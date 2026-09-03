"""Recovery budget controller: two attempts/incident and four/episode."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
import random
from typing import Callable

from web_agent.runtime.action_parameters import (
    validate_observation_bound_action_parameters,
)
from web_agent.runtime.contracts import (
    ConcreteAction,
    PolicyObservation,
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryStrategy,
    RuntimeTaskView,
    TaskSpecification,
    canonical_sha256,
    detached_record_copy,
    runtime_task_view,
)
from web_agent.runtime.observation import (
    assert_oracle_blind_mapping,
    assert_policy_screenshot_integrity,
)
from web_agent.runtime.model_calls import record_model_call
from web_agent.runtime.protocol import RuntimeBudgets
from web_agent.runtime.recovery.strategies import (
    RecoveryPlan,
    actions_share_semantic_target,
    resolve_strategy,
)


class RecoveryBudgetExceeded(RuntimeError):
    pass


class RecoveryActionPlanner(ABC):
    """Frozen oracle-blind planner for REPLAN/ALTERNATIVE_TARGET actions."""

    planner_id: str
    planner_version: str
    frozen: bool = True

    @abstractmethod
    def plan(
        self,
        task: RuntimeTaskView,
        post_failure_observation: PolicyObservation,
        decision: RecoveryDecision,
        failed_action: ConcreteAction,
        *,
        rng: random.Random,
    ) -> ConcreteAction:
        raise NotImplementedError


RecoveryPlannerCallable = Callable[
    [
        RuntimeTaskView,
        PolicyObservation,
        RecoveryDecision,
        ConcreteAction,
        random.Random,
    ],
    ConcreteAction,
]


@dataclass(frozen=True, slots=True)
class CallableRecoveryActionPlanner(RecoveryActionPlanner):
    planner_id: str
    planner_version: str
    callback: RecoveryPlannerCallable
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.planner_id.strip() or not self.planner_version.strip():
            raise ValueError("recovery planner identity/version are required")
        if not self.frozen:
            raise ValueError("evaluation recovery planner must be frozen")

    def plan(
        self,
        task: TaskSpecification | RuntimeTaskView,
        post_failure_observation: PolicyObservation,
        decision: RecoveryDecision,
        failed_action: ConcreteAction,
        *,
        rng: random.Random,
    ) -> ConcreteAction:
        task_view = runtime_task_view(task)
        if decision.strategy not in {
            RecoveryStrategy.REPLAN,
            RecoveryStrategy.ALTERNATIVE_TARGET,
        }:
            raise ValueError("recovery planner is only valid for replanning strategies")
        if post_failure_observation.task_id != task_view.task_id:
            raise ValueError("recovery planner observation belongs to another task")
        assert_policy_screenshot_integrity(post_failure_observation)
        assert_oracle_blind_mapping(
            post_failure_observation.current_page_state,
            location="recovery_planner_input.current_page_state",
        )
        record_model_call(
            stage="recovery_planner",
            component_id=f"{self.planner_id}@{self.planner_version}",
        )
        protected = (
            task_view,
            post_failure_observation,
            decision,
            failed_action,
        )
        protected_hashes = tuple(item.record_sha256 for item in protected)
        callback_inputs = tuple(detached_record_copy(item) for item in protected)
        callback_hashes = tuple(item.record_sha256 for item in callback_inputs)
        callback_error: BaseException | None = None
        result: object | None = None
        try:
            result = self.callback(
                callback_inputs[0],  # type: ignore[arg-type]
                callback_inputs[1],  # type: ignore[arg-type]
                callback_inputs[2],  # type: ignore[arg-type]
                callback_inputs[3],  # type: ignore[arg-type]
                rng,
            )
        except BaseException as exc:
            callback_error = exc
        mutated: list[str] = []
        for label, records, hashes in (
            ("runtime", protected, protected_hashes),
            ("callback", callback_inputs, callback_hashes),
        ):
            for index, (item, expected) in enumerate(zip(records, hashes)):
                try:
                    actual = item.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(f"{label}[{index}]")
        if mutated:
            mutation = ValueError(
                "recovery planner mutated protected callback input: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise mutation from callback_error
            raise mutation
        if callback_error is not None:
            raise callback_error
        if type(result) is not ConcreteAction:
            raise TypeError("recovery planner returned an invalid action contract")
        assert_oracle_blind_mapping(
            result.parameters,
            location="recovery_planner_output.parameters",
        )
        if result.recovery_attempt_id is not None:
            raise ValueError("planner cannot preassign a recovery attempt ID")
        validate_observation_bound_action_parameters(
            result.action_type,
            result.parameters,
            result.bbox,
            observation=post_failure_observation,
        )
        if decision.strategy is RecoveryStrategy.ALTERNATIVE_TARGET:
            same_target = actions_share_semantic_target(failed_action, result)
            if same_target is True:
                raise ValueError(
                    "alternative-target planner returned the failed semantic target"
                )
            if same_target is None:
                raise ValueError(
                    "alternative-target planner did not prove a different semantic target"
                )
        return detached_record_copy(result)


class RecoveryController:
    def __init__(
        self,
        budgets: RuntimeBudgets,
        *,
        action_planner: RecoveryActionPlanner | None = None,
    ) -> None:
        self.budgets = budgets
        if action_planner is not None and not action_planner.frozen:
            raise ValueError("evaluation recovery planner must be frozen")
        self.action_planner = action_planner
        self._incident_attempts: dict[str, int] = defaultdict(int)
        self._episode_attempts = 0

    @property
    def episode_attempts(self) -> int:
        return self._episode_attempts

    def incident_attempts(self, incident_id: str) -> int:
        return self._incident_attempts[incident_id]

    def reset(self) -> None:
        self._incident_attempts.clear()
        self._episode_attempts = 0

    def require_attempt_available(self, incident_id: str) -> None:
        """Check both caps before shadow/retrieval work for a new attempt."""
        incident_count = self._incident_attempts[incident_id]
        if incident_count >= self.budgets.max_recovery_attempts_per_incident:
            raise RecoveryBudgetExceeded(
                f"incident {incident_id} exhausted "
                f"{self.budgets.max_recovery_attempts_per_incident} attempts"
            )
        if self._episode_attempts >= self.budgets.max_recovery_attempts_per_episode:
            raise RecoveryBudgetExceeded(
                "episode exhausted "
                f"{self.budgets.max_recovery_attempts_per_episode} recovery attempts"
            )

    def resolve(
        self,
        decision: RecoveryDecision,
        failed_action: ConcreteAction,
        *,
        task: TaskSpecification | RuntimeTaskView | None = None,
        post_failure_observation: PolicyObservation | None = None,
        rng: random.Random | None = None,
    ) -> RecoveryPlan:
        """Resolve one bounded recovery attempt through the registered path.

        ``begin_attempt`` remains the implementation and compatibility entry
        point used by existing runners.  This wrapper exposes the exact public
        interface named in the Table 2 architecture without changing budget or
        planner semantics.
        """

        return self.begin_attempt(
            decision,
            failed_action,
            task=task,
            post_failure_observation=post_failure_observation,
            rng=rng,
        )

    def begin_attempt(
        self,
        decision: RecoveryDecision,
        failed_action: ConcreteAction,
        *,
        task: TaskSpecification | RuntimeTaskView | None = None,
        post_failure_observation: PolicyObservation | None = None,
        rng: random.Random | None = None,
    ) -> RecoveryPlan:
        self.require_attempt_available(decision.incident_id)
        # The high-level invocation itself is the attempt.  Invalid planner
        # output is therefore represented as a rejected attempt rather than an
        # exception that escapes without accounting.
        self._incident_attempts[decision.incident_id] += 1
        self._episode_attempts += 1
        incident_index = self._incident_attempts[decision.incident_id]
        episode_index = self._episode_attempts
        attempt_id = f"{decision.incident_id}:attempt:{incident_index}"
        resolved_decision = decision
        try:
            if (
                decision.strategy
                in {RecoveryStrategy.REPLAN, RecoveryStrategy.ALTERNATIVE_TARGET}
                and decision.planned_action is None
                and self.action_planner is not None
            ):
                if task is None or post_failure_observation is None or rng is None:
                    raise ValueError(
                        "recovery planner requires current task, post-failure "
                        "observation, and stage-keyed RNG"
                    )
                resolved_decision = replace(
                    decision,
                    planned_action=self._plan_action_isolated(
                        runtime_task_view(task),
                        post_failure_observation,
                        decision,
                        failed_action,
                        rng=rng,
                    ),
                )
        except (TypeError, ValueError) as exc:
            return RecoveryPlan(
                attempt_id=attempt_id,
                incident_id=decision.incident_id,
                strategy=decision.strategy,
                incident_attempt_index=incident_index,
                episode_attempt_index=episode_index,
                actions=(),
                resolution_status="REJECTED",
                rejection_reason=f"recovery planner rejected: {exc}",
            )
        except Exception as exc:
            # The invocation has already consumed its registered attempt.  An
            # unexpected planner/backend failure must therefore return a
            # schema-valid rejected plan so EpisodeRunner can log the attempt.
            # Preserve only the exception class and a commitment to its text;
            # model/provider error strings may contain private page content.
            error_sha256 = canonical_sha256(
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return RecoveryPlan(
                attempt_id=attempt_id,
                incident_id=decision.incident_id,
                strategy=decision.strategy,
                incident_attempt_index=incident_index,
                episode_attempt_index=episode_index,
                actions=(),
                resolution_status="REJECTED",
                rejection_reason=(
                    "recovery planner failed: "
                    f"{type(exc).__name__}; error_sha256={error_sha256}"
                ),
            )
        return resolve_strategy(
            resolved_decision,
            failed_action,
            attempt_id=attempt_id,
            incident_attempt_index=incident_index,
            episode_attempt_index=episode_index,
            task=task,
            post_failure_observation=post_failure_observation,
        )

    def _plan_action_isolated(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        decision: RecoveryDecision,
        failed_action: ConcreteAction,
        *,
        rng: random.Random,
    ) -> ConcreteAction:
        """Protect controller-owned recovery decisions from custom planners."""

        assert self.action_planner is not None
        protected = (task, observation, decision, failed_action)
        protected_hashes = tuple(item.record_sha256 for item in protected)
        callback_inputs = tuple(detached_record_copy(item) for item in protected)
        callback_hashes = tuple(item.record_sha256 for item in callback_inputs)
        callback_error: BaseException | None = None
        result: object | None = None
        try:
            result = self.action_planner.plan(
                callback_inputs[0],  # type: ignore[arg-type]
                callback_inputs[1],  # type: ignore[arg-type]
                callback_inputs[2],  # type: ignore[arg-type]
                callback_inputs[3],  # type: ignore[arg-type]
                rng=rng,
            )
        except BaseException as exc:
            callback_error = exc
        mutated: list[str] = []
        for label, records, hashes in (
            ("runtime", protected, protected_hashes),
            ("callback", callback_inputs, callback_hashes),
        ):
            for index, (item, expected) in enumerate(zip(records, hashes)):
                try:
                    actual = item.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(f"{label}[{index}]")
        if mutated:
            mutation = ValueError(
                "recovery planner mutated protected callback input: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise mutation from callback_error
            raise mutation
        if callback_error is not None:
            raise callback_error
        if type(result) is not ConcreteAction:
            raise TypeError("recovery planner returned an invalid action contract")
        return detached_record_copy(result)

    @staticmethod
    def finish_attempt(
        plan: RecoveryPlan,
        *,
        predicted_assessment_id: str | None = None,
        executed_action_ids: tuple[str, ...] | None = None,
        completed: bool | None = None,
    ) -> RecoveryAttempt:
        planned_action_ids = tuple(action.action_id for action in plan.actions)
        recorded_action_ids = (
            planned_action_ids
            if executed_action_ids is None
            else tuple(executed_action_ids)
        )
        if recorded_action_ids != planned_action_ids[: len(recorded_action_ids)]:
            raise ValueError(
                "executed recovery action IDs must be an ordered prefix of the plan"
            )
        if completed is not None and type(completed) is not bool:
            raise ValueError("recovery completed override must be an exact boolean")
        return RecoveryAttempt(
            attempt_id=plan.attempt_id,
            incident_id=plan.incident_id,
            strategy=plan.strategy,
            incident_attempt_index=plan.incident_attempt_index,
            episode_attempt_index=plan.episode_attempt_index,
            action_ids=recorded_action_ids,
            completed=(
                plan.resolution_status in {"READY", "ABORT"}
                if completed is None
                else completed
            ),
            predicted_assessment_id=predicted_assessment_id,
        )
