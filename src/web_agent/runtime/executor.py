"""Shared browser executor with exact Table 2 budget accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Callable, TypeVar

from web_agent.benchmarks.base import AdapterExecution, BenchmarkAdapter
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionEvidence,
    ExecutionResult,
    ExecutionStatus,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    TaskSpecification,
    VerifierReceiptBinding,
    VersionedRecord,
    detached_record_copy,
)
from web_agent.runtime.deadline import interrupt_after
from web_agent.runtime.protocol import (
    REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS,
    RuntimeBudgets,
)


class ExecutorBudgetExceeded(RuntimeError):
    pass


class EpisodeTimeout(RuntimeError):
    pass


class ExecutorCleanupTimeout(EpisodeTimeout):
    """Adapter cleanup exhausted the boundary; this is not an agent timeout."""


class ExecutorInputMutation(RuntimeError):
    """A benchmark callback mutated its detached concrete-action input."""

    infrastructure_invalid = True


_T = TypeVar("_T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ExecutorSnapshot:
    steps_used: int
    steps_remaining: int
    elapsed_seconds: float


class Executor:
    """Count every agent-requested low-level action, even rejected requests."""

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        *,
        budgets: RuntimeBudgets,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.adapter = adapter
        self.budgets = budgets
        self._clock = clock
        self._steps_used = 0
        self._started_at: float | None = None
        self._task: TaskSpecification | None = None
        self._closed = False

    @property
    def steps_used(self) -> int:
        return self._steps_used

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    @property
    def snapshot(self) -> ExecutorSnapshot:
        return ExecutorSnapshot(
            steps_used=self._steps_used,
            steps_remaining=max(
                0,
                self.budgets.max_executor_steps - self._steps_used,
            ),
            elapsed_seconds=self.elapsed_seconds,
        )

    @property
    def deadline_expired(self) -> bool:
        """Whether the active episode has exhausted its registered wall clock."""

        return (
            self._started_at is not None
            and self.elapsed_seconds >= self.budgets.episode_timeout_seconds
        )

    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        if self._closed:
            raise RuntimeError("executor is closed")
        self._steps_used = 0
        self._started_at = self._clock()
        self._task = task
        observation = self.run_blocking(
            "environment reset",
            lambda: self.adapter.reset(
                task,
                episode_id=episode_id,
                seed=seed,
            ),
        )
        if observation.stage is not ObservationStage.RESET:
            raise RuntimeError("benchmark reset returned the wrong observation stage")
        return observation

    def reset_state_receipt(self) -> VersionedRecord | None:
        """Read the adapter's immutable post-reset evidence, if implemented."""

        return self.adapter.reset_state_receipt()

    def execute(self, action: ConcreteAction) -> ExecutionResult:
        started_at_utc = _utc_now()
        step = self._consume_request()
        safety_error = self._safety_error(action)
        if safety_error is not None:
            ended_at_utc = _utc_now()
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.REJECTED,
                executor_step=step,
                state_changed=False,
                environment_error=False,
                error_kind="safety_rejection",
                message=safety_error,
                evidence=ExecutionEvidence(
                    action_id=action.action_id,
                    started_at_utc=started_at_utc,
                    ended_at_utc=ended_at_utc,
                    status=ExecutionStatus.REJECTED,
                ),
            )
        action_sha256 = action.record_sha256
        callback_action = detached_record_copy(action)
        callback_sha256 = callback_action.record_sha256

        def execute_isolated():
            callback_error: BaseException | None = None
            raw_result = None
            try:
                raw_result = self.adapter.execute(callback_action)
            except BaseException as exc:
                callback_error = exc
            mutated: list[str] = []
            for label, protected, expected in (
                ("runtime", action, action_sha256),
                ("callback", callback_action, callback_sha256),
            ):
                try:
                    actual = protected.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(label)
            if mutated:
                error = ExecutorInputMutation(
                    "benchmark adapter mutated protected concrete-action input: "
                    + ", ".join(mutated)
                )
                if callback_error is not None:
                    raise error from callback_error
                raise error
            if callback_error is not None:
                raise callback_error
            return raw_result

        try:
            raw = self.run_blocking(
                "browser execute",
                execute_isolated,
            )
        except BaseException as exc:
            partial = getattr(exc, "adapter_execution", None)
            if type(partial) is AdapterExecution:
                evidence = partial.evidence or ExecutionEvidence(
                    action_id=action.action_id,
                    started_at_utc=started_at_utc,
                    ended_at_utc=_utc_now(),
                    status=partial.status,
                )
                setattr(
                    exc,
                    "execution_result",
                    ExecutionResult(
                        action_id=action.action_id,
                        status=partial.status,
                        executor_step=step,
                        state_changed=partial.state_changed,
                        environment_error=partial.environment_error,
                        error_kind=partial.error_kind,
                        message=partial.message,
                        internal_retry_count=partial.internal_retry_count,
                        latency_ms=partial.latency_ms,
                        evidence=evidence,
                    ),
                )
            raise
        if raw is None:  # pragma: no cover - callback return invariant
            raise RuntimeError("benchmark adapter returned no execution contract")
        evidence = raw.evidence or ExecutionEvidence(
            action_id=action.action_id,
            started_at_utc=started_at_utc,
            ended_at_utc=_utc_now(),
            status=raw.status,
        )
        return ExecutionResult(
            action_id=action.action_id,
            status=raw.status,
            executor_step=step,
            state_changed=raw.state_changed,
            environment_error=raw.environment_error,
            error_kind=raw.error_kind,
            message=raw.message,
            internal_retry_count=raw.internal_retry_count,
            latency_ms=raw.latency_ms,
            evidence=evidence,
        )

    def reject_unresolved_request(
        self,
        *,
        action_id: str,
        reason: str,
        error_kind: str = "parameter_resolution_rejected",
    ) -> ExecutionResult:
        """Charge one step when the shared provider cannot produce an action."""
        started_at_utc = _utc_now()
        step = self._consume_request()
        ended_at_utc = _utc_now()
        return ExecutionResult(
            action_id=action_id,
            status=ExecutionStatus.REJECTED,
            executor_step=step,
            state_changed=False,
            environment_error=False,
            error_kind=error_kind,
            message=reason,
            evidence=ExecutionEvidence(
                action_id=action_id,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                status=ExecutionStatus.REJECTED,
            ),
        )

    def reject_pre_action_parse_request(
        self,
        *,
        request_id: str,
    ) -> ExecutionResult:
        """Charge one step for an invalid base-policy parser request.

        No ``ConcreteAction`` is accepted here: this is an executor-accounting
        receipt for a request that is structurally incapable of reaching the
        browser.
        """

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("pre-action parse rejection requires a request ID")
        started_at_utc = _utc_now()
        step = self._consume_request()
        ended_at_utc = _utc_now()
        return ExecutionResult(
            action_id=request_id,
            status=ExecutionStatus.REJECTED,
            executor_step=step,
            state_changed=False,
            environment_error=False,
            error_kind="pre_action_parse_rejected",
            message="invalid pre-action parser output",
            evidence=ExecutionEvidence(
                action_id=request_id,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                status=ExecutionStatus.REJECTED,
            ),
        )

    def observe_after(
        self,
        action: ConcreteAction,
        *,
        recovery: bool = False,
    ) -> Observation:
        return self.run_blocking(
            "post-action observation",
            lambda: self.adapter.observe(
                stage=(
                    ObservationStage.POST_RECOVERY
                    if recovery
                    else ObservationStage.POST_ACTION
                ),
                prior_action_id=action.action_id,
            ),
        )

    def observe_current(self) -> Observation:
        return self.run_blocking(
            "current observation",
            lambda: self.adapter.observe(stage=ObservationStage.PRE_ACTION),
        )

    def terminal_signal(
        self,
        binding: VerifierReceiptBinding,
    ) -> OpaqueTerminalSignal:
        if self._task is None:
            raise RuntimeError("executor has no active task")
        return self.run_blocking(
            "sealed terminal evaluation",
            lambda: self.adapter.terminal_signal(self._task, binding),
        )

    def run_blocking(self, operation: str, callback: Callable[[], _T]) -> _T:
        """Run one potentially blocking call inside the hard episode deadline.

        The post-call check is essential: a browser/model/evaluator call that
        starts before 600 seconds but returns after it may not influence the
        episode.  If a call itself raises after the deadline, timeout takes
        precedence so the overrun cannot be mislabeled as another outcome.
        """

        if not operation.strip():
            raise ValueError("deadline-guarded operation name cannot be empty")
        self.require_time_remaining(operation=operation)
        remaining = max(
            0.0,
            float(self.budgets.episode_timeout_seconds) - self.elapsed_seconds,
        )

        def timeout_error() -> EpisodeTimeout:
            return EpisodeTimeout(
                f"episode reached {self.budgets.episode_timeout_seconds:.0f}s"
                f" during {operation}"
            )

        try:
            with interrupt_after(remaining, exception_factory=timeout_error):
                result = callback()
        except Exception as exc:
            try:
                self.require_time_remaining(operation=operation)
            except EpisodeTimeout as timeout:
                raise timeout from exc
            raise
        self.require_time_remaining(operation=operation)
        return result

    def require_time_remaining(self, *, operation: str | None = None) -> None:
        if self._started_at is None:
            raise RuntimeError("executor has not been reset")
        if self.elapsed_seconds >= self.budgets.episode_timeout_seconds:
            suffix = f" during {operation}" if operation else ""
            raise EpisodeTimeout(
                f"episode reached {self.budgets.episode_timeout_seconds:.0f}s"
                f"{suffix}"
            )

    def _consume_request(self) -> int:
        self.require_time_remaining()
        if self._steps_used >= self.budgets.max_executor_steps:
            raise ExecutorBudgetExceeded(
                f"executor request budget exhausted at {self._steps_used}"
            )
        self._steps_used += 1
        return self._steps_used

    def _safety_error(self, action: ConcreteAction) -> str | None:
        task = self._task
        if task is None:
            return "executor has no active task"
        if action.destructive and not (
            task.destructive_actions_allowed and task.development_partition
        ):
            return "destructive action is not permitted by the registered task"
        if action.action_type in {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}:
            for name in ("target_x", "target_y"):
                value = action.parameters.get(name)
                if not isinstance(value, (int, float)):
                    return f"{name} is missing or non-numeric"
                if not 0.0 <= float(value) <= 1.0:
                    return f"{name} lies outside the normalized viewport"
        return None

    def close(self) -> None:
        """Close the adapter without creating an unbounded cleanup gap.

        Environment cleanup is part of the registered episode boundary.  When
        time remains, the callback receives only that remaining time.  If the
        episode deadline was already reached, invoking arbitrary adapter code
        would necessarily extend the frozen task duration, so the callback is
        not entered.  Marking the executor closed before dispatch also prevents
        a timed-out cleanup from being retried outside the deadline.

        Executors that never reached ``reset`` have no live episode clock; the
        shorter registered pre-browser/episode duration is used as a
        fail-closed upper bound for constructor-time adapter cleanup.
        """

        if self._closed:
            return
        self._closed = True
        if self._started_at is not None:
            if self.deadline_expired:
                return
            try:
                self.run_blocking("environment close", self.adapter.close)
            except EpisodeTimeout as exc:
                raise ExecutorCleanupTimeout(
                    "environment close exhausted the registered episode boundary"
                ) from exc
            return

        timeout_seconds = min(
            float(self.budgets.episode_timeout_seconds),
            float(REGISTERED_PRE_BROWSER_SETUP_TIMEOUT_SECONDS),
        )

        def timeout_error() -> ExecutorCleanupTimeout:
            return ExecutorCleanupTimeout(
                f"executor cleanup reached {timeout_seconds:.0f}s"
                " during environment close"
            )

        with interrupt_after(timeout_seconds, exception_factory=timeout_error):
            self.adapter.close()
