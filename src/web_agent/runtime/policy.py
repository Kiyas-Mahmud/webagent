"""Frozen policy interface used by E0-E3 without importing training code."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import random

from web_agent.runtime.contracts import (
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryTransitionInput,
    RuntimeTaskView,
    TaskSpecification,
    TransitionAssessment,
    TransitionInput,
    VersionedRecord,
    canonical_sha256,
    detached_record_copy,
    runtime_task_view,
)
from web_agent.runtime.observation import (
    assert_oracle_blind_mapping,
    assert_policy_screenshot_integrity,
)
from web_agent.runtime.model_calls import record_model_call
from web_agent.runtime.protocol import SystemSwitches


class PolicyError(RuntimeError):
    pass


def _guarded_record_callback(
    *,
    records: tuple[VersionedRecord, ...],
    callback: Callable[[tuple[VersionedRecord, ...]], object],
    boundary: str,
) -> object:
    """Call untrusted model glue with detached inputs and mutation canaries.

    Frozen dataclasses may still contain mutable dictionaries.  Each callback
    therefore receives a strict schema reconstruction.  Both the runtime-owned
    inputs and callback-only copies are hashed before and after invocation; a
    callback that mutates either graph fails closed even when it also raises.
    """

    originals = records
    original_hashes = tuple(item.record_sha256 for item in originals)
    detached = tuple(detached_record_copy(item) for item in originals)
    detached_hashes = tuple(item.record_sha256 for item in detached)

    callback_error: BaseException | None = None
    result: object | None = None
    try:
        result = callback(detached)
    except BaseException as exc:
        callback_error = exc

    mutated: list[str] = []
    for index, (item, expected) in enumerate(zip(originals, original_hashes)):
        try:
            actual = item.record_sha256
        except Exception:
            actual = "<unhashable>"
        if actual != expected:
            mutated.append(f"runtime[{index}]")
    for index, (item, expected) in enumerate(zip(detached, detached_hashes)):
        try:
            actual = item.record_sha256
        except Exception:
            actual = "<unhashable>"
        if actual != expected:
            mutated.append(f"callback[{index}]")
    if mutated:
        mutation = PolicyError(
            f"{boundary} mutated protected callback input: {', '.join(mutated)}"
        )
        if callback_error is not None:
            raise mutation from callback_error
        raise mutation
    if callback_error is not None:
        raise callback_error
    return result


class ActionParseError(PolicyError):
    """The base-policy parser could not produce a valid pre-action contract.

    Backends must raise this type only after model inference has completed and
    parser output is invalid.  Other inference/runtime failures remain ordinary
    ``PolicyError`` instances and therefore do not consume executor requests.
    """

    error_kind = "invalid_pre_action_parser_output"

    def __init__(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("action parse errors require a non-empty message")
        super().__init__(message)
        self.error_sha256 = canonical_sha256(
            {
                "error_kind": self.error_kind,
                "message": message,
            }
        )


class PolicyKind(str, Enum):
    BASE = "base"
    TRAINED = "trained"


class PolicyAdapter(ABC):
    """A frozen base or trained policy with strict causal entry points."""

    policy_id: str
    policy_version: str
    kind: PolicyKind
    checkpoint_sha256: str | None

    @abstractmethod
    def predict_action(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        *,
        rng: random.Random,
    ) -> PreActionDecision:
        raise NotImplementedError

    @abstractmethod
    def assess_transition(
        self,
        task: RuntimeTaskView,
        transition: TransitionInput,
        *,
        rng: random.Random,
    ) -> TransitionAssessment:
        raise NotImplementedError

    @abstractmethod
    def assess_recovery(
        self,
        task: RuntimeTaskView,
        transition: RecoveryTransitionInput,
        *,
        rng: random.Random,
    ) -> RecoveryAssessment:
        """Predict outcome from the actually executed recovery transition."""
        raise NotImplementedError


ActionPredictor = Callable[
    [RuntimeTaskView, PolicyObservation, random.Random],
    PreActionDecision,
]
TransitionPredictor = Callable[
    [RuntimeTaskView, TransitionInput, random.Random],
    TransitionAssessment,
]
RecoveryPredictor = Callable[
    [RuntimeTaskView, RecoveryTransitionInput, random.Random],
    RecoveryAssessment,
]


@dataclass(frozen=True, slots=True)
class CallablePolicyAdapter(PolicyAdapter):
    """Dependency-injected adapter for a frozen model or deterministic fixture.

    Model-specific checkpoint construction remains outside this module.  The
    callbacks receive only stage-valid contracts and a stage-keyed RNG.
    """

    policy_id: str
    policy_version: str
    kind: PolicyKind
    action_predictor: ActionPredictor
    transition_predictor: TransitionPredictor | None = None
    recovery_predictor: RecoveryPredictor | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_id or not self.policy_version:
            raise ValueError("policy identity and version must be frozen")
        if self.kind is PolicyKind.TRAINED and not self.checkpoint_sha256:
            raise ValueError("trained policies require a checkpoint SHA-256")

    def predict_action(
        self,
        task: TaskSpecification | RuntimeTaskView,
        observation: PolicyObservation,
        *,
        rng: random.Random,
    ) -> PreActionDecision:
        task_view = runtime_task_view(task)
        if observation.task_id != task_view.task_id:
            raise PolicyError("policy observation belongs to another task")
        assert_policy_screenshot_integrity(observation)
        assert_oracle_blind_mapping(observation.current_page_state)
        record_model_call(
            stage="pre_action_policy",
            component_id=f"{self.policy_id}@{self.policy_version}",
        )
        result = _guarded_record_callback(
            records=(task_view, observation),
            callback=lambda inputs: self.action_predictor(
                inputs[0],  # type: ignore[arg-type]
                inputs[1],  # type: ignore[arg-type]
                rng,
            ),
            boundary="pre-action policy",
        )
        if type(result) is not PreActionDecision:
            self._raise_invalid_action_output(
                "action predictor returned an invalid contract"
            )
        if result.observation_id != observation.observation_id:
            self._raise_invalid_action_output(
                "action decision cites another observation"
            )
        if result.policy_id != self.policy_id or result.policy_version != self.policy_version:
            self._raise_invalid_action_output(
                "action decision policy identity mismatch"
            )
        assert_oracle_blind_mapping(
            result.parameter_hints,
            location="pre_action_decision.parameter_hints",
        )
        return detached_record_copy(result)

    def _raise_invalid_action_output(self, message: str) -> None:
        if self.kind is PolicyKind.BASE:
            raise ActionParseError(message)
        raise PolicyError(message)

    def assess_transition(
        self,
        task: TaskSpecification | RuntimeTaskView,
        transition: TransitionInput,
        *,
        rng: random.Random,
    ) -> TransitionAssessment:
        if self.transition_predictor is None:
            raise PolicyError("this policy has no post-action diagnosis path")
        task_view = runtime_task_view(task)
        if transition.task_id != task_view.task_id:
            raise PolicyError("transition belongs to another task")
        assert_policy_screenshot_integrity(transition.pre_observation)
        assert_policy_screenshot_integrity(transition.post_observation)
        assert_oracle_blind_mapping(
            transition.pre_observation.current_page_state,
            location="transition.pre_observation.current_page_state",
        )
        assert_oracle_blind_mapping(
            transition.post_observation.current_page_state,
            location="transition.post_observation.current_page_state",
        )
        record_model_call(
            stage="post_action_assessment",
            component_id=f"{self.policy_id}@{self.policy_version}",
        )
        result = _guarded_record_callback(
            records=(task_view, transition),
            callback=lambda inputs: self.transition_predictor(  # type: ignore[misc]
                inputs[0],  # type: ignore[arg-type]
                inputs[1],  # type: ignore[arg-type]
                rng,
            ),
            boundary="post-action policy",
        )
        if type(result) is not TransitionAssessment:
            raise PolicyError("transition predictor returned an invalid contract")
        if (
            result.pre_observation_id != transition.pre_observation.observation_id
            or result.post_observation_id != transition.post_observation.observation_id
            or result.executed_action_id != transition.executed_action.action_id
        ):
            raise PolicyError("transition assessment cites inconsistent causal inputs")
        return detached_record_copy(result)

    def assess_recovery(
        self,
        task: TaskSpecification | RuntimeTaskView,
        transition: RecoveryTransitionInput,
        *,
        rng: random.Random,
    ) -> RecoveryAssessment:
        if self.recovery_predictor is None:
            raise PolicyError("this policy has no recovery-transition assessment path")
        task_view = runtime_task_view(task)
        if transition.task_id != task_view.task_id:
            raise PolicyError("recovery transition belongs to another task")
        assert_policy_screenshot_integrity(
            transition.pre_recovery_observation
        )
        assert_policy_screenshot_integrity(
            transition.post_recovery_observation
        )
        assert_oracle_blind_mapping(
            transition.pre_recovery_observation.current_page_state,
            location="recovery.pre_observation.current_page_state",
        )
        assert_oracle_blind_mapping(
            transition.post_recovery_observation.current_page_state,
            location="recovery.post_observation.current_page_state",
        )
        record_model_call(
            stage="recovery_assessment",
            component_id=f"{self.policy_id}@{self.policy_version}",
        )
        result = _guarded_record_callback(
            records=(task_view, transition),
            callback=lambda inputs: self.recovery_predictor(  # type: ignore[misc]
                inputs[0],  # type: ignore[arg-type]
                inputs[1],  # type: ignore[arg-type]
                rng,
            ),
            boundary="recovery-assessment policy",
        )
        if type(result) is not RecoveryAssessment:
            raise PolicyError("recovery predictor returned an invalid contract")
        if (
            result.incident_id != transition.incident_id
            or result.attempt_id != transition.attempt_id
            or result.pre_recovery_observation_id
            != transition.pre_recovery_observation.observation_id
            or result.post_recovery_observation_id
            != transition.post_recovery_observation.observation_id
            or result.recovery_action_ids
            != tuple(action.action_id for action in transition.recovery_actions)
        ):
            raise PolicyError("recovery assessment cites inconsistent causal inputs")
        return detached_record_copy(result)


@dataclass(frozen=True, slots=True)
class SystemPolicy:
    """Bind a policy to one registered E0-E3 switch set."""

    adapter: PolicyAdapter
    switches: SystemSwitches

    def __post_init__(self) -> None:
        if self.switches.trained_pre_action_policy:
            if self.adapter.kind is not PolicyKind.TRAINED:
                raise ValueError(
                    f"{self.switches.system_id.value} requires the trained policy"
                )
        elif self.adapter.kind is not PolicyKind.BASE:
            raise ValueError("E0 requires the unadapted selected-backbone policy")
        if isinstance(self.adapter, CallablePolicyAdapter):
            if (
                self.switches.post_action_diagnosis
                and self.adapter.transition_predictor is None
            ):
                raise ValueError("diagnosis-enabled system requires transition predictor")
            if (
                self.switches.recovery_controller
                and self.adapter.recovery_predictor is None
            ):
                raise ValueError(
                    "recovery-enabled system requires executed-transition assessment"
                )

    def predict_action(
        self,
        task: TaskSpecification | RuntimeTaskView,
        observation: PolicyObservation,
        *,
        rng: random.Random,
    ) -> PreActionDecision:
        return self.adapter.predict_action(runtime_task_view(task), observation, rng=rng)

    def assess_transition(
        self,
        task: TaskSpecification | RuntimeTaskView,
        transition: TransitionInput,
        *,
        rng: random.Random,
    ) -> TransitionAssessment:
        if not self.switches.post_action_diagnosis:
            raise PolicyError(
                f"{self.switches.system_id.value} cannot run post-action diagnosis"
            )
        return self.adapter.assess_transition(
            runtime_task_view(task), transition, rng=rng
        )

    def assess_recovery(
        self,
        task: TaskSpecification | RuntimeTaskView,
        transition: RecoveryTransitionInput,
        *,
        rng: random.Random,
    ) -> RecoveryAssessment:
        if not self.switches.recovery_controller:
            raise PolicyError(
                f"{self.switches.system_id.value} cannot assess recovery transitions"
            )
        return self.adapter.assess_recovery(
            runtime_task_view(task), transition, rng=rng
        )
