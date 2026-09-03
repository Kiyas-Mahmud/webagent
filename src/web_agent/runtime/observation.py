"""Causal observation construction and oracle-leakage validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from web_agent.runtime.contracts import (
    CausalHistoryEntry,
    ConcreteAction,
    ExecutionResult,
    JsonValue,
    Observation,
    ObservationStage,
    PolicyObservation,
    TaskSpecification,
    TransitionInput,
    VersionedRecord,
    completed_causal_history_entry,
    detached_record_copy,
)


class CausalBoundaryError(ValueError):
    """Raised when future, label, or sealed evaluator data reaches policy input."""


class ProcessorParityError(CausalBoundaryError):
    """Training and runtime multimodal processor identities do not match."""


def assert_policy_screenshot_integrity(
    observation: Observation | PolicyObservation,
) -> None:
    """Bind a policy-visible screenshot path to its exact regular-file bytes.

    Synthetic fixtures intentionally use ``screenshot_path=None`` and remain
    compatible.  Any real path, however, must be a non-symlink regular file
    whose current bytes reproduce the observation's registered SHA-256.
    """

    if observation.screenshot_path is None:
        return
    source = Path(observation.screenshot_path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise CausalBoundaryError(
            "policy screenshot artifact is absent or inaccessible"
        ) from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise CausalBoundaryError(
            "policy screenshot artifact must be a non-symlink regular file"
        )
    if metadata.st_nlink != 1:
        raise CausalBoundaryError("policy screenshot artifact must not be hard-linked")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise CausalBoundaryError("policy screenshot artifact must be read-only")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != observation.screenshot_sha256:
        raise CausalBoundaryError(
            "policy screenshot bytes differ from the registered SHA-256"
        )


_FORBIDDEN_KEY_TOKENS = {
    "oracle",
    "oraclesuccess",
    "oraclefailure",
    "oracleprogress",
    "verifier",
    "verification",
    "groundtruth",
    "referenceaction",
    "referenceanswer",
    "referencetrajectory",
    "expectedaction",
    "expectedtarget",
    "correctaction",
    "correcttarget",
    "successlabel",
    "failurelabel",
    "recoverylabel",
    "failuretypegroundtruth",
    "recoverystrategygroundtruth",
    "relevancelabel",
    "futurescreenshot",
    "futurestate",
    "stateafter",
    "taskreward",
    "failureresolved",
    "verifiedsuccess",
    "verifiedprogress",
    "tasksuccess",
}

_FORBIDDEN_SEALED_KEY_TOKENS = {
    "oracle",
    "oraclesuccess",
    "oraclefailure",
    "oracleprogress",
    "verifier",
    "verification",
    "groundtruth",
    "referenceaction",
    "referenceanswer",
    "referencetrajectory",
    "expectedaction",
    "expectedtarget",
    "correctaction",
    "correcttarget",
    "successlabel",
    "failurelabel",
    "recoverylabel",
    "failuretypegroundtruth",
    "recoverystrategygroundtruth",
    "relevancelabel",
    "taskreward",
    "failureresolved",
    "verifiedsuccess",
    "verifiedprogress",
    "tasksuccess",
}


def _normalise_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def assert_oracle_blind_mapping(
    value: Mapping[str, Any],
    *,
    location: str = "policy input",
) -> None:
    """Reject sealed/future keys anywhere inside a policy-visible mapping."""

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalised = _normalise_key(key)
                if normalised in _FORBIDDEN_KEY_TOKENS or any(
                    token in normalised
                    for token in (
                        "oracleanswer",
                        "referencetrajectory",
                        "groundtruth",
                    )
                ):
                    raise CausalBoundaryError(
                        f"forbidden sealed/future field at {path}.{key}"
                    )
                visit(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, location)


def assert_sealed_truth_blind_mapping(
    value: Mapping[str, Any],
    *,
    location: str = "post-action processor mapping",
) -> None:
    """Allow causal post-state fields while rejecting sealed/oracle labels."""

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalised = _normalise_key(key)
                if normalised in _FORBIDDEN_SEALED_KEY_TOKENS or any(
                    token in normalised
                    for token in (
                        "oracleanswer",
                        "referencetrajectory",
                        "groundtruth",
                    )
                ):
                    raise CausalBoundaryError(
                        f"forbidden sealed field at {path}.{key}"
                    )
                visit(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, location)


@dataclass(frozen=True, slots=True)
class ProcessorParityContract(VersionedRecord):
    """Frozen processor identity plus explicit causal field routing.

    A training manifest should persist one instance and runtime construction
    should independently build the other. Equality is required before any
    episode interaction; importing the training stack is unnecessary.
    """

    processor_class: str
    processor_revision: str
    processor_config_sha256: str
    pre_action_field_mapping: Mapping[str, str]
    post_action_field_mapping: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.processor_class or not self.processor_revision:
            raise ValueError("processor class/revision must be frozen")
        if len(self.processor_config_sha256) != 64:
            raise ValueError("processor configuration hash must be SHA-256")
        try:
            int(self.processor_config_sha256, 16)
        except ValueError as exc:
            raise ValueError("processor configuration hash is not hexadecimal") from exc
        for name, mapping in (
            ("pre_action_field_mapping", self.pre_action_field_mapping),
            ("post_action_field_mapping", self.post_action_field_mapping),
        ):
            if not mapping:
                raise ValueError(f"{name} cannot be empty")
            if any(
                not str(source).strip() or not str(destination).strip()
                for source, destination in mapping.items()
            ):
                raise ValueError(f"{name} contains an empty source/destination")
        # Pre-action routing cannot mention future state at all. Post-action
        # routing legitimately carries executed-action/state-after tensors,
        # but neither side may name sealed labels, references, or oracle truth.
        pre_names = {
            str(source): {str(destination): None}
            for source, destination in self.pre_action_field_mapping.items()
        }
        post_names = {
            str(source): {str(destination): None}
            for source, destination in self.post_action_field_mapping.items()
        }
        assert_oracle_blind_mapping(
            pre_names,
            location="pre_action_field_mapping",
        )
        assert_sealed_truth_blind_mapping(
            post_names,
            location="post_action_field_mapping",
        )


def validate_processor_parity(
    training: ProcessorParityContract,
    runtime: ProcessorParityContract,
) -> None:
    """Fail closed on any processor, revision, config, or field-map drift."""
    differences: list[str] = []
    for field_name in (
        "processor_class",
        "processor_revision",
        "processor_config_sha256",
        "pre_action_field_mapping",
        "post_action_field_mapping",
    ):
        if getattr(training, field_name) != getattr(runtime, field_name):
            differences.append(field_name)
    if differences:
        raise ProcessorParityError(
            "training/runtime processor parity mismatch: "
            + ", ".join(differences)
        )


@dataclass(frozen=True, slots=True)
class ObservationBuilder:
    """Project benchmark observations onto a minimal policy-visible schema."""

    processor_contract: ProcessorParityContract | None = None

    def pre_action(
        self,
        task: TaskSpecification,
        observation: Observation,
        *,
        causal_history: tuple[CausalHistoryEntry, ...] = (),
    ) -> PolicyObservation:
        if observation.stage not in {
            ObservationStage.RESET,
            ObservationStage.PRE_ACTION,
            ObservationStage.POST_ACTION,
            ObservationStage.POST_RECOVERY,
        }:
            raise CausalBoundaryError(
                f"unsupported pre-action observation stage: {observation.stage}"
            )
        # A prior post-action state becomes the next step's current pre-state;
        # only its currently observable page representation is propagated.
        return self._policy_view(
            task,
            observation,
            causal_history=causal_history,
        )

    def post_action(
        self,
        task: TaskSpecification,
        observation: Observation,
        action: ConcreteAction,
        *,
        causal_history: tuple[CausalHistoryEntry, ...] = (),
    ) -> PolicyObservation:
        if observation.stage not in {
            ObservationStage.POST_ACTION,
            ObservationStage.POST_RECOVERY,
        }:
            raise CausalBoundaryError(
                "post-action assessment requires a post-action/recovery observation"
            )
        if observation.prior_action_id != action.action_id:
            raise CausalBoundaryError(
                "post-action observation does not cite the action actually executed"
            )
        return self._policy_view(
            task,
            observation,
            causal_history=causal_history,
        )

    def transition(
        self,
        task: TaskSpecification,
        pre_observation: Observation,
        action: ConcreteAction,
        execution_result: ExecutionResult,
        post_observation: Observation,
        *,
        causal_history: tuple[CausalHistoryEntry, ...] = (),
    ) -> TransitionInput:
        prior_history = self._detached_history(causal_history)
        current_entry = completed_causal_history_entry(
            history_index=len(prior_history) + 1,
            action=action,
            execution=execution_result,
            post_observation=post_observation,
        )
        completed_history = (*prior_history, current_entry)
        return TransitionInput(
            task_id=task.task_id,
            pre_observation=self.pre_action(
                task,
                pre_observation,
                causal_history=prior_history,
            ),
            executed_action=detached_record_copy(action),
            execution_result=detached_record_copy(execution_result),
            post_observation=self.post_action(
                task,
                post_observation,
                action,
                causal_history=completed_history,
            ),
        )

    @staticmethod
    def _policy_view(
        task: TaskSpecification,
        observation: Observation,
        *,
        causal_history: tuple[CausalHistoryEntry, ...] = (),
    ) -> PolicyObservation:
        assert_policy_screenshot_integrity(observation)
        assert_oracle_blind_mapping(
            observation.page_state,
            location=f"observation[{observation.observation_id}].page_state",
        )
        return PolicyObservation(
            task_id=task.task_id,
            goal=task.goal,
            observation_id=observation.observation_id,
            screenshot_sha256=observation.screenshot_sha256,
            screenshot_path=observation.screenshot_path,
            width=observation.width,
            height=observation.height,
            url=observation.url,
            title=observation.title,
            current_page_state=dict(observation.page_state),
            causal_history=ObservationBuilder._detached_history(causal_history),
        )

    @staticmethod
    def _detached_history(
        causal_history: tuple[CausalHistoryEntry, ...],
    ) -> tuple[CausalHistoryEntry, ...]:
        if type(causal_history) is not tuple:
            raise CausalBoundaryError("causal history must be an immutable tuple")
        if any(type(item) is not CausalHistoryEntry for item in causal_history):
            raise CausalBoundaryError(
                "causal history contains an unregistered record type"
            )
        return tuple(detached_record_copy(item) for item in causal_history)


def safe_provider_context(
    observation: PolicyObservation,
    hints: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate and combine only data permitted for action-parameter resolution."""
    assert_policy_screenshot_integrity(observation)
    assert_oracle_blind_mapping(observation.current_page_state)
    assert_oracle_blind_mapping(hints, location="parameter_hints")
    return {
        "goal": observation.goal,
        "url": observation.url,
        "title": observation.title,
        "current_page_state": dict(observation.current_page_state),
        "parameter_hints": dict(hints),
    }
