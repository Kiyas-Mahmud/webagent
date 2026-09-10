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
    "reward",
    "done",
    "terminated",
    "truncated",
    "evaluatorreward",
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
    "futurescreenshot",
    "futurestate",
    "taskreward",
    "failureresolved",
    "verifiedsuccess",
    "verifiedprogress",
    "tasksuccess",
    "reward",
    "done",
    "terminated",
    "truncated",
    "evaluatorreward",
}

# Match semantic identifier tokens rather than arbitrary substrings. This still
# rejects ``official_task_success`` while avoiding collisions such as
# ``task_successor_id`` or ``email_verification_status``. The live browser
# mapper separately constrains page-state keys to a positive schema.
_FORBIDDEN_IDENTIFIER_PHRASES = (
    ("oracle",),
    ("verifier",),
    ("task", "success"),
    ("task", "progress"),
    ("ground", "truth"),
    ("relevance", "label"),
    ("reference", "action"),
    ("reference", "answer"),
    ("reference", "trajectory"),
    ("expected", "action"),
    ("expected", "target"),
    ("correct", "action"),
    ("correct", "target"),
    ("success", "label"),
    ("failure", "label"),
    ("recovery", "label"),
    ("task", "reward"),
    ("evaluator", "reward"),
    ("verified", "failure"),
    ("verified", "agent", "failure"),
    ("verified", "success"),
    ("verified", "progress"),
    ("recovery", "success"),
    ("failure", "resolved"),
    ("incident", "resolved"),
    ("registered", "progress"),
    ("memory", "relevance"),
    ("relevant", "ids"),
    ("success", "evidence"),
    ("progress", "evidence"),
)
_FUTURE_IDENTIFIER_PHRASES = (
    ("future", "state"),
    ("future", "screenshot"),
)
_COSMETIC_WRAPPER_PREFIXES = {
    "env",
    "environment",
    "episode",
    "evaluator",
    "final",
    "hidden",
    "is",
    "official",
    "oracle",
    "posthoc",
    "reported",
    "sealed",
}
_COSMETIC_WRAPPER_SUFFIXES = {
    "annotation",
    "answer",
    "evidence",
    "flag",
    "label",
    "output",
    "result",
    "reward",
    "score",
    "trajectory",
    "truth",
    "value",
}
_ALLOWED_PRE_ACTION_KEY_TOKENS = {"predictedfailureresolved"}
_ALLOWED_POST_ACTION_KEY_TOKENS = {
    "predictedfailureresolved",
    # P1 is intentionally post-action and consumes the executed causal state.
    # Cosmetic wrappers such as ``official_state_after`` remain forbidden.
    "stateafter",
}
_INDIRECT_CONTROL_KEY_TOKENS = {
    "attribute",
    "attributename",
    "attributes",
    "attributenames",
    "column",
    "columnname",
    "columns",
    "columnnames",
    "field",
    "fieldname",
    "fields",
    "fieldnames",
    "feature",
    "featurename",
    "features",
    "featurenames",
    "keyname",
    "keynames",
    "label",
    "labelname",
    "labels",
    "labelnames",
    "metric",
    "metricname",
    "metrics",
    "metricnames",
    "property",
    "propertyname",
    "properties",
    "propertynames",
}


def _normalise_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _identifier_tokens(value: object) -> tuple[str, ...]:
    rendered = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return tuple(token.lower() for token in re.findall(r"[A-Za-z0-9]+", rendered))


def _contains_identifier_phrase(
    tokens: tuple[str, ...], phrase: tuple[str, ...]
) -> bool:
    width = len(phrase)
    return any(tokens[index : index + width] == phrase for index in range(len(tokens) - width + 1))


def _wrapped_compound_match(normalised: str, compound: str) -> bool:
    start = normalised.find(compound)
    if start < 0:
        return False
    prefix = normalised[:start]
    suffix = normalised[start + len(compound) :]
    return bool(prefix or suffix) and _segmented_affix(
        prefix, _COSMETIC_WRAPPER_PREFIXES
    ) and _segmented_affix(suffix, _COSMETIC_WRAPPER_SUFFIXES)


def _segmented_affix(value: str, vocabulary: set[str]) -> bool:
    if not value:
        return True
    reachable = {0}
    for start in range(len(value)):
        if start not in reachable:
            continue
        for token in vocabulary:
            if value.startswith(token, start):
                reachable.add(start + len(token))
    return len(value) in reachable


def _is_indirect_control_key(key_token: str) -> bool:
    if key_token in _INDIRECT_CONTROL_KEY_TOKENS:
        return True
    return any(
        key_token.endswith(base)
        and _segmented_affix(
            key_token[: -len(base)],
            _COSMETIC_WRAPPER_PREFIXES,
        )
        for base in _INDIRECT_CONTROL_KEY_TOKENS
    )


def _collect_indirect_identifiers(value: Any) -> list[str]:
    identifiers: list[str] = []
    if isinstance(value, str):
        identifiers.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                identifiers.append(item)
            elif isinstance(item, (Mapping, list, tuple)):
                identifiers.extend(_collect_indirect_identifiers(item))
    elif isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            nested_token = _normalise_key(nested_key)
            if nested_token in {"name", "names"} or _is_indirect_control_key(
                nested_token
            ):
                identifiers.extend(_collect_indirect_identifiers(nested_value))
            elif isinstance(nested_value, (Mapping, list, tuple)):
                identifiers.extend(_collect_indirect_identifiers(nested_value))
    return identifiers


def _forbidden_key_matches(
    raw_key: object,
    normalised: str,
    *,
    exact_tokens: set[str],
    allowed_tokens: set[str],
    allow_causal_state_after: bool,
) -> set[str]:
    if normalised in allowed_tokens:
        return set()
    tokens = _identifier_tokens(raw_key)
    matches: set[str] = set()
    if normalised in exact_tokens:
        matches.add(normalised)
    for phrase in (*_FORBIDDEN_IDENTIFIER_PHRASES, *_FUTURE_IDENTIFIER_PHRASES):
        compound = "".join(phrase)
        if _contains_identifier_phrase(tokens, phrase) or compound in tokens:
            matches.add(compound)
        elif _wrapped_compound_match(normalised, compound):
            matches.add(compound)
    # ``verification`` alone is evaluator-like, but may appear causally in a
    # browser-origin name such as ``email_verification_status``. Reject it only
    # as an exact field or when cosmetically wrapped.
    if _wrapped_compound_match(normalised, "verification"):
        matches.add("verification")
    for raw_signal in ("done", "reward", "terminated", "truncated"):
        if _wrapped_compound_match(normalised, raw_signal):
            matches.add(raw_signal)
    state_after = _contains_identifier_phrase(tokens, ("state", "after")) or (
        "stateafter" in tokens
    )
    if not allow_causal_state_after and state_after:
        matches.add("stateafter")
    elif _wrapped_compound_match(normalised, "stateafter"):
        matches.add("stateafter")
    return matches


def _assert_ascii_control_key(key: object, *, path: str) -> None:
    rendered = str(key)
    if not rendered.isascii():
        raise CausalBoundaryError(
            f"non-ASCII control-plane field at {path}.{rendered}"
        )


def _forbidden_indirect_value(
    key_token: str,
    value: Any,
    *,
    exact_tokens: set[str],
    allowed_tokens: set[str],
) -> set[str]:
    """Detect truth-field names hidden in common metadata envelopes.

    Browser text and form values are not scanned. Only source-controlled
    metadata-name fields (for example ``field_name``) receive this check.
    """

    if not _is_indirect_control_key(key_token):
        return set()
    identifiers = _collect_indirect_identifiers(value)
    matches: set[str] = set()
    for identifier in identifiers:
        if not identifier.isascii():
            matches.add("nonasciiidentifier")
            continue
        matches.update(
            _forbidden_key_matches(
                identifier,
                _normalise_key(identifier),
                exact_tokens=exact_tokens,
                allowed_tokens=allowed_tokens,
                allow_causal_state_after=False,
            )
        )
    return matches


def assert_oracle_blind_mapping(
    value: Mapping[str, Any],
    *,
    location: str = "policy input",
) -> None:
    """Reject sealed/future keys anywhere inside a policy-visible mapping."""

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                _assert_ascii_control_key(key, path=path)
                normalised = _normalise_key(key)
                matches = _forbidden_key_matches(
                    key,
                    normalised,
                    exact_tokens=_FORBIDDEN_KEY_TOKENS,
                    allowed_tokens=_ALLOWED_PRE_ACTION_KEY_TOKENS,
                    allow_causal_state_after=False,
                )
                matches.update(
                    _forbidden_indirect_value(
                        normalised,
                        nested,
                        exact_tokens=_FORBIDDEN_KEY_TOKENS,
                        allowed_tokens=_ALLOWED_PRE_ACTION_KEY_TOKENS,
                    )
                )
                if matches:
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
                _assert_ascii_control_key(key, path=path)
                normalised = _normalise_key(key)
                matches = _forbidden_key_matches(
                    key,
                    normalised,
                    exact_tokens=_FORBIDDEN_SEALED_KEY_TOKENS,
                    allowed_tokens=_ALLOWED_POST_ACTION_KEY_TOKENS,
                    allow_causal_state_after=True,
                )
                matches.update(
                    _forbidden_indirect_value(
                        normalised,
                        nested,
                        exact_tokens=_FORBIDDEN_SEALED_KEY_TOKENS,
                        allowed_tokens=_ALLOWED_POST_ACTION_KEY_TOKENS,
                    )
                )
                if matches:
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
