"""Pure conversion from one high-level P1 strategy to concrete actions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from web_agent.runtime.action_parameters import (
    ParameterResolutionError,
    validate_observation_bound_action_parameters,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    PolicyObservation,
    RecoveryDecision,
    RecoveryStrategy,
    RuntimeTaskView,
    TaskSpecification,
    VersionedRecord,
    canonical_sha256,
    runtime_task_view,
)
from web_agent.runtime.observation import (
    assert_oracle_blind_mapping,
    assert_policy_screenshot_integrity,
)


class RecoveryResolutionError(ValueError):
    pass


_TARGET_REFERENCE_KEYS = frozenset(
    {
        "aria_ref",
        "backend_node_id",
        "element",
        "element_id",
        "element_ref",
        "node_id",
        "selector",
        "target",
        "target_id",
        "target_ref",
    }
)
RECOVERY_TARGET_EVIDENCE_KEY = "recovery_target_evidence"
RECOVERY_TARGET_EVIDENCE_SCHEMA = "oracle-blind-visible-targets-v1"
_SEMANTIC_TARGET_FINGERPRINT_SCHEMA = "semantic-target-v1"
_TARGET_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "observation_id",
        "task_id",
        "task_goal_sha256",
        "registered_visible_targets",
    }
)
_VISIBLE_TARGET_FIELDS = frozenset(
    {"semantic_target_sha256", "compatible_action_types"}
)
_GROUNDED_TARGET_ACTIONS = frozenset(
    {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}
)


def _normalised_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _semantic_target_evidence(action: ConcreteAction) -> dict[str, set[str]]:
    """Return target-only evidence, excluding action modifiers such as click count."""

    evidence: dict[str, set[str]] = {}
    parameters: Mapping[str, Any] = action.parameters

    references = {
        canonical_sha256(value)
        for key, value in parameters.items()
        if str(key).lower() in _TARGET_REFERENCE_KEYS and value is not None
    }
    if references:
        evidence["reference"] = references

    bbox = action.bbox
    if bbox is None:
        bbox = _normalised_bbox(parameters.get("target_bbox"))
    if bbox is not None:
        evidence["bbox"] = {canonical_sha256(tuple(float(item) for item in bbox))}

    target_x = parameters.get("target_x")
    target_y = parameters.get("target_y")
    if (
        not isinstance(target_x, bool)
        and isinstance(target_x, (int, float))
        and not isinstance(target_y, bool)
        and isinstance(target_y, (int, float))
    ):
        evidence["coordinates"] = {
            canonical_sha256((float(target_x), float(target_y)))
        }

    # A scrollable element can be a semantic target even though SCROLL has no bbox.
    if "container" in parameters and parameters["container"] is not None:
        evidence["container"] = {canonical_sha256(parameters["container"])}
    if action.action_type is ActionType.NAVIGATE and parameters.get("url") is not None:
        evidence["destination"] = {canonical_sha256(parameters["url"])}
    if action.action_type is ActionType.PRESS_KEY and parameters.get("key") is not None:
        evidence["command"] = {canonical_sha256(parameters["key"])}
    return evidence


def semantic_target_fingerprint(action: ConcreteAction) -> str | None:
    """Return one deterministic fingerprint over target-only action evidence."""

    evidence = _semantic_target_evidence(action)
    for kind in (
        "reference",
        "bbox",
        "coordinates",
        "container",
        "destination",
        "command",
    ):
        values = evidence.get(kind)
        if values:
            return canonical_sha256(
                {
                    "schema_version": _SEMANTIC_TARGET_FINGERPRINT_SCHEMA,
                    "kind": kind,
                    "evidence_sha256": sorted(values),
                }
            )
    return None


def actions_share_semantic_target(
    failed_action: ConcreteAction,
    candidate_action: ConcreteAction,
) -> bool | None:
    """Compare target identity independently of non-target action parameters.

    ``None`` means that the two contracts contain no comparable target evidence.
    Callers must reject that case: absence of evidence cannot prove that an
    alternative target is genuinely different.
    """

    failed = _semantic_target_evidence(failed_action)
    candidate = _semantic_target_evidence(candidate_action)
    comparable = set(failed).intersection(candidate)
    if not comparable:
        return None
    if any(failed[kind].intersection(candidate[kind]) for kind in comparable):
        return True
    return False


def build_recovery_target_evidence(
    *,
    task: TaskSpecification | RuntimeTaskView,
    observation_id: str,
    compatible_actions: Sequence[ConcreteAction],
) -> dict[str, Any]:
    """Build the registered oracle-blind target view for one current observation.

    The observation mapper must pass only actions whose target/destination is
    visibly present and compatible with the current task. Action modifiers such
    as text, option value, click count, and scroll amount never enter the target
    fingerprint.
    """

    task_view = runtime_task_view(task)
    if not observation_id.strip():
        raise ValueError("target evidence requires an observation ID")
    registrations: dict[str, set[str]] = {}
    for action in compatible_actions:
        if not isinstance(action, ConcreteAction):
            raise TypeError("target evidence actions must use ConcreteAction")
        fingerprint = semantic_target_fingerprint(action)
        if fingerprint is None:
            raise ValueError("compatible action has no semantic target evidence")
        registrations.setdefault(fingerprint, set()).add(action.action_type.value)
    return {
        "schema_version": RECOVERY_TARGET_EVIDENCE_SCHEMA,
        "observation_id": observation_id,
        "task_id": task_view.task_id,
        "task_goal_sha256": canonical_sha256(
            {"task_id": task_view.task_id, "goal": task_view.goal}
        ),
        "registered_visible_targets": [
            {
                "semantic_target_sha256": fingerprint,
                "compatible_action_types": sorted(action_types),
            }
            for fingerprint, action_types in sorted(registrations.items())
        ],
    }


def _require_lowercase_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecoveryResolutionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _registered_visible_targets(
    *,
    task_view: RuntimeTaskView,
    observation: PolicyObservation,
) -> dict[str, frozenset[ActionType]]:
    raw = observation.current_page_state.get(RECOVERY_TARGET_EVIDENCE_KEY)
    if not isinstance(raw, Mapping):
        raise RecoveryResolutionError(
            "post-failure observation lacks registered recovery target evidence"
        )
    if set(raw) != _TARGET_EVIDENCE_FIELDS:
        raise RecoveryResolutionError(
            "post-failure recovery target evidence fields are not registered"
        )
    if raw.get("schema_version") != RECOVERY_TARGET_EVIDENCE_SCHEMA:
        raise RecoveryResolutionError("recovery target evidence schema mismatch")
    if raw.get("observation_id") != observation.observation_id:
        raise RecoveryResolutionError(
            "recovery target evidence belongs to another observation"
        )
    if raw.get("task_id") != task_view.task_id:
        raise RecoveryResolutionError("recovery target evidence belongs to another task")
    expected_goal_sha256 = canonical_sha256(
        {"task_id": task_view.task_id, "goal": task_view.goal}
    )
    if raw.get("task_goal_sha256") != expected_goal_sha256:
        raise RecoveryResolutionError("recovery target evidence task binding mismatch")
    rows = raw.get("registered_visible_targets")
    if not isinstance(rows, (list, tuple)):
        raise RecoveryResolutionError(
            "registered visible recovery targets must be an array"
        )
    registrations: dict[str, frozenset[ActionType]] = {}
    previous_fingerprint: str | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _VISIBLE_TARGET_FIELDS:
            raise RecoveryResolutionError(
                f"registered visible recovery target[{index}] is malformed"
            )
        fingerprint = _require_lowercase_sha256(
            f"registered visible recovery target[{index}] fingerprint",
            row.get("semantic_target_sha256"),
        )
        if previous_fingerprint is not None and fingerprint <= previous_fingerprint:
            raise RecoveryResolutionError(
                "registered visible recovery targets must be unique and sorted"
            )
        raw_action_types = row.get("compatible_action_types")
        if not isinstance(raw_action_types, (list, tuple)) or not raw_action_types:
            raise RecoveryResolutionError(
                f"registered visible recovery target[{index}] action types are invalid"
            )
        if any(type(value) is not str for value in raw_action_types):
            raise RecoveryResolutionError(
                f"registered visible recovery target[{index}] action types are invalid"
            )
        action_type_names = tuple(raw_action_types)
        if action_type_names != tuple(sorted(set(action_type_names))):
            raise RecoveryResolutionError(
                "compatible recovery action types must be unique and sorted"
            )
        try:
            action_types = frozenset(ActionType(value) for value in action_type_names)
        except ValueError as exc:
            raise RecoveryResolutionError(
                f"registered visible recovery target[{index}] has unknown action type"
            ) from exc
        registrations[fingerprint] = action_types
        previous_fingerprint = fingerprint
    return registrations


def validate_recovery_target_evidence(
    action: ConcreteAction,
    *,
    task: TaskSpecification | RuntimeTaskView | None,
    post_failure_observation: PolicyObservation | None,
) -> None:
    """Prove current visibility and task compatibility without verifier truth."""

    if task is None or post_failure_observation is None:
        raise RecoveryResolutionError(
            "recovery target validation requires current task and observation"
        )
    try:
        task_view = runtime_task_view(task)
    except (TypeError, ValueError) as exc:
        raise RecoveryResolutionError(
            f"recovery target task validation failed: {exc}"
        ) from exc
    if not isinstance(post_failure_observation, PolicyObservation):
        raise RecoveryResolutionError(
            "recovery target validation requires PolicyObservation"
        )
    if (
        post_failure_observation.task_id != task_view.task_id
        or post_failure_observation.goal != task_view.goal
    ):
        raise RecoveryResolutionError(
            "recovery target observation belongs to another task"
        )
    try:
        assert_policy_screenshot_integrity(post_failure_observation)
        assert_oracle_blind_mapping(
            post_failure_observation.current_page_state,
            location="recovery_target_evidence.current_page_state",
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RecoveryResolutionError(
            f"recovery target observation validation failed: {exc}"
        ) from exc
    fingerprint = semantic_target_fingerprint(action)
    if fingerprint is None:
        raise RecoveryResolutionError(
            "recovery action has no deterministic semantic target evidence"
        )
    registrations = _registered_visible_targets(
        task_view=task_view,
        observation=post_failure_observation,
    )
    compatible_types = registrations.get(fingerprint)
    if compatible_types is None:
        raise RecoveryResolutionError(
            "recovery action target is not visibly present in the current observation"
        )
    if action.action_type not in compatible_types:
        raise RecoveryResolutionError(
            "recovery action target is not registered task-compatible for its action type"
        )


def _validate_retry_action(
    failed_action: ConcreteAction,
    *,
    task: TaskSpecification | RuntimeTaskView | None,
    post_failure_observation: PolicyObservation | None,
) -> None:
    """Revalidate a retry solely from registered, oracle-blind current inputs."""

    if task is None or post_failure_observation is None:
        raise RecoveryResolutionError(
            "RETRY requires the current task and post-failure observation"
        )
    try:
        task_view = runtime_task_view(task)
    except (TypeError, ValueError) as exc:
        raise RecoveryResolutionError(f"RETRY task revalidation failed: {exc}") from exc
    if not isinstance(post_failure_observation, PolicyObservation):
        raise RecoveryResolutionError(
            "RETRY requires a PolicyObservation current-state contract"
        )
    if (
        post_failure_observation.task_id != task_view.task_id
        or post_failure_observation.goal != task_view.goal
    ):
        raise RecoveryResolutionError(
            "RETRY post-failure observation belongs to another task"
        )
    if not post_failure_observation.observation_id.strip():
        raise RecoveryResolutionError(
            "RETRY post-failure observation has no observation ID"
        )
    if post_failure_observation.width <= 0 or post_failure_observation.height <= 0:
        raise RecoveryResolutionError(
            "RETRY post-failure observation has invalid viewport dimensions"
        )
    try:
        assert_policy_screenshot_integrity(post_failure_observation)
        assert_oracle_blind_mapping(
            post_failure_observation.current_page_state,
            location="retry_revalidation.current_page_state",
        )
        assert_oracle_blind_mapping(
            failed_action.parameters,
            location="retry_revalidation.action_parameters",
        )
        validate_observation_bound_action_parameters(
            failed_action.action_type,
            failed_action.parameters,
            failed_action.bbox,
            observation=post_failure_observation,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RecoveryResolutionError(
            f"RETRY failed registered action revalidation: {exc}"
        ) from exc

    if failed_action.destructive:
        if not (
            isinstance(task, TaskSpecification)
            and task.destructive_actions_allowed
            and task.development_partition
        ):
            raise RecoveryResolutionError(
                "RETRY cannot reissue an action that is unsafe for the registered task"
            )

    if failed_action.action_type in {
        ActionType.CLICK,
        ActionType.TYPE,
        ActionType.SELECT,
    }:
        bbox = failed_action.bbox
        if bbox is None or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise RecoveryResolutionError(
                "RETRY target is absent from the current grounded action contract"
            )
        parameter_bbox = failed_action.parameters.get("target_bbox")
        if parameter_bbox is not None and _normalised_bbox(parameter_bbox) != tuple(bbox):
            raise RecoveryResolutionError(
                "RETRY target bbox disagrees with the registered grounding"
            )
        target_x = float(failed_action.parameters["target_x"])
        target_y = float(failed_action.parameters["target_y"])
        x, y, width, height = bbox
        if not (x <= target_x <= x + width and y <= target_y <= y + height):
            raise RecoveryResolutionError(
                "RETRY target coordinates no longer satisfy the registered grounding"
            )
        validate_recovery_target_evidence(
            failed_action,
            task=task,
            post_failure_observation=post_failure_observation,
        )


def _validate_resolved_recovery_action(
    action: ConcreteAction,
    *,
    strategy: RecoveryStrategy,
    task: TaskSpecification | RuntimeTaskView | None,
    post_failure_observation: PolicyObservation | None,
) -> None:
    """Apply the common provider/safety gate to every executable recovery."""

    try:
        assert_oracle_blind_mapping(
            action.parameters,
            location="resolved_recovery_action.parameters",
        )
        validate_observation_bound_action_parameters(
            action.action_type,
            action.parameters,
            action.bbox,
            observation=post_failure_observation,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RecoveryResolutionError(
            f"recovery action failed registered parameter validation: {exc}"
        ) from exc
    if action.destructive and not (
        isinstance(task, TaskSpecification)
        and task.destructive_actions_allowed
        and task.development_partition
    ):
        raise RecoveryResolutionError(
            "recovery action is destructive but the registered task does not permit it"
        )
    if strategy in {
        RecoveryStrategy.REPLAN,
        RecoveryStrategy.ALTERNATIVE_TARGET,
    }:
        validate_recovery_target_evidence(
            action,
            task=task,
            post_failure_observation=post_failure_observation,
        )


@dataclass(frozen=True, slots=True)
class RecoveryPlan(VersionedRecord):
    attempt_id: str
    incident_id: str
    strategy: RecoveryStrategy
    incident_attempt_index: int
    episode_attempt_index: int
    actions: tuple[ConcreteAction, ...]
    resolution_status: str
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if self.incident_attempt_index <= 0 or self.episode_attempt_index <= 0:
            raise ValueError("recovery attempt indices are one-based")
        if self.strategy is RecoveryStrategy.ABORT and self.actions:
            raise ValueError("ABORT must use zero browser actions")
        if self.resolution_status not in {"READY", "ABORT", "REJECTED"}:
            raise ValueError(f"invalid recovery resolution status: {self.resolution_status}")


def resolve_strategy(
    decision: RecoveryDecision,
    failed_action: ConcreteAction,
    *,
    attempt_id: str,
    incident_attempt_index: int,
    episode_attempt_index: int,
    task: TaskSpecification | RuntimeTaskView | None = None,
    post_failure_observation: PolicyObservation | None = None,
) -> RecoveryPlan:
    """Resolve exactly one high-level strategy without executing or verifying it."""
    strategy = decision.strategy
    if strategy is RecoveryStrategy.ABORT:
        return RecoveryPlan(
            attempt_id=attempt_id,
            incident_id=decision.incident_id,
            strategy=strategy,
            incident_attempt_index=incident_attempt_index,
            episode_attempt_index=episode_attempt_index,
            actions=(),
            resolution_status="ABORT",
        )
    try:
        if strategy is RecoveryStrategy.RETRY:
            _validate_retry_action(
                failed_action,
                task=task,
                post_failure_observation=post_failure_observation,
            )
            action = replace(
                failed_action,
                action_id=f"{attempt_id}:action:1",
                source_decision_id=decision.decision_id,
                recovery_attempt_id=attempt_id,
            )
        elif strategy is RecoveryStrategy.BACKTRACK:
            action = ConcreteAction(
                action_id=f"{attempt_id}:action:1",
                source_decision_id=decision.decision_id,
                action_type=ActionType.PRESS_KEY,
                parameters={"key": "ALT+LEFT"},
                recovery_attempt_id=attempt_id,
            )
        elif strategy in {
            RecoveryStrategy.REPLAN,
            RecoveryStrategy.ALTERNATIVE_TARGET,
        }:
            if decision.planned_action is None:
                raise RecoveryResolutionError(
                    f"{strategy.value} requires a frozen-provider planned action"
                )
            action = replace(
                decision.planned_action,
                action_id=f"{attempt_id}:action:1",
                source_decision_id=decision.decision_id,
                recovery_attempt_id=attempt_id,
            )
            if strategy is RecoveryStrategy.ALTERNATIVE_TARGET:
                same_target = actions_share_semantic_target(failed_action, action)
                if same_target is True:
                    raise RecoveryResolutionError(
                        "ALTERNATIVE_TARGET reused the failed semantic target"
                    )
                if same_target is None:
                    raise RecoveryResolutionError(
                        "ALTERNATIVE_TARGET could not prove a different semantic target"
                    )
        elif strategy is RecoveryStrategy.NONE:
            raise RecoveryResolutionError("NONE cannot start a recovery attempt")
        else:  # pragma: no cover - exhaustive enum guard
            raise RecoveryResolutionError(f"unsupported strategy: {strategy}")
        _validate_resolved_recovery_action(
            action,
            strategy=strategy,
            task=task,
            post_failure_observation=post_failure_observation,
        )
    except (RecoveryResolutionError, ParameterResolutionError) as exc:
        return RecoveryPlan(
            attempt_id=attempt_id,
            incident_id=decision.incident_id,
            strategy=strategy,
            incident_attempt_index=incident_attempt_index,
            episode_attempt_index=episode_attempt_index,
            actions=(),
            resolution_status="REJECTED",
            rejection_reason=str(exc),
        )
    return RecoveryPlan(
        attempt_id=attempt_id,
        incident_id=decision.incident_id,
        strategy=strategy,
        incident_attempt_index=incident_attempt_index,
        episode_attempt_index=episode_attempt_index,
        actions=(action,),
        resolution_status="READY",
    )
