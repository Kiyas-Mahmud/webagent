"""Authenticated JSON framing and causal schemas for isolated process roles."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import socket
import stat
import struct
from typing import Any, Mapping
from urllib.parse import urlsplit

from web_agent.benchmarks.base import AdapterExecution
from web_agent.runtime.contracts import (
    ConcreteAction,
    Observation,
    OpaqueTerminalSignal,
    VerifierReceiptBinding,
)
from web_agent.runtime.state_reset import (
    WebArenaResetStateReceipt,
    WebArenaResetStateRequest,
)

from .common import canonical_json_bytes, sha256_file


PROCESS_BROKER_PROTOCOL_VERSION = "table2-process-page-broker-ipc-v2"
MAX_PROCESS_BROKER_MESSAGE_BYTES = 1_048_576
RUNTIME_BROKER_OPERATIONS = frozenset(
    {
        "runtime_reset",
        "runtime_observe",
        "runtime_execute",
        "runtime_terminal",
        "runtime_close",
    }
)
RUNTIME_INNER_SCHEMA_PATHS = (
    "runtime_reset.request",
    "runtime_reset.result.observation",
    "runtime_reset.result.reset_state_receipt",
    "runtime_execute.request.action",
    "runtime_observe.result.observation",
    "runtime_execute.result.execution",
    "runtime_terminal.request.receipt_binding",
    "runtime_terminal.result.opaque_terminal_signal",
)
PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION = (
    "table2-process-broker-inner-schema-registry-v2"
)
PROCESS_BROKER_INNER_SCHEMA_REGISTRY = (
    {
        "path": "runtime_reset.request",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "WebArenaResetStateRequest",
        "validator_id": "webarena-reset-request-fields-v1",
    },
    {
        "path": "runtime_reset.result.observation",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "Observation",
        "validator_id": (
            "browsergym-causal-observation-request-bound-"
            "root-confined-screenshot-v2"
        ),
    },
    {
        "path": "runtime_reset.result.reset_state_receipt",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "WebArenaResetStateReceipt",
        "validator_id": "webarena-hashes-only-reset-receipt-bound-v1",
    },
    {
        "path": "runtime_execute.request.action",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "ConcreteAction",
        "validator_id": "six-class-live-browser-action-subset-v1",
    },
    {
        "path": "runtime_observe.result.observation",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "Observation",
        "validator_id": (
            "browsergym-causal-observation-request-bound-"
            "root-confined-screenshot-v2"
        ),
    },
    {
        "path": "runtime_execute.result.execution",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "AdapterExecution",
        "validator_id": "adapter-execution-with-action-evidence-v1",
    },
    {
        "path": "runtime_terminal.request.receipt_binding",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "VerifierReceiptBinding",
        "validator_id": "causal-observation-action-receipt-binding-v1",
    },
    {
        "path": "runtime_terminal.result.opaque_terminal_signal",
        "root_schema_version": "table2.runtime.v1",
        "record_type": "OpaqueTerminalSignal",
        "validator_id": "opaque-terminal-event-token-v1",
    },
)
PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256 = hashlib.sha256(
    canonical_json_bytes(list(PROCESS_BROKER_INNER_SCHEMA_REGISTRY))
).hexdigest()
# Compatibility import for older engineering tests. These paths are no longer
# arbitrary: every one is covered by the source-bound registry above.
ARBITRARY_RUNTIME_MAPPING_PATHS = RUNTIME_INNER_SCHEMA_PATHS
PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS = (
    "ATTEST_RUNTIME_VALUE_PROVENANCE",
    "REGISTER_EXTERNAL_DEPLOYMENT_RECEIPT_SCHEMA_AND_TRUST_ANCHOR",
)
_SEALED_FIELD_TOKENS = frozenset(
    {
        "evaluator",
        "judgment",
        "oracle",
        "page_handle",
        "raw_page",
        "reward",
        "score",
        "success",
        "verifier",
    }
)
_RUNTIME_ACTION_TYPES = frozenset(
    {"CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY"}
)
_ACTION_PARAMETER_FIELDS = {
    "CLICK": {"target_x", "target_y", "target_bbox", "button", "click_count"},
    "TYPE": {"target_x", "target_y", "target_bbox", "text"},
    "SELECT": {
        "target_x",
        "target_y",
        "target_bbox",
        "option",
        "candidate_options",
    },
    "SCROLL": {"direction", "amount", "container"},
    "NAVIGATE": {"url"},
    "PRESS_KEY": {"key"},
}
_OBSERVATION_PAGE_STATE_FIELDS = {
    "schema_version",
    "visible_text",
    "visible_controls",
    "has_browser_error",
    "browser_error_kind",
    "observable_select_controls",
    "recovery_target_evidence",
}
_VISIBLE_CONTROL_FIELDS = {
    "tag",
    "role",
    "input_type",
    "name",
    "text",
    "target_bbox",
    "candidate_options",
    "destination",
}
_SELECT_CONTROL_FIELDS = {"target_bbox", "candidate_options"}
_RECOVERY_TARGET_EVIDENCE_FIELDS = {
    "schema_version",
    "observation_id",
    "task_id",
    "task_goal_sha256",
    "registered_visible_targets",
}
_VISIBLE_TARGET_FIELDS = {"semantic_target_sha256", "compatible_action_types"}
POLICY_SCREENSHOT_TRANSPORT_CONTRACT = (
    "FIXTURE_NONE_OR_ROOT_CONFINED_READ_ONLY_CONTENT_ADDRESSED_PNG_V1"
)


class ProcessBrokerProtocolError(RuntimeError):
    """An IPC frame is malformed, unauthenticated, replayed, or oversized."""


def _normalized_key(value: object) -> str:
    return str(value).casefold().replace("-", "_").replace(" ", "_")


def forbidden_runtime_fields(value: Any) -> set[str]:
    """Return registered sensitive *key aliases* in a runtime JSON value.

    This is deliberately not a semantic inspection of values under neutral
    keys. Operation-specific schemas are registered separately below; external
    value provenance remains a future promotion requirement.
    """

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if any(token in normalized for token in _SEALED_FIELD_TOKENS):
                found.add(normalized)
            found.update(forbidden_runtime_fields(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(forbidden_runtime_fields(nested))
    return found


def _exact_fields(
    value: object, expected: set[str], *, context: str
) -> dict[str, Any]:
    mapping = _json_object(value, context=context)
    if set(mapping) != expected:
        raise ProcessBrokerProtocolError(f"{context} fields differ from schema")
    return mapping


def _runtime_identity_fields(payload: Mapping[str, Any], *, context: str) -> None:
    for field in ("episode_id", "task_id"):
        value = payload.get(field)
        if type(value) is not str or not value or len(value) > 512:
            raise ProcessBrokerProtocolError(f"{context} {field} is invalid")


def _strict_record(
    value: object, record_type: type[Any], *, context: str
) -> tuple[dict[str, Any], Any]:
    """Decode one existing runtime record and reject every wire-level drift."""

    mapping = _json_object(value, context=context)
    try:
        restored = record_type.from_dict(mapping)
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            f"{context} does not satisfy {record_type.__name__}"
        ) from exc
    if canonical_json_bytes(restored.to_dict()) != canonical_json_bytes(mapping):
        raise ProcessBrokerProtocolError(
            f"{context} differs from canonical {record_type.__name__}"
        )
    return mapping, restored


def _finite_number(value: object, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ProcessBrokerProtocolError(f"{context} must be a finite number")
    return float(value)


def _normalized_bbox(value: object, *, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ProcessBrokerProtocolError(f"{context} must be a four-value JSON array")
    result = tuple(
        _finite_number(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ProcessBrokerProtocolError(f"{context} must be normalized to [0, 1]")
    x, y, width, height = result
    if x + width > 1.0 or y + height > 1.0:
        raise ProcessBrokerProtocolError(f"{context} falls outside the viewport")
    return result


def _lowercase_sha256(value: object, *, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessBrokerProtocolError(f"{context} must be lowercase SHA-256")
    return value


def validated_policy_screenshot_root(value: object) -> Path | None:
    """Return one canonical screenshot directory or reject its capability."""

    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ProcessBrokerProtocolError(
            "policy screenshot root must be an absolute directory path"
        )
    unresolved = Path(value)
    if not unresolved.is_absolute() or unresolved.name != "screenshots":
        raise ProcessBrokerProtocolError(
            "policy screenshot root must be an absolute screenshots directory"
        )
    try:
        metadata = unresolved.lstat()
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ProcessBrokerProtocolError(
            "policy screenshot root is absent or inaccessible"
        ) from exc
    if (
        unresolved.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != unresolved
    ):
        raise ProcessBrokerProtocolError(
            "policy screenshot root is not a canonical non-symlink directory"
        )
    return resolved


def _validate_policy_screenshot_path(
    *, observation: Observation, policy_screenshot_root: str | Path | None
) -> None:
    path_value = observation.screenshot_path
    root = validated_policy_screenshot_root(policy_screenshot_root)
    if path_value is None:
        # Deterministic non-browser fixtures intentionally have neither a root
        # nor an image. Registering a root denotes an image-backed live
        # session, for which the selected Qwen policy requires every frame.
        if root is not None:
            raise ProcessBrokerProtocolError(
                "runtime observation requires a screenshot in the registered root"
            )
        return
    if type(path_value) is not str:
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot path must be text"
        )
    if root is None:
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot lacks a registered path root"
        )
    unresolved = Path(path_value)
    if not unresolved.is_absolute():
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot path must be absolute"
        )
    try:
        metadata = unresolved.lstat()
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot is absent or inaccessible"
        ) from exc
    if (
        unresolved.is_symlink()
        or resolved != unresolved
        or resolved.parent != root
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or bool(metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    ):
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot escapes its immutable root contract"
        )
    if not re.fullmatch(
        rf"[0-9]{{6}}-{re.escape(observation.screenshot_sha256)}\.png",
        resolved.name,
    ):
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot filename is not content-addressed"
        )
    if sha256_file(resolved) != observation.screenshot_sha256:
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot bytes differ from registered SHA-256"
        )
    try:
        with resolved.open("rb") as handle:
            signature = handle.read(8)
    except OSError as exc:
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot cannot be read"
        ) from exc
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ProcessBrokerProtocolError(
            "runtime observation screenshot is not a PNG artifact"
        )


def _select_scalar(value: object, *, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ProcessBrokerProtocolError(
            f"{context} must be a string, integer, or finite float"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ProcessBrokerProtocolError(
            f"{context} must be a string, integer, or finite float"
        )


def _absolute_runtime_url(value: object, *, context: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\\" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ProcessBrokerProtocolError(f"{context} is not a safe absolute URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ProcessBrokerProtocolError(f"{context} is not parseable") from exc
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or port == 0
    ):
        raise ProcessBrokerProtocolError(
            f"{context} is outside the registered URL schema"
        )
    return value


def _validate_action_parameters(action: ConcreteAction) -> None:
    for field, value in (
        ("action_id", action.action_id),
        ("source_decision_id", action.source_decision_id),
    ):
        if type(value) is not str or not value or len(value) > 512:
            raise ProcessBrokerProtocolError(
                f"runtime action {field} is not bounded text"
            )
    if action.recovery_attempt_id is not None and (
        type(action.recovery_attempt_id) is not str
        or not action.recovery_attempt_id
        or len(action.recovery_attempt_id) > 512
    ):
        raise ProcessBrokerProtocolError(
            "runtime action recovery_attempt_id is not bounded text"
        )
    if type(action.destructive) is not bool or action.destructive:
        raise ProcessBrokerProtocolError(
            "runtime action destructive flag must be exact false"
        )
    action_name = action.action_type.value
    if action_name not in _RUNTIME_ACTION_TYPES:
        raise ProcessBrokerProtocolError("runtime action type is not registered")
    parameters = dict(action.parameters)
    expected = _ACTION_PARAMETER_FIELDS[action_name]
    if set(parameters) != expected:
        raise ProcessBrokerProtocolError(
            f"runtime {action_name} parameter fields differ from schema"
        )
    if action_name in {"CLICK", "TYPE", "SELECT"}:
        if action.bbox is None:
            raise ProcessBrokerProtocolError(f"runtime {action_name} requires grounding")
        bbox = _normalized_bbox(list(action.bbox), context=f"runtime {action_name} bbox")
        target_bbox = _normalized_bbox(
            parameters["target_bbox"], context=f"runtime {action_name} target_bbox"
        )
        if target_bbox != bbox:
            raise ProcessBrokerProtocolError(
                f"runtime {action_name} target_bbox differs from grounding"
            )
        target_x = _finite_number(parameters["target_x"], context="runtime target_x")
        target_y = _finite_number(parameters["target_y"], context="runtime target_y")
        x, y, width, height = bbox
        if not (
            0.0 <= target_x <= 1.0
            and 0.0 <= target_y <= 1.0
            and x <= target_x <= x + width
            and y <= target_y <= y + height
        ):
            raise ProcessBrokerProtocolError(
                f"runtime {action_name} target point is outside target_bbox"
            )
    if action_name == "CLICK":
        if parameters["button"] not in {"left", "middle", "right"}:
            raise ProcessBrokerProtocolError("runtime CLICK button is not registered")
        if (
            type(parameters["click_count"]) is not int
            or parameters["click_count"] not in {1, 2, 3}
        ):
            raise ProcessBrokerProtocolError("runtime CLICK count is not registered")
    elif action_name == "TYPE":
        if type(parameters["text"]) is not str or not parameters["text"]:
            raise ProcessBrokerProtocolError(
                "runtime TYPE text must be a non-empty string"
            )
    elif action_name == "SELECT":
        _select_scalar(parameters["option"], context="runtime SELECT option")
        candidates = parameters["candidate_options"]
        if not isinstance(candidates, list) or not candidates:
            raise ProcessBrokerProtocolError(
                "runtime SELECT candidates must be a non-empty JSON array"
            )
        for index, candidate in enumerate(candidates):
            _select_scalar(candidate, context=f"runtime SELECT candidate[{index}]")
        if not any(
            type(candidate) is type(parameters["option"])
            and candidate == parameters["option"]
            for candidate in candidates
        ):
            raise ProcessBrokerProtocolError(
                "runtime SELECT option is absent from candidate_options"
            )
    elif action_name == "SCROLL":
        if parameters["direction"] not in {"up", "down", "left", "right"}:
            raise ProcessBrokerProtocolError("runtime SCROLL direction is not registered")
        if _finite_number(
            parameters["amount"], context="runtime SCROLL amount"
        ) not in {0.5, 0.75}:
            raise ProcessBrokerProtocolError("runtime SCROLL amount is not registered")
        if parameters["container"] != "viewport":
            raise ProcessBrokerProtocolError("runtime SCROLL container is not registered")
    elif action_name == "NAVIGATE":
        _absolute_runtime_url(parameters["url"], context="runtime NAVIGATE url")
    elif action_name == "PRESS_KEY" and parameters["key"] not in {
        "ENTER",
        "TAB",
        "ESCAPE",
        "ARROWDOWN",
        "ARROWUP",
        "SPACE",
        "ALT+LEFT",
    }:
        raise ProcessBrokerProtocolError("runtime PRESS_KEY command is not registered")


def _validate_runtime_action(value: object) -> dict[str, Any]:
    mapping, action = _strict_record(
        value, ConcreteAction, context="runtime_execute request action"
    )
    _validate_action_parameters(action)
    return mapping


def _validate_visible_control(value: object, *, index: int) -> dict[str, Any]:
    control = _exact_fields(
        value,
        _VISIBLE_CONTROL_FIELDS,
        context=f"runtime observation visible_controls[{index}]",
    )
    limits = {
        "tag": 32,
        "role": 64,
        "input_type": 32,
        "name": 160,
        "text": 240,
    }
    for field, maximum in limits.items():
        field_value = control[field]
        if type(field_value) is not str or len(field_value) > maximum:
            raise ProcessBrokerProtocolError(
                f"runtime observation control {field} is invalid"
            )
    if control["tag"] != control["tag"].lower() or control["input_type"] != control[
        "input_type"
    ].lower():
        raise ProcessBrokerProtocolError(
            "runtime observation control tag/input_type is not normalized"
        )
    control_bbox = _normalized_bbox(
        control["target_bbox"],
        context=f"runtime observation control[{index}] target_bbox",
    )
    if control_bbox[2] <= 0.0 or control_bbox[3] <= 0.0:
        raise ProcessBrokerProtocolError(
            "runtime observation visible-control bbox must have positive area"
        )
    candidates = control["candidate_options"]
    if not isinstance(candidates, list) or len(candidates) > 256:
        raise ProcessBrokerProtocolError(
            "runtime observation candidate_options schema differs"
        )
    for candidate_index, candidate in enumerate(candidates):
        _select_scalar(
            candidate,
            context=(
                f"runtime observation control[{index}] "
                f"candidate_options[{candidate_index}]"
            ),
        )
    destination = control["destination"]
    if type(destination) is not str:
        raise ProcessBrokerProtocolError(
            "runtime observation control destination must be text"
        )
    if destination:
        try:
            parsed = urlsplit(destination)
            port = parsed.port
        except ValueError as exc:
            raise ProcessBrokerProtocolError(
                "runtime observation control destination is not parseable"
            ) from exc
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "@" in parsed.netloc
            or port == 0
        ):
            raise ProcessBrokerProtocolError(
                "runtime observation control destination is not causal HTTP(S) data"
            )
    return control


def _validate_observation_page_state(
    value: object,
    *,
    observation: Observation,
    request_payload: Mapping[str, Any] | None,
) -> None:
    state = _exact_fields(
        value,
        _OBSERVATION_PAGE_STATE_FIELDS,
        context="runtime observation page_state",
    )
    if state["schema_version"] != "table2-browsergym-causal-observation-v1":
        raise ProcessBrokerProtocolError("runtime observation page_state schema differs")
    if observation.page_settled is not True:
        raise ProcessBrokerProtocolError(
            "runtime BrowserGym observation must be settled"
        )
    if (
        observation.stage.value in {"reset", "pre_action"}
        and observation.prior_action_id is not None
    ):
        raise ProcessBrokerProtocolError(
            "runtime pre-action/reset observation cannot cite a prior action"
        )
    if type(state["visible_text"]) is not str or len(state["visible_text"]) > 8192:
        raise ProcessBrokerProtocolError("runtime observation visible_text is invalid")
    controls_value = state["visible_controls"]
    if not isinstance(controls_value, list) or len(controls_value) > 512:
        raise ProcessBrokerProtocolError("runtime observation controls are invalid")
    controls = [
        _validate_visible_control(control, index=index)
        for index, control in enumerate(controls_value)
    ]
    has_error = state["has_browser_error"]
    error_kind = state["browser_error_kind"]
    if type(has_error) is not bool or has_error is not observation.environment_error:
        raise ProcessBrokerProtocolError(
            "runtime observation browser error flag contradicts Observation"
        )
    if has_error:
        if type(error_kind) is not str or not re.fullmatch(
            r"[A-Z][A-Z0-9_]{0,95}", error_kind
        ):
            raise ProcessBrokerProtocolError(
                "runtime observation browser_error_kind is invalid"
            )
    elif error_kind is not None:
        raise ProcessBrokerProtocolError(
            "runtime observation non-error cannot name browser_error_kind"
        )
    select_controls = state["observable_select_controls"]
    if not isinstance(select_controls, list):
        raise ProcessBrokerProtocolError(
            "runtime observation observable_select_controls must be an array"
        )
    expected_select_controls = [
        {
            "target_bbox": list(control["target_bbox"]),
            "candidate_options": list(control["candidate_options"]),
        }
        for control in controls
        if control["tag"] == "select" and control["candidate_options"]
    ]
    for index, select_control in enumerate(select_controls):
        normalized = _exact_fields(
            select_control,
            _SELECT_CONTROL_FIELDS,
            context=f"runtime observation observable_select_controls[{index}]",
        )
        _normalized_bbox(
            normalized["target_bbox"],
            context=f"runtime observation select_control[{index}] target_bbox",
        )
        candidates = normalized["candidate_options"]
        if not isinstance(candidates, list):
            raise ProcessBrokerProtocolError(
                "runtime observation select-control candidates must be an array"
            )
        for candidate_index, candidate in enumerate(candidates):
            _select_scalar(
                candidate,
                context=(
                    f"runtime observation select_control[{index}] "
                    f"candidate_options[{candidate_index}]"
                ),
            )
    if select_controls != expected_select_controls:
        raise ProcessBrokerProtocolError(
            "runtime observation select controls differ from visible controls"
        )
    recovery = _exact_fields(
        state["recovery_target_evidence"],
        _RECOVERY_TARGET_EVIDENCE_FIELDS,
        context="runtime observation recovery_target_evidence",
    )
    if recovery["schema_version"] != "oracle-blind-visible-targets-v1":
        raise ProcessBrokerProtocolError(
            "runtime observation recovery-target schema differs"
        )
    if recovery["observation_id"] != observation.observation_id:
        raise ProcessBrokerProtocolError(
            "runtime observation recovery evidence has another observation ID"
        )
    _lowercase_sha256(
        recovery["task_goal_sha256"],
        context="runtime observation recovery task_goal_sha256",
    )
    if request_payload is not None and (
        observation.episode_id != request_payload.get("episode_id")
        or recovery["task_id"] != request_payload.get("task_id")
    ):
        raise ProcessBrokerProtocolError(
            "runtime observation result belongs to another episode/task"
        )
    targets = recovery["registered_visible_targets"]
    if not isinstance(targets, list):
        raise ProcessBrokerProtocolError(
            "runtime observation registered_visible_targets must be an array"
        )
    previous: str | None = None
    for index, target in enumerate(targets):
        row = _exact_fields(
            target,
            _VISIBLE_TARGET_FIELDS,
            context=f"runtime observation registered_visible_targets[{index}]",
        )
        fingerprint = _lowercase_sha256(
            row["semantic_target_sha256"],
            context=f"runtime observation target[{index}] fingerprint",
        )
        if previous is not None and fingerprint <= previous:
            raise ProcessBrokerProtocolError(
                "runtime observation target fingerprints must be unique and sorted"
            )
        action_types = row["compatible_action_types"]
        if (
            not isinstance(action_types, list)
            or not action_types
            or any(
                type(item) is not str or item not in _RUNTIME_ACTION_TYPES
                for item in action_types
            )
            or action_types != sorted(set(action_types))
        ):
            raise ProcessBrokerProtocolError(
                "runtime observation compatible action types are invalid"
            )
        previous = fingerprint


def _validate_runtime_observation(
    value: object,
    *,
    request_payload: Mapping[str, Any] | None,
    policy_screenshot_root: str | Path | None,
) -> dict[str, Any]:
    if request_payload is None:
        raise ProcessBrokerProtocolError(
            "runtime_observe result requires its validated request binding"
        )
    mapping, observation = _strict_record(
        value, Observation, context="runtime_observe result observation"
    )
    _lowercase_sha256(
        observation.screenshot_sha256,
        context="runtime observation screenshot_sha256",
    )
    _validate_policy_screenshot_path(
        observation=observation,
        policy_screenshot_root=policy_screenshot_root,
    )
    _absolute_runtime_url(
        observation.url,
        context="runtime observation url",
    )
    if (
        type(observation.observation_id) is not str
        or not observation.observation_id
        or len(observation.observation_id) > 512
        or type(observation.episode_id) is not str
        or len(observation.episode_id) > 512
        or type(observation.title) is not str
        or len(observation.title) > 1024
        or type(observation.width) is not int
        or not 1 <= observation.width <= 8192
        or type(observation.height) is not int
        or not 1 <= observation.height <= 8192
        or type(observation.environment_error) is not bool
    ):
        raise ProcessBrokerProtocolError(
            "runtime observation display metadata exceeds registered bounds"
        )
    requested_stage = request_payload.get("stage")
    requested_prior_action_id = request_payload.get("prior_action_id")
    if (
        observation.stage.value != requested_stage
        or observation.prior_action_id != requested_prior_action_id
    ):
        raise ProcessBrokerProtocolError(
            "runtime observation stage/prior action differs from its request"
        )
    _validate_observation_page_state(
        observation.page_state,
        observation=observation,
        request_payload=request_payload,
    )
    return mapping


def _validate_runtime_execution(
    value: object, *, request_payload: Mapping[str, Any] | None
) -> dict[str, Any]:
    if request_payload is None:
        raise ProcessBrokerProtocolError(
            "runtime_execute result requires its validated request binding"
        )
    mapping, execution = _strict_record(
        value, AdapterExecution, context="runtime_execute result execution"
    )
    if (
        type(execution.state_changed) is not bool
        or type(execution.environment_error) is not bool
        or type(execution.internal_retry_count) is not int
        or type(execution.message) is not str
        or len(execution.message) > 4096
        or (
            execution.error_kind is not None
            and (
                type(execution.error_kind) is not str
                or not execution.error_kind
                or len(execution.error_kind) > 128
            )
        )
    ):
        raise ProcessBrokerProtocolError(
            "runtime execution scalar fields differ from schema"
        )
    if execution.evidence is None:
        raise ProcessBrokerProtocolError(
            "runtime execution requires action-bound ExecutionEvidence"
        )
    if execution.internal_retry_count != 0:
        raise ProcessBrokerProtocolError(
            "runtime execution internal retries are outside the pilot schema"
        )
    action = request_payload.get("action")
    if not isinstance(action, Mapping) or execution.evidence.action_id != action.get(
        "action_id"
    ):
        raise ProcessBrokerProtocolError(
            "runtime execution evidence belongs to another action"
        )
    return mapping


def _validate_reset_request(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "episode_id",
        "task_id",
        "benchmark_version",
        "start_state_id",
        "reset_stage_seed",
    }
    mapping = _exact_fields(value, expected, context="runtime_reset request")
    for field, maximum in (
        ("episode_id", 512),
        ("task_id", 512),
        ("benchmark_version", 256),
    ):
        field_value = mapping[field]
        if (
            type(field_value) is not str
            or not field_value
            or field_value != field_value.strip()
            or len(field_value) > maximum
        ):
            raise ProcessBrokerProtocolError(
                f"runtime_reset request {field} is invalid"
            )
    _lowercase_sha256(
        mapping["start_state_id"], context="runtime_reset request start_state_id"
    )
    reset_stage_seed = mapping["reset_stage_seed"]
    if (
        type(reset_stage_seed) is not int
        or reset_stage_seed < 0
        or reset_stage_seed > 2**63 - 1
    ):
        raise ProcessBrokerProtocolError(
            "runtime_reset request reset_stage_seed is invalid"
        )
    try:
        request = WebArenaResetStateRequest(**mapping)
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            "runtime_reset request does not satisfy WebArenaResetStateRequest"
        ) from exc
    if request.episode_id != mapping["episode_id"] or request.task_id != mapping[
        "task_id"
    ]:
        raise ProcessBrokerProtocolError("runtime_reset request identity changed")
    return mapping


def _validate_reset_receipt(
    value: object, *, request_payload: Mapping[str, Any] | None
) -> dict[str, Any]:
    if request_payload is None:
        raise ProcessBrokerProtocolError(
            "runtime_reset result requires its validated request binding"
        )
    mapping, receipt = _strict_record(
        value,
        WebArenaResetStateReceipt,
        context="runtime_reset result reset_state_receipt",
    )
    for field in (
        "episode_id",
        "task_id",
        "benchmark_version",
        "start_state_id",
        "reset_stage_seed",
    ):
        if getattr(receipt, field) != request_payload.get(field):
            raise ProcessBrokerProtocolError(
                f"runtime_reset receipt {field} differs from request"
            )
    return mapping


def _validate_terminal_binding(value: object) -> dict[str, Any]:
    mapping, _binding = _strict_record(
        value,
        VerifierReceiptBinding,
        context="runtime_terminal request receipt_binding",
    )
    return mapping


def _validate_terminal_signal(value: object) -> dict[str, Any]:
    mapping, _signal = _strict_record(
        value,
        OpaqueTerminalSignal,
        context="runtime_terminal result opaque_terminal_signal",
    )
    return mapping


def expected_verifier_receipt_binding(
    *, observation: Mapping[str, Any], action: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Build the only receipt binding allowed for the current causal state."""

    observation_id = observation.get("observation_id")
    if type(observation_id) is not str or not observation_id:
        raise ProcessBrokerProtocolError(
            "terminal binding source observation has no identity"
        )
    if action is None:
        binding = VerifierReceiptBinding(
            receipt_kind="after_reset",
            observation_id=observation_id,
            observation_sha256=hashlib.sha256(
                canonical_json_bytes(observation)
            ).hexdigest(),
        )
    else:
        action_id = action.get("action_id")
        if type(action_id) is not str or not action_id:
            raise ProcessBrokerProtocolError(
                "terminal binding source action has no identity"
            )
        binding = VerifierReceiptBinding(
            receipt_kind=(
                "after_recovery_action"
                if action.get("recovery_attempt_id") is not None
                else "after_normal_action"
            ),
            observation_id=observation_id,
            observation_sha256=hashlib.sha256(
                canonical_json_bytes(observation)
            ).hexdigest(),
            action_id=action_id,
            action_sha256=hashlib.sha256(canonical_json_bytes(action)).hexdigest(),
        )
    return binding.to_dict()


def validate_runtime_request_payload(
    operation: object, payload: object
) -> dict[str, Any]:
    """Validate one exact operation payload before replay state is consumed."""

    if operation not in RUNTIME_BROKER_OPERATIONS:
        raise ProcessBrokerProtocolError("runtime broker operation is forbidden")
    expected = {"episode_id", "task_id"}
    if operation == "runtime_reset":
        expected.update(
            {"benchmark_version", "start_state_id", "reset_stage_seed"}
        )
    elif operation == "runtime_execute":
        expected.add("action")
    elif operation == "runtime_observe":
        expected.update({"stage", "prior_action_id"})
    elif operation == "runtime_terminal":
        expected.add("receipt_binding")
    normalized = _exact_fields(
        payload, expected, context=f"{operation} request payload"
    )
    _runtime_identity_fields(normalized, context=str(operation))
    if operation == "runtime_reset":
        normalized = _validate_reset_request(normalized)
    elif operation == "runtime_observe":
        stage = normalized.get("stage")
        prior_action_id = normalized.get("prior_action_id")
        if type(stage) is not str or stage not in {
            "post_action",
            "post_recovery",
        }:
            raise ProcessBrokerProtocolError(
                "runtime_observe stage is not registered"
            )
        if (
            type(prior_action_id) is not str
            or not prior_action_id
            or len(prior_action_id) > 512
        ):
            raise ProcessBrokerProtocolError(
                "runtime post observation requires a bounded prior action ID"
            )
    if operation == "runtime_execute":
        action = normalized.get("action")
        if not isinstance(action, Mapping):
            raise ProcessBrokerProtocolError("runtime_execute action must be an object")
    forbidden = forbidden_runtime_fields(normalized)
    if forbidden:
        raise ProcessBrokerProtocolError(
            "runtime payload contains forbidden named keys: "
            + ",".join(sorted(forbidden))
        )
    if operation == "runtime_execute":
        normalized["action"] = _validate_runtime_action(normalized["action"])
    elif operation == "runtime_terminal":
        normalized["receipt_binding"] = _validate_terminal_binding(
            normalized["receipt_binding"]
        )
    return normalized


def validate_runtime_result(
    operation: object,
    result: object,
    *,
    request_payload: Mapping[str, Any] | None = None,
    policy_screenshot_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact result envelope for one registered operation."""

    expected_by_operation = {
        "runtime_reset": {"observation", "reset_state_receipt"},
        "runtime_observe": {"observation"},
        "runtime_execute": {"execution"},
        "runtime_terminal": {"opaque_terminal_signal"},
        "runtime_close": {"closed"},
    }
    expected = expected_by_operation.get(operation)
    if expected is None:
        raise ProcessBrokerProtocolError("runtime result operation is forbidden")
    normalized = _exact_fields(
        result, expected, context=f"{operation} result"
    )
    object_fields = {
        "runtime_reset": ("observation", "reset_state_receipt"),
        "runtime_observe": ("observation",),
        "runtime_execute": ("execution",),
        "runtime_terminal": ("opaque_terminal_signal",),
    }
    for field in object_fields.get(str(operation), ()):
        if not isinstance(normalized.get(field), Mapping):
            raise ProcessBrokerProtocolError(f"{operation} {field} must be an object")
    if operation == "runtime_close" and type(normalized.get("closed")) is not bool:
        raise ProcessBrokerProtocolError("runtime_close closed must be boolean")
    forbidden_view: Mapping[str, Any] = normalized
    if operation == "runtime_reset":
        # This one exact field is a registered negative attestation inside a
        # strict hashes-only record. It is not evaluator truth.
        receipt_view = dict(normalized["reset_state_receipt"])
        receipt_view.pop("oracle_labels_observed", None)
        forbidden_view = {**normalized, "reset_state_receipt": receipt_view}
    forbidden = forbidden_runtime_fields(forbidden_view)
    if forbidden:
        raise ProcessBrokerProtocolError(
            "sealed backend returned forbidden named keys: "
            + ",".join(sorted(forbidden))
        )
    if operation == "runtime_reset":
        synthetic_request = {
            "episode_id": request_payload.get("episode_id")
            if request_payload is not None
            else None,
            "task_id": request_payload.get("task_id")
            if request_payload is not None
            else None,
            "stage": "reset",
            "prior_action_id": None,
        }
        normalized["observation"] = _validate_runtime_observation(
            normalized["observation"],
            request_payload=synthetic_request,
            policy_screenshot_root=policy_screenshot_root,
        )
        normalized["reset_state_receipt"] = _validate_reset_receipt(
            normalized["reset_state_receipt"], request_payload=request_payload
        )
    elif operation == "runtime_observe":
        normalized["observation"] = _validate_runtime_observation(
            normalized["observation"],
            request_payload=request_payload,
            policy_screenshot_root=policy_screenshot_root,
        )
    elif operation == "runtime_execute":
        normalized["execution"] = _validate_runtime_execution(
            normalized["execution"], request_payload=request_payload
        )
    elif operation == "runtime_terminal":
        normalized["opaque_terminal_signal"] = _validate_terminal_signal(
            normalized["opaque_terminal_signal"]
        )
    return normalized


def _json_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessBrokerProtocolError(f"{context} must be a JSON object")
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(f"{context} is not canonical JSON") from exc
    if len(encoded) > MAX_PROCESS_BROKER_MESSAGE_BYTES:
        raise ProcessBrokerProtocolError(f"{context} exceeds the IPC size limit")
    # Never retain nested references supplied by a caller or backend.  The
    # round-trip is also the single canonical JSON type normalization point.
    detached = json.loads(encoded)
    if not isinstance(detached, dict):  # defensive; ``value`` was a Mapping
        raise ProcessBrokerProtocolError(f"{context} must be a JSON object")
    return detached


def authenticated_envelope(
    body: Mapping[str, Any], *, authentication_key: bytes
) -> dict[str, Any]:
    body_value = _json_object(body, context="process-broker body")
    if len(authentication_key) < 32:
        raise ProcessBrokerProtocolError("process-broker key must contain 256 bits")
    return {
        "body": body_value,
        "hmac_sha256": hmac.new(
            authentication_key,
            canonical_json_bytes(body_value),
            hashlib.sha256,
        ).hexdigest(),
    }


def verify_authenticated_envelope(
    value: object, *, authentication_key: bytes
) -> dict[str, Any]:
    envelope = _json_object(value, context="process-broker envelope")
    if set(envelope) != {"body", "hmac_sha256"}:
        raise ProcessBrokerProtocolError("process-broker envelope fields differ")
    body = _json_object(envelope.get("body"), context="process-broker body")
    supplied = envelope.get("hmac_sha256")
    expected = hmac.new(
        authentication_key, canonical_json_bytes(body), hashlib.sha256
    ).hexdigest()
    if type(supplied) is not str or not hmac.compare_digest(supplied, expected):
        raise ProcessBrokerProtocolError("process-broker authentication failed")
    return body


def send_frame(connection: socket.socket, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(_json_object(value, context="process-broker frame"))
    if len(payload) > MAX_PROCESS_BROKER_MESSAGE_BYTES:
        raise ProcessBrokerProtocolError("process-broker frame exceeds size limit")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProcessBrokerProtocolError("process-broker connection closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(connection: socket.socket) -> dict[str, Any]:
    header = _read_exact(connection, 4)
    size = struct.unpack("!I", header)[0]
    if size <= 0 or size > MAX_PROCESS_BROKER_MESSAGE_BYTES:
        raise ProcessBrokerProtocolError("process-broker frame length is invalid")
    payload = _read_exact(connection, size)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ProcessBrokerProtocolError(
                    "process-broker frame contains a duplicate JSON key"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProcessBrokerProtocolError("process-broker frame is not JSON") from exc
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            "process-broker frame is not canonical JSON"
        ) from exc
    if canonical != payload:
        raise ProcessBrokerProtocolError(
            "process-broker frame is not canonical JSON bytes"
        )
    return _json_object(value, context="process-broker frame")
