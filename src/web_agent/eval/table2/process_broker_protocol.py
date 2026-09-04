"""Authenticated JSON framing shared by isolated Table 2 process roles."""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
from typing import Any, Mapping

from .common import canonical_json_bytes


PROCESS_BROKER_PROTOCOL_VERSION = "table2-process-page-broker-ipc-v1"
MAX_PROCESS_BROKER_MESSAGE_BYTES = 1_048_576
RUNTIME_BROKER_OPERATIONS = frozenset(
    {"runtime_observe", "runtime_execute", "runtime_terminal", "runtime_close"}
)
ARBITRARY_RUNTIME_MAPPING_PATHS = (
    "runtime_execute.request.action",
    "runtime_observe.result.observation",
    "runtime_execute.result.execution",
)
PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS = (
    "REGISTER_EXACT_OPERATION_SPECIFIC_INNER_SCHEMAS",
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


class ProcessBrokerProtocolError(RuntimeError):
    """An IPC frame is malformed, unauthenticated, replayed, or oversized."""


def _normalized_key(value: object) -> str:
    return str(value).casefold().replace("-", "_").replace(" ", "_")


def forbidden_runtime_fields(value: Any) -> set[str]:
    """Return registered sensitive *key aliases* in a runtime JSON value.

    This is deliberately not a semantic inspection of values under neutral
    keys. Operation-specific inner schemas and value provenance remain
    unregistered future promotion requirements.
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


def validate_runtime_request_payload(
    operation: object, payload: object
) -> dict[str, Any]:
    """Validate one exact operation payload before replay state is consumed."""

    if operation not in RUNTIME_BROKER_OPERATIONS:
        raise ProcessBrokerProtocolError("runtime broker operation is forbidden")
    expected = {"episode_id", "task_id"}
    if operation == "runtime_execute":
        expected.add("action")
    normalized = _exact_fields(
        payload, expected, context=f"{operation} request payload"
    )
    _runtime_identity_fields(normalized, context=str(operation))
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
    return normalized


def validate_runtime_result(operation: object, result: object) -> dict[str, Any]:
    """Validate the exact result envelope for one registered operation."""

    expected_by_operation = {
        "runtime_observe": {"observation"},
        "runtime_execute": {"execution"},
        "runtime_terminal": {"opaque_terminal"},
        "runtime_close": {"closed"},
    }
    expected = expected_by_operation.get(operation)
    if expected is None:
        raise ProcessBrokerProtocolError("runtime result operation is forbidden")
    normalized = _exact_fields(
        result, expected, context=f"{operation} result"
    )
    if operation in {"runtime_observe", "runtime_execute"}:
        field = "observation" if operation == "runtime_observe" else "execution"
        if not isinstance(normalized.get(field), Mapping):
            raise ProcessBrokerProtocolError(f"{operation} {field} must be an object")
    else:
        field = "opaque_terminal" if operation == "runtime_terminal" else "closed"
        if type(normalized.get(field)) is not bool:
            raise ProcessBrokerProtocolError(f"{operation} {field} must be boolean")
    forbidden = forbidden_runtime_fields(normalized)
    if forbidden:
        raise ProcessBrokerProtocolError(
            "sealed backend returned forbidden named keys: "
            + ",".join(sorted(forbidden))
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
    return dict(value)


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
    try:
        value = json.loads(_read_exact(connection, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProcessBrokerProtocolError("process-broker frame is not JSON") from exc
    return _json_object(value, context="process-broker frame")
