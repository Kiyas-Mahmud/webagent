"""Runtime-only client for the process-isolated page/evaluator owner.

This module intentionally imports neither the worker nor an evaluator module.
Its public API contains only observation, execution, opaque-terminal, and
session-close operations. The server independently enforces that operation
allow-list and exact outer envelopes. Observation, execution, and action values
are still arbitrary mappings: recursive sensitive-key rejection is not value-
provenance or semantic oracle-isolation evidence.
"""

from __future__ import annotations

import secrets
import socket
from pathlib import Path
from typing import Any, Mapping

from .process_broker_protocol import (
    PROCESS_BROKER_PROTOCOL_VERSION,
    RUNTIME_BROKER_OPERATIONS,
    ProcessBrokerProtocolError,
    authenticated_envelope,
    receive_frame,
    send_frame,
    validate_runtime_request_payload,
    validate_runtime_result,
    verify_authenticated_envelope,
)


class ProcessIsolatedRuntimeClient:
    """Operation-limited authenticated capability with no evaluator operation."""

    __slots__ = ("__endpoint", "__key", "__session_id", "__sequence", "__timeout")

    def __init__(
        self,
        *,
        endpoint: str | Path,
        authentication_key: bytes,
        session_id: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not session_id or len(authentication_key) < 32:
            raise ValueError("runtime broker client requires session/key identity")
        self.__endpoint = str(endpoint)
        self.__key = bytes(authentication_key)
        self.__session_id = session_id
        self.__sequence = 0
        self.__timeout = float(timeout_seconds)

    def __request(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        validated_payload = validate_runtime_request_payload(operation, payload)
        sequence = self.__sequence
        self.__sequence += 1
        nonce = secrets.token_hex(32)
        body = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "runtime",
            "session_id": self.__session_id,
            "sequence": sequence,
            "nonce": nonce,
            "operation": operation,
            "payload": validated_payload,
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.__timeout)
            connection.connect(self.__endpoint)
            send_frame(
                connection,
                authenticated_envelope(body, authentication_key=self.__key),
            )
            response = verify_authenticated_envelope(
                receive_frame(connection), authentication_key=self.__key
            )
        expected = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "sealed_runtime_response",
            "session_id": self.__session_id,
            "sequence": sequence,
            "nonce": nonce,
        }
        if any(response.get(field) != value for field, value in expected.items()):
            raise ProcessBrokerProtocolError("process-broker response binding differs")
        status = response.get("status")
        expected_fields = set(expected) | {"status", "result"}
        if status == "REJECTED":
            expected_fields.add("error_code")
        if set(response) != expected_fields:
            raise ProcessBrokerProtocolError("process-broker response fields differ")
        if status != "PASS":
            raise ProcessBrokerProtocolError(
                f"process-broker request rejected: {response.get('error_code')}"
            )
        result = response.get("result")
        return validate_runtime_result(operation, result)

    def observe(self, *, episode_id: str, task_id: str) -> dict[str, Any]:
        result = self.__request(
            "runtime_observe", {"episode_id": episode_id, "task_id": task_id}
        )
        return dict(result["observation"])

    def execute(
        self, *, episode_id: str, task_id: str, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = self.__request(
            "runtime_execute",
            {"episode_id": episode_id, "task_id": task_id, "action": dict(action)},
        )
        return dict(result["execution"])

    def terminal_signal(self, *, episode_id: str, task_id: str) -> bool:
        result = self.__request(
            "runtime_terminal", {"episode_id": episode_id, "task_id": task_id}
        )
        if set(result) != {"opaque_terminal"} or type(result["opaque_terminal"]) is not bool:
            raise ProcessBrokerProtocolError("terminal response is not opaque")
        return result["opaque_terminal"]

    def close(self, *, episode_id: str, task_id: str) -> dict[str, Any]:
        return self.__request(
            "runtime_close", {"episode_id": episode_id, "task_id": task_id}
        )
