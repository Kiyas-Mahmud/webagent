"""Runtime-only client for the process-isolated page/evaluator owner.

This module intentionally imports neither the worker nor an evaluator module.
Its public API contains only reset, observation, execution, opaque-terminal,
and session-close operations. The server independently enforces that operation
allow-list, exact outer envelopes, and all eight registered reset/action/
observation/execution/verifier request and result schemas. Schema conformance
is additionally bound to one episode/task and a reset/execute/observe/verifier
causal state. It does not attest where scalar runtime-visible values
originated; external value-provenance and deployment evidence remain
mandatory.
"""

from __future__ import annotations

import json
import secrets
import socket
from pathlib import Path
from typing import Any, Mapping

from .process_broker_protocol import (
    PROCESS_BROKER_PROTOCOL_VERSION,
    RUNTIME_BROKER_OPERATIONS,
    ProcessBrokerProtocolError,
    authenticated_envelope,
    expected_verifier_receipt_binding,
    receive_frame,
    send_frame,
    validate_runtime_request_payload,
    validate_runtime_result,
    validated_policy_screenshot_root,
    verify_authenticated_envelope,
)
from .common import canonical_json_bytes
from web_agent.runtime.contracts import OpaqueTerminalSignal, VerifierReceiptBinding


def _detached_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical deep snapshot for state retained across calls."""

    detached = json.loads(canonical_json_bytes(value))
    if not isinstance(detached, dict):  # pragma: no cover - Mapping is a JSON object
        raise ProcessBrokerProtocolError("process-broker state is not an object")
    return detached


class ProcessIsolatedRuntimeClient:
    """Operation-limited authenticated capability with no evaluator operation."""

    __slots__ = (
        "__closed",
        "__endpoint",
        "__episode_task",
        "__failed",
        "__key",
        "__last_action",
        "__last_observation",
        "__pending_action_id",
        "__pending_is_recovery",
        "__policy_screenshot_root",
        "__reset_state_receipt",
        "__sequence",
        "__session_id",
        "__terminal_required",
        "__terminated",
        "__timeout",
    )

    def __init__(
        self,
        *,
        endpoint: str | Path,
        authentication_key: bytes,
        session_id: str,
        policy_screenshot_root: str | Path | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not session_id or len(authentication_key) < 32:
            raise ValueError("runtime broker client requires session/key identity")
        self.__endpoint = str(endpoint)
        self.__key = bytes(authentication_key)
        self.__session_id = session_id
        self.__sequence = 0
        self.__timeout = float(timeout_seconds)
        self.__episode_task: tuple[str, str] | None = None
        self.__pending_action_id: str | None = None
        self.__pending_is_recovery = False
        self.__last_action: dict[str, Any] | None = None
        self.__last_observation: dict[str, Any] | None = None
        self.__reset_state_receipt: dict[str, Any] | None = None
        self.__policy_screenshot_root = validated_policy_screenshot_root(
            policy_screenshot_root
        )
        self.__terminal_required = False
        self.__terminated = False
        self.__closed = False
        self.__failed = False

    def __require_open_identity(
        self, episode_id: str, task_id: str, *, allow_failed: bool = False
    ) -> None:
        if self.__closed:
            raise ProcessBrokerProtocolError("process-broker runtime session is closed")
        if self.__failed and not allow_failed:
            raise ProcessBrokerProtocolError(
                "process-broker runtime session failed closed"
            )
        if self.__episode_task is None:
            raise ProcessBrokerProtocolError(
                "process-broker runtime session has not published reset observation"
            )
        if self.__episode_task != (episode_id, task_id):
            raise ProcessBrokerProtocolError(
                "process-broker runtime episode/task identity differs"
            )

    def __request(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        validated_payload = validate_runtime_request_payload(operation, payload)
        sequence = self.__sequence
        self.__sequence += 1
        try:
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
                raise ProcessBrokerProtocolError(
                    "process-broker response binding differs"
                )
            status = response.get("status")
            expected_fields = set(expected) | {"status", "result"}
            if status == "REJECTED":
                expected_fields.add("error_code")
            if set(response) != expected_fields:
                raise ProcessBrokerProtocolError(
                    "process-broker response fields differ"
                )
            if status != "PASS":
                if status != "REJECTED" or response.get("error_code") != (
                    "REGISTERED_REQUEST_REJECTED"
                ):
                    raise ProcessBrokerProtocolError(
                        "process-broker rejection envelope differs"
                    )
                raise ProcessBrokerProtocolError(
                    f"process-broker request rejected: {response.get('error_code')}"
                )
            return validate_runtime_result(
                operation,
                response.get("result"),
                request_payload=validated_payload,
                policy_screenshot_root=self.__policy_screenshot_root,
            )
        except BaseException:
            # Once a sequence has been allocated, a transport/authentication
            # failure leaves server-side consumption indeterminate. No later
            # runtime decision may use this client; cleanup can still be tried.
            self.__failed = True
            raise

    def reset(
        self,
        *,
        episode_id: str,
        task_id: str,
        benchmark_version: str,
        start_state_id: str,
        reset_stage_seed: int,
    ) -> dict[str, Any]:
        if self.__closed or self.__episode_task is not None:
            raise ProcessBrokerProtocolError(
                "process-broker reset must be the first operation"
            )
        if self.__failed:
            raise ProcessBrokerProtocolError(
                "process-broker runtime session failed closed"
            )
        result = self.__request(
            "runtime_reset",
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "benchmark_version": benchmark_version,
                "start_state_id": start_state_id,
                "reset_stage_seed": reset_stage_seed,
            },
        )
        self.__episode_task = (episode_id, task_id)
        self.__last_observation = _detached_mapping(result["observation"])
        self.__last_action = None
        self.__reset_state_receipt = _detached_mapping(
            result["reset_state_receipt"]
        )
        self.__terminal_required = True
        return _detached_mapping(self.__last_observation)

    def reset_state_receipt(self) -> dict[str, Any] | None:
        if self.__reset_state_receipt is None:
            return None
        return _detached_mapping(self.__reset_state_receipt)

    def observe(
        self,
        *,
        episode_id: str,
        task_id: str,
        stage: str,
        prior_action_id: str | None,
    ) -> dict[str, Any]:
        if self.__closed:
            raise ProcessBrokerProtocolError("process-broker runtime session is closed")
        self.__require_open_identity(episode_id, task_id)
        if self.__pending_action_id is None:
            raise ProcessBrokerProtocolError(
                "process-broker observation has no pending executed action"
            )
        expected_stage = (
            "post_recovery" if self.__pending_is_recovery else "post_action"
        )
        if stage != expected_stage or prior_action_id != self.__pending_action_id:
            raise ProcessBrokerProtocolError(
                "process-broker observation differs from pending action"
            )
        result = self.__request(
            "runtime_observe",
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "stage": stage,
                "prior_action_id": prior_action_id,
            },
        )
        self.__pending_action_id = None
        self.__pending_is_recovery = False
        self.__last_observation = _detached_mapping(result["observation"])
        self.__terminal_required = True
        return _detached_mapping(self.__last_observation)

    def execute(
        self, *, episode_id: str, task_id: str, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.__require_open_identity(episode_id, task_id)
        if self.__terminal_required:
            raise ProcessBrokerProtocolError(
                "process-broker execution requires the pending terminal receipt"
            )
        if self.__terminated:
            raise ProcessBrokerProtocolError(
                "process-broker opaque terminal signal already ended the episode"
            )
        if self.__pending_action_id is not None:
            raise ProcessBrokerProtocolError(
                "process-broker execution awaits its post observation"
            )
        prepared_payload = validate_runtime_request_payload(
            "runtime_execute",
            {"episode_id": episode_id, "task_id": task_id, "action": action},
        )
        action_snapshot = prepared_payload["action"]
        result = self.__request(
            "runtime_execute", prepared_payload
        )
        action_id = action_snapshot.get("action_id")
        if type(action_id) is not str:  # request validator should have rejected first
            raise ProcessBrokerProtocolError("process-broker action ID is invalid")
        self.__pending_action_id = action_id
        self.__pending_is_recovery = (
            action_snapshot.get("recovery_attempt_id") is not None
        )
        self.__last_action = _detached_mapping(action_snapshot)
        return _detached_mapping(result["execution"])

    def terminal_signal(
        self,
        *,
        episode_id: str,
        task_id: str,
        receipt_binding: VerifierReceiptBinding | Mapping[str, Any],
    ) -> OpaqueTerminalSignal:
        self.__require_open_identity(episode_id, task_id)
        if self.__pending_action_id is not None:
            raise ProcessBrokerProtocolError(
                "process-broker terminal check requires post observation"
            )
        if not self.__terminal_required or self.__last_observation is None:
            raise ProcessBrokerProtocolError(
                "process-broker terminal receipt is duplicate or out of order"
            )
        supplied = (
            receipt_binding.to_dict()
            if type(receipt_binding) is VerifierReceiptBinding
            else dict(receipt_binding)
        )
        expected = expected_verifier_receipt_binding(
            observation=self.__last_observation,
            action=self.__last_action,
        )
        if supplied != expected:
            raise ProcessBrokerProtocolError(
                "process-broker terminal receipt binding differs from causal state"
            )
        result = self.__request(
            "runtime_terminal",
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "receipt_binding": supplied,
            },
        )
        signal = OpaqueTerminalSignal.from_dict(result["opaque_terminal_signal"])
        self.__terminal_required = False
        self.__terminated = signal.terminate
        return signal

    def close(self, *, episode_id: str, task_id: str) -> dict[str, Any]:
        self.__require_open_identity(episode_id, task_id, allow_failed=True)
        if not self.__failed:
            if self.__pending_action_id is not None:
                raise ProcessBrokerProtocolError(
                    "process-broker close awaits its post observation"
                )
            if self.__terminal_required:
                raise ProcessBrokerProtocolError(
                    "process-broker close requires the pending terminal receipt"
                )
        result = self.__request(
            "runtime_close", {"episode_id": episode_id, "task_id": task_id}
        )
        if result != {"closed": True}:
            raise ProcessBrokerProtocolError("process-broker close was not acknowledged")
        self.__closed = True
        return result
