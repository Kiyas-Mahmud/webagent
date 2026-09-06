"""Runtime-only client for the process-isolated page/evaluator owner.

This module intentionally imports neither the worker nor an evaluator module.
Its public API contains only reset, observation, execution, nondispatched
rejected-action registration, opaque-terminal, and session-close operations.
The server independently enforces that six-operation allow-list, exact outer
envelopes, and all twelve registered reset/action/
observation/execution/verifier/error request and result schemas. Schema conformance
is additionally bound to one episode/task and a reset/execute/observe/verifier
causal state. It does not attest where scalar runtime-visible values
originated; external value-provenance and deployment evidence remain
mandatory. Each request uses one absolute monotonic deadline across connect,
send, and all partial receives; a trickling peer cannot restart the timeout.
"""

from __future__ import annotations

import json
import math
import secrets
import socket
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

from web_agent.benchmarks.base import AdapterExecution

from .process_broker_protocol import (
    PROCESS_BROKER_INFRASTRUCTURE_INVALID_STATUS,
    PROCESS_BROKER_PROTOCOL_VERSION,
    RUNTIME_BROKER_OPERATIONS,
    ProcessBrokerTranscript,
    ProcessBrokerProtocolError,
    authenticated_envelope,
    expected_verifier_receipt_binding,
    receive_frame,
    set_socket_timeout_to_deadline,
    send_frame,
    validate_runtime_infrastructure_invalid,
    validate_runtime_request_payload,
    validate_runtime_result,
    validated_policy_screenshot_root,
    verify_authenticated_envelope,
)
from .common import canonical_json_bytes, sha256_file
from .execution_guard import InfrastructureInvalidError
from .process_broker_timeout import RUNTIME_TIMEOUT_OPERATIONS
from web_agent.runtime.contracts import OpaqueTerminalSignal, VerifierReceiptBinding


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())


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
        "__control_cleanup_only",
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
        "__timeouts",
        "__transcript",
    )

    def __init__(
        self,
        *,
        endpoint: str | Path,
        authentication_key: bytes,
        session_id: str,
        policy_screenshot_root: str | Path | None = None,
        timeout_seconds: float = 10.0,
        timeout_seconds_by_operation: Mapping[str, float] | None = None,
        transcript: ProcessBrokerTranscript | None = None,
    ) -> None:
        if not session_id or len(authentication_key) < 32:
            raise ValueError("runtime broker client requires session/key identity")
        self.__endpoint = str(endpoint)
        self.__key = bytes(authentication_key)
        self.__session_id = session_id
        if transcript is not None and type(transcript) is not ProcessBrokerTranscript:
            raise TypeError("runtime broker transcript capability has wrong type")
        self.__transcript = transcript or ProcessBrokerTranscript(
            session_id=session_id
        )
        self.__sequence = 0
        if timeout_seconds_by_operation is None:
            timeout_value = float(timeout_seconds)
            if not math.isfinite(timeout_value) or timeout_value <= 0.0:
                raise ValueError("runtime broker timeout must be positive and finite")
            self.__timeouts = {
                operation: timeout_value for operation in RUNTIME_TIMEOUT_OPERATIONS
            }
        else:
            if set(timeout_seconds_by_operation) != set(RUNTIME_TIMEOUT_OPERATIONS):
                raise ValueError("runtime broker timeout operation closure differs")
            resolved: dict[str, float] = {}
            for operation in RUNTIME_TIMEOUT_OPERATIONS:
                raw = timeout_seconds_by_operation[operation]
                if (
                    isinstance(raw, bool)
                    or not isinstance(raw, (int, float))
                    or not math.isfinite(float(raw))
                    or float(raw) <= 0.0
                ):
                    raise ValueError(
                        f"runtime broker {operation} timeout must be positive and finite"
                    )
                resolved[operation] = float(raw)
            self.__timeouts = resolved
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
        self.__control_cleanup_only = False
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
            # Rejected-action registration performs no browser work.  It uses
            # the already calibrated execute IPC bound instead of inventing an
            # outcome-dependent timing class after calibration is frozen.
            timeout_operation = (
                "runtime_execute"
                if operation == "runtime_register_rejected"
                else operation
            )
            deadline = monotonic() + self.__timeouts[timeout_operation]
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
            request_frame = authenticated_envelope(
                body, authentication_key=self.__key
            )
            with self.__transcript.exchange():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    set_socket_timeout_to_deadline(connection, deadline)
                    connection.connect(self.__endpoint)
                    send_frame(
                        connection,
                        request_frame,
                        deadline_monotonic=deadline,
                    )
                    self.__transcript.commit_authenticated_frame(
                        direction="request",
                        role="runtime",
                        sequence=sequence,
                        operation=operation,
                        frame=request_frame,
                        payload=validated_payload,
                    )
                    response_frame = receive_frame(
                        connection, deadline_monotonic=deadline
                    )
                    response = verify_authenticated_envelope(
                        response_frame,
                        authentication_key=self.__key,
                    )
                    # Authentication is the commitment boundary. Even a
                    # subsequently rejected response shape is retained so an
                    # ambiguous or adversarial exchange cannot disappear from
                    # the final cleanup transcript.
                    self.__transcript.commit_authenticated_frame(
                        direction="response",
                        role="runtime",
                        sequence=sequence,
                        operation=operation,
                        frame=response_frame,
                        payload=response.get("result"),
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
            if status == PROCESS_BROKER_INFRASTRUCTURE_INVALID_STATUS:
                infrastructure = validate_runtime_infrastructure_invalid(
                    response.get("result"),
                    request_operation=operation,
                    episode_id=str(validated_payload["episode_id"]),
                    request_payload=validated_payload,
                )
                self.__control_cleanup_only = True
                error = InfrastructureInvalidError(
                    reason_code=infrastructure["reason_code"],
                    adapter_id=infrastructure["adapter_id"],
                    adapter_version=infrastructure["adapter_version"],
                    operation=infrastructure["operation"],
                    adapter_evidence=infrastructure["adapter_evidence"],
                )
                adapter_execution = infrastructure.get("adapter_execution")
                if isinstance(adapter_execution, Mapping):
                    setattr(
                        error,
                        "adapter_execution",
                        AdapterExecution.from_dict(adapter_execution),
                    )
                raise error
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
        task_specification_sha256: str,
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
                "task_specification_sha256": task_specification_sha256,
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

    def register_rejected_action(
        self,
        *,
        episode_id: str,
        task_id: str,
        action: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> None:
        """Bind one executor-local rejection without dispatching browser execute."""

        self.__require_open_identity(episode_id, task_id)
        if self.__terminal_required:
            raise ProcessBrokerProtocolError(
                "process-broker rejection registration requires the pending "
                "terminal receipt"
            )
        if self.__terminated:
            raise ProcessBrokerProtocolError(
                "process-broker opaque terminal signal already ended the episode"
            )
        if self.__pending_action_id is not None:
            raise ProcessBrokerProtocolError(
                "process-broker rejection registration awaits post observation"
            )
        prepared_payload = validate_runtime_request_payload(
            "runtime_register_rejected",
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "action": action,
                "execution": execution,
            },
        )
        action_snapshot = prepared_payload["action"]
        result = self.__request(
            "runtime_register_rejected",
            prepared_payload,
        )
        if result != {"registered": True}:
            raise ProcessBrokerProtocolError(
                "process-broker rejected-action registration changed result"
            )
        action_id = action_snapshot.get("action_id")
        if type(action_id) is not str:  # validated above
            raise ProcessBrokerProtocolError("process-broker action ID is invalid")
        self.__pending_action_id = action_id
        self.__pending_is_recovery = (
            action_snapshot.get("recovery_attempt_id") is not None
        )
        self.__last_action = _detached_mapping(action_snapshot)
        return None

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
        if self.__control_cleanup_only:
            raise ProcessBrokerProtocolError(
                "process-broker infrastructure invalid requires control cleanup"
            )
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
