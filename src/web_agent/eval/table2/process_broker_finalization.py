"""Orchestration-only finalization for the process-isolated page broker.

The sealed evaluator/browser child owns the append-only verifier sink for the
whole episode. Runtime terminal calls return the *outer* sealed event/token;
post-decision finalization additionally returns only a content-addressed
identity for the completed child-owned callback-state guard. No raw-evidence
socket, seal key, sink, or verifier writer is retained by either client here.

This remains local architecture evidence. Finalization does not yet have an
independently measured and externally cross-bound timeout class, so the live
EVALUATION scope stays fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import secrets
import socket
from time import monotonic
from types import MappingProxyType
from typing import Any, Mapping

from web_agent.runtime.contracts import EpisodeSummary, OpaqueTerminalSignal

from .common import canonical_json_bytes, sha256_file
from .process_broker_protocol import (
    MAX_PROCESS_BROKER_MESSAGE_BYTES,
    PROCESS_BROKER_PROTOCOL_VERSION,
    ProcessBrokerTranscript,
    ProcessBrokerProtocolError,
    authenticated_envelope,
    receive_frame,
    set_socket_timeout_to_deadline,
    send_frame,
    verify_authenticated_envelope,
)


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

SEALED_FINALIZATION_OPERATION = "sealed_finalize_episode"
SEALED_FINALIZATION_SCHEMA_VERSION = "table2-process-broker-finalization-v4"
SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION = (
    "table2-process-broker-sealed-callback-state-guard-sidecar-identity-v1"
)
SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE = (
    "ProcessBrokerSealedCallbackStateGuardSidecarIdentity"
)
SEALED_CALLBACK_GUARD_RELATIVE_PATH = (
    "sealed_callback_state_guard.child.jsonl"
)
PROCESS_BROKER_FINALIZATION_RECEIPT_SCHEMA_VERSION = (
    "table2-process-broker-sealed-finalization-receipt-v1"
)


def _detached_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessBrokerProtocolError(f"{context} must be a JSON object")
    try:
        encoded = canonical_json_bytes(value)
        detached = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProcessBrokerProtocolError(
            f"{context} must be detached canonical JSON"
        ) from exc
    if len(encoded) > MAX_PROCESS_BROKER_MESSAGE_BYTES:
        raise ProcessBrokerProtocolError(f"{context} exceeds the IPC size limit")
    if not isinstance(detached, dict):  # pragma: no cover - Mapping encodes object
        raise ProcessBrokerProtocolError(f"{context} must be a JSON object")
    return detached


def validate_finalization_request_payload(value: object) -> dict[str, Any]:
    """Validate the exact post-decision request and typed episode summary."""

    payload = _detached_object(value, context="sealed finalization request")
    if set(payload) != {
        "episode_id",
        "task_id",
        "system_id",
        "repeat_id",
        "model_seed",
        "episode_summary",
    }:
        raise ProcessBrokerProtocolError(
            "sealed finalization request fields differ from schema"
        )
    for field in ("episode_id", "task_id", "system_id"):
        item = payload.get(field)
        if (
            type(item) is not str
            or not item
            or item != item.strip()
            or len(item) > 512
        ):
            raise ProcessBrokerProtocolError(
                f"sealed finalization {field} is invalid"
            )
    for field in ("repeat_id", "model_seed"):
        item = payload.get(field)
        if type(item) is not int or item < 0:
            raise ProcessBrokerProtocolError(
                f"sealed finalization {field} is invalid"
            )
    summary_value = payload.get("episode_summary")
    if not isinstance(summary_value, Mapping):
        raise ProcessBrokerProtocolError(
            "sealed finalization summary must be an object"
        )
    try:
        summary = EpisodeSummary.from_dict(summary_value)
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            "sealed finalization summary is invalid"
        ) from exc
    if type(summary) is not EpisodeSummary or canonical_json_bytes(
        summary.to_dict()
    ) != canonical_json_bytes(summary_value):
        raise ProcessBrokerProtocolError(
            "sealed finalization summary is not canonical"
        )
    if (
        summary.episode_id != payload["episode_id"]
        or summary.task_id != payload["task_id"]
        or summary.system_id.value != payload["system_id"]
        or summary.repeat_id != payload["repeat_id"]
        or summary.model_seed != payload["model_seed"]
    ):
        raise ProcessBrokerProtocolError(
            "sealed finalization summary identity differs"
        )
    payload["episode_summary"] = summary.to_dict()
    return payload


def validate_finalization_result(value: object) -> dict[str, Any]:
    """Accept one opaque signal plus its child-owned state-guard identity."""

    result = _detached_object(value, context="sealed finalization result")
    if set(result) != {
        "opaque_terminal_signal",
        "sealed_callback_guard_identity",
    } or not isinstance(
        result.get("opaque_terminal_signal"), Mapping
    ):
        raise ProcessBrokerProtocolError(
            "sealed finalization result fields differ from schema"
        )
    try:
        signal = OpaqueTerminalSignal.from_dict(result["opaque_terminal_signal"])
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            "sealed finalization acknowledgement is not opaque"
        ) from exc
    if type(signal) is not OpaqueTerminalSignal or signal.terminate is not True:
        raise ProcessBrokerProtocolError(
            "sealed finalization acknowledgement must terminate"
        )
    guard = validate_sealed_callback_guard_identity(
        result.get("sealed_callback_guard_identity")
    )
    return {
        "opaque_terminal_signal": signal.to_dict(),
        "sealed_callback_guard_identity": guard,
    }


def validate_sealed_callback_guard_identity(value: object) -> dict[str, Any]:
    """Validate the final child-owned sealed-callback guard sidecar identity."""

    identity = _detached_object(
        value, context="sealed callback state-guard identity"
    )
    if set(identity) != {
        "schema_version",
        "record_type",
        "relative_path",
        "record_count",
        "tail_sha256",
        "content_sha256",
    }:
        raise ProcessBrokerProtocolError(
            "sealed callback state-guard identity fields differ"
        )
    if (
        identity.get("schema_version")
        != SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION
        or identity.get("record_type")
        != SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE
        or identity.get("relative_path")
        != SEALED_CALLBACK_GUARD_RELATIVE_PATH
    ):
        raise ProcessBrokerProtocolError(
            "sealed callback state-guard identity differs"
        )
    count = identity.get("record_count")
    if type(count) is not int or count < 1:
        raise ProcessBrokerProtocolError(
            "sealed callback state-guard identity lacks a completed final guard"
        )
    for field in ("tail_sha256", "content_sha256"):
        digest = identity.get(field)
        if type(digest) is not str or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ProcessBrokerProtocolError(
                f"sealed callback state-guard {field} is not lowercase SHA-256"
            )
    return identity


@dataclass(frozen=True, slots=True)
class ProcessIsolatedFinalizationReceipt:
    """Immutable parent receipt for one authenticated finalization response."""

    schema_version: str
    record_type: str
    opaque_terminal_signal: Mapping[str, Any]
    sealed_callback_guard_identity: Mapping[str, Any]
    authenticated_response_payload_sha256: str
    transcript_root_sha256: str
    transcript_entry_count: int

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != PROCESS_BROKER_FINALIZATION_RECEIPT_SCHEMA_VERSION
            or self.record_type != "ProcessIsolatedFinalizationReceipt"
        ):
            raise ProcessBrokerProtocolError(
                "process finalization receipt identity differs"
            )
        try:
            signal = OpaqueTerminalSignal.from_dict(self.opaque_terminal_signal)
        except (TypeError, ValueError) as exc:
            raise ProcessBrokerProtocolError(
                "process finalization receipt signal is invalid"
            ) from exc
        if type(signal) is not OpaqueTerminalSignal or signal.terminate is not True:
            raise ProcessBrokerProtocolError(
                "process finalization receipt signal must terminate"
            )
        guard = validate_sealed_callback_guard_identity(
            self.sealed_callback_guard_identity
        )
        for field in (
            "authenticated_response_payload_sha256",
            "transcript_root_sha256",
        ):
            digest = getattr(self, field)
            if type(digest) is not str or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ProcessBrokerProtocolError(
                    f"process finalization receipt {field} is not SHA-256"
                )
        if type(self.transcript_entry_count) is not int or (
            self.transcript_entry_count < 2
            or self.transcript_entry_count % 2 != 0
        ):
            raise ProcessBrokerProtocolError(
                "process finalization receipt transcript count is invalid"
            )
        object.__setattr__(
            self,
            "opaque_terminal_signal",
            MappingProxyType(signal.to_dict()),
        )
        object.__setattr__(
            self,
            "sealed_callback_guard_identity",
            MappingProxyType(dict(guard)),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {field.name: getattr(self, field.name) for field in fields(self)}
        value["opaque_terminal_signal"] = dict(self.opaque_terminal_signal)
        value["sealed_callback_guard_identity"] = dict(
            self.sealed_callback_guard_identity
        )
        return value


class ProcessIsolatedFinalizationClient:
    """Single-use orchestration capability with no sealed-output capability."""

    __slots__ = (
        "__endpoint",
        "__failed",
        "__finalized",
        "__key",
        "__receipt",
        "__sequence",
        "__session_id",
        "__timeout_seconds",
        "__transcript",
    )

    def __init__(
        self,
        *,
        endpoint: str | Path,
        authentication_key: bytes,
        session_id: str,
        timeout_seconds: float,
        transcript: ProcessBrokerTranscript | None = None,
    ) -> None:
        timeout = float(timeout_seconds)
        if (
            len(authentication_key) < 32
            or not session_id
            or not math.isfinite(timeout)
            or timeout <= 0.0
        ):
            raise ValueError(
                "process finalization client requires key, session and timeout"
            )
        self.__endpoint = str(endpoint)
        self.__key = bytes(authentication_key)
        self.__session_id = session_id
        if transcript is not None and type(transcript) is not ProcessBrokerTranscript:
            raise TypeError("process finalization transcript capability has wrong type")
        self.__transcript = transcript or ProcessBrokerTranscript(
            session_id=session_id
        )
        self.__timeout_seconds = timeout
        self.__sequence = 0
        self.__finalized = False
        self.__failed = False
        self.__receipt: ProcessIsolatedFinalizationReceipt | None = None

    @property
    def finalization_receipt(self) -> ProcessIsolatedFinalizationReceipt:
        if self.__receipt is None:
            raise ProcessBrokerProtocolError(
                "process finalization receipt is unavailable before finalization"
            )
        return self.__receipt

    @property
    def sealed_callback_guard_identity(self) -> Mapping[str, Any]:
        return self.finalization_receipt.sealed_callback_guard_identity

    def finalize_episode(
        self,
        *,
        episode_summary: EpisodeSummary,
    ) -> OpaqueTerminalSignal:
        if self.__finalized or self.__failed:
            raise ProcessBrokerProtocolError(
                "process finalization capability is consumed or failed"
            )
        if type(episode_summary) is not EpisodeSummary:
            raise TypeError("process finalization requires exact EpisodeSummary")
        summary_payload = episode_summary.to_dict()
        payload = validate_finalization_request_payload(
            {
                "episode_id": episode_summary.episode_id,
                "task_id": episode_summary.task_id,
                "system_id": summary_payload.get("system_id"),
                "repeat_id": episode_summary.repeat_id,
                "model_seed": episode_summary.model_seed,
                "episode_summary": summary_payload,
            }
        )
        sequence = self.__sequence
        self.__sequence += 1
        deadline = monotonic() + self.__timeout_seconds
        nonce = secrets.token_hex(32)
        body = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "orchestration",
            "session_id": self.__session_id,
            "sequence": sequence,
            "nonce": nonce,
            "operation": SEALED_FINALIZATION_OPERATION,
            "payload": payload,
        }
        try:
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
                        role="orchestration",
                        sequence=sequence,
                        operation=SEALED_FINALIZATION_OPERATION,
                        frame=request_frame,
                        payload=payload,
                    )
                    response_frame = receive_frame(
                        connection, deadline_monotonic=deadline
                    )
                    response = verify_authenticated_envelope(
                        response_frame,
                        authentication_key=self.__key,
                    )
                    finalization_transcript = (
                        self.__transcript.commit_authenticated_frame(
                            direction="response",
                            role="orchestration",
                            sequence=sequence,
                            operation=SEALED_FINALIZATION_OPERATION,
                            frame=response_frame,
                            payload=response.get("result"),
                        )
                    )
            expected = {
                "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
                "role": "sealed_orchestration_response",
                "session_id": self.__session_id,
                "sequence": sequence,
                "nonce": nonce,
            }
            if any(response.get(field) != item for field, item in expected.items()):
                raise ProcessBrokerProtocolError(
                    "process finalization response binding differs"
                )
            expected_fields = set(expected) | {"status", "result"}
            if response.get("status") == "REJECTED":
                expected_fields.add("error_code")
            if set(response) != expected_fields:
                raise ProcessBrokerProtocolError(
                    "process finalization response fields differ"
                )
            if response.get("status") != "PASS":
                if (
                    response.get("status") != "REJECTED"
                    or response.get("error_code")
                    != "REGISTERED_REQUEST_REJECTED"
                    or response.get("result") != {}
                ):
                    raise ProcessBrokerProtocolError(
                        "process finalization rejection envelope differs"
                    )
                raise ProcessBrokerProtocolError(
                    "process finalization request was rejected"
                )
            result = validate_finalization_result(response.get("result"))
            signal = OpaqueTerminalSignal.from_dict(
                result["opaque_terminal_signal"]
            )
            self.__receipt = ProcessIsolatedFinalizationReceipt(
                schema_version=(
                    PROCESS_BROKER_FINALIZATION_RECEIPT_SCHEMA_VERSION
                ),
                record_type="ProcessIsolatedFinalizationReceipt",
                opaque_terminal_signal=signal.to_dict(),
                sealed_callback_guard_identity=result[
                    "sealed_callback_guard_identity"
                ],
                authenticated_response_payload_sha256=(
                    hashlib.sha256(canonical_json_bytes(result)).hexdigest()
                ),
                transcript_root_sha256=str(
                    finalization_transcript["root_sha256"]
                ),
                transcript_entry_count=int(
                    finalization_transcript["entry_count"]
                ),
            )
            self.__finalized = True
            return signal
        except BaseException:
            self.__failed = True
            raise


@dataclass(frozen=True, slots=True)
class ProcessIsolatedEpisodeFinalizationBinding:
    """Final-evidence binding retained only by campaign orchestration."""

    client: ProcessIsolatedFinalizationClient
    episode_runtime_dir: Path

    def __post_init__(self) -> None:
        if type(self.client) is not ProcessIsolatedFinalizationClient:
            raise TypeError("process finalization binding requires exact client")
        root = Path(self.episode_runtime_dir)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError(
                "process finalization binding requires a canonical runtime directory"
            )
        object.__setattr__(self, "episode_runtime_dir", root.resolve(strict=True))

    def finalize_episode_evidence(
        self,
        summary: EpisodeSummary,
        runtime_dir: Path,
    ) -> OpaqueTerminalSignal:
        if Path(runtime_dir).resolve(strict=True) != self.episode_runtime_dir:
            raise ProcessBrokerProtocolError(
                "process finalization runtime directory differs"
            )
        return self.client.finalize_episode(episode_summary=summary)

    @property
    def finalization_receipt(self) -> ProcessIsolatedFinalizationReceipt:
        return self.client.finalization_receipt


__all__ = [
    "PROCESS_BROKER_FINALIZATION_RECEIPT_SCHEMA_VERSION",
    "ProcessIsolatedEpisodeFinalizationBinding",
    "ProcessIsolatedFinalizationClient",
    "ProcessIsolatedFinalizationReceipt",
    "SEALED_FINALIZATION_OPERATION",
    "SEALED_FINALIZATION_SCHEMA_VERSION",
    "SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE",
    "SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION",
    "SEALED_CALLBACK_GUARD_RELATIVE_PATH",
    "validate_finalization_request_payload",
    "validate_finalization_result",
    "validate_sealed_callback_guard_identity",
]
