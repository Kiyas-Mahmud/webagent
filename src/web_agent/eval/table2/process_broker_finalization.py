"""Orchestration-only finalization for the process-isolated page broker.

The sealed evaluator/browser child owns the append-only verifier sink for the
whole episode. Runtime terminal calls and post-decision finalization therefore
return the *outer* sealed event/token directly. No raw-evidence socket, seal
key, sink, or verifier writer is retained by either client in this module.

This remains local architecture evidence. Finalization does not yet have an
independently measured and externally cross-bound timeout class, so the live
EVALUATION scope stays fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import secrets
import socket
from time import monotonic
from typing import Any, Mapping

from web_agent.runtime.contracts import EpisodeSummary, OpaqueTerminalSignal

from .common import canonical_json_bytes, sha256_file
from .process_broker_protocol import (
    MAX_PROCESS_BROKER_MESSAGE_BYTES,
    PROCESS_BROKER_PROTOCOL_VERSION,
    ProcessBrokerProtocolError,
    authenticated_envelope,
    receive_frame,
    set_socket_timeout_to_deadline,
    send_frame,
    verify_authenticated_envelope,
)


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

SEALED_FINALIZATION_OPERATION = "sealed_finalize_episode"
SEALED_FINALIZATION_SCHEMA_VERSION = "table2-process-broker-finalization-v3"


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
    """Accept only one outer opaque terminal acknowledgement from the child."""

    result = _detached_object(value, context="sealed finalization result")
    if set(result) != {"opaque_terminal_signal"} or not isinstance(
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
    return {"opaque_terminal_signal": signal.to_dict()}


class ProcessIsolatedFinalizationClient:
    """Single-use orchestration capability with no sealed-output capability."""

    __slots__ = (
        "__endpoint",
        "__failed",
        "__finalized",
        "__key",
        "__sequence",
        "__session_id",
        "__timeout_seconds",
    )

    def __init__(
        self,
        *,
        endpoint: str | Path,
        authentication_key: bytes,
        session_id: str,
        timeout_seconds: float,
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
        self.__timeout_seconds = timeout
        self.__sequence = 0
        self.__finalized = False
        self.__failed = False

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
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                set_socket_timeout_to_deadline(connection, deadline)
                connection.connect(self.__endpoint)
                send_frame(
                    connection,
                    authenticated_envelope(body, authentication_key=self.__key),
                    deadline_monotonic=deadline,
                )
                response = verify_authenticated_envelope(
                    receive_frame(connection, deadline_monotonic=deadline),
                    authentication_key=self.__key,
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


__all__ = [
    "ProcessIsolatedEpisodeFinalizationBinding",
    "ProcessIsolatedFinalizationClient",
    "SEALED_FINALIZATION_OPERATION",
    "SEALED_FINALIZATION_SCHEMA_VERSION",
    "validate_finalization_request_payload",
    "validate_finalization_result",
]
