"""Sealed evaluator/browser-owner worker for authenticated process broker IPC.

Only this process imports the configured backend. The runtime channel exposes
four exact outer operation envelopes and rejects registered sensitive key
aliases recursively. Its action/observation/execution inner mappings are not
operation-specific and carry no value-provenance attestation, so this source
does not claim that neutral-key values are semantically oracle-free.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
from typing import Any, Mapping

from .common import canonical_json_bytes, sha256_file
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


CONTROL_OPERATION = "sealed_control_shutdown"


def _load_factory(
    entrypoint: str, *, source_path: Path, source_sha256: str
) -> Any:
    module_name, separator, attribute_name = entrypoint.partition(":")
    if separator != ":" or not module_name or not attribute_name or "." in attribute_name:
        raise ProcessBrokerProtocolError("sealed backend entrypoint is malformed")
    module = importlib.import_module(module_name)
    loaded_source = Path(str(getattr(module, "__file__", ""))).resolve()
    if (
        loaded_source != source_path.resolve()
        or loaded_source.suffix != ".py"
        or sha256_file(loaded_source) != source_sha256
    ):
        raise ProcessBrokerProtocolError(
            "sealed backend loaded module identity differs"
        )
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ProcessBrokerProtocolError("sealed backend factory is not callable")
    backend = factory()
    for operation in RUNTIME_BROKER_OPERATIONS:
        if not callable(getattr(backend, operation, None)):
            raise ProcessBrokerProtocolError(
                f"sealed backend lacks registered operation {operation}"
            )
    return backend


def _peer_identity(connection: socket.socket) -> tuple[int, int, int] | None:
    if not hasattr(socket, "SO_PEERCRED"):
        return None
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    return struct.unpack("3i", raw)


def _verify_imported_broker_sources(config: Mapping[str, Any]) -> None:
    repository_root = Path(str(config["repository_root"])).resolve()
    rows = config.get("source_files")
    if not isinstance(rows, list):
        raise ProcessBrokerProtocolError("broker source closure is absent")
    expected = {
        str(row.get("relative_path")): str(row.get("sha256"))
        for row in rows
        if isinstance(row, Mapping)
    }
    modules = {
        "web_agent": "src/web_agent/__init__.py",
        "web_agent.eval": "src/web_agent/eval/__init__.py",
        "web_agent.eval.table2": "src/web_agent/eval/table2/__init__.py",
        "web_agent.eval.table2.common": "src/web_agent/eval/table2/common.py",
        "web_agent.eval.table2.process_broker_protocol": (
            "src/web_agent/eval/table2/process_broker_protocol.py"
        ),
        "web_agent.eval.table2.process_broker_worker": (
            "src/web_agent/eval/table2/process_broker_worker.py"
        ),
    }
    for module_name, relative in modules.items():
        module = sys.modules.get(module_name)
        loaded = Path(str(getattr(module, "__file__", ""))).resolve()
        registered = repository_root / relative
        if (
            loaded != registered
            or expected.get(relative) != sha256_file(loaded)
            or loaded.suffix != ".py"
        ):
            raise ProcessBrokerProtocolError(
                f"imported broker module identity differs: {module_name}"
            )


def serve(config: Mapping[str, Any]) -> int:
    endpoint = Path(str(config["endpoint"]))
    runtime_key = base64.b64decode(str(config["runtime_key_b64"]), validate=True)
    control_key = base64.b64decode(str(config["control_key_b64"]), validate=True)
    session_id = str(config["session_id"])
    allowed_runtime_pid = int(config["allowed_runtime_pid"])
    allowed_uid = int(config["allowed_uid"])
    backend_source = Path(str(config["backend_source_path"]))
    if (
        backend_source.is_symlink()
        or not backend_source.is_file()
        or sha256_file(backend_source) != config.get("backend_source_sha256")
    ):
        raise ProcessBrokerProtocolError("sealed backend source identity differs")
    _verify_imported_broker_sources(config)
    backend = _load_factory(
        str(config["backend_entrypoint"]),
        source_path=backend_source,
        source_sha256=str(config["backend_source_sha256"]),
    )
    sequences = {"runtime": 0, "control": 0}
    seen_nonces: set[str] = set()
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    if endpoint.exists():
        raise ProcessBrokerProtocolError("process-broker endpoint already exists")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(endpoint))
        os.chmod(endpoint, 0o600)
        listener.listen(8)
        readiness = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "sealed_readiness",
            "session_id": session_id,
            "startup_nonce": str(config["startup_nonce"]),
            "status": "READY",
            "evaluator_pid": os.getpid(),
            "endpoint_mode": oct(endpoint.stat().st_mode & 0o777),
        }
        print(
            canonical_json_bytes(
                authenticated_envelope(readiness, authentication_key=control_key)
            ).decode("utf-8"),
            flush=True,
        )
        running = True
        while running:
            connection, _ = listener.accept()
            with connection:
                role: object = None
                sequence: object = None
                nonce: object = None
                try:
                    peer = _peer_identity(connection)
                    envelope = receive_frame(connection)
                    unsigned = envelope.get("body")
                    role = unsigned.get("role") if isinstance(unsigned, Mapping) else None
                    if role not in {"runtime", "control"}:
                        raise ProcessBrokerProtocolError("process-broker role is forbidden")
                    key = runtime_key if role == "runtime" else control_key
                    body = verify_authenticated_envelope(
                        envelope, authentication_key=key
                    )
                    if set(body) != {
                        "protocol_version",
                        "role",
                        "session_id",
                        "sequence",
                        "nonce",
                        "operation",
                        "payload",
                    }:
                        raise ProcessBrokerProtocolError(
                            "process-broker request fields differ from schema"
                        )
                    if (
                        body.get("protocol_version") != PROCESS_BROKER_PROTOCOL_VERSION
                        or body.get("session_id") != session_id
                    ):
                        raise ProcessBrokerProtocolError("process-broker session differs")
                    if peer is not None:
                        peer_pid, peer_uid, _peer_gid = peer
                        if peer_uid != allowed_uid or peer_pid != allowed_runtime_pid:
                            raise ProcessBrokerProtocolError(
                                "process-broker peer process identity differs"
                            )
                    sequence = body.get("sequence")
                    nonce = body.get("nonce")
                    if (
                        type(sequence) is not int
                        or sequence != sequences[role]
                        or type(nonce) is not str
                        or len(nonce) != 64
                        or nonce in seen_nonces
                    ):
                        raise ProcessBrokerProtocolError(
                            "process-broker sequence/nonce is replayed or invalid"
                        )
                    operation = body.get("operation")
                    payload = body.get("payload")
                    if role == "runtime":
                        validated_payload = validate_runtime_request_payload(
                            operation, payload
                        )
                        # Consume replay state only after authentication and the
                        # complete request schema have passed. Backend/result
                        # failures still consume the synchronized request.
                        sequences[role] += 1
                        seen_nonces.add(nonce)
                        result = validate_runtime_result(
                            operation,
                            getattr(backend, operation)(validated_payload),
                        )
                        response_role = "sealed_runtime_response"
                    else:
                        if operation != CONTROL_OPERATION or payload != {}:
                            raise ProcessBrokerProtocolError(
                                "control requested an unregistered operation"
                            )
                        sequences[role] += 1
                        seen_nonces.add(nonce)
                        shutdown = getattr(backend, "shutdown", None)
                        if callable(shutdown):
                            shutdown()
                        result = {"shutdown": True}
                        response_role = "sealed_control_response"
                        running = False
                    response = {
                        "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
                        "role": response_role,
                        "session_id": session_id,
                        "sequence": sequence,
                        "nonce": nonce,
                        "status": "PASS",
                        "result": dict(result),
                    }
                    send_frame(
                        connection,
                        authenticated_envelope(response, authentication_key=key),
                    )
                except Exception as exc:
                    # Do not reflect exception text, backend state, or evaluator
                    # details across the runtime boundary.
                    try:
                        role = (
                            role if role in {"runtime", "control"} else "runtime"
                        )
                        key = runtime_key if role == "runtime" else control_key
                        response = {
                            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
                            "role": (
                                "sealed_runtime_response"
                                if role == "runtime"
                                else "sealed_control_response"
                            ),
                            "session_id": session_id,
                            "sequence": (
                                sequence if type(sequence) is int else -1
                            ),
                            "nonce": nonce if type(nonce) is str else "",
                            "status": "REJECTED",
                            "error_code": type(exc).__name__,
                            "result": {},
                        }
                        send_frame(
                            connection,
                            authenticated_envelope(response, authentication_key=key),
                        )
                    except Exception:
                        pass
    finally:
        listener.close()
        try:
            endpoint.unlink()
        except FileNotFoundError:
            pass
    return 0


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        raise ProcessBrokerProtocolError("process-broker worker lacks launch config")
    value = json.loads(line)
    if not isinstance(value, Mapping):
        raise ProcessBrokerProtocolError("process-broker launch config is not an object")
    return serve(value)


if __name__ == "__main__":
    raise SystemExit(main())
