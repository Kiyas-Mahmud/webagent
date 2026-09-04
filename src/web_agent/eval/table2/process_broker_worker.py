"""Sealed evaluator/browser-owner worker for authenticated process broker IPC.

Only this process imports the configured backend. The runtime channel exposes
five exact operation envelopes and rejects registered sensitive key aliases
recursively. Its eight reset/action/observation/execution/verifier request and
result paths satisfy registered schemas and one-session causal state. Those
schemas carry no external value-provenance attestation, so this source does not
claim that neutral-key values are semantically oracle-free.
"""

from __future__ import annotations

import base64
import hashlib
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
    expected_verifier_receipt_binding,
    receive_frame,
    send_frame,
    validate_runtime_request_payload,
    validate_runtime_result,
    validated_policy_screenshot_root,
    verify_authenticated_envelope,
)


CONTROL_OPERATION = "sealed_control_shutdown"


def _load_factory(
    entrypoint: str,
    *,
    source_path: Path,
    source_sha256: str,
    backend_config: Mapping[str, Any],
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
    backend = factory(dict(backend_config))
    for operation in RUNTIME_BROKER_OPERATIONS:
        if not callable(getattr(backend, operation, None)):
            raise ProcessBrokerProtocolError(
                f"sealed backend lacks registered operation {operation}"
            )
    return backend


def _detach_backend_result(value: object) -> dict[str, Any]:
    """Materialize backend-controlled objects before the final source scan."""

    try:
        encoded = canonical_json_bytes(value)
        detached = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProcessBrokerProtocolError(
            "sealed backend result is not detached canonical JSON"
        ) from exc
    if not isinstance(detached, dict):
        raise ProcessBrokerProtocolError(
            "sealed backend result must be a JSON object"
        )
    return detached


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
        "web_agent.benchmarks": "src/web_agent/benchmarks/__init__.py",
        "web_agent.benchmarks.base": "src/web_agent/benchmarks/base.py",
        "web_agent.eval": "src/web_agent/eval/__init__.py",
        "web_agent.eval.table2": "src/web_agent/eval/table2/__init__.py",
        "web_agent.eval.table2.common": "src/web_agent/eval/table2/common.py",
        "web_agent.eval.table2.process_broker_protocol": (
            "src/web_agent/eval/table2/process_broker_protocol.py"
        ),
        "web_agent.eval.table2.process_broker_worker": (
            "src/web_agent/eval/table2/process_broker_worker.py"
        ),
        "web_agent.runtime": "src/web_agent/runtime/__init__.py",
        "web_agent.runtime.contracts": "src/web_agent/runtime/contracts.py",
        "web_agent.runtime.deadline": "src/web_agent/runtime/deadline.py",
        "web_agent.runtime.state_reset": "src/web_agent/runtime/state_reset.py",
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
    # The worker starts in an isolated interpreter.  Every repository-local
    # module executed before readiness, including backend imports, must be in
    # the authenticated launch closure.  This catches future eager package
    # imports instead of silently expanding the trusted computing base.
    for module_name, module in tuple(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        unresolved = Path(str(raw_path))
        if not unresolved.is_absolute():
            unresolved = Path.cwd() / unresolved
        loaded = unresolved.resolve()
        unresolved_inside_repository = False
        try:
            unresolved.absolute().relative_to(repository_root)
            unresolved_inside_repository = True
        except ValueError:
            pass
        try:
            relative = loaded.relative_to(repository_root).as_posix()
        except ValueError:
            if unresolved_inside_repository:
                raise ProcessBrokerProtocolError(
                    "loaded repository module escapes broker source closure: "
                    f"{module_name}"
                )
            continue
        if loaded.suffix != ".py" or expected.get(relative) != sha256_file(loaded):
            raise ProcessBrokerProtocolError(
                f"loaded repository module is outside broker source closure: {module_name}"
            )


def serve(config: Mapping[str, Any]) -> int:
    endpoint = Path(str(config["endpoint"]))
    runtime_key = base64.b64decode(str(config["runtime_key_b64"]), validate=True)
    control_key = base64.b64decode(str(config["control_key_b64"]), validate=True)
    session_id = str(config["session_id"])
    allowed_runtime_pid = int(config["allowed_runtime_pid"])
    allowed_uid = int(config["allowed_uid"])
    policy_screenshot_root = validated_policy_screenshot_root(
        config.get("policy_screenshot_root")
    )
    backend_config = config.get("sealed_backend_config")
    if not isinstance(backend_config, Mapping):
        raise ProcessBrokerProtocolError("sealed backend config is absent")
    backend_config_sha256 = hashlib.sha256(
        canonical_json_bytes(backend_config)
    ).hexdigest()
    if backend_config_sha256 != config.get("sealed_backend_config_sha256"):
        raise ProcessBrokerProtocolError("sealed backend config identity differs")
    screenshot_root_identity = (
        hashlib.sha256(str(policy_screenshot_root).encode("utf-8")).hexdigest()
        if policy_screenshot_root is not None
        else None
    )
    if screenshot_root_identity != config.get(
        "policy_screenshot_root_identity_sha256"
    ):
        raise ProcessBrokerProtocolError("policy screenshot root identity differs")
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
        backend_config=backend_config,
    )
    _verify_imported_broker_sources(config)
    sequences = {"runtime": 0, "control": 0}
    seen_nonces: set[str] = set()
    bound_episode_task: tuple[str, str] | None = None
    pending_action: tuple[str, bool] | None = None
    last_action: dict[str, Any] | None = None
    last_observation: dict[str, Any] | None = None
    terminal_required = False
    episode_terminated = False
    runtime_closed = False
    runtime_failed = False
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
            "sealed_backend_config_sha256": backend_config_sha256,
            "policy_screenshot_root_identity_sha256": screenshot_root_identity,
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
                        requested_identity = (
                            str(validated_payload["episode_id"]),
                            str(validated_payload["task_id"]),
                        )
                        if runtime_closed:
                            raise ProcessBrokerProtocolError(
                                "runtime broker session is closed"
                            )
                        if runtime_failed and operation != "runtime_close":
                            raise ProcessBrokerProtocolError(
                                "runtime broker session failed closed"
                            )
                        if bound_episode_task is None:
                            if (
                                operation != "runtime_reset"
                            ):
                                raise ProcessBrokerProtocolError(
                                    "first runtime operation must be registered reset"
                                )
                        elif requested_identity != bound_episode_task:
                            raise ProcessBrokerProtocolError(
                                "runtime broker episode/task identity differs"
                            )
                        if operation == "runtime_reset" and bound_episode_task is not None:
                            raise ProcessBrokerProtocolError(
                                "runtime reset may occur exactly once"
                            )
                        if operation == "runtime_observe":
                            if pending_action is None:
                                raise ProcessBrokerProtocolError(
                                    "runtime observation has no pending action"
                                )
                            pending_action_id, pending_is_recovery = pending_action
                            expected_stage = (
                                "post_recovery"
                                if pending_is_recovery
                                else "post_action"
                            )
                            if (
                                validated_payload.get("stage") != expected_stage
                                or validated_payload.get("prior_action_id")
                                != pending_action_id
                            ):
                                raise ProcessBrokerProtocolError(
                                    "runtime observation differs from pending action"
                                )
                        elif operation == "runtime_execute":
                            if pending_action is not None:
                                raise ProcessBrokerProtocolError(
                                    "runtime execution awaits its post observation"
                                )
                            if terminal_required:
                                raise ProcessBrokerProtocolError(
                                    "runtime execution requires pending terminal receipt"
                                )
                            if episode_terminated:
                                raise ProcessBrokerProtocolError(
                                    "runtime episode already terminated"
                                )
                        elif operation == "runtime_terminal":
                            if pending_action is not None:
                                raise ProcessBrokerProtocolError(
                                    "runtime terminal check requires post observation"
                                )
                            if not terminal_required or last_observation is None:
                                raise ProcessBrokerProtocolError(
                                    "runtime terminal receipt is duplicate or out of order"
                                )
                            expected_binding = expected_verifier_receipt_binding(
                                observation=last_observation,
                                action=last_action,
                            )
                            if canonical_json_bytes(
                                validated_payload.get("receipt_binding")
                            ) != canonical_json_bytes(expected_binding):
                                raise ProcessBrokerProtocolError(
                                    "runtime terminal receipt binding differs from causal state"
                                )
                        elif operation == "runtime_close" and not runtime_failed:
                            if pending_action is not None:
                                raise ProcessBrokerProtocolError(
                                    "runtime close awaits its post observation"
                                )
                            if terminal_required:
                                raise ProcessBrokerProtocolError(
                                    "runtime close requires pending terminal receipt"
                                )
                        # Consume replay state only after authentication and the
                        # complete request and causal-state schemas have passed.
                        # Backend/result failures consume the synchronized
                        # request and poison every operation except cleanup.
                        sequences[role] += 1
                        seen_nonces.add(nonce)
                        try:
                            backend_result = _detach_backend_result(
                                getattr(backend, operation)(validated_payload)
                            )
                            # A backend may import code lazily from an
                            # operation. Recheck before any result crosses the
                            # boundary so readiness cannot bypass the source
                            # closure.
                            _verify_imported_broker_sources(config)
                            result = validate_runtime_result(
                                operation,
                                backend_result,
                                request_payload=validated_payload,
                                policy_screenshot_root=policy_screenshot_root,
                            )
                            _verify_imported_broker_sources(config)
                        except Exception:
                            runtime_failed = True
                            raise
                        if operation == "runtime_reset":
                            bound_episode_task = requested_identity
                            last_observation = dict(result["observation"])
                            last_action = None
                            terminal_required = True
                        elif operation == "runtime_observe":
                            pending_action = None
                            last_observation = dict(result["observation"])
                            terminal_required = True
                        elif operation == "runtime_execute":
                            action = validated_payload["action"]
                            pending_action = (
                                str(action["action_id"]),
                                action.get("recovery_attempt_id") is not None,
                            )
                            last_action = dict(action)
                        elif operation == "runtime_terminal":
                            terminal_required = False
                            signal = result["opaque_terminal_signal"]
                            episode_terminated = signal["terminate"] is True
                        elif operation == "runtime_close":
                            if result != {"closed": True}:
                                runtime_failed = True
                                raise ProcessBrokerProtocolError(
                                    "runtime close was not acknowledged"
                                )
                            runtime_closed = True
                        response_role = "sealed_runtime_response"
                    else:
                        # Control shutdown is lifecycle cleanup only. It is
                        # deliberately available from an incomplete/failed
                        # runtime phase and does not certify causal episode
                        # completion; only runtime_close carries that ordering
                        # contract.
                        if operation != CONTROL_OPERATION or payload != {}:
                            raise ProcessBrokerProtocolError(
                                "control requested an unregistered operation"
                            )
                        sequences[role] += 1
                        seen_nonces.add(nonce)
                        shutdown = getattr(backend, "shutdown", None)
                        if callable(shutdown):
                            shutdown()
                        _verify_imported_broker_sources(config)
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
                            # A backend-controlled exception class name would be
                            # another neutral scalar channel. Keep rejection
                            # disclosure closed and deterministic.
                            "error_code": "REGISTERED_REQUEST_REJECTED",
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
