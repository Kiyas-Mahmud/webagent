"""Sealed evaluator/browser-owner worker for authenticated process broker IPC.

Only this process imports the configured backend. The runtime channel exposes
six exact operation envelopes and rejects registered sensitive key aliases
recursively. Its registered reset/action/observation/execution/verifier/error request
and result paths satisfy registered schemas and one-session causal state. Those
schemas carry no external value-provenance attestation, so this source does not
claim that neutral-key values are semantically oracle-free.  A separate
orchestration key exposes one post-decision finalization operation. The child
owns the append-only sealed stream. Runtime receives only outer opaque signals;
orchestration additionally receives a content-addressed identity for the
completed child-owned sealed-callback state guard.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import stat
import struct
import sys
from time import monotonic
from typing import Any, Mapping

from . import sealed_verifier as _sealed_verifier_source  # noqa: F401
from .common import canonical_json_bytes, sha256_file
from .process_broker_protocol import (
    PROCESS_BROKER_INFRASTRUCTURE_INVALID_STATUS,
    PROCESS_BROKER_CHILD_ENVIRONMENT_ALLOWLIST_VERSION,
    PROCESS_BROKER_CHILD_ENVIRONMENT_FIXED_NAMES,
    PROCESS_BROKER_CHILD_ENVIRONMENT_INHERITED_ALLOWLIST,
    PROCESS_BROKER_PROTOCOL_VERSION,
    RUNTIME_BROKER_OPERATIONS,
    ProcessBrokerInfrastructureInvalidResponse,
    ProcessBrokerProtocolError,
    ProcessBrokerTranscript,
    authenticated_envelope,
    expected_verifier_receipt_binding,
    receive_frame,
    send_frame,
    validate_runtime_infrastructure_invalid,
    validate_runtime_request_payload,
    validate_runtime_result,
    validate_child_cleanup_result,
    validated_policy_screenshot_root,
    verify_authenticated_envelope,
)
from .process_broker_finalization import (
    SEALED_FINALIZATION_OPERATION,
    validate_finalization_request_payload,
    validate_finalization_result,
)
from .process_broker_timeout import (
    MEASURED_TIMEOUT_MODE,
    binding_sha256 as timeout_binding_sha256,
    validate_process_broker_timeout_binding,
    validate_process_broker_timeout_calibration,
)


CONTROL_OPERATION = "sealed_control_shutdown"


def _verify_child_environment(
    config: Mapping[str, Any], *, endpoint: Path
) -> tuple[str, ...]:
    """Require the exec environment to equal the registered non-secret closure."""

    if config.get("child_environment_allowlist_version") != (
        PROCESS_BROKER_CHILD_ENVIRONMENT_ALLOWLIST_VERSION
    ):
        raise ProcessBrokerProtocolError(
            "process-broker child environment allowlist version differs"
        )
    configured = config.get("child_environment_variable_names")
    if (
        type(configured) is not list
        or any(type(name) is not str for name in configured)
        or configured != sorted(set(configured))
    ):
        raise ProcessBrokerProtocolError(
            "process-broker child environment name closure is invalid"
        )
    registered = set(PROCESS_BROKER_CHILD_ENVIRONMENT_INHERITED_ALLOWLIST) | set(
        PROCESS_BROKER_CHILD_ENVIRONMENT_FIXED_NAMES
    )
    actual = tuple(sorted(os.environ))
    if set(configured) != set(actual) or not set(actual).issubset(registered):
        raise ProcessBrokerProtocolError(
            "process-broker child environment escaped its exact allowlist"
        )
    forbidden_tokens = (
        "API_KEY",
        "AUTH",
        "COOKIE",
        "CREDENTIAL",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    )
    if any(
        token in name.upper() for name in actual for token in forbidden_tokens
    ):
        raise ProcessBrokerProtocolError(
            "process-broker child environment contains a credential alias"
        )
    root = endpoint.parent.resolve(strict=True)
    fixed_paths = {
        "HOME": root / "home",
        "TMPDIR": root,
        "XDG_CACHE_HOME": root / "cache",
    }
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ProcessBrokerProtocolError(
            "process-broker child user-site isolation differs"
        )
    for name, expected in fixed_paths.items():
        supplied = os.environ.get(name)
        path = Path(supplied) if type(supplied) is str else None
        if (
            path is None
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_dir()
            or path.resolve(strict=True) != expected.resolve(strict=True)
        ):
            raise ProcessBrokerProtocolError(
                f"process-broker child fixed environment path differs: {name}"
            )
    return actual


def _load_factory(
    entrypoint: str,
    *,
    source_path: Path,
    source_sha256: str,
    backend_config: Mapping[str, Any],
    require_sealed_finalization: bool,
) -> Any:
    module_name, separator, attribute_name = entrypoint.partition(":")
    if separator != ":" or not module_name or not attribute_name or "." in attribute_name:
        raise ProcessBrokerProtocolError("sealed backend entrypoint is malformed")
    module = importlib.import_module(module_name)
    loaded_source = Path(str(getattr(module, "__file__", ""))).resolve()
    if (
        loaded_source != source_path.resolve()
        or loaded_source.suffix != ".py"
        or loaded_source.stat().st_nlink != 1
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
    if not callable(getattr(backend, "shutdown", None)):
        raise ProcessBrokerProtocolError(
            "sealed backend lacks registered child cleanup operation"
        )
    if require_sealed_finalization and not (
        callable(getattr(backend, SEALED_FINALIZATION_OPERATION, None))
        and getattr(backend, "sealed_finalization_available", False) is True
    ):
        raise ProcessBrokerProtocolError(
            "sealed backend lacks required orchestration finalizer"
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
        "web_agent.eval.table2.process_broker_finalization": (
            "src/web_agent/eval/table2/process_broker_finalization.py"
        ),
        "web_agent.eval.table2.process_broker_timeout": (
            "src/web_agent/eval/table2/process_broker_timeout.py"
        ),
        "web_agent.eval.table2.sealed_verifier": (
            "src/web_agent/eval/table2/sealed_verifier.py"
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
            or loaded.stat().st_nlink != 1
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
        if (
            loaded.suffix != ".py"
            or loaded.stat().st_nlink != 1
            or expected.get(relative) != sha256_file(loaded)
        ):
            raise ProcessBrokerProtocolError(
                f"loaded repository module is outside broker source closure: {module_name}"
            )


def serve(
    config: Mapping[str, Any],
    *,
    readiness_connection: socket.socket,
) -> int:
    if not isinstance(readiness_connection, socket.socket):
        raise ProcessBrokerProtocolError("dedicated readiness channel is absent")
    endpoint = Path(str(config["endpoint"]))
    child_environment_variable_names = _verify_child_environment(
        config, endpoint=endpoint
    )
    runtime_key = base64.b64decode(str(config["runtime_key_b64"]), validate=True)
    orchestration_key = base64.b64decode(
        str(config["orchestration_key_b64"]), validate=True
    )
    control_key = base64.b64decode(str(config["control_key_b64"]), validate=True)
    if (
        min(
            len(runtime_key),
            len(orchestration_key),
            len(control_key),
        )
        < 32
        or len({runtime_key, orchestration_key, control_key}) != 3
    ):
        raise ProcessBrokerProtocolError(
            "process-broker role capabilities are absent or aliased"
        )
    sealed_finalization_required = config.get("sealed_finalization_required")
    if type(sealed_finalization_required) is not bool:
        raise ProcessBrokerProtocolError(
            "process-broker finalization requirement is not boolean"
        )
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
    timeout_evidence = config.get("ipc_timeout_calibration_evidence")
    timeout_binding = validate_process_broker_timeout_binding(
        config.get("ipc_timeout_binding"),
        calibration_evidence=timeout_evidence,
    )
    if (
        timeout_binding["mode"] == MEASURED_TIMEOUT_MODE
        and timeout_evidence is None
    ):
        raise ProcessBrokerProtocolError(
            "measured process-broker timeout evidence is absent"
        )
    if timeout_evidence is not None:
        validate_process_broker_timeout_calibration(timeout_evidence)
    measured_timeout_binding_sha256 = timeout_binding_sha256(timeout_binding)
    if measured_timeout_binding_sha256 != config.get("ipc_timeout_binding_sha256"):
        raise ProcessBrokerProtocolError("process-broker timeout binding differs")
    execution_scope = config.get("execution_scope")
    if execution_scope not in {
        "ENGINEERING_FIXTURE_ONLY",
        "MEASURED_CALIBRATION_REPLAY_ONLY",
        "PILOT_EVALUATION",
    }:
        raise ProcessBrokerProtocolError(
            "worker execution scope lacks local non-authorizing registration"
        )
    if (
        execution_scope == "ENGINEERING_FIXTURE_ONLY"
        and timeout_binding["mode"] == MEASURED_TIMEOUT_MODE
    ) or (
        execution_scope == "MEASURED_CALIBRATION_REPLAY_ONLY"
        and timeout_binding["mode"] != MEASURED_TIMEOUT_MODE
    ) or (
        execution_scope == "PILOT_EVALUATION"
        and timeout_binding["mode"] != MEASURED_TIMEOUT_MODE
    ):
        raise ProcessBrokerProtocolError("worker timeout mode differs from scope")
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
        or backend_source.stat().st_nlink != 1
        or sha256_file(backend_source) != config.get("backend_source_sha256")
    ):
        raise ProcessBrokerProtocolError("sealed backend source identity differs")
    _verify_imported_broker_sources(config)
    backend = _load_factory(
        str(config["backend_entrypoint"]),
        source_path=backend_source,
        source_sha256=str(config["backend_source_sha256"]),
        backend_config=backend_config,
        require_sealed_finalization=sealed_finalization_required,
    )
    _verify_imported_broker_sources(config)
    finalization_available = (
        callable(getattr(backend, SEALED_FINALIZATION_OPERATION, None))
        and getattr(backend, "sealed_finalization_available", False) is True
    )
    sequences = {"runtime": 0, "orchestration": 0, "control": 0}
    transcript = ProcessBrokerTranscript(session_id=session_id)
    seen_nonces: set[str] = set()
    bound_episode_task: tuple[str, str] | None = None
    pending_action: tuple[str, bool] | None = None
    last_action: dict[str, Any] | None = None
    last_observation: dict[str, Any] | None = None
    terminal_required = False
    episode_terminated = False
    runtime_closed = False
    runtime_failed = False
    runtime_control_cleanup_only = False
    finalization_complete = False
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
            "ipc_timeout_binding_sha256": measured_timeout_binding_sha256,
            "sealed_finalization_available": finalization_available,
            "sealed_finalization_required": sealed_finalization_required,
            "child_environment_allowlist_version": (
                PROCESS_BROKER_CHILD_ENVIRONMENT_ALLOWLIST_VERSION
            ),
            "child_environment_variable_names": list(
                child_environment_variable_names
            ),
        }
        startup_deadline = monotonic() + (
            timeout_binding["timeout_milliseconds"]["broker_startup"] / 1000.0
        )
        send_frame(
            readiness_connection,
            authenticated_envelope(readiness, authentication_key=control_key),
            deadline_monotonic=startup_deadline,
        )
        readiness_connection.close()
        running = True
        while running:
            connection, _ = listener.accept()
            with connection:
                connection_deadline = monotonic() + (
                    max(timeout_binding["timeout_milliseconds"].values())
                    / 1000.0
                )
                role: object = None
                sequence: object = None
                nonce: object = None
                operation: object = None
                payload: object = None
                runtime_effect_committed = False
                request_transcript_committed = False
                try:
                    peer = _peer_identity(connection)
                    envelope = receive_frame(
                        connection,
                        deadline_monotonic=connection_deadline,
                    )
                    unsigned = envelope.get("body")
                    role = unsigned.get("role") if isinstance(unsigned, Mapping) else None
                    if role not in {"runtime", "orchestration", "control"}:
                        raise ProcessBrokerProtocolError("process-broker role is forbidden")
                    key = {
                        "runtime": runtime_key,
                        "orchestration": orchestration_key,
                        "control": control_key,
                    }[str(role)]
                    body = verify_authenticated_envelope(
                        envelope, authentication_key=key
                    )
                    sequence = body.get("sequence")
                    nonce = body.get("nonce")
                    operation = body.get("operation")
                    payload = body.get("payload")
                    if (
                        type(sequence) is not int
                        or sequence < 0
                        or type(operation) is not str
                        or not operation
                    ):
                        raise ProcessBrokerProtocolError(
                            "process-broker authenticated transcript metadata is invalid"
                        )
                    transcript.commit_authenticated_frame(
                        direction="request",
                        role=str(role),
                        sequence=sequence,
                        operation=operation,
                        frame=envelope,
                        payload=payload,
                    )
                    request_transcript_committed = True
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
                    if (
                        sequence != sequences[role]
                        or type(nonce) is not str
                        or len(nonce) != 64
                        or nonce in seen_nonces
                    ):
                        raise ProcessBrokerProtocolError(
                            "process-broker sequence/nonce is replayed or invalid"
                        )
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
                        if runtime_control_cleanup_only:
                            raise ProcessBrokerProtocolError(
                                "runtime broker infrastructure invalid requires "
                                "control cleanup"
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
                        elif operation in {
                            "runtime_execute",
                            "runtime_register_rejected",
                        }:
                            if pending_action is not None:
                                raise ProcessBrokerProtocolError(
                                    "runtime action request awaits its post observation"
                                )
                            if terminal_required:
                                raise ProcessBrokerProtocolError(
                                    "runtime action request requires pending terminal receipt"
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
                        except ProcessBrokerInfrastructureInvalidResponse:
                            runtime_failed = True
                            runtime_control_cleanup_only = True
                            raise
                        except Exception:
                            runtime_failed = True
                            raise
                        # From this point a browser/reset/rejection/terminal
                        # effect may be durable. If the authenticated response
                        # cannot be delivered, the caller cannot distinguish
                        # committed from uncommitted state and the episode must
                        # remain poisoned for finalization.
                        runtime_effect_committed = True
                        if operation == "runtime_reset":
                            bound_episode_task = requested_identity
                            last_observation = dict(result["observation"])
                            last_action = None
                            terminal_required = True
                        elif operation == "runtime_observe":
                            pending_action = None
                            last_observation = dict(result["observation"])
                            terminal_required = True
                        elif operation in {
                            "runtime_execute",
                            "runtime_register_rejected",
                        }:
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
                    elif role == "orchestration":
                        if operation != SEALED_FINALIZATION_OPERATION:
                            raise ProcessBrokerProtocolError(
                                "orchestration requested an unregistered operation"
                            )
                        validated_payload = validate_finalization_request_payload(
                            payload
                        )
                        requested_identity = (
                            str(validated_payload["episode_id"]),
                            str(validated_payload["task_id"]),
                        )
                        if bound_episode_task is None:
                            raise ProcessBrokerProtocolError(
                                "sealed finalization requires a reset episode"
                            )
                        if requested_identity != bound_episode_task:
                            raise ProcessBrokerProtocolError(
                                "sealed finalization episode/task identity differs"
                            )
                        if finalization_complete:
                            raise ProcessBrokerProtocolError(
                                "sealed finalization may execute exactly once"
                            )
                        if pending_action is not None:
                            raise ProcessBrokerProtocolError(
                                "sealed finalization awaits a post-action observation"
                            )
                        if terminal_required:
                            raise ProcessBrokerProtocolError(
                                "sealed finalization awaits a causal terminal receipt"
                            )
                        if runtime_failed or runtime_control_cleanup_only:
                            raise ProcessBrokerProtocolError(
                                "sealed finalization cannot consume a failed runtime"
                            )
                        finalizer = getattr(
                            backend, SEALED_FINALIZATION_OPERATION, None
                        )
                        if not callable(finalizer):
                            raise ProcessBrokerProtocolError(
                                "sealed backend has no orchestration finalizer"
                            )
                        # Possession of the distinct orchestration key is the
                        # post-decision boundary.  Atomically close the runtime
                        # role even when an EpisodeTimeout prevented its
                        # best-effort runtime_close call.
                        runtime_closed = True
                        sequences[role] += 1
                        seen_nonces.add(nonce)
                        try:
                            backend_result = _detach_backend_result(
                                finalizer(validated_payload)
                            )
                            _verify_imported_broker_sources(config)
                            result = validate_finalization_result(backend_result)
                            _verify_imported_broker_sources(config)
                        except Exception:
                            runtime_failed = True
                            raise
                        finalization_complete = True
                        response_role = "sealed_orchestration_response"
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
                        if not callable(shutdown):  # checked during factory load
                            raise ProcessBrokerProtocolError(
                                "sealed backend child cleanup operation disappeared"
                            )
                        child_cleanup_result = validate_child_cleanup_result(
                            shutdown()
                        )
                        disposition = child_cleanup_result[
                            "cleanup_disposition"
                        ]
                        if runtime_closed and disposition != (
                            "RUNTIME_CLOSE_ACKNOWLEDGED"
                        ):
                            raise ProcessBrokerProtocolError(
                                "child cleanup disposition differs from closed runtime"
                            )
                        if not runtime_closed and disposition == (
                            "RUNTIME_CLOSE_ACKNOWLEDGED"
                        ):
                            raise ProcessBrokerProtocolError(
                                "child cleanup disposition invents runtime close"
                            )
                        if (
                            bound_episode_task is not None
                            and disposition == "NO_BROWSER_CREATED"
                        ):
                            raise ProcessBrokerProtocolError(
                                "child cleanup denies a reset browser session"
                            )
                        _verify_imported_broker_sources(config)
                        transcript_before_response = transcript.snapshot()
                        result = {
                            "shutdown": True,
                            "child_cleanup_result": child_cleanup_result,
                            "transcript_root_before_response_sha256": (
                                transcript_before_response["root_sha256"]
                            ),
                            "transcript_entry_count_before_response": (
                                transcript_before_response["entry_count"]
                            ),
                        }
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
                    response_frame = authenticated_envelope(
                        response, authentication_key=key
                    )
                    transcript.commit_authenticated_frame(
                        direction="response",
                        role=str(role),
                        sequence=int(sequence),
                        operation=str(operation),
                        frame=response_frame,
                        payload=response["result"],
                    )
                    send_frame(
                        connection,
                        response_frame,
                        deadline_monotonic=connection_deadline,
                    )
                except Exception as exc:
                    if role == "runtime" and runtime_effect_committed:
                        # A durable sealed append followed by response loss is
                        # ambiguous to the caller. Never permit finalization or
                        # another runtime request from that causal prefix.
                        runtime_failed = True
                    # Do not reflect exception text, backend state, or evaluator
                    # details across the runtime boundary.
                    try:
                        role = (
                            role
                            if role in {"runtime", "orchestration", "control"}
                            else "runtime"
                        )
                        key = {
                            "runtime": runtime_key,
                            "orchestration": orchestration_key,
                            "control": control_key,
                        }[str(role)]
                        infrastructure_result: dict[str, Any] | None = None
                        if role == "runtime" and isinstance(
                            exc,
                            ProcessBrokerInfrastructureInvalidResponse,
                        ):
                            try:
                                _verify_imported_broker_sources(config)
                                infrastructure_result = (
                                    validate_runtime_infrastructure_invalid(
                                        exc.result,
                                        request_operation=str(operation),
                                        episode_id=str(
                                            validated_payload["episode_id"]
                                        ),
                                        request_payload=validated_payload,
                                    )
                                )
                            except Exception:
                                infrastructure_result = None
                        response = {
                            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
                            "role": (
                                "sealed_runtime_response"
                                if role == "runtime"
                                else (
                                    "sealed_orchestration_response"
                                    if role == "orchestration"
                                    else "sealed_control_response"
                                )
                            ),
                            "session_id": session_id,
                            "sequence": (
                                sequence if type(sequence) is int else -1
                            ),
                            "nonce": nonce if type(nonce) is str else "",
                            "status": (
                                PROCESS_BROKER_INFRASTRUCTURE_INVALID_STATUS
                                if infrastructure_result is not None
                                else "REJECTED"
                            ),
                            "result": (
                                infrastructure_result
                                if infrastructure_result is not None
                                else {}
                            ),
                        }
                        if infrastructure_result is None:
                            # A backend-controlled exception class name would
                            # be another neutral scalar channel. Keep generic
                            # rejection disclosure closed and deterministic.
                            response["error_code"] = "REGISTERED_REQUEST_REJECTED"
                        response_frame = authenticated_envelope(
                            response, authentication_key=key
                        )
                        if request_transcript_committed:
                            transcript.commit_authenticated_frame(
                                direction="response",
                                role=str(role),
                                sequence=int(sequence),
                                operation=str(operation),
                                frame=response_frame,
                                payload=response["result"],
                            )
                        send_frame(
                            connection,
                            response_frame,
                            deadline_monotonic=connection_deadline,
                        )
                    except Exception:
                        pass
    finally:
        try:
            readiness_connection.close()
        except OSError:
            pass
        listener.close()
        try:
            endpoint.unlink()
        except FileNotFoundError:
            pass
    return 0


def main() -> int:
    raise ProcessBrokerProtocolError(
        "process-broker worker requires the source-verifying dedicated-FD launcher"
    )


if __name__ == "__main__":
    raise SystemExit(main())
