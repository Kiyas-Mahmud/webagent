"""Lifecycle owner for a source-bound, process-isolated Table 2 broker.

The receipt produced here is local architecture evidence only. The worker is a
same-UID child and that fact is not a hostile-code or secret boundary. It cannot
authorize a live campaign: deployment must separately prove process, source,
peer-credential, secret-scrubbing and filesystem boundaries under an external
trust anchor registered by the execution guard.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from time import monotonic
from types import MappingProxyType
from typing import Any, Mapping

from .common import canonical_json_bytes, SchemaError, sha256_file, sha256_json
from .process_broker_protocol import (
    MAX_PROCESS_BROKER_MESSAGE_BYTES,
    POLICY_SCREENSHOT_TRANSPORT_CONTRACT,
    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256,
    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION,
    PROCESS_BROKER_PROTOCOL_VERSION,
    PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS,
    RUNTIME_INNER_SCHEMA_PATHS,
    authenticated_envelope,
    receive_frame,
    remaining_deadline_seconds,
    set_socket_timeout_to_deadline,
    send_frame,
    validated_policy_screenshot_root,
    verify_authenticated_envelope,
)
from .process_broker_runtime import ProcessIsolatedRuntimeClient
# Import the sink implementation eagerly so startup verifies the same source
# bytes before the isolated child receives its child-owned stream target.
from . import sealed_verifier as _sealed_verifier_source  # noqa: F401
from .process_broker_finalization import (
    ProcessIsolatedFinalizationClient,
)
from .process_broker_timeout import (
    MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES,
    MEASURED_TIMEOUT_MODE,
    ProcessBrokerTimeoutExpectedAuthority,
    binding_sha256 as timeout_binding_sha256,
    engineering_timeout_binding,
    load_authority_bound_timeout_calibration,
    measured_timeout_binding,
    validate_process_broker_timeout_binding,
    validate_process_broker_timeout_calibration,
)
from ...runtime.deadline import interrupt_after


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

CONTROL_OPERATION = "sealed_control_shutdown"


_PROCESS_BROKER_STDLIB_LAUNCHER = r"""
import hashlib,json,os,pathlib,socket,stat,struct,sys
channel=socket.socket(fileno=int(sys.argv[2]))
def read_exact(size):
    chunks=[]
    while size:
        chunk=channel.recv(size)
        if not chunk:
            raise RuntimeError("launch channel closed early")
        chunks.append(chunk)
        size-=len(chunk)
    return b"".join(chunks)
def identity(value):
    return (value.st_dev,value.st_ino,value.st_mode,value.st_nlink,value.st_uid,value.st_gid,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
size=struct.unpack("!I",read_exact(4))[0]
if size <= 0 or size > 1048576:
    raise RuntimeError("launch config frame length differs")
payload=read_exact(size)
config=json.loads(payload.decode("utf-8"))
raw_root=config.get("repository_root")
if type(raw_root) is not str or not pathlib.Path(raw_root).is_absolute():
    raise RuntimeError("unsafe process-broker repository root")
root=pathlib.Path(os.path.abspath(raw_root))
directory_flags=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)
file_flags=os.O_RDONLY|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0)
root_fd=os.open(root,directory_flags)
try:
    root_identity=identity(os.fstat(root_fd))
    root_path_identity=identity(os.stat(root,follow_symlinks=False))
    if root_identity != root_path_identity or not stat.S_ISDIR(root_identity[2]):
        raise RuntimeError("unsafe process-broker repository root")
    rows=config.get("source_files")
    if type(rows) is not list or not rows:
        raise RuntimeError("process-broker source closure is absent")
    seen=set()
    for row in rows:
        if type(row) is not dict or set(row) != {"relative_path","sha256"}:
            raise RuntimeError("malformed process-broker source row")
        relative=row["relative_path"]
        expected=row["sha256"]
        lexical=pathlib.PurePosixPath(relative) if type(relative) is str else None
        if (lexical is None or not relative or relative != relative.strip() or "\\" in relative or "\x00" in relative or lexical.is_absolute() or lexical.as_posix() != relative or ".." in lexical.parts or "." in lexical.parts or lexical.suffix != ".py" or relative in seen):
            raise RuntimeError("unsafe process-broker source")
        if type(expected) is not str or len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
            raise RuntimeError("malformed process-broker source digest")
        seen.add(relative)
        descriptors=[os.dup(root_fd)]
        bindings=[]
        try:
            parent=descriptors[0]
            for component in lexical.parts[:-1]:
                before=os.stat(component,dir_fd=parent,follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise RuntimeError("unsafe process-broker source ancestry")
                child=os.open(component,directory_flags,dir_fd=parent)
                descriptors.append(child)
                opened=os.fstat(child)
                after=os.stat(component,dir_fd=parent,follow_symlinks=False)
                if identity(before) != identity(opened) or identity(opened) != identity(after):
                    raise RuntimeError("process-broker source ancestry changed")
                bindings.append((parent,component,child,identity(opened)))
                parent=child
            leaf=lexical.parts[-1]
            before=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RuntimeError("process-broker source is not a single-link regular file")
            source_fd=os.open(leaf,file_flags,dir_fd=parent)
            descriptors.append(source_fd)
            opened=os.fstat(source_fd)
            rebound=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
            if identity(before) != identity(opened) or identity(opened) != identity(rebound) or opened.st_nlink != 1:
                raise RuntimeError("process-broker source changed before read")
            digest=hashlib.sha256()
            total=0
            while True:
                block=os.read(source_fd,1048576)
                if not block:
                    break
                total+=len(block)
                digest.update(block)
            opened_after=os.fstat(source_fd)
            rebound_after=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
            if total != opened.st_size or identity(opened) != identity(opened_after) or identity(opened_after) != identity(rebound_after) or opened_after.st_nlink != 1:
                raise RuntimeError("process-broker source changed during read")
            for bound_parent,component,child,bound_identity in bindings:
                if bound_identity != identity(os.fstat(child)) or bound_identity != identity(os.stat(component,dir_fd=bound_parent,follow_symlinks=False)):
                    raise RuntimeError("process-broker source ancestry changed")
            if digest.hexdigest() != expected:
                raise RuntimeError("process-broker source changed before import")
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
    if root_identity != identity(os.fstat(root_fd)) or root_identity != identity(os.stat(root,follow_symlinks=False)):
        raise RuntimeError("process-broker repository root changed before import")
finally:
    os.close(root_fd)
sys.path.insert(0,sys.argv[1])
from web_agent.eval.table2.process_broker_worker import serve
raise SystemExit(serve(config, readiness_connection=channel))
"""


PROCESS_BROKER_RECEIPT_SCHEMA_VERSION = "table2-process-page-broker-receipt-v9"
PROCESS_BROKER_CLEANUP_RECEIPT_SCHEMA_VERSION = (
    "table2-process-page-broker-cleanup-receipt-v1"
)
PROCESS_BROKER_ENGINEERING_SCOPE = "ENGINEERING_FIXTURE_ONLY"
PROCESS_BROKER_MEASURED_REPLAY_SCOPE = "MEASURED_CALIBRATION_REPLAY_ONLY"
PROCESS_BROKER_EVALUATION_SCOPE = "EVALUATION"
PROCESS_BROKER_SOURCE_PATHS = (
    "src/web_agent/__init__.py",
    "src/web_agent/benchmarks/__init__.py",
    "src/web_agent/benchmarks/base.py",
    "src/web_agent/eval/__init__.py",
    "src/web_agent/eval/table2/__init__.py",
    "src/web_agent/eval/table2/common.py",
    "src/web_agent/eval/table2/execution_guard.py",
    "src/web_agent/eval/table2/process_broker.py",
    "src/web_agent/eval/table2/process_broker_finalization.py",
    "src/web_agent/eval/table2/process_broker_protocol.py",
    "src/web_agent/eval/table2/process_broker_runtime.py",
    "src/web_agent/eval/table2/sealed_verifier.py",
    "src/web_agent/eval/table2/process_broker_timeout.py",
    "src/web_agent/eval/table2/process_broker_webarena_backend.py",
    "src/web_agent/eval/table2/process_broker_worker.py",
    "src/web_agent/runtime/__init__.py",
    "src/web_agent/runtime/contracts.py",
    "src/web_agent/runtime/deadline.py",
    "src/web_agent/runtime/state_reset.py",
)

if MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES > (
    MAX_PROCESS_BROKER_MESSAGE_BYTES * 3 // 8
):  # pragma: no cover - import-time contract
    raise RuntimeError(
        "timeout calibration payload no longer leaves registered launch overhead"
    )


def _validated_repository_python_source(
    repository_root: Path,
    relative_path: object,
    *,
    label: str,
) -> tuple[Path, str]:
    """Resolve one explicit repository source without following symlink ancestry."""

    if (
        type(relative_path) is not str
        or not relative_path
        or relative_path != relative_path.strip()
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise SchemaError(f"{label} must be a canonical relative Python path")
    lexical = Path(relative_path)
    if (
        lexical.is_absolute()
        or ".." in lexical.parts
        or "." in lexical.parts
        or lexical.suffix != ".py"
        or lexical.as_posix() != relative_path
    ):
        raise SchemaError(f"{label} must be a canonical relative Python path")
    candidate = repository_root
    opened_components: list[tuple[Path, tuple[int, ...]]] = []
    try:
        for part in lexical.parts:
            candidate = candidate / part
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise SchemaError(f"{label} contains a symlink component")
            opened_components.append(
                (
                    candidate,
                    (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mode,
                        metadata.st_nlink,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    ),
                )
            )
    except OSError as exc:
        raise SchemaError(f"{label} is missing") from exc
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise SchemaError(f"{label} escaped repository") from exc
    metadata = resolved.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SchemaError(f"{label} is not a single-link regular Python source")
    try:
        for component, before in opened_components:
            after = component.lstat()
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before != after_identity or stat.S_ISLNK(after.st_mode):
                raise SchemaError(f"{label} changed during validation")
    except OSError as exc:
        raise SchemaError(f"{label} changed during validation") from exc
    return resolved, lexical.as_posix()


def _validated_readiness_envelope(
    value: object,
    *,
    control_key: bytes,
    session_id: str,
    startup_nonce: str,
    child_pid: int,
    sealed_backend_config_sha256: str,
    policy_screenshot_root_identity_sha256: str | None,
    ipc_timeout_binding_sha256: str,
    sealed_finalization_required: bool,
) -> tuple[int, bool]:
    ready = verify_authenticated_envelope(value, authentication_key=control_key)
    if set(ready) != {
        "protocol_version",
        "role",
        "session_id",
        "startup_nonce",
        "status",
        "evaluator_pid",
        "endpoint_mode",
        "sealed_backend_config_sha256",
        "policy_screenshot_root_identity_sha256",
        "ipc_timeout_binding_sha256",
        "sealed_finalization_available",
        "sealed_finalization_required",
    }:
        raise RuntimeError("process-broker worker readiness fields differ")
    evaluator_pid = ready.get("evaluator_pid")
    if (
        ready.get("protocol_version") != PROCESS_BROKER_PROTOCOL_VERSION
        or ready.get("role") != "sealed_readiness"
        or ready.get("session_id") != session_id
        or ready.get("startup_nonce") != startup_nonce
        or ready.get("status") != "READY"
        or ready.get("endpoint_mode") != "0o600"
        or ready.get("sealed_backend_config_sha256")
        != sealed_backend_config_sha256
        or ready.get("policy_screenshot_root_identity_sha256")
        != policy_screenshot_root_identity_sha256
        or ready.get("ipc_timeout_binding_sha256")
        != ipc_timeout_binding_sha256
        or type(ready.get("sealed_finalization_available")) is not bool
        or ready.get("sealed_finalization_required")
        is not sealed_finalization_required
        or (
            sealed_finalization_required
            and ready.get("sealed_finalization_available") is not True
        )
        or type(evaluator_pid) is not int
        or evaluator_pid != child_pid
    ):
        raise RuntimeError("process-broker worker readiness failed")
    return evaluator_pid, bool(ready["sealed_finalization_available"])


def _validate_shutdown_response(
    value: object,
    *,
    control_key: bytes,
    session_id: str,
    sequence: int,
    nonce: str,
) -> None:
    response = verify_authenticated_envelope(value, authentication_key=control_key)
    expected = {
        "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
        "role": "sealed_control_response",
        "session_id": session_id,
        "sequence": sequence,
        "nonce": nonce,
        "status": "PASS",
        "result": {"shutdown": True},
    }
    if response != expected:
        raise RuntimeError("process-broker shutdown was rejected")


def _verify_parent_imported_sources(
    repository_root: Path, source_rows: tuple[dict[str, str], ...]
) -> None:
    expected = {row["relative_path"]: row["sha256"] for row in source_rows}
    modules = {
        "web_agent": "src/web_agent/__init__.py",
        "web_agent.benchmarks": "src/web_agent/benchmarks/__init__.py",
        "web_agent.benchmarks.base": "src/web_agent/benchmarks/base.py",
        "web_agent.eval": "src/web_agent/eval/__init__.py",
        "web_agent.eval.table2": "src/web_agent/eval/table2/__init__.py",
        "web_agent.eval.table2.common": "src/web_agent/eval/table2/common.py",
        "web_agent.eval.table2.execution_guard": (
            "src/web_agent/eval/table2/execution_guard.py"
        ),
        "web_agent.eval.table2.process_broker": (
            "src/web_agent/eval/table2/process_broker.py"
        ),
        "web_agent.eval.table2.process_broker_finalization": (
            "src/web_agent/eval/table2/process_broker_finalization.py"
        ),
        "web_agent.eval.table2.process_broker_protocol": (
            "src/web_agent/eval/table2/process_broker_protocol.py"
        ),
        "web_agent.eval.table2.process_broker_runtime": (
            "src/web_agent/eval/table2/process_broker_runtime.py"
        ),
        "web_agent.eval.table2.sealed_verifier": (
            "src/web_agent/eval/table2/sealed_verifier.py"
        ),
        "web_agent.eval.table2.process_broker_timeout": (
            "src/web_agent/eval/table2/process_broker_timeout.py"
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
        expected_sha256 = expected.get(relative)
        import_time_sha256 = getattr(
            module,
            "PROCESS_BROKER_IMPORT_SOURCE_SHA256",
            None,
        )
        if (
            loaded != registered
            or loaded.suffix != ".py"
            or loaded.stat().st_nlink != 1
            or expected_sha256 != sha256_file(loaded)
            or type(import_time_sha256) is not str
            or import_time_sha256 != expected_sha256
        ):
            raise SchemaError(
                f"imported parent broker module identity differs: {module_name}"
            )


def _verify_optional_parent_backend_source(
    *,
    repository_root: Path,
    backend_entrypoint: str,
    backend_source: Path,
    source_rows: tuple[dict[str, str], ...],
) -> None:
    """Bind an already imported parent-side backend/bridge to receipt bytes."""

    module_name = backend_entrypoint.partition(":")[0]
    module = sys.modules.get(module_name)
    if module is None:
        return
    loaded = Path(str(getattr(module, "__file__", ""))).resolve()
    expected = {
        row["relative_path"]: row["sha256"] for row in source_rows
    }.get(backend_source.relative_to(repository_root).as_posix())
    import_time_sha256 = getattr(
        module,
        "PROCESS_BROKER_IMPORT_SOURCE_SHA256",
        None,
    )
    if (
        loaded != backend_source
        or loaded.suffix != ".py"
        or loaded.stat().st_nlink != 1
        or expected != sha256_file(loaded)
        or type(import_time_sha256) is not str
        or import_time_sha256 != expected
    ):
        raise SchemaError("imported parent backend module identity differs")


@dataclass(frozen=True, slots=True)
class ProcessBrokerReceipt:
    schema_version: str
    record_type: str
    claim_scope: str
    status: str
    execution_scope: str
    runtime_pid: int
    sealed_evaluator_pid: int
    process_ids_distinct: bool
    transport: str
    protocol_version: str
    peer_credentials_enforced: bool
    role_separated_authentication: bool
    outer_envelope_fields_exact: bool
    forbidden_named_keys_rejected_recursively: bool
    operation_specific_inner_schema_paths: tuple[str, ...]
    inner_schema_registry_version: str
    inner_schema_registry_sha256: str
    operation_specific_inner_schemas_registered: bool
    loaded_source_closure_enforced: bool
    single_episode_task_session_enforced: bool
    observation_stage_prior_action_bound: bool
    verifier_receipt_causal_binding_enforced: bool
    reset_operation_registered: bool
    executor_local_rejection_registration_enforced: bool
    policy_screenshot_transport_contract: str
    policy_screenshot_root_identity_sha256: str | None
    policy_screenshot_root_bound_per_session: bool
    canonical_wire_json_enforced: bool
    runtime_client_ambiguous_failure_poisoned: bool
    sealed_backend_config_hash_bound: bool
    ipc_timeout_binding: Mapping[str, Any]
    ipc_timeout_binding_sha256: str
    measured_ipc_timeout_calibration_complete: bool
    timeout_calibration_replay_only: bool
    immutable_timeout_authority_bundle_validated: bool
    external_timeout_authority_cross_binding_present: bool
    ipc_timeout_calibration_live_campaign_authority: bool
    runtime_value_provenance_attested: bool
    evaluator_operation_in_runtime_allowlist: bool
    sealed_finalization_capability_available: bool
    sealed_finalization_required: bool
    sealed_finalization_operation_in_runtime_allowlist: bool
    child_owned_sealed_sink: bool
    runtime_adapter_sealed_capability_free: bool
    separate_evidence_transport_present: bool
    runtime_terminal_returns_outer_sealed_signal: bool
    sealed_finalization_returns_outer_sealed_signal: bool
    sealed_finalization_timeout_campaign_authority: bool
    future_promotion_requirements: tuple[str, ...]
    backend_entrypoint: str
    backend_entrypoint_sha256: str
    backend_source_relative_path: str
    backend_source_sha256: str
    sealed_backend_config_sha256: str
    endpoint_identity_sha256: str
    runtime_key_identity_sha256: str
    orchestration_key_identity_sha256: str
    control_key_identity_sha256: str
    source_files: tuple[dict[str, str], ...]
    source_set_sha256: str
    external_deployment_authority: bool

    def __post_init__(self) -> None:
        fixed = {
            "schema_version": PROCESS_BROKER_RECEIPT_SCHEMA_VERSION,
            "record_type": "ProcessBrokerInnerSchemaCompatibilityReceipt",
            "claim_scope": (
                "LOCAL_DISTINCT_PROCESS_KEY_ENVELOPE_AND_INNER_SCHEMA_EVIDENCE_"
                "NOT_VALUE_PROVENANCE_OR_DEPLOYMENT_AUTHORITY"
            ),
            "status": (
                "LOCAL_DISTINCT_PROCESS_ENVELOPE_AND_INNER_SCHEMA_"
                "CONFORMANCE_PASS"
            ),
            "process_ids_distinct": True,
            "transport": "AF_UNIX_JSON_HMAC_SHA256_PEERCRED",
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "peer_credentials_enforced": True,
            "role_separated_authentication": True,
            "outer_envelope_fields_exact": True,
            "forbidden_named_keys_rejected_recursively": True,
            "inner_schema_registry_version": (
                PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION
            ),
            "inner_schema_registry_sha256": (
                PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256
            ),
            "operation_specific_inner_schemas_registered": True,
            "loaded_source_closure_enforced": True,
            "single_episode_task_session_enforced": True,
            "observation_stage_prior_action_bound": True,
            "verifier_receipt_causal_binding_enforced": True,
            "reset_operation_registered": True,
            "executor_local_rejection_registration_enforced": True,
            "policy_screenshot_transport_contract": (
                POLICY_SCREENSHOT_TRANSPORT_CONTRACT
            ),
            "policy_screenshot_root_bound_per_session": True,
            "canonical_wire_json_enforced": True,
            "runtime_client_ambiguous_failure_poisoned": True,
            "sealed_backend_config_hash_bound": True,
            "ipc_timeout_calibration_live_campaign_authority": False,
            "external_timeout_authority_cross_binding_present": False,
            "runtime_value_provenance_attested": False,
            "evaluator_operation_in_runtime_allowlist": False,
            "sealed_finalization_operation_in_runtime_allowlist": False,
            "runtime_adapter_sealed_capability_free": True,
            "separate_evidence_transport_present": False,
            "sealed_finalization_timeout_campaign_authority": False,
            "external_deployment_authority": False,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise SchemaError(f"process-broker receipt {name} differs")
        if self.execution_scope not in {
            PROCESS_BROKER_ENGINEERING_SCOPE,
            PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
            PROCESS_BROKER_EVALUATION_SCOPE,
        }:
            raise SchemaError("process-broker receipt execution scope is unregistered")
        if self.operation_specific_inner_schema_paths != RUNTIME_INNER_SCHEMA_PATHS:
            raise SchemaError(
                "process-broker receipt inner-schema paths differ"
            )
        if self.future_promotion_requirements != (
            PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
        ):
            raise SchemaError(
                "process-broker receipt future promotion requirements differ"
            )
        timeout_binding = validate_process_broker_timeout_binding(
            self.ipc_timeout_binding,
            calibration_evidence=None,
        )
        if timeout_binding_sha256(timeout_binding) != self.ipc_timeout_binding_sha256:
            raise SchemaError("process-broker receipt timeout-binding hash differs")
        expected_complete = timeout_binding.get("mode") == MEASURED_TIMEOUT_MODE
        if self.measured_ipc_timeout_calibration_complete is not expected_complete:
            raise SchemaError("process-broker receipt timeout-calibration status differs")
        if self.timeout_calibration_replay_only is not (
            self.execution_scope == PROCESS_BROKER_MEASURED_REPLAY_SCOPE
        ):
            raise SchemaError("process-broker receipt replay-only status differs")
        if self.immutable_timeout_authority_bundle_validated and not (
            self.timeout_calibration_replay_only and expected_complete
        ):
            raise SchemaError(
                "process-broker immutable timeout bundle status differs"
            )
        if self.execution_scope == PROCESS_BROKER_EVALUATION_SCOPE:
            raise SchemaError(
                "evaluation broker receipt requires unavailable external timeout authority"
            )
        if self.sealed_finalization_required and not (
            self.sealed_finalization_capability_available
        ):
            raise SchemaError(
                "process-broker receipt requires an unavailable finalizer"
            )
        for name in (
            "sealed_finalization_capability_available",
            "sealed_finalization_required",
            "child_owned_sealed_sink",
            "runtime_terminal_returns_outer_sealed_signal",
            "sealed_finalization_returns_outer_sealed_signal",
        ):
            if type(getattr(self, name)) is not bool:
                raise SchemaError(f"process-broker receipt {name} is not boolean")
        if any(
            value is not self.sealed_finalization_capability_available
            for value in (
                self.child_owned_sealed_sink,
                self.runtime_terminal_returns_outer_sealed_signal,
                self.sealed_finalization_returns_outer_sealed_signal,
            )
        ):
            raise SchemaError(
                "process-broker child-owned sealed capabilities differ"
            )
        if (
            self.execution_scope == PROCESS_BROKER_ENGINEERING_SCOPE
            and expected_complete
        ) or (
            self.execution_scope == PROCESS_BROKER_MEASURED_REPLAY_SCOPE
            and not expected_complete
        ):
            raise SchemaError("process-broker receipt timeout mode differs from scope")
        if self.runtime_pid <= 0 or self.sealed_evaluator_pid <= 0:
            raise SchemaError("process-broker receipt process identity is invalid")
        rows = list(self.source_files)
        if rows != sorted(rows, key=lambda row: row["relative_path"]):
            raise SchemaError("process-broker receipt sources are not sorted")
        if self.source_set_sha256 != sha256_json(rows):
            raise SchemaError("process-broker receipt source-set hash differs")
        if hashlib.sha256(self.backend_entrypoint.encode("utf-8")).hexdigest() != (
            self.backend_entrypoint_sha256
        ):
            raise SchemaError("process-broker backend entrypoint hash differs")
        matching_backend = [
            row
            for row in rows
            if row["relative_path"] == self.backend_source_relative_path
        ]
        if len(matching_backend) != 1 or matching_backend[0]["sha256"] != (
            self.backend_source_sha256
        ):
            raise SchemaError("process-broker backend source is not receipt-bound")
        for name in (
            "sealed_backend_config_sha256",
            "runtime_key_identity_sha256",
            "orchestration_key_identity_sha256",
            "control_key_identity_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise SchemaError(f"process-broker receipt {name} is not SHA-256")
        root_identity = self.policy_screenshot_root_identity_sha256
        if root_identity is not None and (
            len(root_identity) != 64
            or any(c not in "0123456789abcdef" for c in root_identity)
        ):
            raise SchemaError(
                "process-broker screenshot-root identity is not SHA-256"
            )

    def to_dict(self) -> dict[str, Any]:
        value = {item.name: getattr(self, item.name) for item in fields(self)}
        value["source_files"] = [dict(row) for row in self.source_files]
        value["operation_specific_inner_schema_paths"] = list(
            self.operation_specific_inner_schema_paths
        )
        value["future_promotion_requirements"] = list(
            self.future_promotion_requirements
        )
        value["ipc_timeout_binding"] = dict(self.ipc_timeout_binding)
        value["ipc_timeout_binding"]["timeout_milliseconds"] = dict(
            self.ipc_timeout_binding["timeout_milliseconds"]
        )
        return value


@dataclass(frozen=True, slots=True)
class ProcessBrokerCleanupReceipt:
    """Source/session-bound proof that the evaluator process and socket ended."""

    schema_version: str
    record_type: str
    claim_scope: str
    status: str
    launch_receipt_sha256: str
    session_identity_sha256: str
    sealed_evaluator_pid: int
    worker_exit_code: int
    graceful_authenticated_shutdown: bool
    endpoint_removed: bool
    temporary_directory_removed: bool
    source_set_sha256: str
    external_deployment_authority: bool

    def __post_init__(self) -> None:
        fixed = {
            "schema_version": PROCESS_BROKER_CLEANUP_RECEIPT_SCHEMA_VERSION,
            "record_type": "ProcessIsolatedPageBrokerCleanupReceipt",
            "claim_scope": "LOCAL_ARCHITECTURE_EVIDENCE_NOT_DEPLOYMENT_AUTHORITY",
            "status": "LOCAL_CLEANUP_PASS",
            "worker_exit_code": 0,
            "graceful_authenticated_shutdown": True,
            "endpoint_removed": True,
            "temporary_directory_removed": True,
            "external_deployment_authority": False,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise SchemaError(f"process-broker cleanup receipt {name} differs")
        for name in (
            "launch_receipt_sha256",
            "session_identity_sha256",
            "source_set_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise SchemaError(f"process-broker cleanup {name} is not SHA-256")
        if self.sealed_evaluator_pid <= 0:
            raise SchemaError("process-broker cleanup evaluator PID is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class ProcessIsolatedBroker:
    """Orchestrator capability holding control key separately from runtime client."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        backend_entrypoint: str,
        backend_source_relative_path: str,
        backend_dependency_source_relative_paths: tuple[str, ...] = (),
        sealed_backend_config: Mapping[str, Any] | None = None,
        policy_screenshot_root: str | Path | None = None,
        startup_timeout_seconds: float | None = None,
        runtime_timeout_seconds: float | None = None,
        control_shutdown_timeout_seconds: float | None = None,
        timeout_calibration_evidence: Mapping[str, Any] | None = None,
        timeout_calibration_artifact_path: str | Path | None = None,
        timeout_calibration_expected_authority: (
            ProcessBrokerTimeoutExpectedAuthority | None
        ) = None,
        execution_scope: str = PROCESS_BROKER_ENGINEERING_SCOPE,
        require_sealed_finalization: bool = False,
    ) -> None:
        if type(require_sealed_finalization) is not bool:
            raise SchemaError(
                "process-broker finalization requirement must be boolean"
            )
        self._sealed_finalization_required = require_sealed_finalization
        self._repo = Path(repository_root).resolve()
        self._backend_entrypoint = backend_entrypoint
        self._backend_source, backend_relative = (
            _validated_repository_python_source(
                self._repo,
                backend_source_relative_path,
                label="process-broker backend source",
            )
        )
        if type(backend_dependency_source_relative_paths) is not tuple or any(
            type(relative) is not str
            for relative in backend_dependency_source_relative_paths
        ):
            raise SchemaError(
                "process-broker backend dependency sources must be a tuple of paths"
            )
        dependency_relatives: list[str] = []
        for relative in backend_dependency_source_relative_paths:
            _source, canonical_relative = _validated_repository_python_source(
                self._repo,
                relative,
                label="process-broker backend dependency source",
            )
            if canonical_relative == backend_relative:
                raise SchemaError(
                    "process-broker backend source is duplicated in dependency closure"
                )
            if canonical_relative in dependency_relatives:
                raise SchemaError(
                    "process-broker backend dependency source is duplicated"
                )
            dependency_relatives.append(canonical_relative)
        self._backend_dependency_source_relatives = tuple(dependency_relatives)
        module_name, separator, attribute = backend_entrypoint.partition(":")
        if separator != ":" or not module_name or not attribute or "." in attribute:
            raise SchemaError("process-broker backend entrypoint is malformed")
        expected_path = Path(*module_name.split(".")).with_suffix(".py")
        try:
            actual_relative = self._backend_source.relative_to(self._repo)
        except ValueError as exc:
            raise SchemaError("process-broker backend escaped repository") from exc
        # Source trees use a src/ prefix while import names do not.
        if actual_relative not in {expected_path, Path("src") / expected_path}:
            raise SchemaError("process-broker backend source differs from entrypoint")
        raw_backend_config: Mapping[str, Any] = sealed_backend_config or {
            "schema_version": "table2-empty-sealed-backend-config-v1"
        }
        try:
            detached_backend_config = json.loads(
                canonical_json_bytes(raw_backend_config)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SchemaError("sealed backend config must be canonical JSON") from exc
        if (
            not isinstance(detached_backend_config, dict)
            or type(detached_backend_config.get("schema_version")) is not str
            or not detached_backend_config["schema_version"].strip()
        ):
            raise SchemaError(
                "sealed backend config requires a versioned object"
            )
        self._sealed_backend_config = detached_backend_config
        self._sealed_backend_config_sha256 = sha256_json(detached_backend_config)
        try:
            self._policy_screenshot_root = validated_policy_screenshot_root(
                policy_screenshot_root
            )
        except Exception as exc:
            raise SchemaError("policy screenshot root is invalid") from exc
        self._policy_screenshot_root_identity_sha256 = (
            hashlib.sha256(str(self._policy_screenshot_root).encode("utf-8")).hexdigest()
            if self._policy_screenshot_root is not None
            else None
        )
        if execution_scope not in {
            PROCESS_BROKER_ENGINEERING_SCOPE,
            PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
            PROCESS_BROKER_EVALUATION_SCOPE,
        }:
            raise SchemaError("process-broker execution scope is unregistered")
        supplied_bundle = (
            timeout_calibration_artifact_path is not None
            or timeout_calibration_expected_authority is not None
        )
        if execution_scope == PROCESS_BROKER_EVALUATION_SCOPE:
            if timeout_calibration_evidence is not None:
                raise SchemaError(
                    "evaluation broker forbids self-declared in-memory timeout calibration"
                )
            if (
                timeout_calibration_artifact_path is None
                or timeout_calibration_expected_authority is None
            ):
                raise SchemaError(
                    "evaluation broker requires an immutable calibration artifact "
                    "and typed expected-authority binding"
                )
            if type(timeout_calibration_expected_authority) is not (
                ProcessBrokerTimeoutExpectedAuthority
            ):
                raise SchemaError(
                    "evaluation timeout authority must use the registered type"
                )
            supplied = Path(timeout_calibration_artifact_path).absolute()
            registered = Path(
                timeout_calibration_expected_authority.calibration_artifact_path
            ).absolute()
            if supplied != registered:
                raise SchemaError(
                    "evaluation timeout artifact path differs from expected authority"
                )
            # Validate every local relationship before reporting the residual
            # external gate. This bundle remains explicitly non-authorizing.
            load_authority_bound_timeout_calibration(
                timeout_calibration_expected_authority
            )
            raise SchemaError(
                "evaluation broker remains blocked until an external authority "
                "cross-binds the complete timeout calibration bundle"
            )
        self._execution_scope = execution_scope
        if execution_scope == PROCESS_BROKER_ENGINEERING_SCOPE:
            if timeout_calibration_evidence is not None or supplied_bundle:
                raise SchemaError(
                    "engineering broker cannot claim measured timeout calibration"
                )
            self._timeout_calibration_evidence = None
            self._timeout_authority_bundle_validated = False
            self._timeout_binding = engineering_timeout_binding(
                startup_timeout_seconds=(
                    10.0
                    if startup_timeout_seconds is None
                    else startup_timeout_seconds
                ),
                runtime_timeout_seconds=(
                    10.0
                    if runtime_timeout_seconds is None
                    else runtime_timeout_seconds
                ),
                control_shutdown_timeout_seconds=(
                    3.0
                    if control_shutdown_timeout_seconds is None
                    else control_shutdown_timeout_seconds
                ),
            )
        else:
            if execution_scope != PROCESS_BROKER_MEASURED_REPLAY_SCOPE:
                raise SchemaError("process-broker measured timeout scope differs")
            if any(
                value is not None
                for value in (
                    startup_timeout_seconds,
                    runtime_timeout_seconds,
                    control_shutdown_timeout_seconds,
                )
            ):
                raise SchemaError(
                    "measured broker calibration forbids caller-injected timeouts"
                )
            if timeout_calibration_evidence is not None and supplied_bundle:
                raise SchemaError(
                    "measured replay must use either raw test evidence or one "
                    "immutable authority bundle"
                )
            if timeout_calibration_evidence is not None:
                self._timeout_authority_bundle_validated = False
                self._timeout_calibration_evidence = (
                    validate_process_broker_timeout_calibration(
                        timeout_calibration_evidence
                    )
                )
            else:
                if (
                    timeout_calibration_artifact_path is None
                    or timeout_calibration_expected_authority is None
                ):
                    raise SchemaError(
                        "measured replay requires calibration evidence"
                    )
                if type(timeout_calibration_expected_authority) is not (
                    ProcessBrokerTimeoutExpectedAuthority
                ):
                    raise SchemaError(
                        "replay timeout authority must use the registered type"
                    )
                supplied = Path(timeout_calibration_artifact_path).absolute()
                registered = Path(
                    timeout_calibration_expected_authority.calibration_artifact_path
                ).absolute()
                if supplied != registered:
                    raise SchemaError(
                        "replay timeout artifact path differs from expected authority"
                    )
                self._timeout_calibration_evidence = (
                    load_authority_bound_timeout_calibration(
                        timeout_calibration_expected_authority
                    )
                )
                self._timeout_authority_bundle_validated = True
            self._timeout_binding = measured_timeout_binding(
                self._timeout_calibration_evidence
            )
        timeout_ms = self._timeout_binding["timeout_milliseconds"]
        self._startup_timeout = timeout_ms["broker_startup"] / 1000.0
        self._runtime_timeouts = {
            operation: timeout_ms[operation] / 1000.0
            for operation in (
                "runtime_reset",
                "runtime_observe",
                "runtime_execute",
                "runtime_terminal",
                "runtime_close",
            )
        }
        self._control_shutdown_timeout = (
            timeout_ms["control_shutdown"] / 1000.0
        )
        self._runtime_key = secrets.token_bytes(32)
        self._orchestration_key = secrets.token_bytes(32)
        self._control_key = secrets.token_bytes(32)
        self._session_id = secrets.token_hex(32)
        self._control_sequence = 0
        self._temp_root = Path(tempfile.mkdtemp(prefix="t2pb-", dir="/tmp"))
        self._endpoint = self._temp_root / "broker.sock"
        self._process: subprocess.Popen[str] | None = None
        self._receipt: ProcessBrokerReceipt | None = None
        self._launch_receipt_sha256: str | None = None
        self._cleanup_receipt: ProcessBrokerCleanupReceipt | None = None
        self._runtime_client_issued = False
        self._finalization_client_issued = False
        self._sealed_finalization_available = False
        self._start_attempted = False

    def __enter__(self) -> "ProcessIsolatedBroker":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    @property
    def receipt(self) -> ProcessBrokerReceipt:
        if self._receipt is None:
            raise RuntimeError("process broker has not started")
        return self._receipt

    @property
    def cleanup_receipt(self) -> ProcessBrokerCleanupReceipt:
        if self._cleanup_receipt is None:
            raise RuntimeError("process broker has not completed authenticated cleanup")
        return self._cleanup_receipt

    def runtime_client(self) -> ProcessIsolatedRuntimeClient:
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("process broker is not running")
        if self._runtime_client_issued:
            raise RuntimeError("process broker runtime capability may be issued once")
        self._runtime_client_issued = True
        return ProcessIsolatedRuntimeClient(
            endpoint=self._endpoint,
            authentication_key=self._runtime_key,
            session_id=self._session_id,
            policy_screenshot_root=self._policy_screenshot_root,
            timeout_seconds_by_operation=self._runtime_timeouts,
        )

    def finalization_client(self) -> ProcessIsolatedFinalizationClient:
        """Issue the distinct single-use post-decision capability once."""

        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("process broker is not running")
        if not self._sealed_finalization_available:
            raise RuntimeError(
                "process broker backend has no sealed finalization capability"
            )
        if self._finalization_client_issued:
            raise RuntimeError(
                "process broker finalization capability may be issued once"
            )
        self._finalization_client_issued = True
        # Final episode verification has not yet received its own independently
        # measured operation class.  Local engineering/replay evidence reuses
        # the larger of terminal/close bounds and the receipt explicitly marks
        # this as non-authorizing for a campaign.
        timeout_seconds = max(
            self._runtime_timeouts["runtime_terminal"],
            self._runtime_timeouts["runtime_close"],
        )
        return ProcessIsolatedFinalizationClient(
            endpoint=self._endpoint,
            authentication_key=self._orchestration_key,
            session_id=self._session_id,
            timeout_seconds=timeout_seconds,
        )

    def _source_rows(self) -> tuple[dict[str, str], ...]:
        backend_relative = self._backend_source.relative_to(self._repo).as_posix()
        rows = []
        for relative in dict.fromkeys(
            (
                *PROCESS_BROKER_SOURCE_PATHS,
                backend_relative,
                *self._backend_dependency_source_relatives,
            )
        ):
            source, canonical_relative = _validated_repository_python_source(
                self._repo,
                relative,
                label="process-broker source",
            )
            if canonical_relative != relative:
                raise SchemaError(
                    f"process-broker source path is not canonical: {relative}"
                )
            rows.append(
                {"relative_path": relative, "sha256": sha256_file(source)}
            )
            rebound, _ = _validated_repository_python_source(
                self._repo,
                relative,
                label="process-broker source",
            )
            if rebound != source:
                raise SchemaError(
                    f"process-broker source changed during hashing: {relative}"
                )
        return tuple(sorted(rows, key=lambda row: row["relative_path"]))

    def start(self) -> ProcessBrokerReceipt:
        if self._start_attempted:
            raise RuntimeError("process broker may start exactly once")
        self._start_attempted = True
        startup_deadline = monotonic() + self._startup_timeout
        try:
            with interrupt_after(
                remaining_deadline_seconds(startup_deadline),
                exception_factory=lambda: TimeoutError(
                    "process-broker absolute startup deadline expired"
                ),
            ):
                return self._start_with_deadline(startup_deadline)
        except BaseException:
            self.stop(force=True)
            raise

    def _start_with_deadline(
        self, startup_deadline: float
    ) -> ProcessBrokerReceipt:
        try:
            rows = self._source_rows()
            _verify_parent_imported_sources(self._repo, rows)
            _verify_optional_parent_backend_source(
                repository_root=self._repo,
                backend_entrypoint=self._backend_entrypoint,
                backend_source=self._backend_source,
                source_rows=rows,
            )
        except BaseException:
            self.stop(force=True)
            raise
        source_hashes = {row["relative_path"]: row["sha256"] for row in rows}
        startup_nonce = secrets.token_hex(32)
        config = {
            "repository_root": str(self._repo),
            "source_files": list(rows),
            "endpoint": str(self._endpoint),
            "runtime_key_b64": base64.b64encode(self._runtime_key).decode("ascii"),
            "orchestration_key_b64": base64.b64encode(
                self._orchestration_key
            ).decode("ascii"),
            "control_key_b64": base64.b64encode(self._control_key).decode("ascii"),
            "session_id": self._session_id,
            "startup_nonce": startup_nonce,
            "allowed_runtime_pid": os.getpid(),
            "allowed_uid": os.getuid(),
            "backend_entrypoint": self._backend_entrypoint,
            "backend_source_path": str(self._backend_source),
            "backend_source_sha256": source_hashes[
                self._backend_source.relative_to(self._repo).as_posix()
            ],
            "sealed_backend_config": self._sealed_backend_config,
            "sealed_backend_config_sha256": self._sealed_backend_config_sha256,
            "policy_screenshot_root": (
                str(self._policy_screenshot_root)
                if self._policy_screenshot_root is not None
                else None
            ),
            "policy_screenshot_root_identity_sha256": (
                self._policy_screenshot_root_identity_sha256
            ),
            "ipc_timeout_binding": self._timeout_binding,
            "ipc_timeout_binding_sha256": timeout_binding_sha256(
                self._timeout_binding
            ),
            "ipc_timeout_calibration_evidence": (
                self._timeout_calibration_evidence
            ),
            "execution_scope": self._execution_scope,
            "sealed_finalization_required": self._sealed_finalization_required,
        }
        if len(canonical_json_bytes(config)) > MAX_PROCESS_BROKER_MESSAGE_BYTES:
            self.stop(force=True)
            raise SchemaError(
                "process-broker launch configuration exceeds IPC frame bound"
            )
        launch_parent: socket.socket | None = None
        launch_child: socket.socket | None = None
        try:
            launch_parent, launch_child = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_STREAM
            )
            # This stdlib-only launcher verifies all registered source bytes
            # before importing package code. A fresh pycache prefix prevents an
            # ignored/stale repository __pycache__ from supplying bytecode.
            # Launch configuration and authenticated readiness use one private
            # framed socket. Backend stdout/stderr cannot impersonate readiness
            # and cannot fill an undrained pipe before readiness is published.
            # Retain the Popen object before its constructor can return. A
            # SIGALRM may interrupt POSIX Popen after fork while it is blocked
            # on the child exec-error pipe. The finally assignment preserves
            # ownership of any positive child PID so forced cleanup can always
            # terminate and reap it instead of orphaning the worker.
            process = subprocess.Popen.__new__(subprocess.Popen)
            try:
                subprocess.Popen.__init__(
                    process,
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-X",
                        f"pycache_prefix={self._temp_root / 'pycache'}",
                        "-c",
                        _PROCESS_BROKER_STDLIB_LAUNCHER,
                        str(self._repo / "src"),
                        str(launch_child.fileno()),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    cwd=self._repo,
                    pass_fds=(launch_child.fileno(),),
                )
            finally:
                launched_pid = getattr(process, "pid", None)
                if type(launched_pid) is int and launched_pid > 0:
                    self._process = process
            if self._process is not process:
                raise RuntimeError(
                    "process-broker launcher returned without owned child PID"
                )
            launch_child.close()
            send_frame(
                launch_parent,
                config,
                deadline_monotonic=startup_deadline,
            )
            readiness_envelope = receive_frame(
                launch_parent,
                deadline_monotonic=startup_deadline,
            )
            (
                evaluator_pid,
                self._sealed_finalization_available,
            ) = _validated_readiness_envelope(
                readiness_envelope,
                control_key=self._control_key,
                session_id=self._session_id,
                startup_nonce=startup_nonce,
                child_pid=process.pid,
                sealed_backend_config_sha256=self._sealed_backend_config_sha256,
                policy_screenshot_root_identity_sha256=(
                    self._policy_screenshot_root_identity_sha256
                ),
                ipc_timeout_binding_sha256=timeout_binding_sha256(
                    self._timeout_binding
                ),
                sealed_finalization_required=(
                    self._sealed_finalization_required
                ),
            )
        except BaseException:
            if launch_child is not None:
                launch_child.close()
            self.stop(force=True)
            raise
        finally:
            if launch_parent is not None:
                launch_parent.close()
            if launch_child is not None:
                launch_child.close()
        try:
            self._receipt = ProcessBrokerReceipt(
                schema_version=PROCESS_BROKER_RECEIPT_SCHEMA_VERSION,
                record_type="ProcessBrokerInnerSchemaCompatibilityReceipt",
                claim_scope=(
                    "LOCAL_DISTINCT_PROCESS_KEY_ENVELOPE_AND_INNER_SCHEMA_EVIDENCE_"
                    "NOT_VALUE_PROVENANCE_OR_DEPLOYMENT_AUTHORITY"
                ),
                status=(
                    "LOCAL_DISTINCT_PROCESS_ENVELOPE_AND_INNER_SCHEMA_"
                    "CONFORMANCE_PASS"
                ),
                execution_scope=self._execution_scope,
                runtime_pid=os.getpid(),
                sealed_evaluator_pid=evaluator_pid,
                process_ids_distinct=evaluator_pid != os.getpid(),
                transport="AF_UNIX_JSON_HMAC_SHA256_PEERCRED",
                protocol_version=PROCESS_BROKER_PROTOCOL_VERSION,
                peer_credentials_enforced=hasattr(socket, "SO_PEERCRED"),
                role_separated_authentication=(
                    len(
                        {
                            self._runtime_key,
                            self._orchestration_key,
                            self._control_key,
                        }
                    )
                    == 3
                ),
                outer_envelope_fields_exact=True,
                forbidden_named_keys_rejected_recursively=True,
                operation_specific_inner_schema_paths=RUNTIME_INNER_SCHEMA_PATHS,
                inner_schema_registry_version=(
                    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION
                ),
                inner_schema_registry_sha256=(
                    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256
                ),
                operation_specific_inner_schemas_registered=True,
                loaded_source_closure_enforced=True,
                single_episode_task_session_enforced=True,
                observation_stage_prior_action_bound=True,
                verifier_receipt_causal_binding_enforced=True,
                reset_operation_registered=True,
                executor_local_rejection_registration_enforced=True,
                policy_screenshot_transport_contract=(
                    POLICY_SCREENSHOT_TRANSPORT_CONTRACT
                ),
                policy_screenshot_root_identity_sha256=(
                    self._policy_screenshot_root_identity_sha256
                ),
                policy_screenshot_root_bound_per_session=True,
                canonical_wire_json_enforced=True,
                runtime_client_ambiguous_failure_poisoned=True,
                sealed_backend_config_hash_bound=True,
                ipc_timeout_binding=MappingProxyType(
                    {
                        **self._timeout_binding,
                        "timeout_milliseconds": MappingProxyType(
                            dict(
                                self._timeout_binding[
                                    "timeout_milliseconds"
                                ]
                            )
                        ),
                    }
                ),
                ipc_timeout_binding_sha256=timeout_binding_sha256(
                    self._timeout_binding
                ),
                measured_ipc_timeout_calibration_complete=(
                    self._timeout_binding["mode"] == MEASURED_TIMEOUT_MODE
                ),
                timeout_calibration_replay_only=(
                    self._execution_scope
                    == PROCESS_BROKER_MEASURED_REPLAY_SCOPE
                ),
                immutable_timeout_authority_bundle_validated=(
                    self._timeout_authority_bundle_validated
                ),
                external_timeout_authority_cross_binding_present=False,
                ipc_timeout_calibration_live_campaign_authority=False,
                runtime_value_provenance_attested=False,
                evaluator_operation_in_runtime_allowlist=False,
                sealed_finalization_capability_available=(
                    self._sealed_finalization_available
                ),
                sealed_finalization_required=self._sealed_finalization_required,
                sealed_finalization_operation_in_runtime_allowlist=False,
                child_owned_sealed_sink=(
                    self._sealed_finalization_available
                ),
                runtime_adapter_sealed_capability_free=True,
                separate_evidence_transport_present=False,
                runtime_terminal_returns_outer_sealed_signal=(
                    self._sealed_finalization_available
                ),
                sealed_finalization_returns_outer_sealed_signal=(
                    self._sealed_finalization_available
                ),
                sealed_finalization_timeout_campaign_authority=False,
                future_promotion_requirements=(
                    PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
                ),
                backend_entrypoint=self._backend_entrypoint,
                backend_entrypoint_sha256=hashlib.sha256(
                    self._backend_entrypoint.encode("utf-8")
                ).hexdigest(),
                backend_source_relative_path=self._backend_source.relative_to(
                    self._repo
                ).as_posix(),
                backend_source_sha256=source_hashes[
                    self._backend_source.relative_to(self._repo).as_posix()
                ],
                sealed_backend_config_sha256=self._sealed_backend_config_sha256,
                endpoint_identity_sha256=hashlib.sha256(
                    str(self._endpoint).encode("utf-8")
                ).hexdigest(),
                runtime_key_identity_sha256=hashlib.sha256(self._runtime_key).hexdigest(),
                orchestration_key_identity_sha256=hashlib.sha256(
                    self._orchestration_key
                ).hexdigest(),
                control_key_identity_sha256=hashlib.sha256(self._control_key).hexdigest(),
                source_files=tuple(
                    MappingProxyType(dict(row)) for row in rows
                ),
                source_set_sha256=sha256_json(list(rows)),
                external_deployment_authority=False,
            )
        except BaseException:
            self.stop(force=True)
            raise
        self._launch_receipt_sha256 = sha256_json(self._receipt.to_dict())
        return self._receipt

    def stop(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            shutil.rmtree(self._temp_root, ignore_errors=True)
            return
        graceful = process.poll() is None and not force
        shutdown_deadline = monotonic() + self._control_shutdown_timeout
        if graceful:
            sequence = self._control_sequence
            self._control_sequence += 1
            nonce = secrets.token_hex(32)
            body = {
                "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
                "role": "control",
                "session_id": self._session_id,
                "sequence": sequence,
                "nonce": nonce,
                "operation": CONTROL_OPERATION,
                "payload": {},
            }
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    set_socket_timeout_to_deadline(connection, shutdown_deadline)
                    connection.connect(str(self._endpoint))
                    send_frame(
                        connection,
                        authenticated_envelope(
                            body, authentication_key=self._control_key
                        ),
                        deadline_monotonic=shutdown_deadline,
                    )
                    _validate_shutdown_response(
                        receive_frame(
                            connection,
                            deadline_monotonic=shutdown_deadline,
                        ),
                        control_key=self._control_key,
                        session_id=self._session_id,
                        sequence=sequence,
                        nonce=nonce,
                    )
            except Exception:
                graceful = False
                if process.poll() is None:
                    process.terminate()
        elif process.poll() is None:
            # Startup/source failures have no authenticated lifecycle to
            # preserve; terminate immediately and use the separately
            # calibrated startup/reap bound below.
            process.terminate()
        if process.poll() is None and not force:
            try:
                process.wait(
                    timeout=remaining_deadline_seconds(shutdown_deadline)
                )
            except Exception:
                graceful = False
        if process.poll() is None:
            graceful = False
            process.kill()
            # Once the control lifecycle is exhausted, reaping a SIGKILLed
            # startup/worker process uses a separate calibrated setup bound;
            # no unmeasured five-second constant remains in evaluation code.
            force_reap_deadline = monotonic() + self._startup_timeout
            process.wait(timeout=remaining_deadline_seconds(force_reap_deadline))
        exit_code = int(process.returncode)
        evaluator_pid = int(process.pid)
        self._process = None
        shutil.rmtree(self._temp_root, ignore_errors=True)
        if (
            self._receipt is not None
            and self._launch_receipt_sha256 is not None
            and graceful
            and exit_code == 0
        ):
            self._cleanup_receipt = ProcessBrokerCleanupReceipt(
                schema_version=PROCESS_BROKER_CLEANUP_RECEIPT_SCHEMA_VERSION,
                record_type="ProcessIsolatedPageBrokerCleanupReceipt",
                claim_scope="LOCAL_ARCHITECTURE_EVIDENCE_NOT_DEPLOYMENT_AUTHORITY",
                status="LOCAL_CLEANUP_PASS",
                launch_receipt_sha256=self._launch_receipt_sha256,
                session_identity_sha256=hashlib.sha256(
                    self._session_id.encode("utf-8")
                ).hexdigest(),
                sealed_evaluator_pid=evaluator_pid,
                worker_exit_code=exit_code,
                graceful_authenticated_shutdown=True,
                endpoint_removed=not self._endpoint.exists(),
                temporary_directory_removed=not self._temp_root.exists(),
                source_set_sha256=self._receipt.source_set_sha256,
                external_deployment_authority=False,
            )

    @property
    def cleaned(self) -> bool:
        return self._process is None and not self._temp_root.exists()
