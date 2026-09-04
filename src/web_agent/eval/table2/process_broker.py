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
import selectors
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .common import SchemaError, sha256_file, sha256_json
from .process_broker_protocol import (
    ARBITRARY_RUNTIME_MAPPING_PATHS,
    PROCESS_BROKER_PROTOCOL_VERSION,
    PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS,
    authenticated_envelope,
    receive_frame,
    send_frame,
    verify_authenticated_envelope,
)
from .process_broker_runtime import ProcessIsolatedRuntimeClient

CONTROL_OPERATION = "sealed_control_shutdown"


PROCESS_BROKER_RECEIPT_SCHEMA_VERSION = "table2-process-page-broker-receipt-v2"
PROCESS_BROKER_CLEANUP_RECEIPT_SCHEMA_VERSION = (
    "table2-process-page-broker-cleanup-receipt-v1"
)
PROCESS_BROKER_SOURCE_PATHS = (
    "src/web_agent/__init__.py",
    "src/web_agent/eval/__init__.py",
    "src/web_agent/eval/table2/__init__.py",
    "src/web_agent/eval/table2/common.py",
    "src/web_agent/eval/table2/process_broker.py",
    "src/web_agent/eval/table2/process_broker_protocol.py",
    "src/web_agent/eval/table2/process_broker_runtime.py",
    "src/web_agent/eval/table2/process_broker_worker.py",
)


def _validated_readiness_envelope(
    value: object,
    *,
    control_key: bytes,
    session_id: str,
    startup_nonce: str,
    child_pid: int,
) -> int:
    ready = verify_authenticated_envelope(value, authentication_key=control_key)
    if set(ready) != {
        "protocol_version",
        "role",
        "session_id",
        "startup_nonce",
        "status",
        "evaluator_pid",
        "endpoint_mode",
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
        or type(evaluator_pid) is not int
        or evaluator_pid != child_pid
    ):
        raise RuntimeError("process-broker worker readiness failed")
    return evaluator_pid


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
        "web_agent.eval": "src/web_agent/eval/__init__.py",
        "web_agent.eval.table2": "src/web_agent/eval/table2/__init__.py",
        "web_agent.eval.table2.common": "src/web_agent/eval/table2/common.py",
        "web_agent.eval.table2.process_broker": (
            "src/web_agent/eval/table2/process_broker.py"
        ),
        "web_agent.eval.table2.process_broker_protocol": (
            "src/web_agent/eval/table2/process_broker_protocol.py"
        ),
        "web_agent.eval.table2.process_broker_runtime": (
            "src/web_agent/eval/table2/process_broker_runtime.py"
        ),
    }
    for module_name, relative in modules.items():
        module = sys.modules.get(module_name)
        loaded = Path(str(getattr(module, "__file__", ""))).resolve()
        registered = repository_root / relative
        if (
            loaded != registered
            or loaded.suffix != ".py"
            or expected.get(relative) != sha256_file(loaded)
        ):
            raise SchemaError(
                f"imported parent broker module identity differs: {module_name}"
            )


@dataclass(frozen=True, slots=True)
class ProcessBrokerReceipt:
    schema_version: str
    record_type: str
    claim_scope: str
    status: str
    runtime_pid: int
    sealed_evaluator_pid: int
    process_ids_distinct: bool
    transport: str
    protocol_version: str
    peer_credentials_enforced: bool
    role_separated_authentication: bool
    outer_envelope_fields_exact: bool
    forbidden_named_keys_rejected_recursively: bool
    arbitrary_nested_mapping_paths: tuple[str, ...]
    operation_specific_inner_schemas_registered: bool
    runtime_value_provenance_attested: bool
    evaluator_operation_in_runtime_allowlist: bool
    future_promotion_requirements: tuple[str, ...]
    backend_entrypoint: str
    backend_entrypoint_sha256: str
    backend_source_relative_path: str
    backend_source_sha256: str
    endpoint_identity_sha256: str
    runtime_key_identity_sha256: str
    control_key_identity_sha256: str
    source_files: tuple[dict[str, str], ...]
    source_set_sha256: str
    external_deployment_authority: bool

    def __post_init__(self) -> None:
        fixed = {
            "schema_version": PROCESS_BROKER_RECEIPT_SCHEMA_VERSION,
            "record_type": "ProcessBrokerEnvelopeCompatibilityReceipt",
            "claim_scope": (
                "LOCAL_DISTINCT_PROCESS_AND_KEY_ENVELOPE_EVIDENCE_"
                "NOT_VALUE_PROVENANCE_OR_DEPLOYMENT_AUTHORITY"
            ),
            "status": "LOCAL_DISTINCT_PROCESS_AND_ENVELOPE_CONFORMANCE_PASS",
            "process_ids_distinct": True,
            "transport": "AF_UNIX_JSON_HMAC_SHA256_PEERCRED",
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "peer_credentials_enforced": True,
            "role_separated_authentication": True,
            "outer_envelope_fields_exact": True,
            "forbidden_named_keys_rejected_recursively": True,
            "operation_specific_inner_schemas_registered": False,
            "runtime_value_provenance_attested": False,
            "evaluator_operation_in_runtime_allowlist": False,
            "external_deployment_authority": False,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise SchemaError(f"process-broker receipt {name} differs")
        if self.arbitrary_nested_mapping_paths != ARBITRARY_RUNTIME_MAPPING_PATHS:
            raise SchemaError(
                "process-broker receipt arbitrary nested mapping paths differ"
            )
        if self.future_promotion_requirements != (
            PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
        ):
            raise SchemaError(
                "process-broker receipt future promotion requirements differ"
            )
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

    def to_dict(self) -> dict[str, Any]:
        value = {item.name: getattr(self, item.name) for item in fields(self)}
        value["source_files"] = [dict(row) for row in self.source_files]
        value["arbitrary_nested_mapping_paths"] = list(
            self.arbitrary_nested_mapping_paths
        )
        value["future_promotion_requirements"] = list(
            self.future_promotion_requirements
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
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        self._repo = Path(repository_root).resolve()
        self._backend_entrypoint = backend_entrypoint
        self._backend_source = (self._repo / backend_source_relative_path).resolve()
        if (
            self._repo not in self._backend_source.parents
            or self._backend_source.is_symlink()
            or not self._backend_source.is_file()
        ):
            raise SchemaError("process-broker backend source is unsafe or missing")
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
        self._runtime_key = secrets.token_bytes(32)
        self._control_key = secrets.token_bytes(32)
        self._session_id = secrets.token_hex(32)
        self._control_sequence = 0
        self._temp_root = Path(tempfile.mkdtemp(prefix="t2pb-", dir="/tmp"))
        self._endpoint = self._temp_root / "broker.sock"
        self._process: subprocess.Popen[str] | None = None
        self._receipt: ProcessBrokerReceipt | None = None
        self._cleanup_receipt: ProcessBrokerCleanupReceipt | None = None
        self._runtime_client_issued = False
        self._start_attempted = False
        self._startup_timeout = startup_timeout_seconds

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
        )

    def _source_rows(self) -> tuple[dict[str, str], ...]:
        backend_relative = self._backend_source.relative_to(self._repo).as_posix()
        rows = []
        for relative in dict.fromkeys((*PROCESS_BROKER_SOURCE_PATHS, backend_relative)):
            source = self._repo / relative
            if source.is_symlink() or not source.is_file():
                raise SchemaError(f"process-broker source is missing: {relative}")
            rows.append({"relative_path": relative, "sha256": sha256_file(source)})
        return tuple(sorted(rows, key=lambda row: row["relative_path"]))

    def start(self) -> ProcessBrokerReceipt:
        if self._start_attempted:
            raise RuntimeError("process broker may start exactly once")
        self._start_attempted = True
        try:
            rows = self._source_rows()
            _verify_parent_imported_sources(self._repo, rows)
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
        }
        try:
            # This stdlib-only launcher verifies all registered source bytes
            # before importing package code. A fresh pycache prefix prevents an
            # ignored/stale repository __pycache__ from supplying bytecode.
            launcher = """
import hashlib,json,pathlib,sys
config=json.loads(sys.stdin.readline())
root=pathlib.Path(config["repository_root"]).resolve()
for row in config["source_files"]:
    path=(root/row["relative_path"]).resolve()
    if root not in path.parents or path.suffix != ".py":
        raise RuntimeError("unsafe process-broker source")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError("process-broker source changed before import")
sys.path.insert(0,sys.argv[1])
from web_agent.eval.table2.process_broker_worker import serve
raise SystemExit(serve(config))
"""
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-X",
                    f"pycache_prefix={self._temp_root / 'pycache'}",
                    "-c",
                    launcher,
                    str(self._repo / "src"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._repo,
            )
            self._process = process
            assert process.stdin is not None
            process.stdin.write(json.dumps(config, sort_keys=True) + "\n")
            process.stdin.flush()
            process.stdin.close()
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            try:
                selector.register(process.stdout, selectors.EVENT_READ)
                if not selector.select(self._startup_timeout):
                    raise RuntimeError("process-broker worker did not become ready")
                line = process.stdout.readline()
            finally:
                selector.close()
            try:
                readiness_envelope = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "process-broker worker readiness is malformed"
                ) from exc
            evaluator_pid = _validated_readiness_envelope(
                readiness_envelope,
                control_key=self._control_key,
                session_id=self._session_id,
                startup_nonce=startup_nonce,
                child_pid=process.pid,
            )
        except BaseException:
            self.stop(force=True)
            raise
        try:
            self._receipt = ProcessBrokerReceipt(
                schema_version=PROCESS_BROKER_RECEIPT_SCHEMA_VERSION,
                record_type="ProcessBrokerEnvelopeCompatibilityReceipt",
                claim_scope=(
                    "LOCAL_DISTINCT_PROCESS_AND_KEY_ENVELOPE_EVIDENCE_"
                    "NOT_VALUE_PROVENANCE_OR_DEPLOYMENT_AUTHORITY"
                ),
                status="LOCAL_DISTINCT_PROCESS_AND_ENVELOPE_CONFORMANCE_PASS",
                runtime_pid=os.getpid(),
                sealed_evaluator_pid=evaluator_pid,
                process_ids_distinct=evaluator_pid != os.getpid(),
                transport="AF_UNIX_JSON_HMAC_SHA256_PEERCRED",
                protocol_version=PROCESS_BROKER_PROTOCOL_VERSION,
                peer_credentials_enforced=hasattr(socket, "SO_PEERCRED"),
                role_separated_authentication=self._runtime_key != self._control_key,
                outer_envelope_fields_exact=True,
                forbidden_named_keys_rejected_recursively=True,
                arbitrary_nested_mapping_paths=ARBITRARY_RUNTIME_MAPPING_PATHS,
                operation_specific_inner_schemas_registered=False,
                runtime_value_provenance_attested=False,
                evaluator_operation_in_runtime_allowlist=False,
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
                endpoint_identity_sha256=hashlib.sha256(
                    str(self._endpoint).encode("utf-8")
                ).hexdigest(),
                runtime_key_identity_sha256=hashlib.sha256(self._runtime_key).hexdigest(),
                control_key_identity_sha256=hashlib.sha256(self._control_key).hexdigest(),
                source_files=rows,
                source_set_sha256=sha256_json(list(rows)),
                external_deployment_authority=False,
            )
        except BaseException:
            self.stop(force=True)
            raise
        return self._receipt

    def stop(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            shutil.rmtree(self._temp_root, ignore_errors=True)
            return
        graceful = process.poll() is None and not force
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
                    connection.settimeout(3.0)
                    connection.connect(str(self._endpoint))
                    send_frame(
                        connection,
                        authenticated_envelope(
                            body, authentication_key=self._control_key
                        ),
                    )
                    _validate_shutdown_response(
                        receive_frame(connection),
                        control_key=self._control_key,
                        session_id=self._session_id,
                        sequence=sequence,
                        nonce=nonce,
                    )
            except Exception:
                graceful = False
                process.terminate()
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            graceful = False
            process.kill()
            process.wait(timeout=5.0)
        exit_code = int(process.returncode)
        evaluator_pid = int(process.pid)
        self._process = None
        shutil.rmtree(self._temp_root, ignore_errors=True)
        if self._receipt is not None and graceful and exit_code == 0:
            self._cleanup_receipt = ProcessBrokerCleanupReceipt(
                schema_version=PROCESS_BROKER_CLEANUP_RECEIPT_SCHEMA_VERSION,
                record_type="ProcessIsolatedPageBrokerCleanupReceipt",
                claim_scope="LOCAL_ARCHITECTURE_EVIDENCE_NOT_DEPLOYMENT_AUTHORITY",
                status="LOCAL_CLEANUP_PASS",
                launch_receipt_sha256=sha256_json(self._receipt.to_dict()),
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
