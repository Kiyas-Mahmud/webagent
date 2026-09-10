"""Credential-free, outcome-blind process-broker timeout collection.

This module is deliberately an evidence collector, not a campaign launcher and
not an authority issuer.  It consumes an externally frozen safe-probe manifest
whose action and failure identities have already been approved.  It never
selects a task, action, recovery, failure injection, timeout, or outcome.  A
source-attested harness performs only the requested probe operation; this
module owns every ``time.monotonic_ns`` read and accepts no caller-supplied
timestamps.

The collector has no reward, evaluator, oracle, paper-metric, model, or
credential interface.  Harness operations return no payload.  The sole data
returned by the harness are pre/post reset-state SHA-256 fingerprints used by
the registered non-persistence receipt.

Successful output is the existing v2 five-artifact timeout bundle plus a local
expected-identity record and a compact collection manifest.  Every record is
non-authorizing.  The output directory is reserved exactly once and is made
read-only on success or failure; a second attempt at the same path is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from .common import SchemaError, canonical_json_bytes, sha256_file, sha256_json
from .process_broker_timeout import (
    ACTION_PHASE_CLASSES,
    CALIBRATION_OPERATION_CLASSES,
    HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION,
    MEASUREMENT_SOURCE_RECEIPT_SCHEMA_VERSION,
    MONOTONIC_CLOCK_ID,
    NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
    RECOVERY_PHASE_CLASSES,
    TIMEOUT_CALIBRATION_FILENAME,
    TIMEOUT_CALIBRATION_CLAIM_SCOPE,
    TIMEOUT_CALIBRATION_STATUS,
    TIMEOUT_COLLECTION_FAILURE_FILENAME,
    TIMEOUT_COLLECTION_MANIFEST_FILENAME,
    TIMEOUT_COLLECTION_MANIFEST_SCHEMA_VERSION,
    TIMEOUT_HARNESS_SOURCE_RECEIPT_FILENAME,
    TIMEOUT_MEASUREMENT_SOURCE_RECEIPT_FILENAME,
    TIMEOUT_NON_PERSISTENCE_RECEIPT_FILENAME,
    TIMEOUT_SAFE_PROBE_FILENAME,
    ProcessBrokerTimeoutExpectedAuthority,
    _load_immutable_strict_json,
    _require_sha256,
    _validate_browsergym_configuration,
    _validate_budgets,
    _validate_harness_source_receipt,
    _validate_host_identity,
    _validate_measurement_source_receipt,
    _validate_non_persistence_receipt,
    _validate_safe_probe_manifest,
    _validate_task_registry,
    build_process_broker_timeout_calibration,
    load_authority_bound_timeout_calibration,
)


TIMEOUT_COLLECTOR_INPUT_SCHEMA_VERSION = (
    "table2-process-broker-timeout-collector-input-v1"
)
TIMEOUT_COLLECTOR_INPUT_STATUS = "EXTERNALLY_FROZEN_NON_AUTHORIZING"
TIMEOUT_COLLECTOR_MANIFEST_SCHEMA_VERSION = (
    TIMEOUT_COLLECTION_MANIFEST_SCHEMA_VERSION
)
TIMEOUT_COLLECTOR_FAILURE_SCHEMA_VERSION = (
    "table2-process-broker-timeout-collection-failure-v1"
)
TIMEOUT_COLLECTOR_ENTRYPOINT = (
    "web_agent.eval.table2.process_broker_timeout_collector:"
    "collect_process_broker_timeout_calibration"
)
TIMEOUT_COLLECTOR_SOURCE_RELATIVE_PATH = (
    "src/web_agent/eval/table2/process_broker_timeout_collector.py"
)

CALIBRATION_FILENAME = TIMEOUT_CALIBRATION_FILENAME
SAFE_PROBE_FILENAME = TIMEOUT_SAFE_PROBE_FILENAME
HARNESS_SOURCE_RECEIPT_FILENAME = TIMEOUT_HARNESS_SOURCE_RECEIPT_FILENAME
MEASUREMENT_SOURCE_RECEIPT_FILENAME = TIMEOUT_MEASUREMENT_SOURCE_RECEIPT_FILENAME
NON_PERSISTENCE_RECEIPT_FILENAME = TIMEOUT_NON_PERSISTENCE_RECEIPT_FILENAME
LOCAL_EXPECTED_AUTHORITY_FILENAME = "local_expected_authority.json"
COLLECTION_MANIFEST_FILENAME = TIMEOUT_COLLECTION_MANIFEST_FILENAME
COLLECTION_FAILURE_FILENAME = TIMEOUT_COLLECTION_FAILURE_FILENAME
STAGING_DIRECTORY_NAME = ".staging"

CORE_ARTIFACT_FILENAMES = frozenset(
    {
        CALIBRATION_FILENAME,
        SAFE_PROBE_FILENAME,
        HARNESS_SOURCE_RECEIPT_FILENAME,
        MEASUREMENT_SOURCE_RECEIPT_FILENAME,
        NON_PERSISTENCE_RECEIPT_FILENAME,
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_TOPOLOGIES = frozenset({"SINGLE_DGX_HOST", "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"})
_NESTED_SETTLE = {
    "runtime_reset": "page_settle_reset",
    "runtime_execute_action": "page_settle_post_action",
    "runtime_execute_recovery": "page_settle_post_recovery",
}
_COLLECTOR_INPUT_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "claim_scope",
    "paper_table_status",
    "production_dispatch_authorized",
    "live_campaign_authority",
    "collected_before_campaign_outcomes",
    "reward_interface_available_to_collector",
    "evaluator_interface_available_to_collector",
    "oracle_interface_available_to_collector",
    "paper_metric_interface_available_to_collector",
    "credentials_available_to_collector",
    "safe_probe_manifest_content_sha256",
    "measurement_harness_source_receipt_content_sha256",
    "harness_factory_entrypoint",
    "deployment_topology",
    "deployment_preflight_content_sha256",
    "semantic_dependency_lock_sha256",
    "service_url_map_content_sha256",
    "browser_host_identity",
    "browsergym_runtime_configuration",
    "task_registry",
    "campaign_budgets",
}


@dataclass(frozen=True, slots=True)
class FrozenProbeOperation:
    """One operation identity selected exclusively by the frozen manifest."""

    task_order: int
    task_id: str
    operation_class: str
    phase: str | None
    action_type: str | None
    probe_action_sha256: str | None
    failure_injection_sha256: str | None
    safety_receipt_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_order": self.task_order,
            "task_id": self.task_id,
            "operation_class": self.operation_class,
            "phase": self.phase,
            "action_type": self.action_type,
            "probe_action_sha256": self.probe_action_sha256,
            "failure_injection_sha256": self.failure_injection_sha256,
            "safety_receipt_sha256": self.safety_receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenResetFingerprintRequest:
    """Narrow request for one reset-and-fingerprint non-persistence check."""

    task_order: int
    task_id: str
    checkpoint: str
    failure_injection_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenProbeCleanupRequest:
    """Best-effort cleanup identity used only after collector failure."""

    task_order: int
    task_id: str
    failed_operation_class: str | None


@runtime_checkable
class ProcessBrokerTimeoutProbeHarness(Protocol):
    """Minimal live harness capability accepted by the collector.

    ``run_registered_operation`` must invoke ``nested_operation`` exactly once
    when it is non-``None`` and must return ``None``.  It may not return browser
    state, scores, rewards, labels, or evaluator output.
    """

    def run_registered_operation(
        self,
        request: FrozenProbeOperation,
        nested_operation: Callable[[], None] | None,
    ) -> None: ...

    def reset_and_fingerprint(
        self, request: FrozenResetFingerprintRequest
    ) -> str: ...

    def cleanup_after_collection_failure(
        self, request: FrozenProbeCleanupRequest
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _ArtifactSnapshot:
    path: Path
    identity: tuple[int, int, int, int, int, int, int, int, int]
    raw_sha256: str
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: Path
    identity: tuple[int, int, int, int, int, int, int, int, int]
    sha256: str


def _exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise SchemaError(f"{label} fields differ from the registered closure")


def _detach_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    try:
        detached = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SchemaError(f"{label} must be canonical JSON") from exc
    if not isinstance(detached, dict):  # pragma: no cover - Mapping encodes object
        raise SchemaError(f"{label} must be an object")
    return detached


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _absolute_path(value: str | Path, label: str) -> Path:
    raw = Path(os.fspath(value))
    if not raw.is_absolute():
        raise SchemaError(f"{label} must be an absolute path")
    return Path(os.path.abspath(raw))


def _assert_no_symlink_ancestry(path: Path, label: str) -> None:
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current = current / component
            if stat.S_ISLNK(current.lstat().st_mode):
                raise SchemaError(f"{label} must not use symlink ancestry")
    except OSError as exc:
        raise SchemaError(f"{label} ancestry is absent") from exc


def _load_frozen_json(
    path: str | Path,
    *,
    expected_canonical_sha256: str,
    label: str,
) -> tuple[dict[str, Any], _ArtifactSnapshot]:
    source = _absolute_path(path, label)
    expected = _require_sha256(expected_canonical_sha256, f"expected {label}")
    value = _load_immutable_strict_json(source, label=label)
    canonical_sha = sha256_json(value)
    if canonical_sha != expected:
        raise SchemaError(f"{label} content differs from frozen identity")
    metadata = source.lstat()
    snapshot = _ArtifactSnapshot(
        path=source,
        identity=_file_identity(metadata),
        raw_sha256=sha256_file(source),
        canonical_sha256=canonical_sha,
    )
    return value, snapshot


def _assert_frozen_json_unchanged(snapshot: _ArtifactSnapshot, label: str) -> None:
    try:
        metadata = snapshot.path.lstat()
    except OSError as exc:
        raise SchemaError(f"{label} disappeared during collection") from exc
    if (
        _file_identity(metadata) != snapshot.identity
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
        or sha256_file(snapshot.path) != snapshot.raw_sha256
    ):
        raise SchemaError(f"{label} changed during collection")
    value = _load_immutable_strict_json(snapshot.path, label=label)
    if sha256_json(value) != snapshot.canonical_sha256:
        raise SchemaError(f"{label} canonical content changed during collection")


def validate_timeout_collector_input(value: object) -> dict[str, Any]:
    """Validate one frozen, explicitly non-authorizing collector input."""

    payload = _detach_mapping(value, "timeout collector input")
    _exact_fields(payload, _COLLECTOR_INPUT_FIELDS, "timeout collector input")
    fixed = {
        "schema_version": TIMEOUT_COLLECTOR_INPUT_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutCollectorInput",
        "status": TIMEOUT_COLLECTOR_INPUT_STATUS,
        "claim_scope": (
            "PRE_CAMPAIGN_SAFE_PROBE_INPUT_NOT_DEPLOYMENT_OR_CAMPAIGN_AUTHORITY"
        ),
        "paper_table_status": "N/R",
        "production_dispatch_authorized": False,
        "live_campaign_authority": False,
        "collected_before_campaign_outcomes": True,
        "reward_interface_available_to_collector": False,
        "evaluator_interface_available_to_collector": False,
        "oracle_interface_available_to_collector": False,
        "paper_metric_interface_available_to_collector": False,
        "credentials_available_to_collector": False,
    }
    for field, expected in fixed.items():
        if type(payload.get(field)) is not type(expected) or payload.get(field) != expected:
            raise SchemaError(f"timeout collector input {field} differs")
    for field in (
        "safe_probe_manifest_content_sha256",
        "measurement_harness_source_receipt_content_sha256",
        "deployment_preflight_content_sha256",
        "semantic_dependency_lock_sha256",
        "service_url_map_content_sha256",
    ):
        _require_sha256(payload.get(field), f"timeout collector input {field}")
    entrypoint = payload.get("harness_factory_entrypoint")
    if type(entrypoint) is not str or _ENTRYPOINT_RE.fullmatch(entrypoint) is None:
        raise SchemaError("timeout collector harness factory entrypoint is malformed")
    if payload.get("deployment_topology") not in _TOPOLOGIES:
        raise SchemaError("timeout collector deployment topology is unregistered")
    payload["browser_host_identity"] = _validate_host_identity(
        payload.get("browser_host_identity")
    )
    payload["browsergym_runtime_configuration"] = (
        _validate_browsergym_configuration(
            payload.get("browsergym_runtime_configuration")
        )
    )
    registry, _task_ids = _validate_task_registry(payload.get("task_registry"))
    payload["task_registry"] = registry
    payload["campaign_budgets"] = _validate_budgets(
        payload.get("campaign_budgets")
    )
    return payload


def load_frozen_timeout_collector_input(
    path: str | Path, *, expected_content_sha256: str
) -> tuple[dict[str, Any], _ArtifactSnapshot]:
    """Load one immutable collector input and bind its external digest."""

    value, snapshot = _load_frozen_json(
        path,
        expected_canonical_sha256=expected_content_sha256,
        label="timeout collector input",
    )
    return validate_timeout_collector_input(value), snapshot


def _safe_repository_source(
    repository_root: Path, relative_path: str, *, label: str
) -> Path:
    if (
        not relative_path
        or relative_path != relative_path.strip()
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise SchemaError(f"{label} path is not canonical")
    lexical = Path(relative_path)
    if (
        lexical.is_absolute()
        or lexical.suffix != ".py"
        or lexical.as_posix() != relative_path
        or "." in lexical.parts
        or ".." in lexical.parts
    ):
        raise SchemaError(f"{label} path is not canonical")
    candidate = repository_root
    try:
        for part in lexical.parts:
            candidate = candidate / part
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise SchemaError(f"{label} path contains a symlink")
    except OSError as exc:
        raise SchemaError(f"{label} source is absent") from exc
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SchemaError(f"{label} must be a single-link regular source")
    try:
        candidate.resolve(strict=True).relative_to(repository_root)
    except ValueError as exc:
        raise SchemaError(f"{label} escaped the repository") from exc
    return candidate


def _snapshot_harness_sources(
    repository_root: Path,
    receipt: Mapping[str, Any],
) -> tuple[dict[str, _SourceSnapshot], dict[str, str]]:
    snapshots: dict[str, _SourceSnapshot] = {}
    registered: dict[str, str] = {}
    for row in receipt["source_files"]:
        relative = str(row["relative_path"])
        source = _safe_repository_source(
            repository_root, relative, label="timeout harness"
        )
        before = source.lstat()
        digest = sha256_file(source)
        after = source.lstat()
        if _file_identity(before) != _file_identity(after):
            raise SchemaError(f"timeout harness source changed while hashing: {relative}")
        if digest != row["sha256"]:
            raise SchemaError(f"timeout harness source differs: {relative}")
        snapshots[relative] = _SourceSnapshot(
            path=source,
            identity=_file_identity(after),
            sha256=digest,
        )
        registered[relative] = digest
    mandatory = {
        TIMEOUT_COLLECTOR_SOURCE_RELATIVE_PATH,
        "src/web_agent/eval/table2/common.py",
        "src/web_agent/eval/table2/process_broker_timeout.py",
    }
    if not mandatory.issubset(registered):
        raise SchemaError("timeout harness source closure omits collector dependencies")
    for relative in mandatory:
        source = _safe_repository_source(
            repository_root, relative, label="timeout collector dependency"
        )
        if registered[relative] != sha256_file(source):
            raise SchemaError("timeout collector dependency source differs")
    return snapshots, registered


def _assert_sources_unchanged(
    snapshots: Mapping[str, _SourceSnapshot],
) -> None:
    for relative, snapshot in snapshots.items():
        try:
            metadata = snapshot.path.lstat()
        except OSError as exc:
            raise SchemaError(
                f"timeout harness source disappeared during collection: {relative}"
            ) from exc
        if (
            _file_identity(metadata) != snapshot.identity
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or sha256_file(snapshot.path) != snapshot.sha256
        ):
            raise SchemaError(
                f"timeout harness source changed during collection: {relative}"
            )


def _module_relative_source(
    module: ModuleType, *, repository_root: Path, registered: Mapping[str, str]
) -> str:
    raw = getattr(module, "__file__", None)
    if type(raw) is not str or not raw:
        raise SchemaError("timeout harness module has no source identity")
    source = Path(raw)
    if source.suffix == ".pyc":
        source = Path(inspect.getsourcefile(module) or "")
    try:
        relative = source.resolve(strict=True).relative_to(repository_root).as_posix()
    except (OSError, ValueError) as exc:
        raise SchemaError("timeout harness module escaped source closure") from exc
    if relative not in registered or sha256_file(source) != registered[relative]:
        raise SchemaError("timeout harness module is outside frozen source closure")
    return relative


def _load_harness(
    *,
    entrypoint: str,
    repository_root: Path,
    registered_sources: Mapping[str, str],
) -> ProcessBrokerTimeoutProbeHarness:
    module_name, attribute_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    _module_relative_source(
        module, repository_root=repository_root, registered=registered_sources
    )
    factory = getattr(module, attribute_name, None)
    if (
        not callable(factory)
        or getattr(factory, "__module__", None) != module_name
        or getattr(factory, "__name__", None) != attribute_name
    ):
        raise SchemaError("timeout harness factory identity differs")
    harness = factory()
    if not isinstance(harness, ProcessBrokerTimeoutProbeHarness):
        raise SchemaError("timeout harness lacks the registered narrow interface")
    harness_module = sys.modules.get(type(harness).__module__)
    if not isinstance(harness_module, ModuleType):
        raise SchemaError("timeout harness implementation module is absent")
    _module_relative_source(
        harness_module,
        repository_root=repository_root,
        registered=registered_sources,
    )
    return harness


def _operation_request(
    *,
    task_order: int,
    task_id: str,
    operation_class: str,
    ordinary_probe: Mapping[str, Any],
    recovery_probe: Mapping[str, Any],
    failure_probe: Mapping[str, Any],
) -> FrozenProbeOperation:
    if operation_class in ACTION_PHASE_CLASSES:
        probe = ordinary_probe
        phase = "ordinary"
        failure_sha = None
    elif operation_class in RECOVERY_PHASE_CLASSES:
        probe = recovery_probe
        phase = "recovery"
        failure_sha = None
    elif operation_class == "control_shutdown_after_failed_runtime":
        probe = failure_probe
        phase = None
        failure_sha = str(probe["failure_injection_sha256"])
    else:
        probe = None
        phase = None
        failure_sha = None
    return FrozenProbeOperation(
        task_order=task_order,
        task_id=task_id,
        operation_class=operation_class,
        phase=phase,
        action_type=(str(probe["action_type"]) if phase is not None else None),
        probe_action_sha256=(
            str(probe["probe_action_sha256"]) if phase is not None else None
        ),
        failure_injection_sha256=failure_sha,
        safety_receipt_sha256=(
            sha256_json(probe["safety_approval"]) if probe is not None else None
        ),
    )


def _measurement_row(
    request: FrozenProbeOperation, started: int, ended: int
) -> dict[str, Any]:
    if (
        type(started) is not int
        or type(ended) is not int
        or started < 0
        or ended <= started
    ):
        raise SchemaError(
            "timeout collector did not observe a positive monotonic_ns interval"
        )
    return {
        "task_order": request.task_order,
        "task_id": request.task_id,
        "operation_class": request.operation_class,
        "started_monotonic_ns": started,
        "ended_monotonic_ns": ended,
        "elapsed_ns": ended - started,
        "action_type": request.action_type,
        "probe_action_sha256": request.probe_action_sha256,
        "failure_injection_sha256": request.failure_injection_sha256,
        "safety_receipt_sha256": request.safety_receipt_sha256,
    }


def _time_simple_operation(
    harness: ProcessBrokerTimeoutProbeHarness,
    request: FrozenProbeOperation,
) -> dict[str, Any]:
    started = time.monotonic_ns()
    result = harness.run_registered_operation(request, None)
    ended = time.monotonic_ns()
    if result is not None:
        raise SchemaError("timeout harness operation returned forbidden data")
    return _measurement_row(request, started, ended)


def _time_operation_with_nested_settle(
    harness: ProcessBrokerTimeoutProbeHarness,
    outer: FrozenProbeOperation,
    nested: FrozenProbeOperation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nested_rows: list[dict[str, Any]] = []
    outer_active = False

    def measure_nested() -> None:
        if not outer_active or nested_rows:
            raise SchemaError(
                "timeout harness must invoke the registered settle exactly once "
                "in its outer operation"
            )
        nested_rows.append(_time_simple_operation(harness, nested))

    started = time.monotonic_ns()
    outer_active = True
    try:
        result = harness.run_registered_operation(outer, measure_nested)
    finally:
        outer_active = False
    ended = time.monotonic_ns()
    if result is not None:
        raise SchemaError("timeout harness operation returned forbidden data")
    if len(nested_rows) != 1:
        raise SchemaError(
            "timeout harness omitted the registered nested settle measurement"
        )
    return _measurement_row(outer, started, ended), nested_rows[0]


def _fingerprint(
    harness: ProcessBrokerTimeoutProbeHarness,
    request: FrozenResetFingerprintRequest,
) -> str:
    value = harness.reset_and_fingerprint(request)
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SchemaError(
            "timeout harness reset fingerprint must be one lowercase SHA-256 digest"
        )
    return value


def _collect_measurements(
    *,
    harness: ProcessBrokerTimeoutProbeHarness,
    safe_manifest: Mapping[str, Any],
    progress: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    non_persistence_entries: list[dict[str, Any]] = []
    task_ids = list(safe_manifest["ordered_task_ids"])
    ordinary_rows = list(safe_manifest["ordinary_probes"])
    recovery_rows = list(safe_manifest["recovery_probes"])
    failure_rows = list(safe_manifest["failed_runtime_cleanup_probes"])

    for task_order, task_id in enumerate(task_ids):
        progress.update(task_order=task_order, task_id=task_id, operation_class=None)
        ordinary = ordinary_rows[task_order]
        recovery = recovery_rows[task_order]
        failure = failure_rows[task_order]
        failure_sha = str(failure["failure_injection_sha256"])
        pre = _fingerprint(
            harness,
            FrozenResetFingerprintRequest(
                task_order=task_order,
                task_id=task_id,
                checkpoint="pre_probe_reset",
                failure_injection_sha256=failure_sha,
            ),
        )

        by_operation: dict[str, dict[str, Any]] = {}
        skip_nested: set[str] = set()
        for operation_class in CALIBRATION_OPERATION_CLASSES:
            if operation_class in skip_nested:
                continue
            progress["operation_class"] = operation_class
            request = _operation_request(
                task_order=task_order,
                task_id=task_id,
                operation_class=operation_class,
                ordinary_probe=ordinary,
                recovery_probe=recovery,
                failure_probe=failure,
            )
            nested_class = _NESTED_SETTLE.get(operation_class)
            if nested_class is None:
                by_operation[operation_class] = _time_simple_operation(
                    harness, request
                )
                continue
            nested_request = _operation_request(
                task_order=task_order,
                task_id=task_id,
                operation_class=nested_class,
                ordinary_probe=ordinary,
                recovery_probe=recovery,
                failure_probe=failure,
            )
            outer_row, nested_row = _time_operation_with_nested_settle(
                harness, request, nested_request
            )
            by_operation[operation_class] = outer_row
            by_operation[nested_class] = nested_row
            skip_nested.add(nested_class)

        if set(by_operation) != set(CALIBRATION_OPERATION_CLASSES):
            raise SchemaError("timeout collector operation coverage differs")
        rows.extend(by_operation[name] for name in CALIBRATION_OPERATION_CLASSES)
        progress["operation_class"] = "post_probe_reset_fingerprint"
        post = _fingerprint(
            harness,
            FrozenResetFingerprintRequest(
                task_order=task_order,
                task_id=task_id,
                checkpoint="post_probe_reset",
                failure_injection_sha256=failure_sha,
            ),
        )
        if pre != post:
            raise SchemaError(
                "timeout safe probe changed the reset-state fingerprint"
            )
        non_persistence_entries.append(
            {
                "task_order": task_order,
                "task_id": task_id,
                "failure_injection_sha256": failure_sha,
                "pre_probe_reset_state_sha256": pre,
                "post_probe_reset_state_sha256": post,
                "reset_state_equivalent": True,
                "persistent_mutation_detected": False,
            }
        )
    progress.update(task_order=None, task_id=None, operation_class=None)
    return rows, non_persistence_entries


def _write_new_readonly_json(path: Path, value: object) -> str:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        total = 0
        while total < len(payload):
            written = os.write(descriptor, payload[total:])
            if written <= 0:  # pragma: no cover - regular filesystem failure
                raise OSError("short write while sealing timeout artifact")
            total += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return sha256_json(value)


def _reserve_output_directory(path: str | Path) -> Path:
    output = _absolute_path(path, "timeout collector output directory")
    parent = output.parent
    _assert_no_symlink_ancestry(parent, "timeout collector output parent")
    if not parent.is_dir() or parent.is_symlink():
        raise SchemaError("timeout collector output parent must be an existing real directory")
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as exc:
        raise SchemaError(
            "timeout collector refuses rerun or overwrite at an existing output path"
        ) from exc
    return output


def _seal_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.chmod(path, 0o555)


def _record_failed_attempt(
    output: Path,
    *,
    staging: Path,
    progress: Mapping[str, Any],
    safe_probe_manifest_sha256: str,
    harness_source_receipt_sha256: str,
    harness_factory_entrypoint: str,
    collection_error_class: str,
    cleanup_attempted: bool,
    cleanup_succeeded: bool,
    cleanup_error_class: str | None,
) -> None:
    failure = {
        "schema_version": TIMEOUT_COLLECTOR_FAILURE_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutCollectionFailure",
        "status": "COLLECTION_FAILED_NO_CALIBRATION_AUTHORITY",
        "claim_scope": "FAILED_PRE_CAMPAIGN_INFRASTRUCTURE_ATTEMPT_ONLY",
        "paper_table_status": "N/R",
        "production_dispatch_authorized": False,
        "live_campaign_authority": False,
        "rerun_at_same_output_path_permitted": False,
        "failed_task_order": progress.get("task_order"),
        "failed_task_id": progress.get("task_id"),
        "failed_operation_class": progress.get("operation_class"),
        "collection_error_class": collection_error_class,
        "cleanup_attempted": cleanup_attempted,
        "cleanup_succeeded": cleanup_succeeded,
        "cleanup_error_class": cleanup_error_class,
        "safe_probe_manifest_sha256": safe_probe_manifest_sha256,
        "measurement_harness_source_receipt_sha256": (
            harness_source_receipt_sha256
        ),
        "harness_factory_entrypoint": harness_factory_entrypoint,
        "harness_factory_entrypoint_sha256": hashlib.sha256(
            harness_factory_entrypoint.encode("utf-8")
        ).hexdigest(),
    }
    errors: list[BaseException] = []
    for directory in (staging, output):
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            # A writable directory and/or the failure marker independently
            # prevent the authority loader from accepting a staged success.
            os.chmod(directory, 0o700)
            _write_new_readonly_json(
                directory / COLLECTION_FAILURE_FILENAME, failure
            )
            _seal_directory(directory)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def _measurement_source_receipt(
    *,
    evidence: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    harness_source_set_sha256: str,
    safe_probe_manifest_sha256: str,
    non_persistence_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": MEASUREMENT_SOURCE_RECEIPT_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutMeasurementSourceReceipt",
        "status": "LIVE_MONOTONIC_MEASUREMENTS_CAPTURED_NON_AUTHORIZING",
        "measurement_clock": MONOTONIC_CLOCK_ID,
        "measurement_harness_source_set_sha256": harness_source_set_sha256,
        "safe_probe_manifest_sha256": safe_probe_manifest_sha256,
        "non_persistence_audit_receipt_sha256": non_persistence_receipt_sha256,
        "task_manifest_sha256": evidence["task_registry"]["manifest_sha256"],
        "task_set_sha256": evidence["task_registry"]["task_set_sha256"],
        "deployment_topology": evidence["deployment_topology"],
        "deployment_preflight_content_sha256": evidence[
            "deployment_preflight_content_sha256"
        ],
        "semantic_dependency_lock_sha256": evidence[
            "semantic_dependency_lock_sha256"
        ],
        "service_url_map_content_sha256": evidence[
            "service_url_map_content_sha256"
        ],
        "browser_host_identity_sha256": evidence["browser_host_identity_sha256"],
        "browsergym_runtime_configuration_sha256": evidence[
            "browsergym_runtime_configuration_sha256"
        ],
        "campaign_budgets_sha256": evidence["campaign_budgets_sha256"],
        "operation_classes": list(CALIBRATION_OPERATION_CLASSES),
        "measurement_count": len(measurements),
        "measurements_sha256": sha256_json(measurements),
        "collected_before_campaign_outcomes": True,
        "reward_read": False,
        "evaluator_output_read": False,
        "oracle_label_read": False,
        "paper_metric_read": False,
    }


def collect_process_broker_timeout_calibration(
    *,
    repository_root: str | Path,
    output_directory: str | Path,
    safe_probe_manifest_path: str | Path,
    expected_safe_probe_manifest_sha256: str,
    measurement_harness_source_receipt_path: str | Path,
    expected_measurement_harness_source_receipt_sha256: str,
    harness_factory_entrypoint: str,
    deployment_topology: str,
    deployment_preflight_content_sha256: str,
    semantic_dependency_lock_sha256: str,
    service_url_map_content_sha256: str,
    browser_host_identity: Mapping[str, Any],
    browsergym_runtime_configuration: Mapping[str, Any],
    task_registry: Mapping[str, Any],
    campaign_budgets: Mapping[str, Any],
    _additional_frozen_snapshots: Sequence[_ArtifactSnapshot] = (),
) -> dict[str, Any]:
    """Collect and seal one exact, non-authorizing live timeout block.

    The safe-probe and source-receipt paths and their independently supplied
    digests are mandatory.  Existing output, missing coverage, source/input
    mutation, harness-returned data, failed cleanup, or reset-state drift fail
    closed and cannot emit a successful calibration artifact.
    """

    root = _absolute_path(repository_root, "timeout collector repository root")
    _assert_no_symlink_ancestry(root, "timeout collector repository root")
    if not root.is_dir() or root.is_symlink():
        raise SchemaError("timeout collector repository root is invalid")
    if deployment_topology not in _TOPOLOGIES:
        raise SchemaError("timeout collector deployment topology is unregistered")
    for value, label in (
        (deployment_preflight_content_sha256, "deployment preflight"),
        (semantic_dependency_lock_sha256, "semantic dependency lock"),
        (service_url_map_content_sha256, "service URL map"),
    ):
        _require_sha256(value, f"timeout collector {label}")
    host = _validate_host_identity(browser_host_identity)
    configuration = _validate_browsergym_configuration(
        browsergym_runtime_configuration
    )
    registry, task_ids = _validate_task_registry(task_registry)
    budgets = _validate_budgets(campaign_budgets)
    if type(harness_factory_entrypoint) is not str or _ENTRYPOINT_RE.fullmatch(
        harness_factory_entrypoint
    ) is None:
        raise SchemaError("timeout harness factory entrypoint is malformed")

    safe_manifest, safe_snapshot = _load_frozen_json(
        safe_probe_manifest_path,
        expected_canonical_sha256=expected_safe_probe_manifest_sha256,
        label="timeout safe-probe manifest",
    )
    harness_receipt, harness_receipt_snapshot = _load_frozen_json(
        measurement_harness_source_receipt_path,
        expected_canonical_sha256=(
            expected_measurement_harness_source_receipt_sha256
        ),
        label="timeout harness source receipt",
    )
    harness_receipt = _validate_harness_source_receipt(harness_receipt)
    if (
        harness_receipt["schema_version"]
        != HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION
        or harness_receipt["collector_entrypoint"] != TIMEOUT_COLLECTOR_ENTRYPOINT
        or harness_receipt["collector_source_relative_path"]
        != TIMEOUT_COLLECTOR_SOURCE_RELATIVE_PATH
    ):
        raise SchemaError("timeout harness receipt does not bind this collector")
    _validate_safe_probe_manifest(
        safe_manifest,
        evidence={"task_registry": registry, "measurements": []},
        task_ids=task_ids,
    )
    source_snapshots, registered_sources = _snapshot_harness_sources(
        root, harness_receipt
    )
    frozen_snapshots = (
        safe_snapshot,
        harness_receipt_snapshot,
        *tuple(_additional_frozen_snapshots),
    )
    output = _reserve_output_directory(output_directory)
    staging = output / STAGING_DIRECTORY_NAME
    progress: dict[str, Any] = {
        "task_order": None,
        "task_id": None,
        "operation_class": "harness_factory",
    }
    harness: ProcessBrokerTimeoutProbeHarness | None = None
    try:
        os.mkdir(staging, 0o700)
        harness = _load_harness(
            entrypoint=harness_factory_entrypoint,
            repository_root=root,
            registered_sources=registered_sources,
        )
        measurements, non_persistence_entries = _collect_measurements(
            harness=harness,
            safe_manifest=safe_manifest,
            progress=progress,
        )
        for snapshot in frozen_snapshots:
            _assert_frozen_json_unchanged(snapshot, snapshot.path.name)
        _assert_sources_unchanged(source_snapshots)

        safe_sha = safe_snapshot.canonical_sha256
        harness_receipt_sha = harness_receipt_snapshot.canonical_sha256
        non_persistence = {
            "schema_version": NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
            "record_type": "ProcessBrokerTimeoutNonPersistenceReceipt",
            "status": "NO_PERSISTENT_MUTATION_OBSERVED",
            "task_manifest_sha256": registry["manifest_sha256"],
            "ordered_task_ids": list(task_ids),
            "safe_probe_manifest_sha256": safe_sha,
            "persistent_mutation_detected": False,
            "entries": non_persistence_entries,
        }
        non_persistence_sha = sha256_json(non_persistence)
        build_kwargs = {
            "measurement_harness_source_set_sha256": harness_receipt[
                "source_set_sha256"
            ],
            "safe_probe_manifest_sha256": safe_sha,
            "non_persistence_audit_receipt_sha256": non_persistence_sha,
            "deployment_topology": deployment_topology,
            "deployment_preflight_content_sha256": (
                deployment_preflight_content_sha256
            ),
            "semantic_dependency_lock_sha256": semantic_dependency_lock_sha256,
            "service_url_map_content_sha256": service_url_map_content_sha256,
            "browser_host_identity": host,
            "browsergym_runtime_configuration": configuration,
            "task_registry": registry,
            "campaign_budgets": budgets,
            "measurements": measurements,
        }
        provisional = build_process_broker_timeout_calibration(
            measurement_source_receipt_sha256="0" * 64,
            **build_kwargs,
        )
        measurement_receipt = _measurement_source_receipt(
            evidence=provisional,
            measurements=measurements,
            harness_source_set_sha256=harness_receipt["source_set_sha256"],
            safe_probe_manifest_sha256=safe_sha,
            non_persistence_receipt_sha256=non_persistence_sha,
        )
        measurement_receipt_sha = sha256_json(measurement_receipt)
        calibration = build_process_broker_timeout_calibration(
            measurement_source_receipt_sha256=measurement_receipt_sha,
            **build_kwargs,
        )
        _validate_safe_probe_manifest(
            safe_manifest, evidence=calibration, task_ids=task_ids
        )
        _validate_non_persistence_receipt(
            non_persistence,
            evidence=calibration,
            safe_probe_manifest_sha256=safe_sha,
            task_ids=task_ids,
        )
        _validate_measurement_source_receipt(
            measurement_receipt,
            evidence=calibration,
            safe_probe_manifest_sha256=safe_sha,
            harness_source_set_sha256=harness_receipt["source_set_sha256"],
            non_persistence_receipt_sha256=non_persistence_sha,
        )

        artifact_values = {
            CALIBRATION_FILENAME: calibration,
            SAFE_PROBE_FILENAME: safe_manifest,
            HARNESS_SOURCE_RECEIPT_FILENAME: harness_receipt,
            MEASUREMENT_SOURCE_RECEIPT_FILENAME: measurement_receipt,
            NON_PERSISTENCE_RECEIPT_FILENAME: non_persistence,
        }
        artifact_hashes: dict[str, str] = {}
        for filename, value in artifact_values.items():
            artifact_hashes[filename] = _write_new_readonly_json(
                staging / filename, value
            )

        manifest = {
            "schema_version": TIMEOUT_COLLECTOR_MANIFEST_SCHEMA_VERSION,
            "record_type": "ProcessBrokerTimeoutCollectionManifest",
            "status": TIMEOUT_CALIBRATION_STATUS,
            "claim_scope": TIMEOUT_CALIBRATION_CLAIM_SCOPE,
            "evidence_label": "PRE_CAMPAIGN_INFRASTRUCTURE_ONLY",
            "paper_table_status": "N/R",
            "production_dispatch_authorized": False,
            "live_campaign_authority": False,
            "external_cross_binding_present": False,
            "caller_selected_timeout": False,
            "measurement_clock": MONOTONIC_CLOCK_ID,
            "measurement_count": len(measurements),
            "task_count": len(task_ids),
            "operation_classes": list(CALIBRATION_OPERATION_CLASSES),
            "safe_probe_manifest_sha256": safe_sha,
            "measurement_harness_source_receipt_sha256": harness_receipt_sha,
            "measurement_harness_source_set_sha256": harness_receipt[
                "source_set_sha256"
            ],
            "harness_factory_entrypoint": harness_factory_entrypoint,
            "harness_factory_entrypoint_sha256": hashlib.sha256(
                harness_factory_entrypoint.encode("utf-8")
            ).hexdigest(),
            "calibration_content_sha256": artifact_hashes[CALIBRATION_FILENAME],
            "artifact_files": [
                {"relative_path": name, "sha256": artifact_hashes[name]}
                for name in sorted(CORE_ARTIFACT_FILENAMES)
            ],
            "rerun_at_same_output_path_permitted": False,
        }
        manifest_sha = sha256_json(manifest)

        def authority_for(directory: Path) -> ProcessBrokerTimeoutExpectedAuthority:
            return ProcessBrokerTimeoutExpectedAuthority(
                schema_version=(
                    "table2-process-broker-timeout-expected-authority-v1"
                ),
                record_type="ProcessBrokerTimeoutExpectedAuthority",
                claim_scope="LOCAL_EXPECTED_IDENTITIES_NOT_EXTERNAL_AUTHORITY",
                paper_table_status="N/R",
                production_dispatch_authorized=False,
                calibration_artifact_path=directory / CALIBRATION_FILENAME,
                calibration_content_sha256=artifact_hashes[CALIBRATION_FILENAME],
                collection_manifest_path=directory / COLLECTION_MANIFEST_FILENAME,
                collection_manifest_content_sha256=manifest_sha,
                task_manifest_sha256=registry["manifest_sha256"],
                ordered_task_ids=task_ids,
                ordered_upstream_indices=tuple(
                    registry["ordered_upstream_indices"]
                ),
                deployment_topology=deployment_topology,
                browser_host_identity=host,
                browsergym_runtime_configuration=configuration,
                deployment_preflight_content_sha256=(
                    deployment_preflight_content_sha256
                ),
                semantic_dependency_lock_sha256=(
                    semantic_dependency_lock_sha256
                ),
                service_url_map_content_sha256=service_url_map_content_sha256,
                campaign_budgets=budgets,
                safe_probe_manifest_path=directory / SAFE_PROBE_FILENAME,
                safe_probe_manifest_content_sha256=safe_sha,
                measurement_harness_source_receipt_path=(
                    directory / HARNESS_SOURCE_RECEIPT_FILENAME
                ),
                measurement_harness_source_receipt_content_sha256=(
                    harness_receipt_sha
                ),
                measurement_harness_source_set_sha256=harness_receipt[
                    "source_set_sha256"
                ],
                measurement_source_receipt_path=(
                    directory / MEASUREMENT_SOURCE_RECEIPT_FILENAME
                ),
                measurement_source_receipt_content_sha256=(
                    measurement_receipt_sha
                ),
                non_persistence_audit_receipt_path=(
                    directory / NON_PERSISTENCE_RECEIPT_FILENAME
                ),
                non_persistence_audit_receipt_content_sha256=(
                    non_persistence_sha
                ),
                external_cross_binding_present=False,
                live_campaign_authority=False,
            )

        authority = authority_for(output)
        _write_new_readonly_json(
            staging / LOCAL_EXPECTED_AUTHORITY_FILENAME,
            authority.to_dict(),
        )
        # This is the stage's success marker too, so it is written only after
        # every staged artifact and the local expected-identity record.
        _write_new_readonly_json(
            staging / COLLECTION_MANIFEST_FILENAME, manifest
        )
        _seal_directory(staging)
        if (
            load_authority_bound_timeout_calibration(authority_for(staging))
            != calibration
        ):
            raise SchemaError("staged timeout bundle replay differs")
        for snapshot in frozen_snapshots:
            _assert_frozen_json_unchanged(snapshot, snapshot.path.name)
        _assert_sources_unchanged(source_snapshots)

        # Invalidate the staged pathname before any publication.  Core files
        # are then moved into the reserved output, with the success manifest
        # deliberately moved last.  Until the final directory is sealed it is
        # still rejected by the authority loader.
        os.chmod(staging, 0o700)
        staged_manifest = staging / ".validated_collection_manifest.json"
        os.rename(staging / COLLECTION_MANIFEST_FILENAME, staged_manifest)
        for filename in sorted(CORE_ARTIFACT_FILENAMES):
            os.rename(staging / filename, output / filename)
        os.rename(
            staging / LOCAL_EXPECTED_AUTHORITY_FILENAME,
            output / LOCAL_EXPECTED_AUTHORITY_FILENAME,
        )
        os.rename(staged_manifest, output / COLLECTION_MANIFEST_FILENAME)
        os.rmdir(staging)
        _seal_directory(output)
        if load_authority_bound_timeout_calibration(authority) != calibration:
            raise SchemaError("published timeout bundle replay differs")
        return manifest
    except BaseException as collection_error:
        cleanup_attempted = False
        cleanup_succeeded = False
        cleanup_error_class: str | None = None
        if harness is not None and progress.get("task_id") is not None:
            cleanup_attempted = True
            try:
                cleanup_result = harness.cleanup_after_collection_failure(
                    FrozenProbeCleanupRequest(
                        task_order=int(progress["task_order"]),
                        task_id=str(progress["task_id"]),
                        failed_operation_class=(
                            str(progress["operation_class"])
                            if progress.get("operation_class") is not None
                            else None
                        ),
                    )
                )
                if cleanup_result is not None:
                    raise SchemaError(
                        "timeout harness failure cleanup returned forbidden data"
                    )
                cleanup_succeeded = True
            except BaseException as cleanup_error:
                cleanup_error_class = type(cleanup_error).__name__
        try:
            _record_failed_attempt(
                output,
                staging=staging,
                progress=progress,
                safe_probe_manifest_sha256=safe_snapshot.canonical_sha256,
                harness_source_receipt_sha256=(
                    harness_receipt_snapshot.canonical_sha256
                ),
                harness_factory_entrypoint=harness_factory_entrypoint,
                collection_error_class=type(collection_error).__name__,
                cleanup_attempted=cleanup_attempted,
                cleanup_succeeded=cleanup_succeeded,
                cleanup_error_class=cleanup_error_class,
            )
        except BaseException as record_error:
            raise record_error from collection_error
        raise


def collect_from_frozen_timeout_collector_input(
    *,
    repository_root: str | Path,
    output_directory: str | Path,
    collector_input_path: str | Path,
    expected_collector_input_sha256: str,
    safe_probe_manifest_path: str | Path,
    measurement_harness_source_receipt_path: str | Path,
) -> dict[str, Any]:
    """CLI-oriented wrapper around one externally hashed immutable input."""

    payload, input_snapshot = load_frozen_timeout_collector_input(
        collector_input_path,
        expected_content_sha256=expected_collector_input_sha256,
    )
    return collect_process_broker_timeout_calibration(
        repository_root=repository_root,
        output_directory=output_directory,
        safe_probe_manifest_path=safe_probe_manifest_path,
        expected_safe_probe_manifest_sha256=payload[
            "safe_probe_manifest_content_sha256"
        ],
        measurement_harness_source_receipt_path=(
            measurement_harness_source_receipt_path
        ),
        expected_measurement_harness_source_receipt_sha256=payload[
            "measurement_harness_source_receipt_content_sha256"
        ],
        harness_factory_entrypoint=payload["harness_factory_entrypoint"],
        deployment_topology=payload["deployment_topology"],
        deployment_preflight_content_sha256=payload[
            "deployment_preflight_content_sha256"
        ],
        semantic_dependency_lock_sha256=payload[
            "semantic_dependency_lock_sha256"
        ],
        service_url_map_content_sha256=payload[
            "service_url_map_content_sha256"
        ],
        browser_host_identity=payload["browser_host_identity"],
        browsergym_runtime_configuration=payload[
            "browsergym_runtime_configuration"
        ],
        task_registry=payload["task_registry"],
        campaign_budgets=payload["campaign_budgets"],
        _additional_frozen_snapshots=(input_snapshot,),
    )


__all__ = [
    "COLLECTION_FAILURE_FILENAME",
    "COLLECTION_MANIFEST_FILENAME",
    "FrozenProbeCleanupRequest",
    "FrozenProbeOperation",
    "FrozenResetFingerprintRequest",
    "ProcessBrokerTimeoutProbeHarness",
    "TIMEOUT_COLLECTOR_ENTRYPOINT",
    "TIMEOUT_COLLECTOR_INPUT_SCHEMA_VERSION",
    "TIMEOUT_COLLECTOR_INPUT_STATUS",
    "collect_from_frozen_timeout_collector_input",
    "collect_process_broker_timeout_calibration",
    "load_frozen_timeout_collector_input",
    "validate_timeout_collector_input",
]
