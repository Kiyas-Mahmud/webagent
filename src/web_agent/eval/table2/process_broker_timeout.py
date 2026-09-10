"""Outcome-blind calibration contract for Table 2 process-broker timeouts.

The production broker must not inherit an arbitrary developer-selected socket
timeout.  This module defines the preregistered measurement coverage and the
deterministic derivation used to turn live, monotonic timings into per-operation
timeouts.  The calibration is necessary infrastructure evidence, but it is not
deployment authority: a separate independently authenticated deployment
receipt must still bind the artifact to the physical campaign host.

No synthetic browser, data URL, configured upper bound, reward, evaluator
output, task outcome, or paper metric can satisfy this contract.  Collection is
deliberately left to the future source-attested live backend harness because an
execution measurement needs preregistered, non-persistent browser probe
actions.  This module only builds, validates, and consumes the resulting
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from .common import SchemaError, canonical_json_bytes, sha256_file, sha256_json


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

TIMEOUT_CALIBRATION_SCHEMA_VERSION = (
    "table2-process-broker-ipc-timeout-calibration-v3"
)
TIMEOUT_CALIBRATION_RECORD_TYPE = "ProcessBrokerIPCTimeoutCalibration"
TIMEOUT_CALIBRATION_STATUS = "MEASURED_COMPLETE_NON_AUTHORIZING"
TIMEOUT_CALIBRATION_CLAIM_SCOPE = (
    "PRE_CAMPAIGN_OUTCOME_BLIND_INFRASTRUCTURE_CALIBRATION_"
    "NOT_DEPLOYMENT_AUTHORITY"
)
TIMEOUT_BINDING_SCHEMA_VERSION = "table2-process-broker-ipc-timeout-binding-v3"
MEASURED_TIMEOUT_MODE = "MEASURED_PRE_CAMPAIGN_CALIBRATION"
ENGINEERING_TIMEOUT_MODE = "ENGINEERING_FIXTURE_EXPLICIT_NOT_EVALUATION"
TIMEOUT_DERIVATION_ID = "max_observed_x2_plus_1000ms_ceil_100ms_v2"
MONOTONIC_CLOCK_ID = "time.monotonic_ns"
MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES = 8 * 1024 * 1024
# Leaves 640 KiB of the registered 1 MiB IPC frame for source rows, sealed
# backend configuration, identities, HMAC keys, and launch-envelope fields.
MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES = 384 * 1024

RUNTIME_TIMEOUT_OPERATIONS = (
    "runtime_reset",
    "runtime_observe",
    "runtime_execute",
    "runtime_terminal",
    "runtime_close",
)
SEALED_FINALIZATION_TIMEOUT_OPERATION = "sealed_finalize_episode"
BROKER_TIMEOUT_KEYS = (
    "broker_startup",
    *RUNTIME_TIMEOUT_OPERATIONS,
    SEALED_FINALIZATION_TIMEOUT_OPERATION,
    "control_shutdown",
)
CALIBRATION_OPERATION_CLASSES = (
    "broker_startup",
    "runtime_reset",
    "page_settle_reset",
    "runtime_terminal_reset",
    "runtime_execute_action",
    "page_settle_post_action",
    "runtime_observe_action",
    "runtime_terminal_post_action",
    "runtime_execute_recovery",
    "page_settle_post_recovery",
    "runtime_observe_recovery",
    "runtime_terminal_post_recovery",
    "runtime_close",
    "sealed_finalize_episode",
    "control_shutdown_normal",
    "control_shutdown_after_failed_runtime",
)
ACTION_PHASE_CLASSES = frozenset(
    {
        "runtime_execute_action",
        "page_settle_post_action",
        "runtime_observe_action",
        "runtime_terminal_post_action",
    }
)
RECOVERY_PHASE_CLASSES = frozenset(
    {
        "runtime_execute_recovery",
        "page_settle_post_recovery",
        "runtime_observe_recovery",
        "runtime_terminal_post_recovery",
    }
)
ACTION_DEPENDENT_CLASSES = ACTION_PHASE_CLASSES | RECOVERY_PHASE_CLASSES
REGISTERED_ACTION_TYPES = frozenset(
    {"CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY"}
)
DERIVATION_SOURCE_CLASSES = {
    "broker_startup": ("broker_startup",),
    "runtime_reset": ("runtime_reset", "page_settle_reset"),
    "runtime_execute": (
        "runtime_execute_action",
        "page_settle_post_action",
        "runtime_execute_recovery",
        "page_settle_post_recovery",
    ),
    "runtime_observe": (
        "runtime_observe_action",
        "runtime_observe_recovery",
    ),
    "runtime_terminal": (
        "runtime_terminal_reset",
        "runtime_terminal_post_action",
        "runtime_terminal_post_recovery",
    ),
    "runtime_close": ("runtime_close",),
    "sealed_finalize_episode": ("sealed_finalize_episode",),
    "control_shutdown": (
        "runtime_close",
        "control_shutdown_normal",
        "control_shutdown_after_failed_runtime",
    ),
}

SAFE_PROBE_MANIFEST_SCHEMA_VERSION = (
    "table2-process-broker-timeout-safe-probe-manifest-v2"
)
HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION = (
    "table2-process-broker-timeout-harness-source-receipt-v1"
)
MEASUREMENT_SOURCE_RECEIPT_SCHEMA_VERSION = (
    "table2-process-broker-timeout-measurement-source-receipt-v2"
)
NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION = (
    "table2-process-broker-timeout-non-persistence-receipt-v1"
)
TIMEOUT_COLLECTION_MANIFEST_SCHEMA_VERSION = (
    "table2-process-broker-timeout-collection-manifest-v1"
)

TIMEOUT_CALIBRATION_FILENAME = "process_broker_timeout_calibration.json"
TIMEOUT_SAFE_PROBE_FILENAME = "safe_probe_manifest.json"
TIMEOUT_HARNESS_SOURCE_RECEIPT_FILENAME = (
    "measurement_harness_source_receipt.json"
)
TIMEOUT_MEASUREMENT_SOURCE_RECEIPT_FILENAME = "measurement_source_receipt.json"
TIMEOUT_NON_PERSISTENCE_RECEIPT_FILENAME = "non_persistence_audit_receipt.json"
TIMEOUT_COLLECTION_MANIFEST_FILENAME = "collection_manifest.json"
TIMEOUT_COLLECTION_FAILURE_FILENAME = "COLLECTION_FAILED.json"

_TIMEOUT_CORE_ARTIFACT_FILENAMES = (
    TIMEOUT_CALIBRATION_FILENAME,
    TIMEOUT_HARNESS_SOURCE_RECEIPT_FILENAME,
    TIMEOUT_MEASUREMENT_SOURCE_RECEIPT_FILENAME,
    TIMEOUT_NON_PERSISTENCE_RECEIPT_FILENAME,
    TIMEOUT_SAFE_PROBE_FILENAME,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_TOPOLOGIES = frozenset({"SINGLE_DGX_HOST", "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"})
_CALIBRATION_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "claim_scope",
    "evidence_label",
    "paper_table_status",
    "live_campaign_authority",
    "collected_before_campaign_outcomes",
    "reward_read",
    "evaluator_output_read",
    "oracle_label_read",
    "paper_metric_read",
    "outcome_based_tuning",
    "persistent_environment_mutation",
    "measurement_clock",
    "measurement_harness_source_set_sha256",
    "measurement_source_receipt_sha256",
    "safe_probe_manifest_sha256",
    "non_persistence_audit_receipt_sha256",
    "deployment_topology",
    "deployment_preflight_content_sha256",
    "semantic_dependency_lock_sha256",
    "service_url_map_content_sha256",
    "browser_host_identity",
    "browser_host_identity_sha256",
    "browsergym_runtime_configuration",
    "browsergym_runtime_configuration_sha256",
    "task_registry",
    "campaign_budgets",
    "campaign_budgets_sha256",
    "operation_classes",
    "coverage_rule",
    "derivation_contract",
    "measurements",
    "measurement_max_elapsed_ms",
    "derived_timeout_milliseconds",
}
_MEASUREMENT_FIELDS = {
    "task_order",
    "task_id",
    "operation_class",
    "started_monotonic_ns",
    "ended_monotonic_ns",
    "elapsed_ns",
    "action_type",
    "probe_action_sha256",
    "failure_injection_sha256",
    "safety_receipt_sha256",
}
_TASK_REGISTRY_FIELDS = {
    "manifest_id",
    "manifest_sha256",
    "ordered_task_ids",
    "ordered_upstream_indices",
    "task_set_sha256",
}
_HOST_IDENTITY_FIELDS = {
    "system",
    "release",
    "machine",
    "python_version",
    "python_executable_sha256",
}
_BROWSERGYM_CONFIGURATION_FIELDS = {
    "schema_version",
    "record_type",
    "viewport_width",
    "viewport_height",
    "headless",
    "slow_mo_ms",
    "playwright_timeout_ms",
    "locale",
    "timezone_id",
    "tags_to_mark",
    "pre_observation_delay_seconds",
    "network_idle_required",
    "page_settle_timeout_seconds",
    "video_recording_enabled",
    "trace_recording_enabled",
}
_CAMPAIGN_BUDGET_FIELDS = {
    "executor_requests_per_episode",
    "recovery_attempts_per_incident",
    "recovery_attempts_per_episode",
    "model_calls_per_episode",
    "recovery_at_k",
    "task_timeout_seconds",
    "pre_browser_setup_timeout_seconds",
    "whole_block_infrastructure_reruns",
    "infrastructure_invalid_reasons",
    "abort_counts_as_recovery_attempt",
    "abort_executor_steps",
}
_BINDING_FIELDS = {
    "schema_version",
    "mode",
    "calibration_schema_version",
    "calibration_content_sha256",
    "measurement_source_receipt_sha256",
    "timeout_milliseconds",
    "measured_calibration_complete",
    "caller_selected_timeout",
    "live_campaign_authority",
}
_COLLECTION_MANIFEST_FIELDS = {
    "schema_version",
    "record_type",
    "status",
    "claim_scope",
    "evidence_label",
    "paper_table_status",
    "production_dispatch_authorized",
    "live_campaign_authority",
    "external_cross_binding_present",
    "caller_selected_timeout",
    "measurement_clock",
    "measurement_count",
    "task_count",
    "operation_classes",
    "safe_probe_manifest_sha256",
    "measurement_harness_source_receipt_sha256",
    "measurement_harness_source_set_sha256",
    "harness_factory_entrypoint",
    "harness_factory_entrypoint_sha256",
    "calibration_content_sha256",
    "artifact_files",
    "rerun_at_same_output_path_permitted",
}
_COLLECTION_MANIFEST_ARTIFACT_FIELDS = {"relative_path", "sha256"}


@dataclass(frozen=True, slots=True)
class ProcessBrokerTimeoutExpectedAuthority:
    """Exact non-authorizing identities expected by a frozen replay.

    Every field is mandatory.  This object deliberately has no signature or
    trust-anchor field: validating it proves closed internal relationships,
    not that an operator or campaign authority approved those relationships.
    Consequently it can authorize local replay only.  True evaluation remains
    blocked until an external authority cross-binds this complete object.
    """

    schema_version: str
    record_type: str
    claim_scope: str
    paper_table_status: str
    production_dispatch_authorized: bool
    calibration_artifact_path: str | Path
    calibration_content_sha256: str
    collection_manifest_path: str | Path
    collection_manifest_content_sha256: str
    task_manifest_sha256: str
    ordered_task_ids: tuple[str, ...]
    ordered_upstream_indices: tuple[int, ...]
    deployment_topology: str
    browser_host_identity: Mapping[str, Any]
    browsergym_runtime_configuration: Mapping[str, Any]
    deployment_preflight_content_sha256: str
    semantic_dependency_lock_sha256: str
    service_url_map_content_sha256: str
    campaign_budgets: Mapping[str, Any]
    safe_probe_manifest_path: str | Path
    safe_probe_manifest_content_sha256: str
    measurement_harness_source_receipt_path: str | Path
    measurement_harness_source_receipt_content_sha256: str
    measurement_harness_source_set_sha256: str
    measurement_source_receipt_path: str | Path
    measurement_source_receipt_content_sha256: str
    non_persistence_audit_receipt_path: str | Path
    non_persistence_audit_receipt_content_sha256: str
    external_cross_binding_present: bool
    live_campaign_authority: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "claim_scope": self.claim_scope,
            "paper_table_status": self.paper_table_status,
            "production_dispatch_authorized": self.production_dispatch_authorized,
            "calibration_artifact_path": os.fspath(self.calibration_artifact_path),
            "calibration_content_sha256": self.calibration_content_sha256,
            "collection_manifest_path": os.fspath(self.collection_manifest_path),
            "collection_manifest_content_sha256": (
                self.collection_manifest_content_sha256
            ),
            "task_manifest_sha256": self.task_manifest_sha256,
            "ordered_task_ids": list(self.ordered_task_ids),
            "ordered_upstream_indices": list(self.ordered_upstream_indices),
            "deployment_topology": self.deployment_topology,
            "browser_host_identity": _detached_mapping(
                self.browser_host_identity, "timeout authority browser host"
            ),
            "browsergym_runtime_configuration": _detached_mapping(
                self.browsergym_runtime_configuration,
                "timeout authority BrowserGym configuration",
            ),
            "deployment_preflight_content_sha256": (
                self.deployment_preflight_content_sha256
            ),
            "semantic_dependency_lock_sha256": self.semantic_dependency_lock_sha256,
            "service_url_map_content_sha256": self.service_url_map_content_sha256,
            "campaign_budgets": _detached_mapping(
                self.campaign_budgets, "timeout authority budgets"
            ),
            "safe_probe_manifest_path": os.fspath(self.safe_probe_manifest_path),
            "safe_probe_manifest_content_sha256": (
                self.safe_probe_manifest_content_sha256
            ),
            "measurement_harness_source_receipt_path": os.fspath(
                self.measurement_harness_source_receipt_path
            ),
            "measurement_harness_source_receipt_content_sha256": (
                self.measurement_harness_source_receipt_content_sha256
            ),
            "measurement_harness_source_set_sha256": (
                self.measurement_harness_source_set_sha256
            ),
            "measurement_source_receipt_path": os.fspath(
                self.measurement_source_receipt_path
            ),
            "measurement_source_receipt_content_sha256": (
                self.measurement_source_receipt_content_sha256
            ),
            "non_persistence_audit_receipt_path": os.fspath(
                self.non_persistence_audit_receipt_path
            ),
            "non_persistence_audit_receipt_content_sha256": (
                self.non_persistence_audit_receipt_content_sha256
            ),
            "external_cross_binding_present": self.external_cross_binding_present,
            "live_campaign_authority": self.live_campaign_authority,
        }


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise SchemaError(f"{label} fields differ from the registered closure")


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SchemaError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _require_positive_finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SchemaError(f"{label} must be positive and finite")
    return float(value)


def _detached_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    try:
        detached = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SchemaError(f"{label} must be canonical JSON") from exc
    if not isinstance(detached, dict):  # pragma: no cover - Mapping encodes as object
        raise SchemaError(f"{label} must be an object")
    return detached


def _validate_browsergym_configuration(value: object) -> dict[str, Any]:
    config = _detached_mapping(value, "BrowserGym runtime configuration")
    _require_exact_fields(
        config, _BROWSERGYM_CONFIGURATION_FIELDS, "BrowserGym runtime configuration"
    )
    if (
        config.get("schema_version") != "table2.runtime.v1"
        or config.get("record_type") != "BrowserGymRuntimeConfiguration"
        or config.get("headless") is not True
        or config.get("video_recording_enabled") is not False
        or config.get("trace_recording_enabled") is not False
        or config.get("network_idle_required") is not True
    ):
        raise SchemaError("BrowserGym runtime configuration is not campaign compatible")
    for field in ("viewport_width", "viewport_height", "playwright_timeout_ms"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise SchemaError(f"BrowserGym {field} must be a positive exact integer")
    if type(config.get("slow_mo_ms")) is not int or config["slow_mo_ms"] < 0:
        raise SchemaError("BrowserGym slow_mo_ms must be nonnegative")
    if config.get("tags_to_mark") not in {"all", "standard_html"}:
        raise SchemaError("BrowserGym tags_to_mark is unregistered")
    for field in ("locale", "timezone_id"):
        item = config.get(field)
        if item is not None and (
            type(item) is not str
            or not item
            or item != item.strip()
            or len(item) > 128
        ):
            raise SchemaError(f"BrowserGym {field} is invalid")
    _require_positive_finite(
        config.get("pre_observation_delay_seconds"),
        "BrowserGym pre-observation delay",
    )
    _require_positive_finite(
        config.get("page_settle_timeout_seconds"),
        "BrowserGym page-settle timeout",
    )
    return config


def _validate_host_identity(value: object) -> dict[str, Any]:
    identity = _detached_mapping(value, "browser-host identity")
    _require_exact_fields(identity, _HOST_IDENTITY_FIELDS, "browser-host identity")
    for field in ("system", "release", "machine", "python_version"):
        item = identity.get(field)
        if (
            type(item) is not str
            or not item.strip()
            or item != item.strip()
            or len(item) > 256
        ):
            raise SchemaError(f"browser-host identity {field} is absent")
    _require_sha256(
        identity.get("python_executable_sha256"),
        "browser-host Python executable",
    )
    return identity


def _validate_task_registry(value: object) -> tuple[dict[str, Any], tuple[str, ...]]:
    registry = _detached_mapping(value, "timeout-calibration task registry")
    _require_exact_fields(registry, _TASK_REGISTRY_FIELDS, "timeout-calibration task registry")
    manifest_id = registry.get("manifest_id")
    if (
        type(manifest_id) is not str
        or not manifest_id.strip()
        or manifest_id != manifest_id.strip()
        or len(manifest_id) > 256
    ):
        raise SchemaError("timeout-calibration task registry lacks manifest_id")
    _require_sha256(registry.get("manifest_sha256"), "task-registry manifest")
    task_ids = registry.get("ordered_task_ids")
    upstream = registry.get("ordered_upstream_indices")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != 50
        or any(type(item) is not str or _TASK_ID_RE.fullmatch(item) is None for item in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise SchemaError("timeout calibration requires exactly 50 unique ordered task IDs")
    if (
        not isinstance(upstream, list)
        or len(upstream) != len(task_ids)
        or any(type(item) is not int or item < 0 for item in upstream)
        or len(set(upstream)) != len(upstream)
    ):
        raise SchemaError("timeout calibration requires 50 unique upstream indices")
    expected_set_sha256 = sha256_json(
        [
            {"task_id": task_id, "upstream_index": index}
            for task_id, index in zip(task_ids, upstream, strict=True)
        ]
    )
    if registry.get("task_set_sha256") != expected_set_sha256:
        raise SchemaError("timeout-calibration ordered task-set hash differs")
    return registry, tuple(task_ids)


def _validate_budgets(value: object) -> dict[str, Any]:
    budgets = _detached_mapping(value, "timeout-calibration campaign budgets")
    _require_exact_fields(budgets, _CAMPAIGN_BUDGET_FIELDS, "timeout-calibration budgets")
    for field in (
        "executor_requests_per_episode",
        "recovery_attempts_per_incident",
        "recovery_attempts_per_episode",
        "model_calls_per_episode",
        "recovery_at_k",
        "task_timeout_seconds",
        "pre_browser_setup_timeout_seconds",
        "whole_block_infrastructure_reruns",
        "abort_executor_steps",
    ):
        if type(budgets.get(field)) is not int or budgets[field] < 0:
            raise SchemaError(f"timeout-calibration budget {field} is invalid")
    if budgets["task_timeout_seconds"] <= 0 or budgets["pre_browser_setup_timeout_seconds"] <= 0:
        raise SchemaError("timeout calibration requires positive wall-clock budgets")
    if budgets.get("abort_counts_as_recovery_attempt") is not True:
        raise SchemaError("timeout-calibration abort accounting changed")
    reasons = budgets.get("infrastructure_invalid_reasons")
    if (
        not isinstance(reasons, list)
        or len(reasons) > 16
        or reasons != sorted(set(reasons))
        or any(
            type(item) is not str
            or not item
            or item != item.strip()
            or len(item) > 96
            for item in reasons
        )
    ):
        raise SchemaError("timeout-calibration infrastructure reasons are not canonical")
    return budgets


def _expected_derivation_contract() -> dict[str, Any]:
    return {
        "derivation_id": TIMEOUT_DERIVATION_ID,
        "statistic": "maximum_observed_monotonic_elapsed_ns",
        "safety_factor_numerator": 2,
        "safety_factor_denominator": 1,
        "additive_margin_ms": 1000,
        "round_up_quantum_ms": 100,
        "configured_bounds_are_not_measurement_substitutes": True,
        "configured_floor_contract": {
            "reset_execute": (
                "playwright_timeout_ms+page_settle_timeout_ms+"
                "pre_observation_delay_ms+1000ms"
            ),
            "terminal_close_control": "playwright_timeout_ms+1000ms",
            "sealed_finalization": "playwright_timeout_ms+1000ms",
            "startup_observe": "1000ms",
        },
        "timeout_source_classes": {
            key: list(value) for key, value in DERIVATION_SOURCE_CLASSES.items()
        },
        "setup_ceiling_source": "campaign_budgets.pre_browser_setup_timeout_seconds",
        "runtime_ceiling_source": "campaign_budgets.task_timeout_seconds",
    }


def _measurement_rows(
    value: object,
    *,
    task_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(value, list):
        raise SchemaError("timeout-calibration measurements must be an array")
    expected_count = len(task_ids) * len(CALIBRATION_OPERATION_CLASSES)
    if len(value) != expected_count:
        raise SchemaError(
            "timeout calibration lacks exact task-by-operation measurement coverage"
        )
    detached = json.loads(canonical_json_bytes(value))
    registered_operations = set(CALIBRATION_OPERATION_CLASSES)
    expected_order = [
        (task_order, operation)
        for task_order in range(len(task_ids))
        for operation in CALIBRATION_OPERATION_CLASSES
    ]
    actual_order: list[tuple[int, str]] = []
    maximum_ns = {name: 0 for name in CALIBRATION_OPERATION_CLASSES}
    by_task: dict[int, dict[str, dict[str, Any]]] = {}
    observed_action_types = {"ordinary": set(), "recovery": set()}
    for row in detached:
        if not isinstance(row, dict):
            raise SchemaError("timeout-calibration measurement row must be an object")
        _require_exact_fields(row, _MEASUREMENT_FIELDS, "timeout-calibration measurement")
        task_order = row.get("task_order")
        operation = row.get("operation_class")
        if type(task_order) is not int or task_order not in range(len(task_ids)):
            raise SchemaError("timeout-calibration task order is invalid")
        if row.get("task_id") != task_ids[task_order]:
            raise SchemaError("timeout-calibration task identity differs from registry order")
        if operation not in registered_operations:
            raise SchemaError("timeout-calibration operation class is unregistered")
        actual_order.append((task_order, str(operation)))
        started = row.get("started_monotonic_ns")
        ended = row.get("ended_monotonic_ns")
        elapsed = row.get("elapsed_ns")
        if (
            type(started) is not int
            or type(ended) is not int
            or type(elapsed) is not int
            or started < 0
            or ended <= started
            or elapsed != ended - started
        ):
            raise SchemaError("timeout calibration requires exact positive monotonic durations")
        maximum_ns[str(operation)] = max(maximum_ns[str(operation)], elapsed)
        action_type = row.get("action_type")
        action_sha = row.get("probe_action_sha256")
        failure_sha = row.get("failure_injection_sha256")
        safety_sha = row.get("safety_receipt_sha256")
        if operation in ACTION_DEPENDENT_CLASSES:
            if action_type not in REGISTERED_ACTION_TYPES:
                raise SchemaError("timeout-calibration action type is unregistered")
            _require_sha256(action_sha, "timeout-calibration probe action")
            _require_sha256(safety_sha, "timeout-calibration action-safety receipt")
            if failure_sha is not None:
                raise SchemaError("action timeout measurement contains failure metadata")
            phase = "ordinary" if operation in ACTION_PHASE_CLASSES else "recovery"
            observed_action_types[phase].add(str(action_type))
        elif operation == "control_shutdown_after_failed_runtime":
            if action_type is not None or action_sha is not None:
                raise SchemaError("failed-cleanup timing cannot contain an action")
            _require_sha256(
                failure_sha, "timeout-calibration failed-runtime injection"
            )
            _require_sha256(
                safety_sha, "timeout-calibration failed-runtime safety receipt"
            )
        elif any(
            item is not None
            for item in (action_type, action_sha, failure_sha, safety_sha)
        ):
            raise SchemaError("non-action timeout measurement contains action metadata")
        by_task.setdefault(task_order, {})[str(operation)] = row
    if actual_order != expected_order:
        raise SchemaError("timeout-calibration measurements are not in registered causal order")
    for phase, action_types in observed_action_types.items():
        if action_types != set(REGISTERED_ACTION_TYPES):
            raise SchemaError(
                "timeout calibration does not cover all six registered action "
                f"classes in the {phase} phase"
            )

    for task_order, rows in by_task.items():
        ordinary = [
            rows[name]
            for name in CALIBRATION_OPERATION_CLASSES
            if name in ACTION_PHASE_CLASSES
        ]
        recovery = [
            rows[name]
            for name in CALIBRATION_OPERATION_CLASSES
            if name in RECOVERY_PHASE_CLASSES
        ]
        for phase_rows, label in ((ordinary, "action"), (recovery, "recovery")):
            action_identities = {
                (
                    row["action_type"],
                    row["probe_action_sha256"],
                    row["safety_receipt_sha256"],
                )
                for row in phase_rows
            }
            if len(action_identities) != 1:
                raise SchemaError(f"timeout-calibration {label} phase identity drifted")
        if ordinary[0]["probe_action_sha256"] == recovery[0]["probe_action_sha256"]:
            raise SchemaError("ordinary and recovery calibration requests must be distinct")

        reset = rows["runtime_reset"]
        settle_reset = rows["page_settle_reset"]
        execute_action = rows["runtime_execute_action"]
        settle_action = rows["page_settle_post_action"]
        execute_recovery = rows["runtime_execute_recovery"]
        settle_recovery = rows["page_settle_post_recovery"]
        for outer, inner, label in (
            (reset, settle_reset, "reset"),
            (execute_action, settle_action, "post-action"),
            (execute_recovery, settle_recovery, "post-recovery"),
        ):
            if not (
                outer["started_monotonic_ns"] <= inner["started_monotonic_ns"]
                < inner["ended_monotonic_ns"] <= outer["ended_monotonic_ns"]
            ):
                raise SchemaError(f"timeout-calibration {label} settle interval is not nested")
        causal = [
            rows["broker_startup"],
            rows["runtime_reset"],
            rows["runtime_terminal_reset"],
            rows["runtime_execute_action"],
            rows["runtime_observe_action"],
            rows["runtime_terminal_post_action"],
            rows["runtime_execute_recovery"],
            rows["runtime_observe_recovery"],
            rows["runtime_terminal_post_recovery"],
            rows["runtime_close"],
            rows["sealed_finalize_episode"],
            rows["control_shutdown_normal"],
            rows["control_shutdown_after_failed_runtime"],
        ]
        if any(
            left["ended_monotonic_ns"] > right["started_monotonic_ns"]
            for left, right in zip(causal, causal[1:])
        ):
            raise SchemaError(
                f"timeout-calibration task {task_order} operation intervals overlap out of order"
            )
    return detached, maximum_ns


def _ceil_milliseconds(nanoseconds: int) -> int:
    return (nanoseconds + 999_999) // 1_000_000


def _round_up(value: int, quantum: int = 100) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _derive_timeout_milliseconds(
    maximum_ns: Mapping[str, int],
    *,
    configuration: Mapping[str, Any],
    budgets: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    maxima_ms: dict[str, int] = {
        operation: _ceil_milliseconds(int(maximum_ns[operation]))
        for operation in CALIBRATION_OPERATION_CLASSES
    }
    playwright_ms = int(configuration["playwright_timeout_ms"])
    settle_ms = math.ceil(float(configuration["page_settle_timeout_seconds"]) * 1000.0)
    delay_ms = math.ceil(float(configuration["pre_observation_delay_seconds"]) * 1000.0)
    configured_floor = {
        "broker_startup": 1000,
        "runtime_reset": playwright_ms + settle_ms + delay_ms + 1000,
        "runtime_execute": playwright_ms + settle_ms + delay_ms + 1000,
        "runtime_observe": 1000,
        "runtime_terminal": playwright_ms + 1000,
        "runtime_close": playwright_ms + 1000,
        "sealed_finalize_episode": playwright_ms + 1000,
        "control_shutdown": playwright_ms + 1000,
    }
    derived: dict[str, int] = {}
    for timeout_key in BROKER_TIMEOUT_KEYS:
        observed_max = max(
            maxima_ms[source]
            for source in DERIVATION_SOURCE_CLASSES[timeout_key]
        )
        candidate = max(observed_max * 2 + 1000, configured_floor[timeout_key])
        derived[timeout_key] = _round_up(candidate)
    setup_ceiling = int(budgets["pre_browser_setup_timeout_seconds"]) * 1000
    runtime_ceiling = int(budgets["task_timeout_seconds"]) * 1000
    for timeout_key in ("broker_startup", "control_shutdown"):
        if derived[timeout_key] > setup_ceiling:
            raise SchemaError(
                f"measured {timeout_key} timeout exceeds frozen setup budget"
            )
    for timeout_key in (
        *RUNTIME_TIMEOUT_OPERATIONS,
        SEALED_FINALIZATION_TIMEOUT_OPERATION,
    ):
        if derived[timeout_key] > runtime_ceiling:
            raise SchemaError(
                f"measured {timeout_key} timeout exceeds frozen task budget"
            )
    return maxima_ms, derived


def _validate_action_safety_approval(
    value: object,
    *,
    task_id: str,
    phase: str,
    action_type: str,
    probe_action_sha256: str,
) -> dict[str, Any]:
    approval = _detached_mapping(value, "timeout probe action-safety approval")
    _require_exact_fields(
        approval,
        {
            "schema_version",
            "record_type",
            "task_id",
            "phase",
            "action_type",
            "probe_action_sha256",
            "approved",
            "persistent_environment_mutation",
            "destructive",
            "external_side_effect",
            "credential_state_change",
        },
        "timeout probe action-safety approval",
    )
    expected = {
        "schema_version": "table2-process-broker-timeout-action-safety-v1",
        "record_type": "ProcessBrokerTimeoutActionSafetyApproval",
        "task_id": task_id,
        "phase": phase,
        "action_type": action_type,
        "probe_action_sha256": probe_action_sha256,
        "approved": True,
        "persistent_environment_mutation": False,
        "destructive": False,
        "external_side_effect": False,
        "credential_state_change": False,
    }
    if approval != expected:
        raise SchemaError("timeout probe action-safety approval differs")
    return approval


def _validate_failure_safety_approval(
    value: object,
    *,
    task_id: str,
    failure_injection_sha256: str,
) -> dict[str, Any]:
    approval = _detached_mapping(value, "failed-runtime probe safety approval")
    _require_exact_fields(
        approval,
        {
            "schema_version",
            "record_type",
            "task_id",
            "failure_injection_sha256",
            "failure_path",
            "approved",
            "persistent_environment_mutation",
            "destructive",
            "external_side_effect",
            "credential_state_change",
            "reward_or_evaluator_access",
        },
        "failed-runtime probe safety approval",
    )
    expected = {
        "schema_version": "table2-process-broker-timeout-failure-safety-v1",
        "record_type": "ProcessBrokerTimeoutFailureSafetyApproval",
        "task_id": task_id,
        "failure_injection_sha256": failure_injection_sha256,
        "failure_path": "SAFE_NON_PERSISTENT_INJECTED_RUNTIME_FAILURE",
        "approved": True,
        "persistent_environment_mutation": False,
        "destructive": False,
        "external_side_effect": False,
        "credential_state_change": False,
        "reward_or_evaluator_access": False,
    }
    if approval != expected:
        raise SchemaError("failed-runtime probe safety approval differs")
    return approval


def _validate_safe_probe_manifest(
    value: object,
    *,
    evidence: Mapping[str, Any],
    task_ids: tuple[str, ...],
) -> dict[str, Any]:
    manifest = _detached_mapping(value, "timeout safe-probe manifest")
    _require_exact_fields(
        manifest,
        {
            "schema_version",
            "record_type",
            "manifest_id",
            "task_manifest_sha256",
            "ordered_task_ids",
            "ordinary_probes",
            "recovery_probes",
            "failed_runtime_cleanup_probes",
        },
        "timeout safe-probe manifest",
    )
    if (
        manifest.get("schema_version") != SAFE_PROBE_MANIFEST_SCHEMA_VERSION
        or manifest.get("record_type") != "ProcessBrokerTimeoutSafeProbeManifest"
        or type(manifest.get("manifest_id")) is not str
        or not manifest["manifest_id"].strip()
        or manifest.get("task_manifest_sha256")
        != evidence["task_registry"]["manifest_sha256"]
        or manifest.get("ordered_task_ids") != list(task_ids)
    ):
        raise SchemaError("timeout safe-probe manifest identity differs")

    row_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for phase, field in (
        ("ordinary", "ordinary_probes"),
        ("recovery", "recovery_probes"),
    ):
        entries = manifest.get(field)
        if not isinstance(entries, list) or len(entries) != len(task_ids):
            raise SchemaError(f"timeout safe-probe {phase} coverage differs")
        covered: set[str] = set()
        for task_order, entry_value in enumerate(entries):
            entry = _detached_mapping(entry_value, f"timeout {phase} probe")
            _require_exact_fields(
                entry,
                {
                    "task_order",
                    "task_id",
                    "phase",
                    "action_type",
                    "probe_action_sha256",
                    "safety_approval",
                },
                f"timeout {phase} probe",
            )
            task_id = task_ids[task_order]
            action_type = entry.get("action_type")
            probe_sha = _require_sha256(
                entry.get("probe_action_sha256"), f"timeout {phase} probe action"
            )
            if (
                entry.get("task_order") != task_order
                or entry.get("task_id") != task_id
                or entry.get("phase") != phase
                or action_type not in REGISTERED_ACTION_TYPES
            ):
                raise SchemaError(f"timeout {phase} probe identity differs")
            approval = _validate_action_safety_approval(
                entry.get("safety_approval"),
                task_id=task_id,
                phase=phase,
                action_type=str(action_type),
                probe_action_sha256=probe_sha,
            )
            covered.add(str(action_type))
            row_lookup[(task_id, phase)] = {
                "action_type": action_type,
                "probe_action_sha256": probe_sha,
                "safety_receipt_sha256": sha256_json(approval),
            }
        if covered != set(REGISTERED_ACTION_TYPES):
            raise SchemaError(
                f"timeout safe-probe {phase} phase lacks all six actions"
            )

    failed = manifest.get("failed_runtime_cleanup_probes")
    if not isinstance(failed, list) or len(failed) != len(task_ids):
        raise SchemaError("failed-runtime cleanup probe coverage differs")
    failure_lookup: dict[str, dict[str, str]] = {}
    for task_order, entry_value in enumerate(failed):
        entry = _detached_mapping(entry_value, "failed-runtime cleanup probe")
        _require_exact_fields(
            entry,
            {
                "task_order",
                "task_id",
                "failure_injection_sha256",
                "safety_approval",
            },
            "failed-runtime cleanup probe",
        )
        task_id = task_ids[task_order]
        failure_sha = _require_sha256(
            entry.get("failure_injection_sha256"),
            "failed-runtime cleanup injection",
        )
        if entry.get("task_order") != task_order or entry.get("task_id") != task_id:
            raise SchemaError("failed-runtime cleanup probe identity differs")
        approval = _validate_failure_safety_approval(
            entry.get("safety_approval"),
            task_id=task_id,
            failure_injection_sha256=failure_sha,
        )
        failure_lookup[task_id] = {
            "failure_injection_sha256": failure_sha,
            "safety_receipt_sha256": sha256_json(approval),
        }

    for row in evidence["measurements"]:
        operation = row["operation_class"]
        if operation in ACTION_PHASE_CLASSES:
            expected = row_lookup[(row["task_id"], "ordinary")]
        elif operation in RECOVERY_PHASE_CLASSES:
            expected = row_lookup[(row["task_id"], "recovery")]
        elif operation == "control_shutdown_after_failed_runtime":
            expected = failure_lookup[row["task_id"]]
        else:
            continue
        for field, expected_value in expected.items():
            if row.get(field) != expected_value:
                raise SchemaError(
                    "timeout measurement is not a member of the safe-probe manifest"
                )
    return manifest


def _validate_harness_source_receipt(value: object) -> dict[str, Any]:
    receipt = _detached_mapping(value, "timeout harness source receipt")
    _require_exact_fields(
        receipt,
        {
            "schema_version",
            "record_type",
            "receipt_id",
            "collector_entrypoint",
            "collector_source_relative_path",
            "source_files",
            "source_set_sha256",
            "measurement_clock",
            "raw_monotonic_intervals_recorded",
            "reward_interface_imported",
            "evaluator_interface_imported",
            "oracle_interface_imported",
            "outcome_labels_accepted",
        },
        "timeout harness source receipt",
    )
    if (
        receipt.get("schema_version") != HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("record_type") != "ProcessBrokerTimeoutHarnessSourceReceipt"
        or type(receipt.get("receipt_id")) is not str
        or not receipt["receipt_id"].strip()
        or type(receipt.get("collector_entrypoint")) is not str
        or ":" not in receipt["collector_entrypoint"]
        or receipt.get("measurement_clock") != MONOTONIC_CLOCK_ID
        or receipt.get("raw_monotonic_intervals_recorded") is not True
        or any(
            receipt.get(field) is not False
            for field in (
                "reward_interface_imported",
                "evaluator_interface_imported",
                "oracle_interface_imported",
                "outcome_labels_accepted",
            )
        )
    ):
        raise SchemaError("timeout harness source receipt identity differs")
    rows = receipt.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("timeout harness source receipt lacks sources")
    previous = ""
    seen: set[str] = set()
    for row_value in rows:
        row = _detached_mapping(row_value, "timeout harness source row")
        _require_exact_fields(
            row, {"relative_path", "sha256"}, "timeout harness source row"
        )
        relative = row.get("relative_path")
        if (
            type(relative) is not str
            or not relative.endswith(".py")
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative <= previous
            or relative in seen
        ):
            raise SchemaError("timeout harness source paths are not canonical")
        _require_sha256(row.get("sha256"), "timeout harness source")
        previous = relative
        seen.add(relative)
    collector_source = receipt.get("collector_source_relative_path")
    if collector_source not in seen:
        raise SchemaError("timeout collector source is outside its source closure")
    if receipt.get("source_set_sha256") != sha256_json(rows):
        raise SchemaError("timeout harness source-set hash differs")
    return receipt


def _validate_non_persistence_receipt(
    value: object,
    *,
    evidence: Mapping[str, Any],
    safe_probe_manifest_sha256: str,
    task_ids: tuple[str, ...],
) -> dict[str, Any]:
    receipt = _detached_mapping(value, "timeout non-persistence receipt")
    _require_exact_fields(
        receipt,
        {
            "schema_version",
            "record_type",
            "status",
            "task_manifest_sha256",
            "ordered_task_ids",
            "safe_probe_manifest_sha256",
            "persistent_mutation_detected",
            "entries",
        },
        "timeout non-persistence receipt",
    )
    if (
        receipt.get("schema_version") != NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("record_type") != "ProcessBrokerTimeoutNonPersistenceReceipt"
        or receipt.get("status") != "NO_PERSISTENT_MUTATION_OBSERVED"
        or receipt.get("task_manifest_sha256")
        != evidence["task_registry"]["manifest_sha256"]
        or receipt.get("ordered_task_ids") != list(task_ids)
        or receipt.get("safe_probe_manifest_sha256")
        != safe_probe_manifest_sha256
        or receipt.get("persistent_mutation_detected") is not False
    ):
        raise SchemaError("timeout non-persistence receipt identity differs")
    entries = receipt.get("entries")
    if not isinstance(entries, list) or len(entries) != len(task_ids):
        raise SchemaError("timeout non-persistence receipt task coverage differs")
    failed_rows = {
        row["task_id"]: row
        for row in evidence["measurements"]
        if row["operation_class"] == "control_shutdown_after_failed_runtime"
    }
    for task_order, entry_value in enumerate(entries):
        entry = _detached_mapping(entry_value, "timeout non-persistence entry")
        _require_exact_fields(
            entry,
            {
                "task_order",
                "task_id",
                "failure_injection_sha256",
                "pre_probe_reset_state_sha256",
                "post_probe_reset_state_sha256",
                "reset_state_equivalent",
                "persistent_mutation_detected",
            },
            "timeout non-persistence entry",
        )
        task_id = task_ids[task_order]
        before = _require_sha256(
            entry.get("pre_probe_reset_state_sha256"),
            "pre-probe reset state",
        )
        after = _require_sha256(
            entry.get("post_probe_reset_state_sha256"),
            "post-probe reset state",
        )
        if (
            entry.get("task_order") != task_order
            or entry.get("task_id") != task_id
            or entry.get("failure_injection_sha256")
            != failed_rows[task_id]["failure_injection_sha256"]
            or before != after
            or entry.get("reset_state_equivalent") is not True
            or entry.get("persistent_mutation_detected") is not False
        ):
            raise SchemaError("timeout non-persistence entry differs")
    return receipt


def _validate_measurement_source_receipt(
    value: object,
    *,
    evidence: Mapping[str, Any],
    safe_probe_manifest_sha256: str,
    harness_source_set_sha256: str,
    non_persistence_receipt_sha256: str,
) -> dict[str, Any]:
    receipt = _detached_mapping(value, "timeout measurement source receipt")
    expected_fields = {
        "schema_version",
        "record_type",
        "status",
        "measurement_clock",
        "measurement_harness_source_set_sha256",
        "safe_probe_manifest_sha256",
        "non_persistence_audit_receipt_sha256",
        "task_manifest_sha256",
        "task_set_sha256",
        "deployment_topology",
        "deployment_preflight_content_sha256",
        "semantic_dependency_lock_sha256",
        "service_url_map_content_sha256",
        "browser_host_identity_sha256",
        "browsergym_runtime_configuration_sha256",
        "campaign_budgets_sha256",
        "operation_classes",
        "measurement_count",
        "measurements_sha256",
        "collected_before_campaign_outcomes",
        "reward_read",
        "evaluator_output_read",
        "oracle_label_read",
        "paper_metric_read",
    }
    _require_exact_fields(receipt, expected_fields, "timeout measurement source receipt")
    expected = {
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
        "measurement_count": len(evidence["measurements"]),
        "measurements_sha256": sha256_json(evidence["measurements"]),
        "collected_before_campaign_outcomes": True,
        "reward_read": False,
        "evaluator_output_read": False,
        "oracle_label_read": False,
        "paper_metric_read": False,
    }
    if receipt != expected:
        raise SchemaError("timeout measurement source receipt differs")
    return receipt


def validate_process_broker_timeout_calibration(
    evidence: object,
    *,
    expected_task_manifest_sha256: str | None = None,
    expected_task_ids: Sequence[str] | None = None,
    expected_upstream_indices: Sequence[int] | None = None,
    expected_browsergym_runtime_configuration: Mapping[str, Any] | None = None,
    expected_browser_host_identity: Mapping[str, Any] | None = None,
    expected_deployment_preflight_content_sha256: str | None = None,
    expected_semantic_dependency_lock_sha256: str | None = None,
    expected_service_url_map_content_sha256: str | None = None,
    expected_campaign_budgets: Mapping[str, Any] | None = None,
    expected_safe_probe_manifest_sha256: str | None = None,
    expected_deployment_topology: str | None = None,
    expected_measurement_harness_source_set_sha256: str | None = None,
    expected_measurement_source_receipt_sha256: str | None = None,
    expected_non_persistence_audit_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate complete measured evidence and rederive every timeout.

    Expected arguments are the campaign/handoff authorities.  Omitting them is
    useful only for child-process replay of the self-contained artifact; such a
    replay never upgrades the evidence to deployment authority.
    """

    value = _detached_mapping(evidence, "process-broker timeout calibration")
    if len(canonical_json_bytes(value)) > MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES:
        raise SchemaError(
            "process-broker timeout calibration exceeds its canonical payload bound"
        )
    _require_exact_fields(value, _CALIBRATION_FIELDS, "process-broker timeout calibration")
    fixed = {
        "schema_version": TIMEOUT_CALIBRATION_SCHEMA_VERSION,
        "record_type": TIMEOUT_CALIBRATION_RECORD_TYPE,
        "status": TIMEOUT_CALIBRATION_STATUS,
        "claim_scope": TIMEOUT_CALIBRATION_CLAIM_SCOPE,
        "evidence_label": "PRE_CAMPAIGN_INFRASTRUCTURE_ONLY",
        "paper_table_status": "N/R",
        "live_campaign_authority": False,
        "collected_before_campaign_outcomes": True,
        "reward_read": False,
        "evaluator_output_read": False,
        "oracle_label_read": False,
        "paper_metric_read": False,
        "outcome_based_tuning": False,
        "persistent_environment_mutation": False,
        "measurement_clock": MONOTONIC_CLOCK_ID,
        "coverage_rule": "exactly_once_each_registered_operation_per_task_v1",
        "operation_classes": list(CALIBRATION_OPERATION_CLASSES),
        "derivation_contract": _expected_derivation_contract(),
    }
    for field, expected in fixed.items():
        if type(value.get(field)) is not type(expected) or value.get(field) != expected:
            raise SchemaError(f"timeout calibration {field} differs from registration")
    for field in (
        "measurement_harness_source_set_sha256",
        "measurement_source_receipt_sha256",
        "safe_probe_manifest_sha256",
        "non_persistence_audit_receipt_sha256",
        "deployment_preflight_content_sha256",
        "semantic_dependency_lock_sha256",
        "service_url_map_content_sha256",
        "browser_host_identity_sha256",
        "browsergym_runtime_configuration_sha256",
        "campaign_budgets_sha256",
    ):
        _require_sha256(value.get(field), f"timeout calibration {field}")
    if value.get("deployment_topology") not in _TOPOLOGIES:
        raise SchemaError("timeout calibration deployment topology is unregistered")

    host = _validate_host_identity(value.get("browser_host_identity"))
    if sha256_json(host) != value["browser_host_identity_sha256"]:
        raise SchemaError("timeout-calibration browser-host identity hash differs")
    configuration = _validate_browsergym_configuration(
        value.get("browsergym_runtime_configuration")
    )
    if sha256_json(configuration) != value["browsergym_runtime_configuration_sha256"]:
        raise SchemaError("timeout-calibration BrowserGym configuration hash differs")
    registry, task_ids = _validate_task_registry(value.get("task_registry"))
    budgets = _validate_budgets(value.get("campaign_budgets"))
    if sha256_json(budgets) != value["campaign_budgets_sha256"]:
        raise SchemaError("timeout-calibration campaign-budget hash differs")
    measurements, maximum_ns = _measurement_rows(
        value.get("measurements"), task_ids=task_ids
    )
    maxima_ms, derived = _derive_timeout_milliseconds(
        maximum_ns, configuration=configuration, budgets=budgets
    )
    if value.get("measurement_max_elapsed_ms") != maxima_ms:
        raise SchemaError("timeout-calibration measured maxima were not rederived")
    if value.get("derived_timeout_milliseconds") != derived:
        raise SchemaError("timeout-calibration operation timeouts were not rederived")
    if value.get("measurements") != measurements:
        raise SchemaError("timeout-calibration measurements changed during validation")

    if expected_task_manifest_sha256 is not None and registry["manifest_sha256"] != _require_sha256(
        expected_task_manifest_sha256, "expected task manifest"
    ):
        raise SchemaError("timeout calibration uses another task manifest")
    if expected_task_ids is not None and list(expected_task_ids) != registry["ordered_task_ids"]:
        raise SchemaError("timeout calibration uses another ordered task set")
    if (
        expected_upstream_indices is not None
        and list(expected_upstream_indices)
        != registry["ordered_upstream_indices"]
    ):
        raise SchemaError("timeout calibration uses another upstream task order")
    if expected_browsergym_runtime_configuration is not None and _validate_browsergym_configuration(
        expected_browsergym_runtime_configuration
    ) != configuration:
        raise SchemaError("timeout calibration uses another BrowserGym configuration")
    if expected_browser_host_identity is not None and _validate_host_identity(
        expected_browser_host_identity
    ) != host:
        raise SchemaError("timeout calibration uses another browser-host identity")
    expected_hashes = {
        "deployment_preflight_content_sha256": expected_deployment_preflight_content_sha256,
        "semantic_dependency_lock_sha256": expected_semantic_dependency_lock_sha256,
        "service_url_map_content_sha256": expected_service_url_map_content_sha256,
        "safe_probe_manifest_sha256": expected_safe_probe_manifest_sha256,
        "measurement_harness_source_set_sha256": (
            expected_measurement_harness_source_set_sha256
        ),
        "measurement_source_receipt_sha256": (
            expected_measurement_source_receipt_sha256
        ),
        "non_persistence_audit_receipt_sha256": (
            expected_non_persistence_audit_receipt_sha256
        ),
    }
    for field, expected in expected_hashes.items():
        if expected is not None and value[field] != _require_sha256(expected, f"expected {field}"):
            raise SchemaError(f"timeout calibration {field} differs from authority")
    if expected_campaign_budgets is not None and _validate_budgets(
        expected_campaign_budgets
    ) != budgets:
        raise SchemaError("timeout calibration uses another campaign budget")
    if (
        expected_deployment_topology is not None
        and value["deployment_topology"] != expected_deployment_topology
    ):
        raise SchemaError("timeout calibration uses another deployment topology")
    return deepcopy(value)


def _load_immutable_strict_json(
    path: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Open one immutable artifact without symlink/hardlink or JSON ambiguity."""

    source = Path(os.path.abspath(os.fspath(path)))
    current = Path(source.anchor)
    try:
        for component in source.parts[1:]:
            current = current / component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise SchemaError(f"{label} must not use symlink ancestry")
    except OSError as exc:
        raise SchemaError(f"{label} is absent") from exc
    metadata = source.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SchemaError(f"{label} must be a single-link regular file")
    if (
        metadata.st_size <= 0
        or metadata.st_size > MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES
    ):
        raise SchemaError(f"{label} size is outside the registered bound")
    if metadata.st_mode & 0o222:
        raise SchemaError(f"{label} must be read-only")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or opened_before.st_mode & 0o222
            or opened_before.st_size <= 0
            or opened_before.st_size > MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES
            or (opened_before.st_dev, opened_before.st_ino)
            != (metadata.st_dev, metadata.st_ino)
            or opened_before.st_size != metadata.st_size
            or opened_before.st_mode != metadata.st_mode
        ):
            raise SchemaError(f"{label} changed during open")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(opened_before.st_size)
            if len(payload) != opened_before.st_size or stream.read(1):
                raise SchemaError(f"{label} size changed during bounded read")
        opened_after = os.fstat(descriptor)
        if (
            (opened_after.st_dev, opened_after.st_ino, opened_after.st_size)
            != (opened_before.st_dev, opened_before.st_ino, opened_before.st_size)
            or opened_after.st_mode != opened_before.st_mode
            or opened_after.st_mode & 0o222
            or opened_after.st_size > MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES
            or opened_after.st_mtime_ns != opened_before.st_mtime_ns
            or opened_after.st_ctime_ns != opened_before.st_ctime_ns
        ):
            raise SchemaError(f"{label} changed while reading")
    finally:
        os.close(descriptor)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise SchemaError(
                    f"{label} contains duplicate JSON key: {key}"
                )
            result[key] = item
        return result

    def reject_constant(token: str) -> None:
        raise SchemaError(
            f"{label} contains non-finite JSON number: {token}"
        )

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{label} is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{label} must contain one JSON object")
    return value


def load_process_broker_timeout_calibration(
    path: str | Path,
    **expected: Any,
) -> dict[str, Any]:
    value = _load_immutable_strict_json(
        path, label="timeout-calibration artifact"
    )
    return validate_process_broker_timeout_calibration(value, **expected)


def _timeout_authority_bundle_directory(
    authority: ProcessBrokerTimeoutExpectedAuthority,
) -> tuple[Path, tuple[int, int, int]]:
    """Resolve the one sealed directory allowed to contain an authority bundle."""

    expected_names = {
        "calibration_artifact_path": TIMEOUT_CALIBRATION_FILENAME,
        "collection_manifest_path": TIMEOUT_COLLECTION_MANIFEST_FILENAME,
        "safe_probe_manifest_path": TIMEOUT_SAFE_PROBE_FILENAME,
        "measurement_harness_source_receipt_path": (
            TIMEOUT_HARNESS_SOURCE_RECEIPT_FILENAME
        ),
        "measurement_source_receipt_path": (
            TIMEOUT_MEASUREMENT_SOURCE_RECEIPT_FILENAME
        ),
        "non_persistence_audit_receipt_path": (
            TIMEOUT_NON_PERSISTENCE_RECEIPT_FILENAME
        ),
    }
    canonical_directory: Path | None = None
    for field, expected_name in expected_names.items():
        raw_path = Path(os.fspath(getattr(authority, field)))
        if not raw_path.is_absolute():
            raise SchemaError(f"timeout expected-authority {field} must be absolute")
        path = Path(os.path.abspath(raw_path))
        if path.name != expected_name:
            raise SchemaError(
                f"timeout expected-authority {field} must use the registered filename"
            )
        current = Path(path.anchor)
        try:
            for component in path.parent.parts[1:]:
                current = current / component
                metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise SchemaError(
                        "timeout authority bundle must not use symlink ancestry"
                    )
        except OSError as exc:
            raise SchemaError("timeout authority bundle directory is absent") from exc
        resolved_parent = path.parent.resolve(strict=True)
        if canonical_directory is None:
            canonical_directory = resolved_parent
        elif resolved_parent != canonical_directory:
            raise SchemaError(
                "timeout authority artifacts must share one canonical directory"
            )

    assert canonical_directory is not None  # all expected paths are mandatory
    try:
        directory_metadata = canonical_directory.lstat()
    except OSError as exc:  # pragma: no cover - resolved directory already existed
        raise SchemaError("timeout authority bundle directory is absent") from exc
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_ISLNK(directory_metadata.st_mode)
        or directory_metadata.st_mode & 0o222
    ):
        raise SchemaError("timeout authority bundle directory must be read-only")
    if os.path.lexists(canonical_directory / TIMEOUT_COLLECTION_FAILURE_FILENAME):
        raise SchemaError("failed timeout collection cannot provide calibration authority")
    return canonical_directory, (
        directory_metadata.st_dev,
        directory_metadata.st_ino,
        directory_metadata.st_mode,
    )


def _validate_timeout_collection_manifest(
    value: object,
    *,
    authority: ProcessBrokerTimeoutExpectedAuthority,
) -> dict[str, Any]:
    """Validate the success marker and its exact five-artifact closure."""

    manifest = _detached_mapping(value, "timeout collection manifest")
    _require_exact_fields(
        manifest, _COLLECTION_MANIFEST_FIELDS, "timeout collection manifest"
    )
    fixed = {
        "schema_version": TIMEOUT_COLLECTION_MANIFEST_SCHEMA_VERSION,
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
        "rerun_at_same_output_path_permitted": False,
    }
    if any(manifest.get(field) != expected for field, expected in fixed.items()):
        raise SchemaError("timeout collection manifest authority boundary differs")
    for field in ("measurement_count", "task_count"):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise SchemaError(f"timeout collection manifest {field} is invalid")
    if manifest.get("operation_classes") != list(CALIBRATION_OPERATION_CLASSES):
        raise SchemaError("timeout collection manifest operation classes differ")
    expected_identities = {
        "safe_probe_manifest_sha256": (
            authority.safe_probe_manifest_content_sha256
        ),
        "measurement_harness_source_receipt_sha256": (
            authority.measurement_harness_source_receipt_content_sha256
        ),
        "measurement_harness_source_set_sha256": (
            authority.measurement_harness_source_set_sha256
        ),
        "calibration_content_sha256": authority.calibration_content_sha256,
    }
    for field, expected in expected_identities.items():
        if manifest.get(field) != expected:
            raise SchemaError(
                f"timeout collection manifest {field} differs from authority"
            )
    entrypoint = manifest.get("harness_factory_entrypoint")
    if type(entrypoint) is not str or ":" not in entrypoint or not entrypoint.strip():
        raise SchemaError("timeout collection manifest harness entrypoint is invalid")
    if manifest.get("harness_factory_entrypoint_sha256") != hashlib.sha256(
        entrypoint.encode("utf-8")
    ).hexdigest():
        raise SchemaError("timeout collection manifest harness entrypoint hash differs")

    expected_artifacts = {
        TIMEOUT_CALIBRATION_FILENAME: authority.calibration_content_sha256,
        TIMEOUT_SAFE_PROBE_FILENAME: (
            authority.safe_probe_manifest_content_sha256
        ),
        TIMEOUT_HARNESS_SOURCE_RECEIPT_FILENAME: (
            authority.measurement_harness_source_receipt_content_sha256
        ),
        TIMEOUT_MEASUREMENT_SOURCE_RECEIPT_FILENAME: (
            authority.measurement_source_receipt_content_sha256
        ),
        TIMEOUT_NON_PERSISTENCE_RECEIPT_FILENAME: (
            authority.non_persistence_audit_receipt_content_sha256
        ),
    }
    rows = manifest.get("artifact_files")
    if not isinstance(rows, list) or len(rows) != len(expected_artifacts):
        raise SchemaError(
            "timeout collection manifest lacks the exact core artifact set"
        )
    observed: dict[str, str] = {}
    ordered_paths: list[str] = []
    for row_value in rows:
        row = _detached_mapping(row_value, "timeout collection artifact row")
        _require_exact_fields(
            row,
            _COLLECTION_MANIFEST_ARTIFACT_FIELDS,
            "timeout collection artifact row",
        )
        relative = row.get("relative_path")
        digest = _require_sha256(
            row.get("sha256"), "timeout collection artifact"
        )
        if (
            type(relative) is not str
            or relative not in expected_artifacts
            or relative in observed
            or Path(relative).name != relative
        ):
            raise SchemaError(
                "timeout collection manifest core artifact paths differ"
            )
        observed[relative] = digest
        ordered_paths.append(relative)
    if observed != expected_artifacts or ordered_paths != sorted(expected_artifacts):
        raise SchemaError(
            "timeout collection manifest lacks the exact core artifact set"
        )
    return manifest


def load_authority_bound_timeout_calibration(
    authority: ProcessBrokerTimeoutExpectedAuthority,
) -> dict[str, Any]:
    """Load and cross-check the complete immutable, non-authorizing bundle."""

    if type(authority) is not ProcessBrokerTimeoutExpectedAuthority:
        raise SchemaError(
            "timeout calibration requires a typed expected-authority binding"
        )
    if (
        authority.schema_version
        != "table2-process-broker-timeout-expected-authority-v1"
        or authority.record_type
        != "ProcessBrokerTimeoutExpectedAuthority"
        or authority.claim_scope
        != "LOCAL_EXPECTED_IDENTITIES_NOT_EXTERNAL_AUTHORITY"
        or authority.paper_table_status != "N/R"
        or authority.production_dispatch_authorized is not False
        or authority.external_cross_binding_present is not False
        or authority.live_campaign_authority is not False
    ):
        raise SchemaError(
            "local timeout expected-authority binding cannot claim external authority"
        )
    bundle_directory, bundle_directory_identity = (
        _timeout_authority_bundle_directory(authority)
    )
    digest_fields = (
        "calibration_content_sha256",
        "collection_manifest_content_sha256",
        "task_manifest_sha256",
        "deployment_preflight_content_sha256",
        "semantic_dependency_lock_sha256",
        "service_url_map_content_sha256",
        "safe_probe_manifest_content_sha256",
        "measurement_harness_source_receipt_content_sha256",
        "measurement_harness_source_set_sha256",
        "measurement_source_receipt_content_sha256",
        "non_persistence_audit_receipt_content_sha256",
    )
    for field in digest_fields:
        _require_sha256(getattr(authority, field), f"timeout authority {field}")
    if authority.deployment_topology not in _TOPOLOGIES:
        raise SchemaError("timeout authority deployment topology is unregistered")
    if len(authority.ordered_task_ids) != 50 or len(
        authority.ordered_upstream_indices
    ) != 50:
        raise SchemaError("timeout authority requires the exact 50-task registry")

    collection_manifest = _load_immutable_strict_json(
        authority.collection_manifest_path,
        label="timeout collection manifest",
    )
    if (
        sha256_json(collection_manifest)
        != authority.collection_manifest_content_sha256
    ):
        raise SchemaError("timeout collection manifest content differs from authority")
    collection_manifest = _validate_timeout_collection_manifest(
        collection_manifest, authority=authority
    )

    evidence = load_process_broker_timeout_calibration(
        authority.calibration_artifact_path,
        expected_task_manifest_sha256=authority.task_manifest_sha256,
        expected_task_ids=authority.ordered_task_ids,
        expected_upstream_indices=authority.ordered_upstream_indices,
        expected_browsergym_runtime_configuration=(
            authority.browsergym_runtime_configuration
        ),
        expected_browser_host_identity=authority.browser_host_identity,
        expected_deployment_preflight_content_sha256=(
            authority.deployment_preflight_content_sha256
        ),
        expected_semantic_dependency_lock_sha256=(
            authority.semantic_dependency_lock_sha256
        ),
        expected_service_url_map_content_sha256=(
            authority.service_url_map_content_sha256
        ),
        expected_campaign_budgets=authority.campaign_budgets,
        expected_safe_probe_manifest_sha256=(
            authority.safe_probe_manifest_content_sha256
        ),
        expected_deployment_topology=authority.deployment_topology,
        expected_measurement_harness_source_set_sha256=(
            authority.measurement_harness_source_set_sha256
        ),
        expected_measurement_source_receipt_sha256=(
            authority.measurement_source_receipt_content_sha256
        ),
        expected_non_persistence_audit_receipt_sha256=(
            authority.non_persistence_audit_receipt_content_sha256
        ),
    )
    if sha256_json(evidence) != authority.calibration_content_sha256:
        raise SchemaError("timeout calibration artifact content differs from authority")
    task_ids = tuple(evidence["task_registry"]["ordered_task_ids"])

    safe_manifest = _load_immutable_strict_json(
        authority.safe_probe_manifest_path,
        label="timeout safe-probe manifest",
    )
    if sha256_json(safe_manifest) != authority.safe_probe_manifest_content_sha256:
        raise SchemaError("timeout safe-probe manifest content differs from authority")
    _validate_safe_probe_manifest(
        safe_manifest, evidence=evidence, task_ids=task_ids
    )

    harness_receipt = _load_immutable_strict_json(
        authority.measurement_harness_source_receipt_path,
        label="timeout harness source receipt",
    )
    if (
        sha256_json(harness_receipt)
        != authority.measurement_harness_source_receipt_content_sha256
    ):
        raise SchemaError("timeout harness source receipt differs from authority")
    validated_harness = _validate_harness_source_receipt(harness_receipt)
    if (
        validated_harness["source_set_sha256"]
        != authority.measurement_harness_source_set_sha256
    ):
        raise SchemaError("timeout harness source set differs from authority")

    non_persistence = _load_immutable_strict_json(
        authority.non_persistence_audit_receipt_path,
        label="timeout non-persistence receipt",
    )
    if (
        sha256_json(non_persistence)
        != authority.non_persistence_audit_receipt_content_sha256
    ):
        raise SchemaError("timeout non-persistence receipt differs from authority")
    _validate_non_persistence_receipt(
        non_persistence,
        evidence=evidence,
        safe_probe_manifest_sha256=authority.safe_probe_manifest_content_sha256,
        task_ids=task_ids,
    )

    measurement_receipt = _load_immutable_strict_json(
        authority.measurement_source_receipt_path,
        label="timeout measurement source receipt",
    )
    if (
        sha256_json(measurement_receipt)
        != authority.measurement_source_receipt_content_sha256
    ):
        raise SchemaError("timeout measurement source receipt differs from authority")
    _validate_measurement_source_receipt(
        measurement_receipt,
        evidence=evidence,
        safe_probe_manifest_sha256=authority.safe_probe_manifest_content_sha256,
        harness_source_set_sha256=authority.measurement_harness_source_set_sha256,
        non_persistence_receipt_sha256=(
            authority.non_persistence_audit_receipt_content_sha256
        ),
    )
    if collection_manifest["measurement_count"] != len(evidence["measurements"]):
        raise SchemaError("timeout collection manifest measurement count differs")
    if collection_manifest["task_count"] != len(task_ids):
        raise SchemaError("timeout collection manifest task count differs")
    try:
        final_directory_metadata = bundle_directory.lstat()
    except OSError as exc:
        raise SchemaError("timeout authority bundle directory disappeared") from exc
    if (
        (
            final_directory_metadata.st_dev,
            final_directory_metadata.st_ino,
            final_directory_metadata.st_mode,
        )
        != bundle_directory_identity
        or final_directory_metadata.st_mode & 0o222
        or os.path.lexists(
            bundle_directory / TIMEOUT_COLLECTION_FAILURE_FILENAME
        )
    ):
        raise SchemaError("timeout authority bundle changed during validation")
    return evidence


def build_process_broker_timeout_calibration(
    *,
    measurement_harness_source_set_sha256: str,
    measurement_source_receipt_sha256: str,
    safe_probe_manifest_sha256: str,
    non_persistence_audit_receipt_sha256: str,
    deployment_topology: str,
    deployment_preflight_content_sha256: str,
    semantic_dependency_lock_sha256: str,
    service_url_map_content_sha256: str,
    browser_host_identity: Mapping[str, Any],
    browsergym_runtime_configuration: Mapping[str, Any],
    task_registry: Mapping[str, Any],
    campaign_budgets: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct deterministic evidence from an already completed live probe.

    This function does not perform the probe and does not create any receipt or
    trust assertion.  Its exact inputs must come from the source-attested live
    collection harness and independently reviewed safe-action manifest.
    """

    host = _validate_host_identity(browser_host_identity)
    config = _validate_browsergym_configuration(browsergym_runtime_configuration)
    registry, task_ids = _validate_task_registry(task_registry)
    budgets = _validate_budgets(campaign_budgets)
    rows, maximum_ns = _measurement_rows(list(measurements), task_ids=task_ids)
    maxima_ms, derived = _derive_timeout_milliseconds(
        maximum_ns, configuration=config, budgets=budgets
    )
    evidence = {
        "schema_version": TIMEOUT_CALIBRATION_SCHEMA_VERSION,
        "record_type": TIMEOUT_CALIBRATION_RECORD_TYPE,
        "status": TIMEOUT_CALIBRATION_STATUS,
        "claim_scope": TIMEOUT_CALIBRATION_CLAIM_SCOPE,
        "evidence_label": "PRE_CAMPAIGN_INFRASTRUCTURE_ONLY",
        "paper_table_status": "N/R",
        "live_campaign_authority": False,
        "collected_before_campaign_outcomes": True,
        "reward_read": False,
        "evaluator_output_read": False,
        "oracle_label_read": False,
        "paper_metric_read": False,
        "outcome_based_tuning": False,
        "persistent_environment_mutation": False,
        "measurement_clock": MONOTONIC_CLOCK_ID,
        "measurement_harness_source_set_sha256": measurement_harness_source_set_sha256,
        "measurement_source_receipt_sha256": measurement_source_receipt_sha256,
        "safe_probe_manifest_sha256": safe_probe_manifest_sha256,
        "non_persistence_audit_receipt_sha256": non_persistence_audit_receipt_sha256,
        "deployment_topology": deployment_topology,
        "deployment_preflight_content_sha256": deployment_preflight_content_sha256,
        "semantic_dependency_lock_sha256": semantic_dependency_lock_sha256,
        "service_url_map_content_sha256": service_url_map_content_sha256,
        "browser_host_identity": host,
        "browser_host_identity_sha256": sha256_json(host),
        "browsergym_runtime_configuration": config,
        "browsergym_runtime_configuration_sha256": sha256_json(config),
        "task_registry": registry,
        "campaign_budgets": budgets,
        "campaign_budgets_sha256": sha256_json(budgets),
        "operation_classes": list(CALIBRATION_OPERATION_CLASSES),
        "coverage_rule": "exactly_once_each_registered_operation_per_task_v1",
        "derivation_contract": _expected_derivation_contract(),
        "measurements": rows,
        "measurement_max_elapsed_ms": maxima_ms,
        "derived_timeout_milliseconds": derived,
    }
    return validate_process_broker_timeout_calibration(evidence)


def engineering_timeout_binding(
    *,
    startup_timeout_seconds: float = 10.0,
    runtime_timeout_seconds: float = 10.0,
    control_shutdown_timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    """Return an explicit non-evaluation binding for deterministic fixtures."""

    seconds = {
        "broker_startup": startup_timeout_seconds,
        **{operation: runtime_timeout_seconds for operation in RUNTIME_TIMEOUT_OPERATIONS},
        SEALED_FINALIZATION_TIMEOUT_OPERATION: runtime_timeout_seconds,
        "control_shutdown": control_shutdown_timeout_seconds,
    }
    milliseconds: dict[str, int] = {}
    for key in BROKER_TIMEOUT_KEYS:
        value = _require_positive_finite(seconds[key], f"engineering {key} timeout")
        converted = value * 1000.0
        if not converted.is_integer():
            raise SchemaError("engineering broker timeout must resolve to whole milliseconds")
        milliseconds[key] = int(converted)
    binding = {
        "schema_version": TIMEOUT_BINDING_SCHEMA_VERSION,
        "mode": ENGINEERING_TIMEOUT_MODE,
        "calibration_schema_version": None,
        "calibration_content_sha256": None,
        "measurement_source_receipt_sha256": None,
        "timeout_milliseconds": milliseconds,
        "measured_calibration_complete": False,
        "caller_selected_timeout": True,
        "live_campaign_authority": False,
    }
    return validate_process_broker_timeout_binding(binding)


def measured_timeout_binding(evidence: object) -> dict[str, Any]:
    validated = validate_process_broker_timeout_calibration(evidence)
    binding = {
        "schema_version": TIMEOUT_BINDING_SCHEMA_VERSION,
        "mode": MEASURED_TIMEOUT_MODE,
        "calibration_schema_version": TIMEOUT_CALIBRATION_SCHEMA_VERSION,
        "calibration_content_sha256": sha256_json(validated),
        "measurement_source_receipt_sha256": validated[
            "measurement_source_receipt_sha256"
        ],
        "timeout_milliseconds": dict(validated["derived_timeout_milliseconds"]),
        "measured_calibration_complete": True,
        "caller_selected_timeout": False,
        "live_campaign_authority": False,
    }
    return validate_process_broker_timeout_binding(
        binding, calibration_evidence=validated
    )


def validate_process_broker_timeout_binding(
    binding: object,
    *,
    calibration_evidence: object | None = None,
) -> dict[str, Any]:
    value = _detached_mapping(binding, "process-broker timeout binding")
    _require_exact_fields(value, _BINDING_FIELDS, "process-broker timeout binding")
    if value.get("schema_version") != TIMEOUT_BINDING_SCHEMA_VERSION:
        raise SchemaError("process-broker timeout binding version differs")
    milliseconds = value.get("timeout_milliseconds")
    if not isinstance(milliseconds, Mapping) or list(milliseconds) != sorted(BROKER_TIMEOUT_KEYS):
        # Canonical JSON sorts mapping keys. Requiring that order catches custom
        # Mapping objects before a binding is consumed by the socket client.
        if not isinstance(milliseconds, Mapping) or set(milliseconds) != set(BROKER_TIMEOUT_KEYS):
            raise SchemaError("process-broker timeout binding operation closure differs")
    if set(milliseconds) != set(BROKER_TIMEOUT_KEYS):
        raise SchemaError("process-broker timeout binding operation closure differs")
    for key in BROKER_TIMEOUT_KEYS:
        item = milliseconds.get(key)
        if type(item) is not int or item <= 0:
            raise SchemaError(f"process-broker {key} timeout must be positive milliseconds")
    if value.get("live_campaign_authority") is not False:
        raise SchemaError("timeout calibration can never be live campaign authority")
    mode = value.get("mode")
    if mode == ENGINEERING_TIMEOUT_MODE:
        expected = {
            "calibration_schema_version": None,
            "calibration_content_sha256": None,
            "measurement_source_receipt_sha256": None,
            "measured_calibration_complete": False,
            "caller_selected_timeout": True,
        }
        if calibration_evidence is not None:
            raise SchemaError("engineering timeout binding cannot attach calibration evidence")
    elif mode == MEASURED_TIMEOUT_MODE:
        expected = {
            "calibration_schema_version": TIMEOUT_CALIBRATION_SCHEMA_VERSION,
            "measured_calibration_complete": True,
            "caller_selected_timeout": False,
        }
        _require_sha256(value.get("calibration_content_sha256"), "timeout calibration content")
        _require_sha256(
            value.get("measurement_source_receipt_sha256"),
            "timeout measurement source receipt",
        )
        if calibration_evidence is not None:
            evidence = validate_process_broker_timeout_calibration(calibration_evidence)
            if value["calibration_content_sha256"] != sha256_json(evidence):
                raise SchemaError("timeout binding calibration content hash differs")
            if value["measurement_source_receipt_sha256"] != evidence[
                "measurement_source_receipt_sha256"
            ]:
                raise SchemaError("timeout binding source receipt differs")
            if dict(milliseconds) != evidence["derived_timeout_milliseconds"]:
                raise SchemaError("timeout binding operation values differ from calibration")
    else:
        raise SchemaError("process-broker timeout binding mode is unregistered")
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise SchemaError(f"process-broker timeout binding {field} differs")
    return deepcopy(value)


def require_measured_process_broker_timeout_binding(
    binding: object,
    *,
    calibration_evidence: object | None,
) -> dict[str, Any]:
    """Fail closed unless the broker uses rederived measured timeouts."""

    value = validate_process_broker_timeout_binding(
        binding, calibration_evidence=calibration_evidence
    )
    if (
        value["mode"] != MEASURED_TIMEOUT_MODE
        or value["measured_calibration_complete"] is not True
        or value["caller_selected_timeout"] is not False
    ):
        raise SchemaError(
            "evaluation broker requires frozen measured IPC-timeout calibration"
        )
    return value


def timeout_seconds_for_runtime_operations(binding: object) -> dict[str, float]:
    value = validate_process_broker_timeout_binding(binding)
    if value["mode"] == MEASURED_TIMEOUT_MODE:
        raise SchemaError(
            "measured timeout binding must be validated with its calibration evidence"
        )
    return {
        operation: value["timeout_milliseconds"][operation] / 1000.0
        for operation in RUNTIME_TIMEOUT_OPERATIONS
    }


def binding_sha256(binding: object) -> str:
    return sha256_json(_detached_mapping(binding, "process-broker timeout binding"))


__all__ = [
    "BROKER_TIMEOUT_KEYS",
    "CALIBRATION_OPERATION_CLASSES",
    "ENGINEERING_TIMEOUT_MODE",
    "HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION",
    "MEASURED_TIMEOUT_MODE",
    "MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES",
    "MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES",
    "MEASUREMENT_SOURCE_RECEIPT_SCHEMA_VERSION",
    "NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION",
    "PROCESS_BROKER_IMPORT_SOURCE_SHA256",
    "ProcessBrokerTimeoutExpectedAuthority",
    "RUNTIME_TIMEOUT_OPERATIONS",
    "SEALED_FINALIZATION_TIMEOUT_OPERATION",
    "SAFE_PROBE_MANIFEST_SCHEMA_VERSION",
    "TIMEOUT_BINDING_SCHEMA_VERSION",
    "TIMEOUT_CALIBRATION_SCHEMA_VERSION",
    "binding_sha256",
    "build_process_broker_timeout_calibration",
    "engineering_timeout_binding",
    "load_authority_bound_timeout_calibration",
    "load_process_broker_timeout_calibration",
    "measured_timeout_binding",
    "require_measured_process_broker_timeout_binding",
    "validate_process_broker_timeout_binding",
    "validate_process_broker_timeout_calibration",
]
