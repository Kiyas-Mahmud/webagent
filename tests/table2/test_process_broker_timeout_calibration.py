from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from web_agent.benchmarks.browsergym_webarena import BrowserGymRuntimeConfiguration
from web_agent.eval.table2 import process_broker_timeout as timeout_module
from web_agent.eval.table2.common import (
    SchemaError,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.process_broker import (
    PROCESS_BROKER_EVALUATION_SCOPE,
    PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
    ProcessIsolatedBroker,
)
from web_agent.eval.table2.process_broker_timeout import (
    CALIBRATION_OPERATION_CLASSES,
    ENGINEERING_TIMEOUT_MODE,
    HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION,
    MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES,
    MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES,
    MEASURED_TIMEOUT_MODE,
    MEASUREMENT_SOURCE_RECEIPT_SCHEMA_VERSION,
    NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
    SAFE_PROBE_MANIFEST_SCHEMA_VERSION,
    ProcessBrokerTimeoutExpectedAuthority,
    build_process_broker_timeout_calibration,
    engineering_timeout_binding,
    load_process_broker_timeout_calibration,
    load_authority_bound_timeout_calibration,
    measured_timeout_binding,
    require_measured_process_broker_timeout_binding,
    validate_process_broker_timeout_calibration,
    validate_process_broker_timeout_binding,
)


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
ACTION_TYPES = ("CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY")
FIXTURE_BACKEND_SOURCE = (
    "src/web_agent/eval/table2/process_broker_fixture_backend.py"
)
FIXTURE_BACKEND_ENTRYPOINT = (
    "web_agent.eval.table2.process_broker_fixture_backend:create_backend"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _registry() -> dict:
    task_ids = [f"webarena.{index}" for index in range(50)]
    upstream = list(range(50))
    return {
        "manifest_id": "table2-webarena-public-dev-50-test-v1",
        "manifest_sha256": _sha("task-manifest"),
        "ordered_task_ids": task_ids,
        "ordered_upstream_indices": upstream,
        "task_set_sha256": sha256_json(
            [
                {"task_id": task_id, "upstream_index": index}
                for task_id, index in zip(task_ids, upstream, strict=True)
            ]
        ),
    }


def _budgets() -> dict:
    return {
        "executor_requests_per_episode": 30,
        "recovery_attempts_per_incident": 2,
        "recovery_attempts_per_episode": 4,
        "model_calls_per_episode": 102,
        "recovery_at_k": 2,
        "task_timeout_seconds": 600,
        "pre_browser_setup_timeout_seconds": 120,
        "whole_block_infrastructure_reruns": 1,
        "infrastructure_invalid_reasons": [
            "BENCHMARK_SERVICE_UNAVAILABLE",
            "BROWSER_CONTROLLER_DISCONNECTED",
            "ENVIRONMENT_RESET_FAILED",
            "FROZEN_DEPENDENCY_UNAVAILABLE",
        ],
        "abort_counts_as_recovery_attempt": True,
        "abort_executor_steps": 0,
    }


def _host() -> dict:
    return {
        "system": "Linux",
        "release": "fixture",
        "machine": "x86_64",
        "python_version": "3.11.0",
        "python_executable_sha256": _sha("python"),
    }


def _intervals(base: int, scale: int) -> dict[str, tuple[int, int]]:
    unit = 1_000_000 * scale
    return {
        "broker_startup": (base + 0 * unit, base + 10 * unit),
        "runtime_reset": (base + 20 * unit, base + 120 * unit),
        "page_settle_reset": (base + 30 * unit, base + 80 * unit),
        "runtime_terminal_reset": (base + 130 * unit, base + 140 * unit),
        "runtime_execute_action": (base + 150 * unit, base + 250 * unit),
        "page_settle_post_action": (base + 160 * unit, base + 210 * unit),
        "runtime_observe_action": (base + 260 * unit, base + 270 * unit),
        "runtime_terminal_post_action": (base + 280 * unit, base + 290 * unit),
        "runtime_execute_recovery": (base + 300 * unit, base + 400 * unit),
        "page_settle_post_recovery": (base + 310 * unit, base + 360 * unit),
        "runtime_observe_recovery": (base + 410 * unit, base + 420 * unit),
        "runtime_terminal_post_recovery": (base + 430 * unit, base + 440 * unit),
        "runtime_close": (base + 450 * unit, base + 460 * unit),
        "control_shutdown_normal": (base + 470 * unit, base + 480 * unit),
        "control_shutdown_after_failed_runtime": (
            base + 490 * unit,
            base + 530 * unit,
        ),
    }


def _measurements() -> list[dict]:
    rows: list[dict] = []
    for task_order in range(50):
        task_id = f"webarena.{task_order}"
        action_type = ACTION_TYPES[task_order % len(ACTION_TYPES)]
        recovery_type = ACTION_TYPES[(task_order + 3) % len(ACTION_TYPES)]
        action_sha = _sha(f"action:{task_order}")
        recovery_sha = _sha(f"recovery:{task_order}")
        action_safety = _sha(f"action-safety:{task_order}")
        recovery_safety = _sha(f"recovery-safety:{task_order}")
        failure_sha = _sha(f"failure-injection:{task_order}")
        failure_safety = _sha(f"failure-safety:{task_order}")
        intervals = _intervals(task_order * 1_000_000_000, 1 + task_order % 3)
        for operation in CALIBRATION_OPERATION_CLASSES:
            started, ended = intervals[operation]
            if operation in {
                "runtime_execute_action",
                "page_settle_post_action",
                "runtime_observe_action",
                "runtime_terminal_post_action",
            }:
                metadata = {
                    "action_type": action_type,
                    "probe_action_sha256": action_sha,
                    "failure_injection_sha256": None,
                    "safety_receipt_sha256": action_safety,
                }
            elif operation in {
                "runtime_execute_recovery",
                "page_settle_post_recovery",
                "runtime_observe_recovery",
                "runtime_terminal_post_recovery",
            }:
                metadata = {
                    "action_type": recovery_type,
                    "probe_action_sha256": recovery_sha,
                    "failure_injection_sha256": None,
                    "safety_receipt_sha256": recovery_safety,
                }
            elif operation == "control_shutdown_after_failed_runtime":
                metadata = {
                    "action_type": None,
                    "probe_action_sha256": None,
                    "failure_injection_sha256": failure_sha,
                    "safety_receipt_sha256": failure_safety,
                }
            else:
                metadata = {
                    "action_type": None,
                    "probe_action_sha256": None,
                    "failure_injection_sha256": None,
                    "safety_receipt_sha256": None,
                }
            rows.append(
                {
                    "task_order": task_order,
                    "task_id": task_id,
                    "operation_class": operation,
                    "started_monotonic_ns": started,
                    "ended_monotonic_ns": ended,
                    "elapsed_ns": ended - started,
                    **metadata,
                }
            )
    return rows


def _evidence() -> dict:
    return build_process_broker_timeout_calibration(
        measurement_harness_source_set_sha256=_sha("harness"),
        measurement_source_receipt_sha256=_sha("source-receipt"),
        safe_probe_manifest_sha256=_sha("safe-probes"),
        non_persistence_audit_receipt_sha256=_sha("non-persistence"),
        deployment_topology="SINGLE_DGX_HOST",
        deployment_preflight_content_sha256=_sha("preflight"),
        semantic_dependency_lock_sha256=_sha("dependency-lock"),
        service_url_map_content_sha256=_sha("service-map"),
        browser_host_identity=_host(),
        browsergym_runtime_configuration=(
            BrowserGymRuntimeConfiguration().to_dict()
        ),
        task_registry=_registry(),
        campaign_budgets=_budgets(),
        measurements=_measurements(),
    )


def test_complete_live_measurements_are_rederived_and_authority_bound() -> None:
    evidence = _evidence()
    validated = validate_process_broker_timeout_calibration(
        evidence,
        expected_task_manifest_sha256=_registry()["manifest_sha256"],
        expected_task_ids=_registry()["ordered_task_ids"],
        expected_upstream_indices=_registry()["ordered_upstream_indices"],
        expected_browsergym_runtime_configuration=(
            BrowserGymRuntimeConfiguration().to_dict()
        ),
        expected_browser_host_identity=_host(),
        expected_deployment_preflight_content_sha256=_sha("preflight"),
        expected_semantic_dependency_lock_sha256=_sha("dependency-lock"),
        expected_service_url_map_content_sha256=_sha("service-map"),
        expected_campaign_budgets=_budgets(),
        expected_safe_probe_manifest_sha256=_sha("safe-probes"),
    )
    assert validated == evidence
    assert evidence["live_campaign_authority"] is False
    assert evidence["outcome_based_tuning"] is False
    assert evidence["reward_read"] is False
    assert evidence["evaluator_output_read"] is False
    assert evidence["measurement_max_elapsed_ms"]["runtime_reset"] == 300
    # BrowserGym's two registered 10-second bounds plus delay and margin form
    # a conservative floor, but only after complete measured coverage passes.
    assert evidence["derived_timeout_milliseconds"]["runtime_reset"] == 21_500
    binding = measured_timeout_binding(evidence)
    assert binding["mode"] == MEASURED_TIMEOUT_MODE
    assert binding["caller_selected_timeout"] is False
    assert require_measured_process_broker_timeout_binding(
        binding, calibration_evidence=evidence
    ) == binding


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda row: row["measurements"].pop(),
            "exact task-by-operation measurement coverage",
        ),
        (
            lambda row: row.update(reward_read=True),
            "reward_read differs",
        ),
        (
            lambda row: row["measurements"][0].update(elapsed_ns=1),
            "exact positive monotonic durations",
        ),
        (
            lambda row: row["measurements"][4].update(
                started_monotonic_ns=row["measurements"][4]["started_monotonic_ns"]
                - 200_000_000
            ),
            "exact positive monotonic durations|settle interval|overlap",
        ),
        (
            lambda row: row["derived_timeout_milliseconds"].update(
                runtime_execute=1
            ),
            "operation timeouts were not rederived",
        ),
    ],
)
def test_calibration_rejects_missing_outcome_tuned_or_forged_evidence(
    mutation, message: str
) -> None:
    evidence = _evidence()
    mutation(evidence)
    with pytest.raises(SchemaError, match=message):
        validate_process_broker_timeout_calibration(evidence)


def test_configured_bounds_and_engineering_timeout_never_substitute_for_measurement() -> None:
    empty = _evidence()
    empty["measurements"] = []
    with pytest.raises(SchemaError, match="exact task-by-operation"):
        validate_process_broker_timeout_calibration(empty)

    engineering = engineering_timeout_binding()
    assert engineering["mode"] == ENGINEERING_TIMEOUT_MODE
    assert engineering["caller_selected_timeout"] is True
    with pytest.raises(SchemaError, match="requires frozen measured"):
        require_measured_process_broker_timeout_binding(
            engineering, calibration_evidence=None
        )


def test_authority_mismatch_and_action_coverage_fail_closed() -> None:
    evidence = _evidence()
    with pytest.raises(SchemaError, match="another task manifest"):
        validate_process_broker_timeout_calibration(
            evidence, expected_task_manifest_sha256=_sha("other")
        )

    missing_action_type = copy.deepcopy(evidence)
    for row in missing_action_type["measurements"]:
        if row["action_type"] == "SELECT":
            row["action_type"] = "CLICK"
    with pytest.raises(SchemaError, match="all six registered action classes"):
        validate_process_broker_timeout_calibration(missing_action_type)

    recovery_only = copy.deepcopy(evidence)
    for row in recovery_only["measurements"]:
        if (
            row["operation_class"]
            in {
                "runtime_execute_recovery",
                "page_settle_post_recovery",
                "runtime_observe_recovery",
                "runtime_terminal_post_recovery",
            }
            and row["action_type"] == "SELECT"
        ):
            row["action_type"] = "CLICK"
    with pytest.raises(SchemaError, match="recovery phase"):
        validate_process_broker_timeout_calibration(recovery_only)


def test_control_shutdown_derives_from_normal_close_and_failed_runtime_cleanup() -> None:
    config = BrowserGymRuntimeConfiguration().to_dict()
    config["playwright_timeout_ms"] = 1
    evidence = build_process_broker_timeout_calibration(
        measurement_harness_source_set_sha256=_sha("harness"),
        measurement_source_receipt_sha256=_sha("source-receipt"),
        safe_probe_manifest_sha256=_sha("safe-probes"),
        non_persistence_audit_receipt_sha256=_sha("non-persistence"),
        deployment_topology="SINGLE_DGX_HOST",
        deployment_preflight_content_sha256=_sha("preflight"),
        semantic_dependency_lock_sha256=_sha("dependency-lock"),
        service_url_map_content_sha256=_sha("service-map"),
        browser_host_identity=_host(),
        browsergym_runtime_configuration=config,
        task_registry=_registry(),
        campaign_budgets=_budgets(),
        measurements=_measurements(),
    )
    assert evidence["measurement_max_elapsed_ms"][
        "control_shutdown_after_failed_runtime"
    ] == 120
    assert evidence["derived_timeout_milliseconds"]["control_shutdown"] == 1_300

    missing_failed_cleanup = copy.deepcopy(evidence)
    missing_failed_cleanup["measurements"] = [
        row
        for row in missing_failed_cleanup["measurements"]
        if not (
            row["task_id"] == "webarena.0"
            and row["operation_class"]
            == "control_shutdown_after_failed_runtime"
        )
    ]
    with pytest.raises(SchemaError, match="exact task-by-operation"):
        validate_process_broker_timeout_calibration(missing_failed_cleanup)


def test_evaluation_broker_forbids_default_or_in_memory_calibration() -> None:
    with pytest.raises(SchemaError, match="immutable calibration artifact"):
        ProcessIsolatedBroker(
            repository_root=ROOT,
            backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
            backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
            sealed_backend_config={
                "schema_version": "table2-process-broker-fixture-backend-v1"
            },
            execution_scope=PROCESS_BROKER_EVALUATION_SCOPE,
        )

    evidence = _evidence()
    with pytest.raises(SchemaError, match="forbids self-declared in-memory"):
        ProcessIsolatedBroker(
            repository_root=ROOT,
            backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
            backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
            sealed_backend_config={
                "schema_version": "table2-process-broker-fixture-backend-v1"
            },
            execution_scope=PROCESS_BROKER_EVALUATION_SCOPE,
            timeout_calibration_evidence=evidence,
        )


def test_local_measured_replay_forbids_injected_timeout_and_binds_calibration() -> None:
    evidence = _evidence()
    with pytest.raises(SchemaError, match="forbids caller-injected"):
        ProcessIsolatedBroker(
            repository_root=ROOT,
            backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
            backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
            sealed_backend_config={
                "schema_version": "table2-process-broker-fixture-backend-v1"
            },
            execution_scope=PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
            timeout_calibration_evidence=evidence,
            runtime_timeout_seconds=1.0,
        )

    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
        backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
        sealed_backend_config={
            "schema_version": "table2-process-broker-fixture-backend-v1"
        },
        execution_scope=PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
        timeout_calibration_evidence=evidence,
    )
    with broker:
        receipt = broker.receipt
        assert receipt.execution_scope == PROCESS_BROKER_MEASURED_REPLAY_SCOPE
        assert receipt.measured_ipc_timeout_calibration_complete is True
        assert receipt.timeout_calibration_replay_only is True
        assert receipt.immutable_timeout_authority_bundle_validated is False
        assert receipt.external_timeout_authority_cross_binding_present is False
        assert receipt.ipc_timeout_calibration_live_campaign_authority is False
        assert receipt.ipc_timeout_binding["timeout_milliseconds"] == evidence[
            "derived_timeout_milliseconds"
        ]
        assert any(
            row["relative_path"]
            == "src/web_agent/eval/table2/process_broker_timeout.py"
            and row["sha256"]
            == sha256_file(
                ROOT
                / "src/web_agent/eval/table2/process_broker_timeout.py"
            )
            for row in receipt.source_files
        )


def test_measured_binding_requires_exact_evidence_and_source_receipt_hash() -> None:
    evidence = _evidence()
    binding = measured_timeout_binding(evidence)

    wrong_content = copy.deepcopy(binding)
    wrong_content["calibration_content_sha256"] = _sha("wrong")
    with pytest.raises(SchemaError, match="calibration content hash differs"):
        validate_process_broker_timeout_binding(
            wrong_content, calibration_evidence=evidence
        )

    missing_source = copy.deepcopy(binding)
    missing_source["measurement_source_receipt_sha256"] = None
    with pytest.raises(SchemaError, match="source receipt"):
        validate_process_broker_timeout_binding(
            missing_source, calibration_evidence=evidence
        )


def test_worker_independently_revalidates_calibration_to_binding_digest() -> None:
    evidence = _evidence()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
        backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
        sealed_backend_config={
            "schema_version": "table2-process-broker-fixture-backend-v1"
        },
        execution_scope=PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
        timeout_calibration_evidence=evidence,
    )
    # Simulate parent-memory corruption after construction. The worker receives
    # both objects and must rederive the evidence-to-binding relation itself.
    broker._timeout_binding["calibration_content_sha256"] = _sha("corrupt")
    with pytest.raises(Exception):
        broker.start()
    assert broker.cleaned


def _write_calibration(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)
    return path


def _authority_artifacts(
    root: Path,
    *,
    membership_mismatch: bool = False,
) -> tuple[ProcessBrokerTimeoutExpectedAuthority, dict]:
    measurements = _measurements()
    ordinary: list[dict] = []
    recovery: list[dict] = []
    failed: list[dict] = []
    for task_order, task_id in enumerate(_registry()["ordered_task_ids"]):
        for phase, target in (("ordinary", ordinary), ("recovery", recovery)):
            action_type = (
                ACTION_TYPES[task_order % len(ACTION_TYPES)]
                if phase == "ordinary"
                else ACTION_TYPES[(task_order + 3) % len(ACTION_TYPES)]
            )
            probe_sha = _sha(f"{'action' if phase == 'ordinary' else 'recovery'}:{task_order}")
            approval = {
                "schema_version": "table2-process-broker-timeout-action-safety-v1",
                "record_type": "ProcessBrokerTimeoutActionSafetyApproval",
                "task_id": task_id,
                "phase": phase,
                "action_type": action_type,
                "probe_action_sha256": probe_sha,
                "approved": True,
                "persistent_environment_mutation": False,
                "destructive": False,
                "external_side_effect": False,
                "credential_state_change": False,
            }
            target.append(
                {
                    "task_order": task_order,
                    "task_id": task_id,
                    "phase": phase,
                    "action_type": action_type,
                    "probe_action_sha256": probe_sha,
                    "safety_approval": approval,
                }
            )
            operation_suffix = "action" if phase == "ordinary" else "recovery"
            for row in measurements:
                if (
                    row["task_id"] == task_id
                    and row["operation_class"]
                    in {
                        f"runtime_execute_{operation_suffix}",
                        f"page_settle_post_{operation_suffix}",
                        f"runtime_observe_{operation_suffix}",
                        f"runtime_terminal_post_{operation_suffix}",
                    }
                ):
                    row["safety_receipt_sha256"] = sha256_json(approval)
        failure_sha = _sha(f"failure-injection:{task_order}")
        failure_approval = {
            "schema_version": "table2-process-broker-timeout-failure-safety-v1",
            "record_type": "ProcessBrokerTimeoutFailureSafetyApproval",
            "task_id": task_id,
            "failure_injection_sha256": failure_sha,
            "failure_path": "SAFE_NON_PERSISTENT_INJECTED_RUNTIME_FAILURE",
            "approved": True,
            "persistent_environment_mutation": False,
            "destructive": False,
            "external_side_effect": False,
            "credential_state_change": False,
            "reward_or_evaluator_access": False,
        }
        failed.append(
            {
                "task_order": task_order,
                "task_id": task_id,
                "failure_injection_sha256": failure_sha,
                "safety_approval": failure_approval,
            }
        )
        for row in measurements:
            if (
                row["task_id"] == task_id
                and row["operation_class"]
                == "control_shutdown_after_failed_runtime"
            ):
                row["safety_receipt_sha256"] = sha256_json(failure_approval)
    if membership_mismatch:
        for row in measurements:
            if row["task_id"] == "webarena.0" and row[
                "operation_class"
            ] in {
                "runtime_execute_action",
                "page_settle_post_action",
                "runtime_observe_action",
                "runtime_terminal_post_action",
            }:
                row["safety_receipt_sha256"] = _sha("unregistered-safety")
    safe_manifest = {
        "schema_version": SAFE_PROBE_MANIFEST_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutSafeProbeManifest",
        "manifest_id": "fixture-safe-probes-v1",
        "task_manifest_sha256": _registry()["manifest_sha256"],
        "ordered_task_ids": _registry()["ordered_task_ids"],
        "ordinary_probes": ordinary,
        "recovery_probes": recovery,
        "failed_runtime_cleanup_probes": failed,
    }
    safe_sha = sha256_json(safe_manifest)
    harness_rows = [
        {
            "relative_path": "tools/collect_timeout_calibration.py",
            "sha256": _sha("collector-source"),
        }
    ]
    harness = {
        "schema_version": HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutHarnessSourceReceipt",
        "receipt_id": "fixture-harness-v1",
        "collector_entrypoint": "tools.collect_timeout_calibration:collect",
        "collector_source_relative_path": harness_rows[0]["relative_path"],
        "source_files": harness_rows,
        "source_set_sha256": sha256_json(harness_rows),
        "measurement_clock": "time.monotonic_ns",
        "raw_monotonic_intervals_recorded": True,
        "reward_interface_imported": False,
        "evaluator_interface_imported": False,
        "oracle_interface_imported": False,
        "outcome_labels_accepted": False,
    }
    non_persistence = {
        "schema_version": NON_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutNonPersistenceReceipt",
        "status": "NO_PERSISTENT_MUTATION_OBSERVED",
        "task_manifest_sha256": _registry()["manifest_sha256"],
        "ordered_task_ids": _registry()["ordered_task_ids"],
        "safe_probe_manifest_sha256": safe_sha,
        "persistent_mutation_detected": False,
        "entries": [
            {
                "task_order": task_order,
                "task_id": task_id,
                "failure_injection_sha256": _sha(
                    f"failure-injection:{task_order}"
                ),
                "pre_probe_reset_state_sha256": _sha(f"state:{task_order}"),
                "post_probe_reset_state_sha256": _sha(f"state:{task_order}"),
                "reset_state_equivalent": True,
                "persistent_mutation_detected": False,
            }
            for task_order, task_id in enumerate(_registry()["ordered_task_ids"])
        ],
    }
    non_persistence_sha = sha256_json(non_persistence)

    build_kwargs = {
        "measurement_harness_source_set_sha256": harness["source_set_sha256"],
        "safe_probe_manifest_sha256": safe_sha,
        "non_persistence_audit_receipt_sha256": non_persistence_sha,
        "deployment_topology": "SINGLE_DGX_HOST",
        "deployment_preflight_content_sha256": _sha("preflight"),
        "semantic_dependency_lock_sha256": _sha("dependency-lock"),
        "service_url_map_content_sha256": _sha("service-map"),
        "browser_host_identity": _host(),
        "browsergym_runtime_configuration": BrowserGymRuntimeConfiguration().to_dict(),
        "task_registry": _registry(),
        "campaign_budgets": _budgets(),
        "measurements": measurements,
    }
    provisional = build_process_broker_timeout_calibration(
        measurement_source_receipt_sha256=SHA,
        **build_kwargs,
    )
    measurement_receipt = {
        "schema_version": MEASUREMENT_SOURCE_RECEIPT_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutMeasurementSourceReceipt",
        "status": "LIVE_MONOTONIC_MEASUREMENTS_CAPTURED_NON_AUTHORIZING",
        "measurement_clock": "time.monotonic_ns",
        "measurement_harness_source_set_sha256": harness["source_set_sha256"],
        "safe_probe_manifest_sha256": safe_sha,
        "non_persistence_audit_receipt_sha256": non_persistence_sha,
        "task_manifest_sha256": _registry()["manifest_sha256"],
        "task_set_sha256": _registry()["task_set_sha256"],
        "deployment_topology": "SINGLE_DGX_HOST",
        "deployment_preflight_content_sha256": _sha("preflight"),
        "semantic_dependency_lock_sha256": _sha("dependency-lock"),
        "service_url_map_content_sha256": _sha("service-map"),
        "browser_host_identity_sha256": provisional["browser_host_identity_sha256"],
        "browsergym_runtime_configuration_sha256": provisional[
            "browsergym_runtime_configuration_sha256"
        ],
        "campaign_budgets_sha256": provisional["campaign_budgets_sha256"],
        "operation_classes": list(CALIBRATION_OPERATION_CLASSES),
        "measurement_count": len(measurements),
        "measurements_sha256": sha256_json(measurements),
        "collected_before_campaign_outcomes": True,
        "reward_read": False,
        "evaluator_output_read": False,
        "oracle_label_read": False,
        "paper_metric_read": False,
    }
    measurement_sha = sha256_json(measurement_receipt)
    evidence = build_process_broker_timeout_calibration(
        measurement_source_receipt_sha256=measurement_sha,
        **build_kwargs,
    )
    paths = {
        "calibration": _write_calibration(root / "calibration.json", evidence),
        "safe": _write_calibration(root / "safe-probes.json", safe_manifest),
        "harness": _write_calibration(root / "harness.json", harness),
        "measurement": _write_calibration(
            root / "measurement.json", measurement_receipt
        ),
        "non_persistence": _write_calibration(
            root / "non-persistence.json", non_persistence
        ),
    }
    authority = ProcessBrokerTimeoutExpectedAuthority(
        schema_version="table2-process-broker-timeout-expected-authority-v1",
        record_type="ProcessBrokerTimeoutExpectedAuthority",
        claim_scope="LOCAL_EXPECTED_IDENTITIES_NOT_EXTERNAL_AUTHORITY",
        paper_table_status="N/R",
        production_dispatch_authorized=False,
        calibration_artifact_path=paths["calibration"],
        calibration_content_sha256=sha256_json(evidence),
        task_manifest_sha256=_registry()["manifest_sha256"],
        ordered_task_ids=tuple(_registry()["ordered_task_ids"]),
        ordered_upstream_indices=tuple(_registry()["ordered_upstream_indices"]),
        deployment_topology="SINGLE_DGX_HOST",
        browser_host_identity=_host(),
        browsergym_runtime_configuration=BrowserGymRuntimeConfiguration().to_dict(),
        deployment_preflight_content_sha256=_sha("preflight"),
        semantic_dependency_lock_sha256=_sha("dependency-lock"),
        service_url_map_content_sha256=_sha("service-map"),
        campaign_budgets=_budgets(),
        safe_probe_manifest_path=paths["safe"],
        safe_probe_manifest_content_sha256=safe_sha,
        measurement_harness_source_receipt_path=paths["harness"],
        measurement_harness_source_receipt_content_sha256=sha256_json(harness),
        measurement_harness_source_set_sha256=harness["source_set_sha256"],
        measurement_source_receipt_path=paths["measurement"],
        measurement_source_receipt_content_sha256=measurement_sha,
        non_persistence_audit_receipt_path=paths["non_persistence"],
        non_persistence_audit_receipt_content_sha256=non_persistence_sha,
        external_cross_binding_present=False,
        live_campaign_authority=False,
    )
    return authority, evidence


def test_calibration_loader_requires_immutable_unambiguous_json(tmp_path: Path) -> None:
    source = _write_calibration(tmp_path / "calibration.json", _evidence())
    assert load_process_broker_timeout_calibration(source) == _evidence()

    source.chmod(0o644)
    with pytest.raises(SchemaError, match="read-only"):
        load_process_broker_timeout_calibration(source)

    source.chmod(0o444)
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(SchemaError, match="single-link"):
        load_process_broker_timeout_calibration(source)
    hardlink.unlink()

    leaf_link = tmp_path / "leaf-link.json"
    leaf_link.symlink_to(source)
    with pytest.raises(SchemaError, match="symlink ancestry"):
        load_process_broker_timeout_calibration(leaf_link)

    real_parent = tmp_path / "real-parent"
    nested = _write_calibration(real_parent / "nested.json", _evidence())
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SchemaError, match="symlink ancestry"):
        load_process_broker_timeout_calibration(parent_link / nested.name)


def test_calibration_loader_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    canonical = json.dumps(_evidence(), sort_keys=True)
    duplicate = canonical.replace(
        '"schema_version":',
        '"schema_version":"forged","schema_version":',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    duplicate_path.chmod(0o444)
    with pytest.raises(SchemaError, match="duplicate JSON key"):
        load_process_broker_timeout_calibration(duplicate_path)

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text('{"value":NaN}', encoding="utf-8")
    nonfinite_path.chmod(0o444)
    with pytest.raises(SchemaError, match="non-finite JSON number"):
        load_process_broker_timeout_calibration(nonfinite_path)

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES + 1)
    oversized.chmod(0o444)
    with pytest.raises(SchemaError, match="size is outside"):
        load_process_broker_timeout_calibration(oversized)


def test_calibration_payload_bound_fails_before_broker_launch_frame(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    assert len(canonical_json_bytes(evidence)) < (
        MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES
    )
    long_ids: list[str] = []
    for index in range(50):
        prefix = f"task-{index:02d}-"
        long_ids.append(prefix + "x" * (256 - len(prefix)))
    evidence["task_registry"]["ordered_task_ids"] = long_ids
    evidence["task_registry"]["task_set_sha256"] = sha256_json(
        [
            {"task_id": task_id, "upstream_index": index}
            for task_id, index in zip(
                long_ids,
                evidence["task_registry"]["ordered_upstream_indices"],
                strict=True,
            )
        ]
    )
    for row in evidence["measurements"]:
        row["task_id"] = long_ids[row["task_order"]]
    assert len(canonical_json_bytes(evidence)) > (
        MAX_TIMEOUT_CALIBRATION_PAYLOAD_BYTES
    )
    with pytest.raises(SchemaError, match="canonical payload bound"):
        validate_process_broker_timeout_calibration(evidence)

    # Replay construction validates before start(), so an oversized
    # calibration cannot reach subprocess launch or IPC framing.
    with pytest.raises(SchemaError, match="canonical payload bound"):
        ProcessIsolatedBroker(
            repository_root=ROOT,
            backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
            backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
            sealed_backend_config={
                "schema_version": "table2-process-broker-fixture-backend-v1"
            },
            execution_scope=PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
            timeout_calibration_evidence=evidence,
        )

    artifact = _write_calibration(tmp_path / "too-large-calibration.json", evidence)
    with pytest.raises(SchemaError, match="canonical payload bound"):
        load_process_broker_timeout_calibration(artifact)


def test_loader_rechecks_mode_and_size_after_open_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_calibration(tmp_path / "raced.json", _evidence())
    original_open = timeout_module.os.open
    raced = False

    def chmod_and_grow_before_open(path, flags, *args, **kwargs):
        nonlocal raced
        if not raced and Path(path) == source:
            raced = True
            source.chmod(0o644)
            writer = original_open(source, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(writer, b" ")
            finally:
                os.close(writer)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(timeout_module.os, "open", chmod_and_grow_before_open)
    with pytest.raises(SchemaError, match="changed during open"):
        load_process_broker_timeout_calibration(source)


def test_immutable_expected_authority_bundle_cross_checks_typed_receipts(
    tmp_path: Path,
) -> None:
    authority, evidence = _authority_artifacts(tmp_path / "valid")
    assert Path(authority.calibration_artifact_path).stat().st_size < (
        MAX_TIMEOUT_AUTHORITY_ARTIFACT_BYTES
    )
    assert load_authority_bound_timeout_calibration(authority) == evidence
    assert authority.to_dict()["production_dispatch_authorized"] is False
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
        backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
        sealed_backend_config={
            "schema_version": "table2-process-broker-fixture-backend-v1"
        },
        execution_scope=PROCESS_BROKER_MEASURED_REPLAY_SCOPE,
        timeout_calibration_artifact_path=authority.calibration_artifact_path,
        timeout_calibration_expected_authority=authority,
    )
    with broker:
        assert broker.receipt.timeout_calibration_replay_only is True
        assert (
            broker.receipt.immutable_timeout_authority_bundle_validated is True
        )
        assert (
            broker.receipt.external_timeout_authority_cross_binding_present
            is False
        )

    mismatched, _ = _authority_artifacts(
        tmp_path / "mismatch", membership_mismatch=True
    )
    with pytest.raises(SchemaError, match="not a member of the safe-probe"):
        load_authority_bound_timeout_calibration(mismatched)


def test_true_evaluation_validates_bundle_but_stays_closed_without_external_authority(
    tmp_path: Path,
) -> None:
    authority, _ = _authority_artifacts(tmp_path / "authority")
    with pytest.raises(SchemaError, match="external authority cross-binds"):
        ProcessIsolatedBroker(
            repository_root=ROOT,
            backend_entrypoint=FIXTURE_BACKEND_ENTRYPOINT,
            backend_source_relative_path=FIXTURE_BACKEND_SOURCE,
            sealed_backend_config={
                "schema_version": "table2-process-broker-fixture-backend-v1"
            },
            execution_scope=PROCESS_BROKER_EVALUATION_SCOPE,
            timeout_calibration_artifact_path=authority.calibration_artifact_path,
            timeout_calibration_expected_authority=authority,
        )
