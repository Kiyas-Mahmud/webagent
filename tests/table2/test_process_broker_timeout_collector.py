from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_file, sha256_json
from web_agent.eval.table2 import process_broker_timeout_collector as collector
from web_agent.eval.table2.process_broker_timeout import (
    CALIBRATION_OPERATION_CLASSES,
    HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION,
    SAFE_PROBE_MANIFEST_SCHEMA_VERSION,
    load_authority_bound_timeout_calibration,
    ProcessBrokerTimeoutExpectedAuthority,
)
from web_agent.eval.table2.process_broker_timeout_collector import (
    COLLECTION_FAILURE_FILENAME,
    COLLECTION_MANIFEST_FILENAME,
    TIMEOUT_COLLECTOR_ENTRYPOINT,
    TIMEOUT_COLLECTOR_INPUT_SCHEMA_VERSION,
    TIMEOUT_COLLECTOR_INPUT_STATUS,
    collect_from_frozen_timeout_collector_input,
)


ROOT = Path(__file__).resolve().parents[2]
ACTION_TYPES = ("CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY")
_ACTIVE_HARNESS = None


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Harness:
    def __init__(
        self,
        *,
        fingerprint_drift: bool = False,
        return_data: bool = False,
        tamper_path: Path | None = None,
        cleanup_error: bool = False,
    ) -> None:
        self.fingerprint_drift = fingerprint_drift
        self.return_data = return_data
        self.tamper_path = tamper_path
        self.cleanup_error = cleanup_error
        self.operations: list[tuple[int, str, str | None]] = []
        self.cleanup: list[tuple[int, str, str | None]] = []
        self._tampered = False

    def run_registered_operation(self, request, nested_operation):
        self.operations.append(
            (request.task_order, request.operation_class, request.action_type)
        )
        if self.tamper_path is not None and not self._tampered:
            self._tampered = True
            self.tamper_path.chmod(0o644)
            with self.tamper_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
        if nested_operation is not None:
            nested_operation()
        if self.return_data:
            return {"score": 1}
        return None

    def reset_and_fingerprint(self, request):
        label = f"reset:{request.task_id}"
        if self.fingerprint_drift and request.checkpoint == "post_probe_reset":
            label += ":changed"
        return _sha(label)

    def cleanup_after_collection_failure(self, request):
        self.cleanup.append(
            (request.task_order, request.task_id, request.failed_operation_class)
        )
        if self.cleanup_error:
            raise RuntimeError("fixture cleanup failure")
        return None


def create_timeout_probe_harness():
    assert _ACTIVE_HARNESS is not None
    return _ACTIVE_HARNESS


def _write_readonly(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o444)
    return path


def _registry() -> dict:
    task_ids = [f"webarena.{index}" for index in range(50)]
    upstream = list(range(50))
    return {
        "manifest_id": "table2-webarena-public-dev-50-collector-test-v1",
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


def _configuration() -> dict:
    return {
        "schema_version": "table2.runtime.v1",
        "record_type": "BrowserGymRuntimeConfiguration",
        "viewport_width": 1280,
        "viewport_height": 720,
        "headless": True,
        "slow_mo_ms": 0,
        "playwright_timeout_ms": 10_000,
        "locale": "en-US",
        "timezone_id": "UTC",
        "tags_to_mark": "all",
        "pre_observation_delay_seconds": 0.5,
        "network_idle_required": True,
        "page_settle_timeout_seconds": 10.0,
        "video_recording_enabled": False,
        "trace_recording_enabled": False,
    }


def _host() -> dict:
    return {
        "system": "Linux",
        "release": "collector-test",
        "machine": "x86_64",
        "python_version": "3.11.0",
        "python_executable_sha256": _sha("python"),
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


def _safe_manifest() -> dict:
    ordinary = []
    recovery = []
    failed = []
    registry = _registry()
    for task_order, task_id in enumerate(registry["ordered_task_ids"]):
        for phase, destination, offset in (
            ("ordinary", ordinary, 0),
            ("recovery", recovery, 3),
        ):
            action_type = ACTION_TYPES[(task_order + offset) % len(ACTION_TYPES)]
            action_sha = _sha(f"{phase}-action:{task_order}")
            approval = {
                "schema_version": "table2-process-broker-timeout-action-safety-v1",
                "record_type": "ProcessBrokerTimeoutActionSafetyApproval",
                "task_id": task_id,
                "phase": phase,
                "action_type": action_type,
                "probe_action_sha256": action_sha,
                "approved": True,
                "persistent_environment_mutation": False,
                "destructive": False,
                "external_side_effect": False,
                "credential_state_change": False,
            }
            destination.append(
                {
                    "task_order": task_order,
                    "task_id": task_id,
                    "phase": phase,
                    "action_type": action_type,
                    "probe_action_sha256": action_sha,
                    "safety_approval": approval,
                }
            )
        failure_sha = _sha(f"failure:{task_order}")
        failed.append(
            {
                "task_order": task_order,
                "task_id": task_id,
                "failure_injection_sha256": failure_sha,
                "safety_approval": {
                    "schema_version": (
                        "table2-process-broker-timeout-failure-safety-v1"
                    ),
                    "record_type": "ProcessBrokerTimeoutFailureSafetyApproval",
                    "task_id": task_id,
                    "failure_injection_sha256": failure_sha,
                    "failure_path": (
                        "SAFE_NON_PERSISTENT_INJECTED_RUNTIME_FAILURE"
                    ),
                    "approved": True,
                    "persistent_environment_mutation": False,
                    "destructive": False,
                    "external_side_effect": False,
                    "credential_state_change": False,
                    "reward_or_evaluator_access": False,
                },
            }
        )
    return {
        "schema_version": SAFE_PROBE_MANIFEST_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutSafeProbeManifest",
        "manifest_id": "collector-safe-probes-v1",
        "task_manifest_sha256": registry["manifest_sha256"],
        "ordered_task_ids": registry["ordered_task_ids"],
        "ordinary_probes": ordinary,
        "recovery_probes": recovery,
        "failed_runtime_cleanup_probes": failed,
    }


def _source_receipt() -> dict:
    rows = []
    for relative in (
        "src/web_agent/eval/table2/common.py",
        "src/web_agent/eval/table2/process_broker_timeout.py",
        "src/web_agent/eval/table2/process_broker_timeout_collector.py",
        "tests/table2/test_process_broker_timeout_collector.py",
    ):
        rows.append(
            {"relative_path": relative, "sha256": sha256_file(ROOT / relative)}
        )
    rows.sort(key=lambda row: row["relative_path"])
    return {
        "schema_version": HARNESS_SOURCE_RECEIPT_SCHEMA_VERSION,
        "record_type": "ProcessBrokerTimeoutHarnessSourceReceipt",
        "receipt_id": "collector-test-harness-v1",
        "collector_entrypoint": TIMEOUT_COLLECTOR_ENTRYPOINT,
        "collector_source_relative_path": (
            "src/web_agent/eval/table2/process_broker_timeout_collector.py"
        ),
        "source_files": rows,
        "source_set_sha256": sha256_json(rows),
        "measurement_clock": "time.monotonic_ns",
        "raw_monotonic_intervals_recorded": True,
        "reward_interface_imported": False,
        "evaluator_interface_imported": False,
        "oracle_interface_imported": False,
        "outcome_labels_accepted": False,
    }


def _input(safe_sha: str, source_sha: str) -> dict:
    return {
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
        "safe_probe_manifest_content_sha256": safe_sha,
        "measurement_harness_source_receipt_content_sha256": source_sha,
        "harness_factory_entrypoint": f"{__name__}:create_timeout_probe_harness",
        "deployment_topology": "SINGLE_DGX_HOST",
        "deployment_preflight_content_sha256": _sha("preflight"),
        "semantic_dependency_lock_sha256": _sha("dependencies"),
        "service_url_map_content_sha256": _sha("service-map"),
        "browser_host_identity": _host(),
        "browsergym_runtime_configuration": _configuration(),
        "task_registry": _registry(),
        "campaign_budgets": _budgets(),
    }


def _artifacts(tmp_path: Path):
    safe = _safe_manifest()
    source = _source_receipt()
    frozen_input = _input(sha256_json(safe), sha256_json(source))
    return (
        _write_readonly(tmp_path / "safe.json", safe),
        _write_readonly(tmp_path / "source.json", source),
        _write_readonly(tmp_path / "input.json", frozen_input),
        frozen_input,
    )


@pytest.fixture(autouse=True)
def _reset_harness():
    global _ACTIVE_HARNESS
    _ACTIVE_HARNESS = None
    yield
    _ACTIVE_HARNESS = None


def _run(tmp_path: Path, harness: _Harness):
    global _ACTIVE_HARNESS
    safe_path, source_path, input_path, frozen_input = _artifacts(tmp_path)
    _ACTIVE_HARNESS = harness
    counter = iter(range(1_000_000, 100_000_000_000, 1_000_000))
    original_time = collector.time
    collector.time = SimpleNamespace(monotonic_ns=lambda: next(counter))
    try:
        result = collect_from_frozen_timeout_collector_input(
            repository_root=ROOT,
            output_directory=tmp_path / "output",
            collector_input_path=input_path,
            expected_collector_input_sha256=sha256_json(frozen_input),
            safe_probe_manifest_path=safe_path,
            measurement_harness_source_receipt_path=source_path,
        )
    finally:
        collector.time = original_time
    return result, safe_path, source_path, input_path


def _unseal(path: Path) -> None:
    if path.exists():
        path.chmod(0o755)


def test_collector_emits_complete_non_authorizing_bundle(tmp_path: Path) -> None:
    harness = _Harness()
    result, _safe, _source, _input_path = _run(tmp_path, harness)
    output = tmp_path / "output"
    assert result["status"] == "MEASURED_COMPLETE_NON_AUTHORIZING"
    assert result["paper_table_status"] == "N/R"
    assert result["production_dispatch_authorized"] is False
    assert result["live_campaign_authority"] is False
    assert result["measurement_count"] == 50 * len(CALIBRATION_OPERATION_CLASSES)
    assert (output / COLLECTION_MANIFEST_FILENAME).is_file()
    calibration = json.loads(
        (output / "process_broker_timeout_calibration.json").read_text()
    )
    assert calibration["status"] == "MEASURED_COMPLETE_NON_AUTHORIZING"
    assert calibration["measurement_clock"] == "time.monotonic_ns"
    assert len(calibration["measurements"]) == 800
    for phase_operations in (
        {
            row[2]
            for row in harness.operations
            if row[1] == "runtime_execute_action"
        },
        {
            row[2]
            for row in harness.operations
            if row[1] == "runtime_execute_recovery"
        },
    ):
        assert phase_operations == set(ACTION_TYPES)
    assert sum(
        operation == "control_shutdown_after_failed_runtime"
        for _, operation, _ in harness.operations
    ) == 50
    assert harness.cleanup == []
    authority_value = json.loads(
        (output / "local_expected_authority.json").read_text()
    )
    assert authority_value["collection_manifest_path"] == str(
        output / COLLECTION_MANIFEST_FILENAME
    )
    assert authority_value["collection_manifest_content_sha256"] == sha256_json(
        result
    )
    assert {row["relative_path"] for row in result["artifact_files"]} == {
        "process_broker_timeout_calibration.json",
        "safe_probe_manifest.json",
        "measurement_harness_source_receipt.json",
        "measurement_source_receipt.json",
        "non_persistence_audit_receipt.json",
    }
    authority = ProcessBrokerTimeoutExpectedAuthority(**authority_value)
    assert load_authority_bound_timeout_calibration(authority) == calibration
    _unseal(output)


def test_collector_refuses_existing_output_without_running_harness(
    tmp_path: Path,
) -> None:
    harness = _Harness()
    result, safe, source, frozen_path = _run(tmp_path, harness)
    calls = len(harness.operations)
    frozen_input = json.loads(frozen_path.read_text())
    with pytest.raises(SchemaError, match="refuses rerun or overwrite"):
        collect_from_frozen_timeout_collector_input(
            repository_root=ROOT,
            output_directory=tmp_path / "output",
            collector_input_path=frozen_path,
            expected_collector_input_sha256=sha256_json(frozen_input),
            safe_probe_manifest_path=safe,
            measurement_harness_source_receipt_path=source,
        )
    assert result["rerun_at_same_output_path_permitted"] is False
    assert len(harness.operations) == calls
    _unseal(tmp_path / "output")


@pytest.mark.parametrize("mode", ["returned_data", "fingerprint_drift"])
def test_collector_rejects_forbidden_harness_data_or_persistence(
    tmp_path: Path, mode: str
) -> None:
    harness = _Harness(
        return_data=mode == "returned_data",
        fingerprint_drift=mode == "fingerprint_drift",
    )
    with pytest.raises(SchemaError):
        _run(tmp_path, harness)
    output = tmp_path / "output"
    assert (output / COLLECTION_FAILURE_FILENAME).is_file()
    assert not (output / "process_broker_timeout_calibration.json").exists()
    assert harness.cleanup
    failure = json.loads((output / COLLECTION_FAILURE_FILENAME).read_text())
    assert failure["cleanup_attempted"] is True
    assert failure["cleanup_succeeded"] is True
    assert failure["cleanup_error_class"] is None
    _unseal(output)


def test_collector_records_cleanup_error_class_without_masking_failure(
    tmp_path: Path,
) -> None:
    harness = _Harness(return_data=True, cleanup_error=True)
    with pytest.raises(SchemaError, match="returned forbidden data"):
        _run(tmp_path, harness)
    output = tmp_path / "output"
    failure = json.loads((output / COLLECTION_FAILURE_FILENAME).read_text())
    assert failure["cleanup_attempted"] is True
    assert failure["cleanup_succeeded"] is False
    assert failure["cleanup_error_class"] == "RuntimeError"
    _unseal(output)


def test_post_publication_failure_marker_makes_complete_bundle_unloadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_loader = collector.load_authority_bound_timeout_calibration
    calls = 0

    def fail_second_validation(authority):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SchemaError("injected post-publication validation failure")
        return real_loader(authority)

    monkeypatch.setattr(
        collector,
        "load_authority_bound_timeout_calibration",
        fail_second_validation,
    )
    with pytest.raises(SchemaError, match="post-publication"):
        _run(tmp_path, _Harness())
    output = tmp_path / "output"
    assert (output / COLLECTION_MANIFEST_FILENAME).is_file()
    assert (output / COLLECTION_FAILURE_FILENAME).is_file()
    authority = ProcessBrokerTimeoutExpectedAuthority(
        **json.loads((output / "local_expected_authority.json").read_text())
    )
    with pytest.raises(SchemaError, match="failed timeout collection"):
        load_authority_bound_timeout_calibration(authority)
    _unseal(output)


def test_collector_detects_safe_manifest_tamper_and_seals_failure(
    tmp_path: Path,
) -> None:
    safe_path, source_path, input_path, frozen_input = _artifacts(tmp_path)
    global _ACTIVE_HARNESS
    harness = _Harness(tamper_path=safe_path)
    _ACTIVE_HARNESS = harness
    counter = iter(range(1_000_000, 100_000_000_000, 1_000_000))
    original_time = collector.time
    collector.time = SimpleNamespace(monotonic_ns=lambda: next(counter))
    try:
        with pytest.raises(SchemaError, match="changed during collection"):
            collect_from_frozen_timeout_collector_input(
                repository_root=ROOT,
                output_directory=tmp_path / "output",
                collector_input_path=input_path,
                expected_collector_input_sha256=sha256_json(frozen_input),
                safe_probe_manifest_path=safe_path,
                measurement_harness_source_receipt_path=source_path,
            )
    finally:
        collector.time = original_time
    output = tmp_path / "output"
    assert (output / COLLECTION_FAILURE_FILENAME).is_file()
    assert not (output / "process_broker_timeout_calibration.json").exists()
    _unseal(output)


def test_collector_rejects_incomplete_six_action_manifest_before_output(
    tmp_path: Path,
) -> None:
    safe = _safe_manifest()
    for row in safe["ordinary_probes"]:
        if row["action_type"] == "SELECT":
            row["action_type"] = "CLICK"
            row["safety_approval"]["action_type"] = "CLICK"
    source = _source_receipt()
    frozen_input = _input(sha256_json(safe), sha256_json(source))
    safe_path = _write_readonly(tmp_path / "safe.json", safe)
    source_path = _write_readonly(tmp_path / "source.json", source)
    input_path = _write_readonly(tmp_path / "input.json", frozen_input)
    with pytest.raises(SchemaError, match="lacks all six actions"):
        collect_from_frozen_timeout_collector_input(
            repository_root=ROOT,
            output_directory=tmp_path / "output",
            collector_input_path=input_path,
            expected_collector_input_sha256=sha256_json(frozen_input),
            safe_probe_manifest_path=safe_path,
            measurement_harness_source_receipt_path=source_path,
        )
    assert not (tmp_path / "output").exists()


def test_cli_has_no_credential_or_timeout_override_flags() -> None:
    source = (ROOT / "scripts/collect_table2_process_broker_timeout.py").read_text()
    assert "--credential" not in source
    assert "--timeout" not in source
    assert "--task" not in source
    assert "--action" not in source
    assert "--reward" not in source
    assert "--oracle" not in source
