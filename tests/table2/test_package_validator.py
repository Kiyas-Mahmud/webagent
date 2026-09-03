from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest
import yaml

from web_agent.benchmarks.webarena import WebArenaInfrastructureRule
from web_agent.config import load_config

from web_agent.eval.table2.campaign import CampaignRunner, _path_id
from web_agent.eval.table2.common import (
    SCHEMA_VERSION,
    SchemaError,
    Table2Error,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.package_validator import (
    FINAL_READY_STATUS,
    ValidationReport,
    _artifact_payload_descriptor,
    _expected_runner_runtime_identity,
    _processor_contract_from_payload,
    _validate_duplicate_audit_bindings,
    _validate_environment_manifest,
    _validate_model_memory_source_bindings,
    _publication_status,
    append_campaign_ledger_event,
    freeze_campaign,
    load_yaml,
    load_selected_analysis_records,
    validate_manual_adjudication_completion,
    validate_campaign,
)
from web_agent.eval.table2.execution_guard import (
    EVALUATION_RUNNER_SCOPE,
    InfrastructureInvalidError,
    RUNNER_ATTESTATION_SCHEMA_VERSION,
)
from web_agent.eval.table2 import handoff as handoff_preparer
from web_agent.eval.table2 import selection_evidence as selection_evidence_module
from web_agent.eval.table2.live_deployment import (
    LIVE_DEPLOYMENT_BINDING_FIELD,
    stage_pc01_live_deployment_package,
)
from tests.table2.test_live_deployment import (
    _manifest as _valid_live_deployment_manifest,
)
from web_agent.eval.table2.task_interface_audit import (
    TASK_INTERFACE_AUDIT_RELATIVE_PATH,
    build_webarena_task_interface_audit,
)
from web_agent.eval.table2.webarena_export import (
    PINNED_BROWSERGYM_WEBARENA_VERSION,
    PINNED_TASK_DEFINITION_VERSION,
    build_public_pilot_task_export,
)
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_PACKAGES,
    run_webarena_host_preflight,
)
from web_agent.eval.table2.webarena_preflight_binding import (
    PREFLIGHT_ARTIFACT_RELATIVE_PATH,
    PREFLIGHT_BINDING_FIELD,
    PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
    build_deployment_preflight_binding,
)
from web_agent.eval.table2.split_deployment_preflight import SINGLE_HOST_TOPOLOGY
from web_agent.eval.table2.retrieval_metrics import compute_retrieval_diagnostics
from web_agent.eval.table2.schedule import (
    SYSTEM_IDS,
    discover_block_attempt_directories,
    resolve_block_attempts,
)
from web_agent.eval.table2.sealed_verifier import (
    SealedVerifierSink,
    SealedVerifierWriter,
    assert_no_verifier_evidence,
    verify_sealed_stream,
)
from web_agent.eval.table2.summary import (
    _write_aggregate_hashes,
    _write_campaign_evidence_manifest,
    build_manual_audit_selection,
    summarize_campaign,
)
from web_agent.eval.table2.selection_evidence import (
    PC01_PROVISIONAL_SELECTION_MODE,
    THREE_CANDIDATE_FINAL_SELECTION_MODE,
    stage_selection_evidence,
    validate_selection_evidence,
)
from web_agent.memory.builder import build_frozen_store
from web_agent.memory.calibration_builder import build_calibration_evidence
from web_agent.memory.eligibility import (
    EligibleMemoryCandidate,
    EligibilitySelection,
    ProvenanceManifest,
)
from web_agent.memory.verification import (
    FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
    RECOVERY_VERIFICATION_SCHEMA_VERSION,
    recovery_action_evidence_sha256,
    verification_bundle_sha256,
)
from web_agent.runtime.contracts import VerifierReceiptBinding
from web_agent.runtime.duplicate_audit import (
    JointDuplicateClusterNamespace,
    canonical_task_audit_record,
)
from web_agent.train.selection import (
    ALL_GATES_THEN_OUTCOME_RULE,
    controlled_quality_gates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = REPOSITORY_ROOT / "configs/eval/table2/pilot_webarena.yaml"
PROTOCOL = REPOSITORY_ROOT / "configs/eval/table2/protocol.yaml"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
JOINT_DUPLICATE_NAMESPACE = {
    "schema_version": "table2-joint-duplicate-cluster-namespace-v1",
    "namespace_id": "fixture-gold-v2.8-webarena-joint-v1",
    "audit_tool_id": "fixture-joint-duplicate-audit",
    "audit_tool_version": "v1",
    "audit_tool_config_sha256": SHA_A,
    "audit_tool_source_sha256": SHA_C,
}
WEBARENA_TASK_URL_MAP = {
    "__GITLAB__": "http://gitlab.example.test",
    "__MAP__": "http://map.example.test",
    "__REDDIT__": "http://reddit.example.test",
    "__SHOPPING__": "http://shopping.example.test",
    "__SHOPPING_ADMIN__": "http://admin.example.test",
}
WEBARENA_SERVICE_URL_MAP = {
    "WA_SHOPPING": WEBARENA_TASK_URL_MAP["__SHOPPING__"],
    "WA_SHOPPING_ADMIN": WEBARENA_TASK_URL_MAP["__SHOPPING_ADMIN__"],
    "WA_REDDIT": WEBARENA_TASK_URL_MAP["__REDDIT__"],
    "WA_GITLAB": WEBARENA_TASK_URL_MAP["__GITLAB__"],
    "WA_WIKIPEDIA": "http://wikipedia.example.test",
    "WA_MAP": WEBARENA_TASK_URL_MAP["__MAP__"],
    "WA_HOMEPAGE": "http://homepage.example.test",
}


def _memory_verification_evidence(
    *,
    source_id: str,
    recovery_id: str,
    task_id: str,
    episode_id: str,
    strategy: str,
    action: str,
) -> dict[str, Any]:
    transition = {
        "recovery_sample_id": recovery_id,
        "recovery_strategy": strategy,
        "executed_recovery_action": action,
        "recovery_action_value": "",
        "failure_state": f"failure-{source_id}",
        "post_recovery_state": f"recovered-{source_id}",
    }
    recovery = {
        "schema_version": RECOVERY_VERIFICATION_SCHEMA_VERSION,
        "authority_type": "verifier",
        "authority_id": "fixture-recovery-verifier",
        "authority_version": "v1",
        "independent_verification": True,
        "source_sample_id": source_id,
        "recovery_sample_id": recovery_id,
        "canonical_task_id": task_id,
        "episode_id": episode_id,
        "pre_recovery_state_sha256": sha256_json(transition["failure_state"]),
        "executed_recovery_action_sha256": recovery_action_evidence_sha256(
            transition
        ),
        "post_recovery_state_sha256": sha256_json(
            transition["post_recovery_state"]
        ),
        "verified_recovery_success": True,
    }
    recovery["evidence_sha256"] = sha256_json(recovery)
    final_task = {
        "schema_version": FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
        "authority_type": "reviewer",
        "authority_id": "fixture-final-reviewer",
        "authority_version": "rubric-v1",
        "independent_verification": True,
        "source_sample_id": source_id,
        "canonical_task_id": task_id,
        "episode_id": episode_id,
        "task_specification_sha256": SHA_A,
        "terminal_state_sha256": SHA_B,
        "terminal_verifier_output_sha256": SHA_C,
        "verified_final_task_success": True,
    }
    final_task["evidence_sha256"] = sha256_json(final_task)
    return {
        "recovery_verification": recovery,
        "final_task_verification": final_task,
        "verification_evidence_sha256": verification_bundle_sha256(
            source_sample_id=source_id,
            recovery_sample_id=recovery_id,
            canonical_task_id=task_id,
            episode_id=episode_id,
            recovery_evidence_sha256=recovery["evidence_sha256"],
            final_task_evidence_sha256=final_task["evidence_sha256"],
        ),
    }


def _fixture_sealed_evaluator_factory(*_: Any, **__: Any) -> Any:
    """Attested fixture entrypoint; production execution is outside this test."""

    raise AssertionError("fixture evaluator factory must not execute during freeze tests")


def _fixture_train_corpus_binding() -> str:
    value = {
        "records_sha256": SHA_B,
        "provenance_manifest_sha256": SHA_B,
        "duplicate_cluster_namespace": JOINT_DUPLICATE_NAMESPACE,
        "duplicate_cluster_namespace_sha256": sha256_json(
            JOINT_DUPLICATE_NAMESPACE
        ),
    }
    return sha256_json(value)


def _fixture_resolved_candidate_config(model_id: str) -> dict[str, Any]:
    value = deepcopy(
        load_config(REPOSITORY_ROOT / f"configs/backbones/{model_id}.yaml")
    )
    value["name"] = f"{value['name']}_FULL_SEED42"
    value["seeds"] = [42]
    value["train"]["epochs"] = 10
    value["train"]["metrics_csv"] = f"/frozen/{model_id}/epoch_metrics.csv"
    value["train"]["keep_top_k"] = 10
    value["output"]["checkpoint_dir"] = f"/frozen/{model_id}/checkpoints"
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _campaign_config(
    tmp_path: Path,
    *,
    mode: str,
    duplicate_audit_manifest: Path | None = None,
    resolved_task_snapshot: Path | None = None,
) -> Path:
    value = yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8"))
    value["campaign_mode"] = mode
    if duplicate_audit_manifest is not None:
        value["duplicate_audit_manifest"] = str(duplicate_audit_manifest)
        for key, source_name in (("recovery_scenarios", "recovery_scenarios.json"),):
            source = REPOSITORY_ROOT / "benchmarks/table2/pilot" / source_name
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["duplicate_audit_manifest"] = str(duplicate_audit_manifest)
            value[key] = str(_write_json(tmp_path / source_name, payload))
    if resolved_task_snapshot is not None:
        value["resolved_task_snapshot"] = str(resolved_task_snapshot)
        value["runtime_integration_entrypoint"] = (
            "tests.table2.test_package_validator:_NestedEvidenceRunner"
        )
        value["runtime_integration_source"] = str(
            Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
        )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"campaign-{mode}.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _verified_duplicate_audit(path: Path, *, task_manifest: Path) -> Path:
    value = json.loads(
        (
            REPOSITORY_ROOT
            / "benchmarks/table2/pilot/duplicate_audit_manifest.json"
        ).read_text(encoding="utf-8")
    )
    value["manifest_state"] = "FROZEN_REGISTRATION"
    value["normal_task_evidence_status"] = "VERIFIED"
    value["normal_task_runtime_policy"] = "VERIFIED_NONEMPTY_CLUSTERS_REQUIRED"
    task_manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    value["duplicate_cluster_namespace"] = dict(JOINT_DUPLICATE_NAMESPACE)
    namespace = JointDuplicateClusterNamespace.from_mapping(
        JOINT_DUPLICATE_NAMESPACE,
        require_hashes=True,
    )
    content_hashes = {
        str(task["task_id"]): sha256_json(task) for task in task_manifest["tasks"]
    }
    corpus_binding = _fixture_train_corpus_binding()
    for row in value["entries"]:
        if row["task_partition"] != "normal":
            continue
        task_id = str(row["task_id"])
        clusters = [f"verified:{task_id}"]
        audit_record = canonical_task_audit_record(
            task_id=task_id,
            content_sha256=content_hashes[task_id],
            train_corpus_manifest_sha256=corpus_binding,
            cluster_ids=clusters,
            namespace=namespace,
        )
        row.update(
            {
                "status": "VERIFIED",
                "cluster_ids": clusters,
                "content_sha256": content_hashes[task_id],
                "train_corpus_manifest_sha256": corpus_binding,
                "cluster_namespace_id": namespace.namespace_id,
                "audit_tool_id": namespace.audit_tool_id,
                "audit_tool_version": namespace.audit_tool_version,
                "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
                "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
                "audit_record": audit_record,
                "evidence_sha256": sha256_json(audit_record),
            }
        )
    return _write_json(path, value)


def _valid_deployment_preflight(root: Path) -> dict[str, Any]:
    url_map_path = _write_json(
        root / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
        WEBARENA_SERVICE_URL_MAP,
    )
    evidence = run_webarena_host_preflight(
        service_url_map=WEBARENA_SERVICE_URL_MAP,
        version_getter=lambda distribution: PINNED_WEBARENA_PACKAGES[distribution],
        browser_probe=lambda: {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture-1",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
            "screenshot_sha256": SHA_A,
        },
        service_probe=lambda _url: {"status": "PASS", "http_status": 200},
        live_reset_probe=lambda index: {
            "status": "PASS",
            "task_index": index,
            "seed": 42,
            "goal_sha256": SHA_B,
            "current_url_sha256": SHA_C,
            "screenshot_shape": [720, 1280, 3],
            "observation_keys": ["goal", "screenshot", "url"],
            "action_taken": False,
            "reward_read": False,
            "evaluator_output_read": False,
        },
        live_reset_task_index=0,
    )
    evidence_path = _write_json(
        root / PREFLIGHT_ARTIFACT_RELATIVE_PATH,
        evidence,
    )
    return build_deployment_preflight_binding(
        evidence_path=evidence_path,
        service_url_map_path=url_map_path,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        expected_live_reset_task_index=0,
    )


def _valid_environment(path: Path) -> Path:
    dependency_lock = path.parent / "dependency.lock"
    dependency_lock.parent.mkdir(parents=True, exist_ok=True)
    dependency_lock.write_bytes(b"fixture-dependency-lock-v1\n")
    infrastructure_rules = [
        {
            "operation": operation,
            "exception_type": "builtins.ConnectionError",
            "reason_code": reason,
            "failure_class": failure_class,
        }
        for operation, reason, failure_class in (
            (
                "environment_factory",
                "BENCHMARK_SERVICE_UNAVAILABLE",
                "benchmark_service_connection_failure",
            ),
            ("reset", "ENVIRONMENT_RESET_FAILED", "task_reset_connection_failure"),
            ("step", "BROWSER_CONTROLLER_DISCONNECTED", "browser_controller_loss"),
            (
                "screenshot_bytes_reset",
                "BROWSER_CONTROLLER_DISCONNECTED",
                "browser_screenshot_controller_loss_reset",
            ),
            (
                "screenshot_bytes_pre_action",
                "BROWSER_CONTROLLER_DISCONNECTED",
                "browser_screenshot_controller_loss_pre_action",
            ),
            (
                "screenshot_bytes_post_action",
                "BROWSER_CONTROLLER_DISCONNECTED",
                "browser_screenshot_controller_loss_post_action",
            ),
            (
                "screenshot_bytes_post_recovery",
                "BROWSER_CONTROLLER_DISCONNECTED",
                "browser_screenshot_controller_loss_post_recovery",
            ),
        )
    ]
    typed_rule_payloads = [
        WebArenaInfrastructureRule(**rule).to_dict()
        for rule in infrastructure_rules
    ]
    value = {
            "schema_version": "table2-environment-v2",
            "benchmark": "webarena",
            "benchmark_version": "fixture-1",
            "benchmark_repository": "fixture/webarena",
            "benchmark_revision": "fixture-revision-1",
            "benchmark_license": "fixture-license",
            "task_definition_version": "fixture-task-v1",
            "browser": "chromium",
            "browser_version": "fixture-1",
            "playwright_version": PINNED_WEBARENA_PACKAGES["playwright"],
            "operating_system": "fixture-linux",
            "controller_id": "fixture-controller",
            "controller_version": "fixture-1",
            "environment_adapter_id": "fixture-adapter",
            "environment_adapter_version": "fixture-adapter-v1",
            "container_digest": "sha256:fixture",
            "hardware": "fixture-host",
            "gpu": "fixture-gpu",
            "driver_version": "fixture-driver",
            "captured_at_utc": "2026-08-31T00:00:00+00:00",
            "dependency_lock_sha256": sha256_file(dependency_lock),
            "dependency_lock_relative_path": "dependency.lock",
            "locked_mount_preflight": {
                "schema_version": "table2-locked-mount-preflight-v1",
                "configured_path": "locked_benchmark_mount",
                "check_algorithm": "lexists_ismount_access_r_ok_v1",
                "path_exists": False,
                "is_mounted": False,
                "readable": False,
            },
            "reset": {
                "implementation_id": "fixture-reset",
                "implementation_version": "v1",
                "deterministic_start_state": True,
            },
            "environment_state_digester": {
                "digester_id": "fixture-webarena-state",
                "digester_version": "v1",
            },
            "infrastructure_classifier": {
                "classifier_id": "fixture-webarena-infrastructure-taxonomy",
                "classifier_version": "v1",
                "rules": infrastructure_rules,
                "rules_sha256": sha256_json(typed_rule_payloads),
            },
            "page_settle_policy": {
                "policy_id": "fixture-settle-v1",
                "network_idle_required": True,
                "settle_timeout_seconds": 5.0,
            },
            "manual_rescue_guard": {
                "guard_id": "fixture-manual-rescue-guard",
                "guard_version": "v1",
                "evidence_mode": "exclusive_controller_input_audit_v1",
            },
            "model_call_timeout_seconds": 30.0,
            "evaluator": {
                "evaluator_id": "fixture-evaluator",
                "evaluator_version": "v1",
                "entrypoint": (
                    "tests.table2.test_package_validator:"
                    "_fixture_sealed_evaluator_factory"
                ),
                "source_relative_path": str(
                    Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
                ),
                "source_sha256": sha256_file(Path(__file__).resolve()),
                "oracle_rules_sha256": sha256_file(
                    REPOSITORY_ROOT
                    / "benchmarks/table2/pilot/recovery_oracle_rules.json"
                ),
                "model_based": False,
                "prompt_sha256": None,
            },
            "execution_order_algorithm": "sha256_offset_permutation_cycle_v1",
            "failure_taxonomy": [
                "BROWSER_CONTROLLER_ERROR",
                "TASK_RESET_FAILURE",
                "NETWORK_HTTP_INFRASTRUCTURE_FAILURE",
                "SITE_UNAVAILABLE",
                "CAPTCHA_LOGIN_CREDENTIAL_GATE",
                "EVALUATOR_FAILURE",
            ],
            "safety_policy_id": "fixture-safety",
            "safety_policy_version": "v1",
            "credential_policy": "fixture-credentials",
            "destructive_action_policy": "fixture-no-destructive-actions",
            "element_resolution_policy": "fixture-element-resolution",
            "invalid_action_policy": "fixture-invalid-action",
            "invalid_task_policy": "fixture-invalid-task",
            "screenshot_timing_policy": "fixture-screenshot-timing",
            "viewport": {
                "width": 1280,
                "height": 720,
                "device_scale_factor": 1,
            },
    }
    value[PREFLIGHT_BINDING_FIELD] = _valid_deployment_preflight(path.parent)
    live_evidence_root = path.parent / "measured-live-deployment"
    live_manifest = _write_json(
        live_evidence_root / "deployment.json",
        _valid_live_deployment_manifest(live_evidence_root),
    )
    live_deployment = stage_pc01_live_deployment_package(
        manifest_path=live_manifest,
        evidence_root=live_evidence_root,
        repository_root=REPOSITORY_ROOT,
        destination_artifact_root=path.parent,
    )
    value[LIVE_DEPLOYMENT_BINDING_FIELD] = live_deployment.binding
    return _write_json(path, value)


def _valid_resolved_task_snapshot(
    path: Path, *, environment: Path, duplicate_audit_manifest: Path
) -> Path:
    registry_path = REPOSITORY_ROOT / "benchmarks/table2/pilot/task_manifest.json"
    registry = read_json(registry_path)
    environment_value = read_json(environment)
    evaluator = environment_value["evaluator"]
    evaluator_identity = {
        "evaluator_id": evaluator["evaluator_id"],
        "evaluator_version": evaluator["evaluator_version"],
        "source_relative_path": evaluator["source_relative_path"],
        "source_sha256": evaluator["source_sha256"],
        "oracle_rules_sha256": evaluator["oracle_rules_sha256"],
    }
    tasks = []
    for registry_row in registry["tasks"]:
        task_id = str(registry_row["task_id"])
        index = int(registry_row["upstream_index"])
        row = {
            "task_id": task_id,
            "upstream_index": index,
            "benchmark_task_id": str(index),
            "benchmark_task_version": "fixture-task-v1",
            "instruction": f"Fixture WebArena instruction {index}",
            "start_state": {
                "sites": ["fixture-site"],
                "start_url": f"https://fixture.invalid/task/{index}",
            },
            "task_config": {"task_id": index, "intent": f"fixture intent {index}"},
            "evaluator": {
                "evaluator_id": evaluator["evaluator_id"],
                "evaluator_version": evaluator["evaluator_version"],
                "config": {"task_id": index},
            },
        }
        row["source_content_sha256"] = sha256_json(row)
        tasks.append(row)
    return _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "ResolvedWebArenaTaskSnapshot",
            "snapshot_id": "fixture-resolved-webarena-0-49",
            "benchmark": environment_value["benchmark"],
            "benchmark_version": environment_value["benchmark_version"],
            "registry_manifest_sha256": sha256_file(registry_path),
            "environment_manifest_sha256": sha256_file(environment),
            "evaluator_identity_sha256": sha256_json(evaluator_identity),
            "partition": registry["partition"],
            "locked_test_content": registry["locked_test_content"],
            "final_paper_evaluation_eligible": registry[
                "final_paper_evaluation_eligible"
            ],
            "required_task_count": 50,
            "duplicate_audit_manifest": str(duplicate_audit_manifest),
            "tasks": tasks,
        },
    )


def _valid_model_manifest(path: Path) -> Path:
    payload_root = path.parent / "model-payloads"
    selected_resolved_config = _fixture_resolved_candidate_config(
        "qwen2vl_2b_gold_v2_8_dgx"
    )
    processor_contract = {
        "schema_version": "table2.runtime.v1",
        "record_type": "ProcessorParityContract",
        "processor_class": "FixtureProcessor",
        "processor_revision": "fixture-processor-v1",
        "processor_config_sha256": SHA_A,
        "pre_action_field_mapping": {"state_before": "pixels"},
        "post_action_field_mapping": {"state_after": "pixels_after"},
    }
    payload_files = {
        "selected_checkpoint": ("checkpoint.bin", b"fixture selected checkpoint"),
        "resolved_config": (
            "resolved_config.json",
            canonical_json_bytes(selected_resolved_config),
        ),
        "processor_contract": (
            "processor_contract.json",
            canonical_json_bytes(processor_contract),
        ),
        "e0_resolved_config": ("e0_resolved_config.json", b'{"fixture":"e0"}\n'),
        "e0_processor_contract": (
            "e0_processor_contract.json",
            canonical_json_bytes(processor_contract),
        ),
        "e0_parser": ("fixture_e0_parser.py", b"def parse_action(value):\n    return value\n"),
    }
    artifact_payloads: dict[str, dict[str, Any]] = {}
    for role, (name, payload) in payload_files.items():
        artifact_path = payload_root / role / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(payload)
        artifact_payloads[role] = _artifact_payload_descriptor(artifact_path)
    backbone = payload_root / "e0_backbone"
    backbone.mkdir(parents=True)
    (backbone / "weights.bin").write_bytes(b"fixture unadapted backbone")
    artifact_payloads["e0_backbone"] = _artifact_payload_descriptor(backbone)
    artifact_payloads["e0_parser"].update(
        {"module": "fixture_e0_parser", "attribute": "parse_action"}
    )
    hashes = {
        {
            "selected_checkpoint": "selected_checkpoint_sha256",
            "resolved_config": "resolved_config_sha256",
            "processor_contract": "processor_contract_sha256",
            "e0_backbone": "e0_backbone_sha256",
            "e0_resolved_config": "e0_resolved_config_sha256",
            "e0_processor_contract": "e0_processor_contract_sha256",
            "e0_parser": "e0_parser_sha256",
        }[role]: descriptor["sha256"]
        for role, descriptor in artifact_payloads.items()
    }
    return _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "model_seed": 42,
            "selected_model_id": "qwen2vl_2b_gold_v2_8_dgx",
            "selected_epoch": 1,
            "resolved_config_record_sha256": sha256_json(
                selected_resolved_config
            ),
            **hashes,
            "e0_backbone_id": "fixture-unadapted-backbone",
            "e0_backbone_revision": "fixture-pinned-revision",
            "e0_base_prompt_sha256": sha256_file(
                REPOSITORY_ROOT / "configs/eval/table2/prompts/e0_action_v1.txt"
            ),
            "e0_parser_id": "fixture-e0-parser",
            "e0_parser_version": "v1",
            "e0_parser_module": "fixture_e0_parser",
            "e0_parser_attribute": "parse_action",
            "selection_scope": "validation_only",
            "checkpoint_selection": "validation_only",
            "validation_rows_read": 1,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "artifact_payloads": artifact_payloads,
        },
    )


def _valid_selection_evidence_inputs(
    root: Path,
    *,
    selected_checkpoint_sha256: str,
    selected_resolved_config_sha256: str,
    selection_mode: str = THREE_CANDIDATE_FINAL_SELECTION_MODE,
) -> dict[str, Any]:
    model_ids = (
        "qwen2vl_2b_gold_v2_8_dgx",
        "qwen25vl_7b_gold_v2_8_dgx",
        "internvl35_8b_gold_v2_8_dgx",
    )
    outcomes = {
        "qwen2vl_2b_gold_v2_8_dgx": 0.70,
        "qwen25vl_7b_gold_v2_8_dgx": 0.65,
        "internvl35_8b_gold_v2_8_dgx": 0.60,
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    candidate_runs = []
    comparison_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        registered_config = load_config(
            REPOSITORY_ROOT / f"configs/backbones/{model_id}.yaml"
        )
        candidate_root = root / "selection-source" / model_id / "seed_42"
        checkpoint = f"/frozen/{model_id}/epoch-1.ckpt"
        checkpoint_payload = (
            b"fixture selected checkpoint"
            if model_id == "qwen2vl_2b_gold_v2_8_dgx"
            else f"fixture checkpoint for {model_id}".encode("utf-8")
        )
        checkpoint_path = candidate_root / "full/epoch-1.ckpt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(checkpoint_payload)
        checkpoint_sha = sha256_file(checkpoint_path)
        if (
            model_id == "qwen2vl_2b_gold_v2_8_dgx"
            and checkpoint_sha != selected_checkpoint_sha256
        ):
            raise AssertionError("selected fixture checkpoint hash changed")
        resolved_config = _fixture_resolved_candidate_config(model_id)
        resolved_config_sha256 = sha256_json(resolved_config)
        if (
            model_id == "qwen2vl_2b_gold_v2_8_dgx"
            and resolved_config_sha256 != selected_resolved_config_sha256
        ):
            raise AssertionError("selected fixture resolved-config hash changed")
        contract = {
            "candidate": {
                "model_id": model_id,
                "config": f"configs/backbones/{model_id}.yaml",
                "label": model_id,
                "accepted_prior_mini": model_id == "qwen2vl_2b_gold_v2_8_dgx",
                "prior_mini_evidence": "fixture",
            },
            "git_commit": commit,
            "config_name": registered_config["name"],
            "model_revision": registered_config["backbone"]["revision"],
            "seed": 42,
            "maximum_full_epochs": 10,
            "controlled_mini_epochs": 5,
            "effective_batch_size": 32,
            "checkpoint_selection": ALL_GATES_THEN_OUTCOME_RULE,
            "checkpoint_selection_source": "original_gold_validation_only",
            "supplement_validation_selects_checkpoint": False,
            "train_rows": 24_107,
            "original_validation_rows": 7_861,
            "supplement_validation_rows": 194,
            "test_rows_read": 0,
        }
        metrics = {
            "epoch": 1,
            "outcome_mcc": outcomes[model_id],
            "action_acc": 0.50,
            "needs_recovery_macro_f1": 0.50,
            "needs_recovery_majority_macro_f1": 0.20,
            "strategy_attempted_macro_f1": 0.50,
            "strategy_attempted_majority_macro_f1": 0.20,
            "recovery_outcome_mcc": 0.30,
            "bbox_mean_iou": 0.10,
            "bbox_recall_iou50": 0.10,
            "outcome_ece": 0.10,
            "failure_macro_f1": 0.55,
            "failtype_macro_f1": 0.54,
            "action_macro_f1": 0.53,
            "memory_mcc": 0.31,
        }
        report: dict[str, Any] = {
            "history": [metrics],
            "best_epochs": {"outcome_mcc": {"epoch": 1}},
            "epoch_checkpoints": {"1": checkpoint},
            "status": "PASS",
            "seed": 42,
            "requested_epochs": 10,
            "train_rows": 24_107,
            "val_rows": 7_861,
            "test_rows_read": 0,
            "selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
            "selected_epoch": 1,
            "best_checkpoint": checkpoint,
            "selected_checkpoint_sha256": checkpoint_sha,
            "source_validation": {
                "primary_original_gold": {
                    "rows": 7_861,
                    "checkpoint_selection_source": True,
                    "selected_epoch_metrics": metrics,
                },
                "supplement_retry_abort": {"rows": 194},
            },
            "experiment_control": {
                "config_sha256": resolved_config_sha256,
                "checkpoint_selection_source": "original_gold_validation_only",
                "supplement_validation_selects_checkpoint": False,
                "locked_test_read": False,
            },
        }
        report["quality_gates"] = controlled_quality_gates(
            report, ALL_GATES_THEN_OUTCOME_RULE
        )
        contract_path = _write_json(candidate_root / "run_contract.json", contract)
        report_path = _write_json(candidate_root / "full/report.json", report)
        resolved_config_path = _write_json(
            candidate_root / "full/resolved_config.json",
            resolved_config,
        )
        candidate_runs.append(
            {
                "model_id": model_id,
                "run_contract": str(contract_path),
                "full_report": str(report_path),
                "resolved_config": str(resolved_config_path),
                "selected_checkpoint": str(checkpoint_path),
            }
        )
        comparison_rows.append(
            {
                "model_id": model_id,
                "seed": 42,
                "selected_epoch": 1,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                **{
                    name: float(metrics[name])
                    for name in (
                        "outcome_mcc",
                        "failure_macro_f1",
                        "failtype_macro_f1",
                        "action_macro_f1",
                        "needs_recovery_macro_f1",
                        "strategy_attempted_macro_f1",
                        "recovery_outcome_mcc",
                        "memory_mcc",
                        "bbox_mean_iou",
                        "bbox_recall_iou50",
                        "outcome_ece",
                    )
                },
            }
        )
    comparison_rows.sort(
        key=lambda row: (
            -row["outcome_mcc"],
            -row["recovery_outcome_mcc"],
            -row["action_macro_f1"],
            row["outcome_ece"],
            row["model_id"],
        )
    )
    for rank, row in enumerate(comparison_rows, start=1):
        row["validation_rank"] = rank
        row["selected_candidate"] = rank == 1
    comparison_json = _write_json(
        root / "selection-source/three_model_selection.json",
        {
            "status": "PASS",
            "selection_scope": "validation_only",
            "selection_rule": (
                "all_quality_gates_then_outcome_mcc; ties: recovery_outcome_mcc, "
                "action_macro_f1, lower_outcome_ece, model_id"
            ),
            "git_commit": commit,
            "selected_model_id": comparison_rows[0]["model_id"],
            "models": comparison_rows,
            "test_rows_read": 0,
            "next_action": (
                "Promote the validation-selected seed-42 checkpoint to the frozen "
                "final Table 2 campaign; do not add model seeds 43 or 44."
            ),
        },
    )
    comparison_csv = root / "selection-source/three_model_validation_comparison.csv"
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    result = {
        "candidate_runs": candidate_runs,
        "comparison_json": str(comparison_json),
        "comparison_csv": str(comparison_csv),
    }
    if selection_mode == PC01_PROVISIONAL_SELECTION_MODE:
        return {
            "mode": selection_mode,
            "candidate_runs": [candidate_runs[0]],
        }
    if selection_mode != THREE_CANDIDATE_FINAL_SELECTION_MODE:
        raise AssertionError(f"unsupported fixture selection mode: {selection_mode}")
    return {"mode": selection_mode, **result}


def test_validation_selection_is_recomputed_not_self_declared(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    model = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    qwen_report_path = Path(inputs["candidate_runs"][0]["full_report"])
    report = read_json(qwen_report_path)
    report["selected_epoch"] = 9
    _write_json(qwen_report_path, report)

    with pytest.raises(SchemaError, match="reported selection differs from replay"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "staged-selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )


def test_pc01_provisional_selection_stages_one_candidate_without_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_SELECTED_EPOCH",
        model["selected_epoch"],
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_CHECKPOINT_SHA256",
        model["selected_checkpoint_sha256"],
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_CONFIG_SHA256",
        model["resolved_config_record_sha256"],
    )

    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "staged-selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    manifest = read_json(manifest_path)
    assert manifest["selection_mode"] == PC01_PROVISIONAL_SELECTION_MODE
    assert manifest["selection_status"] == "PROVISIONAL_PC01_PILOT"
    assert manifest["evidence_label"] == "PILOT_ONLY"
    assert manifest["provisional"] is True
    assert manifest["final_campaign_eligible"] is False
    assert manifest["comparison_performed"] is False
    assert manifest["candidate_model_ids"] == ["qwen2vl_2b_gold_v2_8_dgx"]
    assert len(manifest["candidates"]) == 1
    assert not (manifest_path.parent / "comparison").exists()
    assert "comparison_script" not in manifest

    validate_selection_evidence(
        manifest_path,
        selected_model_manifest=model,
        expected_selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    with pytest.raises(SchemaError, match="mode differs"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifest=model,
            expected_selection_mode=THREE_CANDIDATE_FINAL_SELECTION_MODE,
        )


def test_pc01_provisional_selection_requires_registered_epoch6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_CHECKPOINT_SHA256",
        model["selected_checkpoint_sha256"],
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_CONFIG_SHA256",
        model["resolved_config_record_sha256"],
    )

    with pytest.raises(SchemaError, match="registered PC-01 selected_epoch"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "staged-selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
            selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
        )


def test_final_three_candidate_evidence_cannot_authorize_pilot_protocol(
    tmp_path: Path,
):
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "staged-selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
    )

    with pytest.raises(SchemaError, match="mode differs"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifest=model,
            expected_selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
        )


def test_validation_selection_replays_staged_config_semantics(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    model = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    staged = tmp_path / "staged-selection"
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=staged,
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
    )

    manifest = read_json(manifest_path)
    candidate = next(
        row
        for row in manifest["candidates"]
        if row["model_id"] == "qwen2vl_2b_gold_v2_8_dgx"
    )
    relative = candidate["registered_config_path"]
    config_path = staged / relative
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "name: Y_QWEN2VL_2B_GOLD_V2_8_DGX",
            "name: FORGED_SELECTION_CONFIG",
        ),
        encoding="utf-8",
    )
    forged_sha = sha256_file(config_path)
    candidate["registered_config_sha256"] = forged_sha
    next(row for row in manifest["artifacts"] if row["path"] == relative)[
        "sha256"
    ] = forged_sha
    _write_json(manifest_path, manifest)

    with pytest.raises(SchemaError, match="staged config differs"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifest=model,
        )


def test_validation_selection_binds_exact_seed42_campaign_model_set(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    seed_42 = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=seed_42["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=seed_42[
            "resolved_config_record_sha256"
        ],
    )
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "staged-selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=seed_42,
    )

    validate_selection_evidence(
        manifest_path,
        repository_root=REPOSITORY_ROOT,
        selected_model_manifests={42: seed_42},
        expected_model_seeds=[42],
    )

    with pytest.raises(SchemaError, match="exact campaign matched-seed set"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifests={42: seed_42},
            expected_model_seeds=[42, 43],
        )

    seed_43 = {
        **seed_42,
        "model_seed": 43,
        "selected_epoch": 2,
        "selected_checkpoint_sha256": "1" * 64,
        "resolved_config_sha256": "2" * 64,
        "resolved_config_record_sha256": "3" * 64,
    }
    with pytest.raises(SchemaError, match="exactly model seed 42"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifests={42: seed_42, 43: seed_43},
            expected_model_seeds=[42, 43],
        )

    wrong_family = {
        42: {**seed_42, "selected_model_id": "unselected-model"}
    }
    with pytest.raises(SchemaError, match="model ID differs"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifests=wrong_family,
            expected_model_seeds=[42],
        )


def test_validation_selection_binds_every_resolved_config_artifact(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    model = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    staged = tmp_path / "staged-selection"
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=staged,
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
    )

    manifest = read_json(manifest_path)
    candidate = next(
        row
        for row in manifest["candidates"]
        if row["model_id"] == "internvl35_8b_gold_v2_8_dgx"
    )
    relative = candidate["resolved_config_path"]
    resolved_config_path = staged / relative
    _write_json(resolved_config_path, {"forged": "self-consistent bytes"})
    forged_sha = sha256_file(resolved_config_path)
    candidate["resolved_config_artifact_sha256"] = forged_sha
    next(row for row in manifest["artifacts"] if row["path"] == relative)[
        "sha256"
    ] = forged_sha
    _write_json(manifest_path, manifest)

    with pytest.raises(
        SchemaError,
        match="resolved configuration differs from its run contract",
    ):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifest=model,
        )


def test_validation_selection_rejects_out_of_domain_metrics(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    model = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    report_path = Path(inputs["candidate_runs"][0]["full_report"])
    report = read_json(report_path)
    report["history"][0]["memory_mcc"] = 1.01
    report["source_validation"]["primary_original_gold"][
        "selected_epoch_metrics"
    ] = report["history"][0]
    report["quality_gates"] = controlled_quality_gates(
        report, ALL_GATES_THEN_OUTCOME_RULE
    )
    _write_json(report_path, report)

    with pytest.raises(SchemaError, match=r"memory_mcc is outside \[-1.0, 1\]"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "staged-selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )


def test_validation_selection_hashes_every_candidate_checkpoint(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    model = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    checkpoint_path = Path(inputs["candidate_runs"][2]["selected_checkpoint"])
    checkpoint_path.write_bytes(b"tampered non-winning checkpoint")

    with pytest.raises(SchemaError, match="selected checkpoint bytes differ"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "staged-selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )


def test_validation_selection_rejects_candidate_bundle_swaps(tmp_path: Path):
    model_path = _valid_model_manifest(tmp_path / "model.json")
    model = read_json(model_path)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model[
            "resolved_config_record_sha256"
        ],
    )
    first = inputs["candidate_runs"][0]
    second = inputs["candidate_runs"][1]
    for field in ("full_report", "resolved_config", "selected_checkpoint"):
        first[field], second[field] = second[field], first[field]

    with pytest.raises(
        SchemaError,
        match="resolved configuration differs from its run contract",
    ):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "staged-selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )


def _valid_runner_attestation(
    path: Path,
    *,
    model: Path,
    memory: Path,
    environment: Path,
    resolved_tasks: Path,
) -> Path:
    source = Path(__file__).resolve()
    relative = str(source.relative_to(REPOSITORY_ROOT))
    live_fixture_source = REPOSITORY_ROOT / "tests/table2/test_live_deployment.py"
    live_validator_source = (
        REPOSITORY_ROOT / "src/web_agent/eval/table2/live_deployment.py"
    )
    source_rows = [
        {
            "relative_path": str(candidate.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(candidate),
        }
        for candidate in (source, live_fixture_source, live_validator_source)
    ]
    identity = _expected_runner_runtime_identity(
        model_by_seed={42: model},
        memory_by_seed={42: memory},
        protocol=yaml.safe_load(PROTOCOL.read_text(encoding="utf-8")),
        prompt_sources={
            "parameter_provider_v1.txt": (
                REPOSITORY_ROOT
                / "configs/eval/table2/prompts/parameter_provider_v1.txt"
            ),
            "e0_action_v1.txt": (
                REPOSITORY_ROOT / "configs/eval/table2/prompts/e0_action_v1.txt"
            ),
        },
        environment_path=environment,
        resolved_task_snapshot_path=resolved_tasks,
        runtime_integration={
            "entrypoint": "tests.table2.test_package_validator:_NestedEvidenceRunner",
            "source_relative_path": relative,
            "source_sha256": sha256_file(source),
        },
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()
    return _write_json(
        path,
        {
            "schema_version": RUNNER_ATTESTATION_SCHEMA_VERSION,
            "attestation_scope": EVALUATION_RUNNER_SCOPE,
            "runner_entrypoint": (
                "tests.table2.test_package_validator:_NestedEvidenceRunner"
            ),
            "runtime_integration_entrypoint": (
                "tests.table2.test_package_validator:_NestedEvidenceRunner"
            ),
            "repository_commit": commit,
            "primary_source_relative_path": relative,
            "source_files": source_rows,
            "source_set_sha256": sha256_json(source_rows),
            "runtime_identity": identity,
        },
    )


def _valid_memory_manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "table2-frozen-memory-v2",
            "model_seed": 42,
            "checkpoint_sha256": SHA_A,
            "source_split": "train",
            "test_rows_read": 0,
            "embedding_stage": "post_action_memory_task_adapter",
            "embedding_dimension": 768,
            "normalization": "l2",
            "similarity": "cosine",
            "top_k": 3,
            "tie_break": "memory_id_ascending",
            "admission_threshold_source": "train_only_calibration",
            "admission_threshold": 0.42,
            "same_task_exclusion": True,
            "duplicate_exclusion": True,
            "runtime_writes_allowed": False,
        },
    )


def _valid_memory_store(path: Path, *, model: Path | None = None) -> Path:
    protocol_sha256 = sha256_file(PROTOCOL)
    model_value = read_json(model) if model is not None else {}
    checkpoint_sha256 = str(model_value.get("selected_checkpoint_sha256", SHA_A))
    resolved_config_sha256 = str(model_value.get("resolved_config_sha256", SHA_B))
    resolved_config_record_sha256 = str(
        model_value.get("resolved_config_record_sha256", SHA_C)
    )
    candidates: list[EligibleMemoryCandidate] = []
    specs = (
        ("RETRY", "CLICK", SHA_B),
        ("RETRY", "CLICK", "d" * 64),
        ("REPLAN", "TYPE", "e" * 64),
    )
    for index, (strategy, action, exact_key) in enumerate(specs, start=1):
        memory_id = f"memory-{index}"
        source_id = f"train-sample-{index}"
        recovery_id = f"train-recovery-{index}"
        task_id = f"train-task-{index}"
        episode_id = f"train-episode-{index}"
        verification = _memory_verification_evidence(
            source_id=source_id,
            recovery_id=recovery_id,
            task_id=task_id,
            episode_id=episode_id,
            strategy=strategy,
            action=action,
        )
        recovery_verification_sha = verification["recovery_verification"][
            "evidence_sha256"
        ]
        final_task_verification_sha = verification["final_task_verification"][
            "evidence_sha256"
        ]
        candidates.append(
            EligibleMemoryCandidate(
                record_index=index - 1,
                memory_id=memory_id,
                source_sample_id=source_id,
                recovery_sample_id=recovery_id,
                canonical_task_id=task_id,
                episode_id=episode_id,
                step_index=index,
                exact_duplicate_key=exact_key,
                near_duplicate_cluster_id=f"train-near-{index}",
                duplicate_cluster_namespace_id=JOINT_DUPLICATE_NAMESPACE[
                    "namespace_id"
                ],
                recovery_verification_evidence_sha256=(
                    recovery_verification_sha
                ),
                final_task_verification_evidence_sha256=(
                    final_task_verification_sha
                ),
                verification_evidence_sha256=verification[
                    "verification_evidence_sha256"
                ],
                verification_evidence=verification,
                item={
                    "schema_version": "table2-memory-eligibility-v1",
                    "memory_id": memory_id,
                    "source_dataset_id": "gold-v2.8",
                    "source_dataset_version": "fixture",
                    "source_split": "train",
                    "source_sample_id": source_id,
                    "recovery_sample_id": recovery_id,
                    "source_task_id": task_id,
                    "source_episode_id": episode_id,
                    "step_index": index,
                    "exact_duplicate_key": exact_key,
                    "duplicate_cluster_id": f"train-near-{index}",
                    "duplicate_cluster_namespace_id": (
                        JOINT_DUPLICATE_NAMESPACE["namespace_id"]
                    ),
                    "failure_type": "NO_EFFECT",
                    "strategy": strategy,
                    "executed_recovery_action": action,
                    "memory_update_flag": True,
                    "verified_recovery_success": True,
                    "final_task_success": True,
                    "provenance_valid": True,
                    "recovery_verification_evidence_sha256": (
                        recovery_verification_sha
                    ),
                    "final_task_verification_evidence_sha256": (
                        final_task_verification_sha
                    ),
                    "verification_evidence_sha256": verification[
                        "verification_evidence_sha256"
                    ],
                },
            )
        )
    selection = EligibilitySelection(
        candidates=tuple(candidates),
        exclusion_counts={},
        input_rows=3,
        pre_dedup_eligible_rows=3,
        duplicate_cluster_namespace=JOINT_DUPLICATE_NAMESPACE,
    )
    embedding = np.zeros((3, 768), dtype=np.float32)
    embedding[0, 0] = 1.0
    embedding[1, 1] = 1.0
    embedding[2, 2] = 1.0
    provenance = ProvenanceManifest.from_mapping(
        {
            "schema_version": "table2-memory-provenance-v1",
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "dataset_id": "gold-v2.8",
            "dataset_version": "fixture",
            "dataset_artifacts_sha256": SHA_C,
            "records_sha256": SHA_B,
            "duplicate_cluster_namespace": JOINT_DUPLICATE_NAMESPACE,
            "records": {
                candidate.source_sample_id: {
                    "source_split": "train",
                    "source_sample_id": candidate.source_sample_id,
                    "provenance_valid": True,
                    "final_task_success": True,
                    "canonical_task_id": candidate.canonical_task_id,
                    "episode_id": candidate.episode_id,
                    "exact_duplicate_key": candidate.exact_duplicate_key,
                    "near_duplicate_cluster_id": (
                        candidate.near_duplicate_cluster_id
                    ),
                    "duplicate_cluster_namespace_id": (
                        JOINT_DUPLICATE_NAMESPACE["namespace_id"]
                    ),
                    **_memory_verification_evidence(
                        source_id=candidate.source_sample_id,
                        recovery_id=candidate.recovery_sample_id,
                        task_id=candidate.canonical_task_id,
                        episode_id=candidate.episode_id,
                        strategy=str(candidate.item["strategy"]),
                        action=str(candidate.item["executed_recovery_action"]),
                    ),
                }
                for candidate in candidates
            },
        }
    )
    calibration_evidence = build_calibration_evidence(
        selection=selection,
        embeddings=embedding,
        provenance=provenance,
        provenance_manifest_sha256=SHA_B,
        model_seed=42,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=SHA_B,
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=resolved_config_sha256,
        resolved_config_record_sha256=resolved_config_record_sha256,
        protocol_sha256=protocol_sha256,
    )
    store = build_frozen_store(
        path,
        selection=selection,
        embeddings=embedding,
        model_seed=42,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=SHA_B,
        dataset_id="gold-v2.8",
        dataset_version="fixture",
        dataset_artifacts_sha256=SHA_C,
        resolved_config_sha256=resolved_config_sha256,
        resolved_config_record_sha256=resolved_config_record_sha256,
        protocol_sha256=protocol_sha256,
        provenance_manifest_sha256=SHA_B,
        threshold_calibration=calibration_evidence["threshold_calibration"],
        calibration_evidence=calibration_evidence,
        transition_report={"status": "PASS"},
    )
    return store.root / "manifest.json"


def _evaluation_registration(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    environment = _valid_environment(tmp_path / "environment.json")
    duplicate = tmp_path / "verified-duplicate-audit.json"
    resolved_tasks = _valid_resolved_task_snapshot(
        tmp_path / "resolved-tasks.json",
        environment=environment,
        duplicate_audit_manifest=duplicate,
    )
    _verified_duplicate_audit(duplicate, task_manifest=resolved_tasks)
    config = _campaign_config(
        tmp_path,
        mode="evaluation",
        duplicate_audit_manifest=duplicate,
        resolved_task_snapshot=resolved_tasks,
    )
    return config, environment, resolved_tasks, duplicate


def _freeze_smoke(tmp_path: Path, *, name: str = "campaign") -> Path:
    destination = tmp_path / name
    freeze_campaign(
        repository_root=REPOSITORY_ROOT,
        campaign_config_path=_campaign_config(tmp_path / f"{name}-input", mode="smoke"),
        campaign_dir=destination,
        campaign_id=f"{name}-id",
        allow_dirty_pilot=True,
    )
    return destination


def test_explicit_smoke_freeze_registers_exact_pilot_and_canonical_layout(tmp_path: Path):
    campaign = _freeze_smoke(tmp_path)
    manifest = read_json(campaign / "campaign_manifest.json")
    schedule = read_jsonl(campaign / "schedule/schedule.jsonl")

    assert manifest["campaign_mode"] == "smoke"
    assert manifest["normal_task_count"] == 50
    assert manifest["recovery_scenario_count"] == 15
    assert manifest["scheduled_block_count"] == 65
    assert manifest["planned_episode_count"] == 260
    assert manifest["normal_block_count"] == 50
    assert manifest["recovery_block_count"] == 15
    assert manifest["evidence_label"] == "PILOT_ONLY"
    assert manifest["publication_status"] == "PILOT_ONLY"
    assert manifest["paper_table_status"] == "N/R"
    exclusion_path = (
        campaign / "frozen/benchmark/pilot_task_exclusion_registry.json"
    )
    assert manifest["pilot_task_exclusion_registry_sha256"] == sha256_file(
        exclusion_path
    )
    provenance = read_json(campaign / "frozen/provenance.json")
    assert provenance["pilot_task_exclusion_authority"]["registry_sha256"] == (
        manifest["pilot_task_exclusion_registry_sha256"]
    )

    assert len(schedule) == 65
    assert sum(row["task_partition"] == "normal" for row in schedule) == 50
    assert sum(row["task_partition"] == "recovery_diagnostic" for row in schedule) == 15
    assert [row["task_partition"] for row in schedule[:15]] == [
        "recovery_diagnostic"
    ] * 15
    assert [row["task_partition"] for row in schedule[15:]] == ["normal"] * 50
    assert all(row["matched_model_seed"] == 42 for row in schedule)
    assert all(row["repeat_id"] == 0 for row in schedule)
    assert all(row["max_block_attempts"] == 2 for row in schedule)
    assert all(tuple(row["systems"]) == SYSTEM_IDS for row in schedule)

    required_paths = (
        "campaign_manifest.json",
        "artifact_hashes.json",
        "access_ledger.jsonl",
        "deviation_ledger.jsonl",
        "frozen/campaign.yaml",
        "frozen/protocol.yaml",
        "frozen/environment.json",
        "frozen/provenance.json",
        "frozen/task_manifest.json",
        "frozen/benchmark/recovery_scenarios.json",
        "frozen/benchmark/recovery_oracle_rules.json",
        "frozen/benchmark/audit_manifest.json",
        "frozen/prompts/e0_action_v1.txt",
        "frozen/prompts/parameter_provider_v1.txt",
        "schedule/schedule.jsonl",
    )
    assert all((campaign / relative).is_file() for relative in required_paths)
    assert all(
        (campaign / relative).is_dir()
        for relative in (
            "frozen/systems",
            "component_test",
            "memory",
            "paired_blocks",
            "aggregate",
            "manual_audit",
        )
    )

    report = validate_campaign(campaign, require_complete=False)
    assert report.passed, report.errors
    assert report.publication_status == "PILOT_ONLY"
    assert report.counts["scheduled_blocks"] == 65
    assert report.counts["block_status_not_started"] == 65


def test_campaign_validation_rejects_changed_pilot_exclusion_provenance_hash(
    tmp_path: Path,
) -> None:
    campaign = _freeze_smoke(tmp_path)
    manifest_path = campaign / "campaign_manifest.json"
    manifest = read_json(manifest_path)
    manifest["pilot_task_exclusion_registry_sha256"] = SHA_B
    _write_json(manifest_path, manifest)

    report = validate_campaign(campaign, require_complete=False)
    assert not report.passed
    assert any("pilot-exclusion registry hash" in error for error in report.errors)


def test_pilot_summary_is_guarded_and_never_promoted_to_paper_table(tmp_path: Path):
    campaign = _freeze_smoke(tmp_path)

    with pytest.raises(Table2Error, match="campaign validation failed"):
        summarize_campaign(campaign)

    with (campaign / "aggregate/table2_main.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["System"] for row in rows] == list(SYSTEM_IDS)
    assert all(row["Evidence Status"] == "PILOT_ONLY" for row in rows)
    assert all(row["Task Success Rate"] == "N/R" for row in rows)
    assert read_json(campaign / "campaign_manifest.json")["paper_table_status"] == "N/R"


def test_evaluation_freeze_requires_environment_model_and_memory_readiness(tmp_path: Path):
    config, environment, resolved_tasks, _ = _evaluation_registration(tmp_path)

    with pytest.raises(SchemaError, match="environment manifest"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "missing-environment",
            allow_dirty_pilot=True,
        )

    with pytest.raises(SchemaError, match="model manifests must cover every matched seed"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "missing-model",
            environment_manifest_path=environment,
            resolved_task_snapshot_path=resolved_tasks,
            allow_dirty_pilot=True,
        )

    model = _valid_model_manifest(tmp_path / "model.json")
    with pytest.raises(SchemaError, match="memory manifests must cover every matched seed"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "missing-memory",
            environment_manifest_path=environment,
            resolved_task_snapshot_path=resolved_tasks,
            model_manifest_paths=[model],
            allow_dirty_pilot=True,
        )

    manifest_only = _valid_memory_manifest(
        tmp_path / "manifest-only-store/manifest.json"
    )
    with pytest.raises(SchemaError, match="invalid frozen memory store"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "manifest-only-memory",
            environment_manifest_path=environment,
            resolved_task_snapshot_path=resolved_tasks,
            model_manifest_paths=[model],
            memory_manifest_paths=[manifest_only],
            allow_dirty_pilot=True,
        )

    memory = _valid_memory_store(tmp_path / "external-memory-store", model=model)
    runner_attestation = _valid_runner_attestation(
        tmp_path / "runner-attestation.json",
        model=model,
        memory=memory,
        environment=environment,
        resolved_tasks=resolved_tasks,
    )
    campaign = tmp_path / "ready-evaluation"
    freeze_campaign(
        repository_root=REPOSITORY_ROOT,
        campaign_config_path=config,
        campaign_dir=campaign,
        environment_manifest_path=environment,
        model_manifest_paths=[model],
        memory_manifest_paths=[memory],
        runner_attestation_path=runner_attestation,
        resolved_task_snapshot_path=resolved_tasks,
        allow_dirty_pilot=True,
    )
    assert (campaign / "frozen/models/seed_42.json").is_file()
    copied_store = campaign / "memory/seed_42"
    assert {
        path.name for path in copied_store.iterdir() if path.is_file()
    } == {
        "calibration_evidence.json",
        "verification_evidence.json",
        "manifest.json",
        "manifest.sha256",
        "embeddings.npy",
        "items.jsonl",
        "threshold_calibration.json",
    }
    assert (copied_store / "embeddings.npy").read_bytes() == (
        memory.parent / "embeddings.npy"
    ).read_bytes()
    report = validate_campaign(campaign, require_complete=False)
    assert report.passed, report.errors

    (copied_store / "items.jsonl").write_text("{}\n", encoding="utf-8")
    tampered = validate_campaign(campaign, require_complete=False)
    assert not tampered.passed
    assert any("hash mismatch" in error for error in tampered.errors)


def test_duplicate_audit_rejects_a_different_memory_cluster_namespace(
    tmp_path: Path,
):
    _, _, resolved_tasks, duplicate_audit = _evaluation_registration(tmp_path)
    task_rows = read_json(resolved_tasks)["tasks"]
    wrong_namespace = {
        **JOINT_DUPLICATE_NAMESPACE,
        "namespace_id": "different-joint-namespace",
    }
    task_content_manifest = {
        "task_content_sha256_by_id": {
            str(row["task_id"]): sha256_json(row) for row in task_rows
        },
        "train_corpus_by_seed": {
            "42": {
                "corpus_binding_sha256": _fixture_train_corpus_binding(),
                "duplicate_cluster_namespace": wrong_namespace,
                "duplicate_cluster_namespace_sha256": sha256_json(
                    wrong_namespace
                ),
            }
        },
    }

    with pytest.raises(SchemaError, match="different cluster namespaces"):
        _validate_duplicate_audit_bindings(
            duplicate_audit,
            task_content_manifest=task_content_manifest,
            require_verified_normal=True,
        )


def test_evaluation_freeze_requires_positive_validation_selection_evidence(
    tmp_path: Path,
):
    config, environment, resolved_tasks, _ = _evaluation_registration(tmp_path)
    model = _valid_model_manifest(tmp_path / "model.json")
    payload = json.loads(model.read_text(encoding="utf-8"))
    payload["validation_rows_read"] = 0
    _write_json(model, payload)

    with pytest.raises(SchemaError, match="positive validation evidence"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "zero-validation-model",
            environment_manifest_path=environment,
            model_manifest_paths=[model],
            resolved_task_snapshot_path=resolved_tasks,
            allow_dirty_pilot=True,
        )


def test_evaluation_freeze_rejects_selection_mode_mismatch(tmp_path: Path):
    config, environment, resolved_tasks, _ = _evaluation_registration(tmp_path)
    model = _valid_model_manifest(tmp_path / "model.json")
    model_value = read_json(model)
    memory = _valid_memory_store(tmp_path / "memory", model=model)
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model_value["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model_value[
            "resolved_config_record_sha256"
        ],
        selection_mode=THREE_CANDIDATE_FINAL_SELECTION_MODE,
    )
    selection_manifest = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "final-selection-evidence",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model_value,
        selection_mode=THREE_CANDIDATE_FINAL_SELECTION_MODE,
    )
    campaign = yaml.safe_load(config.read_text(encoding="utf-8"))
    campaign["checkpoint_selection_evidence"] = str(selection_manifest)
    config.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")

    with pytest.raises(SchemaError, match="mode differs"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "mode-mismatch",
            environment_manifest_path=environment,
            model_manifest_paths=[model],
            memory_manifest_paths=[memory],
            resolved_task_snapshot_path=resolved_tasks,
            allow_dirty_pilot=True,
        )


def test_campaign_freeze_rejects_unregistered_model_seeds(tmp_path: Path):
    config = _campaign_config(tmp_path, mode="smoke")
    campaign = yaml.safe_load(config.read_text(encoding="utf-8"))
    campaign["matched_seeds"] = [42, 43]
    config.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")

    with pytest.raises(SchemaError, match=r"matched_seeds must be exactly \[42\]"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=config,
            campaign_dir=tmp_path / "wrong-seeds",
            allow_dirty_pilot=True,
        )


def test_model_memory_binding_requires_canonical_resolved_config_identity(
    tmp_path: Path,
):
    model = _valid_model_manifest(tmp_path / "model.json")
    memory = _valid_memory_store(tmp_path / "memory", model=model)
    _validate_model_memory_source_bindings(
        {42: model},
        {42: memory},
        protocol_source=PROTOCOL,
    )

    memory_value = read_json(memory)
    memory_value["resolved_config_record_sha256"] = "f" * 64
    _write_json(memory, memory_value)
    with pytest.raises(
        SchemaError,
        match="canonical resolved-config record hashes disagree",
    ):
        _validate_model_memory_source_bindings(
            {42: model},
            {42: memory},
            protocol_source=PROTOCOL,
        )


def test_environment_and_processor_payloads_fail_closed(tmp_path: Path):
    environment = _valid_environment(tmp_path / "environment.json")
    environment_value = read_json(environment)
    del environment_value["evaluator"]["source_sha256"]
    with pytest.raises(SchemaError, match="evaluator requires source_sha256"):
        _validate_environment_manifest(
            environment_value,
            protocol=load_yaml(REPOSITORY_ROOT / "configs/eval/table2/protocol.yaml"),
        )

    model = _valid_model_manifest(tmp_path / "model.json")
    processor_path = Path(
        read_json(model)["artifact_payloads"]["processor_contract"]["path"]
    )
    processor_path.write_bytes(processor_path.read_bytes() + b"\n")
    with pytest.raises(SchemaError, match="exact canonical"):
        _processor_contract_from_payload(processor_path)


def test_supported_handoff_preparer_freezes_a_resolvable_evaluation_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environment_path = _valid_environment(tmp_path / "source-environment.json")
    environment = read_json(environment_path)
    evaluator = dict(environment.pop("evaluator"))
    environment.pop("dependency_lock_sha256")
    environment["benchmark_version"] = PINNED_BROWSERGYM_WEBARENA_VERSION
    environment["task_definition_version"] = PINNED_TASK_DEFINITION_VERSION
    live_evidence_root = tmp_path / "live-deployment-evidence"
    live_deployment_manifest = _write_json(
        live_evidence_root / "deployment.json",
        _valid_live_deployment_manifest(live_evidence_root),
    )

    duplicate_input = read_json(
        REPOSITORY_ROOT / "benchmarks/table2/pilot/duplicate_audit_manifest.json"
    )
    duplicate_input["manifest_state"] = "FROZEN_REGISTRATION"
    duplicate_input["normal_task_evidence_status"] = "VERIFIED"
    duplicate_input["normal_task_runtime_policy"] = (
        "VERIFIED_NONEMPTY_CLUSTERS_REQUIRED"
    )

    url_map = WEBARENA_TASK_URL_MAP
    tokens = sorted(url_map)
    upstream_rows = []
    for index in range(50):
        token = tokens[index % len(tokens)]
        site = token.strip("_").casefold()
        upstream_rows.append(
            {
                "sites": [site],
                "task_id": index,
                "require_login": True,
                "storage_state": f"./.auth/{site}_state.json",
                "start_url": f"{token}/task/{index}",
                "geolocation": (
                    {
                        "latitude": 23.8103,
                        "longitude": 90.4125,
                        "accuracy": 10.0,
                    }
                    if index == 0
                    else None
                ),
                "intent": f"Fixture WebArena instruction {index}",
                "require_reset": False,
                "eval": {
                    "eval_types": ["url_match"],
                    "reference_url": token,
                },
            }
        )
    upstream_payload = json.dumps(upstream_rows, sort_keys=True).encode("utf-8")
    upstream_source_path = tmp_path / "test.raw.json"
    upstream_source_path.write_bytes(upstream_payload)
    upstream_source_sha256 = sha256_bytes(upstream_payload)
    url_map_path = _write_json(tmp_path / "webarena-url-map.json", url_map)
    task_export = build_public_pilot_task_export(
        source=upstream_source_path,
        registry_path=(
            REPOSITORY_ROOT / "benchmarks/table2/pilot/task_manifest.json"
        ),
        site_url_map=url_map,
        snapshot_id="external-webarena-export-v1",
        benchmark_version=PINNED_BROWSERGYM_WEBARENA_VERSION,
        task_definition_version=PINNED_TASK_DEFINITION_VERSION,
        evaluator_id=evaluator["evaluator_id"],
        evaluator_version=evaluator["evaluator_version"],
        expected_source_sha256=upstream_source_sha256,
        authorized_raw_json_sha256=upstream_source_sha256,
    )
    task_export_path = _write_json(
        tmp_path / "upstream-task-export.json", task_export
    )
    task_interface_audit = build_webarena_task_interface_audit(task_export)
    assert task_interface_audit["status"] == "PASS"
    assert task_interface_audit["compatible_task_count"] == 50
    task_interface_audit_path = _write_json(
        tmp_path / "webarena-task-interface-audit.json",
        task_interface_audit,
    )
    authenticated_task_rows = []
    for source_row in task_export["tasks"]:
        row = {
            key: source_row[key]
            for key in (
                "task_id",
                "upstream_index",
                "benchmark_task_id",
                "benchmark_task_version",
                "instruction",
                "start_state",
                "task_config",
                "evaluator",
            )
        }
        row["source_content_sha256"] = sha256_json(row)
        authenticated_task_rows.append(row)

    source_model = _valid_model_manifest(tmp_path / "source-model.json")
    source_model_value = read_json(source_model)
    memory = _valid_memory_store(tmp_path / "source-memory", model=source_model)
    duplicate_input["duplicate_cluster_namespace"] = dict(
        JOINT_DUPLICATE_NAMESPACE
    )
    namespace = JointDuplicateClusterNamespace.from_mapping(
        JOINT_DUPLICATE_NAMESPACE,
        require_hashes=True,
    )
    content_hashes = {
        str(row["task_id"]): sha256_json(row)
        for row in authenticated_task_rows
    }
    for row in duplicate_input["entries"]:
        if row["task_partition"] != "normal":
            continue
        task_id = str(row["task_id"])
        clusters = [f"external:{task_id}"]
        audit_record = canonical_task_audit_record(
            task_id=task_id,
            content_sha256=content_hashes[task_id],
            train_corpus_manifest_sha256=_fixture_train_corpus_binding(),
            cluster_ids=clusters,
            namespace=namespace,
        )
        row.update(
            {
                "status": "VERIFIED",
                "cluster_ids": clusters,
                "content_sha256": content_hashes[task_id],
                "train_corpus_manifest_sha256": _fixture_train_corpus_binding(),
                "cluster_namespace_id": namespace.namespace_id,
                "audit_tool_id": namespace.audit_tool_id,
                "audit_tool_version": namespace.audit_tool_version,
                "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
                "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
                "audit_record": audit_record,
                "evidence_sha256": sha256_json(audit_record),
            }
        )
    duplicate_input_path = _write_json(
        tmp_path / "external-duplicate-audit.json", duplicate_input
    )
    model_fields = (
        "model_seed",
        "selected_model_id",
        "selected_epoch",
        "e0_backbone_id",
        "e0_backbone_revision",
        "e0_parser_id",
        "e0_parser_version",
        "e0_parser_module",
        "e0_parser_attribute",
        "selection_scope",
        "checkpoint_selection",
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
    )
    model_spec = {key: source_model_value[key] for key in model_fields}
    model_spec["artifact_paths"] = {
        role: descriptor["path"]
        for role, descriptor in source_model_value["artifact_payloads"].items()
    }
    selection_evidence = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=source_model_value[
            "selected_checkpoint_sha256"
        ],
        selected_resolved_config_sha256=source_model_value[
            "resolved_config_record_sha256"
        ],
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_SELECTED_EPOCH",
        source_model_value["selected_epoch"],
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_CHECKPOINT_SHA256",
        source_model_value["selected_checkpoint_sha256"],
    )
    monkeypatch.setattr(
        selection_evidence_module,
        "PC01_EXPECTED_CONFIG_SHA256",
        source_model_value["resolved_config_record_sha256"],
    )
    source_relative = str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT))
    production_runner_relative = "src/web_agent/eval/table2/production_runner.py"
    spec_path = _write_json(
        tmp_path / "handoff-input.json",
        {
            "schema_version": "table2-handoff-input-v1",
            "campaign_config": str(PILOT_CONFIG),
            "dependency_lock": str(REPOSITORY_ROOT / "pyproject.toml"),
            "environment": environment,
            "evaluator": evaluator,
            "resolved_task_export": str(task_export_path),
            "webarena_task_interface_audit": str(task_interface_audit_path),
            "webarena_task_source": str(upstream_source_path),
            "authorized_raw_webarena_task_source_sha256": (
                upstream_source_sha256
            ),
            "webarena_site_url_map": str(url_map_path),
            "webarena_service_url_map": str(
                tmp_path / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH
            ),
            "webarena_host_preflight": str(
                tmp_path / PREFLIGHT_ARTIFACT_RELATIVE_PATH
            ),
            "webarena_deployment_topology": SINGLE_HOST_TOPOLOGY,
            "pc01_live_deployment_manifest": str(live_deployment_manifest),
            "pc01_live_deployment_evidence_root": str(live_evidence_root),
            "duplicate_audit": str(duplicate_input_path),
            "selection_evidence": selection_evidence,
            "models": [model_spec],
            "memory_manifests": [str(memory)],
            "runner": {
                "runner_entrypoint": (
                    "web_agent.eval.table2.production_runner:create_runner"
                ),
                "primary_source_relative_path": production_runner_relative,
                "runtime_integration_entrypoint": (
                    "tests.table2.test_package_validator:_NestedEvidenceRunner"
                ),
                "runtime_integration_source_relative_path": source_relative,
                "source_relative_paths": [
                    production_runner_relative,
                    source_relative,
                    "scripts/run_table2_evaluation.py",
                    "scripts/compare_full_models.py",
                    "src/web_agent/train/selection.py",
                    "src/web_agent/eval/table2/selection_evidence.py",
                    "src/web_agent/eval/table2/live_deployment.py",
                    "tests/table2/test_live_deployment.py",
                ],
            },
        },
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    monkeypatch.setattr(handoff_preparer, "_git_commit_clean", lambda _: commit)
    # The production handoff is permanently pinned to the real libwebarena
    # digest.  This fixture substitutes only its synthetic byte identity; it
    # does not mock or bypass the independent rebuild/equality gate.
    monkeypatch.setattr(
        handoff_preparer,
        "PINNED_TASK_SOURCE_SHA256",
        upstream_source_sha256,
    )
    handoff_root = tmp_path / "prepared-handoff"
    handoff = handoff_preparer.prepare_handoff(
        spec_path=spec_path,
        output_dir=handoff_root,
        repository_root=REPOSITORY_ROOT,
    )
    assert handoff["matched_seeds"] == [42]
    assert (handoff_root / "resolved_tasks.json").is_file()
    resolved_tasks = read_json(handoff_root / "resolved_tasks.json")
    assert resolved_tasks["upstream_task_source"] == task_export["source"]
    assert resolved_tasks["upstream_export_file_sha256"] == sha256_file(
        task_export_path
    )
    assert resolved_tasks["upstream_export_content_sha256"] == sha256_json(
        task_export
    )
    assert resolved_tasks["site_url_map_sha256"] == task_export[
        "site_url_map_sha256"
    ]
    assert resolved_tasks["task_action_interface_audit"] == (
        task_interface_audit
    )
    assert resolved_tasks["task_action_interface_audit_file_sha256"] == (
        sha256_file(handoff_root / TASK_INTERFACE_AUDIT_RELATIVE_PATH)
    )
    assert all(
        set(row["start_state"])
        == {
            "sites",
            "start_url",
            "require_login",
            "storage_state",
            "geolocation",
            "require_reset",
        }
        for row in resolved_tasks["tasks"]
    )
    assert (handoff_root / "runner_attestation.json").is_file()
    assert (handoff_root / "live_deployment/manifest.json").is_file()
    assert handoff["pc01_live_deployment"] == read_json(
        handoff_root / "environment.json"
    )["pc01_live_deployment"]
    assert sha256_file(handoff_root / "dependency.lock") == sha256_file(
        REPOSITORY_ROOT / "pyproject.toml"
    )
    assert (handoff_root / "selection_evidence/manifest.json").is_file()
    staged_selection = read_json(
        handoff_root / "selection_evidence/manifest.json"
    )
    assert staged_selection["selection_mode"] == PC01_PROVISIONAL_SELECTION_MODE
    assert staged_selection["evidence_label"] == "PILOT_ONLY"
    assert staged_selection["final_campaign_eligible"] is False
    assert not (handoff_root / "selection_evidence/comparison").exists()
    prepared_model = read_json(handoff_root / "models/seed_42.json")
    source_memory_manifest = read_json(memory)
    assert source_memory_manifest["resolved_config_sha256"] == (
        prepared_model["resolved_config_sha256"]
    )
    assert source_memory_manifest["resolved_config_record_sha256"] == (
        prepared_model["resolved_config_record_sha256"]
    )
    for role, descriptor in prepared_model["artifact_payloads"].items():
        payload_path = (handoff_root / "models" / descriptor["path"]).resolve()
        assert payload_path.exists(), role

    frozen_campaign = tmp_path / "prepared-campaign"
    freeze_arguments = handoff["freeze_arguments"]
    freeze_campaign(
        repository_root=REPOSITORY_ROOT,
        campaign_config_path=freeze_arguments["campaign_config"],
        campaign_dir=frozen_campaign,
        model_manifest_paths=freeze_arguments["model_manifests"],
        memory_manifest_paths=freeze_arguments["memory_manifests"],
        environment_manifest_path=freeze_arguments["environment_manifest"],
        runner_attestation_path=freeze_arguments["runner_attestation"],
        resolved_task_snapshot_path=freeze_arguments["resolved_task_snapshot"],
        allow_dirty_pilot=True,
    )
    campaign_manifest = read_json(frozen_campaign / "campaign_manifest.json")
    assert campaign_manifest["planned_episode_count"] == 260
    assert campaign_manifest["webarena_deployment_topology"] == (
        SINGLE_HOST_TOPOLOGY
    )
    assert campaign_manifest["webarena_preflight_artifact_sha256"] == sha256_file(
        frozen_campaign / "frozen" / PREFLIGHT_ARTIFACT_RELATIVE_PATH
    )
    assert campaign_manifest["runtime_integration_entrypoint"] == (
        "tests.table2.test_package_validator:_NestedEvidenceRunner"
    )
    assert campaign_manifest["pc01_live_deployment"] == read_json(
        frozen_campaign / "frozen/environment.json"
    )["pc01_live_deployment"]
    for descriptor in campaign_manifest["model_payloads_by_seed"]["42"].values():
        assert (frozen_campaign / descriptor["path"]).exists()
    frozen_model = read_json(frozen_campaign / "frozen/models/seed_42.json")
    frozen_memory = read_json(frozen_campaign / "memory/seed_42/manifest.json")
    assert frozen_memory["resolved_config_sha256"] == frozen_model[
        "resolved_config_sha256"
    ]
    assert frozen_memory["resolved_config_record_sha256"] == frozen_model[
        "resolved_config_record_sha256"
    ]
    assert sha256_file(frozen_campaign / "frozen/dependency.lock") == sha256_file(
        REPOSITORY_ROOT / "pyproject.toml"
    )
    report = validate_campaign(frozen_campaign, require_complete=False)
    assert report.passed, report.errors
    frozen_preflight = frozen_campaign / "frozen" / PREFLIGHT_ARTIFACT_RELATIVE_PATH
    preflight_bytes = frozen_preflight.read_bytes()
    frozen_preflight.write_bytes(preflight_bytes + b" ")
    tampered_preflight = validate_campaign(frozen_campaign, require_complete=False)
    assert not tampered_preflight.passed
    assert any("preflight" in error.lower() for error in tampered_preflight.errors)
    frozen_preflight.write_bytes(preflight_bytes)
    assert validate_campaign(frozen_campaign, require_complete=False).passed

    frozen_preflight.unlink()
    missing_preflight = validate_campaign(frozen_campaign, require_complete=False)
    assert not missing_preflight.passed
    assert any(
        "artifact hash manifest" in error.lower()
        for error in missing_preflight.errors
    )
    frozen_preflight.write_bytes(preflight_bytes)
    assert validate_campaign(frozen_campaign, require_complete=False).passed

    frozen_live_readiness = (
        frozen_campaign
        / "frozen/live_deployment/evidence/deterministic_reset.json"
    )
    frozen_live_readiness_bytes = frozen_live_readiness.read_bytes()
    frozen_live_readiness.write_bytes(frozen_live_readiness_bytes + b" ")
    tampered_live = validate_campaign(frozen_campaign, require_complete=False)
    assert not tampered_live.passed
    assert any(
        "live_deployment" in error.lower()
        or "live-deployment" in error.lower()
        for error in tampered_live.errors
    )
    frozen_live_readiness.write_bytes(frozen_live_readiness_bytes)
    assert validate_campaign(frozen_campaign, require_complete=False).passed

    (frozen_campaign / "frozen/dependency.lock").write_bytes(b"tampered\n")
    tampered = validate_campaign(frozen_campaign, require_complete=False)
    assert not tampered.passed
    assert any("dependency.lock" in error for error in tampered.errors)


def test_pilot_freeze_rejects_any_source_inside_registered_locked_mount(tmp_path: Path):
    locked_mount = tmp_path / "locked"
    locked_task_manifest = locked_mount / "task_manifest.json"
    locked_task_manifest.parent.mkdir(parents=True)
    locked_task_manifest.write_bytes(
        (REPOSITORY_ROOT / "benchmarks/table2/pilot/task_manifest.json").read_bytes()
    )

    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["benchmark"]["locked_mount"] = str(locked_mount)
    protocol["parameter_provider"]["prompt"] = str(
        REPOSITORY_ROOT / "configs/eval/table2/prompts/parameter_provider_v1.txt"
    )
    protocol["systems"]["overlay_directory"] = str(
        REPOSITORY_ROOT / "configs/eval/table2/systems"
    )
    protocol_dir = tmp_path / "locked-protocol"
    (protocol_dir / "prompts").mkdir(parents=True)
    (protocol_dir / "prompts/e0_action_v1.txt").write_bytes(
        (REPOSITORY_ROOT / "configs/eval/table2/prompts/e0_action_v1.txt").read_bytes()
    )
    protocol_path = protocol_dir / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    campaign = yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8"))
    campaign.update(
        {
            "campaign_mode": "smoke",
            "protocol": str(protocol_path),
            "task_manifest": str(locked_task_manifest),
        }
    )
    campaign_path = tmp_path / "locked-campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")

    with pytest.raises(SchemaError, match="locked.*mount"):
        freeze_campaign(
            repository_root=REPOSITORY_ROOT,
            campaign_config_path=campaign_path,
            campaign_dir=tmp_path / "forbidden-freeze",
            allow_dirty_pilot=True,
        )


def test_access_ledger_locked_read_flag_invalidates_pilot(tmp_path: Path):
    campaign = _freeze_smoke(tmp_path)
    append_campaign_ledger_event(
        campaign,
        ledger_type="access",
        event_type="forbidden_fixture_read",
        payload={"locked_test_content": True, "source_path": "locked/task.json"},
    )

    report = validate_campaign(campaign, require_complete=False)
    assert not report.passed
    assert any("locked-test read" in error for error in report.errors)
    assert report.publication_status == "PILOT_ONLY"


@pytest.mark.parametrize(
    "leaked_key",
    (
        "Oracle Success",
        "oracle-success",
        "ORACLE.SUCCESS",
        "Task Success",
        "relevance label",
        "Ground-Truth",
    ),
)
def test_oracle_leakage_guard_normalizes_case_and_punctuation(leaked_key: str):
    with pytest.raises(SchemaError, match="leaks sealed verifier keys"):
        assert_no_verifier_evidence(
            {"outer": [{"inner": {leaked_key: True}}]}, context="runtime"
        )


def _final_evidence(
    *, system_id: str = "E0", episode_id: str = "fixture"
) -> dict[str, Any]:
    has_recovery = system_id in {"E2", "E3"}
    incident_id = f"{episode_id}:incident-1"
    attempt_id = f"{episode_id}:attempt-1"
    query_id = f"{episode_id}:query-1"
    evidence: dict[str, Any] = {
        "task_success": True,
        "terminal_reason": "TASK_SUCCESS",
        "loop_detected": False,
        "environment_failure": False,
        "failure_incidents": [],
        "recovery_verifications": [],
        "verified_failure_event_count": 0,
        "repeated_error_event_count": 0,
        "memory_relevance": {},
    }
    if has_recovery:
        evidence.update(
            {
                "failure_incidents": [
                    {
                        "failure_incident_id": incident_id,
                        "verified_agent_failure": True,
                        "resolved": True,
                        "resolved_attempt_index": 1,
                        "attempt_count": 1,
                    }
                ],
                "recovery_verifications": [
                    {
                        "recovery_attempt_id": attempt_id,
                        "failure_incident_id": incident_id,
                        "verified_failure_present": True,
                        "successful": True,
                    }
                ],
                "verified_failure_event_count": 1,
            }
        )
    if system_id == "E3":
        evidence["memory_relevance"] = {
            query_id: {
                "relevant_ids": ["memory-2"],
                "relevance_definition": "registered_item_relevance",
                "useful_intervention": True,
                "harmful_intervention": False,
            }
        }
    return evidence


def test_sealed_verifier_returns_only_opaque_signal_and_detects_tampering(tmp_path: Path):
    sink = SealedVerifierSink(
        tmp_path,
        block_id="task_fixture__seed_42__repeat_000",
        attempt_id=0,
        system_id="E0",
        episode_id="fixture:E0",
        matched_seed=42,
        task_id="fixture",
        repeat_id=0,
    )
    signal = sink.record_episode_final(_final_evidence())

    assert signal.terminate is True
    assert len(signal.token_sha256) == 64
    assert not hasattr(signal, "evidence")
    records = verify_sealed_stream(sink.path)
    assert records[0]["evidence"]["task_success"] is True

    original = sink.path.read_text(encoding="utf-8")
    record = json.loads(original)
    record["opaque_token_sha256"] = "0" * 64
    sink.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="bad opaque token"):
        verify_sealed_stream(sink.path)

    record = json.loads(original)
    record["evidence"]["task_success"] = False
    sink.path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="bad record hash"):
        verify_sealed_stream(sink.path)


def test_evaluator_receives_only_a_write_only_sealed_capability(tmp_path: Path):
    sink = SealedVerifierSink(
        tmp_path,
        block_id="task_fixture__seed_42__repeat_000",
        attempt_id=0,
        system_id="E0",
        episode_id="fixture:E0",
        matched_seed=42,
        task_id="fixture",
        repeat_id=0,
    )
    writer = SealedVerifierWriter(sink)

    for forbidden in (
        "campaign_dir",
        "path",
        "sealed_root",
        "system_root",
        "has_episode_final",
        "_key",
        "_previous_hash",
        "__dict__",
    ):
        with pytest.raises(AttributeError):
            getattr(writer, forbidden)
    signal = writer.record_episode_final(_final_evidence())
    assert signal.terminate is True
    assert sink.has_episode_final


def _status(
    *, launched: bool, completed: bool, infrastructure_invalid: bool = False
) -> dict[str, Any]:
    value = {
        "launched": launched,
        "completed": completed,
        "infrastructure_invalid": infrastructure_invalid,
    }
    if infrastructure_invalid:
        value.update(
            {
                "infrastructure_reason": "ENGINEERING_SMOKE_ONLY",
                "infrastructure_evidence_sha256": SHA_A,
            }
        )
    return value


def _attempt(attempt_id: int, statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"attempt_id": attempt_id, "systems": statuses}


def test_whole_block_rerun_never_selects_a_partial_attempt():
    row = {"block_id": "block-1", "max_block_attempts": 2}
    interrupted = _attempt(
        0,
        {
            "E0": _status(launched=True, completed=True),
            "E1": _status(
                launched=True,
                completed=False,
                infrastructure_invalid=True,
            ),
            "E2": _status(launched=False, completed=False),
            "E3": _status(launched=False, completed=False),
        },
    )
    first = resolve_block_attempts(row, [interrupted])
    assert first["status"] == "RERUN_REQUIRED"
    assert first["selected_attempt_id"] is None
    assert first["attempts"][0]["infrastructure_invalid"] is True

    complete = _attempt(
        1,
        {
            system_id: _status(launched=True, completed=True)
            for system_id in SYSTEM_IDS
        },
    )
    second = resolve_block_attempts(row, [interrupted, complete])
    assert second["status"] == "INCLUDED"
    assert second["selected_attempt_id"] == 1
    assert second["attempts"][0]["valid"] is False
    assert second["attempts"][1]["valid"] is True


def test_partial_attempt_without_typed_infrastructure_evidence_cannot_rerun():
    row = {"block_id": "block-unauthorized", "max_block_attempts": 2}
    partial = _attempt(
        0,
        {
            "E0": _status(launched=True, completed=False),
            "E1": _status(launched=False, completed=False),
            "E2": _status(launched=False, completed=False),
            "E3": _status(launched=False, completed=False),
        },
    )
    assert resolve_block_attempts(row, [partial])["status"] == (
        "INTERRUPTED_UNAUTHORIZED"
    )


class _NestedEvidenceRunner:
    def __init__(
        self,
        *,
        fail_first_launch: bool = False,
        mismatched_reset_system: str | None = None,
    ) -> None:
        self.fail_first_launch = fail_first_launch
        self.mismatched_reset_system = mismatched_reset_system
        self.run_calls = 0

    def run(
        self,
        *,
        system_id: str,
        event_logs: Any,
        rerun_id: int,
        task: dict[str, Any],
        repeat_id: int,
        model_seed: int,
        protocol: dict[str, Any],
        stage_seeds: dict[str, int],
        episode_id: str,
        verifier_writer: SealedVerifierWriter,
        **_: Any,
    ) -> dict[str, Any]:
        self.run_calls += 1
        if self.fail_first_launch and self.run_calls == 1:
            raise InfrastructureInvalidError(
                reason_code="ENGINEERING_SMOKE_ONLY",
                adapter_id="deterministic-smoke-adapter",
                adapter_version="v1",
                operation="fixture_reset",
                adapter_evidence={
                    "adapter_event_id": "smoke-interruption-1",
                    "failure_class": "deterministic_fixture",
                    "diagnostic_sha256": SHA_A,
                    "retryable": True,
                },
            )

        from web_agent.runtime.protocol import StageRNGFactory

        campaign_id = episode_id.split(":", maxsplit=1)[0]
        campaign_seed = int(protocol["statistics"]["seed"])
        rng_factory = StageRNGFactory(
            protocol_id=str(protocol["protocol_id"]),
            campaign_id=campaign_id,
            campaign_seed=campaign_seed,
        )
        rng_key = {
            "campaign_id": campaign_id,
            "task_id": str(task["task_id"]),
            "repeat_id": repeat_id,
            "matched_seed": model_seed,
            "stage": "reset",
            "decision_index": 0,
            "incident_index": 0,
            "attempt_index": 0,
            "stream": "default",
        }
        event_logs.append(
            "environment_events",
            "rng_seed_plan",
            {
                "protocol_id": str(protocol["protocol_id"]),
                "campaign_id": campaign_id,
                "campaign_seed": campaign_seed,
                "task_id": str(task["task_id"]),
                "repeat_id": repeat_id,
                "matched_seed": model_seed,
                "decision_index": 0,
                "stage_seeds": stage_seeds,
            },
        )
        event_logs.append(
            "environment_events",
            "stage_rng_use",
            {
                "protocol_id": str(protocol["protocol_id"]),
                "campaign_seed": campaign_seed,
                "key": rng_key,
                "seed": rng_factory.seed_for(
                    task_id=str(task["task_id"]),
                    repeat_id=repeat_id,
                    matched_seed=model_seed,
                    stage="reset",
                    decision_index=0,
                ),
            },
        )
        reset_observation_id = f"{episode_id}:obs-reset"
        reset_observation = {
            "schema_version": "table2.runtime.v1",
            "record_type": "Observation",
            "observation_id": reset_observation_id,
            "episode_id": episode_id,
            "stage": "reset",
            "screenshot_sha256": SHA_C,
            "screenshot_path": None,
            "width": 1280,
            "height": 720,
            "url": "fixture://reset",
            "title": (
                "Synthetic package-validator fixture (mismatched reset)"
                if system_id == self.mismatched_reset_system
                else "Synthetic package-validator fixture"
            ),
            "page_state": {"state_id": "reset"},
            "page_settled": True,
            "environment_error": False,
            "prior_action_id": None,
        }
        reset_observation_sha256 = sha256_json(reset_observation)
        event_logs.append(
            "environment_events",
            "reset",
            {
                **reset_observation,
                "observation_record_sha256": reset_observation_sha256,
            },
        )
        reset_binding = VerifierReceiptBinding(
            receipt_kind="after_reset",
            observation_id=reset_observation_id,
            observation_sha256=reset_observation_sha256,
        )
        reset_signal = verifier_writer.record_bound_receipt(
            reset_binding,
            {"synthetic_fixture_receipt": "reset"},
            should_terminate=False,
        )
        event_logs.append(
            "terminal_signals",
            "after_reset",
            {
                **reset_signal.to_dict(),
                "receipt_binding": reset_binding.to_dict(),
            },
        )

        recovery_count = 0
        incident_id = f"{episode_id}:incident-1"
        attempt_id = f"{episode_id}:attempt-1"
        query_id = f"{episode_id}:query-1"
        if system_id in {"E2", "E3"}:
            recovery_count = 1
            shadow = {
                "decision_id": f"{episode_id}:shadow-1",
                "incident_id": incident_id,
                "strategy": "REPLAN",
                "trigger_sources": ["policy"],
                "diagnosis": "NO_EFFECT",
                "planned_action": None,
            }
            event_logs.append(
                "transitions",
                "post_action_assessment",
                {"assessment": {"failure_type": "NO_EFFECT", "latency_ms": 0.5}},
            )
            event_logs.append(
                "recoveries",
                "recovery_plan",
                {
                    "shadow_decision": shadow,
                    "final_decision": shadow,
                    "plan": {
                        "attempt_id": attempt_id,
                        "incident_id": incident_id,
                        "strategy": "REPLAN",
                        "incident_attempt_index": 1,
                        "episode_attempt_index": 1,
                        "actions": [],
                        "resolution_status": "REJECTED",
                        "rejection_reason": "synthetic smoke-only fixture",
                    },
                },
            )
            event_logs.append(
                "recoveries",
                "recovery_attempt",
                {
                    "attempt": {
                        "attempt_id": attempt_id,
                        "incident_id": incident_id,
                        "incident_attempt_index": 1,
                        "episode_attempt_index": 1,
                        "strategy": "REPLAN",
                        "action_ids": [],
                        "completed": False,
                        "predicted_assessment_id": None,
                    },
                    "predicted_assessment": {"latency_ms": 0.25},
                },
            )
        if system_id == "E3":
            event_logs.append(
                "memory_queries", "no_memory_shadow_before_retrieval", shadow
            )
            event_logs.append(
                "memory_queries",
                "post_failure_query",
                {
                    "shadow_decision": shadow,
                    "final_decision": shadow,
                    "query_result": {
                        "query_id": query_id,
                        "candidate_ids": ["memory-1", "memory-2", "memory-3"],
                        "scores": [0.9, 0.8, 0.7],
                        "shadow_decision_sha256": sha256_json(shadow),
                        "admitted": False,
                        "changed_strategy": False,
                        "changed_target_or_parameters": False,
                        "latency_ms": 0.75,
                    }
                },
            )
        return {
            "completed": True,
            "infrastructure_invalid": False,
            "step_count": 0,
            "executed_action_count": 0,
            "rejected_action_count": 0,
            "recovery_action_count": 0,
            "recovery_attempt_count": recovery_count,
            "model_call_count": 1,
            "task_wall_clock_seconds": 0.01,
            "rerun_id": rerun_id,
        }

    def finalize_episode(
        self,
        *,
        verifier_writer: SealedVerifierWriter,
        system_id: str,
        episode_id: str,
        task: dict[str, Any],
        **_: Any,
    ) -> Any:
        evidence = _final_evidence(system_id=system_id, episode_id=episode_id)
        task_success = self.task_success(system_id=system_id, task=task)
        evidence["task_success"] = task_success
        evidence["terminal_reason"] = (
            "TASK_SUCCESS" if task_success else "TERMINAL_FAILURE"
        )
        return verifier_writer.record_episode_final(evidence)

    def task_success(self, *, system_id: str, task: dict[str, Any]) -> bool:
        return True


class _CompletePilotRunner(_NestedEvidenceRunner):
    def task_success(self, *, system_id: str, task: dict[str, Any]) -> bool:
        task_id = str(task["task_id"])
        if task_id.startswith("webarena.") and system_id == "E0":
            return False
        if task_id.startswith("webarena.") and system_id == "E3":
            return int(task_id.rsplit(".", maxsplit=1)[1]) >= 3
        return True


class _LateInfrastructureRunner(_NestedEvidenceRunner):
    """Canary: consume one request before the first block is invalidated."""

    def run(self, *, event_logs: Any, episode_id: str, **kwargs: Any) -> dict[str, Any]:
        result = super().run(
            event_logs=event_logs,
            episode_id=episode_id,
            **kwargs,
        )
        if self.run_calls == 1:
            event_logs.append(
                "actions",
                "normal_action",
                {
                    "action": {
                        "action_id": f"{episode_id}:late-infra-action",
                    },
                    "execution": {
                        "action_id": f"{episode_id}:late-infra-action",
                        "status": "ERROR",
                        "executor_step": 1,
                        "state_changed": False,
                        "environment_error": True,
                        "error_kind": "infrastructure_interruption",
                    },
                    "interrupted": True,
                },
            )
            raise InfrastructureInvalidError(
                reason_code="ENGINEERING_SMOKE_ONLY",
                adapter_id="deterministic-smoke-adapter",
                adapter_version="v1",
                operation="fixture_late_controller_disconnect",
                adapter_evidence={
                    "adapter_event_id": "smoke-late-interruption-1",
                    "failure_class": "controller_disconnected_after_request",
                    "diagnostic_sha256": SHA_B,
                    "retryable": True,
                },
            )
        return result


def test_campaign_runner_resumes_whole_block_and_validator_parses_nested_evidence(
    tmp_path: Path,
):
    campaign = _freeze_smoke(tmp_path)
    runner = _NestedEvidenceRunner(fail_first_launch=True)
    campaign_runner = CampaignRunner(campaign_dir=campaign, runner=runner)
    scheduled = read_jsonl(campaign / "schedule/schedule.jsonl")[0]

    resolution = campaign_runner.run_block(scheduled)
    assert resolution["status"] == "INCLUDED"
    assert resolution["selected_attempt_id"] == 1
    assert sum(
        status["launched"]
        for status in resolution["attempts"][0]["systems"].values()
    ) == 1
    assert all(
        status["completed"]
        for status in resolution["attempts"][1]["systems"].values()
    )

    calls_after_completion = runner.run_calls
    resumed = campaign_runner.run_block(str(scheduled["block_id"]))
    assert resumed == resolution
    assert runner.run_calls == calls_after_completion

    block_root = (
        campaign
        / "paired_blocks"
        / f"seed_{int(scheduled['matched_model_seed'])}"
        / _path_id(str(scheduled["task_id"]))
        / f"repeat_{int(scheduled['repeat_id'])}"
    )
    assert (block_root / "rerun_0/block_manifest.json").is_file()
    assert (block_root / "rerun_1/block_manifest.json").is_file()
    assert (block_root / "rerun_1/E0/runtime/episode_summary.json").is_file()
    assert (block_root / "rerun_1/E3/runtime/memory_queries.jsonl").is_file()
    assert (block_root / "rerun_1/E3/sealed").is_dir()

    report = validate_campaign(campaign, require_complete=False)
    assert report.passed, report.errors
    assert report.counts["included_blocks"] == 1
    assert report.counts["block_status_not_started"] == 64
    assert report.publication_status == "PILOT_ONLY"

    append_campaign_ledger_event(
        campaign,
        ledger_type="access",
        event_type="episode_task_load",
        payload={
            "block_id": str(scheduled["block_id"]),
            "task_id": str(scheduled["task_id"]),
            "task_partition": str(scheduled["task_partition"]),
            "system_id": "E0",
            "attempt_id": 99,
            "episode_id": "unregistered-extra-launch",
            "locked_test_content": False,
        },
    )
    unreconciled = validate_campaign(campaign, require_complete=False)
    assert not unreconciled.passed
    assert any(
        "task-load ledger does not match physical launches" in error
        for error in unreconciled.errors
    )


def test_late_infrastructure_fault_preserves_consumed_work(tmp_path: Path) -> None:
    campaign = _freeze_smoke(tmp_path)
    runner = _LateInfrastructureRunner()
    campaign_runner = CampaignRunner(campaign_dir=campaign, runner=runner)
    scheduled = read_jsonl(campaign / "schedule/schedule.jsonl")[0]

    resolution = campaign_runner.run_block(scheduled)
    assert resolution["status"] == "INCLUDED"
    assert resolution["selected_attempt_id"] == 1
    invalid_system = next(
        system_id
        for system_id, status in resolution["attempts"][0]["systems"].items()
        if status["infrastructure_invalid"]
    )
    runtime = (
        campaign
        / "paired_blocks"
        / f"seed_{int(scheduled['matched_model_seed'])}"
        / _path_id(str(scheduled["task_id"]))
        / f"repeat_{int(scheduled['repeat_id'])}"
        / "rerun_0"
        / invalid_system
        / "runtime"
    )
    summary = read_json(runtime / "episode_summary.json")
    assert summary["step_count"] == 1
    assert summary["executed_action_count"] == 0
    assert summary["rejected_action_count"] == 1
    assert summary["task_wall_clock_seconds"] >= 0.0
    assert read_jsonl(runtime / "actions.jsonl")[0]["payload"]["execution"][
        "executor_step"
    ] == 1

    report = validate_campaign(campaign, require_complete=False)
    assert report.passed, report.errors


def test_included_block_rejects_mismatched_e0_e3_reset_fingerprint(
    tmp_path: Path,
):
    campaign = _freeze_smoke(tmp_path)
    runner = _NestedEvidenceRunner(mismatched_reset_system="E1")
    campaign_runner = CampaignRunner(campaign_dir=campaign, runner=runner)
    scheduled = read_jsonl(campaign / "schedule/schedule.jsonl")[0]

    resolution = campaign_runner.run_block(scheduled)
    assert resolution["status"] == "INCLUDED"

    report = validate_campaign(campaign, require_complete=False)
    assert not report.passed
    assert any(
        "E0--E3 normalized reset state fingerprints differ" in error
        for error in report.errors
    )


def test_out_of_range_physical_rerun_directory_is_rejected(tmp_path: Path):
    base = tmp_path / "paired-block"
    for attempt_id in (0, 2):
        rerun = base / f"rerun_{attempt_id}"
        rerun.mkdir(parents=True)
        _write_json(
            rerun / "block_manifest.json",
            {"attempt_id": attempt_id, "block_id": "block", "systems": {}},
        )

    with pytest.raises(SchemaError, match="out-of-range physical attempt"):
        discover_block_attempt_directories(base, maximum=2)


def test_recall_at_five_is_unavailable_when_frozen_retrieval_depth_is_three():
    result = compute_retrieval_diagnostics(
        [
            {
                "query_id": "q1",
                "system_id": "E3",
                "retrieved_ids": ["m1", "m2", "m3"],
                "relevant_ids": ["m3", "m4"],
                "relevance_definition": "registered_item_relevance",
                "abstained": False,
                "admitted": True,
                "changed_decision": True,
                "useful_intervention": True,
                "harmful_intervention": False,
                "shadow_decision_hash": SHA_A,
                "query_latency_ms": 1.0,
                "index_size": 10,
                "retrieval_depth": 3,
            }
        ],
        recall_ks=(1, 3, 5),
    )

    assert result["recall_at_1"]["estimate"] == 0.0
    assert result["recall_at_1"]["numerator"] == 0.0
    assert result["recall_at_1"]["denominator"] == 1
    assert result["recall_at_3"]["estimate"] == 0.5
    assert result["recall_at_3"]["numerator"] == 0.5
    assert result["recall_at_5"]["estimate"] is None
    assert result["recall_at_5"]["display"] == "N/A"


def test_registered_strategy_relevance_emits_only_strategy_hit_names_and_counts():
    result = compute_retrieval_diagnostics(
        [
            {
                "query_id": "q-registered-strategy-1",
                "system_id": "E3",
                "retrieved_ids": ["m-wrong", "m-right", "m-other"],
                "relevant_ids": ["m-right"],
                "relevance_definition": (
                    "matching_verified_successful_correction_strategy_v1"
                ),
                "abstained": True,
                "admitted": False,
                "changed_decision": False,
                "changed_strategy": False,
                "changed_target_or_parameters": False,
                "shadow_decision_hash": "a" * 64,
                "retrieval_depth": 3,
            },
            {
                "query_id": "q-registered-strategy-2",
                "system_id": "E3",
                "retrieved_ids": ["m-right", "m-wrong", "m-other"],
                "relevant_ids": ["m-right"],
                "relevance_definition": (
                    "matching_verified_successful_correction_strategy_v1"
                ),
                "abstained": True,
                "admitted": False,
                "changed_decision": False,
                "changed_strategy": False,
                "changed_target_or_parameters": False,
                "shadow_decision_hash": "b" * 64,
                "retrieval_depth": 3,
            },
        ]
    )

    assert not any(key.startswith("recall_at_") for key in result)
    assert result["strategy_hit_at_1"]["numerator"] == 1
    assert result["strategy_hit_at_1"]["denominator"] == 2
    assert result["strategy_hit_at_1"]["estimate"] == 0.5
    assert result["strategy_hit_at_3"]["numerator"] == 2
    assert result["strategy_hit_at_3"]["denominator"] == 2
    assert result["strategy_hit_at_3"]["estimate"] == 1.0
    assert result["strategy_hit_at_5"]["numerator"] is None
    assert result["strategy_hit_at_5"]["denominator"] == 2
    assert result["strategy_hit_at_5"]["display"] == "N/A"
    assert result["mean_reciprocal_rank"]["numerator"] == 1.5
    assert result["mean_reciprocal_rank"]["denominator"] == 2
    assert result["mean_reciprocal_rank"]["estimate"] == 0.75


def test_complete_pilot_writes_deterministic_audit_digests_and_exact_result_export(
    tmp_path: Path,
):
    campaign = _freeze_smoke(tmp_path, name="complete-pilot")
    completion = CampaignRunner(
        campaign_dir=campaign,
        runner=_CompletePilotRunner(),
    ).run()
    assert completion["status"] == "COMPLETE"
    assert completion["scheduled_block_count"] == 65
    assert completion["included_block_count"] == 65

    results_dir = tmp_path / "results/table2/complete-pilot-id"
    with pytest.raises(
        Table2Error,
        match="final PILOT_ONLY summary/export requires completed",
    ):
        summarize_campaign(campaign, results_dir=results_dir)
    assert not results_dir.exists()

    summary = summarize_campaign(
        campaign,
        results_dir=results_dir,
        draft_pilot=True,
    )
    assert summary["publication_status"] == "DRAFT_PILOT_ONLY"
    assert summary["paper_table_status"] == "N/R"
    assert Path(summary["results_dir"]) == results_dir.resolve()

    expected_results = {
        "pilot_summary.json",
        "table2_main.csv",
        "table2_companion.csv",
        "paired_contrasts.csv",
        "retrieval_metrics.csv",
        "statistics.json",
        "provenance.json",
    }
    assert {path.name for path in results_dir.iterdir() if path.is_file()} == expected_results
    assert read_json(results_dir / "pilot_summary.json")["paper_table_status"] == "N/R"
    assert read_json(results_dir / "provenance.json")["contains_raw_oracle_evidence"] is False
    exported_statistics = read_json(results_dir / "statistics.json")
    for contrast in exported_statistics["contrasts"].values():
        significance = contrast["task_success"]["task_clustered_significance"]
        assert "task_effects" not in significance
        assert significance["per_task_effects_exported"] is False
    with (results_dir / "table2_main.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert all(
            row["Evidence Status"] == "DRAFT_PILOT_ONLY"
            for row in csv.DictReader(handle)
        )

    selection_path = campaign / "manual_audit/selection_manifest.json"
    selection = read_json(selection_path)
    assert selection["selected_count"] == 18
    assert selection["total_shortfall"] == 2
    strata = {row["name"]: row for row in selection["strata"]}
    assert {name: row["selected_count"] for name, row in strata.items()} == {
        "ordinary_success": 5,
        "terminal_failure": 5,
        "recovery": 5,
        "e2_e3_disagreement": 3,
    }
    assert strata["e2_e3_disagreement"]["shortfall"] == 2
    assert all(row["substitution_allowed"] is False for row in strata.values())
    assert all("task_success" not in row for row in selection["selected"])
    labels_path = campaign / "manual_audit/sealed/selection_labels.json"
    assert labels_path.is_file()
    assert labels_path.stat().st_mode & 0o077 == 0
    assert selection["outcome_labels_sha256"] == sha256_file(labels_path)
    repeated_selection = build_manual_audit_selection(
        campaign, load_selected_analysis_records(campaign)
    )
    assert repeated_selection == selection

    aggregate_hashes = read_json(campaign / "aggregate/artifact_hashes.json")
    assert aggregate_hashes["excluded_mutable_files"] == ["validation_report.json"]
    assert aggregate_hashes["files"]
    assert all(
        sha256_file(campaign / "aggregate" / relative) == digest
        for relative, digest in aggregate_hashes["files"].items()
    )
    evidence = read_json(campaign / "campaign_evidence_manifest.json")
    assert evidence["files"]
    assert all(
        sha256_file(campaign / relative) == digest
        for relative, digest in evidence["files"].items()
    )

    validation = validate_campaign(
        campaign, require_complete=True, require_aggregates=True
    )
    assert validation.passed, validation.errors
    assert validation.publication_status == "DRAFT_PILOT_ONLY"

    # Hashes alone are not a numeric trust root: simulate an attacker changing a
    # registered result and regenerating both aggregate/campaign hash manifests.
    metrics_path = campaign / "aggregate/metrics.json"
    tampered_metrics = read_json(metrics_path)
    tampered_metrics["systems"]["E0"]["task_success_rate"]["estimate"] = 0.123456
    _write_json(metrics_path, tampered_metrics)
    _write_aggregate_hashes(campaign / "aggregate")
    _write_campaign_evidence_manifest(campaign)

    tampered_validation = validate_campaign(
        campaign, require_complete=True, require_aggregates=True
    )
    assert not tampered_validation.passed
    assert any(
        "aggregate numeric recomputation mismatch: metrics.json" in error
        for error in tampered_validation.errors
    )


def test_locked_final_readiness_requires_signed_hash_bound_human_adjudication(
    tmp_path: Path,
):
    root = tmp_path / "locked-final"
    audit_root = root / "manual_audit"
    sealed_root = audit_root / "sealed"
    sealed_root.mkdir(parents=True)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "locked-final-1",
        "campaign_kind": "locked_final",
        "evidence_label": "FINAL_LOCKED",
    }
    _write_json(root / "campaign_manifest.json", campaign)
    selection = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "locked-final-1",
        "selected_count": 2,
        "selected": [{"audit_id": "audit-01"}, {"audit_id": "audit-02"}],
    }
    selection_path = _write_json(audit_root / "selection_manifest.json", selection)
    report = ValidationReport(campaign_dir=str(root))
    report.counts["included_blocks"] = 1

    assert _publication_status(root, campaign, report, True) == "N/R"
    pilot = {
        **campaign,
        "campaign_kind": "engineering_pilot",
        "evidence_label": "PILOT_ONLY",
    }
    assert _publication_status(root, pilot, report, True) == "PILOT_ONLY"

    selection_hash = sha256_file(selection_path)
    timestamp = "2026-08-31T12:00:00+00:00"
    reviewers = ["reviewer-a", "reviewer-b"]
    signers = [*reviewers, "adjudicator-c"]
    adjudication = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "locked-final-1",
        "selection_manifest_sha256": selection_hash,
        "status": "COMPLETE",
        "all_selected_items_reviewed": True,
        "all_disagreements_resolved": True,
        "reviewer_ids": reviewers,
        "adjudicator_id": "adjudicator-c",
        "signed_by": signers,
        "signed_at_utc": timestamp,
        "records": [
            {
                "audit_id": "audit-01",
                "reviewer_labels": {
                    "reviewer-a": {"success": True},
                    "reviewer-b": {"success": True},
                },
                "final_label": {"success": True},
                "disagreement": False,
                "resolved": True,
            },
            {
                "audit_id": "audit-02",
                "reviewer_labels": {
                    "reviewer-a": {"success": True},
                    "reviewer-b": {"success": False},
                },
                "final_label": {"success": False},
                "disagreement": True,
                "resolved": True,
            },
        ],
    }
    adjudication_path = _write_json(
        sealed_root / "adjudication.json", adjudication
    )
    adjudication_path.chmod(0o600)
    agreement = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "locked-final-1",
        "selection_manifest_sha256": selection_hash,
        "status": "COMPLETE",
        "reviewer_ids": reviewers,
        "sample_size": 2,
        "disagreement_count": 1,
        "adjudicated_disagreement_count": 1,
        "raw_agreement": 0.5,
        "cohen_kappa": 0.0,
        "calculation_method": "registered two-reviewer categorical agreement",
        "signed_by": reviewers,
        "signed_at_utc": timestamp,
    }
    agreement_path = _write_json(audit_root / "agreement.json", agreement)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "locked-final-1",
        "selection_manifest_sha256": selection_hash,
        "completion_status": "HUMAN_ADJUDICATION_COMPLETE",
        "adjudication_relative_path": "sealed/adjudication.json",
        "adjudication_sha256": sha256_file(adjudication_path),
        "agreement_relative_path": "agreement.json",
        "agreement_sha256": sha256_file(agreement_path),
        "signed_by": signers,
        "signed_at_utc": timestamp,
    }
    _write_json(audit_root / "completion_manifest.json", completion)

    validated = validate_manual_adjudication_completion(root)
    assert validated["selected_count"] == 2
    assert validated["disagreement_count"] == 1
    assert _publication_status(root, campaign, report, True) == FINAL_READY_STATUS

    agreement["raw_agreement"] = 1.0
    _write_json(agreement_path, agreement)
    with pytest.raises(SchemaError):
        validate_manual_adjudication_completion(root)
    assert _publication_status(root, campaign, report, True) == "N/R"
