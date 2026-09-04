from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from web_agent.memory.joint_duplicate_audit import (
    JointDuplicateAuditError,
    build_joint_duplicate_assignment_package,
    finalize_joint_duplicate_audit,
    parse_args as parse_joint_duplicate_args,
    provenance_assignment_binding,
    validate_compact_joint_duplicate_evidence,
    validate_final_joint_duplicate_audit,
    validate_joint_duplicate_assignment_package,
)
from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.package_validator import (
    _validate_registered_joint_duplicate_memory_bindings,
)
from web_agent.memory import build_pipeline
from web_agent.memory.builder import build_frozen_store
from web_agent.memory.calibration_builder import build_calibration_evidence
from web_agent.memory.eligibility import EligibilitySelection, ProvenanceManifest
from web_agent.memory.frozen_store import FrozenMemoryStore
from web_agent.memory.manifest import ManifestError, canonical_sha256, sha256_file
from web_agent.memory.preparation import (
    _prepare_p4_candidate_audit_fixture,
    reconstruct_p4_selection_from_preparation,
)
from web_agent.memory.verification import verification_bundle_sha256
from web_agent.runtime.duplicate_audit import (
    FrozenDuplicateAuditManifest,
    JointDuplicateClusterNamespace,
    canonical_task_audit_record,
    canonical_train_corpus_binding,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/eval/table2/joint_duplicate_audit_v1.json"
)
RECOVERY_SCENARIOS = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/table2/pilot/recovery_scenarios.json"
)
DUPLICATE_REGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/table2/pilot/duplicate_audit_manifest.json"
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _gold_row(
    *,
    sample_id: str,
    task_id: str,
    step: int,
    before: str,
    after: str,
    description: str,
    outcome: str,
    action: str,
    strategy: str,
    recovery_success: bool | None,
    memory_update: bool,
) -> dict:
    return {
        "inputs": {
            "state_before": before,
            "state_after": after,
            "task_description": description,
            "website_domain": "reddit",
        },
        "labels": {
            "outcome_label": outcome,
            "failure_type_4": "ACTION_MISMATCH" if outcome == "FAILURE" else "NONE",
            "action_type": action,
            "action_value": "",
            "recovery_strategy": strategy,
            "recovery_attempted": recovery_success is not None,
            "recovery_success": recovery_success,
            "memory_update_flag": memory_update,
            "reflection_text": "fixture only",
        },
        "meta": {
            "sample_id": sample_id,
            "task_id": task_id,
            "trajectory_id": task_id,
            "step_index": step,
            "review_status": "approved",
            "split": "train",
        },
    }


def _fixture_inputs(
    tmp_path: Path,
    *,
    pending_candidate_indices: frozenset[int] = frozenset(),
    include_direct_supplement: bool = False,
) -> dict[str, Path]:
    gold_root = tmp_path / "gold"
    descriptions = (
        "open account settings and change profile name",
        "open account settings and change profile name now",
        "find the archived invoice for project zephyr",
        "restore the deleted repository milestone for project orion",
        "rebuild the analytics dashboard for project lyra",
    )
    rows: list[dict] = []
    for index, description in enumerate(descriptions):
        before = f"images/{index}/before.png"
        failed = f"images/{index}/failed.png"
        recovered = f"images/{index}/recovered.png"
        for relative, payload in (
            (before, f"before-{index}".encode()),
            (failed, f"failed-{index}".encode()),
            (recovered, f"recovered-{index}".encode()),
        ):
            target = gold_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        candidate = _gold_row(
            sample_id=f"candidate-{index}",
            task_id=f"gold-task-{index}",
            step=0,
            before=before,
            after=failed,
            description=description,
            outcome="FAILURE",
            action="CLICK",
            strategy="REPLAN" if index in {2, 4} else "RETRY",
            recovery_success=True,
            memory_update=True,
        )
        if index in pending_candidate_indices:
            candidate["meta"]["review_status"] = "pending"
        rows.extend(
            [
                candidate,
                _gold_row(
                    sample_id=f"recovery-{index}",
                    task_id=f"gold-task-{index}",
                    step=1,
                    before=failed,
                    after=recovered,
                    description=description,
                    outcome="SUCCESS",
                    action="TYPE",
                    strategy="NONE",
                    recovery_success=None,
                    memory_update=False,
                ),
            ]
        )
    gold_json = _write_json(gold_root / "split_train.json", rows)
    supplement_json: Path | None = None
    supplement_root: Path | None = None
    if include_direct_supplement:
        supplement_root = tmp_path / "supplement"
        direct_before = supplement_root / "images/direct-before.png"
        direct_after = supplement_root / "images/direct-after.png"
        direct_before.parent.mkdir(parents=True, exist_ok=True)
        direct_before.write_bytes(b"direct-failure-state")
        direct_after.write_bytes(b"direct-recovered-state")
        direct = _gold_row(
            sample_id="direct-retry-0",
            task_id="direct-task-0",
            step=0,
            before="images/direct-before.png",
            after="images/direct-after.png",
            description="retry the checkout submit action after a transient failure",
            outcome="SUCCESS",
            action="CLICK",
            strategy="RETRY",
            recovery_success=True,
            memory_update=True,
        )
        direct["labels"]["recovery_attempted"] = True
        supplement_json = _write_json(
            supplement_root / "data/supplement_train.json",
            [direct],
        )
    authority_sources = [
        {
            "file_name": "split_train.json",
            "records": len(rows),
            "role": "original_gold",
            "sha256": sha256_file(gold_json),
        }
    ]
    if supplement_json is not None:
        authority_sources.append(
            {
                "file_name": "supplement_train.json",
                "records": 1,
                "role": "retry_abort_supplement_v2",
                "sha256": sha256_file(supplement_json),
            }
        )
    source_authority = _write_json(
        tmp_path / "source-authority.json",
        {
            "authority_id": "table2-synthetic-unit-source-authority",
            "authority_version": "fixture-v1",
            "dataset_id": "fixture-gold",
            "dataset_version": "fixture-v1",
            "schema_version": "table2-p4-source-authority-v1",
            "sources": authority_sources,
        },
    )
    preparation = _prepare_p4_candidate_audit_fixture(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        supplement_train_json=supplement_json,
        supplement_data_root=supplement_root,
        output_dir=tmp_path / "preparation",
        dataset_id="fixture-gold",
        dataset_version="fixture-v1",
        source_authority_path=source_authority,
    )
    assert preparation.status == "REVIEW_REQUIRED"
    assert preparation.candidate_count == 5 + int(include_direct_supplement)

    registry = {
        "schema_version": "1.0",
        "manifest_id": "user-approved-compatible-fixture-50",
        "benchmark": "webarena",
        "partition": "development",
        "registration_status": "FROZEN_DEVELOPMENT_EXCLUSION",
        "required_task_count": 50,
        "final_paper_evaluation_eligible": False,
        "locked_test_content": False,
        "tasks": [
            {"task_id": f"webarena.{100 + index}", "upstream_index": 100 + index}
            for index in range(50)
        ],
    }
    registry_path = _write_json(tmp_path / "approved-registry.json", registry)
    tasks = []
    for index in range(50):
        if index == 0:
            instruction = descriptions[0]
            site = "reddit"
        elif index == 1:
            instruction = descriptions[1]
            site = "reddit"
        else:
            instruction = f"perform unique operation {index} for catalog item {index}"
            site = f"site-{index}"
        task_id = f"webarena.{100 + index}"
        task_config = {
            "sites": [site],
            "intent": instruction,
            "start_url": f"https://{site}.example.test/start/{index}",
            "require_login": True,
            "storage_state": f"./.auth/{site}.json",
            "geolocation": None,
            "require_reset": False,
            "eval": {"eval_types": ["url_match"], "reference_url": "redacted"},
        }
        tasks.append(
            {
                "task_id": task_id,
                "upstream_index": 100 + index,
                "benchmark_task_id": str(100 + index),
                "benchmark_task_version": "fixture-v1",
                "instruction": instruction,
                "start_state": {
                    key: task_config[key]
                    for key in (
                        "sites",
                        "start_url",
                        "require_login",
                        "storage_state",
                        "geolocation",
                        "require_reset",
                    )
                },
                "task_config": task_config,
                "evaluator": {
                    "evaluator_id": "fixture-official",
                    "evaluator_version": "fixture-v1",
                    "config": task_config["eval"],
                },
            }
        )
    export = {
        "schema_version": "table2-webarena-public-task-export-v1",
        "record_type": "SealedWebArenaDevelopmentTaskExport",
        "snapshot_id": "fixture-approved-50",
        "benchmark": "webarena",
        "benchmark_version": "fixture-v1",
        "task_definition_version": "fixture-v1",
        "browsergym_webarena_version": "fixture-v1",
        "libwebarena_version": "fixture-v1",
        "partition": "development",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "locked_test_content": False,
        "final_paper_evaluation_eligible": False,
        "selection_rule": "fixture user-approved compatible development tasks",
        "required_task_count": 50,
        "registry_manifest_id": registry["manifest_id"],
        "registry_manifest_sha256": sha256_file(registry_path),
        "source": {
            "container_kind": "raw_json",
            "container_name": "fixture.json",
            "container_sha256": SHA_A,
            "container_authorization": "FIXTURE",
            "authorized_container_sha256": SHA_A,
            "member": None,
            "task_source_sha256": SHA_B,
        },
        "site_url_map_sha256": SHA_C,
        "resolved_task_set_sha256": canonical_sha256(tasks),
        "tasks": tasks,
    }
    export_path = _write_json(tmp_path / "resolved-export.json", export)
    result = {
        "gold": gold_json,
        "source_authority": source_authority,
        "preparation": preparation.root,
        "registry": registry_path,
        "export": export_path,
        "recovery_scenarios": RECOVERY_SCENARIOS,
        "duplicate_registration": DUPLICATE_REGISTRATION,
    }
    if supplement_json is not None:
        result["supplement"] = supplement_json
    return result


def _build(
    tmp_path: Path,
    *,
    pending_candidate_indices: frozenset[int] = frozenset(),
    include_direct_supplement: bool = False,
) -> tuple[dict[str, Path], object]:
    inputs = _fixture_inputs(
        tmp_path,
        pending_candidate_indices=pending_candidate_indices,
        include_direct_supplement=include_direct_supplement,
    )
    package = build_joint_duplicate_assignment_package(
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        supplement_train_json=inputs.get("supplement"),
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        output_dir=tmp_path / "assignments",
    )
    return inputs, package


def test_joint_cli_requires_and_plumbs_source_authority():
    common = [
        "--config", "config.json",
        "--source-authority", "authority.json",
        "--preparation-package", "preparation",
        "--gold-train-json", "split_train.json",
        "--resolved-task-export", "tasks.json",
        "--approved-task-registry", "registry.json",
        "--recovery-scenarios", "recovery.json",
        "--duplicate-audit-registration", "registration.json",
    ]
    parsed = parse_joint_duplicate_args(
        ["build-assignments", *common, "--output-dir", "assignments"]
    )
    assert parsed.source_authority == Path("authority.json")
    with pytest.raises(SystemExit):
        parse_joint_duplicate_args(
            [
                "build-assignments",
                *common[:2],
                *common[4:],
                "--output-dir",
                "assignments",
            ]
        )


def _kwargs(inputs: dict[str, Path], package_root: Path) -> dict:
    return {
        "package_root": package_root,
        "config_path": CONFIG,
        "source_authority_path": inputs["source_authority"],
        "preparation_root": inputs["preparation"],
        "gold_train_json": inputs["gold"],
        "supplement_train_json": inputs.get("supplement"),
        "resolved_task_export_path": inputs["export"],
        "approved_task_registry_path": inputs["registry"],
        "recovery_scenarios_path": inputs["recovery_scenarios"],
        "duplicate_audit_registration_path": inputs["duplicate_registration"],
    }


def _compact_kwargs(inputs: dict[str, Path], package_root: Path) -> dict:
    values = _kwargs(inputs, package_root)
    values.pop("gold_train_json")
    values.pop("supplement_train_json")
    return values


def test_joint_assignment_detects_exact_near_and_keeps_one_namespace(tmp_path: Path):
    _, package = _build(tmp_path)
    by_id = {row["entity_id"]: row for row in package.entities}
    candidate = by_id["candidate:candidate-0"]
    exact_task = by_id["task:webarena.100"]
    near_task = by_id["task:webarena.101"]

    assert candidate["exact_duplicate_key"] == exact_task["exact_duplicate_key"]
    assert candidate["exact_duplicate_key"] != near_task["exact_duplicate_key"]
    assert candidate["near_duplicate_cluster_id"] == exact_task[
        "near_duplicate_cluster_id"
    ]
    assert candidate["near_duplicate_cluster_id"] == near_task[
        "near_duplicate_cluster_id"
    ]
    assert len(package.candidates) == 5
    assert len(package.tasks) == 50
    assert package.clusters["cross_corpus_cluster_count"] == 1
    ledger = json.loads((package.root / "read_ledger.json").read_text())
    assert ledger["validation_rows_read"] == 0
    assert ledger["test_rows_read"] == 0
    assert ledger["locked_test_rows_read"] == 0
    assert ledger["independent_recovery_reviews_read"] == 0
    assert ledger["independent_final_task_reviews_read"] == 0


def test_joint_assignment_is_byte_deterministic(tmp_path: Path):
    inputs, first = _build(tmp_path)
    second = build_joint_duplicate_assignment_package(
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        output_dir=tmp_path / "assignments-second",
    )
    for name in (
        "audit_config.json",
        "entities.jsonl",
        "clusters.json",
        "read_ledger.json",
        "source_manifest.json",
        "assignment_manifest.json",
        "assignment_manifest.sha256",
    ):
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes()


def test_joint_assignment_has_no_validation_or_test_input_surface(tmp_path: Path):
    inputs = _fixture_inputs(tmp_path)
    forbidden = tmp_path / "split_test.json"
    forbidden.write_bytes(inputs["gold"].read_bytes())
    with pytest.raises(JointDuplicateAuditError, match="accepts only split_train.json"):
        build_joint_duplicate_assignment_package(
            config_path=CONFIG,
            source_authority_path=inputs["source_authority"],
            preparation_root=inputs["preparation"],
            gold_train_json=forbidden,
            resolved_task_export_path=inputs["export"],
            approved_task_registry_path=inputs["registry"],
            recovery_scenarios_path=inputs["recovery_scenarios"],
            duplicate_audit_registration_path=inputs["duplicate_registration"],
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_joint_assignment_replay_rejects_self_consistent_entity_tamper(tmp_path: Path):
    inputs, package = _build(tmp_path)
    package.root.chmod(0o755)
    entities_path = package.root / "entities.jsonl"
    entities_path.chmod(0o644)
    rows = [json.loads(line) for line in entities_path.read_text().splitlines()]
    rows[0]["normalized_goal"] = "tampered goal"
    entities_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = package.root / "assignment_manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["entities.jsonl"] = {
        "sha256": sha256_file(entities_path),
        "bytes": entities_path.stat().st_size,
    }
    manifest["entities_sha256"] = canonical_sha256(rows)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    sidecar = package.root / "assignment_manifest.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")

    with pytest.raises(JointDuplicateAuditError, match="exact source replay"):
        validate_joint_duplicate_assignment_package(**_kwargs(inputs, package.root))


def test_joint_assignment_replay_rejects_changed_state_artifact_bytes(
    tmp_path: Path,
):
    """The raw-source gate must authenticate state bytes, not echoed hashes."""

    inputs, package = _build(tmp_path)
    changed_state = inputs["gold"].parent / "images/0/failed.png"
    changed_state.write_bytes(b"attacker-replaced-failure-state")

    with pytest.raises(
        JointDuplicateAuditError,
        match="source material differs from reopened train bytes.*pre_recovery_state_sha256",
    ):
        validate_joint_duplicate_assignment_package(**_kwargs(inputs, package.root))


def test_compact_replay_rejects_resealed_dependency_source_manifest(
    tmp_path: Path,
):
    """A package cannot replace the registered helper-source attestation."""

    inputs, package = _build(tmp_path)
    package.root.chmod(0o755)
    source_manifest_path = package.root / "source_manifest.json"
    source_manifest_path.chmod(0o644)
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["files"][0]["sha256"] = "9" * 64
    source_manifest["files_sha256"] = canonical_sha256(source_manifest["files"])
    _write_json(source_manifest_path, source_manifest)

    manifest_path = package.root / "assignment_manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["source_manifest.json"] = {
        "sha256": sha256_file(source_manifest_path),
        "bytes": source_manifest_path.stat().st_size,
    }
    manifest["input_binding"]["audit_dependency_source_set_sha256"] = (
        source_manifest["files_sha256"]
    )
    _write_json(manifest_path, manifest)
    sidecar = package.root / "assignment_manifest.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")

    with pytest.raises(
        JointDuplicateAuditError,
        match="input binding differs from compact evidence bytes|source dependencies differ",
    ):
        validate_compact_joint_duplicate_evidence(
            **_compact_kwargs(inputs, package.root)
        )


def _completed_provenance(inputs: dict[str, Path], package: object) -> dict:
    preparation_manifest = json.loads(
        (inputs["preparation"] / "preparation_manifest.json").read_text()
    )
    queue = {
        row["source_sample_id"]: row
        for row in (
            json.loads(line)
            for line in (inputs["preparation"] / "review_queue.jsonl")
            .read_text()
            .splitlines()
        )
    }
    assignments = {row["source_id"]: row for row in package.candidates}
    records = {}
    for sample_id, row in queue.items():
        recovery = {
            "schema_version": "table2-memory-recovery-verification-evidence-v1",
            "authority_type": "reviewer",
            "authority_id": "fixture-independent-recovery-reviewer",
            "authority_version": "fixture-v1",
            "independent_verification": True,
            "source_sample_id": sample_id,
            "recovery_sample_id": row["recovery_sample_id"],
            "canonical_task_id": row["canonical_task_id"],
            "episode_id": row["episode_id"],
            "pre_recovery_state_sha256": row["causal_material"][
                "pre_recovery_state_sha256"
            ],
            "executed_recovery_action_sha256": row["causal_material"][
                "executed_recovery_action_sha256"
            ],
            "post_recovery_state_sha256": row["causal_material"][
                "post_recovery_state_sha256"
            ],
            "verified_recovery_success": True,
        }
        recovery["evidence_sha256"] = canonical_sha256(recovery)
        final_task = {
            "schema_version": "table2-memory-final-task-verification-evidence-v1",
            "authority_type": "reviewer",
            "authority_id": "fixture-independent-terminal-reviewer",
            "authority_version": "fixture-v1",
            "independent_verification": True,
            "source_sample_id": sample_id,
            "canonical_task_id": row["canonical_task_id"],
            "episode_id": row["episode_id"],
            "task_specification_sha256": SHA_A,
            "terminal_state_sha256": SHA_B,
            "terminal_verifier_output_sha256": SHA_C,
            "verified_final_task_success": True,
        }
        final_task["evidence_sha256"] = canonical_sha256(final_task)
        assignment = assignments[sample_id]
        evidence = {
            "source_split": "train",
            "source_sample_id": sample_id,
            "provenance_valid": True,
            "final_task_success": True,
            "canonical_task_id": row["canonical_task_id"],
            "episode_id": row["episode_id"],
            "exact_duplicate_key": assignment["exact_duplicate_key"],
            "near_duplicate_cluster_id": assignment["near_duplicate_cluster_id"],
            "duplicate_cluster_namespace_id": package.namespace.namespace_id,
            "recovery_verification": recovery,
            "final_task_verification": final_task,
        }
        evidence["verification_evidence_sha256"] = verification_bundle_sha256(
            source_sample_id=sample_id,
            recovery_sample_id=row["recovery_sample_id"],
            canonical_task_id=row["canonical_task_id"],
            episode_id=row["episode_id"],
            recovery_evidence_sha256=recovery["evidence_sha256"],
            final_task_evidence_sha256=final_task["evidence_sha256"],
        )
        if row["source_review_status"] == "pending":
            label_review = {
                "schema_version": "table2-memory-p4-label-review-v1",
                "authority_type": "reviewer",
                "authority_id": "fixture-independent-p4-label-reviewer",
                "authority_version": "fixture-rubric-v1",
                "independent_verification": True,
                "source_sample_id": sample_id,
                "source_record_sha256": row["source_record_sha256"],
                "memory_item_source_material_sha256": canonical_sha256(
                    row["memory_item_source_material"]
                ),
                "approved_for_p4_memory": True,
            }
            label_review["evidence_sha256"] = canonical_sha256(label_review)
            evidence["p4_label_review"] = label_review
            evidence["p4_label_review_evidence_sha256"] = label_review[
                "evidence_sha256"
            ]
        records[sample_id] = evidence
    return {
        "schema_version": "table2-memory-provenance-v1",
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "dataset_id": preparation_manifest["dataset_id"],
        "dataset_version": preparation_manifest["dataset_version"],
        "dataset_artifacts_sha256": preparation_manifest[
            "dataset_artifacts_sha256"
        ],
        "records_sha256": preparation_manifest["records_sha256"],
        "duplicate_cluster_namespace": package.namespace.to_dict(),
        "joint_duplicate_assignment_binding": provenance_assignment_binding(package),
        "records": records,
    }


def _fixture_store_binding(
    *,
    inputs: dict[str, Path],
    package: object,
    provenance_path: Path,
    audit_path: Path,
) -> dict:
    return {
        "schema_version": "table2-memory-joint-duplicate-evidence-binding-v2",
        "preparation_manifest_sha256": sha256_file(
            inputs["preparation"] / "preparation_manifest.json"
        ),
        "assignment_manifest_sha256": package.assignment_manifest_sha256,
        "entities_sha256": package.manifest["entities_sha256"],
        "clusters_sha256": package.manifest["clusters_sha256"],
        "audit_config_sha256": package.namespace.audit_tool_config_sha256,
        "audit_source_sha256": package.namespace.audit_tool_source_sha256,
        "recovery_scenarios_sha256": sha256_file(inputs["recovery_scenarios"]),
        "duplicate_audit_registration_sha256": sha256_file(
            inputs["duplicate_registration"]
        ),
        "source_authority_sha256": sha256_file(inputs["source_authority"]),
        "final_duplicate_audit_sha256": sha256_file(audit_path),
        "provenance_manifest_sha256": sha256_file(provenance_path),
        "duplicate_cluster_namespace": package.namespace.to_dict(),
    }


def _fixture_embeddings(selection: EligibilitySelection) -> np.ndarray:
    """Give the compact fixture both positive and negative calibration pairs."""

    matrix = np.zeros((len(selection.candidates), 768), dtype=np.float32)
    retry_index = 0
    for index, candidate in enumerate(selection.candidates):
        if candidate.item["strategy"] == "RETRY":
            matrix[index, 0] = 1.0
            matrix[index, 2] = 0.05 * retry_index
            retry_index += 1
        else:
            matrix[index, 1] = 1.0
    return matrix


def _build_fixture_store(
    *,
    root: Path,
    inputs: dict[str, Path],
    package: object,
    provenance_path: Path,
    audit_path: Path,
    selection: EligibilitySelection | None = None,
    provenance_payload: dict | None = None,
) -> Path:
    if selection is None:
        selection = reconstruct_p4_selection_from_preparation(
            package_root=inputs["preparation"],
            provenance_manifest_path=provenance_path,
        )
    if provenance_payload is None:
        provenance_payload = json.loads(provenance_path.read_text())
    provenance = ProvenanceManifest.from_mapping(provenance_payload)
    embeddings = _fixture_embeddings(selection)
    preparation_manifest = json.loads(
        (inputs["preparation"] / "preparation_manifest.json").read_text()
    )
    transition_report = json.loads(
        (inputs["preparation"] / "candidate_audit.json").read_text()
    )["transition_report"]
    provenance_sha256 = sha256_file(provenance_path)
    calibration = build_calibration_evidence(
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=provenance_sha256,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=preparation_manifest["records_sha256"],
        dataset_artifacts_sha256=preparation_manifest[
            "dataset_artifacts_sha256"
        ],
        resolved_config_sha256=SHA_B,
        resolved_config_record_sha256=SHA_C,
        protocol_sha256=SHA_A,
    )
    store = build_frozen_store(
        root,
        selection=selection,
        embeddings=embeddings,
        model_seed=42,
        checkpoint_sha256=SHA_A,
        records_sha256=preparation_manifest["records_sha256"],
        dataset_id=preparation_manifest["dataset_id"],
        dataset_version=preparation_manifest["dataset_version"],
        dataset_artifacts_sha256=preparation_manifest[
            "dataset_artifacts_sha256"
        ],
        resolved_config_sha256=SHA_B,
        resolved_config_record_sha256=SHA_C,
        protocol_sha256=SHA_A,
        provenance_manifest_sha256=provenance_sha256,
        threshold_calibration=calibration["threshold_calibration"],
        calibration_evidence=calibration,
        transition_report=transition_report,
        joint_duplicate_audit_binding=_fixture_store_binding(
            inputs=inputs,
            package=package,
            provenance_path=provenance_path,
            audit_path=audit_path,
        ),
    )
    return store.root / "manifest.json"


def _fixture_campaign_repository(
    root: Path,
    *,
    source_authority: Path,
) -> Path:
    repository = root / "fixture-repository"
    copies = {
        "configs/eval/table2/joint_duplicate_audit_v1.json": CONFIG,
        "configs/eval/table2/p4_source_authority_v1.json": source_authority,
        "src/web_agent/memory/joint_duplicate_audit.py": (
            Path(__file__).resolve().parents[2]
            / "src/web_agent/memory/joint_duplicate_audit.py"
        ),
        "benchmarks/table2/pilot/duplicate_audit_manifest.json": (
            DUPLICATE_REGISTRATION
        ),
    }
    from web_agent.memory.joint_duplicate_audit import (
        AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS,
    )

    for relative in AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS:
        copies[relative] = Path(__file__).resolve().parents[2] / relative
    for relative, source in copies.items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return repository


def _bound_store_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pending_candidate_indices: frozenset[int] = frozenset(),
) -> dict[str, object]:
    inputs, package = _build(
        tmp_path,
        pending_candidate_indices=pending_candidate_indices,
    )
    provenance_path = _write_json(
        tmp_path / "provenance.json", _completed_provenance(inputs, package)
    )
    audit_path = finalize_joint_duplicate_audit(
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=provenance_path,
        output_path=tmp_path / "audit.json",
    )
    selection = reconstruct_p4_selection_from_preparation(
        package_root=inputs["preparation"],
        provenance_manifest_path=provenance_path,
    )
    manifest_path = _build_fixture_store(
        root=tmp_path / "valid-store",
        inputs=inputs,
        package=package,
        provenance_path=provenance_path,
        audit_path=audit_path,
        selection=selection,
    )
    repository = _fixture_campaign_repository(
        tmp_path,
        source_authority=inputs["source_authority"],
    )
    from web_agent.eval.table2 import package_validator as validator_module

    monkeypatch.setattr(
        validator_module,
        "P4_REGISTERED_SOURCE_AUTHORITY_SHA256",
        sha256_file(inputs["source_authority"]),
    )
    common = {
        "memory_by_seed": {42: manifest_path},
        "duplicate_audit_path": audit_path,
        "required": True,
        "repository_root": repository,
        "assignment_package_root": package.root,
        "preparation_package_root": inputs["preparation"],
        "resolved_task_export_path": inputs["export"],
        "approved_task_registry_path": inputs["registry"],
        "registered_recovery_scenarios_path": inputs["recovery_scenarios"],
        "duplicate_audit_registration_path": (
            repository
            / "benchmarks/table2/pilot/duplicate_audit_manifest.json"
        ),
        "provenance_manifest_path": provenance_path,
    }
    _validate_registered_joint_duplicate_memory_bindings(**common)
    return {
        "inputs": inputs,
        "package": package,
        "provenance_path": provenance_path,
        "audit_path": audit_path,
        "selection": selection,
        "manifest_path": manifest_path,
        "common": common,
    }


def _reseal_store_manifest(store_root: Path) -> Path:
    manifest_path = store_root / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    memory_ids = [
        json.loads(line)["memory_id"]
        for line in (store_root / "items.jsonl").read_text().splitlines()
    ]
    store_identity = canonical_sha256(
        {
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "records_sha256": manifest["records_sha256"],
            "dataset_artifacts_sha256": manifest["dataset_artifacts_sha256"],
            "resolved_config_sha256": manifest["resolved_config_sha256"],
            "resolved_config_record_sha256": manifest[
                "resolved_config_record_sha256"
            ],
            "protocol_sha256": manifest["protocol_sha256"],
            "provenance_manifest_sha256": manifest[
                "provenance_manifest_sha256"
            ],
            "threshold_calibration_sha256": manifest[
                "threshold_calibration_sha256"
            ],
            "calibration_evidence_sha256": manifest[
                "calibration_evidence_sha256"
            ],
            "verification_evidence_sha256": manifest[
                "verification_evidence_sha256"
            ],
            "verification_records_sha256": manifest[
                "verification_records_sha256"
            ],
            "model_seed": manifest["model_seed"],
            "admission_threshold": manifest["admission_threshold"],
            "memory_ids": memory_ids,
            "duplicate_cluster_namespace": manifest[
                "duplicate_cluster_namespace"
            ],
            "joint_duplicate_audit_binding": manifest[
                "joint_duplicate_audit_binding"
            ],
        }
    )
    manifest["store_id"] = (
        f"table2-memory-{manifest['model_seed']}-{store_identity[:16]}"
    )
    _write_json(manifest_path, manifest)
    sidecar = store_root / "manifest.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")
    return manifest_path


def test_finalize_refuses_provenance_without_exact_assignment_citation(tmp_path: Path):
    inputs, package = _build(tmp_path)
    provenance = _completed_provenance(inputs, package)
    provenance.pop("joint_duplicate_assignment_binding")
    provenance_path = _write_json(tmp_path / "provenance.json", provenance)
    with pytest.raises(JointDuplicateAuditError, match="does not cite the exact"):
        finalize_joint_duplicate_audit(
            assignment_package_root=package.root,
            config_path=CONFIG,
            source_authority_path=inputs["source_authority"],
            preparation_root=inputs["preparation"],
            gold_train_json=inputs["gold"],
            resolved_task_export_path=inputs["export"],
            approved_task_registry_path=inputs["registry"],
            recovery_scenarios_path=inputs["recovery_scenarios"],
            duplicate_audit_registration_path=inputs["duplicate_registration"],
            provenance_manifest_path=provenance_path,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_finalize_emits_runtime_consumable_50_plus_15_manifest(tmp_path: Path):
    inputs, package = _build(tmp_path)
    provenance_path = _write_json(
        tmp_path / "provenance.json", _completed_provenance(inputs, package)
    )
    audit_path = finalize_joint_duplicate_audit(
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=provenance_path,
        output_path=tmp_path / "joint-duplicate-audit.json",
    )
    frozen = FrozenDuplicateAuditManifest.from_path(audit_path)
    validated = validate_final_joint_duplicate_audit(
        audit_path=audit_path,
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=provenance_path,
    )
    assert len(frozen.task_ids) == 65
    assert len(validated.task_ids) == 65
    payload = json.loads(audit_path.read_text())
    registration = json.loads(inputs["duplicate_registration"].read_text())
    registered_diagnostics = [
        row
        for row in registration["entries"]
        if row["task_partition"] == "recovery_diagnostic"
    ]
    assert payload["entries"][50:] == registered_diagnostics
    assert payload["normal_task_evidence_status"] == "VERIFIED"
    assert payload["synthetic_recovery_evidence_scope"] == "DIAGNOSTIC_ONLY_NOT_PRIMARY"
    assert payload["independent_recovery_or_final_success_evidence_created"] == 0
    assert payload["train_corpus_binding"]["provenance_manifest_sha256"] == sha256_file(
        provenance_path
    )
    compact = validate_compact_joint_duplicate_evidence(
        **_compact_kwargs(inputs, package.root),
        provenance_manifest_path=provenance_path,
        final_audit_path=audit_path,
    )
    assert compact.assignment_manifest_sha256 == package.assignment_manifest_sha256


def test_authenticated_direct_retry_reaches_a_loadable_frozen_store(
    tmp_path: Path,
):
    inputs, package = _build(tmp_path, include_direct_supplement=True)
    provenance_path = _write_json(
        tmp_path / "provenance.json", _completed_provenance(inputs, package)
    )
    audit_path = finalize_joint_duplicate_audit(
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        supplement_train_json=inputs["supplement"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=provenance_path,
        output_path=tmp_path / "audit.json",
    )
    selection = reconstruct_p4_selection_from_preparation(
        package_root=inputs["preparation"],
        provenance_manifest_path=provenance_path,
    )
    direct = next(
        candidate
        for candidate in selection.candidates
        if candidate.source_sample_id == "direct-retry-0"
    )
    assert direct.item["source_transition_kind"] == (
        "direct_recovery_from_observed_failure_state"
    )
    assert direct.item["failure_type"] == "UNAVAILABLE"
    assert direct.item["failure_type_available"] is False
    assert direct.item["failed_action"] == "UNAVAILABLE"
    assert direct.item["failed_action_available"] is False
    assert direct.item["executed_recovery_action"] == "CLICK"
    manifest_path = _build_fixture_store(
        root=tmp_path / "direct-store",
        inputs=inputs,
        package=package,
        provenance_path=provenance_path,
        audit_path=audit_path,
        selection=selection,
    )
    FrozenMemoryStore.load(manifest_path.parent)


def test_compact_replay_rejects_candidate_material_and_cluster_rebinding(
    tmp_path: Path,
):
    inputs, package = _build(tmp_path)
    package.root.chmod(0o755)
    entities_path = package.root / "entities.jsonl"
    entities_path.chmod(0o644)
    rows = [json.loads(line) for line in entities_path.read_text().splitlines()]
    candidate = next(
        row for row in rows if row["entity_kind"] == "gold_train_candidate"
    )
    candidate["normalized_goal"] = "attacker rewritten candidate goal"
    candidate["goal_tokens"] = ["attacker", "rewritten", "candidate", "goal"]
    candidate["site_keys"] = ["attacker-site"]
    candidate["cluster_ids"] = ["exact:" + SHA_A, "near:" + SHA_B]
    candidate["near_duplicate_cluster_id"] = "near:" + SHA_B
    entities_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_path = package.root / "assignment_manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["entities.jsonl"] = {
        "sha256": sha256_file(entities_path),
        "bytes": entities_path.stat().st_size,
    }
    manifest["entities_sha256"] = canonical_sha256(rows)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    sidecar = package.root / "assignment_manifest.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")

    with pytest.raises(
        JointDuplicateAuditError,
        match="candidate normalization/features are not canonical|compact candidate",
    ):
        validate_compact_joint_duplicate_evidence(
            **_compact_kwargs(inputs, package.root)
        )


def test_compact_final_replay_rejects_rehashed_task_cluster_rebinding(
    tmp_path: Path,
):
    inputs, package = _build(tmp_path)
    provenance_path = _write_json(
        tmp_path / "provenance.json", _completed_provenance(inputs, package)
    )
    audit_path = finalize_joint_duplicate_audit(
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=provenance_path,
        output_path=tmp_path / "audit.json",
    )
    audit_path.chmod(0o644)
    payload = json.loads(audit_path.read_text())
    row = payload["entries"][0]
    row["cluster_ids"] = ["exact:" + SHA_A, "near:" + SHA_B]
    namespace = JointDuplicateClusterNamespace.from_mapping(
        payload["duplicate_cluster_namespace"], require_hashes=True
    )
    row["audit_record"] = canonical_task_audit_record(
        task_id=row["task_id"],
        content_sha256=row["content_sha256"],
        train_corpus_manifest_sha256=row["train_corpus_manifest_sha256"],
        cluster_ids=row["cluster_ids"],
        namespace=namespace,
    )
    row["evidence_sha256"] = canonical_sha256(row["audit_record"])
    _write_json(audit_path, payload)

    with pytest.raises(
        JointDuplicateAuditError,
        match="final duplicate audit differs from compact deterministic replay",
    ):
        validate_compact_joint_duplicate_evidence(
            **_compact_kwargs(inputs, package.root),
            provenance_manifest_path=provenance_path,
            final_audit_path=audit_path,
        )


def test_campaign_binding_rejects_frozen_store_item_cluster_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inputs, package = _build(tmp_path)
    provenance_path = _write_json(
        tmp_path / "provenance.json", _completed_provenance(inputs, package)
    )
    audit_path = finalize_joint_duplicate_audit(
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=provenance_path,
        output_path=tmp_path / "audit.json",
    )
    selection = reconstruct_p4_selection_from_preparation(
        package_root=inputs["preparation"],
        provenance_manifest_path=provenance_path,
    )
    manifest_path = _build_fixture_store(
        root=tmp_path / "valid-store",
        inputs=inputs,
        package=package,
        provenance_path=provenance_path,
        audit_path=audit_path,
        selection=selection,
    )
    binding = dict(
        json.loads(manifest_path.read_text())["joint_duplicate_audit_binding"]
    )
    repository = _fixture_campaign_repository(
        tmp_path, source_authority=inputs["source_authority"]
    )
    from web_agent.eval.table2 import package_validator as validator_module

    monkeypatch.setattr(
        validator_module,
        "P4_REGISTERED_SOURCE_AUTHORITY_SHA256",
        sha256_file(inputs["source_authority"]),
    )
    copied_registration = (
        repository / "benchmarks/table2/pilot/duplicate_audit_manifest.json"
    )
    common = {
        "memory_by_seed": {42: manifest_path},
        "duplicate_audit_path": audit_path,
        "required": True,
        "repository_root": repository,
        "assignment_package_root": package.root,
        "preparation_package_root": inputs["preparation"],
        "resolved_task_export_path": inputs["export"],
        "approved_task_registry_path": inputs["registry"],
        "registered_recovery_scenarios_path": inputs["recovery_scenarios"],
        "duplicate_audit_registration_path": copied_registration,
        "provenance_manifest_path": provenance_path,
    }
    assert _validate_registered_joint_duplicate_memory_bindings(**common) == binding

    # Rebuild a fully self-consistent store around a malicious cluster rewrite.
    # Its own hashes/calibration/load checks pass, but the immutable compact
    # preparation+assignment+provenance closure must still reject it.
    target = selection.candidates[0]
    malicious_cluster = "near:" + "9" * 64
    malicious_item = dict(target.item)
    malicious_item["duplicate_cluster_id"] = malicious_cluster
    malicious_candidates = [
        replace(
            candidate,
            near_duplicate_cluster_id=malicious_cluster,
            item=malicious_item,
        )
        if candidate.memory_id == target.memory_id
        else candidate
        for candidate in selection.candidates
    ]
    malicious_selection = EligibilitySelection(
        candidates=tuple(malicious_candidates),
        exclusion_counts=dict(selection.exclusion_counts),
        input_rows=selection.input_rows,
        pre_dedup_eligible_rows=selection.pre_dedup_eligible_rows,
        duplicate_cluster_namespace=dict(selection.duplicate_cluster_namespace),
    )
    malicious_provenance = deepcopy(json.loads(provenance_path.read_text()))
    malicious_provenance["records"][target.source_sample_id][
        "near_duplicate_cluster_id"
    ] = malicious_cluster
    malicious_manifest = _build_fixture_store(
        root=tmp_path / "malicious-store",
        inputs=inputs,
        package=package,
        provenance_path=provenance_path,
        audit_path=audit_path,
        selection=malicious_selection,
        provenance_payload=malicious_provenance,
    )
    FrozenMemoryStore.load(malicious_manifest.parent)
    common["memory_by_seed"] = {42: malicious_manifest}
    with pytest.raises(SchemaError, match="exact compact eligibility/dedup replay"):
        _validate_registered_joint_duplicate_memory_bindings(**common)


def test_campaign_binding_rejects_omitted_eligible_memory_in_rebuilt_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _bound_store_context(tmp_path, monkeypatch)
    selection = context["selection"]
    assert isinstance(selection, EligibilitySelection)
    counts: dict[str, int] = {}
    for candidate in selection.candidates:
        strategy = str(candidate.item["strategy"])
        counts[strategy] = counts.get(strategy, 0) + 1
    removable = next(
        candidate
        for candidate in selection.candidates
        if counts[str(candidate.item["strategy"])] >= 2
    )
    omitted_selection = replace(
        selection,
        candidates=tuple(
            candidate
            for candidate in selection.candidates
            if candidate.memory_id != removable.memory_id
        ),
    )
    inputs = context["inputs"]
    package = context["package"]
    assert isinstance(inputs, dict)
    malicious_manifest = _build_fixture_store(
        root=tmp_path / "omitted-store",
        inputs=inputs,
        package=package,
        provenance_path=context["provenance_path"],
        audit_path=context["audit_path"],
        selection=omitted_selection,
    )
    FrozenMemoryStore.load(malicious_manifest.parent)
    common = dict(context["common"])
    common["memory_by_seed"] = {42: malicious_manifest}
    with pytest.raises(SchemaError, match="exact compact eligibility/dedup replay"):
        _validate_registered_joint_duplicate_memory_bindings(**common)


def test_campaign_binding_rejects_strategy_and_reflection_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _bound_store_context(tmp_path, monkeypatch)
    selection = context["selection"]
    assert isinstance(selection, EligibilitySelection)
    target = next(
        candidate
        for candidate in selection.candidates
        if candidate.item["strategy"] == "RETRY"
    )
    malicious_item = dict(target.item)
    malicious_item["strategy"] = "REPLAN"
    malicious_item["reflection_text"] = "attacker-authored runtime advice"
    malicious_selection = replace(
        selection,
        candidates=tuple(
            replace(candidate, item=malicious_item)
            if candidate.memory_id == target.memory_id
            else candidate
            for candidate in selection.candidates
        ),
    )
    inputs = context["inputs"]
    package = context["package"]
    assert isinstance(inputs, dict)
    malicious_manifest = _build_fixture_store(
        root=tmp_path / "rewritten-store",
        inputs=inputs,
        package=package,
        provenance_path=context["provenance_path"],
        audit_path=context["audit_path"],
        selection=malicious_selection,
    )
    FrozenMemoryStore.load(malicious_manifest.parent)
    common = dict(context["common"])
    common["memory_by_seed"] = {42: malicious_manifest}
    with pytest.raises(SchemaError, match="exact compact eligibility/dedup replay"):
        _validate_registered_joint_duplicate_memory_bindings(**common)


def test_pending_label_review_is_frozen_and_replayed_after_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _bound_store_context(
        tmp_path,
        monkeypatch,
        pending_candidate_indices=frozenset({4}),
    )
    manifest_path = context["manifest_path"]
    assert isinstance(manifest_path, Path)
    FrozenMemoryStore.load(manifest_path.parent)
    frozen_items = [
        json.loads(line)
        for line in (manifest_path.parent / "items.jsonl").read_text().splitlines()
    ]
    pending_item = next(
        item
        for item in frozen_items
        if item["source_review_status"] == "pending"
    )
    verification_path = manifest_path.parent / "verification_evidence.json"
    verification_path.chmod(0o644)
    payload = json.loads(verification_path.read_text())
    row = next(
        item
        for item in payload["records"]
        if item["memory_id"] == pending_item["memory_id"]
    )
    assert isinstance(row["p4_label_review"], dict)
    row["p4_label_review"] = None
    row["p4_label_review_evidence_sha256"] = None
    row["record_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    payload["records_sha256"] = canonical_sha256(payload["records"])
    _write_json(verification_path, payload)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["verification_evidence.json"] = {
        "sha256": sha256_file(verification_path),
        "bytes": verification_path.stat().st_size,
    }
    manifest["verification_records_sha256"] = canonical_sha256(payload)
    manifest_path.chmod(0o644)
    _write_json(manifest_path, manifest)
    _reseal_store_manifest(manifest_path.parent)

    with pytest.raises(
        ManifestError,
        match="p4_label_review_evidence_sha256|lacks its full P4 label-review record",
    ):
        FrozenMemoryStore.load(manifest_path.parent)


def test_final_validation_rejects_assignment_tamper_in_provenance(tmp_path: Path):
    inputs, package = _build(tmp_path)
    provenance = _completed_provenance(inputs, package)
    provenance["records"]["candidate-0"]["near_duplicate_cluster_id"] = "near:tampered"
    provenance_path = _write_json(tmp_path / "provenance.json", provenance)
    with pytest.raises(JointDuplicateAuditError, match="assignment mismatch"):
        finalize_joint_duplicate_audit(
            assignment_package_root=package.root,
            config_path=CONFIG,
            source_authority_path=inputs["source_authority"],
            preparation_root=inputs["preparation"],
            gold_train_json=inputs["gold"],
            resolved_task_export_path=inputs["export"],
            approved_task_registry_path=inputs["registry"],
            recovery_scenarios_path=inputs["recovery_scenarios"],
            duplicate_audit_registration_path=inputs["duplicate_registration"],
            provenance_manifest_path=provenance_path,
            output_path=tmp_path / "must-not-exist.json",
        )


def test_production_prebuild_rejects_self_consistent_cluster_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A generic-valid namespace/corpus rewrite cannot bypass source replay."""

    inputs, package = _build(tmp_path)
    provenance = _completed_provenance(inputs, package)
    malicious_cluster = "near:" + "9" * 64
    provenance["records"]["candidate-0"][
        "near_duplicate_cluster_id"
    ] = malicious_cluster
    provenance_path = _write_json(tmp_path / "malicious-provenance.json", provenance)

    # Construct the kind of internally consistent evidence the older generic
    # gate accepted: same registered namespace, new provenance-file hash, and
    # canonical task records.  The registered producer replay must still reject
    # it because its candidate/task assignments differ from source bytes.
    valid_audit = finalize_joint_duplicate_audit(
        assignment_package_root=package.root,
        config_path=CONFIG,
        source_authority_path=inputs["source_authority"],
        preparation_root=inputs["preparation"],
        gold_train_json=inputs["gold"],
        resolved_task_export_path=inputs["export"],
        approved_task_registry_path=inputs["registry"],
        recovery_scenarios_path=inputs["recovery_scenarios"],
        duplicate_audit_registration_path=inputs["duplicate_registration"],
        provenance_manifest_path=_write_json(
            tmp_path / "valid-provenance.json", _completed_provenance(inputs, package)
        ),
        output_path=tmp_path / "valid-audit.json",
    )
    audit = json.loads(valid_audit.read_text())
    namespace = JointDuplicateClusterNamespace.from_mapping(
        audit["duplicate_cluster_namespace"], require_hashes=True
    )
    corpus = canonical_train_corpus_binding(
        records_sha256=provenance["records_sha256"],
        provenance_manifest_sha256=sha256_file(provenance_path),
        duplicate_cluster_namespace=namespace,
    )["corpus_binding_sha256"]
    for index, row in enumerate(audit["entries"][:50]):
        clusters = list(row["cluster_ids"])
        if index == 0:
            clusters = [
                malicious_cluster if value.startswith("near:") else value
                for value in clusters
            ]
        record = canonical_task_audit_record(
            task_id=row["task_id"],
            content_sha256=row["content_sha256"],
            train_corpus_manifest_sha256=corpus,
            cluster_ids=sorted(clusters),
            namespace=namespace,
        )
        row["cluster_ids"] = sorted(clusters)
        row["train_corpus_manifest_sha256"] = corpus
        row["audit_record"] = record
        row["evidence_sha256"] = canonical_sha256(record)
    audit["provenance_manifest_sha256"] = sha256_file(provenance_path)
    audit["train_corpus_binding"] = canonical_train_corpus_binding(
        records_sha256=provenance["records_sha256"],
        provenance_manifest_sha256=sha256_file(provenance_path),
        duplicate_cluster_namespace=namespace,
    )
    malicious_audit = _write_json(tmp_path / "malicious-audit.json", audit)

    def generic_gate_must_not_run(**_: object) -> None:
        raise AssertionError("registered replay did not fail before generic evidence gate")

    monkeypatch.setattr(
        build_pipeline, "validate_p4_build_prerequisites", generic_gate_must_not_run
    )
    monkeypatch.setattr(
        build_pipeline,
        "P4_REGISTERED_SOURCE_AUTHORITY_SHA256",
        sha256_file(inputs["source_authority"]),
    )
    with pytest.raises(JointDuplicateAuditError, match="assignment mismatch"):
        build_pipeline.validate_registered_p4_build_prerequisites(
            provenance_manifest_path=provenance_path,
            resolved_task_export_path=inputs["export"],
            webarena_task_source_path=tmp_path / "unused-source",
            webarena_task_registry_path=inputs["registry"],
            webarena_site_url_map_path=tmp_path / "unused-url-map",
            task_interface_audit_path=tmp_path / "unused-interface-audit",
            duplicate_audit_path=malicious_audit,
            joint_assignment_package_path=package.root,
                joint_audit_config_path=CONFIG,
                p4_source_authority_path=inputs["source_authority"],
                p4_preparation_package_path=inputs["preparation"],
            gold_train_json_path=inputs["gold"],
            recovery_scenarios_path=inputs["recovery_scenarios"],
            duplicate_audit_registration_path=inputs["duplicate_registration"],
        )
