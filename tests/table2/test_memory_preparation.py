from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from web_agent.memory.manifest import canonical_sha256, sha256_file
from web_agent.memory.preparation import (
    P4_REGISTERED_SOURCE_AUTHORITY_SHA256,
    P4PreparationError,
    create_memory_store_transfer_manifest,
    main as preparation_main,
    _prepare_p4_candidate_audit_fixture,
    prepare_p4_candidate_audit as prepare_registered_p4_candidate_audit,
    reconstruct_p4_selection_from_preparation,
    validate_memory_store_transfer_manifest,
    validate_p4_preparation_package,
    validate_p4_provenance_against_preparation,
)
from web_agent.memory.verification import verification_bundle_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _row(
    *,
    sample_id: str,
    task_id: str,
    trajectory_id: str,
    step_index: int,
    state_before: str,
    state_after: str,
    outcome: str,
    action: str,
    strategy: str,
    recovery_success: bool | None,
    memory_update: bool,
    split: str = "train",
) -> dict:
    return {
        "inputs": {
            "state_before": state_before,
            "state_after": state_after,
            "task_description": "complete the fixture task",
            "website_domain": "fixture.test",
        },
        "labels": {
            "outcome_label": outcome,
            "failure_type_4": (
                "ACTION_MISMATCH" if outcome == "FAILURE" else "NONE"
            ),
            "action_type": action,
            "action_value": "fixture value",
            "recovery_strategy": strategy,
            "recovery_success": recovery_success,
            "memory_update_flag": memory_update,
            "reflection_text": "fixture reflection",
        },
        "meta": {
            "sample_id": sample_id,
            "task_id": task_id,
            "trajectory_id": trajectory_id,
            "step_index": step_index,
            "review_status": "approved",
            "split": split,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_state(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def prepare_p4_candidate_audit(**kwargs):
    """Attach an explicit synthetic authority to direct unit fixtures."""

    sources = []
    for role, argument, expected_name in (
        ("original_gold", "gold_train_json", "split_train.json"),
        (
            "retry_abort_supplement_v2",
            "supplement_train_json",
            "supplement_train.json",
        ),
    ):
        raw_path = kwargs.get(argument)
        if raw_path is None:
            continue
        path = Path(raw_path)
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rows = []
        sources.append(
            {
                "file_name": expected_name,
                "records": len(rows) if isinstance(rows, list) else 0,
                "role": role,
                "sha256": sha256_file(path),
            }
        )
    authority = Path(kwargs["output_dir"]).parent / (
        "." + Path(kwargs["output_dir"]).name + "-source-authority.json"
    )
    _write_json(
        authority,
        {
            "authority_id": "table2-synthetic-unit-source-authority",
            "authority_version": "fixture-v1",
            "dataset_id": kwargs["dataset_id"],
            "dataset_version": kwargs["dataset_version"],
            "schema_version": "table2-p4-source-authority-v1",
            "sources": sources,
        },
    )
    return _prepare_p4_candidate_audit_fixture(
        **kwargs,
        source_authority_path=authority,
    )


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    gold_root = tmp_path / "gold"
    supplement_root = tmp_path / "supplement"
    for relative, content in (
        ("images/g0-before.png", b"gold before"),
        ("images/g0-after.png", b"gold failure"),
        ("images/g1-after.png", b"gold recovered"),
    ):
        _write_state(gold_root, relative, content)
    gold_rows = [
        _row(
            sample_id="gold-0",
            task_id="gold-task",
            trajectory_id="gold-episode",
            step_index=0,
            state_before="images/g0-before.png",
            state_after="images/g0-after.png",
            outcome="FAILURE",
            action="CLICK",
            strategy="RETRY",
            recovery_success=True,
            memory_update=True,
        ),
        _row(
            sample_id="gold-1",
            task_id="gold-task",
            trajectory_id="gold-episode",
            step_index=1,
            state_before="images/g0-after.png",
            state_after="images/g1-after.png",
            outcome="SUCCESS",
            action="TYPE",
            strategy="NONE",
            recovery_success=None,
            memory_update=False,
        ),
    ]
    gold_json = gold_root / "split_train.json"
    _write_json(gold_json, gold_rows)

    _write_state(supplement_root, "images/s-before.png", b"supplement failure")
    _write_state(supplement_root, "images/s-after.png", b"supplement abort")
    supplement = _row(
        sample_id="supplement-0",
        task_id="supplement-task",
        trajectory_id="supplement-episode",
        step_index=0,
        state_before="images/s-before.png",
        state_after="images/s-after.png",
        outcome="FAILURE",
        action="PRESS_KEY",
        strategy="ABORT",
        recovery_success=True,
        memory_update=False,
    )
    supplement["labels"]["recovery_attempted"] = True
    supplement_json = supplement_root / "data" / "supplement_train.json"
    _write_json(supplement_json, [supplement])
    return gold_root, gold_json, supplement_root, supplement_json


def test_candidate_audit_reads_only_train_and_never_claims_provenance(tmp_path: Path):
    gold_root, gold_json, supplement_root, supplement_json = _source_fixture(
        tmp_path
    )
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        supplement_train_json=supplement_json,
        supplement_data_root=supplement_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "review-package",
    )

    assert package.status == "REVIEW_REQUIRED"
    assert package.candidate_count == 1
    assert package.manifest["independent_verification_records_created"] == 0
    assert package.manifest["provenance_manifest_created"] is False
    assert package.manifest["eligible_memory_items_claimed"] == 0
    for field in (
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
    ):
        assert package.manifest[field] == 0

    audit = json.loads((package.root / "candidate_audit.json").read_text())
    assert audit["source_rows"] == {
        "original_gold": 2,
        "retry_abort_supplement_v2": 1,
    }
    assert audit["local_gate_counts"]["memory_update_flag_false"] == 2
    assert audit["local_gate_counts"]["requires_independent_review"] == 1

    queue = [
        json.loads(line)
        for line in (package.root / "review_queue.jsonl").read_text().splitlines()
    ]
    assert queue[0]["source_sample_id"] == "gold-0"
    assert queue[0]["recovery_sample_id"] == "gold-1"
    assert queue[0]["causal_material"]["executed_recovery_action"] == "TYPE"
    assert all(
        value is None
        for value in queue[0]["required_external_evidence"].values()
    )
    assert "recovery_verification" not in queue[0]
    assert "final_task_verification" not in queue[0]
    validate_p4_preparation_package(package.root)


def test_production_preparation_rejects_self_authored_fixture_authority(
    tmp_path: Path,
):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    forged_authority = tmp_path / "forged-source-authority.json"
    _write_json(
        forged_authority,
        {
            "authority_id": "attacker-self-authored",
            "authority_version": "fixture-v1",
            "dataset_id": "gold-v2.8",
            "dataset_version": "fixture-v1",
            "schema_version": "table2-p4-source-authority-v1",
            "sources": [
                {
                    "file_name": "split_train.json",
                    "records": 2,
                    "role": "original_gold",
                    "sha256": sha256_file(gold_json),
                }
            ],
        },
    )
    assert sha256_file(forged_authority) != P4_REGISTERED_SOURCE_AUTHORITY_SHA256
    with pytest.raises(P4PreparationError, match="registered PC-01 source authority"):
        prepare_registered_p4_candidate_audit(
            gold_train_json=gold_json,
            gold_data_root=gold_root,
            dataset_id="gold-v2.8",
            dataset_version="fixture-v1",
            source_authority_path=forged_authority,
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_fixture_authority_cannot_relabel_dataset_identity(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    authority = tmp_path / "source-authority.json"
    _write_json(
        authority,
        {
            "authority_id": "table2-synthetic-unit-source-authority",
            "authority_version": "fixture-v1",
            "dataset_id": "authorized-dataset",
            "dataset_version": "authorized-version",
            "schema_version": "table2-p4-source-authority-v1",
            "sources": [
                {
                    "file_name": "split_train.json",
                    "records": 2,
                    "role": "original_gold",
                    "sha256": sha256_file(gold_json),
                }
            ],
        },
    )
    with pytest.raises(P4PreparationError, match="dataset identity differs"):
        _prepare_p4_candidate_audit_fixture(
            gold_train_json=gold_json,
            gold_data_root=gold_root,
            dataset_id="attacker-relabeled-dataset",
            dataset_version="authorized-version",
            source_authority_path=authority,
            output_dir=tmp_path / "must-not-exist",
        )


def test_nontraining_declaration_is_rejected_without_a_package(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    rows = json.loads(gold_json.read_text())
    rows[0]["meta"]["split"] = "val"
    _write_json(gold_json, rows)

    destination = tmp_path / "failed-package"
    with pytest.raises(P4PreparationError, match="non-training split"):
        prepare_p4_candidate_audit(
            gold_train_json=gold_json,
            gold_data_root=gold_root,
            dataset_id="gold-v2.8",
            dataset_version="fixture-v1",
            output_dir=destination,
        )
    assert not destination.exists()


def test_audit_rejects_val_or_test_file_names_before_reading(tmp_path: Path):
    root = tmp_path / "gold"
    source = root / "split_test.json"
    _write_json(source, [])

    with pytest.raises(P4PreparationError, match="accepts only split_train.json"):
        prepare_p4_candidate_audit(
            gold_train_json=source,
            gold_data_root=root,
            dataset_id="gold-v2.8",
            dataset_version="fixture-v1",
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_audit_rejects_symlinked_training_json(tmp_path: Path):
    root = tmp_path / "gold"
    _write_json(root / "actual.json", [])
    source = root / "split_train.json"
    source.symlink_to(root / "actual.json")

    with pytest.raises(P4PreparationError, match="must not be a symlink"):
        prepare_p4_candidate_audit(
            gold_train_json=source,
            gold_data_root=root,
            dataset_id="gold-v2.8",
            dataset_version="fixture-v1",
            output_dir=tmp_path / "must-not-exist",
        )


def test_structural_candidate_failure_suppresses_the_complete_queue(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    (gold_root / "images/g1-after.png").unlink()

    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "failed-package",
    )

    assert package.status == "FAIL"
    assert package.candidate_count == 0
    audit = json.loads((package.root / "candidate_audit.json").read_text())
    assert audit["fatal_error_count"] == 1
    assert audit["eligible_memory_items_claimed"] == 0
    assert (package.root / "review_queue.jsonl").read_text() == ""
    validate_p4_preparation_package(package.root)


def test_preparation_package_rejects_fabricated_external_evidence(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "review-package",
    )
    queue_path = package.root / "review_queue.jsonl"
    row = json.loads(queue_path.read_text())
    row["required_external_evidence"]["final_task_success"] = True
    queue_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest_path = package.root / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["review_queue.jsonl"] = {
        "sha256": sha256_file(queue_path),
        "bytes": queue_path.stat().st_size,
    }
    manifest["review_queue_sha256"] = canonical_sha256([row])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (package.root / "preparation_manifest.sha256").write_text(
        sha256_file(manifest_path) + "\n", encoding="utf-8"
    )

    with pytest.raises(P4PreparationError, match="fabricates external evidence"):
        validate_p4_preparation_package(package.root)


def test_preparation_package_rejects_nonzero_test_read_after_rehash(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "review-package",
    )
    ledger_path = package.root / "read_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["test_rows_read"] = 1
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    manifest_path = package.root / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["read_ledger.json"] = {
        "sha256": sha256_file(ledger_path),
        "bytes": ledger_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (package.root / "preparation_manifest.sha256").write_text(
        sha256_file(manifest_path) + "\n", encoding="utf-8"
    )

    with pytest.raises(P4PreparationError, match="test_rows_read=0"):
        validate_p4_preparation_package(package.root)


def _fixture_provenance(package_root: Path) -> dict:
    package = json.loads((package_root / "preparation_manifest.json").read_text())
    queue = json.loads((package_root / "review_queue.jsonl").read_text())
    recovery = {
        "schema_version": "table2-memory-recovery-verification-evidence-v1",
        "authority_type": "reviewer",
        "authority_id": "fixture-reviewer-a",
        "authority_version": "fixture-v1",
        "independent_verification": True,
        "source_sample_id": queue["source_sample_id"],
        "recovery_sample_id": queue["recovery_sample_id"],
        "canonical_task_id": queue["canonical_task_id"],
        "episode_id": queue["episode_id"],
        "pre_recovery_state_sha256": queue["causal_material"][
            "pre_recovery_state_sha256"
        ],
        "executed_recovery_action_sha256": queue["causal_material"][
            "executed_recovery_action_sha256"
        ],
        "post_recovery_state_sha256": queue["causal_material"][
            "post_recovery_state_sha256"
        ],
        "verified_recovery_success": True,
    }
    recovery["evidence_sha256"] = canonical_sha256(recovery)
    final_task = {
        "schema_version": "table2-memory-final-task-verification-evidence-v1",
        "authority_type": "reviewer",
        "authority_id": "fixture-reviewer-b",
        "authority_version": "fixture-v1",
        "independent_verification": True,
        "source_sample_id": queue["source_sample_id"],
        "canonical_task_id": queue["canonical_task_id"],
        "episode_id": queue["episode_id"],
        "task_specification_sha256": SHA_A,
        "terminal_state_sha256": SHA_B,
        "terminal_verifier_output_sha256": SHA_C,
        "verified_final_task_success": True,
    }
    final_task["evidence_sha256"] = canonical_sha256(final_task)
    evidence = {
        "source_split": "train",
        "source_sample_id": queue["source_sample_id"],
        "provenance_valid": True,
        "final_task_success": True,
        "canonical_task_id": queue["canonical_task_id"],
        "episode_id": queue["episode_id"],
        "exact_duplicate_key": SHA_D,
        "near_duplicate_cluster_id": "fixture-near-cluster-1",
        "duplicate_cluster_namespace_id": "fixture-joint-namespace-v1",
        "recovery_verification": recovery,
        "final_task_verification": final_task,
    }
    evidence["verification_evidence_sha256"] = verification_bundle_sha256(
        source_sample_id=queue["source_sample_id"],
        recovery_sample_id=queue["recovery_sample_id"],
        canonical_task_id=queue["canonical_task_id"],
        episode_id=queue["episode_id"],
        recovery_evidence_sha256=recovery["evidence_sha256"],
        final_task_evidence_sha256=final_task["evidence_sha256"],
    )
    return {
        "schema_version": "table2-memory-provenance-v1",
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "dataset_id": package["dataset_id"],
        "dataset_version": package["dataset_version"],
        "dataset_artifacts_sha256": package["dataset_artifacts_sha256"],
        "records_sha256": package["records_sha256"],
        "duplicate_cluster_namespace": {
            "schema_version": "table2-joint-duplicate-cluster-namespace-v1",
            "namespace_id": "fixture-joint-namespace-v1",
            "audit_tool_id": "fixture-joint-auditor",
            "audit_tool_version": "fixture-v1",
            "audit_tool_config_sha256": SHA_E,
            "audit_tool_source_sha256": SHA_F,
        },
        "records": {queue["source_sample_id"]: evidence},
    }


def _attach_p4_label_review(provenance: dict, package_root: Path) -> None:
    queue = json.loads((package_root / "review_queue.jsonl").read_text())
    evidence = provenance["records"][queue["source_sample_id"]]
    review = {
        "schema_version": "table2-memory-p4-label-review-v1",
        "authority_type": "reviewer",
        "authority_id": "fixture-independent-p4-label-reviewer",
        "authority_version": "fixture-rubric-v1",
        "independent_verification": True,
        "source_sample_id": queue["source_sample_id"],
        "source_record_sha256": queue["source_record_sha256"],
        "memory_item_source_material_sha256": canonical_sha256(
            queue["memory_item_source_material"]
        ),
        "approved_for_p4_memory": True,
    }
    review["evidence_sha256"] = canonical_sha256(review)
    evidence["p4_label_review"] = review
    evidence["p4_label_review_evidence_sha256"] = review["evidence_sha256"]


def test_pending_source_requires_separate_label_review_before_admission(
    tmp_path: Path,
):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    rows = json.loads(gold_json.read_text())
    rows[0]["meta"]["review_status"] = "pending"
    _write_json(gold_json, rows)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "pending-package",
    )
    queue = json.loads((package.root / "review_queue.jsonl").read_text())
    assert queue["source_review_status"] == "pending"

    provenance = _fixture_provenance(package.root)
    provenance_path = tmp_path / "pending-without-review.json"
    _write_json(provenance_path, provenance)
    with pytest.raises(P4PreparationError, match="lacks independent label review"):
        reconstruct_p4_selection_from_preparation(
            package_root=package.root,
            provenance_manifest_path=provenance_path,
        )

    _attach_p4_label_review(provenance, package.root)
    reviewed_path = tmp_path / "pending-with-review.json"
    _write_json(reviewed_path, provenance)
    selection = reconstruct_p4_selection_from_preparation(
        package_root=package.root,
        provenance_manifest_path=reviewed_path,
    )
    assert len(selection.candidates) == 1
    assert selection.candidates[0].item["source_review_status"] == "pending"
    assert selection.candidates[0].item[
        "p4_label_review_evidence_sha256"
    ] == provenance["records"]["gold-0"][
        "p4_label_review_evidence_sha256"
    ]


def test_pending_externally_rejected_is_counted_without_label_review(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    rows = json.loads(gold_json.read_text())
    rows[0]["meta"]["review_status"] = "pending"
    _write_json(gold_json, rows)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "pending-package",
    )
    provenance = _fixture_provenance(package.root)
    provenance["records"]["gold-0"]["provenance_valid"] = False
    provenance_path = tmp_path / "pending-rejected.json"
    _write_json(provenance_path, provenance)
    report = validate_p4_provenance_against_preparation(
        package_root=package.root,
        provenance_manifest_path=provenance_path,
    )
    assert report["admitted_records_with_validated_evidence"] == 0
    assert report["externally_excluded_records"] == 1


def test_explicitly_rejected_source_never_enters_review_queue(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    rows = json.loads(gold_json.read_text())
    rows[0]["meta"]["review_status"] = "rejected"
    _write_json(gold_json, rows)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "rejected-package",
    )
    assert package.candidate_count == 0
    assert (package.root / "review_queue.jsonl").read_text() == ""
    audit = json.loads((package.root / "candidate_audit.json").read_text())
    assert audit["local_gate_counts"]["quarantined_or_non_admitted"] == 1


def test_authenticated_direct_retry_preserves_unavailable_prior_failure_fields(
    tmp_path: Path,
):
    gold_root, gold_json, supplement_root, supplement_json = _source_fixture(tmp_path)
    gold = json.loads(gold_json.read_text())
    gold[0]["labels"]["memory_update_flag"] = False
    _write_json(gold_json, gold)
    supplement = json.loads(supplement_json.read_text())
    supplement[0]["labels"].update(
        {
            "outcome_label": "SUCCESS",
            "failure_type_4": "NONE",
            "action_type": "CLICK",
            "recovery_strategy": "RETRY",
            "recovery_success": True,
            "memory_update_flag": True,
        }
    )
    _write_json(supplement_json, supplement)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        supplement_train_json=supplement_json,
        supplement_data_root=supplement_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "direct-package",
    )
    queue = json.loads((package.root / "review_queue.jsonl").read_text())
    assert queue["source_dataset_role"] == "retry_abort_supplement_v2"
    assert queue["source_transition_kind"] == (
        "direct_recovery_from_observed_failure_state"
    )
    material = queue["memory_item_source_material"]
    assert material["failed_action"] == "UNAVAILABLE"
    assert material["failed_action_available"] is False
    assert material["failure_type"] == "UNAVAILABLE"
    assert material["failure_type_available"] is False
    assert material["executed_recovery_action"] == "CLICK"

    provenance_path = tmp_path / "direct-provenance.json"
    _write_json(provenance_path, _fixture_provenance(package.root))
    selection = reconstruct_p4_selection_from_preparation(
        package_root=package.root,
        provenance_manifest_path=provenance_path,
    )
    item = selection.candidates[0].item
    assert item["failed_action"] == "UNAVAILABLE"
    assert item["executed_recovery_action"] == "CLICK"


def test_direct_abort_cannot_be_reinterpreted_as_final_task_success(tmp_path: Path):
    gold_root, gold_json, supplement_root, supplement_json = _source_fixture(tmp_path)
    gold = json.loads(gold_json.read_text())
    gold[0]["labels"]["memory_update_flag"] = False
    _write_json(gold_json, gold)
    supplement = json.loads(supplement_json.read_text())
    supplement[0]["labels"]["memory_update_flag"] = True
    _write_json(supplement_json, supplement)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        supplement_train_json=supplement_json,
        supplement_data_root=supplement_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "abort-package",
    )
    provenance_path = tmp_path / "abort-provenance.json"
    _write_json(provenance_path, _fixture_provenance(package.root))
    with pytest.raises(P4PreparationError, match="final success for direct ABORT"):
        reconstruct_p4_selection_from_preparation(
            package_root=package.root,
            provenance_manifest_path=provenance_path,
        )


def test_original_gold_cannot_spoof_direct_transition_marker(tmp_path: Path):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    rows = json.loads(gold_json.read_text())
    rows[0]["meta"]["_direct_recovery_transition"] = True
    _write_json(gold_json, rows)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "spoof-package",
    )
    queue = json.loads((package.root / "review_queue.jsonl").read_text())
    assert queue["source_dataset_role"] == "original_gold"
    assert queue["source_transition_kind"] == "adjacent_failure_then_recovery"


def test_external_provenance_validator_binds_exact_preparation_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "review-package",
    )
    provenance_path = tmp_path / "provenance.json"
    _write_json(provenance_path, _fixture_provenance(package.root))

    result = validate_p4_provenance_against_preparation(
        package_root=package.root,
        provenance_manifest_path=provenance_path,
    )

    assert result["status"] == "PASS"
    assert result["evidence_role"] == (
        "STRUCTURAL_VALIDATION_ONLY_NOT_REVIEW_AUTHORITY"
    )
    assert result["independent_review_performed_by_tool"] is False
    assert result["admitted_records_with_validated_evidence"] == 1
    assert result["externally_excluded_records"] == 0
    assert preparation_main(
        [
            "validate-provenance",
            "--package-dir",
            str(package.root),
            "--provenance-manifest",
            str(provenance_path),
        ]
    ) == 0
    cli_output = json.loads(capsys.readouterr().out)
    assert cli_output["independent_review_performed_by_tool"] is False


def test_external_provenance_validator_rejects_resealed_causal_mismatch(
    tmp_path: Path,
):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "review-package",
    )
    provenance = _fixture_provenance(package.root)
    record = next(iter(provenance["records"].values()))
    recovery = record["recovery_verification"]
    recovery["pre_recovery_state_sha256"] = SHA_F
    recovery["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in recovery.items() if key != "evidence_sha256"}
    )
    final_task = record["final_task_verification"]
    record["verification_evidence_sha256"] = verification_bundle_sha256(
        source_sample_id=record["source_sample_id"],
        recovery_sample_id=recovery["recovery_sample_id"],
        canonical_task_id=record["canonical_task_id"],
        episode_id=record["episode_id"],
        recovery_evidence_sha256=recovery["evidence_sha256"],
        final_task_evidence_sha256=final_task["evidence_sha256"],
    )
    provenance_path = tmp_path / "provenance.json"
    _write_json(provenance_path, provenance)

    with pytest.raises(P4PreparationError, match="pre_recovery_state_sha256"):
        validate_p4_provenance_against_preparation(
            package_root=package.root,
            provenance_manifest_path=provenance_path,
        )


def test_external_provenance_validator_requires_complete_candidate_coverage(
    tmp_path: Path,
):
    gold_root, gold_json, _, _ = _source_fixture(tmp_path)
    package = prepare_p4_candidate_audit(
        gold_train_json=gold_json,
        gold_data_root=gold_root,
        dataset_id="gold-v2.8",
        dataset_version="fixture-v1",
        output_dir=tmp_path / "review-package",
    )
    provenance = _fixture_provenance(package.root)
    provenance["records"] = {}
    provenance_path = tmp_path / "provenance.json"
    _write_json(provenance_path, provenance)

    with pytest.raises(P4PreparationError, match="candidate coverage is incomplete"):
        validate_p4_provenance_against_preparation(
            package_root=package.root,
            provenance_manifest_path=provenance_path,
        )


class _FakeFrozenStore:
    def __init__(self) -> None:
        self.manifest = {
            "store_id": "table2-memory-42-fixture",
            "model_seed": 42,
            "checkpoint_sha256": SHA_A,
            "records_sha256": SHA_B,
            "dataset_artifacts_sha256": SHA_C,
            "resolved_config_sha256": SHA_D,
            "resolved_config_record_sha256": SHA_E,
            "protocol_sha256": SHA_F,
            "provenance_manifest_sha256": SHA_A,
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "runtime_writes_allowed": False,
            "item_count": 3,
        }
        self.manifest_sha256 = SHA_B
        self.model_seed = 42


def _fake_store_files(root: Path) -> None:
    root.mkdir(parents=True)
    for index, name in enumerate(
        (
            "manifest.json",
            "manifest.sha256",
            "embeddings.npy",
            "items.jsonl",
            "verification_evidence.json",
            "calibration_evidence.json",
            "threshold_calibration.json",
        )
    ):
        (root / name).write_bytes(f"fixture-{index}-{name}".encode())


def test_transfer_manifest_is_portable_complete_and_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source-store"
    destination = tmp_path / "destination-store"
    transfer = tmp_path / "transfer.json"
    _fake_store_files(source)
    fake = _FakeFrozenStore()
    monkeypatch.setattr(
        "web_agent.memory.preparation.FrozenMemoryStore.load",
        lambda _root: fake,
    )

    payload = create_memory_store_transfer_manifest(
        store_root=source,
        manifest_path=transfer,
    )
    assert payload["source_split"] == "train"
    assert payload["runtime_writes_allowed"] is False
    assert set(payload["files"]) == {
        "manifest.json",
        "manifest.sha256",
        "embeddings.npy",
        "items.jsonl",
        "verification_evidence.json",
        "calibration_evidence.json",
        "threshold_calibration.json",
    }
    assert all("/" not in name for name in payload["files"])

    shutil.copytree(source, destination)
    validate_memory_store_transfer_manifest(
        store_root=destination,
        manifest_path=transfer,
    )
    (destination / "items.jsonl").write_bytes(b"tampered")
    with pytest.raises(P4PreparationError, match="descriptors do not match"):
        validate_memory_store_transfer_manifest(
            store_root=destination,
            manifest_path=transfer,
        )


def test_transfer_manifest_rejects_nonzero_read_claim_and_store_internal_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = tmp_path / "store"
    transfer = tmp_path / "transfer.json"
    _fake_store_files(store)
    fake = _FakeFrozenStore()
    monkeypatch.setattr(
        "web_agent.memory.preparation.FrozenMemoryStore.load",
        lambda _root: fake,
    )
    create_memory_store_transfer_manifest(
        store_root=store,
        manifest_path=transfer,
    )

    changed = json.loads(transfer.read_text())
    changed["test_rows_read"] = 1
    transfer.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(P4PreparationError, match="test_rows_read=0"):
        validate_memory_store_transfer_manifest(
            store_root=store,
            manifest_path=transfer,
        )

    with pytest.raises(P4PreparationError, match="outside the immutable store"):
        create_memory_store_transfer_manifest(
            store_root=store,
            manifest_path=store / "transfer.json",
        )
