from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from web_agent.memory.manifest import canonical_sha256, sha256_file
from web_agent.memory.preparation import (
    P4PreparationError,
    create_memory_store_transfer_manifest,
    prepare_p4_candidate_audit,
    validate_memory_store_transfer_manifest,
    validate_p4_preparation_package,
)


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
