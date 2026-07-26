from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from web_agent.data.review_overlay import ReviewOverlay
from web_agent.data.review_session import (
    append_review_event,
    build_review_event,
)


def _record(
    sample_id: str,
    *,
    split: str,
    action: str = "CLICK",
    failure: str = "NONE",
    bbox=None,
) -> dict:
    task_id = f"task-{sample_id}"
    return {
        "inputs": {
            "state_before": f"images/{sample_id}-before.png",
            "state_after": f"images/{sample_id}-after.png",
            "task_description": "Complete the recorded task.",
            "website_domain": "example.test",
        },
        "labels": {
            "outcome_label": "SUCCESS" if failure == "NONE" else "FAILURE",
            "failure_type_4": failure,
            "agent_confidence_before": 0.5,
            "action_type": action,
            "action_coordinates": None,
            "action_target_bbox": bbox,
            "recovery_attempted": False,
            "recovery_strategy": "NONE",
            "recovery_success": None,
            "memory_update_flag": False,
        },
        "meta": {
            "sample_id": sample_id,
            "task_id": task_id,
            "step_index": 0,
            "split": split,
            "review_status": "pending",
        },
    }


def _bbox_target(record: dict) -> dict:
    labels = record["labels"]
    meta = record["meta"]
    return {
        "split": meta["split"],
        "sample_id": meta["sample_id"],
        "task_id": meta["task_id"],
        "step_index": meta["step_index"],
        "reviewer_assignment": "A+B",
        "double_review": True,
        "image_width": 100,
        "image_height": 80,
        "original_bbox": json.dumps(labels["action_target_bbox"]),
        "action_type": labels["action_type"],
    }


def _weak_target(record: dict) -> dict:
    labels = record["labels"]
    meta = record["meta"]
    return {
        "split": meta["split"],
        "sample_id": meta["sample_id"],
        "task_id": meta["task_id"],
        "step_index": meta["step_index"],
        "reviewer_assignment": "A+B",
        "double_review": True,
        "is_focus_row": True,
        "row_focus_targets": f"ACTION:{labels['action_type']}",
        "action_target_bbox": json.dumps(labels["action_target_bbox"]),
        "action_type": labels["action_type"],
        "outcome_label": labels["outcome_label"],
        "failure_type_4": labels["failure_type_4"],
        "recovery_attempted": labels["recovery_attempted"],
        "recovery_strategy": labels["recovery_strategy"],
        "recovery_success": labels["recovery_success"],
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else [
        "split",
        "sample_id",
        "task_id",
        "reviewer_assignment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_passed_reconciliation(tmp_path: Path) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bbox_record = _record(
        "bbox",
        split="train",
        bbox={"x": 10, "y": 90, "width": 20, "height": 10},
    )
    rejected_record = _record(
        "reject",
        split="train",
        action="SCROLL",
        failure="LOOP_DETECTED",
    )
    action_record = _record("action", split="val", action="NAVIGATE")
    bbox_target = _bbox_target(bbox_record)
    weak_targets = [
        _weak_target(rejected_record),
        _weak_target(action_record),
    ]

    bbox_queue = tmp_path / "bbox_review_queue.csv"
    weak_queue = tmp_path / "weak_class_review_queue.csv"
    _write_rows(bbox_queue, [bbox_target])
    _write_rows(weak_queue, weak_targets)

    logs = {}
    for reviewer in ("A", "B"):
        log_path = tmp_path / f"reviewer_{reviewer}.csv"
        append_review_event(
            log_path,
            build_review_event(
                bbox_target,
                queue_kind="bbox",
                reviewer_id=reviewer,
                review_decision="approve_after_correction",
                reason_code="BBOX_WRONG",
                evidence_reference=f"replay:bbox:{reviewer}",
                bbox_ok="no",
                proposed_bbox=(10, 20, 20, 10),
            ),
        )
        append_review_event(
            log_path,
            build_review_event(
                weak_targets[0],
                queue_kind="weak",
                reviewer_id=reviewer,
                review_decision="reject_recollect",
                reason_code="IMG_CAPTURE_INVALID",
            ),
        )
        append_review_event(
            log_path,
            build_review_event(
                weak_targets[1],
                queue_kind="weak",
                reviewer_id=reviewer,
                review_decision="approve_after_correction",
                reason_code="ACTION_LABEL_WRONG",
                evidence_reference=f"replay:action:{reviewer}",
                action_label_ok="no",
                proposed_action_type="SCROLL",
            ),
        )
        logs[reviewer] = log_path

    output = tmp_path / "reconciliation"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/reconcile_gold_reviews.py",
            "--bbox-queue",
            str(bbox_queue),
            "--weak-queue",
            str(weak_queue),
            "--reviewer-log",
            str(logs["A"]),
            "--reviewer-log",
            str(logs["B"]),
            "--output-dir",
            str(output),
            "--require-pass",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return output, {
        "train": [bbox_record, rejected_record],
        "val": [action_record],
        "test": [_record("test", split="test")],
    }


def _write_dataset(root: Path, splits: dict[str, list[dict]]) -> None:
    root.mkdir()
    image_dir = root / "images"
    image_dir.mkdir()
    for split, records in splits.items():
        (root / f"split_{split}.json").write_text(
            json.dumps(records),
            encoding="utf-8",
        )
        for record in records:
            for field in ("state_before", "state_after"):
                image_path = root / record["inputs"][field]
                Image.new("RGB", (100, 80), "white").save(image_path)


def test_review_overlay_applies_corrections_and_exclusions_without_mutation(
    tmp_path,
):
    reconciliation, splits = _build_passed_reconciliation(tmp_path)
    original = json.loads(json.dumps(splits["train"]))
    overlay = ReviewOverlay.load(reconciliation)

    train, train_report = overlay.apply(splits["train"], split="train")
    val, val_report = overlay.apply(splits["val"], split="val")

    assert splits["train"] == original
    assert [row["meta"]["sample_id"] for row in train] == ["bbox"]
    assert train[0]["labels"]["action_target_bbox"] == {
        "x": 10.0,
        "y": 20.0,
        "width": 20.0,
        "height": 10.0,
    }
    assert train[0]["meta"]["review_status"] == "pending"
    assert train[0]["meta"]["targeted_review_overlay"]["status"] == "corrected"
    assert val[0]["labels"]["action_type"] == "SCROLL"
    assert train_report["excluded_rows"] == 1
    assert train_report["corrected_samples"] == 1
    assert val_report["corrected_samples"] == 1
    assert train_report["test_rows_read"] == 0
    provenance = overlay.provenance()
    assert provenance["enabled"] is True
    assert provenance["publication_ready"] is False
    assert len(provenance["reconciliation_report_sha256"]) == 64

    with pytest.raises(ValueError, match="only to train or val"):
        overlay.apply(splits["test"], split="test")


def test_review_overlay_refuses_tampering_and_wrong_source_version(tmp_path):
    reconciliation, splits = _build_passed_reconciliation(tmp_path)
    corrections = reconciliation / "approved_corrections.csv"
    corrections.write_text(
        corrections.read_text(encoding="utf-8-sig") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        ReviewOverlay.load(reconciliation)

    reconciliation, splits = _build_passed_reconciliation(tmp_path / "second")
    splits["train"][0]["labels"]["action_target_bbox"]["y"] = 70
    with pytest.raises(ValueError, match="original value"):
        ReviewOverlay.load(reconciliation).apply(
            splits["train"],
            split="train",
        )


def test_overlay_validator_uses_passed_overlay_without_test_rows(tmp_path):
    reconciliation, splits = _build_passed_reconciliation(tmp_path)
    data_root = tmp_path / "dataset"
    _write_dataset(data_root, splits)
    report_path = tmp_path / "overlay_report.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_gold_review_overlay.py",
            "--data-root",
            str(data_root),
            "--reconciliation-dir",
            str(reconciliation),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["controlled_mini_permitted"] is True
    assert report["publication_ready"] is False
    assert report["test_rows_read"] == 0
    assert report["splits"]["train"]["excluded_rows"] == 1
    assert report["splits"]["val"]["target_counts_after"]["ACTION:SCROLL"] == 1


def test_gold_loader_opt_in_overlay_is_train_val_only_and_compiles():
    path = Path("src/web_agent/data/gold_dataloader.py")
    source = path.read_text(encoding="utf-8")
    assert 'which in {"train", "val"}' in source
    assert 'cfg["data"].get("review_overlay_dir")' in source
    assert "ReviewOverlay.load(overlay_root).apply" in source
    compile(source, str(path), "exec")


def test_training_reports_overlay_and_requires_fair_baseline_comparison():
    stages_path = Path("src/web_agent/train/gold_stages.py")
    stages = stages_path.read_text(encoding="utf-8")
    assert "comparison_requires_baseline_reevaluation_on_overlay" in stages
    assert (
        "v2.7 selected checkpoint re-evaluated on the same review overlay"
        in stages
    )
    assert '"review_overlay": (' in stages
    compile(stages, str(stages_path), "exec")

    runner_path = Path("scripts/run_gold.py")
    runner = runner_path.read_text(encoding="utf-8")
    assert "--review-overlay-dir" in runner
    assert 'if args.stage == "eval"' in runner
    assert "development-only" in runner
    compile(runner, str(runner_path), "exec")


def test_overlay_validation_notebook_is_isolated_output_free_and_compiles():
    path = Path("notebooks/kaggle_gold_review_overlay_validation.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "validate_gold_review_overlay.py" in source
    assert "review_reconciliation_report.json" in source
    assert "test_rows_read" in source
    assert "split_test.json" not in source
    assert "kaggle_gold.ipynb" not in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(
                "".join(cell.get("source", [])),
                f"{path}:cell-{index}",
                "exec",
            )
