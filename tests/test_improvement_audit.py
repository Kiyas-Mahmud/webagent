from __future__ import annotations

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from PIL import Image

from web_agent.data.improvement_audit import (
    bbox_geometry_reasons,
    build_bbox_review_queue,
    build_weak_class_review_queue,
    class_coverage,
    reviewer_assignment,
)


def _row(
    task: str,
    step: int,
    *,
    action: str = "CLICK",
    failure: str = "NONE",
    strategy: str = "NONE",
    recovery_success=None,
    bbox=None,
) -> dict:
    return {
        "inputs": {
            "state_before": f"images/{task}-{step}-before.png",
            "state_after": f"images/{task}-{step}-after.png",
            "task_description": "complete the task",
            "website_domain": "example.test",
        },
        "labels": {
            "action_type": action,
            "action_coordinates": [50, 50],
            "action_target_bbox": bbox,
            "outcome_label": "NONE" if failure == "NONE" else "FAILURE",
            "failure_type_4": failure,
            "recovery_strategy": strategy,
            "recovery_success": recovery_success,
            "action_value": "target",
        },
        "meta": {
            "sample_id": f"{task}-{step}",
            "task_id": task,
            "step_index": step,
            "review_status": "pending",
        },
    }


def test_bbox_geometry_reasons_rejects_overflow_without_clipping():
    box = {"x": 90, "y": 20, "width": 20, "height": 10}
    assert bbox_geometry_reasons(box, (100, 100)) == [
        "right_boundary_overflow"
    ]
    assert box == {"x": 90, "y": 20, "width": 20, "height": 10}


def test_bbox_review_queue_masks_only_invalid_rows_and_keeps_blank_proposal():
    records = [
        _row(
            "t1",
            0,
            bbox={"x": 10, "y": 10, "width": 20, "height": 20},
        ),
        _row(
            "t2",
            0,
            action="SCROLL",
            bbox={"x": 10, "y": 95, "width": 20, "height": 20},
        ),
    ]

    rows, report = build_bbox_review_queue(
        records,
        "train",
        lambda _: ((100, 100), ""),
    )

    assert report["bbox_rows"] == 2
    assert report["valid_bbox_rows"] == 1
    assert report["invalid_bbox_rows"] == 1
    assert report["source_records_mutated"] is False
    assert rows[0]["sample_id"] == "t2-0"
    assert rows[0]["current_training_bbox_mask"] == 0
    assert rows[0]["automatic_correction_allowed"] is False
    assert rows[0]["proposed_bbox_x"] == ""
    assert rows[0]["recommended_disposition"] == "MASK_PENDING_DIRECT_EVIDENCE"
    assert records[1]["labels"]["action_target_bbox"]["y"] == 95


def test_weak_class_queue_preserves_complete_selected_trajectory():
    records = [
        _row("t1", 0, action="SCROLL"),
        _row("t1", 1, action="CLICK"),
        _row("t2", 0, action="TYPE"),
    ]

    rows, report = build_weak_class_review_queue(
        records,
        "train",
        max_tasks_per_class=1,
    )

    assert [row["sample_id"] for row in rows] == ["t1-0", "t1-1"]
    assert rows[0]["row_focus_targets"] == "ACTION:SCROLL"
    assert rows[0]["is_focus_row"] is True
    assert rows[1]["row_focus_targets"] == ""
    assert rows[1]["is_focus_row"] is False
    assert report["selected_unique_tasks"] == 1
    assert report["review_queue_rows_including_trajectory_context"] == 2


def test_recovery_queue_includes_causal_transition_evidence():
    records = [
        _row(
            "t1",
            0,
            failure="ACTION_MISMATCH",
            strategy="BACKTRACK",
            recovery_success=True,
        ),
        _row("t1", 1, action="PRESS_KEY"),
    ]

    rows, _ = build_weak_class_review_queue(
        records,
        "val",
        max_tasks_per_class=1,
    )

    source = rows[0]
    assert source["row_focus_targets"] == "RECOVERY:BACKTRACK"
    assert source["failure_state"] == "images/t1-0-after.png"
    assert source["executed_recovery_action"] == "PRESS_KEY"
    assert source["post_recovery_state"] == "images/t1-1-after.png"


def test_class_coverage_requires_only_zero_support_recovery_collection():
    train = [
        _row("t1", 0, action="SCROLL", failure="LOOP_DETECTED"),
        _row("t2", 0, action="SELECT", strategy="BACKTRACK", recovery_success=True),
        _row("t3", 0, action="NAVIGATE"),
    ]
    val = list(train)
    report = class_coverage(
        {"train": train, "val": val},
        minimum_validation_support=2,
    )

    assert report["target_validation_support"]["ACTION:SCROLL"] == 1
    assert report["target_validation_support"]["RECOVERY:BACKTRACK"] == 1
    assert report["missing_target_classes"] == [
        "RECOVERY:ABORT",
        "RECOVERY:RETRY",
    ]
    assert report["targeted_new_collection_required"] == [
        "RECOVERY:ABORT",
        "RECOVERY:RETRY",
    ]
    assert "ACTION:SCROLL" in report["below_planning_minimum"]


def test_reviewer_assignment_is_stable_per_task_and_has_overlap_flag():
    first = reviewer_assignment("task-123", seed=42)
    second = reviewer_assignment("task-123", seed=42)
    assert first == second
    assert first[0] in {"A", "B", "A+B"}
    assert first[1] is (first[0] == "A+B")


def test_existing_data_improvement_notebook_is_isolated_and_compiles():
    path = Path("notebooks/kaggle_gold_existing_data_improvement.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "audit_gold_existing_data.py" in source
    assert "test_rows_read" in source
    assert "kaggle_gold.ipynb" in source
    assert "subprocess.run(command, check=False)" in source
    assert "source_records_mutated" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")


def test_existing_data_audit_script_streams_zip_without_extraction(tmp_path):
    train = [
        _row(
            "train-task",
            0,
            action="SCROLL",
            bbox={"x": 10, "y": 95, "width": 20, "height": 20},
        ),
        _row("train-task", 1, action="CLICK"),
    ]
    val = [
        _row(
            "val-task",
            0,
            action="SELECT",
            failure="LOOP_DETECTED",
            strategy="BACKTRACK",
            recovery_success=True,
        ),
        _row("val-task", 1, action="NAVIGATE"),
    ]
    archive = tmp_path / "gold.zip"
    image = Image.new("RGB", (100, 100), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    image_bytes = payload.getvalue()

    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/split_train.json", json.dumps(train))
        handle.writestr("dataset/split_val.json", json.dumps(val))
        handle.writestr("dataset/split_test.json", "[]")
        for record in [*train, *val]:
            for field in ("state_before", "state_after"):
                handle.writestr(
                    f"dataset/{record['inputs'][field]}",
                    image_bytes,
                )

    output = tmp_path / "review"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path("src").resolve())
    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_gold_existing_data.py",
            "--data-root",
            str(archive),
            "--output-dir",
            str(output),
            "--review-tasks-per-class",
            "1",
            "--montage-rows-per-category",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(
        (output / "dataset_improvement_audit.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS"
    assert report["source_mode"] == "zip_stream"
    assert report["test_rows_read"] == 0
    assert report["source_images_copied_or_extracted"] is False
    assert report["outputs"]["bbox_review_queue_rows"] == 1
    assert (output / "bbox_mask_manifest.json").is_file()
    assert (output / "weak_class_review_queue.csv").is_file()
    assert not (tmp_path / "images").exists()
