from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from web_agent.data.review_session import (
    EVENT_FIELDS,
    append_review_event,
    build_review_event,
    completion_summary,
    load_review_events,
    review_targets,
)
from web_agent.data.review_reconciliation import (
    cohen_kappa,
    reconcile_reviews,
)


def _target(
    sample_id: str,
    *,
    assignment: str = "A",
    focus: bool = True,
    queue_kind: str = "bbox",
) -> dict:
    row = {
        "split": "train",
        "sample_id": sample_id,
        "task_id": "task-1",
        "step_index": 0,
        "reviewer_assignment": assignment,
        "double_review": assignment == "A+B",
        "image_width": 100,
        "image_height": 80,
    }
    if queue_kind == "weak":
        row.update({
            "is_focus_row": focus,
            "row_focus_targets": "ACTION:SCROLL" if focus else "",
            "selected_task_targets": "ACTION:SCROLL",
        })
    return row


def _approve_weak(target: dict, reviewer_id: str) -> dict[str, str]:
    return build_review_event(
        target,
        queue_kind="weak",
        reviewer_id=reviewer_id,
        review_decision="approve_no_change",
        action_label_ok="yes",
        bbox_ok="not_applicable",
        outcome_ok="yes",
        failure_type_ok="yes",
        recovery_transition_ok="not_applicable",
    )


def test_review_targets_filters_assignment_and_weak_context_rows():
    rows = [
        _target("a", assignment="A", queue_kind="weak"),
        _target("context", assignment="A", focus=False, queue_kind="weak"),
        _target("both", assignment="A+B", queue_kind="weak"),
        _target("b", assignment="B", queue_kind="weak"),
    ]
    targets = review_targets(rows, queue_kind="weak", reviewer_id="A")
    assert [row["sample_id"] for row in targets] == ["a", "both"]


def test_bbox_correction_requires_complete_in_bounds_box_and_evidence():
    target = _target("bbox-1")
    event = build_review_event(
        target,
        queue_kind="bbox",
        reviewer_id="A",
        review_decision="approve_after_correction",
        reason_code="BBOX_WRONG",
        evidence_reference="replay:task-1:step-0",
        bbox_ok="no",
        proposed_bbox=(10, 15, 20, 25),
        event_id=str(uuid.UUID(int=1)),
        reviewed_at_utc="2026-07-27T00:00:00+00:00",
    )
    assert event["step_index"] == "0"
    assert event["proposed_bbox"] == (
        '{"height":25.0,"width":20.0,"x":10.0,"y":15.0}'
    )

    with pytest.raises(ValueError, match="evidence_reference"):
        build_review_event(
            target,
            queue_kind="bbox",
            reviewer_id="A",
            review_decision="approve_after_correction",
            proposed_bbox=(10, 15, 20, 25),
        )
    with pytest.raises(ValueError, match="stay inside"):
        build_review_event(
            target,
            queue_kind="bbox",
            reviewer_id="A",
            review_decision="approve_after_correction",
            evidence_reference="replay",
            proposed_bbox=(90, 70, 20, 20),
        )


def test_mask_decision_is_bbox_only_and_requires_reason():
    target = _target("bbox-1")
    event = build_review_event(
        target,
        queue_kind="bbox",
        reviewer_id="A",
        review_decision="mask_bbox_no_evidence",
        reason_code="BBOX_WRONG",
        bbox_ok="no",
    )
    assert event["review_decision"] == "mask_bbox_no_evidence"

    with pytest.raises(ValueError, match="valid only for bbox"):
        build_review_event(
            _target("weak-1", queue_kind="weak"),
            queue_kind="weak",
            reviewer_id="A",
            review_decision="mask_bbox_no_evidence",
            reason_code="BBOX_WRONG",
        )


def test_approve_no_change_rejects_failed_or_uncertain_checks():
    with pytest.raises(ValueError, match="failed or uncertain"):
        build_review_event(
            _target("weak-1", queue_kind="weak"),
            queue_kind="weak",
            reviewer_id="A",
            review_decision="approve_no_change",
            action_label_ok="no",
        )
    with pytest.raises(ValueError, match="every review check"):
        build_review_event(
            _target("weak-1", queue_kind="weak"),
            queue_kind="weak",
            reviewer_id="A",
            review_decision="approve_no_change",
        )


def test_weak_label_correction_is_recorded_only_with_evidence():
    target = _target("weak-1", queue_kind="weak")
    target.update({
        "outcome_label": "SUCCESS",
        "failure_type_4": "NONE",
    })
    event = build_review_event(
        target,
        queue_kind="weak",
        reviewer_id="A",
        review_decision="approve_after_correction",
        reason_code="OUTCOME_WRONG",
        evidence_reference="replay:weak-1",
        outcome_ok="no",
        failure_type_ok="no",
        proposed_outcome_label="FAILURE",
        proposed_failure_type="PERCEPTION_ERROR",
    )
    assert event["proposed_outcome_label"] == "FAILURE"


def test_corrected_labels_must_remain_logically_consistent():
    target = _target("weak-1", queue_kind="weak")
    target.update({
        "outcome_label": "FAILURE",
        "failure_type_4": "PERCEPTION_ERROR",
    })
    with pytest.raises(ValueError, match="label consistency"):
        build_review_event(
            target,
            queue_kind="weak",
            reviewer_id="A",
            review_decision="approve_after_correction",
            reason_code="OUTCOME_WRONG",
            evidence_reference="replay",
            outcome_ok="no",
            proposed_outcome_label="SUCCESS",
        )


def test_recovery_correction_records_attempted_and_enforces_tuple():
    target = _target("weak-1", queue_kind="weak")
    target.update({
        "recovery_attempted": True,
        "recovery_strategy": "REPLAN",
        "recovery_success": False,
    })
    with pytest.raises(ValueError, match="non-attempted recovery"):
        build_review_event(
            target,
            queue_kind="weak",
            reviewer_id="A",
            review_decision="approve_after_correction",
            reason_code="RECOVERY_LABEL_WRONG",
            evidence_reference="replay",
            recovery_transition_ok="no",
            proposed_recovery_attempted="false",
        )
    event = build_review_event(
        target,
        queue_kind="weak",
        reviewer_id="A",
        review_decision="approve_after_correction",
        reason_code="RECOVERY_LABEL_WRONG",
        evidence_reference="replay",
        recovery_transition_ok="no",
        proposed_recovery_attempted="false",
        proposed_recovery_strategy="NONE",
        proposed_recovery_success="null",
    )
    assert event["proposed_recovery_attempted"] == "false"


def test_append_log_is_immutable_and_completion_uses_latest_event(tmp_path):
    targets = [_target("one"), _target("two")]
    path = tmp_path / "reviewer_A.csv"
    first = build_review_event(
        targets[0],
        queue_kind="bbox",
        reviewer_id="A",
        review_decision="mask_bbox_no_evidence",
        reason_code="BBOX_WRONG",
        bbox_ok="no",
        event_id=str(uuid.UUID(int=1)),
        reviewed_at_utc="2026-07-27T00:00:00+00:00",
    )
    revision = build_review_event(
        targets[0],
        queue_kind="bbox",
        reviewer_id="A",
        review_decision="needs_discussion",
        reason_code="OTHER",
        event_id=str(uuid.UUID(int=2)),
        reviewed_at_utc="2026-07-27T00:01:00+00:00",
    )
    append_review_event(path, first)
    append_review_event(path, revision)

    events = load_review_events(path)
    assert len(events) == 2
    assert [event["event_id"] for event in events] == [
        str(uuid.UUID(int=1)),
        str(uuid.UUID(int=2)),
    ]
    summary = completion_summary(
        targets,
        events,
        queue_kind="bbox",
        reviewer_id="A",
    )
    assert summary["assigned_targets"] == 2
    assert summary["completed_targets"] == 1
    assert summary["remaining_targets"] == 1
    assert summary["unresolved_targets"] == 1
    assert summary["unresolved_sample_ids"] == ["one"]
    assert summary["latest_decision_counts"] == {"needs_discussion": 1}
    assert summary["immutable_event_rows"] == 2
    assert summary["incomplete_sample_ids"] == ["two"]


def test_existing_log_header_must_match(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wrong"])
        writer.writerow(["value"])
    event = {field: "" for field in EVENT_FIELDS}
    with pytest.raises(ValueError, match="incompatible header"):
        append_review_event(path, event)


def test_manual_review_notebook_is_isolated_resumable_and_compiles():
    path = Path("notebooks/kaggle_gold_manual_review.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])

    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "bbox_review_queue.csv" in source
    assert "weak_class_review_queue.csv" in source
    assert "split_train.json" in source
    assert "split_val.json" in source
    assert "split_test.json" not in source
    assert "reviewer_{REVIEWER_ID}_{QUEUE_KIND}_review_events.csv" in source
    assert "resume_matches" in source
    assert "kaggle_gold.ipynb" not in source

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(
                "".join(cell.get("source", [])),
                f"{path}:cell-{index}",
                "exec",
            )


def test_reconciliation_requires_second_review_for_correction():
    bbox = _target("bbox-1", assignment="A")
    bbox.update({
        "original_bbox": '{"x":1,"y":90,"width":10,"height":10}',
        "action_type": "CLICK",
    })
    weak = _target("weak-1", assignment="A+B", queue_kind="weak")
    weak.update({
        "action_type": "SCROLL",
        "outcome_label": "FAILURE",
        "failure_type_4": "PERCEPTION_ERROR",
        "recovery_strategy": "REPLAN",
        "recovery_success": "false",
    })
    primary_bbox = build_review_event(
        bbox,
        queue_kind="bbox",
        reviewer_id="A",
        review_decision="approve_after_correction",
        reason_code="BBOX_WRONG",
        evidence_reference="replay:bbox-1",
        bbox_ok="no",
        proposed_bbox=(10, 15, 20, 25),
    )
    weak_a = _approve_weak(weak, "A")
    weak_b = _approve_weak(weak, "B")

    first = reconcile_reviews([bbox], [weak], [primary_bbox, weak_a, weak_b])
    assert first["report"]["status"] == "INCOMPLETE"
    assert first["report"]["missing_secondary_reviews"] == 1
    assert first["secondary_rows"]["bbox"][0]["reviewer_assignment"] == "B"
    assert first["report"]["controlled_mini_permitted"] is False
    assert first["report"]["publication_ready"] is False

    secondary_target = dict(bbox)
    secondary_target["reviewer_assignment"] = "B"
    secondary_bbox = build_review_event(
        secondary_target,
        queue_kind="bbox",
        reviewer_id="B",
        review_decision="approve_after_correction",
        reason_code="BBOX_WRONG",
        evidence_reference="replay:bbox-1:independent",
        bbox_ok="no",
        proposed_bbox=(10, 15, 20, 25),
    )
    final = reconcile_reviews(
        [bbox],
        [weak],
        [primary_bbox, weak_a, weak_b, secondary_bbox],
    )
    assert final["report"]["status"] == "PASS"
    assert final["report"]["controlled_mini_permitted"] is True
    assert final["report"]["paired_reviews"] == 2
    assert final["report"]["approved_corrections"] == 1
    assert final["corrections"][0]["step_index"] == "0"
    assert final["corrections"][0]["confirmation_event_id"] == (
        secondary_bbox["event_id"]
    )


def test_reconciliation_preserves_disagreement_and_blocks_mini():
    weak = _target("weak-1", assignment="A+B", queue_kind="weak")
    event_a = _approve_weak(weak, "A")
    event_b = build_review_event(
        weak,
        queue_kind="weak",
        reviewer_id="B",
        review_decision="reject_recollect",
        reason_code="IMG_CAPTURE_INVALID",
    )
    result = reconcile_reviews([], [weak], [event_a, event_b])
    assert result["report"]["status"] == "INCOMPLETE"
    assert result["report"]["disagreements"] == 1
    assert result["report"]["controlled_mini_permitted"] is False
    assert len(result["disagreements"]) == 1
    assert result["dispositions"] == []


def test_cohen_kappa_handles_agreement_and_undefined_constant_labels():
    assert cohen_kappa(["yes", "yes", "no"], ["yes", "no", "no"]) == pytest.approx(
        0.4
    )
    assert cohen_kappa(["yes"], ["yes"]) is None
    assert cohen_kappa([], []) is None


def test_reconciliation_cli_writes_auditable_pass_package(tmp_path):
    bbox_path = tmp_path / "bbox.csv"
    with bbox_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle,
            fieldnames=["split", "sample_id", "task_id", "reviewer_assignment"],
        ).writeheader()

    weak = _target("weak-1", assignment="A+B", queue_kind="weak")
    weak.update({
        "action_type": "SCROLL",
        "outcome_label": "FAILURE",
        "failure_type_4": "PERCEPTION_ERROR",
        "recovery_strategy": "REPLAN",
        "recovery_success": "false",
    })
    weak_path = tmp_path / "weak.csv"
    with weak_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weak))
        writer.writeheader()
        writer.writerow(weak)

    log_paths = []
    for reviewer_id in ("A", "B"):
        event = _approve_weak(weak, reviewer_id)
        log_path = tmp_path / f"reviewer_{reviewer_id}.csv"
        append_review_event(log_path, event)
        log_paths.append(log_path)

    output_dir = tmp_path / "result"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/reconcile_gold_reviews.py",
            "--bbox-queue",
            str(bbox_path),
            "--weak-queue",
            str(weak_path),
            "--reviewer-log",
            str(log_paths[0]),
            "--reviewer-log",
            str(log_paths[1]),
            "--output-dir",
            str(output_dir),
            "--require-pass",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (output_dir / "review_reconciliation_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "PASS"
    assert report["test_rows_read"] == 0
    assert report["publication_ready"] is False
    assert (output_dir / "gold_review_reconciliation_package.zip").is_file()
