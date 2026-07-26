"""Pure helpers for immutable human-review sessions.

The review UI is intentionally thin.  Assignment filtering, event validation,
append-only logging, and completion accounting live here so they can be tested
without Jupyter, image libraries, or the large dataset.
"""

from __future__ import annotations

import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REVIEWERS = ("A", "B")
QUEUE_KINDS = ("bbox", "weak")
REVIEW_DECISIONS = (
    "approve_no_change",
    "approve_after_correction",
    "mask_bbox_no_evidence",
    "needs_discussion",
    "reject_recollect",
    "quarantine_policy",
)
CHECK_VALUES = ("", "yes", "no", "uncertain", "not_applicable")
REASON_CODES = (
    "",
    "BBOX_WRONG",
    "ACTION_LABEL_WRONG",
    "ACTION_COORDINATES_WRONG",
    "OUTCOME_WRONG",
    "FAILURE_TYPE_WRONG",
    "RECOVERY_LABEL_WRONG",
    "IMG_CAPTURE_INVALID",
    "IMG_WRONG_PAIR",
    "IMG_WRONG_TRAJECTORY",
    "TASK_ACTION_LEAK",
    "TASK_WRONG_GOAL",
    "ADULT_POLICY_PENDING",
    "OTHER",
)
ACTION_LABELS = ("", "CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY")
OUTCOME_LABELS = ("", "SUCCESS", "FAILURE")
FAILURE_LABELS = (
    "",
    "NONE",
    "ACTION_MISMATCH",
    "PERCEPTION_ERROR",
    "LOOP_DETECTED",
)
RECOVERY_LABELS = (
    "",
    "NONE",
    "RETRY",
    "REPLAN",
    "BACKTRACK",
    "ALTERNATIVE_TARGET",
    "ABORT",
)
RECOVERY_SUCCESS_VALUES = ("", "true", "false", "null")
RECOVERY_ATTEMPTED_VALUES = ("", "true", "false")

EVENT_FIELDS = (
    "event_id",
    "reviewed_at_utc",
    "queue_kind",
    "reviewer_id",
    "split",
    "sample_id",
    "task_id",
    "step_index",
    "reviewer_assignment",
    "double_review",
    "focus_targets",
    "review_decision",
    "reason_code",
    "evidence_reference",
    "reviewer_notes",
    "action_label_ok",
    "bbox_ok",
    "outcome_ok",
    "failure_type_ok",
    "recovery_transition_ok",
    "proposed_bbox",
    "proposed_action_type",
    "proposed_outcome_label",
    "proposed_failure_type",
    "proposed_recovery_attempted",
    "proposed_recovery_strategy",
    "proposed_recovery_success",
)


def load_queue_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _boolean_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value if value is not None else "").strip().lower()


def assigned_to(row: dict, reviewer_id: str) -> bool:
    if reviewer_id not in REVIEWERS:
        raise ValueError(f"reviewer_id must be one of {REVIEWERS}")
    return str(row.get("reviewer_assignment") or "") in {
        reviewer_id,
        "A+B",
    }


def review_targets(
    rows: Sequence[dict],
    *,
    queue_kind: str,
    reviewer_id: str,
) -> list[dict]:
    """Return this reviewer's unique decision targets in stable queue order."""
    if queue_kind not in QUEUE_KINDS:
        raise ValueError(f"queue_kind must be one of {QUEUE_KINDS}")
    selected: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not assigned_to(row, reviewer_id):
            continue
        if queue_kind == "weak" and not _truthy(row.get("is_focus_row")):
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        selected.append(row)
    return selected


def _clean_choice(value: object, choices: tuple[str, ...], field: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned not in choices:
        raise ValueError(f"{field} must be one of {choices}, got {cleaned!r}")
    return cleaned


def _proposed_bbox(
    values: Sequence[object] | None,
    *,
    image_width: object,
    image_height: object,
) -> str:
    if values is None or all(str(value or "").strip() == "" for value in values):
        return ""
    if len(values) != 4 or any(str(value or "").strip() == "" for value in values):
        raise ValueError("all four proposed bbox values are required together")
    try:
        x, y, width, height = (float(value) for value in values)
        canvas_width = float(image_width)
        canvas_height = float(image_height)
    except (TypeError, ValueError):
        raise ValueError("proposed bbox and image dimensions must be numeric") from None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("proposed bbox values must be finite")
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > canvas_width + 1e-6
        or y + height > canvas_height + 1e-6
    ):
        raise ValueError(
            "proposed bbox must have positive size and stay inside the native image"
        )
    return json.dumps(
        {"x": x, "y": y, "width": width, "height": height},
        sort_keys=True,
        separators=(",", ":"),
    )


def build_review_event(
    target: dict,
    *,
    queue_kind: str,
    reviewer_id: str,
    review_decision: str,
    reason_code: str = "",
    evidence_reference: str = "",
    reviewer_notes: str = "",
    action_label_ok: str = "",
    bbox_ok: str = "",
    outcome_ok: str = "",
    failure_type_ok: str = "",
    recovery_transition_ok: str = "",
    proposed_bbox: Sequence[object] | None = None,
    proposed_action_type: str = "",
    proposed_outcome_label: str = "",
    proposed_failure_type: str = "",
    proposed_recovery_attempted: str = "",
    proposed_recovery_strategy: str = "",
    proposed_recovery_success: str = "",
    reviewed_at_utc: str | None = None,
    event_id: str | None = None,
) -> dict[str, str]:
    """Validate one reviewer decision and return a flat immutable event."""
    if queue_kind not in QUEUE_KINDS:
        raise ValueError(f"queue_kind must be one of {QUEUE_KINDS}")
    if not assigned_to(target, reviewer_id):
        raise ValueError("target is not assigned to this reviewer")
    decision = _clean_choice(
        review_decision,
        REVIEW_DECISIONS,
        "review_decision",
    )
    reason = _clean_choice(reason_code, REASON_CODES, "reason_code")
    checks = {
        "action_label_ok": _clean_choice(
            action_label_ok,
            CHECK_VALUES,
            "action_label_ok",
        ),
        "bbox_ok": _clean_choice(bbox_ok, CHECK_VALUES, "bbox_ok"),
        "outcome_ok": _clean_choice(outcome_ok, CHECK_VALUES, "outcome_ok"),
        "failure_type_ok": _clean_choice(
            failure_type_ok,
            CHECK_VALUES,
            "failure_type_ok",
        ),
        "recovery_transition_ok": _clean_choice(
            recovery_transition_ok,
            CHECK_VALUES,
            "recovery_transition_ok",
        ),
    }
    bbox_value = _proposed_bbox(
        proposed_bbox,
        image_width=target.get("image_width"),
        image_height=target.get("image_height"),
    )
    action_value = _clean_choice(
        proposed_action_type,
        ACTION_LABELS,
        "proposed_action_type",
    )
    outcome_value = _clean_choice(
        proposed_outcome_label,
        OUTCOME_LABELS,
        "proposed_outcome_label",
    )
    failure_value = _clean_choice(
        proposed_failure_type,
        FAILURE_LABELS,
        "proposed_failure_type",
    )
    attempted_value = _clean_choice(
        proposed_recovery_attempted,
        RECOVERY_ATTEMPTED_VALUES,
        "proposed_recovery_attempted",
    )
    strategy_value = _clean_choice(
        proposed_recovery_strategy,
        RECOVERY_LABELS,
        "proposed_recovery_strategy",
    )
    success_value = _clean_choice(
        proposed_recovery_success,
        RECOVERY_SUCCESS_VALUES,
        "proposed_recovery_success",
    )
    has_correction = any(
        (
            bbox_value,
            action_value,
            outcome_value,
            failure_value,
            attempted_value,
            strategy_value,
            success_value,
        )
    )
    evidence = str(evidence_reference or "").strip()

    if decision == "approve_after_correction" and not has_correction:
        raise ValueError("approve_after_correction requires a proposed value")
    if has_correction and decision != "approve_after_correction":
        raise ValueError(
            "proposed values require review_decision=approve_after_correction"
        )
    if has_correction and not evidence:
        raise ValueError("every proposed correction requires evidence_reference")
    if decision == "approve_no_change" and any(
        value in {"no", "uncertain"} for value in checks.values()
    ):
        raise ValueError(
            "approve_no_change cannot contain a failed or uncertain check"
        )
    if decision == "approve_no_change" and any(
        value not in {"yes", "not_applicable"} for value in checks.values()
    ):
        raise ValueError(
            "approve_no_change requires every review check to be yes or "
            "not_applicable"
        )
    if decision == "mask_bbox_no_evidence":
        if queue_kind != "bbox":
            raise ValueError("mask_bbox_no_evidence is valid only for bbox queue rows")
        if bbox_value:
            raise ValueError("a masked bbox cannot also propose a replacement")
        if checks["bbox_ok"] not in {"no", "uncertain"}:
            raise ValueError(
                "mask_bbox_no_evidence requires bbox_ok=no or uncertain"
            )
    if decision in {
        "needs_discussion",
        "reject_recollect",
        "quarantine_policy",
        "mask_bbox_no_evidence",
    } and not reason:
        raise ValueError(f"{decision} requires a reason_code")

    correction_checks = {
        "proposed_bbox": (bbox_value, "bbox_ok"),
        "proposed_action_type": (action_value, "action_label_ok"),
        "proposed_outcome_label": (outcome_value, "outcome_ok"),
        "proposed_failure_type": (failure_value, "failure_type_ok"),
        "proposed_recovery_attempted": (
            attempted_value,
            "recovery_transition_ok",
        ),
        "proposed_recovery_strategy": (
            strategy_value,
            "recovery_transition_ok",
        ),
        "proposed_recovery_success": (
            success_value,
            "recovery_transition_ok",
        ),
    }
    for field, (proposed_value, check_field) in correction_checks.items():
        if proposed_value and checks[check_field] not in {"no", "uncertain"}:
            raise ValueError(
                f"{field} requires {check_field}=no or uncertain"
            )

    if outcome_value or failure_value:
        effective_outcome = outcome_value or str(
            target.get("outcome_label") or ""
        )
        effective_failure = failure_value or str(
            target.get("failure_type_4") or ""
        )
        if (
            effective_outcome not in OUTCOME_LABELS[1:]
            or effective_failure not in FAILURE_LABELS[1:]
        ):
            raise ValueError(
                "proposed outcome/failure correction requires complete "
                "existing labels"
            )
        if (
            effective_outcome == "SUCCESS"
            and effective_failure != "NONE"
        ) or (
            effective_outcome == "FAILURE"
            and effective_failure == "NONE"
        ):
            raise ValueError(
                "proposed outcome/failure values violate label consistency"
            )

    if attempted_value or strategy_value or success_value:
        current_attempted = _boolean_text(target.get("recovery_attempted"))
        effective_attempted = attempted_value or current_attempted
        effective_strategy = strategy_value or str(
            target.get("recovery_strategy") or ""
        )
        current_success = str(
            target.get("recovery_success")
            if target.get("recovery_success") is not None
            else "null"
        ).strip().lower()
        effective_success = success_value or current_success
        if (
            effective_attempted not in {"true", "false"}
            or effective_strategy not in RECOVERY_LABELS[1:]
            or effective_success not in {"true", "false", "null"}
        ):
            raise ValueError(
                "proposed recovery correction requires a complete existing "
                "recovery tuple"
            )
        if effective_attempted == "false" and (
            effective_strategy != "NONE" or effective_success != "null"
        ):
            raise ValueError(
                "non-attempted recovery requires strategy NONE and success null"
            )
        if effective_attempted == "true" and (
            effective_strategy in {"", "NONE"}
            or effective_success not in {"true", "false"}
        ):
            raise ValueError(
                "attempted recovery requires a strategy and true/false success"
            )

    timestamp = reviewed_at_utc or datetime.now(timezone.utc).isoformat()
    identifier = event_id or str(uuid.uuid4())
    try:
        uuid.UUID(identifier)
    except (ValueError, AttributeError):
        raise ValueError("event_id must be a UUID") from None

    focus = (
        target.get("row_focus_targets")
        or target.get("selected_task_targets")
        or ""
    )
    event = {
        "event_id": identifier,
        "reviewed_at_utc": timestamp,
        "queue_kind": queue_kind,
        "reviewer_id": reviewer_id,
        "split": str(target.get("split") or ""),
        "sample_id": str(target.get("sample_id") or ""),
        "task_id": str(target.get("task_id") or ""),
        "step_index": str(
            "" if target.get("step_index") is None else target.get("step_index")
        ),
        "reviewer_assignment": str(target.get("reviewer_assignment") or ""),
        "double_review": str(_truthy(target.get("double_review"))),
        "focus_targets": str(focus),
        "review_decision": decision,
        "reason_code": reason,
        "evidence_reference": evidence,
        "reviewer_notes": str(reviewer_notes or "").strip(),
        **checks,
        "proposed_bbox": bbox_value,
        "proposed_action_type": action_value,
        "proposed_outcome_label": outcome_value,
        "proposed_failure_type": failure_value,
        "proposed_recovery_attempted": attempted_value,
        "proposed_recovery_strategy": strategy_value,
        "proposed_recovery_success": success_value,
    }
    if not event["sample_id"] or not event["task_id"]:
        raise ValueError("target must contain sample_id and task_id")
    return {field: str(event[field]) for field in EVENT_FIELDS}


def append_review_event(path: str | Path, event: dict[str, str]) -> None:
    """Append one event; never rewrite or remove prior reviewer decisions."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    missing = [field for field in EVENT_FIELDS if field not in event]
    extras = [field for field in event if field not in EVENT_FIELDS]
    if missing or extras:
        raise ValueError(f"invalid event columns; missing={missing}, extras={extras}")
    exists = destination.is_file() and destination.stat().st_size > 0
    if exists:
        with destination.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != EVENT_FIELDS:
            raise ValueError("existing review log has an incompatible header")
    with destination.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(event)


def load_review_events(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return []
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != EVENT_FIELDS:
        raise ValueError("review log has an incompatible header")
    return rows


def latest_events(
    events: Iterable[dict],
    *,
    queue_kind: str,
    reviewer_id: str,
) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for event in events:
        if (
            event.get("queue_kind") == queue_kind
            and event.get("reviewer_id") == reviewer_id
        ):
            latest[str(event["sample_id"])] = event
    return latest


def completion_summary(
    targets: Sequence[dict],
    events: Iterable[dict],
    *,
    queue_kind: str,
    reviewer_id: str,
) -> dict:
    event_rows = list(events)
    latest = latest_events(
        event_rows,
        queue_kind=queue_kind,
        reviewer_id=reviewer_id,
    )
    target_ids = [str(row["sample_id"]) for row in targets]
    completed = [sample_id for sample_id in target_ids if sample_id in latest]
    incomplete = [sample_id for sample_id in target_ids if sample_id not in latest]
    unresolved = [
        sample_id
        for sample_id in target_ids
        if latest.get(sample_id, {}).get("review_decision") == "needs_discussion"
    ]
    decision_counts: dict[str, int] = {}
    for sample_id in completed:
        decision = str(latest[sample_id]["review_decision"])
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    return {
        "queue_kind": queue_kind,
        "reviewer_id": reviewer_id,
        "assigned_targets": len(target_ids),
        "completed_targets": len(completed),
        "remaining_targets": len(incomplete),
        "completion_fraction": len(completed) / max(len(target_ids), 1),
        "incomplete_sample_ids": incomplete,
        "unresolved_targets": len(unresolved),
        "unresolved_sample_ids": unresolved,
        "latest_decision_counts": dict(sorted(decision_counts.items())),
        "immutable_event_rows": len(event_rows),
    }
