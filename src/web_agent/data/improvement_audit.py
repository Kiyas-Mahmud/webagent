"""Deterministic review queues for improving the existing Gold dataset.

The functions in this module never mutate source records.  They turn train/val
records into:

* a complete ledger of invalid bbox labels that must stay masked unless direct
  replay/screenshot evidence supports a correction;
* trajectory-preserving review queues for weak action, failure, and recovery
  classes; and
* explicit class-coverage evidence showing whether targeted new collection is
  required.

The locked test split is intentionally outside this module's normal workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Callable, Iterable, Sequence

from web_agent.data.recovery_transitions import build_recovery_transition_index


TARGET_ACTIONS = ("SCROLL", "SELECT", "NAVIGATE")
TARGET_FAILURE_TYPES = ("LOOP_DETECTED",)
TARGET_RECOVERY_STRATEGIES = ("BACKTRACK", "RETRY", "ABORT")
REQUIRED_BBOX_KEYS = ("x", "y", "width", "height")


def _view(record: dict) -> tuple[dict, dict, dict]:
    if "inputs" in record and "labels" in record:
        return record["inputs"], record["labels"], record.get("meta", {})
    return record, record, record


def _identity(record: dict, fallback: int) -> tuple[str, str, int]:
    _, _, meta = _view(record)
    task_id = str(
        meta.get("task_id")
        or meta.get("original_task_id")
        or f"__row_{fallback}"
    )
    raw_step = meta.get("step_index", meta.get("step", fallback))
    try:
        step_index = int(raw_step)
    except (TypeError, ValueError):
        step_index = fallback
    sample_id = str(meta.get("sample_id") or f"{task_id}:{step_index}")
    return sample_id, task_id, step_index


def reviewer_assignment(task_id: str, *, seed: int = 42) -> tuple[str, bool]:
    """Assign a complete trajectory to A, B, or both, deterministically.

    Ten percent of tasks are assigned to both reviewers for agreement
    measurement.  Every row with the same task ID receives the same assignment.
    """
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 10:
        return "A+B", True
    return ("A" if bucket % 2 == 0 else "B"), False


def bbox_geometry_reasons(
    box: object,
    image_size: tuple[int, int] | None,
    *,
    image_error: str = "",
    tolerance: float = 1e-6,
) -> list[str]:
    """Return every geometry problem without clipping or repairing the box."""
    reasons: list[str] = []
    if not isinstance(box, dict):
        return ["bbox_not_object"]
    missing = [key for key in REQUIRED_BBOX_KEYS if key not in box]
    if missing:
        reasons.append("missing_keys")
        return reasons
    try:
        values = tuple(float(box[key]) for key in REQUIRED_BBOX_KEYS)
    except (TypeError, ValueError):
        return ["non_numeric"]

    if not all(math.isfinite(value) for value in values):
        reasons.append("non_finite")
    x, y, width, height = values
    if x < 0 or y < 0:
        reasons.append("negative_origin")
    if width <= 0 or height <= 0:
        reasons.append("non_positive_size")

    if image_error:
        reasons.append(image_error)
    elif image_size is None:
        reasons.append("unreadable_state_before_image")
    else:
        image_width, image_height = image_size
        if image_width <= 0 or image_height <= 0:
            reasons.append("invalid_image_size")
        else:
            if x > image_width + tolerance:
                reasons.append("x_origin_outside")
            if y > image_height + tolerance:
                reasons.append("y_origin_outside")
            if x + width > image_width + tolerance:
                reasons.append("right_boundary_overflow")
            if y + height > image_height + tolerance:
                reasons.append("bottom_boundary_overflow")
    return sorted(set(reasons))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _base_review_fields(task_id: str, *, seed: int) -> dict:
    assignment, double_review = reviewer_assignment(task_id, seed=seed)
    return {
        "reviewer_assignment": assignment,
        "double_review": double_review,
        "reviewer_id": "",
        "reviewed_at_utc": "",
        "review_decision": "",
        "reason_code": "",
        "evidence_reference": "",
        "reviewer_notes": "",
        "second_reviewer_id": "",
        "second_reviewer_decision": "",
        "final_decision": "",
        "final_change_applied_by": "",
        "final_change_verified_by": "",
    }


ImageInfoGetter = Callable[[str], tuple[tuple[int, int] | None, str]]


def build_bbox_review_queue(
    records: Sequence[dict],
    split: str,
    image_info: ImageInfoGetter,
    *,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Build a complete invalid-bbox mask/review ledger for one split."""
    rows: list[dict] = []
    reasons: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    bbox_rows = 0
    valid_rows = 0
    image_cache: dict[str, tuple[tuple[int, int] | None, str]] = {}

    for index, record in enumerate(records):
        inputs, labels, meta = _view(record)
        box = labels.get("action_target_bbox")
        if box is None:
            continue
        bbox_rows += 1
        image_path = str(inputs.get("state_before") or "")
        if image_path not in image_cache:
            image_cache[image_path] = image_info(image_path)
        size, image_error = image_cache[image_path]
        row_reasons = bbox_geometry_reasons(
            box,
            size,
            image_error=image_error,
        )
        if not row_reasons:
            valid_rows += 1
            continue

        sample_id, task_id, step_index = _identity(record, index)
        reasons.update(row_reasons)
        action = str(labels.get("action_type") or "")
        action_counts[action] += 1
        width, height = size if size is not None else ("", "")
        original = box if isinstance(box, dict) else {}
        row = {
            "split": split,
            "sample_id": sample_id,
            "task_id": task_id,
            "step_index": step_index,
            "state_before": image_path,
            "state_after": str(inputs.get("state_after") or ""),
            "image_width": width,
            "image_height": height,
            "action_type": action,
            "action_coordinates": _json(labels.get("action_coordinates")),
            "original_bbox": _json(box),
            "original_x": original.get("x", ""),
            "original_y": original.get("y", ""),
            "original_width": original.get("width", ""),
            "original_height": original.get("height", ""),
            "geometry_reasons": "|".join(row_reasons),
            "current_training_bbox_mask": 0,
            "recommended_disposition": "MASK_PENDING_DIRECT_EVIDENCE",
            "automatic_correction_allowed": False,
            "proposed_bbox_x": "",
            "proposed_bbox_y": "",
            "proposed_bbox_width": "",
            "proposed_bbox_height": "",
            "source_review_status": str(meta.get("review_status") or ""),
            **_base_review_fields(task_id, seed=seed),
        }
        rows.append(row)

    return rows, {
        "split": split,
        "records": len(records),
        "bbox_rows": bbox_rows,
        "valid_bbox_rows": valid_rows,
        "invalid_bbox_rows": len(rows),
        "invalid_reason_counts": dict(sorted(reasons.items())),
        "invalid_action_counts": dict(sorted(action_counts.items())),
        "training_disposition": "MASK_INVALID_BBOX_ONLY",
        "source_records_mutated": False,
        "automatic_corrections": 0,
    }


def _focus_tags(labels: dict) -> tuple[str, ...]:
    tags: list[str] = []
    action = str(labels.get("action_type") or "")
    failure = str(labels.get("failure_type_4") or "")
    strategy = str(labels.get("recovery_strategy") or "")
    if action in TARGET_ACTIONS:
        tags.append(f"ACTION:{action}")
    if failure in TARGET_FAILURE_TYPES:
        tags.append(f"FAILURE:{failure}")
    if strategy in TARGET_RECOVERY_STRATEGIES:
        tags.append(f"RECOVERY:{strategy}")
    return tuple(tags)


def _stable_task_order(task_ids: Iterable[str], *, seed: int, category: str) -> list[str]:
    return sorted(
        set(task_ids),
        key=lambda task_id: hashlib.sha256(
            f"{seed}:{category}:{task_id}".encode("utf-8")
        ).digest(),
    )


def build_weak_class_review_queue(
    records: Sequence[dict],
    split: str,
    *,
    max_tasks_per_class: int = 100,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Select complete trajectories for deterministic weak-class review."""
    if max_tasks_per_class <= 0:
        raise ValueError("max_tasks_per_class must be positive")

    task_rows: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    task_categories: dict[str, set[str]] = defaultdict(set)
    class_rows: Counter[str] = Counter()
    class_tasks: dict[str, set[str]] = defaultdict(set)
    identities: dict[int, tuple[str, str, int]] = {}

    for index, record in enumerate(records):
        sample_id, task_id, step_index = _identity(record, index)
        identities[id(record)] = (sample_id, task_id, step_index)
        task_rows[task_id].append((step_index, record))
        _, labels, _ = _view(record)
        for category in _focus_tags(labels):
            class_rows[category] += 1
            class_tasks[category].add(task_id)
            task_categories[task_id].add(category)

    selected_for: dict[str, set[str]] = defaultdict(set)
    selected_counts: dict[str, int] = {}
    for category in sorted(class_tasks):
        selected = _stable_task_order(
            class_tasks[category],
            seed=seed,
            category=f"{split}:{category}",
        )[:max_tasks_per_class]
        selected_counts[category] = len(selected)
        for task_id in selected:
            selected_for[task_id].add(category)

    transitions, transition_report = build_recovery_transition_index(records)
    output: list[dict] = []
    for task_id in sorted(selected_for):
        assignment_fields = _base_review_fields(task_id, seed=seed)
        selected_categories = "|".join(sorted(selected_for[task_id]))
        for _, record in sorted(task_rows[task_id], key=lambda item: item[0]):
            inputs, labels, meta = _view(record)
            sample_id, _, step_index = identities[id(record)]
            row_tags = _focus_tags(labels)
            transition = transitions.get(sample_id, {})
            output.append({
                "split": split,
                "sample_id": sample_id,
                "task_id": task_id,
                "step_index": step_index,
                "selected_task_targets": selected_categories,
                "row_focus_targets": "|".join(row_tags),
                "is_focus_row": bool(row_tags),
                "state_before": str(inputs.get("state_before") or ""),
                "state_after": str(inputs.get("state_after") or ""),
                "task_description": str(inputs.get("task_description") or ""),
                "website_domain": str(inputs.get("website_domain") or ""),
                "action_type": str(labels.get("action_type") or ""),
                "action_coordinates": _json(labels.get("action_coordinates")),
                "action_target_bbox": _json(labels.get("action_target_bbox")),
                "outcome_label": str(labels.get("outcome_label") or ""),
                "failure_type_4": str(labels.get("failure_type_4") or ""),
                "recovery_attempted": labels.get("recovery_success") is not None,
                "recovery_strategy": str(labels.get("recovery_strategy") or ""),
                "recovery_success": labels.get("recovery_success"),
                "failure_state": str(transition.get("failure_state") or ""),
                "executed_recovery_action": str(
                    transition.get("executed_recovery_action") or ""
                ),
                "recovery_action_value": str(
                    transition.get("recovery_action_value") or ""
                ),
                "post_recovery_state": str(
                    transition.get("post_recovery_state") or ""
                ),
                "source_review_status": str(meta.get("review_status") or ""),
                "action_label_ok": "",
                "bbox_ok": "",
                "outcome_ok": "",
                "failure_type_ok": "",
                "recovery_transition_ok": "",
                "proposed_action_type": "",
                "proposed_failure_type": "",
                "proposed_recovery_strategy": "",
                "proposed_recovery_success": "",
                **assignment_fields,
            })

    return output, {
        "split": split,
        "records": len(records),
        "focus_row_counts": dict(sorted(class_rows.items())),
        "focus_task_counts": {
            category: len(task_ids)
            for category, task_ids in sorted(class_tasks.items())
        },
        "selected_tasks_per_class": selected_counts,
        "selected_unique_tasks": len(selected_for),
        "review_queue_rows_including_trajectory_context": len(output),
        "transition_report": transition_report,
        "source_records_mutated": False,
    }


def class_coverage(
    records_by_split: dict[str, Sequence[dict]],
    *,
    minimum_validation_support: int = 100,
) -> dict:
    """Report targeted support and decide whether new collection is required."""
    if minimum_validation_support <= 0:
        raise ValueError("minimum_validation_support must be positive")

    distributions: dict[str, dict[str, dict[str, int]]] = {}
    for split, records in records_by_split.items():
        actions: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        recoveries: Counter[str] = Counter()
        for record in records:
            _, labels, _ = _view(record)
            actions[str(labels.get("action_type") or "")] += 1
            failures[str(labels.get("failure_type_4") or "")] += 1
            recoveries[str(labels.get("recovery_strategy") or "")] += 1
        distributions[split] = {
            "action_type": dict(sorted(actions.items())),
            "failure_type_4": dict(sorted(failures.items())),
            "recovery_strategy": dict(sorted(recoveries.items())),
        }

    val = distributions.get("val", {})
    val_actions = val.get("action_type", {})
    val_failures = val.get("failure_type_4", {})
    val_recoveries = val.get("recovery_strategy", {})
    support = {
        **{
            f"ACTION:{name}": int(val_actions.get(name, 0))
            for name in TARGET_ACTIONS
        },
        **{
            f"FAILURE:{name}": int(val_failures.get(name, 0))
            for name in TARGET_FAILURE_TYPES
        },
        **{
            f"RECOVERY:{name}": int(val_recoveries.get(name, 0))
            for name in TARGET_RECOVERY_STRATEGIES
        },
    }
    missing = sorted(name for name, count in support.items() if count == 0)
    below_minimum = sorted(
        name for name, count in support.items()
        if 0 < count < minimum_validation_support
    )
    collection_required = [
        name for name in missing
        if name.startswith("RECOVERY:")
    ]
    return {
        "distributions": distributions,
        "target_validation_support": support,
        "minimum_validation_support_for_review_planning": minimum_validation_support,
        "missing_target_classes": missing,
        "below_planning_minimum": below_minimum,
        "targeted_new_collection_required": collection_required,
        "existing_data_first": True,
        "interpretation": (
            "Zero-support recovery strategies require real reviewed trajectories "
            "only if the thesis trains or claims those strategies. Nonzero weak "
            "classes should be audited before any collection decision."
        ),
    }
