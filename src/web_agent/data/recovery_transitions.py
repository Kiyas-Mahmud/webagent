"""Causal recovery-transition construction and dataset audits.

The source split remains the single source of truth.  A transition manifest only
references existing rows/images; it never copies the large image dataset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

def _view(record: dict):
    if "inputs" in record and "labels" in record:
        return record["inputs"], record["labels"], record.get("meta", {})
    return record, record, record


def _identity(record: dict, fallback: int) -> tuple[str, int, str]:
    """Return stable (task_id, step_index, sample_id) fields for either layout."""
    _, _, meta = _view(record)
    task_id = str(
        meta.get("task_id") or meta.get("original_task_id") or f"__row_{fallback}"
    )
    raw_step = meta.get("step_index", meta.get("step", fallback))
    try:
        step_index = int(raw_step)
    except (TypeError, ValueError):
        step_index = fallback
    sample_id = str(meta.get("sample_id") or f"{task_id}:{step_index}")
    return task_id, step_index, sample_id


def build_recovery_transition_index(records: Iterable[dict]) -> tuple[dict[str, dict], dict]:
    """Map attempted-recovery sample IDs to their next causal trajectory step.

    The transition is:
      current state_after -> next row's executed action -> next row's state_after.
    The current row supplies the intended strategy and recovery_success target.
    """
    rows = list(records)
    by_task: dict[str, list[tuple[int, int, str, dict]]] = defaultdict(list)
    for position, record in enumerate(rows):
        task_id, step_index, sample_id = _identity(record, position)
        by_task[task_id].append((step_index, position, sample_id, record))

    transitions: dict[str, dict] = {}
    missing_next: list[str] = []
    ambiguous_next: list[str] = []
    invalid_attempted: list[str] = []
    nonconsecutive_next: list[str] = []
    attempted = 0
    for task_rows in by_task.values():
        task_rows.sort(key=lambda item: (item[0], item[1]))
        for offset, (step_index, _, sample_id, record) in enumerate(task_rows):
            current_inputs, current_labels, current_meta = _view(record)
            if current_labels.get("recovery_success") is None:
                continue
            attempted += 1
            if (
                current_labels.get("recovery_strategy") in (None, "NONE")
                or current_labels.get("outcome_label") not in (None, "FAILURE")
            ):
                invalid_attempted.append(sample_id)
                continue
            if offset + 1 >= len(task_rows):
                missing_next.append(sample_id)
                continue
            next_step, _, next_sample_id, next_record = task_rows[offset + 1]
            next_inputs, next_labels, next_meta = _view(next_record)
            if next_step <= step_index:
                ambiguous_next.append(sample_id)
                continue
            if next_step != step_index + 1:
                nonconsecutive_next.append(sample_id)
            if not current_inputs.get("state_after") or not next_inputs.get("state_after"):
                missing_next.append(sample_id)
                continue
            transitions[sample_id] = {
                "source_sample_id": sample_id,
                "recovery_sample_id": next_sample_id,
                "task_id": str(current_meta.get("task_id", "")),
                "source_step_index": step_index,
                "recovery_step_index": next_step,
                "failure_state": current_inputs["state_after"],
                "executed_recovery_action": next_labels.get("action_type", ""),
                "post_recovery_state": next_inputs["state_after"],
                "recovery_strategy": current_labels.get("recovery_strategy", "NONE"),
                "recovery_success": bool(current_labels["recovery_success"]),
                "task_description": current_inputs.get("task_description", ""),
                "website_domain": current_inputs.get("website_domain", ""),
                "recovery_action_value": next_labels.get("action_value", ""),
                "source_meta": {
                    "step_index": current_meta.get("step_index"),
                    "recovery_step_index": next_meta.get("step_index"),
                },
            }

    report = {
        "rows": len(rows),
        "tasks": len(by_task),
        "attempted_recoveries": attempted,
        "proper_transitions": len(transitions),
        "missing_next_step": len(missing_next),
        "ambiguous_next_step": len(ambiguous_next),
        "nonconsecutive_next_step": len(nonconsecutive_next),
        "invalid_attempted_label_contract": len(invalid_attempted),
        "coverage": len(transitions) / max(attempted, 1),
        "missing_next_sample_ids": missing_next[:100],
        "ambiguous_next_sample_ids": ambiguous_next[:100],
        "nonconsecutive_next_sample_ids": nonconsecutive_next[:100],
        "invalid_attempted_sample_ids": invalid_attempted[:100],
    }
    return transitions, report


def recovery_class_audit(records: Iterable[dict]) -> dict:
    """Count the labels that determine whether targeted collection is required."""
    strategies: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    recovery_success: Counter[str] = Counter()
    integrity: Counter[str] = Counter()
    attempted = 0
    rows = list(records)
    for record in rows:
        _, labels, _ = _view(record)
        strategy = str(labels.get("recovery_strategy", "NONE"))
        strategies[strategy] += 1
        failure_types[str(labels.get("failure_type_4", "NONE"))] += 1
        value = labels.get("recovery_success")
        recovery_success["NOT_ATTEMPTED" if value is None else str(bool(value))] += 1
        attempted += value is not None
        if value is not None and strategy == "NONE":
            integrity["attempted_with_none_strategy"] += 1
        if value is None and strategy != "NONE":
            integrity["strategy_without_observed_outcome"] += 1
        explicit_attempt = labels.get("recovery_attempted")
        if explicit_attempt is not None and bool(explicit_attempt) != (value is not None):
            integrity["recovery_attempted_flag_mismatch"] += 1
    tracked = ("RETRY", "ABORT", "BACKTRACK")
    return {
        "rows": len(rows),
        "attempted_recoveries": attempted,
        "recovery_strategy": dict(sorted(strategies.items())),
        "failure_type": dict(sorted(failure_types.items())),
        "recovery_success": dict(sorted(recovery_success.items())),
        "targeted_counts": {
            **{name: strategies.get(name, 0) for name in tracked},
            "LOOP_DETECTED": failure_types.get("LOOP_DETECTED", 0),
        },
        "missing_claimed_classes": [
            name for name in (*tracked, "LOOP_DETECTED")
            if (strategies.get(name, 0) if name != "LOOP_DETECTED"
                else failure_types.get(name, 0)) == 0
        ],
        "integrity_violations": {
            name: integrity.get(name, 0)
            for name in (
                "attempted_with_none_strategy",
                "strategy_without_observed_outcome",
                "recovery_attempted_flag_mismatch",
            )
        },
    }


def transition_manifest(records: Iterable[dict], split: str) -> dict:
    transitions, transition_report = build_recovery_transition_index(records)
    return {
        "schema_version": "recovery-transition-v2",
        "split": split,
        "definition": (
            "failure state -> executed recovery -> post-recovery state -> recovery success"
        ),
        "transitions": list(transitions.values()),
        "transition_report": transition_report,
        "class_audit": recovery_class_audit(records),
    }
