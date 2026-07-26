"""Reconcile immutable Web-Gold-40K reviewer events.

This module never edits dataset records. It validates the two-person review
contract, builds secondary-review queues for risky primary decisions, measures
agreement, and emits correction/disposition ledgers only when both required
reviewers agree.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from web_agent.data.review_session import EVENT_FIELDS, REVIEWERS


RISKY_DECISIONS = {
    "approve_after_correction",
    "needs_discussion",
    "reject_recollect",
    "quarantine_policy",
}
PROPOSAL_FIELDS = (
    "proposed_bbox",
    "proposed_action_type",
    "proposed_outcome_label",
    "proposed_failure_type",
    "proposed_recovery_attempted",
    "proposed_recovery_strategy",
    "proposed_recovery_success",
)
AGREEMENT_FIELDS = (
    "review_decision",
    "action_label_ok",
    "bbox_ok",
    "outcome_ok",
    "failure_type_ok",
    "recovery_transition_ok",
    *PROPOSAL_FIELDS,
)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _decision_targets(rows: Sequence[dict], queue_kind: str) -> dict[str, dict]:
    targets: dict[str, dict] = {}
    for row in rows:
        if queue_kind == "weak" and not _truthy(row.get("is_focus_row")):
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{queue_kind} queue contains an empty sample_id")
        assignment = str(row.get("reviewer_assignment") or "")
        if assignment not in {"A", "B", "A+B"}:
            raise ValueError(
                f"{queue_kind}:{sample_id} has invalid assignment {assignment!r}"
            )
        prior = targets.get(sample_id)
        if prior and (
            str(prior.get("task_id")) != str(row.get("task_id"))
            or str(prior.get("reviewer_assignment")) != assignment
        ):
            raise ValueError(
                f"{queue_kind}:{sample_id} has inconsistent duplicate queue rows"
            )
        targets.setdefault(sample_id, dict(row))
    return targets


def _latest_event_map(events: Iterable[dict]) -> dict[tuple[str, str, str], dict]:
    latest: dict[tuple[str, str, str], dict] = {}
    seen_ids: set[str] = set()
    for event in events:
        missing = [field for field in EVENT_FIELDS if field not in event]
        if missing:
            raise ValueError(f"review event missing fields: {missing}")
        event_id = str(event["event_id"])
        if event_id in seen_ids:
            raise ValueError(f"duplicate immutable event_id: {event_id}")
        seen_ids.add(event_id)
        key = (
            str(event["queue_kind"]),
            str(event["reviewer_id"]),
            str(event["sample_id"]),
        )
        latest[key] = dict(event)
    return latest


def _other_reviewer(reviewer_id: str) -> str:
    return "B" if reviewer_id == "A" else "A"


def _primary_reviewers(assignment: str) -> tuple[str, ...]:
    if assignment == "A+B":
        return REVIEWERS
    return (assignment,)


def _event_signature(event: dict) -> tuple[str, ...]:
    return tuple(str(event.get(field) or "") for field in (
        "review_decision",
        *PROPOSAL_FIELDS,
    ))


def cohen_kappa(values_a: Sequence[str], values_b: Sequence[str]) -> float | None:
    """Return unweighted Cohen's kappa, or None when it is undefined."""
    if len(values_a) != len(values_b):
        raise ValueError("Cohen kappa inputs must have equal length")
    if not values_a:
        return None
    count = len(values_a)
    observed = sum(a == b for a, b in zip(values_a, values_b, strict=True)) / count
    frequencies_a = Counter(values_a)
    frequencies_b = Counter(values_b)
    labels = set(frequencies_a) | set(frequencies_b)
    expected = sum(
        (frequencies_a[label] / count) * (frequencies_b[label] / count)
        for label in labels
    )
    if abs(1.0 - expected) < 1e-12:
        return None
    return (observed - expected) / (1.0 - expected)


def _agreement_metrics(pairs: Sequence[dict]) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for field in AGREEMENT_FIELDS:
        values_a = [str(pair["event_a"].get(field) or "") for pair in pairs]
        values_b = [str(pair["event_b"].get(field) or "") for pair in pairs]
        compared = len(values_a)
        agreements = sum(
            a == b for a, b in zip(values_a, values_b, strict=True)
        )
        metrics[field] = {
            "compared_pairs": compared,
            "agreements": agreements,
            "raw_agreement": agreements / compared if compared else None,
            "cohen_kappa": cohen_kappa(values_a, values_b),
        }
    return metrics


def _correction_row(
    target: dict,
    event: dict,
    *,
    queue_kind: str,
    confirmation_event: dict | None,
) -> dict[str, str]:
    return {
        "queue_kind": queue_kind,
        "split": str(target.get("split") or ""),
        "sample_id": str(target.get("sample_id") or ""),
        "task_id": str(target.get("task_id") or ""),
        "step_index": str(
            "" if target.get("step_index") is None else target.get("step_index")
        ),
        "original_bbox": str(
            target.get("original_bbox")
            or target.get("action_target_bbox")
            or ""
        ),
        "original_action_type": str(target.get("action_type") or ""),
        "original_outcome_label": str(target.get("outcome_label") or ""),
        "original_failure_type": str(target.get("failure_type_4") or ""),
        "original_recovery_attempted": str(
            ""
            if target.get("recovery_attempted") is None
            else target.get("recovery_attempted")
        ),
        "original_recovery_strategy": str(
            target.get("recovery_strategy") or ""
        ),
        "original_recovery_success": str(target.get("recovery_success") or ""),
        **{
            field: str(event.get(field) or "")
            for field in PROPOSAL_FIELDS
        },
        "reason_code": str(event.get("reason_code") or ""),
        "primary_evidence_reference": str(
            event.get("evidence_reference") or ""
        ),
        "primary_event_id": str(event.get("event_id") or ""),
        "confirmation_reason_code": str(
            (confirmation_event or {}).get("reason_code") or ""
        ),
        "confirmation_evidence_reference": str(
            (confirmation_event or {}).get("evidence_reference") or ""
        ),
        "confirmation_event_id": str(
            (confirmation_event or {}).get("event_id") or ""
        ),
    }


def reconcile_reviews(
    bbox_rows: Sequence[dict],
    weak_rows: Sequence[dict],
    events: Iterable[dict],
) -> dict:
    """Validate current review state and derive non-destructive outputs."""
    queue_targets = {
        "bbox": _decision_targets(bbox_rows, "bbox"),
        "weak": _decision_targets(weak_rows, "weak"),
    }
    event_rows = list(events)
    latest = _latest_event_map(event_rows)
    errors: list[str] = []

    for (queue_kind, reviewer_id, sample_id), event in latest.items():
        if queue_kind not in queue_targets:
            errors.append(f"unknown queue_kind in event {event['event_id']}")
            continue
        if reviewer_id not in REVIEWERS:
            errors.append(f"unknown reviewer_id in event {event['event_id']}")
        if sample_id not in queue_targets[queue_kind]:
            errors.append(
                f"{queue_kind}:{sample_id} is not a decision target"
            )
            continue
        target = queue_targets[queue_kind][sample_id]
        for field in ("split", "task_id", "step_index"):
            target_value = (
                ""
                if target.get(field) is None
                else str(target.get(field))
            )
            if str(event.get(field) or "") != target_value:
                errors.append(
                    f"event {event['event_id']} has {field} inconsistent "
                    "with its queue target"
                )

    missing_primary: list[dict] = []
    secondary_rows = {"bbox": [], "weak": []}
    pairs: list[dict] = []
    pair_keys: set[tuple[str, str]] = set()
    unresolved: list[dict] = []
    disagreements: list[dict] = []
    corrections: list[dict] = []
    dispositions: list[dict] = []

    for queue_kind, targets in queue_targets.items():
        for sample_id, target in targets.items():
            assignment = str(target["reviewer_assignment"])
            primary_ids = _primary_reviewers(assignment)
            primary_events = {
                reviewer_id: latest.get((queue_kind, reviewer_id, sample_id))
                for reviewer_id in primary_ids
            }
            for reviewer_id, event in primary_events.items():
                if event is None:
                    missing_primary.append({
                        "queue_kind": queue_kind,
                        "sample_id": sample_id,
                        "task_id": str(target.get("task_id") or ""),
                        "required_reviewer_id": reviewer_id,
                    })

            required_pair: tuple[str, str] | None = None
            if assignment == "A+B":
                required_pair = ("A", "B")
            else:
                primary_event = primary_events[assignment]
                if (
                    primary_event is not None
                    and primary_event["review_decision"] in RISKY_DECISIONS
                ):
                    second_id = _other_reviewer(assignment)
                    required_pair = (assignment, second_id)
                    if latest.get((queue_kind, second_id, sample_id)) is None:
                        secondary = dict(target)
                        secondary["reviewer_assignment"] = second_id
                        secondary["double_review"] = "True"
                        secondary["secondary_review_for_event_id"] = str(
                            primary_event["event_id"]
                        )
                        secondary["primary_reviewer_id"] = assignment
                        secondary_rows[queue_kind].append(secondary)

            consensus_event: dict | None = None
            confirmation_event: dict | None = None
            if required_pair:
                event_first = latest.get(
                    (queue_kind, required_pair[0], sample_id)
                )
                event_second = latest.get(
                    (queue_kind, required_pair[1], sample_id)
                )
                if event_first is not None and event_second is not None:
                    pair_key = (queue_kind, sample_id)
                    if pair_key not in pair_keys:
                        pair_keys.add(pair_key)
                        event_a = (
                            event_first
                            if event_first["reviewer_id"] == "A"
                            else event_second
                        )
                        event_b = (
                            event_first
                            if event_first["reviewer_id"] == "B"
                            else event_second
                        )
                        pairs.append({
                            "queue_kind": queue_kind,
                            "sample_id": sample_id,
                            "task_id": str(target.get("task_id") or ""),
                            "pair_reason": (
                                "assigned_overlap"
                                if assignment == "A+B"
                                else "risky_decision_confirmation"
                            ),
                            "event_a": event_a,
                            "event_b": event_b,
                        })
                    if _event_signature(event_first) == _event_signature(event_second):
                        consensus_event = event_first
                        confirmation_event = event_second
                    else:
                        disagreements.append({
                            "queue_kind": queue_kind,
                            "sample_id": sample_id,
                            "task_id": str(target.get("task_id") or ""),
                            "event_id_a": str(
                                latest[(queue_kind, "A", sample_id)]["event_id"]
                            ),
                            "event_id_b": str(
                                latest[(queue_kind, "B", sample_id)]["event_id"]
                            ),
                            "decision_a": str(
                                latest[(queue_kind, "A", sample_id)][
                                    "review_decision"
                                ]
                            ),
                            "decision_b": str(
                                latest[(queue_kind, "B", sample_id)][
                                    "review_decision"
                                ]
                            ),
                            "reason": "decision_or_proposed_values_disagree",
                        })
            else:
                consensus_event = primary_events[assignment]

            if consensus_event is None:
                continue
            decision = str(consensus_event["review_decision"])
            if decision == "needs_discussion":
                unresolved.append({
                    "queue_kind": queue_kind,
                    "sample_id": sample_id,
                    "task_id": str(target.get("task_id") or ""),
                    "reason": "reviewers_agree_that_adjudication_is_required",
                })
                continue
            if decision == "approve_after_correction":
                corrections.append(
                    _correction_row(
                        target,
                        consensus_event,
                        queue_kind=queue_kind,
                        confirmation_event=confirmation_event,
                    )
                )
            dispositions.append({
                "queue_kind": queue_kind,
                "split": str(target.get("split") or ""),
                "sample_id": sample_id,
                "task_id": str(target.get("task_id") or ""),
                "final_review_decision": decision,
                "primary_event_id": str(consensus_event["event_id"]),
                "confirmation_event_id": str(
                    (confirmation_event or {}).get("event_id") or ""
                ),
            })

    for (queue_kind, reviewer_id, sample_id), event in latest.items():
        target = queue_targets.get(queue_kind, {}).get(sample_id)
        if target is None:
            continue
        assignment = str(target["reviewer_assignment"])
        if reviewer_id not in _primary_reviewers(assignment):
            primary_id = _other_reviewer(reviewer_id)
            primary = latest.get((queue_kind, primary_id, sample_id))
            allowed = (
                primary is not None
                and primary["review_decision"] in RISKY_DECISIONS
            )
            if not allowed:
                errors.append(
                    f"event {event['event_id']} is not an assigned or required "
                    "secondary review"
                )

    status = "PASS"
    if (
        errors
        or missing_primary
        or any(secondary_rows.values())
        or disagreements
        or unresolved
    ):
        status = "INCOMPLETE"

    pair_rows = [
        {
            "queue_kind": pair["queue_kind"],
            "sample_id": pair["sample_id"],
            "task_id": pair["task_id"],
            "pair_reason": pair["pair_reason"],
            "event_id_a": str(pair["event_a"]["event_id"]),
            "event_id_b": str(pair["event_b"]["event_id"]),
            **{
                f"{field}_a": str(pair["event_a"].get(field) or "")
                for field in AGREEMENT_FIELDS
            },
            **{
                f"{field}_b": str(pair["event_b"].get(field) or "")
                for field in AGREEMENT_FIELDS
            },
        }
        for pair in pairs
    ]
    return {
        "report": {
            "status": status,
            "controlled_mini_permitted": status == "PASS",
            "publication_ready": False,
            "test_rows_read": 0,
            "source_records_mutated": False,
            "immutable_event_rows": len(event_rows),
            "decision_targets": {
                kind: len(targets) for kind, targets in queue_targets.items()
            },
            "missing_primary_reviews": len(missing_primary),
            "missing_secondary_reviews": sum(
                len(rows) for rows in secondary_rows.values()
            ),
            "paired_reviews": len(pairs),
            "disagreements": len(disagreements),
            "unresolved": len(unresolved),
            "approved_corrections": len(corrections),
            "finalized_dispositions": len(dispositions),
            "errors": errors,
            "agreement": _agreement_metrics(pairs),
        },
        "missing_primary": missing_primary,
        "secondary_rows": secondary_rows,
        "agreement_pairs": pair_rows,
        "disagreements": disagreements,
        "unresolved": unresolved,
        "corrections": corrections,
        "dispositions": dispositions,
    }
