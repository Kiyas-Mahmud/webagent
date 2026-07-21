"""Deterministic checkpoint selection for controlled Gold experiments.

The trainer records every epoch independently.  This module decides which one
is allowed to represent an experiment without loading a model or reading data,
so the same rule can be unit-tested and replayed on an existing Kaggle report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


LEGACY_OUTCOME_RULE = "outcome_mcc"
ALL_GATES_THEN_OUTCOME_RULE = "all_gates_then_outcome_mcc"
SUPPORTED_SELECTION_RULES = {
    LEGACY_OUTCOME_RULE,
    ALL_GATES_THEN_OUTCOME_RULE,
}

V14_SELECTED_BASELINE = {
    "epoch": 3,
    "outcome_mcc": 0.5216751841203641,
    "action_acc": 0.396,
    "recovery_outcome_mcc": 0.18641092980036,
    "bbox_mae": 0.1862636590092959,
    "outcome_ece": 0.18528690659999844,
}


def _modern_quality_checks(epoch: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate the eight recovery-v2 quality gates for one epoch."""
    return {
        "outcome_mcc_within_0_03_of_v14": (
            float(epoch["outcome_mcc"])
            >= V14_SELECTED_BASELINE["outcome_mcc"] - 0.03
        ),
        "action_accuracy_not_down_more_than_0_03": (
            float(epoch["action_acc"])
            >= V14_SELECTED_BASELINE["action_acc"] - 0.03
        ),
        "needs_recovery_macro_f1_beats_majority_by_0_03": (
            float(epoch["needs_recovery_macro_f1"])
            >= float(epoch["needs_recovery_majority_macro_f1"]) + 0.03
        ),
        "attempted_strategy_macro_f1_beats_majority_by_0_03": (
            float(epoch["strategy_attempted_macro_f1"])
            >= float(epoch["strategy_attempted_majority_macro_f1"]) + 0.03
        ),
        "transition_recovery_outcome_mcc_improves_v14_by_0_01": (
            float(epoch["recovery_outcome_mcc"])
            >= V14_SELECTED_BASELINE["recovery_outcome_mcc"] + 0.01
        ),
        "bbox_mean_iou_at_least_0_05": float(epoch["bbox_mean_iou"]) >= 0.05,
        "bbox_recall_iou50_at_least_0_01": (
            float(epoch["bbox_recall_iou50"]) >= 0.01
        ),
        "outcome_ece_not_up_more_than_0_03": (
            float(epoch["outcome_ece"])
            <= V14_SELECTED_BASELINE["outcome_ece"] + 0.03
        ),
    }


def _legacy_quality_checks(epoch: Mapping[str, Any]) -> dict[str, bool]:
    """Retain the pre-hierarchical recovery gates for old reports."""
    return {
        "outcome_mcc_within_0_02_of_v14": (
            float(epoch["outcome_mcc"])
            >= V14_SELECTED_BASELINE["outcome_mcc"] - 0.02
        ),
        "recovery_macro_f1_beats_majority_by_0_03": (
            float(epoch["recovery_macro_f1"])
            >= float(epoch["recovery_majority_macro_f1"]) + 0.03
        ),
        "recovery_predicts_at_least_3_classes": (
            float(epoch["recovery_pred_classes"]) >= 3
        ),
        "recovery_outcome_mcc_improves_v14_by_0_01": (
            float(epoch["recovery_outcome_mcc"])
            >= V14_SELECTED_BASELINE["recovery_outcome_mcc"] + 0.01
        ),
        "action_accuracy_not_down_more_than_0_02": (
            float(epoch["action_acc"])
            >= V14_SELECTED_BASELINE["action_acc"] - 0.02
        ),
        "bbox_mae_not_up_more_than_0_02": (
            float(epoch["bbox_mae"])
            <= V14_SELECTED_BASELINE["bbox_mae"] + 0.02
        ),
        "outcome_ece_not_up_more_than_0_02": (
            float(epoch["outcome_ece"])
            <= V14_SELECTED_BASELINE["outcome_ece"] + 0.02
        ),
    }


def quality_checks(epoch: Mapping[str, Any]) -> dict[str, bool]:
    """Return the registered gate set matching an epoch's report schema."""
    if "needs_recovery_macro_f1" in epoch:
        return _modern_quality_checks(epoch)
    return _legacy_quality_checks(epoch)


def _validated_history(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = result.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("training result has no epoch history")
    rows = list(history)
    if not rows:
        raise ValueError("training result has empty epoch history")
    epochs = [int(row["epoch"]) for row in rows]
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"epoch history contains duplicates: {epochs}")
    return rows


def _unconstrained_outcome_epoch(
    result: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
) -> int:
    recorded = result.get("best_epochs", {}).get("outcome_mcc", {}).get("epoch")
    computed = int(max(history, key=lambda row: float(row["outcome_mcc"]))["epoch"])
    if recorded is not None and int(recorded) != computed:
        raise ValueError(
            "recorded outcome-MCC epoch disagrees with history: "
            f"recorded={recorded}, computed={computed}"
        )
    return computed


def controlled_quality_gates(
    result: Mapping[str, Any],
    selection_rule: str = LEGACY_OUTCOME_RULE,
) -> dict[str, Any]:
    """Evaluate quality and select one deterministic representative epoch.

    ``all_gates_then_outcome_mcc`` treats the gates as eligibility constraints,
    then maximizes the registered primary metric.  Ties select the earlier epoch.
    When no epoch is eligible, the unconstrained outcome epoch is returned only
    as a diagnostic fallback and the status remains ``FAIL``.
    """
    if selection_rule not in SUPPORTED_SELECTION_RULES:
        raise ValueError(
            f"unsupported checkpoint selection rule {selection_rule!r}; "
            f"expected one of {sorted(SUPPORTED_SELECTION_RULES)}"
        )

    history = _validated_history(result)
    evaluations = []
    for epoch in history:
        checks = quality_checks(epoch)
        evaluations.append(
            {
                "epoch": int(epoch["epoch"]),
                "outcome_mcc": float(epoch["outcome_mcc"]),
                "eligible": all(checks.values()),
                "checks": checks,
            }
        )

    unconstrained_epoch = _unconstrained_outcome_epoch(result, history)
    eligible = [row for row in evaluations if row["eligible"]]
    if selection_rule == ALL_GATES_THEN_OUTCOME_RULE and eligible:
        selected_eval = max(
            eligible,
            key=lambda row: (row["outcome_mcc"], -row["epoch"]),
        )
        selection_reason = "highest outcome_mcc among all-gate-eligible epochs"
    else:
        selected_eval = next(
            row for row in evaluations if row["epoch"] == unconstrained_epoch
        )
        selection_reason = (
            "legacy unconstrained outcome_mcc selection"
            if selection_rule == LEGACY_OUTCOME_RULE
            else "no eligible epoch; unconstrained outcome epoch retained for diagnostics"
        )

    selected_epoch = int(selected_eval["epoch"])
    epoch_checkpoints = result.get("epoch_checkpoints", {})
    selected_checkpoint = (
        epoch_checkpoints.get(str(selected_epoch))
        if isinstance(epoch_checkpoints, Mapping)
        else None
    )
    status = "PASS" if selected_eval["eligible"] else "FAIL"
    return {
        "status": status,
        "selection_rule": selection_rule,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": selected_checkpoint,
        "selected_outcome_mcc": float(selected_eval["outcome_mcc"]),
        "selected_epoch_is_eligible": bool(selected_eval["eligible"]),
        "eligible_epochs": [int(row["epoch"]) for row in eligible],
        "unconstrained_outcome_epoch": unconstrained_epoch,
        "selection_reason": selection_reason,
        "comparison_rule": (
            "all quality gates define eligibility; outcome_mcc ranks eligible epochs"
            if selection_rule == ALL_GATES_THEN_OUTCOME_RULE
            else "all gates use the outcome_mcc-selected checkpoint"
        ),
        "tie_breaker": "earlier epoch",
        "v14_selected_checkpoint_reference": V14_SELECTED_BASELINE,
        "checks": selected_eval["checks"],
        "per_epoch": evaluations,
        "limitations": [
            "recovery-outcome semantics use proper causal transitions",
            "bbox IoU thresholds are minimum functionality gates, not 90-percent claims",
            "full training remains separately gated from this mini experiment",
        ],
    }


def require_selected_checkpoint(
    result: Mapping[str, Any], quality_report: Mapping[str, Any],
) -> str:
    """Return the selected epoch checkpoint or reject an inconsistent report."""
    selected_epoch = int(quality_report["selected_epoch"])
    checkpoints = result.get("epoch_checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("training result has no per-epoch checkpoint mapping")
    checkpoint = checkpoints.get(str(selected_epoch))
    if not checkpoint:
        raise ValueError(
            f"selected epoch {selected_epoch} has no retained checkpoint; "
            f"available epochs are {sorted(checkpoints)}"
        )
    reported = quality_report.get("selected_checkpoint")
    if reported is not None and str(reported) != str(checkpoint):
        raise ValueError(
            "quality report checkpoint disagrees with epoch checkpoint mapping: "
            f"{reported!r} != {checkpoint!r}"
        )
    return str(checkpoint)


def replay_selection_report(
    source_report: Mapping[str, Any],
    *,
    experiment_tag: str = "recovery_v2_7",
) -> dict[str, Any]:
    """Apply the v2.7 selector to an immutable completed-run report.

    This changes report selection metadata only.  It never claims that training
    was rerun or that the newly selected checkpoint was prediction-roundtrip
    tested during replay.
    """
    if int(source_report.get("test_rows_read", -1)) != 0:
        raise ValueError("selection replay requires a report that kept test data locked")

    quality_report = controlled_quality_gates(
        source_report, ALL_GATES_THEN_OUTCOME_RULE
    )
    selected_checkpoint = require_selected_checkpoint(source_report, quality_report)
    selected_epoch = int(quality_report["selected_epoch"])
    selected_history = next(
        row
        for row in source_report["history"]
        if int(row["epoch"]) == selected_epoch
    )
    primary_metric = str(source_report.get("early_stop_metric", "outcome_mcc"))

    replay = deepcopy(dict(source_report))
    source_experiment_control = deepcopy(
        dict(source_report.get("experiment_control", {}))
    )
    source_tag = source_experiment_control.get("experiment_tag", "unknown")
    observational_changes = list(
        source_experiment_control.get("observational_changes", [])
    )
    for change in (
        "all-gates eligibility before outcome-MCC checkpoint ranking",
        "selected-checkpoint identity synchronized across report and exports",
    ):
        if change not in observational_changes:
            observational_changes.append(change)
    source_experiment_control.update(
        {
            "experiment_tag": experiment_tag,
            "quality_selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
            "selection_replay_source_tag": source_tag,
            "selection_replay_changes_model_weights": False,
            "observational_changes": observational_changes,
        }
    )

    replay.update(
        {
            "best_metric": float(selected_history[primary_metric]),
            "best_checkpoint": selected_checkpoint,
            "selected_epoch": selected_epoch,
            "selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
            "unconstrained_best_metric": float(source_report["best_metric"]),
            "unconstrained_best_checkpoint": source_report.get(
                "best_checkpoint", source_report.get("checkpoints", [""])[0]
            ),
            "quality_gates": quality_report,
            "experiment_control": source_experiment_control,
            "selection_replay": {
                "status": quality_report["status"],
                "source_experiment_tag": source_tag,
                "source_checkpoint_roundtrip": source_report.get(
                    "checkpoint_roundtrip"
                ),
                "selected_checkpoint_roundtrip_rerun": False,
                "model_weights_changed": False,
                "test_rows_read": 0,
            },
        }
    )
    return replay
