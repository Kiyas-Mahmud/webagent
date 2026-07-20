"""Small, dependency-free result exporters for training runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def save_mini_result_csv(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write one validation/training result row per completed mini epoch.

    The trainer's JSON report remains the complete machine-readable artifact. This
    CSV is the compact table used for Kaggle download, manual review, and plots.
    """
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("mini report has no epoch history to export")

    selection_metric = report.get("early_stop_metric")
    if not isinstance(selection_metric, str) or not selection_metric:
        raise ValueError("mini report does not identify its selection metric")
    if any(selection_metric not in epoch for epoch in history):
        raise ValueError(f"epoch history is missing selection metric: {selection_metric}")

    best_metric = float(report["best_metric"])
    context = {
        "stage": "mini",
        "status": report.get("status", ""),
        "train_rows": report.get("train_rows", ""),
        "val_rows": report.get("val_rows", ""),
        "selection_metric": selection_metric,
        "best_metric": best_metric,
        "best_checkpoint": report.get("best_checkpoint", ""),
        "checkpoint_roundtrip": report.get("checkpoint_roundtrip", ""),
        "loss_decreased": report.get("loss_decreased", ""),
    }

    rows: list[dict[str, Any]] = []
    metric_fields: list[str] = []
    for epoch_metrics in history:
        if not isinstance(epoch_metrics, Mapping):
            raise ValueError("each mini epoch result must be a mapping")
        for key in epoch_metrics:
            if key not in metric_fields:
                metric_fields.append(key)
        metric_value = float(epoch_metrics[selection_metric])
        rows.append(
            {
                **context,
                **epoch_metrics,
                "is_best": abs(metric_value - best_metric) <= 1e-12,
            }
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*context, *metric_fields, "is_best"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def save_mini_diagnostics_json(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write the detailed, non-tabular evidence that does not belong in CSV."""
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("mini report has no detailed diagnostics to export")
    payload = {
        "stage": "mini",
        "status": report.get("status", ""),
        "train_rows": report.get("train_rows", ""),
        "val_rows": report.get("val_rows", ""),
        "test_rows_read": report.get("test_rows_read", ""),
        "selection_metric": report.get("early_stop_metric", ""),
        "best_metric": report.get("best_metric", ""),
        "best_checkpoint": report.get("best_checkpoint", ""),
        "best_epochs": report.get("best_epochs", {}),
        "epoch_checkpoints": report.get("epoch_checkpoints", {}),
        "class_weights": report.get("class_weights", {}),
        "sampling": report.get("sampling", {}),
        "quality_gates": report.get("quality_gates", {}),
        "train_distribution": report.get("train_distribution", {}),
        "validation_distribution": report.get("validation_distribution", {}),
        "experiment_control": report.get("experiment_control", {}),
        "recovery_transition_reports": report.get("recovery_transition_reports", {}),
        "recovery_class_audit": report.get("recovery_class_audit", {}),
        "epochs": diagnostics,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
