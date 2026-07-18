"""Metric functions (spec). Operate on accumulated numpy/torch arrays.

Primary = Failure Detection F1 (binary on execution_outcome, positive=FAILURE=1):
this drives checkpoint selection and early stopping. The rest are reported each epoch.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)


def failure_detection_f1(y_true, y_pred) -> float:
    """Binary F1 with FAILURE (=1) as the positive class."""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def failure_macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over SUCCESS+FAILURE — collapses to ~0.5 if the model predicts one class."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def outcome_balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def outcome_mcc(y_true, y_pred) -> float:
    """Matthews corr coef — 0 for majority-class collapse, 1 for perfect."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if len(set(yt)) < 2 or len(set(yp)) < 2:
        return 0.0
    return float(matthews_corrcoef(yt, yp))


def classification_mcc(y_true, y_pred) -> float:
    """Binary or multiclass MCC; zero when either side has only one class."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if len(yt) == 0 or len(set(yt)) < 2 or len(set(yp)) < 2:
        return 0.0
    return float(matthews_corrcoef(yt, yp))


def success_recall(y_true, y_pred) -> float:
    """Recall on the SUCCESS class (=0); ~0 means the model never predicts SUCCESS."""
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 for any categorical head, including rare classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def balanced_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(balanced_accuracy_score(y_true, y_pred))


def brier_score(y_true, positive_probability) -> float:
    """Mean squared probability error for a binary positive-class target."""
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(positive_probability, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((probability - y_true) ** 2))


def majority_baseline(y_true) -> dict[str, float | int]:
    """Accuracy/macro-F1 of predicting the most common observed class."""
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return {"class": -1, "accuracy": 0.0, "macro_f1": 0.0}
    labels, counts = np.unique(y_true, return_counts=True)
    majority = int(labels[int(np.argmax(counts))])
    pred = np.full_like(y_true, majority)
    return {
        "class": majority,
        "accuracy": accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred),
    }


def mean_absolute_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).mean()) if len(y_true) else 0.0


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """True hard-binned ECE (a metric, not the training loss)."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(confidences[m].mean() - correct[m].mean())
    return float(ece)


def bbox_mae(pred_boxes, true_boxes, mask) -> float:
    """Mean absolute error over rows that have a bbox (mask==1)."""
    pred = np.asarray(pred_boxes, dtype=float)
    true = np.asarray(true_boxes, dtype=float)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    sel = mask > 0.5
    if sel.sum() == 0:
        return 0.0
    return float(np.abs(pred[sel] - true[sel]).mean())


def bbox_iou_values(pred_boxes, true_boxes, mask) -> np.ndarray:
    """IoU for normalized ``[x, y, width, height]`` boxes with a valid mask."""
    pred = np.asarray(pred_boxes, dtype=float)
    true = np.asarray(true_boxes, dtype=float)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    sel = mask > 0.5
    if sel.sum() == 0:
        return np.array([], dtype=float)
    pred = pred[sel]
    true = true[sel]

    pred_x2 = pred[:, 0] + np.maximum(pred[:, 2], 0.0)
    pred_y2 = pred[:, 1] + np.maximum(pred[:, 3], 0.0)
    true_x2 = true[:, 0] + np.maximum(true[:, 2], 0.0)
    true_y2 = true[:, 1] + np.maximum(true[:, 3], 0.0)
    inter_w = np.maximum(0.0, np.minimum(pred_x2, true_x2) - np.maximum(pred[:, 0], true[:, 0]))
    inter_h = np.maximum(0.0, np.minimum(pred_y2, true_y2) - np.maximum(pred[:, 1], true[:, 1]))
    intersection = inter_w * inter_h
    pred_area = np.maximum(pred[:, 2], 0.0) * np.maximum(pred[:, 3], 0.0)
    true_area = np.maximum(true[:, 2], 0.0) * np.maximum(true[:, 3], 0.0)
    union = pred_area + true_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def bbox_iou_summary(pred_boxes, true_boxes, mask) -> dict[str, float | int]:
    values = bbox_iou_values(pred_boxes, true_boxes, mask)
    if len(values) == 0:
        return {"rows": 0, "mean": 0.0, "median": 0.0, "recall_50": 0.0}
    return {
        "rows": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "recall_50": float((values >= 0.5).mean()),
    }


def classification_diagnostics(
    y_true,
    y_pred,
    *,
    label_names: list[str],
) -> dict:
    """JSON-ready distributions, confusion matrix, and per-class statistics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    true_counts = np.bincount(y_true, minlength=len(labels)) if len(y_true) else np.zeros(len(labels), int)
    pred_counts = np.bincount(y_pred, minlength=len(labels)) if len(y_pred) else np.zeros(len(labels), int)
    return {
        "rows": int(len(y_true)),
        "labels": label_names,
        "true_distribution": {
            name: int(true_counts[index]) for index, name in enumerate(label_names)
        },
        "predicted_distribution": {
            name: int(pred_counts[index]) for index, name in enumerate(label_names)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(label_names)
        },
        "majority_baseline": majority_baseline(y_true),
    }
