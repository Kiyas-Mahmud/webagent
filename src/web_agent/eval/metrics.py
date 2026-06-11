"""Metric functions (spec). Operate on accumulated numpy/torch arrays.

Primary = Failure Detection F1 (binary on execution_outcome, positive=FAILURE=1):
this drives checkpoint selection and early stopping. The rest are reported each epoch.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def failure_detection_f1(y_true, y_pred) -> float:
    """Binary F1 with FAILURE (=1) as the positive class."""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


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
