from __future__ import annotations

import numpy as np
import pytest

from web_agent.eval.metrics import (
    bbox_iou_summary,
    brier_score,
    classification_diagnostics,
    classification_mcc,
    majority_baseline,
)


def test_majority_collapse_is_visible_in_macro_metrics_and_diagnostics():
    true = np.array([0, 0, 0, 1, 2, 2])
    pred = np.zeros_like(true)
    diagnostics = classification_diagnostics(
        true,
        pred,
        label_names=["NONE", "RETRY", "REPLAN"],
    )

    assert diagnostics["majority_baseline"]["accuracy"] == pytest.approx(0.5)
    assert classification_mcc(true, pred) == 0.0
    assert diagnostics["predicted_distribution"] == {
        "NONE": 6,
        "RETRY": 0,
        "REPLAN": 0,
    }
    assert diagnostics["per_class"]["RETRY"]["recall"] == 0.0
    assert diagnostics["confusion_matrix"] == [[3, 0, 0], [1, 0, 0], [2, 0, 0]]


def test_bbox_iou_uses_xywh_and_respects_mask():
    prediction = np.array([
        [0.1, 0.1, 0.2, 0.2],
        [0.0, 0.0, 0.2, 0.2],
        [0.0, 0.0, 1.0, 1.0],
    ])
    target = np.array([
        [0.1, 0.1, 0.2, 0.2],
        [0.1, 0.0, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.0],
    ])
    summary = bbox_iou_summary(prediction, target, np.array([1, 1, 0]))

    assert summary["rows"] == 2
    assert summary["mean"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert summary["median"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert summary["recall_50"] == 0.5


def test_binary_probability_metrics_are_explicit():
    true = np.array([0, 1])
    probability = np.array([0.2, 0.8])
    assert brier_score(true, probability) == pytest.approx(0.04)
    assert majority_baseline([]) == {"class": -1, "accuracy": 0.0, "macro_f1": 0.0}
