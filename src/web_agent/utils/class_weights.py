"""Inverse-frequency class weights for the imbalanced heads (SPEC 6.2).

action_type is dominated by CLICK (83.6%); failure_type's LOOP is only 7.4%.
Without weighting, CrossEntropy just predicts the majority class. We weight each
class by total / (num_classes * count). Classes with zero examples (SCROLL,
NAVIGATE — absent in the source) get weight 0 so they never distort the loss.
"""

from __future__ import annotations

from collections import Counter

import torch

from web_agent.labels import (
    ACTION_TYPE,
    FAILURE_TYPE,
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
)


def _inverse_freq(counts: Counter, num_classes: int) -> torch.Tensor:
    total = sum(counts.values())
    w = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n = counts.get(c, 0)
        w[c] = total / (num_classes * n) if n > 0 else 0.0
    return w


def compute_class_weights(records, limit: int | None = None):
    """Return (action_weights[5], failtype_weights[4]) from label frequencies."""
    rows = records[:limit] if limit else records
    action_counts = Counter(ACTION_TYPE[r["action_type"]] for r in rows)
    failtype_counts = Counter(FAILURE_TYPE[r["failure_type"]] for r in rows)
    return (
        _inverse_freq(action_counts, NUM_ACTION_TYPE),
        _inverse_freq(failtype_counts, NUM_FAILURE_TYPE),
    )
