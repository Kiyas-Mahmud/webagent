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
    EXECUTION_OUTCOME,
    FAILURE_TYPE,
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
    NUM_OUTCOME,
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


def _sklearn_balanced(labels, num_classes: int) -> torch.Tensor:
    """sklearn compute_class_weight('balanced') over present classes; absent -> 0."""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    present = sorted(set(labels))
    w = compute_class_weight("balanced", classes=np.array(present), y=np.array(labels))
    out = torch.zeros(num_classes, dtype=torch.float32)
    for c, wc in zip(present, w):
        out[c] = float(wc)
    return out


def _outcome_weights(labels, num_classes: int, scheme: str, cap: float) -> torch.Tensor:
    """Outcome (binary) weights under a chosen scheme.

    Full sklearn 'balanced' over-swings the binary head (2.56x toward the minority)
    and flips the collapse rather than fixing it. "capped" clamps the minority:
    majority weight=1, minority weight=min(balanced_ratio, cap). "sqrt" uses
    sqrt-inverse-frequency. "balanced" = the raw sklearn weights (ablation). P1-B.
    """
    bal = _sklearn_balanced(labels, num_classes)            # mean ~1, absent -> 0
    if scheme == "balanced":
        return bal
    counts = Counter(labels)
    present = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    if scheme == "sqrt":
        w = torch.zeros(num_classes, dtype=torch.float32)
        inv = {c: 1.0 / (counts[c] ** 0.5) for c in present}
        mean = sum(inv.values()) / len(inv)
        for c in present:
            w[c] = inv[c] / mean
        return w
    if scheme == "capped":
        w = torch.ones(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            if counts.get(c, 0) == 0:
                w[c] = 0.0
        # heavier (minority) class gets up to `cap`; majority stays 1.
        nz = [bal[c].item() for c in present]
        lo = min(nz)
        for c in present:
            w[c] = min(bal[c].item() / lo, cap)
        return w
    raise ValueError(f"unknown outcome weight scheme {scheme!r}")


def balanced_class_weights(records, limit: int | None = None,
                           outcome_scheme: str = "capped", outcome_cap: float = 1.5):
    """Class weights for action_type[5], failure_type[4], outcome[2].

    action/failure_type use sklearn 'balanced'. outcome uses `outcome_scheme`
    ("capped" default, "sqrt", or "balanced") to avoid the over-swing that flips
    the binary collapse. Returns (action_w, failtype_w, outcome_w).
    """
    rows = records[:limit] if limit else records
    action = [ACTION_TYPE[r["action_type"]] for r in rows]
    failtype = [FAILURE_TYPE[r["failure_type"]] for r in rows]
    outcome = [EXECUTION_OUTCOME[r["execution_outcome"]] for r in rows]
    return (
        _sklearn_balanced(action, NUM_ACTION_TYPE),
        _sklearn_balanced(failtype, NUM_FAILURE_TYPE),
        _outcome_weights(outcome, NUM_OUTCOME, outcome_scheme, outcome_cap),
    )


def binary_pos_weight(records, field: str, limit: int | None = None,
                      cap: float = 3.0) -> torch.Tensor:
    """pos_weight (neg/pos, clamped to `cap`) for a BCE head, e.g. recovery_success."""
    rows = records[:limit] if limit else records
    pos = sum(1 for r in rows if bool(r.get(field, False)))
    neg = len(rows) - pos
    ratio = (neg / pos) if pos > 0 else 1.0
    return torch.tensor([min(ratio, cap)], dtype=torch.float32)
