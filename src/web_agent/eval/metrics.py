"""Metric functions (SPEC 8 / IMPLEMENTATION_PLAN 10.1).

Primary: Failure Detection F1 (binary), Failure-Type Acc/macro-F1 (4-way),
Action-Type Acc (5-way), Recovery Success Rate.
Secondary: per-failure-type F1, confidence calibration gap, memory retrieval acc.
"""

from __future__ import annotations


def failure_detection_f1(y_true, y_pred) -> float:
    raise NotImplementedError("Build in Phase 5/10.")


def failure_type_accuracy(y_true, y_pred) -> float:
    raise NotImplementedError


def action_type_accuracy(y_true, y_pred) -> float:
    raise NotImplementedError


def recovery_success_rate(y_true, y_pred) -> float:
    raise NotImplementedError


def confidence_calibration_gap(confidences, outcomes) -> float:
    """SUCCESS avg confidence minus FAILURE avg (target ~0.42)."""
    raise NotImplementedError
