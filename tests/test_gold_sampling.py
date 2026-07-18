from __future__ import annotations

import pytest

from web_agent.data.gold_sampling import (
    gold_joint_stratum,
    label_distribution,
    recovery_aware_batch_indices,
    select_recovery_aware_gold_subset,
    sqrt_inverse_frequency_weights,
)


def _record(index: int, *, attempted: bool, strategy: str, action: str) -> dict:
    return {
        "inputs": {"state_before": f"before-{index}.png"},
        "labels": {
            "failure_type_4": "ACTION_MISMATCH" if attempted else "NONE",
            "action_type": action,
            "recovery_strategy": strategy,
            "recovery_success": (index % 2 == 0) if attempted else None,
            "memory_update_flag": attempted,
            "action_target_bbox": (
                {"x": 0, "y": 0, "width": 1, "height": 1}
                if index % 3 else None
            ),
        },
        "meta": {"sample_id": f"sample-{index}"},
    }


def _records() -> list[dict]:
    rows = []
    for index in range(80):
        attempted = index % 4 == 0
        strategy = "RETRY" if attempted and index % 8 == 0 else "REPLAN" if attempted else "NONE"
        action = ("CLICK", "TYPE", "SELECT", "SCROLL")[index % 4]
        rows.append(_record(index, attempted=attempted, strategy=strategy, action=action))
    return rows


def test_joint_subset_is_exact_unique_deterministic_and_covers_strata():
    records = _records()
    first = select_recovery_aware_gold_subset(records, 32, seed=42)
    second = select_recovery_aware_gold_subset(records, 32, seed=42)

    assert len(first) == 32
    assert len({id(record) for record in first}) == 32
    assert [row["meta"]["sample_id"] for row in first] == [
        row["meta"]["sample_id"] for row in second
    ]
    assert set(map(gold_joint_stratum, records)).issubset(set(map(gold_joint_stratum, first)))


def test_recovery_batches_preserve_every_row_and_spread_attempts():
    records = _records()
    batches = recovery_aware_batch_indices(records, 4, seed=42, epoch=0)
    flat = [index for batch in batches for index in batch]

    assert len(batches) == 20
    assert sorted(flat) == list(range(80))
    assert len(flat) == len(set(flat))
    attempted_batches = sum(
        any(records[index]["labels"]["recovery_success"] is not None for index in batch)
        for batch in batches
    )
    assert attempted_batches == 20
    assert recovery_aware_batch_indices(records, 4, seed=42, epoch=1) != batches


def test_sqrt_weights_are_finite_normalized_capped_and_zero_when_absent():
    labels = [0] * 60 + [1] * 12 + [2] * 3
    weights = sqrt_inverse_frequency_weights(labels, 6, cap=2.5)

    assert weights[0] < weights[1] < weights[2]
    assert max(weights) <= 2.5
    assert sum(weights[:3]) / 3 == pytest.approx(1.0)
    assert weights[3:] == [0.0, 0.0, 0.0]


def test_distribution_reports_recovery_attempt_denominator():
    distribution = label_distribution(_records())
    assert distribution["recovery_attempted"] == {"False": 60, "True": 20}
    assert distribution["recovery_success"] == {
        "NOT_ATTEMPTED": 60,
        "True": 20,
    }
    assert distribution["recovery_strategy"] == {
        "NONE": 60,
        "REPLAN": 10,
        "RETRY": 10,
    }
