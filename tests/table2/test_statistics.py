from __future__ import annotations

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.statistics import (
    cluster_bootstrap,
    compute_paired_contrasts,
)


def test_cluster_bootstrap_is_deterministic_and_task_clustered():
    rows = [
        {"task_id": "a", "value": 0.0},
        {"task_id": "a", "value": 1.0},
        {"task_id": "b", "value": 1.0},
    ]
    first = cluster_bootstrap(rows, value=lambda row: row["value"], samples=200, seed=7)
    second = cluster_bootstrap(rows, value=lambda row: row["value"], samples=200, seed=7)
    assert first == second
    assert first["n_clusters"] == 2
    assert first["n_rows"] == 3
    assert first["estimate"] == 2.0 / 3.0


def test_registered_paired_contrasts_include_e2_e3_memory_regressions():
    rows = []
    outcomes = {
        "task-a": {"E0": False, "E1": True, "E2": True, "E3": False},
        "task-b": {"E0": False, "E1": False, "E2": True, "E3": True},
    }
    for task_id, systems in outcomes.items():
        for system_id, success in systems.items():
            rows.append({
                "block_id": task_id,
                "task_id": task_id,
                "system_id": system_id,
                "task_success": success,
                "step_count": 2,
                "loop_detected": False,
            })

    result = compute_paired_contrasts(rows, bootstrap_samples=100, seed=3)
    assert set(result["contrasts"]) == {
        "E1_minus_E0",
        "E2_minus_E1",
        "E3_minus_E2",
        "E3_minus_E0",
    }
    total = result["contrasts"]["E3_minus_E0"]["task_success"]
    assert total["analysis_role"] == "descriptive_total_system_contrast"
    total_significance = total["task_clustered_significance"]
    assert total_significance["inference_role"] == "descriptive_unadjusted_only"
    assert "holm_adjusted_p" not in total_significance
    assert result["multiple_testing"]["contrast_names"] == [
        "E1_minus_E0",
        "E2_minus_E1",
        "E3_minus_E2",
    ]
    assert result["multiple_testing"]["hypothesis_count"] == 3
    regression = result["paired_memory_regression_rate"]
    assert regression["numerator"] == 1
    assert regression["denominator"] == 2
    assert regression["estimate"] == 0.5
    assert regression["unit"] == "task_id"


def test_memory_regression_rate_counts_tasks_not_seed_repeat_blocks():
    rows = []
    for task_id, seed_outcomes in {
        "task-a": {42: (True, False), 43: (True, True), 44: (True, True)},
        "task-b": {42: (True, True)},
    }.items():
        for matched_seed, (e2_success, e3_success) in seed_outcomes.items():
            block_id = f"{task_id}-seed-{matched_seed}-repeat-0"
            for system_id in ("E0", "E1", "E2", "E3"):
                success = (
                    e2_success
                    if system_id == "E2"
                    else e3_success
                    if system_id == "E3"
                    else False
                )
                rows.append(
                    {
                        "block_id": block_id,
                        "task_id": task_id,
                        "system_id": system_id,
                        "matched_model_seed": matched_seed,
                        "repeat_id": 0,
                        "task_success": success,
                        "step_count": 1,
                        "loop_detected": False,
                    }
                )

    result = compute_paired_contrasts(rows, bootstrap_samples=20, seed=9)
    regression = result["paired_memory_regression_rate"]
    assert regression["numerator"] == 1
    assert regression["denominator"] == 2
    assert regression["estimate"] == 0.5
    assert regression["unique_seed_repeat_cells"] == 4


def test_task_clustered_significance_is_invariant_to_duplicate_seed_repeat_rows():
    rows = []
    for task_index in range(12):
        task_id = f"task-{task_index:02d}"
        for matched_seed in (42, 43):
            if task_index < 6 or matched_seed == 42:
                e0_success, e1_success = False, True
            else:
                e0_success, e1_success = True, False
            outcomes = {
                "E0": e0_success,
                "E1": e1_success,
                "E2": e1_success,
                "E3": e1_success,
            }
            block_id = f"{task_id}-seed-{matched_seed}-repeat-0"
            for system_id, success in outcomes.items():
                rows.append(
                    {
                        "block_id": block_id,
                        "task_id": task_id,
                        "system_id": system_id,
                        "matched_model_seed": matched_seed,
                        "repeat_id": 0,
                        "task_success": success,
                        "step_count": 2,
                        "loop_detected": False,
                    }
                )

    original = compute_paired_contrasts(rows, bootstrap_samples=100, seed=13)
    duplicated = list(rows)
    duplicated.extend(
        {
            **row,
            "block_id": f"{row['block_id']}-accidental-duplicate",
        }
        for row in rows
        if int(row["matched_model_seed"]) == 42
        and int(str(row["task_id"]).split("-")[-1]) >= 6
    )
    repeated = compute_paired_contrasts(
        duplicated,
        bootstrap_samples=100,
        seed=13,
    )

    original_success = original["contrasts"]["E1_minus_E0"]["task_success"]
    repeated_success = repeated["contrasts"]["E1_minus_E0"]["task_success"]
    original_test = original_success["task_clustered_significance"]
    repeated_test = repeated_success["task_clustered_significance"]

    # Six tasks favor E1 and six have one win in each direction. Duplicating
    # only the favorable seed row from the tied tasks would create twelve
    # apparently favorable tasks if repeated rows were treated as evidence.
    assert original_test["right_better_task_clusters"] == 6
    assert original_test["tied_task_clusters"] == 6
    assert original_test["exact_sign_two_sided_p"] == 0.03125
    assert repeated_test["right_better_task_clusters"] == 6
    assert repeated_test["tied_task_clusters"] == 6
    assert repeated_test["duplicate_seed_repeat_cells_collapsed"] == 6
    assert repeated_test["exact_sign_two_sided_p"] == original_test[
        "exact_sign_two_sided_p"
    ]
    assert repeated_test["holm_adjusted_p"] == original_test["holm_adjusted_p"]
    assert original["multiple_testing"]["hypothesis_count"] == 3
    assert repeated["multiple_testing"]["hypothesis_count"] == 3

    # The paired-block diagnostic count does grow, proving that inferential
    # invariance comes from clustering/deduplication rather than the fixture.
    assert repeated_success["discordant"]["E1_only"] > original_success[
        "discordant"
    ]["E1_only"]
    assert repeated_success["discordant"]["inference_role"] == "descriptive_only"


def test_conflicting_duplicate_seed_repeat_success_fails_closed():
    rows = []
    for system_id in ("E0", "E1", "E2", "E3"):
        rows.append(
            {
                "block_id": "original",
                "task_id": "task-a",
                "system_id": system_id,
                "matched_model_seed": 42,
                "repeat_id": 0,
                "task_success": system_id != "E0",
                "step_count": 2,
                "loop_detected": False,
            }
        )
        rows.append(
            {
                "block_id": "conflicting-copy",
                "task_id": "task-a",
                "system_id": system_id,
                "matched_model_seed": 42,
                "repeat_id": 0,
                "task_success": False,
                "step_count": 2,
                "loop_detected": False,
            }
        )

    with pytest.raises(SchemaError, match="conflicting task-success outcomes"):
        compute_paired_contrasts(rows, bootstrap_samples=20, seed=5)
