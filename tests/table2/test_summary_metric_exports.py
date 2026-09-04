from __future__ import annotations

import csv

from web_agent.eval.table2.metrics import rate
from web_agent.eval.table2.package_validator import FINAL_READY_STATUS
from web_agent.eval.table2.schedule import SYSTEM_IDS
from web_agent.eval.table2.statistics import (
    compute_clustered_ratio_contrasts,
    compute_paired_contrasts,
)
from web_agent.eval.table2.summary import (
    PAIRED_CONTRAST_FIELDS,
    RETRIEVAL_METRIC_FIELDS,
    _format_rate,
    _paired_contrast_rows,
    _retrieval_metric_rows,
    _write_main_table,
    _write_metric_csv,
    _write_result_csv,
)


def _clustered_rate(
    numerator: int,
    denominator: int,
    *,
    low: float = 0.1,
    high: float = 0.9,
) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": None if denominator == 0 else numerator / denominator,
        "ci95_low": None if denominator == 0 else low,
        "ci95_high": None if denominator == 0 else high,
        "ci_low": None if denominator == 0 else low,
        "ci_high": None if denominator == 0 else high,
        "confidence": 0.95,
        "interval_method": "percentile_task_cluster_bootstrap",
        "n_task_clusters": 2,
        "display": "N/A" if denominator == 0 else "legacy display",
    }


def _clustered_mean() -> dict:
    return {
        "numerator": 10.0,
        "denominator": 2,
        "estimate": 5.0,
        "mean": 5.0,
        "std": 1.5,
        "median": 5.0,
        "q1": 4.0,
        "q3": 6.0,
        "iqr": 2.0,
        "ci95_low": 3.5,
        "ci95_high": 6.5,
        "ci_low": 3.5,
        "ci_high": 6.5,
        "confidence": 0.95,
        "valid_bootstrap_samples": 100,
        "interval_status": "ESTIMATED",
        "interval_method": "percentile_task_cluster_bootstrap",
        "inference_unit": "task_id",
        "n_task_clusters": 2,
        "missing_seed_repeat_cells": 0,
    }


def test_rate_formatter_and_final_headline_include_counts_and_ci(tmp_path):
    value = _clustered_rate(3, 5, low=0.2, high=0.8)
    assert _format_rate(value) == "60.00% (3/5; 95% CI 20.00%, 80.00%)"
    assert _format_rate(_clustered_rate(0, 0)) == "N/A"
    unavailable = {
        **_clustered_rate(0, 0),
        "reason": "recovery controller disabled for this system",
    }
    assert _format_rate(unavailable) == (
        "N/A (0/0; recovery controller disabled for this system)"
    )
    one_cluster = {
        **value,
        "ci95_low": None,
        "ci95_high": None,
        "ci_low": None,
        "ci_high": None,
        "ci_reason": "fewer than two unique task clusters",
    }
    assert _format_rate(one_cluster) == (
        "60.00% (3/5; 95% CI N/A: fewer than two unique task clusters)"
    )

    systems = {}
    for system_id in SYSTEM_IDS:
        systems[system_id] = {
            "task_success_rate": value,
            "recovery_success_rate": (
                _clustered_rate(0, 0) if system_id in {"E0", "E1"} else value
            ),
            "success_after_initial_failure": value,
            "loop_episode_rate": value,
            "unrecovered_failure_rate": value,
            "steps": {"all": _clustered_mean()},
        }
    metrics = {"systems": systems}

    _write_main_table(tmp_path, metrics, FINAL_READY_STATUS)
    with (tmp_path / "table2_main.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["Task Success Rate"] == (
        "60.00% (3/5; 95% CI 20.00%, 80.00%)"
    )
    assert rows[0]["Recovery Success Rate"] == "N/A"
    assert rows[2]["Recovery Success Rate"] == (
        "60.00% (3/5; 95% CI 20.00%, 80.00%)"
    )
    assert rows[0]["Avg. Browser Actions"] == (
        "5.00 (10.00/2; 95% CI 3.50, 6.50)"
    )

    _write_main_table(tmp_path, metrics, "PILOT_ONLY")
    with (tmp_path / "table2_main.csv").open(encoding="utf-8", newline="") as handle:
        guarded = list(csv.DictReader(handle))
    assert all(row["Task Success Rate"] == "N/R" for row in guarded)
    assert all(row["Evidence Status"] == "PILOT_ONLY" for row in guarded)


def test_metrics_csv_flattens_continuous_distribution_and_environment(tmp_path):
    systems = {
        system_id: {
            "task_success_rate": _clustered_rate(1, 2),
            "steps": {"all": _clustered_mean()},
            "efficiency": {"task_wall_clock_seconds": _clustered_mean()},
        }
        for system_id in SYSTEM_IDS
    }
    metrics = {
        "publication_status": "PILOT_ONLY",
        "systems": systems,
        "environment_failure_rate": {
            "all_launches": rate(2, 20),
            "original_launches": rate(1, 16),
            "rerun_launches": rate(1, 4),
            "by_system": {
                system_id: rate(index, 5)
                for index, system_id in enumerate(SYSTEM_IDS)
            },
        },
    }

    _write_metric_csv(tmp_path, metrics)
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    e2 = next(row for row in rows if row["system_id"] == "E2")
    assert e2["steps_all_std"] == "1.5"
    assert e2["steps_all_q1"] == "4.0"
    assert e2["steps_all_q3"] == "6.0"
    assert e2["steps_all_iqr"] == "2.0"
    assert e2["steps_all_confidence"] == "0.95"
    assert e2["environment_failure_rate_numerator"] == "2"
    assert e2["environment_failure_rate_denominator"] == "5"
    assert e2["environment_failure_rate_interval_method"] == "wilson_score_95"
    assert e2["environment_failure_all_launches_numerator"] == "2"
    assert e2["environment_failure_original_launches_denominator"] == "16"
    assert e2["environment_failure_rerun_launches_denominator"] == "4"


def test_retrieval_csv_keeps_counts_task_cluster_ci_and_na_reasons(tmp_path):
    retrieval = {
        "mean_reciprocal_rank": {
            **_clustered_mean(),
            "query_denominator": 2,
        },
        "strategy_hit_at_5": {
            "numerator": None,
            "denominator": 2,
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "confidence": 0.95,
            "interval_method": "percentile_task_cluster_bootstrap",
            "inference_unit": "task_id",
            "n_task_clusters": 2,
            "valid_bootstrap_samples": 0,
            "interval_status": "NOT_APPLICABLE_REGISTERED_DEPTH",
            "ci_reason": "registered retrieval depth is below 5",
            "reason": "registered retrieval depth is below 5",
            "query_denominator": 2,
            "available_query_count": 0,
            "display": "N/A",
        },
        "query_count": 2,
    }
    rows = _retrieval_metric_rows(
        retrieval,
        publication_status="DRAFT_PILOT_ONLY",
    )
    _write_result_csv(
        tmp_path / "retrieval_metrics.csv",
        rows,
        RETRIEVAL_METRIC_FIELDS,
    )
    with (tmp_path / "retrieval_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        exported = {row["metric"]: row for row in csv.DictReader(handle)}

    mrr = exported["mean_reciprocal_rank"]
    assert mrr["numerator"] == "10.0"
    assert mrr["denominator"] == "2"
    assert mrr["estimate"] == "5.0"
    assert mrr["ci95_low"] == "3.5"
    assert mrr["ci95_high"] == "6.5"
    assert mrr["inference_unit"] == "task_id"
    unavailable = exported["strategy_hit_at_5"]
    assert unavailable["estimate"] == ""
    assert unavailable["ci95_low"] == ""
    assert unavailable["interval_status"] == "NOT_APPLICABLE_REGISTERED_DEPTH"
    assert unavailable["reason"] == "registered retrieval depth is below 5"


def _paired_rows() -> list[dict]:
    output = []
    outcomes = {
        "task-a": {"E0": False, "E1": True, "E2": True, "E3": False},
        "task-b": {"E0": False, "E1": False, "E2": True, "E3": True},
    }
    for task_id, systems in outcomes.items():
        for system_id, success in systems.items():
            output.append(
                {
                    "block_id": task_id,
                    "task_id": task_id,
                    "system_id": system_id,
                    "matched_model_seed": 42,
                    "repeat_id": 0,
                    "task_success": success,
                    "step_count": 2,
                    "loop_detected": False,
                    "saf_success_count": int(success),
                    "verified_failure_episode_count": 1,
                    "successful_recovery_attempt_count": int(
                        system_id in {"E2", "E3"} and success
                    ),
                    "initiated_recovery_attempt_count": int(
                        system_id in {"E2", "E3"}
                    ),
                }
            )
    return output


def test_paired_contrast_csv_keeps_significance_ratios_and_memory_harm(tmp_path):
    episodes = _paired_rows()
    statistics = compute_paired_contrasts(
        episodes, bootstrap_samples=25, seed=7
    )
    statistics["clustered_ratio_contrasts"] = compute_clustered_ratio_contrasts(
        episodes,
        ratio_fields={
            "success_after_initial_failure": (
                "saf_success_count",
                "verified_failure_episode_count",
            ),
            "registered_recovery_success": (
                "successful_recovery_attempt_count",
                "initiated_recovery_attempt_count",
            ),
        },
        bootstrap_samples=25,
        seed=7,
    )
    rows = _paired_contrast_rows(
        statistics, publication_status="DRAFT_PILOT_ONLY"
    )
    _write_result_csv(
        tmp_path / "paired_contrasts.csv", rows, PAIRED_CONTRAST_FIELDS
    )
    with (tmp_path / "paired_contrasts.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        exported = list(csv.DictReader(handle))

    success_rows = [row for row in exported if row["metric"] == "task_success"]
    assert len(success_rows) == 3
    assert all(row["exact_sign_two_sided_p"] != "" for row in success_rows)
    assert all(row["holm_adjusted_p"] != "" for row in success_rows)
    ratio_rows = [
        row for row in exported if row["analysis_family"] == "paired_ratio_difference"
    ]
    assert len(ratio_rows) == 6
    assert {row["metric"] for row in ratio_rows} == {
        "success_after_initial_failure",
        "registered_recovery_success",
    }
    memory = next(
        row
        for row in exported
        if row["metric"] == "paired_memory_regression_rate"
    )
    assert memory["contrast"] == "E3_minus_E2"
    assert memory["analysis_family"] == "task_level_harm_rate"
    assert memory["numerator"] == "1"
    assert memory["denominator"] == "2"
    assert memory["publication_status"] == "DRAFT_PILOT_ONLY"
