from __future__ import annotations

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.metrics import (
    compute_table2_metrics,
    rate,
    task_clustered_mean,
)


def _episode(system_id: str, *, success: bool, steps: int, loop: bool = False) -> dict:
    return {
        "block_id": "block-1",
        "episode_id": f"episode-{system_id}",
        "system_id": system_id,
        "task_id": "task-1",
        "task_success": success,
        "loop_detected": loop,
        "step_count": steps,
        "executed_action_count": steps,
        "rejected_action_count": 0,
        "recovery_action_count": 0 if system_id in {"E0", "E1"} else 1,
        "verified_failure_event_count": 0 if system_id == "E0" else 1,
        "repeated_error_events": 1 if system_id == "E3" else 0,
    }


def _registered_block(
    task_id: str,
    *,
    matched_seed: int,
    repeat_id: int,
    success: bool,
    suffix: str = "",
    overrides: dict[str, bool] | None = None,
) -> list[dict]:
    block_id = f"{task_id}-seed-{matched_seed}-repeat-{repeat_id}{suffix}"
    system_outcomes = {system_id: success for system_id in ("E0", "E1", "E2", "E3")}
    system_outcomes.update(overrides or {})
    return [
        {
            **_episode(system_id, success=system_outcomes[system_id], steps=2),
            "block_id": block_id,
            "episode_id": f"{block_id}-{system_id}",
            "task_id": task_id,
            "matched_model_seed": matched_seed,
            "repeat_id": repeat_id,
            "verified_failure_event_count": 0,
            "repeated_error_events": 0,
        }
        for system_id in ("E0", "E1", "E2", "E3")
    ]


def test_hand_calculated_table2_rates_and_na_denominators():
    episodes = [
        _episode("E0", success=True, steps=2),
        _episode("E1", success=False, steps=3),
        _episode("E2", success=True, steps=4),
        _episode("E3", success=False, steps=5, loop=True),
    ]
    incidents = [
        {"system_id": "E1", "episode_id": "episode-E1", "failure_incident_id": "i1", "verified_agent_failure": True, "attempt_count": 0, "resolved": False},
        {"system_id": "E2", "episode_id": "episode-E2", "failure_incident_id": "i2", "verified_agent_failure": True, "attempt_count": 1, "resolved_attempt_index": 1, "resolved": True},
        {"system_id": "E2", "episode_id": "episode-E2", "failure_incident_id": "i2-no-attempt", "verified_agent_failure": True, "attempt_count": 0, "resolved": False},
        {"system_id": "E3", "episode_id": "episode-E3", "failure_incident_id": "i3", "verified_agent_failure": True, "attempt_count": 2, "resolved": False},
    ]
    attempts = [
        {"recovery_attempt_id": "e2-a1", "system_id": "E2", "episode_id": "episode-E2", "failure_incident_id": "i2", "attempt_index": 1, "initiated": True, "verified_failure_present": True, "successful": True},
        {"recovery_attempt_id": "e3-a1", "system_id": "E3", "episode_id": "episode-E3", "failure_incident_id": "i3", "attempt_index": 1, "initiated": True, "verified_failure_present": True, "successful": False},
        {"recovery_attempt_id": "e3-a2", "system_id": "E3", "episode_id": "episode-E3", "failure_incident_id": "i3", "attempt_index": 2, "initiated": True, "verified_failure_present": True, "successful": False},
    ]

    result = compute_table2_metrics(
        episodes,
        recovery_attempts=attempts,
        failure_incidents=incidents,
        recovery_k=2,
    )["systems"]

    assert result["E0"]["task_success_rate"]["estimate"] == 1.0
    assert result["E1"]["task_success_rate"]["estimate"] == 0.0
    assert result["E0"]["recovery_success_rate"]["display"] == "N/A"
    assert result["E1"]["recovery_success_rate"]["display"] == "N/A"
    assert result["E2"]["recovery_success_rate"]["estimate"] == 1.0
    assert result["E2"]["recovery_at_1"]["estimate"] == 1.0
    assert result["E2"]["recovery_at_1"]["denominator"] == 1
    assert result["E2"]["success_after_initial_failure"]["estimate"] == 1.0
    assert result["E3"]["recovery_success_rate"]["estimate"] == 0.0
    assert result["E3"]["recovery_at_2"]["estimate"] == 0.0
    assert result["E3"]["loop_episode_rate"]["estimate"] == 1.0
    assert result["E3"]["unrecovered_failure_rate"]["estimate"] == 1.0
    assert result["E3"]["repeated_error_event_rate"]["estimate"] == 1.0
    assert result["E3"]["steps"]["all"]["mean"] == 5.0
    assert result["E3"]["steps"]["all"]["numerator"] == 5.0
    assert result["E3"]["steps"]["all"]["denominator"] == 1
    assert result["E3"]["steps"]["all"]["estimate"] == 5.0
    assert result["E3"]["steps"]["all"]["ci95_low"] is None


def test_hand_calculated_environment_failure_rate():
    episodes = [
        _episode(system_id, success=False, steps=1)
        for system_id in ("E0", "E1", "E2", "E3")
    ]
    schedule_attempts = [
        {
            "system_id": "E0",
            "attempt_id": 0,
            "launched": True,
            "infrastructure_invalid": False,
        },
        {
            "system_id": "E1",
            "attempt_id": 0,
            "launched": True,
            "infrastructure_invalid": True,
        },
        {
            "system_id": "E2",
            "attempt_id": 0,
            "launched": False,
            "infrastructure_invalid": False,
        },
        {
            "system_id": "E3",
            "attempt_id": 0,
            "launched": True,
            "infrastructure_invalid": False,
        },
        {
            "system_id": "E0",
            "attempt_id": 1,
            "launched": True,
            "infrastructure_invalid": False,
        },
        {
            "system_id": "E1",
            "attempt_id": 1,
            "launched": True,
            "infrastructure_invalid": False,
        },
    ]

    efr = compute_table2_metrics(
        episodes,
        schedule_attempts=schedule_attempts,
    )["environment_failure_rate"]

    assert efr["all_launches"]["numerator"] == 1
    assert efr["all_launches"]["denominator"] == 5
    assert efr["all_launches"]["estimate"] == pytest.approx(1 / 5)
    assert efr["original_launches"]["estimate"] == pytest.approx(1 / 3)
    assert efr["rerun_launches"]["estimate"] == 0.0
    assert efr["by_system"]["E1"]["estimate"] == pytest.approx(1 / 2)
    assert efr["by_system"]["E2"]["display"] == "N/A"


def test_rate_rejects_impossible_counts_and_uses_na_for_empty_denominator():
    assert rate(0, 0)["display"] == "N/A"
    with pytest.raises(ValueError):
        rate(2, 1)


def test_task_clustered_mean_reports_total_denominator_and_task_interval():
    contributions = []
    for repeat_id in range(10):
        contributions.extend(
            [
                {
                    "task_id": "task-low",
                    "cell": ("matched_seed_repeat", 42, repeat_id),
                    "value": 1,
                },
                {
                    "task_id": "task-high",
                    "cell": ("matched_seed_repeat", 42, repeat_id),
                    "value": 9,
                },
            ]
        )

    result = task_clustered_mean(
        contributions,
        metric_name="browser_actions_all",
        system_id="E0",
        samples=500,
        seed=17,
    )

    assert result["numerator"] == 100.0
    assert result["denominator"] == 20
    assert result["estimate"] == 5.0
    assert result["mean"] == 5.0
    assert result["n_task_clusters"] == 2
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 9.0
    assert result["interval_method"] == "percentile_task_cluster_bootstrap"
    assert result["inference_unit"] == "task_id"


def test_task_clustered_mean_collapses_exact_cells_and_rejects_conflicts():
    base = {
        "task_id": "task-a",
        "cell": ("matched_seed_repeat", 42, 0),
        "value": 3,
    }
    duplicate = task_clustered_mean(
        [base, dict(base)],
        metric_name="browser_actions_all",
        system_id="E0",
        samples=25,
    )
    assert duplicate["denominator"] == 1
    assert duplicate["duplicate_seed_repeat_cells_collapsed"] == 1
    assert duplicate["interval_status"] == (
        "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS"
    )

    with pytest.raises(SchemaError, match="conflicting continuous contributions"):
        task_clustered_mean(
            [base, {**base, "value": 4}],
            metric_name="browser_actions_all",
            system_id="E0",
            samples=25,
        )

    missing = {**base, "value": None}
    missing_duplicate = task_clustered_mean(
        [missing, dict(missing)],
        metric_name="browser_actions_all",
        system_id="E0",
        samples=25,
    )
    assert missing_duplicate["display"] == "N/A"
    assert missing_duplicate["denominator"] == 0
    assert missing_duplicate["missing_seed_repeat_cells"] == 1
    assert missing_duplicate["duplicate_seed_repeat_cells_collapsed"] == 1
    with pytest.raises(SchemaError, match="conflicting continuous contributions"):
        task_clustered_mean(
            [missing, base],
            metric_name="browser_actions_all",
            system_id="E0",
            samples=25,
        )


def test_task_clustered_mean_force_na_clears_every_descriptive_value():
    result = task_clustered_mean(
        [
            {
                "task_id": "task-a",
                "cell": ("matched_seed_repeat", 42, 0),
                "value": 1,
            },
            {
                "task_id": "task-b",
                "cell": ("matched_seed_repeat", 42, 0),
                "value": 9,
            },
        ],
        metric_name="recovery_attempts_all",
        system_id="E0",
        samples=25,
        force_na=True,
    )

    for field in (
        "mean",
        "std",
        "median",
        "q1",
        "q3",
        "iqr",
        "min",
        "max",
        "estimate",
        "ci_low",
        "ci_high",
        "ci95_low",
        "ci95_high",
    ):
        assert result[field] is None
    assert result["display"] == "N/A"


def test_recovery_disabled_systems_use_na_for_continuous_attempt_means():
    episodes = [
        _episode(system_id, success=False, steps=1)
        for system_id in ("E0", "E1", "E2", "E3")
    ]
    metrics = compute_table2_metrics(episodes, bootstrap_samples=25)["systems"]
    for system_id in ("E0", "E1"):
        for metric_name in (
            "recovery_attempts_per_episode",
            "recovery_attempts_per_failure_episode",
        ):
            value = metrics[system_id][metric_name]
            assert value["display"] == "N/A"
            assert value["estimate"] is None
            assert value["mean"] is None
            assert value["numerator"] == 0.0
            assert value["denominator"] == 0


def test_task_clustered_mean_uses_pooled_cells_with_unequal_cluster_sizes():
    contributions = [
        {
            "task_id": "task-a",
            "cell": ("matched_seed_repeat", 42, 0),
            "value": 0,
        },
        *[
            {
                "task_id": "task-b",
                "cell": ("matched_seed_repeat", 42, repeat_id),
                "value": 10,
            }
            for repeat_id in range(3)
        ],
    ]
    result = task_clustered_mean(
        contributions,
        metric_name="browser_actions_all",
        system_id="E0",
        samples=500,
        seed=31,
    )
    assert result["numerator"] == 30.0
    assert result["denominator"] == 4
    assert result["estimate"] == 7.5
    assert result["n_task_clusters"] == 2


def test_per_system_tsr_interval_uses_tasks_not_seed_repeat_rows():
    episodes = []
    for repeat_id in range(12):
        episodes.extend(
            _registered_block(
                "task-always-fails",
                matched_seed=42,
                repeat_id=repeat_id,
                success=False,
            )
        )
        episodes.extend(
            _registered_block(
                "task-always-succeeds",
                matched_seed=42,
                repeat_id=repeat_id,
                success=True,
            )
        )

    result = compute_table2_metrics(
        episodes,
        bootstrap_samples=500,
        confidence=0.95,
        bootstrap_seed=17,
    )
    tsr = result["systems"]["E0"]["task_success_rate"]

    assert tsr["estimate"] == 0.5
    assert tsr["denominator"] == 24
    assert tsr["n_task_clusters"] == 2
    assert tsr["ci95_low"] == 0.0
    assert tsr["ci95_high"] == 1.0
    assert tsr["interval_method"] == "percentile_task_cluster_bootstrap"
    assert tsr["inference_unit"] == "task_id"
    assert tsr["estimand"] == (
        "pooled_ratio_over_unique_registered_seed_repeat_cells"
    )
    assert result["per_system_rate_inference"]["within_task_aggregation"] == (
        "retain_all_unique_matched_seed_repeat_cells_and_linked_counts"
    )


def test_per_system_rate_interval_is_invariant_to_exact_cell_duplication():
    episodes = []
    for task_id, outcomes in {
        "task-a": (False, False),
        "task-b": (True, True),
        "task-c": (False, True),
    }.items():
        for matched_seed, success in zip((42, 43), outcomes):
            episodes.extend(
                _registered_block(
                    task_id,
                    matched_seed=matched_seed,
                    repeat_id=0,
                    success=success,
                )
            )

    original = compute_table2_metrics(
        episodes,
        bootstrap_samples=1_000,
        bootstrap_seed=23,
    )["systems"]["E0"]["task_success_rate"]
    duplicated_rows = [
        *episodes,
        *_registered_block(
            "task-c",
            matched_seed=43,
            repeat_id=0,
            success=True,
            suffix="-accidental-copy",
        ),
    ]
    duplicated = compute_table2_metrics(
        duplicated_rows,
        bootstrap_samples=1_000,
        bootstrap_seed=23,
    )["systems"]["E0"]["task_success_rate"]

    for field in (
        "numerator",
        "denominator",
        "estimate",
        "ci95_low",
        "ci95_high",
        "n_task_clusters",
        "unique_seed_repeat_cells",
        "bootstrap_seed",
    ):
        assert duplicated[field] == original[field]
    assert original["duplicate_seed_repeat_cells_collapsed"] == 0
    assert duplicated["duplicate_seed_repeat_cells_collapsed"] == 1


def test_attempt_rate_interval_is_invariant_to_duplicate_episode_cell():
    first = _registered_block(
        "task-a",
        matched_seed=42,
        repeat_id=0,
        success=False,
    )
    second = _registered_block(
        "task-b",
        matched_seed=42,
        repeat_id=0,
        success=False,
    )

    def recovery_evidence(rows: list[dict], *, successful: bool) -> tuple[dict, dict]:
        episode_id = next(
            row["episode_id"] for row in rows if row["system_id"] == "E2"
        )
        incident_id = f"{episode_id}-incident"
        return (
            {
                "system_id": "E2",
                "episode_id": episode_id,
                "failure_incident_id": incident_id,
                "verified_agent_failure": True,
                "resolved": successful,
                "resolved_attempt_index": 1 if successful else None,
            },
            {
                "recovery_attempt_id": f"{episode_id}-attempt",
                "system_id": "E2",
                "episode_id": episode_id,
                "failure_incident_id": incident_id,
                "attempt_index": 1,
                "initiated": True,
                "verified_failure_present": True,
                "successful": successful,
            },
        )

    first_incident, first_attempt = recovery_evidence(first, successful=True)
    second_incident, second_attempt = recovery_evidence(second, successful=False)
    original = compute_table2_metrics(
        [*first, *second],
        recovery_attempts=[first_attempt, second_attempt],
        failure_incidents=[first_incident, second_incident],
        bootstrap_samples=500,
        bootstrap_seed=31,
    )["systems"]["E2"]["recovery_success_rate"]

    duplicate = _registered_block(
        "task-b",
        matched_seed=42,
        repeat_id=0,
        success=False,
        suffix="-accidental-copy",
    )
    duplicate_incident, duplicate_attempt = recovery_evidence(
        duplicate,
        successful=False,
    )
    repeated = compute_table2_metrics(
        [*first, *second, *duplicate],
        recovery_attempts=[first_attempt, second_attempt, duplicate_attempt],
        failure_incidents=[first_incident, second_incident, duplicate_incident],
        bootstrap_samples=500,
        bootstrap_seed=31,
    )["systems"]["E2"]["recovery_success_rate"]

    for field in (
        "numerator",
        "denominator",
        "estimate",
        "ci95_low",
        "ci95_high",
        "n_task_clusters",
        "unique_seed_repeat_cells",
    ):
        assert repeated[field] == original[field]
    assert repeated["estimate"] == 0.5
    assert repeated["duplicate_seed_repeat_cells_collapsed"] == 1


def test_conflicting_duplicate_cell_rate_contribution_fails_closed():
    episodes = [
        *_registered_block(
            "task-a",
            matched_seed=42,
            repeat_id=0,
            success=True,
        ),
        *_registered_block(
            "task-a",
            matched_seed=42,
            repeat_id=0,
            success=True,
            suffix="-conflicting-copy",
            overrides={"E0": False},
        ),
    ]

    with pytest.raises(SchemaError, match="conflicting rate contributions"):
        compute_table2_metrics(
            episodes,
            bootstrap_samples=50,
            bootstrap_seed=29,
        )


def test_recovery_attempt_must_link_to_a_registered_failure_incident():
    episodes = [
        _episode(system_id, success=False, steps=1) for system_id in ("E0", "E1", "E2", "E3")
    ]
    with pytest.raises(SchemaError, match="linked failure incident"):
        compute_table2_metrics(
            episodes,
            recovery_attempts=[
                {
                    "recovery_attempt_id": "orphan-attempt",
                    "system_id": "E2",
                    "episode_id": "episode-E2",
                    "failure_incident_id": "missing-incident",
                    "initiated": True,
                    "verified_failure_present": True,
                    "successful": False,
                }
            ],
            failure_incidents=[],
        )


def test_efficiency_companion_evidence_reports_measured_values_and_na():
    episodes = [
        {
            **_episode(system_id, success=True, steps=2),
            "input_token_count": 100,
            "output_token_count": 25,
            "model_parameter_count": 7_000_000_000,
            "trainable_parameter_count": 25_000_000,
            "peak_gpu_memory_mb": 12_345.5,
            "peak_system_memory_mb": 4_096.0,
            "training_gpu_hours": 11.25,
            **({"memory_index_size": 321} if system_id == "E3" else {}),
        }
        for system_id in ("E0", "E1", "E2", "E3")
    ]
    systems = compute_table2_metrics(episodes)["systems"]
    assert systems["E0"]["efficiency"]["input_token_count"]["mean"] == 100.0
    assert systems["E2"]["efficiency"]["training_gpu_hours"]["mean"] == 11.25
    assert systems["E3"]["efficiency"]["memory_index_size"]["mean"] == 321.0
    assert systems["E0"]["efficiency"]["memory_index_size"]["mean"] is None

    invalid = [dict(row) for row in episodes]
    invalid[0]["input_token_count"] = True
    with pytest.raises(SchemaError, match="input_token_count"):
        compute_table2_metrics(invalid)
