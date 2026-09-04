from __future__ import annotations

from typing import Any

from web_agent.eval.table2.retrieval_metrics import compute_retrieval_diagnostics


def _query(
    query_id: str,
    *,
    task_id: str,
    retrieved_ids: list[str],
    relevant_ids: list[str],
    latency_ms: float,
    admitted: bool = False,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "episode_id": f"episode-{task_id}",
        "task_id": task_id,
        "system_id": "E3",
        "retrieved_ids": retrieved_ids,
        "relevant_ids": relevant_ids,
        "relevance_definition": "registered_item_relevance",
        "abstained": not admitted,
        "admitted": admitted,
        "changed_decision": admitted,
        "changed_strategy": admitted,
        "changed_target_or_parameters": False,
        "useful_intervention": True if admitted else None,
        "harmful_intervention": False if admitted else None,
        "shadow_decision_hash": query_id[-1] * 64,
        "query_latency_ms": latency_ms,
        "index_size": 10,
        "retrieval_depth": 3,
    }


def test_retrieval_metrics_use_hand_calculated_task_cluster_intervals():
    queries = [
        _query(
            "query-a",
            task_id="task-a",
            retrieved_ids=["m1", "x", "y"],
            relevant_ids=["m1"],
            latency_ms=10.0,
            admitted=True,
        ),
        _query(
            "query-b",
            task_id="task-a",
            retrieved_ids=["x", "m2", "y"],
            relevant_ids=["m2"],
            latency_ms=20.0,
        ),
        _query(
            "query-c",
            task_id="task-b",
            retrieved_ids=["x", "y", "z"],
            relevant_ids=["m3"],
            latency_ms=30.0,
        ),
    ]

    result = compute_retrieval_diagnostics(
        queries,
        bootstrap_samples=2_000,
        confidence=0.95,
        bootstrap_seed=17,
    )

    recall_at_1 = result["recall_at_1"]
    assert recall_at_1["numerator"] == 1.0
    assert recall_at_1["denominator"] == 3
    assert recall_at_1["estimate"] == 1 / 3
    assert recall_at_1["ci95_low"] == 0.0
    assert recall_at_1["ci95_high"] == 0.5

    recall_at_3 = result["recall_at_3"]
    assert recall_at_3["numerator"] == 2.0
    assert recall_at_3["denominator"] == 3
    assert recall_at_3["estimate"] == 2 / 3
    assert recall_at_3["ci95_low"] == 0.0
    assert recall_at_3["ci95_high"] == 1.0

    mrr = result["mean_reciprocal_rank"]
    assert mrr["numerator"] == 1.5
    assert mrr["denominator"] == 3
    assert mrr["estimate"] == 0.5
    assert mrr["ci95_low"] == 0.0
    assert mrr["ci95_high"] == 0.75
    assert mrr["n_task_clusters"] == 2
    assert mrr["inference_unit"] == "task_id"
    assert mrr["valid_bootstrap_samples"] == 2_000

    latency = result["query_latency_ms"]
    assert latency["numerator"] == 60.0
    assert latency["denominator"] == 3
    assert latency["estimate"] == 20.0
    assert latency["ci95_low"] == 15.0
    assert latency["ci95_high"] == 30.0

    unsupported = result["recall_at_5"]
    assert unsupported["numerator"] is None
    assert unsupported["denominator"] == 3
    assert unsupported["estimate"] is None
    assert unsupported["ci95_low"] is None
    assert unsupported["ci95_high"] is None
    assert unsupported["reason"] == "registered retrieval depth is below 5"


def test_every_reported_retrieval_metric_has_full_uncertainty_contract():
    result = compute_retrieval_diagnostics(
        [
            _query(
                "query-d",
                task_id="task-a",
                retrieved_ids=["x", "y", "z"],
                relevant_ids=[],
                latency_ms=1.0,
            ),
            _query(
                "query-e",
                task_id="task-b",
                retrieved_ids=["x", "y", "z"],
                relevant_ids=[],
                latency_ms=2.0,
            ),
        ],
        bootstrap_samples=100,
        bootstrap_seed=19,
    )

    metric_rows = {
        name: value
        for name, value in result.items()
        if isinstance(value, dict) and "estimate" in value
    }
    assert metric_rows
    for name, metric in metric_rows.items():
        assert {
            "numerator",
            "denominator",
            "estimate",
            "ci95_low",
            "ci95_high",
            "confidence",
            "interval_method",
            "inference_unit",
            "interval_status",
        }.issubset(metric), name
        if metric["ci95_low"] is None or metric["ci95_high"] is None:
            assert metric.get("ci_reason"), name
        if metric["estimate"] is None:
            assert metric["display"] == "N/A"
            assert metric.get("reason"), name

    assert result["mean_reciprocal_rank"]["reason"] == (
        "no query has an eligible sealed relevance item"
    )
    assert result["intervention_label_coverage"]["reason"] == (
        "no admitted memory interventions"
    )


def test_retrieval_bootstrap_is_deterministic_for_registered_seed():
    queries = [
        _query(
            "query-f",
            task_id="task-a",
            retrieved_ids=["m1", "x", "y"],
            relevant_ids=["m1"],
            latency_ms=3.0,
        ),
        _query(
            "query-g",
            task_id="task-b",
            retrieved_ids=["x", "y", "z"],
            relevant_ids=["m2"],
            latency_ms=7.0,
        ),
    ]
    first = compute_retrieval_diagnostics(
        queries,
        bootstrap_samples=257,
        bootstrap_seed=23,
    )
    second = compute_retrieval_diagnostics(
        queries,
        bootstrap_samples=257,
        bootstrap_seed=23,
    )

    assert first == second
