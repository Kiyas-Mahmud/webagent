"""Pillar-4 retrieval diagnostics kept separate from end-to-end outcomes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import math
from typing import Any

from .common import SCHEMA_VERSION, SchemaError, strict_bool
from .metrics import task_clustered_mean, task_clustered_rate


STRATEGY_RELEVANCE_DEFINITIONS = frozenset(
    {
        "strategy",
        "strategy_label",
        "same_strategy",
        # Exact registered train-only calibration/relevance proxy in
        # configs/eval/table2/protocol.yaml.  It is a binary strategy match,
        # so publishing it as set-recall would change the preregistered metric.
        "matching_verified_successful_correction_strategy_v1",
    }
)


def compute_retrieval_diagnostics(
    queries: Iterable[Mapping[str, Any]],
    *,
    recall_ks: tuple[int, ...] = (1, 3, 5),
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20250831,
) -> dict[str, Any]:
    """Compute preregistered retrieval metrics without redefining relevance.

    ``relevant_ids`` must be attached by sealed analysis after the retrieval and
    intervention decision.  Queries with no eligible relevant item are omitted
    from Recall@K and MRR, and remain visible in relevance coverage.
    """

    if not recall_ks or any(k <= 0 for k in recall_ks):
        raise ValueError("recall_ks must contain positive integers")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative exact integer")
    rows = [_normalize_query(row) for row in queries]
    if not rows:
        return {
            "schema_version": SCHEMA_VERSION,
            "query_count": 0,
            "status": "N/A",
            "reason": "no E3 memory queries",
            "retrieval_inference": {
                "interval_method": "percentile_task_cluster_bootstrap",
                "inference_unit": "task_id",
                "bootstrap_samples": bootstrap_samples,
                "confidence": confidence,
                "seed": bootstrap_seed,
            },
        }

    query_cells = [(row["episode_id"], row["query_id"]) for row in rows]
    if len(query_cells) != len(set(query_cells)):
        raise SchemaError("retrieval diagnostics contain duplicate episode/query cells")

    def clustered_rate(
        metric_name: str,
        counts: Callable[[Mapping[str, Any]], tuple[int, int]],
        *,
        undefined_reason: str,
    ) -> dict[str, Any]:
        contributions = []
        for row in rows:
            numerator, denominator = counts(row)
            contributions.append(
                {
                    "task_id": row["task_id"],
                    "cell": (row["episode_id"], row["query_id"]),
                    "numerator": numerator,
                    "denominator": denominator,
                }
            )
        result = task_clustered_rate(
            contributions,
            metric_name=metric_name,
            system_id="E3",
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
        )
        return _complete_interval_contract(
            result,
            undefined_reason=undefined_reason,
        )

    def clustered_mean(
        metric_name: str,
        value: Callable[[Mapping[str, Any]], float | int | None],
        *,
        undefined_reason: str,
        display_digits: int = 4,
    ) -> dict[str, Any]:
        result = task_clustered_mean(
            (
                {
                    "task_id": row["task_id"],
                    "cell": (row["episode_id"], row["query_id"]),
                    "value": value(row),
                }
                for row in rows
            ),
            metric_name=metric_name,
            system_id="E3",
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
        )
        result = _complete_interval_contract(
            result,
            undefined_reason=undefined_reason,
        )
        if result["estimate"] is not None:
            result["display"] = f"{float(result['estimate']):.{display_digits}f}"
        return result

    definitions = {row["relevance_definition"] for row in rows if row["relevance_definition"]}
    if len(definitions) > 1:
        raise SchemaError(f"mixed retrieval relevance definitions: {sorted(definitions)}")
    definition = next(iter(definitions), "registered_item_relevance")
    strategy_only = definition in STRATEGY_RELEVANCE_DEFINITIONS
    evaluable = [row for row in rows if row["relevant_ids"]]

    recall_results: dict[str, Any] = {}
    for k in sorted(set(recall_ks)):
        insufficient = [row for row in evaluable if row["retrieval_depth"] < k]
        metric_name = f"strategy_hit_at_{k}" if strategy_only else f"recall_at_{k}"
        if insufficient:
            recall_results[metric_name] = {
                "query_denominator": len(evaluable),
                "available_query_count": len(evaluable) - len(insufficient),
                "numerator": None,
                "denominator": len(evaluable),
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "ci95_low": None,
                "ci95_high": None,
                "confidence": confidence,
                "interval_method": "percentile_task_cluster_bootstrap",
                "inference_unit": "task_id",
                "n_task_clusters": len(
                    {row["task_id"] for row in evaluable}
                ),
                "valid_bootstrap_samples": 0,
                "interval_status": "NOT_APPLICABLE_REGISTERED_DEPTH",
                "display": "N/A",
                "reason": f"registered retrieval depth is below {k}",
                "ci_reason": f"registered retrieval depth is below {k}",
            }
            continue
        if strategy_only:
            recall_results[metric_name] = clustered_rate(
                metric_name,
                lambda row, depth=k: (
                    int(
                        bool(
                            set(row["relevant_ids"]).intersection(
                                row["retrieved_ids"][:depth]
                            )
                        )
                    ),
                    int(bool(row["relevant_ids"])),
                ),
                undefined_reason="no query has an eligible sealed relevance item",
            )
        else:
            recall_results[metric_name] = clustered_mean(
                metric_name,
                lambda row, depth=k: (
                    len(
                        set(row["relevant_ids"]).intersection(
                            row["retrieved_ids"][:depth]
                        )
                    )
                    / len(row["relevant_ids"])
                    if row["relevant_ids"]
                    else None
                ),
                undefined_reason="no query has an eligible sealed relevance item",
            )
        recall_results[metric_name]["query_denominator"] = len(evaluable)

    def reciprocal_rank(row: Mapping[str, Any]) -> float | None:
        if not row["relevant_ids"]:
            return None
        relevant = set(row["relevant_ids"])
        rank = next(
            (
                index
                for index, memory_id in enumerate(row["retrieved_ids"], start=1)
                if memory_id in relevant
            ),
            None,
        )
        return 1.0 / rank if rank is not None else 0.0

    retrieved_count = sum(len(row["retrieved_ids"]) for row in rows)
    interventions = [row for row in rows if row["admitted"]]
    changed = [row for row in interventions if row["changed_decision"]]
    changed_strategy = [row for row in interventions if row["changed_strategy"]]
    changed_target = [row for row in interventions if row["changed_target_or_parameters"]]
    labeled_interventions = [
        row
        for row in interventions
        if row["useful_intervention"] is not None
        and row["harmful_intervention"] is not None
    ]
    useful = [row for row in labeled_interventions if row["useful_intervention"] is True]
    harmful = [row for row in labeled_interventions if row["harmful_intervention"] is True]
    irrelevant_interventions = [
        row
        for row in interventions
        if row["admitted_candidate_id"] not in set(row["relevant_ids"])
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OK",
        "query_count": len(rows),
        "relevance_definition": definition,
        "retrieval_inference": {
            "interval_method": "percentile_task_cluster_bootstrap",
            "inference_unit": "task_id",
            "estimand": "pooled_query_metric_over_task_clusters",
            "within_task_aggregation": "retain_all_unique_episode_query_cells",
            "bootstrap_samples": bootstrap_samples,
            "confidence": confidence,
            "seed": bootstrap_seed,
        },
        **recall_results,
        "mean_reciprocal_rank": {
            **clustered_mean(
                "mean_reciprocal_rank",
                reciprocal_rank,
                undefined_reason="no query has an eligible sealed relevance item",
            ),
            "query_denominator": len(evaluable),
        },
        "retrieval_coverage": clustered_rate(
            "retrieval_coverage",
            lambda row: (int(bool(row["retrieved_ids"])), 1),
            undefined_reason="no E3 memory queries",
        ),
        "relevance_evaluable_coverage": clustered_rate(
            "relevance_evaluable_coverage",
            lambda row: (int(bool(row["relevant_ids"])), 1),
            undefined_reason="no E3 memory queries",
        ),
        "abstention_rate": clustered_rate(
            "abstention_rate",
            lambda row: (int(row["abstained"]), 1),
            undefined_reason="no E3 memory queries",
        ),
        "abstention_coverage": clustered_rate(
            "abstention_coverage",
            lambda row: (int(row["abstained"]), 1),
            undefined_reason="no E3 memory queries",
        ),
        "irrelevant_memory_rate": clustered_rate(
            "irrelevant_memory_rate",
            lambda row: (
                sum(
                    memory_id not in set(row["relevant_ids"])
                    for memory_id in row["retrieved_ids"]
                )
                if row["relevant_ids"]
                else 0,
                len(row["retrieved_ids"]) if row["relevant_ids"] else 0,
            ),
            undefined_reason=(
                "no retrieved items on relevance-evaluable queries"
            ),
        ),
        "intervention_coverage": clustered_rate(
            "intervention_coverage",
            lambda row: (int(row["admitted"]), 1),
            undefined_reason="no E3 memory queries",
        ),
        "intervention_label_coverage": clustered_rate(
            "intervention_label_coverage",
            lambda row: (
                int(row in labeled_interventions),
                int(row["admitted"]),
            ),
            undefined_reason="no admitted memory interventions",
        ),
        "irrelevant_intervention_rate": clustered_rate(
            "irrelevant_intervention_rate",
            lambda row: (
                int(row in irrelevant_interventions),
                int(row["admitted"]),
            ),
            undefined_reason="no admitted memory interventions",
        ),
        "strategy_or_target_change_rate": clustered_rate(
            "strategy_or_target_change_rate",
            lambda row: (
                int(row in changed),
                int(row["admitted"]),
            ),
            undefined_reason="no admitted memory interventions",
        ),
        "strategy_change_rate": clustered_rate(
            "strategy_change_rate",
            lambda row: (
                int(row in changed_strategy),
                int(row["admitted"]),
            ),
            undefined_reason="no admitted memory interventions",
        ),
        "target_or_parameter_change_rate": clustered_rate(
            "target_or_parameter_change_rate",
            lambda row: (
                int(row in changed_target),
                int(row["admitted"]),
            ),
            undefined_reason="no admitted memory interventions",
        ),
        "useful_intervention_rate": clustered_rate(
            "useful_intervention_rate",
            lambda row: (
                int(row in useful),
                int(row in labeled_interventions),
            ),
            undefined_reason=(
                "no admitted intervention has complete sealed usefulness labels"
            ),
        ),
        "harmful_intervention_rate": clustered_rate(
            "harmful_intervention_rate",
            lambda row: (
                int(row in harmful),
                int(row in labeled_interventions),
            ),
            undefined_reason=(
                "no admitted intervention has complete sealed harmfulness labels"
            ),
        ),
        "retrieved_item_count": retrieved_count,
        "query_latency_ms": clustered_mean(
            "query_latency_ms",
            lambda row: row["query_latency_ms"],
            undefined_reason="no finite query-latency observations",
            display_digits=2,
        ),
        "index_sizes": sorted({row["index_size"] for row in rows if row["index_size"] is not None}),
    }


def _complete_interval_contract(
    result: dict[str, Any],
    *,
    undefined_reason: str,
) -> dict[str, Any]:
    """Attach a human-readable reason whenever a metric or its CI is undefined."""

    if result.get("estimate") is None:
        result["reason"] = undefined_reason
    if result.get("ci_low") is None or result.get("ci_high") is None:
        result.setdefault(
            "ci_reason",
            undefined_reason
            if result.get("estimate") is None
            else "fewer than two unique task clusters",
        )
    return result


def _normalize_query(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "query_id",
        "episode_id",
        "task_id",
        "system_id",
        "retrieved_ids",
        "abstained",
        "admitted",
        "changed_decision",
    }
    missing = sorted(required - set(record))
    if missing:
        raise SchemaError(f"memory query is missing required fields: {missing}")
    if str(record["system_id"]) != "E3":
        raise SchemaError("primary retrieval diagnostics may contain only E3 queries")
    if not all(str(record[field]).strip() for field in ("query_id", "episode_id", "task_id")):
        raise SchemaError("memory query identity fields must be nonempty")
    retrieved = record["retrieved_ids"]
    relevant = record.get("relevant_ids", [])
    if not isinstance(retrieved, list) or len(retrieved) != len(set(map(str, retrieved))):
        raise SchemaError(f"query {record['query_id']} has invalid/duplicate retrieved_ids")
    if not isinstance(relevant, list) or len(relevant) != len(set(map(str, relevant))):
        raise SchemaError(f"query {record['query_id']} relevant_ids must be a list")
    admitted = strict_bool(record["admitted"], context="admitted")
    abstained = strict_bool(record["abstained"], context="abstained")
    if abstained == admitted:
        raise SchemaError(
            f"query {record['query_id']} must set abstained to the inverse of admitted"
        )
    shadow_hash = record.get("shadow_decision_hash")
    if not isinstance(shadow_hash, str) or len(shadow_hash) != 64:
        raise SchemaError(f"query {record['query_id']} lacks a SHA-256 no-memory shadow decision")
    retrieval_depth = int(record.get("retrieval_depth", len(retrieved)))
    if retrieval_depth < len(retrieved) or retrieval_depth < 0:
        raise SchemaError(f"query {record['query_id']} has invalid retrieval_depth")
    index_size = (
        int(record["index_size"]) if record.get("index_size") not in (None, "") else None
    )
    if index_size is not None and index_size < len(retrieved):
        raise SchemaError(f"query {record['query_id']} has invalid index_size")
    latency_raw = record.get("query_latency_ms")
    query_latency = None if latency_raw in (None, "") else float(latency_raw)
    if query_latency is not None and (not math.isfinite(query_latency) or query_latency < 0):
        raise SchemaError(f"query {record['query_id']} has invalid query latency")
    useful = _optional_bool(record.get("useful_intervention"))
    harmful = _optional_bool(record.get("harmful_intervention"))
    if useful is True and harmful is True:
        raise SchemaError(f"query {record['query_id']} cannot be both useful and harmful")
    admitted_candidate_id = record.get("admitted_candidate_id")
    if admitted and admitted_candidate_id in (None, ""):
        admitted_candidate_id = retrieved[0] if retrieved else None
    if admitted and admitted_candidate_id is None:
        raise SchemaError(f"query {record['query_id']} admitted without a candidate")
    if admitted_candidate_id is not None and str(admitted_candidate_id) not in set(map(str, retrieved)):
        raise SchemaError(f"query {record['query_id']} admitted candidate was not retrieved")
    changed_strategy = strict_bool(
        record.get("changed_strategy", record["changed_decision"]),
        context="changed_strategy",
    )
    changed_target = strict_bool(
        record.get("changed_target_or_parameters", False),
        context="changed_target_or_parameters",
    )
    changed_decision = strict_bool(record["changed_decision"], context="changed_decision")
    if changed_decision != (changed_strategy or changed_target):
        raise SchemaError(f"query {record['query_id']} has inconsistent change flags")
    return {
        **dict(record),
        "query_id": str(record["query_id"]),
        "episode_id": str(record["episode_id"]),
        "task_id": str(record["task_id"]),
        "system_id": "E3",
        "retrieved_ids": [str(value) for value in retrieved],
        "relevant_ids": [str(value) for value in relevant],
        "relevance_definition": str(record.get("relevance_definition", "registered_item_relevance")),
        "abstained": abstained,
        "admitted": admitted,
        "changed_decision": changed_decision,
        "changed_strategy": changed_strategy,
        "changed_target_or_parameters": changed_target,
        "useful_intervention": useful,
        "harmful_intervention": harmful,
        "admitted_candidate_id": (
            str(admitted_candidate_id) if admitted_candidate_id is not None else None
        ),
        "query_latency_ms": query_latency,
        "index_size": index_size,
        "retrieval_depth": retrieval_depth,
    }


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    return strict_bool(value, context="optional intervention label")
