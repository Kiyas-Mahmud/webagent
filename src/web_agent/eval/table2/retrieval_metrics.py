"""Pillar-4 retrieval diagnostics kept separate from end-to-end outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

from .common import SCHEMA_VERSION, SchemaError, strict_bool
from .metrics import describe, rate


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
) -> dict[str, Any]:
    """Compute preregistered retrieval metrics without redefining relevance.

    ``relevant_ids`` must be attached by sealed analysis after the retrieval and
    intervention decision.  Queries with no eligible relevant item are omitted
    from Recall@K and MRR, and remain visible in relevance coverage.
    """

    if not recall_ks or any(k <= 0 for k in recall_ks):
        raise ValueError("recall_ks must contain positive integers")
    rows = [_normalize_query(row) for row in queries]
    if not rows:
        return {
            "schema_version": SCHEMA_VERSION,
            "query_count": 0,
            "status": "N/A",
            "reason": "no E3 memory queries",
        }

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
                "display": "N/A",
                "reason": f"registered retrieval depth is below {k}",
            }
            continue
        values = []
        hits = 0
        for row in evaluable:
            relevant = set(row["relevant_ids"])
            retrieved = row["retrieved_ids"][:k]
            overlap = len(relevant.intersection(retrieved))
            values.append(overlap / len(relevant))
            hits += int(overlap > 0)
        if strategy_only:
            recall_results[metric_name] = rate(hits, len(evaluable))
        else:
            numerator = sum(values)
            recall_results[metric_name] = {
                "query_denominator": len(evaluable),
                "numerator": numerator,
                "denominator": len(evaluable),
                "estimate": numerator / len(values) if values else None,
                "display": f"{numerator / len(values):.4f}" if values else "N/A",
            }

    reciprocal_ranks: list[float] = []
    for row in evaluable:
        relevant = set(row["relevant_ids"])
        rank = next(
            (index for index, memory_id in enumerate(row["retrieved_ids"], start=1) if memory_id in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    retrieved_count = sum(len(row["retrieved_ids"]) for row in rows)
    irrelevant_count = sum(
        sum(memory_id not in set(row["relevant_ids"]) for memory_id in row["retrieved_ids"])
        for row in evaluable
    )
    evaluable_retrieved_count = sum(len(row["retrieved_ids"]) for row in evaluable)
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
        **recall_results,
        "mean_reciprocal_rank": {
            "query_denominator": len(evaluable),
            "numerator": sum(reciprocal_ranks),
            "denominator": len(evaluable),
            "estimate": (
                sum(reciprocal_ranks) / len(reciprocal_ranks)
                if reciprocal_ranks else None
            ),
            "display": (
                f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}"
                if reciprocal_ranks
                else "N/A"
            ),
        },
        "retrieval_coverage": rate(
            sum(bool(row["retrieved_ids"]) for row in rows), len(rows)
        ),
        "relevance_evaluable_coverage": rate(len(evaluable), len(rows)),
        "abstention_rate": rate(sum(row["abstained"] for row in rows), len(rows)),
        "abstention_coverage": rate(sum(row["abstained"] for row in rows), len(rows)),
        "irrelevant_memory_rate": rate(irrelevant_count, evaluable_retrieved_count),
        "intervention_coverage": rate(len(interventions), len(rows)),
        "intervention_label_coverage": rate(
            len(labeled_interventions), len(interventions)
        ),
        "irrelevant_intervention_rate": rate(
            len(irrelevant_interventions), len(interventions)
        ),
        "strategy_or_target_change_rate": rate(len(changed), len(interventions)),
        "strategy_change_rate": rate(len(changed_strategy), len(interventions)),
        "target_or_parameter_change_rate": rate(len(changed_target), len(interventions)),
        "useful_intervention_rate": rate(len(useful), len(labeled_interventions)),
        "harmful_intervention_rate": rate(len(harmful), len(labeled_interventions)),
        "retrieved_item_count": retrieved_count,
        "query_latency_ms": describe(
            row["query_latency_ms"]
            for row in rows
            if row["query_latency_ms"] is not None
        ),
        "index_sizes": sorted({row["index_size"] for row in rows if row["index_size"] is not None}),
    }


def _normalize_query(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {"query_id", "system_id", "retrieved_ids", "abstained", "admitted", "changed_decision"}
    missing = sorted(required - set(record))
    if missing:
        raise SchemaError(f"memory query is missing required fields: {missing}")
    if str(record["system_id"]) != "E3":
        raise SchemaError("primary retrieval diagnostics may contain only E3 queries")
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
