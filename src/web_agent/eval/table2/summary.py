"""Produce guarded aggregate artifacts from a validated campaign package."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .common import (
    SCHEMA_VERSION,
    SchemaError,
    Table2Error,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_file,
    sha256_json,
    strict_bool,
)
from .metrics import compute_table2_metrics
from .execution_guard import validate_analysis_source_identity
from .package_validator import (
    DRAFT_PILOT_STATUS,
    FINAL_READY_STATUS,
    MANUAL_AUDIT_BLINDING_MODE,
    MANUAL_AUDIT_CASE_CATEGORIES,
    MANUAL_AUDIT_MANIFEST_ID,
    _derive_manual_audit_selection_evidence,
    _load_audit_json_without_duplicate_keys,
    _load_selected_analysis_records_unchecked,
    load_selected_analysis_records,
    load_yaml,
    validate_manual_audit_reviewer_codebook,
    validate_manual_adjudication_completion,
    validate_campaign,
    write_csv,
)
from .retrieval_metrics import compute_retrieval_diagnostics
from .schedule import SYSTEM_IDS
from .sealed_verifier import assert_no_verifier_evidence
from .statistics import compute_clustered_ratio_contrasts, compute_paired_contrasts


MAIN_FIELDS = (
    "System",
    "Task Success Rate",
    "Recovery Success Rate",
    "Success After Initial Failure",
    "Avg. Browser Actions",
    "Loop Episode Rate",
    "Unrecovered Failure Rate",
    "Evidence Status",
)
PAIRED_CONTRAST_FIELDS = (
    "contrast",
    "metric",
    "analysis_family",
    "estimate",
    "ci_low",
    "ci_high",
    "absolute_change",
    "relative_change",
    "left_estimate",
    "right_estimate",
    "numerator",
    "denominator",
    "n_clusters",
    "n_rows",
    "exact_sign_two_sided_p",
    "holm_adjusted_p",
    "right_better_task_clusters",
    "left_better_task_clusters",
    "tied_task_clusters",
    "non_tied_task_clusters",
    "interval_method",
    "inference_unit",
    "confidence",
    "publication_status",
)
RETRIEVAL_METRIC_FIELDS = (
    "metric",
    "numerator",
    "denominator",
    "estimate",
    "ci95_low",
    "ci95_high",
    "confidence",
    "interval_method",
    "inference_unit",
    "n_task_clusters",
    "valid_bootstrap_samples",
    "interval_status",
    "ci_reason",
    "reason",
    "query_denominator",
    "available_query_count",
    "display",
    "publication_status",
)
def summarize_campaign(
    campaign_dir: str | Path,
    *,
    results_dir: str | Path | None = None,
    draft_pilot: bool = False,
) -> dict[str, Any]:
    if type(draft_pilot) is not bool:
        raise TypeError("draft_pilot must be an exact boolean")
    root = Path(campaign_dir).resolve()
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    validation = validate_campaign(root, require_complete=True, require_aggregates=False)
    atomic_write_json(aggregate / "validation_report.json", validation.to_dict())
    if not validation.passed:
        _write_guarded_main_table(aggregate, publication_status=validation.publication_status)
        raise Table2Error("campaign validation failed: " + "; ".join(validation.errors))

    analysis_source_identity = validate_analysis_source_identity(root)
    atomic_write_json(
        aggregate / "analysis_source_identity.json",
        analysis_source_identity,
    )

    protocol = load_yaml(root / "frozen" / "protocol.yaml")
    records = load_selected_analysis_records(root)
    normal_episode_ids = {
        str(row["episode_id"])
        for row in records["episodes"]
        if row.get("task_partition") == "normal"
    }
    recovery_episode_ids = {
        str(row["episode_id"])
        for row in records["episodes"]
        if row.get("task_partition") == "recovery_diagnostic"
    }
    normal_episodes = [
        row for row in records["episodes"] if str(row["episode_id"]) in normal_episode_ids
    ]
    recovery_episodes = [
        row for row in records["episodes"] if str(row["episode_id"]) in recovery_episode_ids
    ]
    recovery_k = int(protocol["budgets"].get("recovery_at_k", 2))
    statistics_cfg = protocol.get("statistics", {})
    rate_inference = {
        "bootstrap_samples": int(statistics_cfg.get("bootstrap_samples", 10_000)),
        "confidence": float(statistics_cfg.get("confidence_level", 0.95)),
        "bootstrap_seed": int(statistics_cfg.get("seed", 20250831)),
    }
    metrics = compute_table2_metrics(
        normal_episodes,
        recovery_attempts=[
            row for row in records["recovery_attempts"] if str(row["episode_id"]) in normal_episode_ids
        ],
        failure_incidents=[
            row for row in records["failure_incidents"] if str(row["episode_id"]) in normal_episode_ids
        ],
        schedule_attempts=records["schedule_attempts"],
        recovery_k=recovery_k,
        **rate_inference,
    )
    diagnostic_metrics = compute_table2_metrics(
        recovery_episodes,
        recovery_attempts=[
            row for row in records["recovery_attempts"] if str(row["episode_id"]) in recovery_episode_ids
        ],
        failure_incidents=[
            row for row in records["failure_incidents"] if str(row["episode_id"]) in recovery_episode_ids
        ],
        recovery_k=recovery_k,
        **rate_inference,
    )
    retrieval = compute_retrieval_diagnostics(
        [
            row for row in records["memory_queries"] if str(row["episode_id"]) in normal_episode_ids
        ],
        bootstrap_samples=rate_inference["bootstrap_samples"],
        confidence=rate_inference["confidence"],
        bootstrap_seed=rate_inference["bootstrap_seed"],
    )
    diagnostic_retrieval = compute_retrieval_diagnostics(
        [
            row for row in records["memory_queries"] if str(row["episode_id"]) in recovery_episode_ids
        ],
        bootstrap_samples=rate_inference["bootstrap_samples"],
        confidence=rate_inference["confidence"],
        bootstrap_seed=rate_inference["bootstrap_seed"],
    )
    statistic_keys = [
        "task_success",
        "step_count",
        "loop_detected",
    ]
    if all(row.get("task_wall_clock_seconds") is not None for row in normal_episodes):
        statistic_keys.append("task_wall_clock_seconds")
    if all(row.get("model_call_count") is not None for row in normal_episodes):
        statistic_keys.append("model_call_count")
    statistics = compute_paired_contrasts(
        normal_episodes,
        metric_keys=tuple(statistic_keys),
        bootstrap_samples=int(statistics_cfg.get("bootstrap_samples", 10_000)),
        confidence=float(statistics_cfg.get("confidence_level", 0.95)),
        seed=int(statistics_cfg.get("seed", 20250831)),
    )
    ratio_rows = _ratio_statistic_rows(
        normal_episodes,
        recovery_attempts=[
            row
            for row in records["recovery_attempts"]
            if str(row["episode_id"]) in normal_episode_ids
        ],
        failure_incidents=[
            row
            for row in records["failure_incidents"]
            if str(row["episode_id"]) in normal_episode_ids
        ],
    )
    statistics["clustered_ratio_contrasts"] = compute_clustered_ratio_contrasts(
        ratio_rows,
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
        bootstrap_samples=int(statistics_cfg.get("bootstrap_samples", 10_000)),
        confidence=float(statistics_cfg.get("confidence_level", 0.95)),
        seed=int(statistics_cfg.get("seed", 20250831)),
    )
    # Official outcome labels are hidden while system condition remains visible;
    # selection must exist before reviewers can work.
    # It is therefore the final permitted write before the publication gate.
    audit_selection = build_manual_audit_selection(root, records)
    campaign_manifest = read_json(root / "campaign_manifest.json")
    publication_status = _summary_publication_status(
        root,
        campaign_manifest,
        validation_status=validation.publication_status,
        draft_pilot=draft_pilot,
    )
    diagnostic_status = (
        publication_status
        if campaign_manifest.get("evidence_label") == "PILOT_ONLY"
        else "PILOT_ONLY"
    )
    metrics["publication_status"] = publication_status
    metrics["headline_task_partition"] = "normal"
    diagnostic_metrics["publication_status"] = diagnostic_status
    diagnostic_metrics["task_partition"] = "recovery_diagnostic"
    retrieval["publication_status"] = publication_status
    diagnostic_retrieval["publication_status"] = diagnostic_status
    statistics["publication_status"] = publication_status
    metrics["paper_table_status"] = (
        "READY" if publication_status == FINAL_READY_STATUS else "N/R"
    )
    atomic_write_json(aggregate / "metrics.json", metrics)
    atomic_write_json(aggregate / "recovery_diagnostic_metrics.json", diagnostic_metrics)
    atomic_write_json(aggregate / "retrieval_diagnostics.json", retrieval)
    atomic_write_json(
        aggregate / "recovery_diagnostic_retrieval.json", diagnostic_retrieval
    )
    atomic_write_json(aggregate / "statistics.json", statistics)
    _write_episode_csv(aggregate, records["episodes"])
    _write_metric_csv(aggregate, metrics)
    _write_main_table(aggregate, metrics, publication_status)
    _write_companion_table(aggregate, metrics, recovery_k, publication_status)
    resolved_results_dir = _resolve_results_dir(
        root,
        campaign_id=str(campaign_manifest["campaign_id"]),
        requested=results_dir,
        draft_pilot=publication_status == DRAFT_PILOT_STATUS,
    )
    results_manifest = _export_results(
        root,
        resolved_results_dir,
        metrics=metrics,
        retrieval=retrieval,
        statistics=statistics,
        validation={**validation.to_dict(), "publication_status": publication_status},
        audit_selection=audit_selection,
    )
    atomic_write_json(aggregate / "results_manifest.json", results_manifest)
    _write_aggregate_hashes(aggregate)
    _write_campaign_evidence_manifest(root)

    final_validation = validate_campaign(root, require_complete=True, require_aggregates=True)
    atomic_write_json(aggregate / "validation_report.json", final_validation.to_dict())
    if not final_validation.passed:
        raise Table2Error("aggregate package validation failed: " + "; ".join(final_validation.errors))
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_dir": str(root),
        "publication_status": final_validation.publication_status,
        "paper_table_status": (
            "READY" if final_validation.publication_status == FINAL_READY_STATUS else "N/R"
        ),
        "aggregate_dir": str(aggregate),
        "manual_audit_manifest": str(root / "manual_audit" / "selection_manifest.json"),
        "results_dir": str(resolved_results_dir),
    }


def _summary_publication_status(
    root: Path,
    campaign_manifest: Mapping[str, Any],
    *,
    validation_status: str,
    draft_pilot: bool,
) -> str:
    """Separate an explicit draft from a human-complete pilot publication."""

    pilot = campaign_manifest.get("evidence_label") == "PILOT_ONLY"
    if draft_pilot:
        if not pilot:
            raise Table2Error("draft_pilot is valid only for a PILOT_ONLY campaign")
        return DRAFT_PILOT_STATUS
    try:
        validate_manual_adjudication_completion(root)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError, SchemaError) as exc:
        boundary = "final PILOT_ONLY" if pilot else "locked-final"
        raise Table2Error(
            f"{boundary} summary/export requires completed, human-attested, "
            f"hash-bound outcome-label-hidden adjudication: {exc}"
        ) from exc
    return "PILOT_ONLY" if pilot else validation_status


def _write_episode_csv(aggregate: Path, episodes: list[dict[str, Any]]) -> None:
    fields = (
        "block_id",
        "episode_id",
        "task_id",
        "task_partition",
        "system_id",
        "matched_model_seed",
        "repeat_id",
        "task_success",
        "loop_detected",
        "environment_failure",
        "step_count",
        "executed_action_count",
        "rejected_action_count",
        "recovery_action_count",
        "recovery_attempt_count",
        "model_call_count",
        "task_wall_clock_seconds",
        "decision_latency_ms",
        "provider_latency_ms",
        "recovery_latency_ms",
        "retrieval_latency_ms",
        "input_token_count",
        "output_token_count",
        "model_parameter_count",
        "trainable_parameter_count",
        "peak_gpu_memory_mb",
        "peak_system_memory_mb",
        "memory_index_size",
        "training_gpu_hours",
        "verified_failure_event_count",
        "repeated_error_events",
    )
    write_csv(aggregate / "episodes.csv", episodes, fields)


def _ratio_statistic_rows(
    episodes: list[dict[str, Any]],
    *,
    recovery_attempts: list[dict[str, Any]],
    failure_incidents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in recovery_attempts:
        attempts_by_episode.setdefault(str(row["episode_id"]), []).append(row)
    incidents_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in failure_incidents:
        incidents_by_episode.setdefault(str(row["episode_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        verified_failure = any(
            strict_bool(
                row["verified_agent_failure"],
                context="statistics.verified_agent_failure",
            )
            for row in incidents_by_episode.get(episode_id, [])
        )
        initiated = [
            row
            for row in attempts_by_episode.get(episode_id, [])
            if strict_bool(row.get("initiated", True), context="statistics.initiated")
        ]
        successful = sum(
            strict_bool(
                row["verified_failure_present"],
                context="statistics.verified_failure_present",
            )
            and strict_bool(row["successful"], context="statistics.recovery_success")
            for row in initiated
        )
        output.append(
            {
                **episode,
                "verified_failure_episode_count": int(verified_failure),
                "saf_success_count": int(
                    verified_failure
                    and strict_bool(
                        episode["task_success"], context="statistics.task_success"
                    )
                ),
                "initiated_recovery_attempt_count": len(initiated),
                "successful_recovery_attempt_count": successful,
            }
        )
    return output


def _write_metric_csv(aggregate: Path, metrics: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for system_id in SYSTEM_IDS:
        values = metrics["systems"][system_id]
        row: dict[str, Any] = {
            "system_id": system_id,
            "publication_status": metrics["publication_status"],
        }

        def add_estimate(prefix: str, value: Mapping[str, Any]) -> None:
            row[f"{prefix}_numerator"] = value.get("numerator")
            row[f"{prefix}_denominator"] = value.get("denominator")
            row[prefix] = value.get("estimate")
            row[f"{prefix}_ci_low"] = value.get(
                "ci_low", value.get("ci95_low")
            )
            row[f"{prefix}_ci_high"] = value.get(
                "ci_high", value.get("ci95_high")
            )
            row[f"{prefix}_interval_method"] = value.get("interval_method")
            row[f"{prefix}_n_task_clusters"] = value.get("n_task_clusters")
            for field in ("std", "median", "q1", "q3", "iqr"):
                if field in value:
                    row[f"{prefix}_{field}"] = value.get(field)
            for field in (
                "confidence",
                "valid_bootstrap_samples",
                "interval_status",
                "missing_seed_repeat_cells",
                "reason",
                "ci_reason",
            ):
                if field in value:
                    row[f"{prefix}_{field}"] = value.get(field)

        for key, value in values.items():
            if isinstance(value, Mapping) and "estimate" in value:
                add_estimate(key, value)
            elif key in {"steps", "efficiency"} and isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, Mapping) and "estimate" in nested_value:
                        add_estimate(f"{key}_{nested_key}", nested_value)

        environment = metrics.get("environment_failure_rate")
        if isinstance(environment, Mapping):
            by_system = environment.get("by_system")
            if isinstance(by_system, Mapping) and isinstance(
                by_system.get(system_id), Mapping
            ):
                add_estimate(
                    "environment_failure_rate",
                    {
                        **by_system[system_id],
                        "interval_method": "wilson_score_95",
                        "confidence": 0.95,
                    },
                )
            for scope in ("all_launches", "original_launches", "rerun_launches"):
                scoped = environment.get(scope)
                if isinstance(scoped, Mapping):
                    add_estimate(
                        f"environment_failure_{scope}",
                        {
                            **scoped,
                            "interval_method": "wilson_score_95",
                            "confidence": 0.95,
                        },
                    )
        rows.append(row)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(aggregate / "metrics.csv", rows, fields)


def _write_main_table(
    aggregate: Path,
    metrics: Mapping[str, Any],
    publication_status: str,
) -> None:
    if publication_status != FINAL_READY_STATUS:
        _write_guarded_main_table(aggregate, publication_status=publication_status)
        return
    rows = []
    for system_id in SYSTEM_IDS:
        values = metrics["systems"][system_id]
        rows.append(
            {
                "System": system_id,
                "Task Success Rate": _format_rate(values["task_success_rate"]),
                "Recovery Success Rate": _format_rate(
                    values["recovery_success_rate"]
                ),
                "Success After Initial Failure": _format_rate(
                    values["success_after_initial_failure"]
                ),
                "Avg. Browser Actions": _format_continuous(values["steps"]["all"]),
                "Loop Episode Rate": _format_rate(values["loop_episode_rate"]),
                "Unrecovered Failure Rate": _format_rate(
                    values["unrecovered_failure_rate"]
                ),
                "Evidence Status": FINAL_READY_STATUS,
            }
        )
    write_csv(aggregate / "table2_main.csv", rows, MAIN_FIELDS)


def _write_guarded_main_table(aggregate: Path, *, publication_status: str) -> None:
    guard = "N/R"
    rows = []
    for system_id in SYSTEM_IDS:
        values = {field: guard for field in MAIN_FIELDS}
        values["System"] = system_id
        values["Evidence Status"] = publication_status
        rows.append(values)
    write_csv(aggregate / "table2_main.csv", rows, MAIN_FIELDS)


def _write_companion_table(
    aggregate: Path,
    metrics: Mapping[str, Any],
    recovery_k: int,
    publication_status: str,
) -> None:
    fields = (
        "System",
        "Recovery@1",
        f"Recovery@{recovery_k}",
        "Verified-failure-only RSR",
        "Episode recovery success rate",
        "False-trigger rate",
        "Unnecessary-intervention rate",
        "Repeated-error event rate",
        "Verified-failure incidence",
        "Avg. recovery attempts/run",
        "Avg. recovery attempts/failure episode",
        "Environment failure rate",
        "Task wall-clock seconds",
        "Model call count",
        "Decision latency ms",
        "Provider latency ms",
        "Recovery latency ms",
        "Retrieval latency ms",
        "Input tokens",
        "Output tokens",
        "Model parameters",
        "Trainable parameters",
        "Peak GPU memory MB",
        "Peak system memory MB",
        "Memory index size",
        "Training GPU hours",
        "Evidence Status",
    )
    guard = None if publication_status == FINAL_READY_STATUS else "N/R"
    rows = []
    for system_id in SYSTEM_IDS:
        values = metrics["systems"][system_id]
        rows.append(
            {
                "System": system_id,
                "Recovery@1": guard or _format_rate(values["recovery_at_1"]),
                f"Recovery@{recovery_k}": guard
                or _format_rate(values[f"recovery_at_{recovery_k}"]),
                "Verified-failure-only RSR": guard
                or _format_rate(values["verified_failure_only_recovery_success_rate"]),
                "Episode recovery success rate": guard
                or _format_rate(values["episode_recovery_success_rate"]),
                "False-trigger rate": guard
                or _format_rate(values["false_trigger_rate"]),
                "Unnecessary-intervention rate": guard
                or _format_rate(values["unnecessary_intervention_rate"]),
                "Repeated-error event rate": guard
                or _format_rate(values["repeated_error_event_rate"]),
                "Verified-failure incidence": guard
                or _format_rate(values["verified_failure_incidence"]),
                "Avg. recovery attempts/run": guard
                or _format_continuous(values["recovery_attempts_per_episode"]),
                "Avg. recovery attempts/failure episode": guard
                or _format_continuous(
                    values["recovery_attempts_per_failure_episode"]
                ),
                "Environment failure rate": guard
                or _format_rate(
                    metrics["environment_failure_rate"]["by_system"][system_id]
                ),
                "Task wall-clock seconds": guard
                or _format_continuous(values["efficiency"]["task_wall_clock_seconds"]),
                "Model call count": guard
                or _format_continuous(values["efficiency"]["model_call_count"]),
                "Decision latency ms": guard
                or _format_continuous(values["efficiency"]["decision_latency_ms"]),
                "Provider latency ms": guard
                or _format_continuous(values["efficiency"]["provider_latency_ms"]),
                "Recovery latency ms": guard
                or _format_continuous(values["efficiency"]["recovery_latency_ms"]),
                "Retrieval latency ms": guard
                or _format_continuous(values["efficiency"]["retrieval_latency_ms"]),
                "Input tokens": guard
                or _format_continuous(values["efficiency"]["input_token_count"]),
                "Output tokens": guard
                or _format_continuous(values["efficiency"]["output_token_count"]),
                "Model parameters": guard
                or _format_continuous(values["efficiency"]["model_parameter_count"]),
                "Trainable parameters": guard
                or _format_continuous(values["efficiency"]["trainable_parameter_count"]),
                "Peak GPU memory MB": guard
                or _format_continuous(values["efficiency"]["peak_gpu_memory_mb"]),
                "Peak system memory MB": guard
                or _format_continuous(values["efficiency"]["peak_system_memory_mb"]),
                "Memory index size": guard
                or _format_continuous(values["efficiency"]["memory_index_size"]),
                "Training GPU hours": guard
                or _format_continuous(values["efficiency"]["training_gpu_hours"]),
                "Evidence Status": publication_status,
            }
        )
    write_csv(aggregate / "table2_companion.csv", rows, fields)


def _format_rate(value: Mapping[str, Any]) -> str:
    """Render a rate with its exact counts and registered 95% interval."""

    estimate = value.get("estimate")
    if estimate is None:
        reason = str(value.get("reason", "")).strip()
        if not reason:
            return "N/A"
        return (
            f"N/A ({value.get('numerator', 0)}/{value.get('denominator', 0)}; "
            f"{reason})"
        )
    numerator = int(value["numerator"])
    denominator = int(value["denominator"])
    low = value.get("ci95_low", value.get("ci_low"))
    high = value.get("ci95_high", value.get("ci_high"))
    point = 100.0 * float(estimate)
    if low is None or high is None:
        reason = str(
            value.get("ci_reason", value.get("interval_status", "not estimable"))
        ).strip()
        return (
            f"{point:.2f}% ({numerator}/{denominator}; "
            f"95% CI N/A: {reason})"
        )
    return (
        f"{point:.2f}% ({numerator}/{denominator}; 95% CI "
        f"{100.0 * float(low):.2f}%, {100.0 * float(high):.2f}%)"
    )


def _format_continuous(value: Mapping[str, Any]) -> str:
    estimate = value.get("estimate", value.get("mean"))
    if estimate is None:
        reason = str(value.get("reason", "")).strip()
        return f"N/A ({reason})" if reason else "N/A"
    low = value.get("ci95_low", value.get("ci_low"))
    high = value.get("ci95_high", value.get("ci_high"))
    denominator = value.get("denominator", value.get("n"))
    numerator = value.get("numerator", value.get("total"))
    ratio = f"{float(numerator):.2f}/{int(denominator)}"
    if low is None or high is None:
        reason = str(
            value.get("ci_reason", value.get("interval_status", "not estimable"))
        ).strip()
        return (
            f"{float(estimate):.2f} ({ratio}; 95% CI N/A: {reason})"
        )
    return (
        f"{float(estimate):.2f} ({ratio}; 95% CI {float(low):.2f}, "
        f"{float(high):.2f})"
    )


def build_manual_audit_selection(
    campaign_dir: str | Path,
    records: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Create the deterministic post-completion, outcome-label-hidden sample."""

    root = Path(campaign_dir).resolve()
    raw_records = _load_selected_analysis_records_unchecked(root)
    if canonical_json_bytes(records) != canonical_json_bytes(raw_records):
        raise SchemaError(
            "manual-audit caller records differ from immutable selected raw evidence"
        )
    records = raw_records
    definition = read_json(root / "frozen" / "benchmark" / "audit_manifest.json")
    codebook_binding = validate_manual_audit_reviewer_codebook(definition)
    if (
        definition.get("manifest_id") != MANUAL_AUDIT_MANIFEST_ID
        or definition.get("blinding_mode") != MANUAL_AUDIT_BLINDING_MODE
        or definition.get("selection_occurs_after_episode_completion") is not True
    ):
        raise SchemaError(
            "manual audit definition has the wrong blinding mode/completion boundary"
        )
    strata_definition = definition.get("strata")
    if not isinstance(strata_definition, list):
        raise SchemaError("manual audit definition lacks strata")
    targets = {
        str(row["name"]): int(row["target"])
        for row in strata_definition
        if isinstance(row, Mapping)
    }
    registered_names = {
        "ordinary_success",
        "terminal_failure",
        "recovery",
        "e2_e3_disagreement",
    }
    if set(targets) != registered_names or any(value != 5 for value in targets.values()):
        raise SchemaError("manual audit must register four strata of five cases")
    required_categories = definition.get("required_case_categories")
    if not isinstance(required_categories, list) or tuple(required_categories) != (
        MANUAL_AUDIT_CASE_CATEGORIES
    ):
        raise SchemaError("manual audit required case categories differ from registration")

    derived = _derive_manual_audit_selection_evidence(records)
    by_episode = derived["episodes_by_id"]
    evidence_by_episode = derived["evidence_by_episode"]
    candidates = derived["candidates"]
    seed = int(definition["selection_seed"])
    campaign_id = str(read_json(root / "campaign_manifest.json")["campaign_id"])

    def score(stratum: str, episode_id: str) -> str:
        return sha256_json(
            {
                "algorithm": "sha256-lowest-v1",
                "campaign_id": campaign_id,
                "selection_seed": seed,
                "stratum": stratum,
                "episode_id": episode_id,
            }
        )

    # Rare/disagreement cases claim their audit unit first.  Remaining strata
    # never substitute for a shortfall and never duplicate a selected episode.
    priority = (
        "e2_e3_disagreement",
        "recovery",
        "terminal_failure",
        "ordinary_success",
    )
    used: set[str] = set()
    selected_by_stratum: dict[str, list[str]] = {}
    for stratum in priority:
        eligible = [episode for episode in candidates[stratum] if episode not in used]
        ordered = sorted(eligible, key=lambda episode: (score(stratum, episode), episode))
        selected = ordered[: targets[stratum]]
        selected_by_stratum[stratum] = selected
        used.update(selected)

    labeled: list[dict[str, Any]] = []
    for stratum, selected in selected_by_stratum.items():
        for episode_id in selected:
            episode = by_episode[episode_id]
            evidence = evidence_by_episode[episode_id]
            labeled.append(
                {
                    "episode_id": episode_id,
                    "block_id": str(episode["block_id"]),
                    "task_id": str(episode["task_id"]),
                    "system_id": str(episode["system_id"]),
                    "stratum": stratum,
                    **evidence,
                    "selection_score": score(stratum, episode_id),
                }
            )
    labeled.sort(
        key=lambda row: (
            score("audit-presentation-order", row["episode_id"]),
            row["episode_id"],
        )
    )
    audit_root = root / "manual_audit"
    reviewer_packet_root = audit_root / "reviewer_packets"
    if reviewer_packet_root.is_symlink():
        raise SchemaError("manual-audit reviewer packet root must not be a symlink")
    if reviewer_packet_root.exists():
        shutil.rmtree(reviewer_packet_root)
    reviewer_packet_root.mkdir(parents=True, exist_ok=True)
    public_selected: list[dict[str, Any]] = []
    for index, row in enumerate(labeled, start=1):
        audit_id = f"audit-{index:02d}"
        row["audit_id"] = audit_id
        primary_source = root / _episode_artifact_path(
            root, str(row["block_id"]), str(row["system_id"])
        )
        paired_source = (
            root / _episode_artifact_path(root, str(row["block_id"]), "E2")
            if row["paired_e2_episode_id"] is not None
            else None
        )
        packet_paths = _write_manual_audit_reviewer_packet(
            root=root,
            reviewer_packet_root=reviewer_packet_root,
            campaign_id=campaign_id,
            audit_id=audit_id,
            primary_source=primary_source,
            primary_episode_id=str(row["episode_id"]),
            primary_system_id=str(row["system_id"]),
            task_id=str(row["task_id"]),
            paired_e2_source=paired_source,
            paired_e2_episode_id=row["paired_e2_episode_id"],
        )
        public_selected.append(
            {
                "audit_id": audit_id,
                "episode_id": row["episode_id"],
                "block_id": row["block_id"],
                "task_id": row["task_id"],
                "system_id": row["system_id"],
                **packet_paths,
            }
        )

    sealed_root = audit_root / "sealed"
    sealed_root.mkdir(parents=True, exist_ok=True)
    os.chmod(sealed_root, 0o700)
    labels_path = sealed_root / "selection_labels.json"
    labels_payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": definition["manifest_id"],
        "campaign_id": campaign_id,
        **codebook_binding,
        "offline_sealed_labels": True,
        "labels": labeled,
    }
    atomic_write_json(labels_path, labels_payload, mode=0o600)
    strata_rows = []
    for row in strata_definition:
        name = str(row["name"])
        achieved = len(selected_by_stratum[name])
        strata_rows.append(
            {
                "name": name,
                "target": targets[name],
                "eligible_count_before_cross_stratum_deduplication": len(
                    set(candidates[name])
                ),
                "selected_count": achieved,
                "shortfall": targets[name] - achieved,
                "substitution_allowed": False,
            }
        )
    selection = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": definition["manifest_id"],
        "campaign_id": campaign_id,
        **codebook_binding,
        "selection_algorithm": "sha256-lowest-v1",
        "selection_seed": seed,
        "blinding_mode": MANUAL_AUDIT_BLINDING_MODE,
        "outcome_labels_location": "sealed/selection_labels.json",
        "outcome_labels_sha256": sha256_file(labels_path),
        "total_target": int(definition["total_target"]),
        "selected_count": len(public_selected),
        "total_shortfall": int(definition["total_target"]) - len(public_selected),
        "strata": strata_rows,
        "category_coverage": [
            {
                "name": category,
                "eligible_count": sum(
                    category in values
                    for values in derived["categories_by_episode"].values()
                ),
                "selected_count": sum(
                    category in set(row["case_categories"]) for row in labeled
                ),
                "status": (
                    "NOT_APPLICABLE"
                    if not any(
                        category in values
                        for values in derived["categories_by_episode"].values()
                    )
                    else "COVERED"
                    if any(category in set(row["case_categories"]) for row in labeled)
                    else "SHORTFALL_NO_SUBSTITUTION"
                ),
                "substitution_allowed": False,
            }
            for category in MANUAL_AUDIT_CASE_CATEGORIES
        ],
        "selected": public_selected,
    }
    atomic_write_json(audit_root / "selection_manifest.json", selection)
    return selection


def _manual_audit_categories(
    episodes: list[dict[str, Any]],
    *,
    recovery_attempts: list[dict[str, Any]],
    failure_incidents: list[dict[str, Any]],
    memory_queries: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Derive evidence-backed case categories for compatibility fixtures.

    Production audit selection does not trust these caller-supplied
    projections: ``build_manual_audit_selection`` reopens the immutable
    campaign records through ``_derive_manual_audit_selection_evidence``. This
    helper retains the existing pure category-contract API and cannot
    authorize a selected audit case.
    """

    categories = {str(row["episode_id"]): set() for row in episodes}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if strict_bool(
            episode["environment_failure"], context="audit.environment_failure"
        ):
            categories[episode_id].add("environment_failure")
        if strict_bool(episode["loop_detected"], context="audit.loop_detected"):
            categories[episode_id].add("loop")
    for attempt in recovery_attempts:
        episode_id = str(attempt.get("episode_id", ""))
        if episode_id not in categories:
            continue
        successful = strict_bool(
            attempt.get("successful", False), context="audit.recovery_successful"
        )
        verified_failure = strict_bool(
            attempt.get("verified_failure_present", False),
            context="audit.verified_failure_present",
        )
        categories[episode_id].add(
            "successful_recovery" if successful else "failed_recovery"
        )
        if not verified_failure:
            categories[episode_id].add("unnecessary_intervention")
    for query in memory_queries:
        episode_id = str(query.get("episode_id", ""))
        if episode_id not in categories:
            continue
        if query.get("useful_intervention") is True:
            categories[episode_id].add("memory_help")
        if query.get("harmful_intervention") is True:
            categories[episode_id].add("memory_harm")
    failure_markers = ("BBOX", "GROUND", "TARGET", "PARAMETER", "INVALID_ACTION")
    for incident in failure_incidents:
        episode_id = str(incident.get("episode_id", ""))
        if episode_id not in categories:
            continue
        descriptor = " ".join(
            str(incident.get(key, ""))
            for key in ("failure_type", "failure_kind", "diagnosis", "error_kind")
        ).upper()
        if any(marker in descriptor for marker in failure_markers):
            categories[episode_id].add("bbox_or_parameter_failure")
    return categories


def _episode_artifact_path(root: Path, block_id: str, system_id: str) -> str:
    matches: list[Path] = []
    for resolution_path in (root / "paired_blocks").glob("seed_*/*/repeat_*/resolution.json"):
        resolution = read_json(resolution_path)
        if str(resolution.get("block_id")) != block_id or resolution.get("status") != "INCLUDED":
            continue
        selected = int(resolution["selected_attempt_id"])
        matches.append(
            resolution_path.parent / f"rerun_{selected}" / system_id / "runtime"
        )
    if len(matches) != 1:
        raise SchemaError(f"manual audit episode does not resolve to one package: {block_id}/{system_id}")
    return str(matches[0].relative_to(root))


def _write_manual_audit_reviewer_packet(
    *,
    root: Path,
    reviewer_packet_root: Path,
    campaign_id: str,
    audit_id: str,
    primary_source: Path,
    primary_episode_id: str,
    primary_system_id: str,
    task_id: str,
    paired_e2_source: Path | None,
    paired_e2_episode_id: Any,
) -> dict[str, str | None]:
    """Copy only runtime-side evidence into a reviewer packet.

    The source campaign has already passed runtime/oracle-separation validation.
    Copying into a dedicated packet prevents the public selection from pointing
    at an ``E*/`` directory whose sibling ``sealed/`` tree contains official
    outcome/evaluator labels.
    """

    packet_root = reviewer_packet_root / audit_id
    resolved_packet_parent = reviewer_packet_root.resolve()
    if packet_root.resolve().parent != resolved_packet_parent:
        raise SchemaError("manual-audit reviewer packet escaped its registered root")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{audit_id}.", dir=reviewer_packet_root)
    )

    def copy_runtime(
        source: Path,
        destination_name: str,
        *,
        expected_episode_id: str,
    ) -> dict[str, str]:
        if source.is_symlink() or source.absolute() != source.resolve():
            raise SchemaError("manual-audit packet source traverses a symlink")
        source = source.resolve()
        if source.name != "runtime" or not source.is_dir():
            raise SchemaError("manual-audit packet source is not a runtime directory")
        summary = read_json(source / "episode_summary.json")
        if summary.get("episode_id") != expected_episode_id:
            raise SchemaError("manual-audit packet source has the wrong episode ID")
        assert_no_verifier_evidence(
            summary, context=str(source / "episode_summary.json")
        )
        destination = temporary / destination_name
        destination.mkdir()
        copied: dict[str, str] = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise SchemaError("manual-audit runtime evidence contains a symlink")
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if "sealed" in {part.lower() for part in relative.parts}:
                raise SchemaError("manual-audit runtime evidence contains a sealed path")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if target.suffix == ".json":
                value = _load_audit_json_without_duplicate_keys(
                    target.read_text(encoding="utf-8"), context=str(target)
                )
                assert_no_verifier_evidence(
                    {"packet_value": value}, context=str(target)
                )
            elif target.suffix == ".jsonl":
                with target.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        value = _load_audit_json_without_duplicate_keys(
                            line, context=f"{target}:{line_number}"
                        )
                        assert_no_verifier_evidence(
                            {"packet_value": value},
                            context=f"{target}:{line_number}",
                        )
            copied[str(relative)] = sha256_file(target)
        if not copied:
            raise SchemaError("manual-audit reviewer packet would be empty")
        return copied

    try:
        task_context = _manual_audit_task_context(root, task_id=task_id)
        atomic_write_json(temporary / "task_context.json", task_context, mode=0o600)
        task_context_sha256 = sha256_file(temporary / "task_context.json")
        primary_files = copy_runtime(
            primary_source,
            "primary_runtime",
            expected_episode_id=primary_episode_id,
        )
        paired_files: dict[str, str] | None = None
        if paired_e2_source is not None:
            if type(paired_e2_episode_id) is not str or not paired_e2_episode_id:
                raise SchemaError("manual-audit paired E2 identity is invalid")
            paired_files = copy_runtime(
                paired_e2_source,
                "paired_e2_runtime",
                expected_episode_id=paired_e2_episode_id,
            )
        elif paired_e2_episode_id is not None:
            raise SchemaError("manual-audit paired E2 source/identity disagree")
        manifest = {
            "schema_version": "table2-manual-audit-reviewer-packet-v1",
            "campaign_id": campaign_id,
            "audit_id": audit_id,
            "blinding_mode": MANUAL_AUDIT_BLINDING_MODE,
            "system_condition_visible": True,
            "official_outcome_labels_included": False,
            "sealed_evaluator_files_included": False,
            "task_context": {
                "task_id": task_id,
                "packet_relative_path": "task_context.json",
                "sha256": task_context_sha256,
                "context_status": task_context["context_status"],
                "source_task_snapshot_sha256": task_context[
                    "source_task_snapshot_sha256"
                ],
                "source_task_record_sha256": task_context[
                    "source_task_record_sha256"
                ],
            },
            "primary": {
                "episode_id": primary_episode_id,
                "system_id": primary_system_id,
                "packet_relative_path": "primary_runtime",
                "source_runtime_relative_path": str(primary_source.relative_to(root)),
                "files": primary_files,
            },
            "paired_e2": (
                {
                    "episode_id": paired_e2_episode_id,
                    "system_id": "E2",
                    "packet_relative_path": "paired_e2_runtime",
                    "source_runtime_relative_path": str(
                        paired_e2_source.relative_to(root)
                    ),
                    "files": paired_files,
                }
                if paired_e2_source is not None
                else None
            ),
        }
        atomic_write_json(temporary / "packet_manifest.json", manifest, mode=0o600)
        if packet_root.exists():
            shutil.rmtree(packet_root)
        os.replace(temporary, packet_root)
        os.chmod(packet_root, 0o700)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "task_context_path": str((packet_root / "task_context.json").relative_to(root)),
        "artifact_path": str(
            (packet_root / "primary_runtime").relative_to(root)
        ),
        "paired_e2_artifact_path": (
            str((packet_root / "paired_e2_runtime").relative_to(root))
            if paired_e2_source is not None
            else None
        ),
    }


def _manual_audit_task_context(root: Path, *, task_id: str) -> dict[str, Any]:
    """Project one frozen task into a reviewer-safe, oracle-free context."""

    candidates = sorted((root / "frozen").glob("task_manifest.*"))
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise SchemaError("manual-audit task context lacks one frozen task snapshot")
    source = candidates[0]
    payload = _load_audit_json_without_duplicate_keys(
        source.read_text(encoding="utf-8"), context=str(source)
    )
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise SchemaError("manual-audit frozen task snapshot is malformed")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise SchemaError("manual-audit task context does not resolve one task")
    task = dict(matches[0])
    instruction = task.get("instruction")
    start_state = task.get("start_state")
    campaign_mode = read_json(root / "campaign_manifest.json").get("campaign_mode")
    if type(instruction) is str and instruction.strip() and isinstance(
        start_state, Mapping
    ):
        observable_start_context = {
            "sites": start_state.get("sites"),
            "start_url": start_state.get("start_url"),
            "geolocation": start_state.get("geolocation"),
            "require_login": start_state.get("require_login"),
            "require_reset": start_state.get("require_reset"),
        }
        context_status = "AVAILABLE"
    elif campaign_mode == "smoke":
        instruction = None
        observable_start_context = None
        context_status = "UNAVAILABLE_ENGINEERING_SMOKE_ONLY"
    else:
        raise SchemaError(
            "manual-audit evaluation task lacks reviewer-safe instruction/start context"
        )
    value = {
        "schema_version": "table2-manual-audit-task-context-v1",
        "task_id": task_id,
        "context_status": context_status,
        "instruction": instruction,
        "observable_start_context": observable_start_context,
        "excluded_source_fields": [
            "evaluator",
            "eval",
            "reference_answer",
            "reference_answers",
            "task_config",
            "storage_state",
        ],
        "source_task_snapshot_sha256": sha256_file(source),
        "source_task_record_sha256": sha256_json(task),
    }
    assert_no_verifier_evidence({"task_context": value}, context="task_context.json")
    return value


def _resolve_results_dir(
    root: Path,
    *,
    campaign_id: str,
    requested: str | Path | None,
    draft_pilot: bool = False,
) -> Path:
    if requested is not None:
        return Path(requested).resolve()
    if root.parent.name == "table2" and root.parent.parent.name == "artifacts":
        repository = root.parent.parent.parent
    else:
        repository = root.parent
    destination = repository / "results" / "table2" / campaign_id
    # Preserve immutable draft evidence while leaving the canonical campaign
    # directory free for the later adjudication-gated PILOT_ONLY export.
    if draft_pilot:
        destination = destination / "draft"
    return destination.resolve()


def _export_results(
    root: Path,
    results_dir: Path,
    *,
    metrics: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    statistics: Mapping[str, Any],
    validation: Mapping[str, Any],
    audit_selection: Mapping[str, Any],
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    campaign = read_json(root / "campaign_manifest.json")
    publication_status = str(metrics["publication_status"])
    pilot_summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "evidence_label": campaign["evidence_label"],
        "publication_status": publication_status,
        "paper_table_status": metrics["paper_table_status"],
        "headline_task_partition": "normal",
        "normal_block_count": campaign["normal_block_count"],
        "recovery_diagnostic_block_count": campaign["recovery_block_count"],
        "manual_audit_selected_count": audit_selection["selected_count"],
        "manual_audit_shortfall": audit_selection["total_shortfall"],
        "validation_status": validation["status"],
    }
    _write_result_json(results_dir / "pilot_summary.json", pilot_summary)
    _write_result_bytes(
        results_dir / "table2_main.csv", (root / "aggregate" / "table2_main.csv").read_bytes()
    )
    _write_result_bytes(
        results_dir / "table2_companion.csv",
        (root / "aggregate" / "table2_companion.csv").read_bytes(),
    )
    _write_result_bytes(
        results_dir / "metrics.csv", (root / "aggregate" / "metrics.csv").read_bytes()
    )
    _write_result_bytes(
        results_dir / "metrics.json", (root / "aggregate" / "metrics.json").read_bytes()
    )

    contrast_rows = _paired_contrast_rows(
        statistics, publication_status=publication_status
    )
    _write_result_csv(
        results_dir / "paired_contrasts.csv",
        contrast_rows,
        PAIRED_CONTRAST_FIELDS,
    )

    retrieval_rows = _retrieval_metric_rows(
        retrieval,
        publication_status=publication_status,
    )
    _write_result_csv(
        results_dir / "retrieval_metrics.csv",
        retrieval_rows,
        RETRIEVAL_METRIC_FIELDS,
    )
    _write_result_json(
        results_dir / "statistics.json",
        _redacted_statistics_for_results(statistics),
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "publication_status": publication_status,
        "repository_commit": campaign["repository_commit"],
        "campaign_manifest_sha256": sha256_file(root / "campaign_manifest.json"),
        "frozen_artifact_hashes_sha256": sha256_file(root / "artifact_hashes.json"),
        "environment_sha256": sha256_file(root / "frozen" / "environment.json"),
        "protocol_sha256": sha256_file(root / "frozen" / "protocol.yaml"),
        "analysis_source_identity_sha256": sha256_file(
            root / "aggregate" / "analysis_source_identity.json"
        ),
        "contains_raw_oracle_evidence": False,
    }
    _write_result_json(results_dir / "provenance.json", provenance)
    names = (
        "pilot_summary.json",
        "table2_main.csv",
        "table2_companion.csv",
        "metrics.csv",
        "metrics.json",
        "paired_contrasts.csv",
        "retrieval_metrics.csv",
        "statistics.json",
        "provenance.json",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign["campaign_id"],
        "results_dir": str(results_dir),
        "files": {name: sha256_file(results_dir / name) for name in names},
        "raw_evidence_exported": False,
    }


def _retrieval_metric_rows(
    retrieval: Mapping[str, Any],
    *,
    publication_status: str,
) -> list[dict[str, Any]]:
    """Flatten retrieval metrics without dropping registered uncertainty fields."""

    rows: list[dict[str, Any]] = []
    for metric, values in retrieval.items():
        if not isinstance(values, Mapping) or not (
            "estimate" in values or "mean" in values
        ):
            continue
        low = values.get("ci95_low", values.get("ci_low"))
        high = values.get("ci95_high", values.get("ci_high"))
        rows.append(
            {
                "metric": metric,
                "numerator": values.get("numerator", values.get("total")),
                "denominator": values.get(
                    "denominator", values.get("query_denominator", values.get("n"))
                ),
                "estimate": values.get("estimate", values.get("mean")),
                "ci95_low": low,
                "ci95_high": high,
                "confidence": values.get("confidence"),
                "interval_method": values.get("interval_method"),
                "inference_unit": values.get("inference_unit"),
                "n_task_clusters": values.get("n_task_clusters"),
                "valid_bootstrap_samples": values.get("valid_bootstrap_samples"),
                "interval_status": values.get("interval_status"),
                "ci_reason": values.get("ci_reason"),
                "reason": values.get("reason"),
                "query_denominator": values.get("query_denominator"),
                "available_query_count": values.get("available_query_count"),
                "display": values.get("display"),
                "publication_status": publication_status,
            }
        )
    return rows


def _paired_contrast_rows(
    statistics: Mapping[str, Any],
    *,
    publication_status: str,
) -> list[dict[str, Any]]:
    """Flatten every registered paired analysis without dropping inference."""

    confidence = statistics.get("confidence")
    bootstrap_unit = statistics.get("bootstrap_unit", "task_id")
    rows: list[dict[str, Any]] = []
    contrasts = statistics.get("contrasts")
    if not isinstance(contrasts, Mapping):
        raise SchemaError("statistics lacks registered paired contrasts")
    for contrast, metric_values in contrasts.items():
        if not isinstance(metric_values, Mapping):
            raise SchemaError(f"statistics contrast is malformed: {contrast}")
        for metric, values in metric_values.items():
            if not isinstance(values, Mapping) or "estimate" not in values:
                continue
            significance = values.get("task_clustered_significance")
            if not isinstance(significance, Mapping):
                significance = {}
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "analysis_family": "paired_mean_difference",
                    "estimate": values.get("estimate"),
                    "ci_low": values.get("ci_low"),
                    "ci_high": values.get("ci_high"),
                    "absolute_change": values.get("absolute_change"),
                    "relative_change": values.get("relative_change"),
                    "left_estimate": values.get("left_mean"),
                    "right_estimate": values.get("right_mean"),
                    "numerator": None,
                    "denominator": None,
                    "n_clusters": values.get("n_clusters"),
                    "n_rows": values.get("n_rows"),
                    "exact_sign_two_sided_p": significance.get(
                        "exact_sign_two_sided_p"
                    ),
                    "holm_adjusted_p": significance.get("holm_adjusted_p"),
                    "right_better_task_clusters": significance.get(
                        "right_better_task_clusters"
                    ),
                    "left_better_task_clusters": significance.get(
                        "left_better_task_clusters"
                    ),
                    "tied_task_clusters": significance.get("tied_task_clusters"),
                    "non_tied_task_clusters": significance.get(
                        "non_tied_task_clusters"
                    ),
                    "interval_method": "percentile_task_cluster_bootstrap",
                    "inference_unit": bootstrap_unit,
                    "confidence": confidence,
                    "publication_status": publication_status,
                }
            )

    ratio_contrasts = statistics.get("clustered_ratio_contrasts", {})
    if not isinstance(ratio_contrasts, Mapping):
        raise SchemaError("statistics clustered ratio contrasts are malformed")
    for contrast, metric_values in ratio_contrasts.items():
        if not isinstance(metric_values, Mapping):
            raise SchemaError(f"ratio contrast is malformed: {contrast}")
        for metric, values in metric_values.items():
            if not isinstance(values, Mapping) or "estimate" not in values:
                raise SchemaError(
                    f"ratio contrast metric is malformed: {contrast}.{metric}"
                )
            rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "analysis_family": "paired_ratio_difference",
                    "estimate": values.get("estimate"),
                    "ci_low": values.get("ci_low"),
                    "ci_high": values.get("ci_high"),
                    "absolute_change": values.get("absolute_change"),
                    "relative_change": values.get("relative_change"),
                    "left_estimate": values.get("left_ratio"),
                    "right_estimate": values.get("right_ratio"),
                    "numerator": None,
                    "denominator": None,
                    "n_clusters": values.get("n_clusters"),
                    "n_rows": None,
                    "exact_sign_two_sided_p": None,
                    "holm_adjusted_p": None,
                    "right_better_task_clusters": None,
                    "left_better_task_clusters": None,
                    "tied_task_clusters": None,
                    "non_tied_task_clusters": None,
                    "interval_method": "percentile_task_cluster_bootstrap",
                    "inference_unit": bootstrap_unit,
                    "confidence": confidence,
                    "publication_status": publication_status,
                }
            )

    memory_regression = statistics.get("paired_memory_regression_rate")
    if not isinstance(memory_regression, Mapping) or "estimate" not in memory_regression:
        raise SchemaError("statistics lacks paired memory-regression rate")
    rows.append(
        {
            "contrast": "E3_minus_E2",
            "metric": "paired_memory_regression_rate",
            "analysis_family": "task_level_harm_rate",
            "estimate": memory_regression.get("estimate"),
            "ci_low": memory_regression.get(
                "ci_low", memory_regression.get("ci95_low")
            ),
            "ci_high": memory_regression.get(
                "ci_high", memory_regression.get("ci95_high")
            ),
            "absolute_change": None,
            "relative_change": None,
            "left_estimate": None,
            "right_estimate": None,
            "numerator": memory_regression.get("numerator"),
            "denominator": memory_regression.get("denominator"),
            "n_clusters": memory_regression.get("n_task_clusters"),
            "n_rows": None,
            "exact_sign_two_sided_p": None,
            "holm_adjusted_p": None,
            "right_better_task_clusters": None,
            "left_better_task_clusters": None,
            "tied_task_clusters": None,
            "non_tied_task_clusters": None,
            "interval_method": "wilson_score_95",
            "inference_unit": memory_regression.get("unit", "task_id"),
            "confidence": 0.95,
            "publication_status": publication_status,
        }
    )
    return rows


def _write_result_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _write_result_bytes(path, payload)


def _redacted_statistics_for_results(
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove per-task identities from the small, shareable result package."""

    output = json.loads(
        json.dumps(statistics, ensure_ascii=False, allow_nan=False)
    )
    contrasts = output.get("contrasts", {})
    if isinstance(contrasts, dict):
        for metrics in contrasts.values():
            if not isinstance(metrics, dict):
                continue
            task_success = metrics.get("task_success")
            if not isinstance(task_success, dict):
                continue
            significance = task_success.get("task_clustered_significance")
            if isinstance(significance, dict):
                significance.pop("task_effects", None)
                significance["per_task_effects_exported"] = False
    return output


def _write_result_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _write_result_bytes(path, output.getvalue().encode("utf-8"))


def _write_result_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise Table2Error(f"refusing to overwrite an existing result artifact: {path}")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_aggregate_hashes(aggregate: Path) -> None:
    files = {
        str(path.relative_to(aggregate)): sha256_file(path)
        for path in sorted(aggregate.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_hashes.json", "validation_report.json"}
    }
    atomic_write_json(
        aggregate / "artifact_hashes.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "excluded_mutable_files": ["validation_report.json"],
            "files": files,
        },
    )


def _write_campaign_evidence_manifest(root: Path) -> None:
    files: dict[str, str] = {}
    for directory_name in (
        "paired_blocks",
        "manual_audit",
        "runtime_readiness",
        "aggregate",
    ):
        directory = root / directory_name
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if relative in {
                "aggregate/validation_report.json",
            }:
                continue
            files[relative] = sha256_file(path)
    completion = root / "completion.json"
    if completion.is_file():
        files["completion.json"] = sha256_file(completion)
    atomic_write_json(
        root / "campaign_evidence_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "campaign_manifest_sha256": sha256_file(root / "campaign_manifest.json"),
            "frozen_artifact_hashes_sha256": sha256_file(root / "artifact_hashes.json"),
            "access_ledger_sha256": sha256_file(root / "access_ledger.jsonl"),
            "deviation_ledger_sha256": sha256_file(root / "deviation_ledger.jsonl"),
            "files": files,
        },
    )
