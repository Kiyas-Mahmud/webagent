"""Registered Table 2 metric definitions with explicit denominators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
import math
import random
from statistics import mean, median, pstdev
from typing import Any

from .common import SCHEMA_VERSION, SchemaError, stable_int_seed, strict_bool
from .schedule import SYSTEM_IDS


RECOVERY_DISABLED_SYSTEMS = frozenset({"E0", "E1"})


def rate(numerator: int, denominator: int, *, force_na: bool = False) -> dict[str, Any]:
    """Return a JSON-safe rate with Wilson 95% interval and N/A semantics."""

    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError(f"invalid rate {numerator}/{denominator}")
    if force_na or denominator == 0:
        return {
            "numerator": int(numerator),
            "denominator": int(denominator),
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "display": "N/A",
        }
    estimate = numerator / denominator
    low, high = _wilson_interval(numerator, denominator)
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "estimate": float(estimate),
        "ci95_low": low,
        "ci95_high": high,
        "display": f"{100.0 * estimate:.2f}% ({numerator}/{denominator})",
    }


def task_clustered_rate(
    contributions: Iterable[Mapping[str, Any]],
    *,
    metric_name: str,
    system_id: str,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20250831,
    force_na: bool = False,
) -> dict[str, Any]:
    """Estimate one pooled rate with task-cluster bootstrap uncertainty.

    The estimand is the ratio of numerator and denominator totals over unique
    registered ``matched_model_seed`` x ``repeat_id`` cells.  Every task is a
    bootstrap cluster: resampling a task retains all of that task's unique
    cells and all attempt/incident counts summarized inside each cell.  Exact
    duplicate cells are collapsed and conflicting duplicates fail closed.
    Thus repeated measurements contribute to the point estimand without being
    treated as independent task replicates for interval estimation.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not metric_name or system_id not in SYSTEM_IDS:
        raise ValueError("clustered rate requires a metric and registered system")

    tasks: dict[str, dict[tuple[Any, ...], tuple[int, int]]] = defaultdict(dict)
    duplicate_cells = 0
    for raw in contributions:
        task_id = str(raw.get("task_id") or "")
        if not task_id:
            raise SchemaError(f"{metric_name} contribution lacks task_id")
        cell = raw.get("cell")
        if not isinstance(cell, tuple) or not cell:
            raise SchemaError(f"{metric_name} contribution lacks a cell key")
        numerator = _nonnegative_int(
            raw.get("numerator"), f"{metric_name}.numerator"
        )
        denominator = _nonnegative_int(
            raw.get("denominator"), f"{metric_name}.denominator"
        )
        if numerator > denominator:
            raise SchemaError(f"{metric_name} numerator exceeds denominator")
        outcome = (numerator, denominator)
        previous = tasks[task_id].get(cell)
        if previous is None:
            tasks[task_id][cell] = outcome
        elif previous != outcome:
            raise SchemaError(
                "conflicting rate contributions for duplicate seed/repeat cell "
                f"{metric_name}.{system_id}.{task_id}.{cell}"
            )
        else:
            duplicate_cells += 1

    task_ids = sorted(tasks)
    if not task_ids:
        raise SchemaError(f"{metric_name} has no task clusters")

    task_totals = {
        task_id: (
            sum(value[0] for value in tasks[task_id].values()),
            sum(value[1] for value in tasks[task_id].values()),
        )
        for task_id in task_ids
    }
    numerator = sum(value[0] for value in task_totals.values())
    denominator = sum(value[1] for value in task_totals.values())
    if numerator > denominator:
        raise SchemaError(f"{metric_name} numerator exceeds denominator")

    derived_seed = stable_int_seed(seed, "per_system_rate", system_id, metric_name)
    common = {
        "numerator": numerator,
        "denominator": denominator,
        "n_task_clusters": len(task_ids),
        "denominator_task_clusters": sum(
            task_denominator > 0
            for _, task_denominator in task_totals.values()
        ),
        "unique_seed_repeat_cells": sum(len(cells) for cells in tasks.values()),
        "duplicate_seed_repeat_cells_collapsed": duplicate_cells,
        "confidence": confidence,
        "interval_method": "percentile_task_cluster_bootstrap",
        "inference_unit": "task_id",
        "estimand": "pooled_ratio_over_unique_registered_seed_repeat_cells",
        "within_task_aggregation": (
            "retain_all_unique_matched_seed_repeat_cells_and_linked_counts"
        ),
        "bootstrap_samples": samples,
        "bootstrap_seed": derived_seed,
    }
    if force_na or denominator == 0:
        return {
            **common,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_bootstrap_samples": 0,
            "interval_status": "NOT_APPLICABLE",
            "display": "N/A",
        }

    estimate = numerator / denominator
    if len(task_ids) < 2:
        return {
            **common,
            "estimate": float(estimate),
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_bootstrap_samples": 0,
            "interval_status": "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS",
            "display": f"{100.0 * estimate:.2f}% ({numerator}/{denominator})",
        }
    rng = random.Random(derived_seed)
    estimates: list[float] = []
    for _ in range(samples):
        chosen = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        sampled_numerator = sum(task_totals[task_id][0] for task_id in chosen)
        sampled_denominator = sum(task_totals[task_id][1] for task_id in chosen)
        if sampled_denominator:
            estimates.append(sampled_numerator / sampled_denominator)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low = _quantile(estimates, alpha) if estimates else None
    high = _quantile(estimates, 1.0 - alpha) if estimates else None
    return {
        **common,
        "estimate": float(estimate),
        "ci_low": low,
        "ci_high": high,
        # Retain the existing field names for the frozen 0.95 protocol.
        "ci95_low": low if confidence == 0.95 else None,
        "ci95_high": high if confidence == 0.95 else None,
        "valid_bootstrap_samples": len(estimates),
        "interval_status": "ESTIMATED" if estimates else "NO_VALID_BOOTSTRAP_RATIOS",
        "display": f"{100.0 * estimate:.2f}% ({numerator}/{denominator})",
    }


def describe(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "n": 0,
            "total": 0.0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }
    q1 = _quantile(ordered, 0.25)
    q3 = _quantile(ordered, 0.75)
    return {
        "n": len(ordered),
        "total": float(sum(ordered)),
        "mean": float(mean(ordered)),
        "std": float(pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "median": float(median(ordered)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def task_clustered_mean(
    contributions: Iterable[Mapping[str, Any]],
    *,
    metric_name: str,
    system_id: str,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20250831,
    force_na: bool = False,
) -> dict[str, Any]:
    """Return a mean and task-cluster bootstrap interval.

    Each contribution is one registered seed/repeat cell. Exact duplicate
    cells are collapsed, conflicting copies fail closed, and a bootstrap draw
    resamples complete tasks so repeated cells from one task are never treated
    as independent tasks. ``numerator`` is the finite value total and
    ``denominator`` is the number of unique contributing cells.
    """

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not metric_name or system_id not in SYSTEM_IDS:
        raise ValueError("clustered mean requires a metric and registered system")

    tasks: dict[str, dict[tuple[Any, ...], float | None]] = defaultdict(dict)
    duplicate_cells = 0
    for raw in contributions:
        task_id = str(raw.get("task_id") or "")
        if not task_id:
            raise SchemaError(f"{metric_name} contribution lacks task_id")
        cell = raw.get("cell")
        if not isinstance(cell, tuple) or not cell:
            raise SchemaError(f"{metric_name} contribution lacks a cell key")
        raw_value = raw.get("value")
        value = (
            None
            if raw_value is None
            else _nonnegative_float(raw_value, f"{metric_name}.value")
        )
        if cell not in tasks[task_id]:
            tasks[task_id][cell] = value
        elif tasks[task_id][cell] != value:
            raise SchemaError(
                "conflicting continuous contributions for duplicate "
                f"seed/repeat cell {metric_name}.{system_id}.{task_id}.{cell}"
            )
        else:
            duplicate_cells += 1

    registered_task_ids = sorted(tasks)
    task_values = {
        task_id: tuple(
            value
            for cell in sorted(tasks[task_id])
            if (value := tasks[task_id][cell]) is not None
        )
        for task_id in registered_task_ids
    }
    task_ids = [task_id for task_id in registered_task_ids if task_values[task_id]]
    ordered = sorted(
        value for task_id in task_ids for value in task_values[task_id]
    )
    descriptive = describe(ordered)
    derived_seed = stable_int_seed(seed, "per_system_mean", system_id, metric_name)
    common = {
        **descriptive,
        "numerator": descriptive["total"],
        "denominator": descriptive["n"],
        "estimate": descriptive["mean"],
        "n_task_clusters": len(task_ids),
        "registered_task_clusters": len(registered_task_ids),
        "unique_seed_repeat_cells": len(ordered),
        "missing_seed_repeat_cells": sum(
            value is None for cells in tasks.values() for value in cells.values()
        ),
        "duplicate_seed_repeat_cells_collapsed": duplicate_cells,
        "confidence": confidence,
        "interval_method": "percentile_task_cluster_bootstrap",
        "inference_unit": "task_id",
        "estimand": "mean_over_unique_registered_seed_repeat_cells",
        "within_task_aggregation": (
            "retain_all_unique_matched_seed_repeat_cells"
        ),
        "bootstrap_samples": samples,
        "bootstrap_seed": derived_seed,
    }
    if force_na:
        return {
            **common,
            "n": 0,
            "total": 0.0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "min": None,
            "max": None,
            "numerator": 0.0,
            "denominator": 0,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_bootstrap_samples": 0,
            "interval_status": "NOT_APPLICABLE",
            "display": "N/A",
        }
    if not ordered:
        return {
            **common,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_bootstrap_samples": 0,
            "interval_status": "NOT_APPLICABLE",
            "display": "N/A",
        }
    if len(task_ids) < 2:
        return {
            **common,
            "ci_low": None,
            "ci_high": None,
            "ci95_low": None,
            "ci95_high": None,
            "valid_bootstrap_samples": 0,
            "interval_status": "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS",
            "display": f"{descriptive['mean']:.2f} ({descriptive['total']:.2f}/{descriptive['n']})",
        }

    rng = random.Random(derived_seed)
    estimates: list[float] = []
    for _ in range(samples):
        chosen = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        sampled_values = [
            value for task_id in chosen for value in task_values[task_id]
        ]
        if sampled_values:
            estimates.append(sum(sampled_values) / len(sampled_values))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low = _quantile(estimates, alpha) if estimates else None
    high = _quantile(estimates, 1.0 - alpha) if estimates else None
    return {
        **common,
        "ci_low": low,
        "ci_high": high,
        "ci95_low": low if confidence == 0.95 else None,
        "ci95_high": high if confidence == 0.95 else None,
        "valid_bootstrap_samples": len(estimates),
        "interval_status": "ESTIMATED" if estimates else "NO_VALID_BOOTSTRAP_MEANS",
        "display": f"{descriptive['mean']:.2f} ({descriptive['total']:.2f}/{descriptive['n']})",
    }


def compute_table2_metrics(
    episodes: Iterable[Mapping[str, Any]],
    *,
    recovery_attempts: Iterable[Mapping[str, Any]] | None = None,
    failure_incidents: Iterable[Mapping[str, Any]] | None = None,
    schedule_attempts: Iterable[Mapping[str, Any]] | None = None,
    recovery_k: int = 2,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20250831,
) -> dict[str, Any]:
    """Compute primary and companion metrics from selected paired blocks.

    ``episodes`` must contain only the one selected valid attempt for every
    included block.  The function rejects incomplete E0--E3 blocks rather than
    silently changing a denominator.  Oracle-derived fields should be joined
    by the analysis loader from ``sealed/`` before calling this function.
    """

    if recovery_k < 1:
        raise ValueError("recovery_k must be at least one")
    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid per-system rate bootstrap configuration")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative exact integer")
    rows = [_normalize_episode(row) for row in episodes]
    _assert_complete_paired_blocks(rows)

    attempt_rows = [dict(row) for row in (recovery_attempts or _embedded(rows, "recovery_attempts"))]
    incident_rows = [dict(row) for row in (failure_incidents or _embedded(rows, "failure_incidents"))]
    episode_system = {row["episode_id"]: row["system_id"] for row in rows}
    attempt_ids: set[str] = set()
    incident_ids: set[tuple[str, str]] = set()
    attempts_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incidents_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in attempt_rows:
        system = str(item.get("system_id", ""))
        if system not in SYSTEM_IDS:
            raise SchemaError(f"recovery attempt has invalid system_id: {system!r}")
        episode_id = str(item.get("episode_id", ""))
        if episode_system.get(episode_id) != system:
            raise SchemaError(f"orphan/mismatched recovery attempt episode: {episode_id!r}")
        attempt_id = str(item.get("recovery_attempt_id", item.get("attempt_id", "")))
        if not attempt_id or attempt_id in attempt_ids:
            raise SchemaError(f"recovery attempt ID is empty/duplicate: {attempt_id!r}")
        attempt_ids.add(attempt_id)
        attempts_by_system[system].append(item)
    for item in incident_rows:
        system = str(item.get("system_id", ""))
        if system not in SYSTEM_IDS:
            raise SchemaError(f"failure incident has invalid system_id: {system!r}")
        episode_id = str(item.get("episode_id", ""))
        if episode_system.get(episode_id) != system:
            raise SchemaError(f"orphan/mismatched failure incident episode: {episode_id!r}")
        incident_id = str(item.get("failure_incident_id", item.get("incident_id", "")))
        identity = (episode_id, incident_id)
        if not incident_id or identity in incident_ids:
            raise SchemaError(f"failure incident ID is empty/duplicate: {identity!r}")
        incident_ids.add(identity)
        if "verified_agent_failure" not in item:
            raise SchemaError(f"failure incident lacks verified_agent_failure: {identity!r}")
        incidents_by_system[system].append(item)
    for item in attempt_rows:
        incident_id = str(
            item.get("failure_incident_id", item.get("incident_id", ""))
        )
        identity = (str(item.get("episode_id", "")), incident_id)
        if not incident_id or identity not in incident_ids:
            raise SchemaError(
                f"recovery attempt lacks a linked failure incident: {identity!r}"
            )
    for system_id in RECOVERY_DISABLED_SYSTEMS:
        if attempts_by_system[system_id]:
            raise SchemaError(f"{system_id} has attempts despite disabled recovery")

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recovery_k": recovery_k,
        "paired_block_count": len({row["block_id"] for row in rows}),
        "per_system_rate_inference": {
            "interval_method": "percentile_task_cluster_bootstrap",
            "inference_unit": "task_id",
            "estimand": "pooled_ratio_over_unique_registered_seed_repeat_cells",
            "within_task_aggregation": (
                "retain_all_unique_matched_seed_repeat_cells_and_linked_counts"
            ),
            "bootstrap_samples": bootstrap_samples,
            "confidence": confidence,
            "seed": bootstrap_seed,
        },
        "systems": {},
    }
    for system_id in SYSTEM_IDS:
        system_episodes = [row for row in rows if row["system_id"] == system_id]
        system_attempts = attempts_by_system[system_id]
        system_incidents = incidents_by_system[system_id]
        output["systems"][system_id] = _system_metrics(
            system_id,
            system_episodes,
            system_attempts,
            system_incidents,
            recovery_k,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
    output["environment_failure_rate"] = _environment_failure_metrics(
        list(schedule_attempts or [])
    )
    return output


def _system_metrics(
    system_id: str,
    episodes: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    recovery_k: int,
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    failure_episode_ids = {
        str(item["episode_id"])
        for item in incidents
        if _bool(item["verified_agent_failure"], "verified_agent_failure")
    }
    initiated = [row for row in attempts if _bool(row.get("initiated", True), "initiated")]
    verified_attempts = [
        row for row in initiated
        if _bool(row.get("verified_failure_present", False), "verified_failure_present")
    ]
    successful_attempts = [
        row for row in verified_attempts
        if _bool(row.get("successful", False), "successful")
    ]
    recovery_disabled = system_id in RECOVERY_DISABLED_SYSTEMS

    verified_incidents = [
        row for row in incidents
        if _bool(row["verified_agent_failure"], "verified_agent_failure")
    ]
    attempted_incident_keys = {
        (
            str(row.get("episode_id", "")),
            str(row.get("failure_incident_id", row.get("incident_id", ""))),
        )
        for row in initiated
    }
    recovery_at_incidents = [
        row
        for row in verified_incidents
        if (
            str(row.get("episode_id", "")),
            str(row.get("failure_incident_id", row.get("incident_id", ""))),
        )
        in attempted_incident_keys
    ]

    def resolved_within(limit: int, incident: Mapping[str, Any]) -> bool:
        resolved_index = incident.get("resolved_attempt_index")
        if resolved_index is not None and str(resolved_index) != "":
            return int(resolved_index) <= limit
        incident_id = str(incident.get("failure_incident_id", incident.get("incident_id", "")))
        episode_id = str(incident.get("episode_id", ""))
        candidates = [
            row for row in verified_attempts
            if str(row.get("failure_incident_id", "")) == incident_id
            and str(row.get("episode_id", "")) == episode_id
            and _bool(row.get("successful", False), "successful")
        ]
        return any(int(row.get("attempt_index", 10**9)) <= limit for row in candidates)

    initiated_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_attempts_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    successful_attempts_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_incidents_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    recovery_at_incidents_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in initiated:
        initiated_by_episode[str(attempt["episode_id"])].append(attempt)
    for attempt in verified_attempts:
        verified_attempts_by_episode[str(attempt["episode_id"])].append(attempt)
    for attempt in successful_attempts:
        successful_attempts_by_episode[str(attempt["episode_id"])].append(attempt)
    for incident in verified_incidents:
        verified_incidents_by_episode[str(incident["episode_id"])].append(incident)
    for incident in recovery_at_incidents:
        recovery_at_incidents_by_episode[str(incident["episode_id"])].append(incident)

    def clustered_rate(
        metric_name: str,
        counts: Callable[[Mapping[str, Any]], tuple[int, int]],
        *,
        force_na: bool = False,
    ) -> dict[str, Any]:
        contributions = []
        for episode in episodes:
            numerator, denominator = counts(episode)
            contributions.append(
                {
                    "task_id": episode["task_id"],
                    "cell": _episode_seed_repeat_cell(episode),
                    "numerator": _nonnegative_int(
                        numerator, f"{metric_name}.numerator"
                    ),
                    "denominator": _nonnegative_int(
                        denominator, f"{metric_name}.denominator"
                    ),
                }
            )
        return task_clustered_rate(
            contributions,
            metric_name=metric_name,
            system_id=system_id,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
            force_na=force_na,
        )

    def clustered_mean(
        metric_name: str,
        value: Callable[[Mapping[str, Any]], float | int | None],
        *,
        include: Callable[[Mapping[str, Any]], bool] | None = None,
        force_na: bool = False,
    ) -> dict[str, Any]:
        contributions: list[dict[str, Any]] = []
        for episode in episodes:
            if include is not None and not include(episode):
                continue
            measured = value(episode)
            contributions.append(
                {
                    "task_id": episode["task_id"],
                    "cell": _episode_seed_repeat_cell(episode),
                    "value": measured,
                }
            )
        return task_clustered_mean(
            contributions,
            metric_name=metric_name,
            system_id=system_id,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed,
            force_na=force_na,
        )

    return {
        "episode_count": len(episodes),
        "task_success_rate": clustered_rate(
            "task_success_rate",
            lambda episode: (int(episode["task_success"]), 1),
        ),
        "recovery_success_rate": clustered_rate(
            "recovery_success_rate",
            lambda episode: (
                len(successful_attempts_by_episode[episode["episode_id"]]),
                len(initiated_by_episode[episode["episode_id"]]),
            ),
            force_na=recovery_disabled,
        ),
        "verified_failure_only_recovery_success_rate": clustered_rate(
            "verified_failure_only_recovery_success_rate",
            lambda episode: (
                len(successful_attempts_by_episode[episode["episode_id"]]),
                len(verified_attempts_by_episode[episode["episode_id"]]),
            ),
            force_na=recovery_disabled,
        ),
        "false_trigger_rate": clustered_rate(
            "false_trigger_rate",
            lambda episode: (
                len(initiated_by_episode[episode["episode_id"]])
                - len(verified_attempts_by_episode[episode["episode_id"]]),
                len(initiated_by_episode[episode["episode_id"]]),
            ),
            force_na=recovery_disabled,
        ),
        "unnecessary_intervention_rate": clustered_rate(
            "unnecessary_intervention_rate",
            lambda episode: (
                len(initiated_by_episode[episode["episode_id"]])
                - len(verified_attempts_by_episode[episode["episode_id"]]),
                len(initiated_by_episode[episode["episode_id"]]),
            ),
            force_na=recovery_disabled,
        ),
        "recovery_at_1": clustered_rate(
            "recovery_at_1",
            lambda episode: (
                sum(
                    resolved_within(1, incident)
                    for incident in recovery_at_incidents_by_episode[
                        episode["episode_id"]
                    ]
                ),
                len(recovery_at_incidents_by_episode[episode["episode_id"]]),
            ),
            force_na=recovery_disabled,
        ),
        f"recovery_at_{recovery_k}": clustered_rate(
            f"recovery_at_{recovery_k}",
            lambda episode: (
                sum(
                    resolved_within(recovery_k, incident)
                    for incident in recovery_at_incidents_by_episode[
                        episode["episode_id"]
                    ]
                ),
                len(recovery_at_incidents_by_episode[episode["episode_id"]]),
            ),
            force_na=recovery_disabled,
        ),
        "success_after_initial_failure": clustered_rate(
            "success_after_initial_failure",
            lambda episode: (
                int(episode["task_success"])
                if episode["episode_id"] in failure_episode_ids
                else 0,
                int(episode["episode_id"] in failure_episode_ids),
            ),
        ),
        "verified_failure_incidence": clustered_rate(
            "verified_failure_incidence",
            lambda episode: (
                int(episode["episode_id"] in failure_episode_ids),
                1,
            ),
        ),
        "loop_episode_rate": clustered_rate(
            "loop_episode_rate",
            lambda episode: (int(episode["loop_detected"]), 1),
        ),
        "repeated_error_event_rate": clustered_rate(
            "repeated_error_event_rate",
            lambda episode: (
                _nonnegative_int(
                    episode.get("repeated_error_events"),
                    "repeated_error_events",
                ),
                _nonnegative_int(
                    episode.get("verified_failure_event_count"),
                    "verified_failure_event_count",
                ),
            ),
        ),
        "unrecovered_failure_rate": clustered_rate(
            "unrecovered_failure_rate",
            lambda episode: (
                sum(
                    not _bool(incident.get("resolved", False), "resolved")
                    for incident in verified_incidents_by_episode[
                        episode["episode_id"]
                    ]
                ),
                len(verified_incidents_by_episode[episode["episode_id"]]),
            ),
        ),
        "episodes_with_any_unresolved_failure": clustered_rate(
            "episodes_with_any_unresolved_failure",
            lambda episode: (
                int(
                    any(
                        not _bool(incident.get("resolved", False), "resolved")
                        for incident in verified_incidents_by_episode[
                            episode["episode_id"]
                        ]
                    )
                ),
                1,
            ),
        ),
        "steps": {
            "all": clustered_mean("browser_actions_all", lambda row: row["step_count"]),
            "successful_tasks": clustered_mean(
                "browser_actions_successful_tasks",
                lambda row: row["step_count"],
                include=lambda row: bool(row["task_success"]),
            ),
            "unsuccessful_tasks": clustered_mean(
                "browser_actions_unsuccessful_tasks",
                lambda row: row["step_count"],
                include=lambda row: not bool(row["task_success"]),
            ),
            "executed": clustered_mean(
                "executed_browser_actions",
                lambda row: row["executed_action_count"],
            ),
            "rejected": clustered_mean(
                "rejected_browser_actions",
                lambda row: row["rejected_action_count"],
            ),
            "recovery": clustered_mean(
                "recovery_browser_actions",
                lambda row: row["recovery_action_count"],
            ),
        },
        "recovery_attempts_per_episode": clustered_mean(
            "recovery_attempts_per_episode",
            lambda row: len(initiated_by_episode[row["episode_id"]]),
            force_na=recovery_disabled,
        ),
        "recovery_attempts_per_failure_episode": clustered_mean(
            "recovery_attempts_per_failure_episode",
            lambda row: len(initiated_by_episode[row["episode_id"]]),
            include=lambda row: row["episode_id"] in failure_episode_ids,
            force_na=recovery_disabled,
        ),
        "episode_recovery_success_rate": clustered_rate(
            "episode_recovery_success_rate",
            lambda episode: (
                int(
                    bool(successful_attempts_by_episode[episode["episode_id"]])
                )
                if episode["episode_id"] in failure_episode_ids
                else 0,
                int(episode["episode_id"] in failure_episode_ids),
            ),
            force_na=recovery_disabled,
        ),
        "efficiency": {
            "task_wall_clock_seconds": clustered_mean(
                "task_wall_clock_seconds",
                lambda row: row["task_wall_clock_seconds"],
            ),
            "model_call_count": clustered_mean(
                "model_call_count",
                lambda row: row["model_call_count"],
            ),
            "decision_latency_ms": clustered_mean(
                "decision_latency_ms",
                lambda row: row["decision_latency_ms"],
            ),
            "provider_latency_ms": clustered_mean(
                "provider_latency_ms",
                lambda row: row["provider_latency_ms"],
            ),
            "recovery_latency_ms": clustered_mean(
                "recovery_latency_ms",
                lambda row: row["recovery_latency_ms"],
            ),
            "retrieval_latency_ms": clustered_mean(
                "retrieval_latency_ms",
                lambda row: row["retrieval_latency_ms"],
            ),
            "input_token_count": clustered_mean(
                "input_token_count",
                lambda row: row["input_token_count"],
            ),
            "output_token_count": clustered_mean(
                "output_token_count",
                lambda row: row["output_token_count"],
            ),
            "model_parameter_count": clustered_mean(
                "model_parameter_count",
                lambda row: row["model_parameter_count"],
            ),
            "trainable_parameter_count": clustered_mean(
                "trainable_parameter_count",
                lambda row: row["trainable_parameter_count"],
            ),
            "peak_gpu_memory_mb": clustered_mean(
                "peak_gpu_memory_mb",
                lambda row: row["peak_gpu_memory_mb"],
            ),
            "peak_system_memory_mb": clustered_mean(
                "peak_system_memory_mb",
                lambda row: row["peak_system_memory_mb"],
            ),
            "memory_index_size": clustered_mean(
                "memory_index_size",
                lambda row: row["memory_index_size"],
            ),
            "training_gpu_hours": clustered_mean(
                "training_gpu_hours",
                lambda row: row["training_gpu_hours"],
            ),
        },
    }


def _normalize_episode(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "system_id",
        "block_id",
        "episode_id",
        "task_id",
        "task_success",
        "loop_detected",
        "step_count",
    }
    missing = sorted(required - set(record))
    if missing:
        raise SchemaError(f"episode is missing required fields: {missing}")
    system_id = str(record["system_id"])
    if system_id not in SYSTEM_IDS:
        raise SchemaError(f"invalid episode system_id: {system_id}")
    normalized = dict(record)
    normalized.update(
        {
            "system_id": system_id,
            "block_id": str(record["block_id"]),
            "episode_id": str(record["episode_id"]),
            "task_id": str(record["task_id"]),
            "task_success": _bool(record["task_success"], "task_success"),
            "loop_detected": _bool(record["loop_detected"], "loop_detected"),
            "step_count": _nonnegative_int(record["step_count"], "step_count"),
            "executed_action_count": _nonnegative_int(
                record.get("executed_action_count", record["step_count"]),
                "executed_action_count",
            ),
            "rejected_action_count": _nonnegative_int(
                record.get("rejected_action_count", 0), "rejected_action_count"
            ),
            "recovery_action_count": _nonnegative_int(
                record.get("recovery_action_count", 0), "recovery_action_count"
            ),
            "model_call_count": _optional_nonnegative_int(
                record.get("model_call_count"), "model_call_count"
            ),
            "task_wall_clock_seconds": _optional_nonnegative_float(
                record.get("task_wall_clock_seconds"), "task_wall_clock_seconds"
            ),
            "decision_latency_ms": _optional_nonnegative_float(
                record.get("decision_latency_ms"), "decision_latency_ms"
            ),
            "provider_latency_ms": _optional_nonnegative_float(
                record.get("provider_latency_ms"), "provider_latency_ms"
            ),
            "recovery_latency_ms": _optional_nonnegative_float(
                record.get("recovery_latency_ms"), "recovery_latency_ms"
            ),
            "retrieval_latency_ms": _optional_nonnegative_float(
                record.get("retrieval_latency_ms"), "retrieval_latency_ms"
            ),
            "input_token_count": _optional_nonnegative_int(
                record.get("input_token_count"), "input_token_count"
            ),
            "output_token_count": _optional_nonnegative_int(
                record.get("output_token_count"), "output_token_count"
            ),
            "model_parameter_count": _optional_nonnegative_int(
                record.get("model_parameter_count"), "model_parameter_count"
            ),
            "trainable_parameter_count": _optional_nonnegative_int(
                record.get("trainable_parameter_count"), "trainable_parameter_count"
            ),
            "peak_gpu_memory_mb": _optional_nonnegative_float(
                record.get("peak_gpu_memory_mb"), "peak_gpu_memory_mb"
            ),
            "peak_system_memory_mb": _optional_nonnegative_float(
                record.get("peak_system_memory_mb"), "peak_system_memory_mb"
            ),
            "memory_index_size": _optional_nonnegative_int(
                record.get("memory_index_size"), "memory_index_size"
            ),
            "training_gpu_hours": _optional_nonnegative_float(
                record.get("training_gpu_hours"), "training_gpu_hours"
            ),
        }
    )
    if normalized["step_count"] != (
        normalized["executed_action_count"] + normalized["rejected_action_count"]
    ):
        raise SchemaError(
            "step_count must equal executed plus rejected budget-consuming executor requests"
        )
    if normalized["recovery_action_count"] > normalized["step_count"]:
        raise SchemaError("recovery_action_count is a subset of steps, not an extra total")
    return normalized


def _assert_complete_paired_blocks(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SchemaError("no selected paired episodes were supplied")
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episode_ids: set[str] = set()
    for row in rows:
        if row["episode_id"] in episode_ids:
            raise SchemaError(f"duplicate episode ID: {row['episode_id']}")
        episode_ids.add(row["episode_id"])
        by_block[row["block_id"]].append(row)
    for block_id, block_rows in by_block.items():
        systems = [row["system_id"] for row in block_rows]
        if len(systems) != 4 or set(systems) != set(SYSTEM_IDS):
            raise SchemaError(
                f"primary block {block_id} is not a complete one-E0-through-E3 block: {systems}"
            )
        if any(_bool(row.get("infrastructure_invalid", False), "infrastructure_invalid") for row in block_rows):
            raise SchemaError(f"infrastructure-invalid block {block_id} entered primary metrics")


def _environment_failure_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    launched = [row for row in rows if _bool(row.get("launched", True), "launched")]
    invalid = [
        row for row in launched
        if _bool(row.get("infrastructure_invalid", False), "infrastructure_invalid")
    ]
    originals = [row for row in launched if int(row.get("attempt_id", 0)) == 0]
    reruns = [row for row in launched if int(row.get("attempt_id", 0)) > 0]
    result = {
        "all_launches": rate(len(invalid), len(launched)),
        "original_launches": rate(
            sum(_bool(row.get("infrastructure_invalid", False), "infrastructure_invalid") for row in originals),
            len(originals),
        ),
        "rerun_launches": rate(
            sum(_bool(row.get("infrastructure_invalid", False), "infrastructure_invalid") for row in reruns),
            len(reruns),
        ),
    }
    result["by_system"] = {
        system_id: rate(
            sum(
                _bool(row.get("infrastructure_invalid", False), "infrastructure_invalid")
                for row in launched
                if str(row.get("system_id")) == system_id
            ),
            sum(str(row.get("system_id")) == system_id for row in launched),
        )
        for system_id in SYSTEM_IDS
    }
    return result


def _episode_seed_repeat_cell(episode: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the registered repeated-measure cell for one system episode."""

    fields = ("matched_model_seed", "repeat_id")
    present = [field in episode for field in fields]
    if not any(present):
        # Small deterministic fixtures predate explicit seed/repeat fields.
        # Their unique block identity is a safe one-cell fallback.
        return ("block_id", str(episode["block_id"]))
    if not all(present):
        raise SchemaError(
            "per-system rate inference requires both matched_model_seed and "
            "repeat_id when either is present"
        )
    values = []
    for field in fields:
        value = episode[field]
        if type(value) is not int or value < 0:
            raise SchemaError(
                f"per-system rate inference requires nonnegative integer {field}"
            )
        values.append(value)
    return ("matched_seed_repeat", *values)


def _embedded(rows: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for episode in rows:
        values = episode.get(key, [])
        if values is None:
            continue
        if not isinstance(values, list) or not all(isinstance(item, Mapping) for item in values):
            raise SchemaError(f"episode {episode.get('episode_id')} has invalid {key}")
        for value in values:
            output.append(
                {
                    "system_id": episode["system_id"],
                    "episode_id": episode["episode_id"],
                    **dict(value),
                }
            )
    return output


def _nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise SchemaError(f"{context} must be an exact integer")
    result = value
    if result < 0:
        raise SchemaError(f"{context} must be nonnegative")
    return result


def _nonnegative_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"{context} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{context} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise SchemaError(f"{context} must be finite and nonnegative")
    return result


def _optional_nonnegative_int(value: Any, context: str) -> int | None:
    if value is None or value == "":
        return None
    return _nonnegative_int(value, context)


def _optional_nonnegative_float(value: Any, context: str) -> float | None:
    if value is None or value == "":
        return None
    return _nonnegative_float(value, context)


def _bool(value: Any, context: str) -> bool:
    return strict_bool(value, context=context)


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
