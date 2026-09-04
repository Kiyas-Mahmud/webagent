"""Deterministic task-cluster bootstrap and registered paired contrasts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import math
import random
from typing import Any

from .common import SCHEMA_VERSION, SchemaError, stable_int_seed, strict_bool
from .metrics import rate
from .schedule import SYSTEM_IDS


REGISTERED_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("E0", "E1"),
    ("E1", "E2"),
    ("E2", "E3"),
)
DESCRIPTIVE_TOTAL_SYSTEM_CONTRASTS: tuple[tuple[str, str], ...] = (("E0", "E3"),)
COMPUTED_CONTRASTS = REGISTERED_CONTRASTS + DESCRIPTIVE_TOTAL_SYSTEM_CONTRASTS


def cluster_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    value: Callable[[Mapping[str, Any]], float],
    cluster_key: str = "task_id",
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20250831,
) -> dict[str, Any]:
    """Bootstrap whole task clusters, preserving within-task dependence."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if cluster_key not in row:
            raise SchemaError(f"bootstrap row lacks cluster key {cluster_key!r}")
        grouped[str(row[cluster_key])].append(dict(row))
    clusters = sorted(grouped)
    if not clusters:
        return {
            "n_clusters": 0,
            "n_rows": 0,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "samples": samples,
            "seed": seed,
        }
    base_values = [float(value(row)) for cluster in clusters for row in grouped[cluster]]
    point = sum(base_values) / len(base_values)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [float(value(row)) for cluster in chosen for row in grouped[cluster]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_clusters": len(clusters),
        "n_rows": len(base_values),
        "estimate": point,
        "ci_low": _quantile(estimates, alpha),
        "ci_high": _quantile(estimates, 1.0 - alpha),
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
    }


def compute_paired_contrasts(
    episodes: Iterable[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = ("task_success", "step_count", "loop_detected"),
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20250831,
) -> dict[str, Any]:
    rows = [dict(row) for row in episodes]
    paired = _pair_rows(rows)
    paired_rows = [
        {
            "block_id": block_id,
            "task_id": str(systems["E0"]["task_id"]),
            "systems": systems,
        }
        for block_id, systems in paired.items()
    ]
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_unit": "task_id",
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "base_seed": seed,
        "contrasts": {},
    }
    for left, right in COMPUTED_CONTRASTS:
        contrast_name = f"{right}_minus_{left}"
        pair_rows = [
            {
                "block_id": pair["block_id"],
                "task_id": pair["task_id"],
                "left": pair["systems"][left],
                "right": pair["systems"][right],
            }
            for pair in paired_rows
        ]
        metric_results: dict[str, Any] = {}
        for metric_key in metric_keys:
            for pair in pair_rows:
                if metric_key not in pair["left"] or metric_key not in pair["right"]:
                    raise SchemaError(f"paired contrast lacks metric {metric_key!r}")
            derived_seed = stable_int_seed(seed, contrast_name, metric_key)
            result = cluster_bootstrap(
                pair_rows,
                value=lambda pair, key=metric_key: _number(pair["right"][key])
                - _number(pair["left"][key]),
                cluster_key="task_id",
                samples=bootstrap_samples,
                confidence=confidence,
                seed=derived_seed,
            )
            result["direction"] = f"{right} minus {left}"
            left_values = [_number(pair["left"][metric_key]) for pair in pair_rows]
            right_values = [_number(pair["right"][metric_key]) for pair in pair_rows]
            left_mean = sum(left_values) / len(left_values)
            right_mean = sum(right_values) / len(right_values)
            result["left_mean"] = left_mean
            result["right_mean"] = right_mean
            result["absolute_change"] = result["estimate"]
            result["relative_change"] = (
                None if left_mean == 0.0 else (right_mean - left_mean) / abs(left_mean)
            )
            metric_results[metric_key] = result
        if "task_success" in metric_keys:
            left_only = sum(
                strict_bool(pair["left"]["task_success"], context="left.task_success")
                and not strict_bool(pair["right"]["task_success"], context="right.task_success")
                for pair in pair_rows
            )
            right_only = sum(
                not strict_bool(pair["left"]["task_success"], context="left.task_success")
                and strict_bool(pair["right"]["task_success"], context="right.task_success")
                for pair in pair_rows
            )
            metric_results["task_success"]["discordant"] = {
                f"{left}_only": left_only,
                f"{right}_only": right_only,
                "unit": "paired_block",
                "inference_role": "descriptive_only",
            }
            significance = _task_clustered_success_significance(
                pair_rows,
                left=left,
                right=right,
            )
            metric_results["task_success"]["task_clustered_significance"] = significance
            if (left, right) in REGISTERED_CONTRASTS:
                significance["inference_role"] = "registered_holm_family_member"
                metric_results["task_success"]["analysis_role"] = (
                    "registered_adjacent_inferential_contrast"
                )
            else:
                significance["inference_role"] = "descriptive_unadjusted_only"
                metric_results["task_success"]["analysis_role"] = (
                    "descriptive_total_system_contrast"
                )
        output["contrasts"][contrast_name] = metric_results

    task_success_p_values = {
        name: metrics["task_success"]["task_clustered_significance"][
            "exact_sign_two_sided_p"
        ]
        for name, metrics in output["contrasts"].items()
        if name
        in {f"{right}_minus_{left}" for left, right in REGISTERED_CONTRASTS}
        if "task_success" in metrics
        and metrics["task_success"]["task_clustered_significance"][
            "exact_sign_two_sided_p"
        ]
        is not None
    }
    holm = _holm_adjust(task_success_p_values)
    for name, adjusted in holm.items():
        output["contrasts"][name]["task_success"]["task_clustered_significance"][
            "holm_adjusted_p"
        ] = adjusted
    output["multiple_testing"] = {
        "family": "registered_task_success_contrasts",
        "method": "Holm",
        "hypothesis_count": len(task_success_p_values),
        "contrast_names": [
            f"{right}_minus_{left}" for left, right in REGISTERED_CONTRASTS
        ],
        "unadjusted_test": "exact_two_sided_sign_test",
        "inference_unit": "task_id",
        "within_task_aggregation": (
            "mean_paired_difference_over_unique_matched_seed_repeat_cells"
        ),
    }

    output["paired_memory_regression_rate"] = _task_level_memory_regression_rate(
        paired_rows
    )
    return output


def _task_level_memory_regression_rate(
    paired_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count memory regressions once per task across registered seed/repeat cells.

    A task enters the denominator when E2 succeeds in at least one unique
    matched-seed/repeat cell. It enters the numerator when any such E2 success
    becomes an E3 failure. This conservative task-level harm diagnostic cannot
    be inflated by adding repeated seeds, repeats, or duplicate block rows.
    """

    tasks: dict[str, dict[tuple[Any, ...], tuple[bool, bool]]] = defaultdict(dict)
    duplicate_cells = 0
    for pair in paired_rows:
        task_id = str(pair["task_id"])
        systems = pair["systems"]
        e2 = systems["E2"]
        e3 = systems["E3"]
        outcome = (
            strict_bool(e2["task_success"], context=f"{task_id}.E2.task_success"),
            strict_bool(e3["task_success"], context=f"{task_id}.E3.task_success"),
        )
        cell_pair = {
            "block_id": pair["block_id"],
            "left": e2,
            "right": e3,
        }
        cell = _paired_seed_repeat_cell(cell_pair)
        previous = tasks[task_id].get(cell)
        if previous is None:
            tasks[task_id][cell] = outcome
        elif previous != outcome:
            raise SchemaError(
                "conflicting E2/E3 outcomes for duplicate paired memory "
                f"seed/repeat cell {task_id}.{cell}"
            )
        else:
            duplicate_cells += 1

    denominator = 0
    numerator = 0
    unique_cells = 0
    for cells in tasks.values():
        unique_cells += len(cells)
        e2_success = any(e2_success for e2_success, _ in cells.values())
        regression = any(
            e2_success and not e3_success
            for e2_success, e3_success in cells.values()
        )
        denominator += int(e2_success)
        numerator += int(regression)
    result = rate(numerator, denominator)
    result.update(
        {
            "unit": "task_id",
            "aggregation": (
                "any_E2_success_denominator_and_any_E2_success_to_E3_failure_"
                "over_unique_matched_seed_repeat_cells"
            ),
            "n_task_clusters": len(tasks),
            "unique_seed_repeat_cells": unique_cells,
            "duplicate_seed_repeat_cells_collapsed": duplicate_cells,
        }
    )
    return result


def compute_clustered_ratio_contrasts(
    episodes: Iterable[Mapping[str, Any]],
    *,
    ratio_fields: Mapping[str, tuple[str, str]],
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20250831,
) -> dict[str, Any]:
    """Bootstrap paired differences of registered numerator/denominator ratios."""

    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid ratio-bootstrap configuration")
    paired = _pair_rows([dict(row) for row in episodes])
    tasks: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for systems in paired.values():
        task_id = str(next(iter(systems.values()))["task_id"])
        tasks[task_id].append(systems)
    task_ids = sorted(tasks)
    output: dict[str, Any] = {}
    for left, right in REGISTERED_CONTRASTS:
        contrast_name = f"{right}_minus_{left}"
        output[contrast_name] = {}
        for metric_name, (numerator_key, denominator_key) in ratio_fields.items():
            def system_ratio(system_id: str, chosen: Sequence[str]) -> float | None:
                numerator = 0
                denominator = 0
                for task_id in chosen:
                    for systems in tasks[task_id]:
                        numerator += _nonnegative_count(
                            systems[system_id].get(numerator_key), numerator_key
                        )
                        denominator += _nonnegative_count(
                            systems[system_id].get(denominator_key), denominator_key
                        )
                if numerator > denominator:
                    raise SchemaError(f"{metric_name} numerator exceeds denominator")
                return None if denominator == 0 else numerator / denominator

            left_point = system_ratio(left, task_ids)
            right_point = system_ratio(right, task_ids)
            if left_point is None or right_point is None:
                output[contrast_name][metric_name] = {
                    "n_clusters": len(task_ids),
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "left_ratio": left_point,
                    "right_ratio": right_point,
                    "absolute_change": None,
                    "relative_change": None,
                    "display": "N/A",
                    "reason": "one or both registered denominators are zero",
                    "samples": bootstrap_samples,
                    "seed": stable_int_seed(seed, contrast_name, metric_name),
                }
                continue
            derived_seed = stable_int_seed(seed, contrast_name, metric_name)
            rng = random.Random(derived_seed)
            estimates: list[float] = []
            for _ in range(bootstrap_samples):
                chosen = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
                left_value = system_ratio(left, chosen)
                right_value = system_ratio(right, chosen)
                if left_value is not None and right_value is not None:
                    estimates.append(right_value - left_value)
            estimates.sort()
            alpha = (1.0 - confidence) / 2.0
            estimate = right_point - left_point
            output[contrast_name][metric_name] = {
                "n_clusters": len(task_ids),
                "estimate": estimate,
                "ci_low": _quantile(estimates, alpha) if estimates else None,
                "ci_high": _quantile(estimates, 1.0 - alpha) if estimates else None,
                "left_ratio": left_point,
                "right_ratio": right_point,
                "absolute_change": estimate,
                "relative_change": (
                    None if left_point == 0.0 else estimate / abs(left_point)
                ),
                "valid_bootstrap_samples": len(estimates),
                "samples": bootstrap_samples,
                "seed": derived_seed,
            }
    return output


def _pair_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        for key in ("block_id", "system_id", "task_id"):
            if key not in row:
                raise SchemaError(f"paired row is missing {key!r}")
        system = str(row["system_id"])
        if system not in SYSTEM_IDS:
            raise SchemaError(f"invalid system in paired rows: {system}")
        block = str(row["block_id"])
        if system in paired[block]:
            raise SchemaError(f"duplicate {system} episode for block {block}")
        if "task_success" in row:
            row["task_success"] = strict_bool(
                row["task_success"], context=f"{block}.{system}.task_success"
            )
        if "loop_detected" in row:
            row["loop_detected"] = strict_bool(
                row["loop_detected"], context=f"{block}.{system}.loop_detected"
            )
        paired[block][system] = row
    for block, systems in paired.items():
        if set(systems) != set(SYSTEM_IDS):
            raise SchemaError(f"incomplete paired block {block}: {sorted(systems)}")
        task_ids = {str(row["task_id"]) for row in systems.values()}
        if len(task_ids) != 1:
            raise SchemaError(f"paired block {block} mixes tasks: {task_ids}")
    if not paired:
        raise SchemaError("no paired rows available for statistical contrasts")
    return dict(paired)


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"contrast value is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise SchemaError("contrast values must be finite")
    return number


def _nonnegative_count(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise SchemaError(f"{context} must be a nonnegative integer")
    return value


def _task_clustered_success_significance(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    """Run an exact paired sign test with task, not episode, as the unit.

    Seed/repeat rows from the same task are correlated repeated measurements.
    They are first reduced to one mean paired effect per task. Exact duplicate
    seed/repeat cells are collapsed so accidental row multiplication cannot
    create significance; a conflicting duplicate fails closed.
    """

    tasks: dict[str, dict[tuple[Any, ...], tuple[bool, bool]]] = defaultdict(dict)
    duplicate_cells = 0
    for pair in pair_rows:
        task_id = str(pair["task_id"])
        left_row = pair["left"]
        right_row = pair["right"]
        left_success = strict_bool(
            left_row["task_success"], context=f"{task_id}.{left}.task_success"
        )
        right_success = strict_bool(
            right_row["task_success"], context=f"{task_id}.{right}.task_success"
        )
        cell = _paired_seed_repeat_cell(pair)
        outcome = (left_success, right_success)
        previous = tasks[task_id].get(cell)
        if previous is None:
            tasks[task_id][cell] = outcome
        elif previous != outcome:
            raise SchemaError(
                "conflicting task-success outcomes for duplicate paired "
                f"seed/repeat cell {task_id}.{cell}"
            )
        else:
            duplicate_cells += 1

    improved = 0
    worsened = 0
    tied = 0
    task_effects: list[dict[str, Any]] = []
    unique_cells = 0
    for task_id in sorted(tasks):
        cells = tasks[task_id]
        unique_cells += len(cells)
        effect_sum = sum(
            int(right_value) - int(left_value)
            for left_value, right_value in cells.values()
        )
        if effect_sum > 0:
            direction = "right_better"
            improved += 1
        elif effect_sum < 0:
            direction = "left_better"
            worsened += 1
        else:
            direction = "tie"
            tied += 1
        task_effects.append(
            {
                "task_id": task_id,
                "unique_seed_repeat_cells": len(cells),
                "paired_difference_mean": effect_sum / len(cells),
                "direction": direction,
            }
        )

    return {
        "method": "exact_two_sided_sign_test",
        "inference_unit": "task_id",
        "within_task_aggregation": (
            "mean_paired_difference_over_unique_matched_seed_repeat_cells"
        ),
        "left_system": left,
        "right_system": right,
        "n_task_clusters": len(tasks),
        "right_better_task_clusters": improved,
        "left_better_task_clusters": worsened,
        "tied_task_clusters": tied,
        "non_tied_task_clusters": improved + worsened,
        "unique_seed_repeat_cells": unique_cells,
        "duplicate_seed_repeat_cells_collapsed": duplicate_cells,
        "exact_sign_two_sided_p": _exact_sign_two_sided(improved, worsened),
        "task_effects": task_effects,
    }


def _paired_seed_repeat_cell(pair: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the repeated-measure cell, falling back for legacy fixtures."""

    left_row = pair["left"]
    right_row = pair["right"]
    fields = ("matched_model_seed", "repeat_id")
    present = [field in left_row or field in right_row for field in fields]
    if not any(present):
        return ("block_id", str(pair["block_id"]))
    if not all(present):
        raise SchemaError(
            "paired success significance requires both matched_model_seed and "
            "repeat_id when either is present"
        )
    values: list[int] = []
    for field in fields:
        if field not in left_row or field not in right_row:
            raise SchemaError(f"paired rows disagree on presence of {field}")
        left_value = left_row[field]
        right_value = right_row[field]
        if (
            type(left_value) is not int
            or type(right_value) is not int
            or left_value < 0
            or right_value < 0
        ):
            raise SchemaError(f"paired {field} must be a nonnegative integer")
        if left_value != right_value:
            raise SchemaError(f"paired rows disagree on {field}")
        values.append(left_value)
    return ("matched_seed_repeat", *values)


def _exact_sign_two_sided(positive: int, negative: int) -> float:
    discordant = positive + negative
    if discordant == 0:
        # With no non-zero task effects the observed statistic is exactly the
        # null value. The planned comparison remains in the Holm family with a
        # valid, maximally non-significant p-value.
        return 1.0
    smaller = min(positive, negative)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def _holm_adjust(values: Mapping[str, float]) -> dict[str, float]:
    """Deterministic Holm family-wise-error adjustment."""

    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of no bootstrap values")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)
