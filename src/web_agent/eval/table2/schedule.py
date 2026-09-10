"""Deterministic paired E0--E3 schedules and whole-block resolution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import itertools
from pathlib import Path
from typing import Any

from .common import (
    SCHEMA_VERSION,
    SchemaError,
    require_keys,
    stable_int_seed,
    strict_bool,
)


SYSTEM_IDS: tuple[str, ...] = ("E0", "E1", "E2", "E3")
REGISTERED_MATCHED_MODEL_SEEDS: tuple[int, ...] = (42,)
EXECUTION_ORDER_ALGORITHM = "sha256_offset_permutation_cycle_v1"
DEFAULT_STAGE_KEYS: tuple[str, ...] = (
    "reset",
    "pre_action",
    "action_parameters",
    "post_action_assessment",
    "recovery_shadow",
    "memory_query",
    "memory_intervention",
    "recovery_resolution",
    "recovery_assessment",
)


def _system_order(block_number: int, campaign_seed: int) -> list[str]:
    """Counterbalance orders without allowing execution order to change RNG."""

    orders = tuple(itertools.permutations(SYSTEM_IDS))
    offset = stable_int_seed("table2-order", campaign_seed) % len(orders)
    return list(orders[(offset + block_number) % len(orders)])


def build_paired_schedule(
    tasks: Iterable[Mapping[str, Any]],
    *,
    model_seeds: Sequence[int] = REGISTERED_MATCHED_MODEL_SEEDS,
    repetitions: int = 1,
    campaign_seed: int = 42,
    protocol_id: str = "table2-protocol",
    campaign_id: str = "table2-campaign",
    max_block_attempts: int = 2,
    stage_keys: Sequence[str] = DEFAULT_STAGE_KEYS,
) -> list[dict[str, Any]]:
    """Create one immutable schedule row per task x seed x repetition.

    Stage seeds intentionally omit ``system_id``: all four systems receive the
    same seed for the same stochastic stage.  A rerun also receives the same
    registered stage seeds; it is a replacement of the complete block, not a
    new opportunity to choose a favourable random realization.
    """

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if max_block_attempts <= 0:
        raise ValueError("max_block_attempts must be positive")
    if (
        any(type(seed) is not int for seed in model_seeds)
        or tuple(model_seeds) != REGISTERED_MATCHED_MODEL_SEEDS
    ):
        raise ValueError(
            "Table 2 registers matched model seed 42 only; seeds 43 and 44 "
            "must not be scheduled"
        )
    if len(set(stage_keys)) != len(stage_keys) or not stage_keys:
        raise ValueError("stage_keys must be non-empty and unique")

    task_rows = [dict(task) for task in tasks]
    for index, task in enumerate(task_rows):
        require_keys(task, ("task_id",), context=f"task[{index}]")
    if len({str(task["task_id"]) for task in task_rows}) != len(task_rows):
        raise SchemaError("task IDs must be unique in the frozen task manifest")

    schedule: list[dict[str, Any]] = []
    block_number = 0
    # Preserve the already-frozen manifest order (the pilot manifest registers
    # upstream indices 0..49 in ascending order). Re-sorting string IDs would
    # incorrectly place task 10 before task 2.
    for task in task_rows:
        task_id = str(task["task_id"])
        for matched_seed in model_seeds:
            for repeat_index in range(repetitions):
                block_id = (
                    f"task_{_safe_id(task_id)}__seed_{int(matched_seed)}"
                    f"__repeat_{repeat_index:03d}"
                )
                stage_seeds = {
                    stage: _runtime_stage_seed(
                        protocol_id=protocol_id,
                        campaign_id=campaign_id,
                        campaign_seed=campaign_seed,
                        task_id=task_id,
                        repeat_id=repeat_index,
                        matched_seed=int(matched_seed),
                        stage=stage,
                    )
                    for stage in stage_keys
                }
                schedule.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "block_id": block_id,
                        "block_number": block_number,
                        "task_id": task_id,
                        "task_partition": task.get(
                            "task_partition", task.get("partition", "unspecified")
                        ),
                        "matched_model_seed": int(matched_seed),
                        "repeat_id": repeat_index,
                        "stage_seed_decision_index": 0,
                        "systems": list(SYSTEM_IDS),
                        "execution_order_algorithm": EXECUTION_ORDER_ALGORITHM,
                        "execution_order": _system_order(block_number, campaign_seed),
                        "stage_seeds": stage_seeds,
                        "max_block_attempts": int(max_block_attempts),
                    }
                )
                block_number += 1
    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(schedule):
        context = f"schedule[{index}]"
        require_keys(
            row,
            (
                "schema_version",
                "block_id",
                "task_id",
                "matched_model_seed",
                "repeat_id",
                "systems",
                "execution_order_algorithm",
                "execution_order",
                "stage_seeds",
                "max_block_attempts",
            ),
            context=context,
        )
        if row["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"{context} uses unsupported schema version")
        block_id = str(row["block_id"])
        if block_id in seen:
            raise SchemaError(f"duplicate schedule block: {block_id}")
        seen.add(block_id)
        if tuple(row["systems"]) != SYSTEM_IDS:
            raise SchemaError(f"{context} must register exactly E0, E1, E2, E3")
        if row["execution_order_algorithm"] != EXECUTION_ORDER_ALGORITHM:
            raise SchemaError(f"{context} uses an unregistered execution-order algorithm")
        if set(row["execution_order"]) != set(SYSTEM_IDS) or len(row["execution_order"]) != 4:
            raise SchemaError(f"{context} execution_order is not a permutation of E0--E3")
        seeds = row["stage_seeds"]
        if not isinstance(seeds, Mapping) or not seeds:
            raise SchemaError(f"{context} has no stage-keyed seeds")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in seeds.values()):
            raise SchemaError(f"{context} stage seeds must be integers")
        if (
            type(row["matched_model_seed"]) is not int
            or int(row["matched_model_seed"]) not in REGISTERED_MATCHED_MODEL_SEEDS
        ):
            raise SchemaError(
                f"{context} must use the registered matched model seed 42 only"
            )
        if int(row["repeat_id"]) < 0:
            raise SchemaError(f"{context} repeat_id cannot be negative")
        if int(row["max_block_attempts"]) <= 0:
            raise SchemaError(f"{context} max_block_attempts must be positive")


def discover_block_attempt_directories(
    base: str | Path,
    *,
    maximum: int,
) -> tuple[tuple[int, Path], ...]:
    """Return the exact consecutive physical attempts permitted by a schedule.

    Reading only ``range(maximum)`` would silently ignore an out-of-range
    ``rerun_*`` directory.  Both execution and package validation use this
    filesystem-closure check so an extra attempt can never escape the frozen
    whole-block rerun limit.
    """

    root = Path(base)
    if type(maximum) is not int or maximum <= 0:
        raise SchemaError("maximum physical block attempts must be positive")
    if not root.exists():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise SchemaError(f"paired-block path is not a real directory: {root}")

    discovered: dict[int, Path] = {}
    for child in root.iterdir():
        if not child.name.startswith("rerun_"):
            continue
        suffix = child.name.removeprefix("rerun_")
        if (
            not suffix.isdigit()
            or str(int(suffix)) != suffix
            or child.is_symlink()
            or not child.is_dir()
        ):
            raise SchemaError(f"invalid physical attempt path: {child}")
        attempt_id = int(suffix)
        if attempt_id >= maximum:
            raise SchemaError(
                f"out-of-range physical attempt {child.name}; maximum is {maximum}"
            )
        manifest = child / "block_manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise SchemaError(f"physical attempt lacks block_manifest.json: {child}")
        if attempt_id in discovered:
            raise SchemaError(f"duplicate physical attempt ordinal: {attempt_id}")
        discovered[attempt_id] = child

    ordinals = sorted(discovered)
    if ordinals != list(range(len(ordinals))):
        raise SchemaError(
            f"physical attempt ordinals must be consecutive from zero: {ordinals}"
        )
    return tuple((attempt_id, discovered[attempt_id]) for attempt_id in ordinals)


def resolve_block_attempts(
    schedule_row: Mapping[str, Any],
    attempts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a complete valid attempt, or require/explain whole-block rerun.

    Agent-caused failure, timeout, ABORT, and step-limit are completed outcomes,
    not infrastructure exclusions.  Only a system result explicitly marked
    ``infrastructure_invalid`` invalidates the complete physical attempt.
    """

    block_id = str(schedule_row["block_id"])
    maximum = int(schedule_row["max_block_attempts"])
    ordered = sorted((dict(item) for item in attempts), key=lambda item: int(item["attempt_id"]))
    if len(ordered) > maximum:
        raise SchemaError(f"{block_id}: more physical attempts exist than preregistered")
    normalized: list[dict[str, Any]] = []
    selected: int | None = None
    for expected_attempt_id, attempt in enumerate(ordered):
        attempt_id = int(attempt.get("attempt_id", -1))
        if attempt_id != expected_attempt_id:
            raise SchemaError(
                f"{block_id}: attempt IDs must be consecutive from zero; got {attempt_id}"
            )
        systems = attempt.get("systems")
        if not isinstance(systems, Mapping) or set(systems) != set(SYSTEM_IDS):
            raise SchemaError(f"{block_id}/attempt_{attempt_id}: missing E0--E3 status")
        system_rows: dict[str, dict[str, Any]] = {}
        for name in SYSTEM_IDS:
            raw = systems[name]
            if not isinstance(raw, Mapping):
                raise SchemaError(
                    f"{block_id}/attempt_{attempt_id}/{name}: status must be an object"
                )
            row = dict(raw)
            launched = strict_bool(row.get("launched", False), context="launched")
            completed = strict_bool(row.get("completed", False), context="completed")
            infrastructure_invalid = strict_bool(
                row.get("infrastructure_invalid", False),
                context="infrastructure_invalid",
            )
            fatal_noninfrastructure = strict_bool(
                row.get("fatal_noninfrastructure_error", False),
                context="fatal_noninfrastructure_error",
            )
            if fatal_noninfrastructure:
                raise SchemaError(
                    f"{block_id}/attempt_{attempt_id}/{name}: "
                    "fatal non-infrastructure error cannot be rerun"
                )
            if completed and not launched:
                raise SchemaError(
                    f"{block_id}/attempt_{attempt_id}/{name}: completed but not launched"
                )
            if infrastructure_invalid:
                if not launched or completed:
                    raise SchemaError(
                        f"{block_id}/attempt_{attempt_id}/{name}: infrastructure "
                        "invalidation requires a launched, incomplete system"
                    )
                reason = str(row.get("infrastructure_reason") or "").strip()
                evidence_hash = str(
                    row.get("infrastructure_evidence_sha256") or ""
                )
                try:
                    valid_evidence_hash = (
                        len(evidence_hash) == 64
                        and int(evidence_hash, 16) >= 0
                    )
                except ValueError:
                    valid_evidence_hash = False
                if not reason or not valid_evidence_hash:
                    raise SchemaError(
                        f"{block_id}/attempt_{attempt_id}/{name}: infrastructure "
                        "invalidation lacks a typed reason/evidence hash"
                    )
            elif row.get("infrastructure_reason") is not None or row.get(
                "infrastructure_evidence_sha256"
            ) is not None:
                raise SchemaError(
                    f"{block_id}/attempt_{attempt_id}/{name}: non-infrastructure "
                    "status carries infrastructure authorization fields"
                )
            row.update(
                {
                    "launched": launched,
                    "completed": completed,
                    "infrastructure_invalid": infrastructure_invalid,
                }
            )
            system_rows[name] = row
        infrastructure_invalid = any(
            row["infrastructure_invalid"] for row in system_rows.values()
        )
        all_completed = all(row["completed"] for row in system_rows.values())
        valid = all_completed and not infrastructure_invalid
        normalized.append(
            {
                "attempt_id": attempt_id,
                "valid": valid,
                "infrastructure_invalid": infrastructure_invalid,
                "all_systems_completed": all_completed,
                "systems": system_rows,
            }
        )
        if valid:
            selected = attempt_id
            break

    if selected is not None and len(ordered) > selected + 1:
        raise SchemaError(f"{block_id}: physical attempts exist after a valid block")

    unauthorized_attempt = any(
        not attempt["valid"] and not attempt["infrastructure_invalid"]
        for attempt in normalized
    )
    if unauthorized_attempt:
        # A physical rerun is authorized only by typed infrastructure
        # invalidation of the preceding whole-block attempt.  In particular,
        # an all-unlaunched or partially completed attempt cannot become an
        # infrastructure exclusion merely because the attempt limit is spent.
        status = "INTERRUPTED_UNAUTHORIZED"
        selected = None
    elif selected is not None:
        status = "INCLUDED"
    elif len(ordered) < maximum:
        status = "RERUN_REQUIRED"
    else:
        status = "EXCLUDED_INFRASTRUCTURE"
    return {
        "schema_version": SCHEMA_VERSION,
        "block_id": block_id,
        "status": status,
        "selected_attempt_id": selected,
        "maximum_attempts": maximum,
        "attempts": normalized,
    }


def _safe_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise SchemaError(f"task ID cannot form a safe artifact path: {value!r}")
    return cleaned[:120]


def _runtime_stage_seed(
    *,
    protocol_id: str,
    campaign_id: str,
    campaign_seed: int,
    task_id: str,
    repeat_id: int,
    matched_seed: int,
    stage: str,
) -> int:
    """Use the runtime's exact registered RNG-key implementation.

    The schedule stores the decision-index-zero anchor for every runtime stage.
    Dynamic decision/incident/attempt streams are derived from the same frozen
    namespace by :class:`StageRNGFactory`; system ID and rerun ordinal never
    enter that namespace.
    """

    try:
        from web_agent.runtime.protocol import StageRNGFactory

        return StageRNGFactory(
            protocol_id=protocol_id,
            campaign_id=campaign_id,
            campaign_seed=int(campaign_seed),
        ).seed_for(
            task_id=task_id,
            repeat_id=int(repeat_id),
            matched_seed=int(matched_seed),
            stage=stage,
            decision_index=0,
        )
    except (ImportError, AttributeError):  # Detached evaluator compatibility.
        return stable_int_seed(
            "sha256_stage_keyed_v1",
            protocol_id,
            campaign_id,
            int(campaign_seed),
            task_id,
            int(repeat_id),
            int(matched_seed),
            stage,
            0,
        )
