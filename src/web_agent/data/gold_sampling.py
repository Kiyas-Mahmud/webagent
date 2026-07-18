"""Pure, deterministic sampling helpers for Gold 40K experiments.

This module deliberately has no torch dependency.  Subset construction and the
batch-index schedule can therefore be regression-tested on a CPU-only machine.
The actual DataLoader adapter lives in :mod:`web_agent.data.gold_dataloader`.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Callable, Hashable, Iterable, Sequence


def _views(record: dict) -> tuple[dict, dict, dict]:
    if "inputs" in record and "labels" in record:
        return record["inputs"], record["labels"], record.get("meta", {})
    return record, record, record


def recovery_attempted(record: dict) -> bool:
    """Return whether a real recovery was attempted for this row."""
    _, labels, _ = _views(record)
    return labels.get("recovery_success") is not None


def gold_joint_stratum(record: dict) -> tuple[Hashable, ...]:
    """Joint mini-training stratum covering every supervised head.

    The tuple is intentionally made only from labels/masks.  It is used to pick
    training rows, never as model input.
    """
    _, labels, _ = _views(record)
    attempted = labels.get("recovery_success") is not None
    recovery_outcome = (
        "NOT_ATTEMPTED"
        if not attempted
        else "SUCCESS" if labels.get("recovery_success") is True else "FAILURE"
    )
    return (
        labels["failure_type_4"],
        labels["action_type"],
        bool(attempted),
        labels["recovery_strategy"],
        recovery_outcome,
        bool(labels.get("memory_update_flag") or False),
        labels.get("action_target_bbox") is not None,
    )


def proportional_stratified_subsample(
    records: Sequence[dict],
    n: int,
    *,
    key: Callable[[dict], Hashable],
    seed: int = 42,
) -> list[dict]:
    """Select exactly ``n`` rows with deterministic proportional joint coverage.

    Every observed stratum receives one row when ``n`` permits it.  Remaining
    slots use Hamilton-style largest-remainder allocation, bounded by each
    stratum's capacity.  Rows are never duplicated.
    """
    if n <= 0:
        raise ValueError("sample size must be positive")
    if n >= len(records):
        return list(records)

    rng = random.Random(seed)
    buckets: dict[Hashable, list[dict]] = defaultdict(list)
    for record in records:
        buckets[key(record)].append(record)
    for rows in buckets.values():
        rng.shuffle(rows)

    keys = sorted(buckets, key=repr)
    quotas = {bucket_key: 0 for bucket_key in keys}
    remaining = n

    if len(keys) <= n:
        for bucket_key in keys:
            quotas[bucket_key] = 1
        remaining -= len(keys)

    capacities = {
        bucket_key: len(buckets[bucket_key]) - quotas[bucket_key]
        for bucket_key in keys
    }
    capacity_total = sum(capacities.values())
    if remaining > 0 and capacity_total > 0:
        exact = {
            bucket_key: remaining * capacities[bucket_key] / capacity_total
            for bucket_key in keys
        }
        for bucket_key in keys:
            additional = min(capacities[bucket_key], math.floor(exact[bucket_key]))
            quotas[bucket_key] += additional
            capacities[bucket_key] -= additional
        remaining = n - sum(quotas.values())

        ranked = sorted(
            keys,
            key=lambda bucket_key: (
                exact[bucket_key] - math.floor(exact[bucket_key]),
                len(buckets[bucket_key]),
                repr(bucket_key),
            ),
            reverse=True,
        )
        while remaining:
            progressed = False
            for bucket_key in ranked:
                if capacities[bucket_key] <= 0:
                    continue
                quotas[bucket_key] += 1
                capacities[bucket_key] -= 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                raise RuntimeError("unable to allocate the requested stratified sample")

    selected: list[dict] = []
    for bucket_key in keys:
        selected.extend(buckets[bucket_key][: quotas[bucket_key]])
    rng.shuffle(selected)
    if len(selected) != n or len({id(row) for row in selected}) != n:
        raise AssertionError("stratified sampling must return unique rows of exact size")
    return selected


def select_recovery_aware_gold_subset(
    records: Sequence[dict], n: int, seed: int = 42,
) -> list[dict]:
    """Proportional Gold mini subset with joint task/recovery/mask coverage."""
    return proportional_stratified_subsample(records, n, key=gold_joint_stratum, seed=seed)


def recovery_aware_batch_indices(
    records: Sequence[dict], batch_size: int, *, seed: int = 42, epoch: int = 0,
) -> list[list[int]]:
    """Arrange every row once while spreading attempted recoveries across batches.

    This is distribution-preserving: no row is added, dropped, or duplicated.
    With the Gold 40K attempt rate (~23.5%) and a physical batch size of four,
    almost every batch receives one attempted recovery.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not records:
        return []

    rng = random.Random(seed + 1_000_003 * epoch)
    attempted = [i for i, record in enumerate(records) if recovery_attempted(record)]
    ordinary = [i for i, record in enumerate(records) if not recovery_attempted(record)]
    rng.shuffle(attempted)
    rng.shuffle(ordinary)

    batch_count = math.ceil(len(records) / batch_size)
    batches: list[list[int]] = [[] for _ in range(batch_count)]

    def distribute(indices: Iterable[int]) -> None:
        cursor = 0
        for index in indices:
            while len(batches[cursor % batch_count]) >= batch_size:
                cursor += 1
            batches[cursor % batch_count].append(index)
            cursor += 1

    distribute(attempted)
    distribute(ordinary)
    rng.shuffle(batches)
    for batch in batches:
        rng.shuffle(batch)

    flat = [index for batch in batches for index in batch]
    if len(flat) != len(records) or sorted(flat) != list(range(len(records))):
        raise AssertionError("recovery-aware batching must use every row exactly once")
    if any(not batch or len(batch) > batch_size for batch in batches):
        raise AssertionError("recovery-aware batching produced an invalid batch")
    return batches


def sqrt_inverse_frequency_weights(
    labels: Sequence[int], num_classes: int, *, cap: float = 3.0,
) -> list[float]:
    """Normalized sqrt-inverse-frequency weights; absent classes stay zero."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if cap <= 0:
        raise ValueError("weight cap must be positive")
    counts = Counter(int(label) for label in labels)
    present = [label for label in range(num_classes) if counts.get(label, 0)]
    weights = [0.0] * num_classes
    if not present:
        return weights
    raw = {label: 1.0 / math.sqrt(counts[label]) for label in present}
    # Find a shared multiplier whose capped values retain mean 1.0. This avoids
    # the common cap-then-renormalize bug that can push a capped weight above cap.
    low = 0.0
    high = cap / min(raw.values())
    for _ in range(80):
        multiplier = (low + high) / 2.0
        mean = sum(min(multiplier * raw[label], cap) for label in present) / len(present)
        if mean < 1.0:
            low = multiplier
        else:
            high = multiplier
    multiplier = (low + high) / 2.0
    for label in present:
        weights[label] = min(multiplier * raw[label], cap)
    return weights


def label_distribution(records: Sequence[dict]) -> dict[str, dict[str, int]]:
    """Return JSON-ready selected-subset distributions for experiment evidence."""
    fields: dict[str, Counter] = {
        "failure_type": Counter(),
        "action_type": Counter(),
        "recovery_attempted": Counter(),
        "recovery_strategy": Counter(),
        "recovery_success": Counter(),
        "memory_update_flag": Counter(),
        "bbox_available": Counter(),
    }
    for record in records:
        _, labels, _ = _views(record)
        attempted = labels.get("recovery_success") is not None
        fields["failure_type"][str(labels["failure_type_4"])] += 1
        fields["action_type"][str(labels["action_type"])] += 1
        fields["recovery_attempted"][str(bool(attempted))] += 1
        fields["recovery_strategy"][str(labels["recovery_strategy"])] += 1
        recovery_value = labels.get("recovery_success")
        recovery_key = "NOT_ATTEMPTED" if recovery_value is None else str(bool(recovery_value))
        fields["recovery_success"][recovery_key] += 1
        fields["memory_update_flag"][str(bool(labels.get("memory_update_flag") or False))] += 1
        fields["bbox_available"][str(labels.get("action_target_bbox") is not None)] += 1
    return {
        field: dict(sorted(counts.items()))
        for field, counts in fields.items()
    }
