"""Ordered authority for the active Table 2 public-development task set.

This contract is deliberately separate from :mod:`pilot_task_exclusion`.
The active 50-task set may be replaced through an explicit research decision,
whereas final-evaluation exclusions are cumulative and must not silently lose
the original public development tasks when that happens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .common import SchemaError, read_json, sha256_file


PUBLIC_DEVELOPMENT_TASK_COUNT = 50
GENERIC_ORDERED_SELECTION_RULE = (
    "exact ordered upstream task identities declared by the tracked registry"
)


@dataclass(frozen=True, slots=True)
class PublicDevelopmentTaskIdentity:
    """One stable task identity in its preregistered execution order."""

    task_id: str
    upstream_index: int
    benchmark_task_id: str


@dataclass(frozen=True, slots=True)
class PublicDevelopmentTaskRegistry:
    """Authenticated ordered public-development task authority."""

    manifest_id: str
    benchmark: str
    selection_rule: str
    tasks: tuple[PublicDevelopmentTaskIdentity, ...]
    registry_sha256: str

    @property
    def ordered_upstream_indices(self) -> tuple[int, ...]:
        return tuple(task.upstream_index for task in self.tasks)

    @property
    def upstream_index_set(self) -> frozenset[int]:
        return frozenset(self.ordered_upstream_indices)


def load_public_development_task_registry(
    path: str | Path,
) -> PublicDevelopmentTaskRegistry:
    """Load exactly 50 unique tasks while preserving declared row order.

    Upstream indices need not be contiguous.  Their order is authority data,
    not a value that downstream code may sort or reconstruct.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise SchemaError("public-development task registry must not be a symlink")
    value = read_json(source)
    exact_metadata = {
        "schema_version": "1.0",
        "benchmark": "webarena",
        "partition": "development",
        "registration_status": "FROZEN_DEVELOPMENT_EXCLUSION",
        "required_task_count": PUBLIC_DEVELOPMENT_TASK_COUNT,
        "final_paper_evaluation_eligible": False,
        "locked_test_content": False,
    }
    for key, expected in exact_metadata.items():
        if value.get(key) != expected:
            raise SchemaError(
                f"public-development task registry {key} differs from authority"
            )

    manifest_id = value.get("manifest_id")
    benchmark = value.get("benchmark")
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        raise SchemaError("public-development task registry lacks a manifest identity")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise SchemaError("public-development task registry lacks a benchmark identity")
    selection_rule = value.get("selection_rule", GENERIC_ORDERED_SELECTION_RULE)
    if not isinstance(selection_rule, str) or not selection_rule.strip():
        raise SchemaError("public-development task registry lacks a selection rule")

    rows = value.get("tasks")
    if not isinstance(rows, list) or len(rows) != PUBLIC_DEVELOPMENT_TASK_COUNT:
        raise SchemaError(
            "public-development task registry must contain exactly 50 tasks"
        )

    tasks: list[PublicDevelopmentTaskIdentity] = []
    upstream_indices: set[int] = set()
    local_ids: set[str] = set()
    benchmark_ids: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SchemaError(
                f"public-development registry task {position} must be an object"
            )
        upstream_index = row.get("upstream_index")
        task_id = row.get("task_id")
        if type(upstream_index) is not int or upstream_index < 0:
            raise SchemaError(
                f"public-development registry task {position} has an invalid "
                "upstream index"
            )
        if not isinstance(task_id, str) or not task_id.strip():
            raise SchemaError(
                f"public-development registry task {position} lacks a task ID"
            )
        normalized_task_id = task_id.strip()
        benchmark_task_id = str(
            row.get("benchmark_task_id", upstream_index)
        ).strip()
        if not benchmark_task_id:
            raise SchemaError(
                f"public-development registry task {position} lacks a benchmark "
                "task identity"
            )
        if (
            upstream_index in upstream_indices
            or normalized_task_id in local_ids
            or benchmark_task_id in benchmark_ids
        ):
            raise SchemaError(
                "public-development task registry contains duplicate stable identity"
            )
        upstream_indices.add(upstream_index)
        local_ids.add(normalized_task_id)
        benchmark_ids.add(benchmark_task_id)
        tasks.append(
            PublicDevelopmentTaskIdentity(
                task_id=normalized_task_id,
                upstream_index=upstream_index,
                benchmark_task_id=benchmark_task_id,
            )
        )

    return PublicDevelopmentTaskRegistry(
        manifest_id=manifest_id.strip(),
        benchmark=benchmark.strip().casefold(),
        selection_rule=selection_rule.strip(),
        tasks=tuple(tasks),
        registry_sha256=sha256_file(source),
    )
