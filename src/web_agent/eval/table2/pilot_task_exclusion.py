"""Permanent metadata-only exclusion authority for Table 2 pilot tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import SchemaError, read_json, sha256_file


PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH = Path(
    "benchmarks/table2/pilot/task_manifest.json"
)
FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH = Path(
    "frozen/benchmark/pilot_task_exclusion_registry.json"
)
PILOT_TASK_EXCLUSION_IDENTITY_VERSION = (
    "benchmark_name_plus_upstream_index_and_benchmark_task_id_v1"
)


@dataclass(frozen=True, slots=True)
class PilotTaskExclusionAuthority:
    manifest_id: str
    benchmark: str
    upstream_indices: frozenset[int]
    benchmark_task_ids: frozenset[str]
    registry_sha256: str


def pilot_task_exclusion_provenance(
    authority: PilotTaskExclusionAuthority,
) -> dict[str, Any]:
    """Canonical campaign-provenance binding for the permanent authority."""

    return {
        "identity_version": PILOT_TASK_EXCLUSION_IDENTITY_VERSION,
        "source_relative_path": str(PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH),
        "frozen_relative_path": str(FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH),
        "manifest_id": authority.manifest_id,
        "benchmark": authority.benchmark,
        "excluded_task_count": len(authority.upstream_indices),
        "registry_sha256": authority.registry_sha256,
    }


def load_pilot_task_exclusion_authority(
    path: str | Path,
) -> PilotTaskExclusionAuthority:
    """Load the tracked 50-task registry without opening any final task content."""

    source = Path(path)
    value = read_json(source)
    exact_metadata = {
        "schema_version": "1.0",
        "partition": "development",
        "registration_status": "FROZEN_DEVELOPMENT_EXCLUSION",
        "final_paper_evaluation_eligible": False,
        "locked_test_content": False,
    }
    for key, expected in exact_metadata.items():
        if value.get(key) != expected:
            raise SchemaError(
                f"pilot exclusion registry {key} differs from permanent authority"
            )
    manifest_id = value.get("manifest_id")
    benchmark = value.get("benchmark")
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        raise SchemaError("pilot exclusion registry lacks a manifest identity")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise SchemaError("pilot exclusion registry lacks a benchmark identity")
    rows = value.get("tasks")
    required_count = value.get("required_task_count")
    if (
        type(required_count) is not int
        or required_count != 50
        or not isinstance(rows, list)
        or len(rows) != required_count
    ):
        raise SchemaError("pilot exclusion registry must contain exactly 50 tasks")

    indices: set[int] = set()
    benchmark_task_ids: set[str] = set()
    local_ids: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SchemaError(f"pilot exclusion task {position} must be an object")
        index = row.get("upstream_index")
        task_id = row.get("task_id")
        if type(index) is not int or index < 0:
            raise SchemaError(
                f"pilot exclusion task {position} has an invalid upstream index"
            )
        if not isinstance(task_id, str) or not task_id.strip():
            raise SchemaError(f"pilot exclusion task {position} lacks a task ID")
        benchmark_task_id = str(row.get("benchmark_task_id", index)).strip()
        if not benchmark_task_id:
            raise SchemaError(
                f"pilot exclusion task {position} lacks a benchmark task identity"
            )
        if index in indices or task_id in local_ids or benchmark_task_id in benchmark_task_ids:
            raise SchemaError("pilot exclusion registry contains duplicate stable identity")
        indices.add(index)
        local_ids.add(task_id)
        benchmark_task_ids.add(benchmark_task_id)
    if indices != set(range(50)):
        raise SchemaError(
            "pilot exclusion registry must preserve upstream WebArena indices 0--49"
        )
    return PilotTaskExclusionAuthority(
        manifest_id=manifest_id,
        benchmark=benchmark.strip().casefold(),
        upstream_indices=frozenset(indices),
        benchmark_task_ids=frozenset(benchmark_task_ids),
        registry_sha256=sha256_file(source),
    )


def validate_locked_final_pilot_exclusion(
    tasks: Sequence[Mapping[str, Any]],
    *,
    task_metadata: Mapping[str, Any],
    authority: PilotTaskExclusionAuthority,
) -> None:
    """Reject final-task overlap using stable metadata, never local task names."""

    benchmark = task_metadata.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise SchemaError(
            "locked-final task metadata lacks a stable benchmark identity"
        )
    if benchmark.strip().casefold() != authority.benchmark:
        return

    seen_indices: set[int] = set()
    seen_benchmark_ids: set[str] = set()
    overlap_indices: set[int] = set()
    overlap_benchmark_ids: set[str] = set()
    for position, row in enumerate(tasks):
        # Deliberately access metadata fields only.  Instructions, start state,
        # evaluator labels/configuration, and other locked content are outside
        # this exclusion authority's capability.
        index = row.get("upstream_index")
        benchmark_task_id = row.get("benchmark_task_id")
        if type(index) is not int or index < 0:
            raise SchemaError(
                f"locked-final task {position} lacks an exact upstream index"
            )
        if not isinstance(benchmark_task_id, str) or not benchmark_task_id.strip():
            raise SchemaError(
                f"locked-final task {position} lacks a benchmark task identity"
            )
        stable_id = benchmark_task_id.strip()
        if index in seen_indices or stable_id in seen_benchmark_ids:
            raise SchemaError("locked-final task set repeats a stable benchmark identity")
        seen_indices.add(index)
        seen_benchmark_ids.add(stable_id)
        if index in authority.upstream_indices:
            overlap_indices.add(index)
        if stable_id in authority.benchmark_task_ids:
            overlap_benchmark_ids.add(stable_id)
    if overlap_indices or overlap_benchmark_ids:
        raise SchemaError(
            "locked-final tasks overlap the permanent pilot exclusion registry "
            f"(upstream_indices={sorted(overlap_indices)}, "
            f"benchmark_task_ids={sorted(overlap_benchmark_ids)})"
        )
