"""Permanent metadata-only exclusion authority for Table 2 pilot tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import SchemaError, read_json, sha256_file


PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH = Path(
    "benchmarks/table2/pilot/final_exclusion_registry_v2.json"
)
FROZEN_PILOT_TASK_EXCLUSION_RELATIVE_PATH = Path(
    "frozen/benchmark/pilot_task_exclusion_registry.json"
)
PILOT_TASK_EXCLUSION_IDENTITY_VERSION = (
    "benchmark_name_plus_upstream_index_and_benchmark_task_id_v2"
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
    """Load the tracked 100-task authority without opening final-task content."""

    source = Path(path)
    value = read_json(source)
    exact_metadata = {
        "schema_version": "2.0",
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
    if value.get("schema_version") != "2.0":
        raise SchemaError("pilot exclusion registry must use aggregate schema 2.0")
    if value.get("identity_version") != PILOT_TASK_EXCLUSION_IDENTITY_VERSION:
        raise SchemaError("pilot exclusion registry changed stable identity semantics")
    rows = value.get("excluded_upstream_indices")
    benchmark_ids = value.get("excluded_benchmark_task_ids")
    required_count = value.get("required_task_count")
    if (
        type(required_count) is not int
        or required_count != 100
        or not isinstance(rows, list)
        or len(rows) != required_count
        or not isinstance(benchmark_ids, list)
        or len(benchmark_ids) != required_count
    ):
        raise SchemaError("pilot exclusion registry must contain exactly 100 tasks")

    indices: set[int] = set()
    benchmark_task_ids: set[str] = set()
    for position, (index, benchmark_task_id) in enumerate(zip(rows, benchmark_ids)):
        if type(index) is not int or index < 0:
            raise SchemaError(
                f"pilot exclusion task {position} has an invalid upstream index"
            )
        if not isinstance(benchmark_task_id, str) or not benchmark_task_id.strip():
            raise SchemaError(
                f"pilot exclusion task {position} lacks a benchmark task identity"
            )
        stable_id = benchmark_task_id.strip()
        if index in indices or stable_id in benchmark_task_ids:
            raise SchemaError("pilot exclusion registry contains duplicate stable identity")
        indices.add(index)
        benchmark_task_ids.add(stable_id)
    historical = set(range(50))
    active = {
        102, 156, 157, 158, 159, 238, 258, 260, 269, 274,
        283, 284, 298, 324, 356, 369, 370, 371, 372, 373,
        374, 375, 377, 378, 379, 380, 381, 676, 677, 678,
        679, 680, 704, 705, 706, 707, 708, 709, 710, 711,
        712, 757, 758, 761, 762, 763, 764, 765, 766, 767,
    }
    if indices != historical | active:
        raise SchemaError(
            "pilot exclusion registry must preserve both registered 50-task sets"
        )
    if benchmark_task_ids != {str(index) for index in indices}:
        raise SchemaError(
            "pilot exclusion benchmark task identities differ from upstream identities"
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
