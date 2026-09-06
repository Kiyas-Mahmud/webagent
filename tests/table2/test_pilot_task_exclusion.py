from __future__ import annotations

from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.pilot_task_exclusion import (
    PILOT_TASK_EXCLUSION_IDENTITY_VERSION,
    PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH,
    load_pilot_task_exclusion_authority,
    pilot_task_exclusion_provenance,
    validate_locked_final_pilot_exclusion,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _authority():
    return load_pilot_task_exclusion_authority(
        REPOSITORY_ROOT / PILOT_TASK_EXCLUSION_REGISTRY_RELATIVE_PATH
    )


def test_tracked_registry_is_the_permanent_combined_exclusion_authority() -> None:
    authority = _authority()

    assert authority.benchmark == "webarena"
    active = {
        102, 156, 157, 158, 159, 238, 258, 260, 269, 274,
        283, 284, 298, 324, 356, 369, 370, 371, 372, 373,
        374, 375, 377, 378, 379, 380, 381, 676, 677, 678,
        679, 680, 704, 705, 706, 707, 708, 709, 710, 711,
        712, 757, 758, 761, 762, 763, 764, 765, 766, 767,
    }
    expected = frozenset(set(range(50)) | active)
    assert authority.upstream_indices == expected
    assert authority.benchmark_task_ids == frozenset(str(index) for index in expected)
    assert pilot_task_exclusion_provenance(authority)["identity_version"] == (
        PILOT_TASK_EXCLUSION_IDENTITY_VERSION
    )


def test_renamed_locked_final_task_cannot_hide_pilot_upstream_index() -> None:
    renamed_overlap = {
        "task_id": "final-study-task-alpha",
        "upstream_index": 17,
        "benchmark_task_id": "17",
        "instruction": "locked content that exclusion validation must not inspect",
        "task_config": {"private": "must remain outside exclusion logic"},
    }

    with pytest.raises(SchemaError, match="permanent pilot exclusion"):
        validate_locked_final_pilot_exclusion(
            [renamed_overlap],
            task_metadata={"benchmark": "WebArena"},
            authority=_authority(),
        )


def test_changed_local_index_cannot_hide_pilot_benchmark_task_identity() -> None:
    disguised_overlap = {
        "task_id": "renamed-and-renumbered-local-task",
        "upstream_index": 117,
        "benchmark_task_id": "17",
    }

    with pytest.raises(SchemaError, match=r"benchmark_task_ids=\['17'\]"):
        validate_locked_final_pilot_exclusion(
            [disguised_overlap],
            task_metadata={"benchmark": "webarena"},
            authority=_authority(),
        )


def test_active_page_state_pilot_task_is_also_permanently_excluded() -> None:
    with pytest.raises(SchemaError, match="permanent pilot exclusion"):
        validate_locked_final_pilot_exclusion(
            [
                {
                    "task_id": "renamed-active-pilot-task",
                    "upstream_index": 676,
                    "benchmark_task_id": "676",
                }
            ],
            task_metadata={"benchmark": "webarena"},
            authority=_authority(),
        )


def test_disjoint_locked_final_metadata_passes_without_reading_task_content() -> None:
    tasks = [
        {
            "task_id": "final-local-name-a",
            "upstream_index": 100,
            "benchmark_task_id": "100",
            "instruction": object(),
            "evaluator": object(),
        },
        {
            "task_id": "final-local-name-b",
            "upstream_index": 101,
            "benchmark_task_id": "101",
            "task_config": object(),
        },
    ]

    validate_locked_final_pilot_exclusion(
        tasks,
        task_metadata={"benchmark": "webarena"},
        authority=_authority(),
    )


def test_exclusion_identity_is_scoped_to_the_registered_benchmark() -> None:
    validate_locked_final_pilot_exclusion(
        [
            {
                "task_id": "another-benchmark-local-task",
                "upstream_index": 17,
                "benchmark_task_id": "17",
            }
        ],
        task_metadata={"benchmark": "not-webarena"},
        authority=_authority(),
    )
