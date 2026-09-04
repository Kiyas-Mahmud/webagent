"""Export the pinned ordered public WebArena task set for Table 2 handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.webarena_export import (
    PINNED_BROWSERGYM_WEBARENA_VERSION,
    PINNED_LIBWEBARENA_VERSION,
    PINNED_LIBWEBARENA_WHEEL_SHA256,
    PINNED_TASK_DEFINITION_VERSION,
    PINNED_TASK_MEMBER,
    PINNED_TASK_SOURCE_SHA256,
    WEBARENA_RUNTIME_START_STATE_FIELDS,
    load_url_map,
    write_public_pilot_task_export,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-task-source",
        type=Path,
        required=True,
        help="pinned test.raw.json or libwebarena wheel containing it",
    )
    parser.add_argument(
        "--authorized-raw-json-sha256",
        default=None,
        help=(
            "explicit separate authorization for an extracted raw JSON source; "
            "omit when using the pinned libwebarena wheel"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/table2/pilot/task_manifest.json",
    )
    parser.add_argument(
        "--url-map",
        type=Path,
        required=True,
        help="operator-owned JSON mapping of the five WebArena URL tokens",
    )
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--benchmark-version",
        default=PINNED_BROWSERGYM_WEBARENA_VERSION,
    )
    parser.add_argument(
        "--task-definition-version",
        default=PINNED_TASK_DEFINITION_VERSION,
    )
    parser.add_argument("--evaluator-id", required=True)
    parser.add_argument(
        "--evaluator-version",
        default=f"libwebarena-{PINNED_LIBWEBARENA_VERSION}",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_public_pilot_task_export(
        args.output,
        source=args.upstream_task_source,
        registry_path=args.registry,
        site_url_map=load_url_map(args.url_map),
        snapshot_id=args.snapshot_id,
        benchmark_version=args.benchmark_version,
        task_definition_version=args.task_definition_version,
        evaluator_id=args.evaluator_id,
        evaluator_version=args.evaluator_version,
        expected_source_sha256=PINNED_TASK_SOURCE_SHA256,
        expected_container_sha256=PINNED_LIBWEBARENA_WHEEL_SHA256,
        authorized_raw_json_sha256=args.authorized_raw_json_sha256,
        archive_member=PINNED_TASK_MEMBER,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "evidence_label": "PILOT_ONLY",
                "paper_table_status": "N/R",
                "output": str(path.resolve()),
                "task_count": len(value["tasks"]),
                "runtime_start_state_fields": sorted(
                    WEBARENA_RUNTIME_START_STATE_FIELDS
                ),
                "source_container_sha256": value["source"][
                    "container_sha256"
                ],
                "source_sha256": value["source"]["task_source_sha256"],
                "site_url_map_sha256": value["site_url_map_sha256"],
                "resolved_task_set_sha256": value["resolved_task_set_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
