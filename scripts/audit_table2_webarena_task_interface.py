"""Audit the frozen WebArena task set against the PC-01 action interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import read_json
from web_agent.eval.table2.task_interface_audit import (
    write_webarena_task_interface_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-export",
        type=Path,
        required=True,
        help="authenticated resolved WebArena public-task export",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_webarena_task_interface_audit(
        args.output,
        task_export=read_json(args.task_export),
    )
    audit = read_json(output)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "handoff_eligible": audit["handoff_eligible"],
                "output": str(output.resolve()),
                "task_export_content_sha256": audit[
                    "task_export_content_sha256"
                ],
                "resolved_task_set_sha256": audit[
                    "resolved_task_set_sha256"
                ],
                "task_count": audit["task_count"],
                "compatible_task_count": audit["compatible_task_count"],
                "incompatible_task_count": audit[
                    "incompatible_task_count"
                ],
                "assistant_answer_required_task_count": audit[
                    "assistant_answer_required_task_count"
                ],
                "page_state_only_task_count": audit[
                    "page_state_only_task_count"
                ],
                "fuzzy_string_match_task_count": audit[
                    "fuzzy_string_match_task_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
