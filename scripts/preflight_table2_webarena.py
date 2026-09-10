"""Run the pinned BrowserGym/WebArena/Chromium host compatibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import SchemaError, atomic_write_json
from web_agent.eval.table2.public_task_registry import (
    load_public_development_task_registry,
)
from web_agent.eval.table2.webarena_preflight import (
    load_service_url_map,
    run_webarena_host_preflight,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-url-map",
        type=Path,
        required=True,
        help="JSON mapping of all seven BrowserGym WA_* service origins",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-registry",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "benchmarks/table2/pilot/task_manifest_page_state_v2.json"
        ),
        help="tracked ordered 50-task public-development registry",
    )
    parser.add_argument("--live-reset-task-index", type=int)
    parser.add_argument(
        "--skip-live-reset",
        action="store_true",
        help="diagnostic only; output remains ineligible for a campaign",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite preflight evidence: {args.output}"
        )
    registry = load_public_development_task_registry(args.task_registry)
    registered_indices = registry.ordered_upstream_indices
    live_reset_task_index = (
        registered_indices[0]
        if args.live_reset_task_index is None
        else args.live_reset_task_index
    )
    if live_reset_task_index != registered_indices[0]:
        raise SchemaError(
            "WebArena host preflight must reset the first task in exact tracked "
            "registry order"
        )
    result = run_webarena_host_preflight(
        service_url_map=load_service_url_map(args.service_url_map),
        run_live_reset=not args.skip_live_reset,
        live_reset_task_index=live_reset_task_index,
        registered_task_indices=registered_indices,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
