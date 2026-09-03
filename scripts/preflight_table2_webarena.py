"""Run the pinned BrowserGym/WebArena/Chromium host compatibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import atomic_write_json
from web_agent.eval.table2.webarena_preflight import (
    load_service_url_map,
    run_webarena_host_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-url-map",
        type=Path,
        required=True,
        help="JSON mapping of all seven BrowserGym WA_* service origins",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-reset-task-index", type=int, default=0)
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
    result = run_webarena_host_preflight(
        service_url_map=load_service_url_map(args.service_url_map),
        run_live_reset=not args.skip_live_reset,
        live_reset_task_index=args.live_reset_task_index,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
