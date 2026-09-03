"""Strictly validate a frozen Table 2 package without exposing sealed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import atomic_write_json
from web_agent.eval.table2.package_validator import validate_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="validate artifacts already written without requiring every block",
    )
    parser.add_argument("--require-aggregates", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_campaign(
        args.campaign_dir,
        require_complete=not args.allow_incomplete,
        require_aggregates=args.require_aggregates,
    )
    payload = report.to_dict()
    if args.report:
        atomic_write_json(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
