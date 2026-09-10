"""Generate guarded Table 2 metrics, statistics, and publication tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.summary import summarize_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "exact redacted output directory; when omitted, drafts use "
            "results/table2/<campaign_id>/draft and adjudicated output uses "
            "results/table2/<campaign_id>"
        ),
    )
    parser.add_argument(
        "--draft-pilot",
        action="store_true",
        help=(
            "emit explicitly DRAFT_PILOT_ONLY artifacts before blinded human "
            "adjudication; omit this flag for the adjudication-gated final pilot export"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_campaign(
        args.campaign_dir,
        results_dir=args.results_dir,
        draft_pilot=args.draft_pilot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
