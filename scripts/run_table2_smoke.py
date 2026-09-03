"""Run the deterministic, browser-free E0--E3 Table 2 engineering smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.benchmarks.fixture import run_all_systems_fixture_smoke
from web_agent.benchmarks.recovery_fixture import (
    run_failure_memory_intervention_smoke,
    run_recovery_diagnostic_fixture_campaign,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty directory for ENGINEERING_SMOKE_ONLY runtime logs",
    )
    parser.add_argument("--campaign-id", default="table2-fixture-smoke")
    parser.add_argument("--matched-seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("success-chain", "failure-memory", "recovery-60"),
        default="success-chain",
        help=(
            "success-chain runs the six-action E0-E3 smoke; failure-memory "
            "exercises P1/P4; recovery-60 runs all 15 diagnostic blocks"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.matched_seed < 0:
        raise ValueError("--matched-seed must be non-negative")
    if args.mode == "success-chain":
        summaries = run_all_systems_fixture_smoke(
            args.output_dir,
            campaign_id=args.campaign_id,
            matched_seed=args.matched_seed,
        )
        payload = {
            "systems": {
                system_id.value: summary.to_dict()
                for system_id, summary in summaries.items()
            }
        }
    elif args.mode == "failure-memory":
        summaries = run_failure_memory_intervention_smoke(
            args.output_dir,
            campaign_id=args.campaign_id,
            matched_seed=args.matched_seed,
        )
        payload = {
            "systems": {
                system_id.value: summary.to_dict()
                for system_id, summary in summaries.items()
            },
            "assertions_file": str(
                (args.output_dir / "smoke_assertions.json").resolve()
            ),
        }
    else:
        results = run_recovery_diagnostic_fixture_campaign(
            args.output_dir,
            campaign_id=args.campaign_id,
            matched_seed=args.matched_seed,
        )
        payload = {
            "registered_scenarios": 15,
            "systems_per_scenario": 4,
            "episode_count": len(results),
            "diagnostic_summary": str(
                (args.output_dir / "diagnostic_summary.json").resolve()
            ),
        }
    print(
        json.dumps(
            {
                "status": "PASS",
                "evidence_label": "ENGINEERING_SMOKE_ONLY",
                "paper_table_status": "N/R",
                "mode": args.mode,
                "output_dir": str(args.output_dir.resolve()),
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
