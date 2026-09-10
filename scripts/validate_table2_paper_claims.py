"""Validate or initialize the research-locked Table 2 paper-claim package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import sha256_file
from web_agent.eval.table2.paper_claims import (
    EVIDENCE_RULES,
    validate_claim_registry,
    validate_claim_report,
    validate_paper_claim_readiness,
    write_claim_evidence_assessment,
    write_initial_claim_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry = subparsers.add_parser("validate-registry")
    registry.add_argument("--registry", required=True, type=Path)

    initialize = subparsers.add_parser("initialize-report")
    initialize.add_argument("--registry", required=True, type=Path)
    initialize.add_argument("--campaign-manifest", required=True, type=Path)
    initialize.add_argument("--output", required=True, type=Path)

    report = subparsers.add_parser("validate-report")
    report.add_argument("--registry", required=True, type=Path)
    report.add_argument("--campaign-dir", required=True, type=Path)
    report.add_argument("--report", required=True, type=Path)

    assess = subparsers.add_parser("assess-evidence")
    assess.add_argument("--registry", required=True, type=Path)
    assess.add_argument("--campaign-dir", required=True, type=Path)
    assess.add_argument(
        "--kind", required=True, choices=tuple(rule.kind for rule in EVIDENCE_RULES)
    )

    readiness = subparsers.add_parser("validate-readiness")
    readiness.add_argument("--registry", required=True, type=Path)
    readiness.add_argument("--campaign-dir", required=True, type=Path)
    readiness.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-registry":
        value = validate_claim_registry(args.registry)
        result = {
            "status": "PASS",
            "registry_id": value["registry_id"],
            "registry_sha256": sha256_file(args.registry),
            "paper_table_status": value["paper_table_status"],
        }
    elif args.command == "initialize-report":
        output = write_initial_claim_report(
            args.output,
            registry_path=args.registry,
            campaign_manifest_path=args.campaign_manifest,
        )
        result = {
            "status": "N/R",
            "report": str(output.resolve()),
            "report_sha256": sha256_file(output),
        }
    elif args.command == "assess-evidence":
        output = write_claim_evidence_assessment(
            kind=args.kind,
            registry_path=args.registry,
            campaign_dir=args.campaign_dir,
        )
        result = {
            "status": "PASS",
            "evidence_kind": args.kind,
            "assessment": str(output.resolve()),
            "assessment_sha256": sha256_file(output),
        }
    else:
        if args.command == "validate-readiness":
            result = validate_paper_claim_readiness(
                args.report,
                registry_path=args.registry,
                campaign_dir=args.campaign_dir,
            )
        else:
            value = validate_claim_report(
                args.report,
                registry_path=args.registry,
                campaign_dir=args.campaign_dir,
            )
            result = {
                "status": "PASS",
                "claim_readiness_status": "NOT_ISSUED",
                "campaign_id": value["campaign_id"],
                "publication_status": value["publication_status"],
                "paper_table_status": value["paper_table_status"],
                "report_sha256": sha256_file(args.report),
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
