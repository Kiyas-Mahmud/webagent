#!/usr/bin/env python3
"""Validate the accepted RETRY/ABORT supplement v2 without reading Gold test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from web_agent.data.recovery_supplement import (
    EXPECTED_ARCHIVE_SHA256,
    validate_supplement,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        help="original Kaggle ZIP; required for archive-level SHA-256 proof",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_supplement(args.supplement_root)
    archive_report = {
        "status": "NOT_CHECKED",
        "expected_sha256": EXPECTED_ARCHIVE_SHA256,
    }
    if args.archive is not None:
        actual = sha256_file(args.archive)
        archive_report = {
            "status": (
                "PASS"
                if actual.lower() == EXPECTED_ARCHIVE_SHA256
                else "FAIL"
            ),
            "path": str(args.archive.resolve()),
            "expected_sha256": EXPECTED_ARCHIVE_SHA256,
            "actual_sha256": actual,
        }
        if archive_report["status"] != "PASS":
            report["errors"].append("Kaggle archive SHA-256 mismatch")
            report["status"] = "FAIL"
    report["archive"] = archive_report
    report["permitted_next_action"] = (
        "SUPPLEMENT_FORWARD_BACKWARD_SANITY"
        if report["status"] == "PASS"
        else "STOP_AND_FIX_REPORTED_PACKAGE_ERRORS"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {args.report.resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
