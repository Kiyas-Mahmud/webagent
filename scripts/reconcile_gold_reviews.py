#!/usr/bin/env python3
"""Reconcile two-person Web-Gold-40K review logs without touching source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Sequence

from web_agent.data.review_reconciliation import reconcile_reviews
from web_agent.data.review_session import load_queue_csv, load_review_events


SECONDARY_METADATA_FIELDS = (
    "secondary_review_for_event_id",
    "primary_reviewer_id",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox-queue", type=Path, required=True)
    parser.add_argument("--weak-queue", type=Path, required=True)
    parser.add_argument(
        "--reviewer-log",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every Reviewer A/B bbox/weak immutable event CSV.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero unless every required review agrees and is resolved.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[dict],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def union_fields(rows: Sequence[dict], fallback: Sequence[str]) -> list[str]:
    fields = list(fallback)
    seen = set(fields)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [args.bbox_queue, args.weak_queue, *args.reviewer_log]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required review inputs not found: {missing}")

    bbox_rows = load_queue_csv(args.bbox_queue)
    weak_rows = load_queue_csv(args.weak_queue)
    events = [
        event
        for log_path in args.reviewer_log
        for event in load_review_events(log_path)
    ]
    result = reconcile_reviews(bbox_rows, weak_rows, events)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "review_reconciliation_report.json"
    report = {
        **result["report"],
        "inputs": {
            str(path): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in inputs
        },
        "outputs": {},
    }

    artifacts: list[Path] = []

    def emit(
        filename: str,
        rows: Sequence[dict],
        fieldnames: Sequence[str],
    ) -> None:
        path = output_dir / filename
        write_csv(path, rows, fieldnames=fieldnames)
        artifacts.append(path)
        report["outputs"][filename] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
        }

    bbox_fields = union_fields(
        bbox_rows,
        ("split", "sample_id", "task_id", "reviewer_assignment"),
    )
    weak_fields = union_fields(
        weak_rows,
        ("split", "sample_id", "task_id", "reviewer_assignment"),
    )
    emit(
        "bbox_secondary_review_queue.csv",
        result["secondary_rows"]["bbox"],
        union_fields(
            result["secondary_rows"]["bbox"],
            (*bbox_fields, *SECONDARY_METADATA_FIELDS),
        ),
    )
    emit(
        "weak_secondary_review_queue.csv",
        result["secondary_rows"]["weak"],
        union_fields(
            result["secondary_rows"]["weak"],
            (*weak_fields, *SECONDARY_METADATA_FIELDS),
        ),
    )
    emit(
        "missing_primary_reviews.csv",
        result["missing_primary"],
        ("queue_kind", "sample_id", "task_id", "required_reviewer_id"),
    )
    emit(
        "agreement_pairs.csv",
        result["agreement_pairs"],
        union_fields(
            result["agreement_pairs"],
            (
                "queue_kind",
                "sample_id",
                "task_id",
                "pair_reason",
                "event_id_a",
                "event_id_b",
            ),
        ),
    )
    emit(
        "review_disagreements.csv",
        result["disagreements"],
        (
            "queue_kind",
            "sample_id",
            "task_id",
            "event_id_a",
            "event_id_b",
            "decision_a",
            "decision_b",
            "reason",
        ),
    )
    emit(
        "unresolved_reviews.csv",
        result["unresolved"],
        ("queue_kind", "sample_id", "task_id", "reason"),
    )
    emit(
        "approved_corrections.csv",
        result["corrections"],
        union_fields(
            result["corrections"],
            (
                "queue_kind",
                "split",
                "sample_id",
                "task_id",
                "step_index",
            ),
        ),
    )
    emit(
        "finalized_dispositions.csv",
        result["dispositions"],
        (
            "queue_kind",
            "split",
            "sample_id",
            "task_id",
            "final_review_decision",
            "primary_event_id",
            "confirmation_event_id",
        ),
    )

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifacts.append(report_path)
    archive_path = output_dir / "gold_review_reconciliation_package.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in artifacts:
            archive.write(path, arcname=path.name)

    print(json.dumps(report, indent=2, sort_keys=True))
    print("Reconciliation package:", archive_path)
    if args.require_pass and report["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
