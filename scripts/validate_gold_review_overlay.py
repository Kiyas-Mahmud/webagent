#!/usr/bin/env python3
"""Validate a passed review overlay against train/validation without writing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from audit_gold_existing_data import ImageReader
from validate_kaggle_gold import find_data_source, load_records
from web_agent.data.improvement_audit import bbox_geometry_reasons
from web_agent.data.review_overlay import ReviewOverlay


TARGET_ACTIONS = {"SCROLL", "SELECT", "NAVIGATE"}
TARGET_FAILURES = {"LOOP_DETECTED"}
TARGET_RECOVERIES = {"BACKTRACK", "RETRY", "ABORT"}


def view(record: dict) -> tuple[dict, dict, dict]:
    if "inputs" in record and "labels" in record:
        return record["inputs"], record["labels"], record.get("meta", {})
    return record, record, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reconciliation-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_counts(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        _, labels, _ = view(record)
        action = str(labels.get("action_type") or "")
        failure = str(labels.get("failure_type_4") or "")
        recovery = str(labels.get("recovery_strategy") or "")
        if action in TARGET_ACTIONS:
            counts[f"ACTION:{action}"] += 1
        if failure in TARGET_FAILURES:
            counts[f"FAILURE:{failure}"] += 1
        if recovery in TARGET_RECOVERIES:
            counts[f"RECOVERY:{recovery}"] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    source = find_data_source(args.data_root)
    overlay = ReviewOverlay.load(args.reconciliation_dir)
    image_reader = ImageReader(source)
    split_results = {}
    bbox_failures = []

    for split, filename in (
        ("train", "split_train.json"),
        ("val", "split_val.json"),
    ):
        records = load_records(source, filename)
        output, application = overlay.apply(records, split=split)
        for record in output:
            inputs, labels, meta = view(record)
            marker = meta.get("targeted_review_overlay") or {}
            if "action_target_bbox" not in marker.get("fields", []):
                continue
            image_size, error = image_reader.info(inputs.get("state_before"))
            reasons = (
                bbox_geometry_reasons(
                    labels.get("action_target_bbox"),
                    image_size,
                )
                if image_size
                else [f"image_error:{error or 'unavailable'}"]
            )
            if reasons:
                bbox_failures.append({
                    "split": split,
                    "sample_id": str(meta.get("sample_id") or ""),
                    "reasons": reasons,
                })
        split_results[split] = {
            **application,
            "target_counts_before": target_counts(records),
            "target_counts_after": target_counts(output),
        }

    report = {
        "status": "PASS" if not bbox_failures else "FAIL",
        "controlled_mini_permitted": not bbox_failures,
        "publication_ready": False,
        "source": getattr(source, "description", str(source)),
        "reconciliation_report_sha256": sha256_file(
            args.reconciliation_dir / "review_reconciliation_report.json"
        ),
        "test_rows_read": 0,
        "source_records_mutated": False,
        "corrected_bbox_failures": bbox_failures,
        "splits": split_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
