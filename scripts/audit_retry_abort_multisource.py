#!/usr/bin/env python3
"""Audit original+supplement loading without reading the locked test split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from web_agent.config import load_config
from web_agent.data.gold_dataloader import (
    load_gold_split,
    load_gold_split_sources,
)
from web_agent.data.gold_dataset import view
from web_agent.data.recovery_supplement import RECOVERY_ONLY_LOSS_MASKS
from web_agent.data.recovery_transitions import build_recovery_transition_index


EXPECTED_ORIGINAL = {"train": 23_499, "val": 7_861}
EXPECTED_SUPPLEMENT = {"train": 608, "val": 194}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="configs/backbones/qwen2vl_2b_gold_v2_8.yaml",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _identities(records: list[dict], field: str) -> set[str]:
    return {
        str(view(record)[2].get(field))
        for record in records
        if view(record)[2].get(field) not in (None, "")
    }


def _strategy_counts(records: list[dict]) -> dict[str, int]:
    return dict(Counter(
        str(view(record)[1].get("recovery_strategy"))
        for record in records
    ))


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["root"] = str(args.original_root.resolve())
    cfg["data"]["recovery_supplement"].update({
        "enabled": True,
        "root": str(args.supplement_root.resolve()),
        "include_in_primary_validation": False,
    })

    errors: list[str] = []
    split_reports = {}
    for split in ("train", "val"):
        sources = load_gold_split_sources(cfg, split)
        original = sources["original_gold"]
        supplement = sources["retry_abort_supplement_v2"]
        if len(original) != EXPECTED_ORIGINAL[split]:
            errors.append(
                f"{split} original rows: expected {EXPECTED_ORIGINAL[split]}, "
                f"found {len(original)}"
            )
        if len(supplement) != EXPECTED_SUPPLEMENT[split]:
            errors.append(
                f"{split} supplement rows: expected "
                f"{EXPECTED_SUPPLEMENT[split]}, found {len(supplement)}"
            )

        invalid_masks = []
        invalid_sources = []
        for index, record in enumerate(supplement):
            _, _, meta = view(record)
            if meta.get("_source_dataset") != "retry_abort_supplement_v2":
                invalid_sources.append(index)
            if meta.get("_loss_masks") != RECOVERY_ONLY_LOSS_MASKS:
                invalid_masks.append(index)
        if invalid_masks:
            errors.append(f"{split} has {len(invalid_masks)} invalid loss masks")
        if invalid_sources:
            errors.append(f"{split} has {len(invalid_sources)} invalid source tags")

        transitions, transition_report = build_recovery_transition_index(
            supplement
        )
        if len(transitions) != len(supplement):
            errors.append(
                f"{split} direct transitions: expected {len(supplement)}, "
                f"found {len(transitions)}"
            )
        identity_overlap = {}
        for field in ("sample_id", "task_id", "trajectory_id", "session_id"):
            overlap = _identities(original, field) & _identities(
                supplement, field
            )
            identity_overlap[field] = len(overlap)
            if overlap:
                errors.append(
                    f"{split} original/supplement {field} overlap: {len(overlap)}"
                )

        split_reports[split] = {
            "original_rows": len(original),
            "supplement_rows": len(supplement),
            "combined_rows": len(original) + len(supplement),
            "supplement_strategy": _strategy_counts(supplement),
            "supplement_direct_transitions": len(transitions),
            "transition_report": transition_report,
            "original_supplement_identity_overlap": identity_overlap,
            "invalid_source_tags": len(invalid_sources),
            "invalid_loss_masks": len(invalid_masks),
        }

    primary_train = load_gold_split(cfg, "train")
    primary_val = load_gold_split(cfg, "val")
    if len(primary_train) != 24_107:
        errors.append(
            f"primary train should combine to 24107 rows; found {len(primary_train)}"
        )
    if len(primary_val) != 7_861:
        errors.append(
            "primary validation must remain original-only (7861 rows); "
            f"found {len(primary_val)}"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "schema": "retry-abort-supplement-v2-multisource-audit",
        "test_rows_read": 0,
        "primary_training_rows": len(primary_train),
        "primary_validation_rows": len(primary_val),
        "supplement_validation_rows_available_separately": (
            split_reports["val"]["supplement_rows"]
        ),
        "checkpoint_selection_validation_source": "original_gold_only",
        "splits": split_reports,
        "errors": errors,
        "permitted_next_action": (
            "SUPPLEMENT_FORWARD_BACKWARD_SANITY"
            if not errors
            else "STOP_AND_FIX_REPORTED_LOADER_ERRORS"
        ),
    }
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
