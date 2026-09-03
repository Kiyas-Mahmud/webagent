"""Validate and rank the three full-run validation reports without test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_IDS = (
    "qwen2vl_2b_gold_v2_8_dgx",
    "qwen25vl_7b_gold_v2_8_dgx",
    "internvl35_8b_gold_v2_8_dgx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def selected_metrics(report: dict) -> dict:
    selected_epoch = int(report["selected_epoch"])
    return next(
        row for row in report["history"]
        if int(row["epoch"]) == selected_epoch
    )


def main() -> None:
    args = parse_args()
    rows = []
    git_commits = set()
    for model_id in MODEL_IDS:
        seed_root = (
            args.comparison_root / model_id / f"seed_{args.seed}"
        )
        contract_path = seed_root / "run_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"missing immutable run contract: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract["candidate"]["model_id"] != model_id:
            raise AssertionError(f"{model_id} run contract identifies another model")
        if int(contract["seed"]) != args.seed:
            raise AssertionError(f"{model_id} run contract used another seed")
        if contract["checkpoint_selection"] != "all_gates_then_outcome_mcc":
            raise AssertionError(f"{model_id} run contract changed selection")
        if int(contract.get("test_rows_read", -1)) != 0:
            raise AssertionError(f"{model_id} run contract permits test reads")
        git_commits.add(contract["git_commit"])

        path = seed_root / "full" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"copy the complete {model_id} seed directory here first: {path}"
            )
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise AssertionError(f"{model_id} did not pass registered gates")
        if int(report.get("seed", -1)) != args.seed:
            raise AssertionError(f"{model_id} full report used another seed")
        if int(report.get("requested_epochs", -1)) != 10:
            raise AssertionError(f"{model_id} changed the 10-epoch maximum")
        if int(report.get("test_rows_read", -1)) != 0:
            raise AssertionError(f"{model_id} accessed locked test rows")
        if int(report.get("train_rows", -1)) != 24_107:
            raise AssertionError(f"{model_id} used the wrong training row count")
        if int(report.get("val_rows", -1)) != 7_861:
            raise AssertionError(f"{model_id} used the wrong primary validation count")
        if report.get("selection_rule") != "all_gates_then_outcome_mcc":
            raise AssertionError(f"{model_id} changed checkpoint selection")
        supplement = report["source_validation"]["supplement_retry_abort"]
        if int(supplement.get("rows", -1)) != 194:
            raise AssertionError(f"{model_id} used the wrong supplement validation")
        metrics = selected_metrics(report)
        rows.append({
            "model_id": model_id,
            "seed": args.seed,
            "selected_epoch": int(report["selected_epoch"]),
            "checkpoint": report["best_checkpoint"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
            **{name: float(metrics[name]) for name in (
                "outcome_mcc",
                "failure_macro_f1",
                "failtype_macro_f1",
                "action_macro_f1",
                "needs_recovery_macro_f1",
                "strategy_attempted_macro_f1",
                "recovery_outcome_mcc",
                "memory_mcc",
                "bbox_mean_iou",
                "bbox_recall_iou50",
                "outcome_ece",
            )},
        })

    if len(git_commits) != 1:
        raise AssertionError(
            f"candidate runs used different Git commits: {sorted(git_commits)}"
        )

    # Preregistered comparison: all gates establish eligibility; validation
    # outcome MCC ranks candidates. Remaining keys are deterministic tie-breaks.
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["outcome_mcc"],
            -row["recovery_outcome_mcc"],
            -row["action_macro_f1"],
            row["outcome_ece"],
            row["model_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["validation_rank"] = rank
        row["selected_candidate"] = rank == 1

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    decision = {
        "status": "PASS",
        "selection_scope": "validation_only",
        "selection_rule": (
            "all_quality_gates_then_outcome_mcc; ties: recovery_outcome_mcc, "
            "action_macro_f1, lower_outcome_ece, model_id"
        ),
        "git_commit": next(iter(git_commits)),
        "selected_model_id": ranked[0]["model_id"],
        "models": ranked,
        "test_rows_read": 0,
        "next_action": (
            "Promote the validation-selected seed-42 checkpoint to the frozen final "
            "Table 2 campaign; do not add model seeds 43 or 44."
        ),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
