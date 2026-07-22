"""Resume the interrupted v2.7 mini run from its completed epoch-3 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from web_agent.config import load_config  # noqa: E402
from web_agent.train.gold_resume import resume_gold_mini  # noqa: E402
from web_agent.utils.results import (  # noqa: E402
    save_mini_diagnostics_json,
    save_mini_result_csv,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/backbones/qwen2vl_2b_gold_v2_7.yaml"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prior-metrics-csv", required=True)
    parser.add_argument("--prior-checkpoint-dir", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--expected-resume-sha256", required=True)
    parser.add_argument("--selected-checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--diagnostics-json", required=True)
    parser.add_argument("--resumed-metrics-csv", required=True)
    parser.add_argument("--checkpoint-output-root", required=True)
    parser.add_argument("--train-rows", type=int, default=5_000)
    parser.add_argument("--val-rows", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume-epoch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=200_704)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    cfg["data"]["root"] = args.data_root
    cfg["data"]["num_workers"] = args.num_workers
    cfg["backbone"]["min_pixels"] = args.min_pixels
    cfg["backbone"]["max_pixels"] = args.max_pixels
    report = resume_gold_mini(
        cfg,
        prior_metrics_csv=args.prior_metrics_csv,
        prior_checkpoint_dir=args.prior_checkpoint_dir,
        resume_checkpoint=args.resume_checkpoint,
        expected_resume_sha256=args.expected_resume_sha256,
        selected_checkpoint_output=args.selected_checkpoint,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        total_epochs=args.epochs,
        resume_epoch=args.resume_epoch,
        seed=args.seed,
        resumed_metrics_csv=args.resumed_metrics_csv,
        checkpoint_output_root=args.checkpoint_output_root,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = save_mini_result_csv(report, args.csv)
    diagnostics_path = save_mini_diagnostics_json(report, args.diagnostics_json)

    summary = {
        "status": report["status"],
        "completed_epochs": report["resume_provenance"]["completed_epochs"],
        "resumed_epochs": report["resume_provenance"]["resumed_epochs"],
        "eligible_epochs": report["quality_gates"]["eligible_epochs"],
        "selected_epoch": report["selected_epoch"],
        "selected_outcome_mcc": report["best_metric"],
        "checkpoint_roundtrip": report["checkpoint_roundtrip"],
        "rng_state_restored": report["resume_provenance"]["rng_state_restored"],
        "test_rows_read": report["test_rows_read"],
        "report": str(report_path),
        "csv": str(csv_path),
        "diagnostics": str(diagnostics_path),
        "selected_checkpoint": report["best_checkpoint"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
