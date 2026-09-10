"""Run the recovery v2.9 full training stage.

This is a NEW entry point.  ``scripts/run_gold.py`` is unchanged and remains the
v2.8 path; nothing here can affect a v2.8 run in progress.

The controlled 5k mini stage is deliberately not offered.  At the throughput
measured on the GB10 (~14 s per physical batch) the mini costs ~24 h and
validates on 500 rows, while epoch 0 of the full run costs ~23 h and validates
on all 7,861 original-Gold rows.  Epoch 0 is therefore the gate, and its abort
thresholds are registered up front in docs/RECOVERY_V2_9_EXPERIMENT.md.  Use
``--check-epoch0`` after the first epoch lands to evaluate them.

The 16-row compatibility smoke still runs through ``scripts/run_gold.py
--stage smoke`` and should be run first; it is fail-closed and costs minutes.

The locked test split is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_agent.config import load_config  # noqa: E402
from web_agent.train.gold_full_v2_9 import run_gold_full_v2_9  # noqa: E402

CONFIG = "configs/backbones/qwen25vl_7b_gold_v2_9.yaml"

# Registered epoch-0 abort gate. These replace the skipped mini stage and are
# fixed BEFORE the run so they cannot be tuned to the result.
#
# The two bbox floors are v2.8's own 7B epoch-0 values. Because the v2.9 centre
# offset is zero-initialised, v2.9 starts numerically identical to v2.8 -- so a
# regression below these means the head swap is wired wrong, not that the idea
# failed. Kill and fix rather than training through it.
EPOCH0_GATE = {
    "outcome_mcc": 0.55,        # v2.8 epoch 0: 0.678 (7B), 0.590 (2B)
    "bbox_mean_iou": 0.155,     # v2.8 7B epoch 0
    "bbox_recall_iou50": 0.083,  # v2.8 7B epoch 0
}


def check_epoch0(metrics_csv: Path) -> int:
    """Evaluate the registered epoch-0 abort gate against the metrics CSV."""
    import csv

    with metrics_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"FAIL: no epoch rows yet in {metrics_csv}")
        return 1
    row = rows[0]
    failures = []

    loss = float(row["train_loss"])
    ok = loss >= 0.0
    print(f"  {'OK  ' if ok else 'FAIL'} train_loss >= 0            : {loss:+.4f}")
    if not ok:
        failures.append("train_loss")

    for key, floor in EPOCH0_GATE.items():
        value = float(row[key])
        ok = value >= floor
        print(f"  {'OK  ' if ok else 'FAIL'} {key:<24s}: {value:.4f}  (floor {floor})")
        if not ok:
            failures.append(key)

    print()
    if failures:
        print(f"EPOCH-0 GATE: FAIL ({', '.join(failures)}) -- stop the run and diagnose")
        return 1
    print("EPOCH-0 GATE: PASS -- continue training")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--supplement-root", required=True)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--report-json")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--check-epoch0",
        metavar="METRICS_CSV",
        help="evaluate the registered epoch-0 abort gate and exit",
    )
    args = parser.parse_args()

    if args.check_epoch0:
        return check_epoch0(Path(args.check_epoch0))

    for required in ("checkpoint_root", "metrics_csv", "report_json"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required for a training run")

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.data_root
    cfg["data"]["num_workers"] = args.num_workers
    cfg["data"].setdefault("recovery_supplement", {}).update({
        "enabled": True,
        "root": args.supplement_root,
        "include_in_primary_validation": False,
    })

    result = run_gold_full_v2_9(
        cfg,
        epochs=args.epochs,
        seed=args.seed,
        checkpoint_root=args.checkpoint_root,
        metrics_csv=args.metrics_csv,
        resume_checkpoint=args.resume_checkpoint,
    )
    report = Path(args.report_json)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {report}")
    print("\nNow evaluate the registered epoch-0 gate:")
    print(f"  python scripts/run_gold_v2_9.py --check-epoch0 {args.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
