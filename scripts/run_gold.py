"""Gold-dataset runner (separate from the synthetic notebook/pipeline).

    python scripts/run_gold.py --stage smoke
    python scripts/run_gold.py --stage mini --data-root /kaggle/input/.../gold
    python scripts/run_gold.py --stage reeval --checkpoint /kaggle/input/.../best.ckpt
    python scripts/run_gold.py --stage eval  --checkpoint checkpoints/Y_QWEN2VL_2B_GOLD/best_*.ckpt

Smoke and mini never read the test split. The causal gold model uses state_before
for action/bbox/confidence-before and before+after for failure/recovery/memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.config import load_config
from web_agent.utils.results import (
    save_mini_diagnostics_json,
    save_mini_result_csv,
    save_source_validation_csv,
)
from web_agent.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backbones/qwen2vl_2b_gold.yaml")
    ap.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "mini", "reeval", "eval"),
    )
    ap.add_argument("--data-root", default=None, help="override cfg.data.root (Kaggle path)")
    ap.add_argument(
        "--supplement-root",
        default=None,
        help=(
            "extracted RETRY/ABORT supplement v2 root; appends only recovery-"
            "supervised train rows and keeps primary validation original-only"
        ),
    )
    ap.add_argument(
        "--review-overlay-dir",
        default=None,
        help=(
            "passed reconciliation directory; applies in memory to train/val "
            "only"
        ),
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="override backbone.min_pixels for a controlled runtime profile",
    )
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="override backbone.max_pixels for a controlled runtime profile",
    )
    ap.add_argument("--train-rows", type=int, default=5_000)
    ap.add_argument("--val-rows", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--result-csv",
        default="/kaggle/working/gold_mini_result.csv",
        help="one-row-per-epoch mini result table",
    )
    ap.add_argument(
        "--diagnostics-json",
        default="/kaggle/working/gold_mini_diagnostics.json",
        help="per-class mini diagnostics, distributions, weights, and controls",
    )
    ap.add_argument(
        "--report-json",
        default="/kaggle/working/gold_mini_report.json",
        help="complete machine-readable mini report",
    )
    ap.add_argument(
        "--source-validation-csv",
        default="/kaggle/working/gold_mini_source_validation.csv",
        help="original and supplement validation metrics as separate rows",
    )
    ap.add_argument(
        "--reeval-json",
        default="/kaggle/working/gold_v14_reeval.json",
        help="validation-only report for a supplied historical checkpoint",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if (args.min_pixels is None) != (args.max_pixels is None):
        ap.error("--min-pixels and --max-pixels must be supplied together")
    if args.min_pixels is not None:
        if args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
            ap.error("pixel bounds must satisfy 0 < min_pixels <= max_pixels")
        cfg["backbone"]["min_pixels"] = args.min_pixels
        cfg["backbone"]["max_pixels"] = args.max_pixels
    if args.supplement_root:
        if args.stage in {"reeval", "eval"}:
            ap.error("--supplement-root is training/smoke-only")
        supplement = cfg["data"].setdefault("recovery_supplement", {})
        supplement.update({
            "enabled": True,
            "root": args.supplement_root,
            "include_in_primary_validation": False,
        })
    if args.review_overlay_dir:
        if args.stage == "eval":
            ap.error(
                "--review-overlay-dir is development-only and cannot be used "
                "for locked-test eval"
            )
        cfg["data"]["review_overlay_dir"] = args.review_overlay_dir
    seed = cfg.get("seeds", [42])[0]
    set_seed(seed)

    from web_agent.train.gold_stages import (
        reevaluate_gold_validation_checkpoint,
        run_gold_mini,
        run_gold_smoke,
    )

    if args.stage == "smoke":
        print(json.dumps(run_gold_smoke(cfg, rows=16, seed=seed), indent=2))
        return

    if args.stage == "mini":
        report = run_gold_mini(
            cfg,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            epochs=args.epochs,
            seed=seed,
        )
        csv_path = save_mini_result_csv(report, args.result_csv)
        diagnostics_path = save_mini_diagnostics_json(report, args.diagnostics_json)
        source_validation_path = save_source_validation_csv(
            report,
            args.source_validation_csv,
        )
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        print("CSV saved:", csv_path)
        print("Diagnostics saved:", diagnostics_path)
        print("Source validation CSV saved:", source_validation_path)
        print("Full report saved:", report_path)
        return

    if args.stage == "reeval":
        if not args.checkpoint:
            ap.error("--checkpoint is required for validation re-evaluation")
        report = reevaluate_gold_validation_checkpoint(
            cfg,
            args.checkpoint,
            val_rows=args.val_rows,
            seed=seed,
        )
        report_path = Path(args.reeval_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("Re-evaluation saved:", report_path)
        return

    if not args.checkpoint:
        ap.error("--checkpoint is required for the final eval stage")
    from web_agent.data.gold_dataloader import build_gold_dataloader
    from web_agent.train.gold_stages import build_gold_components
    from web_agent.train.trainer import collect_predictions, compute_metrics

    components = build_gold_components(cfg)
    _load(components.model, args.checkpoint)
    test_loader = build_gold_dataloader(
        cfg, "test", components.processor, shuffle=False, num_workers=4, seed=seed,
    )
    predictions = collect_predictions(components.model, test_loader, components.device)
    metrics = compute_metrics(predictions)
    print("GOLD TEST:", {key: round(value, 4) for key, value in metrics.items()})


def _load(model, path: str) -> None:
    import torch
    from peft import set_peft_model_state_dict
    ck = torch.load(path, map_location="cuda")
    set_peft_model_state_dict(model.encoder.model, ck["lora"])
    model.adapter.load_state_dict(ck["adapter"])
    if "task_adapters" in ck:
        model.task_adapters.load_state_dict(ck["task_adapters"])
    model.failure_head.load_state_dict(ck["failure"])
    model.action_head.load_state_dict(ck["action"])
    model.memory_head.load_state_dict(ck["memory"])
    model.recovery_outcome_head.load_state_dict(ck["recovery_outcome"])
    print("loaded checkpoint:", path)


if __name__ == "__main__":
    main()
