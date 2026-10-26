"""Score one frozen, validation-selected assessor checkpoint on the locked Web-Gold-40K test split.

This is the final step of the registered protocol: train on train, select on validation,
then open the locked test once with the frozen checkpoint. No parameter is updated here.

Before the test is opened, the same checkpoint is re-scored on the 7,861 primary validation
rows and must reproduce the outcome MCC recorded in its run report. This proves the
checkpoint, config and data were loaded exactly as during selection.

Example (lab DGX):
    python scripts/evaluate_locked_test.py \
        --model-id qwen25vl_7b_gold_v2_8_dgx \
        --checkpoint /path/best_e0_outcome-mcc0.678.ckpt \
        --report /path/full/report.json \
        --data-root /path/final_data_set_40k \
        --supplement-root /path/retry_abort_supplement_v2 \
        --out results/locked_test/qwen25vl_7b_gold_v2_8_dgx.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CONFIGS = {
    "qwen2vl_2b_gold_v2_8_dgx": "configs/backbones/qwen2vl_2b_gold_v2_8_dgx.yaml",
    "qwen25vl_7b_gold_v2_8_dgx": "configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml",
    "internvl35_8b_gold_v2_8_dgx": "configs/backbones/internvl35_8b_gold_v2_8_dgx.yaml",
}
EXPECTED_TEST_ROWS = 7_855
EXPECTED_VAL_ROWS = 7_861
REPORT_METRICS = (
    "outcome_mcc", "failure_macro_f1", "failtype_macro_f1", "action_macro_f1",
    "recovery_macro_f1", "recovery_outcome_mcc", "recovery_outcome_macro_f1",
    "memory_mcc", "memory_macro_f1", "outcome_ece",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--report", type=Path, help="full-run report.json holding the validation best_metric")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--supplement-root", type=Path, help="as used in training; rebuilds identical loss weights")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-tolerance", type=float, default=0.005)
    ap.add_argument("--skip-validation-check", action="store_true",
                    help="not recommended: open the test without reproducing the validation score")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    import torch
    from peft import set_peft_model_state_dict
    from web_agent.config import load_config
    from web_agent.data.gold_dataloader import build_gold_dataloader, load_gold_split
    from web_agent.train.gold_stages import build_gold_components
    from web_agent.train.trainer import build_diagnostics, collect_predictions, compute_metrics
    from web_agent.utils.seed import set_seed

    assert args.checkpoint.is_file(), args.checkpoint
    assert args.checkpoint.stat().st_size > 1_000_000, "checkpoint looks like a Git LFS pointer; run git lfs pull"
    expected_val = None
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        assert Path(report["best_checkpoint"]).name == args.checkpoint.name, (
            "this checkpoint is not the validation-selected one in the report")
        assert report.get("test_rows_read", 0) == 0
        expected_val = float(report["best_metric"])

    cfg = load_config(str(repo / CONFIGS[args.model_id]))
    cfg["data"]["root"] = str(args.data_root)
    cfg["data"]["num_workers"] = args.num_workers
    if args.supplement_root:
        cfg["data"].setdefault("recovery_supplement", {}).update({
            "enabled": True, "root": str(args.supplement_root),
            "include_in_primary_validation": False,
        })
    set_seed(args.seed)

    train_records = load_gold_split(cfg, "train")
    comp = build_gold_components(cfg, None, train_records=train_records, trajectory_records=train_records)
    saved = torch.load(args.checkpoint, map_location=comp.device, weights_only=False)
    set_peft_model_state_dict(comp.model.encoder.model, saved["lora"])
    comp.model.adapter.load_state_dict(saved["adapter"])
    if "task_adapters" in saved:
        comp.model.task_adapters.load_state_dict(saved["task_adapters"])
    for head in ("failure", "action", "memory", "recovery_outcome"):
        getattr(comp.model, f"{head}_head" if head != "failure" else "failure_head").load_state_dict(saved[head])

    def score(which: str, expected_rows: int) -> tuple[dict, dict, int]:
        records = load_gold_split(cfg, which)
        assert len(records) == expected_rows, f"{which}: {len(records)} rows, expected {expected_rows}"
        loader = build_gold_dataloader(cfg, which, comp.processor, records=records, shuffle=False,
                                       num_workers=args.num_workers, seed=args.seed,
                                       trajectory_records=records)
        with torch.no_grad():
            pred = collect_predictions(comp.model, loader, comp.device)
        return compute_metrics(pred), build_diagnostics(pred), len(records)

    validation_check = {"skipped": True}
    if not args.skip_validation_check:
        val_metrics, _, n_val = score("val", EXPECTED_VAL_ROWS)
        got = float(val_metrics["outcome_mcc"])
        validation_check = {"skipped": False, "rows": n_val, "outcome_mcc": got,
                            "expected_outcome_mcc": expected_val,
                            "metrics": {k: val_metrics.get(k) for k in REPORT_METRICS}}
        if expected_val is not None:
            diff = abs(got - expected_val)
            validation_check["abs_difference"] = diff
            assert diff <= args.val_tolerance, (
                f"validation outcome MCC {got:.4f} does not reproduce {expected_val:.4f}; "
                "the locked test was NOT opened. Check config, data root and checkpoint.")
        print(f"validation reproduced: outcome MCC {got:.4f}", flush=True)

    test_metrics, test_diag, n_test = score("test", EXPECTED_TEST_ROWS)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    out = {
        "status": "PASS",
        "model_id": args.model_id,
        "config": CONFIGS[args.model_id],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "code_commit": commit,
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "locked_test_opened": True,
        "test_rows": n_test,
        "validation_check": validation_check,
        "test_metrics": {k: test_metrics.get(k) for k in REPORT_METRICS},
        "test_metrics_all": test_metrics,
        "test_diagnostics": test_diag,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"model_id": args.model_id, "test_rows": n_test,
                      **{k: round(float(v), 4) for k, v in out["test_metrics"].items() if v is not None}},
                     indent=2))
    print("written", args.out)


if __name__ == "__main__":
    main()
