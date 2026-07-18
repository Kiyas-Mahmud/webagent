"""STEP D — evaluate on the 3 test splits separately (spec).

test_task / test_website / test_domain are the generalization splits living inside
split_val.json + split_test.json (the `split` field). We evaluate each separately
and write all metrics per split to a CSV. Test rows are untouched until here.
"""

from __future__ import annotations

import csv
from pathlib import Path

from web_agent.data.dataloader import build_dataloader, load_split
from web_agent.train.trainer import collect_predictions, compute_metrics

EVAL_SUBSPLITS = ("test_task", "test_website", "test_domain")


def _subset(records, split_name):
    return [r for r in records if r.get("split") == split_name]


def evaluate_all_splits(model, cfg, processor, tokenizer=None, device="cuda",
                        out_csv="results/qwen2vl2b_test.csv"):
    """Run eval on each generalization split; return {split: metrics} and write CSV."""
    val = load_split(cfg, "val")
    test = load_split(cfg, "test")
    pool = val + test

    results: dict[str, dict] = {}
    for split_name in EVAL_SUBSPLITS:
        rows = _subset(pool, split_name)
        if not rows:
            continue
        loader = build_dataloader(cfg, "eval_labels", rows, processor, tokenizer,
                                  shuffle=False, num_workers=cfg["data"].get("num_workers", 2))
        preds = collect_predictions(model, loader, device)
        results[split_name] = compute_metrics(preds)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if results:
        keys = list(next(iter(results.values())).keys())
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split"] + keys)
            for split_name, m in results.items():
                w.writerow([split_name] + [f"{m[k]:.5f}" for k in keys])
    return results
