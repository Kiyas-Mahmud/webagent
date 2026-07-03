"""Scalar baselines for execution_outcome — answers "what does the VLM add?".

If a 1-feature logistic regression on visual_diff_score matches the VLM's MCC, the
big model adds nothing on outcome and the headline claim is weak. This is the
decisive cheap check (and a paper baseline row).

Train on split=='train', evaluate on the 3 generalization sub-splits separately.
FAILURE=1 (positive). Judge by MCC / balanced-accuracy, never raw F1.

Run:  python scripts/baseline_scalars.py --data ../FinalData
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

EVAL_SUBSPLITS = ("test_task", "test_website", "test_domain")
FEATURES = {
    "visual_diff_only": ["visual_diff_score"],
    "confidence_only": ["agent_confidence_before"],
    "both_scalars": ["visual_diff_score", "agent_confidence_before"],
}


def load(path: Path, name: str) -> list[dict]:
    with open(path / name, encoding="utf-8") as f:
        return json.load(f)


def xy(rows: list[dict], feats: list[str]):
    X = np.array([[float(r.get(f, 0.0) or 0.0) for f in feats] for r in rows], dtype=float)
    y = np.array([1 if r["execution_outcome"] == "FAILURE" else 0 for r in rows], dtype=int)
    return X, y


def scores(y_true, y_pred) -> str:
    return (f"MCC={matthews_corrcoef(y_true, y_pred):.4f}  "
            f"bal_acc={balanced_accuracy_score(y_true, y_pred):.4f}  "
            f"macro_f1={f1_score(y_true, y_pred, average='macro'):.4f}  "
            f"failure_f1={f1_score(y_true, y_pred, pos_label=1):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../FinalData")
    args = ap.parse_args()
    root = Path(args.data)

    train = [r for r in load(root, "split_train.json") if r.get("split") == "train"]
    pool = load(root, "split_val.json") + load(root, "split_test.json")
    print(f"train rows: {len(train)}   eval pool: {len(pool)}")

    # majority baseline (predict FAILURE always) — the floor
    print("\n=== majority (always FAILURE) ===")
    for sp in EVAL_SUBSPLITS:
        rows = [r for r in pool if r.get("split") == sp]
        _, yt = xy(rows, ["visual_diff_score"])
        print(f"  {sp:13} {scores(yt, np.ones_like(yt))}")

    for name, feats in FEATURES.items():
        print(f"\n=== logistic regression on {name} {feats} ===")
        Xtr, ytr = xy(train, feats)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(Xtr, ytr)
        for sp in EVAL_SUBSPLITS:
            rows = [r for r in pool if r.get("split") == sp]
            Xe, ye = xy(rows, feats)
            print(f"  {sp:13} {scores(ye, clf.predict(Xe))}")


if __name__ == "__main__":
    main()
