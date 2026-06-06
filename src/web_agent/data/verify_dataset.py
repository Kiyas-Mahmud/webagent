"""Phase 1 dataset verification (IMPLEMENTATION_PLAN 1.1-1.3).

Checks counts, split disjointness, all 24 fields present, label distributions,
and logic consistency (no SUCCESS row with non-NONE failure_type, etc.). Writes a
report to results/dataset_verification.txt — this becomes paper evidence.

Run: python -m web_agent.data.verify_dataset
     python -m web_agent.data.verify_dataset --data ../FinalData --strict

Notes confirmed against the real data:
  - JSON files are UTF-8 (Windows cp1252 default FAILS — always pass encoding).
  - bbox dict keys are x / y / width / height (NOT w / h).
  - The `split` field is "train" in split_train.json; the val/test files each hold
    the three eval sub-splits test_task / test_website / test_domain.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# 24 keys = the 23 schema fields (SPEC 3.3) + borrowed_image flag.
EXPECTED_FIELDS = {
    "task_id", "task_description", "website_domain", "original_task_id", "split",
    "pass", "state_before", "state_after", "visual_diff_score", "action_type",
    "action_target_desc", "action_coordinates", "action_target_bbox",
    "execution_outcome", "failure_type", "failure_confidence", "is_augmented",
    "injection_type", "recovery_strategy", "recovery_success",
    "agent_confidence_before", "reflection_text", "memory_update_flag",
    "borrowed_image",
}

EXPECTED_COUNTS = {"train": 38875, "total": 70965}
EVAL_SUBSPLITS = {"test_task", "test_website", "test_domain"}


def _load(path: Path) -> list[dict]:
    """Load a split JSON as a list of records (UTF-8 mandatory)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Report:
    """Collects pass/fail lines; a single failure flips overall status."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.ok = True

    def check(self, label: str, passed: bool, detail: str = "") -> None:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            self.ok = False
        self.lines.append(f"  [{mark}] {label}" + (f"  ->  {detail}" if detail else ""))

    def info(self, label: str, detail: str = "") -> None:
        self.lines.append(f"  [INFO] {label}" + (f"  ->  {detail}" if detail else ""))

    def section(self, title: str) -> None:
        self.lines.append(f"\n=== {title} ===")

    def text(self) -> str:
        head = "DATASET VERIFICATION REPORT\n" + ("=" * 60)
        tail = "\n" + ("=" * 60) + f"\nOVERALL: {'ALL CHECKS PASSED' if self.ok else 'FAILURES PRESENT'}"
        return head + "\n" + "\n".join(self.lines) + tail


def verify(data_dir: Path) -> Report:
    r = Report()
    tr = _load(data_dir / "split_train.json")
    va = _load(data_dir / "split_val.json")
    te = _load(data_dir / "split_test.json")
    all_rows = tr + va + te

    # --- 1.1 Integrity ---
    r.section("1.1 Integrity")
    r.check("train count == 38,875", len(tr) == EXPECTED_COUNTS["train"], str(len(tr)))
    r.check("total == 70,965", len(all_rows) == EXPECTED_COUNTS["total"], str(len(all_rows)))
    r.info("val / test counts", f"{len(va)} / {len(te)}")

    fields_ok = all(EXPECTED_FIELDS.issubset(row.keys()) for row in all_rows)
    r.check("all 24 fields present in every record", fields_ok)

    paths_ok = all(row["state_before"] and row["state_after"] for row in all_rows)
    r.check("no empty state_before / state_after", paths_ok)

    tr_ids = {row["original_task_id"] for row in tr}
    ev_ids = {row["original_task_id"] for row in va} | {row["original_task_id"] for row in te}
    overlap = tr_ids & ev_ids
    r.check("ZERO original_task_id overlap train vs val/test", len(overlap) == 0, f"{len(overlap)} shared")

    r.check("train split field all 'train'", {row["split"] for row in tr} == {"train"})
    r.check("val/test splits are eval sub-splits only",
            ({row["split"] for row in va} | {row["split"] for row in te}) <= EVAL_SUBSPLITS)

    # --- 1.2 Label distributions ---
    r.section("1.2 Label distributions (train)")
    outcome = Counter(row["execution_outcome"] for row in tr)
    fail_pct = 100 * outcome["FAILURE"] / len(tr)
    r.check("outcome ~72% FAILURE", 70 <= fail_pct <= 74, f"{fail_pct:.1f}% FAILURE")
    r.info("failure_type", str(dict(Counter(row["failure_type"] for row in tr))))
    actions = set(Counter(row["action_type"] for row in tr))
    r.check("action_type only CLICK/TYPE/SELECT", actions <= {"CLICK", "TYPE", "SELECT"}, str(actions))
    r.info("recovery_strategy", str(dict(Counter(row["recovery_strategy"] for row in tr))))

    # confidence gap (whole dataset)
    fc = [row["agent_confidence_before"] for row in all_rows if row["execution_outcome"] == "FAILURE"]
    sc = [row["agent_confidence_before"] for row in all_rows if row["execution_outcome"] == "SUCCESS"]
    f_avg, s_avg = sum(fc) / len(fc), sum(sc) / len(sc)
    r.section("1.2 Confidence calibration gap (all rows)")
    r.check("FAILURE avg ~0.41", 0.36 <= f_avg <= 0.46, f"{f_avg:.3f}")
    r.check("SUCCESS avg ~0.83", 0.78 <= s_avg <= 0.88, f"{s_avg:.3f}")
    r.info("gap", f"{s_avg - f_avg:.3f}")

    # --- 1.3 Logic consistency (all rows) ---
    r.section("1.3 Logic consistency (all rows)")
    succ = [row for row in all_rows if row["execution_outcome"] == "SUCCESS"]
    fail = [row for row in all_rows if row["execution_outcome"] == "FAILURE"]
    r.check("no SUCCESS row has non-NONE failure_type",
            all(row["failure_type"] == "NONE" for row in succ))
    r.check("no SUCCESS row has non-NONE recovery_strategy",
            all(row["recovery_strategy"] == "NONE" for row in succ))
    r.check("no FAILURE row has memory_update_flag == False",
            all(row["memory_update_flag"] for row in fail))

    bbox_present = sum(row["action_target_bbox"] is not None for row in all_rows)
    r.info("bbox present count (for bbox regression filter)", str(bbox_present))
    borrowed = sum(bool(row["borrowed_image"]) for row in all_rows)
    r.info("borrowed_image True count", f"{borrowed} ({100*borrowed/len(all_rows):.1f}%)")

    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dataset dir (default ./data symlink)")
    ap.add_argument("--out", default="results/dataset_verification.txt")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any check fails")
    args = ap.parse_args()

    data_dir = Path(args.data)
    report = verify(data_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.text(), encoding="utf-8")
    print(report.text())
    print(f"\nReport written to {out}")

    if args.strict and not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
