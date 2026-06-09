# Phase 1 — Dataset Verification

## 1. Learning topics

- **Data validation** — checking data is what you think before you train on it.
- **Assertions** — statements that must be true, or the program stops.
- **Distributions** — how often each label value appears (class balance).
- **Logic consistency** — cross-field rules (e.g. a SUCCESS row must have no failure type).
- **UTF-8 encoding** — why text files must be read with the right encoding.
- **Set operations** — using `set` intersection to detect overlap fast.

## 2. Why this phase exists

The dataset is the project's strongest asset. If even 1% of labels are wrong or
the train/test split leaks, every model number is meaningless. Phase 1 *proves*
the data is correct and writes a report that becomes evidence in the paper.

A famous rule: **garbage in, garbage out.** Verify first, trust later.

## 3. Code structure

```
src/web_agent/data/verify_dataset.py
  EXPECTED_FIELDS   the 24 columns every record must have
  _load(path)       read one split JSON (UTF-8)
  class Report      collects PASS/FAIL lines, tracks overall status
  verify(data_dir)  runs ALL checks, returns a Report
  main()            CLI entry: writes results/dataset_verification.txt
```

It depends on nothing heavy (no torch) — pure Python (`json`, `collections`,
`pathlib`). That is deliberate: verification must run anywhere, instantly.

## 4. How it works (algorithm)

```
verify(data_dir):
  load split_train.json, split_val.json, split_test.json   (UTF-8!)
  all_rows = train + val + test

  --- 1.1 Integrity ---
  check len(train) == 38,875
  check len(all_rows) == 70,965
  check every record has all 24 fields
  check no empty image paths
  check train IDs and val/test IDs do NOT overlap   (set intersection == empty)
  check train rows all have split=="train"
  check val/test rows are only test_task/website/domain

  --- 1.2 Distributions ---
  count outcomes -> FAILURE should be ~72%
  check action_type only CLICK/TYPE/SELECT (SCROLL/NAVIGATE absent by design)
  average confidence: FAILURE rows ~0.41, SUCCESS rows ~0.83

  --- 1.3 Logic consistency ---
  every SUCCESS row -> failure_type == "NONE"
  every SUCCESS row -> recovery_strategy == "NONE"
  every FAILURE row -> memory_update_flag == True
  count bbox-present rows (67,160) and borrowed images (10,190)

  return Report (PASS/FAIL per check + overall)
```

### The two techniques worth copying

**1. Set intersection for leakage detection.** To check no task appears in both
train and test, we use Python `set`s:
```python
tr_ids = {r["original_task_id"] for r in train}
ev_ids = {r["original_task_id"] for r in val} | {r["original_task_id"] for r in test}
overlap = tr_ids & ev_ids          # & is set intersection
assert len(overlap) == 0
```
Sets make this O(n) instead of O(n²) — instant even on 70k rows.

**2. A `Report` accumulator instead of crashing.** Rather than `assert` (which
stops at the first failure), `Report.check()` records every result and flips a
flag. You see ALL problems in one run, not one at a time.
```python
def check(self, label, passed, detail=""):
    if not passed: self.ok = False
    self.lines.append(f"[{'PASS' if passed else 'FAIL'}] {label} -> {detail}")
```

### The UTF-8 detail
The JSON files contain non-ASCII characters. On Windows the default reader
(cp1252) crashes. Always:
```python
open(path, encoding="utf-8")
```

## 5. Key functions

| Function | In → Out |
|----------|----------|
| `_load(path)` | json path → list of record dicts (UTF-8) |
| `verify(data_dir)` | dataset folder → `Report` object |
| `Report.check(label, passed, detail)` | a result → appended line, updates `.ok` |
| `Report.text()` | → the full formatted report string |
| `main()` | CLI flags → writes report file, exits non-zero if `--strict` and failures |

### What "correct output" looked like (real Kaggle run)
```
[PASS] train count == 38,875
[PASS] ZERO original_task_id overlap        -> 0 shared
[PASS] outcome ~72% FAILURE                 -> 71.9% FAILURE
[PASS] FAILURE avg ~0.41 / SUCCESS avg ~0.83 -> 0.402 / 0.825
[INFO] bbox present count                   -> 67,160
OVERALL: ALL CHECKS PASSED
```
Every number matched the spec → the dataset is trustworthy.
