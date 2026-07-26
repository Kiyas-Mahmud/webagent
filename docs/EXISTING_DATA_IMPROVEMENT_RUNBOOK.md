# Existing-data improvement runbook

## Purpose

This workflow improves Web-Gold-40K without replacing the 39,215-row dataset,
editing the raw split files, extracting the large Kaggle ZIP, or reading the
locked test split.

It addresses two evidence-backed issues:

1. 2,534 train/validation bbox labels fail native-image geometry checks.
2. The v2.7 mini is weak on `SCROLL`, `SELECT`, `NAVIGATE`,
   `LOOP_DETECTED`, and `BACKTRACK`, while `RETRY` and `ABORT` have no
   train/validation support.

The workflow first audits existing records. New collection is permitted only
for a class that remains genuinely absent and is part of the registered thesis
claim.

## Files

- Notebook: `notebooks/kaggle_gold_existing_data_improvement.ipynb`
- Kaggle script: `scripts/audit_gold_existing_data.py`
- Pure audit logic: `src/web_agent/data/improvement_audit.py`
- Tests: `tests/test_improvement_audit.py`
- General review rules: `docs/DATASET_MANUAL_REVIEW_GUIDE.md`

The main `notebooks/kaggle_gold.ipynb` is not changed or called.

## Run on Kaggle

1. Create a new Kaggle notebook from
   `notebooks/kaggle_gold_existing_data_improvement.ipynb`.
2. Use **Add Input** and attach `kiyasmahmud/web-gold-40k`.
3. Use CPU. This audit does not train a model.
4. Run all cells.
5. Download
   `/kaggle/working/gold_existing_data_improvement_review_package.zip`.

The script follows Kaggle version symlinks and streams a compatible nested ZIP
in place. It never extracts the source images.

## Non-destructive bbox decision

Every non-null bbox is checked against the actual `state_before` dimensions:

```text
finite x, y, width, height
x >= 0
y >= 0
width > 0
height > 0
x + width <= image width
y + height <= image height
```

The split schema has no document-to-viewport scroll-offset metadata. Therefore
there is no defensible automatic conversion for the invalid rows. The workflow:

- writes every invalid row to `bbox_review_queue.csv`;
- writes the same IDs to `bbox_mask_manifest.json` with `bbox_mask = 0`;
- keeps the row for every non-localization head;
- leaves proposed correction columns empty;
- makes no source-data mutation.

Do not clamp an overflowing box or center it on `action_coordinates`. Both
operations invent target geometry. A replacement bbox is allowed only when the
replay or source screenshot proves the true native-image target and the second
reviewer confirms it.

## Weak-class review queue

The queue is deterministic and trajectory-preserving:

- up to 100 tasks per target class are selected from train and validation;
- every step belonging to a selected task is included, not only the target row;
- the same task is never split between reviewers;
- 10% of tasks are assigned to `A+B` for agreement measurement;
- attempted-recovery rows include the causal transition:
  failure state, executed recovery action/value, post-recovery state, and
  observed recovery outcome.

The target classes are:

```text
ACTION:SCROLL
ACTION:SELECT
ACTION:NAVIGATE
FAILURE:LOOP_DETECTED
RECOVERY:BACKTRACK
RECOVERY:RETRY
RECOVERY:ABORT
```

The review is a label-quality audit, not permission to rebalance by changing
correct labels. Reviewers must mark ambiguous evidence for discussion.

## Division of work

### Codex/data lead

1. Generate and verify the audit package.
2. Explain queue columns and label rules.
3. Merge immutable reviewer logs without deleting disagreements.
4. Validate proposed corrections against evidence.
5. Apply only approved corrections to the source/export ledger.
6. Regenerate train/validation artifacts and rerun automated gates.
7. Run one controlled 5k experiment using the same validation split and
   selection gates.

### Reviewer A and Reviewer B

1. Review the complete assigned trajectories.
2. Check before/after images, action, target, outcome, and failure type.
3. For recovery, verify all four causal pieces recorded in the queue.
4. Fill correction values only with a reproducible evidence reference.
5. Independently review every `A+B`, correction, rejection, and ambiguous row.
6. Never edit the split JSON files directly.

## Applying reviewer decisions

The audit package itself does not modify data. After review:

1. Preserve both original reviewer decisions.
2. Adjudicate all disagreements.
3. Create one approved correction ledger keyed by `sample_id`.
4. Null only an unresolvable bbox; do not discard the whole row.
5. Correct an action/failure/recovery label only when replay evidence proves
   the old label wrong.
6. Regenerate the dataset from its source ledger.
7. Preserve task/domain split boundaries.
8. Version and checksum the new dataset.
9. Rerun `notebooks/kaggle_gold_data_validation.ipynb`.

## When new data is actually required

`SCROLL`, `SELECT`, `NAVIGATE`, `LOOP_DETECTED`, and `BACKTRACK` already have
existing examples, so audit them before collecting anything.

In the current dataset, `RETRY` and `ABORT` have zero train/validation support.
Choose one registered scope:

- If the thesis claims a learned six-strategy classifier, collect real,
  complete, independently reviewed `RETRY` and `ABORT` transitions in both
  training and validation.
- If `ABORT` remains only a production decision-combiner rule, explicitly
  exclude it from trainable-strategy performance claims and do not manufacture
  examples.

A collected recovery trajectory must contain:

```text
failure state
-> executed recovery action and value
-> post-recovery state
-> observed recovery_success true/false
```

Do not duplicate rows, relabel other strategies, or create synthetic examples
to fill a class.

## Gate before another mini run

Another controlled mini experiment is permitted only after:

- every proposed correction has two-person evidence;
- the bbox mask/correction totals reconcile with the audit report;
- weak-class reviewer disagreements are resolved;
- class counts are regenerated;
- train/validation task and domain separation remains valid;
- test rows read remains zero.

The experiment must compare against v2.7 with the same seed, 5k/500 subsets,
validation-only selection, and eight quality gates. Improvement is judged with
per-class F1/MCC/balanced accuracy and bbox IoU, not headline accuracy alone.
