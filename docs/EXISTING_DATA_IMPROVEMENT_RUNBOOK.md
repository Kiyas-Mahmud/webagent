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
- Interactive reviewer notebook: `notebooks/kaggle_gold_manual_review.ipynb`
- Kaggle script: `scripts/audit_gold_existing_data.py`
- Review reconciliation script: `scripts/reconcile_gold_reviews.py`
- Overlay validation notebook:
  `notebooks/kaggle_gold_review_overlay_validation.ipynb`
- Overlay validator: `scripts/validate_gold_review_overlay.py`
- Pure audit logic: `src/web_agent/data/improvement_audit.py`
- Immutable event validation: `src/web_agent/data/review_session.py`
- Two-person reconciliation: `src/web_agent/data/review_reconciliation.py`
- Tests: `tests/test_improvement_audit.py`, `tests/test_review_session.py`
- General review rules: `docs/DATASET_MANUAL_REVIEW_GUIDE.md`
- Verified Kaggle result: `docs/EXISTING_DATA_IMPROVEMENT_AUDIT_RESULT.md`

The main `notebooks/kaggle_gold.ipynb` is not changed or called.

Workflow order:

1. Generate the audit package using **Generate the audit package on Kaggle
   (first)** below.
2. Run the two-person manual review.
3. Reconcile primary and required secondary reviews until `PASS`.
4. Validate the passed overlay on the real mounted data.
5. Re-evaluate v2.7 and run the controlled mini on that identical overlay.

## Run the two-person manual review

Use `notebooks/kaggle_gold_manual_review.ipynb` in an interactive Kaggle
**Edit** session. Do not commit it as a batch run because it intentionally
waits for human decisions. Start only after the audit package has been
generated.

Each reviewer must:

1. Attach `kiyasmahmud/web-gold-40k`.
2. Attach the accepted output of
   `kiyasmahmud/web-gold-existing-data-improvement`.
3. Use CPU.
4. Set `REVIEWER_ID` to their own identity, set `QUEUE_KIND`, read the guide,
   and set `REVIEWER_CONFIRMED_GUIDE = True`.
5. Finish `bbox`, export its queue-specific event CSV and summary, then repeat
   for `weak`.
6. Download the event CSV frequently. To resume, attach exactly one previously
   exported queue-specific event CSV; cell 3 restores and validates it.

The notebook:

- reads only train and validation;
- shows the complete trajectory around every decision target;
- shows failure state, executed action, and post-recovery state for recovery;
- validates replacement boxes against native image dimensions;
- requires evidence for every proposed correction;
- appends revisions instead of overwriting earlier decisions;
- keeps Reviewer A and Reviewer B in separate logs;
- reads zero test rows and never edits source JSON or screenshots.

The four primary event logs are:

```text
reviewer_A_bbox_review_events.csv
reviewer_A_weak_review_events.csv
reviewer_B_bbox_review_events.csv
reviewer_B_weak_review_events.csv
```

For the accepted version-3 queues, the notebook should report these assigned
decision-target counts (the shared `A+B` overlap appears once for each person):

| Reviewer | BBox targets | Weak-class targets |
| --- | ---: | ---: |
| A | 1,406 | 884 |
| B | 1,371 | 970 |

Stop and report the mismatch if the notebook shows different counts while
using the accepted audit output.

## Reconcile the reviewer logs

After both reviewers finish their primary queues, run:

```bash
python scripts/reconcile_gold_reviews.py \
  --bbox-queue bbox_review_queue.csv \
  --weak-queue weak_class_review_queue.csv \
  --reviewer-log reviewer_A_bbox_review_events.csv \
  --reviewer-log reviewer_A_weak_review_events.csv \
  --reviewer-log reviewer_B_bbox_review_events.csv \
  --reviewer-log reviewer_B_weak_review_events.csv \
  --output-dir gold_review_reconciliation
```

This command is lightweight: it reads CSV logs and queues, not the large image
dataset. It preserves all immutable events and produces:

- missing primary reviews;
- agreement pairs, raw agreement, and Cohen's kappa;
- disagreements and unresolved cases;
- approved two-person corrections;
- finalized dispositions;
- secondary-review queues;
- a checksummed reconciliation report and ZIP.

`A+B` targets always require both reviewers. In addition, any
`approve_after_correction`, `needs_discussion`, `reject_recollect`, or
`quarantine_policy` decision from a singly assigned target automatically
creates a task for the other reviewer. Attach the corresponding
`bbox_secondary_review_queue.csv` or `weak_secondary_review_queue.csv` to the
manual-review notebook, finish those targets, and rerun reconciliation with
the updated logs.

Use `--require-pass` only for the final reconciliation. A `PASS` means the
targeted improvement queues are complete and a controlled mini may proceed. It
does **not** mean the full 39,215-row dataset is publication-ready.

## Validate and use the passed review overlay

Do not create or upload a second full image dataset for the controlled mini.
The training loader supports an opt-in, evidence-checked in-memory overlay.

1. Open `notebooks/kaggle_gold_review_overlay_validation.ipynb`.
2. Attach the existing `kiyasmahmud/web-gold-40k` dataset.
3. Attach exactly one reconciliation output whose report is `PASS`.
4. Use CPU and run all cells.
5. Preserve `gold_review_overlay_validation.json`.

The validator:

- verifies SHA-256 values for the correction and disposition CSVs;
- requires `controlled_mini_permitted=true`, `test_rows_read=0`, and
  `publication_ready=false` in the reconciliation report;
- checks every corrected row's split, sample, task, step, and original value
  against the mounted source version;
- checks corrected bbox geometry against the real native image;
- applies approved corrections in memory;
- excludes independently confirmed `reject_recollect` and
  `quarantine_policy` rows;
- leaves `review_status` unchanged and does not write split JSON or images;
- reads train/validation only.

For the next controlled mini, attach the same reconciliation output and set:

```python
cfg["data"]["review_overlay_dir"] = str(RECONCILIATION_DIR)
```

`load_gold_split` applies it only for `train` and `val`; test remains untouched.
Any changed source value, tampered ledger, missing target, conflicting
correction, invalid recovery tuple, or invalid bbox fails before training.

Because corrected or excluded validation rows change the evaluation data, do
not compare the new mini directly with the historical v2.7 numbers. First
re-evaluate the selected v2.7 checkpoint on the same passed overlay:

```bash
python scripts/run_gold.py \
  --config configs/backbones/qwen2vl_2b_gold_v2_7.yaml \
  --stage reeval \
  --data-root /kaggle/input/datasets/kiyasmahmud/web-gold-40k \
  --review-overlay-dir /kaggle/input/PASSED_RECONCILIATION \
  --checkpoint /kaggle/input/PRIOR_V2_7/best_e4_outcome-mcc0.548.ckpt \
  --reeval-json /kaggle/working/v2_7_reviewed_overlay_reeval.json
```

Then train the new controlled mini on that identical overlay. The mini report
records the reconciliation SHA-256 and explicitly marks historical v14/v2.7
validation metrics as non-comparable. Improvement claims must use the
re-evaluated v2.7 checkpoint as the baseline. The runner rejects
`--review-overlay-dir` for locked-test `eval`.

This overlay is experiment evidence, not the final publication dataset. After
the controlled mini confirms the effect, approved source-level changes must
still be applied to the versioned collection/export ledger, splits regenerated
when required, and the complete publication validator rerun.

## Generate the audit package on Kaggle (first)

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

- review reconciliation passes with `--require-pass`;
- the real mounted-data overlay validation passes;
- every proposed correction has two-person evidence;
- the bbox mask/correction totals reconcile with the audit report;
- weak-class reviewer disagreements are resolved;
- class counts are regenerated;
- train/validation task and domain separation remains valid;
- test rows read remains zero.

The experiment must compare against v2.7 with the same seed, 5k/500 subsets,
validation-only selection, passed overlay, and eight quality gates. Improvement
is judged with per-class F1/MCC/balanced accuracy and bbox IoU, not headline
accuracy alone.
