# Recovery v2 controlled experiment

## Purpose

Recovery v1 proved that sampling and class weighting alone do not solve the causal and
spatial defects. Recovery v2 is a registered structural experiment; it does not overwrite
v1 and it is not a headline/full-data run.

## Data contract

The exporter joins existing rows within each `task_id` by increasing `step_index` and creates
references—not image copies—with this meaning:

`failure state -> executed recovery action -> post-recovery state -> recovery success`

A proper transition requires a non-null `recovery_success`, a non-`NONE` strategy, a failure
source row, a later row in the same task, an executed action, and both state images. Missing,
ambiguous, nonconsecutive, and invalid cases are reported. They are never silently used as
recovery-success supervision.

The current action is included only in the post-action prompt. The pre-action action/bbox/
confidence stream remains `state_before + task/domain`, so the model cannot see the action it
is supposed to predict.

## Registered model changes

1. Binary `needs_recovery` uses all rows.
2. Recovery strategy CE uses attempted-recovery rows only.
3. Recovery-success BCE uses only the sparse causal transition stream.
4. The bbox head attends over preserved image-token features. Its primary loss is SmoothL1
   plus GIoU; primary localization metrics are mean IoU and Recall@IoU50.
5. Policy, diagnosis, memory, and recovery have separate residual adapters.
6. Learnable uncertainty weights balance active task losses; inactive sparse losses do not
   update their uncertainty parameter.

## Reproducible execution

Use `notebooks/kaggle_gold_recovery_v2.ipynb`. Attach the existing Kaggle dataset at
`/kaggle/input/datasets/kiyasmahmud/web-gold-40k`; the notebook discovers the nested split
folder and never downloads or extracts the dataset.

Run `STAGE = 'smoke'` first. After it passes, restart the kernel, set `STAGE = 'mini'`, and
run all cells. Mini is fixed to 5,000 train rows, 500 validation rows, seed 42, and five
epochs. Model training/selection never opens the test split. The separate dataset audit may
count test labels to determine class completeness, but it never creates test predictions.

## Required outputs

- `gold_recovery_v2_result.csv`
- `gold_recovery_v2_diagnostics.json`
- `gold_recovery_v2_mini_report.json`
- `gold_recovery_v2_environment.json`
- `recovery_transition_v2/recovery_class_audit.json`
- one reference-only transition manifest per split
- all five epoch checkpoints

## Advance/block rule

Engineering PASS only proves correct execution. Quality PASS is evaluated at the epoch chosen
by validation outcome MCC and requires outcome/action retention, lift above majority for both
hierarchical recovery levels, improved transition recovery-outcome MCC, mean bbox IoU >= 0.05,
Recall@IoU50 >= 0.01, and bounded calibration regression. Full training stays blocked if any
required human review/hash gate is incomplete or if the class audit shows a class claimed by the
paper cannot be trained/evaluated. Missing classes require targeted completion only; the 39k
dataset must not be recollected wholesale.
