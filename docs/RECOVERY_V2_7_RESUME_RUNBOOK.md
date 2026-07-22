# Recovery v2.7 interrupted-mini continuation

## Purpose

The previous v2.7 Kaggle mini did not produce an epoch-4 validation row because the process ended
after `epoch 4 | step 150/157`. Its log contains no model exception. The preserved epoch-3 artifact
is complete (`epoch=3`, `step=628`, `epoch_complete=true`) and already passes all eight registered
quality gates. This workflow restores that exact training state and runs only the missing epoch 4.

It does not combine v2.6 with v2.7, combine independent v2.7 reruns, read test rows, or change the
data, backbone, architecture, loss, optimizer, seed, validation subset, or quality thresholds.

## Kaggle inputs

Create a new Kaggle notebook from
`notebooks/kaggle_gold_recovery_v2_7_resume.ipynb`, enable a GPU and Internet, and attach:

1. `kiyasmahmud/web-gold-40k` (the original structured dataset).
2. The failed v2.7 notebook output as a Kaggle input. It must contain:
   - `gold_mini_recovery_v2_7_metrics.csv` with exactly epochs 0, 1, 2, and 3;
   - one `best_e{epoch}_*.ckpt` file for each epoch 0–3 in the same checkpoint directory;
   - `gold_recovery_v2_7_environment.json`;
   - no completed epoch-4 checkpoint.

The prior output may be attached as loose files or as the original `results.zip`. When it is a ZIP,
the resume notebook identifies the archive by its contents and extracts only the two metadata files
and four checkpoints into `/kaggle/working/v2_7_prior_artifacts`. It does not extract the Gold data
archive. Approximately 1.2 GB of Kaggle working storage is required for these checkpoints.

Do not attach `results.zip` as the Gold dataset and do not connect `kaggle_gold.ipynb`.

## What the notebook proves before training

- Package and Python versions equal the interrupted run's saved environment.
- The original training commit is `c583e24fdfddac417e84e8cf306b251c305988fe`.
- The CSV contains consecutive completed epochs 0–3 only.
- Every prior epoch has exactly one retained checkpoint.
- The resume artifact is the epoch-3 mapping, not `last.ckpt`.
- Checkpoint metadata says epoch 3, global step 628, completed epoch, and outcome-MCC selection.
- Its full training signature matches the current v2.7 mini configuration.
- Its selection value agrees with the rounded epoch-3 CSV row.
- Train/validation bbox audit passes and the test split remains unopened.

Any mismatch stops before model training.

## What it runs

The model, loss, fixed 5,000-row training subset, fixed 500-row validation subset, and
recovery-aware loader are rebuilt from the same registered configuration. The epoch-3 checkpoint
restores weights, task adapters, heads, learned loss weights, optimizer, scheduler, scaler, and
global step. Training starts at epoch 4 and stops after epoch 4.

The historical checkpoint predates RNG-state saving. Therefore epoch 4 uses the registered seed 42
but cannot reproduce the interrupted process's in-memory random stream bit for bit. This limitation
is written into the report; it is not hidden. New checkpoints now include Python, NumPy, CPU Torch,
and CUDA RNG states so a later interruption can be reproduced exactly.

## Required outputs

After `Run All`, preserve:

- `/kaggle/working/gold_recovery_v2_7_resume_report.json`
- `/kaggle/working/gold_recovery_v2_7_resume_result.csv`
- `/kaggle/working/gold_recovery_v2_7_resume_diagnostics.json`
- `/kaggle/working/gold_recovery_v2_7_resumed_epoch_metrics.csv`
- `/kaggle/working/gold_recovery_v2_7_resume_bbox_audit.json`
- `/kaggle/working/gold_recovery_v2_7_resume_environment.json`
- `/kaggle/working/gold_recovery_v2_7_selected.ckpt`

`PASS` requires exactly epochs `[0, 1, 2, 3, 4]`, at least one all-gate-eligible epoch, synchronized
selection metadata, an explicit selected-checkpoint prediction round-trip, agreement between that
checkpoint's new validation metrics and its recorded history row, and `test_rows_read == 0`.
