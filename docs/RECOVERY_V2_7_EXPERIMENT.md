# Recovery v2.7 - checkpoint selection correction

## Purpose

V2.7 corrects how one checkpoint is selected from a completed controlled run.
It does not change the v2.6 data, architecture, loss, optimizer, seed, validation
rows, thresholds, or model weights.

## Registered selection rule

1. Evaluate the eight existing quality gates independently for every epoch.
2. An epoch is eligible only when every gate passes.
3. Select the eligible epoch with the highest `outcome_mcc`.
4. Break an exact tie in favor of the earlier epoch.
5. If no epoch is eligible, report quality `FAIL`. The unconstrained outcome-MCC
   epoch may be retained for diagnosis, but it cannot be reported as passing.

The gates remain constraints; they are not summed into a post-hoc pillar score.

## V2.6 replay result

The immutable five-epoch v2.6 report gives:

- unconstrained outcome-MCC epoch: 1;
- all-gate-eligible epochs: 3 and 4;
- selected epoch: 4, because its outcome MCC is higher than epoch 3;
- selected outcome MCC: `0.5480665242205773`;
- test rows read: 0;
- all eight selected-epoch quality checks: pass.

V2.6 remains recorded under its original outcome-only rule. The fast replay is
identified as v2.7 and explicitly records that it did not retrain the model.

## Artifact consistency

A fresh v2.7 mini must use the selected epoch for all of the following:

- `selected_epoch`;
- `best_metric`;
- `best_checkpoint`;
- checkpoint loading and prediction round-trip verification;
- quality-gate checks;
- CSV `is_selected` / `is_best` markers.

The unconstrained outcome maximum is retained separately as
`unconstrained_best_metric` and `unconstrained_best_checkpoint`.

## Fast Kaggle replay

Run `notebooks/kaggle_gold_recovery_v2_7_selection.ipynb` without a GPU. This
file preserves the successful Kaggle replay output. It prefers an attached
`gold_recovery_v2_6_mini_report.json`; when none is attached, it reads the
completed report embedded in the repository's executed v2.6 notebook.

Outputs:

- `/kaggle/working/gold_recovery_v2_7_selection_report.json`;
- `/kaggle/working/gold_recovery_v2_7_selection.csv`.

If the prior Kaggle output is attached and contains epoch 4's checkpoint, the
replay also resolves and hashes that file. `NOT_MOUNTED` means only that the old
checkpoint file was not attached. It does not pretend to be a new prediction
round-trip. A fresh training run using
`notebooks/kaggle_gold_recovery_v2_7.ipynb` and
`configs/backbones/qwen2vl_2b_gold_v2_7.yaml` performs the real selected-checkpoint
load and prediction round-trip.

## Full Kaggle workflow

`notebooks/kaggle_gold_recovery_v2_7.ipynb` is the output-free training
notebook. It retains the complete v2.6 stage structure: `audit`, `smoke`,
`bbox_overfit`, `diagnostic`, and `mini`. It defaults to the five-epoch mini
because v2.7 changes no structural training factor and the identical v2.6 gates
were already accepted. Set a different stage only when deliberately repeating
that gate, and restart the kernel before changing stages.

The bbox-overfit assertions now derive the trace length and full-set evaluation
steps from configuration. This matches the inherited v2.6/v2.7 registration of
200 optimizer steps and prevents the old notebook's stale 100-step assertion.
The five-epoch mini additionally verifies the constrained selected epoch,
checkpoint identity, prediction round-trip, and exactly one selected CSV row.
