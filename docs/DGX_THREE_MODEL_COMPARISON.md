# DGX Three-Model Comparison Protocol

## Purpose

This protocol compares three open, unified vision-language backbones while
holding the WebAgent thesis method fixed. It does not declare a winner in
advance. Every candidate uses the same reviewed Gold v2.8 data, causal streams,
four-pillar heads, recovery supplement, losses, validation gates, and
validation-only checkpoint selection.

| Lab PC | Candidate | Registered configuration | Initial status |
| --- | --- | --- | --- |
| PC 1 | Qwen2-VL-2B | `qwen2vl_2b_gold_v2_8_dgx.yaml` | Existing reviewed mini evidence accepted |
| PC 2 | Qwen2.5-VL-7B | `qwen25vl_7b_gold_v2_8_dgx.yaml` | Must pass compatibility and 5k mini gates |
| PC 3 | InternVL3.5-8B-HF | `internvl35_8b_gold_v2_8_dgx.yaml` | Must pass compatibility and 5k mini gates |

All three base checkpoints are free to download and use Apache-2.0-licensed
model repositories. The experiment uses 4-bit QLoRA; it is not training a VLM
from scratch.

## Immutable scientific contract

- Training: 23,499 reviewed original rows plus 608 accepted RETRY/ABORT rows,
  for 24,107 rows total.
- Primary validation: all 7,861 original-Gold validation rows.
- Supplement validation: all 194 supplement rows, reported separately.
- Locked test: zero reads during compatibility, mini, full training, model
  selection, and fixed-seed final promotion.
- Seed for the first comparison: 42.
- Maximum full epochs: 10, with the registered early-stopping behavior.
- Effective batch: 32 for every candidate. Physical micro-batch may differ.
- Selection: an epoch must pass all registered quality gates, then eligible
  epochs are ranked by original-validation outcome MCC.
- Model ranking: outcome MCC; deterministic ties use recovery-outcome MCC,
  action macro-F1, lower outcome ECE, then model ID.
- The task heads, loss masks, causal routing, training rows, validation rows,
  and selection rule may not be changed between candidates.

InternVL normally performs dynamic image tiling. That behavior is explicitly
disabled here. Dynamic tiling creates several local coordinate planes for one
web screenshot, while this thesis bbox target is normalized to one original
page plane. The registered InternVL processor therefore emits one 448x448 patch
per source image. The loader fails closed if this setting is removed.

## One-time setup on every PC

Use the same Git commit and the same extracted data layout on all machines:

```text
/home/aiub/kiyas/webagent
/home/aiub/kiyas/webagent_full/data/original
/home/aiub/kiyas/webagent_full/data/supplement
/home/aiub/kiyas/webagent_comparison
```

From the repository virtual environment, install the project requirements.
Do not replace the lab's working CUDA build of PyTorch with a CPU wheel.
Transformers must be at least 4.57.6 and lower than 5.0 for this registered
run.

Before opening Jupyter or VS Code, pull the same commit on all three machines.
Do not pull new code into a run directory after training has started. A resume
must use the original commit and complete output directory.

## Notebook execution

Open `notebooks/dgx_three_model_comparison.ipynb` using the repository virtual
environment kernel.

On every PC:

1. Run cells 1 through 6.
2. Confirm CUDA, bf16, dependency, data-count, supplement-integrity, and
   zero-test-read checks pass.
3. Run exactly one assigned launch cell:
   - PC 1 runs cell 7 only.
   - PC 2 runs cell 8 only.
   - PC 3 runs cell 9 only.
4. Leave the notebook and terminal running until the candidate completes or
   the machine is intentionally stopped.

The launch cell uses `phase='auto'`:

1. A 16-row model compatibility gate verifies processor inputs, tensor shapes,
   finite loss, backward flow, and causal streams.
2. Qwen2.5-VL-7B and InternVL3.5-8B then run their own controlled 5,000-row,
   500-validation-row, five-epoch mini. A failed mini blocks full training.
3. Qwen2-VL-2B reuses the already accepted reviewed v2.8 mini evidence.
4. An eligible candidate starts or resumes the 24,107-row full run.

The first two steps for a new backbone validate new model code; they do not
re-review or replace the finalized dataset.

## Checkpoints and interruption recovery

Each candidate writes only below:

```text
/home/aiub/kiyas/webagent_comparison/outputs/model_comparison/
  MODEL_ID/seed_42/
    run_contract.json
    model_compatibility_report.json
    mini/                         # required for the two new backbones
    full/
      epoch_metrics.csv
      diagnostics.json
      source_validation.csv
      report.json
      checkpoints/MODEL_FULL_SEED42/last.ckpt
```

`last.ckpt` contains optimizer, scheduler, scaler, RNG, early-stopping,
completed history, and partial-epoch state. Rerun the same assigned cell after
an interruption; it resumes the exact next physical batch. Do not delete the
metrics CSV, copy only one checkpoint, or change the code/config while
resuming. Copy or back up the complete `seed_42` directory.

An existing failed compatibility or mini report is deliberately not
overwritten. Diagnose the evidence, commit a fix, and use a new run directory
or commit-bound experiment instead of silently erasing the failure.

## Comparing the completed runs

After all three full runs pass, copy the three complete model directories onto
one comparison machine under the same `model_comparison` parent. Before running
the comparison, compute the exact SHA-256 of each candidate's
`model_compatibility_report.json` and commit the three distinct model-to-hash
assignments in
`REGISTERED_MODEL_COMPATIBILITY_REPORT_SHA256_BY_MODEL` in
`src/web_agent/eval/table2/selection_evidence.py`. A hash asserted only by the
selection input is not preregistration and is rejected. The production registry
intentionally contains only PC-01 until the real PC-02 and PC-03 bytes exist;
therefore the final selection fails closed today. The Table 2 selection input
must repeat the corresponding registered hash. Its validator preserves each
report as a sixth per-candidate artifact, replays its 16-row smoke invariants,
requires distinct source-registered identities, and binds the current smoke
generator and validator sources. Compatibility-report values are never ranking
inputs; the reports only prove that each candidate exercised the shared
pipeline before training. Then run notebook cell 10. It creates:

```text
outputs/comparison_decision/three_model_validation_comparison.csv
outputs/comparison_decision/three_model_selection.json
```

The comparison script refuses missing, failed, wrong-row-count,
wrong-selection-rule, supplement-mixed, or test-reading reports. It ranks only
eligible validation results; it does not evaluate the locked test.

## Final model decision

Use the three candidates' completed seed-42 validation packages for the
registered validation-only comparison. Do not retrain PC-01 and do not add
seeds 43 or 44 for Table 2. Promote the winning seed-42 checkpoint, and report
explicitly that the paired browser experiment measures task uncertainty but
not model-seed uncertainty. Freeze the model and analysis plan before the
single locked-test evaluation used for the paper.

Until the actual DGX compatibility gate, controlled mini, and full reports
exist, the two new integrations are code-ready but not empirically validated.
