# Table 1 — Three-model full-run validation comparison

Updated: 2026-09-22. Scope: interaction diagnosis, recorded recovery-outcome
assessment, and memory-storage prediction. Every row uses one selected
full-run checkpoint; no metric is taken from another epoch's best result.

## Paper-ready table

**Table 1. Component-level validation performance of the four-pillar WebAgent
with three vision-language backbones.** All candidates use seed 42 and are
evaluated on the same 7,861 Web-Gold-40K validation interactions. Recovery
metrics use the same 1,858 attempted-recovery cases. Higher is better for every
displayed metric. Bold marks the highest observed value. Accuracy is expressed
as a percentage; Macro-F1 and MCC are coefficients.

| Backbone | Outcome MCC ↑ | Failure Macro-F1 ↑ | Failure-type Macro-F1 ↑ | Recovery Accuracy ↑ | Recovery Macro-F1 ↑ | Recovery MCC ↑ | Memory Accuracy ↑ | Memory Macro-F1 ↑ | Memory MCC ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-VL-2B | 0.624 | 0.809 | 0.502 | 90.85% | 0.898 | 0.796 | 84.28% | 0.830 | 0.663 |
| **Qwen2.5-VL-7B** | **0.678** | **0.839** | **0.542** | **93.49%** | **0.925** | **0.852** | **85.77%** | **0.846** | **0.695** |
| InternVL3.5-8B-HF | 0.641 | 0.820 | 0.541 | 92.90% | 0.920 | 0.840 | 83.91% | 0.821 | 0.654 |

**Table note.** Selected epochs are PC1 epoch 6, PC2 epoch 0, and PC3 epoch 0
(zero-based numbering). Memory uses all 7,861 validation cases. These are
single-seed validation measurements, not locked-test results. Different source
commits remain an experimental-control limitation. Bold values indicate
observed differences, not statistical significance.

## Experimental scope

| Item | Definition |
|---|---|
| Data | Reviewed Web-Gold v2.8 full-run data |
| Training | 24,107 cases: 23,499 original plus 608 supplement cases |
| Primary validation | 7,861 original cases |
| Recovery denominator | 1,858 eligible attempted-recovery cases |
| Supplement validation | 194 cases, reported separately and excluded from this table |
| Seed | 42 for all three models |
| Checkpoint rule | Registered quality gates, then original-validation outcome MCC |
| Backbone ranking | Outcome MCC; ties: recovery MCC, action Macro-F1, lower outcome ECE, model ID |
| Additional inference for this table | None; metrics extracted from saved selected epochs |

## Selection statement for the paper

Under the registered numerical ranking criterion, Qwen2.5-VL-7B is the leading
candidate because it achieved the highest outcome MCC (0.678). It also obtained the strongest
recovery-outcome MCC (0.852) and macro-F1 (0.925), memory MCC (0.695) and
macro-F1 (0.846), and bounding-box mean IoU (0.155). Its corresponding raw
accuracies are 93.49% for recovery outcome and 85.77% for memory storage.
InternVL3.5-8B-HF produced the strongest action macro-F1 (0.337),
Recall@IoU50 (0.095), and calibration (ECE 0.040), showing that the backbone
ranking varies by component. These secondary results did not override the
registered primary selection metric. Formal promotion remains blocked by
Cell 10's source-commit provenance check; this document is not a PASS receipt
for that check.

Suggested manuscript wording:

> Qwen2.5-VL-7B achieved the highest observed validation scores across the nine
> reported assessment metrics, including outcome MCC of 0.678, recovery-outcome
> MCC of 0.852, and memory-storage MCC of 0.695. Recovery-outcome and
> memory-storage accuracies were 93.49% and 85.77%, respectively. These
> measurements identify it as the leading candidate under the outcome-MCC
> ranking criterion. The comparison uses one seed per backbone, with
> source-revision differences remaining an experimental-control limitation.

## Required evidence note

These results rank candidates using validation data only. They are from one
seed and do not estimate model-seed uncertainty. The locked test partition was
not read. The three immutable run contracts record different Git commits:

| Backbone | Run-contract commit |
|---|---|
| Qwen2-VL-2B | `2fadf0f508cec42ce6f89b8961db7cfd2adef1df` |
| Qwen2.5-VL-7B | `2bd3d0de067da49f8426bd009a69f6e66a0d1bfa` |
| InternVL3.5-8B-HF | `628c1fa776142c8169466c53865e694aa01d6b8d` |

Accordingly, Table 1 supports a descriptive ranking led by Qwen2.5-VL-7B.
It does not establish a fully source-matched comparison, statistically
significant superiority, or improved live browser completion. The source
contracts must remain unchanged. This comparison also does not independently
isolate a gain from each of the four pillars.

## Metric interpretation

- Outcome MCC is the registered primary model-selection metric.
- Failure macro-F1 evaluates the binary interaction outcome/failure head.
- Failure-type macro-F1 evaluates `NONE`, `ACTION_MISMATCH`,
  `PERCEPTION_ERROR`, and `LOOP_DETECTED` equally.
- Recovery accuracy, MCC and Macro-F1 assess actual recorded recovery outcomes
  on eligible attempted-recovery transitions; they do not measure execution
  of new recoveries by an agent.
- Memory accuracy, MCC and Macro-F1 measure the binary storage decision; they are not
  retrieval Recall@k and do not establish live browser improvement.

Accuracy is the fraction classified correctly. Macro-F1 equally weights the
classes. MCC accounts for all four entries of the binary confusion matrix
and should not be described as percentage accuracy.

Action, strategy, grounding, and calibration results remain in the supporting
CSV for supplementary reporting. Their limitations must be discussed whenever
the manuscript makes claims about those capabilities. The main table focuses
on the agreed assessment endpoints rather than claiming comprehensive agent
performance.

## Exact selected checkpoints

| Backbone | Selected checkpoint | SHA-256 |
|---|---|---|
| Qwen2-VL-2B | `best_e6_outcome-mcc0.624.ckpt` | `9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a` |
| Qwen2.5-VL-7B | `best_e0_outcome-mcc0.678.ckpt` | `71f867cc357d0410337a9b23398b109cc95d6c25c2ffc971138b7b8e3aafef2c` |
| InternVL3.5-8B-HF | `best_e0_outcome-mcc0.641.ckpt` | `35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb` |

## Reproducing the table

Source packages are under
`/home/aiub/kiyas/webagent_comparison/outputs/model_comparison/`, using model
directories `qwen2vl_2b_gold_v2_8_dgx`, `qwen25vl_7b_gold_v2_8_dgx`, and
`internvl35_8b_gold_v2_8_dgx`. Each contains `seed_42/run_contract.json`,
`seed_42/full/report.json`, and `seed_42/full/epoch_metrics.csv`.

Read `selected_epoch` from each report and extract that epoch's metrics from
both `history` and the epoch CSV. The paper columns map to `outcome_mcc`,
`failure_macro_f1`, `failtype_macro_f1`, `recovery_outcome_acc`,
`recovery_outcome_macro_f1`, `recovery_outcome_mcc`, `memory_acc`,
`memory_macro_f1`, and `memory_mcc`. Multiply only accuracy values by 100 for
display; retain all other values as coefficients. Never take per-column maxima
from different epochs.

Exact unrounded values and source identities are in the
[supporting CSV](../results/table1_model_comparison/table1_three_model_validation.csv).
Its `selected` flag identifies the descriptive numerical leader only and does
not override the provenance gate or change the model used in prior experiments.
