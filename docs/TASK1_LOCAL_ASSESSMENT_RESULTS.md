> Completed InternVL follow-up: [seven-row results and audited tables](TASK1_INTERNVL_DUAL_V2_RESULTS.md). The Qwen results below are preserved historical evidence.

Saved-response analysis: [why prompted InternVL coverage is low](TASK1_INTERNVL_COVERAGE_ERROR_ANALYSIS.md). Audit complete; original scores unchanged.

# Task 1: completed local assessment comparison

Current work and authoritative status: [InternVL dual comparison](TASK1_INTERNVL_DUAL_V2.md).

## Historical Qwen comparison and earlier work

CSV exports: [result matrix](../results/task1_qwen_backend_v1/result_matrix.csv) · [folder guide](../results/task1_qwen_backend_v1/README.md).

2026-09-15. **Complete: 1,440/1,440 records. Verification PASS.**
Frozen trained InternVL full epoch 0, seed 42 versus three adapted assessment prompts using the same frozen Qwen2-VL-2B base. These are recorded-transition assessments on the selected validation pilot, not native agent/browser-completion results.

## Interaction assessment — 240 cases

| System | Outcome MCC | Balanced accuracy | Outcome Macro-F1 | Failure-category Macro-F1 | All-case accuracy | Valid outputs |
|---|---:|---:|---:|---:|---:|---:|
| Browser Use assessment + Qwen | 0.0000 | 50.00% | 0.2056 | 0.1248 | 24.58% | 228/240 |
| Agent S2 reflection + Qwen | 0.0000 | 50.00% | 0.1987 | 0.0993 | 24.58% | 238/240 |
| WebVoyager evaluator + Qwen | 0.0000 | 50.00% | 0.2000 | 0.1111 | 22.92% | 220/240 |
| Trained InternVL3.5-8B | 0.5583 | 80.83% | 0.7678 | 0.5287 | 80.42% | 240/240 |

## Recorded recovery assessment — 120 cases

| System | Recovery MCC | Recovery Macro-F1 | All-case accuracy | Valid outputs |
|---|---:|---:|---:|---:|
| Browser Use assessment + Qwen | 0.0000 | 0.3293 | 45.83% | 112/120 |
| Agent S2 reflection + Qwen | 0.0000 | 0.3333 | 50.00% | 120/120 |
| WebVoyager evaluator + Qwen | 0.0000 | 0.3354 | 45.83% | 109/120 |
| Trained InternVL3.5-8B | 0.8844 | 0.9416 | 94.17% | 120/120 |

## Verification and interpretation

- All 1,440 identities/records checked against the frozen profile; every external raw response was reparsed and trained predictions were checked against saved logits. Sixty scalar metric values were independently recomputed with scikit-learn and matched.
- No missing episodes/requests, infrastructure errors or interrupted requests. Parsing failures: Browser Use 20, Agent S2 2, WebVoyager 31, InternVL 0. No abstentions.
- MCC, balanced accuracy and Macro-F1 use each displayed valid denominator. All-case accuracy counts invalid outputs as incorrect. Paired comparisons use common valid cases, with task-group bootstrap intervals (10,000 resamples, seed 20250831) in the detailed JSON.
- InternVL: 193/240 correct interaction outcomes and 113/120 correct recorded recovery outcomes. Interaction labels contain 180 failures and 60 successes; recovery labels are 60/60. A constant-FAILURE interaction predictor would have 75% accuracy but 0 MCC and 50% balanced accuracy.
- Every valid external outcome prediction was SUCCESS. Their zero MCC is reproducible from raw responses, not a rounding artifact. This does not establish the cause of their constant predictions or validate them as strong baselines.
- The comparison involves a dataset-trained 8B model and adapted prompts on an unadapted 2B model. It does not isolate training, model-size or framework effects. Investigate the external constant-SUCCESS behavior before making broad paper claims; retain these frozen results and declare any follow-up as a separate experiment.
- These findings do not demonstrate improved browser task completion, executed recovery or memory benefit. No retraining or locked-test access occurred.

## Local evidence

- [Producing report](../.task1-assets/runs/task1-local-v1/report/results.md)
- [Detailed metrics and paired intervals](../.task1-assets/runs/task1-local-v1/report/results.json)
- [Independent verification](../.task1-assets/runs/task1-local-v1/report/verification.json)
- [Frozen configuration](../.task1-assets/runs/task1-local-v1/freeze.json)
- Raw records: `.task1-assets/runs/task1-local-v1/pilot/` (local ignored evidence; not pushed).
