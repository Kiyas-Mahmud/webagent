# Task 1 — InternVL base, trained decoder and trained heads

## Current status

**2026-09-16: complete. All 2,520 final predictions were scored and independently audited (105 metric checks). The original interrupted attempt and one user-authorized retry remain recorded separately. Results, nine descriptive paired contrasts and execution accounting are available in [the result document](TASK1_INTERNVL_DUAL_V2_RESULTS.md) and [CSV folder](../results/task1_internvl_dual_v2/README.md).** This document is the authoritative status for `internvl_dual_v2`. The completed Qwen comparison remains historical evidence.

## Fixed comparison

Follow-up analysis completed: [saved-response coverage/error audit](TASK1_INTERNVL_COVERAGE_ERROR_ANALYSIS.md). Zero new model calls; original evidence and result exports preserved.

| Row | Method | Weights | Output |
|---|---|---|---|
| browser_use_base | Browser Use-derived assessment | Frozen InternVL base | Decoder JSON |
| agent_s2_base | Agent S2-derived reflection | Frozen InternVL base | Decoder JSON |
| webvoyager_base | WebVoyager-derived evaluation | Frozen InternVL base | Decoder JSON |
| browser_use_trained | Browser Use-derived assessment | Verified trained InternVL LoRA | Decoder JSON |
| agent_s2_trained | Agent S2-derived reflection | Verified trained InternVL LoRA | Decoder JSON |
| webvoyager_trained | WebVoyager-derived evaluation | Verified trained InternVL LoRA | Decoder JSON |
| internvl_heads | Original trained assessment | Same trained checkpoint | Trained logits |

Selection: existing full epoch-0 checkpoint, seed 42, SHA-256 `35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb`. Base revision `3974c115d44f0e13eae8b654f680b1e561c71ae9`. No checkpoint switching, retraining, package upgrades or new dataset review.

Prompts and strict scoring are unchanged. Each decoder sees the same two ordered single-patch 448×448 images. Both variants use NF4 double-quantized 4-bit weights, BF16 compute and SDPA, with the same FP32 preparation of non-quantized frozen parameters; greedy generation has a 128-token limit. The complete generation configuration is frozen before pilot inference: EOS/PAD = 151645, BOS = 151643, cache enabled. PAD now explicitly records the library's previous EOS fallback. Trained generation must execute active, unmerged, verified LoRA layers. Identical predictions are allowed; silent base fallback is not.

Development-v1 is preserved and superseded: its base loader omitted the FP32 preparation applied by the trained LoRA loader. Development-v2 verifies normalized frozen-parameter dtype/shape signatures as well as image inputs and adapter execution. This is an implementation correction, with unchanged prompts/checkpoint.

## Checklist

- [x] Preserve and verify 3,243 historical Qwen/source/evidence files.
- [x] Implement explicit seven-row registry and InternVL-only routing.
- [x] Implement decoder telemetry, adapter hooks and input parity checks.
- [x] Keep original prompt and trained-head ASTs unchanged.
- [x] Strengthen resume to stop on preserved infrastructure/interrupted attempts.
- [x] Pass 60 engineering tests.
- [x] Complete 126 separate development records and trained-head parity.
- [x] Freeze current code, dependencies, assets, prompts and scoring.
- [x] Execute 2,520 pilot records (240 interactions + 120 recoveries per row).
- [x] Reparse raw responses, decode head logits and independently verify metrics.
- [x] Export the seven-row matrix, phase tables and nine descriptive paired contrasts.

## Local commands

From `/home/aiub/kiyas/webagent`, use `.venv/bin/python scripts/external_agents/task1.py COMMAND --profile internvl_dual_v2`. The profile defaults to the preserved selection and `configs/eval/task1/internvl_dual_lab_paths.json`; it never loads Qwen.

- Development: `check --source /home/aiub/kiyas/webagent_full/data/original/final_data_set_40k/split_val.json --out .task1-assets/runs/task1-internvl-dual-v2/development-v2`
- Freeze after PASS: `freeze --development-run .task1-assets/runs/task1-internvl-dual-v2/development-v2 --out .task1-assets/runs/task1-internvl-dual-v2/freeze.json`
- Pilot: `run --freeze .task1-assets/runs/task1-internvl-dual-v2/freeze.json --out .task1-assets/runs/task1-internvl-dual-v2/pilot-reviewed-20260916`
- Score: `score --freeze .task1-assets/runs/task1-internvl-dual-v2/freeze.json --run .task1-assets/runs/task1-internvl-dual-v2/pilot-reviewed-20260916 --out .task1-assets/runs/task1-internvl-dual-v2/report --csv-out results/task1_internvl_dual_v2`

GPU commands require host GPU visibility. No image is uploaded to an API; the locked test split stays unopened. Freeze and report paths refuse overwrite. A repaired implementation needs new development evidence and a new freeze. Completed or rejected model outputs are never selectively rerun.

Pilot monitor:

```bash
watch -n 5 '.venv/bin/python scripts/external_agents/task1.py monitor --profile internvl_dual_v2 --run .task1-assets/runs/task1-internvl-dual-v2/pilot-reviewed-20260916'
```

## Interruption review

The original `pilot/` archive is stopped and preserved. `resume-review-20260916.json` binds the 1,624 copied predictions, original interrupted record and user-authorized retry to `pilot-reviewed-20260916/`. No completed, abstaining or malformed response is rerun. Report the original interruption and one retry separately; the lost attempt’s call count and latency are unknown. The frozen inference code, prompts, checkpoint and scoring are unchanged.

## Interpretation

This is a follow-up on previously observed validation cases, not a new locked test. The six prompt rows are adapted assessment components, not native browser agents. Base/trained decoder contrasts hold method prompts fixed; head/decoder contrasts additionally change the assessment interface. Positive model performance is not an engineering acceptance criterion.

Metrics use valid-output denominators and also report all-case accuracy counting invalid outputs as incorrect. Nine declared contrasts (three trained-minus-base decoder and six head-minus-prompt) use common valid pairs, task-group bootstrap intervals, 10,000 resamples and seed 20250831. No confirmatory significance, browser completion, executed recovery or memory-benefit claim follows from this experiment.
