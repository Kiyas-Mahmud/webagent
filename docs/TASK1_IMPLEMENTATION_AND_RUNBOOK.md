# Task 1 implementation, results and execution checklist

Current work and authoritative status: [InternVL dual comparison](TASK1_INTERNVL_DUAL_V2.md).

## Historical Qwen comparison and earlier work

> **Completed, 2026-09-15:** All 1,440 comparison records are saved and scored.
> Independent raw-output/logit and metric verification passed. InternVL outcome
> MCC = 0.5583; recovery MCC = 0.8844. All three adapted Qwen configurations
> predicted SUCCESS for every valid outcome response (MCC = 0).
> See [completed results and limitations](TASK1_LOCAL_ASSESSMENT_RESULTS.md).
> This completion notice supersedes the earlier running/pending statuses below.

> **Local evaluation update, 2026-09-15:** The existing dataset was verified
> (matching validation hash; all 626 required images decoded). The exact InternVL
> epoch-0 checkpoint was fetched from Git LFS and hash-verified; its pinned base
> snapshot is now local. NVIDIA GB10 CUDA/BF16 checks passed.
> All 72 development records are saved (71 valid, one parsing failure retained).
> InternVL processor/logit parity passed. The configuration is frozen and the
> 1,440-record comparison has been launched serially.
> Monitor `.task1-assets/runs/task1-local-v1/status.json` and `execution.log`.
> Completed comparison scores remain pending until the run and scoring finish.
> This update supersedes earlier host/download/development statuses below.

> **PC identity correction, 2026-09-15:** InternVL is **PC-03**, as assigned in
> `notebooks/dgx_three_model_comparison.ipynb`. Earlier Task 1 PC-02 labels were
> incorrect. The selected InternVL candidate, epoch-0 checkpoint digest and
> historical metrics are unchanged. No comparative inference has run.

| Notebook machine | Candidate | Task 1 role |
|---|---|---|
| PC 1 | `qwen2vl_2b_gold_v2_8_dgx` | Trained Qwen reference; external adapters use its unadapted base |
| PC 2 | `qwen25vl_7b_gold_v2_8_dgx` | Incomplete per lab status; not selected |
| PC 3 | `internvl35_8b_gold_v2_8_dgx` | Selected trained InternVL comparator (`pc03`) |

Original engineering/preflight receipts retain their producing identifiers;
this correction does not rewrite those historical receipts.


Updated 2026-09-15. **Implemented locally; real comparative inference is pending.**
The user authorized the existing local lab dataset on 2026-09-15. Its validation
metadata matches the frozen source, and all 626 required image files are present.
The GB10 GPU supports CUDA and BF16 outside the execution sandbox. InternVL
checkpoint and pinned base are downloaded; the checkpoint SHA-256 matches
the selected full epoch-0 artifact. Full development/parity is running.
External-component development inference has started; comparative inference is pending.
Only selected development images enter the running model check. The locked test
split remains unopened; the 240-case comparative run has not started.

## Fixed experiment

| System | Backend | Behavior |
|---|---|---|
| Our model | PC-03 InternVL3.5-8B-HF, full-trained epoch 0, seed 42 | Original processor, model forward and trained heads |
| Browser Use assessment | Frozen Qwen2-VL-2B-Instruct | Adapted previous-action verification |
| Agent S2 reflection | Same frozen Qwen | Adapted reflection on the recorded transition |
| WebVoyager evaluator | Same frozen Qwen | Whole-task evaluator explicitly adapted to transition assessment |

The 240 interaction cases and 120 linked recovery cases remain unchanged.
Twelve separate development task IDs provide 18 requests (12 interaction and
six recovery requests). None shares a task ID with the pilot or its dependencies.
There will be **1,440 pilot system-phase records**, not 1,440 live episodes.

Source identities and licenses are preserved under `configs/eval/task1/upstream`:

| Component | Pinned commit | Retained part |
|---|---|---|
| Browser Use | `843819cb8131e1370948d381ede9be7f8366ddc4` | Previous-action screenshot verification rule |
| Agent S2 | `3aa272d23d2994c7bbde1acbbe0ef8e8d06b8693` | `REFLECTION_ON_TRAJECTORY` |
| WebVoyager | `5a7896738c10bfb8b9edccce6bb0e0411f8ae569` | Evaluator `SYSTEM_PROMPT` |

Each prompt retains its method-specific excerpt and adds an explicit recorded-
transition, missing-context and compact-output contract. Live tools, private
reasoning history, prior plans and agent final answers are not fabricated.
This is an adapted component comparison; the full agents are not running.
The shared small Qwen backend differs from the projects' original configurations.

## Existing PC-03 results (historical reference only)

These are **epoch 0** original-validation results, not new pilot results.
The exact selected checkpoint SHA-256 is
`35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb`
(645,434,661 bytes). The corresponding report identity and run contract are
preserved under `configs/eval/task1/pc03_reference.json` and
`pc03_run_contract.json`. No weights were downloaded or substituted.

| Capability | Metric | Saved result | Cases |
|---|---|---:|---:|
| Interaction outcome | MCC | 0.6410 | 7,861 |
| Interaction outcome | Balanced accuracy | 82.26% | 7,861 |
| Interaction outcome | Macro-F1 | 0.8201 | 7,861 |
| Failure category | Macro-F1 | 0.5407 | 7,861 |
| Recorded recovery outcome | MCC | 0.8397 | 1,858 |
| Recorded recovery outcome | Macro-F1 | 0.9198 | 1,858 |
| Recorded recovery outcome | Accuracy | 92.90% | 1,858 |
| Memory storage | MCC / Macro-F1 | 0.6543 / 0.8209 | 7,861 |
| Attempted recovery strategy | Macro-F1 | 0.4892 | 1,858 |
| Action type | Macro-F1 | 0.3370 | 7,861 |
| Grounding | Mean IoU / Recall@IoU50 | 0.0922 / 9.54% | 2,349 |

The source name `failure_macro_f1` refers to binary interaction outcome;
`failtype_macro_f1` refers to four failure categories. Do not confuse them or
combine metrics selected from different epochs.

## Tables to fill after inference

### A: same 240 interaction examples

| System | Outcome MCC | Balanced accuracy | Outcome Macro-F1 | Failure-category Macro-F1 | Valid outputs |
|---|---:|---:|---:|---:|---:|
| Browser Use assessment + Qwen | Pending | Pending | Pending | Pending | Pending / 240 |
| Agent S2 reflection + Qwen | Pending | Pending | Pending | Pending | Pending / 240 |
| WebVoyager evaluator + Qwen | Pending | Pending | Pending | Pending | Pending / 240 |
| Our PC-03 InternVL | Pending | Pending | Pending | Pending | Pending / 240 |

### B: same 120 recorded recovery transitions

| System | Recovery MCC | Recovery Macro-F1 | Accuracy | Valid outputs |
|---|---:|---:|---:|---:|
| Browser Use assessment + Qwen | Pending | Pending | Pending | Pending / 120 |
| Agent S2 reflection + Qwen | Pending | Pending | Pending | Pending / 120 |
| WebVoyager evaluator + Qwen | Pending | Pending | Pending | Pending / 120 |
| Our PC-03 InternVL | Pending | Pending | Pending | Pending / 120 |

JSON reporting additionally records valid and all-case accuracy, confusion
matrices, per-class recall, all statuses, known/unknown model-call counts and
latency. Class metrics use the displayed valid subset; invalid/missing predictions
count as wrong for all-case accuracy. Entirely missing passes remain incomplete.
No failed request is silently removed. MCC uses the standard zero convention
for degenerate confusion matrices. Macro-F1 uses the fixed task label vocabulary;
classes absent from both truth and prediction contribute zero F1.

Paired differences use the intersection of valid cases for each system pair.
Resample task groups with replacement, preserving all their rows: 10,000 draws,
seed 20250831, percentile 95% intervals. Intervals condition on valid predictions
and do not measure training-seed uncertainty. Report coverage and all-case
accuracy alongside these conditional comparisons. There is no confirmatory
significance claim in this enriched validation pilot.

## Execution architecture and safety boundaries

- `src/web_agent/eval/task1/core.py`: allowlisted phase inputs, deterministic
  development selection, exact-one-object parsing, exclusive writes, locking,
  and resume of unstarted requests only.
- `backends.py`: source-bound component prompts, local Qwen generation, strict
  PC-03 state restoration, original tensorization/forward and development parity.
- `reporting.py`: completeness checks, shared metrics, paired intervals and tables.
- `cli.py`: prepare/preflight/check/freeze/run/score; model imports are delayed.

Never use `evaluate_all_splits`: it opens locked-test data. The new path loads
only explicitly named original validation metadata and permitted image references.
Labels remain in separate scoring files; the predictor receives only its
allowlisted request. Recorded action type is context for assessment, not a
prediction target in this version. Reference boxes and all outcome/category/
strategy/storage labels stay hidden. Missing action values stay null.

For recovery: source `state_after` -> next recorded action -> next `state_after`.
PC-03 uses its original recovery stream and zero-logit success threshold. The
other heads cannot see future recovery observations during interaction assessment.

Source/config hashes, dependencies, image hashes and model-file identities are
bound before inference. Models use local files only. The inference command
requires the configured absolute local image root and a CUDA device supporting the
unchanged BF16 configuration; an incompatible host must not silently change dtype.
There is no external API client, browser worker, embedding generation or training.

## Commands on Kaggle or the authorized lab host

Use the existing compatible model environment. Copy the repository code and the
prepared metadata, and mount the **existing** selected models and Gold images.
Edit a copy of `configs/eval/task1/kaggle_paths.example.json` outside the source
snapshot to give their actual mounted paths. The example paths are placeholders.
Base snapshot directories must retain their pinned revision names. The local lab configuration is `configs/eval/task1/lab_paths.json`; it points
to the existing local dataset. No image is sent to external APIs.

Run from the repository root. In these commands, `/kaggle/working/task1-paths.json`
is the filled configuration; existing original validation metadata is
`/kaggle/input/web-gold-40k/split_val.json` (substitute its actual mounted path).

The metadata has already been prepared in
`docs/evidence/task1-assessment-prepared-v1`; do not regenerate it unnecessarily.

```bash
python scripts/external_agents/task1.py preflight \
  --prepared docs/evidence/task1-assessment-prepared-v1 \
  --config /kaggle/working/task1-paths.json \
  --out /kaggle/working/task1-preflight.json

python scripts/external_agents/task1.py check \
  --prepared docs/evidence/task1-assessment-prepared-v1 \
  --config /kaggle/working/task1-paths.json \
  --source /kaggle/input/web-gold-40k/split_val.json \
  --out /kaggle/working/task1-development-v1

python scripts/external_agents/task1.py freeze \
  --prepared docs/evidence/task1-assessment-prepared-v1 \
  --config /kaggle/working/task1-paths.json \
  --development-run /kaggle/working/task1-development-v1 \
  --out /kaggle/working/task1-freeze-v1.json

for system in browser_use agent_s2 webvoyager pc03; do
  python scripts/external_agents/task1.py run \
    --prepared docs/evidence/task1-assessment-prepared-v1 \
    --config /kaggle/working/task1-paths.json \
    --freeze /kaggle/working/task1-freeze-v1.json \
    --system "$system" --out /kaggle/working/task1-pilot-v1 || break
done

python scripts/external_agents/task1.py score \
  --prepared docs/evidence/task1-assessment-prepared-v1 \
  --config /kaggle/working/task1-paths.json \
  --freeze /kaggle/working/task1-freeze-v1.json \
  --run /kaggle/working/task1-pilot-v1 \
  --out /kaggle/working/task1-report-v1
```

`check` preserves raw development outputs and compares PC-03 tensors/logits with
original GoldDataset inference. No accuracy threshold selects the prompt. A
PASS receipt establishes code-path/parity execution, not useful predictions.
Inspect schema failure counts before freezing; poor model JSON remains a recorded
failure. Any implementation repair requires a new development output and receipt.

Repeat `run` with the identical arguments to resume only requests never started.
Started but unfinished requests become interrupted records; their original
markers remain. Infrastructure errors stop the pass. Do not relaunch failed
cases under another profile and merge the successful attempts into this result.
`freeze`, prepared files and reports refuse overwrite. Source/environment changes
invalidate a freeze; score using the producing environment and source snapshot.

## Current checklist

- [x] Selected PC-03 full epoch 0 and shared Qwen backend.
- [x] Pinned and preserved three upstream assessment sources and licenses.
- [x] Preserved historical PC-03 metrics and run contract.
- [x] Implemented input views and prepared unchanged 240-case pilot.
- [x] Selected 12 task-disjoint development cases without opening images.
- [x] Implemented all three adapted assessment interfaces.
- [x] Implemented PC-03 loading/forward/parity path (not yet model-executed).
- [x] Implemented serial runner, persistence, resume and reporting.
- [x] Passed local engineering tests; these are not model-performance evidence.
- [x] Locate and authorize the existing local dataset; confirm all 626 required image files.
- [ ] Complete model downloads and pass local image/model/CUDA preflight.
- [ ] Run actual development inference and PC-03 parity.
- [ ] Freeze the verified runtime profile.
- [ ] Execute and audit 1,440 pilot system-phase records.
- [ ] Fill result tables and report uncertainty/cost/limitations.

**Current asset status, 2026-09-15:** Local dataset and Qwen base are present.
InternVL checkpoint exists as a Git LFS pointer on `origin/results/dgx-model-comparison`;
the payload was absent from the checked-out branch/cache. It has now been
fetched and SHA-256 verified at `.task1-assets/best_e0_outcome-mcc0.641.ckpt`.
The pinned InternVL base snapshot was absent and has now downloaded successfully.
GPU verification outside the sandbox passed on NVIDIA GB10 with BF16 support.
This supersedes the earlier Kaggle-only host blocker; historical receipts are retained.

Use `configs/eval/task1/lab_paths.json` on the lab PC and the existing validation
file at `/home/aiub/kiyas/webagent_full/data/original/final_data_set_40k/split_val.json`.
Run preflight, then the separate development/parity check, freeze and pilot commands
above, substituting local output directories. Do not open the locked test split.
