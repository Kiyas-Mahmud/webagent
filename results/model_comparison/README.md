# DGX model-comparison results (Gold v2.8, seed 42)

Lightweight artifacts copied verbatim from the lab run directory
`/home/aiub/kiyas/webagent_comparison` on 2026-09-13. Nothing here was regenerated,
edited, or re-serialized; only the directory nesting was flattened
(`outputs/model_comparison/<run>` -> `runs/<run>`).

Run provenance: `git_commit` `628c1fa`, torch 2.13.0+cu130, transformers 4.57.6,
NVIDIA GB10 (130.7 GB) - see `environment.json`.

## Layout

| Path | Contents |
| --- | --- |
| `environment.json` | Python/torch/transformers/GPU versions and the run's git commit |
| `preflight/` | `multisource_audit.json`, `supplement_validation.json` |
| `runs/internvl35_8b_gold_v2_8_dgx/seed_42/` | InternVL3.5-8B-HF candidate: run contract, compatibility report, `mini/` and `full/` stages |
| `runs/qwen2vl_2b_gold_v2_8_dgx/seed_42/` | Qwen2-VL-2B candidate: run contract and compatibility report only (no mini/full stage artifacts exist in the run directory) |

Each stage directory holds `report.json`, `epoch_metrics.csv`, `diagnostics.json`,
and `source_validation.csv`.

## Headline numbers (InternVL3.5-8B-HF, seed 42)

| Stage | Status | Selection rule | Selected epoch | outcome_mcc |
| --- | --- | --- | --- | --- |
| mini (5 controlled epochs) | PASS | `all_gates_then_outcome_mcc` | 2 | 0.6129 |
| full | PASS | `all_gates_then_outcome_mcc` | 0 | 0.6410 |

Full stage: 24,107 train rows / 7,861 original-validation rows, effective batch 32,
requested max 10 epochs, early-stopped after epochs 0-3, peak GPU 15.69 GB.
`test_rows_read: 0` in every artifact - the held-out test split was never opened.
Selected full checkpoint sha256: `35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb`.

## Checkpoints (Git LFS)

All 11 `.ckpt` files (~616 MB each, ~6.7 GB total) are committed through Git LFS under
`runs/internvl35_8b_gold_v2_8_dgx/seed_42/<stage>/checkpoints/`, matching the convention
already used for the Qwen2-VL-2B run on `Code`. The tracking rule lives in the
repository-root `.gitattributes`.

To fetch them you need `git-lfs` installed (`git lfs install`, then `git lfs pull`);
a plain clone without LFS yields ~130-byte pointer files instead of weights.

The LFS object id of each file is its SHA-256. For the gate-selected full checkpoint,
`best_e0_outcome-mcc0.641.ckpt`, that id is
`35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb` - identical to the
`selected_checkpoint_sha256` recorded independently in `full/report.json`.

Note that the absolute paths recorded inside `report.json` / `epoch_metrics.csv` still
point at the original lab-machine location
(`/home/aiub/kiyas/webagent_comparison/outputs/model_comparison/...`), not at these
committed copies.

## Deliberately not committed

- **`hf_cache/`** - 21 GB of downloaded backbone weights, reproducible from the
  `model_revision` pins in each `run_contract.json`.
