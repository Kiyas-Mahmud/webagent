# DGX Spark Full-Training Runbook

> For the current three-PC backbone comparison, use
> `notebooks/dgx_three_model_comparison.ipynb` and
> `docs/DGX_THREE_MODEL_COMPARISON.md`. This document remains the detailed
> single-model Qwen2-VL-2B recovery/resume reference.

## 1. Purpose

Use `notebooks/dgx_gold_full_training.ipynb` for the first full reviewed-Gold
training seed. The notebook is designed for an NVIDIA DGX Spark or another
single CUDA GPU with sufficient memory and persistent storage.

The first authorized run is:

- model: `Qwen/Qwen2-VL-2B-Instruct`;
- configuration: `configs/backbones/qwen2vl_2b_gold_v2_8.yaml`;
- training source: all 23,499 reviewed original train rows plus all 608 accepted
  RETRY/ABORT supplement train rows (24,107 combined rows);
- checkpoint selection: all 7,861 original-Gold validation rows only;
- supplement validation: all 194 supplement validation rows, reported
  separately and never used for checkpoint selection;
- locked test: not mounted if possible and never read by this notebook;
- seed: 42;
- maximum epochs: 10, with registered early stopping;
- checkpoint interval: every 50 optimizer steps and after every completed
  epoch.

The v2.8 mini completed epochs 0-3 and epoch 3 passed all eight registered
quality gates. Epoch 4 was interrupted by the Kaggle T4 quota. This is an
explicit compute-limit protocol deviation, not a formal five-epoch completion.
The full-run contract records that limitation.

## 2. Data-source options

The notebook supports two explicit modes:

- `DATA_SOURCE = "local"` for datasets already downloaded from Kaggle and
  extracted on the DGX; and
- `DATA_SOURCE = "huggingface"` for immutable Hugging Face snapshots.

For the AIUB lab checkout, the default local paths are:

```text
/home/aiub/kiyas/webagent
/home/aiub/kiyas/webagent_full/data/original
/home/aiub/kiyas/webagent_full/data/supplement
```

Local mode does not require `HF_TOKEN` unless completed artifacts are uploaded
to Hugging Face. The notebook searches recursively below the two data folders,
requires exactly one compatible extracted root for each source, and reports
any ZIP files when extraction is still required. A locally present
`split_test.json` is reported but never opened by the full-training pipeline.

### 2.1 Required Hugging Face sources

Create two private Hugging Face dataset repositories. Do not put an access
token inside the notebook or Git repository. Export it in the lab environment
as `HF_TOKEN`.

#### 2.1.1 Original reviewed Gold repository

Recommended repository ID:

```text
YOUR_HF_USERNAME/web-gold-40k
```

Required structure anywhere under the repository snapshot:

```text
final_data_set_40k/
  split_train.json
  split_val.json
  images/
    ... every image referenced by split_train.json and split_val.json ...
```

`split_test.json` is not required for training and should preferably remain in
a separately controlled locked-test repository. If it is present in the same
repository, the notebook excludes that JSON from download and never calls the
test loader.

The original dataset must remain extracted. Do not upload only a ZIP: training
needs random access to individual images, and keeping both a ZIP and its
extracted copy doubles storage use.

#### 2.1.2 Accepted RETRY/ABORT supplement repository

Recommended repository ID:

```text
YOUR_HF_USERNAME/gold-40k-retry
```

Required accepted package structure:

```text
web_gold_40k_retry_abort_supplement_v2_kaggle/
  SHA256SUMS.txt
  data/
    supplement_train.json
    supplement_val.json
  images/
    ... all images referenced by the two supplement JSON files ...
```

Upload the complete accepted package, including every file listed by
`SHA256SUMS.txt`. The notebook verifies the manifest, the expected 608/194 row
counts, all referenced images, identity separation, direct causal transitions,
loss masks and original/supplement overlap before training.

### 2.2 Model sources downloaded automatically

The current comparison notebook downloads model/processor files from these
public model repositories when their registered candidate is selected:

```text
Qwen/Qwen2-VL-2B-Instruct
Qwen/Qwen2.5-VL-7B-Instruct
OpenGVLab/InternVL3_5-8B-HF
```

The 7B and 8B configurations are code-ready for the same causal Gold
architecture, but each full run is blocked until its own compatibility and
controlled mini gates pass. Neither inherits the 2B model's authorization.

## 3. Upload the datasets with the Hugging Face API

Run this outside the training notebook after setting `HF_TOKEN`:

```python
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])

api.create_repo(
    repo_id="YOUR_HF_USERNAME/web-gold-40k",
    repo_type="dataset",
    private=True,
    exist_ok=True,
)
api.upload_folder(
    repo_id="YOUR_HF_USERNAME/web-gold-40k",
    repo_type="dataset",
    folder_path="/absolute/path/to/final_data_set_40k",
)

api.create_repo(
    repo_id="YOUR_HF_USERNAME/gold-40k-retry",
    repo_type="dataset",
    private=True,
    exist_ok=True,
)
api.upload_folder(
    repo_id="YOUR_HF_USERNAME/gold-40k-retry",
    repo_type="dataset",
    folder_path=(
        "/absolute/path/to/"
        "web_gold_40k_retry_abort_supplement_v2_kaggle"
    ),
)
```

After upload, pin the immutable dataset revision (commit SHA) in the notebook
instead of using a moving `main` revision.

## 4. Persistent lab directories

Set `WORKSPACE_ROOT` in the notebook to storage that survives a notebook/kernel
restart. The default is `/workspace/webagent_full`.

```text
/workspace/webagent_full/
  repo/                         cloned GitHub source
  data/                         Hugging Face snapshots
  hf_cache/                     model/dataset cache
  outputs/
    qwen2vl_2b_gold_v2_8/
      seed_42/
        checkpoints/
          Y_QWEN2VL_2B_GOLD_V2_8_FULL_SEED42/
            last.ckpt
            best_e0_....ckpt
            best_e1_....ckpt
            ...
        epoch_metrics.csv
        diagnostics.json
        full_report.json
        resolved_config.json          # post-run export of selected checkpoint['config']
        source_validation.csv
        run_contract.json
```

`resolved_config.json` is a post-training handoff artifact, not a reason to
alter or restart an active run. Export the selected checkpoint's saved full
`config` mapping as canonical JSON after validation-only selection; do not
reconstruct it from a later checkout. The Table 2 handoff independently
requires that mapping to equal the registered fully resolved candidate config
and records both the file-byte and canonical-mapping SHA-256 identities.

Do not place `WORKSPACE_ROOT` in an ephemeral `/tmp` directory.

## 5. Resume behavior

`last.ckpt` contains:

- LoRA, shared adapter, task adapters and every task head;
- learned uncertainty/loss state;
- optimizer, scheduler and precision-scaler state;
- Python, NumPy, CPU and CUDA RNG states;
- completed metric/diagnostic history;
- early-stopping state and retained checkpoint map;
- current epoch and exact next physical batch;
- partial current-epoch loss accumulators.

Checkpoint writes are atomic: a temporary file is completed before it replaces
`last.ckpt`, so an interruption during serialization does not destroy the last
valid resume point.

With `AUTO_RESUME = True`, rerunning the notebook detects `last.ckpt`, validates
its training signature and continues from its exact next batch. It does not
combine independent runs. If the completed `full_report.json` already exists,
the notebook refuses to start training again.

Keep the output directory unchanged between interruptions. If moving a run to
another machine, copy the complete seed directory, not only `last.ckpt`, so the
previous epoch checkpoints remain available for validation-only selection.

## 6. CSV and report artifacts

`epoch_metrics.csv` contains one row for every completed epoch and identifies
the selected epoch. It includes the honest MCC, macro-F1, balanced accuracy,
recovery, memory, grounding and calibration metrics.

`source_validation.csv` contains two rows:

1. original Gold validation, which selects the checkpoint;
2. RETRY/ABORT supplement validation, which is diagnostic only.

`full_report.json` is the authoritative machine-readable artifact. It records
the code/config/data/checkpoint hashes, row counts, seed, precision, resume
audit, quality gates, bbox denominators, transition reports and zero test reads.

## 7. Model queue and scientific status

| Model | Gold implementation | Full-run status |
| --- | --- | --- |
| Qwen2-VL-2B | Implemented and mini-gate evidence available | Run first |
| Qwen2.5-VL-7B | Official HF VLM contract implemented | Compatibility and mini required before full |
| InternVL3.5-8B-HF | Official HF VLM contract implemented | Compatibility and mini required before full |
| SigLIP + RoBERTa (Y1/Y2) | Dual-encoder path is still `NotImplementedError` | Blocked |
| CLIP + RoBERTa (Y3) | Dual-encoder path is still `NotImplementedError` | Blocked |
| Florence-2 + RoBERTa (Y7) | Encoder/fusion path is not implemented | Blocked |
| InternVL2-2B (historical Y6 plan) | Superseded in the current three-PC screen by InternVL3.5-8B-HF | Not in this run |
| A1-A5 ablations | Separate controlled implementations are absent | Blocked |
| B1-B4 baselines | Separate baseline runners are incomplete | Blocked |

The notebook contains this registry and refuses an unauthorized full run. Add
each later model only after its implementation, smoke, mini, causal checks and
checkpoint round-trip pass. This prevents a model name in a table from being
mistaken for completed experimental evidence.

## 8. Run order

1. For local mode, download and extract both Kaggle datasets below the two
   configured local search roots. For Hugging Face mode, upload both dataset
   repositories, record their commit SHAs and set `HF_TOKEN`.
2. Open `notebooks/dgx_gold_full_training.ipynb`.
3. Edit only the configuration cell. Keep `DATA_SOURCE = "local"` for the AIUB
   Kaggle downloads, or select `"huggingface"` and set repository revisions.
4. Keep `ACTIVE_MODEL_ID = "qwen2vl_2b_gold_v2_8"` for the first run.
5. Run all cells.
6. If interrupted, rerun all cells with the same settings; auto-resume will use
   `last.ckpt`.
7. After completion, preserve the entire seed directory and review the selected
   validation checkpoint before opening the locked test.
