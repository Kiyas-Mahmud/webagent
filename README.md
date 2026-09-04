# Failure-Aware Resilient Autonomous Web Agent

MSc thesis (AIUB). A backbone-agnostic 4-pillar web agent that **detects**,
**diagnoses**, **recovers from**, and **remembers** web-interaction failures.
The earlier 70,965-step Mind2Web-derived corpus is retained as historical
project context; current training and evaluation are governed by the reviewed
Gold v2.8 artifacts and the canonical documents below.

> Read `docs/AGENT.md` first. Full training specs are in
> `docs/PROJECT_SPECIFICATION.md`. For the post-training browser experiment,
> `docs/TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md` is the canonical
> authority and supersedes conflicting runtime wording in older design files.
> Its active model-promotion gate compares exactly PC-01/PC-02/PC-03 at seed 42;
> the 19-model roadmap below is historical supporting/generalization scope.

## The idea

One shared encoder feeds one fused `[B, 768]` embedding to four task heads,
trained together with one combined weighted loss. The architecture is
**backbone-agnostic**: only the front-end encoder swaps across 19 models
(7 backbones + 5 ablations + 7 baselines); the 4 heads and the loss never change.

```
DUAL ENCODER (Y1,Y2,Y3,Y7): vision_enc + text_enc -> CROSS-ATTENTION -> [768] -> 4 heads
UNIFIED VLM  (Y4,Y5,Y6):    VLM(image+text) -> Adapter Linear(D->768)  -> [768] -> 4 heads
```

## Layout

```
docs/        design docs (AGENT, SPEC, ARCHITECTURE, TRAINING, IMPLEMENTATION)
configs/     one YAML per model (base.yaml + backbones/ ablations/ baselines/)
src/web_agent/
  data/      verify_dataset, WebAgentDataset, dataloader (5 mode filters)
  models/    encoders/ (swap) + fusion + adapter + heads (FIXED) + loss (FIXED) + model
  memory/    Pillar 4 retrieval index
  train/     smoke_test -> mini_train -> train, trainer
  eval/      metrics + evaluate (3 splits)
  baselines/ B1-B4
  utils/     seed, checkpoint, logging
scripts/     setup_data.py, run_model.py
notebooks/   kaggle_train.ipynb (thin: pip install -e . then call run_model)
results/     verification report, results_table.csv, tables/ figures/ curves/
tests/       shape + finite-loss tests
```

## Setup

```bash
pip install -e .
python scripts/setup_data.py          # link ../FinalData -> data/, sanity counts
python -m web_agent.data.verify_dataset
```

The browser stack is intentionally separate from the DGX training install.
Only on the eventual Table 2 campaign host, after the selected-checkpoint and
WebArena handoff inputs exist, install the declared extra and browser runtime:

```bash
python3 -m pip install -e '.[table2]'
python3 -m playwright install chromium
```

Do not interpret this installation step as permission to open locked tasks;
the frozen campaign preflight remains the access authority.

## Run a model (test-first, never skip stages)

```bash
python scripts/run_model.py --config configs/backbones/y1_siglip_roberta.yaml --stage smoke   # 16 rows
python scripts/run_model.py --config configs/backbones/y1_siglip_roberta.yaml --stage mini    # 5,000 rows
python scripts/run_model.py --config configs/backbones/y1_siglip_roberta.yaml --stage full    # 38,875 rows
python scripts/run_model.py --config configs/backbones/y1_siglip_roberta.yaml --stage eval    # 3 test splits
```

On Kaggle T4: open `notebooks/kaggle_train.ipynb`, which only does `pip install -e .`
and calls the same entry point.

## Current Gold-data improvement audit

Run `notebooks/kaggle_gold_existing_data_improvement.ipynb` on Kaggle CPU after
attaching `kiyasmahmud/web-gold-40k`. It streams the existing train/validation
data from the attached ZIP, creates invalid-bbox and weak-class reviewer queues,
keeps the locked test split unread, and never changes the source dataset. See
`docs/EXISTING_DATA_IMPROVEMENT_RUNBOOK.md` before applying any correction.
Reviewer A and Reviewer B then use the separate interactive
`notebooks/kaggle_gold_manual_review.ipynb`; their immutable logs are checked
with `scripts/reconcile_gold_reviews.py`. This targeted workflow does not touch
the main `notebooks/kaggle_gold.ipynb`.
After reconciliation passes,
`notebooks/kaggle_gold_review_overlay_validation.ipynb` verifies the approved
in-memory train/validation overlay against the mounted source before another
controlled mini.

## Current AIUB three-PC comparison

Use `notebooks/dgx_three_model_comparison.ipynb` to run Qwen2-VL-2B,
Qwen2.5-VL-7B, and InternVL3.5-8B-HF on separate 128 GB lab machines. Each
candidate has isolated artifacts, exact-batch `last.ckpt` resume, the same
10-epoch maximum, and original-validation-only selection. The two new
backbones must pass their own implementation compatibility and controlled 5k
mini gates before full training. See `docs/DGX_THREE_MODEL_COMPARISON.md`.

## Rules (non-negotiable — see docs/AGENT.md §5)

- Test-first every model: SMOKE -> MINI -> FULL -> EVAL -> CHECKPOINT. Never full-train unverified code.
- Order: Y1 -> A1-A5 -> Y3 -> Y7 -> Y4 -> Y2 -> Y6 -> Y5 (paid, last) -> baselines.
- Primary model chosen by validation results, not pre-fixed.
- FREE-first compute (Kaggle T4); only Y5 paid (Vast.ai A100), stop instance the moment it ends.
- Never split the shared `images/` folder. Don't synthetically fill missing action types.
