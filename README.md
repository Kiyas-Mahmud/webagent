# Failure-Aware Resilient Autonomous Web Agent

MSc thesis (AIUB). A backbone-agnostic 4-pillar web agent that **detects**,
**diagnoses**, **recovers from**, and **remembers** web-interaction failures —
trained on a failure-augmented dataset of 70,965 labeled steps (Mind2Web, 136 sites).

> Read `docs/AGENT.md` first. Full specs in `docs/PROJECT_SPECIFICATION.md`.

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

## Rules (non-negotiable — see docs/AGENT.md §5)

- Test-first every model: SMOKE -> MINI -> FULL -> EVAL -> CHECKPOINT. Never full-train unverified code.
- Order: Y1 -> A1-A5 -> Y3 -> Y7 -> Y4 -> Y2 -> Y6 -> Y5 (paid, last) -> baselines.
- Primary model chosen by validation results, not pre-fixed.
- FREE-first compute (Kaggle T4); only Y5 paid (Vast.ai A100), stop instance the moment it ends.
- Never split the shared `images/` folder. Don't synthetically fill missing action types.
