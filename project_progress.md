# Project Progress Tracker

> **READ THIS FIRST** before planning or writing any code. Update it after every task.
> Companion to `docs/` (the specs) — this file is the live build log.

Project: Failure-Aware Resilient Autonomous Web Agent (MSc thesis).
Repo: https://github.com/Kiyas-Mahmud/webagent (branch `Code`).
Dataset: Kaggle `thesisdata` → `/kaggle/input/datasets/kiyasmahmud/thesisdata/FinalData`
(70,965 labeled steps; train 38,875 / val 16,070 / test 16,020).

## Workflow (how work flows)
- Author code in IDE (`e:/University/thesis/code`) → push to GitHub `Code`.
- Kaggle notebook `git clone`s the package + `sys.path` import (NOT `pip install -e .`
  — PEP 660 editable finder shadows sub-packages).
- Train/verify on **Kaggle T4** (free). Pull results back to IDE.
- Hybrid layout (plan C): stable core in `src/web_agent/`, the training loop + analysis
  live in the notebook so progress is visible.
- Commits: plain messages, **no Claude/AI mention**.

## Key decisions
- **Model = Qwen2-VL-2B-Instruct** (4-bit QLoRA). The spec's "Qwen2.5-VL-0.5B" does
  NOT exist (text-only Qwen2.5-0.5B, can't see images); 2B is the real smallest VLM
  (hidden D=1536). QLoRA for all VLM models for consistency.
- **VLM adapter path** (no cross-attention): VLM(image+text) → mean-pool → Adapter
  Linear(1536→768)+LN+Dropout → 5 heads. Backbone-agnostic (heads + loss never change).
- **Compute**: free Kaggle T4. Full 38,875-row QLoRA is too slow on T4 (~24h/epoch),
  so train on an 8k subsample now; full-data/headline number later on paid A100 (~$20).
  Whole-suite cost estimate: ~$60-160 smart-split (free small models, paid VLMs).
- T4 has **no bf16** → fp16 everywhere.

## Status by phase

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Setup | ✅ done | package scaffold, YAML config (deep-merge), labels.py, seed, checkpoint, logging |
| 1 Dataset verification | ✅ done + PASSED on Kaggle | all checks pass; 71.9% FAILURE, bbox 67,160, 0 leakage |
| 2 DataLoader | ✅ done + verified | WebAgentDataset, 5 mode filters, stratified subsample, VLM joint-processing branch + vlm_collate |
| 3 Model core (smoke+mini) | ✅ done + PASSED | adapter+heads learn; loss drops; checkpoint round-trips |
| 5 Full QLoRA training | 🟡 BUILT, running on Kaggle | 5 heads, 10-term loss, Trainer, eval, memory index — see below |
| 6 Memory index (Pillar 4) | ✅ built (cosine top-3) | retrieval used at eval |
| 7 Ablations | ⬜ not started | A1-A5 on chosen primary |
| 8 Other backbones | ⬜ not started | SigLIP/CLIP/Florence (free T4); InternVL2/Qwen2.5-VL-3B (paid) |
| 9 Baselines | ⬜ not started | B1 random, B2 MindAct, B3 CLIP+MLP, B4 LayoutLMv3 |
| 10 Evaluation/paper | ⬜ not started | 3-split tables, significance |

## Phase 5 — what's built (full QLoRA training)
- Config `configs/backbones/qwen2vl_2b.yaml`: 4-bit nf4, LoRA r8 q/v_proj, 10-term
  loss weights, 2 LR groups (lora 1e-4 / heads 1e-3), early-stop on val Failure-F1.
- `models/encoders/vlm.py`: peft LoRA via `prepare_model_for_kbit_training` +
  `get_peft_model`; grad checkpointing `use_reentrant=False`; mean-pool embedding.
- `models/adapter.py`: Linear(1536→768)+LayerNorm+Dropout(0.1).
- `models/heads.py`: 5 heads — Failure, Action, Memory, **RecoveryOutcome**; BCE heads
  emit LOGITS (BCE-with-logits, autocast-safe). `model.py` returns `fused` (contrastive).
- `data/dataset.py`: multi-step context (last 3 trajectory steps, grouped by
  original_task_id+pass); emits recovery_success + original_task_id; passes
  `mm_token_type_ids` (newer Qwen2-VL M-RoPE requirement).
- `data/dataloader.py`: `TaskPairBatchSampler` (co-batches a task's passes so
  contrastive pairs form); vlm_collate carries task_id + recovery_success + mm_token_type_ids.
- `models/loss.py`: 10-term loss — label smoothing 0.1, sklearn balanced weights,
  conf clip [0.05,0.95], differentiable soft-bin calibration surrogate, margin-0.5
  contrastive, recovery_outcome BCE.
- `train/trainer.py`: 2 param groups, cosine+warmup, fp16 GradScaler, grad accum,
  grad clip, **per-step loss+ETA logging**, early stop patience 3 on val Failure-F1,
  top-3 ckpts by F1, resume `last.ckpt` every 500 steps, metrics CSV.
- `eval/metrics.py`: Failure-F1 (primary), accuracies, true ECE, bbox MAE.
- `eval/evaluate.py`: per-split (test_task/website/domain) eval.
- `memory/index.py`: cosine top-3 retrieval.
- Notebook cells 1-9: install → verify → batch → visual → build → smoke → mini →
  checkpoint → full train (8k subsample, ~5-6h/epoch).

## First full run (8k subsample, Qwen2-VL-2B) — RAN CLEAN, but majority collapse
- Infra perfect: 4 epochs ~78min each, per-step ETA, early stop, top-3 ckpts, round-trip,
  full 3-split eval, CSV. ~5.21GB VRAM.
- **Failure-F1 0.837 = the FAILURE-majority baseline** (71.9% FAILURE → predict-all-FAILURE
  F1 = 0.836). F1 flat across epochs/splits → model collapsed to always-FAILURE. NOT real
  failure detection.
- failtype_acc ~0.28-0.39 (≈majority), recovery_acc ~0.30 (≈majority): also collapsed.
- action_acc ~1.00 (genuinely learned / strongly text-cued). ECE ~0.095 (ok). bbox_mae ~0.20 (weak).
- ROOT CAUSE: outcome loss was NOT class-weighted (only action+failtype were). 
- FIX APPLIED (commit pending): balanced weight on the OUTCOME loss too
  (`balanced_class_weights` now returns outcome weights; `CombinedLoss` takes
  `outcome_class_weights`). Added honest metrics: failure_macro_f1, outcome_bal_acc,
  outcome_mcc, success_recall — so collapse can't hide behind F1. Notebook cells 5/6 pass ow.
  - Re-run cell 9: watch `success_recall` and `outcome_mcc` > 0. If still ~0, bump LoRA
    rank 8→16 / more epochs / more rows.

## Second run (outcome class-weighted) — collapse FLIPPED, not fixed
- After outcome balancing: `success_recall=0.96, failure_f1=0.07, outcome_mcc≈0, bal_acc≈0.50`.
  Model now predicts SUCCESS (minority) for ~everything. **MCC stayed ≈0** the whole time →
  weighting only moved the decision bias and flipped WHICH class collapses. Diagnostic proof:
  the problem is the **representation/inputs the outcome head sees, not loss weighting**.

## Representation-level fixes (this round) — code-ready, validate on Kaggle smoke
Root analysis in `~/.claude/plans/you-are-proffesional-phd-gentle-dewdrop.md`.
- **Pooling (P0-A, primary):** `encoders/vlm.py` now config-selectable `backbone.pooling`
  = `last` (default) | `mean` | `attention`. Was uniform masked-MEAN over the full
  decoder sequence — dominated by hundreds of vision-patch tokens, blurring the causal
  summary where the outcome signal lives. `last` = last non-pad token; `attention` = tiny
  learned pool. (mean kept for the ablation.)
- **state_after (P0-B):** `data.use_state_after` (default false). When true, feed
  before+after screenshots (two images) — failure is often only visible post-action.
  `dataset._vlm_inputs` takes an image list; `image_grid_thw` now `[n_img,3]`; collate
  CATs it (single-image path byte-identical).
- **Contrastive (P1-A):** `loss.contrastive_mode` = `supervised` (default, SupCon on the
  outcome label — works in any batch with both classes) | `task_pair` (old, sparse). The
  old term was ~dead (needed same-task SUCCESS+FAILURE in one micro-batch).
- **Outcome weighting (P1-B):** outcome CE label_smoothing forced to 0; outcome weights
  now `capped` (ratio ≤1.5: SUCCESS 1.5 / FAILURE 1.0) via
  `class_weights.balanced_class_weights(outcome_scheme=...)`, not raw `balanced` (2.56× over-swing).
  `sqrt` scheme also available.
- **recovery_outcome (P2-B):** BCE now uses `pos_weight` (`binary_pos_weight`, ≈2.56);
  loss weight 0.02→0.05.
- **Capacity (P2-A):** LoRA targets q/v → q,k,v,o,gate,up,down; rank 8→16 (2B), alpha 32.
- **Config completeness:** `qwen25vl_3b.yaml` got the full 10-term loss + lr_lora/heads +
  train block (it inherited base's 7-term and would have KeyError'd if run).
- **Notebook:** cell 5 computes capped outcome weights + recovery pos_weight + a
  `DIAG_OUTCOME_ONLY` toggle (zeros the other 9 terms) for the D1/D2/D3 ceiling probe.

## NEXT
- Run the diagnostic protocol FIRST (cell 5 `DIAG_OUTCOME_ONLY=True`): D1 mean-pool,
  D2 last-pool, D3 +state_after. Judge by **val outcome MCC / balanced-acc** (NOT failure_f1).
  Whichever first lifts MCC clearly above 0 is the real fix.
- Then full 10-term run with the winning config. Confirm SupCon `contrastive non-zero` in smoke.
- Then: pooling + input ablations (paper contributions), baselines (majority/random rows),
  other backbones, eval tables.

## Gotchas learned (don't re-hit)
- JSON is UTF-8 (Windows cp1252 fails). bbox keys are `width`/`height` (not w/h).
- `.gitignore data/` once swallowed `src/web_agent/data/` — anchor ignores with `/`.
- Kaggle stale-kernel: after a package fix, re-run cell 1 (purges sys.modules) AND every
  cell that built a reused object (loader, model, loss). Or Restart & Run All.
- `pip install -e .` editable finder shadows subpackages → use sys.path.
- `F.binary_cross_entropy` unsafe under fp16 autocast → use `_with_logits` (heads emit logits).
- Qwen2-VL (newer transformers) needs `mm_token_type_ids` passed through.

## Latest commit
`590ac48` — feasible full-train on T4 (per-step ETA, 8k subsample, warning fixes).
