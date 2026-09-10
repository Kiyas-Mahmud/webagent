# Model Training Implementation Plan — One by One (Test First, Then Full)

> **Read with `PROJECT_SPECIFICATION.md` (v2).**
> **Golden rule:** never launch a full training run on unverified code. For EVERY model: run a tiny SMOKE TEST first, confirm the architecture works, then run the FULL training. Only after one model is fully proven do we move to the next.
>
> **Current authority correction (2026-09-04):** the seven-model/three-seed
> roadmap below is historical generalization scope. The active Table 2 model
> gate is exactly the seed-42 PC-01/PC-02/PC-03 validation-only comparison in
> `DGX_THREE_MODEL_COMPARISON.md`; no PC-01 seeds 43--44 are added.

---

## 0. THE CORE WORKFLOW (applies to every model)

```
For each model M:
  STEP A — SMOKE TEST   (tiny, ~10 min, prove it runs)
  STEP B — MINI TRAIN   (5,000 samples, ~30-60 min, prove it learns)
  STEP C — FULL TRAIN   (all 38,875, hours, the real run)
  STEP D — EVALUATE     (all 3 test splits, record metrics)
  STEP E — CHECKPOINT   (save best weights + log to results table)

Do NOT skip A or B. They cost minutes and save days.
Do NOT start model M+1 until model M passes STEP E.
```

### What each step proves
```
A SMOKE TEST  -> code runs, shapes align, loss is finite, no crash
B MINI TRAIN  -> loss goes DOWN, architecture actually learns, checkpoint saves/loads
C FULL TRAIN  -> real performance on full data
D EVALUATE    -> numbers for the paper
E CHECKPOINT  -> reproducible artifact saved
```

---

## 1. ORDER OF MODELS (free-first, smallest-first)

```
ORDER  MODEL                         WHERE        WHY THIS ORDER
1      Y1 SigLIP + RoBERTa           Kaggle FREE  primary candidate, default architecture
2      A1 Vision-Only                Kaggle FREE  ablation (reuse Y1 code)
3      A2 Text-Only                  Kaggle FREE  ablation
4      A3 Concat-Fusion              Kaggle FREE  ablation
5      A4 No-Memory                  Kaggle FREE  ablation
6      A5 No-Recovery                Kaggle FREE  ablation
7      Y3 CLIP + RoBERTa             Kaggle FREE  swap vision encoder only
8      Y7 Florence-2 + RoBERTa       Kaggle FREE  UI-grounding backbone
9      Y4 Qwen2.5-VL-0.5B            Kaggle FREE  first VLM (adapter path)
10     Y2 SigLIP-large + RoBERTa-L   Kaggle FREE  bigger dual encoder
11     Y6 InternVL2-2B (4-bit)       Kaggle FREE  second VLM
12     Y5 Qwen2.5-VL-3B (QLoRA)      Vast.ai $10  largest, LAST, paid
13     B1 Random                     CPU          baseline (no training)
14     B2 MindAct                    Kaggle FREE  baseline
15     B3 CLIP + MLP                 Kaggle FREE  baseline
16     B4 LayoutLMv3                 Kaggle FREE  baseline
       B5/B6/B7                      —            published numbers, no training
```

**Why this order:** Y1 first because it is the default architecture and primary candidate. Ablations next because they reuse almost all of Y1's code. Then backbone swaps from easiest to hardest. The one paid model (Y5) is dead last, only after the VLM adapter path is already proven free on Y4.

---

## 2. MODEL 1 — Y1 (SigLIP-base + RoBERTa-base) — THE FOUNDATION RUN

> This is the most important run. It proves the entire architecture. Every later model reuses this code.

### STEP A — Smoke test (~10 min)
```
[ ] Load 16 records only (batch of 16)
[ ] Build full model: SigLIP + RoBERTa + cross-attention + 4 heads
[ ] Run ONE forward pass
    CHECK: vision_emb shape = [16, 768]
    CHECK: text_emb shape   = [16, 768]
    CHECK: fused shape      = [16, 768]
    CHECK: all 7 head outputs have correct shapes
[ ] Compute combined loss -> CHECK: single finite scalar (not NaN)
[ ] loss.backward() -> CHECK: no error
[ ] optimizer.step() -> CHECK: one parameter value changed
PASS CRITERIA: no crash, finite loss, weights move
```

### STEP B — Mini train on 5,000 samples (~30-60 min)
```
[ ] Subsample 5,000 train rows (stratified by failure_type)
[ ] Train 3 epochs, encoders FROZEN, lr 1e-3
[ ] Log loss every 10 batches
    CHECK: training loss DROPS within first 50 batches
    CHECK: no NaN/inf anywhere
[ ] Run validation on 500 rows
    CHECK: a non-trivial Failure F1 appears (better than random ~0.4)
[ ] Save checkpoint to Kaggle output
[ ] Reload checkpoint in a fresh cell
    CHECK: reloaded model produces identical predictions
PASS CRITERIA: loss decreases, F1 > random, checkpoint round-trips
```

> **If STEP B fails, STOP. Fix the bug now. Do NOT run full training.**
> Common bugs: wrong label encoding, image normalization wrong, loss weights wrong sign, bbox mask missing.

### STEP C — Full train (all 38,875)
```
[ ] Phase 1: encoders frozen, fusion+heads trainable, lr 1e-3, 3 epochs
[ ] Phase 2: all trainable, encoder lr 1e-5, head lr 1e-4, 7 epochs
[ ] batch 32, AdamW, weight decay 0.01, cosine warmup 10%
[ ] grad clip 1.0, fp16, early stopping patience 3 on val_loss
[ ] checkpoint every 500 steps (so a dead session can resume)
[ ] Repeat for 3 seeds (42, 1, 7) — Y1 only, since it is the primary candidate
```

### STEP D — Evaluate
```
[ ] Build memory index from Y1 train embeddings (top-k cosine, simple)
[ ] Evaluate on test_task, test_website, test_domain separately
[ ] Record: Failure F1, Failure-Type Acc, Action Acc, Recovery SR
[ ] Record per-failure-type F1 (PERCEPTION / ACTION_MISMATCH / LOOP)
[ ] Record mean ± std across 3 seeds
TARGET: Failure Detection F1 > 70%
```

### STEP E — Checkpoint + log
```
[ ] Save best checkpoint (by val Failure F1) + config to storage
[ ] Add Y1 row to the results table
[ ] Save learning curves for the paper appendix
[ ] Back up to Google Drive / HuggingFace
```

**Y1 done. The architecture is proven. Now every later model just swaps a component.**

---

## 3. MODELS 2-6 — ABLATIONS (A1-A5)

> Reuse ALL of Y1's code. Each ablation removes or changes ONE thing. Same smoke-test-first workflow, but smoke test is fast since code is proven.

```
A1 Vision-Only:    delete text encoder + fusion; vision_emb -> heads
A2 Text-Only:      delete vision encoder + fusion; text_emb -> heads
A3 Concat-Fusion:  replace cross-attention with concat + Linear(1536->768)
A4 No-Memory:      remove memory head + memory loss terms
A5 No-Recovery:    remove recovery_strategy outputs + their loss terms
```

### Workflow for EACH ablation
```
STEP A smoke test (~5 min): forward + loss + backward on 16 rows -> no crash
STEP B mini train (~30 min): 5,000 rows, confirm loss drops
STEP C full train: all 38,875, 1 seed (ablations need 1 seed, not 3)
STEP D evaluate: all 3 splits, same metrics
STEP E log: add row to ablation table
```

**Expected story:** every ablation scores LOWER than full Y1.
- A1/A2 lower -> proves both modalities needed
- A3 lower than Y1 -> proves cross-attention beats concat (your novelty)
- A4 lower -> proves memory helps
- A5 lower -> proves recovery supervision helps

---

## 4. MODELS 7-12 — BACKBONE VARIANTS

> Same heads, same loss. Only the encoder changes. Two sub-paths:
> - **Dual-encoder** (Y3, Y7, Y2): swap vision and/or text encoder, keep cross-attention.
> - **Unified VLM** (Y4, Y6, Y5): use the adapter path (spec section 4.5), skip cross-attention.

### MODEL 7 — Y3 (CLIP + RoBERTa) — dual encoder
```
Change: swap SigLIP -> CLIP-ViT-L/14. Everything else identical to Y1.
STEP A smoke: forward/loss/backward on 16 rows
STEP B mini: 5,000 rows, loss drops, CLIP embeddings flow correctly
STEP C full: all data, 1 seed
STEP D eval: 3 splits
STEP E log: backbone table
PURPOSE: proves SigLIP > CLIP (or not) under identical training.
```

### MODEL 8 — Y7 (Florence-2 + RoBERTa)
```
Change: vision encoder = Florence-2-large.
Smoke-first workflow identical.
Florence-2 is strong at visual grounding -> watch bbox metrics.
```

### MODEL 9 — Y4 (Qwen2.5-VL-0.5B) — FIRST VLM, adapter path
```
NEW PATH: this is the first unified VLM. Extra care on STEP A.
STEP A smoke (CRITICAL):
  [ ] Load Qwen2.5-VL-0.5B
  [ ] Feed image+text together (VLM native format)
  [ ] Get pooled output [B, D]
  [ ] Adapter Linear(D -> 768) -> [B, 768]
  [ ] CHECK: cross-attention is SKIPPED (VLM already fused)
  [ ] 4 heads on adapted embedding -> CHECK shapes
  [ ] loss finite, backward works
STEP B mini: 5,000 rows -> CONFIRM the adapter path learns
  -> This validates the whole VLM path for Y5 and Y6 later.
STEP C full: all data, 1 seed (freeze VLM or LoRA, train adapter+heads)
STEP D/E: eval + log
PURPOSE: proves the architecture works on a VLM, cheaply and free.
```

### MODEL 10 — Y2 (SigLIP-large + RoBERTa-large) — dual encoder
```
Change: larger encoders, images at 384px.
Watch VRAM: reduce batch to 16, gradient accumulation 2 -> effective 32.
Smoke-first workflow identical.
```

### MODEL 11 — Y6 (InternVL2-2B, 4-bit) — VLM
```
Reuse the adapter path proven in Y4.
Load InternVL2-2B in 4-bit to fit Kaggle T4.
Enable gradient checkpointing.
Smoke-first, then mini, then full (1 seed).
```

### MODEL 12 — Y5 (Qwen2.5-VL-3B, QLoRA) — LAST, PAID
```
Only run AFTER Y4 and Y6 prove the VLM path works.
Platform: Vast.ai A100 40GB (~$0.35/hr, ~$10 total).
[ ] Smoke test FIRST on Vast.ai before committing hours
[ ] QLoRA: load_in_4bit, nf4, double quant, bf16 compute
[ ] LoRA rank 16, alpha 32, target q_proj/v_proj
[ ] Train adapter + heads + LoRA, gradient accumulation 8 -> effective 32
[ ] STOP THE INSTANCE the moment training ends
STEP D/E: eval + log
```

---

## 5. MODELS 13-16 — BASELINES

```
B1 Random:      no training. Predict majority class. Compute metrics directly.
B2 MindAct:     DeBERTa-v3 + Flan-T5. Smoke-first, mini, full, eval.
B3 CLIP + MLP:  CLIP-ViT-B/32 + simple MLP head. Smoke-first, mini, full, eval.
B4 LayoutLMv3:  LayoutLMv3-base. Smoke-first, mini, full, eval.
B5/B6/B7:       BacktrackAgent / SeeAct / OSCAR -> copy published numbers.
```

Baselines deliberately do NOT get the 4-pillar treatment — they are the comparison floor.

---

## 6. AFTER ALL MODELS — SELECT PRIMARY

```
[ ] Put all 19 models in the main results table
[ ] Compare validation Failure F1 + Recovery SR across Y1-Y7
[ ] The best-performing backbone = PRIMARY MODEL
[ ] Mark it as primary in the paper; report the rest as generalization evidence
[ ] Re-run the primary with 3 seeds if not already done
```

---

## 7. UNIVERSAL CHECKLIST (pin this — use for EVERY model)

```
BEFORE training:
  [ ] Seeds set (42)
  [ ] GPU confirmed (nvidia-smi)
  [ ] Correct DataLoader mode/filter selected
  [ ] Class weights set (CLICK 83.6%, LOOP 7.4%)

SMOKE TEST (16 rows):
  [ ] Forward runs, all shapes correct
  [ ] Loss is finite (not NaN/inf)
  [ ] Backward + optimizer step works
  [ ] One parameter changed

MINI TRAIN (5,000 rows):
  [ ] Loss drops in first 50 batches
  [ ] Val F1 > random
  [ ] Checkpoint saves AND reloads identically

FULL TRAIN:
  [ ] Two-phase schedule applied
  [ ] Checkpoint every 500 steps (resume-safe)
  [ ] Early stopping active

EVALUATE:
  [ ] All 3 test splits separately
  [ ] All primary + per-failure-type metrics recorded

DONE:
  [ ] Best checkpoint + config saved
  [ ] Row added to results table
  [ ] Curves saved
  [ ] Backed up off-platform
  [ ] (VLM/paid) instance stopped
```

---

## 8. RED FLAGS — STOP AND FIX IMMEDIATELY

```
Loss is NaN/inf            -> lr too high, or bad normalization, or log(0)
Loss does not drop         -> label encoding wrong, or labels/inputs misaligned
Val F1 = random forever    -> heads not connected to fused embedding correctly
Bbox loss explodes         -> bbox not normalized, or mask not applied
VLM OOM on Kaggle          -> use 4-bit/QLoRA, reduce batch, grad checkpointing
Checkpoint reload differs  -> not saving/loading all module states
Session dies mid-run       -> resume from last 500-step checkpoint
```

---

## 9. ONE-LINE SUMMARY (for the coding agent)

Train models one at a time in the order Y1 -> A1-A5 -> Y3 -> Y7 -> Y4 -> Y2 -> Y6 -> Y5 -> baselines. For EVERY model, first run a 16-row smoke test (prove it runs), then a 5,000-row mini train (prove it learns and the checkpoint round-trips), and only then the full 38,875-row training followed by evaluation on all three test splits. Y1 establishes and proves the whole architecture; every later model reuses that code and swaps only the backbone (dual-encoder configs keep cross-attention; VLM configs use the single adapter path and skip fusion). The one paid model (Y5 on Vast.ai) runs last, only after the free VLM path is already proven on Y4. After all models are evaluated, the best-performing backbone is selected as the primary model.
