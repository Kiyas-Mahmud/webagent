# AGENT.md — Project Context & Operating Rules

> **Read this first, every session.** Then read the four reference files listed below before writing any code. This file is the entry point; the four files are the full detail.

> **Table 2 authority:** for post-training browser runtime, causal pillar
> ordering, E0-E3 systems, recovery budgets, frozen memory, and evaluation,
> also read `TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md`. That document
> supersedes conflicting legacy runtime text here or in the four historical
> reference files; it does not modify the registered training experiment.

---

## 1. PROJECT IN ONE PARAGRAPH

We are building a **failure-aware autonomous web agent** for an MSc thesis (AIUB, graduation Dec 2026) targeting a **Q1 journal + thesis competition**. Web agents today are trained only on successful demonstrations, so they act but cannot detect, diagnose, recover from, or remember failures. Earlier project iterations used a purpose-built **70,965-step failure-augmented corpus** derived from Mind2Web; that count is historical context, not the authority for current experiments. Current registered training and evaluation use the reviewed Gold v2.8 artifacts and their run-specific manifests. The model is a **unified 4-pillar architecture** with one shared encoder feeding four task heads, trained with one combined weighted loss. The architecture is **backbone-agnostic**: it works with dual encoders (SigLIP + RoBERTa via cross-attention fusion) and with unified VLMs (Qwen2.5-VL, InternVL2 via a single adapter layer). We train the same architecture across candidate backbones and select the **best-performing backbone as the primary model by validation results** — it is NOT pre-fixed.

---

## 2. THE FOUR PILLARS (what the model learns)

```
Pillar 1  Failure-Aware Resilience    -> detect failure, type, confidence, recovery
Pillar 2  Unified Multimodal Decision -> fuse vision + text into one embedding
Pillar 3  Adaptive Multi-Tool Action  -> choose action type + bbox
Pillar 4  Memory-Driven Planning      -> store experience + retrieve past recoveries
```

All four share ONE fused embedding and train together with ONE combined loss.

---

## 3. TRUE CONTRIBUTION (state this correctly)

The novelty is the **unified 4-pillar framework + the failure dataset**, and the fact that it is **backbone-agnostic**. Cross-attention fusion is only ONE implementation of Pillar 2 for dual encoders — it is a detail, NOT the headline. Do not describe cross-attention as "the novelty." The headline is: one framework that learns detection + recovery + memory together and improves any backbone.

---

## 4. READ THE CORE FILES BEFORE CODING

```
PROJECT_SPECIFICATION.md   THE WHAT
   dataset stats, full schema, label mappings, model list,
   architecture specs, combined loss, constraints, what-not-to-do

ARCHITECTURE_DESIGN.md     HOW IT CONNECTS
   how the 4 pillars connect, each head's inputs/outputs/shapes,
   dual-encoder path vs VLM adapter path, full data flow

MODEL_TRAINING_PLAN.md     TRAINING ORDER (test-first)
   smoke test -> mini train -> full train, one model at a time,
   exact order of all 19 models

IMPLEMENTATION_PLAN.md     FULL TIMELINE
   phase-by-phase from setup to paper, requirements,
   techniques, exit criteria, risk register

TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md
   canonical post-training E0-E3 browser runtime and Table 2 evaluation
```

Always reconcile any decision against these files. If something here conflicts with them, ask the user before proceeding.

---

## 5. NON-NEGOTIABLE OPERATING RULES

### 5.1 Test-first for EVERY model (most important rule)
```
For each model, in order:
  A. SMOKE TEST  (16 rows)    -> code runs, shapes ok, loss finite, backward works
  B. MINI TRAIN  (5,000 rows) -> loss drops, val F1 > random, checkpoint round-trips
  C. FULL TRAIN  (38,875)     -> only after A and B pass
  D. EVALUATE    (3 test splits)
  E. CHECKPOINT  (save best + log to results table)

NEVER run full training on unverified code.
NEVER start the next model until the current one passes E.
```

### 5.2 Model order (free-first, smallest-first)

> **BUILD ORDER OVERRIDE (v3, current): Qwen-first.** The foundation/first run is
> now **Qwen2.5-VL-3B-Instruct in 4-bit (QLoRA)** via the VLM adapter path, not Y1.
> This is a REORDER, not a replacement — SigLIP+RoBERTa and the other dual-encoder
> backbones are still trained afterwards, so the backbone-agnostic comparison (the
> paper's novelty) is preserved. Valid because the 4 heads + combined loss never
> change (5.4). Note: "Qwen2.5-VL-0.5B" in older docs does not exist — 3B is the
> smallest Qwen2.5-VL. New current order:
> ```
> Qwen2.5-VL-3B (4-bit) -> A1..A5 ablations -> SigLIP+RoBERTa (Y1) -> Y3 -> Y7
> -> Y2 -> Y6 -> Qwen2.5-VL-3B full QLoRA / larger -> baselines B1..B4
> ```

Original (v2) order, kept for reference:
```
Y1 (SigLIP+RoBERTa) -> A1..A5 ablations -> Y3 -> Y7 -> Y4 (first VLM)
-> Y2 -> Y6 -> Y5 (paid, LAST) -> baselines B1..B4
```
The first model built proves the whole architecture (heads + loss + training loop);
every later model reuses its code and swaps only the backbone front-end.

### 5.3 Primary model is chosen by RESULTS
Do not hardcode a primary model. The former seven-backbone rule describes the
historical broad program; the current registered Table 2 promotion gate
compares PC-01, PC-02, and PC-03 at seed 42 using validation evidence only.
Only that registered comparison may promote the final Table 2 backbone.

### 5.4 Backbone-agnostic implementation
```
Dual encoder (Y1,Y2,Y3,Y7): vision_enc + text_enc -> CROSS-ATTENTION -> [768] -> 4 heads
Unified VLM  (Y4,Y5,Y6):     VLM(image+text) -> Adapter Linear(D->768) -> [768] -> 4 heads
The 4 heads and the combined loss NEVER change. Only the front-end changes.
```

### 5.5 Compute = FREE first
```
Kaggle Free T4 16GB for everything possible (Y1,Y2,Y3,Y4,Y6,Y7,ablations,baselines)
Vast.ai A100 (~$10) ONLY for Y5 Qwen2.5-VL-3B, and LAST
ALWAYS stop the paid instance the moment training ends
Checkpoint every 500 steps so a dead session can resume
```

---

## 6. LEGACY DATASET QUICK FACTS (HISTORICAL)

The figures below describe the earlier 70,965-row synthetic corpus only. They
are retained to explain project history and must not be used as current Gold
v2.8 corpus authority. Use the reviewed-Gold manifests and the registered DGX
run documents for current training claims.

```
Files:   split_train.json (38,875), split_val.json (~16k), split_test.json (~16k)
Images:  ONE shared images/ folder — never split it
Total:   70,965 steps, 2,022 tasks, 136 websites
Outcome: 71.9% FAILURE / 28.1% SUCCESS
Failure types: ACTION_MISMATCH 39.1%, PERCEPTION_ERROR 25.4%, LOOP 7.4%, NONE 28.1%
Actions: CLICK 83.6%, TYPE 12.5%, SELECT 3.9% (SCROLL/NAVIGATE ABSENT — document, do not fill)
Confidence gap: FAILURE 0.41 vs SUCCESS 0.83
Borrowed images: 14.4% (borrowed_image=True) — exclude from VISUAL training/eval only
bbox present: 67,160 rows (filter for bbox regression)
```

Label maps and the full 23-field schema are in PROJECT_SPECIFICATION.md section 3.

---

## 7. COMBINED LOSS (do not change without ablation)

```
L_total = 0.25*L_outcome + 0.20*L_failtype + 0.18*L_action
        + 0.12*L_bbox(masked) + 0.10*L_memory + 0.10*L_recovery
        + 0.05*L_confidence
class-weight action_type (CLICK dominates) and failure_type (LOOP rare)
bbox loss masked to rows where bbox exists
```

---

## 8. WHAT TO DO RIGHT NOW (first session)

```
1. Read the four reference files.
2. Write dataset verification script -> confirm counts, splits, label logic.
3. Write Dataset class + DataLoader (mode filters in spec 6.3).
4. VERIFY one batch: shapes correct, images open, labels aligned.
5. Build Y1 model (SigLIP + RoBERTa + cross-attention + 4 heads) + combined loss.
6. SMOKE TEST on 16 rows.
7. MINI TRAIN on 5,000 rows -> confirm loss drops + checkpoint round-trips.
8. STOP and report results before launching full Y1 training.
```

---

## 9. RED FLAGS — STOP AND FIX

```
loss NaN/inf            -> lr too high / bad normalization / log(0)
loss not dropping       -> label encoding wrong / inputs-labels misaligned
val F1 = random forever -> heads not wired to fused embedding
bbox loss explodes      -> bbox not normalized / mask not applied
VLM OOM on Kaggle       -> 4-bit/QLoRA, smaller batch, grad checkpointing
checkpoint reload differs-> not saving/loading all module states
```

---

## 10. CORE PRINCIPLE

> A simple system evaluated rigorously beats a complex system evaluated poorly.
> Build 40%, evaluate 60%. Be honest about every limitation in the paper.
> The dataset is the strongest asset — protect its integrity above all.
