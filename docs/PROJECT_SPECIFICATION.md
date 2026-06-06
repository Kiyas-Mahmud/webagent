# Failure-Aware Resilient Autonomous Web Agent — Complete Project Specification (v2)

> **Read this entire document before writing any code.**
> This is the single source of truth for the project. Every architectural decision, dataset detail, model configuration, and training step is defined here.
>
> **v2 changes:** (1) The architecture is backbone-agnostic — the primary model is chosen by results, NOT pre-fixed. (2) Compute plan is FREE-first (Kaggle T4), so VLM sizes are reduced to fit free hardware. (3) Model list updated accordingly.

---

## 1. PROJECT OVERVIEW

### 1.1 What we are building

A failure-aware web agent that does four things current web agents cannot do together:

1. **Detect** when a web interaction step has failed
2. **Diagnose** what type of failure occurred
3. **Recover** by selecting an appropriate corrective strategy
4. **Remember** past failures to avoid repeating them

All four capabilities are trained on **one dataset**, in **one training loop**, with **one combined loss function**, sharing **one encoder backbone**.

### 1.2 Why it matters

Every existing web agent (SkillWeaver, OSCAR, BacktrackAgent, Devil's Advocate, ReUseIt, Multimodal Auto-Validation, WebSuite, Pipeline-Failures) solves only one piece of the failure-handling problem, and most rely on expensive proprietary APIs (GPT-4o, GPT-4V). Our system combines all pieces into one trained architecture using free open-source models that run offline after training.

### 1.3 The core idea (READ THIS FIRST)

The contribution is the **architecture + dataset + methodology**, NOT the backbone size.

- Our architecture (cross-attention fusion + 4 task heads + combined loss) is **backbone-agnostic**. ANY vision/text encoder or vision-language model can be plugged in.
- We train the SAME architecture with MULTIPLE backbones.
- We look at the results.
- **The best-performing backbone becomes the primary model** — chosen by data, not assumed in advance.

This is a stronger and more honest research story than pre-selecting one model.

### 1.4 Two end goals

| Goal | Description | Timeline |
|------|-------------|----------|
| **Q1 Paper** | Publish in a Q1 journal + win thesis competition | Month 3 |
| **Production Agent** | A real browser agent with LLM planner support | After paper |

### 1.5 Hard constraints

- **Graduation deadline:** December 2026
- **Paper submission target:** End of Month 3
- **Compute:** FREE first — Kaggle T4 16GB (30 GPU-hrs/week). Minimal paid GPU (Vast.ai ~$10) only for the one model that does not fit free.
- **Storage:** Kaggle Datasets (free), HuggingFace (free), Google Drive 15GB (free)
- **Team:** 2 researchers
- **Language:** Python, PyTorch

---

## 2. THE FOUR PILLARS

The architecture is organized around four research pillars. Each pillar maps to specific dataset fields and specific model components.

### Pillar 1 — Failure-Aware Resilient Agent
**Job:** Detect failures, classify failure type, predict confidence, select recovery strategy.
**Model component:** Failure Head (4 outputs)
**Trained on:** `execution_outcome`, `failure_type`, `failure_confidence`, `recovery_strategy`, `agent_confidence_before`

### Pillar 2 — Unified Multimodal Decision Architecture
**Job:** Combine vision and text so they inform each other before any decision. **This is the architectural novelty.**
**Model component:** Cross-Attention Fusion layer (for dual-encoder configs). For unified VLM backbones, a single adapter layer replaces fusion (see 4.5).
**Trained on:** `state_before`, `task_description`, `action_target_desc`, `website_domain`, `visual_diff_score`, `action_target_bbox`

### Pillar 3 — Adaptive Multi-Tool Orchestration
**Job:** Select the correct action type (CLICK/TYPE/SELECT/SCROLL/NAVIGATE) and target location. Overridden by recovery when failure detected.
**Model component:** Action Head (action type + bbox)
**Trained on:** `action_type`, `action_coordinates`, `action_target_bbox`

### Pillar 4 — Memory-Driven Corrective Planning
**Job:** Remember which experiences to store, retrieve similar past failures, guide recovery using proven strategies.
**Model component:** Memory Head + retrieval index (simple top-k cosine first; FAISS later)
**Trained on:** `memory_update_flag`, `recovery_strategy`, `recovery_success`, `original_task_id`, `pass`, `reflection_text`

---

## 3. THE DATASET

### 3.1 Files

| File | Records | Purpose |
|------|---------|---------|
| `FINAL_Trajectories_ENRICHED.json` | 70,965 | Complete dataset (backup) |
| `split_train.json` | 38,875 | Training (all train-split tasks) |
| `split_val.json` | ~16,000 | Validation (50% of test_ tasks) |
| `split_test.json` | ~16,000 | Final evaluation (50% of test_ tasks) |
| `images/` | 131,721 images | Shared by all splits — NOT split |

**Critical:** All three split JSONs reference the SAME `images/` folder. Do not split images.

### 3.2 Dataset statistics

```
Total steps:          70,965
Unique tasks:         2,022
Unique websites:      136
Screenshots:          131,721 (256px width, 60% JPEG)
Augmentation passes:  5 (14,193 steps each — balanced)
Schema fields:        23 + borrowed_image flag

Execution outcome:
  FAILURE:  51,036 (71.9%)
  SUCCESS:  19,929 (28.1%)

Failure types:
  ACTION_MISMATCH:   27,738 (39.1%)  [WRONG_OPERATION + MISCLICK]
  PERCEPTION_ERROR:  18,043 (25.4%)
  LOOP_DETECTED:      5,255 (7.4%)
  NONE (success):    19,929 (28.1%)

Action types:
  CLICK:   59,335 (83.6%)
  TYPE:     8,870 (12.5%)
  SELECT:   2,760 (3.9%)
  SCROLL:   absent in source — see 3.5
  NAVIGATE: absent in source — see 3.5

Recovery strategies:
  RETRY:               22,975 (32.4%)
  NONE:                19,929 (28.1%)
  REPLAN:              15,268 (21.5%)
  BACKTRACK:           12,470 (17.6%)
  ALTERNATIVE_TARGET:     323 (0.5%)

Confidence calibration gap: 0.42
  FAILURE avg confidence: 0.41
  SUCCESS avg confidence: 0.83

Image integrity:
  Real screenshots:  60,775 (85.6%)
  Borrowed images:   10,190 (14.4%, flagged borrowed_image=True)
  Placeholder/empty: 0
```

### 3.3 Schema — all 23 fields

| Field | Type | Example | Used by |
|-------|------|---------|---------|
| `task_id` | String | `train_pass1_401c4e6f_0` | All |
| `task_description` | String | `rent a car in Brooklyn from April 9` | P2, P3 |
| `website_domain` | String | `budget` | P2, P3 |
| `original_task_id` | String | `401c4e6f-6b0b-...` | All, P4 |
| `split` | String | `train` / `test_domain` / `test_task` / `test_website` | Eval |
| `pass` | String | `pass1` ... `pass5` | P4 |
| `state_before` | Path | `images/.../step_0000_before.jpg` | P2 |
| `state_after` | Path | `images/.../step_0000_after.jpg` | P2 |
| `visual_diff_score` | Float [0-1] | `0.013` | P1, P2 |
| `action_type` | Categorical | `CLICK` / `TYPE` / `SELECT` | P3 |
| `action_target_desc` | String | `large button or link` | P2, P3 |
| `action_coordinates` | List[Float] | `[164.0, 647.89]` | P2, P3 |
| `action_target_bbox` | Dict/None | `{x:144, y:628, w:39, h:38}` | P2 |
| `execution_outcome` | Categorical | `SUCCESS` / `FAILURE` | P1 |
| `failure_type` | Categorical | `PERCEPTION_ERROR` / etc | P1 |
| `failure_confidence` | Float [0-1] | `0.786` | P1 |
| `is_augmented` | Boolean | `true` | Meta |
| `injection_type` | Cat/None | `TARGET_MISSING` / etc | Meta |
| `recovery_strategy` | Categorical | `RETRY` / `REPLAN` / etc | P1, P4 |
| `recovery_success` | Boolean | `true` | P1, P4 |
| `agent_confidence_before` | Float [0-1] | `0.267` | P1, P4 |
| `reflection_text` | String | `Target element masked or missing` | P4 |
| `memory_update_flag` | Boolean | `true` | P4 |
| `borrowed_image` | Boolean | `false` | DataLoader filter |

### 3.4 Label mappings (for encoding to integers)

```python
EXECUTION_OUTCOME = {"SUCCESS": 0, "FAILURE": 1}

FAILURE_TYPE = {"NONE": 0, "PERCEPTION_ERROR": 1,
                "ACTION_MISMATCH": 2, "LOOP_DETECTED": 3}

ACTION_TYPE = {"CLICK": 0, "TYPE": 1, "SELECT": 2,
               "SCROLL": 3, "NAVIGATE": 4}

RECOVERY_STRATEGY = {"NONE": 0, "RETRY": 1, "REPLAN": 2,
                     "BACKTRACK": 3, "ALTERNATIVE_TARGET": 4, "ABORT": 5}

MEMORY_FLAG = {False: 0, True: 1}
```

### 3.5 Known limitations (document in paper, do not hide)

1. **SCROLL/NAVIGATE absent** — Mind2Web source only recorded CLICK/TYPE/SELECT. Action head supports all 5 classes but only 3 have training data. Document this in methods section.
2. **14.4% borrowed images** — flagged with `borrowed_image=True`. Exclude from visual encoder fine-tuning AND from visual evaluation metrics. Keep for label-based heads.
3. **Synthetic failures** — failures injected programmatically, not observed. Defense: failure categories match Mind2Web's own real error analysis.
4. **REASONING_ERROR has zero examples** — Mind2Web does not annotate it. Note as gap/future work.
5. **3,805 steps missing bbox** — filter `action_target_bbox is not None` (67,160 usable) for bbox regression only.

---

## 4. ARCHITECTURE DESIGN

> The architecture is the contribution. It is backbone-agnostic. Sections 4.2–4.4 describe the dual-encoder path (SigLIP+RoBERTa style). Section 4.5 describes how a unified VLM (Qwen2.5-VL, InternVL2, Florence-2) plugs into the SAME heads and SAME loss.

### 4.1 High-level flow (dual-encoder path)

```
INPUT (image + text)
      |
      v
DUAL ENCODER (Pillar 2)
  vision encoder + text encoder
      |
      v
CROSS-ATTENTION FUSION (Pillar 2 — the novelty)
  produces one [B, 768] fused embedding
      |
      v
FOUR TASK HEADS (share the fused embedding)
  +-- Failure Head    (Pillar 1)
  +-- Action Head     (Pillar 3)
  +-- Memory Head     (Pillar 4)
      |
      v
OUTPUT (predictions for all pillars)
```

### 4.2 Component specifications

#### Vision Encoder (default dual-encoder config)
```
Default:  google/siglip-base-patch16-256
Input:    state_before image, 256x256x3
Output:   [B, 768] visual embedding
Training: Frozen in Phase 1, unfrozen (lr 1e-5) in Phase 2
```

#### Text Encoder (default dual-encoder config)
```
Default:  roberta-base
Input:    concatenated text:
          task_description [SEP] action_target_desc [SEP] website_domain
          max 128 tokens
Output:   [B, 768] text embedding (CLS token)
Training: Frozen in Phase 1, unfrozen (lr 1e-5) in Phase 2
```

#### Cross-Attention Fusion (BUILD THIS — Pillar 2 contribution)
```
Input:    vision_emb [B, 768], text_emb [B, 768]
Layers:   2 transformer cross-attention blocks
          - vision attends to text (Q=vision, K=V=text)
          - text attends to vision (Q=text, K=V=vision)
          - concatenate + LayerNorm + Linear(1536 -> 768)
Dropout:  0.1
Output:   fused_embedding [B, 768]
```

#### Failure Head (Pillar 1)
```
Shared trunk: Linear(768 -> 256) -> ReLU -> Dropout(0.3)
Output 1: Linear(256 -> 2)            -> execution_outcome (CrossEntropy)
Output 2: Linear(256 -> 4)            -> failure_type (CrossEntropy)
Output 3: Linear(256 -> 1) -> Sigmoid -> confidence (MSE)
Output 4: Linear(256 -> 6)            -> recovery_strategy (CrossEntropy)
```

#### Action Head (Pillar 3)
```
Shared trunk: Linear(768 -> 256) -> ReLU -> Dropout(0.3)
Output 1: Linear(256 -> 5)   -> action_type (CrossEntropy, class-weighted)
Output 2: Linear(256 -> 4)   -> bbox [x,y,w,h] (MSE, only where bbox exists)
```

#### Memory Head (Pillar 4)
```
Shared trunk: Linear(768 -> 256) -> ReLU -> Dropout(0.3)
Output 1: Linear(256 -> 1) -> Sigmoid -> memory_update_flag (BCE)
Output 2: Linear(256 -> 6)            -> recovery_strategy (CrossEntropy)
+ retrieval index at inference time (see 4.4)
```

### 4.3 Combined Loss Function

```python
L_total = (0.25 * L_failure_outcome     # CrossEntropy
         + 0.20 * L_failure_type        # CrossEntropy
         + 0.18 * L_action_type         # CrossEntropy (class-weighted)
         + 0.12 * L_bbox                # MSE (masked, only where bbox exists)
         + 0.10 * L_memory_flag         # BCE
         + 0.10 * L_recovery_strategy   # CrossEntropy
         + 0.05 * L_confidence)         # MSE
```
Loss weights are a hyperparameter ablation in the paper. Alternative: λ_outcome=0.40.

### 4.4 Memory Index (Pillar 4 — inference time)

```
Build: index over training embeddings WHERE
       memory_update_flag == True AND recovery_success == True
Store: {embedding, failure_type, recovery_strategy, reflection_text, website_domain}
Query: current fused_embedding -> top-3 cosine similar past failures
Return: their recovery_strategy as guidance
Episodic pairs: same original_task_id across 5 passes = natural memory pairs
```
**Simplification for v1:** start with simple top-k cosine similarity (numpy / sklearn). Upgrade to FAISS for production.

### 4.5 Plugging in a unified VLM (backbone-agnostic path)

For unified vision-language backbones (Qwen2.5-VL, InternVL2, Florence-2), vision and text are fused INSIDE the model. The architecture adapts cleanly:

```
VLM([image + text])      -> [B, D]   (D depends on the VLM, e.g. 896, 1536, 2048)
Adapter: Linear(D -> 768) -> [B, 768] (single linear adapter)
   (cross-attention fusion is SKIPPED — the VLM already fused modalities)
4 TASK HEADS (same as 4.2) -> outputs
Combined loss (same as 4.3)
```

**Key point:** the 4 heads and the loss NEVER change. Only the front-end encoder changes. This is what makes the architecture backbone-agnostic and is the central experimental claim of the paper.

---

## 5. MODEL LIST (FREE-FIRST)

> All 7 backbone configs use the SAME architecture (heads + loss). VLM configs use the 4.5 adapter path. Sizes chosen so almost everything trains FREE on Kaggle T4 16GB.

### 5.1 Your system — 7 backbone configurations

| ID | Backbone | Params | VRAM | Where | Cost |
|----|----------|--------|------|-------|------|
| Y1 | SigLIP-base-256 + RoBERTa-base | ~400M | ~6 GB | Kaggle T4 | FREE |
| Y2 | SigLIP-large-384 + RoBERTa-large | ~900M | ~12 GB | Kaggle T4 | FREE |
| Y3 | CLIP-ViT-L/14 + RoBERTa-base | ~450M | ~7 GB | Kaggle T4 | FREE |
| Y4 | Qwen2.5-VL-0.5B (adapter path) | ~0.5B | ~6 GB | Kaggle T4 | FREE |
| Y5 | Qwen2.5-VL-3B (adapter path, QLoRA) | ~3B | ~18 GB | Vast.ai A100 | ~$10 |
| Y6 | InternVL2-2B (adapter path, 4-bit) | ~2B | ~12 GB | Kaggle T4 | FREE |
| Y7 | Florence-2-large + RoBERTa-base | ~800M | ~10 GB | Kaggle T4 | FREE |

**Primary model = chosen AFTER training, by validation results.** Do NOT pre-fix it. Report all 7 in the backbone-generalization table and mark the empirical winner as primary in the paper.

### 5.2 Ablations — 5 variants (all use the SAME dual-encoder backbone, default Y1)

| ID | Name | What changed |
|----|------|--------------|
| A1 | Vision-Only | No text encoder, no fusion |
| A2 | Text-Only | No vision encoder, no fusion |
| A3 | Concat-Fusion | Concatenation instead of cross-attention |
| A4 | No-Memory | Remove Pillar 4 |
| A5 | No-Recovery | Remove recovery_strategy head |

### 5.3 Baselines — 7 systems

| ID | System | Model | How to get numbers |
|----|--------|-------|--------------------|
| B1 | Random | none | Compute yourself |
| B2 | MindAct | DeBERTa-v3 + Flan-T5 | Run on your test data |
| B3 | CLIP + MLP | CLIP-ViT-B/32 | Run on your test data |
| B4 | LayoutLMv3 | LayoutLMv3-base | Run on your test data |
| B5 | BacktrackAgent† | Qwen2-VL-7B | Published numbers |
| B6 | SeeAct† | GPT-4V | Published numbers |
| B7 | OSCAR† | GPT-4o | Published numbers |

† Use published Mind2Web numbers — too expensive to rerun.

**Total: 19 models** (7 backbones + 5 ablations + 7 baselines).

---

## 6. TRAINING PROCEDURE

### 6.1 Two-phase training (per dual-encoder model)

**Phase 1 — Warm up heads (3-4 hours)**
```
Vision encoder: FROZEN
Text encoder:   FROZEN
Fusion:         TRAINABLE (lr 1e-3)
Heads:          TRAINABLE (lr 1e-3)
Epochs:         3
```

**Phase 2 — Joint fine-tuning (10-14 hours)**
```
Vision encoder: TRAINABLE (lr 1e-5)
Text encoder:   TRAINABLE (lr 1e-5)
Fusion:         TRAINABLE (lr 1e-4)
Heads:          TRAINABLE (lr 1e-4)
Epochs:         7
```

For VLM configs (Y4, Y5, Y6, Y7): freeze the VLM (or QLoRA/4-bit), train the adapter + 4 heads in full precision. Same two-phase idea: heads first, then unfreeze adapter / LoRA layers.

### 6.2 Hyperparameters

```
Batch size:        32 (reduce to 8-16 for VLMs; use gradient accumulation to keep effective 32)
Optimizer:         AdamW
Weight decay:      0.01
Scheduler:         Cosine with warmup (10% warmup steps)
Early stopping:    patience 3 on val_loss
Checkpoint:        every 500 steps to free storage (Kaggle output / Drive / HF)
Gradient clip:     1.0
Mixed precision:   fp16 on T4 (bf16 on A100)
Class weights:     apply to action_type (CLICK is 83.6%)
                   apply to failure_type (LOOP is only 7.4%)
Seeds:             3 seeds (42, 1, 7) for the winning/primary model; 1 seed for others if time-limited
```

### 6.3 DataLoader patterns (filters, NOT new files)

| Use case | Filter | Rows |
|----------|--------|------|
| Full label training | `split == "train"` | 38,875 |
| Visual encoder fine-tune | `split == "train" and not borrowed_image` | ~33,300 |
| Clean visual eval | `test split and not borrowed_image` | ~27,500 |
| Label-only eval | `test split` | 32,090 |
| BBox regression | `action_target_bbox is not None` | 67,160 |

---

## 7. STEP-BY-STEP EXECUTION PLAN (FREE-FIRST)

### Phase 0 — Setup
```
[ ] 1. Kaggle: phone-verify, enable GPU T4, confirm 30h/week quota
[ ] 2. Upload data as a private Kaggle Dataset (JSONs + zipped images)
[ ] 3. HuggingFace: create public dataset repo + public model repo
[ ] 4. Install: torch, transformers, pillow, scikit-learn, tqdm, (faiss-cpu later)
[ ] 5. Set seeds (42), confirm nvidia-smi shows T4
```

### Phase 1 — Foundation
```
[ ] 6.  Run dataset verification script (counts, splits, label logic) -> save report
[ ] 7.  Write Dataset class (reads JSON, loads images, tokenizes, returns tensors)
[ ] 8.  Write DataLoader with mode filters (section 6.3)
[ ] 9.  VERIFY: one batch loads correctly, shapes right, images open, labels aligned
[ ] 10. Write Model class (dual encoder + fusion + 4 heads)
[ ] 11. Write combined loss function (section 4.3)
[ ] 12. Train Y1 on 5,000 samples — PROVE pipeline works end to end
[ ] 13. Confirm loss drops in first 50 batches; checkpoint round-trips
```

### Phase 2 — Train Y1 fully (the first full model)
```
[ ] 14. Train Y1 full — Phase 1 (frozen encoders, 3 epochs)
[ ] 15. Train Y1 full — Phase 2 (unfrozen, 7 epochs), 3 seeds
[ ] 16. Build memory index from Y1 embeddings
[ ] 17. Evaluate Y1 on all 3 test splits
[ ] 18. TARGET: Failure Detection F1 > 70%
```

### Phase 3 — Ablations (all dual-encoder, default backbone)
```
[ ] 19. Train A1 Vision-Only
[ ] 20. Train A2 Text-Only
[ ] 21. Train A3 Concat-Fusion
[ ] 22. Train A4 No-Memory
[ ] 23. Train A5 No-Recovery
[ ] 24. Build ablation table (each vs full model)
```

### Phase 4 — Backbone variants (smallest first)
```
[ ] 25. Train Y4 Qwen2.5-VL-0.5B  (FREE, Kaggle)
[ ] 26. Train Y3 CLIP variant      (FREE, Kaggle)
[ ] 27. Train Y7 Florence-2        (FREE, Kaggle)
[ ] 28. Train Y2 SigLIP-large      (FREE, Kaggle)
[ ] 29. Train Y6 InternVL2-2B 4-bit(FREE, Kaggle)
[ ] 30. Train Y5 Qwen2.5-VL-3B QLoRA (~$10, Vast.ai, LAST)
```

### Phase 5 — Baselines
```
[ ] 31. Compute B1 Random baseline
[ ] 32. Train B2 MindAct on your data
[ ] 33. Train B3 CLIP+MLP on your data
[ ] 34. Train B4 LayoutLMv3 on your data
[ ] 35. Collect B5/B6/B7 published numbers from papers
```

### Phase 6 — Evaluation + Paper
```
[ ] 36. Full evaluation all 19 models, all 3 splits
[ ] 37. SELECT PRIMARY MODEL by validation results
[ ] 38. Per-failure-type breakdown table
[ ] 39. Statistical significance (3 seeds, mean+std, p-values)
[ ] 40. Build 3 paper tables (main results, ablation, backbone generalization)
[ ] 41. Write paper sections 1-8
[ ] 42. Upload dataset + best model to HuggingFace
[ ] 43. Build demo (FailureScope) on HF Spaces ZeroGPU
[ ] 44. Submit paper
```

---

## 8. EVALUATION METRICS

### 8.1 Primary metrics
```
Failure Detection F1      (Pillar 1, binary SUCCESS/FAILURE)
Failure Type Accuracy     (Pillar 1, 4-way)
Action Type Accuracy      (Pillar 3, 5-way)
Recovery Success Rate     (Pillar 1 + 4)
```

### 8.2 Secondary metrics
```
Per-split: test_task, test_website, test_domain (generalization curve)
Per-failure-type F1: PERCEPTION_ERROR, ACTION_MISMATCH, LOOP_DETECTED
Confidence calibration gap (SUCCESS avg vs FAILURE avg)
Memory retrieval accuracy (Pillar 4)
```

### 8.3 Efficiency metrics
```
Inference time per step (ms)
Memory index query time (ms)
GPU memory footprint (GB)
Cost per task vs GPT-4o baselines
```

### 8.4 Three paper tables
1. **Main Results** — all 19 models, primary metrics, per split
2. **Ablation** — 5 ablations vs full model (proves each component matters)
3. **Backbone Generalization** — Y1-Y7 (proves backbone-agnostic; winner = primary)

---

## 9. FREE STORAGE STRUCTURE

```
Kaggle Dataset (free, 100GB):
  data/
    split_train.json / split_val.json / split_test.json
    images/                      <- ONE shared folder, all splits

Kaggle Notebook output (free, per-run):
  checkpoints/  logs/  results/

HuggingFace (free):
  Datasets repo  -> FINAL_Trajectories_ENRICHED.json (public, for paper)
  Model Hub repo -> best model checkpoint (public, for paper)
  Spaces ZeroGPU -> FailureScope demo (public URL for paper)

Google Drive (free 15GB):
  code backup + final results + tables/figures
```

---

## 10. COMPUTE & BUDGET (FREE-FIRST)

```
Kaggle Free T4 16GB:
  Y1, Y2, Y3, Y4, Y6, Y7, A1-A5, B1-B4  -> FREE (~16 models)
  30 GPU-hrs/week; sessions up to 12h; checkpoint to resume

Vast.ai A100 40GB ($0.35/hr):
  Y5 Qwen2.5-VL-3B (QLoRA) only -> ~$10
  ALWAYS stop the instance the moment training ends

Storage: all free (Kaggle + HuggingFace + Drive 15GB)

TOTAL CASH: ~$10  (everything else free)
```

---

## 11. PRODUCTION AGENT (after paper)

```
USER REQUEST
     |
     v
LLM PLANNER (Claude / GPT-4o / Qwen)   <- decomposes task into subtasks
     |                                     called ONCE per task
     v
YOUR PRIMARY MODEL (served via FastAPI) <- executes each step
     |                                     detects + recovers per step
     v                                     called N times (free, fast)
PLAYWRIGHT BROWSER (Chromium)          <- runs actions on real websites
     |
     +-- success -> next step
     +-- ABORT   -> structured failure report -> back to LLM Planner
```

**Cost advantage:** 1 LLM call/task + N free model calls = ~30x cheaper than GPT-4o-per-step systems.

**Stack:** Nuxt 3 + FastAPI + Playwright + FAISS + Hetzner VPS + WebSocket.

---

## 12. WHAT NOT TO DO

```
[X] Do not pre-fix the primary model — let validation results choose it
[X] Do not start with the largest VLM + all 4 pillars + 70k samples on day 1
[X] Do not split the images folder — all splits share it
[X] Do not mix train and test_ records when splitting
[X] Do not use borrowed images for visual encoder fine-tuning
[X] Do not synthetically fill missing action types — document them instead
[X] Do not skip the 5k-sample pipeline test before full training
[X] Do not train all 19 models before confirming the first full model works
[X] Do not leave the Vast.ai instance running after training (wastes money)
[X] Do not build the full decision combiner for the paper — defer to production
```

---

## 13. CORE PRINCIPLE

> **A simple system evaluated rigorously beats a complex system evaluated poorly.**

The dataset is the strongest asset (9/10). The biggest risk is evaluation rigor and dataset realism. Invest 40% effort in building, 60% in evaluation. Be honest about all limitations in the paper.

---

## 14. ONE-PARAGRAPH SUMMARY (for the coding agent)

We are training a multi-head neural network that takes a web page screenshot plus a task description and simultaneously predicts: whether the current step failed, what type of failure it is, what action to take, where to click, whether to store the experience in memory, and what recovery strategy to apply. The architecture is backbone-agnostic: a vision encoder and a text encoder are fused with a custom cross-attention layer (the novelty), and four task heads sit on top of the fused embedding, one group per research pillar. For unified vision-language backbones, a single adapter layer replaces the fusion and the SAME four heads and SAME combined loss are used. Everything trains together on 38,875 labeled web interaction steps using a weighted combined loss across all objectives. We train this SAME architecture with seven different backbones (sized to run free on Kaggle T4, except one small paid run), run five ablations, and compare against seven baselines — 19 models total — to prove the methodology is backbone-agnostic and outperforms prior work. The best-performing backbone, chosen by validation results, becomes the primary model. Start by building the DataLoader, prove the pipeline on 5,000 samples with the default SigLIP+RoBERTa configuration, then scale to the full dataset and the remaining models.
