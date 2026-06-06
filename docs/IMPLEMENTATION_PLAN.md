# Implementation Plan — Failure-Aware Resilient Autonomous Web Agents

> **Target:** Q1 top-tier journal publication + thesis competition.
> **Current first priority:** Train all models one by one. Application comes later.
> **Read this with `PROJECT_SPECIFICATION.md`. That file defines the *what*; this file defines the *how* and the *order*.**

---

## 0. GUIDING PRINCIPLES

1. **Prove the pipeline on a tiny subset before scaling.** Never start a 40-hour run on unverified code.
2. **One thing at a time.** Finish and validate Y1 completely before touching any other model.
3. **Reproducibility is mandatory for Q1.** Fix seeds, log everything, version every checkpoint.
4. **Evaluation is 60% of the work.** A simple model evaluated rigorously beats a complex model evaluated poorly.
5. **Document every limitation honestly.** Reviewers reward transparency and punish hidden flaws.
6. **Checkpoint constantly.** Colab/Kaggle sessions die. Resume must always be possible.

---

## PHASE 0 — ENVIRONMENT & SANITY (Day 1–2)

### 0.1 Requirements
```
Compute:   Kaggle Free (T4 16GB) — primary, 30 GPU-hrs/week
Paid:      Vast.ai A100 40GB — ONLY for Y5 Qwen2.5-VL-3B, last
Storage:   Kaggle Datasets + HuggingFace + Google Drive 15GB, per PROJECT_SPECIFICATION section 9
Python:    3.10+
Libraries: torch, torchvision, transformers, accelerate, bitsandbytes,
           pillow, numpy, pandas, scikit-learn, tqdm,
           faiss-cpu, matplotlib, seaborn, wandb (or tensorboard)
```

### 0.2 Tasks
```
[ ] Kaggle: phone-verify, enable GPU T4, confirm 30h/week quota
[ ] Confirm T4 with: nvidia-smi
[ ] pip install all libraries, pin versions in requirements.txt
[ ] Set global seeds: torch, numpy, random = 42
[ ] Set up experiment logger (Weights & Biases recommended for Q1)
[ ] Create folder tree (data, checkpoints, logs, results, memory_index) in Kaggle output + Drive backup
```

### 0.3 Techniques
- Use `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic = True` for reproducibility.
- Log every run config (model, lr, batch, seed) to W&B so every paper number is traceable.

### 0.4 Exit criteria
- T4 visible, all libraries import, storage ready, one dummy tensor moves to GPU.

---

## PHASE 1 — DATASET VERIFICATION (Day 2–4)

> The dataset is your strongest asset (9/10). Verify it before trusting it.

### 1.1 Integrity checks
```
[ ] Load split_train.json, split_val.json, split_test.json
[ ] Confirm counts: train 38,875 | val ~16,000 | test ~16,000
[ ] Confirm total = 70,965 across all three
[ ] Confirm ZERO original_task_id overlap between train and val/test
[ ] Confirm all 23 fields present in every record
[ ] Confirm zero empty state_before / state_after paths
[ ] Confirm every referenced image exists on disk (sample 1,000, then full scan)
```

### 1.2 Label distribution checks
```
[ ] execution_outcome: ~72% FAILURE / ~28% SUCCESS
[ ] failure_type: 4 classes present in all 3 splits
[ ] action_type: CLICK/TYPE/SELECT present (SCROLL/NAVIGATE absent — expected)
[ ] recovery_strategy: 5 classes present
[ ] memory_update_flag: True for all FAILURE, False for all SUCCESS
[ ] Confirm confidence gap: FAILURE avg ~0.41, SUCCESS avg ~0.83
```

### 1.3 Logic consistency checks
```
[ ] No SUCCESS row has a non-NONE failure_type
[ ] No FAILURE row has memory_update_flag = False
[ ] No SUCCESS row has a non-NONE recovery_strategy
[ ] bbox present count = 67,160 (3,805 missing — for bbox regression filter)
[ ] borrowed_image True count ~10,190 (14.4%)
```

### 1.4 Techniques
- Write a single `verify_dataset.py` that prints a full report and asserts on every rule.
- Save the report to `results/dataset_verification.txt` — this becomes evidence in the paper's dataset section.

### 1.5 Exit criteria
- All assertions pass. Verification report saved. Any anomaly documented as a known limitation.

---

## PHASE 2 — DATALOADER (Day 4–7)

### 2.1 What to build
A `WebAgentDataset` (PyTorch `Dataset`) plus a configurable `DataLoader`.

### 2.2 Dataset class responsibilities
```
Input per record:
  - Load state_before image, resize to 256x256, normalize (SigLIP processor)
  - Build text string: task_description [SEP] action_target_desc [SEP] website_domain
  - Tokenize text (RoBERTa tokenizer, max_len 128, pad/truncate)
  - Normalize bbox to [0,1] range, or zero-mask if missing

Labels per record (encode to ints per PROJECT_SPECIFICATION 3.4):
  - label_outcome, label_failtype, label_action, label_bbox,
    label_memory, label_recovery, label_confidence
  - bbox_mask (1 if bbox present, 0 if missing)
```

### 2.3 DataLoader modes (filters, NOT new files)
```
mode="full_labels"  -> split=="train"                       (38,875)
mode="visual"       -> split=="train" and not borrowed_image (~33,300)
mode="eval_visual"  -> test split and not borrowed_image     (~27,500)
mode="eval_labels"  -> test split                            (32,090)
mode="bbox"         -> action_target_bbox is not None        (67,160)
```

### 2.4 Techniques
- Use `num_workers=2`, `pin_memory=True` for Kaggle T4 throughput.
- Cache the JSON in memory once; load images lazily per batch.
- Handle borrowed images by passing `borrowed_image` flag through so the loss can mask visual terms if needed.
- Use a custom `collate_fn` to batch variable fields cleanly.

### 2.5 Exit criteria — DO NOT SKIP
```
[ ] One batch loads without error
[ ] image_tensor shape = [32, 3, 256, 256]
[ ] text_tokens shape = [32, 128]
[ ] all label tensors have correct shape and dtype
[ ] visually inspect 4 decoded images + their labels to confirm alignment
```

---

## PHASE 3 — MODEL CLASS (Day 7–10)

### 3.1 Build order (bottom up)
```
[ ] 3.1 Vision encoder wrapper (SigLIP) -> [B, 768]
[ ] 3.2 Text encoder wrapper (RoBERTa CLS) -> [B, 768]
[ ] 3.3 Cross-attention fusion module -> [B, 768]   (PILLAR 2 — the novelty)
[ ] 3.4 Failure head (4 outputs)                     (PILLAR 1)
[ ] 3.5 Action head (action type + bbox)             (PILLAR 3)
[ ] 3.6 Memory head (flag + recovery)                (PILLAR 4)
[ ] 3.7 Wrap all into one nn.Module returning a dict of logits
```

### 3.2 Cross-attention fusion (technique detail)
- 2 transformer blocks. Vision attends to text (Q=vision, K=V=text) and text attends to vision (Q=text, K=V=vision).
- Concatenate both attended outputs, LayerNorm, `Linear(1536 -> 768)`, dropout 0.1.
- This module is the defendable architectural contribution — keep it clean and documented.

### 3.3 Combined loss module
```
L_total = 0.25*L_outcome + 0.20*L_failtype + 0.18*L_action
        + 0.12*L_bbox(masked) + 0.10*L_memory + 0.10*L_recovery
        + 0.05*L_confidence
```
- bbox loss masked by `bbox_mask`.
- action_type CrossEntropy uses class weights (CLICK dominates at 83.6%).
- failure_type CrossEntropy uses class weights (LOOP only 7.4%).

### 3.4 Exit criteria
```
[ ] Forward pass on one batch returns all 7 logit tensors with correct shapes
[ ] Loss computes to a finite scalar
[ ] loss.backward() runs without error
[ ] One optimizer.step() changes weights (verify a parameter delta)
```

---

## PHASE 4 — PIPELINE PROOF ON 5,000 SAMPLES (Day 10–12)

> The single most important risk-reduction step. Catches bugs cheaply.

### 4.1 Tasks
```
[ ] Subsample 5,000 train records (stratified by failure_type)
[ ] Train Y1 (SigLIP + RoBERTa) for 3 epochs, frozen encoders
[ ] Confirm training loss DROPS in first 50 batches
[ ] Confirm validation loss is computed correctly
[ ] Confirm checkpoint saves to Kaggle output (+ Drive backup) and reloads correctly
[ ] Produce a first (rough) Failure Detection F1 number
```

### 4.2 Exit criteria
- Loss decreases, no NaNs, checkpoint round-trips, a non-trivial F1 appears.
- **Only after this passes do you scale to the full dataset.**

---

## PHASE 5 — TRAIN PRIMARY MODEL Y1 (Week 2)

> Y1 = SigLIP-base-256 + RoBERTa-base + cross-attention + 4 heads. This is the model you defend.

### 5.1 Two-phase training
```
Phase 1 — warm up heads (3 epochs)
  SigLIP frozen, RoBERTa frozen, fusion+heads trainable, lr 1e-3
Phase 2 — joint fine-tune (7 epochs)
  all trainable, encoder lr 1e-5, head lr 1e-4
```

### 5.2 Hyperparameters
```
batch 32 | AdamW | weight_decay 0.01 | cosine schedule w/ 10% warmup
grad clip 1.0 | fp16/bf16 | early stopping patience 3 on val_loss
checkpoint every 500 steps to Kaggle output (+ Drive/HF backup)
```

### 5.3 Techniques for Q1 rigor
- Train Y1 with **3 random seeds** (42, 1, 7) → report mean ± std.
- Save best checkpoint by validation Failure F1, not just loss.
- Log learning curves to W&B for the paper appendix.

### 5.4 Exit criteria
```
[ ] Y1 trains to completion across all 3 seeds
[ ] Target: Failure Detection F1 > 70% on val
[ ] Checkpoints + logs saved for all seeds
```

---

## PHASE 6 — MEMORY MODULE (Week 2, after Y1)

### 6.1 Build (Pillar 4)
```
[ ] Extract fused embeddings for all train rows where
    memory_update_flag=True AND recovery_success=True
[ ] Build index: start SIMPLE (top-k cosine), upgrade to FAISS later
[ ] Store {embedding, failure_type, recovery_strategy, reflection_text, website_domain}
[ ] Query function: current embedding -> top-3 similar -> return recovery_strategy
```

### 6.2 Technique
- Treat 5 passes of the same `original_task_id` as episodic pairs for evaluation of retrieval quality.
- Measure **memory retrieval accuracy**: does retrieved strategy match the successful recovery?

### 6.3 Exit criteria
- Index builds, queries return sensible neighbours, retrieval accuracy logged.

---

## PHASE 7 — ABLATIONS (Week 3–4)

> All use the Y1 backbone. Each isolates one component. This table answers reviewer questions.

```
[ ] A1 Vision-Only      (no text, no fusion)
[ ] A2 Text-Only        (no vision, no fusion)
[ ] A3 Concat-Fusion    (concat instead of cross-attention)
[ ] A4 No-Memory        (remove Pillar 4)
[ ] A5 No-Recovery      (remove recovery head)
```

### 7.1 Techniques
- Keep every other setting identical to Y1 so differences are attributable to the removed component.
- Run each ablation with at least 1 seed (2 if time permits).
- Expected story: every ablation is worse than full Y1; A3 vs Y1 proves cross-attention; A4/A5 prove memory and recovery matter (ReUseIt showed ~20% drop without recovery).

### 7.2 Exit criteria
- Ablation table fully populated, each variant compared to Y1, deltas computed.

---

## PHASE 8 — BACKBONE VARIANTS (Month 2, Week 1–3)

> Proves the method is backbone-agnostic — the competition-winning claim. Run smallest first. Sizes per PROJECT_SPECIFICATION section 5.1 (free-first).

```
[ ] Y3 CLIP-ViT-L/14 + RoBERTa-base       (Kaggle T4 FREE, dual encoder — proves SigLIP > CLIP)
[ ] Y7 Florence-2-large + RoBERTa-base    (Kaggle T4 FREE, dual encoder — watch bbox)
[ ] Y4 Qwen2.5-VL-0.5B (adapter path)     (Kaggle T4 FREE, first VLM — debug VLM path)
[ ] Y2 SigLIP-large-384 + RoBERTa-large   (Kaggle T4 FREE, dual encoder — batch 16 + grad accum 2)
[ ] Y6 InternVL2-2B (adapter path, 4-bit) (Kaggle T4 FREE, second VLM)
[ ] Y5 Qwen2.5-VL-3B (adapter path, QLoRA)(Vast.ai A100, LAST, paid — stop instance instantly)
```

### 8.1 Techniques
- For VLMs (Y4, Y5, Y6): use the adapter path (PROJECT_SPECIFICATION 4.5) — VLM pooled output [D] -> Adapter Linear(D->768) -> 4 heads, skip cross-attention. Use **4-bit / QLoRA** to fit Kaggle T4 16GB.
- For dual encoders (Y2, Y3, Y7): keep cross-attention fusion, swap only the encoder.
- Reuse the SAME head code and SAME loss across all backbones — only the front-end changes.
- Share the SigLIP/RoBERTa fine-tuned weights where applicable to save ~40% compute.

### 8.2 Exit criteria
- All 7 backbone configs trained and evaluated; backbone-generalization table populated.

---

## PHASE 9 — BASELINES (Month 2, Week 3–4)

```
[ ] B1 Random            (compute analytically — majority class)
[ ] B2 MindAct           (DeBERTa-v3 + Flan-T5-XL, run on your test)
[ ] B3 CLIP + MLP        (CLIP-ViT-B/32, run on your test)
[ ] B4 LayoutLMv3        (run on your test)
[ ] B5 BacktrackAgent†   (cite published Mind2Web numbers)
[ ] B6 SeeAct†           (cite published numbers)
[ ] B7 OSCAR†            (cite published numbers)
```
† Published numbers — too expensive to rerun. Cite the paper and note the comparison is against reported results.

---

## PHASE 10 — EVALUATION & RESULT ANALYSIS (Month 2 Week 4 → Month 3 Week 1)

> This is 60% of the paper's value. Do it thoroughly.

### 10.1 Metrics (compute for every model)
```
Primary:
  Failure Detection F1            (binary)
  Failure Type Accuracy / macro-F1 (4-way)
  Action Type Accuracy            (5-way)
  Recovery Success Rate

Secondary:
  Per split: test_task / test_website / test_domain
  Per failure type: PERCEPTION_ERROR / ACTION_MISMATCH / LOOP_DETECTED
  Confidence calibration gap
  Memory retrieval accuracy

Efficiency:
  Inference time per step (ms)
  GPU memory (GB)
  Cost per task vs GPT-4o baselines
```

### 10.2 Statistical rigor (required for Q1)
```
[ ] 3 seeds for Y1 (and key models) -> report mean ± std
[ ] Significance tests vs strongest baseline (paired t-test or bootstrap, report p-values)
[ ] Confusion matrices for failure_type and action_type
[ ] Calibration plot (predicted confidence vs actual success)
```

### 10.3 Three core tables
```
Table 1 — Main Results: all 19 models, primary metrics, per split
Table 2 — Ablation: A1–A5 vs full Y1
Table 3 — Backbone Generalization: Y1–Y7
```

### 10.4 Figures
```
[ ] Architecture diagram (4 pillars)
[ ] Per-failure-type F1 bar chart
[ ] Cross-domain degradation curve (test_task -> test_website -> test_domain)
[ ] Confidence calibration plot
[ ] Efficiency vs accuracy scatter (your small model vs GPT-4o systems)
```

### 10.5 Analysis questions to answer
```
[ ] When does memory help most? (which failure types / domains)
[ ] Which failure type is hardest to detect and why?
[ ] How much does performance drop on unseen domains?
[ ] Does cross-attention beat concatenation, and by how much?
[ ] Does the method improve every backbone? (yes/no per model)
```

---

## PHASE 11 — PAPER WRITING (Month 3)

### 11.1 Structure (Q1 journal)
```
1. Introduction        — gap, contribution, claims
2. Related Work        — 8 studied papers grouped by theme
3. Dataset             — construction, taxonomy, stats, limitations
4. Methodology         — 4 pillars, fusion, loss, training
5. Experiments         — 19 models, 3 tables, setup
6. Analysis            — the 5 analysis questions above
7. Limitations & Future Work — SCROLL gap, synthetic failures, REASONING_ERROR
8. Conclusion
```

### 11.2 Techniques
- Write Sections 2, 3, 4 **in parallel with training** (they need no results).
- Release dataset + code on HuggingFace/GitHub at submission (Q1 expects artifacts).
- Frame the dataset as "failure-augmentation methodology on Mind2Web," not "new dataset."

### 11.3 Target venues
```
IEEE TNNLS | ACM TOSEM | IEEE TPAMI (stretch) | Pattern Recognition
Conference alternative: NeurIPS / ICLR / ACL (Datasets & Benchmarks)
```

---

## PHASE 12 — APPLICATION (AFTER PAPER — lower priority now)

> Defer until training and paper are done. Listed for completeness.
```
LLM Planner (Claude/GPT-4o/Qwen) -> decomposes task
  -> Y1 served via FastAPI executes each step (failure detect + recover)
  -> Playwright drives Chromium
  -> on ABORT, structured report back to LLM Planner
Stack: Nuxt 3 + FastAPI + Playwright + FAISS + Hetzner VPS + WebSocket
Cost: 1 LLM call/task + N free model calls (~30x cheaper than GPT-4o-per-step)
```

---

## 13. COMPUTE BUDGET (with shared-weight reuse)

```
Y1 (3 seeds)         ~36 h
A1–A5 ablations      ~60 h
Y2,Y3,Y4,Y6,Y7       ~118 h
Y5 (Qwen2.5-VL-7B)   ~45 h
Baselines B2–B4      ~34 h
Evaluation           ~30 h
-------------------------------
Total                ~260–390 h over 3 months (≈3 h A100/day)
```

---

## 14. RISK REGISTER

| Risk | Mitigation |
|------|------------|
| Scope creep (3 papers of work) | Lock scope: dataset + P1 + P2 + P3 mandatory; P4 simplified |
| Dataset realism doubted | Defense: failure types match Mind2Web's own error analysis; offer 500–1000 real-failure eval set as contingency |
| Session death loses progress | Checkpoint every 500 steps to Drive; resume logic mandatory |
| VLM won't fit Kaggle T4 16GB | 4-bit/QLoRA, grad checkpointing, small batch; sizes capped at 0.5B/2B/3B; Y5 (3B) on paid A100 last |
| Weak results vs Qwen | Frame contribution as framework + dataset + efficiency, not raw SOTA |
| Evaluation too thin | 60% effort on eval; 3 seeds; p-values; per-type breakdown |

---

## 15. DEFINITION OF DONE (per model)

```
[ ] Trained across required seeds
[ ] Best checkpoint saved to Kaggle output (+ Drive/HF backup) with config logged
[ ] Evaluated on all 3 test splits
[ ] All primary + secondary metrics computed and logged
[ ] Row added to the relevant results table
[ ] Learning curves saved for appendix
```

---

## 16. IMMEDIATE NEXT ACTIONS (start here)

```
1. Phase 0 — environment + Drive + seeds (today)
2. Phase 1 — run verify_dataset.py, save report
3. Phase 2 — build Dataset + DataLoader, inspect one batch
4. Phase 3 — build Model + loss, confirm forward/backward
5. Phase 4 — train Y1 on 5,000 samples, confirm loss drops
   -> only then scale to full Y1 in Phase 5
```

**Order is non-negotiable. Do not jump ahead. Validate each phase before the next.**
