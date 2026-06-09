# Phase 3+ — Roadmap (preview of what's coming)

These phases aren't built yet. This file previews the topics so you can learn
ahead. Each will get its own full `learn/` file when we build it.

## Phase 3 — Model core

The heart of the project: turn the batch of tensors into 7 predictions.

### Learning topics
- **Encoder** — a neural net that turns raw input into a feature vector
  (SigLIP for images, RoBERTa for text → each gives a `[B, 768]` vector).
- **Embedding** — that feature vector; a compressed numeric meaning.
- **Attention / cross-attention** — how the model lets vision and text "look at"
  each other so they agree before deciding. This is the project's novelty.
- **Task head** — a small net on top of the shared embedding, one per job.
- **Logits** — raw scores before turning into probabilities.

### Code structure (to build)
```
src/web_agent/models/
  encoders/vision.py  SigLIP -> [B,768]
  encoders/text.py    RoBERTa -> [B,768]
  fusion.py           cross-attention: vision+text -> one [B,768]   (NOVELTY)
  adapter.py          (VLM path) Linear(D->768)        [already real]
  heads.py            FailureHead, ActionHead, MemoryHead   (FIXED)
  model.py            wire front-end + heads -> dict of 7 outputs
```

### Algorithm (forward pass)
```
image -> vision encoder -> v[768]
text  -> text encoder   -> t[768]
fusion(v, t)            -> fused[768]      (the shared understanding)
fused -> FailureHead    -> outcome, failure_type, confidence, recovery
fused -> ActionHead     -> action_type, bbox
fused -> MemoryHead     -> memory_flag, recovery
```

## Phase 4 — Combined loss

### Learning topics
- **Loss function** — a number measuring how wrong a prediction is.
- **CrossEntropyLoss** — for "which class?" questions.
- **MSELoss** — for "what value?" (confidence, bbox).
- **BCELoss** — for "yes/no probability?" (store in memory?).
- **Class weighting** — boosting rare classes so the model doesn't ignore them.
- **Masking a loss** — skipping rows with no bbox.
- **Weighted multi-task loss** — adding several losses with importance weights.

### The formula (already in configs/base.yaml)
```
L_total = 0.25*outcome + 0.20*failtype + 0.18*action + 0.12*bbox(masked)
        + 0.10*memory + 0.10*recovery + 0.05*confidence
```

## Phase 5 — Training loop (lives in the notebook, hybrid plan)

### Learning topics
- **Forward pass** — run inputs through the model to get predictions.
- **Backward pass / backpropagation** — compute how to nudge each weight.
- **Optimizer (AdamW)** — applies those nudges.
- **Epoch / step / batch** — one pass over data / one update / one group of rows.
- **Learning rate + warmup + cosine schedule** — how big the nudges are over time.
- **Gradient clipping** — capping nudges so training stays stable.
- **Mixed precision (fp16)** — using 16-bit floats to train faster on the T4.
- **Two-phase training** — first train the new parts (encoders frozen), then
  fine-tune everything gently.
- **Checkpointing** — saving weights so a dead Kaggle session can resume.

### The test-first ladder (never skip)
```
SMOKE (16 rows)  -> does it run? shapes ok, loss finite, one step works
MINI  (5,000)    -> does it learn? loss drops, F1 > random, checkpoint reloads
FULL  (38,875)   -> the real run
```

## Phase 6 — Memory index (Pillar 4)

### Learning topics
- **Embedding similarity / cosine similarity** — measuring how alike two vectors are.
- **Top-k retrieval** — finding the k most similar past cases.
- **FAISS** — a fast library for similarity search (used later for scale).

The idea: store embeddings of past *successful recoveries*; at inference, find
the most similar past failure and reuse its recovery strategy.

## Phase 7-10 — Ablations, backbones, baselines, evaluation

### Learning topics
- **Ablation study** — remove one part, measure the drop, prove that part matters.
- **Baseline** — a simpler/older method to compare against (the floor).
- **F1 score, accuracy, macro-F1** — ways to score classification.
- **Generalization splits** — test on unseen tasks / websites / domains.
- **Statistical significance (seeds, mean±std, p-values)** — proving results
  aren't luck, required for a Q1 paper.

## Suggested learning order

1. Finish understanding Phase 0-2 (already built — read those files + run the cells).
2. Learn: what an embedding is, what attention is (Phase 3).
3. Learn: what a loss function and backpropagation are (Phase 4-5).
4. Then we build Phase 3, and this folder gets a full `phase3_model.md`.
