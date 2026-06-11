# Phase 5 — Full QLoRA Training (Qwen2-VL-2B)

## 1. Learning topics

- **QLoRA** — 4-bit quantized base + small trainable LoRA adapters (fine-tune a 2B
  model on a free T4). See the earlier explanation: LoRA = train tiny side matrices,
  freeze the rest; Q = 4-bit weights.
- **peft** — the library that adds LoRA to a HuggingFace model.
- **Param groups** — different learning rates for different parts (LoRA vs heads).
- **Gradient accumulation** — sum gradients over N small batches to act like one big batch.
- **Mixed precision + GradScaler** — fp16 math with a loss scaler to avoid underflow.
- **Cosine schedule + warmup** — LR ramps up then decays smoothly.
- **Early stopping on a metric** — stop when val Failure-F1 stops improving (not loss).
- **Multi-task loss weighting** — combine 10 objectives with importance weights.
- **Contrastive learning** — pull similar things together, push different apart.
- **Calibration / ECE** — does predicted confidence match real accuracy?
- **Class imbalance** — sklearn "balanced" weights so rare classes aren't ignored.

## 2. Why this phase exists

Earlier phases proved the pipeline runs and learns with a frozen VLM. This phase is
the real training: unfreeze the VLM a little (LoRA) so it adapts to web screenshots,
add the full spec (5 heads, 10-loss-term, multi-step context, memory), and produce
the numbers for the paper.

> Note: the spec named `Qwen2.5-VL-0.5B` (hidden 896) — that doesn't exist (it's the
> text-only Qwen2.5-0.5B, which can't see images). The real smallest vision model is
> `Qwen2-VL-2B` (hidden 1536), used here. Everything else follows the spec.

## 3. Code structure (what changed)

```
configs/backbones/qwen2vl_2b.yaml   4-bit QLoRA, LoRA block, 10-term weights, lr groups
models/encoders/vlm.py              + peft LoRA (r8 on q_proj/v_proj)
models/adapter.py                   Linear(1536->768) -> LayerNorm -> Dropout(0.1)
models/heads.py                     + RecoveryOutcomeHead (5th head)
models/model.py                     wires 5 heads; forward returns `fused` embedding
data/dataset.py                     multi-step context (last 3 trajectory steps)
data/dataloader.py                  TaskPairBatchSampler (contrastive pairs)
models/loss.py                      10-term combined loss
eval/metrics.py                     Failure-F1, accuracies, ECE, bbox MAE
train/trainer.py                    the training loop (param groups, schedule, ckpt)
memory/index.py                     cosine top-3 retrieval
eval/evaluate.py                    per-split generalization eval
```

## 4. How it works (algorithm)

### One training step
```
batch (pair-sampled so a task's passes are together)
  -> VLM (4-bit base + LoRA) processes image+text -> mean-pool -> [B,1536]
  -> adapter Linear+LN+Dropout -> fused [B,768]
  -> 5 heads -> predictions dict (+ fused for contrastive)
  -> 10-term loss vs labels
  -> loss / accum_steps ; scaler.scale(loss).backward()
  every 4 micro-batches:
     unscale -> clip grads -> optimizer step (2 LR groups) -> scheduler step -> zero grad
```

### The 10-term loss
```
0.22 outcome     CE (label smoothing 0.1)
0.18 failure_type CE (sklearn balanced weights)
0.15 action       CE (balanced weights)
0.10 bbox         MSE, masked to rows with a box
0.10 memory_flag  BCE
0.09 recovery     CE, averaged over the Failure and Memory heads
0.05 confidence   MSE (predictions clipped to [0.05, 0.95])
0.05 calibration  differentiable soft-binned |confidence - accuracy| surrogate
0.04 contrastive  margin 0.5 over same-task SUCCESS/FAILURE embeddings
0.02 recovery_outcome BCE (did the recovery succeed?)
```

### Why a pair-aware sampler (contrastive)
Contrastive needs a SUCCESS and a FAILURE of the **same task** in one batch.
Random batching almost never does that. `TaskPairBatchSampler` groups the 5 passes
of an `original_task_id` into the same batch so the pairs exist. If a batch has no
pair, the contrastive term is 0 for that batch (a red flag only if it's ALWAYS 0).

### Why early-stop on Failure-F1 not loss
The loss mixes 10 objectives; a lower loss doesn't always mean better failure
detection (the metric we report). So we checkpoint and stop on val Failure-F1.

## 5. Key functions

| Function | File | Role |
|----------|------|------|
| `VLMEncoder._apply_qlora` | encoders/vlm.py | 4-bit base + LoRA via peft |
| `CombinedLoss.forward` | models/loss.py | the 10-term loss |
| `CombinedLoss._calibration` | models/loss.py | differentiable ECE surrogate |
| `CombinedLoss._contrastive` | models/loss.py | same-task margin loss |
| `TaskPairBatchSampler` | data/dataloader.py | co-batches a task's passes |
| `WebAgentDataset._context_text` | data/dataset.py | last-3-steps prompt |
| `Trainer.fit` | train/trainer.py | epochs, val, early stop, ckpt |
| `collect_predictions` / `compute_metrics` | train/trainer.py | eval arrays + metrics |
| `MemoryIndex.query` | memory/index.py | cosine top-3 recovery retrieval |
| `evaluate_all_splits` | eval/evaluate.py | per-split test metrics |

## 6. Test-first ladder (Kaggle cells)
```
6 SMOKE  : 10 loss terms finite, contrastive non-zero, one LoRA weight changes, VRAM<16GB
7 MINI   : Trainer on a tiny subset, 1 epoch -> val failure_f1 prints, CSV written
8 CKPT   : save LoRA+adapter+5 heads as one dict, reload -> identical predictions
9 FULL   : full train (hours, checkpointed) -> early stop -> 3-split eval
```
Target: **val Failure-F1 > 60%**.
