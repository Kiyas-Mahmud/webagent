# Architecture Design — 4-Pillar Failure-Aware Web Agent

> **Read with `PROJECT_SPECIFICATION.md` (v2) and `MODEL_TRAINING_PLAN.md`.**
> This document shows EXACTLY how the four pillars connect, what flows between them, what each component outputs, and how training and inference differ. Backbone-agnostic: dual-encoder path is the default; the VLM adapter path is in section 8.

---

## 1. THE WHOLE SYSTEM IN ONE VIEW

```
                              INPUT
   state_before.jpg (256x256)      task text (task + target_desc + domain)
            |                                   |
            v                                   v
  ============================ PILLAR 2 ============================
  |  VISION ENCODER                    TEXT ENCODER                |
  |  SigLIP -> [B,768]                 RoBERTa -> [B,768]          |
  |          \                               /                    |
  |           \                             /                     |
  |        CROSS-ATTENTION FUSION (the novelty)                   |
  |        vision<->text attend, 2 blocks -> fused [B,768]        |
  ================================================================
                              |
                    fused_embedding [B,768]
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
  === PILLAR 1 ===     === PILLAR 3 ===       === PILLAR 4 ===
  FAILURE HEAD          ACTION HEAD            MEMORY HEAD
  - outcome (2)         - action_type (5)      - memory_flag (1)
  - failure_type (4)    - bbox (4)             - recovery_strat (6)
  - confidence (1)                             + retrieval index
  - recovery_strat (6)
       |                      |                      |
       +----------------------+----------------------+
                              |
                              v
                    === DECISION COMBINER ===
              (inference only — rules, not trained)
              normal action OR recovery action
                              |
                              v
                           OUTPUT
              action to execute + diagnosis + recovery
```

---

## 2. PILLAR 2 — UNIFIED MULTIMODAL DECISION (the entry + novelty)

**Role:** turn a screenshot and a task into ONE shared understanding before any decision.

### 2.1 Inputs
```
Vision:  state_before image, resized 256x256x3, normalized by SigLIP processor
Text:    "task_description [SEP] action_target_desc [SEP] website_domain"
         tokenized, max 128 tokens
```

### 2.2 Process
```
vision_emb = SigLIP(image)        -> [B, 768]
text_emb   = RoBERTa(text)[CLS]   -> [B, 768]

CROSS-ATTENTION FUSION (2 transformer blocks):
  block: vision attends to text   (Q=vision, K=V=text)
  block: text attends to vision   (Q=text,   K=V=vision)
  concat(att_vision, att_text) -> [B, 1536]
  LayerNorm -> Linear(1536 -> 768) -> Dropout(0.1)
```

### 2.3 Output
```
fused_embedding [B, 768]   <- this single vector feeds ALL three other pillars
```

### 2.4 What it is trained against (indirectly)
Pillar 2 has no direct label of its own. It learns because the gradients from all four heads (and `visual_diff_score`, `action_target_bbox`) flow back through the fusion. Good fusion = all heads improve together.

### 2.5 Why it matters (paper claim)
Text alone is overoptimistic on complex sites; vision alone misses text-heavy failures (proven by Multimodal Auto-Validation paper). Cross-attention forces the two to agree, so the fused vector is more reliable than either modality alone. Ablation A3 (concat) vs full model proves cross-attention > concatenation.

---

## 3. PILLAR 1 — FAILURE-AWARE RESILIENT AGENT (the brain)

**Role:** decide if the step failed, why, how sure, and what to do about it. **Highest-weighted pillar.**

### 3.1 Input
```
fused_embedding [B, 768]
```

### 3.2 Process
```
trunk = Linear(768->256) -> ReLU -> Dropout(0.3)
```

### 3.3 Four outputs
```
OUT 1  execution_outcome   Linear(256->2)            -> SUCCESS / FAILURE
OUT 2  failure_type        Linear(256->4)            -> NONE / PERCEPTION_ERROR /
                                                        ACTION_MISMATCH / LOOP_DETECTED
OUT 3  failure_confidence  Linear(256->1) -> Sigmoid -> 0.0 .. 1.0
OUT 4  recovery_strategy   Linear(256->6)            -> NONE / RETRY / REPLAN /
                                                        BACKTRACK / ALT_TARGET / ABORT
```

### 3.4 Trained against
```
OUT 1 -> execution_outcome field      (CrossEntropy)
OUT 2 -> failure_type field           (CrossEntropy, class-weighted: LOOP only 7.4%)
OUT 3 -> agent_confidence_before field(MSE)
OUT 4 -> recovery_strategy field      (CrossEntropy)
```

### 3.5 How it connects to other pillars
```
-> tells PILLAR 3 whether to keep the normal action or be overridden
-> sends failure_type + confidence to the DECISION COMBINER
-> shares recovery_strategy prediction with PILLAR 4 (cross-check)
```

---

## 4. PILLAR 3 — ADAPTIVE MULTI-TOOL ORCHESTRATION (the hands)

**Role:** choose the action and where to perform it. Gets overridden by recovery when Pillar 1 says failure.

### 4.1 Input
```
fused_embedding [B, 768]
(+ at inference: failure signal from Pillar 1)
```

### 4.2 Process
```
trunk = Linear(768->256) -> ReLU -> Dropout(0.3)
```

### 4.3 Outputs
```
OUT 1  action_type  Linear(256->5)  -> CLICK / TYPE / SELECT / SCROLL / NAVIGATE
OUT 2  bbox         Linear(256->4)  -> [x, y, w, h]  (normalized 0..1)
```

### 4.4 Trained against
```
OUT 1 -> action_type field      (CrossEntropy, class-weighted: CLICK 83.6%)
OUT 2 -> action_target_bbox     (MSE, MASKED — only where bbox exists, 67,160 rows)
```
> SCROLL/NAVIGATE have NO training data (absent in source). The head supports 5 classes
> but only 3 are learnable. Document this; do not synthetically fill.

### 4.5 How it connects
```
Normal case (Pillar 1 = SUCCESS, confidence high):
   action_type + bbox -> DECISION COMBINER -> executed directly
Failure case (Pillar 1 = FAILURE):
   action_type is OVERRIDDEN by recovery strategy (Pillar 1 + Pillar 4)
```

---

## 5. PILLAR 4 — MEMORY-DRIVEN CORRECTIVE PLANNING (the experience)

**Role:** decide what to remember, and retrieve proven recoveries from the past.

### 5.1 Input
```
fused_embedding [B, 768]
```

### 5.2 Process (trained head)
```
trunk = Linear(768->256) -> ReLU -> Dropout(0.3)
OUT 1  memory_update_flag  Linear(256->1) -> Sigmoid -> store? True/False
OUT 2  recovery_strategy   Linear(256->6)          -> which recovery
```

### 5.3 Trained against
```
OUT 1 -> memory_update_flag field   (BCE)   [True for all 51,036 failures]
OUT 2 -> recovery_strategy field    (CrossEntropy)
```

### 5.4 Retrieval index (inference time, NOT trained)
```
BUILD (after training):
  for every TRAIN row where memory_update_flag=True AND recovery_success=True:
     store { fused_embedding, failure_type, recovery_strategy,
             reflection_text, website_domain }
  start simple: numpy top-k cosine; upgrade to FAISS for production

QUERY (at inference):
  given current fused_embedding:
     find top-3 most similar stored failures (cosine)
     return their recovery_strategy as guidance

EPISODIC PAIRS:
  the 5 passes of the same original_task_id are natural memory pairs
  (pass that failed  <->  pass that recovered)
```

### 5.5 How it connects
```
-> at inference, supplies the proven recovery to the DECISION COMBINER
-> cross-checks Pillar 1's recovery_strategy prediction
```

---

## 6. DECISION COMBINER (inference only — rules, not a trained network)

**Role:** turn the four pillars' outputs into ONE action to execute. Not trained, no weights.

```
INPUTS: outcome, failure_type, confidence (P1)
        action_type, bbox (P3)
        memory retrieval result, recovery_strategy (P1+P4)

RULE 1 (normal):
  outcome=SUCCESS AND confidence>=0.65
  -> execute P3 action directly

RULE 2 (pre-action caution):
  confidence < 0.65 (model unsure BEFORE acting)
  -> query P4 memory -> use retrieved recovery as guidance

RULE 3 (post-action recovery):
  outcome=FAILURE
  -> read failure_type
  -> P4 retrieves proven recovery for similar past failure
  -> P1 recovery head confirms strategy
  -> execute recovery action (override P3)
  -> attempt_counter += 1

RULE 4 (loop guard):
  failure_type = LOOP_DETECTED
  -> force BACKTRACK, clear this step's history

RULE 5 (hard stop):
  attempt_counter >= 3
  -> ABORT, emit structured failure report
```

> For the PAPER: train and report the heads (P1/P3/P4) and their metrics.
> The full Decision Combiner is mainly for the PRODUCTION agent. For the paper,
> reporting recovery_strategy accuracy + memory retrieval accuracy is enough.

---

## 7. OUTPUTS — WHAT THE SYSTEM PRODUCES

### 7.1 Training-time outputs (for loss + metrics)
```
execution_outcome     (2-way logits)
failure_type          (4-way logits)
failure_confidence    (scalar)
recovery_strategy     (6-way logits)   [from P1 and P4]
action_type           (5-way logits)
bbox                  (4 floats)
memory_update_flag    (scalar)
```

### 7.2 Combined loss (one number, trains everything)
```
L_total = 0.25*L_outcome + 0.20*L_failtype + 0.18*L_action
        + 0.12*L_bbox(masked) + 0.10*L_memory + 0.10*L_recovery
        + 0.05*L_confidence
```

### 7.3 Inference-time output (per web step)
```
{
  action_type:        "CLICK",
  bbox:               [x, y, w, h],
  outcome:            "FAILURE",
  failure_type:       "PERCEPTION_ERROR",
  failure_confidence: 0.31,
  recovery_strategy:  "RETRY",
  memory_hit:         true,
  reflection_text:    "target element masked or missing",
  attempt_number:     2
}
```

---

## 8. BACKBONE-AGNOSTIC PATH (for VLMs: Y4, Y5, Y6)

The four heads and the loss NEVER change. Only the front-end changes.

```
DUAL-ENCODER (Y1, Y2, Y3, Y7):
  vision_enc + text_enc -> CROSS-ATTENTION FUSION -> fused[768] -> 4 heads

UNIFIED VLM (Y4 Qwen2.5-VL-0.5B, Y5 Qwen2.5-VL-3B, Y6 InternVL2-2B):
  VLM(image+text) -> pooled[D] -> Adapter Linear(D->768) -> fused[768] -> 4 heads
  (cross-attention SKIPPED — the VLM already fuses vision+text internally)
```

This single difference is the entire backbone-agnostic claim: swap the front-end, keep the four pillars, and the methodology still works.

---

## 9. DATA FLOW SUMMARY (one step, end to end)

```
1. screenshot + task text come in
2. PILLAR 2 encodes both, fuses -> fused[768]
3. fused[768] copied to all three heads
4. PILLAR 1 predicts: failed? type? confidence? recovery?
5. PILLAR 3 predicts: action_type + bbox
6. PILLAR 4 predicts: store? + retrieves proven recovery from index
7. DECISION COMBINER (inference): pick normal action or recovery
8. OUTPUT: one action + full diagnosis
9. (production) Playwright executes it; on ABORT -> report to LLM planner
```

---

## 10. PILLAR -> DATASET FIELD -> OUTPUT MAP (quick reference)

```
PILLAR 1  execution_outcome      -> SUCCESS/FAILURE
          failure_type           -> 4-way diagnosis
          agent_confidence_before-> confidence scalar
          recovery_strategy      -> 6-way recovery

PILLAR 2  state_before           -> vision_emb
          task/target/domain text-> text_emb
          visual_diff_score      -> (supervises fusion quality)
          action_target_bbox     -> (shared with P3)
          => fused_embedding[768]

PILLAR 3  action_type            -> 5-way action
          action_target_bbox     -> bbox regression

PILLAR 4  memory_update_flag     -> store? flag
          recovery_strategy      -> 6-way recovery
          recovery_success       -> index filter
          original_task_id+pass  -> episodic pairs
          reflection_text        -> retrieval context
```

---

## 11. ONE-PARAGRAPH SUMMARY (for the coding agent)

A screenshot and a task string enter Pillar 2, where a vision encoder and a text encoder produce two 768-dim vectors that a cross-attention fusion layer merges into one shared 768-dim embedding (for VLM backbones a single adapter replaces fusion). That one embedding is fed to three heads in parallel: Pillar 1 predicts whether the step failed, the failure type, a confidence score, and a recovery strategy; Pillar 3 predicts the action type and bounding box; Pillar 4 predicts whether to store the experience and retrieves proven recoveries from a similarity index built over past successful recoveries. All heads are trained together by one weighted combined loss. At inference a rule-based Decision Combiner reads the four pillars' outputs and emits a single action — the normal predicted action when the model is confident and the step looks successful, or a memory-guided recovery action when failure is detected — with a hard 3-attempt limit before aborting and reporting back to the LLM planner. For the paper, we train and report the heads and their metrics; the full Decision Combiner is primarily for the production agent.
