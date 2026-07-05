# Data Ask — Recovery Outcomes (to enable Pillar 4's recovery head)

> For the data-collection team. Goal: collect enough real `recovery_success` examples
> to train the recovery-outcome head. Today it is **disabled** because the label has
> only **4 "True"** examples in v12 — untrainable.

## What `recovery_success` means
When the agent's action **fails**, it tries a **recovery action** (retry / go back /
click a different element). `recovery_success` = did that recovery **reach the goal**?
(True / False)

## Why it's blocked now
v12 `recovery_success`: non-null on ~285 rows, **True on only 4**. A yes/no head can't
learn from 4 positives → we set its loss weight to 0 (paused). Pillar 4's other output
(`memory_update_flag`) is already trained; only the recovery-outcome part waits on data.

## What to collect
1. **Capture the full sequence, not just the first failure:**
   `action → (if it fails) → recovery action → result`.
2. **After the recovery, record the outcome** with the same url-oracle already in use:
   - goal reached → `recovery_success = True`
   - not reached → `recovery_success = False`
   - **never leave it `null`** when a recovery was attempted.
3. **Get enough of BOTH classes:** aim for **~100+ True and ~100+ False** recoveries
   (today: 4 True). Design some tasks where recovery *can* succeed — e.g. click the wrong
   link → backtrack → click the correct link → success.
4. **Also set** `recovery_attempted = True` and the real `recovery_strategy` used
   (RETRY / REPLAN / BACKTRACK / ALTERNATIVE_TARGET).
5. **Needs multi-step trajectories** — the step *after* a failure. v12 already has up to
   3 steps/task; extend so failed steps get a recorded recovery + its result.

## Honesty rules (non-negotiable — same as all labels)
- `recovery_success` = the **real observed** result of the recovery. **Never** derive it
  from the outcome or any other field.
- It is a **LABEL** (the model predicts it), **never fed as model input**.

## One-line summary
> After a failed action, let the agent try a recovery, then record
> `recovery_success = True/False` (never null). We need ~100+ successful recoveries
> (have 4). Keep it a label, never an input.

## Then — what the ML side does (one switch, no code rewrite)
In `configs/backbones/qwen2vl_2b_gold.yaml`:
```yaml
loss:
  recovery_outcome: 0.09   # was 0.0 -> head trains once data has enough True/False
```
Re-run → the recovery-outcome head trains → **Pillar 4 fully covered.** (If still
imbalanced, add a BCE `pos_weight` — a one-line tweak.)

## Related
See the broader collection spec (task diversity, harder "page-changes-but-wrong"
failures, confidence/memory labels) discussed with the team. This doc covers only the
recovery-outcome gap.
