# External-agent failure-and-recovery evaluation — plan

Frozen before any agent is downloaded or run. 2026-09-12.

## 1. The question

Not *"can the agent click the right thing"*. That is a separate, smaller issue.
The question is the thesis's own:

> **When an agent fails, can it tell that it failed, say why, choose a recovery,
> and know whether that recovery will work?**

Existing GUI agents have no trained mechanism for any of this. This evaluation
measures what they achieve zero-shot on our data, against PC-01's trained heads
on the identical rows.

**No bounding box is used anywhere in this evaluation.** Grounding quality is
irrelevant to every question below.

## 2. Evaluation sample — recovery-enriched, not proportional

Built from `split_val.json` only. The locked test split is never opened.

| Part | Rows | Why |
|---|---:|---|
| Every row with a real recovery attempt | **1,858** | Carries strategy + whether it worked |
| Matched non-recovery rows, stratified by `outcome_label` × `failure_type_4` | **1,858** | Supplies successes and un-recovered failures |
| **Total** | **3,716** | |

Deliberately **enriched**, not the natural distribution. Consequence, declared in
advance: raw accuracy on this sample is **not** comparable to published
full-validation accuracy. Reported metrics are MCC, macro-F1 and balanced
accuracy, and the majority baseline is recomputed on this sample. PC-01 is
scored on the identical rows — its existing validation numbers are not reused.

Selection is seeded and hash-committed before any agent runs.

## 3. The five questions

Every system receives exactly the same input: `state_before`, `state_after`,
`task_description`, `website_domain`, and the action type that was attempted
(known to any agent at that moment — it is context, not the answer).

| # | Question | Label | Classes |
|---|---|---|---|
| Q1 | Did this step fail? | `outcome_label` | SUCCESS / FAILURE |
| Q2 | What kind of failure? | `failure_type_4` | NONE / PERCEPTION_ERROR / ACTION_MISMATCH / LOOP_DETECTED |
| Q3 | How should it recover? | `recovery_strategy` | NONE / RETRY / REPLAN / BACKTRACK / ALTERNATIVE_TARGET |
| Q4 | Will that recovery succeed? | `recovery_success` | on the 1,858 attempted rows only |
| Q5 | Worth remembering? | `memory_update_flag` | true / false |

**Q3 gives the agents our exact strategy vocabulary.** This is deliberate and is
the fairer choice: they are not penalised for phrasing, only for the decision.

## 4. Systems compared

| System | Size | Role |
|---|---:|---|
| **ShowUI-2B** | 2B | Purpose-built GUI agent, **same size as PC-01** — the fair head-to-head |
| **OS-Atlas-Base-7B** | 7B | Larger purpose-built GUI agent |
| **Qwen2.5-VL-7B-Instruct** | 7B | Strong general VLM — the realistic "just prompt a good model" competitor |
| **PC-01 (ours)** | 2B | Trained four-pillar heads |
| Majority baseline | — | Recomputed on this exact sample |

SeeClick is **dropped**: it is a pure coordinate model on the older Qwen-VL
architecture (loading risk under transformers 4.57), and on reasoning questions
it would be a strawman. Qwen2.5-VL-7B replaces it and is the harder opponent —
if PC-01 at 2B beats a 7B general VLM on failure detection, that is the result
worth reporting.

## 5. Fairness rules, fixed in advance

- Identical inputs, identical question wording, identical answer vocabulary.
- Agents are **zero-shot**; PC-01 is **fine-tuned**. This is stated in the paper.
  The honest claim is *"off-the-shelf agents lack this capability"*, never
  *"we beat them at what they were built for"*.
- Unparseable agent output counts as **wrong**, never silently retried or
  re-prompted for a better answer. Parse failures are reported per system.
- One prompt per question per system. No prompt search against the labels.
- The locked test split stays unread throughout.

## 6. Expected outcome, written down now

PC-01 is expected to win Q1, Q2, Q4 and Q5 clearly — they correspond to trained
heads (outcome MCC 0.624, failure-type macro-F1 0.502, recovery-outcome MCC
0.796, memory macro-F1 0.830) and the agents have no such mechanism.

**Q3 is the honest risk.** PC-01's recovery-strategy accuracy is 0.535 against a
0.764 majority baseline — *below* majority. A strong general VLM may beat it.
That result is reported as measured.

## 7. Secondary table, not the headline

Click accuracy on the 1,488 rows with a valid box, scored as
*"does the predicted point fall inside the true box"* — the standard GUI
grounding metric, with PC-01's box centre used as its point. Reported as
context, with the dataset limitation stated: the task description never names
the target element, and `action_target_desc` is absent from the exported data.

## 8. Execution order

1. Re-freeze the recovery-enriched sample (3,716 rows) with the recovery labels.
2. Download ShowUI-2B, OS-Atlas-Base-7B, Qwen2.5-VL-7B (~28 GB).
3. Build the harness; **pilot on 200 rows** to prove parsing and timing.
4. Full run, one system at a time, persisting every raw response.
5. Score PC-01 on the identical rows.
6. Report all five questions, plus parse-failure rates and the majority baseline.

Rough cost: ~3,716 rows × 3 systems ≈ 11,000 calls with two images each;
6–12 GPU hours. The pilot decides whether that is worth committing.
