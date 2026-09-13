# Why E1 scores zero — diagnosis and a declared negative result, 2026-09-12

E1 completed 0/30 in the 120-episode evaluation and 0/4 in every development
run. This document establishes why, corrects an earlier wrong claim, and records
a declared intervention that **did not work** and is therefore left disabled.

## 1. Correction: NAVIGATE is not an untrained class

An earlier note in this workstream claimed NAVIGATE had no training data,
quoting [PROJECT_SPECIFICATION.md:194](PROJECT_SPECIFICATION.md). **That line
describes the historical 70,965-row synthetic corpus**, which `AGENT.md` and
`project_progress.md` both mark as superseded. The registered Gold v2.8 run
tells a different story:

| | CLICK | NAVIGATE | PRESS_KEY | SCROLL | SELECT | TYPE |
|---|---:|---:|---:|---:|---:|---:|
| Train rows | 3,851 | **4,770** | 3,677 | 4,023 | 3,906 | 3,880 |
| Validation rows | 1,287 | **1,499** | 1,194 | 1,293 | 1,292 | 1,296 |

NAVIGATE is the **largest** action class. Action class weights are 0.88–1.10,
essentially uniform. Any proposal to drop NAVIGATE because it is "untrained" is
factually wrong, and dropping it because it produces poor results would be a
method chosen by its outcome. It was not done.

## 2. It is not a loading, quantization or bias defect

- Training used QLoRA with `load_in_4bit: true` and `dtype: float16`; the
  runtime loads the identical 4-bit path. **Quantization is matched.**
- The action head's bias does not favour NAVIGATE: SELECT has the largest bias
  (0.116), NAVIGATE only 0.045, PRESS_KEY is negative.
- Fed a zero embedding the head is almost uniform (0.143–0.182 across six).

## 3. The action head works, and is confidently half-right

| | CLICK | TYPE | SELECT | SCROLL | NAVIGATE | PRESS_KEY |
|---|---:|---:|---:|---:|---:|---:|
| Zero embedding (prior) | 0.163 | 0.173 | 0.182 | 0.171 | 0.168 | 0.143 |
| Real MiniWoB page | **0.310** | 0.021 | 0.021 | **0.297** | **0.328** | 0.022 |

This is far from the prior. The head confidently rejects the three text-type
actions — 17% down to 2% — and splits the remainder across the three
pointer-type actions. It also reads the screen: on `enter-text`, TYPE rises
0.021 → 0.078 and SELECT 0.021 → 0.062.

Its in-domain quality is real but weak: **action accuracy 0.422 against a 0.191
majority baseline**, macro-F1 0.327, MCC 0.341, on a balanced six-class problem.

What it cannot do is separate CLICK from SCROLL from NAVIGATE. NAVIGATE wins by
about 1.5 percentage points, and MiniWoB has no permitted URL, so every proposal
is rejected and the loop guard ends the episode.

## 4. Declared intervention: executable-action selection

Because the generated interfaces already receive per-control `supported_actions`
while the trained policy does not, the policy was the only system permitted to
choose an action the environment cannot perform. That is a fairness gap, not a
prediction, so selection was restricted to the action classes registered as
executable for the exact current observation — the same oracle-blind evidence
recovery validation already uses. The learned distribution is never modified and
is still logged in full; only the set the argmax ranges over narrows.

Implemented behind `executable_action_selection`, **off by default**, and frozen
as profile `miniwob-development-v7` before any episode ran.

## 5. Result: the intervention failed on its own terms

It did what it was designed to do — E1 proposed CLICK instead of NAVIGATE on all
four tasks. **E1 still scored 0/4, rejected at the identical stage.**

The reason is the grounding head, not the action class:

| Task | Predicted box centre | Best IoU with any control | Centre lands on a control |
|---|---|---:|---|
| click-button | (0.571, 0.135) | **0.000** | No |
| enter-text | (0.267, 0.095) | **0.000** | No |
| click-test | (0.281, 0.110) | **0.000** | No |
| click-button-sequence | (0.251, 0.196) | **0.000** | No |

Zero overlap with every control, on every task. The pattern is systematic: the
model predicts y between 0.073 and 0.196, while the page's controls begin at y
between 0.243 and 0.336. That upper band is MiniWoB's yellow instruction banner.
**E1 is pointing at the task description, not at the page.**

It also made E2 and E3 worse, and the mechanism is worth recording. The failed
action's type is part of the recovery planner's causal context. Changing it from
NAVIGATE to CLICK changed that context, and the frozen base generator responded
with PRESS_KEY instead of CLICK — losing `click-button`, which E2 and E3 had
been completing:

| `click-button` | development-v6 | development-v7 |
|---|---|---|
| E1 normal action | NAVIGATE → rejected | CLICK → rejected |
| E2 recovery proposal | CLICK `o2:c0` → **reward 1.0** | PRESS_KEY ×2 → reward 0.0 |
| E2 / E3 result | 2/4 · 2/4 | 1/4 · 1/4 |

## 6. Decision

`executable_action_selection` stays **disabled**. The reason is not that the
score fell — it is that the intervention does not address the actual bottleneck
(E1 remains 0/4, rejected at the same stage for the same reason) while
perturbing another system's inputs. `miniwob-development-v6` remains the
recommended profile. The v7 package is retained as the recorded evidence of a
declared hypothesis that was tested and rejected.

## 7. What this means for Table 2

**E1's failure is a grounding-transfer failure (P3), not an action-class
failure.** The policy was trained on Mind2Web-derived real web pages; on
MiniWoB's 160×210 synthetic panes its box predictions land in the instruction
banner with zero overlap with any control. The action head still carries usable
signal; the grounding head does not transfer at all.

This has to be stated plainly in the thesis, because it changes what the
headline contrast measures. **E2−E1 is not "recovery versus a working trained
policy."** It is "recovery through an observed-control interface versus a policy
whose visual grounding does not transfer to this benchmark." The recovery
increment is real and executed, but the baseline it beats is one that cannot
ground an action at all.

Honest options, none of which are free:

1. **Report it as-is**, with the evidence in this document. Cheapest and fully
   defensible, but the zero-scoring baseline weakens the claim.
2. **Give E1 the same named-target interface the recovery path uses**, so it
   selects a control rather than predicting pixels. This makes E1 a genuinely
   comparable baseline — but it changes what E1 *is*, and the trained grounding
   head would no longer be exercised, which is a P3 claim in the thesis.
3. **Evaluate on a benchmark closer to the training domain**, where the
   grounding head has a chance. The largest change, and out of scope here.

None of these should be chosen by which produces the better number. Option 1 is
the only one that requires no new method decision.
