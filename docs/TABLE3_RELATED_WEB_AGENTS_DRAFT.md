# Table 3 draft — comparison with related web agents

Built 2026-09-12 from the supplied `Results_1.docx` collection. **Every number
below is copied from a published paper; none is computed here.** Cells the
source document does not contain are marked `N/R` and must be filled from the
paper itself before submission.

Per the agreed metadata: main metric is task success rate; comparison is
**Direct** only when the benchmark matches ours, **Contextual** otherwise;
results from different benchmarks must never be presented as a direct ranking.

---

## ⚠️ Read this before using the table

**1. "SR" does not mean the same thing in every paper.**

The World-Models paper reports both on Mind2Web:

| Method | Cross-Task **Step SR** | Cross-Task **Task SR** |
|---|---:|---:|
| AWM+WMA | 67.0% | **25.4%** |
| MindAct | 36.2% | **2.0%** |

CogAgent's Mind2Web "cross-task SR 62.3" sits in the *step* range, not the task
range. **Placing 62.3 and 25.4 in one column would be wrong.** Each row must
record which quantity it is.

**2. Our MiniWoB numbers are directly exposed.**

The WebGUM paper's table is **MiniWoB++**, the same benchmark family as our
Table 2:

| System | MiniWoB++ SR |
|---|---:|
| CC-Net (SL) | 32.0% |
| WebN-T5 | 48.4% |
| WGE | 64.6% |
| WebGUM (Flan-T5-XL + ViT) | **94.2%** |
| CC-Net (SL+RL) / Human | 93.5% |
| Synapse (GPT-3.5) | **98.5%** |
| **Ours (E0/E2/E3)** | **20.0%** (6/30) |

Published MiniWoB systems reach 32–98.5%; ours is 20%. The task subsets differ
(six families × five resets here, versus the standard 56-task suite), so this is
**contextual, not direct** — but it is the closest published comparison to our
own benchmark and a reviewer will find it. It must be addressed, not omitted.

---

## Table 3 — related web agents

| Agent | Main capability | Benchmark | Backbone | Tasks | Task SR | Recovery result | Avg steps | Comparison |
|---|---|---|---|---:|---:|---|---:|---|
| **WebVoyager** | Multimodal navigation | WebVoyager | GPT-4V | 643 | `N/R` | N/R | N/R | Direct only if we run WebVoyager |
| **WebCoach** (best) | Cross-session memory | WebVoyager | Skywork-38B + Qwen3-8B coach | `N/R` | **0.614** | memory **+0.141** over no-memory (0.473) | 10.2 | Contextual |
| WebCoach (no memory) | — | WebVoyager | Skywork-38B | `N/R` | 0.473 | baseline | 10.7 | Contextual |
| WebCoach (no memory) | — | WebVoyager | GPT-4o | `N/R` | 0.653 | baseline | 10.9 | Contextual |
| WebCoach (no memory) | — | WebVoyager | Qwen-VL-7B | `N/R` | 0.328 | baseline | 16.4 | Contextual |
| **BacktrackAgent** | Error detection + backtracking | Auto-UI | Qwen2-VL | `N/R` | **29.72** | **+8.16** over same-backbone SFT (21.56) | N/R | Contextual |
| **BacktrackAgent** | Error detection + backtracking | Mobile3M | Qwen2-VL | `N/R` | **43.25** | **+8.37** over same-backbone SFT (34.88) | N/R | Contextual |
| **ReUseIt** | Reusable workflows | Own web tasks | `N/R` | 15 families | **70.1 ± 16.4** | task-only 24.2 ± 13.2 → 70.1 | N/R | Contextual |
| **SkillWeaver** | Reusable skills | WebArena | GPT-4o | 5 sites | **29.8** | +32% rel. over no-skills (22.6) | N/R | Contextual |
| SkillWeaver | Reusable skills | WebArena | GPT-4o-mini | 5 sites | 14.1 | 9.2 → 14.1 | N/R | Contextual |
| **OSCAR** | State-aware reasoning + re-planning | OSWorld | `N/R` | `N/R` | **24.5** | N/A | N/R | Contextual |
| OSCAR | " | GAIA | `N/R` | `N/R` | 28.7 | N/A | N/R | Contextual |
| OSCAR | " | AndroidWorld | `N/R` | `N/R` | 61.6 | N/A | N/R | Contextual |
| **Devil's Advocate (AR)** | Anticipatory reflection | `N/R` | `N/R` | `N/R` | **23.5%** | +3.7 over Plan+Act (19.8%) | 6.39 first-trial / 7.07 last-trial | Contextual |
| **Self-Refine (Vision)** | Multimodal auto-validation | WebVoyager | `N/R` | `N/R` | **81.24%** | +5.0 over Agent-E text (76.20%) | N/R | Contextual |
| **AWM + WMA** | World model / env. dynamics | Mind2Web | `N/R` | `N/R` | **25.4%** task SR (67.0% step SR) | N/A | N/R | Direct if we run Mind2Web |
| MindAct | Baseline | Mind2Web | `N/R` | `N/R` | 2.0% task SR (36.2% step SR) | N/A | N/R | Direct if we run Mind2Web |
| CogAgent | Visual GUI agent | Mind2Web | CogAgent 18B | `N/R` | 62.3 cross-task *(step-level)* | N/A | N/R | Direct if we run Mind2Web |
| SeeClick | GUI grounding | Mind2Web | Qwen-VL 9.6B | `N/R` | 23.7 cross-task | N/A | N/R | Direct if we run Mind2Web |
| Qwen-VL | General VLM | Mind2Web | 9.6B | `N/R` | 12.6 cross-task | N/A | N/R | Direct if we run Mind2Web |
| AutoWebGLM | Web navigation | Mind2Web | 6B | `N/R` | 66.4 cross-task | N/A | N/R | Direct if we run Mind2Web |
| **Our WebAgent** | **Failure recovery + memory** | Our MiniWoB protocol | Qwen2-VL-2B | 30 pairs | **20.0%** (6/30) | **E2−E1 +0.20** (p=0.031, Holm 0.0625); **E3−E2 = 0** | 1.9 executor steps | Proposed system |

---

## Companion: failure-detection comparison (different metric — keep separate)

Our strongest result is not a task success rate, so it needs its own row set.
From *Detecting Pipeline Failures through Fine-Grained Analysis of Web Agents*:

| System | Setting | Failure-detection accuracy |
|---|---|---:|
| GPT-4o | Textual grounding | **83.79** |
| GPT-4o | Default | 74.44 |
| InternVL2-Llama3-76B | Textual grounding | 82.36 |
| Gemini-1.5-flash | Textual grounding | 80.90 |
| Claude 3.5 Sonnet | Textual grounding | 76.07 |
| GPT-4o-mini | Default | 61.25 |
| **Ours — PC-01 (2B)** | — | **80.4 balanced acc** (MCC 0.624) |

Different dataset, so **contextual**. But the sentence it supports is strong and
defensible: *a 2B model reaches failure-detection accuracy in the same band as
GPT-4o-, Gemini- and Claude-based pipelines that are orders of magnitude larger.*

---

## What this collection tells us about our own position

**1. Every competing mechanism reports a positive gain; ours currently does not.**

| Paper | Mechanism | Gain |
|---|---|---:|
| BacktrackAgent | error detection + backtracking | **+8.2** |
| WebCoach | cross-session memory | **+0.141** |
| SkillWeaver | reusable skills | **+32% rel.** |
| ReUseIt | reusable workflows | 24.2 → **70.1** |
| Devil's Advocate | anticipatory reflection | **+3.7** |
| Self-Refine | multimodal validation | **+5.0** |
| **Ours** | recovery (E2−E1) | **+0.20, not significant** |
| **Ours** | memory (E3−E2) | **0.00** |

This is the single most exposed point in the paper. It needs the grounding
evidence (median IoU 0.000 on our own validation) as its explanation, stated up
front rather than discovered by a reviewer.

**2. BacktrackAgent is the nearest neighbour and must be positioned against
explicitly.** Same idea, same backbone family (Qwen2-VL), and it works.

**3. Vision-only models score low on this task class, which contextualises our
action accuracy.** Qwen-VL at 9.6B reaches 10.2 average on Mind2Web and SeeClick
20.9, while HTML-based HTML-T5-XL reaches 66.9. Our 0.422 action accuracy on a
2B vision model is inside the expected band for the modality, not an anomaly.

**4. Mind2Web is the only route to a Direct comparison.** Nine rows above are
Mind2Web; none of the others matches our benchmark. Running our architecture on
the official Multimodal Mind2Web release converts those rows from contextual to
direct — the single highest-value remaining experiment for external validity.

---

## Cells that still need filling from the papers

| Row | Missing |
|---|---|
| WebVoyager | task SR, recovery, avg steps |
| WebCoach | task count |
| BacktrackAgent | task count, avg steps |
| ReUseIt | backbone, avg steps |
| SkillWeaver | task count, avg steps |
| OSCAR | backbone, task count, avg steps |
| Devil's Advocate | benchmark, backbone, task count |
| Self-Refine | backbone, task count |
| All Mind2Web rows | task counts, action budgets |

The agreed metadata requires benchmark, backbone, task count and action budget
for every row. Those gaps must be closed from the source papers, not inferred.
