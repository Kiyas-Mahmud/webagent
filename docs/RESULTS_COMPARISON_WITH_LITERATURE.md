# Results in context: our model and agent compared with published work

Prepared 2026-10-04. Model-level results are final (Table 1, validation set).
Agent-level results for our system will be filled in when the paired run
`web-v3` completes (42 tasks × 2 repeats × 2 systems). Published numbers are
taken from the cited papers' tables and should be checked against the original
PDFs before use in the thesis.

**Important reading note.** The published systems were evaluated on different
benchmarks (Mind2Web, MiniWoB, WebVoyager, WebArena, Auto-UI, GAIA) with
different actors (GPT-4o, Claude, 70B open models). No number below is a
head-to-head comparison with ours. The tables are meant to place our results
in the range reported by comparable work, and to compare *relative* gains over
each system's own baseline.

---

## Table A. Step-level failure detection (existing results only; no new runs)

### A1. Our trained assessor on our data (Table 1, validation, seed 42)

| Head | Decision | Measured on | Accuracy | MCC | Macro-F1 | Majority baseline |
|---|---|---|---|---|---|---|
| Step outcome (P1) | step succeeded / failed | 7,861 validation transitions | ≈83 | 0.678 | 0.839 | — |
| Recovery outcome (P1) | recovery succeeded / failed | 1,858 attempted recoveries | 93.5 | 0.852 | 0.925 | — |
| Memory storage (P4) | store this incident? | 7,861 validation cases | 85.8 | 0.695 | 0.846 | 62.0 % acc., 0.383 F1 |

### A2. Published prompted detectors, different data (context only) [1]

| Setting | Model | Training | Accuracy |
|---|---|---|---|
| Textual grounding | GPT-4o (closed) | none | 83.8 |
| Textual grounding | InternVL2-Llama3-76B (open) | none | 82.4 |
| Textual grounding | Gemini-1.5-pro | none | 79.7 |
| Default prompt | GPT-4o | none | 74.4 |
| Default prompt | Gemini-1.5-flash | none | 61.5 |

### A3. Live precision of the deployed detector (web-v2 traces, real websites)

| Measurement | Value |
|---|---|
| Confident failure alarms (P ≥ 0.9) on non-terminal steps of *completed* episodes | 2 of 40 steps; both genuine failures |
| First confident failure in baseline-failed tasks | step 0–2 (arXiv, Debian, EPA, TED, bank holidays); step 9 (IETF) |
| Uncertain alarms suppressed by the 0.9 gate | 9 in 7 episodes; all 7 episodes completed |

### What Table A supports

| Claim | Evidence | Wording |
|---|---|---|
| Assessor well above chance, class-balanced | A1 (MCC 0.678; memory 85.8 % vs. majority 62.0 %) | "substantially above the majority baseline" |
| Reliable recovery verification | A1 (MCC 0.852, 93.5 %) | "reliable recovery-outcome verification" |
| Not a backbone accident | Table B (0.62–0.68 MCC on three backbones) | "consistent across three VLM backbones" |
| 7B open model vs. prompted GPT-4o | A1 vs. A2 | "comparable magnitude on different data" — never "outperforms" |
| Works outside the training distribution | A3 | "on live websites, confident alarms were almost always genuine failures" |

Caveats to state: A2 numbers are from [1]'s own benchmark, A1 from our
validation split, so accuracy is not transferable across rows (class balance
differs; hence MCC and the majority baseline). Table 1 values are single-seed
validation results.

---

## Table B. Backbone comparison on our data (identical data, protocol and seed)

| Backbone | Outcome MCC | Failure Macro-F1 | Failure-type Macro-F1 | Recovery-outcome MCC | Recovery-outcome acc. | Memory MCC |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B | 0.624 | 0.809 | 0.502 | 0.796 | 90.9 % | 0.663 |
| **Qwen2.5-VL-7B (selected)** | **0.678** | **0.839** | **0.542** | **0.852** | **93.5 %** | **0.695** |
| InternVL3.5-8B-HF | 0.641 | 0.820 | 0.541 | 0.840 | 92.9 % | 0.654 |

Seed 42; one selected checkpoint per backbone (registered selection rule:
outcome MCC). Values from Table 1 of the thesis. Single-seed; differences are
descriptive, not tested for significance.

---

## Table C. Agent-level gains from adding failure detection / recovery / memory to a base agent

| System | What is added | Actor | Benchmark | Baseline SR | With system SR | Gain (pts) | Paired test | Held-out tasks |
|---|---|---|---|---|---|---|---|---|
| Multimodal Auto Validation [2] | vision-based validator + self-refine | Agent-E (GPT-4 class) | WebVoyager, 15 sites | 76.2 | 81.2 | +5.0 | no | no |
| BacktrackAgent [3] | verifier, judger, reflector; backtracking | Qwen2-VL (SFT+RL) | Auto-UI (mobile) | 21.6 | 29.7 | +8.1 | no | — |
| BacktrackAgent [3] | same | same | Mobile3M | 34.9 | 43.3 | +8.4 | no | — |
| WebCoach [4] | cross-session memory coach | **Qwen-VL-7B** | WebVoyager | 32.8 | 28.8 – 31.1 | **−1.7 to −4.0** | no | no |
| WebCoach [4] | same | Qwen-VL-32B | WebVoyager | 49.5 | 54.7 – 57.1 | +5.2 to +7.6 | no | no |
| WebCoach [4] | same | Skywork-38B | WebVoyager | 47.3 | 55.5 – 61.4 | +8.2 to +14.1 | no | no |
| Devil's Advocate [5] | anticipatory reflection | GPT-4 | WebArena | 19.8 | 23.5 | +3.7 | no | — |
| OSCAR [6] | state-aware re-planning | GPT-4 | GAIA (vs. FRIDAY) | 22.4 | 28.7 | +6.3 | no | — |
| SkillWeaver [7] | self-discovered skills | GPT-4o | WebArena | 22.6 | 29.8 | +7.2 | no | — |
| ReUseIt [8] | reusable workflow memory | — | 15 task families | 24.2 | 70.1 | +45.9 | no | — |
| **Ours — development split** | learned failure detection + adaptive recovery + experience memory | **Qwen2.5-VL-7B base** | 20 real-website tasks, 2 repeats | *pending* | *pending* | *pending* | **yes (exact sign test)** | no |
| **Ours — held-out split** | same | same | 22 real-website tasks, 2 repeats | *pending* | *pending* | *pending* | **yes** | **yes** |

Observations that already hold:

1. Published gains from adding verification, reflection or recovery to an
   existing agent are typically **+4 to +8 points** of task success.
2. **Small actors can be harmed by advice.** With a 7B actor, WebCoach's memory
   guidance *reduced* success (−1.7 to −4.0 points), while the same guidance
   helped 32B–38B actors. We observed the same effect in our first comparison
   (web-v2: ours 20/38 vs. baseline 27/38); our recovery design v3 was built to
   remove it (timing parity, plain-text facts instead of labels, repeat guard,
   numbered options).
3. None of the cited studies uses a paired design with a significance test, or
   separates development tasks from held-out tasks. Ours does both.

---

## Table D. Why base agents fail: published analyses vs. our baseline traces

| Source | Setting | Failure categories (share) |
|---|---|---|
| AutoWebGLM [9] | web navigation, error analysis | hallucination 44 %, poor graphical recognition 28 %, misinterpretation of task context 20 %, pop-up interruption 8 % |
| WebSuite [10] | 14 atomic web actions | operational actions (click, type, select) 76–85 % success; informational actions (find, filter, fill) 40–44 % success |
| **Ours (web-v2 baseline, 11 failed episodes)** | Browser Use + Qwen2.5-VL-7B base, real sites | **no-progress loops 9 (82 %)**: scrolling past the page end 5, re-clicking/re-typing the same element 4; false "done" at step 0: 1; invalid output format: 1 |

Our dominant failure mode (repeating an ineffective action while believing it
worked) corresponds to the "hallucination" class in [9] and is exactly what a
transition-level failure detector can observe; the two residual modes (false
"done" without an action, invalid output) produce no executed step and are out
of scope for a transition assessor.

---

## Our development checks (not the final evaluation; recovery v3–v3.3, system C only)

| Set | Tasks | Baseline (from web-v2) | Ours |
|---|---|---|---|
| Rescue set | arXiv paper, Debian distrib, EPA, IETF, TED | 0 / 10 | 8 / 21 attempts (IETF, TED, EPA repeatedly; arXiv and Debian never) |
| Harm set | Alan Turing, photosynthesis, bank holidays, serendipity, jQuery download | 10 / 10 | 10 / 10 (no harm) |

The arXiv-paper and Debian-distrib tasks were later found invalid as written
(target not reachable from the goal text / target page name absent from the
site) and were replaced in the v2 task suite.

---

## References (arXiv identifiers)

1. Detecting Pipeline Failures through Fine-Grained Analysis of Web Agents — arXiv:2509.14382
2. Multimodal Auto Validation for Self-Refinement in Web Agents — arXiv:2410.00689
3. BacktrackAgent: Enhancing GUI Agent with Error Detection and Backtracking Mechanism — arXiv:2505.20660
4. WebCoach: Self-Evolving Web Agents with Cross-Session Memory Guidance — arXiv:2511.12997
5. Devil's Advocate: Anticipatory Reflection for LLM Agents — arXiv:2405.16334
6. OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning — arXiv:2410.18963
7. SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills — arXiv:2504.07079
8. ReUseIt: Synthesizing Reusable AI Agent Workflows for Web Automation — arXiv:2510.14308
9. AutoWebGLM: A Large Language Model-based Web Navigating Agent — arXiv:2404.03648
10. WebSuite: Systematically Evaluating Why Web Agents Fail — arXiv:2406.01623

Also in the collected results but not directly comparable (grounding or
end-to-end agents without a detection/recovery ablation): CogAgent
(arXiv:2312.08914), WebGUM (arXiv:2305.11854), WMA world-model agents
(arXiv:2410.13232), WebSight (arXiv:2508.16987).
