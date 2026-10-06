# Results in context: our model and agent compared with published work

Prepared 2026-10-04. Model-level results are final (Table 1, validation set).
Agent-level results are from the completed paired run `web-v3`
(2026-10-04 to 2026-10-06; 42 tasks × 2 repeats × 2 systems = 168 episodes). Published numbers are
taken from the cited papers' tables and should be checked against the original
PDFs before use in the thesis.

**Important reading note.** The published systems were evaluated on different
benchmarks (Mind2Web, MiniWoB, WebVoyager, WebArena, Auto-UI, GAIA) with
different actors (GPT-4o, Claude, 70B open models). No number below is a
head-to-head comparison with ours. The tables are meant to place our results
in the range reported by comparable work, and to compare *relative* gains over
each system's own baseline.

---

## Table A. Step-level failure detection: our trained 7B assessor vs. published models

| Model | Params | Access | Training for this task | Evaluation data | Accuracy (%) | MCC | Macro-F1 |
|---|---|---|---|---|---|---|---|
| GPT-4o (textual grounding) [1] | undisclosed | closed API | none (prompted) | failure traces of [1] | 83.8 | — | — |
| InternVL2-Llama3-76B-AWQ (TG) [1] | 76 B | open | none (prompted) | same | 82.4 | — | — |
| Gemini-1.5-pro (TG) [1] | undisclosed | closed API | none (prompted) | same | 79.7 | — | — |
| Claude 3.5 Sonnet (TG) [1] | undisclosed | closed API | none (prompted) | same | 76.1 | — | — |
| GPT-4o (default prompt) [1] | undisclosed | closed API | none (prompted) | same | 74.4 | — | — |
| GPT-4o-mini (TG) [1] | undisclosed | closed API | none (prompted) | same | 73.2 | — | — |
| Gemini-1.5-flash (default) [1] | undisclosed | closed API | none (prompted) | same | 61.5 | — | — |
| **Ours: Qwen2.5-VL-7B + trained heads** | **7 B** | **open, 4-bit** | **QLoRA + heads, Web-Gold-40K** | **Web-Gold-40K validation (7,861 transitions)** | **≈83** | **0.678** | **0.839** |

**Reading.** A 7B open model trained on our dataset and run in 4-bit reaches
step-level failure-detection accuracy of the same magnitude as prompted GPT-4o
(83.8 %) and a 76B open model (82.4 %), at about one-tenth of the size and
without API access. The two groups are measured on different data ([1]'s
failure-trace benchmark vs. our validation split), so accuracy is not directly
transferable; MCC and Macro-F1 are reported for our model for that reason. The
recovery-outcome head (MCC 0.852, 93.5 %) and memory-storage head (MCC 0.695,
85.8 % vs. 62.0 % majority) have no published counterpart in [1].

Live precision of the deployed detector (web-v2, real websites): 2 confident
alarms (P ≥ 0.9) in 40 non-terminal steps of completed episodes, both genuine
failures; first confident alarm at step 0–2 in five of six baseline-failed
tasks.

---

## Table B. Controlled comparison on our data: trained assessor vs. agent-style prompted assessment (same model, same cases)

Task 1 pilots, frozen validation subset: 240 interaction cases and 120 linked
recovery cases, identical labels, phase definitions and scoring rules
(audited exports in `results/task1_qwen25_dual_v1/`, `results/task1_internvl_dual_v2/`,
`results/task1_qwen_backend_v1/`). The prompted rows run the *same frozen
decoder* with the assessment instructions adapted from three public agents
(Browser Use, Agent S2, WebVoyager); they are component adaptations, not the
native agents. Invalid or unparsable outputs count as wrong (all-case accuracy).

### B1. Step-outcome assessment (240 cases)

| Assessor | Backbone | Trained? | MCC | Balanced acc. | Macro-F1 | All-case acc. | Valid outputs |
|---|---|---|---|---|---|---|---|
| Browser Use-style prompt | Qwen2.5-VL-7B base | no | 0.000 | 50.0 % | 0.237 | 11.3 % | 87 / 240 |
| Agent S2-style prompt | Qwen2.5-VL-7B base | no | 0.000 | 50.0 % | 0.183 | 15.8 % | 170 / 240 |
| WebVoyager-style prompt | Qwen2.5-VL-7B base | no | — | — | — | 0.0 % | 0 / 240 |
| Browser Use-style prompt | Qwen2.5-VL-7B + our LoRA decoder | partly | 0.151 | 53.4 % | 0.314 | 16.3 % | 107 / 240 |
| Agent S2-style prompt | Qwen2.5-VL-7B + our LoRA decoder | partly | 0.000 | 50.0 % | 0.238 | 12.5 % | 96 / 240 |
| WebVoyager-style prompt | Qwen2.5-VL-7B + our LoRA decoder | partly | −0.276 | 44.6 % | 0.219 | 10.4 % | 89 / 240 |
| Browser Use-style prompt | Qwen2-VL-2B base | no | 0.000 | 50.0 % | 0.206 | 24.6 % | 228 / 240 |
| Agent S2-style prompt | Qwen2-VL-2B base | no | 0.000 | 50.0 % | 0.199 | 24.6 % | 238 / 240 |
| WebVoyager-style prompt | Qwen2-VL-2B base | no | 0.000 | 50.0 % | 0.200 | 22.9 % | 220 / 240 |
| Trained heads | InternVL3.5-8B-HF | **yes** | 0.558 | 80.8 % | 0.768 | 80.4 % | 240 / 240 |
| **Trained heads (selected)** | **Qwen2.5-VL-7B** | **yes** | **0.649** | **84.7 %** | **0.820** | **85.4 %** | **240 / 240** |

### B2. Recovery-outcome assessment (120 cases)

| Assessor | Backbone | Trained? | MCC | Macro-F1 | All-case acc. | Valid outputs |
|---|---|---|---|---|---|---|
| Browser Use-style prompt | Qwen2.5-VL-7B base | no | −0.122 | 0.400 | 16.7 % | 30 / 120 |
| Agent S2-style prompt | Qwen2.5-VL-7B base | no | −0.104 | 0.331 | 37.5 % | 91 / 120 |
| WebVoyager-style prompt | Qwen2.5-VL-7B base | no | — | — | 0.0 % | 0 / 120 |
| Browser Use-style prompt | Qwen2-VL-2B base | no | 0.000 | 0.329 | 45.8 % | 112 / 120 |
| Agent S2-style prompt | Qwen2-VL-2B base | no | 0.000 | 0.333 | 50.0 % | 120 / 120 |
| WebVoyager-style prompt | Qwen2-VL-2B base | no | 0.000 | 0.335 | 45.8 % | 109 / 120 |
| Trained heads | InternVL3.5-8B-HF | **yes** | 0.884 | 0.942 | 94.2 % | 120 / 120 |
| **Trained heads (selected)** | **Qwen2.5-VL-7B** | **yes** | **0.841** | **0.916** | **91.7 %** | **120 / 120** |

### B3. Full-validation backbone comparison (7,861 / 1,858 cases; Table 1)

| Backbone | Outcome MCC | Failure Macro-F1 | Failure-type Macro-F1 | Recovery-outcome MCC | Recovery acc. | Memory MCC |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B | 0.624 | 0.809 | 0.502 | 0.796 | 90.9 % | 0.663 |
| **Qwen2.5-VL-7B (selected)** | **0.678** | **0.839** | **0.542** | **0.852** | **93.5 %** | **0.695** |
| InternVL3.5-8B-HF | 0.641 | 0.820 | 0.541 | 0.840 | 92.9 % | 0.654 |

**Reading.** On identical cases, every agent-style prompted assessor is at or
below chance (MCC ≤ 0.15, often exactly 0), and the prompted decoders fail to
produce a valid structured verdict for a large share of cases (WebVoyager-style
prompting on the Qwen2.5 base: none). The trained heads reach MCC 0.65 (step
outcome) and 0.84 (recovery outcome) with valid output on every case. This is
the controlled evidence that the gain comes from training on Web-Gold-40K, not
from the backbone: the same frozen Qwen2.5-VL-7B goes from MCC 0.00 (prompted)
to 0.65 (trained heads). The three trained backbones agree within 0.06 MCC on
the full validation set (B3). Note the subset values (B1/B2: 0.649 / 0.841)
differ slightly from the full-validation values (B3: 0.678 / 0.852) because
they are computed on 240 / 120 selected cases; both are reported as such.

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
| **Ours — development split** | learned failure detection + adaptive recovery + experience memory | **Qwen2.5-VL-7B base** | 20 real-website tasks, 2 repeats (40 pairs) | 70.0 | 87.5 | **+17.5** (helped 8, hurt 1, p = 0.039) | **yes (exact sign test)** | no |
| **Ours — held-out split** | same | same | 22 real-website tasks, 2 repeats (44 pairs) | 75.0 | 84.1 | **+9.1** (helped 5, hurt 1, p = 0.219) | **yes** | **yes** |
| **Ours — all tasks** | same | same | 42 tasks, 84 pairs | 72.6 | 85.7 | **+13.1** (helped 13, hurt 2, p = 0.007) | **yes** | mixed |

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
