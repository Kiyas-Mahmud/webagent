# WebFARM: full results report

Thesis: *A multimodal framework for failure detection, adaptive recovery, and
experience memory in autonomous web agents*. Framework name: **WebFARM**
(Failure-Aware Recovery and Memory). Report date: 2026-10-15.

This document collects every result of the study in one place: what was
measured, the numbers, what they mean, and where the raw evidence lives. All
numbers are recomputed from the raw run files by the scripts listed in
Section 9; the figures assert them against the tables, so text, tables and
figures agree.

Companion files:
[Results_Comparison_Handout.pdf](Results_Comparison_Handout.pdf) (Tables 1–20 with notes) ·
[figures_v2/README.md](figures_v2/README.md) (figures and captions) ·
[TASK2_FAILURE_AWARE_AGENT_METHOD.md](TASK2_FAILURE_AWARE_AGENT_METHOD.md) (method).

---

## 1. Summary

1. **The trained assessor works; prompting does not.** WebFARM's Qwen2.5-VL-7B
   heads, trained on Web-Gold-40K, judge "did this step fail?" with MCC 0.678
   on the full validation set (0.649 on the frozen 240-case subset). The same
   model, untrained and prompted the way three public agents check themselves,
   scores MCC 0.00 or gives no readable answer.
2. **WebFARM improves two different open-source agents without any change to
   its model or settings.** With the same frozen 7B actor and 42 real-website
   tasks (84 paired episodes per agent):
   - Browser Use: 72.6 % → **85.7 %** (+13.1 points; 13 helped, 2 hurt; p = 0.007)
   - SeeAct: 61.9 % → **82.1 %** (+20.2 points; 21 helped, 4 hurt; p = 0.0009)
3. **It fixes failures and rarely breaks successes.** Rescue rate 57 % and 66 %
   of the agents' failures; harm rate 3 % and 8 % of their successes.
4. **It transfers.** The recovery layer was designed while watching Browser
   Use on 20 development tasks. On the 22 held-out tasks it still adds +9.1
   (Browser Use) and +22.7 points (SeeAct, p = 0.013), and SeeAct was never
   seen during design.
5. **With WebFARM the weaker agent catches up.** SeeAct + WebFARM (82.1 %) and
   Browser Use + WebFARM (85.7 %) are statistically indistinguishable
   (p = 0.65), and SeeAct + WebFARM completes more tasks than Browser Use alone.
6. **It is cheaper, not more expensive.** Episodes are 20 % (Browser Use) and
   32 % (SeeAct) shorter on average, because the loops that consume the step
   budget are cut.

---

## 2. Study design

### 2.1 Components

| Part | What it is |
|---|---|
| Assessor (P1) | Qwen2.5-VL-7B-Instruct with QLoRA adapters and task heads, trained on Web-Gold-40K (checkpoint `71f867cc…`, PC-02, epoch 0, seed 42). Inputs: screenshot before and after an action, task text, action type. Outputs used live: P(step failed), recovery succeeded?, store in memory? |
| Recovery (v3.3) | After a confident failure (P ≥ 0.9): a plain-text note with page facts (goal-matching links, search box, end of page, unsubmitted field), numbered recovery options (open goal link, search the site, press Enter, scroll to top, go back, 0 = decide myself), and a repeat guard that blocks the exact failed action on the same page. The agent still chooses every action. |
| Experience memory (P4) | The memory head decides which incidents to store; similar past incidents are shown in the note. |
| Agent adapters | Browser Use 0.13.10 (system C) and SeeAct 0.2.9.0 (system T). Same assessor, gate, note text, options, guard and memory; only element identity, note position and option execution differ. |

### 2.2 Systems compared

| Id | System | Run | Dates |
|---|---|---|---|
| A | Browser Use alone | web-v3 | 2026-10-04 to 10-06 |
| C | Browser Use + WebFARM | web-v3 | 2026-10-04 to 10-06 |
| S | SeeAct alone | web-v4-seeact | 2026-10-11 to 10-13 |
| T | SeeAct + WebFARM | web-v5-seeact-ours | 2026-10-14 |

### 2.3 Tasks, budget, scoring

| Item | Value |
|---|---|
| Tasks | 42 real-website tasks (`configs/eval/task2/web_tasks_v2.json`) on 22 sites from the Web-Gold-40K domains: 20 development tasks (inspected while designing recovery) and 22 held-out tasks (written and verified before any run, never inspected) |
| Repeats | 2 per task per system → 84 paired episodes per system |
| Actor | Qwen2.5-VL-7B-Instruct, frozen, 4-bit, greedy, 512 new tokens, for every system |
| Budget | 15 agent steps, 2700 s wall clock |
| Completion | environment rule: the current page equals the target page (host + path), never shown to the agent |
| Statistics | completion rate with Wilson 95 % CI; exact two-sided sign test on discordant pairs (pairs = same task and repeat) |
| Infrastructure | reset failures (DNS outage, page-load timeouts) and episodes whose observations were browser error pages were set aside and rerun under a rule fixed before looking at outcomes |

---

## 3. Task 1: the assessor

### 3.1 Backbones trained on Web-Gold-40K (full validation set)

| Backbone | Step failure MCC | Step failure macro-F1 | Recovery worked? MCC | Recovery worked? acc. | Worth storing? MCC |
|---|---|---|---|---|---|
| Qwen2-VL-2B | 0.624 | 0.809 | 0.796 | 90.9 % | 0.663 |
| **Qwen2.5-VL-7B (selected)** | **0.678** | **0.839** | **0.852** | **93.5 %** | **0.695** |
| InternVL3.5-8B-HF | 0.641 | 0.820 | 0.840 | 92.9 % | 0.654 |

7,861 validation interactions, 1,858 recovery cases, seed 42; one checkpoint
per backbone chosen by a rule fixed in advance. Qwen2.5-VL-7B is best on every
column and is used in WebFARM. Its failure-type head (macro-F1 0.542) and
recovery-strategy head (0.493) are weaker, so the live system only logs them;
its action head (macro-F1 0.318) is too weak to act, which is why WebFARM
leaves acting to an existing agent.

### 3.2 Trained vs prompted on identical cases (frozen 240 / 120 cases)

| How the step is judged | Model | MCC (step) | Readable answers | MCC (recovery) | Readable answers |
|---|---|---|---|---|---|
| Browser Use-style prompt | Qwen2.5-VL-7B | 0.000 | 87 / 240 | −0.122 | 30 / 120 |
| Agent S2-style prompt | Qwen2.5-VL-7B | 0.000 | 170 / 240 | −0.104 | 91 / 120 |
| WebVoyager-style prompt | Qwen2.5-VL-7B | — | 0 / 240 | — | 0 / 120 |
| Prompts on Qwen2-VL-2B (3 styles) | Qwen2-VL-2B | 0.000 | 220–238 / 240 | 0.000 | 109–120 / 120 |
| WebFARM heads | InternVL3.5-8B | 0.558 | 240 / 240 | 0.884 | 120 / 120 |
| **WebFARM heads** | **Qwen2.5-VL-7B** | **0.649** | **240 / 240** | **0.841** | **120 / 120** |

Same model, same cases, only training differs: the improvement comes from the
dataset and training, not from model size.

### 3.3 Confusion matrices of the selected assessor (same 240 / 120 cases)

| | Predicted failure | Predicted success |
|---|---|---|
| **Step: gold failure (180)** | 155 (86 %) | 25 (14 %) |
| **Step: gold success (60)** | 10 (17 %) | 50 (83 %) |
| **Recovery: gold failure (60)** | 59 (98 %) | 1 (2 %) |
| **Recovery: gold success (60)** | 9 (15 %) | 51 (85 %) |

Figure: `R02_task1_confusion_matrices`.

### 3.4 Context: published detectors (different data, background only)

Prompted GPT-4o reaches 83.8 % and a 76B open model 82.4 % on the failure
traces of arXiv:2509.14382; WebFARM's 7B 4-bit assessor reaches ≈ 83 % on
Web-Gold-40K validation. The sentence for the paper is "comparable magnitude
on different data", not "better than GPT-4o".

---

## 4. Task 2: WebFARM on two agents

### 4.1 Main result

**Table 12. Completion on the same 84 paired episodes (Wilson 95 % CI)**

| System | Completed | Rate | 95 % CI |
|---|---|---|---|
| SeeAct | 52 / 84 | 61.9 % | [51.2, 71.6] |
| **SeeAct + WebFARM** | **69 / 84** | **82.1 %** | [72.6, 88.9] |
| Browser Use | 61 / 84 | 72.6 % | [62.3, 81.0] |
| **Browser Use + WebFARM** | **72 / 84** | **85.7 %** | [76.7, 91.6] |

**Table 13. Paired comparisons (exact two-sided sign test)**

| Comparison | First better | Second better | Same | p |
|---|---|---|---|---|
| Browser Use + WebFARM vs Browser Use | 13 | 2 | 69 | 0.007 |
| SeeAct + WebFARM vs SeeAct | 21 | 4 | 59 | 0.0009 |
| Browser Use + WebFARM vs SeeAct + WebFARM | 11 | 8 | 65 | 0.65 |
| Browser Use vs SeeAct (both raw) | 18 | 9 | 57 | 0.12 |
| SeeAct + WebFARM vs Browser Use (raw) | 18 | 10 | 56 | 0.18 |

Figures: `R03_main_result_two_agents`, `R04_transfer_slope`.

### 4.2 Development vs held-out tasks

| Task set | n | SeeAct | SeeAct + WebFARM | helped / hurt | p | Browser Use | Browser Use + WebFARM | helped / hurt | p |
|---|---|---|---|---|---|---|---|---|---|
| Development | 40 | 60.0 | 77.5 | 9 / 2 | 0.065 | 70.0 | 87.5 | 8 / 1 | 0.039 |
| Held-out | 44 | 63.6 | 86.4 | 12 / 2 | 0.013 | 75.0 | 84.1 | 5 / 1 | 0.219 |
| All | 84 | 61.9 | 82.1 | 21 / 4 | 0.0009 | 72.6 | 85.7 | 13 / 2 | 0.007 |

The gap between the two raw agents is 10.7 points overall; with WebFARM it
shrinks to 3.6 points, and on held-out tasks SeeAct + WebFARM is ahead
(86.4 vs 84.1).

### 4.3 Rescue and harm

| Agent | Rescue rate (failures fixed) | Harm rate (successes lost) |
|---|---|---|
| Browser Use | 57 % (13 / 23) | 3 % (2 / 61) |
| SeeAct | 66 % (21 / 32) | 8 % (4 / 52) |

Figure: `R05_paired_outcomes`. For reference, the first recovery design
(web-v2, JSON advice with failure-type labels, Browser Use) had 0 % rescue and
22 % harm; the redesign (v3) removed almost all harm.

### 4.4 Cost

| Measure | SeeAct | SeeAct + WebFARM | Browser Use | Browser Use + WebFARM |
|---|---|---|---|---|
| Mean time per episode (s) | 182 | 123 | 220 | 175 |
| Model calls per step | 2 | 2 (+ assessment, options) | 1 | 1 (+ assessment, options) |
| Mean steps on pairs both variants completed | 3.31 (n = 48) | 3.40 (n = 48) | 3.08 (n = 59) | 3.32 (n = 59) |

Figures: `R06_cost`, `R07_step_budget` (both agents with WebFARM keep
completing tasks after step 5, where the raw agents flatten out in loops).

### 4.5 Per task

Per-task outcomes for all four systems: Handout Table 11 and figure
`A1_per_task_four_systems`. Consistent rescues (raw agent 0 / 2, with WebFARM
2 / 2): EPA climate change and cPanel pricing (Browser Use); GOV.UK state
pension age, IETF meetings, Nextcloud install, NOAA weather (SeeAct). Per
site: figure `A2_per_site_four_systems`.

---

## 5. How WebFARM works in practice

### 5.1 Recovery activity (84 episodes each)

| Measure | On Browser Use | On SeeAct |
|---|---|---|
| Episodes with at least one recovery | 31 | 56 |
| Recovery attempts | 55 | 97 |
| Uncertain alarms held back by the 0.9 gate | 22 | 31 |
| Option prompts / options taken | 86 / 69 | 155 / 143 |
| Options taken: goal link / search / Enter after search / scroll-top-then-search / go back | 26 / 19 / 19 / 4 / 0 | 58 / 42 / 36 / 5 / 2 |
| "0 = decide myself" | 17 | 12 |
| Repeat-guard blocks / overrides | 30 / – | 45 / 4 |
| Memory records written / shown | 34 / 10 | 81 / 15 |

The same three options carry the recoveries on both agents: open a
goal-matching link, search the site, press Enter after the search (figure
`R08_recovery_options_both_agents`). Funnel: `A3_recovery_funnel_both_agents`.

### 5.2 Detector behaviour live

| | Non-terminal steps | Judged successful (P ≤ 0.5) | Uncertain (0.5–0.9), held back | Confident failure (P ≥ 0.9) |
|---|---|---|---|---|
| Browser Use + WebFARM | 244 | 183 | 19 | 42 |
| SeeAct + WebFARM | 223 | 113 | 25 | 85 |

Figure: `R10_detector_scores_both_agents`. Failures are caught early: the
first confident failure comes at a median of step 2 (Browser Use) and step 1
(SeeAct); of the episodes where recovery opened, 26 / 31 and 45 / 56 were
completed, including 12 and 20 that the raw agent failed (figure
`R11_intervention_timeline`).

### 5.3 Why the raw agents fail, and what WebFARM changes

| | Browser Use alone | SeeAct alone |
|---|---|---|
| Failed episodes | 23 | 32 |
| Dominant action in failed episodes | scroll (67 %) | click (85 %; 88 % with select / hover) |
| Episodes with a loop of ≥ 5 identical actions | 16 (15 failed) | 19 (19 failed) |
| … with WebFARM | 7 (3 failed) | 1 (1 failed) |
| Other | invalid output in failed episodes 15 % | 56 executions rejected by the browser (hidden / stale elements); 0 unreadable answers, 0 token-limit hits |

Both agents fail for the same underlying reason, they do not notice that
their last action changed nothing: Browser Use scrolls past the content,
SeeAct clicks the same wrong element. A before/after detector sees both,
which is why one framework helps both (figure `R09_failure_modes`).

---

## 6. Case studies

Each case is a sequence of four separate step images (90 mm each) in
`docs/figures_v2/<case>/step1…step4.{pdf,svg,png}`, with a 2 × 2 preview.
Orange box and callout: element acted on; red tag: P1 failure; blue tag: step
driven by a WebFARM option.

**Browser Use + WebFARM, held-out task "Search Wikipedia for 'Marie Curie'
and open her article" (repeat 1)** (`R12_case_browser_use_wikipedia`):
1. Type "Marie Curie" into the search box.
2. Click a non-interactive element; nothing changes; P1: failure (P = 1.00).
3. WebFARM option "search the site": the query is typed again, the suggestion list opens.
4. WebFARM option "press Enter": the Marie Curie article opens.

Browser Use alone made the same useless click, then clicked the Wikipedia
logo and produced invalid output until it stopped.

**SeeAct + WebFARM, development task "Open the EPA page about climate change"
(repeat 1)** (`R13_case_seeact_epa`):
1. Type "climate change" into the search box.
2. Click the filled search box again; nothing changes; P1: failure (P = 0.99).
3. WebFARM options "search the site" and "press Enter" (steps 3–4).
4. WebFARM option: the result link "Climate Change"; www.epa.gov/climate-change reached at step 5.

SeeAct alone searched correctly but clicked the title row of the first result
(a `<dt>` element) 12 times instead of the link with the same words beside it.

---

## 7. Where WebFARM did not help

**Pairs WebFARM lost**

| Agent | Pair | What happened |
|---|---|---|
| Browser Use | Wikipedia black hole r1 | search query built from the goal ("box article black holes") led to a wrong page |
| Browser Use | jQuery API docs r2 | the actor left the correct path and then produced invalid output (no executed step to judge) |
| SeeAct | GitLab pricing r2 | the first click on a hidden menu item was rejected by the browser; WebFARM blocked it, the actor produced no valid element four times and SeeAct's own five-no-op rule ended the episode |
| SeeAct | Debian security r2 | reached `/security/index.en.html`, the same page as the target `/security/`; the exact-page scoring rule counts it as a different page (no other run ever saw this URL) |
| SeeAct | Wikipedia black hole r1 | same query problem as on Browser Use |
| SeeAct | Europa EU countries r2 | repeated menu clicks and a detour to another EU subdomain |

None of the losses is a wrong failure detection.

**Tasks every system fails in some repeat**: EPA recycle (repeat 1), Debian
news (repeat 2), Wikipedia black hole (repeat 2). On these pages the target
link shares no words with the goal, or the site search misleads, so the page
facts cannot point the agent to it.

---

## 8. Context and limitations

**Context (different benchmarks, relative gains only).** Published systems
that add checking or recovery to an agent report +4 to +8 points (Multimodal
Auto-Validation +5.0, BacktrackAgent +8.1, Devil's Advocate +3.7, OSCAR +6.3,
SkillWeaver +7.2), and WebCoach reports −1.7 to −4.0 points with a 7B actor.
WebFARM adds +13.1 and +20.2 points with a 7B actor and is shown on two
agents. Handout Table 4 has the full list with references.

**Limitations to state in the paper**
- Single seed for training; the locked test split of Web-Gold-40K is unused so far.
- 84 paired episodes per agent; held-out gain on Browser Use alone is not significant (6 discordant pairs).
- Runs were up to ten days apart (web-v3, v4, v5); the tasks every system fails were the same in each week, which suggests little site drift.
- Live websites change; results are tied to the run dates, and raw episodes are kept for audit.
- The agent adapters differ in three places (element identity, note position, option execution); everything else is identical.
- One scoring-rule edge case (Debian `index.en.html`) is reported, not re-scored.
- Memory is shown rarely (10 and 15 times) because each site is visited only twice; its contribution is not isolated.
- Error-analysis labelling (failure taxonomy, critical step, pipeline stage) is planned, not done.

---

## 9. Files and reproduction

| What | Where |
|---|---|
| Raw episodes (lab PC, not in git) | `.task2-assets/comparison/web-v3`, `web-v4-seeact`, `web-v5-seeact-ours` (plan.json with frozen source hashes and recorded amendments, episodes/, interrupted/, browser screenshots) |
| Four-system rows (in git) | `results/task2_web_comparison/four_system_rows.json` |
| Task 1 metrics (in git) | `results/table1_model_comparison/`, `results/task1_*` |
| Run docs | `TASK2_WEB_V3_RESULTS.md`, `TASK2_WEB_V4_SEEACT_RESULTS.md`, `TASK2_WEB_V5_SEEACT_WEBFARM_RESULTS.md` |
| Handout PDF | `Results_Comparison_Handout.pdf` (`.venv/bin/python scripts/make_results_handout.py`) |
| Figures | `docs/figures_v2/` (`.venv/bin/python scripts/make_results_figures_v2.py`, needs the raw episodes) |
| Agent code | `src/web_agent/eval/task2/` (`live.py` Browser Use; `seeact_live.py`, `seeact_framework.py`, `seeact_worker.py` SeeAct) |
| Run a comparison | `scripts/compare_agents.py freeze --out DIR --systems A C` (or `S`, `T`), then `run DIR`, `report DIR` |

## 10. Wording for the paper

"Attached to two different open-source web agents without any change to its
model or settings, WebFARM raised task completion on 42 real-website tasks
from 72.6 % to 85.7 % for Browser Use (p = 0.007) and from 61.9 % to 82.1 %
for SeeAct (p < 0.001), rescuing 57 % and 66 % of the agents' failures while
disturbing 3 % and 8 % of their successes, and shortening episodes by 20 % and
32 %. The gains hold on held-out tasks never inspected during design (+9.1
and +22.7 points), and with WebFARM the two agents are statistically
indistinguishable (p = 0.65)."

Avoid: "state of the art", "outperforms GPT-4o", numeric comparison with
other benchmarks' absolute success rates.
