# Figure plan for the results section

2026-10-06. What Q1 agent papers show, and the figures we can produce from the
data we already have. "Must" = expected by reviewers of applied-AI journals
(ESWA style: method figure, main-result figure with uncertainty, ablation,
error/case analysis, limitations). "Optional" = strengthens the paper if space
allows. Every figure lists its data source; all sources exist on the lab PC
unless marked *new run*.

## What comparable Q1 papers show

| Paper | Figures they use | Lesson for us |
|---|---|---|
| AgentDebug / AgentErrorTaxonomy (arXiv:2509.25370) | taxonomy tree; error-type frequency bars per benchmark; histogram of the critical-error step; one annotated trajectory | show *where in the episode* failures happen, not only how many |
| MAST (arXiv:2503.13657) | taxonomy diagram; stacked bars of failure modes per framework; before/after case-study table | one picture of the taxonomy + one distribution chart |
| WebSuite (arXiv:2406.01623) | nested success tables per action / UI element | per-task or per-site breakdown as a heatmap |
| BacktrackAgent (arXiv:2505.20660) | architecture with verifier/judger/reflector; ablation bars; step-level example | ablation figure of components |
| Multimodal Auto Validation (arXiv:2410.00689) | per-website bar chart (text vs vision validator) | per-site bars with and without the framework |
| WebCoach (arXiv:2511.12997) | success vs actor size; time/steps per configuration | cost figure (time, steps) next to accuracy |
| FPC-VLA, ESWA (S095741742600655X) | pipeline figure; qualitative failure-correction frames; ablation table | screenshot sequence of a correction |
| AutoWebGLM (arXiv:2404.03648) | error-type proportion table only | the minimum; we should exceed it |

## Figure list

### A. Method (Section 3)

| # | Figure | Must / optional | Data | Status |
|---|---|---|---|---|
| F1 | **System architecture**: Browser Use actor ↔ real website; P1 assessor (Qwen2.5-VL-7B + QLoRA heads); recovery layer (facts, note, guard, options); experience memory; what flows where | Must | method doc §1.1 (mermaid draft exists) | draw cleanly (draw.io / TikZ) |
| F2 | **One agent step as a sequence/flow**: capture → actor → execute → score → assess (async) → gate ≥ 0.9 → note/options → next step | Must | method doc §3 | draw |
| F3 | **Trained assessor internals**: before/after screenshots + text → backbone (LoRA) → 768-d adapter → task adapters → heads (outcome, failure type, recovery outcome, memory) | Must | method doc §2.1 | draw |
| F4 | **Recovery note and option prompt** as a boxed example (real text from a run) | Optional | method doc §4.5 | typeset |
| F5 | Dataset overview: domains (305), split sizes, label distribution | Optional (dataset chapter) | Web-Gold-40K metadata | plot |

### B. Task 1: assessor quality (Section 5.1)

| # | Figure | Must / optional | Data | Status |
|---|---|---|---|---|
| F6 | **Grouped bars: trained vs prompted assessment**, MCC per method (Table 2); prompted rows at 0, trained at 0.65 / 0.84 | Must | results/task1_* CSVs | plot |
| F7 | Backbone comparison bars (Table 1 metrics per backbone) | Optional (table may suffice) | Table 1 CSV | plot |
| F8 | **Confusion matrix** of the outcome head (success/failure) and of the recovery-outcome head on validation | Must | validation predictions of the selected checkpoint | plot (predictions must be exported from the Task 1 run) |
| F9 | **Reliability / calibration curve** of P(failure) and the ROC with the 0.9 gate marked | Optional, strong | same predictions | plot |
| F10 | Valid-output coverage bars (prompted methods produce no parsable verdict) | Optional | Task 1 pilots | plot |

### C. Task 2: agent-level results (Section 5.2)

| # | Figure | Must / optional | Data | Status |
|---|---|---|---|---|
| F11 | **Main result: paired bars** completion % for Browser Use vs Browser Use + Ours, three groups (development, held-out, all), Wilson CI whiskers, p above | Must | web-v3 report.json | plot |
| F12 | **Paired-outcome chart**: helped / hurt / same counts per split (stacked or dumbbell) | Must | report.json | plot |
| F13 | **Per-task heatmap**: 42 tasks × 4 columns (A r0, C r0, A r1, C r1), green/red, grouped by split | Must (or appendix) | episodes/*/result.json | plot |
| F14 | **Per-site bars**: completion per website domain, both systems (Auto-Validation style) | Optional | result.json + task file | plot |
| F15 | **Cost**: mean time and mean steps per episode, both systems; steps distribution on both-completed pairs | Must | result.json | plot |
| F16 | **Recovery funnel** for the 84 C episodes: steps assessed → confident failures → options offered → option taken → recovery judged success → episode completed | Must | assessment-*.json, note-*.json, actor-*-choice.json, guard files | plot (needs the labelling script) |
| F17 | Ablation bars: baseline → +detection only → +options → +guard → full (if ablation runs are done) | Optional, *new runs* (~1 day each) | — | only if time |
| F18 | Memory effect: repeat 1 vs repeat 0 completion for tasks where memory was shown | Optional | result.json + memory-*.json | plot (small n; report honestly) |

### D. Error analysis (Section 5.3)

| # | Figure | Must / optional | Data | Status |
|---|---|---|---|---|
| F19 | **Failure taxonomy** as a small tree/table figure (F1 loop, F2 wrong target, F3 premature done, F4 invalid output, F5 grounding, F6 environment) | Must | definitions | typeset |
| F20 | **Failure-class distribution**: grouped bars, baseline vs ours, development vs held-out | Must | labelling script output | plot |
| F21 | **Rescue by failure class**: for each baseline failure class, bars of rescued vs not rescued | Must | labelling script | plot |
| F22 | **Critical-error step histogram**, baseline vs ours (AgentDebug style) | Must | labelling script | plot |
| F23 | **Where help stopped** (pipeline stage of each C failure): detection miss / no option / actor ignored / execution / wrong recovery | Must | labelling script | plot |
| F24 | **Case-study screenshot strips** (3–4 panels each): one rescue (TED or EPA), one hurt (black hole), one both-fail (Debian news), one held-out rescue (cPanel) — with step captions and P1 verdicts | Must | observation-*.png in episode folders | compose |
| F25 | Episode timeline plot for one pair: step vs P(failure), with recovery markers, A and C side by side | Optional, very readable | assessment-*.json | plot |

### E. Discussion / limitations

| # | Figure | Must / optional | Data |
|---|---|---|---|
| F26 | Literature context: our gain vs published gains from adding recovery (Table 4 as a dot plot, actor size on x) | Optional | Table 4 |
| F27 | Boxed examples of what the framework cannot fix (Debian "Other downloads", false "done") | Optional | traces |

## Minimum set for submission
F1, F2, F3 (method) · F6, F8 (assessor) · F11, F12, F13, F15, F16 (agent) ·
F19, F20, F21, F22, F23, F24 (error analysis). That is 16 figures; several can
be merged into multi-panel figures (e.g. F11+F12, F20+F21, F15 as two panels)
to fit a journal page budget of about 10–12.

## Style rules (apply to every plot)
- Two colours only: baseline grey (#8A8A8A), ours blue (#2F4F6F); held-out
  hatched or lighter.
- Always show n and CI or exact counts; p-values on the figure for paired tests.
- Same task order in every per-task figure (as in `web_tasks_v2.json`).
- Vector output (PDF/SVG), font ≥ 8 pt at final size; captions state the data
  source (run id `web-v3`, 84 pairs).

## Production order (lab PC)
1. Labelling script (error taxonomy + pipeline stages) → CSVs for F16, F20–F23.
2. `scripts/make_results_figures.py`: F11, F12, F13, F15, F16, F20–F23, F25 from
   the CSVs and report.json (matplotlib, vector).
3. Export Task 1 validation predictions → F6, F8, F9.
4. Case-study strips F24 from saved screenshots.
5. Method drawings F1–F3 by hand.
