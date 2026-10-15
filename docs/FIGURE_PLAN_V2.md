# Figure plan v2 (2026-10-14): results section with two agents

Supersedes the figure list in [FIGURE_PLAN.md](FIGURE_PLAN.md) (2026-10-06,
one agent). The 15 figures already generated
([figures/README.md](figures/README.md)) were made from web-v3 only; with the
SeeAct runs (web-v4 raw, web-v5 with WebFARM) the headline changes from "the
framework helps Browser Use" to "the same framework helps two different
agents". The figure set must change accordingly.

## 1. What comparable papers show (checked again)

| Paper / venue | Figures they rely on | Take for us |
|---|---|---|
| WebVoyager (ACL 2024), Agent-E (2024) | per-website success bars / tables; annotated trajectory screenshots; error-type examples | per-site figure for four systems; screenshot case study |
| Agent Workflow Memory (ICML 2025) | success vs. amount of experience (learning curve); steps reduced | step-budget curve; memory effect |
| BacktrackAgent, Multimodal Auto-Validation, WebCoach | before/after bars per configuration; ablation bars; per-site bars (text vs. vision) | "without / with" bars per agent |
| AgentDebug / AgentErrorTaxonomy, MAST (2025) | taxonomy tree; failure-mode distribution per system; critical-step histogram; case table | failure-mode contrast between the two agents; critical step |
| WebSuite, AutoWebGLM | nested success tables per action type; error proportions | action-mix of failures |
| FPC-VLA and other ESWA applied-AI papers | pipeline figure; qualitative correction frames; ablation; sometimes radar / confusion matrix for multi-metric models | method figure; one correction case per agent; confusion matrix for the assessor (Task 1) |

Common to all: one method figure, one main-result figure with uncertainty,
one per-site or per-task breakdown, one cost figure, one error analysis, one
qualitative example. Rare and therefore distinctive: paired-test outcomes,
held-out split, the same add-on shown on two agents, detector-score
distributions, option-level analysis of what the recovery actually did.

## 2. Figure list

Status: **keep** = existing figure unchanged; **regen-4** = regenerate with
the four systems (A, C, S, T); **new** = not yet made; **pending** = needs the
error-analysis labelling script. ★ = distinctive, worth the page budget.

### A. Method (Section 3)

| # | Figure | Status | Notes |
|---|---|---|---|
| M1 | WebFARM architecture: agent ↔ website; assessor; recovery (facts, note, guard, options); memory; the thin agent adapter with its six hooks, drawn once with "Browser Use / SeeAct" as the pluggable box | new (drawing) | the two-agent story needs the adapter box visible |
| M2 | One agent step as a sequence: capture → actor → execute → score → assess (async) → gate → note / options → next step | new (drawing) | same figure serves both agents; mark the three adapter points |
| M3 | Assessor internals: before/after screenshots + text → backbone → adapter → heads | new (drawing) | unchanged from plan v1 |

### B. Task 1: assessor (Section 5.1)

| # | Figure | Status | Notes |
|---|---|---|---|
| B1 | Trained vs prompted (fig6) | keep | |
| B2 | Backbones (fig7) | keep, optional | table may suffice |
| B3 | Confusion matrices of the outcome and recovery heads | pending (export validation predictions) | standard in ESWA; cheap once predictions are exported |
| B4 | Reliability curve of P(failure) with the 0.9 gate marked | pending (same export) | explains why the gate is 0.9; distinctive |

### C. Task 2: agent-level results (Section 5.2)

| # | Figure | Status | Notes |
|---|---|---|---|
| C1 ★ | **Main result, two agents**: for Browser Use and SeeAct, "without / with WebFARM" bars with Wilson CIs and p, three panels (development, held-out, all) or one panel with the two agents side by side | regen-4 (replaces fig1) | the headline figure of the paper |
| C2 ★ | **Transfer slope chart**: raw → +WebFARM for each agent and each split as connected points; the two "with" points converge (85.7 / 82.1) | new | shows "weak agent + WebFARM > strong agent alone" in one glance |
| C3 | Paired outcomes per agent: helped / hurt / same, with rescue and harm rates | regen-4 (fig2 or fig8 Sankey ×2) | choose one of bar or Sankey, not both |
| C4 | Per-task matrix: 42 tasks × 4 systems × 2 repeats | regen-4 (fig3) | appendix |
| C5 | Per-site completion, four systems | regen-4 (fig15 as a 4-point dot plot or two dumbbells) | |
| C6 | Cost: time box plots for four systems; steps on both-completed pairs | regen-4 (fig4) | |
| C7 | Step-budget curves, four systems | regen-4 (fig9) | shows where each agent overtakes |
| C8 ★ | **Rescue vs harm** scatter: one point per (agent, split), x = rescue rate, y = harm rate, with the web-v2 first design as a reference point (0 %, 22 %) | new | distinctive; summarises the safety story |

### D. Mechanism (Section 5.2, continued)

| # | Figure | Status | Notes |
|---|---|---|---|
| D1 | Recovery funnel, both agents side by side (steps assessed → confident failures → options → taken → verified → completed) | regen-4 (fig5) | |
| D2 ★ | **Which options worked, on both agents** (fig12 extended): per option kind, offered / taken / verified success, Browser Use vs SeeAct | regen-4 | "the same three options do the work on both agents" |
| D3 ★ | **Failure-mode contrast**: action mix of failed episodes, Browser Use (scroll loops) vs SeeAct (click loops), and loop-length histograms | regen-4 (merge fig10 + fig13) | the "different agents, same underlying failure" argument |
| D4 | Detector score distribution with the gate, both agents (fig14 ×2) | regen-4 | |
| D5 ★ | **Intervention timeline**: for every episode where WebFARM acted, the step of the first confident failure and the step of completion or stop, as a strip plot per agent | new | unique; shows how early failures are caught and how long recovery takes |
| D6 | Memory effect: repeat 2 vs repeat 1 completion where memory was shown (small n; both agents) | new, optional | report honestly |

### E. Error analysis and case studies (Section 5.3)

| # | Figure | Status | Notes |
|---|---|---|---|
| E1 | Case study, Browser Use: cPanel (fig11) | keep | |
| E2 ★ | Case study, SeeAct: EPA climate change r0, raw SeeAct failing vs SeeAct + WebFARM (search → Enter → "Climate Change" link, 9 steps) as a screenshot strip with P1 verdicts | new | mirrors E1 on the second agent |
| E3 | Harm anatomy: the 6 hurt pairs (2 on Browser Use, 4 on SeeAct) with class labels | new (table-figure) | small, honest |
| E4 | Failure taxonomy tree (F1–F6) | pending | |
| E5 | Failure-class distribution per system (4 systems) and rescue by class | pending (labelling script) | |
| E6 | Critical-error-step histogram per system | pending (labelling script) | |
| E7 | Where help stopped (pipeline stage of each T and C failure) | pending (labelling script) | |

## 3. Recommended set for the paper (12 figures)

M1, M2 · B1, B3 · C1, C2, C3, C6 · D2, D3, D5 · E2 — plus C4 and the
pending E5–E7 in the appendix. Drop from the paper: fig8 Sankey if C3 is a
bar chart; fig10 and fig13 as separate figures (merged into D3); fig7 as a
figure (table).

## 4. Style (unchanged)

Elsevier widths 90 / 140 / 190 mm; Liberation Sans 7–9 pt; near-black text
only; vector PDF. Colours: grey = raw agent, blue = with WebFARM, red = hurt;
SeeAct panels use the same two colours with a hatch or a lighter tint for the
second agent so "with / without" stays readable; four-system plots use grey,
blue, light grey, light blue in that fixed order (validate with the palette
script). Captions name the run ids (web-v3 / v4 / v5) and n.

## 5. Production order

1. Regenerate C1, C3, C4, C5, C6, C7, D1, D2, D3, D4 from the three runs
   (extend `scripts/make_results_figures.py` with a four-system loader that
   asserts the Table 12–20 numbers).
2. New C2, C8, D5, E2, E3.
3. Method drawings M1–M3.
4. Export Task 1 validation predictions → B3, B4.
5. Labelling script → E4–E7.
