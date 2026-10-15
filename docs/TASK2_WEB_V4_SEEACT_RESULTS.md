# SeeAct as a second raw agent on the same tasks (run `web-v4-seeact`)

Final. Run 2026-10-11 (69 episodes) and 2026-10-13 (15 episodes, after the
16:45 stop). Raw episodes: `.task2-assets/comparison/web-v4-seeact/` (plan.json
with three recorded amendments, episodes/, interrupted/). Method and
integration notes: [SEEACT_BASELINE_PLAN.md](SEEACT_BASELINE_PLAN.md).
Browser Use numbers below are from run `web-v3`
([TASK2_WEB_V3_RESULTS.md](TASK2_WEB_V3_RESULTS.md)); both runs use the same
plan: 42 tasks × 2 repeats, same goals, start pages, success rules, 15-step and
2700 s budgets, same frozen Qwen2.5-VL-7B-Instruct actor (4-bit, greedy, 512
new tokens).

System names: A = Browser Use, C = Browser Use + WebFARM, S = SeeAct (raw).

## 1. What SeeAct is here

SeeAct (Zheng et al., ICML 2024), package 0.2.9.0, unmodified decision code: per
step it lists the interactive elements inside the viewport, draws set-of-mark
boxes, asks the model to reason (turn 0) and then to pick an element letter
plus ACTION / VALUE (turn 1), and executes with Playwright. Two model calls per
step. It runs attached over CDP to the same headless Chromium that the Browser
Use runs used; completion is scored by the same environment rule after every
step. No assessment, recovery or memory.

## 2. Task completion on the same 84 paired episodes

**Table 12. Completion by agent (84 task-repeats; Wilson 95 % CI)**

| System | Completed | Rate | 95 % CI |
|---|---|---|---|
| SeeAct | 52 / 84 | 61.9 % | [51.2, 71.6] |
| Browser Use | 61 / 84 | 72.6 % | [62.3, 81.0] |
| **Browser Use + WebFARM** | **72 / 84** | **85.7 %** | [76.7, 91.6] |

With the same 7B model and tasks, raw SeeAct completes fewer tasks than raw
Browser Use (61.9 % vs 72.6 %) and far fewer than Browser Use + WebFARM
(85.7 %). The framework's gain over SeeAct is 23.8 points.

**Table 13. Paired comparison (exact two-sided sign test on discordant pairs)**

| Comparison | Pairs | First better | Second better | Same | p |
|---|---|---|---|---|---|
| Browser Use + WebFARM vs SeeAct | 84 | 23 | 3 | 58 | < 0.001 |
| Browser Use vs SeeAct | 84 | 18 | 9 | 57 | 0.122 |

Browser Use + WebFARM beats SeeAct on 23 task-repeats and loses on 3 (TED r0,
black hole r0, jQuery API r1: the three C losses already known from web-v3).
Raw Browser Use vs raw SeeAct is not significant: the two agents fail on
different tasks (SeeAct is better on search-driven Wikipedia tasks, TED and
cPanel; Browser Use on arXiv, GOV.UK, IETF, EPA list pages).

**Table 14. By task set**

| Task set | n | SeeAct | Browser Use | Browser Use + WebFARM |
|---|---|---|---|---|
| Development | 40 | 60.0 % | 70.0 % | 87.5 % |
| Held-out | 44 | 63.6 % | 75.0 % | 84.1 % |

**Table 15. Cost**

| Measure | SeeAct | Browser Use | Browser Use + WebFARM |
|---|---|---|---|
| Mean wall-clock per episode (s) | 182 | 220 | 175 |
| Model calls per step | 2 | 1 | 1 (+ assessment, options) |
| Mean steps on the 49 pairs both SeeAct and WebFARM completed | 3.35 | – | 3.65 |

## 3. How SeeAct fails (32 failed episodes)

| Observation | Count |
|---|---|
| Stop reason: 15-step budget exhausted | 28 of 32 |
| Stop reason: five consecutive no-ops / self-declared done | 3 / 1 |
| Episodes with a loop of ≥ 5 identical actions | 19 of 32 |
| Action mix in failed episodes | CLICK 85 %, TYPE 7 %, SELECT 3 % |
| Executions rejected by Playwright (all 84 episodes) | 56 (click timeout 30, fill on a non-input 17, select on a non-select 8) |
| Navigations outside the task's domains, undone | 11 |
| Model outputs without a parsable action / token-limit hits | 0 / 0 |

SeeAct's dominant failure is **repeated clicking of the same wrong element**:
unlike Browser Use, whose failures were scroll loops (67 %), SeeAct loops on
clicks (85 % of failed-episode actions), typically a navigation-menu entry that
does not lead to the target page, or an element that is listed but not
clickable (its selector resolves to a hidden menu item, so Playwright times out
and SeeAct records "Failed to perform" and tries again). It never produced an
unparsable answer and never hit the token limit, so the 7B model follows
SeeAct's two-turn format well; the failures are decision and grounding
failures, exactly the class WebFARM targets.

Tasks SeeAct fails in both repeats (NN) include arXiv lists, Creative Commons
licenses, GOV.UK bank holidays / passport / driving licence / pension age, IETF
about / RFCs, Nextcloud install, Debian news, EPA recycle, NOAA weather. Browser
Use + WebFARM completes 2/2 on nine of these.

## 4. Caveats

- SeeAct ran 5–7 days after web-v3, so site drift is possible; the four tasks
  that both raw agents fail in every repeat (Debian news, EPA recycle, NOAA
  weather, driving licence r0) were the same in both weeks, which suggests the
  drift is small.
- One infrastructure fix was applied during the run and recorded in
  `plan.json` (TED font-wait screenshot fallback, from the pilot); one
  episode (Debian security r0) exhausted the 2700 s wall clock because SeeAct's
  element listing takes ~5 min per step on planet.debian.org and was rerun under
  the amended "wall clock = failed result" rule; it then completed in 3 steps on
  a different path. Coursera r0 needed four reset attempts (page `load` event
  intermittently never fires on a fresh browser; same in web-v3).
- SeeAct + WebFARM (system T) is implemented and unit-tested but not yet run;
  its result decides whether the framework helps a second agent.

## 5. Wording for the paper

"With the same frozen 7B actor and tasks, raw SeeAct completed 61.9 % and raw
Browser Use 72.6 % of the 84 paired episodes (not significantly different,
p = 0.12). Browser Use + WebFARM completed 85.7 %, more than either raw agent
(vs SeeAct: 23 better, 3 worse, p < 0.001)."
