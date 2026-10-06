# Task 2 results: Browser Use with and without the proposed framework (run `web-v3`)

Final. Run completed 2026-10-06. Plan and raw episodes:
`.task2-assets/comparison/web-v3/` (plan.json, report.json, episodes/, interrupted/).
PDF handout with Tables 1–11 and plain-English notes:
[Results_Comparison_Handout.pdf](Results_Comparison_Handout.pdf)
(rebuild with `.venv/bin/python scripts/make_results_handout.py`).
Method and settings: [TASK2_FAILURE_AWARE_AGENT_METHOD.md](TASK2_FAILURE_AWARE_AGENT_METHOD.md).
Literature context: [RESULTS_COMPARISON_WITH_LITERATURE.md](RESULTS_COMPARISON_WITH_LITERATURE.md).

## 1. Setup

| Item | Value |
|---|---|
| Base agent | Browser Use 0.13.10 driven by Qwen2.5-VL-7B-Instruct (frozen base, 4-bit, greedy, 512 new tokens) |
| Ours | the same Browser Use agent with the framework attached: P1 failure detection (trigger P(failure) ≥ 0.9), adaptive recovery v3.3 (plain-text note, page facts, repeat guard, numbered options), experience memory (P4) |
| Assessor checkpoint | Qwen2.5-VL-7B PC-02, epoch 0, seed 42, sha `71f867cc…` |
| Tasks | 42 real-website tasks, `configs/eval/task2/web_tasks_v2.json`: 20 development (inspected while designing recovery) + 22 held-out (written and live-verified 2026-10-04 before any run; never inspected) |
| Design | each task × 2 repeats × 2 systems = 168 episodes, 84 pairs; A and C run back-to-back, order alternates per repeat; C episodes of a run share one memory store |
| Budget | 15 steps per episode; identical prompts, viewport 1280×720, start pages, allowed domains |
| Completion | environment rule: current page equals the target page (host + path; `page_prefix` for list pages; `site` for the jQuery API site). Never shown to the agent |
| Statistics | exact two-sided sign test on discordant pairs; Wilson 95 % CIs |
| Infrastructure handling | reset failures (DNS outage on 2026-10-04, 13:44–16:44, 32 cases; one Coursera 60 s load timeout) and one C episode whose observations were `chrome-error://` pages were set aside and rerun; rule applied to both systems before looking at outcomes; see `interrupted.jsonl` |

## 2. Main result

**Table 6. Task completion**

| Task set | Pairs | Browser Use | Browser Use + Ours | Δ (pts) | p |
|---|---|---|---|---|---|
| Development | 40 | 70.0 [54.6, 81.9] | 87.5 [73.9, 94.5] | +17.5 | 0.039 |
| Held-out | 44 | 75.0 [60.6, 85.4] | 84.1 [70.6, 92.1] | +9.1 | 0.219 |
| All | 84 | 72.6 [62.3, 81.0] | 85.7 [76.7, 91.6] | +13.1 | 0.007 |

Completion rate in % with Wilson 95 % CI.

Over all 84 paired episodes, Browser Use alone completed 72.6 % of tasks; with
the framework 85.7 %, a gain of 13.1 points (p = 0.007). About half of the base
agent's failures are removed. The gain is +17.5 on development tasks and +9.1 on
held-out tasks; the held-out gain is not significant on its own (6 discordant
pairs) but has the same direction, so the improvement is not an artefact of
tuning to known tasks.

**Table 7. Paired outcomes**

| Task set | Helped | Hurt | Same |
|---|---|---|---|
| Development | 8 | 1 | 31 |
| Held-out | 5 | 1 | 38 |
| All | 13 | 2 | 69 |

In 69 of 84 pairs both systems gave the same result (the framework is silent
unless the detector fires). Of the 15 discordant pairs, ours won 13 and lost 2:
when the framework changes the outcome, it improves it in 87 % of cases.

**Table 8. Rescue and harm rates**

| Task set | Rescue rate | Harm rate |
|---|---|---|
| Development | 67 % (8/12) | 4 % (1/28) |
| Held-out | 45 % (5/11) | 3 % (1/33) |
| All | 57 % (13/23) | 3 % (2/61) |

Rescue = baseline failures completed by ours; harm = baseline successes failed by
ours. The first design (web-v2, synchronous JSON advice with diagnosis labels)
had rescue 0 % and harm 22 %; the redesign (v3) removed the harm and added the
rescues.

**Table 9. Cost per episode**

| Measure | Browser Use | Browser Use + Ours |
|---|---|---|
| Mean wall-clock (s) | 220 | 175 |
| Mean steps on the 59 pairs both completed | 3.08 | 3.32 |

Ours is 20 % faster on average because failed baseline episodes exhaust the
15-step budget in loops; on tasks both complete it adds 0.24 steps.

**Table 10. Recovery activity (84 C episodes)**

| Measure | Count |
|---|---|
| Episodes with ≥ 1 recovery | 31 |
| Recovery attempts | 55 |
| Option selections by the actor | 69 |
| Repeat-guard blocks | 30 |
| Uncertain alarms suppressed (P < 0.9) | 22 |
| Memory records written | 34 |
| Memory records shown | 10 |

In 53 of 84 episodes the detector never fired and the actor's input equalled the
baseline's. Memory reuse is small here (10 exposures) because each site is
visited twice; its storage decisions ran in every incident.

**Table 11. Per-task outcomes (two repeats; Y completed, N failed)**

| Task | Split | Browser Use | Browser Use + Ours | Pair result |
|---|---|---|---|---|
| wikipedia-wiki-alan-turing | dev | N Y | Y Y | helped ×1 |
| wikipedia-wiki-eiffel-tower | dev | N Y | Y Y | helped ×1 |
| gov-renew-adult-passport | dev | N Y | Y Y | helped ×1 |
| gov-bank-holidays | dev | N Y | Y Y | helped ×1 |
| epa-climate-change | dev | N N | Y Y | helped ×2 |
| ietf-process-rfcs | dev | N N | Y N | helped ×1 |
| ted-talks | dev | N N | N Y | helped ×1 |
| jquery-api-jquery-com | dev | Y Y | Y N | hurt ×1 |
| debian-news | dev | N N | N N | both fail |
| 11 other development tasks | dev | Y Y | Y Y | same |
| cpanel-pricing | held-out | N N | Y Y | helped ×2 |
| gov-uk-apply-driving-licence | held-out | N N | N Y | helped ×1 |
| wikipedia-random (community portal) | held-out | N Y | Y Y | helped ×1 |
| wikipedia-wiki-marie-curie | held-out | N Y | Y Y | helped ×1 |
| wikipedia-wiki-black-hole | held-out | Y N | N N | hurt ×1 |
| epa-recycle | held-out | N N | N N | both fail |
| noaa-weather | held-out | N N | N N | both fail |
| 15 other held-out tasks | held-out | Y Y | Y Y | same |

Per repeat: r0 A 28 / C 36; r1 A 33 / C 36 (42 pairs each).

Three patterns: consistent rescues (EPA climate change, cPanel pricing: baseline
0/2, ours 2/2); variance rescues (six tasks where the baseline failed once and
ours never did); tasks beyond the framework (Debian news, EPA recycle, NOAA
weather: 0/4, the target link shares no words with the goal). The two losses
(black hole r0, jQuery API r1) ended with the actor off the correct path and
emitting invalid output, leaving no executed step for the detector to judge;
they are examined in the error analysis.

## 3. How to use these in the thesis
- Results order: Table 1 (backbones) → Table 2 (trained vs prompted) →
  **Table 6** → 7 → 8 → 9 → 10 → error analysis → Table 4 (literature context)
  → Table 11 in the appendix.
- Figure for Table 6: paired bars per split (baseline grey, ours colour) with CI
  whiskers and p above.
- Safe wording: "improved completion by 13 points (p = 0.007)", "recovered 57 %
  of baseline failures while disturbing 3 % of successes", "same direction on
  held-out tasks, not individually significant at n = 44".
- Avoid: "outperforms state of the art"; any direct numeric comparison with
  other benchmarks.

## 4. Next: error analysis (planned protocol)
Taxonomy F1 no-progress loop, F2 wrong target, F3 premature "done", F4 invalid
output, F5 grounding error, F6 environment; critical-error step per failed
episode; pipeline stage for C failures (detection → options → choice →
execution → recovery verdict); Tables E1 (class distribution A vs C, dev vs
held-out), E2 (rescue rate per baseline failure class), E3 (where help stopped);
critical-step histogram; 3–4 screenshot case studies; 20-episode manual
spot-check with Cohen's κ.
