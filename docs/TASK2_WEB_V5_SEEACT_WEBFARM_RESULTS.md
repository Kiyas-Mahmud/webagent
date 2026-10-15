# SeeAct with WebFARM attached (run `web-v5-seeact-ours`): the framework on a second agent

Final. Run 2026-10-14, 84 episodes (one infrastructure rerun: cPanel r0, the
browser page closed after navigation; rerun completed). Raw episodes:
`.task2-assets/comparison/web-v5-seeact-ours/` (plan.json, episodes/,
interrupted/, memory/). Design: [SEEACT_FRAMEWORK_INTEGRATION_PLAN.md](SEEACT_FRAMEWORK_INTEGRATION_PLAN.md).
Raw SeeAct: [TASK2_WEB_V4_SEEACT_RESULTS.md](TASK2_WEB_V4_SEEACT_RESULTS.md).
Browser Use with and without WebFARM: [TASK2_WEB_V3_RESULTS.md](TASK2_WEB_V3_RESULTS.md).
All four runs use the same plan: 42 tasks × 2 repeats, same goals, start pages,
success rules, 15-step and 2700 s budgets, same frozen Qwen2.5-VL-7B-Instruct
actor and the same WebFARM checkpoint, gate (0.9), note, options, guard and
memory settings. WebFARM = Failure-Aware Recovery and Memory (P1 detection,
adaptive recovery v3.3, P4 experience memory).

System ids: A = Browser Use, C = Browser Use + WebFARM, S = SeeAct, T = SeeAct + WebFARM.

## 1. Main result: WebFARM on two agents

**Table 16. Task completion of two open-source agents with and without WebFARM (84 paired episodes each; Wilson 95 % CI)**

| Agent | Without WebFARM | With WebFARM | Gain (pts) | Helped / hurt / same | p (sign test) |
|---|---|---|---|---|---|
| Browser Use | 72.6 % [62.3, 81.0] (61/84) | 85.7 % [76.7, 91.6] (72/84) | +13.1 | 13 / 2 / 69 | 0.007 |
| SeeAct | 61.9 % [51.2, 71.6] (52/84) | 82.1 % [72.6, 88.9] (69/84) | +20.2 | 21 / 4 / 59 | 0.0009 |

The same framework, with no change to its model, settings or prompts, raises
task completion of both agents: by 13 points on Browser Use and by 20 points on
SeeAct. On SeeAct it rescues 66 % of the raw agent's failures (21/32) and
disturbs 8 % of its successes (4/52); on Browser Use 57 % and 3 %.

**Table 17. By task set**

| Task set | n | SeeAct | SeeAct + WebFARM | Helped / hurt | p | Browser Use | Browser Use + WebFARM |
|---|---|---|---|---|---|---|---|
| Development | 40 | 60.0 % | 77.5 % | 9 / 2 | 0.065 | 70.0 % | 87.5 % |
| Held-out | 44 | 63.6 % | 86.4 % | 12 / 2 | 0.013 | 75.0 % | 84.1 % |

On SeeAct the gain is larger on the held-out tasks (+22.7) than on the
development tasks (+17.5): the recovery layer was designed while watching
Browser Use on the development tasks and never saw SeeAct, so this is a
genuine transfer result.

**Table 18. The two agents with WebFARM against each other**

| Comparison | First better | Second better | Same | p |
|---|---|---|---|---|
| Browser Use + WebFARM vs SeeAct + WebFARM | 11 | 8 | 65 | 0.65 |
| SeeAct + WebFARM vs raw Browser Use | 18 | 10 | 56 | 0.18 |

With WebFARM, the weaker agent (SeeAct) reaches the level of the stronger one
(85.7 % vs 82.1 %, not significantly different), and a SeeAct with WebFARM
completes more tasks than Browser Use without it.

**Table 19. Cost**

| Measure | SeeAct | SeeAct + WebFARM | Browser Use | Browser Use + WebFARM |
|---|---|---|---|---|
| Mean wall-clock per episode (s) | 182 | 123 | 220 | 175 |
| Mean steps on the 48 pairs both SeeAct variants completed | 3.31 | 3.40 | – | – |

WebFARM makes SeeAct 32 % faster on average because it ends the click loops
that otherwise run to the 15-step budget; on tasks both complete it adds 0.09
steps.

## 2. What WebFARM did on SeeAct (84 T episodes)

**Table 20. Recovery activity**

| Measure | SeeAct + WebFARM | Browser Use + WebFARM (web-v3) |
|---|---|---|
| Episodes with ≥ 1 recovery | 56 | 31 |
| Recovery attempts | 97 | 55 |
| Steps assessed by P1 | 365 | 319 |
| Confident failures (actions blocked) | 160 | – |
| Uncertain alarms held back by the 0.9 gate | 31 | 22 |
| Option prompts / options taken | 155 / 143 | 86 / 69 |
| Options taken: open goal link / search / Enter after search / scroll-top-then-search / go back | 58 / 42 / 36 / 5 / 2 | 26 / 19 / 19 / 4 / 0 |
| "0 = decide myself" | 12 | 17 |
| Repeat-guard blocks / overrides | 45 / 4 | 30 / – |
| Memory records written / shown | 81 / 15 | 34 / 10 |
| Stop reasons of the 15 failures | 8 SeeAct no-op limit, 7 step budget | – |

WebFARM fires far more often on SeeAct (56 of 84 episodes) than on Browser Use
(31) because SeeAct's characteristic failure, clicking the same wrong or
hidden menu element, is detected by P1 as a confident failure on the first
repetition. The same three options do the work on both agents: open a
goal-matching link, search the site, press Enter.

## 3. The four pairs WebFARM lost on SeeAct

| Pair | What happened | Class |
|---|---|---|
| GitLab pricing r1 | SeeAct's first click on the hidden 'Pricing' menu item timed out (Playwright). Raw SeeAct retried it three times and then found the visible link. With WebFARM the rejected click was blocked, the actor answered with no valid element four times, and SeeAct's own five-no-op rule ended the episode at step 5. | guard × agent no-op rule |
| Debian security r1 | WebFARM's link option took the agent to `/security/index.en.html`, the same page as the target `/security/`; the exact-page rule scores it as a different page. No other run ever observed this URL variant. | scoring-rule variant (candidate for a neutral re-score) |
| Wikipedia black hole r0 | The search query built from the goal was "box article black holes" ("search box" and "article" are goal words), leading to a wrong page. Same shared code and same loss on Browser Use + WebFARM r0. | query construction |
| EU countries r1 | repeated clicks on the same menu, a detour to a food.ec.europa.eu page, budget | agent loop beyond recovery |

Three pairs fail for all four systems: EPA recycle r0, Debian news r1, black
hole r1 (target link shares no words with the goal or the search misleads).

## 4. Caveats

- Runs were a week apart (web-v3 on 2026-10-04/06, web-v4 on 10-11/13,
  web-v5 on 10-14). The three tasks all four systems fail were the same in
  every week.
- One infrastructure rerun (cPanel r0: browser page closed after navigation;
  reran and completed in 3 steps); no plan amendments were needed in web-v5.
- The adapter differs from Browser Use's in three places (element identity,
  note position, option execution); the framework's model, gate, note text,
  option set, guard rule and memory are identical.

## 5. Wording for the paper

"Attached to two different open-source agents without any change to its
model or settings, WebFARM raised task completion from 72.6 % to 85.7 % on
Browser Use (p = 0.007) and from 61.9 % to 82.1 % on SeeAct (p < 0.001),
rescuing 57 % and 66 % of the agents' failures while disturbing 3 % and 8 %
of their successes. With WebFARM the two agents are statistically
indistinguishable (p = 0.65)."
