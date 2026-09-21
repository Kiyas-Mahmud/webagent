# Task 2 native-agent development — completed and audited

2026-09-16. This is development evidence, not the final evaluation.

The native Browser Use + InternVL integration completed all 12 frozen episodes.
Independent evidence audit: **PASS**. Four matched task-reset blocks; no overlap
exclusions, missing episodes or episode infrastructure failures. Eighteen raw
actor generations were independently reparsed; proposal-to-first-action identity,
scoring and total model-call accounting passed.

| System | Completed | Executed actions | Executed recovery steps | Memory queries | Memory generations | Model calls | Episode seconds, total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: native Browser Use + InternVL base | 4/4 | 6 | 0 | 0 | 0 | 6 | 294.05 |
| B: A + trained InternVL assessment/recovery | 4/4 | 6 | 1 | 0 | 0 | 12 | 318.22 |
| C: B + frozen memory | 4/4 | 6 | 1 | 1 | 0 | 13 | 309.80 |

Each system completed `click-button`, `enter-text`, `click-test` and
`click-button-sequence`. B−A and C−B completion differences are both zero.
This small development run does not demonstrate benefit or establish equivalence.
Timing is descriptive and excludes separately recorded model initialization.

## What the live evidence establishes

- The model chose TYPE with the exact requested value, observed the changed
  page, and then chose CLICK Submit. No task-specific action/value was inserted.
- On B/C text episodes the initial proposal contained TYPE and CLICK. Upstream
  Browser Use's `max_actions_per_step=1` selected TYPE; the next CLICK came from
  a new model request after observing the page. Full proposals, selected index
  and deferred counts are retained. Deferred actions were not auto-executed.
- On the sequence task, the trained interaction head labelled the first click
  FAILURE and proposed REPLAN. Those learned signals reached the next actor
  request. The next click executed and the trained recovery head assessed its
  transition as SUCCESS (logit `0.5513469576835632`, unchanged >0 threshold).
- C queried the original PC-01 encoder/store. All three candidates were excluded
  by `BACKTRACK_NO_EXECUTED_IN_EPISODE_NAVIGATION`. No example reached live
  generation. Empty admission retained the no-memory response.
- The admitted-memory generation branch was verified separately with an actual
  native completion-interface engineering fixture and stub responses. It was
  **not exercised by admitted examples in live development**.
- Terminal completion comes from MiniWoB raw reward exactly 1.0. Learned head
  labels and agent self-reports remain separate; some terminal interaction
  assessments disagreed with environment completion and were retained.

## Readiness decision

Ready for the frozen A/B/C package comparison: causal inputs, actual execution,
continuation, learned recovery assessment, live memory queries/abstention and
budget accounting are verified. There is no unresolved episode-level defect.
No positive performance or memory-admission requirement is used to select the
method. Final evaluation must report admissions, exposure and benefit separately.

The native coordinator remained idle after all results were saved and every
worker exited. It was closed after audit, without interrupting inference. Final
launch supervision records the same bounded post-run cleanup; it never changes
an episode or automatically retries one.

## Evidence and preserved attempts

Current archive: `.task2-assets/task2-development-v4/`:
`plan.json`, `audit.json`, `result_matrix.csv`, `final-manifest.json`,
`development-readiness.json`, `coordinator-cleanup.json`, producing sources,
images, native conversations, raw proposals, actions, assessments and retrieval.

Earlier attempts remain separate:

- v1: four completed episodes and one interrupted setup. The baseline text
  episode failed under the incorrectly stricter local proposal rejection.
- v2: frozen but never executed; regression fixture dependency correction.
- v3: stopped during loading, before any episode, after verifying upstream
  Browser Use's native action-list semantics.

Read [the exact diagnosis and corrections](TASK2_NATIVE_FEEDBACK_DIAGNOSIS.md).
These results must not be pooled with historical E0–E3/H0–H3 or Task 1 results.

The final 90-episode comparison is separately frozen at
`.task2-assets/task2-native-evaluation-v1/`; its results remain pending.
