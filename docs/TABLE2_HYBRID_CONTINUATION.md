# Hybrid Task 3: bounded recovery and continuation

2026-09-12. **Task 3 complete as an engineering milestone.** Next: Task 4,
H3 memory integration and isolation. Full H-runner integration and live
development remain pending; final evaluation is not ready.

## Change

[EpisodeRunner](../src/web_agent/runtime/episode.py) now accepts
`hybrid_continuation=False`. Existing callers retain their previous behavior.
The hybrid runner must explicitly enable and bind this setting in Task 5.
The checkpoint, generator, prompts, recovery controller, strategy validation,
assessment heads and memory implementation were unchanged in this task.

With the setting enabled:

| Condition | Behavior |
|---|---|
| Executed recovery receives a negative learned assessment | Retry within the same incident if allowance remains |
| Incident reaches two attempts | Record `exhausted_unresolved`; allow normal acting within remaining budgets |
| Episode reaches four recovery attempts | Skip further recovery planning/retrieval; allow normal acting within remaining budgets |
| Learned assessment predicts resolution/progress | Return to normal acting; progress alone does not clear spent incident attempts |
| Learned assessment predicts resolution | Release that failure binding, preserving the episode-wide spent count |
| ABORT, terminal/truncation stop, loop, timeout or total budget limit | Stop; no continuation override |

An already-predicted ABORT still stops after recovery allowance is exhausted,
without charging an extra recovery attempt. Recovery actions now also enter the
existing loop guard in hybrid mode. The opaque browser stop takes precedence
over a simultaneous loop signal.

Repeated unresolved requests retain their incident. The binding uses action,
issued parameters, box, current URL and observed stable control identity, excluding
observation-bound control IDs. Each new triggering transition gets a logged
action/observation binding. An observation refresh alone cannot create retry
credit. Different requests may create new incidents but cannot exceed four total
recovery attempts. Reused incidents retain their original RNG incident index.

This identity rule is conservative: repeating the same request after other page
changes can retain the old unresolved incident until learned resolution. It
does not infer success from page changes or replenish retries.

Execution, learned assessment and continuation reasons remain separate records.
Observable text/selection/checkbox effects are still logged, but do not override
the hybrid retry table. Plain clicks receive no invented progress credit.

## Evidence

**128 focused tests passed:** 20 new controller cases and 108 compatibility checks
covering existing episode/recovery paths, interruptions, causal memory transitions,
the hybrid adapter, target resolution and model-call accounting.

The complete-controller scripted fixtures verified:

- Text-entry and click-only sequences reaching a terminal state after two
  negative recovery assessments and subsequent normal acting.
- Exactly two attempts for the same unresolved failure despite refreshed
  observations; normal acting stopping at 30 executor requests.
- Different failed requests remaining capped at four episode recovery attempts.
- Rejected plans/browser actions remaining unresolved; no fabricated assessment.
- Learned progress preserving spent attempts; learned resolution preserving
  the global allowance; episode initialization resetting episode-owned state.
- ABORT before and after exhaustion, loop/terminal precedence, timeout and the
  102-call limit.
- Default-off behavior still ending at the original recovery-budget stop.

These are **scripted engineering fixtures, with zero actual model inference,
zero memory writes and zero live evaluation episodes**. They use the existing
E2 fixture switch contract to exercise the runner feature; they are neither
new E-profile experiments nor H-profile performance results. Task 5 will register
the actual H identities and runner wiring.

Evidence directory:
`/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task3-engineering/`

- `controller-tests-r3.txt`, `controller-fixtures-r3/`: final 20-case result and
  recorded controller events, actions, assessments, budgets and continuations.
- `compatibility-tests-final.txt`: final 108 compatibility checks.
- `task3.patch`, `episode-before.py`, `sources/`: exact change and producing sources.
- `baseline.json`, `preservation.json`, `completion.json`, `manifest.json`:
  preservation and evidence bindings.

Earlier fixture logs are preserved. One reset test initially tried to reuse an
executor that episode completion had closed; the corrected fixture supplies a
fresh executor while reusing the runner/controller to test their state reset.

Task 4 remains memory wiring. Task 5 must connect the normal-action context API,
register H systems, bind this flag and freeze the complete runner. Task 6 remains
the one matched 24-episode live development check. Engineering success here does
not establish model-chosen multi-step behavior or a completion improvement.
