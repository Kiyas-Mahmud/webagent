# Task 2 native structured-output feedback correction

2026-09-16. Engineering defect found during development; no final evaluation.

The first native development attempt is preserved at
`.task2-assets/task2-development-v1/`, including its producing-source snapshot.
Four episodes completed: click-button A/B/C succeeded; enter-text A failed after
five two-action proposals. The next enter-text B setup was interrupted with no
model prediction. These are incomplete development results, not a matched final
comparison. An operator stop receipt records the boundary and reason.

## Evidence and cause

`episodes/enter-text/0/A/actor-0001-raw.json` contains a TYPE action with the exact
requested value and a CLICK action. The interface correctly rejected this batch
under the frozen one-action contract; it did not discard a valid single TYPE.

`actor-0001-error.json` records `Exactly one native action per step is required`.
However, `actor-0002-started.json` contains only native history's generic
`Agent failed to output in the right format.` The actual rejection reason is
absent. Subsequent raw responses repeated the batch. This establishes missing
feedback delivery; it does not prove that supplying feedback will fix decisions.

## Minimal correction

The local model adapter retains the preceding rejected response, proposal ID,
validation stage, error type and exact reason. The next request includes this
interface feedback alongside the unchanged native prompt/history and schema.
It is cleared after a valid proposal and isolated to the episode's adapter.
The rule applies equally to A/B/C; it does not supply an action or task answer.
No native upstream prompt, checkpoint, decoding setting, task or reset changed.

A regression fixture checks raw-response preservation, exact feedback delivery,
three charged calls and feedback clearing. Related checks verify that rejected
recovery proposals consume incident budgets and that native failure feedback
cannot leak learned or memory context into A.

Two ancillary engineering corrections: completed-step counts no longer count
the loop's final stopping check; worker shutdown wakes an outstanding IPC wait
instead of leaving the coordinator waiting for its timeout. The stopped v1
records retain their original counts and are not rewritten.

## Follow-up

Use a new development directory, `.task2-assets/task2-development-v3`, with the
same four tasks, browser resets, models and budgets. The new plan hashes all
producing sources. Preserve v1's unfavorable outcomes. Do not promote based on
positive completion: audit the actual integration and retain every model failure.

Version 2 was frozen but never executed: a regression test depended on Pydantic,
which belongs to the isolated native environment. Its test stub was made
dependency-free; all 39 checks passed. A source-bound test receipt is now a
mandatory freeze gate. No package was added or upgraded.

## Additional upstream-contract correction before version 4

Source inspection of Browser Use's `Agent.get_model_output` established that
its native `max_actions_per_step=1` slices the validated proposal to its first
action. The local adapter had instead rejected a multi-action list before that
native handling ran. This imposed an additional constraint on the baseline.
Version 3 was stopped while models were loading, before any episode, to correct
that mismatch.

Version 4 accepts the native nonempty action-list schema, preserves the complete
validated proposal, and leaves first-action selection to the upstream agent.
The executor still receives exactly one action; its selected index and deferred
count are explicit evidence. No TYPE/value is inserted, no later proposed action
is automatically executed, and the browser is observed again before the next
step. Strict single-JSON parsing and exact validation feedback remain active.

An actual native `Agent.get_model_output` fixture passed with a two-action stub
proposal: the first TYPE and its whitespace-sensitive value were preserved;
the second CLICK was deferred. This made no GPU calls or browser episodes. All
39 regressions passed. The new matched directory is
`.task2-assets/task2-development-v4/`. The earlier strict-rejection failure is
not rewritten or pooled with the corrected comparison.
