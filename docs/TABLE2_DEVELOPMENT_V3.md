# Planner and continuation development revision — 2026-09-12

Status: implementation and all 16 live development episodes complete. The revised
evidence audit passes; multi-step readiness is not established. This profile
cannot launch final evaluation.
The [completed 120-episode evaluation](TABLE2_MINIWOB_INTERFACE_V2_RESULTS.md)
remains historical evidence with E0/E1/E2/E3 counts 6/0/6/6 out of 30.

## Changes and limits

- The recovery prompt explains all six actions, exact supplied values, completed
  preparation, and one immediate action. It removes the CLICK-shaped formatting
  example. The frozen model still chooses every action, target and value.
- An opt-in observable-effect check allows continuation after an exact visible
  text insertion, changed selected option, or checkbox/radio state change. The
  learned assessment still runs and remains separately logged. Stable control
  identity, before/after observations and a previously unseen state are required.
  Focus alone, arbitrary screenshot changes and toggling back earn no credit.
  This is action-effect evidence, not a prediction of task completion. Masked
  password fields cannot establish exact inserted text and receive no such credit.
- Memory filtering applies necessary conditions to the existing threshold-admitted
  top three. It excludes unsupported BACKTRACK, repetition of an unresolved action,
  missing required corrective arguments, unavailable control capabilities, and
  unobserved destinations. It does not assert semantic usefulness. An empty context
  preserves the complete E2 decision, with exclusions and abstention logged.
- All 1,974 vectors, their ordering, cosine top-three retrieval, threshold, source
  material, checkpoint and decoding remain unchanged. Missing corrective values
  and reflections remain null. E1 action logits, boxes and processor are unchanged.

The episode controller and memory changes default off for existing callers. The
new launcher is [run_table2_development_v3.py](../scripts/run_table2_development_v3.py).
Its configuration, prompt, sources, assets, dependencies and engineering receipt
are hash-bound before any episode starts. The auditor independently reconstructs
observable effects and memory admission from saved evidence.

## Frozen development protocol

Four existing development families (`click-button`, `enter-text`, `click-test`,
`click-button-sequence`) across E0–E3: 16 episodes. The campaign and reset seeds
match the previous accepted development package. Budgets remain 30 executor
requests, two recovery attempts per incident, four per episode, 102 model calls,
and 600 seconds; checkpoint seed 42. Full completion requires termination without
truncation and raw reward exactly 1.0. Every outcome is retained.

These combined changes test feasibility; this small check cannot attribute a
change to one individual feature or establish a general performance improvement.

## Engineering evidence

140 focused tests passed. A real Playwright fixture verified exact text entry,
selection, checkbox state changes, and refusal to credit focus/repeated states.
A scripted full-controller fixture verified that negative learned assessments
remain active while observable text entry permits another bounded incident.
These fixtures made no model calls and are not live pilot results.

Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-development-v3-engineering/`.
Live package: `/home/aiub/kiyas/table2-evidence/miniwob-development-v3/`.

## Live outcome

| System | Previous matched development | This revision |
|---|---:|---:|
| E0 | 1/4 | 1/4 |
| E1 | 0/4 | 0/4 |
| E2 | 1/4 | 2/4 |
| E3 | 1/4 | 2/4 |

Both E2 and E3 completed `click-button` and `click-test`. Neither completed
`enter-text` or `click-button-sequence`. All 16 outcomes are present; no
infrastructure failure or overlap exclusion occurred. These are four development
pairs, not another final Table 2 result or proof of a general improvement.

E2 and E3 each generated five recovery proposals, all CLICK. Each executed three
recovery actions; all six learned assessments remained negative. Terminal browser
scoring correctly retained the four successful recovery episodes. On `enter-text`,
both systems clicked Submit without entering text. **No observable intermediate
effect or continuation occurred in the live run.** The new continuation branch is
verified by engineering fixtures, but its live multi-step behavior remains untested.

The sequence task failed at parameter resolution on all four recovery proposals:
the proposed control centers intersected two overlapping button boxes
(`TARGET_POINT_MATCH_COUNT:2`). The next attempt received that rejection. The
implementation did not silently move the click or select another button. A future
interface investigation should distinguish geometric box overlap from actual
browser hit testing while preserving the selected target.

E3 made five retrieval queries. Filtering excluded ten candidate occurrences:
five unsupported BACKTRACK, three repetitions of an unresolved action and two
missing corrective arguments. One CLICK example remained in all five generations.
There were zero complete abstentions in this live run; exact E2 fallback is covered
by the engineering tests. Memory exposure added **zero completed task pairs**.

## Audit correction and preservation

The original post-run audit failed on redacted control metadata, not on browser
execution. The auditor had passed hashed `tag`/`input_type` values to capability
validation. It now restores the saved raw observation only after matching all its
logged values/hashes. The corrected audit replayed all four blocks and passed.
No live episode was rerun or altered.

The original failed audit and producing source snapshot remain in the run package.
Corrected audit, diagnostics and auditor sources are separately preserved at
`/home/aiub/kiyas/table2-evidence/miniwob-development-v3-audit-r2/`.
The audit-only entrypoint checks the original two audit/orchestration files against
their frozen snapshots and records the revised auditor identities; every other
source and asset must still match. Normal execution retains strict current-source
verification. The 140 tests also pass after the audit correction.

The inherited `live_path_verified` flag means recovery execution, assessment and
memory exposure occurred. It does **not** verify the newly added continuation
branch. The separate `development-readiness.json` therefore records final readiness
as false. **Another final evaluation is premature:** typing-before-submission and
live multi-step continuation are unresolved; memory benefit remains unobserved.
No further final run is queued.

All 2,353 files bound by the previous 120-episode archive manifest were verified
unchanged. The user's notebook, checkpoint and 1,974 memory embeddings are preserved.
