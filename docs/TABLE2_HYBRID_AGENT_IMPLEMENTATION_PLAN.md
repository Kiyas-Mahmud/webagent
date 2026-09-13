# Hybrid browser-agent implementation plan

Date: 2026-09-13. **Tasks 1–7 complete. Final H evaluation completed; audit PASS.**
The [completed final study](TABLE2_HYBRID_FINAL_RESULTS_V1.md) contains all
120 episodes / 30 eligible matched blocks, with no exclusions or infrastructure
errors. H0 completed 8/30; H1/H2/H3 completed 9/30 each. H2−H1 and H3−H2 were
zero (Holm p=1.0). H3 made two queries, both abstained. No recovery action
executed and no memory example reached generation in this final run.

The approved selective-memory protocol, actor and artifacts remained unchanged.
The result is complete and does not support recovery or memory completion gains.
Use the observed findings for thesis reporting; do not rerun the unchanged study.
Execution checklist: [hybrid-agent TODO](TABLE2_HYBRID_AGENT_TODO.md).

## Progress tracker

**Ordered TODO: 7/7 tasks complete; execution, audit and results archived.**
Evidence: [Task 1 preflight report](TABLE2_HYBRID_PREFLIGHT.md),
[Task 2 adapter report](TABLE2_HYBRID_ACTION_ADAPTER.md),
[Task 3 continuation report](TABLE2_HYBRID_CONTINUATION.md),
[Task 4 memory report](TABLE2_HYBRID_MEMORY.md),
[Task 5 runner report](TABLE2_HYBRID_RUNNER.md),
[Task 6 live results and readiness](TABLE2_HYBRID_DEVELOPMENT_V1.md).

- [x] Write the implementation plan and detailed TODO.
- [x] **1. Verify the starting point** — check jobs, protect existing work,
      verify checkpoint/memory, and freeze development tasks and resets.
- [x] **2. Build action selection** — connect trained advice to the frozen
      generator and execute its selected current control.
- [x] **3. Complete recovery and continuation** — keep learned assessment
      active and allow bounded multi-step acting without resetting budgets.
- [x] **4. Connect H3 memory** — reuse frozen retrieval and verify H2 fallback
      when no advice qualifies; no memory writes.
- [x] **5. Pass engineering checks** — validate actions, targets, history,
      budgets and system isolation; freeze the tested runner/profile.
- [x] **6. Run and audit development** — all 24 H0–H3 episodes saved without
      exclusions; actual behavior and a failed readiness decision recorded.
- [x] **7. Final evaluation and thesis delivery** — 120 episodes complete,
      audit PASS, both primary null results and limitations reported.

This tracker now uses the same task numbering as the ordered TODO. The readiness
decision is recorded in Task 6 even when it blocks final evaluation.

**Later, after the readiness decision:**

- [x] Freeze the final H-study task/reset list, sample count and analysis.
- [x] Run the frozen final evaluation and audit all outcomes.
- [x] Report both primary contrasts and archive the thesis evidence.

Update the current task and completion count as work proceeds. Check a milestone
only when its required evidence is saved; passing tests is not live task success.
Detailed subtasks and completion conditions are in the
[full TODO](TABLE2_HYBRID_AGENT_TODO.md).

This is the next development direction after reviewing the trained model and
the existing browser results. It supersedes the prospective experiment in
[the earlier completion plan](TABLE2_AGENT_COMPLETION_PLAN.md). All E0–E3 system
definitions, results, frozen profiles and evidence remain historical records.
Use new **H0–H3** names for this architecture; never relabel it as the old E1.

## 1. Objective and reason for the change

Build an agent using the existing PC-01 checkpoint, frozen Qwen base and memory.
The model must choose its actions; the browser interface must execute them
faithfully. Measure whether learned advice, recovery and memory help completion.
No performance gain is assumed.

Selected epoch-6 validation results show action accuracy 42.17%, action macro-F1
0.3267, mean box IoU 0.101 and Recall@IoU50 8.09%. The action head predicted no
CLICK or SCROLL on that validation split. Recovery-outcome accuracy was 90.85%
and memory-storage accuracy 84.28%. These are different prediction tasks; their
stronger scores do not establish autonomous planning or useful retrieval.
See [the model results](QWEN2VL_2B_GOLD_V2_8_DGX_SEED42_FULL_RESULTS.md).

The v6 named-control executor works and should be reused. Its planner still
skips typing, chooses the wrong sequence order and returns stale IDs. We will
test a single declared hybrid candidate, not assume DOM grounding solves these
planning errors. See [v6 evidence](TABLE2_DEVELOPMENT_V6.md).

## 2. Proposed systems and the four pillars

| System | Normal action selection | Recovery | Memory |
|---|---|---|---|
| H0 | Frozen base generator, screenshot/current controls and causal action history | No learned recovery controller | None |
| H1 | Same generator and context, plus frozen trained pre-action advice | No learned recovery controller | None |
| H2 | H1 | Frozen trained diagnosis, strategy, generated recovery and learned post-recovery assessment | None |
| H3 | H2 | Identical to H2 before retrieval | Frozen train-only advisory retrieval |

All systems share the action schema, execution interface, basic rejection
feedback, normal-action continuation and total budgets. Recovery is the H2/H3
addition; memory is the H3 addition. H0/H1 must not call the learned recovery
heads or query memory. Record all actual model calls and latency.

| Pillar | Role in this candidate | Limit on the claim |
|---|---|---|
| P1 resilience | Trained transition diagnosis/strategy and post-recovery assessment control the recovery branch in H2/H3 | H2−H1 tests the complete recovery addition, including its extra computation |
| P2 multimodal decision | Registered trained screenshot/task processing supplies pre-action advice; the generator also sees the current screenshot and page context | This comparison does not independently isolate the value of each modality |
| P3 action and grounding | Trained action probabilities and box are advisory; generator selects action/control/value; browser resolves the selected control | This is hybrid grounding, not proof that the trained box head became accurate |
| P4 corrective memory | Trained representation queries the existing frozen store and admits advisory examples only in H3 | H3−H2 tests added completion, not merely context exposure |

H1−H0 is a secondary comparison of trained advice, not the original E1−E0.
The two primary contrasts are H2−H1 and H3−H2. Exposure to trained advice must
not be described as demonstrated influence or benefit without evidence.

## 3. Normal-action interface

Implement one thin generated-action adapter; reuse the existing model loader,
named-target parser, parameter provider and browser worker.

For each decision:

1. Capture the current screenshot and projected visible controls. Bind them to
   episode and observation IDs. Do not read task internals or evaluator state.
2. In H1/H2/H3, run the unchanged trained pre-action processor and forward pass
   once. Preserve all six action probabilities, the argmax, predicted box and
   reported confidence. Do not mask classes or change labels/weights/thresholds.
3. Present those outputs in a versioned `trained_advice` section alongside the
   current goal, current control IDs/capabilities/values, completed actions and
   last rejection. Identify advice as fallible; do not present confidence as
   calibrated probability of success. H0 omits this section and the trained call.
4. The existing frozen base generator returns exactly one object with
   `action_type`, `target`, `bbox`, `value`. Use one shared normal-action prompt
   and unchanged decoding. Instructions cover all six actions and emphasize
   current IDs, exact user-supplied values and unfinished preparation.
5. Resolve a named target uniquely against the current observation. Use the v6
   browser-verified hit point for that same control. Preserve coordinate-only
   proposals and apply their existing validation. Do not move an independently
   predicted coordinate to a different control.
6. Execute, capture the result, persist the record, and obtain the next current
   observation. Record advice, raw generation, resolved action and execution
   separately. A generator action differing from trained advice is an explicit
   H-system decision, not a silently repaired trained-head action.

Do not introduce a second planner model, hard-coded task workflows, automatic
credential entry or another generation call that is hidden from the budget.
Use current observations to assess field state; do not infer a successful edit
from focus or from the model's assertion. Historical control IDs remain invalid.

## 4. Recovery and bounded continuation

H2/H3 retain the existing learned transition assessment, failure trigger,
strategy validation, one-action recovery proposal and post-recovery assessment.
Build the complete H2 decision before any H3 retrieval.

Separate three facts in the logs: **action executed**, **learned assessment**, and
**permission to continue**. The proposed continuation semantics are:

| Condition | Next action |
|---|---|
| Browser termination/truncation, explicit ABORT, loop stop or total budget exhausted | Stop and preserve the outcome |
| No recovery triggered after a nonterminal normal action | Continue normal acting in every H system |
| Recovery executes and learned assessment predicts resolution/progress | Record that prediction; return to normal acting |
| Recovery assessment negative and incident attempt remains | Retry within the same incident using current observation and exact feedback |
| Recovery assessment negative and incident attempts exhausted | Record the incident as exhausted/unresolved; return to normal acting if other stop conditions permit |
| Episode recovery allowance exhausted | No further recovery calls; normal acting may continue within the remaining total budget |

Do not report the exhausted incident as resolved. Do not open a fresh incident
for the same unresolved failure just because an observation ID changed. Maintain
the causal incident identity and spent counters; a new failed transition needs
its own recorded binding. The episode-wide attempt limit remains a hard cap.

Observable text/selection/checkbox effects remain diagnostic evidence. A plain
click needs no invented progress score to permit another normal action. The
learned assessment still decides whether another recovery attempt is needed;
its negative prediction alone no longer ends the whole nonterminal episode.
These are declared H-profile semantics, not a retroactive correction to E runs.

Keep budgets: 30 executor requests, two recovery attempts per incident, four
per episode, 102 total model calls and 600 seconds. Count advice inference,
generation, parameter fallback, diagnosis, assessment and query encoding using
the existing accounting contract. Never reset counters to allow a retry.

## 5. Memory

Reuse all 1,974 vectors, ordering, cosine top three, calibrated threshold and
source-material hashes. Reuse v6 applicability filtering and source context.
No regeneration, new entries, invented reflections or corrective values.

H3 receives examples only after a failure and after the H2 no-memory decision
is preserved. The examples may advise; they may not supply fabricated current
task parameters. Empty admission preserves the complete H2 decision. Log
candidates/scores, exclusions, actual generation context, proposed/executed
actions and final outcomes. Label-backed memory may still add no completion.

## 6. Small implementation surface

Proposed new files, created during implementation:

| File | Responsibility |
|---|---|
| `src/web_agent/runtime/hybrid_action_policy.py` | Advice projection and shared frozen generated-action adapter |
| `configs/eval/table2/miniwob_hybrid_dev_v1.json` | Explicit H-system identities, switches, assets and development tasks |
| `configs/eval/table2/miniwob_hybrid_action_prompt_v1.txt` | One frozen normal-action prompt |
| `scripts/run_table2_hybrid_dev.py` | Thin entrypoint using existing loader/browser/evidence machinery |
| `tests/table2/test_hybrid_agent.py` | Advice isolation, action integrity and continuation regressions |

Extend `episode.py` and recovery accounting only with explicit, default-off
hybrid continuation settings. Reuse `named_target_policy.py`,
`action_parameters.py`, `miniwob_controls.py`, `miniwob_worker.py`,
`causal_recovery_planner.py` and the existing memory modules.

The current reporting logic assumes E0–E3 in some places. Generalize only the
system/contrast configuration required for H0–H3, with explicit profile identity
in every result. Preserve historical E parsing and replay. Do not store H1 as
E1 and rename it only in the report. Avoid a new browser framework, service,
repository or agent orchestration layer.

## 7. Verification and one live development cycle

Engineering acceptance:

- Frozen advice matches the original trained forward output and is bound only
  to the current pre-action observation; H0 receives none.
- Six-action schema, exact values, named identity, viewport coordinates,
  overlapping controls, occlusion, stale IDs and unsupported actions behave as
  specified. Nothing inserts a missing TYPE or changes a wrong selected button.
- A scripted multi-step browser fixture demonstrates typing and click-only
  continuation while the learned assessment is negative. Label it engineering.
- Exhausted incidents remain unresolved, cannot recycle their allowance and
  cannot exceed two/four attempts or the total request/call/time budgets.
- H0/H1 have no learned recovery calls; H0/H1/H2 have no memory queries. H3 empty
  admission preserves H2. Retrieval replay and zero writes are verified.
- Audits replay advice, decisions and continuation from raw hash-bound evidence;
  redacted logs are not mistaken for raw state. Scoring retains raw reward.

Retain development families `click-button`, `enter-text`, `click-test`,
`click-button-sequence`. Provisionally add `enter-text-2` and `click-tab-2`, whose
files are already installed. Before any model outcomes, confirm task support
and eligibility using existing overlap rules and freeze the complete manifest.
These new families are development data once used here.

Run one matched reset per eligible family across H0–H3: **24 planned episodes**
before whole-block overlap exclusions. This replaces the earlier proposed
48-episode E-profile development comparison. Freeze tasks, resets, prompts,
source/artifact/dependency identities and budgets first. Serial execution reuses
loaded models and immediately saves every episode. Preserve interrupted attempts.

This tests one candidate with its own H0 baseline; do not select additional
prompts after inspecting these outcomes. A proven implementation defect may be
fixed with regression evidence and a separately preserved revision. Policy
failures remain results. Four or six development pairs do not establish broad
superiority or reliable deployment capability.

## 8. Readiness and final reporting

Produce a readiness record separating: engineering integrity, live execution,
multi-step behavior actually observed, and measured performance. If no model
trajectory exercises continuation, say so; do not substitute the scripted fixture.
A positive completion difference is not an engineering acceptance criterion.

Before any H-profile final run, settle its exact task/reset manifest, sample
count and analysis. Use evaluation resets not used in development; do not tune
against the old 120 outcomes. If previously observed families are reused,
describe new-reset generalization accurately. Keep existing overlap exclusions.

Report completion (terminated, not truncated, raw reward exactly 1.0), per-family
outcomes, paired H2−H1 and H3−H2 differences, improved/worsened pairs, exact
two-sided paired tests and Holm correction for those two primary contrasts.
Keep the registered paired within-family bootstrap procedure (10,000 samples,
analysis seed 20250831), with sample-size limitations. H1−H0 is secondary and
must not be silently promoted after results. Include calls, latency, validity,
recovery and memory exposure, and representative successes and failures.

H0–H3 are internal ablations, not four published competitors. External-agent
superiority needs an additional matched reproducible comparator; do not import
incomparable paper scores into the ranking. A final H result is separate from
the completed E-profile Table 2 evidence and must be named that way in the thesis.

## 9. Fixed constraints and completion deliverable

Keep PC-01 epoch 6/seed 42, pinned base, processor, weights, decoding and memory.
No retraining, model substitution, extra training seeds, package upgrades, Gold
image reads, locked-test reads or new dataset review. Preserve other jobs,
notebook edits, uncommitted work and all completed experiment directories.

Deliver: minimal implementation, regression/browser-fixture evidence, frozen
H0–H3 development manifest, all actual outcomes, audit, remaining limitations and
a final-evaluation readiness decision. No runtime or model execution is performed
by writing this plan. Better performance remains an empirical question.
