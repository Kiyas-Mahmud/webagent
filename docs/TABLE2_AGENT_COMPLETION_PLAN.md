# Final agent development and comparison plan — 2026-09-12

> **Forward plan superseded:** after reviewing the trained action/grounding
> results, the next proposed implementation is the separately named H0–H3
> [hybrid-agent plan](TABLE2_HYBRID_AGENT_IMPLEMENTATION_PLAN.md), with its
> [ordered TODO](TABLE2_HYBRID_AGENT_TODO.md). It has not been implemented.
> The 48-episode E-profile development comparison below is an earlier proposal,
> not a queued run. Preserve the completed E-profile results.

The trained checkpoint is finished. The next deliverable is a working browser
agent built around it, followed by a reproducible comparison. No retraining,
new model, package upgrade or embedding regeneration is planned. This is a
forward plan; it does not claim implementation or evaluation has occurred.

The latest baseline is [development-v6](TABLE2_DEVELOPMENT_V6.md): E0 1/4,
E1 0/4, E2 2/4, E3 2/4. The overlapping-target defect is fixed. Remaining
failures include skipped typing, incorrect sequence order, stale control IDs,
negative learned assessments and a continuation rule that excludes click-only
effects. Memory has not added completion. Preserve all prior results.

## 1. Finish the agent's decision and continuation loop

Use the existing v6 execution interface, BrowserGym and Playwright. Investigate
one bounded development candidate, with two explicitly recorded parts:

- **Planner interface:** present the current goal, current allowed controls,
  observed field values, completed actions and actual last rejection clearly.
  Ask for one immediate action using a current control ID and exact supplied
  value. Test whether the frozen generator itself types before submitting and
  chooses the required sequence order. Keep the four-field action response.
  Do not remap stale IDs or insert missing actions automatically.
- **Continuation:** specify a bounded rule that permits another recovery step
  after an executed nonterminal action, including ordinary clicks. Separate
  permission to continue from a claim of progress or success. Preserve learned
  diagnosis, strategy and assessment, incident/episode budgets and loop stops.
  A click must not receive invented progress credit. Check how the next incident
  is created and charged; do not reset retry counters to escape their limits.

Changing an inference prompt on declared development tasks is legitimate method
development. It is not weight training. Do not select prompts using final
evaluation outcomes, and preserve every tested candidate and failure.

E1's trained action logits, boxes and processor stay unchanged. E2 and E3 share
the same new recovery interface and continuation rules. E3 alone receives memory.
The candidate must not silently replace the trained normal policy with the base
generator or bypass the trained recovery heads.

## 2. Add task coverage before viewing outcomes

Retain the four existing development families, including the difficult ones.
Add two local MiniWoB families for broader interaction coverage, provisionally
`enter-text-2` and `click-tab-2`, after checking their documented interaction
requirements, availability and existing overlap rules. Their HTML files are
already present locally; no new benchmark installation is needed. Assign these
to development before running them. They are not fresh evaluation families later.

Freeze six development families, reset IDs and budgets before comparing current
v6 with the candidate. One reset per family across E0–E3 gives 24 episodes per
profile, 48 for the matched baseline/candidate comparison. Preserve all outcomes;
this is a functional check, not strong evidence of general superiority. Whole
blocks affected by the existing overlap rules are excluded without choosing an
easier replacement after seeing results.

MiniWoB offers a broader collection of interaction environments; task selection
should reflect the claimed capabilities, not observed winning cases.
[Official benchmark repository](https://github.com/Farama-Foundation/miniwob-plusplus).

Acceptance checks concern correct execution and evidence: exact values, current
targets, actual recovery, causal history, bounded continuation, learned
assessment and read-only memory. Record whether multi-step model behavior occurs.
An accurately executed poor decision remains a policy failure, not a broken
test. A scripted fixture is never counted as a model-completed episode.

## 3. Preserve and measure all four pillars

| Pillar | What stays active | What to measure |
|---|---|---|
| P1 resilience | Learned diagnosis/strategy, executed recovery and post-recovery assessment | Recovery execution, assessment versus actual outcome, E2−E1 completion |
| P2 multimodal decisions | Frozen screenshot/task processing and correct pre/post timing | Input/temporal integrity; component evidence separately from completion |
| P3 action and grounding | Six actions, selected-control identity, browser hit testing | Proposal validity, executed actions, stale/invalid targets and task completion |
| P4 corrective memory | Frozen train-only vectors, retrieval, selective advisory context and E2 fallback | Candidates, admitted/exposed examples, concrete action changes, E3−E2 completion |

Keep the 1,974 embeddings and missing corrective values/reflections unchanged.
Do not describe these label-backed records as complete corrective trajectories.
If memory still adds no completion, report that result. Substantive new memory
construction would be a separate scope decision, not an undocumented repair.

## 4. Freeze one evaluation and finish reporting

After this bounded development cycle, classify remaining issues as demonstrated
implementation defects or model/method limitations. Fix genuine defects with
regression evidence; do not launch an indefinite sequence of prompt trials.
If the candidate is technically valid, freeze the selected method with its
limitations. If it is not, retain v6 as the technically audited baseline.

Before another evaluation, freeze the eligible task families and reset list,
sample count, overlap exclusions, budgets and analysis. Retain challenging
families and include task coverage beyond the development demonstrations.
Do not reuse development resets as final evidence. Reusing previously evaluated
families must be disclosed as new-reset evaluation, not unseen-family testing.
The old 120 outcomes remain historical evidence for the old implementation.

E0–E3 are the required primary comparison. They test the recovery and memory
increments; E0/E1 have different policy interfaces. They are not four independent
published competitor agents. A claim of beating external agents requires an
additional reproducible baseline under the same tasks, resets, observations,
tools and budgets. Literature scores from different settings belong in a
qualified related-work table, not a direct ranking.

Use raw reward exactly 1.0 with termination and no truncation for full completion.
Report task-reset paired differences, improved/worsened pairs, confidence
intervals, the registered exact tests and Holm correction for the two primary
contrasts. Report action validity, recovery, memory exposure, calls and latency.
Choose evaluation size for declared precision/resource limits, not by extending
a run until its p-value becomes favourable.

Deliver the frozen profile, code and evidence manifest, Table 2 and per-family
results, representative successful and failed traces, and the manuscript's
supported claims. Better performance is the hypothesis being tested. Correct
execution plus complete reporting is the completion criterion; positive memory
benefit and journal acceptance cannot be guaranteed.

No model run is launched by this document. Current task: implement and validate
the single planner/continuation candidate, then perform the matched development
comparison. Historical locked-test data, Gold images and unrelated jobs remain
outside this work.
