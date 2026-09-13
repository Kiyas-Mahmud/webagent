# Final H-study protocol: selective memory, including abstention

2026-09-13. **Protocol amendment approved and frozen; evaluation subsequently completed.**
See [all final outcomes and limitations](TABLE2_HYBRID_FINAL_RESULTS_V1.md):
120 episodes, audit PASS, H0 8/30 and H1/H2/H3 9/30 each.
The user approved evaluating the existing agent with legitimate memory abstention
counted as an outcome. This replaces the prospective requirement that at least
one memory example must reach generation before final evaluation can proceed.
It does not alter any historical readiness record or experiment result.

The amendment was made after viewing the completed E/H studies and the bounded
planner probe, before running the new reset manifest. Disclose that timing in
the thesis. This is a prospective evaluation of an already developed candidate;
it is not a claim that the amendment preceded all research outcomes.

## Frozen system and four pillars

Retain the hybrid interface-v2 agent and its existing prompts. Neither prompt
variant from the eight-call probe is adopted. Reuse PC-01 epoch 6, seed 42,
the pinned unadapted base, processor, decoding and all 1,974 memory vectors with
their existing ordering, top-three cosine retrieval, threshold and filtering.

| System | Active components |
|---|---|
| H0 | Frozen base generated actions, current screenshot/controls and causal history |
| H1 | H0 plus unchanged trained multimodal/action/grounding advice: P2/P3 |
| H2 | H1 plus learned diagnosis, strategy, executed recovery and learned assessment: P1 |
| H3 | H2 plus frozen train-only retrieval and selective advisory context: P4 |

P3's trained box/action predictions remain advisory in this hybrid architecture.
The generator chooses the executed action, target and value. P4 can correctly
abstain; neither admission nor exposure is itself evidence of benefit.

## Amended acceptance rules

Still required: verified source/model/memory identities; meaningful engineering
checks; live recovery execution, learned assessment and bounded continuation;
causal H-system isolation; correct frozen retrieval/filtering and H2 fallback;
zero memory writes; complete raw action/outcome accounting.

No longer required: a positive completion difference, successful model-chosen
typing, or a minimum number of examples reaching generation. Positive completion
was already not an engineering gate; lack of typing remains a measured limitation.

Accept abstention only when the recorded query/candidates and frozen threshold or
applicability decisions justify it and the full no-memory H2 decision is preserved.
A missing query on a required path, exception, disabled retrieval, lost admitted
context, altered threshold or broken fallback is an implementation failure, not
legitimate abstention. If examples are admitted, their delivery must be audited.
If no recovery query is triggered, report that separately from queried abstention.

The completed v2 development satisfies the amended development prerequisites:
audit PASS, two executed/assessed recoveries with continuation, six H3 queries
and six justified abstentions, zero memory writes, and 44 matching H2/H3 proposal
pairs. Full memory-generation path coverage remains **false**. Preserve its
historical readiness-false receipt; the new decision is recorded separately in
`/home/aiub/kiyas/table2-evidence/hybrid-final-protocol-v1/amended-readiness.json`.

## Task/reset scope and budgets

Retain the six previously specified evaluation families: `click-link`,
`click-option`, `click-checkboxes`, `enter-password`, `login-user`, `focus-text`.
They are disjoint from the six H-development families but have been observed in
earlier E studies. No unseen-family claim is permitted.

Freeze repeat IDs 0–4: **30 task/reset blocks × four systems = 120 planned
episodes**, before overlap exclusions. The existing stage-seed mechanism uses
namespace `miniwob-hybrid-final-selective-memory-v1`, matched model seed 42.
All 30 full stage seeds and browser seeds are recorded in the
[machine-readable protocol](../configs/eval/table2/miniwob_hybrid_final_protocol_v1.json).
They are unique by task/seed and have no collisions with the ten archived study
plan manifests checked. New seeds do not guarantee unique goals or novel skills.

Every system receives the same reset and limits: 30 executor requests,
two recovery attempts per incident, four per episode, 102 model calls and
600 seconds per episode. Recovery limits apply only where recovery is enabled.
Do not alter the actor, budgets or prompt after viewing these outcomes.

Reuse the existing overlap rule against the frozen prepared training material:
canonical task ID, normalized exact goal or word-trigram Jaccard at least 0.8.
Check the actual reset goal before running its four-system block. Exclude an
affected whole block without substitution. This automated reuse is not a new
dataset review. No Gold images or locked-test data are accessed.

## Execution and scoring

Run serially in the recorded task/repeat/system order with loaded models reused.
Persist each episode immediately. Preserve interrupted attempts and resume only
unstarted blocks under identical bindings; do not rerun started blocks to replace
failures. A missing system outcome leaves an incomplete block, not a success or
an automatic policy failure. Report infrastructure failures and incomplete blocks
separately; do not silently change the paired denominator or backfill samples.

Full completion requires **termination without truncation and raw reward exactly
1.0**. Keep raw and wrapper rewards distinct. Parsing failures, stale targets,
poor actions, loops and policy budget exhaustion remain failures in eligible
paired denominators. Memory abstention never removes an episode or block.

## Analysis and allowed conclusions

Primary unit: eligible complete task/reset pairs. Report H2−H1 (recovery) and
H3−H2 (selective memory) completion differences, counts and improved/worsened
pairs. Use the existing exact two-sided discordant-pair binomial test, Holm
correction across the two primary contrasts at 5%, and 95% paired bootstrap
intervals resampling within each fixed family: 10,000 resamples, seed 20250831.
Report family denominators; empty eligible samples yield N/R, not a zero effect.
H1−H0 remains secondary. Do not increase the sample count to obtain significance.

Report action validity, executed recoveries, learned assessments, model calls,
latency, memory query count, candidates, rejection reasons, admitted examples,
generation exposure, action changes and completion separately. No-query cases,
queried abstentions and admitted-context cases must remain distinguishable.

H3−H2 estimates the effect of the **whole selective pipeline**, including its
choice to abstain and query cost. If exposure is zero, report that the study did
not test the usefulness of delivered corrective advice. Do not claim memory is
universally ineffective, or claim all four pillars independently improved.
Positive superiority claims require a positive difference and the prespecified
adjusted test; otherwise describe the observed negative, null or inconclusive
effect. H0–H3 are internal ablations, not external competing agents.

## Current execution boundary

**The final execution package is complete and audited.** All 120 episodes are
saved. The [final report](TABLE2_HYBRID_FINAL_RESULTS_V1.md) records the two primary
null effects and observed mechanism limitations. No new evaluation is running.

The frozen protocol JSON still records launch as pending at approval time;
separate preparation and completion receipts record subsequent events without
rewriting that historical artifact.

Protocol SHA-256:
`0daff965fb94b3766998aa85605ae69d91eacc1d1e71270f12a1ac5310c97b5f`.
The separate evidence directory contains the frozen protocol copy, seed-check
bindings, new readiness decision and preservation receipt. No retraining,
embedding regeneration, memory enrichment or package change is authorized or
needed for this amendment.
