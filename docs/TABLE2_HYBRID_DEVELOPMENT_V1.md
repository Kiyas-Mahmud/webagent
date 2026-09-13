# H0–H3 development v1: completed run and readiness decision

2026-09-13. **Task 6 complete: all 24 frozen development episodes ran once;
the execution-evidence audit passed. Final H evaluation is not ready.**
The run exposed a model-visible input-matching defect, extensive malformed
proposals, and no live exercise of recovery, continuation or memory.
This is development evidence; historical E results remain unchanged.

## Actual outcomes

| Family, repeat 0 | H0 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| click-button | 0 | 0 | 0 | 0 |
| enter-text | 0 | 0 | 0 | 0 |
| click-test | 1 | 1 | 1 | 0 |
| click-button-sequence | 0 | 0 | 0 | 0 |
| enter-text-2 | 0 | 0 | 0 | 0 |
| click-tab-2 | 0 | 0 | 0 | 0 |
| **Completed / attempted** | **1/6** | **1/6** | **1/6** | **0/6** |

All six paired blocks are present, with zero overlap exclusions, missing episodes,
infrastructure failures or interrupted attempts. Completion requires termination
without truncation and raw reward exactly 1.0. Policy failures remain counted.

| Resource or execution measure | H0 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| Executor requests, including rejected proposals | 151 | 133 | 127 | 151 |
| Resolved and browser-executed normal actions | 1 | 2 | 2 | 1 |
| Model calls | 151 | 266 | 256 | 303 |
| Sum of episode elapsed seconds | 142.02 | 150.61 | 146.01 | 167.90 |
| Recovery attempts / executed recoveries | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Memory queries / generation exposures | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Elapsed time above excludes model loading and campaign-level verification.
Overall, 6/562 proposals resolved and executed (1.07%); all were CLICK.
There were no executed TYPE actions, multi-action episodes, post-recovery
assessments, observable-progress continuations or memory interventions.

## What failed, and at which stage

The 556 rejected proposals divide into four exact recorded errors:

| Stage / error | Count | Raw evidence meaning |
|---|---:|---|
| Parsing: not one strict JSON object | 390 | Every one used Markdown code fences; the frozen H contract requires bare JSON. |
| Target resolution: unregistered action | 134 | Every one copied `CLICK\|TYPE\|SELECT\|SCROLL\|NAVIGATE\|PRESS_KEY` as the action type. |
| Target resolution: value must be text or null | 29 | The issued value violated the four-field response contract. |
| Target resolution: no compatible match | 3 | The proposed CLICK could not be resolved from its supplied target/value. |

These errors are replayed from the actual raw model responses. The parser followed
the frozen contract; it did not discard a valid bare TYPE proposal. The prompt's
literal enumeration was copied despite its instruction to select one action.
That identifies an interface-usability problem to investigate, not proof of a
particular prompt fix. A fenced proposal may also contain a wrong action or target;
accepting its formatting alone would not establish task success.

H1/H2/H3 each clicked Submit immediately on `enter-text`, without typing, and
the browser terminated unsuccessfully. H0/H1/H2 clicked the correct control on
`click-test`. All six executed actions terminated their episodes. Rejected normal
proposals consumed requests and received rejection feedback but produced no
concrete action-conditioned transition for learned diagnosis. Terminal actions
stopped the episode. Consequently, P1 recovery and P4 retrieval were never entered.
Do not invent a transition or continue after browser termination to exercise them.

## Demonstrated input-isolation defect

At the first `click-test` proposal, H2 and H3 had identical prompt templates,
screenshots and substantive trained predictions, but their model-visible context
differed in five places: episode ID, observation ID, the two advice observation-ID
fields, and `trained_advice.source_decision_sha256`.

The IDs contain the system name. The advice hash binds the entire trained
decision, including measured `latency_ms`; H2 recorded 244.459894 ms and H3
241.789220 ms. Timing and audit provenance therefore enter the generation prompt
even before retrieval. This violates the intended equality of H2/H3 inputs before
memory and makes exact semantic-context replay dependent on bookkeeping.

The source boundary is [trained_advice and HybridActionContext](../src/web_agent/runtime/hybrid_action_policy.py).
The frozen auditor checks learned-prediction parity, receipts and causal record
bindings, but does not check equality of the entire pre-memory generation context.
The earlier fixture checks therefore did not establish live input isolation.

**This exposure is verified; its causal effect on completion is not.** The first
raw H2/H3 responses were identical; later outputs diverged. H3 made zero memory
queries, so its lost `click-test` completion cannot be attributed to retrieved
memory. Preserve the run as observed evidence rather than claiming a memory harm.

## Statistical and four-pillar interpretation

The frozen report computes H2−H1 = 0/6, H3−H2 = −1/6, and secondary H1−H0 = 0/6.
Both primary exact tests have p=1 and Holm-adjusted p=1; neither supports an
improvement. These are descriptive development comparisons with a documented
input confound, not clean causal estimates of recovery or memory effects.

The saved within-family bootstrap intervals collapse to their point estimates
because there is only one reset per family. They do not estimate uncertainty
over new resets and must not be presented as precise generalization evidence.

- **P2/P3:** trained advice was supplied to H1/H2/H3, and actual model-selected
  controls executed when proposals were valid. Benefit was not demonstrated.
- **P1:** the learned recovery branch remained wired and budgeted, but was not
  exercised in this live run. Earlier scripted tests remain engineering evidence.
- **P4:** the 1,974-vector store stayed frozen; no live query or intervention
  occurred. This run cannot measure its usefulness.

## Readiness and bounded next correction

**Final readiness: false.** This does not require positive development results.
The named implementation blocker is `pre_memory_generation_context_isolation`;
the live coverage gap is that no nonterminal model action reached recovery or
continuation. The observed malformed responses and skipped preparation are
separate model/interface limitations.

Next, make a minimal, separately versioned correction that keeps provenance IDs,
timings and their hashes in audit receipts while supplying only stable semantic
context to generation. Preserve exact action probabilities, boxes, values,
causal history and observation-bound execution validation. Check the same boundary
in recovery context. Test paired pre-memory prompt equality using differing real
system IDs and latency values, and repeated identical observations.

After that correction, assess the saved schema failures in a bounded development
check. Any format handling must preserve one model-selected action and value;
never choose an action from the copied enumeration or insert task solutions.
Declare any contract or prompt change before new inference. Do not rerun this
completed profile or launch the final study automatically. A subsequent live
check must distinguish implementation correctness from poor model decisions and
record whether a nonterminal action, assessment, continuation and memory query
actually occur. Retraining, replacement models and regenerated embeddings are
not required for these corrections.

## Reproducibility and evidence

The original frozen profile used PC-01 epoch 6, seed 42, the pinned base and
unchanged memory. Per-episode limits remained 30 executor requests, two recovery
attempts per incident, four per episode, 102 model calls and 600 seconds.
No source, prompt, model setting, memory or dependency changed during execution.
The runner reverified 182 source bindings and 61 asset bindings after the run.

Run root: `/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1/`.
Plan SHA-256: `007eaa91fbbd43605f745af536ad7189dbe3d32ac369cadade174c08a51277bb`.

- [Completed episode results](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1/results.json),
  [original PASS audit](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1/audit.json),
  [frozen analysis](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1/analysis.json).
- [Per-episode execution diagnostics](/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task6-session/episode-diagnostics.json)
  and [exact pre-memory context differences](/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task6-session/pre-memory-context-differences.json).
- [Separate readiness decision](/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task6-session/development-readiness.json),
  [preservation receipt](/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task6-session/preservation.json),
  and [session log](/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task6-session/run.log).

The original audit has `live_path_verified: false` and zero recovery-context
audits. Its PASS establishes the checks it actually ran; it does not override
the separate failed readiness decision. The original audit and completed
historical studies are preserved. The model coordinator and its browser workers
exited normally; the idle notebook kernel and unrelated jobs were preserved.
