# Hybrid planner: bounded saved-state diagnosis

2026-09-13. **Eight planned generations completed. Neither intervention solved
the planning failures. No browser episode or memory query ran.** The three
unchanged baselines reproduced their historical raw responses exactly.

## Method and evidence

The plan was frozen before inference at
`/home/aiub/kiyas/table2-evidence/hybrid-planner-probe-v1/plan.json`, SHA-256
`16ad44665328adb46b86b4c646cd049edc5f6751dab69d00a67f6dcdd48c0206`.
It binds three saved H2 normal-generation states from completed hybrid dev-v2,
their screenshots, exact original prompts/context, model configuration, diagnostic
script and reminder. Existing source/artifact bindings passed verification.

The existing loader loaded PC-01 epoch 6 seed 42 and the pinned unadapted base.
The eight calls used the frozen base generator, as the hybrid planner does;
trained advice was reused exactly from the saved observations. Decoding remained
greedy with the registered 128-token cap. Instrumentation counted returned tokens
without changing generation arguments or outputs. Memory was neither queried nor
modified. These are counterfactual single-step proposals, not H-system comparisons.

Each case has an unchanged baseline and a variant appending a generic final
four-field response reminder. The two cases with history also have a history-only
variant: past action objects become text records retaining exact issued values,
old targets and execution outcomes. No other context fields change. No variants
were combined; no additional calls were selected after seeing outcomes.

## All outcomes

| Saved state | Unchanged baseline | Final response reminder | History as past-event text |
|---|---|---|---|
| `enter-text`, proposal 1 | Valid CLICK Submit with the field empty | Valid CLICK empty text field; no TYPE | Not planned: no history |
| `click-button-sequence`, proposal 2 | Rejected stale `o1:c1` | Identical stale proposal | Identical stale proposal |
| `click-tab-2`, proposal 3 | Malformed JSON containing history-style `parameters`; stopped at 128 tokens without EOS | Complete four-field JSON, but rejected stale `o5:c2` | Complete valid CLICK `Tab #3`, resolving to current `o7:c2`; repeats the last executed action and box |

The three baseline calls resolved 1/3 proposals; the three reminder calls also
resolved 1/3. History reformatting resolved 1/2 compared with 0/2 for the matching
baseline states. None of the eight proposals was TYPE. Resolution only establishes
the current target/action contract, not task usefulness or browser success.

For the tab case, the reminder and history variants ended normally after 71 and
70 tokens respectively. The baseline exhausted 128 tokens. This verifies where
that particular malformed output stopped. Its wrong `parameters` schema also
violated the response contract; increasing the token cap would not by itself
establish a valid or useful action. No cap change was tested or applied.

## What this establishes

- The saved text goal and current controls were present. The runtime did not
  discard a generated TYPE: no TYPE was generated in this probe.
- History representation affected the tab response schema in this controlled
  case. It did not correct the repeated action. This supports an input-format
  contribution, not a general explanation of all planning failures.
- Stale sequence targets survived both interventions. Automatic replacement
  with a current target would hide the model's actual proposal and was not done.
- Clicking an empty input could precede typing, but no next observation was
  generated here. It is not evidence of a successful preparation sequence.
- No deployed planner change was adopted from these results. The completed v2
  run remains the current live evidence; no completion improvement is claimed.

Two focused regression tests passed for factor isolation and preservation of
exact history values, with rejection of unrecognized history fields.
`diagnostic-audit.json` checks the eight frozen calls and persisted outputs.
`preservation.json` records historical archive and existing-file preservation.
The diagnostic script is [probe_hybrid_planner_v1.py](../scripts/probe_hybrid_planner_v1.py);
the reminder is [the frozen probe text](../configs/eval/table2/hybrid_planner_contract_probe_v1.txt).

## Next decision and stopping boundary

This bounded diagnosis is finished. Do not repeat these prompts or launch a full
study expecting this probe to improve completion. Current final readiness remains
false under the existing live memory-exposure requirement; the probe does not
satisfy that requirement or demonstrate live typing.

With the current frozen model and memory, the defensible next protocol option is
to evaluate the agent as implemented, treating legitimate memory abstention as an
outcome and allowing null results. That requires an explicit amendment of the
current exposure gate before freezing a new final study. Its claim would be the
effectiveness of this selective memory pipeline on the specified tasks, not the
effectiveness of corrective advice that was never delivered. Historical results
and the lack of memory exposure must remain visible.

If the required thesis claim is instead that substantive corrective memory
improves completion, the current label-only store cannot supply the missing
experience content. Gathering genuine disjoint preparation experience and freezing
a new memory would be a different, separately authorized study; no such change
was made. Neither option guarantees superiority. All four pillars may execute
correctly while a measured contrast remains null.

See [the live v2 results](TABLE2_HYBRID_DEVELOPMENT_V2.md) and
[the monitored TODO](TABLE2_HYBRID_AGENT_TODO.md).
