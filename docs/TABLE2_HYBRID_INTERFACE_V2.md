# Hybrid interface v2: corrections and engineering evidence

> Later-session update: [the 24-episode live v2 run is complete](TABLE2_HYBRID_DEVELOPMENT_V2.md).
> Audit PASS; all systems 2/6. Recovery/assessment/continuation occurred, but
> all six memory queries abstained. Final readiness remains false. The preparation
> status and run command below describe the earlier engineering session; do not
> rerun this completed output.

2026-09-13. **The context and action-format corrections are implemented and
engineering checks pass. No new live model episode has run.** The separate
`miniwob-hybrid-dev-v2` development plan is frozen for the same six task/reset
blocks and H0–H3 systems (24 episodes). Final evaluation remains pending.

## Changes

1. Normal generation now receives an explicit projection of page controls,
   completed action types/targets/values, observed execution/state changes,
   rejection diagnostics and substantive trained advice. System/episode IDs,
   decision hashes containing latency, action IDs and provenance hashes stay in
   receipts. Existing observation-local control IDs remain unchanged; execution
   still validates their current observation and capabilities. Trained processor
   inputs, probabilities, boxes and confidence values are unchanged.
2. Recovery uses the same semantic boundary. The runner still validates full
   episode/incident/observation bindings. Raw planner context and feedback remain
   in receipts; generation receives the learned diagnosis/strategy, exact causal
   action history, preceding rejection and admitted memory examples. Learned
   assessment and retry budgets remain active. Frozen memory content is unchanged.
3. The v2 parser accepts exactly one four-field JSON object, either bare or in
   one complete Markdown fence. It preserves the selected action, target and
   issued value. Multiple objects, duplicate keys, arrays, surrounding commentary,
   missing/extra fields, invalid values and copied action lists remain rejected.
   Stale/ambiguous/incompatible targets remain subject to existing checks.
4. A new normal-action prompt describes the four fields and lists allowed action
   names separately. It removes the literal pipe-separated action enumeration
   from the JSON example. The prompt supplies no task solution or automatic
   preparation action. Recovery instructions and decoding settings are unchanged.
5. Auditing now replays the raw-to-semantic projection, retains raw provenance
   checks, and requires identical initial H1/H2/H3 generation context and prompt
   in v2. A regression deliberately injects provenance and verifies audit failure.

The feature is opt-in through `hybrid_interface_version: 2`. V1 remains the
default for old callers and archived experiments. Do not edit or resume the
completed v1 output with the v2 launcher.

## Verification results

- **166 targeted regression tests passed**, covering both interface versions,
  different system IDs/latencies, repeated observations, detached exact history,
  six action types, fences and invalid proposals, target constraints, controller
  budgets/feedback, recovery-context parity, memory isolation and audit tampering.
- **Real Chromium engineering fixture: PASS.** Eight scripted controller fixtures
  executed 32 requests, including typing and click sequences, eight negative
  learned-assessment callbacks, four frozen-memory queries and memory context
  delivery. There were 68 logical dispatches and **zero actual model inference**.
  These scripted completions are not MiniWoB/model performance results.
- **All 562 saved v1 proposals replayed without execution.** Six previously
  resolved proposals were unchanged; 196 additional responses resolved after
  stripping their single fence; 360 remained rejected. Resolving a saved proposal
  does not show that its action is correct or that the browser task would finish.
  This replay also does not test generation under the new prompt.
- The revised initial semantic contexts match across H1/H2/H3 for all six saved
  task/reset blocks, with identical goals and screenshot hashes.
- The completed v1 study re-audited **PASS with identical analysis** into a separate
  output. All 1,605 files in its completed-run manifest remain unchanged.

Initial failed engineering logs are retained: one new test fixture failed to
vary latency, another referenced `reflection_text` instead of the actual
`reflection` field, and the first browser command omitted the repository root
from `PYTHONPATH`. The corrected full regression and browser runs passed.
No package installation or model change was needed.

## Frozen candidate and next action

The candidate retains PC-01 epoch 6, model seed 42, the pinned base, all 1,974
embeddings, retrieval order/threshold, source material, overlap rules and budgets.
The same previously approved development resets are retained. No retraining,
new embeddings, Gold-image access or locked-test access occurred.

Engineering blockers addressed: model-visible provenance leakage and rejection
of an otherwise single fenced action object. Still unmeasured: whether the new
prompt produces better actions; actual model-chosen preparation and continuation;
and whether live P1 recovery and P4 memory add useful actions or completions.

**Next is one matched v2 development run, not final evaluation.** Its prepared
output has no episode results. After a fresh job check, the exact command is:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_table2_hybrid_dev_v2.py run \
  --output /home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v2
```

Run serially under the prepared plan; preserve every outcome. Report actual
typing, recovery assessment, continuation and memory exposure separately from
completion. Final readiness must be reassessed after that live evidence; a
positive effect is not an engineering readiness requirement.

## Files and evidence

- [Semantic projection and parser](../src/web_agent/runtime/hybrid_interface.py),
  [normal adapter](../src/web_agent/runtime/hybrid_action_policy.py),
  [causal recovery boundary](../src/web_agent/runtime/causal_recovery_planner.py).
- [V2 normal prompt](../configs/eval/table2/miniwob_hybrid_action_prompt_v2.txt),
  [candidate configuration](../configs/eval/table2/miniwob_hybrid_dev_v2.json),
  [launcher](../scripts/run_table2_hybrid_dev_v2.py),
  [new regression coverage](../tests/table2/test_hybrid_interface_v2.py).
- [Engineering receipt](/home/aiub/kiyas/table2-evidence/hybrid-interface-v2-engineering/engineering-checks.json),
  [saved-response replay](/home/aiub/kiyas/table2-evidence/hybrid-interface-v2-engineering/saved-response-replay-r2/report.json),
  [browser fixture](/home/aiub/kiyas/table2-evidence/hybrid-interface-v2-engineering/browser-fixture/engineering-result.json).
- [Frozen unstarted v2 plan](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v2/plan.json),
  [historical v1 replay audit](/home/aiub/kiyas/table2-evidence/hybrid-interface-v2-engineering/historical-v1-audit/audit.json),
  [preservation receipt](/home/aiub/kiyas/table2-evidence/hybrid-interface-v2-engineering/preservation.json).
