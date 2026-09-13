# Training/runtime alignment and the two-stage planning diagnostic

2026-09-13. **Investigation completed; the planner problem is not fixed.**
The new nine-generation diagnostic reproduced the historical baselines and
localized a planning-to-action inconsistency. It did not run browser episodes
or demonstrate improved completion. Existing E/H evaluations remain unchanged.

## Correction to the earlier explanation

The trained checkpoint did not produce the hybrid run's all-CLICK JSON outputs.
Those outputs came from the separate frozen unadapted base generator. Describing
them as “the trained model only learned CLICK” was incorrect.

The selected model was trained on all six actions. The saved aggregate report
contains 24,107 training rows:

| Action | Training examples | Selected epoch-6 validation predictions |
|---|---:|---:|
| CLICK | 3,851 | 0 |
| TYPE | 3,880 | 1,276 |
| SELECT | 3,906 | 384 |
| SCROLL | 4,023 | 0 |
| NAVIGATE | 4,770 | 4,079 |
| PRESS_KEY | 3,677 | 2,122 |

The last column is the predicted class distribution on the 7,861 original
validation examples, not browser completion. TYPE recall was 628/1,296 = 48.46%.
CLICK and SCROLL recall were already zero in that saved validation result.
Training examples establish that a class was represented; they do not establish
that this selected checkpoint predicts it reliably. This does not establish why
the action head underperforms, nor justify changing its label mapping.

Sources: [saved report](../results/qwen2vl_2b_gold_v2_8_dgx/seed_42/full/report.json)
and [per-epoch diagnostics](../results/qwen2vl_2b_gold_v2_8_dgx/seed_42/full/diagnostics.json).
Only aggregate reports were read; no Gold images, locked-test data or new dataset
review were involved.

## Where each output actually goes

| Component | Current implementation | What the latest live evidence establishes |
|---|---|---|
| P2/P3 trained policy advice | The selected checkpoint supplies six-class probabilities and its box to H1–H3. The base generator chooses the executable action. | 22 advice records: 18 NAVIGATE, 4 PRESS_KEY. The 35 normal generated actions were CLICK, including all 22 advised decisions. |
| P3 execution | Current control resolution and BrowserGym/Playwright preserve the generated action/value. | All 35 normal proposals resolved and executed. Earlier browser fixtures exercised all six actions; they do not establish model skill. |
| P1 recovery | Trained diagnosis and strategy select the recovery route; generated REPLAN actions come from the base; trained assessment follows execution. | H2 and H3 each executed three REPLAN clicks and two BACKTRACK keypresses. All ten learned assessments were negative; completion remained 2/6 per system. |
| P4 memory | H3 uses a trained representation for frozen cosine retrieval, followed by strict applicability filtering and advisory generation context. | Five queries, five abstentions, zero examples in generation. Memory had no opportunity to change those recovery generations. |

Dispatch is explicit in [hybrid study wiring](../src/web_agent/eval/table2/hybrid_study.py),
[the action adapter](../src/web_agent/runtime/hybrid_action_policy.py) and
[the two model runtimes](../src/web_agent/runtime/qwen2vl_pc01.py).
This separation is the declared hybrid method, not an accidentally unloaded
checkpoint. Replacing the actor with the trained head would restore a different
system definition rather than silently fix H0–H3.

The [training losses](../src/web_agent/models/loss.py) supervise action
classification, grounding and diagnostic/recovery/memory heads. The action-class
loss is not supervision for producing complete browser-action JSON with typed
values. The [dataset label projection](../src/web_agent/data/gold_dataset.py)
maps `action_type` and `memory_update_flag` to their respective head targets.

## Why strong memory/recovery scores are not the live result

The selected checkpoint's 84.28% memory accuracy measures the binary
storage/update flag. It is not a retrieval relevance score or a browser
completion rate. Recovery-outcome accuracy of 90.85% measures offline assessment
of supplied recovery transitions; it is not a guarantee that the agent generates
an effective corrective action.

All 1,974 current memory items lack corrective values and reflections. The
[strict applicability filter](../src/web_agent/memory/context_applicability.py)
rejects examples requiring missing arguments and unsupported BACKTRACK context.
The latest run excluded 13 candidate occurrences and admitted none. This is a
verified information/admission limitation, not a dropped nonempty prompt or
evidence that memory would be useless with a different method. The earlier
label-exposure studies also failed to show completion benefit; simply forcing
exposure is not an established solution.

Changing labels into advisory hints despite missing arguments would be a new
memory-admission experiment. It must preserve missing fields as null, retain
current-observation grounding and be compared separately. No such change was
made here. All embeddings, ordering, top three and threshold remain frozen.

## New bounded diagnostic: plans versus executable proposals

Implemented [the diagnostic runner](../scripts/probe_hybrid_planning_v2.py) and
[its context-preservation regression](../tests/table2/test_hybrid_planning_probe_v2.py).
Frozen evidence: `/home/aiub/kiyas/table2-evidence/hybrid-two-stage-planning-diagnostic-v2/`.
Read its `plan.json`, nine call receipts and `results.json`.

For each of three saved H2 development observations, the same frozen base ran:
the exact historical action prompt; one generic request for a tentative plan;
then the original action interface with the exact generated plan added as
unexecuted advisory context. There were nine generations and no browser actions,
new episodes or memory queries. Model weights, processor and 128-token greedy
decoding were unchanged. No task solution was inserted by the runner.

| Saved development state | Model-generated plan | Action generated with that plan |
|---|---|---|
| enter-text | Enter the task's text, then press Enter | CLICK the input, value null |
| enter-text-2 | Type the task's text, then press Enter | CLICK the input, value null |
| click-button-sequence | Click ONE, then TWO | CLICK ONE |

All three baseline responses matched historical raw text exactly. All six action
responses resolved. All nine generations ended with EOS without reaching the
token cap. Targets changed in two cases; no TYPE action was generated.

This demonstrates that the base can verbalize text entry in these two states,
while the subsequent action interface still chooses CLICK. Clicking an input
could be a preparatory step, so a single saved-state action cannot establish that
it would never type afterward. Conversely, the plan text cannot be counted as
an executed TYPE, and the sequence's corrected first target is not completion.
SCROLL capability was not tested by this diagnostic.

The exact cause of this inconsistency remains unresolved. The evidence does
not support a parser silently changing TYPE to CLICK, a six-class mapping error,
or token truncation in these calls. Do not describe the two-stage candidate as
a completed agent fix or deploy it as a demonstrated improvement.

## Remaining work and stopping boundary

- [x] Separate trained-head predictions from base-generated actions.
- [x] Verify six-class training coverage and selected-head validation behavior.
- [x] Trace recovery execution and explain the actual memory-admission boundary.
- [x] Implement and execute the declared nine-call two-stage diagnostic.
- [ ] Resolve planning-to-action consistency in a separately declared development
      candidate, with the model supplying the exact action and value. No runtime
      rule may turn a failed CLICK into the task's correct TYPE.
- [ ] Exercise that candidate across successive live observations within the
      same fixed budgets. Record typing, subsequent action, learned assessment
      and failures separately. A first-action diagnostic is insufficient.
- [ ] Evaluate memory intervention only after a functioning action sequence gives
      it a measurable opportunity; preserve the frozen store and report abstentions.
- [ ] Freeze a new final protocol only after the candidate's implementation and
      live execution are auditable. Positive performance is not an audit gate.

No production runtime or prompt was changed by this investigation. No retraining
is proposed as a prerequisite. The new final evaluation remains unlaunched;
neither stronger completion nor a memory benefit has been established.

Validation: the new context-preservation regression passed (1 test); the nine
receipts agree with the frozen diagnostic plan and producing script hash. The
existing v3 source/artifact verification passed after inference, and the user's
notebook matched its preservation hash. The diagnostic process exited normally;
the final GPU process check showed only the existing remote-desktop daemon.
