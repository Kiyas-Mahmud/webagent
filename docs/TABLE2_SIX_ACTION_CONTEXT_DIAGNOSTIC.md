# Six-action input-context diagnostic

2026-09-13. **Completed: six trained forwards and eighteen generations.** This is an action-generation
diagnostic, not a live MiniWoB completion study or a final Table 2 evaluation.

The immediate question is whether trained advice or causal action history
contributes to the generator's repeated CLICK decisions. The existing model,
processor, six-action prompt and greedy 128-token decoding remain unchanged.

## Fixed cases and interventions

- CLICK: saved `click-button-sequence`, H1, proposal 2, development-v3.
- TYPE: saved `enter-text-2`, H1, proposal 2, development-v3.
- SELECT, SCROLL, NAVIGATE, PRESS_KEY: saved six-action browser-engineering
  observation 4, with four explicit diagnostic user goals. The screenshot and
  its three completed setup actions are real; those setup actions were scripted.
  These four cases test basic instruction following and must not be reported
  as autonomous benchmark task completions.

Each case has three variants: current task/controls; added trained advice;
added trained advice and causal history. Omitted history is absent, not falsely
reported as an empty list. The task, screenshot and current controls are identical
within a case. The expected operation name is audit metadata, not a separate
answer supplied to the generator. Explicit user instructions remain the goal.

Exactly six trained pre-action forwards and eighteen base generations are
planned. No memory query, browser command, adaptive prompt trial or extra
generation is permitted by this diagnostic. Parsing and deterministic parameter
validation reuse the existing implementation without a model fallback. Every
raw output and rejection is retained. Matching the intended action class alone
does not establish correct target/value, execution, or eventual completion.

Evidence: `/home/aiub/kiyas/table2-evidence/six-action-context-diagnostic-v1/`.
The `plan.json` binds case sources/screenshots, original run, prompts, producing
scripts, configuration and decoding. Runtime verification checks the original
model/source identities before and after the run.

Entrypoint: [probe_six_action_context.py](../scripts/probe_six_action_context.py).
The [context-preservation regression](../tests/table2/test_six_action_context_probe.py)
passed before freezing. Production profiles and past results are preserved.

## Results and next decision

| Intended operation | Task/controls only | Added advice | Added advice/history |
|---|---|---|---|
| CLICK | CLICK; valid parameters | CLICK; valid parameters | CLICK; valid parameters |
| TYPE | CLICK; valid parameters | CLICK; valid parameters | CLICK; valid parameters |
| SELECT | CLICK; valid parameters | CLICK; valid parameters | CLICK; valid parameters |
| SCROLL | CLICK; valid parameters | CLICK; valid parameters | CLICK; valid parameters |
| NAVIGATE | CLICK; valid parameters | CLICK; valid parameters | CLICK; stale target rejected |
| PRESS_KEY | CLICK; valid parameters | CLICK; valid parameters | CLICK; valid parameters |

All eighteen raw responses selected CLICK. Seventeen passed deterministic
parameter validation; the remaining response used historical `o3:c2` when the
current controls had prefix `o4`. Rejection was appropriate and no target was
silently substituted. All eighteen generations ended at EOS without reaching
the token cap. The two full-context live-development replays matched historical
raw responses exactly. The six trained predictions were five NAVIGATE and one
PRESS_KEY; no generated CLICK is attributed to the trained action head.

Advice/history changed some targets and values, but removing both did not
restore the other operations in these six states. Thus their presence is not
the sole cause of the all-CLICK behavior here. This does not isolate every
possible prompt or representation effect. CLICK can itself be a preparatory
operation on an input/dropdown; a single proposal is not an episode. These
results do not establish eventual completion under any variant.

No production change is justified simply by removing advice/history. The next
separate candidate tests a compact atomic operation contract with exact field
translation. It is explicitly a combined prompt/context change, not a claim
that this diagnostic proved a particular schema defect.

## Compact atomic-action candidate

Frozen after the diagnostic above and before its own six generations:
`/home/aiub/kiyas/table2-evidence/atomic-action-contract-candidate-v1/`.
Entrypoint: [probe_atomic_action_contract.py](../scripts/probe_atomic_action_contract.py).

The model supplies `operation`, `control`, `argument`. A deterministic translator
copies these into `action_type`, `target`, `value` and sets the unused coordinate
box to null for named-control execution. It never reads the task to choose an
action/value. All six operation names and exact strings are preserved. Existing
target/parameter validation remains authoritative. One JSON object, optionally
inside a single code fence, is accepted; duplicates, extra fields, sequences,
missing fields and unknown operations reject.

The candidate uses the same six frozen states, model weights and decoding, with
current controls only. Exactly six generations; no browser actions or memory
queries. Eleven translation regressions passed before freezing.

All six generations completed with EOS, without hitting the token cap:

| Intended operation | Generated operation | Generated argument | Deterministic validation |
|---|---|---|---|
| CLICK | CLICK | null | Valid parameters |
| TYPE | CLICK (Submit) | null | Valid parameters for the wrong operation |
| SELECT | NAVIGATE | `Finish` | Rejected: not an absolute URL |
| SCROLL | SCROLL | `2` | Rejected: not a registered direction |
| NAVIGATE | NAVIGATE | `Finish` | Rejected: not an absolute URL |
| PRESS_KEY | NAVIGATE | `two` | Rejected: not an absolute URL |

The combined contract/context revision changes action selection: the base emits
SCROLL and NAVIGATE, so it is not universally incapable of emitting non-CLICK
actions. It still fails to select TYPE here and supplies invalid arguments.
This is not a usable production improvement: valid parameters fall from six
of six in the original controls-only diagnostic to two of six, and neither
diagnostic measures completion. No case was executed or counted as success.

## Existing parameter-provider follow-up

The production provider already has deterministic extraction followed by one
frozen-base fallback. The compact-contract diagnostic deliberately checked
deterministic parameters only. To avoid attributing a bypassed fallback path
to a production defect, a separate four-call replay is frozen for **all four**
rejected compact proposals, including those that chose the wrong operation.

Evidence: `/home/aiub/kiyas/table2-evidence/atomic-parameter-diagnostic-v1/`.
Entrypoint: [probe_atomic_parameter_resolution.py](../scripts/probe_atomic_parameter_resolution.py).
The existing provider prompt/settings are reused; selected action classes stay
unchanged, and every original argument, rejection and newly generated fallback
argument remains in the records. No direction or URL is inferred by the audit
code and no browser command is executed. The budget is four physical generations
and four ledger entries.

**Completed: all four fallbacks explicitly returned `{"status":"REJECTED"}`.**
The provider retained each selected action and reported both-stage rejection.
The model-call ledger recorded exactly four calls. This was not a parser
discarding valid parameters or an omitted production fallback. All four
generations ended at EOS without reaching the token cap.

## Current decision and remaining work

The first diagnostic and the bounded candidate/fallback tests are complete:
28 base generations and six trained forwards in total, zero browser actions,
zero memory queries and zero new completion results. Production prompts,
models, decoding and memory remain unchanged. The compact translator is a
tested diagnostic candidate, not promoted into the H runner.

There is evidence that interface design affects the emitted action class, but
not evidence that either tested interface yields a functioning agent. The
missing TYPE decision, incorrect other operations and explicit parameter
abstentions remain unresolved. No further prompt trials are part of this work.

- [x] Run the fixed six-action context diagnostic and retain all results.
- [x] Implement and test one compact atomic-action translator (11 regressions).
- [x] Test six frozen model generations through that candidate.
- [x] Replay all four rejected proposals through the existing parameter fallback.
- [ ] Establish a functioning frozen planner before claiming recovery/memory gains.
- [ ] Verify live action sequences and learned recovery on matched development tasks.
- [ ] Resolve the memory information/admission limitation in a separately declared
      method if necessary; no missing corrective content may be invented.

Testing a different frozen planner would be a new actor variant, not a repair
already demonstrated here. It must retain PC-01 for the declared trained roles,
preserve P4 identities and previous E/H results, and account for all added model
calls. It must not silently replace the selected trained checkpoint. No new
model was downloaded or substituted during this investigation.

Concrete proposed next test, requiring a change to the user's existing-files-only
constraint: obtain the project's already-pinned
`Qwen/Qwen2.5-VL-7B-Instruct` revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5` as an **unadapted frozen planner**.
The identity is recorded in
[the existing backbone configuration](../configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml);
its training/LoRA settings are not instructions to train or attach those adapters.
Test exactly the same six states using the compact atomic prompt, six generations,
the same 128-token greedy decoding and existing validation. Declare its native
model/processor/precision identities separately; this is not a same-model ablation.
No production promotion, additional trials, memory changes, or final evaluation
is included. This candidate has not been downloaded or tested, and success is
not assumed. PC-01 epoch 6 remains the trained checkpoint.
