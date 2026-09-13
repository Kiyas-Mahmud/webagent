# Hybrid Task 2: normal-action adapter

2026-09-12. **Task 2 complete as an engineering milestone. No H-profile live
evaluation has run; final evaluation is not ready.** Next: Task 3, bounded
recovery and continuation.

## Implemented

The new [HybridActionPolicy](../src/web_agent/runtime/hybrid_action_policy.py)
uses the existing frozen base generation method, action decoder, named-target
resolver and parameter provider. Existing E-profile source files were unchanged.

- H0 dispatches only generation. H1/H2/H3 first dispatch the guarded trained
  pre-action callback and supply a detached, versioned advice projection.
- Advice preserves all six probabilities, the selected unmasked argmax, box,
  grounding confidence and pre-action confidence. Executable-class masking is
  rejected. Advice never replaces the generated action.
- Every H system uses the same [normal-action prompt](../configs/eval/table2/miniwob_hybrid_action_prompt_v1.txt).
  H0 omits the advice section and cannot receive a trained-policy dependency.
  One bare object with all four fields is required; multiple objects, duplicate
  keys and Markdown fences are rejected. Issued values remain intact.
- The runner supplies `HybridActionContext`: current controls, episode/observation
  binding, exact completed action parameters and the last rejection. Completed
  actions must match every causal-history action ID and fingerprint. Missing
  history, stale context and altered parameters fail before inference. The
  context is detached before entering generation.
- The adapter records the guarded trained decision, advice, prompt/context,
  raw response, target resolution and resulting decision. Receipts never claim
  execution. Concrete actions and actual browser outcomes are separate records
  owned by the executor/runner, as demonstrated by the browser fixture.
- Trained advice and base generation each pass through one guarded
  `CallablePolicyAdapter`, which charges before dispatch. H0 costs one normal
  model call; H1/H2/H3 cost two. The existing parameter fallback remains separately
  charged. A remaining allowance of one permits advice but blocks generation.
  Failed dispatches and parse rejections remain recorded.

H0/H1 reject learned assessment entry points. H2/H3 expose unchanged delegated
assessment methods for the next integration task. This adapter never retrieves
or writes memory.

## Verification

**152 focused tests passed**, covering the new adapter, existing named-target and
overlap rules, action parameters, model-call accounting, processor parity and
Qwen runtime contracts. This is a focused result, not a claim about the full suite.

The [browser fixture](../scripts/check_hybrid_action_adapter.py) executed TYPE,
SELECT, CLICK, PRESS_KEY, SCROLL and NAVIGATE through the adapter, deterministic
parameter provider, refreshed v6 target validation and shared Playwright action
code. Exact text, option, checkbox, focus, scroll and navigation effects passed.
The six proposals carried growing completed-action history. Advice and generated
actions intentionally differed to verify that advice cannot silently replace the
chosen action.

The fixture used the existing model Python for the adapter and browser Python
in a separate process. It made **12 scripted dispatches, zero actual model
inferences, zero memory writes and zero live evaluation episodes**. It loaded no
checkpoint or base weights. Model-chosen multi-step performance is still untested.

Evidence: `/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task2-engineering/`

- `pytest.txt`: final 152-test result.
- `browser-fixture/`: screenshots, raw observations, proposal/advice receipts,
  separate executions, dispatch ledger and PASS result.
- `baseline.json`, `preservation.json`: pre-existing file hashes and preservation.
- `implementation-notes.json`: initial fixture issues and their corrections.
- `sources/`, `completion.json`, `manifest.json`: producing sources and evidence
  bindings. This does not replace Task 5's complete H-runner engineering receipt.

## Remaining integration

Task 3 supplies the declared incident-exhaustion/continuation semantics. Task 4
wires H3 memory. Task 5 must connect `HybridActionPolicy` directly to the H runner
without wrapping it in another charged adapter, pass runner-owned completed
actions/rejections on every decision, and bind proposal receipts to the existing
execution logs. It must also freeze H identities, dependencies, sources and the
new prompt before the 24-episode development run in Task 6.

No episode runner or historical E behavior was changed in Task 2. The development
scope remains the six eligible families and resets frozen in Task 1. Neither
these tests nor the scripted fixture establish a performance improvement.
