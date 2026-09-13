# Recovery input and action-description corrections — 2026-09-10

Three interface problems were identified and corrected without retraining.
In the completed matched development check, E2 and E3 each completed two of
four tasks. Text entry, button sequencing and E1 are still noncompletions; their
remaining failures are not reported as solved.

## Demonstrated implementation problems

1. The recovery callback accepted `recovery_decision` and `failed_action` but
   discarded both when calling the model. It also omitted the supplied causal
   history and visible-control context from generation. The new
   `causal-recovery-input-v1` interface serializes these inputs explicitly.
   It selects observable fields, checks task/history binding, and logs the
   exact suffix and its hash. No verifier reward or future observation enters
   the model. The instruction prompt is unchanged, but the recovery model input
   has changed; this is an explicit development protocol revision.
2. Generated valid JSON inside a single Markdown code fence was rejected before
   action validation. An opt-in interface now unwraps one complete fenced object.
   Strict JSON parsing and action validation still run. Prose, multiple objects,
   duplicate fields and malformed JSON remain rejected. This does not fix the
   earlier stray-quote JSON output by guessing its intended syntax.
3. The prompt permits an element description, while the resolver required an
   exact element label. It rejected `button ONE`, and `target: button` with
   `value: Click Me!`, even though they described visible controls. The opt-in
   `visible-description-action-interface-v4` supports explicit role-plus-label
   matching and a unique compatible bare role. A CLICK value can constrain a
   bare role by exact label. Unknown and ambiguous descriptions still fail;
   no fuzzy match, goal-derived target, ordinal guess or action substitution is
   used. TYPE retains the generated text exactly.

Changes are in `src/web_agent/runtime/causal_recovery_planner.py` and
`src/web_agent/runtime/named_target_policy.py`. E0 uses the same opt-in output
envelope/description parsing; E2/E3 share the revised recovery planner. E1's
trained action selection remains unchanged. Legacy interfaces remain available
through the default options and archived producing sources.

The active development runner is:
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/miniwob_campaign_causal_v3.py`.

## Final matched development check

| Development task | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| click-button | Success | Failure | Success | Success |
| enter-text | Failure | Failure | Failure | Failure |
| click-test | Failure | Failure | Success | Success |
| click-button-sequence | Failure | Failure | Failure | Failure |
| Completed | 1/4 | 0/4 | 2/4 | 2/4 |

All four task families are previously used development tasks, with the same
campaign reset namespace, seed 42, budgets and paired initial states. E1–E3
initial action outputs match exactly. All 16 episodes completed with zero runtime
errors and no overlap exclusions. Ten recovery actions executed and received
post-recovery assessments. One normal action executed.

E2 minus E1 is +50 percentage points on these four development pairs; E3 minus
E2 is zero. The development variants were selected during debugging, so these
are descriptive observations, not final efficacy estimates. The five E3 memory
queries all abstained under the existing applicability gate: E3's completion
does not demonstrate benefit from admitted corrective-memory advice.

## Precise remaining failures

- Text entry: E2/E3 generate `CLICK Submit` while the field is empty. That action
  now parses, grounds and executes. MiniWoB terminates with reward 0. No TYPE
  action was produced in those episodes. E0's object-valued target remains an
  unsupported output shape.
- Button sequence: E2/E3 generate and execute `CLICK button ONE`, then generate
  and execute it again. The second click ends the task with reward 0. The
  resolver does not silently change the second action to TWO.
- E1: repeated NAVIGATE output still lacks a URL and is rejected. The preceding
  audit found exact probability/bbox parity with the model's full forward on
  two saved inputs. These corrections do not change the trained-policy output.
- Other E0 failures remain recorded parser rejections. Passing controller audits
  is not a claim that all generated actions are valid or successful.

## Verification

- Focused interface, PC-01 runtime, action-parameter, recovery and memory tests:
  **140 passed**.
- Independent live evidence audit: **PASS** for all four matched blocks,
  retrieval/admission replay, event-log chains and outcome reconciliation.
- Recovery-context audit: **PASS** for ten generation calls; logged strategies
  and diagnoses match the decisions supplied to the planner, suffix hashes
  match, and resolved action types and TYPE values preserve generated values.
- Separate synthetic TYPE executor check: **PASS**. On the real MiniWoB text
  field, generated executor code entered text with quotes/backslashes and then
  replaced it exactly. This used no model and is not a live policy success.
- All 162 plan-bound source files were verified and archived. All 484 original
  24-episode evaluation artifacts remain unchanged. All six manifest-bound P4
  files pass hash verification. No checkpoint, embedding, package, decoding or
  training-seed change occurred; the notebook kernel was left running.

Evidence directory:
`/home/aiub/kiyas/table2-evidence/miniwob-causal-recovery-v3/` contains `plan.json`,
`results.json`, `analysis.json`, `evidence-audit.json`, `context-audit.json`,
`completion.json`, `tests.log`, raw generation receipts and `source-snapshot/`.

Intermediate development runs are preserved separately: `miniwob-causal-recovery-v1`
exposed the fenced-JSON rejection, and `miniwob-causal-recovery-v2` exposed the
role-description mismatch. Each has its own frozen plan and archived producing
source. They are not pooled with v3 or the final evaluation. The synthetic TYPE
check and original callback audit are under `miniwob-causal-recovery-v1/`.

The original Table 2 evaluation is not overwritten by these development results.
No further evaluation is queued. A later replication must freeze and disclose
the revised interface and prior task exposure. Continuing without retraining is
possible, but all-system success and additional P4 benefit remain unestablished.
