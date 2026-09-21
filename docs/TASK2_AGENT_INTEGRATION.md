# Task 2: native agent, trained recovery assessment and frozen memory

## Authoritative current status

**2026-09-16: implementation and engineering checks complete; matched live
DEVELOPMENT COMPLETE AND AUDITED (A/B/C each 4/4). The 90-episode final
evaluation is frozen as a plan but was not completed. No final Task 2 result is
available.**
The live status is `.task2-assets/task2-native-evaluation-v1/status.json`.
Read [completed development evidence](TASK2_NATIVE_DEVELOPMENT_RESULTS.md).
Engineering PASS does not establish task completion or memory benefit.

The user authorized the original PC-01 encoder **only for memory queries**.
InternVL remains the actor and diagnosis/recovery assessor. No retraining,
checkpoint substitution, embedding regeneration or existing-environment upgrade
has occurred. Browser Use has its own isolated environment.

The first development attempt was stopped after a verified feedback-delivery
defect; its four completed outcomes and next setup record are preserved.
See [the diagnosis and correction](TASK2_NATIVE_FEEDBACK_DIAGNOSIS.md).
Version 2 was never executed; version 3 stopped during loading. Version 4
restores upstream action-list handling and includes a source-bound regression gate.

## Experiment

| System | Native actor | Added module |
|---|---|---|
| A | Browser Use + frozen InternVL base | Native feedback/recovery only |
| B | Identical native actor | Trained InternVL failure diagnosis, advisory strategy and recovery assessment |
| C | Identical native actor | B plus frozen label-backed memory, queried with its original PC-01 encoder |

The actual native Browser Use Agent runs, including its DOM/screenshot context,
message history, action validation and feedback. This is separate from Task 1's
recorded assessment comparison, which used adapted prompt components.

Native revision: `843819cb8131e1370948d381ede9be7f8366ddc4`, version 0.13.10.
The [isolated dependency lock](evidence/task2-start-v1/dependency-lock.json)
records 104 resolved packages. BrowserGym MiniWoB 0.14.3 owns reset and scoring;
native Browser Use connects to the same local Chromium through CDP.

The native tool set supports CLICK, TYPE, SELECT, SCROLL, NAVIGATE and PRESS_KEY,
plus native back navigation and done. Native `max_actions_per_step=1` executes only the first proposed action.
The complete proposal and the selected first action are both preserved.
No action replacement, task solutions or JSON repair is supplied. Additional
proposed actions are logged as deferred; they are not automatically executed.
Native prompts are preserved; a shared local structured-output instruction adds
its actual response schema. All three systems use the same actor and interface.

Actor: pinned InternVL3.5-8B-HF **base**, NF4 inference, greedy generation, 512
maximum generated tokens. This is a declared Task 2 setting; Task 1's frozen
128-token setting and evidence remain unchanged. Screenshots use the unchanged
single 448×448 patch processor contract. Native history retains six items.
No external image API is used.

Assessor: trained InternVL full epoch 0, seed 42, checkpoint SHA-256
`35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb`.
Existing head tensorization, label maps and recovery logit > 0 rule are preserved.
Exact actor targets and values are logged and available in advisory context;
**the unchanged heads consume action type, not those parameters**.

Memory: existing 1,974 × 768 vectors; original query encoder checkpoint SHA-256
`9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a`.
Cosine top-three retrieval and threshold `0.7371385097503662` are unchanged.
Missing corrective values/reflections remain null. Unsupported examples are
excluded by the existing applicability checks. C first saves the complete
no-memory B-style generation; admitted memory then receives a separate,
charged generation. Empty admission keeps the original response.

## Four pillars and claim boundaries

- P1: trained diagnosis/strategy advises the next native action; the browser
  executes it and the trained recovery head assesses the resulting transition.
- P2: current screenshots, observed page information and causal native history
  reach the actor; adjacent pre/post screenshots reach the appropriate heads.
- P3: the existing native agent supplies action choice, grounding and execution.
  This experiment does not establish a gain from our trained grounding head.
- P4: compatible frozen retrieval supplies advisory training-label examples.
  Query, admission, generation exposure, execution and completion are separate.

B−A measures the added module package. C−B measures the memory package, including
its query and generation overhead. Neither independently isolates every pillar.
Assessment training accuracy does not guarantee browser usefulness. A generic
assessor control is not included, so B−A cannot isolate trained-assessor quality
from the benefit of additional structured feedback.

## Engineering evidence

- **39 CPU regression tests PASS**: temporal/label isolation, six actions and
  exact values, recovery thresholds, strict parsing, total budget accounting,
  rejected-proposal incident limits, A/B memory isolation and paired scoring.
  Log: `.task2-assets/task2-regression-v4.log`.
- **Native proposal-to-action handling PASS**: an actual upstream
  `Agent.get_model_output` engineering fixture verifies first-action selection
  and exact values, with the complete two-action proposal retained.
  `.task2-assets/engineering-native-selection-v1/result.json`.
- **Both memory interface branches PASS** with stub generation and the native
  completion type; empty context preserves the original response, admitted
  context preserves its B shadow and charges both calls. This is not efficacy.
  `.task2-assets/engineering-memory-interface-v1/result.json`.
- **Six real native executor actions PASS**, including exact TYPE value and
  intended SELECT option. Scripted fixture, zero model calls; not task efficacy.
  `.task2-assets/engineering-six-actions-v2/result.json`.
- **Head-logit parity and original-encoder retrieval replay PASS** on captured
  real-browser fixture transitions: six engineering forward calls, zero live
  model episodes and zero memory writes.
  `.task2-assets/engineering-model-v1/result.json`.
- Failed setup probes remain preserved in separate engineering directories.
  Their fixes concern localhost serving, CDP cleanup and native history minimum;
  no completed model episode was discarded or replaced.

## Frozen development and planned evaluation

Development: `click-button`, `enter-text`, `click-test`,
`click-button-sequence`; one matched reset each × A/B/C = **12 episodes**.
Plan: `.task2-assets/task2-development-v4/plan.json`.
All four blocks passed the frozen train-memory overlap rules.

After development audit and readiness review, the proposed evaluation contains
`click-link`, `click-option`, `click-checkboxes`, `enter-password`, `login-user`,
`focus-text`; five resets each × A/B/C = **90 episodes**, before whole-block
exclusions. These are previously observed families, not unseen-family evidence.
No task substitutions follow exclusions.

Common limits: 30 executor requests, 102 total model calls, 600 seconds;
maximum two recovery proposals per incident and four per episode. Parsing
failures consume model calls and recovery attempts. Assessment, query and
memory-conditioned generation calls count against the same total budget.
Weights are loaded before episode timing, with initialization time reported
separately. Checkpoint seed remains 42. Browser seeds are derived by the existing
stage-seed mechanism and bound before viewing model outcomes.

Completion requires independent environment termination and **raw reward exactly
1.0**, with no invalid destination. Agent self-reported done and positive binary
reward alone do not establish completion. Evaluator rewards never reach the
actor or assessment model inputs. Nonterminal steps may continue within budgets;
a learned negative outcome is not a browser execution error.

Sources, prompts, assets, model identities, memory manifest and benchmark files
are hash-bound. Each episode/raw response persists immediately. Existing completed
records are immutable; interrupted or infrastructure-error episodes require
review and cannot silently become successful resumes.

## Reporting and readiness

Report completion, paired B−A and C−B differences, improved/worsened pairs,
exact two-sided paired tests with Holm correction, and 95% intervals from
10,000 paired resamples within each fixed task family (seed 20250831).
Report invalid proposals, executed recoveries, memory exposure, model calls,
latency, exclusions and infrastructure failures separately.

Readiness requires audited execution and a verified complete module path. A
poor policy prediction remains a failure; a positive performance difference is
not an engineering gate. Development readiness is recorded in the linked
results. The final Task 2 run and independent audit were not completed in this
session, so Task 2 remains pending. All historical E0–E3/H0–H3 and Task 1
results remain separate and unchanged.

## Monitoring

```bash
cat .task2-assets/task2-native-evaluation-v1/status.json
tail -n 20 .task2-assets/task2-native-evaluation-v1/execution.log
```

| Delivery | Status |
|---|---|
| Native actor and six-action executor | Engineering PASS |
| Trained head bridge and parity | PASS |
| Original query route, frozen memory replay | PASS |
| Incident isolation and total budget checks | PASS |
| Development freeze | Complete, 12 eligible episodes |
| Matched live development | Version 4 complete, audit PASS, A/B/C each 4/4 |
| Final readiness audit | Ready; limitations documented in development results |
| Final comparison, CSVs, paired analysis | Not run; frozen 90-episode plan only |
