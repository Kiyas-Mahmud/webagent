# Hybrid Task 5: runner integration and engineering evidence

2026-09-12. **Task 5 complete; 5/7 milestones complete. Next: Task 6.**
The H0–H3 development runner is implemented and its 24-episode plan is frozen.
No hybrid model evaluation ran in this task. Final H evaluation is not ready
until the planned live development run and readiness decision are recorded.

Follow the [TODO](TABLE2_HYBRID_AGENT_TODO.md) and
[implementation specification](TABLE2_HYBRID_AGENT_IMPLEMENTATION_PLAN.md).

## What changed

The [hybrid launcher](../scripts/run_table2_hybrid_dev.py) uses the existing
MiniWoB runner, model loader, subprocess browser, parameter provider and recovery
controller. Its [configuration](../configs/eval/table2/miniwob_hybrid_dev_v1.json)
permits development only. It rejects final evaluation preparation.

| System | Normal acting | Recovery | Memory |
|---|---|---|---|
| H0 | Frozen base generation | None | None |
| H1 | Same generator with unchanged trained advice | None | None |
| H2 | H1 | Learned diagnosis, strategy, execution and assessment | None |
| H3 | H2 | Same recovery path | Frozen train-only advisory context |

These are real H runtime identities, including serialized summaries; historical
E identities and their definitions remain separate. P2/P3 provide multimodal
trained advice and observed-control execution, P1 operates the recovery loop,
and P4 supplies H3's advisory context. The normal generator chooses the action
and value. Trained outputs are advice in this declared H design.

The episode runner now supplies the hybrid actor with completed actions, their
exact parameters, observed outcomes and the latest rejection. Normal parser
rejections consume an executor request and may be followed by another normal
proposal within the existing budgets. Recovery parser feedback is carried back
to normal acting after incident exhaustion. Successful execution clears an
execution rejection even when learned recovery assessment stays negative.

H runs save raw action receipts with hashes committed in the redacted event log.
This fixes replay of TYPE parameters, whose coordinates as well as text can be
redacted. The auditor verifies raw decisions, controls, values, causal history,
trained advice, dispatch counts, continuation counters and H3 shadow/query
receipts. Scripted engineering runs cannot set the live-path readiness flag.

The H protocol has its own identity and explicitly retains the Task 1 seed
protocol/campaign namespace. Without that binding, a new protocol ID would have
changed the already frozen reset seeds. All six original task/reset bindings
were verified against the registered stage-seed calculation.

Primary contrasts are **H2−H1** and **H3−H2**, with the two-test Holm correction.
**H1−H0** is secondary. Completion still requires termination without truncation
and raw reward exactly 1.0. Missing/invalid paired blocks and overlap exclusions
remain explicit; recorded policy failures stay in eligible denominators.

## Verification

Evidence directory:
`/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task5-engineering/`

| Check | Actual outcome |
|---|---|
| Selected regression suite | **330 passed**, across 24 modules |
| Real Chromium controller fixtures | **PASS**: H0–H3, text and click-only sequences; 32 browser requests |
| Learned-assessment fixture | Eight scripted negative assessments; bounded continuation and subsequent normal acting verified |
| Six executor action classes | **PASS**, six additional scripted browser requests |
| Overlapping-control browser fixture | **PASS**, exact selected control and hit testing retained |
| Frozen memory replay | **PASS**, existing vectors only; bound memory files unchanged |
| Assets/environment | **PASS**, 61 file identities and installed dependency versions checked |
| Historical 120-episode E replay | **PASS**, saved analysis reproduced exactly |

The tests cover H identity round trips, unchanged E enumeration, advice and memory
isolation, causal boundaries, raw action preservation, malformed proposals,
recovery rejection feedback, budgets, target validation, scoring and paired
analysis. Tampered advice and raw action receipts fail independent audit.
Existing processor/forward-parity regressions passed; no new checkpoint inference
was performed. This is a selected suite, not a claim that every repository test ran.

The real-browser fixtures use scripted callbacks and a fixture stop condition.
They made **zero actual model inferences** and are not MiniWoB performance results.
Their 68 logical dispatches exercise accounting, including four saved-vector
memory queries. No embeddings were regenerated or packages changed.

`engineering-checks.json` binds 281 source/test files and the verification
receipts. `tested-source/` preserves the corresponding source version. Earlier
failed engineering attempts remain in their logs; final PASS evidence is named
explicitly. The historical audit is written under `historical-replay/`, outside
the completed E archive. No old episode or result was overwritten.

## Prepared next task

The actual development package is:
`/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1/`

It contains a read-only `plan.json`, its SHA-256 and producing source snapshot.
It has **24 planned episodes and zero started task blocks**: H0–H3 across
`click-button`, `enter-text`, `click-test`, `click-button-sequence`, `enter-text-2`
and `click-tab-2`. Task 1 exclusions, reset seeds, seed-42 checkpoint, prompts,
dependencies and 30/2/4/102/600 budgets are retained.

At the next task, from the existing repository:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_table2_hybrid_dev.py run \
  --output /home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1
```

The runner verifies the frozen sources/assets and checks GPU jobs before loading
models. It runs serially and persists results immediately. Existing started
blocks are preserved, not silently replayed. Do not prepare over this directory.

Task 6 must report real H0–H3 completion, model-chosen typing and click-only
continuation, recovery assessment and memory exposure. Memory benefit and better
completion remain unproven. The prepared development run does not authorize or
establish readiness for a final H evaluation.
