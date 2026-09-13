> **Replication completed.** Read [TABLE2_REPLICATION_RESULTS.md](TABLE2_REPLICATION_RESULTS.md). The readiness/plan record below is retained as history; do not relaunch the completed run.

# Full-path audit and revised Table 2 readiness — 2026-09-10

**READY FOR THE REVISED 24-EPISODE REPLICATION.** No further development test,
retraining, embedding regeneration, package upgrade or dataset review is a
prerequisite. The remaining noncompletions are measured performance limitations;
a positive outcome is not required to execute an honest evaluation.

The package is `/home/aiub/kiyas/table2-evidence/miniwob-revised-replication-v1/`.
Its preflight returned `READY_FOR_REVISED_REPLICATION`. No replication episode
has started, and its result is not yet available.

## Audit findings

| Path | Evidence and conclusion |
|---|---|
| Checkpoint/base loading | Current checkpoint and all 14 pinned base files hash-verified. Strict head loading, frozen evaluation mode and existing state/parity evidence reviewed. No substituted checkpoint or unadapted base used for E1. |
| Observation and action decoding | Existing exact training/runtime processor comparison and two-input numerical full-forward parity pass. Trained argmax maps through the training label inverse. E1's NAVIGATE decisions are not a demonstrated label-translation error. |
| E0–E3 switches | E1 post-action diagnosis/recovery is disabled; its extra generation calls are parameter fallback. E2 enables diagnosis/recovery; E3 adds the frozen memory adapter. E0 is a contextual comparison rather than isolation of a single pillar. |
| Parameter/target execution | Previous name/description and fenced-JSON fixes preserve generated actions. Ambiguous/malformed outputs remain rejected. A separate real-browser synthetic TYPE check verifies coordinates, literal text and replacement semantics. |
| Recovery chronology | Pre/post/recovery observations, executed-action history and immediate recovery transitions remain bound. The failed-action reference advances after a failed recovery; a new independent incident is not invented for each attempt. Incident diagnosis is retained as incident context. |
| Recovery planner inputs | Diagnosis, selected strategy, failed action and observable context now reach generation. The latest ten-call context audit verifies the supplied diagnosis/strategy and suffix hashes. |
| P4 retrieval/admission | All manifest-bound files verified; frozen train-only embeddings, threshold and exclusions unchanged. Exact post-action transition supplies the embedding. Inapplicable top advice abstains without selecting a lower-ranked preferred strategy. Five latest development queries abstained. This is no evidence of extra memory benefit. |
| Budgets and termination | Action/model-call caps and monotonic timeout reviewed. Browser termination alone is not success: positive reward is also required. Premature Submit and repeated ONE produce terminal failure and remain noncompletions. |
| Pairing and scoring | Four-task live audit passes; initial screenshots and E1–E3 outputs match. One CSV exclusion inconsistency was found and corrected below. Both primary contrasts retain all valid paired blocks. |
| Provenance and readiness | 163 producing source files bound and archived; model/browser versions and six exact reset seeds verified. Original 484 evaluation artifacts remain unchanged. Locked-test rows were not read; no new model/browser campaign was launched during this audit. |

The existing negative/partial results do not prove every possible implementation
path is correct. They also do not justify silently replacing the trained action
policy, changing labels, selecting successful episodes, or forcing memory to
intervene. No additional execution defect explaining the observed E1 choices
was found in this audit.

## Reporting defect corrected this session

The original analyzer excluded a task block with one invalid system from paired
contrasts, but its CSV could still count the other three systems in that block.
`miniwob-feasibility/analyze_campaign_v2.py` now uses the same valid paired task
set for every system's CSV totals and denominators.

A synthetic two-block regression reproduced the discrepancy: old CSV
denominators were E0/E1/E2=2 and E3=1, despite a one-pair comparison. Corrected
denominators are 1 for every system. This fix does not alter the existing live
results, which had no invalid blocks, or explain their task failures.

## Validation and current performance

- Broader audit suite: **241 passed**, covering contracts, protocols, episode
  receipts, recovery/interruption, parameters, processor/checkpoint checks,
  immutable CUDA bias behavior, planner inputs, grounding and memory boundaries.
  This is the named audit subset, not a claim that every repository test passes.
- Synthetic invalid-block reporting regression: **PASS**.
- Latest live development: 16 episodes, zero runtime errors; E0 1/4, E1 0/4,
  E2 2/4, E3 2/4; evidence/context/completion audits **PASS**.
- Remaining failures: E1 selects NAVIGATE without a URL; recovery submits an
  empty field or repeats ONE. E3 has no demonstrated added benefit. These
  outcomes remain in the record and are not additional readiness gates.

Audit evidence: `/home/aiub/kiyas/table2-evidence/table2-readiness-audit-v1/`.
Live evidence: `/home/aiub/kiyas/table2-evidence/miniwob-causal-recovery-v3/`.
The model was not rerun merely to repeat an already-passing development check.

## Frozen replication

Six families × E0–E3 = **24 planned episodes**, subject to the existing overlap
exclusions: click-link, click-option, click-checkboxes, enter-password,
login-user and focus-text. Same reset seeds as the original evaluation; frozen
PC-01 epoch 6/seed 42; 30 executor requests, two recovery attempts per incident,
four per episode, 102 model calls and 600 seconds per episode. No task
substitution or model-output repair during execution.

Use `visible-description-action-interface-v4`, `causal-recovery-input-v1`,
`causal-strategy-admission-v1`, and the corrected analyzer. The plan records
prompts, decoding, checkpoint/store/source/package identities, causal input
changes, scoring and the two paired comparisons. The launcher checks those
bindings before execution and prevents overwriting an already-started run.
It leaves any other active GPU compute process untouched.

These six families have already been observed. This is a **post-debugging
replication**, not a new held-out confirmatory test. It can provide the revised
MiniWoB Table 2 with that limitation disclosed; it cannot substantiate claims
about the superseded WebArena design or independent verified-success memory.
The original evaluation remains preserved separately.

Run from the existing repository:

```bash
PYTHONPATH=src .venv/bin/python /home/aiub/kiyas/table2-evidence/miniwob-revised-replication-v1/launch.py --run
```

Omitting `--run` performs preflight only. After execution, the launcher runs
evidence/context audits and paired analysis. Claim a completed Table 2 only
after inspecting the resulting execution completion and reporting every
included/excluded block, both contrasts, and memory exposure. No expected
positive score or extra success target is imposed.
