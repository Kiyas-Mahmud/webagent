<!-- MINIWOB_INTERFACE_V2_STATUS_START -->
# Table 2: completed evaluation; development revision completed

The completed 120-episode result remains unchanged: **E0 6/30, E1 0/30,
E2 6/30, E3 6/30; audit PASS**. See the
[authoritative evaluation results](TABLE2_MINIWOB_INTERFACE_V2_RESULTS.md).
Neither primary contrast met the approved Holm-adjusted 5% criterion.

The authorized [planner and continuation revision](TABLE2_DEVELOPMENT_V3.md)
is implemented and tested: **140 tests passed; all 16 development episodes ran;
corrected evidence audit PASS**. Counts: E0 1/4, E1 0/4, E2 2/4, E3 2/4.
E2/E3 each previously completed 1/4 on these matched development resets.

Remaining limits: both planners still submitted the text form without typing;
no live observable-effect continuation occurred; sequence clicks failed overlapping
target-point validation. Memory filtering excluded ten candidate occurrences but
added no completion. The original audit's redacted-control replay error is fixed,
with its failure and the separate corrected audit preserved.

- [x] Implement clearer planning, bounded observable-effect continuation and selective memory.
- [x] Run engineering checks and one matched 16-episode development check.
- [x] Audit all outcomes, preserve evidence and report failures.
- [ ] Establish typing-before-submission and actual live multi-step continuation.
- [ ] Investigate overlapping target geometry without replacing model-selected actions.

Another final evaluation remains premature and is not queued. No retraining or
embedding regeneration occurred. The checkpoint and all 1,974 embeddings are
unchanged. Historical sections below do not override this current status.
<!-- MINIWOB_INTERFACE_V2_STATUS_END -->

---

The following sections are historical records of earlier profiles and runs.

> **Replication completed.** Read [TABLE2_REPLICATION_RESULTS.md](TABLE2_REPLICATION_RESULTS.md). The readiness/plan record below is retained as history; do not relaunch the completed run.

# Active Table 2 plan: BrowserGym + MiniWoB

Status: revised evaluation completed and audited on 2026-09-10. All 24 episodes completed; all systems 0/6, both paired differences 0 percentage points. See [TABLE2_MINIWOB_EVALUATION_RESULTS.md](TABLE2_MINIWOB_EVALUATION_RESULTS.md). The plan below records the approved design.

Benchmark change approved by the user on 2026-09-09. This document is the active execution direction and supersedes WebArena-specific deployment/task requirements in earlier Table 2 plans. BrowserGym is the interface; MiniWoB is the benchmark. WebArena hosting, maps, website images, URL exports and its 260-episode schedule are not prerequisites for this revised experiment.

## Objective and unchanged comparisons

Measure whether executed failure diagnosis/recovery improves browser-task completion and whether frozen train-only corrective memory adds improvement.

| System | Role |
|---|---|
| E0 | Frozen unadapted selected base model |
| E1 | Frozen PC-01 trained multimodal policy and grounding |
| E2 | E1 plus diagnosis, concrete recovery execution and assessment |
| E3 | E2 plus frozen train-only memory retrieval/intervention |

E1–E2 measures the recovery contribution; E2–E3 measures additional memory contribution. E0–E3 is a total-system contrast. Preserve P1 failure resilience, P2 causal multimodal observations, P3 validated action execution/grounding, and P4 read-only corrective memory. No model replacement to force successful results.

## Completed execution sequence

1. Finish P4 embeddings and read-only retrieval from the existing local training-label-backed memory inputs. Use frozen PC-01 epoch6 seed42, 768-dimensional checkpoint embeddings, L2 normalization, cosine top-3 and train-only threshold calibration. Preserve source identities and exclude evaluation overlap. No additional independent review is requested; report the evidence basis as dataset labels, not independent verification or verified final-task success.
2. Complete a matched E0–E3 development check with the full diagnosis/recovery/memory path. Match tasks, resets, action interfaces and budgets; verify execution, causal observations, memory accounting and task-outcome logging. Invalid outputs and failed recovery remain recorded outcomes. The existing v2 recovery planner and adapter are development implementations, not a completed campaign.
3. Freeze the revised MiniWoB protocol, then run Table 2. Record exact task list, held-out evaluation allocation, reset seeds/repeats, budgets, timeout measurement, success semantics, memory overlap exclusions, frozen prompt/model/store versions and paired analysis before evaluation. Use BrowserGym core/miniwob 0.14.3 and pinned MiniWoB source 7fd85d71a4b60325c6585396ec4f48377d049838 unless a concrete incompatibility requires a separately documented change. Do not reuse already inspected development outcomes as unseen final evaluation evidence.

## Existing assets and limits

- Existing model environment and pinned checkpoint/base snapshot remain unchanged. No retraining, additional training seeds or package upgrades by default.
- Memory inputs: `/home/aiub/kiyas/table2-evidence/p4-local-label-memory-v1/` (1,974 entries). Completed embedded store: `/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1/`; 1,974 entries, CPU verification PASS, threshold 0.7371385097503662. Read-only E3 adapter is wired into the runner; live integration verification passed in development-v5.
- Latest full-runner development diagnostic: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/v2-controls-20260909T100854Z/REPORT.md`. E2 diagnosis/planning ran; recovery target validation rejected both attempts. This historical diagnostic was superseded by development-v5 and the completed revised evaluation; positive P4 benefit was not observed.
- Keep validation/locked-test component data untouched. Use the locally available training material as authorized. No need to move Gold images or rerun completed preparation.
- The revised final MiniWoB task manifest and launcher configuration were frozen before evaluation. Existing WebArena campaign configurations and validators are historical and must not be launched or relabelled as MiniWoB evidence. Reuse benchmark-independent runtime components and implement the revised bindings explicitly.
- Paper Table 2 remains N/R until valid revised evaluation evidence exists. Negative or null results are acceptable; do not tune to obtain a positive result.

## Lab closure checkpoint

P4 generation/calibration/retrieval completed on 2026-09-09. The user deferred
matched development and evaluation to the next session. No browser campaign was
started by the waiting coordinator. See [TABLE2_NEXT_LAB_SESSION.md](TABLE2_NEXT_LAB_SESSION.md).
