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

# Revised Table 2: full-credit MiniWoB evaluation complete

Completed 2026-09-10: **24/24 episodes**, zero runtime errors, overlap exclusions,
invalid paired blocks or episode reruns. All execution/context audits passed.
PC-01 epoch 6 / seed 42 and the existing frozen memory were used without retraining.

Success requires termination without truncation and raw MiniWoB reward 1.0.

| System | Full completions | Rate |
|---|---:|---:|
| E0 | 0/6 | 0.0% |
| E1 | 0/6 | 0.0% |
| E2 | 1/6 | 16.7% |
| E3 | 1/6 | 16.7% |

- **E1→E2 recovery:** +16.7 percentage points; exact/Holm p=1.0.
- **E2→E3 memory:** 0 percentage points; exact/Holm p=1.0.
- Both E2 and E3 completed only `focus-text`.
- E3 made 11 memory queries; all 11 reached recovery generation as context.
  No additional memory completion was demonstrated.
- E2/E3 each had ten recovery generations rejected by target/action validation.

These are six previously observed task families, not a fresh held-out study.
The model still makes invalid action/target choices; passing engineering audits
does not establish model effectiveness. Full results and limitations:
[TABLE2_FULL_CREDIT_RESULTS.md](TABLE2_FULL_CREDIT_RESULTS.md).

Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-table2-completion-v1/`.
This run is complete. No evaluation or retraining is queued.

Earlier results are preserved separately:

- [Previous positive-binary-reward replication](TABLE2_REPLICATION_RESULTS.md):
  E0/E1 0/6, E2/E3 2/6, including the disclosed checkbox partial-credit success.
  Its scores and artifacts have not been changed.
- [Memory-context development](TABLE2_MEMORY_CONTEXT_DIAGNOSIS.md) and
  [subsequent development/readiness history](TABLE2_COMPLETION_READINESS.md).
- Original evaluation: all systems 0/6, preserved under
  `miniwob-revised-evaluation-v1/evaluation/`.
