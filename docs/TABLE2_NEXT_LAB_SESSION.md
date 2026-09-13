<!-- MINIWOB_INTERFACE_V2_STATUS_START -->
# Table 2: completed evaluation; development revision completed

Latest proposed next work: [H0–H3 hybrid-agent implementation](TABLE2_HYBRID_AGENT_IMPLEMENTATION_PLAN.md)
and [ordered TODO](TABLE2_HYBRID_AGENT_TODO.md). Planning only; no H-system is
implemented or evaluated. The E-profile status below remains historical evidence.

For another agent, start with the [current solver handoff](TABLE2_AGENT_SOLVER_HANDOFF.md):
research aim, all four pillars, execution path, verified work, failures and source map.

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

# Table 2 full-credit evaluation completed — 2026-09-10

The requested run and reporting are complete: **24/24 episodes**, with execution
and context audits PASS. E0/E1 scored 0/6; E2/E3 scored 1/6. E3 received memory
context in all 11 recovery calls, with no added completion over E2.

Read [TABLE2_FULL_CREDIT_RESULTS.md](TABLE2_FULL_CREDIT_RESULTS.md) for the
results, actual failure stages, significance tests and limitations. No new
evaluation is queued. Do not relaunch the completed package or regenerate
embeddings. The evidence and earlier runs are preserved; further development
would be a separate experiment.

## Proposed next work: one targeted development cycle

1. **Test the suspected planner regression.** The added next-action request
   explicitly emphasizes TYPE; the completed run often proposed TYPE for
   non-editable controls. Compare that instruction with a neutral action request
   on the existing development cases, changing one factor at a time. This is
   a hypothesis, not a proven cause. Preserve correct action-type selection and
   record regressions, including the earlier E2 decline.
2. **Improve form-target grounding.** The logged password/login controls have
   empty names. Check actual observable labels/accessible names and carry them
   into the control interface when available. Test ambiguity and incompatible
   actions; never silently convert TYPE to CLICK or guess a target.
3. **Run one matched development check.** Use the four existing development
   families across E0–E3, identical resets and budgets. Verify concrete execution,
   rejection feedback, multi-step progress and memory exposure. Keep all outcomes.
4. **Separate memory usefulness from interface correctness.** With the shared
   action interface stable, examine whether the retrieved training examples
   provide applicable advice and improve completion over E2. Reuse the frozen
   embeddings, keep absent source values null, and avoid outcome-driven threshold
   changes. A positive effect is an empirical question, not an audit requirement.
5. **Decide whether another frozen run is justified.** Require a documented,
   verified change and a fixed protocol before any new full run. Preserve the
   completed Table 2 and report every experiment; do not repeat runs until a
   favorable memory result appears.

This is a proposed development plan, not a launched evaluation. It requires no
retraining, package upgrade, embedding regeneration or repeated dataset review.

## Previous completed runs

Latest follow-up: the requested memory implementation investigation and a
separate 16-episode development run are complete. Memory context now reaches
all five E3 recovery generations, but E2/E3 both scored 2/4; no additional
completion benefit was demonstrated. Read
[TABLE2_MEMORY_CONTEXT_DIAGNOSIS.md](TABLE2_MEMORY_CONTEXT_DIAGNOSIS.md) for the
metric mismatch, implementation change, audited results and remaining failures.
No retraining or embedding regeneration occurred, and no new final run is queued.

The 24-episode scores below remain the completed replication results.

The requested frozen run has finished: **24/24 episodes**. E0/E1 scored 0/6;
E2/E3 scored 2/6 under the frozen benchmark rule. Seven memory queries all
abstained. No additional memory benefit was observed.

Read [TABLE2_REPLICATION_RESULTS.md](TABLE2_REPLICATION_RESULTS.md) for the
checkbox partial-credit caveat, remaining failures and disclosed context-audit
correction. No episodes were rerun and the original results were preserved.

No evaluation is queued. Do not relaunch the completed package or regenerate
embeddings. Next research work is to incorporate the audited results and their
limitations into the thesis; further model/interface development would be a
separate experiment, not a prerequisite to reporting this completed run.

Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-revised-replication-v1/`.
