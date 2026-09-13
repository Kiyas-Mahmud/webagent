# Final H runner: preparation record and completed execution

**Later update, 2026-09-13:** the [120-episode evaluation finished](TABLE2_HYBRID_FINAL_RESULTS_V1.md)
with audit PASS. H0 8/30; H1/H2/H3 9/30 each. The preparation/readiness narrative
below is historical. The completed package must not be relaunched.

2026-09-13. **Final execution plan frozen and verification PASS. All 30 blocks
are eligible: 120 episodes. Evaluation has not started.**

The [approved selective-memory protocol](TABLE2_HYBRID_FINAL_PROTOCOL_V1.md)
is connected through [run_table2_hybrid_final.py](../scripts/run_table2_hybrid_final.py).
The launcher reuses the existing H0–H3 episode loop, model loaders, prompts,
browser executor and independent auditor unchanged. It installs a scoped final
plan verifier for this entrypoint; historical plans keep their original verifier
and development-only guards. It cannot launch a historical profile.

## Checks completed

- **97 distinct focused tests passed.** The initial 96-test suite covered the
  final gate, existing H runner, interface, memory and continuation. After adding
  the final-only CLI guard, all 27 final-entrypoint tests passed, including the
  new guard test. These counts overlap and must not be summed as 123 tests.
- The scripted fixture exercised the shared loop with an evaluation partition
  across eight H-system fixture episodes. Fake actors/browser observations were
  used: zero actual model inference or live evaluation episodes.
- The final gate accepts audited abstention and null performance. It rejects
  missing/invalid episodes, failed identity/isolation checks, missing recovery
  or assessment, memory mutation, inconsistent query/exposure accounting and
  failed audits. The independent auditor still replays candidate scores,
  applicability exclusions, admitted context and H2 fallback.
- The completed v2 development evidence was independently replayed into a new
  directory. Audit PASS; analysis exactly equals the historical analysis.
  Six queries abstained; full memory-generation coverage remains false.
- **30 reset-only browser probes** bound actual task goals and replayed existing
  overlap rules against frozen prepared training memory. All blocks eligible;
  no replacement tasks or seeds. Zero executor actions and model calls.
- Source/model/memory identities were checked before and after preparation.
  The standalone final-plan verification passed after freezing.

No probe prompt was adopted. PC-01 epoch 6 seed 42, the pinned Qwen base,
all 1,974 vectors, retrieval threshold and all agent/runtime source files remain
unchanged. No retraining, package changes or additional training seeds occurred.

## Frozen execution package

Output: `/home/aiub/kiyas/table2-evidence/miniwob-hybrid-final-selective-memory-v1`.

Plan SHA-256:
`d6e6cd5ca376ed3f9ed9d94dea0dc103a52ec504c92f9d7e7ad79dd5069da585`.

The plan binds the approved protocol, complete 30-block goal/reset manifest,
overlap decisions, H0–H3 definitions, fixed budgets/scoring/analysis, source
snapshot, artifacts and new engineering receipt. New final configuration is
derived from the exact v2 candidate; verification rejects other agent changes.
The old development configurations are untouched.

Engineering evidence:
`/home/aiub/kiyas/table2-evidence/hybrid-final-runner-v1-engineering/`.
This contains tests, separate development replay, acceptance receipt, preparation
log, standalone verification and readiness/preservation records.

## Next action

From `/home/aiub/kiyas/webagent`, run the existing frozen package:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_table2_hybrid_final.py run \
  --output /home/aiub/kiyas/table2-evidence/miniwob-hybrid-final-selective-memory-v1
```

The command re-verifies identities and checks competing GPU jobs before loading
models. A coordinator lock prevents simultaneous launches. It executes serially,
persists every episode and resumes only unstarted blocks; started partial blocks
are preserved, not repaired. A completed package cannot be relaunched. Missing
episodes or infrastructure failures retain a failed audit and explicit incomplete
paired denominators; no automatic sample backfill is allowed.

The independent auditor reports integrity separately from `live_path_verified`.
Zero legitimate memory exposure does not make an otherwise correct study fail
its amended acceptance rule. It still prevents claims about the usefulness of
delivered corrective advice. Record all policy failures and both primary
contrasts; do not require or promise a positive result.

**Readiness now means the amended study can execute faithfully. It does not mean
the planner's stale targets, skipped typing or memory usefulness were solved.**
