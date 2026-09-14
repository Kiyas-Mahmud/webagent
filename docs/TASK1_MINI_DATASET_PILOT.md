# Task 1: 240-example mini validation pilot

Date: 2026-09-14. **Metadata analysis and selection complete; no agent evaluated.**

Current design: [agent comparison architecture and actual data example](TASK1_AGENT_COMPARISON_ARCHITECTURE.md).
The selected external candidates are Browser Use, Agent S2 and WebVoyager's
adapted assessment components. Their exact versions, backends and runtime
compatibility remain to be verified; no adapter is marked runnable yet.

User direction: first compare existing agents' assessment capabilities on a
small portion of our own dataset. Public-benchmark work and Task 2 integration
are subsequent work. This is a validation pilot, not an independent final test.

## Dataset analysis

Read only the original `split_val.json` metadata at
`/home/aiub/kiyas/webagent_full/data/original/final_data_set_40k/split_val.json`.
No training rows, locked-test rows or Gold images were opened. This was a schema
and sampling analysis, not another annotation-quality review.

The split has 7,861 distinct samples from 6,003 task IDs. Its source SHA-256 is
`42df34295f2b0f81e48a29e4acae1b8b16c862ed2a140c19d2a63c4c950d435c`.

| Executed action in source | Available rows | Selected pilot rows |
|---|---:|---:|
| CLICK | 1,287 | 40 |
| TYPE | 1,296 | 40 |
| SELECT | 1,292 | 40 |
| SCROLL | 1,293 | 40 |
| NAVIGATE | 1,499 | 40 |
| PRESS_KEY | 1,194 | 40 |
| **Total** | **7,861** | **240** |

All source rows contain nonempty before/after screenshot references, goals and
domains. Metadata completeness does not verify image availability or correctness.
There are no `action_value` keys in source labels. Do not invent those values.

There are 1,858 recorded recovery attempts. The existing recovery-transition
builder resolves all 1,858 to consecutive same-task transitions, with zero
reported missing/ambiguous/nonconsecutive transitions. This verifies metadata
linkage, not the semantic truth of recovery labels or image contents.

The explicit `recovery_attempted` flag agrees with non-null `recovery_success`
on every source row. The training loader uses this as the needs-recovery proxy.
Present it as recorded-attempt prediction, not independently labelled necessity.

## Mini selection

Choose 40 cases per executed action using fixed strata, without consulting model
predictions. Within each stratum rank sample IDs by SHA256(`20260914:sample_id`).

| Per-action category | Count | Across six actions |
|---|---:|---:|
| Successful recorded recovery | 10 | 60 |
| Failed recorded recovery | 10 | 60 |
| Successful interaction without recovery | 10 | 60 |
| Unrecovered loop failure | 2 | 12 |
| Unrecovered perception failure | 4 | 24 |
| Unrecovered action mismatch | 4 | 24 |

This deliberately enriches recovery and covers all six actions. It is not the
natural full-split distribution. Aggregate accuracy cannot be compared directly
with saved full-validation accuracy; every comparator, including PC-01, must be
scored on these exact cases. Selection metadata is scoring-only, not agent input.

Actual selection: **240 cases, 235 task IDs, 83 domains**.

| Label | Pilot counts |
|---|---|
| Original interaction outcome | 180 FAILURE; 60 SUCCESS |
| Failure type | 97 ACTION_MISMATCH; 71 PERCEPTION_ERROR; 12 LOOP_DETECTED; 60 NONE |
| Recovery strategy | 68 ALTERNATIVE_TARGET; 33 REPLAN; 19 BACKTRACK; 120 NONE |
| Recovery outcome | 60 succeeded; 60 failed; 120 not attempted |
| Memory storage | 183 true; 57 false |

RETRY and ABORT have no reference examples in this original validation split;
do not claim this pilot measures their recall. No supplement was added to fill
absent categories. Existing evaluation masks are true on all source rows.

## How agents will be tested

1. Pin the three selected existing-agent assessment implementations or explicitly
   adapted multimodal backends, with pinned identities and output contracts.
2. Build the same phase-specific recorded inputs for each system. Give outcome
   assessors before/after screenshots, task/domain and the executed action type.
   For recovery assessment, use the linked failure/post-recovery screenshots and
   the actual recovery action; do not reuse the wrong transition.
3. Score failure/outcome, strategy and storage predictions on eligible cases;
   score observed recovery success on the 120 linked attempted cases only.
4. Keep scoring labels, quotas, strata and future outcomes out of agent prompts.
   Preserve the original model input contract. If optional action classification
   is scored, its pre-action input must not reveal the action-type answer.
5. Save all raw responses and invalid outputs. Report per-class metrics, MCC,
   balanced accuracy and correct denominators. Account for task grouping when
   estimating uncertainty; 240 cases are not 240 independent task IDs.

The linked recovery rows are dependencies, not additional scored examples.
Construct transitions from the complete source metadata **before** filtering to
the mini selection; filtering first would break valid consecutive pairs.

Gold images remain on Kaggle. Before any run, verify the selected image assets
there and use a compatible existing execution environment. The metadata-only
selection does not establish that a local image-based run is permitted or ready.

Do not tune prompts against these 240 cases and then present their scores as
untouched evaluation. Any examples used for adapter debugging must be tracked
and separated from comparative scoring before outcomes are examined.

## Artifacts and monitoring

- [Selection manifest](evidence/task1-mini-validation-240-v1/manifest.json)
- [Sample and recovery dependency references](evidence/task1-mini-validation-240-v1/references.json)
- [Scoring-only labels](evidence/task1-mini-validation-240-v1/scoring-labels.json)
- [Reproducible metadata-only preparer](../scripts/external_agents/prepare_mini_pilot.py)

The preparer refuses a changed source hash or an existing output directory. It
does not open images, import the model stack or generate model predictions.

- [x] Inspect original validation metadata and actual class counts.
- [x] Verify recovery linkage with the existing builder.
- [x] Select and hash-bind 240 examples and scoring labels.
- [x] Select three external candidate components and document the architecture.
- [ ] Pin comparator/backend identities and verify native/adapted interfaces.
- [ ] Freeze question wording, temporal inputs, parser and scoring rules.
- [ ] Verify selected image assets and runtime on the permitted host.
- [ ] Verify harness using separately tracked development examples.
- [ ] Run the mini comparison, preserving all responses.
- [ ] Report results as a balanced validation pilot and decide subsequent work.

No inference or live browser tasks were run. The old 5,000-row
`freeze_sample.py` was inspected but not executed: it opens images and does not
preserve the after-state/recovery fields required by this pilot.
