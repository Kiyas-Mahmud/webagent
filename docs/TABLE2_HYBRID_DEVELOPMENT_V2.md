# H0–H3 development v2: live results and remaining limitations

2026-09-13. **All 24 frozen episodes completed once; execution-evidence audit
PASS. H0, H1, H2 and H3 each completed 2/6 tasks.** The context correction holds
in live execution, and recovery execution/assessment/continuation occurred.
Memory always abstained, so the existing full recovery-and-memory path gate
remains unmet. Final H evaluation has not started.

## Completion and resources

| Family, repeat 0 | H0 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| click-button | 1 | 1 | 1 | 1 |
| enter-text | 0 | 0 | 0 | 0 |
| click-test | 1 | 1 | 1 | 1 |
| click-button-sequence | 0 | 0 | 0 | 0 |
| enter-text-2 | 0 | 0 | 0 | 0 |
| click-tab-2 | 0 | 0 | 0 | 0 |
| **Completed / attempted** | **2/6** | **2/6** | **2/6** | **2/6** |

All six task/reset blocks are present, with no exclusions, infrastructure
failures or interrupted attempts. Completion uses raw reward exactly 1.0,
termination and no truncation. Policy failures remain in the denominator.

| Measure, totals across six episodes | H0 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| Executor requests | 36 | 36 | 39 | 39 |
| Executed normal actions | 7 | 8 | 8 | 8 |
| Recovery proposals / executed actions | 0 / 0 | 0 / 0 | 6 / 1 | 6 / 1 |
| Post-recovery assessments | 0 | 0 | 1 | 1 |
| Memory queries / generation exposures | 0 / 0 | 0 / 0 | 0 / 0 | 6 / 0 |
| Model calls | 36 | 72 | 91 | 97 |
| Sum of episode elapsed seconds | 88.66 | 100.87 | 94.28 | 98.16 |

Elapsed time excludes campaign setup/model loading. The six recovery proposals
per system are across two episodes: two on the sequence task and four on the
tab task. The per-incident limit of two and per-episode limit of four were upheld.

There were 148 normal proposals (31 resolved/executed) and 12 recovery proposals
(two resolved/executed). All 33 executed actions were CLICK. Rejected recovery
proposals consume recovery allowances without becoming executor requests; do
not mix these denominators when reporting action validity.

Both primary differences, H2−H1 and H3−H2, are zero across six pairs; neither has
an improved or worsened completion pair. Exact p=1 and Holm p=1 for both.
Secondary H1−H0 is also zero. The stored bootstrap intervals are [0,0] because
there is one reset per family and every paired difference is zero; they do not
establish precise performance on new resets.

## What now works in the live loop

- Initial H1/H2/H3 prompt/context matching passed in every block. A separate
  replay compared all **44 H2/H3 proposal pairs** (38 normal, six recovery):
  screenshots, prompt hashes, complete model-visible suffixes and raw responses
  matched exactly. No memory examples were admitted in those recovery calls.
  The previous system-ID/timing-hash confound is absent from this run.
- On `click-tab-2`, H2 and H3 each executed one model-proposed recovery click,
  received a negative learned assessment, attempted another recovery and resumed
  normal acting after incident exhaustion. Each ultimately executed four clicks
  and stopped under the existing loop rule. The unresolved incident was not
  relabeled successful, and budgets were not reset.
- Memory retrieval ran six times with the unchanged store and threshold. All
  six abstentions preserved the no-memory decision; H2/H3 generation and final
  outcomes matched. This verifies live retrieval and fallback, not a useful
  memory intervention.

P2/P3 advice and grounding are active; P1 recovery, assessment and bounded
continuation are exercised. P4 retrieval is active, but advisory examples never
reach generation. No TYPE action or successful preparation sequence occurred.
No novel observable-effect credit was recorded; click continuation used the
declared hybrid controller rule, separately from learned success.

## Remaining failures and exact evidence

1. **Preparation is skipped.** All eight text-task episodes (`enter-text` and
   `enter-text-2`, across H0–H3) clicked the observed Submit control on their first
   action. Each terminated unsuccessfully without typing. The issued action was
   CLICK; the interface did not lose or replace a generated TYPE.
2. **Stale targets persist.** Ten of twelve recovery proposals were rejected at
   target resolution. Examples name `o1:c1`, `o3:c2` or `o5:c2` after a newer
   observation supplied current IDs. Another 58 normal proposals reused `o1:c1`
   on the sequence task and were rejected. H2/H3 correctly returned to normal
   planning after two rejected sequence recoveries, then reached their request
   budgets. Explicit stale targets were not silently repaired using their boxes.
3. **Malformed responses persist on the tab task.** Fifty-nine normal proposals
   were not complete strict JSON objects. Saved examples begin emitting nested
   `parameters` resembling history records and end before a complete action
   object. Raw responses establish malformed output; they do not by themselves
   identify whether generation stopped at a token limit. Decoding was unchanged.
4. **No applicable memory context was available among the retrieved candidates.**
   Six queries returned 18 candidate occurrences. Six fell below the fixed
   similarity threshold; the remaining 12 were excluded: five lacked corrective
   arguments and seven requested BACKTRACK without an executed navigation to
   backtrack. The same three IDs account for 15/18 candidate occurrences.
   No replacement candidates, fabricated values or threshold changes were used.

H0/H1 executed two clicks on the sequence task but terminated unsuccessfully.
H0/H1 reached their request budgets on the tab task. These are policy outcomes,
not infrastructure failures. The audit found no new execution/provenance defect
within its checks. Poor proposals remain distinct from implementation errors.

## Readiness decision and next boundary

**The requested v2 development run and audit are complete. Final readiness under
the existing full-path gate remains false:** actual memory context exposure is
zero. This is a coverage limitation, not proof that retrieval is broken. Both
learned post-recovery assessments were negative, but their negativity does not
invalidate faithful execution or require a positive result as a readiness gate.

The v2 interface improves development completion relative to the preserved v1
outcomes (v1: 1/6, 1/6, 1/6, 0/6), but all v2 systems tie. This development change
does not demonstrate an increment from trained advice, recovery or memory and
does not replace the historical E evaluation.

Do not launch another unchanged campaign expecting a different conclusion.
Any further change must address these saved failures on development inputs,
preserve model-selected actions/values and declare a new version. The immediate
methodological decision is how to handle absent applicable corrective content:
retain the measured abstention/null result, or explicitly revise the memory
study within train-only constraints. Do not force admission, invent corrections
or label storage-decision accuracy as browser-memory effectiveness. Final scope
and claims must be frozen before any final H run.

## Evidence and preservation

Run root: `/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v2`.
Plan SHA-256: `a41209da9a44e92f99df9372c6d7f1459defce9e9b265a6c9e70e7c6a1aa48b0`.
PC-01 epoch 6, seed 42, the pinned base, all stored embeddings, six reset seeds,
budgets, dependencies, source and prompts remained frozen. No retraining,
stored-embedding regeneration, package changes, Gold-image or locked-test reads
occurred. The coordinator and browser workers exited normally.

- [Episode results](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v2/results.json),
  [PASS audit](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v2/audit.json),
  [paired analysis](/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v2/analysis.json).
- [Per-episode diagnostics](/home/aiub/kiyas/table2-evidence/hybrid-dev-v2-session/episode-diagnostics.json),
  [failure and memory counts](/home/aiub/kiyas/table2-evidence/hybrid-dev-v2-session/failure-and-memory-diagnostics.json),
  [44-pair live context check](/home/aiub/kiyas/table2-evidence/hybrid-dev-v2-session/h2-h3-live-context-parity.json).
- [Readiness record](/home/aiub/kiyas/table2-evidence/hybrid-dev-v2-session/development-readiness.json),
  [preservation receipt](/home/aiub/kiyas/table2-evidence/hybrid-dev-v2-session/preservation.json).

The original audit remains intact with `live_path_verified: false`; the reason
is missing memory-generation exposure, despite two executed recoveries and two
post-recovery assessments. Historical studies and notebook edits are preserved.
