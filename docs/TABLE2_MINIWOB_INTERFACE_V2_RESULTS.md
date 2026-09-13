# Table 2: frozen MiniWoB interface-v2 evaluation

Completed on 2026-09-10: **120/120 episodes; final execution and evidence audit
PASS; 30 eligible matched task-reset pairs; no overlap exclusions, missing
episodes or infrastructure-invalid blocks.** These are measured results for
the revised BrowserGym + MiniWoB protocol. Earlier experiment results remain
unchanged and separate.

| System | Completed | Rate | Executed normal actions | Executed recovery actions | Model calls | Mean episode seconds |
|---|---:|---:|---:|---:|---:|---:|
| E0: unadapted base | 6/30 | 20% | 23 | 0 | 30 | 3.08 |
| E1: trained P2/P3 | 0/30 | 0% | 0 | 0 | 180 | 3.56 |
| E2: E1 + P1 | 6/30 | 20% | 0 | 26 | 150 | 6.16 |
| E3: E2 + P4 | 6/30 | 20% | 0 | 28 | 206 | 7.35 |

Episode latency excludes campaign model loading. Executor requests, including
rejections, were respectively 30, 90, 56 and 58. E0 made seven requests rejected
before a concrete action; its 23 logged concrete normal actions executed.
E1 made 90 NAVIGATE proposals rejected by parameter resolution. E2/E3 each made
30 rejected normal NAVIGATE proposals before entering recovery. These policy
failures remain in the completion denominator.

## Prespecified thesis contrasts

| Contrast | Improved / worsened pairs | Completion difference | 95% interval | Exact two-sided p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| E2 minus E1: recovery | 6 / 0 | +20 percentage points | +16.67 to +26.67 points | 0.03125 | 0.0625 |
| E3 minus E2: memory | 0 / 0 | 0 points | 0 to 0 points | 1.0 | 1.0 |

Recovery produced six additional completions, but **does not meet the approved
Holm-adjusted 5% criterion**. Memory produced **no additional completions** and
no worsened completion pairs. Neither primary contrast supports a positive
effect under the prespecified decision rule.

Intervals use 10,000 paired resamples within each fixed task family, analysis
seed 20250831. The memory interval is degenerate because every observed paired
difference is zero; it does not establish that memory's population effect is
exactly zero. The conditional bootstrap interval is not the multiplicity-adjusted
hypothesis test and does not override the Holm decision.

## Completion by family

Each cell is full completions out of five browser resets.

| Family | E0 | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| click-link | 4 | 0 | 1 | 1 |
| click-option | 0 | 0 | 0 | 0 |
| click-checkboxes | 0 | 0 | 0 | 0 |
| enter-password | 0 | 0 | 0 | 0 |
| login-user | 0 | 0 | 0 | 0 |
| focus-text | 2 | 0 | 5 | 5 |

Completion requires termination without truncation and raw reward exactly 1.0.
One E0 episode and one E2 episode received positive partial reward without full
completion; both remain failures. No task or reset was substituted.

## What operated, and what did not improve

- The audit replayed **76 recovery generation contexts**, **54 executed recovery
  actions and their learned assessments**, event-log chains, matched screenshots,
  initial E1–E3 predictions, source/artifact bindings and frozen retrieval.
- E2 attempted 34 REPLAN and four BACKTRACK recoveries; E3 attempted 42 REPLAN
  and four BACKTRACK recoveries. Twelve E2 and eighteen E3 proposals failed target
  resolution. Rejected proposals retained their action and value and consumed
  the existing attempt budgets.
- Five E3 generation contexts contained two completed actions and feedback from
  a negative learned recovery assessment. This exercised the live multi-step
  history path and the distinction between browser execution and assessment.
  Target-rejection feedback also reached subsequent permitted attempts.
- E3 made **46 memory queries**; all admitted examples entered its recovery
  decision context. **42 generated REPLAN proposals** received memory examples.
  The four BACKTRACK attempts used their registered executor path, without a
  planner generation. The summary's 46 `memory_interventions` counts changed
  decision context; it is not a count of helpful actions or extra completions.
- Retrieval returned 17 unique examples across 138 candidate slots. The same
  example, `p4-review-f8a13b214b278923c63eec55`, ranked first in **39/46 queries
  (84.8%)**. Candidate scores ranged from 0.76535 to 0.93494. High similarity and
  context exposure did not translate into a completion increment here.
- All 1,974 stored examples still lack corrective action values and reflection
  text. Source goals/domains make them interpretable; missing information was
  not fabricated. Training memory accuracy concerns storage decisions and does
  not validate this retrieval method or browser usefulness.
- E1's NAVIGATE behavior remains after unchanged-processor and initial-prediction
  parity checks. Its actions were not relabeled or replaced. These checks found
  no basis for treating those particular predictions as an interface exception.

The evidence supports functioning recovery and memory delivery, with limited
task completion and no measured memory benefit. It does not support a claim
that all four pillars independently increase performance. E0 and E1 have
different action-generation interfaces; the registered primary comparisons
are E1→E2 and E2→E3.

## Implementation and provenance

The [implementation record](TABLE2_MINIWOB_INTERFACE_V2_IMPLEMENTATION.md)
describes control labels/state/capabilities, action grounding, detached causal
history, rejection feedback, neutral prompts and versioned memory context.
The seven-context prompt diagnostic and scripted six-action browser fixtures
remain engineering evidence, separate from live results.

The first development package completed 16 episodes but failed its metadata
audit: a task-family string overwrote the saved task specification. That package
and the original failed audit are preserved. After the metadata-only fix and
regression test, the corrected 16-episode development package passed: E0 1/4,
E1 0/4, E2 1/4, E3 1/4. No prompt, model, reset or budget was changed for that
repeat. The final 120 episodes were then run once, serially, with no amendments
to their producing implementation.

The focused 220-test run, additional 66-test run, later projection checks and
15-test post-fix run passed; these sets overlap. All 2,561 hash-manifested files
in five prior experiment archives verified unchanged. The existing notebook
diff was also verified unchanged.

Frozen plan SHA-256:
`66d59483d3a459355e944fd1f2afca5c8fa3cd6889b5b892fa89e348a684dfcf`.

- [Final evidence and generated report](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-evaluation/REPORT.md)
- [Final audit](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-evaluation/audit.json)
- [Primary analysis](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-evaluation/analysis.json)
- [Supplemental descriptive diagnostics](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-evaluation/supplemental-diagnostics.json)
- [Accepted development audit](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-development-r2/audit.json)
- [Historical 24-episode results](TABLE2_FULL_CREDIT_RESULTS.md)

PC-01 epoch 6 / seed 42, the pinned Qwen base, packages, all 1,974 embeddings,
their ordering, cosine top-three retrieval and threshold remained unchanged.
There was no retraining, package upgrade, Gold-image access, locked-test access
or new dataset review. Conclusions concern these six previously observed
families and their new resets, with one trained checkpoint.

A subsequent [P4 usefulness diagnosis](TABLE2_P4_USEFULNESS_DIAGNOSIS.md) traces
all saved queries and paired proposals. It does not change these results.
