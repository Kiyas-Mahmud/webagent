# Completed Table 2 H-study: selective memory, final v1

2026-09-13. **All 120 episodes completed; independent evidence audit PASS.**
There are 30 eligible matched task/reset blocks, no overlap exclusions, missing
episodes, invalid paired blocks or infrastructure errors. The job has exited.

The study used the [approved protocol](TABLE2_HYBRID_FINAL_PROTOCOL_V1.md) and
[verified final runner](TABLE2_HYBRID_FINAL_RUNNER_V1.md). No prompt, agent,
checkpoint, decoding, memory or budget change was made during evaluation.

## Completion and the thesis contrasts

Full completion requires termination without truncation and raw reward exactly
1.0. All policy failures remain in the eligible paired denominator.

| System | Active additions | Completed | Completion rate |
|---|---|---:|---:|
| H0 | Frozen unadapted base generated actions | 8/30 | 26.67% |
| H1 | H0 plus trained multimodal/action/grounding advice: P2/P3 | 9/30 | 30.00% |
| H2 | H1 plus learned diagnosis and recovery controller: P1 | 9/30 | 30.00% |
| H3 | H2 plus frozen selective train-only memory: P4 | 9/30 | 30.00% |

| Contrast | Difference | Improved / worsened pairs | Exact two-sided p | Holm p | 95% paired bootstrap interval |
|---|---:|---:|---:|---:|---:|
| H2−H1, primary recovery increment | 0 percentage points | 0 / 0 | 1.0 | 1.0 | [0, 0] pp |
| H3−H2, primary selective-memory increment | 0 percentage points | 0 / 0 | 1.0 | 1.0 | [0, 0] pp |
| H1−H0, secondary trained-advice comparison | +3.33 percentage points | 1 / 0 | 1.0 | Not in primary correction | [0, 10] pp |

All comparisons have 30 paired instances. The one improved pair is `click-link`,
repeat 4. The two primary differences are zero on every observed pair, so the
prespecified within-family bootstrap produces degenerate [0, 0] intervals.
Those intervals do not establish equivalence or zero uncertainty on new tasks.
Analysis uses 10,000 paired within-family resamples, seed 20250831, and Holm
correction across the two primary tests at 5%. Neither primary improvement nor
the secondary advice improvement is statistically supported.

## Per-family results

Each cell has five matched resets.

| Family | H0 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|
| click-link | 2/5 | 3/5 | 3/5 | 3/5 |
| click-option | 0/5 | 0/5 | 0/5 | 0/5 |
| click-checkboxes | 1/5 | 1/5 | 1/5 | 1/5 |
| enter-password | 0/5 | 0/5 | 0/5 | 0/5 |
| login-user | 0/5 | 0/5 | 0/5 | 0/5 |
| focus-text | 5/5 | 5/5 | 5/5 | 5/5 |

## What actually executed

| System | Normal proposals | Resolved proposals | Executed normal actions | Recovery proposals / executed | Model calls | Sum of episode seconds |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 62 | 31 | 31 | 0 / 0 | 62 | 148.72 |
| H1 | 118 | 28 | 27 | 0 / 0 | 237 | 278.46 |
| H2 | 118 | 28 | 27 | 2 / 0 | 267 | 304.74 |
| H3 | 118 | 28 | 27 | 2 / 0 | 269 | 304.28 |

Resolved normal-proposal rates were 50.00% for H0 and 23.73% for H1/H2/H3.
Executed requests were 31/62 (50.00%) and 27/118 (22.88%), respectively.
Resolution and browser execution are separate: H1/H2/H3 each had one proposal
that resolved but failed the registered parameter-provider stages. Latencies
above sum recorded episode time; they exclude model loading and do not imply
H3 is faster than H2. Fixed-order single runs do not establish a latency advantage.

All **112 executed actions were CLICK**. No TYPE, SELECT, SCROLL, NAVIGATE or
PRESS_KEY executed. H0 had 30 target-resolution and one parsing rejection;
H1/H2/H3 each had 30 target-resolution and 60 parsing rejections among normal
proposals. These are retained model/interface-contract failures, not discarded
infrastructure outcomes.

P1 remained connected: H2 and H3 each made 28 post-action assessment calls and
two recovery-generation calls. The recovery attempts occurred on `click-link`,
repeat 2. Both systems returned incomplete JSON with a numeric box in the
`target` field. All four recovery proposals were rejected at the logged
`model_parse` stage. **No recovery action executed and no post-recovery assessment
occurred in this final run.** Each system subsequently recorded one continuation
after exhausting that incident. Do not describe this as successful recovery.

The earlier live development did exercise executed recovery and assessment;
that remains separate evidence. The final study's failure to execute a recovery
does not become a positive P1 result because those development checks passed.

## Memory accounting and interpretation

H3 made **two queries in one episode** (`click-link`, repeat 2). The other
29 H3 episodes made no memory query; they are not classified as abstentions.
Both actual queries abstained, admitted zero examples and produced zero
generation exposures. The complete H2 fallback was preserved.

The same three candidates appeared in both queries: six candidate occurrences,
three distinct candidates, each accounting for 2/6 occurrences. All six scores
were above the frozen admission threshold, approximately 0.737139. Applicability
filtering excluded four occurrences for `CORRECTIVE_ARGUMENT_UNAVAILABLE` and
two for `BACKTRACK_NO_EXECUTED_IN_EPISODE_NAVIGATION`. Missing content was not
invented and the threshold was not adjusted.

All 118 matched H2/H3 normal-generation records and both recovery-generation
records had identical prompt hashes, input suffixes, screenshots and raw
responses. H3 added two embedding calls but no admitted memory context or
completion. Zero memory writes or embedding regeneration occurred.

This is a null result for the **whole selective-memory pipeline as implemented**.
Because no advice reached generation, the study does not measure whether useful
delivered corrective experiences could improve completion. Storage-decision
accuracy from model validation is not evidence of retrieval usefulness.

## Research conclusion and limitations

The amended Table 2 H-study is finished. It demonstrates executable hybrid-agent
evaluation with traceable failures; it **does not demonstrate the thesis's
proposed recovery or memory completion gains**. Adding the trained advice yielded
one additional observed completion, with no statistically supported gain.

Claims are restricted to new resets in these six previously observed MiniWoB
families, one PC-01 epoch-6 seed-42 checkpoint and the existing label-backed memory.
The design does not independently isolate improvement from every pillar.
H0–H3 are internal ablations, not external competing agents; no external
superiority claim is supported. The protocol amendment followed development
and historical E results but preceded these new reset outcomes.

Keep the historical E studies separate: the old 120-episode E run was
E0 6/30, E1 0/30, E2 6/30, E3 6/30. Do not pool it with this H run or relabel the
hybrid actor as the original trained-policy E1. No unchanged rerun is needed to
finish this study; use the actual null findings in thesis reporting.

## Evidence and preservation

Completed package:
`/home/aiub/kiyas/table2-evidence/miniwob-hybrid-final-selective-memory-v1/`.
It contains `plan.json`, `completion.json`, `audit.json`, `analysis.json`,
`table2.csv`, `REPORT.md`, producing-source snapshots and every episode's raw
observations, responses, actions and event logs.

Plan SHA-256:
`d6e6cd5ca376ed3f9ed9d94dea0dc103a52ec504c92f9d7e7ad79dd5069da585`.

Session diagnostics and preservation receipts:
`/home/aiub/kiyas/table2-evidence/hybrid-final-evaluation-v1-session/`.
Post-run diagnostics are separate from producing evidence. Completed archives,
frozen model/memory assets, runtime sources, notebook and unrelated work are
preserved. The model process and browser workers have stopped; the existing
notebook kernel and desktop GPU service remain untouched.
