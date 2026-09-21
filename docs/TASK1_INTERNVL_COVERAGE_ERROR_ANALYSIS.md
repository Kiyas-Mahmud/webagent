# Task 1: saved-response coverage and error audit

2026-09-16. **Audit complete; no new inference or changes to completed scores.**
The [seven-row comparison](TASK1_INTERNVL_DUAL_V2_RESULTS.md) remains complete.
This is a post-hoc diagnostic analysis of its saved evidence, not another experiment.

## Main finding

The six prompted configurations mostly abstained: **1,801 of 2,160 responses
(83.38%)**. There were 319 valid predictions (14.77%) and 40 parsing/contract
errors (1.85%). None reached the token limit; all 2,160 ended with EOS.
Increasing the token budget is therefore not supported as a fix by these traces.

The frozen inputs contain generic task descriptions, an action type and a null
action value, without the executed target or action history. This is a concrete
limitation when assessing whether a particular action achieved its intended
effect. The prompts explicitly permit abstention. However, the evidence does
**not** establish that missing context alone caused every abstention, or that
adding context would necessarily improve accuracy. That needs a separate test.

Existing development evidence verifies backend routing, input parity, active
LoRA execution and trained-head parity. This audit found no evidence that an
unloaded adapter or token truncation explains the low coverage.

## Coverage, with every selected case retained

Each interaction cell uses 240 cases; each recovery cell uses 120. Entries are
**valid / abstention / parse error**, not correct / incorrect predictions.

| Configuration | Interaction: valid / abstain / error | Recovery: valid / abstain / error |
|---|---:|---:|
| Browser Use-derived, base | 2 / 238 / 0 | 2 / 118 / 0 |
| Agent S2-derived, base | 26 / 213 / 1 | 12 / 108 / 0 |
| WebVoyager-derived, base | 27 / 210 / 3 | 9 / 111 / 0 |
| Browser Use-derived, trained | 51 / 159 / 30 | 21 / 99 / 0 |
| Agent S2-derived, trained | 35 / 205 / 0 | 17 / 103 / 0 |
| WebVoyager-derived, trained | 68 / 168 / 4 | 49 / 69 / 2 |
| Trained heads | 240 / 0 / 0 | 120 / 0 / 0 |

The trained heads make forced-choice predictions. The prompted configurations
can abstain. Their coverage difference is part of the comparison, not proof of
better reasoning by itself. Head outcome accuracy remains 193/240 (80.42%) and
recorded-recovery accuracy 113/120 (94.17%). These are agreement with dataset
labels, not live browser completion rates.

## What the 40 errors actually contain

| Contract defect | Count |
|---|---:|
| Missing both `failure_type_4` and `reason` | 29 |
| Missing `failure_type_4` only | 7 |
| Missing `reason` only | 1 |
| Invalid failure category with a decisive outcome | 2 |
| Markdown-fenced response instead of bare JSON | 1 |

**39/40 are syntactically valid JSON.** Of those 39, 37 explicitly declare
`ABSTAIN`; two declare `FAILURE` with an invalid category. The fenced response
also visibly contains an abstention, but remains rejected by the frozen parser.
Repairing format alone would mostly convert errors into abstentions, not produce
useful decisive predictions. No response was repaired or rescored.

There is also a demonstrated schema inconsistency: the frozen parser checks
required keys and reason type, then returns an abstention before validating the
failure-category enumeration. Consequently, **701 accepted abstentions contain
a failure label outside the four declared categories**. They contain no scored
prediction and are already excluded from valid-output metrics; this does not
inflate the valid prediction count. A future output contract should explicitly
define the category field for abstention. The existing parser and results remain
unchanged. See [parser](../src/web_agent/eval/task1/core.py) and the
[regression fixtures](../tests/analysis/test_task1_coverage_audit.py).

## Input evidence and its limits

All 360 phase requests have `executed_action.value = null`. The frozen request
projection supplies no executed target, selector, coordinates or action history.
It contains 137 distinct task descriptions, all matching generic formulations
such as “Look for the requested site information for the current task.”
These descriptions do not explicitly give the intended text, destination or
specific task requirement.

The [input builder](../src/web_agent/eval/task1/core.py) copies the supplied task
description and sets the action value to null under the approved contract. This
audit concerns that frozen projection: **it does not establish whether the
original collector captured richer information, or diagnose an export bug.**
No new collector/dataset review was performed. The recorded action type is
provided evidence; outcome and failure labels remain reserved for scoring.

This matters differently across actions. TYPE, SELECT, PRESS_KEY and NAVIGATE
may require the intended argument to judge correctness. CLICK may require its
executed target. LOOP_DETECTED may require earlier history. A visible change can
still support some assessments, but a screenshot change alone does not establish
success against an unspecified goal.

Exact decoded RGB comparison found:

| Phase | Identical before/after pixels | Different pixels |
|---|---:|---:|
| Interaction | 52 / 240 | 188 / 240 |
| Recovery | 33 / 120 | 87 / 120 |

There were **1,353 abstentions on pairs with different pixels**, so identical
screenshots cannot explain all abstentions. Even Browser Use-derived base
abstained on 186/188 changed interaction pairs and 85/87 changed recovery pairs.
Pixel differences can be cosmetic; they are not success labels or a semantic
adjudication of what the action accomplished.

## Model-stated reasons are diagnostic clues, not verified causes

Literal, overlapping regex indicators in the 1,801 abstention reasons found:
970 mentions of uncertainty, 502 of no change/progress, 150 of missing or
unspecified action/target information, and 22 explicitly mentioning `null`.
The exact rules and every response are saved in the audit CSVs.

Of the 502 no-change/progress mentions, 309 concern images with different pixels.
This does not prove the reasons are false: “no meaningful progress” differs from
“no pixel changed.” These text counts were not independently human-adjudicated.
Likewise, whether retained goal/trajectory-oriented instructions encourage
abstention on these short transitions remains an untested prompt hypothesis.

## What can be written in the paper now

> On a fixed validation pilot of 240 interactions and 120 linked recovery
> transitions, the trained assessment heads achieved outcome MCC 0.5583 and
> recovery MCC 0.8844 with complete output coverage. Six adapted prompted
> configurations produced valid predictions for 319/2,160 requests, with 1,801
> abstentions and 40 output-contract errors. We report conditional classification
> metrics together with coverage and all-case accuracy. The result is specific
> to the supplied recorded-transition information and frozen assessment protocol.

Retain all seven rows, errors and abstentions. Display valid denominators beside
MCC/F1 and all-case accuracy with invalid outputs counted as incorrect. Small
common-valid subsets make some paired intervals uninformative; a zero-width
interval on a tiny or single-class subset is not evidence of equivalence.

The trained heads have supervised exposure to this dataset's training portion;
base prompting does not. Trained decoder prompting also differs from reading
trained classification heads. These are adapted assessment components, not
complete native Browser Use, Agent S2 or WebVoyager agents. This validation
follow-up cannot establish unseen-test generalization, superior live agents,
executed recovery improvement, or memory benefit.

## Next bounded work, before any new comparison

1. Audit whether permitted execution records contain the actual task goal,
   executed target and action argument. Trace their provenance into a proposed
   input contract. Use observable execution metadata, not evaluator answers or
   reference gold bounding boxes; preserve truly missing values.
2. Specify an unambiguous abstention schema, including the failure-category
   field. Verify it on separate engineering/development cases. Do not silently
   repair the completed pilot's responses or remove difficult cases.
3. Check whether a separate, task-disjoint development set contains enough
   observable information for the requested labels, ideally with independent
   annotation. Define this procedure before seeing a new comparison's outcomes.
4. Only then decide whether a newly versioned comparison is justified. Freeze
   its inputs, prompts and scoring first; keep this completed study as historical
   evidence. No rerun or live-agent Task 2 was launched by this audit.

## Reproducibility

[Audit folder and CSV guide](../results/task1_internvl_dual_v2_coverage_audit/README.md)
contains row/phase/action coverage, image comparisons, all 2,160 raw decoder
responses and the 40 error records. The
[audit script](../scripts/analysis/task1_coverage_audit.py) reparses original
responses without modifying predictions. Six focused regression tests passed.
All **5,069 files** in the final evidence manifest and **596 distinct bound pilot
images** passed hash verification. The original interrupted attempt and its
authorized retry remain separately preserved. This audit made zero model calls.
