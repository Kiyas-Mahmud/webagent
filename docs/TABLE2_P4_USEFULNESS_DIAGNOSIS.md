# P4 usefulness diagnosis — completed 2026-09-10

The saved evidence shows **working memory delivery, changes in E3 proposals,
and no added task completions**. This audit found no dropped corrective fields
or incorrect cosine ranking in the current path. The main limitations are the
retrieval objective, the examples selected for these live failures, and the
lack of substantive corrective content. Their individual causal effects have
not been isolated.

The completed 120-episode result remains unchanged: E2 and E3 each completed
6/30, with zero improved or worsened memory pairs. This is a post-hoc diagnosis,
not another evaluation or a basis for tuning against those 30 pairs.

## What the 46 queries actually did

Every saved top-three result, score and threshold admission was replayed from
the frozen vectors. Every generated memory context was compared with its
recorded decision. All checks passed. The E2 shadow strategy, diagnosis and
planned action were preserved before adding examples.

| Stage | Observed result |
|---|---|
| Retrieval | 46 queries; 138 candidate slots; all admitted |
| Generation | 42 REPLAN generations received examples; four BACKTRACK attempts used their registered executor path |
| Comparable first E2/E3 generations | 28 matched screenshots, prompts and substantive non-memory contexts |
| Proposal changes | 17/28 four-field proposals differed; some differences were formatting, aliases or unused CLICK values |
| Both proposals resolved | 19 pairs; five changed the actual concrete action |
| Target validity | Three changed from resolved to rejected; six stayed rejected; none changed from rejected to resolved |
| Browser result | No additional completion in any of the 30 pairs |

Comparison normalizes system identifiers and omits the memory-only context
fields/schema and context commitment. It retains the goal, instruction, control
state, action history, values, diagnosis and previous feedback. This descriptive
comparison does not isolate the effect of every prompt token. Later generations
are traced individually because E2/E3 histories can diverge.

The five changed concrete actions were all in `enter-password`: E2 clicked
Submit immediately; E3 clicked **Verify password**, then clicked Submit on its
second attempt. Both clicks executed. Both learned assessments were negative.
E3 never typed either password, so all five episodes still failed. Across all
42 E3 generations it proposed **32 CLICK and 10 SELECT actions, zero TYPE**.
The executor did not silently substitute an action or discard proposed text.

The three additional first-proposal rejections were `click-option` resets 2
and 4, and `click-checkboxes` reset 0. E3 proposed SELECT on controls incompatible
with that action. E2's corresponding Submit clicks executed but also failed the
tasks. These are action-validity regressions without a worsened completion pair.

## Why high memory metrics did not predict useful retrieval

The [embedding route](../src/web_agent/models/model.py) returns the trained
memory task-adapter representation. Its head learns storage/update and recovery
strategy classification. The contrastive loss operates on the diagnosis/outcome
representation, using outcome labels; it is not a learned relevance objective
for corrective browser examples. Good storage accuracy therefore does not
establish semantic nearest-neighbor usefulness.

The existing training-only calibration calls a pair relevant when its failure
and strategy labels agree. **1,903/1,974 pairs are positive** under that proxy.
The selected threshold achieves F1 0.982194; admitting every pair achieves
0.981687. The improvement is only **0.0507 percentage points**. The selected
threshold excludes two training pairs and admits every candidate in this live
run. These are label-agreement statistics, not browser recovery accuracy.

The dominant example ranked first in **39/46 live queries (84.8%)**, specifically
all 39 queries embedding a NAVIGATE action. It records a generic Nike page goal
and NAVIGATE/BACKTRACK recovery, with no URL or reflection. In the training-only
leave-source-task-out replay, this same example ranked first only **2/1,974**
times; training queries selected 1,252 distinct first neighbors. Thus the
evidence does not support a claim that the entire index has collapsed. The
live-query concentration is consistent with poor transfer and the common
rejected NAVIGATE failure context; it does not prove which input feature causes it.

The stored recovery actions include **260 TYPE and 262 SELECT examples**, but
the 138 live retrieval slots contained only NAVIGATE (49), PRESS_KEY (58), CLICK
(24) and SCROLL (7). Stored strategies include no REPLAN examples. This is a
coverage mismatch between what the planner was trying to repair and what it
received; no ablation has established its independent effect on completion.

## Can the existing material supply the missing correction?

**There is no hidden corrective value or reflection to restore in this export.**
All 1,974 stored source-material objects exactly match the corresponding
hash-bound preparation records. Every `recovery_action_value` and
`reflection_text` is empty before the runtime projection. Goals and domains are
present in all records and already reach generation.

The existing PC-01 training action-value evidence reports zero `action_value`
keys across its 24,107 training rows. This audit used that saved evidence and
the prepared training metadata; it did not reopen or review the original
dataset. The preparation/extraction source preserves available values rather
than deliberately deleting them. This does not establish that no other kind
of useful information could ever be extracted from the dataset; it establishes
what the current authorized material actually provides.

We cannot turn labels such as NAVIGATE/BACKTRACK into a verified destination,
or infer a successful password-entry procedure from an empty reflection.
Reformatting the records cannot create those missing facts.

## Decision and bounded next step

No further full evaluation is queued, and no runtime repair is justified by
this audit alone. The frozen model, all 1,974 vectors, source ordering, threshold
and completed results remain unchanged.

One defensible **development-only** candidate is applicability-based abstention
for the existing top three: check whether an example's advice has the necessary
current-observation evidence, and retain the complete E2 decision when none do.
For example, BACKTRACK advice needs an executed in-episode navigation; advice
requiring a missing URL or key cannot supply that argument. This would reuse
existing recorded facts and capability checks, without inventing corrections,
retrieving replacement neighbors, or changing checkpoint weights.

That is a proposed method change, not a demonstrated fix or an automatic route
to a positive P4 result. It may only remove unhelpful context and recover E2
behavior. It must be specified in a separate profile and assessed on development
tasks before considering another evaluation. Do not ban a particular example,
adjust thresholds or choose prompts because of these final outcomes. A claim
of additional memory benefit still requires independently measured completions.

## Evidence and reproducibility

- [All 46 query-to-action traces](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-p4-diagnosis-complete/query-traces.json)
- [All 30 paired first-proposal records](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-p4-diagnosis-complete/paired-first-proposals.json)
- [Machine-readable summary](/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-p4-diagnosis-complete/summary.json)
- [Analysis script](../scripts/analyze_table2_p4.py)
- [Completed Table 2 results](TABLE2_MINIWOB_INTERFACE_V2_RESULTS.md)

The script hash-verifies all 2,353 archived evaluation files, the frozen memory
files, the preparation queue and action-value evidence. It performs zero model
calls, zero browser episodes and zero memory writes. No original dataset rows,
Gold images or locked-test data were opened for analysis. Screenshot files in
the completed MiniWoB archive were read only to verify their hashes.

Reproduce into a **new** directory:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/analyze_table2_p4.py \
  --evaluation /home/aiub/kiyas/table2-evidence/miniwob-interface-v2-evaluation \
  --output /ABS/new-p4-diagnosis
```
