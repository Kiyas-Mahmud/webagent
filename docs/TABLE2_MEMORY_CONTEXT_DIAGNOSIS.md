# Memory integration diagnosis and development experiment — 2026-09-10

The completed 24-episode replication did not test an effective memory
intervention: all seven E3 queries abstained. E3 and E2 each scored 2/6.
That result remains unchanged. It does not establish that useful corrective
memory cannot help, and it does not establish that the current memory helps.

## Findings from saved evidence

1. **Only a strategy label reached the old decision path.** The adapter retrieved
   neighbors and attempted to substitute the first admitted neighbor's strategy.
   It did not pass that experience's failed/recovery action types to the planner.
   Every query ranked the same training example first:
   `p4-review-f8a13b214b278923c63eec55`, a BACKTRACK example from
   `gold_v16_40k_002128`. All seven were excluded because the live history had no
   executed, state-changing navigation to backtrack from. This exclusion protects
   action validity; removing it would not make the advice useful.

2. **The reported training metric answers a different question.** Epoch-6 memory
   accuracy is approximately 84.3%. `trainer.py` compares the sign of
   `memory_flag` with `label_memory`: whether to update/store a memory. The live
   reader instead uses cosine neighbors of the pre-head 768-dimensional memory
   representation. Good store/update classification does not establish useful
   nearest-neighbor ranking or successful browser recovery. The trained
   `memory_recovery` classifier also exists; it is not the stored corrective
   experience used by this reader.

3. **The calibration F1 is a permissive label-agreement proxy.** There are
   1,903 positives among 1,974 training calibration pairs. The selected threshold
   gives F1 0.982194; admitting every pair already gives 0.981687. The difference
   is about 0.05 percentage points. This is not a 98.2% browser recovery rate.
   The threshold has not been changed using evaluation outcomes.

4. **Prepared experiences contain action types, but no action values or
   reflections.** All 1,974 have empty `recovery_action_value` and
   `reflection_text`. The export's existing training action-value evidence also
   records omission of action-value text. There is no basis for fabricating a
   URL, target label, or text to type. This is a limitation of the available
   material, not a request to repeat dataset review or regenerate embeddings.

## Development change

An explicit `train-experience-context-v1` mode reuses the same frozen store,
embedding route, threshold and top-three retrieval. It supplies all
threshold-admitted examples' recorded failure/action/recovery fields to the
existing frozen recovery planner as historical training-label context. It does
not force their strategies. The shadow strategy, diagnosis, action validity
checks, decoding and execution budgets remain in place. Missing source values
remain null; no success claim or source image is supplied to generation.

The material file must match the SHA-256 recorded when embeddings were built.
Each retrieved ID must match its source task, failure type and strategy.
The old strategy-substitution mode remains available for reproducibility.
These modes cannot be enabled together.

`changed_recovery_context` and `context_candidate_ids` explicitly record context
exposure separately from changed strategies/parameters. Such exposure is not
itself an action change or a successful recovery; actual generation receipts
must demonstrate that the material reached the planner.

This is a revised memory-input experiment, not proof that every prior failure
was an implementation bug. It cannot promise positive E3-minus-E2 results.

## Validation and live outcome

The 114 targeted tests passed, covering source binding, train-only material,
missing values, below-threshold abstention, context versus strategy changes,
causal inputs, contracts, episode handling and existing memory/action interfaces.
Synthetic checks are separate from live task outcomes. The first launch stopped
before any episode because 86 source records explicitly mark their failed
action `UNAVAILABLE`. The reader was corrected to retain that missing action as
null. All 1,974 existing material/index bindings then passed a CPU preflight.
The initial log, source snapshot and plan were preserved; a corrected launch was
frozen before starting episodes.

A single matched development run completed **16/16 episodes, zero runtime
errors, zero episode reruns**: the four existing development families
(`click-button`, `enter-text`, `click-test`, `click-button-sequence`), all E0–E3,
with the same development reset namespace and budgets. Its corrected plan and
163 producing source files were frozen before episode execution and verified
unchanged afterward. The six final replication families were not rerun.

| System | Successes | Executed normal actions | Executed recoveries | Memory context interventions |
|---|---:|---:|---:|---:|
| E0 | 1/4 | 1 | 0 | 0 |
| E1 | 0/4 | 0 | 0 | 0 |
| E2 | 2/4 | 0 | 5 | 0 |
| E3 | 2/4 | 0 | 3 | 5 |

The evidence audit passed matched resets, identical initial E1–E3 outputs,
event chains, retrieval replay, source identity and outcome reconciliation.
The generation-context audit passed all ten E2/E3 calls. In particular, all five
E3 calls contained the exact admitted training examples; E2 contained none.

**The memory-delivery change works, but additional completion benefit remains
unproven.** E2 and E3 both completed `click-button` and `click-test`. Both clicked
Submit without entering text on `enter-text`. E2 clicked ONE twice on
`click-button-sequence`; E3 emitted two comma-separated JSON action objects,
ONE then TWO, on each attempt. These violate the unchanged single-action output
contract and were rejected before execution. E3 changed its generated output
but did not achieve additional success. The two format rejections are model
output failures, not runtime crashes. They were not manually repaired.

The observed development E2-minus-E1 difference is +50 percentage points;
E3-minus-E2 is zero. Four development families cannot support a broad claim.
No positive memory result is claimed, and no new final evaluation is queued.

The remaining work concerns useful retrieval and compliant action generation.
The current sparse, generic source examples and repeated nearest-neighbor hub
are plausible limitations; this run does not isolate their individual causal
effects. The single-action instruction is already explicit in the planner
prompt, so silently extracting a preferred action from invalid output would
change the experiment. Any further input, retrieval or output-interface change
must be evaluated as a disclosed development change before a new frozen run.

This work used the existing checkpoint, base snapshot and embeddings with no
retraining, package changes or memory regeneration. All **538 prior replication
artifacts** remained hash-identical. The development model/browser workers
exited after the run.

Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-memory-context-v1/`.
`root-cause-diagnostic.json` records the seven historical queries and aggregate
material/calibration findings. No original dataset rows, Gold images or
locked-test data were opened for this diagnosis.
