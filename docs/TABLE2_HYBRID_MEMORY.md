# Hybrid Task 4: frozen advisory memory

2026-09-12. **Task 4 complete as an engineering milestone.** Next: Task 5,
H-runner integration and combined engineering checks. No live H evaluation has
run; final evaluation remains pending.

## Implemented

[build_hybrid_memory](../src/web_agent/memory/hybrid_runtime.py) returns no memory
adapter for H0/H1/H2, before touching model dependencies or files. For H3 it
constructs the existing label-backed store and source-context material with
selective applicability enabled. Manifest/material hashes, checkpoint identity,
1,974 × 768 shape and the fixed threshold are checked. Retrieval remains the
existing cosine top three; vectors, ordering, labels and calibration are unchanged.

The H3 wrapper persists the complete no-memory H2 shadow **before** the embedding
dispatch. It validates episode, incident, task, transition, processor and checkpoint
bindings and passes detached records through a mutation guard. It permits advisory
experience context only; strategy, planned action and parameters must remain those
of the shadow. Empty admission returns the original complete shadow object.

The existing adapter charges one query-embedding dispatch; the wrapper adds no
model call. H3 receipts separately record the shadow, candidates/scores,
applicability exclusions, admitted context and repeated-candidate counts.
They do not claim generation exposure or completion benefit. The existing planner
records actual generation context; the runner records execution and assessment.

The episode recovery branch now checks its memory switch instead of an E3 enum
literal. Existing E0–E3 switches give identical behavior. H identities/configuration
and complete runner wiring remain Task 5; H1/H2 cannot receive memory through
the new factory.

## Verification

**93 focused tests passed**, including 19 new memory tests. They cover non-memory
system isolation, complete empty-admission fallback, preserved action values,
causal bindings, immutable shadow input, rejection of strategy overrides, dispatch
limits, repeated-candidate logging and the existing recovery/continuation paths.

A scripted planner test verified that the admitted example reached generation
context while allowing the planner to choose TYPE despite retrieved CLICK advice.
This demonstrates interface behavior, not beneficial memory use.

The [saved-vector replay](../scripts/check_hybrid_memory.py) used existing training
vectors at indices 0, 987 and 1973 through the actual H3 factory/store/material.
All candidate IDs, scores and threshold decisions matched an independent stable
cosine top-three calculation. The three probes admitted 0, 1 and 2 context examples;
the empty result preserved the original shadow. All eight bound source files
had identical before/after hashes, and the vector mapping was read-only.

All **1,974 corrective values and reflections remain null**. No content was
invented. The final replay made three scripted embedding dispatches using saved
vectors, **zero actual model inferences, zero new embeddings, zero memory writes
and zero live evaluation episodes**.

Evidence:
`/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-task4-engineering/`

- `pytest-final.txt`: 93-test result; `adapter-tests-final.txt` and
  `adapter-fixtures-final/`: 19-case detailed adapter evidence.
- `frozen-replay-final/`: actual-store replay, shadow/result receipts, fixed
  identities, null-field checks and before/after file hashes.
- `sources/`, `completion.json`, `preservation.json`, `manifest.json`: producing
  sources, preservation and evidence bindings.

Earlier debugging artifacts remain separate. The initial planner fixture passed
the embedding request's immutable mapping directly to JSON generation; the
corrected fixture uses the detached policy observation, matching the runner's
planner boundary. The existing planner did not need modification.

Task 5 must wire the H3 factory and normal-action context into the thin H runner,
freeze all switches and sources, and verify the complete system before the
planned 24-episode development check. These engineering checks establish neither
model-chosen multi-step behavior nor a memory completion gain.
