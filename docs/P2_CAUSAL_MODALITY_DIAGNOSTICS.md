# Pillar 2 Causal/Modality Companion Diagnostics

## Scientific boundary

This package implements the registered Phase J Pillar 2 diagnostics without
changing training or the E0–E3 experiment. It is deliberately offline and may
run only after the source actions are frozen.

Only an exact, authenticated PC-01 run produces
`P2_COMPANION_MECHANISM_EVIDENCE`. A generic predictor or deterministic fixture
is forcibly labelled `ENGINEERING_DIAGNOSTIC_ONLY_UNPROMOTABLE` with promotion
status `UNPROMOTABLE`; changing its label fails validation. Neither status
creates a Table 2 row, modifies a Table 2 metric, triggers recovery, queries or
writes memory, or claims operational recovery. The paper’s Table 2 status
remains `N/R` during the PC-01 pilot.

The two diagnostic families are:

- Inference-time modality sensitivity: compare the full input with a
  registered same-size neutral image and with fixed neutral text/state. This is
  a sensitivity analysis, **not** a controlled training ablation.
- Post-action input controls: remove executed-action identity, replace only the
  post-state with a deterministic same-size donor, and replace only the action
  with a deterministic different donor while retaining the original post-state.
  These test whether predictions are conditioned on the registered causal
  fields; they do **not** by themselves prove causal reasoning.

## Information separation

The predictor has two narrow methods:

1. `predict_pre` receives only the task ID, causal pre-image, and current
   oracle-blind text/state.
2. `predict_post` receives the completed pre/action/post input.

Offline targets are stored on the runner side and are never passed to either
method. The exact pre-action call is replayed after every post-action control;
any changed prediction fails the diagnostic. Oracle/future keys in model-visible
text or page state are rejected. The runner has no environment adapter,
executor, verifier, recovery controller, or memory store.

## Frozen input requirements

The JSON input uses schema `table2.pillar2-diagnostic-input.v1` and must freeze:

- a validation-selected, frozen, evaluation-mode backend identity;
- the exact predictor module and class plus checkpoint, resolved-config payload,
  resolved-config record, processor-contract, predictor-source, and full Git
  commit hashes, with the exact factory entrypoint and factory-source hash;
- source partition and zero locked-test reads;
- one exact pre/post image pair and offline target record per example;
- one neutral image for every observed image dimension;
- the fixed neutral text control `fixed-neutral-text-state-v1`;
- `actions_frozen_before_diagnostics=true`;
- `runtime_outputs_consumed=false` and `affects_primary_table2=false`.

Every image is resolved below the supplied external image root, hash-checked
before inference, and hash-checked again afterward. Absolute image paths are
not exported. Every condition uses the same stage-keyed seed as its full-input
counterpart. Donor selection is deterministic and recorded by example ID and
source-record hash.

Every reported diagnostic rate contains its numerator and denominator. Every
reported diagnostic mean contains its total and eligible-example count. Both
include the estimate, 95% interval bounds, confidence, registered interval
method, inference/cluster/contribution units, number of task clusters,
bootstrap counts and seed, interval status, and an explicit reason when the
interval is unavailable. Intervals use 10,000 deterministic percentile
resamples of whole `task_id` clusters for canonical PC-01 evidence. A metric
with no eligible examples is `NOT_APPLICABLE`; one with fewer than two task
clusters has a point estimate but no CI. Package validation recomputes every
summary from the per-example records.

## Execution

The command’s standard-library bootstrap first requires the exact repository
top level, the manifest’s full 40-character commit, and an empty
`git status --porcelain=v1 --untracked-files=all`. This happens before any
`web_agent` or deployment predictor import and is repeated immediately before
the predictor import and after inference/package validation. A canonical run
cannot bypass this gate.

The source-attested factory must be a top-level, zero-argument callable that
returns an object implementing `Pillar2DiagnosticPredictor`. The factory and
exact predictor-class source files are hash-checked before inference and again
afterward.

For canonical PC-01 evidence, the factory must return the exact
`PC01Pillar2DiagnosticPredictor` class from its registered module/source and
must call `load_pc01_pillar2_diagnostic_predictor`. The identity is pinned to
the registered PC-01 model ID/revision, epoch-6 checkpoint SHA-256, canonical
resolved-config SHA-256, authenticated processor-contract SHA-256, model seed
42, validation-only selection, and frozen evaluation mode. A look-alike class,
subclass, generic adapter, or changed constant remains unpromotable. The
adapter reuses the authenticated checkpoint loader, exact processor streams,
and existing model heads. The deployment wrapper supplies frozen artifact
paths; it must not read the diagnostic manifest, labels, evaluator output, or
image corpus itself.

```bash
PYTHONPATH=src:. python scripts/run_table2_pillar2_diagnostics.py \
  --input-manifest /external/p2/input_manifest.json \
  --image-root /external/p2/images \
  --output-dir artifacts/table2/<campaign_id>/component_test/pillar2 \
  --predictor-factory your_attested_module:create_pillar2_predictor
```

The command refuses a non-empty output directory and emits:

- `pillar2_diagnostic_report.json`, containing per-condition predictions,
  scores, deltas, deterministic donor identities, and temporal replay evidence;
- `package_manifest.json`, binding the report, input, and backend identity by
  SHA-256.

No scientific result is supplied by this implementation. Values appear only
after the frozen corpus is actually run with the authenticated selected
checkpoint.
