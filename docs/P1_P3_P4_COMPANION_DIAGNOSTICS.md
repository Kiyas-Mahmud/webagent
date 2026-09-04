# P1, P3, and P4 Offline Companion Diagnostics

Status: implementation and schema are available; no diagnostic input, run, score,
or paper value is reported here. The paper's Table 2 remains `N/R`.

## Scientific boundary

These packages provide mechanism evidence beside Table 2. They do not add an
E0--E3 row, alter a browser decision, trigger a recovery, write memory, or
replace the paired end-to-end evaluation.

- **P1** measures frozen post-action failure detection/diagnosis and assessment
  of an already executed recovery transition. It cannot establish operational
  recovery benefit because it neither chooses nor executes recovery.
- **P3** measures six-class pre-action prediction, normalized grounding, and
parameter-provider validity/agreement. A rejected policy or provider output
is recorded as exactly one rejected executor request; there is no repair loop
and no browser request is actually submitted by the diagnostic.
- **P4** measures read-only top-3 retrieval, train-only admission, and the
  resulting offline strategy change. It cannot establish memory benefit;
  end-to-end benefit requires the paired E2-versus-E3 Table 2 contrast.

The only eligible evidence partitions are explicitly named train-diagnostic,
validation-only, public-development, or completed-public-pilot sources. Locked
final-test material is not an eligible companion input, and every manifest must
record zero locked-test reads. The completed-campaign label deliberately says
`completed_public_pilot_campaign`; an ambiguous generic campaign label is
rejected.

## Causal inputs

P1 predictor views contain the task-visible text/state, pre/post screenshots,
and executed action. Recovery-assessment views contain the frozen failure state,
post-recovery state, strategy, and executed recovery action types. Targets such
as failure type, recovery need, resolution, and progress remain runner-owned.
PC-01 has one registered recovery-outcome head, so its canonical adapter uses
the same probability for resolution and progress rather than inventing a new
head.

P3 predictor views contain exactly one pre-action screenshot plus observable
task/page state. Target action, bbox, and concrete parameters are structurally
absent. The canonical adapter uses the selected PC-01 action/grounding head and
the registered deterministic-then-frozen-base parameter provider. The provider
class, implementation bytes, mode/version, prompt bytes, and attempt trace are
all identity-bound.

The trained PC-01 action head is a typed tensor interface rather than E0's text
parser. A processor, model, or infrastructure exception aborts canonical P3
diagnostics; it is never converted into a policy-rejection observation. The
policy-rejection schema remains available for backends with an explicitly
registered parser contract, while the PC-01 provider's terminal rejection is
accounted as one rejected request.

P4 retriever views contain a frozen, normalized 768-dimensional post-failure
memory-adapter embedding; task/episode/duplicate-cluster exclusions; and the E2
shadow strategy. Relevance labels and target strategy remain runner-owned.
Every returned item must be train-only, successful, verified, provenance-valid,
not same-task/episode, and outside the query's duplicate clusters. Store bytes
must be read-only and identical before and after the run.

All page-state maps reject oracle, verifier, reward, future-state, ground-truth,
and reference-answer keys. Screenshots are regular, single-link, read-only files
whose bytes match their registered hashes.

## Promotion and source attestation

**Current promotion status:**
`BLOCKED_AUTHORITATIVE_INPUT_PROVENANCE_REQUIRED`. Every package produced by
this source version is engineering-only and unpromotable. Exact backend hashes
prove which PC-01 implementation is available, but the current input schema's
self-declared partition, zero-read count, and opaque source-record hashes do
not prove where examples or labels came from. For P4, naming and hashing the
registered embedding method also does not prove that it produced a supplied
768-dimensional vector.

Canonical companion evidence therefore remains unavailable until a new
registered authority replays a frozen source manifest, access ledger, and
source-record bytes. P4 additionally requires either in-harness execution of
the exact checkpoint-backed `memory_embedding` method over hash-bound causal
inputs or an independently authenticated embedding-production receipt binding
the input, processed batch, output vector, checkpoint/config/processor, source,
partition, and access ledger. A self-authored receipt or hash is insufficient.

Generic fixtures, third-party predictors, and the currently exact PC-01
adapters all produce `ENGINEERING_DIAGNOSTIC_ONLY_UNPROMOTABLE`. After the
missing provenance authority is implemented, a future schema may use a
canonical P1, P3, or P4 mechanism-evidence role only when all of the following
match:

1. PC-01 epoch 6, seed 42 checkpoint, resolved configuration, processor, model
   revision, and validation-only selection constants;
2. the exact class/module and current source bytes in
   `pc01_companion_diagnostics.py`;
3. a source-attested zero-argument deployment factory; and, for P3, the exact
   hybrid provider and registered prompt; or, for P4, the exact frozen-memory
   manifest plus the exact PC-01 memory-embedding method/module/source;
4. the exact full Git commit with a clean working tree before any project/model
   import, again immediately before factory import, and again after inference;
5. unchanged predictor source, image/store hashes, and typed input views after
   inference.

Changing only a report label cannot promote generic evidence. Canonical runs
must be launched from a committed clean checkout; the implementation does not
commit files itself.

## Reported metrics

Every rate contains its numerator, denominator, estimate, deterministic 95%
task-cluster bootstrap interval, confidence, method, cluster count, status, and
reason. Every mean contains the analogous total and count. Empty or single-task
subgroups explicitly state why an interval is not estimable.

P1 reports failure accuracy/precision/recall and Brier score, failure-type
accuracy, recovery-need accuracy/recall and Brier score, proposed-strategy
hit@1, and executed-recovery resolution/progress assessment. Companion breakdowns
are by failure type, executed recovery strategy, and oracle-independent trigger
source.

P3 reports resolved, policy-rejection, and parameter-rejection rates; six-class
action accuracy both including rejections and conditional on a prediction;
per-class recall including rejections; bbox IoU and IoU@0.5 over every grounded
target (a missing/rejected box is scored as zero); parameter-resolution and exact-match
rates; provider attempts; and rejected-request accounting. Breakdowns are by
all six action classes and target-size bin.

P4 reports retrieval coverage, admission/abstention and admitted relevance,
strategy hit@1/hit@3, MRR, irrelevant-memory fraction, admitted-intervention
coverage, changed-strategy rate, shadow/final/changed-strategy agreement, and
no-write verification. These are retrieval/mechanism diagnostics,
not task-success metrics.

## Running after inputs and factories are frozen

Each command requires a manifest-registered, zero-argument deployment factory.
P1/P3 `--evidence-root` points to frozen screenshots; P4 points to the frozen
memory-store directory. Use ignored/external output directories so generating a
package does not dirty the attested repository.

```bash
python -m scripts.run_table2_pillar1_diagnostics \
  --input-manifest <frozen-p1-input.json> \
  --evidence-root <frozen-image-root> \
  --predictor-factory <module:create_p1_predictor> \
  --output-dir <ignored-p1-output>

python -m scripts.run_table2_pillar3_diagnostics \
  --input-manifest <frozen-p3-input.json> \
  --evidence-root <frozen-image-root> \
  --predictor-factory <module:create_p3_predictor> \
  --output-dir <ignored-p3-output>

python -m scripts.run_table2_pillar4_diagnostics \
  --input-manifest <frozen-p4-input.json> \
  --evidence-root <frozen-memory-store> \
  --predictor-factory <module:create_p4_retriever> \
  --output-dir <ignored-p4-output>
```

Each output directory contains the pillar-specific input JSON, report JSON, and
`package_manifest.json`. Package files are read-only and mutually hash-bound.
No example inputs or result files are included in Git because no authenticated
diagnostic evidence has been supplied or executed yet.
