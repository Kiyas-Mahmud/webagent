# P4 Memory Preparation and Cross-Host Transfer Runbook

## Purpose and scientific boundary

This workflow prepares Pillar 4 inputs without weakening the registered Table 2
memory policy. It has two functions:

1. Audit the two registered PC-01 training sources (original Gold plus the
   recovery supplement), then emit a deterministic queue for independent
   review.
2. Hash and validate the seven files of an already-built frozen memory store so
   that the store can be moved from the Kaggle build environment to the
   WebArena campaign host without losing its identity.

The preparation queue is **not** a provenance manifest, verification record,
duplicate-audit result, or eligible-memory list. Gold labels such as
`recovery_success=true` and `memory_update_flag=true` are only local candidate
gates. They are never converted into independent recovery evidence or final-task
success evidence by this tool.

The existing `scripts/build_table2_memory.py` command remains the only
production store builder. The separate
`scripts/run_table2_joint_duplicate_audit.py` command now produces and replays
the exact/near duplicate assignments. It never produces independent recovery or
final-task-success evidence. The store builder requires both stages of that
audit and reopens their exact train/task/config/source inputs before it opens a
checkpoint, imports a model backend, checks CUDA, or creates a store.

## Recommended host and data flow

Keep the 22 GB Gold dataset and its images in the Kaggle environment where they
are already mounted. Do not copy that corpus to the local PC or DGX merely to
prepare P4. Transfer the selected epoch-6 checkpoint and pinned runtime source
to Kaggle for construction, then transfer only the compact frozen store and
evidence package to the campaign host.

```text
Kaggle Gold data mount
  -> registered PC-01 source-authority check (file names, row counts, hashes,
     dataset identity and source roles)
  -> train-only preparation package (small; no images are copied)
  -> deterministic joint train-candidate/WebArena duplicate assignments
  -> independent reviewer/verifier using those frozen assignments
  -> finalized 50+15 duplicate-audit manifest
  -> real table2-memory-provenance-v1 manifest
  -> strict frozen-store build in Kaggle with the exact selected checkpoint/config
  -> seven-file frozen store + external transfer manifest
  -> WebArena campaign host, if different
  -> destination validation before E3 is allowed to load the store
```

The Hugging Face base model must be the exact revision already bound by the
selected checkpoint/config. Resolve or cache that revision in the Kaggle build
environment; this P4 tool does not download a base model and does not change
model configuration.

After construction, export only the complete seven-file store, preparation and
provenance manifests, transfer manifest, and other small registered evidence
needed for review. Do not transfer the Gold images. Validate the store on the
DGX or split-deployment campaign host before E3 can load it. The transfer
manifest is a verifier, not a copy utility.

The joint duplicate namespace cannot be frozen against an unresolved browser
task registry. The measured WebArena public 0--49 audit found 47
assistant-answer/`string_match` tasks incompatible with the current six-action
P3 interface and only three compatible URL tasks. Therefore P4 candidate
screening may proceed, but the joint Gold-train/WebArena duplicate audit,
provenance closure, and production store freeze must wait for the user-approved
and preregistered task-interface resolution. No P4 tool may silently replace or
drop browser tasks to make that audit pass.

## 1. Prepare the train-only review queue

For the two registered Kaggle datasets, the tracked CPU-only wrapper in
[`TABLE2_P4_KAGGLE_PREPARE_ONLY.md`](TABLE2_P4_KAGGLE_PREPARE_ONLY.md) resolves
the permitted mount layouts, invokes this section's authenticated preparation
and validation operations, and emits a hash-bound execution receipt. It stops
at `REVIEW_REQUIRED`; it does not perform any operation from Section 2 or 3.
For the registered PC-01 source authority, the wrapper is mandatory: a direct
`prepare_table2_p4.py audit-candidates` output without the wrapper's clean-
commit execution receipt is not accepted by the joint audit or store builder.

The following command documents the underlying operation for local/synthetic
diagnostics only. For registered evidence, run the wrapper command in the
linked prepare-only document instead:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py audit-candidates \
  --gold-train-json /kaggle/input/web-gold-40k/split_train.json \
  --gold-data-root /kaggle/input/web-gold-40k \
  --supplement-train-json /kaggle/input/gold-40k-retry/data/supplement_train.json \
  --supplement-data-root /kaggle/input/gold-40k-retry \
  --dataset-id web-gold-v2.8 \
  --dataset-version pc01-train-0522807d-supplement-67ade5e9 \
  --source-authority configs/eval/table2/p4_source_authority_v1.json \
  --output-dir /kaggle/working/table2/p4-preparation-<dataset-hash>
```

The production authority requires both exact sources. It registers
`split_train.json` as 23,499 rows with SHA-256
`0522807d74256fa303a20d5b8f3bb1653de44aad187660a655580c708c7ae3de`
and `supplement_train.json` as 608 rows with SHA-256
`67ade5e971e8fa2f58ec8a269b1e476ae8cc2349a89eb3e3e0d5e161c47c1098`.
Omitting, swapping, renaming, adding to, or changing either source is rejected.
The dataset ID/version must also equal the authority. Synthetic tests use a
separate explicit fixture authority; the production API never auto-authors one.

The command accepts only files named `split_train.json` and
`supplement_train.json`, requires them to resolve beneath their declared data
roots, rejects symlinked input JSON and registered locked-mount path components,
and refuses any source row that explicitly declares a non-training split. It has
no validation or test input argument. It reads referenced pre/post state
artifacts only for rows that pass the local candidate gates.

The output package contains:

- `candidate_audit.json`: deterministic gates, transition report, and fatal
  errors;
- `read_ledger.json`: input hashes/counts and explicit zero validation, test, and
  locked-test reads;
- `review_queue.jsonl`: candidate material with every external-evidence field
  set to JSON `null`;
- `source_authority.json`: the exact registered source-authority bytes;
- `preparation_manifest.json` and `preparation_manifest.sha256`: package closure.

Rows whose registered source review status is `pending` may enter this review
queue, but they are not thereby approved for P4. A pending row can be admitted
only if the completed provenance contains a separate independent
`table2-memory-p4-label-review-v1` record bound to both its source-record hash
and the complete memory-item source-material hash. Explicitly rejected or
quarantined rows never enter the queue. Source-approved rows still require
independent recovery and final-task-success evidence.

The authenticated supplement uses a direct-transition layout: its pre-state is
the observed failure state, its row action/value is the executed recovery, and
its post-state is the recovery result. Therefore a successful `RETRY` row may
legitimately have `outcome_label=SUCCESS` and `failure_type_4=NONE`. The prior
failed action/type are frozen as explicitly unavailable rather than copied from
the recovery action. A direct `ABORT` cannot satisfy final-task success and is
not admissible. Raw rows cannot opt themselves into this interpretation by
adding a marker; only the source-authority-bound supplement role can do so.

`REVIEW_REQUIRED` means the package is structurally valid and still needs real
external evidence. `FAIL` suppresses the complete queue and must be resolved;
the CLI exits with status 2. It is never permission to admit a partial subset.

Validate after any transfer:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py validate-preparation \
  --package-dir /kaggle/working/table2-p4-prepare-only-v1/preparation
```

The validator rechecks file hashes, exact schemas, deterministic queue IDs,
train-only filenames, all zero-read fields, and that no queue row claims
independent evidence or eligibility. This compact check does not reopen the
22 GB corpus. Before compact evidence is transferred, the joint producer's raw
validation must replay every queued transition from the authenticated training
JSON and recompute its pre-state, action and post-state hashes from source
bytes.

The zero-read ledger is an application-level record of this command's explicit
inputs. It does not replace the locked-mount preflight, filesystem isolation, or
campaign access ledger required by the Table 2 protocol.

## 2. Freeze joint duplicate assignments, then produce independent evidence

After a user-approved compatible 50-task development export exists, freeze the
joint duplicate assignments on Kaggle:

```bash
PYTHONPATH=src python scripts/run_table2_joint_duplicate_audit.py build-assignments \
  --config configs/eval/table2/joint_duplicate_audit_v1.json \
  --source-authority configs/eval/table2/p4_source_authority_v1.json \
  --preparation-package /kaggle/working/table2-p4-prepare-only-v1/preparation \
  --gold-train-json /kaggle/input/web-gold-40k/split_train.json \
  --supplement-train-json /kaggle/input/gold-40k-retry/data/supplement_train.json \
  --resolved-task-export /kaggle/working/table2/evidence/webarena-tasks.json \
  --approved-task-registry /kaggle/working/table2/evidence/approved-pilot-task-registry.json \
  --recovery-scenarios benchmarks/table2/pilot/recovery_scenarios.json \
  --duplicate-audit-registration benchmarks/table2/pilot/duplicate_audit_manifest.json \
  --output-dir /kaggle/working/table2/evidence/joint-duplicate-assignments
```

This stage reopens every row of the exact `split_train.json` and
`supplement_train.json` files to authenticate the preparation hashes, but it
clusters only preparation-queue candidates because only those rows can possibly
enter P4. It places those candidates and all 50 approved WebArena tasks in one
namespace. Exact identity is the SHA-256 of normalized task-goal tokens and site
keys. Near links use deterministic word trigrams plus site keys, Jaccard >= 4/5,
a same-site requirement when both sides have site keys, and deterministic
single-link connected components. Exact and near IDs, every accepted link, the
config hash, complete producer/helper source manifest, source-file hashes, and
zero-read ledger are frozen in the package. The registered 15 recovery
diagnostic rows are copied without alteration and remain
`SYNTHETIC_DIAGNOSTIC_ONLY`.

For the registered source authority, this command also revalidates the outer
prepare-only execution receipt against the exact tracked source bytes. Its
receipt hash, executed-source-set hash, and full source commit become part of
the assignment namespace/input binding. Keep the receipt and sidecar in the
parent directory of `preparation/`; copying only the six inner files is an
incomplete evidence package.

Revalidate the package from all source bytes after transfer:

```bash
PYTHONPATH=src python scripts/run_table2_joint_duplicate_audit.py validate-assignments \
  --config configs/eval/table2/joint_duplicate_audit_v1.json \
  --source-authority configs/eval/table2/p4_source_authority_v1.json \
  --preparation-package /kaggle/working/table2-p4-prepare-only-v1/preparation \
  --gold-train-json /kaggle/input/web-gold-40k/split_train.json \
  --supplement-train-json /kaggle/input/gold-40k-retry/data/supplement_train.json \
  --resolved-task-export /kaggle/working/table2/evidence/webarena-tasks.json \
  --approved-task-registry /kaggle/working/table2/evidence/approved-pilot-task-registry.json \
  --recovery-scenarios benchmarks/table2/pilot/recovery_scenarios.json \
  --duplicate-audit-registration benchmarks/table2/pilot/duplicate_audit_manifest.json \
  --assignment-package /kaggle/working/table2/evidence/joint-duplicate-assignments
```

The command prints `joint_duplicate_assignment_binding`. The independently
authored provenance JSON must copy that object exactly under the top-level key
`joint_duplicate_assignment_binding`, and its per-sample
`exact_duplicate_key`, `near_duplicate_cluster_id`, and namespace ID must match
the frozen candidate assignments.

An independent registered reviewer or verifier must still inspect the actual
causal transition and terminal task evidence. That external process must create
a genuine
`table2-memory-provenance-v1` artifact containing, for every potentially
admitted item:

- independently verified recovery success;
- independently verified final task success;
- exact task, episode, source, transition, and state/action hash bindings;
- exact-duplicate and near-duplicate identities;
- the registered joint duplicate-cluster namespace ID;
- for every pending source row proposed for admission, a full independent P4
  label-review record bound to all memory-consumed source material;
- explicit zero validation, test, and locked-test reads.

Do not fill the queue's `null` fields and rename it as provenance. Do not infer
final task success from the Gold row's recovery label. Do not invent reviewer
IDs, verifier outputs, duplicate-cluster hashes, or signing/trust evidence.

The repository does **not** supply the independent reviewer/verifier. After that
process has produced a completed provenance manifest, validate its attachment
to the exact preparation queue before invoking the GPU builder:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py validate-provenance \
  --package-dir /kaggle/working/table2-p4-prepare-only-v1/preparation \
  --provenance-manifest /kaggle/working/table2/evidence/table2-memory-provenance-v1.json
```

This command is deliberately read-only. It requires complete candidate coverage,
matches the dataset/version/logical-record identity, validates the registered
verification schemas and canonical digests, and compares every admitted
recovery's pre-state/action/post-state hashes with the preparation queue. A
`PASS` is structural closure only: it does not perform review, prove that a
named reviewer is independent, reopen state images, or run the joint duplicate
audit. The earlier raw assignment build/validation is the required source-byte
authentication step.

Finalize only after that validation passes:

```bash
PYTHONPATH=src python scripts/run_table2_joint_duplicate_audit.py finalize-audit \
  --config configs/eval/table2/joint_duplicate_audit_v1.json \
  --source-authority configs/eval/table2/p4_source_authority_v1.json \
  --preparation-package /kaggle/working/table2-p4-prepare-only-v1/preparation \
  --gold-train-json /kaggle/input/web-gold-40k/split_train.json \
  --supplement-train-json /kaggle/input/gold-40k-retry/data/supplement_train.json \
  --resolved-task-export /kaggle/working/table2/evidence/webarena-tasks.json \
  --approved-task-registry /kaggle/working/table2/evidence/approved-pilot-task-registry.json \
  --recovery-scenarios benchmarks/table2/pilot/recovery_scenarios.json \
  --duplicate-audit-registration benchmarks/table2/pilot/duplicate_audit_manifest.json \
  --assignment-package /kaggle/working/table2/evidence/joint-duplicate-assignments \
  --provenance-manifest /kaggle/working/table2/evidence/table2-memory-provenance-v1.json \
  --output /kaggle/working/table2/evidence/joint-duplicate-audit.json
```

The finalizer replays all assignments and provenance attachment, writes 50
`VERIFIED` normal-task rows plus the unchanged 15 diagnostic-only rows, and
creates no recovery or final-success decisions itself. Do not start the store
build merely because `validate-provenance` passes; the registered audit
implementation/configuration, approved compatible task export, exact recomputed
task-interface `PASS`, and finalized 50+15 manifest must also exist.

Run the read-only `validate-final` subcommand with the same arguments,
`--assignment-package`, `--provenance-manifest`, and `--audit` after any copy.
It reproduces the complete final manifest; it does not merely accept internally
consistent cluster IDs.

## 3. Build the frozen store with the existing strict builder

Only after the external provenance, approved task export, exact task-interface
audit, and completed joint duplicate audit exist:

The concrete command below is for the provisional PC-01 epoch-6 pilot only.
It must not be reused to bind a final campaign if validation-only promotion
selects PC-02 or PC-03; that campaign must supply its promoted checkpoint path
and manifest-resolved identity without editing the scientific memory rules.

```bash
PYTHONPATH=src python scripts/build_table2_memory.py \
  --config /kaggle/working/table2/selected/resolved_config.json \
  --checkpoint /kaggle/working/table2/selected/best_e6_outcome-mcc0.624.ckpt \
  --data-root /kaggle/input/web-gold-40k \
  --supplement-root /kaggle/input/gold-40k-retry \
  --provenance-manifest /kaggle/working/table2/evidence/table2-memory-provenance-v1.json \
  --resolved-task-export /kaggle/working/table2/evidence/webarena-tasks.json \
  --webarena-task-source /kaggle/input/<pinned-webarena-source>/libwebarena-0.0.4-py3-none-any.whl \
  --webarena-task-registry /kaggle/working/table2/evidence/approved-pilot-task-registry.json \
  --webarena-site-url-map /kaggle/working/table2/evidence/task-url-map.json \
  --task-interface-audit /kaggle/working/table2/evidence/webarena-task-interface-audit.json \
  --duplicate-audit /kaggle/working/table2/evidence/joint-duplicate-audit.json \
  --joint-assignment-package /kaggle/working/table2/evidence/joint-duplicate-assignments \
  --joint-audit-config configs/eval/table2/joint_duplicate_audit_v1.json \
  --p4-source-authority configs/eval/table2/p4_source_authority_v1.json \
  --p4-preparation-package /kaggle/working/table2-p4-prepare-only-v1/preparation \
  --recovery-scenarios benchmarks/table2/pilot/recovery_scenarios.json \
  --duplicate-audit-registration benchmarks/table2/pilot/duplicate_audit_manifest.json \
  --protocol-config configs/eval/table2/protocol.yaml \
  --model-seed 42 \
  --output-dir /kaggle/working/table2/frozen-memory/seed_42
```

Use the selected checkpoint's registered resolved configuration and exact base
revision. Add `--review-overlay-dir` only when that frozen train-only overlay is
part of the registered dataset identity. Never point any argument at validation,
test, final, or locked data.

The builder deterministically reconstructs the complete eligible-and-deduped
selection from the preparation queue and completed provenance, and requires the
raw Gold loader to produce exactly the same ordered memory items and
verification records. It then binds every item field consumed by E3—including
strategy, executed action/value, reflection text, task/domain, source role,
verification hashes and duplicate clusters—to that closure. An eligible item
cannot be omitted, changed, or selected differently between seeds.

Calibration is allowed to fail. In particular, if only source-approved
supplement rows survive review, useful final-success rows may all be `RETRY`
after `ABORT` exclusion, leaving no negative class for the registered
same-strategy calibration. Do not admit `ABORT`, approve pending labels without
the independent review record, or weaken the two-class calibration rule to make
a store build succeed.

The prerequisite gate independently rebuilds the 50-task export from the
supplied pinned task-source bytes, explicit preregistered task registry, and
credential-free URL map. It then exactly recomputes the task-interface audit and
requires all 50 pilot tasks to be compatible. The duplicate audit must be a
frozen `PILOT_ONLY` artifact with all 50 normal rows in that same order, each
`VERIFIED` with a non-empty, canonical cluster record. It must also declare
explicit integer `validation_rows_read`, `test_rows_read`, and
`locked_test_rows_read` fields equal to zero. Every task-content hash is checked,
and every task row must cite the train-corpus commitment derived from the exact
logical-record hash, provenance-file hash, and fully hash-bound namespace used
by the provenance manifest.

When a separately extracted raw task JSON is used instead of the pinned wheel,
also pass `--authorized-raw-task-source-sha256 <sha256>`; the extracted bytes
must still equal the pinned task-source hash. Supplying a registry path is not
approval and this command does not choose a replacement registry. With the
currently registered public 0--49 tasks, the exact task-interface audit remains
`FAIL` (47 incompatible tasks), so the builder correctly stops before checkpoint
access until the user-approved, preregistered protocol resolution exists.

## 4. Create and validate a cross-host transfer manifest

Write the transfer manifest outside the immutable store directory:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py create-transfer-manifest \
  --store-dir /kaggle/working/table2/frozen-memory/seed_42 \
  --manifest /kaggle/working/table2/transfers/seed_42.transfer.json
```

The command first loads the store through `FrozenMemoryStore.load`, replaying the
existing strict manifest, item, embedding, calibration, and verification checks.
It then records SHA-256 and byte size for exactly:

- `manifest.json`
- `manifest.sha256`
- `embeddings.npy`
- `items.jsonl`
- `verification_evidence.json`
- `calibration_evidence.json`
- `threshold_calibration.json`

Copy that complete directory plus the separate transfer manifest. On the
destination host, before starting E3:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py validate-transfer \
  --store-dir /campaign/table2/frozen-memory/seed_42 \
  --manifest /campaign/table2/transfers/seed_42.transfer.json
```

Any missing, changed, symlinked, or descriptor/schema-tampered registered store
file fails validation. The transfer manifest also binds the seed, checkpoint,
logical training records, dataset artifacts, resolved configuration, protocol,
provenance manifest, item count, zero non-training reads, and read-only runtime
policy.

This command validates the seven-file frozen store only. The campaign transfer
must also carry the compact joint-evidence closure: the complete assignment
package (including `source_manifest.json`), preparation package (including
`source_authority.json`), completed provenance, finalized 50+15 audit, resolved
task export, registered recovery scenarios, and the tracked audit
configuration/producer/helper sources. Gold images need not transfer because
the raw-source replay was required on the data host. Campaign handoff copies and
revalidates this compact closure, matches its provenance and audit hashes to
every store, and reconstructs the exact item set before E3 is allowed to run.

## Pilot versus final Table 2

This preparation can support the registered public WebArena engineering pilot.
Pilot task IDs must enter the joint duplicate namespace as development tasks and
remain excluded from the later final campaign. Pilot results stay labelled
`PILOT_ONLY`; they do not fill final Table 2.

For the provisional pilot, the store is bound to PC-01 epoch 6, seed 42, with
no retraining and no seeds 43--44. PC-02 and PC-03 continue independently and
are compared only by the registered validation-only promotion rule. If a
different candidate wins, the PC-01 store and pilot remain engineering
evidence only; the final campaign requires a newly identity-bound store for
the promoted winner.

P4 construction is not Table 1 evaluation: the locked Web-Gold component test
is never opened here. It is also not Table 3: related-paper results do not
define memory eligibility, calibration, or Table 2 scores. Gold contributes
only verified train-split recovery records to the immutable memory store;
WebArena supplies the browser tasks.

Do not open or use the locked benchmark to prepare, tune, calibrate, review, or
rebuild P4. The final frozen campaign may begin only after checkpoint selection,
the memory store, duplicate namespace, protocol, and all other campaign inputs
are frozen under the registered final procedure.
