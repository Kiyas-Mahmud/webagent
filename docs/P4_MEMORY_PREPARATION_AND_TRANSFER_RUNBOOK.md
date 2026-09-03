# P4 Memory Preparation and Cross-Host Transfer Runbook

## Purpose and scientific boundary

This workflow prepares Pillar 4 inputs without weakening the registered Table 2
memory policy. It has two functions:

1. Audit the explicit Gold training file and optional recovery-supplement
   training file, then emit a deterministic queue for independent review.
2. Hash and validate the seven files of an already-built frozen memory store so
   that the store can be moved from the Kaggle build environment to the
   WebArena campaign host without losing its identity.

The preparation queue is **not** a provenance manifest, verification record,
duplicate-audit result, or eligible-memory list. Gold labels such as
`recovery_success=true` and `memory_update_flag=true` are only local candidate
gates. They are never converted into independent recovery evidence or final-task
success evidence by this tool.

The existing `scripts/build_table2_memory.py` command remains the only
production store builder. Its existing `ProvenanceManifest`, verification, joint
duplicate-audit, and eligibility checks remain authoritative.

## Recommended host and data flow

Keep the 22 GB Gold dataset and its images in the Kaggle environment where they
are already mounted. Do not copy that corpus to the local PC or DGX merely to
prepare P4. Transfer the selected epoch-6 checkpoint and pinned runtime source
to Kaggle for construction, then transfer only the compact frozen store and
evidence package to the campaign host.

```text
Kaggle Gold data mount
  -> train-only preparation package (small; no images are copied)
  -> independent reviewer/verifier + joint train/WebArena duplicate audit
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

Run on the host with the extracted training data and referenced state images:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py audit-candidates \
  --gold-train-json /kaggle/input/<registered-web-gold-dataset>/split_train.json \
  --gold-data-root /kaggle/input/<registered-web-gold-dataset> \
  --supplement-train-json /kaggle/input/<registered-recovery-supplement>/data/supplement_train.json \
  --supplement-data-root /kaggle/input/<registered-recovery-supplement> \
  --dataset-id web-gold-v2.8 \
  --dataset-version <frozen-dataset-version> \
  --output-dir /kaggle/working/table2/p4-preparation-<dataset-hash>
```

Omit both supplement arguments when no supplement is registered. Supplying only
one supplement argument is rejected.

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
- `preparation_manifest.json` and `preparation_manifest.sha256`: package closure.

`REVIEW_REQUIRED` means the package is structurally valid and still needs real
external evidence. `FAIL` suppresses the complete queue and must be resolved;
the CLI exits with status 2. It is never permission to admit a partial subset.

Validate after any transfer:

```bash
PYTHONPATH=src python scripts/prepare_table2_p4.py validate-preparation \
  --package-dir /kaggle/working/table2/p4-preparation-<dataset-hash>
```

The validator rechecks file hashes, exact schemas, deterministic queue IDs,
train-only filenames, all zero-read fields, and that no queue row claims
independent evidence or eligibility.

The zero-read ledger is an application-level record of this command's explicit
inputs. It does not replace the locked-mount preflight, filesystem isolation, or
campaign access ledger required by the Table 2 protocol.

## 2. Produce evidence outside this tool

An independent registered reviewer or verifier must inspect the actual causal
transition and terminal task evidence. A registered joint duplicate audit must
compare the training candidates against the relevant frozen WebArena task
namespace. That external process must create a genuine
`table2-memory-provenance-v1` artifact containing, for every potentially
admitted item:

- independently verified recovery success;
- independently verified final task success;
- exact task, episode, source, transition, and state/action hash bindings;
- exact-duplicate and near-duplicate identities;
- the registered joint duplicate-cluster namespace ID;
- explicit zero validation, test, and locked-test reads.

Do not fill the queue's `null` fields and rename it as provenance. Do not infer
final task success from the Gold row's recovery label. Do not invent reviewer
IDs, verifier outputs, duplicate-cluster hashes, or signing/trust evidence.

## 3. Build the frozen store with the existing strict builder

Only after the external provenance artifact exists:

```bash
PYTHONPATH=src python scripts/build_table2_memory.py \
  --config /kaggle/working/table2/selected/resolved_config.json \
  --checkpoint /kaggle/working/table2/selected/best_e6_outcome-mcc0.624.ckpt \
  --data-root /kaggle/input/<registered-web-gold-dataset> \
  --supplement-root /kaggle/input/<registered-recovery-supplement> \
  --provenance-manifest /kaggle/working/table2/evidence/table2-memory-provenance-v1.json \
  --protocol-config configs/eval/table2/protocol.yaml \
  --model-seed 42 \
  --output-dir /kaggle/working/table2/frozen-memory/seed_42
```

Use the selected checkpoint's registered resolved configuration and exact base
revision. Add `--review-overlay-dir` only when that frozen train-only overlay is
part of the registered dataset identity. Never point any argument at validation,
test, final, or locked data.

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
