# Table 2 Kaggle P4 `P4_PREPARE_ONLY` Package

## Boundary

This CPU-only package performs exactly two existing operations, in order:

1. authenticated `audit-candidates` against the registered PC-01 Gold source
   authority;
2. `validate-preparation` against the package just produced.

The successful terminal state is `REVIEW_REQUIRED`, not an eligible-memory
claim. The package cannot run the joint Gold/WebArena duplicate audit, author
or validate external provenance, load a checkpoint, create embeddings, build a
memory store, or calculate Table 2. The paper table therefore remains `N/R`.

The two declared Kaggle inputs are fixed to dataset version **1**:

- `kiyasmahmud/web-gold-40k`, version 1;
- `kiyasmahmud/gold-40k-retry`, version 1.

Kaggle kernel metadata does not cryptographically prove the mounted dataset
version. The runner records version 1 as a declaration and authenticates the
actual training JSON bytes against
`configs/eval/table2/p4_source_authority_v1.json`. A version declaration is not
substituted for that content check.

## Accepted read-only mount layouts

Exactly one mount layout per dataset must exist:

```text
/kaggle/input/web-gold-40k
/kaggle/input/datasets/kiyasmahmud/web-gold-40k

/kaggle/input/gold-40k-retry
/kaggle/input/datasets/kiyasmahmud/gold-40k-retry
```

The original Gold training file may be directly below its mount or below the
known `final_data_set_40k/` directory. The supplement may be direct or below
`web_gold_40k_retry_abort_supplement_v2_kaggle/`. Zero matches, multiple
matches, an unregistered root, or a symlink component fails closed. Directory
discovery may inspect names, but only `split_train.json`,
`supplement_train.json`, and state artifacts referenced by locally gated train
candidates are opened by the preparation operation. Validation and test JSON
files have no CLI argument and are never opened.

## Running from an attested source checkout

Do not publish or run this template from a moving branch or an extracted tree
with unknown Git state. Stage a clean Git checkout at the exact full commit in
Kaggle, attach the two datasets, then run:

```bash
PYTHONPATH=src python scripts/run_table2_p4_kaggle_prepare_only.py \
  --repository-root /kaggle/working/webagent \
  --config configs/eval/table2/kaggle_p4_prepare_only_v1.json \
  --input-root /kaggle/input \
  --output-root /kaggle/working/table2-p4-prepare-only-v1 \
  --source-commit FULL_40_CHARACTER_COMMIT \
  --source-archive /kaggle/input/ATTESTED_SOURCE_MOUNT/table2-source.tar
```

`--source-commit` is mandatory for a successful run. It must be a full
lowercase 40-character Git SHA, must equal `HEAD`, and the checkout must be
clean, including untracked files. The runner hashes the exact local source-file
set used by the prepare-only process before work begins and revalidates that
same set after preparation. `--source-archive` is optional; if supplied, it is
hashed only as transport evidence. The archive is never executed, extracted,
or treated as source authority.

For a Kaggle script, stage
`kaggle/table2_p4_prepare_only/run.py` together with the exact repository
source and start from
`kaggle/table2_p4_prepare_only/kernel-metadata.template.json`. Replace only the
metadata `id`. The template disables GPU and internet and declares exactly the
two datasets. It is a local staging template; this repository does not publish
or launch a Kaggle kernel and needs no Kaggle credentials.

## Output and receipt

The runner refuses to overwrite an existing output path. A successful output
contains:

```text
table2-p4-prepare-only-v1/
├── execution_receipt.json
├── execution_receipt.sha256
└── preparation/
    ├── candidate_audit.json
    ├── preparation_manifest.json
    ├── preparation_manifest.sha256
    ├── read_ledger.json
    ├── review_queue.jsonl
    └── source_authority.json
```

No image, checkpoint, model, provenance manifest, embedding, or memory-store
file is allowed. The preparation directory has a 512 MiB fail-closed size
ceiling. The receipt records the exact dataset declarations and resolved roots,
input and output hashes, the required clean commit and exact executed-source
set, any optional transport-archive hash, argv, a small
dependency/environment inventory, timestamps, stage outcomes, and scientific
non-claims. A failed in-boundary run retains only its receipt and checksum.

Keep `execution_receipt.json` and `execution_receipt.sha256` beside the
`preparation/` directory during every later transfer. The registered joint
duplicate assignment validates this outer receipt and binds its SHA-256, the
executed-source-set SHA-256, and the full source commit. Those identities are
then carried into the frozen memory manifest and campaign package.

All zero validation/test/locked-test read counts in this package are explicitly
labelled `APPLICATION_LEVEL_EXPLICIT_INPUTS_ONLY_NOT_OS_OR_PLATFORM_AUDIT`.
They describe this program's input surface; they do not claim to audit Kaggle,
the operating system, or unrelated processes.

After `REVIEW_REQUIRED`, stop. A future user-approved 50-task WebArena
registry, joint duplicate audit, genuine independent evidence, provenance
closure, and GPU-backed frozen-store construction are later separately
authorized stages in
[the P4 memory runbook](P4_MEMORY_PREPARATION_AND_TRANSFER_RUNBOOK.md).
