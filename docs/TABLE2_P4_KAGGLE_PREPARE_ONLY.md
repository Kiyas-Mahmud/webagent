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

The two scientific data inputs are fixed to dataset version **1**:

- `kiyasmahmud/web-gold-40k`, version 1;
- `kiyasmahmud/gold-40k-retry`, version 1.

Kaggle kernel metadata does not cryptographically prove the mounted dataset
version. The runner records version 1 as a declaration and authenticates the
actual training JSON bytes against
`configs/eval/table2/p4_source_authority_v1.json`. A version declaration is not
substituted for that content check.

The generated kernel metadata also attaches one private source-transport
dataset whose exact ID is supplied by the operator. That third dataset is not
a scientific dataset or source authority. It carries one Git bundle and one
versioned transport manifest so the otherwise self-contained Kaggle script can
materialize the registered repository while internet access remains disabled.

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

## Generate the two local staging directories

Start from the final, clean repository commit that is intended to execute. The
stager requires explicit Kaggle `owner/slug` IDs and an explicit source-dataset
version of `1`; it rejects placeholders, a dirty or symlinked repository,
nested outputs, and every pre-existing output path:

```bash
PYTHONPATH=src python scripts/stage_table2_p4_kaggle_transport.py \
  --repository-root /absolute/path/to/clean/webagent \
  --source-dataset-id EXACT_OWNER/table2-p4-source-transport-v1 \
  --source-dataset-version 1 \
  --kernel-id EXACT_OWNER/table2-p4-prepare-only-v1 \
  --source-output-dir /absolute/new/path/table2-p4-source-dataset \
  --kernel-output-dir /absolute/new/path/table2-p4-kernel
```

Both complete IDs are mandatory; only the explicitly supplied owner varies,
while the two registered slugs are fixed. This command creates local files
only. It does not authenticate to Kaggle,
create a dataset, push a kernel, or start a run. The generated source-dataset
directory contains `dataset-metadata.json`, a private-upload policy, exactly one
`table2-p4-git-bundle-transport-v1.json`, and exactly one `.bundle`. Kaggle's
dataset metadata format has no privacy field; dataset creation is private by
default, and the generated policy requires the operator to avoid `--public`
and verify privacy on Kaggle before attaching it, consistent with the
[official dataset CLI contract](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md#kaggle-datasets-create).
The generated kernel directory contains only `run.py` and
`kernel-metadata.json`; the latter sets
`is_private: true`, disables GPU, TPU, and internet, and declares the two Gold
inputs plus the explicitly supplied source-transport dataset.

Only the metadata-named kernel `code_file` is sent as script source by the
[official Kaggle CLI implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py#L6551-L6569).
Therefore the bootstrap does not assume that an arbitrary sibling configuration
file will appear at runtime. Its fixed, tracked `run.py` discovers exactly one
fixed-name manifest directly at either supported source-dataset mount root,
reads the declared dataset ID, and requires that the manifest actually resides
at the corresponding `owner/slug` mount.

## Operator-only Kaggle upload, run, and download

The commands in this section are external operator actions. The repository does
not execute them, authenticate to Kaggle, or retain Kaggle credentials. Run
them only from an authenticated operator host and substitute the same exact
owner used during staging:

```bash
TABLE2_P4_SOURCE_STAGE=/absolute/path/to/table2-p4-source-dataset
TABLE2_P4_KERNEL_STAGE=/absolute/path/to/table2-p4-kernel
TABLE2_P4_DOWNLOAD_DIR=/absolute/new/path/to/table2-p4-kaggle-download
TABLE2_P4_SOURCE_DATASET=EXACT_OWNER/table2-p4-source-transport-v1
TABLE2_P4_KERNEL=EXACT_OWNER/table2-p4-prepare-only-v1
```

First query the source dataset ID. It must not already exist: this protocol
requires a newly created source-transport dataset whose first and only version
is version 1. If this command reports an existing dataset, stop and resolve the
identity conflict; do not call `kaggle datasets version` and do not attach an
older dataset under the registered name.

```bash
kaggle datasets status "$TABLE2_P4_SOURCE_DATASET"
```

After confirming that the ID is unused, create it from the generated staging
directory. Deliberately omit `--public`: the
[official dataset CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md#kaggle-datasets-create)
defines a newly created dataset as private unless `--public` is supplied.

```bash
kaggle datasets create -p "$TABLE2_P4_SOURCE_STAGE"
kaggle datasets status "$TABLE2_P4_SOURCE_DATASET"
kaggle datasets files "$TABLE2_P4_SOURCE_DATASET" --page-size 20
```

The status and file-list commands establish successful creation and the remote
file names, but the documented CLI output is not an independent privacy or
version attestation. Before pushing the kernel, the operator must open that
exact dataset in Kaggle and record that it is **Private** and at version **1**.
Stop if either value differs. Never use `--public`, never create a second
dataset version, and do not treat the local `private-upload-policy.json` as
proof of remote state.

Then push the generated private CPU/no-internet kernel. Per the
[official kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#kaggle-kernels-push),
`push` uploads the code and metadata and starts the run. Do not add an
accelerator override. Re-run the status command manually until Kaggle reports
the latest run complete; a failed run is not evidence.

```bash
kaggle kernels push -p "$TABLE2_P4_KERNEL_STAGE"
kaggle kernels status "$TABLE2_P4_KERNEL"
kaggle kernels files "$TABLE2_P4_KERNEL" --page-size 200
```

Download all output pages into a new path. The path must not exist, and `-o` /
`--force` is intentionally forbidden because it could mix or overwrite runs.

```bash
test ! -e "$TABLE2_P4_DOWNLOAD_DIR"
kaggle kernels output "$TABLE2_P4_KERNEL" \
  -p "$TABLE2_P4_DOWNLOAD_DIR" \
  --page-size 200
```

The expected downloaded package is
`$TABLE2_P4_DOWNLOAD_DIR/table2-p4-prepare-only-v1`. From the exact clean Git
checkout whose commit was staged, strictly replay the outer receipt, SHA-256
sidecar, inner package, source-file set, and clean-commit binding:

```bash
test -z "$(git status --porcelain)"
PYTHONPATH=src python3 scripts/validate_table2_p4_kaggle_output.py \
  --repository-root "$PWD" \
  --output-root "$TABLE2_P4_DOWNLOAD_DIR/table2-p4-prepare-only-v1"
```

Only exit status 0 with `status: PASS`, `package_status: REVIEW_REQUIRED`, and
`paper_table_status: N/R` is valid. Preserve the complete downloaded directory;
the receipt is not valid without its sidecar and `preparation/` bytes. This
validation authenticates the registered program output against the matching
clean source checkout. It does not prove Kaggle platform behavior, reviewer
independence, memory eligibility, or any Table 2 result.

Replay also enforces the canonical recorded invocation and ordered UTC
timestamps, exact CPU/no-model environment semantics, the two registered
dataset identities/layouts, and exact agreement among dataset receipts,
attached mounts, train-input descriptors, the inner read ledger, and the
registered/embedded source authority. Environment, timestamp, and declared
Kaggle-version fields remain wrapper attestations rather than independent
platform measurements.

The downloaded package root is an exact allowlist: it must contain only the
receipt, its sidecar, and `preparation/`. Extra files, extra directories
(including empty ones), symlinks, hard-linked receipt/sidecar or inner
preparation files, and any extra or changed inner preparation entry fail before
a successful validation result.

## No-argument Kaggle execution and authority boundary

The generated kernel is deliberately no-argument. On Kaggle, `run.py`:

1. refuses every CLI argument and every existing fixed checkout or output;
2. finds exactly one versioned manifest and exactly one Git bundle under its
   declared source-dataset mount;
3. rejects path escape, symlink, hard-link, non-regular-file, byte-count,
   SHA-256, manifest-schema, dataset-ID, and version mismatch;
4. requires source-dataset version `1`, runs `git bundle verify`, and requires
   the bundle to advertise exactly one full commit as `HEAD`;
5. clones with `git clone --no-local` into the absent fixed checkout, checks out
   that commit at detached `HEAD`, and verifies a clean worktree;
6. verifies that the executing bootstrap bytes match both the manifest and the
   bootstrap tracked by the cloned commit; and
7. invokes the registered prepare-only runner with the exact commit and bundle
   path as transport evidence.

The manifest hash is intentionally not embedded into the Git commit: doing so
would create a commit/hash circularity. The manifest and bundle are
materialized transport, not scientific source authority. A caller can replace
transport bytes only by producing a different bundle/commit identity. The
inner runner then independently requires the supplied full commit to equal a
clean repository `HEAD`, hashes the exact executed-source set before work,
rechecks both source and bundle after work, and records those identities in the
receipt. Later joint-audit/handoff replay must match the expected source commit
and executed-source hashes; it must never promote the transport manifest or
dataset merely because their internal hashes are self-consistent.

For a direct non-Kaggle diagnostic invocation, the equivalent inner interface
uses `--source-bundle` (the old `--source-archive` spelling is a deprecated
alias and still accepts only a verified Git bundle). Direct invocation does not
replace the registered no-argument Kaggle route.

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
set, the required Git-bundle transport hash, argv, a small
dependency/environment inventory, timestamps, stage outcomes, and scientific
non-claims. Receipt schema v3 calls the required descriptor `source_transport`,
identifies its format as `git_bundle`, records the exact bundle `HEAD`, and
sets `scientific_source_authority: false`. A failed in-boundary run retains
only its receipt and checksum.

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
