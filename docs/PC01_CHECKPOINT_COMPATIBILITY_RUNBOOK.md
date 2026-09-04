# PC-01 Checkpoint-Backed Compatibility Gate

## Purpose and boundary

This gate converts the authenticated processor-only PC-01 v3 export into a
measured, checkpoint-backed DGX receipt. It is a mandatory runtime-readiness
prerequisite supplied to Table 2 handoff as
`pc01_checkpoint_compatibility_receipt`; it is separate from the seven evidence
files produced by the v3 export and is not an eighth producer-evidence role.

A `PASS` means that the exact epoch-6, seed-42 checkpoint can be reconstructed
on the registered DGX GB10 runtime and that its production callbacks reproduce
direct checkpoint inference on a deterministic causal fixture. It does not:

- approve or freeze the WebArena task registry;
- execute any controlled-recovery or WebArena episode;
- construct or approve the P4 memory store;
- promote PC-01 ahead of PC-02 or PC-03; or
- change the paper's Table 2 status from `N/R`.

The registered input is the v3 export whose manifest SHA-256 is
`63c01942cc653732c9e9e18639cb49bded09a82235fd4ea37fef3fad14c9fa2d`.
The command has no option for substituting another export-manifest identity.

## Required DGX inputs

Run from the exact clean Git commit that contains this command. Keep model
artifacts outside Git.

- `pc01-epoch6-seed42-full-v3/`: the exact 11-file v3 export;
- `best_e6_outcome-mcc0.624.ckpt`: SHA-256
  `9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a`,
  size 291,071,781 bytes;
- the complete local Qwen snapshot at revision
  `895c3a49bc3fa70a340399125c650a463535e71c`, directory-payload SHA-256
  `e002f8290faa3e9f44bf3099eac85a2445e17de738c5bb0cc10d342da837c46c`;
- the PC-01 full `report.json` and `run_contract.json`; and
- the source checkout itself, clean at the explicitly supplied commit.

The environment must report Python `3.12.3`, PyTorch `2.13.0+cu130`,
Transformers `4.57.6`, and CUDA device 0 as `NVIDIA GB10` with
130,662,936,576 bytes. This intentionally prevents the local 2 GB MX450 from
creating DGX evidence.

## Run command on the DGX

First record the committed source identity:

```bash
git -C /absolute/path/to/webagent status --short
git -C /absolute/path/to/webagent rev-parse HEAD
```

The status output must be empty. Then use the returned commit in this command:

```bash
cd /absolute/path/to/webagent
PYTHONPATH=src python scripts/run_pc01_checkpoint_compatibility.py \
  --export-dir /absolute/path/to/pc01-epoch6-seed42-full-v3 \
  --checkpoint /absolute/path/to/best_e6_outcome-mcc0.624.ckpt \
  --base-snapshot /absolute/path/to/Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c \
  --report /absolute/path/to/seed_42/full/report.json \
  --run-contract /absolute/path/to/seed_42/run_contract.json \
  --repository-root /absolute/path/to/webagent \
  --expected-source-commit COMMITTED_40_CHARACTER_SHA \
  --output /absolute/path/to/table2-evidence/pc01_checkpoint_compatibility.json
```

The output path must not exist. The command never overwrites a receipt. It
forces Hugging Face offline mode around all processor/model construction and
does not open Gold data, validation data, locked test data, or WebArena tasks.

## What the gate measures

Before loading a model, the command validates every v3 export file and
cross-link, all 14 base-snapshot files, the checkpoint-saved configuration,
the six-action processor-parity receipt, the train-only action-value omission
evidence, report/run-contract selection facts, and the clean runtime source.

On CUDA it then:

1. loads the selected `WebAgentModel`, restores all seven checkpoint state
   roles, disables gradients, and enters evaluation mode;
2. separately loads the unadapted E0 Hugging Face base and proves that no LoRA,
   task adapter, or custom task head is present;
3. hashes the complete selected in-memory state before inference;
4. generates three deterministic 224×224 read-only PPM states containing no
   benchmark or oracle content;
5. compares direct checkpoint inference with the production runtime callback,
   plus an exact repeat, for pre-action policy/grounding, post-action diagnosis,
   and executed-recovery assessment;
6. verifies that the exact P4 memory-task-adapter tensor is finite, nonzero,
   float32, shaped `[1, 768]`, deterministic, and identical through the runtime
   receipt; and
7. re-hashes the complete selected state and requires byte identity with the
   pre-inference state.

Any mismatch raises an error and produces no `PASS` receipt. The successful
receipt is canonical JSON, read-only, hash-bound to the registered v3 export
and source commit, and explicitly records
`PROVISIONAL_ENGINEERING_PILOT_COMPATIBILITY_ONLY` with
`paper_table_status: N/R`.

## Downstream use

Transfer the exact read-only receipt to the control plane and set the handoff
input field `pc01_checkpoint_compatibility_receipt` to that file. Do not rewrite
or reserialize it. Handoff copies the canonical bytes to
`<handoff>/runtime_readiness/pc01_checkpoint_compatibility_receipt.json`, records
its SHA-256 and complete binding in the runner runtime identity, and passes that
same path to campaign freeze. Freeze stores it below
`<campaign>/frozen/runtime_readiness/`; the campaign-root
`<campaign>/runtime_readiness/` namespace is separately reserved for exactly
the portable live matched-block probe evidence, receipt, and sidecar.

Handoff, freeze, campaign validation, and production-runner construction all
reopen the receipt with
`validate_pc01_checkpoint_compatibility_receipt`. They require its clean
`source_attestation.git_commit` to equal the handoff/campaign source commit,
recompute all 11 source-file rows from the attested source closure, and bind
every receipt artifact digest to the staged executable model, seven-file v3
evidence bundle, and the selected full report/run contract. Missing, writable,
noncanonical, tampered, or rebound evidence fails before browser reset.

Do not place the receipt inside or relabel it as one of the seven v3
producer-evidence files. It is mandatory only for the PC-01 `PILOT_ONLY`
evaluation path. An explicit `ENGINEERING_SMOKE_ONLY` campaign omits it and may
not claim it; the provisional receipt cannot authorize a locked-final campaign.
A valid receipt clears only the checkpoint/CUDA compatibility prerequisite;
live WebArena deployment, approved task resolution, P4 store, and the separate
live matched first-normal-block E0–E3 readiness gate remain required before
pilot execution. That live gate has not run. See
`docs/TABLE2_LIVE_MATCHED_READINESS_RUNBOOK.md` and Section 16 of
`docs/TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md` for the complete
handoff, freeze, readiness, and execution sequence.
