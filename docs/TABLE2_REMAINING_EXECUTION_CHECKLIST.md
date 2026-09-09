# Table 2 Remaining Execution Checklist

Last reviewed: 2026-09-08

Execution source remains pinned to commit
`178c24ac898f1feb17848ed1fa40fa22b97d053f`. This checklist is a progress record;
documentation updates do not replace the producing commit or invalidate its
validated preparation package. Use a clean checkout of the pinned commit for
commands that require a clean execution repository.

## Purpose

This is the operational checklist for completing the research-locked Table 2
engineering pilot. It does not authorize PC-01 retraining, access to the locked
component test, final-paper claims, or changes to the registered scientific
roles of P1--P4.

Status values are `DONE`, `IN_PROGRESS`, `WAITING_FOR_AUTHORIZATION`,
`WAITING_FOR_EXTERNAL_RUNTIME`, and `NOT_STARTED`.

## Fixed scientific interpretation

- E0 versus E1 is contextual evidence for the trained multimodal pre-action
  policy and grounding stack.
- E1 versus E2 is the paired contribution of P1 diagnosis plus concrete
  executed recovery and post-recovery assessment.
- E2 versus E3 is the paired contribution of P4 frozen train-only corrective
  memory.
- E0 versus E3 is a total-system contrast only.
- P2 temporal separation and P3 action/grounding behavior are supported by
  causal and mechanism diagnostics; Table 2 must not mislabel them as isolated
  one-row ablations.

## Completed prerequisites

| ID | Status | Deliverable | Acceptance evidence |
| --- | --- | --- | --- |
| T2-01 | DONE | Table 2 implementation and research-boundary tests | `tests/table2/`: 1,541/1,541 passed. |
| T2-02 | DONE | Full repository regression run | 1,662 passed; four unchanged historical failures remain outside Table 2. |
| T2-03 | DONE | Active public development-task registry | 50 page-state-compatible WebArena tasks registered; historical 0--49 and active 50 are permanently excluded from the later final campaign. |
| T2-04 | DONE | Controlled recovery registry | 15 registered scenarios, executed as a 60-episode deterministic engineering smoke across E0--E3. |
| T2-05 | DONE | Kaggle P4 train-only preparation rehearsal | Downloaded output validates `PASS` with package status `REVIEW_REQUIRED`; 24,107 train rows, 2,065 candidates, zero validation/test/locked-test reads, and zero fatal errors. |
| T2-06 | DONE | Local deterministic runtime smokes | Success-chain E0--E3, failure/memory E0--E3, and 15x4 recovery smoke completed. These remain `ENGINEERING_SMOKE_ONLY`. |

## Remaining work in execution order

### T2-07 — Freeze the active source

Status: `DONE`

- [x] Review worktree scope and confirm no DGX training/checkpoint source was
  intentionally modified.
- [x] Confirm `git diff --check` passes.
- [x] Audit all 66 changed/untracked paths before staging: no raw artifact,
  output, locked mount, checkpoint, model weight, embedding, browser profile,
  credential, screenshot, trace, video, HAR, or environment-secret path; no
  untracked file exceeds 1 MiB; no obvious secret-bearing assignment appears
  in the textual patch; and the staging area remains empty.
- [x] Re-run the complete Table 2 suite from the exact pre-commit worktree:
  1,541/1,541 passed in 292.88 seconds on 2026-09-06.
- [x] Re-run the complete repository suite: 1,662 passed, four unchanged
  historical failures, and two warnings in 314.53 seconds on 2026-09-07. All
  eight source/test files implicated by the four failures are byte-unchanged
  from `HEAD`.
- [x] Commit the verified Table 2 source only after explicit user
  authorization.
- [x] Record the execution commit:
  `178c24ac898f1feb17848ed1fa40fa22b97d053f`.

Stop condition: any new Table 2 failure, whitespace error, unexpected training
change, or unreviewed generated/raw artifact in Git.

### T2-08 — Reissue the authenticated Kaggle preparation package

Status: `DONE`

- [x] Regenerate the Git-bundle transport from the authorized active-source
  commit.
- [x] Update the existing private Kaggle source-transport dataset.
- [x] Re-run the CPU-only, internet-disabled prepare-only job.
- [x] Download and validate the new receipt and package against the exact
  producing commit.
- [x] Require `REVIEW_REQUIRED`, zero fatal errors, and zero
  validation/test/locked-test reads.

Validated `results-178c24a.zip`: 24,107 training rows, 2,065 candidates,
zero fatal errors, and application-recorded zero validation/test/locked reads.
Receipt core SHA-256:
`9d9bf5474b7b0eab44cbd73191f66c6ebc3f07e41928202937c1c9fead82b013`.
This preparation step is complete; no further preparation rerun is scheduled.

### T2-09 — Freeze the active 50-task WebArena content and duplicate assignments

Status: `IN_PROGRESS`

- [x] Complete an explicitly uncommitted rehearsal from the pinned
  `libwebarena-0.0.4` wheel: exact 50-task export `PASS`; interface audit
  `PASS`; 50 compatible, zero incompatible, zero assistant-answer tasks, and
  `handoff_eligible: true`. The resolved task-set SHA-256 is
  `33cc316005b8b5974755f7c382394ce80255ce18eb2900021baee3fb7ded0cc5`.
  This rehearsal is not the final source-frozen artifact.
- [x] Export the exact active 50 tasks from the pinned WebArena source using
  commit `178c24a` and the example URL map. Live deployment URLs remain pending.
- [x] Recompute the six-action/page-state interface audit and require 50/50
  compatible tasks.
- [x] Create a clean detached checkout of `178c24a` outside the worktree and
  revalidate the downloaded preparation receipt there: `PASS`. Checklist edits
  therefore do not require a new Kaggle preparation run.
- [x] Stage the next joint-audit launcher, pinned source bundle, and all eight
  preparation evidence files in
  `/home/kiyas-mahmud/Thesis/table2-inputs/table2-joint-job-178c24a/`.
  Local checks pass: Bash syntax, nine payload hashes, missing-argument
  rejection, and rejection of the example-domain task export before Gold reads.
  The launcher builds then independently replays assignments using the existing
  pinned producer. Raw-image execution has NOT been performed.
- [ ] Bind the approved registry, task-content hashes, source wheel, task URL
  placeholders, and interface-audit result.
- [ ] Run joint exact/near duplicate assignment across the 2,065 P4 candidates
  and all 50 WebArena tasks.
- [ ] Validate the assignment package from the original Gold train JSON,
  recovery supplement JSON, preparation receipt, task export, and tracked
  producer/config bytes.

### T2-10 — Complete P4 provenance and eligibility

Status: `NOT_STARTED`

- [ ] Independently assess executed recovery success and final task success for
  every potentially admitted candidate remaining after duplicate filtering.
- [ ] Provide the additional P4 label review required for source rows whose
  registered review status is pending.
- [ ] Reject incomplete, duplicate, same-task, unsuccessful, unverified, or
  provenance-invalid candidates.
- [ ] Create and validate `table2-memory-provenance-v1` with complete candidate
  coverage and zero non-training reads.
- [ ] Finalize and validate the 50+15 duplicate-audit manifest.

The preparation queue is not itself provenance and currently claims zero
eligible memory items.

### T2-11 — Build and freeze the PC-01 pilot memory store

Status: `NOT_STARTED`

- [ ] Bind PC-01 epoch 6, seed 42, its exact resolved configuration, processor,
  and base-model revision.
- [ ] Generate the exact 768-dimensional Pillar 4 representation for every
  eligible item.
- [ ] Apply L2 normalization, cosine similarity, top-3 retrieval, deterministic
  memory-ID tie-breaking, and train-only leave-one-out threshold calibration.
- [ ] Build the immutable seven-file store and validate its transfer manifest.
- [ ] Require E3 read-only operation and prohibit evaluation-memory writes.

No PC-01 retraining and no seeds 43--44 are part of this step.

### Local CUDA diagnostic prerequisite (not the registered DGX gate)

- [x] Install isolated CUDA-enabled PyTorch at
  `/home/kiyas-mahmud/Thesis/table2-envs/local-cuda-diagnostic-v1`.
  Python 3.14.4, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126.
- [x] Verify CUDA visibility and an FP16 256x256 matrix multiplication on the
  NVIDIA MX450: PASS. Compute capability 7.5; PyTorch reports 1,763,901,440
  device-memory bytes. NVIDIA-SMI reports 2,048 MiB physical VRAM.
- [x] Run a separate checkpoint-backed local load diagnostic with unchanged
  model settings: FAIL with measured CUDA OutOfMemoryError during the existing
  QLoRA model-construction path. Checkpoint SHA-256 matched the registered value
  and deserialization succeeded; model construction requested 892 MiB with only
  79.31 MiB free. No trained-weight attachment or inference completed.
  Result: `/home/kiyas-mahmud/Thesis/table2-inputs/local-pc01-load-diagnostic-result.json`.
  No offloading, precision changes, image-size reduction, or gate bypass was
  applied. Local diagnostic dependencies differ from the DGX environment; this
  records failure of the unchanged loader on this local stack, not all possible
  implementations. Next checkpoint inference test should run at the university
  lab under the registered environment.

The existing CPU environment, driver, model weights, model configuration, and
DGX compatibility gate were not changed. This local environment is not the
registered DGX runtime and does not qualify as a Table 2 compatibility PASS.

### University lab checkpoint prerequisite — measured 2026-09-09

Status: `WAITING_FOR_EXTERNAL_RUNTIME`

These are new lab observations; earlier laptop and synthetic results above do
not describe this host. No full checkpoint compatibility run or inference has
completed, and no compatibility PASS receipt exists.

- [x] Inspect the existing `/home/aiub/kiyas/webagent` repository and safely run
  `git pull --ff-only origin Code`. HEAD is
  `875bbae27095e3e808875403a9f253cd2f0b02a1`; `178c24a` is an ancestor.
  Preserve the pre-existing modified notebook and untracked `.vscode/` files.
- [x] Inventory running processes and GPU use. Two existing ipykernels
  (PIDs 228460 and 228462) were present; neither appeared in NVIDIA-SMI's
  GPU process list. No training command was apparent. No process was stopped.
- [x] Locate and hash the checkpoint, full report, and contract under
  `/home/aiub/kiyas/webagent_comparison/outputs/model_comparison/qwen2vl_2b_gold_v2_8_dgx/seed_42/`.
  The checkpoint is at
  `full/checkpoints/Y_QWEN2VL_2B_GOLD_V2_8_DGX_FULL_SEED42/best_e6_outcome-mcc0.624.ckpt`:
  291,071,781 bytes, SHA-256
  `9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a`.
  `full/report.json` SHA-256 is
  `4d8990ecf4ed642a17c5be54af3a897a7785e76458343170d4edc098fbdcdcc8`;
  `run_contract.json` SHA-256 is
  `7ed17a7c68ff02a4cd693908aeb85710cee7ef0190339a52eb277cabd083a930`.
  Their selection facts match epoch 6, seed 42, the pinned revision, and zero
  test reads. Corresponding copies inside the repository have identical hashes.
  Export-manifest cross-binding still requires the transferred v3 export.
- [x] Measure the existing `.venv`: Python 3.12.3, aarch64, PyTorch
  2.13.0+cu130, Transformers 4.57.6, PEFT 0.20.0, bitsandbytes 0.50.0,
  torchvision 0.28.0+cu130, CUDA available on NVIDIA GB10.
- [x] Invoke the unchanged gate's `_require_registered_dgx_cuda` prerequisite.
  **FAIL:** `DGX CUDA runtime.cuda_device_total_memory_bytes differs: expected
  130662936576, got 130662940672`. The observed value is 4,096 bytes larger.
  This is an exact host-identity rejection, not an OOM, checkpoint corruption,
  or inference-quality result. The full gate checks this before artifact loading.
- [x] Independently invoke its `_repository_attestation` prerequisite.
  **FAIL:** `runtime source repository is dirty; commit all gate/runtime changes first`.
  Local work was not stashed, discarded, hidden, or committed to satisfy it.
- [x] Locate both cached Qwen revision directories under
  `/home/aiub/kiyas/webagent_comparison/hf_cache/hub/` and
  `/home/aiub/kiyas/webagent_full/hf_cache/hub/`, each at
  `models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c/`.
  The unchanged snapshot validator rejects symlinks. Both contain 11 symlinked
  entries and lack the registered `.gitattributes`, `LICENSE`, and `README.md`.
  Neither is yet a verified complete 14-file snapshot; cached files are preserved.
- [x] Locate transferred `results-178c24a.zip` and the exact 11-file v3 export
  in `/home/aiub/kiyas/table2-inputs/` after the user completed the transfer.
  All v3 manifest/payload hashes and report/contract cross-bindings pass.
  Preserve both transferred originals unchanged.
- [x] Inspect all 477 ZIP entry names for traversal, duplicates, and symlinks;
  extract only the eight registered evidence files, read-only, under
  `/home/aiub/kiyas/table2-inputs/kaggle-p4-output-178c24a/table2-p4-prepare-only-v1/`.
  Outer receipt file/core hashes match the handoff. The strict downloaded-output
  validator returns `PASS`, package status `REVIEW_REQUIRED`, using a temporary
  clean verification checkout of the producing `178c24a` commit. That temporary
  checkout was removed; the existing lab repository remains the working repo.
  No preparation rerun or Gold reads occurred.
- [x] Prepare a complete regular-file Qwen snapshot at
  `/home/aiub/kiyas/table2-inputs/qwen2-vl-2b-instruct-895c3a49bc3fa70a340399125c650a463535e71c/`.
  Reuse all 11 cached payloads and fetch only the three missing metadata files
  from the exact pinned Hugging Face revision. Preserve the cache. All 14 file
  sizes/hashes and the full directory payload match the transferred manifest;
  all output files are read-only. No model weights were downloaded.
- [x] Invoke the full unchanged checkpoint compatibility CLI after transfer.
  It exits 1 at `_require_registered_dgx_cuda`, with the same exact 4,096-byte
  memory mismatch above, before artifact loading. No PASS receipt is written.
  Separately run `_validate_pc01_export_bundle` with the complete snapshot:
  checkpoint deserialization/config binding, processor parity, training-source
  identity and snapshot checks pass through to line 705, where clean-source
  attestation rejects the existing local changes. This input validation is
  not a model-inference run or a substitute for the failed CUDA gate.
- [ ] Resolve the registered host-identity mismatch without weakening checks,
  resolve the clean-source prerequisite with the owner, and supply the complete
  snapshot and authenticated export. No gate constant, model setting, package,
  or driver was changed.
- [ ] Run the full real checkpoint-backed gate using verified paths and a new
  output path. Model construction, restoration, E0 absence checks, callback
  parity, P4 tensor checks, and state immutability remain unexecuted on this host.

Local diagnostic inventory and exact prerequisite tracebacks are preserved at
`/home/aiub/kiyas/table2-evidence/lab-preflight-20260909T063546Z/`.
This is `LAB_PREREQUISITE_DIAGNOSTIC_ONLY`, not a compatibility receipt.

Post-transfer evidence is saved under `/home/aiub/kiyas/table2-evidence/`:
`kaggle-transfer-validation-20260909.log`,
`qwen-snapshot-validation-20260909.json`,
`pc01-compatibility-transfer-attempt-20260909.log`, and
`pc01-input-validation-20260909.log`. The full CLI attempt failed before model
construction; the GPU registration and clean-source prerequisites remain open.

Approved follow-up on 2026-09-09: correct only the runtime GPU-memory identity
to the measured `130662940672` bytes. Historical training environment records
and all model/configuration, artifact, inference, and exact-match checks remain
unchanged. Targeted compatibility, receipt-handoff, and selection-binding tests:
25 passed. Before the real rerun, preserve notebook/VS Code edits in a hashed
external backup and a path-scoped Git stash, commit only the approved gate/test
and documentation changes, and restore the unrelated work after execution.
No push is part of this step. Record the measured rerun outcome below.

#### Approved lab rerun outcome — 2026-09-09

**FAIL at final selected-model state immutability, after real inference.**
Execution source: `8879dbb5095da224ee972a024fe866333f6494d6`, committed locally
and clean throughout the gate. No source or model check was bypassed.

- [x] Exact corrected CUDA runtime identity, all artifact bindings, and
  clean-source attestation pass.
- [x] Resolve an initial processor-path rejection by moving only the prepared
  snapshot into the required revision-bearing directory layout:
  `/home/aiub/kiyas/table2-inputs/Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c/`.
  This supersedes its earlier flat path. All payload bytes remain unchanged;
  the gate revalidates the complete snapshot. Initial failure log:
  `/home/aiub/kiyas/table2-evidence/pc01-compatibility-8879dbb.log`.
- [x] Construct the exact selected model on CUDA, restore all seven checkpoint
  roles, and pass frozen/evaluation/CUDA assertions. No OOM occurs.
- [x] Load the separate unadapted E0 base and pass adaptation-absence checks.
- [x] Complete the gate's direct/runtime/repeat pre-action, diagnosis,
  executed-recovery assessment, and P4 embedding checks on its deterministic
  synthetic observations. These are compatibility checks, not live episodes.
- [ ] Pass final whole-model state immutability. The second full attempt exits
  1 at `pc01_checkpoint_compatibility.py:2261` with
  `selected model state changed during inference-only compatibility checks`.
  The before/after full state identities differ. The current failure log does
  not identify the specific changed tensors or establish why they changed;
  do not label this optimizer training or harmless caching without evidence.
  Full log:
  `/home/aiub/kiyas/table2-evidence/pc01-compatibility-8879dbb-attempt2.log`.
- [x] Restore notebook and VS Code settings byte-for-byte against the external
  backup manifest after execution. Backup:
  `/home/aiub/kiyas/table2-evidence/local-work-backup-20260909T125252/`.
  Safety stash `67b617ab83d73fb3a14a716fd9f267731ca211f2` is retained.
  Existing kernels were not stopped; the gate process exited and released GPU
  allocations. No changes were pushed.

No PASS receipt exists. Next investigate exactly which selected-model state
entries change during first inference, preserving the immutability check and
model settings. Pilot execution remains blocked and paper Table 2 stays `N/R`.

#### Per-tensor diagnosis — 2026-09-09

**Cause identified; compatibility still FAIL, no repair applied.** A diagnostic
rerun from clean `8879dbb` used read-only Python tracing of the existing state
hash function's metadata/raw bytes. No gate function, model tensor, library,
model setting, or check was replaced. The original gate again rejected final
state immutability; this diagnostic does not produce a compatibility PASS.

- Exactly 214 of 2,823 state entries change, all backbone bias parameters:
  float32 before inference, float16 afterward. All 2,609 other entries retain
  their exact keys, dtype, shape, byte count, and payload hash.
- The selected state payload changes from 2,990,977,941 to 2,990,112,661 bytes.
  Before SHA-256:
  `eb5daaa74e15a1e4c626a061ab9d58fe324e0f00bda4d5a9790a81879957a75d`;
  after SHA-256:
  `25f572f6315d44152bd6c295ea02a4ee41fdba15484e78d5202b9cc4e38bbfd8`.
- All 214 before/after bias hashes were independently checked against their
  corresponding tensors in the pinned base safetensors files. Before matches
  the configured FP16 load promoted to FP32; after matches that FP16 load
  exactly. The observed changes are fully explained by this dtype conversion.
  Learned adapter/head state entries are among the unchanged entries.
- Installed PEFT 0.20.0, `peft/utils/other.py:197-202`, promotes non-quantized
  FP16/BF16 parameters to FP32 during `prepare_model_for_kbit_training`.
  Installed bitsandbytes 0.50.0, `bitsandbytes/nn/modules.py:630-635`, changes
  `bias.data` in place to the forward compute dtype in `Linear4bit.forward`.
  This source behavior explains the measured first-inference state mutation;
  evaluation mode and disabled gradients do not prevent a `.data` assignment.
- Direct/runtime/repeat and P4 checks again pass, but that does not satisfy
  whole-state immutability. Do not waive the failure or silently warm up before
  hashing. A remedy must preserve the original registered compute arithmetic
  while preventing mutation of stored bias parameters, with separate review
  and validation before any new compatibility claim.

Evidence directory:
`/home/aiub/kiyas/table2-evidence/state-diagnostic-8879dbb/`.
`result.json` contains both per-entry inventories, exact differences, restore
and inference results, and the failure traceback. `bias-cast-verification.json`
binds all 214 changed biases to the pinned base payload and records hashes of
the inspected installed library sources. `diagnose.py` and `run.log` preserve
the diagnostic method and output. Local notebook, VS Code, and prior checklist
edits were restored byte-for-byte before this update; safety stash
`20bd91509dc56d5ba3ebc4ae54a3a643cb452610` and external backups are retained.

Lab live-runtime inspection: the generic broker factory bridge and monotonic
collector/CLI are implemented. The concrete live BrowserGym/sealed-evaluator
factory and calibration harness remain missing; the only production
`run_registered_operation`/`reset_and_fingerprint` declarations found are the
collector protocol. Approved probes, measured timing, reset non-persistence,
and campaign evidence bindings remain pending. No live measurements were made.
Docker daemon inventory is blocked by socket permission denial in this session;
do not infer that no containers exist. BrowserGym, libwebarena, and Playwright
are absent from the inspected model `.venv`. Disk has approximately 3.4 TiB
available and RAM approximately 113 GiB available at inspection; this does not
establish service-image architecture compatibility or map deployment readiness.
No download, service launch, package installation, Gold read, Kaggle rerun,
memory admission, or pilot episode was performed. P4 and final Table 2 remain
unchanged: no eligible store established; paper status `N/R`.

### T2-12 — Bring up the live WebArena runtime

Status: `WAITING_FOR_EXTERNAL_RUNTIME`

- [x] Confirm Docker CLI installation on the local x86 host: Docker Engine
  Community client `29.8.0`, `linux/amd64`.
- [x] Verify Docker daemon access from the execution session: client and server
  `29.8.0`, `linux/amd64`; Docker group access works.
- [x] Inventory existing containers and images: only the stopped
  `awesome_faraday` hello-world container and hello-world image were found.
  Both are preserved; no WebArena services are installed yet.
- [ ] Check per-service download, unpacked-image, runtime-disk, and RAM needs
  against local capacity before downloading (currently 352 GB free disk).
- [ ] Pin compatible WebArena service images and map backend, then download
  and launch benchmark services with access restricted to the experiment hosts.
- [x] Validate the pinned local BrowserGym/WebArena/Playwright packages and
  launch Chromium at the registered viewport.
- [ ] Supply reachable runtime URLs for GitLab, Reddit, Shopping, Shopping
  Admin, Map, Wikipedia, and Homepage.
- [ ] Freeze credentials outside Git, resets, service state, browser/controller
  versions, official evaluator, settle rules, and infrastructure classifier.
- [ ] Pass package, browser, seven-service, reset, and evaluator preflight.

The seven reachable services are runtime inputs; DGX shell access is not
required when inference is exposed through the registered split-deployment
boundary.

### T2-13 — Run the real matched E0--E3 gate

Status: `NOT_STARTED`

- [ ] Run one live development task as one complete matched E0--E3 block.
- [ ] Require identical E1/E2/E3 pre-action outputs under identical observable
  inputs and stage-keyed RNG until a registered intervention occurs.
- [ ] Require E2 zero memory queries, E3 zero writes, and E2/E3 identity until
  E3's first admitted intervention.
- [ ] Require sealed-evaluator blindness, complete logs/hashes, no manual rescue,
  and no partial-system rerun.

### T2-14 — Execute the provisional engineering pilot

Status: `NOT_STARTED`

- [ ] Run 15 controlled recovery scenarios across E0--E3: 60 episodes.
- [ ] Run 50 public development tasks across E0--E3: 200 episodes.
- [ ] Permit at most one whole-block infrastructure rerun.
- [ ] Require zero unexplained missing episodes and a complete matched package
  for every valid block.
- [ ] Preserve all ordinary-task failure incidents as additional P1 recovery
  evidence; the controlled 60 episodes are not the complete recovery sample.

### T2-15 — Audit, analyze, and report

Status: `NOT_STARTED`

- [ ] Complete the registered blinded 20-episode audit.
- [ ] Validate the complete artifact package and reproduce metrics solely from
  frozen logs.
- [ ] Report TSR, RSR, success after initial failure, browser actions, loop
  rate, unrecovered failure rate, paired contrasts, confidence intervals, and
  corrected tests with numerators and denominators.
- [ ] Publish only `PILOT_ONLY` engineering results.
- [ ] Keep final paper Table 2 at `N/R` until the registered PC-01/PC-02/PC-03
  validation-only promotion and later final campaign are complete.

## Immediate next action

Approved bias-mutation remedy implemented on 2026-09-09 in the evaluation
runtime only: preserve the pinned CUDA arithmetic using a temporary bias cast,
with an exact installed bitsandbytes version/forward-source guard. Both selected
and E0 models use the same implementation; installed libraries and training
source are untouched. All 39 targeted tests pass, including real CUDA numerical
equivalence to upstream with/without bias and FP16/FP32 inputs, exact repeated
outputs, and complete layer-state immutability. A new source-bound full gate
receipt is still required; record its measured outcome after the run.

At the lab, the transferred inputs and complete pinned snapshot are now verified.
The approved runtime identity correction and temporary clean-source procedure
allowed real checkpoint-backed inference at commit `8879dbb`. The final state
failure is now traced to 214 bitsandbytes bias dtype mutations, with every change
verified against the pinned base snapshot. Review a non-mutating bias-cast remedy
that preserves compute arithmetic and all checks, then validate it before
rerunning the gate. No compatibility PASS is available.
The earlier Docker daemon access and container inventory were laptop results;
lab daemon access is denied in this session. Service hosting, including Map,
needs a concrete compatible deployment and verified URLs before task binding.

The joint duplicate-assignment job is staged at
`/home/kiyas-mahmud/Thesis/table2-inputs/table2-joint-job-178c24a/README.md`.
Next resolve the actual WebArena service URL map and produce the deployment-bound
active task export, then execute the staged job where the Gold images already
live. No further prepare-only rerun is required. Independent success review,
checkpoint-backed embeddings, live service preflight, and real episodes remain
unchecked; neither preparation nor synthetic smoke substitutes for them.
