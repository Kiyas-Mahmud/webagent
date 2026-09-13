<!-- MINIWOB_INTERFACE_V2_STATUS_START -->
# Table 2: completed evaluation; development revision completed

The completed 120-episode result remains unchanged: **E0 6/30, E1 0/30,
E2 6/30, E3 6/30; audit PASS**. See the
[authoritative evaluation results](TABLE2_MINIWOB_INTERFACE_V2_RESULTS.md).
Neither primary contrast met the approved Holm-adjusted 5% criterion.

The authorized [planner and continuation revision](TABLE2_DEVELOPMENT_V3.md)
is implemented and tested: **140 tests passed; all 16 development episodes ran;
corrected evidence audit PASS**. Counts: E0 1/4, E1 0/4, E2 2/4, E3 2/4.
E2/E3 each previously completed 1/4 on these matched development resets.

Remaining limits: both planners still submitted the text form without typing;
no live observable-effect continuation occurred; sequence clicks failed overlapping
target-point validation. Memory filtering excluded ten candidate occurrences but
added no completion. The original audit's redacted-control replay error is fixed,
with its failure and the separate corrected audit preserved.

- [x] Implement clearer planning, bounded observable-effect continuation and selective memory.
- [x] Run engineering checks and one matched 16-episode development check.
- [x] Audit all outcomes, preserve evidence and report failures.
- [ ] Establish typing-before-submission and actual live multi-step continuation.
- [ ] Investigate overlapping target geometry without replacing model-selected actions.

Another final evaluation remains premature and is not queued. No retraining or
embedding regeneration occurred. The checkpoint and all 1,974 embeddings are
unchanged. Historical sections below do not override this current status.
<!-- MINIWOB_INTERFACE_V2_STATUS_END -->

---

The following sections are historical records of earlier profiles and runs.

# Table 2 Remaining Execution Checklist

## Requested full-credit evaluation completed — 2026-09-10

- [x] Correct missing generation-rejection feedback and verify matched E2/E3 inputs.
- [x] Verify visible custom-link support and full raw-reward scoring.
- [x] Complete the combined matched development check: 16/16, no runtime errors;
  E2 1/4, E3 2/4, six verified memory contexts. Preserve the E2-regression caveat.
- [x] Freeze the next 24-episode package and pass its full asset/source/audit preflight.
- [x] Execute `miniwob-table2-completion-v1/launch.py --run` and report both
  paired contrasts and all failures. Completed 24/24; all audits PASS;
  E0/E1 0/6, E2/E3 1/6; 11 verified E3 contexts, zero additional memory completion.
- [x] Verify frozen source/model/memory integrity after execution and preserve
  all prior results. No episodes were rerun and no training was performed.

Results: [TABLE2_FULL_CREDIT_RESULTS.md](TABLE2_FULL_CREDIT_RESULTS.md).
No further evaluation is queued; the new result is reported rather than N/R.
Earlier entries below describe completed or superseded runs, not new prerequisites.

## Memory follow-up completed — 2026-09-10

- [x] Trace the completed replication's seven abstentions and distinguish
  memory-update classification from retrieval usefulness.
- [x] Add a separate, hash-bound training-experience context path without
  changing weights or regenerating embeddings; 114 targeted tests passed.
- [x] Run all four existing development families across E0–E3 and audit actual
  generation exposure. Completed 16/16; five E3 contexts verified; E2/E3 2/4.
- [ ] Demonstrate added memory completion benefit. **Not achieved:** E3-minus-E2
  remains zero; text-entry and output-format failures remain. No new final run
  is queued, and prior Table 2 results remain unchanged.

Details: [memory diagnosis and development results](TABLE2_MEMORY_CONTEXT_DIAGNOSIS.md).

## Active remaining work — BrowserGym + MiniWoB

User-approved benchmark change, 2026-09-09. Follow
[TABLE2_MINIWOB_ACTIVE_PLAN.md](TABLE2_MINIWOB_ACTIVE_PLAN.md).
WebArena deployment, maps, website downloads, WebArena task exports and the old
260-episode schedule are removed from the active work. The local dataset-label
memory route is authorized; a new independent review is not a prerequisite.

- [x] Finish P4 checkpoint embeddings, training-only calibration and frozen read-only retrieval.
  Completed: 1,974 × 768-dimensional PC-01 embeddings, finite and L2-normalized.
  Training-only leave-source-task-out calibration threshold: `0.7371385097503662`.
  Independent CPU recomputation of all calibration pairs/threshold and artifact,
  input identity, read-only permission, deterministic retrieval and exclusion checks: PASS.
  Receipt: `/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1/verification.json`.
  E3 adapter code is connected in the development runner; live intervention verification
  belongs to the next task and is not claimed complete.
- [x] Complete the full matched E0–E3 MiniWoB development check.
  Completed 2026-09-10: `miniwob-revised-evaluation-v1/development-v5/`.
  16 episodes over all four existing development tasks; matched reset screenshots
  and selected-policy initial outputs, 4 executed recovery actions and assessments,
  8 E3 queries/interventions, zero runtime errors, unchanged memory store.
  Evidence audit PASS. Earlier implementation failures are preserved in development,
  development-v3 and development-v4; they are excluded from final evaluation.
- [x] Freeze and run the revised MiniWoB evaluation and report both paired contrasts.
  Completed 2026-09-10: all 24 episodes, no overlap exclusions or runtime errors;
  matched-reset/source/store/seed/budget/retrieval/outcome audit PASS.
  E0, E1, E2 and E3 each completed 0/6 tasks. E1→E2 and E2→E3 differences:
  0 percentage points, exact/Holm p=1. E3 executed 6 recovery actions and made
  12 memory interventions. No normal policy action executed.
  Results and limitations: [TABLE2_MINIWOB_EVALUATION_RESULTS.md](TABLE2_MINIWOB_EVALUATION_RESULTS.md).

All three requested tasks are complete. The negative result identifies an action
execution limitation; it does not establish general ineffectiveness of recovery
or memory. Historical WebArena paper claims remain N/R. No additional run is
queued, no retraining or package changes were made, and the frozen P4 store is unchanged.

## Post-evaluation interface work — 2026-09-10

- [x] Fix explicit visible-name grounding and nullable action serialization without retraining.
- [x] Run and audit eight matched development episodes: E0/E2 button success,
  E1/E3 button failure; all systems fail text entry. Focused tests: 118 passed.
- [x] Preserve the original 24-episode evidence and frozen P4 store; archive the
  revised producing source and raw model outputs.
- [x] Add and test causal E3 admission/abstention: eight matched development
  episodes audited; E3 button success, three queries/abstentions and zero
  interventions. E1 numerical full-forward parity passes on two saved inputs.
  Focused suite: 128 passed. No claim of added memory benefit.
- [x] Connect supplied recovery diagnosis/strategy/history to generation; support
  single fenced JSON and explicit visible role descriptions. Complete all four
  matched development families: E0 1/4, E1 0/4, E2 2/4, E3 2/4; audits PASS,
  140 focused tests pass. Text entry and sequencing now expose executed wrong
  actions; these and E1 failures remain recorded. Five memory queries abstained;
  no additional memory benefit is claimed.
- [x] Complete full-path audit and freeze the revised 24-episode replication:
  241 audit tests, reporting regression and asset/environment/source preflight
  PASS. No further development test or positive-score requirement.
- [x] Execute and report all 24 replication episodes: E0/E1 0/6, E2/E3 2/6
  benchmark-scored successes; seven memory abstentions and zero interventions.
  Runtime audit PASS; separately amended context audit PASS, with its original
  failure preserved. No episode reruns or scoring changes. Disclose the checkbox
  positive-partial-credit caveat and prior task exposure.
  See [TABLE2_REPLICATION_RESULTS.md](TABLE2_REPLICATION_RESULTS.md).

Details: [TABLE2_CAUSAL_RECOVERY_INTERFACE_REPORT.md](TABLE2_CAUSAL_RECOVERY_INTERFACE_REPORT.md).

## Historical execution log — superseded requirements are not active tasks

The material below preserves earlier evidence, failures and decisions. Its old
WebArena/provenance blockers and next-step suggestions do not override the active
MiniWoB plan above.

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

Status: `DONE` — real compatibility PASS at `ca10c72`; earlier failed attempts
below are retained as history. See the successful outcome in Immediate next action.

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

Completed artifact architecture/storage review (2026-09-09): bounded reads of
the four official upstream Google Drive Docker archives, with PAX-aware tar
header parsing and manifest-to-config binding, confirm Shopping, Shopping Admin,
Reddit, and GitLab are all **linux/amd64**. They cannot run natively on GB10
ARM64. Full archive payload hashes and live service behavior remain unverified.
The canonical S3 map-data endpoint is reachable. The four website archives,
Wikipedia ZIM, and four map archives total approximately **492.82 GB**.
Keeping those files plus one unpacked copy of each tar requires approximately
**890.44 GB** before additional Docker/runtime/reset-copy overhead; this is an
estimate, not a measured installed footprint. Available disk is about 3.4 TiB.
The map contains physical database volumes and routing data whose ARM64/server
version compatibility remains to be tested. Full deployment is blocked on an
x86-64 website host or a separately validated emulation route, not Docker access.

Evidence and exact byte/config identities:
`/home/aiub/kiyas/table2-evidence/service-artifact-probe-20260909/REPORT.md`.
No large archive pull, service launch, package change, or pilot run was made.

Docker-access update, 2026-09-09 (supersedes the denied-access observation below):
the user repaired access and the agent verified Docker client/server 29.2.1,
Compose v5.0.2, and native linux/arm64, without sudo. Two stopped hello-world
containers and the hello-world image are the only current inventory; all were
preserved. Approximately 3.4 TiB disk remains available.

Registry manifest inspection confirms linux/arm64 variants for
`mediagis/nominatim:4.2`, `ghcr.io/project-osrm/osrm-backend:v5.27.1`,
`overv/openstreetmap-tile-server:latest`, and `ghcr.io/kiwix/kiwix-serve:3.3.0`.
Exact platform digests and raw manifests are saved under
`/home/aiub/kiyas/table2-evidence/docker-arm64-review-20260909/`.
This is image-platform availability only, not compatibility of the archived
database volumes, a pinned tile-server version, live service health, or campaign
approval. Shopping/Admin/Forum/GitLab archive architectures, map frontend/data
compatibility, complete storage footprint, and live integration still require
verification. No service image/data pull, launch, or pilot episode was performed.

Lab follow-up after checkpoint PASS (2026-09-09): direct Docker socket access
is denied, and `sudo -n` is unavailable under this session's no-new-privileges
flag. Container/image inventory remains unknown. The intended browser/service
host has been requested. Upstream map bootstrap includes an x86-64 AWS CLI
download and cannot be used unchanged on GB10 ARM64; service image architecture,
immutable identities, and total unpacked/runtime capacity remain unverified.
Metadata-only archive mirror checks did not resolve sizes (TLS timeouts or no
matching artifact); no large downloads or services were started.

Concrete source blocker confirmed:
`browsergym_webarena._load_process_broker_browser_runtime_factory` raises until
validated deployment inputs are provided. The existing factory/adapter is not
a registered live bootstrap. The timing collector exists but its live harness,
approved probes, reset audit, and campaign persistence remain to be completed.
The next Kaggle job still depends on deployment-bound task URLs and independent
P4 evidence, not another preparation run.

Detailed host/service/integration sequence and a syntax-checked read-only host
inventory helper are saved in
`/home/aiub/kiyas/table2-evidence/live-deployment-review-20260909/`.
No executable repository source was changed in this review; the compatibility
receipt remains bound to `ca10c72`, and pilot readiness remains unestablished.

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

### Alternative feasibility route: BrowserGym + MiniWoB++

**User-directed local dataset route (2026-09-09):** user confirms there is no
separate review artifact and instructs reuse of the local reviewed dataset.
Do not request another review for this development route. Located exact original
and supplement train JSONs; hashes match preparation. Built a read-only
train-label-backed development memory dataset from the existing candidates,
verified local transition image bytes, and excluded ABORT/NONE strategies.
Manifest and material: `/home/aiub/kiyas/table2-evidence/p4-local-label-memory-v1/`.
This records dataset-label support and preserves pending source metadata; it does
not manufacture independent verification or final-task success. Embeddings and
runtime integration remain unfinished. Registered campaign provenance is not
silently satisfied by this changed development evidence basis.


**Current consolidated status, 2026-09-09:** completed the bounded real E1/E2
v2 diagnostic with visible-target evidence. E2 detects failure and calls the
planner; both recovery attempts fail registered target validation. No recovery
browser action, no measured reward, no recovery benefit established. E1 consumes
30 rejected requests and stops at its budget. Preserve this failure; do not tune
the prompt or snap predicted coordinates to force success. Source snapshots,
logs and limitations: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/v2-controls-20260909T100854Z/REPORT.md`.
P4 audit confirms all 2,065 candidates still lack required independent review
and duplicate bindings; eligible items=0. No completed provenance/store was found
under table2-inputs or table2-evidence. Input request sent to user. No preparation
rerun, locked-data read or fabricated memory. Full E0–E3 evaluation remains
incomplete. Consolidated tracker: `/home/aiub/kiyas/table2-evidence/TABLE2_COMPLETION_STATUS.md`.


**Recovery planner v2 prompt frozen and tested (2026-09-09):** a separate
explicit-schema development prompt produced a valid CLICK response on the saved
failure observation: target `Next button`, normalized bbox `[0.18,0.45,0.26,0.51]`,
value null. Unchanged strict parser passed. This is format compliance only;
no browser action or recovery success measured. No weights, decoding, E0 prompt,
or parser changes. Prompt hash and raw evidence:
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/recovery-planner-v2/REPORT.md`.
Next bind the frozen v2 prompt to the live diagnostic and execute its uncorrected
prediction. No campaign adoption or paper claim implied. Table 2 N/R.

**Planner raw-output mismatch identified (2026-09-09):** unchanged replay
produced CLICK with `target: {bbox: [197,375,227,404]}` and value `Next`.
Top-level `bbox` is missing, target is an object rather than text/null, and the
coordinates are not normalized. Strict rejection is correct; no grounding or
recovery success is established. Raw bytes retained at
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/planner-raw-capture/response.txt`;
analysis in `REPORT.md` alongside it. No prompt/parser/model changes performed.
Proposed next development change: version and freeze an explicit-schema recovery
planner prompt, keeping E0 baseline and strict parsing unchanged. This requires
transparent planner-interface revision, not retroactive repair of results.

**Concrete diagnostic recovery planner implemented and exercised (2026-09-09):**
`frozen_recovery_planner.py` binds the existing frozen unadapted base predictor
and hybrid parameter provider to RecoveryController. PC-01 still selects the
recovery strategy. Same-reset E1/E2 diagnostic: initial inputs/predictions match;
E2 invokes the planner twice, but both outputs fail strict action-schema parsing:
`E0 action output has missing or additional keys` (confirmed by recorded error
hash). No recovery action executed; no output repair or parser relaxation.
Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/planned-recovery-20260909T095159Z/REPORT.md`.
This is a new development binding, not campaign approval. The older missing-
planner diagnostic below is historical. Next capture the raw planner response
on the same causal input to identify which keys differ. Table 2 N/R.

**Live full-runner diagnostic completed (2026-09-09):** E1/E2 reached the
existing EpisodeRunner on matched MiniWoB click-button resets; first policy
outputs and initial screenshots match. E1 stopped after three rejected requests
(loop). E2 diagnosed the failure and selected REPLAN, but both recovery attempts
were rejected because the concrete frozen-provider planner is missing. Zero
browser recovery actions; no recovery benefit measured. Next implement/bind that
planner before another diagnostic. Evidence:
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/matched-recovery-20260909T094759Z/REPORT.md`.
This incomplete-planner engineering package is not pilot/paper eligible despite
the generic runner summary's internal `valid_for_primary` flag. Table 2 N/R.

**Research-direction correction (2026-09-09):** stop the action-only/E0 detour.
The next work is the full failure → diagnosis → executed recovery → assessment
chain. Weak pre-action performance does not invalidate the recovery question.
Source trace completed in
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/recovery-pipeline-trace.md`.
The existing EpisodeRunner can trigger E2 recovery from a parameter rejection;
the small scripts never reached that path. No new live recovery has run.
Next connect the minimal live benchmark adapter to the existing EpisodeRunner
and registered planner, then perform one matched E1/E2 development diagnostic.
E3 still requires admitted frozen train-only memory. The earlier suggestion to
run E0 alone is superseded; no benchmark/protocol change is implied.


User requested a simple, sequential feasibility check on 2026-09-09. This is
not approval to replace the registered WebArena Table 2 campaign.

- [x] Check package support on ARM64/Python 3.12.3.
- [x] Prepare `/home/aiub/kiyas/table2-envs/miniwob-feasibility` separately from
  the model environment. BrowserGym core/miniwob 0.14.3, Playwright 1.44.0,
  gymnasium 1.0.0; dependency check passes.
- [x] Install ARM64 Chromium 125.0.6422.26, build 1117, outside the model
  environment. Use MiniWoB source commit
  `7fd85d71a4b60325c6585396ec4f48377d049838` under `table2-inputs/`.
- [x] Open `miniwob.click-test` headlessly and receive the goal and screenshot.
- [x] Execute a scripted visible-button click: reward 1.0, terminated true,
  truncated false, no action error. Repeat twice with seed 42; both pass.
- [x] Verify same-seed resets: goal, button position, and raw screenshot hash
  match exactly across both runs. Browser closes after the check.
- [x] Connect the frozen PC-01 checkpoint to a live MiniWoB screenshot and obtain
  a model prediction (2026-09-09).
- [ ] Execute a model-selected action without manual repair.
  **One-action attempt failed at action mapping:** PC-01 selected `NAVIGATE`
  (0.329795), ahead of `CLICK` (0.312867) and `SCROLL` (0.298057), for
  "Click the button." The minimal click-only feasibility adapter cannot execute
  `NAVIGATE`; prediction parameter hints contain no URL. No action was substituted,
  no browser step executed, and no task reward was measured. Inference took
  2039 ms (excluding model loading); this is not timing calibration.
  Result and exact prediction:
  `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/pc01-action-20260909T091847Z/result.json`.
  Screenshot, causal request, and model log are alongside it; runnable scripts
  `pc01_predict.py` and `pc01_browser_action.py` are in the parent evidence folder.
  An initial attempt was rejected because the screenshot was writable; the script
  now saves it read-only and the successful inference retained the integrity check.
  The next task is to assess the six-action mapping and existing causal parameter
  resolver before another action attempt. This single prediction establishes no
  completion rate, recovery benefit, memory benefit, or paired E0–E3 readiness.
  **Blocked at CUDA preflight on 2026-09-09 after reboot:** current kernel
  `6.17.0-1032-nvidia` has no NVIDIA GPU module (`modinfo nvidia` fails),
  `/dev/nvidia*` is absent, and PyTorch reports CUDA unavailable / zero devices.
  The `6.17.0-1026-nvidia` kernel has an installed NVIDIA module. No model
  loading or model-selected action occurred. Restore GPU driver availability
  before retrying; no reboot, package change, or notebook interruption performed.
  Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/pc01-action-preflight.json`.
  **GPU blocker resolved, 2026-09-09 15:15 local:** matching `1032` module
  installed, `dpkg --audit` clean, NVIDIA driver 580.173.02 recognizes GB10,
  and the existing model environment passes a real CUDA tensor calculation.
  The earlier preflight evidence records the historical failure; the model-selected
  browser action remains pending.
- [x] Inspect the existing six-action resolver and browser mapper; run the actual
  registered hybrid resolver on the saved PC-01 NAVIGATE decision (2026-09-09).
  Both stages rejected: missing URL hint, then frozen unadapted base rejection.
  No browser action or reward. Evidence and action requirements:
  `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/parameter-review.md`
  and `parameter-result.json` alongside it. Source support for all six actions
  does not establish live MiniWoB compatibility; SELECT observation evidence
  remains absent and file navigation is not permitted by the existing contract.
- [x] Connect the existing hybrid resolver and six-action code mapper to the
  minimal MiniWoB runner; run one live attempt (2026-09-09).
  `pc01_live_worker.py` calls the registered selected policy, hybrid parameter
  provider, `concretize`, and `_action_code` (the last two only on resolution).
  `pc01_live_attempt.py` keeps the browser open and counts rejection against its
  one-attempt budget. Actual result: `ACTION_REJECTED`, one attempted/rejected
  step, zero browser actions, no measured reward. PC-01 again selected NAVIGATE;
  deterministic and real frozen-base stages both rejected. The passive screenshot
  after rejection is not pixel-identical to the initial observation; no browser
  action was issued. Browser closed normally. No substituted click or retry loop.
  Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/pc01-live-20260909T092507Z/result.json`.
  The rejection path is verified live. Resolved six-action execution remains
  unverified on MiniWoB; this is a small feasibility runner, not EpisodeRunner
  integration, calibrated evaluation, or evidence of recovery/memory benefit.
- [x] Verify the resolved CLICK execution path with an explicitly scripted
  visible-button target through `DeterministicParameterProvider`, `concretize`,
  and the existing `_action_code` mapper. Live `miniwob.click-test`, seed 42:
  reward 1.0, terminated true, truncated false, no action error (2026-09-09).
  Scope: `SCRIPTED_RESOLVER_MAPPER_INTEGRATION_ONLY`; no model inference or
  PC-01 success claimed. This validates CLICK plumbing only, not all six actions.
  Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/scripted-mapped-click-20260909T092859Z/result.json`.
  Runnable scripts `scripted_mapped_click.py` and `scripted_click_resolver.py`
  are in the parent evidence directory. Browser closed normally; no package or
  model-setting changes. PC-01's actual NAVIGATE rejection remains unchanged.
- [x] Predeclare and execute a small unchanged-PC-01 feasibility set:
  `click-button`, `enter-text`, `click-button-sequence`, environment seed 42,
  three attempts per task. All nine predictions were NAVIGATE; both parameter
  stages rejected each one. Zero browser actions, no observed completions,
  no measured rewards. Repeated unchanged observations are not independent trials.
  Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/small-set-20260909T093153Z/REPORT.md`.
- [x] Make the initial expansion decision: pause expansion. These probes do not
  establish task suitability for executed recovery or P4 memory. Next inspect
  training/runtime input and label alignment without model changes; the cause
  of the NAVIGATE predictions is not yet established. No benchmark revision or
  paired campaign is approved or implied by the feasibility scripts.
- [x] Audit training/runtime input preprocessing and label alignment (2026-09-09).
  All three saved first-observation prompt strings and input tensors match the
  Gold training methods exactly, with batch-shape normalization. All five
  exported training-source hashes match; label source is byte-identical to the
  recorded training commit. No input/label discrepancy found in these checks.
  No dataset rows read or model settings changed. Empty file-URL domain and
  498x321 screenshots are observations, not established causes of model behavior.
  Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/input-alignment-review.md`
  and `input-alignment.json`. Next inspect existing selected-epoch aggregate
  action metrics/support before deciding further feasibility work.
- [x] Inspect selected epoch-6 aggregate validation action metrics. Report hash
  matches the v3 export; selected metrics match history epoch 6. Action accuracy
  42.17%, macro-F1 0.3267; bbox mean IoU 0.1010 and IoU50 recall 8.09%.
  NAVIGATE has 4,770 training / 1,499 validation examples; the no-data code comment
  is stale. These are Gold validation aggregates, not browser completion rates.
  Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/epoch6-action-review.md`
  and `epoch6-action-review.json`. No new dataset reads, model changes, or training.
  Pause PC-01 task expansion; next useful feasibility comparison is frozen E0 on
  the same fixed tasks/budget, without changing the selected E1–E3 policy.
- [ ] Assess six-action compatibility, meaningful recovery tasks, and eligible
  P4 memory relevance before designing any paired E0–E3 evaluation.
- [ ] Review and explicitly approve any benchmark/protocol revision.

Result: `SCRIPTED_BROWSER_FEASIBILITY_ONLY`, no model used, no recovery or
memory benefit measured. Default MiniWoB rendering produced 498x321 screenshots;
its scaling must be accounted for when mapping model coordinates. This is not
the registered WebArena viewport and no protocol settings were silently changed.
Evidence, screenshots, short runnable script, log, and package inventory:
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/`.
`browser-result.json` reports PASS. Existing model environment versions remain
unchanged. No model retraining, locked data, Gold transfer, or website image
deployment was involved. The checkpoint receipt below remains bound to `ca10c72`.

**Real PC-01 checkpoint-backed compatibility gate: PASS, 2026-09-09.**

- Execution commit: `ca10c72aad52bf60a5e2c5e8b6c8f9e5a9dd1e1e`, clean during
  execution. The runtime fix and earlier host identity correction are committed
  locally; neither has been pushed.
- Read-only canonical receipt:
  `/home/aiub/kiyas/table2-evidence/pc01-compatibility-ca10c72.json`.
  SHA-256: `119257744a22c7b3391a63e1bb0681147657f12dad45d5b691ea57f5cf71aef1`.
  Full log: `/home/aiub/kiyas/table2-evidence/pc01-compatibility-ca10c72.log`.
- All artifact/environment/source checks, seven restored roles, E0 adaptation
  absence, direct/runtime/repeat inference, P4 `[1,768]` tensor, and complete
  state immutability pass. Before and after selected-state SHA-256 both equal
  `eb5daaa74e15a1e4c626a061ab9d58fe324e0f00bda4d5a9790a81879957a75d`.
- All 16 cross-run raw/semantic/repeat/P4 comparisons also match the prior
  unmodified-runtime diagnostic exactly on the registered fixture. Evidence:
  `/home/aiub/kiyas/table2-evidence/immutable-bias-full-model-equivalence.json`.
- Notebook and VS Code settings restored byte-for-byte after execution;
  external backup and safety stash retained. Existing kernels were preserved.
- Scope remains `PROVISIONAL_ENGINEERING_PILOT_COMPATIBILITY_ONLY`, paper
  Table 2 `N/R`. No live WebArena episode or P4 store is established by this PASS.

Use the exact receipt bytes and execution commit for downstream bindings.
This result update is uncommitted so HEAD remains the attested execution commit;
restored local edits mean the current worktree is no longer clean. Do not bind
this receipt to another commit or claim current clean-source readiness.

Approved bias-mutation remedy implemented on 2026-09-09 in the evaluation
runtime only: preserve the pinned CUDA arithmetic using a temporary bias cast,
with an exact installed bitsandbytes version/forward-source guard. Both selected
and E0 models use the same implementation; installed libraries and training
source are untouched. All 39 targeted tests pass, including real CUDA numerical
equivalence to upstream with/without bias and FP16/FP32 inputs, exact repeated
outputs, and complete layer-state immutability. A new source-bound full gate
receipt is still required; record its measured outcome after the run.

At the lab, the transferred inputs and complete pinned snapshot are now verified.
The successful `ca10c72` gate supersedes the earlier `8879dbb` failure. Continue
with live BrowserGym/sealed-evaluator integration, approved timing probes and
calibration harness, service hosting including Map, and P4 evidence/store work.
The 260-episode pilot remains blocked on those independent prerequisites.
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
