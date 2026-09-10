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

Docker daemon access and container inventory are complete. Do not
download the full WebArena image set until its storage requirements are checked.
Then configure and preflight the actual benchmark websites before binding their
addresses to the task export. Docker installation alone is not website setup.

The joint duplicate-assignment job is staged at
`/home/kiyas-mahmud/Thesis/table2-inputs/table2-joint-job-178c24a/README.md`.
Next resolve the actual WebArena service URL map and produce the deployment-bound
active task export, then execute the staged job where the Gold images already
live. No further prepare-only rerun is required. Independent success review,
checkpoint-backed embeddings, live service preflight, and real episodes remain
unchecked; neither preparation nor synthetic smoke substitutes for them.
