# Table 2 Remaining Execution Checklist

Last reviewed: 2026-09-07

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

Status: `WAITING_FOR_AUTHORIZATION`

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
- [ ] Commit the verified Table 2 source only after explicit user
  authorization.
- [ ] Record the full commit SHA used by every later Table 2 artifact.

Stop condition: any new Table 2 failure, whitespace error, unexpected training
change, or unreviewed generated/raw artifact in Git.

### T2-08 — Reissue the authenticated Kaggle preparation package

Status: `NOT_STARTED`

- [ ] Regenerate the Git-bundle transport from the authorized active-source
  commit.
- [ ] Update the existing private Kaggle source-transport dataset.
- [ ] Re-run the CPU-only, internet-disabled prepare-only job.
- [ ] Download and validate the new receipt and package against the exact
  producing commit.
- [ ] Require `REVIEW_REQUIRED`, zero fatal errors, and zero
  validation/test/locked-test reads.

This is a short data-preparation rerun, not model training.

### T2-09 — Freeze the active 50-task WebArena content and duplicate assignments

Status: `NOT_STARTED`

- [x] Complete an explicitly uncommitted rehearsal from the pinned
  `libwebarena-0.0.4` wheel: exact 50-task export `PASS`; interface audit
  `PASS`; 50 compatible, zero incompatible, zero assistant-answer tasks, and
  `handoff_eligible: true`. The resolved task-set SHA-256 is
  `33cc316005b8b5974755f7c382394ce80255ce18eb2900021baee3fb7ded0cc5`.
  This rehearsal is not the final source-frozen artifact.
- [ ] Export the exact active 50 tasks from the pinned WebArena source.
- [ ] Recompute the six-action/page-state interface audit and require 50/50
  compatible tasks.
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

### T2-12 — Bring up the live WebArena runtime

Status: `WAITING_FOR_EXTERNAL_RUNTIME`

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

The exact pre-commit verification in T2-07 reproduced the prior results and the
stop-before-commit boundary has been reached. Request explicit authorization,
then commit the verified source and begin T2-08.
