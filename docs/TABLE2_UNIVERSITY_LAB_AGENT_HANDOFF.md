# Table 2 university lab agent handoff

Prepared: 2026-09-08; expanded and source-checked 2026-09-09. This is a state snapshot and operational handoff, not a
claim that the live pilot is complete. Read this before making changes.

## 1. What the user wants

Continue the thesis research toward a journal submission using the completed
PC-01 model and the updated Table 2 implementation. Transfer recent source via
GitHub push/pull and generated files via USB. Use the existing lab repository,
environment, checkpoint, and base model where compatible. Do not require a new
checkout by default or reinstall a working training environment unnecessarily.

The laptop cannot load the unchanged model successfully. Move the next real
checkpoint test to the university DGX GB10. This is inference, not retraining.
The user explicitly prohibits changing model settings simply to fit hardware.
Preserve other training jobs and all uncommitted lab work.

## 2. Scientific objective and four pillars

Table 2 asks: under matched browser tasks, environments, and budgets, does
executed failure diagnosis/recovery improve completion, and does frozen
train-only corrective memory add an improvement?

| Pillar | Required role | Never substitute |
| --- | --- | --- |
| P1: failure-aware resilience | Outcome detection, failure diagnosis, recovery decision, concrete recovery execution, post-recovery assessment | Offline strategy prediction for executed recovery success |
| P2: multimodal decision | Screenshot plus task/page context with pre/post/recovery temporal separation | Future state or sealed oracle information in policy inputs |
| P3: multi-tool action and grounding | Six registered action classes, predicted targets, concrete browser execution | Silent repairs or oracle-selected targets |
| P4: corrective memory | Immutable train-only successful-recovery store, retrieval, admitted intervention | Test-time writes or memory-flag accuracy as proof of runtime benefit |

| System/contrast | Interpretation |
| --- | --- |
| E0 | Unadapted selected base model |
| E1 | Trained multimodal pre-action policy and grounding |
| E2 | E1 plus diagnosis and executed recovery |
| E3 | E2 plus frozen corrective-memory retrieval/intervention |
| E0–E1 | Contextual trained-policy contrast unless interfaces are fully matched |
| E1–E2 | Recovery contribution |
| E2–E3 | Additional memory contribution |
| E0–E3 | Total-system difference, not attribution to one pillar |

PC-01 epoch 6, seed 42 is provisional for the engineering pilot. No retraining
and no seeds 43–44. PC-02/PC-03 affect later validation-only final selection;
their current completion status must be checked at the lab, not assumed here.
Do not select models or tune thresholds from locked-test or WebArena outcomes.
Table 1 component evaluation and Table 3 related-paper comparison are separate.
The conversation contained invented example table numbers: they are NOT data,
targets, expected effects, or suitable content for reports.

## 3. Source state: update the lab correctly

At preparation time:

- Local branch: `Code`.
- Latest committed evaluation source:
  `178c24ac898f1feb17848ed1fa40fa22b97d053f`.
- GitHub `Code`, last checked with `git ls-remote`:
  `4d9b3df725b9075407749b135019b421e60bef2f`.
- Therefore GitHub had NOT yet received the latest local evaluation source.
- The progress checklist was modified locally; this handoff is a new document.
  Creating this document does not commit or push anything.
- Local CUDA experiments are outside the repository; they are not changes to
  the registered model/runtime implementation.

### Simple synchronization workflow

1. On the laptop, inspect and commit intended documentation changes and push
   the updated source when authorized. Verify the remote commit afterward.
2. At the lab, inspect `git status --short`, current branch, and running jobs
   before pulling. Do not overwrite lab changes, force-push, reset, or casually
   stash work. Do not update files that active jobs may import without checking.
3. In the existing lab repository on `Code`, use `git pull --ff-only origin Code`
   once clean/safe. If it refuses because of divergence, inspect the histories
   and ask before reconciling; do not invent a merge solution.
4. Record `git rev-parse HEAD` and verify that `178c24a` is an ancestor:
   `git merge-base --is-ancestor 178c24ac898f1feb17848ed1fa40fa22b97d053f HEAD`.

No separate evaluation folder is mandatory. Existing checkpoints need not be
copied just because source is updated. Resolve actual lab paths instead of
assuming the laptop's `/home/kiyas-mahmud/...` paths exist there.

### Historical receipt source identity is a separate concern

The completed Kaggle preparation was produced at exact commit `178c24a`.
Its validator and joint-audit pipeline enforce that producing commit and a
clean checkout. A later documentation commit does not change those historical
bytes, but may fail an exact-HEAD check. Do not edit receipts, bypass checks, or
automatically ask for a Kaggle rerun. The staged joint job creates its own
temporary clean source from the existing bundle for this reason. That temporary
execution directory is not a requirement to reorganize the lab repository.
Any later executable source change needs explicit review of downstream bindings.

## 4. What is completed, and what the tests mean

- Table 2 suite previously passed: **1,541 tests**, 2026-09-06.
- Full repository suite previously reported **1,662 passed, four failures**,
  2026-09-07. These are historical runs, not tests rerun for this handoff.
- Four recorded unrelated failures: notebook smoke-stage expectations in
  `tests/test_bbox_v2_3.py` and `tests/test_bbox_v2_5.py`; historical notebook
  outputs in `tests/test_gold_resume.py`; legacy constructor bypass in
  `tests/test_shapes.py`. Investigate any changed/new failure; do not blanket
  ignore failures based on this summary.
- Synthetic success, failure/memory, and 15×4 recovery smokes completed.
  They are `ENGINEERING_SMOKE_ONLY`, not 60 real recovery pilot episodes.
- Source commit `178c24a` contains the reviewed active page-state runtime work.
- Kaggle train-only preparation completed and its downloaded output validated.
- Exact active 50-task export and interface audit passed with example domains.
  This is structural compatibility, NOT a working website deployment.
- Docker works on the laptop; only a stopped hello-world container was found.
  No WebArena website services have been installed there.
- Local CUDA basic FP16 calculation passed, but actual model construction OOM'd.

## 5. Completed Kaggle result: preserve and reuse

Original download (do not rename or overwrite):
`/home/kiyas-mahmud/Downloads/results-178c24a.zip`.

Validated compact extraction on the laptop:
`/home/kiyas-mahmud/Thesis/table2-inputs/kaggle-p4-output-178c24a-9d9bf547/table2-p4-prepare-only-v1/`.

Eight required files: outer `execution_receipt.json` and
`execution_receipt.sha256`; inner `preparation/candidate_audit.json`,
`preparation/preparation_manifest.json`, its `.sha256`, `read_ledger.json`,
`review_queue.jsonl`, and `source_authority.json`. Keep this hierarchy intact.

- Package status: `REVIEW_REQUIRED`; downloaded-package validation: `PASS`.
- Receipt core SHA-256:
  `9d9bf5474b7b0eab44cbd73191f66c6ebc3f07e41928202937c1c9fead82b013`.
- Receipt file SHA-256:
  `e3e4805078173d6623ca81cecce519f3c59cfbd77b81ecbe3b26d48c08a75edc`.
- Source commit: `178c24ac898f1feb17848ed1fa40fa22b97d053f`.
- 24,107 training rows: 23,499 original and 608 supplement.
- 2,065 candidates: 1,888 original pending label review, 177 supplement
  source-approved labels. Source-approved is not independent success evidence.
- Zero fatal errors; application-recorded zero validation/test/locked reads.
  This is not an operating-system access attestation.
- **Zero eligible memory items claimed. No finished memory store exists.**

Do not rerun prepare-only merely because the files moved to the lab.
The next Kaggle job is duplicate assignment and raw-state replay, not training
and not repeating preparation. Gold data/images remain on Kaggle.

## 6. USB transfer inventory

| Item on laptop | Purpose / destination treatment |
| --- | --- |
| `Downloads/results-178c24a.zip` | Preserve raw Kaggle output; extract safely outside Git |
| `Thesis/table2-inputs/pc01-epoch6-seed42-full-v3/` | Required 11-file runtime/config/processor export; preserve all bytes |
| `Thesis/table2-inputs/table2-joint-job-178c24a.tar.gz` | Staged next-job launcher, pinned bundle and preparation files; not an executed audit |
| `Thesis/table2-inputs/webarena-active50-178c24a/` | Optional structural-reference export/audit; EXAMPLE URLs, never live-ready |
| `Thesis/table2-inputs/local-pc01-load-diagnostic-result.json` | Optional record of laptop OOM; not DGX evidence |

All paths above are relative to `/home/kiyas-mahmud/` on the laptop. On the lab,
choose a generated-input folder outside Git and record its actual absolute path.
Check archive entries for traversal/symlinks before extraction; do not overwrite
old outputs. No cookies, tokens, raw Gold images, or locked tasks enter Git.
The lab already has model resources: inventory and reuse rather than download
or copy large files unnecessarily. Do not transfer the laptop virtualenv.

## 7. Model inputs and first lab test

Required exact checkpoint:
`best_e6_outcome-mcc0.624.ckpt`, 291,071,781 bytes, SHA-256
`9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a`.

Base: `Qwen/Qwen2-VL-2B-Instruct`, revision
`895c3a49bc3fa70a340399125c650a463535e71c`.
Use the full matching base/processor snapshot, not only the adapter checkpoint.
The v3 export manifest identity is
`63c01942cc653732c9e9e18639cb49bded09a82235fd4ea37fef3fad14c9fa2d`.
Also resolve PC-01 full `report.json` and `run_contract.json`.

The existing gate requires Python 3.12.3, PyTorch 2.13.0+cu130,
Transformers 4.57.6, aarch64, and NVIDIA GB10 with 130,662,936,576 bytes.
Inspect the already-working lab environment first. A mismatch is a setup issue
to explain, not permission to change registered requirements or upgrade active
training packages.

Read `docs/PC01_CHECKPOINT_COMPATIBILITY_RUNBOOK.md` completely. Then run this
template after replacing every `/ABS/...` placeholder with a verified lab path:

```bash
# Run inside the clean lab repository, using its existing compatible environment.
PYTHONPATH=src python scripts/run_pc01_checkpoint_compatibility.py \
  --export-dir /ABS/pc01-epoch6-seed42-full-v3 \
  --checkpoint /ABS/best_e6_outcome-mcc0.624.ckpt \
  --base-snapshot /ABS/pinned-qwen-snapshot \
  --report /ABS/report.json \
  --run-contract /ABS/run_contract.json \
  --repository-root "$PWD" \
  --expected-source-commit "$(git rev-parse HEAD)" \
  --output /ABS/new-output/pc01_checkpoint_compatibility.json
```

The output must not already exist. Use the same source identity for downstream
bindings; do not reuse a receipt under another source commit. This gate uses
deterministic synthetic observations, not Gold/locked/WebArena task data.
It verifies direct vs runtime pre-action, diagnosis, recovery assessment,
P4 `[1,768]` representation, E0 adaptation absence, and unchanged model state.
Return its complete output/logs to the local control plane. PASS is compatibility
only, not model promotion or pilot completion.

## 8. What actually failed on the laptop

MX450: 2 GB physical VRAM; local CUDA diagnostic environment used Python 3.14.4
and PyTorch 2.13.0+cu126, not the registered DGX stack. CUDA and an FP16 matrix
calculation passed. The real checkpoint hash matched and deserialization passed.
The unchanged `WebAgentModel` constructor then failed inside the existing PEFT
QLoRA preparation, requesting another 892 MiB with only 79.31 MiB free.
Trained checkpoint state was not attached to a complete model, and inference
never executed. No CPU offload, new quantization, reduced image resolution,
training, or model-code edits were applied. Process ended and GPU memory freed.
Do not describe this as bad accuracy or a checkpoint corruption failure.

## 9. Remaining WebArena work — do not assume implementation is finished

Active registry: `benchmarks/table2/pilot/task_manifest_page_state_v2.json`.
The old indices 0–49 are NOT the active pilot. Both the historical 0–49 and
the active 50 remain excluded from later final evaluation.

Active 50 IDs:
102, 156, 157, 158, 159, 238, 258, 260, 269, 274, 283, 284, 298, 324, 356,
369, 370, 371, 372, 373, 374, 375, 377, 378, 379, 380, 381, 676, 677, 678,
679, 680, 704, 705, 706, 707, 708, 709, 710, 711, 712, 757, 758, 761, 762,
763, 764, 765, 766, 767.

Site distribution: Map 20, Shopping Admin 17, Shopping 10, GitLab 3.
The current preflight additionally requires Reddit, Wikipedia, and Homepage
origins. Do not silently drop map tasks or weaken the preflight.
The official full self-hosted map backend describes roughly 180 GB of downloads;
that is NOT total WebArena disk usage. No such download has been started here.
Map hosting remains unresolved; inspect lab capacity and compatible service
sources before large downloads or deployment. Do not assume a public backend
is available or equivalent. Restrict website access to experiment hosts.

### Unfinished live integration explicitly documented in source runbooks

Read `docs/TABLE2_LIVE_MATCHED_READINESS_RUNBOOK.md` and inspect current source
before promising execution. Its current documented gaps include:

- Concrete live BrowserGym plus sealed-evaluator factory and complete source closure.
- Concrete live calibration harness and approved safe probe-action inputs.
  Correction to older runbook wording: `process_broker_timeout_collector.py`
  and `scripts/collect_table2_process_broker_timeout.py` already exist. They
  require frozen inputs and do not authorize a campaign themselves.
- Actual operation timing measurements and required reset/non-persistence checks.
- Remaining campaign persistence/cross-binding for calibration evidence.
- Live process-isolation/value-origin evidence; synthetic broker tests are not it.

These are implementation/integration tasks, not merely missing downloads.
Prior test counts do not prove these paths have run. Do not fabricate evidence,
replace measurements with configured timeout values, or bypass startup checks.

## 10. Remaining P4 work

1. Resolve actual deployment URLs and export the registered 50 task definitions.
2. Run and replay joint Gold/WebArena duplicate assignments on Kaggle, using the
   completed preparation and original training images. The staged shell launcher
   deliberately rejects example-domain exports and has only local guard tests.
3. Review pending labels and independently establish executed recovery success
   and final task success. Reject unsupported rows; never manufacture evidence.
4. Finalize the provenance and 50+15 duplicate-audit closure.
5. Build embeddings with the exact checkpoint and processor where source images
   exist; freeze L2-normalized 768-dimensional vectors, cosine top-3 retrieval,
   deterministic ID tie-breaking, and train-only leave-one-out threshold.
6. Validate and transfer only compact frozen-store/evidence outputs to the
   campaign host. E2 performs zero retrieval; E3 performs zero memory writes.

Threshold calibration may itself fail if the admitted population cannot support
the registered procedure. Report this; do not tune on WebArena or relax eligibility.
See `docs/P4_MEMORY_PREPARATION_AND_TRANSFER_RUNBOOK.md` for exact commands.

## 11. Pilot execution after prerequisites pass

- First pass the separate live matched E0–E3 readiness block.
- Then 15 controlled recovery scenarios × four systems = 60 episodes.
- Then 50 ordinary tasks × four systems = 200 episodes.
- Seed 42 only; one registered repeat; at most one whole-block infrastructure rerun.
- 30 executor requests, two recovery attempts per incident, four per episode,
  102 logical model calls, Recovery@K with K=2, 600-second task budget.
- Invalid executor requests consume steps; ABORT consumes a recovery attempt
  but no browser executor step and never counts as success.
- No manual rescue, outcome-based task selection, or partial-system reruns.
- Count verified failures from ordinary episodes too: the controlled 60 are not
  the entire P1 evidence sample.
- Run the blinded 20-episode audit, validate raw packages, and reproduce metrics
  with denominators, uncertainty, and paired analyses from frozen logs.
- Draft pilot outputs remain `DRAFT_PILOT_ONLY` until required audit passes;
  report `PILOT_ONLY`, while final paper Table 2 remains `N/R`.
- Final promotion waits for registered validation-only candidate comparison.
  Report that one checkpoint seed does not measure model-seed uncertainty.

## 12. Lab agent's ordered checklist

- [ ] Verify updated source actually reached GitHub; safely pull into existing lab repo.
- [ ] Inventory active jobs and preserve local work/environment.
- [ ] Locate USB files, checkpoint, report, contract, and pinned base snapshot.
- [ ] Check environment identity without unnecessary installation.
- [ ] Run real PC-01 compatibility gate; return PASS or the exact failed stage.
- [ ] Inspect and complete remaining live adapter/calibration implementation.
- [ ] Establish WebArena services, including a valid map backend.
- [ ] Resolve active tasks; execute joint audit and complete P4 eligibility/store.
- [ ] Freeze/revalidate consistent source, model, environment, and memory bindings.
- [ ] Pass live matched readiness; run 260 episodes; audit and summarize.

Update `docs/TABLE2_REMAINING_EXECUTION_CHECKLIST.md` after each genuinely
completed item. Separate "implemented", "synthetically tested", "live tested",
and "blocked" in updates. If a required external action is missing, ask for the
specific action, not a vague request for all evidence or full DGX access.
Do not commit/push unrelated changes or claim work completed on another host
without actual output. Start with the compatibility test, not a training rerun.

## 13. Evaluation folder and file guide

Paths in this section are relative to the repository root so they work at the
lab's existing location. This maps implemented responsibilities, not a claim
that every module has passed a live deployment test. Read executable source
when older narrative documents conflict with the files below.

### A. Scientific configuration — important before every run

| Folder/file | What was implemented / why it matters |
| --- | --- |
| `configs/eval/table2/protocol.yaml` | Shared pilot budgets and mechanism controls; do not tune per system |
| `configs/eval/table2/protocol_final.yaml` | Separate final-campaign boundary; pilot is not final selection |
| `configs/eval/table2/pilot_webarena.yaml` | Active manifests, E0–E3, seed 42, and 260-episode counts |
| `configs/eval/table2/systems/e0.yaml` through `e3.yaml` | Registered feature switches; differences must remain interpretable |
| `configs/eval/table2/prompts/e0_action_v1.txt`, `parameter_provider_v1.txt` | Frozen base-agent and shared parameter-provider prompts |
| `configs/eval/table2/p4_source_authority_v1.json` | Exact allowed training-source identity, not arbitrary data paths |
| `configs/eval/table2/joint_duplicate_audit_v1.json` | Frozen normalization and exact/near duplicate rules |
| `configs/eval/table2/kaggle_p4_prepare_only_v1.json` | Registered preparation wrapper configuration |
| `configs/eval/table2/companion_diagnostics_v1.json`, `paper_claim_registry_v1.json` | Companion diagnostic definitions and permitted paper-claim boundaries |
| `configs/eval/table2/webarena_url_map.example.json`, `webarena_task_url_map.example.json` | Examples only: service origins and task placeholder replacements are distinct inputs |

### B. Benchmark registries — use the active v2 files

All files below are in `benchmarks/table2/pilot/`.

| File | Responsibility |
| --- | --- |
| `task_manifest_page_state_v2.json` | Exact active 50-task registry |
| `webarena_source_authority_page_state_v2.json` | Pinned upstream task-source authority |
| `recovery_scenarios_page_state_v2.json` | Active 15 controlled recovery scenarios |
| `duplicate_audit_manifest_page_state_v2.json` | Registered active duplicate-audit structure; not a completed audit |
| `final_exclusion_registry_v2.json` | Prevent reuse of development-exposed tasks in final evaluation |
| `recovery_oracle_rules.json` | Sealed recovery scoring rules; not policy inputs |
| `audit_manifest.json` | Blinded manual audit registration |
| Unsuffixed historical task/source/scenario/audit manifests | Retained history; do not accidentally substitute for active v2 inputs |

### C. Model and runtime — actual decisions and execution

| Folder/file | Implemented role and important boundary |
| --- | --- |
| `src/web_agent/models/model.py` | Shared trained model, heads, causal routing and P4 representation interface; do not redesign for evaluation |
| `src/web_agent/models/encoders/vlm.py` | Pinned backbone/QLoRA loading; laptop OOM occurred here, not in the benchmark |
| `src/web_agent/runtime/qwen2vl_pc01.py` | Real PC-01 loading and callbacks, trained-state restoration and unadapted E0; critical first lab target |
| `src/web_agent/runtime/checkpoint_inference.py`, `pc01_live_integration.py` | Selected-model/runtime binding and integration interfaces |
| `src/web_agent/runtime/contracts.py`, `protocol.py` | Serializable contracts and allowed system configuration |
| `src/web_agent/runtime/observation.py`, `policy.py`, `decision.py` | Observable inputs, policy calls and oracle-independent decision combination |
| `src/web_agent/runtime/action_parameters.py`, `executor.py` | Concrete parameter resolution and executor accounting; invalid outputs are not silently repaired |
| `src/web_agent/runtime/recovery/controller.py`, `strategies.py` | Bounded recovery control and concrete strategy actions |
| `src/web_agent/runtime/memory_adapter.py`, `duplicate_audit.py` | Runtime retrieval/intervention bridge and duplicate bindings; no evaluation writes |
| `src/web_agent/runtime/episode.py` | Episode state machine and normal/recovery sequencing |
| `src/web_agent/runtime/deadline.py`, `model_calls.py` | Wall-clock and logical-model-call budget enforcement |
| `src/web_agent/runtime/event_log.py`, `manifest.py`, `state_reset.py` | Evidence records, identity and reset boundaries |

### D. Browser adapters — environment interaction, not learned decisions

Files are under `src/web_agent/benchmarks/`.

- `base.py`, `registry.py`: adapter interface and lookup.
- `webarena.py`: registered WebArena adapter behavior.
- `browsergym_webarena.py`: BrowserGym observation/action integration.
- `fixture.py`, `recovery_fixture.py`: deterministic synthetic environments;
  never report their episodes as live WebArena results.

### E. P4 memory — preparation is separate from admission and construction

Files are under `src/web_agent/memory/`.

| File | Implemented role |
| --- | --- |
| `preparation.py` | Candidate queue and preparation package; does not certify success |
| `kaggle_prepare_only.py`, `kaggle_transport_staging.py` | Registered Kaggle execution/receipt and source transport staging |
| `joint_duplicate_audit.py` | Joint candidate/task assignments, raw replay and final audit binding |
| `verification.py`, `eligibility.py` | Evidence/provenance checks and leakage exclusions |
| `calibration_builder.py` | Train-only calibration construction |
| `builder.py`, `build_pipeline.py` | Store building and prerequisite orchestration |
| `frozen_store.py`, `manifest.py` | Read-only store access, identity and transfer closure |
| `index.py` | Existing low-level search component; not a competing second memory system |

### F. Table 2 evaluation and integrity modules

All filenames in this table are under `src/web_agent/eval/table2/`.

| File/group | Implemented role / importance |
| --- | --- |
| `pc01_artifacts.py`, `resolved_config.py`, `pc01_processor_parity.py` | Checkpoint/config/processor identity; prevents reconstructing config from newer YAML |
| `pc01_checkpoint_compatibility.py` | Registered real DGX checkpoint gate; first required lab experiment |
| `selection_evidence.py`, `model_compatibility.py` | Validation-selected model and compatibility bindings |
| `webarena_export.py`, `public_task_registry.py`, `pilot_task_exclusion.py`, `task_interface_audit.py` | Source-bound export, active task checks, permanent exclusions and six-action interface audit |
| `webarena_preflight.py`, `webarena_preflight_binding.py`, `split_deployment_preflight.py`, `locked_mount_preflight.py` | Host/services/browser/source boundary checks; examples do not pass as live services |
| `sealed_verifier.py`, `webarena_page_state_evaluator.py`, `outcome_semantics.py` | Sealed scoring and consistent outcome interpretation |
| `sealed_page_broker.py`, `live_page_broker_assembly.py` | One-way contracts / same-process fixture assembly; not sufficient production separation |
| `process_broker.py`, `process_broker_protocol.py`, `process_broker_worker.py`, `process_broker_runtime.py` | Process transport, typed messages, child worker and runtime boundary |
| `process_broker_webarena_backend.py` | Generic source-bound adapter factory bridge; concrete live deployment closure remains pending |
| `process_broker_episode_factory.py`, `process_broker_finalization.py` | Episode assembly and broker finalization wiring |
| `process_broker_fixture_backend.py` | Synthetic broker test backend only |
| `process_broker_timeout.py`, `process_broker_timeout_collector.py` | Timing contracts and real-clock collection infrastructure; still needs live harness and approved inputs |
| `live_deployment.py`, `live_deployment_validation.py`, `state_isolation_validation.py` | Deployment and isolation evidence validation |
| `deployment_authority.py`, `deployment_authority_registry_v1.json` | Authority validation; do not invent or populate final-campaign authority claims |
| `handoff.py`, `handoff_authority.py`, `dependency_lock.py` | Input assembly, source authority and dependency identity |
| `production_runner.py`, `execution_guard.py`, `live_compatibility.py` | Guarded production execution and matched live-readiness gate |
| `schedule.py`, `campaign.py` | Matched E0–E3 blocks, frozen campaigns and rerun discipline |
| `metrics.py`, `retrieval_metrics.py`, `statistics.py`, `summary.py` | Outcome/retrieval metrics, paired uncertainty and summary generation |
| `package_validator.py`, `evidence_validation.py`, `paper_claims.py` | Artifact closure/replay and limits on publication claims |
| `pillar1_diagnostics.py` through `pillar4_diagnostics.py`, `companion_diagnostics.py` | Pillar-specific companion diagnostics, separate from live task success |
| `pc01_companion_diagnostics.py`, `pc01_pillar2_diagnostics.py` | PC-01 checkpoint-backed diagnostic adapters |
| `common.py` | Shared canonical serialization, hashing and schema errors |

### G. CLI entrypoints — invoke these rather than duplicating business logic

All are under `scripts/`; use `--help` and the corresponding runbook to resolve
real lab paths before execution.

| Stage | Entrypoints |
| --- | --- |
| PC-01 artifacts and parity | `export_pc01_table2_artifacts.py`, `audit_pc01_training_action_values.py`, `verify_pc01_processor_parity.py` |
| First real lab gate | `run_pc01_checkpoint_compatibility.py` |
| Completed preparation workflow | `stage_table2_p4_kaggle_transport.py`, `run_table2_p4_kaggle_prepare_only.py`, `validate_table2_p4_kaggle_output.py`, `prepare_table2_p4.py` |
| Next memory work | `run_table2_joint_duplicate_audit.py`, `build_table2_memory.py` |
| Task/environment preparation | `export_table2_webarena_tasks.py`, `audit_table2_webarena_task_interface.py`, `preflight_table2_webarena.py`, `prepare_table2_split_preflight.py` |
| Live boundary/timing | `validate_table2_live_deployment.py`, `validate_table2_deployment_authority.py`, `collect_table2_process_broker_timeout.py` |
| Freeze and execute | `build_table2_dependency_lock.py`, `prepare_table2_handoff.py`, `freeze_table2_campaign.py`, `run_table2_evaluation.py` |
| Synthetic smoke only | `run_table2_smoke.py` |
| Analyze and check | `validate_table2_artifacts.py`, `summarize_table2.py`, `validate_table2_paper_claims.py` |
| Companion diagnostics | `run_table2_pillar1_diagnostics.py` through `run_table2_pillar4_diagnostics.py`, `table2_companion_diagnostic_bootstrap.py` |

Do not run `run_gold.py`, `run_model.py`, or comparison-training scripts as a
substitute for the inference gate. Their presence does not authorize retraining.

### H. Tests and documents to read with changes

`tests/table2/` contains the synthetic/unit and contract tests. Critical groups:

- `test_pc01_checkpoint_compatibility.py`, `test_pc01_qwen_runtime.py`,
  `test_pc01_processor_parity.py`, `test_pc01_live_integration.py`: model bindings.
- `test_temporal_isolation.py`, `test_oracle_blindness.py`,
  `test_callback_isolation.py`, `test_system_switches.py`: scientific barriers.
- `test_joint_duplicate_audit.py`, `test_memory_build_prerequisites.py`,
  `test_memory_replay_evidence.py`, `test_calibration_builder.py`: P4 eligibility.
- `test_process_broker_*.py`, `test_process_isolated_broker.py`,
  `test_live_deployment.py`: process/timing contracts, not proof of a live service.
- `test_episode.py`, `test_recovery.py`, `test_deadline.py`, `test_model_calls.py`:
  execution accounting and recovery budgets.
- `test_metrics.py`, `test_statistics.py`, `test_aggregate_recomputation.py`,
  `test_package_validator*.py`, `test_pilot_publication_gate.py`: reproducibility.

Read together: `docs/TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md`,
`docs/TABLE2_REMAINING_EXECUTION_CHECKLIST.md`,
`docs/PC01_CHECKPOINT_COMPATIBILITY_RUNBOOK.md`,
`docs/P4_MEMORY_PREPARATION_AND_TRANSFER_RUNBOOK.md`, and
`docs/TABLE2_LIVE_MATCHED_READINESS_RUNBOOK.md`.
The collector absence claim in older live-readiness prose is stale; use the
corrected distinction in Section 9 of this handoff.

### I. Generated outputs and local-only files

- `artifacts/table2/<campaign_id>/`: generated raw paired-block evidence,
  screenshots, logs, frozen inputs, hashes and audits; never push raw artifacts.
- `results/table2/<campaign_id>/`: generated summaries; only explicitly vetted
  redacted outputs may be published. Do not overwrite component comparison data.
- Existing `webagent_comparison/outputs/model_comparison/...`: historical
  training resources; preserve their checkpoint/report identities.
- `/home/kiyas-mahmud/Thesis/table2-inputs/`: laptop external generated inputs,
  transfer packages, exports and diagnostics; Git push does NOT transfer these.
- `/home/kiyas-mahmud/Thesis/table2-envs/`: laptop environments/browser assets;
  not a portable lab environment and not part of the Git handoff.

For the lab agent, the highest-priority files are the compatibility runbook and
gate, `qwen2vl_pc01.py`, active pilot configuration/registries, memory runbook,
and the live-deployment integration modules. Preserve their scientific checks;
complete the missing real integrations rather than disabling failed guards.
