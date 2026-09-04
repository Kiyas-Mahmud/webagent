# Table 2 End-to-End Runtime and Post-Training Implementation Plan

**Project:** Four-pillar failure-aware multimodal web agent

**Document purpose:** Implement and evaluate the corrected Table 2 experiment

**Plan date:** 2026-08-30

**Plan status:** Core offline runtime/evidence architecture and the task/source/
deployment-preflight gates are implemented and the complete Table 2 suite
passes 965 tests. The pinned WebArena 0--49 source audit currently blocks
handoff because 47 tasks require an unsupported assistant-answer/STOP
interface. Strict live-capability, provider-installation, evaluator, memory,
and artifact-package binding is implemented across handoff, frozen campaign,
validation, and production launch. Its genuine measured external evidence and
the other pilot inputs are still unavailable. This document does not claim any
live browser-evaluation result. A causal-boundary audit also confirmed that the
current page broker is same-process reviewed-code engineering machinery, not an
enforced isolation boundary: both typed capabilities remain importable in one
interpreter. The frozen runner records that limitation and production fails
before provider import. No live campaign may run until a separately
authenticated process-isolated broker implementation and receipt are added
under a new registered schema.

**Scope:** Work that can proceed while DGX training runs, followed by the exact post-training evaluation sequence

---

## 0. Decision and authority

The next experimental priority is **Table 2: end-to-end agent performance**.
Component training can continue on the varsity-lab DGX Spark while the
runtime, evaluator, logging, and paired E0-E3 protocol are implemented and
tested on development tasks.

This document governs the Table 2 implementation. It refines and supersedes
conflicting Table 2 wording in older documents, but it does not change:

- the registered Gold v2.8 training configuration;
- validation-only model/checkpoint selection;
- the locked-test boundary;
- any current DGX run;
- the scientific roles of the four pillars.

The broader supporting documents remain:

- `docs/Q1_END_TO_END_IMPLEMENTATION_PLAN.md`;
- `docs/Q1_FINAL_EVALUATION_AND_COMPARISON_PLAN.md`;
- `docs/DGX_THREE_MODEL_COMPARISON.md`;
- `docs/DGX_FULL_TRAINING_RUNBOOK.md`;
- `project_progress.md`.

Where old files still describe the legacy 70,965-row synthetic corpus, a
single shared before-state embedding, the old model order, or the old
Failure-F1 selection rule, the current reviewed-Gold v2.8 causal protocol and
validation-gated DGX comparison take precedence.

### 0.1 Evidence boundary

Offline component performance does not prove that an agent can complete a
browser task. Table 2 is fillable only from executed, fully logged browser
episodes.

During training, development may use:

- synthetic/local browser fixtures;
- a registered development-task subset;
- an existing mini or provisional checkpoint;
- training-only memory items.

During development, the following are forbidden:

- opening the locked Web-Gold test to tune runtime behaviour;
- using final scored benchmark tasks to tune prompts, thresholds, budgets, or
  recovery rules;
- writing validation/test episodes into memory;
- presenting the current partial validation CSV as a final Table 2 result.

The active `configs/eval/table2/protocol.yaml` is identified as
`table2-pc01-pilot-v1` and uses the `pc01_provisional` selection mode. That
mode binds the campaign to the PC-01 candidate, forbids locked-task
eligibility, and can emit only pilot evidence while the paper table remains
`N/R`. Changing only the protocol status or evidence label cannot promote it.
The separate `configs/eval/table2/protocol_final.yaml` uses the
`three_candidate_final` mode and is valid only with the exact three registered
candidates. It is deliberately a non-runnable `FINAL_TEMPLATE_ONLY` profile
with status `AWAITING_MODEL_PROMOTION` and `locked_task_eligible: false`; it
cannot authorize a final campaign or locked-task access. A new materialized
final protocol may be frozen only after PC-02/PC-03 finish, the validation-only
comparison selects one winner, and final task eligibility is resolved. The
template's task-manifest locator points to the external
`locked_benchmark_mount/table2/final/task_manifest.json`; it never reuses the
tracked 0--49 pilot registry. That external manifest is created only after the
registered final eligibility process and remains outside Git.

Table 1's locked Web-Gold component evaluation remains a separate batched
component experiment. Gold training data may supply strictly eligible P4
memories, but it does not supply the WebArena browser episodes in Table 2.
Table 3 remains the related-paper comparison/discussion; values from the
strongly related papers are context only and are never copied into Table 2.

### 0.2 Current implementation boundary

The additive runtime, evaluator, evidence package, deterministic browser
fixtures, and 15-scenario recovery harness are implemented. Their checks are
engineering verification only. Before the registered 260-episode PC-01
engineering pilot may start, the authenticated handoff, campaign-freeze, and
post-freeze live-readiness sequence must establish all of the following:

- the authenticated PC-01 epoch-6 checkpoint and checkpoint-exported
  resolved-config artifact. The checkpoint SHA-256 is
  `9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a`,
  and the canonical checkpoint-saved config SHA-256 is
  `d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f`;
- the completed schema-v3 identity/preprocessing export. It authenticates all
  14 files in the pinned base snapshot at revision
  `895c3a49bc3fa70a340399125c650a463535e71c`, binds the checked training
  environment and preprocessing sources, contains the train-only action-value
  omission audit, and cross-binds that audit to the deterministic processor-
  parity receipt. The snapshot directory-payload SHA-256 is
  `e002f8290faa3e9f44bf3099eac85a2445e17de738c5bb0cc10d342da837c46c`;
- the complete seed-42 report, run contract, and compatibility report already
  present for PC-01, revalidated as part of the eventual runtime bundle. The
  historical compatibility report is fixed at SHA-256
  `1a8e9bb008daab5dfe893db7738daf12ee6e57e1ce2469ae29810d793030a441`;
- the frozen unadapted version of the selected backbone for E0;
- a real checkpoint-backed CUDA inference-parity receipt and all mandatory
  live operational bindings. The processor-only export intentionally records
  `runtime_ready: false` and cannot satisfy this gate;
- a complete train-only, per-seed memory store and its train-only threshold-
  calibration evidence;
- a genuine joint train/WebArena duplicate audit marking all 50 development
  tasks `VERIFIED` with non-empty cluster evidence;
- a measured, version-pinned WebArena/browser environment,
  credentials/services, frozen reset and environment-digest callbacks, and an
  upstream task-index compatibility check;
- a sealed evaluator whose provenance distinguishes the pinned official
  WebArena upstream source from the runtime `libwebarena==0.0.4` compatibility
  port. The port is not described as byte-identical or "the official
  evaluator": it requires pinned upstream and runtime hashes, a compatibility
  delta, independent review evidence, frozen evaluator configuration, and a
  successful availability/schema receipt for the registered judge model;
- a version-pinned WebArena observation mapper that derives
  `observable_select_controls` and `recovery_target_evidence` from the current
  oracle-blind page observation. The generic adapter accepts this integration
  as a frozen callback; the campaign cannot substitute fixture evidence or
  start without a live mapper that has passed its schema/canary smoke;
- the complete resolved 50-task export and the preregistered blinded-audit
  sampling/adjudication protocol. Human audit labels are created only after
  episode execution and enter only the sealed evaluation path;
- a separate frozen readiness-probe campaign that completes the exact first
  normal WebArena block across matched E0--E3. Its target-local, portable
  `runtime_readiness/` receipt must replay successfully before every pilot
  block dispatch. This gate has not run and is non-scored readiness evidence,
  not one of the 260 pilot episodes or a Table 2 result.

The final template remains non-runnable until the complete seed-42
reports and run contracts for all three registered backbone candidates and the
independently replayable comparison JSON/CSV are available. Therefore the live
PC-01 pilot has an authenticated epoch-6 checkpoint and checkpoint-saved
resolved configuration plus an authenticated local base-model/processor
snapshot and exact processor-parity receipt. Its complete runnable model bundle
is still blocked on checkpoint-backed CUDA/model-forward parity and the live
operational callbacks. The exact pinned
WebArena 0--49 export has also exposed a material protocol incompatibility: 47
tasks use `string_match` and require an assistant answer submitted through a
STOP/termination interface that the fixed six-class P3 action contract does
not provide; only indices 44, 45, and 46 are page-state URL tasks compatible
with that contract. Thirteen of the 47 answer tasks additionally depend on the
registered model-based fuzzy judge. The handoff recomputes this audit and
fails closed. It must not silently remap evaluators, drop tasks, or invent a
seventh action. A live pilot therefore requires one explicit, user-approved
and preregistered resolution: either replace the pilot registry with 50 public
page-state-compatible development tasks, or add a scientifically justified
answer/termination interface and update the P3/protocol contract before
outcomes are observed. The exact unapproved replacement and consequences are
recorded in `docs/TABLE2_PILOT_TASK_INTERFACE_DECISION.md`.

The pilot is also blocked on measured WebArena services, sealed-evaluator
compatibility evidence, reset callbacks, the genuine joint train/WebArena
duplicate audit, secrets/credentials, the frozen P4 store, the complete live
capability manifest, and the preregistered audit protocol. Until every input
exists and the task-interface gate passes, evaluation stops before browser
reset and the paper Table 2 remains `N/R`.

### 0.3 Isolated browser-stack diagnostic

The optional Table 2 dependencies are declared in `pyproject.toml` and remain
separate from the active DGX training environment. An isolated local x86
diagnostic environment was created with BrowserGym core/WebArena 0.14.3,
`libwebarena==0.0.4`, Gymnasium 1.0.0, Playwright 1.44.0, and Chromium
125.0.6422.26 (build 1117). Package checks and the 1280x720 Chromium smoke
passed. All seven WebArena services were absent, so service readiness failed,
live reset was not run, and that diagnostic host is not campaign-eligible.

On the eventual intended campaign host, install and freeze the same stack in a
dedicated environment rather than mutating the training environment:

```bash
python3 -m pip install -e '.[table2]'
python3 -m playwright install chromium
```

Record the resulting Python/package/browser versions in the dependency lock
before freezing the campaign. Installing the browser stack does not authorize
access to locked tasks or establish service, reset, credential, evaluator, or
campaign readiness.

---

## 1. Final research question

Table 2 must answer:

> On identical browser tasks, under identical environment and compute budgets,
> does explicit post-action failure diagnosis and executed recovery improve
> task completion, and does frozen train-only corrective memory provide an
> additional improvement?

The registered contrasts are:

- **E0 vs E1:** difference between the unadapted common-schema base-agent
  baseline and the trained project action/grounding policy;
- **E1 vs E2:** effect of the complete registered diagnosis-and-recovery stack;
- **E2 vs E3:** effect of corrective-memory retrieval and intervention.

E0 vs E3 is a total-system comparison. It is not evidence that any individual
pillar caused the complete difference. E0 vs E1 is contextual unless the base
and trained policies are shown to have an architecture- and interface-matched
control. E1 vs E2 does not isolate the learned P1 trigger from deterministic
executor-error and loop safeguards unless a preregistered trigger-source
ablation is also reported.

---

## 2. Four-pillar integrity contract

No runtime shortcut is allowed to weaken, merge, bypass, or relabel a pillar.

| Pillar | Runtime responsibility | Required evidence | Prohibited shortcut |
| --- | --- | --- | --- |
| **P1 Failure-aware resilience** | Detect the observed action outcome, diagnose failure type, decide whether recovery is needed, select a strategy, and evaluate the executed recovery | Failure diagnostics plus executed recovery and post-recovery verification | Calling an offline recovery prediction an executed recovery success |
| **P2 Multimodal decision** | Use screenshots with task/page state while preserving strict pre-action, post-action, and recovery-transition timing | Modality/sensitivity evidence and temporal-leakage tests | Feeding post-action or oracle information into pre-action selection |
| **P3 Multi-tool action and grounding** | Select one of six action classes and ground its target; convert it to a concrete browser action | Executed-action validity, action metrics, target/coordinate evidence | Replacing a failed predicted target with an oracle target silently |
| **P4 Corrective memory** | Retrieve relevant successful recovery experience from an immutable train-only index and allow it to influence recovery decisions | Retrieval metrics, intervention logs, and paired E2-vs-E3 outcome evidence | Test-time writes, same-task retrieval, or claiming memory benefit from store-flag accuracy |

### 2.1 Complete causal chain

```text
task + current pre-action state
    -> P2 multimodal encoding
    -> P3 action class + target grounding
    -> common action-parameter provider
    -> concrete browser execution
    -> observed post-action state
    -> P2 action-conditioned transition encoding
    -> P1 outcome + diagnosis + needs-recovery + strategy
    -> P4 train-only retrieval when the registered gate permits it
    -> concrete recovery resolution and execution
    -> observed post-recovery state
    -> P1 recovery-outcome assessment + independent verifier
    -> continue, succeed, safely abort, time out, or exhaust budget
```

### 2.2 Temporal information barriers

| Stage | Permitted information | Outputs |
| --- | --- | --- |
| Pre-action | current screenshot, task, domain/site, current URL and permitted current-page state | action class, bbox/target, confidence |
| Execution | frozen provider output and controller state | concrete executed action and execution status |
| Post-action | pre-state, executed action type, observed post-state, task/domain; for PC-01, action-value text is omitted to match all 24,107 training rows | outcome, failure type, needs recovery, recovery strategy, memory-update prediction |
| Recovery | observed post-failure state, P1 diagnosis, concrete recovery, observed post-recovery state | predicted recovery outcome plus sealed progress verification |
| Memory query | current post-failure representation and permitted task/site metadata | filtered training-recovery neighbours and optional advice |

Hard prohibitions:

1. Pre-action inference must not receive `state_after`, executed action,
   ground-truth outcome, failure label, recovery label, success-oracle answer,
   or any future state.
2. The model prompt must not contain benchmark reference actions or answers.
3. The post-action stream may receive the action that was actually executed,
   because it is now observed history.
4. The task-success oracle must remain independent of policy prompts and
   recovery generation.
5. No validation/test trajectory or near duplicate may enter P4 memory.
6. The evaluator/oracle is a sealed scoring channel. Its failure, progress,
   reference, and success outputs must not enter the recovery trigger,
   strategy selector, memory query, or action-parameter provider.
7. P4 is post-action corrective memory only. Pre-action retrieval or advice is
   disabled in the primary E0-E3 experiment.

---

## 3. Corrected E0-E3 system definitions

| ID | Exact configuration | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- | --- |
| **E0 Base selected-backbone agent** | Unadapted base version of the validation-selected backbone under the common observation, action schema, provider, executor, and budget | Project controller disabled | Base multimodal input | Prompted/common-schema action generation | Disabled |
| **E1 Trained action agent** | Frozen selected project checkpoint using pre-action action and grounding only | Post-action diagnosis and recovery disabled | Pre-action multimodal stream | Trained action + bbox | Disabled |
| **E2 Recovery agent** | Same E1 checkpoint plus post-action diagnosis and concrete recovery | Enabled | Pre/post/recovery causal streams | Same action/grounding and provider | Retrieval disabled |
| **E3 Complete agent** | Same E2 system plus frozen train-only corrective-memory retrieval and registered intervention | Enabled | Same as E2 | Same as E2 | Enabled |

The PDF row called “Base + failure detection” is replaced by E1. Failure
detection without an executed recovery does not provide the required trained
action/grounding baseline and is not the required E1 configuration.

### 3.1 E0 policy contract

E0 must be instantiated from the unadapted base version of whichever backbone
wins the registered validation-only comparison. It must not be hard-coded as
Qwen because InternVL remains an eligible winner. E0 uses a fixed
prompt/parser that produces the common action schema. Its prompt may be
developed only on development tasks and must be frozen before scored
evaluation. E0 must use the same:

- browser controller;
- observation dimensions;
- action-parameter provider;
- target-coordinate convention;
- task instructions;
- action/time budget;
- success oracle.

If the base-agent framework has hidden retry, reflection, replanning, memory,
or automatic target substitution that cannot be disabled and logged, E0 must
be labelled a **contextual baseline**, not a controlled ablation.

Even with hidden behaviours disabled, E0 remains contextual if its prompted
policy form cannot be made architecture/interface-matched to E1. In that case,
the paper reports an unadapted-baseline difference and does not name the full
E0-vs-E1 delta a pure training effect.

### 3.2 E1-E3 checkpoint contract

E1, E2, and E3 use the same selected backbone and the same seed-specific
checkpoint. At every stage they share, they use the same prompt template,
provider/model version, decoding parameters, RNG schedule, thresholds, browser
settings, and budgets. Only the registered runtime feature switches may
differ, and only E3's admitted post-failure memory-context field may add
retrieved content to its otherwise identical recovery template.

### 3.3 Required feature-switch assertions

| Switch | E0 | E1 | E2 | E3 |
| --- | ---: | ---: | ---: | ---: |
| Project trained pre-action policy | 0 | 1 | 1 | 1 |
| Project post-action diagnosis | 0 | 0 | 1 | 1 |
| Project recovery controller | 0 | 0 | 1 | 1 |
| Corrective-memory query | 0 | 0 | 0 | 1 |
| Corrective-memory intervention | 0 | 0 | 0 | 1 |
| Evaluation-time memory write | 0 | 0 | 0 | 0 |

At startup, the runner must serialize the resolved switches and fail if the
configuration differs from this matrix.

System overlays name these six mechanisms directly. Pillar labels describe
the research architecture; they are not interchangeable with runtime feature
switches (for example, action-type prediction is part of P3 and is not a P2
toggle).

---

## 4. Shared fairness contract

All systems must receive the same experimental opportunity.

### 4.1 Shared inputs and environment

- identical task IDs and natural-language instructions;
- identical start-state/reset procedure;
- identical browser and controller version;
- identical viewport and device scale factor;
- identical screenshot timing and page-settle rules;
- identical available current-page metadata;
- identical credentials and consent state;
- identical task, action, recovery, timeout, and model-call budgets;
- identical evaluator and terminal-state rules.

For every ordinary WebArena episode, reset equality is proved twice. The
existing `EpisodeStateResetEvidence` resets and hashes the five shared
model/provider/recovery/embedding backends. A separate source-attested,
oracle-blind WebArena reset-state attester runs immediately after the external
environment reset and emits only SHA-256 commitments for the service,
account, database, and frozen start state. Its typed receipt contains no
success, failure, progress, reward, relevance, evaluator configuration, or
reference trajectory. Package validation requires one hashed receipt per
episode and identical commitments across the matched E0-E3 block; missing,
unattested, malformed, or unequal receipts fail closed.

The frozen page-settle policy is an executable runtime binding, not manifest
metadata alone. It runs after reset and after every Gym step, but before the
raw observation reaches the observation mapper. The binding is matched to the
frozen benchmark version, policy ID, network-idle flag, and timeout. It receives
no reward, done/truncated flag, `info`, or verifier output, must return the
causal settled observation, and fails the episode closed if it times out,
returns nothing, or the mapped observation does not attest `page_settled=true`.

The live action-safety policy is likewise a required frozen runtime binding,
not a descriptive manifest string. Its ID, version, destructive-action rule,
benchmark version, and attested integration source must match the environment
freeze for every E0-E3 episode. Before action mapping or a browser step, it
receives only a detached `TaskSpecification`, the current causal
`Observation`, and the proposed `ConcreteAction`; no oracle or future state is
available. The integration decides task-specific site/origin boundaries rather
than the evaluator guessing an allow-list. Inputs are hash-guarded against
mutation and the callback must return an action-bound typed allow/deny receipt
with a compact reason code. A denial occurs after the shared executor has
charged the request, but before the mapper or browser is called.

`manual_rescue: forbidden` is also machine-enforced. The protocol and campaign
freeze must both preserve that exact value, and the environment must register
a source-attested `exclusive_controller_input_audit_v1` guard. The guard emits
hash-chained, task/observation/action-bound receipts after reset, before and
after every adapter-level browser request, and before every terminal check.
Each receipt must attest exclusive automation control, the exact cumulative
registered browser-step count, zero non-agent input events, and no oracle-label
access. The adapter hashes live environment state before and after each guard
callback so the observer cannot itself alter the browser. A missing receipt,
broken chain, coverage gap, non-agent input, or guard mutation is a fatal
protocol violation and never authorizes an infrastructure rerun. Package
validation reconstructs the expected receipt sequence independently from the
action, observation, and terminal streams.

### 4.2 Shared action execution

- one action schema;
- one action-parameter provider;
- one coordinate conversion function;
- one element-resolution policy;
- one invalid-action policy;
- one safety/destructive-action policy;
- no manual rescue for any system.

### 4.3 Shared execution order

For controlled/self-hosted tasks, use identical snapshots and deterministic
resets. For live sites, randomize or counterbalance system order within each
task/repetition block so that one system is not consistently exposed to a
newer or easier page state.

Runtime randomness is also frozen at the serialized-key level. The
`sha256_stage_keyed_v1` payload has a namespace containing exactly
`protocol_id` and `campaign_seed`, followed by a `StageRNGKey` containing
exactly `schema_version`, `record_type`, `campaign_id`, `task_id`,
`repeat_id`, `matched_seed`, `stage`, `decision_index`, `incident_index`,
`attempt_index`, and `stream`. `system_id` is deliberately absent from the
namespace and key. Consequently, E0-E3 receive the same stage-specific random
stream for the same matched identity and decision context; a system label
cannot silently change the draw. Startup validation rejects any field,
ordering, namespace, or algorithm declaration that differs from this frozen
contract.

### 4.4 Fair-compute reporting

The primary comparison uses equal hard limits, but the paper also reports
actual consumption:

- low-level browser actions;
- recovery attempts;
- model calls;
- prompt/input and output tokens where available;
- decision latency;
- provider latency;
- retrieval latency;
- wall-clock task time.

No system may receive extra unreported retries after reaching its budget.

The runner owns a frozen cap of **102 logical model calls per episode**: up to
90 calls for 30 browser requests (pre-action policy, parameter fallback, and
post-action assessment) plus up to 12 calls for four recovery incidents
(planner, recovery assessment, and memory embedding). A dispatch is recorded
before backend invocation in an append-only ledger with episode ID, sequential
one-based call index, registered stage, component identity, and the common cap.
The backend receipt count and episode summary must equal that ledger; an
unregistered stage, missing dispatch, or attempt beyond 102 fails closed.

### 4.5 Matched paired-block denominator

The unit of pairing is a registered `task_id x repeat_id x model_seed` block.
For E0, the model-seed field identifies the matched inference/order schedule;
its unadapted weights are not retrained per seed. A block enters the primary
Table 2 analysis only when E0-E3 all completed under the same frozen protocol.

- Agent-caused invalid actions, ABORTs, timeouts, loops, and budget exhaustion
  remain failures; they are never removed to improve a system's denominator.
- A preregistered infrastructure-invalid block is rerun for every system under
  the same rule. If a clean rerun is impossible, the complete E0-E3 block is
  excluded from the primary paired analysis, never only the affected system.
- Freeze a maximum paired-block rerun count. Rerun ordinal `j` replaces ordinal
  `j-1` for the entire E0-E3 block; never choose each system's best attempt or
  combine systems from different ordinals.
- Scheduled, rerun, excluded, and final valid paired-block counts are reported
  by system, task, repetition, and seed.
- Unpaired descriptive results may be shown separately but cannot support the
  registered E0-vs-E1, E1-vs-E2, or E2-vs-E3 contrasts.

---

## 5. Implemented additive runtime architecture

Implementation must be additive. Existing training modules and active DGX
configs remain unchanged.

```text
src/web_agent/runtime/
    __init__.py
    contracts.py
    protocol.py
    observation.py
    policy.py
    action_parameters.py
    executor.py
    decision.py
    episode.py
    event_log.py
    manifest.py
    memory_adapter.py
    checkpoint_inference.py
    duplicate_audit.py
    deadline.py
    model_calls.py
    qwen2vl_pc01.py
    state_reset.py
    recovery/
        controller.py
        strategies.py

src/web_agent/benchmarks/
    base.py
    browsergym_webarena.py
    registry.py
    fixture.py
    recovery_fixture.py
    webarena.py

src/web_agent/memory/
    index.py
    eligibility.py
    builder.py
    build_pipeline.py
    calibration_builder.py
    frozen_store.py
    joint_duplicate_audit.py
    manifest.py
    preparation.py
    verification.py

src/web_agent/eval/table2/
    common.py
    sealed_verifier.py
    schedule.py
    metrics.py
    retrieval_metrics.py
    statistics.py
    campaign.py
    outcome_semantics.py
    execution_guard.py
    evidence_validation.py
    handoff.py
    handoff_authority.py
    live_compatibility.py
    live_deployment.py
    locked_mount_preflight.py
    package_validator.py
    pc01_artifacts.py
    pc01_checkpoint_compatibility.py
    pc01_processor_parity.py
    pilot_task_exclusion.py
    production_runner.py
    resolved_config.py
    sealed_page_broker.py
    selection_evidence.py
    split_deployment_preflight.py
    state_isolation_validation.py
    summary.py
    task_interface_audit.py
    model_compatibility.py
    webarena_export.py
    webarena_preflight.py
    webarena_preflight_binding.py

configs/eval/table2/
    protocol.yaml
    protocol_final.yaml
    pilot_webarena.yaml
    joint_duplicate_audit_v1.json
    p4_source_authority_v1.json
    prompts/
    systems/e0.yaml ... systems/e3.yaml

benchmarks/table2/pilot/
    task_manifest.json
    recovery_scenarios.json
    recovery_oracle_rules.json
    duplicate_audit_manifest.json
    audit_manifest.json
    webarena_source_authority.json

scripts/
    export_pc01_table2_artifacts.py
    audit_pc01_training_action_values.py
    verify_pc01_processor_parity.py
    run_pc01_checkpoint_compatibility.py
    export_table2_webarena_tasks.py
    audit_table2_webarena_task_interface.py
    preflight_table2_webarena.py
    prepare_table2_split_preflight.py
    prepare_table2_p4.py
    run_table2_joint_duplicate_audit.py
    build_table2_memory.py
    prepare_table2_handoff.py
    freeze_table2_campaign.py
    run_table2_smoke.py
    run_table2_evaluation.py
    validate_table2_artifacts.py
    summarize_table2.py

tests/table2/
    deterministic contract, causality, recovery, memory, metric,
    statistics, browser-adapter, and package-integrity tests
```

These tracked paths implement the engineering harness. Generated evidence is
written only below ignored `artifacts/table2/<campaign_id>/` and
`results/table2/<campaign_id>/` paths. The external
`locked_benchmark_mount/` is never committed or read by a pilot campaign.
Tests currently construct their synthetic/sanitized fixtures in code, so no
empty `tests/table2/fixtures/` placeholder is tracked. Generated Table 2 result
files remain ignored even after redaction; any result selected for a paper must
be separately reviewed and deliberately copied into a tracked paper-artifact
location rather than force-added from the generated-results tree.

### 5.1 Runtime contracts

Every contract must be versioned and serializable.

#### `TaskSpecification`

- protocol/task ID;
- benchmark and task version;
- natural-language goal;
- start-state identifier or start URL;
- site/domain;
- permitted credentials/profile;
- maximum browser actions;
- maximum recovery attempts;
- timeout;
- destructive-action permission;
- success-oracle ID/version;
- development or scored partition.

#### `Observation`

- episode, step, and observation IDs;
- adapter-owned, per-observation screenshot path, byte SHA-256, and dimensions;
- current URL and title;
- permitted accessibility/DOM summary, if the protocol allows it;
- timestamp and page-settle status;
- browser/viewport metadata;
- environment-error flags;
- prior executed-action ID for post-action observations.

#### `PreActionDecision`

- action class and class probabilities;
- normalized bbox and grounding confidence;
- confidence-before;
- source checkpoint/config hashes;
- inference latency;
- explicit list of input observation IDs.

#### `ConcreteAction`

- predicted action class;
- provider-resolved parameters;
- target coordinates/element evidence;
- provider status and latency;
- safety validation result;
- exact controller command;
- execution start/end timestamps and status.

#### `TransitionAssessment`

- pre- and post-observation IDs;
- executed-action ID;
- outcome probabilities and predicted outcome;
- failure-type probabilities and diagnosis;
- needs-recovery probability/decision;
- recovery-strategy probabilities/decision;
- recovery-trigger source, using only policy predictions, executor/browser
  status, and the frozen loop guard;
- memory-update prediction for component analysis only;
- confidence/calibration fields;
- inference latency.

#### `SealedVerificationResult` (evaluation package only)

- verifier/oracle version;
- progress status and evidence;
- task-success status and evidence;
- agent failure vs environment failure;
- loop/repeated-state flags;
- terminal reason;
- verifier latency.

This record is defined and stored only in `eval/table2/sealed_verifier.py`.
Runtime contracts contain only `OpaqueTerminalSignal`, which exposes an opaque
event/token hash and a stop/no-stop bit. Full success, progress, failure,
recovery, and relevance truth is forbidden from `runtime/` records.

#### `RecoveryAttempt` and `RecoveryAssessment`

- triggering transition and diagnosis;
- high-level recovery strategy;
- concrete recovery action IDs;
- attempt index and recovery budget state;
- post-recovery observation;
- predicted recovery outcome;
- predicted recovery resolution/progress from the executed recovery
  transition.

Independent verified recovery truth is attached later inside the sealed
evaluation package; it is not stored in either runtime contract.

#### `MemoryItem`

- immutable memory ID;
- training dataset/version/split IDs;
- source task/trajectory/episode/step IDs;
- pre-failure, post-failure, and recovered-state hashes/embeddings;
- failed action and target/tool class;
- verified failure type;
- executed recovery strategy and concrete action summary;
- verified local recovery outcome;
- final task success;
- canonical SHA-256 digests of the independent recovery-verification record,
  independent final-task-verification record, and their identity-bound bundle;
- provenance and duplicate-cluster IDs.

#### `MemoryQueryResult`

- query ID and triggering failure;
- top-k memory IDs and similarity scores;
- all exclusion decisions;
- runtime-visible admission score and compatibility features frozen from
  training/development evidence;
- generated or structured advice;
- hash of the pre-retrieval E2-equivalent shadow decision;
- selected-checkpoint embedding-request SHA-256, processed-batch SHA-256,
  exact float32 query-embedding SHA-256, and their complete binding SHA-256;
- whether advice was admitted by the gate;
- whether advice changed strategy, target, or parameters;
- retrieval and intervention latency.

Sealed relevance annotations use a separate evaluation-only schema and are
joined only after episode actions are complete.

#### `EpisodeSummary`

- system ID, model seed, task ID, and repeat ID;
- oracle-blind runtime terminal reason and opaque terminal token reference;
- browser-action and model-call counts;
- predicted failure and recovery counts;
- first-failure and first-recovery step;
- loop/repeated-error counts;
- memory query/intervention counts;
- environment-failure flags;
- total latency and wall-clock time;
- artifact hashes.

Official task success and verified failure/recovery counts are joined from the
sealed evidence only during offline Table 2 aggregation.

### 5.2 Observation builder

The builder must reproduce the selected checkpoint’s processor, image sizing,
normalization, prompt templates, action encoding, and bbox convention.

Required parity tests:

1. The same saved development observation produces the same processor tensors
   through the training and runtime paths.
2. Post-action fields never appear in pre-action tensors or prompts.
3. Qwen and InternVL processor contracts remain separate and explicit.
4. One source screenshot maps to one registered coordinate plane.
5. Screenshot dimensions, viewport dimensions, and normalized bbox conversion
   are recorded for every action.
6. For production WebArena episodes, the mapper supplies no path authority.
   The adapter copies the registered capture bytes into a fresh non-symlink
   `runtime/screenshots/<observation-index>-<byte-sha256>.png`, recomputes the
   digest, and rejects a mismatched mapper claim. Policy and parameter-provider
   entry points recheck the artifact bytes before consuming the path. The
   browser-free fixture/smoke adapter intentionally retains `screenshot_path=None`.
7. The trusted adapter extracts only slot zero (the post-observation) from the
   pinned Gym step tuple. Reward, `terminated`/`done`, `truncated`, `info`, and
   evaluator truth are discarded inside the adapter and are never passed to an
   integration callback. Runtime execution status and state-change evidence are
   derived only from mapped oracle-blind observations and browser-local
   exceptions; sealed task success remains an offline evaluator output.

### 5.3 Frozen policy adapter

The adapter exposes three distinct calls:

```text
predict_action(task, pre_observation)
    -> PreActionDecision

assess_transition(task, pre_observation, concrete_action, post_observation)
    -> TransitionAssessment

assess_recovery(task, executed_recovery_transition)
    -> RecoveryAssessment
```

The recovery call is mandatory for E2/E3 and is temporally valid only after a
concrete recovery has executed and a post-recovery state has been observed. No
API may simulate a later stage by supplying future inputs to an earlier one.

### 5.4 Independent verifier/oracle

The policy cannot verify its own success. The verifier must use the frozen
benchmark semantics or a preregistered deterministic rule where available.
For this pilot, runtime scoring may be supplied only by an independently
reviewed `libwebarena==0.0.4` compatibility-port release; it is not asserted to
be byte-identical to the pinned official WebArena upstream evaluator. The
repository's current
`reviewed_compatibility_port_pending_external_review` implementation is local
parity evidence only and is intentionally unpromotable, even if somebody adds
or re-hashes a JSON file claiming `PASS`. Promotion requires a new non-pending
implementation identity and source commit. The exact final source bytes must
receive a final-byte parity receipt and an independent review receipt binding
the implementation ID/version/source hash, compiler hash, exact 50-task compile
report, upstream reference, compatibility delta, reviewer/authority identity,
and review time. A joint digest of the exact parity/review files and those
identities must then be pinned in a later reviewed source commit; the allowlist
in this source version is deliberately empty. Self-hashes inside caller JSON
are integrity fields, not external-review authentication. Those identities and
any judge availability receipt are then frozen as separate evidence.

The verifier separately reports:

- task success;
- task progress;
- action execution validity;
- page/environment failure;
- repeated state/loop evidence;
- whether the diagnosed failure was resolved.

Oracle answers, reference actions, or hidden target state must not be injected
into policy or recovery prompts. In the primary system, a recovery trigger may
use only P1 diagnosis/confidence, an observed executor/browser error, and the
frozen loop guard. The verifier/oracle may score an incident and terminate a
completed task, but its outputs are never passed to the decision combiner,
recovery resolver, or memory gate. A system whose trigger receives oracle
failure information must be labelled separately as an **oracle-trigger upper
bound** and cannot appear as primary E2 or E3.

---

## 6. Common action-parameter provider

The learned P3 outputs are action class and target grounding. They do not fully
specify all executable parameters. One frozen provider must serve E0-E3.

| Action | Required concrete parameters | Provider rule |
| --- | --- | --- |
| `CLICK` | target coordinate/element, button, click count | Resolve the predicted bbox/target only; never silently substitute an oracle target |
| `TYPE` | target and text | Derive text from the task and permitted current context using the same frozen provider for all systems |
| `SELECT` | target and option value | Resolve only among currently observable options; log candidate set and selected value |
| `SCROLL` | direction and amount/container | Use a registered discrete direction/amount policy shared by all systems |
| `NAVIGATE` | URL or query | Derive from the task/current state without reference answers or future state |
| `PRESS_KEY` | permitted key/chord | Resolve from a registered key vocabulary and current context |

Provider requirements:

1. It cannot read labels, reference trajectories, future screenshots, or
   success-oracle answers.
2. It cannot use different prompts or models for E0, E1, E2, and E3.
3. Provider failure is logged, not converted into an oracle action.
4. The provider’s model/version, prompt hash, parameters, latency, and output
   are recorded.
5. Any task-specific parameter supplied directly by a benchmark must be
   available identically to all systems and disclosed.

For `SELECT`, the exact current policy observation must carry a strict
`observable_select_controls` list. Each item contains only `target_bbox` and
the ordered, non-empty `candidate_options` visible at that control. Exactly one
item must match the policy-grounded bbox, and the provider's candidate list
must match it in type, value, and order before the selected value is accepted.
This observation-bound check applies equally to normal and recovery actions.
For the pilot, `SCROLL` is frozen to the `viewport` container and registered
amounts `0.5` or `0.75`; arbitrary provider-supplied amounts or containers are
rejected.

An invalid grounded target counts as an agent/action failure unless a
predefined environment-error rule applies.

---

## 7. Concrete recovery resolver

P1 predicts a high-level strategy. The resolver converts it into one or more
concrete, logged actions under a fixed recovery budget.

The oracle-blind recovery trigger is a separate frozen protocol object. When a
post-action assessment exists, the decision combiner forms four Boolean bits
in this exact order: `predicted_failure`,
`failure_probability >= 0.5`, `needs_recovery`, and
`needs_recovery_probability >= 0.5`. The comparison is inclusive. The truth
table maps `0000` to no policy trigger and every other one of the 16 possible
bit patterns to a policy trigger, so each model Boolean and each registered
probability threshold participates independently. An absent assessment is
equivalent to no policy signal. Separately, a non-`EXECUTED` executor status
and a detected frozen-loop condition each trigger recovery. These are the only
three trigger sources (`policy`, `executor`, and `loop_guard`); sealed verifier
or oracle outputs never enter this object. Configuration loading and startup
validation reject any threshold, comparator, signal order, truth-table row, or
executor/loop rule that differs from the registered object.

| Strategy | Primary runtime meaning | Required validity rule |
| --- | --- | --- |
| `NONE` | Continue normal policy execution | Allowed only when recovery is not required |
| `RETRY` | Repeat the prior action once after re-validating target and parameters | Do not retry an invalid, disappeared, or unsafe target blindly |
| `REPLAN` | Invoke the same frozen planner/provider to produce a revised next step | Planner version/prompt and call budget are identical for E2/E3 |
| `BACKTRACK` | Use a permitted browser back/undo/reset operation and verify resulting state | Never assume reversal; confirm the post-backtrack state |
| `ALTERNATIVE_TARGET` | Re-ground and execute a genuinely different compatible target | New target must differ from the failed target and remain task-compatible |
| `ABORT` | Stop safely with an explicit reason | Never count ABORT as task success or recovery success |

Grounded `RETRY`, `REPLAN`, and `ALTERNATIVE_TARGET` actions must also be bound
to an oracle-blind `recovery_target_evidence` record in the exact post-failure
observation. The record binds its schema, observation ID, task ID, task-goal
hash, sorted semantic-target hashes, and the action types compatible with each
visible target. A missing/stale target, a task/goal mismatch, or an
unregistered action type rejects the plan. Target fingerprints exclude action
modifiers such as typed text, selected option, click count, and scroll amount,
so those modifiers cannot masquerade as evidence that a target is present or
different.

### 7.1 Recovery attempt accounting

- A high-level strategy invocation is one recovery attempt.
- Every low-level executor request inside it, including an invalid or rejected
  request, is counted in the shared step/action budget.
- Internal environment/controller retries that repeat the same request for
  infrastructure reasons are logged separately and do not become agent steps.
- A strategy is not successful merely because its command executed.
- Success requires independent evidence that the diagnosed failure was
  resolved or task progress resumed.
- The maximum attempts per failure and per episode are frozen in the protocol.
- Exhausting the recovery budget terminates or returns to normal policy only
  according to a preregistered rule.

### 7.2 No hidden rescue

The runner must not:

- click a known correct target after the model misses;
- repair provider text manually;
- refresh or reset only one system;
- add an unlogged extra retry;
- use a human to guide one failed episode;
- change prompts after seeing a final-task failure.

---

## 8. P4 corrective-memory implementation

### 8.1 Primary memory population rule

For the primary E3 result, construct a separate immutable index for each
seed-specific selected checkpoint. An item is eligible only when:

1. it originates from the registered training split;
2. it has complete causal transition provenance;
3. an actual failure was observed;
4. a concrete recovery was executed;
5. the recovery was independently verified as successful;
6. the episode ultimately achieved valid task success;
7. its registered `memory_update_flag` is true;
8. the item passes exact/near-duplicate and quality filters.

The offline `memory_update_flag` may be reported as a component prediction, but
it is only an eligibility gate and cannot override any provenance, executed-
recovery, verified-success, final-success, or exclusion requirement.

The `verified_recovery_success` and `final_task_success` booleans are also only
gates; they are not sufficient verification evidence. Every potentially
admitted provenance record must contain both of the following:

- `table2-memory-recovery-verification-evidence-v1`, naming an independent
  `verifier` or `reviewer` by non-empty ID and version and binding the source
  sample, recovery sample, canonical task, episode, pre-recovery-state hash,
  executed-recovery-action hash, post-recovery-state hash, successful outcome,
  and its canonical evidence SHA-256;
- `table2-memory-final-task-verification-evidence-v1`, naming an independent
  versioned verifier/reviewer and binding the source sample, canonical task,
  episode, task-specification hash, terminal-state hash, terminal verifier-
  output hash, successful outcome, and its canonical evidence SHA-256.

Eligibility recomputes each canonical evidence digest and requires the
recovery pre/action/post hashes to equal the actual causal recovery transition.
For production transitions, pre/post state paths are resolved beneath the
declared training-data root and the artifact bytes themselves are hashed;
missing files or paths escaping that root fail closed. Browser-free synthetic
fixtures may hash an in-memory canonical state only and cannot claim production
artifact evidence. Terminal task-state/verifier-output hashes remain externally
produced post-training evidence and must be bound by the frozen provenance
trust root.
It also requires the canonical task, episode/trajectory, and source step to
match the source training row and causal transition, preventing a
self-consistently rehashed record from being reattached to another task. It
then closes both evidence digests together with the sample, recovery, task, and
episode identities under
`table2-memory-verification-bundle-v1`. Missing evidence, an unversioned or
non-independent authority, a digest mismatch, an identity mismatch, or a
transition mismatch fails the complete build. Thus an operator cannot admit a
memory using success booleans alone.

The ID/version and `independent_verification=true` fields are content-bound
attestations, not cryptographic proof of authorship by themselves. Operational
independence must come from the preregistered verifier/reviewer process and the
externally frozen provenance/campaign trust root. The present repository has no
real post-training verifier records or signing trust root, so production memory
construction remains blocked until those selected-run artifacts are supplied;
synthetic fixtures establish schema behavior only.

### 8.2 Memory exclusions

Before indexing and again before returning a query result, exclude:

- validation and test records;
- the current task or episode;
- the same original task/template when the registered protocol requires it;
- exact input/image duplicates;
- near-duplicate task/state clusters;
- items without verified final task success;
- provenance-invalid or semantically ambiguous transitions.

Every exclusion count must appear in the memory manifest.

Train candidates and all 50 ordinary WebArena tasks must be clustered in one
joint, hash-bound duplicate namespace. The namespace fixes the audit tool ID,
version, configuration hash, and source hash. Every eligible memory item and
every verified task audit record cites that same namespace, and each task
record's evidence hash is recomputed from its canonical content. Merely
supplying an arbitrary 64-character string is not duplicate-audit evidence.

### 8.3 Frozen memory rule

The primary E3 evaluation permits reads only. It performs zero writes.

```text
build from training -> validate exclusions -> freeze -> hash -> evaluate
```

Online/dynamic memory may be studied only as a separately labelled secondary
protocol. It must never be mixed into the primary leakage-controlled Table 2
result.

Every frozen item carries the recovery-evidence, final-task-evidence, and
combined bundle digests. The store manifest embeds a sorted
`table2-memory-verification-manifest-v1` closure over those per-item digests and
hashes that closure into the store identity. The store also contains
`verification_evidence.json`, which retains the exact versioned recovery and
final-task records for every admitted item under a separate sorted aggregate
hash. Store loading replays each record schema, authority identity/version,
canonical record digest, identity-bound bundle, item binding, and complete
manifest closure before calibration replay or query access. Missing, modified,
or reattached verification evidence therefore fails closed at post-training
handoff and reload.

### 8.4 Query and intervention

The primary query is allowed only after a normal action has been attempted,
its post-action state has been observed, and P1 has produced a temporally valid
failure assessment. Pre-action memory query/advice is prohibited. The
registered recovery trigger may be raised by P1 diagnosis/confidence, an
observed executor/browser error, or the frozen loop guard; it may not use the
evaluation oracle. The query uses the current post-failure representation plus
permitted task/site/action metadata.

The post-failure representation has one exact production bridge. The loaded
validation-selected checkpoint exposes its inference-only
`memory_embedding` forward, which returns the 768-dimensional post-action
memory-task-adapter tensor consumed by the Pillar 4 head. `ProductionRunner`
constructs the validating provider internally from that callback and the
frozen processor contract. `SeedRuntimeBinding` cannot supply a separate,
arbitrary embedding provider.

For every production E3 query, runtime creates a typed recursively immutable
embedding request (including nested page-state and action-parameter mappings)
that contains the exact `TransitionInput` and binds the current query,
post-failure-observation and failed-action IDs; the post-failure observation
SHA-256; the complete post-action-input SHA-256; the processor-contract
SHA-256; and the selected-checkpoint SHA-256. The selected-checkpoint backend
must return a typed receipt containing those identities, the request SHA-256,
the processed-batch SHA-256, the registered embedding-stage identifier,
the 768 float values, and the SHA-256 of their little-endian float32 bytes.
The exact processed batch stays inside the source-attested backend; runtime
and package validation bind its backend-created digest to the request,
checkpoint, processor, and float32 embedding evidence. Before L2-normalizing
the query for cosine retrieval, runtime rejects mutation or identity drift,
rejects nondeterministic output for an identical processed batch, and treats
byte-identical output from two different processed batches as a stale/constant-
output failure. These cross-query checks persist for the matched seed across
episodes. The request, receipt bindings, processed-batch hash, and embedding
hash are retained in the memory-query evidence package. Browser-free smoke
fixture readers remain explicitly non-production and do not claim this
selected-checkpoint evidence.

The E3 memory-query event additionally retains the normalized 768-dimensional
float32 query vector and the SHA-256 of its exact little-endian bytes. Package
validation reloads the seed-matched `FrozenMemoryStore`, reconstructs the
current task/episode/duplicate exclusions, and replays cosine retrieval. It
requires exact neighbour IDs/order, exclusion reasons, considered/eligible
counts, float32 cosine scores, threshold admission, advised strategy, final
strategy, and changed-strategy/final-decision fields. Missing vectors or any
replay mismatch invalidates the episode; a logged hash alone is insufficient.

Before any E3 retrieval, compute and log the E2-equivalent **no-memory shadow
decision** under the same inputs. Retrieval then either preserves or modifies
that frozen shadow decision. This makes the first memory intervention and its
causal effect on strategy, target, or parameters directly auditable.

Any human/benchmark relevance label is sealed scoring data attached only after
the decision. It must not be available to the runtime admission gate, ranking,
advice generator, or intervention logic.

E3 may use retrieved evidence admitted by the frozen runtime-visible score to:

- adjust the recovery-strategy ranking;
- prefer a compatible alternative target/tool;
- supply a concise structured warning or recovery hint;
- abstain when relevance is below the frozen threshold.

The threshold is derived automatically from provenance-bound
`table2-memory-calibration-evidence-v1`; it is not supplied by a caller. For
each seed-specific selected checkpoint, the builder starts from the actual
eligible train-only memory population and its exact normalized float32 P4
embeddings. Each eligible item becomes a leave-one-out query. Self, same-task,
exact-duplicate, and near-duplicate candidates are removed, then the shared
runtime cosine primitive retains the deterministic top 3 with ascending
memory-ID tie-breaking.

The calibration relevance value is also derived, never entered as a caller
label: a retrieved pair is relevant exactly when both endpoints are verified
successful corrections with the same registered recovery strategy. The
executed recovery action remains bound provenance but is not part of the
relevance label because the frozen runtime intervention consumes only the
retrieved strategy. This is a registered train-only calibration proxy, not a
human/benchmark relevance label. Every full evidence row binds
both endpoint IDs, task/episode and duplicate identities, provenance-record
SHA-256, immutable-memory-item SHA-256, normalized float32 embedding SHA-256,
rank, cosine score, and the components from which relevance was derived. The
evidence package binds the selected checkpoint, model seed, logical training
records, dataset artifacts, both resolved-config identities, provenance
manifest, duplicate namespace, and frozen protocol; validation, test, and
locked-test read counts must all be zero.

Inside that evidence package is the compact
`table2-memory-threshold-calibration-v2` replay. Its derived rows are
`{row_id, source_split, cosine_similarity, relevant}` records closed by a
canonical SHA-256. The registered replay sweeps the unique observed cosine
scores, admits scores greater than or equal to each candidate threshold, and
maximizes binary F1 using exact integer/rational comparisons; an exact tie
selects the highest threshold. Construction rebuilds the complete evidence
from the eligible records, provenance and embeddings, and store reload
recomputes the frozen endpoint fields, memory-item/embedding hashes,
exclusions, top-3 ranks, scores, derived labels, and threshold selection from
the frozen items and embeddings while preserving the external provenance-
record commitments. The memory-build CLI accepts no calibration file, score
rows, relevance labels, pair list, or free threshold. Both
`calibration_evidence.json` and the compact
`threshold_calibration.json` are hashed into each seed-specific store.

E3 must log whether memory changed the decision. Merely retrieving neighbours
does not prove a memory-driven agent.

### 8.5 P4 experiments

Primary:

- E2: retrieval disabled;
- E3: successful-recovery index, retrieval plus intervention.

Required diagnostics:

- retrieval enabled, advice withheld;
- random/shuffled retrieval;
- all completed trajectories;
- generic successful trajectories;
- successful-recovery-only memory;
- retrieve after every completed post-action assessment vs failure-triggered
  retrieval, never pre-action;
- top-k and relevance-threshold sensitivity.

Every diagnostic corpus and index remains training-only, immutable, and read-
only. Diagnostics may change composition only according to variants frozen
before scoring; they may not ingest validation/test episodes.

### 8.6 P4 metrics

The primary runtime exposes only its frozen top-3 neighbours and its
intervention consumes only the retrieved strategy. Therefore the primary
diagnostics are **Strategy Hit@1** and **Strategy Hit@3**, not Recall@5. A
separately frozen top-k sensitivity run may retrieve five neighbours and then
report Strategy Hit@5; the primary top-3 campaign must display that cell as
`N/A` rather than pretending the fourth and fifth ranks were observed.

- Strategy Hit@1 and Strategy Hit@3;
- mean reciprocal rank;
- retrieval coverage;
- abstention rate;
- irrelevant-memory rate;
- intervention coverage;
- strategy/target change rate;
- useful-intervention rate;
- harmful-intervention rate;
- E3 minus E2 task-success delta;
- E3 regression count where E2 succeeded and E3 failed;
- index size and query latency.

Retrieval relevance must be defined and validated before scoring. A convenient
similarity score alone is not relevance ground truth.

For a preregistered relevant-item set `R_q`, a diagnostic with sufficient
retrieval depth may report:

```text
Recall@K = mean over queries with |R_q| > 0 of
           |TopK(q) intersection R_q| / |R_q|
```

Queries with no eligible relevant item are reported through coverage and
abstention, not silently inserted into the Recall@K denominator. If relevance
is defined only as matching the recovery-strategy label, name the metric
**Strategy Hit@K**, not Recall@K.

---

## 9. Episode algorithm

```text
1. Load the frozen protocol, system config, model checkpoint, and manifests.
2. Start the registered 120-second pre-browser setup deadline, create and
   verify the task/runtime/evaluator bindings, start measurement, and reset
   the shared model backends. Record its typed completion evidence at the
   exact browser-reset boundary.
3. At that boundary start the separate 600-second task clock, reset the
   registered WebArena environment, obtain the hashes-only hidden-state reset
   receipt, and record the initial observation.
4. Check the independent success oracle in the sealed channel only to detect
   an already-complete reset; expose no score, label, reference, or progress
   value to the policy.
5. Build a pre-action observation.
6. E0/E1/E2/E3 selects a normal action under its registered policy.
7. Resolve concrete parameters with the common provider.
8. Validate safety and execute exactly one concrete action.
9. Wait using the registered settle rule and capture the post-action state.
10. For E2/E3, assess the action-conditioned transition with P1/P2.
11. Decide whether to recover using only the P1 assessment, observed
    executor/browser status, and frozen loop guard.
12. Run the independent verifier in a sealed evaluation channel; it may score
    progress/success or end a completed task, but it cannot change Step 11.
13. If no recovery was triggered, continue to the next normal step.
14. If recovery was triggered:
    a. E2 uses diagnosis and the frozen recovery resolver.
    b. E3 first logs the no-memory shadow decision, then queries the frozen
       train-only index, applies the relevance gate, records any intervention,
       and uses the same resolver.
15. Execute the concrete recovery action(s) under the shared budget.
16. Capture the post-recovery state and verify failure resolution/progress in
    the sealed evaluation channel.
17. Record predicted and verified recovery outcomes.
18. Continue until success, safe abort, timeout, environment failure, loop
    termination, or budget exhaustion.
19. Write schema-complete logs and zero evaluation memories.
```

The 600-second clock starts immediately before `WebArenaAdapter.reset`, so
environment construction and reset consume the task budget and a call that
returns at or after 600 seconds cannot influence the episode. Work before this
boundary is not free or unbounded: the frozen 120-second setup guard covers
task projection, runtime/evaluator binding construction and verification,
measurement setup, shared-backend reset, runtime construction, and all event
setup up to that exact boundary. The runner checks the guard before and after
every external callback and again at completion. An overrun is a failed,
non-scorable run; it cannot silently shift the 600-second start time.

External calls at both the 120-second setup and 600-second task boundaries run
under a POSIX main-thread `SIGALRM`/`ITIMER_REAL` hard-interruption guard, not
only a post-return clock check. A host without that mechanism fails closed.
Because an uninterruptible native/kernel hang may still ignore Python signal
delivery, the final campaign additionally requires a process/container
supervisor with a wall-clock kill limit; the in-process guard does not replace
that supervisor.

### 9.1 Loop detection

Freeze a deterministic rule before scoring. It should combine registered
signals such as:

- repeated state hashes/perceptual hashes;
- repeated equivalent action-target pairs;
- repeated URLs/state signatures with no agent-observable state change under a
  frozen oracle-independent proxy;
- repeated diagnosis/recovery cycles;
- bounded sliding window and threshold.

Model self-report alone is insufficient to define a loop. Sealed verifier
progress and evaluator loop labels are analysis-only and must not enter the
runtime loop guard.

The protocol must state the exact state/action fingerprint, perceptual-
similarity threshold, rolling-window length, repetition count, and A-B-A-B
cycle rule. These values are chosen on development episodes and frozen before
scoring.

### 9.2 Environment failures

Classify separately:

- browser crash/controller error;
- task reset failure;
- network outage/HTTP infrastructure failure;
- site unavailable;
- CAPTCHA/login/credential gate when not caused by the agent;
- evaluator failure.

Agent-caused terminations remain failures in the primary matched analysis.
Infrastructure-invalid blocks follow Section 4.5: rerun the whole paired block
under the frozen rule or exclude the whole block. Report a companion schedule-
level summary containing every attempted run so infrastructure burden remains
visible.

---

## 10. Benchmark and environment registration

### 10.1 Selection gate

Before scored evaluation, produce a benchmark-selection report comparing:

1. BrowserGym/AgentLab with a controlled public benchmark such as WebArena,
   WebArenaVerified, or VisualWebArena;
2. WebVoyager as a live-site secondary protocol when direct WebCoach/WebVoyager
   context is required;
3. a controlled project benchmark only if public integration is infeasible.

Selection criteria:

- action-schema compatibility;
- screenshot and target-grounding availability;
- deterministic reset/start state;
- official or independently valid success oracle;
- reproducibility and versioning;
- failure/recovery opportunities;
- infrastructure and compute cost;
- site safety and credential requirements.

### 10.2 Minimum controlled fallback

If a public benchmark cannot be completed, the fallback requires:

- at least 100 held-out task episodes;
- multiple sites/domains and task templates;
- all executable action classes where support exists;
- a preregistered failure-prone subset;
- fixed start states where possible;
- deterministic or independently adjudicated success;
- identical paired E0-E3 task runs;
- complete action/state/recovery logs.

It must be named a controlled project benchmark, not a direct SOTA leaderboard.

### 10.3 Frozen protocol manifest

Record:

- protocol ID and creation timestamp;
- benchmark repository, commit/release, and license;
- task-definition version and exact task IDs;
- sampling and exclusion rules;
- development vs final scored task IDs;
- start-state/reset implementation;
- browser/controller/Playwright versions;
- OS/container image and package lock;
- viewport and device scale factor;
- page-settle/network-idle rule;
- action and recovery budgets;
- frozen `K` for Recovery@K;
- the 120-second pre-browser setup timeout, the exact browser-reset boundary,
  the 600-second task timeout, and model-call timeout;
- oracle/evaluator version and prompt hash, if model-based;
- repetition count and seeds;
- maximum infrastructure-invalid paired-block reruns and ordinal-replacement
  rule;
- system execution-order randomization;
- credentials and destructive-action policy;
- known invalid/outdated task policy;
- environment-failure classification.

After freezing, task removal or replacement requires a new protocol ID. Never
silently edit the scored task manifest.

---

## 11. Exact Table 2 metrics

Every metric export contains its numerator, denominator, point estimate, and
95% confidence interval where applicable.

### 11.1 Task success rate

```text
TSR = successful runs in valid matched paired blocks /
      all runs in valid matched paired blocks
```

The independent task oracle determines success. ABORT, timeout, step-limit,
unresolved loop, or agent-caused invalid termination is not success. The
denominator is every run in the final valid matched paired-block set defined in
Section 4.5; infrastructure exclusions are disclosed separately.

### 11.2 Recovery success rate

```text
RSR = verified successful recovery attempts /
      all initiated high-level recovery attempts
```

The primary denominator includes false-positive triggers; otherwise an
over-triggering controller would be rewarded by removing its unnecessary
attempts. Report two companions: verified-failure-only RSR and false-trigger
rate. A false-positive trigger contributes zero to the primary RSR numerator,
even if its action later advances the task, because there was no verified
failure to recover from. Otherwise, a successful attempt requires sealed
verification of resolved failure, registered progress, or task success.
Disabled recovery or zero initiated attempts yields `N/A`, not zero.

For a K frozen before scoring, report:

```text
Recovery@1 = verified failure incidents resolved on their first high-level
             recovery attempt /
             verified failure incidents receiving at least one attempt

Recovery@K = verified failure incidents resolved within at most K high-level
             recovery attempts /
             verified failure incidents receiving at least one attempt
```

E0 and E1 receive `N/A` for project-controller recovery metrics because the
registered recovery controller is disabled.

For the pilot, headline RSR/Recovery@K and success-after-failure estimates are
computed from verified failure incidents that naturally occur in the 200
ordinary WebArena episodes. The 15 preregistered controlled scenarios (60
E0--E3 episodes) are a separate mechanism and boundary diagnostic: they test
whether P1 executes and assesses recovery under known conditions, but they are
not substituted for the ordinary-task denominator and are not pooled into a
headline WebArena rate. Both partitions remain fully reported.

### 11.3 Success after an initial verified failure

```text
SAF = successful episodes containing at least one verified agent failure /
      all episodes containing at least one verified agent failure
```

This may be reported for all four systems and captures incidental later
correction in E0/E1 as well as explicit recovery in E2/E3.

Also report verified-failure incidence over all paired episodes. SAF is a
conditional resilience description and, by itself, is not a causal estimate.

### 11.4 Loop episode rate

```text
LER = episodes in valid matched blocks meeting the registered loop rule /
      all episodes in valid matched blocks
```

### 11.5 Repeated-error event rate

```text
RER = repeated equivalent failure events in valid matched blocks /
      all verified agent failure events in valid matched blocks
```

This is not interchangeable with loop episode rate.

### 11.6 Average browser actions

```text
Mean steps = total budget-consuming executor requests in valid matched runs /
             number of valid matched episode runs
```

Count every budget-consuming executor request as one step, including executed,
invalid, rejected, normal-policy, and recovery requests. Internal environment
retries of the same request do not count as agent steps. Report the total plus
executed/rejected/recovery breakdowns, mean, standard deviation, median, IQR,
and the distribution for successful and unsuccessful tasks.

### 11.7 Average recovery attempts

Report both:

- per registered task run;
- per episode with at least one verified failure.

### 11.8 Unrecovered failure rate

The original PDF column is retained with a failure-incident definition. A
verified failure incident begins at an oracle-verified agent failure.
Consecutive ineffective actions responding to that same failure remain inside
the incident. It ends as recovered when sealed verification records registered
progress or task success; otherwise it ends unresolved at episode termination.

```text
UFR = verified agent-failure incidents terminating without registered progress
      / all verified agent-failure incidents
```

The oracle is used only for analysis, never to trigger recovery. Report an
episode-level “any unresolved failure” rate as a companion if useful. UFR is
not simply `1 - RSR` because an incident may contain multiple attempts.

### 11.9 Unnecessary-intervention and harm rates

```text
Unnecessary intervention rate = recovery interventions when the independent
                                verifier found no failure /
                                all recovery interventions

Paired memory regression rate = tasks where E2 succeeds and paired E3 fails /
                                tasks where E2 succeeds
```

### 11.10 Environment-failure rate

```text
EFR_schedule = infrastructure-invalid physically launched attempts /
               all physically launched original plus rerun attempts
```

This schedule-level diagnostic includes blocks excluded from the primary paired
analysis, so infrastructure burden cannot disappear when invalid blocks are
removed. Report original and rerun numerators separately.

### 11.11 Efficiency

Report:

- model and trainable parameters;
- decision/provider/recovery/retrieval latency;
- task wall-clock time;
- model calls and tokens;
- peak GPU/system memory;
- memory-index size and query time;
- training GPU hours separately from evaluation cost.

---

## 12. Paper Table 2 outputs

The number **12** above is only this implementation document's section number.
It does not create a journal table called “Table 12,” “Table 12.1,” or “Table
12.2.” The paper has exactly one headline table in this scope: **Table 2**.

### Headline paper Table 2

This is the single proposed headline **Table 2** for the paper. It reports the
registered E0-E3 end-to-end comparison and is the table the project is
currently implementing.

| System | Task Success Rate | Recovery Success Rate | Success After Initial Failure | Avg. Browser Actions | Loop Episode Rate | Unrecovered Failure Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 Base selected-backbone agent | | N/A | | | | |
| E1 Trained action/grounding, no recovery | | N/A | | | | |
| E2 Failure diagnosis + recovery, no memory | | | | | | |
| E3 Complete four-pillar agent | | | | | | |

### Unnumbered companion diagnostics for paper Table 2

The matrix below is an unnumbered companion diagnostic view in this plan. It
explains the mechanisms and denominators behind the same headline paper Table
2; it is not “Table 12.2,” a second Table 2, or a replacement for Table 2. If a
journal later requires it in an appendix or supplement, it may receive a
separate appendix/supplement label at typesetting time, but it remains
supporting evidence rather than the headline result.

| System | Verified-failure RSR | False-trigger Rate | Recovery@1 | Recovery@K | Failure Incidence | Avg. Recovery Attempts | Repeated-Error Rate | Environment-Failure Rate | Task Time | Model Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | N/A | N/A | N/A | N/A | | N/A | | | | |
| E1 | N/A | N/A | N/A | N/A | | N/A | | | | |
| E2 | | | | | | | | | | |
| E3 | | | | | | | | | | |

Every rate must show `numerator/denominator` in the supplement or machine-
readable result package.

Before the preregistered blinded audit is completed and signed, generated
aggregates are labelled `DRAFT_PILOT_ONLY`. After that audit they may be
labelled `PILOT_ONLY`. Neither label authorizes paper values: the headline
Table 2 remains `N/R` until the later frozen final-paper campaign is complete.
When no output path is supplied, the draft export is written under
`results/table2/<campaign_id>/draft/`, while the adjudication-gated export uses
`results/table2/<campaign_id>/`. If an explicit output path is supplied, the
two stages must use distinct directories; immutable draft files are never
silently overwritten during promotion.

---

## 13. Logging and artifact structure

### 13.1 Required logs

Each system's `runtime/` package contains:

- `episode_manifest.json`;
- `actions.jsonl`;
- `transitions.jsonl`;
- `recoveries.jsonl`;
- `memory_queries.jsonl` for E3 only;
- `environment_events.jsonl`;
- `terminal_signals.jsonl` containing only opaque verifier receipts;
- `episode_summary.json` without oracle truth;
- `artifact_hashes.json`;
- causal input-ID/hash lists, trigger sources, and E3 no-memory shadow
  decisions in the relevant JSONL records;
- for each production E3 retrieval, the recursively frozen embedding request
  plus request, processed-batch, processor, checkpoint, transition,
  post-failure-observation, and float32 embedding SHA-256 bindings;
- registered screenshots or state hashes as permitted.

Each paired `sealed/` package contains the independent verifier events and
post-execution relevance annotations that runtime code is forbidden to read.
Campaign-level aggregates contain the episode CSV, exact metric numerators and
denominators, retrieval diagnostics, statistics, provenance, and the blinded
human-audit selection manifest.

Canonical JSONL records use monotonic event IDs, explicit foreign keys, and a
previous-record hash (or equivalent append-order integrity mechanism). Every
budget decrement must resolve to an action/recovery record, and all published
metrics must be reproducible solely from the frozen logs.

### 13.2 Canonical paired-block output layout

```text
artifacts/table2/<campaign_id>/
    campaign_manifest.json
    access_ledger.jsonl
    deviation_ledger.jsonl
    frozen/
    component_test/
    memory/
        seed_<matched_seed>/
            manifest.json
            manifest.sha256
            embeddings.npy
            items.jsonl
            verification_evidence.json
            calibration_evidence.json
            threshold_calibration.json
    schedule/
    paired_blocks/
        seed_<matched_seed>/
            <task_id>/
                repeat_<repeat_id>/
                    rerun_<rerun_ordinal>/
                        block_manifest.json
                        E0/runtime/
                        E0/sealed/
                        E1/runtime/
                        E1/sealed/
                        E2/runtime/
                        E2/sealed/
                        E3/runtime/
                        E3/sealed/
    aggregate/
    manual_audit/
    runtime_readiness/                  # provisional PC-01 pilot only
        probe_evidence/                 # portable non-scored first-block evidence
        live_matched_e0_e3_receipt.json
        live_matched_e0_e3_receipt.sha256
```

Every system, including E0, is nested under the matched seed. This prevents an
E0 result from being overwritten or paired with the wrong E1-E3 checkpoint
schedule. Runtime records and sealed verifier/oracle evidence are physically
separated inside every system package.

### 13.3 Mandatory provenance

- code commit hash;
- exact resolved-configuration payload-byte SHA-256 and canonical parsed-
  mapping SHA-256 (`resolved_config_sha256` and
  `resolved_config_record_sha256`); the checkpoint-saved full configuration
  must equal that canonical mapping exactly;
- model repository/revision;
- checkpoint SHA-256;
- dataset, overlay, supplement, and split hashes;
- memory-index hash;
- benchmark/task/oracle hashes;
- dependency/container lock;
- hardware and driver information;
- seed/repetition and execution-order schedules;
- start/end timestamps;
- all deviations and interruptions.

The run manifest must bind these hashes into one immutable campaign identity.
Every episode package must reference that identity so screenshots, decisions,
oracle records, memory queries, and aggregates cannot be mixed across protocol
versions silently.

### 13.4 Privacy and safety

Logs must not preserve passwords, session tokens, payment information, private
form contents, or unrelated personal data. Store redacted values and hashes
where necessary. URL sanitization covers user information, sensitive path
segments, query/fragment keys (including `session` and `sid` aliases), mixed
`&`/`;` delimiters, and bounded nested percent-encoded redirect URLs; values
beyond the decoding bound fail closed. Benign URLs remain unchanged so useful
provenance is retained. Destructive or externally consequential tasks require
an explicitly approved sandbox/safety policy.

---

## 14. Statistical analysis

### 14.1 Paired design

Every system attempts the same task IDs and registered repetitions. Use:

- a two-sided exact sign test for task success with `task_id` as the
  inferential unit: first average the paired right-minus-left success
  differences over the unique repeat cells at the fixed matched model seed 42
  within each task, then classify each task effect as right-better,
  left-better, or tied;
- exclude zero task effects from the exact sign-test denominator, retain them
  in the reported support, and treat paired-block discordance counts as
  descriptive only;
- paired bootstrap for recovery and success-after-failure;
- paired bootstrap or a paired non-parametric test for action counts and time;
- absolute and relative changes;
- effect sizes and 95% confidence intervals;
- Holm correction across all three registered task-success contrasts:
  E0-vs-E1, E1-vs-E2, and E2-vs-E3. A contrast with no non-zero task effects
  receives the valid maximally non-significant value `p=1`, so the planned
  family never changes in response to the observed outcomes.

Exact duplicate rows for the same task, fixed matched model seed 42, and repeat are
collapsed before the task effect is calculated, so copying an observation
cannot strengthen significance. Conflicting outcomes for one such cell fail
closed. This replaces row-level McNemar inference, which would incorrectly
treat repeated episodes from the same task as independent trials.

### 14.2 Model-seed and task uncertainty

Table 2 uses the registered seed 42 checkpoint only. PC-01 is not retrained and
seeds 43--44 are not added. PC-02 and PC-03 also contribute their seed-42
validation packages only to final backbone selection. Consequently Table 2
reports task-level uncertainty and must explicitly state that model-seed
uncertainty is not measured.

Do not treat repeated task executions as independent task replicates.
Task-success significance reduces those repeated measurements to
one paired direction per task as specified above; interval estimation uses a
task-cluster bootstrap. For attempt-
level RSR, false-trigger, and failure-incident metrics, cluster resampling by
task/episode so correlated attempts are not treated as independent samples.

### 14.3 Failed and interrupted runs

Never hide an unfavourable completed seed or system run. Interrupted runs are
retained with their terminal reason and excluded only by a preregistered rule.

---

## 15. Work to perform while DGX training continues

This workstream must not modify the active training configuration, checkpoint,
or training artifact directory.

### Phase P0 - Freeze the development boundary

Tasks:

1. Create a runtime-development task manifest distinct from final scored tasks.
2. Register the current training code/config commit and do not change it for
   runtime convenience.
3. Freeze the four-pillar and E0-E3 contracts in this document.
4. Decide the benchmark compatibility spike and fallback criteria.

Exit gate:

- development and final data/task boundaries are explicit;
- zero locked-test reads;
- active DGX run remains untouched.

### Phase P1 - Runtime contracts and invariant tests

Tasks:

1. Implement serializable contracts from Section 5.1.
2. Add resolved E0-E3 feature-switch validation.
3. Add checkpoint/config/protocol hash propagation.
4. Implement hand-calculated metric fixtures.

Exit gate:

- round-trip serialization passes;
- invalid/missing fields fail closed;
- E0-E3 switch matrix tests pass.

### Phase P2 - Observation and policy parity

Tasks:

1. Implement the observation builder.
2. Implement frozen-checkpoint `predict_action` and `assess_transition` paths.
3. Test processor/prompt parity on saved development examples.
4. Test temporal information barriers.

Exit gate:

- pre-action output is invariant when only forbidden future fields change;
- post-action assessment uses the executed action and observed next state;
- no oracle/reference answer appears in policy inputs.

### Phase P3 - Provider, executor, and verifier

Tasks:

1. Implement all six concrete action contracts.
2. Integrate the selected environment adapter.
3. Implement deterministic target/bbox conversion.
4. Implement the independent success/progress verifier.
5. Classify agent vs environment failures.

Exit gate:

- each action either executes or rejects explicitly;
- deterministic browser fixtures pass;
- known success/failure oracle fixtures pass;
- no hidden target replacement occurs.

### Phase P4 - Recovery controller

Tasks:

1. Implement each high-level recovery strategy.
2. Implement recovery validity and budget checks.
3. Capture and verify post-recovery state.
4. Separate predicted recovery outcome from verified recovery success.

Exit gate:

- one injected failure triggers a concrete recovery;
- invalid recovery is rejected and logged;
- ABORT terminates safely and never counts as success.

### Phase P5 - Frozen corrective memory

Tasks:

1. Implement the strict training-only memory builder.
2. Add task/episode/exact/near-duplicate exclusions.
3. Derive provenance-bound leave-one-out calibration evidence and its compact
   threshold replay without accepting caller-authored scores or labels.
4. Add immutable manifest, dual resolved-config identities, and complete hash
   closure.
5. Bind every runtime query to the exact selected-checkpoint post-action
   memory embedding request and backend-created receipt.
6. Implement query, relevance gate, advice, and intervention logging.
7. Implement retrieval metrics and diagnostic variants.

Exit gate:

- zero non-training items can enter or be returned;
- calibration evidence replays exactly from frozen eligible items and
  embeddings;
- a stale, constant, wrong-processor, wrong-input, or wrong-checkpoint P4
  embedding receipt fails closed;
- final-evaluation writes are impossible by configuration;
- empty/irrelevant retrieval leaves E3 behaviour equal to E2;
- intervention effects are observable in logs.

### Phase P6 - Episode runner, logs, and metrics

Tasks:

1. Implement the episode state machine and budgets.
2. Implement schema-complete logs.
3. Implement exact Table 2 formulas.
4. Implement paired aggregation and statistics.
5. Implement artifact hashing and manifest validation.

Exit gate:

- hand-calculated fixtures reproduce every metric;
- empty denominators yield `N/A`, never a misleading zero;
- a complete episode package validates independently.

### Phase P7 - Mini-checkpoint browser smoke

Use an existing mini/provisional checkpoint on 5-10 development tasks only.

Required demonstrations:

1. one ordinary successful episode;
2. one failed action and post-action P1 diagnosis;
3. one E2 concrete recovery;
4. one E3 train-only retrieval and logged intervention;
5. one loop or repeated-state termination;
6. one environment failure classification;
7. zero final-task or locked-test access.

Performance may be low. This phase validates execution and evidence capture,
not the paper claim.

### Phase P8 - Freeze evaluator release candidate

Before final model selection completes:

1. finalize the benchmark/task manifest;
2. finalize oracle, budgets, metrics, logs, and analysis;
3. compile all exact 50 task evaluator configurations and freeze the compile
   authority; the current implementation-specific report remains explicitly
   pending and is not campaign permission;
4. cut a new non-pending evaluator implementation identity from a clean source
   commit, then measure parity on those exact final bytes;
5. obtain independent external review whose receipt binds the final
   implementation, parity, compiler, compile-authority, delta, upstream
   reference, reviewer, authority, and timestamp hashes;
6. freeze evaluator code/config and every bound evidence hash;
7. produce a machine-readable dry-run PASS report and record all known
   limitations.

After this freeze, final model results must not be used to redesign the
evaluator.

---

## 16. Exact sequence after model training finishes

### 16.0 Supported handoff and checkout constraint

The evaluation handoff is prepared only from a clean, committed Git checkout.
The runner, runtime integration factory, evaluator source, protocol, prompts,
and overlays must all be the bytes in that checkout. The handoff and campaign
directories must be outside the checkout so that creating them does not dirty
the source tree. They must also be disjoint: the campaign directory cannot be
equal to, inside, or an ancestor of the handoff package. The freeze copies the
handoff authority, authenticates every handoff-sourced byte before and after
copying, records `frozen/handoff_consumption.json`, and replays the complete
handoff inventory immediately before finalization. Frozen-package validation
then verifies every recorded source-to-campaign digest without reopening the
external handoff directory. `prepare_table2_handoff.py` rejects a dirty/non-Git checkout,
missing inputs, placeholder/unknown environment values, unverified audit
evidence, hash drift, noncanonical processor contracts, or an incomplete task
export. It does not infer any external value.

The supplied dependency lock is copied byte-for-byte to
`dependency.lock` in the handoff and then to `frozen/dependency.lock` in the
campaign. `environment.json`, the campaign artifact-hash closure, package
validation, the pre-import evaluation CLI check, and the per-block execution
guard all verify the same SHA-256. A path or hash alone is not accepted as
dependency evidence.

### 16.0.1 Measured task-interface result and current stop condition

The pinned source authority is `libwebarena==0.0.4` wheel SHA-256
`9ebee3b4371502c4f0f7e727a72e5846235d6750d420db9a3b8a168107654feb`,
containing `webarena/test.raw.json` at SHA-256
`7b50386fd69163dbc05d615d834df4c6ed2c35596e97a1b10d17451c02537652`.
An exact audit of upstream public indices 0--49 against
`pc01-p3-six-browser-actions-v1` gives:

- 50 tasks audited in their registered order;
- 47 `string_match` tasks that require assistant-answer submission and are
  incompatible with the six browser-action interface;
- three compatible page-state `url_match` tasks: indices 44, 45, and 46;
- 13 model-based fuzzy-judge tasks among the 47 answer tasks: indices 8, 16,
  17, 18, 19, 20, 22, 24, 34, 35, 47, 48, and 49.

This is a valid `FAIL` audit outcome, not a reason to edit the evidence. The
current 0--49 registry is not handoff-eligible. The handoff recomputes the
audit and refuses to proceed whenever even one registered task is
incompatible. No operator may silently drop the 47 tasks, replace their
evaluators, map an answer to a TYPE action, or add STOP/ANSWER after seeing
outcomes.

The next scientific decision must be user-approved and preregistered before a
live pilot: either register 50 public page-state-compatible development tasks,
with a versioned exporter/audit/protocol update, or introduce and justify an
answer/termination interface as a material P3 and protocol change. The
original 0--49 tasks remain recorded as development
exposures and are never eligible for the final campaign. All workflow steps
below the interface audit are conditional on resolving this stop condition.

### 16.0.2 Exact fail-closed preparation order

Run these gates in order. The five-token task URL map used to resolve task
content and the seven-origin BrowserGym service map used to attest a deployment
are separate inputs; their five shared origins must agree exactly.

1. Export task content directly from the pinned wheel, never from a manually
   reconstructed JSON file:

   ```bash
   PYTHONPATH=src python3 scripts/export_table2_webarena_tasks.py \
     --upstream-task-source /secure/webarena/libwebarena-0.0.4-py3-none-any.whl \
     --url-map /secure/webarena/task-url-map.json \
     --snapshot-id table2-pc01-public-development-v1 \
     --evaluator-id sealed-libwebarena-compatibility-port-v1 \
     --output /secure/table2-inputs/webarena-tasks.json
   ```

   A separately extracted `test.raw.json` is accepted only with its explicit
   `--authorized-raw-json-sha256`; using the pinned wheel is the normal route
   because it authenticates both the distribution and member bytes.

2. Audit the exact export against the fixed P3 action interface:

   ```bash
   PYTHONPATH=src python3 scripts/audit_table2_webarena_task_interface.py \
     --task-export /secure/table2-inputs/webarena-tasks.json \
     --output /secure/table2-inputs/webarena-task-interface-audit.json
   ```

   For the currently registered 0--49 set this writes the measured `FAIL`
   result above. Stop here until the protocol decision is resolved. Handoff
   accepts only an audit that exactly recomputes from the authenticated export
   and has `handoff_eligible: true`.

3. On a task-compatible protocol, test the DGX as the preferred single host:

   ```bash
   PYTHONPATH=src python3 scripts/preflight_table2_webarena.py \
     --service-url-map /secure/webarena/seven-service-url-map.json \
     --live-reset-task-index 0 \
     --output /secure/table2-inputs/dgx-webarena-preflight.json
   ```

   `--skip-live-reset` is diagnostic only and can never authorize a campaign.
   The preflight must measure the pinned Python packages, Chromium, all seven
   `WA_*` origins, and a real public-task reset. If the DGX fails, run the same
   host preflight on the local x86 browser host and freeze a split deployment:

   ```bash
   PYTHONPATH=src python3 scripts/prepare_table2_split_preflight.py build \
     --local-browser-preflight /secure/table2-inputs/local-webarena-preflight.json \
     --service-url-map /secure/webarena/seven-service-url-map.json \
     --dgx-runtime-identity /secure/table2-measured/dgx-runtime-identity.json \
     --expected-dgx-runtime-identity /secure/table2-authority/dgx-runtime-identity.json \
     --bridge-identity /secure/table2-measured/bridge-identity.json \
     --expected-bridge-identity /secure/table2-authority/bridge-identity.json \
     --exchanges /secure/table2-measured/oracle-blind-bridge-exchanges.json \
     --live-reset-task-index 0 \
     --output /secure/table2-inputs/split-deployment-preflight.json
   PYTHONPATH=src python3 scripts/prepare_table2_split_preflight.py validate \
     --artifact /secure/table2-inputs/split-deployment-preflight.json \
     --service-url-map /secure/webarena/seven-service-url-map.json \
     --expected-dgx-runtime-identity /secure/table2-authority/dgx-runtime-identity.json \
     --expected-bridge-identity /secure/table2-authority/bridge-identity.json \
     --live-reset-task-index 0
   ```

   The actual and expected identity files must be distinct authorities. The
   split artifact includes a hash-chained request/response transcript and must
   report zero reward, oracle, evaluator, progress, failure-label, or other
   forbidden fields crossing into the model plane.

4. The deployment owner then supplies real evidence for exactly seven live
   capabilities and validates the
   `table2-pc01-live-deployment-v2` manifest with
   `load_pc01_live_deployment_manifest` in
   `src/web_agent/eval/table2/live_deployment.py`. This is measured evidence,
   so no fixture or repository command fabricates it. The manifest binds the
   validation-disabled BrowserGym path, six-field start-state application,
   exclusive control, reset, live observation/action mapping, safety and fault
   classification, recovery planning, efficiency accounting, the typed
   same-process sealed-page engineering broker, and cleanup on success,
   evaluator error, and runtime abort. That broker demonstrates reviewed-code
   message flow only; it is not live-campaign authority. The handoff/runner
   source attestation must include the implementation and broker sources, plus
   the exact blocked security non-claim. Production remains stopped until an
   externally authenticated process-isolation implementation and receipt
   replace that unpromotable binding. If the frozen campaign cannot reopen and
   validate the manifest and every readiness file, stop before handoff.

   The sealed evaluator evidence must call `libwebarena==0.0.4` an independently
   reviewed compatibility port, not the official evaluator. The current
   pending module/version cannot be promoted by editing its manifest or adding
   a self-authored `PASS`. An approved release must use a new non-pending
   implementation identity/source commit; run final-byte parity against the
   pinned upstream reference; obtain independent review of those exact bytes;
   bind the resulting parity/review receipts to the embedded 50-task compile
   authority; and pin their joint trust-binding digest in a later reviewed
   source commit. The current source allowlist is empty, so fully re-hashed
   caller-authored JSON still fails. The package also binds official upstream
   repository/revision and source hashes separately from runtime wheel/module
   hashes, the compatibility delta, evaluator configuration, and
   response-schema evidence. The 13 fuzzy tasks require a successful
   availability receipt for the pinned `gpt-4-1106-preview` judge and its
   frozen decoding contract. Substitution or silent fallback is forbidden.

5. In parallel, build P4 on Kaggle where the 22 GB Gold training corpus is
   already mounted, following
   `docs/P4_MEMORY_PREPARATION_AND_TRANSFER_RUNBOOK.md`. Transfer no Gold
   images. Transfer only the seven-file frozen store, preparation/provenance/
   duplicate/calibration evidence, and the external transfer manifest, then
   validate those bytes on the campaign host.

6. Only after gates 1--5 pass, prepare the clean-checkout handoff, freeze the
   campaign, validate the incomplete package, and launch. A failed task audit,
   failed host/split preflight, absent live capability record, or invalid P4
   transfer is a hard stop, not an infrastructure rerun.

The input JSON uses schema `table2-handoff-input-v1` and contains these
operator-supplied fields:

- `campaign_config`, `dependency_lock`, `resolved_task_export`,
  `webarena_task_source`, `webarena_site_url_map`,
  `webarena_task_interface_audit`, and `duplicate_audit` artifact paths. The
  handoff reopens the pinned libwebarena wheel/member bytes, resolves the
  credential-free five-token URL map independently, exactly reproduces the
  submitted task export, and recomputes the action-interface audit; a claimed
  hash in a hand-written export or audit is insufficient;
- `webarena_service_url_map`, `webarena_host_preflight`, and one registered
  `webarena_deployment_topology`. `SINGLE_DGX_HOST` accepts only a measured DGX
  host PASS and no split identities. `SPLIT_LOCAL_BROWSER_DGX_INFERENCE`
  additionally requires distinct frozen `expected_dgx_model_runtime_identity`
  and `expected_bridge_identity` authorities and a validated split preflight;
- `pc01_live_deployment_manifest` and
  `pc01_live_deployment_evidence_root`, pointing to the independently measured
  seven-capability manifest and the exact evidence package it references. The
  handoff stages only the manifest, seven readiness records, and transitively
  referenced evaluator-provenance files; symlinks, extra files, missing
  sources, or changed bytes fail closed. The same binding is revalidated during
  freeze, campaign validation, production-runner construction, and immediately
  before every ordinary browser launch;
- for `pc01_provisional`, `selection_evidence` contains exactly the immutable
  PC-01 seed-42 run contract, complete report, checkpoint-saved resolved
  config, selected checkpoint identity, registered backbone config, and
  `model_compatibility_report.json`. It must not contain three-model comparison
  JSON/CSV. The handoff independently replays PC-01's registered epoch gate,
  requires epoch 6 plus the registered checkpoint/config hashes, pins the
  compatibility report to SHA-256
  `1a8e9bb008daab5dfe893db7738daf12ee6e57e1ce2469ae29810d793030a441`,
  and revalidates its 16-row smoke semantics rather than trusting its `PASS`
  label. This report proves pre-training pipeline/head compatibility only; it
  is not a ranking metric and cannot replace the separate checkpoint-backed
  DGX receipt;
- for the PC-01 `PILOT_ONLY` evaluation, the separate
  `pc01_checkpoint_compatibility_receipt` field points to the immutable
  canonical receipt produced on the registered DGX by
  `scripts/run_pc01_checkpoint_compatibility.py`. It is not an eighth
  `MODEL_EVIDENCE_ROLES` member. Its source commit must equal the clean handoff
  commit; all 11 source-attestation rows are recomputed against the exact
  runner source, and every receipt artifact digest is cross-bound to the staged
  executable/v3 bundle and selected full report/run contract. Handoff copies
  the exact bytes to
  `<handoff>/runtime_readiness/pc01_checkpoint_compatibility_receipt.json`;
  freeze then places it at
  `<campaign>/frozen/runtime_readiness/pc01_checkpoint_compatibility_receipt.json`.
  This is distinct from the campaign-root `runtime_readiness/` directory,
  which is reserved for exactly the portable live matched-block evidence,
  receipt, and SHA sidecar. Freeze,
  package validation, runner attestation, and production startup revalidate the
  same binding before browser reset. See
  `docs/PC01_CHECKPOINT_COMPATIBILITY_RUNBOOK.md` for the DGX command and
  receipt boundary. `ENGINEERING_SMOKE_ONLY` omits this receipt, and the
  provisional receipt cannot authorize a locked-final campaign;
- only the later `three_candidate_final` selection profile accepts all three
  registered seed-42 candidate packages and comparison JSON/CSV. Before final
  comparison/handoff, the three distinct model-to-compatibility-report SHA-256
  assignments must be reviewed and committed in the source registry in
  `selection_evidence.py`; repeating a self-asserted hash in the handoff input
  is insufficient. The production registry intentionally contains only PC-01
  until the real PC-02 and PC-03 bytes exist, so final selection currently
  fails closed. Those reports remain compatibility evidence and never enter
  the ranking rule. This profile cannot be used by the current pilot and the
  checked final template cannot authorize a final campaign;
- the complete measured `environment` mapping registered by
  `table2-environment-v2` and an `evaluator` mapping with ID, version,
  repository-relative source, `model_based`, and (only when applicable) an
  evaluator `prompt_path`. These fields do not replace the separately measured
  PC-01 live-deployment capability/evaluator-provenance package;
- one `models[]` row per registered seed, containing the validation/test read
  counts and identities plus exact `artifact_paths` for
  `selected_checkpoint`, `resolved_config`, `processor_contract`,
  `e0_backbone`, `e0_resolved_config`, `e0_processor_contract`, and
  `e0_parser`; the parser row also supplies its module/attribute. The selected
  `resolved_config` must be a fully materialized JSON/YAML mapping with no
  `extends`; handoff records the original file-byte SHA-256 as
  `resolved_config_sha256` and the canonical parsed-mapping SHA-256 as
  `resolved_config_record_sha256`, while memory construction requires the
  checkpoint-saved full `config` mapping to equal that canonical mapping
  exactly;
- every `models[]` row also supplies exact `model_evidence_paths` for the
  schema-v3 export manifest, full base-snapshot manifest, processor-artifact
  manifest, processor-parity receipt, training environment, training-source
  manifest, and train-only action-value evidence. Handoff wraps those seven
  canonical JSON files in `table2-model-evidence-bundle-v1`, checks all
  internal/cross-artifact hashes, requires the schema-v3 export-manifest byte
  identity `63c01942cc653732c9e9e18639cb49bded09a82235fd4ea37fef3fad14c9fa2d`,
  rebuilds the complete registered base-
  snapshot manifest from the executable E0 directory, and requires exact
  equality. The action-value audit must cover exactly the registered 23,499-
  row Gold train split plus 608-row retry supplement (24,107 rows total). The
  parity receipt must pass the production validator with pre/post/recovery
  tensor streams, all six action classes, exact implementations, the exact
  live names and source hashes of all nine registered callables, and the five
  registered training/preprocessing sources,
  including `src/web_agent/data/recovery_transitions.py`. The bundle preserves
  `runtime_ready: false` as an evidence-only statement. Live runtime readiness
  remains controlled solely by the checkpoint-backed compatibility and
  measured deployment gates;
- one complete frozen-store `manifest.json` path per seed in
  `memory_manifests`;
- `runner.runner_entrypoint` set to the canonical
  `web_agent.eval.table2.production_runner:create_runner`, the post-training
  `runtime_integration_entrypoint`, their exact repository-relative sources,
  the primary production-runner source
  `src/web_agent/eval/table2/production_runner.py`, the pre-import bootstrap
  `scripts/run_table2_evaluation.py`, and the complete attested
  runner/integration/evaluator source list;
- `runner.pc01_operations_provider_bootstrap` containing the exact provider
  factory `module:function`, module, qualname, repository-relative source and
  SHA-256, `runtime_only` source plane, provider-contract schema version, and
  expected provider public-contract SHA-256. The factory must be an exact
  function (not a class, partial, callable object, or another function in the
  same file). Its source must be a frozen runtime-capability source and must
  not be shared with the sealed evaluator or same-process engineering broker.
  This source-plane check is reviewed-code evidence, not process isolation. The
  six live
  capability readiness `deployment_state_sha256` values are the hashes of the
  actual provider operation/config public states, not caller-chosen labels.

Under the current blocked pilot protocol, the task export must contain exactly
the registered upstream indices 0--49 and must reproduce from the pinned
upstream source bytes. An approved replacement registry requires a new
versioned protocol/exporter/audit contract before handoff. Every row supplies
benchmark task ID/version, instruction, all six reset inputs (`sites`,
`start_url`, `require_login`, the storage-state reference, `geolocation`, and
`require_reset`), full task config, and evaluator config. The tracked
`benchmarks/table2/pilot/task_manifest.json` remains the ID/order exclusion
registry; it is not treated as task content. Processor-contract inputs are
reconstructed as `ProcessorParityContract` and staged as the exact canonical
`to_dict()` JSON bytes so their raw-file SHA-256 equals `record_sha256`.

Each memory provenance manifest supplied to `build_table2_memory.py` must bind
the versioned independent recovery and final-task verification records defined
in Section 8.1, including their pre/action/post or terminal hashes and combined
bundle digest. The builder and frozen-store reload reject legacy manifests that
offer only success booleans.

After the task-interface, deployment-preflight, live-capability, and Kaggle P4
transfer gates all pass, run the following commands from the repository root,
replacing only the explicit absolute artifact/output paths. The destination
P4 store must already have passed `prepare_table2_p4.py validate-transfer`;
this host must not reconstruct it from a copied Gold corpus. The production
runner entrypoint is registered and must not be substituted:

```bash
test -z "$(git status --porcelain)"
git rev-parse HEAD
PYTHONPATH=src python3 scripts/prepare_table2_p4.py validate-transfer \
  --store-dir /secure/table2-inputs/frozen-memory/seed_42 \
  --manifest /secure/table2-inputs/transfers/seed_42.transfer.json
PYTHONPATH=src python3 scripts/prepare_table2_handoff.py \
  --repository-root "$PWD" \
  --spec /secure/table2-inputs/handoff-input.json \
  --output-dir /secure/table2-handoff/pilot-260
PYTHONPATH=src python3 scripts/freeze_table2_campaign.py \
  --repository-root "$PWD" \
  --handoff-manifest /secure/table2-handoff/pilot-260/handoff_manifest.json \
  --campaign-config /secure/table2-handoff/pilot-260/campaign.yaml \
  --campaign-dir /secure/table2-campaigns/pilot-260 \
  --resolved-task-snapshot /secure/table2-handoff/pilot-260/resolved_tasks.json \
  --environment-manifest /secure/table2-handoff/pilot-260/environment.json \
  --runner-attestation /secure/table2-handoff/pilot-260/runner_attestation.json \
  --pc01-checkpoint-compatibility-receipt /secure/table2-handoff/pilot-260/runtime_readiness/pc01_checkpoint_compatibility_receipt.json \
  --model-manifest /secure/table2-handoff/pilot-260/models/seed_42.json \
  --memory-manifest /secure/table2-handoff/pilot-260/memory/seed_42/manifest.json \
  --campaign-id table2-pc01-pilot-260-v1
PYTHONPATH=src python3 scripts/freeze_table2_campaign.py \
  --repository-root "$PWD" \
  --handoff-manifest /secure/table2-handoff/pilot-260/handoff_manifest.json \
  --campaign-config /secure/table2-handoff/pilot-260/campaign.yaml \
  --campaign-dir /secure/table2-campaigns/pilot-readiness-probe \
  --resolved-task-snapshot /secure/table2-handoff/pilot-260/resolved_tasks.json \
  --environment-manifest /secure/table2-handoff/pilot-260/environment.json \
  --runner-attestation /secure/table2-handoff/pilot-260/runner_attestation.json \
  --pc01-checkpoint-compatibility-receipt /secure/table2-handoff/pilot-260/runtime_readiness/pc01_checkpoint_compatibility_receipt.json \
  --model-manifest /secure/table2-handoff/pilot-260/models/seed_42.json \
  --memory-manifest /secure/table2-handoff/pilot-260/memory/seed_42/manifest.json \
  --campaign-id table2-pc01-readiness-probe-v1
PYTHONPATH=src python3 scripts/validate_table2_artifacts.py \
  --campaign-dir /secure/table2-campaigns/pilot-260 --allow-incomplete
PYTHONPATH=src python3 scripts/validate_table2_artifacts.py \
  --campaign-dir /secure/table2-campaigns/pilot-readiness-probe --allow-incomplete
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-campaigns/pilot-readiness-probe \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --prepare-pc01-provider-boundary-receipt /secure/table2-runtime-receipts/readiness-probe.provider-boundary.json
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-campaigns/pilot-readiness-probe \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --pc01-provider-boundary-receipt /secure/table2-runtime-receipts/readiness-probe.provider-boundary.json \
  --block-id '<first-normal-block-id-from-the-frozen-probe-schedule>' \
  --live-readiness-probe-for /secure/table2-campaigns/pilot-260
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-campaigns/pilot-260 \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --prepare-pc01-provider-boundary-receipt /secure/table2-runtime-receipts/pilot-260.provider-boundary.json
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-campaigns/pilot-260 \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --pc01-provider-boundary-receipt /secure/table2-runtime-receipts/pilot-260.provider-boundary.json
PYTHONPATH=src python3 scripts/validate_table2_artifacts.py \
  --campaign-dir /secure/table2-campaigns/pilot-260
PYTHONPATH=src python3 scripts/summarize_table2.py \
  --campaign-dir /secure/table2-campaigns/pilot-260 \
  --results-dir /secure/table2-results/pilot-260 \
  --draft-pilot
PYTHONPATH=src python3 scripts/validate_table2_artifacts.py \
  --campaign-dir /secure/table2-campaigns/pilot-260 --require-aggregates
```

The two freezes deliberately consume the same authenticated handoff bytes but
use distinct explicit campaign IDs. The readiness command must use the first
normal block ID from the frozen probe schedule and must produce the exact
three-entry target-local receipt package described in
`docs/TABLE2_LIVE_MATCHED_READINESS_RUNBOOK.md`; the target command fails before
browser reset if that evidence is absent or changed.

Every angle-bracketed provider/credential value above is a required external
deployment input; this repository does not claim that any such implementation,
credential capability, or receipt already exists. The tracked preparation mode
validates the complete frozen campaign, locked-mount absence, clean source,
live deployment and exact factory identity, then writes the deterministic
provider-boundary receipt without importing or invoking the factory. A separate
receipt is required for the probe and target because it binds that campaign's
immutable state. The receipt's scope is
`REVIEWED_CODE_ORACLE_FREE_DATAFLOW_ONLY`: it records a same-process factory and
`kernel_filesystem_sandbox: false`. It proves which typed data the reviewed
entrypoint is passed; it is not a sandbox or a hostile-code security claim.
It also cannot repair the separate page-broker gap: this source version records
`BLOCKED_EXTERNAL_PROCESS_ISOLATION_REQUIRED`, and both provider-boundary
preparation and campaign execution stop before importing the provider factory.
The commands above document the future sequence only; they are intentionally
non-runnable until a separately authenticated process-isolated broker contract
is implemented and preregistered.
The credential capability root must be an existing non-symlink tree that is
disjoint in both directions from the campaign and repository; filesystem root,
campaign/source ancestors or descendants, and symlinked leaf/parent components
are rejected. Boundary-receipt preparation never overwrites different existing
evidence: identical bytes are reused, otherwise preparation fails.

After factory return, the CLI repeats the full authority preflight and freezes a
typed `PC01ProviderInstallationReceipt` containing the expected and actual
provider public-contract hashes, exact factory/source identity, bootstrap and
pre/post campaign-state hashes, boundary-receipt hash, and only a hash of the
credential public identity. The provider registry requires and revalidates that
receipt. Before `ProductionTable2Runner` or any block is constructed, the CLI
appends it as the hash-chained `pc01_provider_installation` access-ledger event;
package validation rejects any `episode_task_load` that lacks or precedes this
event. No credential root or secret is serialized.

The `--runner` value must equal the attested `runner_entrypoint`; the runner
factory must report the exact loaded runtime identity. Before importing the
evaluation package, before constructing the production runner, and before
every block, execution requires a clean checkout at the frozen Git commit.
The guards also recheck all attested source bytes, the copied dependency lock,
environment, evaluator, task snapshot, model/E0 payload identities, provider
configuration, and memory-store identity.

Every callable supplied by the runtime integration or sealed-evaluator
bindings must resolve to a repository source path whose current SHA-256 occurs
in the frozen runner attestation. This catches accidental use of an unregistered
backend, helper, evaluator, or measurement callback. External framework/model
methods therefore need a thin attested repository
wrapper instead of being injected directly. The evaluator remains an
in-process, source-attested trusted component: the callable check and the
before/after backend-state digest are scientific integrity controls, not an OS
sandbox. Hostile attested Python could still inspect globals or arbitrary
closure object graphs; evaluating adversarial third-party code would require a
separate process/container and is outside this pilot's assurance scope.

There is no additional Table 2 model seed: repeat neither the memory build nor
the browser campaign for seeds 43--44. The only matched model seed is 42.

The source-attested integration factory returns an `EvaluationRuntimeBinding`
containing one `SeedRuntimeBinding` per matched seed and a
`WebArenaRuntimeBinding` for each episode. Each seed binding must supply the
typed frozen episode-state resetter, which resets and hashes every shared
model/provider/recovery/embedding backend before each system episode, and the
loaded frozen checkpoint and unadapted backbone backends, the exact E0 parser
identity, `parameter_fallback_backbone_sha256` equal to the frozen E0 backbone
payload SHA-256, the bounded
oracle-blind recovery planner, and complete efficiency measurement callbacks.
The selected-checkpoint backend itself must expose the exact P4
`memory_embedding` forward. `ProductionRunner` source-attests that callback
and constructs the typed validating provider internally, bound to the
selected-checkpoint and processor-contract SHA-256 values; the integration
cannot inject a second embedding provider or independently claim its
checkpoint identity.
For each ordinary task it supplies version-pinned WebArena mappers, a
screenshot-byte provider, a typed frozen environment-state digester, the
frozen infrastructure classifier, a typed frozen page-settle policy, and a
typed frozen reset-state attester. It must also supply the typed frozen
action-safety policy whose source equals the attested runtime integration and
whose ID, version, and destructive-action rule equal the environment manifest;
the binding is mandatory rather than an optional adapter callback. The same is
true of the typed manual-rescue guard: its ID, version, evidence mode, and source
must equal the environment and runtime-integration attestations. The reset
attester's source hash also equals the attested runtime-integration source. The
attester
receives only episode/task/benchmark/start-state-hash/reset-seed identity and
returns source-bound service/account/database/start-state hashes after reset.
It also supplies a transition verifier that writes directly
through the write-only `SealedVerifierWriter` capability, and a final evidence
writer. The production runner
passes the episode's explicit runtime directory to the adapter; external
mappers cannot choose the policy-visible screenshot path. The transition
verifier receives only a detached, recursively immutable observation
snapshot—not the live browser environment.
No execution-result mapper exists in the production binding: trusted adapter
code extracts the post-observation from the pinned Gym tuple and permanently
discards its reward, termination, truncation, and `info` slots before any
integration callback can run.
The adapter rejects any snapshot mutation or any difference between the
registered live-state digests taken immediately before and after the callback.
Neither verifier callback may return full evaluator truth to the runtime.
Construction fails before the first reset if any object, byte hash, seed,
version, processor mapping, memory store, or attested identity differs.
The configured `locked_benchmark_mount/` must also be absent, unmounted, and
unreadable throughout this development pilot; its frozen negative attestation
is rechecked before integration and execution.

### Phase A - Receive and authenticate remote artifacts

For every candidate run, collect:

- run contract;
- complete epoch CSV and report;
- all retained epoch checkpoints;
- resume/interruption history;
- source-validation and diagnostic reports;
- exact fully materialized resolved-config artifact, retaining both its
  payload-byte and canonical-mapping identities;
- environment manifest;
- code/config/data/checkpoint hashes.

Checks:

1. The run reached a documented terminal state.
2. Every CSV epoch has a matching retained checkpoint.
3. Data counts and sources match the v2.8 contract.
4. Original validation alone controlled checkpoint selection.
5. Locked-test rows read equals zero.
6. The retained checkpoint's saved full config exactly equals the canonical
   selected resolved-config mapping.
7. No undocumented protocol deviation occurred.

The resolved-config artifact is exported after selection from the selected
checkpoint's saved full `config` mapping as canonical JSON. It must not be
reconstructed from a later source checkout or used to mutate the completed
training run. The handoff then compares it with the registered candidate
configuration and with the checkpoint before accepting either identity.

Failure of any check blocks model comparison.

### Phase B - Complete the three-candidate validation comparison

Candidates:

- Qwen2-VL-2B-Instruct;
- Qwen2.5-VL-7B-Instruct;
- InternVL3.5-8B-HF.

For seed 42:

1. Apply the registered eight quality gates per epoch.
2. Within each candidate, select the eligible epoch with the highest original-
   validation outcome MCC; an exact tie selects the earlier epoch.
3. Compare the three candidate winners using this exact order: higher original-
   validation outcome MCC, higher recovery-outcome MCC, higher action macro-F1,
   lower ECE, then lexicographically smaller model ID.
4. Produce the fail-closed comparison script/report.
5. Select the backbone using validation evidence only.

No test or final browser score may influence this decision.

### Phase C - Promote one seed-42 backbone without retraining

1. Preserve all completed PC-01, PC-02, and PC-03 seed-42 terminal runs,
   including negative results.
2. Replay the registered validation-only three-candidate comparison.
3. If PC-01 wins, reuse epoch 6 without retraining. If another candidate wins,
   retain the PC-01 pilot as engineering evidence and bind the actual winner.
4. Do not add seeds 43--44 or modify architecture, thresholds, labels, or loss.
5. State explicitly that the final Table 2 measures task uncertainty but not
   model-seed uncertainty.

### Phase D - Freeze the final model family and protocol

Freeze and hash:

- the selected seed-42 checkpoint;
- model/backbone revisions;
- fully materialized resolved configs with both exact payload-byte and
  canonical-mapping SHA-256 identities; require each checkpoint-saved full
  config to equal its selected resolved-config mapping exactly;
- automatically derived train-only P4 calibration evidence and thresholds;
- code commit;
- dataset/supplement/overlay/split manifests;
- metric implementations;
- benchmark and final task manifest;
- success/progress oracle;
- E0-E3 configs and prompts;
- budgets and environment;
- statistical plan;
- paper claim registry.

This freeze occurs before any locked component test or final scored browser
task is opened.

### Phase E - Build and freeze the seed-42 P4 index

For the selected seed-42 checkpoint:

1. Encode only eligible successful training recovery episodes.
2. Validate the independent versioned recovery and final-task evidence,
   recompute the causal pre/action/post and canonical record hashes, and close
   their digests into the identity-bound verification bundle.
3. Apply provenance, final-success, task, exact, and near-duplicate filters.
4. Use the exact inference-only post-action memory-task-adapter tensor from the
   selected checkpoint, write normalized float32 embeddings, and bind both
   resolved-config hashes.
5. Derive `table2-memory-calibration-evidence-v1` automatically by leave-one-
   out, runtime-equivalent filtered top-3 retrieval; derive relevance from the
   registered matching-verified-recovery-strategy proxy, while retaining the
   executed action as provenance, and replay the nested threshold selection.
6. Validate zero validation/test/locked-test reads and reject any caller-
   supplied calibration scores or labels.
7. Export memory schema, counts, exclusion report, the per-item independent-
   verification digest closure, full `verification_evidence.json`,
   `calibration_evidence.json`, compact `threshold_calibration.json`, and
   complete SHA-256 closure.
8. Reload the store and replay verification bundles/manifest closure, frozen
   endpoint fields, item/embedding hashes,
   exclusions, ranking, relevance derivation, and threshold selection while
   retaining the source-provenance commitments.
9. Disable writes permanently for final evaluation.

The index is not tuned using locked-test retrieval results. At the end of this
phase, seal one complete campaign manifest covering every selected checkpoint,
code/config/data/supplement/overlay/split artifact, seed-specific memory index,
task list, oracle, environment, metric implementation, statistical analysis,
and paper-claim rule. No locked component test or scored browser task may be
opened until this all-artifact manifest validates.

### Phase F - Final compatibility smoke and campaign reseal

Before any locked-test access:

1. Load the single frozen seed-42 checkpoint and matching memory index.
2. Run one non-scored development episode per E0-E3.
3. Verify environment reset, logs, hashes, feature switches, prompt/provider
   parity, matched budgets, and paired-block validation.
4. Verify E2 performs no retrieval and E3 performs no writes.
5. Confirm no final scored task or locked component-test row is touched.

Only compatibility failures may be repaired; behavioural tuning remains
forbidden. Any repair invalidates the current campaign manifest. After repair,
rerun the development smoke, regenerate every affected hash, and reseal the
complete manifest before proceeding. This phase ends with the final immutable
campaign identity.

### Phase G - One frozen locked component-test campaign

1. Verify every frozen hash in the final campaign manifest.
2. Load the locked Web-Gold test once under one preregistered batched campaign.
3. In that single opening, evaluate only the already-preregistered seed-42
   candidate checkpoints and seed-42 winning-backbone baselines/ablations
   needed for Table 1; do not add seeds 43--44.
4. Do not inspect intermediate test results or tune between variants.
5. Export predictions, supports, confusion matrices, calibration, retrieval
   diagnostics where defined, and both valid and invalid bbox denominators.
6. Record the exact access time, reason, manifest hash, and evaluation order.
7. Never retrain, retune thresholds, change architecture, or add an ablation
   after observing the locked results.

Where operationally possible, keep locked component outputs sealed from the
runtime/model developers until the frozen paired Table 2 campaign has been
launched or completed. Whether sealed or viewed, they may not cause a runtime,
model, memory, threshold, prompt, metric, or task-manifest change.

The supplement has no second independent locked test. Supplementary analyses
from this campaign are additional views of the same test evidence, not an
independent replication. In particular, do not claim that all six recovery
strategies generalize unless the locked supports and results directly justify
that narrower claim.

If a bug invalidates the campaign, preserve the failed package, document the
root cause, obtain thesis-team approval, assign a new protocol ID, and report
the repeated access transparently. No repair is allowed under the old campaign
identity.

### Phase H - Execute paired E0-E3 evaluation

1. Run E0-E3 using only the promoted winning-backbone seed-42 checkpoint.
2. Generate the preregistered task × repetition × system schedule with
   `matched_model_seed=42` fixed in every paired block.
3. Counterbalance execution order.
4. Run each task from the registered start state.
5. Validate every episode package immediately for schema completeness.
6. Retry only preregistered infrastructure-invalid paired blocks, for every
   system under the same rule; the next whole-block ordinal replaces the prior
   ordinal and execution stops at the frozen maximum.
7. Never manually rescue an agent.
8. Preserve failed, timed-out, aborted, and interrupted episodes.
9. Keep memory frozen and read-only.

### Phase I - Calculate metrics and paired statistics

1. Validate task pairing and denominator counts.
2. Calculate all Section 11 metrics.
3. Produce point estimates and 95% confidence intervals.
4. Run E0-vs-E1, E1-vs-E2, and E2-vs-E3 paired tests.
5. Apply the registered multiple-comparison correction.
6. Report practical cost/latency changes.
7. Run the primary valid matched-block analysis, an agent-intention-to-treat
   summary within those blocks, and the full scheduled/rerun/exclusion audit.

### Phase J - Run registered diagnostic experiments

Required:

- E3 retrieval-only without intervention;
- shuffled/random memory;
- successful-recovery-only vs broader memory;
- full input vs image/text occlusion sensitivity;
- per-failure-type and per-strategy recovery analysis;
- trigger-source-stratified recovery results;
- learned-trigger-only vs executor-error/loop-safeguard trigger ablation if a
  P1-specific contribution is claimed;
- grounding/target-size and action-class breakdown;
- separately labelled oracle-trigger/oracle-recovery upper bounds when
  feasible; they may never be substituted for primary E2/E3;
- causal input tests: remove executed action, swap post-state, and action/post
  mismatch controls.

These diagnostics must not alter the frozen primary Table 2 result.

### Phase K - Manual audit and error analysis

Use a preregistered blinded sample containing:

- successes;
- failures;
- successful and failed recoveries;
- E2/E3 disagreements;
- memory-help and memory-harm cases;
- environment-failure cases;
- bbox/parameter failures;
- loops and unnecessary interventions.

Record adjudicator agreement and retain the sample manifest. Do not relabel the
evaluation dataset silently.

### Phase L - Fill Table 2 and freeze paper claims

Table 2 remains `N/R` through every `PILOT_ONLY` run. It may be filled only
from the later frozen final-paper campaign after all final episode packages,
the final blinded audit, and final package validation pass.

Permitted claims require:

- E1 improvement over E0 for the trained project policy relative to the
  unadapted common-schema baseline; call it a pure training effect only if the
  architecture/interface-matched control requirement is met;
- E2 improvement over E1 for the registered diagnosis-and-recovery stack; a
  learned-P1-specific claim additionally requires the trigger-source ablation;
- E3 improvement over E2 for corrective memory;
- confidence intervals and paired evidence;
- visible latency/action/model-call cost;
- explicit negative or unsupported results.

Do not claim:

- operational recovery from offline recovery-outcome MCC alone;
- memory benefit from memory-update classification or Recall@K alone;
- causal reasoning from pre/post screenshots without controlled causal tests;
- superiority to a paper evaluated on another benchmark;
- all six strategies generalize without locked-test support.

---

## 17. Verification and acceptance tests

### 17.1 Four-pillar invariant tests

| Test | Required result |
| --- | --- |
| Change only forbidden `state_after` during pre-action inference | P3 action/bbox output remains unchanged |
| Change executed action/post-state during post-action assessment | P1/P2 transition input and logged assessment change accordingly |
| Remove executed recovery/post-recovery state | Recovery-outcome evaluation rejects the incomplete transition |
| Change only sealed oracle failure/progress outputs | Recovery trigger, strategy, memory query, and action remain unchanged |
| Change only a sealed memory-relevance label | Retrieval ranking, admission, advice, and intervention remain unchanged |
| Attempt a pre-action memory query | Hard failure before inference |
| Supply a non-training, unsuccessful, unflagged, quarantined, same-task, or duplicate memory item | P4 build/query fails closed |
| Supply only declared recovery/final-success booleans, omit versioned independent verification, alter a pre/action/post or terminal evidence hash, or reattach evidence to another sample/task/episode | Eligibility fails before memory construction |
| Alter/reseal a frozen recovery/final verification record, its item digest, or the sorted store-level closure; remove `verification_evidence.json`; or bind a record to another task/episode/step | Frozen-store load fails before calibration replay or retrieval |
| Supply calibration scores, pair rows, or relevance labels externally | Memory build has no such input and fails rather than accepting them |
| Alter an eligible item, normalized embedding, exclusion, top-3 rank/score, derived calibration label, or nested threshold result | Frozen-store calibration replay fails |
| Reuse one embedding across different processed batches, or return a receipt bound to another transition, processor, or checkpoint | P4 query fails before retrieval/intervention |
| Compare E1/E2/E3 pre-action calls under identical input/RNG | Decisions are identical until an enabled post-action feature changes environment state |
| Query irrelevant memory below threshold | E3 abstains and matches E2 decision path |
| Compare E2 and E3 before E3's first admitted intervention | Decisions and actions are identical under matched inputs/RNG |
| Attempt evaluation-time memory write | Hard failure |

### 17.2 Runtime unit tests

- contract validation and serialization;
- feature-switch isolation;
- checkpoint/config hash checks, including exact resolved-config payload bytes,
  canonical mapping identity, and checkpoint-saved config equality;
- normalized bbox and viewport conversion;
- provider output validation for all actions;
- recovery validity and budget accounting;
- loop/repeated-state rule;
- environment-error taxonomy;
- metric formulas with hand-calculated fixtures covering false triggers,
  multiple attempts in one incident, loops, capped failures, and
  infrastructure-invalid paired blocks;
- empty denominator produces `N/A`;
- paired schedule and missing-task detection;
- all-system exclusion/rerun enforcement for infrastructure-invalid blocks;
- E3 no-memory shadow-decision capture;
- log and manifest completeness.

### 17.3 Browser integration tests

- reset, observe, execute, wait, verify, and close;
- deterministic click/type/select/scroll/navigate/key fixtures;
- missing/stale element handling;
- popup/navigation/timeout handling;
- viewport consistency;
- no-op and wrong-target detection;
- backtracking verification;
- clean episode termination.
- pre-browser setup overrun rejection at the registered 120-second boundary;
- 600-second clock start immediately before WebArena environment reset;
- missing, wrong-source, malformed, or cross-system-different WebArena reset
  receipts rejected before aggregation.

### 17.4 Fault-injection tests

- deliberately wrong bbox;
- action mismatch;
- no visual/state change;
- disappeared element;
- navigation error;
- repeated page/action loop;
- misleading but irrelevant memory;
- unavailable site/browser crash;
- exhausted action/recovery budget.

Each fault must be attributed to agent, provider, memory, verifier, or
environment rather than collapsed into one failure bucket.

---

## 18. Stop conditions

### 18.1 Stop implementation scoring if

- runtime/training processor parity fails;
- any future/oracle field enters pre-action inference;
- a verifier/oracle result can alter the primary recovery trigger, strategy,
  memory query, provider, or action;
- primary P4 retrieval occurs before an executed action and post-action
  diagnosis;
- action parameters differ across E1-E3;
- an action or recovery can silently fail without a log;
- the verifier is undefined or policy-dependent;
- memory exclusions or read-only enforcement fail;
- metric fixtures fail;
- one development episode cannot produce a complete package;
- locked/final tasks were accessed for tuning.

### 18.2 Stop final E0-E3 evaluation if

- candidate training or validation-only selection is incomplete;
- required seed/checkpoint evidence is missing;
- frozen hashes do not match;
- task/start-state manifests differ;
- reset/environment conditions are inconsistent;
- E2 retrieves memory or E3 writes memory;
- E2 and E3 diverge before the first admitted memory intervention under
  matched inputs and randomness;
- hidden/manual rescue occurs;
- budgets or prompts differ unexpectedly;
- required logs are missing;
- a final-task result has already influenced a code/config change.

### 18.3 Stop a paper claim if

- the corresponding pillar lacks its required contrast or mechanism control;
- sample support is zero or too small and undisclosed;
- a metric denominator is undefined;
- benchmark protocols differ;
- uncertainty and cost are omitted;
- a component prediction is being substituted for executed behaviour.

---

## 19. Risk register

| Risk | Threatened pillar/evidence | Mitigation |
| --- | --- | --- |
| Post-action leakage into pre-action policy | P2/P3 | Separate APIs, schemas, prompts, and invariance tests |
| Provider supplies an oracle action/target | P3 | One frozen provider, full input/output logs, no reference data |
| Live mapper omits observation-bound target/option evidence | P1/P3 | Version-pin and smoke the WebArena mapper; fail closed on missing, stale, task-mismatched, or self-attested evidence |
| Offline recovery prediction mistaken for execution | P1 | Require concrete recovery and independent post-recovery verification |
| Same-task or evaluation memory leakage | P4 | Train-only manifests, duplicate filters, query-time exclusions, hashes |
| Memory advice overwhelms a small backbone | P4/Table 2 | Relevance gate, abstention, E2/E3 regression and harm analysis |
| Live-site drift favours one system | Table 2 validity | Counterbalanced order, environment logging, controlled primary benchmark |
| Different action/recovery budgets | Fairness | Shared hard limits and startup assertions |
| Weak grounding causes all runtime failures | P3 | Provider/grounding error taxonomy, oracle-target upper bound, bbox analysis |
| Success oracle is noisy or model-dependent | Table 2 validity | Frozen benchmark semantics; reviewed compatibility-port delta; judge availability/configuration receipt; audit sample; version/prompt hashes |
| Oracle feedback leaks into recovery decisions | P1/P4 causal validity | Sealed scoring channel, trigger-source logs, oracle-blindness tests |
| Environment failures hidden as agent failures | Table 2 validity | Separate taxonomy and intention-to-treat/sensitivity reporting |
| Final task used for prompt tuning | Generalization | Frozen dev/final manifests and access ledger |
| Unfavourable seeds or runs omitted | Statistical validity | Preserve and report every registered terminal run |
| Runtime work alters active training | Reproducibility | Additive modules, frozen training commit/config, separate artifacts |

---

## 20. Deliverables checklist

### While training runs

- [x] The original 0--49 development/exclusion registry and locked-boundary guard are frozen
- [x] The original 0--49 task/action-interface audit records 47 incompatible answer tasks and fails closed
- [ ] User-approved, preregistered task-interface resolution and compatible 50-task export
- [ ] Live WebArena host/split-deployment preflight PASS
- [ ] Complete seven-capability live deployment and sealed-evaluator evidence PASS
- [ ] Live WebArena mapper emits validated `observable_select_controls` and `recovery_target_evidence`
- [x] Runtime schemas and contracts
- [x] E0-E3 feature-switch configs and assertions
- [x] Observation/training parity tests
- [x] Common action-parameter provider
- [x] Browser executor interface and independent sealed verifier
- [x] Pinned, validation-disabled BrowserGym/WebArena production wrapper
- [x] Concrete recovery resolver
- [x] Frozen train-only memory builder/retriever/intervention
- [x] Episode runner and complete logging
- [x] Exact metric/statistics implementation
- [x] Deterministic fixture and 15-scenario recovery smokes PASS
- [x] Production evaluation runner and live-evidence revalidation gates
- [x] Portable matched E0--E3 live-readiness receipt and dispatch guard
- [x] Same-process broker limitation frozen as reviewed-code engineering evidence; production blocked before provider import
- [ ] Separately authenticated process-isolated page broker and deployment receipt
- [ ] Isolated live first-normal-block readiness probe PASS and target-local receipt frozen
- [ ] Selected-checkpoint WebArena compatibility smoke PASS
- [ ] Checkpoint, memory, and measured environment attestations frozen

### After training

- [ ] All three seed-42 candidate packages authenticated
- [ ] Validation-only winning backbone selected
- [ ] Final model/config/data/protocol hashes frozen
- [ ] Selected seed-42 train-only memory index frozen
- [ ] Final E0-E3 compatibility smoke PASS
- [ ] Final all-artifact campaign manifest resealed
- [ ] One locked component-test campaign complete
- [ ] Paired E0-E3 episodes complete
- [ ] Metrics, confidence intervals, and paired tests complete
- [ ] P2 causal/modality and P4 memory diagnostics complete
- [ ] Efficiency and environment-failure reports complete
- [ ] Blinded manual audit and error analysis complete
- [ ] Final Table 2 filled from validated artifacts
- [ ] Paper claims frozen and unsupported claims removed

---

## 21. Definition of done

Table 2 is complete only when all of the following are true:

1. The winning backbone was chosen from validation evidence only.
2. The selected seed-42 checkpoint hash exists and matches its authenticated
   validation-only selection package.
3. E0-E3 used identical tasks, start states, environment, provider, evaluator,
   and budgets.
4. P2 temporal barriers and P3 grounding contracts passed their tests.
5. P1 produced and verified concrete browser recoveries.
6. P4 used an immutable train-only successful-recovery index with zero final-
   evaluation writes.
7. Complete paired episode packages exist for all registered runs.
8. Every metric has a frozen definition, numerator, denominator, and
   uncertainty estimate.
9. Memory benefit is measured by E2 vs E3, not inferred from retrieval alone.
10. Efficiency, environment failures, negative results, and limitations are
    reported.
11. All code, data, model, memory, task, oracle, and environment artifacts are
    versioned and hashed.
12. No final result was used to tune the model or evaluator.

The success criterion is not a universal accuracy target and does not require
a positive result. The required Q1 evidence is a reproducible, preregistered
paired estimate under the registered contrasts and mechanism controls, with
uncertainty, costs, negative or null findings, and pillar-specific limitations
reported honestly while all four pillars remain operational and intact.
