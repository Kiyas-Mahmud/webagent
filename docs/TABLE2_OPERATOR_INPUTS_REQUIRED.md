# Table 2 Operator Inputs Required Before the PC-01 Pilot

Status: `WAITING_FOR_EXTERNAL_INPUTS`

Evidence role: `OPERATIONAL_CHECKLIST_ONLY`

Paper Table 2 status: `N/R`

This checklist records the decisions and deployment evidence that cannot be
created honestly by repository code. Completing a source validator, local
fixture, or self-authored receipt does not satisfy any item below.

## 1. Development-task interface decision

Choose one option explicitly before any WebArena outcome is observed:

- **Option A (recommended):** approve the 50 non-mutating, page-state-scored
  candidate indices in `docs/TABLE2_PILOT_TASK_INTERFACE_DECISION.md`.
- **Option B:** authorize a material protocol change that adds an
  answer/termination action. This is not recommended because PC-01 was not
  trained with that action and will not be retrained.

No tracked task manifest treats Option A as approved. Approval permanently
excludes both the historical indices 0--49 and the replacement 50 from the
later final-paper campaign. The selected tasks still require a deployment
safety review and identical E0--E3 reset-state evidence before registration.

## 2. Kaggle execution authorization

The project owner must complete Kaggle authentication in their own browser.
Do not paste a password, token, OAuth URL, verification code, or API key into
Git, a campaign artifact, or chat. After authentication, the source-pinned
prepare-only job may read only:

- `final_data_set_40k/split_train.json` from `web-gold-40k`;
- `final_data_set_40k/data/supplement_train.json` from `gold-40k-retry`.

Validation and test reads must remain zero. The prepare-only output is a review
queue, not a frozen P4 store. Independent recovery/final-success evidence, the
joint Gold-train/WebArena duplicate audit, checkpoint-backed 768-dimensional
embeddings, and the train-only leave-one-out threshold remain required.

## 3. DGX model-plane access

Provide either the exact authorized DGX SSH target and key path or run the
source-frozen commands on the DGX and return their complete compact evidence
packages. Do not commit private keys or credentials.

The DGX gate must authenticate:

- PC-01 epoch 6, seed 42 checkpoint SHA-256
  `9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a`;
- checkpoint-saved resolved configuration identity
  `d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f`;
- Qwen2-VL-2B base snapshot revision
  `895c3a49bc3fa70a340399125c650a463535e71c`;
- processor parity, E0 adaptation absence, E1--E3 checkpoint parity, CUDA
  execution, and the exact 768-dimensional P4 representation.

The local processor-only PASS cannot substitute for the DGX CUDA/model-forward
receipt.

## 4. Live WebArena deployment

The deployment owner must supply all seven reachable service origins, the
external credential capability, deterministic service/reset state, and the
reviewed evaluator implementation. Secrets remain outside the repository and
outside runtime-visible records.

Required evidence includes:

- exact pinned BrowserGym, WebArena, Playwright, Chromium, and controller
  identities;
- task content/start states and service URL binding for the approved registry;
- reset fingerprints and exclusive-input/manual-rescue evidence;
- the complete concrete BrowserGym plus sealed-evaluator adapter-factory source
  closure;
- the concrete live factory binding for the implemented child-owned sealed
  episode-finalization/write-only channel. The generic six-operation bridge
  and orchestration-only finalizer are local architecture evidence only and
  cannot give the runtime an evaluator or verifier-writer capability;
- the preregistered safe, non-persistent calibration-action manifest covering
  all six P3 action classes separately in the ordinary and recovery phases,
  its inline typed action-safety approvals, a registered safe injected-runtime-
  failure cleanup probe for every task, and an independent typed per-task
  reset-state non-persistence audit receipt;
- a complete outcome-blind monotonic timing artifact covering all registered
  broker/settle operations, normal shutdown, and shutdown after injected
  runtime failure for every approved development task, produced before
  outcomes are observed and rederived by the registered timeout contract (a
  caller-selected timeout or configured upper bound is rejected), together
  with immutable typed harness-source and measurement-source receipts;
- an external authority cross-binding of the exact calibration artifact,
  typed manifests/receipts, ordered task registry, topology, browser host and
  configuration, preflight, dependency lock, service map, budgets, and
  collector source closure. The local typed bundle is non-authorizing by
  design and `MEASURED_CALIBRATION_REPLAY_ONLY` is never production evidence;
- compatibility-port parity and independent evaluator review;
- a successful live first-normal-task matched E0--E3 readiness block.

The separate local x86 environment currently proves only package import and
Chromium launch. It does not prove service reachability, login/reset behavior,
evaluator parity, or campaign readiness.

## 5. Independent deployment authority

Scope-decision status: `AWAITING_PROJECT_OWNER_RATIFICATION`. The supplied
research-locked execution plan requires source/request/response hashes and a
sealed split deployment, but it does not explicitly require an independent
Ed25519 attestor or global replay anchor for the engineering pilot. The
repository currently applies this stronger requirement conservatively and
fails closed. Before pilot dispatch, the project owner must explicitly choose
whether Section 5 remains mandatory for `PILOT_ONLY` or becomes final-campaign
hardening; changing that choice requires a reviewed source/protocol update,
not a runtime flag.

The lab must nominate an independent deployment attestor and provide its
Ed25519 public key, authority/key identifiers, validity interval, permitted
host roles/phases, and independently reviewed measurement-profile hash for
source registration. The private key must remain outside Git, the local/DGX
runtime, browser/evaluator processes, and campaign artifacts.

The independent attestor—not the model runtime—must observe and sign the
startup, model-load, pre-block, and post-block measurements, including the
runtime-visible request/response transcript commitment. A repository-generated
key, locally copied JSON, or a self-consistent SHA-256 file is not independent
deployment authority.

The verification-only command is
`scripts/validate_table2_deployment_authority.py`. It supports the four exact
phases, Ed25519 verification, and cryptographic replay of every consumed
receipt in one local ledger. It does not generate or accept private keys. The
packaged registry currently has zero authorities, and the local ledger is
explicitly neither a global replay anchor nor dispatch authority. Registering
the reviewed public key and wiring authenticated receipts into handoff,
startup, and physical-block dispatch remain required source changes after the
lab supplies the independent authority.

Until a real authority is registered and its signed evidence validates, live
dispatch remains blocked even when all local tests pass.

## 6. Final-model promotion inputs

PC-02 and PC-03 continue independently. When both finish, supply their exact
seed-42 packages for the registered validation-only comparison. PC-01 pilot
evidence cannot preselect the final backbone. If PC-01 does not win, its pilot
remains engineering evidence only.

## 7. Final-only preregistration lock

This section is not a PC-01 pilot-launch input and does not make the final
template runnable. It is a separate final-paper gate. Before any final-task
outcome is opened—including an episode success/failure label, sealed recovery
or progress judgment, aggregate, or outcome-audit/adjudication label—the study
owner must freeze a versioned, hash-bound, machine-executable decision rule.
Its timestamp and final-task access ledger must establish that the freeze
preceded the first such read. Until every item below is frozen, final-task
outcome access is prohibited and all paper claims and Table 2 cells remain
`N/R`.

- [ ] **Primary estimand hierarchy:** name the task-success estimand, order the
  confirmatory contrast(s), distinguish any gatekeeping from a single Holm
  family, and label every remaining contrast secondary or descriptive. The
  hierarchy must preserve E1--E2 as the executed diagnosis/recovery increment
  and E2--E3 as the frozen train-only memory increment; it must not redefine
  the four pillars.
- [ ] **Direction and alpha:** record the right-minus-left sign convention,
  the effect direction required for support, the already planned two-sided
  exact-test alternative, and the numeric familywise alpha. These values may
  not be inferred later from a point estimate or from the presence of a 95%
  interval.
- [ ] **Exact tests and interval interpretation:** retain the task-level exact
  sign-test unit and freeze the Holm family and adjustment procedure for the
  registered E0--E1, E1--E2, and E2--E3 task-success tests. Also freeze the
  task-cluster bootstrap construction and whether a bootstrap 95% confidence
  interval is descriptive estimation, a separate support gate, or part of a
  joint gate with the Holm-adjusted test. State the outcome when the adjusted
  test and interval disagree. E0--E3 remains outside this Holm family.
- [ ] **Claim-status mapping:** define an executable mapping from effect sign,
  adjusted p-value, interval, and every registered condition to `SUPPORTED`
  versus `NEGATIVE_OR_NULL`. A valid unconditional contrast that is null,
  adverse, or misses the frozen support rule must be `NEGATIVE_OR_NULL`, not
  hidden as `N/R` or relabelled `NOT_SUPPORTED`. Reserve `NOT_SUPPORTED` for a
  preregistered unmet identification/provenance condition on a claim whose
  registry permits that status.
- [ ] **Zero-information cases:** preserve ties in effect and interval reports
  and set the exact sign-test p-value to `1` when there are zero non-zero task
  effects. For zero initiated recoveries or zero admitted P4 interventions,
  show numerator and zero denominator, report the affected rate and bootstrap
  interval as `N/A`/not estimable, and do not permit the relevant effect or
  companion claim to become `SUPPORTED`. For the unconditional E1--E2 or
  E2--E3 contrast, an otherwise valid final experiment remains
  `NEGATIVE_OR_NULL`; any conditional diagnostic may use `NOT_SUPPORTED` only
  if its minimum-opportunity condition was frozen before outcome access.
- [ ] **Contrast semantics:** freeze E0--E1 as a contextual trained-policy
  comparison unless the separately registered architecture/interface-matched
  control passes; never call the contextual result a pure training effect.
  Freeze E0--E3 as a descriptive total-system comparison only, outside the
  confirmatory Holm family and without attribution to P1, P2, P3, or P4.
- [ ] **P1--P4 diagnostic thresholds:** for each companion claim, freeze its
  authoritative input population, aggregation unit, required metrics, support
  direction, numeric cutoff(s), minimum denominator/coverage, interval rule,
  and hard invariants. The P1 rule must cover failure detection/diagnosis and
  assessment of executed recovery; P2 must cover temporal and modality causal
  controls; P3 must cover six-class action, grounding/parameters, and rejected
  output accounting; P4 must cover train-only retrieval/admission, provenance,
  and zero-write integrity. These thresholds remain companion diagnostics and
  cannot substitute for the paired browser contrasts.
- [ ] **Pilot firewall:** document the external scientific basis for every
  choice above and attest that no PC-01 pilot outcome—point estimate,
  confidence interval, p-value, discordance count, recovery/intervention rate,
  or companion-diagnostic score—was used directly or indirectly to choose or
  revise the rule. The rule may not be selected because it makes the PC-01
  pilot favorable. Pilot outputs remain `PILOT_ONLY`, and their paper claims
  and Table 2 cells remain `N/R` regardless of their values.

## Pilot launch rule

Under the current conservative implementation, the provisional 60 controlled
recovery episodes and 200 normal WebArena episodes may begin only after
Sections 1--5 pass and the frozen package validator authorizes the matched
readiness block. If the owner makes Section 5 final-only, that decision must
first be implemented and revalidated in a clean source snapshot. The pilot
remains `PILOT_ONLY`; it cannot populate the paper's Table 2.
