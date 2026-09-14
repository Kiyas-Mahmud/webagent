# Hybrid-agent TODO — execute in order

**Immediate Task 1 pilot:** [240-example mini dataset and checklist](TASK1_MINI_DATASET_PILOT.md).
Metadata analysed and six-action selection prepared; no agent inference run.
Selected candidates: Browser Use, Agent S2 and WebVoyager assessment components.
See the [architecture and data examples](TASK1_AGENT_COMPARISON_ARCHITECTURE.md).
Next: pin system identities and implement the shared input builder and first adapter.

**New evaluation direction, 2026-09-14:** follow
[NEXT_EVALUATION_PLAN.md](NEXT_EVALUATION_PLAN.md). First complete the two-dataset
comparison using the same frozen assessment models; afterward evaluate an
existing agent with diagnosis/recovery and then memory. The checklists below
remain historical. The new plan does not launch inference or authorise access
to locked-test data.

**Lab pause / home handoff, 2026-09-13:** resume from
[TABLE2_HOME_SESSION_HANDOFF.md](TABLE2_HOME_SESSION_HANDOFF.md) on branch
`table2/lab-handoff-20260913`. Table 2 work is archived for continuation; no
new evaluation is ready or running. The latest train-replay/export finding and
the unexecuted epoch-0 proposal are recorded there. Compact diagnostic summaries
are now in `docs/evidence/lab-handoff-20260913/`; full assets remain in the lab.

**First bounded task complete:** [six-action input-context diagnostic and atomic-action candidate](TABLE2_SIX_ACTION_CONTEXT_DIAGNOSTIC.md).
The context diagnostic completed: all eighteen raw outputs selected CLICK even
without advice/history. The compact six-call candidate emitted SCROLL/NAVIGATE
but invalid arguments; TYPE still became CLICK. All four existing parameter
fallbacks explicitly rejected. Total: 28 generations, six trained forwards,
zero browser episodes or memory changes. The planning issue is not fixed;
the diagnostic candidate is not promoted and no new final evaluation ran.

**Checkpoint-series audit complete, 2026-09-13:**
[all ten epochs, causes and agent implications](TABLE2_ALL_CHECKPOINT_ACTION_AUDIT.md).
All ten six-class heads and `last.ckpt` were inspected on CPU. The recurring
CLICK/SCROLL/NAVIGATE confusion is present across training; epoch 6 was selected
for outcome MCC under gates that do not require per-class recall. No checkpoint
swap, inference or final evaluation ran. The next proposed task is the bounded
six-action input-contract diagnostic described in that report.

**Latest investigation, 2026-09-13:** [training/runtime alignment and two-stage
diagnostic](TABLE2_TRAINING_RUNTIME_ALIGNMENT_AUDIT.md). The trained model did
include TYPE and SCROLL; the hybrid all-CLICK JSON came from the separate base
generator. Nine saved-state diagnostic generations completed. The generator
planned text entry in words, but still emitted CLICK actions. No new browser
episodes ran and the planner problem remains unresolved. The linked checklist
records the remaining action-consistency, live-continuation and memory work.

Date: 2026-09-13. **Latest task complete: interface v3 fixes and matched development.**
See [the completed v3 checklist and results](TABLE2_HYBRID_INTERFACE_V3.md).
456 focused tests passed; all 24 live episodes completed with audit PASS.
H0/H1/H2/H3 each completed **2/6**, unchanged from v2. Executed recovery actions
rose from 2 to 10; model calls fell from 296 to 90. The frozen generator still
never selected TYPE. H3 queried five times and abstained five times under the
unchanged strict filter. No new final evaluation is running or claimed ready
to establish a stronger-agent/memory benefit.

**Historical 7/7 tasks complete. Final H-study finished; audit PASS.**
All 120 episodes completed with 30 eligible matched blocks and no exclusions,
missing episodes or infrastructure errors. H0 completed **8/30**; H1/H2/H3 each
completed **9/30**. Both primary increments were zero (Holm p=1.0). H3 made two
queries and abstained twice; no recovery action executed or memory example
reached generation.
See [the completed final results](TABLE2_HYBRID_FINAL_RESULTS_V1.md).

Follow-up: [code audit and bounded fix checklist](TABLE2_HYBRID_CODE_AUDIT_AND_FIX_PLAN.md).
The audit identifies rejected-proposal loops, RETRY control identity and memory
admission limitations. Its proposed development work is separate from these
seven completed tasks. The authorized v3 implementation is recorded above;
the completed final study is preserved.

The run and browser workers have stopped. The existing checkpoint, memory,
notebook and historical producing source/prompt snapshots are preserved.
**No unchanged rerun is needed.** The study is complete; the proposed recovery
and memory completion benefits were not demonstrated.

See [v2 live results and readiness](TABLE2_HYBRID_DEVELOPMENT_V2.md).
See [v2 corrections and run command](TABLE2_HYBRID_INTERFACE_V2.md).
See [Task 6 results and readiness](TABLE2_HYBRID_DEVELOPMENT_V1.md).
Follow [the implementation specification](TABLE2_HYBRID_AGENT_IMPLEMENTATION_PLAN.md).
Use H0–H3 identities. Preserve E0–E3 and all completed studies.

## 1. Establish the baseline and freeze the development scope

- [x] Inspect current branch/worktree and active CPU/GPU/browser jobs.
- [x] Record source baseline and notebook hash; preserve unrelated edits.
- [x] Verify existing checkpoint, pinned base, processor and memory manifests.
- [x] Read v6 execution code and regression evidence; reuse the hit-test fix.
- [x] Record H0–H3 definitions and primary/secondary contrasts.
- [x] Confirm the six proposed development families and existing overlap rules;
      freeze eligibility and reset IDs before any model outcomes.

Completed evidence: [Task 1 preflight report](TABLE2_HYBRID_PREFLIGHT.md).
All six families eligible; 24 episodes planned. Twelve reset-only probes passed;
zero model calls, executor actions or memory writes.

Done when: the candidate has a concrete identity, input manifest and protected baseline.

## 2. Build the hybrid normal-action adapter

- [x] Add the versioned, detached trained-advice projection.
- [x] Preserve exact six-class probabilities, argmax, box and confidence.
- [x] Use one shared base-generator interface and normal-action prompt.
- [x] Give H0 no trained advice; supply it only in H1/H2/H3.
- [x] Include current observation/controls, completed actions and last rejection.
- [x] Preserve one four-field response and exact model-issued values.
- [x] Reuse current-control resolution and v6 execution-time validation.
- [x] Log trained advice, raw proposal, resolved target and executed action separately.
- [x] Charge every inference/generation/fallback to the model-call budget.

Completed evidence: [Task 2 adapter report](TABLE2_HYBRID_ACTION_ADAPTER.md).
152 focused tests passed; all six actions executed in a real-browser fixture
with scripted callbacks and zero actual model inference. The adapter/context API
and separate execution receipts are verified; H-runner wiring was completed in Task 5.

Done when: a model-selected current control executes without action substitution,
and trained advice is available through a real, auditable decision path.

## 3. Complete bounded recovery and continuation

- [x] Keep learned diagnosis, strategy and post-recovery assessment in H2/H3.
- [x] Implement the specification's continuation table behind a default-off flag.
- [x] Record executed action, learned assessment and continuation reason separately.
- [x] Preserve exhausted/unresolved incidents; never label them resolved.
- [x] Prevent observation-ID changes or repeated failures from resetting budgets.
- [x] Allow normal acting after incident exhaustion when other stop rules permit.
- [x] Preserve ABORT, terminal/truncation, loop, timeout and total-budget stops.
- [x] Keep observable effects as evidence without inventing click-progress credit.

Completed evidence: [Task 3 continuation report](TABLE2_HYBRID_CONTINUATION.md).
20 controller tests and 108 compatibility checks passed. Scripted text and click
sequences exercised continuation after negative assessment. No model inference
or live evaluation ran; H system registration was subsequently completed in Task 5.

Done when: scripted text-entry and click-only sequences can continue under negative
assessment while all incident and episode limits remain enforced.

## 4. Connect H3 memory and verify H-system isolation

- [x] Preserve the complete H2 decision before H3 retrieval.
- [x] Reuse unchanged vectors, top three, threshold, material and applicability filter.
- [x] Leave missing corrective values/reflections null.
- [x] Verify empty admission preserves the complete H2 decision.
- [x] Verify H0/H1/H2 never query memory and every system performs zero memory writes.
- [x] Distinguish candidate retrieval, generation exposure, action change and benefit.

Completed evidence: [Task 4 memory report](TABLE2_HYBRID_MEMORY.md).
93 focused tests passed. Three saved-vector probes matched independent top-three
retrieval; all bound memory files remained unchanged. No model inference, new
embeddings or live evaluation ran. Full H-runner wiring was subsequently verified in Task 5.

Done when: frozen retrieval is reproducible and the only H2/H3 difference is the
specified memory intervention.

Historical v1 qualification: the store and fixture checks above remained valid, but live
generation also received differing bookkeeping fields before memory. Full
model-visible H2/H3 isolation was not established in v1. V2 corrected this:
all 44 actual H2/H3 proposal pairs match; all six memory queries abstained.

## 5. Add the thin runner and meaningful regression checks

- [x] Add H-profile configuration and launcher using existing environments/loaders.
- [x] Make system identities and analysis contrasts explicit in logs and reports.
- [x] Preserve old E-profile behavior and historical evidence replay.
- [x] Test trained-output parity and causal context boundaries.
- [x] Test all six actions, exact values, stale/occluded/overlapping targets.
- [x] Test negative assessment, click continuation, incident exhaustion and exact budgets.
- [x] Test advice/recovery/memory isolation, no writes and raw-reward scoring.
- [x] Run real-browser engineering fixtures; label them as zero-model-call fixtures.
- [x] Freeze a new engineering receipt for the exact tested source version.

Completed evidence: [Task 5 runner and engineering report](TABLE2_HYBRID_RUNNER.md).
330 selected tests passed. Real Chromium fixtures exercised all six actions and
32 controller requests with negative assessments; zero actual model inference.
Historical 120-episode replay passed with identical analysis. The H development
plan is prepared at `/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v1/plan.json`.

Done when: focused checks pass and remaining suite failures, if any, have explicit
causes and relevance assessed. Do not call the entire suite green if it is not.

## 6. Run one matched live development check

- [x] Freeze source, prompts, assets, installed dependencies, budgets and task/reset manifest.
- [x] Recheck active jobs before loading models.
- [x] Run H0–H3 on six families: 24 planned episodes, less whole-block exclusions.
- [x] Persist every episode/raw proposal immediately; preserve interruptions and failures.
- [x] Audit all identities, counters, advice, continuation and retrieval evidence.
- [x] Report H0/H1/H2/H3 completions, per-family failures and resource costs.
- [x] State whether model-chosen typing and click-only continuation actually occurred.
- [x] State whether H3 added completions; retain null or negative findings.
- [x] Stop this development cycle with an explicit readiness/limitation record.

Completed evidence: [Task 6 live results](TABLE2_HYBRID_DEVELOPMENT_V1.md).
Zero exclusions or infrastructure failures. Six executed clicks from 562 requests;
zero typing, recovery, continuation or memory queries. The saved PASS audit has
`live_path_verified: false`. A separate readiness record identifies the input
isolation defect; H3's lost completion cannot be attributed to memory.

Done when: the complete matched check and audit are saved, regardless of whether
the candidate improves performance. Do not launch repeated prompt trials to find a win.

## 7. Final evaluation and thesis delivery — after the development decision

- [x] Remove system IDs and latency-dependent provenance from model-visible
      context while preserving full audit bindings; test paired pre-memory equality.
- [x] Assess saved format failures under the declared v2 development contract;
      preserve exact actions/values and reject ambiguous or invalid proposals.
- [x] Pass v2 engineering checks and freeze a separate matched development plan.
- [x] Run and audit all 24 frozen v2 development episodes; retain every outcome.
- [x] Verify live recovery execution, assessment, bounded continuation and H3
      retrieval/fallback; report zero typing and zero memory-generation exposures.
- [x] Run the bounded eight-call planner diagnosis; preserve every response and
      report schema changes separately from useful actions or completions.
- [x] Amend the protocol to accept verified abstention; preserve historical
      full-path coverage=false and record the new scope decision separately.
- [x] Retain the audited v2 agent and its prompts; record the planner limitations.
- [x] Freeze 30 task/reset blocks, 120 planned episodes and scoring/analysis;
      retain all six previously specified evaluation families and overlap rules.
- [x] Check reset-seed collisions against archived study manifests and label
      previously observed families honestly; development families remain separate.
- [x] Bind a thin H final launcher and execution plan to the amended protocol;
      test accepted abstention and rejection of broken retrieval/isolation,
      preserve old E/H development gates, and freeze producing-source evidence.
- [x] Run the frozen final study once; preserve all outcomes and exclusions.
- [x] Report primary H2−H1 and H3−H2 paired effects with uncertainty and Holm correction.
- [x] Report secondary H1−H0 separately; do not equate advice exposure with benefit.
- [x] Include valid-action rate, recoveries, memory exposure, calls and latency.
- [x] Keep E-profile and H-profile results distinct in the manuscript.
- [x] Support any external-superiority claim with a matched external baseline;
      otherwise limit claims to the systems actually compared.
- [x] Archive code/evidence and update the handoff/checklist with actual results.

No retraining, package upgrade, new model or embedding regeneration is on this list.
