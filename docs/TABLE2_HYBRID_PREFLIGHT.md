# Hybrid-agent Task 1 — completed

Date: 2026-09-12. **Status: PASS. Next task: build the hybrid normal-action adapter.**
No H-agent implementation or model evaluation was performed during this task.

Evidence: `/home/aiub/kiyas/table2-evidence/hybrid-dev-v1-preflight/`.
Read `completion.json`, `asset-verification.json`, `development-scope.json`,
`baseline-manifest.json` and `preflight.log`. The verification script is retained.

## Repository and environment

- Existing repository and branch: `/home/aiub/kiyas/webagent`, `Code`.
- HEAD: `ca10c72aad52bf60a5e2c5e8b6c8f9e5a9dd1e1e`.
- Preserved 474 source/configuration/document/test/notebook/editor files in a
  read-only baseline snapshot, plus the tracked worktree patch and git status.
- No training/evaluation job was observed. GPU use was remote desktop only.
  Existing notebook kernels and another coding-agent process were left untouched.
  Recheck active jobs before later model loading; this is a timestamped observation.
- Existing model/browser package versions and MiniWoB source identity verified.
  No environment changes, package installs or model loads occurred.

## Frozen artifacts

Verified 54 asset/dependency-file bindings using the existing verifier, including
checkpoint/export/base/processor files, report/run-contract identities, memory
artifacts and installed BrowserGym Python sources. PC-01 remains epoch 6/seed 42.

The memory store contains 1,974 finite normalized 768-dimensional vectors;
its array is opened read-only, store writes are disabled, and the original
cosine top-three threshold is `0.7371385097503662`. All bound source material
was checked against the existing store. No embeddings were regenerated.

The later v7 executable-action-selection experiment exists in the worktree and
was preserved. Its runtime default is off. The H development scope explicitly
sets `executable_action_selection: false` and preserves trained predictions.
The v6 hit-test fix and its existing audit/regression evidence were inspected;
no old experiments or fixtures were rerun as evaluation.

## Development scope frozen before model outcomes

Profile identity: `miniwob-hybrid-dev-v1`. The existing stage-seed namespace
`miniwob-label-memory-development-v5` is retained to reproduce the four original
development resets. This seed namespace does not rename H systems as E systems.
Repeat ID 0 and model seed 42 are fixed for every family.

| Family | Browser reset seed | Eligibility |
|---|---:|---|
| click-button | 606453901 | Eligible |
| enter-text | 2692356497 | Eligible |
| click-test | 1373578930 | Eligible |
| click-button-sequence | 2105833823 | Eligible |
| enter-text-2 | 1255176792 | Eligible |
| click-tab-2 | 4095939871 | Eligible |

The full derived stage seeds, task HTML hashes, bound goals, initial screenshot
hashes and overlap records are in `development-scope.json`; its SHA-256 is saved
alongside it. All six families passed the existing canonical-ID/exact-normalized-
goal/trigram-overlap rules against prepared train-only memory. This is that
specific eligibility check, not a claim that every possible semantic overlap
has been ruled out or a new dataset review.

Each family was reset twice without taking any action. Goals, screenshots and
projected control snapshots matched exactly: **12 reset probes; zero model calls,
zero executor actions, zero memory writes**. These are setup checks, not episodes
of agent performance. Initial observations are development evidence.

`enter-text-2` requires the case transformation stated in its user goal; the model
must derive the text rather than the executor hard-coding a transformation.
`click-tab-2` adds tab exploration and a link click. Hidden tab contents or
evaluator answers must not enter the actor's current-control context.

H0: base generator; H1: base generator plus trained advice; H2: H1 plus learned
recovery; H3: H2 plus frozen memory. Primary contrasts: H2−H1 and H3−H2;
secondary: H1−H0. Six eligible blocks × four systems = **24 planned episodes**.

Budgets remain 30 requests, two attempts per incident, four attempts per episode,
102 model calls and 600 seconds. Full completion requires raw reward exactly 1.0,
termination and no truncation. The development manifest records the analysis rules.

## Boundary for the next task

This freezes task scope and starting artifacts only. **Launch readiness is false**:
H adapters, continuation, prompts, regression receipt and final producing-source
freeze are still pending. Do not use the old E runner to launch this H manifest.

All baseline files and asset bindings matched again at the end of verification.
Only the progress documents are subsequently updated to record Task 1 completion.
Follow [Task 2 in the TODO](TABLE2_HYBRID_AGENT_TODO.md) and the
[implementation plan](TABLE2_HYBRID_AGENT_IMPLEMENTATION_PLAN.md).
