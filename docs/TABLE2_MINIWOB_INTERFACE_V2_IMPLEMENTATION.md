# Table 2 MiniWoB interface v2

Current status: implementation, engineering checks, corrected checkpoint-backed
development and all 120 final episodes completed; audits PASS. Read
[TABLE2_MINIWOB_INTERFACE_V2_RESULTS.md](TABLE2_MINIWOB_INTERFACE_V2_RESULTS.md)
for the measured results and limitations. The earlier
24-episode result remains preserved in `TABLE2_FULL_CREDIT_RESULTS.md`.

## Frozen research objective

E0 is the unadapted selected base; E1 is the epoch-6/seed-42 trained multimodal
policy and grounding; E2 adds executed failure diagnosis/recovery/assessment;
E3 adds frozen train-only memory. E1→E2 and E2→E3 are the primary paired contrasts.
The four systems do not independently identify a gain from every pillar.

No retraining, model/decoding changes, embedding regeneration, package upgrades,
Gold-image reads, locked-test access, or new dataset review is part of this work.

## Implemented interfaces

- `miniwob_controls` exposes visible/associated/sibling labels, observation-bound
  IDs, current form state and one shared action-capability rule. Password values
  remain masked. Names/IDs must resolve uniquely; coordinate actions retain their
  predicted points. Execution rechecks the current target before browser action.
- E0 and the shared parameter provider receive the same control projection used
  by recovery. Trained pre-action processor inputs, logits and boxes are unchanged.
- `RecoveryPlannerContext` carries only completed agent-issued actions and
  previous-attempt feedback. The controller supplies detached copies and records
  parsing, targeting, parameter, strategy, execution and assessment stages.
- Parsing errors are explicitly caught as policy rejections, rather than
  unexpected backend errors. All attempts retain their existing budget costs.
- The neutral recovery instruction treats the six action classes consistently.
- Memory context adds source task/domain and availability flags from the existing
  hash-bound material. The 1,974 vectors, cosine top-three ordering and admission
  threshold remain fixed. Missing corrective values/reflections remain null.
- The versioned MiniWoB runner owns preparation, serial execution, resume of
  unstarted blocks, context/retrieval replay and paired reporting. BrowserGym runs
  in its existing separate environment; model loading occurs once per campaign.

## Engineering evidence

Evidence root: `/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-engineering/`.

The seven-input frozen-source prompt probe changed only the TYPE-emphasizing
sentence. Rejected proposals decreased from 3/7 to 1/7. A repeated text entry
became a Submit proposal; one previously valid proposal gained an invalid trailing
comma. Both outcomes are preserved. This probe executed zero browser episodes.

Browser fixtures exercised all six executor actions and label/state extraction.
A separate adapter fixture exercised observation construction and worker execution.
These are engineering checks, not model-completion evidence. The first browser
fixture failed because its scripted SCROLL omitted the registered `container`
parameter; the corrected fixture and original attempt are both preserved.

The first 16-episode development package completed execution but failed its audit:
the saved block's `task` string overwrote the full task specification. The fix
stores `task_specification` separately and tests its goal and block identity.
The original failed audit and all episodes remain in
`miniwob-interface-v2-development`; a diagnostic replay is explicitly marked
unaccepted. The corrected frozen development package is
`miniwob-interface-v2-development-r2`. This repeat addresses the evidence-record
defect; model decisions, prompts, budgets and resets were not changed.

All 2,561 hash-manifested files across the five earlier experiment archives
were verified unchanged. The engineering directory contains the verification.

## Execution and statistical protocol

Development: four existing families (`click-button`, `enter-text`, `click-test`,
`click-button-sequence`) × E0–E3 = 16 episodes, with their existing reset namespace.
Final: six previously observed families (`click-link`, `click-option`,
`click-checkboxes`, `enter-password`, `login-user`, `focus-text`) × five resets ×
four systems = 120 episodes before overlap exclusions. The actual 30 reset seeds
are frozen before outcomes. Five browser resets are not five training seeds.

Budgets: 30 executor requests, two recovery attempts per incident, four per
episode, 102 model calls, 600 seconds. Full completion requires termination,
no truncation and raw reward exactly 1.0. Positive partial reward stays separate.
Overlap and infrastructure-invalid blocks affect all four systems together;
ordinary model failures remain in the denominator. No outcome-driven reruns.

Primary unit: eligible task-reset pair. Report E2−E1 and E3−E2, improved/worsened
pairs, exact two-sided discordant-pair tests, Holm correction across both
contrasts, and 95% paired bootstrap intervals within fixed families (10,000
resamples; analysis seed 20250831). Report exposure, executed recovery, model
calls and latency separately. Positive support requires a positive difference
and Holm-adjusted p < .05; otherwise improvement is not demonstrated.

The engineering and live-path gates do not require positive completion gains.
Scope is these fixed MiniWoB families with one trained checkpoint, not fresh
held-out-family confirmation or independently verified reflective memory.

## Commands

Run with `PYTHONPATH=src .venv/bin/python scripts/run_table2_miniwob.py`:

1. `prepare --phase development --output /ABS/new-development`
2. `run --output /ABS/new-development`
3. `prepare --phase evaluation --development /ABS/new-development --output /ABS/new-evaluation`
4. `run --output /ABS/new-evaluation`

`audit --output /ABS/run` replays saved evidence under its producing source.
Completed packages cannot be relaunched. Interrupted blocks remain recorded;
resuming executes only blocks that never started. The prepared plan, sources,
assets, prerequisites and dependency identities must still match.

## Accepted development and evaluation launch

The corrected development package completed all 16 episodes with audit PASS: E0
1/4, E1 0/4, E2 1/4, E3 1/4. Six recovery actions executed and received learned
assessments; all six E3 generations included the admitted frozen memory examples.
Eleven generation contexts replayed successfully. Generated recovery contexts
contained one prior action in this run; longer histories are covered by engineering
checks but were not exercised by these live model decisions.

Evaluation was frozen before new outcomes: 30 matched blocks, 120 episodes, plan
SHA-256 `66d59483d3a459355e944fd1f2afca5c8fa3cd6889b5b892fa89e348a684dfcf`.
Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-evaluation/`.
No source, prompt or model changes are permitted during this run.

## Final execution completed

All 120 episodes completed once, with no exclusions, missing episodes or
infrastructure failures. All 30 paired blocks and 76 generated recovery contexts
passed the final audit. E0/E1/E2/E3 completed 6/0/6/6 out of 30 each. The recovery
difference was +20 percentage points (Holm p=0.0625); the memory difference was
zero (Holm p=1). Neither met the prespecified positive-effect criterion.

The final live run exercised two-action histories and negative-assessment
feedback in five E3 generations, supplementing the engineering history tests.
There were 54 executed recoveries with learned assessments, 46 memory queries
and 42 E3 generations exposed to memory. Exposure did not add completions.

The producing source, frozen plan, complete evidence and supplemental descriptive
report are archived with hashes. No further changes or outcome-driven reruns
are part of this completed plan.
