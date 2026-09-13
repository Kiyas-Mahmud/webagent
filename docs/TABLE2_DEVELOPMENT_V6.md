# Overlapping-target execution revision — development-v6 results, 2026-09-12

Status: implementation complete; all 16 matched development episodes ran; the
evidence audit passes with `live_path_verified: true`. **Task completion did not
change.** Final evaluation readiness remains **false**.

Diagnosis and the exact defect are in
[the overlapping-target diagnosis](TABLE2_OVERLAPPING_TARGET_DIAGNOSIS.md). This
document records only what the live run produced.

## 1. What changed, and what did not

Two places decided target identity by counting how many control boxes contain a
point. Both now use the browser's own hit test for a control the model named,
and both keep their original behaviour for everything else.

Every model input is byte-identical to development-v3: the same prompt file, the
same planner context, the same control projection and the same advertised schema
label. The recovery generation inputs were verified byte-identical receipt by
receipt. Tasks, resets, budgets, checkpoint, base, memory store, threshold and
model seed are unchanged. **Only execution differs.**

## 2. Frozen protocol

`click-button`, `enter-text`, `click-test`, `click-button-sequence` across
E0–E3: 16 episodes, one matched reset per family, reusing campaign
`miniwob-label-memory-development-v5` so the reset seeds equal development-v3's
exactly. Budgets remain 30 executor requests, two recovery attempts per incident,
four per episode, 102 model calls, 600 seconds; checkpoint seed 42. Full
completion requires termination without truncation and raw reward exactly 1.0.
Every outcome is retained.

Profile `miniwob-development-v6`, launcher
[run_table2_development_v6.py](../scripts/run_table2_development_v6.py),
configuration
[miniwob_development_v6.json](../configs/eval/table2/miniwob_development_v6.json),
declared revision flag `browser_hit_target_resolution`.

## 3. Live outcome

| System | development-v3 | development-v6 |
|---|---:|---:|
| E0 | 1/4 | 1/4 |
| E1 | 0/4 | 0/4 |
| E2 | 2/4 | 2/4 |
| E3 | 2/4 | 2/4 |

Per episode, the only differences are on `click-button-sequence`:

| | development-v3 | development-v6 |
|---|---|---|
| E2 executed recovery actions | 0 | **1** |
| E3 executed recovery actions | 0 | **1** |
| Rejections at `parameter_resolution` | 4 | **0** |
| Rejections at `target_resolution` | 0 | 2 |

Run totals: E2 and E3 each executed 4 recovery actions (was 3), across 5 attempts
(was 5). E3 made 5 memory queries with 4 generation exposures, 11 context
exclusions and 1 complete abstention. The paired contrasts are
`E2−E1 = +0.5` (2 improved, 0 worsened, exact p = 0.5, Holm p = 1.0) and
`E3−E2 = 0` (p = 1.0); both report `IMPROVEMENT_NOT_DEMONSTRATED`. **Four matched
pairs cannot demonstrate anything**, and this run does not replace or revise the
completed 120-episode evaluation.

## 4. What the sequence episodes actually did

Both E2 and E3 now run: trained NAVIGATE rejected → recovery triggered → the
model names `o2:c1` (TWO) → **the action executes** at the browser-verified point
`(0.3133, 0.3738)`, which the page hit-tests to that button. Under
development-v3 this identical proposal was rejected with
`TARGET_POINT_MATCH_COUNT:2`.

The episode still fails, for two reasons that the fix does not touch:

1. **The model clicked the wrong button first.** The task requires ONE then TWO;
   it chose TWO. One click leaves the task unterminated with reward 0.
2. **The second attempt re-emitted a stale control ID.** The current observation
   listed `o4:c0` (ONE) and `o4:c1` (TWO), and the model returned `o2:c1` — the
   ID from the previous observation. The resolver correctly rejected it with
   `0 exact compatible matches`; there is no fuzzy matching. This is a model
   behaviour, newly visible only because recovery now advances the observation.

## 5. Multi-step continuation is still not demonstrated live

Across the run: 8 post-recovery assessments, all negative; **0 observable effects
and 0 continuations**.

The reason is structural, and it is a finding rather than a defect: the
observable-effect rule credits only an exact visible text insertion, a changed
selected option, or a changed checkbox/radio state. `click-button-sequence`
progresses purely by clicking a button, which changes none of those fields, so
it can never earn continuation credit however correctly it executes. With the
learned assessment also negative, no budget-permitted continuation can occur on
this task.

**Consequence for the handoff's item 6:** a live model-chosen preparation action
with a visible effect cannot be demonstrated on the current four development
families unless either (a) the model chooses `TYPE`/`SELECT`/checkbox actions —
which on `enter-text` it does not — or (b) the effect vocabulary is extended to
cover click-only progress. Extending it is a method change and was not made here.

## 6. Preserved and superseded packages

- `miniwob-development-v6/` — this run: 16 episodes, PASS audit,
  `live_path_verified: true`.
- `miniwob-development-v4/` — **retained as a negative control, not as a
  result.** Its interface bumped the control projection's advertised schema
  label from `…v2` to `…v3`, and that single character reached the E0 prompt.
  E0's `click-button` generation changed from an unresolvable target to a
  resolved one and the episode flipped to success, lifting E0 to 2/4. Keeping it
  documents that a one-character prompt change moves this model, and that the
  E0 gain in that package is a re-prompting artefact rather than an execution
  effect.
- `miniwob-development-v5/` — the intermediate run after pinning the label but
  before the second ambiguity gate was corrected. Completions matched v3
  exactly; only E3's sequence recovery executed.
- All earlier completed packages are unchanged: `miniwob-development-v3`
  (491 files), its audit-r2 (12) and engineering (196) directories,
  `miniwob-interface-v2-evaluation` (2,354) and
  `miniwob-interface-v2-development-r2` (481) — zero files modified.

## 7. Engineering evidence

213 focused regression tests pass, including 13 new overlapping-target cases.
The full Table 2 suite reports **1,503 passed, 65 failed**; every failure is in
the four process-broker modules and is the pre-existing environment condition
described in the diagnosis document. Two real-browser fixtures pass with zero
model calls. Across ten families at three resets, 96 of 96 controls receive a
hit point, 88 of them the unchanged box centre.

## 8. Readiness

**Another final evaluation remains premature.** The named execution defect is
fixed and verified, but:

- completion did not improve on matched development tasks;
- live multi-step continuation is still not demonstrated, and cannot be on these
  four families under the current effect vocabulary;
- the learned assessment remains uniformly negative;
- E1 still cannot act — all its normal proposals remain NAVIGATE without a
  permitted URL, and this change does not affect E1 at all;
- the memory material still contains no corrective action values or reflections.

A final run should wait until at least the continuation question is settled, on
its own declared and frozen terms.
