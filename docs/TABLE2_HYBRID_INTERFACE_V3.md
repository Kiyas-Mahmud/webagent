# Hybrid interface v3: bounded proposal correction and stable retry targets

2026-09-13. **Revision complete. All 24 live development episodes completed;
independent audit PASS, no runtime exceptions or exclusions.** Each system
completed **2/6 tasks**, unchanged from v2. Engineering verification passed:
456 focused tests, the full H0–H3 Chromium fixture, all six executor actions,
and nine stable-target browser cases. This revision is development only and
preserves every completed E/H study, including the final H0 8/30 and H1/H2/H3
9/30 result.

## Declared method

The checkpoint remains PC-01 epoch 6, seed 42. The base model, processor,
greedy decoding and 128-token generation cap are unchanged. No retraining or
model substitution is part of this revision.

- One common H0–H3 interface retains all six actions and the exact four response
  fields. It reports missing/extra fields and invalid types precisely. It
  accepts a current control ID, an unambiguous observed name, or the explicitly
  checked `observed_tag#current_control_id` syntax. No generic CSS or guessed
  target replacement is executed.
- A malformed normal proposal receives at most one correction request on the
  same observable page. Repeated rejection cannot consume 30 identical calls.
  A well-formed action with a rejected target keeps its issued type/value and
  rejection status, and can enter learned diagnosis in H2/H3 without a browser
  command. Unparseable JSON never receives a fabricated action label.
- RETRY alone may rebind an earlier runtime-owned target receipt to a uniquely
  matching current control. Source identity, observed properties, geometry,
  capability and hit evidence must match. Model-selected action/value are
  preserved. New model proposals containing stale IDs still reject.
- Loop fingerprints use stable control evidence instead of capture IDs. Raw
  capture bindings remain in the evidence. The model-visible advice field
  `action_class_confidence` accurately names the old maximum action probability;
  the underlying trained prediction and historical records are unchanged.
- Past action records and proposal correction feedback have distinct semantic
  fields. Generation receipts now record token count, EOS and cap status without
  changing decoding.

**P4 remains the existing strict selective-memory method.** All 1,974 vectors,
ordering, cosine top three, threshold and admission checks remain unchanged.
Missing corrective values/reflections remain null. This revision does not test
the separately proposed advisory-label admission policy. The 86 historical
RETRY supplement vectors' differing transition route remains a documented
limitation; none is rewritten or recalibrated.

P2 supplies the unchanged multimodal trained advice; P3 grounds and executes the
generated action. H2 adds P1 learned diagnosis, strategy, recovery execution and
assessment. H3 adds P4 only after preserving the complete H2 decision. A negative
learned assessment and a successfully executed browser action remain separate
facts. Actual environment termination, retry limits and budgets remain binding.

## Fixed development check

The same six development families and saved reset seeds are reused across
H0–H3: `click-button`, `enter-text`, `click-test`, `click-button-sequence`,
`enter-text-2`, `click-tab-2`. All 24 outcomes will be retained. Existing overlap
rules exclude an affected whole block without replacement.

Limits remain 30 executor requests, two recovery attempts per incident, four per
episode, 102 model calls and 600 seconds. Full completion remains termination
without truncation and raw reward exactly 1.0. Engineering fixtures use scripted
callbacks and are never counted as live task completions.

Configuration: [v3 profile](../configs/eval/table2/miniwob_hybrid_dev_v3.json).
Scope: [inherited task/reset bindings](../configs/eval/table2/miniwob_hybrid_dev_v3_scope.json).
Entrypoint: [v3 runner](../scripts/run_table2_hybrid_dev_v3.py).
The source/prompt/assets/dependency freeze must follow successful engineering
checks and precede inference. No further prompt selection will use this run's
outcomes.

## Completion checklist

- [x] Record the source/notebook baseline and inspect active jobs.
- [x] Declare the new interface and retain strict memory semantics.
- [x] Complete focused regression and real Chromium checks.
- [x] Freeze the exact producing source, prompts, environment and artifacts.
- [x] Execute and independently audit the matched 24 development episodes.
- [x] Report actual planning/recovery/memory behavior and final readiness.

Stable retry identity is observational: a DOM replacement reusing the same source
ID and every recorded presentation property cannot be distinguished by the
existing snapshot. Changed, removed, ambiguous and occluded observed controls
must reject. This limitation is not a claim of persistent browser object identity.

Engineering evidence: `/home/aiub/kiyas/table2-evidence/hybrid-interface-v3-engineering/`.
The matched run is frozen at `/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v3/plan.json`.
The controller fixture executed 32 browser requests with eight negative learned
assessment callbacks and four memory context exposures. These were scripted,
with zero real model inference. Historical v2 replay passed with identical
analysis; its episodes were not rerun. No test is counted as live performance.

## Actual development result

| System | Completion | Executed normal | Executed recovery | Model calls | Memory queries / exposures |
|---|---:|---:|---:|---:|---:|
| H0 | 2/6 | 13 | 0 | 13 | 0 / 0 |
| H1 | 2/6 | 10 | 0 | 20 | 0 / 0 |
| H2 | 2/6 | 6 | 5 | 26 | 0 / 0 |
| H3 | 2/6 | 6 | 5 | 31 | 5 / 0 |

All systems completed `click-button` and `click-test`. None completed the other
four development tasks. Both primary completion differences were zero, with
zero improved/worsened pairs, exact p=1 and Holm p=1. With one reset per family,
the reported bootstrap intervals are degenerate and provide little uncertainty
information. This development comparison does not replace a final evaluation.

The revised interface produced **35 valid normal proposals out of 35**, and
all 35 executed. Every one selected CLICK and ended generation with EOS; none
reached the token cap. The six generated recovery proposals were also CLICK.
H2/H3 each executed three REPLAN clicks and two BACKTRACK keypresses. All ten
post-recovery learned assessments remained negative. No useful text insertion,
selection or checked-state preparation effect was observed. The final stops
were 19 genuine environment terminations and five bounded loop stops.

Compared with the previous matched v2 development run, executed recovery actions
increased from **2 to 10**, while total model calls fell from **296 to 90**.
These are measured interface/execution differences for this combined revision;
they do not isolate which edit caused the change. Completion stayed 2/6 for
each system. The newly repaired malformed-proposal and RETRY paths were not
selected in this live run; their evidence remains the engineering fixtures.

H3 queried the frozen store five times and correctly abstained five times.
Thirteen candidate occurrences were excluded for missing corrective arguments
or unsupported historical BACKTRACK advice. There was no memory generation
exposure, no memory write and no demonstrated memory increment. The reported
`live_path_verified: false` refers to the unexercised complete memory-exposure
path; it does not negate the verified retrieval, abstention or recovery execution.

## Remaining limitation and readiness

The scoped implementation task is complete. No remaining implementation error
was identified in the tested and audited paths. The frozen generator still
skips typing on text tasks and repeats the same button in the sequence task;
these actions were genuinely generated and faithfully executed. The existing
strict memory filter has no admissible corrective context in this run. Further
format repair alone cannot establish the missing task-planning capability.

**A stronger-agent or four-pillar performance claim is not ready.** Useful
model-chosen preparation and memory benefit remain undemonstrated. A negative
or null result is not an engineering failure and is not grounds for repeating
the same final evaluation. No new final run is launched here. Any further
advisory-memory or planner-method experiment needs its own declared development
profile; any new final comparison needs a frozen protocol and matched reference
agents before claiming superiority over others.

Raw evidence and independent analysis:
`/home/aiub/kiyas/table2-evidence/miniwob-hybrid-dev-v3/` (`plan.json`,
`results.json`, `audit.json`, `analysis.json`, `REPORT.md`, per-episode receipts).
The producing source and all 61 bound asset files passed verification after the
run. All 6,139 checked historical files and the user's notebook are unchanged.
The coordinator and browser workers exited; no evaluation remains running.
