> Latest implementation audit: [TABLE2_MEMORY_ADMISSION_CORRECTION_REPORT.md](TABLE2_MEMORY_ADMISSION_CORRECTION_REPORT.md). E3 now completes the button development task after memory abstention; E1 forward parity passes but E1 and text entry still fail.

# Continuing PC-01 without retraining — 2026-09-10

The revised interface produced a real successful E2 recovery using the existing
PC-01 epoch-6 seed-42 checkpoint and frozen Qwen recovery planner. Retraining is
not required to execute this recovery path. This does not resolve every policy
failure or establish a memory benefit.

## Fix and evidence

The planner emitted `CLICK` with the visible target name `next`, an inaccurate
box and an omitted nullable `value`. The earlier interface rejected it. The new
`visible-name-action-interface-v2` explicitly resolves a model-supplied name to
one exact, compatible visible DOM control and defaults omitted nullable fields
to null. A named target takes precedence over its generated box. For CLICK,
a name emitted in `value` is also supported when `target` is null.

This is a changed grounding interface, not a claim that the old output box was
correct. The resolver preserves the emitted action type, rejects ambiguous or
missing names and malformed JSON, and logs raw output and the resolution. It
does not infer actions from task goals, fill in missing navigation URLs or use
success labels. Existing point-target and recovery-semantic validation remain.

Implementation: `src/web_agent/runtime/named_target_policy.py`.
The development runner is
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/miniwob_campaign_named_v2.py`.
It applies the interface to E0 generation and the shared E2/E3 recovery planner.
The selected PC-01 normal policy remains unchanged across E1–E3. Checkpoint,
base snapshot, prompts, decoding, packages and frozen P4 store are unchanged.

## Matched live development results

| Development task | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| click-button | Success | Failure | Success | Failure |
| enter-text | Failure | Failure | Failure | Failure |

E2's initial action was rejected; diagnosis triggered recovery; the frozen
planner named the button; the interface executed the click. MiniWoB returned
reward 1 with task termination. E0 also completed that task with its own output.

E3 first replaced REPLAN with retrieved BACKTRACK, which executed without task
success. Its next retrieved ALTERNATIVE_TARGET advice failed the different
semantic-target check. Enter-text E0 emitted an unsupported object-valued target;
its recovery planner emitted malformed JSON. These failures were retained.
Repeated PC-01 NAVIGATE predictions without URLs remain a policy limitation.

Across the two development pairs, E2 minus E1 is +50 percentage points and E3
minus E2 is -50 percentage points. These are development observations from an
interface selected during debugging, not final efficacy estimates. Do not pool
them with the earlier evaluation or select only the successful task for reporting.

## Verification and preservation

- Eight live episodes completed; matched reset screenshots and E1–E3 initial
  policy outputs passed the audit, with zero runtime errors.
- One normal action, three recovery actions, four memory queries/interventions
  and three post-recovery assessments were recorded.
- Focused interface/runtime/recovery tests: 118 passed.
- All 158 plan-bound source files matched and were archived. All 484 artifacts
  of the previous 24-episode evaluation remained byte-identical.
- Frozen memory manifest remains
  `4b577b79e2f8e04424572aa25223a81fa15a8f1f3cb483ef181dda982aa072d6`.
- No training, embedding regeneration or notebook interruption occurred. The
  development job exited; no new evaluation is queued.

Evidence directory:
`/home/aiub/kiyas/table2-evidence/miniwob-named-target-v2/` contains `plan.json`,
`analysis.json`, `evidence-audit.json`, `completion.json`, `tests.log`, raw
generation receipts and the producing `source-snapshot/`. The earlier v1
development attempt is preserved separately.

## Remaining work

Continue with the same checkpoint. Review whether memory strategy applicability
can be established from observable episode history before intervention; any new
abstention rule must be versioned and audited, not chosen from final outcomes.
Text-entry output validity also remains unresolved. Positive E3 improvement is
not an execution requirement and must not be forced by changing thresholds or
manually correcting actions.

Before a further evaluation, freeze the revised grounding interface and any
memory-admission change. The previous six evaluation families have already been
observed: a rerun is an explicitly disclosed post-debugging replication, not a
fresh held-out test. Preserve all four pillars and matched E0–E3 comparisons.
The original 24-episode results remain 0/6 per system; this development check
does not replace the paper Table 2 entry.
