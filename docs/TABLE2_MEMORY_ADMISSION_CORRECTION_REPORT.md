> Latest interface corrections and four-task development check: [TABLE2_CAUSAL_RECOVERY_INTERFACE_REPORT.md](TABLE2_CAUSAL_RECOVERY_INTERFACE_REPORT.md). E0: 1/4, E1: 0/4, E2: 2/4, E3: 2/4; development results, not replacement final Table 2 evidence.

# E1 implementation audit and E3 admission correction — 2026-09-10

The E3 similarity-only intervention lacked a strategy-applicability check.
After adding conservative admission conditions, E3 completed the matched
click-button development task that previously failed. E1 still failed; its
runtime prediction matched the model's full forward numerically on both saved
diagnostic inputs. No retraining or model-setting changes were made.

## E1: what was checked

The selected runtime's pre-action branch uses the policy adapter and action
head, with the training label inverse to decode its argmax. The previous input
alignment check established exact processor tensor/prompt equality and matching
training label source. This session additionally compared the runtime output
with `model.forward` using the same two saved development pre-action inputs.
The diagnostic duplicated that pre stream into the post stream solely to call
the complete causal forward; no future observation or dataset row was read.

Both inputs produced exactly equal action probabilities (maximum difference
0.0), identical bounded boxes and NAVIGATE argmax in both paths. On the saved
button input, NAVIGATE had probability 0.31147, CLICK 0.29586 and SCROLL 0.28814.
The executor cannot execute NAVIGATE without a URL; both registered parameter
provider stages rejected it, and E1 stopped after three repeated rejections.

This finds no action-translation discrepancy on the checked path. It does not
prove correctness for all possible inputs or identify why the checkpoint prefers
NAVIGATE. Changing NAVIGATE to CLICK or replacing E1 with the base planner would
change the trained-policy baseline, not repair an evidenced decoding bug.

The existing aggregate epoch-6 validation report records action accuracy
0.42170, action macro-F1 0.32668, mean bbox IoU 0.10095 and memory-flag accuracy
0.84277. These measure different heads, not live task completion. The report
also records 4,770 training NAVIGATE labels; the old label-file comment claiming
no NAVIGATE data is contradicted by this aggregate artifact and was not used to
mask that class. No dataset rows or locked-test data were inspected.

## E3: implementation gap and correction

Previously, the first above-threshold retrieval always replaced the shadow
recovery strategy. In the failed button episode it chose BACKTRACK despite the
only NAVIGATE request having been rejected. After the ineffective backtrack,
ALTERNATIVE_TARGET also failed its semantic-target validation.

The versioned `causal-strategy-admission-v1` option now checks the first admitted
candidate against the exact failed transition before intervention:

- BACKTRACK requires evidence of an executed, state-changing NAVIGATE within
  the current episode, without an environment error.
- RETRY and ALTERNATIVE_TARGET abstain after `parameter_resolution_rejected`:
  repeating or retargeting alone does not resolve the missing action parameters.
- When excluded, keep the complete original shadow decision and log the reason.
  Do not substitute a lower-ranked candidate. Existing planner, target and
  execution checks still apply to any retained or admitted strategy.

These are conservative necessary conditions, not a complete browser-history
model or a guarantee that admitted advice will help. Click-induced navigation
is not proven by the current history fields, so it cannot satisfy the BACKTRACK
condition. No task name, goal, reward or final outcome enters this gate.

Implementation: `src/web_agent/memory/label_runtime.py`; focused cases in
`tests/table2/test_label_memory_runtime.py`. The option is explicitly enabled in
`/home/aiub/kiyas/table2-evidence/miniwob-feasibility/miniwob_campaign_admission_v1.py`.
Historical runners retain their original admission behavior by default.
Embeddings, cosine scores, top-3 retrieval and the calibrated threshold are
unchanged. This admission change was frozen before the new live run.

## Matched live development results

| Task | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| click-button | Success | Failure | Success | Success |
| enter-text | Failure | Failure | Failure | Failure |

E3 queried memory, excluded the unsupported BACKTRACK candidate, retained
REPLAN, executed the named button click and received reward 1 with termination.
It therefore avoided the harm observed before this correction. Across both
tasks there were three queries, three abstentions and zero memory interventions.
E3's success is preserved E2 recovery, not evidence that retrieved advice adds
benefit. E2 minus E1 is +50 percentage points; E3 minus E2 is zero on these two
development pairs. This debugging allocation is not a final efficacy estimate.

Text entry retains the previous unsupported/malformed generated action outputs;
no text or target was manually supplied to make it pass. E1 and text entry are
not reported as solved.

## Verification and preserved evidence

- Eight episodes, matched reset screenshots and E1–E3 first outputs: PASS.
  These initial screenshots and selected outputs also match the preceding
  named-target-v2 development run.
- Independent retrieval/admission replay, event logs, causal bindings and
  browser outcomes: PASS; zero runtime errors. Two recovery actions executed
  and received assessments; one normal action executed.
- Focused runtime, grounding, recovery and memory suite: 128 passed. Applicable
  advice remains enabled in synthetic admission tests; synthetic tests are not
  live memory-benefit evidence. An initial test import error and an incorrect
  test filename were corrected before the passing suite and live run.
- All 160 plan-bound source files were verified and archived. All 484 artifacts
  in the original final evaluation are unchanged. All six manifest-bound P4
  files passed hash verification; its manifest remains
  `4b577b79e2f8e04424572aa25223a81fa15a8f1f3cb483ef181dda982aa072d6`.
- Model/base files, prompts, decoding, packages and the notebook were unchanged.
  The development process exited; no evaluation is queued.

Evidence: `/home/aiub/kiyas/table2-evidence/miniwob-memory-admission-v1/`, including
`plan.json`, `e1-forward-parity.json`, `e1-training-report-diagnostic.json`,
`evidence-audit.json`, `analysis.json`, `completion.json`, `tests.log`, per-episode
logs and `source-snapshot/`.

The current checkpoint can continue to support the diagnosis/recovery
experiment. Neither a positive P4 contribution nor full model competence has
been established. Any further study must freeze this revised interface and
admission profile and disclose the previous debugging exposure. The original
24-episode results and paper Table 2 status are not replaced by this development
run.
