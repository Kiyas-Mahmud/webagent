# Table 2 completion-interface check — 2026-09-10

> **Run completed:** 24/24 episodes and both audits PASS. E0/E1 0/6, E2/E3 1/6;
> no additional memory completion. Read [TABLE2_FULL_CREDIT_RESULTS.md](TABLE2_FULL_CREDIT_RESULTS.md).
> The readiness record below is historical; do not relaunch the completed package.

**READY for the revised 24-episode MiniWoB run.** The launcher preflight passed
at 2026-09-10 07:38:38 UTC. No final episodes have started. Readiness means that
the frozen implementation can be evaluated; it does not guarantee successful
actions or a positive memory contrast.

This is a follow-up to the memory context diagnosis. It uses PC-01 epoch 6,
seed 42, the same Qwen base snapshot and all 1,974 frozen memory embeddings.
There is no retraining, package installation, embedding regeneration or change
to decoding or episode budgets. Existing notebook kernels remain running.

## Remaining issues and changes

1. **Identical retry after invalid generation.** Both sequence-task E3 calls in
   `miniwob-memory-context-v1` used identical screenshots, input suffixes and
   generated responses. The planner now receives its completed generation's
   rejection message on the existing next attempt at that same observation and
   incident. Feedback clears on success and cannot cross incident/observation
   boundaries. No extra model call or manually corrected action is introduced.

2. **Immediate action request after long context.** A versioned request repeats
   the current goal after historical examples and asks for one next unfinished
   operation in one JSON object. It describes preparation before submission.
   E2 and E3 use identical instructions; only E3 receives train memory. This is
   an input-design change, not a claim that a correct input guarantees a correct
   model response. Multiple objects and malformed JSON remain rejected.

3. **Custom clickable controls omitted.** The new MiniWoB observation interface
   includes visible pointer-cursor elements and explicit link roles, in addition
   to native controls. It retains their real DOM tags and records clickable and
   link-appearance flags. It excludes hidden/plain text elements and descendants
   already represented by native controls. Exact model-supplied target matching
   and ambiguity rejection remain in place. No task generator, event-handler
   body or hidden target answer is used to discover or choose controls.

4. **Partial reward counted as full completion.** BrowserGym MiniWoB 0.14.3
   converts every positive raw reward to binary reward 1. The new worker also
   records `info.task_info.RAW_REWARD_GLOBAL`, already returned after execution.
   The proposed new evaluation counts success only on termination without
   truncation and raw reward exactly 1.0. Partial credit remains in outcome
   logs; reward data never enters policy observations, memory or recovery.
   Missing/invalid raw reward fails scoring rather than falling back to the
   binary reward. Original completed evaluation scores are preserved.

5. **Missing memory values are absent in the sources.** A field-level audit of
   the hash-matched 23,499 original and 608 supplement training records found
   no nonempty reflection, action-value or parameter fields. This confirms that
   the prepared memory reader did not drop existing corrective text. No images
   or locked-test rows were opened and no new dataset review was requested.
   Missing values stay null; good memory-update classification remains a
   different metric from live retrieval benefit.

## Validation

114 targeted tests passed. The 38 tests affected by the final interface receipt
version update were rechecked and passed. A scripted browser infrastructure
smoke discovered/executed a custom link, excluded hidden/plain text controls and
verified that the real BrowserGym task info supplies raw reward. Scripted smoke
actions and synthetic tests are not checkpoint-backed task outcomes.

Two separate development packages preserve all runs:

- `miniwob-step-feedback-v1`: 16/16 episodes; E0 1/4, E1 0/4, E2 1/4, E3 2/4.
  E3-minus-E2 is +25 percentage points on four development pairs, while E3's
  absolute completion stayed at 2/4 relative to the preceding context experiment.
  The difference therefore includes an E2 regression and is not evidence of
  broad memory effectiveness. Both systems now typed the requested text but
  repeated typing. Feedback changed invalid generations without ensuring
  valid actions. Runtime evidence audit passed; all 13 generation contexts
  passed after correcting a separate auditor's handling of redacted numeric
  TYPE parameters. The original auditor/failure were preserved; no episode
  was rerun or rescored.
- `miniwob-completion-development-v1`: the combined control, scoring, memory
  context and retry interfaces, all four existing development families and E0–E3.
  **16/16 episodes completed, no runtime errors or invalid/overlap exclusions.**
  E0 1/4, E1 0/4, E2 1/4, E3 2/4. All 13 generation contexts passed auditing;
  six E3 calls contained the exact retrieved examples. All 165 producing source
  files remained unchanged. The paired E3-minus-E2 development difference is
  +25 percentage points (one improved pair, exact p=1, Holm p=1).

The combined run retains the same small contrast and its E2-regression caveat.
E3 succeeds at `click-test` where E2 attempts TYPE on a button; both succeed at
`click-button`. Both type the text correctly but repeat typing, and neither
completes the button sequence. These are retained model/action-selection
failures. The corrected interfaces have not eliminated all task failures.

## Proposed full run

`/home/aiub/kiyas/table2-evidence/miniwob-table2-completion-v1/` contains a frozen
24-episode package: six existing evaluation families × E0–E3, with matched
resets, unchanged budgets, frozen training-only memory and the new full-credit
scoring rule. Overlap exclusions remove an entire paired family without
replacement. The plan was frozen before the combined development outcomes were
available; readiness is conditional on valid execution/audits, not a favorable
memory result.

This is a post-debugging MiniWoB replication on previously observed families,
not fresh held-out confirmation. It cannot supply a WebArena result or guarantee
positive memory improvement. The new table remains N/R until its episodes and
audits complete; the earlier tables remain separately available.

The launcher verified the checkpoint, all 14 base snapshot files, export files,
frozen memory, installed versions, source hashes, seed derivation, development
audits and prior evaluation artifacts. Its receipt is
`miniwob-table2-completion-v1/readiness.json`, status
`READY_FOR_FULL_CREDIT_MINIWOB_REPLICATION`, with `executed: false`.

The frozen plan SHA-256 is
`cbb7ae53c4f26192979b72d47853106fe57d209442698be40091604868c76b9d`.
The plan records its earlier pending status at freeze time; the subsequent
readiness receipt records the passed checks without changing that plan.

To execute the prepared run from the existing repository:

```bash
PYTHONPATH=src .venv/bin/python /home/aiub/kiyas/table2-evidence/miniwob-table2-completion-v1/launch.py --run
```

The launcher rechecks the frozen inputs, refuses to overwrite a started run,
and stops if another model is using the GPU. It then runs the 24 paired episodes,
audits and reporting. No task replacement, output correction, retraining or
further calibration is part of this package. Further full-run outcomes must be
reported as they occur, including null or negative memory differences.
