# PC-01 checkpoint series: six-action behavior and agent implications

2026-09-13. **All ten saved epoch checkpoints (0–9) and the rolling `last.ckpt`
were inspected on CPU.** Stored diagnostics were checked against the exported
report; no model inference, training, dataset-row/image access, or checkpoint
substitution took place. This covers the PC-01 v2.8 DGX seed-42 full run, not
every historical backbone experiment in the project.

Reproducible audit: [script](../scripts/analyze_pc01_checkpoint_series.py).
Full hashes, confusion matrices, weight checks, metrics and selection replay:
[machine-readable evidence](evidence/pc01_checkpoint_series_audit.json).

## What the checkpoint files establish

- Each epoch has a six-class linear classifier with weight shape `[6, 256]` and
  bias shape `[6]`. All action-head tensors checked are finite.
- All ten classifier weight hashes differ; every adjacent epoch has a nonzero
  classifier-weight change. No class has a zero classifier-weight row. These
  checks establish saved parameter updates, not adequate learning of every class.
- All ten checkpoint configs have the same identity. The epoch-6 file hash
  matches the selected-checkpoint hash in the report.
- Each epoch's embedded diagnostic record exactly matches `diagnostics.json`.
  Action accuracy and macro-F1 recomputed from it agree with `report.json`.
- `last.ckpt` is epoch 9, marked complete, with the same action-classifier weights
  as the epoch-9 file. It is not an eleventh independently trained candidate.

Training counts are CLICK 3,851; TYPE 3,880; SELECT 3,906; SCROLL 4,023;
NAVIGATE 4,770; PRESS_KEY 3,677. Action-class weights are nonzero, approximately
0.88–1.10. Missing classes and a zero action-loss coefficient are not supported
explanations. The observed raw action loss decreased from 1.2616 to 1.0074;
decreasing loss did not yield reliable six-class predictions.

## All epoch results

The table uses the same 7,861 original validation examples for every epoch.
Recall is the fraction of examples of that action predicted correctly; it is
not browser task completion. High recall alone can result from predicting a
class too often.

| Epoch | Accuracy | Macro-F1 | CLICK recall | TYPE recall | SELECT recall | SCROLL recall | NAVIGATE recall | PRESS_KEY recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38.54% | 0.3094 | 97.98% | 70.06% | 28.41% | 1.86% | 0.00% | 39.36% |
| 1 | 37.41% | 0.2749 | 0.00% | 13.19% | 28.56% | 99.92% | 0.00% | 92.88% |
| 2 | 38.02% | 0.2853 | 100.00% | 16.59% | 28.56% | 0.00% | 0.00% | 93.63% |
| 3 | 38.94% | 0.3094 | 100.00% | 36.96% | 28.56% | 0.00% | 0.00% | 77.55% |
| 4 | 41.53% | 0.3257 | 3.26% | 31.40% | 28.56% | 0.00% | 97.73% | 82.24% |
| 5 | 42.31% | 0.3261 | 0.00% | 40.97% | 28.56% | 0.00% | 100.00% | 77.64% |
| 6 | 42.17% | 0.3267 | 0.00% | 48.46% | 28.56% | 0.00% | 100.00% | 68.59% |
| 7 | 39.02% | 0.3096 | 0.00% | 35.26% | 28.56% | 98.92% | 0.07% | 80.49% |
| 8 | 39.13% | 0.3091 | 0.00% | 33.10% | 28.56% | 99.46% | 0.13% | 82.91% |
| 9 | 41.60% | 0.3319 | 0.08% | 31.10% | 28.56% | 6.26% | 94.53% | 83.67% |

The most informative pattern is the confusion among CLICK, SCROLL and NAVIGATE:

- Their validation supports total 1,287 + 1,293 + 1,499 = **4,079**.
- Epochs 2 and 3 predict **CLICK for every one** of those examples.
- Epochs 5 and 6 predict **NAVIGATE for every one** of them.
- Epoch 1 predicts SCROLL for 4,078 of them; epochs 7 and 8 again predict
  SCROLL for almost the entire group.

Thus this is not just epoch 6 forgetting SCROLL late in training. The saved
classifier repeatedly fails to distinguish these three categories and changes
which category wins. No saved epoch demonstrates reliable performance across
all six classes. Epoch 9 has the best macro-F1, 0.3319, but only 0.08% CLICK
recall and 6.26% SCROLL recall. Swapping to it is not an established solution.

## Why epoch 6 was selected despite this problem

The replayed rule is `all_gates_then_outcome_mcc`. All ten epochs pass its
eligibility gates; epoch 6 has the highest outcome MCC, 0.6242.

The [selection code](../src/web_agent/train/selection.py) requires action
accuracy at least 0.396 − 0.03 = **0.366**. It has no per-action minimum recall
gate and does not rank eligible checkpoints by action macro-F1 or task
completion. It also uses minimum box functionality gates, not a requirement for
reliable browser localization. Epoch 6's mean box IoU is 0.1010 and its IoU≥0.5
recall is 8.09% on the 2,349 valid-box examples. Across epochs, mean IoU ranges
from 0.0724 to 0.1182.

**The selection implementation follows its declared rule. The rule was
insufficient to establish autonomous six-action policy competence.** Treating
its PASS as agent readiness was an interpretation error. Do not retroactively
replace the registered rule or historical results to hide that limitation.

## What the training path explains, and what remains unknown

Verified from [the label projection](../src/web_agent/data/gold_dataset.py),
[model routing](../src/web_agent/models/model.py),
[loss](../src/web_agent/models/loss.py) and
[metric collection](../src/web_agent/train/trainer.py):

1. The action head receives the causal pre-action screenshot/task stream.
   Diagnosis and memory use the post-action stream. The chosen action is not
   supplied as an answer in the pre-action input.
2. Action supervision uses `labels.action_type`; original rows receive the
   default action loss mask of one, with no success-only filter. This predicts
   the recorded action, including an action from a failed attempt. It is not
   automatically a label for the best next action or corrective sequence.
   Correctly reviewed labels can accurately describe an unsuccessful attempt.
3. The action loss is six-way cross entropy; it has a nonzero nominal weight
   of 0.15, label smoothing 0.1, nonzero class weights and dynamic uncertainty
   weighting. The logged action multiplier increases during training. There
   is no evidence here that action learning was disabled.
4. Label encoding and validation decoding use the shared six-class map.
   Validation directly takes the head argmax; there is no executor or JSON
   conversion in these saved validation results.
5. The heads train classification and box objectives, not an autoregressive
   loss for emitting browser JSON with the exact TYPE text, URL or key. Those
   required execution parameters need a separate model-generated interface.

The confusion pattern supports **failure to separate these classes in the
learned decision**, but it does not identify the exact underlying cause.
Insufficient distinguishing information in pre-action inputs, representation
limitations and optimization effects remain hypotheses. Aggregate confusion
matrices and final weights cannot prove that images were duplicated, labels
were wrong, gradients conflicted, or a processor lost information. No such
dataset defect is claimed. Existing saved reports are enough to establish the
limitation; this audit does not reopen the dataset review.

The current hybrid all-CLICK issue is separate: its JSON actor is the frozen
unadapted base, whereas the selected checkpoint supplies advice. See the
[training/runtime alignment diagnostic](TABLE2_TRAINING_RUNTIME_ALIGNMENT_AUDIT.md).
That diagnostic found plans mentioning typing followed by CLICK JSON; it did
not fix the issue or establish live continuation.

## Build the agent while preserving the research objective

The proposal's four-pillar objective remains failure-aware multimodal acting
with executed recovery and useful corrective memory. Having four implemented
components and proving four independent improvements are different claims.

The historical `ARCHITECTURE_DESIGN.md` still contains five-action and
“SCROLL/NAVIGATE have no training data” passages for an older source. Those do
not describe this Gold v2.8 run: the saved configs, six-row heads and aggregate
counts above establish its actual six-action contract. Its existing supersession
notice must be respected when reading those historical sections.

There are two existing system definitions; do not switch them silently:

| Definition | Trained action head's authority | Current limitation |
|---|---|---|
| Original E1–E3 | Selects normal action class and box directly; recovery is added in E2/E3 | Weak class separation and localization remain faithful model failures. |
| Approved hybrid H1–H3 | Supplies action/box advice; frozen base chooses action/control/value | The base does not reliably turn a correct verbal plan into a useful action sequence. |

**Continue the approved hybrid development under its own identity, retaining
epoch 6.** It allows us to test the trained failure/recovery and memory modules
without pretending the action head is already a competent sole actor. This is
the current architecture, not a newly discovered cure. It still needs a working
generated-action sequence, and it cannot be described as a demonstration that
the trained box head itself became accurate.

| Pillar | Required implementation and evidence |
|---|---|
| P2 multimodal decision | Preserve trained screenshot/task processing and show the exact observation used for each model decision. |
| P3 action/grounding | Retain all six available actions, log trained advice separately, and execute the generator's exact action/value on its selected current target. Compare action selection with and without advice on development states. |
| P1 resilience | Feed completed failed transitions to learned diagnosis, select and execute recovery, then record learned assessment separately from execution and actual termination. |
| P4 memory | Preserve train-only frozen retrieval and the no-memory decision. Record admission, exposure, changes to actions and completion separately; do not equate storage accuracy with retrieval usefulness. |

Concrete next development work:

1. Declare a small six-action **input-contract diagnostic** before inference.
   Use observable development states representing each operation, not final
   evaluation outcomes. Compare the same generator with task/current controls,
   then added trained advice, then causal history. Keep screenshots and decoding
   fixed. This tests whether advice/history contributes to the action mismatch;
   it must not force the expected action or insert task answers.
2. Use those findings to make one documented action-interface revision if they
   support it. Keep the model as the source of action, target and exact value.
   A tentative plan is not an executed action; do not translate a wrong CLICK
   into a correct TYPE through a task-specific rule.
3. Verify successive live actions on development tasks, including text entry
   and continuation. Preserve all poor decisions. Test execution support for
   all six separately from whether the model chooses each appropriately.
4. Only then assess recovery and memory increments in matched H systems. The
   present strict memory filter may continue to abstain. Changing it to admit
   incomplete label advice is a separate declared method change, not proof of
   useful memory or a reason to fabricate corrective values/reflections.
5. Freeze the revised method and evaluation once implementation is auditable.
   Preserve original E/H results, equal budgets and both primary contrasts.
   Successful demonstrations alone cannot establish an overall performance gain.

If the proposal specifically requires the trained action head alone to be a
reliable six-action actor, the existing results do not establish that capability.
We cannot promise to obtain it through wiring changes with frozen weights.
The hybrid route preserves the four functional pillars with the stated change
in P3 authority; its completion benefit must still be measured.

## Delivery and limits

- [x] Inspect all ten PC-01 epoch files and the rolling checkpoint.
- [x] Verify six-class weights, embedded diagnostics and selected identity.
- [x] Compare all six recalls and locate the recurrent confusion pattern.
- [x] Replay checkpoint selection and identify its competence-gate limitation.
- [x] Explain training targets and distinguish evidence from possible causes.
- [x] Map the existing agent definitions to the research objective.
- [ ] Execute the newly proposed six-action contract diagnostic under a frozen
      development plan; it has not been run by this checkpoint audit.

No checkpoint was changed or promoted, no head ensemble was created, and no
final evaluation was launched. Per-class best epochs must not be combined using
the true action label; that would use information unavailable to the agent.
