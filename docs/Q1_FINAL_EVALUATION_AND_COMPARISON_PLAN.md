# Q1 Final Evaluation and Related-Work Comparison Plan

**Status:** final evidence plan  
**Date:** 2026-07-28  
**Project:** four-pillar failure-aware multimodal web agent  
**PC-01 provisional candidate currently available:** Qwen2-VL-2B with QLoRA;
the final primary backbone remains unselected pending the registered
validation-only PC-01/PC-02/PC-03 comparison

> **Current-state correction (2026-09-04):** Sections 3.1--3.3 preserve the
> 2026-07-28 v2.7 mini snapshot and its then-current blockers; they are not the
> present experiment state. PC-01 has since completed its Gold v2.8 seed-42
> validation run and is provisional only. Use
> `docs/QWEN2VL_2B_GOLD_V2_8_DGX_SEED42_FULL_RESULTS.md` for that candidate and
> `docs/TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md` for the controlling
> E0--E3 protocol. PC-02/PC-03 and final validation-only promotion remain open.

This document defines the experiments, metrics, comparisons, statistical
tests, artifacts, and reporting rules required before the thesis can claim
that the proposed web agent works. It replaces the evaluation/model-order
instructions in older planning documents where they conflict with this plan.
Older mini results remain diagnostic evidence; they are not final test or
end-to-end results.

---

## 1. Central thesis objective

The project is not trying to maximize one classification accuracy value. The
central objective is:

> Build and evaluate a web agent that uses multimodal page state to choose a
> grounded action, detects when an executed action fails, selects and evaluates
> a recovery, and uses previous recovery experience to improve later decisions.

The four pillars are:

1. **Pillar 1 - Failure-aware resilience:** identify execution failure, diagnose
   its type, estimate confidence, and determine whether recovery is required.
2. **Pillar 2 - Multimodal decision:** use screenshot and text/state
   information together.
3. **Pillar 3 - Multi-tool action and grounding:** select an action and, when
   applicable, localize its UI target.
4. **Pillar 4 - Corrective memory:** store useful completed experiences,
   retrieve relevant prior recoveries, and use retrieved evidence during
   decision making.

The final evidence must answer two different questions:

- **Component performance:** did each learned module predict its own targets
  correctly?
- **End-to-end agent performance:** did the complete agent finish tasks and
  recover from failures during sequential execution?

Neither type of evaluation replaces the other.

---

## 2. Research questions

The paper should register the following research questions before opening the
locked test split.

### RQ1 - Failure understanding

How accurately does the model distinguish successful and failed actions and
identify the type of failure under class imbalance?

### RQ2 - Multimodal contribution

Does using screenshot and text/state together improve prediction or end-to-end
success compared with removing either modality?

### RQ3 - Action selection and grounding

How accurately does the model select the required browser action and localize
the target element?

### RQ4 - Recovery

After a verified failure, how accurately does the agent choose a recovery
strategy, and how often does the executed recovery restore progress?

### RQ5 - Corrective memory

Does retrieving relevant completed experiences improve recovery and task
success compared with the same agent without memory guidance?

### RQ6 - Complete system

Does the complete four-pillar agent improve task success and reduce repeated
errors relative to controlled baselines under the same tasks, backbone,
environment, budget, and evaluator?

---

## 3. Historical 2026-07-28 evidence snapshot and its correct interpretation

### 3.1 Controlled 5k mini result

The then-latest locally preserved v2.7 resume report used:

- 5,000 training rows;
- 500 validation rows;
- five zero-based epochs, `0` through `4`;
- validation-only checkpoint selection;
- selected epoch `3`;
- `test_rows_read = 0`;
- checkpoint round-trip verification;
- selection rule `all_gates_then_outcome_mcc`.

Selected-checkpoint validation results include:

| Metric | Current mini value |
| --- | ---: |
| Outcome MCC | 0.5616 |
| Failure macro-F1 | 0.7806 |
| Failure-type macro-F1 | 0.5577 |
| Action macro-F1 | 0.3341 |
| Recovery-strategy macro-F1, attempted rows only | 0.5544 |
| Recovery-outcome MCC | 0.6963 |
| Memory-update balanced accuracy | 0.8266 |
| BBox mean IoU | 0.0788 |
| BBox Recall@IoU50 | 0.0577 |

The source artifact is:
`kaggle_outputs/notebook8052d2dcc0-2026-07-27/gold_recovery_v2_7_resume_report.json`.

These values prove that the controlled pipeline runs, learns non-trivial
signals, saves and reloads a checkpoint, and respects the locked test boundary.
They do **not** prove final generalization or end-to-end task completion.

### 3.2 Historical data blockers at that snapshot

The accepted existing-data audit found:

- 2,534 invalid bbox annotations among 11,602 bbox-labelled train/validation
  rows;
- invalid bbox labels are masked, not automatically corrected;
- reviewer decisions completed: zero at the time of the accepted audit;
- RETRY support in train/validation: zero;
- ABORT support in train/validation: zero;
- all existing attempted recoveries have a causal next-step transition.

Therefore:

- headline/full training is not yet permitted by the accepted audit;
- the two-person review must be reconciled first;
- RETRY and ABORT require genuine targeted collection if the paper claims that
  all six strategies are learned;
- corrected/excluded validation data cannot be directly compared with old
  v2.7/v14 values until the old selected checkpoint is re-evaluated on the
  exact same review overlay.

### 3.3 Historical pillar status at that snapshot

| Pillar | Current mini evidence | Remaining final evidence |
| --- | --- | --- |
| P1 | Outcome, failure type, needs-recovery, and recovery-outcome heads run and learn | Full-data test metrics and end-to-end recovery |
| P2 | Qwen receives image and text/state together | Modality comparison or sensitivity study |
| P3 | Action and bbox heads run | Reviewed bbox labels, stronger grounding, end-to-end action execution |
| P4 | Memory-update and recovery heads run | Train-only retrieval index, runtime use, retrieval evaluation, no-memory comparison |

---

## 4. Why two evaluation levels are required

### 4.1 Component evaluation

Component evaluation treats each held-out record or causal transition as a
labelled prediction problem. It can isolate which module is strong or weak.

Example:

```text
failure state + executed recovery + post-recovery state
    -> predicted recovery outcome
    -> compare with labelled recovery outcome
```

This is necessary for diagnosing class imbalance, localization failure,
calibration, and recovery-strategy collapse.

### 4.2 End-to-end evaluation

End-to-end evaluation treats an entire task episode as the unit:

```text
task instruction
    -> observe state
    -> choose and execute action
    -> observe post-action state
    -> diagnose result
    -> recover if needed
    -> repeat until success, abort, timeout, or step limit
```

A model can have good record-level metrics and still fail tasks because a
single wrong action can invalidate a long trajectory. Conversely, task success
alone can hide a model that repeatedly makes errors but succeeds through many
retries. Q1-level evidence therefore reports both.

---

## 5. Component metrics

### 5.1 Outcome/failure MCC and macro-F1

#### Definitions

- **Macro-F1:** calculate F1 separately for every class and take the unweighted
  mean.
- **Matthews correlation coefficient (MCC):** correlation between prediction
  and ground truth. For binary outcome it uses all four confusion-matrix cells.
  Multiclass MCC is used for failure type.

#### Why selected

The dataset is imbalanced. Plain accuracy can appear high when a model predicts
the majority class. Macro-F1 gives each class equal importance, while MCC
penalizes majority collapse and summarizes the complete confusion matrix.

#### Required reporting

- outcome MCC, macro-F1, balanced accuracy, success recall, and failure recall;
- failure-type MCC and macro-F1;
- per-class precision, recall, F1, and support;
- confusion matrices;
- majority and stratified-random baselines.

#### Related-paper connection

BacktrackAgent trains a binary Judger using the action, pre-action page, and
post-action page, but reports downstream task/step accuracy rather than MCC.
Our MCC and macro-F1 are additional imbalance-aware diagnostic metrics.

Source:
<https://arxiv.org/pdf/2505.20660>

### 5.2 Action macro-F1

#### Definition

Compute F1 for every action class, then average all supported classes equally.
Also report accuracy, balanced accuracy, MCC, per-class F1, and class support.

#### Why selected

An overall action accuracy can be dominated by CLICK. Macro-F1 exposes weak
SCROLL, SELECT, NAVIGATE, PRESS_KEY, and TYPE behaviour.

#### Required reporting

- action macro-F1 and MCC;
- per-action precision, recall, F1, and support;
- predicted and true class distributions;
- majority baseline;
- action-validity rate during end-to-end execution.

#### Related-paper connection

BacktrackAgent reports task-level and step-level action accuracy, including IoU
and text matching. OSCAR additionally reports operation F1 in its static GUI
navigation appendix. Our action macro-F1 is the class-balanced analogue needed
for this dataset.

Sources:
<https://arxiv.org/pdf/2505.20660> and
<https://arxiv.org/pdf/2410.18963>

### 5.3 Recovery-strategy macro-F1

#### Definition

Calculate macro-F1 only on rows where recovery was genuinely attempted. Report
the six-class metric only when all claimed classes have non-zero test support.
Also report a supported-class macro-F1 that explicitly lists the included
classes.

#### Why selected

A recovery classifier can achieve high accuracy by predicting NONE for most
rows. Attempted-only macro-F1 measures whether the model can distinguish the
actual recovery strategies.

#### Required reporting

- attempted-only macro-F1 and MCC;
- per-strategy precision, recall, F1, and support;
- number of predicted classes;
- majority-strategy baseline;
- RETRY and ABORT reported only after genuine examples exist;
- no zero-support class may be presented as learned.

#### Related-paper connection

BacktrackAgent evaluates an explicit Reflector, while ReUseIt evaluates
fallback actions against unguided retry. Neither reports a six-class
recovery-strategy macro-F1. This is a thesis-specific diagnostic that makes the
recovery mechanism auditable.

Sources:
<https://arxiv.org/pdf/2505.20660> and
<https://arxiv.org/html/2510.14308>

### 5.4 Recovery-outcome MCC

#### Definition

On complete causal transitions, predict whether the executed recovery
succeeded:

```text
failure state
  + executed recovery action
  + post-recovery state
  -> recovery success/failure
```

Calculate MCC, macro-F1, balanced accuracy, Brier score, and per-class recall.

#### Why selected

Choosing a recovery label does not prove that the recovery worked. This metric
tests causal post-action evidence and remains informative when recovery
success/failure counts are imbalanced.

#### Related-paper connection

BacktrackAgent explicitly uses the actual page after action execution and
reports that actual execution added 5.65 task-success points, while simulated
execution added only 0.70 points in its ablation. ReUseIt shows that guided
fallback actions increase average success from 50.1% without fallback actions
to 70.1% with them. These papers motivate post-action recovery evaluation,
although they do not report recovery-outcome MCC.

Sources:
<https://arxiv.org/pdf/2505.20660> and
<https://arxiv.org/html/2510.14308>

### 5.5 Memory metrics

Memory must be evaluated at two levels.

#### Memory-update prediction

Report:

- MCC;
- macro-F1;
- balanced accuracy;
- per-class precision/recall/F1;
- majority baseline.

This tests whether the model predicts which experiences should be stored. It
does not prove that retrieval helps the agent.

#### Retrieval quality

Build the memory index from training episodes only. For validation/test queries
exclude:

- the current episode;
- the same task ID;
- near-duplicate trajectories;
- every test-derived memory item.

Report:

- Recall@1, Recall@3, and Recall@5 for retrieving a relevant recovery;
- mean reciprocal rank (MRR);
- strategy agreement@K;
- retrieval coverage;
- irrelevant-memory rate;
- intervention precision: proportion of memory interventions that improve or
  preserve the next decision;
- task-success delta between memory enabled and disabled.

#### Why selected

Pillar 4 claims memory-driven planning. A memory-update classifier alone cannot
support that claim. Retrieval must influence a decision and improve an
observable outcome.

#### Related-paper connection

WebCoach stores completed trajectories, retrieves top-5 experiences through a
FAISS/HNSW index, and injects advice through a Coach. On WebVoyager it reports:

- Qwen-VL-32B: 49.5% to 57.1% task success;
- Skywork-38B: 47.3% to 61.4%;
- Qwen-VL-7B: 32.8% to 31.1%.

The 7B degradation demonstrates why retrieval quality and downstream effect
must both be measured.

Source:
<https://arxiv.org/html/2511.12997>

### 5.6 BBox IoU and Recall@IoU50

#### Definitions

- **Intersection over Union (IoU):** intersection area divided by union area
  between the predicted and reference bbox.
- **Mean IoU:** average IoU across all valid bbox-labelled examples.
- **Median IoU:** included because localization errors can be highly skewed.
- **Recall@IoU50:** proportion of valid bbox examples with IoU at least 0.50.

#### Why selected

Coordinate MAE does not prove that two boxes overlap. A prediction can have
moderate coordinate error and still miss the element completely. Mean IoU
measures overlap quality; Recall@IoU50 measures how often localization reaches
a practically meaningful threshold.

#### Required reporting

- mean and median IoU;
- Recall@IoU25 and Recall@IoU50;
- bbox MAE only as a secondary metric;
- results by action type and element-size bucket;
- valid bbox support and masked-invalid count;
- no invalid or manually rejected annotation in the denominator.

#### Related-paper connection

BacktrackAgent reports both task-level and step-level IoU action accuracy.
OSCAR uses Set-of-Mark GUI grounding from screenshots and accessibility-tree
elements. Our IoU/Recall metrics are the explicit localization evidence for
Pillar 3.

Sources:
<https://arxiv.org/pdf/2505.20660> and
<https://arxiv.org/pdf/2410.18963>

### 5.7 Calibration

#### Definitions

- expected calibration error (ECE);
- Brier score for binary outcome and recovery outcome;
- reliability diagrams;
- confidence MAE where a continuous target is available.

#### Why selected

The Decision Combiner uses confidence to decide when to act, query memory,
recover, or abort. A model can be accurate but dangerously overconfident.
Calibration measures whether a prediction made with approximately 80%
confidence is correct approximately 80% of the time.

#### Required reporting

- outcome ECE and Brier score;
- recovery-outcome ECE and Brier score;
- reliability plots;
- coverage-risk curve for any confidence threshold;
- threshold selected on validation only.

#### Related-paper connection

The five selected related papers primarily report success/accuracy and do not
provide the same calibration suite. Calibration is an additional reliability
contribution of this thesis and must not be described as a metric copied from
those papers.

---

## 6. End-to-end metrics

All end-to-end metrics must be computed from frozen episode logs produced by
the same evaluator, environment, task set, action budget, timeout, and success
oracle for every compared system.

### 6.1 Task success rate

```text
TSR = successfully completed tasks / all attempted tasks
```

#### Why selected

This is the clearest measure that the web agent works. It is the primary metric
used across the five related papers.

#### Related-paper values

- OSCAR: 24.5% on OSWorld and 61.6% on AndroidWorld.
- BacktrackAgent: 54.11% task success on Mobile3M.
- ReUseIt: 24.2% task-only to 70.1% with reusable workflows.
- SkillWeaver: 22.6% to 29.8% on WebArena and 40.2% to 56.2% on its live-site
  evaluation.
- WebCoach: 47.3% to 61.4% with Skywork-38B on WebVoyager.

These values belong to different benchmarks and are contextual references, not
a single common leaderboard.

### 6.2 Recovery success rate

```text
Recovery success rate =
  recovery attempts that restore valid task progress
  / all executed recovery attempts
```

A recovery counts as successful only when post-recovery evidence satisfies a
pre-registered progress rule. Merely executing RETRY, REPLAN, BACKTRACK,
ALTERNATIVE_TARGET, or ABORT is not success.

Report:

- overall recovery success;
- per-strategy recovery success;
- per-failure-type recovery success;
- 95% confidence intervals;
- denominator and number of unique episodes.

#### Why selected

Task success cannot reveal whether the recovery mechanism was responsible.
This metric isolates the operational effectiveness of Pillars 1 and 4.

#### Related-paper connection

BacktrackAgent and ReUseIt demonstrate recovery through task-success ablations,
but do not report this exact normalized metric. It is an additional
mechanism-level end-to-end measure for our system.

### 6.3 Success after an initial failure

```text
Success-after-failure =
  tasks that contain at least one verified failure and later finish successfully
  / tasks that contain at least one verified failure
```

#### Why selected

Overall task success mixes easy no-failure tasks with genuinely recovered
tasks. This metric directly answers: “When the agent first goes wrong, can it
still finish?”

#### Related-paper connection

It operationalizes the behaviour targeted by BacktrackAgent's backtracking,
ReUseIt's fallback actions, and WebCoach's failure-aware advice. The selected
papers do not report this exact formula, so it is a thesis-specific primary
recovery metric.

### 6.4 Repeated-error/loop rate

Primary episode-level definition:

```text
Loop episode rate =
  tasks containing at least one confirmed repeated-error loop
  / all attempted tasks
```

Secondary event-level definition:

```text
Repeated-error event rate =
  repeated ineffective state-action events
  / all executed actions
```

A repeated error must be identified using a frozen rule, for example:

- repeated equivalent state and equivalent ineffective action;
- repeated visits to the same state cycle;
- LOOP_DETECTED confirmed by the evaluator;
- no measurable task progress across the registered window.

#### Why selected

An agent may finish some tasks while wasting many actions in cycles. A
failure-aware agent should reduce repeated mistakes, not merely increase retry
count.

#### Related-paper connection

ReUseIt reports that unguided retries often repeat the first error. WebCoach
targets repeated links, dead ends, and navigation loops and reports fewer or
similar average steps with stronger models. The exact rate above is our
explicit operational measure.

Sources:
<https://arxiv.org/html/2510.14308> and
<https://arxiv.org/html/2511.12997>

### 6.5 Average task steps

```text
Average task steps = total executed actions / all attempted tasks
```

Also report:

- median and interquartile range;
- successful-task steps;
- failed-task steps;
- step-limit termination rate.

#### Why selected

Success achieved through excessive trial and error may be impractical. Steps
measure navigation efficiency and help determine whether recovery avoids or
adds waste.

#### Related-paper connection

WebCoach reports success rate and average steps together. Its dynamic
Skywork-38B configuration improves success from 47.3% to 61.4% while changing
average steps from 10.7 to 10.2.

Source:
<https://arxiv.org/html/2511.12997>

### 6.6 Average recovery attempts

Primary definition:

```text
Average recovery attempts =
  total executed recovery actions
  / tasks containing at least one recovery
```

Also report:

- median and maximum;
- attempts per successful recovered task;
- attempts per unrecovered task;
- hard-abort rate;
- recovery-budget exhaustion rate.

#### Why selected

Two agents can have the same recovery success while one requires many more
attempts. This metric measures recovery efficiency and verifies the hard-stop
policy.

#### Related-paper connection

BacktrackAgent repeats verifier-judger-reflector cycles until acceptance or a
maximum reflection count. OSCAR studies replanning occurrences. Neither of the
selected papers uses this exact mean as a headline metric; we include it to
make recovery cost explicit.

Sources:
<https://arxiv.org/pdf/2505.20660> and
<https://arxiv.org/pdf/2410.18963>

---

## 7. Related-work numerical context

| Paper | Backbone/system | Benchmark | Baseline | Proposed | Main mechanism |
| --- | --- | --- | ---: | ---: | --- |
| OSCAR | GPT-4o for OSWorld | OSWorld | FRIDAY 18.8% | 24.5% | GUI grounding, state machine, verification, replanning |
| OSCAR | adapted mobile agent | AndroidWorld | AppAgent 59.9% | 61.6% | same state-aware planning approach |
| BacktrackAgent | Qwen2-VL-7B | Mobile3M | ReachAgent 46.52% | 54.11% | Verifier, Judger, Reflector, actual post-action page |
| BacktrackAgent | Qwen2-VL-7B | Auto-UI | MobileVLM 25.53% | 29.72% | trained judgment/reflection datasets |
| ReUseIt | Magentic-UI workflow agent | 15 repetitive task families | Task-only 24.2% | 70.1% | condition checks and fallback actions |
| SkillWeaver | GPT-4o CodeAct agent | WebArena | 22.6% | 29.8% | explored and tested reusable APIs |
| SkillWeaver | GPT-4o CodeAct agent | live-site tasks | 40.2% | 56.2% | synthesized website skills |
| WebCoach | Qwen-VL-32B | WebVoyager | 49.5% | 57.1% | retrieved cross-session memory and coaching |
| WebCoach | Skywork-38B | WebVoyager | 47.3% | 61.4% | dynamic self-memory and coaching |

### Interpretation rule

Do not place our result beside these values as if every number came from the
same test. A direct claim such as “our 65% is better than OSCAR's 24.5%” is
invalid when the benchmark, task distribution, backbone, success oracle, and
action budget differ.

Use two result tables in the paper:

1. **Controlled same-protocol comparison:** all our baselines/ablations and the
   full model on exactly the same tasks.
2. **Published contextual comparison:** the table above, with benchmark and
   backbone visible.

A direct comparison to a published system is allowed only when we reproduce it
or evaluate our system on the same public benchmark and protocol.

---

## 8. Systems that must be compared

### 8.1 Required component baselines

1. Majority-class baseline.
2. Stratified-random baseline.
3. Frozen/unadapted Qwen feature baseline where technically meaningful.
4. Full trained model.

### 8.2 Required end-to-end systems

| ID | Configuration | Question answered |
| --- | --- | --- |
| E0 | Base Qwen web agent | What can the backbone do without our framework? |
| E1 | Trained action model without failure-triggered recovery | Does task-specific training help action execution? |
| E2 | E1 + failure detection and recovery, memory disabled | What do Pillar 1 and recovery add? |
| E3 | Full four-pillar agent with memory retrieval | Does corrective memory add value? |

Every system must use:

- identical backbone size where the comparison is intended as an ablation;
- identical task instructions and start states;
- identical action and recovery budgets;
- identical browser/environment versions;
- identical timeout and success oracle;
- identical random seeds or paired task repeats.

### 8.3 Pillar 2 experiment

The preferred Q1 ablation retrains controlled variants:

- full image + text/state;
- image-only;
- text/state-only.

Use the same train/validation split, seed, optimizer, epoch budget, and
selection rule. This is the strongest evidence that multimodal input adds
value.

If compute does not permit retraining, perform inference-time modality
occlusion:

- replace image with a registered neutral image;
- replace text/state with fixed neutral text;
- preserve all other inputs.

This cheaper experiment must be called a **modality sensitivity analysis**, not
a causal training ablation.

### 8.4 Pillar 4 experiment

Compare the same selected checkpoint with:

- retrieval disabled;
- retrieval enabled but advice not injected, as a retrieval-only diagnostic;
- retrieval enabled and advice/strategy guidance injected.

This isolates indexing, retrieval relevance, and intervention effect.

---

## 9. End-to-end benchmark decision

Before full training begins, freeze one primary end-to-end evaluation protocol.

### Preferred Q1 design

- use a recognized public web-agent benchmark compatible with the action
  executor;
- run the official success evaluator;
- report the complete benchmark where resources permit;
- otherwise register the subset and sampling rule before testing;
- supplement it with the project's held-out failure/recovery task suite.

WebVoyager is particularly relevant to WebCoach comparison, WebArena to
SkillWeaver, and Mobile3M/Auto-UI to BacktrackAgent. They are not
interchangeable. Benchmark selection must consider environment availability,
action schema compatibility, reproducibility, and compute.

### Minimum defensible design if a public benchmark cannot be integrated

- at least 100 held-out task episodes;
- multiple websites and task types;
- fixed start states where possible;
- predetermined success oracle;
- at least one initial-failure subset;
- complete action/state/recovery logs;
- paired evaluation of E0-E3;
- report this as a controlled project benchmark, not direct SOTA comparison.

The evaluator must be implemented and smoke-tested with the mini checkpoint
before full training. Final end-to-end values are calculated only after the
checkpoint and protocol are frozen.

---

## 10. Statistical reporting

### 10.1 Repeated runs

- Primary full model: target three training seeds.
- Expensive ablations: at least one fixed seed; three where resources permit.
- Stochastic browser agents: repeat tasks when environment cost permits.
- Never hide failed seeds or interrupted runs.

### 10.2 Uncertainty

Report:

- mean and standard deviation across training seeds;
- 95% confidence intervals for task/recovery success;
- bootstrap confidence intervals for macro-F1, MCC, and TSR where suitable;
- sample counts and class support beside every result.

### 10.3 Paired tests

Because systems attempt the same task set:

- use McNemar's test or paired bootstrap for success/failure outcomes;
- use paired bootstrap or a paired non-parametric test for step counts;
- report absolute improvement, relative improvement, confidence interval, and
  effect size;
- apply Holm correction when many ablations are tested.

Statistical significance must not replace practical significance. A very small
gain with large latency may not be useful.

---

## 11. Efficiency and implementation evidence

The project also claims an implemented agent, so report:

- trainable and total parameter counts;
- model/backbone size;
- GPU type and GPU hours;
- peak GPU memory;
- inference latency per decision;
- recovery/memory retrieval latency;
- average task wall-clock time;
- average input/output token use where applicable;
- retrieval-index size and query latency;
- checkpoint and code commit hashes.

Related systems use much larger models in some comparisons. For example,
WebCoach's strongest result uses a 38B actor plus an 8B Coach, while the
provisional PC-01 candidate is 2B. The final primary backbone is not yet
selected. Model size and cost must remain visible so that a smaller but
competitive model is evaluated fairly.

---

## 12. Leakage and validity controls

### Data

- group splits by task/trajectory/site as registered;
- never split individual steps from one episode across train and test;
- remove exact and near duplicates across splits;
- preserve the original dataset and use versioned overlays/supplements;
- record every correction and exclusion with provenance.

### Training and selection

- train on training only;
- select epochs, thresholds, loss weights, and gates using validation only;
- read the locked test only after final model/configuration freeze;
- never tune after observing locked-test results.

### Memory

- build the final memory store from training trajectories only;
- exclude the current/same task and near duplicates from retrieval;
- do not store or retrieve locked-test trajectories during evaluation;
- reset or snapshot dynamic memory consistently between compared systems;
- report whether memory is frozen or allowed to evolve.

### End-to-end comparison

- use the same environment snapshot and evaluator;
- randomize or counterbalance system execution order when live-site drift is
  possible;
- log environment failures separately from agent failures;
- predefine timeout, step limit, and abort policy;
- do not manually rescue one system but not another.

---

## 13. Execution schedule

### Phase A - Data readiness, before full training

1. Complete two-person bbox and weak-class review.
2. Reconcile disagreements and complete secondary review.
3. Validate the non-destructive review overlay.
4. Collect genuine RETRY/ABORT transitions if all six strategies remain in
   scope.
5. Validate causal order, duplicate leakage, class counts, and provenance.
6. Freeze dataset version, split hashes, and review/supplement hashes.
7. Re-evaluate the v2.7 selected checkpoint on the reviewed validation overlay
   to establish the comparable pre-full-training baseline.

**Exit gate:** data/reconciliation/overlay reports PASS, test rows read zero.

### Phase B - Evaluation preparation, before or parallel with full training

1. Freeze RQ1-RQ6 and all metric formulas.
2. Select the end-to-end benchmark/task suite.
3. Implement the episode logger and success oracle.
4. Implement E0-E3 configuration switches.
5. Integrate train-only memory retrieval into the action decision.
6. Smoke-test the evaluator with the mini checkpoint.
7. Produce a machine-readable dry-run report.

**Exit gate:** one complete task can be replayed/executed and every required
field is logged without reading the locked test.

### Phase C - Reviewed 5k confirmation

1. Run one controlled 5k mini on the reviewed overlay.
2. Require finite loss, checkpoint round-trip, no test access, and registered
   validation gates.
3. Compare old selected checkpoint and new mini on the same reviewed
   validation overlay.

**Exit gate:** reviewed mini passes; no new architecture factor is changed
afterward without another smoke/mini test.

### Phase D - Full training

1. Run the full training split.
2. Save per-epoch CSV, environment, configuration, and checkpoints.
3. Use validation-only early stopping and registered selection.
4. First complete one seed; run additional primary-model seeds after the
   pipeline succeeds.
5. Preserve every run, including failed runs.

**Exit gate:** one checkpoint is selected and frozen using validation only.

### Phase E - Final component test

1. Load the frozen checkpoint.
2. Verify its SHA-256 and configuration.
3. Run the locked component test once.
4. Export metrics, per-class tables, confusion matrices, calibration plots,
   and prediction files.
5. Do not modify the model after opening the locked test.

### Phase F - End-to-end evaluation

1. Run E0-E3 on identical tasks.
2. Preserve complete episode logs and environment metadata.
3. Calculate the six primary end-to-end metrics.
4. Calculate memory retrieval and efficiency metrics.
5. Run paired statistical comparisons.
6. Perform a blinded manual audit of a registered sample of successes,
   recoveries, and failures.

### Phase G - Paper results

1. Fill the controlled component table.
2. Fill the controlled end-to-end table.
3. Fill the ablation table.
4. Fill the contextual related-work table.
5. Write error analysis by class, website, failure type, and recovery strategy.
6. State limitations, unsupported classes, environment failures, and compute
   limitations.

---

## 14. Required final result tables

### Table A - Component results

| Model | Outcome MCC | Failure macro-F1 | Failtype macro-F1 | Action macro-F1 | Strategy macro-F1 | Recovery MCC | Memory MCC | BBox mIoU | R@IoU50 | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority/random | | | | | | | | | | |
| Base model | | | | | | | | | | |
| Full model | | | | | | | | | | |

### Table B - End-to-end results

| System | Task success | Recovery success | Success after failure | Loop episode rate | Avg. steps | Avg. recovery attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 Base Qwen | | | | | | |
| E1 Trained action, no recovery | | | | | | |
| E2 Failure + recovery, no memory | | | | | | |
| E3 Full four-pillar | | | | | | |

### Table C - Pillar ablations

| Variant | Outcome MCC | Action macro-F1 | Recovery MCC | Task success | Delta vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full model | | | | | |
| Image occluded/image-only | | | | | |
| Text occluded/text-only | | | | | |
| No recovery | | | | | |
| No memory retrieval | | | | | |

### Table D - Efficiency

| System | Parameters | GPU hours | Decision latency | Retrieval latency | Task time | Avg. steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | | | | N/A | | |
| E2 | | | | N/A | | |
| E3 | | | | | | |

### Table E - Contextual related work

Use the numerical table in Section 7. Keep benchmark, backbone, and metric
visible. Never label it as a direct leaderboard unless protocols match.

---

## 15. Required figures

1. Four-pillar architecture and causal data flow.
2. Pre-action versus post-action routing diagram.
3. Validation learning curves across epochs.
4. Outcome/failure confusion matrices.
5. Action and recovery per-class F1 chart.
6. BBox IoU distribution and qualitative examples.
7. Reliability/calibration diagram.
8. End-to-end success and recovery comparison with confidence intervals.
9. Memory retrieval example showing query, retrieved experience, intervention,
   and result.
10. Failure taxonomy and representative unrecovered cases.

---

## 16. Q1 claim rules

Claims permitted only after evidence:

- “uses multimodal input” after verifying both inputs enter the model;
- “multimodal input improves performance” only after controlled modality
  comparison;
- “detects failures” after locked-test MCC/macro-F1;
- “recovers from failures” after recovery success and success-after-failure
  evaluation;
- “memory-driven” only after retrieval affects decisions;
- “memory improves performance” only after paired no-memory comparison;
- “outperforms a related method” only on the same benchmark/protocol or a
  faithful reproduction;
- “all six recovery strategies are learned” only with non-zero train,
  validation, and test support for all six.

Do not use “90% accuracy” as the global objective. Each task has a different
metric and difficulty. The paper should prioritize a clear, reproducible gain
over controlled baselines, class-balanced component performance, end-to-end
task success, and honest uncertainty.

---

## 17. Reproducibility artifact checklist

- immutable dataset/supplement/overlay manifests and hashes;
- split IDs and leakage report;
- manual-review and reconciliation reports;
- exact training configuration;
- code commit hash;
- package/environment lock;
- seed list;
- GPU information;
- per-epoch metrics CSV;
- all checkpoint metadata and hashes;
- frozen checkpoint-selection report;
- component predictions;
- end-to-end episode logs;
- evaluator version and success-oracle definition;
- analysis scripts;
- generated tables and figures;
- model card with limitations;
- explicit record of test access.

---

## 18. Primary references

1. OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning.  
   <https://arxiv.org/abs/2410.18963>
2. BacktrackAgent: Enhancing GUI Agent with Error Detection and Backtracking
   Mechanism.  
   <https://arxiv.org/abs/2505.20660>
3. ReUseIt: Synthesizing Reusable AI Agent Workflows for Web Automation.  
   <https://arxiv.org/abs/2510.14308>
4. SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills.  
   <https://arxiv.org/abs/2504.07079>
5. WebCoach: Self-Evolving Web Agents with Cross-Session Memory Guidance.  
   <https://arxiv.org/abs/2511.12997>

---

## 19. Immediate next action

The immediate next action is **not** the final full run. Complete and reconcile
the human review, validate the review overlay, and finish genuine RETRY/ABORT
collection if those strategies remain in scope. In parallel, implement and
smoke-test the end-to-end evaluator using the existing mini checkpoint. When
both the data gate and evaluator dry run pass, run one reviewed 5k confirmation
and then launch full training.
