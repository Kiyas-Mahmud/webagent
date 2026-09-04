# Q1 Component and End-to-End Implementation Plan

**Project:** four-pillar failure-aware multimodal web agent  
**Plan status:** implementation-ready, no code implemented by this document  
**Date:** 2026-07-28  
**PC-01 provisional pilot candidate:** Qwen2-VL-2B with QLoRA; the final
backbone remains unselected pending the registered validation-only PC-01/
PC-02/PC-03 comparison
**Historical controlled evidence when this document was drafted:** 5,000 train rows, 500 validation rows, zero test
rows read  

This is the repository-specific plan for turning the existing offline
multitask model into a measurable web-agent system and producing Q1-level
component, end-to-end, ablation, and related-work evidence.

This document is self-contained. For the broader metric justification and
published-paper values, also see
`docs/Q1_FINAL_EVALUATION_AND_COMPARISON_PLAN.md`.

---

## 1. Final outcome required

At the end of this plan, the repository must be able to:

1. train the reviewed full dataset without reading the locked test split;
2. select one checkpoint using validation only;
3. calculate final component metrics on the locked test split;
4. connect the frozen checkpoint to a controlled browser/task environment;
5. execute actions and observe post-action states;
6. detect failures after actions;
7. execute concrete recovery actions;
8. retrieve train-only recovery memories;
9. compare the same tasks under four controlled system configurations;
10. export reproducible tables, episode logs, confidence intervals, and
    failure analysis.

The final comparison must answer:

> Does the complete four-pillar system finish more tasks and recover from more
> failures than the same agent without recovery and without memory?

---

## 2. What already exists

### 2.1 Model

`src/web_agent/models/model.py` already implements:

- Qwen VLM encoding;
- a shared adapter;
- task-specific adapters;
- pre-action encoding;
- post-action encoding;
- a sparse causal recovery-transition stream;
- failure, action, memory, and recovery-outcome heads;
- spatial tokens for bbox grounding.

The current causal routing is scientifically appropriate:

| Stream | Available information | Current consumers |
| --- | --- | --- |
| Pre-action | state before execution + task/domain | action, bbox, confidence |
| Post-action | state before + state after + executed action/value | outcome, failure type, needs-recovery, recovery strategy, memory update |
| Recovery transition | failure state + executed recovery + post-recovery state | recovery outcome |

### 2.2 Dataset

`src/web_agent/data/gold_dataset.py` already:

- loads state-before and state-after screenshots;
- builds separate pre/post prompts;
- keeps executed action out of the pre-action stream;
- includes executed action in the post-action stream;
- constructs causal recovery transitions;
- masks invalid bbox supervision;
- exports the labels required by the existing heads.

### 2.3 Component evaluation

`src/web_agent/eval/metrics.py` already supports:

- failure F1 and macro-F1;
- balanced accuracy;
- MCC;
- majority baselines;
- Brier score;
- ECE;
- bbox MAE;
- bbox IoU and Recall@IoU50;
- per-class diagnostics and confusion matrices.

`src/web_agent/train/trainer.py` already collects validation predictions and
calculates the current component metrics.

### 2.4 Checkpoint selection

`src/web_agent/train/selection.py` already implements:

```text
all quality gates define eligibility
    -> highest outcome MCC among eligible epochs
    -> earlier epoch wins exact ties
```

This selection rule must remain validation-only.

### 2.5 Review overlay

The repository already provides:

- human-review sessions;
- two-person reconciliation;
- secondary review for risky decisions;
- a non-destructive train/validation overlay;
- validation that prevents the overlay from being applied during locked-test
  evaluation.

### 2.6 Initial memory index

`src/web_agent/memory/index.py` already provides:

- L2 normalization;
- cosine-similarity retrieval;
- top-K metadata results.

It is not integrated into the model's runtime decision path and does not yet
enforce task/episode exclusions.

---

## 3. What does not yet exist

The current repository is an offline supervised training and evaluation
pipeline. It is not yet a complete browser-executing agent.

Missing parts:

1. no browser/environment controller;
2. no episode-level task runner;
3. no action executor;
4. no official/deterministic task-success oracle;
5. no concrete recovery executor;
6. no runtime Decision Combiner;
7. no runtime integration of `MemoryIndex`;
8. no end-to-end episode logger;
9. no task-success/recovery/loop/step metric calculator;
10. no E0-E3 system comparison runner.

### 3.1 Action-parameter gap

The current action head predicts:

```text
action_type
bbox
```

It does not predict:

- text to enter for TYPE;
- option value for SELECT;
- URL/query for NAVIGATE;
- key value for PRESS_KEY;
- direction/distance for SCROLL.

End-to-end execution therefore requires a separate action-parameter provider.
It must be identical across E0-E3, or the comparison will confound recovery
with parameter generation.

### 3.2 High-level recovery gap

The recovery head predicts a strategy:

```text
NONE
RETRY
REPLAN
BACKTRACK
ALTERNATIVE_TARGET
ABORT
```

It does not produce the concrete browser action sequence required by REPLAN or
ALTERNATIVE_TARGET. A fixed recovery action resolver is required.

---

## 4. Frozen system configurations

The end-to-end evaluation will compare four systems.

| ID | Model/policy | Failure diagnosis | Recovery | Memory |
| --- | --- | --- | --- | --- |
| E0 | Base Qwen web agent | base-agent behaviour only | base-agent behaviour only | disabled |
| E1 | Trained pre-action policy | disabled | disabled | disabled |
| E2 | Same trained model | enabled | enabled | disabled |
| E3 | Same trained model | enabled | enabled | enabled |

### 4.1 Fairness rules

E1, E2, and E3 must share:

- the same selected checkpoint;
- the same backbone;
- the same observation builder;
- the same action-parameter provider;
- the same browser executor;
- the same success oracle;
- the same task set;
- the same time, step, and recovery budgets;
- the same seed/repetition schedule.

Only the registered feature switches may differ.

### 4.2 What each comparison proves

| Comparison | Permitted interpretation |
| --- | --- |
| E0 vs E1 | effect of project-specific supervised policy training |
| E1 vs E2 | effect of failure diagnosis and recovery controller |
| E2 vs E3 | effect of corrective memory retrieval and use |
| E0 vs E3 | total system difference, not isolated causal contribution |

---

## 5. Proposed repository structure

The implementation should use new runtime modules and preserve the existing
training code.

```text
src/web_agent/runtime/
    __init__.py
    contracts.py
    observation.py
    policy.py
    action_parameters.py
    executor.py
    verifier.py
    recovery.py
    memory_runtime.py
    decision.py
    episode.py
    logging.py

src/web_agent/eval/
    end_to_end.py
    episode_metrics.py
    statistics.py

scripts/
    run_end_to_end.py
    summarize_end_to_end.py

configs/evaluation/
    q1_smoke.yaml
    q1_primary.yaml

tests/runtime/
    test_observation.py
    test_policy.py
    test_action_parameters.py
    test_executor.py
    test_verifier.py
    test_recovery.py
    test_memory_runtime.py
    test_decision.py
    test_episode.py
    test_logging.py
    test_episode_metrics.py
```

No implementation should be placed inside the main training notebook.
Notebooks may call the tested scripts, but runtime logic belongs in modules.

---

## 6. Runtime contracts

The first implementation task is to define typed, serializable contracts.

### 6.1 Task specification

Required fields:

```text
task_id
instruction
start_url or environment reset ID
website/domain
success_oracle ID
maximum_steps
maximum_recovery_attempts
timeout_seconds
required capabilities
```

Optional fields:

```text
expected final URL pattern
expected visible text
expected structured state
destructive_action_allowed
credentials profile
```

### 6.2 Observation

Required fields:

```text
episode_id
step_index
timestamp
url
domain
screenshot path/hash
viewport width/height
page text or accessibility-state summary
environment status
```

The observation must not include:

- ground-truth next action;
- ground-truth outcome;
- future state;
- test label;
- evaluator-only success information.

### 6.3 Policy decision

Required fields:

```text
action_type
normalized bbox
action confidence
predicted outcome, when post-action
failure type, when post-action
needs recovery
recovery strategy
memory-update decision
model checkpoint hash
configuration ID
```

### 6.4 Concrete browser action

Required fields:

```text
action_type
target coordinates or element reference
value
source:
    model
    parameter provider
    recovery resolver
attempt ID
```

### 6.5 Verification result

Required fields:

```text
environment action executed: true/false
page changed: true/false
task progress: true/false/unknown
task success: true/false
verified failure: true/false
environment failure: true/false
failure reason
oracle version
```

### 6.6 Recovery episode

A recovery episode begins when recovery is triggered and ends at the first of:

- verified progress;
- verified task success;
- another verified failure;
- ABORT;
- recovery-budget exhaustion;
- environment failure.

Required fields:

```text
recovery_episode_id
triggering failure
predicted strategy
concrete recovery actions
memory items used
start state hash
end state hash
verified recovery success
number of browser actions
latency
```

---

## 7. Observation builder

### 7.1 Purpose

Convert browser state into the same semantic format used by
`GoldDataset._vlm_inputs`.

### 7.2 Pre-action observation

Input:

```text
current screenshot
task instruction
domain
```

Prompt role:

```text
Choose the next action from the page before execution.
```

The pre-action observation feeds:

- action type;
- bbox;
- confidence-before.

### 7.3 Post-action observation

Input:

```text
state before
state after
task instruction
domain
executed action type
executed action value
```

Prompt role:

```text
Assess the result by comparing the page before and after execution.
```

The post-action observation feeds:

- outcome;
- failure type;
- needs-recovery;
- recovery strategy;
- memory-update prediction.

### 7.4 Required consistency tests

Tests must verify:

- training and runtime use the same image processor;
- runtime uses the same prompt wording/template;
- image ordering is before then after;
- executed action is absent from pre-action input;
- future/test labels cannot enter either prompt;
- bbox coordinates use the same normalized `[x, y, width, height]` convention;
- viewport-to-pixel conversion is reversible within rounding tolerance.

---

## 8. Policy adapter

The policy adapter loads one frozen checkpoint and exposes two operations.

### 8.1 `predict_action`

Input:

```text
TaskSpecification
current Observation
```

Output:

```text
action type
bbox
pre-action confidence
pre-action embedding
```

It must not run the post-action stream before an action is executed.

### 8.2 `assess_transition`

Input:

```text
TaskSpecification
pre-action Observation
executed ConcreteAction
post-action Observation
```

Output:

```text
predicted outcome
failure type
needs-recovery
recovery strategy
memory-update prediction
post-action embedding
```

### 8.3 Checkpoint controls

Every output report must record:

- checkpoint path;
- SHA-256;
- config hash;
- selected epoch;
- selection report hash;
- dataset/overlay/supplement hashes.

The runtime must refuse a checkpoint that does not match the frozen selection
report.

---

## 9. Action-parameter provider

### 9.1 Purpose

Complete action parameters that the current trained head does not predict.

### 9.2 Fair implementation

Use one fixed provider across E0-E3. It may use:

- task instruction;
- current page state;
- predicted action type;
- predicted target region;
- a fixed planner/LLM prompt.

It must not use:

- ground-truth action value;
- future state;
- test label;
- hidden success-oracle answer.

### 9.3 Action contracts

| Action | Required parameter |
| --- | --- |
| CLICK | target point/bbox |
| TYPE | target + text |
| SELECT | target + option |
| SCROLL | direction + distance |
| NAVIGATE | URL or allowed destination |
| PRESS_KEY | registered key |

### 9.4 Logging

Record separately:

- model-selected action type;
- model bbox;
- provider-generated parameter;
- concrete executed action;
- provider latency;
- parameter-generation failure.

This prevents action-parameter quality from being incorrectly attributed to
the four-pillar model.

---

## 10. Browser/environment controller

### 10.1 Required interface

```text
reset(task)
observe()
execute(concrete action)
evaluate_task()
close()
```

### 10.2 Environment rules

- Use an isolated browser profile per episode.
- Reset cookies/storage according to benchmark rules.
- Fix viewport size and browser version.
- Save environment metadata.
- Separate agent failures from environment failures.
- Do not manually intervene during scored runs.
- Use the same environment snapshot for E0-E3 where possible.

### 10.3 Executor behaviour

The executor must:

- reject out-of-range coordinates;
- reject unsupported actions;
- record Playwright/browser exceptions;
- capture state before and after every action;
- enforce action timeout;
- wait using one registered page-settle rule;
- never silently choose a different element when the model misses.

### 10.4 BBox conversion

For normalized `[x, y, width, height]`:

```text
pixel_x = normalized_x * viewport_width
pixel_y = normalized_y * viewport_height
pixel_w = normalized_width * viewport_width
pixel_h = normalized_height * viewport_height
click_x = pixel_x + pixel_w / 2
click_y = pixel_y + pixel_h / 2
```

Record both normalized and pixel values.

---

## 11. Independent verifier and success oracle

The model cannot be the source of its own ground truth.

### 11.1 Task-success oracle

Preferred order:

1. official benchmark evaluator;
2. deterministic structured-state evaluator;
3. deterministic URL/text/DOM condition;
4. blinded two-person human adjudication for unresolved cases.

### 11.2 Progress oracle

Recovery success requires observable progress. A registered progress oracle
may use:

- required subgoal completion;
- transition to a valid next workflow state;
- removal of an error/blocking state;
- successful return to a recoverable prior state;
- final task success.

Page change alone is insufficient. Dynamic timestamps or animations can change
without task progress.

### 11.3 Environment failure

Examples:

- website unavailable;
- benchmark service failure;
- browser crash;
- expired credentials;
- CAPTCHA outside the registered agent scope.

Environment failures must be reported separately and excluded/included only
according to a frozen rule applied equally to all systems.

---

## 12. Decision Combiner

**Superseded clarification:** the canonical Table 2 protocol is now
`docs/TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md`. Earlier revisions
called the two workflow headings below “12.1” and “12.2.” Those labels were
document subsection numbers, never paper-table numbers. They are renamed here
to Workflow A and Workflow B to prevent confusion. Neither workflow replaces
the single headline paper **Table 2** or its unnumbered companion diagnostics.

The Decision Combiner operates after the relevant runtime-visible information
exists. Sealed verifier/oracle labels never enter its decision path.

### Workflow A — Normal step

```text
observe current page
    -> pre-action prediction
    -> parameter provider
    -> execute action
    -> observe post-action page
    -> post-action diagnosis
```

Pillar 1 cannot use `state_after` to cancel an action that has already
executed. It can only determine the next recovery decision.

### Workflow B — Recovery trigger

Recovery is triggered using a frozen rule based on:

- predicted outcome;
- needs-recovery prediction;
- confidence threshold selected on validation;
- observed executor rejection/error or agent-visible browser result;
- loop guard.

Independent verifier evidence is attached only after the decision is complete
and is used for scoring, failure confirmation, and false-trigger/missed-failure
analysis. It cannot trigger, suppress, select, or modify recovery.

The paper must distinguish:

- model-triggered recovery;
- oracle-confirmed failure;
- false-positive recovery;
- missed failure.

### 12.3 Pre-action caution

Low pre-action confidence may:

- request parameter-provider reconsideration;
- choose a safe registered fallback.

It must not access post-action information or query P4 memory. In the canonical
Table 2 protocol, memory retrieval is post-failure only.

### 12.4 Hard stop

The final policy must freeze:

- maximum task actions;
- maximum recoveries per failure;
- maximum total recoveries;
- timeout;
- ABORT conditions.

---

## 13. Concrete recovery resolver

The strategy label must map to executable behaviour.

| Strategy | Concrete behaviour |
| --- | --- |
| NONE | continue normal policy |
| RETRY | repeat the failed action once after registered state refresh/check |
| REPLAN | ask the fixed planner for a new action plan from the current state |
| BACKTRACK | execute browser/environment back action, then verify returned state |
| ALTERNATIVE_TARGET | request another candidate target for the same subgoal |
| ABORT | terminate safely and emit structured failure |

### 13.1 Strategy-specific validity

- RETRY is not valid if it repeats an action after the target disappeared.
- BACKTRACK succeeds only if it returns to a valid recoverable state.
- REPLAN succeeds only after measurable progress, not after producing text.
- ALTERNATIVE_TARGET must select a genuinely different candidate.
- ABORT is operationally successful when termination is safe, but it is not
  task success.

### 13.2 Recovery-attempt counting

One high-level recovery decision counts as one recovery attempt. Also record
the number of low-level browser actions inside that attempt.

---

## 14. Runtime memory integration

### 14.1 Build set

Build the memory index only from training trajectories satisfying:

```text
completed trajectory
memory_update_flag = true
recovery attempted
recovery_success = true
valid provenance
not rejected/quarantined
```

### 14.2 Stored metadata

Each item should contain:

```text
memory_id
embedding
training episode/task ID
domain/site
failure type
executed recovery strategy
recovery action/value
recovery outcome
short reflection/context
source data version
```

### 14.3 Query

Use the current post-failure embedding as the query.

Required filters:

- exclude same episode;
- exclude same task ID when required by protocol;
- exclude exact/near duplicates;
- exclude all validation/test-derived memories;
- optionally constrain by compatible domain/action schema.

### 14.4 Retrieval outputs

Return:

```text
top-K memory IDs
similarity scores
strategies
failure types
recovery evidence
filter counts
```

### 14.5 Memory intervention

The Decision Combiner must record whether it:

- ignored memory;
- used the top memory;
- used majority strategy among top-K;
- passed retrieved context to the planner;
- changed the original recovery decision.

### 14.6 Frozen versus dynamic memory

Primary evaluation should use frozen train-only memory for repeatability.

Dynamic memory can be a secondary experiment. If enabled:

- task order must be fixed or counterbalanced;
- test task IDs must never retrieve themselves;
- all systems must start from the same initial memory snapshot;
- results must be labelled dynamic and order-dependent.

---

## 15. Episode runner

### 15.1 Episode algorithm

```text
reset task
capture initial observation

while not terminal:
    enforce task/time/action budgets
    predict pre-action action and bbox
    resolve missing action parameters
    execute concrete action
    capture post-action observation
    independently evaluate task success/progress
    run post-action diagnosis

    if task success:
        finish SUCCESS

    if recovery trigger:
        retrieve memory when E3
        select concrete recovery
        execute recovery episode
        verify recovery outcome

    detect repeated-error loops

finish with:
    SUCCESS
    ABORT
    STEP_LIMIT
    TIMEOUT
    ENVIRONMENT_FAILURE
```

### 15.2 No hidden rescue

During scored evaluation:

- no manual clicks;
- no manual text correction;
- no hidden retries outside logged recovery attempts;
- no task-specific prompt edits after seeing failures;
- no different provider or oracle between systems.

---

## 16. Episode logging

### 16.1 Action-level CSV/JSONL

One record per executed browser action:

```text
run_id
system_id
task_id
episode_id
step_index
timestamp
pre_state_hash
pre_screenshot
url_before
predicted_action_type
predicted_bbox_normalized
predicted_bbox_pixels
pre_action_confidence
parameter_provider_output
executed_action
execution_status
post_state_hash
post_screenshot
url_after
predicted_outcome
predicted_failure_type
predicted_needs_recovery
predicted_recovery_strategy
oracle_progress
oracle_task_success
verified_failure
environment_failure
memory_query_id
recovery_episode_id
model_latency_ms
provider_latency_ms
execution_latency_ms
```

### 16.2 Recovery-level CSV/JSONL

One record per high-level recovery attempt:

```text
recovery_episode_id
trigger_step
triggering_failure
strategy
memory_enabled
retrieved_memory_ids
memory_scores
memory_changed_decision
concrete_recovery_actions
recovery_success
actions_used
latency_ms
termination_reason
```

### 16.3 Episode-summary CSV

One record per task attempt:

```text
run_id
system_id
task_id
seed/repetition
task_success
contained_verified_failure
success_after_failure
recovery_attempts
successful_recoveries
loop_detected
repeated_error_events
total_browser_actions
normal_actions
recovery_actions
wall_time
environment_failure
terminal_reason
```

### 16.4 Manifest

Every run directory must contain:

```text
manifest.json
config.yaml
environment.json
checkpoint.json
tasks.json
actions.jsonl
recoveries.jsonl
episodes.csv
metrics.json
metrics.csv
statistics.json
```

---

## 17. Exact end-to-end metric definitions

### 17.1 Task success rate

```text
successful tasks / all valid attempted tasks
```

Report numerator, denominator, percentage, and 95% confidence interval.

### 17.2 Recovery success rate

```text
verified successful recovery episodes / all executed recovery episodes
```

Also report per-strategy and per-failure-type rates.

### 17.3 Success after an initial failure

```text
successful tasks containing at least one earlier verified failure
/
tasks containing at least one verified failure
```

This is the primary operational evidence that the agent recovers.

### 17.4 Loop episode rate

```text
episodes containing a confirmed repeated-error loop
/
all valid attempted episodes
```

### 17.5 Repeated-error event rate

```text
confirmed repeated ineffective state-action events
/
all executed browser actions
```

### 17.6 Average task steps

```text
all executed browser actions / all valid attempted tasks
```

Also report median, IQR, successful-task steps, and failed-task steps.

### 17.7 Average recovery attempts

```text
all high-level recovery attempts
/
tasks containing at least one recovery
```

Also report median, maximum, and recovery-budget exhaustion.

### 17.8 Loop-detection rule

Freeze an observation signature combining:

- canonical URL;
- page/accessibility-state fingerprint;
- screenshot perceptual hash;
- action type and target.

Flag a loop when either:

- the same ineffective state-action pair repeats within the registered rolling
  window; or
- an A-B-A-B state cycle occurs without progress; or
- the official environment/evaluator confirms a loop.

The rolling-window size and similarity thresholds must be tuned on development
episodes only.

---

## 18. Exact component evaluation

After validation-only checkpoint selection, the locked component test must
produce:

### Outcome/failure

- outcome MCC;
- outcome macro-F1;
- balanced accuracy;
- success recall;
- failure recall;
- failure-type MCC and macro-F1;
- per-class precision/recall/F1/support;
- confusion matrices.

### Action

- action macro-F1;
- action MCC;
- balanced accuracy;
- per-action precision/recall/F1/support;
- predicted class count/distribution.

### Recovery

- needs-recovery MCC and macro-F1;
- attempted-only recovery-strategy macro-F1 and MCC;
- per-strategy metrics;
- recovery-outcome MCC, macro-F1, balanced accuracy, and Brier score.

### Memory

- memory-update MCC, macro-F1, and balanced accuracy;
- retrieval Recall@1/3/5;
- MRR;
- strategy agreement@K;
- retrieval coverage;
- irrelevant-memory rate.

### Grounding

- mean/median IoU;
- Recall@IoU25;
- Recall@IoU50;
- bbox MAE as secondary;
- results by action type and target-size bucket;
- number of valid and masked rows.

### Calibration

- outcome ECE and Brier score;
- recovery-outcome ECE and Brier score;
- reliability diagrams;
- threshold/coverage-risk curve.

MCC values must remain decimals in `[-1, 1]`; do not present MCC as task-success
percentages.

---

## 19. Data completion before full training

### 19.1 Manual review

Reviewers only need to decide:

1. does the state support the action label?
2. did the labelled recovery actually occur?
3. does the post-action state support success/failure?
4. is the bbox on the intended element?
5. if uncertain, should the row be discussed?

They must not guess or silently rewrite records.

### 19.2 Reconciliation

Required sequence:

```text
Reviewer A + Reviewer B
    -> agreement calculation
    -> risky/disputed secondary queue
    -> secondary decisions
    -> reconciliation PASS
    -> approved non-destructive overlay
    -> overlay validation PASS
```

### 19.3 RETRY/ABORT

If all six strategies remain claimed, collect genuine complete transitions.

Minimum target:

- RETRY: 300 train + 100 validation;
- ABORT: 300 train + 100 validation.

Preferred target:

- RETRY: 400 train + 150 validation;
- ABORT: 400 train + 150 validation.

The final locked test also needs non-zero independent support. Test examples
must be collected/split before training and remain locked; they must not be
copied from train/validation.

### 19.4 Final data gate

Required PASS artifacts:

- reconciliation report;
- overlay validation report;
- supplementary-data audit, if used;
- duplicate/near-duplicate report;
- split leakage report;
- causal-transition report;
- bbox audit;
- file hashes;
- zero test access record.

---

## 20. Benchmark protocol

### 20.1 Preferred primary benchmark

Use WebArena when compatible with the executor because it provides a controlled
environment and official evaluation and is related to SkillWeaver.

### 20.2 Secondary external benchmark

Use WebVoyager when resources permit because it relates directly to WebCoach.
Record live-site drift, CAPTCHAs, and environment failures.

### 20.3 Minimum controlled fallback

If public integration cannot be completed:

- at least 100 held-out task episodes;
- at least five websites/domains where possible;
- action-class diversity;
- a registered failure-prone subset;
- identical tasks for E0-E3;
- deterministic or independently adjudicated success;
- complete logs.

This fallback must be labelled a project benchmark, not a direct SOTA
leaderboard.

### 20.4 Direct related-paper comparison rule

| Paper | Direct comparison requires |
| --- | --- |
| OSCAR | same OSWorld/AndroidWorld protocol and comparable base model |
| BacktrackAgent | same Mobile3M/Auto-UI split and metrics |
| ReUseIt | same repetitive-task protocol or faithful reproduction |
| SkillWeaver | WebArena or its documented live-site protocol |
| WebCoach | full WebVoyager protocol and visible backbone/memory differences |

Otherwise, published numbers are contextual only.

---

## 21. Smoke, mini, full, and final-test sequence

### Stage 1 - Runtime unit tests

No GPU/browser scoring yet.

Pass requirements:

- contracts serialize/deserialize;
- bbox conversion is correct;
- feature switches isolate E1/E2/E3;
- memory exclusions work;
- episode metric formulas pass hand-calculated fixtures;
- logs contain required fields.

### Stage 2 - Browser integration smoke

Use deterministic local/simple tasks.

Pass requirements:

- reset, observe, execute, verify, close;
- screenshots and state hashes saved;
- all six action contracts either execute or reject clearly;
- timeouts/environment errors are classified correctly.

### Stage 3 - Mini-checkpoint end-to-end smoke

Use the existing mini checkpoint on 5-10 development tasks.

Pass requirements:

- one normal episode completes;
- one failure triggers recovery;
- E2 runs without memory;
- E3 retrieves train-only memory;
- no test data is read;
- final manifests and metrics are generated.

Performance does not need to be high.

### Stage 4 - Reviewed 5k confirmation

Run one 5k/500 validation experiment on the final reviewed overlay and accepted
supplement.

Pass requirements:

- finite loss;
- checkpoint round-trip;
- registered quality gates;
- validation-only selection;
- zero test reads;
- short end-to-end dry run works.

### Stage 5 - Full training

Run one full seed first.

Required:

- immutable config;
- per-epoch metrics;
- checkpoint every epoch/resume support;
- validation-only selection;
- early stopping;
- checkpoint/config/data hashes.

For the current research-locked Table 2 protocol, retain seed 42 only. Do not
retrain PC-01 or add seeds 43--44; PC-02 and PC-03 contribute only their
completed seed-42 validation packages to final model promotion.

### Stage 6 - Frozen component test

After selecting one checkpoint:

- freeze code/config/thresholds;
- open locked component test once;
- export all predictions and metrics;
- do not retrain or retune afterward.

### Stage 7 - End-to-end E0-E3

Run identical task IDs and repetitions for every system. Randomize or
counterbalance order when live-site drift is possible.

### Stage 8 - Ablations

Required:

- E1 vs E2;
- E2 vs E3;
- full input vs image-occluded vs text-occluded sensitivity.

Preferred when compute allows:

- separately trained image-only;
- separately trained text-only;
- separately trained no-recovery variant.

Inference occlusion must be called sensitivity analysis, not a full training
ablation.

---

## 22. Statistical analysis

### 22.1 Training variation

- use the registered seed-42 checkpoint only for Table 2;
- report explicitly that model-seed uncertainty is not measured;
- preserve every completed candidate result, including unfavourable results.

### 22.2 End-to-end uncertainty

For each E0-E3 metric, report:

- numerator;
- denominator;
- percentage/mean;
- 95% confidence interval.

### 22.3 Paired comparisons

Because the same tasks are used:

- McNemar test or paired bootstrap for task success;
- paired bootstrap for success-after-failure;
- paired bootstrap/non-parametric test for step counts;
- absolute and relative improvement;
- Holm correction for multiple ablation comparisons.

### 22.4 Practical significance

The paper must discuss:

- task-success gain;
- recovery gain;
- latency/cost increase;
- additional steps;
- environment failures;
- model size.

A statistically significant but operationally tiny result is not sufficient by
itself.

---

## 23. Final tables

### 23.1 Component table

| System | Outcome MCC | Failure macro-F1 | Action macro-F1 | Strategy macro-F1 | Recovery MCC | Memory MCC | BBox mIoU | R@IoU50 | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority/random | | | | | | | | | |
| E1 | | | | N/A | N/A | N/A | | | |
| E2 | | | | | | | | | |
| E3 | | | | | | | | | |

### 23.2 End-to-end table

| System | Task success | Recovery success | Success after failure | Loop rate | Avg. steps | Avg. recoveries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | | | | | | |
| E1 | | N/A | N/A | | | N/A |
| E2 | | | | | | |
| E3 | | | | | | |

### 23.3 Memory table

| System | R@1 | R@3 | R@5 | MRR | Intervention coverage | Useful intervention rate | TSR delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E2, memory off | N/A | N/A | N/A | N/A | N/A | N/A | reference |
| E3, memory on | | | | | | | |

### 23.4 Efficiency table

| System | Parameters | Decision latency | Provider latency | Retrieval latency | Task time | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | | | | N/A | | N/A |
| E1 | | | | N/A | | |
| E2 | | | | N/A | | |
| E3 | | | | | | |

---

## 24. Implementation task ledger

### Workstream A - Data

| ID | Task | Depends on | Output | Gate |
| --- | --- | --- | --- | --- |
| D1 | Complete primary human review | accepted audit package | four reviewer logs | assignments complete |
| D2 | Reconcile and create secondary queues | D1 | reconciliation report/queues | no unresolved required rows |
| D3 | Complete secondary review | D2 | updated logs | risky rows independently reviewed |
| D4 | Final reconciliation | D3 | approved corrections/dispositions | PASS |
| D5 | Validate overlay | D4 | overlay validation report | PASS |
| D6 | Collect RETRY/ABORT if in scope | frozen schema | versioned supplement | targets met |
| D7 | Audit supplement and splits | D6 | integrity/leakage/causal reports | PASS |
| D8 | Freeze final dataset manifest | D5, D7 | hashes and version ID | immutable |

### Workstream B - Runtime foundation

| ID | Task | Depends on | Output | Gate |
| --- | --- | --- | --- | --- |
| R1 | Define runtime contracts | none | `runtime/contracts.py` | serialization tests |
| R2 | Implement observation builder | R1 | `runtime/observation.py` | training/runtime parity tests |
| R3 | Implement policy adapter | R1, R2 | `runtime/policy.py` | mini checkpoint inference |
| R4 | Implement parameter provider | R1 | `runtime/action_parameters.py` | all action contracts covered |
| R5 | Implement browser executor | R1, R4 | `runtime/executor.py` | deterministic browser tests |
| R6 | Implement verifier/oracle adapter | R1, R5 | `runtime/verifier.py` | known success/failure fixtures |
| R7 | Implement recovery resolver | R4-R6 | `runtime/recovery.py` | each strategy tested |
| R8 | Integrate memory | R3, R7 | `runtime/memory_runtime.py` | train-only exclusion tests |
| R9 | Implement Decision Combiner | R3, R6-R8 | `runtime/decision.py` | E1/E2/E3 switch tests |
| R10 | Implement episode runner | R5-R9 | `runtime/episode.py` | complete dry-run episode |
| R11 | Implement logger | R1, R10 | `runtime/logging.py` | schema-complete manifests |

### Workstream C - Evaluation

| ID | Task | Depends on | Output | Gate |
| --- | --- | --- | --- | --- |
| V1 | Implement episode metrics | R11 | `eval/episode_metrics.py` | hand-calculated tests |
| V2 | Implement E0-E3 runner | R10, R11 | `eval/end_to_end.py` | paired task execution |
| V3 | Implement statistics | V1 | `eval/statistics.py` | bootstrap/paired-test fixtures |
| V4 | Implement CLI scripts | V1-V3 | scripts/configs | reproducible dry run |
| V5 | Freeze benchmark/task manifest | V2 | task IDs + oracle versions | no later edits |
| V6 | Mini-checkpoint smoke | V4, V5 | smoke run package | PASS |

### Workstream D - Training and final evidence

| ID | Task | Depends on | Output | Gate |
| --- | --- | --- | --- | --- |
| T1 | Re-evaluate old checkpoint on reviewed overlay | D8 | comparable validation baseline | zero test reads |
| T2 | Reviewed 5k confirmation | D8, V6 | mini report/checkpoint | all gates PASS |
| T3 | Full seed 1 | T2 | full report/checkpoints | completed |
| T4 | Select and freeze checkpoint | T3 | selection report/hash | validation only |
| T5 | Additional seeds | T3 | seed reports | resources permitting |
| T6 | Locked component test | T4 | component result package | one frozen run |
| T7 | E0-E3 evaluation | T4, V5 | episode/result packages | same tasks/protocol |
| T8 | Sensitivity/ablations | T7 | ablation result package | registered variants |
| T9 | Statistical analysis | T6-T8 | tables/CIs/tests | reproducible |
| T10 | Error analysis | T7-T9 | qualitative audit | blinded sample |

---

## 25. Responsibility split

### Codex/code agent

- implement runtime/evaluation modules after explicit implementation approval;
- write automated tests;
- preserve existing notebooks and training pipeline;
- validate schemas, metrics, hashes, and leakage controls;
- generate reproducible result packages;
- analyze failures without inventing labels.

### Human reviewers

- inspect label/state consistency;
- inspect bbox target correctness;
- flag uncertainty;
- independently adjudicate risky rows;
- never modify source JSON directly.

### Data-collection agent

- collect genuine RETRY/ABORT trajectories;
- preserve complete causal transitions;
- attach provenance;
- run duplication/leakage/schema checks;
- never synthesize outcomes or use locked test data.

### Thesis team

- approve benchmark and task scope;
- approve success/progress oracle;
- provide/approve any credentials and destructive-action policy;
- freeze final claims before locked test;
- review qualitative examples and paper wording.

---

## 26. Suggested schedule

### Before full training this week

1. Freeze benchmark, metrics, budgets, and E0-E3 definitions.
2. Complete data review/reconciliation and RETRY/ABORT decision.
3. Implement contracts, observation parity, parameter provider, and executor.
4. Implement verifier, recovery resolver, memory integration, and Decision
   Combiner.
5. Implement episode logging and metrics.
6. Smoke-test with the existing mini checkpoint.
7. Run the reviewed 5k confirmation.
8. Launch the first full seed only after both data and runtime gates pass.

### During full training

1. Monitor registered validation metrics.
2. Preserve checkpoints and environment artifacts.
3. Draft methodology from frozen architecture/contracts.
4. Prepare task manifests and result-table templates.
5. Do not open locked test.

### After full training

1. Select and hash the validation-eligible checkpoint.
2. Run locked component test.
3. Run E0-E3 end-to-end evaluation.
4. Run Pillar 2 sensitivity and Pillar 4 memory comparison.
5. Calculate confidence intervals and paired tests.
6. Perform error analysis.
7. Fill final tables and write results/discussion.

---

## 27. Stop conditions

Stop full-training launch if:

- reconciliation or overlay validation is not PASS;
- RETRY/ABORT are claimed but required data is not validated;
- split leakage is detected;
- reviewed mini fails registered gates;
- evaluator cannot finish and log a dry-run episode;
- action parameters differ across E0-E3;
- memory can retrieve validation/test or same-task leakage;
- success oracle is undefined or model-dependent.

Stop final evaluation if:

- checkpoint/config hashes do not match the frozen selection report;
- task manifests differ between systems;
- environment reset is inconsistent;
- hidden/manual rescue occurs;
- the locked test has already been used for tuning.

---

## 28. Final readiness definition

The project is ready to write the final results section only when all of the
following exist:

- reviewed, versioned dataset and supplement;
- frozen splits and hashes;
- one validation-selected frozen checkpoint;
- locked component-test package;
- complete E0-E3 episode packages;
- component and end-to-end metric tables;
- memory retrieval evaluation;
- confidence intervals and paired comparisons;
- efficiency table;
- per-class and per-strategy error analysis;
- documented environment failures;
- reproducible code/config/commit/checkpoint identifiers;
- honest statement of unsupported or negative results.

The final success criterion is not a universal 90% number. The required
evidence is a reproducible, statistically supported improvement of the complete
agent over controlled same-protocol baselines, combined with strong
class-balanced component diagnostics and transparent limitations.
