# Next evaluation plan: model comparison first, agent integration second

> **Current status, 2026-09-21:** the [Task 1 InternVL mini comparison](TASK1_INTERNVL_DUAL_V2_RESULTS.md) is complete and pushed. [Task 2 agent integration](TASK2_AGENT_INTEGRATION.md) is implementation/development evidence only: engineering checks and the 12-episode development audit passed (A/B/C each 4/4), while the planned 90-episode final evaluation was not completed. The user approved the original PC-01 encoder for memory queries only. InternVL remains actor and assessor. The public-benchmark extension has not run; older pending statuses and PC-01-only choices are historical.

Date: 2026-09-14

**Implementation update, 2026-09-15:** use the
[Task 1 implementation and runbook](TASK1_IMPLEMENTATION_AND_RUNBOOK.md).
The user selected PC-03 full epoch 0 for this comparison and shared frozen Qwen
for three adapted external components. The original PC-01-only constraint below
remains historical for older experiments. Local harness tests pass; actual
external development has started on the local lab PC; InternVL parity and
comparative inference remain pending.

**Latest Task 1 design:** [agent-component architecture and data examples](TASK1_AGENT_COMPARISON_ARCHITECTURE.md).
Browser Use, Agent S2 and WebVoyager assessment components are the three selected
external candidates. This supersedes the earlier proposed comparator roster
below. Exact backend/checkpoint identities and adapter compatibility remain
pending; none of these candidates has been evaluated on the mini subset.

**Current bounded scope:** start with the
[240-example mini validation pilot](TASK1_MINI_DATASET_PILOT.md) on our dataset.
Its metadata selection is complete; comparator selection and inference remain
pending. Complete this pilot before progressing to the public benchmark.

**Status: planning only. Task 1 is the immediate priority. No new evaluation,
model download, checkpoint change or live-agent integration is launched by this
document.**

## 1. Purpose and execution order

The current paper will evaluate the useful capabilities of the existing frozen
model. We will first compare its predictions with other models on our data and
on a compatible independent public benchmark. We will subsequently investigate
whether adding its diagnosis/recovery module, followed by memory, improves an
existing executing agent.

1. **Task 1 — Two-dataset comparison using the same models.** Establish matched
   offline performance and external generalisation.
2. **Task 2 — Existing agent plus our modules.** Measure the practical effect of
   diagnosis/recovery and the additional effect of memory through live execution.

Task 1 must be completed and reported before Task 2 implementation begins. Its
completion requires valid execution and reporting, not positive results. The
findings determine which capabilities can reasonably be integrated in Task 2.

This plan defines the next work sequence. Older E0–E3 and H0–H3 experiments,
results and source snapshots remain historical evidence. Do not overwrite or
rename them to represent either new experiment.

## 2. Fixed constraints and scientific scope

- Reuse PC-01 epoch 6, seed 42 and its verified processor/base artifacts.
- No retraining, checkpoint mixing/substitution or changes to learned thresholds.
- Preserve the existing 1,974 embeddings, retrieval configuration and source
  identities. No regeneration, invented corrective values or reflections.
- Reuse the lab environment; avoid package changes unless a specific documented
  incompatibility requires them. Check existing jobs before inference.
- Keep Gold images on Kaggle. The existing locked-test restriction remains in
  force until access is explicitly authorised under the frozen protocol.
- Preserve notebooks, unrelated worktree changes and all completed experiments.
- Do not choose models, subsets, prompts or reporting metrics because they make
  our completed evaluation results look better.
- Distinguish engineering tests, validation results, final-test results and live
  task completion. None is a substitute for another.

| Pillar | Task 1 evidence | Task 2 role and limitation |
|---|---|---|
| P1: failure-aware resilience | Failure detection/type, recovery need/strategy and observed recovery-outcome assessment | Diagnose an observed failure, inform a recovery action and assess its execution |
| P2: multimodal decision | Predictions from causally available screenshots/text; optional declared input sensitivity | Supply the module with the appropriate current and completed-transition observations |
| P3: actions and grounding | Supporting six-class and grounding results, including weaknesses | The existing agent supplies actions and grounding; its performance is not evidence that our trained grounding head improved |
| P4: memory | Storage-decision prediction on examples with valid storage labels | Frozen advisory retrieval; benefit requires an additional measured completion improvement |

The new modular integration will not independently isolate gains from all four
pillars. Claims must reflect the actual implementation and measured comparisons.

## 3. Task 1 — Two-dataset comparison using the same models

### 3.1 Questions

- Does our trained model outperform suitable comparison models on our labelled
  prediction tasks under matched information?
- Does a compatible capability transfer to independently collected public data?
- Which classes, source domains and observed-agent behaviours remain difficult?

These are prediction/assessment questions, not a browser-completion contest.
Comparison models are acting as assessors of recorded interactions.

### 3.2 Comparison systems

| System | Role | Selection status |
|---|---|---|
| PC-01 epoch 6, seed 42 | Frozen proposed model | Fixed |
| Pinned unadapted Qwen base | Reference for the trained system's starting model | Existing artifact; assessment interface to specify |
| Independent multimodal model | External comparator beyond the original model family | Select for relevant capabilities, availability and resource fit |
| Optional stronger multimodal model | More demanding reference | Include only if resources and access permit |
| Constant baseline | Class-frequency reference | Define decisions from permitted training/development data, never final-test labels |

Freeze exact identities and prompts before final evaluation. Use the same core
model roster on both datasets. Do not require an unsuitable coordinate-only GUI
model to answer diagnostic questions merely because it is called an agent.

PC-01 is task-trained; prompted baselines may be zero-shot. Disclose that
asymmetry. The comparison does not isolate the benefit of the architecture from
the benefit of task-specific training.

### 3.3 Dataset A — Our dataset

Use existing split identities and label definitions. Record which data selected
the checkpoint or informed development. Checkpoint-selection validation results
are not independent final-test evidence.

| Task | Required evaluation boundary | Measurements |
|---|---|---|
| Interaction outcome | Completed action-conditioned transition | MCC, balanced accuracy, macro-F1 |
| Failure type | Documented labels and causally available context | Macro-F1, per-class precision/recall, confusion matrix |
| Recovery need | Failure-stage observation before the recovery outcome | Macro-F1 and per-class results |
| Recovery strategy | Information available when selecting recovery | Macro-F1, per-class results, valid-label denominator |
| Recovery-outcome assessment | Actual observed recovery transition | MCC, balanced accuracy, macro-F1 |
| Memory-storage decision | Inputs available at the defined storage decision | MCC, macro-F1, positive-class precision/recall |

Do not call post-recovery assessment a prediction of whether an unexecuted
recovery will succeed. Audit each head's actual input contract and label mask.
Do not expose recovery outcomes while assessing an earlier strategy decision.

Prefer the natural evaluation distribution for the main comparison. If an
enriched recovery subset is useful, freeze it as a separate analysis, disclose
its sampling, and rescore every system on the identical rows. Do not mix its
accuracy with full-split results or count repeated rows as independent evidence.

Retain supporting action and bounding-box results in the report/appendix, with
a main-text limitation if grounding remains part of the proposed architecture.

### 3.4 Dataset B — Independent public benchmark

**First candidate: McGill NLP AgentRewardBench.** It evaluates judgments about
recorded web-agent trajectories and supplies trajectories, screenshots and
annotations. Reusing those records does not require hosting the source websites.

- [Paper](https://arxiv.org/abs/2504.08942)
- [Official code and scoring workflow](https://github.com/McGill-NLP/agent-reward-bench)
- [Dataset](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench)

**Alternative candidate: Agent-RewardBench, ACL 2025.** This is a different
benchmark of multimodal reward modelling, including web-related scenarios.

- [Paper](https://aclanthology.org/2025.acl-long.857/)
- [Official repository](https://github.com/Quester-one/Agent-RewardBench)

Neither candidate is selected yet. Complete a small development-schema check:

1. Confirm available observations, action representation and annotation meaning.
2. Compare them with the frozen model's actual processor and head contracts.
3. Specify any deterministic adapter, missing-data handling and eligibility rules.
4. Establish whether the task is trajectory assessment, transition assessment or
   pairwise preference. These are different targets.
5. Verify overlap against existing source identities and pin the dataset revision.
6. Freeze the shared evaluation view, official scoring and any adapted subset.

Never propagate whole-trajectory failure labels to every step, fabricate absent
recovery/storage labels, or silently use the last-step probability as whole-task
success. If a trajectory aggregation rule is proposed, validate and freeze it
using development data and identify it as an additional method component.

If no defensible mapping exists, report the incompatibility and select a better
matched benchmark before test access. Do not force an external score. An adapted
subset must not be presented as the full official leaderboard evaluation.

### 3.5 Matched inference and analysis

- Give all systems the same permitted semantic information and examples. Apply
  documented model-specific image/token formatting without leaking labels.
- Use fixed structured prompts for prompted baselines and preserve raw outputs.
- Declare output validation and retry rules in advance. Report invalid outputs;
  never silently drop them or manually correct answers.
- Save example/task/episode identifiers, predictions, probabilities where
  available, reference labels in scoring-only records, latency and resource use.
- Freeze primary endpoints, paired comparisons and multiplicity handling before
  test outcomes. Report effect sizes and 95% confidence intervals.
- Resample appropriate task/episode groups for correlated observations. State
  that these intervals do not estimate variation across training seeds.
- Use official public-benchmark scoring plus justified supplementary metrics.
- Keep public transfer results separate from our dataset results. Report domain
  and generating-agent breakdowns where metadata and sample sizes support them.

### 3.6 Task 1 checklist, in execution order

- [ ] T1.1 Write the capability/input/label matrix from the actual frozen model.
- [ ] T1.2 Select the comparison models and document resource/access needs.
- [ ] T1.3 Check public development schemas and choose a compatible task.
- [ ] T1.4 Specify our evaluation split, access status and all eligibility rules.
- [ ] T1.5 Freeze manifests, prompts, adapters, endpoints and scoring rules.
- [ ] T1.6 Verify the offline harness on development examples only.
- [ ] T1.7 Resolve required final-test access before opening restricted data.
- [ ] T1.8 Run all selected models on the same eligible examples of our dataset.
- [ ] T1.9 Run the same models on the selected public evaluation task.
- [ ] T1.10 Audit completeness, invalid outputs, split isolation and paired scoring.
- [ ] T1.11 Report every declared endpoint, uncertainty and limitations.
- [ ] T1.12 Summarise which capabilities are supported for Task 2 integration.

Deliverables: frozen protocol, model/data manifests, tested adapters, raw
prediction records, comparison tables, error analysis and a Task 1 results report.
No comparison is marked complete because only PC-01's old report exists.

## 4. Task 2 — Existing agent plus diagnosis/recovery, then memory

**Start after Task 1 is completed and reported.** Use new experiment identities
A/B/C; do not replace historical E/H configurations.

### 4.1 Systems and contrasts

| System | Behaviour | Meaning |
|---|---|---|
| A | Existing executing agent with its declared standard recovery behaviour | Baseline |
| B | Same agent plus PC-01 diagnosis, recovery guidance and outcome assessment | B−A: effect of the added module as a package |
| C | B plus the existing frozen advisory memory | C−B: additional memory effect |

Select an agent that supports the required action types and exposes a documented
extension point for transition assessment and recovery. Verify its action
execution on development tasks; choose on functional fit, not final-test wins.
The exact agent, version and model backend remain to be selected.

### 4.2 Execution loop

1. The existing agent observes the page and chooses a normal browser action.
2. The executor validates and performs that action, then captures the result.
3. In B/C, the frozen module assesses the completed transition and provides the
   declared failure/strategy signals when recovery is triggered.
4. The existing planner selects the concrete recovery action and its values.
5. In C only, frozen memory may supply applicable advisory context at the declared
   point; retain the corresponding no-memory decision for audit.
6. The browser executes the model-selected action. The trained module assesses
   the recovery transition, and the controller follows fixed continuation rules.
7. Stop on task termination, truncation or the frozen budget. Score independently.

Specify before implementation exactly how each prediction changes control flow
or planner context. Merely logging predictions does not constitute integration.
Keep terminal success, observed intermediate changes and learned assessment
separate. Do not replace model choices with scripted task solutions.

### 4.3 Fairness and causal interpretation

- Match tasks, resets, actor/backend, observation access, execution interface and
  total budgets across A/B/C. Disclose module overhead and actual resource use.
- Give A comparable opportunities to recover. Add a generic-assessor control if
  B otherwise receives an additional planner/assessment opportunity. Without
  that control, B−A measures the package rather than the trained assessor alone.
- Make memory the only intended B/C difference. Empty applicable memory must
  preserve B behaviour. Freeze retrieval settings and perform zero memory writes.
- Retain failed proposals and outcomes. Apply identical infrastructure-failure
  handling and exclusions; keep policy failures in the completion denominator.
- Never provide evaluator answers or reward as planning/assessment inputs.

### 4.4 Development, freeze and evaluation

First verify real action execution, diagnosis delivery, executed recovery,
post-recovery assessment and memory admission/abstention in a bounded matched
development run. Include multi-step execution in the functional checks. Scripted
fixtures establish engineering correctness, not live model effectiveness.

Then freeze task/reset selection, overlap exclusions, budgets, stopping rules,
completion scoring, analysis and source identities before final outcomes.
Use available benchmark infrastructure where suitable. Do not inherit old
episode counts or budgets without specifying the new protocol.

Primary endpoints are paired task completion differences B−A and C−B. Report
counts, improved/worsened pairs, uncertainty and prespecified paired tests with
multiplicity handling. Secondary evidence includes recovery execution, memory
exposure, decision changes, model calls, executor requests and latency.

Exposure to memory is not proof of a beneficial intervention. A correct module
may have a null or negative effect; report it rather than repeatedly evaluating
until an improvement appears.

### 4.5 Task 2 checklist

- [ ] T2.1 Review Task 1 findings and choose the existing agent/backend.
- [ ] T2.2 Freeze development scope, A/B/C definitions and any generic control.
- [ ] T2.3 Specify and implement the minimal diagnosis/recovery adapter.
- [ ] T2.4 Connect frozen memory and verify B/C isolation and zero writes.
- [ ] T2.5 Test temporal boundaries, values, feedback and exact budget accounting.
- [ ] T2.6 Run and audit matched live development episodes.
- [ ] T2.7 Resolve implementation defects; record faithful poor predictions.
- [ ] T2.8 Freeze the live evaluation and confirm execution readiness.
- [ ] T2.9 Run matched systems and persist every episode immediately.
- [ ] T2.10 Report both primary contrasts, costs, failures and limitations.

## 5. Paper outputs and claim boundaries

| Evidence | Defensible claim if supported |
|---|---|
| Task 1 on our held-out data | Better prediction on the specified labelled tasks than the evaluated comparators |
| Task 1 on independent public data | Transfer of the specified assessment capability under the declared public protocol |
| Positive supported B−A result | Adding the frozen module improved the evaluated existing agent under matched conditions |
| Positive supported C−B result | Frozen memory provided additional benefit for that agent and task set |

Task 1 does not establish improved browser completion. Task 2 does not establish
that PC-01's trained action/grounding heads became stronger, since the external
agent supplies those functions. Neither guarantees journal acceptance.

Proposed manuscript outputs:

- Dataset/split and eligible-label table.
- Main offline comparison on our dataset.
- Independent public benchmark comparison and transfer breakdowns.
- Failure, recovery, storage and supporting action/grounding diagnostics.
- Separate A/B/C live-agent table after Task 2, with both contrasts and costs.
- Reproducibility appendix and preserved historical E/H results.

## 6. Relation to earlier plans

[EXTERNAL_AGENT_RECOVERY_EVALUATION_PLAN.md](EXTERNAL_AGENT_RECOVERY_EVALUATION_PLAN.md)
contains an earlier validation-enriched comparison proposal. Its fixed model
roster, sample size, timing estimate and expected winners are not adopted here.
Its future-success wording must not override the actual post-recovery assessment
contract. Do not infer that existing agents universally lack recovery mechanisms.

[TABLE2_HYBRID_AGENT_TODO.md](TABLE2_HYBRID_AGENT_TODO.md) records completed and
proposed historical hybrid work. Its open engineering items are not the immediate
priority under this new sequence.

**Next action: T1.1–T1.3 — produce the shared task/input/label specification,
proposed comparator roster and public-benchmark compatibility decision. Do this
before downloading models or opening final-test data.**
