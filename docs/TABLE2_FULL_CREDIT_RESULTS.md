# Revised Table 2 full-credit results — 2026-09-10

**Completed: 24/24 episodes; execution and context audits PASS.** No runtime
errors, overlap exclusions, invalid paired blocks, episode reruns or changes to
the frozen implementation occurred. This is the requested BrowserGym + MiniWoB
run with PC-01 epoch 6 / seed 42, frozen train-only memory context and the revised
control/retry interfaces. No retraining or package changes occurred.

## Full-completion results

Success requires task termination without truncation and raw MiniWoB reward
exactly 1.0. Positive partial reward is not full completion. Scoring receives
task-validation output after execution; it is not passed to action selection.

| System | Full completions | Rate | Executed normal actions | Executed recoveries | Memory queries | Memory context interventions |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0/6 | 0.0% | 1 | 0 | 0 | 0 |
| E1 | 0/6 | 0.0% | 0 | 0 | 0 | 0 |
| E2 | 1/6 | 16.7% | 0 | 1 | 0 | 0 |
| E3 | 1/6 | 16.7% | 0 | 1 | 11 | 11 |

| Task family | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| click-link | Failure | Failure | Failure | Failure |
| click-option | Failure | Failure | Failure | Failure |
| click-checkboxes | Failure | Failure | Failure | Failure |
| enter-password | Failure | Failure | Failure | Failure |
| login-user | Failure | Failure | Failure | Failure |
| focus-text | Failure | Failure | Success | Success |

| Paired contrast | Pairs | Completion difference | Improved / worsened | Exact two-sided p | Holm p |
|---|---:|---:|---:|---:|---:|
| E2 minus E1: recovery | 6 | +16.7 percentage points | 1 / 0 | 1.0 | 1.0 |
| E3 minus E2: memory | 6 | 0.0 percentage points | 0 / 0 | 1.0 | 1.0 |

Recovery added one observed completion. Neither contrast is statistically
significant. **This run did not demonstrate additional completion from memory.**
Memory exposure is verified in all 11 E3 generation calls, so zero improvement
cannot be explained by complete memory abstention in this run.

## Actual failure stages

- E1 emitted NAVIGATE without an executable URL; no normal trained action
  reached the browser. Its failures remain included.
- E2 and E3 each made 11 recovery generation calls. Ten per system were rejected
  at **named-target/action compatibility validation**, before browser execution.
  These are model/interface failures, separate from runtime crashes.
- On link, option and checkbox tasks, the planner commonly proposed TYPE on
  non-editable controls. One E2 checkbox retry proposed SELECT on an input.
  Visible custom-link controls were available; the link failure was no longer
  an empty discovery list.
- Password/login outputs used labels absent from the exposed control-name
  schema or ambiguous descriptions. An E2 login attempt named a bare `input`
  matching two controls. Exact matching correctly did not choose an arbitrary
  target; target-description/label coverage remains a limitation.
- The two successful `focus-text` episodes used TYPE, which focuses the input
  before inserting text. Both inserted the goal sentence and obtained terminal
  raw reward 1.0. This satisfies the frozen task score but is not evidence of a
  minimal or consistently appropriate action choice.
- E0 executed one action on `click-option`, which terminated with raw reward
  -1. Its other episodes failed before executing an action.

No recorded episode obtained positive partial credit in this run. The previous
checkbox partial-credit result belongs to an older frozen run and remains
unchanged. The previous and current runs differ in both planner inputs and
scoring, so the change from 2/6 to 1/6 must not be attributed solely to scoring.

## Evidence integrity and interpretation

All six paired resets and initial E1–E3 policy outputs matched. Event-chain,
action/outcome, retrieval replay, source binding, memory exposure and generation
context audits passed. All **22 recovery generation calls** passed the frozen
context auditor without an amendment or episode rerun.

Post-run checks verified all 165 producing source files, the checkpoint, export,
all 14 base-snapshot files and frozen memory unchanged. Earlier artifact sets
remained hash-identical: 484 original evaluation artifacts, 538 prior replication
artifacts and 584 memory-context development artifacts.

The four pillars and matched E0–E3 design were retained. The material is explicitly
training-label-backed memory; its absent corrective values/reflections remain
null. This six-family, one-reset-per-family run uses previously observed tasks.
It is a post-debugging MiniWoB replication, not fresh held-out confirmation,
a WebArena result or evidence of broad memory effectiveness. The small positive
memory contrast from development did not transfer to these six families.

## Artifacts and status

Evidence root:
`/home/aiub/kiyas/table2-evidence/miniwob-table2-completion-v1/`

- `plan.json`, `plan.sha256`, `source-snapshot/`: frozen protocol and sources.
- `results.json`, `analysis.json`, `table2.csv`: all outcomes and paired analysis.
- `evidence-audit.json`, `context-audit.json`: passed audits.
- `execution-completion.json`: successful launcher completion.
- `report-metrics.json`: verified counts, exact failure receipts and post-run
  integrity checks.
- Per-task/system directories: raw model outputs, screenshots, runtime logs and
  binary/raw rewards.

Plan SHA-256:
`cbb7ae53c4f26192979b72d47853106fe57d209442698be40091604868c76b9d`.

The new run is **complete**, so its results are now reported rather than N/R.
The pre-launch readiness receipt and frozen plan retain their historical status;
`execution-completion.json` records execution. No further evaluation is queued.
The completed package must not be relaunched or overwritten.
