# Revised Table 2 replication results — 2026-09-10

All **24 frozen episodes executed**, with no runtime errors, overlap exclusions,
invalid paired blocks or episode reruns. This is the revised BrowserGym + MiniWoB
replication using PC-01 epoch 6 / seed 42, the corrected action/recovery interfaces
and frozen training-label memory. No retraining or package changes occurred.

## Frozen benchmark scores

| System | Benchmark-scored successes | Rate | Executed normal actions | Executed recovery actions | Memory queries | Memory interventions |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0/6 | 0.0% | 1 | 0 | 0 | 0 |
| E1 | 0/6 | 0.0% | 0 | 0 | 0 | 0 |
| E2 | 2/6 | 33.3% | 0 | 5 | 0 | 0 |
| E3 | 2/6 | 33.3% | 0 | 5 | 7 | 0 |

Success follows the frozen rule: task termination with positive BrowserGym
environment reward. This is a benchmark score, not independent verification that
every task instruction was fulfilled; see the checkbox scoring caveat below.

| Task family | E0 | E1 | E2 | E3 |
|---|---|---|---|---|
| click-link | Failure | Failure | Failure | Failure |
| click-option | Failure | Failure | Failure | Failure |
| click-checkboxes | Failure | Failure | Success* | Success* |
| enter-password | Failure | Failure | Failure | Failure |
| login-user | Failure | Failure | Failure | Failure |
| focus-text | Failure | Failure | Success | Success |

| Paired contrast | Pairs | Score difference | Improved / worsened | Exact two-sided p | Holm p |
|---|---:|---:|---:|---:|---:|
| E2 minus E1 | 6 | +33.3 percentage points | 2 / 0 | 0.5 | 1.0 |
| E3 minus E2 | 6 | 0.0 percentage points | 0 / 0 | 1.0 | 1.0 |

The observed recovery contrast is positive under the registered scoring rule,
but it is not statistically significant. E3 made seven retrieval queries and
abstained on all seven; there is no observed additional memory contribution.
These six previously observed families provide post-debugging replication
evidence, not a fresh held-out confirmatory study or a broad generalization claim.

## What the execution shows

- E1 again predicted unusable NAVIGATE actions without a URL. No normal trained
  action reached the browser; loop detection terminated those episodes.
- E2/E3 executed five recoveries each. Focusing the textbox succeeded. On the
  option, password and login tasks, the planner clicked Submit/Login before
  supplying the required values, and the benchmark returned failure.
- `click-link` uses styled clickable `span.alink` elements. The frozen visible
  control selector did not include them, so its control list was empty and
  named-target recovery was rejected. This discovery limitation is retained as
  a task failure; the selector and episodes were not changed during evaluation.
- E0 executed one normal action, which failed; its other episodes ended on
  invalid-output rejection. All those failures remain included.

**\*Checkbox scoring caveat:** The task requested selecting `BGRd` and `bmvbrq`.
E2/E3 only clicked Submit, yet BrowserGym returned reward 1 and termination.
The pinned task computes positive/negative credit over both selected and
unselected boxes; BrowserGym MiniWoB 0.14.3 maps any positive raw reward to 1.
Therefore this counted success does **not** demonstrate selecting the requested
checkboxes correctly. We retain the frozen score and explicitly disclose the
limitation rather than changing scoring after seeing results. The underlying
raw reward was not recorded by the worker; its exact value is not claimed.

Source inspection for that interpretation used the pinned local
`miniwob/html/miniwob/click-checkboxes.html` and installed
`browsergym/miniwob/base.py` after execution; no task verifier data was fed into
the policy or recovery decisions.

## Audit completion and disclosed correction

The frozen launcher completed all episodes and the runtime evidence audit, then
stopped in the context auditor. That auditor incorrectly asserted that every
generation must have a nonempty visible-control list. The four `click-link`
recovery calls faithfully received an empty list from their recorded observation.

The original auditor and its failure were preserved. A separate auditor,
`verify_context_v2.py`, compared the generated context with actual logged
controls, causal history, latest failed action and screenshot identity. It also
checks redacted strings against their SHA-256 commitments. All **14 recovery
generation calls passed**. The original source/plan, model outputs, scores,
task allocation and episodes were unchanged; no episode was rerun.

The frozen analyzer then completed. Audits verify matched resets, matched
E1–E3 initial policy outputs, event-log chains, retrieval/admission replay, causal
bindings and outcome reconciliation. All 163 producing source files and all 484
original evaluation artifacts remained unchanged. Frozen memory file hashes
also passed verification. Prior preflight/audit tests numbered 241; these are
engineering checks, separate from the 24 live episodes.

## Evidence and status

Evidence root:
`/home/aiub/kiyas/table2-evidence/miniwob-revised-replication-v1/`

- `plan.json` and `source-snapshot/`: frozen configuration and producing sources.
- `results.json`, `analysis.json`, `table2.csv`, `report-metrics.json`: full outcomes
  and the two prespecified paired contrasts.
- `evidence-audit.json`, `context-audit.json`: completed audits.
- `context-audit-amendment.json`: original failure and separate auditor correction.
- `execution-completion.json`: final execution/verification status and limitations.
- Per-task/system directories: raw generations, screenshots, browser outcomes
  and runtime event logs.

The replication is **complete**. No evaluation or retraining is queued. Original
results remain under `miniwob-revised-evaluation-v1/evaluation/` and are not
overwritten. This report can supply the revised MiniWoB Table 2 only with its
scoring, prior-exposure, small-sample and training-label-memory qualifications;
it does not replace the superseded WebArena registration with a WebArena claim.
