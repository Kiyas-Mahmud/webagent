# Task 1 — InternVL dual assessment results

Status: **complete**. These are validation-pilot results.
Post-hoc diagnostic audit: [coverage, abstentions, input limits and output errors](TASK1_INTERNVL_COVERAGE_ERROR_ANALYSIS.md). No predictions or scores were changed.
No full-validation scores are substituted. Metrics use the displayed valid denominator; all-case accuracy uses all selected cases and counts invalid outputs as incorrect. Rates are fractions (0.8042 = 80.42%).

## Table A — interaction assessment

| System | MCC | Balanced accuracy | Macro-F1 | Failure macro-F1 | All-case accuracy | Valid / eligible |
|---|---:|---:|---:|---:|---:|---:|
| browser_use_base | 0.0000 | 1.0000 | 0.5000 | 0.1667 | 0.0083 | 2 / 240 |
| agent_s2_base | 0.2530 | 0.5500 | 0.4812 | 0.2067 | 0.0708 | 26 / 240 |
| webvoyager_base | 0.0000 | 0.5000 | 0.4923 | 0.1874 | 0.0667 | 27 / 240 |
| browser_use_trained | 0.2704 | 0.6448 | 0.5462 | 0.2037 | 0.1167 | 51 / 240 |
| agent_s2_trained | -0.0934 | 0.4542 | 0.4500 | 0.2097 | 0.0667 | 35 / 240 |
| webvoyager_trained | -0.0273 | 0.4921 | 0.2184 | 0.1083 | 0.0667 | 68 / 240 |
| internvl_heads | 0.5583 | 0.8083 | 0.7678 | 0.5287 | 0.8042 | 240 / 240 |

## Table B — recovery assessment

| System | Recovery MCC | Macro-F1 | Accuracy (valid) | All-case accuracy | Valid / eligible |
|---|---:|---:|---:|---:|---:|
| agent_s2_base | 0.5292 | 0.6975 | 0.7500 | 0.0750 | 12 / 120 |
| agent_s2_trained | 0.0913 | 0.5143 | 0.5294 | 0.0750 | 17 / 120 |
| browser_use_base | 0.0000 | 0.3333 | 0.5000 | 0.0083 | 2 / 120 |
| browser_use_trained | 0.2236 | 0.6101 | 0.6667 | 0.1167 | 21 / 120 |
| internvl_heads | 0.8844 | 0.9416 | 0.9417 | 0.9417 | 120 / 120 |
| webvoyager_base | 0.0000 | 0.1818 | 0.2222 | 0.0167 | 9 / 120 |
| webvoyager_trained | 0.0000 | 0.3288 | 0.4898 | 0.2000 | 49 / 120 |

Detailed JSON includes statuses, confusion matrices, calls, latency and paired task-group 95% intervals.
No confirmatory significance claim or live browser-completion claim is made.


This is a follow-up on previously observed validation cases, not a new locked test. The six prompted configurations are adapted component instructions, not native browser agents. Base-versus-trained decoder contrasts hold the prompt and generation settings fixed; head-versus-decoder contrasts also differ in assessment interface. No live task completion, executed recovery, memory gain or confirmatory significance claim is established.


Files: [seven-row matrix](../results/task1_internvl_dual_v2/result_matrix.csv), [interaction metrics](../results/task1_internvl_dual_v2/interaction_metrics.csv), [recovery metrics](../results/task1_internvl_dual_v2/recovery_metrics.csv), [paired differences](../results/task1_internvl_dual_v2/paired_differences.csv), [execution accounting](../results/task1_internvl_dual_v2/execution_summary.csv), [audit](../results/task1_internvl_dual_v2/audit_summary.json).


## Interruption and retry accounting

The original run stopped after 1,624 saved predictions, with one additional request started but unfinished. The user explicitly authorized one audited retry. The reviewed continuation copied all 1,624 predictions byte-for-byte and generated 896 additional responses: 895 previously unstarted requests plus the authorized retry. The retry abstained. Its original interrupted attempt remains in the original archive. There are **2,520 final predictions and 2,521 unique attempts**, including that interruption. No completed, malformed or abstaining response was rerun.

`execution_summary.csv` reports the interruption/retry, coverage and known inference costs. The original interrupted attempt's call count and latency are unknown. Cost totals exclude model loading, development and engineering checks. `audit_summary.json` records preservation and independent metric checks.

## Coverage and uncertainty

Prompted interaction coverage ranges from 2/240 to 68/240; prompted recovery coverage ranges from 2/120 to 49/120. Conditional-valid metrics cannot be read as scores over the entire selection. Head coverage is 240/240 and 120/120. Common-valid base/trained pair counts are only 2, 7 and 11 for interaction, and 1, 4 and 9 for recovery (Browser Use, Agent S2, WebVoyager order). Degenerate [0,0] bootstrap intervals in tiny or single-class subsets do not establish equivalence or precise uncertainty. These are descriptive follow-up results.

The trained-head route has the highest all-case accuracy and full coverage on this selection: interaction MCC 0.5583, failure-category Macro-F1 0.5287, recovery MCC 0.8844, with full coverage. Its outcome predictions match 193/240 interaction labels and 113/120 recovery labels. This supports recorded-transition assessment performance under this protocol; it does not demonstrate improved browser execution or a memory intervention. The prompt routes use different assessment interfaces and have much lower coverage.
