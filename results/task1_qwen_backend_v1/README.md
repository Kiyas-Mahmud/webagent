# Task 1 results — completed Qwen-backend comparison

- **result_matrix.csv**: four-system overview; start here.
- **interaction_metrics.csv**: detailed metrics for 240 interaction cases per system.
- **recovery_metrics.csv**: detailed metrics for 120 recovery cases per system.

These files export the completed `task1-local-v1` run. The three external rows use
adapted assessment prompts with frozen Qwen2-VL-2B-Instruct; our row uses the
trained InternVL3.5-8B full epoch-0 checkpoint. This is not an all-InternVL run
or a native browser-agent task-completion comparison.

Scores are numeric fractions (0.8041667 means 80.41667%). MCC, balanced accuracy
and Macro-F1 use valid predictions; all-case accuracy counts invalid predictions
as incorrect. Every valid external outcome prediction was SUCCESS. No new
inference or recalculation was performed for this export.

Source: [verified report](../../.task1-assets/runs/task1-local-v1/report/results.json).
Interpretation: [results document](../../docs/TASK1_LOCAL_ASSESSMENT_RESULTS.md).
