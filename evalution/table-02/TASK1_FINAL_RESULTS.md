# Task 1 final results: recorded interaction and recovery assessment

## Purpose

Task 1 tests whether the recorded dataset supports reliable assessment of
browser interactions and recovery transitions across model configurations.
The system receives the recorded observations and action context while the
gold outcome remains hidden. It predicts the assessment, which is then scored
against the held-out label.

This is an offline recorded-transition evaluation. It is not a live browser
completion experiment and does not measure memory intervention.

## Fixed evaluation

| Phase | Cases | Input | Output assessed |
|---|---:|---|---|
| Interaction assessment | 240 | Before observation, recorded action, after observation | Outcome and failure category |
| Recovery assessment | 120 | Linked failure observation, recovery action, post-recovery observation | Recovery outcome |

The seven configurations produced 2,520 system-phase records. All rows used
the same cases, seed 42 and strict single-object output contract.

## Table 1A — Interaction assessment

The following paper-facing rows have reportable metric values. The WebVoyager
base configuration produced no valid structured assessment and is retained in
the coverage audit rather than represented by a fabricated score.

| System | Outcome MCC ↑ | Balanced accuracy ↑ | Macro-F1 ↑ | Failure Macro-F1 ↑ | Valid accuracy ↑ | All-case accuracy ↑ | Valid outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Browser Use prompt + Qwen base | 0.0000 | 0.5000 | 0.2368 | 0.1184 | 0.3103 | 0.1125 | 87/240 |
| Agent S2 prompt + Qwen base | 0.0000 | 0.5000 | 0.1827 | 0.0918 | 0.2235 | 0.1583 | 170/240 |
| Browser Use prompt + trained Qwen | 0.1511 | 0.5342 | 0.3141 | 0.1623 | 0.3645 | 0.1625 | 107/240 |
| Agent S2 prompt + trained Qwen | 0.0000 | 0.5000 | 0.2381 | 0.1190 | 0.3125 | 0.1250 | 96/240 |
| WebVoyager prompt + trained Qwen | −0.2757 | 0.4464 | 0.2193 | 0.1096 | 0.2809 | 0.1042 | 89/240 |
| **Our trained Qwen2.5 heads** | **0.6487** | **0.8472** | **0.8196** | **0.5564** | **0.8542** | **0.8542** | **240/240** |

## Table 1B — Recovery assessment

| System | Recovery MCC ↑ | Balanced accuracy ↑ | Macro-F1 ↑ | Valid accuracy ↑ | All-case accuracy ↑ | Valid outputs |
|---|---:|---:|---:|---:|---:|---:|
| Browser Use prompt + Qwen base | −0.1216 | 0.4762 | 0.4000 | 0.6667 | 0.1667 | 30/120 |
| Agent S2 prompt + Qwen base | −0.1043 | 0.4891 | 0.3309 | 0.4945 | 0.3750 | 91/120 |
| Browser Use prompt + trained Qwen | −0.0616 | 0.4839 | 0.4615 | 0.8571 | 0.2500 | 35/120 |
| Agent S2 prompt + trained Qwen | −0.0756 | 0.4833 | 0.4462 | 0.8056 | 0.2417 | 36/120 |
| WebVoyager prompt + trained Qwen | 0.0000 | 0.5000 | 0.4412 | 0.7895 | 0.1250 | 19/120 |
| **Our trained Qwen2.5 heads** | **0.8408** | **0.9167** | **0.9163** | **0.9167** | **0.9167** | **120/120** |

## Result statement for the paper

The trained Qwen2.5 heads produced valid structured outputs for every selected
case and achieved the strongest outcome and recovery-assessment scores in the
Task 1 comparison. The prompted decoder configurations had lower valid-output
coverage and lower classification scores. These results support the use of the
trained assessment heads for the subsequent failure-diagnosis and memory
experiments; they do not by themselves establish live browser completion.

## Recommended Task 1 figures

Use the following files directly in the paper. Every figure is available as a
high-resolution PNG and a vector PDF.

### Figure 1 — Core assessment metrics

[PNG](imagesorpdf/task1_core_metrics.png) ·
[PDF](imagesorpdf/task1_core_metrics.pdf)

Two rows compare interaction and recovery MCC, balanced accuracy and Macro-F1.
The selected Qwen2.5 heads are highlighted in green, base decoder rows in blue,
and trained decoder rows in orange.

### Figure 2 — Output coverage and accuracy

[PNG](imagesorpdf/task1_coverage_accuracy.png) ·
[PDF](imagesorpdf/task1_coverage_accuracy.pdf)

The panels show valid-output coverage, valid versus all-case accuracy, and
interaction failure-category Macro-F1. This figure makes the structured-output
coverage difference visible without adding unavailable metric labels.

### Figure 3 — Response validity audit

[PNG](imagesorpdf/task1_response_validity.png) ·
[PDF](imagesorpdf/task1_response_validity.pdf)

The stacked bars show valid responses, parse errors and abstentions for every
configuration in both phases. Use this as a supplementary figure if the main
paper needs a shorter visual section.

## Reproducibility files

- [`tables/interaction_metrics.csv`](tables/interaction_metrics.csv)
- [`tables/recovery_metrics.csv`](tables/recovery_metrics.csv)
- [`tables/result_matrix.csv`](tables/result_matrix.csv)
- [`tables/paired_differences.csv`](tables/paired_differences.csv)
- [`audit/freeze.json`](audit/freeze.json)
- [`audit/preflight.json`](audit/preflight.json)
- [`audit/verification.json`](audit/verification.json)
- [`imagesorpdf/generate_task1_figures.py`](imagesorpdf/generate_task1_figures.py)
