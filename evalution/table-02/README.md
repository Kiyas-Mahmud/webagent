# Task 1 evaluation archive

This directory contains the completed Task 1 recorded-transition assessment.
It is separate from the later live-agent and memory evaluations.

## Scope

- 240 recorded interaction cases.
- 120 linked recovery cases.
- Seven frozen assessment configurations.
- 2,520 system-phase records in total.
- Qwen2.5-VL-7B, epoch 0, seed 42 for the selected model.
- No live browser execution was performed.

The paper-facing result and figure descriptions are in
[`TASK1_FINAL_RESULTS.md`](TASK1_FINAL_RESULTS.md). The machine-readable CSV
exports in [`tables/`](tables/) remain the authoritative result source. The
figures in [`imagesorpdf/`](imagesorpdf/) are exported in both PNG and PDF
formats.

## Archive layout

```text
table-02/
├── TASK1_FINAL_RESULTS.md
├── TASK1_QWEN25_DUAL_V1_RESULTS.md
├── README.md
├── audit/
│   ├── freeze.json
│   ├── preflight.json
│   └── verification.json
├── imagesorpdf/
│   ├── task1_core_metrics.{png,pdf}
│   ├── task1_coverage_accuracy.{png,pdf}
│   ├── task1_response_validity.{png,pdf}
│   └── generate_task1_figures.py
└── tables/
    ├── interaction_metrics.csv
    ├── recovery_metrics.csv
    ├── result_matrix.csv
    ├── paired_differences.csv
    └── README.md
```

The figures omit text labels for unavailable metric cells so the paper visuals
remain clean. The corresponding zero-valid-output case is retained in the CSV
exports and shown in the coverage and response-validity figures.
