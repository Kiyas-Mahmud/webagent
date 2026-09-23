# Task 1: Qwen2.5-VL-7B selected-model assessment results

Status: complete and independently audited on 2026-09-23.

This follow-up evaluates the same frozen 240 interaction cases and 120 linked
recovery cases used by the completed InternVL Task 1 pilot. It uses the selected
Qwen2.5-VL-7B epoch-0, seed-42 checkpoint and the pinned Qwen2.5 base snapshot.
There were seven rows and 2,520 system-phase records in total. No live browser
execution was performed.

The frozen assets and raw records are in
`.task1-assets/runs/task1-qwen25-dual-v1/`. The machine-readable exports are in
`results/task1_qwen25_dual_v1/`. The verification receipt reports 2,520 records,
90 independently recomputed metric values, and `PASS`.

## Main result: trained Qwen2.5 heads

| Phase | MCC | Balanced accuracy | Macro-F1 | Failure Macro-F1 | Accuracy | Valid / eligible |
|---|---:|---:|---:|---:|---:|---:|
| Interaction assessment | 0.6487 | 0.8472 | 0.8196 | 0.5564 | 0.8542 | 240 / 240 |
| Recovery assessment | 0.8408 | 0.9167 | 0.9163 | N/R | 0.9167 | 120 / 120 |

These are scores on the selected 240/120 comparison subset. They must not be
replaced with the checkpoint's full-validation scores. The full exports also
include memory, strategy, action and grounding metrics from the original model
validation run; those are separate reference evidence and were not recomputed
by this assessment pilot.

## Prompted decoder rows

The six decoder rows use the same frozen Qwen2.5 base or trained LoRA decoder
with adapted Browser Use, Agent S2, or WebVoyager assessment instructions. They
are component adaptations, not native implementations of those agents.

| Row | Interaction MCC | Interaction Macro-F1 | Recovery MCC | Recovery Macro-F1 | Interaction valid | Recovery valid |
|---|---:|---:|---:|---:|---:|---:|
| Browser Use, base | 0.0000 | 0.2368 | -0.1216 | 0.4000 | 87 / 240 | 30 / 120 |
| Agent S2, base | 0.0000 | 0.1827 | -0.1043 | 0.3309 | 170 / 240 | 91 / 120 |
| WebVoyager, base | N/R | N/R | N/R | N/R | 0 / 240 | 0 / 120 |
| Browser Use, trained | 0.1511 | 0.3141 | -0.0616 | 0.4615 | 107 / 240 | 35 / 120 |
| Agent S2, trained | 0.0000 | 0.2381 | -0.0756 | 0.4462 | 96 / 240 | 36 / 120 |
| WebVoyager, trained | -0.2757 | 0.2193 | 0.0000 | 0.4412 | 89 / 240 | 19 / 120 |

The adapted decoder rows have high parse-error and abstention rates. WebVoyager
base produced no valid structured output. Therefore these rows are useful
diagnostics of the frozen decoder and output contract, but they do not establish
that the corresponding native agents perform poorly.

## Comparison with the completed InternVL pilot

The trained-head rows can be compared descriptively because both pilots use the
same selected cases, labels, phase definitions and scoring rules.

| Metric | Qwen2.5 heads | InternVL heads | Qwen2.5 minus InternVL |
|---|---:|---:|---:|
| Interaction MCC | 0.6487 | 0.5583 | +0.0904 |
| Interaction balanced accuracy | 0.8472 | 0.8083 | +0.0389 |
| Interaction Macro-F1 | 0.8196 | 0.7678 | +0.0519 |
| Failure-category Macro-F1 | 0.5564 | 0.5287 | +0.0277 |
| Recovery MCC | 0.8408 | 0.8844 | -0.0436 |
| Recovery Macro-F1 | 0.9163 | 0.9416 | -0.0253 |
| Recovery all-case accuracy | 0.9167 | 0.9417 | -0.0250 |

Under the selected-model rule, Qwen2.5 is the stronger overall choice for the
interaction/outcome assessment. InternVL is slightly stronger on the linked
recovery subset. The evidence does not support claiming that one backbone is
universally superior across every capability.

## What this establishes

The selected Qwen2.5 checkpoint can reliably produce the structured outcome,
failure-category and recovery assessments used by the study. It achieved full
valid-output coverage on the trained-head route and strong MCC/Macro-F1 scores
on the frozen comparison subset. This supports using Qwen2.5 for the next
failure-diagnosis and memory analysis, subject to the same offline assessment
scope.

The prompted rows show that a general decoder prompt is not equivalent to the
trained assessment heads. Their invalid outputs must remain in the denominator
for all-case accuracy and cannot be hidden by reporting valid-only metrics.

This experiment does not establish live browser completion, native external-agent
performance, executed recovery, or a memory improvement. It is a recorded
transition assessment follow-up on previously observed validation cases, not a
new locked test. Confirmatory significance claims are not made.

## Reproduction and audit files

- `results/task1_qwen25_dual_v1/result_matrix.csv`
- `results/task1_qwen25_dual_v1/interaction_metrics.csv`
- `results/task1_qwen25_dual_v1/recovery_metrics.csv`
- `results/task1_qwen25_dual_v1/paired_differences.csv`
- `.task1-assets/runs/task1-qwen25-dual-v1/report/verification.json`
- `.task1-assets/runs/task1-qwen25-dual-v1/freeze.json`

