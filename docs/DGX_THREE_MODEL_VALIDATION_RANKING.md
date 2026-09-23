# Three-model validation ranking

The three full seed-42 reports were pulled from the GitHub result branches and
checked with the section-10 validator. Each report passed its individual gates:
24,107 training rows, 7,861 original-validation rows, 194 supplement rows and
zero locked-test rows.

## Descriptive validation order

The registered ranking rule is outcome MCC, followed by recovery-outcome MCC,
action Macro-F1, lower outcome ECE and model ID. Applying that metric order to
the three reports gives this descriptive order:

| Rank | PC | Model | Selected epoch | Outcome MCC | Recovery MCC | Failure Macro-F1 | Action Macro-F1 | Memory MCC | Mean IoU | Recall@IoU50 | ECE |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PC2 | Qwen2.5-VL-7B | 0 | 0.6783 | 0.8522 | 0.8391 | 0.3182 | 0.6945 | 0.1549 | 0.0830 | 0.0725 |
| 2 | PC3 | InternVL3.5-8B-HF | 0 | 0.6410 | 0.8397 | 0.8201 | 0.3370 | 0.6543 | 0.0922 | 0.0954 | 0.0402 |
| 3 | PC1 | Qwen2-VL-2B | 6 | 0.6242 | 0.7955 | 0.8094 | 0.3267 | 0.6625 | 0.1010 | 0.0809 | 0.1626 |

PC2 is first on the preregistered outcome-MCC ranking. PC3 has the highest
action Macro-F1, grounding Recall@IoU50 and calibration (lowest ECE), while PC2
has the strongest outcome, recovery, failure-category and memory MCC values.

## Quality-gate status

The final section-10 validator did not promote a winner because the run
contracts were generated from different source commits:

| PC | Run-contract commit |
|---|---|
| PC1 | `2fadf0f508cec42ce6f89b8961db7cfd2adef1df` |
| PC2 | `2bd3d0de067da49f8426bd009a69f6e66a0d1bfa` |
| PC3 | `628c1fa776142c8169466c53865e694aa01d6b8d` |

Therefore the order above is **provisional descriptive evidence**, not a
fully matched final model-selection result. The locked test was not opened.
To promote a final winner, either rerun the three candidates from one frozen
source commit or obtain an explicit protocol decision allowing these source
commit differences.

Diagnostic outputs are saved outside the repository at
`/home/aiub/kiyas/webagent_comparison/outputs/comparison_decision/`:

- `three_model_validation_comparison_diagnostic.csv`
- `three_model_selection_diagnostic.json`
- `three_model_quality_gate_failure.json`
