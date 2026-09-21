# InternVL dual v2: post-hoc coverage audit

Read the [analysis and interpretation](../../docs/TASK1_INTERNVL_COVERAGE_ERROR_ANALYSIS.md).
This folder supplements the completed comparison; it does not replace or edit
the original metrics in `results/task1_internvl_dual_v2/`.

| File | Contents |
|---|---|
| `summary.json` | Counts, literal reason-theme rules, source verification and zero-inference declaration |
| `coverage_by_phase.csv` | Seven configurations × two phases, valid/abstention/error counts |
| `coverage_by_action.csv` | Same counts separated by recorded action class |
| `coverage_by_pixel_change.csv` | Coverage separated by exact RGB equality of the image pair |
| `input_audit.csv` | 360 frozen requests: input availability, descriptions and image dimensions/equality |
| `response_audit.csv` | 2,160 decoder responses with original raw text, source path/hash and diagnostics |
| `parse_errors.csv` | The 40 rejected outputs; no repair or relabelling |
| `abstention_reasons.csv` | Exact reason-string frequencies by configuration and phase |
| `audit_receipt.json` | Test result, artifact hashes and final preservation checks |

`valid` means a scorable prediction, not a correct prediction. Coverage uses all
eligible requests. Heads appear in coverage tables but have no generated-text
rows. Reason themes overlap and describe literal model statements, not proven
causes. Exact image equality compares decoded RGB dimensions and bytes; different
pixels do not establish task progress. Repeated cases across configurations and
linked recovery phases are not independent observations.

The input builder was inspected to establish the absence of target/history in
the frozen contract. The audit does not claim those fields are absent from the
original collector. Every input image used here already belonged to this pilot.

Reproduction command (requires the preserved local evidence and image assets;
choose a new output directory):

```bash
PYTHONPATH=src .venv/bin/python scripts/analysis/task1_coverage_audit.py --out /tmp/task1-coverage-audit-copy
```

The script refuses to overwrite its output directory and verifies the original
5,069-file evidence manifest before and after analysis. The receipt additionally
binds this report and its regression fixtures. No inference, prompt revision,
package change or completed-score change was performed.
