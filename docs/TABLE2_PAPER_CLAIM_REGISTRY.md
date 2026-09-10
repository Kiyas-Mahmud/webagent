# Table 2 Paper-Claim Registry

The machine-readable authority is
`configs/eval/table2/paper_claim_registry_v1.json`. It freezes the permitted
interpretation of every Table 2 contrast and the complete evidence catalog
before evaluation. It does not assert that any effect is positive, significant,
or ready for publication. Every claim begins as `N/R`.

The registry distinguishes:

- the contextual E0–E1 policy difference from the stronger, conditional pure-
  training claim;
- the E1–E2 executed diagnosis/recovery increment;
- the E2–E3 frozen corrective-memory increment;
- the E0–E3 total-system difference;
- P1-specific and P1/P2/P3/P4 companion-diagnostic claims;
- cost/latency and complete negative/null-result disclosure.

It also freezes prohibited substitutions. Offline recovery MCC cannot prove
operational recovery; Recall@K cannot prove memory benefit; screenshots cannot
replace controlled causal tests; related-paper scores cannot support a cross-
benchmark superiority claim; development scenarios cannot prove all recovery
strategies generalize; pilot values cannot become final Table 2; and one model
seed cannot estimate model-seed uncertainty.

The P1 companion claim is limited to offline failure detection, diagnosis, and
assessment of an already executed recovery transition. The P4 companion claim
is limited to immutable train-only retrieval, admission, provenance, and
no-write integrity. Neither can replace the paired browser evidence: operational
recovery remains an E1-versus-E2 question and end-to-end memory benefit remains
an E2-versus-E3 question.

The P3 companion claim is named
`p3_grounded_action_parameter_diagnostic_support`. It covers offline six-class
action, grounding, parameter-provider, and invalid-output diagnostics. It does
not claim that the companion package executed a browser action; browser
execution belongs to the paired campaign evidence.

## Claim statuses

The statuses are intentionally not synonyms:

- `N/R` means not evaluated. It has no attached evidence or unmet-condition
  assessment.
- `SUPPORTED` means every registered evidence artifact and every registered
  condition is satisfied and the registered analysis supports the claim.
- `NOT_SUPPORTED` means the claim is not scientifically testable because a
  registered identification or provenance condition is unmet. It is allowed
  only for claims that register such a condition. The report must enumerate
  every unmet condition and bind it to its canonical `UNMET` assessment.
- `NEGATIVE_OR_NULL` means the experiment was valid and every condition was
  satisfied, but the effect was negative, null, or did not meet the registered
  support criterion. A measured non-positive result must not be hidden as
  `NOT_SUPPORTED`.

Primary unconditional contrasts permit `SUPPORTED` or `NEGATIVE_OR_NULL`, not
`NOT_SUPPORTED`. Required cost and complete-disclosure rows must be
`SUPPORTED` before paper-claim readiness can be issued.

The present pre-evaluation registry intentionally records
`UNREGISTERED_BLOCKS_SUPPORTED_OR_NEGATIVE_OR_NULL` for every effect and
companion-performance claim. The protocol registers confidence intervals,
sign tests, and Holm correction, but it does not yet register a claim-specific
decision threshold that maps those outputs to `SUPPORTED` versus
`NEGATIVE_OR_NULL`. The validator therefore rejects either label for those
claims. It does not infer a threshold from prose or accept an author-selected
direction. A later decision rule must be scientifically agreed, versioned, and
frozen before any outcome is inspected; until then these claims remain `N/R`.
An `UNMET` evidence assessment may document a failed identification condition,
but the present conditional claim still remains `N/R`. `NOT_SUPPORTED` becomes
available only after a future, pre-outcome registry revision defines an
executable rule for that claim. This prevents even a testability label from
being chosen after outcomes are visible.

## Evidence authority

A path plus a SHA-256 is not sufficient evidence. Each evidence kind has one
unique canonical assessment path, exact producer/schema identity, exact source
paths, and a kind-specific content validator. Validation reopens both the
assessment and its source artifacts, rejects duplicate JSON keys and symlink
traversal, recomputes every hash, and recomputes `SATISFIED` or `UNMET` from the
source content. Arbitrary JSON, a copied assessment, or one file relabelled as a
different kind fails closed.

Some source reports are deliberately shared across related facets—for example,
the registered P3 report contains action-class, grounding/parameter, and
invalid-output results. That sharing is explicit in the frozen catalog. The
three evidence assessments remain distinct and each applies its own facet
validator. Companion reports with blocked authoritative-input provenance
produce a structured `UNMET` evidence assessment, but cannot resolve the claim
while its decision rule remains unregistered. The separate
`authoritative_companion_input_provenance` condition must be `SATISFIED` before
a future registered rule could support the claim.

The v1 architecture/interface-control and learned-trigger-ablation documents
contain only author-entered booleans. They are retained as explicit records of
what is missing, but their validators always return `UNMET`; those booleans
cannot establish either conditional claim. Satisfying either condition requires
a later, pre-outcome registry revision with a source-replaying, hash-bound
paired-control package. Likewise, P4 checkpoint execution is accepted only
from the full PC-01 compatibility receipt, and companion provenance is accepted
only after all four diagnostic packages are reopened and independently
validated.

Cost evidence is checked as finite numerator/denominator accounting for every
E0–E3 system. Browser-action and model-call totals must be exact counts, and
wall-clock totals must be finite and nonnegative. The complete-disclosure
artifact enumerates every registered contrast and every registered claim
outcome using the controlled status/conclusion/limitation vocabulary; it also
uses exact, non-positive limitation codes for P1–P4 and the seed-42 model-seed
uncertainty limitation. Placeholder prose such as `none`, or positive-result
prose placed in a limitation field, is not accepted.

## Validation workflow

Validate the pre-evaluation registry:

```bash
PYTHONPATH=src python scripts/validate_table2_paper_claims.py \
  validate-registry \
  --registry configs/eval/table2/paper_claim_registry_v1.json
```

After a campaign is frozen, initialize an explicitly unevaluated report:

```bash
PYTHONPATH=src python scripts/validate_table2_paper_claims.py \
  initialize-report \
  --registry CAMPAIGN/frozen/paper_claim_registry.json \
  --campaign-manifest CAMPAIGN/campaign_manifest.json \
  --output CAMPAIGN/paper_claim_report.json
```

After final aggregate and companion artifacts exist, create each registered
assessment through the source-replaying producer. For example:

```bash
PYTHONPATH=src python scripts/validate_table2_paper_claims.py \
  assess-evidence \
  --registry CAMPAIGN/frozen/paper_claim_registry.json \
  --campaign-dir CAMPAIGN \
  --kind paired_task_effect
```

The command selects the output and source paths from the frozen catalog; callers
cannot substitute another path. It refuses to overwrite an existing
assessment. Repeat it for the evidence kinds required by each claim, then fill
the report limitations and any structured unmet conditions. Do not assign an
effect status until its frozen decision-rule field is replaced through a
pre-outcome registry revision. With the present registry, even a final campaign
whose package is `READY_FOR_TABLE2` retains `N/R` for those claims.

A pilot report must keep every claim and the paper table at `N/R`, even if
pilot metrics are favorable. A final report can resolve claims only when it is
bound to the exact passing `aggregate/validation_report.json` whose publication
status is `READY_FOR_TABLE2`; the immutable campaign manifest itself remains
pre-evaluation `N/R`. The required negative/null disclosure claim must itself
be supported, and every prohibited claim must remain `NOT_CLAIMED`.
The saved validation report is not trusted merely because it contains the text
`READY_FOR_TABLE2`: its complete content must equal a fresh strict campaign
validation result.

Validate a populated report with:

```bash
PYTHONPATH=src python scripts/validate_table2_paper_claims.py \
  validate-report \
  --registry CAMPAIGN/frozen/paper_claim_registry.json \
  --campaign-dir CAMPAIGN \
  --report CAMPAIGN/paper_claim_report.json
```

This validator checks evidence identity and publication eligibility. It never
writes paper prose and never infers a scientific conclusion from a point
estimate.

`READY_FOR_TABLE2` is only the campaign-package gate. It never means the claim
report is ready. After `validate-report` passes, issue the separate claim gate:

```bash
PYTHONPATH=src python scripts/validate_table2_paper_claims.py \
  validate-readiness \
  --registry CAMPAIGN/frozen/paper_claim_registry.json \
  --campaign-dir CAMPAIGN \
  --report CAMPAIGN/paper_claim_report.json
```

Only this command may return `READY_FOR_PAPER_CLAIMS`, and only after it
independently confirms `READY_FOR_TABLE2`, every resolved claim, every source-
replayed evidence assessment, complete negative/null disclosure, and all
prohibitions remaining `NOT_CLAIMED`. Because the present registry deliberately
lacks claim-specific effect decision rules, this gate currently fails closed
even for a structurally valid final campaign report. The engineering pilot can
never pass this gate; its status and Table 2 remain `N/R`.
