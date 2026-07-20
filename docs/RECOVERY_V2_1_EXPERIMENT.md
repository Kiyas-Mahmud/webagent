# Recovery v2.1 evidence-driven correction

## Why this version exists

The first two recovery-v2 epochs established three facts:

- causal recovery-outcome prediction works: MCC increased from 0.4970 to 0.7268;
- the unweighted needs-recovery head predicted only the negative class (MCC 0);
- bbox metrics were effectively unchanged at mean IoU 0.0079 and Recall@IoU50 0.

They also exposed an evaluation defect: attempted-strategy metrics used the composed recovery
decision. When the binary gate predicted `NONE`, the evaluator discarded the conditional
strategy prediction and reported strategy accuracy 0. This did not prove that the strategy
classifier itself failed.

## Registered changes from v2

1. Evaluate attempted-recovery strategy directly from conditional strategy logits. Retain the
   composed `needs_recovery -> strategy/NONE` result as a separate end-to-end recovery metric.
2. Derive needs-recovery BCE `pos_weight = negative / positive` from the exact selected training
   rows, capped at 5.0. The observed 5k v2 subset gives `3809 / 1191 = 3.20`; validation labels
   never affect the weight.
3. Give bbox grounding its own residual adapter rather than sharing the action-classification
   policy adapter. This directly implements task-specific isolation for localization.
4. Record first-step gradient norms and per-epoch parameter-update norms for bbox, needs recovery,
   strategy, recovery outcome, and the grounding adapter.
5. Record learned uncertainty log-variances/multipliers, nominal weighted loss separately from
   the possibly negative uncertainty objective, spatial-token counts in smoke, and prediction/
   target bbox coordinate distributions in diagnostics.
6. Expand checkpoint round-trip verification to action, bbox, recovery strategy, needs recovery,
   recovery outcome, and the grounding adapter.

No data, split, backbone, validation subset, batch size, accumulation, learning rates, or primary
selection metric changes. No new data collection is justified by these two epochs.

## Execution gates

Use `notebooks/kaggle_gold_recovery_v2_1.ipynb` in this order:

1. `smoke`: 16 rows. Every named head/adapter must have finite nonzero gradient and update norms;
   bbox supervision and image-token masks must be nonempty.
2. `diagnostic`: one controlled 5k/500 epoch. Inspect both binary recovery classes, raw conditional
   strategy metrics, bbox IoU/distribution, gradient/update norms, and uncertainty multipliers.
3. `mini`: five epochs only after diagnostic evidence is healthy.

The one-epoch diagnostic is deliberately not called a quality result. Full/headline training stays
blocked until the complete five-epoch experiment and existing human-review/hash gates pass.
