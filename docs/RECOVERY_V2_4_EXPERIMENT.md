# Recovery v2.4 — Registered BBox Saturation Correction

## Decision

Recovery v2.3 is not permitted to proceed to the 5,000-row mini experiment.
Its fixed 32-row localization probe failed the two registered IoU gates.
Recovery v2.4 changes only the demonstrated width/height failure mechanism and
must pass smoke and the same fixed micro-overfit experiment before mini training.

## Evidence from the v2.3 Kaggle output

- The exact 32 selected targets passed the geometry audit; no test rows were read.
- All 100 optimizer steps completed with finite, nonzero bbox and grounding
  gradients and nonzero parameter updates.
- Attention entropy fell from about 5.52 to 1.01, so the spatial attention became
  selective, and public centre coordinates moved toward the targets.
- Mean IoU finished at `0.0`, despite bbox MAE improving to about `0.121`.
- Predicted public width and height both finished at exactly `0.0`, with zero
  standard deviation. The targets averaged roughly `0.156` width and `0.050`
  height.
- The v2.3 head computed size as `sigmoid(linear_output)`. Optimization pushed
  those logits far negative; the decoded sizes and sigmoid derivative collapsed
  to zero. This is a model parameterization defect, not evidence that the audited
  32 annotations, centre attention, optimizer, or FP32 branch failed.

## Single registered structural correction

V2.4 keeps the coordinate-softargmax centre. It predicts unconstrained
`log(width)` and `log(height)`, decodes public/internal sizes with a bounded
exponential, and supervises the **unclamped** predicted log-sizes with SmoothL1.
The direct log-size gradient therefore remains useful even if the bounded decode
is at its minimum.

The final size layer is initialized to the geometric mean width and height from
valid bbox rows in the exact selected **training** records. Invalid localization
labels are excluded in the same way as bbox supervision. Validation and test
rows are never used for this prior.

The registered bbox loss is:

```text
coordinate = centre_L1 + SmoothL1(predicted_log_wh, target_log_wh)
bbox_loss  = 5 * coordinate + 2 * GIoU_loss
```

## Fixed controls inherited from v2.3

- dataset and invalid-target masking policy;
- deterministic selected training rows (`32`, seed `42`);
- human montage review requirement;
- Qwen backbone, coordinate-aware attention, centre soft-argmax, separate
  grounding adapter, and FP32 trainable localization path;
- batch size, optimizer, learning rate, weight decay, gradient clipping;
- 100 completed finite-gradient steps with dropout disabled;
- minimum mean-IoU gain `0.10`, minimum final mean IoU `0.20`, and maximum
  full-screen fraction `0.10`;
- zero test rows read.

V2.4 adds one collapse-specific gate: every final public and internal predicted
width/height must be at least `1e-6`. It does **not** lower either IoU threshold.

## Execution order and stopping rules

1. Run `STAGE = 'smoke'`. It must prove shapes, finite loss/backward, nonzero
   gradients/updates, and checkpoint-independent forward behavior.
2. Review the fixed montage. Then set `STAGE = 'bbox_overfit'` and
   `BBOX_MONTAGE_REVIEWED = True` in the setup cell.
3. The overfit report must be `PASS`, every check must be true, decoded sizes
   must stay positive, and the report must show the train-only prior plus the
   10-step optimizer trace.
4. Only then run `STAGE = 'mini'`. Mini training remains a controlled 5,000-row,
   five-epoch validation experiment and must pass its existing quality gates.
5. If the overfit probe fails, stop. Do not tune thresholds or launch mini/full
   training. Diagnose the saved v2.4 report first.

## Interpretation limit

Passing the 32-row probe proves that the localization branch can memorize a tiny
audited training set without collapsing. It does not establish generalization or
a paper result. Those claims require the subsequent validation-only mini run and
later held-out evaluation under the project protocol.
