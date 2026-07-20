# Recovery v2.2 — bbox-collapse correction

## Evidence that authorizes this change

The recovery-v2.1 one-epoch diagnostic predicted nearly the full image for all 194 valid
validation boxes:

- mean prediction: `x=0.00014`, `y≈0`, `width=0.99986`, `height=0.999998`;
- mean IoU `0.0079`, median IoU `0.0052`, Recall@IoU50 `0`;
- bbox gradient norm `0.0611`, bbox update norm `0.2214`, grounding-adapter update norm
  `2.4073`, and uncertainty multiplier `1.082`.

The bbox branch was therefore active and adequately weighted. More epochs, a larger generic
coefficient, or another arbitrary learning rate is not the registered response. The failure is a
geometry/objective collapse, not evidence that the entire 39k dataset needs recollection.

## Registered changes from v2.1

1. Audit train and validation boxes against each actual `state_before` image. A box must have
   finite nonnegative origin, positive size, and satisfy `x + width <= image_width` and
   `y + height <= image_height`. Invalid examples are reported by record ID and are never silently
   clipped or repaired. Test labels remain unopened.
2. Predict normalized centre-format `cx, cy, width, height` internally. Decode to bounded top-left
   `xywh` only for the existing model-output and metric contract.
3. Replace equal mean-SmoothL1/GIoU with separately logged DETR-style coordinate L1 and GIoU:
   `5 * L1(cxcywh) + 2 * GIoU(xyxy)`. The outer multitask bbox coefficient and learned uncertainty
   weighting do not change.
4. Add a bbox-only 32-row/100-step micro-overfit gate. It freezes the VLM and unrelated task heads,
   trains only the shared projection, bbox grounding adapter/attention/trunk/output, and avoids the
   post-action/recovery streams.

The official DETR implementation is the source for centre-format prediction, coordinate L1 plus
GIoU, and the 5:2 coefficients:

- https://github.com/facebookresearch/detr/blob/main/models/detr.py
- https://github.com/facebookresearch/detr/blob/main/main.py
- GIoU: https://arxiv.org/abs/1902.09630

## Fixed factors

- dataset and domain-held-out splits;
- 5,000 training and fixed 500-row validation selection;
- seed 42, Qwen2-VL-2B backbone, QLoRA setup, physical/effective batches;
- learning rates, outer multitask coefficients, recovery-v2.1 heads and balancing;
- validation selection metric and all recovery/outcome/action gates.

## Required Kaggle order

Use `notebooks/kaggle_gold_recovery_v2_2.ipynb` and restart the kernel between stages:

1. `audit`: full train/validation geometry must pass. If it fails, correct only the reported source
   annotations and rerun; do not train through invalid boxes. Keep every dataset row: reconstruct
   the bbox from the actual image/replay when possible, otherwise set only that bbox to `null` so the
   row remains available to the non-localization heads.
2. `smoke`: the normal 16-row multimodal engineering smoke must pass.
3. `bbox_overfit`: registered checks require decreasing loss, mean-IoU gain >= 0.10, final training
   mean IoU >= 0.20, full-screen fraction <= 0.10, nonconstant predictions, and nonzero gradient/
   update for both bbox output and grounding adapter. These are engineering memorization thresholds,
   not paper-quality claims.
4. `diagnostic`: one 5k/500 epoch. It must pass mean IoU >= 0.05, Recall@IoU50 >= 0.01,
   hierarchical-recovery lift, transition-outcome improvement, and calibration. Outcome/action
   retention remains visible but is judged after the complete mini because the diagnostic has only
   one epoch.
5. `mini`: five epochs only after all previous stages pass.

Full/headline training remains blocked by the existing human-review, full-image-hash, missing-class,
and controlled-mini quality gates.
