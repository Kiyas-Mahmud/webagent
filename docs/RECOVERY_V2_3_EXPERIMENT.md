# Recovery v2.3: coordinate-aware bbox grounding

## Why this experiment exists

Recovery v2.2 proved that the bbox branch updates and substantially reduces
coordinate error, but it did **not** pass the registered localization gate on
the fixed 32-row probe. Its exact result was:

- mean IoU: `0.00406 -> 0.02465` (`+0.02059`, required `+0.10`)
- final mean IoU: `0.02465` (required `0.20`)
- bbox MAE: `0.31535 -> 0.11328`
- final predicted y mean/std: `0.03018 / 0.00391`, showing top-edge collapse
- first bbox and grounding gradient norms: non-finite
- bbox and grounding parameter updates: non-zero

This evidence rules out “the head never trained.” It instead identifies a
spatial-representation problem plus an unreliable first-gradient measurement
under scaled FP16. Lowering the IoU thresholds would hide those failures and is
not permitted.

## Registered v2.3 change

V2.3 changes only the localization mechanism and its numerical execution:

1. Derive normalized `(x, y)` patch centres from Qwen `image_grid_thw` after
   spatial merging, aligned exactly with the preserved image-token positions.
2. Add those coordinates to the attention keys/values.
3. Predict bbox centre with a differentiable attention soft-argmax; predict
   width and height with the existing bbox MLP.
4. Keep the frozen/quantized VLM in FP16 and run the trainable grounding
   adapter, attention, coordinate projection, bbox branch, and bbox loss in
   FP32.
5. Disable dropout only during the engineering micro-overfit test.
6. Report the first accepted finite gradients, total attempted steps, rejected
   non-finite steps, and attention entropy.

The public bbox remains normalized top-left `xywh`. No label is added to model
input, no invalid annotation is repaired, and `split_test.json` remains unopened.

## Fixed controls

The following stay identical to v2.2:

- the audited dataset and invalid-target masking policy
- the exact deterministic 32 training records selected with seed `42`
- 100 accepted optimizer steps
- AdamW learning rates and weight decay
- centre L1 ratio `5.0` and GIoU ratio `2.0`
- minimum mean-IoU gain `0.10`
- minimum final mean IoU `0.20`
- maximum full-screen prediction fraction `0.10`
- backbone, causal streams, recovery heads, task adapters, and validation split

## Required execution order

Use `notebooks/kaggle_gold_recovery_v2_3.ipynb` only. Restart the Kaggle kernel
between stages.

1. `audit`: accept only the existing target-level masking disposition; fatal
   image/file errors still block execution.
2. `smoke`: verify output shapes, finite backward, and non-zero updates for the
   bbox, coordinate projection, grounding adapter, and all recovery probes.
3. Inspect the generated montage of the exact 32 micro-overfit rows. Confirm
   each green rectangle marks the intended interactive target, then set
   `BBOX_MONTAGE_REVIEWED = True`.
4. `bbox_overfit`: require every registered check, including zero rejected
   non-finite gradient steps.
5. `diagnostic`: run one controlled 5k/500 epoch only after overfit passes.
6. `mini`: run five epochs only after all diagnostic functionality gates pass.

## Decision rule

A v2.3 pass is not guaranteed in advance. Local tests can prove tensor/grid
alignment, differentiability, config isolation, and notebook correctness; only
the Kaggle GPU run can establish the empirical IoU result. If the unchanged
micro-overfit thresholds fail again, stop before the 5k diagnostic and inspect
the saved attention entropy, coordinate distribution, montage, gradients, and
updates. Do not tune the pass thresholds after seeing the result.
