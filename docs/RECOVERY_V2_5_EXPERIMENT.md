# Recovery v2.5: supervised patch-attention experiment

## Purpose

Recovery v2.5 is a single controlled correction to the bbox grounding path. The
v2.4 micro-overfit run fixed the earlier width/height collapse and passed the
registered IoU-gain, gradient, finite-value, nonconstant-coordinate, and minimum
size gates. It still failed the unchanged final mean-IoU gate: `0.1291 < 0.20`.
Its attention entropy fell from `5.5185` to `1.1042`, but the predicted vertical
centre stayed near the page top. The remaining defect is therefore sharp but
incorrect patch selection, not raw bbox geometry or size decoding.

## Registered change

V2.5 adds one auxiliary loss to valid bbox rows:

`KL(target patch distribution || predicted grounding attention) / log(valid patches)`

The target distribution is a masked Gaussian over Qwen visual-patch centres. It
is centred on the target bbox and uses half of the bbox width and height as its
axis scales. Padding receives zero probability. Grounding-attention dropout is
set to zero so the supervised probabilities are stable; all other trunk/head
dropout remains inherited.

The public bbox continues to use v2.4's coordinate soft-argmax, train-only
geometric log-size prior, direct log-size SmoothL1 term, bounded exponential
size decoding, and GIoU term.

## Controls that must not change

- fixed seed: `42`
- bbox probe rows: `32`
- bbox probe optimizer steps: `100`
- optimizer and learning rate: inherited unchanged from v2.4
- centre/log-size multiplier: `5.0`
- GIoU multiplier: `2.0`
- minimum IoU gain: `0.10`
- minimum final mean IoU: `0.20`
- maximum full-screen prediction fraction: `0.10`
- FP32 grounding, disabled probe dropout, and finite-gradient requirement
- train and validation selection, recovery design, and all non-bbox objectives
- no access to `split_test.json`

The registered attention-KL multiplier is `1.0`. It is not tuned within this
run. Pass thresholds must not be reduced after seeing the result.

## Pre-action feasibility gate

Before GPU construction, the fixed 32-row probe hashes each `state_before`
image by decoded RGB pixels and dimensions, then combines that identity with the exact task description
and website domain. If an identical deterministic pre-action input has different
normalized bbox targets, the probe returns a structured failure. This prevents
an impossible memorization target from being mistaken for a model defect.

The audit does not edit or repair data. A conflict must be reviewed at source.

## Diagnostics

The report evaluates the complete fixed set at optimizer steps
`0, 25, 50, 75, 100`. Each checkpoint records:

- mean/median IoU, Recall@IoU50, and bbox MAE
- coordinate means, standard deviations, minima, and maxima
- attention KL and attention entropy
- internal centre/size and raw log-size distributions
- every row's prediction, target, IoU, centre error, attention peak, and entropy
- the ten worst final rows

These are diagnostic measurements only. The registered optimizer still performs
exactly 100 updates.

## Kaggle execution order

Use only `notebooks/kaggle_gold_recovery_v2_5.ipynb` with the Web-Gold-40K
dataset attached at `/kaggle/input/datasets/kiyasmahmud/web-gold-40k`.
Restart the kernel before every stage.

1. `STAGE = 'audit'`, `BBOX_MONTAGE_REVIEWED = False`
2. `STAGE = 'smoke'`, `BBOX_MONTAGE_REVIEWED = False`
3. Review all 32 green montage boxes manually.
4. `STAGE = 'bbox_overfit'`, `BBOX_MONTAGE_REVIEWED = True`
5. Only after a PASS: `STAGE = 'diagnostic'`
6. Only after a PASS: `STAGE = 'mini'` for the controlled five-epoch run

Any failed assertion is an intentional stop gate. Preserve the environment,
audit, stage report, montage, CSV, diagnostics, and checkpoint artifacts.

## Decision rule

V2.5 proceeds to the one-epoch diagnostic only if every existing v2.4 bbox gate
passes and attention KL decreases. A passing diagnostic may proceed to the
five-epoch mini. A failure is analyzed from the fixed-step and per-row evidence;
it is not answered by recollecting all data, lowering thresholds, or launching
full training.
