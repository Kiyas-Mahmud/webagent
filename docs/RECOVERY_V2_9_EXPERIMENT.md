# Recovery v2.9 — corrected training recipe

Status: **registered, not yet run.**
Config: `configs/backbones/qwen25vl_7b_gold_v2_9.yaml`
Entry point: `scripts/run_gold_v2_9.py`
Backbone, data, causal routing, heads, selection rule and effective batch are
inherited unchanged from v2.8.

## Why

Three defects measured on the v2.8 runs. None are speculative; each cites the
evidence it was derived from.

### 1. The reported training loss was unbounded below

`src/web_agent/models/loss.py:511` aggregates as `exp(-log_var) * L + log_var`.
The `+ log_var` term has no lower bound, and the nine `log_var` parameters sit
in the *heads* AdamW group at `lr 1e-3` with no clamp.

Measured on the **completed** 2B v2.8 seed-42 full run:

| epoch | 0 | 3 | 6 | 9 |
|---|---|---|---|---|
| `train_loss` | +0.566 | −9.971 | −15.690 | **−17.458** |
| `outcome_mcc` | 0.590 | 0.612 | 0.624 | 0.616 |
| `log_var_outcome` | −0.314 | −1.821 | −2.684 | −2.844 |

The loss falls monotonically by 18 points while the metric is flat. It is not a
progress signal.

The weighting was also **not performing its function**. Mid-epoch-2 on the 7B,
all nine `log_vars` lay inside a **0.29** band while their Kendall stationary
points `log(L_weighted)` span **5.85** (from −6.94 for `confidence` to −1.09 for
`bbox`). Because AdamW normalises per-parameter gradient magnitude, every
`log_var` descended at ≈`lr` per step regardless of its own task loss — a
lockstep ramp, not per-task weighting. Its only other effect was `exp(-log_var)`
acting as a uniform, growing global gradient multiplier (1.00 → 1.28 → 1.97 →
2.57), so the *effective* step size rose while the cosine schedule believed it
was decaying.

**Fix.** `loss.dynamic_weighting: fixed`, a branch `CombinedLoss` already
implements. The total becomes a sum of non-negative weighted terms and is
bounded below by zero. No new loss code, which matters for a run with no mini
gate. Removing a component that demonstrably did not work is not a capability
loss; restoring it correctly (e.g. a `softplus` regulariser with a `log_var`
clamp and its own low-LR parameter group) is recorded as future work.

### 2. Early stopping could fire off a warmup epoch

`warmup = int(steps_per_epoch × epochs × warmup_ratio) = int(754 × 10 × 0.1) =
754` — **exactly 1.00 epochs**. Epoch 0 therefore never reached peak LR, yet its
score set `best_metric`. On the 7B v2.8 run, epoch 0 (`outcome_mcc` 0.6783) is
still the selected checkpoint, and with `bad_epochs` already at 1 the run could
stop at epoch 3 and ship a warmup-only model.

A second, latent defect: `configs/base.yaml:43` sets
`optim.early_stopping_patience`, but `trainer.py:367` reads
`train.early_stop_patience` — different section, different name. The value is
never read, yet `optim` is inside the resume signature, so editing it breaks
resume and changes no behaviour.

**Fix.** `train.warmup_steps: 190` (≈0.25 epoch, absolute); `train.min_epochs: 3`
so early stopping cannot arm before epoch 3; `TrainerV29` fails closed if
`optim.early_stopping_patience` carries a real value.

### 3. Grounding is vertically under-resolved

At `max_pixels: 200704`, a 1280×720 screenshot (all screenshots in the dataset
are exactly this size) resizes to 588×336 → a **21×12** token grid, ≈61×60
source px per cell. Measured over the 2,349 clean validation boxes:

| axis | precision needed | grid cell | verdict |
|---|---|---|---|
| horizontal | ±80 px | 61 px | adequate |
| vertical | **±18 px** | 60 px | **3.3× too coarse** |

Raising resolution does not close it — at `max_pixels 802816` (**4× compute**)
the vertical cell is still 30 px, 1.7× too coarse.

The head is **not broken**. Against a constant mean-box predictor it is
decisively better (mean IoU 0.160 vs 0.015, recall@IoU50 0.086 vs 0.000), so the
`coordinate_softargmax` architecture is sound — it is resolution-starved on the
one axis that decides whether a click lands.

**Fix.** (a) A zero-initialised sub-cell centre offset, bounded by `tanh` to
±0.10 (one vertical grid cell is ≈0.083). Soft-argmax interpolates between cell
centres but is unbiased only when the attention mass is symmetric about the true
point; this learns the correction. (b) Shift the box loss toward GIoU
(`l1 5.0→2.0`, `giou 2.0→5.0`): plain L1 weights a 20 px error on a 160 px-wide
box the same as on a 36 px-tall box, so the gradient was dominated by the axis
already solved.

**Not included:** the anisotropic pixel budget (pre-resizing 1280×720 → 640×720
to buy vertical resolution at identical token count). It required two mini arms
to choose between and the mini stage was skipped for time. Future work.

## Registered protocol deviation — the mini stage is skipped

At the throughput measured on the GB10 (**~14 s per physical batch**, bounded
9–16 s):

| | cost | validates on |
|---|---|---|
| controlled mini (5k × 5) | 24.3 h | 500 rows |
| **full run epoch 0** | 23.4 h | **7,861 rows** |

Epoch 0 costs the same as the entire mini and validates on 15× the rows, on the
real distribution. The mini is therefore skipped and **epoch 0 is the gate**.
This follows the precedent set on 2026-08-01, where a time-limited v2.8 mini was
accepted with an explicit recorded deviation.

The 16-row compatibility smoke still runs first — but it must be launched as
`scripts/run_gold_v2_9.py --stage smoke`, **not** `scripts/run_gold.py --stage
smoke`. The v2.8 script would verify the v2.9 *config* while forwarding and
backwarding through the v2.8 *action head*, so the only cheap pre-flight before
a multi-day run would test nothing new. (`build_gold_components` is defined in
`gold_stages` and imported into `gold_full` — two independent name bindings, so
`v2_9_bindings` patches both.) It is fail-closed and costs minutes.

### Epoch-0 abort gate — fixed before the run

Evaluate with `python scripts/run_gold_v2_9.py --check-epoch0 <metrics.csv>`.
**Any failure means stop the run and diagnose.**

| check | threshold | source |
|---|---|---|
| `train_loss` ≥ 0 | hard | the point of fix 1 |
| all loss terms finite | hard | standard |
| `outcome_mcc` | ≥ **0.55** | v2.8 epoch 0: 0.678 (7B), 0.590 (2B) |
| `bbox_mean_iou` | ≥ **0.155** | v2.8 7B epoch 0 |
| `bbox_recall_iou50` | ≥ **0.083** | v2.8 7B epoch 0 |

The two bbox floors are the real test of fix 3a. Because the centre offset is
zero-initialised, v2.9 begins numerically identical to v2.8 — proven by
`tests/test_v2_9.py::test_offset_head_is_identity_at_initialisation`. A
regression below these floors therefore means the head swap is wired wrong, not
that the idea failed.

## Comparability

**v2.9 results are not comparable to v2.8 results.** The loss changed. Report
v2.9 as a separate corrected recipe alongside the v2.8 three-model comparison —
do not substitute it into that table. Re-running all three backbones under v2.9
would restore comparability at roughly 3× the cost and is not currently planned.

## Cost

At ~14 s/batch: **~23 h per epoch**, ~9.8 days for the full 10 epochs. Early
stopping (armed from epoch 3) will likely end it sooner — the 2B v2.8 run peaked
at epoch 6. `checkpoint_every_steps` is lowered 50 → 10, cutting crash exposure
from ~1.5 h to ~18 min.

## Implementation — additive only

Every file the running v2.8 job has loaded is byte-identical; this is asserted by
`tests/test_v2_9.py::test_v2_8_files_are_untouched_by_v2_9`.

| new file | role |
|---|---|
| `src/web_agent/models/heads_v2_9.py` | `ActionHeadV29` — zero-init centre offset |
| `src/web_agent/train/trainer_v2_9.py` | `TrainerV29` — warmup, min-epochs, dead-key guard, loss-sign check |
| `src/web_agent/train/gold_full_v2_9.py` | scoped injection of the two components |
| `configs/backbones/qwen25vl_7b_gold_v2_9.yaml` | the complete diff from v2.8 |
| `scripts/run_gold_v2_9.py` | entry point + `--check-epoch0` |
| `tests/test_v2_9.py` | 23 tests |

`gold_full_v2_9` rebinds two names inside the `gold_full` module namespace for
the duration of one call rather than forking `run_gold_full`. This is deliberate:
the eight registered quality gates, resume validation, split hashing, checkpoint
round-trip and supplement reporting must stay bit-identical to v2.8, and two
copies of that logic could silently drift — a reviewer could then no longer tell
whether a v2.9 result passed the same gates. The rebinding is scoped by a context
manager, asserted on entry and exit, and covered by two tests including the
exception path.
