# Qwen2-VL-2B Gold v2.8 — Full Run, Seed 42 (DGX) — Results

Status: **PASS** — `FULL CANDIDATE PASSED`
Run directory (now tracked through Git LFS): `webagent_comparison/outputs/model_comparison/qwen2vl_2b_gold_v2_8_dgx/seed_42/full/`
Evidence in this repo: `results/qwen2vl_2b_gold_v2_8_dgx/seed_42/full/` (`diagnostics.json`, `report.json`, `source_validation.csv`, `epoch_metrics.csv`)

This is the first completed full-data PC-01 validation candidate: all 24,107 combined Gold train rows,
10 epochs, checkpoint selected only on the 7,861-row original-Gold validation split, zero
locked-test rows read. These are component/selection results, not Table 2 browser results.

## 1. Run configuration

| | |
|---|---|
| Config | `configs/backbones/qwen2vl_2b_gold_v2_8_dgx.yaml` |
| Backbone | Qwen2-VL-2B-Instruct, 4-bit QLoRA, pixel bounds 50,176–200,704 |
| Seed | 42 |
| Train rows | 24,107 (23,499 original Gold + 608 retry/abort supplement) |
| Validation rows (checkpoint selection) | 7,861, `original_gold` only |
| Held-out supplement validation (not used for selection) | 194 rows, `retry_abort_supplement_v2` |
| Test rows read | 0 (locked-test isolation held) |
| Epochs requested / completed | 10 / 10, early stopping **not** triggered |
| Batch | physical 16, grad-accum 2 (effective 32) |
| Mixed precision | fp16 (GB10 supports bf16 but fp16 was kept to stay comparable with the earlier accepted mini-run gate thresholds) |
| Peak GPU memory | 10.18 GB (of 130.7 GB unified memory on the DGX Spark GB10) |
| Checkpoint round-trip / loss-decrease sanity checks | both PASS |
| Config hash | `d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f` |

## 2. Selected checkpoint

Selection rule: `all_gates_then_outcome_mcc` — every quality gate must pass, then rank eligible
epochs by `outcome_mcc`. All 10 epochs turned out gate-eligible; **epoch 6** ranked highest.

| Metric | Value |
|---|---|
| Selected epoch | 6 of 9 (0-indexed) |
| `outcome_mcc` (PC-01 validation-selection metric) | **0.6242** |
| Checkpoint | `best_e6_outcome-mcc0.624.ckpt`, SHA-256 `9eaab6d2…c94895a` |

### vs. the prior accepted checkpoint (v14, 5k-subset mini, epoch 3)

| Metric | v14 (reference) | v2.8 full (epoch 6) | Change |
|---|---|---|---|
| `outcome_mcc` | 0.5217 | **0.6242** | +0.1025 |
| `action_acc` | 0.396 | 0.4217 | +0.026 |
| `recovery_outcome_mcc` (causal transitions) | 0.1864 | **0.7955** | +0.609 |
| `bbox_mae` | 0.1863 | 0.0915 | −0.095 (better) |
| `outcome_ece` | 0.1853 | 0.1626 | −0.023 (better) |

All 8 predeclared quality gates passed at epoch 6 (see §4). The full-data run is a clear
improvement over v14 on every gated axis, not just a marginal pass — the recovery-outcome jump
in particular reflects the earlier fix to feed the causal recovery-transition head only proper
`(failure → recovery action → post-recovery state)` triples instead of leaking future state.

## 3. PC-01 validation metrics at the selected checkpoint (epoch 6, original-Gold validation, n=7,861)

| Head | Metric | Value | Majority baseline |
|---|---|---|---|
| **Outcome** (primary) | MCC | 0.6242 | — |
| | Balanced accuracy | 0.8041 | 0.5838 (acc) |
| | Failure-F1 | 0.8519 | — |
| | Failure macro-F1 | 0.8094 | 0.3686 |
| | Success recall | 0.7158 | — |
| | Brier | 0.1685 | — |
| | ECE | 0.1626 | — |
| **Failure type** | Accuracy / macro-F1 | 0.5812 / 0.5022 | 0.4162 / 0.1470 |
| | MCC | 0.4536 | — |
| **Action** (6-way) | Accuracy / macro-F1 | 0.4217 / 0.3267 | 0.1907 / 0.0534 |
| | MCC | 0.3412 | — |
| **Needs-recovery** (binary gate) | Macro-F1 | 0.5768 | 0.4330 |
| | MCC | 0.3027 | — |
| **Recovery strategy** (attempted-only, n=1,858) | Accuracy / macro-F1 | 0.6604 / 0.4687 | 0.5382 / 0.2333 |
| **Recovery outcome** (causal transitions, n=1,858) | Accuracy / macro-F1 | 0.9085 / 0.8977 | 0.6663 / 0.3999 |
| | MCC | 0.7955 | — |
| **Memory** | Accuracy / macro-F1 | 0.8428 / 0.8303 | 0.6199 / 0.3827 |
| | MCC | 0.6625 | — |
| **Confidence calibration** | MAE | 0.0437 | — |
| | Success-ECE | 0.2338 | — |
| **Bbox** (n=2,349 valid boxes) | MAE | 0.0915 | — |
| | Mean IoU | 0.1010 | — |
| | Median IoU | 0.0000 | — |
| | Recall@IoU50 | 0.0809 | — |

Every head beats its majority-class baseline with real margin. The two weakest heads are
**action** (macro-F1 0.33 on a 6-way task — real signal, but far from saturated) and **bbox
localization** (mean IoU 0.10, median IoU 0 — passes the minimum-functionality gate but grounding
is still weak; see §6).

## 4. Quality gates (all PASS at epoch 6)

| Gate | Result |
|---|---|
| `outcome_mcc` within 0.03 of v14 | PASS (actually +0.10 above) |
| Action accuracy not down >0.03 vs v14 | PASS |
| `needs_recovery` macro-F1 beats majority by ≥0.03 | PASS |
| Attempted-strategy macro-F1 beats majority by ≥0.03 | PASS |
| Transition recovery-outcome MCC improves v14 by ≥0.01 | PASS |
| Bbox mean IoU ≥ 0.05 | PASS |
| Bbox Recall@IoU50 ≥ 0.01 | PASS |
| Outcome ECE not up >0.03 vs v14 | PASS |

## 5. Per-epoch trend (original-Gold validation)

| Epoch | outcome_mcc | failure_macro_f1 | failtype_macro_f1 | action_acc | needs_recovery_f1 | recovery_outcome_mcc | memory_mcc | bbox_mean_iou | outcome_ece |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5899 | 0.7916 | 0.4485 | 0.3854 | 0.5660 | 0.8142 | 0.6639 | 0.1115 | 0.0362 |
| 1 | 0.6009 | 0.7917 | 0.5438 | 0.3741 | 0.5149 | 0.8233 | 0.6482 | 0.0982 | 0.1206 |
| 2 | 0.6161 | 0.8033 | 0.4165 | 0.3802 | 0.5416 | 0.7726 | 0.6187 | 0.1018 | 0.1398 |
| 3 | 0.6116 | 0.7999 | 0.5352 | 0.3894 | 0.5565 | 0.7975 | 0.6375 | 0.0724 | 0.1595 |
| 4 | 0.6175 | 0.8072 | 0.5117 | 0.4153 | 0.5544 | 0.8068 | 0.6255 | **0.1182** | 0.1438 |
| 5 | 0.6032 | 0.7923 | 0.5315 | 0.4231 | 0.5427 | 0.7967 | **0.6701** | 0.1020 | 0.1673 |
| **6*** | **0.6242** | 0.8094 | 0.5022 | 0.4217 | 0.5768 | 0.7955 | 0.6625 | 0.1010 | 0.1626 |
| 7 | 0.6162 | 0.8027 | **0.5713** | 0.3902 | 0.5853 | 0.8041 | 0.6515 | 0.1030 | 0.1725 |
| 8 | 0.6136 | 0.8036 | 0.5618 | 0.3913 | 0.5927 | 0.8068 | 0.6630 | 0.1037 | 0.1748 |
| 9 | 0.6158 | 0.8056 | 0.5628 | **0.4160**† | 0.5975 | 0.8043 | 0.6507 | 0.1030 | 0.1751 |

\* selected checkpoint. † action_acc actually peaks at epoch 5 (0.4231); shown for the last epoch
for reference. Bold marks each column's per-epoch best, independent of selection.

Reading the trend: `outcome_mcc` rises fairly monotonically through epoch 6 then plateaus/wobbles
(0.616→0.614→0.616) — early-stopping never triggered because none of those later epochs actually
regressed past the patience window, but there is no further real gain after epoch 6, so this
looks like a converged plateau rather than an under-trained run. `outcome_ece` **worsens** steadily
after epoch 0 (0.036 → 0.175) even as MCC improves — the model becomes a better classifier but a
worse-calibrated one; ECE was still inside the "don't-regress-more-than-0.03-vs-v14" gate only
because v14's own ECE (0.185) was already mediocre. `strategy_attempted_macro_f1` peaks early
(epoch 1, 0.616) and *degrades* by epoch 9 (0.354) — the recovery-strategy head overfits/drifts
while other heads keep improving, which is why the all-gates-then-MCC selection rule (not
"just pick the last epoch") matters here.

## 6. Known weaknesses / honest caveats

- **Bbox geometry data quality**: 21.8% of all non-null boxes in the combined corpus are invalid
  under the strict contract (finite x/y, positive width/height, box fully inside the actual
  image) — 2,042/8,761 train, 492/2,841 validation. Dominant reasons are
  `bottom_boundary_overflow` (1,835 train / 393 val) and `y_origin_outside` (1,755 train / 381
  val). These rows are **masked out of bbox loss/eval**, not corrected or dropped from the row —
  the row still contributes to every other head. This caps bbox training/eval signal at ~6.7k
  train / ~2.3k val boxes instead of the full non-null counts.
- **Bbox localization remains weak in absolute terms**: mean IoU 0.10, median IoU 0.0,
  Recall@IoU50 0.08. This clears the predeclared *minimum-functionality* gates (≥0.05 mean IoU,
  ≥0.01 Recall@IoU50) but is not remotely close to a usable grounding model — most predicted boxes
  simply don't overlap the target well enough to count at IoU 0.5.
- **Action head is the weakest multi-way classifier**: macro-F1 0.327 / accuracy 0.422 on 6
  balanced-ish classes, versus a 0.19 majority baseline. Real signal, but well short of the
  ~0.96 action accuracy seen on the older, leakier v12/synthetic runs — consistent with the
  causal fix that removed `state_after` (and the text-shortcut wording) from the pre-action
  stream. This is the expected/intended cost of closing that leak, not a new regression.
- **Calibration degrades with training**: `outcome_ece` moves from 0.036 (epoch 0, effectively
  untrained) to ~0.16–0.18 by epoch 6–9; `confidence_success_ece` is 0.234 at the selected
  checkpoint. The outcome/failure classifiers are increasingly overconfident as they improve.
- **Recovery-strategy generalization is fragile**: `strategy_attempted_macro_f1` falls from 0.616
  (epoch 1) to 0.354 (epoch 9) while the epoch-6 checkpoint sits mid-pack at 0.469. `RETRY` and
  `ABORT` have zero support in the *validation* recovery_strategy distribution (only present in
  train and the supplement set — see `missing_claimed_classes` in `recovery_class_audit`), so the
  4-way validation strategy metric never actually exercises those two classes.

## 7. Held-out supplement check (`retry_abort_supplement_v2`, n=194)

This 194-row set (RETRY/ABORT-only, not used for checkpoint selection) is a small extra
generalization probe for the two strategy classes the main validation split under-represents.

| Metric | Value | Majority baseline |
|---|---|---|
| `strategy_attempted_acc` / macro-F1 | 0.6546 / 0.6505 | 0.5258 / 0.3446 |
| `recovery_outcome_acc` / macro-F1 | 0.9124 / 0.9118 | 0.5206 / 0.3424 |
| `recovery_outcome_mcc` | 0.8261 | — |

RETRY vs ABORT: precision/recall of 0.75/0.52 (F1 0.61) on RETRY and 0.60/0.80 (F1 0.69) on
ABORT — the model over-predicts ABORT (123 predicted vs. 92 true) and under-predicts RETRY (71 vs
102 true), but both classes are learned well above chance. `recovery_outcome` on this held-out set
(MCC 0.826) is consistent with — even slightly better than — the main validation figure (0.796),
which is a good sign for that head's generalization.

## 8. Data distributions (for reference)

Train (24,107 rows) failure_type: NONE 10,310 (42.8%) / ACTION_MISMATCH 6,796 (28.2%) /
PERCEPTION_ERROR 6,315 (26.2%) / LOOP_DETECTED 686 (2.8%). Action types are close to balanced
(3,677–4,770 per class across 6 classes). `recovery_attempted` True for 6,126/24,107 (25.4%);
of those, `recovery_success` True for 2,228 / False 3,898 (36.4% success rate on attempted
recoveries). Validation (7,861 rows) mirrors these ratios closely (see `diagnostics.json:
train_distribution` / `validation_distribution` for exact validation-side counts).

## 9. Artifacts

- **In this repo** (`results/qwen2vl_2b_gold_v2_8_dgx/seed_42/full/`): `diagnostics.json` (full
  per-epoch diagnostics, gates, distributions, bbox/recovery audits), `report.json` (superset,
  includes resume/checkpoint-lineage audit and per-epoch history), `epoch_metrics.csv` (10 rows,
  one per epoch, ~140 columns), `source_validation.csv` (per-source-split summary row).
- **Tracked through Git LFS** (`webagent_comparison/outputs/model_comparison/qwen2vl_2b_gold_v2_8_dgx/seed_42/full/checkpoints/`):
  11 checkpoints (`best_e0`…`best_e9` + `last.ckpt`), ~278 MB each, ~3 GB total. Their Git-LFS
  storage status does not replace the authenticated runtime-export, processor, or DGX
  checkpoint-compatibility gates required for Table 2.

## 10. Bottom line

The full-data Qwen2-VL-2B Gold v2.8 run (seed 42) is a genuine, gate-verified improvement over the
prior accepted checkpoint on every predeclared axis, with the causal recovery-outcome fix being
the largest single jump (MCC 0.186 → 0.796). It is a strong PC-01 validation candidate for this
backbone/seed, not a Table 2 result or final promoted backbone. It is **not** yet a finished
evaluation: no locked-test-split number exists for it
(intentionally — test isolation was preserved), bbox grounding and calibration are still weak in
absolute terms, and this is a single seed on one backbone. The research-locked Table 2 design
intentionally adds no PC-01 seeds 43--44, so model-seed uncertainty will remain unmeasured and
must be reported. PC-02/PC-03 validation-only comparison, final model promotion, browser-time
evaluation, and supporting diagnostics remain outstanding per `project_progress.md`.
