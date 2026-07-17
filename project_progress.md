# Project Progress Tracker

> **READ THIS FIRST** before planning or writing any code. Update it after every task.
> Companion to `docs/` (the specs) — this file is the live build log.

Project: Failure-Aware Resilient Autonomous Web Agent (MSc thesis).
Repo: https://github.com/Kiyas-Mahmud/webagent (branch `Code`).
Dataset: Kaggle `thesisdata` → `/kaggle/input/datasets/kiyasmahmud/thesisdata/FinalData`
(70,965 labeled steps; train 38,875 / val 16,070 / test 16,020).

## Workflow (how work flows)
- Author code in IDE (`e:/University/thesis/code`) → push to GitHub `Code`.
- Kaggle notebook `git clone`s the package + `sys.path` import (NOT `pip install -e .`
  — PEP 660 editable finder shadows sub-packages).
- Train/verify on **Kaggle T4** (free). Pull results back to IDE.
- Hybrid layout (plan C): stable core in `src/web_agent/`, the training loop + analysis
  live in the notebook so progress is visible.
- Commits: plain messages, **no Claude/AI mention**.

## Key decisions
- **Model = Qwen2-VL-2B-Instruct** (4-bit QLoRA). The spec's "Qwen2.5-VL-0.5B" does
  NOT exist (text-only Qwen2.5-0.5B, can't see images); 2B is the real smallest VLM
  (hidden D=1536). QLoRA for all VLM models for consistency.
- **VLM adapter path** (no cross-attention): VLM(image+text) → mean-pool → Adapter
  Linear(1536→768)+LN+Dropout → 5 heads. Backbone-agnostic (heads + loss never change).
- **Compute**: free Kaggle T4. Full 38,875-row QLoRA is too slow on T4 (~24h/epoch),
  so train on an 8k subsample now; full-data/headline number later on paid A100 (~$20).
  Whole-suite cost estimate: ~$60-160 smart-split (free small models, paid VLMs).
- T4 has **no bf16** → fp16 everywhere.

## Status by phase

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Setup | ✅ done | package scaffold, YAML config (deep-merge), labels.py, seed, checkpoint, logging |
| 1 Dataset verification | ✅ done + PASSED on Kaggle | all checks pass; 71.9% FAILURE, bbox 67,160, 0 leakage |
| 2 DataLoader | ✅ done + verified | WebAgentDataset, 5 mode filters, stratified subsample, VLM joint-processing branch + vlm_collate |
| 3 Model core (smoke+mini) | ✅ done + PASSED | adapter+heads learn; loss drops; checkpoint round-trips |
| 5 Full QLoRA training | 🟡 BUILT, running on Kaggle | 5 heads, 10-term loss, Trainer, eval, memory index — see below |
| 6 Memory index (Pillar 4) | ✅ built (cosine top-3) | retrieval used at eval |
| 7 Ablations | ⬜ not started | A1-A5 on chosen primary |
| 8 Other backbones | ⬜ not started | SigLIP/CLIP/Florence (free T4); InternVL2/Qwen2.5-VL-3B (paid) |
| 9 Baselines | ⬜ not started | B1 random, B2 MindAct, B3 CLIP+MLP, B4 LayoutLMv3 |
| 10 Evaluation/paper | ⬜ not started | 3-split tables, significance |

## Phase 5 — what's built (full QLoRA training)
- Config `configs/backbones/qwen2vl_2b.yaml`: 4-bit nf4, LoRA r8 q/v_proj, 10-term
  loss weights, 2 LR groups (lora 1e-4 / heads 1e-3), early-stop on val Failure-F1.
- `models/encoders/vlm.py`: peft LoRA via `prepare_model_for_kbit_training` +
  `get_peft_model`; grad checkpointing `use_reentrant=False`; mean-pool embedding.
- `models/adapter.py`: Linear(1536→768)+LayerNorm+Dropout(0.1).
- `models/heads.py`: 5 heads — Failure, Action, Memory, **RecoveryOutcome**; BCE heads
  emit LOGITS (BCE-with-logits, autocast-safe). `model.py` returns `fused` (contrastive).
- `data/dataset.py`: multi-step context (last 3 trajectory steps, grouped by
  original_task_id+pass); emits recovery_success + original_task_id; passes
  `mm_token_type_ids` (newer Qwen2-VL M-RoPE requirement).
- `data/dataloader.py`: `TaskPairBatchSampler` (co-batches a task's passes so
  contrastive pairs form); vlm_collate carries task_id + recovery_success + mm_token_type_ids.
- `models/loss.py`: 10-term loss — label smoothing 0.1, sklearn balanced weights,
  conf clip [0.05,0.95], differentiable soft-bin calibration surrogate, margin-0.5
  contrastive, recovery_outcome BCE.
- `train/trainer.py`: 2 param groups, cosine+warmup, fp16 GradScaler, grad accum,
  grad clip, **per-step loss+ETA logging**, early stop patience 3 on val Failure-F1,
  top-3 ckpts by F1, resume `last.ckpt` every 500 steps, metrics CSV.
- `eval/metrics.py`: Failure-F1 (primary), accuracies, true ECE, bbox MAE.
- `eval/evaluate.py`: per-split (test_task/website/domain) eval.
- `memory/index.py`: cosine top-3 retrieval.
- Notebook cells 1-9: install → verify → batch → visual → build → smoke → mini →
  checkpoint → full train (8k subsample, ~5-6h/epoch).

## First full run (8k subsample, Qwen2-VL-2B) — RAN CLEAN, but majority collapse
- Infra perfect: 4 epochs ~78min each, per-step ETA, early stop, top-3 ckpts, round-trip,
  full 3-split eval, CSV. ~5.21GB VRAM.
- **Failure-F1 0.837 = the FAILURE-majority baseline** (71.9% FAILURE → predict-all-FAILURE
  F1 = 0.836). F1 flat across epochs/splits → model collapsed to always-FAILURE. NOT real
  failure detection.
- failtype_acc ~0.28-0.39 (≈majority), recovery_acc ~0.30 (≈majority): also collapsed.
- action_acc ~1.00 (genuinely learned / strongly text-cued). ECE ~0.095 (ok). bbox_mae ~0.20 (weak).
- ROOT CAUSE: outcome loss was NOT class-weighted (only action+failtype were). 
- FIX APPLIED (commit pending): balanced weight on the OUTCOME loss too
  (`balanced_class_weights` now returns outcome weights; `CombinedLoss` takes
  `outcome_class_weights`). Added honest metrics: failure_macro_f1, outcome_bal_acc,
  outcome_mcc, success_recall — so collapse can't hide behind F1. Notebook cells 5/6 pass ow.
  - Re-run cell 9: watch `success_recall` and `outcome_mcc` > 0. If still ~0, bump LoRA
    rank 8→16 / more epochs / more rows.

## Second run (outcome class-weighted) — collapse FLIPPED, not fixed
- After outcome balancing: `success_recall=0.96, failure_f1=0.07, outcome_mcc≈0, bal_acc≈0.50`.
  Model now predicts SUCCESS (minority) for ~everything. **MCC stayed ≈0** the whole time →
  weighting only moved the decision bias and flipped WHICH class collapses. Diagnostic proof:
  the problem is the **representation/inputs the outcome head sees, not loss weighting**.

## Representation-level fixes (this round) — code-ready, validate on Kaggle smoke
Root analysis in `~/.claude/plans/you-are-proffesional-phd-gentle-dewdrop.md`.
- **Pooling (P0-A, primary):** `encoders/vlm.py` now config-selectable `backbone.pooling`
  = `last` (default) | `mean` | `attention`. Was uniform masked-MEAN over the full
  decoder sequence — dominated by hundreds of vision-patch tokens, blurring the causal
  summary where the outcome signal lives. `last` = last non-pad token; `attention` = tiny
  learned pool. (mean kept for the ablation.)
- **state_after (P0-B):** `data.use_state_after` (default false). When true, feed
  before+after screenshots (two images) — failure is often only visible post-action.
  `dataset._vlm_inputs` takes an image list; `image_grid_thw` now `[n_img,3]`; collate
  CATs it (single-image path byte-identical).
- **Contrastive (P1-A):** `loss.contrastive_mode` = `supervised` (default, SupCon on the
  outcome label — works in any batch with both classes) | `task_pair` (old, sparse). The
  old term was ~dead (needed same-task SUCCESS+FAILURE in one micro-batch).
- **Outcome weighting (P1-B):** outcome CE label_smoothing forced to 0; outcome weights
  now `capped` (ratio ≤1.5: SUCCESS 1.5 / FAILURE 1.0) via
  `class_weights.balanced_class_weights(outcome_scheme=...)`, not raw `balanced` (2.56× over-swing).
  `sqrt` scheme also available.
- **recovery_outcome (P2-B):** BCE now uses `pos_weight` (`binary_pos_weight`, ≈2.56);
  loss weight 0.02→0.05.
- **Capacity (P2-A):** LoRA targets q/v → q,k,v,o,gate,up,down; rank 8→16 (2B), alpha 32.
- **Config completeness:** `qwen25vl_3b.yaml` got the full 10-term loss + lr_lora/heads +
  train block (it inherited base's 7-term and would have KeyError'd if run).
- **Notebook:** cell 5 computes capped outcome weights + recovery pos_weight + a
  `DIAG_OUTCOME_ONLY` toggle (zeros the other 9 terms) for the D1/D2/D3 ceiling probe.

## BREAKTHROUGH — input was the wall; before+after fixed it
- Diagnostic (outcome-only, last-pool, before-only): **MCC frozen 0.0000** across 3 epochs →
  proved the before-only screenshot can't predict outcome. Not loss/pooling.
- Audited all 23 JSON fields: only legit missing inputs were **state_after image** +
  **visual_diff_score** (spec Pillar-2 input). All other non-input fields are prediction
  targets (outcome/failure_type/action/recovery/confidence/bbox/memory) or leaks
  (reflection_text, injection_type). Verified `FinalData.zip` has NO extra text (DOM/HTML) —
  dataset is screenshot-only.
- Fed before+after images (`use_state_after`) + `visual_change` text → **outcome MCC 0 → ~0.95
  on ALL 3 test splits** (task 0.954 / website 0.96 / domain 0.95). bal_acc 0.97, success_recall 0.93.
- **CAVEAT (must resolve before claiming):** this run still fed `visual_diff_score` as text, and
  it ~= the label (0.32 vs 0.07). `loss→0.000`, ece~0.30, converged epoch 0 = likely a SCALAR
  SHORTCUT, not screenshot understanding. Gated it behind `data.use_visual_diff_text` (default
  OFF) so the honest run learns from IMAGES only.
- (This run was DIAG_OUTCOME_ONLY=True, so action/failtype/recovery heads are ~0 by design.)

## NEXT
- **Honest ablation (decisive):** rerun outcome-only with `use_visual_diff_text=false`
  (before+after images, NO scalar). MCC stays ~0.95 → vision works, publishable. MCC →0 →
  it was the scalar shortcut; revise the claim.
- Then `use_visual_diff_text=true` as the "+diff feature" ablation row (paper table).
- Then DIAG_OUTCOME_ONLY=False → full 10-term multitask; confirm the other heads recover.
- Then: pooling/input ablations, baselines (majority/random), other backbones, eval tables.
- NOTE: notebook cells do NOT update via cell-1 git pull (pull updates the package+configs only).
  Config flags (use_state_after, use_visual_diff_text) DO take effect after re-running cell 3/5.

## Gotchas learned (don't re-hit)
- JSON is UTF-8 (Windows cp1252 fails). bbox keys are `width`/`height` (not w/h).
- `.gitignore data/` once swallowed `src/web_agent/data/` — anchor ignores with `/`.
- Kaggle stale-kernel: after a package fix, re-run cell 1 (purges sys.modules) AND every
  cell that built a reused object (loader, model, loss). Or Restart & Run All.
- `pip install -e .` editable finder shadows subpackages → use sys.path.
- `F.binary_cross_entropy` unsafe under fp16 autocast → use `_with_logits` (heads emit logits).
- Qwen2-VL (newer transformers) needs `mm_token_type_ids` passed through.

## GOLD DATASET — real, leak-safe (the pivot)
- Honest images-only run on synthetic confirmed: outcome MCC=0 from images; the 0.95 was the
  `visual_diff_score` scalar shortcut. Synthetic 70k = leaked + non-visual → demoted to
  synthetic-pretrain/debug only (kept frozen per supervisor).
- Team collected a GOLD dataset: `code/data/web_agent_gold_v8_approved_2032_real_browser/`
  — 2032 rows, real Playwright-observed (url-oracle), human-reviewed, leak fields stripped,
  balanced (FAILURE 1058 / SUCCESS 974), disjoint splits 1219/406/407. Gitignored (not in repo).
- Built a SEPARATE gold path (synthetic code byte-unchanged): `data/gold_dataset.py`,
  `data/gold_dataloader.py`, `configs/backbones/qwen2vl_2b_gold.yaml` (disables
  confidence/memory/recovery_outcome — no gold labels), `scripts/run_gold.py`.
  Honest input = state_before+state_after+task only. Reuses model/loss/Trainer/metrics.
- Gold label values map cleanly to existing enums (verified 0 unmapped on 1219 train rows).

## NEXT
- Upload the gold folder as a Kaggle Dataset; `python scripts/run_gold.py --stage smoke` then
  `--stage train --data-root <kaggle path>`. Headline = `outcome_mcc` on gold TEST split.
  Ignore memory_acc/recovery_outcome_acc/ece (heads disabled on gold).
- Optional: fine-tune from a synthetic-pretrained ckpt via `--checkpoint`.

## Latest commit
`683c0e3` — separate gold training path (4 new files; synthetic pipeline untouched).

## 2026-07-17 — full repository and Q1-readiness audit
- Scope inspected: all 71 tracked files, all Markdown/spec/config/source/test/result files,
  both notebooks and their saved outputs, the v8/v12/v14 local gold artifacts, and all
  11,864 images referenced by the v14 splits (all open successfully at 1280x720 PNG).
- **Publication stop:** v14 does not satisfy `docs/GOLD_40K_SPEC.md` yet. All 32 domains
  and all 15 task templates occur in train/val/test; task text predicts action type with
  100% accuracy; 67.9% of test rows have the exact same complete model input in train.
  The 11,864 referenced image paths reduce to only 2,113 distinct SHA-256 images; 81.9%
  of test before-images and 80.3% of test after-images already occur in train.
- Exact-input memorization baseline on v14 test (no neural model): outcome MCC 0.764,
  balanced accuracy 0.869, macro-F1 0.871. Existing pilot metrics therefore cannot be
  treated as generalization evidence. There are also 84 identical-input groups (855
  rows) with conflicting outcome/failure/action labels.
- **Temporal design issue:** the shared representation sees `state_after`, then predicts
  pre-action action/bbox and `agent_confidence_before`. Those heads receive future
  information. Split the system into a pre-action policy path (before+goal -> action,
  bbox, confidence) and post-action verifier/recovery path (before+after+executed action
  -> outcome, failure, recovery, memory), or enforce equivalent causal masking.
- **Implementation completeness:** only the Qwen VLM path is implemented. Dual encoders,
  cross-attention, generic smoke/mini/full CLI stages, and B1-B4 baselines are stubs.
  `pytest` currently fails because labels define 6 actions while the sole test expects 5;
  the notebook's MINI block is stored as Markdown and its checkpoint cell has a saved
  `NameError` output.
- **Metric/reproducibility issues:** recovery-outcome accuracy includes null sentinel
  labels in the shared metric function; ECE compares predicted pre-action confidence to
  model correctness despite a different training target; checkpoint files omit optimizer,
  scheduler and scaler state; precision/checkpoint settings are partly hard-coded; package
  dependencies remain unpinned.
- No implementation code was changed by this audit. Next safe order: freeze headline
  training -> build automated gold leak validator -> regenerate goal-only tasks and
  group/domain/image-disjoint splits -> resolve causal architecture -> repair tests and
  resumable training -> run cheap non-neural/text/image-retrieval baselines -> only then
  launch multi-seed model experiments.
- Commit audited: `191b554` on branch `Code`.

## 2026-07-17 — 39k candidate dataset handoff report received
- External handoff report: `C:\Users\kiyas\Downloads\DATASET_REPORT.md`; large dataset is
  stored outside Git and intended for Kaggle, which is appropriate for the image volume.
- Reported candidate: 39,215 rows, 29,978 trajectories, 509 domains, 23,499/7,861/7,855
  train/val/test rows, six balanced action classes, and 9,237 recovery attempts
  (3,153 success / 6,084 failure). Export claims task-disjoint and domain-disjoint splits.
- This is a different and much stronger candidate than local v14, but the claims have not
  yet been independently verified against the uploaded Kaggle files. Do not transfer the
  old v14 leakage verdict to it, and do not mark it publishable solely from the report.
- Remaining data gates explicitly reported: all rows still `review_status=pending`; 71
  probable blank/browser-error rows in 62 trajectories; 1,354 rows from 16 adult domains
  need a protocol decision; success share is 43.70% rather than the planned >=45%.
- Required verification after the Kaggle slug/path is available: checksums/version manifest,
  zero domain/task/trajectory overlap, zero exact/near image-hash overlap, exact full-input
  duplicate/conflict audit, quantitative task-text-only action baseline, schema/image scan,
  and approved-only review gates.
- Dataset improvement does not resolve the temporal model issue: `state_after` must not feed
  the pre-action action/bbox or `agent_confidence_before` heads. Resolve causal head inputs
  before training the 39k candidate for headline results.

## 2026-07-17 — Kaggle access path confirmed (no local dataset transfer)
- Kaggle handle supplied: `kiyasmahmud/web-gold-40k`.
- Do not call `kagglehub.dataset_load` or `dataset_download` from this local workspace:
  outside Kaggle, KaggleHub downloads the selected resource into a local cache, and this
  machine does not have enough free space.
- The supplied example also needs a concrete tabular `file_path`; an empty path cannot be
  used to load the mixed/large dataset as one Pandas DataFrame. `load_dataset` is deprecated
  in current KaggleHub in favor of `dataset_load`.
- Safe execution route: create a Kaggle Notebook, attach `web-gold-40k` as an Input, and run
  validation/training there. KaggleHub is already authenticated in Kaggle Notebooks and
  serves attached resources from Kaggle's shared resource cache rather than the VM disk.
- No API token was stored, echoed, committed, or used during this check; no dataset file was
  downloaded. If a token was pasted into any chat/plaintext location, revoke and regenerate it.

## 2026-07-17 — separate Kaggle 40k data-validation notebook
- Added `notebooks/kaggle_gold_data_validation.ipynb`; it is CPU-only, auto-detects the
  attached `kiyasmahmud/web-gold-40k` split directory, and never calls KaggleHub/download.
- Added reusable `scripts/validate_kaggle_gold.py`. It checks the 39,215-row handoff
  manifest, exact nested input allow-list, required labels, approved-only status, label
  logic/distributions, unique IDs, domain/task separation, exact-input duplicates and
  conflicts, every image reference, sampled image decodes, and text/confidence shortcut
  baselines. It also blocks publication readiness until the reported adult-domain inclusion
  or quarantine decision is confirmed. It writes only a small JSON report under
  `/kaggle/working`.
- Optional `--full-image-hash` performs streaming SHA-256 plus 64-bit difference hashes over
  unique referenced images. It blocks exact content overlap and cross-split perceptual pairs
  within dHash Hamming distance <=3. It is off in the quick notebook run and must be enabled
  for the final publication audit.
- Local smoke validation ran end-to-end on the older v14 fixture without modifying it. The
  validator correctly reproduced v14's known task-text leak (action accuracy 1.0000) and
  domain overlap, while passing schema, image-reference, sampled-decode, and confidence
  shortcut checks. Python compilation and all notebook code-cell compilation pass.
- The validation commit does not edit `notebooks/kaggle_gold.ipynb`. After rebasing onto
  the two newer training-notebook commits already on `origin/Code`, the local file matches
  the remote version exactly (SHA-256
  `340FEF1FC9A5703B4D0D0D8BA64080114FB89BCACA826B7E58D4A137A15EBE1E`).

## 2026-07-17 — Kaggle ZIP-in-place validation support
- The uploaded 40k dataset is a ZIP inside the attached Kaggle Dataset, not an extracted
  directory. Updated the separate validator and notebook to auto-detect either layout.
- ZIP mode locates the common folder containing `split_train.json`, `split_val.json`, and
  `split_test.json`, loads the JSON members directly, resolves image references against the
  same archive prefix, and streams image decode/SHA-256/dHash reads from the archive.
- No `extract`, `extractall`, KaggleHub download, or image copy is used. Only the small JSON
  validation report is written to `/kaggle/working`.
- End-to-end ZIP fixture test passed: nested split discovery, unique/safe member gate, six
  image-reference resolutions, six image decodes, exact SHA-256 audit, and dHash audit all
  completed directly from the archive. Directory-mode regression smoke also still passes.
- `notebooks/kaggle_gold.ipynb` remains outside the validation diff and unchanged at SHA-256
  `340FEF1FC9A5703B4D0D0D8BA64080114FB89BCACA826B7E58D4A137A15EBE1E`.

## 2026-07-18 — pulled Kaggle output and fixed deep dataset mount discovery
- Fast-forwarded three Kaggle commits (`03f56d7`, `3b7b453`, `f36b0e5`) containing the
  executed `kaggle_gold_data_validation.ipynb` output.
- Saved traceback: cell 2 raised `AssertionError: Dataset not found` even though Kaggle
  metadata showed dataset source ID `18087168` attached. Root cause was the notebook and
  validator searching ZIPs only to a fixed shallow depth; Kaggle can mount versioned inputs
  under a deeper `/kaggle/input/datasets/<owner>/<slug>/versions/<n>/...` hierarchy.
- ZIP and split discovery now falls back to recursive filename search at arbitrary mount
  depth. The notebook no longer fails in preflight: if it cannot choose a candidate itself,
  it passes `/kaggle/input` to the validator and prints candidate paths for diagnosis.
- Cleared the stale saved traceback/output and reset code-cell execution counts. A synthetic
  deep Kaggle-mount fixture passed nested ZIP discovery, direct split JSON reads, six image
  reference resolutions, and six direct archive image decodes without extraction.
- `notebooks/kaggle_gold.ipynb` was not edited and remains SHA-256
  `340FEF1FC9A5703B4D0D0D8BA64080114FB89BCACA826B7E58D4A137A15EBE1E`.

## 2026-07-18 — confirmed Kaggle dataset root
- User-confirmed attached root: `/kaggle/input/datasets/kiyasmahmud/web-gold-40k`.
- The validator and separate validation notebook now prefer this exact directory, then
  recursively locate the versioned ZIP beneath it. Generic auto-discovery remains as a
  fallback if Kaggle changes the mount layout in a future dataset version.
- No extraction/download behavior changed; the main `kaggle_gold.ipynb` remains untouched.

## 2026-07-18 — fixed Kaggle version-link traversal
- The pinned root still raised `No sibling split JSON files or compatible ZIP archive found`.
  Discovery reached the correct directory but used `Path.rglob()`, which does not descend
  through Kaggle's linked version directories.
- Replaced recursive discovery with a loop-safe `os.walk(..., followlinks=True)` traversal.
  It now finds ZIP archives and nested extracted split folders behind version links while
  continuing to stream ZIP contents without downloading, copying, or extracting them.
- Failed discovery now reports rejected ZIP reasons and the first files actually visible
  below the supplied root, so a future Kaggle layout difference can be diagnosed from one
  saved notebook traceback rather than another generic error.
- A Windows directory-junction fixture (the local equivalent of the linked Kaggle mount)
  passed nested ZIP discovery. A separate nested extracted-split fixture also passed.
- Simplified the separate validation notebook preflight to pass the confirmed root directly
  to the validator and display whether root entries are links. `notebooks/kaggle_gold.ipynb`
  remains outside the change.

## 2026-07-18 — first successful Kaggle 39k validation result
- Pulled Kaggle output commit `7eb78e8`. The validator found the attached dataset at
  `/kaggle/input/datasets/kiyasmahmud/web-gold-40k/final_data_set_40k` and completed the
  quick audit in about 2 minutes 13 seconds without downloading or extracting it.
- Result: 31 PASS, 1 FAIL, 1 WARN, 3 SKIP; `publication_ready=false`.
- Strong verified properties: 39,215 expected rows; valid schema and label logic; unique
  sample IDs; zero domain overlap; zero task/trajectory overlap; zero exact full-input
  overlap or conflicts; all 78,430 image references exist; zero image-path overlap; and
  600/600 sampled images decode successfully.
- Shortcut checks passed their registered gates: task-text-only action accuracy 0.3537
  versus 0.1901 majority accuracy (macro-F1 0.2374), and confidence-only outcome
  MCC 0.1132 / ROC-AUC 0.5609. The task wording still contains moderate action signal and
  must remain a reported baseline/ablation even though it is below the blocker threshold.
- Publication blockers: all 39,215 rows remain `review_status=pending`; exact SHA-256 and
  near-image dHash split-overlap audits were not run; and the protocol decision for 1,354
  rows (3.45%) from 16 adult domains remains unresolved.
- Non-blocking balance warning: SUCCESS is 43.70%, 1.30 percentage points below the stated
  45% lower target. Action classes are well balanced (largest 18.98%).
- Do not start headline Q1 training yet. First complete/record human review, resolve or
  quarantine the adult-domain rows, rerun split construction if rows change, then enable
  the full image-hash audit and rerun every validation gate.

## 2026-07-18 — 39,215-row retention and training boundary
- User will retain the current 39,215-row corpus rather than add filler to reach an exact
  40,000 rows, and a teammate will perform the manual review. Report the exact 39,215 count
  consistently; the rounded dataset name does not require synthetic padding.
- The 1,354 adult-domain rows may remain only with a written inclusion, ethics, and safety
  protocol. Keeping rows means no split rebuild is needed unless manual review rejects or
  modifies records; any removal or label correction requires rerunning all split gates.
- Pipeline smoke testing on 16 rows may start while review and full image hashing run in
  parallel. Do not launch headline/full training until those publication gates pass.
- Before mini multitask training, fix two methodology issues in the current gold runner:
  `state_after` presently reaches the shared action/bbox/confidence heads (future-information
  leakage), and the `train` stage automatically evaluates the test split. Use causal pre/post
  head routing and reserve test evaluation for the final frozen model selection protocol.

## 2026-07-18 — modular causal smoke/mini notebook
- Reworked `notebooks/kaggle_gold.ipynb` as the explicitly authorized smoke/mini driver for
  the 39,215-row dataset. It has a single `STAGE = "smoke" | "mini"` control, asserts the
  new 23,499/7,861 train/validation counts, clears stale outputs, records package and commit
  versions, and never loads test records during development stages.
- Added causal gold routing while preserving the Gold 40K input allow-list. The same shared
  VLM+adapter encodes two views: `pre_*` contains state-before plus task/domain and feeds
  action, bbox, and confidence-before; `post_*` contains before+after plus task/domain and
  feeds outcome, failure type, recovery, memory, and recovery outcome. No label is inserted
  into either prompt.
- Added `train/gold_stages.py` so the notebook orchestrates rather than duplicates model
  construction. Smoke selects 16 label-covering rows and checks stream shapes, finite loss,
  finite backward gradients, and a real parameter update. Mini uses a deterministic,
  failure-stratified 5,000-row train subset plus 500 validation rows, requires decreasing
  loss and positive validation MCC, and perturbs/reloads a checkpoint to prove identical
  predictions return.
- Gold checkpoint selection now honors `train.early_stop_metric=outcome_mcc`; checkpoint
  files include LoRA, adapter, all heads, optimizer, scheduler, scaler, epoch, and step.
  Gradient accumulation now flushes the final partial group instead of silently dropping it.
- Corrected evaluation semantics: recovery-outcome accuracy/MCC mask null (not-attempted)
  labels; agent-confidence is reported as MAE against its pre-action target; outcome ECE is
  computed from outcome softmax confidence; failure/action macro-F1 are logged. The
  confidence calibration loss now targets observed action success rather than whether the
  separate outcome head happened to classify the sample correctly.
- Corrected the stale five-action test/documentation to the six Gold 40K actions, including
  PRESS_KEY. Local verification: all edited Python files and all notebook code cells compile,
  notebook JSON is valid/output-free, merged config checks pass, and pytest reports 1 passed
  with GPU/torch-dependent tests skipped because this local Python environment has no PyTorch.
  The required real forward/backward smoke remains the first Kaggle execution gate.
