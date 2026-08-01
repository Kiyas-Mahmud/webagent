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

## 2026-07-18 — five-epoch mini result export
- Changed the Gold mini-training default from 3 to 5 epochs in the Kaggle notebook,
  reusable stage function, and command-line runner. Mini training raises early-stop patience
  to the requested epoch count so this fixed development gate produces all five results.
- Added a dependency-free CSV exporter. A successful mini run writes one row per completed
  epoch, all training/validation metrics, the selection metric, best-epoch flag, dataset
  sizes, and checkpoint status to `/kaggle/working/gold_mini_result.csv`.
- The notebook now asserts that exactly five epoch records exist and that the final CSV was
  created before declaring the mini stage passed.
- Local verification passed: CSV export regression plus the existing dependency-free guard
  report 2 passed / 4 PyTorch-dependent skipped; changed Python files and every output-free
  notebook code cell compile, notebook JSON is valid, and `git diff --check` passes.

## 2026-07-18 — two-reviewer manual dataset guide
- Added `docs/DATASET_MANUAL_REVIEW_GUIDE.md` as a beginner-friendly handoff for the two
  human reviewers. It explains the 39,215-row dataset, nested schema, current distributions,
  label meanings, trajectory-level image/action/outcome/failure/recovery checks, and the
  distinction between the 7,559-row sample package and full approved-only review.
- Defined a two-person protocol: shared calibration, 10% independent overlap, trajectory-level
  work allocation, second review for every correction/rejection, disagreement adjudication,
  immutable review logs, and agreement/kappa reporting.
- Documented evidence-based change rules, prohibited edits, known blank/error and adult-domain
  queues, review-log columns/reason codes, safe correction/versioning workflow, final full-image
  hash audit, and the exact deliverables required before claiming publication readiness.

## 2026-07-18 — recovery-controlled mini v1 implementation
- Pulled and audited `kiyasmahmud/kaggle-gold-v14`: the five-epoch engineering run passed,
  but outcome MCC plateaued at 0.5217, recovery accuracy stayed at the ~0.75 NONE-class
  prior, selected-checkpoint recovery-outcome MCC was 0.1864, and bbox/calibration degraded.
- Added deterministic joint-proportional selection for the 5,000 training rows across failure,
  action, recovery attempt/strategy/outcome, memory, and bbox availability. Added a recovery-aware
  physical-batch schedule that uses every selected row exactly once and spreads attempted
  recoveries without validation oversampling or train-row duplication.
- Recovery strategy now uses normalized sqrt-inverse-frequency weights (cap 3.0) on both the
  failure and memory recovery logits. Action/failure/outcome/recovery and recovery-outcome BCE
  weights are derived from the exact selected training rows; validation labels never affect loss
  weights. Loss coefficients, architecture, learning rates, seed, and primary outcome-MCC
  selection remain unchanged.
- Preserved the exact legacy v14 failure-stratified 500-row validation selector for the controlled
  comparison. The new run uses the unique `recovery_v1` experiment tag and retains all five epoch
  checkpoints. A validation-only historical-checkpoint re-evaluation stage can compute the newly
  added metrics for v14 without opening the test split.
- Expanded honest evaluation: macro-F1/balanced accuracy/MCC and majority baselines for every
  categorical head; Brier scores; recovery denominators and predicted-class count; bbox mean/median
  IoU and Recall@0.5; per-class precision/recall/F1, distributions, confusion matrices; raw and
  weighted loss terms; independent best epochs; and explicit offline-recovery terminology. Scalar
  history stays in CSV and non-tabular evidence is exported to `gold_mini_diagnostics.json`.
- Added predeclared quality gates at the outcome-selected checkpoint: outcome retention, recovery
  lift/diversity/outcome MCC, and non-regression bounds for action, bbox, and calibration. Engineering
  PASS and quality PASS/FAIL are deliberately separate so a completed weak experiment is reported
  honestly instead of crashing or being hidden.
- Separate `notebooks/kaggle_gold_recovery_v1.ipynb` integration is output-free and asserts five
  checkpoints, zero duplicated scheduled rows, fixed validation comparability, CSV/diagnostics
  creation, and test-split isolation. The teammate's executed `kaggle_gold.ipynb` v14 result is
  preserved unchanged. Downloaded Kaggle artifacts are ignored by Git and pytest is constrained
  to the authoritative `tests/` tree.
- Local verification: Ruff clean; `10 passed / 5 skipped` (GPU/PyTorch-dependent checks require
  Kaggle); all Python files and notebook code cells compile; notebook JSON is valid with zero saved
  outputs/execution counts; merged Gold config assertions and `git diff --check` pass. Next required
  runtime gate is the Kaggle 16-row GPU smoke, followed by the controlled five-epoch mini. Headline
  training remains blocked by manual review/adult-domain protocol and full SHA-256/dHash audit.

## 2026-07-19 — recovery-controlled mini v1 Kaggle result analysis
- Downloaded the latest `kiyasmahmud/kaggle-gold-v14` output to the ignored local folder
  `kaggle_outputs/kaggle-gold-v14-latest/`. Verified all five epoch checkpoints plus `last.ckpt`
  are complete (~234 MB each), and the CSV, diagnostics, environment manifest, and mini report are
  present. The run used commit `be30558715e1876bbd58a1c11c4bb8987ff82982`.
- Engineering status is PASS: five epochs completed, train loss decreased 0.9686 -> 0.5264,
  checkpoint round-trip passed, exactly 5,000 unique train rows were scheduled with zero duplicates,
  and no test rows were read. The predeclared quality status is FAIL.
- At the outcome-selected epoch 4: outcome MCC 0.5080 / balanced accuracy 0.7527, failure macro-F1
  0.7538, fail-type macro-F1 0.4874, action accuracy 0.4140 / macro-F1 0.3793, recovery macro-F1
  0.3403 / MCC 0.1994, recovery-outcome MCC 0.1452, memory macro-F1 0.6677, outcome ECE 0.2104,
  bbox mean IoU 0.0108, median IoU 0, and Recall@IoU50 0.
- Versus the original v14 outcome-selected checkpoint, outcome MCC changed 0.5217 -> 0.5080,
  action accuracy 0.396 -> 0.414, recovery-outcome MCC 0.1864 -> 0.1452, and ECE
  0.1853 -> 0.2104. Sampling/weighting successfully broke the recovery NONE-only collapse
  (four predicted classes; macro-F1 beats the 0.2143 majority macro-F1), but it did not produce a
  better unified checkpoint.
- Class diagnostics expose structural blockers: LOOP_DETECTED F1 is 0 despite 15 validation rows;
  BACKTRACK F1 is 0; validation has no RETRY or ABORT rows and the 5k training subset also has none,
  so a six-way recovery claim cannot be learned or evaluated from this experiment. Action predictions
  overproduce NAVIGATE/PRESS_KEY and underproduce CLICK/SELECT/SCROLL. Bbox MAE looks superficially
  acceptable while IoU proves localization is effectively nonfunctional.
- Loss-scale audit: at epoch 4 the nominally 0.10-weighted bbox term contributes only ~0.0024 of
  total loss 0.5264 (~0.5%); fail-type/action/recovery terms dominate. More epochs alone are not the
  remedy: recovery-outcome MCC peaks at epoch 0 (0.2074) then degrades while training loss continues
  downward, showing overfitting/negative multitask transfer.
- Causal audit found a deeper recovery issue. Dataset review rules state `recovery_success` is verified
  from later trajectory steps, but `GoldDataset` feeds the recovery-outcome head only the current row's
  before/after action screenshots plus goal/domain. The target therefore depends on evidence absent
  from the model input. Fix by re-exporting existing replay trajectories as recovery transitions
  `(failure state, executed recovery, post-recovery state, success)` or remove recovery-success as a
  prediction head and retain it only as a memory-index filter. This should use existing replays rather
  than recollecting the whole dataset.
- Recommended next controlled order: (1) finish human review and full hash gate; (2) build the causal
  recovery-transition export; (3) factor recovery into `needs_recovery` plus an attempted-row-only
  strategy classifier over supported classes; (4) condition post-action diagnosis on the causally
  observed executed action; (5) replace pooled-vector bbox MSE with a spatial grounding head and
  SmoothL1+GIoU/IoU supervision; (6) normalize or dynamically balance multitask gradients and use a
  predeclared composite/Pareto checkpoint rule; (7) rerun 5k smoke/mini before any full-data run.

## 2026-07-19 — supplied v12 metrics and leakage re-audit
- Inspected `C:\Users\kiyas\Downloads\gold_v12_metrics.csv` and the supplied terminal screenshot.
  The reported v12 gold-test scores are outcome MCC 0.8372, balanced accuracy 0.9254, failure
  macro-F1 0.9114, action accuracy 0.9612, recovery accuracy 0.8760, memory accuracy 0.9018,
  ECE 0.2175, and bbox MAE 0.0688. The screenshot itself correctly says to ignore
  recovery-outcome accuracy because that head was disabled/sparse.
- Re-audited the local v12 corpus (1,970 rows; 1,185/398/387 train/val/test) with the current
  validator, including full SHA-256 and dHash checks. Saved the ignored report at
  `kaggle_outputs/v12_reaudit_2026-07-19.json`.
- v12 is not comparable to the current 39k domain-held-out experiment: all 26 v12 domains occur in
  all three splits; 163 exact image-content hashes cross splits; and 304,471 cross-split image pairs
  have dHash distance <=3. Task IDs are disjoint, but domain and visual-template generalization are
  not tested.
- The v12 task-text-only action baseline reaches 0.8682 accuracy / 0.8819 macro-F1 on test versus
  the neural model's 0.9612 action accuracy. Much of the apparent 96% action result is therefore
  available from task wording alone. By comparison, the verified 39k task-text baseline is 0.3537
  and the recovery-v1 model reaches 0.4140: the absolute number is lower because the shortcut was
  deliberately removed, while the incremental multimodal lift is of similar order.
- v12 training predates causal routing. Its single shared VLM input contains both `state_before` and
  `state_after`, then feeds action, bbox, and pre-action confidence heads. Those pre-action outputs
  therefore receive future information. The 39k causal runner intentionally restricts them to
  `state_before`; the score drop is expected and methodologically necessary.
- v12 has only six successful recoveries versus 464 failed recoveries; recovery-outcome was not a
  meaningful trained result. Its high recovery-strategy accuracy cannot substitute for recovery
  success evidence. All 1,970 rows also remain `review_status=pending`.
- The original 39k v14 controlled checkpoint is not far better than recovery-v1: outcome MCC
  0.5217 vs 0.5080, while action accuracy is 0.396 vs 0.414. Its ~0.75 recovery accuracy was the
  NONE-class prior; recovery-v1 lowers raw accuracy while increasing class diversity and honest
  recovery macro-F1. Preserve the old scores as pilot/easy-split evidence, not Q1 headline claims.
- Improvement must target honest generalization rather than recreate v12 leakage: add executed action
  only to the post-action verifier; construct temporal context for LOOP_DETECTED; re-export proper
  recovery transitions from existing replays; use hierarchical conditional recovery heads; replace
  global pooled bbox regression with spatial grounding; increase/tiling image resolution for small UI
  text; and control multitask gradient/loss scale before another 5k mini run.

## 2026-07-19 — causal recovery-v2 implementation complete; Kaggle runtime pending
- Added a reference-only recovery transition builder and CLI. It joins rows within a task by
  `step_index` as `failure state -> next executed action -> post-recovery state -> success`, copies
  no images, rejects invalid/ambiguous/missing transition supervision, and reports coverage plus
  integrity violations. The full audit reports split-wise counts for RETRY, ABORT, BACKTRACK, and
  LOOP_DETECTED so any additional collection can be narrowly targeted.
- Gold v2 now supplies the executed current action only to the post-action verifier prompt. The
  pre-action stream remains action-free and continues to own action type, bbox, and confidence.
  Recovery-success uses a separate sparse VLM stream containing only proper transition rows, so
  future recovery state cannot leak into the ordinary outcome/action heads.
- Replaced flat all-row recovery supervision in the v2 config with a binary `needs_recovery` loss
  on all rows and strategy CE only on attempted-recovery rows. Added composed inference plus honest
  needs-recovery and attempted-strategy accuracy/macro-F1/balanced-accuracy/MCC, majority baselines,
  per-class diagnostics, and transition-only recovery-outcome metrics.
- V2 preserves Qwen image-token hidden states for an attention grounding bbox head. Predictions are
  bounded normalized xywh boxes; training uses masked SmoothL1 + GIoU. Mean IoU and Recall@IoU50
  are explicit quality gates, with MAE retained only as a supporting metric.
- Added separate residual adapters for policy, diagnosis, memory, and recovery plus learned
  uncertainty weighting for active multitask losses. Loss parameters are in the optimizer and both
  task-adapter and loss states are saved/restored in checkpoints. Inactive sparse terms do not push
  their uncertainty parameters.
- Added `configs/backbones/qwen2vl_2b_gold_v2.yaml`, the registered protocol
  `docs/RECOVERY_V2_EXPERIMENT.md`, recovery-v2 additions to the two-reviewer guide, and the isolated
  output-free `notebooks/kaggle_gold_recovery_v2.ipynb`. Both earlier Kaggle notebooks remain
  untouched. The new notebook performs the dataset audit/manifest export, enforces a 16-row smoke,
  then runs exactly 5,000 train / 500 validation rows for five epochs and saves CSV, diagnostics,
  environment, reports, manifests, and all checkpoints.
- Controlled-report metadata no longer falsely lists architecture/loss coefficients as fixed for
  v2. Predeclared v2 gates evaluate the outcome-MCC-selected epoch and separate engineering PASS
  from outcome/action retention, hierarchical recovery lift, transition-outcome MCC, bbox IoU/
  Recall@IoU50, and calibration quality.
- Local schema validation on the small v12 corpus found 285/285 proper train transitions and 98/98
  proper validation transitions with no missing, ambiguous, nonconsecutive, or invalid joins. This
  validates the join contract only; it is not evidence about the current 39k counts or model quality.
- Local verification passed: Ruff clean; `13 passed / 5 skipped` (PyTorch/GPU checks require Kaggle);
  Python compileall passed; merged v2 config assertions passed; notebook JSON and every code cell
  compile with no saved outputs/execution counts; and `git diff --check` passed. Required next gate:
  run recovery-v2 smoke on Kaggle, restart the kernel, run the five-epoch mini, then inspect the
  generated full class audit and quality gates before authorizing any headline/full-data training.

## 2026-07-20 — recovery-v2 epoch audit and evidence-driven v2.1 correction
- Audited the supplied recovery-v2 epoch 0/1 logs. The causal transition head is validated as a
  useful change: recovery-outcome MCC improved 0.4970 -> 0.7268 and macro-F1 0.7415 -> 0.8566.
  Outcome MCC reached 0.5514 (above the v14 0.5217 reference), memory MCC reached 0.6635, and
  outcome ECE fell to 0.0625. Action accuracy remained within the registered non-regression band.
- Confirmed two code defects rather than assuming the data was bad. First, attempted-strategy
  metrics consumed the composed recovery prediction, so an all-negative needs-recovery gate forced
  every attempted strategy prediction to `NONE` and produced a false zero. Second, the binary
  needs-recovery BCE was unweighted even though the exact 5k subset has 1,191 positive versus 3,809
  negative rows; the printed 1.93 weight applied only to success/failure within attempted recovery.
- Corrected evaluation to retain raw conditional-strategy predictions separately from the composed
  end-to-end recovery decision. Attempted-only accuracy/macro-F1/class diversity and diagnostics now
  use raw strategy logits; overall recovery metrics still honestly include the binary gate.
- Added exact selected-training-row needs-recovery `pos_weight = negative / positive` (cap 5.0),
  recorded its counts/scheme in the class-weight report, and never uses validation labels. For the
  observed v2 subset this resolves to approximately 3.20.
- Did not guess a new bbox loss or arbitrary coefficient. The unchanged v2 localization metrics
  (MAE 0.6106, mean IoU 0.0079, Recall@IoU50 0 across epochs 0/1) instead triggered direct
  instrumentation: first-step gradient norms, epoch parameter-update norms, bbox coordinate
  prediction/target distributions, spatial-token counts, and learned uncertainty multipliers.
- Added a bbox-specific residual grounding adapter so localization no longer shares the policy
  adapter with action classification. Strengthened smoke to require real bbox supervision plus
  finite nonzero gradients and optimizer updates for bbox, grounding adapter, needs recovery,
  strategy, and recovery outcome. Checkpoint round-trip now verifies every affected output/module.
- Separated the nominal weighted loss from the uncertainty objective. The latter may legitimately
  become negative because of learned log-variance terms, so the loss-decrease gate now uses the
  interpretable nominal weighted loss and CSV logs both log-variances and effective multipliers.
- Added registered config `qwen2vl_2b_gold_v2_1.yaml`, protocol
  `docs/RECOVERY_V2_1_EXPERIMENT.md`, and isolated output-free notebook
  `notebooks/kaggle_gold_recovery_v2_1.ipynb`. Required order is smoke -> one-epoch 5k diagnostic
  -> five-epoch mini only after diagnostics are healthy; v2 artifacts remain preserved.
- Local verification passed: Ruff clean; `13 passed / 8 skipped` (the added tensor/head tests and
  all GPU paths require the Kaggle PyTorch runtime); Python compileall passed; v2.1 merged-config and
  notebook-cell assertions passed; all older Kaggle notebooks have zero diff; and
  `git diff --check` passed. The next evidence must come from the v2.1 Kaggle smoke, not another
  assumption-driven training run.

## 2026-07-20 — bbox v2.2 correction implemented after diagnostic collapse
- Audited the supplied v2.1 one-epoch diagnostic before changing code. Its bbox prediction is a
  near-constant full-image box (`x=0.00014`, `y≈0`, `width=0.99986`, `height=0.999998`) with mean
  IoU 0.0079 and Recall@IoU50 0. The bbox output and grounding adapter both had nonzero gradients and
  substantial optimizer updates, while uncertainty slightly upweighted bbox. This rules out frozen
  parameters or a starved generic task coefficient and supports a geometry/objective correction.
- Added config-gated centre-format bbox prediction. V2.2 predicts normalized `cxcywh` internally,
  keeps the existing public/metric output as bounded top-left `xywh`, and replaces equal
  mean-SmoothL1/GIoU with separately logged DETR-style `5*L1(cxcywh) + 2*GIoU(xyxy)`. Older v2/v2.1
  configs and outputs retain their original path.
- Added a read-only full train/validation bbox audit against each actual `state_before` image plus
  strict v2.2 dataset enforcement. It reports exact IDs/reasons and never clips bad thesis labels.
  Reviewers must reconstruct the box from image/replay evidence or set only the unrecoverable bbox
  to `null`; the row remains available for every non-localization task.
- Running the audit on the local structured v14 reference copy found real coordinate-space defects:
  85/1,796 non-null train boxes and 23/621 validation boxes are invalid, dominated by vertical
  coordinates such as `y=1223.5` for a 1280x720 image. This local copy is smaller than the current
  Kaggle 39k corpus, so its counts are not substituted for the required Kaggle full audit. The
  supplied v2.1 validation distribution independently showed normalized target `y` reaching 1 with
  positive height, consistent with the same issue in the current run.
- Added a bbox-only 32-row/100-step micro-overfit gate that freezes the VLM and unrelated heads,
  runs only the pre-action stream, and trains the projection/grounding/bbox path. It requires loss
  decrease, IoU gain >=0.10, final training mean IoU >=0.20, full-screen fraction <=0.10,
  nonconstant predictions, and nonzero bbox/grounding gradient and update before another 5k epoch.
- Added registered config `qwen2vl_2b_gold_v2_2.yaml`, protocol
  `docs/RECOVERY_V2_2_EXPERIMENT.md`, reviewer guidance for mismatched coordinate spaces, and the
  isolated output-free `notebooks/kaggle_gold_recovery_v2_2.ipynb`. Required order is
  `audit -> smoke -> bbox_overfit -> diagnostic -> mini`; `kaggle_gold.ipynb` and all v2/v2.1
  notebooks remain untouched.
- Local verification passed: Ruff clean; `17 passed / 10 skipped` (PyTorch/GPU paths require
  Kaggle); Python compileall passed; v2.2 merged-config assertions passed; notebook JSON and every
  code cell compile with no outputs/execution counts; `git diff --check` passed. The first Kaggle
  action is `STAGE='audit'`; do not bypass its reported data corrections.

## 2026-07-20 — full Kaggle bbox audit resolved with target-level masking
- Analyzed the user-supplied full Kaggle audit rather than inferring from the earlier local subset.
  Train has 23,499 records, 8,761 non-null boxes, 6,719 valid boxes, and 2,042 invalid boxes;
  validation has 7,861 records, 2,841 non-null boxes, 2,349 valid boxes, and 492 invalid boxes.
  Combined, 2,534/11,602 non-null boxes are invalid (`21.84%`), while 9,068 remain valid.
- The dominant errors are bottom-boundary overflow and y-origin outside 1280x720 viewport images,
  including document-space y values in the thousands. This is a systematic coordinate-space
  mismatch, so clipping or manually guessing all 2,534 replacements would create label noise.
- V2.2 now retains every record and every non-bbox label while setting `bbox_mask=0` only for an
  invalid localization target. The original split JSON is neither edited nor silently clipped.
  The audit separately counts bbox-only maskable defects and fatal missing/unreadable image errors;
  only the former can pass as `PASS_WITH_INVALID_BBOX_MASKED`.
- The smoke stage guarantees at least one verified valid bbox target. The 32-row micro-overfit stage
  now samples only boxes verified against their actual images, eliminating random audit failures.
  Mini reports include the raw train/validation bbox geometry for the selected 5k/500 rows so paper
  denominators remain explicit.
- Updated the isolated output-free `notebooks/kaggle_gold_recovery_v2_2.ipynb`, v2.2 experiment
  protocol, reviewer guide, config, and regression tests. `kaggle_gold.ipynb` and older recovery
  notebooks remain untouched. Local verification: Ruff and compileall pass; `18 passed / 12 skipped`
  (tensor/GPU paths require Kaggle); notebook JSON/cell compilation and `git diff --check` pass.
- Next Kaggle action: connect `notebooks/kaggle_gold_recovery_v2_2.ipynb`, run `STAGE='audit'`, and
  require `training_disposition='PASS_WITH_INVALID_BBOX_MASKED'`, `masked_bbox_rows=2534`,
  `bbox_supervision_rows=9068`, and `fatal_invalid_bbox_rows=0` before moving to `STAGE='smoke'`.

## 2026-07-20 — v2.2 smoke result accepted; report-contract fix
- Pulled Kaggle result commit `2fa4a40`. It executed the intended code commit `667ed69` on a T4.
  The full audit passed its training disposition with 31,360 records retained, 9,068 valid bbox
  targets, 2,534 invalid bbox targets masked, zero fatal image errors, and zero test rows read.
- The core 16-row smoke returned `status=PASS`: all 16 rows were processed, loss was finite
  (`1.6658`), four valid bbox targets were supervised, all required bbox/recovery/grounding probes
  had positive gradients and optimizer updates, every sampled row retained 252 spatial tokens, and
  peak allocated GPU memory was 4.16 GB.
- The notebook's final policy cell stopped only because the smoke report omitted the observational
  field `test_rows_read`; this was a report-schema `KeyError`, not a model, loss, data, or GPU
  failure. Added the explicit constant `test_rows_read: 0` to `run_gold_smoke` plus an AST-based
  regression test. No learning behavior or registered experiment factor changed.
- Cleared the stale executed/error output from `notebooks/kaggle_gold_recovery_v2_2.ipynb` while
  retaining `STAGE='smoke'`. Rerun this cheap stage once to produce a completely green publication
  artifact; only then change the stage to `bbox_overfit`.

## 2026-07-20 — bbox micro-overfit gate authorized
- No additional Kaggle commit followed the smoke-report fix. Reassessed whether repeating the GPU
  smoke would add evidence: the original `run_gold_smoke` had already returned `status=PASS` only
  after processing all 16 rows, checking finite loss/backward, verifying every required gradient and
  parameter update, confirming valid bbox supervision/spatial tokens, and avoiding the test split.
  The later `KeyError` was outside that function and the fix added only the constant report field
  `test_rows_read: 0`; it did not affect data selection, forward/backward, loss, or optimization.
- Accepted that existing smoke as the engineering gate rather than spending another Kaggle GPU run
  solely to reproduce an observational field. Advanced the clean isolated notebook to
  `STAGE='bbox_overfit'`. The registered next gate remains 32 verified valid bbox rows for 100
  steps; diagnostic training stays blocked until every micro-overfit check passes.

## 2026-07-20 — recovery-v2.3 coordinate grounding implemented
- Analyzed the completed v2.2 32-row/100-step micro-overfit report before changing code. Bbox MAE
  improved from 0.31535 to 0.11328 and both bbox/grounding parameters updated, but mean IoU improved
  only 0.00406 -> 0.02465 (+0.02059 versus the registered +0.10 gate), final mean IoU missed the
  registered 0.20 gate, and predicted y collapsed near the top edge (mean 0.03018, std 0.00391).
  The first reported bbox and grounding gradients were non-finite under scaled FP16. This evidence
  supports a coordinate-representation and numerical-stability correction, not a lower threshold,
  generic loss-weight increase, or wholesale data recollection.
- Added explicit normalized Qwen patch centres derived from `image_grid_thw` after spatial merging,
  with a strict token-count alignment check. Recovery-v2.3 adds these coordinates to the spatial
  attention tokens, obtains the bbox centre through differentiable attention soft-argmax, predicts
  width/height through the bbox MLP, and preserves the external normalized top-left `xywh` contract.
  All legacy/content-attention configs remain on their previous code path.
- The frozen/quantized VLM remains in FP16 while the v2.3 trainable grounding adapter, coordinate
  projection, attention, bbox trunk/output, and DETR bbox loss run in FP32. The overfit probe disables
  dropout, records the first accepted finite gradients, counts attempted/rejected non-finite steps,
  and requires zero rejected steps. The original 32 rows, seed 42, 100 accepted updates, L1:GIoU
  ratio 5:2, IoU thresholds, and full-screen threshold remain unchanged.
- Added a deterministic visual montage of the exact 32 overfit records with their original bbox
  targets drawn in green. The isolated v2.3 notebook starts at `STAGE='smoke'` because the bbox
  architecture changed and refuses `bbox_overfit` until a reviewer explicitly sets
  `BBOX_MONTAGE_REVIEWED=True` after inspecting all targets.
- Added `configs/backbones/qwen2vl_2b_gold_v2_3.yaml`,
  `docs/RECOVERY_V2_3_EXPERIMENT.md`, output-free
  `notebooks/kaggle_gold_recovery_v2_3.ipynb`, spatial utilities, and config/notebook/gradient-path
  regression tests. V2.2 and `notebooks/kaggle_gold.ipynb` were not modified.
- Local verification passed: Ruff clean; `21 passed / 15 skipped` (PyTorch/GPU tensor execution
  requires Kaggle); Python compileall passed; merged config and every notebook code cell compile;
  notebook outputs/execution counts are empty; and `git diff --check` passed. This does not guarantee
  the empirical IoU gate. The next evidence must be the v2.3 Kaggle `smoke`, followed by manual
  montage review and only then the unchanged `bbox_overfit` gate.

## 2026-07-20 — v2.3 bbox overfit failure isolated to zero-size saturation
- Analyzed the user-supplied v2.3 audit and complete 32-row/100-step overfit report. The audit is an
  accepted PASS: 31,360 records retained, 9,068 valid bbox targets, 2,534 invalid targets masked
  only from localization, zero fatal image errors, and zero test rows read. The exact overfit sample
  contains 32/32 valid boxes, so invalid dataset geometry did not cause this failure.
- V2.3 fixed the earlier numerical and centre-grounding defects. All 100 optimizer steps had finite
  gradients; bbox/grounding first-gradient norms were 6.315/2.205 and update norms were 0.150/3.646.
  Attention entropy sharpened from 5.519 to 1.007 and the final x/y predictions became nonconstant
  (std 0.155/0.065). The loss decreased from a 7.188 first-window mean to 5.324, with the last
  pre-update step at 3.686.
- The remaining failure is the width/height decoder: public predicted width and height both became
  exactly zero with zero variance, versus target means 0.1559 and 0.0502. Consequently final mean
  IoU and Recall@IoU50 were both zero, and IoU gain was -0.00539. The current coordinate head uses
  `sigmoid(size_logits)` while the inherited head LR is 1e-3. It starts near width/height 0.54/0.53,
  far above these small targets; once logits overshoot deeply negative, sigmoid underflow produces
  zero size and its derivative cannot recover reliably. This mechanism matches both the code and
  the observed exact zero-size output; it is not a speculative data diagnosis.
- Do not run the 5k diagnostic and do not lower the IoU gates. The registered next correction should
  preserve v2.3 centre attention, FP32 grounding, rows, seed, steps, 5:2 outer loss ratio, and gates,
  while replacing only the size branch with train-only-prior initialization plus direct log-width/
  log-height supervision. Log internal centre/size/logit distributions at fixed step intervals so
  saturation or final-step overshoot is visible. Implement this as isolated v2.4 artifacts only
  after explicit approval; v2.3 remains the failed controlled result.

## 2026-07-20 — recovery-v2.4 log-size correction implemented
- Implemented the approved correction as an isolated v2.4 experiment; v2.3 and
  `notebooks/kaggle_gold.ipynb` remain unchanged. The coordinate-softargmax centre, FP32 grounding,
  deterministic 32 rows, seed 42, 100 steps, optimizer settings, 5:2 bbox ratios, and registered
  IoU/full-screen gates are inherited without relaxation.
- Added config-gated log-space width/height prediction. Public/internal sizes use a bounded
  exponential decode, while SmoothL1 reads the **unclamped** predicted log-width/log-height. This
  leaves a direct nonzero recovery gradient even when the decoded size reaches its numerical floor,
  removing the exact saturated-sigmoid mechanism demonstrated by v2.3.
- The size head now starts from the geometric mean width/height calculated only from valid bbox rows
  in the exact selected training records. Invalid localization targets are excluded consistently
  with their bbox mask; validation and test records never contribute to this prior. The prior and
  its source counts are saved in smoke/mini or overfit reports.
- Expanded the micro-overfit evidence with public coordinate minima/maxima, internal cxcywh size
  ranges, raw log-size distributions, centre/log-size/GIoU sublosses, and a named pre-update trace at
  optimizer steps 10,20,...,100. V2.4 additionally fails if any final public/internal width or
  height returns below `1e-6`; the original mean-IoU gain and final-IoU thresholds are unchanged.
- Added `configs/backbones/qwen2vl_2b_gold_v2_4.yaml`,
  `docs/RECOVERY_V2_4_EXPERIMENT.md`, output-free
  `notebooks/kaggle_gold_recovery_v2_4.ipynb`, and regression tests for training-only prior
  provenance, saturated-logit gradients, head initialization/decoding, combined-loss routing,
  inherited controls, and notebook compilation. The notebook deliberately starts at `smoke`
  because the size parameterization and loss changed.
- Local verification passed: Ruff clean; Python compileall passed; full local suite `24 passed / 18
  skipped` (PyTorch/GPU execution is unavailable in this Windows runtime and remains a Kaggle gate);
  merged v2.4 config proves optimizer equality with v2.3; notebook JSON and every code cell compile
  with empty outputs/execution counts; and `git diff --check` passed. The next permitted action is
  v2.4 `smoke`, manual confirmation of the unchanged montage, then `bbox_overfit`. Diagnostic/mini
  training remains blocked until every saved overfit check is true.
- During push, remote commit `916a698` arrived with the completed v2.3 Kaggle notebook evidence
  (`STAGE='bbox_overfit'`, montage reviewed, six executed cells with outputs). It was preserved by a
  clean rebase and not edited. Consequently the old source-hygiene test that requires the protected
  v2.3 notebook to remain at `STAGE='smoke'` and output-free now reports one expected failure; the
  post-rebase v2.4/v2.2 targeted suite remains green (`9 passed / 7 skipped`), Ruff remains clean,
  and compileall passes. Do not "fix" that failure by deleting the user's v2.3 evidence inside this
  v2.4 task; archive/clean the executed notebook only under a separately approved artifact policy.

## 2026-07-20 — v2.4 overfit result fixes size collapse; centre attention remains gated
- Analyzed the supplied full Kaggle v2.4 audit and 32-row/100-step overfit report. The full audit
  remains accepted with 31,360 records retained, 9,068 valid bbox targets, 2,534 invalid targets
  masked only from localization, zero fatal rows, and zero test rows read. The fixed probe itself is
  32/32 geometry-valid, so raw full-corpus geometry status `FAIL` is not the overfit cause.
- V2.4 solved the exact v2.3 saturation defect: final public widths/heights are positive and
  nonconstant, internal size minima are 0.0775/0.0358, log-size outputs are finite, every gradient
  step is finite, loss falls 6.772 -> 4.804, bbox MAE improves 0.1588 -> 0.0746, mean-IoU gain is
  0.1291 (passing the registered +0.10 gate), and Recall@IoU50 rises 0 -> 0.125.
- Only the unchanged final mean-IoU gate fails (`0.1291 < 0.20`). Size means are now close to target,
  and horizontal location is reasonable. The remaining error is vertical localization: internal
  predicted centre-y averages 0.0925 versus target centre-y about 0.1962, with much less spread.
  Attention entropy collapses 5.519 -> 1.104, showing a sharp but usually wrong patch selection.
- Verified against Hugging Face's Qwen2-VL implementation that merged vision positions are flattened
  temporal-height-width/raster order, matching the repository coordinate construction. Do not spend
  another run merely swapping coordinates or lower the IoU threshold.
- Next registered experiment should remain isolated: preserve v2.4 size/prior and all 100-step
  controls; first add a feasibility audit for duplicate pre-action inputs with conflicting bbox
  targets, then add direct target-distribution supervision to the existing attention map and log
  per-row/step localization. Do not simultaneously extend steps or tune the threshold. Diagnostic,
  mini, and full training remain blocked until the unchanged 0.20 micro-overfit gate passes.

## 2026-07-20 — recovery-v2.5 supervised patch attention implemented
- Implemented the evidence-selected v2.5 correction without changing v2.4, earlier recovery
  notebooks, or `notebooks/kaggle_gold.ipynb`. The fixed 32 rows, seed 42, 100 optimizer updates,
  optimizer settings, centre/log-size ratio 5.0, GIoU ratio 2.0, minimum +0.10 IoU gain, minimum
  0.20 final mean IoU, full-screen limit, FP32 grounding, size prior/decoder, recovery design, and
  every non-bbox objective remain inherited unchanged.
- Added a normalized `KL(target patch distribution || grounding attention)` auxiliary term on valid
  bbox rows. The masked target distribution is centred on the target bbox, scales with half its
  width/height, and assigns zero mass to padding. Only grounding-attention dropout is registered as
  zero so the supervised probabilities are stable; the inherited trunk/head dropout is unchanged.
- Added a pre-GPU feasibility gate for the exact fixed probe. It hashes decoded RGB pixels plus
  dimensions and combines them with the exact pre-action task/domain text. Identical deterministic
  inputs with conflicting normalized bbox targets now return a structured FAIL instead of being
  misdiagnosed as an optimization defect. The audit is read-only and reports zero test rows read.
- Expanded the overfit evidence to complete fixed-set evaluations at steps 0, 25, 50, 75, and 100.
  Every checkpoint records attention KL/entropy, bbox metrics and distributions, and all 32 rows'
  IDs, predictions, targets, IoUs, centre errors, attention peaks, and raw log sizes, plus the ten
  worst rows. V2.5 additionally requires attention KL to decrease; no empirical gate was weakened.
- Added `configs/backbones/qwen2vl_2b_gold_v2_5.yaml`,
  `docs/RECOVERY_V2_5_EXPERIMENT.md`, output-free
  `notebooks/kaggle_gold_recovery_v2_5.ipynb`, and focused regression tests for conflict detection,
  target distribution/masking, differentiability, head output contracts, combined-loss routing,
  inherited controls, and notebook isolation/compilation. The notebook starts at `STAGE='smoke'`.
- Local verification: Ruff clean; Python compileall passed; v2.5 non-PyTorch tests `4 passed` and
  PyTorch-dependent tests `4 skipped` because no local Python runtime has PyTorch; notebook JSON and
  every code cell compile with empty outputs/execution counts; and `git diff --check` passed. The
  wider bbox suite is otherwise green but retains the single known v2.3 notebook-hygiene failure
  caused by preserved executed Kaggle evidence. Actual tensor/GPU behavior remains intentionally
  gated by v2.5 Kaggle `smoke`, followed by manual montage review and then `bbox_overfit`; diagnostic
  and mini remain blocked until the unchanged bbox overfit checks all pass.

## 2026-07-21 — recovery-v2.7 checkpoint selection correction
- Preserved the completed v2.6 training evidence and model weights. V2.7 changes no data,
  architecture, loss, optimizer, seed, thresholds, or validation rows; it changes only how a
  completed epoch is permitted to represent the controlled experiment.
- Registered `all_gates_then_outcome_mcc`: evaluate all eight quality gates independently for every
  epoch, keep only all-gate-eligible epochs, then maximize the primary outcome MCC with an
  earlier-epoch tie break. If no epoch is eligible, the run remains quality `FAIL` and the
  unconstrained outcome epoch is retained only for diagnosis.
- Replaying the exact five-epoch v2.6 report produces eligible epochs `[3, 4]` and selects epoch 4
  (`outcome_mcc=0.5480665`). Epoch 1 remains recorded separately as the unconstrained maximum
  (`0.5542802`) and is not misreported as the selected quality checkpoint. All eight epoch-4 gates
  pass and zero test rows were read.
- Synchronized the selected epoch, checkpoint path, selected metric, quality report, CSV flags, and
  prediction round-trip load path. A fresh v2.7 mini now loads and round-trip checks the selected
  epoch checkpoint, rather than always loading the trainer's unconstrained first checkpoint.
- Replaced omission-prone hard-coded v2 metadata sets with numeric version comparison, so v2.6 and
  v2.7 reports retain every inherited recovery/bbox control. Added a selection-only config, pure
  replay utility, output-free Kaggle notebook, and regression tests covering the v2.6 epoch history,
  no-eligible failure, deterministic ties, missing checkpoint rejection, config invariance, report
  synchronization, CSV selection markers, and notebook compilation.
- Local verification: exact v2.6 replay `PASS`; selected epoch 4 and eligible epochs `[3, 4]`; Ruff
  clean; Python compilation passed; focused suite `13 passed`; full suite `37 passed / 22 skipped / 2
  known failures`. The two failures are pre-existing hygiene assertions against preserved executed
  v2.3/v2.5 Kaggle evidence and are unrelated to v2.7. The fast v2.7 notebook does not retrain or
  claim a new checkpoint prediction round-trip; future fresh training through the v2.7 config does.

## 2026-07-21 - recovery-v2.7 full workflow and replay evidence split
- Preserved the completed no-GPU Kaggle selection replay, including its PASS transcript, as
  `notebooks/kaggle_gold_recovery_v2_7_selection.ipynb`. It records eligible epochs `[3, 4]`,
  selected epoch 4, outcome MCC `0.5480665242205773`, zero test rows, and the honest
  `NOT_MOUNTED` status for the old physical checkpoint.
- Replaced `notebooks/kaggle_gold_recovery_v2_7.ipynb` with the full output-free controlled
  workflow. It retains audit, smoke, 32-row bbox overfit, one-epoch diagnostic, and five-epoch mini
  stages; uses only the v2.7 config; exports versioned environment/report/CSV/diagnostic artifacts;
  and never opens the test split.
- The full mini contract now proves all-gate eligibility before ranking, derives the expected
  selected epoch from the fresh history, synchronizes the selected checkpoint and metric, requires
  the explicit selected-checkpoint prediction round-trip, retains the unconstrained best metric for
  diagnosis, and confirms exactly one selected row in the CSV.
- Corrected stale bbox-overfit notebook assumptions: v2.6/v2.7 register 200 optimizer steps, while
  the old notebook asserted a 100-step trace. V2.7 derives trace length and full-set evaluation
  steps directly from configuration. The core trajectory description also reports its actual final
  step instead of always claiming step 100.
- Local verification: both notebook JSON documents and all code cells compile; the full notebook has
  zero outputs/execution counts; the exact v2.6 selection replay passes and selects epoch 4; Ruff and
  `git diff --check` pass; focused selection/export tests are `14 passed`; the full suite is `38
  passed / 22 skipped / 2 known failures`. The two unchanged failures are the preserved executed
  v2.3/v2.5 notebook-hygiene checks and do not involve v2.7.

## 2026-07-22 - time-safe v2.7 epoch-4 continuation

- Inspected the complete failed-run archive instead of inferring from the screenshot. The log ends
  without a traceback at about 11 h 47 min after epoch-4 step 150/157, so the absent epoch-4 row is
  a finalization/session-time failure rather than evidence of a model-quality failure.
- Verified the retained epoch-3 checkpoint metadata: `epoch=3`, `step=628`,
  `epoch_complete=true`, selection metric `outcome_mcc`, value `0.5615662253`, and SHA-256
  `2f4a7e415c6382f0983d2c705d2cd6526d5e766c42319ff40f2adcc71240f9ba`. The four-row CSV and
  current quality gates select epoch 3 and pass all eight checks. `last.ckpt` is an incomplete
  rolling checkpoint from epoch 3 step 500 and is explicitly forbidden for this continuation.
- Added an output-free `kaggle_gold_recovery_v2_7_resume.ipynb` and modular resume runner. It accepts
  only the same v2.7 lineage, validates environment/config/epoch/step/metric/checkpoint mappings
  before GPU training, restores weights plus optimizer/scheduler/scaler, trains only epoch 4, then
  applies the registered selector to the complete 0–4 history.
- The selected physical artifact is copied to Kaggle working output, prediction-roundtrip tested,
  re-evaluated on the unchanged locked 500-row validation subset, and compared with its historical
  gate metrics before report/CSV/diagnostic export. Independent run rows and test rows are rejected.
- Recorded the only unavoidable limitation honestly: the historical checkpoint did not save RNG
  states, so the continued epoch uses registered seed 42 but is not a bitwise replay of the aborted
  in-memory epoch-4 stream. Trainer checkpoints now preserve Python, NumPy, CPU Torch, and CUDA RNG
  states for exact future interruption recovery.
- Added `docs/RECOVERY_V2_7_RESUME_RUNBOOK.md` plus pure provenance/history/checkpoint discovery and
  notebook-contract tests. The actual extracted epoch-3 metadata passes the new signature validator
  with expected step 628 and a `0.00000377` CSV-rounding difference.

## 2026-07-27 - existing-data bbox and weak-class improvement workflow

- Added a separate, CPU-only Kaggle workflow for improving the existing Web-Gold-40K export without
  touching `notebooks/kaggle_gold.ipynb`, extracting the attached ZIP, modifying source records, or
  reading the locked test split. The workflow streams only train/validation and records
  `test_rows_read=0`.
- Implemented a complete invalid-bbox review ledger and mask manifest. Every box is checked against
  its native `state_before` dimensions; invalid rows remain usable for every non-localization head
  while bbox loss stays masked. Because the exported input/label/meta schema has no verified
  document-to-viewport scroll offset, the system performs zero automatic bbox corrections and
  leaves replacement fields blank pending direct replay/screenshot evidence.
- Implemented deterministic trajectory-level review queues for `SCROLL`, `SELECT`, `NAVIGATE`,
  `LOOP_DETECTED`, `BACKTRACK`, `RETRY`, and `ABORT`. Complete trajectories stay together; 10% are
  assigned to both reviewers for agreement; recovery rows include failure state, executed recovery
  action/value, post-recovery state, and observed outcome.
- Added class-coverage evidence that audits existing nonzero classes first and requests targeted real
  collection only for zero-support recovery strategies that remain inside the registered thesis
  claim. In the current train/validation export, this applies to `RETRY` and `ABORT`; the workflow
  never duplicates or relabels data to fill them.
- Added reviewer montages, CSV/JSON outputs, a small downloadable review-package ZIP, an exact
  runbook, and synchronization with the existing manual-review guide. The raw 39,215-row dataset
  remains unchanged.
- Verification: focused audit/recovery suite `11 passed`; synthetic end-to-end nested-ZIP test
  confirms streaming, output creation, zero test rows, and no extraction. After rebasing onto the
  latest remote Kaggle evidence, the full repository suite is
  `57 passed / 22 skipped / 3 known failures`; the unchanged failures are preserved executed
  v2.3/v2.5/resume notebook-hygiene assertions and do not involve this workflow.
- Kaggle kernel `kiyasmahmud/web-gold-existing-data-improvement` version 3 completed on CPU.
  The accepted package has SHA-256
  `48534e7e815135b5530514887835be2b046da1efc1416c2d3d7124dc425e1770`; all 11 ZIP
  members pass integrity reading. The real output reconciles 2,534 unique invalid bbox rows
  (2,042 train + 492 validation) between the review CSV and mask manifest, with zero test rows,
  zero source mutation, and zero automatic correction.
- The bbox evidence is systematic: 2,132 rows have both bottom overflow and y-origin-outside,
  only 289 distinct invalid bbox values are repeated across 2,534 rows, and all affected
  `action_coordinates` are null. `SELECT` contributes 1,618 invalid rows, `TYPE` 552, and `CLICK`
  364. No exported scroll-offset evidence exists, so masking remains the only defensible automatic
  action.
- Existing validation support is SCROLL 1,293, SELECT 1,292, NAVIGATE 1,499, LOOP_DETECTED 240,
  and BACKTRACK 200. RETRY and ABORT remain zero-support classes requiring real targeted
  collection only if retained in the learned-strategy claim. The package contains 2,534 bbox rows
  and 1,977 trajectory-preserving weak-class rows; all human decision/correction fields remain
  intentionally blank.
- Added the separate interactive `kaggle_gold_manual_review.ipynb` workspace for the two human
  reviewers. It reads train/validation only, restores queue-specific immutable logs, skips already
  reviewed targets, validates native-image bbox proposals, requires correction evidence, shows
  causal recovery transitions, and leaves the source export untouched.
- Added review reconciliation that preserves both reviewers' events, measures raw agreement and
  Cohen's kappa, blocks unresolved/disagreeing decisions, and automatically creates a secondary
  task for the other reviewer whenever a singly assigned correction, rejection, quarantine, or
  ambiguity occurs. A reconciliation `PASS` permits only the controlled 5k improvement mini; it
  explicitly does not claim that all 39,215 rows are publication-ready.
- Current verification: the focused review/audit/overlay suite passes `28/28`; Ruff and
  `git diff --check` pass. The repository suite is
  `77 passed / 22 skipped / 3 known failures`. The same three
  pre-existing failures remain in executed v2.3, v2.5, and v2.7-resume notebook-hygiene checks and
  are outside this manual-review change.
- Added an evidence-checked runtime review overlay for the post-reconciliation controlled mini.
  It refuses incomplete/tampered reconciliation artifacts, verifies each original field before
  applying a correction, excludes confirmed reject/quarantine rows, validates corrected recovery
  tuples and bbox geometry, applies only to train/validation, reads zero test rows, and never
  rewrites the 39,215-row source. A separate CPU Kaggle notebook validates the overlay against the
  real mounted dataset before GPU work.
- Corrected the comparison contract for reviewed validation labels: historical v2.7 metrics are
  not directly comparable after corrections/exclusions. The selected v2.7 checkpoint must first
  be re-evaluated on the exact same checksummed overlay; the subsequent mini report records this
  requirement and provenance, and the CLI forbids a development overlay during locked-test eval.

## 2026-08-01 - DGX full-training workflow prepared
- Accepted the time-limited v2.8 mini only with an explicit protocol deviation: epochs 0-3
  completed and epoch 3 passed all eight registered gates; interrupted epoch 4 is not reported as
  a formal five-epoch completion. The first full seed remains Qwen2-VL-2B, seed 42.
- Added a real reviewed-Gold `full` stage. It trains all 24,107 combined train rows, selects only
  on all 7,861 original-Gold validation rows, reports all 194 RETRY/ABORT validation rows
  separately, retains every completed epoch checkpoint, exports CSV/JSON evidence and reads zero
  locked-test rows.
- Strengthened `Trainer` recovery for lab interruptions. `last.ckpt` now records the exact next
  physical batch, partial epoch accumulators, all model/loss/optimizer/scheduler/scaler states,
  Python/NumPy/Torch RNG, completed history, early-stopping state and checkpoint lineage. Resume
  no longer restarts an incomplete epoch from batch zero.
- Added `notebooks/dgx_gold_full_training.ipynb` and
  `docs/DGX_FULL_TRAINING_RUNBOOK.md`. They use immutable Hugging Face dataset revisions,
  persistent DGX storage, automatic same-run resume, per-epoch and cross-model CSVs, source-aware
  validation and optional explicit artifact upload.
- Added a causal Gold v2.8 candidate config for Qwen2.5-VL-3B. Its full run remains blocked until
  that backbone completes its own smoke and controlled mini. Dual-encoder, InternVL, ablation and
  baseline entries remain honestly blocked because their implementations are incomplete.
