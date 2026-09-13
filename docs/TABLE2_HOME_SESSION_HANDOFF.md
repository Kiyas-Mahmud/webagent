# Table 2: resume at home after the lab pause

Prepared 2026-09-13. Branch: `table2/lab-handoff-20260913`.
This is a work-in-progress archive, not a claim that the agent is fixed or a new
final evaluation is ready. The user requested stopping research work and pushing
the Table 2 work for continuation at home. No model/browser processes appeared
in the process inspection at handoff. Recheck before any later lab inference.

## Read first

1. This handoff and [the monitored TODO](TABLE2_HYBRID_AGENT_TODO.md).
2. [Latest six-action diagnostics](TABLE2_SIX_ACTION_CONTEXT_DIAGNOSTIC.md).
3. [Training/runtime alignment](TABLE2_TRAINING_RUNTIME_ALIGNMENT_AUDIT.md).
4. [All ten checkpoint results](TABLE2_ALL_CHECKPOINT_ACTION_AUDIT.md).
5. [Hybrid interface v3](TABLE2_HYBRID_INTERFACE_V3.md) and
   [hybrid implementation plan](TABLE2_HYBRID_AGENT_IMPLEMENTATION_PLAN.md).

Historical banners and unfinished checkboxes can describe earlier sessions.
Use the latest dated evidence; do not repeat already completed diagnostics.

## Aim and four pillars

Test whether executed diagnosis/recovery improves browser completion, and
whether frozen training-only corrective memory improves on recovery. Preserve
P1 learned failure diagnosis, strategy and assessment; P2 causal multimodal
observations; P3 six-action selection, grounding and actual execution; P4
training-only retrieval and auditable memory intervention.

Original E0–E3 and hybrid H0–H3 are different studies. In the current hybrid,
the frozen unadapted Qwen base generates executable actions; PC-01 epoch 6
supplies policy advice, diagnosis, assessment and memory-query representations.
Do not call base-generated CLICK proposals outputs of the trained action head.
The parameter fallback is also the separate unadapted base, not PC-01's trained
decoder. Playwright is already used through BrowserGym + MiniWoB.

## Completed evidence and current blocker

- Historical E interface-v2 final: E0 6/30, E1 0/30, E2 6/30, E3 6/30.
- Historical hybrid final: H0 8/30; H1/H2/H3 9/30 each. No primary recovery or
  memory increment was demonstrated.
- Latest hybrid development v3: 24 episodes, H0/H1/H2/H3 2/6 each; audit PASS.
  456 focused tests passed previously. All six executor actions were exercised
  in engineering fixtures. These are not six demonstrated model skills.
- Latest context diagnostic: 18 generated actions all CLICK, including contexts
  without trained advice or history. The compact six-action candidate generated
  some SCROLL/NAVIGATE but invalid arguments. Four parameter fallbacks explicitly
  rejected. Together: 28 generations + six trained forwards; no browser episodes.
- A preceding nine-generation experiment verbalized text entry, but its second
  generation still chose CLICK. A verbal plan is not an executed TYPE.
- No candidate from these diagnostics was promoted. No further inference or
  native-text-command parser was implemented after those experiments.

The remaining action problem is model-chosen action/target/value consistency
and live multi-step continuation. Do not fix it by inserting the task's correct
TYPE, silently moving predicted boxes or manufacturing successful outcomes.

## New source/export finding: partially investigated, not a completed audit

The exact local original and supplement train JSON files were checked against
the existing `training_action_value_evidence.json` hashes. All 24,107 rows lack
`action_value`. Original rows have 22 direct fields across inputs/labels/meta;
supplement rows have 25. The claim that these are simply the old specification's
23 fields is not established.

Crucially, training-task replay files exist locally under:
`/home/aiub/kiyas/webagent_full/data/original/final_data_set_40k/replays/`.
Several replays selected by task IDs from the authenticated train split were
inspected without opening images or test splits. TYPE has `action.text_to_type`,
SELECT has `action.option_value`, SCROLL has `action.scroll_amount`, and PRESS_KEY
has `action.key`. A NAVIGATE example was actually `navigate_back`, so do not
invent a destination URL or conflate browser back with URL navigation.

This establishes omitted action-specific source information in sampled training
replays. It does NOT establish all-row coverage, that all 63 collector fields
were populated, or the exact exporter script responsible. The collector/exporter
source was not found in the inspected repository/local source locations.
Do not read mixed `gold_audit*.jsonl` files indiscriminately: they may include
locked-test tasks. Use a train-only task allow-list and exact task/step matching.

Training has class/box objectives, not an autoregressive action-value target.
Restoring export fields cannot retroactively train the frozen checkpoint or
explain every independent failure of the unadapted base generator. Values may
still occur in goal text; absence of a dedicated field does not prove otherwise.

Potential next data task: finish train-only replay/export matching, count
recoverable arguments and verify provenance. Any recovered memory content needs
a new declared material binding; preserve the old 1,974 vectors/store/archive.
No enrichment, re-export, embedding regeneration or training happened here.

## Epoch 0 proposal: assessed, not executed or approved for production

The user asked whether switching epoch 6 to epoch 0 and enabling executable
action restriction would solve the problem. Our recommendation was a bounded
development comparison, not a production replacement.

- Epoch 0 TYPE recall 70.06% vs epoch 6 48.46%, but NAVIGATE recall 0% and
  SCROLL 1.86%; overall action accuracy is lower. No epoch is reliable on all six.
- The restriction is a union across current page controls, not a chosen-textbox
  filter. CLICK may still win. With epoch 6 it was already tested in development
  v7: E1 0/4, with incorrect boxes; see `TABLE2_E1_DIAGNOSIS.md`.
- Loading epoch 0 requires its own authenticated checkpoint bindings, not just
  editing a path under epoch 6's manifest.
- The frozen memory vectors, threshold and runtime query identity belong to
  epoch 6. Do not query them with epoch 0 and bypass the mismatch check. Retaining
  epoch 6 for diagnosis/query and using epoch 0 only for policy would be a
  separately declared multi-checkpoint method, not an unchanged E/H system.
- No retraining is required for a diagnostic, but model inference needs GPU
  resources. One successful `enter-text` episode would not validate P1/P4 or
  justify immediately launching a full Table 2 study.

No checkpoint switch, restriction change, new selection gate, planner download
or new final evaluation has been performed. A stronger planner download was
previously proposed but not approved. Do not assume authorization for it.

## Files available at home versus only in the lab

Git includes Table 2 runtime/source, configs/prompts, tests, scripts, reports,
the checkpoint-series aggregate audit and compact latest diagnostic summaries
under `docs/evidence/lab-handoff-20260913/`. Those summaries are copied unchanged;
the adjacent manifest binds each source path and SHA-256. They reference other
lab-only files and are not a standalone replayable experiment bundle.

Lab assets remain at the locations in `TABLE2_AGENT_SOLVER_HANDOFF.md` and the
configs, including:

- `/home/aiub/kiyas/table2-evidence/`: immutable full studies, raw observations,
  screenshots, producing source snapshots and the P4 store/material.
- `/home/aiub/kiyas/table2-inputs/`: pinned Qwen base, PC-01 v3 export, browser
  binaries, MiniWoB source and the transferred results ZIP.
- `/home/aiub/kiyas/webagent_comparison/outputs/`: saved training checkpoints.
- `/home/aiub/kiyas/webagent_full/data/`: train JSONs and original replay files.

No weights, dataset rows, Gold images or full evidence archives are pushed.
The training notebook's local execution outputs and `.vscode/` settings are
preserved on the lab PC, outside this Table 2 commit. The tracked notebook source
remains available in Git. Laptop CUDA detection alone does not prove inference
will fit; the earlier unchanged model construction ran out of GPU memory.

## Home-session instructions

Inspect branch, local changes and available files before doing anything. Work
in the existing repository. Fetch and switch to this branch without overwriting
local edits. Start with the documented evidence and source, not another final
run. Finish the replay/export audit if the train-only replay files are available;
otherwise identify precisely which lab inputs are needed and continue source
analysis/CPU checks. Prepare any epoch comparison as a separate development
profile before lab inference, preserving all original artifact checks.

No retraining, package upgrades, Gold-image or locked-test access, dataset
re-review, fabricated memory, unrelated pushes, or interference with other jobs.
Document exact causes, actual changes, checks and unresolved stages. Do not
promise completion or memory improvement before measurement.
