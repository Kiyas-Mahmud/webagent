# InternVL dual comparison

Start with `result_matrix.csv`. Detailed phase metrics and nine declared paired contrasts are in the other CSV files.

All rates are fractions, not percentages. MCC/F1/balanced accuracy use valid outputs; all-case accuracy counts missing/invalid outputs as incorrect. Empty metrics mean unavailable, not zero.

This is a follow-up on previously observed validation cases, not a new locked test. The six prompted configurations are adapted component instructions, not native browser agents. Base-versus-trained decoder contrasts hold the prompt and generation settings fixed; head-versus-decoder contrasts also differ in assessment interface. No live task completion, executed recovery, memory gain or confirmatory significance claim is established.

Producing report: /home/aiub/kiyas/webagent/.task1-assets/runs/task1-internvl-dual-v2/report


## Interruption and retry accounting

The original run stopped after 1,624 saved predictions, with one additional request started but unfinished. The user explicitly authorized one audited retry. The reviewed continuation copied all 1,624 predictions byte-for-byte and generated 896 additional responses: 895 previously unstarted requests plus the authorized retry. The retry abstained. Its original interrupted attempt remains in the original archive. There are **2,520 final predictions and 2,521 unique attempts**, including that interruption. No completed, malformed or abstaining response was rerun.

`execution_summary.csv` reports the interruption/retry, coverage and known inference costs. The original interrupted attempt's call count and latency are unknown. Cost totals exclude model loading, development and engineering checks. `audit_summary.json` records preservation and independent metric checks.

## Coverage and uncertainty

Prompted interaction coverage ranges from 2/240 to 68/240; prompted recovery coverage ranges from 2/120 to 49/120. Conditional-valid metrics cannot be read as scores over the entire selection. Head coverage is 240/240 and 120/120. Common-valid base/trained pair counts are only 2, 7 and 11 for interaction, and 1, 4 and 9 for recovery (Browser Use, Agent S2, WebVoyager order). Degenerate [0,0] bootstrap intervals in tiny or single-class subsets do not establish equivalence or precise uncertainty. These are descriptive follow-up results.
