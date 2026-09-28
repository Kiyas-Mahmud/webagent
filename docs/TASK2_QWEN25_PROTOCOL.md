# Task 2 with Qwen2.5-VL-7B — protocol `task2-qwen25-v1`

Decided 2026-09-27, before any Qwen Task 2 browser outcome was observed.
Config: [`configs/eval/task2/qwen25_v1.json`](../configs/eval/task2/qwen25_v1.json).
Supersedes the InternVL `task2-native-evaluation-v1` plan (stopped at 3/90 by a
host power-off; see its `stop-receipt-20260927.json`). Earlier Task 2 evidence
stays separate and is never pooled.

## Question

Does adding our trained failure/recovery assessment (B) and frozen memory (C)
to a public browser agent (A) raise task completion under matched conditions?

| System | Actor | Added |
|---|---|---|
| A | Native Browser Use 0.13.10 + Qwen2.5-VL-7B base (frozen, 4-bit, greedy) | — |
| B | same | Trained Qwen2.5-VL-7B heads (PC-02 epoch 0, seed 42): outcome, failure type, strategy, recovery outcome → advisory feedback |
| C | same | B + frozen 1,974-item label memory, queried with the original PC-01 encoder |

Browser Use chooses and grounds every action (element indices), so our weak
grounding head is not on the path. Primary contrasts: **B−A** and **C−B**.

## Why the task families changed

In all saved v1 and development episodes the base agent finished in one step
(`click-link`, `click-button`, `enter-text`...). With no failure to recover
from, B−A is zero by construction. The new protocol measures where the base
agent actually fails.

## Phases

1. **Headroom screen** — A only, 15 harder families × 5 reset seeds
   (key `task2-qwen25-v1-screen`).
2. **Selection (frozen rule, applied mechanically):** keep families where A
   completed some but not all eligible resets; if more than 10, keep those
   closest to 50%, ties by name; if fewer than 6, add 0-success families in name
   order. All-success families are never kept.
3. **Development** — A/B/C, selected families × 1 new reset
   (key `task2-qwen25-v1-development`). Audit PASS required; only
   implementation defects may be fixed, and a fix means a new version.
4. **Evaluation** — A/B/C, selected families × ⌈100 / families⌉ resets
   (key `task2-qwen25-v1-evaluation`), seeds checked disjoint from phases 1–3.

## Fixed settings

- Budgets per episode: 30 executor requests, 102 model calls (actor,
  assessment, memory query and memory generation all count), recovery at most
  2 per incident and 4 per episode. Wall clock 1,800 s is a safety cap only, so
  assessment latency cannot be what ends a B/C episode; latency is reported.
- Completion: MiniWoB terminated, not truncated/invalid, raw reward exactly 1.0.
  Reward never reaches the actor or heads.
- Assessment rule unchanged: outcome argmax; recovery outcome logit > 0.
  Outcome probabilities are logged, not used.
- Analysis: exact two-sided discordant-pair test, Holm over the two contrasts,
  95% bootstrap (10,000 resamples within family, seed 20250831).

## Claim boundaries

- Selecting families by A's screen outcome is a declared design step on
  screen-only seeds; evaluation seeds are new. It raises headroom, not B.
- B−A measures the added package (trained assessment + extra feedback turn);
  there is no generic-assessor control.
- MiniWoB is out of domain for heads trained on Mind2Web-derived pages.
- Browser Use puts a random Chromium-derived tab id in every prompt. It was the
  only A/B prompt difference in smoke v3 and flipped a greedy decision. The noise
  is exchangeable across systems, so the paired test stays valid; it costs power.
  The native prompt is not edited.
- In smoke v3, C queried memory 5 times and admitted 0 of 15 candidates
  (12 backtrack examples without in-episode navigation, 3 without a stored
  corrective value). If this persists, C−B is structurally near zero. The frozen
  retrieval/admission rules are not relaxed after observation; exposure is
  reported with every C result.

## Interface alignment with upstream Browser Use 0.13.10

Found in live engineering smokes, fixed before any screen outcome, each pinned
by a regression test:

| Smoke | Finding | Change |
|---|---|---|
| v1 | Whole-response ```` ```json ```` fence rejected | Upstream Ollama `_JSON_FENCE_RE` rule; body never edited |
| v2 | Host OOM freeze: allocator kept ~12 GB per call | `expandable_segments`, `empty_cache()` per op, 16 GB worker guard, 12 GB supervisor kill |
| v3 | Empty action list rejected locally | Left to upstream's one clarification retry, then its `done(success=False)` noop |
- Null or negative results are reported as measured; no re-running until an
  improvement appears.

## Operation

```bash
# freeze (main env), then run under the shutdown inhibitor
PYTHONPATH=src .venv/bin/python -m web_agent.eval.task2.campaign_v2 freeze --phase screen --out .task2-assets/task2-qwen25-v1-screen
PYTHONPATH=src .venv/bin/python scripts/analysis/task2_supervise_v2.py .task2-assets/task2-qwen25-v1-screen
# development / evaluation take --previous <audited earlier phase>
cat .task2-assets/<run>/status.json   # includes coordinator pid
```
