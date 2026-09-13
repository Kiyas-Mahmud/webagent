# Q1 paper table re-plan — 2026-09-12

Nothing is removed. Every experiment already run is kept and reported. What
changes is which result is the **headline** and which is a **limitation study**,
so the paper leads with what the evidence actually supports.

## 1. Why re-plan

The original Table 2 asks: *does executed recovery increase browser task
completion?* The measured answer is flat — E0 6/30, E1 0/30, E2 6/30, E3 6/30 —
and the training report explains why. The grounding head has **median IoU 0.000
at every epoch on its own validation set**, with recall@IoU 0.5 of 0.081. An
agent that cannot point at a target cannot complete browser tasks, no matter how
good its failure diagnosis is.

Meanwhile the two pillars that *did* train well were never given a headline
table. That is the mismatch this re-plan fixes.

## 2. What the training evidence actually supports

| Head | Score | Majority | Margin | Claimable |
|---|---:|---:|---:|---|
| Outcome (failure detection) MCC | 0.624 | — | — | ✅ |
| Outcome balanced accuracy | 0.804 | 0.584 | +0.220 | ✅ |
| Failure-type macro-F1 | 0.502 | 0.147 | +0.355 | ✅ |
| Recovery-outcome MCC | 0.796 | — | — | ✅ |
| Memory macro-F1 | 0.830 | 0.383 | +0.447 | ✅ |
| Memory accuracy | 0.843 | 0.620 | +0.223 | ✅ |
| Confidence MAE | 0.044 | — | — | ✅ |
| Needs-recovery macro-F1 | 0.577 | 0.433 | +0.144 | 🟡 |
| Attempted-strategy macro-F1 | 0.469 | 0.233 | +0.236 | 🟡 |
| Action-type macro-F1 | 0.327 | 0.053 | +0.274 | 🟡 |
| Recovery-strategy accuracy | 0.535 | **0.764** | **−0.229** | ❌ |
| bbox median IoU | **0.000** | — | — | ❌ |
| bbox recall @ IoU 0.5 | 0.081 | — | — | ❌ |

## 3. New table structure

| Table | Content | Status |
|---|---|---|
| **1** | Corpus: Gold v2.8, 24,107 train rows, six balanced action classes, failure-type distribution | ✅ have |
| **2 — NEW HEADLINE** | **Failure-awareness and memory** vs baselines and ablations | ⛔ needs work |
| **3** | Action selection and grounding — *including* the bbox failure, stated plainly | ⛔ needs same rows |
| **4** | Backbone-agnostic comparison PC-01 / PC-02 / PC-03 | ⛔ only PC-01 exists |
| **5 — was Table 2** | End-to-end browser agent E0–E3 | ✅ have, reframed |

### Table 2 (new headline) — proposed layout

| System | Outcome MCC | Outcome bal-acc | Fail-type macro-F1 | Needs-recovery macro-F1 | Memory macro-F1 | Conf MAE |
|---|---|---|---|---|---|---|
| B1 Majority / Random | — | 0.584 | 0.147 | 0.433 | 0.383 | — |
| B2 MindAct (DeBERTa+Flan-T5) | ? | ? | ? | ? | ? | ? |
| B3 CLIP + MLP | ? | ? | ? | ? | ? | ? |
| B4 LayoutLMv3 | ? | ? | ? | ? | ? | ? |
| A4 Ours − memory head | ? | ? | ? | ? | n/a | ? |
| A5 Ours − recovery supervision | ? | ? | ? | n/a | ? | ? |
| **Ours (PC-01, 4 pillars)** | **0.624** | **0.804** | **0.502** | **0.577** | **0.830** | **0.044** |

Only the last row exists today.

### Table 5 (was Table 2) — kept, reframed

Reported exactly as measured, as a **feasibility and limitation study**:
E0 6/30 · E1 0/30 · E2 6/30 · E3 6/30; E2−E1 = +0.20 (6 improved, 0 worsened,
exact p = 0.031, Holm 0.0625); E3−E2 = 0. Plus the family breakdown — only
`focus-text` (12) and `click-link` (6) ever succeed; `click-option`,
`click-checkboxes`, `enter-password` and `login-user` are 0/20 for every system.

Framed honestly this *strengthens* the paper: it identifies precisely why strong
failure detection does not yet translate into task completion, and names the
grounding head as the bottleneck with in-domain evidence.

## 4. What is missing before "better" can be claimed

Today the only comparison is **one seed, one backbone, validation split, against
a majority baseline**. For Q1 that is not sufficient, and a reviewer will say so.

| Gap | Current state | Needed |
|---|---|---|
| **Locked test split** | `test_rows_read: 0` — never touched | Run once, at the very end |
| **Seed variance** | seed 42 only | ≥3 seeds → mean ± std |
| **Own ablations** | none run | A4 (no-memory), A5 (no-recovery) |
| **External baselines** | none run | B2 MindAct at minimum |
| **Backbone-agnostic** | PC-01 only | PC-02, PC-03 |
| **Training data** | not on this machine | Gold v2.8 must be re-acquired from Kaggle |

The locked test split being untouched is **good discipline** — it is still clean,
and it is the number the paper should report. It must be read exactly once.

## 5. Recommended order

**Phase A — prove the pillars contribute (cheapest, highest value).**
Train A4 (no-memory) and A5 (no-recovery) at seed 42. These directly test the
thesis's own claim that the unified design helps. Both reuse PC-01's code with
one thing removed. *If A4 and A5 do not score lower than PC-01, the unified-design
claim is not supported and the paper must say so.*

**Phase B — add seed variance.** PC-01 at two further seeds → mean ± std on the
headline metrics. Without this there is no uncertainty estimate at all.

**Phase C — one external baseline.** B2 MindAct is the Mind2Web-relevant
comparison and the one reviewers will look for.

**Phase D — backbone-agnostic.** PC-02 and PC-03, which is the stated novelty in
`AGENT.md` and currently has a single data point.

**Phase E — locked test, once.** Freeze everything, read the test split one time,
report.

Grounding (P3) is a separate track. If the bbox head can be fixed, Table 5 comes
alive; if not, it stays a documented limitation. Either way it must not block
Phases A–E, which do not depend on it.

## 6. Hard constraints carried forward

- **PC-01 must not be retrained or substituted.** The frozen Table 2/5 browser
  evidence is hash-bound to that exact checkpoint.
- No result is deleted. Development-v3/v6/v7 and the 120-episode evaluation all
  remain in the record, including the rejected `executable_action_selection`
  hypothesis.
- The locked test split stays unread until Phase E.
- The recovery-strategy accuracy deficit (0.535 vs 0.764 majority) is reported,
  not hidden. Its macro-F1 clears the registered gate, which is why it was not
  caught earlier.
