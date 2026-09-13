# Final evaluation tables — current state

2026-09-12. Follows the agreed five-table structure. **Bold = measured and
verified. `XX` = not yet run.** No number here is estimated or projected.

---

## Table 1 — Component performance

Dataset Web-Gold-40K · unit = one interaction step · **currently validation
split, seed 42, one backbone**. The locked test portion has never been read.

| Model / Backbone | Failure macro-F1 | Failure MCC | Action macro-F1 | Recovery-strategy macro-F1 | Recovery-outcome MCC | BBox mean IoU | Recall@IoU50 | Memory macro-F1 † |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen2-VL-2B** | **0.809** | **0.624** | **0.327** | **0.389** | **0.796** | **0.101** | **0.081** | **0.830** |
| Qwen2.5-VL-7B | XX | XX | XX | XX | XX | XX | XX | XX |
| InternVL3-8B | XX | XX | XX | XX | XX | XX | XX | XX |
| **Selected model** | XX | XX | XX | XX | XX | XX | XX | XX |

Majority baselines on the same split: failure-type 0.147 · outcome acc 0.584 ·
action macro-F1 0.053 · recovery-strategy acc 0.764 · memory macro-F1 0.383.

**† The `Memory Recall@1/@3/@5` columns in the original design cannot be filled.**
The trained memory head is a **binary store / don't-store classifier**, not a
retriever. Recall@k needs relevance labels and a retrieval evaluation that does
not exist. Either build that evaluation or replace those three columns with the
storage metrics above (macro-F1 0.830, MCC 0.670, accuracy 0.843).

**Still required:** test-portion results · 3 seeds for mean ± std · two further
backbones.

---

## Table 2 — End-to-end agent performance ✅ complete

Controlled browser environment · BrowserGym + Playwright · MiniWoB · six task
families × five resets × four systems = **120 episodes** · unit = complete
episode · success = terminated, not truncated, raw reward exactly 1.0 · audit
PASS, no exclusions.

| Agent configuration | Task SR | Recovery SR | Success after initial failure | Avg steps | Loop rate | Unrecovered failure rate |
|---|---:|---:|---:|---:|---:|---:|
| Base agent (E0) | **20.0%** | N/A | N/A | **1.00** | **0.0%** | **0.0%** |
| Base + failure detection (E1) | **0.0%** | N/A | N/A | **3.00** | **100.0%** | **0.0%** |
| Base + failure detection + recovery (E2) | **20.0%** | **20.0%** | **20.0%** | **1.87** | **0.0%** | **80.0%** |
| **Complete agent + recovery + memory (E3)** | **20.0%** | **20.0%** | **20.0%** | **1.93** | **0.0%** | **80.0%** |

Supporting counts: E2 executed **26 recovery actions across 38 attempts**; E3
executed **28 across 46 attempts** with 46 memory queries. All 30 E2/E3 episodes
raised a failure incident. All 30 E1 episodes terminated on the loop guard.

Paired contrasts: **E2 − E1 = +0.20** (6 improved, 0 worsened, exact p = 0.031,
Holm p = 0.0625 — *not* significant at the adjusted 5% level).
**E3 − E2 = 0.00** (p = 1.0).

Per-family: only `focus-text` (12 successes) and `click-link` (6) ever succeed.
`click-option`, `click-checkboxes`, `enter-password` and `login-user` are **0/20
for every system**.

---

## Table 3 — Comparison with related web agents

Drafted in full from published sources — see
[the Table 3 draft](TABLE3_RELATED_WEB_AGENTS_DRAFT.md). Headline rows:

| Agent | Capability | Benchmark | Task SR | Recovery result | Comparison |
|---|---|---|---:|---|---|
| BacktrackAgent | Error detection + backtracking | Auto-UI | 29.72 | **+8.16** over same-backbone SFT | Contextual |
| WebCoach | Cross-session memory | WebVoyager | 0.614 | **+0.141** over no memory | Contextual |
| SkillWeaver | Reusable skills | WebArena | 29.8 | +32% relative | Contextual |
| ReUseIt | Reusable workflows | Own tasks | 70.1 ± 16.4 | 24.2 → 70.1 | Contextual |
| **Our WebAgent** | **Failure recovery + memory** | Our MiniWoB protocol | **20.0%** | **E2−E1 +0.20 (n.s.); E3−E2 0.00** | Proposed |

**Companion — failure detection (separate metric, do not merge):**

| System | Failure-detection accuracy |
|---|---:|
| GPT-4o (textual grounding) | 83.79 |
| InternVL2-Llama3-**76B** | 82.36 |
| Gemini-1.5-flash | 80.90 |
| **Ours — Qwen2-VL-2B** | **80.4** (MCC 0.624) |
| Claude 3.5 Sonnet | 76.07 |

This is the strongest comparison in the paper: **a 2B model in the same
accuracy band as GPT-4o- and 76B-scale pipelines.**

---

## Table 4 — Agents using our data

Another open-source agent trained on the **same Web-Gold-40K learning portion**
and evaluated on the **same unseen test portion**, via an agent-specific adapter.

| Agent | Dataset | Action macro-F1 | BBox mean IoU | Recall@IoU50 | Failure MCC | Recovery macro-F1 | Recovery-outcome MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| Other open-source agent | Web-Gold-40K | XX | XX | XX | N/R | N/R | N/R |
| **Our WebAgent** | Web-Gold-40K | XX | XX | XX | XX | XX | XX |

**Not started.** Only shared metrics decide which agent performs better; having
extra heads does not make ours superior.

---

## Table 5 — Agents using other data

Our architecture on the official **Multimodal Mind2Web** release, using its
cross-task / cross-website / cross-domain test portions.

| Agent | Dataset | Action score | Element grounding | Cross-task | Cross-website | Cross-domain |
|---|---|---:|---:|---:|---:|---:|
| Other open-source agent | Multimodal Mind2Web | XX | XX | XX | XX | XX |
| **Our WebAgent** | Multimodal Mind2Web | XX | XX | XX | XX | XX |

**Not started — and this is now the highest-value remaining experiment.**
Mind2Web is the *only* benchmark where our numbers become **directly**
comparable with published work (CogAgent, SeeClick, Qwen-VL, MindAct,
AutoWebGLM, AWM+WMA all report on it). Everywhere else the comparison is
contextual. Our dataset is Mind2Web-derived, so the transfer is natural.

---

## Status summary

| Table | State | Remaining work |
|---|---|---|
| 1 | 8/8 metric columns filled for one backbone, validation only | test split · 3 seeds · 2 backbones · resolve memory-recall metric |
| **2** | ✅ **complete, audited** | none |
| 3 | ✅ drafted from literature | fill `N/R` cells from the papers |
| 4 | not started | adapter + 1 training run |
| 5 | not started | Mind2Web data + 1–2 training runs |

## Three things that must be stated in the paper, not discovered by a reviewer

1. **Grounding does not work, and it is a data limitation.** BBox median IoU is
   **0.000 at every epoch** on our own validation. The instruction field never
   names the target element, `action_target_desc` is absent from the exported
   data, 19% of boxes fall outside their own screenshot, and the median target
   occupies 0.69% of the page. No model can learn this task as posed.
2. **The end-to-end gain is zero, and (1) is why.** Every competing paper
   reports a positive gain from its mechanism. Ours does not, because an agent
   that cannot point cannot complete tasks however well it diagnoses failure.
3. **Recovery-strategy accuracy (0.535) is below the majority baseline (0.764).**
   Its macro-F1 clears the registered gate, which is why it was not caught
   earlier. Report it.

## What the paper can claim today

- Failure detection at **MCC 0.624 / 80.4% balanced accuracy** from a 2B model,
  in the band of GPT-4o-scale pipelines.
- Failure-type classification at **macro-F1 0.502** against a 0.147 baseline.
- Recovery-outcome prediction at **MCC 0.796**.
- Memory storage decisions at **macro-F1 0.830** against 0.383.
- A complete, audited end-to-end study with **honest negative results** and a
  documented cause.

## What it cannot claim yet

- Any backbone-agnostic property — one backbone, one seed.
- Any generalisation — the locked test split is unread.
- That the unified four-pillar design helps — A4/A5 ablations never run.
- Superiority over any external agent — no matched external baseline exists.
