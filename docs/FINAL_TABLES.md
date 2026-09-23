# Final evaluation tables — current state

2026-09-22. Follows the agreed five-table structure. **Bold = measured and
verified. `XX` = not yet run.** No number here is estimated or projected.

---

## Table 1 — Three-backbone component comparison

Web-Gold-40K original validation split · seed 42 · 7,861 interaction steps ·
1,858 attempted-recovery cases · 2,349 grounded-action cases. The locked test
portion has never been read. Arrows indicate the preferred direction; bold
marks the best value in each column.

| Backbone | Epoch | Outcome MCC ↑ | Failure macro-F1 ↑ | Failure-type macro-F1 ↑ | Action macro-F1 ↑ | Recovery-strategy macro-F1 ↑ | Recovery MCC ↑ | Recovery macro-F1 ↑ | Memory MCC ↑ | Memory macro-F1 ↑ | BBox mIoU ↑ | R@IoU50 ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2-VL-2B | 6 | 0.624 | 0.809 | 0.502 | 0.327 | 0.469 | 0.796 | 0.898 | 0.663 | 0.830 | 0.101 | 0.081 | 0.163 |
| **Qwen2.5-VL-7B (selected)** | **0** | **0.678** | **0.839** | **0.542** | 0.318 | **0.493** | **0.852** | **0.925** | **0.695** | **0.846** | **0.155** | 0.083 | 0.072 |
| InternVL3.5-8B-HF | 0 | 0.641 | 0.820 | 0.541 | **0.337** | 0.489 | 0.840 | 0.920 | 0.654 | 0.821 | 0.092 | **0.095** | **0.040** |

Qwen2.5-VL-7B is selected by the registered primary rule: highest validation
outcome MCC after the per-run quality gates. It also leads recovery-outcome MCC
and macro-F1, memory MCC and macro-F1, and bounding-box mean IoU.
InternVL3.5-8B-HF leads action macro-F1, Recall@IoU50 and calibration, but those
are not the primary selection metric.

Majority baselines on the same split: failure-type macro-F1 0.147 · outcome
accuracy 0.584 · action macro-F1 0.053 · recovery-strategy macro-F1 0.233 ·
memory macro-F1 0.383.

**† The `Memory Recall@1/@3/@5` columns in the original design cannot be filled.**
The trained memory head is a **binary store / don't-store classifier**, not a
retriever. Recall@k needs relevance labels and a retrieval evaluation that does
not exist. Table 1 therefore reports the selected model's storage-decision
metrics instead (macro-F1 0.846, MCC 0.695, accuracy 0.858).

**Evidence boundary:** these are single-seed validation results, not locked-test
results or mean ± standard deviation over multiple seeds. The immutable run
contracts record different source commits. The numerical ranking is therefore
the validation-selected order, while a claim of a fully commit-matched
comparison remains blocked by the provenance audit.

Paper-ready details and exact values are in
[the Table 1 comparison record](TABLE1_THREE_MODEL_COMPARISON.md).

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
| 1 | Three-backbone, seed-42 validation comparison complete; PC2 selected descriptively | locked test · multiple seeds · reconcile source-commit provenance |
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

- Qwen2.5-VL-7B is the validation-selected backbone at **outcome MCC 0.678**
  under the registered primary ranking rule.
- Its failure-type classification reaches **macro-F1 0.542** against a 0.147
  majority baseline.
- Its recorded recovery-outcome prediction reaches **MCC 0.852**.
- Its memory-storage decisions reach **MCC 0.695 / macro-F1 0.846**; these are
  storage-label metrics, not retrieval or live-task success.
- A complete, audited end-to-end study with **honest negative results** and a
  documented cause.

## What it cannot claim yet

- Multi-seed robustness — every backbone currently has only seed 42.
- A fully source-matched three-backbone comparison — the immutable run
  contracts record different Git commits.
- Any generalisation — the locked test split is unread.
- That the unified four-pillar design helps — A4/A5 ablations never run.
- Superiority over any external agent — no matched external baseline exists.
