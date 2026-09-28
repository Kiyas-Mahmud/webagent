# Pillar 4 — experience memory for the Browser Use agent

2026-09-27. Development of the agent system; no evaluation claim.

## Why the previous memory never helped

The Task 2 C system queried a frozen store of 1,974 training recovery examples
(`p4-local-label-memory-v1`, PC-01 embeddings). In every live run, 0 examples
reached the agent. The cause is structural:

| Problem | Evidence |
|---|---|
| No corrective values | `recovery_action_value` empty for all 1,974; `reflection_text` empty for all |
| Admission rejects valueless non-click recoveries | 1,836 / 1,974 (93%) always excluded as `CORRECTIVE_ARGUMENT_UNAVAILABLE` |
| Backtrack items need in-episode navigation | 432 BACKTRACK items, never admissible on single-page MiniWoB |
| No element information | Items say `CLICK`, not what was clicked — even admitted advice would be empty |
| Different model | Embedded by the PC-01 Qwen2-VL-2B encoder, not the selected Qwen2.5 model |
| Read-only | `write_enabled = False`: the agent never stores its own experience |

The thesis defines P4 as "store experience + retrieve past recoveries". That
loop did not exist.

## Design

Module: `src/web_agent/memory/experience.py`; live wiring in
`src/web_agent/eval/task2/live.py` (system C with `memory_mode="experience"`).

```
P1 flags FAILURE ──► open incident
   │                  key   = Qwen2.5 memory embedding of the failed transition
   │                  store = sigmoid(memory head) — trained store/don't-store
   │                  record: goal, page URL, failed action + element hit,
   │                          failure type, strategy
   ├─ query memory ─► same page · top-3 by cosine · ≥ threshold · element visible
   │                  → concrete advice to Browser Use ("retrieved_past_experiences")
   ├─ each recovery ► append action, element, value, P1's assessed outcome
   └─ incident closes (P1 says SUCCESS → resolved; budget spent → not resolved)
                     └─ write if P(store) > 0.5   (resolved AND unresolved kept)
```

- **Element information** comes from Browser Use's own `interacted_element`
  (tag, visible text, id/name/type/placeholder/aria-label/role/title/href).
- **Unresolved incidents are kept**: "this did not work" is failure-aware memory.
- **No evaluator input**: outcomes are P1's learned assessments; MiniWoB reward
  never reaches memory.
- **Same checkpoint as P1**: the store refuses vectors from another checkpoint
  (`MEMORY_EMBEDDING_SPACE_MISMATCH`).
- **Read-only mode** (`write: false`) for any later evaluation policy.

## Threshold calibration and a finding

`scripts/memory/build_qwen25_memory_calibration.py` embedded the 1,974 training
transitions with the Qwen2.5 memory embedding
(`/home/aiub/kiyas/table2-evidence/p4-qwen25-embeddings-v1`, 26 min).
Leave-source-task-out top-1; relevant = same strategy **and** same recovery
action type; max F1 → **threshold 0.839**.

| Measure | Value |
|---|---:|
| Median top-1 cosine | 0.989 |
| AUC of similarity for relevance | 0.60 |
| Precision at thresholds 0.80–0.92 | 0.848–0.851 (flat) |
| Memory head P(store) > 0.5 on training failures | 64% |

The memory embedding was trained for the store decision, not retrieval, and
separates relevant neighbours only weakly. Therefore retrieval is **scoped to
the same observable page** (URL host + path), ranked by the embedding, and must
show a recorded element on the current page. The memory head is selective and
is used as trained.

## Known dependency on P1

On `click-checkboxes-soft`, P1 labelled nearly every checkbox click FAILURE
(P ≈ 0.97–1.00), including correct partial progress. Every such step opens an
incident, so memory contents inherit P1's out-of-domain errors. The memory
head's store decision is the only filter.

## Live verification (development demos, not evaluation)

| Demo | Written | Admitted → shown | Completed |
|---|---:|---|---:|
| `click-checkboxes-soft`, seeds 7 8 9 | 5 | 1 (a record with no element; now excluded as `NO_RECORDED_ELEMENT`) | 1/3 |
| `login-user`, seeds 1 2 3 (fresh store) | 3 | ep2: 1 → shown; ep3: 2 → shown | 3/3 |

What works: every incident was written (P(store) 0.77–1.00), retrieved on the
same page in later episodes, admitted, and shown to Browser Use as concrete
advice (`#username`, `#password` with the typed values, the `Login` button).

What limits it:

- **P1 mislabels on MiniWoB.** On `login-user` the correct username entry was
  judged FAILURE (P 0.92–0.94) and each correct next step FAILURE, so the stored
  experience reads "not resolved" for a sequence that completed the task. Memory
  content is only as good as P1's verdicts. The actor's action was unchanged by
  this advice here.
- **Randomised pages do not transfer.** `click-checkboxes-soft` changes the words
  on every reset; element-level advice from another reset would point at the
  wrong word, so it is correctly excluded. Such families need abstract lessons,
  which this design does not generate.
- A format failure in one episode (malformed `done`, 5 retries) came from
  requests without memory or advice; it is not caused by P4.

Store: `.task2-assets/experience-memory/{development,demo-login-user}/`.

## Run it

```bash
PYTHONPATH=src:scripts .task2-assets/browser-use-env/bin/python scripts/run_agent.py \
    --task click-checkboxes-soft --mode ours --seed 7 8 9       # shared store, one model load
# --memory-readonly  retrieve only · --memory label  old store · --mode baseline  no pillars
```

Store: `.task2-assets/experience-memory/development/` (`records.jsonl`,
`embeddings.npy`, `manifest.json`). Tests: `tests/task2/test_experience_memory.py`.
