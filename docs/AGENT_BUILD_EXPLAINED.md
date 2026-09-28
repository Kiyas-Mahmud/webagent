# How we built our failure-aware web agent — a step-by-step explanation

2026-09-28. This document explains the agent in plain language: what it is, how
each part works, what went wrong on the way and how it was fixed, and how we
test whether our system makes the agent better. For exact numbers and file
details, follow the links to the technical documents.
(Bengali version: [AGENT_BUILD_EXPLAINED_BN.md](AGENT_BUILD_EXPLAINED_BN.md))

---

## Step 1 — The idea: let a strong agent act, let our model judge

Our trained model has four pillars (P1–P4). Its action pillar (P3: choosing
where to click, type or scroll) is weak: its predicted click boxes rarely land
on the right element. Its failure/recovery pillar (P1) is strong: on our own
validation data it tells good steps from failed ones with MCC ≈ 0.65.

So the agent is split into two roles:

- **An open-source agent does the actions.** We use **Browser Use**, driven by
  the frozen, untrained **Qwen2.5-VL-7B** base model. It clicks elements by
  their index in the page, so it never needs pixel coordinates.
- **Our trained model judges and remembers.** After every action it decides
  whether the step succeeded, what kind of failure happened and which recovery
  strategy fits (P1). It also stores and retrieves past experiences (P4).

Our model never takes over the actions. It only gives advice; Browser Use still
decides the next action itself.

---

## Step 2 — The parts of the agent

```
                ┌──────────────────────────────────────┐
  Real website  │  Chromium browser, 1280×720           │
  (Wikipedia,   │  sites taken from our own dataset     │
   arXiv, …)    └───────────▲──────────────┬───────────┘
                            │ action        │ screenshot + page elements
                ┌───────────┴──────────────▼───────────┐
  ① ACTS        │  Browser Use + Qwen2.5-VL-7B (base)   │
                │  click · type · scroll · select ·     │
                │  navigate · press key                 │
                └───────────┬──────────────▲───────────┘
                            │ before/after  │ advice
                            │ screenshots   │ (recovery + memory)
                ┌───────────▼──────────────┴───────────┐
  ② JUDGES      │  Our trained Qwen2.5 model — P1        │
                │  step outcome · failure type ·        │
                │  recovery strategy · recovery outcome │
                └───────────┬──────────────▲───────────┘
                            │ experience    │ similar past experiences
                ┌───────────▼──────────────┴───────────┐
  ③ REMEMBERS   │  Experience memory — P4                │
                │  memory head decides what to keep     │
                └──────────────────────────────────────┘
```

| Pillar | What it does | Where it lives in the agent |
|---|---|---|
| P1 Failure detection & recovery | Judges each step; picks a recovery strategy; checks whether a recovery worked | Our trained heads (`assessment.py`) |
| P2 Multimodal understanding | Reads screenshots and text together | Inside the same Qwen model |
| P3 Action | Click, type, scroll… | **Browser Use** (our weak P3 is not used) |
| P4 Memory | Stores and retrieves failure→recovery experiences | `memory/experience.py` |

---

## Step 3 — What happens in one step

1. **Browser Use looks at the page** (screenshot + list of interactive elements)
   and chooses one action, e.g. *type "Alan Turing" into the search box*.
2. **The browser performs it**, and a new screenshot is taken.
3. **P1 compares the before and after screenshots** and answers: success or
   failure? what kind of failure? which recovery strategy (RETRY, REPLAN,
   BACKTRACK, ALTERNATIVE_TARGET)?
4. **If P1 says failure:**
   - **P4 searches memory** for experiences on the same page: what failed there
     before, and what was tried to fix it.
   - The diagnosis and any retrieved experiences are sent to Browser Use as
     **advice**.
   - Browser Use chooses the next action itself.
5. **After a recovery action**, P1 judges whether the recovery worked.
6. **When the incident ends** (fixed, or recovery attempts used up), the trained
   **memory head** decides whether the experience is worth keeping. Both fixed
   and unfixed incidents are kept: "this did not work" is useful memory too.
7. The loop repeats until the task is complete or 15 steps are used.

Limits that keep things fair: at most 2 recovery attempts per incident and 4 per
episode; the same step and model-call budgets for every system.

---

## Step 4 — Why real websites and not MiniWoB

We first tested on **MiniWoB**, a set of small toy web pages. There, P1 called
**100% of 122 steps a failure**, even in tasks the agent completed. We checked
carefully that this was not a bug:

- Running 240 rows of our own validation data through the exact agent code gave
  accuracy 0.825 and MCC 0.647 (same as Task 1), so the code uses the model
  correctly.
- Changing the screenshot size, the website name or the task text did not help.

The real reason is what the model learned from our dataset. In our data, a
**successful step usually changes the page a lot** (median 36% of pixels,
because a correct click on a real site often opens a new page); a failed step
changes almost nothing (median 0.7%). A MiniWoB checkbox tick changes only
about 0.2% of the pixels, so the model sees every MiniWoB step as a failure.

**Fix:** run the agent on real websites taken from our own dataset, with
screenshots at the training size (1280×720). There, P1 judges correctly:

| Steps | P1 says SUCCESS |
|---|---:|
| MiniWoB, any step | 0% |
| Real sites, steps of **completed** tasks | **57%** (median P(failure) 0.34) |
| Real sites, steps of **failed** tasks | 18% (median P(failure) 1.00) |

Details: [TASK2_WEB_SUITE.md](TASK2_WEB_SUITE.md).

---

## Step 5 — How memory (P4) was rebuilt

The first memory could never help: its 1,974 stored examples had **no element
information and no corrective values**, 93% were always rejected by its rules,
it used an older model's embedding space, and the agent could not write to it.

The new **experience memory** stores the agent's own incidents: the failed
action and the element it hit (taken from Browser Use), P1's diagnosis, every
recovery attempt with P1's verdict, and whether it was resolved. Keys come from
our Qwen2.5 model's memory embedding; the memory head decides what to store.

A measurement showed that the memory embedding alone is a weak retriever
(AUC 0.60 on training data) — it was trained to decide *what to store*, not to
find similar cases. So retrieval is limited to **the same web page**, ranked by
the embedding, and only experiences whose elements are visible on the current
page are shown. Details: [P4_EXPERIENCE_MEMORY.md](P4_EXPERIENCE_MEMORY.md).

---

## Step 6 — Problems found and fixed on the way

| Problem | Fix |
|---|---|
| The lab PC froze (memory leak: ~12 GB kept after every model call) | Release GPU memory after each call; stop a run below 16 GB free; supervisor kills a run below 12 GB |
| The actor wrapped correct answers in Markdown code fences | Unwrap only a fence around the whole answer (the same rule Browser Use uses) |
| Empty action lists rejected more strictly than Browser Use does | Left to Browser Use's own retry logic |
| A run could hang silently after an error | The runner prints every error and continues with the next task |
| Some sites block robots (DuckDuckGo) or start file downloads (Opera, Debian button) | Those tasks removed or reworded before any comparison |
| A power-off killed a long run | Runs hold a shutdown lock; they can stop and resume without losing results |

---

## Step 7 — How we test whether our system helps

We compare two systems on the same real-website tasks:

- **Baseline** = Browser Use alone.
- **Ours** = Browser Use + P1 recovery + P4 memory.

### Why the baseline and ours both run every task — even tasks the baseline completes

1. **The same task does not always give the same result.** The Wikipedia
   "Alan Turing" task completed in one run and failed in another. We cannot know
   in advance which tasks are "easy".
2. **Our system can also cause harm.** P1 sometimes calls a correct step a
   failure, and an unnecessary recovery could break a task the baseline would
   have completed. That harm must be measured too.
3. **Choosing only the baseline's failures would bias the result.** A task that
   failed once may succeed next time by chance; if we only ran ours on such
   tasks, that lucky success would be wrongly credited to our system.

### Paired comparison

Each task is run by both systems, one after the other on the same day (the order
alternates, so website changes affect both equally). Each pair is one of four
outcomes:

| Baseline | Ours | Meaning |
|---|---|---|
| ✅ | ✅ | Same — both completed |
| ❌ | ❌ | Same — neither completed |
| ❌ | ✅ | **Our system helped** |
| ✅ | ❌ | **Our system hurt** |

We then count "helped" against "hurt" and apply an **exact two-sided sign test**.
Our system is clearly better only if "helped" is much larger than "hurt" —
for example about 7 helped and 0 hurt, or 10 helped and 1 hurt.

Whether a task is complete is decided only by the environment (the page URL
matches the task's goal). The agent never sees this check.

### Why the expected gain is modest, and how to make it clearer

Our system can only help when the baseline fails **and** the failure can be
fixed **and** P1 notices it **and** the agent follows the advice. In the
20-task development run, the agent completed 12 of 19 tasks; of the 7 failures,
3 were the actor producing invalid output — something our system cannot fix.
Ways to make a real difference visible (decided before a final run, never after
seeing its results): harder multi-step tasks, more runs per task (so memory can
help later attempts), and triggering recovery only when P1 is very confident.

Even without a large completion gain, the evidence that P1 correctly detects
failures on real sites, and concrete rescued cases, remain valid results.

---

## Step 8 — How to run it

```bash
cd /home/aiub/kiyas/webagent

# One or more tasks, with or without our system, with a per-step trace
PYTHONPATH=src:scripts .task2-assets/browser-use-env/bin/python scripts/run_agent.py \
    --suite web --task wikipedia-wiki-alan-turing --mode ours

# The frozen baseline-vs-ours comparison (resumable)
PYTHONPATH=src:scripts .task2-assets/browser-use-env/bin/python -u scripts/compare_agents.py \
    run .task2-assets/comparison/web-v1
PYTHONPATH=src .venv/bin/python scripts/compare_agents.py report .task2-assets/comparison/web-v1

# Live progress bar in a second terminal
python3 scripts/progress.py
```

Full run instructions: [TASK2_WEB_COMPARISON_RUNBOOK.md](TASK2_WEB_COMPARISON_RUNBOOK.md).

---

## Where things stand (2026-09-28)

- The agent is complete: Browser Use acting, P1 judging, P4 remembering, all
  working live on real websites.
- The 20-task development run: 12/19 tasks completed; P1 separates good from bad
  steps.
- The baseline-vs-ours comparison (20 tasks × 2 runs × 2 systems = 80 episodes)
  is frozen and ready. A 6-task pilot has started.
- Next: run the comparison and report how many tasks each system completes.
