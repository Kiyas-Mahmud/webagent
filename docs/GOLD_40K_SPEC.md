# Gold Dataset Collection Spec — 40k Target (thesis dataset)

> For the data-collection team. The pilots (v8/v12/v14) validated the FORMAT and the
> pipeline. This spec fixes the two leaks they exposed so the 40k dataset gives honest,
> publishable results. **Keep the file format; fix the content + splits.**

## 0. Goal
~40,000 rows of real, leak-free, human-reviewed web-agent steps that train all 4 pillars
(detect / diagnose / recover / remember) and produce honest generalization numbers.

## 1. Format — KEEP (already validated)
- Nested per row: `inputs` / `labels` / `meta`.
- `inputs` = `state_before`, `state_after` (images), `task_description`, `website_domain` — **only these are fed to the model.**
- `labels` = outcome_label, failure_type_4, action_type, action_target_bbox, recovery_strategy, recovery_success, agent_confidence_before, memory_update_flag.
- Result-only fields (`ssim`, `pixel_diff`, `url_after`, `url_before`, `error_message`, oracle, etc.) stay in the **audit file, stripped from the split files** (as you already do).
- `review_status: approved` (human-reviewed).

## 2. The TWO leaks to eliminate (critical)

### Leak A — task text must NOT name the action
In the pilots, every task template named its action, so the model read the answer from the text (`action_acc` = 1.0, fake).
- ❌ "Scroll to find more content", "Choose the dropdown option", "Navigate to the target page", "Submit the search form"
- ✅ **Goal-only** text, action NOT stated: "Find the installation instructions", "Get the current pricing", "Open the community page", "Locate the API reference"
- Rule: a human reading only the task text must **not** be able to tell whether it's a click / scroll / type / select. The action must be inferable from the **page**, not the sentence.

### Leak B — split by DOMAIN, never by row-id
In the pilots, 100% of (domain, task) pairs in test also appeared in train → the model memorized instead of generalizing.
- Split by **website_domain**: some domains go entirely to train, others entirely to val, others to test.
- **Test domains must NEVER appear in train.** Example: train on python.org / numpy / django / …, test on rust-lang / flask / nasa / … .
- Optionally also a harder "unseen-task" split (task-templates held out), but domain-holdout is the minimum.
- Never split by `task_id` (it's just a row number).

## 3. Task diversity
- **Hundreds of distinct tasks**, not 10–15 templates. Real, varied instructions across many goals per site.
- The same goal should appear on many different sites, and sites should support many different goals.

## 4. Distributions (targets)
**Outcome:** SUCCESS 45–55% / FAILURE 45–55%. Same task should sometimes succeed, sometimes fail (so outcome depends on the visual result, not the template).

**Failure types (of FAILURE rows):**
- ACTION_MISMATCH: highest — clicks that **change the page but to the WRONG place** (the hard, visually-real failures).
- PERCEPTION_ERROR: present but NOT dominant (these are often "no change" = easy).
- LOOP_DETECTED: a healthy share (scale up with the 40k).
- NONE = success rows only.

**Actions:** CLICK, TYPE, SELECT, SCROLL, PRESS_KEY, NAVIGATE — no single type dominates (cap CLICK ≤ ~40%, keep realistic).

**Recovery (Pillar 4):**
- `recovery_success = True`: many (scale up — aim for thousands at 40k).
- `recovery_success = False`: many.
- `recovery_attempted = True` must have a **real follow-up step** (action → fail → recovery action → recorded result).
- Leave `recovery_success = null` only when no recovery was attempted.

## 5. Honesty rules (non-negotiable)
- Only `state_before`, `state_after`, `task_description`, `website_domain` are model **inputs**. Everything else is a **label**, predicted, never fed in.
- No label is **derived from the outcome**. `agent_confidence_before` = the agent's real pre-action score, recorded at decision time; its SUCCESS/FAILURE ranges must **overlap** (not disjoint).
- Result-only fields never enter the split files.

## 6. Verify before shipping (self-check)
Run these on the final splits:
1. **Action-text leak:** a classifier on `task_description` alone must NOT predict `action_type` well. If task→action is ~1-to-1, rewrite the tasks.
2. **Split leak:** **0%** of (domain) in test may appear in train. Confirm test domains are unseen.
3. **Confidence leak:** `agent_confidence_before` alone must NOT predict outcome at MCC≈1 (ranges overlap).
4. **Balance:** outcome ~50/50, actions no-dominance, failure types as above, both classes present in every split.
5. `review_status = approved` on all rows.

## 7. One-line summary
> 40k rows, **keep the nested format**. Task text = goal only (never name the action). **Split by domain** (test sites unseen in train), not by row-id. Diverse tasks, balanced outcomes/actions, harder ACTION_MISMATCH failures, thousands of real recovery successes/failures. Every non-input field is a label, never derived from the outcome.
