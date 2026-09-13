# Overlapping-target and skipped-preparation diagnosis — 2026-09-12

Follow-up to [the solver handoff](TABLE2_AGENT_SOLVER_HANDOFF.md), which named two
concrete development-v3 failures and asked for each to be traced separately to a
demonstrated code defect or a model choice. They have different answers.

| Handoff problem | Finding | Evidence |
|---|---|---|
| Overlapping controls rejected | **Implementation defect.** The validator decided target identity by counting box intersections, which is not how the page assigns a click. The model's named control was discarded and re-derived from a point. | Live browser hit test on the exact recorded reset; four archived rejections |
| Preparation is skipped | **Model choice, not a defect.** The frozen base generator received the goal, the screenshot, the empty TYPE-capable field and the v3 preparation instruction, and returned CLICK on Submit. No TYPE was generated, lost or rewritten. | The archived prompt hash, input suffix and raw response |

Nothing here retrains, re-prompts, changes decoding, touches memory content or
alters the trained heads.

## 1. Overlapping controls: the defect

`click-button-sequence` places both buttons with `position:absolute` and
`z-index:auto`, so each button's box can contain the other's centre and the
later button in DOM order paints on top. The development-v3 reset
(`stage_reset_seed 5528761065506371935`) produced exactly that:

| Control | Text | Box `[x, y, w, h]` | Browser element at box centre |
|---|---|---|---|
| `o2:c0` | ONE | `0.271084, 0.327103, 0.120482, 0.186916` | **TWO** — occluded |
| `o2:c1` | TWO | `0.253012, 0.280374, 0.120482, 0.186916` | TWO — its own centre is fine |

Both E2 attempts named `o2:c1`, and E3's second attempt named `o2:c0`. All four
were rejected at `parameter_resolution` with
`HybridParameterResolutionError … TARGET_POINT_MATCH_COUNT:2`, so
`recovery_actions` was 0 and both episodes ended at
`recovery_budget_exhausted`.

The chain that produced this:

1. `resolve_named_target()` matched the model's `o2:c1` to exactly one
   observation-bound control and returned only `action_type/target/bbox/value`.
   **The resolved control's identity was dropped there.**
2. `_target_values()` reduced that box to its centre point.
3. `validate_control_action()` counted how many control boxes contain that
   point, and rejected on any count other than one.

Step 3 uses box containment as a proxy for "which control receives this click."
Those are different things, and under overlap the proxy is wrong in both
directions. Probing the live page at that reset:

- TWO's centre is contained by two boxes but is hit-tested by the browser to
  **TWO**. The rejection was a false negative: the model's action was correct,
  executable, and blocked by the interface.
- ONE's centre is contained by two boxes and is hit-tested to **TWO**. Here a
  click would genuinely have reached the wrong button, so *some* rejection was
  right — but for a reason the match count does not express, and the model's
  correct choice of ONE (the required first button) had no executable path.

Reproduction: `scripts/check_miniwob_overlap_v4.py`, and the earlier raw probe
recorded in this session's evidence directory.

## 2. The change

One rule changes: **when the model names a target, identity comes from the
resolved control and the executed point comes from the browser's own hit test.**

- `CONTROL_JAVASCRIPT` now reports, per control, a `hit_point`: a point inside
  that control's own visible box at which `document.elementFromPoint` resolves
  to that control. It tries the centre first, then a fixed deterministic inset
  grid, and reports `null` when no point inside the control reaches it. This is
  ordinary observable DOM state — no task variable, evaluator state or answer.
- `resolve_named_target()` keeps returning the model's own `target` string and
  the resolved control's box, and now also records `resolved_control_id` in the
  resolution receipt. **The action payload is unchanged**, so every archived
  receipt still replays byte-identically.
- `named_control()` recovers the resolved control from the box the resolver
  itself set, and only for a decision that actually named a target. The
  deterministic provider then executes at that control's `hit_point` and records
  `target_control_id` alongside the existing parameters.
- `validate_control_action()` resolves a named target by that identity and
  requires the executed point to be the browser-verified one. Without a named
  control — a coordinate-grounded prediction, or an observation with no hit
  evidence — the original geometric rule is untouched, including its
  `TARGET_POINT_MATCH_COUNT` rejection.
- The browser worker revalidates the same way against the live DOM before
  dispatch, and `hit_point` joins the stale-target field comparison, so a
  control that moved or became covered between resolution and execution is
  rejected rather than clicked.
- `validate_recovery_target_evidence()` contained **the same defect a second
  time**: it looked up the action's own registered rectangle, then overrode that
  answer with `_visible_point_target()`, which returns `None` whenever two
  registered rectangles contain the point. Fixing only the first gate moved the
  failure here — the live run then rejected with *"recovery action target is not
  visibly present in the current observation"* instead. The override now applies
  only when the point actually identifies a target; when it cannot, an action
  grounded on a registered rectangle keeps that registration. An action grounded
  on no registered rectangle still establishes nothing.

What this deliberately does **not** do:

- It does not choose, replace or re-interpret a control. A wrong button cannot
  be made to succeed: the point always lies inside the control the model named.
- It does not repair a model-supplied coordinate. For a named target the
  coordinate was never the model's output — the interface has always synthesised
  it from the control's box, and the box centre was just as much an interface
  choice as the hit point is.
- It does not change the action type, the issued value, the prompt, the planner
  context or any decoding setting. `prompt_controls()` hides `hit_point`, so the
  model-facing control projection is byte-identical to development-v3.
- It does not touch the trained heads, the memory store, retrieval, the
  threshold or the applicability filter.

### Which systems this affects

E1's trained action head supplies no target hint, so E1 is unaffected and its
NAVIGATE-without-a-permitted-URL problem is untouched. E0, E2 and E3 all reach
the browser through the shared named-target interface and therefore all get the
corrected execution, as the shared action-execution contract requires. **The
E2−E1 contrast changes on the E2 side only, and the E0 baseline changes too.**

### A model-visible leak that had to be closed

The first live run under this change (`miniwob-development-v4`) bumped the
control projection's advertised schema label with the observation interface
version. That label is in the E0 prompt, so **one character** of E0's model
input changed — the two suffixes differ at exactly one position. It was enough:
E0's `click-button` generation went from an unresolvable target to a resolved
one and the episode flipped to success, lifting E0 from 1/4 to 2/4.

That would have confounded execution with re-prompting, so the model-facing
schema label is now pinned and no longer follows the interface version. The E0
control suffix reproduces the archived development-v3 bytes exactly. The v4
package is retained as a negative control, not as a result.

`_visible_point_target()` itself still returns `None` for an ambiguous point, so
overlapping controls remain unidentified for same-target/repeat diagnosis when
the action is not grounded on a registered rectangle. That path does not gate
execution.

## 3. Skipped preparation: not a defect

For `enter-text`, the archived receipt
(`miniwob-development-v3/enter-text/repeat-0/E2/named-recovery-outputs/output-0001.json`)
binds `prompt_sha256 354f8141…`, which is exactly
`configs/eval/table2/miniwob_recovery_prompt_v3.txt`, together with the
observation screenshot. Its serialized `input_suffix` shows the model was given:

- the goal `Enter "Thaddeus" into the text field and press Submit.`;
- `o2:c0` — `tag: input`, `input_type: text`, `value: ""`,
  `supported_actions: ["CLICK", "TYPE"]`;
- `o2:c1` — the Submit button, `supported_actions: ["CLICK"]`;
- the v3 instruction *"Complete required preparation before a later submission
  step."*

The raw response was `{"action_type": "CLICK", "target": "o2:c1", …}`. All six
recovery proposals in the run were CLICK. There is no lost or rewritten TYPE:
had the model emitted `TYPE` on `o2:c0`, that field's box contains no other
control and the action resolves and executes under both the old and the new
rule.

This is a capability limit of the frozen unadapted 2B base generator on this
interface. Re-prompting it until `enter-text` passes would be tuning on the
outcome, so the prompt is unchanged.

## 4. Evidence and verification

- **Regression tests** — `tests/table2/test_development_v4.py`: overlapping
  identity resolution, the preserved geometric rule for coordinate targets,
  rejection of a covered control, rejection of a point the browser does not
  verify, stale observations, identical-box fallback, capability preservation,
  value preservation, and the unchanged model-facing projection.
- **Live browser fixture** — `scripts/check_miniwob_overlap_v4.py` replays the
  exact failing reset and checks the browser's own hit target for each control.
  It also runs the scripted ONE-then-TWO sequence to completion
  (`raw_reward 1.0`). **This is a scripted fixture with zero model calls and is
  not a model success.**
- **Coverage survey** — across all ten development and evaluation families at
  three resets each, 96 of 96 projected controls received a hit point and none
  was unreachable. **88 of those points are the box centre**, i.e. byte-identical
  to the previous behaviour; only the 8 genuinely occluded or ambiguous controls
  move, and every point stayed inside its own control's box.
- **Archive replay** — all 139 archived recovery/E0 receipts across
  `miniwob-development-v3`, `miniwob-interface-v2-evaluation` and
  `miniwob-interface-v2-development-r2` resolve identically under the patched
  code, and all 97 archived planner control projections reproduce exactly. The
  control interface is versioned `miniwob-observable-controls-v3`; `…-v2`
  observations keep the original geometric path.
- **Live matched development run** — `miniwob-development-v6`, 16 episodes,
  audit PASS, `live_path_verified: true`. Results and their limits are in
  [the development-v6 results](TABLE2_DEVELOPMENT_V6.md). In short: the model's
  chosen control now executes on `click-button-sequence` where it previously
  could not, and **task completion did not change** (E0 1/4, E1 0/4, E2 2/4,
  E3 2/4, identical to development-v3).
- No completed package was rerun, overwritten or modified. The five earlier
  evidence directories show zero modified files.

## 5. Separately observed, not changed by this work

Two problems in the current worktree predate this session and are left alone.
Both were isolated by running the same tests from a copied tree, so the evidence
below is about cause, not about which run happened to be green.

1. **All 65 `tests/table2/test_process_broker_*` failures are environmental.**
   The broker child rejects `_distutils_hack` as a "loaded repository module
   outside broker source closure". That module is setuptools' startup shim in
   `.venv/lib/python3.12/site-packages/`, and `.venv` sits **inside** the
   repository root, so the child's repo-relative check classifies it as project
   code. All 65 failures report that one message, and running the same four
   modules from a tree whose root contains no `.venv` reduces them to 1. The
   child's stderr is `DEVNULL`, which is why the visible symptom is only
   "process-broker connection closed early".
2. **The one remaining genuine failure is not from this change.**
   `test_episode_runner_process_backed_parameter_rejection_is_observed_and_bound`
   expects `action_budget_exhausted` and now gets `loop`. The cause is the
   uncommitted `decision.py` loop-guard change, which strips provenance IDs from
   the state fingerprint, so 30 identical rejections fingerprint as one repeated
   state. Three-way check in the copied tree: reverting **only** this session's
   edit to that file still fails; restoring the committed `decision.py` passes.
   That change is deliberate and is also why every development-v3 E1 episode
   ended at `loop` after three steps rather than at the executor budget — but its
   regression test still encodes the old semantics. **The owner should decide
   whether to update the test or revert the loop-guard change**; it was not
   touched here.
