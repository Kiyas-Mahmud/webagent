# Web-Gold-40K Manual Dataset Review Guide

> **Audience:** two teammates who are new to dataset review.
>
> **Purpose:** explain what the dataset contains, how to inspect it, what may be
> corrected, what must not be invented, and how to produce defensible human-review
> evidence for the thesis and a Q1 journal submission.

## 1. The most important rule

Do not approve a row because it “looks mostly okay.” Approve it only when the images,
task, action, outcome, failure type, and any recovery information agree with one another.

Never change a label merely to improve class balance or model performance. If the evidence
is unclear, mark the row `needs_discussion`; if the evidence is unusable, reject or recollect
it. An honest smaller dataset is better than a larger dataset with invented labels.

## 2. Dataset overview

Web-Gold-40K contains browser-interaction steps for training a failure-aware web agent. Each
row shows the page before an action, the page after the action, the user's goal, and labels
describing what action occurred and whether it worked.

The final dataset currently contains **39,215 rows**. The name “40K” is rounded; the team
does not need to fabricate rows to reach exactly 40,000.

The large dataset is stored as the Kaggle Dataset `kiyasmahmud/web-gold-40k` and is mounted
inside Kaggle at `/kaggle/input/datasets/kiyasmahmud/web-gold-40k`. Do not download or
extract it onto a low-storage local machine. Reviewers may instead use the provided HTML
review packages, screenshots, replays, and audit records supplied by the data owner.

| Item | Current value |
| --- | ---: |
| Total rows | 39,215 |
| Tasks/trajectories | 29,978 |
| Domains | 509 |
| Train rows | 23,499 |
| Validation rows | 7,861 |
| Test rows | 7,855 |
| SUCCESS | 17,136 (43.70%) |
| FAILURE | 22,079 (56.30%) |
| Recovery attempted | 9,237 |
| Recovery succeeded | 3,153 |
| Recovery failed | 6,084 |

The split is domain- and task-disjoint: a domain or task occurs in only one of train,
validation, or test. Reviewers must not move individual rows between splits.

### 2.1 Action distribution

| Action | Rows |
| --- | ---: |
| CLICK | 6,427 |
| NAVIGATE | 7,444 |
| TYPE | 6,469 |
| SELECT | 6,465 |
| SCROLL | 6,463 |
| PRESS_KEY | 5,947 |

### 2.2 Failure distribution

| Failure type | Rows | Simple meaning |
| --- | ---: | --- |
| NONE | 17,136 | The step succeeded; no failure occurred. |
| ACTION_MISMATCH | 10,832 | An action occurred, but it was the wrong action or wrong target. |
| PERCEPTION_ERROR | 10,053 | The page or target was misunderstood, missing, blocked, or incorrectly perceived. |
| LOOP_DETECTED | 1,194 | The trajectory repeated an unsuccessful state/action instead of progressing. |

## 3. What one dataset row contains

Every row has three top-level sections: `inputs`, `labels`, and `meta`.

### 3.1 Inputs: the only information given to the model

`inputs` must contain exactly these four fields:

| Field | What it means | What the reviewer checks |
| --- | --- | --- |
| `state_before` | Screenshot before the action | Correct page, readable, and belongs to this task/step. |
| `state_after` | Screenshot after the action | Correct result image, readable, and chronologically after `state_before`. |
| `task_description` | The user's goal | Goal is correct and does not explicitly reveal the required action. |
| `website_domain` | Website/domain identifier | Matches the page and original audit record. |

No label, result, URL oracle, similarity score, browser status, or error message may be
added to model inputs.

### 3.2 Labels: answers the model must learn to predict

| Field | Allowed meaning/value | What the reviewer checks |
| --- | --- | --- |
| `outcome_label` | `SUCCESS` or `FAILURE` | Did this exact step achieve its intended immediate result? |
| `failure_type_4` | `NONE`, `ACTION_MISMATCH`, `PERCEPTION_ERROR`, `LOOP_DETECTED` | Does the failure category match the visual and trajectory evidence? |
| `agent_confidence_before` | Number from 0 to 1 | Must be the recorded pre-action value, not guessed after seeing the outcome. |
| `action_type` | `CLICK`, `NAVIGATE`, `TYPE`, `SELECT`, `SCROLL`, `PRESS_KEY` | Does it match the action that was actually executed? |
| `action_coordinates` | Coordinates or action-specific value | Within the screenshot and consistent with the target/action. |
| `action_target_bbox` | Target box or null | Box covers the true target and stays inside the source image. |
| `recovery_attempted` | Boolean | True only when a real follow-up recovery action exists. |
| `recovery_strategy` | `NONE`, `RETRY`, `REPLAN`, `BACKTRACK`, `ALTERNATIVE_TARGET`, or `ABORT` | Matches the recovery that actually occurred. |
| `recovery_success` | `true`, `false`, or `null` | True/false only when recovery was attempted; otherwise null. |
| `memory_update_flag` | Boolean | Matches the collection policy for whether this experience should be stored. Do not decide by intuition alone. |

### 3.3 Metadata: identification and review evidence

`meta` contains identifiers such as `sample_id`, `task_id`, step information, collection
source/time, evaluation masks, and `review_status`.

- `sample_id` identifies one row and must be unique.
- `task_id` identifies the full task/trajectory. Review all steps of a trajectory together.
- `review_status` is currently `pending` for every row.
- A row becomes `approved` only after the required human review and resolution are complete.

Do not rename IDs, change split membership, or change a domain merely to make a validation
test pass.

## 4. Files the reviewers may receive

The handoff report lists these artifacts:

- `gold_audit.jsonl`: full collection/audit evidence; use it to resolve facts that are not
  visible in model inputs.
- `split_train.json`, `split_val.json`, `split_test.json`: exported model data. Reviewers
  should not edit these files directly.
- `images/`: before/after screenshots.
- `replays/`: complete browser trajectories.
- `review_5000/review.html`: the existing manual-review interface.
- `gold_export_summary.json`: counts and split manifest.

The existing `review_5000` package contains 7,559 rows from 4,818 complete tasks. It is a
stratified review sample, not the whole 39,215-row dataset. Completing only this package does
**not** permit all 39,215 rows to be marked approved.

If the project keeps the current approved-only publication gate, every retained row must be
reviewed. The data owner should provide additional review batches covering all remaining
trajectories. Never mark unreviewed rows approved to make the validator pass.

## 5. Two-person review plan

Always assign complete trajectories, not individual rows. This is necessary for checking
loops and recovery outcomes.

### Phase A: training and calibration

1. Both reviewers read this guide completely.
2. Both independently review the same 100 trajectories, selected across all three splits,
   six actions, two outcomes, and four failure labels.
3. Compare decisions field by field.
4. Discuss disagreements and write short examples of the agreed rules.
5. Repeat with another small batch if major disagreements remain.

Do not begin the main review until both reviewers apply `SUCCESS`, `ACTION_MISMATCH`,
`PERCEPTION_ERROR`, and `LOOP_DETECTED` consistently.

### Phase B: production review

- Select a stratified **10% of trajectories** for independent review by both people. This
  overlap is used to measure inter-reviewer agreement.
- Divide the remaining 90% approximately equally between Reviewer A and Reviewer B.
- Keep every task and all its steps with one primary reviewer.
- Every proposed correction, rejection, image problem, adult-domain case, or uncertain row
  must receive a second-person decision.
- Neither reviewer overwrites the other's original decision.

### Phase C: agreement and adjudication

For the double-reviewed subset, report agreement for at least:

- approve/reject decision;
- `outcome_label`;
- `failure_type_4`;
- `action_type`;
- recovery correctness.

Calculate raw percentage agreement and Cohen's kappa for categorical labels. The data/ML
lead can calculate these numbers from the review log. If the two reviewers cannot resolve a
disagreement, escalate it to the thesis lead or supervisor; do not settle it by guessing.

## 6. Review one trajectory: exact procedure

### Step 1: confirm identity and sequence

- Record `split`, `sample_id`, `task_id`, and step index.
- Open the whole trajectory in step order.
- Confirm no row belongs to a different task or website.
- Confirm before/after images are attached to the correct step.

### Step 2: read the task without looking at the labels

Ask: “What is the user's goal?” The text should state the goal, not the answer/action.

Bad task text leaks the label:

- “Click the pricing button.”
- “Scroll down to find installation.”
- “Select an option from the dropdown.”

Better goal-only text:

- “Find the current pricing.”
- “Locate the installation instructions.”
- “Choose the required plan.”

Only rewrite task text when the original goal is known from the audit/replay. Do not invent a
new goal from the screenshots.

### Step 3: inspect `state_before`

Check that the screenshot:

- opens correctly and is not corrupt;
- is readable enough to judge the page;
- shows the expected domain/task context;
- is the page immediately before the recorded action;
- has not been manually edited, painted, cropped, or replaced.

### Step 4: verify the action and target

Use the replay/audit evidence when the screenshots alone are insufficient.

| Action | Manual evidence to check |
| --- | --- |
| CLICK | The coordinate/box covers the element actually clicked. |
| TYPE | The correct field was focused and the intended text entry occurred. |
| SELECT | The correct control and option were used. |
| SCROLL | The viewport moved in the recorded direction/amount; it is not a static duplicate caused by capture failure. |
| PRESS_KEY | The recorded key caused a compatible result, such as Enter submitting or Tab changing focus. |
| NAVIGATE | The browser reached the intended recorded destination through the executed navigation action. |

For a non-null bounding box, check it against the image's native size:

- `x >= 0`, `y >= 0`, `width > 0`, and `height > 0`;
- `x + width` does not exceed image width;
- `y + height` does not exceed image height;
- the box covers the actual target, not nearby text or empty space;
- action coordinates lie inside the image and, when applicable, inside the target box.

Do not assume a fixed screenshot size; inspect the image's real dimensions.

If a box uses coordinates from a taller/different screenshot (for example, `y=1223.5` for a
1280x720 image), do not clamp it to the bottom edge. Reconstruct the box by visually locating the
target in the attached `state_before` image. If the correct target cannot be established from replay
evidence, set only `action_target_bbox` to `null` and keep the row for its other supervised tasks.
Record the decision and reason in the review sheet.

### Step 5: inspect `state_after` and decide the outcome

Compare the two screenshots and the task's immediate goal.

Use `SUCCESS` only when the intended step result actually happened. A page change alone does
not prove success. Reaching the wrong page is a failure even if the website visibly changed.

Use `FAILURE` when the action did not achieve its intended immediate result, targeted the
wrong element, produced an error, or repeated without progress.

### Step 6: choose the failure type

Use this decision order:

1. If the step succeeded, `failure_type_4 = NONE`.
2. If the trajectory repeats a previous unsuccessful state/action without progress, use
   `LOOP_DETECTED`.
3. If an action occurred but it was the wrong operation or wrong target, especially when it
   caused a meaningful wrong page change, use `ACTION_MISMATCH`.
4. If the agent misunderstood what was visible, believed a missing/blocked element existed,
   or could not correctly perceive the target, use `PERCEPTION_ERROR`.
5. If the evidence cannot distinguish the last two, use `needs_discussion`; do not choose the
   larger class for balance.

Required logic:

- `SUCCESS` must have `failure_type_4 = NONE`.
- `FAILURE` must not have `failure_type_4 = NONE`.

### Step 7: verify recovery using later steps

Recovery cannot be checked from an isolated row. Inspect the next step(s) in the trajectory.

- `recovery_attempted = true` only when a real recovery action followed the failure.
- The selected `recovery_strategy` must describe the action that actually occurred.
- `recovery_success = true` only when the observed recovery reached its intended goal.
- `recovery_success = false` when recovery was attempted but did not reach the goal.
- `recovery_success = null` only when no recovery was attempted.
- When no recovery was attempted, the strategy should be `NONE` under the current label map.

Typical meanings:

| Strategy | Meaning |
| --- | --- |
| `NONE` | No recovery was performed. |
| `RETRY` | Try the same intended action again. |
| `REPLAN` | Change the plan or choose a new sequence. |
| `BACKTRACK` | Return to a previous page/state. |
| `ALTERNATIVE_TARGET` | Use another element that can achieve the same goal. |
| `ABORT` | Stop after recovery is no longer safe/useful; use only if genuinely recorded. |

For the recovery-v2 export, Reviewer A must record these four pieces together for every
attempted recovery: (1) the failure `state_after`, (2) the next executed action and value,
(3) the next observed `state_after`, and (4) the resulting true/false success label. Reviewer
B must independently confirm all four before approval. A success label based only on the
failure row is invalid.

Before changing or collecting any rows, run the recovery class audit and report separate
counts for `RETRY`, `ABORT`, `BACKTRACK`, and `LOOP_DETECTED`. If one is absent or too small
to support the paper's claim, add only reviewed trajectories for that missing class. Do not
rebalance by relabeling evidence, duplicating rows, or recollecting the entire dataset.

### Step 8: verify confidence and memory labels

`agent_confidence_before` must come from the value recorded before the action. It must be in
the range 0 to 1. Never increase it because the row succeeded or decrease it because the row
failed. If the original pre-action value cannot be recovered, flag the row; do not invent one.

For `memory_update_flag`, follow the documented collection/export rule and verify it against
the audit record. Do not set every failure to true or every success to false unless that is
the actual versioned policy. If the policy is missing or ambiguous, flag the row and ask the
data lead to define the rule once for the entire dataset.

### Step 9: record the decision

Use one of these review decisions:

- `approve_no_change`: all evidence and labels agree.
- `approve_after_correction`: a documented correction was made and independently confirmed.
- `needs_discussion`: evidence is ambiguous or a rule is unclear.
- `reject_recollect`: evidence is wrong, corrupt, missing, unsafe, or cannot be recovered.
- `quarantine_policy`: row is held outside final data pending an ethics/safety decision.

Only `approve_no_change` and fully resolved `approve_after_correction` rows may receive final
`review_status = approved`.

## 7. Known issues to review first

### 7.1 Probable blank or browser-error images

The handoff found **71 rows in 62 trajectories** with probable blank or browser-error frames.
Review these first.

A blank/error page is not automatically invalid:

- Keep it if it is a real browser result, the capture is readable, and the failure labels
  honestly describe what happened.
- Reject or recollect it if it came from screenshot/capture corruption, a loading race, the
  wrong browser tab, or a missing image rather than a real agent interaction.

Record which case applies. Do not delete all 71 automatically.

### 7.2 Adult-domain subset

The historical catalog reports **1,354 rows from 16 adult domains**. The project currently
intends to keep the 39,215-row corpus, but these rows require a written supervisor/ethics and
safety protocol before publication.

Reviewers must:

- flag every affected row and trajectory;
- avoid opening content if institutional policy or reviewer safety does not permit it;
- never mark the policy as confirmed themselves;
- record whether the final protocol allows inclusion or requires quarantine.

If retained, document the research necessity, access controls, reviewer protection, privacy,
and reporting policy. If not permitted, quarantine the affected rows/domains and regenerate
the final splits. Do not silently remove or silently approve them.

### 7.3 Outcome balance warning

SUCCESS is 43.70%, slightly below the planned 45% lower target. This is a warning, not proof
that labels are wrong.

- Do not convert failures into successes to reach 45%.
- It is acceptable to keep the honest distribution and disclose it.
- If the team wants closer balance, collect additional real successful trajectories; do not
  duplicate or synthesize filler rows.

### 7.4 Task-text action signal

The automated text-only baseline reached action accuracy 0.3537 versus a 0.1901 majority
baseline. It passed the blocker, but task text still contains moderate action signal.

During review, flag instructions that explicitly say click, scroll, type, select, press, or
navigate. Rewrite only from the verified original goal and obtain second-review approval.

## 8. What may be changed

All corrections must be supported by screenshots, replay, browser/audit logs, or the original
collection record. Record both the old and proposed value.

| Data item | Change allowed? | Rule |
| --- | --- | --- |
| Outcome/failure/action labels | Yes | Only when direct evidence proves the existing label is wrong. |
| Recovery labels | Yes | Must inspect the real follow-up step/result. |
| Bounding box/coordinates | Yes | Must use the true target and native image coordinates. |
| Task description | Limited | Correct errors or action-leaking wording only when the original goal is known. |
| Image path | Limited | Replace only with the correct existing image from the same trajectory. |
| Screenshot pixels | No | Never edit visual content to make a row pass. Recollect instead. |
| Confidence | Very limited | Restore only from the recorded pre-action source; never infer it after the result. |
| Memory flag | Limited | Correct only under one documented dataset-wide policy. |
| `sample_id`, `task_id`, split | No manual change | Fix at the source/export level; preserve domain/task holdout. |
| Website domain | Source correction only | Verify from the audit record; changing it may require rebuilding splits. |
| Review status | Yes, after review | Set approved only after all issues are resolved. |

## 9. What must never be done

- Do not approve rows that were not actually reviewed.
- Do not edit exported split JSON files as the source of truth.
- Do not infer labels from class targets or desired model performance.
- Do not relabel rows to make SUCCESS/FAILURE exactly 50/50.
- Do not copy rows or images to increase the dataset count.
- Do not expose result-only audit fields as model inputs.
- Do not move a row to another split; domain/task separation may break.
- Do not split one trajectory across reviewers or dataset splits.
- Do not guess pre-action confidence after seeing the result.
- Do not mark recovery successful without observing the follow-up result.
- Do not use `NONE` failure type for a failed row.
- Do not hide rejected rows, disagreements, adult-domain decisions, or limitations from the
  final review report.

## 10. Review log format

Use a shared CSV or spreadsheet. Keep one immutable row per reviewer decision; add final
resolution columns instead of overwriting the original decision.

Recommended columns:

```text
reviewer_id
review_round
reviewed_at_utc
split
sample_id
task_id
step_index
decision
before_image_ok
after_image_ok
task_goal_only
action_type_ok
coordinates_bbox_ok
outcome_ok
failure_type_ok
recovery_ok
confidence_source_ok
memory_label_ok
blank_or_error_flag
adult_domain_flag
reason_code
old_value
proposed_value
evidence_reference
reviewer_notes
second_reviewer_id
second_reviewer_decision
final_decision
final_change_applied_by
final_change_verified_by
```

Use consistent reason codes:

```text
IMG_CAPTURE_INVALID
IMG_WRONG_PAIR
IMG_WRONG_TRAJECTORY
TASK_ACTION_LEAK
TASK_WRONG_GOAL
ACTION_LABEL_WRONG
ACTION_COORDINATES_WRONG
BBOX_WRONG
OUTCOME_WRONG
FAILURE_TYPE_WRONG
RECOVERY_LABEL_WRONG
CONFIDENCE_SOURCE_MISSING
MEMORY_POLICY_UNCLEAR
ADULT_POLICY_PENDING
DUPLICATE_OR_ID_PROBLEM
OTHER
```

An `evidence_reference` should point to the replay step, audit-record ID, screenshot paths,
or another reproducible source. “I think this is wrong” is not enough evidence.

## 11. Applying corrections safely

The reviewers propose and verify changes; the data owner applies them.

1. Freeze and checksum the original dataset version.
2. Merge both review logs without removing disagreements.
3. Resolve every `needs_discussion`, proposed correction, rejection, and policy case.
4. Apply approved corrections to the source audit/export data, not only to a split file.
5. Set `review_status = approved` only for finalized retained rows.
6. Remove/quarantine rejected rows and preserve their IDs/reasons in the audit log.
7. Regenerate all split artifacts from the corrected source when any row, task, domain,
   image reference, or label changes.
8. Preserve domain- and task-disjoint splits.
9. Record a new dataset version, timestamp, row counts, checksums, and change summary.
10. Rerun every automated validation gate.

If corrections reduce the dataset below 39,215 rows, report the new honest count. Do not add
filler. New rows may be added only through real collection and the same review process.

## 12. Final automated validation after review

The data/ML lead should run
`notebooks/kaggle_gold_data_validation.ipynb` on Kaggle after uploading the reviewed version.
The reviewers do not need to download the large dataset.

For the final publication audit:

```python
FULL_IMAGE_HASH_AUDIT = True
RUN_SHORTCUT_BASELINES = True
ADULT_DOMAIN_POLICY_CONFIRMED = True  # only after written approval or completed quarantine
```

The final audit must confirm:

- all retained rows have `review_status = approved`;
- schema and the four-input allow-list pass;
- label logic passes;
- all image paths exist and all images decode;
- sample IDs remain unique;
- domain and task overlap across splits remains zero;
- exact full-input overlap/conflicts remain zero;
- exact SHA-256 image-content overlap across splits is zero;
- near-image dHash overlap is resolved/documented;
- task-text-only and confidence-only shortcut blockers pass;
- adult-domain protocol is genuinely resolved.

The current automated result is **31 PASS, 1 FAIL, 1 WARN, 3 SKIP** with
`publication_ready = false`. The remaining publication blockers are:

1. 39,215 pending rows and zero approved rows;
2. full exact-image-content hashing not yet run;
3. full near-image perceptual hashing not yet run;
4. adult-domain protocol not yet confirmed.

The 43.70% SUCCESS share is a non-blocking warning that must be reported honestly.

## 13. Reviewer completion checklist

Each reviewer signs only when all applicable items are true:

- [ ] I reviewed complete trajectories, not isolated rows.
- [ ] I checked both screenshots and the task goal.
- [ ] I verified action and target evidence.
- [ ] I verified outcome and failure type using the decision rules.
- [ ] I inspected later steps for every recovery label I approved.
- [ ] I did not infer confidence from the observed result.
- [ ] I recorded every correction with old value, new value, reason, and evidence.
- [ ] I flagged blank/error captures and adult-domain cases.
- [ ] I did not approve any row I did not inspect.
- [ ] I preserved my original decisions even when later adjudicated.
- [ ] All my unresolved cases were escalated.

## 14. Final deliverables from the two reviewers

Provide the thesis/data lead with:

1. Reviewer A's immutable review log.
2. Reviewer B's immutable review log.
3. The overlap/agreement table and Cohen's kappa results.
4. The adjudication log for every disagreement.
5. A corrections file containing sample ID, old value, new value, reason, and evidence.
6. A rejection/quarantine list with reasons.
7. A list of the 71 suspicious rows/62 trajectories and their final decisions.
8. The adult-domain review/policy list without falsely declaring policy approval.
9. Final counts: reviewed, approved unchanged, corrected, rejected, quarantined, unresolved.
10. Signed reviewer checklist and review dates.

Only after these deliverables are merged, corrections are applied, and all final automated
gates pass may the dataset be described as fully reviewed and publication-ready.
