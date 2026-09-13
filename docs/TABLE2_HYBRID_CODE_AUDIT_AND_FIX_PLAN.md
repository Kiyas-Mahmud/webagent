# Hybrid agent: code audit, causes and bounded fix plan

2026-09-13. **Update: the authorized v3 fixes are implemented and 456 focused
checks passed. The new 24-episode matched development run completed with audit
PASS; H0–H3 remain 2/6 each.** Follow
[v3 implementation and results](TABLE2_HYBRID_INTERFACE_V3.md) for current status;
the audit findings below retain their historical context. The completed H-study remains unchanged: H0 8/30,
H1/H2/H3 9/30, no demonstrated recovery or memory increment.

The code has repairable integration defects, but the evidence does not support
attributing every bad decision to implementation. All 416 saved normal responses
already name CLICK; no generated TYPE was discarded. The immediate task is to
repair rejection handling and control identity, then test whether the existing
generator can make useful multi-step decisions through the corrected interface.

## 1. What was inspected

Reviewed model loading/generation, hybrid input projection, parsing, parameter
resolution, browser control projection/execution, episode/recovery flow, memory
construction/admission and matched final/development evidence. Reviewed the
existing eight-call planner probe to avoid recommending the same failed reminder
experiment again. Relevant runtime sources and prompts were checked against
the final study's producing snapshots; the audited versions match.

No training, checkpoint substitution, generation-setting change, memory write,
embedding regeneration, source-dataset review, Gold-image read or locked-test
access was performed. The saved browser evaluation traces are evidence from
completed studies, not newly opened locked-test data. The current sandbox does
not establish that the host GPU is idle; no live inference was attempted.

## 2. Why recovery and memory had almost no opportunity

For **each** H2 and H3, the 118 normal proposals follow this accounting:

| Route | Count | Consequence |
|---|---:|---|
| Parsing/target-resolution rejection | 90 | Return to normal generation before P1, memory or the normal loop guard. |
| Resolved action executes CLICK | 27 | Every click immediately terminates the environment: nine successes and 18 failures. Recovery cannot legally continue. |
| Parameter-provider rejection | 1 | The only nonterminal assessed incident; two recovery proposals are malformed and never execute. |

The 90 rejects are three episodes with 30 requests each. Attempts 2–30 in each
stuck episode have identical prompts, semantic contexts and screenshots. Greedy
generation repeatedly receives the same input. The learned assessor predicts
failure on all 28 assessed requests, including successful terminal clicks.
Therefore a less sensitive recovery threshold is not the explanation for the
missing calls, and raising recovery budgets would not address the main routes.

Evidence: [final results](TABLE2_HYBRID_FINAL_RESULTS_V1.md) and
`/home/aiub/kiyas/table2-evidence/miniwob-hybrid-final-selective-memory-v1/`.
The three stuck H2/H3 episodes are `click-link/repeat-1`,
`login-user/repeat-2` and `login-user/repeat-4`. The sole recovery incident is
`click-link/repeat-2`.

## 3. Confirmed code defects and concrete repairs

| Finding | Source and reproduction | Minimal repair |
|---|---|---|
| Rejected normal proposals bypass loop detection and P1 | [episode.py:551](../src/web_agent/runtime/episode.py#L551) catches syntax and target errors, then continues at line 595. Loop recording and P1 occur later. A scripted actual H2 runner makes 30 identical invalid generations and 30 advice calls, with zero loop checks, browser actions or P1 calls. | Add a typed rejected-proposal record and bounded repair path. Preserve the known proposed action, exact issued fields, current observation and diagnostic. Detect repeated identical rejected input/output. Pure malformed JSON has no executed action and must not be fabricated into a learned transition. |
| RETRY fails on a newly captured ID for the same control | [miniwob_controls.py:129](../src/web_agent/benchmarks/miniwob_controls.py#L129) refreshes IDs. [strategies.py:434](../src/web_agent/runtime/recovery/strategies.py#L434) validates the old ID against the new page, then otherwise copies it. Same source control, box and hit point: observation 1 is READY; observation 2 rejects `o1:c0`. | Store evidence of the selected stable control identity. For RETRY only, rebind to a uniquely verified current instance of that same control. Preserve action/value; validate current capability and hit point. Ambiguity or a changed/removed control must reject. This is not permission to remap stale targets invented by a new model proposal. |
| ABAB cycle detection includes capture-specific target IDs | [decision.py:70](../src/web_agent/runtime/decision.py#L70) hashes parameters containing `target_control_id`. A/B/A/B with fresh IDs fails the fourth-step cycle check; the fifth capture finally hits the state-count limit. | Use stable observed target identity in the loop fingerprint while retaining raw IDs in evidence. Never collapse different controls or different issued values. Same-state repetition already works. |
| The exposed grounding-confidence name is misleading | [qwen2vl_pc01.py:892](../src/web_agent/runtime/qwen2vl_pc01.py#L892) assigns the maximum action-class probability. [hybrid_action_policy.py:46](../src/web_agent/runtime/hybrid_action_policy.py#L46) exposes it as grounding confidence. | Explain/rename it as action-class confidence in a new semantic projection; preserve original raw model records. It is not a localization confidence estimate. |

The RETRY and ABAB defects are reproduced integration defects, **not causes
demonstrated in the completed final run**: that run did not select RETRY and
had no recorded loop stop. Their fixes improve correctness, without predicting
new completions.

## 4. Interface weaknesses versus model choices

The generator receives the goal, screenshot, editable controls and values.
[qwen2vl_pc01.py:1115](../src/web_agent/runtime/qwen2vl_pc01.py#L1115) uses the
pinned chat template and decodes the generated suffix after the input tokens.
No missing chat-template step, shifted decoding offset or dropped TYPE was
found. The trained checkpoint supplies advice/assessments; a separate frozen
unadapted base deliberately supplies action text under the H architecture.

Two H1/H2/H3 login resets account for 60 parsing failures per system: complete
three-field CLICK Login objects omit `value`. These are not truncated outputs.
Accepting an omitted nullable value would merely execute the model's premature
Login click; it would not insert the missing username/password.

Another episode repeatedly proposes `span#o1:c0`, whereas the declared resolver
accepts `o1:c0` or an observed name. That is a grammar mismatch, not a broken
Playwright click. A future grammar could explicitly accept
`observed_tag#current_control_id`, checking both components against the current
observation. General CSS execution or guessed replacements are unnecessary.
[Target resolver](../src/web_agent/runtime/named_target_policy.py#L106).

The current missing-field diagnostic only says that four fields are required.
It should identify missing/extra keys, expected types, accepted target syntax and
the exact rejected proposal. Keep this correction feedback separate from past
executor records. Those records use `parameters`/numeric boxes, competing with
the requested four-field schema; saved responses copy their structure and
trained-advice boxes. The previous probe demonstrated a formatting contribution,
but neither its reminder nor history rewrite established useful planning.
[Parser](../src/web_agent/runtime/hybrid_interface.py#L51),
[probe results](TABLE2_HYBRID_PLANNER_PROBE_V1.md).

Log generated-token count and EOS/cap stop reason without changing decoding.
The final receipts contain only decoded text. Incomplete text alone cannot
establish token-cap exhaustion. JSON-constrained decoding would be a separate
method/model-setting change; it is not included under the current frozen-decoding
constraint. Increasing the cap alone does not repair a numeric target or a
wrong action schema.

## 5. P4 is constrained by the material and its admission policy

All 1,974 records lack corrective values and reflections. The fixed
[applicability gate](../src/web_agent/memory/context_applicability.py#L23)
necessarily excludes 1,836 non-CLICK records for missing arguments, unless an
earlier rule excludes them first. Only 138 CLICK records potentially survive
that check. Both final queries retrieve PRESS_KEY, PRESS_KEY and NAVIGATE/
BACKTRACK examples; none supplies its required argument. Abstention is correct
under the registered policy. H2/H3 input equality and fallback were preserved.

There is a mismatch between describing these records as advisory labels and
requiring executable historical arguments before admitting most of them. A
**separately versioned advisory-label experiment** could retain source goals,
failure/strategy/action labels and explicit nulls, allowing the current model
to choose current arguments. Keep vectors, ordering, top three and threshold
fixed; do not search lower ranks for preferred examples. This changes the
intervention, does not repair the historical result, and cannot guarantee benefit.
Earlier delivered label-only context also showed no completion improvement.
Substantive corrective experience would require genuine permitted preparation
trajectories and a different, explicitly authorized memory study.

A secondary provenance limitation was found: 86 train-only RETRY supplement
vectors encode before/after **successful historical recovery**, while online
queries describe failed-action transitions. The other 1,888 follow the failure
transition route. The saved build applied the same serialization to both row
types. This is representation mismatch, not evaluation leakage; none of the 86
was a final retrieved hit. Do not alter these vectors or claim this caused final
abstention. Record the limitation and any future source-route exclusion as a
new policy. The unchanged threshold was calibrated on label agreement over the
mixed store, not on browser usefulness.
Evidence: [saved build.py:35](/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1/build.py:35),
[calibration](/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1/calibration.json).

## 6. Implementation order and stopping conditions

Use one new development profile; preserve the completed E/H archives and the
7/7 completed historical TODO. No new framework or model is needed.

- [x] **Repair proposal rejection handling.** Add precise diagnostics and a
  proposal-level repetition guard shared across H0–H3. Permit at most one
  explicitly charged correction request for the same rejected observation/
  proposal; if that corrected request repeats the same failure, stop or report
  unresolved. Do not silently repeat the same greedy call 29 times. Retain
  all original request/model-call accounting and hard limits.
  Separate valid action proposals with failed target resolution from malformed
  JSON: retain their known action type/value and route their rejected request
  through H2/H3 diagnosis/replanning, as parameter-provider rejections already
  do. Keep status REJECTED, no browser command, and the actual unchanged page
  transition; do not invent a target, successful execution or an action type
  for an unparseable response. This route needs explicit contract/regression
  coverage and a separate format-repair metric for H0/H1.
- [x] **Repair target identity and semantics.** Implement evidence-bound RETRY
  rebinding, stable loop fingerprints and accurate advice labels. If adding
  the restricted tag/current-ID grammar, version and test it explicitly.
  Cover removed, replaced, duplicate, occluded and disabled controls and exact
  value preservation. Do not automatically repair model-chosen stale IDs.
- [x] **Verify one minimal planner-interface revision on development only.**
  Use an unambiguous current-action contract and structured rejection details;
  preserve model weights, processor and decoding. Record format validity
  separately from useful model-chosen preparation and subsequent action.
  Repeating the previous reminder probe or accepting premature clicks is not
  evidence of progress. Keep learned diagnosis and post-execution assessment.
- [x] **Make the memory method explicit before running.** Retain strict
  selective-memory semantics, or approve/freeze advisory-label semantics as a
  separate intervention. Do not relabel missing content as verified experience.
  Document the 86-vector source-route limitation without rewriting the store.
- [x] **Run one matched development check and close the revision.** Reuse the
  six existing development families and matched reset/budget design across
  H0–H3. Include a separately labelled controlled recoverable-failure check if
  testing P1 opportunities; match perturbations across systems and keep its
  denominator/results separate from unmodified MiniWoB episodes. Do not remove
  genuine task termination, reset failed episodes, or select tasks for wins.

Readiness requires correct execution and evidence for the proposed path, not a
positive H2−H1 or H3−H2 difference. If the generator still never chooses a useful
preparation action, report that limitation and close this bounded revision.
Another full final evaluation would not establish the missing capability.
Once development is settled, freeze new evaluation instances and include a
matched external reference agent before claiming superiority over others.

## 7. Audit verification and deliverables

The focused run produced **92 passes and four environment failures**: the four
scripted runner tests still invoke real `nvidia-smi`, which returned exit 9 in
this sandbox. Repeating only those four with the GPU-inventory response mocked
inside their already-scripted model/browser harness produced **four passes**.
The production GPU check was not changed. These checks are engineering evidence,
not live model capability or GPU availability evidence.

[analyze_hybrid_failure_routes.py](../scripts/analyze_hybrid_failure_routes.py)
reproduces rejection-loop, named RETRY and ABAB behavior using CPU-only scripted
callbacks through the actual runtime. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/analyze_hybrid_failure_routes.py
```

Expected audited behavior before fixes: 30 generator + 30 advice callbacks,
zero P1/loop checks for each invalid-proposal case; refreshed named RETRY is
rejected; ABAB detection is delayed until the fifth capture. The script creates
fresh `/tmp` evidence. Its counted dispatches are scripted, not actual inference.

The original audit delivered this diagnosis, its implementation checklist,
and the reproduction script. The later authorized v3 revision implements the
runtime/interface fixes and retains strict memory admission. Live verification
and any performance differences are recorded in the linked v3 report.
