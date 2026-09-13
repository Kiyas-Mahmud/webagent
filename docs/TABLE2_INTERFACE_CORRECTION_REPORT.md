> Latest no-retraining interface fix and live E2 success: [TABLE2_NAMED_TARGET_INTERFACE_REPORT.md](TABLE2_NAMED_TARGET_INTERFACE_REPORT.md). E3 memory benefit and text-entry success remain unresolved; these are development results.

# Interface corrections — 2026-09-10

Two interface defects were confirmed and corrected without retraining, changing
weights/prompts/decoding, rebuilding memory, or altering the completed evaluation.
The corrections improve validity and loop handling; they did not produce model
task completion in the eight-episode development check.

## Confirmed defects and changes

1. **Exact-box grounding rejected valid points.** Recovery target registration
   hashed exact DOM bounding boxes. A predicted box could put its click point
   inside the correct control and still fail because its dimensions differed.
   The new explicit `visible_point_targets` evidence maps the unchanged execution
   point to one registered, compatible visible control. Geometry is bound to the
   current task, goal and observation and checked against the original target
   registrations. Off-control, overlapping/ambiguous, stale and incompatible
   targets fail closed. No coordinates are snapped, replaced or inferred from
   task-success labels. The old exact contract remains the default when this
   evidence is absent. Alternative-target checks now recognize two different
   predicted boxes inside the same control as the same semantic target.
2. **Loop comparison included changing capture identities.** The page-state
   fingerprint included `recovery_target_evidence.observation_id`, making an
   unchanged page look new at each capture. Loop comparison now excludes only
   these capture IDs from the two target-evidence structures; their bindings
   remain available and are validated at use. Actual screenshot, page content,
   target geometry and actions remain in the comparison.
3. **Transition ordering validation retained.** Each normal/recovery-memory
   transition stream is checked for chronological order before the streams are
   merged for foreign-key validation. Sorting cannot hide an out-of-order stream.
4. **Raw-output diagnostics added to the revised development launcher.** Exact
   E0 and parameter-fallback responses are saved before parsing. This does not
   rewrite responses or supply missing action parameters.

## Validation

- 71 relevant recovery, episode, contract and interface tests passed.
- A browser-only synthetic click-test regression demonstrated the defect:
  old exact-box check rejected an inset box; the corrected check accepted the
  unchanged click point; browser execution produced reward 1 and termination.
  This contributes no model-evaluation success or paper result.
- Eight matched development episodes: click-button and enter-text, E0–E3,
  reusing development-v5 reset/RNG identities in a separate directory.
- All initial screenshots and initial selected-policy probabilities/actions/
  boxes/confidence matched the previous development run exactly.
- E1 now stops after three repeated rejected actions (`loop`) instead of thirty.
- Live audit PASS: no runtime errors, two executed recovery actions and
  assessments, four E3 queries/interventions, unchanged frozen memory store.
- All systems still completed 0/2 development tasks. Repeated development runs
  must not be pooled or described as new held-out evaluation results.

## What remains unsupported by the model output

The newly captured first E0 response was:

```json
{"action_type":"CLICK","target":null,"bbox":null,"value":"next"}
```

The parser correctly rejects a CLICK without a grounded box under its frozen
schema. The response does not supply an executable point. Treating its `value`
field as a target label and filling in a DOM box would be a further explicit
interface/protocol change, not a demonstrated correction of coordinate scaling.

PC-01 still chooses NAVIGATE on these development observations. The deterministic
provider reports that the URL hint is missing; the base fallback literally
returns `{"status":"REJECTED"}`. Prior input-alignment evidence found identical
training/runtime preprocessing and label mapping. No evidence justifies swapping
class labels, masking NAVIGATE, or manually changing it to CLICK.

The remaining failure is therefore not explained by the two corrected defects.
There is no claim that the system's action-generation problem is solved.

## Evidence and execution source

- Root: `/home/aiub/kiyas/table2-evidence/miniwob-interface-correction-v1/`
- Live results/audit: `development/results.json`, `development/evidence-audit.json`
- Exact raw responses: `development/raw-base-*.json`
- Browser geometry regression: `browser-point-check/result.json`
- Test output: `tests.log`
- Revised launcher:
  `/home/aiub/kiyas/table2-evidence/miniwob-feasibility/miniwob_campaign_interface_v2.py`
- The complete original evaluation source was verified and copied by SHA-256
  into `prior-evaluation-source/` before changes. `paths.json` binds original
  paths to those files. Original evaluation results and their sealed manifest
  were not edited. Its producing source is now this archive, not current HEAD.

Use the revised development launcher for interface checks. The previous
`miniwob_campaign.py` is historical; do not overwrite or rerun the sealed
24-episode result package as though these corrections were already present.
