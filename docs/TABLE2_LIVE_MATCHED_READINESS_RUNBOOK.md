# Table 2 Live Matched E0–E3 Readiness Gate

## Purpose and scientific boundary

The provisional PC-01 `PILOT_ONLY` campaign must not begin its registered
260-episode schedule until the exact frozen runtime completes one live,
matched E0–E3 WebArena block. This is an operational compatibility gate. It is
not a scored pilot episode, model-selection evidence, an eighth model-evidence
producer role, or a paper Table 2 result.

The probe must use the first registered normal WebArena block. A recovery
scenario or a later normal task cannot be substituted after observing an
outcome. The probe campaign and target pilot are separate frozen campaign
directories with distinct campaign IDs. The target must have no physical
episode artifacts, completion record, or `episode_task_load` access-ledger
entry when the receipt is issued.

Engineering smoke campaigns do not require this receipt. A final campaign
must not contain or use the provisional PC-01 receipt; final readiness follows
the later validation-only PC-01/PC-02/PC-03 selection and final freeze.

## Freeze prerequisites

Freeze the target pilot and a separate readiness-probe clone from the same
clean source commit and the same handoff inputs. Their immutable model,
processor, checkpoint-compatibility receipt, runtime integration, tasks,
environment, prompts, systems, protocol, dependency lock, and read-only P4
memory must be byte-identical. Schedule and provenance bytes may differ only
where the distinct campaign identity requires it.

Both runner attestations and their frozen `runner_source/` trees must include
the exact current bytes for:

- `src/web_agent/eval/table2/live_compatibility.py`
- `src/web_agent/eval/table2/campaign.py`
- `scripts/run_table2_evaluation.py`
- `src/web_agent/eval/table2/evidence_validation.py`
- `src/web_agent/eval/table2/common.py`
- `src/web_agent/eval/table2/schedule.py`
- `src/web_agent/eval/table2/package_validator.py`

The gate refuses to authorize the probe if any of these rows is absent, stale,
symlinked, or different from both the frozen source and current clean source.
This requirement must be added to the handoff source list before freezing; it
is intentionally not inferred after freeze.

## Running the isolated probe

Find the first row whose `task_partition` is `normal` in the probe's frozen
`schedule/schedule.jsonl`. Pass that exact `block_id`:

```bash
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-pc01-readiness-probe \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --block-id '<first-normal-block-id>' \
  --live-readiness-probe-for /secure/table2-pc01-pilot-260
```

The command refuses recovery blocks, later normal blocks, multiple blocks, and
`--maximum-blocks`. In the selected included attempt, all E0–E3 systems must
complete the same block normally. One earlier attempt may be retained only when
its typed infrastructure evidence authorizes the registered whole-block rerun;
the complete interrupted attempt, including exactly empty directories for
systems that were not launched, remains part of the sealed readiness evidence.

After package validation passes, the gate writes these read-only files in the
target campaign:

```text
runtime_readiness/
├── probe_evidence/
│   ├── probe_evidence_manifest.json
│   ├── campaign_manifest.json
│   ├── artifact_hashes.json
│   ├── access_ledger.jsonl
│   ├── deviation_ledger.jsonl
│   ├── schedule/schedule.jsonl
│   ├── frozen/protocol.yaml
│   └── paired_blocks/.../<first-normal-block>/...
├── live_matched_e0_e3_receipt.json
└── live_matched_e0_e3_receipt.sha256
```

The receipt binds the exact target campaign manifest, schedule, immutable
runtime files, required source rows, portable probe-evidence inventory, probe
campaign, matched block tree, E0–E3 episode summaries, access/deviation
ledgers, and causal validation. The sealed probe package contains only the
small records needed to reproduce that binding; it does not duplicate model,
processor, checkpoint, or P4 payload bytes. Their complete digest map is
cross-checked against the target campaign, whose real frozen bytes are
reopened. The
causal evidence includes normalized E1/E2/E3 trained-policy actions,
executions, resets and post-action observations through the input to the first
P1 recovery intervention, or their full shared trace when no P1 intervention
occurs. It separately binds the complete E2/E3 comparison scope through the
first admitted memory intervention, or the entire trace if E3 abstains. E2 must
have zero memory queries and E3 must have zero memory writes.

The target `runtime_readiness/` directory has an exact three-entry closure:
the portable `probe_evidence/` directory, receipt, and sidecar. The probe
package itself has an exact recursive file-and-directory closure and permits
only its six fixed campaign records, its evidence manifest, and the selected
block tree. Extra files, empty branches, unrelated blocks, symlinks, hard
links, writable evidence files, or replacement names invalidate the gate.

## Starting or resuming the 260-episode pilot

Run the target normally after the receipt exists:

```bash
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-pc01-pilot-260 \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory
```

`CampaignRunner` reopens the receipt, its SHA sidecar, the target-local sealed
probe package, and the target campaign's frozen inputs immediately before
every block dispatch. Missing, modified, rebound, writable, symlinked, or
hard-linked evidence fails before the browser runner can reset the task. After
successful receipt issuance, validation no longer reads the external probe
directory. The complete target campaign may be moved as one directory, and
the original probe may be deleted or archived, without breaking replay.

If runtime source, a frozen input, task/environment identity, memory, or probe
evidence changes, do not replace the receipt. Freeze new probe and target
campaigns from the new clean commit and repeat the readiness gate. The receipt
must never be copied to another campaign.
