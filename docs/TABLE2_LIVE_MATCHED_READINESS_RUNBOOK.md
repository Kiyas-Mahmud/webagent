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

**Current hard blocker:** the tracked page broker is an in-process typed
engineering fixture. Runtime and sealed-evaluator accessors are importable in
the same interpreter, so it supplies reviewed-code message-flow evidence only,
not enforceable isolation. The handoff freezes this truth as
`BLOCKED_EXTERNAL_PROCESS_ISOLATION_REQUIRED`; the canonical bootstrap stops
before importing the provider factory. Consequently none of the commands below
can authorize or start a live campaign in this source version. They document
the post-isolation sequence for use only after a separately authenticated
process-isolated broker implementation and receipt are preregistered under a
new schema. A self-authored Boolean or receipt cannot lift this block.

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

The handoff runner record must also freeze
`pc01_operations_provider_bootstrap`: the exact provider factory entrypoint,
module, qualname, source path/SHA-256, runtime-only source plane, provider
contract schema, and expected public-contract SHA-256. The provider source may
not be shared with the sealed evaluator or page broker. The factory and the
external credential capability are deployment inputs; the repository does not
invent them or claim they are already available.

The credential capability must name one existing directory whose unresolved
leaf and every parent component are non-symlinks. It must be tree-disjoint in
both directions from the campaign and source checkout: `/`, either tree, a
descendant, or an ancestor such as the campaign/source parent is rejected.

After a future registered process-isolation implementation replaces the
current blocker, generate the probe's provider-boundary receipt with the
tracked CLI. The command below documents that later deterministic preparation;
in the current source version it fails closed at the page-broker gate:

```bash
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-pc01-readiness-probe \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --prepare-pc01-provider-boundary-receipt /secure/table2-runtime-receipts/readiness-probe.provider-boundary.json
```

The receipt attests only the reviewed-code, oracle-free dataflow boundary. It
explicitly records that the factory is same-process and that no kernel
filesystem sandbox is claimed. It verifies that no campaign directory, model,
memory, task, evaluator or sealed-artifact path/content is passed to the
factory; only the external non-secret credential capability is path-bearing.
Preparation is immutable: an existing byte-identical receipt is accepted
without rewrite, while any different existing file fails closed. Creation uses
an exclusive same-directory temporary file followed by atomic replacement.

## Future post-isolation probe sequence

Find the first row whose `task_partition` is `normal` in the probe's frozen
`schedule/schedule.jsonl`. Pass that exact `block_id`:

```bash
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-pc01-readiness-probe \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --pc01-provider-boundary-receipt /secure/table2-runtime-receipts/readiness-probe.provider-boundary.json \
  --block-id '<first-normal-block-id>' \
  --live-readiness-probe-for /secure/table2-pc01-pilot-260
```

After the exact factory returns, the CLI repeats campaign/live authority
checks, derives the actual provider public-contract hash, and creates a typed
`PC01ProviderInstallationReceipt`. It registers that receipt with the provider
and appends the complete path-free receipt as a hash-chained
`pc01_provider_installation` access-ledger event before constructing
`ProductionTable2Runner`. Direct provider registration or runner construction
without this event fails. The receipt includes only hashes for the external
credential identity and boundary receipt; it never serializes credential paths
or secret material.

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

Generate the target campaign's own provider-boundary receipt, then run after
both the boundary receipt and matched-readiness receipt exist:

```bash
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-pc01-pilot-260 \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --prepare-pc01-provider-boundary-receipt /secure/table2-runtime-receipts/pilot-260.provider-boundary.json
PYTHONPATH=src python3 scripts/run_table2_evaluation.py \
  --campaign-dir /secure/table2-pc01-pilot-260 \
  --runner web_agent.eval.table2.production_runner:create_runner \
  --runner-factory \
  --pc01-operations-provider-factory '<EXACT_FROZEN_MODULE:FUNCTION>' \
  --pc01-credential-capability-root /secure/table2-runtime/credentials \
  --pc01-credential-capability-id '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_ID>' \
  --pc01-credential-capability-version '<REGISTERED_EXTERNAL_CREDENTIAL_CAPABILITY_VERSION>' \
  --pc01-provider-boundary-receipt /secure/table2-runtime-receipts/pilot-260.provider-boundary.json
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
