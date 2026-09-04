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

**Current hard blocker:** the repository now contains a source-bound AF_UNIX
process-broker architecture with HMAC-authenticated JSON messages, peer
PID/UID checks, distinct runtime/control keys, four allowlisted operations,
exact outer request/result envelopes, recursive registered sensitive-key
rejection, source- and PID-bound authenticated readiness, and authenticated
cleanup receipts. Its `action`, `observation`, and `execution` inner values are
still arbitrary mappings. A neutral key such as `content` can carry data whose
origin cannot be inferred from the schema, so these checks do not establish
semantic oracle isolation or value provenance.
Its local fixture tests are key/envelope architecture evidence only. The active
WebArena deployment has not supplied an independently authenticated receipt proving
that these process/source/credential boundaries were used on the campaign
host. Handoff therefore records
`BLOCKED_INNER_SCHEMAS_VALUE_PROVENANCE_AND_EXTERNAL_RECEIPT_REQUIRED`; the canonical bootstrap still
stops before importing the provider factory. The older same-process broker is
retained for engineering fixtures and is explicitly unpromotable. A
self-authored Boolean, local test receipt, or copied receipt cannot lift this
block.

The local broker child currently shares the runtime UID and inherits its launch
environment. That is address-space separation, not a same-UID hostile-code or
secret-confidentiality claim. A promotable external deployment receipt must be
bound to the exact attested source/session and independently prove a separate
UID or container/process sandbox, a filesystem policy that prevents runtime
inspection of evaluator memory/state, an allowlisted scrubbed evaluator
environment with no inherited runtime/provider secrets, authenticated peer
identity, and cleanup under those same controls. Promotion additionally
requires exact operation-specific schemas for all three arbitrary inner mapping
paths and externally reviewable provenance showing where every runtime-visible
value came from. The registered schema in this source version intentionally has
neither those inner contracts nor a trust anchor capable of accepting a
deployment receipt.

## Freeze prerequisites

Freeze the target pilot and a separate readiness-probe clone from the same
clean source commit and the same handoff inputs. Their immutable model,
processor, checkpoint-compatibility receipt, runtime integration, tasks,
environment, prompts, systems, protocol, dependency lock, and read-only P4
memory must be byte-identical. Schedule and provenance bytes may differ only
where the distinct campaign identity requires it.

For a split local-browser/DGX-inference deployment, the frozen semantic
dependency lock must be `table2-semantic-dependency-lock-v2`. Its
`browser_host` record is derived from the measured local WebArena preflight;
its `dgx_host` record is derived from the separately pinned, caller-supplied
`table2-dgx-model-runtime-identity-v2`, which includes explicit
Python/platform and sorted package measurements as well as model-runtime,
source-set, and runtime-environment identities. The model/source/environment
hashes are derived from the nested typed records, and the PC-01 model hashes are
exactly registered. Missing DGX inventory is a hard stop. The evaluation CLI
remeasures the browser host during bootstrap and the live capability gate
repeats its runtime check per block. The comparison CLI can run on the DGX with
`--remeasure-host-role dgx_host` and
`--supplied-dgx-runtime-identity <CURRENT_V2_IDENTITY.json>`. It remeasures the
executing Python/platform/package stack but only compares the supplied
model/source/environment record; it does not independently generate that
record. A future startup/model-load/per-block receipt must bind campaign ID,
block ID, fresh nonce, semantic-lock hash, DGX host/runtime identities, phase,
and issue time. No receipt schema or external trust anchor exists in this source
version. Split dispatch therefore fails closed; a browser-side self-claim or
copied JSON cannot substitute. Single-host v1 locks remain supported.

Both runner attestations and their frozen `runner_source/` trees must include
the exact current bytes for:

- `src/web_agent/__init__.py`
- `src/web_agent/eval/__init__.py`
- `src/web_agent/eval/table2/__init__.py`
- `src/web_agent/eval/table2/live_compatibility.py`
- `src/web_agent/eval/table2/campaign.py`
- `scripts/run_table2_evaluation.py`
- `src/web_agent/eval/table2/evidence_validation.py`
- `src/web_agent/eval/table2/common.py`
- `src/web_agent/eval/table2/schedule.py`
- `src/web_agent/eval/table2/package_validator.py`
- `src/web_agent/eval/table2/dependency_lock.py`
- `src/web_agent/eval/table2/process_broker.py`
- `src/web_agent/eval/table2/process_broker_protocol.py`
- `src/web_agent/eval/table2/process_broker_runtime.py`
- `src/web_agent/eval/table2/process_broker_worker.py`

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

After the deployment owner supplies and preregisters independent process-
isolation evidence for the actual campaign host, generate the probe's
provider-boundary receipt with the tracked CLI. The command below documents
that later deterministic preparation; in the current source version it fails
closed at the external-deployment-receipt gate:

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
