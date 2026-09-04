# Table 2 Pilot Task-Interface Decision

Status: `AWAITING_USER_APPROVAL`

Evidence role: `PROPOSAL_ONLY_NOT_A_CAMPAIGN_INPUT`

Paper Table 2 status: `N/R`

## Why a decision is required

The authenticated public WebArena indices 0--49 cannot be run honestly under
PC-01's frozen six browser-action classes. Forty-seven tasks use
`string_match` and require the agent to return an assistant answer through a
termination/answer interface; PC-01 was not trained with such an action. Only
indices 44--46 are page-state URL tasks compatible with the current P3
contract. Silently translating, dropping, or auto-answering the other tasks
would change the experiment and could falsely inflate E0--E3 results.

## Option A — page-state replacement set requiring reset proof (recommended)

Approve an outcome-blind set of 50 public tasks whose official evaluators use
page state (`url_match` and/or `program_html`) and therefore remain executable
through the existing six P3 browser actions. The first proposal preserved the
original site quotas by including Reddit upvote/subscribe tasks 404--406 and
595--596. That proposal is withdrawn: those tasks persistently change shared
site state, while the pinned public task records provide no task-level reset.
Using them in a matched E0--E3 block could let an earlier system change a later
system's start state.

The revised proposal excludes persistent content/account mutations. Because
the pinned release has no page-state-scored, non-mutating Reddit task, the five
Reddit slots are redistributed before any model outcome is observed: two to
Map, two to Shopping Admin, and one to Shopping. This is a task-interface and
state-isolation decision, not a result-based choice:

| Site | Count |
| --- | ---: |
| GitLab | 3 |
| Map | 20 |
| Reddit | 0 |
| Shopping | 10 |
| Shopping Admin | 17 |

Proposed upstream indices, in candidate order (not registered or frozen):

```text
102, 156, 157, 158, 159, 238, 258, 260, 269, 274,
283, 284, 298, 324, 356, 369, 370, 371, 372, 373,
374, 375, 377, 378, 379, 380, 381, 676, 677, 678,
679, 680, 704, 705, 706, 707, 708, 709, 710, 711,
712, 757, 758, 761, 762, 763, 764, 765, 766, 767
```

All 50 tasks are navigation, search, filtering, preview, lookup, directions, or
report-view tasks and are selected to avoid persistent account/content
mutation. Before registration, the deployment-level safety review must still
verify this classification from the complete upstream task and evaluator
configuration and prove identical reset fingerprints across E0--E3. A failed
classification or reset check stops registration; it does not authorize a
silent task substitution.

Approval consequences:

- both the historical 0--49 set and this replacement set become permanent
  development exclusions from the later final campaign, which requires a
  separate active-50 registry and combined 100-task exclusion authority;
- the full content export, start states, evaluator configurations, safety
  evidence, exact interface audit, and joint Gold/WebArena duplicate audit
  must be regenerated and frozen before the pilot;
- every exact evaluator config must pass the frozen 50-row compile authority,
  but compilation alone is not evaluator approval. The repository's current
  `reviewed_compatibility_port_pending_external_review` identity is
  intentionally non-runnable; a new non-pending release/source commit needs a
  final-byte parity receipt and independent review receipt bound to the exact
  implementation, compiler, compile report, upstream reference, reviewer,
  authority, and timestamp hashes;
- no pilot result is created merely by approving the registry.

## Option B — add an answer/termination interface

Add a STOP/ANSWER action and rework the training/runtime parser, policy
contract, parameter provider, executor, E0 interface, budgets, and tests. This
is a material P3 and protocol change, not a small adapter fix. Because PC-01
will not be retrained and time is limited, this option creates a substantial
validity risk and is not recommended for the current pilot.

## Approval record required

No code or tracked manifest currently treats Option A as approved. To proceed,
the project owner must explicitly choose one option. If Option A is approved,
the approval is recorded before any WebArena outcome is observed, then a new
versioned pilot registry/export is generated and the existing fail-closed
handoff is replayed.
