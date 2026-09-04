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

## Option A — page-state replacement set (recommended)

Approve an outcome-blind set of 50 public tasks whose official evaluators use
page state (`url_match` and/or `program_html`) and therefore remain executable
through the existing six P3 browser actions. The set preserves the original
site quotas:

| Site | Count |
| --- | ---: |
| GitLab | 3 |
| Map | 18 |
| Reddit | 5 |
| Shopping | 9 |
| Shopping Admin | 15 |

Proposed upstream indices, in frozen order:

```text
102, 156, 157, 158, 238, 258, 260, 269, 274, 283,
284, 298, 324, 356, 369, 370, 371, 372, 373, 374,
375, 377, 378, 379, 380, 404, 405, 406, 595, 596,
676, 677, 678, 679, 704, 705, 706, 707, 709, 710,
711, 712, 757, 758, 761, 762, 763, 764, 765, 766
```

Forty-five tasks are read-only. Reddit tasks 404, 405, 406, 595, and 596 make
bounded upvote/subscribe changes. They may be admitted only after reset parity
and the registered second safety review pass. If either check fails, task
eligibility must be resolved before execution without looking at any model
outcome.

Approval consequences:

- both the historical 0--49 set and this replacement set become permanent
  development exclusions from the later final campaign;
- the full content export, start states, evaluator configurations, safety
  evidence, exact interface audit, and joint Gold/WebArena duplicate audit
  must be regenerated and frozen before the pilot;
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
