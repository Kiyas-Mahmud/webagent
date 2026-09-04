from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from web_agent.benchmarks.webarena import (
    WebArenaManualRescueCheck,
    WebArenaManualRescueReceipt,
)
from web_agent.eval.table2.common import SchemaError, atomic_write_json
from web_agent.eval.table2.execution_guard import (
    PC01_PROVIDER_BOUNDARY_CLAIM_SCOPE,
    PC01_PROVIDER_INSTALLATION_RECEIPT_SCHEMA_VERSION,
    PC01ProviderInstallationReceipt,
    PRODUCTION_RUNNER_ENTRYPOINT,
)
from web_agent.eval.table2.package_validator import (
    _validate_abort_terminal_semantics,
    _validate_evaluator_backend_guards,
    _validate_final_evidence,
    _validate_frozen_runtime_budget_evidence,
    _validate_manual_rescue_guard_events,
    _validate_pc01_provider_installation_ledger,
    _provider_campaign_state_sha256,
)
from web_agent.runtime.contracts import canonical_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64


def _provider_installation_fixture(root: Path) -> tuple[dict, dict]:
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    expected_contract = "e" * 64
    binding = {
        "factory_entrypoint": "fixture.provider:create_provider",
        "factory_module": "fixture.provider",
        "factory_qualname": "create_provider",
        "source_relative_path": "src/fixture_provider.py",
        "source_sha256": "f" * 64,
        "expected_provider_public_contract_sha256": expected_contract,
    }
    manifest = {
        "campaign_mode": "evaluation",
        "runner_entrypoint": PRODUCTION_RUNNER_ENTRYPOINT,
    }
    atomic_write_json(root / "campaign_manifest.json", manifest)
    atomic_write_json(
        frozen / "runner_attestation.json",
        {"pc01_operations_provider_bootstrap": binding},
    )
    atomic_write_json(frozen / "environment.json", {"fixture": True})
    (frozen / "protocol.yaml").write_text("{}\n", encoding="utf-8")
    state_sha256 = _provider_campaign_state_sha256(root)
    receipt = PC01ProviderInstallationReceipt(
        schema_version=PC01_PROVIDER_INSTALLATION_RECEIPT_SCHEMA_VERSION,
        record_type="PC01ProviderInstallationReceipt",
        status="PASS",
        claim_scope=PC01_PROVIDER_BOUNDARY_CLAIM_SCOPE,
        same_process_factory=True,
        kernel_filesystem_sandbox=False,
        factory_entrypoint=binding["factory_entrypoint"],
        factory_module=binding["factory_module"],
        factory_qualname=binding["factory_qualname"],
        factory_source_relative_path=binding["source_relative_path"],
        factory_source_sha256=binding["source_sha256"],
        bootstrap_context_sha256="1" * 64,
        pre_factory_campaign_state_sha256=state_sha256,
        post_factory_campaign_state_sha256=state_sha256,
        provider_boundary_receipt_sha256="2" * 64,
        credential_public_identity_sha256="3" * 64,
        expected_provider_public_contract_sha256=expected_contract,
        actual_provider_public_contract_sha256=expected_contract,
    )
    event = {
        "event_type": "pc01_provider_installation",
        "payload": {
            "installation_receipt": receipt.to_dict(),
            "installation_receipt_sha256": receipt.receipt_sha256,
            "locked_test_content": False,
        },
    }
    return manifest, event


def test_provider_installation_ledger_precedes_tasks_and_binds_actual_hash(
    tmp_path: Path,
) -> None:
    manifest, installation = _provider_installation_fixture(tmp_path)
    task_load = {
        "event_type": "episode_task_load",
        "payload": {"locked_test_content": False},
    }
    with pytest.raises(SchemaError, match="precedes required"):
        _validate_pc01_provider_installation_ledger(
            tmp_path,
            manifest,
            access_records=[task_load],
        )
    with pytest.raises(SchemaError, match="must precede"):
        _validate_pc01_provider_installation_ledger(
            tmp_path,
            manifest,
            access_records=[task_load, installation],
        )
    _validate_pc01_provider_installation_ledger(
        tmp_path,
        manifest,
        access_records=[installation, task_load],
    )

    tampered = deepcopy(installation)
    tampered["payload"]["installation_receipt"][
        "actual_provider_public_contract_sha256"
    ] = "d" * 64
    with pytest.raises(SchemaError, match="actual provider contract"):
        _validate_pc01_provider_installation_ledger(
            tmp_path,
            manifest,
            access_records=[tampered, task_load],
        )


def _manual_rescue_event(
    *,
    index: int,
    stage: str,
    previous: str,
    observation_id: str,
) -> tuple[dict, str]:
    check = WebArenaManualRescueCheck(
        episode_id="episode",
        task_id="webarena.0",
        task_sha256=SHA_A,
        guard_id="fixture-guard",
        guard_version="v1",
        evidence_mode="exclusive_controller_input_audit_v1",
        check_index=index,
        stage=stage,
        expected_registered_browser_steps=0,
        observation_id=observation_id,
        observation_sha256=SHA_B,
        action_id=None,
        action_sha256=None,
        previous_receipt_sha256=previous,
    )
    receipt = WebArenaManualRescueReceipt(
        guard_id="fixture-guard",
        guard_version="v1",
        source_sha256=SHA_A,
        evidence_mode="exclusive_controller_input_audit_v1",
        check_sha256=check.record_sha256,
        check_index=index,
        stage=stage,
        expected_registered_browser_steps=0,
        observed_registered_browser_steps=0,
        automation_session_sha256=SHA_B,
        controller_input_audit_sha256=canonical_sha256(check.to_dict()),
        exclusive_automation_control=True,
        non_agent_input_event_count=0,
        manual_rescue_detected=False,
    )
    return (
        {
            "event_type": "manual_rescue_guard",
            "payload": {
                "check": check.to_dict(),
                "check_sha256": check.record_sha256,
                "receipt": receipt.to_dict(),
                "receipt_sha256": receipt.record_sha256,
            },
        },
        receipt.record_sha256,
    )


def _protocol() -> dict:
    return {
        "budgets": {
            "executor_requests_per_episode": 30,
            "recovery_attempts_per_incident": 2,
            "recovery_attempts_per_episode": 4,
            "model_calls_per_episode": 102,
            "task_timeout_seconds": 600,
        }
    }


def _summary(*, steps: int = 0, attempts: int = 0, seconds: float = 1.0) -> dict:
    return {
        "step_count": steps,
        "recovery_attempt_count": attempts,
        "model_call_count": 0,
        "task_wall_clock_seconds": seconds,
        "terminal_reason": "CLOSED",
    }


def test_manual_rescue_package_guard_requires_clean_complete_chain() -> None:
    reset = {
        "event_type": "reset",
        "payload": {
            "observation_id": "reset-observation",
            "observation_record_sha256": SHA_B,
        },
    }
    after_reset, tail = _manual_rescue_event(
        index=1,
        stage="after_reset",
        previous="0" * 64,
        observation_id="reset-observation",
    )
    before_terminal, _ = _manual_rescue_event(
        index=2,
        stage="before_terminal",
        previous=tail,
        observation_id="reset-observation",
    )
    terminals = [
        {
            "event_type": "after_reset",
            "payload": {
                "receipt_binding": {
                    "observation_id": "reset-observation",
                    "observation_sha256": SHA_B,
                    "action_id": None,
                    "action_sha256": None,
                }
            },
        }
    ]
    kwargs = {
        "environment_events": [reset, after_reset, before_terminal],
        "actions": [],
        "terminal_signals": terminals,
        "environment": {
            "manual_rescue_guard": {
                "guard_id": "fixture-guard",
                "guard_version": "v1",
                "evidence_mode": "exclusive_controller_input_audit_v1",
            }
        },
        "campaign_manifest": {
            "runtime_integration": {"source_sha256": SHA_A}
        },
        "episode_id": "episode",
        "task_id": "webarena.0",
    }
    _validate_manual_rescue_guard_events(**kwargs)

    missing = dict(kwargs)
    missing["environment_events"] = [reset, after_reset]
    with pytest.raises(SchemaError, match="coverage differs"):
        _validate_manual_rescue_guard_events(**missing)

    tampered = deepcopy(kwargs)
    receipt = tampered["environment_events"][2]["payload"]["receipt"]
    receipt["non_agent_input_event_count"] = 1
    receipt["manual_rescue_detected"] = True
    receipt["record_type"] = "WebArenaManualRescueReceipt"
    with pytest.raises(SchemaError, match="digest differs|unclean"):
        _validate_manual_rescue_guard_events(**tampered)


def _normal_action(step: int) -> dict:
    return {
        "event_type": "normal_action",
        "payload": {
            "execution": {
                "action_id": f"normal-{step}",
                "executor_step": step,
                "status": "EXECUTED",
            }
        },
    }


def _recovery_action(step: int, attempt_id: str, action_id: str) -> dict:
    return {
        "event_type": "recovery_action",
        "payload": {
            "attempt_id": attempt_id,
            "action": {
                "action_id": action_id,
                "recovery_attempt_id": attempt_id,
            },
            "execution": {
                "action_id": action_id,
                "executor_step": step,
                "status": "EXECUTED",
            },
        },
    }


def _recovery_pair(
    *,
    attempt_id: str,
    incident_id: str,
    incident_index: int,
    episode_index: int,
    strategy: str = "REPLAN",
    action_ids: tuple[str, ...] = (),
    resolution_status: str = "REJECTED",
) -> list[dict]:
    return [
        {
            "event_type": "recovery_plan",
            "payload": {
                "plan": {
                    "attempt_id": attempt_id,
                    "incident_id": incident_id,
                    "strategy": strategy,
                    "incident_attempt_index": incident_index,
                    "episode_attempt_index": episode_index,
                    "actions": [
                        {
                            "action_id": action_id,
                            "recovery_attempt_id": attempt_id,
                        }
                        for action_id in action_ids
                    ],
                    "resolution_status": resolution_status,
                }
            },
        },
        {
            "event_type": "recovery_attempt",
            "payload": {
                "attempt": {
                    "attempt_id": attempt_id,
                    "incident_id": incident_id,
                    "strategy": strategy,
                    "incident_attempt_index": incident_index,
                    "episode_attempt_index": episode_index,
                    "action_ids": list(action_ids),
                    "completed": True,
                }
            },
        },
    ]


def _validate_budget(
    *,
    summary: dict,
    actions: list[dict] | None = None,
    recoveries: list[dict] | None = None,
    environment_events: list[dict] | None = None,
    evaluation_mode: bool = True,
) -> None:
    _validate_frozen_runtime_budget_evidence(
        summary=summary,
        actions=actions or [],
        recovery_events=recoveries or [],
        environment_events=environment_events or [],
        protocol=_protocol(),
        evaluation_mode=evaluation_mode,
    )


def _model_call(index: int, *, maximum: int = 102) -> dict:
    return {
        "event_type": "model_call_dispatch",
        "payload": {
            "episode_id": "episode-1",
            "model_call_index": index,
            "stage": "pre_action_policy",
            "component_id": "policy-v1",
            "maximum_model_calls": maximum,
        },
    }


def test_frozen_runtime_budget_accepts_exact_registered_boundaries():
    actions = [_normal_action(index) for index in range(1, 31)]
    recoveries: list[dict] = []
    for episode_index in range(1, 5):
        recoveries.extend(
            _recovery_pair(
                attempt_id=f"attempt-{episode_index}",
                incident_id=f"incident-{(episode_index + 1) // 2}",
                incident_index=1 if episode_index % 2 else 2,
                episode_index=episode_index,
            )
        )

    _validate_budget(
        summary={
            **_summary(steps=30, attempts=4, seconds=600.0),
            "episode_id": "episode-1",
            "model_call_count": 102,
        },
        actions=actions,
        recoveries=recoveries,
        environment_events=[_model_call(index) for index in range(1, 103)],
    )


def test_frozen_runtime_budget_replays_runner_owned_model_calls():
    summary = {
        **_summary(),
        "episode_id": "episode-1",
        "model_call_count": 2,
    }
    _validate_budget(
        summary=summary,
        environment_events=[_model_call(1), _model_call(2)],
    )

    with pytest.raises(SchemaError, match="dispatch count"):
        _validate_budget(summary=summary, environment_events=[_model_call(1)])

    with pytest.raises(SchemaError, match="sequential"):
        _validate_budget(
            summary=summary,
            environment_events=[_model_call(1), _model_call(3)],
        )

    with pytest.raises(SchemaError, match="different frozen episode cap"):
        _validate_budget(
            summary=summary,
            environment_events=[_model_call(1), _model_call(2, maximum=101)],
        )


def test_frozen_runtime_budget_rejects_model_call_overrun_and_missing_evidence():
    overrun = _summary()
    overrun["model_call_count"] = 103
    with pytest.raises(SchemaError, match="model-call budget"):
        _validate_budget(summary=overrun)

    missing = _summary()
    missing.pop("model_call_count")
    with pytest.raises(SchemaError, match="runner-owned model-call evidence"):
        _validate_budget(summary=missing)


def test_frozen_runtime_budget_rejects_requests_timeout_and_step_gaps():
    with pytest.raises(SchemaError, match="executor-request budget"):
        _validate_budget(
            summary=_summary(steps=31),
            actions=[_normal_action(index) for index in range(1, 32)],
        )

    with pytest.raises(SchemaError, match="task timeout"):
        _validate_budget(summary=_summary(seconds=600.000001))

    executor_elapsed = _summary(seconds=599.0)
    executor_elapsed["elapsed_seconds"] = 601.0
    with pytest.raises(SchemaError, match="elapsed_seconds.*task timeout"):
        _validate_budget(summary=executor_elapsed)

    with pytest.raises(SchemaError, match="sequential"):
        _validate_budget(
            summary=_summary(steps=2),
            actions=[_normal_action(1), _normal_action(3)],
        )

    uncharged = _normal_action(1)
    uncharged["payload"]["budget_consumed"] = False
    with pytest.raises(SchemaError, match="uncharged executor request"):
        _validate_budget(summary=_summary(), actions=[uncharged])


def test_frozen_runtime_budget_rejects_episode_and_incident_attempt_overruns():
    five_attempts: list[dict] = []
    for index in range(1, 6):
        five_attempts.extend(
            _recovery_pair(
                attempt_id=f"attempt-{index}",
                incident_id=f"incident-{index}",
                incident_index=1,
                episode_index=index,
            )
        )
    with pytest.raises(SchemaError, match="per-episode recovery budget"):
        _validate_budget(
            summary=_summary(attempts=5),
            recoveries=five_attempts,
        )

    three_for_one_incident: list[dict] = []
    for index in range(1, 4):
        three_for_one_incident.extend(
            _recovery_pair(
                attempt_id=f"incident-attempt-{index}",
                incident_id="incident-1",
                incident_index=index,
                episode_index=index,
            )
        )
    with pytest.raises(SchemaError, match="incident-1 exceeded"):
        _validate_budget(
            summary=_summary(attempts=3),
            recoveries=three_for_one_incident,
        )


@pytest.mark.parametrize(
    "strategy",
    ("RETRY", "REPLAN", "BACKTRACK", "ALTERNATIVE_TARGET"),
)
def test_each_nonabort_strategy_has_one_registered_low_level_action(strategy: str):
    attempt_id = f"{strategy.lower()}-attempt"
    valid_action_id = "recovery-action-1"
    valid_recoveries = _recovery_pair(
        attempt_id=attempt_id,
        incident_id="incident-1",
        incident_index=1,
        episode_index=1,
        strategy=strategy,
        action_ids=(valid_action_id,),
        resolution_status="READY",
    )
    _validate_budget(
        summary=_summary(steps=1, attempts=1),
        actions=[_recovery_action(1, attempt_id, valid_action_id)],
        recoveries=valid_recoveries,
    )

    actions = [
        _recovery_action(1, attempt_id, "recovery-action-1"),
        _recovery_action(2, attempt_id, "recovery-action-2"),
    ]
    recoveries = _recovery_pair(
        attempt_id=attempt_id,
        incident_id="incident-1",
        incident_index=1,
        episode_index=1,
        strategy=strategy,
        action_ids=("recovery-action-1", "recovery-action-2"),
        resolution_status="READY",
    )
    with pytest.raises(SchemaError, match="low-level action limit"):
        _validate_budget(
            summary=_summary(steps=2, attempts=1),
            actions=actions,
            recoveries=recoveries,
        )


def test_abort_zero_step_contract_is_replayed():
    valid_abort = _recovery_pair(
        attempt_id="abort-attempt",
        incident_id="incident-1",
        incident_index=1,
        episode_index=1,
        strategy="ABORT",
        resolution_status="ABORT",
    )
    _validate_budget(summary=_summary(attempts=1), recoveries=valid_abort)

    invalid_abort = deepcopy(valid_abort)
    invalid_abort[0]["payload"]["plan"]["actions"] = [
        {"action_id": "abort-action", "recovery_attempt_id": "abort-attempt"}
    ]
    invalid_abort[1]["payload"]["attempt"]["action_ids"] = ["abort-action"]
    with pytest.raises(SchemaError, match="low-level action limit"):
        _validate_budget(
            summary=_summary(steps=1, attempts=1),
            actions=[_recovery_action(1, "abort-attempt", "abort-action")],
            recoveries=invalid_abort,
        )


def test_abort_safe_terminal_truth_is_always_required():
    abort_attempt = _recovery_pair(
        attempt_id="abort-attempt",
        incident_id="incident-1",
        incident_index=1,
        episode_index=1,
        strategy="ABORT",
        resolution_status="ABORT",
    )[1]
    evidence = {
        "task_success": False,
        "terminal_reason": "ABORT",
        "recovery_verifications": [
            {
                "recovery_attempt_id": "abort-attempt",
                "successful": False,
            }
        ],
    }
    summary = {"runtime_terminal_reason": "ABORT"}
    _validate_abort_terminal_semantics(
        evidence=evidence,
        summary=summary,
        recovery_attempt_events=[abort_attempt],
    )

    successful = deepcopy(evidence)
    successful["task_success"] = True
    with pytest.raises(SchemaError, match="task success"):
        _validate_abort_terminal_semantics(
            evidence=successful,
            summary=summary,
            recovery_attempt_events=[abort_attempt],
        )

    successful_recovery = deepcopy(evidence)
    successful_recovery["recovery_verifications"][0]["successful"] = True
    with pytest.raises(SchemaError, match="recovery success"):
        _validate_abort_terminal_semantics(
            evidence=successful_recovery,
            summary=summary,
            recovery_attempt_events=[abort_attempt],
        )

    # Evaluator and runtime terminal taxonomies remain independent.  A sealed
    # evaluator may describe this as a generic failure while runtime records
    # the causal ABORT reason.
    _validate_abort_terminal_semantics(
        evidence={**evidence, "terminal_reason": "TERMINAL_FAILURE"},
        summary=summary,
        recovery_attempt_events=[abort_attempt],
    )

    with pytest.raises(SchemaError, match="runtime terminal_reason=ABORT"):
        _validate_abort_terminal_semantics(
            evidence=evidence,
            summary={"runtime_terminal_reason": "CLOSED"},
            recovery_attempt_events=[abort_attempt],
        )


def _validate_recovery_truth(
    *,
    successful: list[bool],
    resolved: bool,
    resolved_attempt_index: int | None,
    verification_order: list[int] | None = None,
) -> None:
    incident_id = "incident-1"
    attempt_events = [
        _recovery_pair(
            attempt_id=f"attempt-{index}",
            incident_id=incident_id,
            incident_index=index,
            episode_index=index,
        )[1]
        for index in range(1, len(successful) + 1)
    ]
    order = verification_order or list(range(1, len(successful) + 1))
    evidence = {
        "task_success": True,
        "terminal_reason": "TASK_SUCCESS",
        "loop_detected": False,
        "environment_failure": False,
        "failure_incidents": [
            {
                "failure_incident_id": incident_id,
                "verified_agent_failure": True,
                "resolved": resolved,
                "resolved_attempt_index": resolved_attempt_index,
                "attempt_count": len(successful),
            }
        ],
        "recovery_verifications": [
            {
                "recovery_attempt_id": f"attempt-{index}",
                "failure_incident_id": incident_id,
                "verified_failure_present": True,
                "successful": successful[index - 1],
            }
            for index in order
        ],
        "verified_failure_event_count": 1,
        "repeated_error_event_count": 0,
        "memory_relevance": {},
    }
    _validate_final_evidence(
        evidence,
        system_id="E2",
        summary={
            "runtime_terminal_reason": "OPAQUE_VERIFIER_TERMINAL",
            "infrastructure_invalid": False,
        },
        recovery_attempt_events=attempt_events,
        post_queries=[],
        context="manipulation-canary",
    )


def test_sealed_incident_resolution_matches_linked_successful_attempt():
    _validate_recovery_truth(
        successful=[False, True],
        resolved=True,
        resolved_attempt_index=2,
    )


@pytest.mark.parametrize("successful", ([True, False], [True, True]))
def test_first_verified_recovery_remains_resolved_when_runtime_attempts_again(
    successful: list[bool],
):
    _validate_recovery_truth(
        successful=successful,
        resolved=True,
        resolved_attempt_index=1,
    )


@pytest.mark.parametrize(
    ("successful", "resolved", "resolved_attempt_index", "message"),
    (
        ([False], True, 1, "resolved status differs"),
        ([True], False, None, "resolved status differs"),
        ([True], True, 2, "resolved_attempt_index differs"),
        ([False], False, 1, "cannot declare resolved_attempt_index"),
        ([True, True], True, 2, "resolved_attempt_index differs"),
    ),
)
def test_sealed_incident_resolution_manipulations_fail_closed(
    successful: list[bool],
    resolved: bool,
    resolved_attempt_index: int | None,
    message: str,
):
    with pytest.raises(SchemaError, match=message):
        _validate_recovery_truth(
            successful=successful,
            resolved=resolved,
            resolved_attempt_index=resolved_attempt_index,
        )


def test_sealed_recovery_verifications_must_preserve_runtime_attempt_order():
    with pytest.raises(SchemaError, match="ordered like runtime attempts"):
        _validate_recovery_truth(
            successful=[False, False],
            resolved=False,
            resolved_attempt_index=None,
            verification_order=[2, 1],
        )


def _guard(index: int, callback_kind: str, digest: str = SHA_A) -> dict:
    return {
        "event_type": "sealed_evaluator_backend_guard",
        "payload": {
            "guard_index": index,
            "callback_kind": callback_kind,
            "before_sha256": digest,
            "after_sha256": digest,
            "unchanged": True,
        },
    }


def test_evaluator_backend_guards_are_monotonic_and_cover_callbacks_one_to_one():
    terminal_signals = [
        {"event_type": "after_reset"},
        {"event_type": "after_normal_action"},
        {"event_type": "episode_final_receipt"},
    ]
    guards = [
        _guard(1, "transition_verifier"),
        _guard(2, "transition_verifier"),
        _guard(3, "episode_final_verifier"),
    ]
    _validate_evaluator_backend_guards(
        environment_events=guards,
        terminal_signals=terminal_signals,
        required=True,
    )

    missing = guards[:-1]
    with pytest.raises(SchemaError, match="1:1"):
        _validate_evaluator_backend_guards(
            environment_events=missing,
            terminal_signals=terminal_signals,
            required=True,
        )

    nonmonotonic = deepcopy(guards)
    nonmonotonic[1]["payload"]["guard_index"] = 3
    with pytest.raises(SchemaError, match="not monotonic"):
        _validate_evaluator_backend_guards(
            environment_events=nonmonotonic,
            terminal_signals=terminal_signals,
            required=True,
        )

    wrong_callback = deepcopy(guards)
    wrong_callback[1]["payload"]["callback_kind"] = "episode_final_verifier"
    with pytest.raises(SchemaError, match="callback order"):
        _validate_evaluator_backend_guards(
            environment_events=wrong_callback,
            terminal_signals=terminal_signals,
            required=True,
        )

    changed = deepcopy(guards)
    changed[0]["payload"]["after_sha256"] = SHA_B
    with pytest.raises(SchemaError, match="digest changed"):
        _validate_evaluator_backend_guards(
            environment_events=changed,
            terminal_signals=terminal_signals,
            required=True,
        )

    false_unchanged = deepcopy(guards)
    false_unchanged[0]["payload"]["unchanged"] = False
    with pytest.raises(SchemaError, match="unchanged=true"):
        _validate_evaluator_backend_guards(
            environment_events=false_unchanged,
            terminal_signals=terminal_signals,
            required=True,
        )


def test_custom_engineering_smoke_does_not_forge_backend_guard_evidence():
    _validate_evaluator_backend_guards(
        environment_events=[],
        terminal_signals=[{"event_type": "synthetic_smoke_receipt"}],
        required=False,
    )
