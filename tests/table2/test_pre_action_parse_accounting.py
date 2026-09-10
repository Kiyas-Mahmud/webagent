from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from web_agent.benchmarks.fixture import DeterministicFixtureAdapter
from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.package_validator import (
    _validate_pre_action_parse_rejection,
)
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.contracts import (
    ActionType,
    ObservationStage,
    PreActionDecision,
    SystemID,
    TaskSpecification,
    TerminalReason,
    probability_map,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import Executor
from web_agent.runtime.observation import ObservationBuilder
from web_agent.runtime.policy import (
    ActionParseError,
    CallablePolicyAdapter,
    PolicyError,
    PolicyKind,
    SystemPolicy,
)
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.recovery.controller import RecoveryController


SHA = "a" * 64
BBOX = (0.10, 0.10, 0.20, 0.10)


class _CountingFixtureAdapter(DeterministicFixtureAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.execute_calls = 0
        self.post_action_observations = 0

    def execute(self, action):
        self.execute_calls += 1
        return super().execute(action)

    def observe(self, *, stage, prior_action_id=None):
        if stage in {ObservationStage.POST_ACTION, ObservationStage.POST_RECOVERY}:
            self.post_action_observations += 1
        return super().observe(stage=stage, prior_action_id=prior_action_id)


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="fixture-six-actions",
        goal="exercise parser request accounting",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _protocol() -> RuntimeProtocol:
    return RuntimeProtocol(
        protocol_id="parse-accounting-protocol",
        campaign_id="parse-accounting-campaign",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
    )


def _logs(root: Path, *, system_id: SystemID) -> EpisodeEventLogs:
    task = _task()
    episode_id = (
        f"parse-accounting-campaign:{system_id.value}:{task.task_id}:"
        "repeat-0:seed-42"
    )
    return EpisodeEventLogs(
        root,
        episode_id=episode_id,
        include_memory=False,
        system_id=system_id.value,
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )


def _read_actions(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_e0_parser_failure_consumes_one_rejected_step_without_browser_action(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "e0"
    adapter = _CountingFixtureAdapter()

    def invalid_parser_output(task, observation, rng):
        del task, observation, rng
        raise ActionParseError("base parser could not decode the action envelope")

    policy = SystemPolicy(
        CallablePolicyAdapter(
            policy_id="base-parser-policy",
            policy_version="v1",
            kind=PolicyKind.BASE,
            action_predictor=invalid_parser_output,
        ),
        switches_for(SystemID.E0),
    )
    protocol = _protocol()
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=policy,
        provider=DeterministicParameterProvider(),
        executor=Executor(adapter, budgets=protocol.budgets),
        event_logs=_logs(runtime, system_id=SystemID.E0),
    )
    summary = runner.run(_task(), repeat_id=0, model_seed=42)

    assert summary.terminal_reason is TerminalReason.POLICY_ERROR
    assert summary.executor_steps == summary.normal_actions == 1
    assert summary.recovery_actions == 0
    assert adapter.execute_calls == 0
    assert adapter.post_action_observations == 0
    receipt = runner.contract_validation_receipt
    assert receipt is not None
    assert receipt["validation"]["status"] == "PASS"
    assert receipt["validation"]["actions"] == 0
    assert receipt["validation"]["executions"] == 1

    actions = _read_actions(runtime)
    assert len(actions) == 1
    assert actions[0]["event_type"] == "pre_action_parse_rejection"
    payload = actions[0]["payload"]
    assert payload["decision"] is None
    assert payload["parameters"] is None
    assert payload["action"] is None
    assert payload["action_sha256"] is None
    assert payload["execution"]["status"] == "rejected"
    assert payload["execution"]["executor_step"] == 1
    assert payload["execution"]["error_kind"] == "pre_action_parse_rejected"
    assert "base parser could not decode" not in json.dumps(payload)
    _validate_pre_action_parse_rejection(payload, system_id="E0")

    tampered = dict(payload)
    tampered["action"] = {}
    with pytest.raises(SchemaError, match="must not invent an action"):
        _validate_pre_action_parse_rejection(tampered, system_id="E0")

    mismatched = dict(payload)
    mismatched["execution"] = {
        **payload["execution"],
        "action_id": "different-request",
    }
    with pytest.raises(SchemaError, match="receipt is inconsistent"):
        _validate_pre_action_parse_rejection(mismatched, system_id="E0")
    with pytest.raises(SchemaError, match="registered only for E0"):
        _validate_pre_action_parse_rejection(payload, system_id="E1")


def test_invalid_returned_base_contract_is_classified_as_parse_error() -> None:
    adapter = CallablePolicyAdapter(
        policy_id="base-parser-policy",
        policy_version="v1",
        kind=PolicyKind.BASE,
        action_predictor=lambda *_: object(),
    )
    fixture = DeterministicFixtureAdapter()
    task = _task()
    observation = fixture.reset(task, episode_id="parse-unit", seed=42)
    policy_observation = ObservationBuilder().pre_action(task, observation)
    with pytest.raises(ActionParseError, match="invalid contract"):
        adapter.predict_action(task, policy_observation, rng=random.Random(0))


def test_trained_pre_action_error_cannot_use_e0_rejection_accounting(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "e1"
    fixture = _CountingFixtureAdapter()

    def invalid_trained_output(task, observation, rng):
        del task, observation, rng
        raise ActionParseError("trained head emitted an invalid value")

    policy = SystemPolicy(
        CallablePolicyAdapter(
            policy_id="trained-policy",
            policy_version="v1",
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=SHA,
            action_predictor=invalid_trained_output,
        ),
        switches_for(SystemID.E1),
    )
    protocol = _protocol()
    summary = EpisodeRunner(
        protocol=protocol,
        system_policy=policy,
        provider=DeterministicParameterProvider(),
        executor=Executor(fixture, budgets=protocol.budgets),
        event_logs=_logs(runtime, system_id=SystemID.E1),
    ).run(_task(), repeat_id=0, model_seed=42)

    assert summary.terminal_reason is TerminalReason.POLICY_ERROR
    assert summary.executor_steps == summary.normal_actions == 0
    assert fixture.execute_calls == 0
    assert _read_actions(runtime) == []


def test_post_action_policy_error_does_not_add_rejected_executor_step(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "e2"
    adapter = _CountingFixtureAdapter()
    policy_id = "trained-post-action-policy"

    def action_predictor(task, observation, rng):
        del task, rng
        return PreActionDecision(
            decision_id=f"decision:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=ActionType.CLICK,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType),
                ActionType.CLICK.value,
            ),
            bbox=BBOX,
            grounding_confidence=1.0,
            confidence_before=1.0,
            input_observation_ids=(observation.observation_id,),
            policy_id=policy_id,
            policy_version="v1",
        )

    def failed_post_action_assessment(task, transition, rng):
        del task, transition, rng
        raise PolicyError("unrelated post-action head failure")

    def unused_recovery_assessment(task, transition, rng):
        del task, transition, rng
        raise AssertionError("recovery assessment must not run")

    policy = SystemPolicy(
        CallablePolicyAdapter(
            policy_id=policy_id,
            policy_version="v1",
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=SHA,
            action_predictor=action_predictor,
            transition_predictor=failed_post_action_assessment,
            recovery_predictor=unused_recovery_assessment,
        ),
        switches_for(SystemID.E2),
    )
    protocol = _protocol()
    summary = EpisodeRunner(
        protocol=protocol,
        system_policy=policy,
        provider=DeterministicParameterProvider(),
        executor=Executor(adapter, budgets=protocol.budgets),
        recovery_controller=RecoveryController(protocol.budgets),
        event_logs=_logs(runtime, system_id=SystemID.E2),
    ).run(_task(), repeat_id=0, model_seed=42)

    assert summary.terminal_reason is TerminalReason.POLICY_ERROR
    assert summary.executor_steps == summary.normal_actions == 1
    assert summary.recovery_actions == 0
    assert adapter.execute_calls == 1
    assert adapter.post_action_observations == 1
    actions = _read_actions(runtime)
    assert [row["event_type"] for row in actions] == ["normal_action"]
