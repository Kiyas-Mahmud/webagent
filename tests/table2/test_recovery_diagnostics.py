from __future__ import annotations

import json

from web_agent.benchmarks.recovery_fixture import (
    RecoveryDiagnosticAdapter,
    RecoveryFixtureCampaignRunner,
    load_registered_recovery_diagnostics,
    run_failure_memory_intervention_smoke,
    run_recovery_diagnostic_fixture_campaign,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    SystemID,
    TaskSpecification,
    OpaqueTerminalSignal,
    canonical_sha256,
)
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.protocol import RuntimeProtocol


def test_all_registered_recovery_scenarios_execute_as_60_matched_episodes(tmp_path):
    results = run_recovery_diagnostic_fixture_campaign(tmp_path / "diagnostics")
    assert len(results) == 15 * 4 == 60
    assert {result.scenario_id for result in results} == {
        row.scenario_id for row in load_registered_recovery_diagnostics()[0]
    }
    for result in results:
        if result.system_id not in {SystemID.E2, SystemID.E3}:
            continue
        if result.expected_strategy.value == "ABORT":
            assert result.rule_passed is True
            assert result.recovery_resolved is False
        else:
            assert result.recovery_resolved is True
    assert all(
        result.actual_strategy is None
        for result in results
        if result.system_id in {SystemID.E0, SystemID.E1}
    )
    for result in results:
        runtime = (
            tmp_path
            / "diagnostics"
            / "paired_blocks"
            / "seed_42"
            / result.scenario_id
            / "repeat_0"
            / "rerun_0"
            / result.system_id.value
            / "runtime"
        )
        assert (runtime / "episode_manifest.json").is_file()
        assert (runtime / "rng_provenance.json").is_file()
        assert (runtime / "episode_summary.json").is_file()
        assert (runtime / "artifact_hashes.json").is_file()
        assert (runtime / "actions.jsonl").is_file()


def test_failure_memory_smoke_exercises_p1_p4_and_causal_order(tmp_path):
    summaries = run_failure_memory_intervention_smoke(tmp_path / "memory-smoke")
    assertions = json.loads(
        (tmp_path / "memory-smoke" / "smoke_assertions.json").read_text(
            encoding="utf-8"
        )
    )["assertions"]
    assert all(assertions.values())
    assert summaries[SystemID.E2].failure_incidents == 1
    assert summaries[SystemID.E2].memory_queries == 0
    assert summaries[SystemID.E3].memory_queries == 1
    assert summaries[SystemID.E3].memory_interventions == 1

    recovery_rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "memory-smoke"
            / "E2"
            / "runtime"
            / "recoveries.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    attempt = next(
        row["payload"]["transition"]
        for row in recovery_rows
        if row["event_type"] == "recovery_attempt"
    )
    pre_history = attempt["pre_recovery_observation"]["causal_history"]
    post_history = attempt["post_recovery_observation"]["causal_history"]
    assert len(pre_history) == 1
    assert len(post_history) == 2
    assert post_history[:1] == pre_history
    assert pre_history[-1]["recovery_attempt_id"] is None
    assert post_history[-1]["recovery_attempt_id"] == attempt["attempt_id"]
    assert post_history[-1]["post_observation_id"] == (
        attempt["post_recovery_observation"]["observation_id"]
    )
    history_text = json.dumps(post_history).lower()
    assert "parameters" not in history_text
    assert "message" not in history_text
    assert "oracle" not in history_text
    assert "verifier" not in history_text


def test_opaque_fixture_token_does_not_encode_changed_sealed_resolution():
    scenario = next(
        row
        for row in load_registered_recovery_diagnostics()[0]
        if row.scenario_id == "R04_missing_type_target"
    )
    task = TaskSpecification(
        task_id=scenario.scenario_id,
        goal="diagnostic",
        benchmark_id="recovery_fixture",
        benchmark_version="v1",
        start_state_id="failure-present",
        metadata={"task_partition": "recovery_diagnostic"},
    )
    unresolved = RecoveryDiagnosticAdapter(scenario)
    unresolved.reset(task, episode_id="same-episode", seed=42)
    unresolved_signal = unresolved.terminal_signal(task)

    resolved = RecoveryDiagnosticAdapter(scenario)
    resolved.reset(task, episode_id="same-episode", seed=42)
    resolved.execute(
        ConcreteAction(
            action_id="recovery",
            source_decision_id="decision",
            action_type=ActionType.TYPE,
            parameters={"target_x": 0.8, "target_y": 0.75, "text": "fixture"},
            bbox=(0.7, 0.7, 0.2, 0.1),
            recovery_attempt_id="attempt",
        )
    )
    resolved_signal = resolved.terminal_signal(task)
    assert unresolved_signal.event_id == resolved_signal.event_id
    assert unresolved_signal.token_sha256 == resolved_signal.token_sha256
    assert unresolved_signal.terminate is False
    assert resolved_signal.terminate is True


def test_recovery_fixture_bridge_uses_campaign_logs_and_sealed_finalizer(tmp_path):
    class Sink:
        def __init__(self):
            self.events = []
            self.final = None

        def record(self, evidence, *, should_terminate, event_kind):
            self.events.append((event_kind, evidence, should_terminate))
            index = len(self.events)
            return OpaqueTerminalSignal(
                event_id=f"sealed:{index}",
                token_sha256=canonical_sha256({"sealed_event": index}),
                terminate=should_terminate,
            )

        def record_bound_receipt(self, binding, evidence, *, should_terminate):
            return self.record(
                {**evidence, "runtime_binding": binding.to_dict()},
                should_terminate=should_terminate,
                event_kind=binding.receipt_kind,
            )

        def record_episode_final(self, evidence):
            self.final = evidence
            return self.record(
                {"final_receipt": True},
                should_terminate=True,
                event_kind="episode_final",
            )

    scenario = load_registered_recovery_diagnostics()[0][0]
    campaign_id = "bridge-campaign"
    episode_id = f"{campaign_id}:E2:{scenario.scenario_id}:repeat-0:seed-42"
    runtime = tmp_path / "runtime"
    logs = EpisodeEventLogs(
        runtime,
        episode_id=episode_id,
        include_memory=False,
        system_id="E2",
        task_id=scenario.scenario_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    protocol = RuntimeProtocol(
        protocol_id="bridge-protocol",
        campaign_id=campaign_id,
    )
    stage_seeds = protocol.rng_factory().seed_plan(
        task_id=scenario.scenario_id,
        repeat_id=0,
        matched_seed=42,
    ).stage_seeds
    sink = Sink()
    summary = RecoveryFixtureCampaignRunner().run(
        task={
            "task_id": scenario.scenario_id,
            "task_partition": "recovery_diagnostic",
            "failure_kind": scenario.failure_kind,
            "expected_strategy": scenario.expected_strategy.value,
            "resolution_event": scenario.resolution_event,
        },
        repeat_id=0,
        model_seed=42,
        system_id="E2",
        protocol={"protocol_id": protocol.protocol_id},
        stage_seeds=stage_seeds,
        runtime_dir=runtime,
        event_logs=logs,
        verifier_sink=sink,
        episode_id=episode_id,
    )
    assert summary.system_id is SystemID.E2
    assert sink.final is not None
    assert sink.final["diagnostic_evidence_only"] is True
    assert sink.final["recovery_verifications"][0]["successful"] is True
    terminal_rows = [
        json.loads(line)
        for line in (runtime / "terminal_signals.jsonl").read_text().splitlines()
    ]
    assert terminal_rows[-1]["event_type"] == "episode_final_receipt"
