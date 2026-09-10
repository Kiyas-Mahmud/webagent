"""Executable deterministic recovery diagnostics for the Table 2 pilot.

The runtime sees only observable fixture state and executor outcomes. Registered
oracle rules are evaluated from frozen logs after ``EpisodeRunner.run`` returns
and are written below a sibling ``sealed`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from web_agent.benchmarks.base import AdapterExecution, BenchmarkAdapter, BenchmarkStateError
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    EpisodeSummary,
    ExecutionStatus,
    JsonValue,
    MemoryCandidate,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryStrategy,
    RecoveryTransitionInput,
    RuntimeTaskView,
    SystemID,
    TaskSpecification,
    TerminalReason,
    TransitionAssessment,
    TransitionInput,
    VerifierReceiptBinding,
    canonical_json,
    canonical_sha256,
    probability_map,
    runtime_task_view,
)
from web_agent.runtime.duplicate_audit import (
    FrozenDuplicateAuditManifest,
    validate_pilot_duplicate_audit_manifest,
)
from web_agent.runtime.event_log import RUNTIME_EVENT_FILES, verify_event_log
from web_agent.runtime.recovery.strategies import (
    RECOVERY_TARGET_EVIDENCE_KEY,
    build_recovery_target_evidence,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PILOT_ROOT = _REPO_ROOT / "benchmarks" / "table2" / "pilot"
_BBOX = (0.10, 0.10, 0.20, 0.10)
_ALTERNATIVE_BBOX = (0.70, 0.70, 0.20, 0.10)


@dataclass(frozen=True, slots=True)
class RecoveryScenarioRegistration:
    scenario_id: str
    failure_kind: str
    expected_strategy: RecoveryStrategy
    resolution_event: str


@dataclass(frozen=True, slots=True)
class RecoveryOracleRule:
    rule_id: str
    requirements: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class RecoveryDiagnosticResult:
    scenario_id: str
    system_id: SystemID
    episode_id: str
    expected_strategy: RecoveryStrategy
    actual_strategy: RecoveryStrategy | None
    strategy_match: bool
    resolution_event: str
    rule_passed: bool
    recovery_resolved: bool
    task_success: bool
    terminal_reason: TerminalReason
    executor_steps: int
    runtime_evidence_sha256: str
    oracle_rule_sha256: str
    labels_attached_after_execution: bool = True
    runtime_read_access: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "table2-recovery-diagnostic-result-v1",
            "scenario_id": self.scenario_id,
            "system_id": self.system_id.value,
            "episode_id": self.episode_id,
            "expected_strategy": self.expected_strategy.value,
            "actual_strategy": (
                self.actual_strategy.value if self.actual_strategy is not None else None
            ),
            "strategy_match": self.strategy_match,
            "resolution_event": self.resolution_event,
            "rule_passed": self.rule_passed,
            "recovery_resolved": self.recovery_resolved,
            "task_success": self.task_success,
            "terminal_reason": self.terminal_reason.value,
            "executor_steps": self.executor_steps,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "oracle_rule_sha256": self.oracle_rule_sha256,
            "labels_attached_after_execution": self.labels_attached_after_execution,
            "runtime_read_access": self.runtime_read_access,
        }


class RecoveryDiagnosticAdapter(BenchmarkAdapter):
    """One observable failure followed by a bounded recovery opportunity."""

    benchmark_id = "table2-recovery-fixture"
    benchmark_version = "v1"

    def __init__(
        self,
        scenario: RecoveryScenarioRegistration,
        *,
        accept_any_recovery: bool = False,
        terminal_signal_callback: Callable[
            [
                RecoveryScenarioRegistration,
                bool,
                int,
                str,
                VerifierReceiptBinding | None,
            ],
            OpaqueTerminalSignal,
        ]
        | None = None,
    ) -> None:
        self.scenario = scenario
        self.accept_any_recovery = accept_any_recovery
        self.terminal_signal_callback = terminal_signal_callback
        self._episode_id: str | None = None
        self._task_view: RuntimeTaskView | None = None
        self._observation_index = 0
        self._verification_index = 0
        self._resolved = False
        self._closed = False
        self._last_normal_action: ConcreteAction | None = None
        self.transcript: list[dict[str, JsonValue]] = []

    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        del seed
        if self._closed:
            raise BenchmarkStateError("recovery diagnostic adapter is closed")
        if task.task_id != self.scenario.scenario_id:
            raise BenchmarkStateError("recovery task/scenario mismatch")
        if task.benchmark_id not in {self.benchmark_id, "recovery_fixture"}:
            raise BenchmarkStateError("recovery task benchmark mismatch")
        self._episode_id = episode_id
        self._task_view = runtime_task_view(task)
        self._observation_index = 0
        self._verification_index = 0
        self._resolved = False
        self._last_normal_action = None
        self.transcript.clear()
        return self.observe(stage=ObservationStage.RESET)

    def observe(
        self,
        *,
        stage: ObservationStage,
        prior_action_id: str | None = None,
    ) -> Observation:
        if self._episode_id is None:
            raise BenchmarkStateError("recovery fixture has not been reset")
        if stage in {ObservationStage.POST_ACTION, ObservationStage.POST_RECOVERY}:
            if not prior_action_id:
                raise BenchmarkStateError("post-action observation requires action ID")
        self._observation_index += 1
        observation_id = f"{self._episode_id}:obs:{self._observation_index}"
        if self._task_view is None:
            raise BenchmarkStateError("recovery fixture has no current task view")
        state_id = "resolved" if self._resolved else "failure-present"
        flags = _observable_resolution_flags(
            self.scenario.resolution_event,
            resolved=self._resolved,
        )
        observable: dict[str, JsonValue] = {
            "scenario_id": self.scenario.scenario_id,
            "state_id": state_id,
            "failure_kind": self.scenario.failure_kind,
            "diagnostic_flags": flags,
        }
        if self.scenario.failure_kind == "INVALID_PARAMETER":
            observable["observable_select_controls"] = [
                {
                    "target_bbox": list(_BBOX),
                    "candidate_options": ["stale-option"],
                }
            ]
        # This commitment represents rendered page content. The target receipt
        # added below is controller metadata with an observation-specific ID;
        # including it would give identical E2/E3 pixels different hashes.
        screenshot_sha256 = canonical_sha256(observable)
        observable[RECOVERY_TARGET_EVIDENCE_KEY] = build_recovery_target_evidence(
            task=self._task_view,
            observation_id=observation_id,
            compatible_actions=_diagnostic_visible_compatible_actions(
                self.scenario.failure_kind,
                self._last_normal_action,
            ),
        )
        return Observation(
            observation_id=observation_id,
            episode_id=self._episode_id,
            stage=stage,
            screenshot_sha256=screenshot_sha256,
            width=1280,
            height=720,
            url=f"fixture://recovery/{self.scenario.scenario_id}/{state_id}",
            title=f"Recovery diagnostic {self.scenario.scenario_id}",
            page_state=observable,
            page_settled=True,
            environment_error=False,
            prior_action_id=prior_action_id,
        )

    def execute(self, action: ConcreteAction) -> AdapterExecution:
        if self._episode_id is None:
            raise BenchmarkStateError("recovery fixture has not been reset")
        is_recovery = action.recovery_attempt_id is not None
        if not is_recovery:
            self._last_normal_action = action
            status, error_kind = _registered_initial_outcome(self.scenario.failure_kind)
            self.transcript.append(
                {
                    "phase": "normal",
                    "action_id": action.action_id,
                    "action_fingerprint": action.fingerprint,
                    "status": status.value,
                    "state_changed": False,
                    "target_changed": False,
                }
            )
            return AdapterExecution(
                status=status,
                state_changed=False,
                environment_error=False,
                error_kind=error_kind,
                message=f"registered diagnostic failure {self.scenario.failure_kind}",
            )

        valid, target_changed = self._validate_recovery(action)
        if valid:
            self._resolved = True
        self.transcript.append(
            {
                "phase": "recovery",
                "action_id": action.action_id,
                "action_fingerprint": action.fingerprint,
                "status": ExecutionStatus.EXECUTED.value,
                "state_changed": valid,
                "target_changed": target_changed,
            }
        )
        return AdapterExecution(
            status=ExecutionStatus.EXECUTED,
            state_changed=valid,
            environment_error=False,
            message=(
                f"resolved {self.scenario.resolution_event}"
                if valid
                else "recovery action did not satisfy the observable fixture contract"
            ),
        )

    def _validate_recovery(self, action: ConcreteAction) -> tuple[bool, bool]:
        failed = self._last_normal_action
        if failed is None:
            # Parameter-resolution rejection is charged by the shared executor
            # and therefore never reaches the benchmark adapter.  The fixture
            # still permits a causally planned, newly grounded TYPE recovery.
            if self.scenario.failure_kind == "TARGET_MISSING":
                return _valid_replanned_action(self.scenario.failure_kind, action), True
            return False, False
        target_changed = action.fingerprint != failed.fingerprint
        if self.accept_any_recovery:
            return True, target_changed
        strategy = self.scenario.expected_strategy
        if strategy is RecoveryStrategy.RETRY:
            return action.fingerprint == failed.fingerprint, False
        if strategy is RecoveryStrategy.BACKTRACK:
            valid = (
                action.action_type is ActionType.PRESS_KEY
                and action.parameters.get("key") == "ALT+LEFT"
            )
            return valid, target_changed
        if strategy is RecoveryStrategy.REPLAN:
            return _valid_replanned_action(self.scenario.failure_kind, action), target_changed
        if strategy is RecoveryStrategy.ALTERNATIVE_TARGET:
            return target_changed and action.bbox == _ALTERNATIVE_BBOX, target_changed
        return False, target_changed

    def terminal_signal(
        self,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None = None,
    ) -> OpaqueTerminalSignal:
        if task.task_id != self.scenario.scenario_id or self._episode_id is None:
            raise BenchmarkStateError("terminal task does not match recovery fixture")
        self._verification_index += 1
        event_id = f"{self._episode_id}:terminal:{self._verification_index}"
        if self.terminal_signal_callback is not None:
            signal = self.terminal_signal_callback(
                self.scenario,
                self._resolved,
                self._verification_index,
                event_id,
                binding,
            )
            if not isinstance(signal, OpaqueTerminalSignal):
                raise BenchmarkStateError(
                    "sealed terminal callback returned an invalid opaque signal"
                )
            return signal
        return OpaqueTerminalSignal(
            event_id=event_id,
            token_sha256=canonical_sha256(
                {
                    "event_id": event_id,
                    "benchmark_id": self.benchmark_id,
                    "opaque_receipt_domain": "recovery-fixture-terminal-v1",
                    "causal_binding_sha256": (
                        binding.record_sha256 if binding is not None else None
                    ),
                }
            ),
            terminate=self._resolved,
        )

    def close(self) -> None:
        self._closed = True
        self._episode_id = None
        self._task_view = None


def load_registered_recovery_diagnostics(
    *,
    scenario_path: str | Path = _PILOT_ROOT / "recovery_scenarios.json",
    rule_path: str | Path = _PILOT_ROOT / "recovery_oracle_rules.json",
) -> tuple[
    tuple[RecoveryScenarioRegistration, ...],
    Mapping[str, RecoveryOracleRule],
]:
    scenario_payload = _read_json_object(scenario_path)
    if scenario_payload.get("schema_version") != "1.0":
        raise ValueError("recovery scenario schema version mismatch")
    if scenario_payload.get("deterministic") is not True:
        raise ValueError("recovery scenarios must be deterministic")
    rows = scenario_payload.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError("recovery scenarios must be an array")
    scenarios: list[RecoveryScenarioRegistration] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"recovery scenario[{index}] is not an object")
        if set(row) != {
            "scenario_id",
            "failure_kind",
            "expected_strategy",
            "resolution_event",
        }:
            raise ValueError(f"recovery scenario[{index}] fields are not registered")
        scenarios.append(
            RecoveryScenarioRegistration(
                scenario_id=str(row["scenario_id"]),
                failure_kind=str(row["failure_kind"]),
                expected_strategy=RecoveryStrategy(str(row["expected_strategy"])),
                resolution_event=str(row["resolution_event"]),
            )
        )
    required_count = scenario_payload.get("required_scenario_count")
    if type(required_count) is not int or required_count != 15:
        raise ValueError("recovery manifest must register exactly 15 scenarios")
    if len(scenarios) != required_count or len({row.scenario_id for row in scenarios}) != 15:
        raise ValueError("recovery scenario count/uniqueness mismatch")

    rule_payload = _read_json_object(rule_path)
    if (
        rule_payload.get("schema_version") != "1.0"
        or rule_payload.get("sealed") is not True
        or rule_payload.get("attach_after_execution") is not True
        or rule_payload.get("runtime_read_access") is not False
    ):
        raise ValueError("recovery oracle boundary is not registered")
    raw_rules = rule_payload.get("rules")
    if not isinstance(raw_rules, Mapping):
        raise ValueError("recovery oracle rules must be an object")
    rules = {
        str(rule_id): RecoveryOracleRule(str(rule_id), dict(requirements))
        for rule_id, requirements in raw_rules.items()
        if isinstance(requirements, Mapping)
    }
    missing_rules = {row.resolution_event for row in scenarios} - set(rules)
    if missing_rules or len(rules) != len(raw_rules):
        raise ValueError(f"recovery scenario oracle rules are incomplete: {missing_rules}")
    return tuple(scenarios), rules


def load_pilot_duplicate_audit_registration() -> FrozenDuplicateAuditManifest:
    task_payload = _read_json_object(_PILOT_ROOT / "task_manifest.json")
    recovery_payload = _read_json_object(_PILOT_ROOT / "recovery_scenarios.json")
    normal_ids = tuple(str(row["task_id"]) for row in task_payload["tasks"])
    recovery_ids = tuple(
        str(row["scenario_id"]) for row in recovery_payload["scenarios"]
    )
    return validate_pilot_duplicate_audit_manifest(
        _PILOT_ROOT / "duplicate_audit_manifest.json",
        normal_task_ids=normal_ids,
        recovery_scenario_ids=recovery_ids,
        require_normal_verified=False,
    )


class RecoveryFixtureCampaignRunner:
    """Bridge the 15 diagnostics into the canonical paired CampaignRunner.

    The default backend is the deterministic engineering policy used by local
    smoke tests.  A real pilot launcher must inject ``episode_executor``; that
    callback receives the sealed adapter, frozen stage seeds, existing
    CampaignRunner logs, and duplicate registry so it can bind the
    validation-selected policy/provider/memory without changing the benchmark.
    """

    def __init__(
        self,
        *,
        episode_executor: Callable[[Mapping[str, Any]], EpisodeSummary] | None = None,
        duplicate_audit: FrozenDuplicateAuditManifest | None = None,
        scenario_path: str | Path = _PILOT_ROOT / "recovery_scenarios.json",
        rule_path: str | Path = _PILOT_ROOT / "recovery_oracle_rules.json",
    ) -> None:
        self.episode_executor = episode_executor
        self.duplicate_audit = duplicate_audit or load_pilot_duplicate_audit_registration()
        # Evaluation callers bind these paths to the copied campaign closure.
        # Repository defaults remain available only for the explicit local
        # engineering-smoke helpers.
        scenarios, rules = load_registered_recovery_diagnostics(
            scenario_path=scenario_path,
            rule_path=rule_path,
        )
        self._scenarios = {row.scenario_id: row for row in scenarios}
        self._rules = dict(rules)

    @property
    def evidence_mode(self) -> str:
        return (
            "EXTERNAL_VALIDATION_SELECTED_BINDING"
            if self.episode_executor is not None
            else "ENGINEERING_DIAGNOSTIC_ONLY"
        )

    def run(
        self,
        *,
        task: Mapping[str, Any],
        repeat_id: int,
        model_seed: int,
        system_id: str,
        protocol: Mapping[str, Any],
        stage_seeds: Mapping[str, int],
        runtime_dir: str | Path,
        event_logs: Any,
        verifier_sink: Any,
        episode_id: str,
        **context: Any,
    ) -> EpisodeSummary:
        task_id = str(task.get("task_id") or task.get("scenario_id") or "")
        try:
            scenario = self._scenarios[task_id]
        except KeyError as exc:
            raise ValueError(f"campaign task is not a registered recovery scenario: {task_id}") from exc
        if str(task.get("task_partition")) != "recovery_diagnostic":
            raise ValueError("recovery fixture runner cannot execute a normal benchmark task")
        for key, expected in (
            ("failure_kind", scenario.failure_kind),
            ("expected_strategy", scenario.expected_strategy.value),
            ("resolution_event", scenario.resolution_event),
        ):
            if str(task.get(key)) != expected:
                raise ValueError(f"frozen recovery scenario field changed: {key}")
        resolved_system = SystemID(system_id)
        campaign_marker = f":{resolved_system.value}:{task_id}:"
        if campaign_marker not in episode_id:
            raise ValueError("campaign episode identity does not match recovery scenario")
        campaign_id = episode_id.split(campaign_marker, 1)[0]
        protocol_id = str(protocol.get("protocol_id") or "")
        if not protocol_id:
            raise ValueError("frozen campaign protocol lacks protocol_id")

        def terminal_callback(
            registered: RecoveryScenarioRegistration,
            resolved: bool,
            verification_index: int,
            local_event_id: str,
            binding: VerifierReceiptBinding | None,
        ) -> OpaqueTerminalSignal:
            del local_event_id
            if binding is None:
                raise ValueError("canonical recovery verifier requires causal binding")
            return verifier_sink.record_bound_receipt(
                binding,
                {
                    "scenario_id": registered.scenario_id,
                    "verification_index": verification_index,
                    "diagnostic_transition_resolved": resolved,
                },
                should_terminate=resolved,
            )

        adapter = RecoveryDiagnosticAdapter(
            scenario,
            terminal_signal_callback=terminal_callback,
        )
        if self.episode_executor is None:
            summary = _run_recovery_episode(
                scenario,
                system_id=resolved_system,
                adapter=adapter,
                runtime_dir=Path(runtime_dir),
                campaign_id=campaign_id,
                repeat_id=repeat_id,
                matched_seed=model_seed,
                duplicate_audit=self.duplicate_audit,
                event_logs_override=event_logs,
                stage_seeds_override=stage_seeds,
                expected_episode_id=episode_id,
                protocol_id=protocol_id,
                campaign_seed=int(
                    (
                        protocol.get("statistics", {})
                        if isinstance(protocol.get("statistics", {}), Mapping)
                        else {}
                    ).get("seed", 20_250_831)
                ),
                require_frozen_stage_seed_plan=True,
            )
        else:
            summary = self.episode_executor(
                {
                    "scenario": scenario,
                    "task": dict(task),
                    "system_id": resolved_system,
                    "adapter": adapter,
                    "repeat_id": repeat_id,
                    "model_seed": model_seed,
                    "protocol": dict(protocol),
                    "stage_seeds": dict(stage_seeds),
                    "runtime_dir": Path(runtime_dir),
                    "event_logs": event_logs,
                    "episode_id": episode_id,
                    "duplicate_audit": self.duplicate_audit,
                    "campaign_context": dict(context),
                }
            )
            if not isinstance(summary, EpisodeSummary):
                raise TypeError("external recovery episode executor returned an invalid summary")
        evidence = _campaign_final_recovery_evidence(
            scenario,
            self._rules[scenario.resolution_event],
            summary=summary,
            runtime_dir=Path(runtime_dir),
            adapter_transcript=tuple(adapter.transcript),
        )
        final_signal = verifier_sink.record_episode_final(evidence)
        event_logs.append("terminal_signals", "episode_final_receipt", final_signal)
        return summary
def run_recovery_diagnostic_fixture_campaign(
    output_root: str | Path,
    *,
    campaign_id: str = "table2-recovery-diagnostic-smoke",
    repeat_id: int = 0,
    matched_seed: int = 42,
) -> tuple[RecoveryDiagnosticResult, ...]:
    """Execute the registered 15 x E0--E3 = 60 diagnostic episodes."""

    destination = _require_new_output_root(output_root)
    scenarios, rules = load_registered_recovery_diagnostics()
    duplicate_audit = load_pilot_duplicate_audit_registration()
    results: list[RecoveryDiagnosticResult] = []
    for scenario in scenarios:
        block = (
            destination
            / "paired_blocks"
            / f"seed_{matched_seed}"
            / scenario.scenario_id
            / f"repeat_{repeat_id}"
            / "rerun_0"
        )
        block.mkdir(parents=True, exist_ok=False)
        statuses: dict[str, Any] = {}
        for system_id in SystemID:
            package = block / system_id.value
            runtime_dir = package / "runtime"
            sealed_dir = package / "sealed"
            runtime_dir.mkdir(parents=True)
            sealed_dir.mkdir(parents=True)
            adapter = RecoveryDiagnosticAdapter(scenario)
            summary = _run_recovery_episode(
                scenario,
                system_id=system_id,
                adapter=adapter,
                runtime_dir=runtime_dir,
                campaign_id=campaign_id,
                repeat_id=repeat_id,
                matched_seed=matched_seed,
                duplicate_audit=duplicate_audit,
            )
            (runtime_dir / "episode_summary.json").write_text(
                summary.to_json(indent=2) + "\n",
                encoding="utf-8",
            )
            _write_engineering_artifact_hashes(runtime_dir)
            result = score_recovery_diagnostic_after_execution(
                scenario,
                rules[scenario.resolution_event],
                summary=summary,
                runtime_dir=runtime_dir,
                adapter_transcript=tuple(adapter.transcript),
                sealed_dir=sealed_dir,
            )
            results.append(result)
            statuses[system_id.value] = {
                "launched": True,
                "completed": True,
                "infrastructure_invalid": False,
                "episode_id": summary.episode_id,
                "error": None,
                "runtime_summary_sha256": canonical_sha256(summary.to_dict()),
                "sealed_result_sha256": canonical_sha256(result.to_dict()),
            }
        (block / "block_manifest.json").write_text(
            canonical_json(
                {
                    "schema_version": "table2-recovery-diagnostic-block-v1",
                    "block_id": (
                        f"task_{scenario.scenario_id}__seed_{matched_seed}"
                        f"__repeat_{repeat_id:03d}"
                    ),
                    "attempt_id": 0,
                    "campaign_id": campaign_id,
                    "scenario_id": scenario.scenario_id,
                    "repeat_id": repeat_id,
                    "matched_seed": matched_seed,
                    "evidence_label": "ENGINEERING_DIAGNOSTIC_ONLY",
                    "systems": statuses,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    if len(results) != 60:
        raise RuntimeError(f"registered recovery campaign produced {len(results)} episodes")
    (destination / "diagnostic_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "table2-recovery-diagnostic-summary-v1",
                "campaign_id": campaign_id,
                "evidence_label": "ENGINEERING_DIAGNOSTIC_ONLY",
                "paper_table_status": "N/R",
                "registered_scenarios": 15,
                "systems_per_scenario": 4,
                "episode_count": len(results),
                "results": [result.to_dict() for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(results)


def run_failure_memory_intervention_smoke(
    output_root: str | Path,
    *,
    campaign_id: str = "table2-p1-p4-intervention-smoke",
    matched_seed: int = 42,
) -> Mapping[SystemID, EpisodeSummary]:
    """Exercise P1, recovery, and a real E3 intervention with causal logs."""

    destination = _require_new_output_root(output_root)
    scenarios, _ = load_registered_recovery_diagnostics()
    scenario = next(row for row in scenarios if row.scenario_id == "R03_click_no_change")
    duplicate_audit = load_pilot_duplicate_audit_registration()
    summaries: dict[SystemID, EpisodeSummary] = {}
    for system_id in SystemID:
        runtime_dir = destination / system_id.value / "runtime"
        runtime_dir.mkdir(parents=True)
        summaries[system_id] = _run_recovery_episode(
            scenario,
            system_id=system_id,
            adapter=RecoveryDiagnosticAdapter(
                scenario,
                accept_any_recovery=True,
            ),
            runtime_dir=runtime_dir,
            campaign_id=campaign_id,
            repeat_id=0,
            matched_seed=matched_seed,
            duplicate_audit=duplicate_audit,
            memory_candidate=(
                MemoryCandidate(
                    memory_id="train-memory-intervention-smoke",
                    source_split="train",
                    source_task_id="train.fixture.other-task",
                    source_episode_id="train.fixture.other-episode",
                    duplicate_cluster_id="train.fixture.nonduplicate-cluster",
                    strategy=RecoveryStrategy.RETRY,
                    similarity=0.99,
                    memory_update_flag=True,
                    verified_recovery_success=True,
                    final_task_success=True,
                    advice="retry the transient no-effect action once",
                )
                if system_id is SystemID.E3
                else None
            ),
        )
        (runtime_dir / "episode_summary.json").write_text(
            summaries[system_id].to_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _write_engineering_artifact_hashes(runtime_dir)
    e2 = summaries[SystemID.E2]
    e3 = summaries[SystemID.E3]
    e3_events = _read_jsonl(destination / "E3" / "runtime" / "memory_queries.jsonl")
    event_types = [str(row["event_type"]) for row in e3_events]
    shadow_index = event_types.index("no_memory_shadow_before_retrieval")
    query_index = event_types.index("post_failure_query")
    assertions = {
        "p1_transition_exercised": summaries[SystemID.E2].failure_incidents > 0,
        "recovery_exercised": summaries[SystemID.E2].recovery_attempts > 0,
        "e2_zero_retrieval": e2.memory_queries == 0,
        "e2_memory_log_absent": not (
            destination / "E2" / "runtime" / "memory_queries.jsonl"
        ).exists(),
        "e3_query_exercised": e3.memory_queries == 1,
        "e3_intervention_exercised": e3.memory_interventions == 1,
        "e3_no_writes": all("write" not in item.lower() for item in event_types),
        "shadow_before_query": shadow_index < query_index,
    }
    if not all(assertions.values()):
        raise RuntimeError(f"failure/memory smoke invariant failed: {assertions}")
    (destination / "smoke_assertions.json").write_text(
        json.dumps(
            {
                "schema_version": "table2-p1-p4-smoke-v1",
                "evidence_label": "ENGINEERING_SMOKE_ONLY",
                "paper_table_status": "N/R",
                "assertions": assertions,
                "summaries": {
                    key.value: value.to_dict() for key, value in summaries.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summaries


def score_recovery_diagnostic_after_execution(
    scenario: RecoveryScenarioRegistration,
    rule: RecoveryOracleRule,
    *,
    summary: EpisodeSummary,
    runtime_dir: str | Path,
    adapter_transcript: Sequence[Mapping[str, JsonValue]],
    sealed_dir: str | Path,
) -> RecoveryDiagnosticResult:
    """Attach deterministic labels only after runtime action completion."""

    root = Path(runtime_dir)
    event_hashes: dict[str, str] = {}
    for stream, filename in RUNTIME_EVENT_FILES.items():
        path = root / filename
        if path.exists():
            verification = verify_event_log(
                path,
                expected_stream=stream,
                expected_episode_id=summary.episode_id,
            )
            event_hashes[filename] = verification.file_sha256
    recovery_events = _read_jsonl(root / "recoveries.jsonl")
    actual_strategy = _actual_recovery_strategy(recovery_events)
    strategy_match = actual_strategy is scenario.expected_strategy
    resolved = _score_oracle_requirements(
        rule.requirements,
        summary=summary,
        transcript=adapter_transcript,
        runtime_dir=root,
    )
    result = RecoveryDiagnosticResult(
        scenario_id=scenario.scenario_id,
        system_id=summary.system_id,
        episode_id=summary.episode_id,
        expected_strategy=scenario.expected_strategy,
        actual_strategy=actual_strategy,
        strategy_match=strategy_match,
        resolution_event=scenario.resolution_event,
        rule_passed=resolved,
        recovery_resolved=(
            resolved
            and strategy_match
            and actual_strategy is not RecoveryStrategy.ABORT
        ),
        task_success=False,
        terminal_reason=summary.terminal_reason,
        executor_steps=summary.executor_steps,
        runtime_evidence_sha256=canonical_sha256(event_hashes),
        oracle_rule_sha256=canonical_sha256(
            {"rule_id": rule.rule_id, "requirements": dict(rule.requirements)}
        ),
    )
    destination = Path(sealed_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "recovery_oracle_result.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed result: {output}")
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _campaign_final_recovery_evidence(
    scenario: RecoveryScenarioRegistration,
    rule: RecoveryOracleRule,
    *,
    summary: EpisodeSummary,
    runtime_dir: Path,
    adapter_transcript: Sequence[Mapping[str, JsonValue]],
) -> Mapping[str, Any]:
    """Build the evaluator's final schema from completed diagnostic evidence."""

    recovery_events = _read_jsonl(runtime_dir / "recoveries.jsonl")
    actual_strategy = _actual_recovery_strategy(recovery_events)
    rule_passed = _score_oracle_requirements(
        rule.requirements,
        summary=summary,
        transcript=adapter_transcript,
        runtime_dir=runtime_dir,
    )
    recovery_success = (
        rule_passed
        and actual_strategy is scenario.expected_strategy
        and actual_strategy is not RecoveryStrategy.ABORT
    )
    attempts: list[Mapping[str, Any]] = []
    for event in recovery_events:
        if event.get("event_type") != "recovery_attempt":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        attempt = payload.get("attempt", payload)
        if isinstance(attempt, Mapping):
            attempts.append(attempt)
    incident_id = (
        str(attempts[0].get("incident_id"))
        if attempts
        else f"{summary.episode_id}:sealed-diagnostic-incident:1"
    )
    recovery_verifications = []
    for index, attempt in enumerate(attempts):
        attempt_id = str(attempt.get("attempt_id") or "")
        if not attempt_id:
            raise ValueError("runtime recovery attempt lacks attempt_id")
        recovery_verifications.append(
            {
                "recovery_attempt_id": attempt_id,
                "failure_incident_id": incident_id,
                "verified_failure_present": True,
                "successful": recovery_success and index == len(attempts) - 1,
                "registered_resolution_event": scenario.resolution_event,
                "strategy_match": actual_strategy is scenario.expected_strategy,
            }
        )
    relevance: dict[str, Mapping[str, Any]] = {}
    memory_path = runtime_dir / "memory_queries.jsonl"
    for event in _read_jsonl(memory_path):
        if event.get("event_type") != "post_failure_query":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        query_result = payload.get("query_result", payload)
        if not isinstance(query_result, Mapping):
            continue
        query_id = str(query_result.get("query_id") or "")
        if not query_id:
            raise ValueError("runtime memory query lacks query_id")
        admitted = query_result.get("admitted") is True
        relevance[query_id] = {
            "relevant_ids": [],
            "relevance_definition": (
                "synthetic recovery diagnostic; no train-memory item is oracle-required"
            ),
            "useful_intervention": False if admitted else None,
            "harmful_intervention": True if admitted else None,
        }
    return {
        "task_success": False,
        "terminal_reason": summary.terminal_reason.value,
        "loop_detected": summary.terminal_reason is TerminalReason.LOOP,
        "environment_failure": summary.environment_failure,
        "failure_incidents": [
            {
                "failure_incident_id": incident_id,
                "verified_agent_failure": True,
                "resolved": recovery_success,
                "resolved_attempt_index": len(attempts) if recovery_success else None,
                "attempt_count": len(attempts),
                "failure_kind": scenario.failure_kind,
                "registered_resolution_event": scenario.resolution_event,
            }
        ],
        "recovery_verifications": recovery_verifications,
        "verified_failure_event_count": 1,
        "repeated_error_event_count": (
            1 if scenario.failure_kind == "ACTION_LOOP" else 0
        ),
        "memory_relevance": relevance,
        "diagnostic_rule_passed": rule_passed,
        "diagnostic_expected_strategy": scenario.expected_strategy.value,
        "diagnostic_actual_strategy": (
            actual_strategy.value if actual_strategy is not None else None
        ),
        "diagnostic_evidence_only": True,
    }


def _run_recovery_episode(
    scenario: RecoveryScenarioRegistration,
    *,
    system_id: SystemID,
    adapter: RecoveryDiagnosticAdapter,
    runtime_dir: Path,
    campaign_id: str,
    repeat_id: int,
    matched_seed: int,
    duplicate_audit: FrozenDuplicateAuditManifest,
    memory_candidate: MemoryCandidate | None = None,
    event_logs_override: Any | None = None,
    stage_seeds_override: Mapping[str, int] | None = None,
    expected_episode_id: str | None = None,
    protocol_id: str = "table2-recovery-diagnostic-v1",
    campaign_seed: int = 42,
    require_frozen_stage_seed_plan: bool = False,
) -> EpisodeSummary:
    from web_agent.runtime.action_parameters import DeterministicParameterProvider
    from web_agent.runtime.episode import EpisodeRunner
    from web_agent.runtime.event_log import EpisodeEventLogs
    from web_agent.runtime.executor import Executor
    from web_agent.runtime.memory_adapter import InMemoryFrozenReader, MemoryAdapter
    from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
    from web_agent.runtime.protocol import RuntimeProtocol, switches_for
    from web_agent.runtime.recovery.controller import (
        CallableRecoveryActionPlanner,
        RecoveryController,
    )

    switches = switches_for(system_id)
    policy_id = (
        "diagnostic-base-policy" if system_id is SystemID.E0 else "diagnostic-trained-policy"
    )
    policy = CallablePolicyAdapter(
        policy_id=policy_id,
        policy_version="v1",
        kind=PolicyKind.BASE if system_id is SystemID.E0 else PolicyKind.TRAINED,
        checkpoint_sha256=(
            None
            if system_id is SystemID.E0
            else canonical_sha256("diagnostic-selected-validation-checkpoint-v1")
        ),
        action_predictor=_diagnostic_action_predictor(policy_id),
        transition_predictor=(
            _diagnostic_transition_predictor if switches.post_action_diagnosis else None
        ),
        recovery_predictor=(
            _diagnostic_recovery_predictor if switches.recovery_controller else None
        ),
    )
    protocol = RuntimeProtocol(
        protocol_id=protocol_id,
        campaign_id=campaign_id,
        campaign_seed=campaign_seed,
        require_frozen_stage_seed_plan=require_frozen_stage_seed_plan,
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
        benchmark_id=RecoveryDiagnosticAdapter.benchmark_id,
        benchmark_version=RecoveryDiagnosticAdapter.benchmark_version,
        metadata={"evidence_label": "ENGINEERING_DIAGNOSTIC_ONLY"},
    )
    controller = None
    if switches.recovery_controller:
        controller = RecoveryController(
            protocol.budgets,
            action_planner=CallableRecoveryActionPlanner(
                planner_id="diagnostic-oracle-blind-action-planner",
                planner_version="v1",
                callback=_diagnostic_recovery_action,
            ),
        )
    memory = None
    if system_id is SystemID.E3:
        reader = InMemoryFrozenReader(
            reader_id="diagnostic-immutable-train-reader",
            index_sha256=canonical_sha256(
                {
                    "scenario": scenario.scenario_id,
                    "candidate": memory_candidate.to_dict() if memory_candidate else None,
                }
            ),
            candidates=(memory_candidate,) if memory_candidate is not None else (),
            model_seed=matched_seed,
            registered_admission_threshold=0.5,
        )
        memory = MemoryAdapter(switches, reader=reader, admission_threshold=0.5)
    task = TaskSpecification(
        task_id=scenario.scenario_id,
        goal=f"resolve the observable failure {scenario.failure_kind}",
        benchmark_id="recovery_fixture",
        benchmark_version="v1",
        start_state_id="failure-present",
        metadata={"task_partition": "recovery_diagnostic"},
    )
    episode_id = (
        f"{campaign_id}:{system_id.value}:{task.task_id}:"
        f"repeat-{repeat_id}:seed-{matched_seed}"
    )
    if expected_episode_id is not None and episode_id != expected_episode_id:
        raise ValueError("campaign episode ID differs from deterministic runtime identity")
    generated_stage_seeds = protocol.rng_factory().seed_plan(
        task_id=task.task_id,
        repeat_id=repeat_id,
        matched_seed=matched_seed,
    ).stage_seeds
    stage_seeds = (
        dict(stage_seeds_override)
        if stage_seeds_override is not None
        else generated_stage_seeds
    )
    if event_logs_override is None:
        _write_engineering_runtime_provenance(
            runtime_dir,
            episode_id=episode_id,
            campaign_id=campaign_id,
            protocol_id=protocol.protocol_id,
            system_id=system_id,
            task_id=task.task_id,
            task_partition="recovery_diagnostic",
            repeat_id=repeat_id,
            matched_seed=matched_seed,
            stage_seeds=stage_seeds,
        )
        logs = EpisodeEventLogs(
            runtime_dir,
            episode_id=episode_id,
            include_memory=system_id is SystemID.E3,
            system_id=system_id.value,
            task_id=task.task_id,
            repeat_id=repeat_id,
            matched_seed=matched_seed,
            timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
        )
    else:
        logs = event_logs_override
        if getattr(logs, "episode_id", None) != episode_id:
            raise ValueError("campaign event logs belong to another episode")
    runner = EpisodeRunner(
        protocol=protocol,
        system_policy=SystemPolicy(policy, switches),
        provider=DeterministicParameterProvider(),
        executor=Executor(adapter, budgets=protocol.budgets, clock=lambda: 0.0),
        recovery_controller=controller,
        memory_adapter=memory,
        duplicate_audit_registry=(
            duplicate_audit if system_id is SystemID.E3 else None
        ),
        event_logs=logs,
    )
    return runner.run(
        task,
        repeat_id=repeat_id,
        model_seed=matched_seed,
        stage_seeds=stage_seeds,
    )


def _diagnostic_action_predictor(policy_id: str):
    def predict(
        task: TaskSpecification,
        observation: PolicyObservation,
        rng: object,
    ) -> PreActionDecision:
        del task, rng
        failure_kind = str(observation.current_page_state["failure_kind"])
        action_type, hints, bbox = _initial_action_for_failure(failure_kind)
        return PreActionDecision(
            decision_id=f"{policy_id}:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=action_type,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType), action_type.value
            ),
            bbox=bbox,
            grounding_confidence=0.9,
            confidence_before=0.9,
            input_observation_ids=(observation.observation_id,),
            policy_id=policy_id,
            policy_version="v1",
            parameter_hints=hints,
        )

    return predict


def _diagnostic_transition_predictor(
    task: TaskSpecification,
    transition: TransitionInput,
    rng: object,
) -> TransitionAssessment:
    del task, rng
    failure_kind = str(transition.post_observation.current_page_state["failure_kind"])
    strategy = _policy_strategy_for_failure(failure_kind)
    return TransitionAssessment(
        assessment_id=f"diagnosis:{transition.executed_action.action_id}",
        pre_observation_id=transition.pre_observation.observation_id,
        post_observation_id=transition.post_observation.observation_id,
        executed_action_id=transition.executed_action.action_id,
        predicted_failure=True,
        failure_probability=1.0,
        failure_type=failure_kind,
        failure_type_probabilities={failure_kind: 1.0},
        needs_recovery=True,
        needs_recovery_probability=1.0,
        recovery_strategy=strategy,
        recovery_probabilities={strategy.value: 1.0},
    )


def _diagnostic_recovery_predictor(
    task: TaskSpecification,
    transition: RecoveryTransitionInput,
    rng: object,
) -> RecoveryAssessment:
    del task, rng
    changed = (
        transition.pre_recovery_observation.current_page_state.get("state_id")
        != transition.post_recovery_observation.current_page_state.get("state_id")
    )
    return RecoveryAssessment(
        assessment_id=f"recovery-assessment:{transition.attempt_id}",
        incident_id=transition.incident_id,
        attempt_id=transition.attempt_id,
        pre_recovery_observation_id=(
            transition.pre_recovery_observation.observation_id
        ),
        post_recovery_observation_id=(
            transition.post_recovery_observation.observation_id
        ),
        recovery_action_ids=tuple(
            action.action_id for action in transition.recovery_actions
        ),
        predicted_failure_resolved=changed,
        predicted_resolution_probability=1.0 if changed else 0.0,
        predicted_progress=changed,
        predicted_progress_probability=1.0 if changed else 0.0,
    )


def _diagnostic_recovery_action(
    task: TaskSpecification,
    post_failure_observation: PolicyObservation,
    decision: Any,
    failed_action: ConcreteAction,
    rng: object,
) -> ConcreteAction:
    del task, failed_action, rng
    failure_kind = str(post_failure_observation.current_page_state["failure_kind"])
    if failure_kind not in decision.diagnosis:
        raise ValueError("recovery plan diagnosis/current observation mismatch")
    return _diagnostic_planned_action(
        failure_kind,
        decision.strategy,
        decision_id=decision.decision_id,
    )


def _diagnostic_planned_action(
    failure_kind: str,
    strategy: RecoveryStrategy,
    *,
    decision_id: str,
) -> ConcreteAction:
    if strategy is RecoveryStrategy.REPLAN and failure_kind == "TARGET_MISSING":
        return ConcreteAction(
            action_id="diagnostic-planned-type",
            source_decision_id=decision_id,
            action_type=ActionType.TYPE,
            parameters={
                "target_x": 0.8,
                "target_y": 0.75,
                "target_bbox": list(_ALTERNATIVE_BBOX),
                "text": "fixture",
            },
            bbox=_ALTERNATIVE_BBOX,
        )
    if strategy is RecoveryStrategy.REPLAN and failure_kind == "STALE_PAGE":
        return ConcreteAction(
            action_id="diagnostic-planned-refresh",
            source_decision_id=decision_id,
            action_type=ActionType.NAVIGATE,
            parameters={"url": "fixture://recovery/fresh"},
        )
    return ConcreteAction(
        action_id="diagnostic-planned-alternative",
        source_decision_id=decision_id,
        action_type=ActionType.CLICK,
        parameters={
            "target_x": 0.8,
            "target_y": 0.75,
            "target_bbox": list(_ALTERNATIVE_BBOX),
            "button": "left",
            "click_count": 1,
        },
        bbox=_ALTERNATIVE_BBOX,
    )


def _diagnostic_visible_compatible_actions(
    failure_kind: str,
    failed_action: ConcreteAction | None,
) -> tuple[ConcreteAction, ...]:
    strategy = _policy_strategy_for_failure(failure_kind)
    if strategy is RecoveryStrategy.RETRY:
        return (failed_action,) if failed_action is not None else ()
    if strategy in {
        RecoveryStrategy.REPLAN,
        RecoveryStrategy.ALTERNATIVE_TARGET,
    }:
        planned = _diagnostic_planned_action(
            failure_kind,
            strategy,
            decision_id="diagnostic-target-evidence",
        )
        if failure_kind == "NO_STATE_CHANGE" and failed_action is not None:
            return (failed_action, planned)
        return (planned,)
    return ()


def _initial_action_for_failure(
    failure_kind: str,
) -> tuple[
    ActionType,
    Mapping[str, JsonValue],
    tuple[float, float, float, float] | None,
]:
    if failure_kind == "TARGET_MISSING":
        return ActionType.TYPE, {}, None
    if failure_kind == "INVALID_PARAMETER":
        return ActionType.SELECT, {
            "option": "stale-option",
            "candidate_options": ["stale-option"],
        }, _BBOX
    if failure_kind in {"NAVIGATION_LOOP", "STALE_PAGE", "OFF_PATH"}:
        return ActionType.NAVIGATE, {"url": "fixture://recovery/stale"}, None
    return ActionType.CLICK, {}, _BBOX


def _policy_strategy_for_failure(failure_kind: str) -> RecoveryStrategy:
    mapping = {
        "REJECTED_ACTION": RecoveryStrategy.RETRY,
        "STALE_TARGET": RecoveryStrategy.REPLAN,
        "NO_STATE_CHANGE": RecoveryStrategy.ALTERNATIVE_TARGET,
        "TARGET_MISSING": RecoveryStrategy.REPLAN,
        "INVALID_PARAMETER": RecoveryStrategy.RETRY,
        "NAVIGATION_LOOP": RecoveryStrategy.BACKTRACK,
        "OBSTRUCTED_TARGET": RecoveryStrategy.ALTERNATIVE_TARGET,
        "CONTEXT_MISMATCH": RecoveryStrategy.BACKTRACK,
        "TRANSIENT_EXECUTOR_ERROR": RecoveryStrategy.RETRY,
        "STALE_PAGE": RecoveryStrategy.REPLAN,
        "OFF_PATH": RecoveryStrategy.BACKTRACK,
        "AMBIGUOUS_TARGET": RecoveryStrategy.ALTERNATIVE_TARGET,
        "ACTION_LOOP": RecoveryStrategy.ABORT,
        "RECOVERY_BUDGET_EXHAUSTED": RecoveryStrategy.ABORT,
        "ENVIRONMENT_FAILURE": RecoveryStrategy.ABORT,
    }
    try:
        return mapping[failure_kind]
    except KeyError as exc:
        raise ValueError(f"unregistered diagnostic failure: {failure_kind}") from exc


def _registered_initial_outcome(
    failure_kind: str,
) -> tuple[ExecutionStatus, str | None]:
    if failure_kind == "REJECTED_ACTION":
        return ExecutionStatus.REJECTED, "registered_rejection"
    if failure_kind == "TRANSIENT_EXECUTOR_ERROR":
        return ExecutionStatus.ERROR, "transient_executor_error"
    return ExecutionStatus.EXECUTED, None


def _valid_replanned_action(failure_kind: str, action: ConcreteAction) -> bool:
    if failure_kind == "TARGET_MISSING":
        return action.action_type is ActionType.TYPE and bool(action.parameters.get("text"))
    if failure_kind == "STALE_PAGE":
        return (
            action.action_type is ActionType.NAVIGATE
            and action.parameters.get("url") == "fixture://recovery/fresh"
        )
    return action.action_type is ActionType.CLICK and action.bbox == _ALTERNATIVE_BBOX


def _observable_resolution_flags(
    event: str,
    *,
    resolved: bool,
) -> Mapping[str, JsonValue]:
    flags: dict[str, JsonValue] = {
        "text_entered": False,
        "option_selected": False,
        "modal_absent": False,
        "expected_context": False,
        "fresh_state": False,
        "prior_state": False,
    }
    if resolved:
        event_to_flag = {
            "text_entered": "text_entered",
            "option_selected": "option_selected",
            "modal_closed": "modal_absent",
            "expected_context_restored": "expected_context",
            "fresh_state_observed": "fresh_state",
            "prior_state_restored": "prior_state",
        }
        flag = event_to_flag.get(event)
        if flag is not None:
            flags[flag] = True
    return flags


def _score_oracle_requirements(
    requirements: Mapping[str, JsonValue],
    *,
    summary: EpisodeSummary,
    transcript: Sequence[Mapping[str, JsonValue]],
    runtime_dir: Path,
) -> bool:
    recovery_rows = [row for row in transcript if row.get("phase") == "recovery"]
    last = recovery_rows[-1] if recovery_rows else None
    flags: Mapping[str, Any] = {}
    environment_rows = _read_jsonl(runtime_dir / "environment_events.jsonl")
    for row in reversed(environment_rows):
        if row.get("event_type") == "post_recovery_observation":
            page_state = row.get("payload", {}).get("page_state", {})
            if isinstance(page_state, Mapping):
                raw_flags = page_state.get("diagnostic_flags", {})
                if isinstance(raw_flags, Mapping):
                    flags = raw_flags
            break
    for key, expected in requirements.items():
        if key == "requires_executor_status":
            actual = str(last.get("status", "")).upper() if last else ""
            if actual != str(expected).upper():
                return False
        elif key == "requires_state_change":
            if bool(last and last.get("state_changed")) is not bool(expected):
                return False
        elif key == "requires_target_change":
            if bool(last and last.get("target_changed")) is not bool(expected):
                return False
        elif key == "requires_observation_flag":
            if flags.get(str(expected)) is not True:
                return False
        elif key == "forbids_loop_signature":
            if bool(expected) and not bool(last and last.get("state_changed")):
                return False
        elif key == "requires_terminal_reason":
            actual_terminal = (
                "ABORTED"
                if summary.terminal_reason is TerminalReason.ABORT
                else summary.terminal_reason.value.upper()
            )
            if actual_terminal != str(expected).upper():
                return False
        elif key == "counts_as_success":
            if bool(expected) is not False:
                return False
        else:
            raise ValueError(f"unregistered recovery oracle requirement: {key}")
    return True


def _actual_recovery_strategy(
    recovery_events: Sequence[Mapping[str, Any]],
) -> RecoveryStrategy | None:
    for row in recovery_events:
        if row.get("event_type") != "recovery_plan":
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        plan = payload.get("plan")
        if isinstance(plan, Mapping) and isinstance(plan.get("strategy"), str):
            return RecoveryStrategy(plan["strategy"])
    return None


def _read_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {source}:{line_number}")
        rows.append(value)
    return rows


def _require_new_output_root(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"diagnostic output root must be new or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _write_engineering_runtime_provenance(
    runtime_dir: Path,
    *,
    episode_id: str,
    campaign_id: str,
    protocol_id: str,
    system_id: SystemID,
    task_id: str,
    task_partition: str,
    repeat_id: int,
    matched_seed: int,
    stage_seeds: Mapping[str, int],
) -> None:
    """Create canonical smoke provenance unless CampaignRunner already did."""

    rng_path = runtime_dir / "rng_provenance.json"
    manifest_path = runtime_dir / "episode_manifest.json"
    if rng_path.exists() or manifest_path.exists():
        if not (rng_path.is_file() and manifest_path.is_file()):
            raise FileExistsError("partial runtime provenance package exists")
        return
    rng = {
        "schema_version": "1.0",
        "algorithm": "sha256_stage_keyed_v1",
        "protocol_id": protocol_id,
        "campaign_id": campaign_id,
        "campaign_seed": 42,
        "task_id": task_id,
        "repeat_id": repeat_id,
        "matched_seed": matched_seed,
        "anchor_decision_index": 0,
        "stage_seed_anchors": dict(stage_seeds),
        "dynamic_key_fields": [
            "campaign_id",
            "task_id",
            "repeat_id",
            "matched_seed",
            "stage",
            "decision_index",
            "incident_index",
            "attempt_index",
            "stream",
        ],
        "namespace_fields": ["protocol_id", "campaign_seed"],
        "system_id_in_seed_key": False,
        "rerun_id_in_seed_key": False,
    }
    rng_path.write_text(json.dumps(rng, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    episode_manifest = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "campaign_id": campaign_id,
        "protocol_id": protocol_id,
        "system_id": system_id.value,
        "task_id": task_id,
        "task_partition": task_partition,
        "matched_model_seed": matched_seed,
        "repeat_id": repeat_id,
        "attempt_id": 0,
        "stage_seeds": dict(stage_seeds),
        "rng_provenance_sha256": _sha256_file(rng_path),
        "evidence_label": "ENGINEERING_DIAGNOSTIC_ONLY",
        "paper_table_status": "N/R",
    }
    manifest_path.write_text(
        json.dumps(episode_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_engineering_artifact_hashes(runtime_dir: Path) -> None:
    path = runtime_dir / "artifact_hashes.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime artifact hashes: {path}")
    files = {
        item.name: _sha256_file(item)
        for item in sorted(runtime_dir.iterdir())
        if item.is_file() and item.name != path.name
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "hash_algorithm": "sha256",
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
