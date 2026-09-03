"""Deterministic in-process browser fixture for causal runtime smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from web_agent.benchmarks.base import (
    AdapterExecution,
    BenchmarkAdapter,
    BenchmarkStateError,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    EpisodeSummary,
    ExecutionStatus,
    JsonValue,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryStrategy,
    RecoveryTransitionInput,
    SystemID,
    TaskSpecification,
    TransitionAssessment,
    TransitionInput,
    VerifierReceiptBinding,
    VersionedRecord,
    canonical_sha256,
    probability_map,
)


@dataclass(frozen=True, slots=True)
class FixtureState(VersionedRecord):
    state_id: str
    url: str
    title: str
    observable: Mapping[str, JsonValue] = field(default_factory=dict)
    progress_rank: int = 0
    success: bool = False
    environment_failure: bool = False

    def __post_init__(self) -> None:
        if not self.state_id or self.progress_rank < 0:
            raise ValueError("fixture state requires an ID and non-negative progress")


@dataclass(frozen=True, slots=True)
class FixtureTransition(VersionedRecord):
    transition_id: str
    source_state_id: str
    action_type: ActionType
    destination_state_id: str
    required_parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    required_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class FixtureScenario(VersionedRecord):
    scenario_id: str
    task_id: str
    initial_state_id: str
    states: tuple[FixtureState, ...]
    transitions: tuple[FixtureTransition, ...]

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.task_id or not self.initial_state_id:
            raise ValueError("fixture scenario identifiers must be non-empty")
        state_ids = [state.state_id for state in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("fixture state IDs must be unique")
        if self.initial_state_id not in state_ids:
            raise ValueError("fixture initial state does not exist")
        for transition in self.transitions:
            if transition.source_state_id not in state_ids:
                raise ValueError(f"unknown source state: {transition.source_state_id}")
            if transition.destination_state_id not in state_ids:
                raise ValueError(f"unknown destination state: {transition.destination_state_id}")


class DeterministicFixtureAdapter(BenchmarkAdapter):
    benchmark_id = "table2-fixture"
    benchmark_version = "v1"

    def __init__(
        self,
        scenarios: Mapping[str, FixtureScenario] | None = None,
        *,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        default = default_fixture_scenario()
        self._scenarios = dict(scenarios or {default.task_id: default})
        self._width = width
        self._height = height
        self._scenario: FixtureScenario | None = None
        self._states: dict[str, FixtureState] = {}
        self._state_id: str | None = None
        self._episode_id: str | None = None
        self._observation_index = 0
        self._verification_index = 0
        self._closed = False

    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        del seed  # Scenario transitions are deliberately deterministic.
        if self._closed:
            raise BenchmarkStateError("fixture adapter is closed")
        try:
            scenario = self._scenarios[task.task_id]
        except KeyError as exc:
            raise BenchmarkStateError(f"unknown fixture task: {task.task_id}") from exc
        if task.benchmark_id not in {self.benchmark_id, "fixture"}:
            raise BenchmarkStateError(
                f"task benchmark {task.benchmark_id!r} does not match fixture"
            )
        self._scenario = scenario
        self._states = {state.state_id: state for state in scenario.states}
        self._state_id = scenario.initial_state_id
        self._episode_id = episode_id
        self._observation_index = 0
        self._verification_index = 0
        return self.observe(stage=ObservationStage.RESET)

    @property
    def _state(self) -> FixtureState:
        if self._state_id is None or self._scenario is None:
            raise BenchmarkStateError("fixture task has not been reset")
        return self._states[self._state_id]

    def observe(
        self,
        *,
        stage: ObservationStage,
        prior_action_id: str | None = None,
    ) -> Observation:
        state = self._state
        if stage in {ObservationStage.POST_ACTION, ObservationStage.POST_RECOVERY}:
            if not prior_action_id:
                raise BenchmarkStateError("post-action fixture observation needs action ID")
        self._observation_index += 1
        visible_identity = {
            "scenario_id": self._scenario.scenario_id if self._scenario else "",
            "state_id": state.state_id,
            "observable": state.observable,
        }
        return Observation(
            observation_id=f"{self._episode_id}:obs:{self._observation_index}",
            episode_id=self._episode_id or "uninitialised",
            stage=stage,
            screenshot_sha256=canonical_sha256(visible_identity),
            width=self._width,
            height=self._height,
            url=state.url,
            title=state.title,
            page_state=dict(state.observable),
            page_settled=True,
            environment_error=state.environment_failure,
            prior_action_id=prior_action_id,
        )

    def execute(self, action: ConcreteAction) -> AdapterExecution:
        source = self._state
        if source.environment_failure:
            return AdapterExecution(
                status=ExecutionStatus.ERROR,
                state_changed=False,
                environment_error=True,
                error_kind="fixture_environment_failure",
                message="fixture entered a registered environment-failure state",
            )
        transition = next(
            (
                candidate
                for candidate in self._scenario.transitions
                if candidate.source_state_id == source.state_id
                and _matches(candidate, action)
            ),
            None,
        )
        if transition is None:
            # A wrong but structurally valid browser action executes as a no-op;
            # no hidden correct target/action is revealed to the agent.
            return AdapterExecution(
                status=ExecutionStatus.EXECUTED,
                state_changed=False,
                message="fixture action produced no observable state change",
            )
        self._state_id = transition.destination_state_id
        destination = self._state
        return AdapterExecution(
            status=(
                ExecutionStatus.ERROR
                if destination.environment_failure
                else ExecutionStatus.EXECUTED
            ),
            state_changed=destination.state_id != source.state_id,
            environment_error=destination.environment_failure,
            error_kind=(
                "fixture_environment_failure"
                if destination.environment_failure
                else None
            ),
            message=f"fixture transition {transition.transition_id}",
        )

    def terminal_signal(
        self,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None = None,
    ) -> OpaqueTerminalSignal:
        if self._scenario is None or task.task_id != self._scenario.task_id:
            raise BenchmarkStateError("terminal-signal task does not match active fixture")
        state = self._state
        self._verification_index += 1
        event_id = f"{self._episode_id}:terminal:{self._verification_index}"
        # The full fixture truth remains inside the sealed benchmark.  Runtime
        # receives only an uninformative digest and a stop/no-stop bit.
        return OpaqueTerminalSignal(
            event_id=event_id,
            token_sha256=canonical_sha256(
                {
                    "event_id": event_id,
                    "benchmark_id": self.benchmark_id,
                    "opaque_receipt_domain": "fixture-terminal-v1",
                    "causal_binding_sha256": (
                        binding.record_sha256 if binding is not None else None
                    ),
                }
            ),
            terminate=state.success,
        )

    def close(self) -> None:
        self._closed = True
        self._scenario = None
        self._states = {}
        self._state_id = None


FixtureAdapter = DeterministicFixtureAdapter


def _matches(transition: FixtureTransition, action: ConcreteAction) -> bool:
    if action.action_type is not transition.action_type:
        return False
    for key, expected in transition.required_parameters.items():
        if action.parameters.get(key) != expected:
            return False
    if transition.required_bbox is not None:
        if action.bbox is None:
            return False
        if any(
            abs(float(actual) - float(expected)) > 1e-9
            for actual, expected in zip(action.bbox, transition.required_bbox)
        ):
            return False
    return True


def default_fixture_scenario() -> FixtureScenario:
    """A six-action chain covering every registered P3 action class."""
    bbox = (0.10, 0.10, 0.20, 0.10)
    states = tuple(
        FixtureState(
            state_id=f"s{index}",
            url=f"fixture://table2/state/{index}",
            title=f"Fixture state {index}",
            observable={
                "state_id": f"s{index}",
                "instruction": (
                    "complete" if index == 6 else f"perform step {index + 1}"
                ),
                **(
                    {
                        "observable_select_controls": [
                            {
                                "target_bbox": list(bbox),
                                "candidate_options": ["fixture-option"],
                            }
                        ]
                    }
                    if index == 2
                    else {}
                ),
            },
            progress_rank=index,
            success=index == 6,
        )
        for index in range(7)
    )
    transitions = (
        FixtureTransition("t-click", "s0", ActionType.CLICK, "s1", required_bbox=bbox),
        FixtureTransition(
            "t-type", "s1", ActionType.TYPE, "s2",
            required_parameters={"text": "fixture"}, required_bbox=bbox,
        ),
        FixtureTransition(
            "t-select", "s2", ActionType.SELECT, "s3",
            required_parameters={"option": "fixture-option"}, required_bbox=bbox,
        ),
        FixtureTransition(
            "t-scroll", "s3", ActionType.SCROLL, "s4",
            required_parameters={"direction": "down"},
        ),
        FixtureTransition(
            "t-navigate", "s4", ActionType.NAVIGATE, "s5",
            required_parameters={"url": "fixture://table2/final"},
        ),
        FixtureTransition(
            "t-key", "s5", ActionType.PRESS_KEY, "s6",
            required_parameters={"key": "ENTER"},
        ),
    )
    return FixtureScenario(
        scenario_id="registered-six-action-fixture-v1",
        task_id="fixture-six-actions",
        initial_state_id="s0",
        states=states,
        transitions=transitions,
    )


_SMOKE_ACTIONS: Mapping[
    str,
    tuple[ActionType, Mapping[str, JsonValue], tuple[float, float, float, float] | None],
] = {
    "s0": (ActionType.CLICK, {}, (0.10, 0.10, 0.20, 0.10)),
    "s1": (
        ActionType.TYPE,
        {"text": "fixture"},
        (0.10, 0.10, 0.20, 0.10),
    ),
    "s2": (
        ActionType.SELECT,
        {
            "option": "fixture-option",
            "candidate_options": ["fixture-option"],
        },
        (0.10, 0.10, 0.20, 0.10),
    ),
    "s3": (ActionType.SCROLL, {"direction": "down"}, None),
    "s4": (
        ActionType.NAVIGATE,
        {"url": "fixture://table2/final"},
        None,
    ),
    "s5": (ActionType.PRESS_KEY, {"key": "ENTER"}, None),
}


def run_all_systems_fixture_smoke(
    output_root: str | Path,
    *,
    campaign_id: str = "table2-fixture-smoke",
    repeat_id: int = 0,
    matched_seed: int = 42,
) -> dict[SystemID, EpisodeSummary]:
    """Run deterministic E0--E3 episodes and write only below ``output_root``.

    This is an engineering smoke, not paper evidence. It uses the in-process
    fixture, deterministic policy/provider callbacks, a fixed clock/timestamp,
    and an empty immutable E3 reader. No WebArena/browser/model installation is
    imported or required. Existing non-empty output roots are never reused.
    """
    destination = Path(output_root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"fixture smoke output root must be new or empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    # Imports stay local to avoid coupling benchmark registration to runtime
    # orchestration and to keep ordinary fixture adapter imports lightweight.
    from web_agent.runtime.action_parameters import DeterministicParameterProvider
    from web_agent.runtime.episode import EpisodeRunner
    from web_agent.runtime.event_log import EpisodeEventLogs
    from web_agent.runtime.executor import Executor
    from web_agent.runtime.memory_adapter import InMemoryFrozenReader, MemoryAdapter
    from web_agent.runtime.policy import (
        CallablePolicyAdapter,
        PolicyKind,
        SystemPolicy,
    )
    from web_agent.runtime.protocol import RuntimeProtocol, switches_for
    from web_agent.runtime.recovery.controller import RecoveryController

    protocol = RuntimeProtocol(
        protocol_id="table2-fixture-smoke-v1",
        campaign_id=campaign_id,
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
        benchmark_id=DeterministicFixtureAdapter.benchmark_id,
        benchmark_version=DeterministicFixtureAdapter.benchmark_version,
        metadata={"evidence_label": "ENGINEERING_SMOKE_ONLY"},
    )
    task = TaskSpecification(
        task_id=default_fixture_scenario().task_id,
        goal="complete the deterministic six-action fixture",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
        metadata={"task_partition": "recovery_diagnostic"},
    )
    summaries: dict[SystemID, EpisodeSummary] = {}
    from web_agent.runtime.duplicate_audit import FrozenDuplicateAuditManifest

    duplicate_audit = FrozenDuplicateAuditManifest.synthetic_diagnostic(
        task_id=task.task_id,
        cluster_ids=("fixture-evaluation-cluster",),
        content_identity={"scenario": default_fixture_scenario().to_dict()},
    )
    for system_id in SystemID:
        switches = switches_for(system_id)
        policy_id = (
            "fixture-base-policy"
            if system_id is SystemID.E0
            else "fixture-trained-policy"
        )
        kind = PolicyKind.BASE if system_id is SystemID.E0 else PolicyKind.TRAINED
        adapter = CallablePolicyAdapter(
            policy_id=policy_id,
            policy_version="v1",
            kind=kind,
            checkpoint_sha256=(
                None
                if kind is PolicyKind.BASE
                else canonical_sha256("fixture-trained-checkpoint-v1")
            ),
            action_predictor=_smoke_action_predictor(policy_id),
            transition_predictor=(
                _smoke_transition_predictor
                if switches.post_action_diagnosis
                else None
            ),
            recovery_predictor=(
                _smoke_recovery_predictor
                if switches.recovery_controller
                else None
            ),
        )
        system_policy = SystemPolicy(adapter=adapter, switches=switches)
        recovery_controller = (
            RecoveryController(protocol.budgets)
            if switches.recovery_controller
            else None
        )
        memory_adapter = None
        if system_id is SystemID.E3:
            reader = InMemoryFrozenReader(
                reader_id="fixture-empty-frozen-reader",
                index_sha256=canonical_sha256("fixture-empty-memory-index-v1"),
                candidates=(),
                model_seed=matched_seed,
                registered_admission_threshold=0.5,
            )
            memory_adapter = MemoryAdapter(
                switches,
                reader=reader,
                admission_threshold=0.5,
            )
        episode_id = (
            f"{campaign_id}:{system_id.value}:{task.task_id}:"
            f"repeat-{repeat_id}:seed-{matched_seed}"
        )
        logs = EpisodeEventLogs(
            destination / system_id.value / "runtime",
            episode_id=episode_id,
            include_memory=system_id is SystemID.E3,
            system_id=system_id.value,
            task_id=task.task_id,
            repeat_id=repeat_id,
            matched_seed=matched_seed,
            timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
        )
        runner = EpisodeRunner(
            protocol=protocol,
            system_policy=system_policy,
            provider=DeterministicParameterProvider(),
            executor=Executor(
                DeterministicFixtureAdapter(),
                budgets=protocol.budgets,
                clock=lambda: 0.0,
            ),
            recovery_controller=recovery_controller,
            memory_adapter=memory_adapter,
            duplicate_audit_registry=(
                duplicate_audit if system_id is SystemID.E3 else None
            ),
            event_logs=logs,
        )
        summaries[system_id] = runner.run(
            task,
            repeat_id=repeat_id,
            model_seed=matched_seed,
            stage_seeds=protocol.rng_factory().seed_plan(
                task_id=task.task_id,
                repeat_id=repeat_id,
                matched_seed=matched_seed,
            ).stage_seeds,
        )
    return summaries


def _smoke_action_predictor(policy_id: str):
    def predict(
        task: TaskSpecification,
        observation: PolicyObservation,
        rng: object,
    ) -> PreActionDecision:
        del task, rng
        state_id = str(observation.current_page_state["state_id"])
        action_type, hints, bbox = _SMOKE_ACTIONS[state_id]
        return PreActionDecision(
            decision_id=f"{policy_id}:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=action_type,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType),
                action_type.value,
            ),
            bbox=bbox,
            grounding_confidence=1.0,
            confidence_before=1.0,
            input_observation_ids=(observation.observation_id,),
            policy_id=policy_id,
            policy_version="v1",
            parameter_hints=dict(hints),
        )

    return predict


def _smoke_transition_predictor(
    task: TaskSpecification,
    transition: TransitionInput,
    rng: object,
) -> TransitionAssessment:
    del task, rng
    changed = (
        transition.pre_observation.current_page_state.get("state_id")
        != transition.post_observation.current_page_state.get("state_id")
    )
    return TransitionAssessment(
        assessment_id=f"assessment:{transition.executed_action.action_id}",
        pre_observation_id=transition.pre_observation.observation_id,
        post_observation_id=transition.post_observation.observation_id,
        executed_action_id=transition.executed_action.action_id,
        predicted_failure=not changed,
        failure_probability=0.0 if changed else 1.0,
        failure_type="NONE" if changed else "NO_EFFECT",
        failure_type_probabilities={"NONE" if changed else "NO_EFFECT": 1.0},
        needs_recovery=not changed,
        needs_recovery_probability=0.0 if changed else 1.0,
        recovery_strategy=(
            RecoveryStrategy.NONE if changed else RecoveryStrategy.RETRY
        ),
        recovery_probabilities={
            ("NONE" if changed else "RETRY"): 1.0,
        },
    )


def _smoke_recovery_predictor(
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
