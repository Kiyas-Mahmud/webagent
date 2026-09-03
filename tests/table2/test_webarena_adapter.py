from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from web_agent.benchmarks.base import (
    BenchmarkAdapter,
    BenchmarkStateError,
    BenchmarkUnavailableError,
    EnvironmentAdapter,
)
from web_agent.benchmarks.webarena import (
    FrozenWebArenaActionSafetyPolicy,
    FrozenWebArenaEnvironmentStateDigester,
    FrozenWebArenaInfrastructureFaultClassifier,
    FrozenWebArenaManualRescueGuard,
    FrozenWebArenaObservationSnapshot,
    FrozenWebArenaPageSettlePolicy,
    WebArenaAdapter,
    WebArenaActionSafetyDecision,
    WebArenaInfrastructureClassification,
    WebArenaInfrastructureRule,
    WebArenaManualRescueCheck,
    WebArenaManualRescueReceipt,
)
from web_agent.eval.table2.execution_guard import InfrastructureInvalidError
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    MemoryQuery,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    RecoveryStrategy,
    SystemID,
    TaskSpecification,
    TransitionAssessment,
    VerifierReceiptBinding,
    canonical_json,
    canonical_sha256,
)
from web_agent.runtime.decision import DecisionCombiner
from web_agent.runtime.observation import (
    CausalBoundaryError,
    ObservationBuilder,
)
from web_agent.runtime.executor import Executor
from web_agent.runtime.protocol import (
    REGISTERED_BUDGETS,
    REGISTERED_RECOVERY_TRIGGER_CONFIG,
    switches_for,
)
from web_agent.runtime.state_reset import (
    FrozenWebArenaResetStateAttester,
    REGISTERED_WEBARENA_RESET_COMMITMENTS,
    WebArenaResetStateRequest,
    webarena_reset_receipt_from_commitments,
)


SHA = "a" * 64


def test_environment_adapter_is_the_public_name_for_the_benchmark_contract():
    assert EnvironmentAdapter is BenchmarkAdapter


class _Environment:
    def __init__(self) -> None:
        self.state = {"page": 0}
        self.closed = False
        self.step_calls = 0

    def reset(self, *, seed: int):
        self.state = {"page": seed % 2}
        return dict(self.state), {"ignored": True}

    def step(self, native_action):
        self.step_calls += 1
        self.state = {"page": int(native_action["next_page"])}
        return dict(self.state), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class _ResetFailureEnvironment(_Environment):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def reset(self, *, seed: int):
        del seed
        raise self.error


class _StepFailureEnvironment(_Environment):
    def step(self, native_action):
        del native_action
        raise ConnectionError("browser controller disconnected")


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="webarena.0",
        goal="visit the next page",
        benchmark_id="webarena",
        benchmark_version="fixture-version",
        start_state_id="registered-reset",
    )


def _observation(raw, episode_id, stage, prior_action_id):
    return Observation(
        observation_id=f"{episode_id}:{stage.value}:{raw['page']}",
        episode_id=episode_id,
        stage=stage,
        screenshot_sha256=hashlib.sha256(_screenshot_bytes(raw, None)).hexdigest(),
        page_state={"page": raw["page"]},
        prior_action_id=prior_action_id,
    )


def _screenshot_bytes(raw, observation):
    del observation
    return f"fixture-screenshot-page-{raw['page']}".encode("utf-8")


def _terminal(snapshot, task, binding):
    del task, binding
    assert type(snapshot) is FrozenWebArenaObservationSnapshot
    return OpaqueTerminalSignal(
        event_id="sealed-event",
        token_sha256="f" * 64,
        terminate=snapshot.observation.page_state["page"] == 1,
    )


def _state_digester() -> FrozenWebArenaEnvironmentStateDigester:
    return FrozenWebArenaEnvironmentStateDigester(
        digester_id="fixture-webarena-state",
        digester_version="v1",
        benchmark_version="fixture-version",
        callback=lambda environment: canonical_sha256(environment.state),
    )


def _reset_state_attester() -> FrozenWebArenaResetStateAttester:
    def attest(request: WebArenaResetStateRequest):
        return webarena_reset_receipt_from_commitments(
            request,
            attester_id="fixture-reset-state-attester",
            attester_version="v1",
            attester_source_sha256=SHA,
            commitments_sha256={
                role: canonical_sha256({"role": role, "state": "clean"})
                for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
            },
        )

    return FrozenWebArenaResetStateAttester(
        attester_id="fixture-reset-state-attester",
        attester_version="v1",
        source_sha256=SHA,
        callback=attest,
    )


def _page_settle_policy(callback=None) -> FrozenWebArenaPageSettlePolicy:
    return FrozenWebArenaPageSettlePolicy(
        policy_id="fixture-settle-v1",
        benchmark_version="fixture-version",
        network_idle_required=True,
        settle_timeout_seconds=5.0,
        callback=(
            callback
            if callback is not None
            else lambda environment, raw, stage, timeout, network_idle: raw
        ),
    )


def _action_safety_policy(callback) -> FrozenWebArenaActionSafetyPolicy:
    return FrozenWebArenaActionSafetyPolicy(
        safety_policy_id="fixture-safety",
        safety_policy_version="v1",
        destructive_action_policy="fixture-no-destructive-actions",
        benchmark_version="fixture-version",
        source_sha256=SHA,
        callback=callback,
    )


def _manual_rescue_guard(callback=None) -> FrozenWebArenaManualRescueGuard:
    def clean(environment, check: WebArenaManualRescueCheck):
        del environment
        return WebArenaManualRescueReceipt(
            guard_id="fixture-manual-rescue-guard",
            guard_version="v1",
            source_sha256=SHA,
            evidence_mode="exclusive_controller_input_audit_v1",
            check_sha256=check.record_sha256,
            check_index=check.check_index,
            stage=check.stage,
            expected_registered_browser_steps=(
                check.expected_registered_browser_steps
            ),
            observed_registered_browser_steps=(
                check.expected_registered_browser_steps
            ),
            automation_session_sha256="b" * 64,
            controller_input_audit_sha256=canonical_sha256(check.to_dict()),
            exclusive_automation_control=True,
            non_agent_input_event_count=0,
            manual_rescue_detected=False,
        )

    return FrozenWebArenaManualRescueGuard(
        guard_id="fixture-manual-rescue-guard",
        guard_version="v1",
        benchmark_version="fixture-version",
        evidence_mode="exclusive_controller_input_audit_v1",
        source_sha256=SHA,
        callback=callback or clean,
    )


def test_evaluation_webarena_requires_registered_manual_rescue_guard(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        manual_rescue_evidence_sink=lambda check, receipt: None,
        require_manual_rescue_guard=True,
    )
    with pytest.raises(BenchmarkUnavailableError, match="manual_rescue_guard"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_manual_rescue_guard_emits_bound_clean_lifecycle_receipts(
    tmp_path: Path,
) -> None:
    evidence: list[tuple[WebArenaManualRescueCheck, WebArenaManualRescueReceipt]] = []
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        manual_rescue_guard=_manual_rescue_guard(),
        manual_rescue_evidence_sink=lambda check, receipt: evidence.append(
            (check, receipt)
        ),
        require_manual_rescue_guard=True,
    )
    task = _task()
    reset = adapter.reset(task, episode_id="episode", seed=42)
    adapter.terminal_signal(
        task,
        VerifierReceiptBinding(
            receipt_kind="after_reset",
            observation_id=reset.observation_id,
            observation_sha256=reset.record_sha256,
        ),
    )
    action = ConcreteAction(
        action_id="guarded-action",
        source_decision_id="decision",
        action_type=ActionType.NAVIGATE,
        parameters={"url": "https://example.invalid/"},
    )
    adapter.execute(action)
    post = adapter.observe(
        stage=ObservationStage.POST_ACTION,
        prior_action_id=action.action_id,
    )
    adapter.terminal_signal(
        task,
        VerifierReceiptBinding(
            receipt_kind="after_normal_action",
            observation_id=post.observation_id,
            observation_sha256=post.record_sha256,
            action_id=action.action_id,
            action_sha256=action.record_sha256,
        ),
    )

    assert [check.stage for check, _ in evidence] == [
        "after_reset",
        "before_terminal",
        "before_step",
        "after_step",
        "before_terminal",
    ]
    assert [check.check_index for check, _ in evidence] == [1, 2, 3, 4, 5]
    assert evidence[0][0].previous_receipt_sha256 == "0" * 64
    for index in range(1, len(evidence)):
        assert evidence[index][0].previous_receipt_sha256 == (
            evidence[index - 1][1].record_sha256
        )
    assert all(not receipt.manual_rescue_detected for _, receipt in evidence)


def test_manual_rescue_detection_fails_before_browser_step_and_is_evidenced(
    tmp_path: Path,
) -> None:
    environment = _Environment()
    evidence = []

    def detect(live_environment, check: WebArenaManualRescueCheck):
        del live_environment
        count = 1 if check.stage == "before_step" else 0
        return WebArenaManualRescueReceipt(
            guard_id="fixture-manual-rescue-guard",
            guard_version="v1",
            source_sha256=SHA,
            evidence_mode="exclusive_controller_input_audit_v1",
            check_sha256=check.record_sha256,
            check_index=check.check_index,
            stage=check.stage,
            expected_registered_browser_steps=check.expected_registered_browser_steps,
            observed_registered_browser_steps=check.expected_registered_browser_steps,
            automation_session_sha256="b" * 64,
            controller_input_audit_sha256=canonical_sha256(check.to_dict()),
            exclusive_automation_control=True,
            non_agent_input_event_count=count,
            manual_rescue_detected=bool(count),
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        manual_rescue_guard=_manual_rescue_guard(detect),
        manual_rescue_evidence_sink=lambda check, receipt: evidence.append(receipt),
        require_manual_rescue_guard=True,
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="manual browser intervention"):
        adapter.execute(
            ConcreteAction(
                action_id="blocked-by-input-audit",
                source_decision_id="decision",
                action_type=ActionType.NAVIGATE,
                parameters={"url": "https://example.invalid/"},
            )
        )
    assert evidence[-1].manual_rescue_detected is True
    assert environment.step_calls == 0


def test_webarena_reset_produces_typed_hashes_only_state_receipt(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        reset_state_attester=_reset_state_attester(),
    )
    task = replace(_task(), start_state_id="c" * 64)
    adapter.reset(task, episode_id="episode", seed=42)
    receipt = adapter.reset_state_receipt()

    assert receipt is not None
    assert receipt.episode_id == "episode"
    assert receipt.start_state_id == "c" * 64
    assert receipt.oracle_labels_observed is False
    assert set(receipt.commitments_sha256) == set(
        REGISTERED_WEBARENA_RESET_COMMITMENTS
    )


def _frozen_fault_classifier():
    def classify(error: Exception, operation: str):
        if not isinstance(error, ConnectionError):
            return None
        return WebArenaInfrastructureClassification(
            reason_code=(
                "BENCHMARK_SERVICE_UNAVAILABLE"
                if operation == "environment_factory"
                else (
                    "ENVIRONMENT_RESET_FAILED"
                    if operation == "reset"
                    else "BROWSER_CONTROLLER_DISCONNECTED"
                )
            ),
            failure_class=f"fixture_{operation}_connection_failure",
        )

    return FrozenWebArenaInfrastructureFaultClassifier(
        classifier_id="fixture-webarena-infrastructure-taxonomy",
        classifier_version="v1",
        benchmark_version="fixture-version",
        callback=classify,
        rules=tuple(
            WebArenaInfrastructureRule(
                operation=operation,
                exception_type="builtins.ConnectionError",
                reason_code=(
                    "BENCHMARK_SERVICE_UNAVAILABLE"
                    if operation == "environment_factory"
                    else (
                        "ENVIRONMENT_RESET_FAILED"
                        if operation == "reset"
                        else "BROWSER_CONTROLLER_DISCONNECTED"
                    )
                ),
                failure_class=f"fixture_{operation}_connection_failure",
            )
            for operation in (
                "environment_factory",
                "reset",
                "step",
                "screenshot_bytes_reset",
                "screenshot_bytes_pre_action",
                "screenshot_bytes_post_action",
                "screenshot_bytes_post_recovery",
            )
        ),
    )


def test_webarena_adapter_fails_closed_without_registered_integration() -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="module_that_does_not_exist_table2",
    )
    with pytest.raises(BenchmarkUnavailableError, match="unavailable"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_evaluation_webarena_requires_reset_state_attester(tmp_path: Path) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        require_reset_state_receipt=True,
    )
    with pytest.raises(BenchmarkUnavailableError, match="reset_state_attester"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_evaluation_webarena_requires_registered_page_settle_policy(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        require_page_settle_policy=True,
    )
    with pytest.raises(BenchmarkUnavailableError, match="page_settle_policy"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_evaluation_webarena_requires_registered_action_safety_policy(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        require_action_safety_policy=True,
    )
    with pytest.raises(BenchmarkUnavailableError, match="action_safety_policy"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_registered_action_safety_denial_consumes_request_without_browser_step(
    tmp_path: Path,
) -> None:
    environment = _Environment()
    mapped_actions: list[ConcreteAction] = []

    def map_action(action: ConcreteAction):
        mapped_actions.append(action)
        return {"next_page": 1}

    def deny(task, observation, action):
        assert type(task) is TaskSpecification
        assert type(observation) is Observation
        assert observation.stage is ObservationStage.RESET
        return WebArenaActionSafetyDecision(
            safety_policy_id="fixture-safety",
            safety_policy_version="v1",
            action_id=action.action_id,
            allowed=False,
            reason_code="DENY_UNREGISTERED_NAVIGATION_BOUNDARY",
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=map_action,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        action_safety_policy=_action_safety_policy(deny),
        require_action_safety_policy=True,
    )
    executor = Executor(adapter, budgets=REGISTERED_BUDGETS)
    executor.reset(_task(), episode_id="episode", seed=42)
    action = ConcreteAction(
        action_id="navigation-action",
        source_decision_id="decision",
        action_type=ActionType.NAVIGATE,
        parameters={"url": "https://outside.invalid/"},
    )

    result = executor.execute(action)

    assert result.status is ExecutionStatus.REJECTED
    assert result.executor_step == 1
    assert result.error_kind == "registered_safety_denial"
    assert "DENY_UNREGISTERED_NAVIGATION_BOUNDARY" in result.message
    assert result.evidence is not None
    assert result.evidence.status is ExecutionStatus.REJECTED
    assert result.evidence.controller_command is None
    assert result.evidence.safety_receipt is not None
    assert result.evidence.safety_receipt.allowed is False
    assert result.evidence.safety_receipt.reason_code == (
        "DENY_UNREGISTERED_NAVIGATION_BOUNDARY"
    )
    assert executor.steps_used == 1
    assert environment.step_calls == 0
    assert mapped_actions == []


def test_registered_action_safety_rejects_callback_input_mutation_before_step(
    tmp_path: Path,
) -> None:
    environment = _Environment()
    mapped_actions: list[ConcreteAction] = []

    def mutate(task, observation, action):
        del task
        observation.page_state["page"] = 999
        return WebArenaActionSafetyDecision(
            safety_policy_id="fixture-safety",
            safety_policy_version="v1",
            action_id=action.action_id,
            allowed=True,
            reason_code="ALLOW_REGISTERED_ACTION",
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: mapped_actions.append(action) or {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        action_safety_policy=_action_safety_policy(mutate),
        require_action_safety_policy=True,
    )
    executor = Executor(adapter, budgets=REGISTERED_BUDGETS)
    reset = executor.reset(_task(), episode_id="episode", seed=42)
    action = ConcreteAction(
        action_id="safe-action",
        source_decision_id="decision",
        action_type=ActionType.NAVIGATE,
        parameters={"url": "https://example.invalid/"},
    )

    with pytest.raises(BenchmarkStateError, match="action-safety policy failed") as exc:
        executor.execute(action)

    assert "mutated protected causal input" in str(exc.value.__cause__)
    assert reset.page_state == {"page": 0}
    assert executor.steps_used == 1
    assert environment.step_calls == 0
    assert mapped_actions == []


def test_registered_action_safety_rejects_wrong_decision_identity_before_step(
    tmp_path: Path,
) -> None:
    environment = _Environment()

    def wrong_identity(task, observation, action):
        del task, observation
        return WebArenaActionSafetyDecision(
            safety_policy_id="different-policy",
            safety_policy_version="v1",
            action_id=action.action_id,
            allowed=True,
            reason_code="ALLOW_REGISTERED_ACTION",
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        action_safety_policy=_action_safety_policy(wrong_identity),
        require_action_safety_policy=True,
    )
    executor = Executor(adapter, budgets=REGISTERED_BUDGETS)
    executor.reset(_task(), episode_id="episode", seed=42)

    with pytest.raises(BenchmarkStateError, match="action-safety policy failed") as exc:
        executor.execute(
            ConcreteAction(
                action_id="identity-bound-action",
                source_decision_id="decision",
                action_type=ActionType.NAVIGATE,
                parameters={"url": "https://example.invalid/"},
            )
        )

    assert "identity differs" in str(exc.value.__cause__)
    assert executor.steps_used == 1
    assert environment.step_calls == 0


def test_registered_page_settle_runs_before_reset_and_post_action_mapping(
    tmp_path: Path,
) -> None:
    calls: list[tuple[ObservationStage, float, bool]] = []

    def settle(environment, raw, stage, timeout, network_idle):
        assert isinstance(environment, _Environment)
        calls.append((stage, timeout, network_idle))
        return {"page": raw["page"] + 10}

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        page_settle_policy=_page_settle_policy(settle),
        require_page_settle_policy=True,
    )
    reset = adapter.reset(_task(), episode_id="episode", seed=42)
    assert reset.page_state == {"page": 10}
    adapter.execute(
        ConcreteAction(
            action_id="action",
            source_decision_id="decision",
            action_type=ActionType.NAVIGATE,
            parameters={"next_page": 1},
        )
    )
    post = adapter.observe(
        stage=ObservationStage.POST_ACTION,
        prior_action_id="action",
    )
    assert post.page_state == {"page": 11}
    assert calls == [
        (ObservationStage.RESET, 5.0, True),
        (ObservationStage.POST_ACTION, 5.0, True),
    ]


def test_registered_page_settle_fails_closed_before_mapper(
    tmp_path: Path,
) -> None:
    mapped: list[object] = []

    def mapper(raw, episode_id, stage, prior_action_id):
        mapped.append(raw)
        return _observation(raw, episode_id, stage, prior_action_id)

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=mapper,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        page_settle_policy=_page_settle_policy(
            lambda environment, raw, stage, timeout, network_idle: None
        ),
        require_page_settle_policy=True,
    )
    with pytest.raises(BenchmarkStateError, match="page-settle policy failed"):
        adapter.reset(_task(), episode_id="episode", seed=42)
    assert mapped == []


def test_registered_page_settle_requires_mapper_attestation(tmp_path: Path) -> None:
    def unsettled_mapper(raw, episode_id, stage, prior_action_id):
        return replace(
            _observation(raw, episode_id, stage, prior_action_id),
            page_settled=False,
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=unsettled_mapper,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        page_settle_policy=_page_settle_policy(),
        require_page_settle_policy=True,
    )
    with pytest.raises(BenchmarkStateError, match="settled state"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_webarena_adapter_uses_explicit_mappers_without_guessing_truth(
    tmp_path: Path,
) -> None:
    environment = _Environment()
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": action.parameters["next_page"]},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
    )
    reset = adapter.reset(_task(), episode_id="episode", seed=42)
    assert reset.page_state == {"page": 0}
    screenshot = Path(reset.screenshot_path or "")
    assert screenshot.parent == tmp_path / "screenshots"
    assert screenshot.name == f"000001-{reset.screenshot_sha256}.png"
    assert hashlib.sha256(screenshot.read_bytes()).hexdigest() == reset.screenshot_sha256
    result = adapter.execute(
        ConcreteAction(
            action_id="action",
            source_decision_id="decision",
            action_type=ActionType.NAVIGATE,
            parameters={"next_page": 1},
        )
    )
    assert result.state_changed is True
    assert result.evidence is not None
    assert result.evidence.status is ExecutionStatus.EXECUTED
    assert result.evidence.controller_command is not None
    assert result.evidence.controller_command.action_id == "action"
    assert result.evidence.controller_command.adapter_id == "fixture-adapter"
    expected_native = {"next_page": 1}
    assert result.evidence.controller_command.command_sha256 == canonical_sha256(
        expected_native
    )
    assert result.evidence.controller_command.canonical_byte_length == len(
        canonical_json(expected_native).encode("utf-8")
    )
    assert result.evidence.started_at_utc <= result.evidence.ended_at_utc
    observed = adapter.observe(
        stage=ObservationStage.POST_ACTION,
        prior_action_id="action",
    )
    assert observed.page_state == {"page": 1}
    assert Path(observed.screenshot_path or "").name == (
        f"000002-{observed.screenshot_sha256}.png"
    )
    signal = adapter.terminal_signal(_task())
    assert signal.terminate is True
    assert set(signal.to_dict()) == {
        "schema_version",
        "record_type",
        "event_id",
        "token_sha256",
        "terminate",
    }
    adapter.close()
    assert environment.closed is True


def test_gym_reward_done_and_info_cannot_change_runtime_recovery_or_memory(
    tmp_path: Path,
) -> None:
    action = ConcreteAction(
        action_id="action",
        source_decision_id="decision",
        action_type=ActionType.NAVIGATE,
        parameters={"next_page": 1},
    )

    def run_variant(
        name: str,
        *,
        reward: object,
        terminated: object,
        truncated: object,
        info: object,
    ) -> dict[str, object]:
        mapper_inputs: list[dict[str, int]] = []

        class OracleBearingStepEnvironment(_Environment):
            def step(self, native_action):
                self.state = {"page": int(native_action["next_page"])}
                return dict(self.state), reward, terminated, truncated, info

        def recording_observation(raw, episode_id, stage, prior_action_id):
            mapper_inputs.append(dict(raw))
            return _observation(raw, episode_id, stage, prior_action_id)

        runtime = tmp_path / name
        runtime.mkdir()
        adapter = WebArenaAdapter(
            benchmark_version="fixture-version",
            dependency_module="json",
            environment_factory=lambda task, seed: OracleBearingStepEnvironment(),
            observation_mapper=recording_observation,
            action_mapper=lambda concrete: {
                "next_page": concrete.parameters["next_page"]
            },
            screenshot_bytes_provider=_screenshot_bytes,
            episode_runtime_dir=runtime,
            terminal_signal_mapper=_terminal,
            environment_state_digester=_state_digester(),
        )
        before = adapter.reset(_task(), episode_id="episode", seed=42)
        mapped_execution = adapter.execute(action)
        after = adapter.observe(
            stage=ObservationStage.POST_ACTION,
            prior_action_id=action.action_id,
        )
        execution = ExecutionResult(
            action_id=action.action_id,
            status=mapped_execution.status,
            executor_step=1,
            state_changed=mapped_execution.state_changed,
            environment_error=mapped_execution.environment_error,
            error_kind=mapped_execution.error_kind,
            message=mapped_execution.message,
            internal_retry_count=mapped_execution.internal_retry_count,
            latency_ms=0.0,
        )
        assessment = TransitionAssessment(
            assessment_id="assessment",
            pre_observation_id=before.observation_id,
            post_observation_id=after.observation_id,
            executed_action_id=action.action_id,
            predicted_failure=True,
            failure_probability=0.9,
            failure_type="NO_PROGRESS",
            failure_type_probabilities={"NO_PROGRESS": 1.0},
            needs_recovery=True,
            needs_recovery_probability=0.9,
            recovery_strategy=RecoveryStrategy.RETRY,
            recovery_probabilities={"RETRY": 1.0},
        )
        decision = DecisionCombiner(
            switches_for(SystemID.E3), REGISTERED_RECOVERY_TRIGGER_CONFIG
        )
        trigger = decision.recovery_trigger(
            assessment=assessment,
            execution=execution,
            loop_detected=False,
        )
        shadow = decision.no_memory_shadow(
            trigger=trigger,
            assessment=assessment,
            incident_id="incident",
        )
        query = MemoryQuery(
            query_id="query",
            task_id=_task().task_id,
            episode_id=after.episode_id,
            incident_id="incident",
            post_failure_observation_id=after.observation_id,
            failed_action_id=action.action_id,
            diagnosis=trigger.diagnosis,
            duplicate_cluster_ids=("cluster",),
        )
        return {
            "mapper_inputs": mapper_inputs,
            "action": action.to_dict(),
            "execution": execution.to_dict(),
            "post_visible": {
                "screenshot_sha256": after.screenshot_sha256,
                "url": after.url,
                "title": after.title,
                "page_state": dict(after.page_state),
                "environment_error": after.environment_error,
            },
            "trigger": trigger.to_dict(),
            "recovery": shadow.to_dict(),
            "memory_query": query.to_dict(),
        }

    ordinary = run_variant(
        "ordinary",
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"oracle_success": False, "task_reward": 0.0},
    )
    hostile = run_variant(
        "hostile",
        reward=10_000.0,
        terminated=True,
        truncated=True,
        info={
            "oracle_success": True,
            "task_reward": 10_000.0,
            "reference_action": "leak",
        },
    )
    assert ordinary == hostile
    assert ordinary["mapper_inputs"] == [{"page": 0}, {"page": 1}]


def test_webarena_step_contract_rejects_non_tuple_before_any_post_callback(
    tmp_path: Path,
) -> None:
    mapper_inputs: list[dict[str, int]] = []

    class MalformedStepEnvironment(_Environment):
        def step(self, native_action):
            self.state = {"page": int(native_action["next_page"])}
            return {"observation": dict(self.state), "oracle_success": True}

    def recording_observation(raw, episode_id, stage, prior_action_id):
        mapper_inputs.append(dict(raw))
        return _observation(raw, episode_id, stage, prior_action_id)

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: MalformedStepEnvironment(),
        observation_mapper=recording_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="Gym 4/5-tuple"):
        adapter.execute(
            ConcreteAction(
                action_id="action",
                source_decision_id="decision",
                action_type=ActionType.NAVIGATE,
                parameters={"next_page": 1},
            )
        )
    assert mapper_inputs == [{"page": 0}]


def test_webarena_rejects_mapper_hash_that_differs_from_captured_bytes(
    tmp_path: Path,
) -> None:
    def wrong_hash_observation(raw, episode_id, stage, prior_action_id):
        return replace(
            _observation(raw, episode_id, stage, prior_action_id),
            screenshot_sha256="0" * 64,
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=wrong_hash_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
    )
    with pytest.raises(BenchmarkStateError, match="hash differs from captured bytes"):
        adapter.reset(_task(), episode_id="episode", seed=42)
    assert not list((tmp_path / "screenshots").iterdir())


def test_shared_latest_source_is_copied_and_policy_rejects_artifact_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "latest.png"
    initial = b"shared-latest-before-mapping"
    source.write_bytes(initial)

    def source_observation(raw, episode_id, stage, prior_action_id):
        return replace(
            _observation(raw, episode_id, stage, prior_action_id),
            screenshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=source_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=lambda raw, observation: source.read_bytes(),
        episode_runtime_dir=runtime,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
    )
    observation = adapter.reset(_task(), episode_id="episode", seed=42)
    artifact = Path(observation.screenshot_path or "")
    assert artifact != source
    assert artifact.read_bytes() == initial

    source.write_bytes(b"shared-latest-after-mapping")
    policy_view = ObservationBuilder().pre_action(_task(), observation)
    assert Path(policy_view.screenshot_path or "").read_bytes() == initial

    artifact.chmod(0o600)
    artifact.write_bytes(b"mutated-after-map")
    with pytest.raises(CausalBoundaryError, match="read-only|bytes differ"):
        ObservationBuilder().pre_action(_task(), observation)


def test_mapper_screenshot_path_escape_symlink_and_future_canaries(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    symlink = tmp_path / "latest.png"
    symlink.symlink_to(outside)
    future = tmp_path / "future.png"

    for index, hostile_path in enumerate((outside, symlink, future), start=1):
        runtime = tmp_path / f"runtime-{index}"
        runtime.mkdir()

        def path_observation(raw, episode_id, stage, prior_action_id):
            return replace(
                _observation(raw, episode_id, stage, prior_action_id),
                screenshot_path=str(hostile_path),
            )

        adapter = WebArenaAdapter(
            benchmark_version="fixture-version",
            dependency_module="json",
            environment_factory=lambda task, seed: _Environment(),
            observation_mapper=path_observation,
            action_mapper=lambda action: action.parameters,
            screenshot_bytes_provider=_screenshot_bytes,
            episode_runtime_dir=runtime,
            terminal_signal_mapper=_terminal,
            environment_state_digester=_state_digester(),
        )
        with pytest.raises(BenchmarkStateError, match="must not supply a screenshot path"):
            adapter.reset(_task(), episode_id=f"episode-{index}", seed=42)


def test_adapter_rejects_symlink_screenshot_artifact_directory(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime / "screenshots").symlink_to(outside, target_is_directory=True)
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=runtime,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
    )
    with pytest.raises(BenchmarkStateError, match="must not be a symlink"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_terminal_mapper_receives_only_a_deep_frozen_snapshot(tmp_path: Path) -> None:
    environment = _Environment()

    def hostile_mapper(snapshot, task, binding):
        del task, binding
        assert type(snapshot) is FrozenWebArenaObservationSnapshot
        assert not hasattr(snapshot, "step")
        snapshot.observation.page_state["page"] = 99

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=hostile_mapper,
        environment_state_digester=_state_digester(),
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="immutable observation contract"):
        adapter.terminal_signal(_task())
    assert environment.state == {"page": 0}


def test_terminal_mapper_snapshot_digest_rejects_low_level_mutation(
    tmp_path: Path,
) -> None:
    environment = _Environment()

    def hostile_mapper(snapshot, task, binding):
        del task, binding
        object.__setattr__(snapshot.observation, "title", "tampered")
        return OpaqueTerminalSignal(
            event_id="hostile-event",
            token_sha256="f" * 64,
            terminate=False,
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=hostile_mapper,
        environment_state_digester=_state_digester(),
    )
    original = adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="mutated.*snapshot"):
        adapter.terminal_signal(_task())
    assert original.title == ""


def test_terminal_mapper_live_state_digest_rejects_closure_side_effect(
    tmp_path: Path,
) -> None:
    environment = _Environment()

    def hostile_mapper(snapshot, task, binding):
        del snapshot, task, binding
        environment.state["page"] = 99
        return OpaqueTerminalSignal(
            event_id="hostile-event",
            token_sha256="f" * 64,
            terminate=False,
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: environment,
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=hostile_mapper,
        environment_state_digester=_state_digester(),
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="changed live environment state"):
        adapter.terminal_signal(_task())


def test_webarena_requires_registered_environment_state_digester(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
    )
    with pytest.raises(BenchmarkUnavailableError, match="environment_state_digester"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_webarena_observation_mapper_cannot_expose_oracle_fields(
    tmp_path: Path,
) -> None:
    def leaking_observation(raw, episode_id, stage, prior_action_id):
        value = _observation(raw, episode_id, stage, prior_action_id)
        return Observation(
            **{
                **{
                    name: getattr(value, name)
                    for name in value.__dataclass_fields__
                },
                "page_state": {"oracle_success": True},
            }
        )

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=leaking_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
    )
    with pytest.raises(ValueError, match="oracle"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_webarena_explicit_classifier_types_registered_reset_fault(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _ResetFailureEnvironment(
            ConnectionError("registered service outage")
        ),
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        infrastructure_fault_classifier=_frozen_fault_classifier(),
    )
    with pytest.raises(InfrastructureInvalidError) as caught:
        adapter.reset(_task(), episode_id="episode", seed=42)
    assert caught.value.reason_code == "ENVIRONMENT_RESET_FAILED"
    assert caught.value.operation == "reset"
    assert caught.value.adapter_id == "fixture-adapter"
    assert caught.value.adapter_version == "fixture-adapter-v1"
    assert caught.value.adapter_evidence["retryable"] is True
    assert caught.value.adapter_evidence["task_id"] == "webarena.0"
    assert caught.value.adapter_evidence["classifier_rules_sha256"] == (
        _frozen_fault_classifier().rules_sha256
    )


def test_webarena_explicit_classifier_types_registered_step_fault(
    tmp_path: Path,
) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _StepFailureEnvironment(),
        observation_mapper=_observation,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        infrastructure_fault_classifier=_frozen_fault_classifier(),
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(InfrastructureInvalidError) as caught:
        adapter.execute(
            ConcreteAction(
                action_id="action",
                source_decision_id="decision",
                action_type=ActionType.NAVIGATE,
                parameters={"next_page": 1},
            )
        )
    assert caught.value.reason_code == "BROWSER_CONTROLLER_DISCONNECTED"
    assert caught.value.operation == "step"
    partial = getattr(caught.value, "adapter_execution")
    assert partial.status is ExecutionStatus.ERROR
    assert partial.evidence is not None
    assert partial.evidence.controller_command is not None
    assert partial.evidence.controller_command.command_sha256 == canonical_sha256(
        {"next_page": 1}
    )


def test_webarena_observation_mapper_fault_is_never_infrastructure(
    tmp_path: Path,
) -> None:
    def failed_mapper(raw, episode_id, stage, prior_action_id):
        if stage is ObservationStage.POST_ACTION:
            raise ConnectionError("registered mapper-side controller loss")
        return _observation(raw, episode_id, stage, prior_action_id)

    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=failed_mapper,
        action_mapper=lambda action: {"next_page": 1},
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        infrastructure_fault_classifier=_frozen_fault_classifier(),
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="fatal integration defects"):
        adapter.execute(
            ConcreteAction(
                action_id="action",
                source_decision_id="decision",
                action_type=ActionType.NAVIGATE,
                parameters={"next_page": 1},
            )
        )


def test_webarena_action_mapper_fault_is_never_infrastructure(tmp_path: Path) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _Environment(),
        observation_mapper=_observation,
        action_mapper=lambda action: (_ for _ in ()).throw(
            ConnectionError("mapper defect")
        ),
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        infrastructure_fault_classifier=_frozen_fault_classifier(),
    )
    adapter.reset(_task(), episode_id="episode", seed=42)
    with pytest.raises(BenchmarkStateError, match="fatal integration defects"):
        adapter.execute(
            ConcreteAction(
                action_id="action",
                source_decision_id="decision",
                action_type=ActionType.NAVIGATE,
                parameters={"next_page": 1},
            )
        )


def test_webarena_classifier_output_must_match_frozen_exact_rule(
    tmp_path: Path,
) -> None:
    classifier = FrozenWebArenaInfrastructureFaultClassifier(
        classifier_id="fixture-classifier",
        classifier_version="v1",
        benchmark_version="fixture-version",
        callback=lambda error, operation: WebArenaInfrastructureClassification(
            reason_code="BROWSER_CONTROLLER_DISCONNECTED",
            failure_class="forged_reset_class",
        ),
        rules=(
            WebArenaInfrastructureRule(
                operation="reset",
                exception_type="builtins.ConnectionError",
                reason_code="ENVIRONMENT_RESET_FAILED",
                failure_class="registered_reset_class",
            ),
        ),
    )
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _ResetFailureEnvironment(
            ConnectionError("registered type, forged classification")
        ),
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        infrastructure_fault_classifier=classifier,
    )
    with pytest.raises(ValueError, match="differs from its exact frozen rule"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_webarena_unclassified_failure_remains_fatal(tmp_path: Path) -> None:
    adapter = WebArenaAdapter(
        benchmark_version="fixture-version",
        adapter_id="fixture-adapter",
        adapter_version="fixture-adapter-v1",
        dependency_module="json",
        environment_factory=lambda task, seed: _ResetFailureEnvironment(
            ValueError("unregistered integration drift")
        ),
        observation_mapper=_observation,
        action_mapper=lambda action: action.parameters,
        screenshot_bytes_provider=_screenshot_bytes,
        episode_runtime_dir=tmp_path,
        terminal_signal_mapper=_terminal,
        environment_state_digester=_state_digester(),
        infrastructure_fault_classifier=_frozen_fault_classifier(),
    )
    with pytest.raises(BenchmarkStateError, match="will not guess"):
        adapter.reset(_task(), episode_id="episode", seed=42)


def test_webarena_classifier_must_match_pinned_benchmark_version() -> None:
    classifier = FrozenWebArenaInfrastructureFaultClassifier(
        classifier_id="fixture-classifier",
        classifier_version="v1",
        benchmark_version="different-version",
        callback=lambda error, operation: None,
        rules=(
            WebArenaInfrastructureRule(
                operation="reset",
                exception_type="builtins.ConnectionError",
                reason_code="ENVIRONMENT_RESET_FAILED",
                failure_class="fixture_reset_connection_failure",
            ),
        ),
    )
    with pytest.raises(ValueError, match="versions differ"):
        WebArenaAdapter(
            benchmark_version="fixture-version",
            infrastructure_fault_classifier=classifier,
        )
