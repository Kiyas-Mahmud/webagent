from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from web_agent.eval.table2 import process_broker_webarena_backend as backend_module
from web_agent.eval.table2 import execution_guard as execution_guard_module
from web_agent.eval.table2.common import (
    SchemaError,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.execution_guard import InfrastructureInvalidError
from web_agent.eval.table2.package_validator import (
    _validate_infrastructure_invalid_package,
)
from web_agent.eval.table2.process_broker import ProcessIsolatedBroker
from web_agent.eval.table2.process_broker_protocol import (
    PROCESS_BROKER_INFRASTRUCTURE_INVALID_SCHEMA_VERSION,
    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION,
    ProcessBrokerProtocolError,
    expected_verifier_receipt_binding,
    registered_browser_error_observation_url,
    validate_runtime_infrastructure_invalid,
)
from web_agent.eval.table2.process_broker_webarena_backend import (
    PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
    PROCESS_BROKER_WEBARENA_BACKEND_SCHEMA_VERSION,
    PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
    ProcessBrokerWebArenaEnvironmentAdapter,
    create_backend,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionStatus,
    ObservationStage,
    PreActionDecision,
    RuntimeStartState,
    SystemID,
    TaskSpecification,
    TerminalReason,
    VerifierReceiptBinding,
    probability_map,
)
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.executor import Executor
from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import (
    REGISTERED_BUDGETS,
    RuntimeProtocol,
    switches_for,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_FACTORY_SOURCE = (
    "src/web_agent/eval/table2/process_broker_fixture_backend.py"
)
FIXTURE_FACTORY_ENTRYPOINT = (
    "web_agent.eval.table2.process_broker_fixture_backend:"
    "create_environment_adapter"
)


def _task() -> TaskSpecification:
    start_state = RuntimeStartState(
        sites=("shopping",),
        start_url="https://fixture.invalid/start",
        require_login=False,
        storage_state=None,
        geolocation=None,
        require_reset=True,
    )
    return TaskSpecification(
        task_id="task-1",
        goal="complete the visible fixture task",
        benchmark_id="webarena",
        benchmark_version="webarena-fixture-v1",
        start_state_id=start_state.start_state_sha256,
        site="shopping",
        start_url=start_state.start_url,
        development_partition=True,
        destructive_actions_allowed=False,
        metadata={
            "task_partition": "normal",
            "upstream_index": 1,
            "benchmark_task_id": "webarena.1",
            "source_content_sha256": "b" * 64,
        },
        runtime_start_state=start_state,
    )


def _backend_config(task: TaskSpecification, runtime_dir: Path) -> dict:
    return {
        "schema_version": PROCESS_BROKER_WEBARENA_BACKEND_SCHEMA_VERSION,
        "adapter_factory_entrypoint": FIXTURE_FACTORY_ENTRYPOINT,
        "adapter_factory_source_relative_path": FIXTURE_FACTORY_SOURCE,
        "adapter_factory_source_sha256": sha256_file(ROOT / FIXTURE_FACTORY_SOURCE),
        "task_specification": task.to_dict(),
        "episode_runtime_dir": str(runtime_dir),
    }


def _binding(
    observation: object,
    action: ConcreteAction | None,
) -> VerifierReceiptBinding:
    assert hasattr(observation, "to_dict")
    return VerifierReceiptBinding.from_dict(
        expected_verifier_receipt_binding(
            observation=observation.to_dict(),
            action=action.to_dict() if action is not None else None,
        )
    )


def test_real_backend_entrypoint_runs_environment_adapter_through_strict_broker(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    fixture_module = "web_agent.eval.table2.process_broker_fixture_backend"
    preexisting_fixture = sys.modules.get(fixture_module)
    if preexisting_fixture is None:
        sys.modules.pop(fixture_module, None)

    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )
    with broker:
        registered_sources = {
            row["relative_path"]: row["sha256"]
            for row in broker.receipt.source_files
        }
        assert registered_sources[PROCESS_BROKER_WEBARENA_BACKEND_SOURCE] == (
            sha256_file(ROOT / PROCESS_BROKER_WEBARENA_BACKEND_SOURCE)
        )
        assert registered_sources[FIXTURE_FACTORY_SOURCE] == sha256_file(
            ROOT / FIXTURE_FACTORY_SOURCE
        )
        execution_guard_source = "src/web_agent/eval/table2/execution_guard.py"
        assert registered_sources[execution_guard_source] == sha256_file(
            ROOT / execution_guard_source
        )
        assert broker.receipt.runtime_value_provenance_attested is False
        assert broker.receipt.external_deployment_authority is False
        assert broker.receipt.inner_schema_registry_version == (
            PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION
        )

        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        assert not hasattr(adapter, "evaluate")
        assert not hasattr(adapter, "evaluator")
        assert not hasattr(adapter, "oracle")

        reset_observation = adapter.reset(task, episode_id="episode-1", seed=42)
        reset_receipt = adapter.reset_state_receipt()
        assert reset_observation.stage is ObservationStage.RESET
        assert reset_receipt is not None
        assert reset_receipt.oracle_labels_observed is False
        assert adapter.terminal_signal(
            task, _binding(reset_observation, None)
        ).terminate is False

        action = ConcreteAction(
            action_id="action-1",
            source_decision_id="decision-1",
            action_type=ActionType.NAVIGATE,
            parameters={"url": "https://fixture.invalid/done"},
        )
        execution = adapter.execute(action)
        assert execution.evidence is not None
        assert execution.evidence.action_id == action.action_id
        post_observation = adapter.observe(
            stage=ObservationStage.POST_ACTION,
            prior_action_id=action.action_id,
        )
        assert post_observation.prior_action_id == action.action_id
        assert adapter.terminal_signal(
            task, _binding(post_observation, action)
        ).terminate is True
        adapter.close()

    assert broker.cleanup_receipt.external_deployment_authority is False
    if preexisting_fixture is None:
        assert fixture_module not in sys.modules
    else:
        assert sys.modules.get(fixture_module) is preexisting_fixture


def test_process_backed_executor_registers_safety_rejection_without_browser_step(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )
    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        executor = Executor(adapter, budgets=REGISTERED_BUDGETS)
        reset = executor.reset(task, episode_id="episode-safety", seed=42)
        assert executor.terminal_signal(_binding(reset, None)).terminate is False
        action = ConcreteAction(
            action_id="rejected-safety-action",
            source_decision_id="decision-safety",
            action_type=ActionType.NAVIGATE,
            parameters={"url": "https://fixture.invalid/never-dispatched"},
            recovery_attempt_id="recovery-attempt-1",
            destructive=True,
        )
        execution = executor.execute(action)
        assert execution.status is ExecutionStatus.REJECTED
        assert execution.error_kind == "safety_rejection"
        assert execution.executor_step == executor.steps_used == 1
        assert execution.state_changed is False
        assert execution.environment_error is False
        post = executor.observe_after(action, recovery=True)
        assert post.stage is ObservationStage.POST_RECOVERY
        assert post.prior_action_id == action.action_id
        assert post.url == reset.url
        assert post.page_state["visible_text"] == "fixture action count 0"
        assert executor.terminal_signal(_binding(post, action)).terminate is False
        assert executor.steps_used == 1
        executor.close()


def test_episode_runner_process_backed_parameter_rejection_is_observed_and_bound(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )

    def unresolved_type(_task, observation, _rng):
        return PreActionDecision(
            decision_id=f"decision:{observation.observation_id}",
            observation_id=observation.observation_id,
            action_type=ActionType.TYPE,
            action_probabilities=probability_map(
                tuple(item.value for item in ActionType),
                ActionType.TYPE.value,
            ),
            bbox=(0.1, 0.1, 0.2, 0.2),
            grounding_confidence=1.0,
            confidence_before=1.0,
            input_observation_ids=(observation.observation_id,),
            policy_id="rejected-parameter-policy",
            policy_version="v1",
            # Missing text intentionally forces the registered local
            # parameter-resolution rejection.
            parameter_hints={},
        )

    policy = SystemPolicy(
        CallablePolicyAdapter(
            policy_id="rejected-parameter-policy",
            policy_version="v1",
            kind=PolicyKind.TRAINED,
            checkpoint_sha256="a" * 64,
            action_predictor=unresolved_type,
        ),
        switches_for(SystemID.E1),
    )
    protocol = RuntimeProtocol(
        protocol_id="table2-rejected-parameter-test-v1",
        # This fixture namespace has a reset-stage seed inside the broker's
        # registered signed-64-bit transport range.
        campaign_id="fixture-smoke",
        provider_id="deterministic-parameter-provider",
        provider_version="v1",
        benchmark_id="webarena",
        benchmark_version=task.benchmark_version,
    )
    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        runner = EpisodeRunner(
            protocol=protocol,
            system_policy=policy,
            provider=DeterministicParameterProvider(),
            executor=Executor(adapter, budgets=protocol.budgets),
        )
        summary = runner.run(task, repeat_id=0, model_seed=42)
        # Every rejected request is charged exactly once.  The process-backed
        # post-action observation and sealed receipt remain valid even though
        # none of the requests reaches the child browser/controller action
        # counter, so the episode deterministically exhausts its registered
        # executor budget.
        assert summary.terminal_reason is TerminalReason.ACTION_BUDGET_EXHAUSTED
        assert summary.executor_steps == summary.normal_actions == 30
        assert summary.recovery_actions == 0


def test_backend_rejects_oracle_config_and_unattested_factory_source(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    config = _backend_config(_task(), runtime_dir)

    with_oracle = copy.deepcopy(config)
    with_oracle["task_specification"]["metadata"]["oracle"] = {
        "reference_answer": "secret"
    }
    with pytest.raises(
        ProcessBrokerProtocolError,
        match="evaluator/oracle aliases",
    ):
        create_backend(with_oracle)

    wrong_source = copy.deepcopy(config)
    wrong_source["adapter_factory_source_sha256"] = "0" * 64
    with pytest.raises(ProcessBrokerProtocolError, match="source hash differs"):
        create_backend(wrong_source)

    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        sealed_backend_config=config,
        startup_timeout_seconds=1.0,
    )
    with pytest.raises(Exception):
        broker.start()
    assert broker.cleaned


def test_child_entrypoint_shape_is_rejected_before_module_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    config = _backend_config(_task(), runtime_dir)
    config.update(
        {
            "adapter_factory_entrypoint": (
                "web_agent.benchmarks.browsergym_webarena:"
                "BrowserGymWebArenaRuntimeFactory"
            ),
            "adapter_factory_source_relative_path": (
                "src/web_agent/benchmarks/browsergym_webarena.py"
            ),
            "adapter_factory_source_sha256": sha256_file(
                ROOT / "src/web_agent/benchmarks/browsergym_webarena.py"
            ),
        }
    )
    imported = False

    def forbidden_import(_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("malformed entrypoint must fail before import")

    monkeypatch.setattr(backend_module.importlib, "import_module", forbidden_import)
    with pytest.raises(
        ProcessBrokerProtocolError,
        match="top-level synchronous function",
    ):
        create_backend(config)
    assert imported is False


def test_sealed_callback_guard_is_separate_from_zero_manual_rescue_chain(
    tmp_path: Path,
) -> None:
    from web_agent.eval.table2.process_broker_fixture_backend import (
        create_environment_adapter,
    )

    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    adapter = create_environment_adapter(
        task=_task(),
        episode_runtime_dir=runtime_dir,
    )
    guard = backend_module._SealedCallbackGuardSidecar(runtime_dir)
    assert backend_module._call_state_preserving_sealed_callback(
        adapter,
        lambda: "opaque-only-placeholder",
        callback_kind="sealed_transition",
        context="fixture sealed transition",
        evidence_sidecar=guard,
    ) == "opaque-only-placeholder"
    manual_identity = adapter.process_broker_manual_rescue_sidecar_identity()
    callback_identity = guard.identity()
    assert manual_identity["record_count"] == 0
    assert manual_identity["tail_sha256"] is None
    assert callback_identity["record_count"] == 1
    assert callback_identity["tail_sha256"] is not None
    assert callback_identity["relative_path"] == (
        "sealed_callback_state_guard.child.jsonl"
    )
    guard.close()
    adapter.close()


def test_sealed_callback_guard_rejects_environment_mutation_without_receipt(
    tmp_path: Path,
) -> None:
    from web_agent.eval.table2.process_broker_fixture_backend import (
        create_environment_adapter,
    )

    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    adapter = create_environment_adapter(
        task=_task(),
        episode_runtime_dir=runtime_dir,
    )
    guard = backend_module._SealedCallbackGuardSidecar(runtime_dir)

    def mutate_page() -> None:
        adapter._backend._raw_page.url = "https://fixture.invalid/mutated"

    with pytest.raises(
        ProcessBrokerProtocolError,
        match="mutated the child-owned environment state",
    ):
        backend_module._call_state_preserving_sealed_callback(
            adapter,
            mutate_page,
            callback_kind="sealed_transition",
            context="fixture sealed transition",
            evidence_sidecar=guard,
        )
    assert guard.identity()["record_count"] == 0
    guard.close()
    adapter.close()


def test_reset_binds_complete_oracle_blind_task_record_across_processes(
    tmp_path: Path,
) -> None:
    configured_task = _task()
    runtime_task = replace(
        configured_task,
        goal="a different runtime-visible goal",
        metadata={
            **configured_task.metadata,
            "source_content_sha256": "c" * 64,
        },
    )
    assert runtime_task.record_sha256 != configured_task.record_sha256
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(configured_task, runtime_dir),
    )

    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=configured_task.benchmark_version,
        )
        with pytest.raises(
            ProcessBrokerProtocolError,
            match="REGISTERED_REQUEST_REJECTED",
        ):
            adapter.reset(runtime_task, episode_id="episode-mismatch", seed=42)
        adapter.close()


def test_registered_infrastructure_invalid_round_trips_sanitized_and_poisoned(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )

    with broker:
        client = broker.runtime_client()
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=client,
            benchmark_version=task.benchmark_version,
        )
        executor = Executor(adapter, budgets=REGISTERED_BUDGETS)
        reset_observation = executor.reset(
            task,
            episode_id="episode-infrastructure",
            seed=42,
        )
        executor.terminal_signal(_binding(reset_observation, None))
        action = ConcreteAction(
            action_id="emit-infrastructure-invalid",
            source_decision_id="decision-infrastructure",
            action_type=ActionType.PRESS_KEY,
            parameters={"key": "ENTER"},
        )
        with pytest.raises(InfrastructureInvalidError) as captured:
            executor.execute(action)
        error = captured.value
        assert type(error) is InfrastructureInvalidError
        assert error.reason_code == "BROWSER_CONTROLLER_DISCONNECTED"
        assert error.adapter_id == "fixture-process-webarena"
        assert error.adapter_version == "v1"
        assert error.operation == "step"
        assert error.adapter_evidence == {
            "adapter_event_id": "episode-infrastructure:infrastructure:1",
            "failure_class": "fixture_controller_disconnected",
            "diagnostic_sha256": "c" * 64,
            "retryable": True,
        }
        partial = getattr(error, "adapter_execution", None)
        assert partial is not None
        assert partial.status is ExecutionStatus.ERROR
        assert partial.message == "native controller step did not return"
        assert partial.evidence is not None
        assert partial.evidence.action_id == action.action_id
        assert partial.evidence.controller_command is not None
        assert partial.evidence.controller_command.action_id == action.action_id
        assert partial.evidence.safety_receipt is not None
        assert partial.evidence.safety_receipt.action_id == action.action_id
        preserved = getattr(error, "execution_result", None)
        assert preserved is not None
        assert preserved.status is ExecutionStatus.ERROR
        assert preserved.evidence is not None
        assert preserved.evidence.controller_command is not None
        assert preserved.evidence.controller_command.action_id == action.action_id
        assert not hasattr(error, "raw_exception_text")
        with pytest.raises(
            ProcessBrokerProtocolError,
            match="requires control cleanup",
        ):
            client.close(
                episode_id="episode-infrastructure",
                task_id=task.task_id,
            )
        adapter.close()

    assert broker.cleanup_receipt.graceful_authenticated_shutdown is True


def test_cross_operation_infrastructure_claim_is_rejected_not_authorized(
    tmp_path: Path,
) -> None:
    task = replace(
        _task(),
        goal="fixture forged terminal infrastructure operation",
    )
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )

    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        reset_observation = adapter.reset(
            task,
            episode_id="episode-cross-operation",
            seed=42,
        )
        with pytest.raises(
            ProcessBrokerProtocolError,
            match="REGISTERED_REQUEST_REJECTED",
        ):
            adapter.terminal_signal(task, _binding(reset_observation, None))
        adapter.close()

    assert broker.cleanup_receipt.graceful_authenticated_shutdown is True


def test_cross_action_infrastructure_execution_is_rejected_not_authorized(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )

    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        reset_observation = adapter.reset(
            task,
            episode_id="episode-cross-action",
            seed=42,
        )
        adapter.terminal_signal(task, _binding(reset_observation, None))
        action = ConcreteAction(
            action_id="emit-infrastructure-invalid-wrong-action",
            source_decision_id="decision-cross-action",
            action_type=ActionType.PRESS_KEY,
            parameters={"key": "ENTER"},
        )
        with pytest.raises(
            ProcessBrokerProtocolError,
            match="REGISTERED_REQUEST_REJECTED",
        ):
            adapter.execute(action)
        adapter.close()

    assert broker.cleanup_receipt.graceful_authenticated_shutdown is True


def test_environment_error_observation_round_trips_through_real_broker_ipc(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )

    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        reset_observation = adapter.reset(task, episode_id="episode-error", seed=42)
        assert adapter.terminal_signal(
            task,
            _binding(reset_observation, None),
        ).terminate is False
        action = ConcreteAction(
            action_id="emit-browser-error",
            source_decision_id="decision-error",
            action_type=ActionType.PRESS_KEY,
            parameters={"key": "ENTER"},
        )
        adapter.execute(action)
        observation = adapter.observe(
            stage=ObservationStage.POST_ACTION,
            prior_action_id=action.action_id,
        )
        expected_url = registered_browser_error_observation_url("TIMEOUTERROR")
        assert observation.environment_error is True
        assert observation.url == expected_url
        assert observation.page_state["has_browser_error"] is True
        assert observation.page_state["browser_error_kind"] == "TIMEOUTERROR"
        assert adapter.terminal_signal(
            task,
            _binding(observation, action),
        ).terminate is True
        adapter.close()


def test_incomplete_episode_close_defers_to_control_cleanup_without_masking_error(
    tmp_path: Path,
) -> None:
    task = _task()
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(task, runtime_dir),
    )

    with broker:
        adapter = ProcessBrokerWebArenaEnvironmentAdapter(
            client=broker.runtime_client(),
            benchmark_version=task.benchmark_version,
        )
        reset_observation = adapter.reset(task, episode_id="episode-failed", seed=42)
        adapter.terminal_signal(task, _binding(reset_observation, None))
        action = ConcreteAction(
            action_id="action-before-assessment-error",
            source_decision_id="decision-before-assessment-error",
            action_type=ActionType.PRESS_KEY,
            parameters={"key": "ENTER"},
        )
        adapter.execute(action)
        adapter.observe(
            stage=ObservationStage.POST_ACTION,
            prior_action_id=action.action_id,
        )

        with pytest.raises(RuntimeError, match="assessment failed"):
            try:
                raise RuntimeError("assessment failed")
            finally:
                adapter.close()

    assert broker.cleanup_receipt.graceful_authenticated_shutdown is True


def test_process_broker_dependency_closure_rejects_unsafe_or_duplicate_paths(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    common = {
        "repository_root": ROOT,
        "backend_entrypoint": PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        "backend_source_relative_path": PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        "sealed_backend_config": _backend_config(_task(), runtime_dir),
    }
    for closure in (
        [FIXTURE_FACTORY_SOURCE],
        ("../outside.py",),
        (FIXTURE_FACTORY_SOURCE, FIXTURE_FACTORY_SOURCE),
        (PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,),
    ):
        with pytest.raises(Exception):
            ProcessIsolatedBroker(
                **common,
                backend_dependency_source_relative_paths=closure,
            )


def test_process_broker_dependency_closure_rejects_symlink_ancestry(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    package = repository / "src" / "package"
    real_directory = package / "real"
    real_directory.mkdir(parents=True)
    (package / "backend.py").write_text("def create(config):\n    return config\n")
    (real_directory / "dependency.py").write_text("VALUE = 1\n")
    (package / "alias").symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(Exception, match="symlink component"):
        ProcessIsolatedBroker(
            repository_root=repository,
            backend_entrypoint="package.backend:create",
            backend_source_relative_path="src/package/backend.py",
            backend_dependency_source_relative_paths=(
                "src/package/alias/dependency.py",
            ),
        )


def test_parent_runtime_bridge_import_identity_is_receipt_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    monkeypatch.setattr(
        backend_module,
        "PROCESS_BROKER_IMPORT_SOURCE_SHA256",
        "0" * 64,
    )
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(_task(), runtime_dir),
    )
    with pytest.raises(
        SchemaError,
        match="imported parent backend module identity differs",
    ):
        broker.start()
    assert broker.cleaned


def test_parent_infrastructure_error_definition_is_receipt_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "episode"
    runtime_dir.mkdir()
    monkeypatch.setattr(
        execution_guard_module,
        "PROCESS_BROKER_IMPORT_SOURCE_SHA256",
        "0" * 64,
    )
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
        backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
        backend_dependency_source_relative_paths=(FIXTURE_FACTORY_SOURCE,),
        sealed_backend_config=_backend_config(_task(), runtime_dir),
    )
    with pytest.raises(
        SchemaError,
        match="imported parent broker module identity differs",
    ):
        broker.start()
    assert broker.cleaned


def test_infrastructure_invalid_schema_rejects_extra_or_unregistered_data() -> None:
    value = {
        "schema_version": PROCESS_BROKER_INFRASTRUCTURE_INVALID_SCHEMA_VERSION,
        "record_type": "InfrastructureInvalidError",
        "reason_code": "BROWSER_CONTROLLER_DISCONNECTED",
        "adapter_id": "fixture-process-webarena",
        "adapter_version": "v1",
        "operation": "step",
        "adapter_evidence": {
            "adapter_event_id": "episode-1:infrastructure:1",
            "failure_class": "fixture_controller_disconnected",
            "diagnostic_sha256": "c" * 64,
            "retryable": True,
        },
    }
    assert validate_runtime_infrastructure_invalid(value) == value
    request_identity = {"episode_id": "episode-1", "task_id": "task-1"}
    assert validate_runtime_infrastructure_invalid(
        value,
        request_operation="runtime_execute",
        episode_id="episode-1",
        request_payload=request_identity,
    ) == value
    with pytest.raises(ProcessBrokerProtocolError, match="not causal"):
        validate_runtime_infrastructure_invalid(
            value,
            request_operation="runtime_terminal",
            episode_id="episode-1",
            request_payload=request_identity,
        )
    with pytest.raises(ProcessBrokerProtocolError, match="not episode-bound"):
        validate_runtime_infrastructure_invalid(
            value,
            request_operation="runtime_execute",
            episode_id="episode-2",
            request_payload={"episode_id": "episode-2", "task_id": "task-1"},
        )

    mutations = []
    extra = copy.deepcopy(value)
    extra["adapter_evidence"]["raw_exception_text"] = "secret detail"
    mutations.append(extra)
    wrong_reason = copy.deepcopy(value)
    wrong_reason["reason_code"] = "ENVIRONMENT_RESET_FAILED"
    mutations.append(wrong_reason)
    non_retryable = copy.deepcopy(value)
    non_retryable["adapter_evidence"]["retryable"] = False
    mutations.append(non_retryable)
    forbidden = copy.deepcopy(value)
    forbidden["adapter_evidence"]["oracle"] = "secret"
    mutations.append(forbidden)

    for mutation in mutations:
        with pytest.raises(ProcessBrokerProtocolError):
            validate_runtime_infrastructure_invalid(mutation)


def test_compact_infrastructure_evidence_is_reconstructed_from_one_frozen_rule(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    package = campaign / "attempt"
    frozen = campaign / "frozen"
    sealed = package / "sealed"
    frozen.mkdir(parents=True)
    sealed.mkdir(parents=True)
    episode_id = "campaign:E1:task-1:repeat-0:seed-42"
    rule = {
        "operation": "step",
        "exception_type": "fixture.ControllerDisconnected",
        "reason_code": "BROWSER_CONTROLLER_DISCONNECTED",
        "failure_class": "fixture_controller_disconnected",
    }
    classifier = {
        "classifier_id": "fixture-classifier",
        "classifier_version": "v1",
        "rules_sha256": sha256_json([rule]),
        "rules": [rule],
    }
    environment = {
        "benchmark_version": "webarena-fixture-v1",
        "environment_adapter_id": "fixture-process-webarena",
        "environment_adapter_version": "v1",
        "infrastructure_classifier": classifier,
    }
    expected_diagnostic = {
        "classifier_id": classifier["classifier_id"],
        "classifier_version": classifier["classifier_version"],
        "classifier_rules_sha256": classifier["rules_sha256"],
        "benchmark_version": environment["benchmark_version"],
        "operation": rule["operation"],
        "episode_id": episode_id,
        "task_id": "task-1",
        "failure_class": rule["failure_class"],
        "exception_type": rule["exception_type"],
    }
    error = InfrastructureInvalidError(
        reason_code=rule["reason_code"],
        adapter_id=environment["environment_adapter_id"],
        adapter_version=environment["environment_adapter_version"],
        operation=rule["operation"],
        adapter_evidence={
            "adapter_event_id": f"{episode_id}:infrastructure:1",
            "failure_class": rule["failure_class"],
            "diagnostic_sha256": sha256_json(expected_diagnostic),
            "retryable": True,
        },
    )
    evidence = error.evidence_record(
        campaign_mode="evaluation",
        block_id="block-1",
        attempt_id=0,
        system_id="E1",
        episode_id=episode_id,
    )
    evidence_path = sealed / "infrastructure_invalid.json"
    atomic_write_json(campaign / "campaign_manifest.json", {"campaign_mode": "evaluation"})
    atomic_write_json(frozen / "environment.json", environment)
    atomic_write_json(evidence_path, evidence)
    schedule_row = {"block_id": "block-1", "task_id": "task-1"}
    status = {
        "episode_id": episode_id,
        "infrastructure_reason": rule["reason_code"],
        "infrastructure_evidence_sha256": sha256_file(evidence_path),
    }

    _validate_infrastructure_invalid_package(
        campaign,
        package,
        schedule_row=schedule_row,
        attempt_id=0,
        system_id="E1",
        status=status,
    )

    wrong_event = copy.deepcopy(evidence)
    wrong_event["adapter_evidence"]["adapter_event_id"] = (
        "another-episode:infrastructure:1"
    )
    wrong_event["adapter_evidence_sha256"] = sha256_json(
        wrong_event["adapter_evidence"]
    )
    atomic_write_json(evidence_path, wrong_event)
    status["infrastructure_evidence_sha256"] = sha256_file(evidence_path)
    with pytest.raises(SchemaError, match="not episode-bound"):
        _validate_infrastructure_invalid_package(
            campaign,
            package,
            schedule_row=schedule_row,
            attempt_id=0,
            system_id="E1",
            status=status,
        )

    atomic_write_json(evidence_path, evidence)
    status["infrastructure_evidence_sha256"] = sha256_file(evidence_path)
    duplicate_rule = {**rule, "exception_type": "fixture.AnotherDisconnect"}
    duplicate_environment = copy.deepcopy(environment)
    duplicate_environment["infrastructure_classifier"]["rules"] = [
        rule,
        duplicate_rule,
    ]
    atomic_write_json(frozen / "environment.json", duplicate_environment)
    with pytest.raises(SchemaError, match="one unique frozen rule"):
        _validate_infrastructure_invalid_package(
            campaign,
            package,
            schedule_row=schedule_row,
            attempt_id=0,
            system_id="E1",
            status=status,
        )
