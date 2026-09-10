from __future__ import annotations

from pathlib import Path

import pytest

from web_agent.eval.table2 import process_broker_episode_factory as episode_factory
from web_agent.eval.table2.common import sha256_file, sha256_json
from web_agent.eval.table2.live_deployment import ValidatedPC01LiveDeployment
from web_agent.eval.table2.process_broker import (
    PROCESS_BROKER_PILOT_EVALUATION_SCOPE,
    PROCESS_BROKER_RECEIPT_SCHEMA_VERSION,
    PROCESS_BROKER_SOURCE_PATHS,
)
from web_agent.eval.table2.process_broker_finalization import (
    ProcessIsolatedFinalizationClient,
)
from web_agent.eval.table2.process_broker_runtime import (
    ProcessIsolatedRuntimeClient,
)
from web_agent.eval.table2.process_broker_timeout import (
    MEASURED_TIMEOUT_MODE,
    ProcessBrokerTimeoutExpectedAuthority,
)
from web_agent.eval.table2.production_runner import (
    ProcessIsolatedWebArenaEpisodeBinding,
)
from web_agent.eval.table2.sealed_verifier import (
    SEALED_VERIFIER_STREAM_TARGET_SCHEMA_VERSION,
    SealedVerifierStreamTarget,
)
from web_agent.runtime.contracts import RuntimeStartState, TaskSpecification


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SOURCE = "src/web_agent/benchmarks/browsergym_webarena.py"
SEALED_SOURCE = "src/web_agent/eval/table2/webarena_page_state_evaluator.py"


def _task() -> TaskSpecification:
    start = RuntimeStartState(
        sites=("shopping",),
        start_url="https://shopping.example/start",
        require_login=False,
        storage_state=None,
        geolocation=None,
        require_reset=True,
    )
    return TaskSpecification(
        task_id="webarena-dev-102",
        goal="inspect the visible page state",
        benchmark_id="webarena",
        benchmark_version="0.14.3",
        start_state_id=start.start_state_sha256,
        site="shopping",
        start_url=start.start_url,
        development_partition=True,
        destructive_actions_allowed=False,
        metadata={
            "task_partition": "normal",
            "upstream_index": 102,
            "benchmark_task_id": "webarena.102",
            "source_content_sha256": "a" * 64,
        },
        runtime_start_state=start,
    )


def _target(tmp_path: Path, task: TaskSpecification) -> SealedVerifierStreamTarget:
    campaign = tmp_path / "campaign"
    campaign.mkdir(exist_ok=True)
    return SealedVerifierStreamTarget(
        schema_version=SEALED_VERIFIER_STREAM_TARGET_SCHEMA_VERSION,
        campaign_dir=str(campaign.resolve()),
        block_id="task_webarena-dev-102__seed_42__repeat_000",
        attempt_id=0,
        system_id="E2",
        episode_id="pilot:E2:webarena-dev-102:repeat-0:seed-42",
        matched_seed=42,
        task_id=task.task_id,
        repeat_id=0,
        seal_key_sha256="b" * 64,
        initial_event_count=0,
    )


def _timeout_authority(
    tmp_path: Path,
    *,
    browser_host_identity: dict | None = None,
) -> ProcessBrokerTimeoutExpectedAuthority:
    calibration = tmp_path / "timeout-calibration.json"
    calibration.write_text("{}\n", encoding="utf-8")
    probe = tmp_path / "safe-probes.json"
    harness = tmp_path / "harness.json"
    measurement = tmp_path / "measurement.json"
    audit = tmp_path / "audit.json"
    collection = tmp_path / "collection.json"
    for path in (probe, harness, measurement, audit, collection):
        path.write_text("{}\n", encoding="utf-8")
    return ProcessBrokerTimeoutExpectedAuthority(
        schema_version="test-timeout-authority-v1",
        record_type="ProcessBrokerTimeoutExpectedAuthority",
        claim_scope="TEST_NON_AUTHORIZING",
        paper_table_status="N/R",
        production_dispatch_authorized=False,
        calibration_artifact_path=calibration.resolve(),
        calibration_content_sha256="1" * 64,
        task_manifest_sha256="2" * 64,
        ordered_task_ids=("webarena-dev-102",),
        ordered_upstream_indices=(102,),
        deployment_topology="same_host",
        browser_host_identity=(
            {"host": "test"}
            if browser_host_identity is None
            else browser_host_identity
        ),
        browsergym_runtime_configuration={"headless": True},
        deployment_preflight_content_sha256="3" * 64,
        semantic_dependency_lock_sha256="4" * 64,
        service_url_map_content_sha256="5" * 64,
        campaign_budgets={"maximum_executor_requests": 30},
        safe_probe_manifest_path=probe.resolve(),
        safe_probe_manifest_content_sha256="6" * 64,
        measurement_harness_source_receipt_path=harness.resolve(),
        measurement_harness_source_receipt_content_sha256="7" * 64,
        measurement_harness_source_set_sha256="8" * 64,
        measurement_source_receipt_path=measurement.resolve(),
        measurement_source_receipt_content_sha256="9" * 64,
        non_persistence_audit_receipt_path=audit.resolve(),
        non_persistence_audit_receipt_content_sha256="a" * 64,
        collection_manifest_path=collection.resolve(),
        collection_manifest_content_sha256="b" * 64,
        external_cross_binding_present=False,
        live_campaign_authority=False,
    )


def _live_deployment(tmp_path: Path) -> ValidatedPC01LiveDeployment:
    package = tmp_path / "frozen" / "live_deployment"
    package.mkdir(parents=True)
    adapter_sha = sha256_file(ROOT / ADAPTER_SOURCE)
    sealed_sha = sha256_file(ROOT / SEALED_SOURCE)
    source_rows = [
        {"relative_path": ADAPTER_SOURCE, "sha256": adapter_sha},
        {"relative_path": SEALED_SOURCE, "sha256": sealed_sha},
    ]
    binding = {
        "schema_version": "table2-live-deployment-binding-v1",
        "package_root_relative_path": "live_deployment",
        "manifest_relative_path": "live_deployment/manifest.json",
        "manifest_sha256": "c" * 64,
        "manifest_content_sha256": "d" * 64,
        "package_files": [],
        "package_file_set_sha256": sha256_json([]),
        "capability_source_files": source_rows,
        "capability_source_set_sha256": sha256_json(source_rows),
    }
    manifest = {
        "capabilities": {
            "oracle_blind_browser_mapping": {
                "source_relative_path": ADAPTER_SOURCE,
                "source_sha256": adapter_sha,
            },
            "sealed_webarena_evaluator": {
                "source_relative_path": SEALED_SOURCE,
                "source_sha256": sealed_sha,
            },
        }
    }
    return ValidatedPC01LiveDeployment(
        binding=binding,
        manifest=manifest,
        evaluator_requirements={},
        package_root=package.resolve(),
        package_files=(),
        capability_source_files=(
            (ROOT / ADAPTER_SOURCE).resolve(),
            (ROOT / SEALED_SOURCE).resolve(),
        ),
    )


def _measured_binding() -> dict:
    return {
        "schema_version": "table2-process-broker-ipc-timeout-binding-v3",
        "mode": MEASURED_TIMEOUT_MODE,
        "calibration_schema_version": (
            "table2-process-broker-ipc-timeout-calibration-v3"
        ),
        "calibration_content_sha256": "d" * 64,
        "measurement_source_receipt_sha256": "e" * 64,
        "timeout_milliseconds": {"broker_startup": 1000},
        "measured_calibration_complete": True,
        "caller_selected_timeout": False,
        "live_campaign_authority": False,
    }


def _descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    authority: ProcessBrokerTimeoutExpectedAuthority | None = None,
    adapter_attribute: str = "create_environment_adapter",
    dependency_sources: tuple[episode_factory.SourceAttestedPython, ...] = (),
) -> episode_factory.ProcessBrokerLiveDeploymentDescriptor:
    live = _live_deployment(tmp_path)
    measured = _measured_binding()
    monkeypatch.setattr(
        episode_factory,
        "validate_pc01_live_deployment_binding",
        lambda value, **_: ValidatedPC01LiveDeployment(
            binding=value,
            manifest=live.manifest,
            evaluator_requirements=live.evaluator_requirements,
            package_root=live.package_root,
            package_files=live.package_files,
            capability_source_files=live.capability_source_files,
        ),
    )
    monkeypatch.setattr(
        episode_factory,
        "load_authority_bound_timeout_calibration",
        lambda value: {"authority": value.to_dict()},
    )
    monkeypatch.setattr(
        episode_factory,
        "measured_timeout_binding",
        lambda _value: dict(measured),
    )
    timeout = authority or _timeout_authority(tmp_path)
    return episode_factory.ProcessBrokerLiveDeploymentDescriptor(
        repository_root=ROOT,
        live_deployment=live,
        adapter_factory=episode_factory.SourceAttestedEntrypoint(
            source_relative_path=ADAPTER_SOURCE,
            source_sha256=sha256_file(ROOT / ADAPTER_SOURCE),
            entrypoint=(
                "web_agent.benchmarks.browsergym_webarena:"
                f"{adapter_attribute}"
            ),
        ),
        sealed_transition_callback=episode_factory.SourceAttestedEntrypoint(
            source_relative_path=SEALED_SOURCE,
            source_sha256=sha256_file(ROOT / SEALED_SOURCE),
            entrypoint=(
                "web_agent.eval.table2.webarena_page_state_evaluator:"
                "evaluate_environment_transition"
            ),
        ),
        sealed_finalizer=episode_factory.SourceAttestedEntrypoint(
            source_relative_path=SEALED_SOURCE,
            source_sha256=sha256_file(ROOT / SEALED_SOURCE),
            entrypoint=(
                "web_agent.eval.table2.webarena_page_state_evaluator:"
                "finalize_environment_episode"
            ),
        ),
        backend_dependency_sources=dependency_sources,
        timeout_calibration_artifact_path=Path(
            timeout.calibration_artifact_path
        ),
        timeout_expected_authority=timeout,
    )


class _FakeCleanupReceipt:
    def to_dict(self) -> dict:
        return {
            "schema_version": "table2-process-page-broker-cleanup-receipt-v1",
            "status": "LOCAL_CLEANUP_PASS",
        }


class _FakeReceipt:
    def __init__(self, source_files, timeout_sha256: str) -> None:
        self.schema_version = PROCESS_BROKER_RECEIPT_SCHEMA_VERSION
        self.execution_scope = PROCESS_BROKER_PILOT_EVALUATION_SCOPE
        self.source_files = tuple(source_files)
        self.immutable_timeout_authority_bundle_validated = True
        self.measured_ipc_timeout_calibration_complete = True
        self.ipc_timeout_binding_sha256 = timeout_sha256
        self.sealed_finalization_capability_available = True
        self.sealed_finalization_required = True
        self.child_owned_sealed_sink = True
        self.runtime_adapter_sealed_capability_free = True

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "execution_scope": self.execution_scope,
            "process_ids_distinct": True,
            "loaded_source_closure_enforced": True,
            "operation_specific_inner_schemas_registered": True,
            "single_episode_task_session_enforced": True,
            "verifier_receipt_causal_binding_enforced": True,
            "measured_ipc_timeout_calibration_complete": True,
            "timeout_calibration_replay_only": False,
            "immutable_timeout_authority_bundle_validated": True,
            "external_timeout_authority_cross_binding_present": False,
            "ipc_timeout_calibration_pilot_eligible": True,
            "sealed_finalization_capability_available": True,
            "sealed_finalization_required": True,
            "sealed_finalization_operation_in_runtime_allowlist": False,
            "child_owned_sealed_sink": True,
            "runtime_adapter_sealed_capability_free": True,
            "separate_evidence_transport_present": False,
            "runtime_terminal_returns_outer_sealed_signal": True,
            "sealed_finalization_returns_outer_sealed_signal": True,
            "sealed_finalization_timeout_measured": True,
            "runtime_value_provenance_attested": False,
            "external_deployment_authority": False,
        }


class _FakeBroker:
    instances: list["_FakeBroker"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.cleaned = False
        self.stops: list[bool] = []
        self.cleanup_receipt = _FakeCleanupReceipt()
        source_files = [
            {
                "relative_path": source,
                "sha256": sha256_file(ROOT / source),
            }
            for source in (
                *PROCESS_BROKER_SOURCE_PATHS,
                *kwargs["backend_dependency_source_relative_paths"],
            )
        ]
        timeout_sha = episode_factory.timeout_binding_sha256(
            _measured_binding()
        )
        self.receipt = _FakeReceipt(source_files, timeout_sha)
        self.__class__.instances.append(self)

    def start(self):
        return self.receipt

    def runtime_client(self):
        return object.__new__(ProcessIsolatedRuntimeClient)

    def finalization_client(self):
        return object.__new__(ProcessIsolatedFinalizationClient)

    def stop(self, *, force: bool = False) -> None:
        self.stops.append(force)
        self.cleaned = True


def test_factory_requires_prevalidated_deployment_and_exact_timeout_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = _live_deployment(tmp_path)
    with pytest.raises(
        episode_factory.ProcessBrokerEpisodeFactoryError,
        match="exact validated live-deployment type",
    ):
        episode_factory.ProcessBrokerLiveDeploymentDescriptor(
            repository_root=ROOT,
            live_deployment=object(),
            adapter_factory=object(),
            sealed_transition_callback=object(),
                sealed_finalizer=object(),
                backend_dependency_sources=(),
            timeout_calibration_artifact_path=tmp_path / "missing",
            timeout_expected_authority=object(),
        )

    # Reach the timeout type gate with otherwise valid, reopened deployment
    # bytes.  No fallback timeout or caller-selected duration is accepted.
    monkeypatch.setattr(
        episode_factory,
        "validate_pc01_live_deployment_binding",
        lambda value, **_: ValidatedPC01LiveDeployment(
            binding=value,
            manifest=live.manifest,
            evaluator_requirements={},
            package_root=live.package_root,
            package_files=(),
            capability_source_files=live.capability_source_files,
        ),
    )
    adapter = episode_factory.SourceAttestedEntrypoint(
        ADAPTER_SOURCE,
        sha256_file(ROOT / ADAPTER_SOURCE),
        "web_agent.benchmarks.browsergym_webarena:create_environment_adapter",
    )
    sealed = episode_factory.SourceAttestedEntrypoint(
        SEALED_SOURCE,
        sha256_file(ROOT / SEALED_SOURCE),
        "web_agent.eval.table2.webarena_page_state_evaluator:evaluate_environment_transition",
    )
    with pytest.raises(
        episode_factory.ProcessBrokerEpisodeFactoryError,
        match="exact measured-timeout authority type",
    ):
        episode_factory.ProcessBrokerLiveDeploymentDescriptor(
            repository_root=ROOT,
            live_deployment=live,
            adapter_factory=adapter,
            sealed_transition_callback=sealed,
            sealed_finalizer=episode_factory.SourceAttestedEntrypoint(
                SEALED_SOURCE,
                sha256_file(ROOT / SEALED_SOURCE),
                "web_agent.eval.table2.webarena_page_state_evaluator:finalize_environment_episode",
            ),
            backend_dependency_sources=(),
            timeout_calibration_artifact_path=tmp_path / "missing",
            timeout_expected_authority=object(),
        )


def test_factory_builds_only_explicit_pilot_process_capabilities_and_cleans_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor(tmp_path, monkeypatch)
    _FakeBroker.instances.clear()
    monkeypatch.setattr(episode_factory, "ProcessIsolatedBroker", _FakeBroker)
    task = _task()
    runtime_dir = tmp_path / "episode-runtime"
    runtime_dir.mkdir()

    binding = episode_factory.CredentialFreeProcessIsolatedWebArenaEpisodeFactory(
        descriptor
    )(task, runtime_dir.resolve(), _target(tmp_path, task))

    assert type(binding) is ProcessIsolatedWebArenaEpisodeBinding
    assert binding.benchmark_version == task.benchmark_version
    assert binding.oracle_labels_exposed_to_runtime is False
    broker = _FakeBroker.instances[-1]
    assert broker.kwargs["execution_scope"] == PROCESS_BROKER_PILOT_EVALUATION_SCOPE
    assert broker.kwargs["require_sealed_finalization"] is True
    assert broker.kwargs["policy_screenshot_root"] == (
        runtime_dir.resolve() / "screenshots"
    )
    assert "timeout_calibration_evidence" not in broker.kwargs
    assert broker.kwargs["timeout_calibration_artifact_path"] == (
        descriptor.timeout_calibration_artifact_path
    )
    config = broker.kwargs["sealed_backend_config"]
    assert set(config) == {
        "schema_version",
        "adapter_factory_entrypoint",
        "adapter_factory_source_relative_path",
        "adapter_factory_source_sha256",
        "sealed_transition_callback_entrypoint",
        "sealed_transition_callback_source_relative_path",
        "sealed_transition_callback_source_sha256",
        "sealed_finalizer_entrypoint",
        "sealed_finalizer_source_relative_path",
        "sealed_finalizer_source_sha256",
        "task_specification",
        "episode_runtime_dir",
        "sealed_stream_target",
    }
    assert not any("credential" in field.casefold() for field in config)
    assert descriptor.credential_material_embedded is False

    first_cleanup = binding.cleanup_episode()
    second_cleanup = binding.cleanup_episode()
    assert first_cleanup is second_cleanup
    assert dict(first_cleanup) == broker.cleanup_receipt.to_dict()
    assert broker.stops == [False]


def test_abort_is_identity_bound_and_idempotent_after_browser_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor(tmp_path, monkeypatch)
    _FakeBroker.instances.clear()
    monkeypatch.setattr(episode_factory, "ProcessIsolatedBroker", _FakeBroker)
    task = _task()
    runtime_dir = tmp_path / "episode-runtime"
    runtime_dir.mkdir()
    target = _target(tmp_path, task)
    binding = episode_factory.ProcessBrokerEpisodeFactory(descriptor)(
        task,
        runtime_dir.resolve(),
        target,
    )
    # A reset receipt is the parent-visible causal proof that browser reset ran.
    object.__setattr__(binding.environment_adapter, "_reset_receipt", object())

    receipt = binding.abort_episode(target.episode_id, task.task_id)
    repeated = binding.abort_episode(target.episode_id, task.task_id)
    assert receipt is repeated
    assert receipt.outcome == "BROKER_ABORTED"
    assert receipt.broker_abort_receipt_sha256 is not None
    assert _FakeBroker.instances[-1].stops == [False]
    cleanup = binding.cleanup_episode()
    assert dict(cleanup) == _FakeBroker.instances[-1].cleanup_receipt.to_dict()
    assert _FakeBroker.instances[-1].stops == [False]

    with pytest.raises(
        episode_factory.ProcessBrokerEpisodeFactoryError,
        match="another identity",
    ):
        binding.abort_episode("different-episode", task.task_id)


def test_mutated_nested_timeout_authority_fails_before_broker_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = {"host": "before"}
    authority = _timeout_authority(tmp_path, browser_host_identity=host)
    descriptor = _descriptor(tmp_path, monkeypatch, authority=authority)
    host["host"] = "after"
    _FakeBroker.instances.clear()
    monkeypatch.setattr(episode_factory, "ProcessIsolatedBroker", _FakeBroker)
    task = _task()
    runtime_dir = tmp_path / "episode-runtime"
    runtime_dir.mkdir()

    with pytest.raises(
        episode_factory.ProcessBrokerEpisodeFactoryError,
        match="authority changed",
    ):
        episode_factory.ProcessBrokerEpisodeFactory(descriptor)(
            task,
            runtime_dir.resolve(),
            _target(tmp_path, task),
        )
    assert _FakeBroker.instances == []
