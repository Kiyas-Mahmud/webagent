from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
import hashlib
from pathlib import Path

import pytest

from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymRuntimeConfiguration,
    BrowserGymTaskStateResetReceipt,
    BrowserGymWebArenaRuntimeFactory,
    FrozenBrowserGymTaskStateResetter,
    callable_source_sha256,
    module_source_sha256,
)
from web_agent.benchmarks.webarena import (
    FrozenWebArenaActionSafetyPolicy,
    FrozenWebArenaInfrastructureFaultClassifier,
    FrozenWebArenaManualRescueGuard,
    WebArenaActionSafetyDecision,
    WebArenaInfrastructureClassification,
    WebArenaInfrastructureRule,
    WebArenaManualRescueCheck,
    WebArenaManualRescueReceipt,
)
from web_agent.eval.table2.common import sha256_file, sha256_json
from web_agent.eval.table2.pc01_artifacts import (
    PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_MODEL_REVISION,
)
from web_agent.eval.table2.production_runner import (
    EvaluationRuntimeBinding,
    FrozenRuntimeContext,
    RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION,
    RUNTIME_LIVE_CAPABILITY_IDS,
    SeedRuntimeBinding,
    validate_runtime_capability_authority,
)
from web_agent.eval.table2.sealed_page_broker import (
    create_one_way_sealed_page_broker,
)
from web_agent.eval.table2.live_page_broker_assembly import (
    process_runtime_page_publisher,
)
from web_agent.eval.table2.webarena_preflight_binding import (
    ValidatedDeploymentPreflight,
)
from web_agent.runtime import pc01_live_integration as live
from web_agent.runtime.checkpoint_inference import (
    LoadedSelectedBackboneBackend,
    LoadedSelectedCheckpointBackend,
)
from web_agent.runtime.contracts import (
    RuntimeStartState,
    SystemID,
    TaskSpecification,
    canonical_sha256,
)
from web_agent.runtime.observation import CausalBoundaryError, ProcessorParityContract
from web_agent.runtime.qwen2vl_pc01 import parse_e0_action_output
from web_agent.runtime.recovery.controller import CallableRecoveryActionPlanner
from web_agent.runtime.state_reset import (
    REGISTERED_STATEFUL_BACKEND_ROLES,
    REGISTERED_WEBARENA_RESET_COMMITMENTS,
    CallableEpisodeStateResetter,
    EpisodeStateResetRequest,
    FrozenWebArenaResetStateAttester,
    WebArenaResetStateRequest,
    evidence_from_backend_digests,
    webarena_reset_receipt_from_commitments,
)


SERVICE_URLS = {
    "WA_SHOPPING": "http://shopping.test",
    "WA_SHOPPING_ADMIN": "http://shopping-admin.test",
    "WA_REDDIT": "http://reddit.test",
    "WA_GITLAB": "http://gitlab.test",
    "WA_WIKIPEDIA": "http://wikipedia.test",
    "WA_MAP": "http://map.test",
    "WA_HOMEPAGE": "http://homepage.test",
}


@pytest.fixture(autouse=True)
def _isolated_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_OPERATIONS_REGISTRY", live._ImmutableOperationsRegistry())


def _test_source_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _processor() -> ProcessorParityContract:
    return ProcessorParityContract(
        processor_class="MeasuredProcessor",
        processor_revision="revision-1",
        processor_config_sha256="a" * 64,
        pre_action_field_mapping={"image": "pixel_values"},
        post_action_field_mapping={
            "before_image": "before_pixel_values",
            "after_image": "after_pixel_values",
        },
    )


def _task_state_reset(
    task_id: str,
    start_state: RuntimeStartState,
    seed: int,
) -> BrowserGymTaskStateResetReceipt:
    return BrowserGymTaskStateResetReceipt(
        resetter_id="measured-task-state-reset",
        resetter_version="v1",
        resetter_source_sha256=callable_source_sha256(_task_state_reset),
        task_id=task_id,
        start_state_sha256=start_state.start_state_sha256,
        reset_seed=seed,
        require_reset=start_state.require_reset,
        reset_performed=start_state.require_reset,
        service_url_map_sha256=canonical_sha256(SERVICE_URLS),
        service_state_commitments={
            site: canonical_sha256({"site": site, "seed": seed})
            for site in start_state.sites
        },
    )


def _webarena_reset(request: WebArenaResetStateRequest):
    return webarena_reset_receipt_from_commitments(
        request,
        attester_id="measured-webarena-reset",
        attester_version="v1",
        attester_source_sha256=_test_source_sha256(),
        commitments_sha256={
            role: canonical_sha256(
                {"role": role, "seed": request.reset_stage_seed}
            )
            for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
        },
    )


def _action_safety(task, observation, action):
    del task, observation
    return WebArenaActionSafetyDecision(
        safety_policy_id="measured-action-safety",
        safety_policy_version="v1",
        action_id=action.action_id,
        allowed=True,
        reason_code="ALLOW_REGISTERED_ACTION",
    )


def _manual_rescue(environment, check: WebArenaManualRescueCheck):
    del environment
    return WebArenaManualRescueReceipt(
        guard_id="measured-exclusive-input",
        guard_version="v1",
        source_sha256=_test_source_sha256(),
        evidence_mode="exclusive_controller_input_audit_v1",
        check_sha256=check.record_sha256,
        check_index=check.check_index,
        stage=check.stage,
        expected_registered_browser_steps=check.expected_registered_browser_steps,
        observed_registered_browser_steps=check.expected_registered_browser_steps,
        automation_session_sha256="b" * 64,
        controller_input_audit_sha256=canonical_sha256(check.to_dict()),
        exclusive_automation_control=True,
        non_agent_input_event_count=0,
        manual_rescue_detected=False,
    )


def _classify(error: Exception, operation: str):
    if isinstance(error, ConnectionError) and operation == "environment_factory":
        return WebArenaInfrastructureClassification(
            reason_code="BENCHMARK_SERVICE_UNAVAILABLE",
            failure_class="measured_service_connection_failure",
        )
    return None


def _recovery_plan(task, observation, decision, failed_action, rng):
    del task, observation, decision, rng
    return failed_action


def _begin_measurement(task, system):
    return {"task_id": task.task_id, "system": system.value}


def _alternate_begin_measurement(task, system):
    return {"system": system.value, "task_id": task.task_id}


_PROVIDER_TO_MUTATE_DURING_MEASUREMENT = None


def _mutating_begin_measurement(task, system):
    del task, system
    provider = _PROVIDER_TO_MUTATE_DURING_MEASUREMENT
    assert provider is not None
    provider.browser_runtime_factory.validated_preflight.binding["drift"] = True
    return {"mutated": True}


def _finish_measurement(token, summary):
    del token, summary
    return {
        "model_call_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "model_parameter_count": 1,
        "trainable_parameter_count": 0,
        "peak_gpu_memory_mb": 0.0,
        "peak_system_memory_mb": 0.0,
        "training_gpu_hours": 0.0,
    }


def _source_rows() -> list[dict[str, str]]:
    repository = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repository / "src/web_agent/benchmarks/browsergym_webarena.py",
    )
    return [
        {
            "relative_path": path.relative_to(repository).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _runtime_capability_rows(
    provider: live.PC01LiveOperationsProvider,
) -> dict[str, dict]:
    contracts = provider.capability_public_contracts()
    return {
        capability_id: {
            "capability_id": capability_id,
            "implementation_id": contracts[capability_id]["implementation_id"],
            "implementation_version": contracts[capability_id][
                "implementation_version"
            ],
            "source_relative_path": contracts[capability_id][
                "source_relative_path"
            ],
            "source_sha256": contracts[capability_id]["source_sha256"],
            "deployment_state_sha256": contracts[capability_id][
                "public_state_sha256"
            ],
            "readiness_evidence_sha256": f"{index + 7:x}" * 64,
            "runtime_return_contract": "measured-runtime-output-only",
        }
        for index, capability_id in enumerate(RUNTIME_LIVE_CAPABILITY_IDS)
    }


def _runtime_capability_authority(
    rows: dict[str, dict],
    preflight_binding: dict,
    provider_public_contract_sha256: str,
) -> dict:
    ordered = [rows[item] for item in RUNTIME_LIVE_CAPABILITY_IDS]
    return validate_runtime_capability_authority(
        {
            "schema_version": RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION,
            "deployment_preflight_binding_sha256": sha256_json(preflight_binding),
            "expected_provider_public_contract_sha256": (
                provider_public_contract_sha256
            ),
            "capabilities": rows,
            "capability_set_sha256": sha256_json(ordered),
        }
    )


def _model_manifest() -> dict:
    processor = _processor()
    return {
        "model_seed": 42,
        "selected_checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "resolved_config_sha256": "2" * 64,
        "resolved_config_record_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "processor_contract_sha256": processor.record_sha256,
        "model_evidence_bundle_sha256": "4" * 64,
        "e0_backbone_id": "Qwen/Qwen2-VL-2B-Instruct",
        "e0_backbone_revision": PC01_MODEL_REVISION,
        "e0_backbone_sha256": PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
        "e0_resolved_config_sha256": "6" * 64,
        "e0_processor_contract_sha256": processor.record_sha256,
        "e0_base_prompt_sha256": "7" * 64,
        "e0_parser_id": "strict-e0-json-action-parser",
        "e0_parser_version": "v1",
        "e0_parser_sha256": "8" * 64,
    }


def _runtime_identity(manifest: dict, *, checkpoint_systems=None) -> dict:
    return {
        "checkpoint_systems": (
            ["E1", "E2", "E3"]
            if checkpoint_systems is None
            else checkpoint_systems
        ),
        "selected_checkpoint_by_seed": {
            "42": {
                key: manifest[key]
                for key in (
                    "selected_checkpoint_sha256",
                    "resolved_config_sha256",
                    "resolved_config_record_sha256",
                    "processor_contract_sha256",
                    "model_evidence_bundle_sha256",
                )
            }
        },
        "e0_unadapted_backbone_by_seed": {
            "42": {
                "backbone_id": manifest["e0_backbone_id"],
                "backbone_revision": manifest["e0_backbone_revision"],
                "backbone_sha256": manifest["e0_backbone_sha256"],
                "resolved_config_sha256": manifest["e0_resolved_config_sha256"],
                "processor_contract_sha256": manifest[
                    "e0_processor_contract_sha256"
                ],
                "base_prompt_sha256": manifest["e0_base_prompt_sha256"],
                "parser_id": manifest["e0_parser_id"],
                "parser_version": manifest["e0_parser_version"],
                "parser_sha256": manifest["e0_parser_sha256"],
            }
        },
        "environment": {"benchmark": "webarena", "benchmark_version": "0.14.3"},
        "runtime_integration": {
            "entrypoint": live.PC01_LIVE_INTEGRATION_ENTRYPOINT,
            "source_relative_path": live.PC01_LIVE_INTEGRATION_SOURCE,
            "source_sha256": live.integration_source_sha256(),
        },
    }


def _context(
    tmp_path: Path,
    runtime_identity: dict,
    manifest: dict,
    runtime_capability_authority: dict,
):
    manifest_path = tmp_path / "seed_42.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return FrozenRuntimeContext(
        model_manifest_paths={42: manifest_path},
        model_payload_paths={42: {}},
        model_evidence_paths={42: {}},
        runtime_identity=runtime_identity,
        runtime_capability_authority=runtime_capability_authority,
    )


def _external_episode_resetter(state: dict) -> CallableEpisodeStateResetter:
    source_sha256 = _test_source_sha256()

    def reset(request: EpisodeStateResetRequest):
        state["digests"] = {
            role: canonical_sha256(
                {"role": role, "seed": request.reset_stage_seed, "state": "clean"}
            )
            for role in REGISTERED_STATEFUL_BACKEND_ROLES
        }
        return evidence_from_backend_digests(
            request,
            resetter_id="measured-model-state-reset",
            resetter_version="v1",
            resetter_source_sha256=source_sha256,
            backend_state_sha256=state["digests"],
        )

    def digest():
        return dict(state["digests"])

    return CallableEpisodeStateResetter(
        resetter_id="measured-model-state-reset",
        resetter_version="v1",
        source_sha256=source_sha256,
        callback=reset,
        digest_callback=digest,
    )


def _browser_factory(tmp_path: Path):
    source_sha256 = _test_source_sha256()
    preflight = ValidatedDeploymentPreflight(
        binding={"deployment": "measured-single-host"},
        evidence={"status": "PASS"},
        service_url_map=dict(SERVICE_URLS),
        evidence_path=tmp_path / "preflight.json",
        service_url_map_path=tmp_path / "urls.json",
    )
    task_resetter = FrozenBrowserGymTaskStateResetter(
        resetter_id="measured-task-state-reset",
        resetter_version="v1",
        source_sha256=source_sha256,
        service_url_map_sha256=canonical_sha256(SERVICE_URLS),
        callback=_task_state_reset,
    )
    raw_reset = FrozenWebArenaResetStateAttester(
        attester_id="measured-webarena-reset",
        attester_version="v1",
        source_sha256=source_sha256,
        callback=_webarena_reset,
    )
    raw_safety = FrozenWebArenaActionSafetyPolicy(
        safety_policy_id="measured-action-safety",
        safety_policy_version="v1",
        destructive_action_policy="deny-destructive-actions-v1",
        benchmark_version="0.14.3",
        source_sha256=source_sha256,
        callback=_action_safety,
    )
    raw_manual = FrozenWebArenaManualRescueGuard(
        guard_id="measured-exclusive-input",
        guard_version="v1",
        benchmark_version="0.14.3",
        evidence_mode="exclusive_controller_input_audit_v1",
        source_sha256=source_sha256,
        callback=_manual_rescue,
    )
    classifier = FrozenWebArenaInfrastructureFaultClassifier(
        classifier_id="measured-infrastructure-taxonomy",
        classifier_version="v1",
        benchmark_version="0.14.3",
        callback=_classify,
        rules=(
            WebArenaInfrastructureRule(
                operation="environment_factory",
                exception_type="builtins.ConnectionError",
                reason_code="BENCHMARK_SERVICE_UNAVAILABLE",
                failure_class="measured_service_connection_failure",
            ),
        ),
    )
    factory = BrowserGymWebArenaRuntimeFactory(
        validated_preflight=preflight,
        credential_bundle_root=tmp_path,
        task_state_resetter=task_resetter,
        runtime_page_publisher=process_runtime_page_publisher(),
        configuration=BrowserGymRuntimeConfiguration(),
        benchmark_version="0.14.3",
        environment_adapter_id="browsergym-webarena-validation-disabled",
        environment_adapter_version="v1",
        environment_state_digester_id="browsergym-page-state",
        environment_state_digester_version="v1",
        page_settle_policy_id="browsergym-network-idle-v1",
        infrastructure_fault_classifier=classifier,
        reset_state_attester=live.bind_webarena_reset_state_attester(raw_reset),
        action_safety_policy=live.bind_action_safety_policy(raw_safety),
        manual_rescue_guard=live.bind_manual_rescue_guard(raw_manual),
        wrapper_source_sha256=module_source_sha256(),
    )
    return factory, task_resetter


def _provider_and_context(
    tmp_path: Path,
    *,
    checkpoint_systems=None,
):
    manifest = _model_manifest()
    identity = _runtime_identity(manifest, checkpoint_systems=checkpoint_systems)
    state = {
        "digests": {
            role: canonical_sha256({"role": role, "state": "uninitialized"})
            for role in REGISTERED_STATEFUL_BACKEND_ROLES
        }
    }
    external_resetter = _external_episode_resetter(state)
    integration_resetter = live.bind_episode_state_resetter(external_resetter)
    browser_factory, task_resetter = _browser_factory(tmp_path)
    recovery = CallableRecoveryActionPlanner(
        planner_id="measured-recovery-planner",
        planner_version="v1",
        callback=_recovery_plan,
    )
    provider = live.PC01LiveOperationsProvider(
        provider_id="pc01-dgx-live-operations",
        provider_version="v1",
        expected_runtime_identity_sha256=sha256_json(identity),
        efficiency_measurement_id="measured-efficiency_measurement",
        efficiency_measurement_version="v1",
        credential_capability=live.PC01ExternalCredentialCapability(
            capability_id="measured-browser-credentials",
            capability_version="v1",
            root=tmp_path,
        ),
        browser_runtime_factory=browser_factory,
        task_state_resetter=task_resetter,
        episode_state_resetter=integration_resetter,
        recovery_action_planner=recovery,
        begin_measurement=_begin_measurement,
        finish_measurement=_finish_measurement,
        parameter_fallback_contract=live.PC01_PARAMETER_FALLBACK_CONTRACT,
        parameter_fallback_source_sha256=sha256_file(
            Path(live.__file__).with_name("qwen2vl_pc01.py")
        ),
    )
    context = _context(
        tmp_path,
        identity,
        manifest,
        _runtime_capability_authority(
            _runtime_capability_rows(provider),
            dict(browser_factory.validated_preflight.binding),
            provider.public_contract_sha256,
        ),
    )
    return provider, context, state


def _installation_receipt(
    provider: live.PC01LiveOperationsProvider,
) -> live.PC01ProviderInstallationReceipt:
    relative = Path(__file__).resolve().relative_to(
        Path(__file__).resolve().parents[2]
    ).as_posix()
    return live.PC01ProviderInstallationReceipt(
        schema_version="table2-pc01-provider-installation-receipt-v1",
        record_type="PC01ProviderInstallationReceipt",
        status="PASS",
        claim_scope="REVIEWED_CODE_ORACLE_FREE_DATAFLOW_ONLY",
        same_process_factory=True,
        kernel_filesystem_sandbox=False,
        factory_entrypoint=(
            "tests.table2.test_pc01_live_integration:_begin_measurement"
        ),
        factory_module="tests.table2.test_pc01_live_integration",
        factory_qualname="_begin_measurement",
        factory_source_relative_path=relative,
        factory_source_sha256=_test_source_sha256(),
        bootstrap_context_sha256="1" * 64,
        pre_factory_campaign_state_sha256="2" * 64,
        post_factory_campaign_state_sha256="2" * 64,
        provider_boundary_receipt_sha256="3" * 64,
        credential_public_identity_sha256="4" * 64,
        expected_provider_public_contract_sha256=provider.public_contract_sha256,
        actual_provider_public_contract_sha256=provider.public_contract_sha256,
    )


def _register(provider: live.PC01LiveOperationsProvider) -> str:
    return live.register_pc01_live_operations(
        provider,
        _installation_receipt(provider),
    )


class _FakePC01Factory:
    def __init__(
        self,
        *,
        create_webarena_runtime,
        seed_services,
        create_process_isolated_webarena=None,
    ):
        self.create_webarena_runtime = create_webarena_runtime
        self.seed_services = seed_services
        self.create_process_isolated_webarena = (
            create_process_isolated_webarena
        )

    def __call__(self, context):
        manifest = _model_manifest()
        processor = _processor()
        service = self.seed_services[42]
        selected = LoadedSelectedCheckpointBackend(
            checkpoint_sha256=manifest["selected_checkpoint_sha256"],
            resolved_config_sha256=manifest["resolved_config_sha256"],
            processor_contract=processor,
            action_predictor=parse_e0_action_output,
            transition_predictor=parse_e0_action_output,
            recovery_predictor=parse_e0_action_output,
            memory_embedding=parse_e0_action_output,
        )
        base = LoadedSelectedBackboneBackend(
            backbone_id=manifest["e0_backbone_id"],
            backbone_revision=manifest["e0_backbone_revision"],
            backbone_sha256=manifest["e0_backbone_sha256"],
            resolved_config_sha256=manifest["e0_resolved_config_sha256"],
            processor_contract=processor,
            base_prompt_sha256=manifest["e0_base_prompt_sha256"],
            parser_id=manifest["e0_parser_id"],
            parser_version=manifest["e0_parser_version"],
            action_predictor=parse_e0_action_output,
            frozen=True,
            training=False,
            adaptation_loaded=False,
            task_heads_loaded=False,
        )
        seed_binding = SeedRuntimeBinding(
            model_seed=42,
            episode_state_resetter=service.episode_state_resetter,
            selected_checkpoint_backend=selected,
            selected_backbone_backend=base,
            e0_parser_sha256=manifest["e0_parser_sha256"],
            parameter_fallback_resolver=parse_e0_action_output,
            parameter_fallback_backbone_sha256=manifest["e0_backbone_sha256"],
            recovery_action_planner=service.recovery_action_planner,
            begin_measurement=service.begin_measurement,
            finish_measurement=service.finish_measurement,
        )
        return EvaluationRuntimeBinding(
            runtime_identity=dict(context.runtime_identity),
            seed_bindings={42: seed_binding},
            create_webarena_runtime=self.create_webarena_runtime,
            create_process_isolated_webarena=(
                self.create_process_isolated_webarena
            ),
        )


def _install_and_create(monkeypatch, provider, context):
    monkeypatch.setattr(live, "PC01EvaluationRuntimeFactory", _FakePC01Factory)
    _register(provider)
    return live.create_pc01_live_runtime(context)


def test_missing_provider_fails_before_context_or_model_use() -> None:
    with pytest.raises(live.PC01LiveIntegrationError, match="not registered"):
        live.create_pc01_live_runtime(object())


def test_process_isolated_episode_factory_threads_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    process_factory = object.__new__(
        live.CredentialFreeProcessIsolatedWebArenaEpisodeFactory
    )
    provider = replace(
        provider,
        create_process_isolated_webarena=process_factory,
    )

    binding = _install_and_create(monkeypatch, provider, context)

    assert binding.create_process_isolated_webarena is process_factory


def test_provider_rejects_untyped_process_episode_factory(tmp_path: Path) -> None:
    provider, _, _ = _provider_and_context(tmp_path)
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match="exact credential-free source-attested implementation",
    ):
        replace(provider, create_process_isolated_webarena=lambda *_: None)


def test_provider_registry_is_one_install_and_identity_is_reproducible(
    tmp_path: Path,
) -> None:
    provider, _, _ = _provider_and_context(tmp_path)
    with pytest.raises(live.PC01LiveIntegrationError, match="post-installation receipt"):
        live.register_pc01_live_operations(provider)
    first = _register(provider)
    assert first == provider.contract_sha256
    assert live._OPERATIONS_REGISTRY.require() is provider
    assert provider.contract_sha256 == first
    with pytest.raises(live.PC01LiveIntegrationError, match="already registered"):
        _register(provider)


def test_provider_rejects_disconnected_sealed_page_broker(
    tmp_path: Path,
) -> None:
    provider, _, _ = _provider_and_context(tmp_path)
    disconnected, _ = create_one_way_sealed_page_broker()
    changed_factory = replace(
        provider.browser_runtime_factory,
        runtime_page_publisher=disconnected,
    )
    with pytest.raises(live.PC01LiveIntegrationError, match="disconnected"):
        replace(provider, browser_runtime_factory=changed_factory)


def test_registry_detects_nested_provider_mutation(tmp_path: Path) -> None:
    provider, _, _ = _provider_and_context(tmp_path)
    _register(provider)
    provider.browser_runtime_factory.validated_preflight.binding["drift"] = True
    with pytest.raises(live.PC01LiveIntegrationError, match="changed after installation"):
        live._OPERATIONS_REGISTRY.require()


def test_runtime_identity_mismatch_fails_before_checkpoint_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    _register(provider)
    changed = dict(context.runtime_identity)
    changed["unregistered_drift"] = True
    changed_context = FrozenRuntimeContext(
        model_manifest_paths=context.model_manifest_paths,
        model_payload_paths=context.model_payload_paths,
        model_evidence_paths=context.model_evidence_paths,
        runtime_identity=changed,
        runtime_capability_authority=context.runtime_capability_authority,
    )
    called = False

    class MustNotLoad:
        def __init__(self, **kwargs):
            del kwargs
            nonlocal called
            called = True

    monkeypatch.setattr(live, "PC01EvaluationRuntimeFactory", MustNotLoad)
    with pytest.raises(live.PC01LiveIntegrationError, match="runtime identity differs"):
        live.create_pc01_live_runtime(changed_context)
    assert called is False


def test_oracle_fields_are_rejected_before_browser_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    binding = _install_and_create(monkeypatch, provider, context)
    start = RuntimeStartState(
        sites=("gitlab",),
        start_url="http://gitlab.test/project",
        require_login=True,
        storage_state=".auth/gitlab.json",
        geolocation=None,
        require_reset=False,
    )
    task = TaskSpecification(
        task_id="webarena.44",
        goal="Open the project",
        benchmark_id="webarena",
        benchmark_version="0.14.3",
        start_state_id=start.start_state_sha256,
        site="gitlab",
        start_url=start.start_url,
        metadata={"oracle_success": True},
        runtime_start_state=start,
    )
    with pytest.raises(CausalBoundaryError, match="forbidden sealed/future field"):
        binding.create_webarena_runtime(task)


def test_provider_schema_has_no_sealed_evaluator_or_reward_payload() -> None:
    names = {item.name for item in fields(live.PC01LiveOperationsProvider)}
    assert not {
        "sealed_evaluator",
        "verifier",
        "reward",
        "oracle_labels",
        "reference_trajectory",
    } & names
    assert {
        "credential_capability",
        "task_state_resetter",
        "browser_runtime_factory",
        "episode_state_resetter",
        "recovery_action_planner",
        "efficiency_measurement_id",
        "efficiency_measurement_version",
        "begin_measurement",
        "finish_measurement",
        "parameter_fallback_contract",
    } <= names


def test_runtime_context_rejects_sealed_evaluator_metadata(tmp_path: Path) -> None:
    _, context, _ = _provider_and_context(tmp_path)
    with pytest.raises(ValueError, match="exclude evaluator"):
        FrozenRuntimeContext(
            model_manifest_paths=context.model_manifest_paths,
            model_payload_paths=context.model_payload_paths,
            model_evidence_paths=context.model_evidence_paths,
            runtime_identity={
                **dict(context.runtime_identity),
                "evaluator": {"evaluator_id": "must-not-cross"},
            },
            runtime_capability_authority=context.runtime_capability_authority,
        )


def test_provider_capability_state_drift_fails_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    _register(provider)
    capabilities = {
        key: dict(value)
        for key, value in context.runtime_capability_authority["capabilities"].items()
    }
    capabilities["recovery_action_planner"]["deployment_state_sha256"] = "f" * 64
    ordered = [capabilities[item] for item in RUNTIME_LIVE_CAPABILITY_IDS]
    changed_authority = dict(context.runtime_capability_authority)
    changed_authority["capabilities"] = capabilities
    changed_authority["capability_set_sha256"] = sha256_json(ordered)
    changed_context = FrozenRuntimeContext(
        model_manifest_paths=context.model_manifest_paths,
        model_payload_paths=context.model_payload_paths,
        model_evidence_paths=context.model_evidence_paths,
        runtime_identity=context.runtime_identity,
        runtime_capability_authority=changed_authority,
    )
    called = False

    class MustNotLoad:
        def __init__(self, **kwargs):
            del kwargs
            nonlocal called
            called = True

    monkeypatch.setattr(live, "PC01EvaluationRuntimeFactory", MustNotLoad)
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match="capability public contract differs",
    ):
        live.create_pc01_live_runtime(changed_context)
    assert called is False


def test_provider_preflight_drift_fails_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    _register(provider)
    changed_preflight = {"deployment": "different-host"}
    changed_authority = dict(context.runtime_capability_authority)
    changed_authority["deployment_preflight_binding_sha256"] = sha256_json(
        changed_preflight
    )
    changed_context = FrozenRuntimeContext(
        model_manifest_paths=context.model_manifest_paths,
        model_payload_paths=context.model_payload_paths,
        model_evidence_paths=context.model_evidence_paths,
        runtime_identity=context.runtime_identity,
        runtime_capability_authority=changed_authority,
    )
    called = False

    class MustNotLoad:
        def __init__(self, **kwargs):
            del kwargs
            nonlocal called
            called = True

    monkeypatch.setattr(live, "PC01EvaluationRuntimeFactory", MustNotLoad)
    with pytest.raises(live.PC01LiveIntegrationError, match="preflight differs"):
        live.create_pc01_live_runtime(changed_context)
    assert called is False


def test_efficiency_callbacks_require_frozen_measurement_identity(
    tmp_path: Path,
) -> None:
    provider, _, _ = _provider_and_context(tmp_path)
    _, context, _ = _provider_and_context(tmp_path)
    changed = replace(provider, efficiency_measurement_version="v2")
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match="efficiency_measurement.implementation_version",
    ):
        live.validate_provider_public_contract(
            changed,
            context.runtime_capability_authority,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "oracle_labels_exposed_to_runtime",
        "reward_slots_exposed_to_runtime",
        "sealed_evaluator_capability_present",
    ),
)
def test_provider_rejects_any_oracle_reward_or_evaluator_capability(
    tmp_path: Path,
    field_name: str,
) -> None:
    provider, _, _ = _provider_and_context(tmp_path)
    with pytest.raises(live.PC01LiveIntegrationError, match="no oracle/reward/evaluator"):
        replace(provider, **{field_name: True})


def test_e0_is_unadapted_and_e1_e3_share_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    binding = _install_and_create(monkeypatch, provider, context)
    seed = binding.seed_bindings[42]
    assert seed.selected_backbone_backend.adaptation_loaded is False
    assert seed.selected_backbone_backend.task_heads_loaded is False
    assert seed.selected_backbone_backend.training is False
    assert (
        seed.selected_checkpoint_backend.checkpoint_sha256
        == PC01_EXPECTED_CHECKPOINT_SHA256
    )
    assert binding.runtime_identity["checkpoint_systems"] == ["E1", "E2", "E3"]
    assert "E0" not in binding.runtime_identity["checkpoint_systems"]
    assert (
        seed.parameter_fallback_backbone_sha256
        == PC01_EXPECTED_BASE_SNAPSHOT_SHA256
    )


def test_checkpoint_system_drift_is_rejected_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(
        tmp_path,
        checkpoint_systems=["E0", "E1", "E2", "E3"],
    )
    _register(provider)
    monkeypatch.setattr(live, "PC01EvaluationRuntimeFactory", _FakePC01Factory)
    with pytest.raises(live.PC01LiveIntegrationError, match="E1, E2, and E3 exactly"):
        live.create_pc01_live_runtime(context)


def test_reset_wrapper_discards_contamination_and_is_system_independent(
    tmp_path: Path,
) -> None:
    provider, _, state = _provider_and_context(tmp_path)
    resetter = provider.episode_state_resetter
    request_e0 = EpisodeStateResetRequest(
        episode_id="e0-episode",
        system_id=SystemID.E0,
        reset_stage_seed=991,
    )
    evidence_e0 = resetter.reset(request_e0)
    state["digests"] = {
        role: hashlib.sha256(f"contaminated:{role}".encode()).hexdigest()
        for role in REGISTERED_STATEFUL_BACKEND_ROLES
    }
    request_e3 = EpisodeStateResetRequest(
        episode_id="e3-episode",
        system_id=SystemID.E3,
        reset_stage_seed=991,
    )
    evidence_e3 = resetter.reset(request_e3)
    assert evidence_e0.backend_state_sha256 == evidence_e3.backend_state_sha256
    assert evidence_e0.initial_state_sha256 == evidence_e3.initial_state_sha256
    assert evidence_e0.resetter_source_sha256 == live.integration_source_sha256()
    assert evidence_e3.resetter_source_sha256 == live.integration_source_sha256()


def test_repeated_entrypoint_calls_are_deterministic_and_do_not_replace_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    monkeypatch.setattr(live, "PC01EvaluationRuntimeFactory", _FakePC01Factory)
    fingerprint = _register(provider)
    first = live.create_pc01_live_runtime(context)
    second = live.create_pc01_live_runtime(context)
    assert first.runtime_identity == second.runtime_identity == context.runtime_identity
    assert first.seed_bindings[42].selected_checkpoint_backend.checkpoint_sha256 == (
        second.seed_bindings[42].selected_checkpoint_backend.checkpoint_sha256
    )
    assert first.seed_bindings[42].selected_backbone_backend.backbone_sha256 == (
        second.seed_bindings[42].selected_backbone_backend.backbone_sha256
    )
    assert live._OPERATIONS_REGISTRY.require() is provider
    assert provider.contract_sha256 == fingerprint


def test_provider_capability_contract_is_derived_not_declared(tmp_path: Path) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    contracts = provider.capability_public_contracts()
    assert tuple(contracts) == RUNTIME_LIVE_CAPABILITY_IDS
    assert contracts["recovery_action_planner"]["implementation_id"] == (
        provider.recovery_action_planner.planner_id
    )
    assert contracts["efficiency_measurement"]["implementation_id"] == (
        provider.efficiency_measurement_id
    )
    arbitrary = {
        key: dict(value)
        for key, value in context.runtime_capability_authority["capabilities"].items()
    }
    arbitrary["recovery_action_planner"]["deployment_state_sha256"] = "9" * 64
    authority = dict(context.runtime_capability_authority)
    authority["capabilities"] = arbitrary
    authority["capability_set_sha256"] = sha256_json(
        [arbitrary[item] for item in RUNTIME_LIVE_CAPABILITY_IDS]
    )
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match="recovery_action_planner.deployment_state_sha256",
    ):
        live.validate_provider_public_contract(provider, authority)


@pytest.mark.parametrize(
    ("capability_id", "mutation"),
    (
        (
            "deterministic_reset",
            lambda provider: replace(
                provider,
                episode_state_resetter=replace(
                    provider.episode_state_resetter,
                    resetter_id="different-resetter",
                ),
            ),
        ),
        (
            "exclusive_input_audit",
            lambda provider: replace(
                provider,
                browser_runtime_factory=replace(
                    provider.browser_runtime_factory,
                    manual_rescue_guard=replace(
                        provider.browser_runtime_factory.manual_rescue_guard,
                        guard_id="different-guard",
                    ),
                ),
            ),
        ),
        (
            "oracle_blind_browser_mapping",
            lambda provider: replace(
                provider,
                browser_runtime_factory=replace(
                    provider.browser_runtime_factory,
                    environment_adapter_id="different-browser-adapter",
                ),
            ),
        ),
        (
            "action_safety_fault_classification",
            lambda provider: replace(
                provider,
                browser_runtime_factory=replace(
                    provider.browser_runtime_factory,
                    action_safety_policy=replace(
                        provider.browser_runtime_factory.action_safety_policy,
                        safety_policy_id="different-safety-policy",
                    ),
                ),
            ),
        ),
        (
            "recovery_action_planner",
            lambda provider: replace(
                provider,
                recovery_action_planner=replace(
                    provider.recovery_action_planner,
                    planner_id="different-recovery-planner",
                ),
            ),
        ),
        (
            "efficiency_measurement",
            lambda provider: replace(
                provider,
                efficiency_measurement_id="different-efficiency-measurement",
            ),
        ),
    ),
)
def test_actual_capability_id_version_must_match_frozen(
    tmp_path: Path,
    capability_id: str,
    mutation,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    changed = mutation(provider)
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match=capability_id,
    ):
        live.validate_provider_public_contract(
            changed,
            context.runtime_capability_authority,
        )


def test_operation_qualname_substitution_in_same_file_fails(tmp_path: Path) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    changed = replace(provider, begin_measurement=_alternate_begin_measurement)
    original = provider.capability_public_contracts()["efficiency_measurement"]
    substituted = changed.capability_public_contracts()["efficiency_measurement"]
    assert original["source_sha256"] == substituted["source_sha256"]
    assert original["operations"] != substituted["operations"]
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match="public-contract hash differs",
    ):
        live.validate_provider_public_contract(
            changed,
            context.runtime_capability_authority,
        )


@pytest.mark.parametrize("field", ("provider_id", "provider_version"))
def test_provider_public_contract_hash_is_frozen_compared_and_logged(
    tmp_path: Path,
    field: str,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    assert context.runtime_capability_authority[
        "expected_provider_public_contract_sha256"
    ] == provider.public_contract_sha256
    changed = replace(provider, **{field: f"different-{field}"})
    with pytest.raises(
        live.PC01LiveIntegrationError,
        match="public-contract hash differs",
    ):
        live.validate_provider_public_contract(
            changed,
            context.runtime_capability_authority,
        )


def test_provider_bootstrap_context_is_typed_deep_frozen_and_path_free(
    tmp_path: Path,
) -> None:
    provider, context, _ = _provider_and_context(tmp_path)
    preflight_view = {
        "schema_version": "table2-pc01-runtime-preflight-view-v1",
        "source_binding_sha256": "1" * 64,
        "deployment_topology": "SINGLE_DGX_HOST",
        "validator_contract": "measured-validator-v1",
        "expected_live_reset_task_index": 0,
        "preflight_content_sha256": "2" * 64,
        "service_url_map_content_sha256": sha256_json(SERVICE_URLS),
        "expected_dgx_runtime_identity_sha256": None,
        "expected_bridge_identity_sha256": None,
        "preflight_status": "PASS",
    }
    authority = dict(context.runtime_capability_authority)
    authority["deployment_preflight_binding_sha256"] = sha256_json(preflight_view)
    bootstrap = live.PC01ProviderBootstrapContext(
        schema_version=live.PC01_PROVIDER_BOOTSTRAP_SCHEMA_VERSION,
        protocol_id="table2-pc01-pilot-v1",
        model_seed=42,
        repository_commit="a" * 40,
        runtime_identity_sha256=provider.expected_runtime_identity_sha256,
        runtime_capability_authority=authority,
        runtime_environment={"benchmark": "webarena", "version": "0.14.3"},
        deployment_preflight_view=preflight_view,
        service_url_map=SERVICE_URLS,
        credential_capability=provider.credential_capability,
    )
    assert {item.name for item in fields(bootstrap)} == {
        "schema_version",
        "protocol_id",
        "model_seed",
        "repository_commit",
        "runtime_identity_sha256",
        "runtime_capability_authority",
        "runtime_environment",
        "deployment_preflight_view",
        "service_url_map",
        "credential_capability",
    }
    assert not {
        "campaign_dir",
        "task_path",
        "evaluator_path",
        "memory_path",
        "model_path",
        "sealed_path",
    } & {item.name for item in fields(bootstrap)}
    with pytest.raises(TypeError):
        bootstrap.runtime_environment["drift"] = True

    def walk(value):
        if isinstance(value, Mapping):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from walk(item)
        else:
            yield value

    assert not any(
        isinstance(item, Path)
        for field_name in (
            "runtime_capability_authority",
            "runtime_environment",
            "deployment_preflight_view",
            "service_url_map",
        )
        for item in walk(getattr(bootstrap, field_name))
    )


def test_nested_mutation_between_registry_require_and_delegate_fails(
    tmp_path: Path,
) -> None:
    global _PROVIDER_TO_MUTATE_DURING_MEASUREMENT
    provider, _, _ = _provider_and_context(tmp_path)
    changed = replace(provider, begin_measurement=_mutating_begin_measurement)
    _register(changed)
    _PROVIDER_TO_MUTATE_DURING_MEASUREMENT = changed
    try:
        dispatch = live._MeasurementDispatch(
            delegate=changed.begin_measurement,
            operation_id="begin_measurement",
        )
        with pytest.raises(
            live.PC01LiveIntegrationError,
            match="changed after installation",
        ):
            dispatch(object(), object())
    finally:
        _PROVIDER_TO_MUTATE_DURING_MEASUREMENT = None
