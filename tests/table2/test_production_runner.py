from __future__ import annotations

import functools
import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_table2_evaluation as evaluation_cli
from web_agent.eval.table2 import package_validator as package_validator_module
from web_agent.eval.table2 import production_runner as production_runner_module

from web_agent.eval.table2 import execution_guard
from web_agent.eval.table2.common import SchemaError, sha256_file, sha256_json
from web_agent.eval.table2.dependency_lock import build_semantic_dependency_lock
from web_agent.eval.table2.execution_guard import (
    EVALUATION_RUNNER_SCOPE,
    PC01_PAGE_BROKER_SECURITY_BLOCKED_STATUS,
    PC01_PAGE_BROKER_SECURITY_CLAIM_SCOPE,
    PC01_PAGE_BROKER_SECURITY_FIELD,
    PRODUCTION_RUNNER_ENTRYPOINT,
    RUNNER_ATTESTATION_SCHEMA_VERSION,
    assert_pc01_page_broker_production_authorized,
    assert_clean_git_checkout,
    blocked_pc01_page_broker_security_binding,
    validate_attested_callable_source,
    validate_pc01_page_broker_security_binding,
    validate_runner_attestation_payload,
)
from web_agent.eval.table2.live_deployment import (
    LIVE_DEPLOYMENT_BINDING_FIELD,
    stage_pc01_live_deployment_package,
)
from web_agent.eval.table2.webarena_preflight import PINNED_WEBARENA_PACKAGES
from web_agent.eval.table2.split_deployment_preflight import (
    SINGLE_HOST_TOPOLOGY,
    SPLIT_HOST_TOPOLOGY,
    build_dgx_model_runtime_identity,
)
from web_agent.eval.table2.package_validator import (
    PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD,
    PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH,
    _copy_model_payloads,
)
from web_agent.eval.table2.production_runner import (
    EvaluationRuntimeBinding,
    FrozenRuntimeContext,
    ProductionRunnerError,
    ProductionTable2Runner,
    RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION,
    RUNTIME_LIVE_CAPABILITY_IDS,
    WebArenaRuntimeBinding,
    _ActiveWebArenaSession,
    _load_processor_contract,
    _normal_task_specification,
    _validated_efficiency,
    build_runtime_capability_authority,
    create_runner,
    runtime_context_identity,
    validate_runtime_capability_authority,
)
from web_agent.runtime.contracts import (
    OpaqueTerminalSignal,
    RuntimeGeolocation,
    RuntimeStartState,
    canonical_json,
    canonical_sha256,
)
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.observation import ProcessorParityContract
from web_agent.runtime.state_reset import (
    REGISTERED_STATEFUL_BACKEND_ROLES,
    CallableEpisodeStateResetter,
)
from tests.table2.test_live_deployment import (
    _bound_environment as _live_deployment_environment,
    _manifest as _valid_live_deployment_manifest,
)
from tests.table2.test_package_validator import _valid_model_manifest


def _processor() -> ProcessorParityContract:
    return ProcessorParityContract(
        processor_class="FrozenProcessor",
        processor_revision="revision-1",
        processor_config_sha256="a" * 64,
        pre_action_field_mapping={"image": "pixel_values"},
        post_action_field_mapping={
            "before_image": "before_pixel_values",
            "after_image": "after_pixel_values",
        },
    )


def _efficiency() -> dict[str, int | float]:
    return {
        "model_call_count": 4,
        "input_token_count": 100,
        "output_token_count": 10,
        "model_parameter_count": 1_000,
        "trainable_parameter_count": 100,
        "peak_gpu_memory_mb": 256.0,
        "peak_system_memory_mb": 512.0,
        "training_gpu_hours": 3.5,
    }


def test_processor_payload_must_equal_runtime_record_bytes(tmp_path: Path) -> None:
    contract = _processor()
    canonical = tmp_path / "canonical.json"
    canonical.write_bytes(canonical_json(contract.to_dict()).encode("utf-8"))
    assert _load_processor_contract(canonical) == contract

    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProductionRunnerError, match="exact canonical"):
        _load_processor_contract(reformatted)


def test_task_projection_excludes_evaluator_and_reference_content() -> None:
    frozen_start_state = {
        "sites": ["example"],
        "start_url": "https://example.invalid/",
        "require_login": True,
        "storage_state": "./.auth/example_state.json",
        "geolocation": {
            "latitude": 23.8103,
            "longitude": 90.4125,
            "accuracy": 5,
        },
        "require_reset": False,
    }
    task = _normal_task_specification(
        {
            "task_id": "webarena.0",
            "instruction": "Open the visible account page",
            "upstream_index": 0,
            "benchmark_task_id": "0",
            "source_content_sha256": "b" * 64,
            "start_state": frozen_start_state,
            "task_config": {"reference_answer": "must stay evaluator-side"},
            "evaluator": {"oracle_success": True},
        },
        {"benchmark": "webarena", "benchmark_version": "0.14.3"},
    )
    assert task.metadata == {
        "task_partition": "normal",
        "upstream_index": 0,
        "benchmark_task_id": "0",
        "source_content_sha256": "b" * 64,
    }
    assert "evaluator" not in task.metadata
    assert "task_config" not in task.metadata
    assert type(task.runtime_start_state) is RuntimeStartState
    assert task.runtime_start_state == RuntimeStartState(
        sites=("example",),
        start_url="https://example.invalid/",
        require_login=True,
        storage_state="./.auth/example_state.json",
        geolocation=RuntimeGeolocation(
            latitude=23.8103,
            longitude=90.4125,
            accuracy=5,
        ),
        require_reset=False,
    )
    assert task.runtime_start_state.to_webarena_mapping() == frozen_start_state
    assert task.start_state_id == canonical_sha256(frozen_start_state)
    assert (
        RuntimeStartState.from_dict(task.runtime_start_state.to_dict())
        == task.runtime_start_state
    )
    serialized = json.dumps(task.to_dict(), sort_keys=True)
    assert "reference_answer" not in serialized
    assert "oracle_success" not in serialized


@pytest.mark.parametrize(
    "start_state",
    (
        {
            "sites": ["example"],
            "start_url": "https://example.invalid/",
            "require_login": True,
            "storage_state": "./.auth/example_state.json",
            "geolocation": None,
            # A missing reset input must not silently acquire a default.
        },
        {
            "sites": ["example"],
            "start_url": "https://example.invalid/",
            "require_login": True,
            "storage_state": "./.auth/example_state.json",
            "geolocation": None,
            "require_reset": False,
            "reference_answer": "sealed content",
        },
        {
            "sites": ["example"],
            "start_url": "https://user:secret@example.invalid/",
            "require_login": True,
            "storage_state": "./.auth/example_state.json",
            "geolocation": None,
            "require_reset": False,
        },
        {
            "sites": ["example"],
            "start_url": "https://example.invalid/",
            "require_login": True,
            "storage_state": '{"cookies":[{"value":"secret"}]}',
            "geolocation": None,
            "require_reset": False,
        },
    ),
)
def test_task_projection_rejects_incomplete_or_credential_bearing_start_state(
    start_state: dict,
) -> None:
    with pytest.raises(ProductionRunnerError, match="runtime start_state"):
        _normal_task_specification(
            {
                "task_id": "webarena.0",
                "instruction": "Open the visible account page",
                "upstream_index": 0,
                "benchmark_task_id": "0",
                "source_content_sha256": "b" * 64,
                "start_state": start_state,
            },
            {"benchmark": "webarena", "benchmark_version": "0.14.3"},
        )


def test_policy_integration_context_has_no_task_memory_or_sealed_capability() -> None:
    names = {field.name for field in fields(FrozenRuntimeContext)}
    assert names == {
        "model_manifest_paths",
        "model_payload_paths",
        "model_evidence_paths",
        "runtime_identity",
        "runtime_capability_authority",
    }
    assert not names & {
        "campaign_dir",
        "resolved_task_snapshot_path",
        "memory_store_paths",
        "sealed_root",
        "verifier_sink",
        "sealed_evaluator",
    }


def test_runtime_context_identity_removes_sealed_and_task_snapshot_metadata() -> None:
    projected = runtime_context_identity(
        {
            "checkpoint_systems": ["E1", "E2", "E3"],
            "environment": {
                "manifest_sha256": "a" * 64,
                "benchmark": "webarena",
                "benchmark_version": "0.14.3",
                LIVE_DEPLOYMENT_BINDING_FIELD: {
                    "manifest_sha256": "b" * 64,
                },
            },
            "evaluator": {
                "evaluator_id": "sealed-webarena",
                "source_sha256": "c" * 64,
            },
            "resolved_task_snapshot": {
                "sha256": "d" * 64,
                "record_type": "contains-sealed-config",
            },
        }
    )
    assert projected == {
        "checkpoint_systems": ["E1", "E2", "E3"],
        "environment": {
            "benchmark": "webarena",
            "benchmark_version": "0.14.3",
        },
    }
    assert "sealed" not in json.dumps(projected, sort_keys=True).lower()


def test_runtime_capability_authority_is_six_runtime_rows_only(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evidence_root = tmp_path / "measured"
    manifest_path = evidence_root / "deployment.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(_valid_live_deployment_manifest(evidence_root), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest_path,
        evidence_root=evidence_root,
        repository_root=repository_root,
        destination_artifact_root=tmp_path / "frozen",
    )
    service_urls = {
        "WA_SHOPPING": "http://shopping.test",
        "WA_SHOPPING_ADMIN": "http://shopping-admin.test",
        "WA_REDDIT": "http://reddit.test",
        "WA_GITLAB": "http://gitlab.test",
        "WA_WIKIPEDIA": "http://wikipedia.test",
        "WA_MAP": "http://map.test",
        "WA_HOMEPAGE": "http://homepage.test",
    }
    preflight = SimpleNamespace(
        binding={
            "schema_version": "fixture-binding-v1",
            "deployment_topology": "SINGLE_DGX_HOST",
            "validator_contract": "fixture-validator-v1",
            "expected_live_reset_task_index": 0,
            "preflight_content_sha256": "a" * 64,
            "service_url_map_content_sha256": sha256_json(service_urls),
            "expected_dgx_model_runtime_identity_sha256": None,
            "expected_bridge_identity_sha256": None,
        },
        evidence={"status": "PASS"},
        service_url_map=service_urls,
    )
    authority = build_runtime_capability_authority(
        staged,
        preflight,
        expected_provider_public_contract_sha256="f" * 64,
    )
    assert authority["schema_version"] == RUNTIME_CAPABILITY_AUTHORITY_SCHEMA_VERSION
    assert tuple(authority["capabilities"]) == RUNTIME_LIVE_CAPABILITY_IDS
    assert "sealed_webarena_evaluator" not in authority["capabilities"]
    assert authority["expected_provider_public_contract_sha256"] == "f" * 64
    assert validate_runtime_capability_authority(authority) == authority
    for capability_id, row in authority["capabilities"].items():
        readiness = json.loads(
            (
                staged.package_root
                / staged.manifest["capabilities"][capability_id][
                    "readiness_evidence_path"
                ]
            ).read_text(encoding="utf-8")
        )
        assert row["deployment_state_sha256"] == readiness[
            "deployment_state_sha256"
        ]


def test_production_runner_reopens_frozen_model_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_base_snapshot",
        lambda path, _backbone: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_export_manifest",
        lambda _path: None,
    )
    source = _valid_model_manifest(tmp_path / "source-model.json")
    campaign = tmp_path / "campaign"
    _, payloads, evidence = _copy_model_payloads(
        campaign_root=campaign,
        seed=42,
        manifest_path=source,
        manifest=json.loads(source.read_text(encoding="utf-8")),
        copied=[],
    )
    runner = object.__new__(ProductionTable2Runner)
    runner.root = campaign
    runner.manifest = {
        "matched_seeds": [42],
        "model_payloads_by_seed": {"42": payloads},
        "model_evidence_by_seed": {"42": evidence},
    }
    manifests, model_paths, evidence_paths = runner._verify_model_payloads()
    assert set(manifests) == {42}
    assert set(model_paths[42]) == set(payloads)
    assert set(evidence_paths[42]) == set(evidence["artifacts"])

    action_evidence = evidence_paths[42]["training_action_value_evidence"]
    action_evidence.write_bytes(action_evidence.read_bytes() + b" ")
    with pytest.raises(ProductionRunnerError, match="evidence bundle"):
        runner._verify_model_payloads()


@pytest.mark.parametrize("alias_kind", ("symlink-parent", "external-hardlink"))
def test_campaign_model_payload_rejects_path_and_inode_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_base_snapshot",
        lambda path, _backbone: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_export_manifest",
        lambda _path: None,
    )
    source = _valid_model_manifest(tmp_path / "source-model.json")
    campaign = tmp_path / "campaign"
    frozen_manifest, payloads, evidence = _copy_model_payloads(
        campaign_root=campaign,
        seed=42,
        manifest_path=source,
        manifest=json.loads(source.read_text(encoding="utf-8")),
        copied=[],
    )
    checkpoint = campaign / payloads["selected_checkpoint"]["path"]
    if alias_kind == "symlink-parent":
        relocated = campaign / "relocated-checkpoint-parent"
        checkpoint.parent.rename(relocated)
        checkpoint.parent.symlink_to(relocated, target_is_directory=True)
        expected = "symlink"
    else:
        external = tmp_path / "external-checkpoint.bin"
        external.write_bytes(checkpoint.read_bytes())
        checkpoint.unlink()
        checkpoint.hardlink_to(external)
        expected = "hard-linked"

    frozen_model = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    with pytest.raises(SchemaError, match=expected):
        package_validator_module._validate_model_artifact_payloads(
            frozen_manifest,
            frozen_model,
            path_base=campaign,
        )

    runner = object.__new__(ProductionTable2Runner)
    runner.root = campaign
    runner.manifest = {
        "matched_seeds": [42],
        "model_payloads_by_seed": {"42": payloads},
        "model_evidence_by_seed": {"42": evidence},
    }
    with pytest.raises(ProductionRunnerError, match="unsafe model payload path"):
        runner._verify_model_payloads()


def test_package_model_payload_second_pass_rejects_cross_role_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_base_snapshot",
        lambda path, _backbone: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_export_manifest",
        lambda _path: None,
    )
    source = _valid_model_manifest(tmp_path / "source-model.json")
    campaign = tmp_path / "campaign"
    frozen_manifest, payloads, _ = _copy_model_payloads(
        campaign_root=campaign,
        seed=42,
        manifest_path=source,
        manifest=json.loads(source.read_text(encoding="utf-8")),
        copied=[],
    )
    checkpoint = campaign / payloads["selected_checkpoint"]["path"]
    original = package_validator_module._validated_campaign_relative_model_payload
    calls = 0

    def replace_after_first_role_pass(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == len(package_validator_module.MODEL_PAYLOAD_ROLES):
            replacement = checkpoint.with_suffix(".replacement")
            replacement.write_bytes(b"cross-role replacement")
            replacement.replace(checkpoint)
        return result

    monkeypatch.setattr(
        package_validator_module,
        "_validated_campaign_relative_model_payload",
        replace_after_first_role_pass,
    )
    with pytest.raises(SchemaError, match="return authentication|before return"):
        package_validator_module._validate_model_artifact_payloads(
            frozen_manifest,
            json.loads(frozen_manifest.read_text(encoding="utf-8")),
            path_base=campaign,
        )
    assert calls > len(package_validator_module.MODEL_PAYLOAD_ROLES)


def test_package_model_evidence_second_pass_rejects_cross_role_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_base_snapshot",
        lambda path, _backbone: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_export_manifest",
        lambda _path: None,
    )
    source = _valid_model_manifest(tmp_path / "source-model.json")
    campaign = tmp_path / "campaign"
    frozen_manifest, _, evidence_bundle = _copy_model_payloads(
        campaign_root=campaign,
        seed=42,
        manifest_path=source,
        manifest=json.loads(source.read_text(encoding="utf-8")),
        copied=[],
    )
    model = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    executable = package_validator_module._validate_model_artifact_payloads(
        frozen_manifest,
        model,
        path_base=campaign,
    )
    export = campaign / evidence_bundle["artifacts"]["export_manifest"]["path"]
    original = package_validator_module._validate_pc01_model_evidence_cross_consistency

    def replace_after_semantic_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        replacement = export.with_suffix(".replacement")
        replacement.write_bytes(b"cross-role replacement")
        replacement.replace(export)
        return result

    monkeypatch.setattr(
        package_validator_module,
        "_validate_pc01_model_evidence_cross_consistency",
        replace_after_semantic_validation,
    )
    with pytest.raises(SchemaError, match="return authentication|before return"):
        package_validator_module._validate_model_evidence_bundle(
            frozen_manifest,
            model,
            path_base=campaign,
            executable_payloads=executable,
        )


def test_production_runner_final_pass_rejects_cross_role_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_base_snapshot",
        lambda path, _backbone: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        package_validator_module,
        "_validate_registered_pc01_export_manifest",
        lambda _path: None,
    )
    source = _valid_model_manifest(tmp_path / "source-model.json")
    campaign = tmp_path / "campaign"
    _, payloads, evidence = _copy_model_payloads(
        campaign_root=campaign,
        seed=42,
        manifest_path=source,
        manifest=json.loads(source.read_text(encoding="utf-8")),
        copied=[],
    )
    checkpoint = campaign / payloads["selected_checkpoint"]["path"]
    original = production_runner_module._validated_campaign_relative_model_payload
    calls = 0

    def replace_after_first_role_pass(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == len(package_validator_module.MODEL_PAYLOAD_ROLES):
            replacement = checkpoint.with_suffix(".replacement")
            replacement.write_bytes(b"cross-role replacement")
            replacement.replace(checkpoint)
        return result

    monkeypatch.setattr(
        production_runner_module,
        "_validated_campaign_relative_model_payload",
        replace_after_first_role_pass,
    )
    runner = object.__new__(ProductionTable2Runner)
    runner.root = campaign
    runner.manifest = {
        "matched_seeds": [42],
        "model_payloads_by_seed": {"42": payloads},
        "model_evidence_by_seed": {"42": evidence},
    }
    with pytest.raises(ProductionRunnerError, match="changed before runtime handoff"):
        runner._verify_model_payloads()
    assert calls == len(package_validator_module.MODEL_PAYLOAD_ROLES)


def test_runtime_and_evaluator_factories_are_separate_capabilities() -> None:
    names = {field.name for field in fields(EvaluationRuntimeBinding)}
    assert "create_webarena_runtime" in names
    assert "create_sealed_evaluator" not in names
    assert "create_webarena_episode" not in names


def test_production_webarena_binding_requires_typed_action_safety_policy() -> None:
    names = {field.name for field in fields(WebArenaRuntimeBinding)}
    assert "action_safety_policy" in names
    assert "manual_rescue_guard" in names


def test_live_action_safety_identity_must_match_frozen_environment() -> None:
    source_sha256 = "a" * 64
    runner = object.__new__(ProductionTable2Runner)
    runner.environment = {
        "benchmark_version": "fixture-version",
        "environment_adapter_id": "fixture-adapter",
        "environment_adapter_version": "v1",
        "environment_state_digester": {
            "digester_id": "fixture-digester",
            "digester_version": "v1",
        },
        "infrastructure_classifier": {
            "classifier_id": "fixture-classifier",
            "classifier_version": "v1",
            "rules_sha256": "b" * 64,
        },
        "page_settle_policy": {
            "policy_id": "fixture-settle",
            "network_idle_required": True,
            "settle_timeout_seconds": 5.0,
        },
        "manual_rescue_guard": {
            "guard_id": "fixture-manual-rescue-guard",
            "guard_version": "v1",
            "evidence_mode": "exclusive_controller_input_audit_v1",
        },
        "safety_policy_id": "fixture-safety",
        "safety_policy_version": "v1",
        "destructive_action_policy": "fixture-no-destructive-actions",
        "evaluator": {
            "evaluator_id": "fixture-evaluator",
            "evaluator_version": "v1",
        },
    }
    runner.integration = SimpleNamespace(
        runtime_identity={
            "runtime_integration": {"source_sha256": source_sha256}
        }
    )
    runtime = SimpleNamespace(
        benchmark_version="fixture-version",
        environment_adapter_id="fixture-adapter",
        environment_adapter_version="v1",
        environment_state_digester=SimpleNamespace(
            digester_id="fixture-digester",
            digester_version="v1",
        ),
        infrastructure_fault_classifier=SimpleNamespace(
            classifier_id="fixture-classifier",
            classifier_version="v1",
            rules_sha256="b" * 64,
        ),
        page_settle_policy=SimpleNamespace(
            policy_id="fixture-settle",
            network_idle_required=True,
            settle_timeout_seconds=5.0,
        ),
        action_safety_policy=SimpleNamespace(
            safety_policy_id="different-safety-policy",
            safety_policy_version="v1",
            destructive_action_policy="fixture-no-destructive-actions",
            source_sha256=source_sha256,
        ),
        manual_rescue_guard=SimpleNamespace(
            guard_id="fixture-manual-rescue-guard",
            guard_version="v1",
            evidence_mode="exclusive_controller_input_audit_v1",
            source_sha256=source_sha256,
        ),
        reset_state_attester=SimpleNamespace(source_sha256=source_sha256),
    )
    evaluator = SimpleNamespace(
        benchmark_version="fixture-version",
        evaluator_id="fixture-evaluator",
        evaluator_version="v1",
    )

    with pytest.raises(ProductionRunnerError, match="action-safety safety_policy_id"):
        runner._verify_episode_bindings(runtime, evaluator)

    runtime.action_safety_policy.safety_policy_id = "fixture-safety"
    runtime.action_safety_policy.source_sha256 = "c" * 64
    with pytest.raises(ProductionRunnerError, match="action-safety source differs"):
        runner._verify_episode_bindings(runtime, evaluator)


def test_sealed_evaluator_cannot_mutate_registered_runtime_backends(
    tmp_path: Path,
) -> None:
    mutable = {"state": "clean"}

    def digests() -> dict[str, str]:
        return {
            role: canonical_sha256({"role": role, "state": mutable["state"]})
            for role in REGISTERED_STATEFUL_BACKEND_ROLES
        }

    resetter = CallableEpisodeStateResetter(
        resetter_id="guard-fixture",
        resetter_version="v1",
        source_sha256="a" * 64,
        callback=lambda request: (_ for _ in ()).throw(AssertionError(request)),
        digest_callback=digests,
    )
    logs = EpisodeEventLogs(
        tmp_path / "runtime",
        episode_id="guard:E0:task:repeat-0:seed-42",
        include_memory=False,
        system_id="E0",
        task_id="task",
        repeat_id=0,
        matched_seed=42,
    )
    session = _ActiveWebArenaSession(
        evaluator=SimpleNamespace(),
        resetter=resetter,
        event_logs=logs,
        task_id="task",
        abort_episode=lambda episode_id, task_id: None,
    )
    runner = object.__new__(ProductionTable2Runner)
    signal = OpaqueTerminalSignal(
        event_id="opaque",
        token_sha256="b" * 64,
        terminate=False,
    )
    assert (
        runner._invoke_guarded_evaluator(
            session,
            callback_kind="transition_verifier",
            callback=lambda: signal,
        )
        == signal
    )

    def mutate() -> OpaqueTerminalSignal:
        mutable["state"] = "oracle-contaminated"
        return signal

    with pytest.raises(ProductionRunnerError, match="mutated"):
        runner._invoke_guarded_evaluator(
            session,
            callback_kind="transition_verifier",
            callback=mutate,
        )


def test_efficiency_probe_is_complete_and_oracle_blind() -> None:
    result = _validated_efficiency(
        _efficiency(),
        task_wall_clock_seconds=1.25,
        memory_index_size=4096,
    )
    assert result["model_call_count"] == 4
    assert result["task_wall_clock_seconds"] == 1.25
    assert result["memory_index_size"] == 4096

    with pytest.raises(ProductionRunnerError, match="dispatch ledger"):
        _validated_efficiency(
            _efficiency(),
            task_wall_clock_seconds=1.25,
            memory_index_size=4096,
            expected_model_call_count=3,
        )

    incomplete = _efficiency()
    del incomplete["training_gpu_hours"]
    with pytest.raises(ProductionRunnerError, match="incomplete"):
        _validated_efficiency(
            incomplete,
            task_wall_clock_seconds=1.0,
            memory_index_size=0,
        )

    leaked = {**_efficiency(), "oracle_success": True}
    with pytest.raises(Exception, match="leaks sealed verifier keys"):
        _validated_efficiency(
            leaked,
            task_wall_clock_seconds=1.0,
            memory_index_size=0,
        )


def test_production_factory_fails_closed_without_frozen_campaign(tmp_path: Path) -> None:
    with pytest.raises((ProductionRunnerError, FileNotFoundError)):
        create_runner(campaign_dir=tmp_path)


def test_production_runner_revalidates_checkpoint_receipt_before_runtime_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {
        "receipt_sha256": "a" * 64,
        "schema_version": "fixture",
        "gate_id": "fixture-gate",
    }
    receipt = {"status": "PASS"}
    calls: list[dict] = []

    def validate(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return receipt, binding, ()

    monkeypatch.setattr(
        production_runner_module,
        "_validate_pc01_checkpoint_compatibility_readiness",
        validate,
    )
    runner = object.__new__(ProductionTable2Runner)
    runner.root = tmp_path / "campaign"
    runner.repository_root = tmp_path / "source"
    runner.manifest = {
        "repository_commit": "1" * 40,
        "pc01_checkpoint_compatibility_receipt_sha256": "a" * 64,
        PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD: binding,
    }
    runner.attestation = {
        "runtime_identity": {
            PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD: binding,
        }
    }
    runner._revalidate_pc01_checkpoint_compatibility()
    assert runner.checkpoint_compatibility_receipt == receipt
    assert runner.checkpoint_compatibility_binding == binding
    assert calls == [
        {
            "path": (
                runner.root
                / "frozen"
                / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
            ),
            "repository_root": runner.repository_root,
            "expected_source_commit": "1" * 40,
            "model_manifest_path": runner.root
            / "frozen/models/seed_42.json",
            "selection_evidence_path": runner.root
            / "frozen/selection_evidence/manifest.json",
            "path_base": runner.root,
        }
    ]

    runner.manifest["pc01_checkpoint_compatibility_receipt_sha256"] = "b" * 64
    with pytest.raises(ProductionRunnerError, match="bindings differ"):
        runner._revalidate_pc01_checkpoint_compatibility()


def test_production_runner_fails_closed_when_checkpoint_receipt_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args, **_kwargs):
        raise SchemaError("tampered checkpoint receipt")

    monkeypatch.setattr(
        production_runner_module,
        "_validate_pc01_checkpoint_compatibility_readiness",
        reject,
    )
    runner = object.__new__(ProductionTable2Runner)
    runner.root = tmp_path / "campaign"
    runner.repository_root = tmp_path / "source"
    runner.manifest = {"repository_commit": "1" * 40}
    runner.attestation = {"runtime_identity": {}}
    with pytest.raises(
        ProductionRunnerError,
        match="checkpoint compatibility revalidation failed",
    ):
        runner._revalidate_pc01_checkpoint_compatibility()


def test_injected_callback_source_must_be_in_attested_hash_set() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = Path(__file__).resolve()
    relative = source.relative_to(repository_root).as_posix()
    hashes = {relative: sha256_file(source)}

    def registered_callback() -> None:
        return None

    assert (
        validate_attested_callable_source(
            registered_callback,
            repository_root=repository_root,
            source_hashes=hashes,
            field="fixture.registered_callback",
        )
        == relative
    )
    with pytest.raises(SchemaError, match="outside the attested repository"):
        validate_attested_callable_source(
            Path.exists,
            repository_root=repository_root,
            source_hashes=hashes,
            field="fixture.external_callback",
        )
    with pytest.raises(SchemaError, match="source hash differs"):
        validate_attested_callable_source(
            registered_callback,
            repository_root=repository_root,
            source_hashes={relative: "0" * 64},
            field="fixture.modified_callback",
        )


def test_clean_checkout_guard_checks_untracked_and_tracked_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def clean_output(command: list[str], **_: object) -> str:
        calls.append(tuple(command))
        return "f" * 40 if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(execution_guard.subprocess, "check_output", clean_output)
    assert assert_clean_git_checkout(Path(__file__).parent) == "f" * 40
    assert (
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ) in calls

    def dirty_output(command: list[str], **_: object) -> str:
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "f" * 40
        return " M src/web_agent/eval/table2/production_runner.py\n"

    monkeypatch.setattr(execution_guard.subprocess, "check_output", dirty_output)
    with pytest.raises(SchemaError, match="clean Git checkout"):
        assert_clean_git_checkout(Path(__file__).parent)


def test_evaluation_cli_bootstrap_closes_source_and_dependency_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = Path(evaluation_cli.__file__).resolve()
    relative = script.relative_to(repository_root).as_posix()
    campaign = tmp_path / "campaign"
    frozen_source = campaign / "frozen/runner_source" / relative
    frozen_source.parent.mkdir(parents=True)
    frozen_source.write_bytes(script.read_bytes())
    preflight = {
        "host": {
            "system": evaluation_cli.platform.system(),
            "release": evaluation_cli.platform.release(),
            "machine": evaluation_cli.platform.machine(),
            "python_version": evaluation_cli.platform.python_version(),
            "python_executable_sha256": sha256_file(Path(sys.executable)),
        },
        "package_check": {
            "status": "PASS",
            "packages": [
                {
                    "distribution": name,
                    "expected_version": version,
                    "actual_version": version,
                    "status": "PASS",
                }
                for name, version in sorted(PINNED_WEBARENA_PACKAGES.items())
            ]
        },
        "browser_check": {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture-chromium",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
        },
    }
    environment = {
        "benchmark": "webarena",
        "benchmark_version": "fixture-benchmark",
        "benchmark_revision": "fixture-revision",
        "operating_system": f"fixture-{evaluation_cli.platform.system()}",
        "browser": "chromium",
        "browser_version": "fixture-chromium",
        "playwright_version": PINNED_WEBARENA_PACKAGES["playwright"],
        "controller_id": "fixture-controller",
        "controller_version": "fixture-controller-v1",
        "environment_adapter_id": "fixture-adapter",
        "environment_adapter_version": "fixture-adapter-v1",
        "container_digest": "sha256:fixture",
    }
    (campaign / "frozen/webarena_deployment_preflight.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    dependency_lock = campaign / "frozen/dependency.lock"
    dependency_lock.write_text(
        json.dumps(
            build_semantic_dependency_lock(
                environment=environment,
                deployment_preflight=preflight,
                deployment_topology=SINGLE_HOST_TOPOLOGY,
            )
        ),
        encoding="utf-8",
    )
    environment.update(
        {
            "dependency_lock_relative_path": "dependency.lock",
            "dependency_lock_sha256": sha256_file(dependency_lock),
        }
    )
    (campaign / "frozen/environment.json").write_text(
        json.dumps(
            environment
        ),
        encoding="utf-8",
    )
    commit = "f" * 40
    runner_entrypoint = "fixture.runner:create"
    attestation = {
        "runner_entrypoint": runner_entrypoint,
        "repository_commit": commit,
        "source_files": [
            {"relative_path": relative, "sha256": sha256_file(script)}
        ],
    }
    attestation_path = campaign / "frozen/runner_attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_kind": "engineering_pilot",
                "campaign_mode": "evaluation",
                "evidence_label": "PILOT_ONLY",
                "runner_identity_scope": evaluation_cli.EVALUATION_RUNNER_SCOPE,
                "runner_attestation_sha256": sha256_file(attestation_path),
                "repository_commit": commit,
            }
        ),
        encoding="utf-8",
    )

    def clean_output(command: list[str], **_: object) -> str:
        return commit if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(evaluation_cli.subprocess, "check_output", clean_output)
    monkeypatch.setattr(
        evaluation_cli.importlib_metadata,
        "version",
        lambda distribution: PINNED_WEBARENA_PACKAGES[distribution],
    )
    evaluation_cli._bootstrap_verify_evaluation_source(
        campaign,
        runner_entrypoint=runner_entrypoint,
    )

    dependency_lock.write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="dependency-lock hash mismatch"):
        evaluation_cli._bootstrap_verify_evaluation_source(
            campaign,
            runner_entrypoint=runner_entrypoint,
        )


def test_evaluation_cli_bootstrap_accepts_only_registered_smoke_profile(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "smoke-campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_kind": "engineering_pilot",
                "campaign_mode": "smoke",
                "evidence_label": "PILOT_ONLY",
            }
        ),
        encoding="utf-8",
    )

    evaluation_cli._bootstrap_verify_evaluation_source(
        campaign,
        runner_entrypoint="fixture.runner:create",
    )


@pytest.mark.parametrize(
    ("campaign_kind", "evidence_label"),
    (
        ("unknown", "PILOT_ONLY"),
        ("engineering_pilot", "FINAL_LOCKED"),
        ("locked_final", "PILOT_ONLY"),
    ),
)
def test_evaluation_cli_bootstrap_rejects_unregistered_smoke_before_runner_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign_kind: str,
    evidence_label: str,
) -> None:
    campaign = tmp_path / "bad-smoke-campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_kind": campaign_kind,
                "campaign_mode": "smoke",
                "evidence_label": evidence_label,
            }
        ),
        encoding="utf-8",
    )
    imported = tmp_path / "runner-imported"
    (tmp_path / "profile_probe_runner.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n"
        "def run(**kwargs):\n"
        "    return kwargs\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    args = SimpleNamespace(
        campaign_dir=campaign,
        runner="profile_probe_runner:run",
        runner_factory=False,
        pc01_operations_provider_factory=None,
        pc01_credential_capability_root=None,
        pc01_credential_capability_id=None,
        pc01_credential_capability_version=None,
        pc01_provider_boundary_receipt=None,
        prepare_pc01_provider_boundary_receipt=None,
        block_id=None,
        maximum_blocks=None,
        live_readiness_probe_for=None,
    )
    monkeypatch.setattr(evaluation_cli, "parse_args", lambda: args)

    with pytest.raises(RuntimeError, match="profile is not registered"):
        evaluation_cli.main()

    assert not imported.exists()
    assert "profile_probe_runner" not in sys.modules


def test_evaluation_cli_bootstrap_rejects_mixed_smoke_before_provider_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = tmp_path / "bad-provider-smoke-campaign"
    campaign.mkdir()
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "campaign_kind": "engineering_pilot",
                "campaign_mode": "smoke",
                "evidence_label": "FINAL_LOCKED",
            }
        ),
        encoding="utf-8",
    )
    imported = tmp_path / "provider-imported"
    (tmp_path / "profile_probe_provider.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n"
        "def create(bootstrap_context):\n"
        "    return bootstrap_context\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    args = SimpleNamespace(
        campaign_dir=campaign,
        runner=evaluation_cli.PC01_PRODUCTION_RUNNER_ENTRYPOINT,
        runner_factory=True,
        pc01_operations_provider_factory="profile_probe_provider:create",
        pc01_credential_capability_root=tmp_path / "credentials",
        pc01_credential_capability_id="fixture-credentials",
        pc01_credential_capability_version="v1",
        pc01_provider_boundary_receipt=tmp_path / "boundary.json",
        prepare_pc01_provider_boundary_receipt=None,
        block_id=None,
        maximum_blocks=None,
        live_readiness_probe_for=None,
    )
    monkeypatch.setattr(evaluation_cli, "parse_args", lambda: args)

    with pytest.raises(RuntimeError, match="profile is not registered"):
        evaluation_cli.main()

    assert not imported.exists()
    assert "profile_probe_provider" not in sys.modules


def test_evaluation_cli_bootstrap_remeasures_split_browser_but_blocks_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_host = {
        "system": evaluation_cli.platform.system(),
        "release": evaluation_cli.platform.release(),
        "machine": evaluation_cli.platform.machine(),
        "python_version": evaluation_cli.platform.python_version(),
        "python_executable_sha256": sha256_file(Path(sys.executable)),
    }
    local = {
        "host": browser_host,
        "package_check": {
            "status": "PASS",
            "packages": [
                {
                    "distribution": name,
                    "expected_version": version,
                    "actual_version": version,
                    "status": "PASS",
                }
                for name, version in sorted(PINNED_WEBARENA_PACKAGES.items())
            ],
        },
        "browser_check": {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture-chromium",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
        },
    }
    dgx_dependencies = {
        "host": {
            "system": "Linux",
            "release": "fixture-dgx-release",
            "machine": "aarch64",
            "python_version": "3.12.3",
            "python_executable_sha256": "a" * 64,
        },
        "packages": [
            {"distribution": "accelerate", "version": "1.0.0"},
            {"distribution": "bitsandbytes", "version": "0.49.0"},
            {"distribution": "peft", "version": "0.18.0"},
            {"distribution": "torch", "version": "2.13.0"},
            {"distribution": "transformers", "version": "4.57.6"},
        ],
    }
    dgx = build_dgx_model_runtime_identity(
        host_identity_sha256=sha256_json(dgx_dependencies["host"]),
        dependency_identity=dgx_dependencies,
        runtime_identity=evaluation_cli.PC01_SPLIT_RUNTIME_IDENTITY,
        runtime_source_files=[
            {"relative_path": "src/runtime.py", "sha256": "c" * 64}
        ],
        runtime_environment={
            "python_version": "3.12.3",
            "python_executable_sha256": "a" * 64,
            "torch_version": "2.13.0",
            "transformers_version": "4.57.6",
            "cuda_available": True,
            "cuda_runtime_version": "13.0",
            "device_type": "cuda",
            "device_name": "NVIDIA GB10",
            "device_count": 1,
            "container_digest": "sha256:" + "d" * 64,
        },
    )
    preflight = {
        "local_browser_preflight": local,
        "dgx_model_runtime_identity": dgx,
    }
    environment = {
        "benchmark": "webarena",
        "benchmark_version": "fixture-benchmark",
        "benchmark_revision": "fixture-revision",
        "operating_system": f"fixture-{evaluation_cli.platform.system()}",
        "browser": "chromium",
        "browser_version": "fixture-chromium",
        "playwright_version": PINNED_WEBARENA_PACKAGES["playwright"],
        "controller_id": "fixture-controller",
        "controller_version": "v1",
        "environment_adapter_id": "fixture-adapter",
        "environment_adapter_version": "v1",
        "container_digest": "sha256:fixture",
    }
    lock = build_semantic_dependency_lock(
        environment=environment,
        deployment_preflight=preflight,
        deployment_topology=SPLIT_HOST_TOPOLOGY,
    )
    campaign = tmp_path / "split-campaign"
    (campaign / "frozen").mkdir(parents=True)
    (campaign / "frozen/webarena_deployment_preflight.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    lock_path = campaign / "frozen/dependency.lock"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(
        evaluation_cli.importlib_metadata,
        "version",
        lambda name: PINNED_WEBARENA_PACKAGES[name],
    )
    with pytest.raises(RuntimeError, match="split deployment dispatch is blocked"):
        evaluation_cli._bootstrap_validate_semantic_dependency_lock(
            campaign_root=campaign,
            environment=environment,
            dependency_lock=lock_path,
        )

    missing = json.loads(json.dumps(preflight))
    del missing["dgx_model_runtime_identity"]["dependency_identity"]
    (campaign / "frozen/webarena_deployment_preflight.json").write_text(
        json.dumps(missing), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="DGX dependency/runtime inventory"):
        evaluation_cli._bootstrap_validate_semantic_dependency_lock(
            campaign_root=campaign,
            environment=environment,
            dependency_lock=lock_path,
        )


def _wrong_pc01_provider_factory(bootstrap_context: object) -> object:
    assert bootstrap_context is not None
    return object()


def _alternate_pc01_provider_factory(bootstrap_context: object) -> object:
    assert bootstrap_context is not None
    return object()


class _CallableProviderFactory:
    def __init__(self) -> None:
        self.captured = {"oracle_success": True, "evaluator_path": "/sealed"}

    def __call__(self, bootstrap_context: object) -> object:
        del bootstrap_context
        return object()


_PARTIAL_PROVIDER_FACTORY = functools.partial(_wrong_pc01_provider_factory)
_CALLABLE_PROVIDER_FACTORY = _CallableProviderFactory()


def test_pc01_cli_requires_explicit_same_process_provider_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        campaign_dir=tmp_path,
        runner=evaluation_cli.PC01_PRODUCTION_RUNNER_ENTRYPOINT,
        runner_factory=True,
        pc01_operations_provider_factory=None,
        block_id=None,
        maximum_blocks=None,
        live_readiness_probe_for=None,
    )
    monkeypatch.setattr(evaluation_cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluation_cli,
        "_bootstrap_verify_evaluation_source",
        lambda *unused_args, **unused_kwargs: None,
    )
    with pytest.raises(ValueError, match="explicit.*provider-factory"):
        evaluation_cli.main()


def test_pc01_cli_provider_factory_must_return_exact_attested_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    relative = Path(__file__).resolve().relative_to(repository_root).as_posix()
    row = {"relative_path": relative, "sha256": sha256_file(Path(__file__))}
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    credential_root = tmp_path / "credentials"
    credential_root.mkdir()
    isolation = tmp_path / "isolation.json"
    isolation.write_text("{}", encoding="utf-8")
    binding = {
        "factory_entrypoint": (
            "tests.table2.test_production_runner:_wrong_pc01_provider_factory"
        ),
        "factory_module": "tests.table2.test_production_runner",
        "factory_qualname": "_wrong_pc01_provider_factory",
        "source_relative_path": relative,
        "source_sha256": row["sha256"],
    }
    monkeypatch.setattr(
        evaluation_cli,
        "_preflight_pc01_provider_install",
        lambda *unused_args, **unused_kwargs: evaluation_cli._ProviderInstallPreflight(
            context=object(),
            binding=binding,
            source_hashes={relative: row["sha256"]},
            campaign_state_sha256="a" * 64,
        ),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_validate_pc01_provider_boundary_receipt",
        lambda *unused_args, **unused_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="wrong exact type"):
        evaluation_cli._install_pc01_operations_provider(
            campaign,
            provider_factory_entrypoint=binding["factory_entrypoint"],
            credential_capability_root=credential_root,
            credential_capability_id="fixture-credentials",
            credential_capability_version="v1",
            provider_boundary_receipt=isolation,
        )


def _provider_factory_binding(function_name: str) -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    relative = Path(__file__).resolve().relative_to(repository_root).as_posix()
    return {
        "schema_version": "table2-pc01-provider-bootstrap-v1",
        "factory_entrypoint": (
            f"tests.table2.test_production_runner:{function_name}"
        ),
        "factory_module": "tests.table2.test_production_runner",
        "factory_qualname": function_name,
        "source_relative_path": relative,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "provider_contract_schema_version": (
            "table2-pc01-provider-public-contract-v1"
        ),
        "expected_provider_public_contract_sha256": "e" * 64,
        "source_plane": "runtime_only",
    }


def test_provider_factory_entrypoint_must_equal_frozen_identity() -> None:
    binding = _provider_factory_binding("_wrong_pc01_provider_factory")
    with pytest.raises(RuntimeError, match="entrypoint differs from frozen"):
        evaluation_cli._load_exact_provider_factory(
            "tests.table2.test_production_runner:_alternate_pc01_provider_factory",
            binding=binding,
            source_hashes={
                binding["source_relative_path"]: binding["source_sha256"]
            },
        )


@pytest.mark.parametrize(
    "function_name",
    ("_PARTIAL_PROVIDER_FACTORY", "_CALLABLE_PROVIDER_FACTORY", "_CallableProviderFactory"),
)
def test_provider_factory_rejects_partial_class_and_callable_objects(
    function_name: str,
) -> None:
    binding = _provider_factory_binding(function_name)
    with pytest.raises(RuntimeError, match="exact function"):
        evaluation_cli._load_exact_provider_factory(
            binding["factory_entrypoint"],
            binding=binding,
            source_hashes={
                binding["source_relative_path"]: binding["source_sha256"]
            },
        )


@pytest.mark.parametrize("forbidden_plane", ("sealed", "broker"))
def test_provider_factory_source_cannot_be_sealed_or_broker_plane(
    forbidden_plane: str,
) -> None:
    binding = _provider_factory_binding("_wrong_pc01_provider_factory")
    other_source = {
        "source_relative_path": "src/web_agent/eval/table2/production_runner.py",
        "source_sha256": sha256_file(
            Path(__file__).resolve().parents[2]
            / "src/web_agent/eval/table2/production_runner.py"
        ),
    }
    provider_source = {
        "source_relative_path": binding["source_relative_path"],
        "source_sha256": binding["source_sha256"],
    }
    sealed = provider_source if forbidden_plane == "sealed" else other_source
    broker = provider_source if forbidden_plane == "broker" else other_source
    deployment = SimpleNamespace(
        manifest={
            "capabilities": {
                "deterministic_reset": {
                    **provider_source,
                    "capability_id": "deterministic_reset",
                },
                "sealed_webarena_evaluator": {
                    **sealed,
                    "capability_id": "sealed_webarena_evaluator",
                },
            },
            "sealed_page_broker": broker,
        }
    )
    with pytest.raises(RuntimeError, match="never sealed, broker, or shared-plane"):
        evaluation_cli._validate_provider_source_plane(
            provider_binding=binding,
            validated_live=deployment,
        )


def test_locked_mount_and_campaign_validation_precede_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"factory_load": 0}

    def reject_preflight(*_args, **_kwargs):
        raise RuntimeError("locked mount or campaign mismatch")

    def load_factory(*_args, **_kwargs):
        calls["factory_load"] += 1
        return _wrong_pc01_provider_factory

    monkeypatch.setattr(
        evaluation_cli, "_preflight_pc01_provider_install", reject_preflight
    )
    monkeypatch.setattr(evaluation_cli, "_load_exact_provider_factory", load_factory)
    with pytest.raises(RuntimeError, match="locked mount or campaign mismatch"):
        evaluation_cli._install_pc01_operations_provider(
            tmp_path / "campaign",
            provider_factory_entrypoint=(
                "tests.table2.test_production_runner:_wrong_pc01_provider_factory"
            ),
            credential_capability_root=tmp_path / "credentials",
            credential_capability_id="fixture-credentials",
            credential_capability_version="v1",
            provider_boundary_receipt=tmp_path / "boundary.json",
        )
    assert calls["factory_load"] == 0


def test_same_process_page_broker_binding_is_truthful_and_unpromotable() -> None:
    binding = blocked_pc01_page_broker_security_binding()
    assert binding == {
        "schema_version": "table2-pc01-page-broker-security-v1",
        "status": PC01_PAGE_BROKER_SECURITY_BLOCKED_STATUS,
        "claim_scope": PC01_PAGE_BROKER_SECURITY_CLAIM_SCOPE,
        "architecture": "same_process_in_memory_typed_capabilities",
        "same_process_broker": True,
        "kernel_process_isolation": False,
        "runtime_process_can_import_sealed_capability": True,
        "external_process_isolation_evidence_present": False,
        "production_dispatch_authorized": False,
    }
    assert validate_pc01_page_broker_security_binding(binding) == binding
    with pytest.raises(SchemaError, match="externally evidenced process isolation"):
        assert_pc01_page_broker_production_authorized(binding)

    forged = dict(binding)
    forged["production_dispatch_authorized"] = True
    with pytest.raises(SchemaError, match="non-isolated production blocker"):
        validate_pc01_page_broker_security_binding(forged)


def test_process_isolation_block_prevents_adversarial_provider_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"factory_load": 0}

    def blocked_preflight(*_args, **_kwargs):
        # A malicious provider could import the sealed evaluator accessor as a
        # module side effect. The production gate must fire before its module is
        # resolved or executed at all.
        assert_pc01_page_broker_production_authorized(
            blocked_pc01_page_broker_security_binding()
        )

    def adversarial_factory_load(*_args, **_kwargs):
        calls["factory_load"] += 1
        from web_agent.eval.table2.live_page_broker_assembly import (
            process_sealed_page_evaluator_capability,
        )

        process_sealed_page_evaluator_capability()
        return _wrong_pc01_provider_factory

    monkeypatch.setattr(
        evaluation_cli, "_preflight_pc01_provider_install", blocked_preflight
    )
    monkeypatch.setattr(
        evaluation_cli, "_load_exact_provider_factory", adversarial_factory_load
    )
    with pytest.raises(SchemaError, match="externally evidenced process isolation"):
        evaluation_cli._install_pc01_operations_provider(
            tmp_path / "campaign",
            provider_factory_entrypoint=(
                "tests.table2.test_production_runner:_wrong_pc01_provider_factory"
            ),
            credential_capability_root=tmp_path / "credentials",
            credential_capability_id="fixture-credentials",
            credential_capability_version="v1",
            provider_boundary_receipt=tmp_path / "boundary.json",
        )
    assert calls["factory_load"] == 0


def test_canonical_bootstrap_records_broker_block_before_package_import() -> None:
    attestation = {
        PC01_PAGE_BROKER_SECURITY_FIELD: blocked_pc01_page_broker_security_binding()
    }
    with pytest.raises(RuntimeError, match="blocked before provider import"):
        evaluation_cli._bootstrap_assert_pc01_page_broker_isolation(attestation)


def test_evaluation_bootstrap_mirrors_process_broker_promotion_requirements() -> None:
    assert evaluation_cli.PC01_PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS == list(
        execution_guard.PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
    )


def test_direct_preflight_cannot_import_or_use_sealed_broker_capability(
    tmp_path: Path,
) -> None:
    """An isolated direct caller stops before either broker plane is imported."""

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "runner_attestation.json").write_text(
        json.dumps(
            {
                PC01_PAGE_BROKER_SECURITY_FIELD: (
                    blocked_pc01_page_broker_security_binding()
                )
            }
        ),
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[2]
    probe = """
import sys
from pathlib import Path
from scripts import run_table2_evaluation as cli

try:
    cli._preflight_pc01_provider_install(
        Path(sys.argv[1]),
        credential_capability_root=Path(sys.argv[1]) / "credentials",
        credential_capability_id="adversarial",
        credential_capability_version="v1",
    )
except RuntimeError as exc:
    assert "blocked before provider import" in str(exc)
else:
    raise AssertionError("blocked broker preflight unexpectedly returned")

for forbidden in (
    "web_agent.runtime.pc01_live_integration",
    "web_agent.eval.table2.live_page_broker_assembly",
    "web_agent.eval.table2.sealed_page_broker",
):
    assert forbidden not in sys.modules, forbidden
"""
    completed = evaluation_cli.subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _typed_bootstrap_fixture(tmp_path: Path):
    from tests.table2.test_pc01_live_integration import (
        SERVICE_URLS,
        _provider_and_context,
    )
    from web_agent.runtime import pc01_live_integration as pc01_live

    provider, runtime_context, _ = _provider_and_context(tmp_path)
    preflight_view = {
        "schema_version": "table2-pc01-runtime-preflight-view-v1",
        "source_binding_sha256": "1" * 64,
        "deployment_topology": "SINGLE_DGX_HOST",
        "validator_contract": "fixture-validator-v1",
        "expected_live_reset_task_index": 0,
        "preflight_content_sha256": "2" * 64,
        "service_url_map_content_sha256": sha256_json(SERVICE_URLS),
        "expected_dgx_runtime_identity_sha256": None,
        "expected_bridge_identity_sha256": None,
        "preflight_status": "PASS",
    }
    authority = dict(runtime_context.runtime_capability_authority)
    authority["deployment_preflight_binding_sha256"] = sha256_json(preflight_view)
    context = pc01_live.PC01ProviderBootstrapContext(
        schema_version=pc01_live.PC01_PROVIDER_BOOTSTRAP_SCHEMA_VERSION,
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
    binding = _provider_factory_binding("_wrong_pc01_provider_factory")
    binding["expected_provider_public_contract_sha256"] = (
        provider.public_contract_sha256
    )
    return context, binding, provider


def test_provider_factory_receives_oracle_free_typed_bootstrap_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_agent.runtime.pc01_live_integration import PC01ProviderBootstrapContext

    context, binding, _ = _typed_bootstrap_fixture(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    boundary = tmp_path / "boundary.json"
    boundary.write_text("{}", encoding="utf-8")
    captured: list[object] = []

    def capture(value):
        captured.append(value)
        return object()

    monkeypatch.setattr(
        evaluation_cli,
        "_preflight_pc01_provider_install",
        lambda *unused_args, **unused_kwargs: evaluation_cli._ProviderInstallPreflight(
            context=context,
            binding=binding,
            source_hashes={
                binding["source_relative_path"]: binding["source_sha256"]
            },
            campaign_state_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_validate_pc01_provider_boundary_receipt",
        lambda *unused_args, **unused_kwargs: {},
    )
    monkeypatch.setattr(
        evaluation_cli, "_load_exact_provider_factory", lambda *_args, **_kwargs: capture
    )
    with pytest.raises(RuntimeError, match="wrong exact type"):
        evaluation_cli._install_pc01_operations_provider(
            campaign,
            provider_factory_entrypoint=binding["factory_entrypoint"],
            credential_capability_root=tmp_path,
            credential_capability_id="fixture",
            credential_capability_version="v1",
            provider_boundary_receipt=boundary,
        )
    assert len(captured) == 1
    assert type(captured[0]) is PC01ProviderBootstrapContext
    serialized = json.dumps(
        evaluation_cli._bootstrap_context_identity(captured[0]), sort_keys=True
    ).lower()
    assert str(campaign) not in serialized
    assert not any(
        token in serialized
        for token in (
            "evaluator_path",
            "memory_path",
            "model_path",
            "sealed_path",
            "oracle_success",
        )
    )


def test_provider_boundary_receipt_is_generated_and_scope_honest(
    tmp_path: Path,
) -> None:
    context, binding, _ = _typed_bootstrap_fixture(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    preflight = evaluation_cli._ProviderInstallPreflight(
        context=context,
        binding=binding,
        source_hashes={binding["source_relative_path"]: binding["source_sha256"]},
        campaign_state_sha256="b" * 64,
    )
    output = evaluation_cli._write_pc01_provider_boundary_receipt(
        tmp_path / "external/boundary.json",
        campaign_root=campaign,
        preflight=preflight,
    )
    receipt = evaluation_cli._validate_pc01_provider_boundary_receipt(
        output,
        campaign_root=campaign,
        preflight=preflight,
    )
    assert receipt["claim_scope"] == "REVIEWED_CODE_ORACLE_FREE_DATAFLOW_ONLY"
    assert receipt["kernel_filesystem_sandbox"] is False
    assert receipt["campaign_directory_argument_passed"] is False
    original_bytes = output.read_bytes()
    assert evaluation_cli._write_pc01_provider_boundary_receipt(
        output,
        campaign_root=campaign,
        preflight=preflight,
    ) == output
    assert output.read_bytes() == original_bytes
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable evidence"):
        evaluation_cli._write_pc01_provider_boundary_receipt(
            output,
            campaign_root=campaign,
            preflight=preflight,
        )


@pytest.mark.parametrize(
    "candidate_kind", ("filesystem_root", "campaign_parent", "repo_parent")
)
def test_credential_capability_rejects_broad_or_overlapping_roots(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    from web_agent.runtime.pc01_live_integration import (
        PC01LiveIntegrationError,
        validate_external_credential_capability_root,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    repository = Path(__file__).resolve().parents[2]
    candidates = {
        "filesystem_root": Path("/"),
        "campaign_parent": campaign.parent,
        "repo_parent": repository.parent,
    }
    with pytest.raises(PC01LiveIntegrationError, match="root|tree-disjoint"):
        validate_external_credential_capability_root(
            candidates[candidate_kind],
            forbidden_roots=(campaign, repository),
        )


@pytest.mark.parametrize("symlink_kind", ("leaf", "parent"))
def test_credential_capability_rejects_symlink_leaf_and_parent(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    from web_agent.runtime.pc01_live_integration import (
        PC01LiveIntegrationError,
        validate_external_credential_capability_root,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    real_parent = tmp_path / "real_external"
    credentials = real_parent / "credentials"
    credentials.mkdir(parents=True)
    if symlink_kind == "leaf":
        candidate = tmp_path / "credential_link"
        candidate.symlink_to(credentials, target_is_directory=True)
    else:
        linked_parent = tmp_path / "external_parent_link"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        candidate = linked_parent / "credentials"
    with pytest.raises(PC01LiveIntegrationError, match="symlink"):
        validate_external_credential_capability_root(
            candidate,
            forbidden_roots=(campaign, Path(__file__).resolve().parents[2]),
        )


@pytest.mark.parametrize("symlink_kind", ("leaf", "parent"))
def test_provider_boundary_receipt_rejects_symlink_leaf_and_parent(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    context, binding, _ = _typed_bootstrap_fixture(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    preflight = evaluation_cli._ProviderInstallPreflight(
        context=context,
        binding=binding,
        source_hashes={binding["source_relative_path"]: binding["source_sha256"]},
        campaign_state_sha256="b" * 64,
    )
    real = evaluation_cli._write_pc01_provider_boundary_receipt(
        tmp_path / "external/real.json",
        campaign_root=campaign,
        preflight=preflight,
    )
    if symlink_kind == "leaf":
        candidate = tmp_path / "receipt_link.json"
        candidate.symlink_to(real)
    else:
        parent_link = tmp_path / "receipt_parent_link"
        parent_link.symlink_to(real.parent, target_is_directory=True)
        candidate = parent_link / real.name
    with pytest.raises(RuntimeError, match="symlink"):
        evaluation_cli._validate_pc01_provider_boundary_receipt(
            candidate,
            campaign_root=campaign,
            preflight=preflight,
        )


@pytest.mark.parametrize("symlink_kind", ("leaf", "parent"))
def test_provider_boundary_receipt_write_rejects_symlink_leaf_and_parent(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    context, binding, _ = _typed_bootstrap_fixture(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    preflight = evaluation_cli._ProviderInstallPreflight(
        context=context,
        binding=binding,
        source_hashes={binding["source_relative_path"]: binding["source_sha256"]},
        campaign_state_sha256="b" * 64,
    )
    real_parent = tmp_path / "real_output_parent"
    real_parent.mkdir()
    if symlink_kind == "leaf":
        real_target = real_parent / "existing.json"
        real_target.write_text("{}", encoding="utf-8")
        candidate = tmp_path / "output_link.json"
        candidate.symlink_to(real_target)
    else:
        parent_link = tmp_path / "output_parent_link"
        parent_link.symlink_to(real_parent, target_is_directory=True)
        candidate = parent_link / "boundary.json"
    with pytest.raises(RuntimeError, match="symlink"):
        evaluation_cli._write_pc01_provider_boundary_receipt(
            candidate,
            campaign_root=campaign,
            preflight=preflight,
        )


def test_provider_installation_receipt_is_registered_and_hash_chain_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_agent.eval.table2.common import read_jsonl
    from web_agent.runtime import pc01_live_integration as pc01_live

    context, binding, provider = _typed_bootstrap_fixture(tmp_path)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    boundary = tmp_path / "boundary.json"
    boundary.write_text("{}\n", encoding="utf-8")
    preflight = evaluation_cli._ProviderInstallPreflight(
        context=context,
        binding=binding,
        source_hashes={binding["source_relative_path"]: binding["source_sha256"]},
        campaign_state_sha256="b" * 64,
    )
    monkeypatch.setattr(
        pc01_live, "_OPERATIONS_REGISTRY", pc01_live._ImmutableOperationsRegistry()
    )
    monkeypatch.setattr(
        pc01_live,
        "validate_provider_public_contract",
        lambda *unused_args, **unused_kwargs: {},
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_preflight_pc01_provider_install",
        lambda *unused_args, **unused_kwargs: preflight,
    )
    monkeypatch.setattr(
        evaluation_cli,
        "_validate_pc01_provider_boundary_receipt",
        lambda *unused_args, **unused_kwargs: {},
    )

    def factory(_bootstrap):
        return provider

    monkeypatch.setattr(
        evaluation_cli,
        "_load_exact_provider_factory",
        lambda *unused_args, **unused_kwargs: factory,
    )
    receipt = evaluation_cli._install_pc01_operations_provider(
        campaign,
        provider_factory_entrypoint=binding["factory_entrypoint"],
        credential_capability_root=tmp_path,
        credential_capability_id="fixture",
        credential_capability_version="v1",
        provider_boundary_receipt=boundary,
    )
    assert receipt.actual_provider_public_contract_sha256 == (
        provider.public_contract_sha256
    )
    assert pc01_live._OPERATIONS_REGISTRY.require_installation_receipt() == receipt
    records = read_jsonl(campaign / "access_ledger.jsonl")
    assert [record["event_type"] for record in records] == [
        "pc01_provider_installation"
    ]
    payload = records[0]["payload"]
    assert payload["installation_receipt_sha256"] == receipt.receipt_sha256
    assert str(tmp_path) not in json.dumps(payload, sort_keys=True)


def test_production_attestation_must_include_preimport_cli_source() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    production_relative = "src/web_agent/eval/table2/production_runner.py"
    fixture_relative = Path(__file__).resolve().relative_to(repository_root).as_posix()
    rows = [
        {
            "relative_path": production_relative,
            "sha256": sha256_file(repository_root / production_relative),
        },
        {
            "relative_path": fixture_relative,
            "sha256": sha256_file(repository_root / fixture_relative),
        },
    ]
    integration = {
        "entrypoint": "tests.table2.test_production_runner:_processor",
        "source_relative_path": fixture_relative,
        "source_sha256": sha256_file(repository_root / fixture_relative),
    }
    evaluator = {
        "source_relative_path": fixture_relative,
        "source_sha256": sha256_file(repository_root / fixture_relative),
    }
    payload = {
        "schema_version": RUNNER_ATTESTATION_SCHEMA_VERSION,
        "attestation_scope": EVALUATION_RUNNER_SCOPE,
        "runner_entrypoint": PRODUCTION_RUNNER_ENTRYPOINT,
        "runtime_integration_entrypoint": integration["entrypoint"],
        "repository_commit": "f" * 40,
        "primary_source_relative_path": production_relative,
        "source_files": rows,
        "source_set_sha256": sha256_json(rows),
        "runtime_identity": {
            "runtime_integration": integration,
            "evaluator": evaluator,
        },
    }
    with pytest.raises(SchemaError, match="pre-import evaluation CLI"):
        validate_runner_attestation_payload(
            payload,
            repository_root=repository_root,
            repository_commit="f" * 40,
            expected_runtime_identity=payload["runtime_identity"],
        )


def test_production_runner_reopens_live_capability_bytes_before_launch(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evidence_root = tmp_path / "measured"
    manifest_path = evidence_root / "deployment.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(_valid_live_deployment_manifest(evidence_root), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    campaign_root = tmp_path / "campaign"
    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest_path,
        evidence_root=evidence_root,
        repository_root=repository_root,
        destination_artifact_root=campaign_root / "frozen",
    )
    runner = object.__new__(ProductionTable2Runner)
    runner.root = campaign_root
    runner.repository_root = repository_root
    runner.environment = _live_deployment_environment(staged)
    runner.manifest = {LIVE_DEPLOYMENT_BINDING_FIELD: staged.binding}
    runner._attested_source_hashes = {
        str(row["relative_path"]): str(row["sha256"])
        for row in staged.binding["capability_source_files"]
    }
    validator_relative = "src/web_agent/eval/table2/live_deployment.py"
    runner._attested_source_hashes[validator_relative] = sha256_file(
        repository_root / validator_relative
    )
    assert runner._revalidate_live_deployment().binding == staged.binding

    readiness = campaign_root / "frozen/live_deployment/evidence/deterministic_reset.json"
    readiness.write_bytes(readiness.read_bytes() + b" ")
    with pytest.raises(ProductionRunnerError, match="live-deployment preflight"):
        runner._revalidate_live_deployment()
