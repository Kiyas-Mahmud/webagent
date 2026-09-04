from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.run_table2_evaluation as evaluation_cli
from web_agent.eval.table2 import package_validator as package_validator_module
from web_agent.eval.table2 import production_runner as production_runner_module

from web_agent.eval.table2 import execution_guard
from web_agent.eval.table2.common import SchemaError, sha256_file, sha256_json
from web_agent.eval.table2.execution_guard import (
    EVALUATION_RUNNER_SCOPE,
    PRODUCTION_RUNNER_ENTRYPOINT,
    RUNNER_ATTESTATION_SCHEMA_VERSION,
    assert_clean_git_checkout,
    validate_attested_callable_source,
    validate_runner_attestation_payload,
)
from web_agent.eval.table2.live_deployment import (
    LIVE_DEPLOYMENT_BINDING_FIELD,
    stage_pc01_live_deployment_package,
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
    WebArenaRuntimeBinding,
    _ActiveWebArenaSession,
    _load_processor_contract,
    _normal_task_specification,
    _validated_efficiency,
    create_runner,
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
    }
    assert not names & {
        "campaign_dir",
        "resolved_task_snapshot_path",
        "memory_store_paths",
        "sealed_root",
        "verifier_sink",
    }


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
    dependency_lock = campaign / "frozen/dependency.lock"
    dependency_lock.write_bytes(b"locked-dependencies\n")
    (campaign / "frozen/environment.json").write_text(
        json.dumps(
            {
                "dependency_lock_relative_path": "dependency.lock",
                "dependency_lock_sha256": sha256_file(dependency_lock),
            }
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
                "campaign_mode": "evaluation",
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
    runner.environment = {LIVE_DEPLOYMENT_BINDING_FIELD: staged.binding}
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
