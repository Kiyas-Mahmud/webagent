from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import pytest

from web_agent.eval.table2.common import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from web_agent.eval.table2.pc01_artifacts import (
    PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_TRAINING_ENVIRONMENT_SHA256,
)
from web_agent.eval.table2.pc01_checkpoint_compatibility import (
    PC01_CHECKPOINT_COMPATIBILITY_GATE_ID,
    PC01_CHECKPOINT_COMPATIBILITY_ROLE,
    PC01_CHECKPOINT_COMPATIBILITY_SCHEMA,
    PC01_REGISTERED_GPU_MEMORY_BYTES,
    PC01_REGISTERED_GPU_NAME,
    PC01_REGISTERED_PYTHON_VERSION,
    PC01_REGISTERED_TORCH_VERSION,
    PC01_REGISTERED_EXPORT_MANIFEST_SHA256,
    PC01CheckpointCompatibilityError,
    _EXPORT_FILES,
    _MANIFEST_PAYLOAD_FILES,
    _SOURCE_PATHS,
    _float32_embedding_tensor_sha256,
    _state_mapping_identity,
    _validate_export_manifest,
    _write_receipt,
    run_pc01_checkpoint_compatibility,
    validate_pc01_checkpoint_compatibility_receipt,
)
from web_agent.runtime.contracts import float32_vector_sha256


SHA_A = "a" * 64
SHA_B = "b" * 64


def _receipt() -> dict:
    artifact_hashes = {
        "export_manifest_sha256": PC01_REGISTERED_EXPORT_MANIFEST_SHA256,
        "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": 291_071_781,
        "resolved_config_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "processor_contract_sha256": SHA_A,
        "processor_artifact_manifest_sha256": SHA_A,
        "processor_parity_receipt_sha256": SHA_A,
        "base_snapshot_manifest_sha256": SHA_A,
        "base_snapshot_directory_payload_sha256": (
            PC01_EXPECTED_BASE_SNAPSHOT_SHA256
        ),
        "training_environment_record_sha256": SHA_A,
        "training_environment_source_sha256": (
            PC01_TRAINING_ENVIRONMENT_SHA256
        ),
        "training_source_manifest_sha256": SHA_A,
        "training_action_value_evidence_sha256": SHA_A,
        "report_sha256": SHA_A,
        "run_contract_sha256": SHA_A,
    }
    selected = {
        "class": "web_agent.models.model.WebAgentModel",
        "eval_mode": True,
        "frozen": True,
        "all_parameters_cuda": True,
        "parameter_count": 1,
        "module_count": 1,
    }
    e0 = {
        **selected,
        "class": "transformers.Qwen2VLForConditionalGeneration",
        "adaptation_loaded": False,
        "task_heads_loaded": False,
        "forbidden_state_hits": [],
    }
    source_files = [
        {"path": path, "sha256": SHA_A, "size_bytes": 10 + index}
        for index, path in enumerate(_SOURCE_PATHS)
    ]
    source_files[
        _SOURCE_PATHS.index(
            "src/web_agent/eval/table2/pc01_checkpoint_compatibility.py"
        )
    ]["sha256"] = SHA_B
    state_identity = {
        "sha256": SHA_A,
        "tensor_count": 1,
        "total_tensor_bytes": 4,
    }
    roles = {
        role: {
            "checkpoint": dict(state_identity),
            "loaded": dict(state_identity),
            "exact_match": True,
        }
        for role in (
            "lora",
            "adapter",
            "task_adapters",
            "failure",
            "action",
            "memory",
            "recovery_outcome",
        )
    }
    inference_common = {
        "direct_raw_tensor_sha256": SHA_A,
        "repeat_raw_tensor_sha256": SHA_A,
        "direct_semantic_sha256": SHA_B,
        "runtime_semantic_sha256": SHA_B,
        "repeat_semantic_sha256": SHA_B,
        "direct_runtime_exact": True,
        "repeat_exact": True,
    }
    embedding_values = [1.0] + [0.0] * 767
    return {
        "schema_version": PC01_CHECKPOINT_COMPATIBILITY_SCHEMA,
        "artifact_role": PC01_CHECKPOINT_COMPATIBILITY_ROLE,
        "gate_id": PC01_CHECKPOINT_COMPATIBILITY_GATE_ID,
        "status": "PASS",
        "scope": "PROVISIONAL_ENGINEERING_PILOT_COMPATIBILITY_ONLY",
        "paper_table_status": "N/R",
        "model_id": "qwen2vl_2b_gold_v2_8_dgx",
        "model_seed": 42,
        "checkpoint_epoch": 6,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_bindings": artifact_hashes,
        "source_attestation": {
            "git_commit": "1" * 40,
            "git_clean": True,
            "source_file_count": len(_SOURCE_PATHS),
            "source_files": source_files,
            "source_manifest_sha256": sha256_bytes(
                canonical_json_bytes(source_files)
            ),
        },
        "host_attestation": {
            "host_role": "DGX_MODEL_PLANE",
            "architecture": "aarch64",
            "python_version": PC01_REGISTERED_PYTHON_VERSION,
            "torch_version": PC01_REGISTERED_TORCH_VERSION,
            "cuda_runtime_version": "13.0",
            "cudnn_version": 1,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_index": 0,
            "cuda_device_name": PC01_REGISTERED_GPU_NAME,
            "cuda_device_capability": [12, 1],
            "cuda_device_total_memory_bytes": PC01_REGISTERED_GPU_MEMORY_BYTES,
            "package_versions": {
                "accelerate": "1",
                "bitsandbytes": "1",
                "peft": "1",
                "transformers": "4.57.6",
            },
        },
        "model_load": {
            "selected_checkpoint": selected,
            "e0_unadapted_base": e0,
        },
        "checkpoint_restore": {
            "roles": roles,
            "all_roles_exact": True,
            "selected_state_sha256_before": SHA_B,
            "selected_state_sha256_after": SHA_B,
            "selected_state_tensor_count": 1,
            "selected_state_tensor_bytes": 4,
            "state_unchanged_after_inference": True,
        },
        "fixture": {
            "fixture_id": "pc01-causal-three-state-ppm-v1",
            "generator_source_sha256": SHA_B,
            "width": 224,
            "height": 224,
            "screenshot_sha256": ["c" * 64, "d" * 64, "e" * 64],
            "task_sha256": SHA_A,
            "transition_sha256": SHA_A,
            "recovery_transition_sha256": SHA_A,
            "embedding_request_sha256": SHA_B,
            "contains_webarena_task_content": False,
            "contains_oracle_content": False,
        },
        "inference": {
            "pre_action": {
                "processed_batch_sha256": "c" * 64,
                **inference_common,
            },
            "post_action_diagnosis": {
                "processed_pre_batch_sha256": "c" * 64,
                "processed_post_batch_sha256": "d" * 64,
                **inference_common,
            },
            "executed_recovery_assessment": {
                "processed_pre_batch_sha256": "e" * 64,
                "processed_post_batch_sha256": "f" * 64,
                "processed_recovery_batch_sha256": "9" * 64,
                **inference_common,
            },
            "p4_memory_embedding": {
                "processed_post_batch_sha256": "d" * 64,
                "shape": [1, 768],
                "dimension": 768,
                "dtype_after_export": "float32",
                "embedding_sha256": float32_vector_sha256(embedding_values),
                "embedding_values": embedding_values,
                "direct_tensor_sha256": _float32_embedding_tensor_sha256(
                    embedding_values
                ),
                "repeat_tensor_sha256": _float32_embedding_tensor_sha256(
                    embedding_values
                ),
                "l2_norm": 1.0,
                "finite": True,
                "nonzero": True,
                "direct_runtime_exact": True,
                "repeat_exact": True,
                "request_sha256": SHA_B,
            },
            "checkpoint_forward_executed": True,
            "runtime_callbacks_executed": True,
            "all_parity_checks_passed": True,
        },
        "causal_and_scientific_boundaries": {
            "offline_mode_enforced": True,
            "network_access_used": False,
            "weight_updates_performed": False,
            "training_rows_read": 0,
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "webarena_tasks_executed": 0,
            "task_registry_frozen": False,
            "pc01_final_backbone_promoted": False,
            "table2_populated": False,
            "action_value_text_omitted": True,
        },
    }


def test_module_import_remains_dependency_lazy() -> None:
    repository = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        "import web_agent.eval.table2.pc01_checkpoint_compatibility; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'peft' not in sys.modules"
    )
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(repository / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_cli_rejects_caller_selected_export_manifest_identity(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    output = tmp_path / "must-not-exist.json"
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(repository / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_pc01_checkpoint_compatibility.py"),
            "--export-dir",
            str(tmp_path / "export"),
            "--checkpoint",
            str(tmp_path / "checkpoint.ckpt"),
            "--base-snapshot",
            str(tmp_path / "snapshot"),
            "--report",
            str(tmp_path / "report.json"),
            "--run-contract",
            str(tmp_path / "run-contract.json"),
            "--repository-root",
            str(repository),
            "--expected-source-commit",
            "1" * 40,
            "--output",
            str(output),
            "--expected-export-manifest-sha256",
            SHA_B,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --expected-export-manifest-sha256" in result.stderr
    assert not output.exists()


def test_valid_receipt_stays_pilot_only_and_binds_export_and_source() -> None:
    receipt = _receipt()
    assert validate_pc01_checkpoint_compatibility_receipt(
        receipt,
        expected_source_commit="1" * 40,
    ) == receipt


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("paper_table_status",), "READY", "paper_table_status"),
        (
            ("model_load", "e0_unadapted_base", "adaptation_loaded"),
            True,
            "adaptation_loaded",
        ),
        (
            ("inference", "p4_memory_embedding", "dimension"),
            512,
            "dimension",
        ),
        (
            ("causal_and_scientific_boundaries", "locked_test_rows_read"),
            1,
            "locked_test_rows_read",
        ),
        (
            ("checkpoint_restore", "selected_state_sha256_after"),
            SHA_A,
            "restore/immutability",
        ),
    ),
)
def test_receipt_rejects_false_or_scientifically_expansive_claims(
    path: tuple[str, ...], value: object, message: str
) -> None:
    receipt = deepcopy(_receipt())
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(PC01CheckpointCompatibilityError, match=message):
        validate_pc01_checkpoint_compatibility_receipt(receipt)


def test_receipt_rejects_another_selected_export_or_source_commit() -> None:
    receipt = _receipt()
    receipt["artifact_bindings"]["export_manifest_sha256"] = SHA_B
    with pytest.raises(PC01CheckpointCompatibilityError, match="export_manifest_sha256"):
        validate_pc01_checkpoint_compatibility_receipt(receipt)
    receipt = _receipt()
    with pytest.raises(PC01CheckpointCompatibilityError, match="another source commit"):
        validate_pc01_checkpoint_compatibility_receipt(
            receipt, expected_source_commit="2" * 40
        )


def test_receipt_rejects_dropped_or_unbound_nested_evidence() -> None:
    cases: list[tuple[dict, str]] = []

    receipt = deepcopy(_receipt())
    receipt["artifact_bindings"]["unexpected"] = SHA_A
    cases.append((receipt, "artifact binding fields changed"))

    receipt = deepcopy(_receipt())
    receipt["source_attestation"]["source_files"].pop()
    cases.append((receipt, "source-file coverage changed"))

    receipt = deepcopy(_receipt())
    receipt["source_attestation"]["source_files"][0]["sha256"] = SHA_B
    cases.append((receipt, "source manifest hash differs"))

    receipt = deepcopy(_receipt())
    receipt["host_attestation"]["package_versions"].pop("peft")
    cases.append((receipt, "package-version coverage changed"))

    receipt = deepcopy(_receipt())
    receipt["checkpoint_restore"]["roles"].pop("lora")
    cases.append((receipt, "restore role coverage changed"))

    receipt = deepcopy(_receipt())
    receipt["checkpoint_restore"]["roles"]["memory"]["loaded"]["sha256"] = SHA_B
    cases.append((receipt, "role memory is not exact"))

    receipt = deepcopy(_receipt())
    receipt["fixture"]["generator_source_sha256"] = SHA_A
    cases.append((receipt, "generator differs"))

    receipt = deepcopy(_receipt())
    receipt["inference"]["pre_action"].pop("processed_batch_sha256")
    cases.append((receipt, "pre-action inference evidence fields changed"))

    receipt = deepcopy(_receipt())
    receipt["inference"]["executed_recovery_assessment"][
        "repeat_raw_tensor_sha256"
    ] = SHA_B
    cases.append((receipt, "direct/repeat raw tensor hashes differ"))

    receipt = deepcopy(_receipt())
    receipt["inference"]["post_action_diagnosis"][
        "runtime_semantic_sha256"
    ] = SHA_A
    cases.append((receipt, "semantic hashes differ"))

    receipt = deepcopy(_receipt())
    receipt["inference"]["p4_memory_embedding"]["embedding_values"][0] = 2.0
    cases.append((receipt, "P4 embedding hash/norm/request bindings are invalid"))

    receipt = deepcopy(_receipt())
    receipt["inference"]["p4_memory_embedding"]["request_sha256"] = SHA_A
    cases.append((receipt, "P4 embedding hash/norm/request bindings are invalid"))

    for candidate, message in cases:
        with pytest.raises(PC01CheckpointCompatibilityError, match=message):
            validate_pc01_checkpoint_compatibility_receipt(candidate)


def test_receipt_writer_is_canonical_read_only_and_never_overwrites(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    destination = _write_receipt(tmp_path / "receipt.json", receipt)
    assert destination.read_bytes() == canonical_json_bytes(receipt)
    assert destination.stat().st_mode & 0o222 == 0
    with pytest.raises(PC01CheckpointCompatibilityError, match="overwrite"):
        _write_receipt(destination, receipt)


def test_state_mapping_hash_covers_keys_shapes_dtypes_and_bytes() -> None:
    torch = pytest.importorskip("torch")
    first = {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "bias": torch.tensor([3.0], dtype=torch.float16),
    }
    same = {key: value.clone() for key, value in first.items()}
    changed = {key: value.clone() for key, value in first.items()}
    changed["weight"][0, 0] = 9.0
    identity = _state_mapping_identity(first, torch=torch)
    assert identity == _state_mapping_identity(same, torch=torch)
    assert identity["sha256"] != _state_mapping_identity(
        changed, torch=torch
    )["sha256"]
    assert identity["tensor_count"] == 2
    with pytest.raises(PC01CheckpointCompatibilityError, match="not a tensor"):
        _state_mapping_identity({"weight": "forged"}, torch=torch)


def _fake_export_manifest(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    export = tmp_path / "export"
    export.mkdir()
    for name in sorted(_MANIFEST_PAYLOAD_FILES):
        (export / name).write_bytes(b"{}")
    checkpoint = tmp_path / "selected.ckpt"
    report = tmp_path / "report.json"
    contract = tmp_path / "run_contract.json"
    checkpoint.write_bytes(b"checkpoint")
    report.write_bytes(b"report")
    contract.write_bytes(b"contract")
    files = {
        name: {
            "sha256": sha256_file(export / name),
            "size_bytes": (export / name).stat().st_size,
        }
        for name in sorted(_MANIFEST_PAYLOAD_FILES)
    }
    value = {
        "schema_version": "table2.pc01-export.v3",
        "model_id": "qwen2vl_2b_gold_v2_8_dgx",
        "config_name": "Y_QWEN2VL_2B_GOLD_V2_8_DGX",
        "model_seed": 42,
        "model_revision": "895c3a49bc3fa70a340399125c650a463535e71c",
        "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "resolved_config_record_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "resolved_config_payload_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "processor_contract_sha256": SHA_A,
        "processor_artifact_manifest_sha256": files[
            "processor_artifact_manifest.json"
        ]["sha256"],
        "processor_parity_receipt_sha256": files[
            "processor_parity_receipt.json"
        ]["sha256"],
        "processor_parity_verified": True,
        "base_snapshot_manifest_sha256": files[
            "base_snapshot_manifest.json"
        ]["sha256"],
        "base_snapshot_directory_payload_sha256": (
            PC01_EXPECTED_BASE_SNAPSHOT_SHA256
        ),
        "base_snapshot_file_count": 14,
        "base_snapshot_copied_into_export": False,
        "training_environment_source_sha256": (
            PC01_TRAINING_ENVIRONMENT_SHA256
        ),
        "training_environment_record_sha256": files[
            "training_environment.json"
        ]["sha256"],
        "training_source_manifest_sha256": files[
            "training_source_manifest.json"
        ]["sha256"],
        "training_action_value_evidence_sha256": files[
            "training_action_value_evidence.json"
        ]["sha256"],
        "training_transformers_version": "4.57.6",
        "runtime_processor_implementation": {
            "transformers_version": "4.57.6",
            "processor_class": "test.Processor",
            "image_processor_class": "test.ImageProcessor",
            "use_fast": True,
        },
        "report_sha256": sha256_file(report),
        "run_contract_sha256": sha256_file(contract),
        "selection_scope": "validation_only",
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "network_access_used": False,
        "weight_updates_performed": False,
        "runtime_ready": False,
        "files": files,
    }
    manifest = export / "pc01_export_manifest.json"
    manifest.write_bytes(canonical_json_bytes(value))
    assert {path.name for path in export.iterdir()} == _EXPORT_FILES
    return export, checkpoint, report, contract, value


def test_export_manifest_authentication_detects_payload_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export, checkpoint, report, contract, value = _fake_export_manifest(tmp_path)
    monkeypatch.setattr(
        "web_agent.eval.table2.pc01_checkpoint_compatibility."
        "PC01_EXPECTED_CHECKPOINT_SIZE_BYTES",
        checkpoint.stat().st_size,
    )
    identity = sha256_file(export / "pc01_export_manifest.json")
    _validate_export_manifest(
        value,
        export_dir=export,
        expected_manifest_sha256=identity,
        checkpoint_path=checkpoint,
        report_path=report,
        run_contract_path=contract,
    )
    (export / "processor_parity_receipt.json").write_bytes(b'{"tampered":true}')
    with pytest.raises(PC01CheckpointCompatibilityError, match="payload differs"):
        _validate_export_manifest(
            value,
            export_dir=export,
            expected_manifest_sha256=identity,
            checkpoint_path=checkpoint,
            report_path=report,
            run_contract_path=contract,
        )


def test_public_runner_fails_on_non_dgx_before_reading_artifacts_or_writing_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "must-not-exist.json"
    calls: list[str] = []

    def reject_host():
        raise PC01CheckpointCompatibilityError("requires CUDA on the DGX model plane")

    def forbidden_bundle(**kwargs):
        calls.append("bundle")
        raise AssertionError(kwargs)

    monkeypatch.setattr(
        "web_agent.eval.table2.pc01_checkpoint_compatibility._require_registered_dgx_cuda",
        reject_host,
    )
    monkeypatch.setattr(
        "web_agent.eval.table2.pc01_checkpoint_compatibility._validate_pc01_export_bundle",
        forbidden_bundle,
    )
    with pytest.raises(PC01CheckpointCompatibilityError, match="requires CUDA"):
        run_pc01_checkpoint_compatibility(
            export_dir=tmp_path / "missing-export",
            checkpoint_path=tmp_path / "missing.ckpt",
            base_snapshot_path=tmp_path / "missing-base",
            report_path=tmp_path / "missing-report.json",
            run_contract_path=tmp_path / "missing-contract.json",
            repository_root=tmp_path,
            expected_source_commit="1" * 40,
            output_path=output,
        )
    assert calls == []
    assert not output.exists()
