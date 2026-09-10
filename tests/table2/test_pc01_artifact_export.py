from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.eval.table2.common import canonical_json_bytes, sha256_bytes, sha256_file
from web_agent.eval.table2.pc01_artifacts import (
    PC01_CONFIG_NAME,
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_EXPECTED_SNAPSHOT_FILES,
    PC01_IMAGE_PROCESSOR_CLASS,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_PRIMARY_TRAIN_ROWS,
    PC01_PRIMARY_TRAIN_SHA256,
    PC01_PROCESSOR_CLASS,
    PC01_SAVED_FULL_CONFIG_NAME,
    PC01_SUPPLEMENT_TRAIN_ROWS,
    PC01_SUPPLEMENT_TRAIN_SHA256,
    PC01_TOTAL_TRAIN_ROWS,
    PC01_TRAINING_ACTION_VALUE_SCHEMA,
    PC01_TRAINING_ENVIRONMENT_SHA256,
    PC01_TRAINING_GIT_COMMIT,
    PC01_TRAINING_SOURCE_SHA256,
    PC01_TRANSFORMERS_VERSION,
    PC01ArtifactError,
    build_pc01_base_snapshot_manifest,
    build_pc01_training_action_value_evidence,
    export_pc01_artifacts,
    export_pc01_checkpoint_identity,
    load_pinned_local_processor,
    load_processor_contract,
    validate_pc01_training_sources,
)
from web_agent.eval.table2.pc01_processor_parity import (
    current_pc01_processor_parity_source_identity,
)
from web_agent.runtime.manifest import sha256_directory


def _config() -> dict:
    return {
        "name": PC01_SAVED_FULL_CONFIG_NAME,
        "seeds": [42],
        "fused_dim": 768,
        "backbone": {
            "path": "vlm",
            "family": "qwen2_vl",
            "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
            "revision": PC01_MODEL_REVISION,
            "trust_remote_code": False,
            "qlora": True,
            "min_pixels": 50_176,
            "max_pixels": 200_704,
        },
        "data": {
            "causal_routing": True,
            "use_state_after": True,
            "recovery_transitions": True,
        },
        "model": {"task_adapters": {"enabled": True}},
    }


def _checkpoint_payload(config: dict) -> dict:
    return {
        "lora": {},
        "adapter": {},
        "task_adapters": {},
        "failure": {},
        "action": {},
        "memory": {},
        "recovery_outcome": {},
        "config": config,
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict, str, str]:
    config = _config()
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"materialized-test-checkpoint\0" + b"x" * 2048)
    checkpoint_sha256 = sha256_file(checkpoint)
    config_sha256 = sha256_bytes(canonical_json_bytes(config))
    report = {
        "seed": 42,
        "stage": "full",
        "status": "PASS",
        "test_rows_read": 0,
        "selection_rule": "all_gates_then_outcome_mcc",
        "selected_checkpoint_sha256": checkpoint_sha256,
        "experiment_control": {
            "config_sha256": config_sha256,
            "checkpoint_selection_source": "original_gold_validation_only",
            "supplement_validation_selects_checkpoint": False,
            "locked_test_read": False,
        },
    }
    run_contract = {
        "candidate": {"model_id": PC01_MODEL_ID},
        "config_name": PC01_CONFIG_NAME,
        "model_revision": PC01_MODEL_REVISION,
        "seed": 42,
        "checkpoint_selection_source": "original_gold_validation_only",
        "supplement_validation_selects_checkpoint": False,
        "test_rows_read": 0,
        "git_commit": PC01_TRAINING_GIT_COMMIT,
    }
    report_path = tmp_path / "report.json"
    contract_path = tmp_path / "run_contract.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    contract_path.write_text(json.dumps(run_contract), encoding="utf-8")
    return (
        checkpoint,
        report_path,
        contract_path,
        config,
        config_sha256,
        checkpoint_sha256,
    )


class _FakeImageProcessor:
    min_pixels = 50_176
    max_pixels = 200_704
    _commit_hash = PC01_MODEL_REVISION


class _FakeProcessor:
    _commit_hash = PC01_MODEL_REVISION
    image_processor = _FakeImageProcessor()

    def __call__(self, *args, **kwargs):  # pragma: no cover - export only checks API
        return {}

    def apply_chat_template(self, *args, **kwargs):
        return "chat"

    def batch_decode(self, *args, **kwargs):  # pragma: no cover - API check only
        return ["{}"]


_FakeImageProcessor.__module__, _FakeImageProcessor.__qualname__ = (
    PC01_IMAGE_PROCESSOR_CLASS.rsplit(".", 1)
)
_FakeProcessor.__module__, _FakeProcessor.__qualname__ = PC01_PROCESSOR_CLASS.rsplit(
    ".", 1
)


def _training_environment(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "environment.json"
    path.write_text(
        json.dumps(
            {
                "python": "3.12.3",
                "torch": "2.13.0+cu130",
                "transformers": PC01_TRANSFORMERS_VERSION,
                "gpu": "test fixture",
                "gpu_total_memory_gb": 1.0,
                "git_commit": PC01_TRAINING_GIT_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    return path, sha256_file(path)


def _snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "hub" / "snapshots" / PC01_MODEL_REVISION
    return _snapshot_at(snapshot)


def _snapshot_at(snapshot: Path) -> Path:
    snapshot.mkdir(parents=True)
    for index, name in enumerate(sorted(PC01_EXPECTED_SNAPSHOT_FILES)):
        (snapshot / name).write_bytes(f"fixture-{index}-{name}".encode("utf-8"))
    return snapshot


def _training_action_value_evidence(
    tmp_path: Path,
    *,
    report: Path,
    run_contract: Path,
) -> Path:
    zero_counts = {
        "top_level": 0,
        "inputs": 0,
        "labels": 0,
        "recursive_total": 0,
    }
    value = {
        "schema_version": PC01_TRAINING_ACTION_VALUE_SCHEMA,
        "training_git_commit": PC01_TRAINING_GIT_COMMIT,
        "action_value_mode": "OMITTED_FOR_ALL_TRAINING_ROWS",
        "runtime_requirement": "OMIT_ACTION_VALUE_TEXT",
        "executed_action_type_supplied_to_processor": True,
        "action_value_supplied_to_processor": False,
        "sources": [
            {
                "source_id": "original_gold",
                "file_name": "split_train.json",
                "sha256": PC01_PRIMARY_TRAIN_SHA256,
                "rows": PC01_PRIMARY_TRAIN_ROWS,
                "action_value_key_occurrences": zero_counts,
            },
            {
                "source_id": "retry_abort_supplement_v2",
                "file_name": "supplement_train.json",
                "sha256": PC01_SUPPLEMENT_TRAIN_SHA256,
                "rows": PC01_SUPPLEMENT_TRAIN_ROWS,
                "action_value_key_occurrences": zero_counts,
            },
        ],
        "combined_train_rows": PC01_TOTAL_TRAIN_ROWS,
        "report_sha256": sha256_file(report),
        "run_contract_sha256": sha256_file(run_contract),
        "training_rows_read": PC01_TOTAL_TRAIN_ROWS,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    path = tmp_path / "training_action_value_evidence.json"
    path.write_bytes(canonical_json_bytes(value))
    return path


def _tensor_bundle(seed: str) -> dict:
    tensor_keys = {
        "attention_mask",
        "image_counts",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    }
    bundle_sha = sha256_bytes(f"bundle:{seed}".encode("utf-8"))
    return {
        "training_bundle_sha256": bundle_sha,
        "runtime_bundle_sha256": bundle_sha,
        "tensor_count": len(tensor_keys),
        "tensors": {
            name: {
                "dtype": "int64",
                "shape": [1, 1],
                "payload_sha256": sha256_bytes(
                    f"tensor:{seed}:{name}".encode("utf-8")
                ),
            }
            for name in sorted(tensor_keys)
        },
    }


def _valid_parity_receipt(**kwargs) -> dict:
    implementation = {
        "transformers_version": PC01_TRANSFORMERS_VERSION,
        "processor_class": PC01_PROCESSOR_CLASS,
        "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
        "use_fast": True,
    }
    action_types = ("CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY")
    return {
        "schema_version": "table2.pc01-processor-parity.v1",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "resolved_config_record_sha256": sha256_bytes(
            canonical_json_bytes(dict(kwargs["config"]))
        ),
        "processor_contract_sha256": kwargs["processor_contract"].record_sha256,
        "processor_artifact_manifest_sha256": kwargs[
            "processor_artifact_manifest_sha256"
        ],
        "base_snapshot_directory_payload_sha256": kwargs[
            "base_snapshot_directory_payload_sha256"
        ],
        "training_environment_sha256": kwargs["training_environment_sha256"],
        "training_environment_record_sha256": kwargs[
            "training_environment_record_sha256"
        ],
        "training_source_manifest_sha256": kwargs[
            "training_source_manifest_sha256"
        ],
        "training_action_value_evidence_sha256": kwargs[
            "training_action_value_evidence_sha256"
        ],
        "training_source_manifest": validate_pc01_training_sources(),
        "run_contract_sha256": kwargs["run_contract_sha256"],
        "implementation_identity": implementation,
        "training_processor_implementation": {
            "transformers_version": PC01_TRANSFORMERS_VERSION,
            "processor_class": PC01_PROCESSOR_CLASS,
            "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
            "factory": "web_agent.train.gold_stages.build_processor",
            "use_fast_argument": "omitted",
        },
        "runtime_processor_implementation": implementation,
        "source_identity": current_pc01_processor_parity_source_identity(),
        "fixture": {
            "schema_version": "table2.pc01-processor-parity-fixture.v1",
            "width": 64,
            "height": 48,
            "before_sha256": "1" * 64,
            "after_sha256": "2" * 64,
            "task_id": "fixture",
            "goal": "fixture",
            "url": "https://fixture.invalid",
            "regular_action": "TYPE",
            "regular_action_parameter": "text",
            "recovery_action": "PRESS_KEY",
            "recovery_action_parameter": "ESCAPE",
            "processor_action_value_omitted": True,
        },
        "streams": {
            phase: _tensor_bundle(phase) for phase in ("pre", "post", "recovery")
        },
        "action_class_matrix": {
            action_type: {
                "parameter_sha256": sha256_bytes(action_type.encode("utf-8")),
                "processor_action_value_omitted": True,
                "post": _tensor_bundle(f"{action_type}:post"),
                "recovery": _tensor_bundle(f"{action_type}:recovery"),
            }
            for action_type in action_types
        },
        "parity_verified": True,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "network_access_used": False,
    }


def test_config_only_export_authenticates_exact_checkpoint_saved_config(
    tmp_path: Path,
) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    destination = tmp_path / "identity"
    result = export_pc01_checkpoint_identity(
        checkpoint_path=checkpoint,
        report_path=report,
        run_contract_path=run_contract,
        output_dir=destination,
        expected_config_sha256=config_sha,
        expected_checkpoint_sha256=checkpoint_sha,
        checkpoint_loader=lambda _: _checkpoint_payload(config),
    )
    assert result.config_sha256 == config_sha
    assert result.checkpoint_sha256 == checkpoint_sha
    assert result.resolved_config_path.read_bytes() == canonical_json_bytes(config)
    receipt = json.loads(result.identity_receipt_path.read_text(encoding="utf-8"))
    assert receipt["checkpoint_saved_config_name"] == PC01_SAVED_FULL_CONFIG_NAME
    assert receipt["run_contract_config_name"] == PC01_CONFIG_NAME
    assert receipt["processor_validated"] is False
    assert receipt["runtime_ready"] is False
    assert receipt["network_access_used"] is False
    assert receipt["weight_updates_performed"] is False


def test_full_export_builds_canonical_processor_contract_from_local_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    snapshot = _snapshot(tmp_path)
    training_environment, training_environment_sha = _training_environment(tmp_path)
    action_value_evidence = _training_action_value_evidence(
        tmp_path,
        report=report,
        run_contract=run_contract,
    )
    monkeypatch.setattr(
        "web_agent.eval.table2.pc01_processor_parity.build_pc01_processor_parity_receipt",
        _valid_parity_receipt,
    )
    seen_kwargs: dict = {}

    def processor_loader(path: Path, kwargs: dict):
        assert path == snapshot.resolve()
        seen_kwargs.update(kwargs)
        return _FakeProcessor()

    destination = tmp_path / "full"
    result = export_pc01_artifacts(
        checkpoint_path=checkpoint,
        report_path=report,
        run_contract_path=run_contract,
        processor_source=snapshot,
        training_environment_path=training_environment,
        training_action_value_evidence_path=action_value_evidence,
        output_dir=destination,
        expected_config_sha256=config_sha,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_base_snapshot_sha256=sha256_directory(snapshot),
        checkpoint_loader=lambda _: _checkpoint_payload(config),
        processor_loader=processor_loader,
        version_loader=lambda: PC01_TRANSFORMERS_VERSION,
        expected_training_environment_sha256=training_environment_sha,
    )
    assert seen_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
        "min_pixels": 50_176,
        "max_pixels": 200_704,
    }
    contract = load_processor_contract(result.processor_contract_path)
    assert contract.processor_revision == PC01_MODEL_REVISION
    assert contract.record_sha256 == result.processor_contract_sha256
    assert result.e0_processor_contract_path.read_bytes() == (
        result.processor_contract_path.read_bytes()
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "table2.pc01-export.v3"
    assert manifest["selection_scope"] == "validation_only"
    assert manifest["test_rows_read"] == 0
    assert manifest["locked_test_rows_read"] == 0
    assert manifest["network_access_used"] is False
    assert manifest["weight_updates_performed"] is False
    assert manifest["runtime_ready"] is False
    assert manifest["processor_parity_verified"] is True
    assert manifest["training_transformers_version"] == PC01_TRANSFORMERS_VERSION
    assert manifest["runtime_processor_implementation"] == {
        "transformers_version": PC01_TRANSFORMERS_VERSION,
        "processor_class": PC01_PROCESSOR_CLASS,
        "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
        "use_fast": True,
    }
    processor_artifacts = json.loads(
        (destination / "processor_artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert processor_artifacts["schema_version"] == (
        "table2.pc01-processor-artifacts.v3"
    )
    assert processor_artifacts["implementation_identity"] == manifest[
        "runtime_processor_implementation"
    ]
    base_manifest = json.loads(
        result.base_snapshot_manifest_path.read_text(encoding="utf-8")
    )
    assert base_manifest["file_count"] == 14
    assert {item["path"] for item in base_manifest["files"]} == (
        PC01_EXPECTED_SNAPSHOT_FILES
    )
    assert {
        item["path"]
        for item in base_manifest["files"]
        if item["path"].endswith(".safetensors")
    } == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert result.processor_parity_receipt_path.is_file()
    assert result.training_action_value_evidence_path.read_bytes() == (
        action_value_evidence.read_bytes()
    )
    assert result.training_environment_path.read_bytes() == canonical_json_bytes(
        json.loads(training_environment.read_text(encoding="utf-8"))
    )
    training_sources = json.loads(
        result.training_source_manifest_path.read_text(encoding="utf-8")
    )
    assert training_sources["git_commit"] == PC01_TRAINING_GIT_COMMIT
    assert {item["path"] for item in training_sources["files"]} == {
        "src/web_agent/data/gold_dataset.py",
        "src/web_agent/data/gold_dataloader.py",
        "src/web_agent/data/recovery_transitions.py",
        "src/web_agent/train/gold_stages.py",
        "src/web_agent/models/encoders/vlm_contract.py",
    }


def test_export_rejects_report_hash_drift_before_writing(tmp_path: Path) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    value = json.loads(report.read_text(encoding="utf-8"))
    value["experiment_control"]["config_sha256"] = "0" * 64
    report.write_text(json.dumps(value), encoding="utf-8")
    destination = tmp_path / "must-not-exist"
    with pytest.raises(PC01ArtifactError, match="experiment_control"):
        export_pc01_checkpoint_identity(
            checkpoint_path=checkpoint,
            report_path=report,
            run_contract_path=run_contract,
            output_dir=destination,
            expected_config_sha256=config_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            checkpoint_loader=lambda _: _checkpoint_payload(config),
        )
    assert not destination.exists()


def test_train_only_action_value_audit_proves_omission_and_rejects_any_key(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "split_train.json"
    supplement = tmp_path / "supplement_train.json"
    primary.write_text(
        json.dumps([{"inputs": {"task": "one"}, "labels": {"action": "CLICK"}}]),
        encoding="utf-8",
    )
    supplement.write_text(
        json.dumps([{"inputs": {"task": "two"}, "labels": {"action": "TYPE"}}]),
        encoding="utf-8",
    )
    primary_sha = sha256_file(primary)
    supplement_sha = sha256_file(supplement)
    report = tmp_path / "audit_report.json"
    run_contract = tmp_path / "audit_run_contract.json"
    report.write_text(
        json.dumps(
            {
                "train_rows": 2,
                "test_rows_read": 0,
                "experiment_control": {
                    "development_split_sha256": {"train": primary_sha}
                },
                "train_distribution": {
                    "source_dataset": {
                        "original_gold": 1,
                        "retry_abort_supplement_v2": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    run_contract.write_text(
        json.dumps(
            {
                "train_rows": 2,
                "test_rows_read": 0,
                "git_commit": PC01_TRAINING_GIT_COMMIT,
            }
        ),
        encoding="utf-8",
    )

    evidence = build_pc01_training_action_value_evidence(
        primary_train_path=primary,
        supplement_train_path=supplement,
        report_path=report,
        run_contract_path=run_contract,
        expected_primary_sha256=primary_sha,
        expected_supplement_sha256=supplement_sha,
        expected_primary_rows=1,
        expected_supplement_rows=1,
    )
    assert evidence["action_value_mode"] == "OMITTED_FOR_ALL_TRAINING_ROWS"
    assert evidence["action_value_supplied_to_processor"] is False
    assert evidence["validation_rows_read"] == evidence["test_rows_read"] == 0

    supplement.write_text(
        json.dumps(
            [
                {
                    "inputs": {"task": "two"},
                    "labels": {"action": "TYPE", "nested": {"action_value": "x"}},
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(PC01ArtifactError, match="contains action_value evidence"):
        build_pc01_training_action_value_evidence(
            primary_train_path=primary,
            supplement_train_path=supplement,
            report_path=report,
            run_contract_path=run_contract,
            expected_primary_sha256=primary_sha,
            expected_supplement_sha256=sha256_file(supplement),
            expected_primary_rows=1,
            expected_supplement_rows=1,
        )


def test_full_export_rejects_unproved_processor_revision(tmp_path: Path) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    snapshot = _snapshot_at(tmp_path / "processor-without-snapshot-identity")

    class NoRevisionProcessor(_FakeProcessor):
        _commit_hash = None

    NoRevisionProcessor.image_processor = type(
        "ImageProcessor",
        (),
        {"min_pixels": 50_176, "max_pixels": 200_704, "_commit_hash": None},
    )()
    training_environment, training_environment_sha = _training_environment(tmp_path)
    action_value_evidence = _training_action_value_evidence(
        tmp_path,
        report=report,
        run_contract=run_contract,
    )
    with pytest.raises(PC01ArtifactError, match="no embedded or snapshot-path revision"):
        export_pc01_artifacts(
            checkpoint_path=checkpoint,
            report_path=report,
            run_contract_path=run_contract,
            processor_source=snapshot,
            training_environment_path=training_environment,
            training_action_value_evidence_path=action_value_evidence,
            output_dir=tmp_path / "must-not-exist",
            expected_config_sha256=config_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            checkpoint_loader=lambda _: _checkpoint_payload(config),
            processor_loader=lambda _path, _kwargs: NoRevisionProcessor(),
            version_loader=lambda: PC01_TRANSFORMERS_VERSION,
            expected_training_environment_sha256=training_environment_sha,
            expected_base_snapshot_sha256=sha256_directory(snapshot),
        )


def test_base_snapshot_manifest_matches_runtime_directory_hash(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    manifest = build_pc01_base_snapshot_manifest(snapshot)
    assert manifest["directory_payload_sha256"] == sha256_directory(snapshot)
    assert manifest["file_count"] == len(PC01_EXPECTED_SNAPSHOT_FILES) == 14
    assert manifest["total_size_bytes"] == sum(
        (snapshot / name).stat().st_size for name in PC01_EXPECTED_SNAPSHOT_FILES
    )


def test_base_snapshot_manifest_rejects_missing_extra_and_symlink_files(
    tmp_path: Path,
) -> None:
    missing_snapshot = _snapshot(tmp_path / "missing")
    (missing_snapshot / "vocab.json").unlink()
    with pytest.raises(PC01ArtifactError, match="file set differs"):
        build_pc01_base_snapshot_manifest(missing_snapshot)

    extra_snapshot = _snapshot(tmp_path / "extra")
    (extra_snapshot / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(PC01ArtifactError, match="file set differs"):
        build_pc01_base_snapshot_manifest(extra_snapshot)

    symlink_snapshot = _snapshot(tmp_path / "symlink")
    (symlink_snapshot / "nested-link").symlink_to(symlink_snapshot / "vocab.json")
    with pytest.raises(PC01ArtifactError, match="symlink"):
        build_pc01_base_snapshot_manifest(symlink_snapshot)


def test_full_export_requires_authenticated_training_environment_before_checkpoint_load(
    tmp_path: Path,
) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    checkpoint_loaded = False

    def checkpoint_loader(_path: Path) -> dict:
        nonlocal checkpoint_loaded
        checkpoint_loaded = True
        return _checkpoint_payload(config)

    with pytest.raises(PC01ArtifactError, match="requires the checked-in"):
        export_pc01_artifacts(
            checkpoint_path=checkpoint,
            report_path=report,
            run_contract_path=run_contract,
            processor_source=_snapshot(tmp_path),
            output_dir=tmp_path / "must-not-exist",
            expected_config_sha256=config_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            checkpoint_loader=checkpoint_loader,
        )
    assert checkpoint_loaded is False

    environment, environment_sha = _training_environment(tmp_path)
    action_value_evidence = _training_action_value_evidence(
        tmp_path,
        report=report,
        run_contract=run_contract,
    )
    environment.write_text("{}", encoding="utf-8")
    with pytest.raises(PC01ArtifactError, match="environment bytes differ"):
        export_pc01_artifacts(
            checkpoint_path=checkpoint,
            report_path=report,
            run_contract_path=run_contract,
            processor_source=tmp_path / "hub" / "snapshots" / PC01_MODEL_REVISION,
            training_environment_path=environment,
            training_action_value_evidence_path=action_value_evidence,
            output_dir=tmp_path / "must-not-exist-either",
            expected_config_sha256=config_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_training_environment_sha256=environment_sha,
            checkpoint_loader=checkpoint_loader,
        )
    assert checkpoint_loaded is False


def test_processor_requires_exact_transformers_and_concrete_fast_classes(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    with pytest.raises(PC01ArtifactError, match="transformers==4.57.6"):
        load_pinned_local_processor(
            _config(),
            snapshot,
            processor_loader=lambda _path, _kwargs: _FakeProcessor(),
            version_loader=lambda: "4.57.5",
            training_action_value_evidence_sha256="1" * 64,
        )

    class WrongProcessor(_FakeProcessor):
        pass

    with pytest.raises(PC01ArtifactError, match="processor class differs"):
        load_pinned_local_processor(
            _config(),
            snapshot,
            processor_loader=lambda _path, _kwargs: WrongProcessor(),
            version_loader=lambda: PC01_TRANSFORMERS_VERSION,
            training_action_value_evidence_sha256="1" * 64,
        )


def test_checkpoint_saved_full_name_is_distinct_from_run_contract_name() -> None:
    assert PC01_SAVED_FULL_CONFIG_NAME == "Y_QWEN2VL_2B_GOLD_V2_8_DGX_FULL_SEED42"
    assert PC01_CONFIG_NAME == "Y_QWEN2VL_2B_GOLD_V2_8_DGX"
    assert PC01_SAVED_FULL_CONFIG_NAME != PC01_CONFIG_NAME


def test_registered_constants_match_checked_in_pc01_report_and_contract() -> None:
    repository = Path(__file__).resolve().parents[2]
    run_root = (
        repository
        / "webagent_comparison/outputs/model_comparison"
        / "qwen2vl_2b_gold_v2_8_dgx/seed_42"
    )
    report = json.loads((run_root / "full/report.json").read_text(encoding="utf-8"))
    contract = json.loads((run_root / "run_contract.json").read_text(encoding="utf-8"))
    assert report["selected_checkpoint_sha256"] == PC01_EXPECTED_CHECKPOINT_SHA256
    assert report["experiment_control"]["config_sha256"] == PC01_EXPECTED_CONFIG_SHA256
    assert contract["config_name"] == PC01_CONFIG_NAME
    assert contract["model_revision"] == PC01_MODEL_REVISION
    assert contract["test_rows_read"] == 0
    environment = repository / "webagent_comparison/environment.json"
    assert sha256_file(environment) == PC01_TRAINING_ENVIRONMENT_SHA256
    source_manifest = validate_pc01_training_sources(repository)
    assert source_manifest["git_commit"] == contract["git_commit"]
    assert all(
        sha256_file(repository / item["path"]) == item["sha256"]
        for item in source_manifest["files"]
    )


def test_training_source_manifest_binds_recovery_transition_construction(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    recovery_source = "src/web_agent/data/recovery_transitions.py"
    assert PC01_TRAINING_SOURCE_SHA256[recovery_source] == (
        "e0145a9b81eb9d23a61d2560454d7e4b33d884ac0ee5188d259e0fe7198182db"
    )
    for relative in PC01_TRAINING_SOURCE_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository / relative).read_bytes())
    tampered = tmp_path / recovery_source
    tampered.write_bytes(tampered.read_bytes() + b"\n# parity tamper\n")
    with pytest.raises(PC01ArtifactError, match="recovery_transitions.py"):
        validate_pc01_training_sources(tmp_path)
