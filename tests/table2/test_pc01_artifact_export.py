from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.eval.table2.common import canonical_json_bytes, sha256_bytes, sha256_file
from web_agent.eval.table2.pc01_artifacts import (
    PC01_CONFIG_NAME,
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_SAVED_FULL_CONFIG_NAME,
    PC01ArtifactError,
    export_pc01_artifacts,
    export_pc01_checkpoint_identity,
    load_processor_contract,
)


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
) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    snapshot = tmp_path / "hub" / "snapshots" / PC01_MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "preprocessor_config.json").write_text(
        '{"processor":"qwen2-vl"}', encoding="utf-8"
    )
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
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
        output_dir=destination,
        expected_config_sha256=config_sha,
        expected_checkpoint_sha256=checkpoint_sha,
        checkpoint_loader=lambda _: _checkpoint_payload(config),
        processor_loader=processor_loader,
    )
    assert seen_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
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
    assert manifest["selection_scope"] == "validation_only"
    assert manifest["test_rows_read"] == 0
    assert manifest["locked_test_rows_read"] == 0
    assert manifest["network_access_used"] is False
    assert manifest["weight_updates_performed"] is False


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


def test_full_export_rejects_unproved_processor_revision(tmp_path: Path) -> None:
    checkpoint, report, run_contract, config, config_sha, checkpoint_sha = _inputs(
        tmp_path
    )
    snapshot = tmp_path / "processor-without-snapshot-identity"
    snapshot.mkdir()
    (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    class NoRevisionProcessor(_FakeProcessor):
        _commit_hash = None

    NoRevisionProcessor.image_processor = type(
        "ImageProcessor",
        (),
        {"min_pixels": 50_176, "max_pixels": 200_704, "_commit_hash": None},
    )()
    with pytest.raises(PC01ArtifactError, match="no embedded or snapshot-path revision"):
        export_pc01_artifacts(
            checkpoint_path=checkpoint,
            report_path=report,
            run_contract_path=run_contract,
            processor_source=snapshot,
            output_dir=tmp_path / "must-not-exist",
            expected_config_sha256=config_sha,
            expected_checkpoint_sha256=checkpoint_sha,
            checkpoint_loader=lambda _: _checkpoint_payload(config),
            processor_loader=lambda _path, _kwargs: NoRevisionProcessor(),
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
