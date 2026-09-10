from __future__ import annotations

from pathlib import Path

import pytest

from tests.table2.test_package_validator import (
    FIXTURE_MODEL_COMPATIBILITY_SHA256_BY_MODEL,
    _valid_model_manifest,
    _valid_selection_evidence_inputs,
    _write_json,
)
from web_agent.eval.table2 import selection_evidence as selection_module
from web_agent.eval.table2.common import SchemaError, read_json, sha256_file
from web_agent.eval.table2.pc01_artifacts import (
    PC01_EXPECTED_EXPORT_MANIFEST_SHA256,
)
from web_agent.eval.table2.selection_evidence import (
    MODEL_COMPATIBILITY_EVIDENCE_ROLE,
    PC01_PROVISIONAL_SELECTION_MODE,
    stage_selection_evidence,
    validate_selection_evidence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _register_synthetic_candidate_compatibility_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "REGISTERED_MODEL_COMPATIBILITY_REPORT_SHA256_BY_MODEL",
        dict(FIXTURE_MODEL_COMPATIBILITY_SHA256_BY_MODEL),
    )


def _provisional_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict]:
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model["resolved_config_record_sha256"],
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    monkeypatch.setattr(
        selection_module,
        "PC01_EXPECTED_SELECTED_EPOCH",
        model["selected_epoch"],
    )
    monkeypatch.setattr(
        selection_module,
        "PC01_EXPECTED_CHECKPOINT_SHA256",
        model["selected_checkpoint_sha256"],
    )
    monkeypatch.setattr(
        selection_module,
        "PC01_EXPECTED_CONFIG_SHA256",
        model["resolved_config_record_sha256"],
    )
    return model, inputs


def test_selection_package_binds_six_candidate_artifacts_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, inputs = _provisional_inputs(tmp_path, monkeypatch)
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    manifest = read_json(manifest_path)
    assert len(manifest["artifacts"]) == 6
    assert manifest["model_compatibility_evidence_role"] == (
        MODEL_COMPATIBILITY_EVIDENCE_ROLE
    )
    candidate = manifest["candidates"][0]
    assert candidate["model_compatibility_report_used_for_ranking"] is False
    assert candidate["model_compatibility_report_checkpoint_runtime_proof"] is False
    assert candidate["model_compatibility_report_contains_model_identity"] is False
    assert candidate["candidate_evidence_binding_sha256"]
    for field in (
        "compatibility_implementation",
        "compatibility_report_writer",
        "compatibility_validator",
        "selection_evidence_implementation",
    ):
        assert len(manifest[field]["sha256"]) == 64
    validate_selection_evidence(
        manifest_path,
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
        expected_selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )


def test_candidate_binding_rejects_a_rehashed_artifact_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, inputs = _provisional_inputs(tmp_path, monkeypatch)
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    manifest = read_json(manifest_path)
    candidate = manifest["candidates"][0]
    contract_relative = candidate["run_contract_path"]
    contract_path = manifest_path.parent / contract_relative
    contract = read_json(contract_path)
    contract["post_freeze_note"] = "attempted rebinding"
    _write_json(contract_path, contract)
    changed_digest = sha256_file(contract_path)
    candidate["run_contract_sha256"] = changed_digest
    descriptor = next(
        row for row in manifest["artifacts"] if row["path"] == contract_relative
    )
    descriptor["sha256"] = changed_digest
    _write_json(manifest_path, manifest)

    with pytest.raises(SchemaError, match="candidate evidence closure"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifest=model,
            expected_selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
        )


def test_validator_source_identity_is_not_a_free_manifest_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, inputs = _provisional_inputs(tmp_path, monkeypatch)
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
        selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
    )
    manifest = read_json(manifest_path)
    manifest["compatibility_validator"]["sha256"] = "a" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(SchemaError, match="validator source differs"):
        validate_selection_evidence(
            manifest_path,
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
            expected_selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
        )


def test_registered_v3_export_cross_binds_pc01_run_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, inputs = _provisional_inputs(tmp_path, monkeypatch)
    model["model_evidence_bundle"]["artifacts"]["export_manifest"][
        "sha256"
    ] = PC01_EXPECTED_EXPORT_MANIFEST_SHA256

    with pytest.raises(SchemaError, match="registered v3 export run_contract"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
            selection_mode=PC01_PROVISIONAL_SELECTION_MODE,
        )


def test_final_candidate_reports_are_frozen_but_never_ranked(
    tmp_path: Path,
) -> None:
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model["resolved_config_record_sha256"],
    )
    manifest_path = stage_selection_evidence(
        selection_spec=inputs,
        spec_path=tmp_path / "handoff-input.json",
        output_dir=tmp_path / "selection",
        repository_root=REPOSITORY_ROOT,
        selected_model_manifest=model,
    )
    manifest = read_json(manifest_path)
    assert all(
        "compatibility" not in field
        for row in manifest["recomputed_decision"]["models"]
        for field in row
    )
    future = next(
        row
        for row in manifest["candidates"]
        if row["model_id"] == "qwen25vl_7b_gold_v2_8_dgx"
    )
    assert future["model_compatibility_report_identity_scope"] == (
        "source_registered_candidate_sha256"
    )
    staged_report = manifest_path.parent / future[
        "model_compatibility_report_path"
    ]
    staged_report.write_bytes(staged_report.read_bytes() + b"\n")
    with pytest.raises(SchemaError, match="artifact hash mismatch"):
        validate_selection_evidence(
            manifest_path,
            selected_model_manifest=model,
        )


def test_final_selection_rejects_unregistered_pending_candidate_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model["resolved_config_record_sha256"],
    )
    monkeypatch.setattr(
        selection_module,
        "REGISTERED_MODEL_COMPATIBILITY_REPORT_SHA256_BY_MODEL",
        {
            "qwen2vl_2b_gold_v2_8_dgx": (
                FIXTURE_MODEL_COMPATIBILITY_SHA256_BY_MODEL[
                    "qwen2vl_2b_gold_v2_8_dgx"
                ]
            )
        },
    )

    with pytest.raises(SchemaError, match="not source-preregistered"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )


def test_final_selection_rejects_reused_candidate_report_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model["resolved_config_record_sha256"],
    )
    repeated = FIXTURE_MODEL_COMPATIBILITY_SHA256_BY_MODEL[
        "qwen2vl_2b_gold_v2_8_dgx"
    ]
    monkeypatch.setattr(
        selection_module,
        "REGISTERED_MODEL_COMPATIBILITY_REPORT_SHA256_BY_MODEL",
        {model_id: repeated for model_id in FIXTURE_MODEL_COMPATIBILITY_SHA256_BY_MODEL},
    )

    with pytest.raises(SchemaError, match="distinct source-registered"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )


def test_final_selection_rejects_misassigned_candidate_report_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = read_json(_valid_model_manifest(tmp_path / "model.json"))
    inputs = _valid_selection_evidence_inputs(
        tmp_path,
        selected_checkpoint_sha256=model["selected_checkpoint_sha256"],
        selected_resolved_config_sha256=model["resolved_config_record_sha256"],
    )
    swapped = dict(FIXTURE_MODEL_COMPATIBILITY_SHA256_BY_MODEL)
    swapped["qwen25vl_7b_gold_v2_8_dgx"], swapped[
        "internvl35_8b_gold_v2_8_dgx"
    ] = (
        swapped["internvl35_8b_gold_v2_8_dgx"],
        swapped["qwen25vl_7b_gold_v2_8_dgx"],
    )
    monkeypatch.setattr(
        selection_module,
        "REGISTERED_MODEL_COMPATIBILITY_REPORT_SHA256_BY_MODEL",
        swapped,
    )

    with pytest.raises(SchemaError, match="differs from source registration"):
        stage_selection_evidence(
            selection_spec=inputs,
            spec_path=tmp_path / "handoff-input.json",
            output_dir=tmp_path / "selection",
            repository_root=REPOSITORY_ROOT,
            selected_model_manifest=model,
        )
