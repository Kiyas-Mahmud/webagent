from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from web_agent.eval.table2 import package_validator as validator
from web_agent.eval.table2.common import (
    SchemaError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from web_agent.eval.table2.pc01_checkpoint_compatibility import (
    PC01CheckpointCompatibilityError,
    _SOURCE_PATHS,
)
from tests.table2.test_pc01_checkpoint_compatibility import _receipt


def _write_canonical(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o444)
    return path


def _source_bound_receipt(root: Path, *, commit: str) -> dict:
    receipt = _receipt()
    rows = []
    for index, relative in enumerate(_SOURCE_PATHS):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{index}".encode("utf-8"))
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    receipt["source_attestation"].update(
        {
            "git_commit": commit,
            "source_files": rows,
            "source_manifest_sha256": sha256_bytes(canonical_json_bytes(rows)),
        }
    )
    receipt["fixture"]["generator_source_sha256"] = rows[
        _SOURCE_PATHS.index(
            "src/web_agent/eval/table2/pc01_checkpoint_compatibility.py"
        )
    ]["sha256"]
    return receipt


def _cross_binding_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict, str]:
    commit = "1" * 40
    source_root = tmp_path / "source"
    receipt = _source_bound_receipt(source_root, commit=commit)

    payload_root = tmp_path / "payloads"
    selected = payload_root / "selected.ckpt"
    processor = payload_root / "processor.json"
    backbone = payload_root / "backbone.bin"
    for path, payload in (
        (selected, b"selected"),
        (processor, b"processor"),
        (backbone, b"backbone"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    evidence_root = tmp_path / "evidence"
    evidence_paths: dict[str, Path] = {}
    for role in validator.MODEL_EVIDENCE_ROLES:
        if role == "export_manifest":
            continue
        source = evidence_root / validator.MODEL_EVIDENCE_FILENAMES[role]
        _write_canonical(source, {"role": role})
        evidence_paths[role] = source

    report = _write_canonical(tmp_path / "selection/full_report.json", {"ok": 1})
    contract = _write_canonical(tmp_path / "selection/run_contract.json", {"ok": 2})
    export = {
        "training_environment_source_sha256": "f" * 64,
        "report_sha256": sha256_file(report),
        "run_contract_sha256": sha256_file(contract),
    }
    export_path = _write_canonical(
        evidence_root / "pc01_export_manifest.json", export
    )
    evidence_paths["export_manifest"] = export_path

    executable = {
        "selected_checkpoint": (
            selected,
            {"sha256": sha256_file(selected)},
        ),
        "processor_contract": (
            processor,
            {"sha256": sha256_file(processor)},
        ),
        "e0_backbone": (
            backbone,
            {"sha256": sha256_file(backbone)},
        ),
    }
    evidence = {
        role: (path, {"sha256": sha256_file(path)})
        for role, path in evidence_paths.items()
    }
    model = {
        "selected_model_id": receipt["model_id"],
        "model_seed": receipt["model_seed"],
        "selected_epoch": receipt["checkpoint_epoch"],
        "resolved_config_record_sha256": "e" * 64,
    }
    model_path = _write_canonical(tmp_path / "model.json", model)
    selection = {
        "selected_model_id": receipt["model_id"],
        "candidates": [
            {
                "model_id": receipt["model_id"],
                "full_report_path": report.relative_to(report.parent).as_posix(),
                "full_report_sha256": sha256_file(report),
                "run_contract_path": contract.relative_to(contract.parent).as_posix(),
                "run_contract_sha256": sha256_file(contract),
            }
        ],
    }
    # Both files are direct children of the selection package root.
    selection_path = _write_canonical(
        report.parent / "manifest.json", selection
    )

    receipt["artifact_bindings"] = {
        "export_manifest_sha256": sha256_file(export_path),
        "checkpoint_sha256": sha256_file(selected),
        "checkpoint_size_bytes": selected.stat().st_size,
        "resolved_config_sha256": model["resolved_config_record_sha256"],
        "processor_contract_sha256": sha256_file(processor),
        "processor_artifact_manifest_sha256": sha256_file(
            evidence_paths["processor_artifact_manifest"]
        ),
        "processor_parity_receipt_sha256": sha256_file(
            evidence_paths["processor_parity_receipt"]
        ),
        "base_snapshot_manifest_sha256": sha256_file(
            evidence_paths["base_snapshot_manifest"]
        ),
        "base_snapshot_directory_payload_sha256": sha256_file(backbone),
        "training_environment_record_sha256": sha256_file(
            evidence_paths["training_environment"]
        ),
        "training_environment_source_sha256": export[
            "training_environment_source_sha256"
        ],
        "training_source_manifest_sha256": sha256_file(
            evidence_paths["training_source_manifest"]
        ),
        "training_action_value_evidence_sha256": sha256_file(
            evidence_paths["training_action_value_evidence"]
        ),
        "report_sha256": sha256_file(report),
        "run_contract_sha256": sha256_file(contract),
    }
    receipt_path = _write_canonical(tmp_path / "receipt.json", receipt)

    def structural(value, *, expected_source_commit=None):
        if value["source_attestation"]["git_commit"] != expected_source_commit:
            raise PC01CheckpointCompatibilityError("another source commit")
        return dict(value)

    monkeypatch.setattr(
        validator,
        "validate_pc01_checkpoint_compatibility_receipt",
        structural,
    )
    monkeypatch.setattr(
        validator,
        "_validate_model_artifact_payloads",
        lambda *_args, **_kwargs: executable,
    )
    monkeypatch.setattr(
        validator,
        "_validate_model_evidence_bundle",
        lambda *_args, **_kwargs: evidence,
    )
    return receipt_path, source_root, model_path, receipt, commit


def test_checkpoint_readiness_cross_binds_every_artifact_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, source_root, model_path, receipt, commit = _cross_binding_fixture(
        tmp_path, monkeypatch
    )
    selection_path = tmp_path / "selection/manifest.json"
    _, binding, source_files = validator._validate_pc01_checkpoint_compatibility_readiness(
        receipt_path,
        repository_root=source_root,
        expected_source_commit=commit,
        model_manifest_path=model_path,
        selection_evidence_path=selection_path,
    )
    assert len(source_files) == 11
    assert binding["receipt_sha256"] == sha256_file(receipt_path)
    assert binding["artifact_bindings"] == receipt["artifact_bindings"]

    for field in receipt["artifact_bindings"]:
        rebound = deepcopy(receipt)
        if field == "checkpoint_size_bytes":
            rebound["artifact_bindings"][field] += 1
        else:
            rebound["artifact_bindings"][field] = "0" * 64
        rebound_path = tmp_path / f"rebound-{field}.json"
        _write_canonical(rebound_path, rebound)
        with pytest.raises(SchemaError, match="rebound"):
            validator._validate_pc01_checkpoint_compatibility_readiness(
                rebound_path,
                repository_root=source_root,
                expected_source_commit=commit,
                model_manifest_path=model_path,
                selection_evidence_path=selection_path,
            )


def test_checkpoint_receipt_rejects_reformat_write_access_and_source_tamper(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    receipt = _source_bound_receipt(source_root, commit="1" * 40)
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    reformatted.chmod(0o444)
    with pytest.raises(SchemaError, match="exact canonical"):
        validator._read_canonical_pc01_checkpoint_compatibility_receipt(
            reformatted
        )

    writable = _write_canonical(tmp_path / "writable.json", receipt)
    writable.chmod(0o644)
    with pytest.raises(SchemaError, match="read-only"):
        validator._read_canonical_pc01_checkpoint_compatibility_receipt(writable)

    for source_relative in _SOURCE_PATHS:
        source_path = source_root / source_relative
        original = source_path.read_bytes()
        source_path.write_bytes(b"tampered")
        with pytest.raises(SchemaError, match="source bytes differ"):
            validator._pc01_checkpoint_compatibility_source_files(
                receipt,
                repository_root=source_root,
            )
        source_path.write_bytes(original)
