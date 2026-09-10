from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_file
from web_agent.eval.table2.handoff_authority import (
    validate_handoff_freeze_authority,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PAPER_CLAIM_REGISTRY = (
    REPOSITORY_ROOT / "configs/eval/table2/paper_claim_registry_v1.json"
)

def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "handoff"
    inputs = {
        "campaign_config_path": _write(root / "campaign.json", {"kind": "pilot"}),
        "resolved_task_snapshot_path": _write(root / "resolved.json", {"tasks": []}),
        "environment_manifest_path": _write(root / "environment.json", {"browser": "x"}),
        "runner_attestation_path": _write(root / "runner.json", {"entrypoint": "x"}),
        "selection_evidence_path": _write(
            root / "selection" / "manifest.json", {"winner": "pc01"}
        ),
        "paper_claim_registry_path": _write(
            root / "paper_claim_registry.json",
            json.loads(PAPER_CLAIM_REGISTRY.read_text(encoding="utf-8")),
        ),
        "model_manifest_paths": [_write(root / "models" / "seed_42.json", {"seed": 42})],
        "memory_manifest_paths": [_write(root / "memory" / "seed_42.json", {"seed": 42})],
        "pc01_checkpoint_compatibility_receipt_path": _write(
            root / "receipt.json", {"status": "PASS"}
        ),
    }
    arguments = {
        "handoff_manifest": str(root / "handoff_manifest.json"),
        "campaign_config": str(inputs["campaign_config_path"]),
        "resolved_task_snapshot": str(inputs["resolved_task_snapshot_path"]),
        "environment_manifest": str(inputs["environment_manifest_path"]),
        "runner_attestation": str(inputs["runner_attestation_path"]),
        "checkpoint_selection_evidence": str(inputs["selection_evidence_path"]),
        "paper_claim_registry": str(inputs["paper_claim_registry_path"]),
        "model_manifests": [str(path) for path in inputs["model_manifest_paths"]],
        "memory_manifests": [str(path) for path in inputs["memory_manifest_paths"]],
        "pc01_checkpoint_compatibility_receipt": str(
            inputs["pc01_checkpoint_compatibility_receipt_path"]
        ),
    }
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = _write(
        root / "handoff_manifest.json",
        {
            "schema_version": "table2-handoff-bundle-v1",
            "repository_root": str(tmp_path.resolve()),
            "repository_commit": "a" * 40,
            "campaign_mode": "evaluation",
            "selection_mode": "pc01_provisional",
            "matched_seeds": [42],
            "paper_claim_registry_id": "table2-research-locked-claims-v1",
            "paper_claim_registry_sha256": sha256_file(
                inputs["paper_claim_registry_path"]
            ),
            "freeze_arguments": arguments,
            "files": files,
        },
    )
    return manifest, inputs


def _validate(tmp_path: Path, manifest: Path, inputs: dict[str, object]):
    return validate_handoff_freeze_authority(
        manifest,
        repository_root=tmp_path,
        repository_commit="a" * 40,
        **inputs,
    )


def test_handoff_authority_replays_inventory_and_all_freeze_paths(tmp_path: Path) -> None:
    manifest, inputs = _bundle(tmp_path)
    value = _validate(tmp_path, manifest, inputs)
    assert value["campaign_mode"] == "evaluation"


def test_handoff_authority_rejects_extra_or_changed_bytes(tmp_path: Path) -> None:
    manifest, inputs = _bundle(tmp_path)
    extra = manifest.parent / "unregistered.json"
    _write(extra, {"extra": True})
    with pytest.raises(SchemaError, match="inventory differs"):
        _validate(tmp_path, manifest, inputs)
    extra.unlink()
    Path(inputs["resolved_task_snapshot_path"]).write_text(
        json.dumps({"tasks": ["changed"]}), encoding="utf-8"
    )
    with pytest.raises(SchemaError, match="inventory differs"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_uninventoried_empty_directory(
    tmp_path: Path,
) -> None:
    manifest, inputs = _bundle(tmp_path)
    (manifest.parent / "unregistered-empty").mkdir()

    with pytest.raises(SchemaError, match="directory closure differs"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_hard_linked_registered_input(
    tmp_path: Path,
) -> None:
    manifest, inputs = _bundle(tmp_path)
    campaign = Path(inputs["campaign_config_path"])
    external = _write(tmp_path / "same-campaign.json", {"kind": "pilot"})
    campaign.unlink()
    os.link(external, campaign)

    with pytest.raises(SchemaError, match="hard-linked|singly linked regular file"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_replaced_freeze_input(tmp_path: Path) -> None:
    manifest, inputs = _bundle(tmp_path)
    replacement = _write(tmp_path / "replacement.json", {"tasks": []})
    inputs["resolved_task_snapshot_path"] = replacement
    with pytest.raises(SchemaError, match="resolved_task_snapshot differs"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_replaced_selection_evidence(
    tmp_path: Path,
) -> None:
    manifest, inputs = _bundle(tmp_path)
    replacement = _write(tmp_path / "replacement-selection.json", {"winner": "pc01"})
    inputs["selection_evidence_path"] = replacement
    with pytest.raises(SchemaError, match="checkpoint_selection_evidence differs"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_replaced_paper_claim_registry(
    tmp_path: Path,
) -> None:
    manifest, inputs = _bundle(tmp_path)
    replacement = _write(
        tmp_path / "replacement-paper-claims.json",
        json.loads(PAPER_CLAIM_REGISTRY.read_text(encoding="utf-8")),
    )
    inputs["paper_claim_registry_path"] = replacement
    with pytest.raises(SchemaError, match="paper_claim_registry differs"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_registered_input_outside_package(
    tmp_path: Path,
) -> None:
    manifest, inputs = _bundle(tmp_path)
    external = _write(tmp_path / "external-model.json", {"seed": 42})
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["freeze_arguments"]["model_manifests"] = [str(external)]
    manifest.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    inputs["model_manifest_paths"] = [external]

    with pytest.raises(SchemaError, match="escaped the authenticated package"):
        _validate(tmp_path, manifest, inputs)


def test_handoff_authority_rejects_different_source_commit(tmp_path: Path) -> None:
    manifest, inputs = _bundle(tmp_path)
    with pytest.raises(SchemaError, match="repository commit differs"):
        validate_handoff_freeze_authority(
            manifest,
            repository_root=tmp_path,
            repository_commit="b" * 40,
            **inputs,
        )
