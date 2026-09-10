"""DGX-only checkpoint-backed compatibility gate for the PC-01 Table 2 pilot.

This module turns the processor-only PC-01 export into measured model evidence.
It deliberately has no PyTorch/Transformers imports at module import time.  The
public runner first requires the registered DGX CUDA host, then authenticates
the complete v3 export, checkpoint, Qwen snapshot, training-data prompt fact,
and clean runtime source before loading either model.

A successful receipt proves only engineering compatibility.  It does not
promote PC-01 over PC-02/PC-03, freeze a task registry, run WebArena, or populate
Table 2.  Failures raise and never create a PASS receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import stat
import struct
import subprocess
import tempfile
from typing import Any

from web_agent.eval.table2.common import canonical_json_bytes, sha256_bytes, sha256_file
from web_agent.eval.table2.pc01_artifacts import (
    PC01_CONFIG_NAME,
    PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_EXPECTED_EXPORT_MANIFEST_SHA256,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_MODEL_SEED,
    PC01_TRAINING_ENVIRONMENT_SHA256,
    build_pc01_base_snapshot_manifest,
    load_checkpoint_saved_config,
    load_pc01_base_snapshot_manifest,
    load_pc01_training_action_value_evidence,
    load_processor_contract,
    registered_pc01_training_source_manifest,
    validate_pc01_training_sources,
)
from web_agent.eval.table2.pc01_processor_parity import (
    validate_pc01_processor_parity_receipt,
)
from web_agent.eval.table2.resolved_config import (
    assert_checkpoint_config_matches,
    load_resolved_config_identity,
)
from web_agent.labels import (
    ACTION_TYPE_INV,
    EXECUTION_OUTCOME,
    FAILURE_TYPE_INV,
    RECOVERY_STRATEGY_INV,
)
from web_agent.runtime.checkpoint_inference import (
    ValidationSelectedBackbone,
    ValidationSelectedCheckpoint,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    PolicyObservation,
    RecoveryTransitionInput,
    RuntimeTaskView,
    TransitionInput,
    canonical_sha256,
    float32_vector_sha256,
)
from web_agent.runtime.memory_adapter import PostFailureEmbeddingRequest
from web_agent.runtime.qwen2vl_pc01 import (
    E0_ACTION_PROMPT,
    E0_PARSER_ID,
    E0_PARSER_VERSION,
    PC01RuntimeArtifacts,
    _RuntimeBatchBuilder,
    _SelectedCheckpointRuntime,
    _bounded_bbox,
    _hash_tensor_batch,
    _load_selected_model,
    _load_unadapted_base,
    _strict_offline_hf,
    _tensor_probabilities,
)


PC01_CHECKPOINT_COMPATIBILITY_SCHEMA = (
    "table2.pc01-checkpoint-compatibility.v1"
)
PC01_CHECKPOINT_COMPATIBILITY_ROLE = "checkpoint_compatibility_receipt"
PC01_CHECKPOINT_COMPATIBILITY_GATE_ID = "pc01-epoch6-seed42-dgx-cuda-v1"
PC01_REGISTERED_EXPORT_MANIFEST_SHA256 = PC01_EXPECTED_EXPORT_MANIFEST_SHA256
PC01_CHECKPOINT_EPOCH = 6
PC01_REGISTERED_PYTHON_VERSION = "3.12.3"
PC01_REGISTERED_TORCH_VERSION = "2.13.0+cu130"
PC01_REGISTERED_GPU_NAME = "NVIDIA GB10"
PC01_REGISTERED_GPU_MEMORY_BYTES = 130_662_936_576
PC01_EXPECTED_CHECKPOINT_SIZE_BYTES = 291_071_781

_EXPORT_FILES = frozenset(
    {
        "resolved_config.json",
        "processor_contract.json",
        "e0_processor_contract.json",
        "e0_resolved_config.json",
        "processor_artifact_manifest.json",
        "base_snapshot_manifest.json",
        "processor_parity_receipt.json",
        "training_environment.json",
        "training_source_manifest.json",
        "training_action_value_evidence.json",
        "pc01_export_manifest.json",
    }
)
_MANIFEST_PAYLOAD_FILES = _EXPORT_FILES - {"pc01_export_manifest.json"}
_EXPORT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "config_name",
        "model_seed",
        "model_revision",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "resolved_config_record_sha256",
        "resolved_config_payload_sha256",
        "processor_contract_sha256",
        "processor_artifact_manifest_sha256",
        "processor_parity_receipt_sha256",
        "processor_parity_verified",
        "base_snapshot_manifest_sha256",
        "base_snapshot_directory_payload_sha256",
        "base_snapshot_file_count",
        "base_snapshot_copied_into_export",
        "training_environment_source_sha256",
        "training_environment_record_sha256",
        "training_source_manifest_sha256",
        "training_action_value_evidence_sha256",
        "training_transformers_version",
        "runtime_processor_implementation",
        "report_sha256",
        "run_contract_sha256",
        "selection_scope",
        "test_rows_read",
        "locked_test_rows_read",
        "network_access_used",
        "weight_updates_performed",
        "runtime_ready",
        "files",
    }
)
_CHECKPOINT_STATE_ROLES = (
    "lora",
    "adapter",
    "task_adapters",
    "failure",
    "action",
    "memory",
    "recovery_outcome",
)
_SOURCE_PATHS = (
    "scripts/run_pc01_checkpoint_compatibility.py",
    "src/web_agent/eval/table2/pc01_artifacts.py",
    "src/web_agent/eval/table2/pc01_checkpoint_compatibility.py",
    "src/web_agent/eval/table2/pc01_processor_parity.py",
    "src/web_agent/models/encoders/vlm.py",
    "src/web_agent/models/encoders/vlm_contract.py",
    "src/web_agent/models/model.py",
    "src/web_agent/runtime/checkpoint_inference.py",
    "src/web_agent/runtime/memory_adapter.py",
    "src/web_agent/runtime/observation.py",
    "src/web_agent/runtime/qwen2vl_pc01.py",
)


class PC01CheckpointCompatibilityError(RuntimeError):
    """The PC-01 CUDA compatibility gate cannot produce admissible evidence."""


@dataclass(frozen=True, slots=True)
class ValidatedPC01CompatibilityInputs:
    """Authenticated local paths and records used by the CUDA portion."""

    export_dir: Path
    checkpoint_path: Path
    base_snapshot_path: Path
    report_path: Path
    run_contract_path: Path
    export_manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    run_contract: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    checkpoint_payload: Mapping[str, Any]
    processor_contract: Any
    source_attestation: Mapping[str, Any]
    artifact_bindings: Mapping[str, Any]


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if not _is_sha256(text):
        raise PC01CheckpointCompatibilityError(f"{label} must be lowercase SHA-256")
    return text


def _require_regular_file(path: str | Path, *, label: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise PC01CheckpointCompatibilityError(
            f"{label} is absent or inaccessible: {source}"
        ) from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise PC01CheckpointCompatibilityError(
            f"{label} must be a regular non-symlink file: {source}"
        )
    return source.resolve()


def _require_directory(path: str | Path, *, label: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise PC01CheckpointCompatibilityError(
            f"{label} is absent or inaccessible: {source}"
        ) from exc
    if source.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PC01CheckpointCompatibilityError(
            f"{label} must be a non-symlink directory: {source}"
        )
    return source.resolve()


def _read_unique_json(
    path: str | Path,
    *,
    label: str,
    canonical: bool = False,
) -> dict[str, Any]:
    source = _require_regular_file(path, label=label)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PC01CheckpointCompatibilityError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except PC01CheckpointCompatibilityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PC01CheckpointCompatibilityError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PC01CheckpointCompatibilityError(f"{label} must contain one object")
    if canonical and source.read_bytes() != canonical_json_bytes(value):
        raise PC01CheckpointCompatibilityError(f"{label} is not canonical JSON")
    return value


def _exact_mapping(value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for field, required in expected.items():
        if value.get(field) != required:
            raise PC01CheckpointCompatibilityError(
                f"{label}.{field} differs: expected {required!r}, got {value.get(field)!r}"
            )


def _validate_export_manifest(
    value: Mapping[str, Any],
    *,
    export_dir: Path,
    expected_manifest_sha256: str,
    checkpoint_path: Path,
    report_path: Path,
    run_contract_path: Path,
) -> dict[str, Any]:
    """Validate every v3 export byte before any model constructor runs."""

    _require_sha256(expected_manifest_sha256, label="expected export manifest identity")
    if set(value) != _EXPORT_MANIFEST_FIELDS:
        raise PC01CheckpointCompatibilityError(
            "PC-01 v3 export manifest fields changed"
        )
    manifest_path = export_dir / "pc01_export_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise PC01CheckpointCompatibilityError(
            "PC-01 export manifest differs from the explicitly selected v3 export"
        )
    _exact_mapping(
        value,
        {
            "schema_version": "table2.pc01-export.v3",
            "model_id": PC01_MODEL_ID,
            "config_name": PC01_CONFIG_NAME,
            "model_seed": PC01_MODEL_SEED,
            "model_revision": PC01_MODEL_REVISION,
            "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
            "resolved_config_record_sha256": PC01_EXPECTED_CONFIG_SHA256,
            "resolved_config_payload_sha256": PC01_EXPECTED_CONFIG_SHA256,
            "processor_parity_verified": True,
            "base_snapshot_directory_payload_sha256": (
                PC01_EXPECTED_BASE_SNAPSHOT_SHA256
            ),
            "base_snapshot_file_count": 14,
            "base_snapshot_copied_into_export": False,
            "training_environment_source_sha256": (
                PC01_TRAINING_ENVIRONMENT_SHA256
            ),
            "training_transformers_version": "4.57.6",
            "selection_scope": "validation_only",
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "network_access_used": False,
            "weight_updates_performed": False,
            "runtime_ready": False,
        },
        label="PC-01 export manifest",
    )
    if (
        value.get("checkpoint_size_bytes") != PC01_EXPECTED_CHECKPOINT_SIZE_BYTES
        or checkpoint_path.stat().st_size != PC01_EXPECTED_CHECKPOINT_SIZE_BYTES
    ):
        raise PC01CheckpointCompatibilityError(
            "PC-01 export checkpoint size differs from the supplied checkpoint"
        )
    direct_bindings = {
        "report_sha256": sha256_file(report_path),
        "run_contract_sha256": sha256_file(run_contract_path),
    }
    _exact_mapping(value, direct_bindings, label="PC-01 export manifest")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != _MANIFEST_PAYLOAD_FILES:
        raise PC01CheckpointCompatibilityError(
            "PC-01 export manifest payload file coverage changed"
        )
    for name in sorted(_MANIFEST_PAYLOAD_FILES):
        source = _require_regular_file(export_dir / name, label=f"PC-01 export {name}")
        row = files[name]
        if not isinstance(row, Mapping) or set(row) != {"sha256", "size_bytes"}:
            raise PC01CheckpointCompatibilityError(
                f"PC-01 export manifest row for {name} is malformed"
            )
        expected = {"sha256": sha256_file(source), "size_bytes": source.stat().st_size}
        if dict(row) != expected:
            raise PC01CheckpointCompatibilityError(
                f"PC-01 export payload differs from its manifest: {name}"
            )
    cross_links = {
        "processor_artifact_manifest_sha256": files[
            "processor_artifact_manifest.json"
        ]["sha256"],
        "processor_parity_receipt_sha256": files[
            "processor_parity_receipt.json"
        ]["sha256"],
        "base_snapshot_manifest_sha256": files[
            "base_snapshot_manifest.json"
        ]["sha256"],
        "training_environment_record_sha256": files[
            "training_environment.json"
        ]["sha256"],
        "training_source_manifest_sha256": files[
            "training_source_manifest.json"
        ]["sha256"],
        "training_action_value_evidence_sha256": files[
            "training_action_value_evidence.json"
        ]["sha256"],
    }
    _exact_mapping(value, cross_links, label="PC-01 export manifest")
    return dict(files)


def _repository_attestation(
    repository_root: str | Path,
    *,
    expected_source_commit: str,
) -> dict[str, Any]:
    root = _require_directory(repository_root, label="runtime source repository")
    if (
        len(expected_source_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_source_commit)
    ):
        raise PC01CheckpointCompatibilityError(
            "expected runtime source commit must be a lowercase 40-character Git commit"
        )
    try:
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status_text = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility requires a Git source checkout"
        ) from exc
    if top != root:
        raise PC01CheckpointCompatibilityError(
            "repository root must be the exact Git top-level directory"
        )
    if commit != expected_source_commit:
        raise PC01CheckpointCompatibilityError(
            "runtime source commit differs from the explicitly selected commit"
        )
    if status_text.strip():
        raise PC01CheckpointCompatibilityError(
            "runtime source repository is dirty; commit all gate/runtime changes first"
        )
    files: list[dict[str, Any]] = []
    for relative in _SOURCE_PATHS:
        source = _require_regular_file(root / relative, label=f"runtime source {relative}")
        files.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    return {
        "git_commit": commit,
        "git_clean": True,
        "source_file_count": len(files),
        "source_files": files,
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _validate_pc01_export_bundle(
    *,
    export_dir: str | Path,
    checkpoint_path: str | Path,
    base_snapshot_path: str | Path,
    report_path: str | Path,
    run_contract_path: str | Path,
    repository_root: str | Path,
    expected_source_commit: str,
) -> ValidatedPC01CompatibilityInputs:
    export = _require_directory(export_dir, label="PC-01 v3 export")
    checkpoint = _require_regular_file(checkpoint_path, label="PC-01 checkpoint")
    base_snapshot = _require_directory(
        base_snapshot_path, label="PC-01 pinned Qwen base snapshot"
    )
    report_source = _require_regular_file(report_path, label="PC-01 full report")
    contract_source = _require_regular_file(
        run_contract_path, label="PC-01 run contract"
    )
    entries = {path.name for path in export.iterdir()}
    if entries != _EXPORT_FILES or any(not path.is_file() for path in export.iterdir()):
        raise PC01CheckpointCompatibilityError(
            "PC-01 v3 export has missing, extra, or non-file entries"
        )
    manifest = _read_unique_json(
        export / "pc01_export_manifest.json",
        label="PC-01 export manifest",
        canonical=True,
    )
    _validate_export_manifest(
        manifest,
        export_dir=export,
        expected_manifest_sha256=PC01_REGISTERED_EXPORT_MANIFEST_SHA256,
        checkpoint_path=checkpoint,
        report_path=report_source,
        run_contract_path=contract_source,
    )
    if sha256_file(checkpoint) != PC01_EXPECTED_CHECKPOINT_SHA256:
        raise PC01CheckpointCompatibilityError(
            "supplied checkpoint differs from registered PC-01 epoch 6"
        )

    report = _read_unique_json(report_source, label="PC-01 full report")
    run_contract = _read_unique_json(contract_source, label="PC-01 run contract")
    _exact_mapping(
        report,
        {
            "seed": PC01_MODEL_SEED,
            "stage": "full",
            "status": "PASS",
            "selected_epoch": PC01_CHECKPOINT_EPOCH,
            "selected_checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
            "test_rows_read": 0,
        },
        label="PC-01 full report",
    )
    val_rows = report.get("val_rows")
    if type(val_rows) is not int or val_rows <= 0:
        raise PC01CheckpointCompatibilityError(
            "PC-01 report lacks positive validation-only selection rows"
        )
    _exact_mapping(
        run_contract,
        {
            "seed": PC01_MODEL_SEED,
            "model_revision": PC01_MODEL_REVISION,
            "checkpoint_selection_source": "original_gold_validation_only",
            "supplement_validation_selects_checkpoint": False,
            "test_rows_read": 0,
        },
        label="PC-01 run contract",
    )

    config_identity = load_resolved_config_identity(export / "resolved_config.json")
    if (
        config_identity.payload_sha256 != PC01_EXPECTED_CONFIG_SHA256
        or config_identity.record_sha256 != PC01_EXPECTED_CONFIG_SHA256
    ):
        raise PC01CheckpointCompatibilityError(
            "PC-01 resolved config does not reproduce its registered identity"
        )
    saved_config, checkpoint_sha256, checkpoint_payload = load_checkpoint_saved_config(
        checkpoint,
        expected_checkpoint_sha256=PC01_EXPECTED_CHECKPOINT_SHA256,
    )
    if checkpoint_sha256 != PC01_EXPECTED_CHECKPOINT_SHA256:
        raise PC01CheckpointCompatibilityError("PC-01 checkpoint identity changed")
    assert_checkpoint_config_matches(saved_config, config_identity)

    processor_contract = load_processor_contract(export / "processor_contract.json")
    e0_contract = load_processor_contract(export / "e0_processor_contract.json")
    if e0_contract != processor_contract:
        raise PC01CheckpointCompatibilityError(
            "selected and E0 processor contracts are not identical"
        )
    if processor_contract.record_sha256 != manifest.get("processor_contract_sha256"):
        raise PC01CheckpointCompatibilityError(
            "processor contract differs from the v3 export manifest"
        )

    supplied_snapshot_manifest = load_pc01_base_snapshot_manifest(
        export / "base_snapshot_manifest.json"
    )
    actual_snapshot_manifest = build_pc01_base_snapshot_manifest(base_snapshot)
    if actual_snapshot_manifest != supplied_snapshot_manifest:
        raise PC01CheckpointCompatibilityError(
            "live Qwen base snapshot differs from every file in the exported manifest"
        )

    training_sources = _read_unique_json(
        export / "training_source_manifest.json",
        label="PC-01 training source manifest",
        canonical=True,
    )
    authenticated_sources = validate_pc01_training_sources(repository_root)
    if training_sources != authenticated_sources or training_sources != (
        registered_pc01_training_source_manifest()
    ):
        raise PC01CheckpointCompatibilityError(
            "PC-01 exported/current/registered training sources differ"
        )
    action_value_evidence = load_pc01_training_action_value_evidence(
        export / "training_action_value_evidence.json",
        report_path=report_source,
        run_contract_path=contract_source,
    )
    action_value_sha256 = sha256_file(
        export / "training_action_value_evidence.json"
    )

    training_environment = _read_unique_json(
        export / "training_environment.json",
        label="PC-01 training environment",
        canonical=True,
    )
    _exact_mapping(
        training_environment,
        {
            "python": PC01_REGISTERED_PYTHON_VERSION,
            "torch": PC01_REGISTERED_TORCH_VERSION,
            "transformers": "4.57.6",
            "gpu": PC01_REGISTERED_GPU_NAME,
            "gpu_total_memory_gb": 130.662936576,
            "git_commit": "2fadf0f508cec42ce6f89b8961db7cfd2adef1df",
        },
        label="PC-01 training environment",
    )

    processor_artifacts = _read_unique_json(
        export / "processor_artifact_manifest.json",
        label="PC-01 processor artifact manifest",
        canonical=True,
    )
    _exact_mapping(
        processor_artifacts,
        {
            "schema_version": "table2.pc01-processor-artifacts.v3",
            "model_id": "Qwen/Qwen2-VL-2B-Instruct",
            "revision": PC01_MODEL_REVISION,
            "training_environment_sha256": PC01_TRAINING_ENVIRONMENT_SHA256,
            "training_source_manifest_sha256": sha256_file(
                export / "training_source_manifest.json"
            ),
            "training_action_value_evidence_sha256": action_value_sha256,
            "training_transformers_version": "4.57.6",
            "implementation_identity": manifest.get(
                "runtime_processor_implementation"
            ),
        },
        label="PC-01 processor artifact manifest",
    )
    if sha256_bytes(canonical_json_bytes(processor_artifacts)) != (
        processor_contract.processor_config_sha256
    ):
        raise PC01CheckpointCompatibilityError(
            "processor contract is not bound to the exported processor artifacts"
        )

    parity = _read_unique_json(
        export / "processor_parity_receipt.json",
        label="PC-01 processor parity receipt",
        canonical=True,
    )
    validate_pc01_processor_parity_receipt(
        parity,
        resolved_config_record_sha256=PC01_EXPECTED_CONFIG_SHA256,
        processor_contract_sha256=processor_contract.record_sha256,
        processor_artifact_manifest_sha256=sha256_file(
            export / "processor_artifact_manifest.json"
        ),
        base_snapshot_directory_payload_sha256=(
            PC01_EXPECTED_BASE_SNAPSHOT_SHA256
        ),
        training_environment_sha256=PC01_TRAINING_ENVIRONMENT_SHA256,
        training_environment_record_sha256=sha256_file(
            export / "training_environment.json"
        ),
        training_source_manifest_sha256=sha256_file(
            export / "training_source_manifest.json"
        ),
        training_action_value_evidence_sha256=action_value_sha256,
        run_contract_sha256=sha256_file(contract_source),
    )
    if not (
        manifest.get("runtime_processor_implementation")
        == processor_artifacts.get("implementation_identity")
        == parity.get("implementation_identity")
        == parity.get("runtime_processor_implementation")
    ):
        raise PC01CheckpointCompatibilityError(
            "v3 export, processor artifacts, and parity receipt cite different "
            "runtime processor implementations"
        )

    e0_config = _read_unique_json(
        export / "e0_resolved_config.json",
        label="PC-01 E0 resolved config",
        canonical=True,
    )
    if set(e0_config) != {
        "schema_version",
        "system_id",
        "adaptation_loaded",
        "task_heads_loaded",
        "backbone",
    }:
        raise PC01CheckpointCompatibilityError("PC-01 E0 config fields changed")
    _exact_mapping(
        e0_config,
        {
            "schema_version": "table2.pc01-e0-resolved-config.v1",
            "system_id": "E0",
            "adaptation_loaded": False,
            "task_heads_loaded": False,
            "backbone": config_identity.mapping.get("backbone"),
        },
        label="PC-01 E0 config",
    )

    source_attestation = _repository_attestation(
        repository_root, expected_source_commit=expected_source_commit
    )
    artifact_bindings = {
        "export_manifest_sha256": PC01_REGISTERED_EXPORT_MANIFEST_SHA256,
        "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "resolved_config_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "processor_contract_sha256": processor_contract.record_sha256,
        "processor_artifact_manifest_sha256": sha256_file(
            export / "processor_artifact_manifest.json"
        ),
        "processor_parity_receipt_sha256": sha256_file(
            export / "processor_parity_receipt.json"
        ),
        "base_snapshot_manifest_sha256": sha256_file(
            export / "base_snapshot_manifest.json"
        ),
        "base_snapshot_directory_payload_sha256": (
            PC01_EXPECTED_BASE_SNAPSHOT_SHA256
        ),
        "training_environment_record_sha256": sha256_file(
            export / "training_environment.json"
        ),
        "training_environment_source_sha256": (
            PC01_TRAINING_ENVIRONMENT_SHA256
        ),
        "training_source_manifest_sha256": sha256_file(
            export / "training_source_manifest.json"
        ),
        "training_action_value_evidence_sha256": action_value_sha256,
        "report_sha256": sha256_file(report_source),
        "run_contract_sha256": sha256_file(contract_source),
    }
    if action_value_evidence.get("runtime_requirement") != "OMIT_ACTION_VALUE_TEXT":
        raise PC01CheckpointCompatibilityError(
            "training action-value evidence does not require runtime omission"
        )
    return ValidatedPC01CompatibilityInputs(
        export_dir=export,
        checkpoint_path=checkpoint,
        base_snapshot_path=base_snapshot,
        report_path=report_source,
        run_contract_path=contract_source,
        export_manifest=manifest,
        report=report,
        run_contract=run_contract,
        resolved_config=config_identity.mapping,
        checkpoint_payload=checkpoint_payload,
        processor_contract=processor_contract,
        source_attestation=source_attestation,
        artifact_bindings=artifact_bindings,
    )


def _require_registered_dgx_cuda() -> tuple[Any, dict[str, Any]]:
    """Return PyTorch only on the exact registered GB10 model plane."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility requires the pinned PyTorch runtime"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility requires CUDA on the DGX model plane"
        )
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    actual = {
        "host_role": "DGX_MODEL_PLANE",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_index": 0,
        "cuda_device_name": str(properties.name),
        "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
    }
    exact = {
        "architecture": "aarch64",
        "python_version": PC01_REGISTERED_PYTHON_VERSION,
        "torch_version": PC01_REGISTERED_TORCH_VERSION,
        "cuda_runtime_version": "13.0",
        "cuda_device_name": PC01_REGISTERED_GPU_NAME,
        "cuda_device_total_memory_bytes": PC01_REGISTERED_GPU_MEMORY_BYTES,
    }
    _exact_mapping(actual, exact, label="DGX CUDA runtime")
    return torch, actual


def _state_mapping_identity(values: Mapping[str, Any], *, torch: Any) -> dict[str, Any]:
    """Hash exact state keys, dtype/shape metadata, and tensor bytes."""

    if not isinstance(values, Mapping) or not values:
        raise PC01CheckpointCompatibilityError("checkpoint/model state mapping is empty")
    digest = hashlib.sha256()
    tensor_count = 0
    total_tensor_bytes = 0
    keys = sorted(values)
    for key in keys:
        if not isinstance(key, str) or not key:
            raise PC01CheckpointCompatibilityError("state mapping key is invalid")
        value = values[key]
        if not torch.is_tensor(value):
            raise PC01CheckpointCompatibilityError(
                f"state mapping value is not a tensor: {key}"
            )
        if str(value.device) == "meta":
            raise PC01CheckpointCompatibilityError(f"state tensor is unmaterialized: {key}")
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        metadata = {
            "key": key,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "num_bytes": len(raw),
        }
        encoded = canonical_json_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        tensor_count += 1
        total_tensor_bytes += len(raw)
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "total_tensor_bytes": total_tensor_bytes,
    }


def _tensor_mapping_identity(values: Mapping[str, Any], *, torch: Any) -> dict[str, Any]:
    tensors = {key: value for key, value in values.items() if torch.is_tensor(value)}
    if set(tensors) != set(values):
        raise PC01CheckpointCompatibilityError(
            "model inference returned a non-tensor output"
        )
    return _state_mapping_identity(tensors, torch=torch)


def _verify_checkpoint_restore(
    *, model: Any, checkpoint: Mapping[str, Any], torch: Any
) -> dict[str, Any]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise PC01CheckpointCompatibilityError(
            "checkpoint restore validation requires PEFT"
        ) from exc
    loaded_states = {
        "lora": get_peft_model_state_dict(model.encoder.model),
        "adapter": model.adapter.state_dict(),
        "task_adapters": model.task_adapters.state_dict(),
        "failure": model.failure_head.state_dict(),
        "action": model.action_head.state_dict(),
        "memory": model.memory_head.state_dict(),
        "recovery_outcome": model.recovery_outcome_head.state_dict(),
    }
    roles: dict[str, Any] = {}
    for role in _CHECKPOINT_STATE_ROLES:
        source = checkpoint.get(role)
        if not isinstance(source, Mapping):
            raise PC01CheckpointCompatibilityError(
                f"checkpoint state role is missing: {role}"
            )
        checkpoint_identity = _state_mapping_identity(source, torch=torch)
        loaded_identity = _state_mapping_identity(loaded_states[role], torch=torch)
        if checkpoint_identity != loaded_identity:
            raise PC01CheckpointCompatibilityError(
                f"loaded model state differs from checkpoint role {role}"
            )
        roles[role] = {
            "checkpoint": checkpoint_identity,
            "loaded": loaded_identity,
            "exact_match": True,
        }
    return {"roles": roles, "all_roles_exact": True}


def _assert_eval_frozen_cuda(model: Any, *, label: str) -> dict[str, Any]:
    modules = tuple(model.modules())
    parameters = tuple(model.parameters())
    if model.training or any(module.training for module in modules):
        raise PC01CheckpointCompatibilityError(f"{label} is not wholly in eval mode")
    if not parameters:
        raise PC01CheckpointCompatibilityError(f"{label} has no materialized parameters")
    if any(parameter.requires_grad for parameter in parameters):
        raise PC01CheckpointCompatibilityError(f"{label} has trainable parameters")
    devices = {parameter.device.type for parameter in parameters}
    if devices != {"cuda"}:
        raise PC01CheckpointCompatibilityError(
            f"{label} parameters are not all on CUDA: {sorted(devices)}"
        )
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "eval_mode": True,
        "frozen": True,
        "all_parameters_cuda": True,
        "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "module_count": len(modules),
    }


def _assert_e0_unadapted(model: Any) -> dict[str, Any]:
    identity = _assert_eval_frozen_cuda(model, label="E0 base model")
    forbidden = (
        "lora_",
        "task_adapters",
        "failure_head",
        "action_head",
        "memory_head",
        "recovery_outcome_head",
    )
    names = tuple(name for name, _ in model.named_modules()) + tuple(
        name for name, _ in model.named_parameters()
    )
    hits = sorted({token for token in forbidden if any(token in name for name in names)})
    class_name = identity["class"].casefold()
    if hits or class_name.startswith("peft.") or "peftmodel" in class_name:
        raise PC01CheckpointCompatibilityError(
            f"E0 contains adaptation/task-head state: {hits}"
        )
    return {
        **identity,
        "adaptation_loaded": False,
        "task_heads_loaded": False,
        "forbidden_state_hits": [],
    }


def _write_ppm(path: Path, *, width: int, height: int, salt: int) -> None:
    payload = bytearray()
    for y in range(height):
        for x in range(width):
            payload.extend(
                (
                    (17 * x + 3 * y + salt) % 256,
                    (5 * x + 19 * y + 2 * salt) % 256,
                    (13 * x + 7 * y + 3 * salt) % 256,
                )
            )
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)
    path.chmod(0o444)


def _compatibility_fixture(root: Path, *, processor_contract_sha256: str) -> dict[str, Any]:
    width = height = 224
    paths = tuple(root / f"state-{index}.ppm" for index in range(3))
    for index, path in enumerate(paths):
        _write_ppm(path, width=width, height=height, salt=31 + 47 * index)
    hashes = tuple(sha256_file(path) for path in paths)
    task = RuntimeTaskView(
        task_id="pc01-checkpoint-compatibility-fixture",
        goal="Type the requested text into the visible field, then submit it.",
    )
    observations = tuple(
        PolicyObservation(
            task_id=task.task_id,
            goal=task.goal,
            observation_id=f"fixture-observation-{index}",
            screenshot_sha256=hashes[index],
            screenshot_path=str(paths[index]),
            width=width,
            height=height,
            url=f"https://fixture.invalid/state-{index}",
            title=f"Fixture state {index}",
            current_page_state={
                "visible_text_sha256": canonical_sha256(
                    {"state": index, "kind": "compatibility_fixture"}
                )
            },
        )
        for index in range(3)
    )
    failed_action = ConcreteAction(
        action_id="fixture-action-type",
        source_decision_id="fixture-decision-type",
        action_type=ActionType.TYPE,
        parameters={
            "target_x": 0.5,
            "target_y": 0.5,
            "target_bbox": [0.25, 0.25, 0.5, 0.5],
            "text": "failure-aware web agent",
        },
        bbox=(0.25, 0.25, 0.5, 0.5),
    )
    transition = TransitionInput(
        task_id=task.task_id,
        pre_observation=observations[0],
        executed_action=failed_action,
        execution_result=ExecutionResult(
            action_id=failed_action.action_id,
            status=ExecutionStatus.EXECUTED,
            executor_step=1,
            state_changed=True,
        ),
        post_observation=observations[1],
    )
    recovery_action = ConcreteAction(
        action_id="fixture-recovery-press-key",
        source_decision_id="fixture-recovery-decision",
        action_type=ActionType.PRESS_KEY,
        parameters={"key": "ENTER"},
        recovery_attempt_id="fixture-recovery-attempt",
    )
    recovery = RecoveryTransitionInput(
        task_id=task.task_id,
        incident_id="fixture-incident",
        attempt_id="fixture-recovery-attempt",
        pre_recovery_observation=observations[1],
        recovery_actions=(recovery_action,),
        post_recovery_observation=observations[2],
    )
    request = PostFailureEmbeddingRequest(
        query_id="fixture-memory-query",
        post_failure_observation_id=observations[1].observation_id,
        failed_action_id=failed_action.action_id,
        post_action_input=transition,
        post_failure_observation_sha256=observations[1].record_sha256,
        post_action_input_sha256=transition.record_sha256,
        processor_contract_sha256=processor_contract_sha256,
        checkpoint_sha256=PC01_EXPECTED_CHECKPOINT_SHA256,
    )
    return {
        "task": task,
        "observations": observations,
        "transition": transition,
        "recovery": recovery,
        "embedding_request": request,
        "evidence": {
            "fixture_id": "pc01-causal-three-state-ppm-v1",
            "generator_source_sha256": sha256_file(Path(__file__)),
            "width": width,
            "height": height,
            "screenshot_sha256": list(hashes),
            "task_sha256": task.record_sha256,
            "transition_sha256": transition.record_sha256,
            "recovery_transition_sha256": recovery.record_sha256,
            "embedding_request_sha256": request.record_sha256,
            "contains_webarena_task_content": False,
            "contains_oracle_content": False,
        },
    }


def _action_semantics(
    *, model: Any, batch: Mapping[str, Any], torch: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    pre_encoded = model.encode(dict(batch), prefix="pre_")
    adapters = model.task_adapters
    pre_fused = adapters["policy"](pre_encoded["fused"])
    confidence = model.failure_head(
        pre_fused, confidence_fused=pre_fused
    )["confidence"]
    bbox_source = pre_encoded["fused"].float() if model.bbox_fp32_grounding else pre_encoded["fused"]
    bbox_fused = adapters["grounding"](bbox_source) if model.separate_bbox_adapter else pre_fused
    spatial_tokens = pre_encoded.get("spatial_tokens")
    if spatial_tokens is not None:
        if model.bbox_fp32_grounding:
            spatial_tokens = spatial_tokens.float()
        spatial_tokens = adapters[
            "grounding" if model.separate_bbox_adapter else "policy"
        ](spatial_tokens)
    predictions = model.action_head(
        pre_fused,
        bbox_fused=bbox_fused,
        spatial_tokens=spatial_tokens,
        spatial_mask=pre_encoded.get("spatial_mask"),
        spatial_coords=pre_encoded.get("spatial_coords"),
        force_fp32_bbox=model.bbox_fp32_grounding,
    )
    probabilities = _tensor_probabilities(torch, predictions["action_type"])
    action_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    bbox = _bounded_bbox(predictions["bbox"][0].detach().float().cpu().tolist())
    semantic = {
        "action_type": ACTION_TYPE_INV[action_index],
        "action_probabilities": {
            ACTION_TYPE_INV[index]: probability
            for index, probability in enumerate(probabilities)
        },
        "bbox": list(bbox),
        "grounding_confidence": max(probabilities),
        "confidence_before": float(confidence[0].detach().float().cpu().item()),
    }
    raw = {**predictions, "confidence": confidence}
    return semantic, _tensor_mapping_identity(raw, torch=torch)


def _transition_semantics(
    *, predictions: Mapping[str, Any], torch: Any
) -> dict[str, Any]:
    outcome = _tensor_probabilities(torch, predictions["outcome"])
    failure_types = _tensor_probabilities(torch, predictions["failure_type"])
    recovery = _tensor_probabilities(torch, predictions["recovery"])
    failure_index = max(range(len(failure_types)), key=failure_types.__getitem__)
    recovery_index = max(range(len(recovery)), key=recovery.__getitem__)
    needs_probability = float(
        torch.sigmoid(predictions["needs_recovery"].float())[0]
        .detach()
        .cpu()
        .item()
    )
    memory_probability = float(
        torch.sigmoid(predictions["memory_flag"].float())[0]
        .detach()
        .cpu()
        .item()
    )
    return {
        "predicted_failure": (
            max(range(len(outcome)), key=outcome.__getitem__)
            == EXECUTION_OUTCOME["FAILURE"]
        ),
        "failure_probability": outcome[EXECUTION_OUTCOME["FAILURE"]],
        "failure_type": FAILURE_TYPE_INV[failure_index],
        "failure_type_probabilities": {
            FAILURE_TYPE_INV[index]: probability
            for index, probability in enumerate(failure_types)
        },
        "needs_recovery": needs_probability >= 0.5,
        "needs_recovery_probability": needs_probability,
        "recovery_strategy": RECOVERY_STRATEGY_INV[recovery_index],
        "recovery_probabilities": {
            RECOVERY_STRATEGY_INV[index]: probability
            for index, probability in enumerate(recovery)
        },
        "memory_update_prediction": memory_probability >= 0.5,
    }


def _runtime_action_semantics(value: Any) -> dict[str, Any]:
    return {
        "action_type": value.action_type.value,
        "action_probabilities": dict(value.action_probabilities),
        "bbox": list(value.bbox) if value.bbox is not None else None,
        "grounding_confidence": value.grounding_confidence,
        "confidence_before": value.confidence_before,
    }


def _runtime_transition_semantics(value: Any) -> dict[str, Any]:
    return {
        "predicted_failure": value.predicted_failure,
        "failure_probability": value.failure_probability,
        "failure_type": value.failure_type,
        "failure_type_probabilities": dict(value.failure_type_probabilities),
        "needs_recovery": value.needs_recovery,
        "needs_recovery_probability": value.needs_recovery_probability,
        "recovery_strategy": value.recovery_strategy.value,
        "recovery_probabilities": dict(value.recovery_probabilities),
        "memory_update_prediction": value.memory_update_prediction,
    }


def _runtime_recovery_semantics(value: Any) -> dict[str, Any]:
    return {
        "predicted_failure_resolved": value.predicted_failure_resolved,
        "predicted_resolution_probability": value.predicted_resolution_probability,
        "predicted_progress": value.predicted_progress,
        "predicted_progress_probability": value.predicted_progress_probability,
    }


def _matching_semantic_evidence(
    *, label: str, direct: Mapping[str, Any], runtime: Mapping[str, Any], repeated: Mapping[str, Any]
) -> dict[str, Any]:
    if direct != runtime or runtime != repeated:
        raise PC01CheckpointCompatibilityError(
            f"{label} direct/runtime/repeat inference parity failed"
        )
    digest = sha256_bytes(canonical_json_bytes(direct))
    return {
        "direct_semantic_sha256": digest,
        "runtime_semantic_sha256": sha256_bytes(canonical_json_bytes(runtime)),
        "repeat_semantic_sha256": sha256_bytes(canonical_json_bytes(repeated)),
        "direct_runtime_exact": True,
        "repeat_exact": True,
    }


def _run_inference_checks(
    *,
    model: Any,
    processor: Any,
    torch: Any,
    config: Mapping[str, Any],
    processor_contract: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = _SelectedCheckpointRuntime(
        model=model,
        processor=processor,
        torch=torch,
        config=config,
        checkpoint_sha256=PC01_EXPECTED_CHECKPOINT_SHA256,
        processor_contract=processor_contract,
    )
    with tempfile.TemporaryDirectory(prefix="pc01-checkpoint-gate-") as name:
        fixture = _compatibility_fixture(
            Path(name), processor_contract_sha256=processor_contract.record_sha256
        )
        task = fixture["task"]
        observations = fixture["observations"]
        transition = fixture["transition"]
        recovery = fixture["recovery"]
        embedding_request = fixture["embedding_request"]
        builder = _RuntimeBatchBuilder(processor=processor, config=config, torch=torch)
        pre_stream = builder.stream(task=task, observations=(observations[0],), phase="pre")
        pre_batch = builder.prefix(pre_stream, "pre_")
        transition_batch = builder.transition(task, transition)
        recovery_batch = builder.recovery(task, recovery)

        with torch.inference_mode():
            direct_action, direct_action_raw = _action_semantics(
                model=model, batch=pre_batch, torch=torch
            )
            repeated_action, repeated_action_raw = _action_semantics(
                model=model, batch=pre_batch, torch=torch
            )
        if direct_action_raw != repeated_action_raw or direct_action != repeated_action:
            raise PC01CheckpointCompatibilityError(
                "pre-action raw checkpoint inference is not deterministic"
            )
        runtime_action = _runtime_action_semantics(
            runtime.predict_action(task, observations[0], random.Random(42))
        )
        runtime_action_repeat = _runtime_action_semantics(
            runtime.predict_action(task, observations[0], random.Random(42))
        )
        action_evidence = {
            "processed_batch_sha256": _hash_tensor_batch(pre_batch, prefix="pre_"),
            "direct_raw_tensor_sha256": direct_action_raw["sha256"],
            "repeat_raw_tensor_sha256": repeated_action_raw["sha256"],
            **_matching_semantic_evidence(
                label="pre-action",
                direct=direct_action,
                runtime=runtime_action,
                repeated=runtime_action_repeat,
            ),
        }

        with torch.inference_mode():
            direct_transition_raw = model(transition_batch)
            repeated_transition_raw = model(transition_batch)
        direct_transition_identity = _tensor_mapping_identity(
            direct_transition_raw, torch=torch
        )
        repeated_transition_identity = _tensor_mapping_identity(
            repeated_transition_raw, torch=torch
        )
        if direct_transition_identity != repeated_transition_identity:
            raise PC01CheckpointCompatibilityError(
                "post-action raw checkpoint inference is not deterministic"
            )
        direct_transition = _transition_semantics(
            predictions=direct_transition_raw, torch=torch
        )
        repeated_direct_transition = _transition_semantics(
            predictions=repeated_transition_raw, torch=torch
        )
        if direct_transition != repeated_direct_transition:
            raise PC01CheckpointCompatibilityError(
                "post-action decoded checkpoint inference is not deterministic"
            )
        runtime_transition = _runtime_transition_semantics(
            runtime.assess_transition(task, transition, random.Random(42))
        )
        runtime_transition_repeat = _runtime_transition_semantics(
            runtime.assess_transition(task, transition, random.Random(42))
        )
        transition_evidence = {
            "processed_pre_batch_sha256": _hash_tensor_batch(
                transition_batch, prefix="pre_"
            ),
            "processed_post_batch_sha256": _hash_tensor_batch(
                transition_batch, prefix="post_"
            ),
            "direct_raw_tensor_sha256": direct_transition_identity["sha256"],
            "repeat_raw_tensor_sha256": repeated_transition_identity["sha256"],
            **_matching_semantic_evidence(
                label="post-action",
                direct=direct_transition,
                runtime=runtime_transition,
                repeated=runtime_transition_repeat,
            ),
        }

        with torch.inference_mode():
            direct_recovery_raw = model(recovery_batch)
            repeated_recovery_raw = model(recovery_batch)
        direct_recovery_identity = _tensor_mapping_identity(
            direct_recovery_raw, torch=torch
        )
        repeated_recovery_identity = _tensor_mapping_identity(
            repeated_recovery_raw, torch=torch
        )
        if direct_recovery_identity != repeated_recovery_identity:
            raise PC01CheckpointCompatibilityError(
                "recovery raw checkpoint inference is not deterministic"
            )
        direct_probability = float(
            torch.sigmoid(direct_recovery_raw["recovery_outcome"].float())[0]
            .detach()
            .cpu()
            .item()
        )
        direct_recovery = {
            "predicted_failure_resolved": direct_probability >= 0.5,
            "predicted_resolution_probability": direct_probability,
            "predicted_progress": direct_probability >= 0.5,
            "predicted_progress_probability": direct_probability,
        }
        runtime_recovery = _runtime_recovery_semantics(
            runtime.assess_recovery(task, recovery, random.Random(42))
        )
        runtime_recovery_repeat = _runtime_recovery_semantics(
            runtime.assess_recovery(task, recovery, random.Random(42))
        )
        recovery_evidence = {
            "processed_pre_batch_sha256": _hash_tensor_batch(
                recovery_batch, prefix="pre_"
            ),
            "processed_post_batch_sha256": _hash_tensor_batch(
                recovery_batch, prefix="post_"
            ),
            "processed_recovery_batch_sha256": _hash_tensor_batch(
                recovery_batch, prefix="recovery_"
            ),
            "direct_raw_tensor_sha256": direct_recovery_identity["sha256"],
            "repeat_raw_tensor_sha256": repeated_recovery_identity["sha256"],
            **_matching_semantic_evidence(
                label="recovery",
                direct=direct_recovery,
                runtime=runtime_recovery,
                repeated=runtime_recovery_repeat,
            ),
        }

        direct_embedding_batch = builder.transition(task, transition)
        with torch.inference_mode():
            direct_embedding = model.memory_embedding(direct_embedding_batch)
            repeated_embedding = model.memory_embedding(direct_embedding_batch)
        if tuple(direct_embedding.shape) != (1, 768):
            raise PC01CheckpointCompatibilityError(
                "direct memory-task-adapter output is not [1, 768]"
            )
        if direct_embedding.dtype != torch.float32 or repeated_embedding.dtype != torch.float32:
            raise PC01CheckpointCompatibilityError(
                "memory-task-adapter output is not exact float32"
            )
        if not torch.isfinite(direct_embedding).all() or not torch.equal(
            direct_embedding.detach().cpu(), repeated_embedding.detach().cpu()
        ):
            raise PC01CheckpointCompatibilityError(
                "direct memory-task-adapter output is non-finite or nondeterministic"
            )
        direct_values = tuple(
            float(item)
            for item in direct_embedding[0].detach().float().cpu().tolist()
        )
        runtime_embedding = runtime.memory_embedding(embedding_request)
        runtime_embedding_repeat = runtime.memory_embedding(embedding_request)
        if (
            runtime_embedding.values != direct_values
            or runtime_embedding_repeat.values != direct_values
            or runtime_embedding.embedding_sha256
            != float32_vector_sha256(direct_values)
            or runtime_embedding_repeat.embedding_sha256
            != runtime_embedding.embedding_sha256
        ):
            raise PC01CheckpointCompatibilityError(
                "direct/runtime/repeat P4 embedding parity failed"
            )
        norm = math.hypot(*direct_values)
        if not math.isfinite(norm) or norm <= 1e-12:
            raise PC01CheckpointCompatibilityError(
                "P4 embedding has zero or non-finite norm"
            )
        expected_batch_sha256 = _hash_tensor_batch(
            direct_embedding_batch, prefix="post_"
        )
        if (
            runtime_embedding.processed_batch_sha256 != expected_batch_sha256
            or runtime_embedding_repeat.processed_batch_sha256
            != expected_batch_sha256
        ):
            raise PC01CheckpointCompatibilityError(
                "P4 runtime receipt is bound to another processed batch"
            )
        embedding_evidence = {
            "processed_post_batch_sha256": expected_batch_sha256,
            "shape": [1, 768],
            "dimension": 768,
            "dtype_after_export": "float32",
            "embedding_sha256": runtime_embedding.embedding_sha256,
            "embedding_values": list(direct_values),
            "direct_tensor_sha256": _tensor_mapping_identity(
                {"embedding": direct_embedding}, torch=torch
            )["sha256"],
            "repeat_tensor_sha256": _tensor_mapping_identity(
                {"embedding": repeated_embedding}, torch=torch
            )["sha256"],
            "l2_norm": norm,
            "finite": True,
            "nonzero": True,
            "direct_runtime_exact": True,
            "repeat_exact": True,
            "request_sha256": embedding_request.record_sha256,
        }
        return fixture["evidence"], {
            "pre_action": action_evidence,
            "post_action_diagnosis": transition_evidence,
            "executed_recovery_assessment": recovery_evidence,
            "p4_memory_embedding": embedding_evidence,
            "checkpoint_forward_executed": True,
            "runtime_callbacks_executed": True,
            "all_parity_checks_passed": True,
        }


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("accelerate", "bitsandbytes", "peft", "transformers"):
        try:
            result[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise PC01CheckpointCompatibilityError(
                f"DGX runtime is missing required package {package}"
            ) from exc
    if result["transformers"] != "4.57.6":
        raise PC01CheckpointCompatibilityError(
            "DGX runtime Transformers differs from registered PC-01"
        )
    return result


def _validate_state_identity(value: object, *, label: str) -> dict[str, Any]:
    expected = {"sha256", "tensor_count", "total_tensor_bytes"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PC01CheckpointCompatibilityError(f"{label} identity is malformed")
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA-256")
    tensor_count = value.get("tensor_count")
    total_bytes = value.get("total_tensor_bytes")
    if type(tensor_count) is not int or tensor_count <= 0:
        raise PC01CheckpointCompatibilityError(
            f"{label} tensor count must be a positive exact integer"
        )
    if type(total_bytes) is not int or total_bytes <= 0:
        raise PC01CheckpointCompatibilityError(
            f"{label} tensor bytes must be a positive exact integer"
        )
    return {
        "sha256": digest,
        "tensor_count": tensor_count,
        "total_tensor_bytes": total_bytes,
    }


def _validate_inference_stage(
    value: object,
    *,
    label: str,
    batch_fields: set[str],
) -> dict[str, Any]:
    hash_fields = {
        *batch_fields,
        "direct_raw_tensor_sha256",
        "repeat_raw_tensor_sha256",
        "direct_semantic_sha256",
        "runtime_semantic_sha256",
        "repeat_semantic_sha256",
    }
    expected_fields = {*hash_fields, "direct_runtime_exact", "repeat_exact"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise PC01CheckpointCompatibilityError(
            f"{label} inference evidence fields changed"
        )
    for field in sorted(hash_fields):
        _require_sha256(value.get(field), label=f"{label}.{field}")
    if value.get("direct_runtime_exact") is not True or value.get("repeat_exact") is not True:
        raise PC01CheckpointCompatibilityError(f"{label} parity flags are false")
    if value["direct_raw_tensor_sha256"] != value["repeat_raw_tensor_sha256"]:
        raise PC01CheckpointCompatibilityError(
            f"{label} direct/repeat raw tensor hashes differ"
        )
    semantic_hashes = {
        value["direct_semantic_sha256"],
        value["runtime_semantic_sha256"],
        value["repeat_semantic_sha256"],
    }
    if len(semantic_hashes) != 1:
        raise PC01CheckpointCompatibilityError(
            f"{label} direct/runtime/repeat semantic hashes differ"
        )
    return dict(value)


def _float32_embedding_tensor_sha256(values: list[float]) -> str:
    raw = struct.pack("<768f", *values)
    metadata = {
        "key": "embedding",
        "dtype": "float32",
        "shape": [1, 768],
        "num_bytes": len(raw),
    }
    encoded = canonical_json_bytes(metadata)
    digest = hashlib.sha256()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _validate_receipt_shape(value: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "artifact_role",
        "gate_id",
        "status",
        "scope",
        "paper_table_status",
        "model_id",
        "model_seed",
        "checkpoint_epoch",
        "executed_at_utc",
        "artifact_bindings",
        "source_attestation",
        "host_attestation",
        "model_load",
        "checkpoint_restore",
        "fixture",
        "inference",
        "causal_and_scientific_boundaries",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility receipt fields changed"
        )
    _exact_mapping(
        value,
        {
            "schema_version": PC01_CHECKPOINT_COMPATIBILITY_SCHEMA,
            "artifact_role": PC01_CHECKPOINT_COMPATIBILITY_ROLE,
            "gate_id": PC01_CHECKPOINT_COMPATIBILITY_GATE_ID,
            "status": "PASS",
            "scope": "PROVISIONAL_ENGINEERING_PILOT_COMPATIBILITY_ONLY",
            "paper_table_status": "N/R",
            "model_id": PC01_MODEL_ID,
            "model_seed": PC01_MODEL_SEED,
            "checkpoint_epoch": PC01_CHECKPOINT_EPOCH,
        },
        label="checkpoint compatibility receipt",
    )
    try:
        timestamp = datetime.fromisoformat(str(value.get("executed_at_utc")))
    except ValueError as exc:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility timestamp is invalid"
        ) from exc
    if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility timestamp must use UTC"
        )
    artifacts = value.get("artifact_bindings")
    required_artifact_hashes = {
        "export_manifest_sha256",
        "checkpoint_sha256",
        "resolved_config_sha256",
        "processor_contract_sha256",
        "processor_artifact_manifest_sha256",
        "processor_parity_receipt_sha256",
        "base_snapshot_manifest_sha256",
        "base_snapshot_directory_payload_sha256",
        "training_environment_record_sha256",
        "training_environment_source_sha256",
        "training_source_manifest_sha256",
        "training_action_value_evidence_sha256",
        "report_sha256",
        "run_contract_sha256",
    }
    artifact_fields = {*required_artifact_hashes, "checkpoint_size_bytes"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_fields:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility artifact binding fields changed"
        )
    for field in required_artifact_hashes:
        _require_sha256(artifacts[field], label=f"artifact binding {field}")
    _exact_mapping(
        artifacts,
        {
            "export_manifest_sha256": PC01_REGISTERED_EXPORT_MANIFEST_SHA256,
            "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
            "resolved_config_sha256": PC01_EXPECTED_CONFIG_SHA256,
            "base_snapshot_directory_payload_sha256": (
                PC01_EXPECTED_BASE_SNAPSHOT_SHA256
            ),
            "training_environment_source_sha256": (
                PC01_TRAINING_ENVIRONMENT_SHA256
            ),
        },
        label="checkpoint compatibility artifact bindings",
    )
    if (
        type(artifacts.get("checkpoint_size_bytes")) is not int
        or artifacts["checkpoint_size_bytes"] != PC01_EXPECTED_CHECKPOINT_SIZE_BYTES
    ):
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility checkpoint size is invalid"
        )
    source = value.get("source_attestation")
    source_fields = {
        "git_commit",
        "git_clean",
        "source_file_count",
        "source_files",
        "source_manifest_sha256",
    }
    if not isinstance(source, Mapping) or set(source) != source_fields:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility source-attestation fields changed"
        )
    commit = source.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or source.get("git_clean") is not True
        or source.get("source_file_count") != len(_SOURCE_PATHS)
    ):
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility source attestation is invalid"
        )
    source_files = source.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != len(_SOURCE_PATHS):
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility source-file coverage changed"
        )
    if tuple(
        row.get("path") if isinstance(row, Mapping) else None
        for row in source_files
    ) != _SOURCE_PATHS:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility source-file order/set changed"
        )
    for row in source_files:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise PC01CheckpointCompatibilityError(
                "checkpoint compatibility source-file row is malformed"
            )
        _require_sha256(row.get("sha256"), label=f"source file {row.get('path')}")
        if type(row.get("size_bytes")) is not int or row["size_bytes"] <= 0:
            raise PC01CheckpointCompatibilityError(
                f"checkpoint compatibility source size is invalid: {row.get('path')}"
            )
    source_manifest_sha256 = sha256_bytes(canonical_json_bytes(source_files))
    if source.get("source_manifest_sha256") != source_manifest_sha256:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility source manifest hash differs from its rows"
        )
    host = value.get("host_attestation")
    host_fields = {
        "host_role",
        "architecture",
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_device_capability",
        "cuda_device_total_memory_bytes",
        "package_versions",
    }
    if not isinstance(host, Mapping) or set(host) != host_fields:
        raise PC01CheckpointCompatibilityError("host-attestation fields changed")
    _exact_mapping(
        host,
        {
            "host_role": "DGX_MODEL_PLANE",
            "architecture": "aarch64",
            "python_version": PC01_REGISTERED_PYTHON_VERSION,
            "torch_version": PC01_REGISTERED_TORCH_VERSION,
            "cuda_runtime_version": "13.0",
            "cuda_available": True,
            "cuda_device_index": 0,
            "cuda_device_name": PC01_REGISTERED_GPU_NAME,
            "cuda_device_total_memory_bytes": PC01_REGISTERED_GPU_MEMORY_BYTES,
        },
        label="checkpoint compatibility host",
    )
    if type(host.get("cudnn_version")) is not int or host["cudnn_version"] <= 0:
        raise PC01CheckpointCompatibilityError("host cuDNN version is invalid")
    if type(host.get("cuda_device_count")) is not int or host["cuda_device_count"] < 1:
        raise PC01CheckpointCompatibilityError("host CUDA device count is invalid")
    capability = host.get("cuda_device_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(item) is not int or item < 0 for item in capability)
    ):
        raise PC01CheckpointCompatibilityError("host CUDA capability is invalid")
    package_versions = host.get("package_versions")
    if not isinstance(package_versions, Mapping) or set(package_versions) != {
        "accelerate",
        "bitsandbytes",
        "peft",
        "transformers",
    }:
        raise PC01CheckpointCompatibilityError("host package-version coverage changed")
    if any(
        not isinstance(version, str) or not version.strip()
        for version in package_versions.values()
    ) or package_versions.get("transformers") != "4.57.6":
        raise PC01CheckpointCompatibilityError("host package-version evidence is invalid")
    model_load = value.get("model_load")
    if not isinstance(model_load, Mapping) or set(model_load) != {
        "selected_checkpoint",
        "e0_unadapted_base",
    }:
        raise PC01CheckpointCompatibilityError("model-load evidence fields changed")
    for role in ("selected_checkpoint", "e0_unadapted_base"):
        row = model_load.get(role)
        expected_row_fields = {
            "class",
            "eval_mode",
            "frozen",
            "all_parameters_cuda",
            "parameter_count",
            "module_count",
        }
        if role == "e0_unadapted_base":
            expected_row_fields |= {
                "adaptation_loaded",
                "task_heads_loaded",
                "forbidden_state_hits",
            }
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise PC01CheckpointCompatibilityError(
                f"model-load evidence fields changed for {role}"
            )
        _exact_mapping(
            row,
            {"eval_mode": True, "frozen": True, "all_parameters_cuda": True},
            label=f"model-load {role}",
        )
        if (
            not isinstance(row.get("class"), str)
            or not row["class"].strip()
            or type(row.get("parameter_count")) is not int
            or row["parameter_count"] <= 0
            or type(row.get("module_count")) is not int
            or row["module_count"] <= 0
        ):
            raise PC01CheckpointCompatibilityError(
                f"model-load class/count evidence is invalid for {role}"
            )
    if model_load["selected_checkpoint"]["class"] != (
        "web_agent.models.model.WebAgentModel"
    ):
        raise PC01CheckpointCompatibilityError(
            "selected checkpoint loaded through another model class"
        )
    e0_class = model_load["e0_unadapted_base"]["class"]
    if (
        not e0_class.endswith(".Qwen2VLForConditionalGeneration")
        or e0_class.casefold().startswith("peft.")
        or "peftmodel" in e0_class.casefold()
    ):
        raise PC01CheckpointCompatibilityError("E0 loaded through an adapted model class")
    _exact_mapping(
        model_load["e0_unadapted_base"],
        {
            "adaptation_loaded": False,
            "task_heads_loaded": False,
            "forbidden_state_hits": [],
        },
        label="model-load E0",
    )
    restore = value.get("checkpoint_restore")
    restore_fields = {
        "roles",
        "all_roles_exact",
        "selected_state_sha256_before",
        "selected_state_sha256_after",
        "selected_state_tensor_count",
        "selected_state_tensor_bytes",
        "state_unchanged_after_inference",
    }
    if not isinstance(restore, Mapping) or set(restore) != restore_fields:
        raise PC01CheckpointCompatibilityError(
            "checkpoint restore/immutability fields changed"
        )
    roles = restore.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(_CHECKPOINT_STATE_ROLES):
        raise PC01CheckpointCompatibilityError(
            "checkpoint restore role coverage changed"
        )
    for role in _CHECKPOINT_STATE_ROLES:
        row = roles[role]
        if not isinstance(row, Mapping) or set(row) != {
            "checkpoint",
            "loaded",
            "exact_match",
        }:
            raise PC01CheckpointCompatibilityError(
                f"checkpoint restore role {role} is malformed"
            )
        checkpoint_identity = _validate_state_identity(
            row["checkpoint"], label=f"checkpoint restore {role}.checkpoint"
        )
        loaded_identity = _validate_state_identity(
            row["loaded"], label=f"checkpoint restore {role}.loaded"
        )
        if row.get("exact_match") is not True or checkpoint_identity != loaded_identity:
            raise PC01CheckpointCompatibilityError(
                f"checkpoint restore role {role} is not exact"
            )
    for field in ("selected_state_sha256_before", "selected_state_sha256_after"):
        _require_sha256(restore.get(field), label=f"checkpoint restore {field}")
    if (
        restore.get("all_roles_exact") is not True
        or restore.get("state_unchanged_after_inference") is not True
        or restore.get("selected_state_sha256_before")
        != restore.get("selected_state_sha256_after")
        or type(restore.get("selected_state_tensor_count")) is not int
        or restore["selected_state_tensor_count"] <= 0
        or type(restore.get("selected_state_tensor_bytes")) is not int
        or restore["selected_state_tensor_bytes"] <= 0
    ):
        raise PC01CheckpointCompatibilityError(
            "checkpoint restore/immutability evidence is invalid"
        )
    fixture = value.get("fixture")
    fixture_fields = {
        "fixture_id",
        "generator_source_sha256",
        "width",
        "height",
        "screenshot_sha256",
        "task_sha256",
        "transition_sha256",
        "recovery_transition_sha256",
        "embedding_request_sha256",
        "contains_webarena_task_content",
        "contains_oracle_content",
    }
    if not isinstance(fixture, Mapping) or set(fixture) != fixture_fields:
        raise PC01CheckpointCompatibilityError("compatibility fixture fields changed")
    _exact_mapping(
        fixture,
        {
            "fixture_id": "pc01-causal-three-state-ppm-v1",
            "width": 224,
            "height": 224,
            "contains_webarena_task_content": False,
            "contains_oracle_content": False,
        },
        label="checkpoint compatibility fixture",
    )
    for field in (
        "generator_source_sha256",
        "task_sha256",
        "transition_sha256",
        "recovery_transition_sha256",
        "embedding_request_sha256",
    ):
        _require_sha256(fixture.get(field), label=f"compatibility fixture {field}")
    screenshots = fixture.get("screenshot_sha256")
    if (
        not isinstance(screenshots, list)
        or len(screenshots) != 3
        or len(set(screenshots)) != 3
        or any(not _is_sha256(item) for item in screenshots)
    ):
        raise PC01CheckpointCompatibilityError(
            "compatibility fixture screenshot identities are invalid"
        )
    source_by_path = {row["path"]: row for row in source_files}
    if fixture["generator_source_sha256"] != source_by_path[
        "src/web_agent/eval/table2/pc01_checkpoint_compatibility.py"
    ]["sha256"]:
        raise PC01CheckpointCompatibilityError(
            "compatibility fixture generator differs from attested source"
        )
    inference = value.get("inference")
    inference_fields = {
        "pre_action",
        "post_action_diagnosis",
        "executed_recovery_assessment",
        "p4_memory_embedding",
        "checkpoint_forward_executed",
        "runtime_callbacks_executed",
        "all_parity_checks_passed",
    }
    if not isinstance(inference, Mapping) or set(inference) != inference_fields:
        raise PC01CheckpointCompatibilityError("inference evidence fields changed")
    _exact_mapping(
        inference,
        {
            "checkpoint_forward_executed": True,
            "runtime_callbacks_executed": True,
            "all_parity_checks_passed": True,
        },
        label="checkpoint compatibility inference",
    )
    pre_action = _validate_inference_stage(
        inference.get("pre_action"),
        label="pre-action",
        batch_fields={"processed_batch_sha256"},
    )
    post_action = _validate_inference_stage(
        inference.get("post_action_diagnosis"),
        label="post-action diagnosis",
        batch_fields={
            "processed_pre_batch_sha256",
            "processed_post_batch_sha256",
        },
    )
    recovery = _validate_inference_stage(
        inference.get("executed_recovery_assessment"),
        label="executed recovery assessment",
        batch_fields={
            "processed_pre_batch_sha256",
            "processed_post_batch_sha256",
            "processed_recovery_batch_sha256",
        },
    )
    if pre_action["processed_batch_sha256"] != post_action[
        "processed_pre_batch_sha256"
    ]:
        raise PC01CheckpointCompatibilityError(
            "pre-action and transition pre-stream batch hashes differ"
        )
    if recovery["processed_pre_batch_sha256"] == recovery[
        "processed_post_batch_sha256"
    ]:
        raise PC01CheckpointCompatibilityError(
            "recovery fixture pre/post batches unexpectedly have the same identity"
        )
    embedding = inference.get("p4_memory_embedding")
    embedding_fields = {
        "processed_post_batch_sha256",
        "shape",
        "dimension",
        "dtype_after_export",
        "embedding_sha256",
        "embedding_values",
        "direct_tensor_sha256",
        "repeat_tensor_sha256",
        "l2_norm",
        "finite",
        "nonzero",
        "direct_runtime_exact",
        "repeat_exact",
        "request_sha256",
    }
    if not isinstance(embedding, Mapping) or set(embedding) != embedding_fields:
        raise PC01CheckpointCompatibilityError("P4 embedding evidence fields changed")
    _exact_mapping(
        embedding,
        {
            "shape": [1, 768],
            "dimension": 768,
            "dtype_after_export": "float32",
            "finite": True,
            "nonzero": True,
            "direct_runtime_exact": True,
            "repeat_exact": True,
        },
        label="P4 compatibility evidence",
    )
    for field in (
        "processed_post_batch_sha256",
        "embedding_sha256",
        "direct_tensor_sha256",
        "repeat_tensor_sha256",
        "request_sha256",
    ):
        _require_sha256(embedding.get(field), label=f"P4 compatibility {field}")
    embedding_values = embedding.get("embedding_values")
    if (
        not isinstance(embedding_values, list)
        or len(embedding_values) != 768
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in embedding_values
        )
    ):
        raise PC01CheckpointCompatibilityError(
            "P4 embedding values are not 768 finite numbers"
        )
    expected_embedding_sha256 = float32_vector_sha256(embedding_values)
    expected_tensor_sha256 = _float32_embedding_tensor_sha256(
        [float(item) for item in embedding_values]
    )
    expected_norm = math.hypot(*(float(item) for item in embedding_values))
    if (
        embedding["direct_tensor_sha256"] != embedding["repeat_tensor_sha256"]
        or embedding["direct_tensor_sha256"] != expected_tensor_sha256
        or embedding["embedding_sha256"] != expected_embedding_sha256
        or embedding["processed_post_batch_sha256"]
        != post_action["processed_post_batch_sha256"]
        or embedding["request_sha256"] != fixture["embedding_request_sha256"]
        or isinstance(embedding.get("l2_norm"), bool)
        or not isinstance(embedding.get("l2_norm"), (int, float))
        or not math.isfinite(float(embedding["l2_norm"]))
        or float(embedding["l2_norm"]) <= 1e-12
        or not math.isclose(
            float(embedding["l2_norm"]),
            expected_norm,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PC01CheckpointCompatibilityError(
            "P4 embedding hash/norm/request bindings are invalid"
        )
    boundaries = value.get("causal_and_scientific_boundaries")
    boundary_fields = {
        "offline_mode_enforced",
        "network_access_used",
        "weight_updates_performed",
        "training_rows_read",
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
        "webarena_tasks_executed",
        "task_registry_frozen",
        "pc01_final_backbone_promoted",
        "table2_populated",
        "action_value_text_omitted",
    }
    if not isinstance(boundaries, Mapping) or set(boundaries) != boundary_fields:
        raise PC01CheckpointCompatibilityError(
            "scientific-boundary evidence fields changed"
        )
    _exact_mapping(
        boundaries,
        {
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
        label="checkpoint compatibility scientific boundaries",
    )
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility receipt is not canonicalizable"
        ) from exc


def validate_pc01_checkpoint_compatibility_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate a PASS receipt and optional external export/source bindings."""

    _validate_receipt_shape(receipt)
    if receipt["artifact_bindings"]["export_manifest_sha256"] != (
        PC01_REGISTERED_EXPORT_MANIFEST_SHA256
    ):
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility receipt cites an unregistered v3 export"
        )
    if expected_source_commit is not None and receipt["source_attestation"].get(
        "git_commit"
    ) != expected_source_commit:
        raise PC01CheckpointCompatibilityError(
            "checkpoint compatibility receipt cites another source commit"
        )
    return dict(receipt)


def _write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(receipt)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PC01CheckpointCompatibilityError(
            f"refusing to overwrite compatibility receipt: {destination}"
        ) from exc
    destination.chmod(0o444)
    return destination


def run_pc01_checkpoint_compatibility(
    *,
    export_dir: str | Path,
    checkpoint_path: str | Path,
    base_snapshot_path: str | Path,
    report_path: str | Path,
    run_contract_path: str | Path,
    repository_root: str | Path,
    expected_source_commit: str,
    output_path: str | Path,
) -> Path:
    """Run the real PC-01 checkpoint gate and create one immutable PASS receipt.

    The CUDA/DGX check is intentionally first.  Calling this function on a CPU
    host or the local 2 GB MX450 cannot deserialize the checkpoint, load a model,
    or write a receipt that resembles DGX evidence.
    """

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise PC01CheckpointCompatibilityError(
            f"refusing to overwrite compatibility receipt: {output}"
        )
    torch, host = _require_registered_dgx_cuda()
    inputs = _validate_pc01_export_bundle(
        export_dir=export_dir,
        checkpoint_path=checkpoint_path,
        base_snapshot_path=base_snapshot_path,
        report_path=report_path,
        run_contract_path=run_contract_path,
        repository_root=repository_root,
        expected_source_commit=expected_source_commit,
    )
    host["package_versions"] = _package_versions()

    from web_agent.utils.seed import set_seed

    set_seed(PC01_MODEL_SEED, deterministic=True)
    artifacts = PC01RuntimeArtifacts(
        resolved_config_path=inputs.export_dir / "resolved_config.json",
        processor_contract_path=inputs.export_dir / "processor_contract.json",
        processor_source=inputs.base_snapshot_path,
        e0_resolved_config_path=inputs.export_dir / "e0_resolved_config.json",
        e0_processor_contract_path=inputs.export_dir / "e0_processor_contract.json",
        e0_backbone_path=inputs.base_snapshot_path,
        export_manifest_path=inputs.export_dir / "pc01_export_manifest.json",
        training_action_value_evidence_path=(
            inputs.export_dir / "training_action_value_evidence.json"
        ),
    )
    selected = ValidationSelectedCheckpoint(
        manifest_id="pc01-epoch6-seed42-checkpoint-compatibility",
        model_seed=PC01_MODEL_SEED,
        checkpoint_path=inputs.checkpoint_path,
        selected_checkpoint_sha256=PC01_EXPECTED_CHECKPOINT_SHA256,
        resolved_config_sha256=PC01_EXPECTED_CONFIG_SHA256,
        processor_contract_sha256=inputs.processor_contract.record_sha256,
        validation_rows_read=int(inputs.report["val_rows"]),
        checkpoint_selection="validation_only",
        selection_scope="validation_only",
        test_rows_read=0,
        locked_test_rows_read=0,
    )
    e0_identity = load_resolved_config_identity(
        inputs.export_dir / "e0_resolved_config.json"
    )
    e0 = ValidationSelectedBackbone(
        manifest_id="pc01-epoch6-seed42-e0-checkpoint-compatibility",
        backbone_id="Qwen/Qwen2-VL-2B-Instruct",
        backbone_revision=PC01_MODEL_REVISION,
        backbone_path=inputs.base_snapshot_path,
        backbone_sha256=PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
        resolved_config_sha256=e0_identity.payload_sha256,
        processor_contract_sha256=inputs.processor_contract.record_sha256,
        base_prompt_sha256=sha256_bytes(E0_ACTION_PROMPT.encode("utf-8")),
        parser_id=E0_PARSER_ID,
        parser_version=E0_PARSER_VERSION,
        validation_rows_read=int(inputs.report["val_rows"]),
        selection_scope="validation_only",
        test_rows_read=0,
        locked_test_rows_read=0,
    )

    with _strict_offline_hf():
        model, processor, loaded_torch, config, loaded_contract = _load_selected_model(
            selection=selected, artifacts=artifacts
        )
        if loaded_torch is not torch:
            raise PC01CheckpointCompatibilityError(
                "selected runtime imported another PyTorch module instance"
            )
        selected_load = _assert_eval_frozen_cuda(
            model, label="selected PC-01 checkpoint"
        )
        selected_names = tuple(name for name, _ in model.named_modules())
        if not all(
            any(token in name for name in selected_names)
            for token in (
                "task_adapters",
                "failure_head",
                "action_head",
                "memory_head",
                "recovery_outcome_head",
            )
        ):
            raise PC01CheckpointCompatibilityError(
                "selected model lacks one or more registered task-head modules"
            )
        restore = _verify_checkpoint_restore(
            model=model, checkpoint=inputs.checkpoint_payload, torch=torch
        )
        torch.cuda.synchronize(0)
        selected_state_before = _state_mapping_identity(
            model.state_dict(), torch=torch
        )

        e0_model, e0_torch = _load_unadapted_base(
            selection=e0,
            artifacts=artifacts,
            selected_config=config,
            processor=processor,
            processor_contract=loaded_contract,
        )
        if e0_torch is not torch:
            raise PC01CheckpointCompatibilityError(
                "E0 runtime imported another PyTorch module instance"
            )
        e0_load = _assert_e0_unadapted(e0_model)

        fixture_evidence, inference = _run_inference_checks(
            model=model,
            processor=processor,
            torch=torch,
            config=config,
            processor_contract=loaded_contract,
        )
        torch.cuda.synchronize(0)
        selected_state_after = _state_mapping_identity(model.state_dict(), torch=torch)
    if selected_state_before != selected_state_after:
        raise PC01CheckpointCompatibilityError(
            "selected model state changed during inference-only compatibility checks"
        )
    restore = {
        **restore,
        "selected_state_sha256_before": selected_state_before["sha256"],
        "selected_state_sha256_after": selected_state_after["sha256"],
        "selected_state_tensor_count": selected_state_before["tensor_count"],
        "selected_state_tensor_bytes": selected_state_before["total_tensor_bytes"],
        "state_unchanged_after_inference": True,
    }
    receipt = {
        "schema_version": PC01_CHECKPOINT_COMPATIBILITY_SCHEMA,
        "artifact_role": PC01_CHECKPOINT_COMPATIBILITY_ROLE,
        "gate_id": PC01_CHECKPOINT_COMPATIBILITY_GATE_ID,
        "status": "PASS",
        "scope": "PROVISIONAL_ENGINEERING_PILOT_COMPATIBILITY_ONLY",
        "paper_table_status": "N/R",
        "model_id": PC01_MODEL_ID,
        "model_seed": PC01_MODEL_SEED,
        "checkpoint_epoch": PC01_CHECKPOINT_EPOCH,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_bindings": dict(inputs.artifact_bindings),
        "source_attestation": dict(inputs.source_attestation),
        "host_attestation": host,
        "model_load": {
            "selected_checkpoint": selected_load,
            "e0_unadapted_base": e0_load,
        },
        "checkpoint_restore": restore,
        "fixture": fixture_evidence,
        "inference": inference,
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
    validated = validate_pc01_checkpoint_compatibility_receipt(
        receipt,
        expected_source_commit=expected_source_commit,
    )
    destination = _write_receipt(output, validated)
    # Verify durable bytes before reporting success.
    reopened = _read_unique_json(
        destination, label="checkpoint compatibility receipt", canonical=True
    )
    validate_pc01_checkpoint_compatibility_receipt(
        reopened,
        expected_source_commit=expected_source_commit,
    )
    return destination
