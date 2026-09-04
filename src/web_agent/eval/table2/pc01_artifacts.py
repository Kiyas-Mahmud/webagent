"""Fail-closed export of the PC-01 model artifacts used by Table 2.

The training checkpoint is the authority for the resolved model configuration.
This module deliberately imports neither PyTorch nor Transformers at import
time.  Those optional dependencies are loaded only after all cheap path,
report, and run-contract checks pass.

No network fallback is permitted.  A processor must be supplied as a local
Hugging Face snapshot whose embedded/path revision proves that it is the exact
revision registered by the PC-01 run contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from web_agent.eval.table2.common import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from web_agent.runtime.observation import ProcessorParityContract


PC01_MODEL_ID = "qwen2vl_2b_gold_v2_8_dgx"
PC01_CONFIG_NAME = "Y_QWEN2VL_2B_GOLD_V2_8_DGX"
PC01_SAVED_FULL_CONFIG_NAME = "Y_QWEN2VL_2B_GOLD_V2_8_DGX_FULL_SEED42"
PC01_MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
PC01_MODEL_SEED = 42
PC01_TRANSFORMERS_VERSION = "4.57.6"
PC01_PROCESSOR_CLASS = (
    "transformers.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor"
)
PC01_IMAGE_PROCESSOR_CLASS = (
    "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast."
    "Qwen2VLImageProcessorFast"
)
PC01_TRAINING_GIT_COMMIT = "2fadf0f508cec42ce6f89b8961db7cfd2adef1df"
PC01_TRAINING_ENVIRONMENT_SHA256 = (
    "d72c7e12c8bdf23098b81626186bce54d892113cc66a018dd3a1b89cb69d75d5"
)
PC01_TRAINING_SOURCE_SHA256 = {
    "src/web_agent/data/gold_dataset.py": (
        "c99a0684d744d803e19b4167e67dd95dd10ea9eeebfdc330fe8b0786e35b3a8d"
    ),
    "src/web_agent/data/gold_dataloader.py": (
        "ccb9534d1bc81d6e6a7386f5a53e76fc4afa5a44a513a5a0d47f3387120e0999"
    ),
    "src/web_agent/data/recovery_transitions.py": (
        "e0145a9b81eb9d23a61d2560454d7e4b33d884ac0ee5188d259e0fe7198182db"
    ),
    "src/web_agent/train/gold_stages.py": (
        "601b76a14e28885b254ff609484532bd4777d3fc95e356db8fa46394dd977216"
    ),
    "src/web_agent/models/encoders/vlm_contract.py": (
        "47e0bb10b5bbeff37f94df39bafdf528cbf9d963a5b7842fbab74e082ec1a47a"
    ),
}
PC01_EXPECTED_SNAPSHOT_FILES = frozenset(
    {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
PC01_EXPECTED_CONFIG_SHA256 = (
    "d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f"
)
PC01_EXPECTED_CHECKPOINT_SHA256 = (
    "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a"
)
PC01_EXPECTED_BASE_SNAPSHOT_SHA256 = (
    "e002f8290faa3e9f44bf3099eac85a2445e17de738c5bb0cc10d342da837c46c"
)
PC01_EXPECTED_EXPORT_MANIFEST_SHA256 = (
    "63c01942cc653732c9e9e18639cb49bded09a82235fd4ea37fef3fad14c9fa2d"
)
PC01_PRIMARY_TRAIN_SHA256 = (
    "0522807d74256fa303a20d5b8f3bb1653de44aad187660a655580c708c7ae3de"
)
PC01_SUPPLEMENT_TRAIN_SHA256 = (
    "67ade5e971e8fa2f58ec8a269b1e476ae8cc2349a89eb3e3e0d5e161c47c1098"
)
PC01_PRIMARY_TRAIN_ROWS = 23_499
PC01_SUPPLEMENT_TRAIN_ROWS = 608
PC01_TOTAL_TRAIN_ROWS = PC01_PRIMARY_TRAIN_ROWS + PC01_SUPPLEMENT_TRAIN_ROWS
PC01_TRAINING_ACTION_VALUE_SCHEMA = (
    "table2.pc01-training-action-value-evidence.v1"
)

PC01_REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "lora",
        "adapter",
        "task_adapters",
        "failure",
        "action",
        "memory",
        "recovery_outcome",
        "config",
    }
)

PRE_ACTION_FIELD_MAPPING = {
    "task.goal": "text.current_task",
    "observation.url.hostname": "text.domain",
    "observation.screenshot_path": "images[0]",
}
POST_ACTION_FIELD_MAPPING = {
    "task.goal": "text.current_task",
    "transition.pre_observation.url.hostname": "text.domain",
    "transition.pre_observation.screenshot_path": "images[0]",
    "transition.executed_action.action_type": "text.executed_action",
    "transition.post_observation.screenshot_path": "images[1]",
}

_KNOWN_PROCESSOR_FILES = frozenset(
    {
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    }
)
_MODEL_WEIGHT_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".gguf",
        ".h5",
        ".onnx",
        ".ot",
        ".pt",
        ".pth",
        ".safetensors",
    }
)


class PC01ArtifactError(RuntimeError):
    """A PC-01 artifact is absent, ambiguous, mutable, or inconsistent."""


CheckpointLoader = Callable[[Path], Any]
ProcessorLoader = Callable[[Path, Mapping[str, Any]], Any]
VersionLoader = Callable[[], str]


@dataclass(frozen=True, slots=True)
class LoadedPinnedProcessor:
    """A locally loaded processor and the byte-derived parity evidence."""

    processor: Any
    contract: ProcessorParityContract
    artifact_manifest: Mapping[str, Any]
    implementation_identity: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PC01ExportResult:
    output_dir: Path
    resolved_config_path: Path
    processor_contract_path: Path
    e0_processor_contract_path: Path
    e0_resolved_config_path: Path
    processor_parity_receipt_path: Path
    base_snapshot_manifest_path: Path
    training_environment_path: Path
    training_source_manifest_path: Path
    training_action_value_evidence_path: Path
    manifest_path: Path
    config_sha256: str
    checkpoint_sha256: str
    processor_contract_sha256: str


@dataclass(frozen=True, slots=True)
class PC01CheckpointIdentityResult:
    """Config-only export that is independent of base-model/processor supply."""

    output_dir: Path
    resolved_config_path: Path
    identity_receipt_path: Path
    config_sha256: str
    checkpoint_sha256: str


def _require_regular_file(path: str | Path, *, label: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise PC01ArtifactError(f"{label} is absent or inaccessible: {source}") from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise PC01ArtifactError(f"{label} must be a regular non-symlink file: {source}")
    return source.resolve()


def _require_local_directory(path: str | Path, *, label: str) -> Path:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise PC01ArtifactError(f"{label} is absent or inaccessible: {source}") from exc
    if source.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise PC01ArtifactError(f"{label} must be a local non-symlink directory: {source}")
    return source.resolve()


def _read_unique_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _require_regular_file(path, label=label)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PC01ArtifactError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except PC01ArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PC01ArtifactError(f"cannot read valid JSON from {label}: {source}") from exc
    if not isinstance(parsed, dict):
        raise PC01ArtifactError(f"{label} must contain one JSON object")
    return parsed


def _read_unique_json_array(path: str | Path, *, label: str) -> list[Any]:
    """Read one JSON array while rejecting duplicate object keys."""

    source = _require_regular_file(path, label=label)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PC01ArtifactError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except PC01ArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PC01ArtifactError(f"cannot read valid JSON from {label}: {source}") from exc
    if not isinstance(parsed, list):
        raise PC01ArtifactError(f"{label} must contain one JSON array")
    return parsed


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def registered_pc01_training_source_manifest() -> dict[str, Any]:
    """Return the immutable preprocessing-source identities from the training commit."""

    return {
        "schema_version": "table2.pc01-training-sources.v1",
        "git_commit": PC01_TRAINING_GIT_COMMIT,
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(PC01_TRAINING_SOURCE_SHA256.items())
        ],
    }


def validate_pc01_training_sources(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Require current preprocessing sources to equal the checked-in training commit."""

    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[4]
    )
    manifest = registered_pc01_training_source_manifest()
    for item in manifest["files"]:
        source = _require_regular_file(
            root / item["path"],
            label=f"PC-01 training source {item['path']}",
        )
        if sha256_file(source) != item["sha256"]:
            raise PC01ArtifactError(
                "current preprocessing source differs from the authenticated "
                f"PC-01 training commit: {item['path']}"
            )
    return manifest


def _assert_checkpoint_materialized(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise PC01ArtifactError(
            "selected checkpoint is still a Git LFS pointer; materialize its bytes first"
        )
    if path.stat().st_size < 1024:
        raise PC01ArtifactError("selected checkpoint is implausibly small or truncated")


def _default_checkpoint_loader(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional DGX stack
        raise PC01ArtifactError(
            "PyTorch is required to inspect the checkpoint; no fallback parser is allowed"
        ) from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - exercised on the DGX artifact
        raise PC01ArtifactError(f"cannot deserialize selected checkpoint: {path}") from exc


def load_checkpoint_saved_config(
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    checkpoint_loader: CheckpointLoader | None = None,
) -> tuple[dict[str, Any], str, Mapping[str, Any]]:
    """Return the exact checkpoint-saved config after structural validation."""

    source = _require_regular_file(checkpoint_path, label="selected checkpoint")
    _assert_checkpoint_materialized(source)
    digest = sha256_file(source)
    if (
        expected_checkpoint_sha256 is not None
        and digest != expected_checkpoint_sha256
    ):
        raise PC01ArtifactError(
            "checkpoint bytes differ from the registered PC-01 checkpoint"
        )
    loader = checkpoint_loader or _default_checkpoint_loader
    try:
        payload = loader(source)
    except PC01ArtifactError:
        raise
    except Exception as exc:
        raise PC01ArtifactError("checkpoint loader rejected the selected checkpoint") from exc
    if not isinstance(payload, Mapping):
        raise PC01ArtifactError("selected checkpoint must contain a state mapping")
    missing = sorted(PC01_REQUIRED_CHECKPOINT_KEYS - set(payload))
    if missing:
        raise PC01ArtifactError(f"selected checkpoint is missing required state: {missing}")
    saved = payload.get("config")
    if not isinstance(saved, Mapping):
        raise PC01ArtifactError("checkpoint config must be a mapping")
    config = deepcopy(dict(saved))
    if "extends" in config:
        raise PC01ArtifactError(
            "checkpoint config is not fully materialized (contains 'extends')"
        )
    try:
        canonical_json_bytes(config)
    except (TypeError, ValueError) as exc:
        raise PC01ArtifactError(
            "checkpoint config must contain finite JSON-compatible values"
        ) from exc
    return config, digest, payload


def _validate_pc01_report_and_contract(
    *,
    report: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    checkpoint_sha256: str,
    config_sha256: str,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
) -> None:
    exact_report = {
        "seed": PC01_MODEL_SEED,
        "stage": "full",
        "status": "PASS",
        "test_rows_read": 0,
        "selection_rule": "all_gates_then_outcome_mcc",
    }
    for field, expected in exact_report.items():
        if report.get(field) != expected:
            raise PC01ArtifactError(
                f"PC-01 report {field} differs: expected {expected!r}, got {report.get(field)!r}"
            )
    if report.get("selected_checkpoint_sha256") != checkpoint_sha256:
        raise PC01ArtifactError("checkpoint bytes differ from the PC-01 report")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise PC01ArtifactError("checkpoint bytes differ from the registered PC-01 checkpoint")

    control = report.get("experiment_control")
    if not isinstance(control, Mapping):
        raise PC01ArtifactError("PC-01 report lacks experiment_control evidence")
    expected_control = {
        "config_sha256": expected_config_sha256,
        "checkpoint_selection_source": "original_gold_validation_only",
        "supplement_validation_selects_checkpoint": False,
        "locked_test_read": False,
    }
    for field, expected in expected_control.items():
        if control.get(field) != expected:
            raise PC01ArtifactError(f"PC-01 experiment_control changed {field}")
    if config_sha256 != expected_config_sha256:
        raise PC01ArtifactError(
            "checkpoint-saved config does not reproduce the report's canonical hash"
        )

    candidate = run_contract.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("model_id") != PC01_MODEL_ID:
        raise PC01ArtifactError("run contract does not identify PC-01")
    exact_contract = {
        "config_name": PC01_CONFIG_NAME,
        "model_revision": PC01_MODEL_REVISION,
        "seed": PC01_MODEL_SEED,
        "checkpoint_selection_source": "original_gold_validation_only",
        "supplement_validation_selects_checkpoint": False,
        "test_rows_read": 0,
    }
    for field, expected in exact_contract.items():
        if run_contract.get(field) != expected:
            raise PC01ArtifactError(f"PC-01 run contract changed {field}")


def _validate_saved_pc01_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("name") != PC01_SAVED_FULL_CONFIG_NAME
        or config.get("seeds") != [PC01_MODEL_SEED]
    ):
        raise PC01ArtifactError("checkpoint config does not identify the registered PC-01 run")
    if int(config.get("fused_dim", -1)) != 768:
        raise PC01ArtifactError("PC-01 checkpoint does not use the registered 768-D fusion")
    data = config.get("data")
    model = config.get("model")
    adapters = model.get("task_adapters") if isinstance(model, Mapping) else None
    if not isinstance(data, Mapping) or data.get("causal_routing") is not True:
        raise PC01ArtifactError("PC-01 checkpoint lacks causal pre/post routing")
    if data.get("use_state_after") is not True or data.get("recovery_transitions") is not True:
        raise PC01ArtifactError("PC-01 checkpoint lacks registered transition inputs")
    if not isinstance(adapters, Mapping) or adapters.get("enabled") is not True:
        raise PC01ArtifactError("PC-01 checkpoint lacks registered task adapters")


def _verified_checkpoint_inputs(
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    run_contract_path: str | Path,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    checkpoint_loader: CheckpointLoader | None,
) -> tuple[dict[str, Any], bytes, str, Path, Path]:
    if not _is_sha256(expected_config_sha256) or not _is_sha256(
        expected_checkpoint_sha256
    ):
        raise PC01ArtifactError("expected PC-01 identities must be lowercase SHA-256")
    report_source = _require_regular_file(report_path, label="PC-01 report")
    contract_source = _require_regular_file(run_contract_path, label="PC-01 run contract")
    report = _read_unique_json(report_source, label="PC-01 report")
    run_contract = _read_unique_json(contract_source, label="PC-01 run contract")
    config, checkpoint_sha256, _ = load_checkpoint_saved_config(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_loader=checkpoint_loader,
    )
    config_bytes = canonical_json_bytes(config)
    config_sha256 = sha256_bytes(config_bytes)
    _validate_pc01_report_and_contract(
        report=report,
        run_contract=run_contract,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    _validate_saved_pc01_config(config)
    return config, config_bytes, checkpoint_sha256, report_source, contract_source


def _processor_file_manifest(
    source: Path,
    *,
    implementation_identity: Mapping[str, Any],
    training_environment_sha256: str,
    training_source_manifest_sha256: str,
    training_action_value_evidence_sha256: str,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    known_found = False
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PC01ArtifactError(
                f"processor snapshot contains a symlink and is not self-contained: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if path.name in _KNOWN_PROCESSOR_FILES:
            known_found = True
        if path.suffix.lower() in _MODEL_WEIGHT_SUFFIXES:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not known_found:
        raise PC01ArtifactError(
            "local snapshot contains none of the registered processor/tokenizer files"
        )
    if not files:
        raise PC01ArtifactError("local processor artifact set is empty")
    return {
        "schema_version": "table2.pc01-processor-artifacts.v3",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "training_environment_sha256": training_environment_sha256,
        "training_source_manifest_sha256": training_source_manifest_sha256,
        "training_action_value_evidence_sha256": (
            training_action_value_evidence_sha256
        ),
        "training_transformers_version": PC01_TRANSFORMERS_VERSION,
        "implementation_identity": dict(implementation_identity),
        "files": files,
    }


def build_pc01_base_snapshot_manifest(
    processor_source: str | Path,
) -> dict[str, Any]:
    """Hash the complete pinned base snapshot without copying or loading weights.

    ``directory_payload_sha256`` intentionally reproduces
    :func:`web_agent.runtime.manifest.sha256_directory`: for every sorted file,
    hash the eight-byte big-endian relative-path length, UTF-8 path, and raw
    payload.  This implementation additionally rejects links, nested
    directories, special files, missing files, and extras before reading the
    4.43 GB production payload.
    """

    source = _require_local_directory(processor_source, label="base-model snapshot")
    paths: list[tuple[str, Path, os.stat_result]] = []
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise PC01ArtifactError(
                f"base-model snapshot entry is inaccessible: {path}"
            ) from exc
        if path.is_symlink():
            raise PC01ArtifactError(
                f"base-model snapshot contains a symlink: {path}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            raise PC01ArtifactError(
                f"base-model snapshot contains an unexpected directory: {path}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise PC01ArtifactError(
                f"base-model snapshot entry is not a regular file: {path}"
            )
        paths.append((path.relative_to(source).as_posix(), path, metadata))

    observed = {relative for relative, _, _ in paths}
    if observed != PC01_EXPECTED_SNAPSHOT_FILES:
        missing = sorted(PC01_EXPECTED_SNAPSHOT_FILES - observed)
        extra = sorted(observed - PC01_EXPECTED_SNAPSHOT_FILES)
        raise PC01ArtifactError(
            "base-model snapshot file set differs from the registered Qwen "
            f"snapshot (missing={missing}, extra={extra})"
        )

    directory_digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    total_size = 0
    for relative, path, before in paths:
        relative_bytes = relative.encode("utf-8")
        directory_digest.update(len(relative_bytes).to_bytes(8, "big"))
        directory_digest.update(relative_bytes)
        file_digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    file_digest.update(block)
                    directory_digest.update(block)
        except OSError as exc:
            raise PC01ArtifactError(
                f"cannot hash base-model snapshot file: {path}"
            ) from exc
        after = path.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise PC01ArtifactError(
                f"base-model snapshot changed while it was being hashed: {path}"
            )
        total_size += after.st_size
        files.append(
            {
                "path": relative,
                "size_bytes": after.st_size,
                "sha256": file_digest.hexdigest(),
            }
        )

    return {
        "schema_version": "table2.pc01-base-snapshot.v1",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "directory_payload_sha256": directory_digest.hexdigest(),
        "files": files,
    }


def load_pc01_base_snapshot_manifest(
    path: str | Path,
    *,
    expected_directory_payload_sha256: str = PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    """Load and structurally authenticate a canonical full-snapshot manifest."""

    source = _require_regular_file(path, label="PC-01 base snapshot manifest")
    value = _read_unique_json(source, label="PC-01 base snapshot manifest")
    if source.read_bytes() != canonical_json_bytes(value):
        raise PC01ArtifactError("PC-01 base snapshot manifest is not canonical")
    exact = {
        "schema_version": "table2.pc01-base-snapshot.v1",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "file_count": len(PC01_EXPECTED_SNAPSHOT_FILES),
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise PC01ArtifactError(f"PC-01 base snapshot manifest changed {field}")
    if set(value) != {
        "schema_version",
        "model_id",
        "revision",
        "file_count",
        "total_size_bytes",
        "directory_payload_sha256",
        "files",
    }:
        raise PC01ArtifactError("PC-01 base snapshot manifest fields changed")
    if not _is_sha256(expected_directory_payload_sha256):
        raise PC01ArtifactError("expected base snapshot identity must be SHA-256")
    if not _is_sha256(value.get("directory_payload_sha256")):
        raise PC01ArtifactError("base snapshot directory identity is not SHA-256")
    if value["directory_payload_sha256"] != expected_directory_payload_sha256:
        raise PC01ArtifactError(
            "base snapshot directory differs from the registered Qwen snapshot"
        )
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(PC01_EXPECTED_SNAPSHOT_FILES):
        raise PC01ArtifactError("base snapshot manifest has the wrong file count")
    observed: list[str] = []
    total_size = 0
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise PC01ArtifactError("base snapshot manifest contains an invalid file row")
        relative = str(item["path"])
        size = item["size_bytes"]
        if type(size) is not int or size < 0 or not _is_sha256(item["sha256"]):
            raise PC01ArtifactError(
                f"base snapshot manifest has invalid metadata for {relative!r}"
            )
        observed.append(relative)
        total_size += size
    if observed != sorted(PC01_EXPECTED_SNAPSHOT_FILES):
        raise PC01ArtifactError("base snapshot manifest file order/set changed")
    if value.get("total_size_bytes") != total_size:
        raise PC01ArtifactError("base snapshot manifest total size is inconsistent")
    return value


def _processor_revision_markers(processor: Any, source: Path) -> set[str]:
    markers: set[str] = set()
    parts = source.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            markers.add(parts[index + 1])
    candidates = (
        processor,
        getattr(processor, "tokenizer", None),
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        direct = getattr(candidate, "_commit_hash", None)
        if isinstance(direct, str) and direct:
            markers.add(direct)
        init_kwargs = getattr(candidate, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping):
            marker = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
            if isinstance(marker, str) and marker:
                markers.add(marker)
    return markers


def _default_processor_loader(source: Path, kwargs: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional DGX stack
        raise PC01ArtifactError(
            "Transformers is required to validate the processor; no network fallback is allowed"
        ) from exc
    try:
        return AutoProcessor.from_pretrained(str(source), **dict(kwargs))
    except Exception as exc:  # pragma: no cover - exercised on the DGX artifact
        raise PC01ArtifactError(
            "cannot load the pinned processor exclusively from local files"
        ) from exc


def _default_transformers_version() -> str:
    try:
        return importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise PC01ArtifactError(
            "Transformers is required to validate the processor"
        ) from exc


def _qualified_class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _processor_implementation_identity(
    processor: Any,
    *,
    transformers_version: str,
) -> dict[str, Any]:
    if transformers_version != PC01_TRANSFORMERS_VERSION:
        raise PC01ArtifactError(
            "PC-01 processor requires transformers=="
            f"{PC01_TRANSFORMERS_VERSION}, got {transformers_version!r}"
        )
    processor_class = _qualified_class_name(processor)
    if processor_class != PC01_PROCESSOR_CLASS:
        raise PC01ArtifactError(
            "loaded processor class differs from the registered PC-01 processor"
        )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise PC01ArtifactError("Qwen2-VL processor lacks an image processor")
    image_processor_class = _qualified_class_name(image_processor)
    if image_processor_class != PC01_IMAGE_PROCESSOR_CLASS:
        raise PC01ArtifactError(
            "loaded image processor is not the registered Qwen2-VL fast processor"
        )
    return {
        "transformers_version": transformers_version,
        "processor_class": processor_class,
        "image_processor_class": image_processor_class,
        "use_fast": True,
    }


def load_pinned_local_processor(
    config: Mapping[str, Any],
    processor_source: str | Path,
    *,
    processor_loader: ProcessorLoader | None = None,
    version_loader: VersionLoader | None = None,
    training_environment_sha256: str = PC01_TRAINING_ENVIRONMENT_SHA256,
    training_source_manifest_sha256: str | None = None,
    training_action_value_evidence_sha256: str | None = None,
) -> LoadedPinnedProcessor:
    """Load and fingerprint the exact local processor registered by ``config``."""

    source = _require_local_directory(processor_source, label="processor snapshot")
    if not _is_sha256(training_environment_sha256):
        raise PC01ArtifactError(
            "processor training-environment identity must be lowercase SHA-256"
        )
    if training_source_manifest_sha256 is None:
        training_source_manifest_sha256 = sha256_bytes(
            canonical_json_bytes(registered_pc01_training_source_manifest())
        )
    if not _is_sha256(training_source_manifest_sha256):
        raise PC01ArtifactError(
            "processor training-source identity must be lowercase SHA-256"
        )
    if not _is_sha256(training_action_value_evidence_sha256):
        raise PC01ArtifactError(
            "processor training action-value evidence must be lowercase SHA-256"
        )
    backbone = config.get("backbone")
    if not isinstance(backbone, Mapping):
        raise PC01ArtifactError("resolved config lacks a backbone mapping")
    exact = {
        "family": "qwen2_vl",
        "path": "vlm",
        "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "trust_remote_code": False,
    }
    for field, expected in exact.items():
        if backbone.get(field) != expected:
            raise PC01ArtifactError(f"resolved PC-01 backbone changed {field}")
    min_pixels = backbone.get("min_pixels")
    max_pixels = backbone.get("max_pixels")
    if type(min_pixels) is not int or type(max_pixels) is not int or min_pixels <= 0:
        raise PC01ArtifactError("PC-01 processor pixel bounds are invalid")
    if max_pixels < min_pixels:
        raise PC01ArtifactError("PC-01 processor maximum pixels precedes minimum")
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "use_fast": True,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    transformers_version = (version_loader or _default_transformers_version)()
    if transformers_version != PC01_TRANSFORMERS_VERSION:
        raise PC01ArtifactError(
            "PC-01 processor requires transformers=="
            f"{PC01_TRANSFORMERS_VERSION}, got {transformers_version!r}"
        )
    loader = processor_loader or _default_processor_loader
    try:
        processor = loader(source, kwargs)
    except PC01ArtifactError:
        raise
    except Exception as exc:
        raise PC01ArtifactError("processor loader rejected the pinned local snapshot") from exc
    if (
        not callable(getattr(processor, "apply_chat_template", None))
        or not callable(getattr(processor, "batch_decode", None))
        or not callable(processor)
    ):
        raise PC01ArtifactError("loaded processor lacks the required multimodal API")
    markers = _processor_revision_markers(processor, source)
    if not markers:
        raise PC01ArtifactError(
            "processor snapshot provides no embedded or snapshot-path revision evidence"
        )
    if markers != {PC01_MODEL_REVISION}:
        raise PC01ArtifactError(
            f"processor revision evidence differs from PC-01: {sorted(markers)}"
        )
    implementation_identity = _processor_implementation_identity(
        processor,
        transformers_version=transformers_version,
    )
    image_processor = processor.image_processor
    for field, expected in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        actual = getattr(image_processor, field, None)
        if actual is None or int(actual) != expected:
            raise PC01ArtifactError(f"loaded processor changed {field}")

    artifact_manifest = _processor_file_manifest(
        source,
        implementation_identity=implementation_identity,
        training_environment_sha256=training_environment_sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_action_value_evidence_sha256=(
            str(training_action_value_evidence_sha256)
        ),
    )
    processor_config_sha256 = sha256_bytes(canonical_json_bytes(artifact_manifest))
    contract = ProcessorParityContract(
        processor_class=implementation_identity["processor_class"],
        processor_revision=PC01_MODEL_REVISION,
        processor_config_sha256=processor_config_sha256,
        pre_action_field_mapping=PRE_ACTION_FIELD_MAPPING,
        post_action_field_mapping=POST_ACTION_FIELD_MAPPING,
    )
    return LoadedPinnedProcessor(
        processor=processor,
        contract=contract,
        artifact_manifest=artifact_manifest,
        implementation_identity=implementation_identity,
    )


def load_pc01_training_environment(
    path: str | Path,
    *,
    run_contract_path: str | Path | None = None,
    expected_sha256: str = PC01_TRAINING_ENVIRONMENT_SHA256,
) -> dict[str, Any]:
    """Authenticate the exact environment evidence attached to the PC-01 run."""

    source = _require_regular_file(path, label="PC-01 training environment")
    if sha256_file(source) != expected_sha256:
        raise PC01ArtifactError(
            "PC-01 training environment bytes differ from registered evidence"
        )
    environment = _read_unique_json(source, label="PC-01 training environment")
    if environment.get("transformers") != PC01_TRANSFORMERS_VERSION:
        raise PC01ArtifactError("PC-01 training Transformers version changed")
    if environment.get("git_commit") != PC01_TRAINING_GIT_COMMIT:
        raise PC01ArtifactError("PC-01 training environment git commit changed")
    if run_contract_path is not None:
        contract = _read_unique_json(
            run_contract_path,
            label="PC-01 run contract",
        )
        if contract.get("git_commit") != environment["git_commit"]:
            raise PC01ArtifactError(
                "PC-01 run contract and training environment commits differ"
            )
    return environment


def _count_json_key(value: Any, key: str) -> int:
    if isinstance(value, Mapping):
        return int(key in value) + sum(
            _count_json_key(item, key) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_count_json_key(item, key) for item in value)
    return 0


def build_pc01_training_action_value_evidence(
    *,
    primary_train_path: str | Path,
    supplement_train_path: str | Path,
    report_path: str | Path,
    run_contract_path: str | Path,
    expected_primary_sha256: str = PC01_PRIMARY_TRAIN_SHA256,
    expected_supplement_sha256: str = PC01_SUPPLEMENT_TRAIN_SHA256,
    expected_primary_rows: int = PC01_PRIMARY_TRAIN_ROWS,
    expected_supplement_rows: int = PC01_SUPPLEMENT_TRAIN_ROWS,
) -> dict[str, Any]:
    """Audit the train-only data fact that PC-01 never saw action values.

    The runtime prompt must reproduce what the selected checkpoint actually
    consumed, not what the dataset implementation could have consumed if a
    nonexistent label had been present.  This audit reads only the two training
    JSON files and emits compact evidence; it never reads images, validation, or
    test data.
    """

    primary = _require_regular_file(primary_train_path, label="PC-01 primary train JSON")
    supplement = _require_regular_file(
        supplement_train_path,
        label="PC-01 supplement train JSON",
    )
    if sha256_file(primary) != expected_primary_sha256:
        raise PC01ArtifactError("PC-01 primary train JSON differs from registered bytes")
    if sha256_file(supplement) != expected_supplement_sha256:
        raise PC01ArtifactError(
            "PC-01 supplement train JSON differs from registered bytes"
        )
    primary_rows = _read_unique_json_array(primary, label="PC-01 primary train JSON")
    supplement_rows = _read_unique_json_array(
        supplement,
        label="PC-01 supplement train JSON",
    )
    sources = (
        (
            "original_gold",
            primary,
            primary_rows,
            expected_primary_sha256,
            expected_primary_rows,
        ),
        (
            "retry_abort_supplement_v2",
            supplement,
            supplement_rows,
            expected_supplement_sha256,
            expected_supplement_rows,
        ),
    )
    source_evidence: list[dict[str, Any]] = []
    for source_id, source_path, rows, expected_sha256, expected_rows in sources:
        if len(rows) != expected_rows:
            raise PC01ArtifactError(
                f"PC-01 {source_id} train row count changed: {len(rows)} != {expected_rows}"
            )
        counts = {"top_level": 0, "inputs": 0, "labels": 0, "recursive_total": 0}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise PC01ArtifactError(
                    f"PC-01 {source_id} train row {index} is not an object"
                )
            inputs = row.get("inputs")
            labels = row.get("labels")
            if not isinstance(inputs, Mapping) or not isinstance(labels, Mapping):
                raise PC01ArtifactError(
                    f"PC-01 {source_id} train row {index} lacks inputs/labels"
                )
            counts["top_level"] += int("action_value" in row)
            counts["inputs"] += int("action_value" in inputs)
            counts["labels"] += int("action_value" in labels)
            counts["recursive_total"] += _count_json_key(row, "action_value")
        if any(counts.values()):
            raise PC01ArtifactError(
                f"PC-01 {source_id} contains action_value evidence: {counts}"
            )
        source_evidence.append(
            {
                "source_id": source_id,
                "file_name": source_path.name,
                "sha256": expected_sha256,
                "rows": expected_rows,
                "action_value_key_occurrences": counts,
            }
        )

    report_source = _require_regular_file(report_path, label="PC-01 report")
    contract_source = _require_regular_file(
        run_contract_path,
        label="PC-01 run contract",
    )
    report = _read_unique_json(report_source, label="PC-01 report")
    contract = _read_unique_json(contract_source, label="PC-01 run contract")
    total_rows = expected_primary_rows + expected_supplement_rows
    if report.get("train_rows") != total_rows or contract.get("train_rows") != total_rows:
        raise PC01ArtifactError("PC-01 train-row evidence differs from the two train inputs")
    experiment_control = report.get("experiment_control")
    split_hashes = (
        experiment_control.get("development_split_sha256")
        if isinstance(experiment_control, Mapping)
        else None
    )
    if not isinstance(split_hashes, Mapping) or split_hashes.get("train") != expected_primary_sha256:
        raise PC01ArtifactError(
            "PC-01 report does not authenticate the primary training split"
        )
    train_distribution = report.get("train_distribution")
    source_distribution = (
        train_distribution.get("source_dataset")
        if isinstance(train_distribution, Mapping)
        else None
    )
    expected_distribution = {
        "original_gold": expected_primary_rows,
        "retry_abort_supplement_v2": expected_supplement_rows,
    }
    if source_distribution != expected_distribution:
        raise PC01ArtifactError("PC-01 report training-source distribution changed")
    if contract.get("git_commit") != PC01_TRAINING_GIT_COMMIT:
        raise PC01ArtifactError("PC-01 run contract training commit changed")
    if report.get("test_rows_read") != 0 or contract.get("test_rows_read") != 0:
        raise PC01ArtifactError("PC-01 action-value audit cannot bind a test-reading run")

    return {
        "schema_version": PC01_TRAINING_ACTION_VALUE_SCHEMA,
        "training_git_commit": PC01_TRAINING_GIT_COMMIT,
        "action_value_mode": "OMITTED_FOR_ALL_TRAINING_ROWS",
        "runtime_requirement": "OMIT_ACTION_VALUE_TEXT",
        "executed_action_type_supplied_to_processor": True,
        "action_value_supplied_to_processor": False,
        "sources": source_evidence,
        "combined_train_rows": total_rows,
        "report_sha256": sha256_file(report_source),
        "run_contract_sha256": sha256_file(contract_source),
        "training_rows_read": total_rows,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
    }


def load_pc01_training_action_value_evidence(
    path: str | Path,
    *,
    report_path: str | Path,
    run_contract_path: str | Path,
) -> dict[str, Any]:
    """Authenticate a compact train-only action-value audit receipt."""

    source = _require_regular_file(path, label="PC-01 training action-value evidence")
    value = _read_unique_json(source, label="PC-01 training action-value evidence")
    if source.read_bytes() != canonical_json_bytes(value):
        raise PC01ArtifactError("PC-01 training action-value evidence is not canonical")
    return validate_pc01_training_action_value_evidence(
        value,
        report_sha256=sha256_file(
            _require_regular_file(report_path, label="PC-01 report")
        ),
        run_contract_sha256=sha256_file(
            _require_regular_file(run_contract_path, label="PC-01 run contract")
        ),
    )


def validate_pc01_training_action_value_evidence(
    value: Mapping[str, Any],
    *,
    report_sha256: str,
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Validate exact production action-value evidence without source paths.

    Handoff already authenticates the report/run-contract hashes through the
    export manifest, so this pure validator lets that boundary reuse the same
    registered source identities and row counts without copying either large
    training corpus into the campaign package.
    """

    if not _is_sha256(report_sha256) or not _is_sha256(run_contract_sha256):
        raise PC01ArtifactError(
            "PC-01 action-value report/run-contract identities must be SHA-256"
        )
    expected_fields = {
        "schema_version",
        "training_git_commit",
        "action_value_mode",
        "runtime_requirement",
        "executed_action_type_supplied_to_processor",
        "action_value_supplied_to_processor",
        "sources",
        "combined_train_rows",
        "report_sha256",
        "run_contract_sha256",
        "training_rows_read",
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
    }
    if set(value) != expected_fields:
        raise PC01ArtifactError("PC-01 training action-value evidence fields changed")
    exact = {
        "schema_version": PC01_TRAINING_ACTION_VALUE_SCHEMA,
        "training_git_commit": PC01_TRAINING_GIT_COMMIT,
        "action_value_mode": "OMITTED_FOR_ALL_TRAINING_ROWS",
        "runtime_requirement": "OMIT_ACTION_VALUE_TEXT",
        "executed_action_type_supplied_to_processor": True,
        "action_value_supplied_to_processor": False,
        "combined_train_rows": PC01_TOTAL_TRAIN_ROWS,
        "report_sha256": report_sha256,
        "run_contract_sha256": run_contract_sha256,
        "training_rows_read": PC01_TOTAL_TRAIN_ROWS,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise PC01ArtifactError(
                f"PC-01 training action-value evidence changed {field}"
            )
    expected_sources = [
        ("original_gold", "split_train.json", PC01_PRIMARY_TRAIN_SHA256, PC01_PRIMARY_TRAIN_ROWS),
        (
            "retry_abort_supplement_v2",
            "supplement_train.json",
            PC01_SUPPLEMENT_TRAIN_SHA256,
            PC01_SUPPLEMENT_TRAIN_ROWS,
        ),
    ]
    rows = value.get("sources")
    if not isinstance(rows, list) or len(rows) != len(expected_sources):
        raise PC01ArtifactError("PC-01 training action-value sources changed")
    zero_counts = {"top_level": 0, "inputs": 0, "labels": 0, "recursive_total": 0}
    for row, expected in zip(rows, expected_sources):
        if not isinstance(row, Mapping) or set(row) != {
            "source_id",
            "file_name",
            "sha256",
            "rows",
            "action_value_key_occurrences",
        }:
            raise PC01ArtifactError("PC-01 training action-value source row is malformed")
        source_id, file_name, digest, count = expected
        if dict(row) != {
            "source_id": source_id,
            "file_name": file_name,
            "sha256": digest,
            "rows": count,
            "action_value_key_occurrences": zero_counts,
        }:
            raise PC01ArtifactError("PC-01 training action-value source evidence changed")
    return value


def write_pc01_training_action_value_evidence(
    path: str | Path,
    evidence: Mapping[str, Any],
) -> Path:
    """Write one immutable canonical training-data audit receipt."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(evidence)))
    except FileExistsError as exc:
        raise PC01ArtifactError(
            f"refusing to overwrite training action-value evidence: {destination}"
        ) from exc
    destination.chmod(0o444)
    return destination


def load_pc01_resolved_config(
    path: str | Path,
    *,
    expected_sha256: str = PC01_EXPECTED_CONFIG_SHA256,
) -> dict[str, Any]:
    """Load exact canonical PC-01 checkpoint-saved configuration bytes."""

    if not _is_sha256(expected_sha256):
        raise PC01ArtifactError("expected resolved-config identity must be SHA-256")
    source = _require_regular_file(path, label="PC-01 resolved config")
    config = _read_unique_json(source, label="PC-01 resolved config")
    canonical = canonical_json_bytes(config)
    if source.read_bytes() != canonical:
        raise PC01ArtifactError("PC-01 resolved config is not exact canonical bytes")
    if sha256_bytes(canonical) != expected_sha256:
        raise PC01ArtifactError("PC-01 resolved config differs from registered evidence")
    _validate_saved_pc01_config(config)
    return config


def load_processor_contract(path: str | Path) -> ProcessorParityContract:
    """Load a canonical contract file and require byte/record identity."""

    source = _require_regular_file(path, label="processor contract")
    value = _read_unique_json(source, label="processor contract")
    fields = (
        "processor_class",
        "processor_revision",
        "processor_config_sha256",
        "pre_action_field_mapping",
        "post_action_field_mapping",
    )
    try:
        contract = ProcessorParityContract(**{field: value[field] for field in fields})
    except (KeyError, TypeError, ValueError) as exc:
        raise PC01ArtifactError("processor contract cannot be reconstructed") from exc
    if set(value) != set(contract.to_dict()):
        raise PC01ArtifactError("processor contract contains missing or unregistered fields")
    if source.read_bytes() != canonical_json_bytes(contract.to_dict()):
        raise PC01ArtifactError("processor contract is not exact canonical record bytes")
    return contract


@contextmanager
def _atomic_output_directory(destination: Path):
    parent = destination.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PC01ArtifactError(f"refusing to overwrite export directory: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        yield temporary
        os.replace(temporary, destination)
    except BaseException:
        # Only the exporter-owned fresh temporary directory is removed.
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))
    path.chmod(0o444)


def export_pc01_checkpoint_identity(
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    run_contract_path: str | Path,
    output_dir: str | Path,
    expected_config_sha256: str = PC01_EXPECTED_CONFIG_SHA256,
    expected_checkpoint_sha256: str = PC01_EXPECTED_CHECKPOINT_SHA256,
    checkpoint_loader: CheckpointLoader | None = None,
) -> PC01CheckpointIdentityResult:
    """Export only the authenticated checkpoint config and its identity receipt.

    This deliberately does not load or require a processor/base-model snapshot,
    so checkpoint authentication can be completed as soon as the ``.ckpt``,
    report, and run contract are present.  It is not a complete runtime bundle.
    """

    config, config_bytes, checkpoint_sha256, report_source, contract_source = (
        _verified_checkpoint_inputs(
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            run_contract_path=run_contract_path,
            expected_config_sha256=expected_config_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            checkpoint_loader=checkpoint_loader,
        )
    )
    config_sha256 = sha256_bytes(config_bytes)
    destination = Path(output_dir).resolve()
    with _atomic_output_directory(destination) as temporary:
        resolved_config = temporary / "resolved_config.json"
        receipt = temporary / "pc01_checkpoint_identity.json"
        resolved_config.write_bytes(config_bytes)
        resolved_config.chmod(0o444)
        _write_canonical(
            receipt,
            {
                "schema_version": "table2.pc01-checkpoint-identity.v1",
                "model_id": PC01_MODEL_ID,
                "run_contract_config_name": PC01_CONFIG_NAME,
                "checkpoint_saved_config_name": config["name"],
                "model_seed": PC01_MODEL_SEED,
                "model_revision": PC01_MODEL_REVISION,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_size_bytes": Path(checkpoint_path).stat().st_size,
                "resolved_config_record_sha256": config_sha256,
                "resolved_config_payload_sha256": sha256_file(resolved_config),
                "report_sha256": sha256_file(report_source),
                "run_contract_sha256": sha256_file(contract_source),
                "selection_scope": "validation_only",
                "test_rows_read": 0,
                "locked_test_rows_read": 0,
                "processor_validated": False,
                "runtime_ready": False,
                "network_access_used": False,
                "weight_updates_performed": False,
            },
        )
    return PC01CheckpointIdentityResult(
        output_dir=destination,
        resolved_config_path=destination / "resolved_config.json",
        identity_receipt_path=destination / "pc01_checkpoint_identity.json",
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )


def export_pc01_artifacts(
    *,
    checkpoint_path: str | Path,
    report_path: str | Path,
    run_contract_path: str | Path,
    processor_source: str | Path,
    output_dir: str | Path,
    training_environment_path: str | Path | None = None,
    training_action_value_evidence_path: str | Path | None = None,
    expected_config_sha256: str = PC01_EXPECTED_CONFIG_SHA256,
    expected_checkpoint_sha256: str = PC01_EXPECTED_CHECKPOINT_SHA256,
    expected_training_environment_sha256: str = PC01_TRAINING_ENVIRONMENT_SHA256,
    expected_base_snapshot_sha256: str = PC01_EXPECTED_BASE_SNAPSHOT_SHA256,
    checkpoint_loader: CheckpointLoader | None = None,
    processor_loader: ProcessorLoader | None = None,
    version_loader: VersionLoader | None = None,
) -> PC01ExportResult:
    """Export canonical PC-01 handoff evidence without changing model weights."""

    if training_environment_path is None:
        raise PC01ArtifactError(
            "full PC-01 export requires the checked-in training environment"
        )
    if training_action_value_evidence_path is None:
        raise PC01ArtifactError(
            "full PC-01 export requires train-only action-value evidence"
        )
    training_environment_source = _require_regular_file(
        training_environment_path,
        label="PC-01 training environment",
    )
    training_environment = load_pc01_training_environment(
        training_environment_source,
        run_contract_path=run_contract_path,
        expected_sha256=expected_training_environment_sha256,
    )
    training_environment_sha256 = sha256_file(training_environment_source)
    training_environment_record_sha256 = sha256_bytes(
        canonical_json_bytes(training_environment)
    )
    training_source_manifest_value = validate_pc01_training_sources()
    training_source_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(training_source_manifest_value)
    )
    training_action_value_evidence_source = _require_regular_file(
        training_action_value_evidence_path,
        label="PC-01 training action-value evidence",
    )
    training_action_value_evidence_value = load_pc01_training_action_value_evidence(
        training_action_value_evidence_source,
        report_path=report_path,
        run_contract_path=run_contract_path,
    )
    training_action_value_evidence_sha256 = sha256_file(
        training_action_value_evidence_source
    )

    config, config_bytes, checkpoint_sha256, report_source, contract_source = (
        _verified_checkpoint_inputs(
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            run_contract_path=run_contract_path,
            expected_config_sha256=expected_config_sha256,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            checkpoint_loader=checkpoint_loader,
        )
    )
    config_sha256 = sha256_bytes(config_bytes)

    base_snapshot_manifest_value = build_pc01_base_snapshot_manifest(processor_source)
    if not _is_sha256(expected_base_snapshot_sha256):
        raise PC01ArtifactError("expected base snapshot identity must be SHA-256")
    if (
        base_snapshot_manifest_value["directory_payload_sha256"]
        != expected_base_snapshot_sha256
    ):
        raise PC01ArtifactError(
            "base-model snapshot bytes differ from the registered Qwen snapshot"
        )
    loaded_processor = load_pinned_local_processor(
        config,
        processor_source,
        processor_loader=processor_loader,
        version_loader=version_loader,
        training_environment_sha256=training_environment_sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_action_value_evidence_sha256=(
            training_action_value_evidence_sha256
        ),
    )
    processor_payload = loaded_processor.contract.to_dict()
    processor_contract_sha256 = loaded_processor.contract.record_sha256
    processor_artifact_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(loaded_processor.artifact_manifest)
    )
    from web_agent.eval.table2.pc01_processor_parity import (
        build_pc01_processor_parity_receipt,
        validate_pc01_processor_parity_receipt,
    )

    try:
        processor_parity_receipt_value = dict(
            build_pc01_processor_parity_receipt(
                config=config,
                processor=loaded_processor.processor,
                processor_source=processor_source,
                processor_contract=loaded_processor.contract,
                implementation_identity=loaded_processor.implementation_identity,
                processor_artifact_manifest_sha256=(
                    processor_artifact_manifest_sha256
                ),
                base_snapshot_directory_payload_sha256=(
                    base_snapshot_manifest_value["directory_payload_sha256"]
                ),
                training_environment_sha256=training_environment_sha256,
                training_environment_record_sha256=(
                    training_environment_record_sha256
                ),
                training_source_manifest_sha256=training_source_manifest_sha256,
                training_action_value_evidence_sha256=(
                    training_action_value_evidence_sha256
                ),
                run_contract_sha256=sha256_file(contract_source),
            )
        )
    except PC01ArtifactError:
        raise
    except Exception as exc:
        raise PC01ArtifactError(
            "processor-only training/runtime parity validation failed"
        ) from exc
    try:
        processor_parity_receipt_value = validate_pc01_processor_parity_receipt(
            processor_parity_receipt_value,
            resolved_config_record_sha256=config_sha256,
            processor_contract_sha256=processor_contract_sha256,
            processor_artifact_manifest_sha256=processor_artifact_manifest_sha256,
            base_snapshot_directory_payload_sha256=(
                base_snapshot_manifest_value["directory_payload_sha256"]
            ),
            training_environment_sha256=training_environment_sha256,
            training_environment_record_sha256=training_environment_record_sha256,
            training_source_manifest_sha256=training_source_manifest_sha256,
            training_action_value_evidence_sha256=(
                training_action_value_evidence_sha256
            ),
            run_contract_sha256=sha256_file(contract_source),
        )
    except Exception as exc:
        raise PC01ArtifactError("processor parity receipt validation failed") from exc
    e0_config = {
        "schema_version": "table2.pc01-e0-resolved-config.v1",
        "system_id": "E0",
        "adaptation_loaded": False,
        "task_heads_loaded": False,
        "backbone": deepcopy(config["backbone"]),
    }

    destination = Path(output_dir).resolve()
    with _atomic_output_directory(destination) as temporary:
        resolved_config = temporary / "resolved_config.json"
        processor_contract = temporary / "processor_contract.json"
        e0_processor_contract = temporary / "e0_processor_contract.json"
        e0_resolved_config = temporary / "e0_resolved_config.json"
        processor_manifest = temporary / "processor_artifact_manifest.json"
        base_snapshot_manifest = temporary / "base_snapshot_manifest.json"
        processor_parity_receipt = temporary / "processor_parity_receipt.json"
        training_environment_artifact = temporary / "training_environment.json"
        training_source_manifest = temporary / "training_source_manifest.json"
        training_action_value_evidence = (
            temporary / "training_action_value_evidence.json"
        )
        manifest = temporary / "pc01_export_manifest.json"
        resolved_config.write_bytes(config_bytes)
        resolved_config.chmod(0o444)
        _write_canonical(processor_contract, processor_payload)
        _write_canonical(e0_processor_contract, processor_payload)
        _write_canonical(e0_resolved_config, e0_config)
        _write_canonical(processor_manifest, loaded_processor.artifact_manifest)
        _write_canonical(base_snapshot_manifest, base_snapshot_manifest_value)
        _write_canonical(processor_parity_receipt, processor_parity_receipt_value)
        _write_canonical(training_environment_artifact, training_environment)
        _write_canonical(training_source_manifest, training_source_manifest_value)
        _write_canonical(
            training_action_value_evidence,
            training_action_value_evidence_value,
        )
        manifest_value = {
            "schema_version": "table2.pc01-export.v3",
            "model_id": PC01_MODEL_ID,
            "config_name": PC01_CONFIG_NAME,
            "model_seed": PC01_MODEL_SEED,
            "model_revision": PC01_MODEL_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": Path(checkpoint_path).stat().st_size,
            "resolved_config_record_sha256": config_sha256,
            "resolved_config_payload_sha256": sha256_file(resolved_config),
            "processor_contract_sha256": processor_contract_sha256,
            "processor_artifact_manifest_sha256": sha256_file(processor_manifest),
            "processor_parity_receipt_sha256": sha256_file(
                processor_parity_receipt
            ),
            "processor_parity_verified": True,
            "base_snapshot_manifest_sha256": sha256_file(base_snapshot_manifest),
            "base_snapshot_directory_payload_sha256": (
                base_snapshot_manifest_value["directory_payload_sha256"]
            ),
            "base_snapshot_file_count": base_snapshot_manifest_value["file_count"],
            "base_snapshot_copied_into_export": False,
            "training_environment_source_sha256": training_environment_sha256,
            "training_environment_record_sha256": sha256_file(
                training_environment_artifact
            ),
            "training_source_manifest_sha256": sha256_file(
                training_source_manifest
            ),
            "training_action_value_evidence_sha256": sha256_file(
                training_action_value_evidence
            ),
            "training_transformers_version": training_environment["transformers"],
            "runtime_processor_implementation": dict(
                loaded_processor.implementation_identity
            ),
            "report_sha256": sha256_file(report_source),
            "run_contract_sha256": sha256_file(contract_source),
            "selection_scope": "validation_only",
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "network_access_used": False,
            "weight_updates_performed": False,
            "runtime_ready": False,
            "files": {
                name: {
                    "sha256": sha256_file(temporary / name),
                    "size_bytes": (temporary / name).stat().st_size,
                }
                for name in (
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
                )
            },
        }
        _write_canonical(manifest, manifest_value)

    return PC01ExportResult(
        output_dir=destination,
        resolved_config_path=destination / "resolved_config.json",
        processor_contract_path=destination / "processor_contract.json",
        e0_processor_contract_path=destination / "e0_processor_contract.json",
        e0_resolved_config_path=destination / "e0_resolved_config.json",
        processor_parity_receipt_path=destination / "processor_parity_receipt.json",
        base_snapshot_manifest_path=destination / "base_snapshot_manifest.json",
        training_environment_path=destination / "training_environment.json",
        training_source_manifest_path=destination / "training_source_manifest.json",
        training_action_value_evidence_path=(
            destination / "training_action_value_evidence.json"
        ),
        manifest_path=destination / "pc01_export_manifest.json",
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        processor_contract_sha256=processor_contract_sha256,
    )
