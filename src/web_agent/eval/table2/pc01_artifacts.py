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
PC01_EXPECTED_CONFIG_SHA256 = (
    "d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f"
)
PC01_EXPECTED_CHECKPOINT_SHA256 = (
    "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a"
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
    "transition.executed_action.parameters": "text.action_value",
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


@dataclass(frozen=True, slots=True)
class LoadedPinnedProcessor:
    """A locally loaded processor and the byte-derived parity evidence."""

    processor: Any
    contract: ProcessorParityContract
    artifact_manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PC01ExportResult:
    output_dir: Path
    resolved_config_path: Path
    processor_contract_path: Path
    e0_processor_contract_path: Path
    e0_resolved_config_path: Path
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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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
    checkpoint_loader: CheckpointLoader | None = None,
) -> tuple[dict[str, Any], str, Mapping[str, Any]]:
    """Return the exact checkpoint-saved config after structural validation."""

    source = _require_regular_file(checkpoint_path, label="selected checkpoint")
    _assert_checkpoint_materialized(source)
    digest = sha256_file(source)
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


def _processor_file_manifest(source: Path) -> dict[str, Any]:
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
        "schema_version": "table2.pc01-processor-artifacts.v1",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "files": files,
    }


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


def load_pinned_local_processor(
    config: Mapping[str, Any],
    processor_source: str | Path,
    *,
    processor_loader: ProcessorLoader | None = None,
) -> LoadedPinnedProcessor:
    """Load and fingerprint the exact local processor registered by ``config``."""

    source = _require_local_directory(processor_source, label="processor snapshot")
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
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
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
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise PC01ArtifactError("Qwen2-VL processor lacks an image processor")
    for field, expected in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        actual = getattr(image_processor, field, None)
        if actual is None or int(actual) != expected:
            raise PC01ArtifactError(f"loaded processor changed {field}")

    artifact_manifest = _processor_file_manifest(source)
    processor_config_sha256 = sha256_bytes(canonical_json_bytes(artifact_manifest))
    processor_class = f"{type(processor).__module__}.{type(processor).__qualname__}"
    contract = ProcessorParityContract(
        processor_class=processor_class,
        processor_revision=PC01_MODEL_REVISION,
        processor_config_sha256=processor_config_sha256,
        pre_action_field_mapping=PRE_ACTION_FIELD_MAPPING,
        post_action_field_mapping=POST_ACTION_FIELD_MAPPING,
    )
    return LoadedPinnedProcessor(
        processor=processor,
        contract=contract,
        artifact_manifest=artifact_manifest,
    )


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
    expected_config_sha256: str = PC01_EXPECTED_CONFIG_SHA256,
    expected_checkpoint_sha256: str = PC01_EXPECTED_CHECKPOINT_SHA256,
    checkpoint_loader: CheckpointLoader | None = None,
    processor_loader: ProcessorLoader | None = None,
) -> PC01ExportResult:
    """Export canonical PC-01 handoff evidence without changing model weights."""

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

    loaded_processor = load_pinned_local_processor(
        config,
        processor_source,
        processor_loader=processor_loader,
    )
    processor_payload = loaded_processor.contract.to_dict()
    processor_contract_sha256 = loaded_processor.contract.record_sha256
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
        manifest = temporary / "pc01_export_manifest.json"
        resolved_config.write_bytes(config_bytes)
        resolved_config.chmod(0o444)
        _write_canonical(processor_contract, processor_payload)
        _write_canonical(e0_processor_contract, processor_payload)
        _write_canonical(e0_resolved_config, e0_config)
        _write_canonical(processor_manifest, loaded_processor.artifact_manifest)
        manifest_value = {
            "schema_version": "table2.pc01-export.v1",
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
            "report_sha256": sha256_file(report_source),
            "run_contract_sha256": sha256_file(contract_source),
            "selection_scope": "validation_only",
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "network_access_used": False,
            "weight_updates_performed": False,
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
        manifest_path=destination / "pc01_export_manifest.json",
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        processor_contract_sha256=processor_contract_sha256,
    )
