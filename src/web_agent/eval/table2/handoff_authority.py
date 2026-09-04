"""Authenticate the complete Table 2 handoff before campaign freeze.

The evaluation freeze accepts many individual paths for testability.  A real
campaign must not be assembled from independently self-consistent files,
however: it must consume the exact, source-attested bundle produced by the
handoff builder.  This module closes that boundary by replaying the handoff
inventory and cross-checking every freeze input against the registered bundle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import stat
from typing import Any

from .common import SchemaError, read_json, require_keys, sha256_file


HANDOFF_BUNDLE_SCHEMA_VERSION = "table2-handoff-bundle-v1"
HANDOFF_MANIFEST_NAME = "handoff_manifest.json"


def _resolved_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise SchemaError(f"{field} is missing, not a file, or symlinked")
    metadata = candidate.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SchemaError(f"{field} must be a singly linked regular file")
    return candidate.resolve()


def _registered_path(value: object, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise SchemaError(f"handoff freeze arguments lack {field}")
    path = Path(text)
    if not path.is_absolute():
        raise SchemaError(f"handoff freeze argument {field} must be an absolute path")
    return path.resolve()


def _require_inventoried_file(
    path: Path,
    *,
    package_root: Path,
    inventory: Mapping[str, str],
    field: str,
) -> None:
    try:
        relative = path.relative_to(package_root).as_posix()
    except ValueError as exc:
        raise SchemaError(f"handoff {field} escaped the authenticated package") from exc
    if inventory.get(relative) != sha256_file(path):
        raise SchemaError(f"handoff {field} is not byte-inventoried")


def _require_same_path(
    actual: str | Path | None,
    registered: object,
    *,
    field: str,
    package_root: Path,
    inventory: Mapping[str, str],
) -> Path:
    if actual is None:
        raise SchemaError(f"evaluation freeze requires handoff-bound {field}")
    resolved = _resolved_file(actual, field=field)
    if resolved != _registered_path(registered, field=field):
        raise SchemaError(f"evaluation freeze {field} differs from handoff authority")
    _require_inventoried_file(
        resolved,
        package_root=package_root,
        inventory=inventory,
        field=field,
    )
    return resolved


def _require_same_path_list(
    actual: Sequence[str | Path],
    registered: object,
    *,
    field: str,
    package_root: Path,
    inventory: Mapping[str, str],
) -> tuple[Path, ...]:
    if not isinstance(registered, list) or not all(
        isinstance(value, str) and value.strip() for value in registered
    ):
        raise SchemaError(f"handoff freeze arguments {field} must be a path array")
    actual_paths = tuple(_resolved_file(value, field=field) for value in actual)
    registered_paths = tuple(
        _registered_path(value, field=f"{field}[{index}]")
        for index, value in enumerate(registered)
    )
    if actual_paths != registered_paths:
        raise SchemaError(f"evaluation freeze {field} differ from handoff authority")
    for index, path in enumerate(actual_paths):
        _require_inventoried_file(
            path,
            package_root=package_root,
            inventory=inventory,
            field=f"{field}[{index}]",
        )
    return actual_paths


def validate_handoff_freeze_authority(
    handoff_manifest_path: str | Path,
    *,
    repository_root: str | Path,
    repository_commit: str,
    campaign_config_path: str | Path,
    selection_evidence_path: str | Path | None,
    model_manifest_paths: Sequence[str | Path],
    memory_manifest_paths: Sequence[str | Path],
    environment_manifest_path: str | Path | None,
    runner_attestation_path: str | Path | None,
    resolved_task_snapshot_path: str | Path | None,
    pc01_checkpoint_compatibility_receipt_path: str | Path | None,
) -> dict[str, Any]:
    """Replay one handoff inventory and bind every evaluation-freeze input.

    Exact byte inventory is checked before any individual path is trusted.
    Extra files, missing files, symlinks, a different source checkout/commit,
    or replacement freeze arguments all fail closed.
    """

    manifest_path = _resolved_file(
        handoff_manifest_path, field="handoff authority manifest"
    )
    if manifest_path.name != HANDOFF_MANIFEST_NAME:
        raise SchemaError(
            f"handoff authority must be named {HANDOFF_MANIFEST_NAME}"
        )
    package_root = manifest_path.parent
    value = read_json(manifest_path)
    if not isinstance(value, Mapping):
        raise SchemaError("handoff authority manifest must be a JSON object")
    require_keys(
        value,
        (
            "schema_version",
            "repository_root",
            "repository_commit",
            "campaign_mode",
            "selection_mode",
            "matched_seeds",
            "freeze_arguments",
            "files",
        ),
        context="handoff authority manifest",
    )
    if value.get("schema_version") != HANDOFF_BUNDLE_SCHEMA_VERSION:
        raise SchemaError("handoff authority schema is not registered")
    if value.get("campaign_mode") != "evaluation":
        raise SchemaError("handoff authority is not an evaluation bundle")
    if value.get("matched_seeds") != [42]:
        raise SchemaError("handoff authority matched seeds must be exactly [42]")

    repo = Path(repository_root).resolve()
    if Path(str(value.get("repository_root") or "")).resolve() != repo:
        raise SchemaError("handoff authority repository root differs")
    if str(value.get("repository_commit") or "") != repository_commit:
        raise SchemaError("handoff authority repository commit differs")

    inventory = value.get("files")
    if not isinstance(inventory, Mapping) or not inventory:
        raise SchemaError("handoff authority file inventory must be non-empty")
    registered_inventory: dict[str, str] = {}
    for raw_relative, raw_digest in inventory.items():
        relative = str(raw_relative)
        digest = str(raw_digest)
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise SchemaError("handoff authority contains an invalid inventory entry")
        if relative == HANDOFF_MANIFEST_NAME or relative in registered_inventory:
            raise SchemaError("handoff authority contains a forbidden inventory entry")
        registered_inventory[relative] = digest

    expected_directories: set[str] = set()
    for relative in registered_inventory:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_inventory: dict[str, str] = {}
    actual_directories: set[str] = set()
    for path in sorted(package_root.rglob("*")):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise SchemaError("handoff authority package contains a symlink")
        if path.is_dir():
            actual_directories.add(path.relative_to(package_root).as_posix())
        elif path.is_file():
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SchemaError(
                    "handoff authority package contains a non-regular/hard-linked file"
                )
            relative = path.relative_to(package_root).as_posix()
            actual_inventory[relative] = sha256_file(path)
        else:
            raise SchemaError(
                "handoff authority package contains a non-regular filesystem entry"
            )
    if actual_inventory != registered_inventory:
        raise SchemaError(
            "handoff authority file inventory differs from exact byte replay"
        )
    if actual_directories != expected_directories:
        raise SchemaError(
            "handoff authority directory closure differs from its file inventory"
        )

    arguments = value.get("freeze_arguments")
    if not isinstance(arguments, Mapping):
        raise SchemaError("handoff authority freeze_arguments must be a mapping")
    require_keys(
        arguments,
        (
            "handoff_manifest",
            "campaign_config",
            "resolved_task_snapshot",
            "environment_manifest",
            "runner_attestation",
            "checkpoint_selection_evidence",
            "model_manifests",
            "memory_manifests",
        ),
        context="handoff authority freeze_arguments",
    )
    registered_manifest = _registered_path(
        arguments["handoff_manifest"], field="handoff_manifest"
    )
    if registered_manifest != manifest_path:
        raise SchemaError(
            "evaluation freeze handoff_manifest differs from handoff authority"
        )
    _require_same_path(
        campaign_config_path,
        arguments["campaign_config"],
        field="campaign_config",
        package_root=package_root,
        inventory=registered_inventory,
    )
    _require_same_path(
        resolved_task_snapshot_path,
        arguments["resolved_task_snapshot"],
        field="resolved_task_snapshot",
        package_root=package_root,
        inventory=registered_inventory,
    )
    _require_same_path(
        environment_manifest_path,
        arguments["environment_manifest"],
        field="environment_manifest",
        package_root=package_root,
        inventory=registered_inventory,
    )
    _require_same_path(
        runner_attestation_path,
        arguments["runner_attestation"],
        field="runner_attestation",
        package_root=package_root,
        inventory=registered_inventory,
    )
    _require_same_path(
        selection_evidence_path,
        arguments["checkpoint_selection_evidence"],
        field="checkpoint_selection_evidence",
        package_root=package_root,
        inventory=registered_inventory,
    )
    _require_same_path_list(
        model_manifest_paths,
        arguments["model_manifests"],
        field="model_manifests",
        package_root=package_root,
        inventory=registered_inventory,
    )
    _require_same_path_list(
        memory_manifest_paths,
        arguments["memory_manifests"],
        field="memory_manifests",
        package_root=package_root,
        inventory=registered_inventory,
    )

    registered_receipt = arguments.get("pc01_checkpoint_compatibility_receipt")
    if registered_receipt is None:
        if pc01_checkpoint_compatibility_receipt_path is not None:
            raise SchemaError(
                "evaluation freeze supplied a checkpoint receipt absent from handoff"
            )
    else:
        _require_same_path(
            pc01_checkpoint_compatibility_receipt_path,
            registered_receipt,
            field="pc01_checkpoint_compatibility_receipt",
            package_root=package_root,
            inventory=registered_inventory,
        )

    return dict(value)
