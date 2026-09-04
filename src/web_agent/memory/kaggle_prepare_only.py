"""Fail-closed Kaggle wrapper for train-only P4 candidate preparation.

This module deliberately exposes only the authenticated ``audit-candidates``
and ``validate-preparation`` stages.  It cannot author provenance, run a joint
duplicate audit, load a checkpoint, build a memory store, or read a validation
or test JSON file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from web_agent.memory.manifest import canonical_sha256, sha256_file
from web_agent.memory.preparation import (
    P4PreparationPackage,
    prepare_p4_candidate_audit,
    validate_p4_preparation_package,
)


CONFIG_SCHEMA_VERSION = "table2-p4-kaggle-prepare-config-v1"
RECEIPT_SCHEMA_VERSION = "table2-p4-kaggle-prepare-receipt-v2"
MODE = "P4_PREPARE_ONLY"
PAPER_TABLE_STATUS = "N/R"
SOURCE_AUTHORITY_RELATIVE = "configs/eval/table2/p4_source_authority_v1.json"
PREPARE_ONLY_CONFIG_RELATIVE = (
    "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
)
EXPECTED_DATASET_ID = "web-gold-v2.8"
EXPECTED_DATASET_VERSION = "pc01-train-0522807d-supplement-67ade5e9"
MAX_PREPARATION_OUTPUT_BYTES = 536_870_912
APPLICATION_ZERO_READ_SCOPE = (
    "APPLICATION_LEVEL_EXPLICIT_INPUTS_ONLY_NOT_OS_OR_PLATFORM_AUDIT"
)

# Every repository file whose Python bytes can participate in the prepare-only
# process is committed explicitly.  Third-party/runtime identities remain in
# the receipt's dependency/platform section; this set closes the local source
# boundary that a bare ``git rev-parse HEAD`` cannot establish on a dirty tree.
EXECUTED_SOURCE_RELATIVE_PATHS = (
    "kaggle/table2_p4_prepare_only/run.py",
    "scripts/run_table2_p4_kaggle_prepare_only.py",
    "src/web_agent/__init__.py",
    "src/web_agent/data/__init__.py",
    "src/web_agent/data/recovery_transitions.py",
    "src/web_agent/memory/__init__.py",
    "src/web_agent/memory/builder.py",
    "src/web_agent/memory/calibration_builder.py",
    "src/web_agent/memory/eligibility.py",
    "src/web_agent/memory/frozen_store.py",
    "src/web_agent/memory/index.py",
    "src/web_agent/memory/kaggle_prepare_only.py",
    "src/web_agent/memory/manifest.py",
    "src/web_agent/memory/preparation.py",
    "src/web_agent/memory/verification.py",
    "src/web_agent/runtime/__init__.py",
    "src/web_agent/runtime/contracts.py",
    "src/web_agent/runtime/duplicate_audit.py",
    "src/web_agent/runtime/model_calls.py",
    "src/web_agent/runtime/protocol.py",
)
SOURCE_COMMIT_VERIFICATION = (
    "MATCHED_CLEAN_REPOSITORY_GIT_HEAD_AND_EXECUTED_SOURCE_SET"
)
SOURCE_ARCHIVE_EVIDENCE_ROLE = (
    "TRANSPORT_ONLY_NOT_EXECUTED_NOT_SOURCE_AUTHORITY"
)
PREPARATION_EXECUTION_RECEIPT_VALIDATED = (
    "VALIDATED_REGISTERED_KAGGLE_PREPARE_ONLY_RECEIPT"
)
PREPARATION_EXECUTION_RECEIPT_NOT_APPLICABLE = (
    "NOT_APPLICABLE_NONREGISTERED_SOURCE_AUTHORITY"
)

PREPARATION_OUTPUT_FILES = (
    "candidate_audit.json",
    "preparation_manifest.json",
    "preparation_manifest.sha256",
    "read_ledger.json",
    "review_queue.jsonl",
    "source_authority.json",
)
ALLOWED_OPERATIONS = ("audit-candidates", "validate-preparation")


class KaggleP4PrepareOnlyError(ValueError):
    """The Kaggle preparation-only boundary or an input contract failed."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    role: str
    slug: str
    kaggle_dataset_version: int
    mount_layouts: tuple[str, ...]
    data_root_layouts: tuple[str, ...]
    train_json_relative_path: str


@dataclass(frozen=True, slots=True)
class ResolvedDatasetMount:
    spec: DatasetSpec
    mount_layout: str
    mount_root: Path
    data_root_layout: str
    data_root: Path
    train_json: Path

    def receipt(self) -> dict[str, Any]:
        return {
            "role": self.spec.role,
            "slug": self.spec.slug,
            "declared_kaggle_dataset_version": self.spec.kaggle_dataset_version,
            "kaggle_platform_version_verification": (
                "DECLARED_ONLY_NOT_QUERIED_BY_WRAPPER"
            ),
            "mount_layout": self.mount_layout,
            "resolved_mount_root": str(self.mount_root),
            "data_root_layout": self.data_root_layout,
            "resolved_data_root": str(self.data_root),
            "train_json": str(self.train_json),
            "train_json_sha256": sha256_file(self.train_json),
            "train_json_bytes": self.train_json.stat().st_size,
        }


EXPECTED_DATASETS = (
    DatasetSpec(
        role="original_gold",
        slug="kiyasmahmud/web-gold-40k",
        kaggle_dataset_version=1,
        mount_layouts=(
            "web-gold-40k",
            "datasets/kiyasmahmud/web-gold-40k",
        ),
        data_root_layouts=(".", "final_data_set_40k"),
        train_json_relative_path="split_train.json",
    ),
    DatasetSpec(
        role="retry_abort_supplement_v2",
        slug="kiyasmahmud/gold-40k-retry",
        kaggle_dataset_version=1,
        mount_layouts=(
            "gold-40k-retry",
            "datasets/kiyasmahmud/gold-40k-retry",
        ),
        data_root_layouts=(
            ".",
            "web_gold_40k_retry_abort_supplement_v2_kaggle",
        ),
        train_json_relative_path="data/supplement_train.json",
    ),
)


def _expected_config() -> dict[str, Any]:
    return {
        "allowed_operations": list(ALLOWED_OPERATIONS),
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_version": EXPECTED_DATASET_VERSION,
        "datasets": [
            {
                "data_root_layouts": list(spec.data_root_layouts),
                "kaggle_dataset_version": spec.kaggle_dataset_version,
                "mount_layouts": list(spec.mount_layouts),
                "role": spec.role,
                "slug": spec.slug,
                "train_json_relative_path": spec.train_json_relative_path,
            }
            for spec in EXPECTED_DATASETS
        ],
        "execution_resource": "CPU_ONLY_NO_MODEL_OR_CHECKPOINT",
        "max_preparation_output_bytes": MAX_PREPARATION_OUTPUT_BYTES,
        "mode": MODE,
        "paper_table_status": PAPER_TABLE_STATUS,
        "preparation_output_files": list(PREPARATION_OUTPUT_FILES),
        "schema_version": CONFIG_SCHEMA_VERSION,
        "source_authority": SOURCE_AUTHORITY_RELATIVE,
    }


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise KaggleP4PrepareOnlyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_prepare_only_config(path: str | Path) -> Mapping[str, Any]:
    """Load the one registered config; semantically changed copies are rejected."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise KaggleP4PrepareOnlyError(
            "prepare-only config must be a regular non-symlink file"
        )
    try:
        payload = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise KaggleP4PrepareOnlyError(f"invalid prepare-only config: {error}") from error
    if payload != _expected_config():
        raise KaggleP4PrepareOnlyError(
            "prepare-only config differs from the registered exact contract"
        )
    return payload


def _safe_relative(value: str, *, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        if value == ".":
            return PurePosixPath(".")
        raise KaggleP4PrepareOnlyError(f"{field} is not a safe relative path")
    return path


def _lexical_child(root: Path, relative: str, *, field: str) -> Path:
    safe = _safe_relative(relative, field=field)
    return root if str(safe) == "." else root.joinpath(*safe.parts)


def _assert_no_symlink_chain(root: Path, candidate: Path, *, field: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise KaggleP4PrepareOnlyError(f"{field} escapes the input root") from error
    current = root
    if current.is_symlink():
        raise KaggleP4PrepareOnlyError(f"{field} uses a symlinked input root")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise KaggleP4PrepareOnlyError(
                f"{field} contains a symlink component: {current}"
            )


def _discover_named_mounts(input_root: Path, basename: str) -> set[Path]:
    """Find mount-shaped slug directories to reject unexpected alternatives.

    Kaggle has used both ``input/<slug>`` and
    ``input/datasets/<owner>/<slug>``.  A bounded three-level walk observes
    those layouts and unexpected variants without opening any dataset file.
    """

    found: set[Path] = set()
    root_depth = len(input_root.parts)
    for current, directories, _ in os.walk(input_root, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        if depth >= 3:
            directories[:] = []
            continue
        retained: list[str] = []
        for name in sorted(directories):
            child = current_path / name
            if name == basename:
                found.add(child)
            if depth < 2 and not child.is_symlink():
                retained.append(name)
        directories[:] = retained
    return found


def _discover_train_files(mount_root: Path, filename: str) -> set[Path]:
    """Find train JSON locations without opening JSON or validation/test files."""

    found: set[Path] = set()
    for current, directories, files in os.walk(mount_root, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in sorted(directories):
            child = current_path / name
            if name not in {"images", ".git", "__pycache__"} and not child.is_symlink():
                retained.append(name)
        directories[:] = retained
        if filename in files:
            found.add(current_path / filename)
    return found


def resolve_dataset_mount(
    input_root: str | Path,
    spec: DatasetSpec,
) -> ResolvedDatasetMount:
    """Resolve exactly one registered Kaggle mount and one known data layout."""

    root = Path(os.path.abspath(os.fspath(input_root)))
    if root.is_symlink() or not root.is_dir():
        raise KaggleP4PrepareOnlyError(
            f"Kaggle input root is not a regular non-symlink directory: {root}"
        )
    allowed_mounts: dict[Path, str] = {}
    for layout in spec.mount_layouts:
        path = _lexical_child(root, layout, field=f"{spec.role}.mount_layout")
        _assert_no_symlink_chain(root, path, field=f"{spec.role}.mount_root")
        allowed_mounts[path] = layout

    basename = spec.slug.rsplit("/", 1)[-1]
    discovered = _discover_named_mounts(root, basename)
    unexpected = sorted(discovered - set(allowed_mounts), key=str)
    if unexpected:
        raise KaggleP4PrepareOnlyError(
            f"{spec.role} has unexpected mount root(s): "
            + ", ".join(map(str, unexpected))
        )
    existing = [path for path in allowed_mounts if path.is_dir()]
    if len(existing) != 1:
        raise KaggleP4PrepareOnlyError(
            f"{spec.role} requires exactly one permitted Kaggle mount root; "
            f"found {len(existing)}: {sorted(map(str, existing))}"
        )
    mount_root = existing[0]

    expected_train_paths: dict[Path, tuple[str, Path]] = {}
    for data_layout in spec.data_root_layouts:
        data_root = _lexical_child(
            mount_root,
            data_layout,
            field=f"{spec.role}.data_root_layout",
        )
        train_json = _lexical_child(
            data_root,
            spec.train_json_relative_path,
            field=f"{spec.role}.train_json_relative_path",
        )
        _assert_no_symlink_chain(root, train_json, field=f"{spec.role}.train_json")
        expected_train_paths[train_json] = (data_layout, data_root)

    discovered_train = _discover_train_files(
        mount_root, Path(spec.train_json_relative_path).name
    )
    unexpected_train = sorted(discovered_train - set(expected_train_paths), key=str)
    if unexpected_train:
        raise KaggleP4PrepareOnlyError(
            f"{spec.role} has train JSON at unexpected root(s): "
            + ", ".join(map(str, unexpected_train))
        )
    existing_train = [
        path
        for path in expected_train_paths
        if path.is_file() and not path.is_symlink()
    ]
    if len(existing_train) != 1:
        raise KaggleP4PrepareOnlyError(
            f"{spec.role} requires exactly one permitted training JSON; "
            f"found {len(existing_train)}: {sorted(map(str, existing_train))}"
        )
    train_json = existing_train[0]
    data_layout, data_root = expected_train_paths[train_json]
    if data_root.is_symlink() or not data_root.is_dir():
        raise KaggleP4PrepareOnlyError(
            f"{spec.role} data root is not a regular directory"
        )
    return ResolvedDatasetMount(
        spec=spec,
        mount_layout=allowed_mounts[mount_root],
        mount_root=mount_root.resolve(strict=True),
        data_root_layout=data_layout,
        data_root=data_root.resolve(strict=True),
        train_json=train_json.resolve(strict=True),
    )


def resolve_registered_dataset_mounts(
    input_root: str | Path,
) -> tuple[ResolvedDatasetMount, ...]:
    return tuple(resolve_dataset_mount(input_root, spec) for spec in EXPECTED_DATASETS)


def _file_descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise KaggleP4PrepareOnlyError(f"output is not a regular file: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_compact_preparation_outputs(package_root: str | Path) -> dict[str, Any]:
    """Enforce the compact preparation-only file allowlist and size ceiling."""

    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise KaggleP4PrepareOnlyError(
            "preparation output must be a regular non-symlink directory"
        )
    files: list[Path] = []
    directories: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise KaggleP4PrepareOnlyError(f"preparation output contains symlink: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            files.append(path)
        else:
            raise KaggleP4PrepareOnlyError(f"unsupported preparation output: {path}")
    if directories:
        raise KaggleP4PrepareOnlyError("preparation output contains nested directories")
    relative = {str(path.relative_to(root)): path for path in files}
    if set(relative) != set(PREPARATION_OUTPUT_FILES):
        raise KaggleP4PrepareOnlyError(
            "preparation output file set differs from the compact allowlist"
        )
    descriptors = {
        name: _file_descriptor(relative[name]) for name in PREPARATION_OUTPUT_FILES
    }
    total_bytes = sum(row["bytes"] for row in descriptors.values())
    if total_bytes > MAX_PREPARATION_OUTPUT_BYTES:
        raise KaggleP4PrepareOnlyError(
            "preparation output exceeds the registered compact-output limit"
        )
    return {
        "files": descriptors,
        "total_bytes": total_bytes,
        "allowlist": list(PREPARATION_OUTPUT_FILES),
        "contains_images": False,
        "contains_model_or_checkpoint": False,
        "contains_provenance_manifest": False,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("web-agent", "numpy", "pillow"):
        try:
            result[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _source_receipt(
    *,
    repository_root: Path,
    source_commit: str | None,
    source_archive: str | Path | None,
) -> dict[str, Any]:
    commit = None if source_commit is None else source_commit.strip().lower()
    if commit is None:
        raise KaggleP4PrepareOnlyError(
            "prepare-only evidence requires --source-commit with the full Git SHA"
        )
    if (
        len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise KaggleP4PrepareOnlyError("source commit must be a full lowercase Git SHA")

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    inside = git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        raise KaggleP4PrepareOnlyError(
            "prepare-only evidence requires an authenticated Git checkout; "
            "an unverified extracted archive is insufficient"
        )
    head_result = git("rev-parse", "HEAD")
    if head_result.returncode != 0:
        raise KaggleP4PrepareOnlyError("unable to resolve repository Git HEAD")
    git_head = head_result.stdout.strip().lower()
    if git_head != commit:
        raise KaggleP4PrepareOnlyError(
            "supplied source commit differs from repository Git HEAD"
        )
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise KaggleP4PrepareOnlyError("unable to verify repository worktree state")
    if status.stdout:
        raise KaggleP4PrepareOnlyError(
            "prepare-only evidence requires a clean Git checkout"
        )

    source_rows: list[dict[str, str]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise KaggleP4PrepareOnlyError(
                f"executed source is missing or symlinked: {relative}"
            )
        source_rows.append({"relative_path": relative, "sha256": sha256_file(path)})

    archive: dict[str, Any] | None = None
    if source_archive is not None:
        path = Path(source_archive)
        if path.is_symlink() or not path.is_file():
            raise KaggleP4PrepareOnlyError(
                "source archive must be a regular non-symlink file"
            )
        resolved = path.resolve(strict=True)
        archive = {
            "path": str(resolved),
            **_file_descriptor(resolved),
            "evidence_role": SOURCE_ARCHIVE_EVIDENCE_ROLE,
        }
    return {
        "repository_root": str(repository_root),
        "source_commit_supplied": commit,
        "repository_git_head": git_head,
        "source_commit_verification": SOURCE_COMMIT_VERIFICATION,
        "repository_clean": True,
        "executed_source_files": source_rows,
        "executed_source_set_sha256": canonical_sha256(source_rows),
        "source_archive": archive,
    }


def _validate_source_mapping(
    value: object,
    *,
    repository_root: Path,
    require_clean_git_checkout: bool = True,
) -> dict[str, Any]:
    """Reopen the clean checkout and reproduce the executed-source binding."""

    if not isinstance(value, Mapping) or set(value) != {
        "repository_root",
        "source_commit_supplied",
        "repository_git_head",
        "source_commit_verification",
        "repository_clean",
        "executed_source_files",
        "executed_source_set_sha256",
        "source_archive",
    }:
        raise KaggleP4PrepareOnlyError("prepare-only source receipt schema mismatch")
    commit = value.get("source_commit_supplied")
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise KaggleP4PrepareOnlyError("source receipt lacks a full lowercase Git SHA")
    if (
        value.get("repository_git_head") != commit
        or value.get("source_commit_verification") != SOURCE_COMMIT_VERIFICATION
        or value.get("repository_clean") is not True
    ):
        raise KaggleP4PrepareOnlyError(
            "source receipt does not attest a clean matching Git checkout"
        )

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    if require_clean_git_checkout:
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        if (
            head.returncode != 0
            or head.stdout.strip().lower() != commit
            or status.returncode != 0
            or status.stdout
        ):
            raise KaggleP4PrepareOnlyError(
                "executed source checkout is missing, dirty, or at another commit"
            )
    expected_rows: list[dict[str, str]] = []
    for relative in EXECUTED_SOURCE_RELATIVE_PATHS:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise KaggleP4PrepareOnlyError(
                f"executed source is missing or symlinked: {relative}"
            )
        expected_rows.append(
            {"relative_path": relative, "sha256": sha256_file(path)}
        )
    if value.get("executed_source_files") != expected_rows or value.get(
        "executed_source_set_sha256"
    ) != canonical_sha256(expected_rows):
        raise KaggleP4PrepareOnlyError(
            "executed source file set differs from the clean checkout"
        )
    archive = value.get("source_archive")
    if archive is not None:
        if not isinstance(archive, Mapping) or set(archive) != {
            "path",
            "bytes",
            "sha256",
            "evidence_role",
        }:
            raise KaggleP4PrepareOnlyError("source archive descriptor is malformed")
        if archive.get("evidence_role") != SOURCE_ARCHIVE_EVIDENCE_ROLE:
            raise KaggleP4PrepareOnlyError(
                "source archive was incorrectly promoted to source authority"
            )
        if (
            type(archive.get("path")) is not str
            or not archive["path"]
            or type(archive.get("bytes")) is not int
            or archive["bytes"] < 0
            or type(archive.get("sha256")) is not str
            or len(archive["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in archive["sha256"]
            )
        ):
            raise KaggleP4PrepareOnlyError("source archive descriptor is invalid")
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def locate_prepare_only_execution_receipt(
    preparation_root: str | Path,
) -> tuple[Path, Path]:
    """Locate the outer wrapper receipt for an inner preparation package."""

    preparation = Path(preparation_root).resolve()
    candidates = []
    for root in (preparation.parent, preparation):
        receipt = root / "execution_receipt.json"
        sidecar = root / "execution_receipt.sha256"
        if receipt.exists() or sidecar.exists():
            if (
                receipt.is_symlink()
                or sidecar.is_symlink()
                or not receipt.is_file()
                or not sidecar.is_file()
            ):
                raise KaggleP4PrepareOnlyError(
                    "prepare-only execution receipt/sidecar is incomplete or unsafe"
                )
            candidates.append((receipt, sidecar))
    if len(candidates) != 1:
        raise KaggleP4PrepareOnlyError(
            "preparation package requires exactly one outer execution receipt"
        )
    return candidates[0]


def validate_prepare_only_execution_receipt(
    *,
    preparation_root: str | Path,
    repository_root: str | Path,
    require_clean_git_checkout: bool = True,
) -> dict[str, Any]:
    """Validate the source-attested outer receipt and its exact inner outputs."""

    preparation = Path(preparation_root).resolve()
    repository = Path(repository_root).resolve()
    receipt_path, sidecar_path = locate_prepare_only_execution_receipt(preparation)
    try:
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise KaggleP4PrepareOnlyError(
            f"invalid prepare-only execution receipt: {error}"
        ) from error
    if not isinstance(receipt, dict):
        raise KaggleP4PrepareOnlyError("prepare-only execution receipt must be an object")
    required = {
        "schema_version",
        "mode",
        "scientific_role",
        "paper_table_status",
        "status",
        "package_status",
        "started_at_utc",
        "ended_at_utc",
        "argv",
        "allowed_operations",
        "stage_status",
        "config",
        "source",
        "environment",
        "datasets",
        "attached_mounts",
        "inputs",
        "outputs",
        "zero_read_claims",
        "scientific_nonclaims",
        "error",
        "receipt_core_sha256",
    }
    if set(receipt) != required:
        raise KaggleP4PrepareOnlyError(
            "prepare-only execution receipt fields differ from schema"
        )
    core = dict(receipt)
    recorded_core = core.pop("receipt_core_sha256")
    if recorded_core != canonical_sha256(core):
        raise KaggleP4PrepareOnlyError("prepare-only receipt core hash mismatch")
    if sidecar_path.read_text(encoding="utf-8").strip() != sha256_file(receipt_path):
        raise KaggleP4PrepareOnlyError("prepare-only receipt sidecar mismatch")
    fixed = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "mode": MODE,
        "scientific_role": "PREPARATION_ONLY_NOT_PROVENANCE_OR_MEMORY_STORE",
        "paper_table_status": PAPER_TABLE_STATUS,
        "status": "REVIEW_REQUIRED",
        "package_status": "REVIEW_REQUIRED",
        "allowed_operations": list(ALLOWED_OPERATIONS),
        "stage_status": {
            "audit-candidates": "REVIEW_REQUIRED",
            "validate-preparation": "PASS",
            "joint-duplicate-audit": "FORBIDDEN_NOT_RUN",
            "provenance-authoring": "FORBIDDEN_NOT_RUN",
            "memory-store-build": "FORBIDDEN_NOT_RUN",
        },
        "error": None,
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt mismatch at {field}"
            )
    if not all(
        type(receipt.get(field)) is str and bool(receipt[field])
        for field in ("started_at_utc", "ended_at_utc")
    ):
        raise KaggleP4PrepareOnlyError("prepare-only receipt timestamps are missing")
    if not isinstance(receipt.get("argv"), list) or not all(
        isinstance(value, str) for value in receipt["argv"]
    ):
        raise KaggleP4PrepareOnlyError("prepare-only receipt argv is malformed")
    _validate_source_mapping(
        receipt.get("source"),
        repository_root=repository,
        require_clean_git_checkout=require_clean_git_checkout,
    )

    config = receipt.get("config")
    authority = receipt.get("inputs", {}).get("source_authority") if isinstance(
        receipt.get("inputs"), Mapping
    ) else None
    expected_inputs = (
        (
            config,
            repository / PREPARE_ONLY_CONFIG_RELATIVE,
            "config",
        ),
        (
            authority,
            repository / SOURCE_AUTHORITY_RELATIVE,
            "source authority",
        ),
    )
    for descriptor, path, role in expected_inputs:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise KaggleP4PrepareOnlyError(
                f"prepare-only {role} descriptor is malformed"
            )
        if path.is_symlink() or not path.is_file() or descriptor.get(
            "sha256"
        ) != sha256_file(path) or descriptor.get("bytes") != path.stat().st_size:
            raise KaggleP4PrepareOnlyError(
                f"prepare-only {role} differs from the validated repository"
            )

    validated_package = validate_p4_preparation_package(preparation)
    if validated_package.status != "REVIEW_REQUIRED":
        raise KaggleP4PrepareOnlyError(
            "receipt is attached to a non-reviewable preparation package"
        )
    expected_outputs = validate_compact_preparation_outputs(preparation)
    if receipt.get("outputs") != expected_outputs:
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt output descriptors differ from preparation bytes"
        )
    zero_reads = receipt.get("zero_read_claims")
    if not isinstance(zero_reads, Mapping) or zero_reads != {
        "scope": APPLICATION_ZERO_READ_SCOPE,
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "claim_limit": (
            "Records only explicit application inputs; it is not a kernel, "
            "filesystem, operating-system, or Kaggle-platform access audit."
        ),
    }:
        raise KaggleP4PrepareOnlyError("prepare-only zero-read claim is malformed")
    if receipt.get("scientific_nonclaims") != {
        "independent_review_performed": False,
        "provenance_manifest_created": False,
        "joint_duplicate_audit_performed": False,
        "eligible_memory_items_claimed": 0,
        "memory_store_built": False,
        "table2_value_created": False,
    }:
        raise KaggleP4PrepareOnlyError(
            "prepare-only scientific nonclaims are malformed"
        )
    return {
        "execution_receipt_sha256": sha256_file(receipt_path),
        "receipt_core_sha256": recorded_core,
        "executed_source_set_sha256": receipt["source"][
            "executed_source_set_sha256"
        ],
        "source_commit": receipt["source"]["source_commit_supplied"],
    }


@dataclass(frozen=True, slots=True)
class PrepareOnlyExecution:
    status: str
    output_root: Path
    receipt: Mapping[str, Any]

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "REVIEW_REQUIRED" else 2


def _run_prepare_only_impl(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    argv: Sequence[str],
    source_commit: str | None = None,
    source_archive: str | Path | None = None,
    prepare: Callable[..., P4PreparationPackage],
    validate: Callable[[str | Path], P4PreparationPackage],
) -> PrepareOnlyExecution:
    """Implementation seam; production binds the two registered operations."""

    started = _utc_now()
    repository = Path(repository_root).resolve()
    input_path = Path(os.path.abspath(os.fspath(input_root)))
    destination = Path(os.path.abspath(os.fspath(output_root)))
    if destination.exists() or destination.is_symlink():
        raise KaggleP4PrepareOnlyError(
            f"refusing to overwrite prepare-only output: {destination}"
        )
    if not repository.is_dir():
        raise KaggleP4PrepareOnlyError(f"repository root is missing: {repository}")
    try:
        destination.relative_to(input_path)
    except ValueError:
        pass
    else:
        raise KaggleP4PrepareOnlyError("prepare-only output cannot be inside Kaggle input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )

    mounts: tuple[ResolvedDatasetMount, ...] = ()
    source: Mapping[str, Any] | None = None
    source_authority: Path | None = None
    stage_status = {
        "audit-candidates": "NOT_RUN",
        "validate-preparation": "NOT_RUN",
        "joint-duplicate-audit": "FORBIDDEN_NOT_RUN",
        "provenance-authoring": "FORBIDDEN_NOT_RUN",
        "memory-store-build": "FORBIDDEN_NOT_RUN",
    }
    output_evidence: Mapping[str, Any] = {
        "files": {},
        "total_bytes": 0,
        "allowlist": list(PREPARATION_OUTPUT_FILES),
        "contains_images": False,
        "contains_model_or_checkpoint": False,
        "contains_provenance_manifest": False,
    }
    status = "FAIL"
    error_payload: dict[str, str] | None = None
    package_status: str | None = None
    try:
        load_prepare_only_config(config_path)
        source = _source_receipt(
            repository_root=repository,
            source_commit=source_commit,
            source_archive=source_archive,
        )
        source_authority = repository / SOURCE_AUTHORITY_RELATIVE
        if source_authority.is_symlink() or not source_authority.is_file():
            raise KaggleP4PrepareOnlyError(
                "registered P4 source authority is missing or symlinked"
            )
        mounts = resolve_registered_dataset_mounts(input_path)
        by_role = {row.spec.role: row for row in mounts}
        package = prepare(
            gold_train_json=by_role["original_gold"].train_json,
            gold_data_root=by_role["original_gold"].data_root,
            supplement_train_json=by_role["retry_abort_supplement_v2"].train_json,
            supplement_data_root=by_role["retry_abort_supplement_v2"].data_root,
            dataset_id=EXPECTED_DATASET_ID,
            dataset_version=EXPECTED_DATASET_VERSION,
            source_authority_path=source_authority,
            output_dir=temporary / "preparation",
        )
        package_status = package.status
        stage_status["audit-candidates"] = package.status
        validated = validate(temporary / "preparation")
        stage_status["validate-preparation"] = "PASS"
        if validated.status != "REVIEW_REQUIRED" or package.status != "REVIEW_REQUIRED":
            raise KaggleP4PrepareOnlyError(
                "authenticated preparation did not end in REVIEW_REQUIRED"
            )
        output_evidence = validate_compact_preparation_outputs(
            temporary / "preparation"
        )
        # Reopen the checkout after preparation so the receipt cannot bless a
        # source tree that changed while the candidate package was produced.
        _validate_source_mapping(source, repository_root=repository)
        status = "REVIEW_REQUIRED"
    except Exception as error:  # receipt is required for every in-boundary failure
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
        }
        preparation = temporary / "preparation"
        if preparation.exists():
            shutil.rmtree(preparation)

    dataset_receipts = [row.receipt() for row in mounts]
    config_descriptor = None
    config_candidate = Path(config_path)
    if config_candidate.is_file() and not config_candidate.is_symlink():
        config_descriptor = {
            "path": str(config_candidate.resolve()),
            **_file_descriptor(config_candidate.resolve()),
        }
    authority_descriptor = None
    if source_authority is not None and source_authority.is_file():
        authority_descriptor = {
            "path": str(source_authority.resolve()),
            **_file_descriptor(source_authority.resolve()),
        }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "mode": MODE,
        "scientific_role": "PREPARATION_ONLY_NOT_PROVENANCE_OR_MEMORY_STORE",
        "paper_table_status": PAPER_TABLE_STATUS,
        "status": status,
        "package_status": package_status,
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "argv": list(argv),
        "allowed_operations": list(ALLOWED_OPERATIONS),
        "stage_status": stage_status,
        "config": config_descriptor,
        "source": source,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "execution_resource": "CPU_ONLY_NO_MODEL_OR_CHECKPOINT",
            "gpu_or_checkpoint_loaded_by_wrapper": False,
            "dependencies": _dependency_versions(),
        },
        "datasets": dataset_receipts,
        "attached_mounts": [
            {
                "slug": row.spec.slug,
                "declared_kaggle_dataset_version": row.spec.kaggle_dataset_version,
                "resolved_mount_root": str(row.mount_root),
            }
            for row in mounts
        ],
        "inputs": {
            "source_authority": authority_descriptor,
            "train_json_files": [
                {
                    "role": row["role"],
                    "path": row["train_json"],
                    "sha256": row["train_json_sha256"],
                    "bytes": row["train_json_bytes"],
                }
                for row in dataset_receipts
            ],
        },
        "outputs": output_evidence,
        "zero_read_claims": {
            "scope": APPLICATION_ZERO_READ_SCOPE,
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "claim_limit": (
                "Records only explicit application inputs; it is not a kernel, "
                "filesystem, operating-system, or Kaggle-platform access audit."
            ),
        },
        "scientific_nonclaims": {
            "independent_review_performed": False,
            "provenance_manifest_created": False,
            "joint_duplicate_audit_performed": False,
            "eligible_memory_items_claimed": 0,
            "memory_store_built": False,
            "table2_value_created": False,
        },
        "error": error_payload,
    }
    receipt["receipt_core_sha256"] = canonical_sha256(receipt)
    _write_json(temporary / "execution_receipt.json", receipt)
    receipt_sha = sha256_file(temporary / "execution_receipt.json")
    (temporary / "execution_receipt.sha256").write_text(
        receipt_sha + "\n", encoding="utf-8"
    )
    temporary.rename(destination)
    return PrepareOnlyExecution(status=status, output_root=destination, receipt=receipt)


def run_prepare_only(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    argv: Sequence[str],
    source_commit: str | None = None,
    source_archive: str | Path | None = None,
) -> PrepareOnlyExecution:
    """Run only authenticated candidate preparation followed by validation."""

    return _run_prepare_only_impl(
        repository_root=repository_root,
        config_path=config_path,
        input_root=input_root,
        output_root=output_root,
        argv=argv,
        source_commit=source_commit,
        source_archive=source_archive,
        prepare=prepare_p4_candidate_audit,
        validate=validate_p4_preparation_package,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/table2/kaggle_p4_prepare_only_v1.json"),
    )
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/table2-p4-prepare-only-v1"),
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--source-archive", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(arguments)
    repository = args.repository_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = repository / config_path
    try:
        result = run_prepare_only(
            repository_root=repository,
            config_path=config_path,
            input_root=args.input_root,
            output_root=args.output_root,
            argv=arguments,
            source_commit=args.source_commit,
            source_archive=args.source_archive,
        )
    except KaggleP4PrepareOnlyError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "mode": MODE,
                "output_root": str(result.output_root),
                "receipt_core_sha256": result.receipt["receipt_core_sha256"],
                "paper_table_status": PAPER_TABLE_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
