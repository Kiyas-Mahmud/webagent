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
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from web_agent.memory.manifest import canonical_sha256, sha256_file
from web_agent.memory.preparation import (
    P4_SOURCE_AUTHORITY_SCHEMA_VERSION,
    P4PreparationPackage,
    prepare_p4_candidate_audit,
    validate_p4_preparation_package,
)


CONFIG_SCHEMA_VERSION = "table2-p4-kaggle-prepare-config-v1"
RECEIPT_SCHEMA_VERSION = "table2-p4-kaggle-prepare-receipt-v3"
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
SOURCE_TRANSPORT_EVIDENCE_ROLE = (
    "GIT_BUNDLE_TRANSPORT_NOT_SCIENTIFIC_SOURCE_AUTHORITY"
)
# Compatibility import name only; receipt schema v3 calls this evidence a
# source transport and accepts only a verified Git bundle.
SOURCE_ARCHIVE_EVIDENCE_ROLE = SOURCE_TRANSPORT_EVIDENCE_ROLE
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
DOWNLOADED_OUTPUT_ENTRIES = (
    "execution_receipt.json",
    "execution_receipt.sha256",
    "preparation",
)
ALLOWED_OPERATIONS = ("audit-candidates", "validate-preparation")
_RECEIPT_ARGV_FLAGS = (
    "--repository-root",
    "--config",
    "--input-root",
    "--output-root",
    "--source-commit",
    "--source-bundle",
)
_ENVIRONMENT_DEPENDENCIES = ("web-agent", "numpy", "pillow")


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
    metadata = path.stat()
    if metadata.st_nlink != 1:
        raise KaggleP4PrepareOnlyError(
            f"preparation output must not be hard-linked: {path}"
        )
    return {"bytes": metadata.st_size, "sha256": sha256_file(path)}


def _regular_unlinked_file(path: str | Path, *, role: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise KaggleP4PrepareOnlyError(f"{role} is missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise KaggleP4PrepareOnlyError(f"{role} contains a symlink component")
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise KaggleP4PrepareOnlyError(f"{role} must be a regular file")
    if metadata.st_nlink != 1:
        raise KaggleP4PrepareOnlyError(f"{role} must not be hard-linked")
    return candidate


def _regular_directory_without_symlink_ancestry(
    path: str | Path, *, role: str
) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise KaggleP4PrepareOnlyError(f"{role} is missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise KaggleP4PrepareOnlyError(f"{role} contains a symlink component")
    metadata = candidate.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise KaggleP4PrepareOnlyError(f"{role} must be a regular directory")
    return candidate


def _validate_downloaded_output_root(path: str | Path) -> tuple[Path, Path]:
    """Close the downloaded wrapper tree before trusting its inner receipt."""

    root = _regular_directory_without_symlink_ancestry(
        path,
        role="downloaded prepare-only output root",
    )
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise KaggleP4PrepareOnlyError(
            "downloaded prepare-only output root is inaccessible"
        ) from error
    if {entry.name for entry in entries} != set(DOWNLOADED_OUTPUT_ENTRIES):
        raise KaggleP4PrepareOnlyError(
            "downloaded prepare-only output root differs from the exact allowlist"
        )

    preparation = _regular_directory_without_symlink_ancestry(
        root / "preparation",
        role="downloaded prepare-only preparation directory",
    )
    _regular_unlinked_file(
        root / "execution_receipt.json",
        role="downloaded prepare-only execution receipt",
    )
    _regular_unlinked_file(
        root / "execution_receipt.sha256",
        role="downloaded prepare-only execution receipt sidecar",
    )
    return root, preparation


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


def _canonical_absolute_path(value: object, *, field: str) -> str:
    """Validate a portable, lexical absolute path without reopening it.

    Receipt replay normally happens on a different host from the Kaggle run,
    so original paths cannot be required to exist.  Their spelling can still
    be exact and can be cross-bound to the recorded invocation and datasets.
    """

    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise KaggleP4PrepareOnlyError(f"{field} must be a canonical absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or os.path.normpath(value) != value
    ):
        raise KaggleP4PrepareOnlyError(f"{field} must be a canonical absolute path")
    return value


def _canonical_receipt_argv(
    *,
    repository_root: Path,
    config_path: Path,
    input_root: Path,
    output_root: Path,
    source_commit: str,
    source_bundle: Path,
) -> list[str]:
    """Return the one normalized semantic invocation stored in a PASS receipt."""

    return [
        "--repository-root",
        str(repository_root),
        "--config",
        str(config_path),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
        "--source-commit",
        source_commit,
        "--source-bundle",
        str(source_bundle),
    ]


def _validate_receipt_argv(value: object) -> dict[str, str]:
    if not isinstance(value, list) or len(value) != 2 * len(_RECEIPT_ARGV_FLAGS):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt argv is not the canonical registered invocation"
        )
    parsed: dict[str, str] = {}
    for index, flag in enumerate(_RECEIPT_ARGV_FLAGS):
        flag_index = 2 * index
        argument = value[flag_index + 1]
        if value[flag_index] != flag or type(argument) is not str:
            raise KaggleP4PrepareOnlyError(
                "prepare-only receipt argv is not the canonical registered invocation"
            )
        parsed[flag.removeprefix("--").replace("-", "_")] = argument
    for field in (
        "repository_root",
        "config",
        "input_root",
        "output_root",
        "source_bundle",
    ):
        parsed[field] = _canonical_absolute_path(
            parsed[field], field=f"receipt.argv.{field}"
        )
    commit = parsed["source_commit"]
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise KaggleP4PrepareOnlyError(
            "receipt.argv.source_commit must be a full lowercase Git SHA"
        )
    return parsed


def _validate_receipt_timestamps(receipt: Mapping[str, Any]) -> None:
    parsed: list[datetime] = []
    for field in ("started_at_utc", "ended_at_utc"):
        value = receipt.get(field)
        if type(value) is not str or not value:
            raise KaggleP4PrepareOnlyError(
                "prepare-only receipt timestamps are missing"
            )
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as error:
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt {field} is not canonical ISO-8601"
            ) from error
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
            or timestamp.isoformat() != value
        ):
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt {field} is not canonical UTC ISO-8601"
            )
        parsed.append(timestamp)
    if parsed[1] < parsed[0]:
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt ended before it started"
        )


def _validate_receipt_environment(value: object) -> dict[str, Any]:
    required = {
        "python",
        "implementation",
        "platform",
        "cpu_count",
        "execution_resource",
        "gpu_or_checkpoint_loaded_by_wrapper",
        "dependencies",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt environment schema mismatch"
        )
    for field in ("python", "implementation", "platform"):
        text = value.get(field)
        if type(text) is not str or not text or text != text.strip():
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt environment.{field} is malformed"
            )
    python_parts = str(value["python"]).split(".")
    if len(python_parts) != 3 or not all(part.isdigit() for part in python_parts):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt environment.python is malformed"
        )
    cpu_count = value.get("cpu_count")
    if cpu_count is not None and (type(cpu_count) is not int or cpu_count < 1):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt environment.cpu_count is malformed"
        )
    if (
        value.get("execution_resource") != "CPU_ONLY_NO_MODEL_OR_CHECKPOINT"
        or value.get("gpu_or_checkpoint_loaded_by_wrapper") is not False
    ):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt violates the fixed CPU/no-model boundary"
        )
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        _ENVIRONMENT_DEPENDENCIES
    ):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt dependency inventory schema mismatch"
        )
    for name in _ENVIRONMENT_DEPENDENCIES:
        version = dependencies[name]
        if version is not None and (
            type(version) is not str or not version or version != version.strip()
        ):
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt dependency version is malformed: {name}"
            )
    return dict(value)


def _safe_git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "TZ": "UTC",
    }


def _system_git_executable() -> str:
    candidate = shutil.which("git", path=os.defpath)
    if candidate is None:
        raise KaggleP4PrepareOnlyError("system-default Git executable is unavailable")
    resolved = Path(candidate).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise KaggleP4PrepareOnlyError(
            "system-default Git executable is not a regular executable"
        )
    return str(resolved)


def _source_receipt(
    *,
    repository_root: Path,
    source_commit: str | None,
    source_bundle: str | Path | None,
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
            [_system_git_executable(), "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=_safe_git_environment(),
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

    transport: dict[str, Any] | None = None
    if source_bundle is not None:
        resolved = _regular_unlinked_file(source_bundle, role="source bundle")
        verify = git("bundle", "verify", str(resolved))
        if verify.returncode != 0:
            raise KaggleP4PrepareOnlyError("source transport failed git bundle verify")
        heads = git("bundle", "list-heads", str(resolved))
        if heads.returncode != 0 or heads.stdout.strip().splitlines() != [
            f"{commit} HEAD"
        ]:
            raise KaggleP4PrepareOnlyError(
                "source bundle must advertise exactly the supplied commit as HEAD"
            )
        transport = {
            "path": str(resolved),
            **_file_descriptor(resolved),
            "format": "git_bundle",
            "bundle_head": commit,
            "evidence_role": SOURCE_TRANSPORT_EVIDENCE_ROLE,
            "scientific_source_authority": False,
            "repository_authentication": SOURCE_COMMIT_VERIFICATION,
        }
    return {
        "repository_root": str(repository_root),
        "source_commit_supplied": commit,
        "repository_git_head": git_head,
        "source_commit_verification": SOURCE_COMMIT_VERIFICATION,
        "repository_clean": True,
        "executed_source_files": source_rows,
        "executed_source_set_sha256": canonical_sha256(source_rows),
        "source_transport": transport,
    }


def _validate_source_mapping(
    value: object,
    *,
    repository_root: Path,
    require_clean_git_checkout: bool = True,
    recorded_repository_root: str | None = None,
    recorded_source_commit: str | None = None,
    recorded_source_bundle: str | None = None,
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
        "source_transport",
    }:
        raise KaggleP4PrepareOnlyError("prepare-only source receipt schema mismatch")
    source_repository_root = _canonical_absolute_path(
        value.get("repository_root"), field="source.repository_root"
    )
    expected_repository_root = (
        str(repository_root)
        if recorded_repository_root is None
        else recorded_repository_root
    )
    if source_repository_root != expected_repository_root:
        raise KaggleP4PrepareOnlyError(
            "source.repository_root differs from the canonical invocation"
        )
    commit = value.get("source_commit_supplied")
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise KaggleP4PrepareOnlyError("source receipt lacks a full lowercase Git SHA")
    if recorded_source_commit is not None and commit != recorded_source_commit:
        raise KaggleP4PrepareOnlyError(
            "source commit differs from the canonical invocation"
        )
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
            [_system_git_executable(), "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=_safe_git_environment(),
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
    transport = value.get("source_transport")
    if transport is None:
        raise KaggleP4PrepareOnlyError(
            "registered prepare-only evidence requires Git-bundle transport"
        )
    if transport is not None:
        if not isinstance(transport, Mapping) or set(transport) != {
            "path",
            "bytes",
            "sha256",
            "format",
            "bundle_head",
            "evidence_role",
            "scientific_source_authority",
            "repository_authentication",
        }:
            raise KaggleP4PrepareOnlyError("source transport descriptor is malformed")
        if (
            transport.get("format") != "git_bundle"
            or transport.get("bundle_head") != commit
            or transport.get("evidence_role") != SOURCE_TRANSPORT_EVIDENCE_ROLE
            or transport.get("scientific_source_authority") is not False
            or transport.get("repository_authentication")
            != SOURCE_COMMIT_VERIFICATION
        ):
            raise KaggleP4PrepareOnlyError(
                "source transport role or commit binding is invalid"
            )
        if (
            type(transport.get("path")) is not str
            or not transport["path"]
            or type(transport.get("bytes")) is not int
            or transport["bytes"] < 1
            or type(transport.get("sha256")) is not str
            or len(transport["sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in transport["sha256"]
            )
        ):
            raise KaggleP4PrepareOnlyError("source transport descriptor is invalid")
        transport_path = _canonical_absolute_path(
            transport.get("path"), field="source.source_transport.path"
        )
        if (
            recorded_source_bundle is not None
            and transport_path != recorded_source_bundle
        ):
            raise KaggleP4PrepareOnlyError(
                "source transport path differs from the canonical invocation"
            )
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


def _load_strict_json_object(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise KaggleP4PrepareOnlyError(f"{role} must be a regular non-symlink file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KaggleP4PrepareOnlyError(f"invalid {role}: {error}") from error
    if not isinstance(payload, dict):
        raise KaggleP4PrepareOnlyError(f"{role} must be a JSON object")
    return payload


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise KaggleP4PrepareOnlyError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_receipt_file_descriptor(
    value: object,
    *,
    role: str,
    recorded_path: str,
    replay_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise KaggleP4PrepareOnlyError(
            f"prepare-only {role} descriptor is malformed"
        )
    if _canonical_absolute_path(value.get("path"), field=f"{role}.path") != recorded_path:
        raise KaggleP4PrepareOnlyError(
            f"prepare-only {role} path differs from the canonical invocation"
        )
    byte_count = value.get("bytes")
    digest = _require_sha256(value.get("sha256"), field=f"{role}.sha256")
    if type(byte_count) is not int or byte_count < 1:
        raise KaggleP4PrepareOnlyError(
            f"prepare-only {role} byte count is malformed"
        )
    if (
        replay_path.is_symlink()
        or not replay_path.is_file()
        or replay_path.stat().st_size != byte_count
        or sha256_file(replay_path) != digest
    ):
        raise KaggleP4PrepareOnlyError(
            f"prepare-only {role} differs from the validated repository"
        )
    return dict(value)


def _recorded_child(root: str, relative: str, *, field: str) -> str:
    safe = _safe_relative(relative, field=field)
    child = Path(root) if str(safe) == "." else Path(root).joinpath(*safe.parts)
    return _canonical_absolute_path(str(child), field=field)


def _validate_receipt_dataset_bindings(
    receipt: Mapping[str, Any],
    *,
    invocation: Mapping[str, str],
    preparation: Path,
    repository: Path,
) -> None:
    """Close receipt datasets against registered inputs and inner evidence."""

    recorded_repository = invocation["repository_root"]
    expected_recorded_config = _recorded_child(
        recorded_repository,
        PREPARE_ONLY_CONFIG_RELATIVE,
        field="receipt.config.path",
    )
    if invocation["config"] != expected_recorded_config:
        raise KaggleP4PrepareOnlyError(
            "receipt argv config is not the registered repository config"
        )
    _validate_receipt_file_descriptor(
        receipt.get("config"),
        role="config",
        recorded_path=expected_recorded_config,
        replay_path=repository / PREPARE_ONLY_CONFIG_RELATIVE,
    )

    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "source_authority",
        "train_json_files",
    }:
        raise KaggleP4PrepareOnlyError("prepare-only receipt inputs schema mismatch")
    expected_recorded_authority = _recorded_child(
        recorded_repository,
        SOURCE_AUTHORITY_RELATIVE,
        field="receipt.inputs.source_authority.path",
    )
    authority_descriptor = _validate_receipt_file_descriptor(
        inputs.get("source_authority"),
        role="source authority",
        recorded_path=expected_recorded_authority,
        replay_path=repository / SOURCE_AUTHORITY_RELATIVE,
    )

    ledger = _load_strict_json_object(
        preparation / "read_ledger.json", role="preparation read ledger"
    )
    inner_authority_path = preparation / "source_authority.json"
    inner_authority = _load_strict_json_object(
        inner_authority_path, role="preparation source authority"
    )
    replay_authority = _load_strict_json_object(
        repository / SOURCE_AUTHORITY_RELATIVE,
        role="registered replay source authority",
    )
    if inner_authority != replay_authority:
        raise KaggleP4PrepareOnlyError(
            "preparation source authority differs from the registered checkout"
        )
    if (
        authority_descriptor["sha256"] != sha256_file(inner_authority_path)
        or ledger.get("source_authority_sha256") != authority_descriptor["sha256"]
    ):
        raise KaggleP4PrepareOnlyError(
            "receipt/read-ledger/source-authority hash binding mismatch"
        )
    authority_required = {
        "authority_id",
        "authority_version",
        "dataset_id",
        "dataset_version",
        "schema_version",
        "sources",
    }
    if set(inner_authority) != authority_required or (
        inner_authority.get("schema_version") != P4_SOURCE_AUTHORITY_SCHEMA_VERSION
        or inner_authority.get("dataset_id") != EXPECTED_DATASET_ID
        or inner_authority.get("dataset_version") != EXPECTED_DATASET_VERSION
    ):
        raise KaggleP4PrepareOnlyError(
            "preparation source authority differs from the registered dataset identity"
        )

    dataset_rows = receipt.get("datasets")
    ledger_rows = ledger.get("train_json_files")
    authority_rows = inner_authority.get("sources")
    if not all(isinstance(rows, list) for rows in (dataset_rows, ledger_rows, authority_rows)):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt dataset evidence must use ordered lists"
        )
    if not (
        len(dataset_rows) == len(ledger_rows) == len(authority_rows) == len(EXPECTED_DATASETS)
    ):
        raise KaggleP4PrepareOnlyError(
            "prepare-only receipt dataset count differs from the registered contract"
        )

    expected_train_inputs: list[dict[str, Any]] = []
    expected_attached_mounts: list[dict[str, Any]] = []
    dataset_fields = {
        "role",
        "slug",
        "declared_kaggle_dataset_version",
        "kaggle_platform_version_verification",
        "mount_layout",
        "resolved_mount_root",
        "data_root_layout",
        "resolved_data_root",
        "train_json",
        "train_json_sha256",
        "train_json_bytes",
    }
    ledger_fields = {"role", "file_name", "records", "bytes", "sha256"}
    authority_fields = {"role", "file_name", "records", "sha256"}
    for index, spec in enumerate(EXPECTED_DATASETS):
        dataset = dataset_rows[index]
        ledger_row = ledger_rows[index]
        authority_row = authority_rows[index]
        if not isinstance(dataset, Mapping) or set(dataset) != dataset_fields:
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt datasets[{index}] schema mismatch"
            )
        if not isinstance(ledger_row, Mapping) or set(ledger_row) != ledger_fields:
            raise KaggleP4PrepareOnlyError(
                f"preparation read-ledger train_json_files[{index}] schema mismatch"
            )
        if not isinstance(authority_row, Mapping) or set(authority_row) != authority_fields:
            raise KaggleP4PrepareOnlyError(
                f"preparation source-authority sources[{index}] schema mismatch"
            )
        if (
            dataset.get("role") != spec.role
            or dataset.get("slug") != spec.slug
            or dataset.get("declared_kaggle_dataset_version")
            != spec.kaggle_dataset_version
            or dataset.get("kaggle_platform_version_verification")
            != "DECLARED_ONLY_NOT_QUERIED_BY_WRAPPER"
            or dataset.get("mount_layout") not in spec.mount_layouts
            or dataset.get("data_root_layout") not in spec.data_root_layouts
        ):
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt datasets[{index}] identity/layout mismatch"
            )

        mount_root = _recorded_child(
            invocation["input_root"],
            str(dataset["mount_layout"]),
            field=f"receipt.datasets[{index}].resolved_mount_root",
        )
        data_root = _recorded_child(
            mount_root,
            str(dataset["data_root_layout"]),
            field=f"receipt.datasets[{index}].resolved_data_root",
        )
        train_json = _recorded_child(
            data_root,
            spec.train_json_relative_path,
            field=f"receipt.datasets[{index}].train_json",
        )
        if (
            _canonical_absolute_path(
                dataset.get("resolved_mount_root"),
                field=f"receipt.datasets[{index}].resolved_mount_root",
            )
            != mount_root
            or _canonical_absolute_path(
                dataset.get("resolved_data_root"),
                field=f"receipt.datasets[{index}].resolved_data_root",
            )
            != data_root
            or _canonical_absolute_path(
                dataset.get("train_json"),
                field=f"receipt.datasets[{index}].train_json",
            )
            != train_json
        ):
            raise KaggleP4PrepareOnlyError(
                f"prepare-only receipt datasets[{index}] path binding mismatch"
            )
        byte_count = dataset.get("train_json_bytes")
        digest = _require_sha256(
            dataset.get("train_json_sha256"),
            field=f"receipt.datasets[{index}].train_json_sha256",
        )
        if type(byte_count) is not int or byte_count < 1:
            raise KaggleP4PrepareOnlyError(
                f"receipt.datasets[{index}].train_json_bytes is malformed"
            )
        expected_file_name = Path(spec.train_json_relative_path).name
        records = ledger_row.get("records")
        if type(records) is not int or records < 0:
            raise KaggleP4PrepareOnlyError(
                f"preparation read-ledger records[{index}] is malformed"
            )
        expected_ledger = {
            "role": spec.role,
            "file_name": expected_file_name,
            "records": records,
            "bytes": byte_count,
            "sha256": digest,
        }
        if dict(ledger_row) != expected_ledger:
            raise KaggleP4PrepareOnlyError(
                f"receipt dataset differs from read-ledger source {spec.role}"
            )
        expected_authority = {
            "role": spec.role,
            "file_name": expected_file_name,
            "records": records,
            "sha256": digest,
        }
        if dict(authority_row) != expected_authority:
            raise KaggleP4PrepareOnlyError(
                f"receipt dataset differs from source authority {spec.role}"
            )
        expected_train_inputs.append(
            {
                "role": spec.role,
                "path": train_json,
                "sha256": digest,
                "bytes": byte_count,
            }
        )
        expected_attached_mounts.append(
            {
                "slug": spec.slug,
                "declared_kaggle_dataset_version": spec.kaggle_dataset_version,
                "resolved_mount_root": mount_root,
            }
        )

    if inputs.get("train_json_files") != expected_train_inputs:
        raise KaggleP4PrepareOnlyError(
            "receipt train inputs differ from registered dataset/read-ledger bindings"
        )
    if receipt.get("attached_mounts") != expected_attached_mounts:
        raise KaggleP4PrepareOnlyError(
            "receipt attached mounts differ from registered dataset bindings"
        )


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
    _validate_receipt_timestamps(receipt)
    invocation = _validate_receipt_argv(receipt.get("argv"))
    _validate_receipt_environment(receipt.get("environment"))
    _validate_source_mapping(
        receipt.get("source"),
        repository_root=repository,
        require_clean_git_checkout=require_clean_git_checkout,
        recorded_repository_root=invocation["repository_root"],
        recorded_source_commit=invocation["source_commit"],
        recorded_source_bundle=invocation["source_bundle"],
    )

    validated_package = validate_p4_preparation_package(preparation)
    if validated_package.status != "REVIEW_REQUIRED":
        raise KaggleP4PrepareOnlyError(
            "receipt is attached to a non-reviewable preparation package"
        )
    _validate_receipt_dataset_bindings(
        receipt,
        invocation=invocation,
        preparation=preparation,
        repository=repository,
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


def _build_downloaded_output_validation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a downloaded Table 2 Kaggle P4 prepare-only output "
            "against the exact clean source checkout that produced it."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "downloaded table2-p4-prepare-only-v1 directory containing "
            "execution_receipt.json, its sidecar, and preparation/"
        ),
    )
    return parser


def validate_downloaded_output_main(argv: Sequence[str] | None = None) -> int:
    """CLI boundary for strict replay of a downloaded outer receipt."""

    args = _build_downloaded_output_validation_parser().parse_args(argv)
    output_root = args.output_root
    try:
        output_root, preparation = _validate_downloaded_output_root(output_root)
        binding = validate_prepare_only_execution_receipt(
            preparation_root=preparation,
            repository_root=args.repository_root,
            require_clean_git_checkout=True,
        )
    except (KaggleP4PrepareOnlyError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "validation_scope": (
                        "DOWNLOADED_PREPARE_ONLY_RECEIPT_AND_OUTPUT_BYTES"
                    ),
                    "paper_table_status": PAPER_TABLE_STATUS,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "validation_scope": (
                    "DOWNLOADED_PREPARE_ONLY_RECEIPT_AND_OUTPUT_BYTES"
                ),
                "package_status": "REVIEW_REQUIRED",
                "paper_table_status": PAPER_TABLE_STATUS,
                "output_root": str(output_root.resolve()),
                **binding,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
    source_bundle: str | Path | None = None,
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
            source_bundle=source_bundle,
        )
        if source["source_transport"] is None:
            raise KaggleP4PrepareOnlyError(
                "registered prepare-only execution requires --source-bundle"
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
        # Reopen and rehash the Git bundle as well as the source checkout.
        # Later receipt replay may legitimately lack the transport dataset, so
        # this byte-level recheck happens inside the original Kaggle boundary.
        if _source_receipt(
            repository_root=repository,
            source_commit=source_commit,
            source_bundle=source_bundle,
        ) != source:
            raise KaggleP4PrepareOnlyError(
                "source checkout or Git-bundle transport changed during preparation"
            )
        expected_argv = _canonical_receipt_argv(
            repository_root=repository,
            config_path=Path(config_path).resolve(),
            input_root=input_path,
            output_root=destination,
            source_commit=str(source["source_commit_supplied"]),
            source_bundle=Path(str(source["source_transport"]["path"])),
        )
        if list(argv) != expected_argv:
            raise KaggleP4PrepareOnlyError(
                "prepare-only invocation does not match the canonical registered argv"
            )
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
    source_bundle: str | Path | None = None,
    source_archive: str | Path | None = None,
) -> PrepareOnlyExecution:
    """Run only authenticated candidate preparation followed by validation."""

    if source_bundle is not None and source_archive is not None:
        raise KaggleP4PrepareOnlyError(
            "supply source_bundle only; source_archive is a deprecated alias"
        )
    transport = source_bundle if source_bundle is not None else source_archive

    return _run_prepare_only_impl(
        repository_root=repository_root,
        config_path=config_path,
        input_root=input_root,
        output_root=output_root,
        argv=argv,
        source_commit=source_commit,
        source_bundle=transport,
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
    parser.add_argument(
        "--source-bundle",
        "--source-archive",
        dest="source_bundle",
        type=Path,
        help=(
            "Git-bundle transport evidence; --source-archive is a deprecated "
            "compatibility spelling and still requires a Git bundle"
        ),
    )
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
            source_bundle=args.source_bundle,
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
