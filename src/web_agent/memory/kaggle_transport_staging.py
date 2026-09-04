"""Local-only staging for the registered Kaggle P4 Git-bundle transport.

The staging operation creates upload-ready *local directories* and never calls
Kaggle. Dataset and kernel IDs must be supplied explicitly; placeholder IDs,
dirty repositories, and existing destinations fail closed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


TRANSPORT_MANIFEST_NAME = "table2-p4-git-bundle-transport-v1.json"
TRANSPORT_MANIFEST_SCHEMA = "table2-p4-git-bundle-transport-v1"
TRANSPORT_EVIDENCE_ROLE = (
    "GIT_BUNDLE_MATERIALIZED_TRANSPORT_NOT_SCIENTIFIC_SOURCE_AUTHORITY"
)
SOURCE_DATASET_SLUG = "table2-p4-source-transport-v1"
KERNEL_SLUG = "table2-p4-prepare-only-v1"
SOURCE_DATASET_POLICY_SCHEMA = "table2-p4-source-dataset-private-policy-v1"
BUNDLE_NAME = "webagent-table2-p4-source-v1.bundle"
CHECKOUT_RELATIVE_PATH = "webagent-table2-p4-source-v1"
OUTPUT_RELATIVE_PATH = "table2-p4-prepare-only-v1"
BOOTSTRAP_RELATIVE_PATH = "kaggle/table2_p4_prepare_only/run.py"
REGISTERED_DATASET_SOURCES = (
    "kiyasmahmud/web-gold-40k",
    "kiyasmahmud/gold-40k-retry",
)
_KAGGLE_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/[a-z0-9][a-z0-9-]{2,49}\Z"
)


class KaggleTransportStagingError(ValueError):
    """The local source transport could not be staged safely."""


@dataclass(frozen=True, slots=True)
class StagedKaggleTransport:
    source_dataset_directory: Path
    kernel_directory: Path
    repository_commit: str
    bundle_sha256: str
    transport_manifest_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _validate_kaggle_id(value: str, *, field: str) -> str:
    if _KAGGLE_ID_RE.fullmatch(value) is None:
        raise KaggleTransportStagingError(
            f"{field} must be an explicit owner/slug Kaggle ID"
        )
    lowered = value.casefold()
    if any(
        token in lowered
        for token in ("placeholder", "replace_with", "replace-", "your_", "your-")
    ):
        raise KaggleTransportStagingError(f"{field} contains a placeholder")
    return value


def _title_from_id(identifier: str) -> str:
    slug = identifier.split("/", 1)[1]
    title = " ".join(part.capitalize() for part in re.split(r"[-_.]+", slug) if part)
    if not 6 <= len(title) <= 50:
        raise KaggleTransportStagingError(
            "Kaggle ID slug does not produce a 6-50 character metadata title"
        )
    return title


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    candidate = shutil.which("git", path=os.defpath)
    if candidate is None:
        raise KaggleTransportStagingError(
            "system-default Git executable is unavailable"
        )
    git_path = Path(candidate).resolve()
    if not git_path.is_file() or not os.access(git_path, os.X_OK):
        raise KaggleTransportStagingError(
            "system-default Git executable is not a regular executable"
        )
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "TZ": "UTC",
    }
    try:
        return subprocess.run(
            [str(git_path), "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise KaggleTransportStagingError(f"unable to execute Git safely: {error}") from error


def _git_output(repository: Path, *arguments: str, operation: str) -> str:
    result = _git(repository, *arguments)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise KaggleTransportStagingError(f"{operation} failed: {detail[:500]}")
    return result.stdout.strip()


def _clean_source_commit(repository: Path) -> str:
    inside = _git_output(
        repository,
        "rev-parse",
        "--is-inside-work-tree",
        operation="repository verification",
    )
    if inside != "true":
        raise KaggleTransportStagingError("repository root is not a Git worktree")
    commit = _git_output(
        repository, "rev-parse", "HEAD", operation="source HEAD resolution"
    ).lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise KaggleTransportStagingError("source HEAD is not a full Git commit")
    status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        operation="source cleanliness verification",
    )
    if status:
        raise KaggleTransportStagingError(
            "source repository must be clean, including untracked files"
        )
    for relative in (
        BOOTSTRAP_RELATIVE_PATH,
        "scripts/run_table2_p4_kaggle_prepare_only.py",
        "configs/eval/table2/kaggle_p4_prepare_only_v1.json",
    ):
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise KaggleTransportStagingError(
                f"registered source file is missing or symlinked: {relative}"
            )
        _git_output(
            repository,
            "ls-files",
            "--error-unmatch",
            relative,
            operation=f"tracked-source verification for {relative}",
        )
    return commit


def _assert_new_destination(path: Path, *, role: str) -> Path:
    destination = Path(os.path.abspath(os.fspath(path)))
    if destination.exists() or destination.is_symlink():
        raise KaggleTransportStagingError(f"refusing to overwrite {role}: {destination}")
    current = Path(destination.anchor)
    for part in destination.parent.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise KaggleTransportStagingError(
                f"{role} parent path contains a symlink: {current}"
            )
    return destination


def _assert_outside_repository(repository: Path, destination: Path, *, role: str) -> None:
    try:
        destination.relative_to(repository)
    except ValueError:
        return
    raise KaggleTransportStagingError(f"{role} cannot be staged inside the source repository")


def stage_kaggle_transport(
    *,
    repository_root: str | Path,
    source_dataset_id: str,
    source_dataset_version: int,
    kernel_id: str,
    source_output_dir: str | Path,
    kernel_output_dir: str | Path,
) -> StagedKaggleTransport:
    """Create local dataset/kernel staging directories without remote actions."""

    source_id = _validate_kaggle_id(source_dataset_id, field="source_dataset_id")
    selected_kernel_id = _validate_kaggle_id(kernel_id, field="kernel_id")
    if source_id.split("/", 1)[1] != SOURCE_DATASET_SLUG:
        raise KaggleTransportStagingError(
            f"source_dataset_id slug must be exactly {SOURCE_DATASET_SLUG}"
        )
    if selected_kernel_id.split("/", 1)[1] != KERNEL_SLUG:
        raise KaggleTransportStagingError(
            f"kernel_id slug must be exactly {KERNEL_SLUG}"
        )
    if source_id in REGISTERED_DATASET_SOURCES:
        raise KaggleTransportStagingError(
            "source transport dataset must not reuse a registered Gold dataset ID"
        )
    if source_dataset_version != 1:
        raise KaggleTransportStagingError(
            "source_dataset_version must be explicitly supplied as 1 for schema v1"
        )
    repository_candidate = Path(os.path.abspath(os.fspath(repository_root)))
    if repository_candidate.is_symlink() or not repository_candidate.is_dir():
        raise KaggleTransportStagingError(
            "repository root must be a regular non-symlink directory"
        )
    repository = repository_candidate.resolve(strict=True)
    commit = _clean_source_commit(repository)
    source_destination = _assert_new_destination(
        Path(source_output_dir), role="source-dataset staging directory"
    )
    kernel_destination = _assert_new_destination(
        Path(kernel_output_dir), role="kernel staging directory"
    )
    if source_destination == kernel_destination:
        raise KaggleTransportStagingError("source and kernel staging directories must differ")
    for outer, inner in (
        (source_destination, kernel_destination),
        (kernel_destination, source_destination),
    ):
        try:
            inner.relative_to(outer)
        except ValueError:
            continue
        raise KaggleTransportStagingError(
            "source and kernel staging directories must not be nested"
        )
    _assert_outside_repository(
        repository, source_destination, role="source-dataset staging directory"
    )
    _assert_outside_repository(repository, kernel_destination, role="kernel staging directory")

    created: list[Path] = []
    try:
        source_destination.parent.mkdir(parents=True, exist_ok=True)
        kernel_destination.parent.mkdir(parents=True, exist_ok=True)
        for destination, role in (
            (source_destination, "source-dataset staging directory"),
            (kernel_destination, "kernel staging directory"),
        ):
            if destination.parent.is_symlink() or not destination.parent.is_dir():
                raise KaggleTransportStagingError(
                    f"{role} parent must be a regular non-symlink directory"
                )
        source_destination.mkdir()
        created.append(source_destination)
        kernel_destination.mkdir()
        created.append(kernel_destination)

        bundle = source_destination / BUNDLE_NAME
        result = _git(repository, "bundle", "create", str(bundle), "HEAD")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise KaggleTransportStagingError(
                f"Git bundle creation failed: {detail[:500]}"
            )
        if bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size < 1:
            raise KaggleTransportStagingError("Git did not create a regular non-empty bundle")
        heads = _git_output(
            repository,
            "bundle",
            "list-heads",
            str(bundle),
            operation="Git bundle HEAD inspection",
        ).splitlines()
        if heads != [f"{commit} HEAD"]:
            raise KaggleTransportStagingError(
                "created Git bundle does not advertise exactly source HEAD"
            )
        _git_output(
            repository,
            "bundle",
            "verify",
            str(bundle),
            operation="Git bundle verification",
        )

        bootstrap_source = repository / BOOTSTRAP_RELATIVE_PATH
        transport_manifest = {
            "schema_version": TRANSPORT_MANIFEST_SCHEMA,
            "evidence_role": TRANSPORT_EVIDENCE_ROLE,
            "source_dataset_id": source_id,
            "source_dataset_version": source_dataset_version,
            "repository_commit": commit,
            "bundle": {
                "relative_path": BUNDLE_NAME,
                "bytes": bundle.stat().st_size,
                "sha256": _sha256_file(bundle),
                "format": "git_bundle",
            },
            "bootstrap": {
                "repository_relative_path": BOOTSTRAP_RELATIVE_PATH,
                "sha256": _sha256_file(bootstrap_source),
            },
        }
        manifest_path = source_destination / TRANSPORT_MANIFEST_NAME
        _write_json(manifest_path, transport_manifest)
        _write_json(
            source_destination / "dataset-metadata.json",
            {
                "title": _title_from_id(source_id),
                "id": source_id,
                "licenses": [{"name": "unknown"}],
            },
        )
        _write_json(
            source_destination / "private-upload-policy.json",
            {
                "schema_version": SOURCE_DATASET_POLICY_SCHEMA,
                "dataset_id": source_id,
                "required_visibility": "PRIVATE",
                "remote_action_performed_by_stager": False,
                "privacy_enforcement": (
                    "KAGGLE_DATASETS_CREATE_DEFAULT_PRIVATE_AND_OPERATOR_VERIFICATION_REQUIRED"
                ),
            },
        )

        shutil.copyfile(bootstrap_source, kernel_destination / "run.py")
        _write_json(
            kernel_destination / "kernel-metadata.json",
            {
                "id": selected_kernel_id,
                "title": _title_from_id(selected_kernel_id),
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": [*REGISTERED_DATASET_SOURCES, source_id],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
        )
    except Exception:
        for path in reversed(created):
            shutil.rmtree(path, ignore_errors=True)
        raise

    return StagedKaggleTransport(
        source_dataset_directory=source_destination,
        kernel_directory=kernel_destination,
        repository_commit=commit,
        bundle_sha256=_sha256_file(source_destination / BUNDLE_NAME),
        transport_manifest_sha256=_sha256_file(
            source_destination / TRANSPORT_MANIFEST_NAME
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--source-dataset-version", type=int, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--kernel-output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(arguments)
    try:
        staged = stage_kaggle_transport(
            repository_root=args.repository_root,
            source_dataset_id=args.source_dataset_id,
            source_dataset_version=args.source_dataset_version,
            kernel_id=args.kernel_id,
            source_output_dir=args.source_output_dir,
            kernel_output_dir=args.kernel_output_dir,
        )
    except KaggleTransportStagingError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "STAGED_LOCALLY_ONLY",
                "remote_action_performed": False,
                "source_dataset_directory": str(staged.source_dataset_directory),
                "kernel_directory": str(staged.kernel_directory),
                "repository_commit": staged.repository_commit,
                "bundle_sha256": staged.bundle_sha256,
                "transport_manifest_sha256": staged.transport_manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
