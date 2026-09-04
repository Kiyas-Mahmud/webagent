"""No-argument, fail-closed Kaggle bootstrap for Table 2 P4 preparation.

This file intentionally uses only the Python standard library until it has
materialized and authenticated the pinned Git bundle. The bundle is transport
only: the prepare-only runner separately authenticates the cloned clean Git
checkout and hashes its executed source set.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


TRANSPORT_MANIFEST_NAME = "table2-p4-git-bundle-transport-v1.json"
TRANSPORT_MANIFEST_SCHEMA = "table2-p4-git-bundle-transport-v1"
TRANSPORT_EVIDENCE_ROLE = (
    "GIT_BUNDLE_MATERIALIZED_TRANSPORT_NOT_SCIENTIFIC_SOURCE_AUTHORITY"
)
SOURCE_DATASET_SLUG = "table2-p4-source-transport-v1"
CHECKOUT_RELATIVE_PATH = "webagent-table2-p4-source-v1"
OUTPUT_RELATIVE_PATH = "table2-p4-prepare-only-v1"
PREPARE_CONFIG_RELATIVE_PATH = (
    "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
)
RUNNER_RELATIVE_PATH = "scripts/run_table2_p4_kaggle_prepare_only.py"
BOOTSTRAP_RELATIVE_PATH = "kaggle/table2_p4_prepare_only/run.py"
DEFAULT_INPUT_ROOT = Path("/kaggle/input")
DEFAULT_WORKING_ROOT = Path("/kaggle/working")
_KAGGLE_ID_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]{2,99}\Z"
)
class BootstrapError(ValueError):
    """The source transport or no-argument bootstrap contract failed."""


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BootstrapError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, role: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"{role} must be a regular non-symlink file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid {role}: {error}") from error
    if not isinstance(payload, dict):
        raise BootstrapError(f"{role} must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: object, *, field: str, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BootstrapError(f"{field} must be a lowercase {length}-character hex digest")
    return value


def _validate_kaggle_id(value: object, *, field: str) -> str:
    if type(value) is not str or _KAGGLE_ID_RE.fullmatch(value) is None:
        raise BootstrapError(f"{field} must be an exact owner/slug Kaggle ID")
    lowered = value.casefold()
    if any(token in lowered for token in ("placeholder", "replace_with", "your_")):
        raise BootstrapError(f"{field} contains a placeholder")
    return value


def _safe_relative(value: object, *, field: str) -> PurePosixPath:
    if type(value) is not str:
        raise BootstrapError(f"{field} must be a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BootstrapError(f"{field} must be a safe relative path")
    return path


def _lexical_child(root: Path, relative: PurePosixPath) -> Path:
    return root.joinpath(*relative.parts)


def _assert_regular_symlink_free(root: Path, candidate: Path, *, role: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise BootstrapError(f"{role} escapes its declared root") from error
    current = root
    if current.is_symlink() or not current.is_dir():
        raise BootstrapError(f"{role} root must be a regular non-symlink directory")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise BootstrapError(f"{role} contains a symlink component: {current}")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise BootstrapError(f"{role} must be a regular file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise BootstrapError(f"{role} must be a regular file")
    if metadata.st_nlink != 1:
        raise BootstrapError(f"{role} must not be hard-linked")


def _assert_tree_has_no_symlinks(root: Path, *, role: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise BootstrapError(f"{role} must be a regular non-symlink directory")
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in [*directories, *files]:
            if (base / name).is_symlink():
                raise BootstrapError(f"{role} contains a symlink: {base / name}")


def _resolve_source_mount(input_root: Path, dataset_id: str) -> Path:
    if input_root.is_symlink() or not input_root.is_dir():
        raise BootstrapError("Kaggle input root must be a regular non-symlink directory")
    owner, slug = dataset_id.split("/", 1)
    permitted = (
        input_root / slug,
        input_root / "datasets" / owner / slug,
    )
    existing: list[Path] = []
    for candidate in permitted:
        current = input_root
        for part in candidate.relative_to(input_root).parts:
            current = current / part
            if current.is_symlink():
                raise BootstrapError("source-dataset mount contains a symlink component")
        if candidate.is_dir():
            existing.append(candidate)
    if len(existing) != 1:
        raise BootstrapError(
            "source dataset requires exactly one permitted Kaggle mount; "
            f"found {len(existing)}"
        )
    mount = existing[0]
    _assert_tree_has_no_symlinks(mount, role="source-dataset mount")
    return mount


def _discover_transport_manifest(input_root: Path) -> Path:
    """Find one manifest at the root of a supported Kaggle dataset mount.

    The bounded walk intentionally does not traverse the 22 GB Gold datasets.
    It checks only ``input/<slug>`` and ``input/datasets/<owner>/<slug>`` roots.
    """

    if input_root.is_symlink() or not input_root.is_dir():
        raise BootstrapError("Kaggle input root must be a regular non-symlink directory")
    manifests: list[Path] = []
    for child in sorted(input_root.iterdir(), key=lambda path: path.name):
        if child.name == "datasets":
            continue
        candidate = child / TRANSPORT_MANIFEST_NAME
        if candidate.exists() or candidate.is_symlink():
            manifests.append(candidate)
    datasets_root = input_root / "datasets"
    if datasets_root.exists() or datasets_root.is_symlink():
        if datasets_root.is_symlink() or not datasets_root.is_dir():
            raise BootstrapError("Kaggle datasets root must be a non-symlink directory")
        for owner in sorted(datasets_root.iterdir(), key=lambda path: path.name):
            if owner.is_symlink() or not owner.is_dir():
                continue
            for dataset in sorted(owner.iterdir(), key=lambda path: path.name):
                candidate = dataset / TRANSPORT_MANIFEST_NAME
                if candidate.exists() or candidate.is_symlink():
                    manifests.append(candidate)
    if len(manifests) != 1:
        raise BootstrapError(
            "Kaggle input requires exactly one versioned transport manifest; "
            f"found {len(manifests)}"
        )
    _assert_regular_symlink_free(
        input_root, manifests[0], role="discovered transport manifest"
    )
    return manifests[0]


def _discover_sole_bundle(mount: Path) -> Path:
    bundles: list[Path] = []
    for current, directories, files in os.walk(mount, followlinks=False):
        directories[:] = sorted(directories)
        base = Path(current)
        for name in sorted(files):
            path = base / name
            if path.suffix == ".bundle":
                bundles.append(path)
    if len(bundles) != 1:
        raise BootstrapError(
            "source dataset requires exactly one Git bundle; "
            f"found {len(bundles)}"
        )
    return bundles[0]


def _run_git(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    git_executable = shutil.which("git", path=os.defpath)
    if git_executable is None:
        raise BootstrapError("system-default Git executable is unavailable")
    git_path = Path(git_executable).resolve()
    if not git_path.is_file() or not os.access(git_path, os.X_OK):
        raise BootstrapError("system-default Git executable is not a regular executable")
    environment = _safe_git_environment()
    try:
        return subprocess.run(
            [str(git_path), *arguments],
            cwd=None if cwd is None else str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BootstrapError(f"unable to execute Git safely: {error}") from error


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


def _require_git_success(
    result: subprocess.CompletedProcess[str], *, operation: str
) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BootstrapError(f"{operation} failed: {detail[:500]}")
    return result.stdout.strip()


def _verify_bundle(bundle: Path, commit: str, *, working_root: Path) -> None:
    heads = _require_git_success(
        _run_git(["bundle", "list-heads", str(bundle)]),
        operation="git bundle list-heads",
    ).splitlines()
    if heads != [f"{commit} HEAD"]:
        raise BootstrapError("Git bundle must advertise exactly the registered commit as HEAD")
    verifier = Path(
        tempfile.mkdtemp(prefix=".table2-bundle-verify-", dir=working_root)
    )
    try:
        _require_git_success(
            _run_git(["init", "--bare", "-q", str(verifier)]),
            operation="temporary verifier initialization",
        )
        _require_git_success(
            _run_git(["-C", str(verifier), "bundle", "verify", str(bundle)]),
            operation="git bundle verify",
        )
    finally:
        shutil.rmtree(verifier, ignore_errors=True)


def _load_transport_manifest(
    *,
    manifest_path: Path,
    mount: Path,
    expected_dataset_id: str,
    working_root: Path,
) -> tuple[Mapping[str, Any], Path]:
    _assert_regular_symlink_free(mount, manifest_path, role="transport manifest")
    discovered_manifests = sorted(
        (
            path
            for path in mount.rglob(TRANSPORT_MANIFEST_NAME)
            if path.is_file() or path.is_symlink()
        ),
        key=str,
    )
    if discovered_manifests != [manifest_path]:
        raise BootstrapError(
            "source dataset requires exactly one versioned transport manifest"
        )
    manifest = _load_json(manifest_path, role="transport manifest")
    required = {
        "schema_version",
        "evidence_role",
        "source_dataset_id",
        "source_dataset_version",
        "repository_commit",
        "bundle",
        "bootstrap",
    }
    if set(manifest) != required:
        raise BootstrapError("transport manifest fields differ from the exact schema")
    if manifest.get("schema_version") != TRANSPORT_MANIFEST_SCHEMA:
        raise BootstrapError("transport manifest schema version is not registered")
    if manifest.get("evidence_role") != TRANSPORT_EVIDENCE_ROLE:
        raise BootstrapError("transport manifest improperly claims source authority")
    _validate_kaggle_id(manifest.get("source_dataset_id"), field="source_dataset_id")
    if manifest["source_dataset_id"] != expected_dataset_id:
        raise BootstrapError("transport manifest changed during authenticated discovery")
    if manifest["source_dataset_id"].split("/", 1)[1] != SOURCE_DATASET_SLUG:
        raise BootstrapError("source transport dataset slug is not registered")
    if manifest.get("source_dataset_version") != 1:
        raise BootstrapError("source_dataset_version must be exactly 1 for schema v1")
    if manifest["source_dataset_id"] in {
        "kiyasmahmud/web-gold-40k",
        "kiyasmahmud/gold-40k-retry",
    }:
        raise BootstrapError("source transport must not impersonate a Gold dataset")
    commit = _valid_sha(
        manifest.get("repository_commit"), field="repository_commit", length=40
    )
    bundle_row = manifest.get("bundle")
    if not isinstance(bundle_row, dict) or set(bundle_row) != {
        "relative_path",
        "bytes",
        "sha256",
        "format",
    }:
        raise BootstrapError("transport bundle descriptor is malformed")
    if bundle_row.get("format") != "git_bundle":
        raise BootstrapError("transport format must be git_bundle")
    bundle_relative = _safe_relative(
        bundle_row.get("relative_path"), field="bundle.relative_path"
    )
    bundle = _lexical_child(mount, bundle_relative)
    _assert_regular_symlink_free(mount, bundle, role="Git bundle")
    if type(bundle_row.get("bytes")) is not int or bundle_row["bytes"] < 1:
        raise BootstrapError("bundle.bytes must be a positive integer")
    if bundle.stat().st_size != bundle_row["bytes"]:
        raise BootstrapError("Git bundle byte count mismatch")
    expected_hash = _valid_sha(
        bundle_row.get("sha256"), field="bundle.sha256", length=64
    )
    if _sha256_file(bundle) != expected_hash:
        raise BootstrapError("Git bundle SHA-256 mismatch")
    if _discover_sole_bundle(mount) != bundle:
        raise BootstrapError("declared bundle differs from the sole discovered Git bundle")
    bootstrap_row = manifest.get("bootstrap")
    if not isinstance(bootstrap_row, dict) or set(bootstrap_row) != {
        "repository_relative_path",
        "sha256",
    }:
        raise BootstrapError("transport bootstrap descriptor is malformed")
    if bootstrap_row.get("repository_relative_path") != BOOTSTRAP_RELATIVE_PATH:
        raise BootstrapError("transport bootstrap path differs from the registered path")
    _valid_sha(bootstrap_row.get("sha256"), field="bootstrap.sha256", length=64)
    _verify_bundle(bundle, commit, working_root=working_root)
    return manifest, bundle


def _verify_materialized_checkout(checkout: Path, commit: str) -> None:
    inside = _require_git_success(
        _run_git(["-C", str(checkout), "rev-parse", "--is-inside-work-tree"]),
        operation="checkout repository verification",
    )
    if inside != "true":
        raise BootstrapError("materialized source is not a Git worktree")
    head = _require_git_success(
        _run_git(["-C", str(checkout), "rev-parse", "HEAD"]),
        operation="checkout HEAD verification",
    ).lower()
    if head != commit:
        raise BootstrapError("materialized checkout HEAD differs from transport commit")
    detached = _run_git(["-C", str(checkout), "symbolic-ref", "-q", "HEAD"])
    if detached.returncode == 0:
        raise BootstrapError("materialized checkout HEAD must be detached")
    if detached.returncode not in (1,):
        raise BootstrapError("unable to verify detached checkout HEAD")
    status = _require_git_success(
        _run_git(
            [
                "-C",
                str(checkout),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ]
        ),
        operation="checkout cleanliness verification",
    )
    if status:
        raise BootstrapError("materialized checkout is dirty")


def build_prepare_command(
    *,
    input_root: Path,
    working_root: Path,
    script_path: Path,
) -> list[str]:
    """Materialize the sole pinned bundle and return the exact runner argv."""

    input_root = Path(os.path.abspath(os.fspath(input_root)))
    working_root = Path(os.path.abspath(os.fspath(working_root)))
    script_path = Path(os.path.abspath(os.fspath(script_path)))
    script_dir = script_path.parent
    _assert_regular_symlink_free(script_dir, script_path, role="executing bootstrap")
    if working_root.is_symlink() or not working_root.is_dir():
        raise BootstrapError("Kaggle working root must be a regular non-symlink directory")
    manifest_path = _discover_transport_manifest(input_root)
    preliminary = _load_json(manifest_path, role="transport manifest")
    source_dataset_id = _validate_kaggle_id(
        preliminary.get("source_dataset_id"), field="source_dataset_id"
    )
    mount = _resolve_source_mount(input_root, source_dataset_id)
    if manifest_path.parent != mount:
        raise BootstrapError(
            "transport manifest is not at its declared source-dataset mount root"
        )
    manifest, bundle = _load_transport_manifest(
        manifest_path=manifest_path,
        mount=mount,
        expected_dataset_id=source_dataset_id,
        working_root=working_root,
    )
    bootstrap_sha = _sha256_file(script_path)
    if bootstrap_sha != manifest["bootstrap"]["sha256"]:
        raise BootstrapError("executing bootstrap differs from the transport manifest")

    checkout = working_root / CHECKOUT_RELATIVE_PATH
    output = working_root / OUTPUT_RELATIVE_PATH
    for path, role in ((checkout, "checkout"), (output, "prepare-only output")):
        if path.exists() or path.is_symlink():
            raise BootstrapError(f"refusing to overwrite existing {role}: {path}")

    commit = str(manifest["repository_commit"])
    _require_git_success(
        _run_git(
            [
                "clone",
                "--no-checkout",
                "--no-tags",
                "--no-local",
                "--",
                str(bundle),
                str(checkout),
            ],
            timeout=300,
        ),
        operation="Git bundle clone",
    )
    _require_git_success(
        _run_git(["-C", str(checkout), "checkout", "--detach", commit]),
        operation="detached exact-commit checkout",
    )
    _verify_materialized_checkout(checkout, commit)

    cloned_bootstrap = checkout / BOOTSTRAP_RELATIVE_PATH
    if cloned_bootstrap.is_symlink() or not cloned_bootstrap.is_file():
        raise BootstrapError("cloned commit lacks the tracked registered bootstrap")
    if _sha256_file(cloned_bootstrap) != bootstrap_sha:
        raise BootstrapError("executing bootstrap differs from cloned committed bootstrap")
    runner = checkout / RUNNER_RELATIVE_PATH
    config_path = checkout / PREPARE_CONFIG_RELATIVE_PATH
    for path, role in ((runner, "prepare-only runner"), (config_path, "prepare config")):
        if path.is_symlink() or not path.is_file():
            raise BootstrapError(f"cloned commit lacks the registered {role}")

    return [
        sys.executable,
        "-B",
        str(runner),
        "--repository-root",
        str(checkout),
        "--config",
        str(config_path),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output),
        "--source-commit",
        commit,
        "--source-bundle",
        str(bundle),
    ]


def run_no_argument_bootstrap(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    working_root: Path = DEFAULT_WORKING_ROOT,
    script_path: Path | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    """Run the registered prepare-only command after secure materialization."""

    actual_script = Path(__file__).absolute() if script_path is None else script_path
    normalized_input = Path(os.path.abspath(os.fspath(input_root)))
    normalized_working = Path(os.path.abspath(os.fspath(working_root)))
    command = build_prepare_command(
        input_root=normalized_input,
        working_root=normalized_working,
        script_path=actual_script,
    )
    # This registered stage is CPU-only JSON/provenance auditing. It needs no
    # Kaggle API credential, accelerator, inherited loader path, or user cache.
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(normalized_working / CHECKOUT_RELATIVE_PATH / "src"),
        "TZ": "UTC",
    }
    try:
        result = command_runner(
            command,
            cwd=str(normalized_working / CHECKOUT_RELATIVE_PATH),
            env=environment,
            check=False,
        )
    except OSError as error:
        raise BootstrapError(
            f"unable to launch authenticated prepare-only runner: {error}"
        ) from error
    return int(result.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": "registered Kaggle bootstrap accepts no arguments",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        return run_no_argument_bootstrap()
    except BootstrapError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
