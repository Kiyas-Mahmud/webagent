"""Run frozen offline Pillar-2 companion diagnostics.

The bootstrap portion of this command intentionally imports only the Python
standard library. It verifies the manifest-registered full Git commit and a
clean checkout before importing any ``web_agent`` or deployment predictor
source. The predictor factory is a zero-argument ``module:attribute``
callable; its source and returned predictor identity are rechecked after the
diagnostic run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="external root for manifest-relative diagnostic/neutral images",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--predictor-factory",
        required=True,
        help=(
            "exact zero-argument module:attribute factory frozen by the "
            "manifest backend identity"
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"P2 input manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return text


def _require_full_commit(value: object) -> str:
    text = str(value)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise RuntimeError(
            "P2 backend repository_commit must be one full lowercase Git commit"
        )
    return text


def _bootstrap_backend_identity(path: Path) -> dict[str, str]:
    """Read only the source identity needed before project imports occur."""

    source = path.absolute()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise RuntimeError("P2 input manifest is absent or inaccessible") from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("P2 input manifest must be a non-symlink regular file")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("P2 input manifest is not valid unique-key JSON") from exc
    if not isinstance(value, Mapping) or not isinstance(
        value.get("backend_identity"), Mapping
    ):
        raise RuntimeError("P2 input manifest has no backend identity")
    backend = value["backend_identity"]
    entrypoint = str(backend.get("factory_entrypoint", ""))
    module_name, separator, attribute_name = entrypoint.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute_name
        or "." in attribute_name
    ):
        raise RuntimeError("P2 backend factory entrypoint is malformed")
    return {
        "repository_commit": _require_full_commit(backend.get("repository_commit")),
        "factory_entrypoint": entrypoint,
        "factory_source_sha256": _require_sha256(
            backend.get("factory_source_sha256"), label="factory_source_sha256"
        ),
    }


def _git_output(repository_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("P2 diagnostic requires an accessible Git checkout") from exc


def _verify_clean_repository_before_import(
    repository_root: Path,
    *,
    expected_commit: str,
) -> dict[str, str]:
    """Standard-library-only source gate used before project/deployment import."""

    expected = _require_full_commit(expected_commit)
    root = repository_root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("P2 repository root must be a non-symlink directory")
    root = root.resolve()
    top_level = _git_output(root, "rev-parse", "--show-toplevel").strip()
    commit = _git_output(root, "rev-parse", "HEAD").strip()
    status = _git_output(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if Path(top_level).resolve() != root:
        raise RuntimeError("P2 repository root is not the exact Git top level")
    _require_full_commit(commit)
    if commit != expected:
        raise RuntimeError("P2 repository commit differs from input registration")
    if status.strip():
        raise RuntimeError("P2 diagnostic requires a clean Git checkout")
    return {
        "repository_commit": commit,
        "git_status_porcelain_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
    }


def _verify_committed_source_file(
    repository_root: Path,
    source_path: Path,
    *,
    expected_commit: str,
) -> str:
    """Require loaded source to equal bytes tracked by the registered commit."""

    root = repository_root.resolve()
    source = source_path.absolute()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("P2 loaded source must be a non-symlink regular file")
    source = source.resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError("P2 loaded source is outside the attested repository") from exc
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed = subprocess.check_output(
            ["git", "show", f"{_require_full_commit(expected_commit)}:{relative}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "P2 loaded source is not tracked by the registered commit"
        ) from exc
    if tracked != relative or hashlib.sha256(committed).hexdigest() != _sha256_file(
        source
    ):
        raise RuntimeError("P2 loaded source differs from registered commit bytes")
    return relative


def _load_factory(
    entrypoint: str,
    expected: Any,
    *,
    repository_root: Path,
) -> Any:
    if entrypoint != expected.factory_entrypoint:
        raise RuntimeError("P2 predictor factory differs from frozen backend identity")
    module_name, separator, attribute_name = entrypoint.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute_name
        or "." in attribute_name
    ):
        raise RuntimeError("--predictor-factory must be exact module:attribute")
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError("P2 predictor factory module source is unresolved")
    registered_source = Path(spec.origin).absolute()
    if registered_source.is_symlink() or not registered_source.is_file():
        raise RuntimeError("P2 predictor factory module source must be a regular file")
    registered_source = registered_source.resolve()
    _verify_committed_source_file(
        repository_root,
        registered_source,
        expected_commit=expected.repository_commit,
    )
    if _sha256_file(registered_source) != expected.factory_source_sha256:
        raise RuntimeError("P2 predictor factory source hash mismatch before import")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise RuntimeError("P2 predictor factory is not callable")
    if getattr(factory, "__module__", None) != module_name or getattr(
        factory, "__qualname__", None
    ) != attribute_name:
        raise RuntimeError("P2 predictor factory is an alias or nested callable")
    try:
        source_name = inspect.getsourcefile(factory) or inspect.getfile(factory)
    except (OSError, TypeError) as exc:
        raise RuntimeError("P2 predictor factory source is unresolved") from exc
    if not source_name:
        raise RuntimeError("P2 predictor factory source is unresolved")
    source = Path(source_name).absolute()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("P2 predictor factory source must be a regular file")
    if source.resolve() != registered_source or (
        _sha256_file(source.resolve()) != expected.factory_source_sha256
    ):
        raise RuntimeError("P2 predictor factory source hash mismatch")
    return factory


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    bootstrap = _bootstrap_backend_identity(args.input_manifest)
    if args.predictor_factory != bootstrap["factory_entrypoint"]:
        raise RuntimeError("P2 predictor factory differs from bootstrap registration")

    # This gate precedes every web_agent and deployment predictor import.
    starting_git = _verify_clean_repository_before_import(
        repository_root,
        expected_commit=bootstrap["repository_commit"],
    )

    diagnostics = importlib.import_module(
        "web_agent.eval.table2.pillar2_diagnostics"
    )
    diagnostic_source_name = inspect.getsourcefile(diagnostics)
    if not diagnostic_source_name:
        raise RuntimeError("P2 diagnostic runner source is unresolved")
    _verify_committed_source_file(
        repository_root,
        Path(diagnostic_source_name),
        expected_commit=bootstrap["repository_commit"],
    )
    diagnostic = diagnostics.Pillar2DiagnosticInput.load(args.input_manifest)
    if (
        diagnostic.backend_identity.repository_commit
        != bootstrap["repository_commit"]
        or diagnostic.backend_identity.factory_entrypoint
        != bootstrap["factory_entrypoint"]
        or diagnostic.backend_identity.factory_source_sha256
        != bootstrap["factory_source_sha256"]
    ):
        raise RuntimeError("typed P2 identity differs from bootstrap identity")

    # This second check is immediately before deployment predictor import.
    if _verify_clean_repository_before_import(
        repository_root,
        expected_commit=bootstrap["repository_commit"],
    ) != starting_git:
        raise RuntimeError("P2 repository identity changed during schema import")
    factory = _load_factory(
        args.predictor_factory,
        diagnostic.backend_identity,
        repository_root=repository_root,
    )
    predictor = factory()
    report = diagnostics.run_pillar2_diagnostics(
        diagnostic,
        image_root=args.image_root,
        predictor=predictor,
        repository_root=repository_root,
    )

    # Re-resolve and rehash the registered factory source after inference. The
    # runner separately rehashes the exact predictor class source and images.
    _load_factory(
        args.predictor_factory,
        diagnostic.backend_identity,
        repository_root=repository_root,
    )
    _verify_committed_source_file(
        repository_root,
        Path(diagnostic_source_name),
        expected_commit=bootstrap["repository_commit"],
    )
    diagnostics.validate_pillar2_diagnostic_report(report)
    package = diagnostics.write_pillar2_diagnostic_package(
        args.output_dir,
        diagnostic=diagnostic,
        report=report,
    )
    validation = diagnostics.validate_pillar2_diagnostic_package(args.output_dir)
    ending_git = _verify_clean_repository_before_import(
        repository_root,
        expected_commit=bootstrap["repository_commit"],
    )
    if ending_git != starting_git:
        raise RuntimeError("P2 repository identity changed during diagnostic run")
    package["validation"] = validation
    print(json.dumps(package, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
