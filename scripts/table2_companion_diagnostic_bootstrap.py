"""Standard-library bootstrap for source-attested companion diagnostics.

This module intentionally contains no ``web_agent`` import.  The exact clean
Git commit and factory source are authenticated before the selected diagnostic
schema, model stack, or deployment factory is imported.
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


_PILLARS = {
    "P1": {
        "module": "web_agent.eval.table2.pillar1_diagnostics",
        "input_class": "Pillar1DiagnosticInput",
        "run": "run_pillar1_diagnostics",
        "validate_report": "validate_pillar1_diagnostic_report",
        "write": "write_pillar1_diagnostic_package",
        "validate_package": "validate_pillar1_diagnostic_package",
    },
    "P3": {
        "module": "web_agent.eval.table2.pillar3_diagnostics",
        "input_class": "Pillar3DiagnosticInput",
        "run": "run_pillar3_diagnostics",
        "validate_report": "validate_pillar3_diagnostic_report",
        "write": "write_pillar3_diagnostic_package",
        "validate_package": "validate_pillar3_diagnostic_package",
    },
    "P4": {
        "module": "web_agent.eval.table2.pillar4_diagnostics",
        "input_class": "Pillar4DiagnosticInput",
        "run": "run_pillar4_diagnostics",
        "validate_report": "validate_pillar4_diagnostic_report",
        "write": "write_pillar4_diagnostic_package",
        "validate_package": "validate_pillar4_diagnostic_package",
    },
}


def _parse_args(pillar: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run source-attested offline {pillar} companion diagnostics."
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument(
        "--evidence-root",
        required=True,
        type=Path,
        help="frozen image root for P1/P3, or frozen memory-store root for P4",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--predictor-factory",
        required=True,
        help="exact zero-argument module:attribute factory registered by the input",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"companion input repeats JSON key {key!r}")
        output[key] = value
    return output


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return text


def _require_commit(value: object) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError("companion repository_commit must be one full Git commit")
    return text


def _bootstrap_identity(path: Path, *, pillar: str) -> dict[str, str]:
    source = path.absolute()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise RuntimeError("companion input manifest is absent") from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("companion input must be a non-symlink regular file")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("companion input is not valid unique-key JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        f"table2.pillar{pillar[1:]}-diagnostic-input.v1"
    ) or not isinstance(value.get("backend_identity"), Mapping):
        raise RuntimeError("companion input schema/backend identity is invalid")
    backend = value["backend_identity"]
    if backend.get("pillar") != pillar:
        raise RuntimeError("companion input/backend pillar mismatch")
    entrypoint = str(backend.get("factory_entrypoint", ""))
    module, separator, attribute = entrypoint.partition(":")
    if separator != ":" or not module or not attribute or "." in attribute:
        raise RuntimeError("companion predictor factory entrypoint is malformed")
    return {
        "repository_commit": _require_commit(backend.get("repository_commit")),
        "factory_entrypoint": entrypoint,
        "factory_source_sha256": _require_sha256(
            backend.get("factory_source_sha256"), label="factory_source_sha256"
        ),
    }


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("companion diagnostic requires a Git checkout") from exc


def _verify_clean_repository(root: Path, *, expected_commit: str) -> dict[str, str]:
    expected = _require_commit(expected_commit)
    raw = root.absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError("companion repository root must be a non-symlink directory")
    repository = raw.resolve()
    top = Path(_git_output(repository, "rev-parse", "--show-toplevel").strip())
    commit = _git_output(repository, "rev-parse", "HEAD").strip()
    status = _git_output(
        repository, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if top.resolve() != repository or commit != expected or status.strip():
        raise RuntimeError(
            "companion diagnostic requires the exact clean registered Git commit"
        )
    return {
        "repository_commit": commit,
        "git_status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _verify_committed_source(
    root: Path, source: Path, *, expected_commit: str
) -> str:
    repository = root.resolve()
    candidate = source.absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("companion loaded source must be a regular file")
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(repository).as_posix()
        tracked = subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed = subprocess.check_output(
            ["git", "show", f"{_require_commit(expected_commit)}:{relative}"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("companion loaded source is not in the registered commit") from exc
    if tracked != relative or hashlib.sha256(committed).hexdigest() != _sha256_file(candidate):
        raise RuntimeError("companion loaded source differs from registered commit bytes")
    return relative


def _load_factory(entrypoint: str, identity: Any, *, root: Path) -> Any:
    if entrypoint != identity.factory_entrypoint:
        raise RuntimeError("companion factory differs from typed registration")
    module_name, _, attribute_name = entrypoint.partition(":")
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise RuntimeError("companion predictor factory source is unresolved")
    source = Path(spec.origin).resolve()
    _verify_committed_source(
        root, source, expected_commit=identity.repository_commit
    )
    if _sha256_file(source) != identity.factory_source_sha256:
        raise RuntimeError("companion predictor factory source hash mismatch")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory) or getattr(factory, "__module__", None) != module_name or (
        getattr(factory, "__qualname__", None) != attribute_name
    ):
        raise RuntimeError("companion predictor factory is aliased, nested, or non-callable")
    source_name = inspect.getsourcefile(factory) or inspect.getfile(factory)
    if not source_name or Path(source_name).resolve() != source:
        raise RuntimeError("companion predictor factory source changed after import")
    return factory


def main(pillar: str) -> None:
    if pillar not in _PILLARS:
        raise RuntimeError("companion diagnostic pillar is unregistered")
    args = _parse_args(pillar)
    repository_root = Path(__file__).resolve().parents[1]
    bootstrap = _bootstrap_identity(args.input_manifest, pillar=pillar)
    if args.predictor_factory != bootstrap["factory_entrypoint"]:
        raise RuntimeError("companion factory differs from bootstrap registration")

    # This is deliberately before every web_agent/deployment import.
    starting_git = _verify_clean_repository(
        repository_root, expected_commit=bootstrap["repository_commit"]
    )
    registration = _PILLARS[pillar]
    diagnostics = importlib.import_module(registration["module"])
    diagnostic_source = Path(inspect.getsourcefile(diagnostics) or "")
    if not diagnostic_source:
        raise RuntimeError("companion diagnostic source is unresolved")
    _verify_committed_source(
        repository_root,
        diagnostic_source,
        expected_commit=bootstrap["repository_commit"],
    )
    diagnostic = getattr(diagnostics, registration["input_class"]).load(
        args.input_manifest
    )
    identity = diagnostic.backend_identity
    if (
        identity.repository_commit != bootstrap["repository_commit"]
        or identity.factory_entrypoint != bootstrap["factory_entrypoint"]
        or identity.factory_source_sha256 != bootstrap["factory_source_sha256"]
    ):
        raise RuntimeError("typed companion identity differs from bootstrap identity")
    if _verify_clean_repository(
        repository_root, expected_commit=bootstrap["repository_commit"]
    ) != starting_git:
        raise RuntimeError("repository identity changed during companion schema import")

    factory = _load_factory(args.predictor_factory, identity, root=repository_root)
    predictor = factory()
    report = getattr(diagnostics, registration["run"])(
        diagnostic,
        evidence_root=args.evidence_root,
        predictor=predictor,
        repository_root=repository_root,
    )

    # Re-resolve all executable sources and the exact repository after inference.
    _load_factory(args.predictor_factory, identity, root=repository_root)
    _verify_committed_source(
        repository_root,
        diagnostic_source,
        expected_commit=bootstrap["repository_commit"],
    )
    getattr(diagnostics, registration["validate_report"])(report)
    package = getattr(diagnostics, registration["write"])(
        args.output_dir, diagnostic=diagnostic, report=report
    )
    package["validation"] = getattr(diagnostics, registration["validate_package"])(
        args.output_dir
    )
    ending_git = _verify_clean_repository(
        repository_root, expected_commit=bootstrap["repository_commit"]
    )
    if ending_git != starting_git:
        raise RuntimeError("repository identity changed during companion diagnostics")
    print(json.dumps(package, indent=2, sort_keys=True))
