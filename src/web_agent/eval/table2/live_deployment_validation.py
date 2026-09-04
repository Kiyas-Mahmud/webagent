"""Validate measured Table 2 live-deployment evidence by ephemeral staging.

The validator reuses the production staging and binding path inside an
ephemeral directory and does not mutate the supplied manifest or evidence
tree.  It returns identities and counts only; its output is not a readiness
receipt and cannot authorize handoff, campaign dispatch, or a paper result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Sequence

from .common import SchemaError, sha256_json
from .live_deployment import stage_pc01_live_deployment_package


SUMMARY_SCHEMA_VERSION = "table2-live-deployment-validation-summary-v1"


def _absolute_external_source(
    value: str | Path,
    *,
    repository_root: Path,
    field: str,
    require_directory: bool,
) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise SchemaError(f"{field} must be an absolute external path")
    current = supplied
    while True:
        if current.is_symlink():
            raise SchemaError(f"{field} must not use symlink ancestry")
        if current.parent == current:
            break
        current = current.parent
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise SchemaError(f"{field} is absent or inaccessible") from exc
    expected_kind = resolved.is_dir() if require_directory else resolved.is_file()
    if not expected_kind:
        kind = "directory" if require_directory else "file"
        raise SchemaError(f"{field} must be an existing {kind}")
    if (
        resolved == repository_root
        or repository_root in resolved.parents
        or resolved in repository_root.parents
    ):
        raise SchemaError(f"{field} must be outside the repository tree")
    return resolved


def _require_safe_temporary_parent(
    *,
    repository_root: Path,
    manifest_path: Path,
    evidence_root: Path,
) -> None:
    try:
        temporary_parent = Path(gettempdir()).resolve(strict=True)
    except OSError as exc:
        raise SchemaError("system temporary directory is inaccessible") from exc
    for authority_tree, label in (
        (repository_root, "repository"),
        (evidence_root, "evidence root"),
        (manifest_path.parent, "manifest directory"),
    ):
        if (
            temporary_parent == authority_tree
            or authority_tree in temporary_parent.parents
        ):
            raise SchemaError(
                "system temporary directory must remain outside the "
                f"{label} authority tree"
            )


def build_validation_summary(
    *,
    repository_root: str | Path,
    manifest_path: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Reproduce a package's identities without retaining a staged copy."""

    try:
        repository = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise SchemaError("repository_root is absent or inaccessible") from exc
    if not repository.is_dir():
        raise SchemaError("repository_root must be an existing directory")
    manifest = _absolute_external_source(
        manifest_path,
        repository_root=repository,
        field="manifest",
        require_directory=False,
    )
    evidence = _absolute_external_source(
        evidence_root,
        repository_root=repository,
        field="evidence_root",
        require_directory=True,
    )
    _require_safe_temporary_parent(
        repository_root=repository,
        manifest_path=manifest,
        evidence_root=evidence,
    )
    with TemporaryDirectory(prefix="table2-live-deployment-validation-") as temp:
        validated = stage_pc01_live_deployment_package(
            manifest_path=manifest,
            evidence_root=evidence,
            repository_root=repository,
            destination_artifact_root=Path(temp),
        )
        binding = validated.binding
        requirements = validated.evaluator_requirements
        capability_ids = sorted(validated.manifest["capabilities"])
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "scientific_role": "VALIDATION_SUMMARY_ONLY_NOT_DISPATCH_AUTHORITY",
            "paper_table_status": "N/R",
            "dispatch_authorized": False,
            "handoff_authorized": False,
            "cross_binding_performed": False,
            "manifest_sha256": binding["manifest_sha256"],
            "manifest_content_sha256": binding["manifest_content_sha256"],
            "package_file_count": len(binding["package_files"]),
            "package_file_set_sha256": binding["package_file_set_sha256"],
            "capability_source_file_count": len(
                binding["capability_source_files"]
            ),
            "capability_source_set_sha256": binding[
                "capability_source_set_sha256"
            ],
            "capability_id_count": len(capability_ids),
            "capability_id_set_sha256": sha256_json(capability_ids),
            "evaluator_requirements_content_sha256": sha256_json(requirements),
            "evaluator_requirement_task_count": requirements["task_count"],
            "judge_required_task_count": requirements["judge_required"][
                "task_count"
            ],
            "string_match_task_count": requirements["string_match"][
                "task_count"
            ],
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_validation_summary(
        repository_root=args.repository_root,
        manifest_path=args.manifest,
        evidence_root=args.evidence_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
