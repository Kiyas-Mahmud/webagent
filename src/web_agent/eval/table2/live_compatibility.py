"""Non-scored live WebArena readiness gate for the provisional PC-01 pilot.

The 260-episode development pilot may start only after the exact frozen runtime
has completed one isolated, matched E0--E3 WebArena block.  The probe runs in a
separate campaign directory so its outcomes can never be aggregated with the
pilot.  Receipt issuance then seals the minimal replayable probe evidence into
the target package; validation never depends on the external probe directory.

This module is deliberately outside the seven model-evidence producer roles.
It is runtime-readiness evidence, not model-selection evidence and never paper
Table 2 evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import stat
from typing import Any

from .common import (
    SchemaError,
    Table2Error,
    atomic_write_json,
    read_json,
    read_jsonl,
    safe_relative_path,
    sha256_file,
    sha256_json,
)
from .evidence_validation import validate_included_block_causal_trace
from .schedule import (
    SYSTEM_IDS,
    discover_block_attempt_directories,
    resolve_block_attempts,
    validate_schedule,
)


LIVE_COMPATIBILITY_SCHEMA_VERSION = "table2-live-matched-block-readiness-v2"
LIVE_COMPATIBILITY_RECORD_TYPE = "PC01LiveMatchedE0E3ReadinessReceipt"
LIVE_COMPATIBILITY_EVIDENCE_SCOPE = "PC01_PROVISIONAL_PILOT_ONLY"
LIVE_COMPATIBILITY_EVIDENCE_ROLE = "NON_SCORED_RUNTIME_READINESS_ONLY"
PROBE_EVIDENCE_SCHEMA_VERSION = "table2-live-probe-evidence-v1"
PROBE_EVIDENCE_RECORD_TYPE = "PC01PortableLiveProbeEvidence"
LIVE_COMPATIBILITY_RELATIVE_PATH = Path(
    "runtime_readiness/live_matched_e0_e3_receipt.json"
)
LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH = Path(
    "runtime_readiness/live_matched_e0_e3_receipt.sha256"
)
LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH = Path(
    "runtime_readiness/probe_evidence"
)
PROBE_EVIDENCE_MANIFEST_NAME = "probe_evidence_manifest.json"
_PROBE_EVIDENCE_FIXED_FILES = (
    "campaign_manifest.json",
    "artifact_hashes.json",
    "schedule/schedule.jsonl",
    "access_ledger.jsonl",
    "deviation_ledger.jsonl",
    "frozen/protocol.yaml",
)
LIVE_COMPATIBILITY_SOURCE_RELATIVE_PATH = (
    "src/web_agent/eval/table2/live_compatibility.py"
)
CAMPAIGN_SOURCE_RELATIVE_PATH = "src/web_agent/eval/table2/campaign.py"
EVALUATION_CLI_SOURCE_RELATIVE_PATH = "scripts/run_table2_evaluation.py"
EVIDENCE_VALIDATION_SOURCE_RELATIVE_PATH = (
    "src/web_agent/eval/table2/evidence_validation.py"
)
COMMON_SOURCE_RELATIVE_PATH = "src/web_agent/eval/table2/common.py"
SCHEDULE_SOURCE_RELATIVE_PATH = "src/web_agent/eval/table2/schedule.py"
PACKAGE_VALIDATOR_SOURCE_RELATIVE_PATH = (
    "src/web_agent/eval/table2/package_validator.py"
)

_EXPECTED_PILOT = {
    "campaign_kind": "engineering_pilot",
    "campaign_mode": "evaluation",
    "evidence_label": "PILOT_ONLY",
    "publication_status": "PILOT_ONLY",
    "paper_table_status": "N/R",
    "matched_seeds": [42],
    "normal_block_count": 50,
    "recovery_block_count": 15,
    "scheduled_block_count": 65,
    "planned_episode_count": 260,
    "manual_rescue": "forbidden",
}
_SHARED_MANIFEST_FIELDS = (
    "repository_commit",
    "protocol_id",
    "campaign_seed",
    "repeat_ids",
    "max_block_attempts",
    "runner_identity_scope",
    "runner_entrypoint",
    "runner_attestation_sha256",
    "runtime_integration_entrypoint",
    "runtime_integration",
    "selection_mode",
    "checkpoint_selection_evidence_sha256",
    "pc01_checkpoint_compatibility_receipt_sha256",
    "pc01_checkpoint_compatibility",
    "task_registry_sha256",
    "resolved_task_snapshot_sha256",
    "webarena_deployment_topology",
    "webarena_preflight_artifact_sha256",
    "webarena_preflight_content_sha256",
    "webarena_service_url_map_sha256",
    "webarena_expected_dgx_model_runtime_identity_sha256",
    "webarena_expected_bridge_identity_sha256",
    "webarena_deployment_preflight",
    "pc01_live_deployment",
    "pilot_task_exclusion_registry_relative_path",
    "pilot_task_exclusion_registry_sha256",
    "pilot_task_exclusion_identity_version",
    "model_payloads_by_seed",
    "model_evidence_by_seed",
    "joint_duplicate_evidence_binding",
)
LIVE_COMPATIBILITY_REQUIRED_SOURCE_RELATIVE_PATHS = (
    LIVE_COMPATIBILITY_SOURCE_RELATIVE_PATH,
    CAMPAIGN_SOURCE_RELATIVE_PATH,
    EVALUATION_CLI_SOURCE_RELATIVE_PATH,
    EVIDENCE_VALIDATION_SOURCE_RELATIVE_PATH,
    COMMON_SOURCE_RELATIVE_PATH,
    SCHEDULE_SOURCE_RELATIVE_PATH,
    PACKAGE_VALIDATOR_SOURCE_RELATIVE_PATH,
)


def campaign_requires_live_compatibility(manifest: Mapping[str, Any]) -> bool:
    """Return whether a campaign is the real provisional 260-episode pilot."""

    return (
        manifest.get("campaign_mode") == "evaluation"
        and manifest.get("evidence_label") == "PILOT_ONLY"
    )


def _require_regular_directory(path: str | Path, *, field: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise SchemaError(f"{field} must not be a symlink")
    resolved = supplied.resolve()
    if not resolved.is_dir():
        raise SchemaError(f"{field} is not a directory: {resolved}")
    return resolved


def _require_regular_file(path: Path, *, field: str, read_only: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SchemaError(f"{field} must be a regular non-symlink file: {path}")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SchemaError(f"{field} must be a singly linked regular file: {path}")
    if read_only and metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SchemaError(f"{field} must be read-only: {path}")
    return path


def _require_readiness_directory_closure(target_root: Path) -> Path:
    directory = target_root / LIVE_COMPATIBILITY_RELATIVE_PATH.parent
    if directory.is_symlink() or not directory.is_dir():
        raise SchemaError(
            "target runtime_readiness must be a regular non-symlink directory"
        )
    expected = {
        LIVE_COMPATIBILITY_RELATIVE_PATH.name,
        LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH.name,
        LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH.name,
    }
    entries = list(directory.iterdir())
    actual = {path.name for path in entries}
    evidence = directory / LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH.name
    invalid_entry = any(
        path.is_symlink()
        or (
            path == evidence
            and not path.is_dir()
        )
        or (
            path != evidence
            and not path.is_file()
        )
        for path in entries
    )
    if actual != expected or invalid_entry:
        raise SchemaError(
            "target runtime_readiness lacks the exact canonical "
            "receipt/probe-evidence/sidecar closure"
        )
    return directory


def _validate_pilot_manifest(manifest: Mapping[str, Any], *, field: str) -> None:
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in _EXPECTED_PILOT.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise SchemaError(f"{field} is not the registered PC-01 pilot: {mismatches}")
    if list(manifest.get("systems", ())) != list(SYSTEM_IDS):
        raise SchemaError(f"{field} must register exactly E0--E3")


def _validated_campaign(root: Path, *, field: str) -> dict[str, Any]:
    # Imported lazily to keep the receipt module out of package-validation
    # import cycles and to ensure the normal validator remains authoritative.
    from .package_validator import _validate_campaign_for_live_readiness_replay

    report = _validate_campaign_for_live_readiness_replay(root)
    if not report.passed:
        raise SchemaError(
            f"{field} campaign validation failed: " + "; ".join(report.errors)
        )
    manifest = read_json(root / "campaign_manifest.json")
    _validate_pilot_manifest(manifest, field=field)
    return manifest


def _schedule(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _require_regular_file(root / "schedule" / "schedule.jsonl", field="schedule")
    if sha256_file(path) != manifest.get("schedule_sha256"):
        raise SchemaError("live-readiness schedule differs from campaign manifest")
    rows = read_jsonl(path)
    validate_schedule(rows)
    if len(rows) != 65:
        raise SchemaError("live-readiness pilot schedule must contain exactly 65 blocks")
    return rows


def _first_normal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get("task_partition") == "normal"]
    if len(matches) != 50:
        raise SchemaError("live-readiness schedule must contain exactly 50 normal blocks")
    row = matches[0]
    if row.get("matched_model_seed") != 42 or row.get("repeat_id") != 0:
        raise SchemaError("live-readiness block must use matched seed 42, repeat 0")
    return row


def _path_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    if not cleaned:
        raise SchemaError(f"unsafe empty compatibility artifact path ID: {value!r}")
    return cleaned[:160]


def _block_base(root: Path, row: Mapping[str, Any]) -> Path:
    return (
        root
        / "paired_blocks"
        / f"seed_{int(row['matched_model_seed'])}"
        / _path_id(str(row["task_id"]))
        / f"repeat_{int(row['repeat_id'])}"
    )


def _require_exact_directory_layout(
    root: Path,
    *,
    files: Sequence[str],
    directories: Sequence[str],
    field: str,
) -> Path:
    """Require one registered directory level with no aliases or extras."""

    directory = _require_regular_directory(root, field=field)
    entries = list(directory.iterdir())
    if any(path.is_symlink() for path in entries):
        raise SchemaError(f"{field} contains a symlink")
    expected = set(files) | set(directories)
    actual = {path.name for path in entries}
    if actual != expected:
        raise SchemaError(
            f"{field} does not have the exact registered entries; "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name in files:
        _require_regular_file(directory / name, field=f"{field}/{name}")
    for name in directories:
        _require_regular_directory(directory / name, field=f"{field}/{name}")
    return directory


def _require_exact_tree_layout(
    root: Path,
    *,
    relative_files: Sequence[str],
    field: str,
) -> Path:
    """Require an exact recursive file/directory closure without symlinks.

    Empty directories are evidence too: accepting one would let an unregistered
    branch survive the source-block review while disappearing from the portable
    file inventory.  Directory names are therefore derived solely from the
    registered file paths.
    """

    directory = _require_regular_directory(root, field=field)
    expected_files: set[str] = set()
    expected_directories: set[str] = set()
    for value in relative_files:
        relative = safe_relative_path(value)
        if relative == Path(".") or not relative.name:
            raise SchemaError(f"{field} contains an empty registered path")
        normalized = relative.as_posix()
        if normalized in expected_files:
            raise SchemaError(f"{field} contains a duplicate registered path")
        expected_files.add(normalized)
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise SchemaError(f"{field} contains a symlink")
        relative = path.relative_to(directory).as_posix()
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            _require_regular_file(path, field=f"{field}/{relative}")
            actual_files.add(relative)
        else:
            raise SchemaError(f"{field} contains a non-regular filesystem entry")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise SchemaError(
            f"{field} does not have the exact registered tree; "
            f"expected_files={sorted(expected_files)}, "
            f"actual_files={sorted(actual_files)}, "
            f"expected_directories={sorted(expected_directories)}, "
            f"actual_directories={sorted(actual_directories)}"
        )
    return directory


def _validate_runtime_tree_closure(runtime: Path, *, field: str) -> None:
    """Close the runtime tree against its own canonical artifact inventory."""

    manifest_path = _require_regular_file(
        runtime / "artifact_hashes.json",
        field=f"{field}/artifact_hashes.json",
    )
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SchemaError(f"{field} artifact hash inventory is malformed")
    registered: list[str] = []
    for relative, digest in files.items():
        normalized = safe_relative_path(str(relative))
        if normalized == Path(".") or normalized.name == "artifact_hashes.json":
            raise SchemaError(f"{field} artifact hash path is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SchemaError(f"{field} artifact hash digest is malformed")
        candidate = runtime / normalized
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or sha256_file(candidate) != digest
        ):
            raise SchemaError(f"{field} artifact hash mismatch: {normalized}")
        registered.append(normalized.as_posix())
    _require_exact_tree_layout(
        runtime,
        relative_files=("artifact_hashes.json", *registered),
        field=field,
    )


def _validate_sealed_tree_closure(
    sealed: Path,
    runtime: Path,
    *,
    status: Mapping[str, Any],
    field: str,
) -> None:
    """Require the only sealed layouts emitted by the registered runner."""

    launched = status.get("launched") is True
    completed = status.get("completed") is True
    infrastructure_invalid = status.get("infrastructure_invalid") is True
    if not launched:
        _require_exact_tree_layout(sealed, relative_files=(), field=field)
        return

    episode_id = str(status.get("episode_id") or "")
    if not episode_id:
        summary_path = runtime / "episode_summary.json"
        if summary_path.is_file():
            episode_id = str(read_json(summary_path).get("episode_id") or "")
    if not episode_id:
        raise SchemaError(f"{field} launched system lacks an episode identity")
    episode_path = f"{_path_id(episode_id)}/verifier_events.jsonl"
    terminal_path = runtime / "terminal_signals.jsonl"
    terminal_records = read_jsonl(terminal_path) if terminal_path.is_file() else []

    if completed and not infrastructure_invalid:
        expected = (".seal_key", episode_path)
    elif infrastructure_invalid and not completed:
        expected = [
            ".seal_key",
            "infrastructure_invalid.json",
            "runtime_exception.txt",
        ]
        if terminal_records:
            expected.append(episode_path)
    else:
        raise SchemaError(f"{field} has an unauthorized launched system state")
    _require_exact_tree_layout(sealed, relative_files=expected, field=field)


def _validate_system_tree_closure(
    package: Path,
    *,
    status: Mapping[str, Any],
    field: str,
) -> None:
    runtime = package / "runtime"
    sealed = package / "sealed"
    if status.get("launched") is True:
        _validate_runtime_tree_closure(runtime, field=f"{field}/runtime")
    else:
        _require_exact_tree_layout(
            runtime,
            relative_files=(),
            field=f"{field}/runtime",
        )
    _validate_sealed_tree_closure(
        sealed,
        runtime,
        status=status,
        field=f"{field}/sealed",
    )


def _validate_selected_block_directory_closure(
    root: Path,
    schedule_row: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Replay attempt resolution and the exact registered block topology."""

    selected_base = _require_regular_directory(
        _block_base(root, schedule_row), field="live-readiness selected block"
    )
    resolution_path = _require_regular_file(
        selected_base / "resolution.json",
        field="live-readiness block resolution",
    )
    resolution = read_json(resolution_path)
    selected_attempt = resolution.get("selected_attempt_id")
    if type(selected_attempt) is not int or selected_attempt < 0:
        raise SchemaError("live-readiness block lacks a selected complete attempt")

    discovered = discover_block_attempt_directories(
        selected_base,
        maximum=int(schedule_row["max_block_attempts"]),
    )
    discovered_names = tuple(f"rerun_{attempt_id}" for attempt_id, _ in discovered)
    _require_exact_directory_layout(
        selected_base,
        files=("resolution.json",),
        directories=discovered_names,
        field="live-readiness selected block",
    )
    if not discovered or selected_attempt not in {
        attempt_id for attempt_id, _ in discovered
    }:
        raise SchemaError(
            "live-readiness probe does not retain its selected physical attempt"
        )
    attempts: list[dict[str, Any]] = []
    reruns: dict[int, Path] = {}
    for attempt_id, rerun in discovered:
        _require_exact_directory_layout(
            rerun,
            files=("block_manifest.json",),
            directories=SYSTEM_IDS,
            field=f"live-readiness rerun_{attempt_id}",
        )
        block_manifest = read_json(rerun / "block_manifest.json")
        if (
            int(block_manifest.get("attempt_id", -1)) != attempt_id
            or str(block_manifest.get("block_id", ""))
            != str(schedule_row["block_id"])
        ):
            raise SchemaError(
                "live-readiness block manifest identity differs from schedule"
            )
        for system_id in SYSTEM_IDS:
            package = _require_exact_directory_layout(
                rerun / system_id,
                files=(),
                directories=("runtime", "sealed"),
                field=(
                    f"live-readiness rerun_{attempt_id}/{system_id} system package"
                ),
            )
            status = block_manifest.get("systems", {}).get(system_id)
            if not isinstance(status, Mapping):
                raise SchemaError(
                    "live-readiness block manifest has a malformed system status"
                )
            _validate_system_tree_closure(
                package,
                status=status,
                field=f"live-readiness rerun_{attempt_id}/{system_id}",
            )
        attempts.append(block_manifest)
        reruns[attempt_id] = rerun

    calculated = resolve_block_attempts(schedule_row, attempts)
    if resolution != calculated:
        raise SchemaError(
            "live-readiness block resolution differs from registered attempt evidence"
        )
    if calculated.get("status") != "INCLUDED":
        raise SchemaError("live-readiness block is not a complete included E0--E3 block")
    return (
        selected_base,
        reruns[selected_attempt],
        resolution,
        attempts[selected_attempt],
    )


def _validate_registered_included_block_schema(
    root: Path,
    schedule_row: Mapping[str, Any],
) -> None:
    """Replay the canonical package validator for the completed source block."""

    # package_validator imports this module to enforce the receipt boundary, so
    # this import must remain lazy.
    from .package_validator import ValidationReport, _validate_block

    report = ValidationReport(campaign_dir=str(root))
    outcome = _validate_block(
        root,
        schedule_row,
        report,
        require_complete=True,
    )
    if outcome != "INCLUDED" or report.errors:
        raise SchemaError(
            "live-readiness source block failed the registered included-block schema"
        )


def _tree_binding(root: Path, *, field: str) -> dict[str, Any]:
    descriptors = _tree_descriptors(root, field=field)
    return {
        "file_count": len(descriptors),
        "total_bytes": sum(int(row["bytes"]) for row in descriptors),
        "files_sha256": sha256_json(descriptors),
    }


def _tree_descriptors(root: Path, *, field: str) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise SchemaError(f"{field} tree is missing or symlinked")
    descriptors: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SchemaError(f"{field} tree contains a symlink: {path}")
        if path.is_dir():
            continue
        metadata = path.stat()
        if (
            not path.is_file()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise SchemaError(f"{field} tree contains a non-regular file: {path}")
        descriptors.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": metadata.st_size,
            }
        )
    if not descriptors:
        raise SchemaError(f"{field} tree contains no evidence files")
    return descriptors


def _copy_sealed_file(source: Path, destination: Path, *, field: str) -> None:
    """Copy one regular file and prove that no source race changed its bytes."""

    source = _require_regular_file(source, field=field)
    expected_sha256 = sha256_file(source)
    expected_size = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SchemaError(f"portable probe evidence already contains {destination}")
    with source.open("rb") as reader, destination.open("xb") as writer:
        while True:
            block = reader.read(1024 * 1024)
            if not block:
                break
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    # Portable evidence remains immutable while verifier keys retain private
    # owner-only permissions required by the sealed-stream validator.
    os.chmod(destination, 0o400)
    if (
        destination.stat().st_size != expected_size
        or sha256_file(destination) != expected_sha256
        or source.stat().st_size != expected_size
        or sha256_file(source) != expected_sha256
    ):
        raise SchemaError(f"{field} changed while portable probe evidence was sealed")


def _seal_probe_evidence(
    probe_root: Path,
    target_root: Path,
    *,
    probe_first: Mapping[str, Any],
) -> Path:
    """Copy only the material needed to replay the non-scored probe."""

    evidence_root = target_root / LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH
    if evidence_root.exists() or evidence_root.is_symlink():
        raise SchemaError("portable probe-evidence directory already exists")
    evidence_root.mkdir(parents=True)
    for relative in _PROBE_EVIDENCE_FIXED_FILES:
        _copy_sealed_file(
            probe_root / relative,
            evidence_root / relative,
            field=f"probe evidence {relative}",
        )

    source_block, _, _, _ = _validate_selected_block_directory_closure(
        probe_root, probe_first
    )
    block_relative = source_block.relative_to(probe_root)
    source_descriptors = _tree_descriptors(
        source_block, field="live-readiness source block"
    )
    # File-only copying would erase the deliberately empty runtime/sealed
    # directories of systems that were not launched after an infrastructure
    # interruption.  Preserve the already schema-validated tree shape before
    # copying its files so portable replay sees the exact physical attempt.
    for source_directory in sorted(
        (path for path in source_block.rglob("*") if path.is_dir()),
        key=lambda path: (len(path.parts), path.as_posix()),
    ):
        relative = source_directory.relative_to(source_block)
        (evidence_root / block_relative / relative).mkdir(
            parents=True,
            exist_ok=True,
        )
    for descriptor in source_descriptors:
        relative = Path(str(descriptor["relative_path"]))
        _copy_sealed_file(
            source_block / relative,
            evidence_root / block_relative / relative,
            field=f"probe block {relative.as_posix()}",
        )

    files = _tree_descriptors(evidence_root, field="portable probe evidence")
    manifest = {
        "schema_version": PROBE_EVIDENCE_SCHEMA_VERSION,
        "record_type": PROBE_EVIDENCE_RECORD_TYPE,
        "evidence_scope": LIVE_COMPATIBILITY_EVIDENCE_SCOPE,
        "source_probe_campaign_id": str(
            read_json(probe_root / "campaign_manifest.json")["campaign_id"]
        ),
        "selected_block_relative_path": block_relative.as_posix(),
        "files": files,
        "files_sha256": sha256_json(files),
    }
    atomic_write_json(
        evidence_root / PROBE_EVIDENCE_MANIFEST_NAME,
        manifest,
        mode=0o444,
    )
    return evidence_root


def _validate_probe_evidence_package(
    target_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reopen the exact portable probe package without any external path."""

    evidence_root = _require_regular_directory(
        target_root / LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH,
        field="portable probe evidence",
    )
    manifest_path = _require_regular_file(
        evidence_root / PROBE_EVIDENCE_MANIFEST_NAME,
        field="portable probe evidence manifest",
        read_only=True,
    )
    manifest = read_json(manifest_path)
    if set(manifest) != {
        "schema_version",
        "record_type",
        "evidence_scope",
        "source_probe_campaign_id",
        "selected_block_relative_path",
        "files",
        "files_sha256",
    }:
        raise SchemaError("portable probe evidence manifest fields differ")
    if (
        manifest.get("schema_version") != PROBE_EVIDENCE_SCHEMA_VERSION
        or manifest.get("record_type") != PROBE_EVIDENCE_RECORD_TYPE
        or manifest.get("evidence_scope") != LIVE_COMPATIBILITY_EVIDENCE_SCOPE
    ):
        raise SchemaError("portable probe evidence identity differs")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise SchemaError("portable probe evidence manifest has no file inventory")
    actual_files = [
        descriptor
        for descriptor in _tree_descriptors(
            evidence_root, field="portable probe evidence"
        )
        if descriptor["relative_path"] != PROBE_EVIDENCE_MANIFEST_NAME
    ]
    if expected_files != actual_files or manifest.get("files_sha256") != sha256_json(
        actual_files
    ):
        raise SchemaError("portable probe evidence differs from its exact file inventory")
    for descriptor in actual_files:
        path = evidence_root / str(descriptor["relative_path"])
        _require_regular_file(
            path,
            field=f"portable probe evidence {descriptor['relative_path']}",
            read_only=True,
        )

    probe_manifest = read_json(evidence_root / "campaign_manifest.json")
    _validate_pilot_manifest(probe_manifest, field="sealed probe")
    if str(probe_manifest.get("campaign_id")) != str(
        manifest.get("source_probe_campaign_id")
    ):
        raise SchemaError("portable probe campaign identity differs from its manifest")
    probe_first = _first_normal(_schedule(evidence_root, probe_manifest))
    selected_block_relative = _block_base(evidence_root, probe_first).relative_to(
        evidence_root
    )
    if manifest.get("selected_block_relative_path") != selected_block_relative.as_posix():
        raise SchemaError("portable probe selected-block path differs from its schedule")
    selected_parts = selected_block_relative.parts
    if len(selected_parts) != 4 or selected_parts[0] != "paired_blocks":
        raise SchemaError("portable probe selected-block path has the wrong topology")
    root_files = tuple(
        Path(relative).name
        for relative in _PROBE_EVIDENCE_FIXED_FILES
        if len(Path(relative).parts) == 1
    ) + (PROBE_EVIDENCE_MANIFEST_NAME,)
    _require_exact_directory_layout(
        evidence_root,
        files=root_files,
        directories=("schedule", "frozen", "paired_blocks"),
        field="portable probe evidence root",
    )
    _require_exact_directory_layout(
        evidence_root / "schedule",
        files=("schedule.jsonl",),
        directories=(),
        field="portable probe schedule",
    )
    _require_exact_directory_layout(
        evidence_root / "frozen",
        files=("protocol.yaml",),
        directories=(),
        field="portable probe frozen evidence",
    )
    ancestor = evidence_root / "paired_blocks"
    for depth, child_name in enumerate(selected_parts[1:], start=1):
        _require_exact_directory_layout(
            ancestor,
            files=(),
            directories=(child_name,),
            field=f"portable probe selected-block ancestor depth {depth}",
        )
        ancestor /= child_name
    _validate_selected_block_directory_closure(evidence_root, probe_first)

    allowed_fixed = set(_PROBE_EVIDENCE_FIXED_FILES)
    allowed_prefix = selected_block_relative.as_posix() + "/"
    actual_names = {str(row["relative_path"]) for row in actual_files}
    if not allowed_fixed.issubset(actual_names) or any(
        name not in allowed_fixed and not name.startswith(allowed_prefix)
        for name in actual_names
    ):
        raise SchemaError("portable probe evidence contains a non-minimal file")

    artifact_hashes = read_json(evidence_root / "artifact_hashes.json")
    frozen_files = artifact_hashes.get("files")
    if not isinstance(frozen_files, Mapping) or frozen_files.get(
        "frozen/protocol.yaml"
    ) != sha256_file(evidence_root / "frozen/protocol.yaml"):
        raise SchemaError("portable probe protocol differs from frozen artifact hashes")
    return evidence_root, manifest, probe_manifest, probe_first


def _probe_evidence_binding(target_root: Path) -> dict[str, Any]:
    evidence_root, _, _, _ = _validate_probe_evidence_package(target_root)
    manifest_path = evidence_root / PROBE_EVIDENCE_MANIFEST_NAME
    return {
        "relative_path": LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH.as_posix(),
        "manifest_relative_path": (
            LIVE_COMPATIBILITY_PROBE_EVIDENCE_RELATIVE_PATH
            / PROBE_EVIDENCE_MANIFEST_NAME
        ).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "tree": _tree_binding(evidence_root, field="portable probe evidence"),
    }


def _runtime_input_binding(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hashes_path = _require_regular_file(root / "artifact_hashes.json", field="artifact hashes")
    hashes = read_json(hashes_path)
    files = hashes.get("files")
    if not isinstance(files, Mapping):
        raise SchemaError("campaign artifact hashes lack a file mapping")
    # Campaign provenance contains the campaign ID/freeze time and schedule has
    # campaign-keyed RNG.  Every other frozen/runtime input must be byte-identical
    # between the isolated probe and target pilot.
    selected = {
        str(relative): str(digest)
        for relative, digest in sorted(files.items())
        if (
            (str(relative).startswith("frozen/") and str(relative) != "frozen/provenance.json")
            or str(relative).startswith("memory/")
        )
    }
    if not selected:
        raise SchemaError("campaign has no immutable live-runtime input set")
    for relative, digest in selected.items():
        path = _require_regular_file(root / relative, field=f"runtime input {relative}")
        if sha256_file(path) != digest:
            raise SchemaError(f"runtime input hash mismatch: {relative}")

    attestation_path = _require_regular_file(
        root / "frozen" / "runner_attestation.json",
        field="runner attestation",
    )
    attestation = read_json(attestation_path)
    source_rows = attestation.get("source_files")
    if not isinstance(source_rows, list):
        raise SchemaError("runner attestation lacks source files")
    source_hashes: dict[str, str] = {}
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise SchemaError("runner attestation contains a malformed source row")
        relative = str(row.get("relative_path") or "")
        digest = str(row.get("sha256") or "")
        if not relative or relative in source_hashes:
            raise SchemaError("runner attestation source rows are empty or duplicated")
        source_hashes[relative] = digest
    repository_root = Path(__file__).resolve().parents[4]
    required_source_rows: list[dict[str, str]] = []
    for relative in LIVE_COMPATIBILITY_REQUIRED_SOURCE_RELATIVE_PATHS:
        expected = source_hashes.get(relative)
        live = repository_root / relative
        frozen = root / "frozen" / "runner_source" / relative
        if (
            expected is None
            or not live.is_file()
            or live.is_symlink()
            or not frozen.is_file()
            or frozen.is_symlink()
            or sha256_file(live) != expected
            or sha256_file(frozen) != expected
        ):
            raise SchemaError(
                f"live-readiness gate source is not frozen and attested: {relative}"
            )
        required_source_rows.append({"relative_path": relative, "sha256": expected})

    shared_manifest = {field: manifest.get(field) for field in _SHARED_MANIFEST_FIELDS}
    return {
        "shared_manifest_fields": shared_manifest,
        "shared_manifest_fields_sha256": sha256_json(shared_manifest),
        "immutable_runtime_files": selected,
        "immutable_runtime_files_sha256": sha256_json(selected),
        "required_gate_sources": required_source_rows,
        "required_gate_sources_sha256": sha256_json(required_source_rows),
    }


def _sealed_probe_runtime_binding(
    evidence_root: Path,
    probe_manifest: Mapping[str, Any],
    *,
    target_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the probe runtime identity from compact sealed metadata.

    The large immutable model and memory payloads are not copied a second time.
    Their complete relative-path digest map is retained in the probe's frozen
    artifact manifest and must exactly equal the target package whose real bytes
    are reopened by :func:`_runtime_input_binding`.
    """

    artifact_hashes = read_json(evidence_root / "artifact_hashes.json")
    files = artifact_hashes.get("files")
    if not isinstance(files, Mapping):
        raise SchemaError("portable probe artifact hashes lack a file mapping")
    selected = {
        str(relative): str(digest)
        for relative, digest in sorted(files.items())
        if (
            (
                str(relative).startswith("frozen/")
                and str(relative) != "frozen/provenance.json"
            )
            or str(relative).startswith("memory/")
        )
    }
    if not selected or selected != target_runtime.get("immutable_runtime_files"):
        raise SchemaError(
            "portable probe immutable runtime differs from the target campaign"
        )
    shared_manifest = {
        field: probe_manifest.get(field) for field in _SHARED_MANIFEST_FIELDS
    }
    if shared_manifest != target_runtime.get("shared_manifest_fields"):
        raise SchemaError(
            "portable probe shared manifest differs from the target campaign"
        )
    required_sources = target_runtime.get("required_gate_sources")
    if not isinstance(required_sources, list) or not required_sources:
        raise SchemaError("target runtime has no attested live-gate source set")
    return {
        "shared_manifest_fields": shared_manifest,
        "shared_manifest_fields_sha256": sha256_json(shared_manifest),
        "immutable_runtime_files": selected,
        "immutable_runtime_files_sha256": sha256_json(selected),
        "required_gate_sources": required_sources,
        "required_gate_sources_sha256": sha256_json(required_sources),
    }


def _assert_no_scored_probe_outputs(probe_root: Path, selected_base: Path) -> None:
    paired_root = probe_root / "paired_blocks"
    allowed_ancestors = {
        path
        for path in selected_base.parents
        if path == paired_root or paired_root in path.parents
    }
    for path in paired_root.rglob("*"):
        if path.is_symlink() or not (
            path == selected_base
            or path in allowed_ancestors
            or selected_base in path.parents
        ):
            raise SchemaError(
                "live-readiness probe contains another attempted/scored block"
            )
    for directory_name in ("aggregate", "manual_audit", "component_test"):
        directory = probe_root / directory_name
        if directory.exists() and any(directory.iterdir()):
            raise SchemaError(
                f"live-readiness probe contains forbidden {directory_name} outputs"
            )
    for filename in ("completion.json", "campaign_evidence_manifest.json"):
        if (probe_root / filename).exists():
            raise SchemaError(f"live-readiness probe must not contain {filename}")


def _assert_target_unstarted(target_root: Path) -> None:
    paired = target_root / "paired_blocks"
    if paired.exists() and any(paired.iterdir()):
        raise SchemaError(
            "live-readiness receipt must be issued before any target pilot block starts"
        )
    if (target_root / "completion.json").exists():
        raise SchemaError("live-readiness target already has a completion artifact")
    access_path = _require_regular_file(
        target_root / "access_ledger.jsonl",
        field="target access ledger",
    )
    if any(
        str(record.get("event_type") or "") == "episode_task_load"
        for record in read_jsonl(access_path)
    ):
        raise SchemaError(
            "live-readiness receipt must be issued before any target pilot block starts"
        )


def _cross_bind_campaigns(
    probe_root: Path,
    target_root: Path,
    *,
    require_target_unstarted: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if probe_root == target_root:
        raise SchemaError("live-readiness probe must use a separate campaign directory")
    if probe_root in target_root.parents or target_root in probe_root.parents:
        raise SchemaError(
            "live-readiness probe and target campaign directories must not overlap"
        )
    probe_manifest = _validated_campaign(probe_root, field="probe")
    target_manifest = _validated_campaign(target_root, field="target")
    if probe_manifest.get("campaign_id") == target_manifest.get("campaign_id"):
        raise SchemaError("probe and target campaigns must have distinct campaign IDs")
    probe_runtime = _runtime_input_binding(probe_root, probe_manifest)
    target_runtime = _runtime_input_binding(target_root, target_manifest)
    if probe_runtime != target_runtime:
        raise SchemaError(
            "probe and target campaigns do not use identical "
            "runtime/model/memory/task/environment inputs"
        )
    probe_first = _first_normal(_schedule(probe_root, probe_manifest))
    target_first = _first_normal(_schedule(target_root, target_manifest))
    if (
        probe_first.get("task_id") != target_first.get("task_id")
        or probe_first.get("matched_model_seed") != target_first.get("matched_model_seed")
        or probe_first.get("repeat_id") != target_first.get("repeat_id")
    ):
        raise SchemaError("probe does not use the target pilot's first normal task block")
    if require_target_unstarted:
        _assert_target_unstarted(target_root)
    return probe_manifest, target_manifest, probe_first, target_first


def authorize_live_compatibility_probe(
    probe_campaign_dir: str | Path,
    target_campaign_dir: str | Path,
    *,
    block_id: str,
) -> dict[str, Any]:
    """Authorize exactly the first normal block in an isolated pilot clone."""

    probe_root = _require_regular_directory(probe_campaign_dir, field="probe campaign")
    target_root = _require_regular_directory(target_campaign_dir, field="target campaign")
    _, _, probe_first, _ = _cross_bind_campaigns(
        probe_root,
        target_root,
        require_target_unstarted=True,
    )
    if str(probe_first["block_id"]) != str(block_id):
        raise SchemaError(
            "live-readiness probe must execute the first registered normal WebArena block"
        )
    selected_base = _block_base(probe_root, probe_first)
    _assert_no_scored_probe_outputs(probe_root, selected_base)
    return probe_first


def _matched_block_binding(
    probe_root: Path,
    probe_first: Mapping[str, Any],
) -> dict[str, Any]:
    selected_base, rerun, resolution, block_manifest = (
        _validate_selected_block_directory_closure(probe_root, probe_first)
    )
    _assert_no_scored_probe_outputs(probe_root, selected_base)
    if resolution.get("status") != "INCLUDED":
        raise SchemaError("live-readiness block is not a complete included E0--E3 block")
    selected_attempt = resolution.get("selected_attempt_id")
    if type(selected_attempt) is not int or selected_attempt < 0:
        raise SchemaError("live-readiness block lacks a selected complete attempt")
    statuses = block_manifest.get("systems")
    if not isinstance(statuses, Mapping) or set(statuses) != set(SYSTEM_IDS):
        raise SchemaError("live-readiness block manifest lacks exactly E0--E3")
    if any(
        not isinstance(statuses[system_id], Mapping)
        or statuses[system_id].get("launched") is not True
        or statuses[system_id].get("completed") is not True
        or statuses[system_id].get("infrastructure_invalid") is not False
        for system_id in SYSTEM_IDS
    ):
        raise SchemaError("live-readiness block did not complete every system normally")
    # Keep YAML parsing in the package validator's established helper.
    from .package_validator import load_yaml

    loop_rule = load_yaml(probe_root / "frozen" / "protocol.yaml").get("loop_rule")
    if not isinstance(loop_rule, Mapping) or not isinstance(
        loop_rule.get("state_fingerprint_fields"), list
    ):
        raise SchemaError("live-readiness protocol lacks state fingerprint fields")
    causal = validate_included_block_causal_trace(
        rerun,
        state_fingerprint_fields=loop_rule["state_fingerprint_fields"],
        require_trained_first_pre_action=True,
    )
    e2_memory_path = rerun / "E2" / "runtime" / "memory_queries.jsonl"
    e2_records = read_jsonl(e2_memory_path) if e2_memory_path.exists() else []
    if e2_records:
        raise SchemaError("live-readiness E2 performed forbidden memory retrieval")
    e3_memory_path = _require_regular_file(
        rerun / "E3" / "runtime" / "memory_queries.jsonl",
        field="live-readiness E3 memory stream",
    )
    e3_records = read_jsonl(e3_memory_path)
    e3_queries = sum(
        str(row.get("event_type") or "") == "post_failure_query"
        for row in e3_records
    )
    e3_writes = sum("write" in str(row.get("event_type") or "").casefold() for row in e3_records)
    if e3_writes:
        raise SchemaError("live-readiness E3 performed a forbidden memory write")
    summaries = {
        system_id: sha256_file(
            rerun / system_id / "runtime" / "episode_summary.json"
        )
        for system_id in SYSTEM_IDS
    }
    return {
        "block_id": str(probe_first["block_id"]),
        "task_id": str(probe_first["task_id"]),
        "task_partition": "normal",
        "matched_model_seed": 42,
        "repeat_id": 0,
        "selected_attempt_id": selected_attempt,
        "systems": list(SYSTEM_IDS),
        "stage_seeds_sha256": sha256_json(probe_first["stage_seeds"]),
        "block_tree": _tree_binding(selected_base, field="live-readiness block"),
        "episode_summary_sha256": summaries,
        "causal_validation": causal,
        "e2_memory_query_count": 0,
        "e3_memory_query_count": e3_queries,
        "e3_memory_write_count": 0,
    }


def _campaign_receipt_binding(
    root: Path,
    manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    first_normal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": str(manifest["campaign_id"]),
        "campaign_manifest_sha256": sha256_file(root / "campaign_manifest.json"),
        "frozen_artifact_hashes_sha256": sha256_file(root / "artifact_hashes.json"),
        "schedule_sha256": sha256_file(root / "schedule" / "schedule.jsonl"),
        "repository_commit": str(manifest["repository_commit"]),
        "protocol_id": str(manifest["protocol_id"]),
        "runner_attestation_sha256": str(manifest["runner_attestation_sha256"]),
        "first_normal_block_id": str(first_normal["block_id"]),
        "first_normal_task_id": str(first_normal["task_id"]),
        "first_normal_stage_seeds_sha256": sha256_json(first_normal["stage_seeds"]),
        "immutable_runtime_files_sha256": runtime["immutable_runtime_files_sha256"],
        "required_gate_sources_sha256": runtime["required_gate_sources_sha256"],
    }


def _receipt_core(
    probe_root: Path,
    target_root: Path,
    *,
    probe_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    probe_first: Mapping[str, Any],
    target_first: Mapping[str, Any],
    probe_runtime: Mapping[str, Any],
    target_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": LIVE_COMPATIBILITY_SCHEMA_VERSION,
        "record_type": LIVE_COMPATIBILITY_RECORD_TYPE,
        "status": "PASS",
        "evidence_scope": LIVE_COMPATIBILITY_EVIDENCE_SCOPE,
        "evidence_role": LIVE_COMPATIBILITY_EVIDENCE_ROLE,
        "paper_table_status": "N/R",
        "paper_table_eligible": False,
        "probe_campaign": _campaign_receipt_binding(
            probe_root, probe_manifest, probe_runtime, probe_first
        ),
        "target_campaign": _campaign_receipt_binding(
            target_root, target_manifest, target_runtime, target_first
        ),
        "matched_block": _matched_block_binding(probe_root, probe_first),
        "probe_access_ledger_sha256": sha256_file(
            probe_root / "access_ledger.jsonl"
        ),
        "probe_deviation_ledger_sha256": sha256_file(
            probe_root / "deviation_ledger.jsonl"
        ),
        "shared_runtime_binding_sha256": sha256_json(probe_runtime),
    }


def _external_receipt_core(
    probe_root: Path,
    target_root: Path,
    *,
    require_target_unstarted: bool,
) -> dict[str, Any]:
    probe_manifest, target_manifest, probe_first, target_first = _cross_bind_campaigns(
        probe_root,
        target_root,
        require_target_unstarted=require_target_unstarted,
    )
    _validate_registered_included_block_schema(probe_root, probe_first)
    probe_runtime = _runtime_input_binding(probe_root, probe_manifest)
    target_runtime = _runtime_input_binding(target_root, target_manifest)
    return _receipt_core(
        probe_root,
        target_root,
        probe_manifest=probe_manifest,
        target_manifest=target_manifest,
        probe_first=probe_first,
        target_first=target_first,
        probe_runtime=probe_runtime,
        target_runtime=target_runtime,
    )


def _sealed_receipt_core(
    target_root: Path,
    *,
    require_target_unstarted: bool,
) -> dict[str, Any]:
    evidence_root, _, probe_manifest, probe_first = (
        _validate_probe_evidence_package(target_root)
    )
    target_manifest = _validated_campaign(target_root, field="target")
    if probe_manifest.get("campaign_id") == target_manifest.get("campaign_id"):
        raise SchemaError("sealed probe and target campaigns have the same campaign ID")
    target_first = _first_normal(_schedule(target_root, target_manifest))
    if (
        probe_first.get("task_id") != target_first.get("task_id")
        or probe_first.get("matched_model_seed")
        != target_first.get("matched_model_seed")
        or probe_first.get("repeat_id") != target_first.get("repeat_id")
    ):
        raise SchemaError(
            "portable probe does not use the target pilot's first normal task block"
        )
    if require_target_unstarted:
        _assert_target_unstarted(target_root)
    target_runtime = _runtime_input_binding(target_root, target_manifest)
    probe_runtime = _sealed_probe_runtime_binding(
        evidence_root,
        probe_manifest,
        target_runtime=target_runtime,
    )
    return _receipt_core(
        evidence_root,
        target_root,
        probe_manifest=probe_manifest,
        target_manifest=target_manifest,
        probe_first=probe_first,
        target_first=target_first,
        probe_runtime=probe_runtime,
        target_runtime=target_runtime,
    )


def _complete_receipt(core: Mapping[str, Any], target_root: Path) -> dict[str, Any]:
    body = {**dict(core), "probe_evidence": _probe_evidence_binding(target_root)}
    return {**body, "receipt_id": f"pc01-live-ready-{sha256_json(body)[:24]}"}


def write_live_compatibility_receipt(
    *,
    probe_campaign_dir: str | Path,
    target_campaign_dir: str | Path,
) -> Path:
    """Write one immutable readiness receipt after a valid isolated probe."""

    probe_root = _require_regular_directory(probe_campaign_dir, field="probe campaign")
    target_root = _require_regular_directory(target_campaign_dir, field="target campaign")
    destination = target_root / LIVE_COMPATIBILITY_RELATIVE_PATH
    sidecar = target_root / LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH
    if destination.exists() or sidecar.exists():
        # Idempotent validation is safe; replacement/resealing is not.
        validate_live_compatibility_receipt(target_root)
        return destination
    readiness_directory = destination.parent
    if readiness_directory.exists():
        if readiness_directory.is_symlink() or not readiness_directory.is_dir():
            raise SchemaError(
                "target runtime_readiness must be a regular non-symlink directory"
            )
        if any(readiness_directory.iterdir()):
            raise SchemaError(
                "target runtime_readiness must be empty before receipt issuance"
            )
    external_core = _external_receipt_core(
        probe_root,
        target_root,
        require_target_unstarted=True,
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    probe_manifest = read_json(probe_root / "campaign_manifest.json")
    probe_first = _first_normal(_schedule(probe_root, probe_manifest))
    _seal_probe_evidence(
        probe_root,
        target_root,
        probe_first=probe_first,
    )
    sealed_core = _sealed_receipt_core(
        target_root,
        require_target_unstarted=True,
    )
    if sealed_core != external_core:
        raise SchemaError(
            "portable probe evidence changed while the readiness receipt was sealed"
        )
    receipt = _complete_receipt(sealed_core, target_root)
    # Publish the receipt and then its sidecar only after the portable evidence
    # is complete. An interrupted write therefore remains visibly invalid.
    atomic_write_json(destination, receipt, mode=0o444)
    digest = sha256_file(destination)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(sidecar, flags, 0o444)
    try:
        os.write(fd, (digest + "\n").encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(sidecar, 0o444)
    validate_live_compatibility_receipt(target_root)
    return destination


def validate_live_compatibility_receipt(
    target_campaign_dir: str | Path,
) -> dict[str, Any]:
    """Reproduce the receipt solely from target-local sealed probe evidence."""

    target_root = _require_regular_directory(target_campaign_dir, field="target campaign")
    _require_readiness_directory_closure(target_root)
    destination = _require_regular_file(
        target_root / LIVE_COMPATIBILITY_RELATIVE_PATH,
        field="live-readiness receipt",
        read_only=True,
    )
    sidecar = _require_regular_file(
        target_root / LIVE_COMPATIBILITY_SIDECAR_RELATIVE_PATH,
        field="live-readiness receipt sidecar",
        read_only=True,
    )
    if sidecar.read_text(encoding="ascii").strip() != sha256_file(destination):
        raise SchemaError("live-readiness receipt sidecar hash mismatch")
    supplied = read_json(destination)
    if "probe_campaign_path" in supplied:
        raise SchemaError("live-readiness receipt contains a forbidden external probe path")
    expected = _complete_receipt(
        _sealed_receipt_core(
            target_root,
            require_target_unstarted=False,
        ),
        target_root,
    )
    if supplied != expected:
        raise SchemaError("live-readiness receipt differs from exact evidence replay")
    return expected


def require_live_compatibility_before_execution(
    campaign_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Fail immediately before a provisional pilot episode can reset a browser."""

    root = _require_regular_directory(campaign_dir, field="campaign")
    if manifest.get("campaign_mode") == "smoke":
        return None
    if manifest.get("evidence_label") != "PILOT_ONLY":
        readiness_root = root / LIVE_COMPATIBILITY_RELATIVE_PATH.parent
        if readiness_root.exists() or readiness_root.is_symlink():
            raise Table2Error(
                "locked-final/non-pilot execution cannot accept provisional PC-01 live readiness"
            )
        return None
    _validate_pilot_manifest(manifest, field="execution target")
    return validate_live_compatibility_receipt(root)
