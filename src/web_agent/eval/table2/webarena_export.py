"""Build the sealed WebArena 0--49 development-task export.

This module deliberately has no BrowserGym, Playwright, or model dependency.
It reads the pinned upstream task-definition bytes, checks them against the
permanent development-task registry, resolves only registered site URL
placeholders, and writes the complete task/evaluator records expected by the
Table 2 handoff preparer.  Runtime code receives a separate oracle-blind
projection in :mod:`web_agent.eval.table2.production_runner`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit
import zipfile

from .common import (
    SchemaError,
    atomic_write_json,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from .pilot_task_exclusion import load_pilot_task_exclusion_authority


PINNED_BROWSERGYM_WEBARENA_VERSION = "0.14.3"
PINNED_LIBWEBARENA_VERSION = "0.0.4"
PINNED_LIBWEBARENA_WHEEL_SHA256 = (
    "9ebee3b4371502c4f0f7e727a72e5846235d6750d420db9a3b8a168107654feb"
)
PINNED_TASK_MEMBER = "webarena/test.raw.json"
PINNED_TASK_SOURCE_SHA256 = (
    "7b50386fd69163dbc05d615d834df4c6ed2c35596e97a1b10d17451c02537652"
)
PINNED_TASK_DEFINITION_VERSION = "libwebarena-0.0.4:test.raw.json"
PUBLIC_PILOT_EXPORT_SCHEMA_VERSION = "table2-webarena-public-task-export-v1"
PUBLIC_PILOT_EXPORT_RECORD_TYPE = "SealedWebArenaDevelopmentTaskExport"
PUBLIC_PILOT_SELECTION_RULE = (
    "upstream public task indices 0 through 49 in ascending order"
)
PINNED_REQUIRED_URL_TOKENS = frozenset(
    {
        "__GITLAB__",
        "__MAP__",
        "__REDDIT__",
        "__SHOPPING__",
        "__SHOPPING_ADMIN__",
    }
)
WEBARENA_RUNTIME_START_STATE_FIELDS = frozenset(
    {
        "sites",
        "start_url",
        "require_login",
        "storage_state",
        "geolocation",
        "require_reset",
    }
)

_URL_TOKEN = re.compile(r"__[A-Z][A-Z0-9_]*__")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _task_source_bytes(
    source: str | Path,
    *,
    archive_member: str = PINNED_TASK_MEMBER,
    expected_container_sha256: str = PINNED_LIBWEBARENA_WHEEL_SHA256,
    authorized_raw_json_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Read a raw JSON file or the task member of a wheel/ZIP archive."""

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.is_symlink():
        raise SchemaError("upstream WebArena task source must not be a symlink")
    if path.suffix.casefold() in {".whl", ".zip"}:
        if authorized_raw_json_sha256 is not None:
            raise SchemaError(
                "raw-source authorization must be absent when using the pinned "
                "libwebarena wheel container"
            )
        container_sha256 = sha256_file(path)
        _validate_sha256_identity(
            container_sha256,
            expected_container_sha256,
            context="upstream libwebarena wheel container",
        )
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if names.count(archive_member) != 1:
                    raise SchemaError(
                        "upstream archive must contain exactly one registered "
                        f"task member: {archive_member}"
                    )
                payload = archive.read(archive_member)
        except zipfile.BadZipFile as exc:
            raise SchemaError("upstream WebArena task archive is not a valid ZIP") from exc
        source_record = {
            "container_kind": "zip_archive",
            "container_name": path.name,
            "container_sha256": container_sha256,
            "container_authorization": "PINNED_LIBWEBARENA_WHEEL_SHA256",
            "authorized_container_sha256": str(expected_container_sha256)
            .strip()
            .casefold(),
            "member": archive_member,
        }
    else:
        if authorized_raw_json_sha256 is None:
            raise SchemaError(
                "raw WebArena task JSON requires an explicit separately authorized "
                "raw-source SHA-256"
            )
        container_sha256 = sha256_file(path)
        _validate_sha256_identity(
            container_sha256,
            authorized_raw_json_sha256,
            context="authorized raw WebArena task JSON",
        )
        payload = path.read_bytes()
        source_record = {
            "container_kind": "raw_json",
            "container_name": path.name,
            # For a raw source the container and task-definition payload are
            # the same bytes.  Retain both fields so downstream provenance
            # never has to infer a missing container identity.
            "container_sha256": container_sha256,
            "container_authorization": "EXPLICIT_RAW_JSON_SHA256",
            "authorized_container_sha256": str(authorized_raw_json_sha256)
            .strip()
            .casefold(),
            "member": None,
        }
    if not payload:
        raise SchemaError("upstream WebArena task source is empty")
    source_record["task_source_sha256"] = sha256_bytes(payload)
    return payload, source_record


def _parse_task_rows(payload: bytes) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaError("upstream WebArena task source is not valid UTF-8 JSON") from exc
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise SchemaError("upstream WebArena task source must be an array of objects")
    return [dict(row) for row in value]


def _validate_sha256_identity(actual: str, expected: str, *, context: str) -> None:
    normalized = str(expected).strip().casefold()
    if not _SHA256.fullmatch(normalized):
        raise SchemaError(f"expected {context} SHA-256 must be 64 hex digits")
    if actual != normalized:
        raise SchemaError(f"{context} differs from the registered SHA-256")


def _validate_expected_sha256(actual: str, expected: str) -> None:
    _validate_sha256_identity(
        actual,
        expected,
        context="upstream WebArena task source",
    )


def _registered_url_map(
    value: Mapping[str, Any], *, required_tokens: frozenset[str]
) -> dict[str, str]:
    if set(value) != set(required_tokens):
        missing = sorted(required_tokens - set(value))
        extra = sorted(set(value) - required_tokens)
        raise SchemaError(
            "site URL mapping must exactly cover task-source placeholders "
            f"(missing={missing}, extra={extra})"
        )
    result: dict[str, str] = {}
    for token in sorted(required_tokens):
        raw = value[token]
        if not isinstance(raw, str) or not raw.strip():
            raise SchemaError(f"site URL mapping {token} must be a nonempty URL")
        candidate = raw.strip().rstrip("/")
        parsed = urlsplit(candidate)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise SchemaError(
                f"site URL mapping {token} must be a credential-free HTTP(S) base URL"
            )
        if _URL_TOKEN.search(candidate):
            raise SchemaError(f"site URL mapping {token} contains an unresolved placeholder")
        result[token] = candidate
    return result


def load_url_map(path: str | Path) -> dict[str, str]:
    """Load an operator-owned URL map without accepting credential material."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise SchemaError("WebArena URL-map file must not be a symlink")
    value = read_json(source)
    nested = value.get("url_map")
    if nested is not None and set(value) != {"url_map"}:
        raise SchemaError(
            "wrapped WebArena URL-map file may contain only the url_map field"
        )
    mapping = nested if nested is not None else value
    if not isinstance(mapping, Mapping):
        raise SchemaError("WebArena URL-map file must contain a JSON mapping")
    return _registered_url_map(mapping, required_tokens=PINNED_REQUIRED_URL_TOKENS)


def _tokens_in(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_URL_TOKEN.findall(value))
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            result.update(_tokens_in(key))
            result.update(_tokens_in(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = set()
        for item in value:
            result.update(_tokens_in(item))
        return result
    return set()


def _resolve_tokens(value: Any, url_map: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        resolved = value
        for token in sorted(url_map, key=lambda item: (-len(item), item)):
            resolved = resolved.replace(token, url_map[token])
        return resolved
    if isinstance(value, Mapping):
        return {str(key): _resolve_tokens(item, url_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_tokens(item, url_map) for item in value]
    return value


def _nonempty_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context} must be a nonempty string")
    return value


def _runtime_start_state(
    resolved: Mapping[str, Any], *, upstream_index: int
) -> dict[str, Any]:
    """Return the exact six-field, oracle-blind WebArena reset projection."""

    missing = sorted(WEBARENA_RUNTIME_START_STATE_FIELDS - set(resolved))
    if missing:
        raise SchemaError(
            f"task {upstream_index} lacks WebArena reset fields: {missing}"
        )
    candidate = {
        key: resolved[key] for key in WEBARENA_RUNTIME_START_STATE_FIELDS
    }
    try:
        # This is a dependency-light data contract; it imports no browser or
        # model backend.  Re-serializing also fixes the one canonical reset
        # representation used by the task snapshot and runtime receipt.
        from web_agent.runtime.contracts import RuntimeStartState

        return RuntimeStartState.from_webarena_mapping(
            candidate
        ).to_webarena_mapping()
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"task {upstream_index} has an invalid WebArena runtime start state: {exc}"
        ) from exc


def build_public_pilot_task_export(
    *,
    source: str | Path,
    registry_path: str | Path,
    site_url_map: Mapping[str, Any],
    snapshot_id: str,
    benchmark_version: str,
    task_definition_version: str,
    evaluator_id: str,
    evaluator_version: str,
    expected_source_sha256: str = PINNED_TASK_SOURCE_SHA256,
    expected_container_sha256: str = PINNED_LIBWEBARENA_WHEEL_SHA256,
    authorized_raw_json_sha256: str | None = None,
    archive_member: str = PINNED_TASK_MEMBER,
) -> dict[str, Any]:
    """Return the complete, sealed 50-task handoff export.

    The result intentionally contains official evaluator configuration.  It is
    an evaluator/handoff artifact, not a runtime input; the production runner
    constructs its browser policy input only through its oracle-blind task
    projection.
    """

    snapshot = _nonempty_text(snapshot_id, context="snapshot_id").strip()
    benchmark = _nonempty_text(
        benchmark_version, context="benchmark_version"
    ).strip()
    task_version = _nonempty_text(
        task_definition_version, context="task_definition_version"
    ).strip()
    evaluator_name = _nonempty_text(evaluator_id, context="evaluator_id").strip()
    evaluator_release = _nonempty_text(
        evaluator_version, context="evaluator_version"
    ).strip()
    for context, value in (
        ("snapshot_id", snapshot),
        ("benchmark_version", benchmark),
        ("task_definition_version", task_version),
        ("evaluator_id", evaluator_name),
        ("evaluator_version", evaluator_release),
    ):
        if "unknown" in value.casefold() or _URL_TOKEN.search(value):
            raise SchemaError(f"{context} must be a measured, non-placeholder identity")

    authority = load_pilot_task_exclusion_authority(registry_path)
    if authority.benchmark != "webarena":
        raise SchemaError("pilot task registry is not the WebArena exclusion authority")
    payload, source_record = _task_source_bytes(
        source,
        archive_member=archive_member,
        expected_container_sha256=expected_container_sha256,
        authorized_raw_json_sha256=authorized_raw_json_sha256,
    )
    _validate_expected_sha256(source_record["task_source_sha256"], expected_source_sha256)
    raw_rows = _parse_task_rows(payload)

    by_id: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(raw_rows):
        task_id = row.get("task_id")
        if type(task_id) is not int or task_id < 0:
            raise SchemaError(f"upstream task row {position} has an invalid task_id")
        if task_id in by_id:
            raise SchemaError(f"upstream task source repeats task_id {task_id}")
        by_id[task_id] = row
    missing = sorted(authority.upstream_indices - set(by_id))
    if missing:
        raise SchemaError(f"upstream task source lacks registered tasks: {missing}")

    selected_raw = [by_id[index] for index in sorted(authority.upstream_indices)]
    observed_tokens: set[str] = set()
    for row in selected_raw:
        observed_tokens.update(_tokens_in(row))
    if observed_tokens != PINNED_REQUIRED_URL_TOKENS:
        raise SchemaError(
            "registered WebArena 0--49 placeholder set changed "
            f"(observed={sorted(observed_tokens)})"
        )
    urls = _registered_url_map(
        site_url_map, required_tokens=PINNED_REQUIRED_URL_TOKENS
    )

    tasks: list[dict[str, Any]] = []
    for index, raw in zip(sorted(authority.upstream_indices), selected_raw, strict=True):
        resolved = _resolve_tokens(raw, urls)
        unresolved = _tokens_in(resolved)
        if unresolved:
            raise SchemaError(
                f"resolved WebArena task {index} still has placeholders: {sorted(unresolved)}"
            )
        instruction = _nonempty_text(
            resolved.get("intent"), context=f"task {index} intent"
        )
        sites = resolved.get("sites")
        if not isinstance(sites, list) or not sites or not all(
            isinstance(site, str) and site.strip() for site in sites
        ):
            raise SchemaError(f"task {index} must contain a nonempty sites array")
        start_url = _nonempty_text(
            resolved.get("start_url"), context=f"task {index} start_url"
        )
        evaluator_config = resolved.get("eval")
        if not isinstance(evaluator_config, Mapping) or not evaluator_config:
            raise SchemaError(f"task {index} lacks official evaluator configuration")
        start_state = _runtime_start_state(resolved, upstream_index=index)

        tasks.append(
            {
                "task_id": f"webarena.{index}",
                "upstream_index": index,
                "benchmark_task_id": str(index),
                "benchmark_task_version": task_version,
                "instruction": instruction,
                "start_state": start_state,
                "task_config": resolved,
                "evaluator": {
                    "evaluator_id": evaluator_name,
                    "evaluator_version": evaluator_release,
                    "config": dict(evaluator_config),
                },
            }
        )

    if len(tasks) != 50:
        raise SchemaError("resolved public WebArena pilot export must contain 50 tasks")
    task_set_sha256 = sha256_json(tasks)
    return {
        "schema_version": PUBLIC_PILOT_EXPORT_SCHEMA_VERSION,
        "record_type": PUBLIC_PILOT_EXPORT_RECORD_TYPE,
        "snapshot_id": snapshot,
        "benchmark": "webarena",
        "benchmark_version": benchmark,
        "task_definition_version": task_version,
        "browsergym_webarena_version": PINNED_BROWSERGYM_WEBARENA_VERSION,
        "libwebarena_version": PINNED_LIBWEBARENA_VERSION,
        "partition": "development",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "locked_test_content": False,
        "final_paper_evaluation_eligible": False,
        "selection_rule": PUBLIC_PILOT_SELECTION_RULE,
        "required_task_count": 50,
        "registry_manifest_id": authority.manifest_id,
        "registry_manifest_sha256": authority.registry_sha256,
        "source": source_record,
        "site_url_map_sha256": sha256_json(urls),
        "resolved_task_set_sha256": task_set_sha256,
        "tasks": tasks,
    }


def validate_public_pilot_task_export(
    value: Mapping[str, Any],
    *,
    source: str | Path,
    registry_path: str | Path,
    site_url_map: Mapping[str, Any],
    expected_source_sha256: str = PINNED_TASK_SOURCE_SHA256,
    expected_container_sha256: str = PINNED_LIBWEBARENA_WHEEL_SHA256,
    authorized_raw_json_sha256: str | None = None,
) -> dict[str, Any]:
    """Independently reproduce a handoff export from pinned upstream bytes.

    A claimed source hash inside an export is not authentication: an altered
    file could copy that string.  The handoff therefore reopens the pinned raw
    task source (or wheel member), resolves the supplied credential-free URL
    map again, and requires exact equality with the submitted export.
    """

    if not isinstance(value, Mapping):
        raise SchemaError("resolved WebArena task export must be an object")
    if value.get("benchmark_version") != PINNED_BROWSERGYM_WEBARENA_VERSION:
        raise SchemaError("task export does not use pinned BrowserGym WebArena 0.14.3")
    if value.get("task_definition_version") != PINNED_TASK_DEFINITION_VERSION:
        raise SchemaError("task export does not use the pinned libwebarena task definition")
    rows = value.get("tasks")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
        raise SchemaError("resolved WebArena task export lacks task records")
    evaluator = rows[0].get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError("resolved WebArena task export lacks evaluator identity")
    expected = build_public_pilot_task_export(
        source=source,
        registry_path=registry_path,
        site_url_map=site_url_map,
        snapshot_id=str(value.get("snapshot_id") or ""),
        benchmark_version=PINNED_BROWSERGYM_WEBARENA_VERSION,
        task_definition_version=PINNED_TASK_DEFINITION_VERSION,
        evaluator_id=str(evaluator.get("evaluator_id") or ""),
        evaluator_version=str(evaluator.get("evaluator_version") or ""),
        expected_source_sha256=expected_source_sha256,
        expected_container_sha256=expected_container_sha256,
        authorized_raw_json_sha256=authorized_raw_json_sha256,
        archive_member=PINNED_TASK_MEMBER,
    )
    if dict(value) != expected:
        raise SchemaError(
            "resolved WebArena task export differs from an independent rebuild "
            "of the pinned upstream 0--49 task bytes"
        )
    return expected


def write_public_pilot_task_export(
    output_path: str | Path, **kwargs: Any
) -> Path:
    """Build and atomically write a sealed public-task export."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite WebArena task export: {destination}")
    return atomic_write_json(destination, build_public_pilot_task_export(**kwargs))
