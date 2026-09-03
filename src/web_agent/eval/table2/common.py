"""Dependency-light serialization and validation helpers for Table 2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = "table2.v1"


class Table2Error(RuntimeError):
    """Base error for fail-closed Table 2 operations."""


class SchemaError(Table2Error):
    """Raised when an artifact does not satisfy its registered schema."""


def as_mapping(value: Any) -> dict[str, Any]:
    """Convert a runtime dataclass or mapping to a detached plain dictionary."""

    if is_dataclass(value):
        return dict(asdict(value))
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError(f"expected a mapping-compatible record, got {type(value)!r}")


def json_ready(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_ready(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinity are forbidden in campaign artifacts")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any, *, mode: int | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_ready(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def append_jsonl(path: str | Path, record: Mapping[str, Any], *, mode: int | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(destination, flags, mode or 0o644)
    try:
        payload = canonical_json_bytes(record) + b"\n"
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    if mode is not None:
        os.chmod(destination, mode)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SchemaError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, or CSV records without guessing field semantics."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(source)
    if suffix == ".csv":
        with source.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, dict):
        rows = value.get("records", value.get("tasks"))
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            return [dict(row) for row in rows]
        return [dict(value)]
    raise SchemaError(f"unsupported record container: {source}")


def require_keys(record: Mapping[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = sorted(key for key in keys if key not in record)
    if missing:
        raise SchemaError(f"{context} is missing required keys: {', '.join(missing)}")


def strict_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise SchemaError(f"{context} must be a boolean, got {value!r}")


def safe_relative_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SchemaError(f"path must remain inside its artifact root: {value}")
    return candidate


def stable_int_seed(*parts: Any) -> int:
    """Stable unsigned 32-bit seed independent of Python hash randomization."""

    digest = hashlib.sha256(canonical_json_bytes(list(parts))).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def stable_sort(records: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return sorted((dict(record) for record in records), key=lambda row: tuple(str(row.get(k, "")) for k in keys))
