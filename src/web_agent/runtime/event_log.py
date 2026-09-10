"""Append-only, hash-chained JSONL event streams for runtime evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping
from urllib.parse import quote, quote_plus, unquote, unquote_plus, urlsplit, urlunsplit

from web_agent.runtime.contracts import (
    JsonValue,
    SCHEMA_VERSION,
    VersionedRecord,
    canonical_json,
)


RUNTIME_EVENT_FILES: Mapping[str, str] = {
    "actions": "actions.jsonl",
    "transitions": "transitions.jsonl",
    "recoveries": "recoveries.jsonl",
    "environment_events": "environment_events.jsonl",
    "memory_queries": "memory_queries.jsonl",
    "terminal_signals": "terminal_signals.jsonl",
}

_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "cookie",
    "authorization",
    "creditcard",
    "payment",
    "formvalue",
    "formcontent",
    "fieldvalue",
    "inputvalue",
    "typedtext",
)
_ACTION_VALUE_CONTAINERS = frozenset({"values", "parameters", "parameterhints"})
_PRIVATE_TEXT_CONTAINERS = frozenset(
    {"form", "forms", "formfields", "inputs"}
)
_URL_VALUE_KEYS = frozenset(
    {
        "url",
        "urls",
        "uri",
        "uris",
        "href",
        "hrefs",
        "location",
        "locations",
        "starturl",
        "currenturl",
        "targeturl",
        "sourceurl",
    }
)
_URL_SENSITIVE_KEY_PARTS = (
    *_SENSITIVE_KEY_PARTS,
    "apikey",
    "accesskey",
    "credential",
    "signature",
    "sessionid",
)
_URL_SENSITIVE_EXACT_KEYS = frozenset({"session", "sid"})
_URL_COMPONENT_SEPARATOR = re.compile(r"([&;])")
_URL_SECRET_MARKER = re.compile(
    r"(?:^|[^a-z0-9])(?:password|passwd|secret|token|cookie|authorization|"
    r"apikey|accesskey|credential|signature|session|sid)(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_MAX_NESTED_URL_REDACTION_DEPTH = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload(value: VersionedRecord | Mapping[str, Any]) -> dict[str, JsonValue]:
    raw = value.to_dict() if isinstance(value, VersionedRecord) else value
    redacted = _redact_mapping(raw)
    if "observation_record_sha256" in redacted:
        # The pre-redaction commitment remains the sealed causal identity.  A
        # second digest makes the privacy-preserving logged projection locally
        # reproducible, so changing any non-redacted observation field is
        # detectable without recovering private form contents.
        logged_projection = dict(redacted)
        logged_projection.pop("observation_record_sha256", None)
        logged_projection.pop("logged_observation_sha256", None)
        redacted["logged_observation_sha256"] = hashlib.sha256(
            canonical_json(logged_projection).encode("utf-8")
        ).hexdigest()
    return redacted


def _redact_mapping(
    value: Mapping[str, Any],
    *,
    protect_action_values: bool = False,
    protect_text_values: bool = False,
) -> dict[str, JsonValue]:
    action_type = str(value.get("action_type", "")).upper()
    form_control = _is_form_control(value)
    output: dict[str, JsonValue] = {}
    for key, item in value.items():
        normalised = "".join(
            character for character in str(key).lower() if character.isalnum()
        )
        nested_protection = protect_action_values or (
            action_type == "TYPE" and normalised in _ACTION_VALUE_CONTAINERS
        )
        nested_text_protection = protect_text_values or form_control or (
            normalised in _PRIVATE_TEXT_CONTAINERS
        )
        output[str(key)] = _redact(
            key,
            item,
            protect_action_values=nested_protection,
            protect_text_values=nested_text_protection,
        )
    return output


def _is_form_control(value: Mapping[str, Any]) -> bool:
    role = str(value.get("role", "")).strip().lower()
    tag = str(value.get("tag", value.get("tag_name", ""))).strip().lower()
    return (
        role in {"textbox", "searchbox", "combobox", "spinbutton"}
        or tag in {"input", "textarea", "select", "option"}
        or value.get("is_form_field") is True
    )


def _normalise_key(value: object) -> str:
    return "".join(
        character for character in str(value).lower() if character.isalnum()
    )


def _is_url_secret_key(value: object) -> bool:
    raw = str(value)
    normalised = _normalise_key(raw)
    exact_candidates = {
        _normalise_key(candidate)
        for candidate in re.split(r"[?&#;/]+", raw)
        if candidate
    }
    return bool(exact_candidates.intersection(_URL_SENSITIVE_EXACT_KEYS)) or any(
        part in normalised for part in _URL_SENSITIVE_KEY_PARTS
    )


def _contains_url_secret_marker(value: str) -> bool:
    """Detect credential syntax through a bounded number of encoding layers."""

    candidate = value
    for _ in range(_MAX_NESTED_URL_REDACTION_DEPTH + 1):
        if _URL_SECRET_MARKER.search(candidate):
            return True
        decoded = unquote_plus(candidate)
        if decoded == candidate:
            return False
        candidate = decoded
    # Fail closed when the encoding depth exceeds the registered inspection
    # bound; otherwise an attacker could hide a marker behind unlimited layers.
    return True


def _redact_url_component(component: str, *, depth: int) -> tuple[str, bool]:
    """Redact query/fragment pairs while preserving delimiters and benign text."""

    if not component:
        return component, False
    pieces = _URL_COMPONENT_SEPARATOR.split(component)
    output: list[str] = []
    changed = False
    for piece in pieces:
        if piece in {"&", ";"}:
            output.append(piece)
            continue
        if not piece:
            output.append(piece)
            continue
        if "=" not in piece:
            # Encoded key/value forms can occur in fragments. Decode one layer
            # only for inspection; retain the exact original when it is benign.
            decoded_piece = unquote_plus(piece)
            if "=" not in decoded_piece:
                output.append(piece)
                continue
            encoded_pair = True
            decoded_key, decoded_value = decoded_piece.split("=", 1)
        else:
            encoded_pair = False
            encoded_key, encoded_value = piece.split("=", 1)
            decoded_key = unquote_plus(encoded_key)
            decoded_value = unquote_plus(encoded_value)

        if _is_url_secret_key(decoded_key):
            digest = hashlib.sha256(decoded_value.encode("utf-8")).hexdigest()
            redacted_pair = f"{decoded_key}=sha256:{digest}"
            output.append(
                quote_plus(redacted_pair, safe="")
                if encoded_pair
                else f"{encoded_key}={quote_plus(f'sha256:{digest}', safe=':')}"
            )
            changed = True
            continue

        if depth >= _MAX_NESTED_URL_REDACTION_DEPTH:
            if _contains_url_secret_marker(decoded_value):
                digest = hashlib.sha256(decoded_value.encode("utf-8")).hexdigest()
                nested = f"redacted-nested-url-sha256:{digest}"
            else:
                output.append(piece)
                continue
        else:
            nested = _redacted_url_scalar(decoded_value, _depth=depth + 1)
        if nested == decoded_value:
            output.append(piece)
            continue
        redacted_pair = f"{decoded_key}={nested}"
        output.append(
            quote_plus(redacted_pair, safe="")
            if encoded_pair
            else f"{encoded_key}={quote_plus(nested, safe='')}"
        )
        changed = True
    return "".join(output), changed


def _redacted_url_scalar(value: str, *, _depth: int = 0) -> str:
    """Remove URL-embedded credentials while retaining useful location data."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        # A malformed value containing a credential marker is safer as one
        # opaque commitment than as a partially parsed secret-bearing string.
        if _contains_url_secret_marker(value):
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            return f"redacted-url-sha256:{digest}"
        return value

    changed = False
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        digest = hashlib.sha256(userinfo.encode("utf-8")).hexdigest()
        netloc = f"redacted-userinfo-sha256-{digest}@{host}"
        changed = True

    path_parts = parsed.path.split("/")
    path_changed = False
    redact_next_path_part = False
    for index, encoded_part in enumerate(path_parts):
        decoded_part = unquote(encoded_part)
        if redact_next_path_part and decoded_part:
            digest = hashlib.sha256(decoded_part.encode("utf-8")).hexdigest()
            path_parts[index] = quote(f"sha256:{digest}", safe=":")
            path_changed = True
            redact_next_path_part = False
            continue
        separator = "=" if "=" in decoded_part else ":" if ":" in decoded_part else None
        if separator is not None:
            path_key, path_value = decoded_part.split(separator, 1)
            if _is_url_secret_key(path_key):
                digest = hashlib.sha256(path_value.encode("utf-8")).hexdigest()
                path_parts[index] = quote(
                    f"{path_key}{separator}sha256:{digest}",
                    safe="=:",
                )
                path_changed = True
                continue
        if decoded_part and _is_url_secret_key(decoded_part):
            # Paths often encode credentials as /session_token/<value>. Keep
            # the marker useful for diagnostics but commit only the following
            # value. A trailing marker has no value to expose.
            redact_next_path_part = True
    path = "/".join(path_parts)
    changed = changed or path_changed

    query, query_changed = _redact_url_component(parsed.query, depth=_depth)
    fragment, fragment_changed = _redact_url_component(parsed.fragment, depth=_depth)
    changed = changed or query_changed or fragment_changed
    if not changed:
        return value
    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


def _redact(
    key: object,
    value: Any,
    *,
    protect_action_values: bool = False,
    protect_text_values: bool = False,
) -> JsonValue:
    normalised = _normalise_key(key)
    # Digests and opaque token hashes are already non-secret join keys.
    sensitive_key = any(part in normalised for part in _SENSITIVE_KEY_PARTS)
    protected_scalar = protect_action_values and not isinstance(
        value, (Mapping, list, tuple)
    )
    protected_text = protect_text_values and isinstance(value, str)
    if not normalised.endswith("sha256") and (
        sensitive_key or protected_scalar or protected_text
    ):
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return {"redacted": True, "sha256": digest}
    if isinstance(value, Mapping):
        return _redact_mapping(
            value,
            protect_action_values=protect_action_values,
            protect_text_values=protect_text_values,
        )
    if isinstance(value, (list, tuple)):
        return [
            _redact(
                key,
                item,
                protect_action_values=protect_action_values,
                protect_text_values=protect_text_values,
            )
            for item in value
        ]
    if isinstance(value, str) and normalised in _URL_VALUE_KEYS:
        return _redacted_url_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class HashChainedEventLog:
    """One new immutable stream. Existing non-empty streams are never appended."""

    def __init__(
        self,
        path: str | Path,
        *,
        stream: str,
        episode_id: str,
        context: Mapping[str, JsonValue] | None = None,
        timestamp_factory: Callable[[], str] = _utc_now,
    ) -> None:
        if stream not in RUNTIME_EVENT_FILES:
            raise ValueError(f"unregistered runtime stream: {stream}")
        self.path = Path(path)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"refusing to append to existing event log: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.stream = stream
        if not episode_id.strip():
            raise ValueError("event log episode_id cannot be empty")
        self.episode_id = episode_id
        self.context = dict(context or {})
        self._timestamp_factory = timestamp_factory
        self._sequence = 0
        self._previous_record_hash = "0" * 64

    def append(
        self,
        event_type: str,
        payload: VersionedRecord | Mapping[str, Any],
    ) -> dict[str, JsonValue]:
        if not event_type.strip():
            raise ValueError("event_type cannot be empty")
        self._sequence += 1
        body: dict[str, JsonValue] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "RuntimeEvent",
            "stream": self.stream,
            "episode_id": self.episode_id,
            **self.context,
            "event_id": f"{self.episode_id}:{self.stream}:{self._sequence}",
            "sequence": self._sequence,
            "previous_record_hash": self._previous_record_hash,
            "event_type": event_type,
            "timestamp_utc": self._timestamp_factory(),
            "payload": _payload(payload),
        }
        record_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        record = {**body, "record_hash": record_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous_record_hash = record_hash
        return record

    @property
    def records(self) -> int:
        return self._sequence

    @property
    def tail_sha256(self) -> str:
        return self._previous_record_hash

    @property
    def file_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EventLogVerification(VersionedRecord):
    path: str
    stream: str
    episode_id: str
    records: int
    tail_sha256: str
    file_sha256: str


def verify_event_log(
    path: str | Path,
    *,
    expected_stream: str | None = None,
    expected_episode_id: str | None = None,
) -> EventLogVerification:
    source = Path(path)
    previous = "0" * 64
    stream: str | None = None
    episode_id: str | None = None
    records = 0
    digest = hashlib.sha256()
    with source.open("rb") as binary:
        for line_number, raw in enumerate(binary, start=1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"event record is not an object at line {line_number}")
            claimed_hash = record.pop("record_hash", None)
            actual_hash = hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
            if claimed_hash != actual_hash:
                raise ValueError(f"event hash mismatch at {source}:{line_number}")
            if record.get("previous_record_hash") != previous:
                raise ValueError(f"event chain break at {source}:{line_number}")
            if record.get("sequence") != line_number:
                raise ValueError(f"non-monotonic event sequence at {source}:{line_number}")
            current_stream = str(record.get("stream", ""))
            if stream is None:
                stream = current_stream
            elif stream != current_stream:
                raise ValueError(f"mixed streams in {source}")
            current_episode = str(record.get("episode_id", ""))
            if not current_episode:
                raise ValueError(f"missing episode_id at {source}:{line_number}")
            if episode_id is None:
                episode_id = current_episode
            elif episode_id != current_episode:
                raise ValueError(f"mixed episode IDs in {source}")
            previous = actual_hash
            records += 1
    resolved_stream = stream or expected_stream or source.stem
    if expected_stream is not None and resolved_stream != expected_stream:
        raise ValueError(
            f"event stream mismatch: {resolved_stream!r} != {expected_stream!r}"
        )
    resolved_episode = episode_id or expected_episode_id or ""
    if expected_episode_id is not None and resolved_episode != expected_episode_id:
        raise ValueError(
            f"event episode mismatch: {resolved_episode!r} != {expected_episode_id!r}"
        )
    return EventLogVerification(
        path=str(source),
        stream=resolved_stream,
        episode_id=resolved_episode,
        records=records,
        tail_sha256=previous,
        file_sha256=digest.hexdigest(),
    )


class EpisodeEventLogs:
    """Open the canonical runtime streams under one episode runtime directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        episode_id: str,
        include_memory: bool,
        system_id: str | None = None,
        task_id: str | None = None,
        repeat_id: int | None = None,
        matched_seed: int | None = None,
        timestamp_factory: Callable[[], str] = _utc_now,
    ) -> None:
        destination = Path(root)
        self.episode_id = episode_id
        context: dict[str, JsonValue] = {}
        if system_id is not None:
            context["system_id"] = system_id
        if task_id is not None:
            context["task_id"] = task_id
        if repeat_id is not None:
            context["repeat_id"] = repeat_id
        if matched_seed is not None:
            context["matched_seed"] = matched_seed
        streams = tuple(
            stream
            for stream in RUNTIME_EVENT_FILES
            if include_memory or stream != "memory_queries"
        )
        self.logs = {
            stream: HashChainedEventLog(
                destination / RUNTIME_EVENT_FILES[stream],
                stream=stream,
                episode_id=episode_id,
                context=context,
                timestamp_factory=timestamp_factory,
            )
            for stream in streams
        }

    def append(
        self,
        stream: str,
        event_type: str,
        payload: VersionedRecord | Mapping[str, Any],
    ) -> dict[str, JsonValue]:
        try:
            log = self.logs[stream]
        except KeyError as exc:
            raise ValueError(f"stream is disabled/unregistered: {stream}") from exc
        return log.append(event_type, payload)

    def file_hashes(self) -> dict[str, str]:
        return {stream: log.file_sha256 for stream, log in self.logs.items()}
