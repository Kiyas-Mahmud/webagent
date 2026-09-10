"""Exact identity for a selected full-run resolved configuration artifact.

The Table 2 handoff uses two deliberately different identities:

* ``payload_sha256`` hashes the artifact bytes exactly as supplied; and
* ``record_sha256`` hashes the parsed mapping in canonical JSON form.

Keeping both prevents runtime-only path overrides from being mistaken for the
configuration that produced the validation-selected checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from web_agent.eval.table2.common import (
    canonical_json_bytes,
    json_ready,
    sha256_bytes,
    sha256_file,
)


class ResolvedConfigIdentityError(ValueError):
    """Raised when a resolved-config artifact has no unambiguous identity."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResolvedConfigIdentityError(
                f"resolved configuration contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _require_string_keys(value: Any, *, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ResolvedConfigIdentityError(
                    f"{location} mapping keys must be non-empty strings"
                )
            _require_string_keys(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_string_keys(item, location=f"{location}[{index}]")


def _canonical_mapping_bytes(value: Mapping[str, Any]) -> bytes:
    _require_string_keys(value)
    try:
        ready = json_ready(dict(value))
        if not isinstance(ready, dict):  # pragma: no cover - mapping guarantees this
            raise TypeError("resolved configuration did not normalize to an object")
        return canonical_json_bytes(ready)
    except (TypeError, ValueError) as exc:
        raise ResolvedConfigIdentityError(
            "resolved configuration must be finite, JSON-compatible data"
        ) from exc


@dataclass(frozen=True, slots=True)
class ResolvedConfigIdentity:
    """The exact bytes and canonical mapping for one resolved config."""

    source_path: Path
    mapping: dict[str, Any]
    payload_sha256: str
    record_sha256: str
    canonical_record_bytes: bytes


def load_resolved_config_identity(path: str | Path) -> ResolvedConfigIdentity:
    """Load a fully resolved JSON/YAML config without applying any overrides."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ResolvedConfigIdentityError(
            f"resolved configuration must be a regular non-symlink file: {source}"
        )
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ResolvedConfigIdentityError(
            f"cannot read resolved configuration: {source}"
        ) from exc
    try:
        if source.suffix.lower() == ".json":
            value = json.loads(text, object_pairs_hook=_unique_json_object)
        else:
            value = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except ResolvedConfigIdentityError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ResolvedConfigIdentityError(
            f"invalid resolved configuration {source}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ResolvedConfigIdentityError(
            f"resolved configuration must be a mapping: {source}"
        )
    mapping = dict(value)
    if "extends" in mapping:
        raise ResolvedConfigIdentityError(
            "selected resolved configuration must be fully materialized and "
            "cannot contain an extends directive"
        )
    canonical = _canonical_mapping_bytes(mapping)
    return ResolvedConfigIdentity(
        source_path=source.resolve(),
        mapping=mapping,
        payload_sha256=sha256_file(source),
        record_sha256=sha256_bytes(canonical),
        canonical_record_bytes=canonical,
    )


def assert_checkpoint_config_matches(
    saved_config: Mapping[str, Any],
    identity: ResolvedConfigIdentity,
) -> None:
    """Require the checkpoint-saved full config to equal the supplied mapping."""

    actual = _canonical_mapping_bytes(saved_config)
    if actual != identity.canonical_record_bytes:
        raise ResolvedConfigIdentityError(
            "checkpoint-saved full configuration differs from the selected "
            "resolved-config artifact: "
            f"expected {identity.record_sha256}, found {sha256_bytes(actual)}"
        )
