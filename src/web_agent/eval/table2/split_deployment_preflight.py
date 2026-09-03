"""Fail-closed authorization evidence for split Table 2 deployment.

This module does not run a browser or an inference service. It binds positive,
separately measured evidence into a frozen campaign-eligibility artifact. A
failed DGX browser-host routing report is intentionally outside this schema and
can never authorize split execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from .common import SchemaError, canonical_json_bytes, sha256_json
from .webarena_preflight import validate_webarena_host_preflight


SINGLE_HOST_TOPOLOGY = "SINGLE_DGX_HOST"
SPLIT_HOST_TOPOLOGY = "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"

SPLIT_PREFLIGHT_SCHEMA_VERSION = "table2-webarena-split-deployment-preflight-v1"
SPLIT_PREFLIGHT_RECORD_TYPE = "WebArenaSplitDeploymentPreflight"
SPLIT_PREFLIGHT_EVIDENCE_LABEL = "PRE_CAMPAIGN_COMPATIBILITY_ONLY"
DGX_MODEL_RUNTIME_IDENTITY_SCHEMA_VERSION = "table2-dgx-model-runtime-identity-v1"
SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION = "table2-split-bridge-identity-v1"
SPLIT_BRIDGE_PROTOCOL_ID = "table2-oracle-blind-inference-bridge"
SPLIT_BRIDGE_PROTOCOL_VERSION = "v1"
SPLIT_TRANSCRIPT_SCHEMA_VERSION = "table2-split-bridge-transcript-v1"
SPLIT_CHAIN_ALGORITHM = "sha256_canonical_json_previous_entry_v1"

_FORBIDDEN_CATEGORIES = ("reward", "oracle", "evaluator")


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise SchemaError(
            f"{context} fields are not the registered closure "
            f"(missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected, key=str)})"
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, context: str) -> str:
    if not _is_sha256(value):
        raise SchemaError(f"{context} must be one lowercase SHA-256 digest")
    return str(value)


def _require_nonempty(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context} must be a nonempty string")
    return value


def _forbidden_field_counts(value: Any) -> dict[str, int]:
    counts = {category: 0 for category in _FORBIDDEN_CATEGORIES}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).lower().replace("-", "_").replace(" ", "_")
                for category in _FORBIDDEN_CATEGORIES:
                    if category in normalized:
                        counts[category] += 1
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return counts


def _validate_zero_counts(value: object, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be a mapping")
    _require_exact_fields(value, set(_FORBIDDEN_CATEGORIES), context=context)
    normalized: dict[str, int] = {}
    for category in _FORBIDDEN_CATEGORIES:
        count = value.get(category)
        if type(count) is not int or count != 0:
            raise SchemaError(f"{context}.{category} must be exactly zero")
        normalized[category] = count
    return normalized


def build_dgx_model_runtime_identity(
    *,
    host_identity_sha256: str,
    runtime_identity: Mapping[str, Any],
    runtime_source_set_sha256: str,
    runtime_environment_sha256: str,
) -> dict[str, Any]:
    """Bind an already measured DGX runtime into the registered identity schema."""

    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise SchemaError("DGX nested runtime_identity must be a nonempty mapping")
    try:
        canonical_json_bytes(runtime_identity)
    except (TypeError, ValueError) as exc:
        raise SchemaError("DGX nested runtime_identity is not canonical JSON") from exc
    identity = {
        "schema_version": DGX_MODEL_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "record_type": "DGXModelRuntimeIdentity",
        "host_identity_sha256": _require_sha256(
            host_identity_sha256, context="DGX host identity SHA-256"
        ),
        "model_seed": 42,
        "runtime_identity": deepcopy(dict(runtime_identity)),
        "runtime_identity_sha256": sha256_json(runtime_identity),
        "runtime_source_set_sha256": _require_sha256(
            runtime_source_set_sha256, context="DGX runtime source-set SHA-256"
        ),
        "runtime_environment_sha256": _require_sha256(
            runtime_environment_sha256, context="DGX runtime environment SHA-256"
        ),
        "evaluation_mode": True,
        "weights_mutated": False,
    }
    return _validate_dgx_model_runtime_identity(identity, expected=identity)


def _validate_dgx_model_runtime_identity(
    identity: object, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise SchemaError("DGX model-runtime identity must be a mapping")
    _require_exact_fields(
        identity,
        {
            "schema_version",
            "record_type",
            "host_identity_sha256",
            "model_seed",
            "runtime_identity",
            "runtime_identity_sha256",
            "runtime_source_set_sha256",
            "runtime_environment_sha256",
            "evaluation_mode",
            "weights_mutated",
        },
        context="DGX model-runtime identity",
    )
    exact_values = {
        "schema_version": DGX_MODEL_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "record_type": "DGXModelRuntimeIdentity",
        "model_seed": 42,
        "evaluation_mode": True,
        "weights_mutated": False,
    }
    for field, registered in exact_values.items():
        actual = identity.get(field)
        if type(actual) is not type(registered) or actual != registered:
            raise SchemaError(f"DGX model-runtime identity {field} is not registered")
    for field in (
        "host_identity_sha256",
        "runtime_identity_sha256",
        "runtime_source_set_sha256",
        "runtime_environment_sha256",
    ):
        _require_sha256(identity.get(field), context=f"DGX runtime {field}")
    nested = identity.get("runtime_identity")
    if not isinstance(nested, Mapping) or not nested:
        raise SchemaError("DGX nested runtime_identity must be a nonempty mapping")
    try:
        canonical_json_bytes(nested)
        expected_bytes = canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise SchemaError("DGX model-runtime identity is not canonical JSON") from exc
    if identity.get("runtime_identity_sha256") != sha256_json(nested):
        raise SchemaError("DGX nested runtime identity hash mismatch")
    if canonical_json_bytes(identity) != expected_bytes:
        raise SchemaError("split DGX runtime identity differs from frozen expectation")
    return deepcopy(dict(identity))


def _validate_bridge_identity(
    identity: object,
    *,
    expected: Mapping[str, Any],
    browser_host_identity_sha256: str,
    dgx_model_runtime_identity_sha256: str,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise SchemaError("split bridge identity must be a mapping")
    _require_exact_fields(
        identity,
        {
            "schema_version",
            "record_type",
            "bridge_id",
            "bridge_version",
            "protocol_id",
            "protocol_version",
            "source_files",
            "source_set_sha256",
            "browser_host_identity_sha256",
            "browser_endpoint_identity_sha256",
            "dgx_endpoint_identity_sha256",
            "dgx_model_runtime_identity_sha256",
            "transport_identity_sha256",
            "request_schema_sha256",
            "response_schema_sha256",
            "reward_fields_permitted",
            "oracle_fields_permitted",
            "evaluator_fields_permitted",
        },
        context="split bridge identity",
    )
    exact_values = {
        "schema_version": SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION,
        "record_type": "OracleBlindInferenceBridgeIdentity",
        "protocol_id": SPLIT_BRIDGE_PROTOCOL_ID,
        "protocol_version": SPLIT_BRIDGE_PROTOCOL_VERSION,
        "browser_host_identity_sha256": browser_host_identity_sha256,
        "dgx_model_runtime_identity_sha256": dgx_model_runtime_identity_sha256,
        "reward_fields_permitted": False,
        "oracle_fields_permitted": False,
        "evaluator_fields_permitted": False,
    }
    for field, registered in exact_values.items():
        actual = identity.get(field)
        if type(actual) is not type(registered) or actual != registered:
            raise SchemaError(f"split bridge identity {field} is not registered")
    for field in ("bridge_id", "bridge_version"):
        _require_nonempty(identity.get(field), context=f"split bridge {field}")
    for field in (
        "source_set_sha256",
        "browser_endpoint_identity_sha256",
        "dgx_endpoint_identity_sha256",
        "transport_identity_sha256",
        "request_schema_sha256",
        "response_schema_sha256",
    ):
        _require_sha256(identity.get(field), context=f"split bridge {field}")

    source_files = identity.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise SchemaError("split bridge source_files must be a nonempty array")
    normalized_sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in source_files:
        if not isinstance(row, Mapping):
            raise SchemaError("split bridge source row must be a mapping")
        _require_exact_fields(
            row, {"relative_path", "sha256"}, context="split bridge source row"
        )
        relative = _require_nonempty(
            row.get("relative_path"), context="split bridge source relative_path"
        )
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or relative in seen:
            raise SchemaError("split bridge source paths must be safe and unique")
        seen.add(relative)
        digest = _require_sha256(
            row.get("sha256"), context=f"split bridge source {relative} sha256"
        )
        normalized_sources.append({"relative_path": relative, "sha256": digest})
    if normalized_sources != sorted(
        normalized_sources, key=lambda row: row["relative_path"]
    ):
        raise SchemaError("split bridge source_files must be path-sorted")
    if identity.get("source_set_sha256") != sha256_json(normalized_sources):
        raise SchemaError("split bridge source-set hash mismatch")
    try:
        expected_mapping = dict(expected)
        expected_bytes = canonical_json_bytes(expected_mapping)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            "expected split bridge identity is not canonical JSON"
        ) from exc
    if canonical_json_bytes(identity) != expected_bytes:
        raise SchemaError(
            "split bridge identity differs from the frozen expected identity"
        )
    return deepcopy(dict(identity))


def _chain_seed(
    *,
    local_browser_preflight_sha256: str,
    dgx_model_runtime_identity_sha256: str,
    bridge_identity_sha256: str,
) -> str:
    return sha256_json(
        {
            "schema_version": SPLIT_TRANSCRIPT_SCHEMA_VERSION,
            "local_browser_preflight_sha256": local_browser_preflight_sha256,
            "dgx_model_runtime_identity_sha256": (
                dgx_model_runtime_identity_sha256
            ),
            "bridge_identity_sha256": bridge_identity_sha256,
        }
    )


def build_split_bridge_transcript(
    *,
    exchanges: Sequence[Mapping[str, Any]],
    local_browser_preflight_sha256: str,
    dgx_model_runtime_identity_sha256: str,
    bridge_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash real request/response payloads into a redacted append-only chain."""

    if isinstance(exchanges, (str, bytes)) or not isinstance(exchanges, Sequence):
        raise SchemaError("split bridge exchanges must be a sequence")
    if not exchanges:
        raise SchemaError("split bridge preflight requires at least one exchange")
    _require_sha256(
        local_browser_preflight_sha256,
        context="local browser preflight SHA-256",
    )
    _require_sha256(
        dgx_model_runtime_identity_sha256,
        context="DGX model-runtime identity SHA-256",
    )
    bridge_identity_sha256 = sha256_json(bridge_identity)
    previous = _chain_seed(
        local_browser_preflight_sha256=local_browser_preflight_sha256,
        dgx_model_runtime_identity_sha256=dgx_model_runtime_identity_sha256,
        bridge_identity_sha256=bridge_identity_sha256,
    )
    entries: list[dict[str, Any]] = []
    for sequence_id, exchange in enumerate(exchanges):
        if not isinstance(exchange, Mapping):
            raise SchemaError("split bridge exchange must be a mapping")
        _require_exact_fields(
            exchange, {"request", "response"}, context="split bridge exchange"
        )
        request = exchange.get("request")
        response = exchange.get("response")
        if not isinstance(request, Mapping) or not request:
            raise SchemaError("split bridge request must be a nonempty mapping")
        if not isinstance(response, Mapping) or not response:
            raise SchemaError("split bridge response must be a nonempty mapping")
        try:
            canonical_json_bytes(request)
            canonical_json_bytes(response)
        except (TypeError, ValueError) as exc:
            raise SchemaError("split bridge payload is not canonical JSON") from exc
        request_counts = _forbidden_field_counts(request)
        response_counts = _forbidden_field_counts(response)
        _validate_zero_counts(
            request_counts, context="split bridge request forbidden-field counts"
        )
        _validate_zero_counts(
            response_counts, context="split bridge response forbidden-field counts"
        )
        entry = {
            "sequence_id": sequence_id,
            "previous_entry_sha256": previous,
            "browser_endpoint_identity_sha256": bridge_identity.get(
                "browser_endpoint_identity_sha256"
            ),
            "dgx_endpoint_identity_sha256": bridge_identity.get(
                "dgx_endpoint_identity_sha256"
            ),
            "request_sha256": sha256_json(request),
            "response_sha256": sha256_json(response),
            "request_forbidden_field_counts": request_counts,
            "response_forbidden_field_counts": response_counts,
        }
        entry["entry_sha256"] = sha256_json(entry)
        previous = entry["entry_sha256"]
        entries.append(entry)
    return {
        "schema_version": SPLIT_TRANSCRIPT_SCHEMA_VERSION,
        "record_type": "OracleBlindInferenceBridgeTranscript",
        "chain_algorithm": SPLIT_CHAIN_ALGORITHM,
        "append_only": True,
        "chain_seed_sha256": entries[0]["previous_entry_sha256"],
        "entry_count": len(entries),
        "entries": entries,
        "terminal_entry_sha256": previous,
        "forbidden_field_totals": {
            category: sum(
                entry[direction][category]
                for entry in entries
                for direction in (
                    "request_forbidden_field_counts",
                    "response_forbidden_field_counts",
                )
            )
            for category in _FORBIDDEN_CATEGORIES
        },
    }


def _validate_split_bridge_transcript(
    transcript: object,
    *,
    local_browser_preflight_sha256: str,
    dgx_model_runtime_identity_sha256: str,
    bridge_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(transcript, Mapping):
        raise SchemaError("split bridge transcript must be a mapping")
    _require_exact_fields(
        transcript,
        {
            "schema_version",
            "record_type",
            "chain_algorithm",
            "append_only",
            "chain_seed_sha256",
            "entry_count",
            "entries",
            "terminal_entry_sha256",
            "forbidden_field_totals",
        },
        context="split bridge transcript",
    )
    expected_scalars = {
        "schema_version": SPLIT_TRANSCRIPT_SCHEMA_VERSION,
        "record_type": "OracleBlindInferenceBridgeTranscript",
        "chain_algorithm": SPLIT_CHAIN_ALGORITHM,
        "append_only": True,
    }
    for field, expected in expected_scalars.items():
        actual = transcript.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise SchemaError(f"split bridge transcript {field} is not registered")
    expected_seed = _chain_seed(
        local_browser_preflight_sha256=local_browser_preflight_sha256,
        dgx_model_runtime_identity_sha256=dgx_model_runtime_identity_sha256,
        bridge_identity_sha256=sha256_json(bridge_identity),
    )
    if transcript.get("chain_seed_sha256") != expected_seed:
        raise SchemaError("split bridge transcript chain seed is not identity-bound")
    entries = transcript.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SchemaError("split bridge transcript must contain at least one exchange")
    if type(transcript.get("entry_count")) is not int or transcript.get(
        "entry_count"
    ) != len(entries):
        raise SchemaError("split bridge transcript entry count mismatch")
    browser_endpoint = bridge_identity["browser_endpoint_identity_sha256"]
    dgx_endpoint = bridge_identity["dgx_endpoint_identity_sha256"]
    previous = expected_seed
    expected_entry_fields = {
        "sequence_id",
        "previous_entry_sha256",
        "browser_endpoint_identity_sha256",
        "dgx_endpoint_identity_sha256",
        "request_sha256",
        "response_sha256",
        "request_forbidden_field_counts",
        "response_forbidden_field_counts",
        "entry_sha256",
    }
    for sequence_id, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise SchemaError("split bridge transcript entry must be a mapping")
        _require_exact_fields(
            entry, expected_entry_fields, context="split bridge transcript entry"
        )
        if type(entry.get("sequence_id")) is not int or entry.get(
            "sequence_id"
        ) != sequence_id:
            raise SchemaError("split bridge transcript sequence is not contiguous")
        if entry.get("previous_entry_sha256") != previous:
            raise SchemaError("split bridge transcript previous-entry hash mismatch")
        if (
            entry.get("browser_endpoint_identity_sha256") != browser_endpoint
            or entry.get("dgx_endpoint_identity_sha256") != dgx_endpoint
        ):
            raise SchemaError("split bridge transcript endpoint identity mismatch")
        for field in ("request_sha256", "response_sha256"):
            _require_sha256(entry.get(field), context=f"split bridge entry {field}")
        _validate_zero_counts(
            entry.get("request_forbidden_field_counts"),
            context="split bridge request forbidden-field counts",
        )
        _validate_zero_counts(
            entry.get("response_forbidden_field_counts"),
            context="split bridge response forbidden-field counts",
        )
        unhashed = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if entry.get("entry_sha256") != sha256_json(unhashed):
            raise SchemaError("split bridge transcript entry hash mismatch")
        previous = str(entry["entry_sha256"])
    if transcript.get("terminal_entry_sha256") != previous:
        raise SchemaError("split bridge transcript terminal hash mismatch")
    _validate_zero_counts(
        transcript.get("forbidden_field_totals"),
        context="split bridge transcript forbidden-field totals",
    )
    return deepcopy(dict(transcript))


def build_split_deployment_preflight(
    *,
    local_browser_preflight: Mapping[str, Any],
    service_url_map: Mapping[str, Any],
    dgx_model_runtime_identity: Mapping[str, Any],
    bridge_identity: Mapping[str, Any],
    exchanges: Sequence[Mapping[str, Any]],
    expected_live_reset_task_index: int | None = None,
) -> dict[str, Any]:
    """Bind supplied positive measurements into one split-eligibility artifact."""

    local = validate_webarena_host_preflight(
        local_browser_preflight,
        service_url_map=service_url_map,
        expected_live_reset_task_index=expected_live_reset_task_index,
    )
    dgx = _validate_dgx_model_runtime_identity(
        dgx_model_runtime_identity,
        expected=dgx_model_runtime_identity,
    )
    local_sha = sha256_json(local)
    dgx_sha = sha256_json(dgx)
    browser_host_sha = sha256_json(local["host"])
    bridge = _validate_bridge_identity(
        bridge_identity,
        expected=bridge_identity,
        browser_host_identity_sha256=browser_host_sha,
        dgx_model_runtime_identity_sha256=dgx_sha,
    )
    transcript = build_split_bridge_transcript(
        exchanges=exchanges,
        local_browser_preflight_sha256=local_sha,
        dgx_model_runtime_identity_sha256=dgx_sha,
        bridge_identity=bridge,
    )
    result = {
        "schema_version": SPLIT_PREFLIGHT_SCHEMA_VERSION,
        "record_type": SPLIT_PREFLIGHT_RECORD_TYPE,
        "status": "PASS",
        "campaign_eligible": True,
        "deployment_topology": SPLIT_HOST_TOPOLOGY,
        "evidence_label": SPLIT_PREFLIGHT_EVIDENCE_LABEL,
        "paper_table_status": "N/R",
        "failed_dgx_routing_report_used": False,
        "authorization_inputs": [
            "LOCAL_BROWSER_HOST_PASS",
            "PINNED_DGX_MODEL_RUNTIME_IDENTITY",
            "PINNED_ORACLE_BLIND_BRIDGE_IDENTITY",
            "HASH_CHAINED_ORACLE_BLIND_BRIDGE_TRANSCRIPT",
        ],
        "local_browser_preflight": local,
        "local_browser_preflight_sha256": local_sha,
        "dgx_model_runtime_identity": dgx,
        "dgx_model_runtime_identity_sha256": dgx_sha,
        "bridge_identity": bridge,
        "bridge_identity_sha256": sha256_json(bridge),
        "bridge_transcript": transcript,
        "bridge_transcript_sha256": sha256_json(transcript),
    }
    return validate_split_deployment_preflight(
        result,
        service_url_map=service_url_map,
        expected_live_reset_task_index=expected_live_reset_task_index,
        expected_dgx_model_runtime_identity=dgx,
        expected_bridge_identity=bridge_identity,
    )


def validate_split_deployment_preflight(
    evidence: Mapping[str, Any],
    *,
    service_url_map: Mapping[str, Any],
    expected_dgx_model_runtime_identity: Mapping[str, Any],
    expected_bridge_identity: Mapping[str, Any],
    expected_live_reset_task_index: int | None = None,
) -> dict[str, Any]:
    """Validate one frozen split artifact against externally pinned identities."""

    if not isinstance(evidence, Mapping):
        raise SchemaError("split deployment preflight evidence must be a mapping")
    _require_exact_fields(
        evidence,
        {
            "schema_version",
            "record_type",
            "status",
            "campaign_eligible",
            "deployment_topology",
            "evidence_label",
            "paper_table_status",
            "failed_dgx_routing_report_used",
            "authorization_inputs",
            "local_browser_preflight",
            "local_browser_preflight_sha256",
            "dgx_model_runtime_identity",
            "dgx_model_runtime_identity_sha256",
            "bridge_identity",
            "bridge_identity_sha256",
            "bridge_transcript",
            "bridge_transcript_sha256",
        },
        context="split deployment preflight",
    )
    expected_scalars = {
        "schema_version": SPLIT_PREFLIGHT_SCHEMA_VERSION,
        "record_type": SPLIT_PREFLIGHT_RECORD_TYPE,
        "status": "PASS",
        "campaign_eligible": True,
        "deployment_topology": SPLIT_HOST_TOPOLOGY,
        "evidence_label": SPLIT_PREFLIGHT_EVIDENCE_LABEL,
        "paper_table_status": "N/R",
        "failed_dgx_routing_report_used": False,
        "authorization_inputs": [
            "LOCAL_BROWSER_HOST_PASS",
            "PINNED_DGX_MODEL_RUNTIME_IDENTITY",
            "PINNED_ORACLE_BLIND_BRIDGE_IDENTITY",
            "HASH_CHAINED_ORACLE_BLIND_BRIDGE_TRANSCRIPT",
        ],
    }
    for field, registered in expected_scalars.items():
        actual = evidence.get(field)
        if type(actual) is not type(registered) or actual != registered:
            raise SchemaError(f"split deployment preflight {field} is not registered")

    local = validate_webarena_host_preflight(
        evidence.get("local_browser_preflight"),
        service_url_map=service_url_map,
        expected_live_reset_task_index=expected_live_reset_task_index,
    )
    local_sha = sha256_json(local)
    if evidence.get("local_browser_preflight_sha256") != local_sha:
        raise SchemaError("split local-browser preflight hash mismatch")
    if not isinstance(expected_dgx_model_runtime_identity, Mapping) or not (
        expected_dgx_model_runtime_identity
    ):
        raise SchemaError("split validation requires expected DGX runtime identity")
    actual_dgx = _validate_dgx_model_runtime_identity(
        evidence.get("dgx_model_runtime_identity"),
        expected=expected_dgx_model_runtime_identity,
    )
    dgx_sha = sha256_json(actual_dgx)
    if evidence.get("dgx_model_runtime_identity_sha256") != dgx_sha:
        raise SchemaError("split DGX model-runtime identity hash mismatch")
    if (
        not isinstance(expected_bridge_identity, Mapping)
        or not expected_bridge_identity
    ):
        raise SchemaError("split validation requires expected bridge identity")
    bridge = _validate_bridge_identity(
        evidence.get("bridge_identity"),
        expected=expected_bridge_identity,
        browser_host_identity_sha256=sha256_json(local["host"]),
        dgx_model_runtime_identity_sha256=dgx_sha,
    )
    if evidence.get("bridge_identity_sha256") != sha256_json(bridge):
        raise SchemaError("split bridge identity hash mismatch")
    transcript = _validate_split_bridge_transcript(
        evidence.get("bridge_transcript"),
        local_browser_preflight_sha256=local_sha,
        dgx_model_runtime_identity_sha256=dgx_sha,
        bridge_identity=bridge,
    )
    if evidence.get("bridge_transcript_sha256") != sha256_json(transcript):
        raise SchemaError("split bridge transcript hash mismatch")
    return deepcopy(dict(evidence))


def validate_webarena_deployment_preflight(
    evidence: Mapping[str, Any],
    *,
    deployment_topology: str,
    service_url_map: Mapping[str, Any],
    expected_live_reset_task_index: int | None = None,
    expected_dgx_model_runtime_identity: Mapping[str, Any] | None = None,
    expected_bridge_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch to exactly one registered topology-specific validator."""

    if deployment_topology == SINGLE_HOST_TOPOLOGY:
        return validate_webarena_host_preflight(
            evidence,
            service_url_map=service_url_map,
            expected_live_reset_task_index=expected_live_reset_task_index,
        )
    if deployment_topology == SPLIT_HOST_TOPOLOGY:
        if not isinstance(expected_dgx_model_runtime_identity, Mapping):
            raise SchemaError("split topology requires expected DGX runtime identity")
        if not isinstance(expected_bridge_identity, Mapping):
            raise SchemaError("split topology requires expected bridge identity")
        return validate_split_deployment_preflight(
            evidence,
            service_url_map=service_url_map,
            expected_live_reset_task_index=expected_live_reset_task_index,
            expected_dgx_model_runtime_identity=expected_dgx_model_runtime_identity,
            expected_bridge_identity=expected_bridge_identity,
        )
    raise SchemaError(
        f"unregistered WebArena deployment topology: {deployment_topology!r}"
    )
