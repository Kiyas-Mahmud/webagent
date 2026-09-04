"""Fail-closed compatibility evidence for split Table 2 deployment.

This module does not run a browser or an inference service. It binds positive,
caller-supplied compatibility evidence into a frozen artifact. It does not
attest opaque payload semantics or endpoint origin, and it cannot authorize
split dispatch. A failed DGX browser-host routing report is intentionally
outside this schema.
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

SPLIT_PREFLIGHT_SCHEMA_VERSION = "table2-webarena-split-deployment-preflight-v2"
SPLIT_PREFLIGHT_RECORD_TYPE = "WebArenaSplitDeploymentPreflight"
SPLIT_PREFLIGHT_EVIDENCE_LABEL = "PRE_CAMPAIGN_COMPATIBILITY_ONLY"
DGX_MODEL_RUNTIME_IDENTITY_SCHEMA_VERSION = "table2-dgx-model-runtime-identity-v2"
SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION = "table2-split-bridge-identity-v2"
SPLIT_BRIDGE_PROTOCOL_ID = "table2-redacted-inference-bridge-compatibility"
SPLIT_BRIDGE_PROTOCOL_VERSION = "v2"
SPLIT_TRANSCRIPT_SCHEMA_VERSION = "table2-split-bridge-transcript-v2"
SPLIT_CHAIN_ALGORITHM = "sha256_canonical_json_previous_entry_v1"
SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION = "table2-split-bridge-request-v1"
SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION = "table2-split-bridge-response-v1"
SPLIT_TRANSCRIPT_CLAIM_SCOPE = (
    "REDACTED_EXACT_ENVELOPE_METADATA_NOT_PAYLOAD_CONTENT_OR_ORIGIN_PROOF"
)
SPLIT_DISPATCH_BLOCKER = (
    "BLOCKED_DGX_REMEASUREMENT_RECEIPT_AND_EXTERNAL_TRUST_ANCHOR_REQUIRED"
)

PC01_BACKBONE_ID = "Qwen/Qwen2-VL-2B-Instruct"
PC01_BACKBONE_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
PC01_BASE_SNAPSHOT_SHA256 = (
    "e002f8290faa3e9f44bf3099eac85a2445e17de738c5bb0cc10d342da837c46c"
)
PC01_CHECKPOINT_SHA256 = (
    "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a"
)
PC01_RESOLVED_CONFIG_SHA256 = (
    "d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f"
)
PC01_PROCESSOR_CONTRACT_SHA256 = (
    "b3c9f629a3c4ce6ab7bb71b27c7e0b560e3e21c51a19d1235fd4a507ba437797"
)

BRIDGE_OPERATIONS = frozenset(
    {
        "parameter_fallback",
        "post_action_diagnosis",
        "post_failure_embedding",
        "pre_action_prediction",
        "recovery_assessment",
    }
)
BRIDGE_REQUEST_SCHEMA = {
    "schema_version": SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION,
    "exact_fields": [
        "schema_version",
        "request_id",
        "operation",
        "payload_sha256",
    ],
    "registered_operations": sorted(BRIDGE_OPERATIONS),
    "payload_visibility": "HASH_ONLY_NOT_CONTENT_ATTESTATION",
}
BRIDGE_RESPONSE_SCHEMA = {
    "schema_version": SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION,
    "exact_fields": [
        "schema_version",
        "request_id",
        "operation",
        "payload_sha256",
    ],
    "registered_operations": sorted(BRIDGE_OPERATIONS),
    "payload_visibility": "HASH_ONLY_NOT_CONTENT_ATTESTATION",
}

_FORBIDDEN_CATEGORY_ALIASES = {
    "reward": ("reward", "score"),
    "oracle": (
        "oracle",
        "success",
        "outcome",
        "judgment",
        "judgement",
        "label",
        "verdict",
    ),
    "evaluator": ("evaluator", "verifier"),
}
_FORBIDDEN_CATEGORIES = tuple(_FORBIDDEN_CATEGORY_ALIASES)
_REQUIRED_DGX_DISTRIBUTIONS = frozenset(
    {"accelerate", "bitsandbytes", "peft", "torch", "transformers"}
)


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
    if any(
        marker in value.casefold() for marker in ("placeholder", "unknown", "tbd")
    ):
        raise SchemaError(f"{context} must not be a placeholder value")
    return value


def _validate_pc01_runtime_identity(value: object) -> dict[str, str]:
    """Validate the exact provisional PC-01 model/processor identity."""

    if not isinstance(value, Mapping):
        raise SchemaError("DGX nested runtime_identity must be a mapping")
    expected = {
        "backbone_id": PC01_BACKBONE_ID,
        "backbone_revision": PC01_BACKBONE_REVISION,
        "base_snapshot_sha256": PC01_BASE_SNAPSHOT_SHA256,
        "checkpoint_sha256": PC01_CHECKPOINT_SHA256,
        "resolved_config_sha256": PC01_RESOLVED_CONFIG_SHA256,
        "processor_contract_sha256": PC01_PROCESSOR_CONTRACT_SHA256,
    }
    _require_exact_fields(value, set(expected), context="DGX PC-01 runtime identity")
    for field, registered in expected.items():
        actual = value.get(field)
        if actual != registered:
            raise SchemaError(
                f"DGX PC-01 runtime identity {field} differs from the registered pilot"
            )
    return dict(expected)


def _validate_runtime_source_files(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise SchemaError("DGX runtime source_files must be a nonempty array")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise SchemaError("DGX runtime source row must be a mapping")
        _require_exact_fields(
            row, {"relative_path", "sha256"}, context="DGX runtime source row"
        )
        relative = _require_nonempty(
            row.get("relative_path"), context="DGX runtime source relative_path"
        )
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or relative == "."
            or relative in seen
        ):
            raise SchemaError(
                "DGX runtime source paths must be canonical, safe, and unique"
            )
        seen.add(relative)
        rows.append(
            {
                "relative_path": relative,
                "sha256": _require_sha256(
                    row.get("sha256"), context=f"DGX runtime source {relative} sha256"
                ),
            }
        )
    if rows != sorted(rows, key=lambda row: row["relative_path"]):
        raise SchemaError("DGX runtime source_files must be path-sorted")
    return rows


def _package_version(
    dependency_identity: Mapping[str, Any], distribution: str
) -> str:
    matches = [
        row["version"]
        for row in dependency_identity["packages"]
        if row["distribution"] == distribution
    ]
    if len(matches) != 1:
        raise SchemaError(f"DGX dependency inventory lacks unique {distribution}")
    return str(matches[0])


def _validate_runtime_environment(
    value: object, *, dependency_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate explicit runtime declarations and bind their nested identities."""

    if not isinstance(value, Mapping):
        raise SchemaError("DGX runtime_environment must be a mapping")
    fields = {
        "python_version",
        "python_executable_sha256",
        "torch_version",
        "transformers_version",
        "cuda_available",
        "cuda_runtime_version",
        "device_type",
        "device_name",
        "device_count",
        "container_digest",
    }
    _require_exact_fields(value, fields, context="DGX runtime environment")
    environment = {
        field: _require_nonempty(value.get(field), context=f"DGX runtime {field}")
        for field in (
            "python_version",
            "torch_version",
            "transformers_version",
            "cuda_runtime_version",
            "device_type",
            "device_name",
            "container_digest",
        )
    }
    environment["python_executable_sha256"] = _require_sha256(
        value.get("python_executable_sha256"),
        context="DGX runtime python_executable_sha256",
    )
    if type(value.get("cuda_available")) is not bool or value.get(
        "cuda_available"
    ) is not True:
        raise SchemaError("DGX runtime declaration requires cuda_available=true")
    if type(value.get("device_count")) is not int or value["device_count"] < 1:
        raise SchemaError("DGX runtime device_count must be a positive integer")
    environment["cuda_available"] = True
    environment["device_count"] = int(value["device_count"])
    if environment["device_type"] != "cuda":
        raise SchemaError("DGX runtime device_type must be cuda")
    container_digest = environment["container_digest"]
    if not container_digest.startswith("sha256:") or not _is_sha256(
        container_digest.removeprefix("sha256:")
    ):
        raise SchemaError(
            "DGX runtime container_digest must be a sha256:<64 lowercase hex> digest"
        )
    dependency_host = dependency_identity["host"]
    if (
        environment["python_version"] != dependency_host["python_version"]
        or environment["python_executable_sha256"]
        != dependency_host["python_executable_sha256"]
        or environment["torch_version"]
        != _package_version(dependency_identity, "torch")
        or environment["transformers_version"]
        != _package_version(dependency_identity, "transformers")
    ):
        raise SchemaError(
            "DGX runtime environment differs from dependency identity"
        )
    return environment


def _split_dispatch_requirement() -> dict[str, Any]:
    """Describe the deliberately unavailable external split-dispatch authority."""

    return {
        "schema_version": "table2-split-dgx-remeasurement-requirement-v1",
        "status": SPLIT_DISPATCH_BLOCKER,
        "required_phases": [
            "DGX_INFERENCE_SERVICE_STARTUP",
            "BEFORE_MODEL_LOAD",
            "BEFORE_EACH_PHYSICAL_BLOCK",
        ],
        "required_receipt_bindings": [
            "campaign_id",
            "block_id",
            "challenge_nonce",
            "semantic_dependency_lock_sha256",
            "dgx_host_identity_sha256",
            "dgx_runtime_identity_sha256",
            "phase",
            "issued_at_utc",
        ],
        "registered_receipt_schema_version": None,
        "registered_external_trust_anchor": None,
        "production_dispatch_authorized": False,
    }


def _validate_dgx_dependency_identity(value: object) -> dict[str, Any]:
    """Validate the supplied DGX Python/platform/package inventory."""

    if not isinstance(value, Mapping):
        raise SchemaError("DGX dependency identity must be a mapping")
    _require_exact_fields(
        value,
        {"host", "packages"},
        context="DGX dependency identity",
    )
    host = value.get("host")
    if not isinstance(host, Mapping):
        raise SchemaError("DGX dependency host must be a mapping")
    host_fields = {
        "system",
        "release",
        "machine",
        "python_version",
        "python_executable_sha256",
    }
    _require_exact_fields(host, host_fields, context="DGX dependency host")
    normalized_host = {
        field: _require_nonempty(host.get(field), context=f"DGX host {field}")
        for field in ("system", "release", "machine", "python_version")
    }
    normalized_host["python_executable_sha256"] = _require_sha256(
        host.get("python_executable_sha256"),
        context="DGX host python_executable_sha256",
    )

    rows = value.get("packages")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("DGX dependency packages must be a nonempty array")
    packages: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise SchemaError("DGX dependency package row must be a mapping")
        _require_exact_fields(
            row, {"distribution", "version"}, context="DGX dependency package row"
        )
        distribution = _require_nonempty(
            row.get("distribution"), context="DGX dependency distribution"
        )
        version = _require_nonempty(
            row.get("version"), context=f"DGX dependency {distribution} version"
        )
        if distribution in seen:
            raise SchemaError("DGX dependency distributions must be unique")
        seen.add(distribution)
        packages.append({"distribution": distribution, "version": version})
    if packages != sorted(packages, key=lambda row: row["distribution"]):
        raise SchemaError("DGX dependency packages must be distribution-sorted")
    if not _REQUIRED_DGX_DISTRIBUTIONS.issubset(seen):
        missing = sorted(_REQUIRED_DGX_DISTRIBUTIONS - seen)
        raise SchemaError(
            f"DGX dependency inventory lacks required model packages: {missing}"
        )
    return {"host": normalized_host, "packages": packages}


def _forbidden_field_counts(value: Any) -> dict[str, int]:
    counts = {category: 0 for category in _FORBIDDEN_CATEGORIES}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).lower().replace("-", "_").replace(" ", "_")
                for category, aliases in _FORBIDDEN_CATEGORY_ALIASES.items():
                    if any(alias in normalized for alias in aliases):
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
    dependency_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    runtime_source_files: Sequence[Mapping[str, Any]],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind explicit provisional PC-01 runtime facts into one typed identity.

    This constructor validates caller-supplied measurements.  It does not prove
    that they came from the currently executing DGX process; production use
    still requires the external per-block receipt described by the split lock.
    """

    dependencies = _validate_dgx_dependency_identity(dependency_identity)
    selected_runtime = _validate_pc01_runtime_identity(runtime_identity)
    source_files = _validate_runtime_source_files(list(runtime_source_files))
    environment = _validate_runtime_environment(
        runtime_environment, dependency_identity=dependencies
    )
    derived_host_sha256 = sha256_json(dependencies["host"])
    if host_identity_sha256 != derived_host_sha256:
        raise SchemaError(
            "DGX host identity SHA-256 differs from its nested Python/platform record"
        )
    identity = {
        "schema_version": DGX_MODEL_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "record_type": "DGXModelRuntimeIdentity",
        "host_identity_sha256": _require_sha256(
            host_identity_sha256, context="DGX host identity SHA-256"
        ),
        "dependency_identity": dependencies,
        "dependency_identity_sha256": sha256_json(dependencies),
        "model_seed": 42,
        "runtime_identity": selected_runtime,
        "runtime_identity_sha256": sha256_json(selected_runtime),
        "runtime_source_files": source_files,
        "runtime_source_set_sha256": sha256_json(source_files),
        "runtime_environment": environment,
        "runtime_environment_sha256": sha256_json(environment),
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
            "dependency_identity",
            "dependency_identity_sha256",
            "model_seed",
            "runtime_identity",
            "runtime_identity_sha256",
            "runtime_source_files",
            "runtime_source_set_sha256",
            "runtime_environment",
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
        "dependency_identity_sha256",
        "runtime_identity_sha256",
        "runtime_source_set_sha256",
        "runtime_environment_sha256",
    ):
        _require_sha256(identity.get(field), context=f"DGX runtime {field}")
    dependencies = _validate_dgx_dependency_identity(
        identity.get("dependency_identity")
    )
    if identity.get("dependency_identity_sha256") != sha256_json(dependencies):
        raise SchemaError("DGX dependency identity hash mismatch")
    if identity.get("host_identity_sha256") != sha256_json(dependencies["host"]):
        raise SchemaError(
            "DGX host identity hash differs from dependency inventory host"
        )
    nested = _validate_pc01_runtime_identity(identity.get("runtime_identity"))
    source_files = _validate_runtime_source_files(
        identity.get("runtime_source_files")
    )
    runtime_environment = _validate_runtime_environment(
        identity.get("runtime_environment"), dependency_identity=dependencies
    )
    try:
        expected_bytes = canonical_json_bytes(expected)
    except (TypeError, ValueError) as exc:
        raise SchemaError("DGX model-runtime identity is not canonical JSON") from exc
    if identity.get("runtime_identity_sha256") != sha256_json(nested):
        raise SchemaError("DGX nested runtime identity hash mismatch")
    if identity.get("runtime_source_set_sha256") != sha256_json(source_files):
        raise SchemaError("DGX runtime source-set hash mismatch")
    if identity.get("runtime_environment_sha256") != sha256_json(
        runtime_environment
    ):
        raise SchemaError("DGX runtime environment hash mismatch")
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
            "reward_envelope_fields_permitted",
            "oracle_envelope_fields_permitted",
            "evaluator_envelope_fields_permitted",
            "payload_content_attested",
            "endpoint_origin_attested",
        },
        context="split bridge identity",
    )
    exact_values = {
        "schema_version": SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION,
        "record_type": "SplitInferenceBridgeCompatibilityIdentity",
        "protocol_id": SPLIT_BRIDGE_PROTOCOL_ID,
        "protocol_version": SPLIT_BRIDGE_PROTOCOL_VERSION,
        "browser_host_identity_sha256": browser_host_identity_sha256,
        "dgx_model_runtime_identity_sha256": dgx_model_runtime_identity_sha256,
        "reward_envelope_fields_permitted": False,
        "oracle_envelope_fields_permitted": False,
        "evaluator_envelope_fields_permitted": False,
        "payload_content_attested": False,
        "endpoint_origin_attested": False,
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
    ):
        _require_sha256(identity.get(field), context=f"split bridge {field}")
    if identity.get("request_schema_sha256") != sha256_json(BRIDGE_REQUEST_SCHEMA):
        raise SchemaError("split bridge request schema hash is not registered")
    if identity.get("response_schema_sha256") != sha256_json(
        BRIDGE_RESPONSE_SCHEMA
    ):
        raise SchemaError("split bridge response schema hash is not registered")

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
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or relative == "."
            or relative in seen
        ):
            raise SchemaError(
                "split bridge source paths must be canonical, safe, and unique"
            )
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


def _validate_bridge_payload_envelope(
    value: object,
    *,
    direction: str,
) -> dict[str, str]:
    if direction == "request":
        schema_version = SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION
    elif direction == "response":
        schema_version = SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION
    else:  # pragma: no cover - private caller invariant
        raise SchemaError("split bridge direction is not registered")
    if not isinstance(value, Mapping):
        raise SchemaError(f"split bridge {direction} must be a mapping")
    forbidden = _forbidden_field_counts(value)
    _validate_zero_counts(
        forbidden, context=f"split bridge {direction} forbidden-field counts"
    )
    fields = {"schema_version", "request_id", "operation", "payload_sha256"}
    _require_exact_fields(value, fields, context=f"split bridge {direction}")
    if value.get("schema_version") != schema_version:
        raise SchemaError(
            f"split bridge {direction} schema version is not registered"
        )
    request_id = _require_nonempty(
        value.get("request_id"), context=f"split bridge {direction} request_id"
    )
    if len(request_id) > 256:
        raise SchemaError(f"split bridge {direction} request_id is too long")
    operation = _require_nonempty(
        value.get("operation"), context=f"split bridge {direction} operation"
    )
    if operation not in BRIDGE_OPERATIONS:
        raise SchemaError(f"split bridge {direction} operation is not registered")
    payload_sha256 = _require_sha256(
        value.get("payload_sha256"),
        context=f"split bridge {direction} payload_sha256",
    )
    return {
        "schema_version": schema_version,
        "request_id": request_id,
        "operation": operation,
        "payload_sha256": payload_sha256,
    }


def _validate_bridge_exchange(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise SchemaError("split bridge exchange must be a mapping")
    _require_exact_fields(value, {"request", "response"}, context="split bridge exchange")
    request = _validate_bridge_payload_envelope(
        value.get("request"), direction="request"
    )
    response = _validate_bridge_payload_envelope(
        value.get("response"), direction="response"
    )
    if (
        response["request_id"] != request["request_id"]
        or response["operation"] != request["operation"]
    ):
        raise SchemaError(
            "split bridge response is not bound to request ID and operation"
        )
    return {"request": request, "response": response}


def build_split_bridge_transcript(
    *,
    exchanges: Sequence[Mapping[str, Any]],
    local_browser_preflight_sha256: str,
    dgx_model_runtime_identity_sha256: str,
    bridge_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize redacted envelope records into a self-consistent hash chain.

    The supplied ``payload_sha256`` values are opaque. This function neither
    observes nor makes a semantic claim about the payloads they name.
    """

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
        validated_exchange = _validate_bridge_exchange(exchange)
        request = validated_exchange["request"]
        response = validated_exchange["response"]
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
            "request_schema_version": request["schema_version"],
            "response_schema_version": response["schema_version"],
            "request_id_sha256": sha256_json(request["request_id"]),
            "operation": request["operation"],
            "request_payload_sha256": request["payload_sha256"],
            "response_payload_sha256": response["payload_sha256"],
            "request_envelope_sha256": sha256_json(request),
            "response_envelope_sha256": sha256_json(response),
            "request_forbidden_envelope_field_counts": request_counts,
            "response_forbidden_envelope_field_counts": response_counts,
        }
        entry["entry_sha256"] = sha256_json(entry)
        previous = entry["entry_sha256"]
        entries.append(entry)
    return {
        "schema_version": SPLIT_TRANSCRIPT_SCHEMA_VERSION,
        "record_type": "SplitInferenceBridgeCompatibilityTranscript",
        "claim_scope": SPLIT_TRANSCRIPT_CLAIM_SCOPE,
        "request_schema_sha256": sha256_json(BRIDGE_REQUEST_SCHEMA),
        "response_schema_sha256": sha256_json(BRIDGE_RESPONSE_SCHEMA),
        "chain_algorithm": SPLIT_CHAIN_ALGORITHM,
        "append_only": True,
        "chain_seed_sha256": entries[0]["previous_entry_sha256"],
        "entry_count": len(entries),
        "entries": entries,
        "terminal_entry_sha256": previous,
        "forbidden_envelope_field_totals": {
            category: sum(
                entry[direction][category]
                for entry in entries
                for direction in (
                    "request_forbidden_envelope_field_counts",
                    "response_forbidden_envelope_field_counts",
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
            "claim_scope",
            "request_schema_sha256",
            "response_schema_sha256",
            "chain_algorithm",
            "append_only",
            "chain_seed_sha256",
            "entry_count",
            "entries",
            "terminal_entry_sha256",
            "forbidden_envelope_field_totals",
        },
        context="split bridge transcript",
    )
    expected_scalars = {
        "schema_version": SPLIT_TRANSCRIPT_SCHEMA_VERSION,
        "record_type": "SplitInferenceBridgeCompatibilityTranscript",
        "claim_scope": SPLIT_TRANSCRIPT_CLAIM_SCOPE,
        "chain_algorithm": SPLIT_CHAIN_ALGORITHM,
        "append_only": True,
    }
    for field, expected in expected_scalars.items():
        actual = transcript.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise SchemaError(f"split bridge transcript {field} is not registered")
    if transcript.get("request_schema_sha256") != sha256_json(
        BRIDGE_REQUEST_SCHEMA
    ) or transcript.get("response_schema_sha256") != sha256_json(
        BRIDGE_RESPONSE_SCHEMA
    ):
        raise SchemaError("split bridge transcript schema identity differs")
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
        "request_schema_version",
        "response_schema_version",
        "request_id_sha256",
        "operation",
        "request_payload_sha256",
        "response_payload_sha256",
        "request_envelope_sha256",
        "response_envelope_sha256",
        "request_forbidden_envelope_field_counts",
        "response_forbidden_envelope_field_counts",
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
        if (
            entry.get("request_schema_version")
            != SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION
            or entry.get("response_schema_version")
            != SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION
            or entry.get("operation") not in BRIDGE_OPERATIONS
        ):
            raise SchemaError("split bridge transcript envelope metadata differs")
        for field in (
            "request_id_sha256",
            "request_payload_sha256",
            "response_payload_sha256",
            "request_envelope_sha256",
            "response_envelope_sha256",
        ):
            _require_sha256(entry.get(field), context=f"split bridge entry {field}")
        _validate_zero_counts(
            entry.get("request_forbidden_envelope_field_counts"),
            context="split bridge request forbidden-envelope-field counts",
        )
        _validate_zero_counts(
            entry.get("response_forbidden_envelope_field_counts"),
            context="split bridge response forbidden-envelope-field counts",
        )
        unhashed = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if entry.get("entry_sha256") != sha256_json(unhashed):
            raise SchemaError("split bridge transcript entry hash mismatch")
        previous = str(entry["entry_sha256"])
    if transcript.get("terminal_entry_sha256") != previous:
        raise SchemaError("split bridge transcript terminal hash mismatch")
    _validate_zero_counts(
        transcript.get("forbidden_envelope_field_totals"),
        context="split bridge transcript forbidden-envelope-field totals",
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
    """Bind supplied inputs into one non-authorizing compatibility artifact."""

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
        "campaign_eligible": False,
        "dispatch_authorized": False,
        "dispatch_blocker": SPLIT_DISPATCH_BLOCKER,
        "deployment_topology": SPLIT_HOST_TOPOLOGY,
        "evidence_label": SPLIT_PREFLIGHT_EVIDENCE_LABEL,
        "paper_table_status": "N/R",
        "failed_dgx_routing_report_used": False,
        "compatibility_inputs": [
            "LOCAL_BROWSER_HOST_PASS",
            "PINNED_DGX_MODEL_RUNTIME_IDENTITY",
            "PINNED_REDACTED_BRIDGE_COMPATIBILITY_IDENTITY",
            "HASH_CHAINED_REDACTED_BRIDGE_ENVELOPE_METADATA",
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
            "dispatch_authorized",
            "dispatch_blocker",
            "deployment_topology",
            "evidence_label",
            "paper_table_status",
            "failed_dgx_routing_report_used",
            "compatibility_inputs",
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
        "campaign_eligible": False,
        "dispatch_authorized": False,
        "dispatch_blocker": SPLIT_DISPATCH_BLOCKER,
        "deployment_topology": SPLIT_HOST_TOPOLOGY,
        "evidence_label": SPLIT_PREFLIGHT_EVIDENCE_LABEL,
        "paper_table_status": "N/R",
        "failed_dgx_routing_report_used": False,
        "compatibility_inputs": [
            "LOCAL_BROWSER_HOST_PASS",
            "PINNED_DGX_MODEL_RUNTIME_IDENTITY",
            "PINNED_REDACTED_BRIDGE_COMPATIBILITY_IDENTITY",
            "HASH_CHAINED_REDACTED_BRIDGE_ENVELOPE_METADATA",
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
