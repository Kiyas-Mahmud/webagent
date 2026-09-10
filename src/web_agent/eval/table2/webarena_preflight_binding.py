"""Hash-bound WebArena deployment preflight artifacts.

The measured preflight is deliberately stored outside ``environment.json`` so
its original bytes survive handoff and campaign freezing.  This module binds
those bytes and the credential-free URL map into the environment, then invokes
the topology-specific semantic validator every time the bundle crosses a
trust boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import SchemaError, read_json, sha256_file, sha256_json
from .webarena_preflight import load_service_url_map


PREFLIGHT_BINDING_FIELD = "webarena_deployment_preflight"
PREFLIGHT_BINDING_SCHEMA_VERSION = (
    "table2-webarena-deployment-preflight-binding-v1"
)
PREFLIGHT_ARTIFACT_RELATIVE_PATH = "webarena_deployment_preflight.json"
PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH = "webarena_service_url_map.json"
PREFLIGHT_VALIDATOR_CONTRACT = (
    "validate_webarena_deployment_preflight:v1"
)


@dataclass(frozen=True, slots=True)
class ValidatedDeploymentPreflight:
    binding: dict[str, Any]
    evidence: dict[str, Any]
    service_url_map: dict[str, str]
    evidence_path: Path
    service_url_map_path: Path


def _semantic_validate(
    evidence: Mapping[str, Any],
    *,
    deployment_topology: str,
    service_url_map: Mapping[str, Any],
    expected_live_reset_task_index: int,
    expected_dgx_model_runtime_identity: Mapping[str, Any] | None = None,
    expected_bridge_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Kept behind one dispatcher so SINGLE_DGX_HOST can never fall through to
    # the split-host schema.  The split validator itself requires both exact
    # expected identities before it will validate split compatibility. Dispatch
    # authority is a separate fail-closed execution-guard decision.
    from .split_deployment_preflight import (
        validate_webarena_deployment_preflight,
    )

    return validate_webarena_deployment_preflight(
        evidence,
        deployment_topology=deployment_topology,
        service_url_map=service_url_map,
        expected_live_reset_task_index=expected_live_reset_task_index,
        expected_dgx_model_runtime_identity=expected_dgx_model_runtime_identity,
        expected_bridge_identity=expected_bridge_identity,
    )


def build_deployment_preflight_binding(
    *,
    evidence_path: str | Path,
    service_url_map_path: str | Path,
    deployment_topology: str,
    expected_live_reset_task_index: int,
    expected_dgx_model_runtime_identity: Mapping[str, Any] | None = None,
    expected_bridge_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate source artifacts and return their closed environment binding."""

    evidence_source = Path(evidence_path)
    url_map_source = Path(service_url_map_path)
    for source, label in (
        (evidence_source, "WebArena deployment preflight"),
        (url_map_source, "WebArena site URL map"),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.is_symlink():
            raise SchemaError(f"{label} must not be a symlink")
    evidence = read_json(evidence_source)
    urls = load_service_url_map(url_map_source)
    validated = _semantic_validate(
        evidence,
        deployment_topology=deployment_topology,
        service_url_map=urls,
        expected_live_reset_task_index=expected_live_reset_task_index,
        expected_dgx_model_runtime_identity=expected_dgx_model_runtime_identity,
        expected_bridge_identity=expected_bridge_identity,
    )
    if validated != evidence:
        raise SchemaError("deployment preflight validation changed evidence content")
    from .split_deployment_preflight import (
        SINGLE_HOST_TOPOLOGY,
        SPLIT_HOST_TOPOLOGY,
    )

    if deployment_topology == SINGLE_HOST_TOPOLOGY:
        if (
            expected_dgx_model_runtime_identity is not None
            or expected_bridge_identity is not None
        ):
            raise SchemaError(
                "single-host preflight must not invent split-host identities"
            )
        dgx_identity = None
        bridge_identity = None
    elif deployment_topology == SPLIT_HOST_TOPOLOGY:
        if not isinstance(expected_dgx_model_runtime_identity, Mapping) or not isinstance(
            expected_bridge_identity, Mapping
        ):
            raise SchemaError(
                "split-host preflight requires frozen DGX and bridge identities"
            )
        dgx_identity = dict(expected_dgx_model_runtime_identity)
        bridge_identity = dict(expected_bridge_identity)
    else:  # The dispatcher already rejects this; keep binding logic explicit.
        raise SchemaError("unregistered deployment topology cannot be bound")
    return {
        "schema_version": PREFLIGHT_BINDING_SCHEMA_VERSION,
        "deployment_topology": deployment_topology,
        "validator_contract": PREFLIGHT_VALIDATOR_CONTRACT,
        "expected_live_reset_task_index": expected_live_reset_task_index,
        "preflight_artifact_path": PREFLIGHT_ARTIFACT_RELATIVE_PATH,
        "preflight_artifact_sha256": sha256_file(evidence_source),
        "preflight_content_sha256": sha256_json(evidence),
        "service_url_map_path": PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
        "service_url_map_artifact_sha256": sha256_file(url_map_source),
        "service_url_map_content_sha256": sha256_json(urls),
        "expected_dgx_model_runtime_identity": dgx_identity,
        "expected_dgx_model_runtime_identity_sha256": (
            sha256_json(dgx_identity) if dgx_identity is not None else None
        ),
        "expected_bridge_identity": bridge_identity,
        "expected_bridge_identity_sha256": (
            sha256_json(bridge_identity) if bridge_identity is not None else None
        ),
    }


def validate_bound_deployment_preflight(
    environment: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    expected_dgx_model_runtime_identity: Mapping[str, Any] | None = None,
    expected_bridge_identity: Mapping[str, Any] | None = None,
) -> ValidatedDeploymentPreflight:
    """Reopen, hash, and semantically validate a bound preflight package."""

    binding = environment.get(PREFLIGHT_BINDING_FIELD)
    if not isinstance(binding, Mapping):
        raise SchemaError(
            f"environment requires {PREFLIGHT_BINDING_FIELD} evidence binding"
        )
    expected_fields = {
        "schema_version",
        "deployment_topology",
        "validator_contract",
        "expected_live_reset_task_index",
        "preflight_artifact_path",
        "preflight_artifact_sha256",
        "preflight_content_sha256",
        "service_url_map_path",
        "service_url_map_artifact_sha256",
        "service_url_map_content_sha256",
        "expected_dgx_model_runtime_identity",
        "expected_dgx_model_runtime_identity_sha256",
        "expected_bridge_identity",
        "expected_bridge_identity_sha256",
    }
    if set(binding) != expected_fields:
        raise SchemaError(
            "WebArena deployment-preflight binding is not an exact field closure"
        )
    if binding.get("schema_version") != PREFLIGHT_BINDING_SCHEMA_VERSION:
        raise SchemaError("WebArena deployment-preflight binding schema is not registered")
    if binding.get("validator_contract") != PREFLIGHT_VALIDATOR_CONTRACT:
        raise SchemaError("WebArena deployment-preflight validator contract changed")
    reset_index = binding.get("expected_live_reset_task_index")
    if type(reset_index) is not int or reset_index < 0:
        raise SchemaError("bound WebArena live-reset task index is invalid")
    if binding.get("preflight_artifact_path") != PREFLIGHT_ARTIFACT_RELATIVE_PATH:
        raise SchemaError("bound WebArena preflight artifact path changed")
    if (
        binding.get("service_url_map_path")
        != PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH
    ):
        raise SchemaError("bound WebArena service URL-map artifact path changed")

    root = Path(artifact_root)
    evidence_path = root / PREFLIGHT_ARTIFACT_RELATIVE_PATH
    url_map_path = root / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH
    for source, label in (
        (evidence_path, "bound WebArena deployment preflight"),
        (url_map_path, "bound WebArena site URL map"),
    ):
        if not source.is_file():
            raise SchemaError(f"{label} is missing")
        if source.is_symlink():
            raise SchemaError(f"{label} must not be a symlink")
    if sha256_file(evidence_path) != binding.get("preflight_artifact_sha256"):
        raise SchemaError("bound WebArena deployment-preflight byte hash differs")
    if sha256_file(url_map_path) != binding.get(
        "service_url_map_artifact_sha256"
    ):
        raise SchemaError("bound WebArena service URL-map byte hash differs")

    evidence = read_json(evidence_path)
    urls = load_service_url_map(url_map_path)
    if sha256_json(evidence) != binding.get("preflight_content_sha256"):
        raise SchemaError("bound WebArena deployment-preflight content hash differs")
    if sha256_json(urls) != binding.get("service_url_map_content_sha256"):
        raise SchemaError("bound WebArena service URL-map content hash differs")
    topology = binding.get("deployment_topology")
    if not isinstance(topology, str) or not topology:
        raise SchemaError("bound WebArena deployment topology is absent")
    from .split_deployment_preflight import (
        SINGLE_HOST_TOPOLOGY,
        SPLIT_HOST_TOPOLOGY,
    )

    bound_dgx = binding.get("expected_dgx_model_runtime_identity")
    bound_bridge = binding.get("expected_bridge_identity")
    bound_dgx_sha = binding.get("expected_dgx_model_runtime_identity_sha256")
    bound_bridge_sha = binding.get("expected_bridge_identity_sha256")
    if topology == SINGLE_HOST_TOPOLOGY:
        if any(
            value is not None
            for value in (bound_dgx, bound_bridge, bound_dgx_sha, bound_bridge_sha)
        ):
            raise SchemaError("single-host binding contains split-host identities")
        if (
            expected_dgx_model_runtime_identity is not None
            or expected_bridge_identity is not None
        ):
            raise SchemaError("single-host validation received split-host identities")
        selected_dgx = None
        selected_bridge = None
    elif topology == SPLIT_HOST_TOPOLOGY:
        if not isinstance(bound_dgx, Mapping) or not isinstance(bound_bridge, Mapping):
            raise SchemaError("split-host binding lacks frozen expected identities")
        if sha256_json(bound_dgx) != bound_dgx_sha:
            raise SchemaError("bound expected DGX runtime identity hash differs")
        if sha256_json(bound_bridge) != bound_bridge_sha:
            raise SchemaError("bound expected bridge identity hash differs")
        if (
            expected_dgx_model_runtime_identity is not None
            and dict(expected_dgx_model_runtime_identity) != dict(bound_dgx)
        ):
            raise SchemaError("external expected DGX runtime identity differs from binding")
        if (
            expected_bridge_identity is not None
            and dict(expected_bridge_identity) != dict(bound_bridge)
        ):
            raise SchemaError("external expected bridge identity differs from binding")
        selected_dgx = dict(bound_dgx)
        selected_bridge = dict(bound_bridge)
    else:
        raise SchemaError(f"unregistered bound deployment topology: {topology!r}")
    validated = _semantic_validate(
        evidence,
        deployment_topology=topology,
        service_url_map=urls,
        expected_live_reset_task_index=reset_index,
        expected_dgx_model_runtime_identity=selected_dgx,
        expected_bridge_identity=selected_bridge,
    )
    if validated != evidence:
        raise SchemaError("bound deployment preflight changed during validation")
    return ValidatedDeploymentPreflight(
        binding=dict(binding),
        evidence=dict(evidence),
        service_url_map=dict(urls),
        evidence_path=evidence_path,
        service_url_map_path=url_map_path,
    )
