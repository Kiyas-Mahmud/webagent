"""Semantic dependency identity for Table 2 campaign hosts.

The file hash remains useful for byte identity, but it is not evidence that the
named dependencies were measured.  This module closes that gap by rebuilding
and validating a small, versioned semantic record from the already validated
WebArena host preflight and the environment manifest. Single-host records bind
that measured host; split records explicitly label the DGX inventory as
caller-supplied compatibility data. This module does not inspect or authorize a
remote host by itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import metadata
import platform
from pathlib import Path
import sys
from typing import Any

from .common import SchemaError, atomic_write_json, read_json, sha256_file, sha256_json
from .split_deployment_preflight import SINGLE_HOST_TOPOLOGY, SPLIT_HOST_TOPOLOGY
from .split_deployment_preflight import (
    SPLIT_DISPATCH_BLOCKER,
    _split_dispatch_requirement,
)
from .webarena_preflight import PINNED_WEBARENA_PACKAGES


SEMANTIC_DEPENDENCY_LOCK_SCHEMA_VERSION = "table2-semantic-dependency-lock-v1"
SPLIT_SEMANTIC_DEPENDENCY_LOCK_SCHEMA_VERSION = (
    "table2-semantic-dependency-lock-v2"
)
SEMANTIC_DEPENDENCY_LOCK_RECORD_TYPE = "Table2SemanticDependencyLock"
SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE = (
    "MEASURED_HOST_DEPENDENCIES_NOT_CAMPAIGN_RESULTS"
)
SPLIT_SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE = (
    "BROWSER_MEASURED_DGX_SUPPLIED_COMPATIBILITY_NOT_DISPATCH_AUTHORITY"
)
SINGLE_HOST_ROLE = "single_host"
BROWSER_HOST_ROLE = "browser_host"
DGX_HOST_ROLE = "dgx_host"
SPLIT_RUNTIME_REMEASUREMENT_RESPONSIBILITIES = {
    BROWSER_HOST_ROLE: {
        "python_platform_packages": [
            "EVALUATION_BOOTSTRAP",
            "BEFORE_EACH_PHYSICAL_BLOCK",
        ],
        "runtime": [
            "LIVE_BROWSER_CAPABILITY_PREFLIGHT",
            "BEFORE_EACH_PHYSICAL_BLOCK",
        ],
    },
    DGX_HOST_ROLE: {
        "python_platform_packages": [
            "DGX_INFERENCE_SERVICE_STARTUP",
            "BEFORE_EACH_PHYSICAL_BLOCK",
        ],
        "runtime": [
            "DGX_INFERENCE_SERVICE_STARTUP",
            "BEFORE_MODEL_LOAD",
            "BEFORE_EACH_PHYSICAL_BLOCK",
        ],
    },
}


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be an object")
    return value


def _require_text(value: object, *, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise SchemaError(f"{context} must be a nonempty measured string")
    if any(marker in value.casefold() for marker in ("placeholder", "unknown", "tbd")):
        raise SchemaError(f"{context} is not a measured value")
    return value


def _host_preflight(
    deployment_preflight: Mapping[str, Any], *, topology: str
) -> Mapping[str, Any]:
    if topology == SINGLE_HOST_TOPOLOGY:
        return deployment_preflight
    if topology == SPLIT_HOST_TOPOLOGY:
        local = deployment_preflight.get("local_browser_preflight")
        if not isinstance(local, Mapping):
            raise SchemaError(
                "split dependency lock lacks local browser-host preflight"
            )
        return local
    raise SchemaError("dependency lock deployment topology is not registered")


def _semantic_core(
    *,
    environment: Mapping[str, Any],
    deployment_preflight: Mapping[str, Any],
    deployment_topology: str,
) -> dict[str, Any]:
    host_preflight = _host_preflight(
        deployment_preflight, topology=deployment_topology
    )
    host = _require_mapping(
        host_preflight.get("host"), context="dependency lock preflight host"
    )
    package_check = _require_mapping(
        host_preflight.get("package_check"),
        context="dependency lock package check",
    )
    browser = _require_mapping(
        host_preflight.get("browser_check"),
        context="dependency lock browser check",
    )
    if package_check.get("status") != "PASS":
        raise SchemaError("dependency lock package check did not pass")
    if browser.get("status") != "PASS":
        raise SchemaError("dependency lock browser check did not pass")
    package_rows = package_check.get("packages")
    if (
        not isinstance(package_rows, list)
        or len(package_rows) != len(PINNED_WEBARENA_PACKAGES)
    ):
        raise SchemaError(
            "dependency lock package rows must exactly cover the pinned stack"
        )
    packages: list[dict[str, str]] = []
    seen_distributions: set[str] = set()
    for row_value in package_rows:
        row = _require_mapping(row_value, context="dependency lock package row")
        distribution = _require_text(
            row.get("distribution"), context="dependency lock distribution"
        )
        if distribution in seen_distributions:
            raise SchemaError(
                "dependency lock package distributions must be unique"
            )
        seen_distributions.add(distribution)
        version = _require_text(
            row.get("actual_version"),
            context=f"dependency lock {distribution} version",
        )
        if (
            row.get("status") != "PASS"
            or row.get("expected_version") != version
            or PINNED_WEBARENA_PACKAGES.get(distribution) != version
        ):
            raise SchemaError(
                f"dependency lock package {distribution} differs from pinned PASS evidence"
            )
        packages.append({"distribution": distribution, "version": version})
    packages.sort(key=lambda row: row["distribution"])
    if {row["distribution"] for row in packages} != set(PINNED_WEBARENA_PACKAGES):
        raise SchemaError("dependency lock does not cover the pinned WebArena stack")

    environment_identity = {
        field: _require_text(
            environment.get(field), context=f"dependency lock environment {field}"
        )
        for field in (
            "benchmark",
            "benchmark_version",
            "benchmark_revision",
            "operating_system",
            "browser",
            "browser_version",
            "playwright_version",
            "controller_id",
            "controller_version",
            "environment_adapter_id",
            "environment_adapter_version",
            "container_digest",
        )
    }
    if environment_identity["browser"].casefold() != "chromium":
        raise SchemaError("dependency lock requires Chromium")
    if environment_identity["browser_version"] != browser.get("browser_version"):
        raise SchemaError("dependency lock Chromium version differs from preflight")
    if (
        environment_identity["playwright_version"]
        != PINNED_WEBARENA_PACKAGES["playwright"]
    ):
        raise SchemaError("dependency lock Playwright version differs from pinned stack")

    host_identity = {
        field: _require_text(host.get(field), context=f"dependency lock host {field}")
        for field in ("system", "release", "machine", "python_version")
    }
    python_digest = _require_text(
        host.get("python_executable_sha256"),
        context="dependency lock Python executable digest",
    )
    if len(python_digest) != 64 or any(c not in "0123456789abcdef" for c in python_digest):
        raise SchemaError("dependency lock Python executable digest is not SHA-256")
    host_identity["python_executable_sha256"] = python_digest
    if host_identity["system"].casefold() not in environment_identity[
        "operating_system"
    ].casefold():
        raise SchemaError(
            "dependency lock operating system differs from measured platform"
        )

    browser_identity = {
        "engine": _require_text(
            browser.get("browser"), context="dependency lock browser engine"
        ),
        "version": _require_text(
            browser.get("browser_version"), context="dependency lock browser version"
        ),
        "viewport": deepcopy(browser.get("viewport")),
        "device_scale_factor": browser.get("device_scale_factor"),
    }
    if browser_identity["engine"] != "chromium":
        raise SchemaError("dependency lock browser engine is not Chromium")
    if browser_identity["viewport"] != {"width": 1280, "height": 720}:
        raise SchemaError("dependency lock viewport differs from the registered viewport")
    if browser_identity["device_scale_factor"] != 1:
        raise SchemaError("dependency lock device scale factor differs")

    return {
        "deployment_topology": deployment_topology,
        "host": host_identity,
        "packages": packages,
        "browser": browser_identity,
        "environment": environment_identity,
        "deployment_preflight_content_sha256": sha256_json(deployment_preflight),
    }


def _split_semantic_core(
    *,
    environment: Mapping[str, Any],
    deployment_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Build distinct browser/DGX identities from split compatibility evidence."""

    local = _host_preflight(deployment_preflight, topology=SPLIT_HOST_TOPOLOGY)
    common = _semantic_core(
        environment=environment,
        deployment_preflight=local,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
    )
    dgx_value = deployment_preflight.get("dgx_model_runtime_identity")
    if not isinstance(dgx_value, Mapping):
        raise SchemaError(
            "split dependency lock requires a supplied DGX model-runtime identity"
        )
    # The split-preflight validator defines this exact v2 object.
    # Replaying it here still prevents a direct dependency-lock build from
    # accepting a missing, legacy, or internally inconsistent inventory.
    from .split_deployment_preflight import _validate_dgx_model_runtime_identity

    dgx = _validate_dgx_model_runtime_identity(dgx_value, expected=dgx_value)
    dependency_identity = _require_mapping(
        dgx.get("dependency_identity"),
        context="split DGX dependency identity",
    )
    browser_runtime = {
        "browser": deepcopy(common["browser"]),
        "environment": deepcopy(common["environment"]),
    }
    dgx_runtime = {
        "dgx_model_runtime_identity_sha256": sha256_json(dgx),
        "model_runtime_identity_sha256": dgx["runtime_identity_sha256"],
        "runtime_source_set_sha256": dgx["runtime_source_set_sha256"],
        "runtime_environment_sha256": dgx["runtime_environment_sha256"],
    }
    return {
        "deployment_topology": SPLIT_HOST_TOPOLOGY,
        "browser_host": {
            "host": deepcopy(common["host"]),
            "packages": deepcopy(common["packages"]),
            "runtime_identity": browser_runtime,
            "runtime_identity_sha256": sha256_json(browser_runtime),
        },
        "dgx_host": {
            "host": deepcopy(dependency_identity["host"]),
            "packages": deepcopy(dependency_identity["packages"]),
            "runtime_identity": dgx_runtime,
            "runtime_identity_sha256": sha256_json(dgx_runtime),
        },
        "runtime_remeasurement_responsibilities": deepcopy(
            SPLIT_RUNTIME_REMEASUREMENT_RESPONSIBILITIES
        ),
        "dgx_dispatch_receipt_requirement": _split_dispatch_requirement(),
        "browser": deepcopy(common["browser"]),
        "environment": deepcopy(common["environment"]),
        "deployment_preflight_content_sha256": sha256_json(deployment_preflight),
    }


def measure_current_python_dependency_identity(
    distributions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure the executing Python host and pinned distributions now.

    This measurement is separate from replaying the frozen preflight. The
    production bootstrap and each physical block use it on the campaign host,
    so a copied or stale preflight cannot hide Python-binary, platform, or
    package drift. Chromium itself remains a live browser-capability check; it
    is not inferred from Python package metadata here.
    """

    packages: list[dict[str, str]] = []
    names = tuple(
        sorted(
            PINNED_WEBARENA_PACKAGES
            if distributions is None
            else distributions
        )
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(type(name) is not str or not name.strip() for name in names)
    ):
        raise SchemaError("dependency measurement distributions are invalid")
    for distribution in names:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise SchemaError(
                f"current campaign host lacks pinned dependency {distribution}"
            ) from exc
        packages.append({"distribution": distribution, "version": version})
    return {
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable_sha256": sha256_file(sys.executable),
        },
        "packages": packages,
    }


def validate_current_host_against_semantic_dependency_lock(
    value: Mapping[str, Any],
    *,
    host_role: str | None = None,
    measured: Mapping[str, Any] | None = None,
    supplied_runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Remeasure host dependencies and compare an optional supplied identity.

    The Python/platform/package inventory is measured by this process when
    ``measured`` is omitted.  A DGX model-runtime JSON is only compared; this
    function does not claim to have generated or independently observed it.
    """

    topology = value.get("deployment_topology")
    if topology == SINGLE_HOST_TOPOLOGY:
        if host_role not in (None, SINGLE_HOST_ROLE):
            raise SchemaError("single-host dependency lock has no split host role")
        expected = {"host": value.get("host"), "packages": value.get("packages")}
        role_label = "campaign-host"
    elif topology == SPLIT_HOST_TOPOLOGY:
        if host_role not in {BROWSER_HOST_ROLE, DGX_HOST_ROLE}:
            raise SchemaError(
                "split dependency remeasurement requires browser_host or dgx_host role"
            )
        target = value.get(host_role)
        if not isinstance(target, Mapping):
            raise SchemaError(f"split dependency lock lacks {host_role} inventory")
        expected = {
            "host": target.get("host"),
            "packages": target.get("packages"),
        }
        role_label = host_role.replace("_", "-")
    else:
        raise SchemaError(
            "semantic dependency lock has an unregistered deployment topology"
        )
    package_rows = expected["packages"]
    if not isinstance(package_rows, list):
        raise SchemaError(f"dependency lock lacks {role_label} packages")
    distributions = [str(row.get("distribution")) for row in package_rows]
    current = dict(
        measured or measure_current_python_dependency_identity(distributions)
    )
    if set(current) != {"host", "packages"}:
        raise SchemaError("current dependency measurement fields differ")
    if current.get("host") != expected["host"]:
        raise SchemaError(
            f"current {role_label} Python/platform identity differs from dependency lock"
        )
    if current.get("packages") != expected["packages"]:
        raise SchemaError(
            f"current {role_label} package versions differ from dependency lock"
        )
    if topology == SPLIT_HOST_TOPOLOGY and host_role == DGX_HOST_ROLE:
        if not isinstance(supplied_runtime_identity, Mapping):
            raise SchemaError(
                "DGX-host dependency remeasurement requires a supplied model-runtime identity"
            )
        from .split_deployment_preflight import _validate_dgx_model_runtime_identity

        runtime = _validate_dgx_model_runtime_identity(
            supplied_runtime_identity,
            expected=supplied_runtime_identity,
        )
        actual_runtime = {
            "dgx_model_runtime_identity_sha256": sha256_json(runtime),
            "model_runtime_identity_sha256": runtime["runtime_identity_sha256"],
            "runtime_source_set_sha256": runtime["runtime_source_set_sha256"],
            "runtime_environment_sha256": runtime["runtime_environment_sha256"],
        }
        target = value[DGX_HOST_ROLE]
        if (
            actual_runtime != target.get("runtime_identity")
            or sha256_json(actual_runtime) != target.get("runtime_identity_sha256")
        ):
            raise SchemaError(
                "current DGX-host model-runtime identity differs from dependency lock"
            )
    elif supplied_runtime_identity is not None:
        raise SchemaError(
            "model-runtime identity may be supplied only for split dgx_host remeasurement"
        )
    return deepcopy(current)


def assert_split_dispatch_authority_registered(value: Mapping[str, Any]) -> None:
    """Fail closed until an externally trusted, per-block DGX receipt exists."""

    if value.get("deployment_topology") != SPLIT_HOST_TOPOLOGY:
        return
    requirement = value.get("dgx_dispatch_receipt_requirement")
    expected = _split_dispatch_requirement()
    if requirement != expected:
        raise SchemaError("split DGX dispatch-receipt requirement differs")
    if (
        requirement.get("status") == SPLIT_DISPATCH_BLOCKER
        and requirement.get("registered_receipt_schema_version") is None
        and requirement.get("registered_external_trust_anchor") is None
        and requirement.get("production_dispatch_authorized") is False
    ):
        raise SchemaError(
            "split dispatch is blocked: no externally trusted DGX startup/per-block "
            "remeasurement receipt is registered; the future receipt must bind "
            "campaign_id, block_id, challenge_nonce, the semantic lock, DGX host, "
            "runtime identity, phase, and issue time"
        )
    raise SchemaError("split DGX dispatch authority is not a registered exact contract")


def build_semantic_dependency_lock(
    *,
    environment: Mapping[str, Any],
    deployment_preflight: Mapping[str, Any],
    deployment_topology: str,
) -> dict[str, Any]:
    """Build a lock from validated evidence; never infer missing host values."""

    if deployment_topology == SINGLE_HOST_TOPOLOGY:
        schema_version = SEMANTIC_DEPENDENCY_LOCK_SCHEMA_VERSION
        claim_scope = SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE
        core = _semantic_core(
            environment=environment,
            deployment_preflight=deployment_preflight,
            deployment_topology=deployment_topology,
        )
    elif deployment_topology == SPLIT_HOST_TOPOLOGY:
        schema_version = SPLIT_SEMANTIC_DEPENDENCY_LOCK_SCHEMA_VERSION
        claim_scope = SPLIT_SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE
        core = _split_semantic_core(
            environment=environment,
            deployment_preflight=deployment_preflight,
        )
    else:
        raise SchemaError("dependency lock deployment topology is not registered")
    value = {
        "schema_version": schema_version,
        "record_type": SEMANTIC_DEPENDENCY_LOCK_RECORD_TYPE,
        "claim_scope": claim_scope,
        "paper_table_status": "N/R",
        **core,
        "semantic_identity_sha256": sha256_json(core),
    }
    return validate_semantic_dependency_lock(
        value,
        environment=environment,
        deployment_preflight=deployment_preflight,
        deployment_topology=deployment_topology,
    )


def validate_semantic_dependency_lock(
    value: object,
    *,
    environment: Mapping[str, Any],
    deployment_preflight: Mapping[str, Any],
    deployment_topology: str,
) -> dict[str, Any]:
    """Cross-check lock semantics against the measured environment evidence."""

    lock = _require_mapping(value, context="semantic dependency lock")
    common_fields = {
        "schema_version",
        "record_type",
        "claim_scope",
        "paper_table_status",
        "deployment_topology",
        "browser",
        "environment",
        "deployment_preflight_content_sha256",
        "semantic_identity_sha256",
    }
    if deployment_topology == SINGLE_HOST_TOPOLOGY:
        expected_fields = common_fields | {"host", "packages"}
        expected_schema_version = SEMANTIC_DEPENDENCY_LOCK_SCHEMA_VERSION
        expected_claim_scope = SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE
    elif deployment_topology == SPLIT_HOST_TOPOLOGY:
        expected_fields = common_fields | {
            "browser_host",
            "dgx_host",
            "runtime_remeasurement_responsibilities",
            "dgx_dispatch_receipt_requirement",
        }
        expected_schema_version = SPLIT_SEMANTIC_DEPENDENCY_LOCK_SCHEMA_VERSION
        expected_claim_scope = SPLIT_SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE
    else:
        raise SchemaError("dependency lock deployment topology is not registered")
    if set(lock) != expected_fields:
        raise SchemaError("semantic dependency-lock fields differ from schema")
    fixed = {
        "schema_version": expected_schema_version,
        "record_type": SEMANTIC_DEPENDENCY_LOCK_RECORD_TYPE,
        "claim_scope": expected_claim_scope,
        "paper_table_status": "N/R",
    }
    for field, expected in fixed.items():
        if lock.get(field) != expected:
            raise SchemaError(f"semantic dependency-lock {field} is not registered")
    expected_core = (
        _semantic_core(
            environment=environment,
            deployment_preflight=deployment_preflight,
            deployment_topology=deployment_topology,
        )
        if deployment_topology == SINGLE_HOST_TOPOLOGY
        else _split_semantic_core(
            environment=environment,
            deployment_preflight=deployment_preflight,
        )
    )
    actual_core = {field: deepcopy(lock.get(field)) for field in expected_core}
    if actual_core != expected_core:
        raise SchemaError(
            "semantic dependency lock differs from measured environment/preflight"
        )
    if lock.get("semantic_identity_sha256") != sha256_json(expected_core):
        raise SchemaError("semantic dependency-lock identity hash differs")
    return deepcopy(dict(lock))


def write_semantic_dependency_lock(
    output: str | Path,
    *,
    environment: Mapping[str, Any],
    deployment_preflight: Mapping[str, Any],
    deployment_topology: str,
) -> Path:
    value = build_semantic_dependency_lock(
        environment=environment,
        deployment_preflight=deployment_preflight,
        deployment_topology=deployment_topology,
    )
    return atomic_write_json(output, value, mode=0o444)


def read_and_validate_semantic_dependency_lock(
    path: str | Path,
    *,
    environment: Mapping[str, Any],
    deployment_preflight: Mapping[str, Any],
    deployment_topology: str,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SchemaError("semantic dependency lock is missing or symlinked")
    return validate_semantic_dependency_lock(
        read_json(source),
        environment=environment,
        deployment_preflight=deployment_preflight,
        deployment_topology=deployment_topology,
    )
