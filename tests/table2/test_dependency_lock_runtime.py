from __future__ import annotations

from copy import deepcopy

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_json
from web_agent.eval.table2.dependency_lock import (
    BROWSER_HOST_ROLE,
    DGX_HOST_ROLE,
    SPLIT_SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE,
    SPLIT_RUNTIME_REMEASUREMENT_RESPONSIBILITIES,
    assert_split_dispatch_authority_registered,
    build_semantic_dependency_lock,
    validate_current_host_against_semantic_dependency_lock,
)
from web_agent.eval.table2.split_deployment_preflight import (
    PC01_BACKBONE_ID,
    PC01_BACKBONE_REVISION,
    PC01_BASE_SNAPSHOT_SHA256,
    PC01_CHECKPOINT_SHA256,
    PC01_PROCESSOR_CONTRACT_SHA256,
    PC01_RESOLVED_CONFIG_SHA256,
    SINGLE_HOST_TOPOLOGY,
    SPLIT_HOST_TOPOLOGY,
    build_dgx_model_runtime_identity,
)
from web_agent.eval.table2.webarena_preflight import PINNED_WEBARENA_PACKAGES


def _lock_and_measurement() -> tuple[dict, dict]:
    measurement = {
        "host": {
            "system": "Linux",
            "release": "fixture-release",
            "machine": "aarch64",
            "python_version": "3.12.4",
            "python_executable_sha256": "a" * 64,
        },
        "packages": [
            {"distribution": "browsergym-core", "version": "0.14.3"},
            {"distribution": "browsergym-webarena", "version": "0.14.3"},
            {"distribution": "gymnasium", "version": "1.0.0"},
            {"distribution": "libwebarena", "version": "0.0.4"},
            {"distribution": "playwright", "version": "1.44.0"},
        ],
    }
    lock = {
        "deployment_topology": SINGLE_HOST_TOPOLOGY,
        **deepcopy(measurement),
    }
    return lock, measurement


def _environment_and_preflight() -> tuple[dict, dict]:
    environment = {
        "benchmark": "webarena",
        "benchmark_version": "0.14.3",
        "benchmark_revision": "fixture-revision",
        "operating_system": "Linux fixture",
        "browser": "chromium",
        "browser_version": "fixture-chromium",
        "playwright_version": "1.44.0",
        "controller_id": "browsergym",
        "controller_version": "0.14.3",
        "environment_adapter_id": "table2-webarena",
        "environment_adapter_version": "v1",
        "container_digest": "sha256:fixture-container",
    }
    preflight = {
        "host": {
            "system": "Linux",
            "release": "fixture-release",
            "machine": "aarch64",
            "python_version": "3.12.4",
            "python_executable_sha256": "a" * 64,
        },
        "package_check": {
            "status": "PASS",
            "packages": [
                {
                    "distribution": distribution,
                    "expected_version": version,
                    "actual_version": version,
                    "status": "PASS",
                }
                for distribution, version in sorted(
                    PINNED_WEBARENA_PACKAGES.items()
                )
            ],
        },
        "browser_check": {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture-chromium",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
        },
    }
    return environment, preflight


def test_current_host_measurement_must_equal_frozen_lock() -> None:
    lock, measured = _lock_and_measurement()
    assert validate_current_host_against_semantic_dependency_lock(
        lock, measured=measured
    ) == measured

    changed = deepcopy(measured)
    changed["host"]["python_executable_sha256"] = "b" * 64
    with pytest.raises(SchemaError, match="Python/platform identity differs"):
        validate_current_host_against_semantic_dependency_lock(
            lock, measured=changed
        )

    changed = deepcopy(measured)
    changed["packages"][-1]["version"] = "1.45.0"
    with pytest.raises(SchemaError, match="package versions differ"):
        validate_current_host_against_semantic_dependency_lock(
            lock, measured=changed
        )


def test_split_runtime_measurement_requires_and_selects_explicit_host_role() -> None:
    lock, measured = _lock_and_measurement()
    lock["deployment_topology"] = SPLIT_HOST_TOPOLOGY
    lock[BROWSER_HOST_ROLE] = deepcopy(measured)
    lock[DGX_HOST_ROLE] = deepcopy(measured)
    with pytest.raises(SchemaError, match="requires browser_host or dgx_host"):
        validate_current_host_against_semantic_dependency_lock(
            lock, measured=measured
        )
    assert validate_current_host_against_semantic_dependency_lock(
        lock, host_role=BROWSER_HOST_ROLE, measured=measured
    ) == measured
    dgx_changed = deepcopy(measured)
    dgx_changed["host"]["machine"] = "x86_64"
    with pytest.raises(SchemaError, match="dgx-host Python/platform"):
        validate_current_host_against_semantic_dependency_lock(
            lock, host_role=DGX_HOST_ROLE, measured=dgx_changed
        )


def _dgx_runtime_identity() -> dict:
    dependencies = {
        "host": {
            "system": "Linux",
            "release": "dgx-release",
            "machine": "aarch64",
            "python_version": "3.12.3",
            "python_executable_sha256": "d" * 64,
        },
        "packages": [
            {"distribution": "accelerate", "version": "1.0.0"},
            {"distribution": "bitsandbytes", "version": "0.49.0"},
            {"distribution": "peft", "version": "0.18.0"},
            {"distribution": "torch", "version": "2.13.0"},
            {"distribution": "transformers", "version": "4.57.6"},
        ],
    }
    return build_dgx_model_runtime_identity(
        host_identity_sha256=sha256_json(dependencies["host"]),
        dependency_identity=dependencies,
        runtime_identity={
            "backbone_id": PC01_BACKBONE_ID,
            "backbone_revision": PC01_BACKBONE_REVISION,
            "base_snapshot_sha256": PC01_BASE_SNAPSHOT_SHA256,
            "checkpoint_sha256": PC01_CHECKPOINT_SHA256,
            "resolved_config_sha256": PC01_RESOLVED_CONFIG_SHA256,
            "processor_contract_sha256": PC01_PROCESSOR_CONTRACT_SHA256,
        },
        runtime_source_files=[
            {"relative_path": "src/runtime.py", "sha256": "f" * 64}
        ],
        runtime_environment={
            "python_version": "3.12.3",
            "python_executable_sha256": "d" * 64,
            "torch_version": "2.13.0",
            "transformers_version": "4.57.6",
            "cuda_available": True,
            "cuda_runtime_version": "13.0",
            "device_type": "cuda",
            "device_name": "NVIDIA GB10",
            "device_count": 1,
            "container_digest": "sha256:" + "e" * 64,
        },
    )


def test_split_lock_binds_two_host_inventories_and_responsibilities() -> None:
    environment, local = _environment_and_preflight()
    dgx = _dgx_runtime_identity()
    preflight = {
        "local_browser_preflight": local,
        "dgx_model_runtime_identity": dgx,
    }
    lock = build_semantic_dependency_lock(
        environment=environment,
        deployment_preflight=preflight,
        deployment_topology=SPLIT_HOST_TOPOLOGY,
    )
    assert lock["schema_version"] == "table2-semantic-dependency-lock-v2"
    assert lock["claim_scope"] == SPLIT_SEMANTIC_DEPENDENCY_LOCK_CLAIM_SCOPE
    assert lock[BROWSER_HOST_ROLE]["host"] == local["host"]
    assert lock[DGX_HOST_ROLE]["host"] == dgx["dependency_identity"]["host"]
    assert lock[DGX_HOST_ROLE]["packages"] == dgx["dependency_identity"]["packages"]
    assert lock["runtime_remeasurement_responsibilities"] == (
        SPLIT_RUNTIME_REMEASUREMENT_RESPONSIBILITIES
    )
    dgx_measurement = deepcopy(dgx["dependency_identity"])
    with pytest.raises(SchemaError, match="requires a supplied model-runtime"):
        validate_current_host_against_semantic_dependency_lock(
            lock,
            host_role=DGX_HOST_ROLE,
            measured=dgx_measurement,
        )
    assert validate_current_host_against_semantic_dependency_lock(
        lock,
        host_role=DGX_HOST_ROLE,
        measured=dgx_measurement,
        supplied_runtime_identity=dgx,
    ) == dgx_measurement
    changed_runtime = deepcopy(dgx)
    changed_runtime["runtime_identity"]["checkpoint_sha256"] = "2" * 64
    changed_runtime["runtime_identity_sha256"] = sha256_json(
        changed_runtime["runtime_identity"]
    )
    with pytest.raises(SchemaError, match="registered pilot"):
        validate_current_host_against_semantic_dependency_lock(
            lock,
            host_role=DGX_HOST_ROLE,
            measured=dgx_measurement,
            supplied_runtime_identity=changed_runtime,
        )

    assert lock["dgx_dispatch_receipt_requirement"][
        "production_dispatch_authorized"
    ] is False
    with pytest.raises(SchemaError, match="split dispatch is blocked"):
        assert_split_dispatch_authority_registered(lock)

    missing = deepcopy(preflight)
    del missing["dgx_model_runtime_identity"]["dependency_identity"]
    with pytest.raises(SchemaError, match="fields are not the registered closure"):
        build_semantic_dependency_lock(
            environment=environment,
            deployment_preflight=missing,
            deployment_topology=SPLIT_HOST_TOPOLOGY,
        )


def test_dependency_lock_rejects_failed_or_duplicate_measurement_rows() -> None:
    environment, preflight = _environment_and_preflight()
    assert build_semantic_dependency_lock(
        environment=environment,
        deployment_preflight=preflight,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
    )["packages"]

    failed = deepcopy(preflight)
    failed["package_check"]["status"] = "FAIL"
    with pytest.raises(SchemaError, match="package check did not pass"):
        build_semantic_dependency_lock(
            environment=environment,
            deployment_preflight=failed,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
        )

    duplicate = deepcopy(preflight)
    duplicate["package_check"]["packages"][-1] = deepcopy(
        duplicate["package_check"]["packages"][0]
    )
    with pytest.raises(SchemaError, match="unique|cover"):
        build_semantic_dependency_lock(
            environment=environment,
            deployment_preflight=duplicate,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
        )
