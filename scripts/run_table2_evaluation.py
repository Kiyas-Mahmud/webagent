"""Run or resume frozen paired E0--E3 Table 2 blocks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import os
import platform
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping


EVALUATION_RUNNER_SCOPE = "FROZEN_EVALUATION_RUNNER"
FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH = "dependency.lock"
PC01_PRODUCTION_RUNNER_ENTRYPOINT = (
    "web_agent.eval.table2.production_runner:create_runner"
)
PC01_PAGE_BROKER_SECURITY_FIELD = "pc01_page_broker_security"
PC01_PAGE_BROKER_SECURITY_BLOCKED_BINDING = {
    "schema_version": "table2-pc01-page-broker-security-v1",
    "status": "BLOCKED_EXTERNAL_PROCESS_ISOLATION_REQUIRED",
    "claim_scope": "REVIEWED_CODE_DATAFLOW_ONLY_NOT_PROCESS_ISOLATION",
    "architecture": "same_process_in_memory_typed_capabilities",
    "same_process_broker": True,
    "kernel_process_isolation": False,
    "runtime_process_can_import_sealed_capability": True,
    "external_process_isolation_evidence_present": False,
    "production_dispatch_authorized": False,
}
PC01_PROCESS_BROKER_SOURCE_PATHS = (
    "src/web_agent/__init__.py",
    "src/web_agent/eval/__init__.py",
    "src/web_agent/eval/table2/__init__.py",
    "src/web_agent/eval/table2/common.py",
    "src/web_agent/eval/table2/process_broker.py",
    "src/web_agent/eval/table2/process_broker_protocol.py",
    "src/web_agent/eval/table2/process_broker_runtime.py",
    "src/web_agent/eval/table2/process_broker_worker.py",
)
PC01_PROCESS_BROKER_ARBITRARY_MAPPING_PATHS = [
    "runtime_execute.request.action",
    "runtime_observe.result.observation",
    "runtime_execute.result.execution",
]
PC01_PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS = [
    "REGISTER_EXACT_OPERATION_SPECIFIC_INNER_SCHEMAS",
    "ATTEST_RUNTIME_VALUE_PROVENANCE",
    "REGISTER_EXTERNAL_DEPLOYMENT_RECEIPT_SCHEMA_AND_TRUST_ANCHOR",
]
PINNED_SEMANTIC_DEPENDENCIES = {
    "browsergym-core": "0.14.3",
    "browsergym-webarena": "0.14.3",
    "gymnasium": "1.0.0",
    "libwebarena": "0.0.4",
    "playwright": "1.44.0",
}
PC01_SPLIT_RUNTIME_IDENTITY = {
    "backbone_id": "Qwen/Qwen2-VL-2B-Instruct",
    "backbone_revision": "895c3a49bc3fa70a340399125c650a463535e71c",
    "base_snapshot_sha256": (
        "e002f8290faa3e9f44bf3099eac85a2445e17de738c5bb0cc10d342da837c46c"
    ),
    "checkpoint_sha256": (
        "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a"
    ),
    "resolved_config_sha256": (
        "d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f"
    ),
    "processor_contract_sha256": (
        "b3c9f629a3c4ce6ab7bb71b27c7e0b560e3e21c51a19d1235fd4a507ba437797"
    ),
}
SPLIT_DGX_DISPATCH_REQUIREMENT = {
    "schema_version": "table2-split-dgx-remeasurement-requirement-v1",
    "status": (
        "BLOCKED_DGX_REMEASUREMENT_RECEIPT_AND_EXTERNAL_TRUST_ANCHOR_REQUIRED"
    ),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument(
        "--runner",
        required=True,
        help=(
            "module:attribute for a callable/runtime object; the callable may "
            "accept the complete request mapping or supported keyword arguments"
        ),
    )
    parser.add_argument(
        "--runner-factory",
        action="store_true",
        help="call the selected entrypoint once with campaign_dir before execution",
    )
    parser.add_argument(
        "--pc01-operations-provider-factory",
        default=None,
        help=(
            "required only for the canonical PC-01 production runner; an "
            "exact frozen module:function called with one detached, oracle-free "
            "bootstrap record and returning exactly PC01LiveOperationsProvider"
        ),
    )
    parser.add_argument(
        "--pc01-credential-capability-root",
        type=Path,
        default=None,
        help="external deployment-owned credential directory (never copied or read)",
    )
    parser.add_argument("--pc01-credential-capability-id", default=None)
    parser.add_argument("--pc01-credential-capability-version", default=None)
    parser.add_argument(
        "--pc01-provider-boundary-receipt",
        type=Path,
        default=None,
        help=(
            "required external receipt for the reviewed-code/oracle-free dataflow "
            "boundary; this does not assert filesystem or hostile-code isolation"
        ),
    )
    parser.add_argument(
        "--prepare-pc01-provider-boundary-receipt",
        type=Path,
        default=None,
        help=(
            "validate the frozen campaign and write the deterministic external "
            "provider-boundary receipt, then exit without loading a factory"
        ),
    )
    parser.add_argument("--block-id", default=None, help="run/resume only one frozen block")
    parser.add_argument("--maximum-blocks", type=int, default=None)
    parser.add_argument(
        "--live-readiness-probe-for",
        type=Path,
        default=None,
        help=(
            "run exactly one explicit normal E0--E3 block in this isolated "
            "campaign and write a non-scored readiness receipt for the supplied "
            "otherwise-unstarted PILOT_ONLY target campaign"
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bootstrap_validate_split_dependency_lock(
    *, environment: Mapping[str, Any], preflight: Mapping[str, Any], lock: Mapping[str, Any]
) -> None:
    """Stdlib-only replay of the dual-host v2 dependency authority."""

    local = preflight.get("local_browser_preflight")
    dgx = preflight.get("dgx_model_runtime_identity")
    if not isinstance(local, dict) or not isinstance(dgx, dict):
        raise RuntimeError(
            "split dependency authority lacks browser-host or DGX-host inventory"
        )
    host = local.get("host")
    package_check = local.get("package_check")
    browser = local.get("browser_check")
    if not all(isinstance(value, dict) for value in (host, package_check, browser)):
        raise RuntimeError("split browser-host dependency preflight is incomplete")
    package_rows = package_check.get("packages")
    if not isinstance(package_rows, list):
        raise RuntimeError("split browser-host package inventory is absent")
    package_by_name = {
        row.get("distribution"): row
        for row in package_rows
        if isinstance(row, dict)
    }
    if (
        package_check.get("status") != "PASS"
        or len(package_rows) != len(package_by_name)
        or set(package_by_name) != set(PINNED_SEMANTIC_DEPENDENCIES)
    ):
        raise RuntimeError("split browser-host package inventory differs")
    for distribution, version in PINNED_SEMANTIC_DEPENDENCIES.items():
        row = package_by_name[distribution]
        if (
            row.get("status") != "PASS"
            or row.get("expected_version") != version
            or row.get("actual_version") != version
        ):
            raise RuntimeError(
                f"split browser-host dependency {distribution} differs"
            )
    if (
        browser.get("status") != "PASS"
        or browser.get("browser") != "chromium"
        or browser.get("viewport") != {"width": 1280, "height": 720}
        or browser.get("device_scale_factor") != 1
        or environment.get("browser_version") != browser.get("browser_version")
        or environment.get("playwright_version")
        != PINNED_SEMANTIC_DEPENDENCIES["playwright"]
        or str(host.get("system") or "").casefold()
        not in str(environment.get("operating_system") or "").casefold()
    ):
        raise RuntimeError("split browser-host runtime identity differs")

    dependency_identity = dgx.get("dependency_identity")
    runtime_source_files = dgx.get("runtime_source_files")
    runtime_environment = dgx.get("runtime_environment")
    if (
        dgx.get("schema_version") != "table2-dgx-model-runtime-identity-v2"
        or set(dgx)
        != {
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
        }
        or dgx.get("record_type") != "DGXModelRuntimeIdentity"
        or dgx.get("model_seed") != 42
        or dgx.get("evaluation_mode") is not True
        or dgx.get("weights_mutated") is not False
        or not isinstance(dependency_identity, dict)
        or set(dependency_identity) != {"host", "packages"}
        or not isinstance(dependency_identity.get("host"), dict)
        or set(dependency_identity["host"])
        != {
            "system",
            "release",
            "machine",
            "python_version",
            "python_executable_sha256",
        }
        or not isinstance(dependency_identity.get("packages"), list)
        or not dependency_identity["packages"]
        or dgx.get("dependency_identity_sha256")
        != _sha256_json(dependency_identity)
        or dgx.get("host_identity_sha256")
        != _sha256_json(dependency_identity["host"])
        or dgx.get("runtime_identity") != PC01_SPLIT_RUNTIME_IDENTITY
        or dgx.get("runtime_identity_sha256")
        != _sha256_json(dgx["runtime_identity"])
        or not isinstance(runtime_source_files, list)
        or not runtime_source_files
        or dgx.get("runtime_source_set_sha256")
        != _sha256_json(runtime_source_files)
        or not isinstance(runtime_environment, dict)
        or dgx.get("runtime_environment_sha256")
        != _sha256_json(runtime_environment)
    ):
        raise RuntimeError("split DGX dependency/runtime inventory is absent or invalid")
    dgx_host = dependency_identity["host"]
    if (
        any(
            not isinstance(dgx_host.get(field), str)
            or not dgx_host[field]
            or any(
                marker in dgx_host[field].casefold()
                for marker in ("placeholder", "unknown", "tbd")
            )
            for field in ("system", "release", "machine", "python_version")
        )
        or not isinstance(dgx_host.get("python_executable_sha256"), str)
        or len(dgx_host["python_executable_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in dgx_host["python_executable_sha256"]
        )
    ):
        raise RuntimeError("split DGX dependency host identity is malformed")
    dgx_packages = dependency_identity["packages"]
    if any(
        not isinstance(row, dict)
        or set(row) != {"distribution", "version"}
        or not isinstance(row.get("distribution"), str)
        or not row.get("distribution")
        or any(
            marker in row["distribution"].casefold()
            for marker in ("placeholder", "unknown", "tbd")
        )
        or not isinstance(row.get("version"), str)
        or not row.get("version")
        or any(
            marker in row["version"].casefold()
            for marker in ("placeholder", "unknown", "tbd")
        )
        for row in dgx_packages
    ):
        raise RuntimeError("split DGX package inventory is malformed")
    if dgx_packages != sorted(dgx_packages, key=lambda row: row["distribution"]):
        raise RuntimeError("split DGX package inventory is not sorted")
    dgx_names = [row["distribution"] for row in dgx_packages]
    if len(dgx_names) != len(set(dgx_names)) or not {
        "accelerate",
        "bitsandbytes",
        "peft",
        "torch",
        "transformers",
    }.issubset(dgx_names):
        raise RuntimeError("split DGX package inventory is incomplete or duplicated")
    if any(
        not isinstance(row, dict)
        or set(row) != {"relative_path", "sha256"}
        or not isinstance(row.get("relative_path"), str)
        or not row["relative_path"]
        or Path(row["relative_path"]).is_absolute()
        or ".." in Path(row["relative_path"]).parts
        or Path(row["relative_path"]).as_posix() != row["relative_path"]
        or row["relative_path"] == "."
        or not isinstance(row.get("sha256"), str)
        or len(row["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in row["sha256"])
        for row in runtime_source_files
    ):
        raise RuntimeError("split DGX runtime source inventory is malformed")
    if runtime_source_files != sorted(
        runtime_source_files, key=lambda row: row["relative_path"]
    ) or len({row["relative_path"] for row in runtime_source_files}) != len(
        runtime_source_files
    ):
        raise RuntimeError("split DGX runtime sources are unsorted or duplicated")
    expected_runtime_environment_fields = {
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
    dependency_packages = {
        row["distribution"]: row["version"] for row in dgx_packages
    }
    if (
        set(runtime_environment) != expected_runtime_environment_fields
        or runtime_environment.get("python_version")
        != dependency_identity["host"].get("python_version")
        or runtime_environment.get("python_executable_sha256")
        != dependency_identity["host"].get("python_executable_sha256")
        or runtime_environment.get("torch_version")
        != dependency_packages.get("torch")
        or runtime_environment.get("transformers_version")
        != dependency_packages.get("transformers")
        or runtime_environment.get("cuda_available") is not True
        or runtime_environment.get("device_type") != "cuda"
        or type(runtime_environment.get("device_count")) is not int
        or runtime_environment["device_count"] < 1
        or any(
            not isinstance(runtime_environment.get(field), str)
            or not runtime_environment[field]
            or any(
                marker in runtime_environment[field].casefold()
                for marker in ("placeholder", "unknown", "tbd")
            )
            for field in (
                "cuda_runtime_version",
                "device_name",
                "container_digest",
            )
        )
        or not runtime_environment["container_digest"].startswith("sha256:")
        or len(runtime_environment["container_digest"]) != 71
        or any(
            character not in "0123456789abcdef"
            for character in runtime_environment["container_digest"][7:]
        )
    ):
        raise RuntimeError("split DGX runtime environment is malformed or inconsistent")

    packages = [
        {"distribution": name, "version": package_by_name[name]["actual_version"]}
        for name in sorted(package_by_name)
    ]
    host_identity = {
        field: host.get(field)
        for field in (
            "system",
            "release",
            "machine",
            "python_version",
            "python_executable_sha256",
        )
    }
    browser_identity = {
        "engine": browser.get("browser"),
        "version": browser.get("browser_version"),
        "viewport": browser.get("viewport"),
        "device_scale_factor": browser.get("device_scale_factor"),
    }
    environment_identity = {
        field: environment.get(field)
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
    browser_runtime = {
        "browser": browser_identity,
        "environment": environment_identity,
    }
    dgx_runtime = {
        "dgx_model_runtime_identity_sha256": _sha256_json(dgx),
        "model_runtime_identity_sha256": dgx.get("runtime_identity_sha256"),
        "runtime_source_set_sha256": dgx.get("runtime_source_set_sha256"),
        "runtime_environment_sha256": dgx.get("runtime_environment_sha256"),
    }
    responsibilities = {
        "browser_host": {
            "python_platform_packages": [
                "EVALUATION_BOOTSTRAP",
                "BEFORE_EACH_PHYSICAL_BLOCK",
            ],
            "runtime": [
                "LIVE_BROWSER_CAPABILITY_PREFLIGHT",
                "BEFORE_EACH_PHYSICAL_BLOCK",
            ],
        },
        "dgx_host": {
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
    core = {
        "deployment_topology": "SPLIT_LOCAL_BROWSER_DGX_INFERENCE",
        "browser_host": {
            "host": host_identity,
            "packages": packages,
            "runtime_identity": browser_runtime,
            "runtime_identity_sha256": _sha256_json(browser_runtime),
        },
        "dgx_host": {
            "host": dependency_identity["host"],
            "packages": dgx_packages,
            "runtime_identity": dgx_runtime,
            "runtime_identity_sha256": _sha256_json(dgx_runtime),
        },
        "runtime_remeasurement_responsibilities": responsibilities,
        "dgx_dispatch_receipt_requirement": SPLIT_DGX_DISPATCH_REQUIREMENT,
        "browser": browser_identity,
        "environment": environment_identity,
        "deployment_preflight_content_sha256": _sha256_json(preflight),
    }
    expected = {
        "schema_version": "table2-semantic-dependency-lock-v2",
        "record_type": "Table2SemanticDependencyLock",
        "claim_scope": (
            "BROWSER_MEASURED_DGX_SUPPLIED_COMPATIBILITY_NOT_DISPATCH_AUTHORITY"
        ),
        "paper_table_status": "N/R",
        **core,
        "semantic_identity_sha256": _sha256_json(core),
    }
    if lock != expected:
        raise RuntimeError(
            "split dependency lock differs from browser/DGX measured authority"
        )

    # This process owns the browser/control plane. The remote inference service
    # must independently run the registered dgx_host validation at startup and
    # before every physical block; a browser process cannot attest remote state.
    current_host = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable_sha256": _sha256_file(Path(sys.executable)),
    }
    if current_host != expected["browser_host"]["host"]:
        raise RuntimeError(
            "evaluation bootstrap browser-host Python/platform identity differs"
        )
    try:
        current_packages = [
            {
                "distribution": distribution,
                "version": importlib_metadata.version(distribution),
            }
            for distribution in sorted(PINNED_SEMANTIC_DEPENDENCIES)
        ]
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "evaluation bootstrap browser host lacks a pinned dependency"
        ) from exc
    if current_packages != expected["browser_host"]["packages"]:
        raise RuntimeError(
            "evaluation bootstrap browser-host package versions differ from lock"
        )
    raise RuntimeError(
        "split deployment dispatch is blocked: no externally trusted DGX "
        "startup/per-block remeasurement receipt schema or trust anchor is registered"
    )


def _bootstrap_validate_semantic_dependency_lock(
    *, campaign_root: Path, environment: Mapping[str, Any], dependency_lock: Path
) -> None:
    """Stdlib-only semantic replay before importing evaluation package code."""

    lock = _read_mapping(dependency_lock)
    topology = lock.get("deployment_topology")
    expected_schema = (
        "table2-semantic-dependency-lock-v2"
        if topology == "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"
        else "table2-semantic-dependency-lock-v1"
    )
    expected_claim_scope = (
        "BROWSER_MEASURED_DGX_SUPPLIED_COMPATIBILITY_NOT_DISPATCH_AUTHORITY"
        if topology == "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"
        else "MEASURED_HOST_DEPENDENCIES_NOT_CAMPAIGN_RESULTS"
    )
    if (
        lock.get("schema_version") != expected_schema
        or lock.get("record_type") != "Table2SemanticDependencyLock"
        or lock.get("claim_scope") != expected_claim_scope
        or lock.get("paper_table_status") != "N/R"
    ):
        raise RuntimeError("evaluation bootstrap dependency lock is not semantic")
    preflight = _read_mapping(
        campaign_root / "frozen" / "webarena_deployment_preflight.json"
    )
    if topology == "SPLIT_LOCAL_BROWSER_DGX_INFERENCE":
        _bootstrap_validate_split_dependency_lock(
            environment=environment,
            preflight=preflight,
            lock=lock,
        )
        return
    host_preflight = preflight if topology == "SINGLE_DGX_HOST" else None
    if not isinstance(host_preflight, dict):
        raise RuntimeError("evaluation bootstrap dependency topology is invalid")
    package_rows = host_preflight.get("package_check", {}).get("packages")
    browser = host_preflight.get("browser_check")
    host = host_preflight.get("host")
    if (
        not isinstance(package_rows, list)
        or not isinstance(browser, dict)
        or not isinstance(host, dict)
    ):
        raise RuntimeError("evaluation bootstrap dependency preflight is incomplete")
    package_by_name = {
        row.get("distribution"): row
        for row in package_rows
        if isinstance(row, dict)
    }
    if set(package_by_name) != set(PINNED_SEMANTIC_DEPENDENCIES):
        raise RuntimeError("evaluation bootstrap dependency package set differs")
    for distribution, expected_version in PINNED_SEMANTIC_DEPENDENCIES.items():
        row = package_by_name[distribution]
        if (
            row.get("status") != "PASS"
            or row.get("expected_version") != expected_version
            or row.get("actual_version") != expected_version
        ):
            raise RuntimeError(
                f"evaluation bootstrap dependency {distribution} differs"
            )
    if (
        browser.get("browser") != "chromium"
        or browser.get("viewport") != {"width": 1280, "height": 720}
        or browser.get("device_scale_factor") != 1
        or environment.get("browser_version") != browser.get("browser_version")
        or environment.get("playwright_version")
        != PINNED_SEMANTIC_DEPENDENCIES["playwright"]
        or str(host.get("system") or "").casefold()
        not in str(environment.get("operating_system") or "").casefold()
    ):
        raise RuntimeError(
            "evaluation bootstrap browser/platform dependency identity differs"
        )
    packages = sorted(
        [
            {
                "distribution": row.get("distribution"),
                "version": row.get("actual_version"),
            }
            for row in package_rows
            if isinstance(row, dict)
        ],
        key=lambda row: str(row["distribution"]),
    )
    core = {
        "deployment_topology": topology,
        "host": {
            field: host.get(field)
            for field in (
                "system",
                "release",
                "machine",
                "python_version",
                "python_executable_sha256",
            )
        },
        "packages": packages,
        "browser": {
            "engine": browser.get("browser"),
            "version": browser.get("browser_version"),
            "viewport": browser.get("viewport"),
            "device_scale_factor": browser.get("device_scale_factor"),
        },
        "environment": {
            field: environment.get(field)
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
        },
        "deployment_preflight_content_sha256": _sha256_json(preflight),
    }
    expected = {
        "schema_version": "table2-semantic-dependency-lock-v1",
        "record_type": "Table2SemanticDependencyLock",
        "claim_scope": "MEASURED_HOST_DEPENDENCIES_NOT_CAMPAIGN_RESULTS",
        "paper_table_status": "N/R",
        **core,
        "semantic_identity_sha256": _sha256_json(core),
    }
    if lock != expected:
        raise RuntimeError(
            "evaluation bootstrap dependency lock differs from measured preflight/environment"
        )
    current_host = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable_sha256": _sha256_file(Path(sys.executable)),
    }
    if current_host != lock.get("host"):
        raise RuntimeError(
            "evaluation bootstrap current Python/platform identity differs from lock"
        )
    try:
        current_packages = [
            {
                "distribution": distribution,
                "version": importlib_metadata.version(distribution),
            }
            for distribution in sorted(PINNED_SEMANTIC_DEPENDENCIES)
        ]
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "evaluation bootstrap current host lacks a pinned dependency"
        ) from exc
    if current_packages != lock.get("packages"):
        raise RuntimeError(
            "evaluation bootstrap current package versions differ from lock"
        )


def _lexical_absolute_path(value: str | Path) -> Path:
    """Return an absolute path without following symlink components."""

    return Path(os.path.abspath(os.fspath(value)))


def _assert_no_existing_symlink_components(path: Path, *, label: str) -> None:
    """Inspect the unresolved leaf and every existing parent with ``lstat``."""

    absolute = _lexical_absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{label} must not contain a symlink: {current}")


def _assert_tree_disjoint(
    path: Path,
    *,
    protected_roots: tuple[Path, ...],
    label: str,
    must_exist: bool = True,
) -> Path:
    """Require a resolved path to be neither ancestor nor descendant of roots."""

    resolved = path.resolve(strict=must_exist)
    if resolved == Path(resolved.anchor):
        raise RuntimeError(f"{label} cannot be a filesystem root")
    for protected in protected_roots:
        protected_root = protected.resolve()
        if (
            resolved == protected_root
            or resolved in protected_root.parents
            or protected_root in resolved.parents
        ):
            raise RuntimeError(
                f"{label} must be tree-disjoint from campaign and source roots"
            )
    return resolved


def _validated_external_boundary_receipt_path(
    path: Path,
    *,
    campaign_root: Path,
) -> Path:
    unresolved = _lexical_absolute_path(path)
    _assert_no_existing_symlink_components(
        unresolved, label="provider boundary receipt"
    )
    if not unresolved.is_file():
        raise RuntimeError("external provider boundary receipt is missing")
    repository_root = Path(__file__).resolve().parents[1]
    return _assert_tree_disjoint(
        unresolved,
        protected_roots=(campaign_root, repository_root),
        label="provider boundary receipt",
    )


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"evaluation bootstrap expected a JSON object: {path}")
    return value


def _bootstrap_assert_pc01_page_broker_isolation(
    attestation: Mapping[str, Any],
) -> None:
    """Stop before provider import while broker evidence is unpromotable.

    This standard-library-only check intentionally duplicates the frozen
    non-claim in ``execution_guard``. Importing the evaluation package merely
    to discover this blocker would already import runtime integration modules.
    """

    binding = attestation.get(PC01_PAGE_BROKER_SECURITY_FIELD)
    if not isinstance(binding, dict):
        raise RuntimeError(
            "evaluation bootstrap has no authenticated page-broker security status"
        )
    legacy_blocked = binding == PC01_PAGE_BROKER_SECURITY_BLOCKED_BINDING
    source_rows = binding.get("architecture_source_files")
    expected_rows = {
        str(row.get("relative_path")): str(row.get("sha256"))
        for row in attestation.get("source_files", [])
        if isinstance(row, dict)
    }
    process_blocked = (
        binding.get("schema_version") == "table2-pc01-page-broker-security-v3"
        and binding.get("status")
        == "BLOCKED_INNER_SCHEMAS_VALUE_PROVENANCE_AND_EXTERNAL_RECEIPT_REQUIRED"
        and binding.get("claim_scope")
        == (
            "SOURCE_ATTESTED_DISTINCT_PROCESS_AND_KEY_ENVELOPE_ARCHITECTURE_"
            "NOT_VALUE_PROVENANCE_OR_DEPLOYMENT_AUTHORITY"
        )
        and binding.get("architecture")
        == "separate_process_af_unix_json_hmac_sha256_peercred_v1"
        and binding.get("runtime_and_evaluator_process_roles_separate") is True
        and binding.get("outer_envelope_fields_exact") is True
        and binding.get("forbidden_named_keys_rejected_recursively") is True
        and binding.get("arbitrary_nested_mapping_paths")
        == PC01_PROCESS_BROKER_ARBITRARY_MAPPING_PATHS
        and binding.get("operation_specific_inner_schemas_registered") is False
        and binding.get("runtime_value_provenance_attested") is False
        and binding.get("evaluator_operation_in_runtime_allowlist") is False
        and binding.get("authenticated_outer_envelopes") is True
        and binding.get("role_separated_authentication") is True
        and binding.get("future_promotion_requirements")
        == PC01_PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
        and binding.get("same_process_fixture_production_eligible") is False
        and binding.get("local_receipt_schema_version")
        == "table2-process-page-broker-receipt-v2"
        and binding.get("local_cleanup_receipt_schema_version")
        == "table2-process-page-broker-cleanup-receipt-v1"
        and binding.get("external_deployment_receipt_schema_version") is None
        and binding.get("external_deployment_receipt_present") is False
        and binding.get("external_trust_anchor_registered") is False
        and binding.get("production_dispatch_authorized") is False
        and isinstance(source_rows, list)
        and [row.get("relative_path") for row in source_rows]
        == sorted(PC01_PROCESS_BROKER_SOURCE_PATHS)
        and all(
            isinstance(row, dict)
            and set(row) == {"relative_path", "sha256"}
            and expected_rows.get(str(row.get("relative_path"))) == row.get("sha256")
            for row in source_rows
        )
        and binding.get("architecture_source_set_sha256")
        == hashlib.sha256(
            json.dumps(
                source_rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        and set(binding)
        == {
            "schema_version",
            "status",
            "claim_scope",
            "architecture",
            "runtime_and_evaluator_process_roles_separate",
            "outer_envelope_fields_exact",
            "forbidden_named_keys_rejected_recursively",
            "arbitrary_nested_mapping_paths",
            "operation_specific_inner_schemas_registered",
            "runtime_value_provenance_attested",
            "evaluator_operation_in_runtime_allowlist",
            "authenticated_outer_envelopes",
            "role_separated_authentication",
            "future_promotion_requirements",
            "same_process_fixture_production_eligible",
            "local_receipt_schema_version",
            "local_cleanup_receipt_schema_version",
            "architecture_source_files",
            "architecture_source_set_sha256",
            "external_deployment_receipt_schema_version",
            "external_deployment_receipt_present",
            "external_trust_anchor_registered",
            "production_dispatch_authorized",
        }
    )
    if not legacy_blocked and not process_blocked:
        raise RuntimeError(
            "evaluation bootstrap has no authenticated page-broker security status"
        )
    raise RuntimeError(
        "PC-01 live campaign is blocked before provider import: local broker "
        "evidence covers distinct processes, exact outer envelopes, and named-key "
        "rejection only; operation-specific inner schemas, runtime value provenance, "
        "and a separately authenticated deployment receipt are required"
    )


def _bootstrap_verify_evaluation_source(
    campaign_dir: Path,
    *,
    runner_entrypoint: str,
    provider_factory_entrypoint: str | None = None,
) -> None:
    """Use only the standard library before importing the evaluation package."""

    campaign_root = campaign_dir.resolve()
    manifest = _read_mapping(campaign_root / "campaign_manifest.json")
    if manifest.get("campaign_mode") == "smoke":
        return
    if manifest.get("runner_identity_scope") != EVALUATION_RUNNER_SCOPE:
        raise RuntimeError("evaluation bootstrap lacks a frozen runner identity")
    attestation_path = campaign_root / "frozen" / "runner_attestation.json"
    if _sha256_file(attestation_path) != manifest.get("runner_attestation_sha256"):
        raise RuntimeError("evaluation bootstrap runner attestation hash mismatch")
    attestation = _read_mapping(attestation_path)
    if runner_entrypoint != attestation.get("runner_entrypoint"):
        raise RuntimeError("evaluation bootstrap runner entrypoint mismatch")
    provider_binding = attestation.get("pc01_operations_provider_bootstrap")
    if runner_entrypoint == PC01_PRODUCTION_RUNNER_ENTRYPOINT:
        if not isinstance(provider_binding, dict):
            raise RuntimeError("evaluation bootstrap lacks provider-factory identity")
        expected_fields = {
            "schema_version",
            "factory_entrypoint",
            "factory_module",
            "factory_qualname",
            "source_relative_path",
            "source_sha256",
            "provider_contract_schema_version",
            "expected_provider_public_contract_sha256",
            "source_plane",
        }
        if set(provider_binding) != expected_fields:
            raise RuntimeError("evaluation bootstrap provider identity is malformed")
        if (
            provider_factory_entrypoint is None
            or provider_factory_entrypoint != provider_binding["factory_entrypoint"]
        ):
            raise RuntimeError(
                "PC-01 provider factory entrypoint differs from frozen identity"
            )
        module_name, separator, attribute_name = provider_factory_entrypoint.partition(":")
        if (
            separator != ":"
            or not module_name
            or not attribute_name
            or "." in attribute_name
            or provider_binding.get("factory_module") != module_name
            or provider_binding.get("factory_qualname") != attribute_name
            or provider_binding.get("source_plane") != "runtime_only"
        ):
            raise RuntimeError("evaluation bootstrap provider entrypoint is not exact")
        _bootstrap_assert_pc01_page_broker_isolation(attestation)

    repository_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("evaluation bootstrap requires a Git checkout") from exc
    if status:
        raise RuntimeError("evaluation bootstrap requires a clean Git checkout")
    if (
        not commit
        or commit != manifest.get("repository_commit")
        or commit != attestation.get("repository_commit")
    ):
        raise RuntimeError("evaluation bootstrap Git commit differs from campaign")

    rows = attestation.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("evaluation bootstrap source attestation is malformed")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("evaluation bootstrap source row is malformed")
        relative_text = str(row.get("relative_path", ""))
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise RuntimeError("evaluation bootstrap source path is unsafe or duplicate")
        seen.add(relative_text)
        live = (repository_root / relative).resolve()
        frozen = (campaign_root / "frozen" / "runner_source" / relative).resolve()
        expected = row.get("sha256")
        if (
            repository_root not in live.parents
            or campaign_root not in frozen.parents
            or not live.is_file()
            or not frozen.is_file()
            or _sha256_file(live) != expected
            or _sha256_file(frozen) != expected
        ):
            raise RuntimeError(
                f"evaluation bootstrap source identity mismatch: {relative_text}"
            )
    own_relative = Path(__file__).resolve().relative_to(repository_root).as_posix()
    if own_relative not in seen:
        raise RuntimeError("evaluation CLI source is absent from runner attestation")
    if isinstance(provider_binding, dict):
        provider_relative = str(provider_binding.get("source_relative_path") or "")
        if (
            provider_relative not in seen
            or provider_binding.get("source_sha256")
            != next(
                (
                    row.get("sha256")
                    for row in rows
                    if isinstance(row, dict)
                    and row.get("relative_path") == provider_relative
                ),
                None,
            )
        ):
            raise RuntimeError(
                "evaluation bootstrap provider source differs from frozen identity"
            )

    environment = _read_mapping(campaign_root / "frozen" / "environment.json")
    if (
        environment.get("dependency_lock_relative_path")
        != FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH
    ):
        raise RuntimeError("evaluation bootstrap dependency-lock path is not frozen")
    dependency_lock = campaign_root / "frozen" / FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH
    if (
        dependency_lock.is_symlink()
        or not dependency_lock.is_file()
        or _sha256_file(dependency_lock) != environment.get("dependency_lock_sha256")
    ):
        raise RuntimeError("evaluation bootstrap dependency-lock hash mismatch")
    _bootstrap_validate_semantic_dependency_lock(
        campaign_root=campaign_root,
        environment=environment,
        dependency_lock=dependency_lock,
    )


def _callable_source_relative(callback: Any, repository_root: Path) -> str:
    target = callback
    if inspect.ismethod(target):
        target = target.__func__
    elif not (inspect.isfunction(target) or inspect.isclass(target)):
        target = getattr(type(target), "__call__", None)
    if target is None:
        raise RuntimeError("PC-01 operations provider factory source is unresolved")
    try:
        source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "PC-01 operations provider factory source is unresolved"
        ) from exc
    if not source_name:
        raise RuntimeError("PC-01 operations provider factory has no source file")
    unresolved = Path(source_name).absolute()
    if unresolved.is_symlink():
        raise RuntimeError("PC-01 operations provider factory source is a symlink")
    source = unresolved.resolve()
    try:
        return source.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "PC-01 operations provider factory source is outside the repository"
        ) from exc


@dataclass(frozen=True, slots=True)
class _ProviderInstallPreflight:
    context: Any
    binding: Mapping[str, str]
    source_hashes: Mapping[str, str]
    campaign_state_sha256: str


def _campaign_state_sha256(campaign_root: Path) -> str:
    paths = (
        campaign_root / "campaign_manifest.json",
        campaign_root / "frozen/runner_attestation.json",
        campaign_root / "frozen/environment.json",
        campaign_root / "frozen/protocol.yaml",
    )
    return hashlib.sha256(
        "".join(f"{path.name}:{_sha256_file(path)}\n" for path in paths).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_provider_source_plane(
    *,
    provider_binding: Mapping[str, str],
    validated_live: Any,
) -> None:
    manifest = getattr(validated_live, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("PC-01 live-deployment capability planes are absent")
    capabilities = manifest.get("capabilities")
    broker = manifest.get("sealed_page_broker")
    if not isinstance(capabilities, Mapping) or not isinstance(broker, Mapping):
        raise RuntimeError("PC-01 live-deployment capability planes are malformed")
    sealed = capabilities.get("sealed_webarena_evaluator")
    if not isinstance(sealed, Mapping):
        raise RuntimeError("PC-01 sealed evaluator source authority is absent")
    runtime_sources = {
        (
            str(row.get("source_relative_path") or ""),
            str(row.get("source_sha256") or ""),
        )
        for capability_id, row in capabilities.items()
        if capability_id != "sealed_webarena_evaluator" and isinstance(row, Mapping)
    }
    source = (
        str(provider_binding["source_relative_path"]),
        str(provider_binding["source_sha256"]),
    )
    forbidden = {
        (
            str(sealed.get("source_relative_path") or ""),
            str(sealed.get("source_sha256") or ""),
        ),
        (
            str(broker.get("source_relative_path") or ""),
            str(broker.get("source_sha256") or ""),
        ),
    }
    if source not in runtime_sources or source in forbidden:
        raise RuntimeError(
            "PC-01 provider factory source must be runtime-only, never sealed, "
            "broker, or shared-plane"
        )


def _preflight_pc01_provider_install(
    campaign_root: Path,
    *,
    credential_capability_root: Path,
    credential_capability_id: str,
    credential_capability_version: str,
) -> _ProviderInstallPreflight:
    """Run all campaign and causal guards before importing/calling the factory."""

    # Keep this check standard-library-only and ahead of every runtime/provider
    # import.  The current broker is an in-process engineering fixture whose
    # evaluator accessor is importable by any code in this interpreter.  A
    # caller that bypasses ``main`` must therefore fail at the same boundary as
    # the canonical CLI, before importing the integration module that assembles
    # that fixture.
    campaign_root = campaign_root.resolve()
    _bootstrap_assert_pc01_page_broker_isolation(
        _read_mapping(campaign_root / "frozen" / "runner_attestation.json")
    )

    from web_agent.eval.table2.execution_guard import (
        PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD,
        assert_clean_git_checkout,
        attested_source_hashes,
        validate_pc01_provider_bootstrap_binding,
    )
    from web_agent.eval.table2.locked_mount_preflight import (
        assert_production_locked_mount_preflight,
    )
    from web_agent.eval.table2.live_deployment import (
        validate_bound_pc01_live_deployment,
    )
    from web_agent.eval.table2.package_validator import validate_campaign
    from web_agent.eval.table2.production_runner import (
        build_runtime_capability_authority,
        runtime_context_identity,
        runtime_deployment_preflight_view,
        _read_json_or_yaml_mapping,
    )
    from web_agent.eval.table2.webarena_preflight_binding import (
        validate_bound_deployment_preflight,
    )
    from web_agent.runtime.pc01_live_integration import (
        PC01ExternalCredentialCapability,
        PC01ProviderBootstrapContext,
        validate_external_credential_capability_root,
    )

    repository_root = Path(__file__).resolve().parents[1]
    report = validate_campaign(
        campaign_root,
        require_complete=False,
        require_aggregates=False,
    )
    if not report.passed:
        raise RuntimeError(
            "PC-01 provider preflight campaign validation failed: "
            + "; ".join(report.errors)
        )
    commit = assert_clean_git_checkout(repository_root)
    manifest = _read_mapping(campaign_root / "campaign_manifest.json")
    attestation = _read_mapping(campaign_root / "frozen/runner_attestation.json")
    environment = _read_mapping(campaign_root / "frozen/environment.json")
    protocol = _read_json_or_yaml_mapping(campaign_root / "frozen/protocol.yaml")
    if (
        manifest.get("campaign_mode") != "evaluation"
        or manifest.get("evidence_label") != "PILOT_ONLY"
        or protocol.get("protocol_id") != "table2-pc01-pilot-v1"
        or protocol.get("evidence_label") != "PILOT_ONLY"
        or protocol.get("paper_table_status") != "N/R"
        or manifest.get("repository_commit") != commit
        or attestation.get("repository_commit") != commit
    ):
        raise RuntimeError("PC-01 provider preflight violates PILOT_ONLY/N-R identity")
    assert_production_locked_mount_preflight(
        repository_root=repository_root,
        protocol=protocol,
        environment=environment,
    )
    validated_live = validate_bound_pc01_live_deployment(
        environment,
        artifact_root=campaign_root / "frozen",
        repository_root=repository_root,
    )
    deployment_preflight = validate_bound_deployment_preflight(
        environment,
        artifact_root=campaign_root / "frozen",
    )
    source_hashes = attested_source_hashes(attestation)
    provider_binding = validate_pc01_provider_bootstrap_binding(
        attestation.get(PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD),
        repository_root=repository_root,
        source_hashes=source_hashes,
    )
    _validate_provider_source_plane(
        provider_binding=provider_binding,
        validated_live=validated_live,
    )
    try:
        credential_root = validate_external_credential_capability_root(
            credential_capability_root,
            forbidden_roots=(campaign_root, repository_root),
        )
    except Exception as exc:
        raise RuntimeError(
            "credential capability root is not tree-disjoint and symlink-free"
        ) from exc
    credential_capability = PC01ExternalCredentialCapability(
        capability_id=credential_capability_id,
        capability_version=credential_capability_version,
        root=credential_root,
    )
    runtime_identity = runtime_context_identity(attestation["runtime_identity"])
    preflight_view = runtime_deployment_preflight_view(deployment_preflight)
    authority = build_runtime_capability_authority(
        validated_live,
        deployment_preflight,
        expected_provider_public_contract_sha256=provider_binding[
            "expected_provider_public_contract_sha256"
        ],
    )
    context = PC01ProviderBootstrapContext(
        schema_version="table2-pc01-provider-bootstrap-v1",
        protocol_id=str(protocol["protocol_id"]),
        model_seed=42,
        repository_commit=commit,
        runtime_identity_sha256=hashlib.sha256(
            json.dumps(
                runtime_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        runtime_capability_authority=authority,
        runtime_environment=runtime_identity.get("environment", {}),
        deployment_preflight_view=preflight_view,
        service_url_map=dict(deployment_preflight.service_url_map),
        credential_capability=credential_capability,
    )
    return _ProviderInstallPreflight(
        context=context,
        binding=provider_binding,
        source_hashes=source_hashes,
        campaign_state_sha256=_campaign_state_sha256(campaign_root),
    )


def _bootstrap_context_identity(context: Any) -> Mapping[str, Any]:
    return {
        "schema_version": context.schema_version,
        "protocol_id": context.protocol_id,
        "model_seed": context.model_seed,
        "repository_commit": context.repository_commit,
        "runtime_identity_sha256": context.runtime_identity_sha256,
        "runtime_capability_authority": {
            "schema_version": context.runtime_capability_authority["schema_version"],
            "deployment_preflight_binding_sha256": context.runtime_capability_authority[
                "deployment_preflight_binding_sha256"
            ],
            "expected_provider_public_contract_sha256": (
                context.runtime_capability_authority[
                    "expected_provider_public_contract_sha256"
                ]
            ),
            "capability_set_sha256": context.runtime_capability_authority[
                "capability_set_sha256"
            ],
        },
        "runtime_environment": dict(context.runtime_environment),
        "deployment_preflight_view": dict(context.deployment_preflight_view),
        "service_url_map": dict(context.service_url_map),
        "credential_capability": dict(context.credential_capability.public_identity),
    }


def _pc01_provider_boundary_receipt_value(
    preflight: _ProviderInstallPreflight,
) -> Mapping[str, Any]:
    from web_agent.eval.table2.common import sha256_json

    context_identity = _bootstrap_context_identity(preflight.context)
    measurement = {
        "claim_scope": "REVIEWED_CODE_ORACLE_FREE_DATAFLOW_ONLY",
        "factory_entrypoint": preflight.binding["factory_entrypoint"],
        "factory_source_sha256": preflight.binding["source_sha256"],
        "expected_provider_public_contract_sha256": preflight.binding[
            "expected_provider_public_contract_sha256"
        ],
        "runtime_identity_sha256": preflight.context.runtime_identity_sha256,
        "bootstrap_context_sha256": sha256_json(context_identity),
        "campaign_state_sha256": preflight.campaign_state_sha256,
        "same_process_factory": True,
        "kernel_filesystem_sandbox": False,
        "campaign_directory_argument_passed": False,
        "campaign_artifact_path_passed": False,
        "task_evaluator_memory_model_or_sealed_content_passed": False,
        "external_credential_capability_only": True,
    }
    return {
        "schema_version": "table2-pc01-provider-boundary-receipt-v1",
        "record_type": "PC01ProviderBoundaryReceipt",
        "status": "PASS",
        **measurement,
        "measurement_sha256": sha256_json(measurement),
    }


def _validate_pc01_provider_boundary_receipt(
    receipt_path: Path,
    *,
    campaign_root: Path,
    preflight: _ProviderInstallPreflight,
) -> Mapping[str, Any]:
    """Validate the attainable reviewed-code/dataflow boundary receipt.

    This receipt deliberately says that no kernel filesystem sandbox exists.
    It binds the exact source-attested function and the path-free typed argument
    that this CLI actually invokes; it is not evidence against hostile trusted
    source code.
    """

    candidate = _validated_external_boundary_receipt_path(
        receipt_path,
        campaign_root=campaign_root,
    )
    expected = _pc01_provider_boundary_receipt_value(preflight)
    receipt = _read_mapping(candidate)
    if dict(receipt) != expected:
        raise RuntimeError(
            "provider boundary receipt differs from the measured reviewed-code/"
            "oracle-free bootstrap"
        )
    return receipt


def _write_pc01_provider_boundary_receipt(
    output_path: Path,
    *,
    campaign_root: Path,
    preflight: _ProviderInstallPreflight,
) -> Path:
    """Write the deterministic receipt operators later supply to the run."""

    output = _lexical_absolute_path(output_path)
    _assert_no_existing_symlink_components(
        output, label="provider boundary receipt output"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_existing_symlink_components(
        output, label="provider boundary receipt output"
    )
    repository_root = Path(__file__).resolve().parents[1]
    output = _assert_tree_disjoint(
        output,
        protected_roots=(campaign_root, repository_root),
        label="provider boundary receipt output",
        must_exist=False,
    )
    if output.exists() and not output.is_file():
        raise RuntimeError("provider boundary receipt output is not a regular file")
    value = _pc01_provider_boundary_receipt_value(preflight)
    serialized = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if output.exists():
        if output.read_bytes() != serialized:
            raise RuntimeError(
                "existing provider boundary receipt differs; immutable evidence "
                "will not be overwritten"
            )
        return output.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        # Recheck the unresolved destination immediately before atomic replace.
        _assert_no_existing_symlink_components(
            output, label="provider boundary receipt output"
        )
        os.replace(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output.resolve(strict=True)


def _load_exact_provider_factory(
    entrypoint: str,
    *,
    binding: Mapping[str, str],
    source_hashes: Mapping[str, str],
) -> Any:
    from web_agent.eval.table2.campaign import load_entrypoint
    from web_agent.eval.table2.execution_guard import validate_attested_callable_source

    if entrypoint != binding.get("factory_entrypoint"):
        raise RuntimeError("PC-01 provider factory entrypoint differs from frozen identity")
    factory = load_entrypoint(entrypoint)
    if not inspect.isfunction(factory):
        raise RuntimeError(
            "PC-01 operations provider factory must be an exact function, never a "
            "partial, class, or callable object"
        )
    if (
        factory.__module__ != binding["factory_module"]
        or factory.__qualname__ != binding["factory_qualname"]
    ):
        raise RuntimeError("PC-01 operations provider factory callable identity differs")
    repository_root = Path(__file__).resolve().parents[1]
    relative = _callable_source_relative(factory, repository_root)
    if (
        relative != binding["source_relative_path"]
        or _sha256_file(repository_root / relative) != binding["source_sha256"]
    ):
        raise RuntimeError("PC-01 operations provider factory source identity differs")
    validate_attested_callable_source(
        factory,
        repository_root=repository_root,
        source_hashes=source_hashes,
        field="pc01_operations_provider_factory",
        expected_relative_path=binding["source_relative_path"],
    )
    parameters = tuple(inspect.signature(factory).parameters.values())
    if (
        len(parameters) != 1
        or parameters[0].kind
        not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ):
        raise RuntimeError(
            "PC-01 provider factory must accept exactly one positional bootstrap context"
        )
    return factory


def _build_pc01_provider_installation_receipt(
    *,
    provider: Any,
    before: _ProviderInstallPreflight,
    after: _ProviderInstallPreflight,
    provider_boundary_receipt: Path,
    campaign_root: Path,
) -> Any:
    """Build immutable post-factory evidence without serializing secret paths."""

    from web_agent.eval.table2.common import sha256_json
    from web_agent.eval.table2.execution_guard import (
        PC01_PROVIDER_BOUNDARY_CLAIM_SCOPE,
        PC01_PROVIDER_INSTALLATION_RECEIPT_SCHEMA_VERSION,
        PC01ProviderInstallationReceipt,
    )

    boundary_path = _validated_external_boundary_receipt_path(
        provider_boundary_receipt,
        campaign_root=campaign_root,
    )
    return PC01ProviderInstallationReceipt(
        schema_version=PC01_PROVIDER_INSTALLATION_RECEIPT_SCHEMA_VERSION,
        record_type="PC01ProviderInstallationReceipt",
        status="PASS",
        claim_scope=PC01_PROVIDER_BOUNDARY_CLAIM_SCOPE,
        same_process_factory=True,
        kernel_filesystem_sandbox=False,
        factory_entrypoint=str(before.binding["factory_entrypoint"]),
        factory_module=str(before.binding["factory_module"]),
        factory_qualname=str(before.binding["factory_qualname"]),
        factory_source_relative_path=str(before.binding["source_relative_path"]),
        factory_source_sha256=str(before.binding["source_sha256"]),
        bootstrap_context_sha256=sha256_json(
            _bootstrap_context_identity(before.context)
        ),
        pre_factory_campaign_state_sha256=before.campaign_state_sha256,
        post_factory_campaign_state_sha256=after.campaign_state_sha256,
        provider_boundary_receipt_sha256=_sha256_file(boundary_path),
        credential_public_identity_sha256=sha256_json(
            dict(before.context.credential_capability.public_identity)
        ),
        expected_provider_public_contract_sha256=str(
            before.binding["expected_provider_public_contract_sha256"]
        ),
        actual_provider_public_contract_sha256=str(
            provider.public_contract_sha256
        ),
    )


def _install_pc01_operations_provider(
    campaign_dir: Path,
    *,
    provider_factory_entrypoint: str,
    credential_capability_root: Path,
    credential_capability_id: str,
    credential_capability_version: str,
    provider_boundary_receipt: Path,
) -> Any:
    """Load, attest, call and register one provider before runner creation."""

    if not provider_factory_entrypoint or ":" not in provider_factory_entrypoint:
        raise RuntimeError(
            "canonical PC-01 production requires "
            "--pc01-operations-provider-factory module:attribute"
        )
    campaign_root = campaign_dir.resolve()
    before = _preflight_pc01_provider_install(
        campaign_root,
        credential_capability_root=credential_capability_root,
        credential_capability_id=credential_capability_id,
        credential_capability_version=credential_capability_version,
    )
    # Imports below are intentionally behind the fail-closed preflight.  They
    # are reachable only in deterministic tests that substitute an explicitly
    # non-production preflight, until a future registered process-isolation
    # implementation replaces the blocked binding.
    from web_agent.runtime.pc01_live_integration import (
        PC01LiveOperationsProvider,
        register_pc01_live_operations,
        validate_provider_public_contract,
    )
    from web_agent.eval.table2.package_validator import append_campaign_ledger_event
    if provider_factory_entrypoint != before.binding["factory_entrypoint"]:
        raise RuntimeError("PC-01 provider factory entrypoint differs from frozen identity")
    _validate_pc01_provider_boundary_receipt(
        provider_boundary_receipt,
        campaign_root=campaign_root,
        preflight=before,
    )
    factory = _load_exact_provider_factory(
        provider_factory_entrypoint,
        binding=before.binding,
        source_hashes=before.source_hashes,
    )
    provider = factory(before.context)
    if type(provider) is not PC01LiveOperationsProvider:
        raise RuntimeError(
            "PC-01 operations provider factory returned the wrong exact type"
        )
    validate_provider_public_contract(
        provider,
        before.context.runtime_capability_authority,
        expected_preflight_view=before.context.deployment_preflight_view,
    )
    if provider.expected_runtime_identity_sha256 != before.context.runtime_identity_sha256:
        raise RuntimeError("PC-01 provider runtime identity differs from bootstrap")
    after = _preflight_pc01_provider_install(
        campaign_root,
        credential_capability_root=credential_capability_root,
        credential_capability_id=credential_capability_id,
        credential_capability_version=credential_capability_version,
    )
    if (
        after.campaign_state_sha256 != before.campaign_state_sha256
        or after.binding != before.binding
        or after.context != before.context
    ):
        raise RuntimeError("PC-01 campaign/live authority changed during factory call")
    validate_provider_public_contract(
        provider,
        after.context.runtime_capability_authority,
        expected_preflight_view=after.context.deployment_preflight_view,
    )
    installation_receipt = _build_pc01_provider_installation_receipt(
        provider=provider,
        before=before,
        after=after,
        provider_boundary_receipt=provider_boundary_receipt,
        campaign_root=campaign_root,
    )
    register_pc01_live_operations(provider, installation_receipt)
    append_campaign_ledger_event(
        campaign_root,
        ledger_type="access",
        event_type="pc01_provider_installation",
        payload={
            "installation_receipt": installation_receipt.to_dict(),
            "installation_receipt_sha256": installation_receipt.receipt_sha256,
            "locked_test_content": False,
        },
    )
    return installation_receipt


def main() -> None:
    args = parse_args()
    provider_installation_receipt = None
    provider_factory_arg = getattr(args, "pc01_operations_provider_factory", None)
    credential_root_arg = getattr(args, "pc01_credential_capability_root", None)
    credential_id_arg = getattr(args, "pc01_credential_capability_id", None)
    credential_version_arg = getattr(
        args, "pc01_credential_capability_version", None
    )
    boundary_receipt_arg = getattr(args, "pc01_provider_boundary_receipt", None)
    prepare_boundary_receipt_arg = getattr(
        args, "prepare_pc01_provider_boundary_receipt", None
    )
    if args.maximum_blocks is not None and args.maximum_blocks <= 0:
        raise ValueError("--maximum-blocks must be positive")
    if args.live_readiness_probe_for is not None:
        if args.block_id is None:
            raise ValueError("--live-readiness-probe-for requires --block-id")
        if args.maximum_blocks is not None:
            raise ValueError(
                "--live-readiness-probe-for cannot be combined with --maximum-blocks"
            )
    canonical_pc01 = args.runner == PC01_PRODUCTION_RUNNER_ENTRYPOINT
    if canonical_pc01:
        if not args.runner_factory:
            raise ValueError(
                "canonical PC-01 production runner requires --runner-factory"
            )
        if provider_factory_arg is None:
            raise ValueError(
                "canonical PC-01 production runner requires an explicit "
                "--pc01-operations-provider-factory"
            )
        if (
            credential_root_arg is None
            or not credential_id_arg
            or not credential_version_arg
            or (
                boundary_receipt_arg is None
                and prepare_boundary_receipt_arg is None
            )
        ):
            raise ValueError(
                "canonical PC-01 production requires an explicit external "
                "credential capability root/id/version and provider-boundary receipt"
            )
        if (
            boundary_receipt_arg is not None
            and prepare_boundary_receipt_arg is not None
        ):
            raise ValueError(
                "prepare and consume provider-boundary receipt modes are exclusive"
            )
    elif any(
        value is not None
        for value in (
            provider_factory_arg,
            credential_root_arg,
            credential_id_arg,
            credential_version_arg,
            boundary_receipt_arg,
            prepare_boundary_receipt_arg,
        )
    ):
        raise ValueError(
            "--pc01-operations-provider-factory is forbidden for non-PC-01 runners"
        )
    _bootstrap_verify_evaluation_source(
        args.campaign_dir,
        runner_entrypoint=args.runner,
        provider_factory_entrypoint=provider_factory_arg,
    )
    if canonical_pc01:
        if prepare_boundary_receipt_arg is not None:
            preflight = _preflight_pc01_provider_install(
                args.campaign_dir.resolve(),
                credential_capability_root=credential_root_arg,
                credential_capability_id=credential_id_arg,
                credential_capability_version=credential_version_arg,
            )
            output = _write_pc01_provider_boundary_receipt(
                prepare_boundary_receipt_arg,
                campaign_root=args.campaign_dir.resolve(),
                preflight=preflight,
            )
            print(json.dumps({"provider_boundary_receipt": str(output)}, indent=2))
            return
        provider_installation_receipt = _install_pc01_operations_provider(
            args.campaign_dir,
            provider_factory_entrypoint=provider_factory_arg,
            credential_capability_root=credential_root_arg,
            credential_capability_id=credential_id_arg,
            credential_capability_version=credential_version_arg,
            provider_boundary_receipt=boundary_receipt_arg,
        )
    from web_agent.eval.table2.campaign import CampaignRunner, load_entrypoint

    entrypoint = load_entrypoint(args.runner)
    if args.runner_factory:
        entrypoint = entrypoint(campaign_dir=args.campaign_dir)
    elif inspect.isclass(entrypoint):
        entrypoint = entrypoint()
    campaign = CampaignRunner(
        campaign_dir=args.campaign_dir,
        runner=entrypoint,
        runner_entrypoint=args.runner,
        maximum_blocks=args.maximum_blocks,
        live_readiness_probe_target=args.live_readiness_probe_for,
    )
    result = campaign.run_block(args.block_id) if args.block_id else campaign.run()
    if provider_installation_receipt is not None:
        result = {
            **result,
            "pc01_provider_installation_receipt_sha256": (
                provider_installation_receipt.receipt_sha256
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
