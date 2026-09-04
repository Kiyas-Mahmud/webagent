"""Build and preflight a fail-closed Table 2 post-training handoff bundle.

All scientific and operational values come from one explicit JSON input spec.
The command computes hashes, stages immutable payload bytes, and rejects an
incomplete or inconsistent bundle; it never supplies model, audit, task, or
environment values on the operator's behalf.  The repository script is only a
thin command-line entrypoint; all handoff business logic lives in this module.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import yaml

from web_agent.eval.table2.common import (
    SCHEMA_VERSION,
    SchemaError,
    Table2Error,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    require_keys,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.execution_guard import (
    EVALUATION_CLI_SOURCE_RELATIVE_PATH,
    EVALUATION_RUNNER_SCOPE,
    FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH,
    PC01_PAGE_BROKER_SECURITY_FIELD,
    PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD,
    RUNNER_ATTESTATION_SCHEMA_VERSION,
    blocked_pc01_page_broker_security_binding,
    validate_pc01_provider_bootstrap_binding,
    validate_runner_attestation_payload,
)
from web_agent.eval.table2.locked_mount_preflight import (
    LOCKED_MOUNT_ATTESTATION_FIELD,
    LockedMountPreflightError,
    assert_locked_mount_inaccessible,
)
from web_agent.eval.table2.live_deployment import (
    LIVE_DEPLOYMENT_BINDING_FIELD,
    ValidatedPC01LiveDeployment,
    stage_pc01_live_deployment_package,
    validate_evaluator_requirements_resolved_snapshot_binding,
    validate_live_capability_source_plane_disjointness,
)
from web_agent.eval.table2.selection_evidence import stage_selection_evidence
from web_agent.eval.table2.task_interface_audit import (
    TASK_INTERFACE_AUDIT_RELATIVE_PATH,
    build_page_state_compile_authority,
    require_webarena_task_interface_compatible,
    validate_page_state_compile_authority,
    validate_webarena_task_interface_audit,
)
from web_agent.eval.table2.public_task_registry import (
    load_public_development_task_registry,
)
from web_agent.eval.table2.webarena_export import (
    PINNED_TASK_SOURCE_SHA256,
    load_url_map,
    validate_public_pilot_task_export,
)
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_PACKAGES,
    load_service_url_map,
)
from web_agent.eval.table2.webarena_preflight_binding import (
    PREFLIGHT_ARTIFACT_RELATIVE_PATH,
    PREFLIGHT_BINDING_FIELD,
    PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
    build_deployment_preflight_binding,
)
from web_agent.eval.table2.split_deployment_preflight import (
    SINGLE_HOST_TOPOLOGY,
    SPLIT_HOST_TOPOLOGY,
)
from web_agent.eval.table2.resolved_config import (
    ResolvedConfigIdentityError,
    load_resolved_config_identity,
)
from web_agent.eval.table2.package_validator import (
    EVALUATION_CONTROL_SOURCE_RELATIVE_PATHS,
    FROZEN_MEMORY_STORE_FILES,
    MODEL_EVIDENCE_PRODUCER_SCHEMA_VERSION,
    MODEL_EVIDENCE_ROLES,
    MODEL_PAYLOAD_HASH_FIELDS,
    MODEL_PAYLOAD_ROLES,
    JOINT_DUPLICATE_ASSIGNMENT_FILES,
    JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH,
    JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH,
    P4_PREPARATION_EVIDENCE_FILES,
    P4_PREPARATION_EXECUTION_EVIDENCE_FILES,
    P4_SOURCE_AUTHORITY_RELATIVE_PATH,
    PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD,
    PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH,
    _artifact_payload_descriptor,
    _build_task_content_binding_manifest,
    _copy_exact,
    _expected_runner_runtime_identity,
    _load_recovery_scenarios,
    _load_and_verify_frozen_memory_store,
    _load_task_manifest,
    _model_evidence_bundle_value,
    _processor_contract_from_payload,
    _resolve_input,
    _validate_duplicate_audit_bindings,
    _validate_duplicate_audit_manifest,
    _validate_environment_manifest,
    _validate_model_artifact_payloads,
    _validate_model_evidence_bundle,
    _validate_model_memory_source_bindings,
    _validate_registered_joint_duplicate_memory_bindings,
    _validate_pc01_checkpoint_compatibility_readiness,
    _validate_recovery_campaign_counts,
    _validate_registered_protocol,
    _validate_resolved_task_snapshot,
    _validate_seed_manifest_coverage,
    load_yaml,
)
from web_agent.memory.joint_duplicate_audit import (
    AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS,
    AUDIT_TOOL_SOURCE_RELATIVE_PATH,
    JointDuplicateAuditError,
    validate_compact_joint_duplicate_evidence,
)
from web_agent.memory.kaggle_prepare_only import (
    EXECUTED_SOURCE_RELATIVE_PATHS,
    PREPARE_ONLY_CONFIG_RELATIVE,
    locate_prepare_only_execution_receipt,
)


INPUT_SCHEMA_VERSION = "table2-handoff-input-v1"
OUTPUT_SCHEMA_VERSION = "table2-handoff-bundle-v1"
PRODUCTION_RUNNER_ENTRYPOINT = (
    "web_agent.eval.table2.production_runner:create_runner"
)
PRODUCTION_RUNNER_SOURCE = "src/web_agent/eval/table2/production_runner.py"
_TASK_TO_SERVICE_URL_KEYS = {
    "__SHOPPING__": "WA_SHOPPING",
    "__SHOPPING_ADMIN__": "WA_SHOPPING_ADMIN",
    "__REDDIT__": "WA_REDDIT",
    "__GITLAB__": "WA_GITLAB",
    "__MAP__": "WA_MAP",
}


def _validate_handoff_live_capability_source_planes(
    live_deployment: ValidatedPC01LiveDeployment,
) -> None:
    """Reassert runtime/sealed source separation at the handoff boundary."""

    try:
        validate_live_capability_source_plane_disjointness(
            live_deployment.manifest
        )
    except SchemaError as exc:
        raise SchemaError(
            "handoff live-deployment capability source planes are not disjoint: "
            f"{exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _spec_path(spec_path: Path, value: object, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise SchemaError(f"handoff input requires {field}")
    path = Path(text)
    return (path if path.is_absolute() else spec_path.parent / path).absolute()


def _spec_identity(
    spec_path: Path, value: object, *, field: str
) -> dict[str, Any] | None:
    """Load an expected split-host identity from an object or JSON path."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    source = _spec_path(spec_path, value, field=field)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise SchemaError(f"{field} must not be a symlink")
    return read_json(source)


def _repo_source(repo: Path, value: object, *, field: str) -> tuple[str, Path]:
    text = str(value or "").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise SchemaError(f"{field} must be a safe repository-relative path")
    source = (repo / path).resolve()
    if repo not in source.parents or not source.is_file():
        raise SchemaError(f"{field} is missing from the repository checkout")
    return path.as_posix(), source


def _git_commit_clean(repo: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Table2Error("handoff preparation requires a Git checkout") from exc
    if not commit or status:
        raise Table2Error(
            "handoff preparation requires a clean Git checkout; commit every runner, "
            "evaluator, protocol, and configuration change first"
        )
    return commit


def _copy_payload(source: Path, destination_root: Path) -> Path:
    # Descriptor construction rejects absent, empty, or symlinked payloads.
    _artifact_payload_descriptor(source)
    if source.is_file():
        destination = destination_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    if destination_root.exists():
        raise Table2Error(f"refusing to overwrite staged payload: {destination_root}")
    shutil.copytree(source, destination_root, symlinks=False)
    return destination_root


def _canonicalize_processor_contract(source: Path, destination_root: Path) -> Path:
    _artifact_payload_descriptor(source)
    value = read_json(source)
    required = (
        "processor_class",
        "processor_revision",
        "processor_config_sha256",
        "pre_action_field_mapping",
        "post_action_field_mapping",
    )
    require_keys(value, required, context=f"processor contract {source}")
    try:
        from web_agent.runtime.observation import ProcessorParityContract

        contract = ProcessorParityContract(**{key: value[key] for key in required})
    except (ImportError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"processor contract cannot reconstruct ProcessorParityContract: {source}"
        ) from exc
    destination = destination_root / "processor_contract.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(contract.to_dict()))
    _processor_contract_from_payload(destination)
    return destination


def _computed_field(target: dict[str, Any], key: str, value: Any, *, context: str) -> None:
    if key in target and target[key] != value:
        raise SchemaError(f"{context}.{key} differs from explicit artifact bytes")
    target[key] = value


def _build_environment(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    repo: Path,
    campaign: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_path: Path,
    webarena_host_preflight_source: Path,
    webarena_service_url_map_source: Path,
    deployment_topology: str,
    expected_dgx_model_runtime_identity: Mapping[str, Any] | None,
    expected_bridge_identity: Mapping[str, Any] | None,
    live_deployment_manifest_source: Path,
    live_deployment_evidence_root: Path,
    expected_live_reset_task_index: int,
) -> tuple[Path, Path, ValidatedPC01LiveDeployment]:
    raw = spec.get("environment")
    evaluator_spec = spec.get("evaluator")
    if not isinstance(raw, Mapping) or not isinstance(evaluator_spec, Mapping):
        raise SchemaError("handoff input requires environment and evaluator mappings")
    environment = dict(raw)
    _computed_field(
        environment,
        "schema_version",
        "table2-environment-v2",
        context="environment",
    )
    dependency_lock = _spec_path(
        spec_path, spec.get("dependency_lock"), field="dependency_lock"
    )
    if not dependency_lock.is_file():
        raise FileNotFoundError(dependency_lock)
    if dependency_lock.is_symlink():
        raise SchemaError("dependency_lock must not be a symlink")
    frozen_lock = output_path.parent / FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH
    if frozen_lock.exists():
        raise Table2Error(f"refusing to overwrite staged dependency lock: {frozen_lock}")
    frozen_lock.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dependency_lock, frozen_lock)
    dependency_lock_sha256 = sha256_file(dependency_lock)
    if sha256_file(frozen_lock) != dependency_lock_sha256:
        raise SchemaError("staged dependency lock differs from supplied bytes")
    _computed_field(
        environment,
        "dependency_lock_sha256",
        dependency_lock_sha256,
        context="environment",
    )
    _computed_field(
        environment,
        "dependency_lock_relative_path",
        FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH,
        context="environment",
    )
    try:
        locked_mount_attestation = assert_locked_mount_inaccessible(repo, protocol)
    except LockedMountPreflightError as exc:
        raise SchemaError(f"handoff locked-mount preflight failed: {exc}") from exc
    _computed_field(
        environment,
        LOCKED_MOUNT_ATTESTATION_FIELD,
        locked_mount_attestation,
        context="environment",
    )

    evaluator_relative, evaluator_source = _repo_source(
        repo,
        evaluator_spec.get("source_relative_path"),
        field="evaluator.source_relative_path",
    )
    oracle_rules = _resolve_input(repo, campaign["recovery_oracle_rules"])
    evaluator = dict(evaluator_spec)
    evaluator_entrypoint = str(evaluator.get("entrypoint") or "").strip()
    if ":" not in evaluator_entrypoint:
        raise SchemaError(
            "evaluator.entrypoint must name an independent module:factory"
        )
    evaluator_module = evaluator_entrypoint.split(":", 1)[0]
    module_stem = evaluator_module.replace(".", "/")
    if evaluator_relative not in {
        f"{module_stem}.py",
        f"{module_stem}/__init__.py",
    }:
        raise SchemaError(
            "evaluator entrypoint module differs from its attested source"
        )
    _computed_field(
        evaluator, "source_relative_path", evaluator_relative, context="evaluator"
    )
    _computed_field(
        evaluator, "source_sha256", sha256_file(evaluator_source), context="evaluator"
    )
    _computed_field(
        evaluator,
        "oracle_rules_sha256",
        sha256_file(oracle_rules),
        context="evaluator",
    )
    model_based = evaluator.get("model_based")
    if type(model_based) is not bool:
        raise SchemaError("evaluator.model_based must be an explicit boolean")
    prompt_value = evaluator.pop("prompt_path", None)
    if model_based:
        prompt = _spec_path(spec_path, prompt_value, field="evaluator.prompt_path")
        if not prompt.is_file():
            raise FileNotFoundError(prompt)
        _computed_field(
            evaluator, "prompt_sha256", sha256_file(prompt), context="evaluator"
        )
    elif prompt_value is not None:
        raise SchemaError("non-model evaluator must not provide prompt_path")
    else:
        _computed_field(evaluator, "prompt_sha256", None, context="evaluator")
    environment["evaluator"] = evaluator
    source_binding = build_deployment_preflight_binding(
        evidence_path=webarena_host_preflight_source,
        service_url_map_path=webarena_service_url_map_source,
        deployment_topology=deployment_topology,
        expected_live_reset_task_index=expected_live_reset_task_index,
        expected_dgx_model_runtime_identity=expected_dgx_model_runtime_identity,
        expected_bridge_identity=expected_bridge_identity,
    )
    staged_preflight = output_path.parent / PREFLIGHT_ARTIFACT_RELATIVE_PATH
    staged_url_map = output_path.parent / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH
    for source, destination, label in (
        (
            webarena_host_preflight_source,
            staged_preflight,
            "WebArena host preflight",
        ),
        (
            webarena_service_url_map_source,
            staged_url_map,
            "WebArena deployment service URL map",
        ),
    ):
        if destination.exists():
            raise Table2Error(f"refusing to overwrite staged {label}: {destination}")
        shutil.copy2(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise SchemaError(f"staged {label} differs from supplied bytes")
    staged_binding = build_deployment_preflight_binding(
        evidence_path=staged_preflight,
        service_url_map_path=staged_url_map,
        deployment_topology=deployment_topology,
        expected_live_reset_task_index=expected_live_reset_task_index,
        expected_dgx_model_runtime_identity=expected_dgx_model_runtime_identity,
        expected_bridge_identity=expected_bridge_identity,
    )
    if staged_binding != source_binding:
        raise SchemaError("staged WebArena deployment preflight binding changed")
    _computed_field(
        environment,
        PREFLIGHT_BINDING_FIELD,
        staged_binding,
        context="environment",
    )
    live_deployment = stage_pc01_live_deployment_package(
        manifest_path=live_deployment_manifest_source,
        evidence_root=live_deployment_evidence_root,
        repository_root=repo,
        destination_artifact_root=output_path.parent,
    )
    # Reassert the plane boundary at the handoff trust transition.  The staged
    # package validator already enforces it; this explicit handoff check keeps
    # later refactors from silently turning that guarantee into an assumption.
    _validate_handoff_live_capability_source_planes(live_deployment)
    _computed_field(
        environment,
        LIVE_DEPLOYMENT_BINDING_FIELD,
        live_deployment.binding,
        context="environment",
    )
    preflight_evidence = read_json(staged_preflight)
    if deployment_topology == SINGLE_HOST_TOPOLOGY:
        browser_evidence = preflight_evidence.get("browser_check")
    elif deployment_topology == SPLIT_HOST_TOPOLOGY:
        local_preflight = preflight_evidence.get("local_browser_preflight")
        browser_evidence = (
            local_preflight.get("browser_check")
            if isinstance(local_preflight, Mapping)
            else None
        )
    else:  # Semantic validation above already rejects this branch.
        browser_evidence = None
    if not isinstance(browser_evidence, Mapping):
        raise SchemaError("WebArena host preflight lacks browser evidence")
    if (
        str(environment.get("browser", "")).strip().casefold() != "chromium"
        or environment.get("browser_version")
        != browser_evidence.get("browser_version")
        or environment.get("playwright_version")
        != PINNED_WEBARENA_PACKAGES["playwright"]
    ):
        raise SchemaError(
            "environment browser identity differs from measured WebArena preflight"
        )
    _validate_environment_manifest(environment, protocol=protocol)
    atomic_write_json(output_path, environment)
    return output_path, evaluator_source, live_deployment


def _build_resolved_tasks(
    *,
    export_path: Path,
    upstream_task_source_path: Path,
    site_url_map_path: Path,
    authorized_raw_task_source_sha256: str | None,
    task_interface_audit_path: Path,
    output_path: Path,
    registry_path: Path,
    environment_path: Path,
    duplicate_output_path: Path,
    evaluator_requirements: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry_metadata, registry_rows = _load_task_manifest(registry_path)
    submitted_export = read_json(export_path)
    # Use the independently rebuilt value from this point onward.  Exact
    # equality validation prevents a hand-written export from becoming task
    # authority even if it copies the registered source hash into metadata.
    export = validate_public_pilot_task_export(
        submitted_export,
        source=upstream_task_source_path,
        registry_path=registry_path,
        site_url_map=load_url_map(site_url_map_path),
        expected_source_sha256=PINNED_TASK_SOURCE_SHA256,
        authorized_raw_json_sha256=authorized_raw_task_source_sha256,
    )
    if not task_interface_audit_path.is_file():
        raise FileNotFoundError(task_interface_audit_path)
    if task_interface_audit_path.is_symlink():
        raise SchemaError("WebArena task-interface audit must not be a symlink")
    submitted_task_interface_audit = read_json(task_interface_audit_path)
    task_interface_audit = validate_webarena_task_interface_audit(
        submitted_task_interface_audit,
        task_export=export,
    )
    # This is intentionally checked before any task filtering or runtime
    # projection can occur.  A FAIL audit is evidence, not permission to alter
    # the official evaluator or silently keep only compatible tasks.
    require_webarena_task_interface_compatible(task_interface_audit)
    staged_task_interface_audit = output_path.parent / (
        TASK_INTERFACE_AUDIT_RELATIVE_PATH
    )
    if staged_task_interface_audit.exists():
        raise Table2Error(
            "refusing to overwrite staged WebArena task-interface audit: "
            f"{staged_task_interface_audit}"
        )
    shutil.copy2(task_interface_audit_path, staged_task_interface_audit)
    if sha256_file(staged_task_interface_audit) != sha256_file(
        task_interface_audit_path
    ):
        raise SchemaError("staged WebArena task-interface audit bytes changed")
    validate_webarena_task_interface_audit(
        read_json(staged_task_interface_audit),
        task_export=export,
    )
    export_rows = export.get("tasks")
    if not isinstance(export_rows, list) or not all(
        isinstance(row, Mapping) for row in export_rows
    ):
        raise SchemaError("resolved_task_export must contain a tasks array")
    if len(export_rows) != 50:
        raise SchemaError("resolved_task_export must contain exactly 50 task rows")
    environment = read_json(environment_path)
    if (
        export.get("benchmark") != environment.get("benchmark")
        or export.get("benchmark_version") != environment.get("benchmark_version")
    ):
        raise SchemaError("task export benchmark identity differs from environment")
    if export.get("task_definition_version") != environment.get(
        "task_definition_version"
    ):
        raise SchemaError(
            "task export definition identity differs from environment"
        )
    evaluator = environment["evaluator"]
    evaluator_identity = {
        "evaluator_id": evaluator["evaluator_id"],
        "evaluator_version": evaluator["evaluator_version"],
        "source_relative_path": evaluator["source_relative_path"],
        "source_sha256": evaluator["source_sha256"],
        "oracle_rules_sha256": evaluator["oracle_rules_sha256"],
    }
    expected_registry_identity = [
        (
            str(row.get("task_id") or "").strip(),
            row.get("upstream_index"),
            str(
                row.get("benchmark_task_id", row.get("upstream_index"))
            ).strip(),
        )
        for row in registry_rows
    ]
    observed_export_identity = [
        (
            str(row.get("task_id") or "").strip(),
            row.get("upstream_index"),
            str(row.get("benchmark_task_id") or "").strip(),
        )
        for row in export_rows
    ]
    if observed_export_identity != expected_registry_identity:
        raise SchemaError(
            "task export identities or order differ from tracked registry"
        )

    by_index: dict[int, Mapping[str, Any]] = {}
    for row in export_rows:
        index = row.get("upstream_index")
        if type(index) is not int or index in by_index:
            raise SchemaError("task export upstream indices must be unique integers")
        by_index[index] = row
    tasks: list[dict[str, Any]] = []
    content_fields = (
        "task_id",
        "upstream_index",
        "benchmark_task_id",
        "benchmark_task_version",
        "instruction",
        "start_state",
        "task_config",
        "evaluator",
    )
    for registry_row in registry_rows:
        index = int(registry_row["upstream_index"])
        source = dict(by_index[index])
        require_keys(
            source,
            (
                "upstream_index",
                "benchmark_task_id",
                "benchmark_task_version",
                "instruction",
                "start_state",
                "task_config",
                "evaluator",
            ),
            context=f"resolved_task_export[{index}]",
        )
        task_id = str(registry_row["task_id"])
        if "task_id" in source and source["task_id"] != task_id:
            raise SchemaError(f"task export ID differs from registry at upstream index {index}")
        task_evaluator = source["evaluator"]
        if not isinstance(task_evaluator, Mapping):
            raise SchemaError(f"task export evaluator must be a mapping at index {index}")
        for key in ("evaluator_id", "evaluator_version"):
            if task_evaluator.get(key) != evaluator[key]:
                raise SchemaError(f"task evaluator {key} differs at upstream index {index}")
        try:
            from web_agent.runtime.contracts import RuntimeStartState

            RuntimeStartState.from_webarena_mapping(source["start_state"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"task start state is incomplete or unsafe at upstream index {index}"
            ) from exc
        row = {
            "task_id": task_id,
            "upstream_index": index,
            "benchmark_task_id": str(source["benchmark_task_id"]),
            "benchmark_task_version": str(source["benchmark_task_version"]),
            "instruction": source["instruction"],
            "start_state": source["start_state"],
            "task_config": source["task_config"],
            "evaluator": dict(task_evaluator),
        }
        row["source_content_sha256"] = sha256_json(
            {key: row[key] for key in content_fields}
        )
        tasks.append(row)

    page_state_compile_report = build_page_state_compile_authority(tasks)
    validate_page_state_compile_authority(
        page_state_compile_report,
        task_rows=tasks,
    )
    if (
        page_state_compile_report.get("evaluator_id")
        != evaluator_identity["evaluator_id"]
        or page_state_compile_report.get("evaluator_version")
        != evaluator_identity["evaluator_version"]
    ):
        raise SchemaError(
            "page-state compile authority evaluator identity differs from "
            "the frozen environment"
        )

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "ResolvedWebArenaTaskSnapshot",
        "snapshot_id": str(export.get("snapshot_id") or "").strip(),
        "benchmark": environment["benchmark"],
        "benchmark_version": environment["benchmark_version"],
        "task_definition_version": export["task_definition_version"],
        "browsergym_webarena_version": export["browsergym_webarena_version"],
        "libwebarena_version": export["libwebarena_version"],
        "upstream_export_schema_version": export["schema_version"],
        "upstream_export_record_type": export["record_type"],
        "upstream_export_file_sha256": sha256_file(export_path),
        "upstream_export_content_sha256": sha256_json(export),
        "upstream_task_source": dict(export["source"]),
        "site_url_map_sha256": export["site_url_map_sha256"],
        "resolved_task_set_sha256": export["resolved_task_set_sha256"],
        "task_action_interface_audit": task_interface_audit,
        "task_action_interface_audit_file_sha256": sha256_file(
            staged_task_interface_audit
        ),
        "task_action_interface_audit_content_sha256": sha256_json(
            task_interface_audit
        ),
        "page_state_evaluator_compile_report": page_state_compile_report,
        "page_state_evaluator_compile_report_content_sha256": sha256_json(
            page_state_compile_report
        ),
        "selection_rule": export["selection_rule"],
        "registry_manifest_id": export["registry_manifest_id"],
        "upstream_registry_manifest_sha256": export[
            "registry_manifest_sha256"
        ],
        "registry_manifest_sha256": sha256_file(registry_path),
        "environment_manifest_sha256": sha256_file(environment_path),
        "evaluator_identity_sha256": sha256_json(evaluator_identity),
        "partition": registry_metadata.get("partition"),
        "locked_test_content": registry_metadata.get("locked_test_content"),
        "final_paper_evaluation_eligible": registry_metadata.get(
            "final_paper_evaluation_eligible"
        ),
        "required_task_count": len(registry_rows),
        "duplicate_audit_manifest": str(duplicate_output_path),
        "tasks": tasks,
    }
    if not snapshot["snapshot_id"]:
        raise SchemaError("resolved_task_export requires an explicit snapshot_id")
    validate_evaluator_requirements_resolved_snapshot_binding(
        evaluator_requirements,
        task_export=export,
        resolved_task_snapshot=snapshot,
    )
    atomic_write_json(output_path, snapshot)
    return _validate_resolved_task_snapshot(
        output_path,
        registry_path=registry_path,
        registry_metadata=registry_metadata,
        registry_tasks=registry_rows,
        environment_path=environment_path,
    )


def _stage_models(
    *,
    model_specs: object,
    spec_path: Path,
    output_root: Path,
    protocol_path: Path,
    matched_seeds: list[int],
) -> dict[int, Path]:
    if not isinstance(model_specs, list) or not all(
        isinstance(row, Mapping) for row in model_specs
    ):
        raise SchemaError("handoff input models must be an array of mappings")
    model_paths: list[Path] = []
    prompt_hash = sha256_file(protocol_path.parent / "prompts" / "e0_action_v1.txt")
    for raw in model_specs:
        row = dict(raw)
        required = (
            "model_seed",
            "selected_model_id",
            "selected_epoch",
            "e0_backbone_id",
            "e0_backbone_revision",
            "e0_parser_id",
            "e0_parser_version",
            "e0_parser_module",
            "e0_parser_attribute",
            "selection_scope",
            "checkpoint_selection",
            "validation_rows_read",
            "test_rows_read",
            "locked_test_rows_read",
            "artifact_paths",
            "model_evidence_paths",
        )
        require_keys(row, required, context="model handoff input")
        seed = row["model_seed"]
        if type(seed) is not int or seed not in matched_seeds:
            raise SchemaError("model handoff seed is not a registered matched seed")
        artifact_paths = row["artifact_paths"]
        if not isinstance(artifact_paths, Mapping) or set(artifact_paths) != set(
            MODEL_PAYLOAD_ROLES
        ):
            raise SchemaError("model artifact_paths must contain every registered payload role")
        evidence_paths = row["model_evidence_paths"]
        if not isinstance(evidence_paths, Mapping) or set(evidence_paths) != set(
            MODEL_EVIDENCE_ROLES
        ):
            raise SchemaError(
                "model model_evidence_paths must contain every registered evidence role"
            )
        model_dir = output_root / "models"
        model_path = model_dir / f"seed_{seed}.json"
        descriptors: dict[str, dict[str, Any]] = {}
        model_hashes: dict[str, str] = {}
        for role in MODEL_PAYLOAD_ROLES:
            source = _spec_path(
                spec_path, artifact_paths[role], field=f"model[{seed}].artifact_paths.{role}"
            )
            role_root = output_root / "payloads" / f"seed_{seed}" / role
            staged = (
                _canonicalize_processor_contract(source, role_root)
                if role in {"processor_contract", "e0_processor_contract"}
                else _copy_payload(source, role_root)
            )
            stored_path = os.path.relpath(staged, model_dir)
            descriptor = _artifact_payload_descriptor(staged, stored_path=stored_path)
            if role == "e0_parser":
                descriptor.update(
                    {
                        "module": row["e0_parser_module"],
                        "attribute": row["e0_parser_attribute"],
                    }
                )
            descriptors[role] = descriptor
            model_hashes[MODEL_PAYLOAD_HASH_FIELDS[role]] = descriptor["sha256"]
        for role in ("processor_contract", "e0_processor_contract"):
            source = (model_dir / descriptors[role]["path"]).resolve()
            _processor_contract_from_payload(source)
        evidence_descriptors: dict[str, dict[str, Any]] = {}
        for role in MODEL_EVIDENCE_ROLES:
            source = _spec_path(
                spec_path,
                evidence_paths[role],
                field=f"model[{seed}].model_evidence_paths.{role}",
            )
            role_root = output_root / "model_evidence" / f"seed_{seed}" / role
            staged = _copy_payload(source, role_root)
            stored_path = os.path.relpath(staged, model_dir)
            evidence_descriptors[role] = _artifact_payload_descriptor(
                staged,
                stored_path=stored_path,
            )
        evidence_bundle = _model_evidence_bundle_value(
            evidence_descriptors,
            producer_schema_version=MODEL_EVIDENCE_PRODUCER_SCHEMA_VERSION,
        )
        resolved_config_path = (
            model_dir / descriptors["resolved_config"]["path"]
        ).resolve()
        try:
            resolved_config_identity = load_resolved_config_identity(
                resolved_config_path
            )
        except ResolvedConfigIdentityError as exc:
            raise SchemaError(
                f"selected resolved configuration is unreadable for seed {seed}"
            ) from exc
        if (
            resolved_config_identity.payload_sha256
            != model_hashes["resolved_config_sha256"]
        ):
            raise SchemaError(
                "selected resolved-config payload identity changed while staging"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **{
                key: row[key]
                for key in required
                if key not in {"artifact_paths", "model_evidence_paths"}
            },
            **model_hashes,
            "resolved_config_record_sha256": (
                resolved_config_identity.record_sha256
            ),
            "e0_base_prompt_sha256": prompt_hash,
            "artifact_payloads": descriptors,
            "model_evidence_bundle_sha256": evidence_bundle["bundle_sha256"],
            "model_evidence_bundle": evidence_bundle,
        }
        atomic_write_json(model_path, manifest)
        executable = _validate_model_artifact_payloads(model_path, manifest)
        _validate_model_evidence_bundle(
            model_path,
            manifest,
            executable_payloads=executable,
        )
        model_paths.append(model_path)
    return _validate_seed_manifest_coverage(
        model_paths, matched_seeds, kind="model", required=True
    )


def _bind_duplicate_audit(
    *,
    input_path: Path,
    output_path: Path,
    task_rows: list[dict[str, Any]],
    task_snapshot_path: Path,
    recovery_path: Path,
    memory_by_seed: Mapping[int, Path],
    repository_root: Path,
    assignment_package_root: Path,
    preparation_package_root: Path,
    resolved_task_export_path: Path,
    approved_task_registry_path: Path,
    registered_recovery_scenarios_path: Path,
    duplicate_audit_registration_path: Path,
    provenance_manifest_path: Path,
) -> Path:
    payload = read_json(input_path)
    if (
        payload.get("manifest_state") != "FROZEN_REGISTRATION"
        or payload.get("normal_task_evidence_status") != "VERIFIED"
        or payload.get("normal_task_runtime_policy")
        != "VERIFIED_NONEMPTY_CLUSTERS_REQUIRED"
    ):
        raise SchemaError("external duplicate audit is not explicitly frozen and VERIFIED")
    bindings = _build_task_content_binding_manifest(
        task_rows,
        task_manifest_sha256=sha256_file(task_snapshot_path),
        memory_by_seed=memory_by_seed,
        evidence_scope=EVALUATION_RUNNER_SCOPE,
    )
    corpora = {
        value["corpus_binding_sha256"]
        for value in bindings["train_corpus_by_seed"].values()
    }
    if len(corpora) != 1:
        raise SchemaError("one external duplicate audit requires one matched train corpus")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SchemaError("external duplicate audit entries must be an array")
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise SchemaError("external duplicate audit entry must be a mapping")
        if raw.get("task_partition") == "normal":
            task_id = str(raw.get("task_id", ""))
            if raw.get("status") != "VERIFIED":
                raise SchemaError(f"external duplicate audit is not VERIFIED for {task_id}")
            clusters = raw.get("cluster_ids")
            if not isinstance(clusters, list) or not clusters:
                raise SchemaError(f"external duplicate audit has no clusters for {task_id}")
            if not str(raw.get("audit_tool_id", "")).strip() or not str(
                raw.get("audit_tool_version", "")
            ).strip():
                raise SchemaError(f"external duplicate audit tool identity is absent for {task_id}")
            evidence = raw.get("evidence_sha256")
            if not isinstance(evidence, str) or len(evidence) != 64:
                raise SchemaError(f"external duplicate audit evidence hash is absent for {task_id}")
    recovery_rows, _ = _load_recovery_scenarios(recovery_path)
    _validate_duplicate_audit_manifest(
        input_path,
        normal_tasks=task_rows,
        recovery_scenarios=recovery_rows,
        require_verified_normal=True,
    )
    _validate_duplicate_audit_bindings(
        input_path,
        task_content_manifest=bindings,
        require_verified_normal=True,
    )
    _validate_registered_joint_duplicate_memory_bindings(
        memory_by_seed,
        duplicate_audit_path=input_path,
        required=True,
        repository_root=repository_root,
        assignment_package_root=assignment_package_root,
        preparation_package_root=preparation_package_root,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=approved_task_registry_path,
        registered_recovery_scenarios_path=(
            registered_recovery_scenarios_path
        ),
        duplicate_audit_registration_path=duplicate_audit_registration_path,
        provenance_manifest_path=provenance_manifest_path,
    )
    # Copy only after every supplied row, its canonical audit record, and the
    # joint memory/task namespace have passed fail-closed validation.  Handoff
    # must never fill in hashes that the external audit did not itself bind.
    atomic_write_json(output_path, payload)
    return output_path


def prepare_handoff(
    *, spec_path: str | Path, output_dir: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    spec_path = Path(spec_path).resolve()
    repo = Path(repository_root).resolve()
    output = Path(output_dir).resolve()
    if output == repo or repo in output.parents:
        raise Table2Error("handoff output must be outside the source checkout")
    if output.exists():
        raise Table2Error(f"handoff output already exists and will not be overwritten: {output}")
    spec = read_json(spec_path)
    if spec.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise SchemaError("handoff input schema version is not registered")
    commit = _git_commit_clean(repo)

    campaign_source = _spec_path(
        spec_path, spec.get("campaign_config"), field="campaign_config"
    )
    campaign = load_yaml(campaign_source)
    if str(campaign.get("campaign_mode", "evaluation")) != "evaluation":
        raise SchemaError("handoff preparer supports evaluation campaigns only")
    protocol_path = _resolve_input(repo, campaign["protocol"])
    protocol = load_yaml(protocol_path)
    _validate_registered_protocol(protocol)
    selection_config = protocol.get("selection")
    if not isinstance(selection_config, Mapping):
        raise SchemaError("frozen protocol lacks selection configuration")
    selection_mode = str(selection_config.get("mode") or "")
    pilot_only = (
        str(campaign.get("campaign_kind")) != "locked_final"
        or str(campaign.get("evidence_label")) == "PILOT_ONLY"
    )
    requires_pc01_checkpoint_compatibility = pilot_only
    registry_path = _resolve_input(repo, campaign["task_manifest"])
    active_task_registry = (
        load_public_development_task_registry(registry_path)
        if pilot_only
        else None
    )
    expected_live_reset_task_index = (
        active_task_registry.ordered_upstream_indices[0]
        if active_task_registry is not None
        else 0
    )
    recovery_source = _resolve_input(repo, campaign["recovery_scenarios"])
    matched_seeds = [int(seed) for seed in campaign.get("matched_seeds", [])]
    if matched_seeds != [42]:
        raise SchemaError("Table 2 handoff matched_seeds must be exactly [42]")

    upstream_task_source_path = _spec_path(
        spec_path,
        spec.get("webarena_task_source"),
        field="webarena_task_source",
    )
    site_url_map_path = _spec_path(
        spec_path,
        spec.get("webarena_site_url_map"),
        field="webarena_site_url_map",
    )
    raw_source_authority_value = spec.get(
        "authorized_raw_webarena_task_source_sha256"
    )
    authorized_raw_task_source_sha256 = (
        None
        if raw_source_authority_value is None
        else str(raw_source_authority_value).strip()
    )
    if raw_source_authority_value is not None and not (
        authorized_raw_task_source_sha256
    ):
        raise SchemaError(
            "authorized_raw_webarena_task_source_sha256 must be a SHA-256"
        )
    # The five placeholder resolutions and seven deployed BrowserGym services
    # are distinct authorities.  Shared origins must still resolve identically.
    task_urls = load_url_map(site_url_map_path)
    service_url_map_path = _spec_path(
        spec_path,
        spec.get("webarena_service_url_map"),
        field="webarena_service_url_map",
    )
    service_urls = load_service_url_map(service_url_map_path)
    for task_key, service_key in _TASK_TO_SERVICE_URL_KEYS.items():
        if task_urls[task_key] != service_urls[service_key]:
            raise SchemaError(
                f"WebArena task and deployment maps disagree for {task_key}"
            )
    host_preflight_path = _spec_path(
        spec_path,
        spec.get("webarena_host_preflight"),
        field="webarena_host_preflight",
    )
    deployment_topology = str(
        spec.get("webarena_deployment_topology") or ""
    ).strip()
    if deployment_topology not in {SINGLE_HOST_TOPOLOGY, SPLIT_HOST_TOPOLOGY}:
        raise SchemaError(
            "handoff input requires one registered webarena_deployment_topology"
        )
    expected_dgx_identity = _spec_identity(
        spec_path,
        spec.get("expected_dgx_model_runtime_identity"),
        field="expected_dgx_model_runtime_identity",
    )
    expected_bridge_identity = _spec_identity(
        spec_path,
        spec.get("expected_bridge_identity"),
        field="expected_bridge_identity",
    )
    build_deployment_preflight_binding(
        evidence_path=host_preflight_path,
        service_url_map_path=service_url_map_path,
        deployment_topology=deployment_topology,
        expected_live_reset_task_index=expected_live_reset_task_index,
        expected_dgx_model_runtime_identity=expected_dgx_identity,
        expected_bridge_identity=expected_bridge_identity,
    )
    live_deployment_manifest_source = _spec_path(
        spec_path,
        spec.get("pc01_live_deployment_manifest"),
        field="pc01_live_deployment_manifest",
    )
    live_deployment_evidence_root = _spec_path(
        spec_path,
        spec.get("pc01_live_deployment_evidence_root"),
        field="pc01_live_deployment_evidence_root",
    )
    for source, field in (
        (live_deployment_manifest_source, "pc01_live_deployment_manifest"),
        (live_deployment_evidence_root, "pc01_live_deployment_evidence_root"),
    ):
        resolved = source.resolve()
        if resolved == repo or repo in resolved.parents:
            raise SchemaError(
                f"{field} must be supplied from measured evidence outside the source checkout"
            )
    output.mkdir(parents=True)
    environment_path, evaluator_source, live_deployment = _build_environment(
        spec=spec,
        spec_path=spec_path,
        repo=repo,
        campaign=campaign,
        protocol=protocol,
        output_path=output / "environment.json",
        webarena_host_preflight_source=host_preflight_path,
        webarena_service_url_map_source=service_url_map_path,
        deployment_topology=deployment_topology,
        expected_dgx_model_runtime_identity=expected_dgx_identity,
        expected_bridge_identity=expected_bridge_identity,
        live_deployment_manifest_source=live_deployment_manifest_source,
        live_deployment_evidence_root=live_deployment_evidence_root,
        expected_live_reset_task_index=expected_live_reset_task_index,
    )
    duplicate_path = output / "duplicate_audit.json"
    export_path = _spec_path(
        spec_path, spec.get("resolved_task_export"), field="resolved_task_export"
    )
    task_interface_audit_path = _spec_path(
        spec_path,
        spec.get("webarena_task_interface_audit"),
        field="webarena_task_interface_audit",
    )
    _, task_rows = _build_resolved_tasks(
        export_path=export_path,
        upstream_task_source_path=upstream_task_source_path,
        site_url_map_path=site_url_map_path,
        authorized_raw_task_source_sha256=authorized_raw_task_source_sha256,
        task_interface_audit_path=task_interface_audit_path,
        output_path=output / "resolved_tasks.json",
        registry_path=registry_path,
        environment_path=environment_path,
        duplicate_output_path=duplicate_path,
        evaluator_requirements=live_deployment.evaluator_requirements,
    )
    task_snapshot_path = output / "resolved_tasks.json"
    duplicate_input_source = _spec_path(
        spec_path, spec.get("duplicate_audit"), field="duplicate_audit"
    )

    joint_assignment_source = _spec_path(
        spec_path,
        spec.get("joint_duplicate_assignment_package"),
        field="joint_duplicate_assignment_package",
    )
    joint_preparation_source = _spec_path(
        spec_path,
        spec.get("p4_preparation_package"),
        field="p4_preparation_package",
    )
    joint_provenance_source = _spec_path(
        spec_path,
        spec.get("joint_duplicate_provenance_manifest"),
        field="joint_duplicate_provenance_manifest",
    )
    duplicate_registration_source = (
        repo / JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH
    ).resolve()
    try:
        validate_compact_joint_duplicate_evidence(
            package_root=joint_assignment_source,
            config_path=repo / JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH,
            source_authority_path=repo / P4_SOURCE_AUTHORITY_RELATIVE_PATH,
            preparation_root=joint_preparation_source,
            resolved_task_export_path=export_path,
            approved_task_registry_path=registry_path,
            recovery_scenarios_path=recovery_source,
            duplicate_audit_registration_path=duplicate_registration_source,
            provenance_manifest_path=joint_provenance_source,
            final_audit_path=duplicate_input_source,
        )
    except (JointDuplicateAuditError, OSError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"handoff compact joint duplicate evidence is invalid: {exc}"
        ) from exc
    try:
        preparation_receipt_source, preparation_receipt_sidecar_source = (
            locate_prepare_only_execution_receipt(joint_preparation_source)
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"handoff P4 preparation execution receipt is invalid: {exc}"
        ) from exc
    joint_evidence_root = output / "joint_duplicate_evidence"
    for name in JOINT_DUPLICATE_ASSIGNMENT_FILES:
        _copy_exact(
            joint_assignment_source / name,
            joint_evidence_root / "assignment" / name,
        )
    for name in P4_PREPARATION_EVIDENCE_FILES:
        _copy_exact(
            joint_preparation_source / name,
            joint_evidence_root / "preparation" / name,
        )
    for source, name in zip(
        (preparation_receipt_source, preparation_receipt_sidecar_source),
        P4_PREPARATION_EXECUTION_EVIDENCE_FILES,
        strict=True,
    ):
        _copy_exact(source, joint_evidence_root / name)
    _copy_exact(
        export_path,
        joint_evidence_root / "resolved_task_export.json",
    )
    _copy_exact(
        recovery_source,
        joint_evidence_root / "registered_recovery_scenarios.json",
    )
    _copy_exact(
        joint_provenance_source,
        joint_evidence_root / "provenance_manifest.json",
    )
    tracked_evidence_root = joint_evidence_root / "tracked_repository"
    for relative in dict.fromkeys(
        (
            JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH,
            P4_SOURCE_AUTHORITY_RELATIVE_PATH,
            Path(AUDIT_TOOL_SOURCE_RELATIVE_PATH),
            *(Path(value) for value in AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS),
            *(Path(value) for value in EXECUTED_SOURCE_RELATIVE_PATHS),
            Path(PREPARE_ONLY_CONFIG_RELATIVE),
            JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH,
        )
    ):
        _copy_exact(repo / relative, tracked_evidence_root / relative)

    model_by_seed = _stage_models(
        model_specs=spec.get("models"),
        spec_path=spec_path,
        output_root=output,
        protocol_path=protocol_path,
        matched_seeds=matched_seeds,
    )
    if 42 not in model_by_seed:
        raise SchemaError(
            "Table 2 handoff requires seed 42 selection-winner evidence"
        )
    selection_evidence_path = stage_selection_evidence(
        selection_spec=spec.get("selection_evidence"),
        spec_path=spec_path,
        output_dir=output / "selection_evidence",
        repository_root=repo,
        selected_model_manifest=read_json(model_by_seed[42]),
        selection_mode=selection_mode,
    )
    from .selection_evidence import validate_selection_evidence

    validate_selection_evidence(
        selection_evidence_path,
        repository_root=repo,
        selected_model_manifests={
            seed: read_json(path) for seed, path in sorted(model_by_seed.items())
        },
        expected_model_seeds=matched_seeds,
        expected_selection_mode=selection_mode,
    )
    checkpoint_compatibility_binding: dict[str, Any] | None = None
    checkpoint_compatibility_source_files: tuple[Path, ...] = ()
    checkpoint_compatibility_path: Path | None = None
    supplied_checkpoint_compatibility = spec.get(
        "pc01_checkpoint_compatibility_receipt"
    )
    if requires_pc01_checkpoint_compatibility:
        checkpoint_compatibility_source = _spec_path(
            spec_path,
            supplied_checkpoint_compatibility,
            field="pc01_checkpoint_compatibility_receipt",
        )
        resolved_checkpoint_compatibility = checkpoint_compatibility_source.resolve()
        if (
            resolved_checkpoint_compatibility == repo
            or repo in resolved_checkpoint_compatibility.parents
        ):
            raise SchemaError(
                "pc01_checkpoint_compatibility_receipt must be measured outside "
                "the source checkout"
            )
        (
            _,
            checkpoint_compatibility_binding,
            checkpoint_compatibility_source_files,
        ) = _validate_pc01_checkpoint_compatibility_readiness(
            checkpoint_compatibility_source,
            repository_root=repo,
            expected_source_commit=commit,
            model_manifest_path=model_by_seed[42],
            selection_evidence_path=selection_evidence_path,
        )
        checkpoint_compatibility_path = (
            output / PC01_CHECKPOINT_COMPATIBILITY_RECEIPT_RELATIVE_PATH
        )
        checkpoint_compatibility_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            checkpoint_compatibility_source,
            checkpoint_compatibility_path,
        )
        checkpoint_compatibility_path.chmod(0o444)
        (
            _,
            staged_checkpoint_compatibility_binding,
            _,
        ) = _validate_pc01_checkpoint_compatibility_readiness(
            checkpoint_compatibility_path,
            repository_root=repo,
            expected_source_commit=commit,
            model_manifest_path=model_by_seed[42],
            selection_evidence_path=selection_evidence_path,
        )
        if staged_checkpoint_compatibility_binding != checkpoint_compatibility_binding:
            raise SchemaError(
                "staged PC-01 checkpoint compatibility receipt changed identity"
            )
    elif supplied_checkpoint_compatibility is not None:
        raise SchemaError(
            "pc01_checkpoint_compatibility_receipt is provisional PC-01 pilot "
            "evidence and cannot authorize another campaign"
        )
    memory_values = spec.get("memory_manifests")
    if not isinstance(memory_values, list):
        raise SchemaError("handoff input memory_manifests must be an array")
    memory_paths = [
        _spec_path(spec_path, value, field=f"memory_manifests[{index}]")
        for index, value in enumerate(memory_values)
    ]
    source_memory_by_seed = _validate_seed_manifest_coverage(
        memory_paths, matched_seeds, kind="memory", required=True
    )
    _validate_model_memory_source_bindings(
        model_by_seed, source_memory_by_seed, protocol_source=protocol_path
    )
    memory_by_seed: dict[int, Path] = {}
    for seed, source_manifest in sorted(source_memory_by_seed.items()):
        destination = output / "memory" / f"seed_{seed}"
        for name in FROZEN_MEMORY_STORE_FILES:
            _copy_exact(source_manifest.parent / name, destination / name)
        _load_and_verify_frozen_memory_store(destination)
        memory_by_seed[seed] = destination / "manifest.json"
    _validate_model_memory_source_bindings(
        model_by_seed, memory_by_seed, protocol_source=protocol_path
    )

    recovery_payload = read_json(recovery_source)
    recovery_payload["duplicate_audit_manifest"] = str(duplicate_path)
    recovery_path = output / "recovery_scenarios.json"
    atomic_write_json(recovery_path, recovery_payload)
    _bind_duplicate_audit(
        input_path=duplicate_input_source,
        output_path=duplicate_path,
        task_rows=task_rows,
        task_snapshot_path=task_snapshot_path,
        recovery_path=recovery_path,
        memory_by_seed=memory_by_seed,
        repository_root=repo,
        assignment_package_root=joint_assignment_source,
        preparation_package_root=joint_preparation_source,
        resolved_task_export_path=export_path,
        approved_task_registry_path=registry_path,
        registered_recovery_scenarios_path=recovery_source,
        duplicate_audit_registration_path=duplicate_registration_source,
        provenance_manifest_path=joint_provenance_source,
    )
    recovery_rows, recovery_metadata = _load_recovery_scenarios(recovery_path)
    _validate_recovery_campaign_counts(
        task_rows,
        recovery_rows,
        recovery_metadata,
        campaign,
        matched_seeds=matched_seeds,
        repeat_count=len(campaign.get("repeat_ids", [])),
        pilot_only=str(campaign.get("campaign_kind")) != "locked_final",
    )

    runner_spec = spec.get("runner")
    if not isinstance(runner_spec, Mapping):
        raise SchemaError("handoff input requires runner mapping")
    runner_entrypoint = str(runner_spec.get("runner_entrypoint", "")).strip()
    integration_entrypoint = str(
        runner_spec.get("runtime_integration_entrypoint", "")
    ).strip()
    if ":" not in runner_entrypoint or ":" not in integration_entrypoint:
        raise SchemaError("runner and runtime integration entrypoints must use module:attribute")
    if runner_entrypoint != PRODUCTION_RUNNER_ENTRYPOINT:
        raise SchemaError(
            "runner.runner_entrypoint must name the supported production runner"
        )
    primary_relative, _ = _repo_source(
        repo,
        runner_spec.get("primary_source_relative_path"),
        field="runner.primary_source_relative_path",
    )
    if primary_relative != PRODUCTION_RUNNER_SOURCE:
        raise SchemaError(
            "runner.primary_source_relative_path must name the production runner source"
        )
    integration_relative, integration_source = _repo_source(
        repo,
        runner_spec.get("runtime_integration_source_relative_path"),
        field="runner.runtime_integration_source_relative_path",
    )
    provider_bootstrap_value = runner_spec.get(
        PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD
    )
    if not isinstance(provider_bootstrap_value, Mapping):
        raise SchemaError(
            "runner requires a frozen PC-01 operations-provider bootstrap"
        )
    source_values = runner_spec.get("source_relative_paths")
    if not isinstance(source_values, list) or not source_values:
        raise SchemaError("runner.source_relative_paths must be a non-empty array")
    source_rows: list[dict[str, str]] = []
    source_paths: dict[str, Path] = {}
    for index, value in enumerate(source_values):
        relative, source = _repo_source(
            repo, value, field=f"runner.source_relative_paths[{index}]"
        )
        if relative in source_paths:
            raise SchemaError("runner source paths must be unique")
        source_paths[relative] = source
    evaluator_relative = str(evaluator_source.relative_to(repo))
    required_sources = {
        primary_relative,
        integration_relative,
        evaluator_relative,
        EVALUATION_CLI_SOURCE_RELATIVE_PATH,
        "src/web_agent/train/selection.py",
        "src/web_agent/eval/table2/selection_evidence.py",
        "src/web_agent/eval/table2/model_compatibility.py",
        "src/web_agent/train/gold_stages.py",
        "scripts/run_gold.py",
        "src/web_agent/eval/table2/live_deployment.py",
        AUDIT_TOOL_SOURCE_RELATIVE_PATH,
        str(JOINT_DUPLICATE_AUDIT_CONFIG_RELATIVE_PATH),
        str(P4_SOURCE_AUTHORITY_RELATIVE_PATH),
        *AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS,
        *EVALUATION_CONTROL_SOURCE_RELATIVE_PATHS,
    }
    required_sources.update(
        str(path.relative_to(repo))
        for path in checkpoint_compatibility_source_files
    )
    required_sources.update(
        row["relative_path"]
        for row in live_deployment.binding["capability_source_files"]
    )
    if selection_mode == "three_candidate_final":
        required_sources.add("scripts/compare_full_models.py")
    if not required_sources.issubset(source_paths):
        raise SchemaError(
            "runner source set must explicitly include the CLI, primary, integration, "
            "evaluator, validation-selection, compatibility-gate, and generator sources"
        )
    for relative in sorted(source_paths):
        source_rows.append(
            {"relative_path": relative, "sha256": sha256_file(source_paths[relative])}
        )
    provider_bootstrap = validate_pc01_provider_bootstrap_binding(
        provider_bootstrap_value,
        repository_root=repo,
        source_hashes={row["relative_path"]: row["sha256"] for row in source_rows},
    )
    capabilities = live_deployment.manifest.get("capabilities")
    broker = live_deployment.manifest.get("sealed_page_broker")
    if not isinstance(capabilities, Mapping) or not isinstance(broker, Mapping):
        raise SchemaError("live deployment capability planes are malformed")
    _validate_handoff_live_capability_source_planes(live_deployment)
    runtime_source_paths = {
        str(row.get("source_relative_path") or "")
        for capability_id, row in capabilities.items()
        if capability_id != "sealed_webarena_evaluator" and isinstance(row, Mapping)
    }
    forbidden_provider_sources = {
        str(capabilities["sealed_webarena_evaluator"].get("source_relative_path") or ""),
        str(broker.get("source_relative_path") or ""),
    }
    provider_source_relative = provider_bootstrap["source_relative_path"]
    if (
        provider_source_relative not in runtime_source_paths
        or provider_source_relative in forbidden_provider_sources
    ):
        raise SchemaError(
            "PC-01 provider factory source must be runtime-only, never sealed, "
            "broker, or shared-plane"
        )
    integration_identity = {
        "entrypoint": integration_entrypoint,
        "source_relative_path": integration_relative,
        "source_sha256": sha256_file(integration_source),
    }
    expected_identity = _expected_runner_runtime_identity(
        model_by_seed=model_by_seed,
        memory_by_seed=memory_by_seed,
        protocol=protocol,
        prompt_sources={
            "parameter_provider_v1.txt": _resolve_input(
                repo, protocol["parameter_provider"]["prompt"]
            ),
            "e0_action_v1.txt": protocol_path.parent / "prompts" / "e0_action_v1.txt",
        },
        environment_path=environment_path,
        resolved_task_snapshot_path=task_snapshot_path,
        runtime_integration=integration_identity,
        selection_evidence_path=selection_evidence_path,
        checkpoint_compatibility_receipt_path=checkpoint_compatibility_path,
    )
    attestation = {
        "schema_version": RUNNER_ATTESTATION_SCHEMA_VERSION,
        "attestation_scope": EVALUATION_RUNNER_SCOPE,
        "runner_entrypoint": runner_entrypoint,
        "runtime_integration_entrypoint": integration_entrypoint,
        PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD: provider_bootstrap,
        PC01_PAGE_BROKER_SECURITY_FIELD: (
            blocked_pc01_page_broker_security_binding()
        ),
        "repository_commit": commit,
        "primary_source_relative_path": primary_relative,
        "source_files": source_rows,
        "source_set_sha256": sha256_json(source_rows),
        "runtime_identity": expected_identity,
    }
    attestation_path = output / "runner_attestation.json"
    atomic_write_json(attestation_path, attestation)
    validate_runner_attestation_payload(
        attestation,
        repository_root=repo,
        repository_commit=commit,
        expected_runtime_identity=expected_identity,
    )

    prepared_campaign = dict(campaign)
    prepared_campaign.update(
        {
            "handoff_manifest": str(output / "handoff_manifest.json"),
            "resolved_task_snapshot": str(task_snapshot_path),
            "recovery_scenarios": str(recovery_path),
            "duplicate_audit_manifest": str(duplicate_path),
            "environment_manifest": str(environment_path),
            "runner_attestation": str(attestation_path),
            "runtime_integration_entrypoint": integration_entrypoint,
            "runtime_integration_source": integration_relative,
            "checkpoint_selection_evidence": str(selection_evidence_path),
            "joint_duplicate_assignment_package": str(
                joint_evidence_root / "assignment"
            ),
            "p4_preparation_package": str(
                joint_evidence_root / "preparation"
            ),
            "joint_duplicate_resolved_task_export": str(
                joint_evidence_root / "resolved_task_export.json"
            ),
            "joint_duplicate_registered_recovery_scenarios": str(
                joint_evidence_root / "registered_recovery_scenarios.json"
            ),
            "joint_duplicate_audit_registration": str(
                tracked_evidence_root
                / JOINT_DUPLICATE_AUDIT_REGISTRATION_RELATIVE_PATH
            ),
            "joint_duplicate_provenance_manifest": str(
                joint_evidence_root / "provenance_manifest.json"
            ),
        }
    )
    if checkpoint_compatibility_path is not None:
        prepared_campaign["pc01_checkpoint_compatibility_receipt"] = str(
            checkpoint_compatibility_path
        )
    campaign_path = output / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(prepared_campaign, sort_keys=False), encoding="utf-8"
    )

    freeze_arguments = {
        "handoff_manifest": str(output / "handoff_manifest.json"),
        "campaign_config": str(campaign_path),
        "resolved_task_snapshot": str(task_snapshot_path),
        "environment_manifest": str(environment_path),
        "runner_attestation": str(attestation_path),
        "checkpoint_selection_evidence": str(selection_evidence_path),
        "model_manifests": [str(model_by_seed[seed]) for seed in sorted(model_by_seed)],
        "memory_manifests": [str(memory_by_seed[seed]) for seed in sorted(memory_by_seed)],
    }
    if checkpoint_compatibility_path is not None:
        freeze_arguments["pc01_checkpoint_compatibility_receipt"] = str(
            checkpoint_compatibility_path
        )
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "handoff_manifest.json"
    }
    handoff = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "repository_root": str(repo),
        "repository_commit": commit,
        "campaign_mode": "evaluation",
        "selection_mode": selection_mode,
        "matched_seeds": matched_seeds,
        LIVE_DEPLOYMENT_BINDING_FIELD: live_deployment.binding,
        PC01_CHECKPOINT_COMPATIBILITY_BINDING_FIELD: (
            checkpoint_compatibility_binding
        ),
        "freeze_arguments": freeze_arguments,
        "files": files,
    }
    atomic_write_json(output / "handoff_manifest.json", handoff)
    return handoff


def main() -> None:
    args = parse_args()
    result = prepare_handoff(
        spec_path=args.spec,
        output_dir=args.output_dir,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
