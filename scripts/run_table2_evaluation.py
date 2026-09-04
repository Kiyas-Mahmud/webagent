"""Run or resume frozen paired E0--E3 Table 2 blocks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
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
    """Stop before importing a provider while the broker is same-process.

    This standard-library-only check intentionally duplicates the frozen
    non-claim in ``execution_guard``.  Importing the evaluation package merely
    to discover this blocker would already import runtime integration modules
    in the same interpreter that can import the sealed evaluator capability.
    """

    binding = attestation.get(PC01_PAGE_BROKER_SECURITY_FIELD)
    if not isinstance(binding, dict) or binding != (
        PC01_PAGE_BROKER_SECURITY_BLOCKED_BINDING
    ):
        raise RuntimeError(
            "evaluation bootstrap has no authenticated page-broker security status"
        )
    raise RuntimeError(
        "PC-01 live campaign is blocked before provider import: the current "
        "same-process page broker is reviewed-code engineering evidence only; "
        "a separately authenticated process-isolation implementation and receipt "
        "are required"
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
