"""Fail-closed execution identity and infrastructure-rerun contracts.

These guards deliberately remain dependency-light.  They are used both while
freezing a campaign and immediately before executing every physical block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import functools
import inspect
from pathlib import Path
import subprocess
from typing import Any

from .common import (
    SCHEMA_VERSION,
    SchemaError,
    Table2Error,
    canonical_json_bytes,
    read_json,
    safe_relative_path,
    sha256_file,
    sha256_json,
)


RUNNER_ATTESTATION_SCHEMA_VERSION = "table2-evaluation-runner-attestation-v2"
PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD = "pc01_operations_provider_bootstrap"
PC01_PROVIDER_BOOTSTRAP_SCHEMA_VERSION = "table2-pc01-provider-bootstrap-v1"
PC01_PROVIDER_PUBLIC_CONTRACT_SCHEMA_VERSION = (
    "table2-pc01-provider-public-contract-v1"
)
PC01_PROVIDER_INSTALLATION_RECEIPT_SCHEMA_VERSION = (
    "table2-pc01-provider-installation-receipt-v1"
)
PC01_PROVIDER_BOUNDARY_CLAIM_SCOPE = (
    "REVIEWED_CODE_ORACLE_FREE_DATAFLOW_ONLY"
)
PC01_PAGE_BROKER_SECURITY_FIELD = "pc01_page_broker_security"
PC01_PAGE_BROKER_SECURITY_SCHEMA_VERSION = (
    "table2-pc01-page-broker-security-v1"
)
PC01_PAGE_BROKER_SECURITY_CLAIM_SCOPE = (
    "REVIEWED_CODE_DATAFLOW_ONLY_NOT_PROCESS_ISOLATION"
)
PC01_PAGE_BROKER_SECURITY_BLOCKED_STATUS = (
    "BLOCKED_EXTERNAL_PROCESS_ISOLATION_REQUIRED"
)
EVALUATION_RUNNER_SCOPE = "FROZEN_EVALUATION_RUNNER"
ENGINEERING_SMOKE_SCOPE = "ENGINEERING_SMOKE_ONLY"
ENGINEERING_SMOKE_REASON = "ENGINEERING_SMOKE_ONLY"
FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH = "dependency.lock"
PRODUCTION_RUNNER_ENTRYPOINT = (
    "web_agent.eval.table2.production_runner:create_runner"
)
EVALUATION_CLI_SOURCE_RELATIVE_PATH = "scripts/run_table2_evaluation.py"
ANALYSIS_SOURCE_IDENTITY_SCHEMA_VERSION = "table2-analysis-source-identity-v1"
ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS: tuple[str, ...] = (
    "scripts/summarize_table2.py",
    "scripts/validate_table2_artifacts.py",
    "src/web_agent/eval/table2/common.py",
    "src/web_agent/eval/table2/execution_guard.py",
    "src/web_agent/eval/table2/metrics.py",
    "src/web_agent/eval/table2/package_validator.py",
    "src/web_agent/eval/table2/retrieval_metrics.py",
    "src/web_agent/eval/table2/statistics.py",
    "src/web_agent/eval/table2/summary.py",
)
REGISTERED_INFRASTRUCTURE_REASONS: tuple[str, ...] = (
    "BENCHMARK_SERVICE_UNAVAILABLE",
    "BROWSER_CONTROLLER_DISCONNECTED",
    "ENVIRONMENT_RESET_FAILED",
    "FROZEN_DEPENDENCY_UNAVAILABLE",
)

_PC01_PROVIDER_BOOTSTRAP_FIELDS = frozenset(
    {
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
)


def blocked_pc01_page_broker_security_binding() -> dict[str, Any]:
    """Return the only broker-security claim this source can currently make.

    The repository broker keeps runtime and evaluator methods on distinct typed
    objects, but both objects and the raw live-page state exist in one importable
    Python process.  That is a useful reviewed-code dataflow fixture; it is not
    an enforceable confidentiality boundary.  No caller may turn this record
    into campaign authority by changing a Boolean or attaching self-authored
    evidence.  A future process-isolated transport requires a new registered
    schema and independent deployment evidence.
    """

    return {
        "schema_version": PC01_PAGE_BROKER_SECURITY_SCHEMA_VERSION,
        "status": PC01_PAGE_BROKER_SECURITY_BLOCKED_STATUS,
        "claim_scope": PC01_PAGE_BROKER_SECURITY_CLAIM_SCOPE,
        "architecture": "same_process_in_memory_typed_capabilities",
        "same_process_broker": True,
        "kernel_process_isolation": False,
        "runtime_process_can_import_sealed_capability": True,
        "external_process_isolation_evidence_present": False,
        "production_dispatch_authorized": False,
    }


def validate_pc01_page_broker_security_binding(value: object) -> dict[str, Any]:
    """Validate the exact fail-closed broker-security non-claim."""

    expected = blocked_pc01_page_broker_security_binding()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise SchemaError(
            "PC-01 page-broker security binding must record the current "
            "same-process, non-isolated production blocker exactly"
        )
    return expected


def assert_pc01_page_broker_production_authorized(value: object) -> None:
    """Always stop this source version before live provider/runtime dispatch."""

    binding = validate_pc01_page_broker_security_binding(value)
    if binding["production_dispatch_authorized"] is not True:
        raise SchemaError(
            "PC-01 live campaign is blocked: the in-process page broker is only "
            "a reviewed-code dataflow fixture; externally evidenced process "
            "isolation is required before any provider or browser runtime loads"
        )
    raise SchemaError(
        "PC-01 page-broker authority is unsupported by this registered schema"
    )


@dataclass(frozen=True, slots=True)
class PC01ProviderInstallationReceipt:
    """Immutable post-factory evidence for one same-process provider install.

    This receipt attests only the reviewed source and dataflow boundary.  Its
    explicit ``kernel_filesystem_sandbox=False`` field prevents the record from
    being misread as an operating-system sandbox claim.
    """

    schema_version: str
    record_type: str
    status: str
    claim_scope: str
    same_process_factory: bool
    kernel_filesystem_sandbox: bool
    factory_entrypoint: str
    factory_module: str
    factory_qualname: str
    factory_source_relative_path: str
    factory_source_sha256: str
    bootstrap_context_sha256: str
    pre_factory_campaign_state_sha256: str
    post_factory_campaign_state_sha256: str
    provider_boundary_receipt_sha256: str
    credential_public_identity_sha256: str
    expected_provider_public_contract_sha256: str
    actual_provider_public_contract_sha256: str

    def __post_init__(self) -> None:
        fixed = {
            "schema_version": PC01_PROVIDER_INSTALLATION_RECEIPT_SCHEMA_VERSION,
            "record_type": "PC01ProviderInstallationReceipt",
            "status": "PASS",
            "claim_scope": PC01_PROVIDER_BOUNDARY_CLAIM_SCOPE,
            "same_process_factory": True,
            "kernel_filesystem_sandbox": False,
        }
        for field_name, expected in fixed.items():
            if getattr(self, field_name) != expected:
                raise SchemaError(
                    f"PC-01 provider installation receipt {field_name} mismatch"
                )
        if self.factory_entrypoint.count(":") != 1:
            raise SchemaError(
                "PC-01 provider installation factory entrypoint is malformed"
            )
        module_name, attribute_name = self.factory_entrypoint.split(":", 1)
        if (
            self.factory_module != module_name
            or self.factory_qualname != attribute_name
            or not module_name
            or not attribute_name
            or "." in attribute_name
        ):
            raise SchemaError(
                "PC-01 provider installation factory identity is inconsistent"
            )
        relative = safe_relative_path(self.factory_source_relative_path).as_posix()
        if relative != self.factory_source_relative_path:
            raise SchemaError(
                "PC-01 provider installation source path is not canonical"
            )
        for field_name in (
            "factory_source_sha256",
            "bootstrap_context_sha256",
            "pre_factory_campaign_state_sha256",
            "post_factory_campaign_state_sha256",
            "provider_boundary_receipt_sha256",
            "credential_public_identity_sha256",
            "expected_provider_public_contract_sha256",
            "actual_provider_public_contract_sha256",
        ):
            if not is_sha256(getattr(self, field_name)):
                raise SchemaError(
                    f"PC-01 provider installation {field_name} must be SHA-256"
                )
        if (
            self.pre_factory_campaign_state_sha256
            != self.post_factory_campaign_state_sha256
        ):
            raise SchemaError(
                "PC-01 campaign authority changed during provider installation"
            )
        if (
            self.expected_provider_public_contract_sha256
            != self.actual_provider_public_contract_sha256
        ):
            raise SchemaError(
                "PC-01 actual provider contract differs from frozen expectation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @property
    def receipt_sha256(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: object) -> "PC01ProviderInstallationReceipt":
        if not isinstance(value, Mapping):
            raise SchemaError("PC-01 provider installation receipt must be an object")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise SchemaError(
                "PC-01 provider installation receipt fields differ from schema"
            )
        return cls(**{name: value[name] for name in expected})

    def validate_provider_contract(self, actual_sha256: str) -> None:
        if (
            not is_sha256(actual_sha256)
            or actual_sha256 != self.actual_provider_public_contract_sha256
            or actual_sha256 != self.expected_provider_public_contract_sha256
        ):
            raise SchemaError(
                "registered provider differs from its installation receipt"
            )


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class InfrastructureInvalidError(Table2Error):
    """Typed authorization request for one preregistered whole-block rerun.

    Generic connection/time-out exceptions are intentionally unrelated to
    this type.  A benchmark adapter must classify the fault at its boundary
    and attach immutable diagnostic evidence before orchestration may consume
    a rerun.
    """

    infrastructure_invalid = True

    def __init__(
        self,
        *,
        reason_code: str,
        adapter_id: str,
        adapter_version: str,
        operation: str,
        adapter_evidence: Mapping[str, Any],
    ) -> None:
        self.reason_code = _nonempty(reason_code, field="reason_code")
        self.adapter_id = _nonempty(adapter_id, field="adapter_id")
        self.adapter_version = _nonempty(adapter_version, field="adapter_version")
        self.operation = _nonempty(operation, field="operation")
        if not isinstance(adapter_evidence, Mapping) or not adapter_evidence:
            raise ValueError("adapter_evidence must be a non-empty mapping")
        evidence = dict(adapter_evidence)
        required = {
            "adapter_event_id",
            "failure_class",
            "diagnostic_sha256",
            "retryable",
        }
        missing = required - set(evidence)
        if missing:
            raise ValueError(
                "adapter_evidence is missing required fields: "
                + ", ".join(sorted(missing))
            )
        for field in ("adapter_event_id", "failure_class"):
            _nonempty(evidence[field], field=f"adapter_evidence.{field}")
        if not is_sha256(evidence["diagnostic_sha256"]):
            raise ValueError("adapter_evidence.diagnostic_sha256 must be SHA-256")
        if evidence["retryable"] is not True:
            raise ValueError("adapter_evidence.retryable must be exactly true")
        # Round-trip through the canonical encoder now so unsupported values
        # fail at the adapter boundary rather than after a partial block.
        canonical_json_bytes(evidence)
        self.adapter_evidence = evidence
        super().__init__(
            f"{self.reason_code} at {self.adapter_id}:{self.operation}"
        )


    def evidence_record(
        self,
        *,
        campaign_mode: str,
        block_id: str,
        attempt_id: int,
        system_id: str,
        episode_id: str,
    ) -> dict[str, Any]:
        allowed = set(REGISTERED_INFRASTRUCTURE_REASONS)
        engineering_smoke = campaign_mode == "smoke"
        if engineering_smoke:
            allowed.add(ENGINEERING_SMOKE_REASON)
        if self.reason_code not in allowed:
            raise SchemaError(
                f"infrastructure reason is not preregistered: {self.reason_code}"
            )
        if self.reason_code == ENGINEERING_SMOKE_REASON and not engineering_smoke:
            raise SchemaError("ENGINEERING_SMOKE_ONLY cannot authorize an evaluation rerun")
        evidence_sha = sha256_json(self.adapter_evidence)
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "InfrastructureInvalidEvidence",
            "evidence_scope": (
                ENGINEERING_SMOKE_SCOPE if engineering_smoke else EVALUATION_RUNNER_SCOPE
            ),
            "reason_code": self.reason_code,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "operation": self.operation,
            "adapter_evidence": self.adapter_evidence,
            "adapter_evidence_sha256": evidence_sha,
            "block_id": block_id,
            "attempt_id": int(attempt_id),
            "system_id": system_id,
            "episode_id": episode_id,
        }


def validate_pc01_provider_bootstrap_binding(
    value: object,
    *,
    repository_root: Path,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate the exact runtime-only provider-factory registration."""

    if not isinstance(value, Mapping) or set(value) != _PC01_PROVIDER_BOOTSTRAP_FIELDS:
        raise SchemaError("PC-01 provider bootstrap binding fields differ from schema")
    result = {str(key): str(item) for key, item in value.items()}
    fixed = {
        "schema_version": PC01_PROVIDER_BOOTSTRAP_SCHEMA_VERSION,
        "provider_contract_schema_version": (
            PC01_PROVIDER_PUBLIC_CONTRACT_SCHEMA_VERSION
        ),
        "source_plane": "runtime_only",
    }
    for field, expected in fixed.items():
        if result.get(field) != expected:
            raise SchemaError(f"PC-01 provider bootstrap {field} mismatch")
    entrypoint = _nonempty(
        result.get("factory_entrypoint"), field="provider factory entrypoint"
    )
    if entrypoint.count(":") != 1:
        raise SchemaError("provider factory entrypoint must use module:attribute")
    module_name, attribute_name = entrypoint.split(":", 1)
    if (
        result.get("factory_module") != module_name
        or result.get("factory_qualname") != attribute_name
        or "." in attribute_name
    ):
        raise SchemaError(
            "provider factory module/qualname differs from exact entrypoint"
        )
    relative = safe_relative_path(result.get("source_relative_path", "")).as_posix()
    if relative != result.get("source_relative_path"):
        raise SchemaError("provider factory source path is not canonical")
    root = repository_root.resolve()
    source = (root / relative).resolve()
    if (
        root not in source.parents
        or source.is_symlink()
        or not source.is_file()
        or not is_sha256(result.get("source_sha256"))
        or sha256_file(source) != result.get("source_sha256")
    ):
        raise SchemaError("provider factory source identity differs from checkout")
    if source_hashes is not None and source_hashes.get(relative) != result.get(
        "source_sha256"
    ):
        raise SchemaError("provider factory source is absent from runner attestation")
    if not is_sha256(result.get("expected_provider_public_contract_sha256")):
        raise SchemaError("provider public-contract hash must be SHA-256")
    return result


def validate_infrastructure_evidence_record(
    value: Mapping[str, Any],
    *,
    campaign_mode: str,
    block_id: str,
    attempt_id: int,
    system_id: str,
    episode_id: str,
) -> None:
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "InfrastructureInvalidEvidence",
        "evidence_scope": (
            ENGINEERING_SMOKE_SCOPE
            if campaign_mode == "smoke"
            else EVALUATION_RUNNER_SCOPE
        ),
        "block_id": block_id,
        "attempt_id": int(attempt_id),
        "system_id": system_id,
        "episode_id": episode_id,
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise SchemaError(f"infrastructure evidence {key} mismatch")
    reason = str(value.get("reason_code", ""))
    allowed = set(REGISTERED_INFRASTRUCTURE_REASONS)
    if campaign_mode == "smoke":
        allowed.add(ENGINEERING_SMOKE_REASON)
    if reason not in allowed:
        raise SchemaError(f"infrastructure evidence reason is not registered: {reason}")
    if reason == ENGINEERING_SMOKE_REASON and campaign_mode != "smoke":
        raise SchemaError("engineering-smoke infrastructure evidence entered evaluation")
    for field in ("adapter_id", "adapter_version", "operation"):
        _nonempty(value.get(field), field=f"infrastructure.{field}")
    evidence = value.get("adapter_evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise SchemaError("infrastructure adapter evidence is absent")
    if sha256_json(evidence) != value.get("adapter_evidence_sha256"):
        raise SchemaError("infrastructure adapter evidence hash mismatch")
    # Reuse the exception constructor as the strict evidence schema checker.
    InfrastructureInvalidError(
        reason_code=reason,
        adapter_id=str(value["adapter_id"]),
        adapter_version=str(value["adapter_version"]),
        operation=str(value["operation"]),
        adapter_evidence=evidence,
    )


def validate_runner_attestation_payload(
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
    repository_commit: str,
    expected_runtime_identity: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Validate an evaluation runner attestation against the freeze inputs."""

    if payload.get("schema_version") != RUNNER_ATTESTATION_SCHEMA_VERSION:
        raise SchemaError("evaluation runner attestation schema mismatch")
    if payload.get("attestation_scope") != EVALUATION_RUNNER_SCOPE:
        raise SchemaError("evaluation runner attestation has a non-evaluation scope")
    entrypoint = _nonempty(payload.get("runner_entrypoint"), field="runner_entrypoint")
    if ":" not in entrypoint:
        raise SchemaError("runner_entrypoint must use module:attribute syntax")
    integration_entrypoint = _nonempty(
        payload.get("runtime_integration_entrypoint"),
        field="runtime_integration_entrypoint",
    )
    if ":" not in integration_entrypoint:
        raise SchemaError("runtime_integration_entrypoint must use module:attribute syntax")
    if payload.get("repository_commit") != repository_commit:
        raise SchemaError("runner attestation repository commit differs from campaign freeze")
    if payload.get("runtime_identity") != dict(expected_runtime_identity):
        raise SchemaError("runner attestation runtime identity differs from frozen inputs")
    source_rows = payload.get("source_files")
    if not isinstance(source_rows, list) or not source_rows:
        raise SchemaError("runner attestation requires non-empty source_files")
    primary = str(payload.get("primary_source_relative_path") or "")
    resolved: list[Path] = []
    seen: set[str] = set()
    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            raise SchemaError(f"runner source_files[{index}] must be an object")
        relative = str(safe_relative_path(str(row.get("relative_path", ""))))
        if not relative or relative in seen:
            raise SchemaError("runner source paths must be non-empty and unique")
        seen.add(relative)
        source = (repository_root / relative).resolve()
        if repository_root not in source.parents or not source.is_file():
            raise SchemaError(f"runner source is outside/missing from repository: {relative}")
        actual = sha256_file(source)
        if row.get("sha256") != actual:
            raise SchemaError(f"runner source hash differs from checkout: {relative}")
        resolved.append(source)
        normalized_rows.append({"relative_path": relative, "sha256": actual})
    if primary not in seen:
        raise SchemaError("primary runner source is not listed in source_files")
    if (
        entrypoint == PRODUCTION_RUNNER_ENTRYPOINT
        and EVALUATION_CLI_SOURCE_RELATIVE_PATH not in seen
    ):
        raise SchemaError(
            "production runner source set omits the pre-import evaluation CLI"
        )
    integration = expected_runtime_identity.get("runtime_integration")
    if not isinstance(integration, Mapping):
        raise SchemaError("expected runtime integration identity is absent")
    integration_source = str(integration.get("source_relative_path", ""))
    if (
        integration.get("entrypoint") != integration_entrypoint
        or integration_source not in seen
        or _source_hash_for_relative(normalized_rows, integration_source)
        != integration.get("source_sha256")
    ):
        raise SchemaError("runtime integration entrypoint/source is not frozen")
    evaluator = expected_runtime_identity.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise SchemaError("expected evaluator identity is absent")
    evaluator_source = str(evaluator.get("source_relative_path", ""))
    if (
        evaluator_source not in seen
        or _source_hash_for_relative(normalized_rows, evaluator_source)
        != evaluator.get("source_sha256")
    ):
        raise SchemaError("evaluator source is not included in frozen runner source set")
    if payload.get("source_set_sha256") != sha256_json(normalized_rows):
        raise SchemaError("runner source-set hash mismatch")
    provider_binding = payload.get(PC01_PROVIDER_BOOTSTRAP_BINDING_FIELD)
    if entrypoint == PRODUCTION_RUNNER_ENTRYPOINT:
        validate_pc01_provider_bootstrap_binding(
            provider_binding,
            repository_root=repository_root,
            source_hashes={
                row["relative_path"]: row["sha256"] for row in normalized_rows
            },
        )
        validate_pc01_page_broker_security_binding(
            payload.get(PC01_PAGE_BROKER_SECURITY_FIELD)
        )
    elif provider_binding is not None:
        raise SchemaError(
            "non-production runner attestation must not bind a PC-01 provider"
        )
    elif payload.get(PC01_PAGE_BROKER_SECURITY_FIELD) is not None:
        raise SchemaError(
            "non-production runner attestation must not claim PC-01 broker security"
        )
    return tuple(resolved)


def attested_source_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the already-validated source set as an exact path/hash map.

    This helper still validates shape and uniqueness so callers cannot turn a
    malformed attestation into a last-row-wins callback allow-list.
    """

    rows = payload.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise SchemaError("runner attestation source_files is malformed")
    result: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SchemaError(f"runner source_files[{index}] must be an object")
        relative = str(safe_relative_path(str(row.get("relative_path", ""))))
        digest = str(row.get("sha256", ""))
        if not relative or relative in result or not is_sha256(digest):
            raise SchemaError("runner callback source set is malformed or duplicated")
        result[relative] = digest
    return result


def validate_attested_callable_source(
    callback: Any,
    *,
    repository_root: Path,
    source_hashes: Mapping[str, str],
    field: str,
    expected_relative_path: str | None = None,
) -> str:
    """Require an injected callable's executable source to be hash-attested.

    The production architecture treats the attested Python sources as trusted.
    Consequently this checks the callable that was actually supplied, but does
    not attempt to sandbox its globals or recursively police arbitrary object
    graphs captured by a trusted closure.
    """

    if not callable(callback):
        raise SchemaError(f"{field} must be callable")
    target = callback.func if isinstance(callback, functools.partial) else callback
    if inspect.ismethod(target):
        target = target.__func__
    elif not (
        inspect.isfunction(target)
        or inspect.isclass(target)
        or inspect.ismethoddescriptor(target)
    ):
        target = getattr(type(target), "__call__", None)
    if target is None:
        raise SchemaError(f"cannot resolve source for injected callback {field}")
    try:
        source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise SchemaError(
            f"cannot resolve source for injected callback {field}"
        ) from exc
    if not source_name:
        raise SchemaError(f"cannot resolve source for injected callback {field}")
    root = repository_root.resolve()
    unresolved_source = Path(source_name).absolute()
    if unresolved_source.is_symlink():
        raise SchemaError(f"injected callback {field} source must not be a symlink")
    source = unresolved_source.resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise SchemaError(
            f"injected callback {field} is outside the attested repository: {source}"
        ) from exc
    expected = source_hashes.get(relative)
    if expected is None:
        raise SchemaError(
            f"injected callback {field} source is not attested: {relative}"
        )
    if expected_relative_path is not None and relative != expected_relative_path:
        raise SchemaError(
            f"injected callback {field} source differs from its registered entrypoint: "
            f"{relative!r} != {expected_relative_path!r}"
        )
    if not source.is_file() or source.is_symlink() or sha256_file(source) != expected:
        raise SchemaError(f"injected callback {field} source hash differs: {relative}")
    return relative


def assert_clean_git_checkout(repository_root: Path) -> str:
    """Return HEAD only when the executing checkout has no tracked/untracked drift."""

    root = repository_root.resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SchemaError("evaluation execution requires a Git checkout") from exc
    if not commit or status:
        raise SchemaError(
            "evaluation execution requires a clean Git checkout at the frozen commit"
        )
    return commit


def validate_dependency_lock_for_environment(
    environment_path: Path,
    environment: Mapping[str, Any] | None = None,
) -> Path:
    """Verify the lock file named and hashed by one environment manifest."""

    environment_path = environment_path.resolve()
    if environment is None:
        environment = read_json(environment_path)
    relative = str(environment.get("dependency_lock_relative_path", ""))
    if relative != FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH:
        raise SchemaError(
            "environment dependency lock must use the registered frozen path"
        )
    expected = environment.get("dependency_lock_sha256")
    if not is_sha256(expected):
        raise SchemaError("environment dependency lock hash is malformed")
    candidate = environment_path.parent / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise SchemaError("dependency lock is missing or symlinked")
    if sha256_file(candidate) != expected:
        raise SchemaError("dependency lock hash differs from environment")
    return candidate


def validate_frozen_dependency_lock(campaign_root: Path) -> Path:
    """Verify the exact dependency-lock bytes copied into an evaluation package."""

    return validate_dependency_lock_for_environment(
        campaign_root.resolve() / "frozen" / "environment.json"
    )


def validate_analysis_source_identity(
    campaign_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate the code that computes or revalidates Table 2 results.

    Engineering-smoke summaries retain an explicit non-paper identity. A real
    evaluation summary must run from the clean frozen commit and every
    registered analysis source must match both the live checkout and the copy
    carried in the runner attestation.
    """

    campaign_path = Path(campaign_root).resolve()
    campaign = read_json(campaign_path / "campaign_manifest.json")
    if campaign.get("campaign_mode") == "smoke":
        return {
            "schema_version": ANALYSIS_SOURCE_IDENTITY_SCHEMA_VERSION,
            "identity_scope": ENGINEERING_SMOKE_SCOPE,
            "paper_table_eligible": False,
            "repository_commit": None,
            "runner_attestation_sha256": None,
            "sources": [],
            "source_set_sha256": sha256_json([]),
        }
    if campaign.get("runner_identity_scope") != EVALUATION_RUNNER_SCOPE:
        raise SchemaError("evaluation analysis lacks frozen runner identity scope")

    repo = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[4]
    )
    commit = assert_clean_git_checkout(repo)
    if commit != campaign.get("repository_commit"):
        raise SchemaError(
            "Table 2 analysis checkout commit differs from the frozen campaign"
        )

    attestation_path = campaign_path / "frozen" / "runner_attestation.json"
    if attestation_path.is_symlink() or not attestation_path.is_file():
        raise SchemaError("Table 2 analysis lacks a regular runner attestation")
    if sha256_file(attestation_path) != campaign.get("runner_attestation_sha256"):
        raise SchemaError("Table 2 analysis runner attestation hash mismatch")
    attestation = read_json(attestation_path)
    source_hashes = attested_source_hashes(attestation)
    frozen_source_root = campaign_path / "frozen" / "runner_source"
    rows: list[dict[str, str]] = []
    for relative in ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS:
        expected = source_hashes.get(relative)
        live = repo / safe_relative_path(relative)
        frozen = frozen_source_root / safe_relative_path(relative)
        if (
            expected is None
            or live.is_symlink()
            or frozen.is_symlink()
            or not live.is_file()
            or not frozen.is_file()
            or sha256_file(live) != expected
            or sha256_file(frozen) != expected
        ):
            raise SchemaError(
                f"Table 2 analysis source is not live/frozen attested: {relative}"
            )
        rows.append({"relative_path": relative, "sha256": expected})
    return {
        "schema_version": ANALYSIS_SOURCE_IDENTITY_SCHEMA_VERSION,
        "identity_scope": "FROZEN_TABLE2_ANALYSIS_SOURCE",
        "paper_table_eligible": True,
        "repository_commit": commit,
        "runner_attestation_sha256": sha256_file(attestation_path),
        "sources": rows,
        "source_set_sha256": sha256_json(rows),
    }


def _source_hash_for_relative(
    rows: Sequence[Mapping[str, str]], relative: str
) -> str | None:
    for row in rows:
        if row.get("relative_path") == relative:
            return row.get("sha256")
    return None


def verify_runner_before_execution(
    campaign_root: Path,
    *,
    runner: Any,
    runner_entrypoint: str | None,
) -> None:
    """Verify live runner, loaded identities, source bytes, and checkout commit."""

    campaign = read_json(campaign_root / "campaign_manifest.json")
    mode = str(campaign.get("campaign_mode", "evaluation"))
    if mode == "smoke":
        if campaign.get("runner_identity_scope") != ENGINEERING_SMOKE_SCOPE:
            raise SchemaError("smoke runner bypass lacks ENGINEERING_SMOKE_ONLY scope")
        return
    if campaign.get("runner_identity_scope") != EVALUATION_RUNNER_SCOPE:
        raise SchemaError("evaluation campaign lacks a frozen runner identity")
    if not runner_entrypoint:
        raise SchemaError("evaluation execution requires the attested runner entrypoint")
    attestation_path = campaign_root / "frozen" / "runner_attestation.json"
    attestation = read_json(attestation_path)
    if sha256_file(attestation_path) != campaign.get("runner_attestation_sha256"):
        raise SchemaError("frozen runner attestation hash mismatch")
    if runner_entrypoint != attestation.get("runner_entrypoint"):
        raise SchemaError("CLI/direct runner entrypoint differs from frozen attestation")

    validate_frozen_dependency_lock(campaign_root)

    target = _runner_target(runner)
    source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    if not source_name:
        raise SchemaError("cannot resolve evaluation runner source file")
    source_path = Path(source_name).resolve()
    git_root = _git_root(source_path.parent)
    if git_root is None or source_path == git_root or git_root not in source_path.parents:
        raise SchemaError("evaluation runner source is not inside a Git checkout")
    current_commit = assert_clean_git_checkout(git_root)
    if current_commit != attestation.get("repository_commit"):
        raise SchemaError("executing runner checkout commit differs from frozen commit")

    source_rows = attestation.get("source_files")
    if not isinstance(source_rows, list):
        raise SchemaError("runner attestation source_files is malformed")
    by_relative = {
        str(row.get("relative_path")): str(row.get("sha256"))
        for row in source_rows
        if isinstance(row, Mapping)
    }
    primary = str(attestation.get("primary_source_relative_path", ""))
    try:
        actual_primary = str(source_path.relative_to(git_root))
    except ValueError as exc:
        raise SchemaError("runner source escaped its checkout") from exc
    if actual_primary != primary:
        raise SchemaError(
            "loaded runner source path differs from frozen primary source: "
            f"{actual_primary!r} != {primary!r}"
        )
    for relative, expected_hash in by_relative.items():
        live = (git_root / safe_relative_path(relative)).resolve()
        frozen = campaign_root / "frozen" / "runner_source" / safe_relative_path(relative)
        if (
            git_root not in live.parents
            or not live.is_file()
            or not frozen.is_file()
            or sha256_file(live) != expected_hash
            or sha256_file(frozen) != expected_hash
        ):
            raise SchemaError(f"runner source identity mismatch: {relative}")

    attest = getattr(runner, "evaluation_runner_attestation", None)
    if not callable(attest):
        raise SchemaError(
            "evaluation runner must expose evaluation_runner_attestation() for loaded identities"
        )
    actual_identity = attest()
    if not isinstance(actual_identity, Mapping):
        raise SchemaError("runner loaded-identity attestation must be a mapping")
    if dict(actual_identity) != attestation.get("runtime_identity"):
        raise SchemaError("actual loaded runner identities differ from frozen attestation")


def _runner_target(runner: Any) -> Any:
    if hasattr(runner, "run") and callable(runner.run):
        return runner.run
    if callable(runner):
        return runner
    raise SchemaError("runner must be callable or expose .run()")


def _git_root(path: Path) -> Path | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(value).resolve()


def _nonempty(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise SchemaError(f"{field} must be non-empty")
    return normalized
