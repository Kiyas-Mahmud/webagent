"""Credential-free assembly of one process-isolated WebArena episode.

This module is orchestration glue, not a live-deployment implementation.  It
does not discover credentials, construct a reset policy, choose an evaluator,
or infer any callback from a capability label.  A caller must supply a
previously validated live-deployment package, three explicit source-attested
child entrypoints, their complete repository dependency closure, and the
immutable measured timeout authority.  Screenshot ownership is derived from
each fresh episode runtime directory and is never shared across episodes.

The returned binding exposes only the runtime operation client plus distinct
finalization, abort, and cleanup capabilities.  The browser adapter and sealed
verifier stream remain owned by the child process.  The local broker receipt is
component evidence for the ``PILOT_ONLY`` architecture; it is not final-paper
deployment authority.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
import json
from pathlib import Path
import stat
from threading import RLock
from types import MappingProxyType
from typing import Any

from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymEpisodeAbortReceipt,
)
from web_agent.eval.table2.common import (
    SchemaError,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.live_deployment import (
    ValidatedPC01LiveDeployment,
    validate_pc01_live_deployment_binding,
)
from web_agent.eval.table2.process_broker import (
    PROCESS_BROKER_PILOT_EVALUATION_SCOPE,
    PROCESS_BROKER_SOURCE_PATHS,
    ProcessIsolatedBroker,
)
from web_agent.eval.table2.process_broker_finalization import (
    ProcessIsolatedEpisodeFinalizationBinding,
)
from web_agent.eval.table2.process_broker_protocol import (
    validated_policy_screenshot_root,
)
from web_agent.eval.table2.process_broker_timeout import (
    MEASURED_TIMEOUT_MODE,
    ProcessBrokerTimeoutExpectedAuthority,
    binding_sha256 as timeout_binding_sha256,
    load_authority_bound_timeout_calibration,
    measured_timeout_binding,
)
from web_agent.eval.table2.process_broker_webarena_backend import (
    PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
    PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
    PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION,
    ProcessBrokerWebArenaEnvironmentAdapter,
)
from web_agent.eval.table2.production_runner import (
    ProcessIsolatedWebArenaEpisodeBinding,
)
from web_agent.eval.table2.sealed_verifier import (
    SealedVerifierStreamTarget,
)
from web_agent.runtime.contracts import TaskSpecification


PROCESS_BROKER_EPISODE_FACTORY_SCHEMA_VERSION = (
    "table2-process-broker-live-episode-factory-v1"
)
PROCESS_BROKER_EPISODE_FACTORY_CLAIM_SCOPE = (
    "PILOT_ONLY_CREDENTIAL_FREE_ORCHESTRATION_NOT_FINAL_DEPLOYMENT_AUTHORITY"
)
PROCESS_BROKER_EPISODE_DESCRIPTOR_IDENTITY_SCHEMA_VERSION = (
    "table2-process-broker-episode-factory-descriptor-public-identity-v1"
)
PROCESS_BROKER_EPISODE_DESCRIPTOR_IDENTITY_RECORD_TYPE = (
    "ProcessBrokerEpisodeFactoryDescriptorPublicIdentity"
)

_ADAPTER_FACTORY_KEYWORD_PARAMETERS = ("task", "episode_runtime_dir")
_SEALED_TRANSITION_KEYWORD_PARAMETERS = (
    "task",
    "adapter",
    "receipt_binding",
    "evidence_writer",
)
_SEALED_FINALIZER_KEYWORD_PARAMETERS = (
    "task",
    "adapter",
    "episode_summary",
    "episode_runtime_dir",
    "evidence_writer",
)


class ProcessBrokerEpisodeFactoryError(RuntimeError):
    """A process episode could not be assembled from the frozen inputs."""


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_relative_python_path(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} must be a canonical relative Python path"
        )
    lexical = Path(value)
    if (
        lexical.is_absolute()
        or lexical.suffix != ".py"
        or lexical.as_posix() != value
        or "." in lexical.parts
        or ".." in lexical.parts
    ):
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} must be a canonical relative Python path"
        )
    return value


def _validated_repository_source(
    repository_root: Path,
    source: "SourceAttestedPython",
    *,
    field_name: str,
) -> Path:
    relative = _canonical_relative_python_path(
        source.source_relative_path,
        field_name=f"{field_name}.source_relative_path",
    )
    candidate = repository_root
    try:
        for component in Path(relative).parts:
            candidate /= component
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProcessBrokerEpisodeFactoryError(
                    f"{field_name} source contains a symlink component"
                )
    except OSError as exc:
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} source is unavailable"
        ) from exc
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} source escaped the repository"
        ) from exc
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} source must be a single-link regular file"
        )
    if sha256_file(resolved) != source.source_sha256:
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} source hash differs from its frozen identity"
        )
    return resolved


def _validate_entrypoint_source_contract(
    source_path: Path,
    entrypoint: "SourceAttestedEntrypoint",
    *,
    keyword_parameters: tuple[str, ...],
    field_name: str,
) -> None:
    """Prove the advertised top-level function exists before child import."""

    attribute_name = entrypoint.entrypoint.partition(":")[2]
    try:
        tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
    except (OSError, SyntaxError, ValueError) as exc:
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} source cannot be parsed"
        ) from exc
    declarations = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == attribute_name
    ]
    if len(declarations) != 1 or not isinstance(declarations[0], ast.FunctionDef):
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} is not one top-level synchronous function"
        )
    function = declarations[0]
    arguments = function.args
    if (
        function.decorator_list
        or arguments.posonlyargs
        or arguments.args
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or tuple(item.arg for item in arguments.kwonlyargs) != keyword_parameters
        or any(default is not None for default in arguments.kw_defaults)
    ):
        raise ProcessBrokerEpisodeFactoryError(
            f"{field_name} signature differs from the child call contract"
        )


@dataclass(frozen=True, slots=True)
class SourceAttestedPython:
    """One explicit repository source identity; no import is performed."""

    source_relative_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        _canonical_relative_python_path(
            self.source_relative_path,
            field_name="source-attested Python path",
        )
        if not _is_sha256(self.source_sha256):
            raise ProcessBrokerEpisodeFactoryError(
                "source-attested Python digest must be lowercase SHA-256"
            )


@dataclass(frozen=True, slots=True)
class SourceAttestedEntrypoint(SourceAttestedPython):
    """One explicit ``module:attribute`` bound to exact repository bytes."""

    entrypoint: str

    def __post_init__(self) -> None:
        super(SourceAttestedEntrypoint, self).__post_init__()
        module_name, separator, attribute_name = self.entrypoint.partition(":")
        if (
            separator != ":"
            or not module_name
            or any(not component.isidentifier() for component in module_name.split("."))
            or not attribute_name.isidentifier()
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "source-attested entrypoint must use module:attribute syntax"
            )
        expected = Path(*module_name.split(".")).with_suffix(".py")
        supplied = Path(self.source_relative_path)
        if supplied not in {expected, Path("src") / expected}:
            raise ProcessBrokerEpisodeFactoryError(
                "source-attested entrypoint differs from its source path"
            )


@dataclass(frozen=True, slots=True)
class ProcessBrokerLiveDeploymentDescriptor:
    """Immutable inputs required before a pilot broker may be constructed.

    ``live_deployment`` is an ``InitVar`` on purpose: mutable dictionaries from
    the validator are never retained.  Only canonical bytes and content hashes
    are stored, and the package is reopened from disk before every episode.
    """

    repository_root: Path
    live_deployment: InitVar[ValidatedPC01LiveDeployment]
    adapter_factory: SourceAttestedEntrypoint
    sealed_transition_callback: SourceAttestedEntrypoint
    sealed_finalizer: SourceAttestedEntrypoint
    backend_dependency_sources: tuple[SourceAttestedPython, ...]
    timeout_calibration_artifact_path: Path
    timeout_expected_authority: ProcessBrokerTimeoutExpectedAuthority
    schema_version: str = PROCESS_BROKER_EPISODE_FACTORY_SCHEMA_VERSION
    claim_scope: str = PROCESS_BROKER_EPISODE_FACTORY_CLAIM_SCOPE
    credential_material_embedded: bool = False
    frozen: bool = True
    _live_artifact_root: Path = field(init=False, repr=False)
    _live_binding_json: bytes = field(init=False, repr=False)
    _live_binding_sha256: str = field(init=False, repr=False)
    _live_manifest_sha256: str = field(init=False, repr=False)
    _timeout_authority_json: bytes = field(init=False, repr=False)
    _timeout_binding_sha256: str = field(init=False, repr=False)

    def __post_init__(self, live_deployment: ValidatedPC01LiveDeployment) -> None:
        if self.schema_version != PROCESS_BROKER_EPISODE_FACTORY_SCHEMA_VERSION:
            raise ProcessBrokerEpisodeFactoryError(
                "process episode descriptor schema version changed"
            )
        if self.claim_scope != PROCESS_BROKER_EPISODE_FACTORY_CLAIM_SCOPE:
            raise ProcessBrokerEpisodeFactoryError(
                "process episode descriptor claim scope changed"
            )
        if self.credential_material_embedded is not False or self.frozen is not True:
            raise ProcessBrokerEpisodeFactoryError(
                "process episode descriptor must be frozen and credential-free"
            )
        if type(live_deployment) is not ValidatedPC01LiveDeployment:
            raise ProcessBrokerEpisodeFactoryError(
                "descriptor requires the exact validated live-deployment type"
            )

        unresolved_repository = Path(self.repository_root).absolute()
        if (
            not unresolved_repository.is_dir()
            or unresolved_repository.is_symlink()
            or unresolved_repository.resolve(strict=True) != unresolved_repository
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "descriptor repository root must be canonical and non-symlinked"
            )
        repository = unresolved_repository.resolve(strict=True)
        object.__setattr__(self, "repository_root", repository)

        package_root = Path(live_deployment.package_root)
        if package_root.name != "live_deployment":
            raise ProcessBrokerEpisodeFactoryError(
                "validated live-deployment package root is not registered"
            )
        artifact_root = package_root.parent.resolve(strict=True)
        binding_json = canonical_json_bytes(live_deployment.binding)
        try:
            binding = json.loads(binding_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - canonical encoder
            raise ProcessBrokerEpisodeFactoryError(
                "validated live-deployment binding is not canonical JSON"
            ) from exc
        try:
            reopened = validate_pc01_live_deployment_binding(
                binding,
                artifact_root=artifact_root,
                repository_root=repository,
            )
        except (OSError, SchemaError, TypeError, ValueError) as exc:
            raise ProcessBrokerEpisodeFactoryError(
                "live-deployment package did not survive descriptor revalidation"
            ) from exc
        if (
            canonical_json_bytes(reopened.binding) != binding_json
            or canonical_json_bytes(reopened.manifest)
            != canonical_json_bytes(live_deployment.manifest)
            or reopened.package_root != package_root.resolve(strict=True)
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "reopened live-deployment package differs from validated input"
            )
        object.__setattr__(self, "_live_artifact_root", artifact_root)
        object.__setattr__(self, "_live_binding_json", binding_json)
        object.__setattr__(self, "_live_binding_sha256", sha256_json(binding))
        object.__setattr__(
            self,
            "_live_manifest_sha256",
            sha256_json(reopened.manifest),
        )

        for name in (
            "adapter_factory",
            "sealed_transition_callback",
            "sealed_finalizer",
        ):
            if type(getattr(self, name)) is not SourceAttestedEntrypoint:
                raise ProcessBrokerEpisodeFactoryError(
                    f"descriptor {name} requires an exact source-attested entrypoint"
                )
        if type(self.backend_dependency_sources) is not tuple or any(
            type(item) is not SourceAttestedPython
            for item in self.backend_dependency_sources
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "backend dependency closure must be a tuple of exact source identities"
            )

        self._validate_live_source_roles(reopened)
        self._validate_source_closure(reopened)
        for field_name, entrypoint, parameters in (
            (
                "adapter factory",
                self.adapter_factory,
                _ADAPTER_FACTORY_KEYWORD_PARAMETERS,
            ),
            (
                "sealed transition callback",
                self.sealed_transition_callback,
                _SEALED_TRANSITION_KEYWORD_PARAMETERS,
            ),
            (
                "sealed finalizer",
                self.sealed_finalizer,
                _SEALED_FINALIZER_KEYWORD_PARAMETERS,
            ),
        ):
            source_path = _validated_repository_source(
                repository,
                entrypoint,
                field_name=field_name,
            )
            _validate_entrypoint_source_contract(
                source_path,
                entrypoint,
                keyword_parameters=parameters,
                field_name=field_name,
            )

        if type(self.timeout_expected_authority) is not (
            ProcessBrokerTimeoutExpectedAuthority
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "descriptor requires the exact measured-timeout authority type"
            )
        supplied_timeout_path = Path(
            self.timeout_calibration_artifact_path
        ).absolute()
        registered_timeout_path = Path(
            self.timeout_expected_authority.calibration_artifact_path
        ).absolute()
        if supplied_timeout_path != registered_timeout_path:
            raise ProcessBrokerEpisodeFactoryError(
                "timeout artifact path differs from its expected authority"
            )
        try:
            calibration = load_authority_bound_timeout_calibration(
                self.timeout_expected_authority
            )
            timeout_binding = measured_timeout_binding(calibration)
        except (OSError, SchemaError, TypeError, ValueError) as exc:
            raise ProcessBrokerEpisodeFactoryError(
                "immutable measured-timeout authority did not validate"
            ) from exc
        if (
            timeout_binding.get("mode") != MEASURED_TIMEOUT_MODE
            or timeout_binding.get("measured_calibration_complete") is not True
            or timeout_binding.get("caller_selected_timeout") is not False
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "descriptor timeout binding is not measured and outcome-blind"
            )
        authority_json = canonical_json_bytes(
            self.timeout_expected_authority.to_dict()
        )
        object.__setattr__(self, "timeout_calibration_artifact_path", supplied_timeout_path)
        object.__setattr__(self, "_timeout_authority_json", authority_json)
        object.__setattr__(
            self,
            "_timeout_binding_sha256",
            timeout_binding_sha256(timeout_binding),
        )

    @property
    def live_deployment_binding_sha256(self) -> str:
        return self._live_binding_sha256

    @property
    def live_deployment_manifest_sha256(self) -> str:
        return self._live_manifest_sha256

    @property
    def measured_timeout_binding_sha256(self) -> str:
        return self._timeout_binding_sha256

    def public_identity(self) -> dict[str, object]:
        """Return the credential/path-free descriptor identity persisted by runs."""

        def entrypoint_identity(value: SourceAttestedEntrypoint) -> dict[str, str]:
            return {
                "entrypoint": value.entrypoint,
                "source_relative_path": value.source_relative_path,
                "source_sha256": value.source_sha256,
            }

        dependencies = sorted(
            (
                {
                    "source_relative_path": item.source_relative_path,
                    "source_sha256": item.source_sha256,
                }
                for item in self.backend_dependency_sources
            ),
            key=lambda item: item["source_relative_path"],
        )
        return {
            "schema_version": (
                PROCESS_BROKER_EPISODE_DESCRIPTOR_IDENTITY_SCHEMA_VERSION
            ),
            "record_type": PROCESS_BROKER_EPISODE_DESCRIPTOR_IDENTITY_RECORD_TYPE,
            "claim_scope": self.claim_scope,
            "live_deployment_binding_sha256": self._live_binding_sha256,
            "live_deployment_manifest_sha256": self._live_manifest_sha256,
            "measured_timeout_binding_sha256": self._timeout_binding_sha256,
            "adapter_factory": entrypoint_identity(self.adapter_factory),
            "sealed_transition_callback": entrypoint_identity(
                self.sealed_transition_callback
            ),
            "sealed_finalizer": entrypoint_identity(self.sealed_finalizer),
            "backend_dependency_sources": dependencies,
            "broker_backend_entrypoint": PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
            "broker_backend_source_relative_path": (
                PROCESS_BROKER_WEBARENA_BACKEND_SOURCE
            ),
            "broker_backend_source_sha256": sha256_file(
                self.repository_root / PROCESS_BROKER_WEBARENA_BACKEND_SOURCE
            ),
            "credential_material_embedded": False,
            "frozen": True,
        }

    @property
    def descriptor_sha256(self) -> str:
        return sha256_json(self.public_identity())

    def _validate_live_source_roles(
        self,
        deployment: ValidatedPC01LiveDeployment,
    ) -> None:
        capabilities = deployment.manifest.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ProcessBrokerEpisodeFactoryError(
                "validated live deployment lacks capability identities"
            )
        runtime_source = capabilities.get("oracle_blind_browser_mapping")
        sealed_source = capabilities.get("sealed_webarena_evaluator")
        if not isinstance(runtime_source, Mapping) or not isinstance(
            sealed_source, Mapping
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "live deployment lacks runtime or sealed source authority"
            )
        expected_runtime = (
            runtime_source.get("source_relative_path"),
            runtime_source.get("source_sha256"),
        )
        expected_sealed = (
            sealed_source.get("source_relative_path"),
            sealed_source.get("source_sha256"),
        )
        if (
            self.adapter_factory.source_relative_path,
            self.adapter_factory.source_sha256,
        ) != expected_runtime:
            raise ProcessBrokerEpisodeFactoryError(
                "adapter factory source differs from frozen browser-mapping authority"
            )
        for name, entrypoint in (
            ("sealed transition callback", self.sealed_transition_callback),
            ("sealed finalizer", self.sealed_finalizer),
        ):
            if (
                entrypoint.source_relative_path,
                entrypoint.source_sha256,
            ) != expected_sealed:
                raise ProcessBrokerEpisodeFactoryError(
                    f"{name} source differs from frozen sealed-evaluator authority"
                )

    def _declared_sources(self) -> tuple[SourceAttestedPython, ...]:
        return (
            SourceAttestedPython(
                self.adapter_factory.source_relative_path,
                self.adapter_factory.source_sha256,
            ),
            SourceAttestedPython(
                self.sealed_transition_callback.source_relative_path,
                self.sealed_transition_callback.source_sha256,
            ),
            SourceAttestedPython(
                self.sealed_finalizer.source_relative_path,
                self.sealed_finalizer.source_sha256,
            ),
            *self.backend_dependency_sources,
        )

    def _validate_source_closure(
        self,
        deployment: ValidatedPC01LiveDeployment,
    ) -> None:
        declared: dict[str, str] = {}
        for index, source in enumerate(self._declared_sources()):
            previous = declared.get(source.source_relative_path)
            if previous is not None and previous != source.source_sha256:
                raise ProcessBrokerEpisodeFactoryError(
                    "child source closure repeats a path with different hashes"
                )
            declared[source.source_relative_path] = source.source_sha256
            _validated_repository_source(
                self.repository_root,
                source,
                field_name=f"child source closure[{index}]",
            )
        if PROCESS_BROKER_WEBARENA_BACKEND_SOURCE in declared:
            raise ProcessBrokerEpisodeFactoryError(
                "generic broker backend cannot be repeated as a child dependency"
            )
        fixed: dict[str, str] = {}
        for relative in PROCESS_BROKER_SOURCE_PATHS:
            fixed_source = SourceAttestedPython(
                relative,
                sha256_file(self.repository_root / relative),
            )
            _validated_repository_source(
                self.repository_root,
                fixed_source,
                field_name="fixed process-broker source",
            )
            fixed[relative] = fixed_source.source_sha256
        required_rows = deployment.binding.get("capability_source_files")
        if not isinstance(required_rows, list) or not required_rows:
            raise ProcessBrokerEpisodeFactoryError(
                "live deployment lacks its capability source closure"
            )
        required: dict[str, str] = {}
        for index, row in enumerate(required_rows):
            if not isinstance(row, Mapping) or set(row) != {
                "relative_path",
                "sha256",
            }:
                raise ProcessBrokerEpisodeFactoryError(
                    "live capability source row is malformed"
                )
            relative = _canonical_relative_python_path(
                row.get("relative_path"),
                field_name=f"live capability source[{index}].relative_path",
            )
            digest = row.get("sha256")
            if not _is_sha256(digest) or relative in required:
                raise ProcessBrokerEpisodeFactoryError(
                    "live capability source closure is duplicated or malformed"
                )
            required[relative] = digest
            _validated_repository_source(
                self.repository_root,
                SourceAttestedPython(relative, digest),
                field_name=f"live capability source[{index}]",
            )
        for relative, digest in required.items():
            if relative in fixed and fixed[relative] != digest:
                raise ProcessBrokerEpisodeFactoryError(
                    "live capability closure conflicts with fixed broker source"
                )
        expected_declared = {
            relative: digest
            for relative, digest in required.items()
            if relative not in fixed
        }
        if declared != expected_declared:
            raise ProcessBrokerEpisodeFactoryError(
                "declared child source closure differs from the exact frozen "
                "live capability closure"
            )

    def revalidate(self) -> ValidatedPC01LiveDeployment:
        """Reopen all mutable external bytes immediately before one launch."""

        if canonical_json_bytes(self.timeout_expected_authority.to_dict()) != (
            self._timeout_authority_json
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "measured-timeout authority changed after descriptor construction"
            )
        try:
            binding = json.loads(self._live_binding_json)
            deployment = validate_pc01_live_deployment_binding(
                binding,
                artifact_root=self._live_artifact_root,
                repository_root=self.repository_root,
            )
            calibration = load_authority_bound_timeout_calibration(
                self.timeout_expected_authority
            )
        except (OSError, SchemaError, TypeError, ValueError) as exc:
            raise ProcessBrokerEpisodeFactoryError(
                "external deployment or timeout bytes changed before launch"
            ) from exc
        if (
            canonical_json_bytes(deployment.binding) != self._live_binding_json
            or sha256_json(deployment.manifest) != self._live_manifest_sha256
            or timeout_binding_sha256(measured_timeout_binding(calibration))
            != self._timeout_binding_sha256
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "external deployment or measured timeout identity changed"
            )
        self._validate_live_source_roles(deployment)
        self._validate_source_closure(deployment)
        return deployment

    def backend_dependency_paths(self) -> tuple[str, ...]:
        """Return a stable, duplicate-free child import closure."""

        paths = {
            source.source_relative_path for source in self._declared_sources()
        }
        paths.discard(PROCESS_BROKER_WEBARENA_BACKEND_SOURCE)
        return tuple(sorted(paths))


class _ProcessBrokerEpisodeLifecycle:
    """Single owner for authenticated shutdown and idempotent cleanup."""

    __slots__ = (
        "_abort_receipt",
        "_adapter",
        "_broker",
        "_cleaned",
        "_cleanup_receipt",
        "_lock",
        "_task_id",
    )

    def __init__(
        self,
        *,
        broker: ProcessIsolatedBroker,
        adapter: ProcessBrokerWebArenaEnvironmentAdapter,
        task_id: str,
    ) -> None:
        self._broker = broker
        self._adapter = adapter
        self._task_id = task_id
        self._lock = RLock()
        self._cleaned = False
        self._cleanup_receipt: Mapping[str, Any] | None = None
        self._abort_receipt: BrowserGymEpisodeAbortReceipt | None = None

    def _stop_authenticated(self) -> Mapping[str, Any]:
        try:
            self._broker.stop()
        except BaseException:
            try:
                self._broker.stop(force=True)
            except BaseException:
                pass
            raise
        if not self._broker.cleaned:
            raise ProcessBrokerEpisodeFactoryError(
                "process broker did not complete cleanup"
            )
        try:
            receipt_value = self._broker.cleanup_receipt.to_dict()
            detached = json.loads(canonical_json_bytes(receipt_value))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProcessBrokerEpisodeFactoryError(
                "authenticated process broker cleanup receipt is unavailable"
            ) from exc
        if not isinstance(detached, dict) or not detached:
            raise ProcessBrokerEpisodeFactoryError(
                "authenticated process broker cleanup receipt is malformed"
            )
        self._cleanup_receipt = MappingProxyType(detached)
        self._cleaned = True
        return self._cleanup_receipt

    def cleanup_episode(self) -> Mapping[str, Any]:
        with self._lock:
            if self._cleaned:
                if self._cleanup_receipt is None:  # pragma: no cover - invariant
                    raise ProcessBrokerEpisodeFactoryError(
                        "process cleanup state lacks its authenticated receipt"
                    )
                return self._cleanup_receipt
            return self._stop_authenticated()

    def abort_episode(
        self,
        episode_id: str,
        task_id: str,
    ) -> BrowserGymEpisodeAbortReceipt:
        with self._lock:
            if (
                type(episode_id) is not str
                or not episode_id.strip()
                or type(task_id) is not str
                or task_id != self._task_id
            ):
                raise ProcessBrokerEpisodeFactoryError(
                    "process broker abort identity differs from the episode"
                )
            if self._abort_receipt is not None:
                if (
                    self._abort_receipt.episode_id != episode_id
                    or self._abort_receipt.task_id != task_id
                ):
                    raise ProcessBrokerEpisodeFactoryError(
                        "process broker abort was repeated with another identity"
                    )
                return self._abort_receipt

            browser_reset = self._adapter.reset_state_receipt() is not None
            already_cleaned = self._cleaned
            if not self._cleaned:
                self._stop_authenticated()

            if not browser_reset:
                receipt = BrowserGymEpisodeAbortReceipt(
                    episode_id=episode_id,
                    task_id=task_id,
                    outcome="NO_BROWSER_CREATED",
                    underlying_browser_created=False,
                    underlying_browser_close_called=False,
                )
            elif already_cleaned:
                receipt = BrowserGymEpisodeAbortReceipt(
                    episode_id=episode_id,
                    task_id=task_id,
                    outcome="ALREADY_CLEANED_BY_SEALED_BROKER",
                    underlying_browser_created=True,
                    underlying_browser_close_called=True,
                )
            else:
                try:
                    cleanup_receipt = self._broker.cleanup_receipt
                except RuntimeError as exc:
                    raise ProcessBrokerEpisodeFactoryError(
                        "authenticated broker abort lacks its cleanup receipt"
                    ) from exc
                abort_sha256 = sha256_json(
                    {
                        "episode_id": episode_id,
                        "task_id": task_id,
                        "cleanup_receipt": cleanup_receipt.to_dict(),
                    }
                )
                receipt = BrowserGymEpisodeAbortReceipt(
                    episode_id=episode_id,
                    task_id=task_id,
                    outcome="BROKER_ABORTED",
                    underlying_browser_created=True,
                    underlying_browser_close_called=True,
                    broker_abort_receipt_sha256=abort_sha256,
                )
            self._abort_receipt = receipt
            return receipt


def _stop_after_construction_failure(broker: ProcessIsolatedBroker) -> None:
    """Best-effort reap without replacing the fail-closed construction error."""

    try:
        broker.stop()
    except BaseException:
        try:
            broker.stop(force=True)
        except BaseException:
            pass


@dataclass(frozen=True, slots=True)
class CredentialFreeProcessIsolatedWebArenaEpisodeFactory:
    """Build one child-owned WebArena episode from explicit frozen inputs."""

    descriptor: ProcessBrokerLiveDeploymentDescriptor

    def __post_init__(self) -> None:
        if type(self.descriptor) is not ProcessBrokerLiveDeploymentDescriptor:
            raise ProcessBrokerEpisodeFactoryError(
                "process episode factory requires the exact descriptor type"
            )

    def __call__(
        self,
        task: TaskSpecification,
        episode_runtime_dir: Path,
        sealed_stream_target: SealedVerifierStreamTarget,
    ) -> ProcessIsolatedWebArenaEpisodeBinding:
        if type(task) is not TaskSpecification:
            raise ProcessBrokerEpisodeFactoryError(
                "process episode factory requires exact TaskSpecification"
            )
        if (
            task.benchmark_id.casefold() != "webarena"
            or task.development_partition is not True
            or task.destructive_actions_allowed is not False
            or task.runtime_start_state is None
            or task.metadata.get("task_partition") != "normal"
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "process episode task is outside the registered WebArena pilot"
            )
        if type(sealed_stream_target) is not SealedVerifierStreamTarget:
            raise ProcessBrokerEpisodeFactoryError(
                "process episode requires the exact sealed stream target"
            )
        if (
            sealed_stream_target.task_id != task.task_id
            or sealed_stream_target.matched_seed != 42
            or sealed_stream_target.initial_event_count != 0
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "sealed stream target differs from the PC-01 episode"
            )
        runtime_dir = Path(episode_runtime_dir).absolute()
        if (
            not runtime_dir.is_dir()
            or runtime_dir.is_symlink()
            or runtime_dir.resolve(strict=True) != runtime_dir
        ):
            raise ProcessBrokerEpisodeFactoryError(
                "episode runtime directory must be canonical and pre-existing"
            )
        self.descriptor.revalidate()
        screenshot_root = runtime_dir / "screenshots"
        try:
            screenshot_root.mkdir(mode=0o700, exist_ok=False)
            screenshot_root = validated_policy_screenshot_root(screenshot_root)
        except Exception as exc:
            raise ProcessBrokerEpisodeFactoryError(
                "episode screenshot root must be fresh, canonical, and child-specific"
            ) from exc
        if screenshot_root is None or any(screenshot_root.iterdir()):
            raise ProcessBrokerEpisodeFactoryError(
                "episode screenshot root was not created empty"
            )
        config = {
            "schema_version": (
                PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION
            ),
            "adapter_factory_entrypoint": self.descriptor.adapter_factory.entrypoint,
            "adapter_factory_source_relative_path": (
                self.descriptor.adapter_factory.source_relative_path
            ),
            "adapter_factory_source_sha256": (
                self.descriptor.adapter_factory.source_sha256
            ),
            "sealed_transition_callback_entrypoint": (
                self.descriptor.sealed_transition_callback.entrypoint
            ),
            "sealed_transition_callback_source_relative_path": (
                self.descriptor.sealed_transition_callback.source_relative_path
            ),
            "sealed_transition_callback_source_sha256": (
                self.descriptor.sealed_transition_callback.source_sha256
            ),
            "sealed_finalizer_entrypoint": self.descriptor.sealed_finalizer.entrypoint,
            "sealed_finalizer_source_relative_path": (
                self.descriptor.sealed_finalizer.source_relative_path
            ),
            "sealed_finalizer_source_sha256": (
                self.descriptor.sealed_finalizer.source_sha256
            ),
            "task_specification": task.to_dict(),
            "episode_runtime_dir": str(runtime_dir),
            "sealed_stream_target": sealed_stream_target.to_dict(),
        }
        broker = ProcessIsolatedBroker(
            repository_root=self.descriptor.repository_root,
            backend_entrypoint=PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT,
            backend_source_relative_path=PROCESS_BROKER_WEBARENA_BACKEND_SOURCE,
            backend_dependency_source_relative_paths=(
                self.descriptor.backend_dependency_paths()
            ),
            sealed_backend_config=config,
            policy_screenshot_root=screenshot_root,
            timeout_calibration_artifact_path=(
                self.descriptor.timeout_calibration_artifact_path
            ),
            timeout_calibration_expected_authority=(
                self.descriptor.timeout_expected_authority
            ),
            execution_scope=PROCESS_BROKER_PILOT_EVALUATION_SCOPE,
            require_sealed_finalization=True,
        )
        try:
            receipt = broker.start()
            source_rows = {
                str(row["relative_path"]): str(row["sha256"])
                for row in receipt.source_files
            }
            expected_source_rows = {
                relative: sha256_file(self.descriptor.repository_root / relative)
                for relative in PROCESS_BROKER_SOURCE_PATHS
            }
            for source in self.descriptor._declared_sources():
                expected_source_rows[source.source_relative_path] = (
                    source.source_sha256
                )
            if (
                len(source_rows) != len(receipt.source_files)
                or source_rows != expected_source_rows
            ):
                raise ProcessBrokerEpisodeFactoryError(
                    "started broker source receipt differs from the exact descriptor closure"
                )
            if (
                receipt.execution_scope != PROCESS_BROKER_PILOT_EVALUATION_SCOPE
                or receipt.immutable_timeout_authority_bundle_validated is not True
                or receipt.measured_ipc_timeout_calibration_complete is not True
                or receipt.ipc_timeout_binding_sha256
                != self.descriptor.measured_timeout_binding_sha256
                or receipt.sealed_finalization_capability_available is not True
                or receipt.sealed_finalization_required is not True
                or receipt.child_owned_sealed_sink is not True
                or receipt.runtime_adapter_sealed_capability_free is not True
            ):
                raise ProcessBrokerEpisodeFactoryError(
                    "started broker receipt lacks the registered pilot isolation profile"
                )
            runtime_client = broker.runtime_client()
            finalization_client = broker.finalization_client()
            adapter = ProcessBrokerWebArenaEnvironmentAdapter(
                client=runtime_client,
                benchmark_version=task.benchmark_version,
            )
            finalization = ProcessIsolatedEpisodeFinalizationBinding(
                client=finalization_client,
                episode_runtime_dir=runtime_dir,
            )
            lifecycle = _ProcessBrokerEpisodeLifecycle(
                broker=broker,
                adapter=adapter,
                task_id=task.task_id,
            )
            public_identity = MappingProxyType(
                self.descriptor.public_identity()
            )
            return ProcessIsolatedWebArenaEpisodeBinding(
                benchmark_version=task.benchmark_version,
                environment_adapter=adapter,
                finalize_episode_evidence=finalization.finalize_episode_evidence,
                finalization_receipt=lambda: (
                    finalization.finalization_receipt.to_dict()
                ),
                abort_episode=lifecycle.abort_episode,
                cleanup_episode=lifecycle.cleanup_episode,
                broker_receipt=receipt.to_dict(),
                live_deployment_binding_sha256=(
                    self.descriptor.live_deployment_binding_sha256
                ),
                live_deployment_manifest_sha256=(
                    self.descriptor.live_deployment_manifest_sha256
                ),
                measured_timeout_binding_sha256=(
                    self.descriptor.measured_timeout_binding_sha256
                ),
                episode_factory_public_identity=public_identity,
                episode_factory_descriptor_sha256=(
                    self.descriptor.descriptor_sha256
                ),
                frozen=True,
                oracle_labels_exposed_to_runtime=False,
            )
        except BaseException:
            _stop_after_construction_failure(broker)
            raise


# Concise aliases for operator integrations without weakening exact-type checks.
ProcessBrokerEpisodeFactory = CredentialFreeProcessIsolatedWebArenaEpisodeFactory
ProcessBrokerEpisodeDescriptor = ProcessBrokerLiveDeploymentDescriptor


__all__ = [
    "CredentialFreeProcessIsolatedWebArenaEpisodeFactory",
    "PROCESS_BROKER_EPISODE_FACTORY_CLAIM_SCOPE",
    "PROCESS_BROKER_EPISODE_FACTORY_SCHEMA_VERSION",
    "ProcessBrokerEpisodeDescriptor",
    "ProcessBrokerEpisodeFactory",
    "ProcessBrokerEpisodeFactoryError",
    "ProcessBrokerLiveDeploymentDescriptor",
    "SourceAttestedEntrypoint",
    "SourceAttestedPython",
]
