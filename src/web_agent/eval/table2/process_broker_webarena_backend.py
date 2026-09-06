"""Source-bound WebArena adapter backend for the strict process broker.

The generic process-broker worker imports :func:`create_backend` in its isolated
child.  This module then imports one explicitly named, hash-bound repository
factory which receives only the oracle-blind ``TaskSpecification`` already
available to the browser runtime and its episode output directory.  The factory
must construct the complete WebArena ``EnvironmentAdapter`` -- including any
sealed terminal callback -- inside the child.  No evaluator object, oracle
record, reward, reference trajectory, or final label is accepted in the backend
configuration or returned over IPC.

This is repository-side integration machinery, not deployment authority.  The
current worker is still a same-UID child, scalar value provenance is not
externally attested, and the canonical evaluation hard stop remains mandatory.
No concrete live BrowserGym binding plus sealed-evaluator factory or credential/
configuration capability is registered by this module; that deployment closure
remains pending.

An actual deployment must register every repository-local source imported by
its adapter factory through ``backend_dependency_source_relative_paths`` and
must separately satisfy the external receipt/trust-anchor gates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
from typing import Any

from web_agent.benchmarks.base import AdapterExecution, EnvironmentAdapter
from web_agent.runtime.contracts import (
    ConcreteAction,
    EpisodeSummary,
    ExecutionResult,
    ExecutionStatus,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    TaskSpecification,
    VerifierReceiptBinding,
    detached_record_copy,
)
from web_agent.runtime.state_reset import WebArenaResetStateReceipt

from .common import SchemaError, canonical_json_bytes, sha256_file
from .process_broker_protocol import (
    PROCESS_BROKER_INFRASTRUCTURE_INVALID_SCHEMA_VERSION,
    ProcessBrokerInfrastructureInvalidMixin,
    ProcessBrokerInfrastructureInvalidResponse,
    ProcessBrokerProtocolError,
    forbidden_runtime_fields,
    validate_runtime_infrastructure_invalid,
    validate_runtime_request_payload,
)
from .process_broker_runtime import ProcessIsolatedRuntimeClient
from .sealed_verifier import (
    SealedVerifierSink,
    SealedVerifierStreamTarget,
    SealedVerifierWriter,
)


PROCESS_BROKER_WEBARENA_BACKEND_SCHEMA_VERSION = (
    "table2-process-broker-webarena-backend-v1"
)
PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION = (
    "table2-process-broker-webarena-finalizing-backend-v3"
)
PROCESS_BROKER_WEBARENA_BACKEND_ENTRYPOINT = (
    "web_agent.eval.table2.process_broker_webarena_backend:create_backend"
)
PROCESS_BROKER_WEBARENA_BACKEND_SOURCE = (
    "src/web_agent/eval/table2/process_broker_webarena_backend.py"
)
PROCESS_BROKER_CHILD_CLEANUP_SCHEMA_VERSION = (
    "table2-process-broker-child-cleanup-result-v1"
)
PROCESS_BROKER_CHILD_CLEANUP_RECORD_TYPE = "ProcessBrokerChildCleanupResult"
PROCESS_BROKER_CHILD_CLEANUP_DISPOSITIONS = frozenset(
    {
        "RUNTIME_CLOSE_ACKNOWLEDGED",
        "CONTROL_ABORT_COMPLETED",
        "NO_BROWSER_CREATED",
    }
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_SCHEMA_VERSION = (
    "table2-process-broker-sealed-callback-state-guard-v1"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_RECORD_TYPE = (
    "ProcessBrokerSealedCallbackStateGuard"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION = (
    "table2-process-broker-sealed-callback-state-guard-sidecar-identity-v1"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE = (
    "ProcessBrokerSealedCallbackStateGuardSidecarIdentity"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_SIDECAR_FILENAME = (
    "sealed_callback_state_guard.child.jsonl"
)
# Captured when the parent-side runtime bridge is imported. The broker compares
# this with its launch receipt so later file replacement cannot pair stale
# in-memory bridge code with newly hashed child bytes.
PROCESS_BROKER_IMPORT_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

_BACKEND_CONFIG_FIELDS = {
    "schema_version",
    "adapter_factory_entrypoint",
    "adapter_factory_source_relative_path",
    "adapter_factory_source_sha256",
    "task_specification",
    "episode_runtime_dir",
}
_FINALIZING_BACKEND_CONFIG_FIELDS = _BACKEND_CONFIG_FIELDS | {
    "sealed_transition_callback_entrypoint",
    "sealed_transition_callback_source_relative_path",
    "sealed_transition_callback_source_sha256",
    "sealed_finalizer_entrypoint",
    "sealed_finalizer_source_relative_path",
    "sealed_finalizer_source_sha256",
    "sealed_stream_target",
}
_RUNTIME_TASK_METADATA_FIELDS = {
    "task_partition",
    "upstream_index",
    "benchmark_task_id",
    "source_content_sha256",
}
WebArenaEpisodeAdapterFactory = Callable[..., EnvironmentAdapter]
WebArenaEpisodeSealedFinalizer = Callable[..., OpaqueTerminalSignal]
WebArenaSealedTransitionEvaluator = Callable[..., OpaqueTerminalSignal]

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


def _lowercase_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProcessBrokerProtocolError(f"{label} must be lowercase SHA-256")
    return value


def _detached_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessBrokerProtocolError(
            "WebArena broker backend config is not an object"
        )
    try:
        detached = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend config is not canonical JSON"
        ) from exc
    if not isinstance(detached, dict):  # pragma: no cover - Mapping encodes as object
        raise ProcessBrokerProtocolError(
            "WebArena broker backend config is not an object"
        )
    schema_version = detached.get("schema_version")
    expected_fields = (
        _FINALIZING_BACKEND_CONFIG_FIELDS
        if schema_version
        == PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION
        else _BACKEND_CONFIG_FIELDS
    )
    if set(detached) != expected_fields:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend config fields differ from schema"
        )
    forbidden = forbidden_runtime_fields(detached)
    if forbidden:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend config contains evaluator/oracle aliases"
        )
    if schema_version not in {
        PROCESS_BROKER_WEBARENA_BACKEND_SCHEMA_VERSION,
        PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION,
    }:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend config version differs"
        )
    return detached


def _validated_runtime_task(value: object) -> TaskSpecification:
    if not isinstance(value, Mapping):
        raise ProcessBrokerProtocolError(
            "WebArena broker backend requires a typed task specification"
        )
    try:
        task = TaskSpecification.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend task specification is invalid"
        ) from exc
    if type(task) is not TaskSpecification or canonical_json_bytes(task.to_dict()) != (
        canonical_json_bytes(value)
    ):
        raise ProcessBrokerProtocolError(
            "WebArena broker backend task specification is not canonical"
        )
    if (
        task.benchmark_id.casefold() != "webarena"
        or task.runtime_start_state is None
        or task.development_partition is not True
        or task.destructive_actions_allowed is not False
    ):
        raise ProcessBrokerProtocolError(
            "WebArena broker backend task is outside the registered pilot runtime"
        )
    metadata = task.metadata
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != _RUNTIME_TASK_METADATA_FIELDS
    ):
        raise ProcessBrokerProtocolError(
            "WebArena broker backend task metadata differs from oracle-blind projection"
        )
    if metadata.get("task_partition") != "normal":
        raise ProcessBrokerProtocolError(
            "WebArena broker backend accepts ordinary tasks only"
        )
    upstream_index = metadata.get("upstream_index")
    benchmark_task_id = metadata.get("benchmark_task_id")
    if type(upstream_index) is not int or upstream_index < 0:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend task upstream index is invalid"
        )
    if type(benchmark_task_id) is not str or not benchmark_task_id.strip():
        raise ProcessBrokerProtocolError(
            "WebArena broker backend benchmark task identity is invalid"
        )
    _lowercase_sha256(
        metadata.get("source_content_sha256"),
        label="WebArena broker backend task source content",
    )
    return task


def _validated_episode_runtime_dir(value: object) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise ProcessBrokerProtocolError(
            "WebArena broker backend episode runtime directory is invalid"
        )
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ProcessBrokerProtocolError(
            "WebArena broker backend episode runtime directory must be absolute"
        )
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProcessBrokerProtocolError(
            "WebArena broker backend episode runtime directory is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != candidate
    ):
        raise ProcessBrokerProtocolError(
            "WebArena broker backend episode runtime directory is not canonical"
        )
    return resolved


def _validated_repository_factory_source(
    *, repository_root: Path, relative_path: object
) -> tuple[Path, str]:
    if (
        type(relative_path) is not str
        or not relative_path
        or relative_path != relative_path.strip()
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source path is invalid"
        )
    lexical = Path(relative_path)
    if (
        lexical.is_absolute()
        or ".." in lexical.parts
        or "." in lexical.parts
        or lexical.suffix != ".py"
        or lexical.as_posix() != relative_path
    ):
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source path is invalid"
        )
    candidate = repository_root
    try:
        for part in lexical.parts:
            candidate = candidate / part
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProcessBrokerProtocolError(
                    "WebArena adapter factory source contains a symlink component"
                )
    except OSError as exc:
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source is unavailable"
        ) from exc
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source escaped repository"
        ) from exc
    if not stat.S_ISREG(resolved.lstat().st_mode):
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source is not a regular file"
        )
    return resolved, lexical.as_posix()


def _validated_entrypoint_parts(
    value: object,
    *,
    label: str,
) -> tuple[str, str]:
    if type(value) is not str:
        raise ProcessBrokerProtocolError(f"{label} entrypoint is invalid")
    module_name, separator, attribute_name = value.partition(":")
    if (
        separator != ":"
        or not module_name
        or any(not part.isidentifier() for part in module_name.split("."))
        or not attribute_name.isidentifier()
    ):
        raise ProcessBrokerProtocolError(f"{label} entrypoint is malformed")
    return module_name, attribute_name


def _validate_preimport_function_contract(
    source: Path,
    *,
    attribute_name: str,
    keyword_parameters: tuple[str, ...],
    label: str,
) -> None:
    """Reject an absent or impossible child entrypoint before module import.

    Source hashing alone does not prove that ``module:attribute`` exists or has
    the call shape used by the worker.  Parsing the already-attested bytes first
    also prevents a malformed module's top-level code from running merely to
    discover that its advertised attribute was a class, alias, or wrong-shaped
    function.
    """

    try:
        tree = ast.parse(source.read_bytes(), filename=str(source))
    except (OSError, SyntaxError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            f"{label} source cannot be parsed before import"
        ) from exc
    declarations = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == attribute_name
    ]
    if len(declarations) != 1 or not isinstance(declarations[0], ast.FunctionDef):
        raise ProcessBrokerProtocolError(
            f"{label} entrypoint is not one top-level synchronous function"
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
        raise ProcessBrokerProtocolError(
            f"{label} entrypoint signature differs from the child contract"
        )


def _validate_loaded_function_contract(
    callback: object,
    *,
    module_name: str,
    attribute_name: str,
    keyword_parameters: tuple[str, ...],
    label: str,
) -> Callable[..., Any]:
    if (
        not inspect.isfunction(callback)
        or getattr(callback, "__module__", None) != module_name
        or getattr(callback, "__name__", None) != attribute_name
    ):
        raise ProcessBrokerProtocolError(
            f"{label} entrypoint changed after pre-import validation"
        )
    try:
        parameters = tuple(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError) as exc:
        raise ProcessBrokerProtocolError(
            f"{label} entrypoint signature is unavailable"
        ) from exc
    if (
        tuple(item.name for item in parameters) != keyword_parameters
        or any(item.kind is not inspect.Parameter.KEYWORD_ONLY for item in parameters)
        or any(item.default is not inspect.Parameter.empty for item in parameters)
    ):
        raise ProcessBrokerProtocolError(
            f"{label} entrypoint signature differs from the child contract"
        )
    return callback


def _load_source_attested_function(
    config: Mapping[str, Any],
    *,
    field_prefix: str,
    label: str,
    keyword_parameters: tuple[str, ...],
) -> Callable[..., Any]:
    entrypoint = config.get(f"{field_prefix}_entrypoint")
    module_name, attribute_name = _validated_entrypoint_parts(
        entrypoint,
        label=label,
    )
    repository_root = Path(__file__).resolve().parents[4]
    source, source_relative = _validated_repository_factory_source(
        repository_root=repository_root,
        relative_path=config.get(f"{field_prefix}_source_relative_path"),
    )
    expected_module_path = Path(*module_name.split(".")).with_suffix(".py")
    if Path(source_relative) not in {
        expected_module_path,
        Path("src") / expected_module_path,
    }:
        raise ProcessBrokerProtocolError(f"{label} source differs from entrypoint")
    expected_sha256 = _lowercase_sha256(
        config.get(f"{field_prefix}_source_sha256"),
        label=f"{label} source",
    )
    if sha256_file(source) != expected_sha256:
        raise ProcessBrokerProtocolError(f"{label} source hash differs")
    _validate_preimport_function_contract(
        source,
        attribute_name=attribute_name,
        keyword_parameters=keyword_parameters,
        label=label,
    )
    module = importlib.import_module(module_name)
    loaded_source = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_source != source or sha256_file(loaded_source) != expected_sha256:
        raise ProcessBrokerProtocolError(f"loaded {label} identity differs")
    return _validate_loaded_function_contract(
        getattr(module, attribute_name, None),
        module_name=module_name,
        attribute_name=attribute_name,
        keyword_parameters=keyword_parameters,
        label=label,
    )


def _load_source_attested_adapter_factory(
    config: Mapping[str, Any],
) -> WebArenaEpisodeAdapterFactory:
    return _load_source_attested_function(
        config,
        field_prefix="adapter_factory",
        label="WebArena adapter factory",
        keyword_parameters=_ADAPTER_FACTORY_KEYWORD_PARAMETERS,
    )


def _load_source_attested_finalizer(
    config: Mapping[str, Any],
) -> WebArenaEpisodeSealedFinalizer:
    """Load only the explicitly hash-bound child-side finalizer."""

    return _load_source_attested_function(
        config,
        field_prefix="sealed_finalizer",
        label="WebArena sealed finalizer",
        keyword_parameters=_SEALED_FINALIZER_KEYWORD_PARAMETERS,
    )


def _load_source_attested_transition_evaluator(
    config: Mapping[str, Any],
) -> WebArenaSealedTransitionEvaluator:
    return _load_source_attested_function(
        config,
        field_prefix="sealed_transition_callback",
        label="WebArena sealed transition callback",
        keyword_parameters=_SEALED_TRANSITION_KEYWORD_PARAMETERS,
    )


def _adapter_environment_state_sha256(
    adapter: EnvironmentAdapter,
    *,
    context: str,
) -> str:
    callback = getattr(adapter, "process_broker_environment_state_sha256", None)
    if not callable(callback):
        raise ProcessBrokerProtocolError(
            f"{context} lacks the source-attested environment-state digest"
        )
    try:
        value = callback()
    except BaseException as exc:
        raise ProcessBrokerProtocolError(
            f"{context} environment-state digest failed"
        ) from exc
    return _lowercase_sha256(value, label=f"{context} environment-state digest")


class _SealedCallbackGuardSidecar:
    """Child/backend-owned append-only sealed-callback state evidence."""

    __slots__ = (
        "_closed",
        "_count",
        "_descriptor",
        "_file_identity",
        "_path",
        "_tail_sha256",
    )

    def __init__(self, runtime_dir: Path) -> None:
        path = runtime_dir / PROCESS_BROKER_SEALED_CALLBACK_GUARD_SIDECAR_FILENAME
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise ProcessBrokerProtocolError(
                "sealed-callback guard sidecar must be a fresh child-owned file"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise ProcessBrokerProtocolError(
                "sealed-callback guard sidecar must be a single-link regular file"
            )
        self._path = path
        self._descriptor = descriptor
        self._file_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
        )
        self._count = 0
        self._tail_sha256: str | None = None
        self._closed = False

    def _validate_open(self) -> None:
        if self._closed:
            raise ProcessBrokerProtocolError(
                "sealed-callback guard sidecar is already closed"
            )
        try:
            descriptor_metadata = os.fstat(self._descriptor)
            path_metadata = self._path.lstat()
        except OSError as exc:
            raise ProcessBrokerProtocolError(
                "sealed-callback guard sidecar identity is unavailable"
            ) from exc
        descriptor_identity = (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
            descriptor_metadata.st_mode,
            descriptor_metadata.st_nlink,
        )
        path_identity = (
            path_metadata.st_dev,
            path_metadata.st_ino,
            path_metadata.st_mode,
            path_metadata.st_nlink,
        )
        if (
            descriptor_identity != self._file_identity
            or path_identity != self._file_identity
            or not stat.S_ISREG(descriptor_metadata.st_mode)
            or descriptor_metadata.st_nlink != 1
        ):
            raise ProcessBrokerProtocolError(
                "sealed-callback guard sidecar pathname changed"
            )

    def append(
        self,
        *,
        callback_kind: str,
        environment_state_sha256_before: str,
        environment_state_sha256_after: str,
    ) -> None:
        self._validate_open()
        if callback_kind not in {"sealed_transition", "sealed_finalizer"}:
            raise ProcessBrokerProtocolError(
                "sealed-callback guard kind is unregistered"
            )
        before = _lowercase_sha256(
            environment_state_sha256_before,
            label="sealed-callback guard before state",
        )
        after = _lowercase_sha256(
            environment_state_sha256_after,
            label="sealed-callback guard after state",
        )
        if before != after:
            raise ProcessBrokerProtocolError(
                "mutating sealed callback cannot be recorded as unchanged"
            )
        index = self._count + 1
        body = {
            "schema_version": PROCESS_BROKER_SEALED_CALLBACK_GUARD_SCHEMA_VERSION,
            "record_type": PROCESS_BROKER_SEALED_CALLBACK_GUARD_RECORD_TYPE,
            "callback_guard_index": index,
            "previous_record_sha256": self._tail_sha256 or ("0" * 64),
            "callback_kind": callback_kind,
            "environment_state_sha256_before": before,
            "environment_state_sha256_after": after,
            "environment_state_unchanged": True,
        }
        record_sha256 = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        encoded = canonical_json_bytes(
            {**body, "record_sha256": record_sha256}
        ) + b"\n"
        view = memoryview(encoded)
        while view:
            written = os.write(self._descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file write contract
                raise ProcessBrokerProtocolError(
                    "sealed-callback guard sidecar append made no progress"
                )
            view = view[written:]
        os.fsync(self._descriptor)
        self._count = index
        self._tail_sha256 = record_sha256

    def identity(self) -> dict[str, Any]:
        self._validate_open()
        os.fsync(self._descriptor)
        current_offset = os.lseek(self._descriptor, 0, os.SEEK_CUR)
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        try:
            while True:
                block = os.read(self._descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        finally:
            os.lseek(self._descriptor, current_offset, os.SEEK_SET)
        self._validate_open()
        return {
            "schema_version": (
                PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION
            ),
            "record_type": PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE,
            "relative_path": PROCESS_BROKER_SEALED_CALLBACK_GUARD_SIDECAR_FILENAME,
            "record_count": self._count,
            "tail_sha256": self._tail_sha256,
            "content_sha256": digest.hexdigest(),
        }

    def close(self) -> None:
        if self._closed:
            return None
        self._validate_open()
        os.fsync(self._descriptor)
        os.close(self._descriptor)
        self._closed = True
        return None


def _call_state_preserving_sealed_callback(
    adapter: EnvironmentAdapter,
    callback: Callable[[], Any],
    *,
    callback_kind: str,
    context: str,
    evidence_sidecar: _SealedCallbackGuardSidecar,
) -> Any:
    """Require a sealed callback to preserve the child-owned browser state."""

    before = _adapter_environment_state_sha256(adapter, context=context)
    callback_error: BaseException | None = None
    result: object | None = None
    try:
        result = callback()
    except BaseException as exc:
        callback_error = exc
    try:
        after = _adapter_environment_state_sha256(adapter, context=context)
    except BaseException as digest_error:
        if callback_error is not None:
            raise digest_error from callback_error
        raise
    if after != before:
        error = ProcessBrokerProtocolError(
            f"{context} mutated the child-owned environment state"
        )
        if callback_error is not None:
            raise error from callback_error
        raise error
    try:
        evidence_sidecar.append(
            callback_kind=callback_kind,
            environment_state_sha256_before=before,
            environment_state_sha256_after=after,
        )
    except BaseException as evidence_error:
        if callback_error is not None:
            raise ProcessBrokerProtocolError(
                f"{context} state-guard evidence append failed"
            ) from evidence_error
        raise ProcessBrokerProtocolError(
            f"{context} state-guard evidence append failed"
        ) from evidence_error
    if callback_error is not None:
        raise callback_error
    return result


def _manual_rescue_sidecar_identity(
    adapter: EnvironmentAdapter,
    *,
    required: bool,
) -> tuple[int, str | None]:
    callback = getattr(
        adapter,
        "process_broker_manual_rescue_sidecar_identity",
        None,
    )
    if not callable(callback):
        if required:
            raise ProcessBrokerProtocolError(
                "finalizing WebArena adapter lacks child-owned manual-rescue evidence"
            )
        return 0, None
    try:
        value = callback()
    except BaseException as exc:
        raise ProcessBrokerProtocolError(
            "child-owned manual-rescue sidecar identity failed"
        ) from exc
    if not isinstance(value, Mapping):
        raise ProcessBrokerProtocolError(
            "child-owned manual-rescue sidecar identity is malformed"
        )
    if set(value) != {
        "schema_version",
        "record_type",
        "relative_path",
        "record_count",
        "tail_sha256",
        "content_sha256",
    }:
        raise ProcessBrokerProtocolError(
            "child-owned manual-rescue sidecar identity fields differ"
        )
    if (
        value.get("schema_version")
        != "table2-process-broker-manual-rescue-sidecar-identity-v1"
        or value.get("record_type")
        != "ProcessBrokerManualRescueSidecarIdentity"
        or value.get("relative_path") != "manual_rescue_guard.child.jsonl"
    ):
        raise ProcessBrokerProtocolError(
            "child-owned manual-rescue sidecar identity version differs"
        )
    count = value.get("record_count")
    tail = value.get("tail_sha256")
    if type(count) is not int or count < 0:
        raise ProcessBrokerProtocolError(
            "child-owned manual-rescue sidecar count is invalid"
        )
    if (tail is None) is not (count == 0):
        raise ProcessBrokerProtocolError(
            "child-owned manual-rescue sidecar tail/count disagree"
        )
    if tail is not None:
        _lowercase_sha256(tail, label="child-owned manual-rescue sidecar tail")
    _lowercase_sha256(
        value.get("content_sha256"),
        label="child-owned manual-rescue sidecar content",
    )
    return count, tail


class WebArenaProcessBrokerBackend:
    """Child-owned adapter translated to the five strict broker operations."""

    __slots__ = (
        "_adapter",
        "_closed",
        "_episode_id",
        "_finalized",
        "_finalizer",
        "_reset_receipt",
        "_runtime_closed",
        "_runtime_dir",
        "_sealed_callback_guard",
        "_sealed_sink",
        "_sealed_writer",
        "_task",
        "_transition_evaluator",
    )

    def __init__(
        self,
        *,
        task: TaskSpecification,
        adapter: EnvironmentAdapter,
        runtime_dir: Path | None = None,
        finalizer: WebArenaEpisodeSealedFinalizer | None = None,
        transition_evaluator: WebArenaSealedTransitionEvaluator | None = None,
        sealed_sink: SealedVerifierSink | None = None,
    ) -> None:
        if type(task) is not TaskSpecification:
            raise TypeError("WebArena process backend requires TaskSpecification")
        if not isinstance(adapter, EnvironmentAdapter):
            raise TypeError("WebArena process backend requires EnvironmentAdapter")
        if str(getattr(adapter, "benchmark_id", "")).casefold() != "webarena":
            raise ValueError("WebArena process backend received another benchmark")
        if getattr(adapter, "benchmark_version", None) != task.benchmark_version:
            raise ValueError("WebArena process backend adapter version differs")
        finalizing_components = (
            callable(finalizer),
            callable(transition_evaluator),
            type(sealed_sink) is SealedVerifierSink,
        )
        if any(finalizing_components) and not all(finalizing_components):
            raise ValueError(
                "WebArena process backend finalization components are incomplete"
            )
        if all(finalizing_components):
            if not callable(
                getattr(adapter, "process_broker_environment_state_sha256", None)
            ):
                raise ValueError(
                    "finalizing WebArena adapter lacks its source-attested state digest"
                )
            if not callable(
                getattr(
                    adapter,
                    "process_broker_manual_rescue_sidecar_identity",
                    None,
                )
            ):
                raise ValueError(
                    "finalizing WebArena adapter lacks its child-owned manual-rescue sidecar"
                )
        self._task = detached_record_copy(task)
        self._adapter = adapter
        self._episode_id: str | None = None
        self._reset_receipt: WebArenaResetStateReceipt | None = None
        self._runtime_dir = runtime_dir
        self._finalizer = finalizer
        self._transition_evaluator = transition_evaluator
        self._sealed_sink = sealed_sink
        self._sealed_callback_guard = (
            _SealedCallbackGuardSidecar(runtime_dir)
            if all(finalizing_components) and runtime_dir is not None
            else None
        )
        self._sealed_writer = (
            SealedVerifierWriter(sealed_sink)
            if type(sealed_sink) is SealedVerifierSink
            else None
        )
        self._runtime_closed = False
        self._finalized = False
        self._closed = False

    def _validated_request(
        self, operation: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        request = validate_runtime_request_payload(operation, payload)
        if request["task_id"] != self._task.task_id:
            raise ProcessBrokerProtocolError(
                "WebArena process backend task identity differs"
            )
        if self._episode_id is not None and request["episode_id"] != self._episode_id:
            raise ProcessBrokerProtocolError(
                "WebArena process backend episode identity differs"
            )
        if self._closed:
            raise ProcessBrokerProtocolError("WebArena process backend is closed")
        return request

    def _call_adapter(
        self,
        runtime_operation: str,
        request: Mapping[str, Any],
        callback: Callable[[], Any],
    ) -> Any:
        """Translate only the reviewed infrastructure-rerun exception."""

        try:
            return callback()
        except BaseException as error:
            if not isinstance(error, ProcessBrokerInfrastructureInvalidMixin):
                raise
            adapter_evidence = getattr(error, "adapter_evidence", None)
            if not isinstance(adapter_evidence, Mapping):
                raise ProcessBrokerProtocolError(
                    "WebArena infrastructure error lacks typed adapter evidence"
                ) from None
            error_adapter_id = getattr(error, "adapter_id", None)
            error_adapter_version = getattr(error, "adapter_version", None)
            error_operation = getattr(error, "operation", None)
            configured_adapter_id = getattr(self._adapter, "_adapter_id", None)
            configured_adapter_version = getattr(
                self._adapter,
                "_adapter_version",
                None,
            )
            if (
                type(configured_adapter_id) is not str
                or type(configured_adapter_version) is not str
                or error_adapter_id != configured_adapter_id
                or error_adapter_version != configured_adapter_version
            ):
                raise ProcessBrokerProtocolError(
                    "WebArena infrastructure error adapter identity differs"
                ) from None
            partial_execution = getattr(error, "adapter_execution", None)
            if error_operation == "step" and type(partial_execution) is not (
                AdapterExecution
            ):
                raise ProcessBrokerProtocolError(
                    "WebArena step infrastructure error lacks execution evidence"
                ) from None
            if partial_execution is not None and (
                runtime_operation != "runtime_execute"
                or error_operation != "step"
                or type(partial_execution) is not AdapterExecution
            ):
                raise ProcessBrokerProtocolError(
                    "WebArena infrastructure execution evidence is not causal"
                ) from None
            sanitized_value = {
                "schema_version": (
                    PROCESS_BROKER_INFRASTRUCTURE_INVALID_SCHEMA_VERSION
                ),
                "record_type": "InfrastructureInvalidError",
                "reason_code": getattr(error, "reason_code", None),
                "adapter_id": error_adapter_id,
                "adapter_version": error_adapter_version,
                "operation": error_operation,
                "adapter_evidence": {
                    field: adapter_evidence.get(field)
                    for field in (
                        "adapter_event_id",
                        "failure_class",
                        "diagnostic_sha256",
                        "retryable",
                    )
                },
            }
            if type(partial_execution) is AdapterExecution:
                sanitized_value["adapter_execution"] = partial_execution.to_dict()
            sanitized = validate_runtime_infrastructure_invalid(
                sanitized_value,
                request_operation=runtime_operation,
                episode_id=str(request["episode_id"]),
                request_payload=request,
            )
            raise ProcessBrokerInfrastructureInvalidResponse(
                sanitized,
                request_operation=runtime_operation,
                episode_id=str(request["episode_id"]),
                request_payload=request,
            ) from None

    @staticmethod
    def _observation_result(
        observation: object,
        *,
        episode_id: str,
        stage: ObservationStage,
        prior_action_id: str | None,
    ) -> dict[str, Any]:
        if type(observation) is not Observation:
            raise ProcessBrokerProtocolError(
                "WebArena adapter returned the wrong observation contract"
            )
        if (
            observation.episode_id != episode_id
            or observation.stage is not stage
            or observation.prior_action_id != prior_action_id
        ):
            raise ProcessBrokerProtocolError(
                "WebArena adapter observation differs from causal request"
            )
        return observation.to_dict()

    def runtime_reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self._validated_request("runtime_reset", payload)
        if self._episode_id is not None:
            raise ProcessBrokerProtocolError("WebArena process backend reset repeated")
        if (
            request["benchmark_version"] != self._task.benchmark_version
            or request["start_state_id"] != self._task.start_state_id
            or request["task_specification_sha256"] != self._task.record_sha256
        ):
            raise ProcessBrokerProtocolError(
                "WebArena process backend reset task contract differs"
            )
        observation = self._call_adapter(
            "runtime_reset",
            request,
            lambda: self._adapter.reset(
                detached_record_copy(self._task),
                episode_id=request["episode_id"],
                seed=request["reset_stage_seed"],
            )
        )
        receipt = self._adapter.reset_state_receipt()
        if type(receipt) is not WebArenaResetStateReceipt:
            raise ProcessBrokerProtocolError(
                "WebArena process backend lacks the hashes-only reset receipt"
            )
        self._episode_id = request["episode_id"]
        self._reset_receipt = receipt
        return {
            "observation": self._observation_result(
                observation,
                episode_id=request["episode_id"],
                stage=ObservationStage.RESET,
                prior_action_id=None,
            ),
            "reset_state_receipt": receipt.to_dict(),
        }

    def runtime_observe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self._validated_request("runtime_observe", payload)
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "WebArena process backend has not reset"
            )
        stage = ObservationStage(request["stage"])
        prior_action_id = request["prior_action_id"]
        observation = self._call_adapter(
            "runtime_observe",
            request,
            lambda: self._adapter.observe(
                stage=stage,
                prior_action_id=prior_action_id,
            )
        )
        return {
            "observation": self._observation_result(
                observation,
                episode_id=self._episode_id,
                stage=stage,
                prior_action_id=prior_action_id,
            )
        }

    def runtime_execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = self._validated_request("runtime_execute", payload)
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "WebArena process backend has not reset"
            )
        try:
            action = ConcreteAction.from_dict(request["action"])
        except (TypeError, ValueError) as exc:
            raise ProcessBrokerProtocolError(
                "WebArena process backend action is invalid"
            ) from exc
        execution = self._call_adapter(
            "runtime_execute",
            request,
            lambda: self._adapter.execute(action),
        )
        if type(execution) is not AdapterExecution:
            raise ProcessBrokerProtocolError(
                "WebArena adapter returned the wrong execution contract"
            )
        return {"execution": execution.to_dict()}

    def runtime_register_rejected(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind an executor-local rejection without invoking browser execute."""

        request = self._validated_request("runtime_register_rejected", payload)
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "WebArena process backend has not reset"
            )
        try:
            action = ConcreteAction.from_dict(request["action"])
            execution = ExecutionResult.from_dict(request["execution"])
        except (TypeError, ValueError) as exc:
            raise ProcessBrokerProtocolError(
                "WebArena rejected-action registration is invalid"
            ) from exc
        if (
            type(action) is not ConcreteAction
            or type(execution) is not ExecutionResult
            or execution.action_id != action.action_id
            or execution.status is not ExecutionStatus.REJECTED
            or execution.state_changed is not False
            or execution.environment_error is not False
        ):
            raise ProcessBrokerProtocolError(
                "WebArena rejected-action registration differs"
            )
        protected = (action, execution)
        protected_hashes = tuple(item.record_sha256 for item in protected)
        callback = tuple(detached_record_copy(item) for item in protected)
        callback_hashes = tuple(item.record_sha256 for item in callback)
        callback_error: BaseException | None = None
        callback_result: object | None = None
        try:
            callback_result = self._adapter.register_rejected_action(
                callback[0],  # type: ignore[arg-type]
                callback[1],  # type: ignore[arg-type]
            )
        except BaseException as exc:
            callback_error = exc
        mutated: list[str] = []
        for label, records, hashes in (
            ("backend", protected, protected_hashes),
            ("adapter", callback, callback_hashes),
        ):
            for index, (item, expected) in enumerate(zip(records, hashes)):
                try:
                    actual = item.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(f"{label}[{index}]")
        if mutated:
            error = ProcessBrokerProtocolError(
                "WebArena adapter mutated rejected-action registration: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise error from callback_error
            raise error
        if callback_error is not None:
            raise callback_error
        if callback_result is not None:
            raise ProcessBrokerProtocolError(
                "WebArena rejected-action hook returned data"
            )
        return {"registered": True}

    @property
    def sealed_finalization_available(self) -> bool:
        return (
            callable(self._finalizer)
            and callable(self._transition_evaluator)
            and type(self._sealed_sink) is SealedVerifierSink
            and type(self._sealed_writer) is SealedVerifierWriter
        )

    def _sealed_records(self) -> list[dict[str, Any]]:
        sink = self._sealed_sink
        if type(sink) is not SealedVerifierSink:
            raise ProcessBrokerProtocolError(
                "WebArena process backend lacks its child-owned sealed sink"
            )
        return sink.verified_records()

    @staticmethod
    def _require_signal_matches_record(
        signal: object,
        record: Mapping[str, Any],
        *,
        context: str,
    ) -> OpaqueTerminalSignal:
        if type(signal) is not OpaqueTerminalSignal:
            raise ProcessBrokerProtocolError(f"{context} returned a non-opaque signal")
        if (
            signal.event_id != record.get("event_id")
            or signal.token_sha256 != record.get("opaque_token_sha256")
            or signal.terminate is not record.get("should_terminate")
        ):
            raise ProcessBrokerProtocolError(
                f"{context} signal differs from the outer sealed record"
            )
        return signal

    def runtime_terminal(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = self._validated_request("runtime_terminal", payload)
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "WebArena process backend has not reset"
            )
        try:
            binding = VerifierReceiptBinding.from_dict(request["receipt_binding"])
        except (TypeError, ValueError) as exc:
            raise ProcessBrokerProtocolError(
                "WebArena process backend terminal binding is invalid"
            ) from exc
        if self.sealed_finalization_available:
            assert self._transition_evaluator is not None
            assert self._sealed_writer is not None
            assert type(self._sealed_callback_guard) is _SealedCallbackGuardSidecar
            before = self._sealed_records()
            signal = _call_state_preserving_sealed_callback(
                self._adapter,
                lambda: self._transition_evaluator(
                    task=detached_record_copy(self._task),
                    adapter=self._adapter,
                    receipt_binding=binding,
                    evidence_writer=self._sealed_writer,
                ),
                callback_kind="sealed_transition",
                context="WebArena sealed transition evaluator",
                evidence_sidecar=self._sealed_callback_guard,
            )
            after = self._sealed_records()
            if len(after) != len(before) + 1 or after[:-1] != before:
                raise ProcessBrokerProtocolError(
                    "WebArena transition evaluator did not append exactly one sealed record"
                )
            record = after[-1]
            runtime_binding = record.get("evidence", {}).get("runtime_binding")
            expected_binding = {
                key: binding.to_dict().get(key)
                for key in (
                    "receipt_kind",
                    "observation_id",
                    "observation_sha256",
                    "action_id",
                    "action_sha256",
                )
            }
            if (
                record.get("event_kind") != binding.receipt_kind
                or runtime_binding != expected_binding
            ):
                raise ProcessBrokerProtocolError(
                    "WebArena transition sealed record differs from causal binding"
                )
            signal = self._require_signal_matches_record(
                signal,
                record,
                context="WebArena transition evaluator",
            )
        else:
            signal = self._call_adapter(
                "runtime_terminal",
                request,
                lambda: self._adapter.terminal_signal(
                    detached_record_copy(self._task),
                    binding,
                )
            )
        if type(signal) is not OpaqueTerminalSignal:
            raise ProcessBrokerProtocolError(
                "WebArena adapter exposed a non-opaque terminal result"
            )
        return {"opaque_terminal_signal": signal.to_dict()}

    def runtime_close(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._validated_request("runtime_close", payload)
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "WebArena process backend has not reset"
            )
        result = self._adapter.close()
        if result is not None:
            raise ProcessBrokerProtocolError(
                "WebArena adapter runtime close returned unregistered data"
            )
        self._runtime_closed = True
        return {"closed": True}

    def sealed_finalize_episode(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "WebArena process backend has not reset"
            )
        if self._finalized:
            raise ProcessBrokerProtocolError(
                "WebArena process finalization may execute exactly once"
            )
        if (
            not callable(self._finalizer)
            or self._runtime_dir is None
            or type(self._sealed_writer) is not SealedVerifierWriter
        ):
            raise ProcessBrokerProtocolError(
                "WebArena process backend lacks a sealed finalizer"
            )
        try:
            summary = EpisodeSummary.from_dict(payload["episode_summary"])
        except (KeyError, SchemaError, TypeError, ValueError) as exc:
            raise ProcessBrokerProtocolError(
                "WebArena process finalizer summary is invalid"
            ) from exc
        sink = self._sealed_sink
        assert type(sink) is SealedVerifierSink
        expected_identity = {
            "episode_id": sink.episode_id,
            "task_id": sink.task_id,
            "system_id": sink.system_id,
            "repeat_id": sink.repeat_id,
            "model_seed": sink.matched_seed,
        }
        summary_identity = {
            "episode_id": summary.episode_id,
            "task_id": summary.task_id,
            "system_id": summary.system_id.value,
            "repeat_id": summary.repeat_id,
            "model_seed": summary.model_seed,
        }
        request_identity = {
            field: payload.get(field) for field in expected_identity
        }
        if (
            summary.episode_id != self._episode_id
            or summary.task_id != self._task.task_id
            or summary_identity != expected_identity
            or request_identity != expected_identity
        ):
            raise ProcessBrokerProtocolError(
                "WebArena process finalizer summary identity differs"
            )
        if not self._runtime_closed:
            close_result = self._adapter.close()
            if close_result is not None:
                raise ProcessBrokerProtocolError(
                    "WebArena adapter pre-finalization close returned data"
                )
            self._runtime_closed = True
        before = self._sealed_records()
        guard_sidecar = self._sealed_callback_guard
        if type(guard_sidecar) is not _SealedCallbackGuardSidecar:
            raise ProcessBrokerProtocolError(
                "WebArena finalizer lacks its child-owned callback guard sidecar"
            )
        signal = _call_state_preserving_sealed_callback(
            self._adapter,
            lambda: self._finalizer(
                task=detached_record_copy(self._task),
                adapter=self._adapter,
                episode_summary=summary,
                episode_runtime_dir=self._runtime_dir,
                evidence_writer=self._sealed_writer,
            ),
            callback_kind="sealed_finalizer",
            context="WebArena sealed finalizer",
            evidence_sidecar=guard_sidecar,
        )
        after = self._sealed_records()
        if len(after) != len(before) + 1 or after[:-1] != before:
            raise ProcessBrokerProtocolError(
                "WebArena finalizer did not append exactly one sealed record"
            )
        record = after[-1]
        if record.get("event_kind") != "episode_final":
            raise ProcessBrokerProtocolError(
                "WebArena finalizer did not append episode_final"
            )
        signal = self._require_signal_matches_record(
            signal,
            record,
            context="WebArena finalizer",
        )
        if signal.terminate is not True:
            raise ProcessBrokerProtocolError(
                "WebArena sealed finalizer returned a non-opaque acknowledgement"
            )
        self._finalized = True
        callback_guard_identity = guard_sidecar.identity()
        if (
            callback_guard_identity.get("record_count", 0) < 1
            or callback_guard_identity.get("tail_sha256") is None
        ):
            raise ProcessBrokerProtocolError(
                "WebArena finalizer lacks completed callback-guard evidence"
            )
        return {
            "opaque_terminal_signal": signal.to_dict(),
            "sealed_callback_guard_identity": callback_guard_identity,
        }

    def shutdown(self) -> dict[str, Any]:
        """Close the browser and return only the strict cleanup commitment."""

        if self._closed:
            raise ProcessBrokerProtocolError(
                "WebArena process backend shutdown may run exactly once"
            )
        manual_count, manual_tail = _manual_rescue_sidecar_identity(
            self._adapter,
            required=self._sealed_sink is not None,
        )
        browser_created = self._reset_receipt is not None
        close_attempted = browser_created
        close_completed = False
        try:
            process_close = getattr(
                self._adapter,
                "process_broker_close_browser",
                None,
            )
            if callable(process_close):
                receipt = process_close()
                if receipt is None:
                    browser_created = False
                    close_attempted = False
                    close_completed = False
                else:
                    created_value = getattr(
                        receipt,
                        "underlying_browser_created",
                        None,
                    )
                    closed_value = getattr(
                        receipt,
                        "underlying_browser_close_called",
                        None,
                    )
                    if type(created_value) is not bool or type(closed_value) is not bool:
                        raise ProcessBrokerProtocolError(
                            "WebArena child browser close receipt is malformed"
                        )
                    browser_created = created_value
                    close_attempted = created_value
                    close_completed = closed_value
                    if close_completed is not browser_created:
                        raise ProcessBrokerProtocolError(
                            "WebArena child browser close receipt is incomplete"
                        )
            else:
                result = self._adapter.close()
                if result is not None:
                    raise ProcessBrokerProtocolError(
                        "WebArena adapter shutdown returned unregistered data"
                    )
                close_completed = close_attempted
        finally:
            if type(self._sealed_callback_guard) is _SealedCallbackGuardSidecar:
                self._sealed_callback_guard.close()
            if type(self._sealed_sink) is SealedVerifierSink:
                self._sealed_sink.release_child_ownership()
        if browser_created and not (close_attempted and close_completed):
            raise ProcessBrokerProtocolError(
                "WebArena child cleanup did not complete browser close"
            )
        disposition = (
            "NO_BROWSER_CREATED"
            if not browser_created
            else (
                "RUNTIME_CLOSE_ACKNOWLEDGED"
                if self._runtime_closed
                else "CONTROL_ABORT_COMPLETED"
            )
        )
        if disposition not in PROCESS_BROKER_CHILD_CLEANUP_DISPOSITIONS:
            raise AssertionError("unregistered child cleanup disposition")
        self._closed = True
        return {
            "schema_version": PROCESS_BROKER_CHILD_CLEANUP_SCHEMA_VERSION,
            "record_type": PROCESS_BROKER_CHILD_CLEANUP_RECORD_TYPE,
            "cleanup_disposition": disposition,
            "browser_close_attempted": close_attempted,
            "browser_close_completed": close_completed,
            "manual_rescue_check_count": manual_count,
            "manual_rescue_tail_sha256": manual_tail,
        }


class ProcessBrokerWebArenaEnvironmentAdapter(EnvironmentAdapter):
    """Runtime-only ``EnvironmentAdapter`` over one issued broker capability."""

    benchmark_id = "webarena"

    __slots__ = (
        "_action_observation_pending",
        "_closed",
        "_client",
        "_defer_close_to_control",
        "_episode_id",
        "_reset_receipt",
        "_task_sha256",
        "_terminal_receipt_pending",
        "benchmark_version",
    )

    def __init__(
        self,
        *,
        client: ProcessIsolatedRuntimeClient,
        benchmark_version: str,
    ) -> None:
        if type(client) is not ProcessIsolatedRuntimeClient:
            raise TypeError(
                "process-backed WebArena adapter requires the exact runtime capability"
            )
        if (
            type(benchmark_version) is not str
            or not benchmark_version
            or benchmark_version != benchmark_version.strip()
        ):
            raise ValueError("process-backed WebArena benchmark version is required")
        self._client = client
        self.benchmark_version = benchmark_version
        self._episode_id: str | None = None
        self._task_sha256: str | None = None
        self._reset_receipt: WebArenaResetStateReceipt | None = None
        self._action_observation_pending = False
        self._terminal_receipt_pending = False
        self._defer_close_to_control = False
        self._closed = False

    def _call_client(self, callback: Callable[[], Any]) -> Any:
        try:
            return callback()
        except BaseException:
            # A rejected or transport-ambiguous request must not be followed by
            # a runtime-plane close. The broker owner retains the distinct
            # control capability needed for deterministic child cleanup.
            self._defer_close_to_control = True
            raise

    def reset(
        self,
        task: TaskSpecification,
        *,
        episode_id: str,
        seed: int,
    ) -> Observation:
        if self._closed or self._episode_id is not None:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena reset may run exactly once"
            )
        if (
            type(task) is not TaskSpecification
            or task.benchmark_id.casefold() != self.benchmark_id
            or task.benchmark_version != self.benchmark_version
        ):
            raise ProcessBrokerProtocolError(
                "process-backed WebArena task contract differs"
            )
        observation_value = self._call_client(
            lambda: self._client.reset(
                episode_id=episode_id,
                task_id=task.task_id,
                task_specification_sha256=task.record_sha256,
                benchmark_version=task.benchmark_version,
                start_state_id=task.start_state_id,
                reset_stage_seed=seed,
            )
        )
        # The client has published a reset even if a later defensive decode
        # were to fail.  Preserve its causal state so ``close`` cannot attempt
        # an out-of-order runtime_close and mask the original exception.
        self._episode_id = episode_id
        self._terminal_receipt_pending = True
        receipt_value = self._client.reset_state_receipt()
        if receipt_value is None:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena reset receipt is absent"
            )
        observation = Observation.from_dict(observation_value)
        receipt = WebArenaResetStateReceipt.from_dict(receipt_value)
        if type(observation) is not Observation or type(receipt) is not (
            WebArenaResetStateReceipt
        ):
            raise ProcessBrokerProtocolError(
                "process-backed WebArena reset records changed type"
            )
        self._task_sha256 = task.record_sha256
        self._reset_receipt = receipt
        return observation

    def reset_state_receipt(self) -> WebArenaResetStateReceipt | None:
        return self._reset_receipt

    def _require_open(self) -> str:
        if self._closed:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena adapter is closed"
            )
        if self._episode_id is None:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena adapter has not reset"
            )
        return self._episode_id

    def observe(
        self,
        *,
        stage: ObservationStage,
        prior_action_id: str | None = None,
    ) -> Observation:
        episode_id = self._require_open()
        if type(stage) is not ObservationStage:
            raise TypeError("process-backed observation stage has the wrong type")
        receipt = self._reset_receipt
        assert receipt is not None
        value = self._call_client(
            lambda: self._client.observe(
                episode_id=episode_id,
                task_id=receipt.task_id,
                stage=stage.value,
                prior_action_id=prior_action_id,
            )
        )
        self._action_observation_pending = False
        self._terminal_receipt_pending = True
        observation = Observation.from_dict(value)
        if type(observation) is not Observation:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena observation changed type"
            )
        return observation

    def execute(self, action: ConcreteAction) -> AdapterExecution:
        episode_id = self._require_open()
        if type(action) is not ConcreteAction:
            raise TypeError("process-backed execution requires ConcreteAction")
        receipt = self._reset_receipt
        assert receipt is not None
        # Dispatch is ambiguous until the authenticated result arrives.  Mark
        # the causal slot first so a lost response can only be cleaned through
        # orchestration-owned control shutdown, never an invalid runtime_close.
        self._action_observation_pending = True
        value = self._call_client(
            lambda: self._client.execute(
                episode_id=episode_id,
                task_id=receipt.task_id,
                action=action.to_dict(),
            )
        )
        execution = AdapterExecution.from_dict(value)
        if type(execution) is not AdapterExecution:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena execution changed type"
            )
        return execution

    def register_rejected_action(
        self,
        action: ConcreteAction,
        execution: ExecutionResult,
    ) -> None:
        episode_id = self._require_open()
        if type(action) is not ConcreteAction or type(execution) is not ExecutionResult:
            raise TypeError(
                "process-backed rejection registration requires exact records"
            )
        receipt = self._reset_receipt
        assert receipt is not None
        # Registration is transport-ambiguous until acknowledged.  Reserve the
        # same causal observation slot used by a real execute, but call the
        # dedicated non-dispatch operation.
        self._action_observation_pending = True
        self._call_client(
            lambda: self._client.register_rejected_action(
                episode_id=episode_id,
                task_id=receipt.task_id,
                action=action.to_dict(),
                execution=execution.to_dict(),
            )
        )
        return None

    def terminal_signal(
        self,
        task: TaskSpecification,
        binding: VerifierReceiptBinding | None = None,
    ) -> OpaqueTerminalSignal:
        episode_id = self._require_open()
        if (
            type(task) is not TaskSpecification
            or task.record_sha256 != self._task_sha256
        ):
            raise ProcessBrokerProtocolError(
                "process-backed WebArena terminal task differs"
            )
        if type(binding) is not VerifierReceiptBinding:
            raise ProcessBrokerProtocolError(
                "process-backed WebArena terminal requires causal receipt binding"
            )
        signal = self._call_client(
            lambda: self._client.terminal_signal(
                episode_id=episode_id,
                task_id=task.task_id,
                receipt_binding=binding,
            )
        )
        self._terminal_receipt_pending = False
        return signal

    def close(self) -> None:
        if self._closed:
            return None
        if self._episode_id is None:
            self._closed = True
            return None
        if (
            self._defer_close_to_control
            or self._action_observation_pending
            or self._terminal_receipt_pending
        ):
            # The strict runtime channel intentionally forbids close while an
            # action awaits observation or an observation awaits its causal
            # terminal receipt.  EpisodeRunner can unwind in either state when
            # policy/assessment code raises.  Do not replace that exception
            # with a second protocol error: the orchestration-owned broker
            # control shutdown remains responsible for child cleanup.
            self._closed = True
            return None
        receipt = self._reset_receipt
        assert receipt is not None
        self._client.close(
            episode_id=self._episode_id,
            task_id=receipt.task_id,
        )
        self._closed = True
        return None


def _close_adapter_and_release_sink_best_effort(
    adapter: object,
    sealed_sink: SealedVerifierSink | None,
) -> None:
    """Dispose post-factory resources without replacing the startup error."""

    try:
        adapter.close()  # type: ignore[attr-defined]
    except BaseException:
        pass
    if type(sealed_sink) is SealedVerifierSink:
        try:
            sealed_sink.release_child_ownership()
        except BaseException:
            pass


def create_backend(config: Mapping[str, Any]) -> WebArenaProcessBrokerBackend:
    """Generic worker entrypoint around a source-attested child-only factory."""

    detached = _detached_config(config)
    task = _validated_runtime_task(detached["task_specification"])
    runtime_dir = _validated_episode_runtime_dir(detached["episode_runtime_dir"])
    factory = _load_source_attested_adapter_factory(detached)
    finalizing = (
        detached["schema_version"]
        == PROCESS_BROKER_WEBARENA_FINALIZING_BACKEND_SCHEMA_VERSION
    )
    finalizer = _load_source_attested_finalizer(detached) if finalizing else None
    transition_evaluator = (
        _load_source_attested_transition_evaluator(detached)
        if finalizing
        else None
    )
    sealed_sink: SealedVerifierSink | None = None
    if finalizing:
        try:
            target = SealedVerifierStreamTarget.from_dict(
                detached["sealed_stream_target"]
            )
            if target.task_id != task.task_id:
                raise ValueError("sealed target task identity differs")
            sealed_sink = target.open_child_sink(
                episode_runtime_dir=runtime_dir
            )
        except (KeyError, SchemaError, TypeError, ValueError) as exc:
            raise ProcessBrokerProtocolError(
                "WebArena child-owned sealed stream target is invalid"
            ) from exc
    protected_task_sha256 = task.record_sha256
    factory_task = detached_record_copy(task)
    try:
        adapter = factory(
            task=factory_task,
            episode_runtime_dir=runtime_dir,
        )
    except BaseException:
        if type(sealed_sink) is SealedVerifierSink:
            sealed_sink.release_child_ownership()
        raise
    try:
        task_mutated = (
            task.record_sha256 != protected_task_sha256
            or factory_task.record_sha256 != protected_task_sha256
        )
    except BaseException:
        _close_adapter_and_release_sink_best_effort(adapter, sealed_sink)
        raise
    if task_mutated:
        _close_adapter_and_release_sink_best_effort(adapter, sealed_sink)
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory mutated protected task input"
        )
    try:
        return WebArenaProcessBrokerBackend(
            task=task,
            adapter=adapter,
            runtime_dir=runtime_dir,
            finalizer=finalizer,
            transition_evaluator=transition_evaluator,
            sealed_sink=sealed_sink,
        )
    except BaseException:
        _close_adapter_and_release_sink_best_effort(adapter, sealed_sink)
        raise
