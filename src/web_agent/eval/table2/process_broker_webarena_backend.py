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
import importlib
import json
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


def _load_source_attested_adapter_factory(
    config: Mapping[str, Any],
) -> WebArenaEpisodeAdapterFactory:
    entrypoint = config.get("adapter_factory_entrypoint")
    if type(entrypoint) is not str:
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory entrypoint is invalid"
        )
    module_name, separator, attribute_name = entrypoint.partition(":")
    if (
        separator != ":"
        or any(not part.isidentifier() for part in module_name.split("."))
        or not attribute_name.isidentifier()
    ):
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory entrypoint is malformed"
        )
    repository_root = Path(__file__).resolve().parents[4]
    source, source_relative = _validated_repository_factory_source(
        repository_root=repository_root,
        relative_path=config.get("adapter_factory_source_relative_path"),
    )
    expected_module_path = Path(*module_name.split(".")).with_suffix(".py")
    if Path(source_relative) not in {
        expected_module_path,
        Path("src") / expected_module_path,
    }:
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source differs from entrypoint"
        )
    expected_sha256 = _lowercase_sha256(
        config.get("adapter_factory_source_sha256"),
        label="WebArena adapter factory source",
    )
    if sha256_file(source) != expected_sha256:
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory source hash differs"
        )
    module = importlib.import_module(module_name)
    loaded_source = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_source != source or sha256_file(loaded_source) != expected_sha256:
        raise ProcessBrokerProtocolError(
            "loaded WebArena adapter factory identity differs"
        )
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ProcessBrokerProtocolError(
            "WebArena adapter factory entrypoint is not callable"
        )
    return factory


def _load_source_attested_finalizer(
    config: Mapping[str, Any],
) -> WebArenaEpisodeSealedFinalizer:
    """Load only the explicitly hash-bound child-side finalizer."""

    entrypoint = config.get("sealed_finalizer_entrypoint")
    if type(entrypoint) is not str:
        raise ProcessBrokerProtocolError(
            "WebArena sealed finalizer entrypoint is invalid"
        )
    module_name, separator, attribute_name = entrypoint.partition(":")
    if (
        separator != ":"
        or any(not part.isidentifier() for part in module_name.split("."))
        or not attribute_name.isidentifier()
    ):
        raise ProcessBrokerProtocolError(
            "WebArena sealed finalizer entrypoint is malformed"
        )
    repository_root = Path(__file__).resolve().parents[4]
    source, source_relative = _validated_repository_factory_source(
        repository_root=repository_root,
        relative_path=config.get("sealed_finalizer_source_relative_path"),
    )
    expected_module_path = Path(*module_name.split(".")).with_suffix(".py")
    if Path(source_relative) not in {
        expected_module_path,
        Path("src") / expected_module_path,
    }:
        raise ProcessBrokerProtocolError(
            "WebArena sealed finalizer source differs from entrypoint"
        )
    expected_sha256 = _lowercase_sha256(
        config.get("sealed_finalizer_source_sha256"),
        label="WebArena sealed finalizer source",
    )
    if sha256_file(source) != expected_sha256:
        raise ProcessBrokerProtocolError(
            "WebArena sealed finalizer source hash differs"
        )
    module = importlib.import_module(module_name)
    loaded_source = Path(str(getattr(module, "__file__", ""))).resolve()
    if loaded_source != source or sha256_file(loaded_source) != expected_sha256:
        raise ProcessBrokerProtocolError(
            "loaded WebArena sealed finalizer identity differs"
        )
    finalizer = getattr(module, attribute_name, None)
    if not callable(finalizer):
        raise ProcessBrokerProtocolError(
            "WebArena sealed finalizer entrypoint is not callable"
        )
    return finalizer


def _load_source_attested_transition_evaluator(
    config: Mapping[str, Any],
) -> WebArenaSealedTransitionEvaluator:
    translated = {
        **config,
        "sealed_finalizer_entrypoint": config.get(
            "sealed_transition_callback_entrypoint"
        ),
        "sealed_finalizer_source_relative_path": config.get(
            "sealed_transition_callback_source_relative_path"
        ),
        "sealed_finalizer_source_sha256": config.get(
            "sealed_transition_callback_source_sha256"
        ),
    }
    return _load_source_attested_finalizer(translated)


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
        self._task = detached_record_copy(task)
        self._adapter = adapter
        self._episode_id: str | None = None
        self._reset_receipt: WebArenaResetStateReceipt | None = None
        self._runtime_dir = runtime_dir
        self._finalizer = finalizer
        self._transition_evaluator = transition_evaluator
        self._sealed_sink = sealed_sink
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
            before = self._sealed_records()
            signal = self._transition_evaluator(
                task=detached_record_copy(self._task),
                adapter=self._adapter,
                receipt_binding=binding,
                evidence_writer=self._sealed_writer,
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
        # Close only the runtime decision capability.  The child retains the
        # live page for the separate orchestration finalizer; adapter/browser
        # cleanup occurs during authenticated control shutdown afterwards.
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
        before = self._sealed_records()
        signal = self._finalizer(
            task=detached_record_copy(self._task),
            adapter=self._adapter,
            episode_summary=summary,
            episode_runtime_dir=self._runtime_dir,
            evidence_writer=self._sealed_writer,
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
        self._runtime_closed = True
        self._finalized = True
        return {"opaque_terminal_signal": signal.to_dict()}

    def shutdown(self) -> None:
        """Best-effort child cleanup when control shutdown follows a failed session."""

        if self._closed:
            return None
        try:
            result = self._adapter.close()
        finally:
            if type(self._sealed_sink) is SealedVerifierSink:
                self._sealed_sink.release_child_ownership()
        if result is not None:
            raise ProcessBrokerProtocolError(
                "WebArena adapter shutdown returned unregistered data"
            )
        self._closed = True
        return None


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
