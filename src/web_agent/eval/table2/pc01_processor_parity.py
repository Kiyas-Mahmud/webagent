"""Deterministic, processor-only PC-01 training/runtime parity evidence.

The check deliberately loads no model and performs no forward pass.  It sends
the same two deterministic image fixtures and causal text through the exact
GoldDataset preprocessing/collation functions and the production
``_RuntimeBatchBuilder`` transition/recovery functions, then requires bitwise
tensor equality for the pre-action, post-action, and recovery streams.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from importlib import metadata as importlib_metadata
import inspect
import os
from pathlib import Path
import tempfile
from typing import Any

from web_agent.eval.table2.common import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from web_agent.eval.table2.pc01_artifacts import (
    PC01_IMAGE_PROCESSOR_CLASS,
    PC01_MODEL_REVISION,
    PC01_PROCESSOR_CLASS,
    PC01_TRANSFORMERS_VERSION,
    validate_pc01_training_sources,
)
from web_agent.runtime.observation import ProcessorParityContract


class PC01ProcessorParityError(RuntimeError):
    """Training and production preprocessing did not produce exact parity."""


TrainingProcessorLoader = Callable[[Mapping[str, Any], Path], Any]
_PARITY_ACTION_TYPES = (
    "CLICK",
    "TYPE",
    "SELECT",
    "SCROLL",
    "NAVIGATE",
    "PRESS_KEY",
)
_PARITY_TENSOR_KEYS = {
    "attention_mask",
    "image_counts",
    "image_grid_thw",
    "input_ids",
    "pixel_values",
}


@contextmanager
def _strict_offline_hf():
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _default_training_processor_loader(
    config: Mapping[str, Any],
    processor_source: Path,
) -> Any:
    """Replay the authenticated training factory against the local snapshot."""

    from web_agent.train.gold_stages import build_processor

    local_config = deepcopy(dict(config))
    local_config["backbone"]["vlm_model"] = str(processor_source)
    with _strict_offline_hf():
        return build_processor(local_config)


def _qualified_class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PC01ProcessorParityError(f"{label} must be lowercase SHA-256")
    return text


def _write_ppm(path: Path, *, width: int, height: int, salt: int) -> None:
    payload = bytearray()
    for y in range(height):
        for x in range(width):
            payload.extend(
                (
                    (17 * x + 3 * y + salt) % 256,
                    (5 * x + 19 * y + 2 * salt) % 256,
                    (13 * x + 7 * y + 3 * salt) % 256,
                )
            )
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)
    path.chmod(0o444)


def _callable_identity(value: Any) -> dict[str, str]:
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise PC01ProcessorParityError(
            f"cannot attest preprocessing source for {value!r}"
        ) from exc
    return {
        "callable": f"{value.__module__}.{value.__qualname__}",
        "source_sha256": sha256_bytes(source),
    }


def current_pc01_processor_parity_source_identity() -> dict[str, dict[str, str]]:
    """Recompute the exact source identities used by PC-01 parity.

    This intentionally inspects the live production callables every time it is
    called. Missing modules, non-inspectable callables, or source drift are a
    validation failure; a receipt may not self-assert replacement identities.
    The builder and validator share this function so their registered source
    boundary cannot drift independently.
    """

    try:
        from web_agent.data.gold_dataloader import _collate_stream
        from web_agent.data.gold_dataset import GoldDataset
        from web_agent.runtime.qwen2vl_pc01 import (
            _RuntimeBatchBuilder,
            _context_text as runtime_context_text,
            _domain as runtime_domain,
            _load_image as runtime_load_image,
        )
    except ImportError as exc:
        raise PC01ProcessorParityError(
            "cannot import the registered PC-01 preprocessing sources"
        ) from exc

    return {
        "gold_context": _callable_identity(GoldDataset._context_text),
        "gold_vlm_inputs": _callable_identity(GoldDataset._vlm_inputs),
        "gold_collation": _callable_identity(_collate_stream),
        "runtime_stream": _callable_identity(_RuntimeBatchBuilder.stream),
        "runtime_transition": _callable_identity(_RuntimeBatchBuilder.transition),
        "runtime_recovery": _callable_identity(_RuntimeBatchBuilder.recovery),
        "runtime_context_text": _callable_identity(runtime_context_text),
        "runtime_domain": _callable_identity(runtime_domain),
        "runtime_load_image": _callable_identity(runtime_load_image),
    }


def _tensor_identity(tensor: Any, *, torch: Any) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise PC01ProcessorParityError(
            f"processor emitted a non-tensor value: {type(tensor)!r}"
        )
    detached = tensor.detach().cpu().contiguous()
    raw = detached.view(torch.uint8).numpy().tobytes(order="C")
    metadata = {
        "dtype": str(detached.dtype).removeprefix("torch."),
        "shape": list(detached.shape),
    }
    return {
        **metadata,
        "payload_sha256": sha256_bytes(canonical_json_bytes(metadata) + b"\0" + raw),
    }


def _bundle_identity(values: Mapping[str, Any], *, torch: Any) -> dict[str, Any]:
    tensors = {
        key: _tensor_identity(values[key], torch=torch)
        for key in sorted(values)
    }
    return {
        "sha256": sha256_bytes(canonical_json_bytes(tensors)),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def _require_equal_tensor_bundles(
    *,
    phase: str,
    training: Mapping[str, Any],
    runtime: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    if set(training) != set(runtime):
        raise PC01ProcessorParityError(
            f"{phase} tensor keys differ between training and runtime: "
            f"training={sorted(training)}, runtime={sorted(runtime)}"
        )
    for key in sorted(training):
        left = training[key]
        right = runtime[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise PC01ProcessorParityError(f"{phase}.{key} is not a tensor on both paths")
        if left.dtype != right.dtype:
            raise PC01ProcessorParityError(
                f"{phase}.{key} dtype differs: {left.dtype} != {right.dtype}"
            )
        if tuple(left.shape) != tuple(right.shape):
            raise PC01ProcessorParityError(
                f"{phase}.{key} shape differs: "
                f"{tuple(left.shape)} != {tuple(right.shape)}"
            )
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise PC01ProcessorParityError(
                f"{phase}.{key} tensor bytes differ between training and runtime"
            )
    training_identity = _bundle_identity(training, torch=torch)
    runtime_identity = _bundle_identity(runtime, torch=torch)
    if training_identity["sha256"] != runtime_identity["sha256"]:
        raise PC01ProcessorParityError(
            f"{phase} tensor-bundle identities differ after exact comparison"
        )
    return {
        "training_bundle_sha256": training_identity["sha256"],
        "runtime_bundle_sha256": runtime_identity["sha256"],
        "tensor_count": training_identity["tensor_count"],
        "tensors": training_identity["tensors"],
    }


def _strip_prefix(values: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    selected = {
        key.removeprefix(prefix): value
        for key, value in values.items()
        if key.startswith(prefix) and key != f"{prefix}row_indices"
    }
    if not selected:
        raise PC01ProcessorParityError(f"runtime produced no {prefix!r} tensors")
    return selected


def build_pc01_processor_parity_receipt(
    *,
    config: Mapping[str, Any],
    processor: Any,
    processor_source: str | Path,
    processor_contract: ProcessorParityContract,
    implementation_identity: Mapping[str, Any],
    processor_artifact_manifest_sha256: str,
    base_snapshot_directory_payload_sha256: str,
    training_environment_sha256: str,
    training_environment_record_sha256: str,
    training_source_manifest_sha256: str,
    training_action_value_evidence_sha256: str,
    run_contract_sha256: str,
    training_processor_loader: TrainingProcessorLoader | None = None,
) -> dict[str, Any]:
    """Return canonicalizable bitwise preprocessing-parity evidence.

    Optional ML dependencies are imported only inside this function.  The
    processor is supplied by the already authenticated local-snapshot loader;
    this function never calls a model constructor or reads weight shards.
    """

    try:
        import torch
        from PIL import Image
        from web_agent.data.gold_dataloader import _collate_stream
        from web_agent.data.gold_dataset import GoldDataset
        from web_agent.runtime.contracts import (
            ActionType,
            ConcreteAction,
            ExecutionResult,
            ExecutionStatus,
            PolicyObservation,
            RecoveryTransitionInput,
            RuntimeTaskView,
            TransitionInput,
        )
        from web_agent.runtime.qwen2vl_pc01 import (
            _RuntimeBatchBuilder,
            _context_text as runtime_context_text,
            _domain as runtime_domain,
            _load_image as runtime_load_image,
        )
    except ImportError as exc:  # pragma: no cover - depends on deployment stack
        raise PC01ProcessorParityError(
            "PyTorch and Pillow are required for processor-only parity validation"
        ) from exc

    exact_implementation = {
        "transformers_version": PC01_TRANSFORMERS_VERSION,
        "processor_class": PC01_PROCESSOR_CLASS,
        "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
        "use_fast": True,
    }
    if dict(implementation_identity) != exact_implementation:
        raise PC01ProcessorParityError(
            "processor implementation identity differs from registered PC-01 facts"
        )
    if processor_contract.processor_class != PC01_PROCESSOR_CLASS:
        raise PC01ProcessorParityError("processor contract class is not registered PC-01")
    try:
        installed_transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        raise PC01ProcessorParityError(
            "Transformers is required for processor-only parity validation"
        ) from exc
    if installed_transformers_version != PC01_TRANSFORMERS_VERSION:
        raise PC01ProcessorParityError(
            "processor parity requires transformers=="
            f"{PC01_TRANSFORMERS_VERSION}, got {installed_transformers_version!r}"
        )
    if _qualified_class_name(processor) != PC01_PROCESSOR_CLASS:
        raise PC01ProcessorParityError("runtime processor has the wrong concrete class")
    if (
        _qualified_class_name(getattr(processor, "image_processor", None))
        != PC01_IMAGE_PROCESSOR_CLASS
    ):
        raise PC01ProcessorParityError(
            "runtime processor does not use the concrete fast image processor"
        )
    for label, value in (
        ("processor artifact manifest", processor_artifact_manifest_sha256),
        ("base snapshot directory payload", base_snapshot_directory_payload_sha256),
        ("training environment", training_environment_sha256),
        ("training environment record", training_environment_record_sha256),
        ("training source manifest", training_source_manifest_sha256),
        ("training action-value evidence", training_action_value_evidence_sha256),
        ("run contract", run_contract_sha256),
    ):
        _require_sha256(value, label=label)
    training_source_manifest = validate_pc01_training_sources()
    authenticated_training_source_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(training_source_manifest)
    )
    if (
        training_source_manifest_sha256
        != authenticated_training_source_manifest_sha256
    ):
        raise PC01ProcessorParityError(
            "training source manifest differs from authenticated PC-01 sources"
        )

    source = Path(processor_source)
    loader = training_processor_loader or _default_training_processor_loader
    try:
        training_processor = loader(config, source)
    except Exception as exc:
        raise PC01ProcessorParityError(
            "cannot reconstruct the processor through the authenticated training factory"
        ) from exc
    if _qualified_class_name(training_processor) != PC01_PROCESSOR_CLASS:
        raise PC01ProcessorParityError(
            "training factory resolved the wrong concrete processor class"
        )
    if (
        _qualified_class_name(getattr(training_processor, "image_processor", None))
        != PC01_IMAGE_PROCESSOR_CLASS
    ):
        raise PC01ProcessorParityError(
            "training factory did not resolve the concrete fast image processor"
        )

    cfg = dict(config)
    dataset = object.__new__(GoldDataset)
    dataset.cfg = cfg
    dataset.processor = training_processor
    runtime_builder = _RuntimeBatchBuilder(
        processor=processor,
        config=cfg,
        torch=torch,
    )
    dataset.vlm_contract = runtime_builder.vlm_contract

    fixture_width = 64
    fixture_height = 48
    task = RuntimeTaskView(
        task_id="pc01-processor-parity-fixture",
        goal="Inspect the deterministic processor parity fixture",
    )
    domain = "parity.example.test"
    # The authenticated 24,107-row PC-01 train corpus contains no action_value
    # key.  Both training and runtime must therefore omit action-value text even
    # when the executable browser action itself carries parameters.
    regular_parameter_value = "deterministic fixture text"
    recovery_parameter_value = "ESCAPE"
    processor_action_value = ""

    with tempfile.TemporaryDirectory(prefix="pc01-processor-parity-") as temporary:
        root = Path(temporary)
        before_path = root / "before.ppm"
        after_path = root / "after.ppm"
        _write_ppm(
            before_path,
            width=fixture_width,
            height=fixture_height,
            salt=11,
        )
        _write_ppm(
            after_path,
            width=fixture_width,
            height=fixture_height,
            salt=73,
        )

        def observation(path: Path, *, observation_id: str) -> PolicyObservation:
            return PolicyObservation(
                task_id=task.task_id,
                goal=task.goal,
                observation_id=observation_id,
                screenshot_sha256=sha256_file(path),
                screenshot_path=str(path),
                width=fixture_width,
                height=fixture_height,
                url=f"https://{domain}/fixture",
                title="PC-01 processor parity fixture",
            )

        before_observation = observation(before_path, observation_id="parity-before")
        after_observation = observation(after_path, observation_id="parity-after")
        regular_action = ConcreteAction(
            action_id="parity-regular-action",
            source_decision_id="parity-regular-decision",
            action_type=ActionType.TYPE,
            parameters={"text": regular_parameter_value},
        )
        transition = TransitionInput(
            task_id=task.task_id,
            pre_observation=before_observation,
            executed_action=regular_action,
            execution_result=ExecutionResult(
                action_id=regular_action.action_id,
                status=ExecutionStatus.EXECUTED,
                executor_step=1,
                state_changed=True,
            ),
            post_observation=after_observation,
        )
        recovery_action = ConcreteAction(
            action_id="parity-recovery-action",
            source_decision_id="parity-recovery-decision",
            action_type=ActionType.PRESS_KEY,
            parameters={"key": recovery_parameter_value},
            recovery_attempt_id="parity-recovery-attempt",
        )
        recovery_transition = RecoveryTransitionInput(
            task_id=task.task_id,
            incident_id="parity-incident",
            attempt_id="parity-recovery-attempt",
            pre_recovery_observation=before_observation,
            recovery_actions=(recovery_action,),
            post_recovery_observation=after_observation,
        )

        with Image.open(before_path) as image:
            before_image = image.convert("RGB").copy()
        with Image.open(after_path) as image:
            after_image = image.convert("RGB").copy()
        training_input = {
            "task_description": task.goal,
            "website_domain": domain,
        }

        def training_stream(
            phase: str,
            images: list[Any],
            *,
            executed_action: str = "",
            action_value: str = "",
        ) -> dict[str, Any]:
            unbatched = dataset._vlm_inputs(
                training_input,
                images,
                phase=phase,
                executed_action=executed_action,
                action_value=action_value,
            )
            prefix = f"{phase}_"
            row = {f"{prefix}{key}": value for key, value in unbatched.items()}
            collated = _collate_stream([row], prefix, cfg)
            return _strip_prefix(collated, prefix)

        training = {
            "pre": training_stream("pre", [before_image]),
            "post": training_stream(
                "post",
                [before_image, after_image],
                executed_action=regular_action.action_type.value,
                action_value=processor_action_value,
            ),
            "recovery": training_stream(
                "recovery",
                [before_image, after_image],
                executed_action=recovery_action.action_type.value,
                action_value=processor_action_value,
            ),
        }
        runtime_transition = runtime_builder.transition(task, transition)
        runtime_recovery = runtime_builder.recovery(task, recovery_transition)
        runtime = {
            "pre": _strip_prefix(runtime_transition, "pre_"),
            "post": _strip_prefix(runtime_transition, "post_"),
            "recovery": _strip_prefix(runtime_recovery, "recovery_"),
        }
        streams = {
            phase: _require_equal_tensor_bundles(
                phase=phase,
                training=training[phase],
                runtime=runtime[phase],
                torch=torch,
            )
            for phase in ("pre", "post", "recovery")
        }
        action_specs: dict[ActionType, tuple[dict[str, Any], tuple[float, ...] | None]] = {
            ActionType.CLICK: ({"target_x": 20.0, "target_y": 15.0}, (0.1, 0.1, 0.2, 0.2)),
            ActionType.TYPE: ({"text": "parameter-bearing-type"}, (0.1, 0.1, 0.2, 0.2)),
            ActionType.SELECT: ({"option": "parameter-bearing-option"}, (0.1, 0.1, 0.2, 0.2)),
            ActionType.SCROLL: ({"direction": "down", "amount": 480}, None),
            ActionType.NAVIGATE: ({"url": "https://parity.example.test/next"}, None),
            ActionType.PRESS_KEY: ({"key": "ESCAPE"}, None),
        }
        action_class_matrix: dict[str, Any] = {}
        for action_type in ActionType:
            parameters, bbox = action_specs[action_type]
            normal = ConcreteAction(
                action_id=f"parity-matrix-{action_type.value.lower()}",
                source_decision_id="parity-matrix-decision",
                action_type=action_type,
                parameters=parameters,
                bbox=bbox,
            )
            normal_transition = TransitionInput(
                task_id=task.task_id,
                pre_observation=before_observation,
                executed_action=normal,
                execution_result=ExecutionResult(
                    action_id=normal.action_id,
                    status=ExecutionStatus.EXECUTED,
                    executor_step=1,
                    state_changed=True,
                ),
                post_observation=after_observation,
            )
            recovery_matrix_action = ConcreteAction(
                action_id=f"parity-matrix-recovery-{action_type.value.lower()}",
                source_decision_id="parity-matrix-recovery-decision",
                action_type=action_type,
                parameters=parameters,
                bbox=bbox,
                recovery_attempt_id="parity-matrix-attempt",
            )
            matrix_recovery_transition = RecoveryTransitionInput(
                task_id=task.task_id,
                incident_id="parity-matrix-incident",
                attempt_id="parity-matrix-attempt",
                pre_recovery_observation=before_observation,
                recovery_actions=(recovery_matrix_action,),
                post_recovery_observation=after_observation,
            )
            matrix_runtime_transition = runtime_builder.transition(
                task,
                normal_transition,
            )
            matrix_runtime_recovery = runtime_builder.recovery(
                task,
                matrix_recovery_transition,
            )
            action_class_matrix[action_type.value] = {
                "parameter_sha256": sha256_bytes(canonical_json_bytes(parameters)),
                "processor_action_value_omitted": True,
                "post": _require_equal_tensor_bundles(
                    phase=f"post[{action_type.value}]",
                    training=training_stream(
                        "post",
                        [before_image, after_image],
                        executed_action=action_type.value,
                        action_value="",
                    ),
                    runtime=_strip_prefix(matrix_runtime_transition, "post_"),
                    torch=torch,
                ),
                "recovery": _require_equal_tensor_bundles(
                    phase=f"recovery[{action_type.value}]",
                    training=training_stream(
                        "recovery",
                        [before_image, after_image],
                        executed_action=action_type.value,
                        action_value="",
                    ),
                    runtime=_strip_prefix(matrix_runtime_recovery, "recovery_"),
                    torch=torch,
                ),
            }
        fixture = {
            "schema_version": "table2.pc01-processor-parity-fixture.v1",
            "width": fixture_width,
            "height": fixture_height,
            "before_sha256": sha256_file(before_path),
            "after_sha256": sha256_file(after_path),
            "task_id": task.task_id,
            "goal": task.goal,
            "url": before_observation.url,
            "regular_action": regular_action.action_type.value,
            "regular_action_parameter": regular_parameter_value,
            "recovery_action": recovery_action.action_type.value,
            "recovery_action_parameter": recovery_parameter_value,
            "processor_action_value_omitted": True,
        }

    source_identity = current_pc01_processor_parity_source_identity()
    return {
        "schema_version": "table2.pc01-processor-parity.v1",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "resolved_config_record_sha256": sha256_bytes(canonical_json_bytes(cfg)),
        "processor_contract_sha256": processor_contract.record_sha256,
        "processor_artifact_manifest_sha256": processor_artifact_manifest_sha256,
        "base_snapshot_directory_payload_sha256": (
            base_snapshot_directory_payload_sha256
        ),
        "training_environment_sha256": training_environment_sha256,
        "training_environment_record_sha256": training_environment_record_sha256,
        "training_source_manifest_sha256": training_source_manifest_sha256,
        "training_action_value_evidence_sha256": (
            training_action_value_evidence_sha256
        ),
        "training_source_manifest": training_source_manifest,
        "run_contract_sha256": run_contract_sha256,
        "implementation_identity": exact_implementation,
        "training_processor_implementation": {
            "transformers_version": installed_transformers_version,
            "processor_class": _qualified_class_name(training_processor),
            "image_processor_class": _qualified_class_name(
                training_processor.image_processor
            ),
            "factory": "web_agent.train.gold_stages.build_processor",
            "use_fast_argument": "omitted",
        },
        "runtime_processor_implementation": exact_implementation,
        "source_identity": source_identity,
        "fixture": fixture,
        "streams": streams,
        "action_class_matrix": action_class_matrix,
        "parity_verified": True,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "network_access_used": False,
    }


def _validate_tensor_bundle_evidence(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "training_bundle_sha256",
        "runtime_bundle_sha256",
        "tensor_count",
        "tensors",
    }:
        raise PC01ProcessorParityError(f"{label} tensor-bundle evidence is malformed")
    training_hash = _require_sha256(
        value.get("training_bundle_sha256"),
        label=f"{label} training bundle",
    )
    runtime_hash = _require_sha256(
        value.get("runtime_bundle_sha256"),
        label=f"{label} runtime bundle",
    )
    if training_hash != runtime_hash:
        raise PC01ProcessorParityError(f"{label} training/runtime bundle hashes differ")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _PARITY_TENSOR_KEYS:
        raise PC01ProcessorParityError(f"{label} tensor coverage changed")
    if value.get("tensor_count") != len(_PARITY_TENSOR_KEYS):
        raise PC01ProcessorParityError(f"{label} tensor count changed")
    for name, tensor in tensors.items():
        if not isinstance(tensor, Mapping) or set(tensor) != {
            "dtype",
            "shape",
            "payload_sha256",
        }:
            raise PC01ProcessorParityError(f"{label}.{name} identity is malformed")
        if not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]:
            raise PC01ProcessorParityError(f"{label}.{name} dtype is invalid")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or not shape or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise PC01ProcessorParityError(f"{label}.{name} shape is invalid")
        _require_sha256(tensor.get("payload_sha256"), label=f"{label}.{name}")


def validate_pc01_processor_parity_receipt(
    receipt: Mapping[str, Any],
    *,
    resolved_config_record_sha256: str,
    processor_contract_sha256: str,
    processor_artifact_manifest_sha256: str,
    base_snapshot_directory_payload_sha256: str,
    training_environment_sha256: str,
    training_environment_record_sha256: str,
    training_source_manifest_sha256: str,
    training_action_value_evidence_sha256: str,
    run_contract_sha256: str,
) -> dict[str, Any]:
    """Reject incomplete or self-asserted processor-parity evidence."""

    expected_fields = {
        "schema_version",
        "model_id",
        "revision",
        "resolved_config_record_sha256",
        "processor_contract_sha256",
        "processor_artifact_manifest_sha256",
        "base_snapshot_directory_payload_sha256",
        "training_environment_sha256",
        "training_environment_record_sha256",
        "training_source_manifest_sha256",
        "training_action_value_evidence_sha256",
        "training_source_manifest",
        "run_contract_sha256",
        "implementation_identity",
        "training_processor_implementation",
        "runtime_processor_implementation",
        "source_identity",
        "fixture",
        "streams",
        "action_class_matrix",
        "parity_verified",
        "model_weights_loaded",
        "model_forward_executed",
        "network_access_used",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        raise PC01ProcessorParityError("processor parity receipt fields changed")
    exact = {
        "schema_version": "table2.pc01-processor-parity.v1",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "resolved_config_record_sha256": resolved_config_record_sha256,
        "processor_contract_sha256": processor_contract_sha256,
        "processor_artifact_manifest_sha256": processor_artifact_manifest_sha256,
        "base_snapshot_directory_payload_sha256": (
            base_snapshot_directory_payload_sha256
        ),
        "training_environment_sha256": training_environment_sha256,
        "training_environment_record_sha256": training_environment_record_sha256,
        "training_source_manifest_sha256": training_source_manifest_sha256,
        "training_action_value_evidence_sha256": (
            training_action_value_evidence_sha256
        ),
        "run_contract_sha256": run_contract_sha256,
        "parity_verified": True,
        "model_weights_loaded": False,
        "model_forward_executed": False,
        "network_access_used": False,
    }
    for field, expected in exact.items():
        if receipt.get(field) != expected:
            raise PC01ProcessorParityError(f"processor parity receipt changed {field}")
    authenticated_sources = validate_pc01_training_sources()
    if receipt.get("training_source_manifest") != authenticated_sources:
        raise PC01ProcessorParityError("processor parity training sources changed")
    exact_implementation = {
        "transformers_version": PC01_TRANSFORMERS_VERSION,
        "processor_class": PC01_PROCESSOR_CLASS,
        "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
        "use_fast": True,
    }
    if receipt.get("implementation_identity") != exact_implementation:
        raise PC01ProcessorParityError("processor parity implementation changed")
    if receipt.get("runtime_processor_implementation") != exact_implementation:
        raise PC01ProcessorParityError("runtime processor implementation changed")
    expected_training_implementation = {
        "transformers_version": PC01_TRANSFORMERS_VERSION,
        "processor_class": PC01_PROCESSOR_CLASS,
        "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
        "factory": "web_agent.train.gold_stages.build_processor",
        "use_fast_argument": "omitted",
    }
    if receipt.get("training_processor_implementation") != expected_training_implementation:
        raise PC01ProcessorParityError("training processor implementation changed")
    source_identity = receipt.get("source_identity")
    authenticated_source_identity = current_pc01_processor_parity_source_identity()
    if source_identity != authenticated_source_identity:
        raise PC01ProcessorParityError(
            "processor parity source identity differs from live registered callables"
        )
    fixture = receipt.get("fixture")
    fixture_fields = {
        "schema_version",
        "width",
        "height",
        "before_sha256",
        "after_sha256",
        "task_id",
        "goal",
        "url",
        "regular_action",
        "regular_action_parameter",
        "recovery_action",
        "recovery_action_parameter",
        "processor_action_value_omitted",
    }
    if not isinstance(fixture, Mapping) or set(fixture) != fixture_fields:
        raise PC01ProcessorParityError("processor parity fixture evidence changed")
    if (
        fixture.get("schema_version")
        != "table2.pc01-processor-parity-fixture.v1"
        or fixture.get("processor_action_value_omitted") is not True
    ):
        raise PC01ProcessorParityError("processor parity fixture semantics changed")
    for field in ("before_sha256", "after_sha256"):
        _require_sha256(fixture.get(field), label=f"fixture {field}")
    streams = receipt.get("streams")
    if not isinstance(streams, Mapping) or set(streams) != {"pre", "post", "recovery"}:
        raise PC01ProcessorParityError("processor parity causal streams changed")
    for name, evidence in streams.items():
        _validate_tensor_bundle_evidence(evidence, label=f"stream {name}")
    matrix = receipt.get("action_class_matrix")
    if not isinstance(matrix, Mapping) or set(matrix) != set(_PARITY_ACTION_TYPES):
        raise PC01ProcessorParityError("processor parity action-class coverage changed")
    for action_type, row in matrix.items():
        if not isinstance(row, Mapping) or set(row) != {
            "parameter_sha256",
            "processor_action_value_omitted",
            "post",
            "recovery",
        }:
            raise PC01ProcessorParityError(
                f"processor parity action row {action_type} is malformed"
            )
        _require_sha256(row.get("parameter_sha256"), label=f"{action_type} parameters")
        if row.get("processor_action_value_omitted") is not True:
            raise PC01ProcessorParityError(
                f"processor parity action row {action_type} exposes action_value"
            )
        _validate_tensor_bundle_evidence(row.get("post"), label=f"{action_type} post")
        _validate_tensor_bundle_evidence(
            row.get("recovery"),
            label=f"{action_type} recovery",
        )
    try:
        canonical_json_bytes(dict(receipt))
    except (TypeError, ValueError) as exc:
        raise PC01ProcessorParityError("processor parity receipt is not canonical JSON") from exc
    return dict(receipt)


def write_pc01_processor_parity_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Write one immutable canonical receipt without replacing prior evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(receipt)))
    except FileExistsError as exc:
        raise PC01ProcessorParityError(
            f"refusing to overwrite processor parity receipt: {destination}"
        ) from exc
    destination.chmod(0o444)
    return destination
