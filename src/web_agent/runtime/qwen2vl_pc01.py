"""Real, dependency-lazy Qwen2-VL runtime bridge for the PC-01 checkpoint.

This module exposes the four model callbacks required by the Table 2 runtime:
pre-action policy/grounding, post-action diagnosis and recovery selection,
executed-recovery assessment, and the exact 768-dimensional post-action memory
adapter representation.  It also exposes a separately loaded, unadapted E0
base model for action generation and the common parameter-provider fallback.

Nothing is downloaded at import or load time.  All model/processor inputs must
be local, pinned artifacts.  Missing PyTorch, Transformers, PEFT, CUDA, or local
weights is a hard error; there is no synthetic or heuristic model fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from threading import Lock
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from web_agent.eval.table2.common import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from web_agent.eval.table2.pc01_artifacts import (
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_MODEL_SEED,
    PC01_MODEL_REVISION,
    PC01_SAVED_FULL_CONFIG_NAME,
    PC01ArtifactError,
    load_checkpoint_saved_config,
    load_pinned_local_processor,
    load_processor_contract,
)
from web_agent.eval.table2.resolved_config import (
    assert_checkpoint_config_matches,
    load_resolved_config_identity,
)
from web_agent.labels import (
    ACTION_TYPE,
    ACTION_TYPE_INV,
    EXECUTION_OUTCOME,
    FAILURE_TYPE_INV,
    RECOVERY_STRATEGY_INV,
)
from web_agent.runtime.action_parameters import ParameterResolutionError, ProviderCallable
from web_agent.runtime.checkpoint_inference import (
    LoadedSelectedBackboneBackend,
    LoadedSelectedCheckpointBackend,
    ValidationSelectedBackbone,
    ValidationSelectedCheckpoint,
)
from web_agent.runtime.contracts import (
    ActionType,
    JsonValue,
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryStrategy,
    RecoveryTransitionInput,
    RuntimeTaskView,
    TransitionAssessment,
    TransitionInput,
    canonical_sha256,
    float32_vector_sha256,
)
from web_agent.runtime.memory_adapter import PostFailureEmbedding, PostFailureEmbeddingRequest
from web_agent.runtime.observation import (
    ProcessorParityContract,
    assert_policy_screenshot_integrity,
    validate_processor_parity,
)
from web_agent.runtime.policy import ActionParseError


SELECTED_POLICY_ID = "validation-selected-web-agent"
SELECTED_POLICY_VERSION = "v1"
E0_POLICY_ID = "selected-backbone-unadapted"
E0_POLICY_VERSION = "v1"
E0_PARSER_ID = "strict-e0-json-action-parser"
E0_PARSER_VERSION = "v1"

E0_ACTION_PROMPT = (
    "You are the frozen base web-action policy used only for the E0 baseline.\n"
    "Given the task and current causal browser observation, return exactly one JSON\n"
    "object with keys \"action_type\", \"target\", \"bbox\", and \"value\".\n"
    "\"action_type\" must be exactly one of CLICK, TYPE, SELECT, SCROLL, NAVIGATE,\n"
    "or PRESS_KEY. For CLICK, TYPE, or SELECT, \"bbox\" must be the visible target's\n"
    "normalized [x, y, width, height] box in [0, 1], with positive width and height;\n"
    "for every other action it must be null. Use \"target\" for a short visible\n"
    "element/region description (or null when no target applies), and \"value\" for\n"
    "text, option, direction, URL, or key content required by the chosen action (or\n"
    "null when no value applies). Do not claim success, use evaluator information,\n"
    "invent hidden page state, repair an invalid choice, or emit prose.\n"
)
PARAMETER_PROVIDER_PROMPT = (
    "Resolve only parameters that are necessary to execute the already-selected web\n"
    "action. Return exactly one JSON object. Use visible causal observation content;\n"
    "never use oracle labels, reference trajectories, future states, or memory. If a\n"
    "required parameter cannot be resolved, return {\"status\":\"REJECTED\"} without\n"
    "repairing or replacing the selected action.\n"
)
REGISTERED_GENERATION_KWARGS: Mapping[str, JsonValue] = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 128,
}


class PC01RuntimeError(RuntimeError):
    """The real PC-01 model cannot be loaded or cannot satisfy its contract."""


@dataclass(frozen=True, slots=True)
class PC01RuntimeArtifacts:
    """Local-only payloads needed by both selected and E0 model factories."""

    resolved_config_path: Path
    processor_contract_path: Path
    processor_source: Path
    e0_resolved_config_path: Path
    e0_processor_contract_path: Path
    e0_backbone_path: Path
    export_manifest_path: Path
    training_action_value_evidence_path: Path


@dataclass(frozen=True, slots=True)
class LoadedPC01Backends:
    selected: LoadedSelectedCheckpointBackend
    e0: LoadedSelectedBackboneBackend
    parameter_fallback_resolver: ProviderCallable


class PC01BackendFactories:
    """Matched standard factories sharing one lazily loaded model bundle."""

    def __init__(
        self,
        *,
        selected: ValidationSelectedCheckpoint,
        e0: ValidationSelectedBackbone,
        artifacts: PC01RuntimeArtifacts,
    ) -> None:
        self._selected = selected
        self._e0 = e0
        self._artifacts = artifacts
        self._loaded: LoadedPC01Backends | None = None
        self._load_lock = Lock()

    def _bundle(self) -> LoadedPC01Backends:
        with self._load_lock:
            if self._loaded is None:
                self._loaded = load_pc01_runtime_backends(
                    selected=self._selected,
                    e0=self._e0,
                    artifacts=self._artifacts,
                )
            return self._loaded

    def selected_checkpoint_backend(
        self,
        selection: ValidationSelectedCheckpoint,
    ) -> LoadedSelectedCheckpointBackend:
        if selection != self._selected:
            raise PC01RuntimeError("selected-checkpoint factory received another selection")
        return self._bundle().selected

    def selected_backbone_backend(
        self,
        selection: ValidationSelectedBackbone,
    ) -> LoadedSelectedBackboneBackend:
        if selection != self._e0:
            raise PC01RuntimeError("E0 factory received another backbone selection")
        return self._bundle().e0

    def resolve_parameters(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        rng: random.Random,
    ) -> Mapping[str, JsonValue]:
        return self._bundle().parameter_fallback_resolver(
            task, observation, decision, rng
        )


def _require_local_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise PC01RuntimeError(f"{label} must be a local non-symlink directory: {path}")
    return path.resolve()


def _require_ml_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from peft import set_peft_model_state_dict
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
        from web_agent.models.model import WebAgentModel
    except ImportError as exc:  # pragma: no cover - depends on optional DGX stack
        raise PC01RuntimeError(
            "PC-01 inference requires the pinned PyTorch/Transformers/PEFT runtime"
        ) from exc
    if not torch.cuda.is_available():
        raise PC01RuntimeError("PC-01 Qwen2-VL inference requires an available CUDA device")
    return (
        torch,
        set_peft_model_state_dict,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
        WebAgentModel,
    )


@contextmanager
def _strict_offline_hf():
    """Prevent an incomplete local payload from silently contacting the Hub."""

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


def _validate_selected_config(config: Mapping[str, Any]) -> None:
    if config.get("name") != PC01_SAVED_FULL_CONFIG_NAME:
        raise PC01RuntimeError("resolved configuration is not PC-01")
    if int(config.get("fused_dim", -1)) != 768:
        raise PC01RuntimeError("selected checkpoint must use fused_dim=768")
    backbone = config.get("backbone")
    data = config.get("data")
    model = config.get("model")
    if not isinstance(backbone, Mapping) or not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise PC01RuntimeError("selected configuration lacks model/data/backbone mappings")
    exact = {
        "path": "vlm",
        "family": "qwen2_vl",
        "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": PC01_MODEL_REVISION,
        "trust_remote_code": False,
        "qlora": True,
    }
    for field, expected in exact.items():
        if backbone.get(field) != expected:
            raise PC01RuntimeError(f"selected PC-01 config changed backbone.{field}")
    if data.get("causal_routing") is not True or data.get("use_state_after") is not True:
        raise PC01RuntimeError("selected PC-01 config lacks causal transition routing")
    if data.get("recovery_transitions") is not True:
        raise PC01RuntimeError("selected PC-01 config lacks recovery transitions")
    task_adapters = model.get("task_adapters")
    if not isinstance(task_adapters, Mapping) or task_adapters.get("enabled") is not True:
        raise PC01RuntimeError("selected PC-01 config lacks the task-adapter bank")


def _same_backbone(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "path",
        "family",
        "vlm_model",
        "revision",
        "trust_remote_code",
        "vlm_hidden_dim",
        "min_pixels",
        "max_pixels",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _runtime_action_value_evidence_sha256(
    *,
    selection: ValidationSelectedCheckpoint,
    artifacts: PC01RuntimeArtifacts,
) -> str:
    """Bind processor construction to the already-validated v3 evidence."""

    export_manifest = read_json(artifacts.export_manifest_path)
    action_value_evidence_sha256 = sha256_file(
        artifacts.training_action_value_evidence_path
    )
    if (
        export_manifest.get("schema_version") != "table2.pc01-export.v3"
        or export_manifest.get("runtime_ready") is not False
        or export_manifest.get("checkpoint_sha256")
        != selection.selected_checkpoint_sha256
        or export_manifest.get("resolved_config_payload_sha256")
        != selection.resolved_config_sha256
        or export_manifest.get("processor_contract_sha256")
        != selection.processor_contract_sha256
        or export_manifest.get("training_action_value_evidence_sha256")
        != action_value_evidence_sha256
    ):
        raise PC01RuntimeError(
            "selected runtime artifacts differ from the frozen v3 evidence bundle"
        )
    return action_value_evidence_sha256


def _load_selected_model(
    *,
    selection: ValidationSelectedCheckpoint,
    artifacts: PC01RuntimeArtifacts,
) -> tuple[Any, Any, Any, dict[str, Any], ProcessorParityContract]:
    if selection.model_seed != PC01_MODEL_SEED:
        raise PC01RuntimeError("selected checkpoint is not the registered PC-01 seed")
    if selection.selected_checkpoint_sha256 != PC01_EXPECTED_CHECKPOINT_SHA256:
        raise PC01RuntimeError(
            "selected checkpoint identity is not the registered PC-01 e6 model"
        )
    selection.verify_checkpoint()
    identity = load_resolved_config_identity(artifacts.resolved_config_path)
    if identity.record_sha256 != PC01_EXPECTED_CONFIG_SHA256:
        raise PC01RuntimeError("resolved configuration differs from the registered PC-01 hash")
    if identity.payload_sha256 != selection.resolved_config_sha256:
        raise PC01RuntimeError("selected resolved-config payload differs from the model manifest")
    _validate_selected_config(identity.mapping)
    expected_processor = load_processor_contract(artifacts.processor_contract_path)
    if expected_processor.record_sha256 != selection.processor_contract_sha256:
        raise PC01RuntimeError("selected processor contract differs from the model manifest")
    action_value_evidence_sha256 = _runtime_action_value_evidence_sha256(
        selection=selection,
        artifacts=artifacts,
    )
    try:
        loaded_processor = load_pinned_local_processor(
            identity.mapping,
            artifacts.processor_source,
            training_action_value_evidence_sha256=(
                action_value_evidence_sha256
            ),
        )
    except PC01ArtifactError as exc:
        raise PC01RuntimeError("cannot reconstruct the exact selected processor") from exc
    validate_processor_parity(expected_processor, loaded_processor.contract)

    torch, set_peft_model_state_dict, _, _, WebAgentModel = _require_ml_dependencies()
    saved_config, checkpoint_sha256, checkpoint = load_checkpoint_saved_config(
        selection.checkpoint_path
    )
    if checkpoint_sha256 != selection.selected_checkpoint_sha256:
        raise PC01RuntimeError("loaded checkpoint bytes differ from selection")
    assert_checkpoint_config_matches(saved_config, identity)

    local_backbone = _require_local_directory(
        artifacts.e0_backbone_path,
        label="PC-01 local base-model snapshot",
    )
    runtime_config = deepcopy(identity.mapping)
    runtime_config["backbone"]["vlm_model"] = str(local_backbone)
    try:
        with _strict_offline_hf():
            model = WebAgentModel(runtime_config)
        set_peft_model_state_dict(model.encoder.model, checkpoint["lora"])
        model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
        model.task_adapters.load_state_dict(checkpoint["task_adapters"], strict=True)
        model.failure_head.load_state_dict(checkpoint["failure"], strict=True)
        model.action_head.load_state_dict(checkpoint["action"], strict=True)
        model.memory_head.load_state_dict(checkpoint["memory"], strict=True)
        model.recovery_outcome_head.load_state_dict(
            checkpoint["recovery_outcome"], strict=True
        )
        for module in (
            model.adapter,
            model.task_adapters,
            model.failure_head,
            model.action_head,
            model.memory_head,
            model.recovery_outcome_head,
        ):
            module.to("cuda")
        model.requires_grad_(False)
        model.eval()
    except PC01RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - exercised on the DGX artifact
        raise PC01RuntimeError("failed to construct the exact frozen PC-01 model") from exc
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise PC01RuntimeError("selected PC-01 model is not fully frozen in evaluation mode")
    return model, loaded_processor.processor, torch, identity.mapping, expected_processor


def _load_unadapted_base(
    *,
    selection: ValidationSelectedBackbone,
    artifacts: PC01RuntimeArtifacts,
    selected_config: Mapping[str, Any],
    processor: Any,
    processor_contract: ProcessorParityContract,
) -> tuple[Any, Any]:
    if (
        selection.backbone_id != "Qwen/Qwen2-VL-2B-Instruct"
        or selection.backbone_revision != PC01_MODEL_REVISION
    ):
        raise PC01RuntimeError(
            "E0 is not the unadapted version of the PC-01 backbone"
        )
    selection.verify_backbone()
    e0_identity = load_resolved_config_identity(artifacts.e0_resolved_config_path)
    if e0_identity.payload_sha256 != selection.resolved_config_sha256:
        raise PC01RuntimeError("E0 resolved-config payload differs from the model manifest")
    e0_backbone = e0_identity.mapping.get("backbone")
    selected_backbone = selected_config.get("backbone")
    if not isinstance(e0_backbone, Mapping) or not isinstance(selected_backbone, Mapping):
        raise PC01RuntimeError("E0 resolved configuration lacks a backbone mapping")
    if not _same_backbone(e0_backbone, selected_backbone):
        raise PC01RuntimeError("E0 and selected checkpoint do not use the same pinned backbone")
    expected_e0_processor = load_processor_contract(artifacts.e0_processor_contract_path)
    if expected_e0_processor.record_sha256 != selection.processor_contract_sha256:
        raise PC01RuntimeError("E0 processor contract differs from the model manifest")
    validate_processor_parity(expected_e0_processor, processor_contract)
    if selection.base_prompt_sha256 != sha256_bytes(E0_ACTION_PROMPT.encode("utf-8")):
        raise PC01RuntimeError("E0 prompt bytes differ from the registered prompt")
    if selection.parser_id != E0_PARSER_ID or selection.parser_version != E0_PARSER_VERSION:
        raise PC01RuntimeError("E0 parser identity differs from the PC-01 runtime parser")

    torch, _, AutoModelForImageTextToText, BitsAndBytesConfig, _ = (
        _require_ml_dependencies()
    )
    local_backbone = _require_local_directory(
        artifacts.e0_backbone_path,
        label="PC-01 unadapted E0 backbone",
    )
    backbone = selected_config["backbone"]
    dtype = getattr(torch, str(backbone.get("dtype", "float16")), None)
    if dtype is None:
        raise PC01RuntimeError("E0 backbone dtype is unavailable in the pinned PyTorch")
    quantization_config = None
    if backbone.get("load_in_4bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "revision": PC01_MODEL_REVISION,
        "dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "quantization_config": quantization_config,
    }
    attention = backbone.get("attn_implementation")
    if attention:
        kwargs["attn_implementation"] = attention
    try:
        with _strict_offline_hf():
            model = AutoModelForImageTextToText.from_pretrained(
                str(local_backbone), **kwargs
            )
        model.requires_grad_(False)
        model.eval()
    except Exception as exc:  # pragma: no cover - exercised on the DGX artifact
        raise PC01RuntimeError("failed to load the frozen unadapted E0 base model") from exc
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise PC01RuntimeError("E0 base model is not frozen in evaluation mode")
    return model, torch


def _load_image(observation: PolicyObservation) -> Any:
    assert_policy_screenshot_integrity(observation)
    if observation.screenshot_path is None:
        raise PC01RuntimeError("real PC-01 inference requires a screenshot artifact")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - required by the DGX image stack
        raise PC01RuntimeError("Pillow is required for PC-01 screenshot inference") from exc
    try:
        with Image.open(observation.screenshot_path) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise PC01RuntimeError("cannot decode the policy-visible screenshot") from exc


def _domain(observation: PolicyObservation) -> str:
    return urlparse(observation.url).hostname or ""


def _context_text(
    *,
    phase: str,
    task: RuntimeTaskView,
    domain: str,
    executed_action: str = "",
    action_value: str = "",
) -> str:
    instruction = {
        "pre": "Choose the next action from the page before execution.",
        "post": "Assess the result by comparing the page before and after execution.",
        "recovery": (
            "Assess whether the executed recovery succeeded by comparing the "
            "failure state with the post-recovery state."
        ),
    }[phase]
    action_context = ""
    if executed_action:
        action_context = f" | executed_action: {executed_action}"
        if action_value:
            action_context += f" | action_value: {action_value}"
    return (
        f"{instruction} current_task: {task.goal} | domain: {domain}"
        f"{action_context}"
    )


class _RuntimeBatchBuilder:
    def __init__(self, *, processor: Any, config: Mapping[str, Any], torch: Any) -> None:
        self.processor = processor
        self.config = config
        self.torch = torch
        try:
            from web_agent.models.encoders.vlm_contract import get_vlm_contract
        except ImportError as exc:  # pragma: no cover - dependency belongs to model stack
            raise PC01RuntimeError("cannot load the registered VLM tensor contract") from exc
        self.vlm_contract = get_vlm_contract(dict(config))

    def stream(
        self,
        *,
        task: RuntimeTaskView,
        observations: Sequence[PolicyObservation],
        phase: str,
        executed_action: str = "",
        action_value: str = "",
    ) -> dict[str, Any]:
        if not observations:
            raise PC01RuntimeError("processor stream requires at least one observation")
        images = [_load_image(observation) for observation in observations]
        text = _context_text(
            phase=phase,
            task=task,
            domain=_domain(observations[0]),
            executed_action=executed_action,
            action_value=action_value,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in images),
                    {"type": "text", "text": text},
                ],
            }
        ]
        try:
            chat = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.processor(text=[chat], images=images, return_tensors="pt")
        except Exception as exc:
            raise PC01RuntimeError("pinned processor rejected the causal model input") from exc
        missing = [
            key for key in self.vlm_contract.required_input_keys if key not in encoded
        ]
        if missing:
            raise PC01RuntimeError(f"pinned processor omitted required inputs: {missing}")
        result = {
            key: encoded[key]
            for key in self.vlm_contract.model_input_keys
            if key in encoded
        }
        result["image_counts"] = self.torch.tensor(
            [len(images)], dtype=self.torch.long
        )
        return result

    @staticmethod
    def prefix(stream: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in stream.items()}

    def transition(self, task: RuntimeTaskView, value: TransitionInput) -> dict[str, Any]:
        action = value.executed_action
        pre = self.stream(
            task=task,
            observations=(value.pre_observation,),
            phase="pre",
        )
        post = self.stream(
            task=task,
            observations=(value.pre_observation, value.post_observation),
            phase="post",
            executed_action=action.action_type.value,
            # The authenticated 24,107-row PC-01 training corpus contains no
            # action_value label. Supplying executable parameters here would
            # introduce text that the selected checkpoint never saw.
            action_value="",
        )
        return {**self.prefix(pre, "pre_"), **self.prefix(post, "post_")}

    def recovery(
        self,
        task: RuntimeTaskView,
        value: RecoveryTransitionInput,
    ) -> dict[str, Any]:
        # Gold recovery transitions supervise the one action immediately before
        # the post-recovery screenshot. Bind production preprocessing to that
        # final action type. The registered train-only data audit proves that no
        # action_value key existed, so executable parameters stay out of model text.
        recovery_action = value.recovery_actions[-1]
        action_name = recovery_action.action_type.value
        pre = self.stream(
            task=task,
            observations=(value.pre_recovery_observation,),
            phase="pre",
        )
        post = self.stream(
            task=task,
            observations=(
                value.pre_recovery_observation,
                value.post_recovery_observation,
            ),
            phase="post",
            executed_action=action_name,
            action_value="",
        )
        recovery = self.stream(
            task=task,
            observations=(
                value.pre_recovery_observation,
                value.post_recovery_observation,
            ),
            phase="recovery",
            executed_action=action_name,
            action_value="",
        )
        return {
            **self.prefix(pre, "pre_"),
            **self.prefix(post, "post_"),
            **self.prefix(recovery, "recovery_"),
            "recovery_row_indices": self.torch.tensor([0], dtype=self.torch.long),
        }


def _tensor_probabilities(torch: Any, tensor: Any) -> list[float]:
    probabilities = torch.softmax(tensor.float(), dim=-1)[0].detach().cpu().tolist()
    values = [float(value) for value in probabilities]
    if not values or not all(math.isfinite(value) for value in values):
        raise PC01RuntimeError("model produced invalid categorical probabilities")
    return values


def _bounded_bbox(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4 or not all(math.isfinite(float(value)) for value in values):
        raise PC01RuntimeError("model produced an invalid bbox tensor")
    x, y, width, height = (float(value) for value in values)
    if min(x, y, width, height) < 0.0:
        raise PC01RuntimeError("model produced a bbox with negative coordinates")
    if x > 1.0 or y > 1.0 or x + width > 1.0 or y + height > 1.0:
        raise PC01RuntimeError("model produced a bbox outside the normalized image plane")
    return (x, y, width, height)


def _hash_tensor_batch(batch: Mapping[str, Any], *, prefix: str) -> str:
    digest = hashlib.sha256()
    keys = sorted(key for key in batch if key.startswith(prefix))
    if not keys:
        raise PC01RuntimeError("cannot hash an empty processed model batch")
    for key in keys:
        tensor = batch[key]
        if not hasattr(tensor, "detach"):
            raise PC01RuntimeError(f"processed model batch contains non-tensor {key}")
        cpu = tensor.detach().cpu().contiguous()
        metadata = {
            "key": key,
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
        }
        raw = cpu.numpy().tobytes(order="C")
        digest.update(canonical_json_bytes(metadata))
        digest.update(len(raw).to_bytes(8, byteorder="big", signed=False))
        digest.update(raw)
    return digest.hexdigest()


class _SelectedCheckpointRuntime:
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        torch: Any,
        config: Mapping[str, Any],
        checkpoint_sha256: str,
        processor_contract: ProcessorParityContract,
    ) -> None:
        self.model = model
        self.torch = torch
        self.checkpoint_sha256 = checkpoint_sha256
        self.processor_contract = processor_contract
        self.batch = _RuntimeBatchBuilder(
            processor=processor,
            config=config,
            torch=torch,
        )
        self.lock = Lock()

    def _assert_eval(self) -> None:
        if self.model.training:
            raise PC01RuntimeError("PC-01 model left evaluation mode")

    def predict_action(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        rng: random.Random,
    ) -> PreActionDecision:
        del rng
        started = perf_counter()
        stream = self.batch.stream(
            task=task,
            observations=(observation,),
            phase="pre",
        )
        batch = self.batch.prefix(stream, "pre_")
        with self.lock, self.torch.inference_mode():
            self._assert_eval()
            pre_encoded = self.model.encode(batch, prefix="pre_")
            adapters = self.model.task_adapters
            pre_fused = adapters["policy"](pre_encoded["fused"])
            confidence = self.model.failure_head(
                pre_fused,
                confidence_fused=pre_fused,
            )["confidence"]
            bbox_source = (
                pre_encoded["fused"].float()
                if self.model.bbox_fp32_grounding
                else pre_encoded["fused"]
            )
            bbox_fused = (
                adapters["grounding"](bbox_source)
                if self.model.separate_bbox_adapter
                else pre_fused
            )
            spatial_tokens = pre_encoded.get("spatial_tokens")
            if spatial_tokens is not None:
                if self.model.bbox_fp32_grounding:
                    spatial_tokens = spatial_tokens.float()
                spatial_tokens = adapters[
                    "grounding" if self.model.separate_bbox_adapter else "policy"
                ](spatial_tokens)
            predictions = self.model.action_head(
                pre_fused,
                bbox_fused=bbox_fused,
                spatial_tokens=spatial_tokens,
                spatial_mask=pre_encoded.get("spatial_mask"),
                spatial_coords=pre_encoded.get("spatial_coords"),
                force_fp32_bbox=self.model.bbox_fp32_grounding,
            )
        probabilities = _tensor_probabilities(
            self.torch, predictions["action_type"]
        )
        action_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        action_label = ACTION_TYPE_INV[action_index]
        bbox = _bounded_bbox(
            predictions["bbox"][0].detach().float().cpu().tolist()
        )
        confidence_before = float(confidence[0].detach().float().cpu().item())
        decision_payload = {
            "task_id": task.task_id,
            "observation_id": observation.observation_id,
            "action_probabilities": probabilities,
            "bbox": bbox,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        return PreActionDecision(
            decision_id=f"pc01-action-{canonical_sha256(decision_payload)[:24]}",
            observation_id=observation.observation_id,
            action_type=ActionType(action_label),
            action_probabilities={
                ACTION_TYPE_INV[index]: probability
                for index, probability in enumerate(probabilities)
            },
            bbox=bbox,
            grounding_confidence=max(probabilities),
            confidence_before=confidence_before,
            input_observation_ids=(observation.observation_id,),
            policy_id=SELECTED_POLICY_ID,
            policy_version=SELECTED_POLICY_VERSION,
            parameter_hints={},
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def assess_transition(
        self,
        task: RuntimeTaskView,
        transition: TransitionInput,
        rng: random.Random,
    ) -> TransitionAssessment:
        del rng
        started = perf_counter()
        batch = self.batch.transition(task, transition)
        with self.lock, self.torch.inference_mode():
            self._assert_eval()
            predictions = self.model(batch)
        outcome = _tensor_probabilities(self.torch, predictions["outcome"])
        failure_types = _tensor_probabilities(
            self.torch, predictions["failure_type"]
        )
        recovery = _tensor_probabilities(self.torch, predictions["recovery"])
        failure_index = max(range(len(failure_types)), key=failure_types.__getitem__)
        recovery_index = max(range(len(recovery)), key=recovery.__getitem__)
        if "needs_recovery" not in predictions:
            raise PC01RuntimeError("PC-01 checkpoint omitted the needs-recovery head")
        needs_probability = float(
            self.torch.sigmoid(predictions["needs_recovery"].float())[0]
            .detach()
            .cpu()
            .item()
        )
        memory_probability = float(
            self.torch.sigmoid(predictions["memory_flag"].float())[0]
            .detach()
            .cpu()
            .item()
        )
        assessment_payload = {
            "transition": transition.record_sha256,
            "outcome": outcome,
            "failure_types": failure_types,
            "recovery": recovery,
        }
        return TransitionAssessment(
            assessment_id=f"pc01-transition-{canonical_sha256(assessment_payload)[:24]}",
            pre_observation_id=transition.pre_observation.observation_id,
            post_observation_id=transition.post_observation.observation_id,
            executed_action_id=transition.executed_action.action_id,
            predicted_failure=max(range(len(outcome)), key=outcome.__getitem__)
            == EXECUTION_OUTCOME["FAILURE"],
            failure_probability=outcome[EXECUTION_OUTCOME["FAILURE"]],
            failure_type=FAILURE_TYPE_INV[failure_index],
            failure_type_probabilities={
                FAILURE_TYPE_INV[index]: probability
                for index, probability in enumerate(failure_types)
            },
            needs_recovery=needs_probability >= 0.5,
            needs_recovery_probability=needs_probability,
            recovery_strategy=RecoveryStrategy(RECOVERY_STRATEGY_INV[recovery_index]),
            recovery_probabilities={
                RECOVERY_STRATEGY_INV[index]: probability
                for index, probability in enumerate(recovery)
            },
            memory_update_prediction=memory_probability >= 0.5,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def assess_recovery(
        self,
        task: RuntimeTaskView,
        transition: RecoveryTransitionInput,
        rng: random.Random,
    ) -> RecoveryAssessment:
        del rng
        started = perf_counter()
        batch = self.batch.recovery(task, transition)
        with self.lock, self.torch.inference_mode():
            self._assert_eval()
            predictions = self.model(batch)
        raw = predictions.get("recovery_outcome")
        if raw is None or int(raw.shape[0]) != 1:
            raise PC01RuntimeError("PC-01 recovery-outcome head returned no assessment")
        probability = float(
            self.torch.sigmoid(raw.float())[0].detach().cpu().item()
        )
        if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
            raise PC01RuntimeError("PC-01 recovery outcome probability is invalid")
        # PC-01 has one registered executed-recovery outcome head.  Table 2's
        # progress field therefore records the same model prediction, not a new
        # untrained head or oracle-derived value.
        assessment_payload = {
            "transition": transition.record_sha256,
            "recovery_outcome_probability": probability,
        }
        return RecoveryAssessment(
            assessment_id=f"pc01-recovery-{canonical_sha256(assessment_payload)[:24]}",
            incident_id=transition.incident_id,
            attempt_id=transition.attempt_id,
            pre_recovery_observation_id=(
                transition.pre_recovery_observation.observation_id
            ),
            post_recovery_observation_id=(
                transition.post_recovery_observation.observation_id
            ),
            recovery_action_ids=tuple(
                action.action_id for action in transition.recovery_actions
            ),
            predicted_failure_resolved=probability >= 0.5,
            predicted_resolution_probability=probability,
            predicted_progress=probability >= 0.5,
            predicted_progress_probability=probability,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def memory_embedding(
        self,
        request: PostFailureEmbeddingRequest,
    ) -> PostFailureEmbedding:
        task = RuntimeTaskView(
            task_id=request.post_action_input.task_id,
            goal=request.post_action_input.pre_observation.goal,
        )
        batch = self.batch.transition(task, request.post_action_input)
        processed_batch_sha256 = _hash_tensor_batch(batch, prefix="post_")
        with self.lock, self.torch.inference_mode():
            self._assert_eval()
            representation = self.model.memory_embedding(batch)
        if tuple(representation.shape) != (1, 768):
            raise PC01RuntimeError("PC-01 memory adapter did not return [1, 768]")
        values = tuple(
            float(value)
            for value in representation[0].detach().float().cpu().tolist()
        )
        return PostFailureEmbedding(
            query_id=request.query_id,
            post_failure_observation_id=request.post_failure_observation_id,
            values=values,
            request_sha256=request.record_sha256,
            post_failure_observation_sha256=(
                request.post_failure_observation_sha256
            ),
            post_action_input_sha256=request.post_action_input_sha256,
            processor_contract_sha256=request.processor_contract_sha256,
            checkpoint_sha256=request.checkpoint_sha256,
            processed_batch_sha256=processed_batch_sha256,
            embedding_sha256=float32_vector_sha256(values),
        )


def _strict_json_object(text: str, *, context: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ActionParseError(f"{context} is not one strict JSON object") from exc
    if not isinstance(value, dict):
        raise ActionParseError(f"{context} must be a JSON object")
    return value


def parse_e0_action_output(text: str) -> dict[str, JsonValue]:
    """Parse exactly the registered E0 JSON schema; never repair model text."""

    value = _strict_json_object(text, context="E0 action output")
    expected = {"action_type", "target", "bbox", "value"}
    if set(value) != expected:
        raise ActionParseError("E0 action output has missing or additional keys")
    action = value.get("action_type")
    if action not in ACTION_TYPE:
        raise ActionParseError("E0 action output names an unregistered action")
    for field in ("target", "value"):
        if value[field] is not None and not isinstance(value[field], str):
            raise ActionParseError(f"E0 action output {field} must be text or null")
    raw_bbox = value["bbox"]
    grounded = action in {"CLICK", "TYPE", "SELECT"}
    if grounded:
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ActionParseError(
                "E0 grounded action bbox must be a four-value JSON list"
            )
        if any(isinstance(item, bool) for item in raw_bbox):
            raise ActionParseError("E0 grounded action bbox values must be numeric")
        try:
            bbox = _bounded_bbox(raw_bbox)
        except (PC01RuntimeError, TypeError, ValueError) as exc:
            raise ActionParseError("E0 grounded action bbox is invalid") from exc
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise ActionParseError(
                "E0 grounded action bbox must have positive width and height"
            )
        value["bbox"] = list(bbox)
    elif raw_bbox is not None:
        raise ActionParseError("E0 non-grounded action bbox must be null")
    return value


class _UnadaptedBaseRuntime:
    def __init__(self, *, model: Any, processor: Any, torch: Any) -> None:
        self.model = model
        self.processor = processor
        self.torch = torch
        self.lock = Lock()

    def _generate(
        self,
        *,
        prompt: str,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        suffix: str,
    ) -> str:
        image = _load_image(observation)
        text = (
            f"{prompt}current_task: {task.goal} | domain: {_domain(observation)}"
            f"{suffix}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        try:
            chat = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self.processor(text=[chat], images=[image], return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            input_length = int(inputs["input_ids"].shape[1])
            with self.lock, self.torch.inference_mode():
                if self.model.training:
                    raise PC01RuntimeError("E0 base model left evaluation mode")
                generated = self.model.generate(
                    **inputs,
                    **dict(REGISTERED_GENERATION_KWARGS),
                )
            generated_suffix = generated[:, input_length:]
            decoded = self.processor.batch_decode(
                generated_suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except PC01RuntimeError:
            raise
        except Exception as exc:
            raise PC01RuntimeError("unadapted E0 generation failed") from exc
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise PC01RuntimeError("unadapted E0 model returned an invalid batch")
        return decoded[0]

    def predict_action(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        rng: random.Random,
    ) -> PreActionDecision:
        del rng
        started = perf_counter()
        raw = self._generate(
            prompt=E0_ACTION_PROMPT,
            task=task,
            observation=observation,
            suffix="",
        )
        parsed = parse_e0_action_output(raw)
        action = str(parsed["action_type"])
        probabilities = {
            label: float(label == action) for label in ACTION_TYPE
        }
        hints: dict[str, JsonValue] = {}
        if parsed["target"] is not None:
            hints["target"] = str(parsed["target"])
        if parsed["value"] is not None:
            value = str(parsed["value"])
            hints["value"] = value
            if action == "TYPE":
                hints["text"] = value
            elif action == "SELECT":
                hints["option"] = value
                controls = observation.current_page_state.get(
                    "observable_select_controls"
                )
                if isinstance(controls, list):
                    matching = [
                        control
                        for control in controls
                        if isinstance(control, Mapping)
                        and control.get("target_bbox") == parsed["bbox"]
                        and isinstance(control.get("candidate_options"), list)
                    ]
                    if len(matching) == 1:
                        hints["candidate_options"] = list(
                            matching[0]["candidate_options"]
                        )
            elif action == "SCROLL":
                hints["direction"] = value
            elif action == "NAVIGATE":
                hints["url"] = value
            elif action == "PRESS_KEY":
                hints["key"] = value
        payload = {
            "task_id": task.task_id,
            "observation_id": observation.observation_id,
            "parsed": parsed,
        }
        return PreActionDecision(
            decision_id=f"pc01-e0-{canonical_sha256(payload)[:24]}",
            observation_id=observation.observation_id,
            action_type=ActionType(action),
            action_probabilities=probabilities,
            bbox=(
                tuple(float(item) for item in parsed["bbox"])
                if parsed["bbox"] is not None
                else None
            ),
            grounding_confidence=0.0,
            confidence_before=0.0,
            input_observation_ids=(observation.observation_id,),
            policy_id=E0_POLICY_ID,
            policy_version=E0_POLICY_VERSION,
            parameter_hints=hints,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def resolve_parameters(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        rng: random.Random,
    ) -> Mapping[str, JsonValue]:
        del rng
        suffix = (
            f" | selected_action: {decision.action_type.value}"
            f" | selected_bbox: {json.dumps(decision.bbox)}"
            f" | visible_parameter_hints: "
            f"{canonical_json_bytes(dict(decision.parameter_hints)).decode('utf-8')}"
        )
        raw = self._generate(
            prompt=PARAMETER_PROVIDER_PROMPT,
            task=task,
            observation=observation,
            suffix=suffix,
        )
        try:
            parsed = _strict_json_object(raw, context="parameter-provider output")
        except ActionParseError as exc:
            raise ParameterResolutionError(str(exc)) from exc
        if parsed == {"status": "REJECTED"}:
            raise ParameterResolutionError("frozen base rejected parameter resolution")
        try:
            canonical_json_bytes(parsed)
        except (TypeError, ValueError) as exc:
            raise ParameterResolutionError(
                "frozen base parameters are not finite JSON values"
            ) from exc
        return parsed


def load_pc01_runtime_backends(
    *,
    selected: ValidationSelectedCheckpoint,
    e0: ValidationSelectedBackbone,
    artifacts: PC01RuntimeArtifacts,
) -> LoadedPC01Backends:
    """Load the selected PC-01 model and its separate unadapted E0 base."""

    model, processor, torch, config, contract = _load_selected_model(
        selection=selected,
        artifacts=artifacts,
    )
    selected_runtime = _SelectedCheckpointRuntime(
        model=model,
        processor=processor,
        torch=torch,
        config=config,
        checkpoint_sha256=selected.selected_checkpoint_sha256,
        processor_contract=contract,
    )
    e0_model, e0_torch = _load_unadapted_base(
        selection=e0,
        artifacts=artifacts,
        selected_config=config,
        processor=processor,
        processor_contract=contract,
    )
    e0_runtime = _UnadaptedBaseRuntime(
        model=e0_model,
        processor=processor,
        torch=e0_torch,
    )
    selected_backend = LoadedSelectedCheckpointBackend(
        checkpoint_sha256=selected.selected_checkpoint_sha256,
        resolved_config_sha256=selected.resolved_config_sha256,
        processor_contract=contract,
        action_predictor=selected_runtime.predict_action,
        transition_predictor=selected_runtime.assess_transition,
        recovery_predictor=selected_runtime.assess_recovery,
        memory_embedding=selected_runtime.memory_embedding,
        frozen=True,
        training=False,
    )
    e0_backend = LoadedSelectedBackboneBackend(
        backbone_id=e0.backbone_id,
        backbone_revision=e0.backbone_revision,
        backbone_sha256=e0.backbone_sha256,
        resolved_config_sha256=e0.resolved_config_sha256,
        processor_contract=load_processor_contract(
            artifacts.e0_processor_contract_path
        ),
        base_prompt_sha256=e0.base_prompt_sha256,
        parser_id=e0.parser_id,
        parser_version=e0.parser_version,
        action_predictor=e0_runtime.predict_action,
        frozen=True,
        training=False,
        adaptation_loaded=False,
        task_heads_loaded=False,
    )
    return LoadedPC01Backends(
        selected=selected_backend,
        e0=e0_backend,
        parameter_fallback_resolver=e0_runtime.resolve_parameters,
    )


def make_pc01_backend_factories(
    *,
    selected: ValidationSelectedCheckpoint,
    e0: ValidationSelectedBackbone,
    artifacts: PC01RuntimeArtifacts,
) -> PC01BackendFactories:
    """Create lazy factories compatible with both checkpoint adapters."""

    return PC01BackendFactories(selected=selected, e0=e0, artifacts=artifacts)


@dataclass(frozen=True, slots=True)
class PC01SeedOperationalServices:
    """Non-model services that a real BrowserGym/WebArena deployment must inject."""

    episode_state_resetter: Any
    recovery_action_planner: Any
    begin_measurement: Callable[..., Any]
    finish_measurement: Callable[..., Mapping[str, Any]]

    def __post_init__(self) -> None:
        # Keep imports local so importing this dependency-lazy model bridge
        # still has no BrowserGym, PyTorch, or Transformers side effect.
        from web_agent.runtime.recovery.controller import RecoveryActionPlanner
        from web_agent.runtime.state_reset import EpisodeStateResetter

        if not isinstance(self.episode_state_resetter, EpisodeStateResetter):
            raise PC01RuntimeError(
                "PC-01 operational services require a typed episode-state resetter"
            )
        if self.episode_state_resetter.frozen is not True:
            raise PC01RuntimeError("PC-01 episode-state resetter must be frozen")
        if not isinstance(self.recovery_action_planner, RecoveryActionPlanner):
            raise PC01RuntimeError(
                "PC-01 operational services require a typed recovery-action planner"
            )
        if self.recovery_action_planner.frozen is not True:
            raise PC01RuntimeError("PC-01 recovery-action planner must be frozen")
        if not callable(self.begin_measurement) or not callable(
            self.finish_measurement
        ):
            raise PC01RuntimeError(
                "PC-01 operational services require both measurement callbacks"
            )


# These are capabilities, not file names or guessed environment variables.
# Every item depends on the measured deployment and therefore must be supplied
# by a clean, source-attested campaign integration.  Keeping this list in code
# makes the current fail-closed boundary inspectable without implying that a
# fixture or an unverified BrowserGym default is production-ready.
PC01_LIVE_DEPLOYMENT_CAPABILITIES = (
    "deterministic WebArena service/account/database/start-state reset with "
    "hashes-only receipt",
    "exclusive browser-controller input audit proving zero manual rescue",
    "frozen oracle-blind BrowserGym observation/action/page-settle/screenshot mapping",
    "frozen action-safety and infrastructure-fault classification",
    "concrete REPLAN/ALTERNATIVE_TARGET recovery action planner",
    "complete per-episode model/resource measurement",
    "source-reviewed libwebarena evaluator port with pinned upstream/delta/config "
    "evidence, measured gpt-4-1106-preview availability, and write-only sealed output",
)


@dataclass(frozen=True, slots=True)
class PC01EvaluationRuntimeFactory:
    """Source-attestable factory producing the exact production binding type.

    BrowserGym, WebArena site deployment, credentials, reset semantics, safety
    policy, and screenshot capture remain deployment-owned.  They must arrive
    through ``create_webarena_runtime`` as a typed ``WebArenaRuntimeBinding``;
    omitting any live service fails before model loading.
    """

    create_webarena_runtime: Callable[..., Any]
    seed_services: Mapping[int, PC01SeedOperationalServices]
    create_process_isolated_webarena: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if not callable(self.create_webarena_runtime):
            raise PC01RuntimeError("a concrete BrowserGym/WebArena runtime factory is required")
        if self.create_process_isolated_webarena is not None and not callable(
            self.create_process_isolated_webarena
        ):
            raise PC01RuntimeError(
                "process-isolated WebArena factory must be callable when supplied"
            )
        if set(self.seed_services) != {PC01_MODEL_SEED}:
            raise PC01RuntimeError(
                "PC-01 live operational services must cover seed 42 exactly"
            )
        if type(self.seed_services[PC01_MODEL_SEED]) is not PC01SeedOperationalServices:
            raise PC01RuntimeError(
                "PC-01 seed 42 services must use PC01SeedOperationalServices"
            )

    def __call__(self, context: Any) -> Any:
        # Imports are deferred so merely importing this model bridge does not
        # pull the production runner or any browser/ML dependency into tests.
        from web_agent.eval.table2.production_runner import (
            EvaluationRuntimeBinding,
            FrozenRuntimeContext,
            SeedRuntimeBinding,
        )

        if type(context) is not FrozenRuntimeContext:
            raise PC01RuntimeError("PC-01 integration requires FrozenRuntimeContext")
        expected = set(context.model_manifest_paths)
        if expected != {PC01_MODEL_SEED}:
            raise PC01RuntimeError(
                "this PC-01 integration is bound only to matched model seed 42"
            )
        if (
            set(context.model_payload_paths) != expected
            or set(context.model_evidence_paths) != expected
            or set(self.seed_services) != expected
        ):
            raise PC01RuntimeError("PC-01 integration does not cover matched seeds exactly")
        bindings: dict[int, Any] = {}
        for seed in sorted(expected):
            manifest = read_json(context.model_manifest_paths[seed])
            payloads = context.model_payload_paths[seed]
            evidence = context.model_evidence_paths[seed]
            required = {
                "selected_checkpoint",
                "resolved_config",
                "processor_contract",
                "e0_backbone",
                "e0_resolved_config",
                "e0_processor_contract",
                "e0_parser",
            }
            if set(payloads) != required:
                raise PC01RuntimeError("PC-01 frozen model payload roles are incomplete")
            required_evidence = {
                "export_manifest",
                "base_snapshot_manifest",
                "processor_artifact_manifest",
                "processor_parity_receipt",
                "training_environment",
                "training_source_manifest",
                "training_action_value_evidence",
            }
            if set(evidence) != required_evidence:
                raise PC01RuntimeError(
                    "PC-01 frozen model evidence roles are incomplete"
                )
            selected = ValidationSelectedCheckpoint.from_mapping(
                manifest,
                checkpoint_path=payloads["selected_checkpoint"],
            )
            e0 = ValidationSelectedBackbone(
                manifest_id=str(manifest.get("manifest_id") or f"seed-{seed}-e0"),
                backbone_id=str(manifest["e0_backbone_id"]),
                backbone_revision=str(manifest["e0_backbone_revision"]),
                backbone_path=payloads["e0_backbone"],
                backbone_sha256=str(manifest["e0_backbone_sha256"]),
                resolved_config_sha256=str(manifest["e0_resolved_config_sha256"]),
                processor_contract_sha256=str(
                    manifest["e0_processor_contract_sha256"]
                ),
                base_prompt_sha256=str(manifest["e0_base_prompt_sha256"]),
                parser_id=str(manifest["e0_parser_id"]),
                parser_version=str(manifest["e0_parser_version"]),
                validation_rows_read=int(manifest["validation_rows_read"]),
                selection_scope=str(manifest["selection_scope"]),
                test_rows_read=int(manifest["test_rows_read"]),
                locked_test_rows_read=int(manifest["locked_test_rows_read"]),
            )
            artifacts = PC01RuntimeArtifacts(
                resolved_config_path=payloads["resolved_config"],
                processor_contract_path=payloads["processor_contract"],
                processor_source=payloads["e0_backbone"],
                e0_resolved_config_path=payloads["e0_resolved_config"],
                e0_processor_contract_path=payloads["e0_processor_contract"],
                e0_backbone_path=payloads["e0_backbone"],
                export_manifest_path=evidence["export_manifest"],
                training_action_value_evidence_path=evidence[
                    "training_action_value_evidence"
                ],
            )
            factories = make_pc01_backend_factories(
                selected=selected,
                e0=e0,
                artifacts=artifacts,
            )
            selected_backend = factories.selected_checkpoint_backend(selected)
            e0_backend = factories.selected_backbone_backend(e0)
            services = self.seed_services[seed]
            bindings[seed] = SeedRuntimeBinding(
                model_seed=seed,
                episode_state_resetter=services.episode_state_resetter,
                selected_checkpoint_backend=selected_backend,
                selected_backbone_backend=e0_backend,
                e0_parser_sha256=sha256_file(payloads["e0_parser"]),
                parameter_fallback_resolver=factories.resolve_parameters,
                parameter_fallback_backbone_sha256=e0.backbone_sha256,
                recovery_action_planner=services.recovery_action_planner,
                begin_measurement=services.begin_measurement,
                finish_measurement=services.finish_measurement,
                frozen=True,
            )
        return EvaluationRuntimeBinding(
            runtime_identity=dict(context.runtime_identity),
            seed_bindings=bindings,
            create_webarena_runtime=self.create_webarena_runtime,
            frozen=True,
            oracle_labels_exposed_to_runtime=False,
            create_process_isolated_webarena=(
                self.create_process_isolated_webarena
            ),
        )


def require_live_factory() -> None:
    """Explicit fail-closed entrypoint for deployments that forgot live wiring."""

    raise PC01RuntimeError(
        "construct PC01EvaluationRuntimeFactory only after freezing all live "
        "deployment capabilities: "
        + "; ".join(PC01_LIVE_DEPLOYMENT_CAPABILITIES)
    )
