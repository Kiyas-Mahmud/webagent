"""WebAgentModel — assembles front-end + FIXED heads into one nn.Module (ARCH 1).

Reads cfg["backbone"]["path"]:
  vlm          -> VLMEncoder -> Adapter(D->768)                      -> fused[768]
  dual_encoder -> VisionEncoder + TextEncoder -> CrossAttentionFusion -> fused[768]

Then the SAME FailureHead + ActionHead + MemoryHead return one flat prediction
dict consumed by CombinedLoss. Gold runs can enable causal routing: the shared
encoder/adapter creates a pre-action embedding and a post-action embedding, then
routes each head to information that exists when its prediction is made.

Recovery-v2 optionally preserves spatial image tokens, adds task-specific
residual adapters, and encodes a sparse third stream only for proper recovery
transitions. These switches are config-gated so the completed v1 path is intact.

Qwen-first build order: the `vlm` path is implemented now; the `dual_encoder`
path is added when SigLIP+RoBERTa is built.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from web_agent.models.adapter import Adapter, ResidualTaskAdapter
from web_agent.models.heads import (
    ActionHead,
    FailureHead,
    MemoryHead,
    RecoveryOutcomeHead,
)


class WebAgentModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.path = cfg["backbone"]["path"]
        self.causal_routing = bool(cfg["data"].get("causal_routing", False))
        self.use_recovery_transitions = bool(
            cfg["data"].get("recovery_transitions", False)
        )
        self.hierarchical_recovery = bool(
            cfg.get("loss", {}).get("hierarchical_recovery", False)
        )
        model_cfg = cfg.get("model", {})
        self.spatial_grounding = bool(model_cfg.get("spatial_grounding", False))
        self.bbox_fp32_grounding = bool(
            model_cfg.get("bbox_fp32_grounding", False)
        )
        fused_dim = cfg.get("fused_dim", 768)

        if self.path == "vlm":
            from web_agent.models.encoders.vlm import VLMEncoder
            self.encoder = VLMEncoder(cfg)
            self.adapter = Adapter(
                self.encoder.hidden_dim, fused_dim,
                dropout=cfg.get("adapter", {}).get("dropout", 0.1),
            )
        elif self.path == "dual_encoder":
            raise NotImplementedError(
                "dual_encoder path is built when SigLIP+RoBERTa is added (see plan)."
            )
        else:
            raise ValueError(f"unknown backbone.path {self.path!r}")

        task_adapter_cfg = model_cfg.get("task_adapters", {})
        self.use_task_adapters = bool(task_adapter_cfg.get("enabled", False))
        self.separate_bbox_adapter = self.use_task_adapters and bool(
            task_adapter_cfg.get("separate_bbox", False)
        )
        if self.use_task_adapters:
            kwargs = {
                "dim": fused_dim,
                "bottleneck": int(task_adapter_cfg.get("bottleneck", 128)),
                "dropout": float(task_adapter_cfg.get("dropout", 0.1)),
            }
            task_names = ["policy", "diagnosis", "memory", "recovery"]
            if self.separate_bbox_adapter:
                task_names.append("grounding")
            self.task_adapters = nn.ModuleDict({
                name: ResidualTaskAdapter(**kwargs)
                for name in task_names
            })
        else:
            self.task_adapters = nn.ModuleDict({
                name: nn.Identity()
                for name in ("policy", "diagnosis", "memory", "recovery")
            })

        self.failure_head = FailureHead(
            fused_dim, needs_recovery=self.hierarchical_recovery,
        )
        self.action_head = ActionHead(
            fused_dim,
            spatial_grounding=self.spatial_grounding,
            bbox_parameterization=model_cfg.get(
                "bbox_parameterization", "legacy_xywh",
            ),
            bbox_grounding_mode=model_cfg.get(
                "bbox_grounding_mode", "content_attention",
            ),
            bbox_size_parameterization=model_cfg.get(
                "bbox_size_parameterization", "sigmoid",
            ),
            bbox_log_size_min=float(model_cfg.get(
                "bbox_log_size_min", -9.210340371976184,
            )),
            bbox_log_size_max=float(model_cfg.get(
                "bbox_log_size_max", 0.0,
            )),
            bbox_attention_dropout=model_cfg.get("bbox_attention_dropout"),
        )
        self.memory_head = MemoryHead(fused_dim)
        self.recovery_outcome_head = RecoveryOutcomeHead(fused_dim)

    def encode(self, batch: dict, prefix: str = ""):
        """Front-end -> pooled features and, when enabled, spatial image tokens."""
        if self.path == "vlm":
            encoded = self.encoder(batch, prefix=prefix)
            return self._project_encoder_output(encoded)
        raise NotImplementedError

    def _project_encoder_output(self, encoded) -> dict:
        """Apply the shared adapter while preserving spatial metadata."""
        if not isinstance(encoded, dict):
            return {"fused": self.adapter(encoded.float())}
        result = {
            "fused": self.adapter(encoded["pooled"].float()),
            "spatial_tokens": self.adapter(encoded["spatial_tokens"].float()),
            "spatial_mask": encoded["spatial_mask"],
        }
        if "spatial_coords" in encoded:
            result["spatial_coords"] = encoded["spatial_coords"].float()
        return result

    @torch.inference_mode()
    def memory_embedding(self, batch: dict) -> torch.Tensor:
        """Return the post-action 768-d tensor consumed by ``memory_head``.

        This is an inference-only view of the existing causal route.  It does
        not call any prediction head, normalize the representation, or alter
        ``forward``.  Callers must put the complete model in evaluation mode so
        the returned tensor is exactly the deterministic memory-task-adapter
        input used by the selected checkpoint.
        """
        if self.training:
            raise RuntimeError("memory_embedding requires model.eval()")
        if not getattr(self, "causal_routing", False):
            raise RuntimeError(
                "memory_embedding requires the registered causal post-action route"
            )
        post_encoded = self.encode(batch, prefix="post_")
        memory_fused = self.task_adapters["memory"](post_encoded["fused"])
        if memory_fused.ndim != 2 or memory_fused.shape[-1] != 768:
            raise RuntimeError(
                "memory task-adapter representation must have shape [B, 768], "
                f"got {tuple(memory_fused.shape)}"
            )
        if not torch.isfinite(memory_fused).all():
            raise RuntimeError("memory task-adapter representation is non-finite")
        return memory_fused.detach()

    def forward(self, batch: dict) -> dict:
        if self.causal_routing:
            pre_encoded = self.encode(batch, prefix="pre_")
            post_encoded = self.encode(batch, prefix="post_")
        else:
            pre_encoded = post_encoded = self.encode(batch)

        adapters = getattr(self, "task_adapters", None)
        def adapt(name, value):
            return adapters[name](value) if adapters is not None else value
        pre_fused = adapt("policy", pre_encoded["fused"])
        post_fused = adapt("diagnosis", post_encoded["fused"])
        memory_fused = adapt("memory", post_encoded["fused"])

        preds: dict = {
            "fused": post_fused,  # outcome representation used by contrastive loss
            "fused_pre": pre_fused,
            "fused_post": post_fused,
        }
        if pre_encoded.get("spatial_mask") is not None:
            preds["spatial_token_counts"] = pre_encoded["spatial_mask"].sum(dim=1)
        failure_predictions = self.failure_head(
            post_fused,
            confidence_fused=pre_fused if self.causal_routing else None,
        )
        strategy_source_pre = batch.get("strategy_source_pre")
        if strategy_source_pre is not None:
            strategy_source_pre = strategy_source_pre.to(
                post_fused.device
            ).view(-1).bool()
        if strategy_source_pre is not None and strategy_source_pre.any():
            pre_diagnosis_fused = adapt(
                "diagnosis", pre_encoded["fused"]
            )
            pre_recovery = self.failure_head(
                pre_diagnosis_fused,
                confidence_fused=pre_fused if self.causal_routing else None,
            )["recovery"]
            failure_predictions["recovery"] = torch.where(
                strategy_source_pre[:, None],
                pre_recovery,
                failure_predictions["recovery"],
            )
        preds.update(failure_predictions)
        bbox_context = (
            torch.autocast("cuda", enabled=False)
            if self.bbox_fp32_grounding else contextlib.nullcontext()
        )
        with bbox_context:
            bbox_source = (
                pre_encoded["fused"].float()
                if self.bbox_fp32_grounding else pre_encoded["fused"]
            )
            bbox_fused = (
                adapt("grounding", bbox_source)
                if self.separate_bbox_adapter else pre_fused
            )
            spatial_tokens = pre_encoded.get("spatial_tokens")
            if spatial_tokens is not None:
                if self.bbox_fp32_grounding:
                    spatial_tokens = spatial_tokens.float()
                spatial_tokens = adapt(
                    "grounding" if self.separate_bbox_adapter else "policy",
                    spatial_tokens,
                )
        if getattr(self, "spatial_grounding", False):
            preds.update(self.action_head(
                pre_fused,
                bbox_fused=bbox_fused,
                spatial_tokens=spatial_tokens,
                spatial_mask=pre_encoded.get("spatial_mask"),
                spatial_coords=pre_encoded.get("spatial_coords"),
                force_fp32_bbox=self.bbox_fp32_grounding,
            ))
        else:
            preds.update(self.action_head(pre_fused))
        memory_predictions = self.memory_head(memory_fused)
        if strategy_source_pre is not None and strategy_source_pre.any():
            pre_memory_fused = adapt("memory", pre_encoded["fused"])
            pre_memory_recovery = self.memory_head(
                pre_memory_fused
            )["memory_recovery"]
            memory_predictions["memory_recovery"] = torch.where(
                strategy_source_pre[:, None],
                pre_memory_recovery,
                memory_predictions["memory_recovery"],
            )
        preds.update(memory_predictions)
        if getattr(self, "use_recovery_transitions", False):
            if "recovery_input_ids" in batch:
                recovery_encoded = self.encode(batch, prefix="recovery_")
                recovery_fused = adapt("recovery", recovery_encoded["fused"])
                preds.update(self.recovery_outcome_head(recovery_fused))
                preds["recovery_row_indices"] = batch["recovery_row_indices"]
            else:
                preds["recovery_outcome"] = post_fused.new_empty((0, 1))
                preds["recovery_row_indices"] = post_fused.new_empty(
                    (0,), dtype=torch.long,
                )
        else:
            preds.update(self.recovery_outcome_head(post_fused))
        return preds

    def forward_bbox(self, batch: dict, *, force_fp32_grounding: bool = False) -> dict:
        """Run only the causal pre-action stream and bbox grounding branch.

        This is used by the registered bbox micro-overfit gates. It avoids
        spending compute on post-action and recovery streams while proving
        localization can learn.
        """
        if force_fp32_grounding:
            # Keep the frozen/quantized VLM in its native T4 precision, then run
            # every trainable localization component in FP32. This prevents the
            # scaled-FP16 gradient overflow observed in the v2.2 probe.
            with torch.autocast("cuda", dtype=self.encoder.dtype):
                encoded = self.encoder(batch, prefix="pre_")
            with torch.autocast("cuda", enabled=False):
                pre_encoded = self._project_encoder_output(encoded)
        else:
            pre_encoded = self.encode(batch, prefix="pre_")
        adapters = getattr(self, "task_adapters", None)

        def adapt(name, value):
            return adapters[name](value) if adapters is not None else value

        pre_fused = adapt("policy", pre_encoded["fused"])
        bbox_fused = (
            adapt("grounding", pre_encoded["fused"])
            if self.separate_bbox_adapter else pre_fused
        )
        spatial_tokens = pre_encoded.get("spatial_tokens")
        if spatial_tokens is not None:
            spatial_tokens = adapt(
                "grounding" if self.separate_bbox_adapter else "policy",
                spatial_tokens,
            )
        context = (
            torch.autocast("cuda", enabled=False)
            if force_fp32_grounding else contextlib.nullcontext()
        )
        with context:
            return self.action_head(
                pre_fused,
                bbox_fused=bbox_fused,
                spatial_tokens=spatial_tokens,
                spatial_mask=pre_encoded.get("spatial_mask"),
                spatial_coords=pre_encoded.get("spatial_coords"),
                force_fp32_bbox=force_fp32_grounding,
            )

    def trainable_parameters(self):
        """All params with requires_grad: LoRA + adapter + 5 heads (4-bit base frozen)."""
        return [p for p in self.parameters() if p.requires_grad]

    def head_parameters(self):
        """Adapter + the 5 heads (the high-LR param group)."""
        mods = [self.adapter, self.task_adapters, self.failure_head, self.action_head,
                self.memory_head, self.recovery_outcome_head]
        return [p for m in mods for p in m.parameters() if p.requires_grad]

    def lora_parameters(self):
        """LoRA params inside the VLM encoder (the low-LR param group)."""
        return [p for p in self.encoder.parameters() if p.requires_grad]
