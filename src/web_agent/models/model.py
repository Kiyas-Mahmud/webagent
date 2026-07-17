"""WebAgentModel — assembles front-end + FIXED heads into one nn.Module (ARCH 1).

Reads cfg["backbone"]["path"]:
  vlm          -> VLMEncoder -> Adapter(D->768)                      -> fused[768]
  dual_encoder -> VisionEncoder + TextEncoder -> CrossAttentionFusion -> fused[768]

Then the SAME FailureHead + ActionHead + MemoryHead return one flat prediction
dict consumed by CombinedLoss. Gold runs can enable causal routing: the shared
encoder/adapter creates a pre-action embedding and a post-action embedding, then
routes each head to information that exists when its prediction is made.

Qwen-first build order: the `vlm` path is implemented now; the `dual_encoder`
path is added when SigLIP+RoBERTa is built.
"""

from __future__ import annotations

import torch.nn as nn

from web_agent.models.adapter import Adapter
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

        self.failure_head = FailureHead(fused_dim)
        self.action_head = ActionHead(fused_dim)
        self.memory_head = MemoryHead(fused_dim)
        self.recovery_outcome_head = RecoveryOutcomeHead(fused_dim)

    def encode(self, batch: dict, prefix: str = ""):
        """Front-end -> fused [B, 768]."""
        if self.path == "vlm":
            return self.adapter(self.encoder(batch, prefix=prefix))
        raise NotImplementedError

    def forward(self, batch: dict) -> dict:
        if self.causal_routing:
            pre_fused = self.encode(batch, prefix="pre_")
            post_fused = self.encode(batch, prefix="post_")
        else:
            pre_fused = post_fused = self.encode(batch)

        preds: dict = {
            "fused": post_fused,  # outcome representation used by contrastive loss
            "fused_pre": pre_fused,
            "fused_post": post_fused,
        }
        preds.update(self.failure_head(
            post_fused,
            confidence_fused=pre_fused if self.causal_routing else None,
        ))
        preds.update(self.action_head(pre_fused))
        preds.update(self.memory_head(post_fused))
        preds.update(self.recovery_outcome_head(post_fused))
        return preds

    def trainable_parameters(self):
        """All params with requires_grad: LoRA + adapter + 5 heads (4-bit base frozen)."""
        return [p for p in self.parameters() if p.requires_grad]

    def head_parameters(self):
        """Adapter + the 5 heads (the high-LR param group)."""
        mods = [self.adapter, self.failure_head, self.action_head,
                self.memory_head, self.recovery_outcome_head]
        return [p for m in mods for p in m.parameters() if p.requires_grad]

    def lora_parameters(self):
        """LoRA params inside the VLM encoder (the low-LR param group)."""
        return [p for p in self.encoder.parameters() if p.requires_grad]
