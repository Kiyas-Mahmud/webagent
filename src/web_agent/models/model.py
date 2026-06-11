"""WebAgentModel — assembles front-end + FIXED heads into one nn.Module (ARCH 1).

Reads cfg["backbone"]["path"]:
  vlm          -> VLMEncoder -> Adapter(D->768)                      -> fused[768]
  dual_encoder -> VisionEncoder + TextEncoder -> CrossAttentionFusion -> fused[768]

Then the SAME FailureHead + ActionHead + MemoryHead run on fused[768] and the
forward returns one flat dict of all predictions, consumed by CombinedLoss.

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

    def encode(self, batch: dict):
        """Front-end -> fused [B, 768]."""
        if self.path == "vlm":
            return self.adapter(self.encoder(batch))
        raise NotImplementedError

    def forward(self, batch: dict) -> dict:
        fused = self.encode(batch)
        preds: dict = {"fused": fused}           # exposed for the contrastive loss
        preds.update(self.failure_head(fused))   # outcome, failure_type, confidence, recovery
        preds.update(self.action_head(fused))    # action_type, bbox
        preds.update(self.memory_head(fused))    # memory_flag, memory_recovery
        preds.update(self.recovery_outcome_head(fused))  # recovery_outcome
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
