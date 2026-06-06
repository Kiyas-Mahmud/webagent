"""Unified VLM wrapper -> pooled [B, D] (SPEC 4.5, the adapter path).

Y4 Qwen2.5-VL-0.5B, Y6 InternVL2-2B (4-bit), Y5 Qwen2.5-VL-3B (QLoRA). The VLM
fuses image+text internally, so CROSS-ATTENTION IS SKIPPED — a single adapter
Linear(D->768) (see ../adapter.py) maps the pooled output into the shared heads.
"""

from __future__ import annotations

import torch.nn as nn


class VLMEncoder(nn.Module):
    def __init__(self, model_name: str, load_in_4bit: bool = False, qlora: bool = False):
        super().__init__()
        # TODO(Phase 4/8): load VLM (optionally 4-bit/QLoRA), accept native image+text
        #   format, return pooled hidden state [B, D]. Enable grad checkpointing for T4.
        raise NotImplementedError("Build in Phase 4 (Y4 first; see docs/MODEL_TRAINING_PLAN.md §4).")

    def forward(self, image, text):
        raise NotImplementedError
