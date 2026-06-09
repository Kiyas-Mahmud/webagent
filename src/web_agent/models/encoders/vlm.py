"""Unified VLM encoder -> pooled [B, D] (SPEC 4.5, the adapter path).

Foundation backbone (Qwen-first build order): Qwen2.5-VL-3B-Instruct in 4-bit.
The VLM fuses image+text internally, so CROSS-ATTENTION IS SKIPPED — a single
adapter Linear(D->768) (see ../adapter.py) maps the pooled output into the shared
heads. D = model.config.hidden_size (2048 for Qwen2.5-VL-3B; read at load).

Requires transformers >= 4.49 for the Qwen2.5-VL classes, and bitsandbytes +
accelerate for 4-bit. The Kaggle install cell upgrades transformers.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class VLMEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # AutoModelForImageTextToText auto-selects the right class for any
        # Qwen-VL (Qwen2-VL-2B, Qwen2.5-VL-3B, ...) — backbone-agnostic loader.
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

        bb = cfg["backbone"]
        # T4 (Turing) has no native bf16 — default to fp16; 4-bit uses its own
        # compute dtype.
        self.dtype = getattr(torch, bb.get("dtype", "float16"))

        quant = None
        if bb.get("load_in_4bit"):
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(
            bb["vlm_model"],
            quantization_config=quant,
            torch_dtype=self.dtype,
            device_map={"": 0},          # 2B/3B fits a single T4
        )
        # Qwen2.5-VL nests the LM dims under config.text_config (no top-level
        # hidden_size). Fall back across layouts to stay robust.
        conf = self.model.config
        self.hidden_dim = (
            getattr(conf, "hidden_size", None)
            or getattr(getattr(conf, "text_config", None), "hidden_size", None)
        )
        if self.hidden_dim is None:
            raise RuntimeError("could not resolve VLM hidden size from config")

        # Phase-1 smoke/mini: freeze the VLM, train only adapter + heads.
        # QLoRA (LoRA on q_proj/v_proj) is added later for the full run.
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.config.use_cache = False
        # When fully frozen we don't need the autograd graph through the VLM —
        # run it under no_grad to save VRAM on the T4. Once QLoRA adds trainable
        # params this flips automatically.
        self._grad_checkpoint_enabled = False

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def _frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

    def enable_qlora_grad_checkpointing(self) -> None:
        """Call once LoRA params are added, so the VLM keeps gradients efficiently."""
        self.model.gradient_checkpointing_enable()
        self._grad_checkpoint_enabled = True

    def forward(self, batch: dict) -> torch.Tensor:
        dev = self.device
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            out = self.model(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev),
                pixel_values=batch["pixel_values"].to(dev, dtype=self.dtype),
                image_grid_thw=batch["image_grid_thw"].to(dev),
                output_hidden_states=True,
                use_cache=False,
            )
            h = out.hidden_states[-1]                      # [B, seq, D]
            mask = batch["attention_mask"].to(dev).unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, D]
        return pooled.float()
