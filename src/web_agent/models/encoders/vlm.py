"""Unified VLM encoder -> pooled [B, D] (SPEC 4.5, the adapter path).

Foundation backbone: Qwen2-VL-2B-Instruct in 4-bit QLoRA. The VLM fuses image+text
internally, so CROSS-ATTENTION IS SKIPPED — a single adapter Linear(D->768)
(see ../adapter.py) maps the pooled output into the shared heads. D is read from
the model config at load (1536 for Qwen2-VL-2B).

QLoRA: 4-bit base (frozen) + LoRA on q_proj/v_proj (trainable). When LoRA is
applied the base stays frozen but the LoRA params require grad, so forward runs
WITH the autograd graph (no_grad only while fully frozen, e.g. a frozen smoke run).

Requires transformers (recent), bitsandbytes, accelerate, peft. The Kaggle install
cell provides them.
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

        self.model.config.use_cache = False

        if bb.get("qlora"):
            self._apply_qlora(bb)
        else:
            # Fully frozen (e.g. a quick smoke check). Forward runs under no_grad.
            for p in self.model.parameters():
                p.requires_grad = False

    def _apply_qlora(self, bb: dict) -> None:
        """4-bit base frozen + LoRA on attention projections (trainable)."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Enables gradient checkpointing + makes inputs require grad so gradients
        # flow back to the LoRA layers through the frozen 4-bit base.
        self.model = prepare_model_for_kbit_training(
            self.model, use_gradient_checkpointing=True,
        )
        lora = LoraConfig(
            r=bb.get("lora_rank", 8),
            lora_alpha=bb.get("lora_alpha", 16),
            lora_dropout=bb.get("lora_dropout", 0.05),
            target_modules=bb.get("lora_target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def _frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.parameters())

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
