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

        # Pooling of the [B, seq, D] last hidden state into [B, D]. Qwen2.5-VL is a
        # causal decoder, so the integrated summary lives at the LAST non-pad token,
        # not in a uniform mean (which is dominated by the hundreds of vision-patch
        # tokens). "last" is the default; "mean" and "attention" are kept for the
        # pooling ablation (see docs plan P0-A).
        self.pooling = bb.get("pooling", "last")
        if self.pooling not in ("mean", "last", "attention"):
            raise ValueError(f"unknown pooling {self.pooling!r}")
        if self.pooling == "attention":
            # tiny learned attention pool; trainable, lives in the LoRA param group.
            self.attn_pool = nn.Linear(self.hidden_dim, 1)

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
            gradient_checkpointing_kwargs={"use_reentrant": False},
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
        kwargs = dict(
            input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
            pixel_values=batch["pixel_values"].to(dev, dtype=self.dtype),
            image_grid_thw=batch["image_grid_thw"].to(dev),
            output_hidden_states=True,
            use_cache=False,
        )
        if "mm_token_type_ids" in batch:  # required by newer Qwen2-VL (M-RoPE)
            kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(dev)
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            out = self.model(**kwargs)
            h = out.hidden_states[-1]                      # [B, seq, D]
            am = batch["attention_mask"].to(dev)           # [B, seq] (1=real, 0=pad)
            pooled = self._pool(h, am)                     # [B, D]
        return pooled.float()

    def _pool(self, h: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        """Pool [B, seq, D] -> [B, D] per self.pooling. am = attention mask [B, seq]."""
        if self.pooling == "mean":
            m = am.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.pooling == "last":
            # right-padded -> last real token index = (#real - 1)
            last = am.sum(dim=1).clamp(min=1).long() - 1          # [B]
            return h[torch.arange(h.size(0), device=h.device), last]
        # attention pool: masked softmax over a learned token score (computed in fp32)
        scores = self.attn_pool(h.float()).squeeze(-1)            # [B, seq]
        scores = scores.masked_fill(am == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).to(h.dtype).unsqueeze(-1)  # [B, seq, 1]
        return (h * w).sum(dim=1)
