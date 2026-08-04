"""Unified VLM encoder -> pooled [B, D] (SPEC 4.5, the adapter path).

Official Hugging Face Qwen2/2.5-VL and InternVL-HF backbones fuse image and text
internally, so cross-attention is skipped. A single adapter Linear(D->768)
(see ../adapter.py) maps the pooled output into the unchanged shared heads. D
is validated against the pinned model revision at load time.

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

from web_agent.models.encoders.vlm_contract import get_vlm_contract
from web_agent.models.spatial import (
    normalized_fixed_square_coordinates,
    normalized_spatial_coordinates,
)


class VLMEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # The official Hugging Face formats for Qwen2/2.5-VL and InternVL3.5
        # share this auto-model API. Family-specific tensor differences stay in
        # VLMContract instead of being scattered through the encoder.
        from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

        bb = cfg["backbone"]
        self.contract = get_vlm_contract(cfg)
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
        load_kwargs = {
            "quantization_config": quant,
            "dtype": self.dtype,
            "device_map": {"": 0},
            "revision": bb.get("revision", "main"),
            "trust_remote_code": bool(bb.get("trust_remote_code", False)),
            "low_cpu_mem_usage": True,
        }
        if bb.get("attn_implementation"):
            load_kwargs["attn_implementation"] = bb["attn_implementation"]
        self.model = AutoModelForImageTextToText.from_pretrained(
            bb["vlm_model"], **load_kwargs,
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
        expected_hidden_dim = bb.get("vlm_hidden_dim")
        if (
            expected_hidden_dim is not None
            and int(expected_hidden_dim) != int(self.hidden_dim)
        ):
            raise RuntimeError(
                "configured VLM hidden size does not match the downloaded "
                f"revision: expected {expected_hidden_dim}, got {self.hidden_dim}"
            )

        self.model.config.use_cache = False
        self.preserve_spatial_tokens = bool(bb.get("preserve_spatial_tokens", False))
        self.coordinate_spatial_tokens = (
            cfg.get("model", {}).get("bbox_grounding_mode")
            == "coordinate_softargmax"
        )
        vision_conf = getattr(conf, "vision_config", None)
        self.spatial_merge_size = int(
            getattr(vision_conf, "spatial_merge_size", 2)
        )
        self.image_seq_length = int(
            bb.get(
                "image_seq_length",
                getattr(conf, "image_seq_length", 0) or 0,
            )
        )
        text_conf = getattr(conf, "text_config", None)
        self.image_token_id = (
            getattr(conf, "image_token_id", None)
            or getattr(text_conf, "image_token_id", None)
        )

        # Pooling of the [B, seq, D] last hidden state into [B, D]. Qwen2.5-VL is a
        # causal decoder, so the integrated summary lives at the LAST non-pad token,
        # not in a uniform mean (which is dominated by the hundreds of vision-patch
        # tokens). "last" is the default; "mean" and "attention" are kept for the
        # pooling ablation (see docs plan P0-A).
        self.pooling = bb.get("pooling", "last")
        self.forward_logits_to_keep = bb.get("forward_logits_to_keep")
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

    def forward(self, batch: dict, prefix: str = "") -> torch.Tensor | dict:
        """Encode one VLM stream; causal gold batches use pre_ and post_ prefixes."""
        dev = self.device
        kwargs = dict(
            input_ids=batch[f"{prefix}input_ids"].to(dev),
            attention_mask=batch[f"{prefix}attention_mask"].to(dev),
            output_hidden_states=True,
            use_cache=False,
        )
        for name in self.contract.model_input_keys:
            if name in {"input_ids", "attention_mask"}:
                continue
            key = f"{prefix}{name}"
            if key not in batch:
                continue
            value = batch[key].to(dev)
            if name == "pixel_values":
                value = value.to(dtype=self.dtype)
            kwargs[name] = value
        logits_to_keep = self.forward_logits_to_keep
        if logits_to_keep is not None:
            kwargs["logits_to_keep"] = int(logits_to_keep)
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            out = self.model(**kwargs)
            h = out.hidden_states[-1]                      # [B, seq, D]
            am = batch[f"{prefix}attention_mask"].to(dev)  # [B, seq] (1=real, 0=pad)
            pooled = self._pool(h, am)                     # [B, D]
        if not self.preserve_spatial_tokens:
            return pooled.float()
        input_ids = batch[f"{prefix}input_ids"].to(dev)
        if self.image_token_id is None:
            spatial_mask = am.bool()
        else:
            spatial_mask = input_ids.eq(int(self.image_token_id)) & am.bool()
            empty = ~spatial_mask.any(dim=1)
            if empty.any():
                spatial_mask[empty] = am[empty].bool()
        result = {
            "pooled": pooled.float(),
            "spatial_tokens": h.float(),
            "spatial_mask": spatial_mask,
        }
        if self.coordinate_spatial_tokens:
            if self.contract.coordinate_mode == "qwen_grid":
                result["spatial_coords"] = normalized_spatial_coordinates(
                    spatial_mask,
                    batch[f"{prefix}image_grid_thw"].to(dev),
                    spatial_merge_size=self.spatial_merge_size,
                )
            elif self.contract.coordinate_mode == "fixed_square":
                if self.image_seq_length <= 0:
                    raise RuntimeError(
                        "fixed-square VLM requires a positive image_seq_length"
                    )
                result["spatial_coords"] = normalized_fixed_square_coordinates(
                    spatial_mask,
                    batch[f"{prefix}image_counts"].to(dev),
                    tokens_per_image=self.image_seq_length,
                )
            else:
                raise RuntimeError(
                    f"unsupported coordinate mode {self.contract.coordinate_mode!r}"
                )
        return result

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
