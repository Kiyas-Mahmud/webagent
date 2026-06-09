"""WebAgentDataset — one record -> tensors (IMPLEMENTATION_PLAN 2.2).

Stable core (hybrid plan C): written once, reused by every model. Backbone-agnostic
via two paths, selected by cfg["backbone"]["path"]:

  dual_encoder : separate image + text (SigLIP pixel_values + RoBERTa input_ids).
  vlm          : Qwen2.5-VL processes image+text JOINTLY with its own processor
                 (chat template + vision placeholder tokens). Returns input_ids,
                 attention_mask, pixel_values (flattened patches), image_grid_thw.

Both paths share the SAME label encoding (web_agent.labels) and the SAME bbox
normalization, so the 4 heads + combined loss never change.

Per record:
  bbox  : {x,y,width,height} px -> normalized [0,1] by the image's own size,
          with bbox_mask = 0 when missing so the loss can skip it.
  labels: outcome/failtype/action/recovery (int64), memory/confidence (float32).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from web_agent.labels import (
    ACTION_TYPE,
    EXECUTION_OUTCOME,
    FAILURE_TYPE,
    RECOVERY_STRATEGY,
)

SEP = " </s> "  # dual-encoder field separator (RoBERTa).


class WebAgentDataset(Dataset):
    def __init__(self, records, cfg, processor, tokenizer=None):
        self.records = records
        self.cfg = cfg
        self.processor = processor
        self.tokenizer = tokenizer
        self.root = Path(cfg["data"]["root"])
        self.max_len = cfg["data"].get("text_max_len", 128)
        self.path = cfg["backbone"]["path"]  # "vlm" or "dual_encoder"

    def __len__(self) -> int:
        return len(self.records)

    # ---- shared helpers ----
    def _load_image(self, rel_path: str) -> Image.Image:
        return Image.open(self.root / rel_path).convert("RGB")

    def _bbox(self, rec: dict, img_w: int, img_h: int):
        """Return (bbox[4] normalized, mask[1]). Zeros + mask=0 when bbox missing."""
        b = rec.get("action_target_bbox")
        if not b:
            return torch.zeros(4, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        bbox = torch.tensor([
            b["x"] / img_w,
            b["y"] / img_h,
            b["width"] / img_w,
            b["height"] / img_h,
        ], dtype=torch.float32).clamp_(0.0, 1.0)
        return bbox, torch.ones(1, dtype=torch.float32)

    def _labels(self, rec: dict, bbox, bbox_mask) -> dict:
        return {
            "bbox": bbox,
            "bbox_mask": bbox_mask,
            "borrowed": torch.tensor([float(rec.get("borrowed_image", False))]),
            "label_outcome": torch.tensor(EXECUTION_OUTCOME[rec["execution_outcome"]]),
            "label_failtype": torch.tensor(FAILURE_TYPE[rec["failure_type"]]),
            "label_action": torch.tensor(ACTION_TYPE[rec["action_type"]]),
            "label_recovery": torch.tensor(RECOVERY_STRATEGY[rec["recovery_strategy"]]),
            "label_memory": torch.tensor([float(rec["memory_update_flag"])]),
            "label_confidence": torch.tensor([float(rec["agent_confidence_before"])]),
        }

    def _prompt(self, rec: dict) -> str:
        return (
            f"Task: {rec.get('task_description', '') or ''}\n"
            f"Target: {rec.get('action_target_desc', '') or ''}\n"
            f"Website: {rec.get('website_domain', '') or ''}"
        )

    # ---- dual-encoder path ----
    def _dual_inputs(self, image, rec):
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        text = SEP.join([
            rec.get("task_description", "") or "",
            rec.get("action_target_desc", "") or "",
            rec.get("website_domain", "") or "",
        ])
        enc = self.tokenizer(
            text, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
        }

    # ---- VLM path (Qwen2.5-VL joint processing) ----
    def _vlm_inputs(self, image, rec):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": self._prompt(rec)}],
        }]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.processor(text=[chat], images=[image], return_tensors="pt")
        return {
            "input_ids": enc["input_ids"][0],            # [seq]
            "attention_mask": enc["attention_mask"][0],  # [seq]
            "pixel_values": enc["pixel_values"],         # [num_patches, patch_dim]
            "image_grid_thw": enc["image_grid_thw"][0],  # [3] = (t, h, w)
        }

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = self._load_image(rec["state_before"])
        img_w, img_h = image.size
        bbox, bbox_mask = self._bbox(rec, img_w, img_h)

        inputs = self._vlm_inputs(image, rec) if self.path == "vlm" else self._dual_inputs(image, rec)
        inputs.update(self._labels(rec, bbox, bbox_mask))
        return inputs
