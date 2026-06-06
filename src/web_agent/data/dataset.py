"""WebAgentDataset — one record -> tensors (IMPLEMENTATION_PLAN 2.2).

Stable core (hybrid plan C): written once, reused by every model and every
training loop in the notebook. Backbone-agnostic — the image processor and text
tokenizer are INJECTED (not hardcoded), so the same Dataset serves SigLIP/CLIP/
Florence/VLM configs.

Per record:
  image : open state_before (UTF-8 paths), processor -> pixel_values [3,H,W]
  text  : "task_description [SEP] action_target_desc [SEP] website_domain"
          tokenized to max 128 (pad/truncate)
  bbox  : {x,y,width,height} pixels -> normalized [0,1] by the image's own size,
          with bbox_mask = 0 when missing (3,805 rows) so the loss can skip it
  labels: encoded via web_agent.labels (the single source of truth)

Returned dict (all tensors):
  pixel_values, input_ids, attention_mask,
  bbox [4], bbox_mask [1], borrowed [1],
  label_outcome, label_failtype, label_action, label_recovery,
  label_memory, label_confidence
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

SEP = " </s> "  # RoBERTa separator; tokenizer adds specials too, this just joins fields.


class WebAgentDataset(Dataset):
    def __init__(self, records, cfg, processor, tokenizer):
        self.records = records
        self.cfg = cfg
        self.processor = processor
        self.tokenizer = tokenizer
        self.root = Path(cfg["data"]["root"])
        self.max_len = cfg["data"].get("text_max_len", 128)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> Image.Image:
        return Image.open(self.root / rel_path).convert("RGB")

    def _pixel_values(self, image: Image.Image) -> torch.Tensor:
        # Processor handles resize + normalize for whichever vision backbone is used.
        out = self.processor(images=image, return_tensors="pt")
        return out["pixel_values"][0]

    def _text_tokens(self, rec: dict):
        text = SEP.join([
            rec.get("task_description", "") or "",
            rec.get("action_target_desc", "") or "",
            rec.get("website_domain", "") or "",
        ])
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"][0], enc["attention_mask"][0]

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

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = self._load_image(rec["state_before"])
        img_w, img_h = image.size

        pixel_values = self._pixel_values(image)
        input_ids, attention_mask = self._text_tokens(rec)
        bbox, bbox_mask = self._bbox(rec, img_w, img_h)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
