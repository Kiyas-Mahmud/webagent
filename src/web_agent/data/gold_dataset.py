"""GoldDataset — the real, leak-safe gold dataset (separate from the synthetic path).

Kept fully separate from WebAgentDataset so the synthetic 70k pipeline stays
byte-for-byte unchanged (supervisor constraint). Reuses the SAME tensor keys, so
the shared model / loss / trainer / vlm_collate run on gold with no edits.

Gold schema (already leak-stripped by the data team) -> model keys:
  outcome_label   -> label_outcome      (EXECUTION_OUTCOME)
  failure_type_4  -> label_failtype     (FAILURE_TYPE)
  action_type     -> label_action       (ACTION_TYPE)
  recovery_strategy -> label_recovery   (RECOVERY_STRATEGY)
  action_target_bbox -> bbox (+ mask 0 when null)

Honest input = state_before + state_after images + task_description ONLY. The split
files contain no ssim/pixel_diff/url_after, so no scalar can leak.

Gold has NO labels for confidence / memory / recovery_success. We emit neutral
placeholders ONLY so collate/loss don't KeyError; those three loss terms are set to
weight 0 in configs/backbones/qwen2vl_2b_gold.yaml, so the placeholders never train.
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


class GoldDataset(Dataset):
    def __init__(self, records, cfg, processor):
        self.records = records
        self.cfg = cfg
        self.processor = processor
        self.root = Path(cfg["data"]["root"])
        self.use_state_after = bool(cfg["data"].get("use_state_after", True))

    def __len__(self) -> int:
        return len(self.records)

    # ---- helpers ----
    def _load_image(self, rel_path: str) -> Image.Image:
        return Image.open(self.root / rel_path).convert("RGB")

    def _bbox(self, rec: dict, img_w: int, img_h: int):
        """Return (bbox[4] normalized, mask[1]); zeros + mask 0 when bbox is null."""
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

    def _context_text(self, rec: dict) -> str:
        # gold has no domain/target-desc worth feeding; task only.
        return f"current_task: {rec.get('task_description', '') or ''}"

    def _vlm_inputs(self, rec: dict, images):
        """images: list of PIL images (1 = before only, 2 = before+after)."""
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": self._context_text(rec)})
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.processor(text=[chat], images=list(images), return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],     # [n_img, 3]
        }
        if "mm_token_type_ids" in enc:
            out["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
        return out

    def _labels(self, rec: dict, bbox, bbox_mask) -> dict:
        return {
            "bbox": bbox,
            "bbox_mask": bbox_mask,
            "borrowed": torch.tensor([0.0]),
            "label_outcome": torch.tensor(EXECUTION_OUTCOME[rec["outcome_label"]]),
            "label_failtype": torch.tensor(FAILURE_TYPE[rec["failure_type_4"]]),
            "label_action": torch.tensor(ACTION_TYPE[rec["action_type"]]),
            "label_recovery": torch.tensor(RECOVERY_STRATEGY[rec["recovery_strategy"]]),
            # neutral placeholders — loss weights for these 3 heads are 0 on gold.
            "label_memory": torch.tensor([0.0]),
            "label_confidence": torch.tensor([0.5]),
            "label_recovery_success": torch.tensor([0.0]),
            "original_task_id": rec.get("task_id", ""),
        }

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        before = self._load_image(rec["state_before"])
        img_w, img_h = before.size
        bbox, bbox_mask = self._bbox(rec, img_w, img_h)

        images = [before]
        if self.use_state_after and rec.get("state_after"):
            images.append(self._load_image(rec["state_after"]))

        inputs = self._vlm_inputs(rec, images)
        inputs.update(self._labels(rec, bbox, bbox_mask))
        return inputs
