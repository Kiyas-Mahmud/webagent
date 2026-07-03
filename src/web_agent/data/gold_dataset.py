"""GoldDataset — the real, leak-safe gold dataset (separate from the synthetic path).

Kept fully separate from WebAgentDataset so the synthetic 70k pipeline stays
byte-for-byte unchanged. Reuses the SAME tensor keys, so the shared model / loss /
trainer / vlm_collate run on gold with no edits.

Handles BOTH gold layouts in one code path:
  - v12 (current): nested  {"inputs": {...}, "labels": {...}, "meta": {...}}
  - v8  (pilot):   flat     {state_before, outcome_label, ...}

Honest input = state_before + state_after images + task_description (+ website_domain).
The split files carry no ssim/pixel_diff/url_after, so no scalar can leak.

v12 provides real confidence + memory labels -> those heads are enabled in the gold
config. recovery_success is too sparse -> that head stays disabled (placeholder).
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


def view(rec: dict):
    """Return (inputs, labels, meta) for either layout.

    v12 nested -> the three sub-dicts. v8 flat -> the record itself for all three
    (every field sits at the top level, so the same .get() calls work).
    """
    if "inputs" in rec and "labels" in rec:
        return rec["inputs"], rec["labels"], rec.get("meta", {})
    return rec, rec, rec


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

    def _bbox(self, lab: dict, img_w: int, img_h: int):
        """Return (bbox[4] normalized, mask[1]); zeros + mask 0 when bbox is null."""
        b = lab.get("action_target_bbox")
        if not b:
            return torch.zeros(4, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        bbox = torch.tensor([
            b["x"] / img_w,
            b["y"] / img_h,
            b["width"] / img_w,
            b["height"] / img_h,
        ], dtype=torch.float32).clamp_(0.0, 1.0)
        return bbox, torch.ones(1, dtype=torch.float32)

    def _context_text(self, inp: dict) -> str:
        task = inp.get("task_description", "") or ""
        domain = inp.get("website_domain", "") or ""
        return f"current_task: {task} | domain: {domain}"

    def _vlm_inputs(self, inp: dict, images):
        """images: list of PIL images (1 = before only, 2 = before+after)."""
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": self._context_text(inp)})
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

    def _labels(self, lab: dict, meta: dict, bbox, bbox_mask) -> dict:
        return {
            "bbox": bbox,
            "bbox_mask": bbox_mask,
            "borrowed": torch.tensor([0.0]),
            "label_outcome": torch.tensor(EXECUTION_OUTCOME[lab["outcome_label"]]),
            "label_failtype": torch.tensor(FAILURE_TYPE[lab["failure_type_4"]]),
            "label_action": torch.tensor(ACTION_TYPE[lab["action_type"]]),
            "label_recovery": torch.tensor(RECOVERY_STRATEGY[lab["recovery_strategy"]]),
            # v12 has real confidence + memory; v8 -> neutral default (head disabled there).
            "label_memory": torch.tensor([float(lab.get("memory_update_flag") or False)]),
            "label_confidence": torch.tensor([float(
                lab["agent_confidence_before"]
                if lab.get("agent_confidence_before") is not None else 0.5)]),
            # recovery_success sparse -> head stays disabled; placeholder keeps collate happy.
            "label_recovery_success": torch.tensor([float(lab.get("recovery_success") or False)]),
            "original_task_id": meta.get("task_id", ""),
        }

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        inp, lab, meta = view(rec)
        before = self._load_image(inp["state_before"])
        img_w, img_h = before.size
        bbox, bbox_mask = self._bbox(lab, img_w, img_h)

        images = [before]
        if self.use_state_after and inp.get("state_after"):
            images.append(self._load_image(inp["state_after"]))

        out = self._vlm_inputs(inp, images)
        out.update(self._labels(lab, meta, bbox, bbox_mask))
        return out
