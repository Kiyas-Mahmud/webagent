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

from collections import defaultdict
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
CONTEXT_STEPS = 3  # how many previous steps of the trajectory to include


def _step_index(task_id: str) -> int:
    """Trailing _N of a task_id, e.g. 'train_pass1_<uuid>_0' -> 0."""
    try:
        return int(task_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


class WebAgentDataset(Dataset):
    def __init__(self, records, cfg, processor, tokenizer=None):
        self.records = records
        self.cfg = cfg
        self.processor = processor
        self.tokenizer = tokenizer
        self.root = Path(cfg["data"]["root"])
        self.max_len = cfg["data"].get("text_max_len", 128)
        self.path = cfg["backbone"]["path"]  # "vlm" or "dual_encoder"
        # Failure is often only visible in the post-action state. When enabled (and
        # available) feed state_before + state_after to the failure pillar; this is
        # the input ablation in the plan (P0-B). VLM path only.
        self.use_state_after = bool(cfg["data"].get("use_state_after", False))
        self._build_trajectory_index()

    def _build_trajectory_index(self) -> None:
        """Map each row -> its up-to-3 previous steps in the same trajectory.

        Trajectory = (original_task_id, pass); steps ordered by the task_id suffix.
        """
        groups: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for i, r in enumerate(self.records):
            key = (r.get("original_task_id"), r.get("pass"))
            groups[key].append((_step_index(r["task_id"]), i))
        self._prev_idx: dict[int, list[int]] = {}
        for lst in groups.values():
            lst.sort()
            ordered = [i for _, i in lst]
            for pos, i in enumerate(ordered):
                self._prev_idx[i] = ordered[max(0, pos - CONTEXT_STEPS):pos]

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
            "label_recovery_success": torch.tensor([float(rec.get("recovery_success", False))]),
            # str field for the contrastive pair sampler / loss (carried by collate).
            "original_task_id": rec.get("original_task_id", ""),
        }

    def _context_text(self, idx: int) -> str:
        """Last-3-steps context + current step (spec format), empty-padded at start."""
        prev = self._prev_idx.get(idx, [])
        prev = [None] * (CONTEXT_STEPS - len(prev)) + prev  # left-pad to 3
        parts = []
        for p in prev:
            if p is None:
                parts.append("prev_action:  | prev_outcome: ")
            else:
                pr = self.records[p]
                parts.append(
                    f"prev_action: {pr['action_type']} | prev_outcome: {pr['execution_outcome']}"
                )
        rec = self.records[idx]
        parts.append(f"current_task: {rec.get('task_description', '') or ''}")
        parts.append(f"target: {rec.get('action_target_desc', '') or ''}")
        parts.append(f"domain: {rec.get('website_domain', '') or ''}")
        return " | ".join(parts)

    # ---- dual-encoder path ----
    def _dual_inputs(self, idx, image):
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        enc = self.tokenizer(
            self._context_text(idx), max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
        }

    # ---- VLM path (Qwen joint processing) ----
    def _vlm_inputs(self, idx, images):
        """images: list of PIL images (1 = before only, 2 = before+after)."""
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": self._context_text(idx)})
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        enc = self.processor(text=[chat], images=list(images), return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"][0],            # [seq]
            "attention_mask": enc["attention_mask"][0],  # [seq]
            "pixel_values": enc["pixel_values"],         # [num_patches, patch_dim]
            "image_grid_thw": enc["image_grid_thw"],     # [n_img, 3] = (t, h, w)
        }
        # Newer transformers Qwen2-VL requires per-token image/text markers (M-RoPE).
        if "mm_token_type_ids" in enc:
            out["mm_token_type_ids"] = enc["mm_token_type_ids"][0]  # [seq]
        return out

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = self._load_image(rec["state_before"])
        img_w, img_h = image.size
        bbox, bbox_mask = self._bbox(rec, img_w, img_h)

        if self.path == "vlm":
            images = [image]
            if self.use_state_after and rec.get("state_after"):
                images.append(self._load_image(rec["state_after"]))
            inputs = self._vlm_inputs(idx, images)
        else:
            inputs = self._dual_inputs(idx, image)
        inputs.update(self._labels(rec, bbox, bbox_mask))
        return inputs
