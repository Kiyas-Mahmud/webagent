"""GoldDataset — the real, leak-safe gold dataset (separate from the synthetic path).

Kept fully separate from WebAgentDataset so the synthetic 70k pipeline stays
byte-for-byte unchanged. Reuses the SAME tensor keys, so the shared model / loss /
trainer / vlm_collate run on gold with no edits.

Handles BOTH gold layouts in one code path:
  - v12 (current): nested  {"inputs": {...}, "labels": {...}, "meta": {...}}
  - v8  (pilot):   flat     {state_before, outcome_label, ...}

Honest input = state_before + state_after images + task_description (+ website_domain).
The split files carry no ssim/pixel_diff/url_after, so no scalar can leak.

Gold training uses two causal views of the same row:
  pre_*  = state_before + goal text
  post_* = state_before + state_after + goal text
The action/bbox/confidence-before heads consume only pre_*. Failure, recovery,
and memory heads consume post_*. No label is inserted into either prompt.

Gold 40K provides real confidence, memory, and attempted-recovery outcome labels.
All heads are enabled; recovery outcome is masked to attempted rows only.
When configured, invalid bbox geometry is likewise masked only from localization;
the same row remains available to every other supervised head.
"""

from __future__ import annotations

from pathlib import Path
import math

import torch
from PIL import Image
from torch.utils.data import Dataset

from web_agent.labels import (
    ACTION_TYPE,
    EXECUTION_OUTCOME,
    FAILURE_TYPE,
    RECOVERY_STRATEGY,
)
from web_agent.data.recovery_transitions import build_recovery_transition_index


def view(rec: dict):
    """Return (inputs, labels, meta) for either layout.

    v12 nested -> the three sub-dicts. v8 flat -> the record itself for all three
    (every field sits at the top level, so the same .get() calls work).
    """
    if "inputs" in rec and "labels" in rec:
        return rec["inputs"], rec["labels"], rec.get("meta", {})
    return rec, rec, rec


class GoldDataset(Dataset):
    def __init__(self, records, cfg, processor, trajectory_records=None):
        self.records = records
        self.cfg = cfg
        self.processor = processor
        self.root = Path(cfg["data"]["root"])
        self.use_state_after = bool(cfg["data"].get("use_state_after", True))
        self.causal_routing = bool(cfg["data"].get("causal_routing", False))
        self.executed_action_post = bool(
            cfg["data"].get("executed_action_post", False)
        )
        self.use_recovery_transitions = bool(
            cfg["data"].get("recovery_transitions", False)
        )
        self.strict_bbox_geometry = bool(
            cfg["data"].get("strict_bbox_geometry", False)
        )
        self.invalid_bbox_policy = str(
            cfg["data"].get("invalid_bbox_policy", "error")
        )
        if self.invalid_bbox_policy not in {"error", "mask"}:
            raise ValueError(
                "invalid_bbox_policy must be either 'error' or 'mask', got "
                f"{self.invalid_bbox_policy!r}"
            )
        if self.invalid_bbox_policy == "mask" and not self.strict_bbox_geometry:
            raise ValueError(
                "invalid_bbox_policy='mask' requires strict_bbox_geometry=true"
            )
        source_rows = trajectory_records if trajectory_records is not None else records
        if self.use_recovery_transitions:
            self.recovery_transitions, self.transition_report = (
                build_recovery_transition_index(source_rows)
            )
        else:
            self.recovery_transitions, self.transition_report = {}, {}

    def __len__(self) -> int:
        return len(self.records)

    # ---- helpers ----
    def _record_root(self, meta: dict) -> Path:
        private_root = meta.get("_data_root")
        return Path(private_root) if private_root else self.root

    def _load_image(
        self,
        rel_path: str,
        *,
        root: Path | None = None,
    ) -> Image.Image:
        return Image.open((root or self.root) / rel_path).convert("RGB")

    def _bbox(self, lab: dict, img_w: int, img_h: int):
        """Return (bbox[4] normalized, mask[1]); zeros + mask 0 when bbox is null."""
        b = lab.get("action_target_bbox")
        if not b:
            return torch.zeros(4, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)
        try:
            values = (
                float(b["x"]),
                float(b["y"]),
                float(b["width"]),
                float(b["height"]),
            )
        except (KeyError, TypeError, ValueError):
            if self.strict_bbox_geometry and self.invalid_bbox_policy == "mask":
                return (
                    torch.zeros(4, dtype=torch.float32),
                    torch.zeros(1, dtype=torch.float32),
                )
            raise ValueError(
                f"invalid bbox object {b!r} for state_before image {img_w}x{img_h}"
            ) from None
        if self.strict_bbox_geometry:
            x, y, width, height = values
            valid = (
                all(math.isfinite(value) for value in values)
                and x >= 0
                and y >= 0
                and width > 0
                and height > 0
                and x + width <= img_w + 1e-6
                and y + height <= img_h + 1e-6
            )
            if not valid:
                if self.invalid_bbox_policy == "mask":
                    return (
                        torch.zeros(4, dtype=torch.float32),
                        torch.zeros(1, dtype=torch.float32),
                    )
                raise ValueError(
                    f"invalid bbox {b!r} for state_before image {img_w}x{img_h}"
                )
        bbox = torch.tensor([
            values[0] / img_w,
            values[1] / img_h,
            values[2] / img_w,
            values[3] / img_h,
        ], dtype=torch.float32)
        if not self.strict_bbox_geometry:
            bbox.clamp_(0.0, 1.0)
        return bbox, torch.ones(1, dtype=torch.float32)

    def _context_text(
        self,
        inp: dict,
        phase: str,
        executed_action: str = "",
        action_value: str = "",
    ) -> str:
        task = inp.get("task_description", "") or ""
        domain = inp.get("website_domain", "") or ""
        instruction = {
            "pre": "Choose the next action from the page before execution.",
            "post": "Assess the result by comparing the page before and after execution.",
            "recovery": (
                "Assess whether the executed recovery succeeded by comparing the "
                "failure state with the post-recovery state."
            ),
        }[phase]
        action_context = ""
        if executed_action:
            action_context = f" | executed_action: {executed_action}"
            if action_value:
                action_context += f" | action_value: {action_value}"
        return (
            f"{instruction} current_task: {task} | domain: {domain}"
            f"{action_context}"
        )

    def _vlm_inputs(
        self,
        inp: dict,
        images,
        phase: str,
        executed_action: str = "",
        action_value: str = "",
    ):
        """images: list of PIL images (1 = before only, 2 = before+after)."""
        content = [{"type": "image"} for _ in images]
        content.append({
            "type": "text",
            "text": self._context_text(
                inp, phase, executed_action=executed_action, action_value=action_value,
            ),
        })
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

    def _labels(
        self,
        lab: dict,
        meta: dict,
        bbox,
        bbox_mask,
        has_recovery_transition: bool,
    ) -> dict:
        attempted = lab.get("recovery_success") is not None
        configured_masks = meta.get("_loss_masks", {})

        def loss_mask(name: str) -> torch.Tensor:
            return torch.tensor(
                [float(configured_masks.get(name, 1.0))],
                dtype=torch.float32,
            )

        return {
            "bbox": bbox,
            "bbox_mask": bbox_mask * loss_mask("bbox"),
            "borrowed": torch.tensor([0.0]),
            "label_outcome": torch.tensor(EXECUTION_OUTCOME[lab["outcome_label"]]),
            "label_failtype": torch.tensor(FAILURE_TYPE[lab["failure_type_4"]]),
            "label_action": torch.tensor(ACTION_TYPE[lab["action_type"]]),
            "label_recovery": torch.tensor(RECOVERY_STRATEGY[lab["recovery_strategy"]]),
            "label_needs_recovery": torch.tensor([float(attempted)]),
            # v12 has real confidence + memory; v8 -> neutral default (head disabled there).
            "label_memory": torch.tensor([float(lab.get("memory_update_flag") or False)]),
            "label_confidence": torch.tensor([float(
                lab["agent_confidence_before"]
                if lab.get("agent_confidence_before") is not None else 0.5)]),
            # recovery_success: -1 = not attempted (null) -> masked out in the loss/metric;
            # 0/1 = attempted-failed / attempted-succeeded (the only rows the head learns from).
            "label_recovery_success": torch.tensor([
                -1.0 if not has_recovery_transition else float(lab["recovery_success"])]),
            "original_task_id": meta.get("task_id", ""),
            "source_dataset": str(meta.get("_source_dataset", "original_gold")),
            "strategy_source_pre": torch.tensor([
                float(bool(meta.get("_strategy_source_pre", False)))
            ]),
            "loss_mask_outcome": loss_mask("outcome"),
            "loss_mask_failtype": loss_mask("failure_type"),
            "loss_mask_action": loss_mask("action"),
            "loss_mask_recovery": loss_mask("recovery"),
            "loss_mask_needs_recovery": loss_mask("needs_recovery"),
            "loss_mask_memory": loss_mask("memory"),
            "loss_mask_confidence": loss_mask("confidence"),
            "loss_mask_calibration": loss_mask("calibration"),
            "loss_mask_contrastive": loss_mask("contrastive"),
            "loss_mask_recovery_outcome": loss_mask("recovery_outcome"),
        }

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        inp, lab, meta = view(rec)
        record_root = self._record_root(meta)
        before = self._load_image(inp["state_before"], root=record_root)
        img_w, img_h = before.size
        bbox, bbox_mask = self._bbox(lab, img_w, img_h)

        images = [before]
        if self.use_state_after and inp.get("state_after"):
            images.append(self._load_image(
                inp["state_after"], root=record_root,
            ))

        sample_id = str(meta.get("sample_id") or f"{meta.get('task_id', '')}:{meta.get('step_index', idx)}")
        transition = self.recovery_transitions.get(sample_id)

        if self.causal_routing:
            if len(images) != 2:
                raise ValueError("causal gold routing requires both state_before and state_after")
            pre = self._vlm_inputs(inp, [before], phase="pre")
            post = self._vlm_inputs(
                inp,
                images,
                phase="post",
                executed_action=lab.get("action_type", "") if self.executed_action_post else "",
                action_value=lab.get("action_value", "") if self.executed_action_post else "",
            )
            out = {f"pre_{key}": value for key, value in pre.items()}
            out.update({f"post_{key}": value for key, value in post.items()})
        else:
            out = self._vlm_inputs(inp, images, phase="post")
        if transition is not None:
            transition_root = Path(
                transition.get("_data_root") or record_root
            )
            recovery_inp = {
                "task_description": transition["task_description"],
                "website_domain": transition["website_domain"],
            }
            recovery = self._vlm_inputs(
                recovery_inp,
                [
                    self._load_image(
                        transition["failure_state"], root=transition_root,
                    ),
                    self._load_image(
                        transition["post_recovery_state"], root=transition_root,
                    ),
                ],
                phase="recovery",
                executed_action=transition["executed_recovery_action"],
                action_value=transition["recovery_action_value"],
            )
            out.update({f"recovery_{key}": value for key, value in recovery.items()})
        out.update(self._labels(
            lab,
            meta,
            bbox,
            bbox_mask,
            has_recovery_transition=(
                transition is not None
                if self.use_recovery_transitions
                else lab.get("recovery_success") is not None
            ),
        ))
        return out
