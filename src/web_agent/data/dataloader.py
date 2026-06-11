"""Split loading + the 5 mode filters + stratified subsample (SPEC 6.3).

Stable core (hybrid plan C). The notebook calls build_dataloader() to get a
DataLoader, then drives the training loop itself (visible, tweakable).

Modes are FILTERS over the same JSONs, never new files:
  full_labels -> split == "train"
  visual      -> split == "train" and not borrowed_image
  eval_visual -> eval sub-split  and not borrowed_image
  eval_labels -> eval sub-split
  bbox        -> action_target_bbox is not None

`limit` (smoke=16 / mini=5000) takes a stratified-by-failure_type subsample so
small runs still see every class (LOOP is only 7.4%).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from web_agent.data.dataset import WebAgentDataset

# Tensor label keys shared by both paths; stacked identically in every collate.
_LABEL_KEYS = (
    "bbox", "bbox_mask", "borrowed", "label_outcome", "label_failtype",
    "label_action", "label_recovery", "label_memory", "label_confidence",
    "label_recovery_success",
)

MODE_FILTERS = ("full_labels", "visual", "eval_visual", "eval_labels", "bbox")
EVAL_SUBSPLITS = {"test_task", "test_website", "test_domain"}


def load_split(cfg: dict, which: str) -> list[dict]:
    """which in {train, val, test}. UTF-8 mandatory (cp1252 fails on this data)."""
    fname = cfg["data"][f"{which}_json"]
    with open(Path(cfg["data"]["root"]) / fname, encoding="utf-8") as f:
        return json.load(f)


def _keep(rec: dict, mode: str) -> bool:
    split = rec.get("split", "")
    borrowed = bool(rec.get("borrowed_image", False))
    if mode == "full_labels":
        return split == "train"
    if mode == "visual":
        return split == "train" and not borrowed
    if mode == "eval_labels":
        return split in EVAL_SUBSPLITS
    if mode == "eval_visual":
        return split in EVAL_SUBSPLITS and not borrowed
    if mode == "bbox":
        return rec.get("action_target_bbox") is not None
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODE_FILTERS}")


def filter_records(records: list[dict], mode: str) -> list[dict]:
    return [r for r in records if _keep(r, mode)]


def stratified_subsample(records: list[dict], n: int, seed: int = 42,
                         key: str = "failure_type") -> list[dict]:
    """Take ~n rows keeping each `key` class proportionally represented."""
    if n >= len(records):
        return records
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get(key)].append(r)
    out: list[dict] = []
    for cls, rows in buckets.items():
        take = max(1, round(n * len(rows) / len(records)))
        out.extend(rng.sample(rows, min(take, len(rows))))
    rng.shuffle(out)
    return out[:n]


def _stack_labels(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in _LABEL_KEYS}
    out["original_task_id"] = [b["original_task_id"] for b in batch]  # str list (contrastive)
    return out


class TaskPairBatchSampler(Sampler):
    """Group rows by original_task_id so the passes of a task land in the same
    batch — this lets the contrastive loss form SUCCESS/FAILURE pairs (spec).

    Tasks are shuffled each epoch; rows within a task stay together; batches are
    filled task-by-task up to batch_size.
    """

    def __init__(self, records, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._by_task: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            self._by_task[r.get("original_task_id", "")].append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        tasks = list(self._by_task.keys())
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(tasks)
        batch: list[int] = []
        for t in tasks:
            for idx in self._by_task[t]:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self._by_task.values())
        return (total + self.batch_size - 1) // self.batch_size


def vlm_collate(batch: list[dict]) -> dict:
    """Collate for the Qwen VLM path: variable-length sequences + per-image patches.

    input_ids/attention_mask are left-variable -> right-pad to the longest in the
    batch. pixel_values are flattened patches that differ per image -> concatenate
    along dim 0 (Qwen consumes them with image_grid_thw, which we stack).
    """
    out = _stack_labels(batch)
    out["input_ids"] = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0,
    )
    out["attention_mask"] = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
    )
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    out["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])
    if "mm_token_type_ids" in batch[0]:
        out["mm_token_type_ids"] = pad_sequence(
            [b["mm_token_type_ids"] for b in batch], batch_first=True, padding_value=0,
        )
    return out


def dual_collate(batch: list[dict]) -> dict:
    """Collate for the dual-encoder path: every field is fixed-size -> just stack."""
    out = _stack_labels(batch)
    for k in ("pixel_values", "input_ids", "attention_mask"):
        out[k] = torch.stack([b[k] for b in batch])
    return out


def build_dataloader(cfg, mode, records, processor, tokenizer=None,
                     limit=None, batch_size=None, shuffle=True, seed=42,
                     num_workers=None, pair_sampler=False):
    """Filter -> (optional stratified limit) -> WebAgentDataset -> DataLoader.

    Picks the collate by cfg["backbone"]["path"] (vlm vs dual_encoder).
    `num_workers` overrides cfg; >0 parallelizes the per-sample VLM processing
    (the bottleneck) so the GPU stops starving.
    `pair_sampler=True` groups a task's passes into the same batch (contrastive)."""
    rows = filter_records(records, mode)
    if limit is not None:
        rows = stratified_subsample(rows, limit, seed=seed)
    ds = WebAgentDataset(rows, cfg, processor, tokenizer)
    bs = batch_size or cfg["optim"]["batch_size"]
    nw = cfg["data"].get("num_workers", 2) if num_workers is None else num_workers
    collate = vlm_collate if cfg["backbone"]["path"] == "vlm" else dual_collate
    common = dict(num_workers=nw, pin_memory=True, collate_fn=collate,
                  persistent_workers=(nw > 0))
    if pair_sampler:
        sampler = TaskPairBatchSampler(rows, bs, shuffle=shuffle, seed=seed)
        return DataLoader(ds, batch_sampler=sampler, **common), sampler
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, **common)
