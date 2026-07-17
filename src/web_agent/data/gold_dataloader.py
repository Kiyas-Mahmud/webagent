"""Gold split loading + DataLoader (separate from the synthetic dataloader).

Each gold split file IS the split (no per-row filtering). Reuses the synthetic
`vlm_collate` unchanged — GoldDataset emits exactly the keys it expects.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from web_agent.data.dataloader import stack_labels, vlm_collate
from web_agent.data.gold_dataset import GoldDataset, view
from web_agent.utils.class_weights import balanced_class_weights


def load_gold_split(cfg: dict, which: str) -> list[dict]:
    """which in {train, val, test}. Reads cfg['data'][f'{which}_json'] under data.root."""
    fname = cfg["data"][f"{which}_json"]
    with open(Path(cfg["data"]["root"]) / fname, encoding="utf-8") as f:
        return json.load(f)


def gold_class_weights(records, limit: int | None = None):
    """action/failtype/outcome weights via the existing balanced_class_weights.

    Remaps gold field names onto the synthetic keys that balanced_class_weights
    reads, so we reuse that function with no new weighting code.
    """
    remapped = []
    for r in records:
        _, lab, _ = view(r)   # handles both v12 nested and v8 flat
        remapped.append({
            "action_type": lab["action_type"],
            "failure_type": lab["failure_type_4"],
            "execution_outcome": lab["outcome_label"],
        })
    return balanced_class_weights(remapped, limit=limit)


def stratified_gold_subsample(records, n: int, seed: int = 42):
    """Deterministic proportional sample over the four failure categories."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for record in records:
        _, labels, _ = view(record)
        buckets[labels["failure_type_4"]].append(record)
    selected = []
    for rows in buckets.values():
        take = max(1, round(n * len(rows) / len(records)))
        selected.extend(rng.sample(rows, min(take, len(rows))))
    selected_ids = {id(record) for record in selected}
    remaining = [record for record in records if id(record) not in selected_ids]
    rng.shuffle(selected)
    rng.shuffle(remaining)
    selected.extend(remaining)
    return selected[:n]


def select_smoke_records(records, n: int = 16, seed: int = 42):
    """Small deterministic set covering every failure type and action class."""
    if n < 10:
        raise ValueError("gold smoke test needs at least 10 rows for label coverage")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    selected = []
    covered_failures = set()
    covered_actions = set()
    covered_recovery_outcomes = set()
    for record in shuffled:
        _, labels, _ = view(record)
        failure = labels["failure_type_4"]
        action = labels["action_type"]
        recovery_outcome = labels.get("recovery_success")
        adds_recovery_coverage = (
            recovery_outcome in {True, False}
            and recovery_outcome not in covered_recovery_outcomes
        )
        if (
            failure not in covered_failures
            or action not in covered_actions
            or adds_recovery_coverage
        ):
            selected.append(record)
            covered_failures.add(failure)
            covered_actions.add(action)
            if recovery_outcome in {True, False}:
                covered_recovery_outcomes.add(recovery_outcome)
        if (
            len(covered_failures) == 4
            and len(covered_actions) == 6
            and covered_recovery_outcomes == {True, False}
        ):
            break
    if (
        len(covered_failures) != 4
        or len(covered_actions) != 6
        or covered_recovery_outcomes != {True, False}
    ):
        raise ValueError("gold smoke records do not cover all required label categories")
    chosen_ids = {id(record) for record in selected}
    selected.extend(record for record in shuffled if id(record) not in chosen_ids)
    return selected[:n]


def _collate_stream(batch: list[dict], prefix: str) -> dict:
    out = {}
    for name in ("input_ids", "attention_mask"):
        key = f"{prefix}{name}"
        out[key] = pad_sequence(
            [row[key] for row in batch], batch_first=True, padding_value=0,
        )
    for name in ("pixel_values", "image_grid_thw"):
        key = f"{prefix}{name}"
        out[key] = torch.cat([row[key] for row in batch], dim=0)
    token_type_key = f"{prefix}mm_token_type_ids"
    if token_type_key in batch[0]:
        out[token_type_key] = pad_sequence(
            [row[token_type_key] for row in batch], batch_first=True, padding_value=0,
        )
    return out


def causal_gold_collate(batch: list[dict]) -> dict:
    """Collate independent pre-action and post-action VLM streams."""
    out = stack_labels(batch)
    out.update(_collate_stream(batch, "pre_"))
    out.update(_collate_stream(batch, "post_"))
    return out


def build_gold_dataloader(cfg, which, processor, records=None, limit=None,
                          batch_size=None, shuffle=True, num_workers=None,
                          seed: int = 42, smoke: bool = False):
    """Filter-free loader for a gold split. Random batches (SupCon is supervised,
    so no pair-sampler needed)."""
    rows = records if records is not None else load_gold_split(cfg, which)
    if limit is not None:
        rows = (
            select_smoke_records(rows, limit, seed)
            if smoke
            else stratified_gold_subsample(rows, limit, seed)
        )
    ds = GoldDataset(rows, cfg, processor)
    bs = batch_size or cfg["optim"]["batch_size"]
    nw = cfg["data"].get("num_workers", 2) if num_workers is None else num_workers
    collate = causal_gold_collate if cfg["data"].get("causal_routing") else vlm_collate
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        num_workers=nw, pin_memory=True, collate_fn=collate,
        persistent_workers=(nw > 0),
        generator=generator,
    )
