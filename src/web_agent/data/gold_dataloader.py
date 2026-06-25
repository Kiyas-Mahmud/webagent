"""Gold split loading + DataLoader (separate from the synthetic dataloader).

Each gold split file IS the split (no per-row filtering). Reuses the synthetic
`vlm_collate` unchanged — GoldDataset emits exactly the keys it expects.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from web_agent.data.dataloader import vlm_collate   # reuse, unchanged
from web_agent.data.gold_dataset import GoldDataset
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
    remapped = [
        {
            "action_type": r["action_type"],
            "failure_type": r["failure_type_4"],
            "execution_outcome": r["outcome_label"],
        }
        for r in records
    ]
    return balanced_class_weights(remapped, limit=limit)


def build_gold_dataloader(cfg, which, processor, records=None, limit=None,
                          batch_size=None, shuffle=True, num_workers=None):
    """Filter-free loader for a gold split. Random batches (SupCon is supervised,
    so no pair-sampler needed)."""
    rows = records if records is not None else load_gold_split(cfg, which)
    if limit is not None:
        rows = rows[:limit]
    ds = GoldDataset(rows, cfg, processor)
    bs = batch_size or cfg["optim"]["batch_size"]
    nw = cfg["data"].get("num_workers", 2) if num_workers is None else num_workers
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, drop_last=False,
        num_workers=nw, pin_memory=True, collate_fn=vlm_collate,
        persistent_workers=(nw > 0),
    )
