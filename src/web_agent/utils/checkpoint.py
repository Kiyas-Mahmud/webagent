"""Checkpoint save/load — must round-trip ALL module states (red-flag #6).

Save model + optimizer + scheduler + scaler + step/epoch + config so a dead
Kaggle session resumes from the last 500-step checkpoint with identical state.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, **state) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)
