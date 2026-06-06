"""YAML config loader: merge base.yaml with a per-model override file.

A model YAML may set `extends: base.yaml`; this loads base first, then deep-merges
the model file on top. Returns a plain dict consumed by dataset/model/trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on leaf keys)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a model config, resolving an `extends:` parent relative to configs/."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        parent_cfg = load_config(CONFIG_ROOT / parent)
        cfg = _deep_merge(parent_cfg, cfg)
    return cfg
