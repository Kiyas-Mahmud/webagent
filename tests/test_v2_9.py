"""Recovery v2.9 correctness tests.

The v2.9 full run has no controlled mini stage (skipped for time), so these
tests plus the 16-row smoke are the only cheap evidence before a multi-day run.
Two of them are load-bearing for the registered epoch-0 abort gate:

  * ``test_offset_head_is_identity_at_initialisation`` -- the gate uses v2.8's
    epoch-0 bbox numbers as floors, which is only fair if v2.9 starts identical.
  * ``test_injection_restores_v2_8_bindings`` -- the scoped rebinding in
    ``gold_full_v2_9`` must never leak into a v2.8 run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from web_agent.config import load_config
from web_agent.models.heads import ActionHead
from web_agent.models.heads_v2_9 import ActionHeadV29
from web_agent.train import gold_full, gold_full_v2_9
from web_agent.train.trainer import Trainer

CONFIG = "configs/backbones/qwen25vl_7b_gold_v2_9.yaml"
DIM, TOKENS = 768, 24


def _v2_8_head() -> ActionHead:
    torch.manual_seed(0)
    return ActionHead(
        DIM,
        spatial_grounding=True,
        bbox_parameterization="cxcywh",
        bbox_grounding_mode="coordinate_softargmax",
        bbox_size_parameterization="log_space",
    )


def _inputs(rows: int = 3):
    torch.manual_seed(1)
    return {
        "fused": torch.randn(rows, DIM),
        "bbox_fused": torch.randn(rows, DIM),
        "spatial_tokens": torch.randn(rows, TOKENS, DIM),
        "spatial_mask": torch.ones(rows, TOKENS),
        "spatial_coords": torch.rand(rows, TOKENS, 2),
    }


# --------------------------------------------------------------------------
# Fix 3a -- grounding offset head
# --------------------------------------------------------------------------

def test_offset_head_is_identity_at_initialisation():
    """v2.9 must equal v2.8 at step 0, or the epoch-0 bbox floors are unfair."""
    base = _v2_8_head().eval()
    head = ActionHeadV29.from_v2_8(base).eval()
    data = _inputs()
    with torch.no_grad():
        want = base(data["fused"], data["bbox_fused"],
                    spatial_tokens=data["spatial_tokens"],
                    spatial_mask=data["spatial_mask"],
                    spatial_coords=data["spatial_coords"])
        got = head(data["fused"], data["bbox_fused"],
                   spatial_tokens=data["spatial_tokens"],
                   spatial_mask=data["spatial_mask"],
                   spatial_coords=data["spatial_coords"])
    torch.testing.assert_close(got["bbox"], want["bbox"])
    torch.testing.assert_close(got["bbox_cxcywh"], want["bbox_cxcywh"])
    torch.testing.assert_close(got["action_type"], want["action_type"])
    assert torch.equal(got["bbox_centre_offset"], torch.zeros_like(got["bbox_centre_offset"]))


def test_from_v2_8_preserves_the_bbox_size_prior():
    """build_gold_components applies the prior before the swap; it must survive."""
    base = _v2_8_head()
    base.initialize_bbox_size_prior([-2.0, -3.0])
    head = ActionHeadV29.from_v2_8(base)
    torch.testing.assert_close(head.bbox.bias, base.bbox.bias)
    torch.testing.assert_close(head.bbox.weight, base.bbox.weight)


def test_offset_is_bounded_and_moves_the_centre_once_trained():
    head = ActionHeadV29.from_v2_8(_v2_8_head()).eval()
    torch.manual_seed(2)
    with torch.no_grad():  # simulate a trained offset
        head.centre_offset.weight.normal_(0, 1.0)
        head.centre_offset.bias.normal_(0, 1.0)
    data = _inputs()
    with torch.no_grad():
        out = head(data["fused"], data["bbox_fused"],
                   spatial_tokens=data["spatial_tokens"],
                   spatial_mask=data["spatial_mask"],
                   spatial_coords=data["spatial_coords"])
    offset = out["bbox_centre_offset"]
    assert offset.abs().max() <= head.centre_offset_max + 1e-6
    assert offset.abs().max() > 0.0
    centre = out["bbox_cxcywh"][:, :2]
    assert ((centre >= 0.0) & (centre <= 1.0)).all()


def test_v2_9_head_refuses_unsupported_modes():
    with pytest.raises(ValueError, match="coordinate_softargmax"):
        ActionHeadV29(DIM, spatial_grounding=True, bbox_parameterization="cxcywh",
                      bbox_grounding_mode="content_attention")


# --------------------------------------------------------------------------
# Fix 1 + 2 -- config contract
# --------------------------------------------------------------------------

def test_config_fixes_the_three_defects():
    cfg = load_config(CONFIG)
    # 1. loss sign
    assert cfg["loss"]["dynamic_weighting"] == "fixed"
    # 2. warmup is absolute and well under one epoch; early stop cannot arm early
    assert cfg["train"]["warmup_steps"] < 754 // 2
    assert cfg["train"]["min_epochs"] >= 3
    # the dead key is nulled, not carrying a misleading value
    assert cfg["optim"].get("early_stopping_patience") is None
    # 3. grounding
    assert cfg["model"]["centre_offset_max"] > 0
    assert cfg["loss"]["bbox_giou_ratio"] > cfg["loss"]["bbox_l1_ratio"]
    # registered contract inherited unchanged
    assert cfg["optim"]["batch_size"] * cfg["optim"]["grad_accum"] == 32
    assert cfg["train"]["quality_selection_rule"] == "all_gates_then_outcome_mcc"
    assert cfg["name"] != load_config("configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml")["name"]


def test_fixed_weighting_total_is_non_negative():
    """With fixed weighting the total is a sum of non-negative weighted terms."""
    cfg = load_config(CONFIG)
    weights = {k: v for k, v in cfg["loss"].items() if isinstance(v, (int, float))}
    for name in ("outcome", "failure_type", "action_type", "bbox", "memory_flag",
                 "recovery", "confidence", "calibration", "recovery_outcome"):
        assert weights[name] >= 0, f"loss weight {name} is negative"


# --------------------------------------------------------------------------
# Wiring safety
# --------------------------------------------------------------------------

def test_injection_restores_v2_8_bindings():
    original_trainer = gold_full.Trainer
    original_builder = gold_full.build_gold_components
    assert original_trainer is Trainer
    with gold_full_v2_9.v2_9_bindings():
        assert gold_full.Trainer is not Trainer
        assert gold_full.build_gold_components is not original_builder
    assert gold_full.Trainer is Trainer
    assert gold_full.build_gold_components is original_builder


def test_injection_restores_bindings_even_on_error():
    original_trainer = gold_full.Trainer
    with pytest.raises(RuntimeError, match="boom"):
        with gold_full_v2_9.v2_9_bindings():
            raise RuntimeError("boom")
    assert gold_full.Trainer is original_trainer


def test_v2_9_refuses_uncertainty_weighting():
    cfg = load_config(CONFIG)
    cfg["loss"]["dynamic_weighting"] = "uncertainty"
    with pytest.raises(ValueError, match="dynamic_weighting"):
        gold_full_v2_9.run_gold_full_v2_9(
            cfg, epochs=1, checkpoint_root="/tmp/x", metrics_csv="/tmp/x.csv",
        )


# --------------------------------------------------------------------------
# Safety contract -- the v2.8 run in progress must not be disturbed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "src/web_agent/models/loss.py",
    "src/web_agent/models/heads.py",
    "src/web_agent/models/model.py",
    "src/web_agent/train/trainer.py",
    "src/web_agent/train/gold_full.py",
    "src/web_agent/train/gold_stages.py",
    "src/web_agent/train/resume.py",
    "scripts/run_gold.py",
    "configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml",
])
def test_v2_8_files_are_untouched_by_v2_9(path):
    """v2.9 is additive only. These files are loaded by the running v2.8 job."""
    import subprocess
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", path],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert diff.stdout == "", f"v2.9 work modified {path}:\n{diff.stdout[:400]}"
