"""Shape + finite-loss tests — the cheap guard before any GPU run.

Verifies the FIXED contract that must hold for ALL backbones:
  fused embedding = [B, 768]; head outputs have correct widths (outcome 2,
  failure_type 4, action_type 6, recovery 6, bbox 4, confidence/memory 1);
  combined loss is a finite scalar; backward runs.

These import from web_agent.labels so a label-map change can't silently drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from web_agent.labels import (
    NUM_ACTION_TYPE,
    NUM_FAILURE_TYPE,
    NUM_OUTCOME,
    NUM_RECOVERY,
)


def test_label_widths_match_spec():
    # Guards SPEC 3.4 head widths even before the model is built.
    assert NUM_OUTCOME == 2
    assert NUM_FAILURE_TYPE == 4
    assert NUM_ACTION_TYPE == 6
    assert NUM_RECOVERY == 6


def test_head_widths_match_gold_contract():
    torch = pytest.importorskip("torch")
    from web_agent.models.heads import ActionHead, FailureHead

    fused = torch.randn(4, 8)
    action = ActionHead(dim=8, hidden=4, dropout=0.0)(fused)
    failure = FailureHead(dim=8, hidden=4, dropout=0.0)(fused)
    assert action["action_type"].shape == (4, 6)
    assert action["bbox"].shape == (4, 4)
    assert failure["outcome"].shape == (4, 2)
    assert failure["failure_type"].shape == (4, 4)
    assert failure["confidence"].shape == (4, 1)
    assert failure["recovery"].shape == (4, 6)


def test_confidence_uses_pre_action_embedding():
    torch = pytest.importorskip("torch")
    from web_agent.models.heads import FailureHead

    head = FailureHead(dim=2, hidden=2, dropout=0.0)
    head.trunk = torch.nn.Identity()
    with torch.no_grad():
        head.confidence.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.confidence.bias.zero_()
    post = torch.zeros(1, 2)
    low_pre = torch.tensor([[-2.0, 0.0]])
    high_pre = torch.tensor([[2.0, 0.0]])
    low = head(post, confidence_fused=low_pre)["confidence"]
    high = head(post, confidence_fused=high_pre)["confidence"]
    assert high.item() > low.item()


def test_model_routes_pre_and_post_embeddings_to_correct_heads():
    torch = pytest.importorskip("torch")
    from web_agent.models.model import WebAgentModel

    class Encoder(torch.nn.Module):
        def forward(self, batch, prefix=""):
            return batch[f"{prefix}features"]

    class Failure(torch.nn.Module):
        def forward(self, post, confidence_fused=None):
            return {"failure_source": post, "confidence_source": confidence_fused}

    class Action(torch.nn.Module):
        def forward(self, pre):
            return {"action_source": pre}

    class Memory(torch.nn.Module):
        def forward(self, post):
            return {"memory_source": post}

    class RecoveryOutcome(torch.nn.Module):
        def forward(self, post):
            return {"recovery_outcome_source": post}

    model = WebAgentModel.__new__(WebAgentModel)
    torch.nn.Module.__init__(model)
    model.path = "vlm"
    model.causal_routing = True
    model.encoder = Encoder()
    model.adapter = torch.nn.Identity()
    model.failure_head = Failure()
    model.action_head = Action()
    model.memory_head = Memory()
    model.recovery_outcome_head = RecoveryOutcome()

    pre = torch.tensor([[1.0, 2.0]])
    post = torch.tensor([[3.0, 4.0]])
    predictions = model({"pre_features": pre, "post_features": post})
    assert torch.equal(predictions["action_source"], pre)
    assert torch.equal(predictions["confidence_source"], pre)
    assert torch.equal(predictions["failure_source"], post)
    assert torch.equal(predictions["memory_source"], post)
    assert torch.equal(predictions["recovery_outcome_source"], post)


def test_metrics_mask_unattempted_recoveries_and_separate_confidence():
    pytest.importorskip("torch")
    from web_agent.train.trainer import compute_metrics

    predictions = {
        "outcome_pred": np.array([0, 1, 1]),
        "outcome_true": np.array([0, 1, 0]),
        "failtype_pred": np.array([0, 2, 1]),
        "failtype_true": np.array([0, 2, 1]),
        "action_pred": np.array([0, 5, 3]),
        "action_true": np.array([0, 5, 3]),
        "recovery_pred": np.array([0, 1, 2]),
        "recovery_true": np.array([0, 1, 2]),
        "memory_pred": np.array([0, 1, 1]),
        "memory_true": np.array([0, 1, 1]),
        "recovery_outcome_pred": np.array([1, 1, 0]),
        "recovery_outcome_true": np.array([-1, 1, 0]),
        "outcome_confidence": np.array([0.9, 0.8, 0.6]),
        "confidence": np.array([0.3, 0.7, 0.4]),
        "confidence_true": np.array([0.2, 0.8, 0.4]),
        "bbox_pred": np.zeros((3, 4)),
        "bbox_true": np.zeros((3, 4)),
        "bbox_mask": np.ones(3),
    }
    metrics = compute_metrics(predictions)
    assert metrics["recovery_outcome_acc"] == 1.0
    assert metrics["confidence_mae"] > 0.0
    assert "outcome_ece" in metrics
    assert "ece" not in metrics
