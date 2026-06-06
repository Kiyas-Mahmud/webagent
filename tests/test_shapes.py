"""Shape + finite-loss tests — the cheap guard before any GPU run.

Verifies the FIXED contract that must hold for ALL backbones:
  fused embedding = [B, 768]; head outputs have correct widths (outcome 2,
  failure_type 4, action_type 5, recovery 6, bbox 4, confidence/memory 1);
  combined loss is a finite scalar; backward runs.

These import from web_agent.labels so a label-map change can't silently drift.
"""

from __future__ import annotations

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
    assert NUM_ACTION_TYPE == 5
    assert NUM_RECOVERY == 6


# TODO(Phase 3): once heads/model/loss exist, add:
#   - test_fused_shape: WebAgentModel(cfg) forward on a 4-row dummy -> fused [4,768]
#   - test_head_widths: each head output width matches NUM_* above
#   - test_loss_finite: CombinedLoss returns a finite scalar
#   - test_backward: loss.backward() changes >=1 parameter
