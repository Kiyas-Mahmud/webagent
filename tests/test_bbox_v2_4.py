from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

from web_agent.config import load_config
from web_agent.data.bbox_audit import bbox_log_size_prior


def _record(image_name: str, x: float, y: float, width: float, height: float):
    return {
        "inputs": {"state_before": image_name},
        "labels": {
            "action_target_bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        },
        "meta": {"sample_id": image_name},
    }


def test_train_only_log_size_prior_is_geometric_and_excludes_invalid_rows(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "first.png")
    Image.new("RGB", (100, 100)).save(tmp_path / "second.png")
    Image.new("RGB", (100, 100)).save(tmp_path / "invalid.png")
    records = [
        _record("first.png", 1, 1, 10, 4),
        _record("second.png", 1, 1, 40, 25),
        _record("invalid.png", 90, 90, 20, 20),
    ]

    prior = bbox_log_size_prior(records, tmp_path)

    assert prior["source"] == "valid training bbox rows only"
    assert prior["valid_rows"] == 2
    assert prior["excluded_invalid_rows"] == 1
    assert prior["geometric_mean_wh"] == pytest.approx([0.2, 0.1])
    assert prior["log_wh"] == pytest.approx([math.log(0.2), math.log(0.1)])


def test_log_size_loss_keeps_nonzero_gradient_below_decode_floor():
    torch = pytest.importorskip("torch")
    from web_agent.models.bbox import detr_bbox_log_size_loss_terms

    predicted_log_wh = torch.tensor([[-100.0, -100.0]], requires_grad=True)
    decoded_wh = torch.exp(predicted_log_wh.clamp(min=math.log(1e-4), max=0.0))
    prediction = torch.cat([torch.tensor([[0.5, 0.5]]), decoded_wh], dim=-1)
    target = torch.tensor([[0.35, 0.40, 0.30, 0.20]])

    centre_l1, log_size, giou = detr_bbox_log_size_loss_terms(
        prediction, predicted_log_wh, target,
    )
    (5.0 * (centre_l1 + log_size) + 2.0 * giou).backward()

    assert torch.isfinite(predicted_log_wh.grad).all()
    assert torch.all(predicted_log_wh.grad < 0)
    assert predicted_log_wh.grad.abs().min().item() > 0.0


def test_log_space_action_head_uses_registered_prior_and_exposes_raw_logits():
    torch = pytest.importorskip("torch")
    from web_agent.models.heads import ActionHead

    head = ActionHead(
        dim=8,
        hidden=4,
        dropout=0.0,
        spatial_grounding=True,
        bbox_parameterization="cxcywh",
        bbox_grounding_mode="coordinate_softargmax",
        bbox_size_parameterization="log_space",
    )
    prior = torch.log(torch.tensor([0.2, 0.05]))
    head.initialize_bbox_size_prior(prior)
    output = head(
        torch.randn(2, 8),
        bbox_fused=torch.randn(2, 8),
        spatial_tokens=torch.randn(2, 4, 8),
        spatial_mask=torch.ones(2, 4, dtype=torch.bool),
        spatial_coords=torch.tensor([
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
        ]),
    )

    assert torch.count_nonzero(head.bbox.weight) == 0
    assert torch.allclose(head.bbox.bias, prior)
    assert torch.allclose(output["bbox_log_wh"], prior.expand(2, -1))
    assert torch.allclose(
        output["bbox_cxcywh"][:, 2:],
        torch.tensor([[0.2, 0.05], [0.2, 0.05]]),
    )
    assert torch.all(output["bbox"][:, 2:] > 0)


def test_combined_loss_routes_direct_log_size_term_and_backward():
    torch = pytest.importorskip("torch")
    from web_agent.models.loss import CombinedLoss

    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_4.yaml")
    loss_fn = CombinedLoss(cfg)
    predicted_log_wh = torch.tensor(
        [[-100.0, -100.0], [-2.0, -3.0]], requires_grad=True,
    )
    decoded_wh = torch.exp(predicted_log_wh.clamp(min=math.log(1e-4), max=0.0))
    prediction_cxcywh = torch.cat([
        torch.tensor([[0.5, 0.5], [0.4, 0.4]]), decoded_wh,
    ], dim=-1)
    preds = {
        "outcome": torch.randn(2, 2, requires_grad=True),
        "failure_type": torch.randn(2, 4, requires_grad=True),
        "action_type": torch.randn(2, 6, requires_grad=True),
        "recovery": torch.randn(2, 6, requires_grad=True),
        "memory_recovery": torch.randn(2, 6, requires_grad=True),
        "needs_recovery": torch.randn(2, 1, requires_grad=True),
        "bbox": prediction_cxcywh,
        "bbox_cxcywh": prediction_cxcywh,
        "bbox_log_wh": predicted_log_wh,
        "memory_flag": torch.randn(2, 1, requires_grad=True),
        "recovery_outcome": torch.randn(2, 1, requires_grad=True),
        "confidence": torch.sigmoid(torch.randn(2, 1, requires_grad=True)),
        "fused": torch.randn(2, 768, requires_grad=True),
    }
    batch = {
        "label_outcome": torch.tensor([0, 1]),
        "label_failtype": torch.tensor([0, 1]),
        "label_action": torch.tensor([0, 1]),
        "label_recovery": torch.tensor([1, 2]),
        "label_needs_recovery": torch.ones(2, 1),
        "bbox": torch.tensor([[0.35, 0.40, 0.30, 0.20], [0.30, 0.30, 0.20, 0.10]]),
        "bbox_mask": torch.ones(2, 1),
        "label_memory": torch.tensor([[0.0], [1.0]]),
        "label_recovery_success": torch.tensor([[0.0], [1.0]]),
        "label_confidence": torch.tensor([[0.8], [0.2]]),
        "original_task_id": ["a", "b"],
    }

    terms = loss_fn(preds, batch)
    terms["total"].backward()

    assert torch.isfinite(terms["total"])
    assert float(terms["bbox_l1"]) == pytest.approx(
        float(terms["bbox_center_l1"] + terms["bbox_log_size_smooth_l1"])
    )
    assert torch.isfinite(predicted_log_wh.grad).all()
    assert predicted_log_wh.grad[0].abs().min().item() > 0.0


def test_v2_4_changes_only_registered_size_mechanism_and_keeps_v2_3_gates():
    previous = load_config("configs/backbones/qwen2vl_2b_gold_v2_3.yaml")
    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_4.yaml")

    assert cfg["train"]["controlled_experiment_tag"] == "recovery_v2_4"
    assert cfg["model"]["bbox_size_parameterization"] == "log_space"
    assert cfg["loss"]["bbox_loss"] == "detr_log_size_giou"
    assert cfg["optim"] == previous["optim"]
    for key in (
        "bbox_overfit_rows",
        "bbox_overfit_steps",
        "bbox_overfit_min_iou_gain",
        "bbox_overfit_min_mean_iou",
        "bbox_overfit_max_full_screen_fraction",
        "bbox_overfit_fp32_grounding",
        "bbox_overfit_disable_dropout",
        "bbox_overfit_require_finite_gradients",
    ):
        assert cfg["train"][key] == previous["train"][key]
    assert cfg["loss"]["bbox_l1_ratio"] == previous["loss"]["bbox_l1_ratio"]
    assert cfg["loss"]["bbox_giou_ratio"] == previous["loss"]["bbox_giou_ratio"]
    assert cfg["train"]["bbox_overfit_min_predicted_size"] == pytest.approx(1e-6)


def test_v2_4_notebook_is_isolated_output_free_and_compiles():
    path = Path("notebooks/kaggle_gold_recovery_v2_4.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    all_source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "qwen2vl_2b_gold_v2_4.yaml" in all_source
    assert "STAGE = 'smoke'" in all_source
    assert "BBOX_MONTAGE_REVIEWED = False" in all_source
    assert "bbox_size_parameterization'] == 'log_space'" in all_source
    assert "decoded_sizes_above_registered_minimum" in all_source
    assert "gold_recovery_v2_3" not in all_source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
