from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

from web_agent.config import load_config
from web_agent.data.bbox_audit import audit_bbox_probe_feasibility


def _record(
    image_name: str,
    box: tuple[float, float, float, float],
    *,
    sample_id: str,
    task: str = "click the target",
    domain: str = "example",
) -> dict:
    x, y, width, height = box
    return {
        "inputs": {
            "state_before": image_name,
            "task_description": task,
            "website_domain": domain,
        },
        "labels": {
            "action_target_bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        },
        "meta": {"sample_id": sample_id},
    }


def test_feasibility_audit_detects_content_identical_conflicting_targets(tmp_path):
    pixels = Image.new("RGB", (100, 100), "white")
    pixels.save(tmp_path / "first.png")
    pixels.save(tmp_path / "renamed.png")
    records = [
        _record("first.png", (10, 20, 30, 10), sample_id="one"),
        _record("renamed.png", (10, 60, 30, 10), sample_id="two"),
    ]

    report = audit_bbox_probe_feasibility(records, tmp_path)

    assert report["status"] == "FAIL"
    assert report["unique_pre_action_inputs"] == 1
    assert report["conflicting_input_groups"] == 1
    assert report["conflicting_rows"] == 2
    assert report["test_rows_read"] == 0


def test_feasibility_audit_keeps_text_distinct_and_accepts_same_target(tmp_path):
    pixels = Image.new("RGB", (100, 100), "white")
    for name in ("first.png", "second.png", "third.png"):
        pixels.save(tmp_path / name)
    records = [
        _record("first.png", (10, 20, 30, 10), sample_id="one"),
        _record("second.png", (10, 20, 30, 10), sample_id="two"),
        _record(
            "third.png",
            (10, 60, 30, 10),
            sample_id="three",
            task="click a different target",
        ),
    ]

    report = audit_bbox_probe_feasibility(records, tmp_path)

    assert report["status"] == "PASS"
    assert report["unique_pre_action_inputs"] == 2
    assert report["duplicate_input_groups"] == 1
    assert report["conflicting_rows"] == 0


def test_attention_target_masks_padding_and_peaks_at_target_centre():
    torch = pytest.importorskip("torch")
    from web_agent.models.bbox import bbox_attention_target_distribution

    coordinates = torch.tensor([[[
        0.25, 0.25,
    ], [
        0.75, 0.25,
    ], [
        0.25, 0.75,
    ], [
        0.75, 0.75,
    ], [
        0.0, 0.0,
    ]]])
    mask = torch.tensor([[True, True, True, True, False]])
    target = torch.tensor([[0.65, 0.65, 0.20, 0.20]])

    distribution = bbox_attention_target_distribution(
        coordinates, mask, target,
    )

    assert distribution.shape == mask.shape
    assert distribution.sum().item() == pytest.approx(1.0)
    assert distribution[0, 4].item() == 0.0
    assert distribution.argmax(dim=-1).item() == 3


def test_attention_kl_is_lower_at_target_and_backpropagates():
    torch = pytest.importorskip("torch")
    from web_agent.models.bbox import (
        bbox_attention_kl_loss,
        bbox_attention_target_distribution,
    )

    coordinates = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    target_box = torch.tensor([[0.65, 0.65, 0.20, 0.20]])
    target_attention = bbox_attention_target_distribution(
        coordinates, mask, target_box,
    )
    wrong_logits = torch.tensor([[4.0, -4.0]], requires_grad=True)
    wrong_attention = torch.softmax(wrong_logits, dim=-1)

    correct_loss = bbox_attention_kl_loss(
        target_attention, coordinates, mask, target_box,
    )
    wrong_loss = bbox_attention_kl_loss(
        wrong_attention, coordinates, mask, target_box,
    )
    wrong_loss.backward()

    assert correct_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert wrong_loss.item() > correct_loss.item()
    assert torch.isfinite(wrong_logits.grad).all()
    assert wrong_logits.grad.norm().item() > 0.0


def test_action_head_exposes_stable_attention_contract():
    torch = pytest.importorskip("torch")
    from web_agent.models.heads import ActionHead

    head = ActionHead(
        dim=8,
        hidden=4,
        dropout=0.3,
        spatial_grounding=True,
        bbox_parameterization="cxcywh",
        bbox_grounding_mode="coordinate_softargmax",
        bbox_size_parameterization="log_space",
        bbox_attention_dropout=0.0,
    )
    mask = torch.tensor([[True, True, False]])
    coordinates = torch.tensor([[
        [0.25, 0.25], [0.75, 0.75], [0.0, 0.0],
    ]])
    output = head(
        torch.randn(1, 8),
        bbox_fused=torch.randn(1, 8),
        spatial_tokens=torch.randn(1, 3, 8),
        spatial_mask=mask,
        spatial_coords=coordinates,
    )

    assert head.grounding_attention.dropout == pytest.approx(0.0)
    assert output["bbox_attention_weights"].shape == (1, 3)
    assert torch.equal(output["bbox_spatial_mask"], mask)
    assert torch.equal(output["bbox_spatial_coords"], coordinates)
    assert output["bbox_attention_weights"][0, 2].item() == 0.0
    assert output["bbox_attention_weights"].sum().item() == pytest.approx(1.0)


def test_combined_loss_routes_attention_supervision_and_backward():
    torch = pytest.importorskip("torch")
    from web_agent.models.loss import CombinedLoss

    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_5.yaml")
    loss_fn = CombinedLoss(cfg)
    predicted_log_wh = torch.tensor(
        [[math.log(0.20), math.log(0.10)]], requires_grad=True,
    )
    prediction_cxcywh = torch.cat([
        torch.tensor([[0.75, 0.75]]), torch.exp(predicted_log_wh),
    ], dim=-1)
    attention_logits = torch.tensor([[3.0, -3.0]], requires_grad=True)
    attention = torch.softmax(attention_logits, dim=-1)
    predictions = {
        "outcome": torch.randn(1, 2, requires_grad=True),
        "failure_type": torch.randn(1, 4, requires_grad=True),
        "action_type": torch.randn(1, 6, requires_grad=True),
        "recovery": torch.randn(1, 6, requires_grad=True),
        "memory_recovery": torch.randn(1, 6, requires_grad=True),
        "needs_recovery": torch.randn(1, 1, requires_grad=True),
        "bbox": prediction_cxcywh,
        "bbox_cxcywh": prediction_cxcywh,
        "bbox_log_wh": predicted_log_wh,
        "bbox_attention_weights": attention,
        "bbox_spatial_coords": torch.tensor([[
            [0.25, 0.25], [0.75, 0.75],
        ]]),
        "bbox_spatial_mask": torch.ones(1, 2, dtype=torch.bool),
        "memory_flag": torch.randn(1, 1, requires_grad=True),
        "recovery_outcome": torch.randn(1, 1, requires_grad=True),
        "confidence": torch.sigmoid(torch.randn(1, 1, requires_grad=True)),
        "fused": torch.randn(1, 768, requires_grad=True),
    }
    batch = {
        "label_outcome": torch.tensor([1]),
        "label_failtype": torch.tensor([1]),
        "label_action": torch.tensor([0]),
        "label_recovery": torch.tensor([1]),
        "label_needs_recovery": torch.ones(1, 1),
        "bbox": torch.tensor([[0.65, 0.65, 0.20, 0.10]]),
        "bbox_mask": torch.ones(1, 1),
        "label_memory": torch.ones(1, 1),
        "label_recovery_success": torch.ones(1, 1),
        "label_confidence": torch.tensor([[0.2]]),
        "original_task_id": ["task-a"],
    }

    terms = loss_fn(predictions, batch)
    terms["total"].backward()

    assert terms["bbox_attention_kl"].item() > 0.0
    assert torch.isfinite(terms["total"])
    assert torch.isfinite(attention_logits.grad).all()
    assert attention_logits.grad.norm().item() > 0.0


def test_v2_5_changes_only_registered_attention_controls():
    previous = load_config("configs/backbones/qwen2vl_2b_gold_v2_4.yaml")
    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_5.yaml")

    assert cfg["train"]["controlled_experiment_tag"] == "recovery_v2_5"
    assert cfg["model"]["bbox_attention_dropout"] == pytest.approx(0.0)
    assert cfg["loss"]["bbox_attention_ratio"] == pytest.approx(1.0)
    assert cfg["train"]["bbox_overfit_full_eval_steps"] == [25, 50, 75]
    assert cfg["optim"] == previous["optim"]
    for key in (
        "bbox_overfit_rows",
        "bbox_overfit_steps",
        "bbox_overfit_min_iou_gain",
        "bbox_overfit_min_mean_iou",
        "bbox_overfit_max_full_screen_fraction",
        "bbox_overfit_min_predicted_size",
        "bbox_overfit_fp32_grounding",
        "bbox_overfit_disable_dropout",
        "bbox_overfit_require_finite_gradients",
    ):
        assert cfg["train"][key] == previous["train"][key]
    for key in ("bbox_l1_ratio", "bbox_giou_ratio", "bbox_loss"):
        assert cfg["loss"][key] == previous["loss"][key]
    assert cfg["model"]["bbox_size_parameterization"] == "log_space"


def test_v2_5_notebook_is_isolated_output_free_and_compiles():
    path = Path("notebooks/kaggle_gold_recovery_v2_5.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    all_source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "qwen2vl_2b_gold_v2_5.yaml" in all_source
    assert "STAGE = 'smoke'" in all_source
    assert "BBOX_MONTAGE_REVIEWED = False" in all_source
    assert "bbox_attention_ratio'] == 1.0" in all_source
    assert "grounding_attention" in all_source
    assert "[0, 25, 50, 75, 100]" in all_source
    assert "gold_recovery_v2_4" not in all_source
    assert Path("docs/RECOVERY_V2_5_EXPERIMENT.md").is_file()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
