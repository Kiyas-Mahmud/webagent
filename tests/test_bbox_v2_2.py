from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from web_agent.config import load_config
from web_agent.data.bbox_audit import audit_bbox_geometry


def _record(box: dict, sample_id: str = "sample-1") -> dict:
    return {
        "inputs": {"state_before": "images/page.png"},
        "labels": {"action_target_bbox": box},
        "meta": {"sample_id": sample_id},
    }


def test_bbox_audit_accepts_complete_in_image_box(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 80)).save(image_dir / "page.png")

    report = audit_bbox_geometry([
        _record({"x": 10, "y": 20, "width": 30, "height": 15}),
    ], tmp_path)

    assert report["status"] == "PASS"
    assert report["bbox_rows"] == 1
    assert report["invalid_bbox_rows"] == 0
    assert report["normalized_valid_distribution"]["x"]["mean"] == pytest.approx(0.1)


def test_bbox_audit_reports_boundary_overflow_without_repairing(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 80)).save(image_dir / "page.png")
    box = {"x": 90, "y": 75, "width": 20, "height": 10}

    report = audit_bbox_geometry([_record(box)], tmp_path)

    assert report["status"] == "FAIL"
    assert report["invalid_bbox_rows"] == 1
    assert report["invalid_reason_counts"] == {
        "bottom_boundary_overflow": 1,
        "right_boundary_overflow": 1,
    }
    assert report["invalid_examples"][0]["bbox"] == box


def test_centre_bbox_conversion_and_detr_loss_have_useful_gradient():
    torch = pytest.importorskip("torch")
    from web_agent.models.bbox import (
        cxcywh_to_bounded_xywh,
        detr_bbox_loss_terms,
        xywh_to_cxcywh,
    )

    target_xywh = torch.tensor([[0.40, 0.30, 0.10, 0.20]])
    exact = xywh_to_cxcywh(target_xywh)
    decoded = cxcywh_to_bounded_xywh(exact)
    assert torch.allclose(decoded, target_xywh)

    full_screen = torch.tensor([[0.50, 0.50, 1.00, 1.00]], requires_grad=True)
    l1, giou = detr_bbox_loss_terms(full_screen, target_xywh)
    total = 5.0 * l1 + 2.0 * giou
    total.backward()
    assert total.item() > 0.0
    assert full_screen.grad is not None
    assert full_screen.grad.norm().item() > 0.0

    exact_l1, exact_giou = detr_bbox_loss_terms(exact, target_xywh)
    assert exact_l1.item() == pytest.approx(0.0, abs=1e-7)
    assert exact_giou.item() == pytest.approx(0.0, abs=1e-6)


def test_action_head_exposes_internal_centre_box_and_stable_xywh():
    torch = pytest.importorskip("torch")
    from web_agent.models.heads import ActionHead

    head = ActionHead(
        dim=8,
        hidden=4,
        dropout=0.0,
        spatial_grounding=True,
        bbox_parameterization="cxcywh",
    )
    result = head(
        torch.randn(2, 8),
        bbox_fused=torch.randn(2, 8),
        spatial_tokens=torch.randn(2, 3, 8),
        spatial_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    assert result["bbox_cxcywh"].shape == (2, 4)
    assert result["bbox"].shape == (2, 4)
    assert torch.all((result["bbox_cxcywh"] > 0) & (result["bbox_cxcywh"] < 1))
    assert torch.all((result["bbox"] >= 0) & (result["bbox"] <= 1))
    assert torch.all(result["bbox"][:, :2] + result["bbox"][:, 2:] <= 1.0 + 1e-6)


def test_v2_2_config_changes_only_registered_bbox_controls():
    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_2.yaml")
    assert cfg["data"]["strict_bbox_geometry"] is True
    assert cfg["model"]["bbox_parameterization"] == "cxcywh"
    assert cfg["model"]["task_adapters"]["separate_bbox"] is True
    assert cfg["loss"]["bbox_loss"] == "detr_l1_giou"
    assert cfg["loss"]["bbox_l1_ratio"] == 5.0
    assert cfg["loss"]["bbox_giou_ratio"] == 2.0
    assert cfg["loss"]["needs_recovery_weight_scheme"] == "exact_inverse_frequency"
    assert cfg["train"]["controlled_experiment_tag"] == "recovery_v2_2"


def test_v2_2_notebook_is_output_free_and_every_code_cell_compiles():
    path = Path("notebooks/kaggle_gold_recovery_v2_2.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
