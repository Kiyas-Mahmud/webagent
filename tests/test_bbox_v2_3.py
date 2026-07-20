from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.config import load_config


def test_qwen_grid_becomes_normalized_raster_patch_centres():
    torch = pytest.importorskip("torch")
    from web_agent.models.spatial import normalized_spatial_coordinates

    mask = torch.tensor([[True, True, True, True, False]])
    coordinates = normalized_spatial_coordinates(
        mask,
        torch.tensor([[1, 4, 4]]),
        spatial_merge_size=2,
    )

    expected = torch.tensor([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
        [0.00, 0.00],
    ])
    assert torch.allclose(coordinates[0], expected)


def test_spatial_soft_argmax_moves_to_the_attended_patch():
    torch = pytest.importorskip("torch")
    from web_agent.models.spatial import spatial_soft_argmax

    coordinates = torch.tensor([[[0.25, 0.25], [0.75, 0.75], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    first, first_entropy = spatial_soft_argmax(
        torch.tensor([[1.0, 0.0, 0.0]]), coordinates, mask,
    )
    second, second_entropy = spatial_soft_argmax(
        torch.tensor([[0.0, 1.0, 1.0]]), coordinates, mask,
    )

    assert torch.allclose(first, torch.tensor([[0.25, 0.25]]))
    assert torch.allclose(second, torch.tensor([[0.75, 0.75]]))
    assert torch.isfinite(first_entropy).all()
    assert torch.isfinite(second_entropy).all()


def test_coordinate_action_head_is_differentiable_end_to_end():
    torch = pytest.importorskip("torch")
    from web_agent.models.heads import ActionHead

    torch.manual_seed(42)
    head = ActionHead(
        dim=8,
        hidden=4,
        dropout=0.0,
        spatial_grounding=True,
        bbox_parameterization="cxcywh",
        bbox_grounding_mode="coordinate_softargmax",
    )
    spatial_coordinates = torch.tensor([
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
    ])
    output = head(
        torch.randn(2, 8),
        bbox_fused=torch.randn(2, 8),
        spatial_tokens=torch.randn(2, 4, 8),
        spatial_mask=torch.ones(2, 4, dtype=torch.bool),
        spatial_coords=spatial_coordinates,
    )
    target = torch.tensor([[0.2, 0.2, 0.3, 0.2], [0.6, 0.5, 0.2, 0.3]])
    loss = torch.nn.functional.smooth_l1_loss(output["bbox"], target)
    loss.backward()

    assert output["bbox"].shape == (2, 4)
    assert output["bbox_cxcywh"].shape == (2, 4)
    assert output["bbox_attention_entropy"].shape == (2,)
    for module in (
        head.coordinate_projection,
        head.grounding_attention,
        head.bbox,
    ):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.norm().item() > 0 for gradient in gradients)


def test_v2_3_config_keeps_v2_2_gates_and_enables_only_registered_controls():
    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_3.yaml")

    assert cfg["train"]["controlled_experiment_tag"] == "recovery_v2_3"
    assert cfg["model"]["bbox_grounding_mode"] == "coordinate_softargmax"
    assert cfg["model"]["bbox_fp32_grounding"] is True
    assert cfg["train"]["bbox_overfit_fp32_grounding"] is True
    assert cfg["train"]["bbox_overfit_disable_dropout"] is True
    assert cfg["train"]["bbox_overfit_require_finite_gradients"] is True
    assert cfg["train"]["bbox_overfit_rows"] == 32
    assert cfg["train"]["bbox_overfit_steps"] == 100
    assert cfg["train"]["bbox_overfit_min_iou_gain"] == pytest.approx(0.10)
    assert cfg["train"]["bbox_overfit_min_mean_iou"] == pytest.approx(0.20)
    assert cfg["train"]["bbox_overfit_max_full_screen_fraction"] == pytest.approx(0.10)
    assert cfg["loss"]["bbox_l1_ratio"] == pytest.approx(5.0)
    assert cfg["loss"]["bbox_giou_ratio"] == pytest.approx(2.0)


def test_v2_3_notebook_is_isolated_output_free_and_compiles():
    path = Path("notebooks/kaggle_gold_recovery_v2_3.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    all_source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "qwen2vl_2b_gold_v2_3.yaml" in all_source
    assert "STAGE = 'smoke'" in all_source
    assert "BBOX_MONTAGE_REVIEWED = False" in all_source
    assert "save_gold_bbox_probe_montage" in all_source
    assert "coordinate_projection" in all_source
    assert "gold_recovery_v2_2" not in all_source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
