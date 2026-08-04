from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from web_agent.config import load_config
from web_agent.models.encoders.vlm_contract import get_vlm_contract


def test_qwen2_2b_reference_is_pinned_for_comparison():
    cfg = load_config(
        "configs/backbones/qwen2vl_2b_gold_v2_8_dgx.yaml"
    )

    assert cfg["backbone"]["family"] == "qwen2_vl"
    assert len(cfg["backbone"]["revision"]) == 40
    assert cfg["backbone"]["trust_remote_code"] is False
    assert cfg["backbone"]["vlm_hidden_dim"] == 1536
    assert cfg["optim"]["batch_size"] * cfg["optim"]["grad_accum"] == 32
    assert cfg["train"]["epochs"] == 10


def test_qwen25_7b_candidate_preserves_gold_protocol():
    cfg = load_config(
        "configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml"
    )

    assert cfg["backbone"]["family"] == "qwen2_vl"
    assert cfg["backbone"]["vlm_model"] == (
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    assert len(cfg["backbone"]["revision"]) == 40
    assert cfg["backbone"]["vlm_hidden_dim"] == 3584
    assert cfg["backbone"]["min_pixels"] == 50_176
    assert cfg["backbone"]["max_pixels"] == 200_704
    assert cfg["optim"]["batch_size"] * cfg["optim"]["grad_accum"] == 32
    assert cfg["optim"]["mixed_precision"] == "bf16"
    assert cfg["train"]["epochs"] == 10
    assert cfg["data"]["causal_routing"] is True
    assert cfg["data"]["recovery_transitions"] is True
    assert cfg["loss"]["hierarchical_recovery"] is True
    assert cfg["train"]["quality_selection_rule"] == (
        "all_gates_then_outcome_mcc"
    )


def test_internvl35_candidate_preserves_gold_protocol():
    cfg = load_config(
        "configs/backbones/internvl35_8b_gold_v2_8_dgx.yaml"
    )

    assert cfg["backbone"]["family"] == "internvl_hf"
    assert cfg["backbone"]["vlm_model"] == (
        "OpenGVLab/InternVL3_5-8B-HF"
    )
    assert len(cfg["backbone"]["revision"]) == 40
    assert cfg["backbone"]["trust_remote_code"] is False
    assert cfg["backbone"]["vlm_hidden_dim"] == 4096
    assert cfg["backbone"]["crop_to_patches"] is False
    assert cfg["backbone"]["image_seq_length"] == 256
    assert cfg["optim"]["batch_size"] * cfg["optim"]["grad_accum"] == 32
    assert cfg["optim"]["mixed_precision"] == "bf16"
    assert cfg["train"]["epochs"] == 10
    assert cfg["data"]["causal_routing"] is True
    assert cfg["data"]["recovery_transitions"] is True
    assert cfg["loss"]["hierarchical_recovery"] is True
    assert cfg["train"]["quality_selection_rule"] == (
        "all_gates_then_outcome_mcc"
    )


def test_family_contracts_keep_processor_differences_explicit():
    qwen = get_vlm_contract(load_config(
        "configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml"
    ))
    internvl = get_vlm_contract(load_config(
        "configs/backbones/internvl35_8b_gold_v2_8_dgx.yaml"
    ))

    assert "image_grid_thw" in qwen.required_input_keys
    assert qwen.coordinate_mode == "qwen_grid"
    assert qwen.processor_uses_pixel_bounds is True
    assert "image_grid_thw" not in internvl.model_input_keys
    assert "image_sizes" in internvl.optional_input_keys
    assert internvl.coordinate_mode == "fixed_square"
    assert internvl.processor_uses_pixel_bounds is False
    assert internvl.requires_single_patch_images is True


def test_internvl_collate_preserves_multi_image_cardinality():
    torch = pytest.importorskip("torch")
    from web_agent.data.gold_dataloader import causal_gold_collate

    cfg = load_config(
        "configs/backbones/internvl35_8b_gold_v2_8_dgx.yaml"
    )

    def row(token_count: int) -> dict:
        labels = {
            "bbox": torch.zeros(4),
            "bbox_mask": torch.ones(1),
            "borrowed": torch.zeros(1),
            "label_outcome": torch.tensor(0),
            "label_failtype": torch.tensor(0),
            "label_action": torch.tensor(0),
            "label_recovery": torch.tensor(0),
            "label_memory": torch.zeros(1),
            "label_confidence": torch.zeros(1),
            "label_recovery_success": torch.tensor([-1.0]),
            "label_needs_recovery": torch.zeros(1),
            "original_task_id": "task",
            "source_dataset": "original_gold",
        }
        for prefix, images in (("pre_", 1), ("post_", 2)):
            labels[f"{prefix}input_ids"] = torch.ones(
                token_count * images, dtype=torch.long
            )
            labels[f"{prefix}attention_mask"] = torch.ones(
                token_count * images, dtype=torch.long
            )
            labels[f"{prefix}pixel_values"] = torch.zeros(
                images, 3, 4, 4
            )
            labels[f"{prefix}image_count"] = torch.tensor([images])
        return labels

    batch = causal_gold_collate([row(4), row(5)], cfg=cfg)
    assert batch["pre_input_ids"].shape == (2, 5)
    assert batch["post_input_ids"].shape == (2, 10)
    assert batch["pre_pixel_values"].shape[0] == 2
    assert batch["post_pixel_values"].shape[0] == 4
    assert batch["pre_image_counts"].tolist() == [1, 1]
    assert batch["post_image_counts"].tolist() == [2, 2]


def test_fixed_square_coordinates_cover_each_image_grid():
    torch = pytest.importorskip("torch")
    from web_agent.models.spatial import normalized_fixed_square_coordinates

    mask = torch.tensor([
        [False, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, True, True, False],
    ])
    coords = normalized_fixed_square_coordinates(
        mask,
        torch.tensor([1, 2]),
        tokens_per_image=4,
    )
    expected = torch.tensor([
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]
    ])
    assert torch.allclose(coords[0, mask[0]], expected)
    assert torch.allclose(coords[1, mask[1]], expected.repeat(2, 1))


def test_three_model_notebook_is_output_free_and_compiles():
    path = Path("notebooks/dgx_three_model_comparison.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "qwen2vl_2b_gold_v2_8_dgx" in source
    assert "qwen25vl_7b_gold_v2_8_dgx" in source
    assert "internvl35_8b_gold_v2_8_dgx" in source
    assert "PC 1" in source and "PC 2" in source and "PC 3" in source
    assert "all_gates_then_outcome_mcc" in source
    assert "test_rows_read" in source
    assert "split_test.json" in source
    assert "run_candidate" in source
    assert "phase='auto'" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            ast.parse("".join(cell["source"]))
