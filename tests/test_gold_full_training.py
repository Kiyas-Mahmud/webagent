from __future__ import annotations

import ast
import json
from pathlib import Path

from web_agent.config import load_config


def test_qwen25_gold_candidate_keeps_v2_8_causal_contract():
    cfg = load_config("configs/backbones/qwen25vl_3b_gold_v2_8.yaml")

    assert cfg["backbone"]["path"] == "vlm"
    assert cfg["backbone"]["vlm_model"] == (
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    assert cfg["backbone"]["vlm_hidden_dim"] == 2048
    assert cfg["optim"]["mixed_precision"] == "bf16"
    assert cfg["data"]["causal_routing"] is True
    assert cfg["data"]["recovery_transitions"] is True
    assert cfg["loss"]["hierarchical_recovery"] is True
    assert cfg["train"]["quality_selection_rule"] == (
        "all_gates_then_outcome_mcc"
    )


def test_dgx_full_notebook_is_locked_resume_safe_and_compiles():
    path = Path("notebooks/dgx_gold_full_training.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "ACTIVE_MODEL_ID = 'qwen2vl_2b_gold_v2_8'" in source
    assert "MAX_EPOCHS = 15" in source
    assert "CHECKPOINT_EVERY_STEPS = 250" in source
    assert "--stage', 'full'" in source
    assert "--resume-checkpoint" in source
    assert "last.ckpt" in source
    assert "epoch_metrics.csv" in source
    assert "source_validation.csv" in source
    assert "full_model_results.csv" in source
    assert "ignore_patterns=['split_test.json', '**/split_test.json']" in source
    assert "assert not (ORIGINAL_ROOT / 'split_test.json').exists()" in source
    assert "test_rows_read" in source
    assert "24_107" in source
    assert "7_861" in source
    assert "qwen25vl_3b_gold_v2_8" in source
    assert "'full_authorized': False" in source
    assert "HF_TOKEN = os.environ.get('HF_TOKEN')" in source
    assert "REPLACE_WITH_HF_COMMIT_SHA" in source
    assert "kaggle_gold.ipynb" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            ast.parse("".join(cell["source"]))


def test_full_runner_and_exact_batch_resume_contract_are_present():
    cli = Path("scripts/run_gold.py").read_text(encoding="utf-8")
    full = Path("src/web_agent/train/gold_full.py").read_text(
        encoding="utf-8"
    )
    trainer = Path("src/web_agent/train/trainer.py").read_text(
        encoding="utf-8"
    )

    ast.parse(cli)
    ast.parse(full)
    ast.parse(trainer)
    assert 'choices=("smoke", "mini", "full", "reeval", "eval")' in cli
    assert "run_gold_full" in cli
    assert '"test_rows_read": 0' in full
    assert '"checkpoint_selection_source": "original_gold_validation_only"' in full
    assert "supplement_validation_selects_checkpoint" in full
    assert '"batch_in_epoch"' in trainer
    assert '"epoch_state"' in trainer
    assert "self.resume_batch_in_epoch" in trainer
    assert "if i < self.resume_batch_in_epoch" in trainer
    assert 'path.with_suffix(path.suffix + ".tmp")' in trainer
    assert "temporary.replace(path)" in trainer
    assert '"rng_state"' in trainer
    assert '"early_stop_state"' in trainer
