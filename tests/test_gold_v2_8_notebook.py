from __future__ import annotations

import ast
import json
from pathlib import Path

from web_agent.config import load_config


def test_v2_8_config_changes_data_source_not_locked_test_policy():
    cfg = load_config("configs/backbones/qwen2vl_2b_gold_v2_8.yaml")

    assert cfg["train"]["controlled_experiment_tag"] == "recovery_v2_8"
    assert cfg["train"]["quality_selection_rule"] == (
        "all_gates_then_outcome_mcc"
    )
    assert cfg["data"]["recovery_supplement"]["enabled"] is True
    assert (
        cfg["data"]["recovery_supplement"][
            "include_in_primary_validation"
        ]
        is False
    )


def test_v2_8_notebook_is_output_free_locked_and_compiles():
    path = Path("notebooks/kaggle_gold_recovery_v2_8.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )

    assert "qwen2vl_2b_gold_v2_8.yaml" in source
    assert "gold-40k-retry" in source
    assert "review_reconciliation_report.json" in source
    assert "NO_CHANGE_REVIEW_CONFIRMED = True" in source
    assert "NO_CHANGE_REVIEWER_COUNT = 2" in source
    assert "two_person_manual_no_change_attestation" in source
    assert "no row-level reconciliation" in source
    assert "'--train-rows', str(TRAIN_ROWS)" in source
    assert "'--val-rows', str(ORIGINAL_VAL_ROWS)" in source
    assert "'--epochs', str(EPOCHS)" in source
    assert "EPOCHS = 5" in source
    assert "TRAIN_ROWS = 5_000" in source
    assert "SUPPLEMENT_VAL_ROWS = 194" in source
    assert "include_in_primary_validation'] = False" in source
    assert "sys.path.insert(0, str(SOURCE_ROOT))" in source
    assert "os.environ['PYTHONPATH']" in source
    assert "import web_agent" in source
    assert "test_rows_read" in source
    assert "source_validation" in source
    assert "FULL_TRAINING_PERMITTED" not in source
    assert "kaggle_gold.ipynb" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            ast.parse("".join(cell["source"]))


def test_mini_status_follows_quality_gate_status():
    source = Path("src/web_agent/train/gold_stages.py").read_text(
        encoding="utf-8"
    )

    assert '"status": quality_gates["status"]' in source
    assert '"training_disposition": (' in source
    assert '"STOP_BEFORE_FULL_TRAINING"' in source
