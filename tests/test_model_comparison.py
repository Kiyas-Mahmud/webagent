from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_full_models import MODEL_IDS, main


def _write_candidate(
    root: Path,
    model_id: str,
    *,
    outcome_mcc: float,
    git_commit: str = "a" * 40,
) -> None:
    seed_root = root / model_id / "seed_42"
    full_root = seed_root / "full"
    full_root.mkdir(parents=True)
    contract = {
        "candidate": {"model_id": model_id},
        "git_commit": git_commit,
        "seed": 42,
        "checkpoint_selection": "all_gates_then_outcome_mcc",
        "test_rows_read": 0,
    }
    (seed_root / "run_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    metrics = {
        "epoch": 3,
        "outcome_mcc": outcome_mcc,
        "failure_macro_f1": 0.70,
        "failtype_macro_f1": 0.50,
        "action_macro_f1": 0.40,
        "needs_recovery_macro_f1": 0.60,
        "strategy_attempted_macro_f1": 0.35,
        "recovery_outcome_mcc": 0.55,
        "memory_mcc": 0.65,
        "bbox_mean_iou": 0.10,
        "bbox_recall_iou50": 0.05,
        "outcome_ece": 0.08,
    }
    report = {
        "status": "PASS",
        "seed": 42,
        "requested_epochs": 10,
        "test_rows_read": 0,
        "train_rows": 24_107,
        "val_rows": 7_861,
        "selection_rule": "all_gates_then_outcome_mcc",
        "selected_epoch": 3,
        "best_checkpoint": f"{model_id}/best.ckpt",
        "selected_checkpoint_sha256": "b" * 64,
        "history": [metrics],
        "source_validation": {
            "supplement_retry_abort": {"rows": 194},
        },
    }
    (full_root / "report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_comparison_ranks_only_same_commit_validation_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "model_comparison"
    scores = (0.51, 0.61, 0.56)
    for model_id, score in zip(MODEL_IDS, scores):
        _write_candidate(root, model_id, outcome_mcc=score)
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_full_models.py",
            "--comparison-root", str(root),
            "--seed", "42",
            "--csv", str(csv_path),
            "--json", str(json_path),
        ],
    )

    main()

    decision = json.loads(json_path.read_text(encoding="utf-8"))
    assert decision["status"] == "PASS"
    assert decision["selected_model_id"] == MODEL_IDS[1]
    assert decision["git_commit"] == "a" * 40
    assert decision["test_rows_read"] == 0
    assert csv_path.is_file()


def test_comparison_rejects_mixed_code_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "model_comparison"
    for index, model_id in enumerate(MODEL_IDS):
        _write_candidate(
            root,
            model_id,
            outcome_mcc=0.50 + index / 100,
            git_commit=("a" if index < 2 else "c") * 40,
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_full_models.py",
            "--comparison-root", str(root),
            "--seed", "42",
            "--csv", str(tmp_path / "comparison.csv"),
            "--json", str(tmp_path / "comparison.json"),
        ],
    )

    with pytest.raises(AssertionError, match="different Git commits"):
        main()
