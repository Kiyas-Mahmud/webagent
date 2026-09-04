from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from web_agent.eval.table2.common import SchemaError, atomic_write_json
from web_agent.eval.table2 import package_validator as validator


def _install_deterministic_recomputation_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        "episodes": [
            {
                "episode_id": "normal-E0",
                "task_id": "normal-task",
                "task_partition": "normal",
                "system_id": "E0",
                "task_success": True,
                "step_count": 1,
                "loop_detected": False,
                "task_wall_clock_seconds": None,
                "model_call_count": None,
            },
            {
                "episode_id": "recovery-E0",
                "task_id": "recovery-task",
                "task_partition": "recovery_diagnostic",
                "system_id": "E0",
                "task_success": False,
                "step_count": 2,
                "loop_detected": True,
                "task_wall_clock_seconds": None,
                "model_call_count": None,
            },
        ],
        "recovery_attempts": [],
        "failure_incidents": [],
        "memory_queries": [
            {"episode_id": "normal-E0", "task_partition": "normal"},
            {
                "episode_id": "recovery-E0",
                "task_partition": "recovery_diagnostic",
            },
        ],
        "schedule_attempts": [{"task_partition": "normal"}],
    }
    monkeypatch.setattr(
        validator,
        "_load_selected_analysis_records_unchecked",
        lambda _root: deepcopy(records),
    )

    def fake_metrics(episodes: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        partition = str(episodes[0]["task_partition"])
        return {
            "schema_version": "table2.v1",
            "systems": {
                system_id: {"partition": partition, "estimate": 0.625}
                for system_id in ("E0", "E1", "E2", "E3")
            },
        }

    monkeypatch.setattr(validator, "compute_table2_metrics", fake_metrics)
    monkeypatch.setattr(
        validator,
        "compute_retrieval_diagnostics",
        lambda queries, **_kwargs: {
            "schema_version": "table2.v1",
            "query_count": len(list(queries)),
            "estimate": 0.75,
        },
    )
    monkeypatch.setattr(
        validator,
        "compute_paired_contrasts",
        lambda *_args, **_kwargs: {
            "schema_version": "table2.v1",
            "contrasts": {"E1_minus_E0": {"estimate": 0.125}},
        },
    )
    monkeypatch.setattr(
        validator,
        "compute_clustered_ratio_contrasts",
        lambda *_args, **_kwargs: {"E1_minus_E0": {"estimate": 0.25}},
    )
    # This unit isolates the five canonical JSON recomputations. Exact CSV and
    # redacted-result byte replay have separate full-package tamper canaries.
    monkeypatch.setattr(
        validator,
        "_validate_canonical_summary_exports",
        lambda *_args, **_kwargs: None,
    )


def _write_expected_aggregates(root: Path) -> dict[str, dict[str, Any]]:
    expected = {
        "metrics.json": {
            "schema_version": "table2.v1",
            "systems": {
                system_id: {"partition": "normal", "estimate": 0.625}
                for system_id in ("E0", "E1", "E2", "E3")
            },
            "publication_status": "DRAFT_PILOT_ONLY",
            "headline_task_partition": "normal",
            "paper_table_status": "N/R",
        },
        "recovery_diagnostic_metrics.json": {
            "schema_version": "table2.v1",
            "systems": {
                system_id: {
                    "partition": "recovery_diagnostic",
                    "estimate": 0.625,
                }
                for system_id in ("E0", "E1", "E2", "E3")
            },
            "publication_status": "DRAFT_PILOT_ONLY",
            "task_partition": "recovery_diagnostic",
        },
        "retrieval_diagnostics.json": {
            "schema_version": "table2.v1",
            "query_count": 1,
            "estimate": 0.75,
            "publication_status": "DRAFT_PILOT_ONLY",
        },
        "recovery_diagnostic_retrieval.json": {
            "schema_version": "table2.v1",
            "query_count": 1,
            "estimate": 0.75,
            "publication_status": "DRAFT_PILOT_ONLY",
        },
        "statistics.json": {
            "schema_version": "table2.v1",
            "contrasts": {"E1_minus_E0": {"estimate": 0.125}},
            "clustered_ratio_contrasts": {
                "E1_minus_E0": {"estimate": 0.25}
            },
            "publication_status": "DRAFT_PILOT_ONLY",
        },
    }
    for filename, payload in expected.items():
        atomic_write_json(root / "aggregate" / filename, payload)
    return expected


def test_recomputation_rejects_every_tampered_numeric_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {
        "budgets": {"recovery_at_k": 2},
        "statistics": {
            "bootstrap_samples": 10,
            "confidence_level": 0.95,
            "seed": 42,
        },
    }
    protocol_path = tmp_path / "frozen" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    _install_deterministic_recomputation_stubs(monkeypatch)
    expected = _write_expected_aggregates(tmp_path)
    campaign_manifest = {"evidence_label": "PILOT_ONLY"}

    # The untouched aggregate is accepted, proving the five independent
    # reconstruction branches agree on their registered source evidence.
    validator._validate_recomputed_aggregate_outputs(
        tmp_path,
        publication_status="DRAFT_PILOT_ONLY",
        campaign_manifest=campaign_manifest,
    )

    for filename in expected:
        _write_expected_aggregates(tmp_path)
        tampered = deepcopy(expected[filename])
        if filename in {
            "metrics.json",
            "recovery_diagnostic_metrics.json",
        }:
            tampered["systems"]["E0"]["estimate"] = 0.999
        elif filename in {
            "retrieval_diagnostics.json",
            "recovery_diagnostic_retrieval.json",
        }:
            tampered["estimate"] = 0.999
        else:
            tampered["contrasts"]["E1_minus_E0"]["estimate"] = 0.999
        atomic_write_json(tmp_path / "aggregate" / filename, tampered)
        with pytest.raises(
            SchemaError,
            match=rf"aggregate numeric recomputation mismatch: {filename}",
        ):
            validator._validate_recomputed_aggregate_outputs(
                tmp_path,
                publication_status="DRAFT_PILOT_ONLY",
                campaign_manifest=campaign_manifest,
            )


def test_json_mismatch_is_type_sensitive_for_boolean_number_canary() -> None:
    mismatch = validator._first_json_mismatch(
        {"estimate": 1},
        {"estimate": True},
    )
    assert mismatch == ("$.estimate", 1, True)
