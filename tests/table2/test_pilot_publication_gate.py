from __future__ import annotations

from pathlib import Path
import json

import pytest

from web_agent.eval.table2.common import SchemaError, Table2Error
from web_agent.eval.table2.package_validator import (
    DRAFT_PILOT_STATUS,
    ValidationReport,
    _publication_status,
    adjudication_gated_pilot_publication_status,
)
from web_agent.eval.table2 import package_validator as package_validator_module
from web_agent.eval.table2 import campaign as campaign_module
from web_agent.eval.table2 import summary as summary_module


PILOT = {
    "campaign_id": "pilot",
    "campaign_kind": "engineering_pilot",
    "campaign_mode": "evaluation",
    "evidence_label": "PILOT_ONLY",
}


def _write_terminal_pilot_completion(root: Path) -> None:
    (root / "campaign_manifest.json").write_text(
        json.dumps(PILOT), encoding="utf-8"
    )
    (root / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": "table2.v1",
                "campaign_id": PILOT["campaign_id"],
                "status": "COMPLETE",
                "scheduled_block_count": 2,
                "processed_block_count": 2,
                "included_block_count": 2,
                "infrastructure_excluded_block_count": 0,
                "publication_status": DRAFT_PILOT_STATUS,
            }
        ),
        encoding="utf-8",
    )


def test_pre_adjudication_export_requires_explicit_draft_status(tmp_path: Path) -> None:
    with pytest.raises(
        Table2Error,
        match="final PILOT_ONLY summary/export requires completed",
    ):
        summary_module._summary_publication_status(
            tmp_path,
            PILOT,
            validation_status="PILOT_ONLY",
            draft_pilot=False,
        )

    assert (
        summary_module._summary_publication_status(
            tmp_path,
            PILOT,
            validation_status="PILOT_ONLY",
            draft_pilot=True,
        )
        == DRAFT_PILOT_STATUS
    )


def test_completed_adjudication_promotes_pilot_from_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_terminal_pilot_completion(tmp_path)
    calls: list[Path] = []

    def completed(root: Path) -> dict:
        calls.append(root)
        return {"completion_status": "HUMAN_ADJUDICATION_COMPLETE"}

    monkeypatch.setattr(
        package_validator_module,
        "validate_manual_adjudication_completion",
        completed,
    )
    assert (
        summary_module._summary_publication_status(
            tmp_path,
            PILOT,
            validation_status=DRAFT_PILOT_STATUS,
            draft_pilot=False,
        )
        == "PILOT_ONLY"
    )
    assert calls == [tmp_path]


def test_default_draft_and_adjudicated_results_use_distinct_directories(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "artifacts" / "table2" / "pilot"
    assert summary_module._resolve_results_dir(
        campaign,
        campaign_id="pilot",
        requested=None,
        draft_pilot=True,
    ) == (tmp_path / "results" / "table2" / "pilot" / "draft").resolve()
    assert summary_module._resolve_results_dir(
        campaign,
        campaign_id="pilot",
        requested=None,
        draft_pilot=False,
    ) == (tmp_path / "results" / "table2" / "pilot").resolve()


def test_draft_flag_cannot_downgrade_a_locked_final_campaign(tmp_path: Path) -> None:
    with pytest.raises(Table2Error, match="valid only for a PILOT_ONLY"):
        summary_module._summary_publication_status(
            tmp_path,
            {
                "campaign_id": "final",
                "campaign_kind": "locked_final",
                "campaign_mode": "evaluation",
                "evidence_label": "FINAL_LOCKED",
            },
            validation_status="READY_FOR_TABLE2",
            draft_pilot=True,
        )


def test_validator_downgrades_unadjudicated_pilot_only_aggregate_to_draft(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    (aggregate / "metrics.json").write_text(
        json.dumps({"publication_status": "PILOT_ONLY"}),
        encoding="utf-8",
    )
    report = ValidationReport(campaign_dir=str(tmp_path))

    assert _publication_status(tmp_path, PILOT, report, True) == DRAFT_PILOT_STATUS


def test_validator_defaults_unadjudicated_pilot_without_aggregates_to_draft(
    tmp_path: Path,
) -> None:
    report = ValidationReport(campaign_dir=str(tmp_path))

    assert _publication_status(tmp_path, PILOT, report, False) == DRAFT_PILOT_STATUS


def test_duplicate_completion_publication_status_cannot_promote_pilot(
    tmp_path: Path,
) -> None:
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(PILOT), encoding="utf-8"
    )
    (tmp_path / "completion.json").write_text(
        (
            '{"schema_version":"table2.v1","campaign_id":"pilot",'
            '"status":"COMPLETE","scheduled_block_count":2,'
            '"processed_block_count":2,"included_block_count":2,'
            '"infrastructure_excluded_block_count":0,'
            '"publication_status":"READY_FOR_TABLE2",'
            '"publication_status":"DRAFT_PILOT_ONLY"}'
        ),
        encoding="utf-8",
    )

    assert (
        adjudication_gated_pilot_publication_status(tmp_path)
        == DRAFT_PILOT_STATUS
    )

    with pytest.raises(SchemaError, match="duplicate JSON key"):
        package_validator_module._validate_campaign_completion_publication_status(
            tmp_path,
            PILOT,
            scheduled_block_count=2,
            processed_block_count=2,
            included_block_count=2,
            infrastructure_excluded_block_count=0,
        )


def test_pilot_gate_promotes_only_after_terminal_completion_and_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_terminal_pilot_completion(tmp_path)
    calls: list[Path] = []

    def completed(root: Path) -> dict[str, str]:
        calls.append(root)
        return {"completion_status": "HUMAN_ADJUDICATION_COMPLETE"}

    monkeypatch.setattr(
        package_validator_module,
        "validate_manual_adjudication_completion",
        completed,
    )

    assert adjudication_gated_pilot_publication_status(tmp_path) == "PILOT_ONLY"
    assert calls == [tmp_path.resolve()]


def test_pilot_gate_keeps_valid_adjudication_draft_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        package_validator_module,
        "validate_manual_adjudication_completion",
        lambda _root: {"completion_status": "HUMAN_ADJUDICATION_COMPLETE"},
    )

    assert (
        adjudication_gated_pilot_publication_status(tmp_path)
        == DRAFT_PILOT_STATUS
    )


@pytest.mark.parametrize(
    ("maximum_blocks", "expected_completion", "expected_publication"),
    (
        (1, "INCOMPLETE", DRAFT_PILOT_STATUS),
        (None, "COMPLETE", "PILOT_ONLY"),
    ),
)
def test_campaign_completion_requires_terminal_campaign_and_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maximum_blocks: int | None,
    expected_completion: str,
    expected_publication: str,
) -> None:
    runner = object.__new__(campaign_module.CampaignRunner)
    runner.root = tmp_path
    runner.manifest = dict(PILOT)
    runner.schedule = [{"block_id": "one"}, {"block_id": "two"}]
    runner.maximum_blocks = maximum_blocks
    runner.live_readiness_probe_target = None
    runner.run_block = lambda _row: {"status": "INCLUDED"}
    monkeypatch.setattr(
        campaign_module,
        "require_live_compatibility_before_execution",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        campaign_module,
        "adjudication_gated_pilot_publication_status",
        lambda _root, *, campaign_complete: (
            "PILOT_ONLY" if campaign_complete else DRAFT_PILOT_STATUS
        ),
    )

    completion = runner.run()

    assert completion["status"] == expected_completion
    assert completion["publication_status"] == expected_publication


@pytest.mark.parametrize(
    "manifest",
    (
        {**PILOT, "campaign_kind": "unknown"},
        {**PILOT, "evidence_label": "FINAL_LOCKED"},
        {
            **PILOT,
            "campaign_kind": "locked_final",
            "evidence_label": "PILOT_ONLY",
        },
        {
            **PILOT,
            "campaign_kind": "locked_final",
            "campaign_mode": "smoke",
            "evidence_label": "FINAL_LOCKED",
        },
    ),
)
def test_publication_authority_rejects_unknown_or_inconsistent_profile(
    tmp_path: Path,
    manifest: dict[str, object],
) -> None:
    report = ValidationReport(campaign_dir=str(tmp_path))

    with pytest.raises(SchemaError, match="profile is not registered|cannot use smoke"):
        _publication_status(tmp_path, manifest, report, True)
    with pytest.raises(SchemaError, match="profile is not registered|cannot use smoke"):
        summary_module._summary_publication_status(
            tmp_path,
            manifest,
            validation_status="N/R",
            draft_pilot=True,
        )
