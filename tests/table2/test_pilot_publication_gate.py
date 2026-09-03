from __future__ import annotations

from pathlib import Path
import json

import pytest

from web_agent.eval.table2.common import Table2Error
from web_agent.eval.table2.package_validator import (
    DRAFT_PILOT_STATUS,
    ValidationReport,
    _publication_status,
)
from web_agent.eval.table2 import summary as summary_module


PILOT = {
    "campaign_id": "pilot",
    "campaign_kind": "engineering_pilot",
    "evidence_label": "PILOT_ONLY",
}


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
    calls: list[Path] = []

    def completed(root: Path) -> dict:
        calls.append(root)
        return {"completion_status": "HUMAN_ADJUDICATION_COMPLETE"}

    monkeypatch.setattr(
        summary_module,
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
