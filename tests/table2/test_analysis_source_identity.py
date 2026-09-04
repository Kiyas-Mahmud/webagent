from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.eval.table2 import execution_guard
from web_agent.eval.table2.common import SchemaError, sha256_file, sha256_json
from web_agent.eval.table2.execution_guard import (
    ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS,
    ENGINEERING_SMOKE_SCOPE,
    EVALUATION_RUNNER_SCOPE,
    validate_analysis_source_identity,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    campaign = tmp_path / "campaign"
    commit = "a" * 40
    rows: list[dict[str, str]] = []
    for index, relative in enumerate(ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS):
        live = repo / relative
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(f"analysis-source-{index}\n", encoding="utf-8")
        frozen = campaign / "frozen/runner_source" / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(live.read_bytes())
        rows.append({"relative_path": relative, "sha256": sha256_file(live)})
    attestation = _write_json(
        campaign / "frozen/runner_attestation.json",
        {"source_files": rows, "source_set_sha256": sha256_json(rows)},
    )
    _write_json(
        campaign / "campaign_manifest.json",
        {
            "campaign_mode": "evaluation",
            "runner_identity_scope": EVALUATION_RUNNER_SCOPE,
            "repository_commit": commit,
            "runner_attestation_sha256": sha256_file(attestation),
        },
    )
    return repo, campaign, commit


def test_smoke_analysis_identity_is_explicitly_non_paper(tmp_path: Path) -> None:
    campaign = tmp_path / "smoke"
    _write_json(
        campaign / "campaign_manifest.json",
        {"campaign_mode": "smoke"},
    )
    value = validate_analysis_source_identity(campaign)
    assert value["identity_scope"] == ENGINEERING_SMOKE_SCOPE
    assert value["paper_table_eligible"] is False
    assert value["sources"] == []


def test_evaluation_analysis_replays_live_and_frozen_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, campaign, commit = _evaluation_fixture(tmp_path)
    monkeypatch.setattr(
        execution_guard,
        "assert_clean_git_checkout",
        lambda supplied: commit if supplied == repo.resolve() else "wrong",
    )
    value = validate_analysis_source_identity(campaign, repository_root=repo)
    assert value["paper_table_eligible"] is True
    assert value["repository_commit"] == commit
    assert len(value["sources"]) == len(ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS)

    relative = ANALYSIS_REQUIRED_SOURCE_RELATIVE_PATHS[0]
    (campaign / "frozen/runner_source" / relative).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(SchemaError, match="not live/frozen attested"):
        validate_analysis_source_identity(campaign, repository_root=repo)


def test_evaluation_analysis_rejects_commit_or_source_omission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, campaign, commit = _evaluation_fixture(tmp_path)
    monkeypatch.setattr(
        execution_guard,
        "assert_clean_git_checkout",
        lambda supplied: "b" * 40,
    )
    with pytest.raises(SchemaError, match="checkout commit differs"):
        validate_analysis_source_identity(campaign, repository_root=repo)

    monkeypatch.setattr(
        execution_guard,
        "assert_clean_git_checkout",
        lambda supplied: commit,
    )
    attestation_path = campaign / "frozen/runner_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["source_files"] = attestation["source_files"][1:]
    attestation["source_set_sha256"] = sha256_json(attestation["source_files"])
    _write_json(attestation_path, attestation)
    manifest_path = campaign / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runner_attestation_sha256"] = sha256_file(attestation_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(SchemaError, match="not live/frozen attested"):
        validate_analysis_source_identity(campaign, repository_root=repo)
