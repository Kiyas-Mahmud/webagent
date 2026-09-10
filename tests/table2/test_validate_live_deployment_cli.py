from __future__ import annotations

import json
from pathlib import Path

import pytest

import web_agent.eval.table2.live_deployment as live_deployment
from web_agent.eval.table2.common import SchemaError, sha256_file
from web_agent.eval.table2 import live_deployment_validation as live_cli
from tests.table2.test_live_deployment import _manifest


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SUMMARY_FIELDS = {
    "schema_version",
    "scientific_role",
    "paper_table_status",
    "dispatch_authorized",
    "handoff_authorized",
    "cross_binding_performed",
    "manifest_sha256",
    "manifest_content_sha256",
    "package_file_count",
    "package_file_set_sha256",
    "capability_source_file_count",
    "capability_source_set_sha256",
    "capability_id_count",
    "capability_id_set_sha256",
    "evaluator_requirements_content_sha256",
    "evaluator_requirement_task_count",
    "judge_required_task_count",
    "string_match_task_count",
}


@pytest.fixture(autouse=True)
def _isolate_synthetic_review_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live_deployment,
        "PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S",
        frozenset(),
    )


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_live_deployment_cli_summary_replays_strict_ephemeral_staging(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "measured-evidence"
    manifest = _write_json(evidence / "deployment.json", _manifest(evidence))
    before = {
        path.relative_to(evidence).as_posix(): sha256_file(path)
        for path in evidence.rglob("*")
        if path.is_file()
    }

    summary = live_cli.build_validation_summary(
        repository_root=ROOT,
        manifest_path=manifest,
        evidence_root=evidence,
    )

    assert summary["schema_version"] == (
        "table2-live-deployment-validation-summary-v1"
    )
    assert summary["manifest_sha256"] == sha256_file(manifest)
    assert summary["capability_id_count"] == 7
    assert summary["evaluator_requirement_task_count"] == 50
    assert summary["scientific_role"] == (
        "VALIDATION_SUMMARY_ONLY_NOT_DISPATCH_AUTHORITY"
    )
    assert set(summary) == EXPECTED_SUMMARY_FIELDS
    assert summary["paper_table_status"] == "N/R"
    assert summary["dispatch_authorized"] is False
    assert summary["handoff_authorized"] is False
    assert summary["cross_binding_performed"] is False
    assert "status" not in summary
    assert "manifest_path" not in summary
    assert before == {
        path.relative_to(evidence).as_posix(): sha256_file(path)
        for path in evidence.rglob("*")
        if path.is_file()
    }


def test_live_deployment_cli_stdout_is_exact_non_authorizing_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = tmp_path / "measured-evidence"
    manifest = _write_json(evidence / "deployment.json", _manifest(evidence))

    result = live_cli.main(
        [
            "--repository-root",
            str(ROOT),
            "--manifest",
            str(manifest),
            "--evidence-root",
            str(evidence),
        ]
    )

    assert result == 0
    stdout = capsys.readouterr().out
    assert stdout.endswith("\n")
    assert stdout.count("\n{") == 0
    summary = json.loads(stdout)
    assert set(summary) == EXPECTED_SUMMARY_FIELDS
    assert summary["paper_table_status"] == "N/R"
    assert summary["dispatch_authorized"] is False
    assert summary["handoff_authorized"] is False
    assert summary["cross_binding_performed"] is False


@pytest.mark.parametrize("relative_field", ["manifest_path", "evidence_root"])
def test_live_deployment_cli_requires_absolute_authority_paths(
    tmp_path: Path,
    relative_field: str,
) -> None:
    evidence = tmp_path / "measured-evidence"
    manifest = _write_json(evidence / "deployment.json", _manifest(evidence))
    arguments = {
        "repository_root": ROOT,
        "manifest_path": manifest,
        "evidence_root": evidence,
    }
    arguments[relative_field] = Path("relative-authority-path")

    with pytest.raises(SchemaError, match="must be an absolute external path"):
        live_cli.build_validation_summary(**arguments)


@pytest.mark.parametrize("repo_local_field", ["manifest_path", "evidence_root"])
def test_live_deployment_cli_rejects_repository_local_authority_paths(
    tmp_path: Path,
    repo_local_field: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    manifest = _write_json(external / "deployment.json", {})
    evidence = external
    local_evidence = repository / "evidence"
    local_evidence.mkdir()
    local_manifest = _write_json(repository / "deployment.json", {})
    arguments = {
        "repository_root": repository,
        "manifest_path": manifest,
        "evidence_root": evidence,
    }
    arguments[repo_local_field] = (
        local_manifest if repo_local_field == "manifest_path" else local_evidence
    )

    with pytest.raises(SchemaError, match="must be outside the repository tree"):
        live_cli.build_validation_summary(**arguments)


def test_live_deployment_cli_rejects_temporary_root_inside_evidence_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "measured-evidence"
    manifest = _write_json(evidence / "deployment.json", _manifest(evidence))
    temporary_parent = evidence / "operator-configured-tmp"
    temporary_parent.mkdir()
    before = sorted(path.relative_to(evidence) for path in evidence.rglob("*"))
    monkeypatch.setattr(live_cli, "gettempdir", lambda: str(temporary_parent))

    with pytest.raises(
        SchemaError,
        match="temporary directory must remain outside the evidence root",
    ):
        live_cli.build_validation_summary(
            repository_root=ROOT,
            manifest_path=manifest,
            evidence_root=evidence,
        )

    assert sorted(path.relative_to(evidence) for path in evidence.rglob("*")) == before


def test_live_deployment_cli_rejects_changed_measured_evidence(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "measured-evidence"
    value = _manifest(evidence)
    manifest = _write_json(evidence / "deployment.json", value)
    readiness = evidence / value["capabilities"]["deterministic_reset"][
        "readiness_evidence_path"
    ]
    readiness.write_text("{}", encoding="utf-8")

    with pytest.raises(SchemaError, match="readiness evidence hash differs"):
        live_cli.build_validation_summary(
            repository_root=ROOT,
            manifest_path=manifest,
            evidence_root=evidence,
        )
