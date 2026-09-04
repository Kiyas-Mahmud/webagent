from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

import scripts.audit_table2_webarena_task_interface as task_interface_cli

from web_agent.eval.table2.common import (
    SchemaError,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2 import handoff as handoff_preparer
from web_agent.eval.table2.production_runner import _normal_task_specification
from web_agent.eval.table2.task_interface_audit import (
    PC01_BROWSER_ACTIONS,
    build_webarena_task_interface_audit,
    require_webarena_task_interface_compatible,
    validate_resolved_task_interface_binding,
    validate_webarena_task_interface_audit,
    write_webarena_task_interface_audit,
)
from web_agent.eval.table2.webarena_export import (
    PINNED_BROWSERGYM_WEBARENA_VERSION,
    PINNED_LIBWEBARENA_VERSION,
    PINNED_LIBWEBARENA_WHEEL_SHA256,
    PINNED_REQUIRED_URL_TOKENS,
    PINNED_TASK_DEFINITION_VERSION,
    PINNED_TASK_MEMBER,
    PINNED_TASK_SOURCE_SHA256,
    WEBARENA_RUNTIME_START_STATE_FIELDS,
    build_public_pilot_task_export,
    load_url_map,
    validate_public_pilot_task_export,
)


URL_MAP = {
    "__GITLAB__": "http://gitlab.example.test",
    "__MAP__": "http://map.example.test",
    "__REDDIT__": "http://reddit.example.test",
    "__SHOPPING__": "http://shopping.example.test",
    "__SHOPPING_ADMIN__": "http://admin.example.test",
}


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _registry(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "manifest_id": "fixture-public-dev-50",
                "benchmark": "webarena",
                "partition": "development",
                "registration_status": "FROZEN_DEVELOPMENT_EXCLUSION",
                "required_task_count": 50,
                "final_paper_evaluation_eligible": False,
                "locked_test_content": False,
                "tasks": [
                    {"task_id": f"webarena.{index}", "upstream_index": index}
                    for index in range(50)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _upstream_rows() -> list[dict]:
    tokens = sorted(URL_MAP)
    rows = []
    for index in range(50):
        token = tokens[index % len(tokens)]
        rows.append(
            {
                "sites": [token.strip("_").casefold()],
                "task_id": index,
                "require_login": True,
                "storage_state": f"./.auth/site_{index}_state.json",
                "start_url": f"{token}/start/{index}",
                "geolocation": None,
                "intent": f"Complete public development task {index}",
                "require_reset": False,
                "eval": {
                    "eval_types": ["string_match"],
                    "reference_answers": {"exact_match": f"answer-{index}"},
                    "reference_url": token,
                },
            }
        )
    return rows


def _source(path: Path) -> tuple[Path, str]:
    payload = json.dumps(_upstream_rows(), sort_keys=True).encode()
    path.write_bytes(payload)
    return path, sha256_bytes(payload)


def _build(tmp_path: Path, **overrides):
    source, digest = _source(tmp_path / "test.raw.json")
    kwargs = {
        "source": source,
        "registry_path": _registry(tmp_path / "registry.json"),
        "site_url_map": URL_MAP,
        "snapshot_id": "webarena-public-0-49-fixture",
        "benchmark_version": "0.14.3",
        "task_definition_version": "fixture-task-v1",
        "evaluator_id": "official-webarena",
        "evaluator_version": "fixture-v1",
        "expected_source_sha256": digest,
    }
    kwargs.update(overrides)
    if Path(kwargs["source"]).suffix.casefold() not in {".whl", ".zip"}:
        kwargs.setdefault(
            "authorized_raw_json_sha256", kwargs["expected_source_sha256"]
        )
    return build_public_pilot_task_export(**kwargs)


def _build_interface_fixture(
    tmp_path: Path,
) -> tuple[dict, Path, Path, Path, str]:
    rows = _upstream_rows()
    rows[8]["eval"]["reference_answers"] = {
        "fuzzy_match": ["a semantically matched answer"]
    }
    for index in (44, 45, 46):
        token = sorted(URL_MAP)[index % len(URL_MAP)]
        rows[index]["eval"] = {
            "eval_types": ["url_match"],
            "reference_url": token,
        }
    payload = json.dumps(rows, sort_keys=True).encode()
    source = tmp_path / "interface-test.raw.json"
    source.write_bytes(payload)
    digest = sha256_bytes(payload)
    registry = _registry(tmp_path / "interface-registry.json")
    url_map_path = tmp_path / "interface-url-map.json"
    url_map_path.write_text(json.dumps(URL_MAP), encoding="utf-8")
    export = build_public_pilot_task_export(
        source=source,
        registry_path=registry,
        site_url_map=URL_MAP,
        snapshot_id="webarena-public-0-49-interface-fixture",
        benchmark_version=PINNED_BROWSERGYM_WEBARENA_VERSION,
        task_definition_version=PINNED_TASK_DEFINITION_VERSION,
        evaluator_id="official-webarena",
        evaluator_version="libwebarena-fixture",
        expected_source_sha256=digest,
        authorized_raw_json_sha256=digest,
    )
    return export, source, registry, url_map_path, digest


def test_export_is_exactly_registered_public_0_to_49_and_pilot_only(
    tmp_path: Path,
) -> None:
    value = _build(tmp_path)

    assert value["evidence_label"] == "PILOT_ONLY"
    assert value["paper_table_status"] == "N/R"
    assert value["locked_test_content"] is False
    assert value["final_paper_evaluation_eligible"] is False
    assert [row["upstream_index"] for row in value["tasks"]] == list(range(50))
    assert len(value["tasks"]) == 50
    assert not any("__" in row["start_state"]["start_url"] for row in value["tasks"])
    assert all(
        set(row["start_state"]) == WEBARENA_RUNTIME_START_STATE_FIELDS
        for row in value["tasks"]
    )


def test_handoff_validation_rebuilds_export_from_upstream_bytes(tmp_path: Path) -> None:
    source, digest = _source(tmp_path / "test.raw.json")
    registry = _registry(tmp_path / "registry.json")
    value = build_public_pilot_task_export(
        source=source,
        registry_path=registry,
        site_url_map=URL_MAP,
        snapshot_id="webarena-public-0-49-authenticated-fixture",
        benchmark_version=PINNED_BROWSERGYM_WEBARENA_VERSION,
        task_definition_version=PINNED_TASK_DEFINITION_VERSION,
        evaluator_id="official-webarena",
        evaluator_version="fixture-v1",
        expected_source_sha256=digest,
        authorized_raw_json_sha256=digest,
    )

    assert validate_public_pilot_task_export(
        value,
        source=source,
        registry_path=registry,
        site_url_map=URL_MAP,
        expected_source_sha256=digest,
        authorized_raw_json_sha256=digest,
    ) == value

    tampered = json.loads(json.dumps(value))
    tampered["tasks"][0]["instruction"] = "altered after export"
    tampered["resolved_task_set_sha256"] = "0" * 64
    with pytest.raises(SchemaError, match="independent rebuild"):
        validate_public_pilot_task_export(
            tampered,
            source=source,
            registry_path=registry,
            site_url_map=URL_MAP,
            expected_source_sha256=digest,
            authorized_raw_json_sha256=digest,
        )

    minimal = {
        "snapshot_id": value["snapshot_id"],
        "benchmark_version": value["benchmark_version"],
        "task_definition_version": value["task_definition_version"],
        "tasks": value["tasks"],
    }
    with pytest.raises(SchemaError, match="independent rebuild"):
        validate_public_pilot_task_export(
            minimal,
            source=source,
            registry_path=registry,
            site_url_map=URL_MAP,
            expected_source_sha256=digest,
            authorized_raw_json_sha256=digest,
        )


def test_task_interface_audit_reports_47_answer_tasks_and_3_page_state_tasks(
    tmp_path: Path,
) -> None:
    export, _, _, _, _ = _build_interface_fixture(tmp_path)

    audit = build_webarena_task_interface_audit(export)

    assert audit["status"] == "FAIL"
    assert audit["handoff_eligible"] is False
    assert audit["task_count"] == 50
    assert audit["compatible_task_count"] == 3
    assert audit["incompatible_task_count"] == 47
    assert audit["assistant_answer_required_task_count"] == 47
    assert audit["page_state_only_task_count"] == 3
    assert audit["fuzzy_string_match_task_count"] == 1
    assert audit["evaluator_type_counts"] == {
        "string_match": 47,
        "url_match": 3,
    }
    assert audit["action_interface"]["action_classes"] == list(
        PC01_BROWSER_ACTIONS
    )
    assert audit["action_interface"]["stop_action_supported"] is False
    assert audit["action_interface"][
        "assistant_answer_submission_supported"
    ] is False
    assert audit["task_export_content_sha256"] == sha256_json(export)
    assert audit["resolved_task_set_sha256"] == export[
        "resolved_task_set_sha256"
    ]
    assert audit["tasks"][8]["answer_matching_modes"] == ["fuzzy_match"]
    assert audit["tasks"][8]["evaluator_mode"] == (
        "ASSISTANT_ANSWER_REQUIRED"
    )
    assert audit["tasks"][44]["compatible"] is True
    assert audit["tasks"][44]["evaluator_mode"] == "PAGE_STATE_ONLY"
    with pytest.raises(SchemaError, match="rejects evaluation handoff"):
        require_webarena_task_interface_compatible(audit)


def test_task_interface_audit_is_exactly_recomputed_and_writer_is_no_overwrite(
    tmp_path: Path,
) -> None:
    export, _, _, _, _ = _build_interface_fixture(tmp_path)
    audit = build_webarena_task_interface_audit(export)
    assert validate_webarena_task_interface_audit(
        audit, task_export=export
    ) == audit

    tampered = json.loads(json.dumps(audit))
    tampered["compatible_task_count"] = 50
    with pytest.raises(SchemaError, match="exact recomputation"):
        validate_webarena_task_interface_audit(tampered, task_export=export)
    minimal = {
        "status": "PASS",
        "handoff_eligible": True,
        "task_export_content_sha256": sha256_json(export),
    }
    with pytest.raises(SchemaError, match="exact recomputation"):
        validate_webarena_task_interface_audit(minimal, task_export=export)

    output = tmp_path / "interface-audit.json"
    write_webarena_task_interface_audit(output, task_export=export)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_webarena_task_interface_audit(output, task_export=export)


def test_resolved_snapshot_recomputes_and_cannot_forge_interface_pass(
    tmp_path: Path,
) -> None:
    export, _, _, _, _ = _build_interface_fixture(tmp_path)
    audit = build_webarena_task_interface_audit(export)
    snapshot = {
        "snapshot_id": export["snapshot_id"],
        "upstream_export_schema_version": export["schema_version"],
        "upstream_export_record_type": export["record_type"],
        "upstream_export_content_sha256": sha256_json(export),
        "upstream_task_source": export["source"],
        "site_url_map_sha256": export["site_url_map_sha256"],
        "resolved_task_set_sha256": export["resolved_task_set_sha256"],
        "task_action_interface_audit": audit,
        "task_action_interface_audit_content_sha256": sha256_json(audit),
        "tasks": export["tasks"],
    }
    assert validate_resolved_task_interface_binding(snapshot) == audit

    forged = json.loads(json.dumps(snapshot))
    forged["task_action_interface_audit"]["status"] = "PASS"
    forged["task_action_interface_audit"]["handoff_eligible"] = True
    forged["task_action_interface_audit_content_sha256"] = sha256_json(
        forged["task_action_interface_audit"]
    )
    with pytest.raises(SchemaError, match="task-row recomputation"):
        validate_resolved_task_interface_binding(forged)

    changed_task = json.loads(json.dumps(snapshot))
    changed_task["tasks"][0]["evaluator"]["config"] = {
        "eval_types": ["url_match"],
        "reference_url": "http://gitlab.example.test",
    }
    changed_task["tasks"][0]["task_config"]["eval"] = dict(
        changed_task["tasks"][0]["evaluator"]["config"]
    )
    with pytest.raises(SchemaError, match="task-row recomputation"):
        validate_resolved_task_interface_binding(changed_task)


def test_task_interface_audit_cli_writes_fail_evidence_without_mutating_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export, _, _, _, _ = _build_interface_fixture(tmp_path)
    export_path = tmp_path / "resolved-export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    output = tmp_path / "task-interface-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_table2_webarena_task_interface.py",
            "--task-export",
            str(export_path),
            "--output",
            str(output),
        ],
    )

    task_interface_cli.main()

    receipt = json.loads(capsys.readouterr().out)
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["handoff_eligible"] is False
    assert receipt["incompatible_task_count"] == 47
    assert [row["upstream_index"] for row in audit["tasks"]] == list(
        range(50)
    )


def test_handoff_task_builder_rejects_incompatible_audit_without_task_dropping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export, source, registry, url_map_path, digest = _build_interface_fixture(
        tmp_path
    )
    export_path = tmp_path / "interface-export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    audit_path = tmp_path / "interface-audit.json"
    audit_path.write_text(
        json.dumps(build_webarena_task_interface_audit(export)),
        encoding="utf-8",
    )
    monkeypatch.setattr(handoff_preparer, "PINNED_TASK_SOURCE_SHA256", digest)

    with pytest.raises(
        SchemaError,
        match="47 incompatible task.*47 task.*assistant-answer submission",
    ):
        handoff_preparer._build_resolved_tasks(
            export_path=export_path,
            upstream_task_source_path=source,
            site_url_map_path=url_map_path,
            authorized_raw_task_source_sha256=digest,
            task_interface_audit_path=audit_path,
            output_path=tmp_path / "handoff/resolved_tasks.json",
            registry_path=registry,
            environment_path=tmp_path / "must-not-be-reached.json",
            duplicate_output_path=tmp_path / "handoff/duplicate_audit.json",
        )
    assert not (tmp_path / "handoff/resolved_tasks.json").exists()


def test_tracked_source_authority_matches_the_executable_export_contract() -> None:
    authority = json.loads(
        (
            REPOSITORY_ROOT
            / "benchmarks/table2/pilot/webarena_source_authority.json"
        ).read_text(encoding="utf-8")
    )
    assert authority["task_indices"] == {"first": 0, "last": 49, "count": 50}
    assert authority["browsergym_webarena"]["version"] == (
        PINNED_BROWSERGYM_WEBARENA_VERSION
    )
    assert authority["libwebarena"]["version"] == PINNED_LIBWEBARENA_VERSION
    assert authority["libwebarena"]["task_member"] == PINNED_TASK_MEMBER
    assert authority["libwebarena"]["task_member_sha256"] == (
        PINNED_TASK_SOURCE_SHA256
    )
    assert authority["libwebarena"]["wheel_sha256"] == (
        PINNED_LIBWEBARENA_WHEEL_SHA256
    )
    assert frozenset(authority["required_site_url_tokens"]) == (
        PINNED_REQUIRED_URL_TOKENS
    )
    assert authority["final_paper_evaluation_eligible"] is False


def test_sealed_evaluator_truth_is_not_in_oracle_blind_runtime_projection(
    tmp_path: Path,
) -> None:
    task = _build(tmp_path)["tasks"][0]
    assert task["evaluator"]["config"]["reference_answers"]
    assert task["task_config"]["eval"]["reference_answers"]

    specification = _normal_task_specification(
        task, {"benchmark": "webarena", "benchmark_version": "0.14.3"}
    )
    payload = specification.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    assert "reference_answers" not in serialized
    assert "answer-0" not in serialized
    assert "evaluator" not in serialized


def test_source_hash_mismatch_fails_before_export(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="registered SHA-256"):
        _build(tmp_path, expected_source_sha256="0" * 64)


def test_url_map_requires_exact_credential_free_registered_tokens(
    tmp_path: Path,
) -> None:
    missing = dict(URL_MAP)
    missing.pop("__MAP__")
    with pytest.raises(SchemaError, match="exactly cover"):
        _build(tmp_path, site_url_map=missing)

    credentialed = dict(URL_MAP)
    credentialed["__MAP__"] = "https://username:password@map.example.test"
    with pytest.raises(SchemaError, match="credential-free"):
        _build(tmp_path, site_url_map=credentialed)

    wrapped = tmp_path / "wrapped-url-map.json"
    wrapped.write_text(
        json.dumps({"url_map": URL_MAP, "credentials": "must-not-be-accepted"}),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="only the url_map field"):
        load_url_map(wrapped)


def test_every_upstream_reset_field_is_required_before_export(tmp_path: Path) -> None:
    rows = _upstream_rows()
    del rows[0]["geolocation"]
    payload = json.dumps(rows, sort_keys=True).encode()
    source = tmp_path / "missing-reset-field.json"
    source.write_bytes(payload)

    with pytest.raises(SchemaError, match="lacks WebArena reset fields"):
        _build(
            tmp_path,
            source=source,
            expected_source_sha256=sha256_bytes(payload),
        )


def test_changed_task_placeholder_set_fails_closed(tmp_path: Path) -> None:
    rows = _upstream_rows()
    rows[0]["start_url"] = "__UNREGISTERED_SITE__/start"
    payload = json.dumps(rows, sort_keys=True).encode()
    source = tmp_path / "changed.json"
    source.write_bytes(payload)
    with pytest.raises(SchemaError, match="placeholder set changed"):
        _build(
            tmp_path,
            source=source,
            expected_source_sha256=sha256_bytes(payload),
        )


def test_wheel_member_is_hashed_as_task_source_not_as_container(tmp_path: Path) -> None:
    payload = json.dumps(_upstream_rows(), sort_keys=True).encode()
    wheel = tmp_path / "libwebarena-fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("webarena/test.raw.json", payload)

    value = _build(
        tmp_path,
        source=wheel,
        expected_source_sha256=sha256_bytes(payload),
        expected_container_sha256=sha256_file(wheel),
    )
    assert value["source"]["container_kind"] == "zip_archive"
    assert value["source"]["container_sha256"]
    assert value["source"]["task_source_sha256"] == sha256_bytes(payload)

    repacked = tmp_path / "repacked-libwebarena-fixture.whl"
    with zipfile.ZipFile(repacked, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("webarena/test.raw.json", payload)
    assert sha256_file(repacked) != sha256_file(wheel)
    with pytest.raises(SchemaError, match="wheel container differs"):
        _build(
            tmp_path,
            source=repacked,
            expected_source_sha256=sha256_bytes(payload),
            expected_container_sha256=sha256_file(wheel),
        )


def test_raw_source_preserves_container_and_payload_identity(tmp_path: Path) -> None:
    value = _build(tmp_path)
    assert value["source"]["container_kind"] == "raw_json"
    assert value["source"]["container_sha256"] == value["source"][
        "task_source_sha256"
    ]
    source, digest = _source(tmp_path / "unauthorized-raw.json")
    with pytest.raises(SchemaError, match="separately authorized"):
        build_public_pilot_task_export(
            source=source,
            registry_path=_registry(tmp_path / "unauthorized-registry.json"),
            site_url_map=URL_MAP,
            snapshot_id="unauthorized-raw",
            benchmark_version=PINNED_BROWSERGYM_WEBARENA_VERSION,
            task_definition_version=PINNED_TASK_DEFINITION_VERSION,
            evaluator_id="official-webarena",
            evaluator_version="fixture-v1",
            expected_source_sha256=digest,
        )


def test_duplicate_or_missing_upstream_identity_fails_closed(tmp_path: Path) -> None:
    rows = _upstream_rows()
    rows[49]["task_id"] = 48
    payload = json.dumps(rows, sort_keys=True).encode()
    source = tmp_path / "duplicate.json"
    source.write_bytes(payload)
    with pytest.raises(SchemaError, match="repeats task_id"):
        _build(
            tmp_path,
            source=source,
            expected_source_sha256=sha256_bytes(payload),
        )
