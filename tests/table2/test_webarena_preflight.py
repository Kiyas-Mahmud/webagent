from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_file, sha256_json
from web_agent.eval.table2.split_deployment_preflight import SINGLE_HOST_TOPOLOGY
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_PACKAGES,
    PINNED_WEBARENA_SERVICE_URL_KEYS,
    run_webarena_host_preflight,
    validate_webarena_host_preflight,
)
from web_agent.eval.table2.webarena_preflight_binding import (
    PREFLIGHT_ARTIFACT_RELATIVE_PATH,
    PREFLIGHT_BINDING_FIELD,
    PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH,
    build_deployment_preflight_binding,
    validate_bound_deployment_preflight,
)


URL_MAP = {
    "WA_GITLAB": "http://gitlab.private.example",
    "WA_MAP": "http://map.private.example",
    "WA_REDDIT": "http://reddit.private.example",
    "WA_SHOPPING": "http://shopping.private.example",
    "WA_SHOPPING_ADMIN": "http://admin.private.example",
    "WA_WIKIPEDIA": "http://wikipedia.private.example",
    "WA_HOMEPAGE": "http://homepage.private.example",
}


def _version(distribution: str) -> str:
    return PINNED_WEBARENA_PACKAGES[distribution]


def _pass_browser():
    return {
        "status": "PASS",
        "browser": "chromium",
        "browser_version": "fixture",
        "viewport": {"width": 1280, "height": 720},
        "device_scale_factor": 1,
        "screenshot_sha256": "a" * 64,
    }


def _pass_service(_url: str):
    return {"status": "PASS", "http_status": 200}


def _pass_reset(index: int):
    return {
        "status": "PASS",
        "task_index": index,
        "seed": 42,
        "goal_sha256": "b" * 64,
        "current_url_sha256": "c" * 64,
        "screenshot_shape": [720, 1280, 3],
        "observation_keys": ["goal", "screenshot", "url"],
        "action_taken": False,
        "reward_read": False,
        "evaluator_output_read": False,
    }


def test_all_measured_checks_are_required_for_single_dgx_compatibility() -> None:
    result = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
    )

    assert result["status"] == "PASS"
    assert result["campaign_eligible"] is True
    assert result["deployment_decision"] == "SINGLE_DGX_COMPATIBLE"
    assert all(result["checks"].values())
    assert {row["service_key"] for row in result["service_checks"]} == set(
        PINNED_WEBARENA_SERVICE_URL_KEYS
    )


def test_preflight_requires_all_seven_browsergym_service_origins() -> None:
    missing_wikipedia = dict(URL_MAP)
    missing_wikipedia.pop("WA_WIKIPEDIA")

    with pytest.raises(SchemaError, match="WA_WIKIPEDIA"):
        run_webarena_host_preflight(
            service_url_map=missing_wikipedia,
            version_getter=_version,
            browser_probe=_pass_browser,
            service_probe=_pass_service,
            live_reset_probe=_pass_reset,
        )


def test_five_task_placeholder_urls_cannot_replace_seven_service_origins() -> None:
    task_placeholder_map = {
        "__GITLAB__": URL_MAP["WA_GITLAB"],
        "__MAP__": URL_MAP["WA_MAP"],
        "__REDDIT__": URL_MAP["WA_REDDIT"],
        "__SHOPPING__": URL_MAP["WA_SHOPPING"],
        "__SHOPPING_ADMIN__": URL_MAP["WA_SHOPPING_ADMIN"],
    }

    with pytest.raises(SchemaError, match="WA_HOMEPAGE"):
        run_webarena_host_preflight(
            service_url_map=task_placeholder_map,
            version_getter=_version,
            browser_probe=_pass_browser,
            service_probe=_pass_service,
            live_reset_probe=_pass_reset,
        )


def test_browser_failure_requires_split_host_instead_of_partial_campaign() -> None:
    result = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=lambda: {"status": "FAIL", "error_type": "FixtureFailure"},
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
    )

    assert result["status"] == "FAIL"
    assert result["campaign_eligible"] is False
    assert result["deployment_decision"] == (
        "SPLIT_LOCAL_BROWSER_DGX_INFERENCE_REQUIRED"
    )


def test_skipping_live_reset_cannot_produce_compatibility_evidence() -> None:
    result = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        run_live_reset=False,
    )

    assert result["status"] == "INCOMPLETE"
    assert result["checks"]["live_reset"] is False
    assert result["campaign_eligible"] is False


def test_package_version_drift_fails_the_host_gate() -> None:
    def drifted(distribution: str) -> str:
        return "9.9.9" if distribution == "playwright" else _version(distribution)

    result = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=drifted,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["packages"] is False


def test_skipped_reset_does_not_hide_an_already_failed_infrastructure_check() -> None:
    result = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=lambda: {"status": "FAIL", "error_type": "FixtureFailure"},
        service_probe=_pass_service,
        run_live_reset=False,
    )

    assert result["status"] == "FAIL"
    assert result["deployment_decision"] == (
        "SPLIT_LOCAL_BROWSER_DGX_INFERENCE_REQUIRED"
    )


def test_report_hashes_service_origins_instead_of_copying_them() -> None:
    result = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
    )
    serialized = json.dumps(result, sort_keys=True)
    assert "private.example" not in serialized
    assert all(row["origin_sha256"] for row in result["service_checks"])


def _valid_evidence() -> dict:
    return run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
    )


def _bound_preflight(tmp_path: Path) -> tuple[dict, Path, Path]:
    evidence_path = tmp_path / PREFLIGHT_ARTIFACT_RELATIVE_PATH
    url_map_path = tmp_path / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH
    evidence_path.write_text(
        json.dumps(_valid_evidence(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    url_map_path.write_text(
        json.dumps(URL_MAP, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding = build_deployment_preflight_binding(
        evidence_path=evidence_path,
        service_url_map_path=url_map_path,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        expected_live_reset_task_index=0,
    )
    return {PREFLIGHT_BINDING_FIELD: binding}, evidence_path, url_map_path


def test_bound_preflight_reopens_exact_bytes_and_revalidates_semantics(
    tmp_path: Path,
) -> None:
    environment, evidence_path, _ = _bound_preflight(tmp_path)

    receipt = validate_bound_deployment_preflight(
        environment,
        artifact_root=tmp_path,
    )

    assert receipt.evidence_path == evidence_path
    assert receipt.binding["deployment_topology"] == SINGLE_HOST_TOPOLOGY
    assert receipt.evidence["status"] == "PASS"


def test_bound_preflight_rejects_missing_or_byte_tampered_evidence(
    tmp_path: Path,
) -> None:
    environment, evidence_path, _ = _bound_preflight(tmp_path)
    original = evidence_path.read_bytes()
    evidence_path.write_bytes(original + b" ")
    with pytest.raises(SchemaError, match="byte hash differs"):
        validate_bound_deployment_preflight(environment, artifact_root=tmp_path)

    evidence_path.unlink()
    with pytest.raises(SchemaError, match="is missing"):
        validate_bound_deployment_preflight(environment, artifact_root=tmp_path)


def test_bound_preflight_does_not_trust_rehashed_fail_status(tmp_path: Path) -> None:
    environment, evidence_path, _ = _bound_preflight(tmp_path)
    forged = _valid_evidence()
    forged["status"] = "FAIL"
    evidence_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding = environment[PREFLIGHT_BINDING_FIELD]
    binding["preflight_artifact_sha256"] = sha256_file(evidence_path)
    binding["preflight_content_sha256"] = sha256_json(forged)

    with pytest.raises(SchemaError, match="status is not campaign-eligible"):
        validate_bound_deployment_preflight(environment, artifact_root=tmp_path)


def test_bound_preflight_topology_discriminator_cannot_fall_through(
    tmp_path: Path,
) -> None:
    environment, _, _ = _bound_preflight(tmp_path)
    environment[PREFLIGHT_BINDING_FIELD]["deployment_topology"] = (
        "SPLIT_LOCAL_BROWSER_DGX_INFERENCE"
    )

    with pytest.raises(SchemaError, match="split-host binding lacks"):
        validate_bound_deployment_preflight(environment, artifact_root=tmp_path)


def test_stored_pass_evidence_is_rederived_against_the_url_map() -> None:
    evidence = _valid_evidence()

    validated = validate_webarena_host_preflight(
        evidence,
        service_url_map=URL_MAP,
        expected_live_reset_task_index=0,
    )

    assert validated == evidence
    assert validated is not evidence
    assert validated["browser_check"] is not evidence["browser_check"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(status="INCOMPLETE"), "status"),
        (
            lambda row: row["checks"].update(services=False),
            "check services",
        ),
        (
            lambda row: row["package_check"]["packages"][0].update(
                actual_version="9.9.9"
            ),
            "package",
        ),
        (
            lambda row: row["browser_check"].update(
                viewport={"width": 1920, "height": 1080}
            ),
            "viewport",
        ),
        (
            lambda row: row["service_checks"][0].update(http_status=500),
            "service check",
        ),
        (
            lambda row: row["service_checks"][0].update(origin_sha256="d" * 64),
            "service check",
        ),
        (
            lambda row: row["live_reset_check"].update(action_taken=True),
            "no-action_taken",
        ),
        (
            lambda row: row["live_reset_check"].update(reward_read=True),
            "no-reward_read",
        ),
        (
            lambda row: row["live_reset_check"].update(
                evaluator_output_read=True
            ),
            "no-evaluator_output_read",
        ),
        (
            lambda row: row["live_reset_check"].update(
                observation_keys=["goal", "url"]
            ),
            "observation keys",
        ),
        (lambda row: row.update(unregistered=True), "registered closure"),
    ],
)
def test_campaign_preflight_validation_rejects_tampering(mutation, message) -> None:
    evidence = deepcopy(_valid_evidence())
    mutation(evidence)

    with pytest.raises(SchemaError, match=message):
        validate_webarena_host_preflight(evidence, service_url_map=URL_MAP)


def test_campaign_preflight_rejects_a_different_url_map() -> None:
    evidence = _valid_evidence()
    changed = dict(URL_MAP)
    changed["WA_MAP"] = "http://other-map.private.example"

    with pytest.raises(SchemaError, match="service_url_map_sha256"):
        validate_webarena_host_preflight(evidence, service_url_map=changed)


def test_campaign_preflight_binds_the_requested_public_reset_task() -> None:
    evidence = _valid_evidence()

    with pytest.raises(SchemaError, match="requested task"):
        validate_webarena_host_preflight(
            evidence,
            service_url_map=URL_MAP,
            expected_live_reset_task_index=1,
        )


def test_preflight_uses_noncontiguous_tracked_registry_membership() -> None:
    indices = tuple(300 + ((position * 23) % 107) for position in range(50))
    evidence = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
        live_reset_task_index=indices[0],
        registered_task_indices=indices,
    )

    assert evidence["live_reset_check"]["task_index"] == indices[0]
    assert validate_webarena_host_preflight(
        evidence,
        service_url_map=URL_MAP,
        expected_live_reset_task_index=indices[0],
        registered_task_indices=indices,
    ) == evidence
    with pytest.raises(SchemaError, match="absent from the tracked public registry"):
        run_webarena_host_preflight(
            service_url_map=URL_MAP,
            version_getter=_version,
            browser_probe=_pass_browser,
            service_probe=_pass_service,
            live_reset_probe=_pass_reset,
            live_reset_task_index=9999,
            registered_task_indices=indices,
        )


def test_bound_preflight_accepts_registry_attested_noncontiguous_index(
    tmp_path: Path,
) -> None:
    indices = tuple(500 + ((position * 29) % 109) for position in range(50))
    evidence = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=_version,
        browser_probe=_pass_browser,
        service_probe=_pass_service,
        live_reset_probe=_pass_reset,
        live_reset_task_index=indices[0],
        registered_task_indices=indices,
    )
    evidence_path = tmp_path / PREFLIGHT_ARTIFACT_RELATIVE_PATH
    url_map_path = tmp_path / PREFLIGHT_SERVICE_URL_MAP_RELATIVE_PATH
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    url_map_path.write_text(json.dumps(URL_MAP), encoding="utf-8")
    binding = build_deployment_preflight_binding(
        evidence_path=evidence_path,
        service_url_map_path=url_map_path,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        expected_live_reset_task_index=indices[0],
    )

    validated = validate_bound_deployment_preflight(
        {PREFLIGHT_BINDING_FIELD: binding}, artifact_root=tmp_path
    )

    assert validated.binding["expected_live_reset_task_index"] == indices[0]


@pytest.mark.parametrize(
    "field", ["action_taken", "reward_read", "evaluator_output_read"]
)
def test_generator_does_not_issue_pass_for_forbidden_reset_claim(field: str) -> None:
    def invalid_reset(index: int) -> dict:
        row = _pass_reset(index)
        row[field] = True
        return row

    with pytest.raises(SchemaError, match=f"no-{field}"):
        run_webarena_host_preflight(
            service_url_map=URL_MAP,
            version_getter=_version,
            browser_probe=_pass_browser,
            service_probe=_pass_service,
            live_reset_probe=invalid_reset,
        )


def test_generator_does_not_issue_pass_for_status_only_service_probe() -> None:
    with pytest.raises(SchemaError, match="service check"):
        run_webarena_host_preflight(
            service_url_map=URL_MAP,
            version_getter=_version,
            browser_probe=_pass_browser,
            service_probe=lambda _url: {"status": "PASS", "http_status": 500},
            live_reset_probe=_pass_reset,
        )
