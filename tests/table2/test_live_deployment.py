from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import web_agent.eval.table2.live_deployment as live_deployment
from web_agent.eval.table2.common import SchemaError, sha256_file, sha256_json
from web_agent.eval.table2.task_interface_audit import (
    build_webarena_task_interface_audit,
)
from web_agent.eval.table2.live_deployment import (
    COMPATIBILITY_REVIEW_SCHEMA_VERSION,
    EVALUATOR_REQUIREMENTS_SCHEMA_VERSION,
    JUDGE_API_NOT_APPLICABLE,
    LIVE_CAPABILITY_EVIDENCE_RECORD_TYPE,
    LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
    LIVE_DEPLOYMENT_RECORD_TYPE,
    LIVE_DEPLOYMENT_SCHEMA_VERSION,
    PINNED_LIBWEBARENA_VERSION,
    PINNED_LIBWEBARENA_WHEEL_SHA256,
    PINNED_UPSTREAM_EVALUATORS_SHA256,
    PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
    PINNED_UPSTREAM_WEBARENA_REPOSITORY,
    PINNED_UPSTREAM_WEBARENA_REVISION,
    PINNED_WEBARENA_JUDGE_DECODING,
    PINNED_WEBARENA_JUDGE_MODEL_ID,
    REGISTERED_BROWSERGYM_VERSION,
    REGISTERED_SEALED_BROKER_CONTRACT,
    REGISTERED_START_STATE_CONTRACT,
    REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
    REQUIRED_CAPABILITY_CLAIMS,
    REQUIRED_LIVE_CAPABILITIES,
    UPSTREAM_WEBARENA_START_STATE_FIELDS,
    LIVE_DEPLOYMENT_BINDING_FIELD,
    build_evaluator_requirements_artifact,
    stage_pc01_live_deployment_package,
    validate_evaluator_requirements_artifact,
    validate_evaluator_requirements_derivation,
    validate_evaluator_requirements_resolved_snapshot_binding,
    validate_bound_pc01_live_deployment,
    validate_live_capability_source_plane_disjointness,
    validate_pc01_live_deployment_manifest,
)
from web_agent.eval.table2.handoff import (
    _validate_handoff_live_capability_source_planes,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE = Path(__file__).resolve().relative_to(ROOT).as_posix()
SEALED_SOURCE_RELATIVE = "src/web_agent/eval/table2/sealed_verifier.py"
BROKER_SOURCE_RELATIVE = "src/web_agent/eval/table2/live_deployment.py"


@pytest.fixture(autouse=True)
def _isolated_synthetic_external_review_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give each test an isolated stand-in for a pre-pinned review allowlist.

    Production intentionally has no registered external-review binding yet.
    Existing manifest tests exercise downstream validation with synthetic
    evidence, so their helper explicitly pins only the exact bytes it creates.
    """

    monkeypatch.setattr(
        live_deployment,
        "PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S",
        set(),
    )


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_bytes(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _task_export_and_audit(*, page_state_only: bool) -> tuple[dict, dict]:
    tasks = []
    for index in range(50):
        if page_state_only or index >= 47:
            evaluator_config = {
                "eval_types": ["url_match"],
                "reference_answers": None,
                "reference_url": f"https://site.example/done/{index}",
                "program_html": [],
                "url_note": "GOLD in PRED",
            }
        else:
            answer_mode = "fuzzy_match" if index < 13 else "exact_match"
            evaluator_config = {
                "eval_types": ["string_match"],
                "reference_answers": {answer_mode: f"answer-{index}"},
            }
        task_config = {
            "task_id": index,
            "sites": ["shopping"],
            "start_url": f"https://site.example/start/{index}",
            "require_login": True,
            "storage_state": f"./.auth/site-{index}.json",
            "geolocation": None,
            "require_reset": False,
            "intent": f"Synthetic task {index}",
            "eval": evaluator_config,
        }
        tasks.append(
            {
                "task_id": f"webarena.{index}",
                "upstream_index": index,
                "benchmark_task_id": str(index),
                "benchmark_task_version": "synthetic-v1",
                "instruction": task_config["intent"],
                "start_state": {
                    key: task_config[key]
                    for key in (
                        "sites",
                        "start_url",
                        "require_login",
                        "storage_state",
                        "geolocation",
                        "require_reset",
                    )
                },
                "task_config": task_config,
                "evaluator": {
                    "evaluator_id": "synthetic-sealed-webarena",
                    "evaluator_version": "synthetic-v1",
                    "config": evaluator_config,
                },
            }
        )
    task_export = {
        "schema_version": "table2-webarena-public-task-export-v1",
        "record_type": "SealedWebArenaDevelopmentTaskExport",
        "snapshot_id": "synthetic-derived-evaluator-profile",
        "benchmark": "webarena",
        "benchmark_version": REGISTERED_BROWSERGYM_VERSION,
        "task_definition_version": "synthetic-v1",
        "source": {"task_source_sha256": "7" * 64},
        "site_url_map_sha256": "6" * 64,
        "required_task_count": 50,
        "registry_manifest_id": "synthetic-active-public-development-50",
        "registry_manifest_sha256": "5" * 64,
        "tasks": tasks,
    }
    task_export["resolved_task_set_sha256"] = sha256_json(tasks)
    audit = build_webarena_task_interface_audit(task_export)
    return task_export, audit


def _resolved_snapshot(task_export: dict, task_audit: dict) -> dict:
    compile_report = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=task_audit,
    )["page_state_compile_report"]
    return {
        "snapshot_id": task_export["snapshot_id"],
        "upstream_export_schema_version": task_export["schema_version"],
        "upstream_export_record_type": task_export["record_type"],
        "upstream_export_content_sha256": sha256_json(task_export),
        "upstream_task_source": task_export["source"],
        "site_url_map_sha256": task_export["site_url_map_sha256"],
        "resolved_task_set_sha256": task_export["resolved_task_set_sha256"],
        "registry_manifest_id": task_export["registry_manifest_id"],
        "upstream_registry_manifest_sha256": task_export[
            "registry_manifest_sha256"
        ],
        "task_action_interface_audit": task_audit,
        "task_action_interface_audit_content_sha256": sha256_json(task_audit),
        "page_state_evaluator_compile_report": compile_report,
        "page_state_evaluator_compile_report_content_sha256": (
            sha256_json(compile_report) if compile_report is not None else None
        ),
        "tasks": task_export["tasks"],
    }


def test_evaluator_requirements_bind_to_exact_resolved_task_snapshot() -> None:
    task_export, task_audit = _task_export_and_audit(page_state_only=True)
    requirements = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=task_audit,
    )
    snapshot = _resolved_snapshot(task_export, task_audit)

    assert validate_evaluator_requirements_resolved_snapshot_binding(
        requirements,
        task_export=task_export,
        resolved_task_snapshot=snapshot,
    ) == requirements

    different_export = json.loads(json.dumps(task_export))
    different_export["tasks"][0]["task_config"]["eval"][
        "reference_url"
    ] = "https://site.example/a-different-outcome"
    different_export["tasks"][0]["evaluator"]["config"] = dict(
        different_export["tasks"][0]["task_config"]["eval"]
    )
    different_export["resolved_task_set_sha256"] = sha256_json(
        different_export["tasks"]
    )
    different_audit = build_webarena_task_interface_audit(different_export)
    different_requirements = build_evaluator_requirements_artifact(
        task_export=different_export,
        task_interface_audit=different_audit,
    )
    with pytest.raises(SchemaError, match="differs from the frozen resolved-task"):
        validate_evaluator_requirements_resolved_snapshot_binding(
            different_requirements,
            task_export=different_export,
            resolved_task_snapshot=snapshot,
        )

    forged_snapshot = json.loads(json.dumps(snapshot))
    forged_snapshot["page_state_evaluator_compile_report"]["tasks"][0][
        "criterion_count"
    ] = 999
    forged_snapshot[
        "page_state_evaluator_compile_report_content_sha256"
    ] = sha256_json(forged_snapshot["page_state_evaluator_compile_report"])
    with pytest.raises(SchemaError, match="exact 50-task recompilation"):
        validate_evaluator_requirements_resolved_snapshot_binding(
            requirements,
            task_export=task_export,
            resolved_task_snapshot=forged_snapshot,
        )


def _evaluator_provenance(
    tmp_path: Path,
    *,
    page_state_only: bool = False,
    task_export: dict | None = None,
    task_audit: dict | None = None,
    sealed_evaluator_identity: dict,
) -> dict:
    evidence = tmp_path / "evaluator-provenance"
    generated_task_authority = task_export is None and task_audit is None
    if task_export is None or task_audit is None:
        if task_export is not None or task_audit is not None:
            raise ValueError("task export and audit must be supplied together")
        task_export, task_audit = _task_export_and_audit(
            page_state_only=page_state_only
        )
    if generated_task_authority and page_state_only:
        for row in task_export["tasks"]:
            row["evaluator"]["evaluator_id"] = sealed_evaluator_identity[
                "evaluator_id"
            ]
            row["evaluator"]["evaluator_version"] = sealed_evaluator_identity[
                "evaluator_version"
            ]
        task_export["resolved_task_set_sha256"] = sha256_json(
            task_export["tasks"]
        )
        task_audit = build_webarena_task_interface_audit(task_export)
    requirements = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=task_audit,
    )
    requirements_path = _write_json(
        evidence / "evaluator_requirements.json", requirements
    )
    runtime_evaluator = _write_bytes(
        evidence / "runtime_evaluators.py",
        b"# measured libwebarena evaluator module\n",
    )
    delta = _write_json(
        evidence / "compatibility_delta.json",
        {"classification": "compatibility_port", "differences": ["relative imports"]},
    )
    upstream_reference = _write_bytes(
        evidence / "upstream_parity_reference.json",
        (
            ROOT
            / "tests/table2/fixtures/webarena_page_state_reference.json"
        ).read_bytes(),
    )
    compile_report = requirements["page_state_compile_report"]
    parity_value = {
        "schema_version": "table2-webarena-final-byte-parity-v1",
        "record_type": "WebArenaEvaluatorFinalByteParityReceipt",
        "status": "PASS",
        "parity_scope": "FINAL_EVALUATOR_BYTES_AGAINST_PINNED_REFERENCE",
        "implementation_id": sealed_evaluator_identity["evaluator_id"],
        "implementation_version": sealed_evaluator_identity[
            "evaluator_version"
        ],
        "implementation_source_sha256": sealed_evaluator_identity[
            "source_sha256"
        ],
        "page_state_compiler_source_sha256": (
            compile_report["compiler_source_sha256"]
            if compile_report is not None
            else None
        ),
        "upstream_reference_sha256": sha256_file(upstream_reference),
        "case_count": 12,
        "all_cases_matched": True,
        "state_mutation_observed": False,
        "oracle_output_returned_to_runtime": False,
        "task_outcomes_observed": False,
        "measured_at_utc": "2026-09-04T10:30:00+06:00",
    }
    parity_value["parity_statement_sha256"] = sha256_json(parity_value)
    parity = _write_json(
        evidence / "compatibility_parity.json", parity_value
    )
    review_value = {
            "schema_version": COMPATIBILITY_REVIEW_SCHEMA_VERSION,
            "record_type": "WebArenaEvaluatorCompatibilityPortReview",
            "status": "PASS",
            "review_scope": "READ_ONLY_SEALED_EVALUATOR_COMPATIBILITY_PORT",
            "evaluator_access_mode": "READ_ONLY_SEALED",
            "state_mutation_permitted": False,
            "evaluator_remapping_permitted": False,
            "task_outcomes_observed_before_freeze": False,
            "registry_manifest_sha256": requirements["registry_manifest_sha256"],
            "resolved_task_set_sha256": requirements["resolved_task_set_sha256"],
            "evaluator_requirements_sha256": sha256_file(requirements_path),
            "page_state_compile_report_content_sha256": requirements[
                "page_state_compile_report_content_sha256"
            ],
            "page_state_compiler_source_sha256": (
                requirements["page_state_compile_report"][
                    "compiler_source_sha256"
                ]
                if requirements["page_state_compile_report"] is not None
                else None
            ),
            "implementation_id": sealed_evaluator_identity["evaluator_id"],
            "implementation_version": sealed_evaluator_identity[
                "evaluator_version"
            ],
            "implementation_source_sha256": sealed_evaluator_identity[
                "source_sha256"
            ],
            "implementation_classification": (
                "externally_reviewed_compatibility_port"
            ),
            "compatibility_delta_sha256": sha256_file(delta),
            "compatibility_parity_sha256": sha256_file(parity),
            "review_authority_id": "synthetic-independent-review-board",
            "review_authority_type": "INDEPENDENT_EXTERNAL_TECHNICAL_REVIEW",
            "reviewer_id": "synthetic-reviewer-01",
            "reviewer_organization": "Synthetic Independent QA",
            "reviewer_role": "external evaluator compatibility reviewer",
            "reviewer_independent_of_implementation": True,
            "reviewer_independent_of_campaign_execution": True,
            "reviewed_at_utc": "2026-09-04T11:00:00+06:00",
    }
    review_value["review_statement_sha256"] = sha256_json(review_value)
    review = _write_json(
        evidence / "compatibility_review.json",
        review_value,
    )
    synthetic_trust = set(
        live_deployment.PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S
    )
    synthetic_trust.add(
        live_deployment._external_review_trust_binding_sha256(
            review_receipt=review_value,
            review_file_sha256=sha256_file(review),
            parity_receipt=parity_value,
            parity_file_sha256=sha256_file(parity),
        )
    )
    # Test-only source: callers in other fixture modules do not inherit this
    # module's autouse fixture, so install their exact synthetic binding too.
    live_deployment.PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S = (
        synthetic_trust
    )
    evaluator_configuration = _write_json(
        evidence / "evaluator_configuration.json",
        {"task_set": "public-development-0-49", "sealed": True},
    )
    def relative(path: Path) -> str:
        return str(path.relative_to(tmp_path))

    judge_count = requirements["judge_required"]["task_count"]
    provenance = {
        "upstream_repository": PINNED_UPSTREAM_WEBARENA_REPOSITORY,
        "upstream_revision": PINNED_UPSTREAM_WEBARENA_REVISION,
        "upstream_evaluator_source_sha256": PINNED_UPSTREAM_EVALUATORS_SHA256,
        "upstream_judge_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
        "runtime_distribution": "libwebarena",
        "runtime_distribution_version": PINNED_LIBWEBARENA_VERSION,
        "runtime_distribution_sha256": PINNED_LIBWEBARENA_WHEEL_SHA256,
        "runtime_evaluator_module": "webarena.evaluation_harness.evaluators",
        "runtime_evaluator_module_evidence_path": relative(runtime_evaluator),
        "runtime_evaluator_module_sha256": sha256_file(runtime_evaluator),
        "implementation_classification": "externally_reviewed_compatibility_port",
        "byte_identical_to_upstream": False,
        "compatibility_delta_evidence_path": relative(delta),
        "compatibility_delta_sha256": sha256_file(delta),
        "upstream_parity_reference_evidence_path": relative(
            upstream_reference
        ),
        "upstream_parity_reference_sha256": sha256_file(upstream_reference),
        "compatibility_parity_evidence_path": relative(parity),
        "compatibility_parity_sha256": sha256_file(parity),
        "compatibility_review_evidence_path": relative(review),
        "compatibility_review_sha256": sha256_file(review),
        "evaluator_configuration_evidence_path": relative(evaluator_configuration),
        "evaluator_configuration_sha256": sha256_file(evaluator_configuration),
        "evaluator_requirements_evidence_path": relative(requirements_path),
        "evaluator_requirements_sha256": sha256_file(requirements_path),
        "judge_required_task_count": judge_count,
        "judge_required_task_set_sha256": requirements["judge_required"][
            "task_id_set_sha256"
        ],
        "string_match_task_count": requirements["string_match"]["task_count"],
        "string_match_task_set_sha256": requirements["string_match"][
            "task_id_set_sha256"
        ],
        "judge_model_substitution_allowed": False,
        "runtime_fallback_allowed": False,
        "judge_outputs_returned_to_runtime": 0,
    }
    if judge_count:
        runtime_judge = _write_bytes(
            evidence / "runtime_helper_functions.py",
            b"# measured libwebarena judge module\n",
        )
        response_schema = _write_json(
            evidence / "judge_response_schema.json",
            {"type": "string", "allowed_verdicts": ["correct", "incorrect"]},
        )
        availability = _write_json(
            evidence / "judge_api_availability.json",
            {
                "schema_version": "table2-webarena-judge-api-preflight-v1",
                "record_type": "WebArenaJudgeAPIPreflightReceipt",
                "model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
                "availability_status": "PASS",
                "decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
                "prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
                "response_schema_sha256": sha256_file(response_schema),
                "request_succeeded": True,
                "response_conformed_to_schema": True,
                "credential_material_embedded": False,
                "evaluator_output_returned_to_runtime": False,
                "response_content_sha256": "8" * 64,
                "measured_at_utc": "2026-09-04T12:00:00+06:00",
            },
        )
        provenance.update(
            {
                "runtime_judge_module_evidence_path": relative(runtime_judge),
                "runtime_judge_module_sha256": sha256_file(runtime_judge),
                "judge_model_id": PINNED_WEBARENA_JUDGE_MODEL_ID,
                "judge_prompt_source_sha256": PINNED_UPSTREAM_JUDGE_SOURCE_SHA256,
                "judge_decoding_parameters": dict(PINNED_WEBARENA_JUDGE_DECODING),
                "judge_api_response_schema_evidence_path": relative(response_schema),
                "judge_api_response_schema_sha256": sha256_file(response_schema),
                "judge_api_availability_receipt_evidence_path": relative(availability),
                "judge_api_availability_receipt_sha256": sha256_file(availability),
                "judge_api_preflight_status": "PASS",
                "unavailable_task_policy": (
                    "preregistered_exclusion_or_disclosed_evaluator_change_before_outcomes"
                ),
            }
        )
    else:
        provenance.update(
            {
                "runtime_judge_module_evidence_path": None,
                "runtime_judge_module_sha256": None,
                "judge_model_id": None,
                "judge_prompt_source_sha256": None,
                "judge_decoding_parameters": None,
                "judge_api_response_schema_evidence_path": None,
                "judge_api_response_schema_sha256": None,
                "judge_api_availability_receipt_evidence_path": None,
                "judge_api_availability_receipt_sha256": None,
                "judge_api_preflight_status": "NOT_APPLICABLE",
                "unavailable_task_policy": JUDGE_API_NOT_APPLICABLE,
            }
        )
    return provenance


def _manifest(
    tmp_path: Path,
    *,
    page_state_only: bool = False,
    task_export: dict | None = None,
    task_audit: dict | None = None,
    sealed_evaluator_identity: dict | None = None,
) -> dict:
    source_sha256 = sha256_file(ROOT / SOURCE_RELATIVE)
    effective_sealed_identity = sealed_evaluator_identity or {
        "evaluator_id": "measured-sealed_webarena_evaluator",
        "evaluator_version": "v1",
        "source_relative_path": SEALED_SOURCE_RELATIVE,
        "source_sha256": sha256_file(ROOT / SEALED_SOURCE_RELATIVE),
    }
    evaluator_provenance = _evaluator_provenance(
        tmp_path,
        page_state_only=page_state_only,
        task_export=task_export,
        task_audit=task_audit,
        sealed_evaluator_identity=effective_sealed_identity,
    )
    capabilities = {}
    for index, (capability_id, (plane, return_contract)) in enumerate(
        sorted(REQUIRED_LIVE_CAPABILITIES.items()), start=1
    ):
        identity = (
            effective_sealed_identity
            if capability_id == "sealed_webarena_evaluator"
            else {
                "evaluator_id": f"measured-{capability_id}",
                "evaluator_version": "v1",
                "source_relative_path": SOURCE_RELATIVE,
                "source_sha256": source_sha256,
            }
        )
        capability_source_sha256 = identity["source_sha256"]
        evidence = _write_json(
            tmp_path / "evidence" / f"{capability_id}.json",
            {
                "schema_version": LIVE_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
                "record_type": LIVE_CAPABILITY_EVIDENCE_RECORD_TYPE,
                "capability_id": capability_id,
                "status": "PASS",
                "implementation_source_sha256": capability_source_sha256,
                "deployment_state_sha256": f"{index:x}" * 64,
                "measured_at_utc": "2026-09-04T12:00:00+06:00",
                "locked_test_rows_read": 0,
                "oracle_values_returned_to_runtime": 0,
                "claims": REQUIRED_CAPABILITY_CLAIMS[capability_id],
                "provenance": (
                    evaluator_provenance
                    if capability_id == "sealed_webarena_evaluator"
                    else {}
                ),
            },
        )
        capabilities[capability_id] = {
            "capability_id": capability_id,
            "plane": plane,
            "implementation_id": identity["evaluator_id"],
            "implementation_version": identity["evaluator_version"],
            "source_relative_path": identity["source_relative_path"],
            "source_sha256": capability_source_sha256,
            "readiness_evidence_path": str(evidence.relative_to(tmp_path)),
            "readiness_evidence_sha256": sha256_file(evidence),
            "runtime_return_contract": return_contract,
            "frozen": True,
        }
    return {
        "schema_version": LIVE_DEPLOYMENT_SCHEMA_VERSION,
        "record_type": LIVE_DEPLOYMENT_RECORD_TYPE,
        "protocol_id": "table2-pc01-pilot-v1",
        "model_id": "qwen2vl_2b_gold_v2_8_dgx:epoch6:seed42",
        "model_seed": 42,
        "campaign_scope": "PROVISIONAL_ENGINEERING_PILOT",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "benchmark": "webarena",
        "benchmark_version": REGISTERED_BROWSERGYM_VERSION,
        "credential_material_embedded": False,
        "locked_test_rows_read": 0,
        "browsergym_validation_boundary": {
            "browsergym_version": REGISTERED_BROWSERGYM_VERSION,
            "execution_path": REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
            "default_step_called": False,
            "task_validation_called": False,
            "reward_slot_read": False,
            "termination_slots_read": False,
            "info_slot_read": False,
            "oracle_output_exposed_to_runtime": False,
            "upstream_start_state_fields": list(
                UPSTREAM_WEBARENA_START_STATE_FIELDS
            ),
            "generic_webarena_task_used": False,
            "adapter_close_deferred_until_sealed_finalization": True,
            "cleanup_on_final_success": True,
            "cleanup_on_final_error": True,
            "cleanup_on_runtime_abort": True,
        },
        "sealed_page_broker": {
            "contract": REGISTERED_SEALED_BROKER_CONTRACT,
            "source_relative_path": BROKER_SOURCE_RELATIVE,
            "source_sha256": sha256_file(ROOT / BROKER_SOURCE_RELATIVE),
            "runtime_can_evaluate": False,
            "evaluator_can_publish": False,
            "live_page_returned_to_runtime": False,
            "evaluation_result_return_contract": "opaque_terminal_signal_only",
            "start_state_contract": REGISTERED_START_STATE_CONTRACT,
            "generic_webarena_task_allowed": False,
            "adapter_close_deferred_until_sealed_finalization": True,
            "cleanup_on_final_success": True,
            "cleanup_on_final_error": True,
            "cleanup_on_runtime_abort": True,
        },
        "capabilities": capabilities,
    }


def _bound_environment(staged) -> dict:
    sealed = staged.manifest["capabilities"]["sealed_webarena_evaluator"]
    return {
        LIVE_DEPLOYMENT_BINDING_FIELD: staged.binding,
        "evaluator": {
            "evaluator_id": sealed["implementation_id"],
            "evaluator_version": sealed["implementation_version"],
            "source_relative_path": sealed["source_relative_path"],
            "source_sha256": sealed["source_sha256"],
        },
    }


def test_complete_measured_deployment_manifest_validates(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    assert validate_pc01_live_deployment_manifest(
        value,
        repository_root=ROOT,
        evidence_root=tmp_path,
    ) == value


RUNTIME_CAPABILITY_IDS = tuple(
    sorted(
        set(REQUIRED_LIVE_CAPABILITIES)
        - {"sealed_webarena_evaluator"}
    )
)


def _replace_runtime_capability_source(
    value: dict,
    *,
    capability_id: str,
    forbidden_source: dict,
    evidence_root: Path,
) -> None:
    row = value["capabilities"][capability_id]
    row["source_relative_path"] = forbidden_source["source_relative_path"]
    row["source_sha256"] = forbidden_source["source_sha256"]
    evidence_path = evidence_root / row["readiness_evidence_path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["implementation_source_sha256"] = forbidden_source[
        "source_sha256"
    ]
    _write_json(evidence_path, evidence)
    row["readiness_evidence_sha256"] = sha256_file(evidence_path)


@pytest.mark.parametrize("capability_id", RUNTIME_CAPABILITY_IDS)
@pytest.mark.parametrize(
    "forbidden_plane",
    ("sealed_webarena_evaluator", "sealed_page_broker"),
)
def test_each_runtime_capability_source_is_disjoint_from_forbidden_planes(
    tmp_path: Path,
    capability_id: str,
    forbidden_plane: str,
) -> None:
    value = _manifest(tmp_path)
    forbidden_source = (
        value["capabilities"][forbidden_plane]
        if forbidden_plane == "sealed_webarena_evaluator"
        else value[forbidden_plane]
    )
    _replace_runtime_capability_source(
        value,
        capability_id=capability_id,
        forbidden_source=forbidden_source,
        evidence_root=tmp_path,
    )

    with pytest.raises(
        SchemaError,
        match=(
            "runtime capability source plane overlaps the sealed evaluator "
            f"or broker: {capability_id}"
        ),
    ):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


@pytest.mark.parametrize(
    "forbidden_plane",
    ("sealed_webarena_evaluator", "sealed_page_broker"),
)
def test_handoff_reasserts_runtime_capability_source_plane_disjointness(
    tmp_path: Path,
    forbidden_plane: str,
) -> None:
    value = _manifest(tmp_path)
    forbidden_source = (
        value["capabilities"][forbidden_plane]
        if forbidden_plane == "sealed_webarena_evaluator"
        else value[forbidden_plane]
    )
    value["capabilities"]["deterministic_reset"].update(
        {
            "source_relative_path": forbidden_source[
                "source_relative_path"
            ],
            "source_sha256": forbidden_source["source_sha256"],
        }
    )

    with pytest.raises(
        SchemaError,
        match="handoff live-deployment capability source planes are not disjoint",
    ):
        _validate_handoff_live_capability_source_planes(
            SimpleNamespace(manifest=value)
        )


def test_source_plane_validator_rejects_same_bytes_under_a_path_alias(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path)
    sealed = value["capabilities"]["sealed_webarena_evaluator"]
    runtime = value["capabilities"]["deterministic_reset"]
    runtime["source_sha256"] = sealed["source_sha256"]

    with pytest.raises(
        SchemaError,
        match="runtime capability source plane overlaps the sealed evaluator",
    ):
        validate_live_capability_source_plane_disjointness(value)


def test_pending_page_state_evaluator_identity_cannot_be_self_promoted(
    tmp_path: Path,
) -> None:
    source_relative = (
        "src/web_agent/eval/table2/webarena_page_state_evaluator.py"
    )
    value = _manifest(
        tmp_path,
        page_state_only=True,
        sealed_evaluator_identity={
            "evaluator_id": "table2-sealed-webarena-page-state-compatibility-port",
            "evaluator_version": (
                "reviewed_compatibility_port_pending_external_review"
            ),
            "source_relative_path": source_relative,
            "source_sha256": sha256_file(ROOT / source_relative),
        },
    )
    with pytest.raises(SchemaError, match="pending external-review identity"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_compile_report_rejects_a_page_state_config_outside_strict_grammar() -> None:
    task_export, task_audit = _task_export_and_audit(page_state_only=True)
    task_export["tasks"][17]["evaluator"]["config"][
        "unreviewed_extension"
    ] = True
    task_export["tasks"][17]["task_config"]["eval"] = dict(
        task_export["tasks"][17]["evaluator"]["config"]
    )
    task_export["resolved_task_set_sha256"] = sha256_json(task_export["tasks"])
    task_audit = build_webarena_task_interface_audit(task_export)
    with pytest.raises(SchemaError, match="rejected exact task config"):
        build_evaluator_requirements_artifact(
            task_export=task_export,
            task_interface_audit=task_audit,
        )


def test_handoff_compile_authority_binds_pending_local_report_without_promoting_it() -> None:
    task_export, _ = _task_export_and_audit(page_state_only=True)
    for row in task_export["tasks"]:
        row["evaluator"]["evaluator_id"] = (
            "table2-sealed-webarena-page-state-compatibility-port"
        )
        row["evaluator"]["evaluator_version"] = (
            "reviewed_compatibility_port_pending_external_review"
        )
    task_export["resolved_task_set_sha256"] = sha256_json(task_export["tasks"])
    audit = build_webarena_task_interface_audit(task_export)
    requirements = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=audit,
    )
    report = requirements["page_state_compile_report"]
    assert report["campaign_authority"] is False
    assert report["implementation_specific_report_status"] == (
        "BOUND_PENDING_EXTERNAL_REVIEW_INPUT"
    )
    assert report["implementation_specific_compile_report"][
        "campaign_ready"
    ] is False
    assert report["implementation_specific_compile_report_content_sha256"] == (
        sha256_json(report["implementation_specific_compile_report"])
    )


def test_external_review_identity_and_statement_hash_are_mandatory(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path, page_state_only=True)
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    provenance = readiness["provenance"]
    review_path = tmp_path / provenance["compatibility_review_evidence_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewer_id"] = review["implementation_id"]
    unsigned = dict(review)
    unsigned.pop("review_statement_sha256")
    review["review_statement_sha256"] = sha256_json(unsigned)
    _write_json(review_path, review)
    provenance["compatibility_review_sha256"] = sha256_file(review_path)
    _write_json(readiness_path, readiness)
    evaluator_row["readiness_evidence_sha256"] = sha256_file(readiness_path)

    with pytest.raises(SchemaError, match="reviewer cannot be"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_final_byte_parity_cannot_describe_different_implementation_bytes(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path, page_state_only=True)
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    provenance = readiness["provenance"]
    parity_path = tmp_path / provenance["compatibility_parity_evidence_path"]
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["implementation_source_sha256"] = "1" * 64
    unsigned = dict(parity)
    unsigned.pop("parity_statement_sha256")
    parity["parity_statement_sha256"] = sha256_json(unsigned)
    _write_json(parity_path, parity)
    provenance["compatibility_parity_sha256"] = sha256_file(parity_path)
    _write_json(readiness_path, readiness)
    evaluator_row["readiness_evidence_sha256"] = sha256_file(readiness_path)

    with pytest.raises(
        SchemaError,
        match="final-byte compatibility parity implementation_source_sha256",
    ):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_staging_copies_only_transitively_referenced_live_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    value = _manifest(source)
    manifest = _write_json(source / "deployment.json", value)
    _write_bytes(source / "credentials.json", b"must-not-be-copied")
    destination = tmp_path / "handoff"

    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest,
        evidence_root=source,
        repository_root=ROOT,
        destination_artifact_root=destination,
    )
    assert not (staged.package_root / "credentials.json").exists()
    assert staged.binding["manifest_sha256"] == sha256_file(
        staged.package_root / "manifest.json"
    )
    assert staged.evaluator_requirements["judge_required"]["task_count"] == 13
    assert staged.evaluator_requirements["resolved_task_set_sha256"] == (
        json.loads(
            (
                staged.package_root
                / "evaluator-provenance/evaluator_requirements.json"
            ).read_text(encoding="utf-8")
        )["resolved_task_set_sha256"]
    )
    environment = _bound_environment(staged)
    reopened = validate_bound_pc01_live_deployment(
        environment,
        artifact_root=destination,
        repository_root=ROOT,
    )
    assert reopened.binding == staged.binding
    assert reopened.evaluator_requirements == staged.evaluator_requirements
    assert len(reopened.capability_source_files) == 3


def test_staged_live_evidence_rejects_tampering_and_unreferenced_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    manifest = _write_json(source / "deployment.json", _manifest(source))
    destination = tmp_path / "handoff"
    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest,
        evidence_root=source,
        repository_root=ROOT,
        destination_artifact_root=destination,
    )
    environment = _bound_environment(staged)
    extra = _write_json(staged.package_root / "unreferenced.json", {"secret": True})
    with pytest.raises(SchemaError, match="unreferenced evidence"):
        validate_bound_pc01_live_deployment(
            environment,
            artifact_root=destination,
            repository_root=ROOT,
        )
    extra.unlink()
    readiness = next(
        path
        for path in staged.package_files
        if path.name == "deterministic_reset.json"
    )
    readiness.write_bytes(readiness.read_bytes() + b" ")
    with pytest.raises(SchemaError, match="readiness evidence hash differs"):
        validate_bound_pc01_live_deployment(
            environment,
            artifact_root=destination,
            repository_root=ROOT,
        )


def test_bound_deployment_rejects_environment_evaluator_substitution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    manifest = _write_json(source / "deployment.json", _manifest(source))
    destination = tmp_path / "handoff"
    staged = stage_pc01_live_deployment_package(
        manifest_path=manifest,
        evidence_root=source,
        repository_root=ROOT,
        destination_artifact_root=destination,
    )
    environment = _bound_environment(staged)
    environment["evaluator"]["evaluator_id"] = "different-evaluator"
    with pytest.raises(
        SchemaError, match="environment evaluator differs.*evaluator_id"
    ):
        validate_bound_pc01_live_deployment(
            environment,
            artifact_root=destination,
            repository_root=ROOT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["capabilities"].pop("deterministic_reset"),
            "seven registered capabilities",
        ),
        (
            lambda value: value["browsergym_validation_boundary"].update(
                {"task_validation_called": True}
            ),
            "validation-disabled execution",
        ),
        (
            lambda value: value["sealed_page_broker"].update(
                {"runtime_can_evaluate": True}
            ),
            "runtime_can_evaluate",
        ),
        (
            lambda value: value["capabilities"]["recovery_action_planner"].update(
                {"implementation_id": "TBD placeholder"}
            ),
            "non-production marker",
        ),
    ),
)
def test_manifest_fails_closed_on_missing_or_weakened_capabilities(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    value = _manifest(tmp_path)
    mutation(value)
    with pytest.raises(SchemaError, match=message):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_readiness_claims_and_evidence_bytes_are_not_self_attested(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["exclusive_input_audit"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_value["claims"]["manual_input_event_count"] = 1
    _write_json(evidence, evidence_value)

    with pytest.raises(SchemaError, match="readiness evidence hash differs"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )

    row["readiness_evidence_sha256"] = sha256_file(evidence)
    with pytest.raises(SchemaError, match="readiness claims differ"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_evaluator_capability_cannot_return_oracle_values_to_runtime(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_value["oracle_values_returned_to_runtime"] = 1
    _write_json(evidence, evidence_value)
    row["readiness_evidence_sha256"] = sha256_file(evidence)
    with pytest.raises(SchemaError, match="oracle_values_returned_to_runtime mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "implementation_classification",
            "official_webarena_evaluator",
            "implementation_classification",
        ),
        ("judge_model_id", "newer-judge", "judge_model_id"),
        ("judge_api_preflight_status", "UNAVAILABLE", "judge_api_preflight_status"),
        ("runtime_fallback_allowed", True, "runtime_fallback_allowed"),
    ),
)
def test_evaluator_provenance_forbids_bare_official_claim_or_judge_fallback(
    tmp_path: Path,
    field: str,
    replacement,
    message: str,
) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    evidence_value["provenance"][field] = replacement
    _write_json(evidence, evidence_value)
    row["readiness_evidence_sha256"] = sha256_file(evidence)
    with pytest.raises(SchemaError, match=message):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_evaluator_compatibility_review_and_api_receipt_bytes_are_required(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    evidence = tmp_path / row["readiness_evidence_path"]
    evidence_value = json.loads(evidence.read_text(encoding="utf-8"))
    provenance = evidence_value["provenance"]
    review = tmp_path / provenance["compatibility_review_evidence_path"]
    review.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="compatibility_review_sha256 differs"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_measured_api_receipt_cannot_silently_use_a_newer_judge(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness = tmp_path / row["readiness_evidence_path"]
    readiness_value = json.loads(readiness.read_text(encoding="utf-8"))
    provenance = readiness_value["provenance"]
    receipt = tmp_path / provenance["judge_api_availability_receipt_evidence_path"]
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["model_id"] = "newer-judge"
    _write_json(receipt, receipt_value)
    provenance["judge_api_availability_receipt_sha256"] = sha256_file(receipt)
    _write_json(readiness, readiness_value)
    row["readiness_evidence_sha256"] = sha256_file(readiness)
    with pytest.raises(SchemaError, match="availability receipt model_id mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_manifest_rejects_boolean_zero_for_strict_leakage_counts(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    value["locked_test_rows_read"] = False
    with pytest.raises(SchemaError, match="locked_test_rows_read mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_evaluator_requirements_bind_task_order_registry_and_resolved_hash() -> None:
    task_export, audit = _task_export_and_audit(page_state_only=True)
    requirements = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=audit,
    )

    assert requirements["registry_manifest_id"] == task_export[
        "registry_manifest_id"
    ]
    assert requirements["registry_manifest_sha256"] == task_export[
        "registry_manifest_sha256"
    ]
    assert requirements["resolved_task_set_sha256"] == task_export[
        "resolved_task_set_sha256"
    ]
    assert requirements["ordered_task_ids_sha256"] == sha256_json(
        [row["task_id"] for row in task_export["tasks"]]
    )
    assert validate_evaluator_requirements_derivation(
        requirements,
        task_export=task_export,
        task_interface_audit=audit,
    ) == requirements

    reordered_export = json.loads(json.dumps(task_export))
    reordered_export["tasks"][0], reordered_export["tasks"][1] = (
        reordered_export["tasks"][1],
        reordered_export["tasks"][0],
    )
    reordered_export["resolved_task_set_sha256"] = sha256_json(
        reordered_export["tasks"]
    )
    reordered_audit = build_webarena_task_interface_audit(reordered_export)
    with pytest.raises(SchemaError, match="exact task-export/interface-audit"):
        validate_evaluator_requirements_derivation(
            requirements,
            task_export=reordered_export,
            task_interface_audit=reordered_audit,
        )

    changed_registry_export = json.loads(json.dumps(task_export))
    changed_registry_export["registry_manifest_sha256"] = "4" * 64
    changed_registry_audit = build_webarena_task_interface_audit(
        changed_registry_export
    )
    with pytest.raises(SchemaError, match="exact task-export/interface-audit"):
        validate_evaluator_requirements_derivation(
            requirements,
            task_export=changed_registry_export,
            task_interface_audit=changed_registry_audit,
        )


def test_evaluator_requirements_reject_falsified_counts_and_task_sets() -> None:
    task_export, audit = _task_export_and_audit(page_state_only=False)
    requirements = build_evaluator_requirements_artifact(
        task_export=task_export,
        task_interface_audit=audit,
    )

    false_count = json.loads(json.dumps(requirements))
    false_count["judge_required"]["task_count"] = 12
    with pytest.raises(SchemaError, match="judge-required counts"):
        validate_evaluator_requirements_artifact(false_count)

    false_set = json.loads(json.dumps(requirements))
    false_set["judge_required"]["task_ids"] = false_set["judge_required"][
        "task_ids"
    ][1:]
    false_set["judge_required"]["task_count"] = len(
        false_set["judge_required"]["task_ids"]
    )
    false_set["judge_required"]["task_id_set_sha256"] = sha256_json(
        false_set["judge_required"]["task_ids"]
    )
    with pytest.raises(SchemaError, match="judge-required counts"):
        validate_evaluator_requirements_artifact(false_set)


def test_page_state_only_profile_requires_zero_judge_na_and_no_fake_pass(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path, page_state_only=True)
    assert validate_pc01_live_deployment_manifest(
        value,
        repository_root=ROOT,
        evidence_root=tmp_path,
    ) == value
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    provenance = readiness["provenance"]
    assert provenance["judge_required_task_count"] == 0
    assert provenance["string_match_task_count"] == 0
    assert provenance["judge_model_id"] is None
    assert provenance["runtime_judge_module_evidence_path"] is None
    assert provenance["judge_api_availability_receipt_evidence_path"] is None
    assert provenance["judge_api_preflight_status"] == "NOT_APPLICABLE"
    assert not (tmp_path / "evaluator-provenance/runtime_helper_functions.py").exists()
    assert not (tmp_path / "evaluator-provenance/judge_api_availability.json").exists()

    provenance["judge_api_preflight_status"] = "PASS"
    _write_json(readiness_path, readiness)
    evaluator_row["readiness_evidence_sha256"] = sha256_file(readiness_path)
    with pytest.raises(SchemaError, match="zero-judge.*judge_api_preflight_status"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_nonzero_judge_profile_requires_measured_api_evidence(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    provenance = readiness["provenance"]
    assert provenance["judge_required_task_count"] == 13
    receipt = tmp_path / provenance[
        "judge_api_availability_receipt_evidence_path"
    ]
    receipt.unlink()

    with pytest.raises(SchemaError, match="unsafe, missing, or empty"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_read_only_compatibility_review_cannot_be_self_resealed(
    tmp_path: Path,
) -> None:
    value = _manifest(tmp_path, page_state_only=True)
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    provenance = readiness["provenance"]
    review_path = tmp_path / provenance["compatibility_review_evidence_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["state_mutation_permitted"] = True
    _write_json(review_path, review)
    provenance["compatibility_review_sha256"] = sha256_file(review_path)
    _write_json(readiness_path, readiness)
    evaluator_row["readiness_evidence_sha256"] = sha256_file(readiness_path)

    with pytest.raises(SchemaError, match="read-only, outcome-blind review"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_fully_rehashed_review_and_parity_fabrication_lacks_trusted_authority(
    tmp_path: Path,
) -> None:
    """Recomputing every caller-controlled hash cannot create review authority."""

    value = _manifest(tmp_path, page_state_only=True)
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    provenance = readiness["provenance"]

    parity_path = tmp_path / provenance["compatibility_parity_evidence_path"]
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["measured_at_utc"] = "2026-09-04T10:45:00+06:00"
    parity.pop("parity_statement_sha256")
    parity["parity_statement_sha256"] = sha256_json(parity)
    _write_json(parity_path, parity)
    provenance["compatibility_parity_sha256"] = sha256_file(parity_path)

    review_path = tmp_path / provenance["compatibility_review_evidence_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        {
            "compatibility_parity_sha256": sha256_file(parity_path),
            "review_authority_id": "caller-authored-fake-authority",
            "reviewer_id": "caller-authored-fake-reviewer",
            "reviewed_at_utc": "2026-09-04T11:15:00+06:00",
        }
    )
    review.pop("review_statement_sha256")
    review["review_statement_sha256"] = sha256_json(review)
    _write_json(review_path, review)
    provenance["compatibility_review_sha256"] = sha256_file(review_path)

    _write_json(readiness_path, readiness)
    evaluator_row["readiness_evidence_sha256"] = sha256_file(readiness_path)

    with pytest.raises(SchemaError, match="source-tracked trust authority"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_self_hashed_external_review_is_unpromotable_without_pinned_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _manifest(tmp_path, page_state_only=True)
    monkeypatch.setattr(
        live_deployment,
        "PINNED_EXTERNAL_EVALUATOR_REVIEW_BINDING_SHA256S",
        frozenset(),
    )
    with pytest.raises(SchemaError, match="self-hashed caller JSON cannot promote"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )


def test_provenance_counts_must_equal_derived_requirements(tmp_path: Path) -> None:
    value = _manifest(tmp_path)
    evaluator_row = value["capabilities"]["sealed_webarena_evaluator"]
    readiness_path = tmp_path / evaluator_row["readiness_evidence_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["provenance"]["string_match_task_count"] = 0
    _write_json(readiness_path, readiness)
    evaluator_row["readiness_evidence_sha256"] = sha256_file(readiness_path)

    with pytest.raises(SchemaError, match="string_match_task_count mismatch"):
        validate_pc01_live_deployment_manifest(
            value,
            repository_root=ROOT,
            evidence_root=tmp_path,
        )
