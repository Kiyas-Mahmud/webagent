from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from web_agent.eval.table2.common import SchemaError, sha256_file
from web_agent.eval.table2.paper_claims import (
    CLAIM_READY_STATUS,
    CLAIM_RULES,
    DECISION_RULE_VALIDATORS,
    DISCLOSED_CLAIM_IDS,
    EVIDENCE_RULES,
    EXECUTABLE_DECISION_RULES,
    NR_LIMITATION,
    PROHIBITED_CLAIMS,
    REGISTERED_PILLAR_LIMITATIONS,
    STATUS_TEXT,
    build_claim_evidence_assessment,
    build_initial_claim_report,
    validate_claim_registry,
    validate_claim_registry_binding,
    validate_claim_registry_mapping,
    validate_claim_report,
    validate_paper_claim_readiness,
    write_claim_evidence_assessment,
    write_initial_claim_report,
)
from web_agent.runtime.protocol import validate_frozen_protocol_mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPOSITORY_ROOT / "configs/eval/table2/paper_claim_registry_v1.json"


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _campaign(tmp_path: Path, *, ready: bool) -> tuple[Path, dict]:
    root = tmp_path / ("final" if ready else "pilot")
    campaign = {
        "schema_version": "table2.v1",
        "campaign_id": "claim-fixture",
        "campaign_kind": "locked_final" if ready else "engineering_pilot",
        "campaign_mode": "evaluation",
        "evidence_label": "FINAL_LOCKED" if ready else "PILOT_ONLY",
        "publication_status": "N/R" if ready else "DRAFT_PILOT_ONLY",
        "paper_table_status": "N/R",
        "paper_claim_registry_id": "table2-research-locked-claims-v1",
        "paper_claim_registry_sha256": sha256_file(REGISTRY),
    }
    _write_json(root / "campaign_manifest.json", campaign)
    return root, campaign


def _metric_system() -> dict:
    metric = {"estimate": 0.5, "numerator": 1, "denominator": 2}
    return {
        "steps": {"all": dict(metric)},
        "efficiency": {
            "model_call_count": dict(metric),
            "task_wall_clock_seconds": dict(metric),
        },
        "recovery_success_rate": dict(metric),
        "success_after_initial_failure": dict(metric),
        "unrecovered_failure_rate": dict(metric),
    }


def _companion_envelope(pillar: str) -> dict:
    return {
        "schema_version": f"table2.pillar{pillar[-1]}-diagnostic-report.v1",
        "pillar": pillar,
        "campaign_id": "claim-fixture",
        "paper_table_status": "N/R",
        "affects_primary_table2": False,
        "locked_test_rows_read": 0,
        "evidence_role": f"{pillar}_COMPANION_MECHANISM_EVIDENCE",
        "promotion_status": "CANONICAL_PC01_COMPANION_EVIDENCE",
        "records": [{"record_type": "fixture"}],
    }


def _seed_canonical_sources(root: Path, *, condition_met: bool = True) -> None:
    _write_json(
        root / "aggregate/validation_report.json",
        {
            "campaign_dir": str(root),
            "status": "PASS",
            "publication_status": "READY_FOR_TABLE2",
            "errors": [],
            "warnings": [],
            "counts": {"included_blocks": 1},
        },
    )
    paired = {
        name: {
            "task_success": {
                "estimate": 0.1,
                "ci_low": -0.1,
                "ci_high": 0.3,
                "n_clusters": 1,
                "analysis_role": (
                    "descriptive_total_system_contrast"
                    if name == "E3_minus_E0"
                    else "registered_adjacent_inferential_contrast"
                ),
                "task_clustered_significance": (
                    {"inference_role": "descriptive_unadjusted_only"}
                    if name == "E3_minus_E0"
                    else {"inference_role": "registered_holm_family_member"}
                ),
            }
        }
        for name in (
            "E1_minus_E0",
            "E2_minus_E1",
            "E3_minus_E2",
            "E3_minus_E0",
        )
    }
    _write_json(
        root / "aggregate/statistics.json",
        {
            "schema_version": "table2.v1",
            "publication_status": "READY_FOR_TABLE2",
            "contrasts": paired,
            "multiple_testing": {
                "family": "registered_task_success_contrasts",
                "method": "Holm",
                "hypothesis_count": 3,
                "contrast_names": [
                    "E1_minus_E0",
                    "E2_minus_E1",
                    "E3_minus_E2",
                ],
            },
        },
    )
    metrics = {
        "schema_version": "table2.v1",
        "publication_status": "READY_FOR_TABLE2",
        "systems": {system: _metric_system() for system in ("E0", "E1", "E2", "E3")},
    }
    _write_json(root / "aggregate/metrics.json", metrics)
    _write_json(
        root / "aggregate/recovery_diagnostic_metrics.json",
        {
            **metrics,
            "publication_status": "PILOT_ONLY",
            "task_partition": "recovery_diagnostic",
        },
    )
    _write_json(
        root / "aggregate/architecture_interface_matched_control.json",
        {
            "schema_version": "table2-architecture-interface-control-v1",
            "campaign_id": "claim-fixture",
            "architecture_matched": condition_met,
            "interface_matched": condition_met,
            "assessment": "DECLARATIVE_ONLY_NOT_SOURCE_ATTESTED",
        },
    )
    _write_json(
        root / "memory/seed_42/manifest.json",
        {
            "schema_version": "table2-frozen-memory-v2",
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "model_seed": 42,
            "embedding_dimension": 768,
            "normalization": "l2",
            "similarity": "cosine",
            "top_k": 3,
            "tie_break": "memory_id_ascending",
            "admission_threshold_source": "train_only_calibration",
            "same_task_exclusion": True,
            "duplicate_exclusion": True,
            "runtime_writes_allowed": False,
            "item_count": 2,
        },
    )
    _write_json(
        root / "aggregate/retrieval_diagnostics.json",
        {
            "schema_version": "table2.v1",
            "publication_status": "READY_FOR_TABLE2",
            "status": "OK",
            "query_count": 1,
            "retrieval_coverage": {"estimate": 1.0, "numerator": 1, "denominator": 1},
            "intervention_coverage": {
                "estimate": 1.0,
                "numerator": 1,
                "denominator": 1,
            },
        },
    )
    _write_json(
        root / "aggregate/trigger_source_stratified_recovery.json",
        {
            "schema_version": "table2-trigger-source-recovery-v1",
            "campaign_id": "claim-fixture",
            "publication_status": "READY_FOR_TABLE2",
            "by_trigger_source": {"learned": {"episode_count": 1}},
        },
    )
    _write_json(
        root / "component_test/pillar1/learned_trigger_only_ablation.json",
        {
            "schema_version": "table2-learned-trigger-ablation-v1",
            "campaign_id": "claim-fixture",
            "ablation_executed": condition_met,
            "isolates_learned_trigger": condition_met,
            "assessment": "DECLARATIVE_ONLY_NOT_PAIRED_EXECUTION_EVIDENCE",
        },
    )
    p1 = {
        **_companion_envelope("P1"),
        "summary": {
            "overall": {
                "failure_detection_accuracy": {"estimate": 1.0},
                "failure_type_accuracy_on_failures": {"estimate": 1.0},
                "recovery_resolution_accuracy": {"estimate": 1.0},
                "recovery_progress_accuracy": {"estimate": 1.0},
            },
            "by_failure_type": {"NONE": {}},
            "by_recovery_strategy": {"RETRY": {}},
        },
    }
    _write_json(root / "component_test/pillar1/pillar1_diagnostic_report.json", p1)
    _write_json(
        root / "component_test/companion_input_provenance.json",
        {
            "schema_version": "table2-companion-input-provenance-v1",
            "campaign_id": "claim-fixture",
            "status": "PASS" if condition_met else "FAIL",
            "locked_test_rows_read": 0,
            "pillars": {
                pillar: {
                    "package_manifest_sha256": character * 64,
                    "authoritative_input_replay": condition_met,
                }
                for pillar, character in zip(("P1", "P2", "P3", "P4"), "abcd")
            },
        },
    )
    p2 = {
        **_companion_envelope("P2"),
        "runtime_outputs_consumed": False,
        "condition_definitions": {"image_occluded": "registered"},
        "temporal_replay": {"status": "PASS", "records": [{"identical": True}]},
        "summary": {
            "pre_action_modality_sensitivity": {"full": {}},
            "post_action_controls": {"full": {}},
        },
    }
    _write_json(root / "component_test/pillar2/pillar2_diagnostic_report.json", p2)
    p3 = {
        **_companion_envelope("P3"),
        "summary": {
            "overall": {
                "bbox_iou_mean": {"estimate": 0.5},
                "parameter_resolution_rate": {"estimate": 1.0},
            },
            "by_action_class": {"CLICK": {}},
            "by_target_size": {"small": {}},
            "invalid_output_count": 0,
            "accounted_rejected_executor_requests": 0,
        },
    }
    _write_json(root / "component_test/pillar3/pillar3_diagnostic_report.json", p3)
    p4 = {
        **_companion_envelope("P4"),
        "summary": {
            "query_count": 1,
            "retrieval_coverage": {"estimate": 1.0},
            "admission_rate": {"estimate": 1.0},
            "abstention_rate": {"estimate": 0.0},
        },
        "store_integrity": {"unchanged": True, "write_enabled": False},
        "memory_writes_executed": 0,
    }
    _write_json(root / "component_test/pillar4/pillar4_diagnostic_report.json", p4)
    _write_json(
        root / "runtime_readiness/checkpoint_embedding_execution_evidence.json",
        {
            "schema_version": "table2-checkpoint-embedding-execution-evidence-v1",
            "campaign_id": "claim-fixture",
            "checkpoint_sha256": "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a",
            "embedding_dimension": 768,
            "checkpoint_forward_executed": condition_met,
            "exact_p4_tensor_consumed": condition_met,
            "locked_test_rows_read": 0,
        },
    )
    _write_json(
        root / "paper_claim_evidence/disclosures.json",
        {
            "schema_version": "table2-paper-disclosures-v1",
            "campaign_id": "claim-fixture",
            "registered_contrasts_disclosed": {
                public: {
                    "statistics_contrast": statistics,
                    "estimate": paired[statistics]["task_success"]["estimate"],
                    "ci_low": paired[statistics]["task_success"]["ci_low"],
                    "ci_high": paired[statistics]["task_success"]["ci_high"],
                }
                for public, statistics in {
                    "E0_vs_E1": "E1_minus_E0",
                    "E1_vs_E2": "E2_minus_E1",
                    "E2_vs_E3": "E3_minus_E2",
                    "E0_vs_E3": "E3_minus_E0",
                }.items()
            },
            "claim_statuses_disclosed": {
                claim_id: "N/R" for claim_id in DISCLOSED_CLAIM_IDS
            },
            "claim_outcomes_disclosed": {
                claim_id: {
                    "status": "N/R",
                    "conclusion": STATUS_TEXT["N/R"][0],
                    "limitation": STATUS_TEXT["N/R"][1],
                }
                for claim_id in DISCLOSED_CLAIM_IDS
            },
            "pillar_limitations": REGISTERED_PILLAR_LIMITATIONS,
            "model_seed_scope": "seed_42_only_task_uncertainty_not_model_seed_uncertainty",
        },
    )


def _assessment(root: Path, kind: str, *, finding: str | None = None) -> dict[str, str]:
    value = build_claim_evidence_assessment(
        kind=kind,
        registry_path=REGISTRY,
        campaign_dir=root,
        finding=finding,
    )
    rule = next(item for item in EVIDENCE_RULES if item.kind == kind)
    path = _write_json(root / rule.relative_path, value)
    return {"kind": kind, "relative_path": rule.relative_path, "sha256": sha256_file(path)}


def _existing_assessment(root: Path, kind: str) -> dict[str, str]:
    rule = next(item for item in EVIDENCE_RULES if item.kind == kind)
    path = root / rule.relative_path
    return {"kind": kind, "relative_path": rule.relative_path, "sha256": sha256_file(path)}


def _ready_report(root: Path) -> dict:
    _seed_canonical_sources(root)
    registry = validate_claim_registry(REGISTRY)
    claims = [
        {
            "claim_id": rule.claim_id,
            "status": "N/R",
            "conclusion": "NOT_EVALUATED",
            "limitation": NR_LIMITATION,
            "evidence": [],
            "unmet_conditions": [],
        }
        for rule in CLAIM_RULES
    ]
    campaign_path = root / "campaign_manifest.json"
    validation_path = root / "aggregate/validation_report.json"
    return {
        "schema_version": "table2-paper-claim-report-v2",
        "registry_id": registry["registry_id"],
        "registry_sha256": sha256_file(REGISTRY),
        "campaign_id": "claim-fixture",
        "campaign_manifest_sha256": sha256_file(campaign_path),
        "publication_authority_relative_path": "aggregate/validation_report.json",
        "publication_authority_sha256": sha256_file(validation_path),
        "publication_status": "READY_FOR_TABLE2",
        "paper_table_status": "N/R",
        "claims": claims,
        "prohibited_claims": [
            {"claim_id": row.claim_id, "status": "NOT_CLAIMED"}
            for row in PROHIBITED_CLAIMS
        ],
    }


def _validate_ready(path: Path, root: Path) -> dict:
    authority = json.loads(
        (root / "aggregate/validation_report.json").read_text(encoding="utf-8")
    )
    validation = SimpleNamespace(
        passed=True,
        publication_status="READY_FOR_TABLE2",
        errors=[],
        to_dict=lambda: authority,
    )
    with patch(
        "web_agent.eval.table2.package_validator.validate_campaign",
        return_value=validation,
    ):
        return validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root)


def _validate_claim_readiness(path: Path, root: Path) -> dict:
    authority_path = root / "aggregate/validation_report.json"
    authority = (
        json.loads(authority_path.read_text(encoding="utf-8"))
        if authority_path.is_file()
        else {}
    )
    validation = SimpleNamespace(
        passed=True,
        publication_status="READY_FOR_TABLE2",
        errors=[],
        to_dict=lambda: authority,
    )
    with patch(
        "web_agent.eval.table2.package_validator.validate_campaign",
        return_value=validation,
    ):
        return validate_paper_claim_readiness(
            path,
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def _attach_registered_evidence(root: Path, claim: dict, rule_index: int) -> None:
    for kind in CLAIM_RULES[rule_index].required_evidence:
        rule = next(item for item in EVIDENCE_RULES if item.kind == kind)
        if not (root / rule.relative_path).is_file():
            _assessment(root, kind)
    claim["evidence"] = [
        _existing_assessment(root, kind)
        for kind in CLAIM_RULES[rule_index].required_evidence
    ]


def test_registered_claim_registry_is_exact_initially_nr_and_catalogued() -> None:
    value = validate_claim_registry(REGISTRY)
    assert value["paper_table_status"] == "N/R"
    assert value["publication_rules"]["paper_claim_ready_status"] == CLAIM_READY_STATUS
    assert value["publication_rules"]["ready_for_table2_does_not_imply_claim_readiness"]
    assert [row["claim_id"] for row in value["claims"]] == [
        row.claim_id for row in CLAIM_RULES
    ]
    assert len(value["evidence_catalog"]) == len(EVIDENCE_RULES)
    assert len({row["relative_path"] for row in value["evidence_catalog"]}) == len(
        EVIDENCE_RULES
    )
    claims = {row["claim_id"]: row for row in value["claims"]}
    assert "p3_grounded_action_execution_support" not in claims
    assert "browser execution is not measured" in claims[
        "p3_grounded_action_parameter_diagnostic_support"
    ]["interpretation"]
    assert claims["e1_e2_operational_recovery_increment"]["decision_rule"] == (
        "UNREGISTERED_BLOCKS_SUPPORTED_OR_NEGATIVE_OR_NULL"
    )
    assert set(DECISION_RULE_VALIDATORS) == set(EXECUTABLE_DECISION_RULES)


def test_registry_keeps_p1_and_p4_diagnostics_separate_from_browser_effects() -> None:
    claims = {row["claim_id"]: row for row in validate_claim_registry(REGISTRY)["claims"]}
    assert "not an operational recovery-effect claim" in claims[
        "p1_offline_diagnostic_support"
    ]["interpretation"]
    assert "final_package_validation" not in claims["p1_offline_diagnostic_support"][
        "required_evidence"
    ]
    assert "not an end-to-end memory-benefit claim" in claims[
        "p4_retrieval_admission_diagnostic_support"
    ]["interpretation"]
    assert claims["e1_e2_operational_recovery_increment"]["contrast"] == "E1_vs_E2"
    assert claims["e2_e3_corrective_memory_increment"]["contrast"] == "E2_vs_E3"


@pytest.mark.parametrize("name", ["protocol.yaml", "protocol_final.yaml"])
def test_protocol_binds_exact_claim_registry(name: str) -> None:
    protocol = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/eval/table2" / name).read_text(encoding="utf-8")
    )
    validate_frozen_protocol_mapping(protocol)
    protocol["paper_claims"]["pilot_resolution_forbidden"] = False
    with pytest.raises(ValueError, match="paper_claims"):
        validate_frozen_protocol_mapping(protocol)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["claims"].reverse(), "claim contract differs"),
        (
            lambda value: value["evidence_catalog"][0].update(
                {"relative_path": "paper_claim_evidence/assessments/junk.json"}
            ),
            "evidence rule differs",
        ),
        (
            lambda value: value["claim_status_semantics"].update(
                {"NOT_SUPPORTED": "no positive result"}
            ),
            "status semantics differ",
        ),
        (
            lambda value: value["claims"][-2].update(
                {"decision_rule": "caller_selected_status_v1"}
            ),
            "claim contract differs",
        ),
    ],
)
def test_registry_rejects_semantic_weakening(mutation, message: str) -> None:
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutation(value)
    with pytest.raises(SchemaError, match=message):
        validate_claim_registry_mapping(value)


def test_campaign_and_protocol_registry_copies_must_be_byte_identical(tmp_path: Path) -> None:
    staged = tmp_path / "paper_claim_registry.json"
    staged.write_bytes(REGISTRY.read_bytes())
    assert validate_claim_registry_binding(
        campaign_registry_path=staged,
        protocol_registry_path=REGISTRY,
        protocol_registry_id="table2-research-locked-claims-v1",
    )["registry_sha256"] == sha256_file(REGISTRY)
    value = json.loads(staged.read_text(encoding="utf-8"))
    value["claims"][0]["interpretation"] += " altered"
    _write_json(staged, value)
    with pytest.raises(SchemaError):
        validate_claim_registry_binding(
            campaign_registry_path=staged,
            protocol_registry_path=REGISTRY,
            protocol_registry_id="table2-research-locked-claims-v1",
        )


def test_initial_pilot_report_is_nr_and_hash_bound(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=False)
    report = build_initial_claim_report(
        registry_path=REGISTRY, campaign_manifest_path=root / "campaign_manifest.json"
    )
    assert report["paper_table_status"] == "N/R"
    assert {row["status"] for row in report["claims"]} == {"N/R"}
    assert all(not row["evidence"] and not row["unmet_conditions"] for row in report["claims"])
    path = _write_json(root / "paper_claim_report.json", report)
    assert validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root) == report
    with pytest.raises(SchemaError, match="paper-claim readiness"):
        _validate_claim_readiness(path, root)


@pytest.mark.parametrize(
    "publication_status",
    ("N/R", "DRAFT_PILOT_ONLY", "PILOT_ONLY"),
)
def test_locked_final_report_cannot_use_nonfinal_pilot_claim_path(
    tmp_path: Path,
    publication_status: str,
) -> None:
    root, campaign = _campaign(tmp_path, ready=True)
    campaign["publication_status"] = publication_status
    _write_json(root / "campaign_manifest.json", campaign)
    report = build_initial_claim_report(
        registry_path=REGISTRY,
        campaign_manifest_path=root / "campaign_manifest.json",
    )
    path = _write_json(root / "paper_claim_report.json", report)

    with pytest.raises(SchemaError, match="only a pilot campaign may use a non-final"):
        validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root)


@pytest.mark.parametrize(
    ("campaign_kind", "evidence_label"),
    (
        ("unknown", "PILOT_ONLY"),
        ("engineering_pilot", "FINAL_LOCKED"),
        ("locked_final", "PILOT_ONLY"),
    ),
)
def test_paper_claim_authority_rejects_unknown_or_mixed_campaign_profile(
    tmp_path: Path,
    campaign_kind: str,
    evidence_label: str,
) -> None:
    root, campaign = _campaign(tmp_path, ready=False)
    campaign["campaign_kind"] = campaign_kind
    campaign["evidence_label"] = evidence_label
    _write_json(root / "campaign_manifest.json", campaign)

    with pytest.raises(SchemaError, match="profile is not registered"):
        build_initial_claim_report(
            registry_path=REGISTRY,
            campaign_manifest_path=root / "campaign_manifest.json",
        )


def test_nonfinal_report_cannot_resolve_or_attach_evidence(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=False)
    report = build_initial_claim_report(
        registry_path=REGISTRY, campaign_manifest_path=root / "campaign_manifest.json"
    )
    report["claims"][0]["status"] = "SUPPORTED"
    report["claims"][0]["conclusion"], report["claims"][0]["limitation"] = STATUS_TEXT[
        "SUPPORTED"
    ]
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="non-final campaign resolves"):
        validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root)


def test_final_report_requires_independent_campaign_validation_then_claim_gate(
    tmp_path: Path,
) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="independently recomputed"):
        validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root)
    assert _validate_ready(path, root) == report
    with pytest.raises(SchemaError, match="unresolved claims"):
        _validate_claim_readiness(path, root)


def test_saved_final_authority_must_equal_recomputed_validation(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    path = _write_json(root / "paper_claim_report.json", report)
    authority = json.loads(
        (root / "aggregate/validation_report.json").read_text(encoding="utf-8")
    )
    recomputed = {**authority, "counts": {"included_blocks": 2}}
    validation = SimpleNamespace(
        passed=True,
        publication_status="READY_FOR_TABLE2",
        errors=[],
        to_dict=lambda: recomputed,
    )
    with patch(
        "web_agent.eval.table2.package_validator.validate_campaign",
        return_value=validation,
    ), pytest.raises(SchemaError, match="differs from independently recomputed"):
        validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root)


@pytest.mark.parametrize("status", ["SUPPORTED", "NEGATIVE_OR_NULL"])
def test_caller_cannot_choose_effect_status_without_frozen_decision_rule(
    tmp_path: Path, status: str
) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    claim = report["claims"][0]
    claim["status"] = status
    claim["conclusion"], claim["limitation"] = STATUS_TEXT[status]
    _attach_registered_evidence(root, claim, 0)
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="claim-specific decision rule"):
        _validate_ready(path, root)
    with pytest.raises(SchemaError, match="claim-specific decision rule"):
        _validate_claim_readiness(path, root)


def test_nr_status_cannot_carry_a_positive_conclusion(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    report["claims"][0]["conclusion"] = "The trained system improved success."
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="controlled vocabulary"):
        _validate_ready(path, root)


def test_arbitrary_json_cannot_satisfy_registered_evidence_kind(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    claim = report["claims"][-2]
    claim["status"] = "SUPPORTED"
    claim["conclusion"], claim["limitation"] = STATUS_TEXT["SUPPORTED"]
    _attach_registered_evidence(root, claim, len(CLAIM_RULES) - 2)
    row = claim["evidence"][0]
    junk = _write_json(root / row["relative_path"], {"kind": row["kind"], "fixture": True})
    row["sha256"] = sha256_file(junk)
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="keys differ"):
        _validate_ready(path, root)


def test_one_assessment_file_cannot_be_reused_or_relabelled(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    cost = report["claims"][-2]
    cost["status"] = "SUPPORTED"
    cost["conclusion"], cost["limitation"] = STATUS_TEXT["SUPPORTED"]
    _attach_registered_evidence(root, cost, len(CLAIM_RULES) - 2)
    paired = next(row for row in cost["evidence"] if row["kind"] == "browser_action_counts")
    ci = next(
        row for row in cost["evidence"] if row["kind"] == "model_call_counts"
    )
    ci.update({"relative_path": paired["relative_path"], "sha256": paired["sha256"]})
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="path differs from catalog"):
        _validate_ready(path, root)

    report = _ready_report(root)
    cost = report["claims"][-2]
    cost["status"] = "SUPPORTED"
    cost["conclusion"], cost["limitation"] = STATUS_TEXT["SUPPORTED"]
    _attach_registered_evidence(root, cost, len(CLAIM_RULES) - 2)
    paired = next(row for row in cost["evidence"] if row["kind"] == "browser_action_counts")
    ci = next(
        row for row in cost["evidence"] if row["kind"] == "model_call_counts"
    )
    target = root / ci["relative_path"]
    target.write_bytes((root / paired["relative_path"]).read_bytes())
    ci["sha256"] = sha256_file(target)
    _write_json(path, report)
    with pytest.raises(SchemaError, match="identity differs"):
        _validate_ready(path, root)


def test_evidence_source_hash_and_content_are_replayed(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    control_path = root / "aggregate/architecture_interface_matched_control.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control.update(
        {
            "architecture_matched": False,
            "interface_matched": False,
            "assessment": "DECLARATIVE_ONLY_NOT_SOURCE_ATTESTED",
        }
    )
    _write_json(control_path, control)
    control_evidence = _assessment(
        root,
        "architecture_interface_matched_control",
    )
    claim = report["claims"][1]
    claim["status"] = "NOT_SUPPORTED"
    claim["conclusion"], claim["limitation"] = STATUS_TEXT["NOT_SUPPORTED"]
    _attach_registered_evidence(root, claim, 1)
    claim["unmet_conditions"] = [
        {
            "condition": "architecture_interface_matched_control",
            "finding": "architecture_interface_matched_control:UNMET",
            "assessment_evidence": control_evidence,
        }
    ]
    source = root / "aggregate/statistics.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    del value["contrasts"]["E2_minus_E1"]
    _write_json(source, value)
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="source SHA-256 mismatch"):
        _validate_ready(path, root)
    with pytest.raises(SchemaError, match="lacks a registered contrast"):
        build_claim_evidence_assessment(
            kind="paired_task_effect",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_conditional_claim_remains_nr_until_decision_rule_is_registered(
    tmp_path: Path,
) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    control_path = root / "aggregate/architecture_interface_matched_control.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["architecture_matched"] = False
    control["interface_matched"] = False
    control["assessment"] = "DECLARATIVE_ONLY_NOT_SOURCE_ATTESTED"
    _write_json(control_path, control)
    evidence_row = _assessment(
        root,
        "architecture_interface_matched_control",
    )
    claim = report["claims"][1]
    claim["status"] = "NOT_SUPPORTED"
    claim["conclusion"], claim["limitation"] = STATUS_TEXT["NOT_SUPPORTED"]
    _attach_registered_evidence(root, claim, 1)
    claim["unmet_conditions"] = [
        {
            "condition": "architecture_interface_matched_control",
            "finding": "architecture_interface_matched_control:UNMET",
            "assessment_evidence": evidence_row,
        }
    ]
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="must remain N/R"):
        _validate_ready(path, root)


def test_not_supported_is_not_a_synonym_for_negative_or_null(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    primary = report["claims"][2]
    primary["status"] = "NOT_SUPPORTED"
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="invalid status"):
        _validate_ready(path, root)

    report = _ready_report(root)
    primary = report["claims"][2]
    primary["status"] = "NEGATIVE_OR_NULL"
    primary["conclusion"], primary["limitation"] = STATUS_TEXT["NEGATIVE_OR_NULL"]
    _attach_registered_evidence(root, primary, 2)
    _write_json(path, report)
    with pytest.raises(SchemaError, match="claim-specific decision rule"):
        _validate_ready(path, root)


def test_bare_control_and_ablation_assertions_remain_unmet(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root, condition_met=True)
    for kind in (
        "architecture_interface_matched_control",
        "learned_trigger_only_ablation",
    ):
        assessment = build_claim_evidence_assessment(
            kind=kind,
            registry_path=REGISTRY,
            campaign_dir=root,
        )
        assert assessment["assessment_status"] == "UNMET"
        assert assessment["finding"] == f"{kind}:UNMET"


def test_final_draft_remains_nr_and_prohibited_claims_remain_unclaimed(
    tmp_path: Path,
) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    report["claims"][-1]["status"] = "N/R"
    path = _write_json(root / "paper_claim_report.json", report)
    assert _validate_ready(path, root) == report
    with pytest.raises(SchemaError, match="unresolved claims"):
        _validate_claim_readiness(path, root)

    report = _ready_report(root)
    report["prohibited_claims"][0]["status"] = "CLAIMED"
    _write_json(path, report)
    with pytest.raises(SchemaError, match="prohibition was altered or claimed"):
        _validate_ready(path, root)


def test_claim_evidence_writer_uses_canonical_path_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    output = write_claim_evidence_assessment(
        kind="final_package_validation",
        registry_path=REGISTRY,
        campaign_dir=root,
    )
    assert output.relative_to(root).as_posix() == (
        "paper_claim_evidence/assessments/final_package_validation.json"
    )
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(SchemaError, match="already exists"):
        write_claim_evidence_assessment(
            kind="final_package_validation",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_initial_report_writer_refuses_overwrite(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=False)
    output = root / "paper_claim_report.json"
    write_initial_claim_report(
        output,
        registry_path=REGISTRY,
        campaign_manifest_path=root / "campaign_manifest.json",
    )
    with pytest.raises(SchemaError, match="already exists"):
        write_initial_claim_report(
            output,
            registry_path=REGISTRY,
            campaign_manifest_path=root / "campaign_manifest.json",
        )


def test_total_system_statistics_are_required_and_outside_holm(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    statistics_path = root / "aggregate/statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    assert statistics["multiple_testing"]["contrast_names"] == [
        "E1_minus_E0",
        "E2_minus_E1",
        "E3_minus_E2",
    ]
    assert "holm_adjusted_p" not in statistics["contrasts"]["E3_minus_E0"][
        "task_success"
    ]["task_clustered_significance"]
    del statistics["contrasts"]["E3_minus_E0"]
    _write_json(statistics_path, statistics)
    with pytest.raises(SchemaError, match="lacks a registered contrast"):
        build_claim_evidence_assessment(
            kind="paired_task_effect",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_metric_evidence_requires_finite_consistent_summaries(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    metrics_path = root / "aggregate/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["systems"]["E0"]["steps"]["all"]["estimate"] = None
    _write_json(metrics_path, metrics)
    with pytest.raises(SchemaError, match="finite number"):
        build_claim_evidence_assessment(
            kind="browser_action_counts",
            registry_path=REGISTRY,
            campaign_dir=root,
        )

    metrics["systems"]["E0"]["steps"]["all"] = {
        "estimate": 0.75,
        "numerator": 1,
        "denominator": 2,
    }
    _write_json(metrics_path, metrics)
    with pytest.raises(SchemaError, match="differs from numerator/denominator"):
        build_claim_evidence_assessment(
            kind="browser_action_counts",
            registry_path=REGISTRY,
            campaign_dir=root,
        )

    metrics["systems"]["E0"]["steps"]["all"] = {
        "estimate": 0.75,
        "numerator": 1.5,
        "denominator": 2,
    }
    _write_json(metrics_path, metrics)
    with pytest.raises(SchemaError, match="exact count"):
        build_claim_evidence_assessment(
            kind="browser_action_counts",
            registry_path=REGISTRY,
            campaign_dir=root,
        )

def test_nonstandard_nan_json_and_free_finding_are_rejected(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    with pytest.raises(SchemaError, match="machine-controlled"):
        build_claim_evidence_assessment(
            kind="browser_action_counts",
            registry_path=REGISTRY,
            campaign_dir=root,
            finding="The costs prove superiority.",
        )

    metrics_path = root / "aggregate/metrics.json"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8").replace('"estimate": 0.5', '"estimate": NaN', 1),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="strict UTF-8 JSON"):
        build_claim_evidence_assessment(
            kind="browser_action_counts",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_nr_limitation_cannot_smuggle_positive_prose(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=False)
    report = build_initial_claim_report(
        registry_path=REGISTRY,
        campaign_manifest_path=root / "campaign_manifest.json",
    )
    report["claims"][0]["limitation"] = "The trained system significantly improves E0."
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(SchemaError, match="controlled vocabulary"):
        validate_claim_report(path, registry_path=REGISTRY, campaign_dir=root)


def test_disclosure_is_bound_to_statistics_and_report_statuses(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    disclosure = report["claims"][-1]
    disclosure["status"] = "SUPPORTED"
    disclosure["conclusion"], disclosure["limitation"] = STATUS_TEXT["SUPPORTED"]
    _attach_registered_evidence(root, disclosure, len(CLAIM_RULES) - 1)
    path = _write_json(root / "paper_claim_report.json", report)
    assert _validate_ready(path, root) == report

    source_path = root / "paper_claim_evidence/disclosures.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["registered_contrasts_disclosed"]["E0_vs_E3"]["estimate"] = 999.0
    _write_json(source_path, source)
    with pytest.raises(SchemaError, match="source SHA-256 mismatch"):
        _validate_ready(path, root)

    report = _ready_report(root)
    cost = report["claims"][-2]
    cost["status"] = "SUPPORTED"
    cost["conclusion"], cost["limitation"] = STATUS_TEXT["SUPPORTED"]
    _attach_registered_evidence(root, cost, len(CLAIM_RULES) - 2)
    disclosure = report["claims"][-1]
    disclosure["status"] = "SUPPORTED"
    disclosure["conclusion"], disclosure["limitation"] = STATUS_TEXT["SUPPORTED"]
    _attach_registered_evidence(root, disclosure, len(CLAIM_RULES) - 1)
    _write_json(path, report)
    with pytest.raises(SchemaError, match="claim statuses differ"):
        _validate_ready(path, root)


def test_disclosure_rejects_placeholder_or_positive_limitation_text(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    source_path = root / "paper_claim_evidence/disclosures.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["pillar_limitations"]["P4"] = "none"
    _write_json(source_path, source)
    with pytest.raises(SchemaError, match="pillar limitations differ"):
        build_claim_evidence_assessment(
            kind="pillar_limitations_disclosure",
            registry_path=REGISTRY,
            campaign_dir=root,
        )

    _seed_canonical_sources(root)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["claim_outcomes_disclosed"][DISCLOSED_CLAIM_IDS[0]]["limitation"] = (
        "THE_SYSTEM_SIGNIFICANTLY_IMPROVES_SUCCESS"
    )
    _write_json(source_path, source)
    with pytest.raises(SchemaError, match="differs from controlled status"):
        build_claim_evidence_assessment(
            kind="complete_contrast_disclosure",
            registry_path=REGISTRY,
            campaign_dir=root,
        )

    _seed_canonical_sources(root)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["model_seed_scope"] = "seed_42_proves_model_seed_stability"
    _write_json(source_path, source)
    with pytest.raises(SchemaError, match="model-seed limitation"):
        build_claim_evidence_assessment(
            kind="model_seed_uncertainty_limitation",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_bare_companion_provenance_booleans_are_not_authority(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    for pillar in range(1, 5):
        _write_json(root / f"component_test/pillar{pillar}/package_manifest.json", {})
    with pytest.raises(SchemaError, match="keys differ"):
        build_claim_evidence_assessment(
            kind="authoritative_companion_input_provenance",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_bare_checkpoint_execution_booleans_are_not_authority(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    _write_json(
        root / "runtime_readiness/pc01_checkpoint_compatibility_receipt.json",
        {
            "checkpoint_sha256": "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a",
            "checkpoint_forward_executed": True,
            "exact_p4_tensor_consumed": True,
        },
    )
    with pytest.raises(Exception, match="receipt"):
        build_claim_evidence_assessment(
            kind="checkpoint_backed_embedding_execution_evidence",
            registry_path=REGISTRY,
            campaign_dir=root,
        )


def test_evidence_writer_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    _seed_canonical_sources(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "paper_claim_evidence/assessments").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(SchemaError, match="parent (escapes|traverses)"):
        write_claim_evidence_assessment(
            kind="final_package_validation",
            registry_path=REGISTRY,
            campaign_dir=root,
        )
    assert not (outside / "final_package_validation.json").exists()


def test_old_caller_supplied_publication_status_is_not_accepted(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path, ready=True)
    report = _ready_report(root)
    path = _write_json(root / "paper_claim_report.json", report)
    with pytest.raises(TypeError, match="validated_publication_status"):
        validate_claim_report(
            path,
            registry_path=REGISTRY,
            campaign_dir=root,
            validated_publication_status="READY_FOR_TABLE2",  # type: ignore[call-arg]
        )
