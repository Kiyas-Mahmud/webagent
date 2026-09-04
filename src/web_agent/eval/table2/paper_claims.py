"""Frozen publication-claim contracts for the Table 2 evidence package.

This module does not infer whether a scientific claim is true.  It validates
that every permitted or prohibited claim was registered before evaluation and
that a later claim report is bound to the exact campaign artifacts it cites.
Pilot packages remain ``N/R`` regardless of their observed metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .common import (
    SchemaError,
    atomic_write_json,
    safe_relative_path,
    sha256_file,
)


REGISTRY_SCHEMA_VERSION = "table2-paper-claim-registry-v1"
REPORT_SCHEMA_VERSION = "table2-paper-claim-report-v2"
EVIDENCE_SCHEMA_VERSION = "table2-paper-claim-evidence-assessment-v1"
READINESS_SCHEMA_VERSION = "table2-paper-claim-readiness-v1"
REGISTRY_ID = "table2-research-locked-claims-v1"
SOURCE_REGISTRY_RELATIVE_PATH = Path(
    "configs/eval/table2/paper_claim_registry_v1.json"
)
FROZEN_REGISTRY_RELATIVE_PATH = Path("frozen/paper_claim_registry.json")
INITIAL_STATUS = "N/R"
FINAL_READY_STATUS = "READY_FOR_TABLE2"
CLAIM_READY_STATUS = "READY_FOR_PAPER_CLAIMS"
EVIDENCE_PRODUCER = "web_agent.eval.table2.paper_claims:evidence-assessment-v1"
PILOT_PUBLICATION_STATUSES = frozenset({"N/R", "DRAFT_PILOT_ONLY", "PILOT_ONLY"})
CONDITIONAL_STATUSES = ("N/R", "SUPPORTED", "NOT_SUPPORTED", "NEGATIVE_OR_NULL")
EFFECT_STATUSES = ("N/R", "SUPPORTED", "NEGATIVE_OR_NULL")
DISCLOSURE_STATUSES = ("N/R", "SUPPORTED")
UNREGISTERED_DECISION_RULE = "UNREGISTERED_BLOCKS_SUPPORTED_OR_NEGATIVE_OR_NULL"
EXECUTABLE_DECISION_RULES = frozenset(
    {
        "machine_validated_complete_cost_accounting_v1",
        "machine_validated_complete_disclosure_v1",
    }
)
NR_CONCLUSION = "NOT_EVALUATED"
NR_LIMITATION = "REQUIRED_FINAL_EVIDENCE_NOT_VALIDATED"
STATUS_TEXT = {
    "N/R": (NR_CONCLUSION, NR_LIMITATION),
    "SUPPORTED": (
        "REGISTERED_CLAIM_SUPPORTED",
        "REGISTERED_EVIDENCE_AND_DECISION_RULE_SATISFIED",
    ),
    "NOT_SUPPORTED": (
        "REGISTERED_CLAIM_NOT_TESTABLE",
        "REGISTERED_IDENTIFICATION_OR_PROVENANCE_CONDITION_UNMET",
    ),
    "NEGATIVE_OR_NULL": (
        "REGISTERED_EFFECT_NEGATIVE_OR_NULL",
        "REGISTERED_EFFECT_DID_NOT_MEET_SUPPORT_CRITERION",
    ),
}
REGISTERED_PILLAR_LIMITATIONS = {
    "P1": "OPERATIONAL_RECOVERY_REQUIRES_PAIRED_E1_E2_EVIDENCE",
    "P2": "COMPANION_DIAGNOSTIC_IS_NOT_A_TRAINING_ABLATION",
    "P3": "COMPANION_DIAGNOSTIC_DOES_NOT_MEASURE_BROWSER_EXECUTION",
    "P4": "MEMORY_BENEFIT_REQUIRES_PAIRED_E2_E3_EVIDENCE",
}
DISCLOSURE_OUTCOME_KEYS = {"status", "conclusion", "limitation"}


@dataclass(frozen=True, slots=True)
class ClaimRule:
    claim_id: str
    family: str
    contrast: str | None
    interpretation: str
    required_evidence: tuple[str, ...]
    conditions: tuple[str, ...] = ()
    permitted_statuses: tuple[str, ...] = EFFECT_STATUSES
    decision_rule: str = UNREGISTERED_DECISION_RULE


@dataclass(frozen=True, slots=True)
class EvidenceRule:
    kind: str
    validator: str
    source_relative_paths: tuple[str, ...]

    @property
    def relative_path(self) -> str:
        return f"paper_claim_evidence/assessments/{self.kind}.json"


@dataclass(frozen=True, slots=True)
class ProhibitedClaim:
    claim_id: str
    prohibition: str
    forbidden_substitutions: tuple[str, ...]


CLAIM_RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        claim_id="e0_e1_contextual_policy_difference",
        family="primary_contrast",
        contrast="E0_vs_E1",
        interpretation=(
            "trained multimodal policy versus the unadapted common-schema baseline; "
            "contextual unless architecture and interface matching are proven"
        ),
        required_evidence=(
            "final_package_validation",
            "paired_task_effect",
            "task_cluster_confidence_interval",
            "cost_accounting",
        ),
    ),
    ClaimRule(
        claim_id="e0_e1_pure_training_effect",
        family="conditional_causal_claim",
        contrast="E0_vs_E1",
        interpretation="pure training effect under an architecture/interface-matched control",
        required_evidence=(
            "final_package_validation",
            "paired_task_effect",
            "task_cluster_confidence_interval",
            "architecture_interface_matched_control",
        ),
        conditions=("architecture_interface_matched_control",),
        permitted_statuses=CONDITIONAL_STATUSES,
    ),
    ClaimRule(
        claim_id="e1_e2_operational_recovery_increment",
        family="primary_contrast",
        contrast="E1_vs_E2",
        interpretation="increment from executed diagnosis and bounded concrete recovery",
        required_evidence=(
            "final_package_validation",
            "paired_task_effect",
            "task_cluster_confidence_interval",
            "verified_failure_recovery_evidence",
            "cost_accounting",
        ),
    ),
    ClaimRule(
        claim_id="e2_e3_corrective_memory_increment",
        family="primary_contrast",
        contrast="E2_vs_E3",
        interpretation="increment from frozen train-only corrective-memory intervention",
        required_evidence=(
            "final_package_validation",
            "paired_task_effect",
            "task_cluster_confidence_interval",
            "frozen_train_only_memory_evidence",
            "paired_memory_intervention_trace",
            "cost_accounting",
        ),
    ),
    ClaimRule(
        claim_id="e0_e3_total_system_difference",
        family="total_system_contrast",
        contrast="E0_vs_E3",
        interpretation="total-system difference only; not a single-pillar attribution",
        required_evidence=(
            "final_package_validation",
            "paired_task_effect",
            "task_cluster_confidence_interval",
            "cost_accounting",
        ),
    ),
    ClaimRule(
        claim_id="p1_learned_trigger_specific_contribution",
        family="conditional_pillar_claim",
        contrast="E1_vs_E2",
        interpretation=(
            "learned P1 trigger contribution separated from executor and loop safeguards"
        ),
        required_evidence=(
            "final_package_validation",
            "trigger_source_stratified_recovery",
            "learned_trigger_only_ablation",
        ),
        conditions=("learned_trigger_only_ablation",),
        permitted_statuses=CONDITIONAL_STATUSES,
    ),
    ClaimRule(
        claim_id="p1_offline_diagnostic_support",
        family="companion_diagnostic",
        contrast=None,
        interpretation=(
            "offline failure detection, failure diagnosis, and executed-transition "
            "recovery-assessment support; not an operational recovery-effect claim"
        ),
        required_evidence=(
            "p1_failure_detection_diagnostics",
            "p1_failure_diagnosis_diagnostics",
            "p1_executed_recovery_assessment_diagnostics",
            "authoritative_companion_input_provenance",
        ),
        conditions=("authoritative_companion_input_provenance",),
        permitted_statuses=CONDITIONAL_STATUSES,
    ),
    ClaimRule(
        claim_id="p2_temporal_multimodal_diagnostic_support",
        family="companion_diagnostic",
        contrast=None,
        interpretation="temporal and modality sensitivity support from registered causal controls",
        required_evidence=(
            "p2_temporal_causal_diagnostics",
            "p2_modality_occlusion_diagnostics",
            "authoritative_companion_input_provenance",
        ),
        conditions=("authoritative_companion_input_provenance",),
        permitted_statuses=CONDITIONAL_STATUSES,
    ),
    ClaimRule(
        claim_id="p3_grounded_action_parameter_diagnostic_support",
        family="companion_diagnostic",
        contrast=None,
        interpretation=(
            "offline six-class action, grounding, parameter and invalid-output "
            "diagnostics; browser execution is not measured"
        ),
        required_evidence=(
            "p3_action_class_diagnostics",
            "p3_grounding_parameter_diagnostics",
            "p3_invalid_output_accounting",
            "authoritative_companion_input_provenance",
        ),
        conditions=("authoritative_companion_input_provenance",),
        permitted_statuses=CONDITIONAL_STATUSES,
    ),
    ClaimRule(
        claim_id="p4_retrieval_admission_diagnostic_support",
        family="companion_diagnostic",
        contrast=None,
        interpretation=(
            "frozen train-only retrieval, admission, provenance, and no-write "
            "mechanism support; not an end-to-end memory-benefit claim"
        ),
        required_evidence=(
            "p4_retrieval_admission_diagnostics",
            "p4_store_provenance_integrity",
            "p4_no_write_integrity",
            "authoritative_companion_input_provenance",
            "checkpoint_backed_embedding_execution_evidence",
        ),
        conditions=(
            "authoritative_companion_input_provenance",
            "checkpoint_backed_embedding_execution_evidence",
        ),
        permitted_statuses=CONDITIONAL_STATUSES,
    ),
    ClaimRule(
        claim_id="cost_and_latency_tradeoff",
        family="required_cost_disclosure",
        contrast="E0_E1_E2_E3",
        interpretation="visible browser-action, model-call and wall-clock costs",
        required_evidence=(
            "browser_action_counts",
            "model_call_counts",
            "wall_clock_latency",
        ),
        permitted_statuses=DISCLOSURE_STATUSES,
        decision_rule="machine_validated_complete_cost_accounting_v1",
    ),
    ClaimRule(
        claim_id="negative_null_and_unsupported_results_disclosed",
        family="required_complete_disclosure",
        contrast="ALL_REGISTERED_CONTRASTS",
        interpretation="all negative, null and unsupported results and limitations are disclosed",
        required_evidence=(
            "complete_contrast_disclosure",
            "pillar_limitations_disclosure",
            "model_seed_uncertainty_limitation",
        ),
        permitted_statuses=DISCLOSURE_STATUSES,
        decision_rule="machine_validated_complete_disclosure_v1",
    ),
)

DISCLOSED_CLAIM_IDS = tuple(
    rule.claim_id
    for rule in CLAIM_RULES
    if rule.claim_id != "negative_null_and_unsupported_results_disclosed"
)


# Every evidence kind has one unique assessment path and an exact set of
# campaign-relative source artifacts.  An assessment is not authority by
# itself: its status is recomputed from these source bytes during validation.
EVIDENCE_RULES: tuple[EvidenceRule, ...] = (
    EvidenceRule(
        "final_package_validation",
        "final_package_validation_v1",
        ("aggregate/validation_report.json",),
    ),
    EvidenceRule(
        "paired_task_effect",
        "paired_task_effect_v1",
        ("aggregate/statistics.json",),
    ),
    EvidenceRule(
        "task_cluster_confidence_interval",
        "task_cluster_confidence_interval_v1",
        ("aggregate/statistics.json",),
    ),
    EvidenceRule(
        "cost_accounting",
        "cost_accounting_v1",
        ("aggregate/metrics.json",),
    ),
    EvidenceRule(
        "architecture_interface_matched_control",
        "architecture_interface_matched_control_v1",
        ("aggregate/architecture_interface_matched_control.json",),
    ),
    EvidenceRule(
        "verified_failure_recovery_evidence",
        "verified_failure_recovery_evidence_v1",
        (
            "aggregate/metrics.json",
            "aggregate/recovery_diagnostic_metrics.json",
        ),
    ),
    EvidenceRule(
        "frozen_train_only_memory_evidence",
        "frozen_train_only_memory_evidence_v1",
        ("memory/seed_42/manifest.json",),
    ),
    EvidenceRule(
        "paired_memory_intervention_trace",
        "paired_memory_intervention_trace_v1",
        ("aggregate/retrieval_diagnostics.json",),
    ),
    EvidenceRule(
        "trigger_source_stratified_recovery",
        "trigger_source_stratified_recovery_v1",
        ("aggregate/trigger_source_stratified_recovery.json",),
    ),
    EvidenceRule(
        "learned_trigger_only_ablation",
        "learned_trigger_only_ablation_v1",
        ("component_test/pillar1/learned_trigger_only_ablation.json",),
    ),
    EvidenceRule(
        "p1_failure_detection_diagnostics",
        "p1_failure_detection_diagnostics_v1",
        ("component_test/pillar1/pillar1_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p1_failure_diagnosis_diagnostics",
        "p1_failure_diagnosis_diagnostics_v1",
        ("component_test/pillar1/pillar1_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p1_executed_recovery_assessment_diagnostics",
        "p1_executed_recovery_assessment_diagnostics_v1",
        ("component_test/pillar1/pillar1_diagnostic_report.json",),
    ),
    EvidenceRule(
        "authoritative_companion_input_provenance",
        "authoritative_companion_input_provenance_v1",
        (
            "component_test/companion_input_provenance.json",
            "component_test/pillar1/package_manifest.json",
            "component_test/pillar2/package_manifest.json",
            "component_test/pillar3/package_manifest.json",
            "component_test/pillar4/package_manifest.json",
        ),
    ),
    EvidenceRule(
        "p2_temporal_causal_diagnostics",
        "p2_temporal_causal_diagnostics_v1",
        ("component_test/pillar2/pillar2_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p2_modality_occlusion_diagnostics",
        "p2_modality_occlusion_diagnostics_v1",
        ("component_test/pillar2/pillar2_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p3_action_class_diagnostics",
        "p3_action_class_diagnostics_v1",
        ("component_test/pillar3/pillar3_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p3_grounding_parameter_diagnostics",
        "p3_grounding_parameter_diagnostics_v1",
        ("component_test/pillar3/pillar3_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p3_invalid_output_accounting",
        "p3_invalid_output_accounting_v1",
        ("component_test/pillar3/pillar3_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p4_retrieval_admission_diagnostics",
        "p4_retrieval_admission_diagnostics_v1",
        ("component_test/pillar4/pillar4_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p4_store_provenance_integrity",
        "p4_store_provenance_integrity_v1",
        ("component_test/pillar4/pillar4_diagnostic_report.json",),
    ),
    EvidenceRule(
        "p4_no_write_integrity",
        "p4_no_write_integrity_v1",
        ("component_test/pillar4/pillar4_diagnostic_report.json",),
    ),
    EvidenceRule(
        "checkpoint_backed_embedding_execution_evidence",
        "checkpoint_backed_embedding_execution_evidence_v1",
        ("runtime_readiness/pc01_checkpoint_compatibility_receipt.json",),
    ),
    EvidenceRule(
        "browser_action_counts",
        "browser_action_counts_v1",
        ("aggregate/metrics.json",),
    ),
    EvidenceRule(
        "model_call_counts",
        "model_call_counts_v1",
        ("aggregate/metrics.json",),
    ),
    EvidenceRule(
        "wall_clock_latency",
        "wall_clock_latency_v1",
        ("aggregate/metrics.json",),
    ),
    EvidenceRule(
        "complete_contrast_disclosure",
        "complete_contrast_disclosure_v1",
        (
            "paper_claim_evidence/disclosures.json",
            "aggregate/statistics.json",
        ),
    ),
    EvidenceRule(
        "pillar_limitations_disclosure",
        "pillar_limitations_disclosure_v1",
        ("paper_claim_evidence/disclosures.json",),
    ),
    EvidenceRule(
        "model_seed_uncertainty_limitation",
        "model_seed_uncertainty_limitation_v1",
        ("paper_claim_evidence/disclosures.json",),
    ),
)

EVIDENCE_RULE_BY_KIND = {rule.kind: rule for rule in EVIDENCE_RULES}
if len(EVIDENCE_RULE_BY_KIND) != len(EVIDENCE_RULES):  # pragma: no cover
    raise RuntimeError("duplicate registered paper-claim evidence kind")


PROHIBITED_CLAIMS: tuple[ProhibitedClaim, ...] = (
    ProhibitedClaim(
        "offline_mcc_proves_operational_recovery",
        "Do not claim operational recovery from offline recovery-outcome MCC alone.",
        ("offline_recovery_outcome_mcc",),
    ),
    ProhibitedClaim(
        "retrieval_metric_alone_proves_memory_benefit",
        "Do not claim memory benefit from memory-update classification or Recall@K alone.",
        ("memory_update_classification", "recall_at_k", "mrr"),
    ),
    ProhibitedClaim(
        "screenshots_alone_prove_causal_reasoning",
        "Do not claim causal reasoning from screenshots without registered causal controls.",
        ("pre_post_screenshot_observation",),
    ),
    ProhibitedClaim(
        "cross_benchmark_superiority",
        "Do not claim superiority to work evaluated on another benchmark.",
        ("related_paper_score",),
    ),
    ProhibitedClaim(
        "all_recovery_strategies_generalize",
        "Do not claim all six recovery strategies generalize without locked-task support.",
        ("development_scenario_result",),
    ),
    ProhibitedClaim(
        "pilot_result_is_final_table2",
        "Do not present engineering-pilot values as final Table 2 evidence.",
        ("pilot_metric", "pilot_confidence_interval"),
    ),
    ProhibitedClaim(
        "single_seed_estimates_model_seed_uncertainty",
        "Do not claim model-seed uncertainty from the registered seed-42-only design.",
        ("task_cluster_bootstrap",),
    ),
)


REGISTRY_KEYS = {
    "schema_version",
    "registry_id",
    "status",
    "paper_table_status",
    "seed_scope",
    "claim_status_values",
    "claim_status_semantics",
    "publication_rules",
    "evidence_catalog",
    "claims",
    "prohibited_claims",
}
EVIDENCE_CATALOG_KEYS = {
    "kind",
    "relative_path",
    "schema_version",
    "producer",
    "validator",
    "source_relative_paths",
}
CLAIM_KEYS = {
    "claim_id",
    "family",
    "contrast",
    "interpretation",
    "initial_status",
    "permitted_statuses",
    "required_evidence",
    "conditions",
    "decision_rule",
}
PROHIBITED_KEYS = {
    "claim_id",
    "prohibition",
    "status",
    "forbidden_evidence_substitutions",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(map(str, value))
    if actual != expected:
        raise SchemaError(
            f"{context} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _string_list(value: Any, *, context: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{context} must be a string array")
    normalized = tuple(item.strip() for item in value)
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise SchemaError(f"{context} must contain unique nonempty values")
    if not allow_empty and not normalized:
        raise SchemaError(f"{context} must not be empty")
    return normalized


def validate_claim_registry_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact registered claim ontology and return a detached mapping."""

    if not isinstance(value, Mapping):
        raise SchemaError("paper claim registry must be a JSON object")
    _exact_keys(value, REGISTRY_KEYS, "paper claim registry")
    expected_header = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "status": "FROZEN_PRE_EVALUATION_TEMPLATE",
        "paper_table_status": INITIAL_STATUS,
        "seed_scope": "seed_42_only_task_uncertainty_not_model_seed_uncertainty",
    }
    for key, expected in expected_header.items():
        if value.get(key) != expected:
            raise SchemaError(f"paper claim registry {key} differs from registration")
    statuses = _string_list(value.get("claim_status_values"), context="claim_status_values")
    if statuses != ("N/R", "SUPPORTED", "NOT_SUPPORTED", "NEGATIVE_OR_NULL"):
        raise SchemaError("paper claim status order differs from registration")
    expected_semantics = {
        "N/R": "not evaluated; no evidence or conclusion may be attached",
        "SUPPORTED": (
            "all registered evidence and conditions are satisfied and the "
            "registered analysis supports the stated claim"
        ),
        "NOT_SUPPORTED": (
            "the claim is not scientifically testable because at least one "
            "registered identification or provenance condition is unmet"
        ),
        "NEGATIVE_OR_NULL": (
            "all registered evidence and conditions are satisfied, but the "
            "registered effect is negative, null, or does not meet the support criterion"
        ),
    }
    if value.get("claim_status_semantics") != expected_semantics:
        raise SchemaError("paper claim status semantics differ from registration")
    publication_rules = value.get("publication_rules")
    expected_publication_rules = {
        "pilot_claim_status": "N/R",
        "pilot_table_status": "N/R",
        "final_ready_status": FINAL_READY_STATUS,
        "paper_claim_ready_status": CLAIM_READY_STATUS,
        "ready_for_table2_does_not_imply_claim_readiness": True,
        "unregistered_decision_rules_block_claim_resolution": True,
        "final_claims_must_resolve": True,
        "negative_results_disclosure_required": True,
        "evidence_paths_must_be_campaign_relative_and_hash_bound": True,
    }
    if publication_rules != expected_publication_rules:
        raise SchemaError("paper claim publication rules differ from registration")

    catalog = value.get("evidence_catalog")
    if not isinstance(catalog, list) or len(catalog) != len(EVIDENCE_RULES):
        raise SchemaError("paper claim evidence catalog has the wrong entry count")
    catalog_paths: set[str] = set()
    for index, (row, rule) in enumerate(zip(catalog, EVIDENCE_RULES)):
        if not isinstance(row, Mapping):
            raise SchemaError(f"paper claim evidence catalog[{index}] must be an object")
        _exact_keys(row, EVIDENCE_CATALOG_KEYS, f"evidence catalog[{index}]")
        expected = {
            "kind": rule.kind,
            "relative_path": rule.relative_path,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "producer": EVIDENCE_PRODUCER,
            "validator": rule.validator,
            "source_relative_paths": list(rule.source_relative_paths),
        }
        if dict(row) != expected:
            raise SchemaError(f"paper claim evidence rule differs: {rule.kind}")
        if rule.relative_path in catalog_paths:
            raise SchemaError("paper claim evidence catalog reuses an assessment path")
        catalog_paths.add(rule.relative_path)

    registered_evidence = {kind for rule in CLAIM_RULES for kind in rule.required_evidence}
    if registered_evidence != set(EVIDENCE_RULE_BY_KIND):
        raise SchemaError("paper claim evidence catalog does not exactly cover claims")

    claims = value.get("claims")
    if not isinstance(claims, list) or len(claims) != len(CLAIM_RULES):
        raise SchemaError("paper claim registry has the wrong claim count")
    for index, (row, rule) in enumerate(zip(claims, CLAIM_RULES)):
        if not isinstance(row, Mapping):
            raise SchemaError(f"paper claim[{index}] must be an object")
        _exact_keys(row, CLAIM_KEYS, f"paper claim[{index}]")
        expected = {
            "claim_id": rule.claim_id,
            "family": rule.family,
            "contrast": rule.contrast,
            "interpretation": rule.interpretation,
            "initial_status": INITIAL_STATUS,
            "permitted_statuses": list(rule.permitted_statuses),
            "required_evidence": list(rule.required_evidence),
            "conditions": list(rule.conditions),
            "decision_rule": rule.decision_rule,
        }
        if dict(row) != expected:
            raise SchemaError(f"paper claim contract differs from registration: {rule.claim_id}")
        if (
            rule.decision_rule != UNREGISTERED_DECISION_RULE
            and rule.decision_rule not in EXECUTABLE_DECISION_RULES
        ):
            raise SchemaError(
                f"paper claim decision rule has no executable validator: {rule.decision_rule}"
            )

    prohibited = value.get("prohibited_claims")
    if not isinstance(prohibited, list) or len(prohibited) != len(PROHIBITED_CLAIMS):
        raise SchemaError("paper claim registry has the wrong prohibition count")
    for index, (row, rule) in enumerate(zip(prohibited, PROHIBITED_CLAIMS)):
        if not isinstance(row, Mapping):
            raise SchemaError(f"prohibited claim[{index}] must be an object")
        _exact_keys(row, PROHIBITED_KEYS, f"prohibited claim[{index}]")
        expected = {
            "claim_id": rule.claim_id,
            "prohibition": rule.prohibition,
            "status": "PROHIBITED",
            "forbidden_evidence_substitutions": list(rule.forbidden_substitutions),
        }
        if dict(row) != expected:
            raise SchemaError(
                f"prohibited claim contract differs from registration: {rule.claim_id}"
            )
    return dict(value)


def validate_claim_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or source.absolute() != source.resolve()
    ):
        raise SchemaError("paper claim registry must be a regular non-symlink file")
    return validate_claim_registry_mapping(
        _read_strict_json(source, context="paper claim registry")
    )


def validate_claim_registry_binding(
    *,
    campaign_registry_path: str | Path,
    protocol_registry_path: str | Path,
    protocol_registry_id: object,
) -> dict[str, str]:
    """Bind campaign and protocol locators to the same registered bytes.

    A handoff rewrites the campaign locator to its staged copy while the
    protocol retains the tracked source locator.  Therefore path text is not
    authority: both regular files must validate independently and have the
    exact same SHA-256 and registered ID.
    """

    campaign_source = Path(campaign_registry_path)
    protocol_source = Path(protocol_registry_path)
    campaign_registry = validate_claim_registry(campaign_source)
    protocol_registry = validate_claim_registry(protocol_source)
    if protocol_registry_id != REGISTRY_ID:
        raise SchemaError("protocol paper-claim registry ID differs from registration")
    if campaign_registry["registry_id"] != protocol_registry["registry_id"]:
        raise SchemaError("campaign/protocol paper-claim registry IDs differ")
    campaign_digest = sha256_file(campaign_source)
    if campaign_digest != sha256_file(protocol_source):
        raise SchemaError("campaign/protocol paper-claim registry bytes differ")
    return {"registry_id": REGISTRY_ID, "registry_sha256": campaign_digest}


def build_initial_claim_report(
    *,
    registry_path: str | Path,
    campaign_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build an explicitly unevaluated report; never infer a claim from metrics."""

    registry_input = Path(registry_path)
    campaign_input = Path(campaign_manifest_path)
    if campaign_input.is_symlink() or not campaign_input.is_file():
        raise SchemaError("campaign manifest must be a regular non-symlink file")
    registry = validate_claim_registry(registry_input)
    registry_source = registry_input.resolve()
    campaign_source = campaign_input.resolve()
    campaign = _read_strict_json(campaign_source, context="campaign manifest")
    campaign_id = str(campaign.get("campaign_id", "")).strip()
    if not campaign_id:
        raise SchemaError("campaign manifest lacks campaign_id")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "registry_sha256": sha256_file(registry_source),
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": sha256_file(campaign_source),
        "publication_authority_relative_path": "campaign_manifest.json",
        "publication_authority_sha256": sha256_file(campaign_source),
        "publication_status": str(campaign.get("publication_status", "N/R")),
        "paper_table_status": INITIAL_STATUS,
        "claims": [
            {
                "claim_id": row["claim_id"],
                "status": INITIAL_STATUS,
                "conclusion": NR_CONCLUSION,
                "limitation": NR_LIMITATION,
                "evidence": [],
                "unmet_conditions": [],
            }
            for row in registry["claims"]
        ],
        "prohibited_claims": [
            {"claim_id": row["claim_id"], "status": "NOT_CLAIMED"}
            for row in registry["prohibited_claims"]
        ],
    }


REPORT_KEYS = {
    "schema_version",
    "registry_id",
    "registry_sha256",
    "campaign_id",
    "campaign_manifest_sha256",
    "publication_authority_relative_path",
    "publication_authority_sha256",
    "publication_status",
    "paper_table_status",
    "claims",
    "prohibited_claims",
}
REPORT_CLAIM_KEYS = {
    "claim_id",
    "status",
    "conclusion",
    "limitation",
    "evidence",
    "unmet_conditions",
}
EVIDENCE_KEYS = {"kind", "relative_path", "sha256"}
EVIDENCE_ASSESSMENT_KEYS = {
    "schema_version",
    "producer",
    "evidence_kind",
    "campaign_id",
    "assessment_status",
    "finding",
    "sources",
}
EVIDENCE_SOURCE_KEYS = {"relative_path", "sha256"}
UNMET_CONDITION_KEYS = {"condition", "finding", "assessment_evidence"}
REPORT_PROHIBITED_KEYS = {"claim_id", "status"}


def _require_sha256(value: object, *, context: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SchemaError(f"{context} SHA-256 is malformed")
    return digest


def _campaign_file(root: Path, relative_text: str, *, context: str) -> Path:
    relative = safe_relative_path(relative_text)
    if not relative.parts or str(relative) in {"", "."}:
        raise SchemaError(f"{context} path is empty")
    candidate = root / relative
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise SchemaError(f"{context} escapes campaign")
    if candidate.absolute() != resolved or candidate.is_symlink() or not candidate.is_file():
        raise SchemaError(f"{context} is missing, symlinked, or traverses a symlink: {relative}")
    return resolved


def _read_strict_json(path: Path, *, context: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError(f"{context} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaError(f"{context} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{context} must be a JSON object")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be an object")
    return value


def _positive_int(value: object, *, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise SchemaError(f"{context} must be a positive exact integer")
    return value


def _nonnegative_int(value: object, *, context: str) -> int:
    if type(value) is not int or value < 0:
        raise SchemaError(f"{context} must be a nonnegative exact integer")
    return value


def _finite_number(
    value: object,
    *,
    context: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise SchemaError(f"{context} must be a finite nonnegative number")
    return result


def _metric_summary(
    value: object,
    *,
    context: str,
    integer_numerator: bool = False,
) -> Mapping[str, Any]:
    metric = _mapping(value, context=context)
    if not {"estimate", "numerator", "denominator"}.issubset(metric):
        raise SchemaError(f"{context} lacks estimate/numerator/denominator")
    estimate = _finite_number(metric.get("estimate"), context=f"{context}.estimate", nonnegative=True)
    numerator = _finite_number(
        metric.get("numerator"), context=f"{context}.numerator", nonnegative=True
    )
    # The aggregate producer serializes continuous totals as floats, including
    # totals of integer-valued step/model-call observations.  Require exact
    # integrality without imposing a JSON representation that the producer
    # itself does not use.
    if integer_numerator and not numerator.is_integer():
        raise SchemaError(f"{context}.numerator must be an exact count")
    denominator = _positive_int(metric.get("denominator"), context=f"{context}.denominator")
    expected = numerator / denominator
    if not math.isclose(estimate, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise SchemaError(f"{context} estimate differs from numerator/denominator")
    return metric


def _safe_output_file(root: Path, relative_text: str, *, context: str) -> Path:
    relative = safe_relative_path(relative_text)
    candidate = root / relative
    resolved_parent = candidate.parent.resolve()
    if root != resolved_parent and root not in resolved_parent.parents:
        raise SchemaError(f"{context} parent escapes campaign")
    if candidate.parent.absolute() != resolved_parent:
        raise SchemaError(f"{context} parent traverses a symlink")
    if candidate.is_symlink():
        raise SchemaError(f"{context} must not be a symlink")
    return candidate


def _validate_companion_report(
    value: Mapping[str, Any],
    *,
    campaign_id: str,
    pillar: str,
    schema_version: str,
    summary_keys: set[str],
) -> None:
    expected_role = f"{pillar}_COMPANION_MECHANISM_EVIDENCE"
    role = value.get("evidence_role")
    promotion = value.get("promotion_status")
    valid_promotion = (
        role == expected_role and promotion == "CANONICAL_PC01_COMPANION_EVIDENCE"
    ) or (
        role == "ENGINEERING_DIAGNOSTIC_ONLY_UNPROMOTABLE"
        and promotion == "UNPROMOTABLE_INPUT_PROVENANCE_BLOCKED"
    )
    if (
        value.get("schema_version") != schema_version
        or value.get("campaign_id") != campaign_id
        or value.get("paper_table_status") != "N/R"
        or value.get("affects_primary_table2") is not False
        or value.get("locked_test_rows_read") != 0
        or not valid_promotion
    ):
        raise SchemaError(f"{pillar} companion report is not registered evidence")
    records = value.get("records")
    summary = _mapping(value.get("summary"), context=f"{pillar} summary")
    if not isinstance(records, list) or not records or not summary_keys.issubset(summary):
        raise SchemaError(f"{pillar} companion report lacks registered diagnostic content")


def _registered_metric_systems(value: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    if value.get("schema_version") != "table2.v1" or value.get(
        "publication_status"
    ) != FINAL_READY_STATUS:
        raise SchemaError(f"{context} is not a final registered aggregate")
    systems = _mapping(value.get("systems"), context=f"{context}.systems")
    if set(systems) != {"E0", "E1", "E2", "E3"}:
        raise SchemaError(f"{context} does not contain the complete E0-E3 system set")
    return systems


STATISTICS_CONTRAST_BY_CLAIM_CONTRAST = {
    "E0_vs_E1": "E1_minus_E0",
    "E1_vs_E2": "E2_minus_E1",
    "E2_vs_E3": "E3_minus_E2",
    "E0_vs_E3": "E3_minus_E0",
}
ADJACENT_STATISTICS_CONTRASTS = (
    "E1_minus_E0",
    "E2_minus_E1",
    "E3_minus_E2",
)


def _validated_task_success_contrasts(
    value: Mapping[str, Any],
    *,
    require_interval: bool,
) -> Mapping[str, Any]:
    if (
        value.get("schema_version") != "table2.v1"
        or value.get("publication_status") != FINAL_READY_STATUS
    ):
        raise SchemaError("paired-statistics evidence is not final")
    contrasts = _mapping(value.get("contrasts"), context="statistics.contrasts")
    expected = set(STATISTICS_CONTRAST_BY_CLAIM_CONTRAST.values())
    if not expected.issubset(contrasts):
        raise SchemaError("paired-statistics evidence lacks a registered contrast")
    for contrast in expected:
        task_success = _mapping(
            _mapping(contrasts[contrast], context=f"statistics.{contrast}").get(
                "task_success"
            ),
            context=f"statistics.{contrast}.task_success",
        )
        estimate = _finite_number(
            task_success.get("estimate"), context=f"statistics.{contrast}.estimate"
        )
        if require_interval:
            low = _finite_number(
                task_success.get("ci_low"), context=f"statistics.{contrast}.ci_low"
            )
            high = _finite_number(
                task_success.get("ci_high"), context=f"statistics.{contrast}.ci_high"
            )
            _positive_int(
                task_success.get("n_clusters"),
                context=f"statistics.{contrast}.n_clusters",
            )
            if low > estimate or estimate > high:
                raise SchemaError(
                    f"statistics.{contrast} interval does not contain its estimate"
                )
        if contrast == "E3_minus_E0" and task_success.get("analysis_role") != (
            "descriptive_total_system_contrast"
        ):
            raise SchemaError("E3-minus-E0 statistics lost descriptive-only scope")
    multiple = _mapping(value.get("multiple_testing"), context="statistics.multiple_testing")
    if (
        multiple.get("family") != "registered_task_success_contrasts"
        or multiple.get("method") != "Holm"
        or multiple.get("hypothesis_count") != 3
        or multiple.get("contrast_names") != list(ADJACENT_STATISTICS_CONTRASTS)
    ):
        raise SchemaError("statistics Holm family differs from the three adjacent contrasts")
    total_significance = _mapping(
        _mapping(contrasts["E3_minus_E0"], context="statistics.E3_minus_E0")[
            "task_success"
        ].get("task_clustered_significance"),
        context="statistics.E3_minus_E0.task_clustered_significance",
    )
    if (
        total_significance.get("inference_role") != "descriptive_unadjusted_only"
        or "holm_adjusted_p" in total_significance
    ):
        raise SchemaError("E3-minus-E0 must remain outside the Holm family")
    return contrasts


def _validate_evidence_source_content(
    rule: EvidenceRule,
    sources: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str,
    claim_statuses: Mapping[str, str] | None = None,
) -> str:
    """Return the independently recomputed SATISFIED/UNMET assessment."""

    values = [
        _read_strict_json(Path(str(row["_resolved_path"])), context=f"{rule.kind} source")
        for row in sources
    ]
    value = values[0]
    validator = rule.validator

    if validator == "final_package_validation_v1":
        counts = value.get("counts")
        if (
            value.get("status") != "PASS"
            or value.get("publication_status") != FINAL_READY_STATUS
            or value.get("errors") not in ([], None)
            or not isinstance(counts, Mapping)
            or _positive_int(
                counts.get("included_blocks"), context="validation included_blocks"
            )
            <= 0
        ):
            raise SchemaError("final package validation evidence is not a passing final report")
    elif validator in {"paired_task_effect_v1", "task_cluster_confidence_interval_v1"}:
        _validated_task_success_contrasts(
            value,
            require_interval=validator == "task_cluster_confidence_interval_v1",
        )
    elif validator in {
        "cost_accounting_v1",
        "browser_action_counts_v1",
        "model_call_counts_v1",
        "wall_clock_latency_v1",
    }:
        systems = _registered_metric_systems(value, context="metric evidence")
        if validator == "browser_action_counts_v1":
            fields = {"step_count"}
        elif validator == "model_call_counts_v1":
            fields = {"model_call_count"}
        elif validator == "wall_clock_latency_v1":
            fields = {"task_wall_clock_seconds"}
        else:
            fields = {"step_count", "model_call_count", "task_wall_clock_seconds"}
        for system_id, system_value in systems.items():
            system = _mapping(system_value, context=f"metrics.{system_id}")
            steps = _mapping(system.get("steps"), context=f"metrics.{system_id}.steps")
            efficiency = _mapping(
                system.get("efficiency"), context=f"metrics.{system_id}.efficiency"
            )
            registered = {
                "step_count": steps.get("all"),
                "model_call_count": efficiency.get("model_call_count"),
                "task_wall_clock_seconds": efficiency.get("task_wall_clock_seconds"),
            }
            for field in fields:
                _metric_summary(
                    registered.get(field),
                    context=f"metrics.{system_id}.{field}",
                    integer_numerator=field in {"step_count", "model_call_count"},
                )
    elif validator == "architecture_interface_matched_control_v1":
        _exact_keys(
            value,
            {
                "schema_version",
                "campaign_id",
                "architecture_matched",
                "interface_matched",
                "assessment",
            },
            "architecture/interface control assessment",
        )
        if (
            value.get("schema_version") != "table2-architecture-interface-control-v1"
            or value.get("campaign_id") != campaign_id
            or type(value.get("architecture_matched")) is not bool
            or type(value.get("interface_matched")) is not bool
            or value.get("assessment")
            != "DECLARATIVE_ONLY_NOT_SOURCE_ATTESTED"
        ):
            raise SchemaError("architecture/interface control assessment is malformed")
        # This v1 document is only an author assertion.  It does not bind the
        # two systems to source-attested runtime/model/interface manifests, so
        # even positive booleans cannot establish the conditional causal
        # control.  A future pre-outcome registry revision may introduce a
        # source-replaying validator; until then this condition is UNMET.
        return "UNMET"
    elif validator == "verified_failure_recovery_evidence_v1":
        for source_index, source_value in enumerate(values):
            if source_index == 0:
                systems = _registered_metric_systems(
                    source_value, context="normal-task recovery metric evidence"
                )
            else:
                if (
                    source_value.get("schema_version") != "table2.v1"
                    or source_value.get("publication_status") != "PILOT_ONLY"
                    or source_value.get("task_partition") != "recovery_diagnostic"
                ):
                    raise SchemaError("controlled recovery evidence changed diagnostic scope")
                systems = _mapping(
                    source_value.get("systems"), context="controlled recovery systems"
                )
                if set(systems) != {"E0", "E1", "E2", "E3"}:
                    raise SchemaError("controlled recovery evidence lacks E0-E3")
            for system_id in ("E2", "E3"):
                system = _mapping(systems[system_id], context=f"recovery.{system_id}")
                required_recovery = {
                    "recovery_success_rate",
                    "success_after_initial_failure",
                    "unrecovered_failure_rate",
                }
                if not required_recovery.issubset(system):
                    raise SchemaError("recovery metric evidence lacks registered outcomes")
                for metric_name in required_recovery:
                    _metric_summary(
                        system[metric_name],
                        context=f"recovery.{system_id}.{metric_name}",
                    )
    elif validator == "frozen_train_only_memory_evidence_v1":
        required = {
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
        }
        if any(value.get(key) != expected for key, expected in required.items()):
            raise SchemaError("memory evidence violates the frozen train-only contract")
        _positive_int(value.get("item_count"), context="memory item_count")
    elif validator == "paired_memory_intervention_trace_v1":
        if (
            value.get("schema_version") != "table2.v1"
            or value.get("publication_status") != FINAL_READY_STATUS
            or value.get("status") != "OK"
        ):
            raise SchemaError("memory trace evidence is not a final retrieval aggregate")
        _positive_int(value.get("query_count"), context="memory query_count")
        if not {"retrieval_coverage", "intervention_coverage"}.issubset(value):
            raise SchemaError("memory trace evidence lacks retrieval/intervention accounting")
        _metric_summary(value["retrieval_coverage"], context="retrieval coverage")
        _metric_summary(value["intervention_coverage"], context="intervention coverage")
    elif validator == "trigger_source_stratified_recovery_v1":
        if (
            value.get("schema_version") != "table2-trigger-source-recovery-v1"
            or value.get("campaign_id") != campaign_id
            or value.get("publication_status") != FINAL_READY_STATUS
            or not isinstance(value.get("by_trigger_source"), Mapping)
            or not value["by_trigger_source"]
        ):
            raise SchemaError("trigger-source evidence is malformed or empty")
    elif validator == "learned_trigger_only_ablation_v1":
        _exact_keys(
            value,
            {
                "schema_version",
                "campaign_id",
                "ablation_executed",
                "isolates_learned_trigger",
                "assessment",
            },
            "learned-trigger ablation assessment",
        )
        if (
            value.get("schema_version") != "table2-learned-trigger-ablation-v1"
            or value.get("campaign_id") != campaign_id
            or type(value.get("ablation_executed")) is not bool
            or type(value.get("isolates_learned_trigger")) is not bool
            or value.get("assessment")
            != "DECLARATIVE_ONLY_NOT_PAIRED_EXECUTION_EVIDENCE"
        ):
            raise SchemaError("learned-trigger ablation assessment is malformed")
        # As above, the v1 file contains no hash-bound paired ablation package.
        # Its booleans are descriptive declarations, not execution authority.
        return "UNMET"
    elif validator.startswith("p1_"):
        from .pillar1_diagnostics import validate_pillar1_diagnostic_report

        validate_pillar1_diagnostic_report(value)
        _validate_companion_report(
            value,
            campaign_id=campaign_id,
            pillar="P1",
            schema_version="table2.pillar1-diagnostic-report.v1",
            summary_keys={"overall", "by_failure_type", "by_recovery_strategy"},
        )
        overall = _mapping(value["summary"]["overall"], context="P1 overall")
        expected = {
            "p1_failure_detection_diagnostics_v1": {"failure_detection_accuracy"},
            "p1_failure_diagnosis_diagnostics_v1": {"failure_type_accuracy_on_failures"},
            "p1_executed_recovery_assessment_diagnostics_v1": {
                "recovery_resolution_accuracy",
                "recovery_progress_accuracy",
            },
        }[validator]
        if not expected.issubset(overall):
            raise SchemaError("P1 report lacks the requested diagnostic facet")
    elif validator == "authoritative_companion_input_provenance_v1":
        _exact_keys(
            value,
            {"schema_version", "campaign_id", "locked_test_rows_read", "pillars"},
            "companion input-provenance assessment",
        )
        if (
            value.get("schema_version") != "table2-companion-input-provenance-v1"
            or value.get("campaign_id") != campaign_id
            or value.get("locked_test_rows_read") != 0
        ):
            raise SchemaError("companion input-provenance assessment is malformed")
        pillars = _mapping(value.get("pillars"), context="companion provenance pillars")
        if set(pillars) != {"P1", "P2", "P3", "P4"}:
            raise SchemaError("companion input-provenance assessment lacks a pillar")
        package_sources = {
            f"P{index}": sources[index] for index in range(1, 5)
        }
        report_names = {
            "P1": "pillar1_diagnostic_report.json",
            "P2": "pillar2_diagnostic_report.json",
            "P3": "pillar3_diagnostic_report.json",
            "P4": "pillar4_diagnostic_report.json",
        }
        package_validators: dict[str, Any] = {}
        from .pillar1_diagnostics import validate_pillar1_diagnostic_package
        from .pillar2_diagnostics import validate_pillar2_diagnostic_package
        from .pillar3_diagnostics import validate_pillar3_diagnostic_package
        from .pillar4_diagnostics import validate_pillar4_diagnostic_package

        package_validators.update(
            {
                "P1": validate_pillar1_diagnostic_package,
                "P2": validate_pillar2_diagnostic_package,
                "P3": validate_pillar3_diagnostic_package,
                "P4": validate_pillar4_diagnostic_package,
            }
        )
        canonical: list[bool] = []
        for pillar in ("P1", "P2", "P3", "P4"):
            row = pillars[pillar]
            record = _mapping(row, context=f"companion provenance {pillar}")
            _exact_keys(
                record,
                {"package_manifest_relative_path", "package_manifest_sha256"},
                f"companion provenance {pillar}",
            )
            source_row = package_sources[pillar]
            if (
                record.get("package_manifest_relative_path")
                != source_row.get("relative_path")
                or record.get("package_manifest_sha256")
                != source_row.get("sha256")
            ):
                raise SchemaError(
                    f"companion provenance {pillar} is not bound to its package manifest"
                )
            manifest_path = Path(str(source_row["_resolved_path"]))
            result = package_validators[pillar](manifest_path.parent)
            if (
                result.get("status") != "PASS"
                or result.get("package_manifest_sha256") != source_row.get("sha256")
            ):
                raise SchemaError(f"companion provenance {pillar} package did not validate")
            report = _read_strict_json(
                manifest_path.parent / report_names[pillar],
                context=f"companion provenance {pillar} report",
            )
            if report.get("campaign_id") != campaign_id:
                raise SchemaError(f"companion provenance {pillar} campaign differs")
            canonical.append(
                report.get("evidence_role") == f"{pillar}_COMPANION_MECHANISM_EVIDENCE"
                and report.get("promotion_status")
                == "CANONICAL_PC01_COMPANION_EVIDENCE"
            )
        return "SATISFIED" if all(canonical) else "UNMET"
    elif validator.startswith("p2_"):
        from .pillar2_diagnostics import validate_pillar2_diagnostic_report

        validate_pillar2_diagnostic_report(value)
        p2_role = value.get("evidence_role")
        p2_promotion = value.get("promotion_status")
        p2_valid_promotion = (
            p2_role == "P2_COMPANION_MECHANISM_EVIDENCE"
            and p2_promotion == "CANONICAL_PC01_COMPANION_EVIDENCE"
        ) or (
            p2_role == "ENGINEERING_DIAGNOSTIC_ONLY_UNPROMOTABLE"
            and p2_promotion == "UNPROMOTABLE_INPUT_PROVENANCE_BLOCKED"
        )
        if (
            value.get("schema_version") != "table2.pillar2-diagnostic-report.v1"
            or value.get("campaign_id") != campaign_id
            or value.get("paper_table_status") != "N/R"
            or value.get("affects_primary_table2") is not False
            or value.get("runtime_outputs_consumed") is not False
            or value.get("locked_test_rows_read") != 0
            or not p2_valid_promotion
        ):
            raise SchemaError("P2 companion report is not registered evidence")
        if validator == "p2_temporal_causal_diagnostics_v1":
            replay = _mapping(value.get("temporal_replay"), context="P2 temporal replay")
            if replay.get("status") != "PASS" or not isinstance(replay.get("records"), list):
                raise SchemaError("P2 temporal replay is absent")
        else:
            summary = _mapping(value.get("summary"), context="P2 summary")
            if not {
                "pre_action_modality_sensitivity",
                "post_action_controls",
            }.issubset(summary) or not isinstance(value.get("condition_definitions"), Mapping):
                raise SchemaError("P2 modality conditions are absent")
    elif validator.startswith("p3_"):
        from .pillar3_diagnostics import validate_pillar3_diagnostic_report

        validate_pillar3_diagnostic_report(value)
        _validate_companion_report(
            value,
            campaign_id=campaign_id,
            pillar="P3",
            schema_version="table2.pillar3-diagnostic-report.v1",
            summary_keys={"overall", "by_action_class", "by_target_size"},
        )
        summary = value["summary"]
        if validator == "p3_action_class_diagnostics_v1" and not summary["by_action_class"]:
            raise SchemaError("P3 action-class diagnostics are empty")
        if validator == "p3_grounding_parameter_diagnostics_v1" and not {
            "bbox_iou_mean",
            "parameter_resolution_rate",
        }.issubset(_mapping(summary["overall"], context="P3 overall")):
            raise SchemaError("P3 grounding/parameter diagnostics are absent")
        if validator == "p3_invalid_output_accounting_v1" and not {
            "invalid_output_count",
            "accounted_rejected_executor_requests",
        }.issubset(summary):
            raise SchemaError("P3 invalid-output accounting is absent")
    elif validator.startswith("p4_"):
        from .pillar4_diagnostics import validate_pillar4_diagnostic_report

        validate_pillar4_diagnostic_report(value)
        _validate_companion_report(
            value,
            campaign_id=campaign_id,
            pillar="P4",
            schema_version="table2.pillar4-diagnostic-report.v1",
            summary_keys={"query_count", "retrieval_coverage", "admission_rate"},
        )
        if validator == "p4_retrieval_admission_diagnostics_v1" and not {
            "retrieval_coverage",
            "admission_rate",
            "abstention_rate",
        }.issubset(value["summary"]):
            raise SchemaError("P4 retrieval/admission diagnostics are absent")
        if validator == "p4_store_provenance_integrity_v1":
            integrity = _mapping(value.get("store_integrity"), context="P4 store integrity")
            if (
                integrity.get("unchanged") is not True
                or integrity.get("write_enabled") is not False
            ):
                raise SchemaError("P4 store integrity is not immutable")
        if validator == "p4_no_write_integrity_v1" and value.get(
            "memory_writes_executed"
        ) != 0:
            raise SchemaError("P4 diagnostics wrote evaluation memory")
    elif validator == "checkpoint_backed_embedding_execution_evidence_v1":
        from .pc01_checkpoint_compatibility import (
            validate_pc01_checkpoint_compatibility_receipt,
        )

        receipt = validate_pc01_checkpoint_compatibility_receipt(value)
        artifacts = _mapping(
            receipt.get("artifact_bindings"), context="checkpoint artifact bindings"
        )
        inference = _mapping(receipt.get("inference"), context="checkpoint inference")
        embedding = _mapping(
            inference.get("p4_memory_embedding"), context="checkpoint P4 embedding"
        )
        if (
            receipt.get("status") != "PASS"
            or artifacts.get("checkpoint_sha256")
            != "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a"
            or embedding.get("dimension") != 768
            or inference.get("checkpoint_forward_executed") is not True
            or embedding.get("direct_runtime_exact") is not True
            or embedding.get("repeat_exact") is not True
            or _mapping(
                receipt.get("causal_and_scientific_boundaries"),
                context="checkpoint scientific boundaries",
            ).get("locked_test_rows_read")
            != 0
        ):
            raise SchemaError("checkpoint-backed P4 embedding assessment is malformed")
        return "SATISFIED"
    elif validator in {
        "complete_contrast_disclosure_v1",
        "pillar_limitations_disclosure_v1",
        "model_seed_uncertainty_limitation_v1",
    }:
        disclosure_keys = {
            "schema_version",
            "campaign_id",
            "registered_contrasts_disclosed",
            "claim_statuses_disclosed",
            "claim_outcomes_disclosed",
            "pillar_limitations",
            "model_seed_scope",
        }
        _exact_keys(value, disclosure_keys, "paper disclosure evidence")
        if (
            value.get("schema_version") != "table2-paper-disclosures-v1"
            or value.get("campaign_id") != campaign_id
        ):
            raise SchemaError("paper disclosure evidence is malformed")
        disclosed_statuses = _mapping(
            value.get("claim_statuses_disclosed"), context="disclosed claim statuses"
        )
        if set(disclosed_statuses) != set(DISCLOSED_CLAIM_IDS):
            raise SchemaError("paper disclosure does not enumerate every registered claim")
        rules_by_id = {rule.claim_id: rule for rule in CLAIM_RULES}
        for claim_id in DISCLOSED_CLAIM_IDS:
            status = disclosed_statuses[claim_id]
            if type(status) is not str or status not in rules_by_id[claim_id].permitted_statuses:
                raise SchemaError(f"paper disclosure has invalid status for {claim_id}")
        if claim_statuses is not None and dict(disclosed_statuses) != {
            claim_id: claim_statuses[claim_id] for claim_id in DISCLOSED_CLAIM_IDS
        }:
            raise SchemaError("paper disclosure claim statuses differ from the claim report")

        disclosed_outcomes = _mapping(
            value.get("claim_outcomes_disclosed"),
            context="disclosed claim outcomes",
        )
        if set(disclosed_outcomes) != set(DISCLOSED_CLAIM_IDS):
            raise SchemaError("paper disclosure does not enumerate every claim outcome")
        for claim_id in DISCLOSED_CLAIM_IDS:
            outcome = _mapping(
                disclosed_outcomes[claim_id],
                context=f"disclosed claim outcome {claim_id}",
            )
            _exact_keys(
                outcome,
                DISCLOSURE_OUTCOME_KEYS,
                f"disclosed claim outcome {claim_id}",
            )
            status = disclosed_statuses[claim_id]
            conclusion, limitation = STATUS_TEXT[status]
            expected_outcome = {
                "status": status,
                "conclusion": conclusion,
                "limitation": limitation,
            }
            if dict(outcome) != expected_outcome:
                raise SchemaError(
                    f"paper disclosure claim outcome differs from controlled status: {claim_id}"
                )
            if claim_statuses is not None and status != claim_statuses[claim_id]:
                raise SchemaError(
                    f"paper disclosure claim outcome differs from the claim report: {claim_id}"
                )

        limitations = _mapping(value.get("pillar_limitations"), context="pillar limitations")
        if dict(limitations) != REGISTERED_PILLAR_LIMITATIONS:
            raise SchemaError("paper disclosure pillar limitations differ from registration")

        disclosed_contrasts = _mapping(
            value.get("registered_contrasts_disclosed"),
            context="disclosed registered contrasts",
        )
        if set(disclosed_contrasts) != set(STATISTICS_CONTRAST_BY_CLAIM_CONTRAST):
            raise SchemaError("paper disclosure omits a registered contrast")
        if validator == "complete_contrast_disclosure_v1":
            statistics = values[1]
            contrasts = _validated_task_success_contrasts(
                statistics, require_interval=True
            )
            for public_name, statistics_name in (
                STATISTICS_CONTRAST_BY_CLAIM_CONTRAST.items()
            ):
                row = _mapping(
                    disclosed_contrasts[public_name],
                    context=f"paper disclosure {public_name}",
                )
                _exact_keys(
                    row,
                    {"statistics_contrast", "estimate", "ci_low", "ci_high"},
                    f"paper disclosure {public_name}",
                )
                source = _mapping(
                    _mapping(
                        contrasts[statistics_name], context=f"statistics.{statistics_name}"
                    ).get("task_success"),
                    context=f"statistics.{statistics_name}.task_success",
                )
                expected = {
                    "statistics_contrast": statistics_name,
                    "estimate": source["estimate"],
                    "ci_low": source["ci_low"],
                    "ci_high": source["ci_high"],
                }
                if dict(row) != expected:
                    raise SchemaError(
                        f"paper disclosure {public_name} differs from final statistics"
                    )
        if validator == "model_seed_uncertainty_limitation_v1" and value.get(
            "model_seed_scope"
        ) != "seed_42_only_task_uncertainty_not_model_seed_uncertainty":
            raise SchemaError("paper disclosure changes the model-seed limitation")
    else:  # pragma: no cover - registry validation makes this unreachable.
        raise SchemaError(f"unimplemented paper-claim evidence validator: {validator}")
    return "SATISFIED"


def build_claim_evidence_assessment(
    *,
    kind: str,
    registry_path: str | Path,
    campaign_dir: str | Path,
    finding: str | None = None,
) -> dict[str, Any]:
    """Recompute and serialize one registered evidence assessment."""

    validate_claim_registry(registry_path)
    rule = EVIDENCE_RULE_BY_KIND.get(kind)
    if rule is None:
        raise SchemaError(f"unregistered paper-claim evidence kind: {kind!r}")
    root = Path(campaign_dir).resolve()
    campaign = _read_strict_json(
        _campaign_file(root, "campaign_manifest.json", context="campaign manifest"),
        context="campaign manifest",
    )
    campaign_id = str(campaign.get("campaign_id", "")).strip()
    if not campaign_id:
        raise SchemaError("campaign manifest lacks campaign_id")
    sources: list[dict[str, Any]] = []
    internal_sources: list[dict[str, Any]] = []
    for relative in rule.source_relative_paths:
        source = _campaign_file(root, relative, context=f"{kind} source")
        public = {"relative_path": relative, "sha256": sha256_file(source)}
        sources.append(public)
        internal_sources.append({**public, "_resolved_path": str(source)})
    assessment_status = _validate_evidence_source_content(
        rule, internal_sources, campaign_id=campaign_id
    )
    controlled_finding = f"{rule.kind}:{assessment_status}"
    if finding is not None and str(finding) != controlled_finding:
        raise SchemaError(
            "paper-claim evidence finding is machine-controlled; expected "
            f"{controlled_finding}"
        )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "producer": EVIDENCE_PRODUCER,
        "evidence_kind": rule.kind,
        "campaign_id": campaign_id,
        "assessment_status": assessment_status,
        "finding": controlled_finding,
        "sources": sources,
    }


def write_claim_evidence_assessment(
    *,
    kind: str,
    registry_path: str | Path,
    campaign_dir: str | Path,
    finding: str | None = None,
) -> Path:
    rule = EVIDENCE_RULE_BY_KIND.get(kind)
    if rule is None:
        raise SchemaError(f"unregistered paper-claim evidence kind: {kind!r}")
    root = Path(campaign_dir).resolve()
    output = _safe_output_file(
        root, rule.relative_path, context="paper-claim evidence output"
    )
    if output.exists() or output.is_symlink():
        raise SchemaError(f"paper-claim evidence output already exists: {output}")
    payload = build_claim_evidence_assessment(
        kind=kind,
        registry_path=registry_path,
        campaign_dir=root,
        finding=finding,
    )
    return atomic_write_json(output, payload, mode=0o444)


def _validate_assessment_file(
    *,
    rule: EvidenceRule,
    evidence_path: Path,
    campaign_root: Path,
    campaign_id: str,
    claim_statuses: Mapping[str, str],
    context: str,
) -> tuple[str, str]:
    assessment = _read_strict_json(evidence_path, context=context)
    _exact_keys(assessment, EVIDENCE_ASSESSMENT_KEYS, context)
    if (
        assessment.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or assessment.get("producer") != EVIDENCE_PRODUCER
        or assessment.get("evidence_kind") != rule.kind
        or assessment.get("campaign_id") != campaign_id
    ):
        raise SchemaError(f"{context} identity differs from the evidence catalog")
    finding = str(assessment.get("finding", ""))
    source_rows = assessment.get("sources")
    if not isinstance(source_rows, list) or len(source_rows) != len(
        rule.source_relative_paths
    ):
        raise SchemaError(f"{context} source inventory differs from registration")
    internal: list[dict[str, Any]] = []
    for index, (row, expected_relative) in enumerate(
        zip(source_rows, rule.source_relative_paths)
    ):
        if not isinstance(row, Mapping):
            raise SchemaError(f"{context}.sources[{index}] must be an object")
        _exact_keys(row, EVIDENCE_SOURCE_KEYS, f"{context}.sources[{index}]")
        if row.get("relative_path") != expected_relative:
            raise SchemaError(f"{context} source path differs from registration")
        source = _campaign_file(
            campaign_root, expected_relative, context=f"{context} source"
        )
        digest = _require_sha256(row.get("sha256"), context=f"{context} source")
        if sha256_file(source) != digest:
            raise SchemaError(f"{context} source SHA-256 mismatch")
        internal.append({**dict(row), "_resolved_path": str(source)})
    recomputed = _validate_evidence_source_content(
        rule,
        internal,
        campaign_id=campaign_id,
        claim_statuses=claim_statuses,
    )
    if assessment.get("assessment_status") != recomputed:
        raise SchemaError(f"{context} assessment status differs from source evidence")
    expected_finding = f"{rule.kind}:{recomputed}"
    if finding != expected_finding:
        raise SchemaError(f"{context} finding is not the controlled assessment finding")
    return recomputed, finding


def _validate_evidence_rows(
    rows: Any,
    *,
    required: Sequence[str],
    campaign_root: Path,
    campaign_id: str,
    claim_statuses: Mapping[str, str],
    status: str,
    context: str,
) -> dict[str, tuple[Mapping[str, Any], str, str]]:
    if not isinstance(rows, list):
        raise SchemaError(f"{context}.evidence must be an array")
    seen: dict[str, tuple[Mapping[str, Any], str, str]] = {}
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SchemaError(f"{context}.evidence[{index}] must be an object")
        _exact_keys(row, EVIDENCE_KEYS, f"{context}.evidence[{index}]")
        kind = str(row.get("kind", "")).strip()
        rule = EVIDENCE_RULE_BY_KIND.get(kind)
        if rule is None or kind in seen:
            raise SchemaError(f"{context} evidence kinds are unregistered or duplicated")
        if row.get("relative_path") != rule.relative_path:
            raise SchemaError(f"{context} evidence path differs from catalog: {kind}")
        if rule.relative_path in seen_paths:
            raise SchemaError(f"{context} reuses one assessment file for multiple kinds")
        seen_paths.add(rule.relative_path)
        source = _campaign_file(campaign_root, rule.relative_path, context=f"{context} evidence")
        digest = _require_sha256(row.get("sha256"), context=f"{context} evidence")
        if sha256_file(source) != digest:
            raise SchemaError(f"{context} evidence SHA-256 mismatch: {rule.relative_path}")
        assessment_status, finding = _validate_assessment_file(
            rule=rule,
            evidence_path=source,
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            claim_statuses=claim_statuses,
            context=f"{context} evidence {kind}",
        )
        seen[kind] = (row, assessment_status, finding)
    expected = set(required)
    actual = set(seen)
    if status == INITIAL_STATUS:
        if actual:
            raise SchemaError(f"{context} must not attach evidence before evaluation")
    elif actual != expected:
        raise SchemaError(
            f"{context} evidence kinds differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return seen


def _validate_unmet_conditions(
    value: Any,
    *,
    rule: ClaimRule,
    evidence: Mapping[str, tuple[Mapping[str, Any], str, str]],
    status: str,
    context: str,
) -> None:
    if not isinstance(value, list):
        raise SchemaError(f"{context}.unmet_conditions must be an array")
    seen: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise SchemaError(f"{context}.unmet_conditions[{index}] must be an object")
        _exact_keys(row, UNMET_CONDITION_KEYS, f"{context}.unmet_conditions[{index}]")
        condition = str(row.get("condition", "")).strip()
        if condition not in rule.conditions or condition in seen:
            raise SchemaError(f"{context} has an unregistered or duplicate unmet condition")
        seen.add(condition)
        finding = row.get("finding")
        if type(finding) is not str:
            raise SchemaError(f"{context} unmet-condition finding is not controlled text")
        reference = row.get("assessment_evidence")
        if not isinstance(reference, Mapping):
            raise SchemaError(f"{context} unmet-condition evidence must be an object")
        _exact_keys(reference, EVIDENCE_KEYS, f"{context} unmet-condition evidence")
        evidence_entry = evidence.get(condition)
        if evidence_entry is None or dict(reference) != dict(evidence_entry[0]):
            raise SchemaError(f"{context} unmet condition is not bound to its assessment")
        if evidence_entry[1] != "UNMET" or finding != evidence_entry[2]:
            raise SchemaError(f"{context} unmet-condition finding differs from source evidence")

    assessed_unmet = {
        condition
        for condition in rule.conditions
        if condition in evidence and evidence[condition][1] == "UNMET"
    }
    if status == "NOT_SUPPORTED":
        if not rule.conditions:
            raise SchemaError(
                f"{context} NOT_SUPPORTED is reserved for a registered unmet condition; "
                "use NEGATIVE_OR_NULL for a measured non-positive result"
            )
        if not seen or seen != assessed_unmet:
            raise SchemaError(f"{context} must enumerate every assessed unmet condition")
        nonconditions = set(rule.required_evidence) - set(rule.conditions)
        if any(evidence[kind][1] != "SATISFIED" for kind in nonconditions):
            raise SchemaError(f"{context} has invalid non-condition evidence")
    else:
        if seen:
            raise SchemaError(f"{context} attaches unmet conditions to status {status}")
        if status in {"SUPPORTED", "NEGATIVE_OR_NULL"} and any(
            evidence[kind][1] != "SATISFIED" for kind in rule.required_evidence
        ):
            raise SchemaError(f"{context} resolved status requires satisfied evidence")


def _complete_cost_accounting_decision(
    evidence: Mapping[str, tuple[Mapping[str, Any], str, str]],
) -> str:
    expected = {"browser_action_counts", "model_call_counts", "wall_clock_latency"}
    if set(evidence) != expected or any(
        evidence[kind][1] != "SATISFIED" for kind in expected
    ):
        raise SchemaError("complete-cost decision lacks validated cost evidence")
    return "SUPPORTED"


def _complete_disclosure_decision(
    evidence: Mapping[str, tuple[Mapping[str, Any], str, str]],
) -> str:
    expected = {
        "complete_contrast_disclosure",
        "pillar_limitations_disclosure",
        "model_seed_uncertainty_limitation",
    }
    if set(evidence) != expected or any(
        evidence[kind][1] != "SATISFIED" for kind in expected
    ):
        raise SchemaError("complete-disclosure decision lacks validated disclosure evidence")
    return "SUPPORTED"


DECISION_RULE_VALIDATORS = {
    "machine_validated_complete_cost_accounting_v1": _complete_cost_accounting_decision,
    "machine_validated_complete_disclosure_v1": _complete_disclosure_decision,
}
if set(DECISION_RULE_VALIDATORS) != set(EXECUTABLE_DECISION_RULES):  # pragma: no cover
    raise RuntimeError("paper-claim decision-rule registry is inconsistent")


def _validate_executable_decision_rule(
    *,
    rule: ClaimRule,
    status: str,
    evidence: Mapping[str, tuple[Mapping[str, Any], str, str]],
    context: str,
) -> None:
    if rule.decision_rule == UNREGISTERED_DECISION_RULE:
        if status != INITIAL_STATUS:
            raise SchemaError(
                f"claim {rule.claim_id} must remain N/R until a frozen "
                "claim-specific decision rule exists"
            )
        return
    validator = DECISION_RULE_VALIDATORS.get(rule.decision_rule)
    if validator is None:
        raise SchemaError(
            f"{context} references an unknown executable decision rule: "
            f"{rule.decision_rule}"
        )
    if status == INITIAL_STATUS:
        return
    expected = validator(evidence)
    if status != expected:
        raise SchemaError(
            f"{context} status {status} differs from decision-rule result {expected}"
        )


def validate_claim_report(
    report_path: str | Path,
    *,
    registry_path: str | Path,
    campaign_dir: str | Path,
) -> dict[str, Any]:
    """Validate a claim report against its frozen registry and campaign bytes."""

    report_input = Path(report_path)
    registry_input = Path(registry_path)
    root = Path(campaign_dir).resolve()
    if (
        report_input.is_symlink()
        or not report_input.is_file()
        or report_input.absolute() != report_input.resolve()
    ):
        raise SchemaError("paper claim report must be a regular non-symlink file")
    report_source = report_input.resolve()
    if root != report_source and root not in report_source.parents:
        raise SchemaError("paper claim report must be inside the campaign directory")
    registry = validate_claim_registry(registry_input)
    registry_source = registry_input.resolve()
    campaign_source = _campaign_file(
        root, "campaign_manifest.json", context="campaign manifest"
    )
    campaign = _read_strict_json(campaign_source, context="campaign manifest")
    report = _read_strict_json(report_source, context="paper claim report")
    _exact_keys(report, REPORT_KEYS, "paper claim report")
    registry_digest = sha256_file(registry_source)
    if (
        campaign.get("paper_claim_registry_id") != REGISTRY_ID
        or campaign.get("paper_claim_registry_sha256") != registry_digest
    ):
        raise SchemaError("campaign manifest is not bound to the supplied paper-claim registry")
    expected_header = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "registry_id": REGISTRY_ID,
        "registry_sha256": registry_digest,
        "campaign_id": str(campaign.get("campaign_id", "")),
        "campaign_manifest_sha256": sha256_file(campaign_source),
    }
    for key, expected in expected_header.items():
        if report.get(key) != expected:
            raise SchemaError(f"paper claim report {key} differs from frozen evidence")

    authority_relative = safe_relative_path(
        str(report.get("publication_authority_relative_path", ""))
    )
    if authority_relative not in {
        Path("campaign_manifest.json"),
        Path("aggregate/validation_report.json"),
    }:
        raise SchemaError("paper claim publication authority path is unregistered")
    authority_path = _campaign_file(
        root, str(authority_relative), context="paper claim publication authority"
    )
    if report.get("publication_authority_sha256") != sha256_file(authority_path):
        raise SchemaError("paper claim publication authority SHA-256 mismatch")
    authority = _read_strict_json(authority_path, context="paper claim publication authority")
    publication_status = str(authority.get("publication_status", ""))
    if report.get("publication_status") != publication_status:
        raise SchemaError("paper claim report publication status differs from authority")
    ready = publication_status == FINAL_READY_STATUS
    if ready:
        from .package_validator import validate_campaign

        validation = validate_campaign(
            root,
            require_complete=True,
            require_aggregates=True,
        )
        if not validation.passed or validation.publication_status != FINAL_READY_STATUS:
            raise SchemaError(
                "final-ready claims require an independently recomputed "
                "READY_FOR_TABLE2 campaign validation"
            )
        if authority_relative != Path("aggregate/validation_report.json"):
            raise SchemaError("final-ready claims require aggregate validation authority")
        if authority.get("status") != "PASS" or authority.get("errors") not in ([], None):
            raise SchemaError("final-ready claim authority is not a passing validation report")
        try:
            recomputed_authority = validation.to_dict()
        except AttributeError as exc:  # pragma: no cover - production validator contract.
            raise SchemaError(
                "campaign validator did not return serializable publication authority"
            ) from exc
        if dict(authority) != recomputed_authority:
            raise SchemaError(
                "saved final-ready authority differs from independently recomputed validation"
            )
        if str(campaign.get("campaign_kind")) != "locked_final":
            raise SchemaError("only a locked-final campaign may resolve paper claims")
    else:
        if authority_relative != Path("campaign_manifest.json"):
            raise SchemaError("non-final claims must use the frozen campaign manifest authority")
        if publication_status not in PILOT_PUBLICATION_STATUSES:
            raise SchemaError("unregistered non-final publication status")
        if report.get("paper_table_status") != INITIAL_STATUS:
            raise SchemaError("non-final claim report must keep paper_table_status=N/R")

    claims = report.get("claims")
    if not isinstance(claims, list) or len(claims) != len(CLAIM_RULES):
        raise SchemaError("paper claim report has the wrong claim count")
    claim_statuses: dict[str, str] = {}
    for index, (row, rule) in enumerate(zip(claims, CLAIM_RULES)):
        if not isinstance(row, Mapping):
            raise SchemaError(f"paper claim report row[{index}] must be an object")
        _exact_keys(row, REPORT_CLAIM_KEYS, f"paper claim report row[{index}]")
        if row.get("claim_id") != rule.claim_id:
            raise SchemaError("paper claim report order/identity differs from registry")
        raw_status = row.get("status")
        if type(raw_status) is not str or raw_status not in rule.permitted_statuses:
            raise SchemaError(f"paper claim report has invalid status: {rule.claim_id}")
        claim_statuses[rule.claim_id] = raw_status

    for index, (row, rule) in enumerate(zip(claims, CLAIM_RULES)):
        status = claim_statuses[rule.claim_id]
        expected_conclusion, expected_limitation = STATUS_TEXT[status]
        if (
            row.get("conclusion") != expected_conclusion
            or row.get("limitation") != expected_limitation
        ):
            raise SchemaError(
                f"paper claim status text differs from controlled vocabulary: {rule.claim_id}"
            )
        if not ready and status != INITIAL_STATUS:
            raise SchemaError(f"non-final campaign resolves a paper claim: {rule.claim_id}")
        if status != INITIAL_STATUS and rule.conditions and not set(rule.conditions).issubset(
            rule.required_evidence
        ):
            raise SchemaError(f"registered claim condition is not evidence-bound: {rule.claim_id}")
        evidence = _validate_evidence_rows(
            row.get("evidence"),
            required=rule.required_evidence,
            campaign_root=root,
            campaign_id=str(report["campaign_id"]),
            claim_statuses=claim_statuses,
            status=status,
            context=f"claim {rule.claim_id}",
        )
        _validate_unmet_conditions(
            row.get("unmet_conditions"),
            rule=rule,
            evidence=evidence,
            status=status,
            context=f"claim {rule.claim_id}",
        )
        _validate_executable_decision_rule(
            rule=rule,
            status=status,
            evidence=evidence,
            context=f"claim {rule.claim_id}",
        )

    if ready:
        unresolved = any(row.get("status") == INITIAL_STATUS for row in claims)
        expected_table_status = INITIAL_STATUS if unresolved else "READY"
        if report.get("paper_table_status") != expected_table_status:
            raise SchemaError(
                "final claim report paper_table_status differs from claim resolution state"
            )

    prohibited = report.get("prohibited_claims")
    if not isinstance(prohibited, list) or len(prohibited) != len(PROHIBITED_CLAIMS):
        raise SchemaError("paper claim report has the wrong prohibition count")
    for index, (row, rule) in enumerate(zip(prohibited, PROHIBITED_CLAIMS)):
        if not isinstance(row, Mapping):
            raise SchemaError(f"paper prohibition row[{index}] must be an object")
        _exact_keys(row, REPORT_PROHIBITED_KEYS, f"paper prohibition row[{index}]")
        if row != {"claim_id": rule.claim_id, "status": "NOT_CLAIMED"}:
            raise SchemaError(f"paper prohibition was altered or claimed: {rule.claim_id}")
    return report


def validate_paper_claim_readiness(
    report_path: str | Path,
    *,
    registry_path: str | Path,
    campaign_dir: str | Path,
) -> dict[str, Any]:
    """Issue the distinct paper-claim gate after campaign and claim validation.

    ``READY_FOR_TABLE2`` authenticates the campaign package.  It is a necessary
    input to this function, never an alias for ``READY_FOR_PAPER_CLAIMS``.
    """

    report = validate_claim_report(
        report_path,
        registry_path=registry_path,
        campaign_dir=campaign_dir,
    )
    if report.get("publication_status") != FINAL_READY_STATUS:
        raise SchemaError(
            "paper-claim readiness is available only after READY_FOR_TABLE2 campaign validation"
        )
    unresolved = [
        str(row.get("claim_id"))
        for row in report["claims"]
        if row.get("status") == INITIAL_STATUS
    ]
    if unresolved:
        raise SchemaError(
            "paper-claim readiness has unresolved claims: " + ", ".join(unresolved)
        )
    if report.get("paper_table_status") != "READY":
        raise SchemaError("paper-claim readiness requires paper_table_status=READY")
    negative_row = next(
        row
        for row in report["claims"]
        if row.get("claim_id") == "negative_null_and_unsupported_results_disclosed"
    )
    if negative_row.get("status") != "SUPPORTED":
        raise SchemaError("paper-claim readiness requires complete negative-result disclosure")
    source = Path(report_path).resolve()
    registry_source = Path(registry_path).resolve()
    root = Path(campaign_dir).resolve()
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": CLAIM_READY_STATUS,
        "campaign_id": report["campaign_id"],
        "campaign_publication_status": FINAL_READY_STATUS,
        "paper_table_status": "READY",
        "registry_sha256": sha256_file(registry_source),
        "claim_report_sha256": sha256_file(source),
        "campaign_manifest_sha256": sha256_file(root / "campaign_manifest.json"),
        "publication_authority_sha256": report["publication_authority_sha256"],
    }


def write_initial_claim_report(
    output_path: str | Path,
    *,
    registry_path: str | Path,
    campaign_manifest_path: str | Path,
) -> Path:
    output = Path(output_path)
    if output.parent.absolute() != output.parent.resolve():
        raise SchemaError("paper claim report output parent traverses a symlink")
    if output.exists() or output.is_symlink():
        raise SchemaError(f"paper claim report output already exists: {output}")
    payload = build_initial_claim_report(
        registry_path=registry_path,
        campaign_manifest_path=campaign_manifest_path,
    )
    return atomic_write_json(output, payload)
