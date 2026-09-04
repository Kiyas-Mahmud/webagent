from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.package_validator import (
    _validate_protocol_access_boundary,
    _validate_system_overlay,
    _validate_task_boundary,
)
from web_agent.eval.table2.selection_evidence import CANDIDATE_MODEL_IDS
from web_agent.runtime.contracts import SystemID
from web_agent.runtime.protocol import (
    REGISTERED_BUDGETS,
    REGISTERED_FINAL_CANDIDATE_IDS,
    REGISTERED_FINAL_SELECTION_MODE,
    REGISTERED_PARAMETER_PROVIDER_DECODING_PARAMETERS,
    REGISTERED_PC01_CANDIDATE_IDS,
    REGISTERED_PC01_PILOT_PROTOCOL_ID,
    REGISTERED_PC01_SELECTION_MODE,
    REGISTERED_RECOVERY_POLICY_SIGNAL_ORDER,
    REGISTERED_RECOVERY_POLICY_TRUTH_TABLE,
    REGISTERED_RNG_KEY_FIELDS,
    REGISTERED_RNG_NAMESPACE_FIELDS,
    REGISTERED_RNG_SYSTEM_ID_IN_KEY,
    RuntimeStage,
    LoopRule,
    load_protocol_bundle,
    switches_for,
    validate_frozen_protocol_mapping,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "eval" / "table2"
PILOT_ROOT = ROOT / "benchmarks" / "table2" / "pilot"


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_registered_protocol_freezes_pilot_boundaries_and_budgets():
    protocol = _yaml(CONFIG_ROOT / "protocol.yaml")
    pilot = _yaml(CONFIG_ROOT / "pilot_webarena.yaml")

    assert protocol["protocol_id"] == REGISTERED_PC01_PILOT_PROTOCOL_ID
    assert protocol["evidence_label"] == "PILOT_ONLY"
    assert protocol["paper_table_status"] == "N/R"
    assert protocol["selection"] == {
        "mode": REGISTERED_PC01_SELECTION_MODE,
        "candidate_ids": list(REGISTERED_PC01_CANDIDATE_IDS),
        "required_candidate_count": 1,
        "checkpoint_selection": "validation_only",
        "locked_task_eligible": False,
    }
    assert protocol["benchmark"]["allow_locked_reads"] is False
    assert protocol["verification"]["runtime_can_read_full_labels"] is False
    assert protocol["memory"]["runtime_mode"] == "immutable_read_only"
    assert protocol["memory"]["embedding_dimension"] == 768
    assert protocol["memory"]["top_k"] == 3
    assert protocol["parameter_provider"]["decoding_parameters"] == (
        REGISTERED_PARAMETER_PROVIDER_DECODING_PARAMETERS
    )
    assert pilot["normal_episode_count"] == 50 * 4
    assert pilot["recovery_episode_count"] == 15 * 4
    assert pilot["planned_episode_count"] == 260
    assert pilot["locked_test_access"] == "forbidden"
    assert tuple(protocol["randomness"]["namespace_fields"]) == (
        REGISTERED_RNG_NAMESPACE_FIELDS
    )
    assert tuple(protocol["randomness"]["key_fields"]) == REGISTERED_RNG_KEY_FIELDS
    assert (
        protocol["randomness"]["system_id_in_key"]
        is REGISTERED_RNG_SYSTEM_ID_IN_KEY
        is False
    )
    trigger = protocol["recovery_trigger"]
    assert tuple(trigger["policy_signal_order"]) == (
        REGISTERED_RECOVERY_POLICY_SIGNAL_ORDER
    )
    assert trigger["policy_truth_table"] == dict(
        REGISTERED_RECOVERY_POLICY_TRUTH_TABLE
    )

    budgets = protocol["budgets"]
    assert budgets["executor_requests_per_episode"] == REGISTERED_BUDGETS.max_executor_steps
    assert budgets["recovery_attempts_per_incident"] == REGISTERED_BUDGETS.max_recovery_attempts_per_incident
    assert budgets["recovery_attempts_per_episode"] == REGISTERED_BUDGETS.max_recovery_attempts_per_episode
    assert budgets["task_timeout_seconds"] == REGISTERED_BUDGETS.episode_timeout_seconds
    assert budgets["pre_browser_setup_timeout_seconds"] == 120


def test_declared_rng_fields_exactly_match_hashed_stage_key_and_exclude_system():
    bundle = load_protocol_bundle(
        CONFIG_ROOT / "protocol.yaml",
        campaign_id="rng-field-audit",
    )
    key = bundle.protocol.rng_factory().key(
        task_id="task-1",
        repeat_id=0,
        matched_seed=42,
        stage=RuntimeStage.RECOVERY_RESOLUTION,
        decision_index=3,
        incident_index=2,
        attempt_index=1,
        stream="planner",
    )
    assert tuple(key.to_dict()) == REGISTERED_RNG_KEY_FIELDS
    assert "system_id" not in key.to_dict()


def test_parameter_provider_decoding_parameters_are_exactly_frozen():
    protocol = _yaml(CONFIG_ROOT / "protocol.yaml")
    validate_frozen_protocol_mapping(protocol)

    changed = deepcopy(protocol)
    changed["parameter_provider"]["decoding_parameters"]["temperature"] = 0.5
    with pytest.raises(
        ValueError,
        match="parameter_provider.decoding_parameters.temperature",
    ):
        validate_frozen_protocol_mapping(changed)

    extended = deepcopy(protocol)
    extended["parameter_provider"]["decoding_parameters"]["num_beams"] = 2
    with pytest.raises(ValueError, match="exactly the registered keys"):
        validate_frozen_protocol_mapping(extended)


def test_protocol_freezes_exact_loop_semantics():
    protocol = _yaml(CONFIG_ROOT / "protocol.yaml")
    validate_frozen_protocol_mapping(protocol)
    assert protocol["loop_rule"] == {
        "state_fingerprint_algorithm": "canonical_sha256_v1",
        "state_fingerprint_fields": [
            "screenshot_sha256",
            "url",
            "title",
            "page_state",
        ],
        "action_target_fingerprint_algorithm": "canonical_sha256_v1",
        "action_target_fingerprint_fields": [
            "action_type",
            "parameters",
            "bbox",
        ],
        "perceptual_similarity_metric": "exact_hash_match",
        "perceptual_similarity_threshold": 1.0,
        "rolling_window": 6,
        "equivalent_repetition_count": 3,
        "detect_abab_cycle": True,
    }

    weakened = deepcopy(protocol)
    weakened["loop_rule"]["equivalent_repetition_count"] = 4
    with pytest.raises(ValueError, match="loop_rule.equivalent_repetition_count"):
        validate_frozen_protocol_mapping(weakened)
    with pytest.raises(TypeError, match="exact registered types"):
        LoopRule(detect_abab_cycle=1)  # type: ignore[arg-type]


def test_frozen_protocol_loads_only_sibling_prompt_and_system_overlays(tmp_path: Path):
    frozen = tmp_path / "campaign" / "frozen"
    prompts = frozen / "prompts"
    systems = frozen / "systems"
    prompts.mkdir(parents=True)
    systems.mkdir()
    protocol_path = frozen / "protocol.yaml"
    protocol_path.write_bytes((CONFIG_ROOT / "protocol.yaml").read_bytes())
    prompt_path = prompts / "parameter_provider_v1.txt"
    prompt_path.write_bytes(
        (CONFIG_ROOT / "prompts" / "parameter_provider_v1.txt").read_bytes()
    )
    for system_id in ("e0", "e1", "e2", "e3"):
        (systems / f"{system_id}.yaml").write_bytes(
            (CONFIG_ROOT / "systems" / f"{system_id}.yaml").read_bytes()
        )

    bundle = load_protocol_bundle(protocol_path, campaign_id="frozen-campaign")
    assert bundle.protocol.provider_prompt_sha256 == hashlib.sha256(
        prompt_path.read_bytes()
    ).hexdigest()
    assert tuple(system.system_id.value for system in bundle.systems) == (
        "E0",
        "E1",
        "E2",
        "E3",
    )

    prompt_path.unlink()
    with pytest.raises(FileNotFoundError, match="campaign closure"):
        load_protocol_bundle(protocol_path, campaign_id="frozen-campaign")


def test_final_template_cannot_authorize_locked_access():
    pilot_protocol = _yaml(CONFIG_ROOT / "protocol.yaml")
    pilot_campaign = _yaml(CONFIG_ROOT / "pilot_webarena.yaml")
    _validate_protocol_access_boundary(
        pilot_protocol, pilot_campaign, pilot_only=True
    )

    missing_manual_guard = deepcopy(pilot_campaign)
    missing_manual_guard.pop("manual_rescue")
    with pytest.raises(SchemaError, match="manual_rescue=forbidden"):
        _validate_protocol_access_boundary(
            pilot_protocol, missing_manual_guard, pilot_only=True
        )

    weakened_manual_guard = deepcopy(pilot_protocol)
    weakened_manual_guard["manual_rescue"]["policy"] = "operator_allowed"
    with pytest.raises(SchemaError, match="manual rescue"):
        _validate_protocol_access_boundary(
            weakened_manual_guard, pilot_campaign, pilot_only=True
        )

    unsafe_pilot = deepcopy(pilot_protocol)
    unsafe_pilot["benchmark"]["allow_locked_reads"] = True
    with pytest.raises(SchemaError, match="allow_locked_reads=false"):
        _validate_protocol_access_boundary(
            unsafe_pilot, pilot_campaign, pilot_only=True
        )

    final_protocol = _yaml(CONFIG_ROOT / "protocol_final.yaml")
    validate_frozen_protocol_mapping(final_protocol)
    final_campaign = deepcopy(pilot_campaign)
    final_campaign.update(
        {
            "campaign_kind": "locked_final",
            "evidence_label": "FINAL_LOCKED",
            "locked_test_access": "single_frozen_campaign",
        }
    )
    with pytest.raises(SchemaError, match="final template cannot authorize"):
        _validate_protocol_access_boundary(
            final_protocol, final_campaign, pilot_only=False
        )
    assert final_protocol["benchmark"]["allow_locked_reads"] is False

    with pytest.raises(SchemaError, match="locked/final tasks"):
        _validate_task_boundary(
            [{"task_id": "locked-task-1", "partition": "locked_test"}],
            pilot_campaign,
            pilot_protocol,
            metadata={},
            pilot_only=True,
        )

def test_selection_profiles_cannot_cross_pilot_and_final_boundaries():
    pilot = _yaml(CONFIG_ROOT / "protocol.yaml")
    final = _yaml(CONFIG_ROOT / "protocol_final.yaml")
    pilot_campaign = _yaml(CONFIG_ROOT / "pilot_webarena.yaml")
    assert REGISTERED_FINAL_CANDIDATE_IDS == CANDIDATE_MODEL_IDS
    assert pilot_campaign["protocol"] == "configs/eval/table2/protocol.yaml"
    assert final["selection"] == {
        "mode": REGISTERED_FINAL_SELECTION_MODE,
        "candidate_ids": list(REGISTERED_FINAL_CANDIDATE_IDS),
        "required_candidate_count": 3,
        "checkpoint_selection": "validation_only",
        "locked_task_eligible": False,
    }
    assert final["protocol_status"] == "AWAITING_MODEL_PROMOTION"
    assert final["evidence_label"] == "FINAL_TEMPLATE_ONLY"
    assert final["paper_table_status"] == "N/R"
    validate_frozen_protocol_mapping(pilot)
    validate_frozen_protocol_mapping(final)

    shared_sections = set(pilot) - {
        "protocol_id",
        "protocol_status",
        "evidence_label",
        "selection",
        "benchmark",
    }
    assert shared_sections == set(final) - {
        "protocol_id",
        "protocol_status",
        "evidence_label",
        "selection",
        "benchmark",
    }
    assert all(pilot[key] == final[key] for key in shared_sections)
    assert pilot["benchmark"] == {
        **final["benchmark"],
        "task_manifest": "benchmarks/table2/pilot/task_manifest.json",
    }
    assert final["benchmark"]["task_manifest"] == (
        "locked_benchmark_mount/table2/final/task_manifest.json"
    )

    final_bundle = load_protocol_bundle(
        CONFIG_ROOT / "protocol_final.yaml",
        campaign_id="final-protocol-schema-test",
    )
    assert final_bundle.protocol.metadata["selection_mode"] == (
        REGISTERED_FINAL_SELECTION_MODE
    )

    relabeled_pilot = deepcopy(pilot)
    relabeled_pilot.update(
        {
            "protocol_id": "table2-final-template-v1",
            "protocol_status": "AWAITING_MODEL_PROMOTION",
            "evidence_label": "FINAL_TEMPLATE_ONLY",
        }
    )
    with pytest.raises(ValueError, match="selection.mode"):
        validate_frozen_protocol_mapping(relabeled_pilot)

    missing_candidate = deepcopy(final)
    missing_candidate["selection"]["candidate_ids"].pop()
    with pytest.raises(ValueError, match="candidate gate"):
        validate_frozen_protocol_mapping(missing_candidate)


def test_e0_e3_overlays_match_the_registered_mechanism_switches():
    expected = {
        "E0": (False, False, False, False, False, False),
        "E1": (True, False, False, False, False, False),
        "E2": (True, True, True, False, False, False),
        "E3": (True, True, True, True, True, False),
    }

    for system_id, registered in expected.items():
        overlay = _yaml(CONFIG_ROOT / "systems" / f"{system_id.lower()}.yaml")
        features = overlay["features"]
        actual = (
            features["trained_pre_action_policy"],
            features["post_action_diagnosis"],
            features["recovery_controller"],
            features["memory_query"],
            features["memory_intervention"],
            features["evaluation_memory_write"],
        )
        assert actual == registered

        runtime = switches_for(SystemID(system_id))
        assert (
            runtime.trained_pre_action_policy,
            runtime.post_action_diagnosis,
            runtime.recovery_controller,
            runtime.memory_query,
            runtime.memory_intervention,
            runtime.evaluation_memory_write,
        ) == registered
        assert runtime.evaluation_memory_write is False


def test_package_validator_rejects_legacy_pillar_overlay_keys():
    legacy = _yaml(CONFIG_ROOT / "systems" / "e1.yaml")
    legacy["features"] = {
        "pillar1_failure_recovery": False,
        "pillar2_action_type": True,
        "pillar3_grounding": True,
        "pillar4_memory": False,
    }
    with pytest.raises(SchemaError, match="registered mechanism matrix"):
        _validate_system_overlay(legacy, "E1", "legacy-e1.yaml")


def test_pilot_manifests_register_exact_counts_without_locked_content():
    tasks = _json(PILOT_ROOT / "task_manifest.json")
    recovery = _json(PILOT_ROOT / "recovery_scenarios.json")
    audit = _json(PILOT_ROOT / "audit_manifest.json")

    task_ids = [row["task_id"] for row in tasks["tasks"]]
    scenario_ids = [row["scenario_id"] for row in recovery["scenarios"]]
    assert len(task_ids) == tasks["required_task_count"] == 50
    assert len(set(task_ids)) == 50
    assert len(scenario_ids) == recovery["required_scenario_count"] == 15
    assert len(set(scenario_ids)) == 15
    assert tasks["final_paper_evaluation_eligible"] is False
    assert tasks["locked_test_content"] is False
    assert recovery["final_paper_evaluation_eligible"] is False
    assert sum(row["target"] for row in audit["strata"]) == audit["total_target"] == 20
    codebook = audit["reviewer_codebook"]
    assert audit["blinding_mode"] == (
        "OUTCOME_LABELS_HIDDEN_SYSTEM_CONDITION_VISIBLE"
    )
    assert "blinded" not in audit
    assert codebook["schema_version"] == "table2-manual-audit-reviewer-codebook-v2"
    assert codebook["label_vector_order"] == [
        field["name"] for field in codebook["fields"]
    ]
    assert codebook["agreement_method"] == (
        "unweighted_cohen_kappa_over_complete_ordered_label_vectors_v1"
    )
    assert codebook["composite_exact_agreement_method"] == (
        "complete_ordered_label_vector_exact_agreement_v1"
    )
    assert codebook["per_field_agreement_method"] == (
        "unweighted_cohen_kappa_per_registered_field_v1"
    )
    assert codebook["final_vs_sealed_comparison_method"] == (
        "adjudicated_labels_vs_selected_sealed_evidence_counts_v1"
    )
    assert codebook["applicability_rules"] == [
        (
            "NO_RECOVERY_ATTEMPT_if_and_only_if_selected_evidence_has_no_"
            "executed_recovery"
        ),
        (
            "NOT_APPLICABLE_memory_effect_if_and_only_if_no_E3_admitted_memory_"
            "intervention_with_paired_E2_evidence"
        ),
        (
            "NO_INTERVENTION_if_and_only_if_selected_evidence_has_neither_"
            "executed_recovery_nor_admitted_memory_intervention"
        ),
    ]
    assert codebook["undefined_kappa"] == {
        "status": "UNDEFINED",
        "reason": "EXPECTED_AGREEMENT_EQUALS_ONE",
        "condition": "expected_chance_agreement_equals_one",
    }


def test_raw_artifact_and_locked_mount_rules_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    required = {
        "/artifacts/",
        "/outputs/",
        "/locked_benchmark_mount/",
        "/browser_profiles/",
        "*.har",
        "*.faiss",
        "*.embeddings.npy",
    }
    assert required.issubset(set(ignore))
