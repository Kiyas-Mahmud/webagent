from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

import pytest
import yaml

from web_agent.runtime.checkpoint_inference import (
    LoadedSelectedBackboneBackend,
    LoadedSelectedCheckpointBackend,
    SelectedBackbonePolicyAdapter,
    SelectedCheckpointError,
    SelectedCheckpointPolicyAdapter,
    ValidationSelectedBackbone,
    ValidationSelectedCheckpoint,
)
from web_agent.runtime.observation import ProcessorParityContract
from web_agent.runtime.contracts import (
    MemoryCandidate,
    MemoryQuery,
    RecoveryDecision,
    RecoveryStrategy,
    SystemID,
)
from web_agent.runtime.memory_adapter import (
    FrozenMemoryReader,
    MemoryAdapter,
    MemoryBoundaryError,
    ReaderQueryResult,
)
from web_agent.runtime.protocol import (
    switches_for,
    validate_frozen_protocol_mapping,
)


ROOT = Path(__file__).resolve().parents[2]


def test_protocol_mapping_rejects_every_newly_locked_semantic():
    source = yaml.safe_load(
        (ROOT / "configs" / "eval" / "table2" / "protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_frozen_protocol_mapping(source)
    truth_table_with_false_case_enabled = dict(
        source["recovery_trigger"]["policy_truth_table"]
    )
    truth_table_with_false_case_enabled["0000"] = True
    mutations = (
        (("evidence_label",), "FINAL_LOCKED"),
        (("paper_table_status",), "READY"),
        (("benchmark", "name"), "visualwebarena"),
        (("benchmark", "adapter"), "other.Adapter"),
        (("benchmark", "task_manifest"), "benchmarks/other.json"),
        (("benchmark", "locked_mount"), "another_locked_mount"),
        (("benchmark", "allow_locked_reads"), True),
        (("benchmark", "invalidate_if_successful_after_reset"), False),
        (
            ("causal_boundary", "pre_action_inputs"),
            ["task_instruction", "causal_history", "current_observation"],
        ),
        (
            ("causal_boundary", "post_action_inputs"),
            ["pre_action_observation", "post_action_observation"],
        ),
        (
            ("causal_boundary", "forbidden_runtime_inputs"),
            ["oracle_success", "oracle_failure"],
        ),
        (("causal_boundary", "sealed_verifier_returns"), "full_evidence"),
        (
            ("recovery_trigger", "failure_probability_threshold"),
            0.6,
        ),
        (
            ("recovery_trigger", "needs_recovery_probability_threshold"),
            0.6,
        ),
        (("recovery_trigger", "probability_comparator"), "greater_than"),
        (
            ("recovery_trigger", "policy_signal_order"),
            [
                "needs_recovery",
                "failure_probability_at_or_above_threshold",
                "predicted_failure",
                "needs_recovery_probability_at_or_above_threshold",
            ],
        ),
        (
            ("recovery_trigger", "policy_truth_table"),
            truth_table_with_false_case_enabled,
        ),
        (("recovery_trigger", "executor_status_rule"), "ignore_rejection"),
        (("recovery_trigger", "loop_rule"), "ignore_loop"),
        (("budgets", "recovery_at_k"), 3),
        (("budgets", "whole_block_infrastructure_reruns"), 2),
        (("parameter_provider", "mode"), "model_only"),
        (("parameter_provider", "prompt"), "prompts/unregistered.txt"),
        (("parameter_provider", "invalid_output_policy"), "repair"),
        (
            ("parameter_provider", "decoding_parameters"),
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 128,
            },
        ),
        (("memory", "embedding_dimension"), 512),
        (("memory", "normalization"), "none"),
        (("memory", "top_k"), 5),
        (("memory", "tie_break"), "random"),
        (("memory", "admission_threshold_source"), "validation"),
        (("memory", "duplicate_exclusion"), False),
        (("verification", "final_success_source"), "self_report"),
        (
            ("verification", "recovery_evidence_sources"),
            ["preregistered_deterministic_scenarios"],
        ),
        (("verification", "headline_recovery_partition"), "controlled"),
        (("verification", "controlled_recovery_role"), "headline"),
        (("randomness", "algorithm"), "python_random"),
        (("randomness", "namespace_fields"), ["protocol_id"]),
        (
            ("randomness", "key_fields"),
            [
                "campaign_id",
                "task_id",
                "repeat_id",
                "matched_seed",
                "stage",
                "decision_index",
            ],
        ),
        (("randomness", "system_id_in_key"), True),
        (("artifacts", "root"), "other-artifacts"),
        (("artifacts", "paired_block_first"), False),
        (("artifacts", "append_only_jsonl"), False),
        (("artifacts", "hash_algorithm"), "md5"),
        (("artifacts", "raw_artifacts_in_git"), "allowed"),
        (("artifacts", "result_root"), "other-results"),
        (("statistics", "bootstrap_samples"), 100),
        (("statistics", "bootstrap_unit"), "episode_id"),
        (("statistics", "seed"), 42),
    )
    for path, value in mutations:
        changed = json.loads(json.dumps(source))
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError):
            validate_frozen_protocol_mapping(changed)

    changed = json.loads(json.dumps(source))
    changed["recovery_trigger"]["unregistered_signal"] = True
    with pytest.raises(ValueError, match="exactly the registered keys"):
        validate_frozen_protocol_mapping(changed)


class _BadOrderReader(FrozenMemoryReader):
    reader_id = "bad-order"
    index_sha256 = "a" * 64
    model_seed = 42
    registered_admission_threshold = 0.5
    frozen = True
    write_enabled = False

    def __init__(self, candidates):
        self.candidates = tuple(candidates)

    def query(self, query, *, k, rng):
        del query, k, rng
        return ReaderQueryResult(
            candidates=self.candidates,
            exclusion_reasons={},
            considered_count=len(self.candidates),
            eligible_count=len(self.candidates),
        )


def _candidate(memory_id: str, score: float) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        source_split="train",
        source_task_id=f"train-{memory_id}",
        source_episode_id=f"episode-{memory_id}",
        duplicate_cluster_id=f"cluster-{memory_id}",
        strategy=RecoveryStrategy.RETRY,
        similarity=score,
        memory_update_flag=True,
        verified_recovery_success=True,
        final_task_success=True,
    )


def _apply(reader: FrozenMemoryReader):
    adapter = MemoryAdapter(
        switches_for(SystemID.E3),
        reader=reader,
        admission_threshold=0.5,
    )
    return adapter.apply_post_failure(
        query=MemoryQuery(
            query_id="query",
            task_id="eval-task",
            episode_id="eval-episode",
            incident_id="incident",
            post_failure_observation_id="post",
            failed_action_id="failed",
            diagnosis="NO_EFFECT",
            duplicate_cluster_ids=("eval-cluster",),
        ),
        shadow_decision=RecoveryDecision(
            decision_id="shadow",
            incident_id="incident",
            strategy=RecoveryStrategy.BACKTRACK,
            trigger_sources=("policy",),
            diagnosis="NO_EFFECT",
        ),
        rng=random.Random(42),
    )


def test_memory_requires_exact_top3_and_deterministic_unique_reader_order():
    with pytest.raises(ValueError, match="registered as 3"):
        MemoryAdapter(
            switches_for(SystemID.E3),
            reader=_BadOrderReader(()),
            top_k=2,
            admission_threshold=0.5,
        )
    with pytest.raises(MemoryBoundaryError, match="ordering"):
        _apply(_BadOrderReader((_candidate("low", 0.6), _candidate("high", 0.9))))
    duplicate = _candidate("same", 0.9)
    with pytest.raises(MemoryBoundaryError, match="duplicate candidate"):
        _apply(_BadOrderReader((duplicate, replace(duplicate, similarity=0.8))))


def test_validation_selected_checkpoint_rejects_nonvalidation_or_test_reads(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"frozen")
    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base = {
        "model_seed": 42,
        "selected_checkpoint_sha256": digest,
        "resolved_config_sha256": "b" * 64,
        "processor_contract_sha256": "c" * 64,
        "selection_scope": "validation_only",
        "checkpoint_selection": "validation_only",
        "validation_rows_read": 100,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    selection = ValidationSelectedCheckpoint.from_mapping(
        base,
        checkpoint_path=checkpoint,
    )
    selection.verify_checkpoint()
    with pytest.raises(SelectedCheckpointError):
        ValidationSelectedCheckpoint.from_mapping(
            {**base, "test_rows_read": 1},
            checkpoint_path=checkpoint,
        )
    with pytest.raises(SelectedCheckpointError):
        ValidationSelectedCheckpoint.from_mapping(
            {**base, "selection_scope": "test"},
            checkpoint_path=checkpoint,
        )


def _processor() -> ProcessorParityContract:
    return ProcessorParityContract(
        processor_class="FixtureProcessor",
        processor_revision="rev-1",
        processor_config_sha256="d" * 64,
        pre_action_field_mapping={"image": "pre_image"},
        post_action_field_mapping={"image": "post_image"},
    )


def test_loaded_checkpoint_must_match_frozen_resolved_config(tmp_path):
    import hashlib

    checkpoint = tmp_path / "selected.bin"
    checkpoint.write_bytes(b"selected")
    processor = _processor()
    selection = ValidationSelectedCheckpoint(
        manifest_id="selected-seed-42",
        model_seed=42,
        checkpoint_path=checkpoint,
        selected_checkpoint_sha256=hashlib.sha256(b"selected").hexdigest(),
        resolved_config_sha256="a" * 64,
        processor_contract_sha256=processor.record_sha256,
        validation_rows_read=100,
    )
    backend = LoadedSelectedCheckpointBackend(
        checkpoint_sha256=selection.selected_checkpoint_sha256,
        resolved_config_sha256="b" * 64,
        processor_contract=processor,
        action_predictor=lambda *args: None,
        transition_predictor=lambda *args: None,
        recovery_predictor=lambda *args: None,
        memory_embedding=lambda request: None,
    )
    adapter = SelectedCheckpointPolicyAdapter(
        selection=selection,
        backend_factory=lambda _: backend,
        runtime_processor_contract=processor,
    )
    with pytest.raises(SelectedCheckpointError, match="configuration"):
        adapter._load()


def test_e0_selected_backbone_rejects_adaptation_or_task_heads(tmp_path):
    import hashlib

    backbone = tmp_path / "base.bin"
    backbone.write_bytes(b"unadapted-base")
    processor = _processor()
    selection = ValidationSelectedBackbone(
        manifest_id="selected-backbone",
        backbone_id="winner-family",
        backbone_revision="pinned-revision",
        backbone_path=backbone,
        backbone_sha256=hashlib.sha256(b"unadapted-base").hexdigest(),
        resolved_config_sha256="a" * 64,
        processor_contract_sha256=processor.record_sha256,
        base_prompt_sha256="b" * 64,
        parser_id="e0-json-parser",
        parser_version="v1",
        validation_rows_read=100,
    )
    with pytest.raises(SelectedCheckpointError, match="unadapted"):
        LoadedSelectedBackboneBackend(
            backbone_id=selection.backbone_id,
            backbone_revision=selection.backbone_revision,
            backbone_sha256=selection.backbone_sha256,
            resolved_config_sha256=selection.resolved_config_sha256,
            processor_contract=processor,
            base_prompt_sha256=selection.base_prompt_sha256,
            parser_id=selection.parser_id,
            parser_version=selection.parser_version,
            action_predictor=lambda *args: None,
            adaptation_loaded=True,
        )
