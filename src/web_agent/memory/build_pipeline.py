"""Build one immutable, train-only Table 2 corrective-memory store.

This command never reads validation or test splits.  It requires a separately
frozen provenance manifest that supplies final-task-success and duplicate-cluster
evidence for every potentially admitted source sample.  The repository script
is a thin entrypoint; checkpoint/data orchestration lives in this module.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from web_agent.eval.table2.resolved_config import (
    ResolvedConfigIdentity,
    assert_checkpoint_config_matches,
    load_resolved_config_identity,
)
from web_agent.memory import (
    ProvenanceManifest,
    build_frozen_store,
    select_eligible_candidates,
)
from web_agent.memory.calibration_builder import (
    CALIBRATION_EMBEDDING_HASH,
    CALIBRATION_EVIDENCE_SCHEMA_VERSION,
    CALIBRATION_PAIR_POLICY,
    CALIBRATION_RELEVANCE_DEFINITION,
    build_calibration_evidence,
    validate_calibration_evidence,
)
from web_agent.memory.manifest import (
    THRESHOLD_CALIBRATION_ADMISSION_RULE,
    THRESHOLD_CALIBRATION_CANDIDATES,
    THRESHOLD_CALIBRATION_METHOD,
    THRESHOLD_CALIBRATION_OBJECTIVE,
    THRESHOLD_CALIBRATION_ROWS_STORAGE,
    THRESHOLD_CALIBRATION_SCHEMA_VERSION,
    THRESHOLD_CALIBRATION_TIE_BREAK,
    canonical_sha256,
    sha256_file,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "exact fully materialized selected full-run config artifact; its "
            "canonical mapping must equal checkpoint['config']"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--supplement-root",
        type=Path,
        default=None,
        help="optional extracted RETRY/ABORT training supplement root",
    )
    parser.add_argument(
        "--review-overlay-dir",
        type=Path,
        default=None,
        help="passed train-only review/reconciliation overlay",
    )
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        required=True,
        help=(
            "table2-memory-provenance-v1 JSON with source_split=train, "
            "validation_rows_read=test_rows_read=locked_test_rows_read=0, "
            "dataset-artifact/logical-record hashes, and per-sample task/episode/"
            "duplicate evidence plus independently verified recovery and final-"
            "task evidence records with canonical digests"
        ),
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/eval/table2/protocol.yaml"),
    )
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def _public_value(value: Any) -> Any:
    """Remove loader-only private keys before hashing the logical row corpus."""
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_public_value(item) for item in value]
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"missing JSON manifest: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def _validate_protocol(path: Path) -> None:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("memory"), dict):
        raise ValueError("Table 2 protocol must contain a memory mapping")
    memory = payload["memory"]
    required = {
        "runtime_mode": "immutable_read_only",
        "source_split": "train",
        "embedding_stage": "post_action_memory_task_adapter",
        "embedding_dimension": 768,
        "normalization": "l2",
        "similarity": "cosine",
        "top_k": 3,
        "tie_break": "memory_id_ascending",
        "admission_threshold_source": "train_only_calibration",
        "same_task_exclusion": True,
        "duplicate_exclusion": True,
        "e3_writes": "forbidden",
    }
    mismatches = {
        key: {"expected": expected, "actual": memory.get(key)}
        for key, expected in required.items()
        if memory.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Table 2 memory protocol mismatch: {mismatches}")
    calibration = memory.get("threshold_calibration")
    registered_calibration = {
        "schema_version": THRESHOLD_CALIBRATION_SCHEMA_VERSION,
        "method": THRESHOLD_CALIBRATION_METHOD,
        "objective": THRESHOLD_CALIBRATION_OBJECTIVE,
        "tie_break": THRESHOLD_CALIBRATION_TIE_BREAK,
        "candidate_threshold_policy": THRESHOLD_CALIBRATION_CANDIDATES,
        "admission_rule": THRESHOLD_CALIBRATION_ADMISSION_RULE,
        "rows_storage": THRESHOLD_CALIBRATION_ROWS_STORAGE,
        "evidence_schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        "pair_policy": CALIBRATION_PAIR_POLICY,
        "relevance_definition": CALIBRATION_RELEVANCE_DEFINITION,
        "embedding_hash_algorithm": CALIBRATION_EMBEDDING_HASH,
    }
    if not isinstance(calibration, Mapping) or dict(calibration) != (
        registered_calibration
    ):
        raise ValueError(
            "Table 2 memory threshold-calibration protocol differs from "
            f"registration: expected {registered_calibration}, got {calibration!r}"
        )


def _validate_model_config(cfg: Mapping[str, Any]) -> None:
    if int(cfg.get("fused_dim", -1)) != 768:
        raise ValueError("Table 2 memory requires fused_dim=768")
    data = cfg.get("data", {})
    if not data.get("causal_routing") or not data.get("use_state_after"):
        raise ValueError("Table 2 memory requires causal post-action routing")
    adapters = cfg.get("model", {}).get("task_adapters", {})
    if not adapters.get("enabled"):
        raise ValueError("selected checkpoint must enable the memory task adapter")


def _load_checkpoint(
    model,
    path: Path,
    cfg: Mapping[str, Any],
    *,
    resolved_config_identity: ResolvedConfigIdentity,
) -> None:
    import torch
    from peft import set_peft_model_state_dict

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state mapping")
    required = {
        "lora",
        "adapter",
        "task_adapters",
        "failure",
        "action",
        "memory",
        "recovery_outcome",
        "config",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required state: {missing}")
    saved_cfg = checkpoint["config"]
    if not isinstance(saved_cfg, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    assert_checkpoint_config_matches(saved_cfg, resolved_config_identity)
    current_backbone = cfg.get("backbone", {})
    saved_backbone = saved_cfg.get("backbone", {})
    for field in ("family", "vlm_model", "revision", "path"):
        if saved_backbone.get(field) != current_backbone.get(field):
            raise ValueError(f"checkpoint/config backbone mismatch for {field}")
    if int(saved_cfg.get("fused_dim", -1)) != 768:
        raise ValueError("checkpoint was not trained with fused_dim=768")

    set_peft_model_state_dict(model.encoder.model, checkpoint["lora"])
    model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
    model.task_adapters.load_state_dict(checkpoint["task_adapters"], strict=True)
    model.failure_head.load_state_dict(checkpoint["failure"], strict=True)
    model.action_head.load_state_dict(checkpoint["action"], strict=True)
    model.memory_head.load_state_dict(checkpoint["memory"], strict=True)
    model.recovery_outcome_head.load_state_dict(
        checkpoint["recovery_outcome"], strict=True
    )


def main() -> None:
    args = parse_args()
    if args.model_seed < 0:
        raise ValueError("--model-seed must be non-negative")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {args.checkpoint}")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen memory output: {args.output_dir}"
        )

    _validate_protocol(args.protocol_config)
    resolved_config_identity = load_resolved_config_identity(args.config)
    # Runtime-only data locations are applied to a detached copy.  Neither
    # identity below may be derived from this mutable build configuration.
    cfg = deepcopy(resolved_config_identity.mapping)
    cfg["data"]["root"] = str(args.data_root.resolve())
    if args.review_overlay_dir is not None:
        cfg["data"]["review_overlay_dir"] = str(
            args.review_overlay_dir.resolve()
        )
    if args.supplement_root is not None:
        supplement = cfg["data"].setdefault("recovery_supplement", {})
        supplement.update({
            "enabled": True,
            "root": str(args.supplement_root.resolve()),
            "include_in_primary_validation": False,
        })
    _validate_model_config(cfg)

    # This loader call is intentionally train-only.  No validation/test helper is
    # imported or invoked anywhere in this command.
    from web_agent.data.gold_dataloader import (
        build_gold_dataloader,
        load_gold_split,
    )
    from web_agent.data.recovery_transitions import (
        build_recovery_transition_index,
    )

    records = load_gold_split(cfg, "train")
    records_sha256 = canonical_sha256(_public_value(records))
    provenance = ProvenanceManifest.from_mapping(
        _load_json_object(args.provenance_manifest)
    )
    if records_sha256 != provenance.records_sha256:
        raise ValueError(
            "logical training-record hash does not match provenance manifest: "
            f"expected {provenance.records_sha256}, found {records_sha256}"
        )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    protocol_sha256 = sha256_file(args.protocol_config)
    provenance_manifest_sha256 = sha256_file(args.provenance_manifest)

    transitions, transition_report = build_recovery_transition_index(records)
    # Production verification commits to state artifact bytes, not reusable path
    # strings. Supplement transitions retain their own declared root; ordinary
    # Gold transitions resolve beneath the selected train-data root.
    for transition in transitions.values():
        if not transition.get("_data_root"):
            transition["_data_root"] = str(args.data_root.resolve())
    selection = select_eligible_candidates(
        records,
        source_split="train",
        transitions=transitions,
        provenance=provenance,
    )

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("selected VLM memory embedding requires a CUDA device")
    from web_agent.models.model import WebAgentModel
    from web_agent.train.gold_stages import build_processor

    processor = build_processor(cfg)
    model = WebAgentModel(cfg)
    _load_checkpoint(
        model,
        args.checkpoint,
        cfg,
        resolved_config_identity=resolved_config_identity,
    )
    for module in (
        model.adapter,
        model.task_adapters,
        model.failure_head,
        model.action_head,
        model.memory_head,
        model.recovery_outcome_head,
    ):
        module.to("cuda")
    model.eval()

    selected_records = [
        records[candidate.record_index] for candidate in selection.candidates
    ]
    loader = build_gold_dataloader(
        cfg,
        "train",
        processor,
        records=selected_records,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.model_seed,
        recovery_aware=False,
        trajectory_records=records,
    )
    batches: list[np.ndarray] = []
    embedded_rows = 0
    for batch in loader:
        tensor = model.memory_embedding(batch).float().cpu().numpy()
        batches.append(tensor)
        embedded_rows += int(tensor.shape[0])
    if embedded_rows != len(selection.candidates):
        raise RuntimeError(
            "memory embedding row count changed during loading: "
            f"expected {len(selection.candidates)}, found {embedded_rows}"
        )
    embeddings = np.concatenate(batches, axis=0)

    calibration_evidence = build_calibration_evidence(
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=provenance_manifest_sha256,
        model_seed=args.model_seed,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=records_sha256,
        dataset_artifacts_sha256=provenance.dataset_artifacts_sha256,
        resolved_config_sha256=resolved_config_identity.payload_sha256,
        resolved_config_record_sha256=resolved_config_identity.record_sha256,
        protocol_sha256=protocol_sha256,
    )
    verified_calibration = validate_calibration_evidence(
        calibration_evidence,
        selection=selection,
        embeddings=embeddings,
        provenance=provenance,
        provenance_manifest_sha256=provenance_manifest_sha256,
        model_seed=args.model_seed,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=records_sha256,
        dataset_artifacts_sha256=provenance.dataset_artifacts_sha256,
        resolved_config_sha256=resolved_config_identity.payload_sha256,
        resolved_config_record_sha256=resolved_config_identity.record_sha256,
        protocol_sha256=protocol_sha256,
    )
    calibration = dict(verified_calibration.threshold_calibration)

    store = build_frozen_store(
        args.output_dir,
        selection=selection,
        embeddings=embeddings,
        model_seed=args.model_seed,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=records_sha256,
        dataset_id=provenance.dataset_id,
        dataset_version=provenance.dataset_version,
        dataset_artifacts_sha256=provenance.dataset_artifacts_sha256,
        resolved_config_sha256=resolved_config_identity.payload_sha256,
        resolved_config_record_sha256=resolved_config_identity.record_sha256,
        protocol_sha256=protocol_sha256,
        provenance_manifest_sha256=provenance_manifest_sha256,
        threshold_calibration=calibration,
        calibration_evidence=calibration_evidence,
        transition_report=transition_report,
    )
    print(json.dumps({
        "status": "PASS",
        "store": str(store.root),
        "store_id": store.manifest["store_id"],
        "manifest_sha256": store.manifest_sha256,
        "model_seed": store.model_seed,
        "items": len(store),
        "admission_threshold": calibration["admission_threshold"],
        "calibration_evidence_sha256": (
            verified_calibration.evidence_sha256
        ),
        "calibration_queries": calibration_evidence["query_count"],
        "calibration_retrieved_pairs": calibration_evidence[
            "calibration_sample_count"
        ],
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "runtime_writes_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
