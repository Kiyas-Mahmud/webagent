"""Build one immutable, train-only Table 2 corrective-memory store.

This command never reads validation or test splits. It requires a separately
frozen provenance manifest plus an authenticated compatible WebArena task export
and a completed joint train/task duplicate audit before checkpoint or model
access. The repository script is a thin entrypoint; checkpoint/data
orchestration lives in this module.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from web_agent.eval.table2.resolved_config import (
    ResolvedConfigIdentity,
    assert_checkpoint_config_matches,
    load_resolved_config_identity,
)
from web_agent.eval.table2.task_interface_audit import (
    require_webarena_task_interface_compatible,
    validate_webarena_task_interface_audit,
)
from web_agent.eval.table2.webarena_export import (
    PINNED_TASK_SOURCE_SHA256,
    load_url_map,
    validate_public_pilot_task_export,
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
    require_sha256,
    sha256_file,
)
from web_agent.memory.joint_duplicate_audit import (
    validate_final_joint_duplicate_audit,
    validate_joint_duplicate_assignment_package,
)
from web_agent.memory.preparation import (
    P4_REGISTERED_SOURCE_AUTHORITY_SHA256,
    reconstruct_p4_selection_from_preparation,
)
from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    FrozenDuplicateAuditManifest,
    canonical_train_corpus_binding,
    duplicate_audit_task_content_sha256,
)


_PILOT_TASK_COUNT = 50
_FORBIDDEN_READ_FIELDS = (
    "validation_rows_read",
    "test_rows_read",
    "locked_test_rows_read",
)


@dataclass(frozen=True)
class P4BuildPrerequisiteValidation:
    """Validated external evidence required before any P4 model work."""

    provenance: ProvenanceManifest
    provenance_manifest_sha256: str
    resolved_task_export_sha256: str
    task_interface_audit_sha256: str
    duplicate_audit_sha256: str
    train_corpus_binding_sha256: str
    duplicate_cluster_namespace_id: str
    verified_task_count: int
    preparation_manifest_sha256: str | None = None
    joint_assignment_manifest_sha256: str | None = None
    joint_assignment_entities_sha256: str | None = None
    joint_assignment_clusters_sha256: str | None = None
    joint_audit_config_sha256: str | None = None
    joint_audit_source_sha256: str | None = None
    recovery_scenarios_sha256: str | None = None
    duplicate_audit_registration_sha256: str | None = None
    source_authority_sha256: str | None = None

    def report(self) -> dict[str, Any]:
        report = {
            "status": "PASS",
            "evidence_role": "VALIDATION_ONLY_EXTERNAL_EVIDENCE_NOT_AUTHORED",
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "resolved_task_export_sha256": self.resolved_task_export_sha256,
            "task_interface_audit_sha256": self.task_interface_audit_sha256,
            "duplicate_audit_sha256": self.duplicate_audit_sha256,
            "train_corpus_binding_sha256": self.train_corpus_binding_sha256,
            "duplicate_cluster_namespace_id": (
                self.duplicate_cluster_namespace_id
            ),
            "verified_task_count": self.verified_task_count,
            "validation_rows_read": self.provenance.validation_rows_read,
            "test_rows_read": self.provenance.test_rows_read,
            "locked_test_rows_read": self.provenance.locked_test_rows_read,
        }
        registered = self.registered_duplicate_evidence_binding()
        if registered is not None:
            report["registered_joint_duplicate_evidence"] = registered
        return report

    def registered_duplicate_evidence_binding(self) -> dict[str, Any] | None:
        values = {
            "preparation_manifest_sha256": self.preparation_manifest_sha256,
            "assignment_manifest_sha256": self.joint_assignment_manifest_sha256,
            "entities_sha256": self.joint_assignment_entities_sha256,
            "clusters_sha256": self.joint_assignment_clusters_sha256,
            "audit_config_sha256": self.joint_audit_config_sha256,
            "audit_source_sha256": self.joint_audit_source_sha256,
            "recovery_scenarios_sha256": self.recovery_scenarios_sha256,
            "duplicate_audit_registration_sha256": (
                self.duplicate_audit_registration_sha256
            ),
            "source_authority_sha256": self.source_authority_sha256,
        }
        if all(value is None for value in values.values()):
            return None
        if any(value is None for value in values.values()):
            raise ValueError("registered joint duplicate-evidence binding is partial")
        return {
            "schema_version": "table2-memory-joint-duplicate-evidence-binding-v2",
            **values,
            "final_duplicate_audit_sha256": self.duplicate_audit_sha256,
            "provenance_manifest_sha256": self.provenance_manifest_sha256,
            "duplicate_cluster_namespace": (
                self.provenance.duplicate_cluster_namespace.to_dict()
            ),
        }


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
        "--resolved-task-export",
        type=Path,
        required=True,
        help=(
            "authenticated, fully resolved 50-task WebArena pilot export; it is "
            "reconstructed from the separately supplied source, registry, and URL map"
        ),
    )
    parser.add_argument(
        "--webarena-task-source",
        type=Path,
        required=True,
        help="pinned libwebarena wheel or separately authorized raw task JSON",
    )
    parser.add_argument(
        "--webarena-task-registry",
        type=Path,
        required=True,
        help="user-approved and preregistered pilot task exclusion registry",
    )
    parser.add_argument(
        "--webarena-site-url-map",
        type=Path,
        required=True,
        help="credential-free URL-token map used to reproduce the resolved task export",
    )
    parser.add_argument(
        "--authorized-raw-task-source-sha256",
        default=None,
        help=(
            "required only when --webarena-task-source is a separately extracted "
            "raw JSON file; the content must still match the pinned task-source hash"
        ),
    )
    parser.add_argument(
        "--task-interface-audit",
        type=Path,
        required=True,
        help="exact recomputable PASS audit for all resolved pilot tasks",
    )
    parser.add_argument(
        "--duplicate-audit",
        type=Path,
        required=True,
        help=(
            "completed joint Gold-train/WebArena audit with canonical VERIFIED "
            "records for every resolved pilot task"
        ),
    )
    parser.add_argument(
        "--joint-assignment-package",
        type=Path,
        required=True,
        help="immutable assignment package from run_table2_joint_duplicate_audit.py",
    )
    parser.add_argument(
        "--joint-audit-config",
        type=Path,
        required=True,
        help="canonical frozen joint_duplicate_audit_v1.json",
    )
    parser.add_argument(
        "--p4-preparation-package",
        type=Path,
        required=True,
        help="exact train-only P4 preparation package used for assignments",
    )
    parser.add_argument(
        "--p4-source-authority",
        type=Path,
        required=True,
        help="tracked PC-01 train-source authority with exact hashes/counts",
    )
    parser.add_argument(
        "--recovery-scenarios",
        type=Path,
        required=True,
        help="registered 15-scenario diagnostic manifest",
    )
    parser.add_argument(
        "--duplicate-audit-registration",
        type=Path,
        required=True,
        help="canonical pilot registration supplying diagnostic-only rows",
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


def _load_nonsymlink_json(path: Path, *, artifact: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{artifact} must not be a symlink")
    return _load_json_object(path)


def _validate_pilot_export_boundary(task_export: Mapping[str, Any]) -> None:
    expected = {
        "benchmark": "webarena",
        "partition": "development",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "locked_test_content": False,
        "final_paper_evaluation_eligible": False,
        "required_task_count": _PILOT_TASK_COUNT,
    }
    mismatches = {
        key: {"expected": value, "actual": task_export.get(key)}
        for key, value in expected.items()
        if task_export.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "resolved WebArena pilot export boundary mismatch: "
            f"{mismatches}"
        )
    rows = task_export.get("tasks")
    if not isinstance(rows, list) or len(rows) != _PILOT_TASK_COUNT:
        raise ValueError("resolved WebArena pilot export must contain exactly 50 tasks")
    task_ids = [str(row.get("task_id") or "").strip() for row in rows]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(
        set(task_ids)
    ):
        raise ValueError("resolved WebArena pilot task IDs must be non-empty and unique")
    indices = [row.get("upstream_index") for row in rows]
    if indices != list(range(_PILOT_TASK_COUNT)):
        raise ValueError(
            "resolved WebArena pilot tasks must remain in approved index order 0--49"
        )
    require_sha256(
        task_export.get("resolved_task_set_sha256"),
        field="resolved_task_export.resolved_task_set_sha256",
    )
    if task_export["resolved_task_set_sha256"] != canonical_sha256(rows):
        raise ValueError("resolved WebArena pilot task-set hash mismatch")
    for field in ("site_url_map_sha256", "registry_manifest_sha256"):
        require_sha256(
            task_export.get(field),
            field=f"resolved_task_export.{field}",
        )
    source = task_export.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("resolved WebArena pilot export lacks source authority")
    for field in (
        "container_sha256",
        "authorized_container_sha256",
        "task_source_sha256",
    ):
        require_sha256(source.get(field), field=f"resolved_task_export.source.{field}")


def validate_p4_build_prerequisites(
    *,
    provenance_manifest_path: str | Path,
    resolved_task_export_path: str | Path,
    webarena_task_source_path: str | Path,
    webarena_task_registry_path: str | Path,
    webarena_site_url_map_path: str | Path,
    task_interface_audit_path: str | Path,
    duplicate_audit_path: str | Path,
    authorized_raw_task_source_sha256: str | None = None,
    expected_task_source_sha256: str = PINNED_TASK_SOURCE_SHA256,
) -> P4BuildPrerequisiteValidation:
    """Validate externally produced task/audit evidence without authoring it.

    The source bytes, task registry and URL map are reopened so a hand-written
    export cannot authorize a build merely by copying expected hash strings.
    The returned train-corpus digest is the same commitment checked again at
    campaign handoff after the store exists.
    """

    provenance_path = Path(provenance_manifest_path)
    export_path = Path(resolved_task_export_path)
    task_source_path = Path(webarena_task_source_path)
    registry_path = Path(webarena_task_registry_path)
    url_map_path = Path(webarena_site_url_map_path)
    interface_path = Path(task_interface_audit_path)
    duplicate_path = Path(duplicate_audit_path)

    provenance_payload = _load_nonsymlink_json(
        provenance_path,
        artifact="memory provenance manifest",
    )
    provenance = ProvenanceManifest.from_mapping(provenance_payload)
    provenance_sha256 = sha256_file(provenance_path)

    submitted_export = _load_nonsymlink_json(
        export_path,
        artifact="resolved WebArena task export",
    )
    _load_nonsymlink_json(registry_path, artifact="WebArena task registry")
    _load_nonsymlink_json(url_map_path, artifact="WebArena site URL map")
    task_export = validate_public_pilot_task_export(
        submitted_export,
        source=task_source_path,
        registry_path=registry_path,
        site_url_map=load_url_map(url_map_path),
        expected_source_sha256=expected_task_source_sha256,
        authorized_raw_json_sha256=authorized_raw_task_source_sha256,
    )
    _validate_pilot_export_boundary(task_export)
    task_rows = task_export["tasks"]
    task_ids = [str(row["task_id"]) for row in task_rows]

    submitted_interface_audit = _load_nonsymlink_json(
        interface_path,
        artifact="WebArena task-interface audit",
    )
    interface_audit = validate_webarena_task_interface_audit(
        submitted_interface_audit,
        task_export=task_export,
    )
    require_webarena_task_interface_compatible(interface_audit)
    if (
        interface_audit.get("task_count") != _PILOT_TASK_COUNT
        or interface_audit.get("compatible_task_count") != _PILOT_TASK_COUNT
        or interface_audit.get("incompatible_task_count") != 0
    ):
        raise ValueError(
            "P4 build requires exactly 50 compatible WebArena pilot tasks"
        )

    duplicate_payload = _load_nonsymlink_json(
        duplicate_path,
        artifact="joint duplicate-audit manifest",
    )
    for field in _FORBIDDEN_READ_FIELDS:
        if type(duplicate_payload.get(field)) is not int or duplicate_payload[field] != 0:
            raise ValueError(
                "joint duplicate audit requires explicit "
                f"{field}=0"
            )
    required_top_level = {
        "manifest_state": "FROZEN_REGISTRATION",
        "evidence_label": "PILOT_ONLY",
        "normal_task_evidence_status": "VERIFIED",
        "normal_task_runtime_policy": "VERIFIED_NONEMPTY_CLUSTERS_REQUIRED",
        "required_normal_task_count": _PILOT_TASK_COUNT,
    }
    for field, expected in required_top_level.items():
        if duplicate_payload.get(field) != expected:
            raise ValueError(
                "joint duplicate audit is not a completed pilot artifact at "
                f"{field}"
            )
    try:
        duplicate_manifest = FrozenDuplicateAuditManifest.from_path(duplicate_path)
    except DuplicateAuditError as error:
        raise ValueError(f"invalid joint duplicate-audit manifest: {error}") from error
    namespace = duplicate_manifest.duplicate_cluster_namespace
    if namespace is None or not namespace.is_evaluation_ready:
        raise ValueError("joint duplicate-audit namespace is not fully hash-bound")
    if namespace.to_dict() != provenance.duplicate_cluster_namespace.to_dict():
        raise ValueError(
            "joint duplicate audit and memory provenance use different namespaces"
        )

    entries = duplicate_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("joint duplicate-audit entries must be an array")
    normal_entries = [
        row
        for row in entries
        if isinstance(row, Mapping) and row.get("task_partition") == "normal"
    ]
    normal_ids = [str(row.get("task_id") or "") for row in normal_entries]
    if normal_ids != task_ids:
        raise ValueError(
            "joint duplicate audit must cover the exact 50 resolved pilot tasks "
            "in approved order"
        )

    corpus_binding = canonical_train_corpus_binding(
        records_sha256=provenance.records_sha256,
        provenance_manifest_sha256=provenance_sha256,
        duplicate_cluster_namespace=provenance.duplicate_cluster_namespace,
    )
    expected_corpus_sha256 = corpus_binding["corpus_binding_sha256"]
    for task_row, audit_row in zip(task_rows, normal_entries, strict=True):
        task_id = str(task_row["task_id"])
        try:
            clusters = duplicate_manifest.clusters_for(
                task_id,
                task_partition="normal",
            )
        except DuplicateAuditError as error:
            raise ValueError(
                f"joint duplicate audit is not VERIFIED for {task_id}: {error}"
            ) from error
        if not clusters:
            raise ValueError(
                f"joint duplicate audit has no cluster evidence for {task_id}"
            )
        expected_content_sha256 = duplicate_audit_task_content_sha256(task_row)
        if audit_row.get("content_sha256") != expected_content_sha256:
            raise ValueError(
                f"joint duplicate audit task-content hash mismatch for {task_id}"
            )
        if audit_row.get("train_corpus_manifest_sha256") != expected_corpus_sha256:
            raise ValueError(
                f"joint duplicate audit train-corpus hash mismatch for {task_id}"
            )

    return P4BuildPrerequisiteValidation(
        provenance=provenance,
        provenance_manifest_sha256=provenance_sha256,
        resolved_task_export_sha256=sha256_file(export_path),
        task_interface_audit_sha256=sha256_file(interface_path),
        duplicate_audit_sha256=duplicate_manifest.manifest_sha256,
        train_corpus_binding_sha256=expected_corpus_sha256,
        duplicate_cluster_namespace_id=namespace.namespace_id,
        verified_task_count=len(normal_entries),
    )


def validate_registered_p4_build_prerequisites(
    *,
    provenance_manifest_path: str | Path,
    resolved_task_export_path: str | Path,
    webarena_task_source_path: str | Path,
    webarena_task_registry_path: str | Path,
    webarena_site_url_map_path: str | Path,
    task_interface_audit_path: str | Path,
    duplicate_audit_path: str | Path,
    joint_assignment_package_path: str | Path,
    joint_audit_config_path: str | Path,
    p4_source_authority_path: str | Path,
    p4_preparation_package_path: str | Path,
    gold_train_json_path: str | Path,
    recovery_scenarios_path: str | Path,
    duplicate_audit_registration_path: str | Path,
    supplement_train_json_path: str | Path | None = None,
    authorized_raw_task_source_sha256: str | None = None,
    expected_task_source_sha256: str = PINNED_TASK_SOURCE_SHA256,
) -> P4BuildPrerequisiteValidation:
    """Production gate that replays the registered duplicate-audit producer.

    The older :func:`validate_p4_build_prerequisites` remains a dependency-light
    validator for already-authenticated evidence.  This wrapper is the only gate
    used by the production CLI: it reopens the exact train sources, preparation
    package, registered algorithm/config, task export/registry, provenance,
    recovery registration, and final 50+15 audit before any checkpoint access.
    """

    if sha256_file(p4_source_authority_path) != (
        P4_REGISTERED_SOURCE_AUTHORITY_SHA256
    ):
        raise ValueError("P4 source authority differs from registered PC-01 authority")

    assignment = validate_joint_duplicate_assignment_package(
        package_root=joint_assignment_package_path,
        config_path=joint_audit_config_path,
        source_authority_path=p4_source_authority_path,
        preparation_root=p4_preparation_package_path,
        gold_train_json=gold_train_json_path,
        supplement_train_json=supplement_train_json_path,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=webarena_task_registry_path,
        recovery_scenarios_path=recovery_scenarios_path,
        duplicate_audit_registration_path=duplicate_audit_registration_path,
    )
    validate_final_joint_duplicate_audit(
        audit_path=duplicate_audit_path,
        assignment_package_root=joint_assignment_package_path,
        config_path=joint_audit_config_path,
        source_authority_path=p4_source_authority_path,
        preparation_root=p4_preparation_package_path,
        gold_train_json=gold_train_json_path,
        supplement_train_json=supplement_train_json_path,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=webarena_task_registry_path,
        recovery_scenarios_path=recovery_scenarios_path,
        duplicate_audit_registration_path=duplicate_audit_registration_path,
        provenance_manifest_path=provenance_manifest_path,
    )
    base = validate_p4_build_prerequisites(
        provenance_manifest_path=provenance_manifest_path,
        resolved_task_export_path=resolved_task_export_path,
        webarena_task_source_path=webarena_task_source_path,
        webarena_task_registry_path=webarena_task_registry_path,
        webarena_site_url_map_path=webarena_site_url_map_path,
        task_interface_audit_path=task_interface_audit_path,
        duplicate_audit_path=duplicate_audit_path,
        authorized_raw_task_source_sha256=authorized_raw_task_source_sha256,
        expected_task_source_sha256=expected_task_source_sha256,
    )
    if assignment.namespace.to_dict() != (
        base.provenance.duplicate_cluster_namespace.to_dict()
    ):
        raise ValueError(
            "registered assignment and validated provenance use different namespaces"
        )
    return replace(
        base,
        preparation_manifest_sha256=sha256_file(
            Path(p4_preparation_package_path) / "preparation_manifest.json"
        ),
        joint_assignment_manifest_sha256=assignment.assignment_manifest_sha256,
        joint_assignment_entities_sha256=str(
            assignment.manifest["entities_sha256"]
        ),
        joint_assignment_clusters_sha256=str(
            assignment.manifest["clusters_sha256"]
        ),
        joint_audit_config_sha256=str(
            assignment.manifest["audit_tool_config_sha256"]
        ),
        joint_audit_source_sha256=str(
            assignment.manifest["audit_tool_source_sha256"]
        ),
        recovery_scenarios_sha256=sha256_file(recovery_scenarios_path),
        duplicate_audit_registration_sha256=sha256_file(
            duplicate_audit_registration_path
        ),
        source_authority_sha256=sha256_file(p4_source_authority_path),
    )


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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.model_seed < 0:
        raise ValueError("--model-seed must be non-negative")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    # This is deliberately the first artifact gate. It parses no checkpoint,
    # imports no model backend, queries no accelerator, and creates no store.
    prerequisites = validate_registered_p4_build_prerequisites(
        provenance_manifest_path=args.provenance_manifest,
        resolved_task_export_path=args.resolved_task_export,
        webarena_task_source_path=args.webarena_task_source,
        webarena_task_registry_path=args.webarena_task_registry,
        webarena_site_url_map_path=args.webarena_site_url_map,
        task_interface_audit_path=args.task_interface_audit,
        duplicate_audit_path=args.duplicate_audit,
        joint_assignment_package_path=args.joint_assignment_package,
        joint_audit_config_path=args.joint_audit_config,
        p4_source_authority_path=args.p4_source_authority,
        p4_preparation_package_path=args.p4_preparation_package,
        gold_train_json_path=args.data_root / "split_train.json",
        supplement_train_json_path=(
            args.supplement_root / "data" / "supplement_train.json"
            if args.supplement_root is not None
            else None
        ),
        recovery_scenarios_path=args.recovery_scenarios,
        duplicate_audit_registration_path=(
            args.duplicate_audit_registration
        ),
        authorized_raw_task_source_sha256=(
            args.authorized_raw_task_source_sha256
        ),
    )
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
    provenance = prerequisites.provenance
    if records_sha256 != provenance.records_sha256:
        raise ValueError(
            "logical training-record hash does not match provenance manifest: "
            f"expected {provenance.records_sha256}, found {records_sha256}"
        )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    protocol_sha256 = sha256_file(args.protocol_config)
    provenance_manifest_sha256 = prerequisites.provenance_manifest_sha256

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
    compact_selection = reconstruct_p4_selection_from_preparation(
        package_root=args.p4_preparation_package,
        provenance_manifest_path=args.provenance_manifest,
    )
    selection_contract = {
        "input_rows": selection.input_rows,
        "pre_dedup_eligible_rows": selection.pre_dedup_eligible_rows,
        "exclusion_counts": dict(selection.exclusion_counts),
        "duplicate_cluster_namespace": dict(
            selection.duplicate_cluster_namespace
        ),
        "items": [dict(candidate.item) for candidate in selection.candidates],
        "verification_evidence": [
            dict(candidate.verification_evidence)
            for candidate in selection.candidates
        ],
    }
    compact_contract = {
        "input_rows": compact_selection.input_rows,
        "pre_dedup_eligible_rows": compact_selection.pre_dedup_eligible_rows,
        "exclusion_counts": dict(compact_selection.exclusion_counts),
        "duplicate_cluster_namespace": dict(
            compact_selection.duplicate_cluster_namespace
        ),
        "items": [
            dict(candidate.item) for candidate in compact_selection.candidates
        ],
        "verification_evidence": [
            dict(candidate.verification_evidence)
            for candidate in compact_selection.candidates
        ],
    }
    if selection_contract != compact_contract:
        raise ValueError(
            "raw-source P4 selection differs from the authenticated compact "
            "preparation/provenance replay"
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
        joint_duplicate_audit_binding=(
            prerequisites.registered_duplicate_evidence_binding()
        ),
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
        "p4_build_prerequisites": prerequisites.report(),
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "runtime_writes_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
