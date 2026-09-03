from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    FrozenDuplicateAuditManifest,
    JointDuplicateClusterNamespace,
    canonical_task_audit_record,
    validate_pilot_duplicate_audit_manifest,
)
from web_agent.runtime.contracts import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "benchmarks" / "table2" / "pilot"


def _ids():
    tasks = json.loads((PILOT / "task_manifest.json").read_text(encoding="utf-8"))
    recovery = json.loads(
        (PILOT / "recovery_scenarios.json").read_text(encoding="utf-8")
    )
    return (
        [row["task_id"] for row in tasks["tasks"]],
        [row["scenario_id"] for row in recovery["scenarios"]],
    )


def test_pilot_duplicate_audit_has_exact_50_plus_15_coverage():
    normal, recovery = _ids()
    manifest = validate_pilot_duplicate_audit_manifest(
        PILOT / "duplicate_audit_manifest.json",
        normal_task_ids=normal,
        recovery_scenario_ids=recovery,
        require_normal_verified=False,
    )
    assert len(manifest.task_ids) == 65
    assert manifest.clusters_for(
        recovery[0],
        task_partition="recovery_diagnostic",
        allow_synthetic_diagnostic=True,
    )


def test_pending_normal_duplicate_evidence_is_rejected_before_e3():
    normal, recovery = _ids()
    manifest = FrozenDuplicateAuditManifest.from_path(
        PILOT / "duplicate_audit_manifest.json",
        expected_task_ids=normal + recovery,
    )
    with pytest.raises(DuplicateAuditError, match="pending"):
        manifest.clusters_for(normal[0], task_partition="normal")
    with pytest.raises(DuplicateAuditError, match="not verified"):
        manifest.require_evaluation_ready(normal + recovery)
    with pytest.raises(DuplicateAuditError):
        validate_pilot_duplicate_audit_manifest(
            PILOT / "duplicate_audit_manifest.json",
            normal_task_ids=normal,
            recovery_scenario_ids=recovery,
            require_normal_verified=True,
        )


def test_synthetic_clusters_cannot_be_relabelled_as_normal_evidence():
    manifest = FrozenDuplicateAuditManifest.synthetic_diagnostic(
        task_id="R-fixture",
        cluster_ids=("synthetic-cluster",),
        content_identity={"fixture": "v1"},
    )
    with pytest.raises(DuplicateAuditError, match="partition"):
        manifest.clusters_for(
            "R-fixture",
            task_partition="normal",
            allow_synthetic_diagnostic=True,
        )


def test_verified_normal_evidence_rejects_an_arbitrary_64_char_hash(tmp_path: Path):
    namespace_value = {
        "schema_version": "table2-joint-duplicate-cluster-namespace-v1",
        "namespace_id": "joint-fixture-v1",
        "audit_tool_id": "fixture-clusterer",
        "audit_tool_version": "v1",
        "audit_tool_config_sha256": "a" * 64,
        "audit_tool_source_sha256": "b" * 64,
    }
    namespace = JointDuplicateClusterNamespace.from_mapping(
        namespace_value,
        require_hashes=True,
    )
    audit_record = canonical_task_audit_record(
        task_id="webarena.0",
        content_sha256="c" * 64,
        train_corpus_manifest_sha256="d" * 64,
        cluster_ids=("cluster-0",),
        namespace=namespace,
    )
    entry = {
        "task_id": "webarena.0",
        "task_partition": "normal",
        "status": "VERIFIED",
        "cluster_ids": ["cluster-0"],
        "content_sha256": "c" * 64,
        "train_corpus_manifest_sha256": "d" * 64,
        "audit_tool_id": namespace.audit_tool_id,
        "audit_tool_version": namespace.audit_tool_version,
        "evidence_sha256": "f" * 64,
        "cluster_namespace_id": namespace.namespace_id,
        "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
        "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
        "audit_record": audit_record,
    }
    path = tmp_path / "fabricated-evidence.json"
    path.write_text(
        json.dumps({
            "schema_version": "table2-duplicate-audit-v1",
            "manifest_id": "fabricated-evidence-canary",
            "manifest_state": "FROZEN_REGISTRATION",
            "duplicate_cluster_namespace": namespace_value,
            "entries": [entry],
        }),
        encoding="utf-8",
    )

    assert canonical_sha256(audit_record) != entry["evidence_sha256"]
    with pytest.raises(DuplicateAuditError, match="not canonical"):
        FrozenDuplicateAuditManifest.from_path(path)
