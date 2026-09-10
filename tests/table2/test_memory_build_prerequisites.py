from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_bytes, sha256_file
from web_agent.eval.table2.package_validator import (
    _build_task_content_binding_manifest,
    _train_corpus_binding,
)
from web_agent.eval.table2.task_interface_audit import (
    build_webarena_task_interface_audit,
)
from web_agent.eval.table2.webarena_export import (
    PINNED_BROWSERGYM_WEBARENA_VERSION,
    PINNED_TASK_DEFINITION_VERSION,
    build_public_pilot_task_export,
)
from web_agent.memory import build_pipeline
from web_agent.memory.build_pipeline import validate_p4_build_prerequisites
from web_agent.runtime.contracts import canonical_sha256
from web_agent.runtime.duplicate_audit import (
    JointDuplicateClusterNamespace,
    canonical_duplicate_audit_task_content,
    canonical_task_audit_record,
    canonical_train_corpus_binding,
    duplicate_audit_task_content_sha256,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
URL_MAP = {
    "__GITLAB__": "http://gitlab.example.test",
    "__MAP__": "http://map.example.test",
    "__REDDIT__": "http://reddit.example.test",
    "__SHOPPING__": "http://shopping.example.test",
    "__SHOPPING_ADMIN__": "http://admin.example.test",
}
NAMESPACE_VALUE = {
    "schema_version": "table2-joint-duplicate-cluster-namespace-v1",
    "namespace_id": "fixture-gold-train-webarena-joint-v1",
    "audit_tool_id": "fixture-external-joint-auditor",
    "audit_tool_version": "fixture-v1",
    "audit_tool_config_sha256": SHA_A,
    "audit_tool_source_sha256": SHA_C,
}


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _fixture_inputs(
    root: Path,
    *,
    incompatible_task: bool = False,
    task_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    indices = task_indices if task_indices is not None else tuple(range(50))
    assert len(indices) == 50 and len(set(indices)) == 50
    registry = _write_json(
        root / "pilot-task-registry.json",
        {
            "schema_version": "1.0",
            "manifest_id": "fixture-approved-public-dev-50",
            "benchmark": "webarena",
            "partition": "development",
            "registration_status": "FROZEN_DEVELOPMENT_EXCLUSION",
            "required_task_count": 50,
            "final_paper_evaluation_eligible": False,
            "locked_test_content": False,
            "tasks": [
                {"task_id": f"webarena.{index}", "upstream_index": index}
                for index in indices
            ],
        },
    )
    url_map = _write_json(root / "task-url-map.json", URL_MAP)
    tokens = sorted(URL_MAP)
    source_rows: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        token = tokens[position % len(tokens)]
        evaluator = {
            "eval_types": ["url_match"],
            "reference_url": f"{token}/result/{index}",
        }
        if incompatible_task and position == 0:
            evaluator = {
                "eval_types": ["string_match"],
                "reference_answers": {"exact_match": "fixture-answer"},
            }
        source_rows.append(
            {
                "sites": [token.strip("_").casefold()],
                "task_id": index,
                "require_login": True,
                "storage_state": f"./.auth/site_{index}_state.json",
                "start_url": f"{token}/start/{index}",
                "geolocation": None,
                "intent": f"Fixture development task {index}",
                "require_reset": False,
                "eval": evaluator,
            }
        )
    source_bytes = json.dumps(source_rows, sort_keys=True).encode("utf-8")
    source = root / "test.raw.json"
    source.write_bytes(source_bytes)
    source_sha256 = sha256_bytes(source_bytes)

    task_export = build_public_pilot_task_export(
        source=source,
        registry_path=registry,
        site_url_map=URL_MAP,
        snapshot_id="fixture-approved-compatible-pilot-v1",
        benchmark_version=PINNED_BROWSERGYM_WEBARENA_VERSION,
        task_definition_version=PINNED_TASK_DEFINITION_VERSION,
        evaluator_id="fixture-sealed-webarena-evaluator",
        evaluator_version="fixture-v1",
        expected_source_sha256=source_sha256,
        authorized_raw_json_sha256=source_sha256,
    )
    export_path = _write_json(root / "resolved-task-export.json", task_export)
    interface_audit = build_webarena_task_interface_audit(task_export)
    interface_path = _write_json(
        root / "task-interface-audit.json",
        interface_audit,
    )

    provenance_path = _write_json(
        root / "provenance.json",
        {
            "schema_version": "table2-memory-provenance-v1",
            "source_split": "train",
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "dataset_id": "fixture-gold",
            "dataset_version": "fixture-v1",
            "dataset_artifacts_sha256": SHA_A,
            "records_sha256": SHA_B,
            "duplicate_cluster_namespace": NAMESPACE_VALUE,
            "records": {},
        },
    )
    namespace = JointDuplicateClusterNamespace.from_mapping(
        NAMESPACE_VALUE,
        require_hashes=True,
    )
    corpus_binding = canonical_train_corpus_binding(
        records_sha256=SHA_B,
        provenance_manifest_sha256=sha256_file(provenance_path),
        duplicate_cluster_namespace=namespace,
    )
    audit_entries = []
    for task_row in task_export["tasks"]:
        task_id = str(task_row["task_id"])
        clusters = [f"fixture-cluster:{task_id}"]
        audit_record = canonical_task_audit_record(
            task_id=task_id,
            content_sha256=duplicate_audit_task_content_sha256(task_row),
            train_corpus_manifest_sha256=corpus_binding[
                "corpus_binding_sha256"
            ],
            cluster_ids=clusters,
            namespace=namespace,
        )
        audit_entries.append(
            {
                "task_id": task_id,
                "task_partition": "normal",
                "status": "VERIFIED",
                "cluster_ids": clusters,
                "content_sha256": audit_record["content_sha256"],
                "train_corpus_manifest_sha256": audit_record[
                    "train_corpus_manifest_sha256"
                ],
                "audit_tool_id": namespace.audit_tool_id,
                "audit_tool_version": namespace.audit_tool_version,
                "evidence_sha256": canonical_sha256(audit_record),
                "cluster_namespace_id": namespace.namespace_id,
                "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
                "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
                "audit_record": audit_record,
            }
        )
    duplicate_path = _write_json(
        root / "joint-duplicate-audit.json",
        {
            "schema_version": "table2-duplicate-audit-v1",
            "manifest_id": "fixture-completed-joint-audit-v1",
            "manifest_state": "FROZEN_REGISTRATION",
            "evidence_label": "PILOT_ONLY",
            "normal_task_evidence_status": "VERIFIED",
            "normal_task_runtime_policy": "VERIFIED_NONEMPTY_CLUSTERS_REQUIRED",
            "required_normal_task_count": 50,
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "duplicate_cluster_namespace": NAMESPACE_VALUE,
            "entries": audit_entries,
        },
    )
    return {
        "provenance_manifest_path": provenance_path,
        "resolved_task_export_path": export_path,
        "webarena_task_source_path": source,
        "webarena_task_registry_path": registry,
        "webarena_site_url_map_path": url_map,
        "task_interface_audit_path": interface_path,
        "duplicate_audit_path": duplicate_path,
        "authorized_raw_task_source_sha256": source_sha256,
        "expected_task_source_sha256": source_sha256,
    }


def test_p4_build_prerequisite_accepts_exact_external_fixture(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)

    result = validate_p4_build_prerequisites(**inputs)

    assert result.verified_task_count == 50
    assert result.train_corpus_binding_sha256
    assert result.report()["evidence_role"] == (
        "VALIDATION_ONLY_EXTERNAL_EVIDENCE_NOT_AUTHORED"
    )
    assert result.report()["validation_rows_read"] == 0


def test_p4_prerequisite_accepts_exact_noncontiguous_registry_order(
    tmp_path: Path,
) -> None:
    indices = tuple(200 + ((position * 19) % 103) for position in range(50))
    inputs = _fixture_inputs(tmp_path, task_indices=indices)

    result = validate_p4_build_prerequisites(**inputs)

    export = json.loads(
        inputs["resolved_task_export_path"].read_text(encoding="utf-8")
    )
    assert [row["upstream_index"] for row in export["tasks"]] == list(indices)
    assert result.verified_task_count == 50


def test_prebuild_bindings_equal_campaign_handoff_bindings(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)
    provenance_sha256 = sha256_file(inputs["provenance_manifest_path"])
    expected_corpus = canonical_train_corpus_binding(
        records_sha256=SHA_B,
        provenance_manifest_sha256=provenance_sha256,
        duplicate_cluster_namespace=NAMESPACE_VALUE,
    )
    assert _train_corpus_binding(
        {
            "records_sha256": SHA_B,
            "provenance_manifest_sha256": provenance_sha256,
            "duplicate_cluster_namespace": NAMESPACE_VALUE,
        }
    ) == expected_corpus

    export = json.loads(
        inputs["resolved_task_export_path"].read_text(encoding="utf-8")
    )
    resolved_row = canonical_duplicate_audit_task_content(export["tasks"][0])
    handoff_binding = _build_task_content_binding_manifest(
        [resolved_row],
        task_manifest_sha256=SHA_A,
        memory_by_seed={},
        evidence_scope="EVALUATION_RUNNER",
    )
    assert handoff_binding["task_content_sha256_by_id"][
        resolved_row["task_id"]
    ] == duplicate_audit_task_content_sha256(export["tasks"][0])


def test_p4_build_prerequisite_rejects_incompatible_task_export(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path, incompatible_task=True)

    with pytest.raises(SchemaError, match="rejects evaluation handoff"):
        validate_p4_build_prerequisites(**inputs)


@pytest.mark.parametrize(
    "field",
    ["validation_rows_read", "test_rows_read", "locked_test_rows_read"],
)
def test_p4_build_prerequisite_rejects_duplicate_audit_forbidden_reads(
    tmp_path: Path,
    field: str,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    duplicate_path = inputs["duplicate_audit_path"]
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    duplicate[field] = 1
    _write_json(duplicate_path, duplicate)

    with pytest.raises(ValueError, match=field):
        validate_p4_build_prerequisites(**inputs)


@pytest.mark.parametrize("mode", ["pending", "empty_clusters"])
def test_p4_build_prerequisite_requires_verified_nonempty_task_evidence(
    tmp_path: Path,
    mode: str,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    duplicate_path = inputs["duplicate_audit_path"]
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    row = duplicate["entries"][0]
    row["cluster_ids"] = []
    if mode == "pending":
        row.update(
            {
                "status": "PENDING_EXTERNAL_AUDIT",
                "content_sha256": None,
                "train_corpus_manifest_sha256": None,
                "evidence_sha256": None,
                "audit_record": None,
            }
        )
    _write_json(duplicate_path, duplicate)

    expected = "pending" if mode == "pending" else "complete evidence"
    with pytest.raises(ValueError, match=expected):
        validate_p4_build_prerequisites(**inputs)


@pytest.mark.parametrize("binding", ["task_content", "train_corpus"])
def test_p4_build_prerequisite_rejects_self_consistently_resealed_binding(
    tmp_path: Path,
    binding: str,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    duplicate_path = inputs["duplicate_audit_path"]
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    namespace = JointDuplicateClusterNamespace.from_mapping(
        duplicate["duplicate_cluster_namespace"],
        require_hashes=True,
    )
    row = duplicate["entries"][0]
    content_sha256 = row["content_sha256"]
    corpus_sha256 = row["train_corpus_manifest_sha256"]
    if binding == "task_content":
        content_sha256 = SHA_C
    else:
        corpus_sha256 = SHA_C
    record = canonical_task_audit_record(
        task_id=row["task_id"],
        content_sha256=content_sha256,
        train_corpus_manifest_sha256=corpus_sha256,
        cluster_ids=row["cluster_ids"],
        namespace=namespace,
    )
    row["content_sha256"] = content_sha256
    row["train_corpus_manifest_sha256"] = corpus_sha256
    row["audit_record"] = record
    row["evidence_sha256"] = canonical_sha256(record)
    _write_json(duplicate_path, duplicate)

    expected = "task-content" if binding == "task_content" else "train-corpus"
    with pytest.raises(ValueError, match=expected):
        validate_p4_build_prerequisites(**inputs)


def test_p4_build_prerequisite_rejects_self_consistent_namespace_mismatch(
    tmp_path: Path,
) -> None:
    inputs = _fixture_inputs(tmp_path)
    duplicate_path = inputs["duplicate_audit_path"]
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    changed_namespace = {
        **duplicate["duplicate_cluster_namespace"],
        "namespace_id": "fixture-different-joint-namespace-v1",
    }
    namespace = JointDuplicateClusterNamespace.from_mapping(
        changed_namespace,
        require_hashes=True,
    )
    corpus = canonical_train_corpus_binding(
        records_sha256=SHA_B,
        provenance_manifest_sha256=sha256_file(
            inputs["provenance_manifest_path"]
        ),
        duplicate_cluster_namespace=namespace,
    )
    duplicate["duplicate_cluster_namespace"] = changed_namespace
    for row in duplicate["entries"]:
        record = canonical_task_audit_record(
            task_id=row["task_id"],
            content_sha256=row["content_sha256"],
            train_corpus_manifest_sha256=corpus["corpus_binding_sha256"],
            cluster_ids=row["cluster_ids"],
            namespace=namespace,
        )
        row.update(
            {
                "train_corpus_manifest_sha256": corpus[
                    "corpus_binding_sha256"
                ],
                "cluster_namespace_id": namespace.namespace_id,
                "audit_tool_id": namespace.audit_tool_id,
                "audit_tool_version": namespace.audit_tool_version,
                "audit_tool_config_sha256": namespace.audit_tool_config_sha256,
                "audit_tool_source_sha256": namespace.audit_tool_source_sha256,
                "audit_record": record,
                "evidence_sha256": canonical_sha256(record),
            }
        )
    _write_json(duplicate_path, duplicate)

    with pytest.raises(ValueError, match="different namespaces"):
        validate_p4_build_prerequisites(**inputs)


def test_memory_build_main_runs_prerequisite_before_checkpoint_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_before_model_work(**_: Any) -> None:
        raise RuntimeError("prerequisite-sentinel")

    monkeypatch.setattr(
        build_pipeline,
        "validate_registered_p4_build_prerequisites",
        reject_before_model_work,
    )
    argv = [
        "--config",
        "missing-config.json",
        "--checkpoint",
        "missing-checkpoint.pt",
        "--data-root",
        "missing-data",
        "--provenance-manifest",
        "missing-provenance.json",
        "--resolved-task-export",
        "missing-tasks.json",
        "--webarena-task-source",
        "missing-webarena.whl",
        "--webarena-task-registry",
        "missing-registry.json",
        "--webarena-site-url-map",
        "missing-url-map.json",
        "--task-interface-audit",
        "missing-interface-audit.json",
        "--duplicate-audit",
        "missing-duplicate-audit.json",
        "--joint-assignment-package",
        "missing-joint-assignments",
        "--joint-audit-config",
        "missing-joint-config.json",
        "--p4-preparation-package",
        "missing-preparation",
        "--p4-source-authority",
        "missing-source-authority.json",
        "--recovery-scenarios",
        "missing-recovery-scenarios.json",
        "--duplicate-audit-registration",
        "missing-duplicate-registration.json",
        "--model-seed",
        "42",
        "--output-dir",
        "missing-output",
    ]

    with pytest.raises(RuntimeError, match="prerequisite-sentinel"):
        build_pipeline.main(argv)
