"""Deterministic Gold-train/WebArena duplicate-audit producer.

This module deliberately separates duplicate assignment from scientific
recovery review.  The first stage clusters train-only P4 review candidates and
an approved 50-task WebArena development export.  The second stage can emit the
runtime 50+15 ``FrozenDuplicateAuditManifest`` only after an independently
authored, complete provenance manifest cites the exact first-stage package; the
15 registered diagnostic rows remain synthetic and diagnostic-only.

No command accepts a validation, test, or locked-benchmark input.  No command
creates recovery-success, final-task-success, reviewer, or verifier evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata
from urllib.parse import urlsplit

from web_agent.eval.table2.common import canonical_json_bytes
from web_agent.memory.eligibility import ProvenanceManifest
from web_agent.memory.manifest import ManifestError, sha256_file
from web_agent.memory.preparation import (
    P4_SOURCE_AUTHORITY_FILE_NAME,
    P4PreparationError,
    validate_p4_preparation_package,
    validate_p4_provenance_against_preparation,
)
from web_agent.memory.verification import (
    VerificationEvidenceError,
    canonical_p4_memory_source_material,
    recovery_action_evidence_sha256,
    recovery_state_evidence_sha256,
)
from web_agent.data.recovery_transitions import build_recovery_transition_index
from web_agent.runtime.contracts import canonical_sha256
from web_agent.runtime.duplicate_audit import (
    DUPLICATE_AUDIT_SCHEMA_VERSION,
    DuplicateAuditError,
    FrozenDuplicateAuditManifest,
    JointDuplicateClusterNamespace,
    canonical_task_audit_record,
    canonical_train_corpus_binding,
    duplicate_audit_task_content_sha256,
    validate_pilot_duplicate_audit_manifest,
)


AUDIT_CONFIG_SCHEMA_VERSION = "table2-joint-duplicate-audit-config-v1"
ASSIGNMENT_PACKAGE_SCHEMA_VERSION = (
    "table2-joint-duplicate-assignment-package-v3"
)
ASSIGNMENT_ENTITIES_SCHEMA_VERSION = (
    "table2-joint-duplicate-assignment-entity-v2"
)
ASSIGNMENT_CLUSTERS_SCHEMA_VERSION = (
    "table2-joint-duplicate-clusters-v1"
)
ASSIGNMENT_READ_LEDGER_SCHEMA_VERSION = (
    "table2-joint-duplicate-read-ledger-v1"
)
PROVENANCE_ASSIGNMENT_BINDING_SCHEMA_VERSION = (
    "table2-provenance-joint-duplicate-assignment-binding-v1"
)
AUDIT_TOOL_SOURCE_RELATIVE_PATH = (
    "src/web_agent/memory/joint_duplicate_audit.py"
)
AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS = (
    "src/web_agent/data/recovery_transitions.py",
    "src/web_agent/eval/table2/common.py",
    "src/web_agent/memory/eligibility.py",
    "src/web_agent/memory/manifest.py",
    "src/web_agent/memory/preparation.py",
    "src/web_agent/memory/verification.py",
    "src/web_agent/runtime/contracts.py",
    "src/web_agent/runtime/duplicate_audit.py",
)
ASSIGNMENT_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "table2-joint-duplicate-source-manifest-v1"
)
PILOT_TASK_COUNT = 50
RECOVERY_SCENARIO_COUNT = 15

_CONFIG_FILE_NAME = "audit_config.json"
_ENTITIES_FILE_NAME = "entities.jsonl"
_CLUSTERS_FILE_NAME = "clusters.json"
_READ_LEDGER_FILE_NAME = "read_ledger.json"
_SOURCE_MANIFEST_FILE_NAME = "source_manifest.json"
_MANIFEST_FILE_NAME = "assignment_manifest.json"
_MANIFEST_SIDECAR_NAME = "assignment_manifest.sha256"
_PAYLOAD_FILES = (
    _CONFIG_FILE_NAME,
    _ENTITIES_FILE_NAME,
    _CLUSTERS_FILE_NAME,
    _READ_LEDGER_FILE_NAME,
    _SOURCE_MANIFEST_FILE_NAME,
)
_ZERO_READ_FIELDS = (
    "validation_rows_read",
    "test_rows_read",
    "locked_test_rows_read",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[a-z0-9]+")
_FORBIDDEN_PATH_PARTS = frozenset(
    {"locked_benchmark_mount", "locked-test", "locked_test"}
)

_REGISTERED_CONFIG: dict[str, Any] = {
    "audit_tool_id": "table2-deterministic-joint-duplicate-auditor",
    "audit_tool_version": "1.0.0",
    "cluster_id_policy": {
        "exact": "exact:<sha256-of-normalized-goal-and-sites>",
        "near": "near:<sha256-of-namespace-and-sorted-component-members>",
        "task_runtime_clusters": "sorted_exact_and_near_ids",
    },
    "exact_match": {
        "identity": "sha256_of_canonical_normalized_goal_tokens_and_site_keys",
    },
    "near_match": {
        "feature_representation": "word_3_shingle_set_plus_site_keys",
        "jaccard_threshold_denominator": 5,
        "jaccard_threshold_numerator": 4,
        "linkage": "deterministic_connected_components_single_link",
        "require_site_overlap_when_both_nonempty": True,
        "shingle_size": 3,
        "singleton_policy": "retain_unique_cluster",
    },
    "normalization": {
        "case": "unicode_casefold",
        "empty_goal": "forbidden",
        "site_key": "lowercase_host_or_label_without_www_or_trailing_dot",
        "token_pattern": "[a-z0-9]+",
        "unicode_form": "NFKC",
        "whitespace": "single_ascii_space",
    },
    "schema_version": AUDIT_CONFIG_SCHEMA_VERSION,
}


class JointDuplicateAuditError(ValueError):
    """The duplicate audit input, package, or binding is invalid."""


@dataclass(frozen=True, slots=True)
class JointDuplicateAssignmentPackage:
    """Validated immutable first-stage assignment evidence."""

    root: Path
    manifest: Mapping[str, Any]
    namespace: JointDuplicateClusterNamespace
    entities: tuple[Mapping[str, Any], ...]
    clusters: Mapping[str, Any]

    @property
    def assignment_manifest_sha256(self) -> str:
        return sha256_file(self.root / _MANIFEST_FILE_NAME)

    @property
    def candidates(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            row for row in self.entities if row["entity_kind"] == "gold_train_candidate"
        )

    @property
    def tasks(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            row
            for row in self.entities
            if row["entity_kind"] == "webarena_development_task"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JointDuplicateAuditError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    _require_regular_file(path, role=role)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except json.JSONDecodeError as error:
        raise JointDuplicateAuditError(f"invalid {role} JSON: {error}") from error
    if not isinstance(value, dict):
        raise JointDuplicateAuditError(f"{role} must be a JSON object")
    return value


def _load_json_array(path: Path, *, role: str) -> list[dict[str, Any]]:
    _require_regular_file(path, role=role)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except json.JSONDecodeError as error:
        raise JointDuplicateAuditError(f"invalid {role} JSON: {error}") from error
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise JointDuplicateAuditError(f"{role} must be a JSON array of objects")
    return value


def _require_regular_file(path: Path, *, role: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise JointDuplicateAuditError(
            f"{role} must be a regular non-symlink file: {path}"
        )
    if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in path.resolve().parts):
        raise JointDuplicateAuditError(f"{role} resolves beneath a locked path")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise JointDuplicateAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_zero_reads(value: Mapping[str, Any], *, role: str) -> None:
    for field in _ZERO_READ_FIELDS:
        if type(value.get(field)) is not int or value[field] != 0:
            raise JointDuplicateAuditError(f"{role} requires explicit {field}=0")


def _canonical_pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_config(path: Path) -> tuple[dict[str, Any], str, str]:
    value = _load_json(path, role="joint duplicate-audit config")
    if value != _REGISTERED_CONFIG:
        raise JointDuplicateAuditError(
            "joint duplicate-audit config differs from the registered v1 method"
        )
    if path.read_bytes() != _canonical_pretty_bytes(_REGISTERED_CONFIG):
        raise JointDuplicateAuditError(
            "joint duplicate-audit config is not in its canonical frozen encoding"
        )
    source = Path(__file__).resolve()
    _require_regular_file(source, role="joint duplicate-audit implementation")
    return value, sha256_file(path), sha256_file(source)


def _audit_source_manifest(config_path: Path) -> dict[str, Any]:
    """Bind every local implementation file that defines audit semantics.

    The assignment producer imports normalization-independent evidence and
    transition helpers.  Hashing only this module would allow those helper
    semantics to drift while retaining the same declared producer identity.
    The config must therefore live at its canonical repository-relative path,
    and every dependency byte must match the code currently executing.
    """

    resolved_config = config_path.resolve()
    try:
        repository_root = resolved_config.parents[3]
    except IndexError as error:  # pragma: no cover - defensive path contract
        raise JointDuplicateAuditError(
            "joint duplicate-audit config has no repository root"
        ) from error
    expected_config = (
        repository_root / "configs/eval/table2/joint_duplicate_audit_v1.json"
    ).resolve()
    if resolved_config != expected_config:
        raise JointDuplicateAuditError(
            "joint duplicate-audit config must use its canonical repository path"
        )
    execution_root = Path(__file__).resolve().parents[3]
    rows: list[dict[str, Any]] = []
    for relative in (
        AUDIT_TOOL_SOURCE_RELATIVE_PATH,
        *AUDIT_TOOL_DEPENDENCY_RELATIVE_PATHS,
    ):
        tracked = (repository_root / relative).resolve()
        executing = (execution_root / relative).resolve()
        _require_regular_file(tracked, role=f"audit source dependency {relative}")
        _require_regular_file(executing, role=f"executing source dependency {relative}")
        tracked_sha256 = sha256_file(tracked)
        if sha256_file(executing) != tracked_sha256:
            raise JointDuplicateAuditError(
                "tracked joint duplicate dependency differs from executing code: "
                f"{relative}"
            )
        rows.append(
            {
                "relative_path": relative,
                "sha256": tracked_sha256,
                "bytes": tracked.stat().st_size,
            }
        )
    rows.sort(key=lambda row: row["relative_path"])
    return {
        "schema_version": ASSIGNMENT_SOURCE_MANIFEST_SCHEMA_VERSION,
        "files": rows,
        "files_sha256": canonical_sha256(rows),
    }


def _public_value(value: Any) -> Any:
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


def _declares_train_only(record: Mapping[str, Any]) -> bool:
    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        return False
    declarations: list[str] = []
    for authority in (record, meta):
        for field in ("split", "source_split", "partition"):
            raw = authority.get(field)
            if raw not in (None, ""):
                declarations.append(str(raw).strip().casefold())
    return all(value == "train" for value in declarations)


def _source_sample_id(record: Mapping[str, Any], *, index: int) -> str:
    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        raise JointDuplicateAuditError(f"training row {index} lacks meta")
    value = str(meta.get("sample_id") or meta.get("row_id") or meta.get("id") or "").strip()
    if not value:
        raise JointDuplicateAuditError(f"training row {index} lacks sample_id")
    return value


def _normalize_goal(value: Any) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, str):
        raise JointDuplicateAuditError("duplicate-audit goal must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = tuple(_TOKEN.findall(normalized))
    if not tokens:
        raise JointDuplicateAuditError("duplicate-audit goal becomes empty after normalization")
    return " ".join(tokens), tokens


def _normalize_site(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = unicodedata.normalize("NFKC", value).strip().casefold()
    if "://" in text:
        parsed = urlsplit(text)
        text = parsed.hostname or ""
    else:
        text = text.split("/", 1)[0].split(":", 1)[0]
    text = text.rstrip(".")
    if text.startswith("www."):
        text = text[4:]
    return text


def _features(tokens: Sequence[str], sites: Sequence[str]) -> tuple[str, ...]:
    size = int(_REGISTERED_CONFIG["near_match"]["shingle_size"])
    if len(tokens) >= size:
        task_features = {
            "task:" + "\x1f".join(tokens[index : index + size])
            for index in range(len(tokens) - size + 1)
        }
    else:
        task_features = {f"task:{token}" for token in tokens}
    return tuple(sorted(task_features | {f"site:{site}" for site in sites}))


def _base_entity(
    *,
    entity_id: str,
    entity_kind: str,
    source_id: str,
    content_sha256: str,
    goal: str,
    sites: Sequence[str],
) -> dict[str, Any]:
    normalized_goal, tokens = _normalize_goal(goal)
    site_keys = tuple(sorted({site for raw in sites if (site := _normalize_site(raw))}))
    feature_values = _features(tokens, site_keys)
    exact_key = canonical_sha256(
        {"goal_tokens": list(tokens), "site_keys": list(site_keys)}
    )
    return {
        "schema_version": ASSIGNMENT_ENTITIES_SCHEMA_VERSION,
        "entity_id": entity_id,
        "entity_kind": entity_kind,
        "source_id": source_id,
        "content_sha256": _require_sha256(content_sha256, field="entity.content_sha256"),
        "normalized_goal": normalized_goal,
        "goal_tokens": list(tokens),
        "site_keys": list(site_keys),
        "features": list(feature_values),
        "features_sha256": canonical_sha256(list(feature_values)),
        "exact_duplicate_key": exact_key,
    }


def _load_queue_rows(package_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = package_root / "review_queue.jsonl"
    _require_regular_file(path, role="P4 review queue")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise JointDuplicateAuditError(
                f"blank P4 review-queue line {line_number}"
            )
        try:
            row = json.loads(raw, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise JointDuplicateAuditError(
                f"invalid P4 review-queue line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise JointDuplicateAuditError(
                f"P4 review-queue line {line_number} must be an object"
            )
        rows.append(row)
    return rows


def _load_training_candidate_entities(
    *,
    package_root: Path,
    gold_train_json: Path,
    supplement_train_json: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    read_ledger = _load_json(
        package_root / "read_ledger.json", role="P4 preparation read ledger"
    )
    _require_zero_reads(read_ledger, role="P4 preparation read ledger")
    descriptors = read_ledger.get("train_json_files")
    if not isinstance(descriptors, list):
        raise JointDuplicateAuditError("P4 preparation lacks train source descriptors")
    descriptor_by_role = {
        str(row.get("role")): row for row in descriptors if isinstance(row, Mapping)
    }
    if set(descriptor_by_role) not in (
        {"original_gold"},
        {"original_gold", "retry_abort_supplement_v2"},
    ):
        raise JointDuplicateAuditError("P4 preparation train-source roles are invalid")
    supplied: dict[str, Path] = {"original_gold": gold_train_json}
    if "retry_abort_supplement_v2" in descriptor_by_role:
        if supplement_train_json is None:
            raise JointDuplicateAuditError(
                "preparation package requires supplement_train.json"
            )
        supplied["retry_abort_supplement_v2"] = supplement_train_json
    elif supplement_train_json is not None:
        raise JointDuplicateAuditError(
            "supplement_train.json was supplied but is absent from preparation"
        )

    expected_names = {
        "original_gold": "split_train.json",
        "retry_abort_supplement_v2": "supplement_train.json",
    }
    source_records: dict[str, tuple[str, dict[str, Any]]] = {}
    source_roles: dict[str, str] = {}
    source_roots: dict[str, Path] = {}
    replay_records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    total_rows = 0
    for role in sorted(supplied):
        path = supplied[role]
        if path.name != expected_names[role]:
            raise JointDuplicateAuditError(
                f"{role} accepts only {expected_names[role]}"
            )
        rows = _load_json_array(path, role=f"{role} training source")
        if role == "original_gold":
            source_roots[role] = path.resolve().parent
        else:
            if path.parent.name != "data":
                raise JointDuplicateAuditError(
                    "registered supplement_train.json must be inside its data/ "
                    "directory so the immutable dataset root is derivable"
                )
            source_roots[role] = path.resolve().parent.parent
        descriptor = descriptor_by_role[role]
        expected = {
            "role": role,
            "file_name": path.name,
            "records": len(rows),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if dict(descriptor) != expected:
            raise JointDuplicateAuditError(
                f"{role} bytes/count differ from the P4 preparation ledger"
            )
        for index, record in enumerate(rows):
            if not _declares_train_only(record):
                raise JointDuplicateAuditError(
                    f"{role} row {index} declares a non-training split"
                )
            sample_id = _source_sample_id(record, index=index)
            if sample_id in source_records:
                raise JointDuplicateAuditError(
                    f"duplicate training source sample ID: {sample_id}"
                )
            source_records[sample_id] = (role, record)
            source_roles[sample_id] = role
            replay_record = deepcopy(record)
            replay_meta = replay_record.setdefault("meta", {})
            if not isinstance(replay_meta, dict):
                raise JointDuplicateAuditError(
                    f"{role} row {index} meta is not an object"
                )
            replay_meta["_source_dataset"] = role
            replay_meta["_direct_recovery_transition"] = (
                role == "retry_abort_supplement_v2"
            )
            replay_meta["_data_root"] = str(source_roots[role])
            replay_records.append(replay_record)
        total_rows += len(rows)
        source_summaries.append(expected)

    if read_ledger.get("train_rows_read") != total_rows:
        raise JointDuplicateAuditError(
            "P4 preparation train row count differs from reopened sources"
        )
    replay_transitions, _ = build_recovery_transition_index(replay_records)
    for sample_id, transition in replay_transitions.items():
        role = source_roles.get(sample_id)
        if role is None:
            raise JointDuplicateAuditError(
                f"replayed transition has no authenticated source role: {sample_id}"
            )
        transition["_data_root"] = str(source_roots[role])

    queue_rows = _load_queue_rows(package_root)
    entities: list[dict[str, Any]] = []
    queue_ids: set[str] = set()
    for queue_row in queue_rows:
        sample_id = str(queue_row.get("source_sample_id") or "").strip()
        if not sample_id or sample_id in queue_ids:
            raise JointDuplicateAuditError(
                "P4 review queue contains an empty or duplicate sample ID"
            )
        queue_ids.add(sample_id)
        try:
            source_role, record = source_records[sample_id]
        except KeyError as error:
            raise JointDuplicateAuditError(
                f"P4 review candidate is absent from reopened train sources: {sample_id}"
            ) from error
        if canonical_sha256(_public_value(record)) != queue_row.get(
            "source_record_sha256"
        ):
            raise JointDuplicateAuditError(
                f"P4 review candidate source hash differs for {sample_id}"
            )
        inputs = record.get("inputs")
        labels = record.get("labels")
        meta = record.get("meta", {})
        if not all(isinstance(value, Mapping) for value in (inputs, labels, meta)):
            raise JointDuplicateAuditError(
                f"training candidate {sample_id} lacks nested inputs/labels/meta"
            )
        expected_duplicate_material = {
            "task_description": str(inputs.get("task_description") or "").strip(),
            "website_domain": str(inputs.get("website_domain") or "").strip(),
        }
        if queue_row.get("duplicate_audit_material") != expected_duplicate_material:
            raise JointDuplicateAuditError(
                f"P4 duplicate-audit material differs from source for {sample_id}"
            )
        transition = replay_transitions.get(sample_id)
        if not isinstance(transition, Mapping):
            raise JointDuplicateAuditError(
                f"P4 review candidate transition cannot be replayed: {sample_id}"
            )
        public_transition = _public_value(
            {key: value for key, value in transition.items() if key != "source_meta"}
        )
        if canonical_sha256(public_transition) != queue_row.get("transition_sha256"):
            raise JointDuplicateAuditError(
                f"P4 review candidate transition hash differs for {sample_id}"
            )
        try:
            source_material = canonical_p4_memory_source_material(
                queue_row.get("memory_item_source_material")
            )
        except VerificationEvidenceError as error:
            raise JointDuplicateAuditError(
                f"P4 source material is invalid for {sample_id}: {error}"
            ) from error
        source_review_status = str(meta.get("review_status") or "").strip().lower()
        source_task_id = str(
            meta.get("task_id") or meta.get("original_task_id") or ""
        ).strip()
        source_episode_id = str(
            meta.get("episode_id")
            or meta.get("trajectory_id")
            or source_task_id
        ).strip()
        source_step = meta.get("step_index", transition.get("source_step_index"))
        if type(source_step) is bool:
            raise JointDuplicateAuditError(
                f"P4 source step is invalid for {sample_id}"
            )
        try:
            source_step = int(source_step)
        except (TypeError, ValueError) as error:
            raise JointDuplicateAuditError(
                f"P4 source step is invalid for {sample_id}"
            ) from error
        direct = source_role == "retry_abort_supplement_v2"
        try:
            replay_pre_hash = recovery_state_evidence_sha256(
                transition, field="failure_state"
            )
            replay_action_hash = recovery_action_evidence_sha256(transition)
            replay_post_hash = recovery_state_evidence_sha256(
                transition, field="post_recovery_state"
            )
        except VerificationEvidenceError as error:
            raise JointDuplicateAuditError(
                f"P4 state/action evidence cannot be replayed for {sample_id}: "
                f"{error}"
            ) from error
        expected_material = {
            "source_dataset_role": source_role,
            "source_review_status": source_review_status,
            "source_sample_id": sample_id,
            "recovery_sample_id": str(transition.get("recovery_sample_id") or ""),
            "canonical_task_id": source_task_id,
            "episode_id": source_episode_id,
            "source_step_index": source_step,
            "source_record_sha256": canonical_sha256(_public_value(record)),
            "transition_sha256": canonical_sha256(public_transition),
            "source_transition_kind": (
                "direct_recovery_from_observed_failure_state"
                if direct
                else "adjacent_failure_then_recovery"
            ),
            "task_description": expected_duplicate_material["task_description"],
            "website_domain": expected_duplicate_material["website_domain"],
            "observed_failure_basis": (
                "source_attested_direct_pre_recovery_state"
                if direct
                else "source_outcome_failure"
            ),
            "source_outcome_label": str(labels.get("outcome_label") or "").strip(),
            "source_failure_type": str(labels.get("failure_type_4") or "").strip(),
            "source_action_type": str(labels.get("action_type") or "").strip(),
            "failure_type": (
                "UNAVAILABLE"
                if direct
                else str(labels.get("failure_type_4") or "").strip()
            ),
            "failure_type_available": not direct,
            "failed_action": (
                "UNAVAILABLE"
                if direct
                else str(labels.get("action_type") or "").strip()
            ),
            "failed_action_available": not direct,
            "recovery_strategy": str(
                labels.get("recovery_strategy") or ""
            ).strip(),
            "reflection_text": str(labels.get("reflection_text") or ""),
            "pre_recovery_state_sha256": replay_pre_hash,
            "executed_recovery_action": str(
                transition.get("executed_recovery_action") or ""
            ).strip(),
            "recovery_action_value": str(
                transition.get("recovery_action_value") or ""
            ),
            "executed_recovery_action_sha256": replay_action_hash,
            "post_recovery_state_sha256": replay_post_hash,
        }
        for field, expected in expected_material.items():
            if source_material.get(field) != expected:
                raise JointDuplicateAuditError(
                    f"P4 source material differs from reopened train bytes for "
                    f"{sample_id} at {field}"
                )
        task_id = str(queue_row.get("canonical_task_id") or "").strip()
        base_entity = _base_entity(
                entity_id=f"candidate:{sample_id}",
                entity_kind="gold_train_candidate",
                source_id=sample_id,
                content_sha256=str(queue_row["source_record_sha256"]),
                goal=expected_duplicate_material["task_description"],
                sites=(expected_duplicate_material["website_domain"],),
            )
        entities.append(
            base_entity
            | {
                "canonical_task_id": task_id,
                "source_review_status": source_review_status,
                "source_transition_kind": source_material[
                    "source_transition_kind"
                ],
                "memory_item_source_material_sha256": canonical_sha256(
                    source_material
                ),
            }
        )
    return entities, {
        "train_json_files": source_summaries,
        "train_rows_read": total_rows,
        "train_candidate_rows_read": len(queue_rows),
    }


def _load_approved_task_entities(
    *, resolved_task_export_path: Path, approved_task_registry_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_export = _load_json(
        resolved_task_export_path, role="resolved WebArena development-task export"
    )
    registry = _load_json(
        approved_task_registry_path, role="approved WebArena development registry"
    )
    boundary = {
        "schema_version": "table2-webarena-public-task-export-v1",
        "record_type": "SealedWebArenaDevelopmentTaskExport",
        "benchmark": "webarena",
        "partition": "development",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "locked_test_content": False,
        "final_paper_evaluation_eligible": False,
        "required_task_count": PILOT_TASK_COUNT,
    }
    mismatches = {
        field: {"expected": expected, "actual": task_export.get(field)}
        for field, expected in boundary.items()
        if task_export.get(field) != expected
    }
    if mismatches:
        raise JointDuplicateAuditError(
            f"resolved WebArena development boundary mismatch: {mismatches}"
        )
    registry_boundary = {
        "schema_version": "1.0",
        "benchmark": "webarena",
        "partition": "development",
        "registration_status": "FROZEN_DEVELOPMENT_EXCLUSION",
        "required_task_count": PILOT_TASK_COUNT,
        "final_paper_evaluation_eligible": False,
        "locked_test_content": False,
    }
    for field, expected in registry_boundary.items():
        if registry.get(field) != expected:
            raise JointDuplicateAuditError(
                f"approved WebArena registry mismatch at {field}"
            )
    registry_hash = sha256_file(approved_task_registry_path)
    if task_export.get("registry_manifest_sha256") != registry_hash:
        raise JointDuplicateAuditError(
            "resolved task export cites a different approved registry"
        )
    if task_export.get("registry_manifest_id") != registry.get("manifest_id"):
        raise JointDuplicateAuditError(
            "resolved task export cites a different registry identity"
        )
    task_rows = task_export.get("tasks")
    registry_rows = registry.get("tasks")
    if (
        not isinstance(task_rows, list)
        or len(task_rows) != PILOT_TASK_COUNT
        or not all(isinstance(row, Mapping) for row in task_rows)
        or not isinstance(registry_rows, list)
        or len(registry_rows) != PILOT_TASK_COUNT
        or not all(isinstance(row, Mapping) for row in registry_rows)
    ):
        raise JointDuplicateAuditError(
            "approved registry/export must each contain exactly 50 task objects"
        )
    if task_export.get("resolved_task_set_sha256") != canonical_sha256(task_rows):
        raise JointDuplicateAuditError("resolved WebArena task-set hash mismatch")
    source = task_export.get("source")
    if not isinstance(source, Mapping):
        raise JointDuplicateAuditError("resolved task export lacks source attestation")
    for field in (
        "container_sha256",
        "authorized_container_sha256",
        "task_source_sha256",
    ):
        _require_sha256(source.get(field), field=f"task_export.source.{field}")

    entities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for position, (task, registered) in enumerate(
        zip(task_rows, registry_rows, strict=True)
    ):
        task_id = str(task.get("task_id") or "").strip()
        registry_task_id = str(registered.get("task_id") or "").strip()
        upstream_index = task.get("upstream_index")
        if (
            not task_id
            or task_id != registry_task_id
            or type(upstream_index) is not int
            or upstream_index < 0
            or upstream_index != registered.get("upstream_index")
            or task_id in seen_ids
            or upstream_index in seen_indices
        ):
            raise JointDuplicateAuditError(
                f"resolved/approved WebArena identity mismatch at position {position}"
            )
        seen_ids.add(task_id)
        seen_indices.add(upstream_index)
        task_config = task.get("task_config")
        if not isinstance(task_config, Mapping):
            raise JointDuplicateAuditError(f"WebArena task {task_id} lacks task_config")
        raw_sites = task_config.get("sites")
        if not isinstance(raw_sites, list):
            raise JointDuplicateAuditError(f"WebArena task {task_id} lacks sites")
        sites = [str(value) for value in raw_sites]
        entities.append(
            _base_entity(
                entity_id=f"task:{task_id}",
                entity_kind="webarena_development_task",
                source_id=task_id,
                content_sha256=duplicate_audit_task_content_sha256(task),
                goal=str(task.get("instruction") or ""),
                sites=sites,
            )
            | {
                "upstream_index": upstream_index,
                "benchmark_task_id": str(task.get("benchmark_task_id") or ""),
            }
        )
    return entities, {
        "resolved_task_export_sha256": sha256_file(resolved_task_export_path),
        "resolved_task_set_sha256": task_export["resolved_task_set_sha256"],
        "approved_task_registry_sha256": registry_hash,
        "approved_task_registry_id": registry["manifest_id"],
        "development_tasks_read": len(task_rows),
        "task_rows": task_rows,
    }


def _load_registered_recovery_diagnostics(
    *,
    recovery_scenarios_path: Path,
    duplicate_audit_registration_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Copy only the canonical diagnostic-only rows from tracked registration.

    These rows are not inferred from Gold data and are never relabelled
    ``VERIFIED``.  Their exact registered bytes and the scenario identities are
    both bound into the first-stage package.
    """

    scenarios_payload = _load_json(
        recovery_scenarios_path, role="registered recovery-scenario manifest"
    )
    fixed = {
        "schema_version": "1.0",
        "manifest_id": "table2-deterministic-recovery-15-v1",
        "evidence_label": "PILOT_ONLY",
        "required_scenario_count": RECOVERY_SCENARIO_COUNT,
        "deterministic": True,
        "final_paper_evaluation_eligible": False,
        "duplicate_audit_scope": "SYNTHETIC_DIAGNOSTIC_ONLY",
    }
    for field, expected in fixed.items():
        if scenarios_payload.get(field) != expected:
            raise JointDuplicateAuditError(
                f"registered recovery-scenario mismatch at {field}"
            )
    scenarios = scenarios_payload.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != RECOVERY_SCENARIO_COUNT
        or not all(isinstance(row, Mapping) for row in scenarios)
    ):
        raise JointDuplicateAuditError(
            "registered recovery manifest must contain exactly 15 scenarios"
        )
    scenario_ids: list[str] = []
    for index, row in enumerate(scenarios):
        if set(row) != {
            "scenario_id",
            "failure_kind",
            "expected_strategy",
            "resolution_event",
        }:
            raise JointDuplicateAuditError(
                f"registered recovery scenario {index} fields differ from schema"
            )
        scenario_id = str(row.get("scenario_id") or "").strip()
        if not scenario_id or any(not str(row.get(field) or "").strip() for field in (
            "failure_kind",
            "expected_strategy",
            "resolution_event",
        )):
            raise JointDuplicateAuditError(
                f"registered recovery scenario {index} is incomplete"
            )
        scenario_ids.append(scenario_id)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise JointDuplicateAuditError("registered recovery scenario IDs repeat")

    registration_payload = _load_json(
        duplicate_audit_registration_path,
        role="registered pilot duplicate-audit manifest",
    )
    if (
        registration_payload.get("manifest_id")
        != "table2-pilot-duplicate-audit-registration-v1"
        or registration_payload.get("manifest_state") != "FROZEN_REGISTRATION"
        or registration_payload.get("synthetic_recovery_evidence_scope")
        != "DIAGNOSTIC_ONLY_NOT_PRIMARY"
        or registration_payload.get("required_recovery_scenario_count")
        != RECOVERY_SCENARIO_COUNT
    ):
        raise JointDuplicateAuditError(
            "pilot duplicate registration is not the canonical diagnostic authority"
        )
    try:
        registered = FrozenDuplicateAuditManifest.from_path(
            duplicate_audit_registration_path
        )
    except DuplicateAuditError as error:
        raise JointDuplicateAuditError(
            f"invalid registered pilot duplicate audit: {error}"
        ) from error
    entries = registration_payload.get("entries")
    if not isinstance(entries, list):
        raise JointDuplicateAuditError("registered duplicate audit lacks entries")
    diagnostic_rows = [
        dict(row)
        for row in entries
        if isinstance(row, Mapping)
        and row.get("task_partition") == "recovery_diagnostic"
    ]
    if [str(row.get("task_id") or "") for row in diagnostic_rows] != scenario_ids:
        raise JointDuplicateAuditError(
            "registered diagnostic rows differ from recovery-scenario order"
        )
    for scenario_id in scenario_ids:
        registered.clusters_for(
            scenario_id,
            task_partition="recovery_diagnostic",
            allow_synthetic_diagnostic=True,
        )
    return [dict(row) for row in scenarios], diagnostic_rows, {
        "recovery_scenarios_sha256": sha256_file(recovery_scenarios_path),
        "duplicate_audit_registration_sha256": sha256_file(
            duplicate_audit_registration_path
        ),
        "recovery_scenario_count": len(scenario_ids),
        "recovery_scenario_ids": scenario_ids,
    }


def _namespace(
    *, config_sha256: str, source_sha256: str, input_binding: Mapping[str, Any]
) -> JointDuplicateClusterNamespace:
    namespace_suffix = canonical_sha256(
        {
            "audit_tool_config_sha256": config_sha256,
            "audit_tool_source_sha256": source_sha256,
            "input_binding": input_binding,
        }
    )[:24]
    return JointDuplicateClusterNamespace(
        namespace_id=f"table2-gold-train-webarena-{namespace_suffix}",
        audit_tool_id=_REGISTERED_CONFIG["audit_tool_id"],
        audit_tool_version=_REGISTERED_CONFIG["audit_tool_version"],
        audit_tool_config_sha256=config_sha256,
        audit_tool_source_sha256=source_sha256,
    )


def _jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[int, int]:
    left_features = set(str(value) for value in left["features"])
    right_features = set(str(value) for value in right["features"])
    union = left_features | right_features
    return len(left_features & right_features), len(union)


def _linked(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[bool, str, int, int]:
    numerator, denominator = _jaccard(left, right)
    if left["exact_duplicate_key"] == right["exact_duplicate_key"]:
        return True, "EXACT", numerator, denominator
    left_sites = set(left["site_keys"])
    right_sites = set(right["site_keys"])
    if left_sites and right_sites and not (left_sites & right_sites):
        return False, "SITE_MISMATCH", numerator, denominator
    threshold_numerator = int(
        _REGISTERED_CONFIG["near_match"]["jaccard_threshold_numerator"]
    )
    threshold_denominator = int(
        _REGISTERED_CONFIG["near_match"]["jaccard_threshold_denominator"]
    )
    accepted = (
        denominator > 0
        and numerator * threshold_denominator
        >= threshold_numerator * denominator
    )
    return accepted, "NEAR" if accepted else "BELOW_THRESHOLD", numerator, denominator


def _cluster_entities(
    base_entities: Sequence[Mapping[str, Any]],
    *, namespace: JointDuplicateClusterNamespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entities = [dict(row) for row in sorted(base_entities, key=lambda row: row["entity_id"])]
    ids = [str(row["entity_id"]) for row in entities]
    if len(ids) != len(set(ids)):
        raise JointDuplicateAuditError("duplicate joint-audit entity ID")
    parent = list(range(len(entities)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if ids[left_root] <= ids[right_root]:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    accepted_links: list[dict[str, Any]] = []
    # A Jaccard-positive pair must share at least one registered feature.  The
    # inverted index therefore removes provably-zero pairs without changing a
    # single clustering decision.  Sorted pair replay keeps results invariant
    # to source row order and dictionary iteration order.
    feature_members: dict[str, list[int]] = {}
    for index, entity in enumerate(entities):
        for feature in entity["features"]:
            feature_members.setdefault(str(feature), []).append(index)
    candidate_pairs: set[tuple[int, int]] = set()
    for feature in sorted(feature_members):
        members = feature_members[feature]
        for left_offset, left_index in enumerate(members):
            for right_index in members[left_offset + 1 :]:
                candidate_pairs.add((left_index, right_index))
    for left_index, right_index in sorted(candidate_pairs):
        left = entities[left_index]
        right = entities[right_index]
        accepted, relation, numerator, denominator = _linked(left, right)
        if not accepted:
            continue
        union(left_index, right_index)
        accepted_links.append(
            {
                "left_entity_id": left["entity_id"],
                "right_entity_id": right["entity_id"],
                "relation": relation,
                "jaccard_numerator": numerator,
                "jaccard_denominator": denominator,
            }
        )

    components: dict[int, list[int]] = {}
    for index in range(len(entities)):
        components.setdefault(find(index), []).append(index)
    cluster_rows: list[dict[str, Any]] = []
    for member_indices in components.values():
        member_ids = sorted(ids[index] for index in member_indices)
        near_id = "near:" + canonical_sha256(
            {
                "namespace_id": namespace.namespace_id,
                "member_entity_ids": member_ids,
                "linkage": _REGISTERED_CONFIG["near_match"]["linkage"],
            }
        )
        member_kinds = sorted({entities[index]["entity_kind"] for index in member_indices})
        cluster_rows.append(
            {
                "near_duplicate_cluster_id": near_id,
                "member_entity_ids": member_ids,
                "member_count": len(member_ids),
                "member_kinds": member_kinds,
                "contains_cross_corpus_members": len(member_kinds) > 1,
                "exact_duplicate_keys": sorted(
                    {entities[index]["exact_duplicate_key"] for index in member_indices}
                ),
            }
        )
        for index in member_indices:
            exact_cluster = "exact:" + str(entities[index]["exact_duplicate_key"])
            entities[index]["near_duplicate_cluster_id"] = near_id
            entities[index]["cluster_ids"] = sorted([exact_cluster, near_id])

    cluster_rows.sort(key=lambda row: row["near_duplicate_cluster_id"])
    accepted_links.sort(
        key=lambda row: (row["left_entity_id"], row["right_entity_id"])
    )
    cluster_payload = {
        "schema_version": ASSIGNMENT_CLUSTERS_SCHEMA_VERSION,
        "duplicate_cluster_namespace": namespace.to_dict(),
        "entity_count": len(entities),
        "train_candidate_count": sum(
            row["entity_kind"] == "gold_train_candidate" for row in entities
        ),
        "development_task_count": sum(
            row["entity_kind"] == "webarena_development_task" for row in entities
        ),
        "exact_group_count": len({row["exact_duplicate_key"] for row in entities}),
        "near_cluster_count": len(cluster_rows),
        "cross_corpus_cluster_count": sum(
            row["contains_cross_corpus_members"] for row in cluster_rows
        ),
        "accepted_link_count": len(accepted_links),
        "accepted_links": accepted_links,
        "clusters": cluster_rows,
    }
    return entities, cluster_payload


def _compute_assignment(
    *,
    config_path: Path,
    source_authority_path: Path,
    preparation_root: Path,
    gold_train_json: Path,
    resolved_task_export_path: Path,
    approved_task_registry_path: Path,
    recovery_scenarios_path: Path,
    duplicate_audit_registration_path: Path,
    supplement_train_json: Path | None,
) -> dict[str, Any]:
    config, config_sha256, source_sha256 = _load_config(config_path)
    source_manifest = _audit_source_manifest(config_path)
    try:
        package = validate_p4_preparation_package(preparation_root)
    except P4PreparationError as error:
        raise JointDuplicateAuditError(
            f"invalid P4 preparation package: {error}"
        ) from error
    if package.status != "REVIEW_REQUIRED" or package.candidate_count <= 0:
        raise JointDuplicateAuditError(
            "joint duplicate assignment requires a nonempty REVIEW_REQUIRED preparation"
        )
    _require_regular_file(source_authority_path, role="P4 source authority")
    if (package.root / P4_SOURCE_AUTHORITY_FILE_NAME).read_bytes() != (
        source_authority_path.read_bytes()
    ):
        raise JointDuplicateAuditError(
            "P4 preparation source authority differs from registered authority"
        )
    train_entities, train_reads = _load_training_candidate_entities(
        package_root=package.root,
        gold_train_json=gold_train_json,
        supplement_train_json=supplement_train_json,
    )
    task_entities, task_reads = _load_approved_task_entities(
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=approved_task_registry_path,
    )
    recovery_scenarios, recovery_entries, recovery_reads = (
        _load_registered_recovery_diagnostics(
            recovery_scenarios_path=recovery_scenarios_path,
            duplicate_audit_registration_path=duplicate_audit_registration_path,
        )
    )
    preparation_manifest_sha256 = sha256_file(
        package.root / "preparation_manifest.json"
    )
    input_binding = {
        "preparation_manifest_sha256": preparation_manifest_sha256,
        "preparation_records_sha256": package.manifest["records_sha256"],
        "review_queue_sha256": package.manifest["review_queue_sha256"],
        "source_authority_sha256": package.manifest[
            "source_authority_sha256"
        ],
        "dataset_artifacts_sha256": package.manifest[
            "dataset_artifacts_sha256"
        ],
        "resolved_task_export_sha256": task_reads["resolved_task_export_sha256"],
        "resolved_task_set_sha256": task_reads["resolved_task_set_sha256"],
        "approved_task_registry_sha256": task_reads[
            "approved_task_registry_sha256"
        ],
        "recovery_scenarios_sha256": recovery_reads[
            "recovery_scenarios_sha256"
        ],
        "duplicate_audit_registration_sha256": recovery_reads[
            "duplicate_audit_registration_sha256"
        ],
        "audit_dependency_source_set_sha256": source_manifest["files_sha256"],
    }
    namespace = _namespace(
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        input_binding=input_binding,
    )
    entities, clusters = _cluster_entities(
        [*train_entities, *task_entities], namespace=namespace
    )
    read_ledger = {
        "schema_version": ASSIGNMENT_READ_LEDGER_SCHEMA_VERSION,
        "source_split": "train_plus_public_development_tasks",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "train_json_files": train_reads["train_json_files"],
        "train_rows_read": train_reads["train_rows_read"],
        "train_candidate_rows_read": train_reads["train_candidate_rows_read"],
        "source_authority_sha256": package.manifest[
            "source_authority_sha256"
        ],
        "development_tasks_read": task_reads["development_tasks_read"],
        "resolved_task_export_sha256": task_reads["resolved_task_export_sha256"],
        "approved_task_registry_sha256": task_reads[
            "approved_task_registry_sha256"
        ],
        "independent_recovery_reviews_read": 0,
        "independent_final_task_reviews_read": 0,
        "registered_recovery_scenarios_read": recovery_reads[
            "recovery_scenario_count"
        ],
    }
    return {
        "config": config,
        "config_bytes": config_path.read_bytes(),
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "source_manifest": source_manifest,
        "namespace": namespace,
        "entities": entities,
        "clusters": clusters,
        "read_ledger": read_ledger,
        "input_binding": input_binding,
        "task_rows": task_reads["task_rows"],
        "recovery_scenarios": recovery_scenarios,
        "recovery_entries": recovery_entries,
        "recovery_scenario_ids": recovery_reads["recovery_scenario_ids"],
    }


def _descriptor(path: Path) -> dict[str, Any]:
    _require_regular_file(path, role=f"assignment payload {path.name}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_pretty_bytes(value))


def _write_entities(path: Path, entities: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as stream:
        for entity in entities:
            stream.write(canonical_json_bytes(entity))
            stream.write(b"\n")


def build_joint_duplicate_assignment_package(
    *,
    config_path: str | Path,
    source_authority_path: str | Path,
    preparation_root: str | Path,
    gold_train_json: str | Path,
    resolved_task_export_path: str | Path,
    approved_task_registry_path: str | Path,
    recovery_scenarios_path: str | Path,
    duplicate_audit_registration_path: str | Path,
    output_dir: str | Path,
    supplement_train_json: str | Path | None = None,
) -> JointDuplicateAssignmentPackage:
    """Build immutable deterministic assignments without authoring review evidence."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite assignment package: {destination}")
    computed = _compute_assignment(
        config_path=Path(config_path),
        source_authority_path=Path(source_authority_path),
        preparation_root=Path(preparation_root),
        gold_train_json=Path(gold_train_json),
        resolved_task_export_path=Path(resolved_task_export_path),
        approved_task_registry_path=Path(approved_task_registry_path),
        recovery_scenarios_path=Path(recovery_scenarios_path),
        duplicate_audit_registration_path=Path(
            duplicate_audit_registration_path
        ),
        supplement_train_json=(
            Path(supplement_train_json) if supplement_train_json is not None else None
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        (temporary / _CONFIG_FILE_NAME).write_bytes(computed["config_bytes"])
        _write_entities(temporary / _ENTITIES_FILE_NAME, computed["entities"])
        _atomic_json(temporary / _CLUSTERS_FILE_NAME, computed["clusters"])
        _atomic_json(temporary / _READ_LEDGER_FILE_NAME, computed["read_ledger"])
        _atomic_json(
            temporary / _SOURCE_MANIFEST_FILE_NAME,
            computed["source_manifest"],
        )
        descriptors = {
            name: _descriptor(temporary / name) for name in _PAYLOAD_FILES
        }
        manifest = {
            "schema_version": ASSIGNMENT_PACKAGE_SCHEMA_VERSION,
            "package_state": "FROZEN_ASSIGNMENTS",
            "evidence_label": "PILOT_ONLY",
            "paper_table_status": "N/R",
            "evidence_role": (
                "DETERMINISTIC_DUPLICATE_ASSIGNMENT_NOT_RECOVERY_OR_SUCCESS_VERIFICATION"
            ),
            "source_implementation_relative_path": AUDIT_TOOL_SOURCE_RELATIVE_PATH,
            "audit_tool_config_sha256": computed["config_sha256"],
            "audit_tool_source_sha256": computed["source_sha256"],
            "duplicate_cluster_namespace": computed["namespace"].to_dict(),
            "input_binding": computed["input_binding"],
            "entity_count": len(computed["entities"]),
            "train_candidate_count": computed["read_ledger"][
                "train_candidate_rows_read"
            ],
            "development_task_count": PILOT_TASK_COUNT,
            "recovery_scenario_count": RECOVERY_SCENARIO_COUNT,
            "entities_sha256": canonical_sha256(computed["entities"]),
            "clusters_sha256": canonical_sha256(computed["clusters"]),
            "validation_rows_read": 0,
            "test_rows_read": 0,
            "locked_test_rows_read": 0,
            "independent_recovery_or_final_success_evidence_created": 0,
            "provenance_manifest_created": False,
            "files": descriptors,
        }
        _atomic_json(temporary / _MANIFEST_FILE_NAME, manifest)
        (temporary / _MANIFEST_SIDECAR_NAME).write_text(
            sha256_file(temporary / _MANIFEST_FILE_NAME) + "\n", encoding="utf-8"
        )
        for path in temporary.iterdir():
            path.chmod(0o444)
        os.replace(temporary, destination)
        destination.chmod(0o555)
    except Exception:
        if temporary.exists():
            for path in temporary.iterdir():
                path.chmod(0o644)
            temporary.chmod(0o755)
            shutil.rmtree(temporary)
        raise
    return validate_joint_duplicate_assignment_package(
        package_root=destination,
        config_path=config_path,
        source_authority_path=source_authority_path,
        preparation_root=preparation_root,
        gold_train_json=gold_train_json,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=approved_task_registry_path,
        recovery_scenarios_path=recovery_scenarios_path,
        duplicate_audit_registration_path=duplicate_audit_registration_path,
        supplement_train_json=supplement_train_json,
    )


def _read_entities(path: Path) -> list[dict[str, Any]]:
    _require_regular_file(path, role="joint duplicate assignment entities")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            raise JointDuplicateAuditError(f"blank assignment entity line {line_number}")
        try:
            row = json.loads(raw, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as error:
            raise JointDuplicateAuditError(
                f"invalid assignment entity line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise JointDuplicateAuditError(
                f"assignment entity line {line_number} must be an object"
            )
        rows.append(row)
    return rows


def validate_joint_duplicate_assignment_package(
    *,
    package_root: str | Path,
    config_path: str | Path,
    source_authority_path: str | Path,
    preparation_root: str | Path,
    gold_train_json: str | Path,
    resolved_task_export_path: str | Path,
    approved_task_registry_path: str | Path,
    recovery_scenarios_path: str | Path,
    duplicate_audit_registration_path: str | Path,
    supplement_train_json: str | Path | None = None,
) -> JointDuplicateAssignmentPackage:
    """Reopen every permitted input and reproduce the complete assignment."""

    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise JointDuplicateAuditError("assignment package must be a non-symlink directory")
    manifest = _load_json(root / _MANIFEST_FILE_NAME, role="assignment manifest")
    expected_fields = {
        "schema_version",
        "package_state",
        "evidence_label",
        "paper_table_status",
        "evidence_role",
        "source_implementation_relative_path",
        "audit_tool_config_sha256",
        "audit_tool_source_sha256",
        "duplicate_cluster_namespace",
        "input_binding",
        "entity_count",
        "train_candidate_count",
        "development_task_count",
        "recovery_scenario_count",
        "entities_sha256",
        "clusters_sha256",
        *_ZERO_READ_FIELDS,
        "independent_recovery_or_final_success_evidence_created",
        "provenance_manifest_created",
        "files",
    }
    if set(manifest) != expected_fields:
        raise JointDuplicateAuditError("assignment manifest fields differ from schema")
    fixed = {
        "schema_version": ASSIGNMENT_PACKAGE_SCHEMA_VERSION,
        "package_state": "FROZEN_ASSIGNMENTS",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "evidence_role": (
            "DETERMINISTIC_DUPLICATE_ASSIGNMENT_NOT_RECOVERY_OR_SUCCESS_VERIFICATION"
        ),
        "source_implementation_relative_path": AUDIT_TOOL_SOURCE_RELATIVE_PATH,
        "development_task_count": PILOT_TASK_COUNT,
        "recovery_scenario_count": RECOVERY_SCENARIO_COUNT,
        "independent_recovery_or_final_success_evidence_created": 0,
        "provenance_manifest_created": False,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise JointDuplicateAuditError(f"assignment manifest mismatch at {field}")
    _require_zero_reads(manifest, role="assignment manifest")
    sidecar = root / _MANIFEST_SIDECAR_NAME
    _require_regular_file(sidecar, role="assignment manifest sidecar")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(
        root / _MANIFEST_FILE_NAME
    ):
        raise JointDuplicateAuditError("assignment manifest sidecar mismatch")
    descriptors = manifest.get("files")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(_PAYLOAD_FILES):
        raise JointDuplicateAuditError("assignment package payload file set is invalid")
    for name in _PAYLOAD_FILES:
        if descriptors[name] != _descriptor(root / name):
            raise JointDuplicateAuditError(f"assignment package payload changed: {name}")

    computed = _compute_assignment(
        config_path=Path(config_path),
        source_authority_path=Path(source_authority_path),
        preparation_root=Path(preparation_root),
        gold_train_json=Path(gold_train_json),
        resolved_task_export_path=Path(resolved_task_export_path),
        approved_task_registry_path=Path(approved_task_registry_path),
        recovery_scenarios_path=Path(recovery_scenarios_path),
        duplicate_audit_registration_path=Path(
            duplicate_audit_registration_path
        ),
        supplement_train_json=(
            Path(supplement_train_json) if supplement_train_json is not None else None
        ),
    )
    if (root / _CONFIG_FILE_NAME).read_bytes() != computed["config_bytes"]:
        raise JointDuplicateAuditError("frozen audit config bytes changed")
    entities = _read_entities(root / _ENTITIES_FILE_NAME)
    clusters = _load_json(root / _CLUSTERS_FILE_NAME, role="assignment clusters")
    ledger = _load_json(root / _READ_LEDGER_FILE_NAME, role="assignment read ledger")
    if entities != computed["entities"]:
        raise JointDuplicateAuditError(
            "joint assignments differ from exact source replay"
        )
    if clusters != computed["clusters"]:
        raise JointDuplicateAuditError("joint clusters differ from exact source replay")
    if ledger != computed["read_ledger"]:
        raise JointDuplicateAuditError("joint audit read ledger differs from source replay")
    source_manifest = _load_json(
        root / _SOURCE_MANIFEST_FILE_NAME,
        role="assignment source manifest",
    )
    if source_manifest != computed["source_manifest"]:
        raise JointDuplicateAuditError(
            "joint audit source dependency manifest differs from source replay"
        )
    semantic_expected = {
        "audit_tool_config_sha256": computed["config_sha256"],
        "audit_tool_source_sha256": computed["source_sha256"],
        "duplicate_cluster_namespace": computed["namespace"].to_dict(),
        "input_binding": computed["input_binding"],
        "entity_count": len(entities),
        "train_candidate_count": len(
            [row for row in entities if row["entity_kind"] == "gold_train_candidate"]
        ),
        "entities_sha256": canonical_sha256(entities),
        "clusters_sha256": canonical_sha256(clusters),
    }
    for field, expected in semantic_expected.items():
        if manifest.get(field) != expected:
            raise JointDuplicateAuditError(f"assignment manifest mismatch at {field}")
    try:
        namespace = JointDuplicateClusterNamespace.from_mapping(
            manifest["duplicate_cluster_namespace"], require_hashes=True
        )
    except DuplicateAuditError as error:
        raise JointDuplicateAuditError(f"invalid assignment namespace: {error}") from error
    return JointDuplicateAssignmentPackage(
        root=root.resolve(),
        manifest=manifest,
        namespace=namespace,
        entities=tuple(entities),
        clusters=clusters,
    )


def validate_compact_joint_duplicate_evidence(
    *,
    package_root: str | Path,
    config_path: str | Path,
    source_authority_path: str | Path,
    preparation_root: str | Path,
    resolved_task_export_path: str | Path,
    approved_task_registry_path: str | Path,
    recovery_scenarios_path: str | Path,
    duplicate_audit_registration_path: str | Path,
    provenance_manifest_path: str | Path | None = None,
    final_audit_path: str | Path | None = None,
) -> JointDuplicateAssignmentPackage:
    """Validate the transferable assignment/preparation closure without Gold.

    The source host's production prebuild gate performs the stronger replay
    against every Gold training row.  This compact verifier is for campaign
    transfer: it revalidates the complete preparation and assignment package,
    tracked algorithm/config identities, all public-task assignments, all
    candidate source-record identities, and deterministically recomputes every
    exact/near cluster.  It never opens Gold, validation, test, or locked data.
    """

    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise JointDuplicateAuditError(
            "assignment package must be a non-symlink directory"
        )
    manifest = _load_json(root / _MANIFEST_FILE_NAME, role="assignment manifest")
    expected_manifest_fields = {
        "schema_version",
        "package_state",
        "evidence_label",
        "paper_table_status",
        "evidence_role",
        "source_implementation_relative_path",
        "audit_tool_config_sha256",
        "audit_tool_source_sha256",
        "duplicate_cluster_namespace",
        "input_binding",
        "entity_count",
        "train_candidate_count",
        "development_task_count",
        "recovery_scenario_count",
        "entities_sha256",
        "clusters_sha256",
        *_ZERO_READ_FIELDS,
        "independent_recovery_or_final_success_evidence_created",
        "provenance_manifest_created",
        "files",
    }
    if set(manifest) != expected_manifest_fields:
        raise JointDuplicateAuditError("assignment manifest fields differ from schema")
    fixed = {
        "schema_version": ASSIGNMENT_PACKAGE_SCHEMA_VERSION,
        "package_state": "FROZEN_ASSIGNMENTS",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "evidence_role": (
            "DETERMINISTIC_DUPLICATE_ASSIGNMENT_NOT_RECOVERY_OR_SUCCESS_VERIFICATION"
        ),
        "source_implementation_relative_path": AUDIT_TOOL_SOURCE_RELATIVE_PATH,
        "development_task_count": PILOT_TASK_COUNT,
        "recovery_scenario_count": RECOVERY_SCENARIO_COUNT,
        "independent_recovery_or_final_success_evidence_created": 0,
        "provenance_manifest_created": False,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise JointDuplicateAuditError(f"assignment manifest mismatch at {field}")
    _require_zero_reads(manifest, role="assignment manifest")
    sidecar = root / _MANIFEST_SIDECAR_NAME
    _require_regular_file(sidecar, role="assignment manifest sidecar")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(
        root / _MANIFEST_FILE_NAME
    ):
        raise JointDuplicateAuditError("assignment manifest sidecar mismatch")
    descriptors = manifest.get("files")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(_PAYLOAD_FILES):
        raise JointDuplicateAuditError("assignment package payload file set is invalid")
    for name in _PAYLOAD_FILES:
        if descriptors[name] != _descriptor(root / name):
            raise JointDuplicateAuditError(f"assignment package payload changed: {name}")

    _, config_sha256, source_sha256 = _load_config(Path(config_path))
    source_manifest = _audit_source_manifest(Path(config_path))
    if (root / _CONFIG_FILE_NAME).read_bytes() != Path(config_path).read_bytes():
        raise JointDuplicateAuditError("frozen audit config bytes changed")
    try:
        preparation = validate_p4_preparation_package(preparation_root)
    except P4PreparationError as error:
        raise JointDuplicateAuditError(
            f"invalid compact P4 preparation package: {error}"
        ) from error
    source_authority = Path(source_authority_path)
    _require_regular_file(source_authority, role="registered P4 source authority")
    if (preparation.root / P4_SOURCE_AUTHORITY_FILE_NAME).read_bytes() != (
        source_authority.read_bytes()
    ):
        raise JointDuplicateAuditError(
            "compact P4 source authority differs from registered authority"
        )
    if preparation.status != "REVIEW_REQUIRED" or preparation.candidate_count <= 0:
        raise JointDuplicateAuditError(
            "compact duplicate evidence requires a nonempty REVIEW_REQUIRED preparation"
        )
    task_entities, task_reads = _load_approved_task_entities(
        resolved_task_export_path=Path(resolved_task_export_path),
        approved_task_registry_path=Path(approved_task_registry_path),
    )
    _, _, recovery_reads = _load_registered_recovery_diagnostics(
        recovery_scenarios_path=Path(recovery_scenarios_path),
        duplicate_audit_registration_path=Path(
            duplicate_audit_registration_path
        ),
    )
    input_binding = {
        "preparation_manifest_sha256": sha256_file(
            preparation.root / "preparation_manifest.json"
        ),
        "preparation_records_sha256": preparation.manifest["records_sha256"],
        "review_queue_sha256": preparation.manifest["review_queue_sha256"],
        "source_authority_sha256": preparation.manifest[
            "source_authority_sha256"
        ],
        "dataset_artifacts_sha256": preparation.manifest[
            "dataset_artifacts_sha256"
        ],
        "resolved_task_export_sha256": task_reads["resolved_task_export_sha256"],
        "resolved_task_set_sha256": task_reads["resolved_task_set_sha256"],
        "approved_task_registry_sha256": task_reads[
            "approved_task_registry_sha256"
        ],
        "recovery_scenarios_sha256": recovery_reads[
            "recovery_scenarios_sha256"
        ],
        "duplicate_audit_registration_sha256": recovery_reads[
            "duplicate_audit_registration_sha256"
        ],
        "audit_dependency_source_set_sha256": source_manifest["files_sha256"],
    }
    if manifest.get("input_binding") != input_binding:
        raise JointDuplicateAuditError(
            "assignment input binding differs from compact evidence bytes"
        )
    namespace = _namespace(
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        input_binding=input_binding,
    )
    if manifest.get("duplicate_cluster_namespace") != namespace.to_dict():
        raise JointDuplicateAuditError(
            "assignment namespace differs from tracked source/config and inputs"
        )
    frozen_source_manifest = _load_json(
        root / _SOURCE_MANIFEST_FILE_NAME,
        role="assignment source manifest",
    )
    if frozen_source_manifest != source_manifest:
        raise JointDuplicateAuditError(
            "assignment source dependencies differ from frozen evidence"
        )

    entities = _read_entities(root / _ENTITIES_FILE_NAME)
    candidate_rows = [
        row for row in entities if row.get("entity_kind") == "gold_train_candidate"
    ]
    task_rows = [
        row
        for row in entities
        if row.get("entity_kind") == "webarena_development_task"
    ]
    if len(candidate_rows) != preparation.candidate_count:
        raise JointDuplicateAuditError(
            "assignment candidate count differs from compact preparation"
        )
    queue_by_sample = {
        str(row.get("source_sample_id") or ""): row
        for row in _load_queue_rows(preparation.root)
    }
    if len(queue_by_sample) != preparation.candidate_count or "" in queue_by_sample:
        raise JointDuplicateAuditError(
            "compact preparation has invalid candidate source identities"
        )

    cluster_fields = {"near_duplicate_cluster_id", "cluster_ids"}
    base_entities: list[dict[str, Any]] = []
    for row in candidate_rows:
        sample_id = str(row.get("source_id") or "")
        queue = queue_by_sample.get(sample_id)
        if queue is None:
            raise JointDuplicateAuditError(
                f"assignment candidate is absent from preparation: {sample_id}"
            )
        expected_identity = {
            "entity_id": f"candidate:{sample_id}",
            "content_sha256": queue.get("source_record_sha256"),
            "canonical_task_id": queue.get("canonical_task_id"),
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                raise JointDuplicateAuditError(
                    f"compact candidate assignment mismatch for {sample_id} at {field}"
                )
        duplicate_material = queue.get("duplicate_audit_material")
        if not isinstance(duplicate_material, Mapping) or set(
            duplicate_material
        ) != {"task_description", "website_domain"}:
            raise JointDuplicateAuditError(
                f"compact preparation lacks duplicate material for {sample_id}"
            )
        source_material = queue.get("memory_item_source_material")
        try:
            canonical_source_material = canonical_p4_memory_source_material(
                source_material
            )
        except VerificationEvidenceError as error:
            raise JointDuplicateAuditError(
                f"compact preparation source material is invalid for "
                f"{sample_id}: {error}"
            ) from error
        reconstructed = _base_entity(
            entity_id=str(row.get("entity_id") or ""),
            entity_kind="gold_train_candidate",
            source_id=sample_id,
            content_sha256=str(row.get("content_sha256") or ""),
            goal=str(duplicate_material.get("task_description") or ""),
            sites=(str(duplicate_material.get("website_domain") or ""),),
        ) | {
            "canonical_task_id": str(row.get("canonical_task_id") or ""),
            "source_review_status": str(queue.get("source_review_status") or ""),
            "source_transition_kind": str(
                queue.get("source_transition_kind") or ""
            ),
            "memory_item_source_material_sha256": canonical_sha256(
                canonical_source_material
            ),
        }
        if {key: row.get(key) for key in reconstructed} != reconstructed:
            raise JointDuplicateAuditError(
                f"candidate normalization/features are not canonical: {sample_id}"
            )
        if set(row) != set(reconstructed) | cluster_fields:
            raise JointDuplicateAuditError(
                f"candidate assignment fields differ from schema: {sample_id}"
            )
        base_entities.append(reconstructed)

    expected_tasks = {str(row["source_id"]): row for row in task_entities}
    if len(task_rows) != len(expected_tasks):
        raise JointDuplicateAuditError(
            "assignment task count differs from resolved development export"
        )
    for row in task_rows:
        task_id = str(row.get("source_id") or "")
        expected = expected_tasks.get(task_id)
        if expected is None:
            raise JointDuplicateAuditError(
                f"assignment task is absent from resolved export: {task_id}"
            )
        if {key: row.get(key) for key in expected} != expected:
            raise JointDuplicateAuditError(
                f"public task assignment content differs for {task_id}"
            )
        if set(row) != set(expected) | cluster_fields:
            raise JointDuplicateAuditError(
                f"task assignment fields differ from schema: {task_id}"
            )
        base_entities.append(expected)

    recomputed_entities, recomputed_clusters = _cluster_entities(
        base_entities, namespace=namespace
    )
    if entities != recomputed_entities:
        raise JointDuplicateAuditError(
            "compact exact/near assignments differ from deterministic replay"
        )
    clusters = _load_json(root / _CLUSTERS_FILE_NAME, role="assignment clusters")
    if clusters != recomputed_clusters:
        raise JointDuplicateAuditError(
            "compact exact/near clusters differ from deterministic replay"
        )
    ledger = _load_json(root / _READ_LEDGER_FILE_NAME, role="assignment read ledger")
    preparation_ledger = _load_json(
        preparation.root / "read_ledger.json",
        role="compact P4 preparation read ledger",
    )
    _require_zero_reads(ledger, role="assignment read ledger")
    if (
        ledger.get("development_tasks_read") != PILOT_TASK_COUNT
        or ledger.get("train_candidate_rows_read") != preparation.candidate_count
        or ledger.get("train_json_files")
        != preparation_ledger.get("train_json_files")
        or ledger.get("train_rows_read")
        != preparation_ledger.get("train_rows_read")
        or ledger.get("resolved_task_export_sha256")
        != input_binding["resolved_task_export_sha256"]
        or ledger.get("approved_task_registry_sha256")
        != input_binding["approved_task_registry_sha256"]
        or ledger.get("registered_recovery_scenarios_read")
        != RECOVERY_SCENARIO_COUNT
        or ledger.get("independent_recovery_reviews_read") != 0
        or ledger.get("independent_final_task_reviews_read") != 0
    ):
        raise JointDuplicateAuditError(
            "compact assignment read ledger differs from registered inputs"
        )
    semantic_expected = {
        "audit_tool_config_sha256": config_sha256,
        "audit_tool_source_sha256": source_sha256,
        "entity_count": len(entities),
        "train_candidate_count": len(candidate_rows),
        "entities_sha256": canonical_sha256(entities),
        "clusters_sha256": canonical_sha256(clusters),
    }
    for field, expected in semantic_expected.items():
        if manifest.get(field) != expected:
            raise JointDuplicateAuditError(f"assignment manifest mismatch at {field}")
    package = JointDuplicateAssignmentPackage(
        root=root.resolve(),
        manifest=manifest,
        namespace=namespace,
        entities=tuple(entities),
        clusters=clusters,
    )
    if final_audit_path is not None and provenance_manifest_path is None:
        raise JointDuplicateAuditError(
            "compact final-audit replay requires completed provenance"
        )
    if provenance_manifest_path is not None:
        provenance, _, provenance_sha256 = _validated_completed_provenance(
            package=package,
            preparation_root=preparation.root,
            provenance_manifest_path=Path(provenance_manifest_path),
        )
        if final_audit_path is not None:
            task_export = _load_json(
                Path(resolved_task_export_path),
                role="compact resolved WebArena development-task export",
            )
            recovery_entries = _load_registered_recovery_diagnostics(
                recovery_scenarios_path=Path(recovery_scenarios_path),
                duplicate_audit_registration_path=Path(
                    duplicate_audit_registration_path
                ),
            )[1]
            expected_audit = _final_audit_payload(
                package=package,
                provenance=provenance,
                provenance_sha256=provenance_sha256,
                task_rows=task_export["tasks"],
                recovery_entries=recovery_entries,
            )
            actual_audit = _load_json(
                Path(final_audit_path), role="compact final duplicate audit"
            )
            if actual_audit != expected_audit:
                raise JointDuplicateAuditError(
                    "final duplicate audit differs from compact deterministic replay"
                )
    return package


def provenance_assignment_binding(
    package: JointDuplicateAssignmentPackage,
) -> dict[str, Any]:
    """Return the exact object an external provenance author must cite."""

    namespace_value = package.namespace.to_dict()
    return {
        "schema_version": PROVENANCE_ASSIGNMENT_BINDING_SCHEMA_VERSION,
        "assignment_manifest_sha256": package.assignment_manifest_sha256,
        "entities_sha256": package.manifest["entities_sha256"],
        "clusters_sha256": package.manifest["clusters_sha256"],
        "duplicate_cluster_namespace_sha256": canonical_sha256(namespace_value),
    }


def _validated_completed_provenance(
    *,
    package: JointDuplicateAssignmentPackage,
    preparation_root: Path,
    provenance_manifest_path: Path,
) -> tuple[ProvenanceManifest, dict[str, Any], str]:
    provenance_payload = _load_json(
        provenance_manifest_path, role="completed external P4 provenance manifest"
    )
    try:
        provenance = ProvenanceManifest.from_mapping(provenance_payload)
    except (ManifestError, TypeError, ValueError) as error:
        raise JointDuplicateAuditError(f"invalid completed provenance: {error}") from error
    try:
        validation = validate_p4_provenance_against_preparation(
            package_root=preparation_root,
            provenance_manifest_path=provenance_manifest_path,
        )
    except P4PreparationError as error:
        raise JointDuplicateAuditError(
            f"completed provenance is not attached to preparation: {error}"
        ) from error
    expected_binding = provenance_assignment_binding(package)
    if provenance_payload.get("joint_duplicate_assignment_binding") != expected_binding:
        raise JointDuplicateAuditError(
            "completed provenance does not cite the exact joint assignment package"
        )
    if provenance.duplicate_cluster_namespace.to_dict() != package.namespace.to_dict():
        raise JointDuplicateAuditError(
            "completed provenance cites a different duplicate namespace"
        )
    candidates = {str(row["source_id"]): row for row in package.candidates}
    if set(provenance.records) != set(candidates):
        raise JointDuplicateAuditError(
            "completed provenance must exactly cover joint-audit train candidates"
        )
    for sample_id, assignment in candidates.items():
        evidence = provenance.records[sample_id]
        expected = {
            "exact_duplicate_key": assignment["exact_duplicate_key"],
            "near_duplicate_cluster_id": assignment[
                "near_duplicate_cluster_id"
            ],
            "duplicate_cluster_namespace_id": package.namespace.namespace_id,
        }
        for field, expected_value in expected.items():
            if evidence.get(field) != expected_value:
                raise JointDuplicateAuditError(
                    f"provenance assignment mismatch for {sample_id} at {field}"
                )
    return provenance, dict(validation), sha256_file(provenance_manifest_path)


def _final_audit_payload(
    *,
    package: JointDuplicateAssignmentPackage,
    provenance: ProvenanceManifest,
    provenance_sha256: str,
    task_rows: Sequence[Mapping[str, Any]],
    recovery_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    corpus_binding = canonical_train_corpus_binding(
        records_sha256=provenance.records_sha256,
        provenance_manifest_sha256=provenance_sha256,
        duplicate_cluster_namespace=package.namespace,
    )
    assignment_by_task = {str(row["source_id"]): row for row in package.tasks}
    entries: list[dict[str, Any]] = []
    for task in task_rows:
        task_id = str(task["task_id"])
        assignment = assignment_by_task.get(task_id)
        if assignment is None:
            raise JointDuplicateAuditError(
                f"assignment package lacks WebArena task {task_id}"
            )
        cluster_ids = tuple(sorted(str(value) for value in assignment["cluster_ids"]))
        content_sha256 = duplicate_audit_task_content_sha256(task)
        if content_sha256 != assignment["content_sha256"]:
            raise JointDuplicateAuditError(
                f"task content changed after duplicate assignment: {task_id}"
            )
        audit_record = canonical_task_audit_record(
            task_id=task_id,
            content_sha256=content_sha256,
            train_corpus_manifest_sha256=corpus_binding[
                "corpus_binding_sha256"
            ],
            cluster_ids=cluster_ids,
            namespace=package.namespace,
        )
        entries.append(
            {
                "task_id": task_id,
                "task_partition": "normal",
                "status": "VERIFIED",
                "cluster_ids": list(cluster_ids),
                "content_sha256": content_sha256,
                "train_corpus_manifest_sha256": corpus_binding[
                    "corpus_binding_sha256"
                ],
                "audit_tool_id": package.namespace.audit_tool_id,
                "audit_tool_version": package.namespace.audit_tool_version,
                "evidence_sha256": canonical_sha256(audit_record),
                "cluster_namespace_id": package.namespace.namespace_id,
                "audit_tool_config_sha256": (
                    package.namespace.audit_tool_config_sha256
                ),
                "audit_tool_source_sha256": (
                    package.namespace.audit_tool_source_sha256
                ),
                "audit_record": audit_record,
            }
        )
    identity = canonical_sha256(
        {
            "assignment_manifest_sha256": package.assignment_manifest_sha256,
            "provenance_manifest_sha256": provenance_sha256,
            "task_ids": [row["task_id"] for row in entries],
            "recovery_scenario_ids": [row["task_id"] for row in recovery_entries],
        }
    )[:24]
    return {
        "schema_version": DUPLICATE_AUDIT_SCHEMA_VERSION,
        "manifest_id": f"table2-joint-duplicate-audit-{identity}",
        "manifest_state": "FROZEN_REGISTRATION",
        "evidence_label": "PILOT_ONLY",
        "paper_table_status": "N/R",
        "normal_task_evidence_status": "VERIFIED",
        "normal_task_runtime_policy": "VERIFIED_NONEMPTY_CLUSTERS_REQUIRED",
        "synthetic_recovery_evidence_scope": "DIAGNOSTIC_ONLY_NOT_PRIMARY",
        "required_normal_task_count": PILOT_TASK_COUNT,
        "required_recovery_scenario_count": RECOVERY_SCENARIO_COUNT,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "independent_recovery_or_final_success_evidence_created": 0,
        "joint_duplicate_assignment_binding": provenance_assignment_binding(package),
        "provenance_manifest_sha256": provenance_sha256,
        "train_corpus_binding": corpus_binding,
        "duplicate_cluster_namespace": package.namespace.to_dict(),
        "entries": [*entries, *(dict(row) for row in recovery_entries)],
    }


def finalize_joint_duplicate_audit(
    *,
    assignment_package_root: str | Path,
    config_path: str | Path,
    source_authority_path: str | Path,
    preparation_root: str | Path,
    gold_train_json: str | Path,
    resolved_task_export_path: str | Path,
    approved_task_registry_path: str | Path,
    recovery_scenarios_path: str | Path,
    duplicate_audit_registration_path: str | Path,
    provenance_manifest_path: str | Path,
    output_path: str | Path,
    supplement_train_json: str | Path | None = None,
) -> Path:
    """Emit VERIFIED duplicate evidence after, never instead of, external review."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite duplicate audit: {destination}")
    package = validate_joint_duplicate_assignment_package(
        package_root=assignment_package_root,
        config_path=config_path,
        source_authority_path=source_authority_path,
        preparation_root=preparation_root,
        gold_train_json=gold_train_json,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=approved_task_registry_path,
        recovery_scenarios_path=recovery_scenarios_path,
        duplicate_audit_registration_path=duplicate_audit_registration_path,
        supplement_train_json=supplement_train_json,
    )
    provenance, _, provenance_sha256 = _validated_completed_provenance(
        package=package,
        preparation_root=Path(preparation_root),
        provenance_manifest_path=Path(provenance_manifest_path),
    )
    task_export = _load_json(
        Path(resolved_task_export_path), role="resolved WebArena development-task export"
    )
    payload = _final_audit_payload(
        package=package,
        provenance=provenance,
        provenance_sha256=provenance_sha256,
        task_rows=task_export["tasks"],
        recovery_entries=_load_registered_recovery_diagnostics(
            recovery_scenarios_path=Path(recovery_scenarios_path),
            duplicate_audit_registration_path=Path(
                duplicate_audit_registration_path
            ),
        )[1],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical_pretty_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.chmod(0o644)
            temporary.unlink()
    validate_final_joint_duplicate_audit(
        audit_path=destination,
        assignment_package_root=assignment_package_root,
        config_path=config_path,
        source_authority_path=source_authority_path,
        preparation_root=preparation_root,
        gold_train_json=gold_train_json,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=approved_task_registry_path,
        recovery_scenarios_path=recovery_scenarios_path,
        duplicate_audit_registration_path=duplicate_audit_registration_path,
        provenance_manifest_path=provenance_manifest_path,
        supplement_train_json=supplement_train_json,
    )
    return destination


def validate_final_joint_duplicate_audit(
    *,
    audit_path: str | Path,
    assignment_package_root: str | Path,
    config_path: str | Path,
    source_authority_path: str | Path,
    preparation_root: str | Path,
    gold_train_json: str | Path,
    resolved_task_export_path: str | Path,
    approved_task_registry_path: str | Path,
    recovery_scenarios_path: str | Path,
    duplicate_audit_registration_path: str | Path,
    provenance_manifest_path: str | Path,
    supplement_train_json: str | Path | None = None,
) -> FrozenDuplicateAuditManifest:
    """Reproduce and validate final evidence from all permitted source bytes."""

    package = validate_joint_duplicate_assignment_package(
        package_root=assignment_package_root,
        config_path=config_path,
        source_authority_path=source_authority_path,
        preparation_root=preparation_root,
        gold_train_json=gold_train_json,
        resolved_task_export_path=resolved_task_export_path,
        approved_task_registry_path=approved_task_registry_path,
        recovery_scenarios_path=recovery_scenarios_path,
        duplicate_audit_registration_path=duplicate_audit_registration_path,
        supplement_train_json=supplement_train_json,
    )
    provenance, _, provenance_sha256 = _validated_completed_provenance(
        package=package,
        preparation_root=Path(preparation_root),
        provenance_manifest_path=Path(provenance_manifest_path),
    )
    task_export = _load_json(
        Path(resolved_task_export_path), role="resolved WebArena development-task export"
    )
    expected = _final_audit_payload(
        package=package,
        provenance=provenance,
        provenance_sha256=provenance_sha256,
        task_rows=task_export["tasks"],
        recovery_entries=_load_registered_recovery_diagnostics(
            recovery_scenarios_path=Path(recovery_scenarios_path),
            duplicate_audit_registration_path=Path(
                duplicate_audit_registration_path
            ),
        )[1],
    )
    actual = _load_json(Path(audit_path), role="final joint duplicate audit")
    if actual != expected:
        raise JointDuplicateAuditError(
            "final joint duplicate audit differs from exact source replay"
        )
    try:
        recovery_scenarios = _load_registered_recovery_diagnostics(
            recovery_scenarios_path=Path(recovery_scenarios_path),
            duplicate_audit_registration_path=Path(
                duplicate_audit_registration_path
            ),
        )[0]
        frozen = validate_pilot_duplicate_audit_manifest(
            audit_path,
            normal_task_ids=[str(row["task_id"]) for row in task_export["tasks"]],
            recovery_scenario_ids=[
                str(row["scenario_id"]) for row in recovery_scenarios
            ],
            require_normal_verified=True,
        )
    except DuplicateAuditError as error:
        raise JointDuplicateAuditError(
            f"final duplicate audit is not runtime-consumable: {error}"
        ) from error
    return frozen


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--preparation-package", type=Path, required=True)
    parser.add_argument("--gold-train-json", type=Path, required=True)
    parser.add_argument("--supplement-train-json", type=Path)
    parser.add_argument("--resolved-task-export", type=Path, required=True)
    parser.add_argument("--approved-task-registry", type=Path, required=True)
    parser.add_argument("--recovery-scenarios", type=Path, required=True)
    parser.add_argument("--duplicate-audit-registration", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-assignments")
    _common_arguments(build)
    build.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate-assignments")
    _common_arguments(validate)
    validate.add_argument("--assignment-package", type=Path, required=True)
    finalize = commands.add_parser("finalize-audit")
    _common_arguments(finalize)
    finalize.add_argument("--assignment-package", type=Path, required=True)
    finalize.add_argument("--provenance-manifest", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    validate_final = commands.add_parser("validate-final")
    _common_arguments(validate_final)
    validate_final.add_argument("--assignment-package", type=Path, required=True)
    validate_final.add_argument("--provenance-manifest", type=Path, required=True)
    validate_final.add_argument("--audit", type=Path, required=True)
    return parser.parse_args(argv)


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "config_path": args.config,
        "source_authority_path": args.source_authority,
        "preparation_root": args.preparation_package,
        "gold_train_json": args.gold_train_json,
        "supplement_train_json": args.supplement_train_json,
        "resolved_task_export_path": args.resolved_task_export,
        "approved_task_registry_path": args.approved_task_registry,
        "recovery_scenarios_path": args.recovery_scenarios,
        "duplicate_audit_registration_path": args.duplicate_audit_registration,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = _common_kwargs(args)
    if args.command == "build-assignments":
        result = build_joint_duplicate_assignment_package(
            **common, output_dir=args.output_dir
        )
        report = {
            "status": "FROZEN_ASSIGNMENTS",
            "assignment_manifest_sha256": result.assignment_manifest_sha256,
            "duplicate_cluster_namespace": result.namespace.to_dict(),
            "joint_duplicate_assignment_binding": provenance_assignment_binding(
                result
            ),
            "train_candidate_count": len(result.candidates),
            "development_task_count": len(result.tasks),
            "recovery_scenario_count": RECOVERY_SCENARIO_COUNT,
            "independent_recovery_or_final_success_evidence_created": 0,
        }
    elif args.command == "validate-assignments":
        result = validate_joint_duplicate_assignment_package(
            **common, package_root=args.assignment_package
        )
        report = {
            "status": "PASS",
            "assignment_manifest_sha256": result.assignment_manifest_sha256,
            "duplicate_cluster_namespace": result.namespace.to_dict(),
            "joint_duplicate_assignment_binding": provenance_assignment_binding(
                result
            ),
            "train_candidate_count": len(result.candidates),
            "development_task_count": len(result.tasks),
        }
    elif args.command == "finalize-audit":
        path = finalize_joint_duplicate_audit(
            **common,
            assignment_package_root=args.assignment_package,
            provenance_manifest_path=args.provenance_manifest,
            output_path=args.output,
        )
        report = {"status": "VERIFIED", "audit": str(path), "sha256": sha256_file(path)}
    else:
        result = validate_final_joint_duplicate_audit(
            **common,
            assignment_package_root=args.assignment_package,
            provenance_manifest_path=args.provenance_manifest,
            audit_path=args.audit,
        )
        report = {
            "status": "PASS",
            "audit_sha256": result.manifest_sha256,
            "verified_task_count": len(result.task_ids),
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
