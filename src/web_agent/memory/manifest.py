"""Integrity helpers for immutable Table 2 corrective-memory stores.

The store manifest is deliberately independent of the browser runtime.  A store
is accepted only after the manifest sidecar and every declared payload hash have
been verified.  This module uses only the Python standard library so a frozen
store can be audited without importing torch or loading a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    JointDuplicateClusterNamespace,
)


STORE_SCHEMA_VERSION = "table2-frozen-memory-v2"
PROVENANCE_SCHEMA_VERSION = "table2-memory-provenance-v1"
THRESHOLD_CALIBRATION_SCHEMA_VERSION = "table2-memory-threshold-calibration-v2"
THRESHOLD_CALIBRATION_METHOD = "observed_cosine_threshold_sweep_v1"
THRESHOLD_CALIBRATION_OBJECTIVE = "maximize_binary_f1"
THRESHOLD_CALIBRATION_TIE_BREAK = "highest_threshold"
THRESHOLD_CALIBRATION_CANDIDATES = "unique_observed_similarity_scores"
THRESHOLD_CALIBRATION_ADMISSION_RULE = "similarity_greater_than_or_equal"
THRESHOLD_CALIBRATION_ROWS_STORAGE = "embedded_canonical_json_sha256"
ELIGIBILITY_POLICY_VERSION = "table2-memory-eligibility-v1"
EMBEDDING_DIMENSION = 768
SIMILARITY_METRIC = "cosine"
RUNTIME_TOP_K = 3

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when memory provenance or store integrity is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable UTF-8 representation used for content hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationReplay:
    """Exact result of replaying the registered train-only threshold sweep."""

    admission_threshold: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    objective_numerator: int
    objective_denominator: int
    candidate_threshold_count: int

    def operating_point(self) -> dict[str, int | float]:
        return {
            "threshold": self.admission_threshold,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "objective_numerator": self.objective_numerator,
            "objective_denominator": self.objective_denominator,
            "candidate_threshold_count": self.candidate_threshold_count,
        }


def _replay_threshold_rows(
    rows: object,
) -> tuple[list[dict[str, Any]], ThresholdCalibrationReplay]:
    if not isinstance(rows, list) or not rows:
        raise ManifestError("calibration_rows must be a non-empty JSON array")

    normalized: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    required_row_keys = {
        "row_id",
        "source_split",
        "cosine_similarity",
        "relevant",
    }
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != required_row_keys:
            raise ManifestError(
                f"calibration_rows[{index}] must contain exactly "
                f"{sorted(required_row_keys)}"
            )
        row_id = raw_row.get("row_id")
        if not isinstance(row_id, str) or not row_id or row_id != row_id.strip():
            raise ManifestError(
                f"calibration_rows[{index}].row_id must be a non-empty canonical string"
            )
        if row_id in row_ids:
            raise ManifestError(f"duplicate calibration row_id: {row_id}")
        row_ids.add(row_id)
        if raw_row.get("source_split") != "train":
            raise ManifestError(
                f"calibration_rows[{index}] must declare source_split=train"
            )
        raw_similarity = raw_row.get("cosine_similarity")
        if isinstance(raw_similarity, bool) or not isinstance(
            raw_similarity, (int, float)
        ):
            raise ManifestError(
                f"calibration_rows[{index}].cosine_similarity must be numeric"
            )
        similarity = float(raw_similarity)
        if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
            raise ManifestError(
                f"calibration_rows[{index}].cosine_similarity must be finite "
                "and within [-1, 1]"
            )
        relevant = raw_row.get("relevant")
        if type(relevant) is not bool:
            raise ManifestError(
                f"calibration_rows[{index}].relevant must be a JSON boolean"
            )
        normalized.append({
            "row_id": row_id,
            "source_split": "train",
            "cosine_similarity": similarity,
            "relevant": relevant,
        })

    positive_count = sum(1 for row in normalized if row["relevant"])
    if positive_count == 0 or positive_count == len(normalized):
        raise ManifestError(
            "threshold calibration requires at least one relevant and one "
            "non-relevant train row"
        )

    thresholds = sorted(
        {float(row["cosine_similarity"]) for row in normalized},
        reverse=True,
    )
    best_replay: ThresholdCalibrationReplay | None = None
    best_objective: Fraction | None = None
    for threshold in thresholds:
        true_positive = sum(
            1
            for row in normalized
            if row["cosine_similarity"] >= threshold and row["relevant"]
        )
        false_positive = sum(
            1
            for row in normalized
            if row["cosine_similarity"] >= threshold and not row["relevant"]
        )
        false_negative = positive_count - true_positive
        true_negative = (
            len(normalized) - positive_count - false_positive
        )
        numerator = 2 * true_positive
        denominator = numerator + false_positive + false_negative
        objective = Fraction(numerator, denominator)
        candidate = ThresholdCalibrationReplay(
            admission_threshold=threshold,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
            objective_numerator=objective.numerator,
            objective_denominator=objective.denominator,
            candidate_threshold_count=len(thresholds),
        )
        # Candidates are visited from highest to lowest.  Keeping the first
        # exact rational tie implements the registered conservative tie-break.
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_replay = candidate

    if best_replay is None:  # pragma: no cover - non-empty rows guarantee this
        raise ManifestError("threshold calibration produced no candidate threshold")
    return normalized, best_replay


def build_threshold_calibration_payload(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_sha256: str,
    records_sha256: str,
    dataset_artifacts_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Create replayable evidence without accepting a caller-chosen threshold."""

    rows = [dict(row) for row in calibration_rows]
    normalized, replay = _replay_threshold_rows(rows)
    payload: dict[str, Any] = {
        "schema_version": THRESHOLD_CALIBRATION_SCHEMA_VERSION,
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "calibration_method": THRESHOLD_CALIBRATION_METHOD,
        "calibration_objective": THRESHOLD_CALIBRATION_OBJECTIVE,
        "calibration_tie_break": THRESHOLD_CALIBRATION_TIE_BREAK,
        "candidate_threshold_policy": THRESHOLD_CALIBRATION_CANDIDATES,
        "admission_rule": THRESHOLD_CALIBRATION_ADMISSION_RULE,
        "calibration_rows_storage": THRESHOLD_CALIBRATION_ROWS_STORAGE,
        "calibration_sample_count": len(normalized),
        "calibration_rows": normalized,
        "calibration_rows_sha256": canonical_sha256(normalized),
        "admission_threshold": replay.admission_threshold,
        "selected_operating_point": replay.operating_point(),
        "checkpoint_sha256": require_sha256(
            checkpoint_sha256, field="checkpoint_sha256"
        ),
        "records_sha256": require_sha256(records_sha256, field="records_sha256"),
        "dataset_artifacts_sha256": require_sha256(
            dataset_artifacts_sha256, field="dataset_artifacts_sha256"
        ),
        "protocol_sha256": require_sha256(protocol_sha256, field="protocol_sha256"),
    }
    validate_threshold_calibration_payload(
        payload,
        checkpoint_sha256=checkpoint_sha256,
        records_sha256=records_sha256,
        dataset_artifacts_sha256=dataset_artifacts_sha256,
        protocol_sha256=protocol_sha256,
    )
    return payload


def validate_threshold_calibration_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    records_sha256: str,
    dataset_artifacts_sha256: str,
    protocol_sha256: str,
) -> ThresholdCalibrationReplay:
    """Validate and exactly replay a train-only admission-threshold manifest."""
    required_keys = {
        "schema_version",
        "source_split",
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
        "calibration_method",
        "calibration_objective",
        "calibration_tie_break",
        "candidate_threshold_policy",
        "admission_rule",
        "calibration_rows_storage",
        "calibration_sample_count",
        "calibration_rows",
        "calibration_rows_sha256",
        "admission_threshold",
        "selected_operating_point",
        "checkpoint_sha256",
        "records_sha256",
        "dataset_artifacts_sha256",
        "protocol_sha256",
    }
    if set(payload) != required_keys:
        missing = sorted(required_keys - set(payload))
        extra = sorted(set(payload) - required_keys)
        raise ManifestError(
            "threshold calibration fields differ from the registered schema: "
            f"missing={missing}, extra={extra}"
        )
    if payload.get("schema_version") != THRESHOLD_CALIBRATION_SCHEMA_VERSION:
        raise ManifestError("unsupported memory threshold-calibration schema")
    if payload.get("source_split") != "train":
        raise ManifestError("memory threshold calibration must be train-only")
    for field in ("validation_rows_read", "test_rows_read", "locked_test_rows_read"):
        if type(payload.get(field)) is not int or payload[field] != 0:
            raise ManifestError(f"threshold calibration requires explicit {field}=0")
    registered_settings = {
        "calibration_method": THRESHOLD_CALIBRATION_METHOD,
        "calibration_objective": THRESHOLD_CALIBRATION_OBJECTIVE,
        "calibration_tie_break": THRESHOLD_CALIBRATION_TIE_BREAK,
        "candidate_threshold_policy": THRESHOLD_CALIBRATION_CANDIDATES,
        "admission_rule": THRESHOLD_CALIBRATION_ADMISSION_RULE,
        "calibration_rows_storage": THRESHOLD_CALIBRATION_ROWS_STORAGE,
    }
    for field, expected in registered_settings.items():
        if payload.get(field) != expected:
            raise ManifestError(
                f"threshold calibration {field} differs from registration"
            )
    normalized_rows, replay = _replay_threshold_rows(payload.get("calibration_rows"))
    if type(payload.get("calibration_sample_count")) is not int or int(
        payload["calibration_sample_count"]
    ) != len(normalized_rows):
        raise ManifestError(
            "threshold calibration_sample_count must exactly match embedded rows"
        )
    registered_rows_hash = require_sha256(
        payload.get("calibration_rows_sha256"),
        field="calibration.calibration_rows_sha256",
    )
    try:
        actual_rows_hash = canonical_sha256(payload["calibration_rows"])
    except (TypeError, ValueError) as error:
        raise ManifestError(
            "embedded calibration rows are not canonical finite JSON"
        ) from error
    if actual_rows_hash != registered_rows_hash:
        raise ManifestError("embedded calibration rows do not match their SHA-256")
    registered = payload.get("admission_threshold")
    if isinstance(registered, bool) or not isinstance(registered, (int, float)):
        raise ManifestError("threshold calibration lacks numeric admission_threshold")
    if not math.isfinite(float(registered)) or not -1.0 <= float(registered) <= 1.0:
        raise ManifestError("admission_threshold must be finite and within [-1, 1]")
    if float(registered) != replay.admission_threshold:
        raise ManifestError(
            "admission_threshold does not exactly match deterministic calibration replay"
        )
    try:
        operating_point_matches = canonical_json_bytes(
            payload.get("selected_operating_point")
        ) == canonical_json_bytes(replay.operating_point())
    except (TypeError, ValueError) as error:
        raise ManifestError(
            "selected_operating_point is not canonical finite JSON"
        ) from error
    if not operating_point_matches:
        raise ManifestError(
            "selected_operating_point does not match deterministic calibration replay"
        )
    expected_hashes = {
        "checkpoint_sha256": require_sha256(
            checkpoint_sha256, field="expected.checkpoint_sha256"
        ),
        "records_sha256": require_sha256(
            records_sha256, field="expected.records_sha256"
        ),
        "dataset_artifacts_sha256": require_sha256(
            dataset_artifacts_sha256,
            field="expected.dataset_artifacts_sha256",
        ),
        "protocol_sha256": require_sha256(
            protocol_sha256, field="expected.protocol_sha256"
        ),
    }
    for field, expected in expected_hashes.items():
        actual = require_sha256(payload.get(field), field=f"calibration.{field}")
        if actual != expected:
            raise ManifestError(f"threshold calibration {field} differs from build input")
    return replay


def file_descriptor(path: str | Path) -> dict[str, int | str]:
    candidate = Path(path)
    if not candidate.is_file():
        raise ManifestError(f"missing memory-store payload: {candidate}")
    return {
        "sha256": sha256_file(candidate),
        "bytes": candidate.stat().st_size,
    }


@dataclass(frozen=True)
class VerifiedStoreManifest:
    """A verified manifest and the digest binding it to its store."""

    payload: Mapping[str, Any]
    sha256: str


def verify_store_manifest(root: str | Path) -> VerifiedStoreManifest:
    """Verify a frozen store manifest and every file it declares."""
    directory = Path(root)
    manifest_path = directory / "manifest.json"
    digest_path = directory / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise ManifestError(
            f"frozen store requires manifest.json and manifest.sha256: {directory}"
        )

    actual_manifest_sha = sha256_file(manifest_path)
    expected_manifest_sha = require_sha256(
        digest_path.read_text(encoding="utf-8").strip(),
        field="manifest.sha256",
    )
    if actual_manifest_sha != expected_manifest_sha:
        raise ManifestError(
            "memory manifest hash mismatch: "
            f"expected {expected_manifest_sha}, found {actual_manifest_sha}"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid memory manifest JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("memory manifest must be a JSON object")
    if payload.get("schema_version") != STORE_SCHEMA_VERSION:
        raise ManifestError(
            "unsupported memory manifest schema: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("source_split") != "train":
        raise ManifestError("frozen memory source_split must be exactly 'train'")
    if payload.get("embedding_stage") != "post_action_memory_task_adapter":
        raise ManifestError(
            "memory embedding_stage must be post_action_memory_task_adapter"
        )
    if int(payload.get("embedding_dimension", -1)) != EMBEDDING_DIMENSION:
        raise ManifestError(
            f"memory embedding dimension must be {EMBEDDING_DIMENSION}"
        )
    if payload.get("normalization") != "l2":
        raise ManifestError("frozen memory normalization must be l2")
    if payload.get("similarity") != SIMILARITY_METRIC:
        raise ManifestError("frozen memory similarity metric must be cosine")
    if int(payload.get("top_k", -1)) != RUNTIME_TOP_K:
        raise ManifestError(f"runtime top-k must be exactly {RUNTIME_TOP_K}")
    if payload.get("tie_break") != "memory_id_ascending":
        raise ManifestError("frozen memory tie_break must be memory_id_ascending")
    if payload.get("admission_threshold_source") != "train_only_calibration":
        raise ManifestError(
            "memory admission threshold must come from train_only_calibration"
        )
    raw_threshold = payload.get("admission_threshold")
    if isinstance(raw_threshold, bool):
        raise ManifestError("admission_threshold must be a number, not boolean")
    try:
        admission_threshold = float(raw_threshold)
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("admission_threshold must be a number") from error
    if not math.isfinite(admission_threshold) or not -1.0 <= admission_threshold <= 1.0:
        raise ManifestError("admission_threshold must be finite and within [-1, 1]")
    if payload.get("same_task_exclusion") is not True:
        raise ManifestError("same-task memory exclusion must be enabled")
    if payload.get("duplicate_exclusion") is not True:
        raise ManifestError("duplicate memory exclusion must be enabled")
    try:
        JointDuplicateClusterNamespace.from_mapping(
            payload.get("duplicate_cluster_namespace"),
            require_hashes=True,
        )
    except DuplicateAuditError as error:
        raise ManifestError(
            f"invalid frozen-memory duplicate-cluster namespace: {error}"
        ) from error
    if payload.get("runtime_writes_allowed") is not False:
        raise ManifestError("frozen memory must declare runtime_writes_allowed=false")
    for field in (
        "validation_rows_read",
        "test_rows_read",
        "locked_test_rows_read",
    ):
        if type(payload.get(field)) is not int or payload[field] != 0:
            raise ManifestError(f"frozen memory requires explicit {field}=0")
    for hash_field in (
        "checkpoint_sha256",
        "dataset_artifacts_sha256",
        "records_sha256",
        "resolved_config_sha256",
        "resolved_config_record_sha256",
        "protocol_sha256",
        "provenance_manifest_sha256",
        "threshold_calibration_sha256",
        "calibration_evidence_sha256",
        "verification_evidence_sha256",
        "verification_records_sha256",
        "transition_report_sha256",
    ):
        require_sha256(payload.get(hash_field), field=hash_field)
    verification_evidence = payload.get("verification_evidence")
    if not isinstance(verification_evidence, Mapping):
        raise ManifestError("frozen memory lacks verification_evidence closure")
    from web_agent.memory.verification import (
        VERIFICATION_DIGEST_ALGORITHM,
        VERIFICATION_MANIFEST_SCHEMA_VERSION,
    )

    expected_verification_fields = {
        "schema_version",
        "digest_algorithm",
        "item_count",
        "items",
        "items_sha256",
    }
    if set(verification_evidence) != expected_verification_fields:
        raise ManifestError(
            "verification_evidence fields differ from the registered schema"
        )
    if verification_evidence.get("schema_version") != (
        VERIFICATION_MANIFEST_SCHEMA_VERSION
    ):
        raise ManifestError("unsupported verification_evidence schema")
    if verification_evidence.get("digest_algorithm") != (
        VERIFICATION_DIGEST_ALGORITHM
    ):
        raise ManifestError("unsupported verification_evidence digest algorithm")
    verification_items = verification_evidence.get("items")
    if not isinstance(verification_items, list):
        raise ManifestError("verification_evidence.items must be an array")
    if (
        type(verification_evidence.get("item_count")) is not int
        or verification_evidence["item_count"] != len(verification_items)
        or verification_evidence["item_count"] != payload.get("item_count")
    ):
        raise ManifestError("verification_evidence item count is inconsistent")
    verification_row_fields = {
        "memory_id",
        "source_sample_id",
        "recovery_verification_evidence_sha256",
        "final_task_verification_evidence_sha256",
        "verification_evidence_sha256",
    }
    prior_memory_id = ""
    for index, row in enumerate(verification_items):
        if not isinstance(row, Mapping) or set(row) != verification_row_fields:
            raise ManifestError(
                f"verification_evidence.items[{index}] fields are invalid"
            )
        memory_id = str(row.get("memory_id") or "")
        if not memory_id or memory_id <= prior_memory_id:
            raise ManifestError(
                "verification_evidence items must have unique ascending memory IDs"
            )
        prior_memory_id = memory_id
        if not str(row.get("source_sample_id") or "").strip():
            raise ManifestError(
                f"verification_evidence.items[{index}] lacks source_sample_id"
            )
        for field in (
            "recovery_verification_evidence_sha256",
            "final_task_verification_evidence_sha256",
            "verification_evidence_sha256",
        ):
            require_sha256(row.get(field), field=f"verification_evidence.{field}")
    if verification_evidence.get("items_sha256") != canonical_sha256(
        verification_items
    ):
        raise ManifestError("verification_evidence item closure hash mismatch")
    if payload.get("verification_evidence_sha256") != canonical_sha256(
        verification_evidence
    ):
        raise ManifestError("verification_evidence manifest hash mismatch")
    verification_records_path = directory / "verification_evidence.json"
    try:
        verification_records = json.loads(
            verification_records_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(
            f"invalid stored independent verification records: {error}"
        ) from error
    if not isinstance(verification_records, dict):
        raise ManifestError("stored independent verification records must be an object")
    if canonical_sha256(verification_records) != payload[
        "verification_records_sha256"
    ]:
        raise ManifestError("stored independent verification-record hash mismatch")
    from web_agent.memory.verification import (
        VerificationEvidenceError,
        validate_verification_records_payload,
    )

    try:
        verification_expected_count = payload.get("item_count")
        validate_verification_records_payload(
            verification_records,
            expected_item_count=(
                verification_expected_count
                if type(verification_expected_count) is int
                else -1
            ),
        )
    except VerificationEvidenceError as error:
        raise ManifestError(
            f"stored independent verification records are invalid: {error}"
        ) from error
    calibration_path = directory / "threshold_calibration.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"invalid stored threshold calibration: {error}") from error
    if not isinstance(calibration, dict):
        raise ManifestError("stored threshold calibration must be an object")
    if canonical_sha256(calibration) != payload["threshold_calibration_sha256"]:
        raise ManifestError("stored threshold calibration hash mismatch")
    calibration_replay = validate_threshold_calibration_payload(
        calibration,
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        records_sha256=str(payload["records_sha256"]),
        dataset_artifacts_sha256=str(payload["dataset_artifacts_sha256"]),
        protocol_sha256=str(payload["protocol_sha256"]),
    )
    if admission_threshold != calibration_replay.admission_threshold:
        raise ManifestError(
            "stored admission_threshold does not exactly match calibration replay"
        )
    calibration_evidence_path = directory / "calibration_evidence.json"
    try:
        calibration_evidence = json.loads(
            calibration_evidence_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"invalid stored calibration evidence: {error}") from error
    if not isinstance(calibration_evidence, dict):
        raise ManifestError("stored calibration evidence must be an object")
    if (
        canonical_sha256(calibration_evidence)
        != payload["calibration_evidence_sha256"]
    ):
        raise ManifestError("stored calibration evidence hash mismatch")
    # Local import avoids a module cycle: the evidence builder itself reuses
    # this module's compact threshold replay.
    from web_agent.memory.calibration_builder import (
        CALIBRATION_EMBEDDING_HASH,
        CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        CALIBRATION_PAIR_POLICY,
        CALIBRATION_RELEVANCE_DEFINITION,
    )

    expected_evidence = {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        "source_split": "train",
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "relevance_definition": CALIBRATION_RELEVANCE_DEFINITION,
        "pair_policy": CALIBRATION_PAIR_POLICY,
        "embedding_stage": "post_action_memory_task_adapter",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalization": "l2",
        "similarity": SIMILARITY_METRIC,
        "embedding_hash_algorithm": CALIBRATION_EMBEDDING_HASH,
        "model_seed": payload.get("model_seed"),
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "records_sha256": payload.get("records_sha256"),
        "dataset_artifacts_sha256": payload.get("dataset_artifacts_sha256"),
        "resolved_config_sha256": payload.get("resolved_config_sha256"),
        "resolved_config_record_sha256": payload.get(
            "resolved_config_record_sha256"
        ),
        "protocol_sha256": payload.get("protocol_sha256"),
        "provenance_manifest_sha256": payload.get("provenance_manifest_sha256"),
        "duplicate_cluster_namespace_id": payload[
            "duplicate_cluster_namespace"
        ]["namespace_id"],
        "retrieval_top_k": RUNTIME_TOP_K,
        "retrieval_tie_break": "memory_id_ascending",
    }
    for field, expected in expected_evidence.items():
        if calibration_evidence.get(field) != expected:
            raise ManifestError(
                f"stored calibration evidence differs from memory at {field}"
            )
    embedded_threshold = calibration_evidence.get("threshold_calibration")
    if not isinstance(embedded_threshold, Mapping) or canonical_sha256(
        embedded_threshold
    ) != payload["threshold_calibration_sha256"]:
        raise ManifestError(
            "stored calibration evidence does not embed the threshold replay"
        )
    if calibration_evidence.get(
        "threshold_calibration_sha256"
    ) != payload["threshold_calibration_sha256"]:
        raise ManifestError("stored calibration evidence threshold hash mismatch")
    evidence_rows = calibration_evidence.get("calibration_rows")
    selection_rows = calibration_evidence.get("eligible_selection")
    if not isinstance(evidence_rows, list) or not isinstance(selection_rows, list):
        raise ManifestError("stored calibration evidence lacks row-level bindings")
    if calibration_evidence.get("calibration_rows_sha256") != canonical_sha256(
        evidence_rows
    ):
        raise ManifestError("stored calibration evidence row hash mismatch")
    if calibration_evidence.get("eligible_selection_sha256") != canonical_sha256(
        selection_rows
    ):
        raise ManifestError("stored calibration eligible-selection hash mismatch")
    if (
        calibration_evidence.get("eligible_item_count") != payload.get("item_count")
        or calibration_evidence.get("query_count") != payload.get("item_count")
        or calibration_evidence.get("calibration_sample_count")
        != len(evidence_rows)
        or len(evidence_rows) > RUNTIME_TOP_K * int(payload.get("item_count", -1))
    ):
        raise ManifestError("stored calibration evidence counts are inconsistent")
    if payload.get("eligibility_policy_version") != ELIGIBILITY_POLICY_VERSION:
        raise ManifestError("unexpected memory eligibility policy version")
    for text_field in ("store_id", "dataset_id", "dataset_version"):
        if not str(payload.get(text_field) or "").strip():
            raise ManifestError(f"memory manifest is missing {text_field}")

    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != {
        "calibration_evidence.json",
        "embeddings.npy",
        "items.jsonl",
        "threshold_calibration.json",
        "verification_evidence.json",
    }:
        raise ManifestError(
            "memory manifest files must contain embeddings, items, calibration, "
            "and independent verification evidence"
        )
    for relative_name, expected in files.items():
        if not isinstance(expected, dict):
            raise ManifestError(f"invalid file descriptor for {relative_name}")
        expected_hash = require_sha256(
            expected.get("sha256"), field=f"files.{relative_name}.sha256"
        )
        try:
            expected_size = int(expected["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError(
                f"files.{relative_name}.bytes must be an integer"
            ) from error
        target = directory / relative_name
        descriptor = file_descriptor(target)
        if descriptor["sha256"] != expected_hash:
            raise ManifestError(f"memory payload hash mismatch: {relative_name}")
        if descriptor["bytes"] != expected_size:
            raise ManifestError(f"memory payload size mismatch: {relative_name}")

    raw_item_count = payload.get("item_count")
    raw_seed = payload.get("model_seed")
    if type(raw_item_count) is not int or type(raw_seed) is not int:
        raise ManifestError("item_count and model_seed must be JSON integers")
    try:
        item_count = int(raw_item_count)
        seed = int(raw_seed)
    except (TypeError, ValueError) as error:
        raise ManifestError("item_count and model_seed must be integers") from error
    if item_count <= 0:
        raise ManifestError("a frozen memory store cannot be empty")
    if seed < 0:
        raise ManifestError("model_seed must be non-negative")

    return VerifiedStoreManifest(payload=payload, sha256=actual_manifest_sha)
