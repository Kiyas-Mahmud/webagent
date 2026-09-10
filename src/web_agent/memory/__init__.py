"""Pillar 4: leakage-controlled corrective memory for Table 2."""

from web_agent.memory.builder import MemoryBuildError, build_frozen_store
from web_agent.memory.calibration_builder import (
    CalibrationEvidenceError,
    VerifiedCalibrationEvidence,
    build_calibration_evidence,
    validate_calibration_evidence,
)
from web_agent.memory.eligibility import (
    EligibilitySelection,
    EligibleMemoryCandidate,
    MemoryEligibilityError,
    ProvenanceManifest,
    select_eligible_candidates,
)
from web_agent.memory.frozen_store import (
    FrozenQueryResult,
    FrozenMemoryStore,
    MemoryHit,
    QueryExclusions,
)
from web_agent.memory.manifest import (
    ManifestError,
    ThresholdCalibrationReplay,
    build_threshold_calibration_payload,
)
from web_agent.memory.verification import (
    FINAL_TASK_VERIFICATION_SCHEMA_VERSION,
    RECOVERY_VERIFICATION_SCHEMA_VERSION,
    VERIFICATION_BUNDLE_SCHEMA_VERSION,
    VERIFICATION_MANIFEST_SCHEMA_VERSION,
    VERIFICATION_RECORDS_SCHEMA_VERSION,
    VerificationEvidenceError,
    recovery_action_evidence_sha256,
    recovery_state_evidence_sha256,
    validate_provenance_verification_evidence,
    verification_bundle_sha256,
    verification_records_payload,
    validate_verification_records_payload,
)

__all__ = [
    "EligibilitySelection",
    "EligibleMemoryCandidate",
    "FrozenMemoryStore",
    "FrozenQueryResult",
    "ManifestError",
    "CalibrationEvidenceError",
    "MemoryBuildError",
    "MemoryEligibilityError",
    "MemoryHit",
    "ProvenanceManifest",
    "QueryExclusions",
    "ThresholdCalibrationReplay",
    "VerifiedCalibrationEvidence",
    "VerificationEvidenceError",
    "FINAL_TASK_VERIFICATION_SCHEMA_VERSION",
    "RECOVERY_VERIFICATION_SCHEMA_VERSION",
    "VERIFICATION_BUNDLE_SCHEMA_VERSION",
    "VERIFICATION_MANIFEST_SCHEMA_VERSION",
    "VERIFICATION_RECORDS_SCHEMA_VERSION",
    "build_calibration_evidence",
    "build_frozen_store",
    "build_threshold_calibration_payload",
    "recovery_action_evidence_sha256",
    "recovery_state_evidence_sha256",
    "select_eligible_candidates",
    "validate_calibration_evidence",
    "validate_provenance_verification_evidence",
    "validate_verification_records_payload",
    "verification_bundle_sha256",
    "verification_records_payload",
]
