"""Offline Pillar-2 modality sensitivity and causal-input controls.

This module is deliberately outside the live episode runner.  It replays a
frozen, already-executed diagnostic corpus through a read-only predictor and
never returns an action, recovery trigger, or memory intervention to runtime.

The registered controls are:

* full input versus a registered neutral image;
* full input versus fixed neutral text/state;
* post-action assessment with the executed action removed;
* post-action assessment with a deterministic donor post-state; and
* post-action assessment with a deterministic different donor action.

Inference-time modality occlusion is sensitivity evidence, not a controlled
training ablation.  The post-action controls test input conditioning; they do
not by themselves prove causal reasoning or operational recovery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import inspect
import math
from pathlib import Path
import stat
import subprocess
from typing import Any, Protocol
from urllib.parse import urlparse

from web_agent.labels import (
    ACTION_TYPE,
    FAILURE_TYPE,
    RECOVERY_STRATEGY,
)
from web_agent.runtime.observation import (
    CausalBoundaryError,
    assert_oracle_blind_mapping,
)

from .common import (
    SchemaError,
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    sha256_json,
    stable_int_seed,
)
from .statistics import cluster_bootstrap


PILLAR2_INPUT_SCHEMA_VERSION = "table2.pillar2-diagnostic-input.v1"
PILLAR2_REPORT_SCHEMA_VERSION = "table2.pillar2-diagnostic-report.v1"
PILLAR2_PACKAGE_SCHEMA_VERSION = "table2.pillar2-diagnostic-package.v1"
PILLAR2_BACKEND_SCHEMA_VERSION = "table2.pillar2-diagnostic-backend.v1"
PILLAR2_EVIDENCE_ROLE = "P2_COMPANION_MECHANISM_EVIDENCE"
ENGINEERING_EVIDENCE_ROLE = "ENGINEERING_DIAGNOSTIC_ONLY_UNPROMOTABLE"
CANONICAL_INPUT_PROVENANCE_STATUS = (
    "BLOCKED_AUTHORITATIVE_INPUT_PROVENANCE_REQUIRED"
)
INPUT_PROVENANCE_BLOCKED_PROMOTION = "UNPROMOTABLE_INPUT_PROVENANCE_BLOCKED"
PAPER_TABLE_STATUS = "N/R"
NEUTRAL_TEXT_ID = "fixed-neutral-text-state-v1"
PILLAR2_BOOTSTRAP_CONFIDENCE = 0.95
PILLAR2_BOOTSTRAP_SAMPLES = 10_000
PILLAR2_BOOTSTRAP_SEED = 20250831
PILLAR2_INTERVAL_METHOD = "percentile_task_cluster_bootstrap"

PILLAR2_CLAIM_SCOPE = {
    "modality": "INFERENCE_TIME_SENSITIVITY_NOT_TRAINING_ABLATION",
    "causal_controls": "INPUT_CONDITIONING_DIAGNOSTIC_NOT_CAUSAL_PROOF",
    "operational_recovery": "NOT_MEASURED",
    "table2_row_or_contrast": "NONE",
}
PILLAR2_CONDITION_DEFINITIONS = {
    "image_occluded": (
        "replace every phase-visible image with a registered same-size neutral image"
    ),
    "text_occluded": (
        "replace task/domain/URL/title/page-state with the fixed registered neutral "
        "text/state"
    ),
    "executed_action_removed": (
        "omit executed-action identity from post-action inference only"
    ),
    "post_state_swapped": (
        "replace only the post-state image with a deterministic same-size donor"
    ),
    "action_post_mismatch": (
        "replace only the executed action with a deterministic different donor action"
    ),
}

_PRE_SUMMARY_METRICS = {
    "action_accuracy": "rate",
    "bbox_iou_mean": "mean",
    "top_action_change_rate_vs_full": "rate",
    "mean_action_total_variation_vs_full": "mean",
    "mean_bbox_l1_vs_full": "mean",
    "mean_confidence_absolute_delta_vs_full": "mean",
}
_POST_SUMMARY_METRICS = {
    "outcome_accuracy": "rate",
    "failure_type_accuracy": "rate",
    "needs_recovery_accuracy": "rate",
    "recovery_strategy_accuracy": "rate",
    "failure_decision_change_rate_vs_full": "rate",
    "mean_failure_probability_delta_vs_full": "mean",
    "failure_type_change_rate_vs_full": "rate",
    "mean_failure_type_total_variation_vs_full": "mean",
    "needs_recovery_change_rate_vs_full": "rate",
    "mean_needs_recovery_probability_delta_vs_full": "mean",
    "recovery_strategy_change_rate_vs_full": "rate",
    "mean_recovery_total_variation_vs_full": "mean",
}

ALLOWED_SOURCE_PARTITIONS = frozenset(
    {
        "train_diagnostic",
        "validation_only",
        "public_development",
        "completed_public_pilot_campaign",
    }
)
ACTION_LABELS = tuple(ACTION_TYPE)
FAILURE_LABELS = tuple(FAILURE_TYPE)
RECOVERY_LABELS = tuple(RECOVERY_STRATEGY)


class Pillar2DiagnosticError(RuntimeError):
    """The offline diagnostic boundary or evidence package is invalid."""


class DiagnosticCondition(str, Enum):
    FULL = "full"
    IMAGE_OCCLUDED = "image_occluded"
    TEXT_OCCLUDED = "text_occluded"
    EXECUTED_ACTION_REMOVED = "executed_action_removed"
    POST_STATE_SWAPPED = "post_state_swapped"
    ACTION_POST_MISMATCH = "action_post_mismatch"


PRE_CONDITIONS = (
    DiagnosticCondition.FULL,
    DiagnosticCondition.IMAGE_OCCLUDED,
    DiagnosticCondition.TEXT_OCCLUDED,
)
POST_CONDITIONS = (
    DiagnosticCondition.FULL,
    DiagnosticCondition.IMAGE_OCCLUDED,
    DiagnosticCondition.TEXT_OCCLUDED,
    DiagnosticCondition.EXECUTED_ACTION_REMOVED,
    DiagnosticCondition.POST_STATE_SWAPPED,
    DiagnosticCondition.ACTION_POST_MISMATCH,
)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise SchemaError(
            f"{context} fields mismatch: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _require_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context} must be non-empty text")
    return value


def _require_sha256(value: object, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaError(f"{context} must be lowercase SHA-256")
    return text


def _require_commit(value: object) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaError("repository_commit must be a full lowercase Git commit")
    return text


def _require_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{context} must be an exact boolean")
    return value


def _require_probability(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SchemaError(f"{context} must be finite and in [0, 1]")
    return result


def _probability_distribution(
    value: Mapping[str, Any],
    *,
    labels: Sequence[str],
    context: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(labels):
        raise SchemaError(f"{context} must contain exactly {list(labels)}")
    result = {
        label: _require_probability(value[label], context=f"{context}.{label}")
        for label in labels
    }
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise SchemaError(f"{context} must sum to one")
    return result


def _bbox(value: object, *, context: str, nullable: bool) -> tuple[float, float, float, float] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SchemaError(f"{context} must be [x, y, width, height]")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SchemaError(f"{context} must be finite")
    x, y, width, height = result
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise SchemaError(f"{context} is outside the normalized image plane")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DiagnosticBackendIdentity:
    predictor_id: str
    predictor_version: str
    predictor_module: str
    predictor_qualname: str
    factory_entrypoint: str
    model_id: str
    model_revision: str
    checkpoint_sha256: str
    resolved_config_sha256: str
    resolved_config_record_sha256: str
    processor_contract_sha256: str
    factory_source_sha256: str
    predictor_source_sha256: str
    repository_commit: str
    model_seed: int
    selection_scope: str = "validation_only"
    frozen: bool = True
    evaluation_mode: bool = True
    schema_version: str = PILLAR2_BACKEND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "predictor_id",
            "predictor_version",
            "predictor_module",
            "predictor_qualname",
            "factory_entrypoint",
            "model_id",
            "model_revision",
        ):
            _require_text(getattr(self, name), context=f"backend.{name}")
        if ":" in self.predictor_module or ":" in self.predictor_qualname:
            raise SchemaError("backend predictor class identity is malformed")
        module_name, separator, attribute_name = self.factory_entrypoint.partition(":")
        if (
            separator != ":"
            or not module_name
            or not attribute_name
            or "." in attribute_name
        ):
            raise SchemaError("backend.factory_entrypoint must be exact module:attribute")
        for name in (
            "checkpoint_sha256",
            "resolved_config_sha256",
            "resolved_config_record_sha256",
            "processor_contract_sha256",
            "factory_source_sha256",
            "predictor_source_sha256",
        ):
            _require_sha256(getattr(self, name), context=f"backend.{name}")
        _require_commit(self.repository_commit)
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise SchemaError("backend.model_seed must be a nonnegative integer")
        if self.selection_scope != "validation_only":
            raise SchemaError("Pillar-2 diagnostics require validation-only model selection")
        if self.frozen is not True or self.evaluation_mode is not True:
            raise SchemaError("Pillar-2 diagnostic backend must be frozen in evaluation mode")
        if self.schema_version != PILLAR2_BACKEND_SCHEMA_VERSION:
            raise SchemaError("unregistered Pillar-2 backend schema")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiagnosticBackendIdentity":
        _require_exact_keys(
            value,
            {
                "predictor_id",
                "predictor_version",
                "predictor_module",
                "predictor_qualname",
                "factory_entrypoint",
                "model_id",
                "model_revision",
                "checkpoint_sha256",
                "resolved_config_sha256",
                "resolved_config_record_sha256",
                "processor_contract_sha256",
                "factory_source_sha256",
                "predictor_source_sha256",
                "repository_commit",
                "model_seed",
                "selection_scope",
                "frozen",
                "evaluation_mode",
                "schema_version",
            },
            context="backend identity",
        )
        if type(value["model_seed"]) is not int:
            raise SchemaError("backend.model_seed must be an exact integer")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ImageArtifact:
    artifact_id: str
    relative_path: str
    sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, context="image.artifact_id")
        _require_text(self.relative_path, context="image.relative_path")
        safe_relative_path(self.relative_path)
        _require_sha256(self.sha256, context="image.sha256")
        if type(self.width) is not int or type(self.height) is not int:
            raise SchemaError("image dimensions must be exact integers")
        if self.width <= 0 or self.height <= 0:
            raise SchemaError("image dimensions must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImageArtifact":
        _require_exact_keys(
            value,
            {"artifact_id", "relative_path", "sha256", "width", "height"},
            context="image artifact",
        )
        if type(value["width"]) is not int or type(value["height"]) is not int:
            raise SchemaError("image dimensions must be exact integers")
        return cls(**dict(value))

    def resolve(self, root: Path) -> "ResolvedImage":
        unresolved = root / safe_relative_path(self.relative_path)
        try:
            metadata = unresolved.lstat()
        except OSError as exc:
            raise Pillar2DiagnosticError(
                f"diagnostic image is absent: {self.artifact_id}"
            ) from exc
        if unresolved.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise Pillar2DiagnosticError(
                f"diagnostic image must be a non-symlink regular file: {self.artifact_id}"
            )
        destination = unresolved.resolve()
        if root.resolve() not in destination.parents:
            raise Pillar2DiagnosticError("diagnostic image escaped its registered root")
        if sha256_file(destination) != self.sha256:
            raise Pillar2DiagnosticError(
                f"diagnostic image hash mismatch: {self.artifact_id}"
            )
        return ResolvedImage(
            artifact_id=self.artifact_id,
            path=destination,
            sha256=self.sha256,
            width=self.width,
            height=self.height,
        )

    def public_identity(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    """Exact image input. Paths are supplied to the predictor but never exported."""

    artifact_id: str
    path: Path
    sha256: str
    width: int
    height: int

    def public_identity(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticTextState:
    task_text: str
    domain: str
    current_url: str
    title: str
    page_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("task_text", "domain", "current_url"):
            _require_text(getattr(self, name), context=f"text_state.{name}")
        if not isinstance(self.title, str) or not isinstance(self.page_state, Mapping):
            raise SchemaError("text_state title/page_state have invalid types")
        hostname = urlparse(self.current_url).hostname or ""
        if hostname != self.domain:
            raise SchemaError("text_state domain must equal the current URL hostname")
        try:
            assert_oracle_blind_mapping(asdict(self), location="P2 diagnostic text/state")
            canonical_json_bytes(asdict(self))
        except (CausalBoundaryError, TypeError, ValueError) as exc:
            raise SchemaError(f"P2 diagnostic text/state is not oracle-blind: {exc}") from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiagnosticTextState":
        _require_exact_keys(
            value,
            {"task_text", "domain", "current_url", "title", "page_state"},
            context="diagnostic text/state",
        )
        return cls(**dict(value))

    @classmethod
    def neutral(cls, text: str) -> "DiagnosticTextState":
        _require_text(text, context="neutral_text")
        return cls(
            task_text=text,
            domain="neutral.invalid",
            current_url="https://neutral.invalid/",
            title=text,
            page_state={"neutral_text": text},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiagnosticTargets:
    """Offline labels withheld from every predictor callback."""

    action_type: str
    action_bbox: tuple[float, float, float, float] | None
    failure: bool
    failure_type: str
    needs_recovery: bool
    recovery_strategy: str

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPE:
            raise SchemaError("target action_type is not registered")
        if self.failure_type not in FAILURE_TYPE:
            raise SchemaError("target failure_type is not registered")
        if self.recovery_strategy not in RECOVERY_STRATEGY:
            raise SchemaError("target recovery_strategy is not registered")
        _require_bool(self.failure, context="target.failure")
        _require_bool(self.needs_recovery, context="target.needs_recovery")
        if self.action_bbox is not None:
            _bbox(self.action_bbox, context="target.action_bbox", nullable=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiagnosticTargets":
        _require_exact_keys(
            value,
            {
                "action_type",
                "action_bbox",
                "failure",
                "failure_type",
                "needs_recovery",
                "recovery_strategy",
            },
            context="diagnostic targets",
        )
        return cls(
            action_type=str(value["action_type"]),
            action_bbox=_bbox(value["action_bbox"], context="target.action_bbox", nullable=True),
            failure=_require_bool(value["failure"], context="target.failure"),
            failure_type=str(value["failure_type"]),
            needs_recovery=_require_bool(
                value["needs_recovery"], context="target.needs_recovery"
            ),
            recovery_strategy=str(value["recovery_strategy"]),
        )


@dataclass(frozen=True, slots=True)
class DiagnosticExample:
    example_id: str
    task_id: str
    pre_image: ImageArtifact
    post_image: ImageArtifact
    text_state: DiagnosticTextState
    executed_action: str
    targets: DiagnosticTargets
    source_record_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.example_id, context="example_id")
        _require_text(self.task_id, context="task_id")
        if self.executed_action not in ACTION_TYPE:
            raise SchemaError("executed_action is not one of the six registered actions")
        _require_sha256(self.source_record_sha256, context="source_record_sha256")
        if self.pre_image.artifact_id == self.post_image.artifact_id:
            raise SchemaError("diagnostic transition requires distinct pre/post images")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiagnosticExample":
        _require_exact_keys(
            value,
            {
                "example_id",
                "task_id",
                "pre_image",
                "post_image",
                "text_state",
                "executed_action",
                "targets",
                "source_record_sha256",
            },
            context="diagnostic example",
        )
        for field in ("pre_image", "post_image", "text_state", "targets"):
            if not isinstance(value[field], Mapping):
                raise SchemaError(f"diagnostic example {field} must be an object")
        return cls(
            example_id=str(value["example_id"]),
            task_id=str(value["task_id"]),
            pre_image=ImageArtifact.from_mapping(value["pre_image"]),
            post_image=ImageArtifact.from_mapping(value["post_image"]),
            text_state=DiagnosticTextState.from_mapping(value["text_state"]),
            executed_action=str(value["executed_action"]),
            targets=DiagnosticTargets.from_mapping(value["targets"]),
            source_record_sha256=str(value["source_record_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class Pillar2DiagnosticInput:
    diagnostic_id: str
    campaign_id: str
    source_partition: str
    registered_seed: int
    bootstrap_samples: int
    bootstrap_confidence: float
    bootstrap_seed: int
    neutral_text_id: str
    neutral_text: str
    neutral_images: tuple[ImageArtifact, ...]
    examples: tuple[DiagnosticExample, ...]
    backend_identity: DiagnosticBackendIdentity
    actions_frozen_before_diagnostics: bool
    runtime_outputs_consumed: bool
    affects_primary_table2: bool
    locked_test_rows_read: int
    evidence_role: str = ENGINEERING_EVIDENCE_ROLE
    paper_table_status: str = PAPER_TABLE_STATUS
    schema_version: str = PILLAR2_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("diagnostic_id", "campaign_id"):
            _require_text(getattr(self, name), context=name)
        if self.source_partition not in ALLOWED_SOURCE_PARTITIONS:
            raise SchemaError(
                "P2 diagnostics source_partition is unregistered or locked"
            )
        if type(self.registered_seed) is not int or self.registered_seed < 0:
            raise SchemaError("registered_seed must be a nonnegative integer")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples <= 0:
            raise SchemaError("bootstrap_samples must be a positive exact integer")
        if self.bootstrap_confidence != PILLAR2_BOOTSTRAP_CONFIDENCE:
            raise SchemaError("P2 diagnostic confidence must remain exactly 0.95")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise SchemaError("bootstrap_seed must be a nonnegative exact integer")
        if self.neutral_text_id != NEUTRAL_TEXT_ID:
            raise SchemaError("neutral text control identity is not registered")
        _require_text(self.neutral_text, context="neutral_text")
        if not self.neutral_images or not self.examples:
            raise SchemaError("P2 diagnostic requires neutral images and examples")
        ids = [example.example_id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise SchemaError("P2 diagnostic example IDs must be unique")
        neutral_sizes = [(image.width, image.height) for image in self.neutral_images]
        if len(neutral_sizes) != len(set(neutral_sizes)):
            raise SchemaError("neutral image dimensions must be unique")
        required_sizes = {
            (image.width, image.height)
            for example in self.examples
            for image in (example.pre_image, example.post_image)
        }
        if not required_sizes <= set(neutral_sizes):
            raise SchemaError("a registered same-size neutral image is missing")
        source_image_hashes = {
            image.sha256
            for example in self.examples
            for image in (example.pre_image, example.post_image)
        }
        if any(image.sha256 in source_image_hashes for image in self.neutral_images):
            raise SchemaError("neutral image must differ from every observed source image")
        if any(
            self.neutral_text
            in {
                example.text_state.task_text,
                example.text_state.title,
            }
            for example in self.examples
        ):
            raise SchemaError("neutral text must differ from observed task/title text")
        if self.actions_frozen_before_diagnostics is not True:
            raise SchemaError("P2 diagnostics may run only after actions are frozen")
        if self.runtime_outputs_consumed is not False:
            raise SchemaError("P2 diagnostics may not feed outputs into runtime")
        if self.affects_primary_table2 is not False:
            raise SchemaError("P2 diagnostics may not alter primary Table 2")
        if type(self.locked_test_rows_read) is not int or self.locked_test_rows_read != 0:
            raise SchemaError("P2 pilot diagnostics require zero locked-test reads")
        if self.evidence_role not in {
            PILLAR2_EVIDENCE_ROLE,
            ENGINEERING_EVIDENCE_ROLE,
        }:
            raise SchemaError("P2 evidence role is unregistered")
        if self.paper_table_status != PAPER_TABLE_STATUS:
            raise SchemaError("P2 diagnostic package must keep paper Table 2 N/R")
        if self.schema_version != PILLAR2_INPUT_SCHEMA_VERSION:
            raise SchemaError("unregistered P2 diagnostic input schema")
        _validate_donor_coverage(self.examples)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Pillar2DiagnosticInput":
        _require_exact_keys(
            value,
            {
                "diagnostic_id",
                "campaign_id",
                "source_partition",
                "registered_seed",
                "bootstrap_samples",
                "bootstrap_confidence",
                "bootstrap_seed",
                "neutral_text_id",
                "neutral_text",
                "neutral_images",
                "examples",
                "backend_identity",
                "actions_frozen_before_diagnostics",
                "runtime_outputs_consumed",
                "affects_primary_table2",
                "locked_test_rows_read",
                "evidence_role",
                "paper_table_status",
                "schema_version",
            },
            context="P2 diagnostic input",
        )
        if not isinstance(value["backend_identity"], Mapping):
            raise SchemaError("P2 backend identity must be an object")
        if not isinstance(value["neutral_images"], list) or not isinstance(
            value["examples"], list
        ):
            raise SchemaError("P2 neutral_images/examples must be arrays")
        return cls(
            diagnostic_id=str(value["diagnostic_id"]),
            campaign_id=str(value["campaign_id"]),
            source_partition=str(value["source_partition"]),
            registered_seed=value["registered_seed"],
            bootstrap_samples=value["bootstrap_samples"],
            bootstrap_confidence=value["bootstrap_confidence"],
            bootstrap_seed=value["bootstrap_seed"],
            neutral_text_id=str(value["neutral_text_id"]),
            neutral_text=str(value["neutral_text"]),
            neutral_images=tuple(
                ImageArtifact.from_mapping(item) for item in value["neutral_images"]
            ),
            examples=tuple(
                DiagnosticExample.from_mapping(item) for item in value["examples"]
            ),
            backend_identity=DiagnosticBackendIdentity.from_mapping(
                value["backend_identity"]
            ),
            actions_frozen_before_diagnostics=_require_bool(
                value["actions_frozen_before_diagnostics"],
                context="actions_frozen_before_diagnostics",
            ),
            runtime_outputs_consumed=_require_bool(
                value["runtime_outputs_consumed"], context="runtime_outputs_consumed"
            ),
            affects_primary_table2=_require_bool(
                value["affects_primary_table2"], context="affects_primary_table2"
            ),
            locked_test_rows_read=value["locked_test_rows_read"],
            evidence_role=str(value["evidence_role"]),
            paper_table_status=str(value["paper_table_status"]),
            schema_version=str(value["schema_version"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Pillar2DiagnosticInput":
        return cls.from_mapping(read_json(path))

    @property
    def record_sha256(self) -> str:
        return sha256_json(_diagnostic_input_public_mapping(self))


@dataclass(frozen=True, slots=True)
class PreDiagnosticView:
    """The only object visible to pre-action diagnostic inference."""

    example_id: str
    task_id: str
    image: ResolvedImage
    text_state: DiagnosticTextState

    def public_mapping(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task_id": self.task_id,
            "image": self.image.public_identity(),
            "text_state": self.text_state.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PostDiagnosticView:
    """Completed causal transition visible only to post-action inference."""

    example_id: str
    task_id: str
    pre_image: ResolvedImage
    post_image: ResolvedImage
    text_state: DiagnosticTextState
    executed_action: str | None

    def __post_init__(self) -> None:
        if self.executed_action is not None and self.executed_action not in ACTION_TYPE:
            raise SchemaError("post diagnostic executed action is not registered")

    def public_mapping(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "task_id": self.task_id,
            "pre_image": self.pre_image.public_identity(),
            "post_image": self.post_image.public_identity(),
            "text_state": self.text_state.to_dict(),
            "executed_action": self.executed_action,
        }


@dataclass(frozen=True, slots=True)
class PreDiagnosticPrediction:
    action_probabilities: Mapping[str, float]
    bbox: tuple[float, float, float, float]
    confidence_before: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_probabilities",
            _probability_distribution(
                self.action_probabilities,
                labels=ACTION_LABELS,
                context="pre.action_probabilities",
            ),
        )
        object.__setattr__(self, "bbox", _bbox(self.bbox, context="pre.bbox", nullable=False))
        object.__setattr__(
            self,
            "confidence_before",
            _require_probability(self.confidence_before, context="pre.confidence_before"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_probabilities": dict(self.action_probabilities),
            "bbox": list(self.bbox),
            "confidence_before": self.confidence_before,
        }


@dataclass(frozen=True, slots=True)
class PostDiagnosticPrediction:
    failure_probability: float
    failure_type_probabilities: Mapping[str, float]
    needs_recovery_probability: float
    recovery_probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failure_probability",
            _require_probability(self.failure_probability, context="post.failure_probability"),
        )
        object.__setattr__(
            self,
            "failure_type_probabilities",
            _probability_distribution(
                self.failure_type_probabilities,
                labels=FAILURE_LABELS,
                context="post.failure_type_probabilities",
            ),
        )
        object.__setattr__(
            self,
            "needs_recovery_probability",
            _require_probability(
                self.needs_recovery_probability,
                context="post.needs_recovery_probability",
            ),
        )
        object.__setattr__(
            self,
            "recovery_probabilities",
            _probability_distribution(
                self.recovery_probabilities,
                labels=RECOVERY_LABELS,
                context="post.recovery_probabilities",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_probability": self.failure_probability,
            "failure_type_probabilities": dict(self.failure_type_probabilities),
            "needs_recovery_probability": self.needs_recovery_probability,
            "recovery_probabilities": dict(self.recovery_probabilities),
        }


class Pillar2DiagnosticPredictor(Protocol):
    """Read-only diagnostic ABI. Neither method can execute browser behavior."""

    @property
    def diagnostic_identity(self) -> DiagnosticBackendIdentity: ...

    def predict_pre(self, view: PreDiagnosticView, *, seed: int) -> PreDiagnosticPrediction: ...

    def predict_post(
        self, view: PostDiagnosticView, *, seed: int
    ) -> PostDiagnosticPrediction: ...


def _validate_donor_coverage(examples: Sequence[DiagnosticExample]) -> None:
    for example in examples:
        post_candidates = [
            row
            for row in examples
            if row.example_id != example.example_id
            and row.post_image.sha256 != example.post_image.sha256
            and (row.post_image.width, row.post_image.height)
            == (example.post_image.width, example.post_image.height)
        ]
        if not post_candidates:
            raise SchemaError(
                f"example {example.example_id} has no same-size different post-state donor"
            )
        if not any(
            row.example_id != example.example_id
            and row.executed_action != example.executed_action
            for row in examples
        ):
            raise SchemaError(
                f"example {example.example_id} has no different-action mismatch donor"
            )


def _diagnostic_input_public_mapping(value: Pillar2DiagnosticInput) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "diagnostic_id": value.diagnostic_id,
        "campaign_id": value.campaign_id,
        "source_partition": value.source_partition,
        "registered_seed": value.registered_seed,
        "bootstrap_samples": value.bootstrap_samples,
        "bootstrap_confidence": value.bootstrap_confidence,
        "bootstrap_seed": value.bootstrap_seed,
        "neutral_text_id": value.neutral_text_id,
        "neutral_text_sha256": sha256_bytes(value.neutral_text.encode("utf-8")),
        "neutral_images": [image.public_identity() for image in value.neutral_images],
        "examples": [
            {
                "example_id": row.example_id,
                "task_id": row.task_id,
                "pre_image": row.pre_image.public_identity(),
                "post_image": row.post_image.public_identity(),
                "text_state_sha256": sha256_json(row.text_state.to_dict()),
                "executed_action": row.executed_action,
                "targets_sha256": sha256_json(asdict(row.targets)),
                "source_record_sha256": row.source_record_sha256,
            }
            for row in value.examples
        ],
        "backend_identity": value.backend_identity.to_dict(),
        "actions_frozen_before_diagnostics": value.actions_frozen_before_diagnostics,
        "runtime_outputs_consumed": value.runtime_outputs_consumed,
        "affects_primary_table2": value.affects_primary_table2,
        "locked_test_rows_read": value.locked_test_rows_read,
        "evidence_role": value.evidence_role,
        "paper_table_status": value.paper_table_status,
    }


def _predictor_source_path(predictor: Pillar2DiagnosticPredictor) -> Path:
    target: Any = type(predictor)
    try:
        source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise Pillar2DiagnosticError("diagnostic predictor source is unresolved") from exc
    if not source_name:
        raise Pillar2DiagnosticError("diagnostic predictor source is unresolved")
    source = Path(source_name).absolute()
    if source.is_symlink() or not source.is_file():
        raise Pillar2DiagnosticError(
            "diagnostic predictor source must be a non-symlink regular file"
        )
    return source.resolve()


def _verify_predictor(
    predictor: Pillar2DiagnosticPredictor,
    expected: DiagnosticBackendIdentity,
) -> tuple[Path, str]:
    actual = predictor.diagnostic_identity
    if not isinstance(actual, DiagnosticBackendIdentity) or actual != expected:
        raise Pillar2DiagnosticError("diagnostic predictor identity differs from registration")
    predictor_type = type(predictor)
    if (
        predictor_type.__module__ != expected.predictor_module
        or predictor_type.__qualname__ != expected.predictor_qualname
    ):
        raise Pillar2DiagnosticError("diagnostic predictor class identity mismatch")
    source = _predictor_source_path(predictor)
    digest = sha256_file(source)
    if digest != expected.predictor_source_sha256:
        raise Pillar2DiagnosticError("diagnostic predictor source hash mismatch")
    return source, digest


def verify_clean_repository(
    repository_root: str | Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Require one exact clean Git checkout and full commit identity."""

    expected = _require_commit(expected_commit)
    root = Path(repository_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise Pillar2DiagnosticError("repository_root must be a non-symlink directory")
    root = root.resolve()
    try:
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Pillar2DiagnosticError(
            "P2 diagnostic requires an accessible Git checkout"
        ) from exc
    if Path(top_level).resolve() != root:
        raise Pillar2DiagnosticError("repository_root is not the Git top level")
    _require_commit(commit)
    if commit != expected:
        raise Pillar2DiagnosticError(
            "P2 diagnostic repository commit differs from registration"
        )
    if status.strip():
        raise Pillar2DiagnosticError("P2 diagnostic requires a clean Git checkout")
    return {
        "status": "CLEAN_VERIFIED",
        "repository_root_sha256": sha256_bytes(str(root).encode("utf-8")),
        "repository_commit": commit,
        "git_status_porcelain_sha256": sha256_bytes(status.encode("utf-8")),
    }


def _verify_committed_repository_source(
    repository_root: str | Path,
    source_path: str | Path,
    *,
    expected_commit: str,
) -> str:
    """Bind a loaded source file to bytes tracked by the registered commit."""

    root = Path(repository_root).resolve()
    source = Path(source_path).resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise Pillar2DiagnosticError(
            "diagnostic predictor source is outside the attested repository"
        ) from exc
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed = subprocess.check_output(
            ["git", "show", f"{_require_commit(expected_commit)}:{relative}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Pillar2DiagnosticError(
            "diagnostic predictor source is not tracked by the registered commit"
        ) from exc
    if tracked != relative or sha256_bytes(committed) != sha256_file(source):
        raise Pillar2DiagnosticError(
            "diagnostic predictor source differs from the registered commit bytes"
        )
    return relative


def _canonical_pc01_error(
    identity: DiagnosticBackendIdentity,
    *,
    predictor: Pillar2DiagnosticPredictor | None = None,
) -> str | None:
    from .pc01_pillar2_diagnostics import canonical_pc01_backend_error

    backend_error = canonical_pc01_backend_error(identity, predictor=predictor)
    if backend_error is not None:
        return backend_error
    return (
        f"{CANONICAL_INPUT_PROVENANCE_STATUS}: P2 requires replay of an "
        "authoritative frozen source manifest, access ledger, and source-record bytes"
    )


def _neutral_image_for(
    neutral: Mapping[tuple[int, int], ResolvedImage], image: ResolvedImage
) -> ResolvedImage:
    try:
        return neutral[(image.width, image.height)]
    except KeyError as exc:  # pragma: no cover - caught at schema construction too
        raise Pillar2DiagnosticError("same-size neutral image is absent") from exc


def _donor(
    candidates: Sequence[DiagnosticExample],
    *,
    diagnostic: Pillar2DiagnosticInput,
    example: DiagnosticExample,
    condition: DiagnosticCondition,
) -> DiagnosticExample:
    if not candidates:
        raise Pillar2DiagnosticError(
            f"no donor for {example.example_id}/{condition.value}"
        )
    return min(
        candidates,
        key=lambda row: sha256_json(
            {
                "diagnostic_id": diagnostic.diagnostic_id,
                "registered_seed": diagnostic.registered_seed,
                "example_id": example.example_id,
                "condition": condition.value,
                "donor_id": row.example_id,
            }
        ),
    )


def _top(probabilities: Mapping[str, float], labels: Sequence[str]) -> str:
    return max(labels, key=lambda label: (probabilities[label], -labels.index(label)))


def _tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return 0.5 * sum(abs(left[key] - right[key]) for key in left)


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _pre_delta(
    prediction: PreDiagnosticPrediction,
    baseline: PreDiagnosticPrediction,
) -> dict[str, Any]:
    return {
        "top_action_changed": _top(prediction.action_probabilities, ACTION_LABELS)
        != _top(baseline.action_probabilities, ACTION_LABELS),
        "action_total_variation": _tv(
            prediction.action_probabilities, baseline.action_probabilities
        ),
        "bbox_l1": sum(abs(a - b) for a, b in zip(prediction.bbox, baseline.bbox)) / 4,
        "confidence_absolute_delta": abs(
            prediction.confidence_before - baseline.confidence_before
        ),
    }


def _post_delta(
    prediction: PostDiagnosticPrediction,
    baseline: PostDiagnosticPrediction,
) -> dict[str, Any]:
    return {
        "failure_decision_changed": (prediction.failure_probability >= 0.5)
        != (baseline.failure_probability >= 0.5),
        "failure_probability_absolute_delta": abs(
            prediction.failure_probability - baseline.failure_probability
        ),
        "failure_type_changed": _top(
            prediction.failure_type_probabilities, FAILURE_LABELS
        )
        != _top(baseline.failure_type_probabilities, FAILURE_LABELS),
        "failure_type_total_variation": _tv(
            prediction.failure_type_probabilities,
            baseline.failure_type_probabilities,
        ),
        "needs_recovery_decision_changed": (
            prediction.needs_recovery_probability >= 0.5
        )
        != (baseline.needs_recovery_probability >= 0.5),
        "needs_recovery_probability_absolute_delta": abs(
            prediction.needs_recovery_probability
            - baseline.needs_recovery_probability
        ),
        "recovery_strategy_changed": _top(
            prediction.recovery_probabilities, RECOVERY_LABELS
        )
        != _top(baseline.recovery_probabilities, RECOVERY_LABELS),
        "recovery_total_variation": _tv(
            prediction.recovery_probabilities, baseline.recovery_probabilities
        ),
    }


def _pre_score(
    prediction: PreDiagnosticPrediction, targets: DiagnosticTargets
) -> dict[str, Any]:
    return {
        "action_correct": _top(prediction.action_probabilities, ACTION_LABELS)
        == targets.action_type,
        "bbox_iou": (
            _bbox_iou(prediction.bbox, targets.action_bbox)
            if targets.action_bbox is not None
            else None
        ),
    }


def _post_score(
    prediction: PostDiagnosticPrediction, targets: DiagnosticTargets
) -> dict[str, bool]:
    return {
        "outcome_correct": (prediction.failure_probability >= 0.5) == targets.failure,
        "failure_type_correct": _top(
            prediction.failure_type_probabilities, FAILURE_LABELS
        )
        == targets.failure_type,
        "needs_recovery_correct": (
            prediction.needs_recovery_probability >= 0.5
        )
        == targets.needs_recovery,
        "recovery_strategy_correct": _top(
            prediction.recovery_probabilities, RECOVERY_LABELS
        )
        == targets.recovery_strategy,
    }


def _clustered_diagnostic_statistic(
    rows: Sequence[Mapping[str, Any]],
    *,
    values: Sequence[bool | float | None],
    metric_name: str,
    kind: str,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Example-level estimand with whole-task percentile bootstrap CIs."""

    if kind not in {"rate", "mean"} or len(rows) != len(values):
        raise SchemaError("P2 diagnostic statistic inputs are inconsistent")
    contributions = [
        {"task_id": str(row["task_id"]), "value": float(value)}
        for row, value in zip(rows, values)
        if value is not None
    ]
    numeric = [row["value"] for row in contributions]
    if any(not math.isfinite(value) for value in numeric):
        raise SchemaError(f"P2 diagnostic metric {metric_name} is non-finite")
    if kind == "rate" and any(value not in {0.0, 1.0} for value in numeric):
        raise SchemaError(f"P2 diagnostic rate {metric_name} is not binary")
    task_ids = sorted({row["task_id"] for row in contributions})
    derived_seed = stable_int_seed(seed, "p2_diagnostic", metric_name)
    estimate = sum(numeric) / len(numeric) if numeric else None
    common = {
        "metric_kind": kind,
        "estimate": estimate,
        "ci_low": None,
        "ci_high": None,
        "ci95_low": None,
        "ci95_high": None,
        "confidence": confidence,
        "interval_method": PILLAR2_INTERVAL_METHOD,
        "inference_unit": "task_id",
        "cluster_unit": "task_id",
        "contribution_unit": "diagnostic_example",
        "n_task_clusters": len(task_ids),
        "bootstrap_samples": samples,
        "bootstrap_seed": derived_seed,
        "valid_bootstrap_samples": 0,
    }
    if kind == "rate":
        kind_fields = {
            "numerator": int(sum(numeric)),
            "denominator": len(numeric),
        }
    else:
        kind_fields = {"total": float(sum(numeric)), "count": len(numeric)}
    if not numeric:
        return {
            **common,
            **kind_fields,
            "interval_status": "NOT_APPLICABLE",
            "reason": "no eligible diagnostic examples",
        }
    if len(task_ids) < 2:
        return {
            **common,
            **kind_fields,
            "interval_status": "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS",
            "reason": "fewer than two unique task clusters",
        }
    result = cluster_bootstrap(
        contributions,
        value=lambda row: float(row["value"]),
        cluster_key="task_id",
        samples=samples,
        confidence=confidence,
        seed=derived_seed,
    )
    if (
        result["n_clusters"] != len(task_ids)
        or result["n_rows"] != len(numeric)
        or not math.isclose(
            float(result["estimate"]), float(estimate), rel_tol=1e-12, abs_tol=1e-15
        )
    ):
        raise SchemaError("P2 task-cluster bootstrap changed the point estimand")
    return {
        **common,
        **kind_fields,
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "ci95_low": result["ci_low"],
        "ci95_high": result["ci_high"],
        "valid_bootstrap_samples": samples,
        "interval_status": "ESTIMATED",
        "reason": None,
    }


def _summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    def statistic(
        rows: Sequence[Mapping[str, Any]],
        *,
        metric_name: str,
        kind: str,
        value: Any,
    ) -> dict[str, Any]:
        return _clustered_diagnostic_statistic(
            rows,
            values=[value(row) for row in rows],
            metric_name=metric_name,
            kind=kind,
            samples=samples,
            confidence=confidence,
            seed=seed,
        )

    pre_summary: dict[str, Any] = {}
    for condition in PRE_CONDITIONS:
        rows = [
            row for row in records if row["phase"] == "pre" and row["condition"] == condition.value
        ]
        prefix = f"pre.{condition.value}"
        pre_summary[condition.value] = {
            "example_count": len(rows),
            "action_accuracy": statistic(
                rows,
                metric_name=f"{prefix}.action_accuracy",
                kind="rate",
                value=lambda row: row["score"]["action_correct"],
            ),
            "bbox_iou_mean": statistic(
                rows,
                metric_name=f"{prefix}.bbox_iou_mean",
                kind="mean",
                value=lambda row: row["score"]["bbox_iou"],
            ),
            "top_action_change_rate_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.top_action_change_rate_vs_full",
                kind="rate",
                value=lambda row: row["delta_from_full"]["top_action_changed"],
            ),
            "mean_action_total_variation_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.mean_action_total_variation_vs_full",
                kind="mean",
                value=lambda row: row["delta_from_full"]["action_total_variation"],
            ),
            "mean_bbox_l1_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.mean_bbox_l1_vs_full",
                kind="mean",
                value=lambda row: row["delta_from_full"]["bbox_l1"],
            ),
            "mean_confidence_absolute_delta_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.mean_confidence_absolute_delta_vs_full",
                kind="mean",
                value=lambda row: row["delta_from_full"][
                    "confidence_absolute_delta"
                ],
            ),
        }

    post_summary: dict[str, Any] = {}
    for condition in POST_CONDITIONS:
        rows = [
            row for row in records if row["phase"] == "post" and row["condition"] == condition.value
        ]
        prefix = f"post.{condition.value}"
        post_summary[condition.value] = {
            "example_count": len(rows),
            "outcome_accuracy": statistic(
                rows,
                metric_name=f"{prefix}.outcome_accuracy",
                kind="rate",
                value=lambda row: row["score"]["outcome_correct"],
            ),
            "failure_type_accuracy": statistic(
                rows,
                metric_name=f"{prefix}.failure_type_accuracy",
                kind="rate",
                value=lambda row: row["score"]["failure_type_correct"],
            ),
            "needs_recovery_accuracy": statistic(
                rows,
                metric_name=f"{prefix}.needs_recovery_accuracy",
                kind="rate",
                value=lambda row: row["score"]["needs_recovery_correct"],
            ),
            "recovery_strategy_accuracy": statistic(
                rows,
                metric_name=f"{prefix}.recovery_strategy_accuracy",
                kind="rate",
                value=lambda row: row["score"]["recovery_strategy_correct"],
            ),
            "failure_decision_change_rate_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.failure_decision_change_rate_vs_full",
                kind="rate",
                value=lambda row: row["delta_from_full"][
                    "failure_decision_changed"
                ],
            ),
            "mean_failure_probability_delta_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.mean_failure_probability_delta_vs_full",
                kind="mean",
                value=lambda row: row["delta_from_full"][
                    "failure_probability_absolute_delta"
                ],
            ),
            "failure_type_change_rate_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.failure_type_change_rate_vs_full",
                kind="rate",
                value=lambda row: row["delta_from_full"]["failure_type_changed"],
            ),
            "mean_failure_type_total_variation_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.mean_failure_type_total_variation_vs_full",
                kind="mean",
                value=lambda row: row["delta_from_full"][
                    "failure_type_total_variation"
                ],
            ),
            "needs_recovery_change_rate_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.needs_recovery_change_rate_vs_full",
                kind="rate",
                value=lambda row: row["delta_from_full"][
                    "needs_recovery_decision_changed"
                ],
            ),
            "mean_needs_recovery_probability_delta_vs_full": statistic(
                rows,
                metric_name=(
                    f"{prefix}.mean_needs_recovery_probability_delta_vs_full"
                ),
                kind="mean",
                value=lambda row: row["delta_from_full"][
                    "needs_recovery_probability_absolute_delta"
                ],
            ),
            "recovery_strategy_change_rate_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.recovery_strategy_change_rate_vs_full",
                kind="rate",
                value=lambda row: row["delta_from_full"][
                    "recovery_strategy_changed"
                ],
            ),
            "mean_recovery_total_variation_vs_full": statistic(
                rows,
                metric_name=f"{prefix}.mean_recovery_total_variation_vs_full",
                kind="mean",
                value=lambda row: row["delta_from_full"]["recovery_total_variation"],
            ),
        }
    return {
        "uncertainty": {
            "confidence": confidence,
            "interval_method": PILLAR2_INTERVAL_METHOD,
            "inference_unit": "task_id",
            "cluster_unit": "task_id",
            "contribution_unit": "diagnostic_example",
            "bootstrap_samples": samples,
            "base_seed": seed,
        },
        "pre_action_modality_sensitivity": pre_summary,
        "post_action_controls": post_summary,
    }


def _record(
    *,
    example: DiagnosticExample,
    phase: str,
    condition: DiagnosticCondition,
    view: PreDiagnosticView | PostDiagnosticView,
    prediction: PreDiagnosticPrediction | PostDiagnosticPrediction,
    baseline: PreDiagnosticPrediction | PostDiagnosticPrediction,
    seed: int,
    donor: DiagnosticExample | None = None,
) -> dict[str, Any]:
    if phase == "pre":
        assert isinstance(prediction, PreDiagnosticPrediction)
        assert isinstance(baseline, PreDiagnosticPrediction)
        delta = _pre_delta(prediction, baseline)
        score = _pre_score(prediction, example.targets)
    else:
        assert isinstance(prediction, PostDiagnosticPrediction)
        assert isinstance(baseline, PostDiagnosticPrediction)
        delta = _post_delta(prediction, baseline)
        score = _post_score(prediction, example.targets)
    prediction_mapping = prediction.to_dict()
    return {
        "example_id": example.example_id,
        "task_id": example.task_id,
        "phase": phase,
        "condition": condition.value,
        "stage_seed": seed,
        "input_sha256": sha256_json(view.public_mapping()),
        "prediction": prediction_mapping,
        "prediction_sha256": sha256_json(prediction_mapping),
        "delta_from_full": delta,
        "score": score,
        "donor": (
            None
            if donor is None
            else {
                "example_id": donor.example_id,
                "source_record_sha256": donor.source_record_sha256,
            }
        ),
    }


def run_pillar2_diagnostics(
    diagnostic: Pillar2DiagnosticInput,
    *,
    image_root: str | Path,
    predictor: Pillar2DiagnosticPredictor,
    repository_root: str | Path | None = None,
    allow_uncommitted_engineering: bool = False,
) -> dict[str, Any]:
    """Run the registered offline controls and return a JSON-ready report.

    The method never receives an environment adapter, action executor, recovery
    controller, sealed verifier, or memory store.  This structural boundary is
    intentional: diagnostic outputs cannot affect Table 2 execution.
    """

    # Verify the exact checkout before the lazy import of the canonical PC-01
    # adapter and before any predictor callback. The public CLI performs an
    # even earlier standard-library-only gate before importing this module.
    repository_receipt: dict[str, Any] | None = None
    if repository_root is not None:
        repository_receipt = verify_clean_repository(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
        )
    canonical_error = _canonical_pc01_error(
        diagnostic.backend_identity, predictor=predictor
    )
    canonical_pc01 = canonical_error is None
    provenance_blocked = bool(
        canonical_error
        and canonical_error.startswith(CANONICAL_INPUT_PROVENANCE_STATUS)
    )
    expected_role = (
        PILLAR2_EVIDENCE_ROLE if canonical_pc01 else ENGINEERING_EVIDENCE_ROLE
    )
    if diagnostic.evidence_role != expected_role:
        raise Pillar2DiagnosticError(
            "diagnostic evidence role is inconsistent with its predictor: "
            f"expected {expected_role}; canonical check: {canonical_error or 'PASS'}"
        )
    if (canonical_pc01 or provenance_blocked) and (
        diagnostic.bootstrap_samples != PILLAR2_BOOTSTRAP_SAMPLES
        or diagnostic.bootstrap_confidence != PILLAR2_BOOTSTRAP_CONFIDENCE
        or diagnostic.bootstrap_seed != PILLAR2_BOOTSTRAP_SEED
        or diagnostic.registered_seed != diagnostic.backend_identity.model_seed
    ):
        raise Pillar2DiagnosticError(
            "canonical PC-01 companion evidence changed registered seed/bootstrap "
            "semantics"
        )
    if repository_receipt is None:
        if provenance_blocked:
            raise Pillar2DiagnosticError(
                "exact PC-01 P2 diagnostics with blocked input provenance require "
                "clean repository verification"
            )
        if allow_uncommitted_engineering and not canonical_pc01:
            repository_receipt = {
                "status": "ENGINEERING_UNCOMMITTED_UNPROMOTABLE",
                "repository_root_sha256": None,
                "repository_commit": diagnostic.backend_identity.repository_commit,
                "git_status_porcelain_sha256": None,
            }
        else:
            raise Pillar2DiagnosticError(
                "P2 diagnostics require clean repository verification; only generic "
                "engineering fixtures may explicitly opt out"
            )
    assert repository_receipt is not None

    root = Path(image_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise Pillar2DiagnosticError("image_root must be a non-symlink directory")
    source_path, source_hash = _verify_predictor(
        predictor, diagnostic.backend_identity
    )
    if repository_root is not None:
        _verify_committed_repository_source(
            repository_root,
            source_path,
            expected_commit=diagnostic.backend_identity.repository_commit,
        )
    resolved_neutral = {
        (image.width, image.height): image.resolve(root)
        for image in diagnostic.neutral_images
    }
    resolved_examples = {
        example.example_id: (
            example.pre_image.resolve(root),
            example.post_image.resolve(root),
        )
        for example in diagnostic.examples
    }
    neutral_text = DiagnosticTextState.neutral(diagnostic.neutral_text)
    records: list[dict[str, Any]] = []
    temporal_replays: list[dict[str, Any]] = []

    for example in diagnostic.examples:
        pre_image, post_image = resolved_examples[example.example_id]
        pre_seed = stable_int_seed(
            diagnostic.diagnostic_id,
            example.example_id,
            "pre",
            diagnostic.registered_seed,
        )
        post_seed = stable_int_seed(
            diagnostic.diagnostic_id,
            example.example_id,
            "post",
            diagnostic.registered_seed,
        )
        full_pre_view = PreDiagnosticView(
            example.example_id, example.task_id, pre_image, example.text_state
        )
        baseline_pre = predictor.predict_pre(full_pre_view, seed=pre_seed)
        if not isinstance(baseline_pre, PreDiagnosticPrediction):
            raise Pillar2DiagnosticError("predict_pre returned an unregistered type")
        pre_views = {
            DiagnosticCondition.FULL: full_pre_view,
            DiagnosticCondition.IMAGE_OCCLUDED: PreDiagnosticView(
                example.example_id,
                example.task_id,
                _neutral_image_for(resolved_neutral, pre_image),
                example.text_state,
            ),
            DiagnosticCondition.TEXT_OCCLUDED: PreDiagnosticView(
                example.example_id,
                example.task_id,
                pre_image,
                neutral_text,
            ),
        }
        for condition, view in pre_views.items():
            prediction = (
                baseline_pre
                if condition is DiagnosticCondition.FULL
                else predictor.predict_pre(view, seed=pre_seed)
            )
            if not isinstance(prediction, PreDiagnosticPrediction):
                raise Pillar2DiagnosticError("predict_pre returned an unregistered type")
            records.append(
                _record(
                    example=example,
                    phase="pre",
                    condition=condition,
                    view=view,
                    prediction=prediction,
                    baseline=baseline_pre,
                    seed=pre_seed,
                )
            )

        post_donor = _donor(
            [
                row
                for row in diagnostic.examples
                if row.example_id != example.example_id
                and row.post_image.sha256 != example.post_image.sha256
                and (row.post_image.width, row.post_image.height)
                == (example.post_image.width, example.post_image.height)
            ],
            diagnostic=diagnostic,
            example=example,
            condition=DiagnosticCondition.POST_STATE_SWAPPED,
        )
        action_donor = _donor(
            [
                row
                for row in diagnostic.examples
                if row.example_id != example.example_id
                and row.executed_action != example.executed_action
            ],
            diagnostic=diagnostic,
            example=example,
            condition=DiagnosticCondition.ACTION_POST_MISMATCH,
        )
        donor_post_image = resolved_examples[post_donor.example_id][1]
        full_post_view = PostDiagnosticView(
            example.example_id,
            example.task_id,
            pre_image,
            post_image,
            example.text_state,
            example.executed_action,
        )
        baseline_post = predictor.predict_post(full_post_view, seed=post_seed)
        if not isinstance(baseline_post, PostDiagnosticPrediction):
            raise Pillar2DiagnosticError("predict_post returned an unregistered type")
        post_views: dict[
            DiagnosticCondition, tuple[PostDiagnosticView, DiagnosticExample | None]
        ] = {
            DiagnosticCondition.FULL: (full_post_view, None),
            DiagnosticCondition.IMAGE_OCCLUDED: (
                PostDiagnosticView(
                    example.example_id,
                    example.task_id,
                    _neutral_image_for(resolved_neutral, pre_image),
                    _neutral_image_for(resolved_neutral, post_image),
                    example.text_state,
                    example.executed_action,
                ),
                None,
            ),
            DiagnosticCondition.TEXT_OCCLUDED: (
                PostDiagnosticView(
                    example.example_id,
                    example.task_id,
                    pre_image,
                    post_image,
                    neutral_text,
                    example.executed_action,
                ),
                None,
            ),
            DiagnosticCondition.EXECUTED_ACTION_REMOVED: (
                PostDiagnosticView(
                    example.example_id,
                    example.task_id,
                    pre_image,
                    post_image,
                    example.text_state,
                    None,
                ),
                None,
            ),
            DiagnosticCondition.POST_STATE_SWAPPED: (
                PostDiagnosticView(
                    example.example_id,
                    example.task_id,
                    pre_image,
                    donor_post_image,
                    example.text_state,
                    example.executed_action,
                ),
                post_donor,
            ),
            DiagnosticCondition.ACTION_POST_MISMATCH: (
                PostDiagnosticView(
                    example.example_id,
                    example.task_id,
                    pre_image,
                    post_image,
                    example.text_state,
                    action_donor.executed_action,
                ),
                action_donor,
            ),
        }
        for condition, (view, donor) in post_views.items():
            prediction = (
                baseline_post
                if condition is DiagnosticCondition.FULL
                else predictor.predict_post(view, seed=post_seed)
            )
            if not isinstance(prediction, PostDiagnosticPrediction):
                raise Pillar2DiagnosticError("predict_post returned an unregistered type")
            records.append(
                _record(
                    example=example,
                    phase="post",
                    condition=condition,
                    view=view,
                    prediction=prediction,
                    baseline=baseline_post,
                    seed=post_seed,
                    donor=donor,
                )
            )

        # Replay the exact pre-action view after all post-action controls.  Any
        # future/post input, oracle target, or mutable stage state that leaks
        # into pre-action inference makes this fail closed.
        replay_pre = predictor.predict_pre(full_pre_view, seed=pre_seed)
        if not isinstance(replay_pre, PreDiagnosticPrediction):
            raise Pillar2DiagnosticError("predict_pre replay returned an unregistered type")
        baseline_hash = sha256_json(baseline_pre.to_dict())
        replay_hash = sha256_json(replay_pre.to_dict())
        if baseline_hash != replay_hash:
            raise Pillar2DiagnosticError(
                f"pre-action temporal replay changed for {example.example_id}"
            )
        temporal_replays.append(
            {
                "example_id": example.example_id,
                "pre_input_sha256": sha256_json(full_pre_view.public_mapping()),
                "baseline_prediction_sha256": baseline_hash,
                "replay_prediction_sha256": replay_hash,
                "identical": True,
            }
        )

    # Inputs and predictor source are rehashed after inference so a callback
    # cannot mutate the evidence it was evaluated on.
    for example in diagnostic.examples:
        example.pre_image.resolve(root)
        example.post_image.resolve(root)
    for image in diagnostic.neutral_images:
        image.resolve(root)
    if sha256_file(source_path) != source_hash:
        raise Pillar2DiagnosticError("diagnostic predictor source changed during inference")
    if predictor.diagnostic_identity != diagnostic.backend_identity:
        raise Pillar2DiagnosticError("diagnostic predictor identity changed during inference")
    if repository_root is not None:
        _verify_committed_repository_source(
            repository_root,
            source_path,
            expected_commit=diagnostic.backend_identity.repository_commit,
        )
        ending_repository_receipt = verify_clean_repository(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
        )
        if ending_repository_receipt != repository_receipt:
            raise Pillar2DiagnosticError(
                "P2 diagnostic repository identity changed during inference"
            )

    return {
        "schema_version": PILLAR2_REPORT_SCHEMA_VERSION,
        "diagnostic_id": diagnostic.diagnostic_id,
        "campaign_id": diagnostic.campaign_id,
        "evidence_role": expected_role,
        "promotion_status": (
            "CANONICAL_PC01_COMPANION_EVIDENCE"
            if canonical_pc01
            else (
                INPUT_PROVENANCE_BLOCKED_PROMOTION
                if provenance_blocked
                else "UNPROMOTABLE"
            )
        ),
        "paper_table_status": PAPER_TABLE_STATUS,
        "affects_primary_table2": False,
        "runtime_outputs_consumed": False,
        "browser_actions_executed": 0,
        "recovery_triggers_emitted": 0,
        "memory_queries_executed": 0,
        "memory_writes_executed": 0,
        "claim_scope": dict(PILLAR2_CLAIM_SCOPE),
        "source_partition": diagnostic.source_partition,
        "locked_test_rows_read": 0,
        "example_count": len(diagnostic.examples),
        "registered_seed": diagnostic.registered_seed,
        "bootstrap": {
            "samples": diagnostic.bootstrap_samples,
            "confidence": diagnostic.bootstrap_confidence,
            "base_seed": diagnostic.bootstrap_seed,
            "interval_method": PILLAR2_INTERVAL_METHOD,
            "inference_unit": "task_id",
            "cluster_unit": "task_id",
            "contribution_unit": "diagnostic_example",
        },
        "repository_verification": repository_receipt,
        "input_manifest_sha256": diagnostic.record_sha256,
        "backend_identity": diagnostic.backend_identity.to_dict(),
        "neutral_controls": {
            "neutral_text_id": diagnostic.neutral_text_id,
            "neutral_text_sha256": sha256_bytes(
                diagnostic.neutral_text.encode("utf-8")
            ),
            "neutral_images": [
                image.public_identity() for image in diagnostic.neutral_images
            ],
        },
        "condition_definitions": dict(PILLAR2_CONDITION_DEFINITIONS),
        "temporal_replay": {
            "status": "PASS",
            "count": len(temporal_replays),
            "records": temporal_replays,
        },
        "records": records,
        "summary": _summarize(
            records,
            samples=diagnostic.bootstrap_samples,
            confidence=diagnostic.bootstrap_confidence,
            seed=diagnostic.bootstrap_seed,
        ),
    }


def write_pillar2_diagnostic_package(
    output_dir: str | Path,
    *,
    diagnostic: Pillar2DiagnosticInput,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an immutable, hash-bound companion package to a new directory."""

    validate_pillar2_diagnostic_report(report)
    if (
        report.get("diagnostic_id") != diagnostic.diagnostic_id
        or report.get("campaign_id") != diagnostic.campaign_id
        or report.get("input_manifest_sha256") != diagnostic.record_sha256
        or report.get("backend_identity") != diagnostic.backend_identity.to_dict()
        or report.get("evidence_role") != diagnostic.evidence_role
    ):
        raise Pillar2DiagnosticError(
            "P2 diagnostic report is not bound to the supplied frozen input"
        )
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise Pillar2DiagnosticError("P2 diagnostic output directory is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    report_path = atomic_write_json(
        destination / "pillar2_diagnostic_report.json", report, mode=0o444
    )
    package = {
        "schema_version": PILLAR2_PACKAGE_SCHEMA_VERSION,
        "diagnostic_id": diagnostic.diagnostic_id,
        "evidence_role": report["evidence_role"],
        "promotion_status": report["promotion_status"],
        "paper_table_status": PAPER_TABLE_STATUS,
        "affects_primary_table2": False,
        "runtime_outputs_consumed": False,
        "report_relative_path": report_path.name,
        "report_sha256": sha256_file(report_path),
        "input_manifest_sha256": diagnostic.record_sha256,
        "backend_identity_sha256": sha256_json(
            diagnostic.backend_identity.to_dict()
        ),
    }
    package_path = atomic_write_json(
        destination / "package_manifest.json", package, mode=0o444
    )
    return {
        **package,
        "package_manifest_sha256": sha256_file(package_path),
        "output_dir": str(destination.resolve()),
    }


_STATISTIC_COMMON_FIELDS = {
    "metric_kind",
    "estimate",
    "ci_low",
    "ci_high",
    "ci95_low",
    "ci95_high",
    "confidence",
    "interval_method",
    "inference_unit",
    "cluster_unit",
    "contribution_unit",
    "n_task_clusters",
    "bootstrap_samples",
    "bootstrap_seed",
    "valid_bootstrap_samples",
    "interval_status",
    "reason",
}


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"{context} must be finite")
    return number


def _validate_diagnostic_statistic(
    value: object,
    *,
    metric_name: str,
    kind: str,
    samples: int,
    confidence: float,
    base_seed: int,
) -> None:
    """Validate every count, estimand, interval, and clustering field."""

    if not isinstance(value, Mapping):
        raise SchemaError(f"P2 summary metric {metric_name} must be an object")
    count_fields = (
        {"numerator", "denominator"}
        if kind == "rate"
        else {"total", "count"}
    )
    _require_exact_keys(
        value,
        _STATISTIC_COMMON_FIELDS | count_fields,
        context=f"P2 summary metric {metric_name}",
    )
    if value["metric_kind"] != kind:
        raise SchemaError(f"P2 summary metric {metric_name} changed kind")
    if (
        isinstance(value["confidence"], bool)
        or not isinstance(value["confidence"], (int, float))
        or float(value["confidence"]) != confidence
        or value["interval_method"] != PILLAR2_INTERVAL_METHOD
        or value["inference_unit"] != "task_id"
        or value["cluster_unit"] != "task_id"
        or value["contribution_unit"] != "diagnostic_example"
        or type(value["bootstrap_samples"]) is not int
        or value["bootstrap_samples"] != samples
        or type(value["bootstrap_seed"]) is not int
        or value["bootstrap_seed"]
        != stable_int_seed(base_seed, "p2_diagnostic", metric_name)
    ):
        raise SchemaError(f"P2 summary metric {metric_name} changed inference method")
    if (
        type(value["n_task_clusters"]) is not int
        or value["n_task_clusters"] < 0
        or type(value["valid_bootstrap_samples"]) is not int
        or value["valid_bootstrap_samples"] < 0
    ):
        raise SchemaError(f"P2 summary metric {metric_name} has invalid counts")

    if kind == "rate":
        if (
            type(value["numerator"]) is not int
            or type(value["denominator"]) is not int
            or value["denominator"] < 0
            or not 0 <= value["numerator"] <= value["denominator"]
        ):
            raise SchemaError(f"P2 summary rate {metric_name} has invalid counts")
        count = value["denominator"]
        expected_estimate = (
            None if count == 0 else value["numerator"] / value["denominator"]
        )
    else:
        if type(value["count"]) is not int or value["count"] < 0:
            raise SchemaError(f"P2 summary mean {metric_name} has invalid count")
        total = _finite_number(value["total"], context=f"{metric_name}.total")
        count = value["count"]
        expected_estimate = None if count == 0 else total / count

    if value["n_task_clusters"] > count:
        raise SchemaError(f"P2 summary metric {metric_name} has too many clusters")
    if expected_estimate is None:
        if value["estimate"] is not None:
            raise SchemaError(f"P2 summary metric {metric_name} must be undefined")
    else:
        estimate = _finite_number(
            value["estimate"], context=f"{metric_name}.estimate"
        )
        if not math.isclose(
            estimate, expected_estimate, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise SchemaError(f"P2 summary metric {metric_name} estimate/count mismatch")

    ci_values = (
        value["ci_low"],
        value["ci_high"],
        value["ci95_low"],
        value["ci95_high"],
    )
    if count == 0:
        expected_status = "NOT_APPLICABLE"
        expected_reason: str | None = "no eligible diagnostic examples"
    elif value["n_task_clusters"] < 2:
        if value["n_task_clusters"] != 1:
            raise SchemaError(
                f"P2 summary metric {metric_name} has examples but no task cluster"
            )
        expected_status = "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS"
        expected_reason = "fewer than two unique task clusters"
    else:
        expected_status = "ESTIMATED"
        expected_reason = None
    if (
        value["interval_status"] != expected_status
        or value["reason"] != expected_reason
    ):
        raise SchemaError(f"P2 summary metric {metric_name} has invalid status/reason")

    if expected_status == "ESTIMATED":
        low = _finite_number(value["ci_low"], context=f"{metric_name}.ci_low")
        high = _finite_number(value["ci_high"], context=f"{metric_name}.ci_high")
        low95 = _finite_number(
            value["ci95_low"], context=f"{metric_name}.ci95_low"
        )
        high95 = _finite_number(
            value["ci95_high"], context=f"{metric_name}.ci95_high"
        )
        if (
            low > high
            or not math.isclose(low95, low, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(high95, high, rel_tol=0.0, abs_tol=0.0)
        ):
            raise SchemaError(f"P2 summary metric {metric_name} has invalid 95% CI")
        if kind == "rate" and not 0.0 <= low <= high <= 1.0:
            raise SchemaError(f"P2 summary rate {metric_name} CI is outside [0, 1]")
        if value["valid_bootstrap_samples"] != samples:
            raise SchemaError(
                f"P2 summary metric {metric_name} has incomplete bootstrap samples"
            )
    elif any(item is not None for item in ci_values) or (
        value["valid_bootstrap_samples"] != 0
    ):
        raise SchemaError(f"P2 summary metric {metric_name} fabricated an interval")


def _validate_diagnostic_summary(
    summary: object,
    *,
    example_count: int,
    samples: int,
    confidence: float,
    base_seed: int,
) -> None:
    if not isinstance(summary, Mapping):
        raise SchemaError("P2 diagnostic summary must be an object")
    _require_exact_keys(
        summary,
        {
            "uncertainty",
            "pre_action_modality_sensitivity",
            "post_action_controls",
        },
        context="P2 diagnostic summary",
    )
    uncertainty = summary["uncertainty"]
    expected_uncertainty = {
        "confidence": confidence,
        "interval_method": PILLAR2_INTERVAL_METHOD,
        "inference_unit": "task_id",
        "cluster_unit": "task_id",
        "contribution_unit": "diagnostic_example",
        "bootstrap_samples": samples,
        "base_seed": base_seed,
    }
    if uncertainty != expected_uncertainty:
        raise SchemaError("P2 diagnostic summary uncertainty registration changed")

    families = (
        (
            "pre_action_modality_sensitivity",
            PRE_CONDITIONS,
            _PRE_SUMMARY_METRICS,
            "pre",
        ),
        ("post_action_controls", POST_CONDITIONS, _POST_SUMMARY_METRICS, "post"),
    )
    for family_name, conditions, metric_kinds, phase in families:
        family = summary[family_name]
        if not isinstance(family, Mapping) or set(family) != {
            condition.value for condition in conditions
        }:
            raise SchemaError(f"P2 summary family {family_name} is malformed")
        for condition in conditions:
            condition_summary = family[condition.value]
            if not isinstance(condition_summary, Mapping):
                raise SchemaError("P2 condition summary must be an object")
            _require_exact_keys(
                condition_summary,
                {"example_count", *metric_kinds},
                context=f"P2 summary {phase}.{condition.value}",
            )
            if (
                type(condition_summary["example_count"]) is not int
                or condition_summary["example_count"] != example_count
            ):
                raise SchemaError("P2 summary example count differs from records")
            for metric_name, kind in metric_kinds.items():
                _validate_diagnostic_statistic(
                    condition_summary[metric_name],
                    metric_name=f"{phase}.{condition.value}.{metric_name}",
                    kind=kind,
                    samples=samples,
                    confidence=confidence,
                    base_seed=base_seed,
                )


def validate_pillar2_diagnostic_report(report: Mapping[str, Any]) -> None:
    """Fail closed on any attempt to relabel diagnostics as Table 2 evidence."""

    _require_exact_keys(
        report,
        {
            "schema_version",
            "diagnostic_id",
            "campaign_id",
            "evidence_role",
            "promotion_status",
            "paper_table_status",
            "affects_primary_table2",
            "runtime_outputs_consumed",
            "browser_actions_executed",
            "recovery_triggers_emitted",
            "memory_queries_executed",
            "memory_writes_executed",
            "claim_scope",
            "source_partition",
            "locked_test_rows_read",
            "example_count",
            "registered_seed",
            "bootstrap",
            "repository_verification",
            "input_manifest_sha256",
            "backend_identity",
            "neutral_controls",
            "condition_definitions",
            "temporal_replay",
            "records",
            "summary",
        },
        context="P2 diagnostic report",
    )
    required = {
        "schema_version": PILLAR2_REPORT_SCHEMA_VERSION,
        "paper_table_status": PAPER_TABLE_STATUS,
        "affects_primary_table2": False,
        "runtime_outputs_consumed": False,
        "browser_actions_executed": 0,
        "recovery_triggers_emitted": 0,
        "memory_queries_executed": 0,
        "memory_writes_executed": 0,
        "locked_test_rows_read": 0,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise SchemaError(f"P2 diagnostic report changed protected field {field}")
    _require_text(report["diagnostic_id"], context="report.diagnostic_id")
    _require_text(report["campaign_id"], context="report.campaign_id")
    if report["source_partition"] not in ALLOWED_SOURCE_PARTITIONS:
        raise SchemaError("P2 diagnostic report source partition is unregistered")
    if type(report["registered_seed"]) is not int or report["registered_seed"] < 0:
        raise SchemaError("P2 diagnostic report registered seed is invalid")
    _require_sha256(
        report["input_manifest_sha256"], context="report.input_manifest_sha256"
    )
    role = report.get("evidence_role")
    promotion_status = report.get("promotion_status")
    if role not in {PILLAR2_EVIDENCE_ROLE, ENGINEERING_EVIDENCE_ROLE}:
        raise SchemaError("P2 diagnostic report evidence role is unregistered")
    backend_value = report.get("backend_identity")
    if not isinstance(backend_value, Mapping):
        raise SchemaError("P2 diagnostic report backend identity is absent")
    backend = DiagnosticBackendIdentity.from_mapping(backend_value)
    canonical_error = _canonical_pc01_error(backend)
    provenance_blocked = bool(
        canonical_error
        and canonical_error.startswith(CANONICAL_INPUT_PROVENANCE_STATUS)
    )
    if role == PILLAR2_EVIDENCE_ROLE:
        if canonical_error is not None:
            raise SchemaError(
                f"P2 companion evidence is not canonical PC-01: {canonical_error}"
            )
        if promotion_status != "CANONICAL_PC01_COMPANION_EVIDENCE":
            raise SchemaError("canonical PC-01 P2 promotion status changed")
        if report["registered_seed"] != backend.model_seed:
            raise SchemaError("canonical PC-01 diagnostic seed changed")
    elif provenance_blocked:
        if promotion_status != INPUT_PROVENANCE_BLOCKED_PROMOTION:
            raise SchemaError(
                "exact PC-01 P2 evidence must remain input-provenance blocked"
            )
        if report["registered_seed"] != backend.model_seed:
            raise SchemaError("input-provenance-blocked PC-01 diagnostic seed changed")
    elif promotion_status != "UNPROMOTABLE":
        raise SchemaError("generic P2 diagnostics must remain unpromotable")
    repository = report.get("repository_verification")
    if not isinstance(repository, Mapping) or set(repository) != {
        "status",
        "repository_root_sha256",
        "repository_commit",
        "git_status_porcelain_sha256",
    }:
        raise SchemaError("P2 repository verification receipt is malformed")
    if repository["repository_commit"] != backend.repository_commit:
        raise SchemaError("P2 repository receipt cites another commit")
    if role == PILLAR2_EVIDENCE_ROLE or provenance_blocked:
        if repository["status"] != "CLEAN_VERIFIED":
            raise SchemaError(
                "exact PC-01 P2 evidence lacks clean Git verification"
            )
        _require_sha256(
            repository["repository_root_sha256"],
            context="repository_root_sha256",
        )
        _require_sha256(
            repository["git_status_porcelain_sha256"],
            context="git_status_porcelain_sha256",
        )
    elif repository["status"] == "CLEAN_VERIFIED":
        _require_sha256(
            repository["repository_root_sha256"],
            context="repository_root_sha256",
        )
        _require_sha256(
            repository["git_status_porcelain_sha256"],
            context="git_status_porcelain_sha256",
        )
    elif repository["status"] == "ENGINEERING_UNCOMMITTED_UNPROMOTABLE":
        if (
            repository["repository_root_sha256"] is not None
            or repository["git_status_porcelain_sha256"] is not None
        ):
            raise SchemaError("uncommitted P2 fixture fabricated a clean Git receipt")
    else:
        raise SchemaError("generic P2 repository status is unregistered")
    bootstrap = report.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or set(bootstrap) != {
        "samples",
        "confidence",
        "base_seed",
        "interval_method",
        "inference_unit",
        "cluster_unit",
        "contribution_unit",
    }:
        raise SchemaError("P2 diagnostic bootstrap registration is malformed")
    if (
        type(bootstrap["samples"]) is not int
        or bootstrap["samples"] <= 0
        or isinstance(bootstrap["confidence"], bool)
        or not isinstance(bootstrap["confidence"], (int, float))
        or float(bootstrap["confidence"]) != PILLAR2_BOOTSTRAP_CONFIDENCE
        or type(bootstrap["base_seed"]) is not int
        or bootstrap["base_seed"] < 0
        or bootstrap["interval_method"] != PILLAR2_INTERVAL_METHOD
        or bootstrap["inference_unit"] != "task_id"
        or bootstrap["cluster_unit"] != "task_id"
        or bootstrap["contribution_unit"] != "diagnostic_example"
    ):
        raise SchemaError("P2 diagnostic bootstrap registration changed")
    if (role == PILLAR2_EVIDENCE_ROLE or provenance_blocked) and (
        bootstrap["samples"] != PILLAR2_BOOTSTRAP_SAMPLES
        or bootstrap["base_seed"] != PILLAR2_BOOTSTRAP_SEED
    ):
        raise SchemaError("exact PC-01 bootstrap registration changed")
    claim_scope = report.get("claim_scope")
    if not isinstance(claim_scope, Mapping) or claim_scope != PILLAR2_CLAIM_SCOPE:
        raise SchemaError("P2 diagnostic claim scope was broadened")
    if report.get("condition_definitions") != PILLAR2_CONDITION_DEFINITIONS:
        raise SchemaError("P2 diagnostic condition definitions changed")
    controls = report.get("neutral_controls")
    if not isinstance(controls, Mapping):
        raise SchemaError("P2 diagnostic neutral controls are absent")
    _require_exact_keys(
        controls,
        {"neutral_text_id", "neutral_text_sha256", "neutral_images"},
        context="P2 diagnostic neutral controls",
    )
    if controls["neutral_text_id"] != NEUTRAL_TEXT_ID:
        raise SchemaError("P2 neutral text identity changed")
    _require_sha256(
        controls["neutral_text_sha256"], context="neutral_text_sha256"
    )
    neutral_images = controls["neutral_images"]
    if not isinstance(neutral_images, list) or not neutral_images:
        raise SchemaError("P2 diagnostic neutral image evidence is absent")
    neutral_ids: set[str] = set()
    for image in neutral_images:
        if not isinstance(image, Mapping):
            raise SchemaError("P2 diagnostic neutral image identity is malformed")
        _require_exact_keys(
            image,
            {"artifact_id", "sha256", "width", "height"},
            context="P2 diagnostic neutral image identity",
        )
        artifact_id = _require_text(
            image["artifact_id"], context="neutral_image.artifact_id"
        )
        if artifact_id in neutral_ids:
            raise SchemaError("P2 diagnostic neutral image identity is duplicate")
        neutral_ids.add(artifact_id)
        _require_sha256(image["sha256"], context="neutral_image.sha256")
        if (
            type(image["width"]) is not int
            or type(image["height"]) is not int
            or image["width"] <= 0
            or image["height"] <= 0
        ):
            raise SchemaError("P2 diagnostic neutral image dimensions are invalid")
    replay = report.get("temporal_replay")
    if not isinstance(replay, Mapping) or replay.get("status") != "PASS":
        raise SchemaError("P2 temporal replay did not pass")
    example_count = report.get("example_count")
    if type(example_count) is not int or example_count <= 0:
        raise SchemaError("P2 diagnostic example_count must be positive")
    replay_rows = replay.get("records")
    if (
        type(replay.get("count")) is not int
        or replay.get("count") != example_count
        or not isinstance(replay_rows, list)
        or len(replay_rows) != example_count
    ):
        raise SchemaError("P2 temporal replay cardinality mismatch")
    replay_ids: set[str] = set()
    for row in replay_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "example_id",
            "pre_input_sha256",
            "baseline_prediction_sha256",
            "replay_prediction_sha256",
            "identical",
        }:
            raise SchemaError("P2 temporal replay row is malformed")
        example_id = str(row["example_id"])
        if not example_id or example_id in replay_ids:
            raise SchemaError("P2 temporal replay IDs are empty or duplicate")
        replay_ids.add(example_id)
        for field in (
            "pre_input_sha256",
            "baseline_prediction_sha256",
            "replay_prediction_sha256",
        ):
            _require_sha256(row[field], context=f"temporal_replay.{field}")
        if (
            row["identical"] is not True
            or row["baseline_prediction_sha256"]
            != row["replay_prediction_sha256"]
        ):
            raise SchemaError("P2 temporal replay prediction changed")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise SchemaError("P2 diagnostic report has no prediction records")
    if len(records) != example_count * (
        len(PRE_CONDITIONS) + len(POST_CONDITIONS)
    ):
        raise SchemaError("P2 diagnostic report condition cardinality mismatch")
    expected_record_fields = {
        "example_id",
        "task_id",
        "phase",
        "condition",
        "stage_seed",
        "input_sha256",
        "prediction",
        "prediction_sha256",
        "delta_from_full",
        "score",
        "donor",
    }
    keyed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != expected_record_fields:
            raise SchemaError("P2 diagnostic prediction row is not an object")
        example_id = str(row["example_id"])
        task_id = str(row["task_id"])
        phase = str(row["phase"])
        condition = str(row["condition"])
        _require_text(example_id, context="prediction.example_id")
        _require_text(task_id, context="prediction.task_id")
        permitted = (
            {item.value for item in PRE_CONDITIONS}
            if phase == "pre"
            else {item.value for item in POST_CONDITIONS}
            if phase == "post"
            else set()
        )
        if condition not in permitted:
            raise SchemaError("P2 diagnostic phase/condition is unregistered")
        key = (example_id, phase, condition)
        if key in keyed:
            raise SchemaError("P2 diagnostic phase/condition row is duplicate")
        keyed[key] = row
        expected_stage_seed = stable_int_seed(
            report["diagnostic_id"],
            example_id,
            phase,
            report["registered_seed"],
        )
        if (
            type(row["stage_seed"]) is not int
            or row["stage_seed"] != expected_stage_seed
        ):
            raise SchemaError("P2 diagnostic stage seed is invalid")
        _require_sha256(row["input_sha256"], context="prediction.input_sha256")
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping) or row.get("prediction_sha256") != sha256_json(
            prediction
        ):
            raise SchemaError("P2 diagnostic prediction hash mismatch")
        try:
            if phase == "pre":
                parsed_prediction: PreDiagnosticPrediction | PostDiagnosticPrediction = (
                    PreDiagnosticPrediction(
                        action_probabilities=prediction["action_probabilities"],
                        bbox=prediction["bbox"],
                        confidence_before=prediction["confidence_before"],
                    )
                )
            else:
                parsed_prediction = PostDiagnosticPrediction(
                    failure_probability=prediction["failure_probability"],
                    failure_type_probabilities=prediction[
                        "failure_type_probabilities"
                    ],
                    needs_recovery_probability=prediction[
                        "needs_recovery_probability"
                    ],
                    recovery_probabilities=prediction["recovery_probabilities"],
                )
        except (KeyError, TypeError, ValueError, SchemaError) as exc:
            raise SchemaError("P2 diagnostic prediction payload is malformed") from exc
        if parsed_prediction.to_dict() != prediction:
            raise SchemaError("P2 diagnostic prediction is not canonical")
        score = row["score"]
        if not isinstance(score, Mapping):
            raise SchemaError("P2 diagnostic score is malformed")
        if phase == "pre":
            _require_exact_keys(
                score,
                {"action_correct", "bbox_iou"},
                context="P2 pre diagnostic score",
            )
            _require_bool(score["action_correct"], context="score.action_correct")
            if score["bbox_iou"] is not None:
                _require_probability(score["bbox_iou"], context="score.bbox_iou")
        else:
            _require_exact_keys(
                score,
                {
                    "outcome_correct",
                    "failure_type_correct",
                    "needs_recovery_correct",
                    "recovery_strategy_correct",
                },
                context="P2 post diagnostic score",
            )
            for score_name in score:
                _require_bool(score[score_name], context=f"score.{score_name}")
        donor = row["donor"]
        needs_donor = condition in {
            DiagnosticCondition.POST_STATE_SWAPPED.value,
            DiagnosticCondition.ACTION_POST_MISMATCH.value,
        }
        if needs_donor:
            if not isinstance(donor, Mapping) or set(donor) != {
                "example_id",
                "source_record_sha256",
            }:
                raise SchemaError("P2 diagnostic donor evidence is missing")
            if str(donor["example_id"]) == example_id:
                raise SchemaError("P2 diagnostic uses its own example as donor")
            _require_sha256(
                donor["source_record_sha256"], context="prediction.donor source"
            )
        elif donor is not None:
            raise SchemaError("P2 diagnostic has an unregistered donor")

    example_ids = {key[0] for key in keyed}
    if example_ids != replay_ids or len(example_ids) != example_count:
        raise SchemaError("P2 prediction/replay example identities differ")
    for example_id in example_ids:
        for phase, conditions in (("pre", PRE_CONDITIONS), ("post", POST_CONDITIONS)):
            if {
                condition.value
                for condition in conditions
                if (example_id, phase, condition.value) in keyed
            } != {condition.value for condition in conditions}:
                raise SchemaError("P2 diagnostic is missing a registered condition")
        baseline_pre = keyed[(example_id, "pre", DiagnosticCondition.FULL.value)]
        if (
            baseline_pre["prediction_sha256"]
            != next(
                row["baseline_prediction_sha256"]
                for row in replay_rows
                if row["example_id"] == example_id
            )
        ):
            raise SchemaError("P2 temporal replay is bound to another baseline")
        pre_baseline_prediction = PreDiagnosticPrediction(
            **baseline_pre["prediction"]
        )
        post_baseline_prediction = PostDiagnosticPrediction(
            **keyed[
                (example_id, "post", DiagnosticCondition.FULL.value)
            ]["prediction"]
        )
        for condition in PRE_CONDITIONS:
            row = keyed[(example_id, "pre", condition.value)]
            parsed = PreDiagnosticPrediction(**row["prediction"])
            if row["delta_from_full"] != _pre_delta(
                parsed, pre_baseline_prediction
            ):
                raise SchemaError("P2 pre-action delta is not reproducible")
        for condition in POST_CONDITIONS:
            row = keyed[(example_id, "post", condition.value)]
            parsed = PostDiagnosticPrediction(**row["prediction"])
            if row["delta_from_full"] != _post_delta(
                parsed, post_baseline_prediction
            ):
                raise SchemaError("P2 post-action delta is not reproducible")
    _validate_diagnostic_summary(
        report.get("summary"),
        example_count=example_count,
        samples=bootstrap["samples"],
        confidence=float(bootstrap["confidence"]),
        base_seed=bootstrap["base_seed"],
    )
    if report.get("summary") != _summarize(
        records,
        samples=bootstrap["samples"],
        confidence=bootstrap["confidence"],
        seed=bootstrap["base_seed"],
    ):
        raise SchemaError("P2 diagnostic summary is not reproducible")


def validate_pillar2_diagnostic_package(
    package_dir: str | Path,
) -> dict[str, Any]:
    """Revalidate a stored standalone P2 companion package from bytes."""

    root = Path(package_dir)
    manifest_path = root / "package_manifest.json"
    report_path = root / "pillar2_diagnostic_report.json"
    for path in (manifest_path, report_path):
        if path.is_symlink() or not path.is_file():
            raise SchemaError(f"P2 package artifact is absent or unsafe: {path.name}")
    manifest = read_json(manifest_path)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "diagnostic_id",
            "evidence_role",
            "promotion_status",
            "paper_table_status",
            "affects_primary_table2",
            "runtime_outputs_consumed",
            "report_relative_path",
            "report_sha256",
            "input_manifest_sha256",
            "backend_identity_sha256",
        },
        context="P2 package manifest",
    )
    if (
        manifest["schema_version"] != PILLAR2_PACKAGE_SCHEMA_VERSION
        or manifest["evidence_role"] not in {
            PILLAR2_EVIDENCE_ROLE,
            ENGINEERING_EVIDENCE_ROLE,
        }
        or manifest["promotion_status"]
        not in {
            "CANONICAL_PC01_COMPANION_EVIDENCE",
            INPUT_PROVENANCE_BLOCKED_PROMOTION,
            "UNPROMOTABLE",
        }
        or manifest["paper_table_status"] != PAPER_TABLE_STATUS
        or manifest["affects_primary_table2"] is not False
        or manifest["runtime_outputs_consumed"] is not False
        or manifest["report_relative_path"] != report_path.name
    ):
        raise SchemaError("P2 package protected identity changed")
    for field in (
        "report_sha256",
        "input_manifest_sha256",
        "backend_identity_sha256",
    ):
        _require_sha256(manifest[field], context=f"P2 package {field}")
    if sha256_file(report_path) != manifest["report_sha256"]:
        raise SchemaError("P2 package report hash mismatch")
    report = read_json(report_path)
    validate_pillar2_diagnostic_report(report)
    if (
        report.get("diagnostic_id") != manifest["diagnostic_id"]
        or report.get("evidence_role") != manifest["evidence_role"]
        or report.get("promotion_status") != manifest["promotion_status"]
        or report.get("input_manifest_sha256") != manifest["input_manifest_sha256"]
        or sha256_json(report.get("backend_identity"))
        != manifest["backend_identity_sha256"]
    ):
        raise SchemaError("P2 package report/manifest bindings differ")
    return {
        "status": "PASS",
        "schema_version": PILLAR2_PACKAGE_SCHEMA_VERSION,
        "diagnostic_id": manifest["diagnostic_id"],
        "paper_table_status": PAPER_TABLE_STATUS,
        "affects_primary_table2": False,
        "package_manifest_sha256": sha256_file(manifest_path),
        "report_sha256": manifest["report_sha256"],
    }
