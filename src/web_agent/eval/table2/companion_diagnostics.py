"""Shared fail-closed primitives for offline pillar companion diagnostics.

The companion packages are deliberately separate from the E0--E3 campaign.
They may inspect frozen, causal evidence and invoke a frozen diagnostic
backend, but they have no browser, verifier, recovery-controller, or memory
write capability.  This module contains only serialization, source identity,
confidence-interval, and package-integrity machinery shared by P1, P3, and P4.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import stat
import subprocess
from typing import Any, Protocol
from urllib.parse import urlparse

from web_agent.runtime.observation import CausalBoundaryError, assert_oracle_blind_mapping

from .common import (
    SchemaError,
    atomic_write_json,
    canonical_json_bytes,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    sha256_json,
    stable_int_seed,
)
from .statistics import cluster_bootstrap


COMPANION_BACKEND_SCHEMA_VERSION = "table2.companion-diagnostic-backend.v1"
COMPANION_PACKAGE_SCHEMA_VERSION = "table2.companion-diagnostic-package.v1"
ENGINEERING_EVIDENCE_ROLE = "ENGINEERING_DIAGNOSTIC_ONLY_UNPROMOTABLE"
CANONICAL_INPUT_PROVENANCE_STATUS = (
    "BLOCKED_AUTHORITATIVE_INPUT_PROVENANCE_REQUIRED"
)
INPUT_PROVENANCE_BLOCKED_PROMOTION = "UNPROMOTABLE_INPUT_PROVENANCE_BLOCKED"
PAPER_TABLE_STATUS = "N/R"
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20250831
INTERVAL_METHOD = "percentile_task_cluster_bootstrap"
ALLOWED_SOURCE_PARTITIONS = frozenset(
    {
        "train_diagnostic",
        "validation_only",
        "public_development",
        "completed_public_pilot_campaign",
    }
)
PC01_MODEL_ID = "qwen2vl_2b_gold_v2_8_dgx"
PC01_MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
PC01_CHECKPOINT_SHA256 = (
    "9eaab6d24719b7bce8d0dd2ccf9169c3ddf83e0a800714a84531679c0c94895a"
)
PC01_CONFIG_SHA256 = (
    "d014050287ae2142e1c2111cff8b00de214dc3f416fedb49504edde8bd61007f"
)
PC01_PROCESSOR_SHA256 = (
    "b3c9f629a3c4ce6ab7bb71b27c7e0b560e3e21c51a19d1235fd4a507ba437797"
)


class CompanionDiagnosticError(RuntimeError):
    """A companion diagnostic violated its scientific or evidence boundary."""


def require_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context} must be non-empty text")
    return value


def require_sha256(value: object, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaError(f"{context} must be one lowercase SHA-256")
    return text


def require_commit(value: object, *, context: str = "repository_commit") -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise SchemaError(f"{context} must be one full lowercase Git commit")
    return text


def require_bool(value: object, *, context: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{context} must be an exact boolean")
    return value


def require_probability(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise SchemaError(f"{context} must be finite in [0, 1]")
    return number


def require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise SchemaError(
            f"{context} fields mismatch (missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)})"
        )


def probability_distribution(
    value: Mapping[str, Any], *, labels: Sequence[str], context: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(labels):
        raise SchemaError(f"{context} must contain exactly the registered labels")
    result = {
        label: require_probability(value[label], context=f"{context}.{label}")
        for label in labels
    }
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SchemaError(f"{context} must sum to one")
    return result


def selected_label(probabilities: Mapping[str, float], labels: Sequence[str]) -> str:
    """Stable argmax with the registered label order as the tie breaker."""

    return max(labels, key=lambda label: (float(probabilities[label]), -labels.index(label)))


def normalized_bbox(
    value: object, *, context: str, nullable: bool = True
) -> tuple[float, float, float, float] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SchemaError(f"{context} must be [x, y, width, height]")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise SchemaError(f"{context} must contain numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise SchemaError(f"{context} must be finite")
    x, y, width, height = result
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise SchemaError(f"{context} is outside the normalized image plane")
    return result  # type: ignore[return-value]


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return 0.0 if union <= 0.0 else intersection / union


def json_mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be a JSON object")
    result = dict(value)
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{context} is not finite JSON") from exc
    try:
        assert_oracle_blind_mapping(result, location=context)
    except CausalBoundaryError as exc:
        raise SchemaError(f"{context} contains sealed/oracle data") from exc
    return result


@dataclass(frozen=True, slots=True)
class CompanionBackendIdentity:
    pillar: str
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
    provider_module: str | None = None
    provider_qualname: str | None = None
    provider_id: str | None = None
    provider_version: str | None = None
    provider_policy_source: str | None = None
    provider_source_sha256: str | None = None
    provider_prompt_sha256: str | None = None
    memory_manifest_sha256: str | None = None
    embedding_provider_id: str | None = None
    embedding_provider_version: str | None = None
    embedding_provider_module: str | None = None
    embedding_provider_qualname: str | None = None
    embedding_provider_source_sha256: str | None = None
    selection_scope: str = "validation_only"
    frozen: bool = True
    evaluation_mode: bool = True
    schema_version: str = COMPANION_BACKEND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.pillar not in {"P1", "P3", "P4"}:
            raise SchemaError("companion backend pillar is unregistered")
        for name in (
            "predictor_id",
            "predictor_version",
            "predictor_module",
            "predictor_qualname",
            "factory_entrypoint",
            "model_id",
            "model_revision",
        ):
            require_text(getattr(self, name), context=f"backend.{name}")
        module_name, separator, attribute = self.factory_entrypoint.partition(":")
        if separator != ":" or not module_name or not attribute or "." in attribute:
            raise SchemaError("backend.factory_entrypoint must be exact module:attribute")
        for name in (
            "checkpoint_sha256",
            "resolved_config_sha256",
            "resolved_config_record_sha256",
            "processor_contract_sha256",
            "factory_source_sha256",
            "predictor_source_sha256",
        ):
            require_sha256(getattr(self, name), context=f"backend.{name}")
        require_commit(self.repository_commit)
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise SchemaError("backend.model_seed must be a nonnegative exact integer")
        if self.selection_scope != "validation_only":
            raise SchemaError("companion diagnostics require validation-only selection")
        if self.frozen is not True or self.evaluation_mode is not True:
            raise SchemaError("companion backend must be frozen in evaluation mode")
        if self.schema_version != COMPANION_BACKEND_SCHEMA_VERSION:
            raise SchemaError("unregistered companion backend schema")
        provider_values = (
            self.provider_module,
            self.provider_qualname,
            self.provider_id,
            self.provider_version,
            self.provider_policy_source,
            self.provider_source_sha256,
            self.provider_prompt_sha256,
        )
        if self.pillar == "P3":
            if any(value is None for value in provider_values):
                raise SchemaError("P3 backend requires the complete parameter-provider identity")
            require_text(self.provider_module, context="backend.provider_module")
            require_text(self.provider_qualname, context="backend.provider_qualname")
            require_text(self.provider_id, context="backend.provider_id")
            require_text(self.provider_version, context="backend.provider_version")
            require_text(
                self.provider_policy_source, context="backend.provider_policy_source"
            )
            require_sha256(
                self.provider_source_sha256, context="backend.provider_source_sha256"
            )
            require_sha256(
                self.provider_prompt_sha256, context="backend.provider_prompt_sha256"
            )
        elif any(value is not None for value in provider_values):
            raise SchemaError("only P3 may carry a parameter-provider identity")
        embedding_values = (
            self.embedding_provider_id,
            self.embedding_provider_version,
            self.embedding_provider_module,
            self.embedding_provider_qualname,
            self.embedding_provider_source_sha256,
        )
        if self.pillar == "P4":
            require_sha256(
                self.memory_manifest_sha256, context="backend.memory_manifest_sha256"
            )
            if any(value is None for value in embedding_values):
                raise SchemaError("P4 backend requires the complete embedding identity")
            for name in (
                "embedding_provider_id",
                "embedding_provider_version",
                "embedding_provider_module",
                "embedding_provider_qualname",
            ):
                require_text(getattr(self, name), context=f"backend.{name}")
            require_sha256(
                self.embedding_provider_source_sha256,
                context="backend.embedding_provider_source_sha256",
            )
        elif self.memory_manifest_sha256 is not None:
            raise SchemaError("only P4 may carry a memory-manifest identity")
        elif any(value is not None for value in embedding_values):
            raise SchemaError("only P4 may carry an embedding-provider identity")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CompanionBackendIdentity":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="backend identity")
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
        require_text(self.artifact_id, context="image.artifact_id")
        safe_relative_path(self.relative_path)
        require_sha256(self.sha256, context="image.sha256")
        if type(self.width) is not int or type(self.height) is not int:
            raise SchemaError("image dimensions must be exact integers")
        if self.width <= 0 or self.height <= 0:
            raise SchemaError("image dimensions must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ImageArtifact":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="image artifact")
        return cls(**dict(value))

    def resolve(self, root: str | Path) -> "ResolvedImage":
        base = Path(root).resolve()
        unresolved = base / safe_relative_path(self.relative_path)
        try:
            metadata = unresolved.lstat()
        except OSError as exc:
            raise CompanionDiagnosticError(
                f"diagnostic image is absent: {self.relative_path}"
            ) from exc
        if unresolved.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise CompanionDiagnosticError("diagnostic image must be a regular non-symlink file")
        if metadata.st_nlink != 1 or metadata.st_mode & 0o222:
            raise CompanionDiagnosticError(
                "diagnostic image must be single-link and read-only"
            )
        path = unresolved.resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise CompanionDiagnosticError("diagnostic image escapes evidence root") from exc
        if sha256_file(path) != self.sha256:
            raise CompanionDiagnosticError("diagnostic image hash mismatch")
        return ResolvedImage(
            artifact_id=self.artifact_id,
            path=path,
            sha256=self.sha256,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, slots=True)
class ResolvedImage:
    artifact_id: str
    path: Path
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ObservableTextState:
    task_text: str
    domain: str
    current_url: str
    title: str
    page_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("task_text", "domain", "current_url"):
            require_text(getattr(self, name), context=f"text_state.{name}")
        if not isinstance(self.title, str):
            raise SchemaError("text_state.title must be text")
        if (urlparse(self.current_url).hostname or "") != self.domain:
            raise SchemaError("text_state domain must equal the current URL hostname")
        object.__setattr__(
            self,
            "page_state",
            json_mapping(self.page_state, context="text_state.page_state"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservableTextState":
        require_exact_keys(value, set(cls.__dataclass_fields__), context="observable text state")
        return cls(**dict(value))


def validate_input_envelope(
    *,
    pillar: str,
    source_partition: str,
    registered_seed: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    actions_frozen_before_diagnostics: bool,
    runtime_outputs_consumed: bool,
    affects_primary_table2: bool,
    locked_test_rows_read: int,
    evidence_role: str,
    canonical_role: str,
    paper_table_status: str,
    backend_identity: CompanionBackendIdentity,
) -> None:
    if pillar != backend_identity.pillar:
        raise SchemaError("input pillar/backend pillar mismatch")
    if source_partition not in ALLOWED_SOURCE_PARTITIONS:
        raise SchemaError("diagnostic source partition is unregistered or locked")
    if type(registered_seed) is not int or registered_seed < 0:
        raise SchemaError("registered_seed must be a nonnegative exact integer")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise SchemaError("bootstrap_samples must be a positive exact integer")
    if bootstrap_confidence != BOOTSTRAP_CONFIDENCE:
        raise SchemaError("diagnostic confidence must remain exactly 0.95")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise SchemaError("bootstrap_seed must be a nonnegative exact integer")
    if actions_frozen_before_diagnostics is not True:
        raise SchemaError("diagnostics may run only after source actions are frozen")
    if runtime_outputs_consumed is not False:
        raise SchemaError("diagnostic outputs may not feed into runtime")
    if affects_primary_table2 is not False:
        raise SchemaError("companion diagnostics may not alter primary Table 2")
    if type(locked_test_rows_read) is not int or locked_test_rows_read != 0:
        raise SchemaError("companion diagnostics require zero locked-test reads")
    if evidence_role not in {canonical_role, ENGINEERING_EVIDENCE_ROLE}:
        raise SchemaError("diagnostic evidence role is unregistered")
    if paper_table_status != PAPER_TABLE_STATUS:
        raise SchemaError("companion diagnostic must keep paper Table 2 N/R")


class DiagnosticPredictor(Protocol):
    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity: ...


def predictor_source(predictor: DiagnosticPredictor) -> tuple[Path, str]:
    target = type(predictor)
    try:
        source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    except (OSError, TypeError) as exc:
        raise CompanionDiagnosticError("diagnostic predictor source is unresolved") from exc
    if not source_name:
        raise CompanionDiagnosticError("diagnostic predictor source is unresolved")
    source = Path(source_name).absolute()
    if source.is_symlink() or not source.is_file():
        raise CompanionDiagnosticError(
            "diagnostic predictor source must be a non-symlink regular file"
        )
    source = source.resolve()
    digest = sha256_file(source)
    if digest != predictor.diagnostic_identity.predictor_source_sha256:
        raise CompanionDiagnosticError("diagnostic predictor source hash mismatch")
    return source, digest


def repository_attestation(
    repository_root: str | Path,
    *,
    expected_commit: str,
    source_path: str | Path,
    factory_entrypoint: str,
    factory_source_sha256: str,
) -> dict[str, Any]:
    raw_root = Path(repository_root).absolute()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise CompanionDiagnosticError(
            "diagnostic repository root must be a non-symlink directory"
        )
    root = raw_root.resolve()
    expected = require_commit(expected_commit)
    try:
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CompanionDiagnosticError("diagnostic requires an accessible Git checkout") from exc
    if top != root or commit != expected or status.strip():
        raise CompanionDiagnosticError(
            "canonical diagnostic requires the exact clean registered repository commit"
        )

    def committed_source(candidate: Path, *, label: str, expected_sha256: str) -> str:
        source = candidate.resolve()
        try:
            relative = source.relative_to(root).as_posix()
            tracked = subprocess.check_output(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            committed = subprocess.check_output(
                ["git", "show", f"{expected}:{relative}"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            raise CompanionDiagnosticError(
                f"diagnostic {label} source is not tracked by the registered commit"
            ) from exc
        actual_sha256 = sha256_file(source)
        if (
            tracked != relative
            or sha256_bytes(committed) != actual_sha256
            or actual_sha256 != expected_sha256
        ):
            raise CompanionDiagnosticError(
                f"diagnostic {label} source differs from registered commit bytes"
            )
        return relative

    predictor_source = Path(source_path).resolve()
    predictor_relative = committed_source(
        predictor_source,
        label="predictor",
        expected_sha256=sha256_file(predictor_source),
    )
    module_name, _, _ = factory_entrypoint.partition(":")
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise CompanionDiagnosticError("diagnostic factory source is unresolved")
    factory_source = Path(spec.origin).absolute()
    if factory_source.is_symlink() or not factory_source.is_file():
        raise CompanionDiagnosticError("diagnostic factory source must be a regular file")
    factory_relative = committed_source(
        factory_source,
        label="factory",
        expected_sha256=require_sha256(
            factory_source_sha256, context="factory source SHA-256"
        ),
    )
    return {
        "repository_commit": commit,
        "git_clean": True,
        "git_status_porcelain_sha256": sha256_bytes(status.encode("utf-8")),
        "predictor_source_path": predictor_relative,
        "predictor_source_sha256": sha256_file(predictor_source),
        "factory_source_path": factory_relative,
        "factory_source_sha256": sha256_file(factory_source),
    }


def verify_predictor(
    predictor: DiagnosticPredictor,
    expected: CompanionBackendIdentity,
) -> tuple[Path, str]:
    if type(predictor.diagnostic_identity) is not CompanionBackendIdentity:
        raise CompanionDiagnosticError("predictor returned an invalid backend identity")
    if predictor.diagnostic_identity != expected:
        raise CompanionDiagnosticError("predictor backend identity differs from frozen input")
    source, digest = predictor_source(predictor)
    if type(predictor).__module__ != expected.predictor_module or (
        type(predictor).__qualname__ != expected.predictor_qualname
    ):
        raise CompanionDiagnosticError("predictor class identity differs from frozen input")
    return source, digest


def engineering_source_attestation(
    *, source_path: Path, source_sha256: str
) -> dict[str, Any]:
    return {
        "repository_commit": None,
        "git_clean": False,
        "git_status_porcelain_sha256": None,
        "predictor_source_path": source_path.name,
        "predictor_source_sha256": source_sha256,
        "status": "ENGINEERING_UNATTESTED_UNPROMOTABLE",
    }


def rehash_images(images: Iterable[ResolvedImage]) -> None:
    for image in images:
        if sha256_file(image.path) != image.sha256:
            raise CompanionDiagnosticError("diagnostic image changed during inference")


def diagnostic_statistic(
    contributions: Iterable[Mapping[str, Any]],
    *,
    metric_kind: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    contribution_unit: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in contributions]
    if metric_kind not in {"rate", "mean"}:
        raise ValueError("diagnostic metric kind is unregistered")
    for row in rows:
        require_text(row.get("task_id"), context="diagnostic statistic task_id")
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError("diagnostic statistic contribution must be numeric")
        if not math.isfinite(float(value)):
            raise SchemaError("diagnostic statistic contribution must be finite")
        if metric_kind == "rate" and float(value) not in {0.0, 1.0}:
            raise SchemaError("diagnostic rate contribution must be binary")
    values = [float(row["value"]) for row in rows]
    clusters = len({str(row["task_id"]) for row in rows})
    total = sum(values)
    estimate = None if not rows else total / len(rows)
    result: dict[str, Any] = {
        "metric_kind": metric_kind,
        "estimate": estimate,
        "ci_low": None,
        "ci_high": None,
        "ci95_low": None,
        "ci95_high": None,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "interval_method": INTERVAL_METHOD,
        "inference_unit": "task_id",
        "cluster_unit": "task_id",
        "contribution_unit": contribution_unit,
        "n_task_clusters": clusters,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "valid_bootstrap_samples": 0,
        "interval_status": "NOT_APPLICABLE" if not rows else (
            "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS" if clusters < 2 else "ESTIMATED"
        ),
        "reason": "no eligible diagnostic examples" if not rows else (
            "fewer than two unique task clusters" if clusters < 2 else None
        ),
    }
    if metric_kind == "rate":
        result.update({"numerator": int(total), "denominator": len(rows)})
    else:
        result.update({"total": total, "count": len(rows)})
    if rows and clusters >= 2:
        bootstrap = cluster_bootstrap(
            rows,
            value=lambda row: float(row["value"]),
            cluster_key="task_id",
            samples=bootstrap_samples,
            confidence=BOOTSTRAP_CONFIDENCE,
            seed=bootstrap_seed,
        )
        if not math.isclose(
            float(bootstrap["estimate"]), float(estimate), rel_tol=0.0, abs_tol=1e-12
        ):
            raise CompanionDiagnosticError("clustered diagnostic point estimate changed")
        result.update(
            {
                "ci_low": bootstrap["ci_low"],
                "ci_high": bootstrap["ci_high"],
                "ci95_low": bootstrap["ci_low"],
                "ci95_high": bootstrap["ci_high"],
                "valid_bootstrap_samples": bootstrap_samples,
            }
        )
    return result


def metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    metric_kind: str,
    value: Callable[[Mapping[str, Any]], float | None],
    bootstrap_samples: int,
    bootstrap_seed: int,
    contribution_unit: str,
) -> dict[str, Any]:
    contributions = [
        {"task_id": str(row["task_id"]), "value": contribution}
        for row in rows
        if (contribution := value(row)) is not None
    ]
    return diagnostic_statistic(
        contributions,
        metric_kind=metric_kind,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=stable_int_seed(bootstrap_seed, name),
        contribution_unit=contribution_unit,
    )


def load_unique_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise CompanionDiagnosticError("diagnostic JSON is absent") from exc
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise CompanionDiagnosticError("diagnostic JSON must be a regular non-symlink file")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise CompanionDiagnosticError(f"diagnostic JSON repeats key {key!r}")
            output[key] = item
        return output

    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompanionDiagnosticError("diagnostic JSON is invalid") from exc
    if not isinstance(value, dict):
        raise CompanionDiagnosticError("diagnostic JSON must contain one object")
    return value


def write_package(
    output_dir: str | Path,
    *,
    pillar: str,
    diagnostic: Any,
    report: Mapping[str, Any],
    input_filename: str,
    report_filename: str,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise CompanionDiagnosticError("diagnostic output path is not a directory")
        if any(destination.iterdir()):
            raise CompanionDiagnosticError("diagnostic output directory is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    input_path = atomic_write_json(destination / input_filename, asdict(diagnostic), mode=0o444)
    report_path = atomic_write_json(destination / report_filename, report, mode=0o444)
    manifest = {
        "schema_version": COMPANION_PACKAGE_SCHEMA_VERSION,
        "pillar": pillar,
        "evidence_role": report["evidence_role"],
        "promotion_status": report["promotion_status"],
        "paper_table_status": PAPER_TABLE_STATUS,
        "affects_primary_table2": False,
        "files": {
            input_filename: {
                "sha256": sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
            },
            report_filename: {
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        },
        "input_manifest_sha256": report["input_manifest_sha256"],
        "report_sha256": sha256_file(report_path),
    }
    manifest_path = atomic_write_json(
        destination / "package_manifest.json", manifest, mode=0o444
    )
    return {**manifest, "package_manifest_sha256": sha256_file(manifest_path)}


def validate_package(
    output_dir: str | Path,
    *,
    pillar: str,
    input_filename: str,
    report_filename: str,
    load_input: Callable[[Mapping[str, Any]], Any],
    validate_report: Callable[[Mapping[str, Any]], None],
) -> dict[str, Any]:
    root = Path(output_dir)
    expected_files = {input_filename, report_filename, "package_manifest.json"}
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected_files:
        raise CompanionDiagnosticError("diagnostic package file set changed")
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise CompanionDiagnosticError("diagnostic package contains a non-file entry")
        if path.stat().st_mode & 0o222:
            raise CompanionDiagnosticError("diagnostic package files must be read-only")
    manifest = load_unique_json(root / "package_manifest.json")
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "pillar",
            "evidence_role",
            "promotion_status",
            "paper_table_status",
            "affects_primary_table2",
            "files",
            "input_manifest_sha256",
            "report_sha256",
        },
        context="companion package manifest",
    )
    if (
        manifest["schema_version"] != COMPANION_PACKAGE_SCHEMA_VERSION
        or manifest["pillar"] != pillar
        or manifest["paper_table_status"] != PAPER_TABLE_STATUS
        or manifest["affects_primary_table2"] is not False
    ):
        raise CompanionDiagnosticError("diagnostic package boundary changed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {input_filename, report_filename}:
        raise CompanionDiagnosticError("diagnostic package inventory changed")
    for name in (input_filename, report_filename):
        row = files[name]
        path = root / name
        if not isinstance(row, Mapping) or set(row) != {"sha256", "size_bytes"}:
            raise CompanionDiagnosticError("diagnostic package file descriptor changed")
        if row != {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}:
            raise CompanionDiagnosticError(f"diagnostic package {name} hash mismatch")
    diagnostic_mapping = load_unique_json(root / input_filename)
    diagnostic = load_input(diagnostic_mapping)
    report = load_unique_json(root / report_filename)
    validate_report(report)
    if sha256_json(asdict(diagnostic)) != report.get("input_manifest_sha256"):
        raise CompanionDiagnosticError("diagnostic package input/report binding changed")
    expected_metadata = {
        "pillar": pillar,
        "diagnostic_id": diagnostic.diagnostic_id,
        "campaign_id": diagnostic.campaign_id,
        "source_partition": diagnostic.source_partition,
        "evidence_role": diagnostic.evidence_role,
        "paper_table_status": diagnostic.paper_table_status,
        "backend_identity": diagnostic.backend_identity.to_dict(),
    }
    if any(report.get(field) != expected for field, expected in expected_metadata.items()):
        raise CompanionDiagnosticError("diagnostic package report/input metadata changed")
    if manifest["report_sha256"] != sha256_file(root / report_filename):
        raise CompanionDiagnosticError("diagnostic package report hash mismatch")
    if manifest["input_manifest_sha256"] != report["input_manifest_sha256"]:
        raise CompanionDiagnosticError("diagnostic package input identity changed")
    if manifest["evidence_role"] != report["evidence_role"] or (
        manifest["promotion_status"] != report["promotion_status"]
    ):
        raise CompanionDiagnosticError("diagnostic package promotion identity changed")
    return {
        "status": "PASS",
        "pillar": pillar,
        "paper_table_status": PAPER_TABLE_STATUS,
        "package_manifest_sha256": sha256_file(root / "package_manifest.json"),
    }


def hash_directory_read_only(root: str | Path) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Hash a diagnostic store and reject links, special files, or writable bytes."""

    directory = Path(root)
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise CompanionDiagnosticError("diagnostic store is absent") from exc
    if directory.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise CompanionDiagnosticError("diagnostic store must be a non-symlink directory")
    if metadata.st_mode & 0o222:
        raise CompanionDiagnosticError("diagnostic store directory must be read-only")
    directory = directory.resolve()
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise CompanionDiagnosticError("diagnostic store contains a non-regular entry")
        if info.st_mode & 0o222:
            raise CompanionDiagnosticError("diagnostic store contains writable files")
        relative = path.relative_to(directory).as_posix()
        relative_bytes = relative.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(payload)
        files.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    if not files:
        raise CompanionDiagnosticError("diagnostic store is empty")
    return digest.hexdigest(), tuple(files)


def canonical_pc01_base_error(identity: CompanionBackendIdentity) -> str | None:
    expected = {
        "model_id": PC01_MODEL_ID,
        "model_revision": PC01_MODEL_REVISION,
        "checkpoint_sha256": PC01_CHECKPOINT_SHA256,
        "resolved_config_sha256": PC01_CONFIG_SHA256,
        "resolved_config_record_sha256": PC01_CONFIG_SHA256,
        "processor_contract_sha256": PC01_PROCESSOR_SHA256,
        "model_seed": 42,
        "selection_scope": "validation_only",
        "frozen": True,
        "evaluation_mode": True,
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            return f"canonical PC-01 backend changed {field}"
    return None


def run_attestation(
    *,
    diagnostic: Any,
    predictor: DiagnosticPredictor,
    canonical_error: str | None,
    canonical_role: str,
    repository_root: str | Path | None,
    allow_uncommitted_engineering: bool,
) -> tuple[str, str, dict[str, Any], Path, str]:
    source_path, source_hash = verify_predictor(predictor, diagnostic.backend_identity)
    canonical = canonical_error is None
    provenance_blocked = bool(
        canonical_error
        and canonical_error.startswith(CANONICAL_INPUT_PROVENANCE_STATUS)
    )
    expected_role = canonical_role if canonical else ENGINEERING_EVIDENCE_ROLE
    if diagnostic.evidence_role != expected_role:
        raise CompanionDiagnosticError(
            "diagnostic evidence role is inconsistent with backend promotion status"
        )
    if canonical or provenance_blocked:
        if repository_root is None:
            raise CompanionDiagnosticError(
                "exact PC-01 diagnostics require clean repository verification"
            )
        attestation = repository_attestation(
            repository_root,
            expected_commit=diagnostic.backend_identity.repository_commit,
            source_path=source_path,
            factory_entrypoint=diagnostic.backend_identity.factory_entrypoint,
            factory_source_sha256=diagnostic.backend_identity.factory_source_sha256,
        )
        promotion = (
            "CANONICAL_PC01_COMPANION_EVIDENCE"
            if canonical
            else INPUT_PROVENANCE_BLOCKED_PROMOTION
        )
    else:
        if not allow_uncommitted_engineering:
            raise CompanionDiagnosticError(
                "generic diagnostic requires explicit unpromotable engineering mode"
            )
        attestation = engineering_source_attestation(
            source_path=source_path, source_sha256=source_hash
        )
        promotion = "UNPROMOTABLE"
    if (canonical or provenance_blocked) and (
        diagnostic.registered_seed != diagnostic.backend_identity.model_seed
        or diagnostic.bootstrap_samples != BOOTSTRAP_SAMPLES
        or diagnostic.bootstrap_confidence != BOOTSTRAP_CONFIDENCE
        or diagnostic.bootstrap_seed != BOOTSTRAP_SEED
    ):
        raise CompanionDiagnosticError(
            "canonical PC-01 companion evidence changed seed/bootstrap registration"
        )
    return expected_role, promotion, attestation, source_path, source_hash


def validate_report_envelope(
    report: Mapping[str, Any],
    *,
    pillar: str,
    report_schema: str,
    canonical_role: str,
    canonical_error: str | None,
    backend_identity: CompanionBackendIdentity,
) -> None:
    if report.get("schema_version") != report_schema or report.get("pillar") != pillar:
        raise SchemaError("companion diagnostic report schema/pillar changed")
    if report.get("paper_table_status") != PAPER_TABLE_STATUS:
        raise SchemaError("companion diagnostic paper_table_status changed")
    if report.get("affects_primary_table2") is not False:
        raise SchemaError("companion diagnostic cannot affect primary Table 2")
    if type(report.get("browser_actions_executed")) is not int or report.get(
        "browser_actions_executed"
    ) != 0:
        raise SchemaError("offline companion diagnostic cannot execute browser actions")
    if type(report.get("memory_writes_executed")) is not int or report.get(
        "memory_writes_executed"
    ) != 0:
        raise SchemaError("offline companion diagnostic cannot write memory")
    role = report.get("evidence_role")
    promotion = report.get("promotion_status")
    if role == canonical_role:
        if canonical_error is not None or promotion != "CANONICAL_PC01_COMPANION_EVIDENCE":
            raise SchemaError("companion evidence is not canonical PC-01")
        expected_fields = {
            "repository_commit",
            "git_clean",
            "git_status_porcelain_sha256",
            "predictor_source_path",
            "predictor_source_sha256",
            "factory_source_path",
            "factory_source_sha256",
        }
        attestation = report.get("source_attestation")
        if not isinstance(attestation, Mapping) or set(attestation) != expected_fields:
            raise SchemaError("canonical companion source attestation changed")
        if (
            attestation["repository_commit"] != backend_identity.repository_commit
            or attestation["git_clean"] is not True
            or attestation["git_status_porcelain_sha256"] != sha256_bytes(b"")
            or attestation["predictor_source_sha256"]
            != backend_identity.predictor_source_sha256
            or attestation["factory_source_sha256"]
            != backend_identity.factory_source_sha256
        ):
            raise SchemaError("canonical companion source attestation is inconsistent")
        safe_relative_path(str(attestation["predictor_source_path"]))
        safe_relative_path(str(attestation["factory_source_path"]))
    elif role == ENGINEERING_EVIDENCE_ROLE:
        attestation = report.get("source_attestation")
        provenance_blocked = bool(
            canonical_error
            and canonical_error.startswith(CANONICAL_INPUT_PROVENANCE_STATUS)
        )
        if promotion == INPUT_PROVENANCE_BLOCKED_PROMOTION:
            expected_fields = {
                "repository_commit",
                "git_clean",
                "git_status_porcelain_sha256",
                "predictor_source_path",
                "predictor_source_sha256",
                "factory_source_path",
                "factory_source_sha256",
            }
            if not provenance_blocked:
                raise SchemaError(
                    "input-provenance-blocked evidence lacks an exact PC-01 backend"
                )
            if not isinstance(attestation, Mapping) or set(attestation) != expected_fields:
                raise SchemaError("blocked companion source attestation changed")
            if (
                attestation["repository_commit"] != backend_identity.repository_commit
                or attestation["git_clean"] is not True
                or attestation["git_status_porcelain_sha256"] != sha256_bytes(b"")
                or attestation["predictor_source_sha256"]
                != backend_identity.predictor_source_sha256
                or attestation["factory_source_sha256"]
                != backend_identity.factory_source_sha256
            ):
                raise SchemaError("blocked companion source attestation is inconsistent")
            safe_relative_path(str(attestation["predictor_source_path"]))
            safe_relative_path(str(attestation["factory_source_path"]))
        else:
            if promotion != "UNPROMOTABLE":
                raise SchemaError("generic companion evidence must remain unpromotable")
            expected_fields = {
                "repository_commit",
                "git_clean",
                "git_status_porcelain_sha256",
                "predictor_source_path",
                "predictor_source_sha256",
                "status",
            }
            if not isinstance(attestation, Mapping) or set(attestation) != expected_fields:
                raise SchemaError("engineering companion source attestation changed")
            if (
                attestation["repository_commit"] is not None
                or attestation["git_clean"] is not False
                or attestation["git_status_porcelain_sha256"] is not None
                or attestation["status"] != "ENGINEERING_UNATTESTED_UNPROMOTABLE"
                or attestation["predictor_source_sha256"]
                != backend_identity.predictor_source_sha256
            ):
                raise SchemaError("engineering companion evidence was relabelled")
            require_text(
                attestation["predictor_source_path"], context="predictor source path"
            )
    else:
        raise SchemaError("companion diagnostic evidence role changed")


def validate_diagnostic_statistic(value: Mapping[str, Any]) -> None:
    """Validate the complete denominator/CI contract of one reported metric."""

    kind = value.get("metric_kind")
    base = {
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
    expected = base | ({"numerator", "denominator"} if kind == "rate" else {
        "total", "count"
    })
    if kind not in {"rate", "mean"} or set(value) != expected:
        raise SchemaError("diagnostic statistic fields/kind changed")
    for field in (
        "n_task_clusters",
        "bootstrap_samples",
        "bootstrap_seed",
        "valid_bootstrap_samples",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise SchemaError(f"diagnostic statistic {field} is invalid")
    if value["bootstrap_samples"] <= 0 or value["confidence"] != BOOTSTRAP_CONFIDENCE:
        raise SchemaError("diagnostic statistic bootstrap registration changed")
    if (
        value["interval_method"] != INTERVAL_METHOD
        or value["inference_unit"] != "task_id"
        or value["cluster_unit"] != "task_id"
    ):
        raise SchemaError("diagnostic statistic inference unit/method changed")
    require_text(value["contribution_unit"], context="diagnostic contribution unit")
    count_field = "denominator" if kind == "rate" else "count"
    count = value[count_field]
    if type(count) is not int or count < 0 or value["n_task_clusters"] > count:
        raise SchemaError("diagnostic statistic count/cluster count is invalid")
    total_field = "numerator" if kind == "rate" else "total"
    total = value[total_field]
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not math.isfinite(
        float(total)
    ):
        raise SchemaError("diagnostic statistic numerator/total is invalid")
    if kind == "rate" and (type(total) is not int or not 0 <= total <= count):
        raise SchemaError("diagnostic rate numerator is invalid")
    expected_estimate = None if count == 0 else float(total) / count
    estimate = value["estimate"]
    if expected_estimate is None:
        if estimate is not None:
            raise SchemaError("empty diagnostic statistic has an estimate")
    elif isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or not math.isclose(
        float(estimate), expected_estimate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SchemaError("diagnostic statistic estimate disagrees with its evidence")
    ci_fields = ("ci_low", "ci_high", "ci95_low", "ci95_high")
    status = value["interval_status"]
    if count == 0:
        expected_status = "NOT_APPLICABLE"
        expected_reason = "no eligible diagnostic examples"
    elif value["n_task_clusters"] < 2:
        expected_status = "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS"
        expected_reason = "fewer than two unique task clusters"
    else:
        expected_status = "ESTIMATED"
        expected_reason = None
    if status != expected_status or value["reason"] != expected_reason:
        raise SchemaError("diagnostic statistic interval status/reason changed")
    if status == "ESTIMATED":
        for field in ci_fields:
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(
                float(item)
            ):
                raise SchemaError("diagnostic statistic confidence interval is invalid")
        if (
            value["ci_low"] != value["ci95_low"]
            or value["ci_high"] != value["ci95_high"]
            or float(value["ci_low"]) > float(value["ci_high"])
            or value["valid_bootstrap_samples"] != value["bootstrap_samples"]
        ):
            raise SchemaError("diagnostic statistic confidence interval changed")
    elif any(value[field] is not None for field in ci_fields) or value[
        "valid_bootstrap_samples"
    ] != 0:
        raise SchemaError("non-estimable diagnostic statistic invented an interval")


def validate_metric_tree(value: object) -> None:
    """Find and validate every statistic object nested in a summary."""

    if isinstance(value, Mapping):
        if "metric_kind" in value:
            validate_diagnostic_statistic(value)
            return
        for item in value.values():
            validate_metric_tree(item)
    elif isinstance(value, list):
        for item in value:
            validate_metric_tree(item)
