"""Dependency-lazy binding for a validation-selected frozen checkpoint.

This module intentionally does not import torch, transformers, or training
code.  A DGX/WebArena launcher supplies one backend factory after freezing the
checkpoint and processor manifests; the adapter verifies those boundaries
before the first inference call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Mapping

from web_agent.runtime.contracts import (
    JsonValue,
    PolicyObservation,
    PreActionDecision,
    RecoveryAssessment,
    RecoveryTransitionInput,
    RuntimeTaskView,
    TransitionAssessment,
    TransitionInput,
)
from web_agent.runtime.manifest import sha256_directory, sha256_file
from web_agent.runtime.memory_adapter import EmbeddingCallable
from web_agent.runtime.observation import (
    ProcessorParityContract,
    validate_processor_parity,
)
from web_agent.runtime.policy import (
    ActionPredictor,
    CallablePolicyAdapter,
    PolicyAdapter,
    PolicyKind,
    RecoveryPredictor,
    TransitionPredictor,
)


class SelectedCheckpointError(RuntimeError):
    """The selected checkpoint, manifest, or loaded backend is not admissible."""


@dataclass(frozen=True, slots=True)
class ValidationSelectedCheckpoint:
    manifest_id: str
    model_seed: int
    checkpoint_path: Path
    selected_checkpoint_sha256: str
    resolved_config_sha256: str
    processor_contract_sha256: str
    validation_rows_read: int
    checkpoint_selection: str = "validation_only"
    selection_scope: str = "validation_only"
    test_rows_read: int = 0
    locked_test_rows_read: int = 0

    def __post_init__(self) -> None:
        if not self.manifest_id.strip() or self.model_seed < 0:
            raise ValueError("selected checkpoint identity/seed are invalid")
        if self.checkpoint_selection != "validation_only" or self.selection_scope != "validation_only":
            raise SelectedCheckpointError("checkpoint was not selected on validation only")
        if self.validation_rows_read <= 0:
            raise SelectedCheckpointError("validation-only selection needs positive validation evidence")
        if self.test_rows_read != 0 or self.locked_test_rows_read != 0:
            raise SelectedCheckpointError("checkpoint selection reports test/locked reads")
        for name in (
            "selected_checkpoint_sha256",
            "resolved_config_sha256",
            "processor_contract_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be SHA-256")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, JsonValue],
        *,
        checkpoint_path: str | Path,
    ) -> "ValidationSelectedCheckpoint":
        required = {
            "model_seed",
            "selected_checkpoint_sha256",
            "resolved_config_sha256",
            "processor_contract_sha256",
            "selection_scope",
            "checkpoint_selection",
            "validation_rows_read",
            "test_rows_read",
            "locked_test_rows_read",
        }
        missing = required - set(value)
        if missing:
            raise SelectedCheckpointError(
                f"model selection manifest is missing {sorted(missing)}"
            )
        integer_fields = (
            "model_seed",
            "validation_rows_read",
            "test_rows_read",
            "locked_test_rows_read",
        )
        if any(type(value[name]) is not int for name in integer_fields):
            raise SelectedCheckpointError("model selection row counts/seed must be exact integers")
        return cls(
            manifest_id=str(value.get("manifest_id") or f"seed-{value['model_seed']}"),
            model_seed=int(value["model_seed"]),
            checkpoint_path=Path(checkpoint_path),
            selected_checkpoint_sha256=str(value["selected_checkpoint_sha256"]),
            resolved_config_sha256=str(value["resolved_config_sha256"]),
            processor_contract_sha256=str(value["processor_contract_sha256"]),
            validation_rows_read=int(value["validation_rows_read"]),
            checkpoint_selection=str(value["checkpoint_selection"]),
            selection_scope=str(value["selection_scope"]),
            test_rows_read=int(value["test_rows_read"]),
            locked_test_rows_read=int(value["locked_test_rows_read"]),
        )

    def verify_checkpoint(self) -> None:
        if self.checkpoint_path.is_file():
            actual = sha256_file(self.checkpoint_path)
        elif self.checkpoint_path.is_dir():
            actual = sha256_directory(self.checkpoint_path)
        else:
            raise FileNotFoundError(f"selected checkpoint is absent: {self.checkpoint_path}")
        if actual != self.selected_checkpoint_sha256:
            raise SelectedCheckpointError(
                "selected checkpoint bytes differ from the validation-selected manifest"
            )


@dataclass(frozen=True, slots=True)
class ValidationSelectedBackbone:
    """Unadapted base model paired to the validation-selected backbone family."""

    manifest_id: str
    backbone_id: str
    backbone_revision: str
    backbone_path: Path
    backbone_sha256: str
    resolved_config_sha256: str
    processor_contract_sha256: str
    base_prompt_sha256: str
    parser_id: str
    parser_version: str
    validation_rows_read: int
    selection_scope: str = "validation_only"
    test_rows_read: int = 0
    locked_test_rows_read: int = 0

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "backbone_id",
            "backbone_revision",
            "parser_id",
            "parser_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"selected base {name} is required")
        if self.selection_scope != "validation_only" or self.validation_rows_read <= 0:
            raise SelectedCheckpointError(
                "E0 backbone must be tied to positive validation-only selection evidence"
            )
        if self.test_rows_read != 0 or self.locked_test_rows_read != 0:
            raise SelectedCheckpointError("E0 backbone selection reports test/locked reads")
        for name in (
            "backbone_sha256",
            "resolved_config_sha256",
            "processor_contract_sha256",
            "base_prompt_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"selected base {name} must be SHA-256")

    def verify_backbone(self) -> None:
        if self.backbone_path.is_file():
            actual = sha256_file(self.backbone_path)
        elif self.backbone_path.is_dir():
            actual = sha256_directory(self.backbone_path)
        else:
            raise FileNotFoundError(f"selected unadapted backbone is absent: {self.backbone_path}")
        if actual != self.backbone_sha256:
            raise SelectedCheckpointError(
                "unadapted E0 backbone bytes differ from the selected-backbone manifest"
            )


@dataclass(frozen=True, slots=True)
class LoadedSelectedBackboneBackend:
    backbone_id: str
    backbone_revision: str
    backbone_sha256: str
    resolved_config_sha256: str
    processor_contract: ProcessorParityContract
    base_prompt_sha256: str
    parser_id: str
    parser_version: str
    action_predictor: ActionPredictor
    frozen: bool = True
    training: bool = False
    adaptation_loaded: bool = False
    task_heads_loaded: bool = False

    def __post_init__(self) -> None:
        for name in (
            "backbone_id",
            "backbone_revision",
            "parser_id",
            "parser_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"loaded selected base {name} is required")
        for name in (
            "backbone_sha256",
            "resolved_config_sha256",
            "base_prompt_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"loaded selected base {name} must be SHA-256")
        if (
            not self.frozen
            or self.training
            or self.adaptation_loaded
            or self.task_heads_loaded
        ):
            raise SelectedCheckpointError(
                "E0 must be frozen, eval-mode, unadapted, and free of trained task heads"
            )


SelectedBackboneBackendFactory = Callable[
    [ValidationSelectedBackbone],
    LoadedSelectedBackboneBackend,
]


class SelectedBackbonePolicyAdapter(PolicyAdapter):
    """Lazy E0 adapter for the unadapted version of the selected backbone."""

    kind = PolicyKind.BASE
    checkpoint_sha256 = None

    def __init__(
        self,
        *,
        selection: ValidationSelectedBackbone,
        backend_factory: SelectedBackboneBackendFactory,
        runtime_processor_contract: ProcessorParityContract,
        policy_id: str = "selected-backbone-unadapted",
        policy_version: str = "v1",
    ) -> None:
        if not policy_id.strip() or not policy_version.strip():
            raise ValueError("E0 policy identity/version are required")
        if runtime_processor_contract.record_sha256 != selection.processor_contract_sha256:
            raise SelectedCheckpointError(
                "E0 runtime processor contract differs from selection manifest"
            )
        self.selection = selection
        self.backend_factory = backend_factory
        self.runtime_processor_contract = runtime_processor_contract
        self.policy_id = policy_id
        self.policy_version = policy_version
        self._delegate: CallablePolicyAdapter | None = None

    def _load(self) -> CallablePolicyAdapter:
        if self._delegate is not None:
            return self._delegate
        self.selection.verify_backbone()
        backend = self.backend_factory(self.selection)
        if not isinstance(backend, LoadedSelectedBackboneBackend):
            raise SelectedCheckpointError("E0 backend factory returned an invalid binding")
        exact = {
            "backbone_id": self.selection.backbone_id,
            "backbone_revision": self.selection.backbone_revision,
            "backbone_sha256": self.selection.backbone_sha256,
            "resolved_config_sha256": self.selection.resolved_config_sha256,
            "base_prompt_sha256": self.selection.base_prompt_sha256,
            "parser_id": self.selection.parser_id,
            "parser_version": self.selection.parser_version,
        }
        for name, expected in exact.items():
            if getattr(backend, name) != expected:
                raise SelectedCheckpointError(f"E0 backend {name} differs from manifest")
        validate_processor_parity(
            backend.processor_contract,
            self.runtime_processor_contract,
        )
        self._delegate = CallablePolicyAdapter(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            kind=PolicyKind.BASE,
            checkpoint_sha256=None,
            action_predictor=backend.action_predictor,
        )
        return self._delegate

    def predict_action(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        *,
        rng: random.Random,
    ) -> PreActionDecision:
        return self._load().predict_action(task, observation, rng=rng)

    def assess_transition(
        self,
        task: RuntimeTaskView,
        transition: TransitionInput,
        *,
        rng: random.Random,
    ) -> TransitionAssessment:
        del task, transition, rng
        raise SelectedCheckpointError("E0 has no trained post-action task head")

    def assess_recovery(
        self,
        task: RuntimeTaskView,
        transition: RecoveryTransitionInput,
        *,
        rng: random.Random,
    ) -> RecoveryAssessment:
        del task, transition, rng
        raise SelectedCheckpointError("E0 has no trained recovery task head")


@dataclass(frozen=True, slots=True)
class LoadedSelectedCheckpointBackend:
    """Callbacks exposed by one already-loaded, eval-mode inference backend."""

    checkpoint_sha256: str
    resolved_config_sha256: str
    processor_contract: ProcessorParityContract
    action_predictor: ActionPredictor
    transition_predictor: TransitionPredictor
    recovery_predictor: RecoveryPredictor
    memory_embedding: EmbeddingCallable
    frozen: bool = True
    training: bool = False

    def __post_init__(self) -> None:
        if not _is_sha256(self.checkpoint_sha256) or not _is_sha256(
            self.resolved_config_sha256
        ):
            raise ValueError("loaded backend checkpoint/config identity is invalid")
        if not self.frozen or self.training:
            raise SelectedCheckpointError("selected checkpoint backend must be frozen/eval-mode")
        if not callable(self.memory_embedding):
            raise TypeError(
                "selected checkpoint backend requires the exact P4 memory-embedding forward"
            )


SelectedCheckpointBackendFactory = Callable[
    [ValidationSelectedCheckpoint],
    LoadedSelectedCheckpointBackend,
]


class SelectedCheckpointPolicyAdapter(PolicyAdapter):
    """Lazy P1/P2/P3 policy bridge with selection and processor parity gates."""

    kind = PolicyKind.TRAINED

    def __init__(
        self,
        *,
        selection: ValidationSelectedCheckpoint,
        backend_factory: SelectedCheckpointBackendFactory,
        runtime_processor_contract: ProcessorParityContract,
        policy_id: str = "validation-selected-web-agent",
        policy_version: str = "v1",
    ) -> None:
        if not policy_id.strip() or not policy_version.strip():
            raise ValueError("selected checkpoint policy identity/version are required")
        if runtime_processor_contract.record_sha256 != selection.processor_contract_sha256:
            raise SelectedCheckpointError(
                "runtime processor contract hash differs from model selection manifest"
            )
        self.selection = selection
        self.backend_factory = backend_factory
        self.runtime_processor_contract = runtime_processor_contract
        self.policy_id = policy_id
        self.policy_version = policy_version
        self.checkpoint_sha256 = selection.selected_checkpoint_sha256
        self._delegate: CallablePolicyAdapter | None = None

    def _load(self) -> CallablePolicyAdapter:
        if self._delegate is not None:
            return self._delegate
        self.selection.verify_checkpoint()
        backend = self.backend_factory(self.selection)
        if not isinstance(backend, LoadedSelectedCheckpointBackend):
            raise SelectedCheckpointError("checkpoint factory returned an invalid backend")
        if backend.checkpoint_sha256 != self.selection.selected_checkpoint_sha256:
            raise SelectedCheckpointError("loaded backend cites another checkpoint")
        if backend.resolved_config_sha256 != self.selection.resolved_config_sha256:
            raise SelectedCheckpointError(
                "loaded backend resolved configuration differs from selection manifest"
            )
        validate_processor_parity(
            backend.processor_contract,
            self.runtime_processor_contract,
        )
        self._delegate = CallablePolicyAdapter(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            kind=PolicyKind.TRAINED,
            checkpoint_sha256=self.checkpoint_sha256,
            action_predictor=backend.action_predictor,
            transition_predictor=backend.transition_predictor,
            recovery_predictor=backend.recovery_predictor,
        )
        return self._delegate

    def predict_action(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        *,
        rng: random.Random,
    ) -> PreActionDecision:
        return self._load().predict_action(task, observation, rng=rng)

    def assess_transition(
        self,
        task: RuntimeTaskView,
        transition: TransitionInput,
        *,
        rng: random.Random,
    ) -> TransitionAssessment:
        return self._load().assess_transition(task, transition, rng=rng)

    def assess_recovery(
        self,
        task: RuntimeTaskView,
        transition: RecoveryTransitionInput,
        *,
        rng: random.Random,
    ) -> RecoveryAssessment:
        return self._load().assess_recovery(task, transition, rng=rng)


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
