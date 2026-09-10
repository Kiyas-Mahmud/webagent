"""Authenticated PC-01 adapter for the offline Pillar-2 diagnostics.

The adapter reuses the exact selected-checkpoint loader and causal processor
streams used by the live PC-01 runtime.  It adds no weights or heads and does
not expose an executor, recovery controller, verifier, or memory store.

A deployment-owned, source-attested zero-argument factory should call
``load_pc01_pillar2_diagnostic_predictor`` with its frozen artifact bindings.
That small wrapper is the factory entrypoint registered in the diagnostic
input manifest; this module supplies the canonical predictor implementation.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from web_agent.labels import (
    ACTION_TYPE_INV,
    EXECUTION_OUTCOME,
    FAILURE_TYPE_INV,
    RECOVERY_STRATEGY_INV,
)
from web_agent.runtime.checkpoint_inference import ValidationSelectedCheckpoint
from web_agent.eval.table2.pc01_artifacts import (
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_MODEL_SEED,
)
from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.qwen2vl_pc01 import (
    PC01RuntimeArtifacts,
    PC01RuntimeError,
    _RuntimeBatchBuilder,
    _SelectedCheckpointRuntime,
    _load_selected_model,
    _tensor_probabilities,
)

from .common import sha256_file
from .pillar2_diagnostics import (
    DiagnosticBackendIdentity,
    Pillar2DiagnosticError,
    PostDiagnosticPrediction,
    PostDiagnosticView,
    PreDiagnosticPrediction,
    PreDiagnosticView,
)


PC01_PILLAR2_PREDICTOR_ID = "pc01-pillar2-offline-diagnostic"
PC01_PILLAR2_PREDICTOR_VERSION = "v1"
PC01_EXPECTED_PROCESSOR_CONTRACT_SHA256 = (
    "b3c9f629a3c4ce6ab7bb71b27c7e0b560e3e21c51a19d1235fd4a507ba437797"
)
PC01_PILLAR2_PREDICTOR_MODULE = __name__
PC01_PILLAR2_PREDICTOR_QUALNAME = "PC01Pillar2DiagnosticPredictor"


def canonical_pc01_backend_error(
    identity: DiagnosticBackendIdentity,
    *,
    predictor: Any | None = None,
) -> str | None:
    """Return why an identity cannot be promoted to PC-01 companion evidence."""

    expected = {
        "predictor_id": PC01_PILLAR2_PREDICTOR_ID,
        "predictor_version": PC01_PILLAR2_PREDICTOR_VERSION,
        "predictor_module": PC01_PILLAR2_PREDICTOR_MODULE,
        "predictor_qualname": PC01_PILLAR2_PREDICTOR_QUALNAME,
        "model_id": PC01_MODEL_ID,
        "model_revision": PC01_MODEL_REVISION,
        "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "resolved_config_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "resolved_config_record_sha256": PC01_EXPECTED_CONFIG_SHA256,
        "processor_contract_sha256": PC01_EXPECTED_PROCESSOR_CONTRACT_SHA256,
        "model_seed": PC01_MODEL_SEED,
        "selection_scope": "validation_only",
        "frozen": True,
        "evaluation_mode": True,
        "predictor_source_sha256": pc01_pillar2_predictor_source_sha256(),
    }
    for field, expected_value in expected.items():
        if getattr(identity, field) != expected_value:
            return f"canonical PC-01 backend changed {field}"
    if predictor is not None and type(predictor) is not PC01Pillar2DiagnosticPredictor:
        return "canonical PC-01 companion evidence requires the exact predictor class"
    return None


def pc01_pillar2_predictor_source_sha256() -> str:
    """Return the exact canonical adapter source identity for registration."""

    return sha256_file(Path(__file__).resolve())


def _observation(
    view: PreDiagnosticView | PostDiagnosticView,
    *,
    image: Any,
    stage: str,
) -> PolicyObservation:
    if stage not in {"pre", "post"}:
        raise Pillar2DiagnosticError("PC-01 P2 diagnostic stage is unregistered")
    return PolicyObservation(
        task_id=view.task_id,
        goal=view.text_state.task_text,
        observation_id=(
            f"p2-{view.example_id}-{stage}-{image.sha256[:20]}"
        ),
        screenshot_sha256=image.sha256,
        screenshot_path=str(image.path),
        width=image.width,
        height=image.height,
        url=view.text_state.current_url,
        title=view.text_state.title,
        current_page_state=dict(view.text_state.page_state),
        causal_history=(),
    )


class PC01Pillar2DiagnosticPredictor:
    """Pure-inference diagnostic view over one authenticated PC-01 runtime."""

    def __init__(
        self,
        *,
        runtime: _SelectedCheckpointRuntime,
        identity: DiagnosticBackendIdentity,
    ) -> None:
        if identity.predictor_id != PC01_PILLAR2_PREDICTOR_ID or (
            identity.predictor_version != PC01_PILLAR2_PREDICTOR_VERSION
        ):
            raise Pillar2DiagnosticError("PC-01 P2 predictor identity is unregistered")
        canonical_error = canonical_pc01_backend_error(identity, predictor=self)
        if canonical_error is not None:
            raise Pillar2DiagnosticError(canonical_error)
        if identity.predictor_source_sha256 != pc01_pillar2_predictor_source_sha256():
            raise Pillar2DiagnosticError("PC-01 P2 predictor source identity mismatch")
        if runtime.checkpoint_sha256 != identity.checkpoint_sha256:
            raise Pillar2DiagnosticError("PC-01 P2 runtime cites another checkpoint")
        if (
            runtime.processor_contract.record_sha256
            != identity.processor_contract_sha256
        ):
            raise Pillar2DiagnosticError("PC-01 P2 runtime cites another processor")
        self._runtime = runtime
        self._identity = identity

    @property
    def diagnostic_identity(self) -> DiagnosticBackendIdentity:
        return self._identity

    def predict_pre(
        self, view: PreDiagnosticView, *, seed: int
    ) -> PreDiagnosticPrediction:
        task = RuntimeTaskView(
            task_id=view.task_id,
            goal=view.text_state.task_text,
        )
        decision = self._runtime.predict_action(
            task,
            _observation(view, image=view.image, stage="pre"),
            random.Random(seed),
        )
        return PreDiagnosticPrediction(
            action_probabilities=dict(decision.action_probabilities),
            bbox=decision.bbox,
            confidence_before=decision.confidence_before,
        )

    def predict_post(
        self, view: PostDiagnosticView, *, seed: int
    ) -> PostDiagnosticPrediction:
        del seed  # PC-01 is deterministic in evaluation mode.
        task = RuntimeTaskView(
            task_id=view.task_id,
            goal=view.text_state.task_text,
        )
        pre_observation = _observation(view, image=view.pre_image, stage="pre")
        post_observation = _observation(view, image=view.post_image, stage="post")
        pre = self._runtime.batch.stream(
            task=task,
            observations=(pre_observation,),
            phase="pre",
        )
        post = self._runtime.batch.stream(
            task=task,
            observations=(pre_observation, post_observation),
            phase="post",
            executed_action=view.executed_action or "",
            # The authenticated PC-01 training corpus has no action-value text.
            action_value="",
        )
        batch = {
            **_RuntimeBatchBuilder.prefix(pre, "pre_"),
            **_RuntimeBatchBuilder.prefix(post, "post_"),
        }
        with self._runtime.lock, self._runtime.torch.inference_mode():
            self._runtime._assert_eval()
            predictions = self._runtime.model(batch)
        outcome = _tensor_probabilities(
            self._runtime.torch, predictions["outcome"]
        )
        failure_types = _tensor_probabilities(
            self._runtime.torch, predictions["failure_type"]
        )
        recovery = _tensor_probabilities(
            self._runtime.torch, predictions["recovery"]
        )
        if "needs_recovery" not in predictions:
            raise PC01RuntimeError("PC-01 checkpoint omitted the needs-recovery head")
        needs_probability = float(
            self._runtime.torch.sigmoid(predictions["needs_recovery"].float())[0]
            .detach()
            .cpu()
            .item()
        )
        return PostDiagnosticPrediction(
            failure_probability=outcome[EXECUTION_OUTCOME["FAILURE"]],
            failure_type_probabilities={
                FAILURE_TYPE_INV[index]: probability
                for index, probability in enumerate(failure_types)
            },
            needs_recovery_probability=needs_probability,
            recovery_probabilities={
                RECOVERY_STRATEGY_INV[index]: probability
                for index, probability in enumerate(recovery)
            },
        )


def load_pc01_pillar2_diagnostic_predictor(
    *,
    selection: ValidationSelectedCheckpoint,
    artifacts: PC01RuntimeArtifacts,
    identity: DiagnosticBackendIdentity,
) -> PC01Pillar2DiagnosticPredictor:
    """Load the exact frozen PC-01 checkpoint for companion diagnostics only."""

    if identity.model_seed != selection.model_seed:
        raise Pillar2DiagnosticError("P2 identity/selection model seeds differ")
    if identity.checkpoint_sha256 != selection.selected_checkpoint_sha256:
        raise Pillar2DiagnosticError("P2 identity/selection checkpoints differ")
    if identity.resolved_config_sha256 != selection.resolved_config_sha256:
        raise Pillar2DiagnosticError("P2 identity/selection configurations differ")
    if identity.processor_contract_sha256 != selection.processor_contract_sha256:
        raise Pillar2DiagnosticError("P2 identity/selection processors differ")
    model, processor, torch, config, processor_contract = _load_selected_model(
        selection=selection,
        artifacts=artifacts,
    )
    runtime = _SelectedCheckpointRuntime(
        model=model,
        processor=processor,
        torch=torch,
        config=config,
        checkpoint_sha256=selection.selected_checkpoint_sha256,
        processor_contract=processor_contract,
    )
    return PC01Pillar2DiagnosticPredictor(runtime=runtime, identity=identity)
