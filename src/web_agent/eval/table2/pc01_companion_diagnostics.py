"""Exact PC-01 adapters for the offline P1, P3, and P4 diagnostics.

These adapters expose only frozen inference and read-only retrieval.  A
deployment-owned, source-attested zero-argument factory must load the required
external artifacts and return one of the exact classes below.  None of the
classes owns a browser, verifier, recovery controller, campaign runner, or
memory-write API.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

from web_agent.eval.table2.pc01_artifacts import PC01_MODEL_SEED
from web_agent.labels import (
    EXECUTION_OUTCOME,
    FAILURE_TYPE_INV,
    RECOVERY_STRATEGY_INV,
)
from web_agent.runtime.action_parameters import (
    CallableActionParameterProvider,
    DeterministicParameterProvider,
    HybridParameterProvider,
    HybridParameterResolutionError,
)
from web_agent.runtime.checkpoint_inference import ValidationSelectedCheckpoint
from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.qwen2vl_pc01 import (
    PC01RuntimeArtifacts,
    PC01RuntimeError,
    _RuntimeBatchBuilder,
    _SelectedCheckpointRuntime,
    _load_selected_model,
    _tensor_probabilities,
)

from .common import sha256_file, sha256_json
from .companion_diagnostics import CompanionBackendIdentity
from .pillar1_diagnostics import (
    P1RecoveryPrediction,
    P1RecoveryView,
    P1TransitionPrediction,
    P1TransitionView,
    canonical_p1_backend_error,
)
from .pillar3_diagnostics import (
    GROUNDED_ACTIONS,
    P3DiagnosticView,
    P3Prediction,
    ProviderAttemptEvidence,
    canonical_p3_backend_error,
)
from .pillar4_diagnostics import (
    P4Candidate,
    P4QueryView,
    P4RetrievalPrediction,
    canonical_p4_backend_error,
)

if TYPE_CHECKING:
    from web_agent.memory.frozen_store import FrozenMemoryStore


PC01_P1_PREDICTOR_ID = "pc01-pillar1-offline-diagnostic"
PC01_P1_PREDICTOR_VERSION = "v1"
PC01_P1_PREDICTOR_QUALNAME = "PC01Pillar1DiagnosticPredictor"
PC01_P3_PREDICTOR_ID = "pc01-pillar3-offline-diagnostic"
PC01_P3_PREDICTOR_VERSION = "v1"
PC01_P3_PREDICTOR_QUALNAME = "PC01Pillar3DiagnosticPredictor"
PC01_P4_RETRIEVER_ID = "pc01-pillar4-offline-diagnostic"
PC01_P4_RETRIEVER_VERSION = "v1"
PC01_P4_RETRIEVER_QUALNAME = "PC01Pillar4DiagnosticRetriever"
PC01_P4_EMBEDDING_PROVIDER_ID = "selected-checkpoint-memory-embedding"
PC01_P4_EMBEDDING_PROVIDER_VERSION = "v1"
PC01_P4_EMBEDDING_PROVIDER_MODULE = "web_agent.runtime.qwen2vl_pc01"
PC01_P4_EMBEDDING_PROVIDER_QUALNAME = "_SelectedCheckpointRuntime.memory_embedding"
PC01_PARAMETER_DECODING = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 128,
}


def pc01_companion_predictor_source_sha256() -> str:
    """Return the byte identity shared by the three exact adapter classes."""

    return sha256_file(Path(__file__).resolve())


def pc01_parameter_provider_source_sha256() -> str:
    """Return the exact registered hybrid-provider implementation identity."""

    import web_agent.runtime.action_parameters as provider_module

    return sha256_file(Path(provider_module.__file__).resolve())


def pc01_memory_embedding_source_sha256() -> str:
    """Return the exact PC-01 memory-adapter inference implementation bytes."""

    import web_agent.runtime.qwen2vl_pc01 as runtime_module

    return sha256_file(Path(runtime_module.__file__).resolve())


def _observation(
    view: P1TransitionView | P1RecoveryView | P3DiagnosticView,
    *,
    image: Any,
    stage: str,
) -> PolicyObservation:
    if stage not in {"pre", "post", "recovery_pre", "recovery_post"}:
        raise PC01RuntimeError("PC-01 companion observation stage is unregistered")
    return PolicyObservation(
        task_id=view.task_id,
        goal=view.text_state.task_text,
        observation_id=(
            f"companion-{view.example_id}-{stage}-{image.sha256[:20]}"
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


def _validate_runtime_identity(
    runtime: _SelectedCheckpointRuntime,
    identity: CompanionBackendIdentity,
) -> None:
    if runtime.checkpoint_sha256 != identity.checkpoint_sha256:
        raise PC01RuntimeError("PC-01 companion runtime cites another checkpoint")
    if runtime.processor_contract.record_sha256 != identity.processor_contract_sha256:
        raise PC01RuntimeError("PC-01 companion runtime cites another processor")


class PC01Pillar1DiagnosticPredictor:
    """P1 component predictions over causal post-transition evidence only."""

    def __init__(
        self,
        *,
        runtime: _SelectedCheckpointRuntime,
        identity: CompanionBackendIdentity,
    ) -> None:
        error = canonical_p1_backend_error(
            identity, predictor=self, require_input_provenance=False
        )
        if error is not None:
            raise PC01RuntimeError(error)
        _validate_runtime_identity(runtime, identity)
        self._runtime = runtime
        self._identity = identity

    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity:
        return self._identity

    def predict_transition(
        self, view: P1TransitionView, *, seed: int
    ) -> P1TransitionPrediction:
        del seed
        task = RuntimeTaskView(task_id=view.task_id, goal=view.text_state.task_text)
        pre_observation = _observation(view, image=view.pre_image, stage="pre")
        post_observation = _observation(view, image=view.post_image, stage="post")
        pre = self._runtime.batch.stream(
            task=task, observations=(pre_observation,), phase="pre"
        )
        post = self._runtime.batch.stream(
            task=task,
            observations=(pre_observation, post_observation),
            phase="post",
            executed_action=view.executed_action,
            action_value="",
        )
        batch = {
            **_RuntimeBatchBuilder.prefix(pre, "pre_"),
            **_RuntimeBatchBuilder.prefix(post, "post_"),
        }
        with self._runtime.lock, self._runtime.torch.inference_mode():
            self._runtime._assert_eval()
            predictions = self._runtime.model(batch)
        outcome = _tensor_probabilities(self._runtime.torch, predictions["outcome"])
        failure_types = _tensor_probabilities(
            self._runtime.torch, predictions["failure_type"]
        )
        recovery = _tensor_probabilities(
            self._runtime.torch, predictions["recovery"]
        )
        raw_needs = predictions.get("needs_recovery")
        if raw_needs is None:
            raise PC01RuntimeError("PC-01 checkpoint omitted the needs-recovery head")
        needs_probability = float(
            self._runtime.torch.sigmoid(raw_needs.float())[0].detach().cpu().item()
        )
        return P1TransitionPrediction(
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

    def predict_recovery(
        self, view: P1RecoveryView, *, seed: int
    ) -> P1RecoveryPrediction:
        del seed
        task = RuntimeTaskView(task_id=view.task_id, goal=view.text_state.task_text)
        pre_observation = _observation(
            view, image=view.pre_recovery_image, stage="recovery_pre"
        )
        post_observation = _observation(
            view, image=view.post_recovery_image, stage="recovery_post"
        )
        final_action = view.executed_recovery_actions[-1]
        pre = self._runtime.batch.stream(
            task=task, observations=(pre_observation,), phase="pre"
        )
        post = self._runtime.batch.stream(
            task=task,
            observations=(pre_observation, post_observation),
            phase="post",
            executed_action=final_action,
            action_value="",
        )
        recovery = self._runtime.batch.stream(
            task=task,
            observations=(pre_observation, post_observation),
            phase="recovery",
            executed_action=final_action,
            action_value="",
        )
        batch = {
            **_RuntimeBatchBuilder.prefix(pre, "pre_"),
            **_RuntimeBatchBuilder.prefix(post, "post_"),
            **_RuntimeBatchBuilder.prefix(recovery, "recovery_"),
            "recovery_row_indices": self._runtime.torch.tensor(
                [0], dtype=self._runtime.torch.long
            ),
        }
        with self._runtime.lock, self._runtime.torch.inference_mode():
            self._runtime._assert_eval()
            predictions = self._runtime.model(batch)
        raw = predictions.get("recovery_outcome")
        if raw is None or int(raw.shape[0]) != 1:
            raise PC01RuntimeError(
                "PC-01 recovery-outcome head returned no diagnostic assessment"
            )
        probability = float(
            self._runtime.torch.sigmoid(raw.float())[0].detach().cpu().item()
        )
        return P1RecoveryPrediction(
            resolution_probability=probability,
            # PC-01 has one registered executed-recovery outcome head; progress
            # therefore reuses that prediction and is not an invented head.
            progress_probability=probability,
        )


def _provider_attempts(trace: Any) -> tuple[ProviderAttemptEvidence, ...]:
    return tuple(
        ProviderAttemptEvidence(
            source=attempt.source,
            status=attempt.status,
            error_sha256=(
                None
                if attempt.status == "RESOLVED"
                else sha256_json(
                    {"source": attempt.source, "reason": attempt.reason}
                )
            ),
        )
        for attempt in trace.attempts
    )


def _error_sha256(*, stage: str, error: BaseException) -> str:
    return sha256_json(
        {
            "stage": stage,
            "exception_type": type(error).__qualname__,
            "message": str(error),
        }
    )


class PC01Pillar3DiagnosticPredictor:
    """Exact PC-01 action head plus the registered two-stage provider."""

    def __init__(
        self,
        *,
        runtime: _SelectedCheckpointRuntime,
        parameter_provider: HybridParameterProvider,
        identity: CompanionBackendIdentity,
    ) -> None:
        error = canonical_p3_backend_error(
            identity, predictor=self, require_input_provenance=False
        )
        if error is not None:
            raise PC01RuntimeError(error)
        _validate_runtime_identity(runtime, identity)
        if type(parameter_provider) is not HybridParameterProvider:
            raise PC01RuntimeError("P3 requires the exact registered hybrid provider")
        expected_provider = {
            "provider_id": identity.provider_id,
            "provider_version": identity.provider_version,
            "policy_source": identity.provider_policy_source,
            "prompt_sha256": identity.provider_prompt_sha256,
        }
        for field, expected in expected_provider.items():
            if getattr(parameter_provider, field) != expected:
                raise PC01RuntimeError(f"P3 hybrid provider changed {field}")
        if type(parameter_provider.deterministic) is not DeterministicParameterProvider:
            raise PC01RuntimeError("P3 deterministic provider class changed")
        fallback = parameter_provider.frozen_base_fallback
        if type(fallback) is not CallableActionParameterProvider:
            raise PC01RuntimeError("P3 frozen-base provider class changed")
        if (
            fallback.frozen is not True
            or fallback.policy_source != "selected_backbone_unadapted"
            or fallback.prompt_sha256 != identity.provider_prompt_sha256
            or dict(fallback.decoding_parameters) != PC01_PARAMETER_DECODING
            or dict(parameter_provider.decoding_parameters) != PC01_PARAMETER_DECODING
        ):
            raise PC01RuntimeError("P3 frozen-base provider registration changed")
        self._runtime = runtime
        self._provider = parameter_provider
        self._identity = identity

    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity:
        return self._identity

    def predict(self, view: P3DiagnosticView, *, seed: int) -> P3Prediction:
        task = RuntimeTaskView(task_id=view.task_id, goal=view.text_state.task_text)
        observation = _observation(view, image=view.image, stage="pre")
        rng = random.Random(seed)
        # The trained PC-01 policy has a typed six-class head, not a text
        # parser. Processor/model/infrastructure exceptions therefore abort the
        # diagnostic instead of being misreported as ordinary invalid output.
        decision = self._runtime.predict_action(task, observation, rng)
        visible_bbox = (
            decision.bbox if decision.action_type.value in GROUNDED_ACTIONS else None
        )
        try:
            parameters = self._provider.resolve(
                task, observation, decision, rng=random.Random(seed)
            )
        except HybridParameterResolutionError as exc:
            return P3Prediction(
                status="PARAMETER_REJECTED",
                action_probabilities=dict(decision.action_probabilities),
                action_type=decision.action_type.value,
                bbox=visible_bbox,
                grounding_confidence=decision.grounding_confidence,
                parameter_status="REJECTED",
                provider_action_type=None,
                parameters=None,
                provider_attempts=_provider_attempts(exc.trace),
                error_sha256=_error_sha256(stage="parameter_provider", error=exc),
                accounted_rejected_executor_requests=1,
            )
        trace = parameters.resolution_trace
        if trace is None:
            raise PC01RuntimeError("P3 provider omitted its resolution trace")
        return P3Prediction(
            status="RESOLVED",
            action_probabilities=dict(decision.action_probabilities),
            action_type=decision.action_type.value,
            bbox=visible_bbox,
            grounding_confidence=decision.grounding_confidence,
            parameter_status="RESOLVED",
            provider_action_type=parameters.action_type.value,
            parameters=dict(parameters.values),
            provider_attempts=_provider_attempts(trace),
            error_sha256=None,
            accounted_rejected_executor_requests=0,
        )


class PC01Pillar4DiagnosticRetriever:
    """Read-only diagnostic projection over the exact frozen P4 store."""

    def __init__(
        self,
        *,
        store: FrozenMemoryStore,
        identity: CompanionBackendIdentity,
    ) -> None:
        error = canonical_p4_backend_error(
            identity, predictor=self, require_input_provenance=False
        )
        if error is not None:
            raise PC01RuntimeError(error)
        from web_agent.memory.frozen_store import FrozenMemoryStore

        if type(store) is not FrozenMemoryStore:
            raise PC01RuntimeError("P4 requires the exact frozen-memory store class")
        if store.manifest_sha256 != identity.memory_manifest_sha256:
            raise PC01RuntimeError("P4 store manifest differs from backend identity")
        if store.model_seed != PC01_MODEL_SEED:
            raise PC01RuntimeError("P4 store cites another model seed")
        bindings = {
            "checkpoint_sha256": identity.checkpoint_sha256,
            "resolved_config_sha256": identity.resolved_config_sha256,
            "resolved_config_record_sha256": identity.resolved_config_record_sha256,
        }
        for field, expected in bindings.items():
            if store.manifest.get(field) != expected:
                raise PC01RuntimeError(f"P4 store changed {field}")
        self._store = store
        self._identity = identity

    @property
    def diagnostic_identity(self) -> CompanionBackendIdentity:
        return self._identity

    @property
    def store_root(self) -> Path:
        return self._store.root

    def retrieve(self, view: P4QueryView, *, seed: int) -> P4RetrievalPrediction:
        from web_agent.memory.frozen_store import QueryExclusions

        del seed  # Frozen cosine retrieval and memory-ID ties are deterministic.
        result = self._store.query(
            view.query_embedding,
            exclusions=QueryExclusions(
                current_task_id=view.task_id,
                current_episode_id=view.episode_id,
                duplicate_cluster_ids=frozenset(view.duplicate_cluster_ids),
            ),
        )
        candidates = tuple(
            P4Candidate(
                rank=hit.rank,
                memory_id=hit.memory_id,
                similarity=hit.cosine_similarity,
                source_split=str(hit.item["source_split"]),
                source_task_id=str(hit.item["source_task_id"]),
                source_episode_id=str(hit.item["source_episode_id"]),
                duplicate_cluster_id=str(hit.item["duplicate_cluster_id"]),
                strategy=str(hit.item["strategy"]),
                memory_update_flag=hit.item["memory_update_flag"],
                verified_recovery_success=hit.item["verified_recovery_success"],
                final_task_success=hit.item["final_task_success"],
                provenance_valid=hit.item["provenance_valid"],
            )
            for hit in result.hits
        )
        return P4RetrievalPrediction(
            query_embedding_sha256=view.query_embedding_sha256,
            store_manifest_sha256=self._store.manifest_sha256,
            admission_threshold=self._store.admission_threshold,
            candidates=candidates,
            exclusion_reasons=dict(result.exclusion_reasons),
            considered_count=result.considered_count,
            eligible_count=result.eligible_count,
            write_enabled=False,
        )


def _load_runtime(
    *,
    selection: ValidationSelectedCheckpoint,
    artifacts: PC01RuntimeArtifacts,
    identity: CompanionBackendIdentity,
) -> _SelectedCheckpointRuntime:
    if selection.model_seed != identity.model_seed:
        raise PC01RuntimeError("companion identity/selection seeds differ")
    if selection.selected_checkpoint_sha256 != identity.checkpoint_sha256:
        raise PC01RuntimeError("companion identity/selection checkpoints differ")
    if selection.resolved_config_sha256 != identity.resolved_config_sha256:
        raise PC01RuntimeError("companion identity/selection configurations differ")
    if selection.processor_contract_sha256 != identity.processor_contract_sha256:
        raise PC01RuntimeError("companion identity/selection processors differ")
    model, processor, torch, config, processor_contract = _load_selected_model(
        selection=selection, artifacts=artifacts
    )
    return _SelectedCheckpointRuntime(
        model=model,
        processor=processor,
        torch=torch,
        config=config,
        checkpoint_sha256=selection.selected_checkpoint_sha256,
        processor_contract=processor_contract,
    )


def load_pc01_pillar1_diagnostic_predictor(
    *,
    selection: ValidationSelectedCheckpoint,
    artifacts: PC01RuntimeArtifacts,
    identity: CompanionBackendIdentity,
) -> PC01Pillar1DiagnosticPredictor:
    return PC01Pillar1DiagnosticPredictor(
        runtime=_load_runtime(
            selection=selection, artifacts=artifacts, identity=identity
        ),
        identity=identity,
    )


def load_pc01_pillar3_diagnostic_predictor(
    *,
    selection: ValidationSelectedCheckpoint,
    artifacts: PC01RuntimeArtifacts,
    parameter_provider: HybridParameterProvider,
    identity: CompanionBackendIdentity,
) -> PC01Pillar3DiagnosticPredictor:
    return PC01Pillar3DiagnosticPredictor(
        runtime=_load_runtime(
            selection=selection, artifacts=artifacts, identity=identity
        ),
        parameter_provider=parameter_provider,
        identity=identity,
    )


def load_pc01_pillar4_diagnostic_retriever(
    *,
    store_root: str | Path,
    identity: CompanionBackendIdentity,
) -> PC01Pillar4DiagnosticRetriever:
    from web_agent.memory.frozen_store import FrozenMemoryStore

    return PC01Pillar4DiagnosticRetriever(
        store=FrozenMemoryStore.load(store_root), identity=identity
    )
