"""Explicit training-label memory intervention for the revised MiniWoB study.

Uses the shared runtime query/decision logs; does not assert independently
verified recovery or final-task success, or implement the old campaign profile.
"""
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from web_agent.runtime.contracts import ActionType, ExecutionStatus, MemoryQueryResult, RecoveryStrategy, float32_vector_sha256, canonical_sha256
from web_agent.runtime.memory_adapter import MemoryDecision, MemoryBoundaryError
from web_agent.runtime.model_calls import record_model_call

APPLICABILITY_VERSION = 'causal-strategy-admission-v1'


def strategy_exclusion(strategy, transition):
    """Conservative necessary conditions, using only causal execution evidence.

    This is not a usefulness score or a guarantee of successful recovery.
    Missing history cannot establish an in-episode destination for BACKTRACK.
    A parameter-rejected action cannot be made executable by copying it or
    changing its target alone. REPLAN is left to the existing planner checks.
    """
    if strategy == RecoveryStrategy.BACKTRACK.value:
        if not any(entry.action_type is ActionType.NAVIGATE
                   and entry.execution_status is ExecutionStatus.EXECUTED
                   and entry.state_changed and not entry.environment_error
                   for entry in transition.post_observation.causal_history):
            return 'BACKTRACK_NO_EXECUTED_IN_EPISODE_NAVIGATION'
    if (strategy in {RecoveryStrategy.RETRY.value, RecoveryStrategy.ALTERNATIVE_TARGET.value}
            and transition.execution_result.status is ExecutionStatus.REJECTED
            and transition.execution_result.error_kind == 'parameter_resolution_rejected'):
        return 'FAILED_ACTION_PARAMETERS_UNRESOLVED'
    return None


class LabelBackedMemoryAdapter:
    requires_causal_embedding_request = True
    evaluation_mode = False  # legacy campaign attestation is not this study's evidence basis
    def __init__(self, *, store, runtime, transitions, excluded_tasks=(), require_strategy_applicability=False, recovery_context_material=None, selective_context=False):
        if require_strategy_applicability and recovery_context_material is not None:
            raise MemoryBoundaryError('strategy override and experience context are separate experiment modes')
        self.store=store; self.runtime=runtime; self.transitions=transitions
        self.recovery_context_material = recovery_context_material
        if selective_context and recovery_context_material is None:
            raise MemoryBoundaryError('selective context requires bound experience material')
        self.selective_context = selective_context
        if (recovery_context_material is not None
                and recovery_context_material.sha256 != store.memory_input_sha256):
            raise MemoryBoundaryError('experience material/store mismatch')
        self.expected_store_manifest_sha256=store.manifest_sha256
        self.excluded_tasks=frozenset(excluded_tasks)
        self.require_strategy_applicability = require_strategy_applicability
        self.reader=SimpleNamespace(model_seed=42,frozen=True,write_enabled=False,reader_id='miniwob-train-label-reader-v1',index_sha256=store.manifest_sha256,store_manifest_sha256=store.manifest_sha256,evidence_scope='TRAIN_LABEL_BACKED_MINIWOB')
        if require_strategy_applicability:
            self.reader.reader_id += ':' + APPLICABILITY_VERSION
        if recovery_context_material is not None:
            self.reader.reader_id += ':' + recovery_context_material.projection_version
        if selective_context:
            from web_agent.memory.context_applicability import VERSION
            self.reader.reader_id += ':' + VERSION
        self.calls=0
    def assert_model_seed(self, seed):
        if seed != 42: raise MemoryBoundaryError('memory checkpoint seed mismatch')
    def apply_post_failure(self, *, query, shadow_decision, rng, embedding_request=None):
        if query.incident_id != shadow_decision.incident_id: raise MemoryBoundaryError('incident mismatch')
        if embedding_request is None:
            raise MemoryBoundaryError('missing exact causal embedding request')
        if (embedding_request.query_id != query.query_id
                or embedding_request.post_failure_observation_id != query.post_failure_observation_id
                or embedding_request.failed_action_id != query.failed_action_id
                or embedding_request.checkpoint_sha256 != self.store.checkpoint_sha256
                or embedding_request.post_action_input_sha256 != query.post_action_input_sha256):
            raise MemoryBoundaryError('memory embedding request binding mismatch')
        transition=embedding_request.post_action_input
        if transition is None: raise MemoryBoundaryError('missing exact causal transition for memory query')
        if transition.executed_action.action_id != query.failed_action_id: raise MemoryBoundaryError('memory failed-action binding mismatch')
        started=perf_counter()
        record_model_call(stage='memory_embedding',component_id='pc01-label-memory-v1')
        from web_agent.runtime.contracts import RuntimeTaskView
        task=RuntimeTaskView(task_id=transition.task_id,goal=transition.post_observation.goal)
        batch=self.runtime.batch.transition(task,transition)
        with self.runtime.lock,self.runtime.torch.inference_mode():
            raw=self.runtime.model.memory_embedding(batch)[0].float().cpu().numpy()
        raw_sha=float32_vector_sha256(tuple(float(x) for x in raw))
        vector=(raw/np.linalg.norm(raw)).astype(np.float32)
        hits=self.store.query(vector,excluded_task_ids=self.excluded_tasks|{query.task_id})
        admitted=next((h for h in hits if h.admitted),None)
        exclusions = {}
        if admitted is not None and self.require_strategy_applicability:
            reason = strategy_exclusion(admitted.strategy, transition)
            if reason is not None:
                exclusions[admitted.memory_id] = reason
                # Abstain, retaining the complete shadow decision. Do not search
                # lower-ranked candidates until a preferred strategy appears.
                admitted = None
        final=shadow_decision if admitted is None else replace(shadow_decision,decision_id=f"{shadow_decision.decision_id}:memory:{query.query_id}:{admitted.memory_id}",strategy=RecoveryStrategy(admitted.strategy))
        context_ids = ()
        if self.recovery_context_material is not None:
            # Fixed top-three threshold results, including unhelpful examples.
            # No task-outcome ranking, fallback search or strategy override.
            experiences = tuple(self.recovery_context_material.for_hit(hit)
                                for hit in hits if hit.admitted)
            if self.selective_context:
                from web_agent.memory.context_applicability import context_exclusion
                eligible = []
                for item in experiences:
                    reason = context_exclusion(item.to_dict(), transition.to_dict())
                    if reason:
                        exclusions[item.memory_id] = reason
                    else:
                        eligible.append(item)
                experiences = tuple(eligible)
                admitted = next((h for h in hits if h.admitted and h.memory_id not in exclusions), None)
            context_ids = tuple(item.memory_id for item in experiences)
            final = (replace(shadow_decision,
                             decision_id=f'{shadow_decision.decision_id}:memory-context:{query.query_id}',
                             memory_experiences=experiences)
                     if experiences else shadow_decision)
        self.calls+=1
        result=MemoryQueryResult(query_id=query.query_id,shadow_decision_sha256=shadow_decision.sha256,candidate_ids=tuple(h.memory_id for h in hits),scores=tuple(h.similarity for h in hits),exclusion_reasons=exclusions,admitted_candidate_id=admitted.memory_id if admitted else None,admitted=admitted is not None,changed_strategy=final.strategy!=shadow_decision.strategy,changed_target_or_parameters=False,final_strategy=final.strategy,latency_ms=(perf_counter()-started)*1000,query_embedding_sha256=raw_sha,reader_id=self.reader.reader_id,store_manifest_sha256=self.store.manifest_sha256,embedding_binding_sha256=canonical_sha256({'query':query.record_sha256,'transition':transition.record_sha256,'checkpoint':self.store.checkpoint_sha256}),normalized_query_embedding=tuple(float(x) for x in vector),normalized_query_embedding_sha256=float32_vector_sha256(tuple(float(x) for x in vector)))
        if self.recovery_context_material is not None:
            result = replace(result, changed_recovery_context=bool(context_ids),
                             context_candidate_ids=context_ids,
                             memory_material_sha256=self.recovery_context_material.sha256)
        return MemoryDecision(shadow_decision=shadow_decision,final_decision=final,query_result=result)
