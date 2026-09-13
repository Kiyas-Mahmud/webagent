"""H3-only wiring for the existing frozen, label-backed advisory memory.

No index building, calibration, model loading or alternative retrieval path.
Generation exposure and execution remain separate planner/runner evidence.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Lock

from web_agent.memory.label_backed_store import LabelBackedMemoryStore
from web_agent.memory.label_experience import LabelExperienceMaterial
from web_agent.memory.label_runtime import LabelBackedMemoryAdapter
from web_agent.runtime.contracts import MemoryQuery, RecoveryDecision
from web_agent.runtime.memory_adapter import MemoryBoundaryError, MemoryDecision, PostFailureEmbeddingRequest
from web_agent.runtime.policy import PolicyError, _guarded_record_callback

VERSION = 'hybrid-advisory-memory-v1'
FROZEN_COUNT = 1974
FROZEN_THRESHOLD = 0.7371385097503662


class HybridMemoryAdapter(LabelBackedMemoryAdapter):
    """Guard and audit the shared context-only adapter; do not charge twice."""

    def __init__(self, *, episode_id, evidence_dir, **kwargs):
        super().__init__(**kwargs, require_strategy_applicability=False, selective_context=True)
        if not episode_id:
            raise MemoryBoundaryError('hybrid memory requires an episode binding')
        self.episode_id = episode_id
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=False)
        self._request_count = 0
        self._candidate_counts = Counter()
        self._request_lock = Lock()

    def _save(self, name, payload):
        with (self.evidence_dir / name).open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')

    def apply_post_failure(self, *, query, shadow_decision, rng, embedding_request=None):
        if (type(query) is not MemoryQuery or type(shadow_decision) is not RecoveryDecision
                or type(embedding_request) is not PostFailureEmbeddingRequest):
            raise MemoryBoundaryError('hybrid memory requires typed causal inputs')
        if query.episode_id != self.episode_id or shadow_decision.memory_experiences:
            raise MemoryBoundaryError('hybrid memory requires the current episode and complete no-memory shadow')
        request = embedding_request
        if (query.task_id != request.post_action_input.task_id
                or query.incident_id != shadow_decision.incident_id
                or any(getattr(query, key) != getattr(request, key) for key in (
                    'query_id', 'post_failure_observation_id', 'failed_action_id',
                    'post_failure_observation_sha256', 'post_action_input_sha256',
                    'processor_contract_sha256', 'checkpoint_sha256'))):
            raise MemoryBoundaryError('hybrid memory query/transition binding mismatch')
        if (self.require_strategy_applicability or not self.selective_context
                or self.recovery_context_material.projection_version != 'train-experience-context-v2'
                or not self.reader.frozen or self.reader.write_enabled
                or not self.store.frozen or self.store.write_enabled
                or self.store.threshold != FROZEN_THRESHOLD):
            raise MemoryBoundaryError('hybrid memory configuration changed')
        with self._request_lock:
            self._request_count += 1
            prefix = f'query-{self._request_count:04d}'
            # Durable shadow first, including failure/budget attempts which later error.
            self._save(prefix+'-shadow.json', {'schema':VERSION, 'system':'H3',
                'query':query.to_dict(), 'embedding_request_sha256':request.record_sha256,
                'shadow_decision':shadow_decision.to_dict(), 'shadow_sha256':shadow_decision.sha256})
            try:
                result = _guarded_record_callback(
                    records=(query, shadow_decision, request),
                    callback=lambda inputs: super(HybridMemoryAdapter,self).apply_post_failure(
                        query=inputs[0], shadow_decision=inputs[1], rng=rng, embedding_request=inputs[2]),
                    boundary='hybrid advisory memory')
                if type(result) is not MemoryDecision:
                    raise MemoryBoundaryError('hybrid memory returned an invalid decision')
                final = result.final_decision
                restored = replace(final, decision_id=shadow_decision.decision_id, memory_experiences=())
                if (result.shadow_decision.sha256 != shadow_decision.sha256
                        or restored.sha256 != shadow_decision.sha256
                        or result.query_result.changed_strategy
                        or result.query_result.changed_target_or_parameters):
                    raise MemoryBoundaryError('advisory memory altered the no-memory action or strategy')
                # Empty admission retains the original complete object, not a
                # reconstructed approximation of the H2 decision.
                result = replace(result, shadow_decision=shadow_decision,
                    final_decision=final if final.memory_experiences else shadow_decision)
                self._candidate_counts.update(result.query_result.candidate_ids)
                self._save(prefix+'-result.json', {'schema':VERSION, 'system':'H3',
                    'decision':result.to_dict(), 'candidate_occurrences':dict(self._candidate_counts),
                    'total_candidate_occurrences':sum(self._candidate_counts.values()),
                    'generation_exposure_measured':False, 'completion_benefit_measured':False})
                return result
            except Exception as exc:
                self._save(prefix+'-error.json', {'schema':VERSION, 'error_type':type(exc).__name__,
                                                'error':str(exc)})
                if isinstance(exc, PolicyError):
                    raise MemoryBoundaryError(f'hybrid memory callback boundary failed: {exc}') from exc
                raise


def build_hybrid_memory(*, system, episode_id, runtime=None, store_directory=None,
                        material_path=None, evidence_dir=None,
                        expected_manifest_sha256=None, expected_material_sha256=None,
                        excluded_tasks=()):
    """Non-memory systems return before touching files or model dependencies."""
    if system not in {'H0','H1','H2','H3'}:
        raise MemoryBoundaryError('unknown hybrid system identity')
    if system != 'H3':
        return None
    if None in (store_directory, material_path, evidence_dir, expected_manifest_sha256, expected_material_sha256):
        raise MemoryBoundaryError('H3 requires the frozen store/material bindings')
    raw = (Path(store_directory)/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise MemoryBoundaryError('hybrid memory manifest mismatch')
    store = LabelBackedMemoryStore(store_directory)
    if (store.memory_input_sha256 != expected_material_sha256
            or store._vectors.shape != (FROZEN_COUNT,768)
            or store.threshold != FROZEN_THRESHOLD
            or getattr(runtime,'checkpoint_sha256',None) != store.checkpoint_sha256):
        raise MemoryBoundaryError('hybrid memory frozen assets mismatch')
    material = LabelExperienceMaterial(material_path,store=store,include_source_context=True)
    return HybridMemoryAdapter(episode_id=episode_id,evidence_dir=evidence_dir,
        store=store,runtime=runtime,transitions={},excluded_tasks=excluded_tasks,
        recovery_context_material=material)
