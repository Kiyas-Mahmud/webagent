"""Read existing, hash-bound train material for a separate context experiment."""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from web_agent.runtime.contracts import (
    ActionType, RecoveryStrategy, RetrievedRecoveryExperience,
)
from web_agent.runtime.memory_adapter import MemoryBoundaryError


class LabelExperienceMaterial:
    """Never opens source dataset rows or image files; never invents missing values."""

    def __init__(self, path, *, store, include_source_context=False):
        self.projection_version = 'train-experience-context-v2' if include_source_context else 'train-experience-context-v1'
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != store.memory_input_sha256:
            raise MemoryBoundaryError('memory material does not match embedding source')
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
        records = {}
        for item in items:
            if (item['source_split'] != 'train'
                    or item['evidence_basis'] != 'training_dataset_labels'):
                raise MemoryBoundaryError('experience material must be train-only')
            mid = item['memory_id']
            if mid in records:
                raise MemoryBoundaryError('duplicate memory material ID')
            material = item['material']
            records[mid] = RetrievedRecoveryExperience(
                memory_id=mid,
                source_task_id=material['canonical_task_id'],
                failure_type=material['failure_type'],
                failed_action=(ActionType(material['failed_action'])
                               if material.get('failed_action_available', True) else None),
                recovery_action=ActionType(material['executed_recovery_action']),
                recorded_strategy=RecoveryStrategy(material['recovery_strategy']),
                recovery_action_value=material['recovery_action_value'] or None,
                reflection=material['reflection_text'] or None,
                material_version='train-experience-v2' if include_source_context else 'train-experience-v1',
                source_task_description=material.get('task_description') if include_source_context else None,
                source_domain=material.get('website_domain') if include_source_context else None,
                availability={'failed_action':bool(material.get('failed_action_available',True)),
                    'recovery_action_value':bool(material['recovery_action_value']),
                    'reflection':bool(material['reflection_text']),
                    'source_task_description':bool(material.get('task_description')),
                    'source_domain':bool(material.get('website_domain'))} if include_source_context else {},
            )
        self._records = MappingProxyType(records)

    def for_hit(self, hit):
        item = self._records.get(hit.memory_id)
        if (item is None or item.source_task_id != hit.source_task
                or item.failure_type != hit.failure_type
                or item.recorded_strategy.value != hit.strategy):
            raise MemoryBoundaryError('retrieved index/material identity mismatch')
        return item
