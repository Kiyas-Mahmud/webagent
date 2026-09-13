"""Read-only retrieval for the explicitly revised, training-label-backed study.

This does not implement the independently verified FrozenMemoryStore contract.
Its results state their weaker evidence basis and do not claim final-task success.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class LabelMemoryHit:
    memory_id: str
    source_task: str
    failure_type: str
    strategy: str
    similarity: float
    admitted: bool
    evidence_basis: str = 'training_dataset_labels'

class LabelBackedMemoryStore:
    frozen = True
    write_enabled = False
    evidence_basis = 'training_dataset_labels'

    def __init__(self, directory: str | Path):
        root = Path(directory)
        raw = (root / 'manifest.json').read_bytes()
        manifest = json.loads(raw)
        if manifest['status'] != 'EMBEDDINGS_AND_TRAIN_ONLY_CALIBRATION_COMPLETE':
            raise ValueError('memory build is incomplete')
        for name, expected in manifest['files'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'memory artifact changed: {name}')
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.checkpoint_sha256 = manifest['checkpoint_sha256']
        self.memory_input_sha256 = manifest['memory_input_sha256']
        self.threshold = float(manifest['admission_threshold'])
        if not -1 <= self.threshold <= 1:
            raise ValueError('invalid admission threshold')
        self._vectors = np.load(root / 'embeddings.npy', mmap_mode='r')
        self._rows = tuple(json.loads((root / 'index.json').read_text()))
        ids = [r['memory_id'] for r in self._rows]
        if ids != sorted(set(ids)) or self._vectors.shape != (len(ids), 768):
            raise ValueError('invalid index ordering or embedding dimensions')
        if not np.isfinite(self._vectors).all() or not np.allclose(np.linalg.norm(self._vectors, axis=1), 1, atol=1e-5):
            raise ValueError('embeddings must be finite unit vectors')

    def query(self, vector, *, excluded_task_ids: Iterable[str] = ()) -> tuple[LabelMemoryHit, ...]:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (768,) or not np.isfinite(vector).all():
            raise ValueError('query must be a finite 768-vector')
        norm = np.linalg.norm(vector)
        if norm <= 0:
            raise ValueError('zero query vector')
        excluded = frozenset(excluded_task_ids)
        allowed = np.array([i for i,r in enumerate(self._rows) if r['source_task'] not in excluded], dtype=np.int64)
        scores = np.clip(self._vectors @ (vector / norm), -1, 1)
        order = allowed[np.argsort(-scores[allowed], kind='stable')][:3]
        return tuple(LabelMemoryHit(
            memory_id=self._rows[i]['memory_id'], source_task=self._rows[i]['source_task'],
            failure_type=self._rows[i]['failure_type'], strategy=self._rows[i]['strategy'],
            similarity=float(scores[i]), admitted=bool(scores[i] >= self.threshold),
        ) for i in order)
