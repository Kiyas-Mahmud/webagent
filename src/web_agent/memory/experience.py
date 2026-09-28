"""Pillar 4 experience memory: the agent's own failure→recovery episodes.

Written during runs, read on the next failure. Each record is one incident:
the failed action and the element it hit, P1's diagnosis and strategy, every
recovery action tried (type, element, value) with P1's assessment of it, and
whether the incident was resolved. Resolved and unresolved incidents are both
kept — "this did not work" is failure-aware memory too.

Keys are the selected Qwen2.5 checkpoint's own memory embedding (the P1 model),
so no second encoder is needed. The trained memory head decides what to store.
Nothing here reads evaluator reward; outcomes are the learned assessments.

Retrieval is page-scoped. On the 1,974 training transitions the memory
embedding separates relevant from irrelevant neighbours only weakly (AUC 0.60,
median top-1 cosine 0.989; p4-qwen25-embeddings-v1): it was trained to decide
storage, not retrieval. So candidates must come from the same observable page
(URL host+path), are ranked by that embedding, and must clear the calibrated
threshold and show a recorded element on the current page.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

import numpy as np

from web_agent.eval.task2.elements import element_summary, normalize_text  # noqa: F401 (re-export)

SCHEMA = 'p4-experience-memory-v1'
TOP_K = 3


def page_key(url):
    """Observable page identity: host and path, without port, query or fragment."""
    parts = urlsplit(url or '')
    return f"{parts.hostname or ''}{parts.path}".lower()


def element_labels(element):
    if not element:
        return set()
    labels = {normalize_text(element.get('text'))}
    labels |= {normalize_text(element['attributes'].get(k)) for k in ('id', 'name', 'aria-label', 'placeholder')}
    return {label for label in labels if label}


def page_labels(controls):
    """Visible text/names/ids of the current page's controls (observable to the agent)."""
    labels = set()
    for control in controls or ():
        labels.add(normalize_text(control.get('text')))
        labels.add(normalize_text(control.get('name')))
        for name in control.get('accessible_names') or ():
            labels.add(normalize_text(name.get('value') if isinstance(name, dict) else name))
    return {label for label in labels if label}


@dataclass(frozen=True)
class ExperienceHit:
    memory_id: str
    similarity: float
    exclusion_reason: str | None
    record: dict

    @property
    def admitted(self):
        return self.exclusion_reason is None


class ExperienceMemory:
    """Append-only experience store on disk: records.jsonl + embeddings.npy."""

    def __init__(self, directory, *, threshold, checkpoint_sha256, write_enabled=True):
        self.root = Path(directory); self.root.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        if not -1 <= self.threshold <= 1:
            raise ValueError('invalid admission threshold')
        self.checkpoint_sha256 = checkpoint_sha256
        self.write_enabled = bool(write_enabled)
        manifest = self.root/'manifest.json'
        identity = {'schema': SCHEMA, 'checkpoint_sha256': checkpoint_sha256, 'dimension': 768}
        if manifest.exists():
            stored = json.loads(manifest.read_text())
            if {k: stored.get(k) for k in identity} != identity:
                raise ValueError('MEMORY_EMBEDDING_SPACE_MISMATCH: store was built by another checkpoint/schema')
        else:
            manifest.write_text(json.dumps(identity, indent=2)+'\n')
        self.records = [json.loads(line) for line in (self.root/'records.jsonl').read_text().splitlines()
                        if line.strip()] if (self.root/'records.jsonl').exists() else []
        vectors = self.root/'embeddings.npy'
        self.vectors = np.load(vectors) if vectors.exists() else np.zeros((0, 768), np.float32)
        if len(self.records) != len(self.vectors):
            raise ValueError('memory records and embeddings are out of step')

    def __len__(self):
        return len(self.records)

    @staticmethod
    def unit(vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (768,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError('memory key must be a finite non-zero 768-vector')
        return vector/np.linalg.norm(vector)

    def query(self, vector, *, current_episode, current_controls, current_url):
        """Same-page top-k by cosine; admission = threshold and observable applicability."""
        page = page_key(current_url)
        candidates = np.array([i for i, r in enumerate(self.records) if page_key(r['page_url']) == page], dtype=np.int64)
        if not len(candidates):
            return []
        scores = np.clip(self.vectors @ self.unit(vector), -1, 1)
        order = candidates[np.argsort(-scores[candidates], kind='stable')][:TOP_K]
        visible = page_labels(current_controls)
        hits = []
        for index in order:
            record = self.records[int(index)]
            similarity = float(scores[index])
            if record['episode_id'] == current_episode:
                reason = 'SAME_EPISODE'
            elif similarity < self.threshold:
                reason = 'BELOW_THRESHOLD'
            else:
                recorded = element_labels(record['failed_action'].get('element'))
                for attempt in record['recovery_attempts']:
                    recorded |= element_labels(attempt.get('element'))
                # Without an observable anchor, applicability cannot be checked and advice is empty.
                reason = ('NO_RECORDED_ELEMENT' if not recorded else
                          'NO_RECORDED_ELEMENT_ON_CURRENT_PAGE' if not recorded & visible else None)
            hits.append(ExperienceHit(record['memory_id'], similarity, reason, record))
        return hits

    def add(self, record, vector):
        if not self.write_enabled:
            raise RuntimeError('memory is read-only')
        vector = self.unit(vector)
        record = dict(record, memory_id=record.get('memory_id') or uuid.uuid4().hex[:12],
                      written_at=record.get('written_at') or time.time(), schema=SCHEMA)
        with (self.root/'records.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True)+'\n')
        self.records.append(record)
        self.vectors = np.vstack([self.vectors, vector[None]])
        temp = self.root/'embeddings.tmp.npy'
        np.save(temp, self.vectors); temp.replace(self.root/'embeddings.npy')
        return record['memory_id']


def advice(hit):
    """What the actor sees: concrete, observable past experience with its learned outcome."""
    record = hit.record
    return {'similarity': round(hit.similarity, 3), 'past_task': record['task_goal'],
            'failed_action': record['failed_action'], 'diagnosed_failure': record['failure_type'],
            'suggested_strategy': record['strategy'],
            'recovery_attempts': [{k: a.get(k) for k in ('action_type', 'element', 'value', 'assessed_outcome')}
                                  for a in record['recovery_attempts']],
            'incident_outcome': 'resolved' if record['resolved'] else 'not resolved'}
