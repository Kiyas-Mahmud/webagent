"""CPU-only preparation, contracts and durable execution. Never opens images."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from web_agent.data.recovery_transitions import build_recovery_transition_index
from web_agent.labels import ACTION_TYPE, EXECUTION_OUTCOME, FAILURE_TYPE

ROOT = Path(__file__).resolve().parents[4]
SYSTEMS = ('browser_use', 'agent_s2', 'webvoyager', 'pc03')
PHASES = ('interaction_assessment', 'recovery_assessment')
SOURCE_HASH = '42df34295f2b0f81e48a29e4acae1b8b16c862ed2a140c19d2a63c4c950d435c'
SELECTION_HASH = '6bb337c47aec52e5b4f3d71fd6a32b4512e81ba6cd9ad521e79bca5c2bd57106'
CHECKPOINT_HASH = '35eec6c940836e581abe597006cbf4d9aedbcba06c574ba3d8c2829f667d28cb'

def encode(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, obj):
    """Exclusive creation: existing evidence is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(encode(obj)); f.flush(); os.fsync(f.fileno())

def request_key(request):
    return sha(encode(request))

def validate_request(r):
    if set(r) != {'case_id', 'task_id', 'phase', 'task_description', 'website_domain',
                  'before_image', 'after_image', 'executed_action'}:
        raise ValueError('Request has missing or forbidden fields')
    if r['phase'] not in PHASES:
        raise ValueError('Unknown phase')
    if set(r['executed_action']) != {'type', 'value'}:
        raise ValueError('Unexpected action fields')
    if r['executed_action']['type'] not in ACTION_TYPE:
        raise ValueError('Unknown action')
    # This source has no values. Introducing them changes the frozen input contract.
    if r['executed_action']['value'] is not None:
        raise ValueError('Unexpected action value')
    for name in ('before_image', 'after_image'):
        p = Path(r[name])
        if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'images':
            raise ValueError('Image must be a relative images/ reference')

def make_views(rows, ids):
    by_id = {r['meta']['sample_id']: r for r in rows}
    if len(by_id) != len(rows) or len(ids) != len(set(ids)):
        raise ValueError('Duplicate source or selected IDs')
    links, audit = build_recovery_transition_index(rows)
    if any(audit[k] for k in ('missing_next_step', 'ambiguous_next_step',
                              'nonconsecutive_next_step', 'invalid_attempted_label_contract')):
        raise ValueError('Recovery linkage failed')
    inputs, targets = [], {}
    for sid in ids:
        row = by_id[sid]; inp, lab, meta = row['inputs'], row['labels'], row['meta']
        common = dict(case_id=sid, task_id=meta['task_id'],
                      task_description=inp['task_description'], website_domain=inp['website_domain'])
        r = dict(common, phase=PHASES[0], before_image=inp['state_before'],
                 after_image=inp['state_after'], executed_action={'type': lab['action_type'], 'value': None})
        validate_request(r); inputs.append(r)
        targets[request_key(r)] = {'outcome_label': lab['outcome_label'], 'failure_type_4': lab['failure_type_4']}
        if sid in links:
            link = links[sid]
            r = dict(common, phase=PHASES[1], before_image=link['failure_state'],
                     after_image=link['post_recovery_state'],
                     executed_action={'type': link['executed_recovery_action'], 'value': None})
            validate_request(r); inputs.append(r)
            targets[request_key(r)] = {'outcome_label': 'SUCCESS' if link['recovery_success'] else 'FAILURE'}
    return inputs, targets, links

def prepare(source, selection, output):
    if file_hash(source) != SOURCE_HASH:
        raise ValueError('Source validation hash mismatch')
    selection = Path(selection)
    if file_hash(selection / 'manifest.json') != SELECTION_HASH:
        raise ValueError('Frozen 240-case selection identity changed')
    manifest = read(selection / 'manifest.json')
    for name, key in [('references.json', 'references_sha256'), ('scoring-labels.json', 'scoring_labels_sha256')]:
        if file_hash(selection / name) != manifest[key]:
            raise ValueError('Selection artifact changed: ' + name)
    if manifest['source_sha256'] != SOURCE_HASH:
        raise ValueError('Selection source mismatch')
    rows = read(source); refs = read(selection / 'references.json')
    ids = [r['sample_id'] for r in refs]
    requests, targets, links = make_views(rows, ids)
    if len(ids) != 240 or len(requests) != 360:
        raise ValueError('Expected 240 interaction + 120 recovery requests')
    for ref in refs:
        if rows[ref['val_index']]['meta']['sample_id'] != ref['sample_id']:
            raise ValueError('Selection index mismatch')
        if ref['recovery_sample_id'] != links.get(ref['sample_id'], {}).get('recovery_sample_id'):
            raise ValueError('Selection recovery dependency mismatch')
    blocked = {r['task_id'] for r in refs}
    by_id = {r['meta']['sample_id']: r for r in rows}
    blocked |= {by_id[r['recovery_sample_id']]['meta']['task_id'] for r in refs if r['recovery_sample_id']}
    dev_ids = []
    # Two per action; one recovery case where available, then one other unique task.
    for action in ACTION_TYPE:
        candidates = sorted((r for r in rows if r['labels']['action_type'] == action), key=lambda r:r['meta']['sample_id'])
        for require_recovery in (True, False):
            eligible = [r for r in candidates if r['meta']['task_id'] not in blocked and
                        (not require_recovery or r['meta']['sample_id'] in links)]
            if not eligible:
                raise ValueError('Not enough disjoint development tasks')
            row = eligible[0]; dev_ids.append(row['meta']['sample_id']); blocked.add(row['meta']['task_id'])
    dev_requests, dev_targets, _ = make_views(rows, dev_ids)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for name, value in [('inputs.json', requests), ('scoring-only.json', targets),
                        ('development-inputs.json', dev_requests), ('development-scoring-only.json', dev_targets)]:
        write_new(output/name, value)
    receipt = {'schema':'task1.prepared.v1','source_sha256':SOURCE_HASH,
               'selection_sha256':file_hash(selection/'manifest.json'),
               'files':{p.name:file_hash(p) for p in output.glob('*.json')},
               'pilot_cases':240,'pilot_requests':360,'development_cases':12,
               'development_requests':len(dev_requests),'images_opened':0,'model_calls':0}
    write_new(output/'manifest.json', receipt)
    return receipt

def load_prepared(folder, development=False):
    folder = Path(folder); manifest = read(folder/'manifest.json')
    for name, digest in manifest['files'].items():
        if file_hash(folder/name) != digest:
            raise ValueError('Prepared evidence changed: '+name)
    requests = read(folder/('development-inputs.json' if development else 'inputs.json'))
    for r in requests: validate_request(r)
    if len({request_key(r) for r in requests}) != len(requests):
        raise ValueError('Duplicate requests')
    return requests

def parse_response(raw, phase):
    """Exactly one JSON object; no prose/fences, repairs, or label guessing."""
    try:
        def unique(pairs):
            d = {}
            for k,v in pairs:
                if k in d: raise ValueError('Duplicate JSON key')
                d[k] = v
            return d
        d = json.loads(raw, object_pairs_hook=unique)
        required = {'outcome_label','reason'} | ({'failure_type_4'} if phase == PHASES[0] else set())
        if not isinstance(d,dict) or set(d) != required or not isinstance(d['reason'],str):
            raise ValueError('Output schema mismatch')
        if d['outcome_label'] == 'ABSTAIN':
            return {'status':'abstention','predictions':{}}
        if d['outcome_label'] not in EXECUTION_OUTCOME: raise ValueError('Unknown outcome')
        if phase == PHASES[0] and d['failure_type_4'] not in FAILURE_TYPE: raise ValueError('Unknown category')
        return {'status':'valid','predictions':{k:v for k,v in d.items() if k != 'reason'}}
    except (ValueError,TypeError):
        return {'status':'parse_error','predictions':{}}

def execute(requests, system, backend, output, binding):
    """One result per phase; started-but-unfinished requests are preserved, never replayed."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # Single writer even if another shell launches the same system.
    import fcntl
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'system':system, 'binding':binding, 'requests_sha256':sha(encode(requests))}
        if (output/'identity.json').exists():
            if read(output/'identity.json') != identity: raise ValueError('Resume identity mismatch')
        else: write_new(output/'identity.json',identity)
        blocked = []
        for request in requests:
            key = request_key(request); path = output/(key+'.json'); started = output/(key+'.started.json')
            if path.exists():
                if read(path).get('status') in ('infrastructure_error', 'interrupted'):
                    blocked.append(key)
            elif started.exists():
                write_new(path,{'request_key':key,'case_id':request['case_id'],'task_id':request['task_id'],
                               'phase':request['phase'],'system':system,'status':'interrupted',
                               'predictions':{},'model_calls':None,'latency_seconds':None})
                blocked.append(key)
        if blocked:
            raise RuntimeError('Review required: preserved infrastructure/interrupted records: '+','.join(blocked))
        for request in requests:
            key = request_key(request); path = output/(key+'.json'); started = output/(key+'.started.json')
            if path.exists(): continue
            write_new(started, {'request_key':key,'started_at':time.time()})
            t = time.monotonic()
            record = {'request_key':key,'case_id':request['case_id'],'task_id':request['task_id'],
                      'phase':request['phase'],'system':system}
            try:
                result = backend.predict(request)
                record.update(result)
                record['latency_seconds'] = time.monotonic()-t
                write_new(path,record)
            except Exception as exc:
                record.update(status='infrastructure_error', predictions={}, error_type=type(exc).__name__,
                              error=str(exc),model_calls=None,latency_seconds=time.monotonic()-t)
                write_new(path,record)
                raise
        return len(list(output.glob('*.started.json')))
