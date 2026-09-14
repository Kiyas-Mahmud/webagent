"""Prepare a metadata-only, reproducible validation pilot; never open images.

The output binds references and scoring labels, not an agent input payload.
This balanced diagnostic sample is not a natural-distribution final test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/aiub/kiyas/webagent_full/data/original/final_data_set_40k/split_val.json')
EXPECTED_SHA256 = '42df34295f2b0f81e48a29e4acae1b8b16c862ed2a140c19d2a63c4c950d435c'
SEED = 20260914
ACTIONS = ('CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY')
QUOTAS = {'recovered': 10, 'recovery_failed': 10, 'success_no_recovery': 10,
          'loop_no_recovery': 2, 'perception_no_recovery': 4, 'mismatch_no_recovery': 4}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def stratum(row):
    lab = row['labels']
    if lab['recovery_success'] is True:
        return 'recovered'
    if lab['recovery_success'] is False:
        return 'recovery_failed'
    if lab['outcome_label'] == 'SUCCESS':
        return 'success_no_recovery'
    return {'LOOP_DETECTED': 'loop_no_recovery',
            'PERCEPTION_ERROR': 'perception_no_recovery',
            'ACTION_MISMATCH': 'mismatch_no_recovery'}[lab['failure_type_4']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    raw = SOURCE.read_bytes()
    if digest(raw) != EXPECTED_SHA256:
        raise ValueError('Validation source identity changed; do not silently resample')
    rows = json.loads(raw)
    assert len(rows) == 7861
    by_id = {r['meta']['sample_id']: i for i, r in enumerate(rows)}
    assert len(by_id) == len(rows)
    module_path = ROOT / 'src/web_agent/data/recovery_transitions.py'
    spec = importlib.util.spec_from_file_location('recovery_transitions', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transitions, transition_audit = module.build_recovery_transition_index(rows)
    assert transition_audit['nonconsecutive_next_step'] == 0

    chosen = []
    pools = {}
    for action in ACTIONS:
        for group, count in QUOTAS.items():
            pool = [i for i, row in enumerate(rows)
                    if row['labels']['action_type'] == action and stratum(row) == group]
            pools[f'{action}/{group}'] = len(pool)
            assert len(pool) >= count, (action, group, len(pool))
            pool.sort(key=lambda i: (digest(f'{SEED}:{rows[i]["meta"]["sample_id"]}'.encode()),
                                     rows[i]['meta']['sample_id']))
            chosen.extend(pool[:count])
    chosen.sort()
    assert len(chosen) == len(set(chosen)) == 240
    references, scoring = [], []
    for i in chosen:
        row = rows[i]
        sid = row['meta']['sample_id']
        transition = transitions.get(sid)
        references.append({
            'sample_id': sid, 'val_index': i, 'task_id': row['meta']['task_id'],
            'stratum': stratum(row),
            'recovery_sample_id': transition['recovery_sample_id'] if transition else None,
            'recovery_val_index': by_id[transition['recovery_sample_id']] if transition else None,
        })
        scoring.append({'sample_id': sid, 'labels': row['labels'],
                        'action_type_eval_mask': row['meta'].get('action_type_eval_mask'),
                        'failure_type_4_eval_mask': row['meta'].get('failure_type_4_eval_mask')})
    selected = [rows[i] for i in chosen]
    distribution = lambda rs, key: dict(Counter(str(r['labels'].get(key)) for r in rs))
    keys = ('action_type', 'outcome_label', 'failure_type_4', 'recovery_strategy',
            'recovery_success', 'memory_update_flag')
    assert all(v == 40 for v in distribution(selected, 'action_type').values())
    assert sum(r['recovery_sample_id'] is not None for r in references) == 120
    assert all(distribution(selected, k).keys() == distribution(rows, k).keys() for k in keys)
    reference_bytes, scoring_bytes = encode(references), encode(scoring)
    manifest = {
        'schema': 'task1.mini-validation-pilot.v1', 'status': 'selected_not_evaluated',
        'purpose': 'balanced diagnostic pilot, not final-test or full-split performance',
        'source': str(SOURCE), 'source_sha256': digest(raw), 'source_rows': len(rows),
        'selection_seed': SEED, 'selection_method': 'SHA256(seed:sample_id) ranked within fixed strata',
        'per_action_quotas': QUOTAS, 'available_strata': pools,
        'selected_rows': len(selected), 'unique_selected_tasks': len({r['meta']['task_id'] for r in selected}),
        'recovery_assessment_rows': 120, 'source_recovery_audit': transition_audit,
        'source_distributions': {k: distribution(rows, k) for k in keys},
        'selected_distributions': {k: distribution(selected, k) for k in keys},
        'selected_domains': len({r['inputs']['website_domain'] for r in selected}),
        'required_input_metadata_missing': {k: sum(not r['inputs'].get(k) for r in selected)
            for k in ('state_before', 'state_after', 'task_description', 'website_domain')},
        'source_action_value_keys': sum('action_value' in r['labels'] for r in rows),
        'images_opened': 0, 'image_availability_verified': False, 'locked_test_rows_read': 0,
        'model_calls': 0, 'references_sha256': digest(reference_bytes),
        'scoring_labels_sha256': digest(scoring_bytes),
        'preparer_sha256': digest(Path(__file__).read_bytes()),
        'recovery_builder_sha256': digest(module_path.read_bytes()),
        'input_rule': 'Never pass this manifest, strata or scoring labels to an agent; construct phase-specific views.',
        'recovery_rule': 'Build transitions from original validation split before selecting; dependency rows are not additional scored cases.',
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'references.json').write_bytes(reference_bytes)
    (args.out / 'scoring-labels.json').write_bytes(scoring_bytes)
    (args.out / 'manifest.json').write_bytes(encode(manifest))
    print(json.dumps({k: manifest[k] for k in ('selected_rows', 'unique_selected_tasks',
          'recovery_assessment_rows', 'selected_domains', 'selected_distributions')}, indent=2))


if __name__ == '__main__':
    main()
