"""Freeze the external-agent evaluation sample before any agent is run.

Selection is seeded, stratified by action_type and committed by hash. It reads
only split_val.json; the locked test split is never opened. Running this twice
must produce the identical sample.
"""
from pathlib import Path
import argparse, hashlib, json, random
from collections import Counter
from PIL import Image

DATA = Path('/home/aiub/kiyas/webagent_full/data/original/final_data_set_40k')
SEED = 20250912
TARGET = 5000
ACTIONS = ('CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY')


def bbox_status(row):
    """Whether this row can be scored for grounding, using the image itself."""
    box = row['labels'].get('action_target_bbox')
    if not box:
        return 'absent', None
    path = DATA / row['inputs']['state_before']
    if not path.exists():
        return 'image_missing', None
    width, height = Image.open(path).size
    x, y, w, h = box['x'], box['y'], box['width'], box['height']
    if w <= 0 or h <= 0:
        return 'degenerate', None
    if x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
        return 'out_of_bounds', None
    return 'valid', {'x': x, 'y': y, 'width': w, 'height': h,
                     'image_width': width, 'image_height': height}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads((DATA / 'split_val.json').read_text())
    assert len(rows) == 7861, len(rows)

    # Stratify by action_type so every class is represented in proportion.
    by_action = {a: [] for a in ACTIONS}
    for index, row in enumerate(rows):
        by_action[row['labels']['action_type']].append(index)
    rng = random.Random(SEED)
    picked = []
    for action in ACTIONS:                      # fixed order, not dict order
        pool = sorted(by_action[action])
        share = round(TARGET * len(pool) / len(rows))
        picked.extend(rng.sample(pool, min(share, len(pool))))
    picked = sorted(picked)[:TARGET]

    selected, grounding_reasons = [], Counter()
    for index in picked:
        row = rows[index]
        status, box = bbox_status(row)
        grounding_reasons[status] += 1
        selected.append({
            'sample_id': row['meta']['sample_id'],
            'val_index': index,
            'state_before': row['inputs']['state_before'],
            'task_description': row['inputs']['task_description'],
            'website_domain': row['inputs']['website_domain'],
            'action_type': row['labels']['action_type'],
            'outcome_label': row['labels']['outcome_label'],
            'failure_type_4': row['labels']['failure_type_4'],
            'memory_update_flag': row['labels']['memory_update_flag'],
            'action_type_eval_mask': row['meta'].get('action_type_eval_mask'),
            'failure_type_4_eval_mask': row['meta'].get('failure_type_4_eval_mask'),
            'grounding_status': status,
            'grounding_box': box,
        })

    manifest = {
        'schema': 'external-agent-eval-sample.v1',
        'purpose': 'Zero-shot evaluation of open-source GUI agents on the Gold v2.8 validation split',
        'source_file': str(DATA / 'split_val.json'),
        'source_sha256': hashlib.sha256((DATA / 'split_val.json').read_bytes()).hexdigest(),
        'locked_test_rows_read': 0,
        'seed': SEED,
        'requested_rows': TARGET,
        'selected_rows': len(selected),
        'stratified_by': 'action_type',
        'action_distribution': dict(Counter(s['action_type'] for s in selected)),
        'outcome_distribution': dict(Counter(s['outcome_label'] for s in selected)),
        'failure_type_distribution': dict(Counter(s['failure_type_4'] for s in selected)),
        'grounding_eligibility': dict(grounding_reasons),
        'grounding_scorable_rows': grounding_reasons['valid'],
        'exclusions_declared_before_running_any_agent': {
            'absent': 'row carries no target box',
            'out_of_bounds': 'labelled box falls outside its own screenshot',
            'degenerate': 'zero or negative width/height',
            'image_missing': 'referenced screenshot not on disk',
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'sample.json').write_text(json.dumps(selected, indent=1) + '\n')
    manifest['sample_sha256'] = hashlib.sha256((args.out / 'sample.json').read_bytes()).hexdigest()
    (args.out / 'sample-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'exclusions_declared_before_running_any_agent'}, indent=1))


if __name__ == '__main__':
    main()
