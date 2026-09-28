"""Embed the 1,974 training recovery transitions in the selected Qwen2.5 memory space.

Same inputs and stream as the historical PC-01 build (failed step's before/after
screenshots, post stream, failed action type), but through the PC-02 Qwen2.5
checkpoint that P1 already uses. Output is used for threshold calibration only:
the training items carry no element or corrective value, so they cannot supply
concrete advice and are not served as live memory.

Calibration: leave-source-task-out top-1 over training items only. A retrieved
pair is relevant when both recoveries used the same strategy AND the same
recovery action type (stricter than the historical strategy+failure-type proxy,
under which 96% of pairs were relevant). Highest F1 wins; higher threshold wins
ties. No MiniWoB or evaluation input is read.

  PYTHONPATH=src .venv/bin/python scripts/memory/build_qwen25_memory_calibration.py OUT
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from web_agent.eval.task1.core import file_hash

ITEMS = Path('/home/aiub/kiyas/table2-evidence/p4-local-label-memory-v1')


def calibrate(vectors, rows, relevant):
    sim = vectors @ vectors.T
    tasks = np.array([r['source_task'] for r in rows])
    scores, labels = [], []
    for i in range(len(rows)):
        allowed = np.flatnonzero(tasks != tasks[i])
        j = int(allowed[np.argsort(-sim[i, allowed], kind='stable')][0])
        scores.append(float(sim[i, j])); labels.append(relevant(rows[i], rows[j]))
    scores, labels = np.array(scores), np.array(labels)
    best = None
    for threshold in sorted(set(scores.tolist()+[1.0]), reverse=True):
        predicted = scores >= threshold
        tp = int((predicted & labels).sum()); fp = int((predicted & ~labels).sum()); fn = int((~predicted & labels).sum())
        f1 = 2*tp/max(2*tp+fp+fn, 1)
        if best is None or f1 > best['f1']:
            best = {'threshold': threshold, 'f1': f1, 'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    return best, int(labels.sum()), scores, labels


def strict(a, b):
    return a['strategy'] == b['strategy'] and a['recovery_action'] == b['recovery_action']


def historical(a, b):
    return a['strategy'] == b['strategy'] and a['failure_type'] == b['failure_type']


def main():
    import torch
    from web_agent.data.gold_dataloader import _collate_stream
    from web_agent.eval.task1.qwen25_backends import QWEN25_CHECKPOINT_HASH, Qwen25Assessor
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ITEMS/'manifest.json').read_text())
    if file_hash(ITEMS/'memory-items.jsonl') != manifest['memory_items_sha256']:
        raise ValueError('Memory items changed')
    items = sorted((json.loads(l) for l in (ITEMS/'memory-items.jsonl').read_text().splitlines()), key=lambda x: x['memory_id'])
    sources, roots = {}, {}
    for role, path in manifest['source_paths'].items():
        path = Path(path)
        if file_hash(path) != manifest['source_sha256'][role]:
            raise ValueError('Source changed: '+role)
        sources[role] = {r['meta']['sample_id']: r for r in json.loads(path.read_text())}
        roots[role] = path.parent if role == 'original_gold' else path.parent.parent
    config = json.loads(Path('configs/eval/task1/qwen25_dual_lab_paths.json').read_text())
    assessor = Qwen25Assessor(config)
    model, cfg = assessor.model, assessor.cfg
    rows = []; started = time.monotonic()
    for n, item in enumerate(items):
        material = item['material']; role = material['source_dataset_role']
        record = sources[role][material['source_sample_id']]; inp = record['inputs']
        paths = [roots[role]/inp[k] for k in ('state_before', 'state_after')]
        dest = out/f'vector-{n:04d}.npy'
        if not dest.exists():
            from PIL import Image
            images = []
            for path in paths:
                with Image.open(path) as image:images.append(image.convert('RGB').copy())
            values = assessor.dataset._vlm_inputs({k: inp[k] for k in ('task_description', 'website_domain')}, images, 'post',
                                                  executed_action=record['labels']['action_type'], action_value='')
            batch = _collate_stream([{'post_'+k: v for k, v in values.items()}], 'post_', cfg)
            with torch.inference_mode():
                vector = model.memory_embedding(batch)
                flag = torch.sigmoid(model.memory_head(vector)['memory_flag']).item()
            vector = vector[0].float().cpu().numpy()
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError('Invalid memory embedding')
            np.save(dest, (vector/np.linalg.norm(vector)).astype(np.float32))
            (out/f'flag-{n:04d}.json').write_text(json.dumps({'store_probability': flag}))
            del batch; torch.cuda.empty_cache()
        rows.append({'memory_id': item['memory_id'], 'source_task': material['canonical_task_id'],
                     'failure_type': material['failure_type'], 'strategy': material['recovery_strategy'],
                     'recovery_action': material['executed_recovery_action'],
                     'store_probability': json.loads((out/f'flag-{n:04d}.json').read_text())['store_probability'],
                     'embedding_image_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]})
        if n % 50 == 0 or n == len(items)-1:
            print(json.dumps({'completed': n+1, 'total': len(items), 'elapsed_seconds': round(time.monotonic()-started)}), flush=True)
    vectors = np.stack([np.load(out/f'vector-{n:04d}.npy') for n in range(len(items))])
    np.save(out/'embeddings.npy', vectors)
    (out/'index.json').write_text(json.dumps(rows, indent=1)+'\n')
    selected, positives, scores, labels = calibrate(vectors, rows, strict)
    reference, reference_positives, _, _ = calibrate(vectors, rows, historical)
    np.savez(out/'calibration-pairs.npz', scores=scores, labels=labels)
    result = {'status': 'QWEN25_EMBEDDINGS_AND_TRAIN_ONLY_CALIBRATION_COMPLETE', 'count': len(items), 'dimension': 768,
              'checkpoint_sha256': QWEN25_CHECKPOINT_HASH, 'memory_input_sha256': manifest['memory_items_sha256'],
              'calibration': {'method': 'leave-source-task-out top1; relevant = same strategy and same recovery action type; max F1, higher threshold wins ties; training items only',
                              'proxy_positive_count': positives, 'selected': selected,
                              'historical_proxy_reference': {'positive_count': reference_positives, 'selected': reference}},
              'admission_threshold': selected['threshold'],
              'store_probability_mean': float(np.mean([r['store_probability'] for r in rows])),
              'script_sha256': file_hash(__file__)}
    result['files'] = {name: file_hash(out/name) for name in ('embeddings.npy', 'index.json', 'calibration-pairs.npz')}
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
