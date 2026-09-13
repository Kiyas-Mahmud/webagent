"""Read saved Table 2 evidence; no inference, browser, training or memory writes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def events(folder, stream, kind):
    p = folder / 'runtime' / (stream + '.jsonl')
    return [e['payload'] for line in p.read_text().splitlines()
            if (e := json.loads(line))['event_type'] == kind] if p.exists() else []


def context(receipt):
    before, sep, after = receipt['input_suffix'].partition('\nnext_action_request: ')
    assert sep
    return json.loads(before.removeprefix('\ncausal_recovery_context: ')), json.loads(after)


def comparable_input(receipt):
    c, request = context(receipt)
    # The memory schema and identity commitment differ by design. Keep all
    # observable controls, values, history, diagnosis and previous feedback.
    for k in ['schema', 'planner_context_sha256', 'retrieved_training_examples', 'memory_usage']:
        c.pop(k, None)
    return json.dumps([c, request], sort_keys=True).replace(':E2:', ':SYSTEM:').replace(':E3:', ':SYSTEM:')


def proposal(receipt):
    raw = receipt['raw_response'].strip()
    if raw.startswith('```json') and raw.endswith('```'):
        raw = raw[7:-3].strip()
    try:
        p = json.loads(raw)
        assert isinstance(p, dict)
        return {k: p.get(k) for k in ['action_type', 'target', 'bbox', 'value']}
    except (ValueError, AssertionError):
        return {'unparsed_raw': raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.evaluation.resolve(), args.output.resolve()
    assert not out.exists(), 'new evidence directory required'
    archive = read(root / 'artifact-manifest.json')
    for rel, h in archive['files'].items():
        assert sha(root / rel) == h, ('evaluation artifact changed', rel)
    plan = read(root / 'plan.json')
    assert read(root / 'audit.json')['status'] == 'PASS'
    config = plan['config']
    store = Path(config['asset_paths']['memory'])
    manifest = read(store / 'manifest.json')
    for rel, h in manifest['files'].items():
        assert sha(store / rel) == h, ('memory artifact changed', rel)
    material_path = Path(config['memory_material'])
    assert sha(material_path) == manifest['memory_input_sha256']
    items = [json.loads(l) for l in material_path.read_text().splitlines()]
    assert len(items) == 1974 and all(x['source_split'] == 'train' for x in items)
    materials = {x['memory_id']: x['material'] for x in items}
    index = read(store / 'index.json')
    vectors = np.load(store / 'embeddings.npy', mmap_mode='r')
    trace, comparisons = [], []
    candidate_counts, top_counts, projected_strategies, projected_actions = (Counter() for _ in range(4))
    query_vectors = []
    for block in plan['blocks']:
        folder = root / block['task'] / f"repeat-{block['repeat_id']}"
        first, first_actions = {}, {}
        for system in ['E2', 'E3']:
            sub = folder / system
            receipts = [read(p) for p in sorted((sub / 'named-recovery-outputs').glob('output-*.json'))]
            if receipts:
                first[system] = receipts[0]
                generated_plans = [p for p in events(sub, 'recoveries', 'recovery_plan')
                                   if p['final_decision']['strategy'] in ['REPLAN', 'ALTERNATIVE_TARGET']]
                first_actions[system] = [{k: a[k] for k in ['action_type', 'bbox', 'parameters']}
                                        for a in generated_plans[0]['plan']['actions']]
            if system != 'E3':
                continue
            plans = events(sub, 'recoveries', 'recovery_plan')
            generated = [p for p in plans if p['final_decision']['strategy'] in ['REPLAN', 'ALTERNATIVE_TARGET']]
            assert len(generated) == len(receipts)
            generations = {p['plan']['attempt_id']: r for p, r in zip(generated, receipts)}
            by_decision = {p['final_decision']['decision_id']: p for p in plans}
            assessments = {p.get('attempt', p)['attempt_id']: p.get('predicted_assessment')
                           for p in events(sub, 'recoveries', 'recovery_attempt')}
            actions = events(sub, 'actions', 'recovery_action')
            for q in events(sub, 'memory_queries', 'post_failure_query'):
                r, decision = q['query_result'], q['final_decision']
                p = by_decision[decision['decision_id']]['plan']
                receipt = generations.get(p['attempt_id'])
                vector = np.asarray(r['normalized_query_embedding'], dtype=np.float32)
                scores = np.clip(vectors @ (vector / np.linalg.norm(vector)), -1, 1)
                allowed = np.array([i for i, row in enumerate(index) if row['source_task'] != q['query']['task_id']])
                order = allowed[np.argsort(-scores[allowed], kind='stable')][:3]
                assert [index[i]['memory_id'] for i in order] == r['candidate_ids']
                np.testing.assert_allclose(scores[order], r['scores'], atol=1e-6, rtol=0)
                admitted = [index[i]['memory_id'] for i in order if scores[i] >= manifest['admission_threshold']]
                assert admitted == r['context_candidate_ids']
                for key in ['strategy', 'diagnosis', 'planned_action', 'incident_id', 'trigger_sources']:
                    assert decision[key] == q['shadow_decision'][key]
                if receipt:
                    c, _ = context(receipt)
                    assert c['retrieved_training_examples'] == decision['memory_experiences']
                execution = [a['execution'] for a in actions
                             if a['action']['recovery_attempt_id'] == p['attempt_id']]
                candidate_counts.update(r['candidate_ids']); top_counts.update(r['candidate_ids'][:1])
                projected_strategies.update(materials[m]['recovery_strategy'] for m in admitted)
                projected_actions.update(materials[m]['executed_recovery_action'] for m in admitted)
                query_vectors.append(vector)
                trace.append({'task': block['task'], 'repeat_id': block['repeat_id'],
                    'query_id': r['query_id'], 'attempt_id': p['attempt_id'],
                    'embedding_action_type': q['embedding_request']['post_action_input']['executed_action']['action_type'],
                    'normalized_query_embedding_sha256': r['normalized_query_embedding_sha256'],
                    'candidates': [{'memory_id': mid, 'score': score, 'admitted': mid in admitted}
                                   for mid, score in zip(r['candidate_ids'], r['scores'])],
                    'source_examples': decision['memory_experiences'],
                    'strategy_preserved': decision['strategy'], 'generation_included': receipt is not None,
                    'proposal': proposal(receipt) if receipt else None,
                    'generation_status': receipt['status'] if receipt else None,
                    'rejection_stage': p['rejection_stage'], 'rejection_reason': p['rejection_reason'],
                    'planned_actions': p['actions'], 'execution': execution,
                    'assessment': assessments.get(p['attempt_id']), 'completion': read(sub / 'result.json')['success']})
        row = {**block, 'E2_completion': read(folder / 'E2/result.json')['success'],
               'E3_completion': read(folder / 'E3/result.json')['success']}
        if len(first) == 2:
            a, b = first['E2'], first['E3']
            row.update(comparable=(a['screenshot_sha256'] == b['screenshot_sha256']
                and a['prompt_sha256'] == b['prompt_sha256'] and comparable_input(a) == comparable_input(b)),
                E2_proposal=proposal(a), E3_proposal=proposal(b),
                E2_status=a['status'], E3_status=b['status'],
                E2_concrete_actions=first_actions['E2'], E3_concrete_actions=first_actions['E3'],
                same_concrete_actions=first_actions['E2'] == first_actions['E3'],
                same_proposal=proposal(a) == proposal(b),
                same_resolved_action=a.get('resolved') == b.get('resolved'))
        else:
            row.update(comparable=False, reason='No paired planner generation; registered BACKTRACK path')
        comparisons.append(row)
    assert len(trace) == 46 and sum(t['generation_included'] for t in trace) == 42
    # Reuse prepared training metadata, not original dataset rows or images.
    prep = Path('/home/aiub/kiyas/table2-inputs/kaggle-p4-output-178c24a/table2-p4-prepare-only-v1/preparation')
    queue_path = prep / 'review_queue.jsonl'
    assert sha(queue_path) == read(prep / 'preparation_manifest.json')['files']['review_queue.jsonl']['sha256']
    queue = {x['queue_id']: x for line in queue_path.read_text().splitlines() if (x := json.loads(line))['queue_id'] in materials}
    assert len(queue) == len(materials)
    for mid, m in materials.items():
        assert queue[mid]['source_split'] == 'train'
        assert queue[mid]['memory_item_source_material'] == m
    value_evidence_path = Path(config['asset_paths']['export']) / 'training_action_value_evidence.json'
    assert sha(value_evidence_path) == plan['asset_sha256'][str(value_evidence_path)]
    value_evidence = read(value_evidence_path)
    calibration = read(store / 'calibration.json')
    pairs = np.load(store / 'calibration-pairs.npz')
    y = pairs['labels']; positives = int(y.sum()); all_f1 = 2 * positives / (len(y) + positives)
    # Training-only geometry describes concentration, without choosing a new metric.
    matrix = vectors @ vectors.T
    tasks = np.array([r['source_task'] for r in index])
    nearest = []
    for i in range(len(index)):
        allowed = np.flatnonzero(tasks != tasks[i])
        nearest.append(int(allowed[np.argmax(matrix[i, allowed])]))
    train_counts = Counter(index[i]['memory_id'] for i in nearest)
    dominant = top_counts.most_common(1)[0][0]
    comparable = [c for c in comparisons if c['comparable']]
    summary = {'scope': 'Post-hoc descriptive diagnosis; no new model calls, browser episodes, tuning or memory writes',
        'status': 'SAVED_RETRIEVAL_AND_DELIVERY_REPLAY_PASS', 'queries': len(trace),
        'generation_exposures': 42, 'candidate_frequency': dict(candidate_counts), 'rank_one_frequency': dict(top_counts),
        'retrieved_strategy_slots': dict(projected_strategies), 'retrieved_action_slots': dict(projected_actions),
        'comparable_first_generations': len(comparable),
        'changed_first_proposals': sum(not c['same_proposal'] for c in comparable),
        'changed_concrete_actions_when_both_resolved': sum(c['E2_status'] == c['E3_status'] == 'RESOLVED'
            and not c['same_concrete_actions'] for c in comparable),
        'generated_action_types': dict(Counter(t['proposal']['action_type'] for t in trace if t['proposal'])),
        'embedding_action_types': dict(Counter(t['embedding_action_type'] for t in trace)),
        'unique_query_vectors': len({t['normalized_query_embedding_sha256'] for t in trace}),
        'first_status_transitions': dict(Counter(c['E2_status']+' -> '+c['E3_status'] for c in comparable)),
        'material': {'count': len(items), 'prepared_source_matches': len(queue),
            'available_fields': {k: sum(bool(m.get(k)) for m in materials.values()) for k in
                ['recovery_action_value', 'reflection_text', 'task_description', 'website_domain']},
            'strategy_counts': dict(Counter(m['recovery_strategy'] for m in materials.values())),
            'recovery_action_counts': dict(Counter(m['executed_recovery_action'] for m in materials.values())),
            'source_action_value_evidence': value_evidence},
        'calibration': {'selected': calibration['selected'], 'all_admit_f1': all_f1,
            'positive_pairs': positives, 'pairs': len(y),
            'f1_increment_over_all_admit': calibration['selected']['f1']-all_f1},
        'training_geometry': {'dominant_evaluation_neighbor_training_top1_count': train_counts[dominant],
            'training_top1_unique': len(train_counts), 'training_top1_most_common': train_counts.most_common(5),
            'mean_pairwise_cosine_excluding_self': float((matrix.sum()-len(index))/(len(index)*(len(index)-1)))},
        'model_calls': 0, 'browser_episodes': 0, 'memory_writes': 0,
        'original_dataset_rows_read': 0, 'images_read_for_content': 0,
        'archive_files_hash_verified': len(archive['files'])}
    out.mkdir(parents=True)
    for name, data in [('summary.json', summary), ('query-traces.json', trace), ('paired-first-proposals.json', comparisons)]:
        (out / name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    shutil.copy2(__file__, out / 'analyze_table2_p4.py')
    inputs = [root/'artifact-manifest.json', store/'manifest.json', material_path, queue_path, value_evidence_path]
    (out/'input-identities.json').write_text(json.dumps({str(p): sha(p) for p in inputs}, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
