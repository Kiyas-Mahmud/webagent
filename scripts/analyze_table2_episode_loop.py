"""Describe the completed MiniWoB loop from hash-verified saved evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.evaluation.resolve(), args.output.resolve()
    assert not out.exists()
    manifest = json.loads((root/'artifact-manifest.json').read_text())
    reads = {}

    def read(rel, lines=False):
        data = (root/rel).read_bytes()
        h = hashlib.sha256(data).hexdigest()
        assert h == manifest['files'][str(rel)], ('artifact changed', str(rel))
        reads[str(rel)] = h
        return [json.loads(l) for l in data.splitlines()] if lines else json.loads(data)

    rows = read(Path('results.json')); plan = read(Path('plan.json'))
    assert read(Path('audit.json'))['status'] == 'PASS' and len(rows) == 120
    details = []; systems = {}
    for system in plan['systems']:
        selected = [r for r in rows if r['system'] == system]
        raw_types = Counter(); assessment_flags = Counter(); first_probabilities = defaultdict(list)
        all_negative_after_completion = 0
        for row in selected:
            folder = Path(row['task'])/f"repeat-{row['repeat_id']}"/system
            def events(stream, kind):
                path = folder/'runtime'/f'{stream}.jsonl'
                return [e['payload'] for e in read(path, lines=True) if e['event_type'] == kind] if (root/path).exists() else []
            actions = events('actions', 'normal_action')
            recoveries = events('actions', 'recovery_action')
            assessments = [r['predicted_assessment'] for r in events('recoveries', 'recovery_attempt') if 'predicted_assessment' in r]
            for a in assessments:
                assessment_flags[f"resolved={a['predicted_failure_resolved']},progress={a['predicted_progress']}"] += 1
            if row['success'] and assessments and not any(a['predicted_failure_resolved'] or a['predicted_progress'] for a in assessments):
                all_negative_after_completion += 1
            if system == 'E1':
                for key, value in actions[0]['decision']['action_probabilities'].items():
                    first_probabilities[key].append(value)
            for p in (root/folder).glob('named-*-outputs/output-*.json'):
                r = read(p.relative_to(root)); raw = r['raw_response'].strip()
                if raw.startswith('```json') and raw.endswith('```'):
                    raw = raw[7:-3]
                try:
                    parsed = json.loads(raw)
                    kind = parsed.get('action_type') if isinstance(parsed, dict) else None
                    raw_types[kind or 'missing_action_type'] += 1
                except ValueError:
                    raw_types['invalid_json'] += 1
            details.append({'task': row['task'], 'repeat_id': row['repeat_id'], 'system': system,
                'success': row['success'], 'terminal_reason': row['summary']['terminal_reason'],
                'executor_requests': row['summary']['executor_steps'],
                'normal_action_types': [a['decision']['action_type'] for a in actions],
                'normal_execution_status': [a['execution']['status'] for a in actions],
                'recovery_action_types': [a['action']['action_type'] for a in recoveries],
                'recovery_execution_status': [a['execution']['status'] for a in recoveries],
                'browser_steps': len(row['environment_outcomes']), 'outcomes': row['environment_outcomes'],
                'recovery_assessments': assessments})
        systems[system] = {'episodes': len(selected), 'completed': sum(r['success'] for r in selected),
            'terminal_reasons': dict(Counter(r['summary']['terminal_reason'] for r in selected)),
            'maximum_executor_requests': max(r['summary']['executor_steps'] for r in selected),
            'maximum_model_calls': max(r['summary']['model_call_count'] for r in selected),
            'maximum_episode_seconds': max(r['summary']['elapsed_seconds'] for r in selected),
            'generated_action_types': dict(raw_types), 'assessment_flags': dict(assessment_flags),
            'completed_episodes_with_negative_assessments': all_negative_after_completion,
            'first_policy_probabilities': {k: {'min': min(v), 'mean': statistics.mean(v), 'max': max(v)}
                                           for k, v in first_probabilities.items()}}
    result = {'scope': 'Post-hoc loop diagnosis; no new evaluation or changed policy',
              'systems': systems, 'frozen_budgets': plan['runtime_protocol']['budgets'],
              'evaluation_episodes': len(rows), 'runtime_errors': sum('error' in r for r in rows),
              'new_model_calls': 0, 'new_browser_episodes': 0, 'memory_writes': 0}
    out.mkdir(parents=True)
    for name, value in [('summary.json', result), ('episode-traces.json', details), ('verified-inputs.json', reads)]:
        (out/name).write_text(json.dumps(value, indent=2)+'\n')
    shutil.copy2(__file__, out/'analyze_table2_episode_loop.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
