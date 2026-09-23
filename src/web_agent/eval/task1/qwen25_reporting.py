"""Monitoring and independently verified exports for the Qwen2.5 seven-row profile."""
from collections import Counter
from pathlib import Path
import csv
import json
import math

from .core import file_hash, read, request_key, write_new
from .qwen25_profiles import PROFILE, ROWS, ROW_IDS, profile_spec
from .reporting import markdown


def monitor(run):
    root = Path(run)
    status = read(root / 'status.json') if (root / 'status.json').exists() else {'stage': 'not_started'}
    expected_total = int(status.get('expected_records', 2520))
    expected_per_row = expected_total // len(ROWS)
    print(PROFILE, json.dumps(status))
    print(f'{"Row":26} {"Saved":>9} {"Valid":>7} {"Parse":>7} {"Abstain":>8} {"Infra":>7} {"Interrupted":>12}')
    total = 0
    for spec in ROWS:
        counts = Counter()
        for path in (root / spec.id).glob('*.json'):
            if path.name == 'identity.json' or path.name.endswith('.started.json'):
                continue
            try:
                counts[read(path)['status']] += 1
            except (OSError, ValueError, KeyError):
                continue
        total += sum(counts.values())
        print(
            f'{spec.id:26} {sum(counts.values()):5}/{expected_per_row:<3} {counts["valid"]:7} '
            f'{counts["parse_error"]:7} {counts["abstention"]:8} '
            f'{counts["infrastructure_error"]:7} {counts["interrupted"]:12}'
        )
    print(f'Total: {total}/{expected_total} ({100 * total / expected_total:.1f}%). Valid means parseable, not correct.')


def independently_verify(report, records, requests, targets):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
    from web_agent.labels import EXECUTION_OUTCOME_INV, FAILURE_TYPE_INV

    checks = 0
    for system in ROW_IDS:
        data = records[system]
        if len(data) != len(requests):
            raise ValueError('Missing Qwen2.5 records in independent audit')
        if system == 'qwen25_heads':
            for record in data.values():
                logits = record['raw_logits']
                if record['phase'] == 'interaction_assessment':
                    outcome = EXECUTION_OUTCOME_INV[
                        max(range(2), key=lambda index: logits['outcome'][0][index])
                    ]
                    failure = FAILURE_TYPE_INV[
                        max(range(4), key=lambda index: logits['failure_type'][0][index])
                    ]
                    if failure != record['predictions']['failure_type_4']:
                        raise ValueError('Qwen2.5 head failure-logit mismatch')
                else:
                    value = logits['recovery_outcome']
                    while isinstance(value, list):
                        value = value[0]
                    outcome = 'SUCCESS' if value > 0 else 'FAILURE'
                if outcome != record['predictions']['outcome_label']:
                    raise ValueError('Qwen2.5 head outcome-logit mismatch')
        for phase in ('interaction_assessment', 'recovery_assessment'):
            subset = [request for request in requests if request['phase'] == phase]
            valid = [request for request in subset if data[request_key(request)]['status'] == 'valid']
            for field, metrics in report['systems'][phase][system]['endpoints'].items():
                truth = [targets[request_key(request)][field] for request in valid]
                predictions = [data[request_key(request)]['predictions'][field] for request in valid]
                if not truth:
                    if any(metrics[key] is not None for key in ('mcc', 'balanced_accuracy', 'macro_f1', 'accuracy')):
                        raise ValueError('Empty Qwen2.5 denominator scored as a real metric')
                    continue
                independent = {
                    'mcc': matthews_corrcoef(truth, predictions),
                    'balanced_accuracy': balanced_accuracy_score(truth, predictions),
                    'macro_f1': f1_score(
                        truth, predictions, labels=metrics['labels'], average='macro', zero_division=0,
                    ),
                    'accuracy': accuracy_score(truth, predictions),
                    'all_case_accuracy': sum(a == b for a, b in zip(truth, predictions)) / len(subset),
                }
                for key, value in independent.items():
                    if not math.isclose(value, metrics[key], abs_tol=1e-12):
                        raise ValueError('Independent Qwen2.5 metric mismatch: ' + system + '/' + phase + '/' + key)
                    checks += 1
    return {
        'status': 'PASS',
        'records': sum(map(len, records.values())),
        'independent_metric_checks': checks,
        'scope': 'Raw replay/adapter/input parity checked by runner; trained logit decoding and sklearn metric recomputation',
    }


def verified_report(report, records, requests, targets, output, csv_output, identity):
    audit = independently_verify(report, records, requests, targets)
    if report['status'] != 'complete':
        raise ValueError('Incomplete Qwen2.5 report cannot be published as complete')
    if output.exists() or csv_output.exists():
        raise FileExistsError('Qwen2.5 report/CSV directory already exists')
    output.mkdir(parents=True)
    csv_output.mkdir(parents=True)
    report['study'] = 'selected-model validation follow-up: Qwen2.5 base, trained decoder and trained heads'
    report['profile'] = profile_spec()
    write_new(output / 'results.json', report)
    write_new(output / 'binding.json', identity)
    audit['report_sha256'] = file_hash(output / 'results.json')
    write_new(output / 'verification.json', audit)
    (output / 'results.md').write_text(markdown(report) + '\n\n' + limitations() + '\n')
    summaries = []
    for row in ROWS:
        interaction = report['systems']['interaction_assessment'][row.id]
        recovery = report['systems']['recovery_assessment'][row.id]
        interaction_outcome = interaction['endpoints']['outcome_label']
        recovery_outcome = recovery['endpoints']['outcome_label']
        summaries.append({
            **row_metadata(row),
            'interaction_mcc': interaction_outcome['mcc'],
            'interaction_balanced_accuracy': interaction_outcome['balanced_accuracy'],
            'interaction_macro_f1': interaction_outcome['macro_f1'],
            'failure_category_macro_f1': interaction['endpoints']['failure_type_4']['macro_f1'],
            'interaction_all_case_accuracy': interaction_outcome['all_case_accuracy'],
            'interaction_valid': interaction['valid'],
            'interaction_eligible': interaction['eligible'],
            'recovery_mcc': recovery_outcome['mcc'],
            'recovery_macro_f1': recovery_outcome['macro_f1'],
            'recovery_all_case_accuracy': recovery_outcome['all_case_accuracy'],
            'recovery_valid': recovery['valid'],
            'recovery_eligible': recovery['eligible'],
        })
    csv_write(csv_output / 'result_matrix.csv', summaries)
    for phase, filename in [
        ('interaction_assessment', 'interaction_metrics.csv'),
        ('recovery_assessment', 'recovery_metrics.csv'),
    ]:
        rows = []
        for row in ROWS:
            data = report['systems'][phase][row.id]
            metrics = data['endpoints']['outcome_label']
            item = {
                **row_metadata(row),
                'phase': phase,
                **{key: metrics[key] for key in ('mcc', 'balanced_accuracy', 'macro_f1', 'accuracy', 'all_case_accuracy')},
                'valid_outputs': data['valid'],
                'eligible_cases': data['eligible'],
            }
            if phase == 'interaction_assessment':
                item['failure_category_macro_f1'] = data['endpoints']['failure_type_4']['macro_f1']
            for status in ('parse_error', 'abstention', 'infrastructure_error', 'interrupted', 'missing'):
                item[status] = data['statuses'].get(status, 0)
            for key in ('model_calls_known', 'model_calls_unknown_records', 'latency_seconds_total'):
                item[key] = data[key]
            item['token_limit_hits'] = sum(
                record.get('limit_hit', False)
                for record in records[row.id].values()
                if record['phase'] == phase
            )
            rows.append(item)
        csv_write(csv_output / filename, rows)
    pairs = []
    for name, comparison in report['paired_comparisons'].items():
        phase, endpoint, first, second = name.split('/')
        for metric in ('mcc', 'balanced_accuracy', 'macro_f1', 'accuracy'):
            interval = (comparison.get('intervals') or {}).get(metric)
            pairs.append({
                'phase': phase,
                'endpoint': endpoint,
                'first': first,
                'second': second,
                'metric': metric,
                'direction': 'second_minus_first',
                'pairs': comparison['pairs'],
                'task_groups': comparison.get('task_groups', 0),
                'difference': interval['difference'] if interval else None,
                'ci95_low': interval['ci95'][0] if interval else None,
                'ci95_high': interval['ci95'][1] if interval else None,
                'resamples': 10000,
                'analysis_seed': 20250831,
            })
    csv_write(csv_output / 'paired_differences.csv', pairs)
    (csv_output / 'README.md').write_text(
        '# Qwen2.5 selected-model Task 1 comparison\n\n'
        'Start with `result_matrix.csv`. Detailed phase metrics and nine declared paired contrasts '
        'are in the other CSV files.\n\nAll rates are fractions, not percentages. '
        'MCC/F1/balanced accuracy use valid outputs; all-case accuracy counts missing/invalid outputs '
        'as incorrect. Empty metrics mean unavailable, not zero.\n\n'
        + limitations() + '\n\nProducing report: ' + str(output.resolve()) + '\n'
    )


def row_metadata(row):
    return {
        'row_id': row.id,
        'method': row.method or 'trained_heads',
        'backend': 'Qwen2.5-VL-7B-Instruct',
        'weight_variant': row.weight_variant,
        'output_mode': row.output_mode,
        'seed': 42,
    }


def csv_write(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def limitations():
    return (
        'This is a follow-up on previously observed validation cases, not a new locked test. '
        'The six prompted configurations are adapted component instructions, not native browser agents. '
        'Base-versus-trained decoder contrasts hold prompts and generation settings fixed; '
        'head-versus-decoder contrasts also differ in assessment interface. '
        'The checkpoint was selected from the completed validation comparison before this follow-up. '
        'No live task completion, executed recovery, memory gain or confirmatory significance claim is established.'
    )
