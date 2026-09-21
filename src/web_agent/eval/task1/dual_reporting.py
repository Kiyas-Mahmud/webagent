"""Read-only monitoring and independently verified exports for seven InternVL rows."""
from collections import Counter
from pathlib import Path
import csv
import json
import math
from .profiles import PROFILE, ROWS, ROW_IDS, profile_spec
from .core import read, write_new, request_key, file_hash
from .reporting import markdown


def monitor(run):
    root = Path(run)
    status = read(root/'status.json') if (root/'status.json').exists() else {'stage':'not_started'}
    print(PROFILE, json.dumps(status))
    print(f'{"Row":26} {"Saved":>9} {"Valid":>7} {"Parse":>7} {"Abstain":>8} {"Infra":>7} {"Interrupted":>12}')
    total = 0
    for spec in ROWS:
        counts = Counter()
        for p in (root/spec.id).glob('*.json'):
            if p.name == 'identity.json' or p.name.endswith('.started.json'): continue
            try: counts[read(p)['status']] += 1
            except (OSError,ValueError,KeyError): continue
        total += sum(counts.values())
        print(f'{spec.id:26} {sum(counts.values()):5}/360 {counts["valid"]:7} {counts["parse_error"]:7} '
              f'{counts["abstention"]:8} {counts["infrastructure_error"]:7} {counts["interrupted"]:12}')
    print(f'Total: {total}/2520 ({100*total/2520:.1f}%). Valid means parseable, not correct.')


def independently_verify(report, records, requests, targets):
    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, accuracy_score
    from web_agent.labels import EXECUTION_OUTCOME_INV, FAILURE_TYPE_INV
    checks = 0
    for system in ROW_IDS:
        data = records[system]
        if len(data) != len(requests): raise ValueError('Missing records in independent audit')
        if system == 'internvl_heads':
            for r in data.values():
                logits = r['raw_logits']
                if r['phase']=='interaction_assessment':
                    outcome = EXECUTION_OUTCOME_INV[max(range(2),key=lambda k:logits['outcome'][0][k])]
                    failure = FAILURE_TYPE_INV[max(range(4),key=lambda k:logits['failure_type'][0][k])]
                    if failure != r['predictions']['failure_type_4']: raise ValueError('Head failure-logit mismatch')
                else:
                    value = logits['recovery_outcome']
                    while isinstance(value,list): value = value[0]
                    outcome = 'SUCCESS' if value > 0 else 'FAILURE'
                if outcome != r['predictions']['outcome_label']: raise ValueError('Head outcome-logit mismatch')
        for phase in ('interaction_assessment','recovery_assessment'):
            subset = [r for r in requests if r['phase']==phase]
            valid = [r for r in subset if data[request_key(r)]['status']=='valid']
            for field,m in report['systems'][phase][system]['endpoints'].items():
                truth = [targets[request_key(r)][field] for r in valid]
                pred = [data[request_key(r)]['predictions'][field] for r in valid]
                if not truth:
                    if any(m[k] is not None for k in ('mcc','balanced_accuracy','macro_f1','accuracy')):
                        raise ValueError('Empty denominator scored as real metric')
                    continue
                other = {'mcc':matthews_corrcoef(truth,pred),
                         'balanced_accuracy':balanced_accuracy_score(truth,pred),
                         'macro_f1':f1_score(truth,pred,labels=m['labels'],average='macro',zero_division=0),
                         'accuracy':accuracy_score(truth,pred),
                         'all_case_accuracy':sum(a==b for a,b in zip(truth,pred))/len(subset)}
                for key,value in other.items():
                    if not math.isclose(value,m[key],abs_tol=1e-12):
                        raise ValueError('Independent metric mismatch: '+system+'/'+phase+'/'+key)
                    checks += 1
    return {'status':'PASS','records':sum(map(len,records.values())),
            'independent_metric_checks':checks,'scope':'Raw replay/adapter/input parity checked by runner; trained logit decoding and sklearn metric recomputation'}


def verified_report(report, records, requests, targets, output, csv_output, identity):
    audit = independently_verify(report,records,requests,targets)
    if report['status'] != 'complete': raise ValueError('Incomplete report cannot be published as complete')
    # No overwrites or partial replacement of the Qwen exports.
    if output.exists() or csv_output.exists(): raise FileExistsError('Report/CSV directory already exists')
    output.mkdir(parents=True); csv_output.mkdir(parents=True)
    report['study'] = 'follow-up validation pilot: InternVL base, trained decoder and trained heads'
    report['profile'] = profile_spec()
    write_new(output/'results.json',report)
    write_new(output/'binding.json',identity)
    audit['report_sha256'] = file_hash(output/'results.json')
    write_new(output/'verification.json',audit)
    (output/'results.md').write_text(markdown(report)+'\n\n'+limitations()+'\n')
    summaries = []
    for row in ROWS:
        a = report['systems']['interaction_assessment'][row.id]
        b = report['systems']['recovery_assessment'][row.id]
        am = a['endpoints']['outcome_label']; bm = b['endpoints']['outcome_label']
        summaries.append({**row_metadata(row), 'interaction_mcc':am['mcc'],
            'interaction_balanced_accuracy':am['balanced_accuracy'],'interaction_macro_f1':am['macro_f1'],
            'failure_category_macro_f1':a['endpoints']['failure_type_4']['macro_f1'],
            'interaction_all_case_accuracy':am['all_case_accuracy'], 'interaction_valid':a['valid'],
            'interaction_eligible':a['eligible'],'recovery_mcc':bm['mcc'],'recovery_macro_f1':bm['macro_f1'],
            'recovery_all_case_accuracy':bm['all_case_accuracy'],'recovery_valid':b['valid'],'recovery_eligible':b['eligible']})
    csv_write(csv_output/'result_matrix.csv',summaries)
    for phase,filename in [('interaction_assessment','interaction_metrics.csv'),('recovery_assessment','recovery_metrics.csv')]:
        rows=[]
        for row in ROWS:
            data=report['systems'][phase][row.id];m=data['endpoints']['outcome_label']
            item={**row_metadata(row),'phase':phase,**{k:m[k] for k in ('mcc','balanced_accuracy','macro_f1','accuracy','all_case_accuracy')},
                  'valid_outputs':data['valid'],'eligible_cases':data['eligible']}
            if phase=='interaction_assessment': item['failure_category_macro_f1']=data['endpoints']['failure_type_4']['macro_f1']
            for status in ('parse_error','abstention','infrastructure_error','interrupted','missing'):
                item[status]=data['statuses'].get(status,0)
            for key in ('model_calls_known','model_calls_unknown_records','latency_seconds_total'): item[key]=data[key]
            item['token_limit_hits']=sum(r.get('limit_hit',False) for r in records[row.id].values() if r['phase']==phase)
            rows.append(item)
        csv_write(csv_output/filename,rows)
    pairs=[]
    for name,comparison in report['paired_comparisons'].items():
        phase,endpoint,first,second=name.split('/')
        for metric in ('mcc','balanced_accuracy','macro_f1','accuracy'):
            interval=(comparison.get('intervals') or {}).get(metric)
            pairs.append({'phase':phase,'endpoint':endpoint,'first':first,'second':second,'metric':metric,
                'direction':'second_minus_first','pairs':comparison['pairs'],'task_groups':comparison.get('task_groups',0),
                'difference':interval['difference'] if interval else None,'ci95_low':interval['ci95'][0] if interval else None,
                'ci95_high':interval['ci95'][1] if interval else None,'resamples':10000,'analysis_seed':20250831})
    csv_write(csv_output/'paired_differences.csv',pairs)
    (csv_output/'README.md').write_text('# InternVL dual comparison\n\nStart with `result_matrix.csv`. '
        'Detailed phase metrics and nine declared paired contrasts are in the other CSV files.\n\n'
        'All rates are fractions, not percentages. MCC/F1/balanced accuracy use valid outputs; '
        'all-case accuracy counts missing/invalid outputs as incorrect. Empty metrics mean unavailable, not zero.\n\n'
        +limitations()+'\n\nProducing report: '+str(output.resolve())+'\n')


def row_metadata(row):
    return {'row_id':row.id,'method':row.method or 'trained_heads','backend':'InternVL3.5-8B-HF',
            'weight_variant':row.weight_variant,'output_mode':row.output_mode,'seed':42}


def csv_write(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def limitations():
    return ('This is a follow-up on previously observed validation cases, not a new locked test. '
            'The six prompted configurations are adapted component instructions, not native browser agents. '
            'Base-versus-trained decoder contrasts hold the prompt and generation settings fixed; '
            'head-versus-decoder contrasts also differ in assessment interface. '
            'No live task completion, executed recovery, memory gain or confirmatory significance claim is established.')
