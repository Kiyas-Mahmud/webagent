"""Independent native-campaign completeness audit and paired completion analysis."""
from collections import Counter
import argparse
import csv
import json
import math
import jsonschema
from pathlib import Path

from web_agent.eval.task1.core import file_hash, write_new
from web_agent.eval.task2.native_model import strict_object


def paired_test(improved, worsened):
    n=improved+worsened
    if not n:return 1.0
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(improved,worsened)+1))/(2**n))


def summarize_pairs(pairs):
    import numpy as np
    rng=np.random.default_rng(20250831)
    contrasts=[]
    for baseline,added in (('A','B'),('B','C')):
        eligible=[p for p in pairs if baseline in p['results'] and added in p['results']]
        differences=[int(p['results'][added]['completion'])-int(p['results'][baseline]['completion']) for p in eligible]
        groups={}
        for pair,diff in zip(eligible,differences):groups.setdefault(pair['task'],[]).append(diff)
        samples=np.zeros(10000)
        for values in groups.values():
            a=np.asarray(values);samples+=rng.choice(a,size=(10000,len(a)),replace=True).sum(axis=1)
        if differences:samples/=len(differences)
        improved=differences.count(1);worsened=differences.count(-1)
        contrasts.append({'contrast':added+'-'+baseline,'pairs':len(differences),'improved':improved,'worsened':worsened,
                          'difference':sum(differences)/len(differences) if differences else None,
                          'ci95':np.quantile(samples,[.025,.975]).tolist() if differences else None,
                          'exact_p':paired_test(improved,worsened)})
    running=0
    for rank,index in enumerate(sorted(range(2),key=lambda i:contrasts[i]['exact_p'])):
        running=max(running,min(1.,contrasts[index]['exact_p']*(2-rank)))
        contrasts[index]['holm_p']=running
    return contrasts


def audit(root):
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    for group in ('sources','bindings'):
        for path,digest in plan[group].items():
            if file_hash(path)!=digest:raise ValueError('Frozen identity changed: '+path)
    rows=[];pairs=[];errors=[];raw_count=0;parse_failures=Counter();memory_exposures=Counter()
    for block in plan['blocks']:
        if not block['eligible']:continue
        paired={'task':block['task'],'repeat':block['repeat_id'],'results':{}}
        for system in plan['systems']:
            folder=root/'episodes'/block['task']/str(block['repeat_id'])/system
            result_path=folder/'result.json'
            if not result_path.exists():errors.append(str(result_path));continue
            result=json.loads(result_path.read_text())
            assert result['system']==system and result['block']==block
            score=result['terminal_score']
            assert result['completion']==bool(score['terminated'] and not score.get('invalid_url') and score['raw_reward']==1.)
            assert sum(result['model_calls'].values())<=plan['settings']['budget']['max_model_calls']
            assert result['executor_requests']<=plan['settings']['budget']['max_executor_requests']
            if (folder/'infrastructure-error.json').exists():errors.append(str(folder/'infrastructure-error.json'))
            if system=='A':
                assert not list(folder.glob('assessment-*.json')) and not list(folder.glob('memory-*.json'))
            if system=='B':assert not list(folder.glob('memory-*.json'))
            raw_paths=list(folder.glob('actor-*-raw.json'))
            for p in raw_paths:
                payload=json.loads(p.read_text());raw_count+=1
                assert payload['backend']=='InternVL3.5-8B-HF' and payload['weight_variant']=='base'
                assert payload['model_calls']==1 and payload['token_count']==len(payload['generated_tokens'])
                if '-memory-' in p.name:memory_exposures[system]+=1
            started=list(folder.glob('actor-*-started.json'))
            assert len(started)==len(raw_paths), 'Unfinished model request'
            for started_path in folder.glob('actor-*-started.json'):
                name=started_path.name
                if '-memory-' in name:continue
                prefix=name[:-len('-started.json')]
                selected=folder/(prefix+'-memory-raw.json')
                if not selected.exists():selected=folder/(prefix+'-raw.json')
                raw=json.loads(selected.read_text())['raw_response']
                try:
                    value=strict_object(raw)
                    assert isinstance(value.get('action'),list) and len(value['action'])>=1
                    # JSON Schema validation is independent of the native Pydantic parser.
                    schema=json.loads(started_path.read_text())['request']['output_schema']
                    jsonschema.validate(value,schema)
                    expected=True
                except (ValueError,AssertionError,jsonschema.ValidationError):expected=False
                parsed=(folder/(prefix+'-parsed.json')).exists()
                if parsed!=expected:
                    # Deadline rejection can discard a syntactically valid late response.
                    error_path=folder/(prefix+'-error.json')
                    detail=json.loads(error_path.read_text())['error'] if error_path.exists() else ''
                    assert 'deadline' in detail.lower() or 'budget' in detail.lower(),(selected,parsed,expected,detail)
                if not parsed:parse_failures[system]+=1
            assert len(raw_paths)==result['model_calls'].get('actor',0)+result['model_calls'].get('memory_actor',0)
            assert len(list(folder.glob('assessment-*.json')))==result['model_calls'].get('assessment',0)
            assert len(list(folder.glob('memory-*.json')))==result['model_calls'].get('memory_query',0)
            actions=list(folder.glob('action-*.json'))
            assert len(actions)==result['agent_steps']
            for action_path in actions:
                event=json.loads(action_path.read_text())
                if event['native_output']:
                    selection=event['native_selection'];assert selection['selected_index']==0
                    proposed=json.loads((folder/(selection['actor_request_id']+'-parsed.json')).read_text())['native_proposal']['action']
                    assert selection['proposed_actions']==len(proposed) and selection['deferred_actions']==len(proposed)-1
                    assert event['native_output']['action']==proposed[:1], 'Native first-action selection changed the issued action'
            paired['results'][system]=result;rows.append(result)
        pairs.append(paired)
    if errors:raise ValueError('Missing/invalid episodes: '+str(errors))
    contrasts=summarize_pairs(pairs)
    summary=[]
    for system in plan['systems']:
        selected=[r for r in rows if r['system']==system]
        summary.append({'system':system,'completed':sum(r['completion'] for r in selected),'episodes':len(selected),
            'model_calls':sum(sum(r['model_calls'].values()) for r in selected),
            'executor_requests':sum(r['executor_requests'] for r in selected),
            'executed_actions':sum(r['counters']['executed'] for r in selected),
            'executed_recoveries':sum(r['counters']['executed_recoveries'] for r in selected),
            'memory_queries':sum(r['counters']['memory_queries'] for r in selected),
            'memory_exposures':memory_exposures[system],
            'invalid_actor_outputs':parse_failures[system],
            'elapsed_seconds':sum(r['elapsed_seconds'] for r in selected)})
    result={'status':'PASS','phase':plan['phase'],'episodes':len(rows),'summary':summary,'contrasts':contrasts,
            'excluded_blocks':[b for b in plan['blocks'] if not b['eligible']],
            'infrastructure_errors':errors,'raw_generations_audited':raw_count,
            'claim_scope':'Native Browser Use configured with local InternVL; package effects on fixed MiniWoB families; no claim of four isolated pillar gains'}
    write_new(root/'audit.json',result)
    with (root/'result_matrix.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write_new(root/'final-manifest.json',{'files':{str(p):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='status.json'}})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    print(json.dumps(audit(args.root),indent=2))
