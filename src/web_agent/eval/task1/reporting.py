"""Completeness-aware scores and paired task-cluster intervals."""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from .core import SYSTEMS, PHASES, read, request_key, write_new
from web_agent.labels import FAILURE_TYPE

def metrics(truth, predictions, labels):
    """Fixed label macro-F1, supported-class balanced accuracy, generalized MCC."""
    if not truth: return {'mcc':None,'balanced_accuracy':None,'macro_f1':None,'accuracy':None}
    mat=np.zeros((len(labels),len(labels)),dtype=float); index={k:i for i,k in enumerate(labels)}
    for t,p in zip(truth,predictions): mat[index[t],index[p]]+=1
    return matrix_metrics(mat)

def matrix_metrics(mat):
    t=mat.sum(axis=-1); p=mat.sum(axis=-2); n=t.sum(axis=-1); c=np.trace(mat,axis1=-2,axis2=-1)
    denom=np.sqrt((n*n-(p*p).sum(axis=-1))*(n*n-(t*t).sum(axis=-1)))
    with np.errstate(divide='ignore',invalid='ignore'):
        mcc=np.divide(c*n-(t*p).sum(axis=-1),denom,out=np.zeros_like(n),where=denom>0)
        tp=np.diagonal(mat,axis1=-2,axis2=-1)
        recall=np.divide(tp,t,out=np.zeros_like(t),where=t>0)
        bal=recall.sum(axis=-1)/np.maximum((t>0).sum(axis=-1),1)
        f1=np.divide(2*tp,t+p,out=np.zeros_like(t),where=t+p>0).mean(axis=-1)
        acc=c/n
    result={'mcc':mcc,'balanced_accuracy':bal,'macro_f1':f1,'accuracy':acc}
    return {k:float(v) if np.ndim(v)==0 else v for k,v in result.items()}

def paired_interval(rows, labels, seed=20250831, resamples=10000):
    if not rows: return {'pairs':0,'intervals':None}
    groups=sorted({r[0] for r in rows}); gi={g:i for i,g in enumerate(groups)}
    li={v:i for i,v in enumerate(labels)}
    counts=np.zeros((len(groups),2,len(labels),len(labels)))
    for group,t,a,b in rows:
        counts[gi[group],0,li[t],li[a]]+=1
        counts[gi[group],1,li[t],li[b]]+=1
    rng=np.random.default_rng(seed); values={k:[] for k in ('mcc','balanced_accuracy','macro_f1','accuracy')}
    for start in range(0,resamples,100):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(100,resamples-start))
        mats=np.einsum('rg,gslm->rslm',weights,counts)
        a=matrix_metrics(mats[:,0]); b=matrix_metrics(mats[:,1])
        for k in values: values[k].extend((b[k]-a[k]).tolist())
    a=matrix_metrics(counts[:,0].sum(axis=0)); b=matrix_metrics(counts[:,1].sum(axis=0))
    return {'pairs':len(rows),'task_groups':len(groups),'resamples':resamples,'seed':seed,
            'direction':'second_minus_first','intervals':{k:{'difference':b[k]-a[k],
                'ci95':np.quantile(v,[.025,.975]).tolist()} for k,v in values.items()}}

def collect_records(run, system, requests, binding):
    folder=Path(run)/system
    if not (folder/'identity.json').exists(): return {}
    from .core import sha,encode
    expected={'system':system,'binding':binding,'requests_sha256':sha(encode(requests))}
    if read(folder/'identity.json')!=expected: raise ValueError('Result identity mismatch')
    allowed={request_key(r):r for r in requests}; records={}
    for path in folder.glob('*.json'):
        if path.name=='identity.json' or path.name.endswith('.started.json'): continue
        row=read(path); key=row['request_key']
        if key not in allowed or path.stem!=key or key in records: raise ValueError('Unexpected or duplicate result')
        req=allowed[key]
        if any(row[k]!=req[k] for k in ('case_id','task_id','phase')) or row['system']!=system:
            raise ValueError('Result metadata mismatch')
        if row['status'] not in ('valid','abstention','parse_error','infrastructure_error','interrupted','unsupported'):
            raise ValueError('Unknown status')
        if row['status']=='valid':
            pred=row['predictions']
            if pred.get('outcome_label') not in ('SUCCESS','FAILURE'): raise ValueError('Invalid normalized outcome')
            if req['phase']==PHASES[0] and pred.get('failure_type_4') not in FAILURE_TYPE:
                raise ValueError('Invalid normalized category')
        records[key]=row
    return records

def score(requests, targets, records, systems=SYSTEMS, contrasts=None):
    summaries={}; comparisons={}; complete=True
    for phase in PHASES:
        subset=[r for r in requests if r['phase']==phase]; summaries[phase]={}
        for system in systems:
            data=records.get(system,{})
            status=Counter(data.get(request_key(r),{}).get('status','missing') for r in subset)
            if status['missing'] or status['infrastructure_error'] or status['interrupted']: complete=False
            valid=[r for r in subset if data.get(request_key(r),{}).get('status')=='valid']
            summary={'eligible':len(subset),'valid':len(valid),'statuses':dict(status),'endpoints':{}}
            for field,labels in [('outcome_label',('SUCCESS','FAILURE'))]+(
                    [('failure_type_4',tuple(FAILURE_TYPE))] if phase==PHASES[0] else []):
                truth=[targets[request_key(r)][field] for r in valid]
                pred=[data[request_key(r)]['predictions'][field] for r in valid]
                m=metrics(truth,pred,labels)
                m['all_case_accuracy']=sum(a==b for a,b in zip(truth,pred))/len(subset) if subset else None
                m['valid_denominator']=len(valid)
                m['per_class_recall']={label:(sum(a==label and b==label for a,b in zip(truth,pred))/truth.count(label)
                                             if label in truth else None) for label in labels}
                m['confusion_matrix']=[[sum(a==x and b==y for a,b in zip(truth,pred)) for y in labels] for x in labels]
                m['labels']=labels; summary['endpoints'][field]=m
            times=[data[request_key(r)].get('latency_seconds') for r in subset if request_key(r) in data]
            calls=[data[request_key(r)].get('model_calls') for r in subset if request_key(r) in data]
            summary['latency_seconds_total']=sum(t for t in times if t is not None)
            summary['model_calls_known']=sum(c for c in calls if c is not None)
            summary['model_calls_unknown_records']=sum(c is None for c in calls)
            summaries[phase][system]=summary
        for a,b in (combinations(systems,2) if contrasts is None else contrasts):
            for field,labels in [('outcome_label',('SUCCESS','FAILURE'))]+(
                    [('failure_type_4',tuple(FAILURE_TYPE))] if phase==PHASES[0] else []):
                paired=[]
                for r in subset:
                    key=request_key(r); x=records.get(a,{}).get(key,{}); y=records.get(b,{}).get(key,{})
                    if x.get('status')==y.get('status')=='valid':
                        paired.append((r['task_id'],targets[key][field],x['predictions'][field],y['predictions'][field]))
                comparisons[f'{phase}/{field}/{a}/{b}']=paired_interval(paired,labels)
    return {'status':'complete' if complete else 'incomplete','study':'balanced validation pilot',
            'metrics_on':'valid responses; all_case_accuracy includes missing/invalid as incorrect',
            'systems':summaries,'paired_comparisons':comparisons}

def markdown(report):
    lines=['# Task 1 mini comparison results','',f"Status: **{report['status']}**. These are validation-pilot results.",
           'No full-validation scores are substituted. Metrics use the displayed valid denominator.', '']
    for phase,title in zip(PHASES,('Table A — interaction assessment','Table B — recovery assessment')):
        lines += ['## '+title,'','| System | MCC | Balanced accuracy | Macro-F1 | Failure macro-F1 | All-case accuracy | Valid / eligible |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        def fmt(x): return 'N/R' if x is None else f'{x:.4f}'
        for system,s in report['systems'][phase].items():
            m=s['endpoints']['outcome_label']; f=s['endpoints'].get('failure_type_4',{}).get('macro_f1')
            lines.append(f"| {system} | {fmt(m['mcc'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['macro_f1'])} | {fmt(f)} | {fmt(m['all_case_accuracy'])} | {s['valid']} / {s['eligible']} |")
        lines.append('')
    lines += ['Detailed JSON includes statuses, confusion matrices, calls, latency and paired task-group 95% intervals.',
              'No confirmatory significance claim or live browser-completion claim is made.']
    return '\n'.join(lines)+'\n'
