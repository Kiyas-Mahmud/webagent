"""Read-only post-hoc audit of saved InternVL assessments; no inference or score repair."""
from __future__ import annotations
import argparse,csv,json,re
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from web_agent.eval.task1.core import read,file_hash,load_prepared,request_key,parse_response,write_new
from web_agent.eval.task1.profiles import ROWS

PATTERNS={
 'null_value_mentioned':r'\bnull\b',
 'no_change_language':r'\bno (?:\w+ ){0,3}(?:change|progress)\b|\bunchanged\b|\bidentical\b|\bsame (?:page|screen)',
 'uncertainty_language':r'insufficient|not enough|cannot|can.not|unable|unclear|unknown|not .*?(?:clear|specified)|does not provide',
 'specific_action_or_target_mentioned':r'specific (?:action|target)|not (?:detailed|specified)|unspecified',
}

def inspect_response(raw,phase):
    """Describe contract failures without accepting/repairing rejected predictions."""
    try:
        pairs=json.loads(raw,object_pairs_hook=lambda x:x)
        value=json.loads(raw)
    except (ValueError,TypeError):
        return {'syntax':'not_bare_json','defects':['not_bare_json'],'declared_outcome':None,'reason':''}
    if not isinstance(value,dict):return {'syntax':'json','defects':['not_object'],'declared_outcome':None,'reason':''}
    keys=[k for k,v in pairs]; defects=[]
    if len(keys)!=len(set(keys)):defects.append('duplicate_keys')
    required={'outcome_label','reason'}|({'failure_type_4'} if phase=='interaction_assessment' else set())
    defects.extend('missing:'+k for k in sorted(required-value.keys()))
    defects.extend('extra:'+k for k in sorted(value.keys()-required))
    if 'reason' in value and not isinstance(value['reason'],str):defects.append('reason_not_string')
    if value.get('outcome_label') not in ('SUCCESS','FAILURE','ABSTAIN'):defects.append('invalid_outcome')
    if phase=='interaction_assessment' and 'failure_type_4' in value and value['failure_type_4'] not in ('NONE','PERCEPTION_ERROR','ACTION_MISMATCH','LOOP_DETECTED'):
        defects.append('invalid_failure_label')
    return {'syntax':'json','defects':defects,'declared_outcome':value.get('outcome_label'),
            'reason':value.get('reason','') if isinstance(value.get('reason',''),str) else ''}

def emit_csv(path,rows):
    with path.open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',required=True);args=a.parse_args()
    root=Path('.task1-assets/runs/task1-internvl-dual-v2');run=root/'pilot-reviewed-20260916';out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    manifest=read(root/'final-evidence-manifest-20260916.json')
    for path,digest in manifest['files'].items():
        if file_hash(path)!=digest:raise ValueError('Evidence changed: '+path)
    requests=load_prepared('docs/evidence/task1-assessment-prepared-v1');image_root=Path(read('configs/eval/task1/internvl_dual_lab_paths.json')['image_root'])
    frozen=read(root/'freeze.json');image_hashes=frozen['assets']['images']
    inputs=[];by_key={};seen=set()
    for r in requests:
        for image_key in ('before_image','after_image'):
            rel=r[image_key]
            if rel not in seen:
                assert file_hash(image_root/rel)==image_hashes[rel],rel;seen.add(rel)
        with Image.open(image_root/r['before_image']) as a,Image.open(image_root/r['after_image']) as b:
            aa=a.convert('RGB');bb=b.convert('RGB');identical=aa.size==bb.size and aa.tobytes()==bb.tobytes()
            item={'request_key':request_key(r),'case_id':r['case_id'],'phase':r['phase'],'action_type':r['executed_action']['type'],
                  'value_missing':r['executed_action']['value'] is None,'target_supplied':False,'history_supplied':False,
                  'task_description':r['task_description'],'before_image':r['before_image'],'after_image':r['after_image'],
                  'before_size':str(aa.size),'after_size':str(bb.size),'identical_decoded_pixels':identical}
        inputs.append(item);by_key[item['request_key']]=item
    rows=[];parse_details=[];reasons=Counter();groups=defaultdict(Counter);action_groups=defaultdict(Counter);pixel_groups=defaultdict(Counter)
    for spec in ROWS:
        for r in requests:
            key=request_key(r);p=run/spec.id/(key+'.json');saved=read(p);inp=by_key[key]
            groups[(spec.id,r['phase'])][saved['status']]+=1
            action_groups[(spec.id,r['phase'],inp['action_type'])][saved['status']]+=1
            pixel_groups[(spec.id,r['phase'],inp['identical_decoded_pixels'])][saved['status']]+=1
            if spec.output_mode!='generation':continue
            parsed=parse_response(saved['raw_response'],r['phase']);assert parsed['status']==saved['status'] and parsed['predictions']==saved['predictions']
            inspection=inspect_response(saved['raw_response'],r['phase'])
            themes=[name for name,pat in PATTERNS.items() if re.search(pat,inspection['reason'],re.I)] if saved['status']=='abstention' else []
            item={'row_id':spec.id,'request_key':key,'case_id':r['case_id'],'phase':r['phase'],'action_type':inp['action_type'],
                  'status':saved['status'],'syntax':inspection['syntax'],'defects':'|'.join(inspection['defects']),
                  'declared_outcome':inspection['declared_outcome'],'reason':inspection['reason'],'themes':'|'.join(themes),
                  'identical_decoded_pixels':inp['identical_decoded_pixels'],'token_count':saved['generated_token_count'],
                  'limit_hit':saved['limit_hit'],'ended_with_eos':saved['ended_with_eos'],
                  'raw_response':saved['raw_response'],'source_path':str(p),'source_sha256':file_hash(p)}
            rows.append(item)
            if saved['status']=='parse_error':parse_details.append(item)
            if saved['status']=='abstention':reasons[(spec.id,r['phase'],inspection['reason'])]+=1
    def summarize(mapping,columns):
        result=[]
        for key,counts in sorted(mapping.items()):
            n=sum(counts.values());result.append(dict(zip(columns,key),eligible=n,valid=counts['valid'],abstention=counts['abstention'],parse_error=counts['parse_error'],coverage=counts['valid']/n))
        return result
    out.mkdir(parents=True)
    emit_csv(out/'input_audit.csv',inputs)
    emit_csv(out/'response_audit.csv',rows)
    emit_csv(out/'parse_errors.csv',parse_details)
    emit_csv(out/'coverage_by_phase.csv',summarize(groups,['row_id','phase']))
    emit_csv(out/'coverage_by_action.csv',summarize(action_groups,['row_id','phase','action_type']))
    emit_csv(out/'coverage_by_pixel_change.csv',summarize(pixel_groups,['row_id','phase','identical_decoded_pixels']))
    emit_csv(out/'abstention_reasons.csv',[dict(row_id=s,phase=p,reason=r,count=n) for (s,p,r),n in sorted(reasons.items(),key=lambda kv:(kv[0][0],-kv[1],kv[0][1:]))])
    summary={'scope':'Post-hoc audit of saved pilot requests/responses and their already-bound images. No new predictions, prompt changes or rescoring.',
      'source_records':2520,'generation_records':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
      'parse_error_syntax':dict(Counter(r['syntax'] for r in parse_details)),
      'parse_error_defect_combinations':dict(Counter(r['defects'] for r in parse_details)),
      'parse_error_declared_outcomes':dict(Counter(str(r['declared_outcome']) for r in parse_details)),
      'accepted_abstentions_with_invalid_failure_label':sum(r['status']=='abstention' and 'invalid_failure_label' in r['defects'] for r in rows),
      'token_limit_hits':sum(r['limit_hit'] for r in rows),'generation_eos_endings':sum(r['ended_with_eos'] for r in rows),
      'null_values':sum(r['value_missing'] for r in inputs),'input_records':len(inputs),'unique_task_descriptions':len({r['task_description'] for r in inputs}),
      'generic_description_pattern_count':sum(bool(re.fullmatch(r'(Locate|Look for) .+ for (the current task|the next task step|the requested result|this page goal|the current page objective)\.',r['task_description'])) for r in inputs),
      'identical_pixels_by_phase':{p:sum(r['identical_decoded_pixels'] for r in inputs if r['phase']==p) for p in ('interaction_assessment','recovery_assessment')},
      'abstention_theme_occurrences':{n:sum(r['status']=='abstention' and n in r['themes'].split('|') for r in rows) for n in PATTERNS},
      'reason_theme_rules':PATTERNS,'theme_limitations':'Overlapping literal-text indicators of model-stated reasons, not verified causal labels or manual adjudication.',
      'frozen_files_verified_unchanged':len(manifest['files']),'bound_images_verified':len(seen),'model_calls':0,'script_sha256':file_hash(__file__)}
    write_new(out/'summary.json',summary)
    for path,digest in manifest['files'].items():assert file_hash(path)==digest,path
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
