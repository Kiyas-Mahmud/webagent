"""Replay interface evidence and report all matched MiniWoB outcomes."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import traceback


def log_matches(raw,logged):
    if isinstance(logged,dict) and set(logged)=={'redacted','sha256'} and logged['redacted'] is True:
        return not isinstance(raw,(dict,list)) and hashlib.sha256(str(raw).encode()).hexdigest()==logged['sha256']
    if isinstance(raw,dict) and isinstance(logged,dict):
        return set(raw)==set(logged) and all(log_matches(raw[k],logged[k]) for k in raw)
    if isinstance(raw,list) and isinstance(logged,list):
        return len(raw)==len(logged) and all(log_matches(a,b) for a,b in zip(raw,logged))
    return raw==logged


def paired_interval(pairs, *, samples=10000,seed=20250831):
    """Resample matched resets within each fixed family, retaining its sample size."""
    if not pairs: return [None,None]
    groups=defaultdict(list)
    for task,value in pairs: groups[task].append(value)
    rng=random.Random(seed); estimates=[]
    for _ in range(samples):
        total=sum(sum(rng.choice(values) for _ in values) for _,values in sorted(groups.items()))
        estimates.append(total/len(pairs))
    estimates.sort()
    def quantile(p):
        index=(len(estimates)-1)*p; lower=int(index); upper=min(lower+1,len(estimates)-1)
        return estimates[lower]+(index-lower)*(estimates[upper]-estimates[lower])
    return [quantile(.025),quantile(.975)]


def analyze(plan,rows):
    keys=[(r['task'],r['repeat_id'],r['system']) for r in rows if 'system' in r]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate episode result')
    by={key:r for key,r in zip(keys,[r for r in rows if 'system' in r])}
    excluded={(r['task'],r['repeat_id']) for r in rows if r.get('status')=='EXCLUDED_TRAIN_OVERLAP'}
    all_blocks=[(b['task'],b['repeat_id']) for b in plan['blocks']]
    if not excluded <= set(all_blocks): raise ValueError('unknown excluded block')
    included=[b for b in all_blocks if b not in excluded]
    expected={(t,r,s) for t,r in included for s in plan['systems']}
    valid=[(t,r) for t,r in included if all((t,r,s) in by and 'summary' in by[t,r,s]
        and by[t,r,s]['summary']['valid_for_primary'] and not by[t,r,s]['summary']['environment_failure']
        for s in plan['systems'])]
    contrasts=plan.get('analysis',{}).get('contrasts',[['E1','E2'],['E2','E3']])
    secondary=plan.get('analysis',{}).get('secondary_contrasts',[])
    if len(contrasts)!=2 or any(a not in plan['systems'] or b not in plan['systems'] or a==b for a,b in contrasts+secondary):
        raise ValueError('invalid configured contrasts')
    comparisons=[]
    for left,right in contrasts+secondary:
        pairs=[(t,int(by[t,r,right]['success'])-int(by[t,r,left]['success'])) for t,r in valid]
        improved=sum(v==1 for _,v in pairs); worsened=sum(v==-1 for _,v in pairs); d=improved+worsened
        p=min(1.,2*sum(math.comb(d,i) for i in range(min(improved,worsened)+1))/2**d) if d else 1.
        comparisons.append({'contrast':right+' minus '+left,'paired_instances':len(pairs),
            'improved':improved,'worsened':worsened,'completion_difference':(improved-worsened)/len(pairs) if pairs else None,
            'ci95':paired_interval(pairs),'exact_two_sided_p':p if pairs else None})
    previous=0
    for rank,i in enumerate(sorted(range(2),key=lambda i:comparisons[i]['exact_two_sided_p'] if comparisons[i]['exact_two_sided_p'] is not None else 2)):
        c=comparisons[i]
        if c['exact_two_sided_p'] is None: c.update(holm_p=None,conclusion='NOT_EVALUABLE');continue
        previous=max(previous,min(1.,(2-rank)*c['exact_two_sided_p']));c['holm_p']=previous
        c['conclusion']='POSITIVE_INCREMENT_SUPPORTED' if c['completion_difference']>0 and previous<.05 else 'IMPROVEMENT_NOT_DEMONSTRATED'
    secondary_results=comparisons[2:]
    comparisons=comparisons[:2]
    for c in secondary_results:
        c.update(role='secondary', multiplicity='not included in the two primary Holm tests')
    systems=[]
    for s in plan['systems']:
        good=[by[t,r,s] for t,r in valid]
        systems.append({'system':s,'episodes':len(good),'completed':sum(r['success'] for r in good),
            **{key:sum(r['summary'][key] for r in good) for key in ('executor_steps','recovery_actions','recovery_attempts','memory_queries','memory_interventions','model_call_count','elapsed_seconds')}})
    return {'all_included_episodes_present':set(by)==expected,'valid_paired_blocks':valid,
        'excluded_overlap_blocks':sorted(excluded),'excluded_invalid_blocks':[b for b in included if b not in valid],
        'runtime_errors':[{'task':r['task'],'repeat_id':r['repeat_id'],'system':r['system'],'error':r['error']} for r in rows if 'error' in r],
        'systems':systems,'paired_contrasts':comparisons,**({'secondary_contrasts':secondary_results} if secondary else {}),'scope':plan['scope'],
        'interval_note':'Stratified paired bootstrap for the fixed families; small or constant samples can produce degenerate intervals.'}


def events(folder,name):
    path=folder/'runtime'/name
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def audit_and_report(out, *, report_dir=None):
    import numpy as np
    from web_agent.eval.table2.miniwob_study import verify_plan,sha,write_json
    from web_agent.benchmarks.miniwob_support import full_completion
    from web_agent.benchmarks.miniwob_controls import CONTROL_INTERFACE,prompt_controls
    from web_agent.runtime.contracts import PolicyObservation,canonical_sha256
    from web_agent.runtime.named_target_policy import resolve_named_target
    from web_agent.runtime.policy import ActionParseError
    from web_agent.runtime.causal_recovery_planner import NEUTRAL_STEP_REQUEST_INSTRUCTION,NEUTRAL_STEP_REQUEST_VERSION
    from web_agent.runtime.event_log import verify_event_log
    from web_agent.memory.label_backed_store import LabelBackedMemoryStore
    from web_agent.memory.label_experience import LabelExperienceMaterial
    out=Path(out)
    plan=verify_plan(out,audit_revision=report_dir is not None)
    report_dir=Path(report_dir) if report_dir is not None else out
    if report_dir != out: report_dir.mkdir(parents=True,exist_ok=False)
    rows=json.loads((out/'results.json').read_text())
    revision=plan['config'].get('development_revision',{})
    store=LabelBackedMemoryStore(plan['config']['asset_paths']['memory'])
    material=LabelExperienceMaterial(plan['config']['memory_material'],store=store,include_source_context=True)
    checks=[];errors=[];contexts=[];failure_stages=Counter();candidate_counts=Counter()
    totals=Counter();system_diagnostics=defaultdict(Counter)
    def check_episode(row,folder,binding):
        system=row['system'];summary=row['summary']
        for stream in (folder/'runtime').glob('*.jsonl'): verify_event_log(stream)
        assert summary['valid_for_primary'] and not summary['environment_failure'],'invalid episode'
        assert summary['model_call_count']<=102 and summary['executor_steps']<=30 and summary['recovery_attempts']<=4
        outcomes=json.loads((folder/'browser-outcomes.json').read_text())
        assert outcomes==row['environment_outcomes'] and row['success']==full_completion(outcomes)
        actions=events(folder,'actions.jsonl')
        normal=[e['payload'] for e in actions if e['event_type']=='normal_action']
        recovery=[e['payload'] for e in actions if e['event_type']=='recovery_action']
        if plan['config'].get('hybrid_profile'):
            def raw_action(record):
                path=folder/'hybrid-action-records'/f"action-{record['execution']['executor_step']:04d}.json"
                assert sha(path)==record['raw_evidence_sha256'], 'raw action receipt changed'
                value=json.loads(path.read_text())
                assert log_matches(value,{k:v for k,v in record.items() if k!='raw_evidence_sha256'})
                return value
            normal=[raw_action(r) for r in normal]
            recovery=[raw_action(r) for r in recovery]
        submitted=sum(x['execution']['status'] in {'executed','error'} for x in normal+recovery)
        assert submitted==len(outcomes),'browser action/outcome count mismatch'
        for kind,values in [('normal',normal),('recovery',recovery)]:
            count=sum(x['execution']['status']=='executed' for x in values)
            totals['executed_'+kind]+=count;system_diagnostics[system]['executed_'+kind]+=count
        rec_events=events(folder,'recoveries.jsonl')
        plans=[e['payload'] for e in rec_events if e['event_type']=='recovery_plan']
        assessment_count=sum(e['event_type']=='recovery_attempt' and 'predicted_assessment' in e['payload'] for e in rec_events)
        totals['post_recovery_assessments']+=assessment_count
        for item in plans:
            p=item['plan'];assert p['incident_attempt_index']<=2
            if p['resolution_status']=='REJECTED':
                stage=p.get('rejection_stage') or 'strategy_validation';failure_stages[stage]+=1;system_diagnostics[system][stage]+=1
                assert stage!='planner_runtime','unexpected planner/backend error'
        logged_contexts={e['payload']['context_sha256']:e['payload']['context'] for e in rec_events if e['event_type']=='planner_context'}
        action_records={x['action']['action_id']:x for x in normal+recovery}
        requests=[x for x in plans if x['final_decision']['strategy'] in {'REPLAN','ALTERNATIVE_TARGET'}]
        receipts=[json.loads(p.read_text()) for p in sorted((folder/'named-recovery-outputs').glob('output-*.json'))]
        assert len(receipts)==len(requests),'unlogged recovery generation'
        raw_by_attempt={p['plan']['attempt_id']:r for p,r in zip(requests,receipts)}
        for item,receipt in zip(requests,receipts):
            if 'recovery_prompt_path' in plan:
                assert receipt['prompt_sha256']==sha(plan['recovery_prompt_path'])
            assert sha(Path(folder/f"observation-{receipt['observation_id'].rsplit(':',1)[1]}.png"))==receipt['screenshot_sha256']
            assert hashlib.sha256(receipt['input_suffix'].encode()).hexdigest()==receipt['input_suffix_sha256']
            interface_version=plan['config'].get('hybrid_interface_version', 1)
            before,separator,after=receipt['input_suffix'].partition('\nnext_action_request: ')
            context=json.loads(before.removeprefix('\ncausal_recovery_context: '))
            if interface_version == 3:
                from web_agent.runtime.hybrid_interface import semantic_context
                from web_agent.eval.table2.hybrid_reporting import audit_generation_metadata
                assert not separator and '\ncurrent_controls: ' not in receipt['input_suffix']
                assert receipt['hybrid_interface_version'] == 3
                bound_context=receipt['audit_context']['context']
                assert context == semantic_context(bound_context, episode_id=summary['episode_id'],
                    recovery=True, interface_version=3)
                audit_generation_metadata(receipt['generation_metadata'], scripted=plan.get('evidence_type') == 'SCRIPTED_ENGINEERING')
                context=bound_context; request=None
            else:
                assert separator
                request=json.loads(after)
            if interface_version == 2:
                from web_agent.runtime.hybrid_interface import semantic_context, feedback_context
                assert receipt['hybrid_interface_version'] == 2
                bound_context = receipt['audit_context']['context']
                bound_request = receipt['audit_context']['request']
                assert context == semantic_context(bound_context, episode_id=summary['episode_id'], recovery=True)
                assert request == {**bound_request, 'previous_generation_rejection':
                    feedback_context(bound_request['previous_generation_rejection'], episode_id=summary['episode_id'])}
                context, request = bound_context, bound_request
            if request is not None:
                assert request['schema']==NEUTRAL_STEP_REQUEST_VERSION and request['instruction']==NEUTRAL_STEP_REQUEST_INSTRUCTION
                assert request['current_goal']==binding['task_specification']['goal']
            logged=logged_contexts[context['planner_context_sha256']]
            assert log_matches(context['completed_actions'],logged['completed_actions'])
            assert log_matches(context['previous_attempt'],logged['previous_attempt'])
            if request is not None:
                assert request['previous_generation_rejection']==context['previous_attempt']
            if context['previous_attempt'] is not None:
                assert context['previous_attempt']['attempt_id'].startswith(logged['incident_id']+':attempt:')
            raw_context={**logged,'completed_actions':context['completed_actions'],'previous_attempt':context['previous_attempt']}
            assert canonical_sha256(raw_context)==context['planner_context_sha256'],'planner context commitment differs'
            for a in context['completed_actions']:
                recorded=action_records[a['action_id']]
                assert log_matches({k:a[k] for k in ('action_type','bbox','parameters')},
                    {k:recorded['action'][k] for k in ('action_type','bbox','parameters')})
                assert a['execution_status']==recorded['execution']['status'] and a['state_changed']==recorded['execution']['state_changed']
            decision=item['final_decision']
            assert context['selected_recovery_strategy']==decision['strategy'] and context['observed_failure_diagnosis']==decision['diagnosis']
            assert log_matches(context.get('retrieved_training_examples',[]),decision['memory_experiences'])
            obs=json.loads((folder/f"observation-{receipt['observation_id'].rsplit(':',1)[1]}.json").read_text())
            view=PolicyObservation(task_id=binding['task_specification']['task_id'],goal=binding['task_specification']['goal'],observation_id=obs['observation_id'],
                screenshot_sha256=obs['screenshot_sha256'],screenshot_path=obs['screenshot_path'],width=obs['width'],height=obs['height'],
                url=obs['url'],title=obs['title'],current_page_state=obs['page_state'])
            assert context['current_visible_controls']==prompt_controls(view)
            try:
                if interface_version in {2, 3}:
                    from web_agent.runtime.hybrid_interface import parse_action
                    parsed,resolution=parse_action(receipt['raw_response'],view,interface_version=interface_version)
                else:
                    parsed,_=resolve_named_target(receipt['raw_response'],view,allow_json_fence=True,allow_role_target=True)
                assert receipt['status']=='RESOLVED' and parsed==receipt['resolved']
                if interface_version == 3:
                    assert receipt['resolution'] == resolution
            except ActionParseError as exc:
                assert receipt['status']=='REJECTED'
                if interface_version == 3:
                    assert receipt['diagnostic'] == exc.diagnostic
                    assert receipt['diagnostic_code'] == exc.diagnostic_code
                    assert receipt['failure_stage'] == exc.failure_stage
                    assert receipt['proposed_action'] == getattr(exc, 'proposed_action', None)
            if context.get('retrieved_training_examples'):
                totals['memory_generation_exposures']+=1
                system_diagnostics[system]['memory_generation_exposures']+=1
            contexts.append({'task':row['task'],'repeat_id':row['repeat_id'],'system':system,'observation_id':receipt['observation_id'],'status':'PASS'})
        progress=[e['payload'] for e in rec_events if e['event_type']=='observable_step_progress']
        if revision.get('observable_progress_continuation') or revision.get('hybrid_continuation'):
            from web_agent.runtime.observable_progress import observable_step_effect,register_novel_effect
            from web_agent.runtime.contracts import ConcreteAction
            assert len(progress)==assessment_count
            seen=set()
            assessments={e['payload']['attempt']['attempt_id']:e['payload']['predicted_assessment']
                for e in rec_events if e['event_type']=='recovery_attempt' and 'predicted_assessment' in e['payload']}
            terminals={e['payload']['receipt_binding']['action_id']:e['payload']['terminate']
                for e in events(folder,'terminal_signals.jsonl') if e['event_type']=='after_recovery_action'}
            def load_view(oid):
                o=json.loads((folder/f"observation-{oid.rsplit(':',1)[1]}.json").read_text())
                return PolicyObservation(task_id=binding['task_specification']['task_id'],goal=binding['task_specification']['goal'],
                    observation_id=o['observation_id'],screenshot_sha256=o['screenshot_sha256'],screenshot_path=o['screenshot_path'],
                    width=o['width'],height=o['height'],url=o['url'],title=o['title'],current_page_state=o['page_state'])
            for record in progress:
                assert record['learned_assessment_id']==assessments[record['attempt_id']]['assessment_id']
                action=dict(action_records[record['action_id']]['action']);parameters=dict(action['parameters'])
                if action['action_type'] in {'TYPE','SELECT'}:
                    key='text' if action['action_type']=='TYPE' else 'option'
                    if plan['config'].get('hybrid_interface_version', 1) == 3:
                        # The committed raw action receipt above already binds
                        # exact values, including RETRY with no generator call.
                        assert isinstance(parameters[key], str)
                        if record['attempt_id'] in raw_by_attempt:
                            assert parameters[key] == raw_by_attempt[record['attempt_id']]['resolved']['value']
                    else:
                        raw=raw_by_attempt[record['attempt_id']]['resolved']['value']
                        assert log_matches(raw,parameters[key]);parameters[key]=raw
                action['parameters']=parameters
                expected=register_novel_effect(observable_step_effect(ConcreteAction.from_dict(action),
                    load_view(record['pre_observation_id']),load_view(record['post_observation_id'])),seen)
                assert log_matches(expected,{k:record[k] for k in expected})
                assert record['continuation_allowed']==(not revision.get('hybrid_continuation',False) and expected['novel_effect'] and not terminals[record['action_id']])
                totals['observable_effects']+=int(expected['observed_effect'])
                totals['observable_continuations']+=int(record['continuation_allowed'])
                system_diagnostics[system]['observable_continuations']+=int(record['continuation_allowed'])
        else:
            assert not progress
        queries=[e['payload'] for e in events(folder,'memory_queries.jsonl') if e['event_type']=='post_failure_query']
        if system not in {'E3','H3'}: assert not queries and summary['memory_queries']==summary['memory_interventions']==0
        else:
            assert len(queries)==summary['memory_queries']
            assert sum(q['query_result']['changed_recovery_context'] for q in queries)==summary['memory_interventions']
            for q in queries:
                r=q['query_result'];query=q['query']
                hits=store.query(r['normalized_query_embedding'],excluded_task_ids={query['task_id']})
                assert r['candidate_ids']==[h.memory_id for h in hits]
                np.testing.assert_allclose(r['scores'],[h.similarity for h in hits],atol=1e-6,rtol=0)
                expected=[material.for_hit(h).to_dict() for h in hits if h.admitted]
                if revision.get('selective_memory_context'):
                    from web_agent.memory.context_applicability import context_exclusion
                    transition=dict(q['embedding_request']['post_action_input'])
                    post=dict(transition['post_observation'])
                    raw_state=load_view(post['observation_id']).current_page_state
                    assert log_matches(raw_state,post['current_page_state'])
                    post['current_page_state']=raw_state
                    transition['post_observation']=post
                    excluded={e['memory_id']:reason for e in expected
                              if (reason:=context_exclusion(e,transition))}
                    assert r['exclusion_reasons']==excluded
                    expected=[e for e in expected if e['memory_id'] not in excluded]
                    assert r['admitted_candidate_id']==(expected[0]['memory_id'] if expected else None)
                    assert r['admitted']==bool(expected)
                    if not expected: assert q['final_decision']==q['shadow_decision']
                    totals['memory_context_exclusions']+=len(excluded)
                    totals['memory_abstentions']+=int(not expected)
                assert q['final_decision']['memory_experiences']==expected
                assert r['context_candidate_ids']==[e['memory_id'] for e in expected]
                assert r['memory_material_sha256']==material.sha256 and r['store_manifest_sha256']==store.manifest_sha256
                for key in ('strategy','diagnosis','planned_action','trigger_sources','incident_id'):
                    assert q['final_decision'][key]==q['shadow_decision'][key]
                assert not r['changed_strategy'] and not r['changed_target_or_parameters']
                er=q['embedding_request'];transition=er['post_action_input']
                assert er['failed_action_id']==query['failed_action_id']==transition['executed_action']['action_id']
                assert er['post_failure_observation_id']==query['post_failure_observation_id']==transition['post_observation']['observation_id']
                assert er['post_action_input_sha256']==query['post_action_input_sha256'] and er['checkpoint_sha256']==store.checkpoint_sha256
                candidate_counts.update(r['context_candidate_ids']);totals['memory_queries']+=1
        if plan['config'].get('hybrid_profile'):
            from web_agent.eval.table2.hybrid_reporting import audit_hybrid_episode
            first_advice=audit_hybrid_episode(plan, row, folder, binding)
            if plan['config'].get('hybrid_interface_version') == 3:
                from web_agent.runtime.observable_progress import observable_step_effect,register_novel_effect
                from web_agent.runtime.contracts import ConcreteAction
                diagnostic=Counter()
                proposals=[json.loads(p.read_text()) for p in sorted((folder/'hybrid-normal-outputs').glob('proposal-*.json'))]
                for proposal in proposals:
                    if 'raw_response' not in proposal: continue
                    known=proposal.get('resolved_proposal') or proposal.get('proposed_action')
                    diagnostic['normal_schema_valid_proposals' if known else 'normal_malformed_proposals']+=1
                    diagnostic['normal_target_resolution_rejections']+=int('proposed_action' in proposal)
                    if known: diagnostic['normal_model_chosen_'+known['action_type']]+=1
                    metadata=proposal['generation_metadata']
                    diagnostic['normal_generation_stop_'+metadata.get('stop_reason','unavailable')]+=1
                repairs=[e['payload'] for e in actions if e['event_type']=='hybrid_proposal_repair']
                diagnostic['normal_format_repair_successes']+=sum(e['outcome']=='corrected' for e in repairs)
                diagnostic['normal_proposal_repair_unresolved']+=sum(e['outcome']=='unresolved' for e in repairs)
                pre_by_oid={p['observation_id']:PolicyObservation.from_dict(p['observation']) for p in proposals}
                post_by_action={e['payload']['prior_action_id']:e['payload']['observation_id']
                    for e in events(folder,'environment_events.jsonl') if e['event_type']=='post_action_observation'}
                seen=set(); prepared=False
                for record in normal:
                    if record['execution']['status'] != 'executed': continue
                    diagnostic['normal_executed_actions_after_preparation_effect']+=int(prepared)
                    action=ConcreteAction.from_dict(record['action'])
                    post_id=post_by_action[action.action_id]
                    raw_post=json.loads((folder/f"observation-{post_id.rsplit(':',1)[1]}.json").read_text())
                    post=PolicyObservation(task_id=binding['task_specification']['task_id'],goal=binding['task_specification']['goal'],
                        observation_id=raw_post['observation_id'],screenshot_sha256=raw_post['screenshot_sha256'],
                        screenshot_path=raw_post['screenshot_path'],width=raw_post['width'],height=raw_post['height'],
                        url=raw_post['url'],title=raw_post['title'],current_page_state=raw_post['page_state'])
                    effect=register_novel_effect(observable_step_effect(action,
                        pre_by_oid[record['decision']['observation_id']],post),seen)
                    diagnostic['normal_observed_preparation_effects']+=int(effect['observed_effect'])
                    diagnostic['normal_novel_preparation_effects']+=int(effect['novel_effect'])
                    prepared=prepared or effect['novel_effect']
                system_diagnostics[system].update(diagnostic); totals.update(diagnostic)
            return first_advice
        return normal[0]['decision'] if system!='E0' else None
    for block in plan['blocks']:
        task=block['task'];repeat=block['repeat_id'];folder=out/task/f'repeat-{repeat}'
        block_rows=[r for r in rows if r['task']==task and r['repeat_id']==repeat]
        try:
            overlap=json.loads((folder/'overlap-audit.json').read_text())
            if len(block_rows)==1 and block_rows[0].get('status')=='EXCLUDED_TRAIN_OVERLAP':
                assert overlap['matching_source_tasks'];continue
            assert not overlap['matching_source_tasks']
            assert len(block_rows)==4 and {r['system'] for r in block_rows}==set(plan['systems'])
            binding=json.loads((folder/'task-binding.json').read_text())
            assert all(binding[k]==block[k] for k in block)
            screens=[];initial=[]
            for row in block_rows:
                assert 'error' not in row,row.get('error')
                sub=folder/row['system'];screens.append(sha(sub/'observation-1.png'))
                d=check_episode(row,sub,binding)
                if d is not None: initial.append({k:d[k] for k in ('action_type','action_probabilities','bbox','confidence_before','parameter_hints')})
            assert len(set(screens))==1,'paired resets differ'
            assert len(initial)==3 and initial[0]==initial[1]==initial[2],'initial trained predictions differ'
            if plan['config'].get('hybrid_interface_version', 1) in {2, 3}:
                first = [json.loads((folder/s/'hybrid-normal-outputs/proposal-0001.json').read_text())
                         for s in ('H1', 'H2', 'H3')]
                assert all(r['input_suffix']==first[0]['input_suffix'] and r['prompt']==first[0]['prompt']
                           for r in first), 'initial pre-memory generation inputs differ'
            checks.append({'task':task,'repeat_id':repeat,'status':'PASS'})
        except Exception as exc:
            errors.append({'task':task,'repeat_id':repeat,'error':str(exc),'exception':type(exc).__name__,'traceback':traceback.format_exc()})
    analysis=analyze(plan,rows)
    for system in analysis['systems']:
        diagnostic=system_diagnostics[system['system']]
        system.update(executed_normal=diagnostic['executed_normal'],executed_recovery=diagnostic['executed_recovery'],
            memory_generation_exposures=diagnostic['memory_generation_exposures'])
        if plan['config'].get('hybrid_interface_version') == 3:
            for key in ('normal_schema_valid_proposals','normal_malformed_proposals',
                        'normal_target_resolution_rejections','normal_format_repair_successes',
                        'normal_proposal_repair_unresolved','normal_observed_preparation_effects',
                        'normal_novel_preparation_effects','normal_executed_actions_after_preparation_effect'):
                system[key]=diagnostic[key]
    if not analysis['all_included_episodes_present']: errors.append({'error':'missing or unexpected episodes'})
    audit={'status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,'totals':dict(totals),
        'live_path_verified':plan.get('evidence_type')!='SCRIPTED_ENGINEERING' and not errors and totals['executed_recovery']>0 and totals['post_recovery_assessments']>0 and totals['memory_generation_exposures']>0,
        'failure_stages':dict(failure_stages),'system_diagnostics':{s:dict(v) for s,v in system_diagnostics.items()},
        'memory_candidate_frequency':dict(candidate_counts),'source_and_assets_verified':True,
        'context_audit_count':len(contexts),'claim':'Exposure and execution integrity are distinct from improvement.'}
    if report_dir != out:
        audit['audit_revision']={'reason':'Audit-only replay of preserved producing sources and raw evidence; no episode reruns.',
            'producing_plan_sha256':sha(out/'plan.json'),
            'auditor_sources':{str(p):sha(p) for p in Path(__file__).resolve().parents[2].rglob('*.py')}}
    write_json(report_dir/'audit.json',audit);write_json(report_dir/'context-audit.json',{'status':audit['status'],'checks':contexts})
    write_json(report_dir/'analysis.json',analysis)
    with (report_dir/'table2.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(analysis['systems'][0]));writer.writeheader();writer.writerows(analysis['systems'])
    lines=[f"# MiniWoB {plan['phase']} — {audit['status']}", '',plan['scope'],'',
        '| System | Completed / eligible | Executed normal | Executed recovery | Memory queries | Context exposure |',
        '|---|---:|---:|---:|---:|---:|']
    for s in analysis['systems']:
        d=system_diagnostics[s['system']]
        lines.append(f"| {s['system']} | {s['completed']}/{s['episodes']} | {d['executed_normal']} | {d['executed_recovery']} | {s['memory_queries']} | {d['memory_generation_exposures']} |")
    lines+=['','Paired contrasts (right minus left):','']
    for c in analysis['paired_contrasts']:
        lines.append(f"- {c['contrast']}: difference {c['completion_difference']}; 95% interval {c['ci95']}; improved/worsened {c['improved']}/{c['worsened']}; exact p={c['exact_two_sided_p']}, Holm p={c['holm_p']}. {c['conclusion']}.")
    for c in analysis.get('secondary_contrasts',[]):
        lines.append(f"- Secondary {c['contrast']}: difference {c['completion_difference']}; 95% interval {c['ci95']}; exact p={c['exact_two_sided_p']} (outside primary Holm family).")
    lines+=['',f"Failure stages: {dict(failure_stages)}.",
        'Memory material is training-label-backed; missing corrective values/reflections remain null.',
        'These six evaluation families were previously observed. One checkpoint does not estimate training-seed uncertainty.',
        ('Engineering/context audit PASS does not establish a completion benefit. H0–H3 share the frozen generator; H1–H3 add trained advice.' if plan['config'].get('hybrid_profile') else 'Engineering/context audit PASS does not establish a completion benefit. E0 and E1 use different action-generation interfaces.'),
        'P2/P3 remain shared capabilities; the paired contrasts do not independently estimate all four pillar effects.',
        analysis['interval_note'],'']
    (report_dir/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'audit':audit['status'],'live_path_verified':audit['live_path_verified'],'systems':analysis['systems'],'errors':errors}),flush=True)
    return audit
