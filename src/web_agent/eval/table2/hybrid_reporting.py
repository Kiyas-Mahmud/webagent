"""Independent replay of hybrid normal-action, isolation and continuation receipts."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import random
from types import SimpleNamespace

from web_agent.runtime.contracts import PolicyObservation, PreActionDecision, ConcreteAction, RuntimeTaskView, canonical_sha256
from web_agent.runtime.hybrid_action_policy import trained_advice, PROMPT_PATH, _HybridGenerator, HybridActionContext
from web_agent.benchmarks.miniwob_controls import prompt_controls
from web_agent.runtime.policy import ActionParseError


def audit_generation_metadata(metadata, *, scripted=False):
    """Check stop evidence independently; decoded text cannot establish a cap."""
    assert isinstance(metadata, dict) and type(metadata.get('available')) is bool
    if not metadata['available']:
        assert scripted and isinstance(metadata.get('reason'), str)
        return
    assert set(metadata) == {'available', 'generated_token_count', 'eos_token_ids',
        'eos_observed', 'token_cap_reached', 'max_new_tokens', 'stop_reason'}
    count = metadata['generated_token_count']
    assert type(count) is int and 0 <= count <= 128 and metadata['max_new_tokens'] == 128
    assert isinstance(metadata['eos_token_ids'], list) and all(type(i) is int for i in metadata['eos_token_ids'])
    assert type(metadata['eos_observed']) is bool and type(metadata['token_cap_reached']) is bool
    assert not metadata['eos_observed'] or (metadata['eos_token_ids'] and count > 0)
    assert metadata['token_cap_reached'] == (count == 128)
    assert metadata['stop_reason'] == ('eos' if metadata['eos_observed'] else
        'max_new_tokens' if count == 128 else 'other')


def audit_hybrid_episode(plan, row, folder, binding):
    from web_agent.eval.table2.miniwob_reporting import events, log_matches
    from web_agent.eval.table2.miniwob_study import sha
    system = row['system']; summary = row['summary']
    version = plan['config'].get('hybrid_interface_version', 1)
    prompt_path = PROMPT_PATH if version == 1 else PROMPT_PATH.with_name(f'miniwob_hybrid_action_prompt_v{version}.txt')
    assert summary['system_id'] == system and summary['protocol_id'] == 'table2-miniwob-hybrid-v1'
    action_events = events(folder, 'actions.jsonl')
    normal = [e['payload'] for e in action_events if e['event_type'] in {'normal_action', 'pre_action_parse_rejection'}]
    records = [json.loads(p.read_text()) for p in sorted((folder/'hybrid-normal-outputs').glob('proposal-*.json'))]
    completed = [r for r in records if r['status'] in {'RESOLVED','REJECTED'}]
    assert len(normal) == len(completed) == summary['normal_actions']
    dispatch = [e['payload'] for e in events(folder,'environment_events.jsonl') if e['event_type']=='model_call_dispatch']
    assert len(dispatch) == summary['model_call_count']
    assert [d['model_call_index'] for d in dispatch] == list(range(1,len(dispatch)+1))
    assert all(d['episode_id']==summary['episode_id'] for d in dispatch)
    if system in {'H0','H1'}:
        assert summary['recovery_actions']==summary['recovery_attempts']==0
        assert not any(d['stage'] in {'post_action_assessment','recovery_assessment','recovery_planner','memory_embedding'} for d in dispatch)
    if system!='H3':
        assert not list((folder/'hybrid-memory-outputs').glob('*'))
        assert not any(d['stage']=='memory_embedding' for d in dispatch)
    # Unfinished inference receipts are allowed only at the bound budget/time stop.
    for r in records:
        if r['status']=='ERROR':
            assert r is records[-1] and summary['terminal_reason'] in {'policy_error','timeout'}
            assert r['error_type'] in {'ModelCallBudgetExceeded','EpisodeTimeout'}
    actual_pre = [d['component_id'] for d in dispatch if d['stage']=='pre_action_policy']
    expected_pre=[]
    for r in records:
        if 'trained_decision' in r: expected_pre.append('validation-selected-web-agent@v1')
        if 'raw_response' in r: expected_pre.append('miniwob-'+system.lower()+f'-normal@hybrid-normal-action-v{version}')
    # A dispatched callback can time out before returning its raw response.
    assert actual_pre[:len(expected_pre)]==expected_pre
    assert len(actual_pre)==len(expected_pre) or (records[-1]['status']=='ERROR' and len(actual_pre)==len(expected_pre)+1)
    first_advice=None
    for receipt, logged in zip(completed,normal):
        assert receipt['episode_id']==summary['episode_id'] and receipt['system']==system
        assert receipt['schema']==f'hybrid-normal-action-v{version}'
        assert receipt['prompt']==prompt_path.read_text() and receipt['prompt_sha256']==sha(prompt_path)
        view=PolicyObservation.from_dict(receipt['observation'])
        assert view.record_sha256==receipt['observation_sha256']
        assert view.task_id==binding['task_specification']['task_id'] and view.goal==binding['task_specification']['goal']
        oid=view.observation_id
        raw=json.loads((folder/f"observation-{oid.rsplit(':',1)[1]}.json").read_text())
        assert view.current_page_state==raw['page_state'] and view.screenshot_sha256==raw['screenshot_sha256']
        assert sha(view.screenshot_path)==view.screenshot_sha256
        context=json.loads(receipt['input_suffix'].removeprefix('\nhybrid_action_context: '))
        assert hashlib.sha256(receipt['input_suffix'].encode()).hexdigest()==receipt['input_suffix_sha256']
        model_context = context
        if version in {2, 3}:
            from web_agent.runtime.hybrid_interface import semantic_context
            context = receipt['audit_context']
            assert model_context == semantic_context(context, episode_id=summary['episode_id'], interface_version=version)
        if version == 3:
            audit_generation_metadata(receipt['generation_metadata'], scripted=plan.get('evidence_type') == 'SCRIPTED_ENGINEERING')
        assert context['current_controls']==prompt_controls(view)
        assert context['observation_id']==oid and context['episode_id']==summary['episode_id']
        # Only actions completed before this observation can occur in history.
        prior=[e['payload'] for e in action_events if e['event_type'] in {'normal_action','recovery_action'}
               and e['payload']['execution']['executor_step'] < logged['execution']['executor_step']]
        assert [a['action_id'] for a in context['completed_actions']]==[p['action']['action_id'] for p in prior]
        assert [h.action_id for h in view.causal_history]==[p['action']['action_id'] for p in prior]
        for a,p,h in zip(context['completed_actions'],prior,view.causal_history):
            assert log_matches({k:a[k] for k in ('action_type','parameters','bbox')},{k:p['action'][k] for k in ('action_type','parameters','bbox')})
            assert a['execution_status']==p['execution']['status']==h.execution_status.value
            assert a['state_changed']==p['execution']['state_changed']==h.state_changed
            assert a['post_observation_id']==h.post_observation_id
        history_actions=tuple(ConcreteAction.from_dict({**p['action'], 'parameters':a['parameters']})
            for p,a in zip(prior,context['completed_actions']))
        projected=HybridActionContext(summary['episode_id'],oid,history_actions,context['last_rejection']).project(view)
        assert projected=={k:v for k,v in context.items() if k!='trained_advice'}
        if system=='H0':
            assert 'trained_advice' not in context and 'trained_decision' not in receipt
        else:
            decision=PreActionDecision.from_dict(receipt['trained_decision'])
            assert decision.observation_id==oid and decision.input_observation_ids==(oid,)
            assert receipt['trained_advice']==context['trained_advice']==trained_advice(decision,
                checkpoint_sha256=plan['config']['checkpoint_and_base_bindings']['checkpoint_sha256'])
            if first_advice is None: first_advice=decision.to_dict()
        replay={}
        replay_base=SimpleNamespace(_generate=lambda **kw:receipt['raw_response'])
        if version == 3:
            replay_base._generate_with_metadata=lambda **kw:(receipt['raw_response'],receipt['generation_metadata'])
        backend=_HybridGenerator(base=replay_base,
            prompt=receipt['prompt'],context=model_context,receipt=replay,policy_id='miniwob-'+system.lower()+'-normal',
            interface_version=version)
        try:
            generated=backend.predict_action(RuntimeTaskView.from_dict(receipt['task']),view,random.Random(42)).to_dict()
        except ActionParseError as exc:
            assert receipt['status']=='REJECTED'
            if version == 3:
                from web_agent.runtime.hybrid_interface import TargetProposalResolutionError
                assert receipt['diagnostic'] == exc.diagnostic
                assert receipt['diagnostic_code'] == exc.diagnostic_code
                assert receipt['failure_stage'] == exc.failure_stage
                if isinstance(exc, TargetProposalResolutionError):
                    assert receipt['proposed_action'] == exc.proposed_action
                    raw_path=folder/'hybrid-action-records'/f"action-{logged['execution']['executor_step']:04d}.json"
                    assert sha(raw_path) == logged['raw_evidence_sha256']
                    action_record=json.loads(raw_path.read_text())
                    assert log_matches(action_record, {k:v for k,v in logged.items() if k != 'raw_evidence_sha256'})
                    rejected=action_record['proposal_rejection']; issued=exc.proposed_action
                    assert rejected['schema'] == 'table2.known-action-proposal-rejection.v3'
                    assert rejected['proposed_action'] == issued and rejected['diagnostic'] == exc.diagnostic
                    assert rejected['failure_stage'] == exc.failure_stage and rejected['diagnostic_code'] == exc.diagnostic_code
                    assert rejected['error_sha256'] == exc.error_sha256
                    decision=action_record['decision']; action=action_record['action']; execution=action_record['execution']
                    assert decision['action_type'] == action['action_type'] == issued['action_type']
                    assert decision['bbox'] == action['bbox'] == issued['bbox']
                    hints={'target':issued['target'], 'value':issued['value']}
                    value_key={'TYPE':'text','SELECT':'option','SCROLL':'direction','NAVIGATE':'url','PRESS_KEY':'key'}.get(issued['action_type'])
                    if value_key and issued['value'] is not None: hints[value_key]=issued['value']
                    assert decision['parameter_hints'] == action['parameters'] == hints
                    assert set(decision['action_probabilities']) == {'CLICK','TYPE','SELECT','SCROLL','NAVIGATE','PRESS_KEY'}
                    assert decision['action_probabilities'] == {k:float(k == issued['action_type']) for k in decision['action_probabilities']}
                    assert decision['grounding_confidence'] == decision['confidence_before'] == 0.
                    assert decision['observation_id'] == view.observation_id
                    assert action['source_decision_id'] == decision['decision_id']
                    assert execution['status'] == 'rejected' and not execution['state_changed'] and not execution['environment_error']
                    assert execution['error_kind'] == 'parameter_resolution_rejected'
                    assert execution['evidence']['controller_command'] is None
                    assert action_record['parameters'] is None and action_record['resolution_attempts'] == []
                else:
                    assert logged['decision'] is None and logged['action'] is None and 'proposed_action' not in receipt
            else:
                assert logged['decision'] is None
        else:
            assert receipt['status']=='RESOLVED'
            assert {k:v for k,v in generated.items() if k not in {'decision_id','latency_ms'}}=={k:v for k,v in receipt['decision'].items() if k not in {'decision_id','latency_ms'}}
            assert log_matches(receipt['decision'],logged['decision'])
    if version == 3:
        repair_events=[e['payload'] for e in action_events if e['event_type']=='hybrid_proposal_repair']
        by_state=Counter()
        for event in repair_events:
            assert event['schema'] == 'table2.hybrid-proposal-repair.v3'
            if event['outcome'] == 'corrected':
                assert not event['correction_request_allowed']
                continue
            by_state[event['proposal_state_sha256']] += 1
            assert event['rejection_index'] == by_state[event['proposal_state_sha256']] <= 2
            assert event['correction_request_allowed'] == (event['rejection_index'] == 1)
            assert event['outcome'] == ('correction_pending' if event['rejection_index'] == 1 else 'unresolved')
    rec_events=events(folder,'recoveries.jsonl')
    attempts=Counter();total=0;assessments={}
    for event in rec_events:
        c=event['payload']
        if event['event_type']=='recovery_plan':
            p=c['plan'];attempts[p['incident_id']]+=1;total+=1
            assert p['incident_attempt_index']==attempts[p['incident_id']]
        if event['event_type']=='recovery_attempt' and 'predicted_assessment' in c:
            assessments[c['attempt']['attempt_id']]=c['predicted_assessment']
        if event['event_type']!='hybrid_continuation':continue
        assert c['incident_attempts']==attempts[c['incident_id']]<=2
        assert c['episode_attempts']==total<=4
        if c['incident_status']=='exhausted_unresolved':assert c['incident_attempts']==2
        if c['incident_status']=='unresolved_episode_allowance_exhausted':assert total==4
        if 'assessment_id' in c:
            assessment=assessments[c['attempt_id']]
            assert c['assessment_id']==assessment['assessment_id']
            resolved=assessment['predicted_failure_resolved'];progress=assessment['predicted_progress']
            assert c['incident_status']==('predicted_resolved' if resolved else 'unresolved')
            assert c['next']==('normal' if resolved or progress else 'check_retry_budget')
    assert total==summary['recovery_attempts']
    if system=='H3':
        queries=[e['payload'] for e in events(folder,'memory_queries.jsonl') if e['event_type']=='post_failure_query']
        results=sorted((folder/'hybrid-memory-outputs').glob('*-result.json'))
        assert len(results)==len(queries)
        candidates=Counter()
        for result_path,q in zip(results,queries):
            result=json.loads(result_path.read_text())
            shadow=json.loads(result_path.with_name(result_path.name.replace('-result','-shadow')).read_text())
            assert log_matches(result['decision'],{k:q[k] for k in result['decision']})
            assert log_matches(shadow['query'],q['query']) and log_matches(shadow['shadow_decision'],q['shadow_decision'])
            candidates.update(result['decision']['query_result']['candidate_ids'])
            assert result['candidate_occurrences']==dict(candidates)
            assert result['total_candidate_occurrences']==sum(candidates.values())
            assert not result['generation_exposure_measured'] and not result['completion_benefit_measured']
    return first_advice
