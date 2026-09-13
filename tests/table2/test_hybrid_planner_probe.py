"""Verify the diagnostic changes only its declared input factor."""
import json
import pytest
from scripts.probe_hybrid_planner_v1 import suffix_for


def test_factor_isolation_and_exact_history_values():
    ctx={'current_controls':[{'control_id':'o3:c0'}], 'trained_advice':{'argmax':'NAVIGATE'},
         'completed_actions':[{'action_type':'TYPE','bbox':[.1,.2,.3,.1],
            'parameters':{'text':' Exact "Case" \n','target_control_id':'o1:c0'},
            'execution_status':'executed','state_changed':True}],
         'last_rejection':{'stage':'target_resolution','detail':'old target o1:c0'}}
    suffix='\nhybrid_action_context: '+json.dumps(ctx,sort_keys=True)
    saved={'input_suffix':suffix}
    assert suffix_for(saved,'baseline','notice')==suffix
    assert suffix_for(saved,'contract_last','notice')==suffix+'\nnext_response_contract: notice'
    changed=json.loads(suffix_for(saved,'history_text','notice').split(': ',1)[1])
    assert {k:v for k,v in changed.items() if k!='completed_actions'}=={k:v for k,v in ctx.items() if k!='completed_actions'}
    history=changed['completed_actions'][0]
    assert 'issued text='+json.dumps(ctx['completed_actions'][0]['parameters']['text']) in history
    assert 'issued target_control_id="o1:c0"' in history
    assert 'observed execution_status="executed"' in history
    assert json.loads(saved['input_suffix'].split(': ',1)[1])==ctx


def test_no_silent_unknown_history_drop_or_empty_intervention():
    for history in ([],[{'action_type':'CLICK','unexpected':'must preserve'}]):
        with pytest.raises(ValueError):suffix_for({'input_suffix':'\nhybrid_action_context: '+json.dumps({'completed_actions':history})},'history_text','')
    with pytest.raises(ValueError):suffix_for({'input_suffix':''},'unplanned_combination','')
