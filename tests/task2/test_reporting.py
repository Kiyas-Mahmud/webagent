import json
import pytest
from web_agent.eval.task2.reporting import paired_test, summarize_pairs, audit


def test_exact_paired_tests_and_holm():
    assert paired_test(0,0)==1
    assert paired_test(6,0)==.03125
    pairs=[{'task':'one','results':{'A':{'completion':False},'B':{'completion':True},'C':{'completion':True}}} for _ in range(6)]
    a,b=summarize_pairs(pairs)
    assert a['difference']==1 and a['pairs']==6 and a['holm_p']==.0625
    assert a['ci95']==[1.,1.] and b['difference']==0 and b['holm_p']==1


def test_pair_denominator_requires_both_systems():
    a,b=summarize_pairs([{'task':'one','results':{'A':{'completion':True}}}])
    assert a['pairs']==b['pairs']==0 and a['difference'] is None


def test_audit_rejects_missing_episodes(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps({'sources':{},'bindings':{},'systems':['A','B','C'],'blocks':[{'eligible':True,'task':'one','repeat_id':0}]}))
    with pytest.raises(ValueError,match='Missing/invalid episodes'):audit(tmp_path)


def test_native_selection_accepts_first_action_and_only_the_upstream_noop():
    from web_agent.eval.task2.reporting import check_native_selection
    first={'click':{'index':5}}; second={'input':{'index':6,'text':'x'}}
    check_native_selection({'native_selection':{'selected_index':0,'proposed_actions':2,'deferred_actions':1},
                            'native_output':{'action':[first]}},[first,second])
    noop={'done':{'success':False,'text':'No next action returned by LLM!','files_to_display':[]}}
    check_native_selection({'native_selection':{'selected_index':None,'proposed_actions':0,'deferred_actions':0,'upstream_noop':True},
                            'native_output':{'action':[noop]}},[])
    with pytest.raises(AssertionError):
        check_native_selection({'native_selection':{'selected_index':None,'proposed_actions':0,'deferred_actions':0,'upstream_noop':True},
                                'native_output':{'action':[first]}},[])
    with pytest.raises(AssertionError):
        check_native_selection({'native_selection':{'selected_index':0,'proposed_actions':2,'deferred_actions':1},
                                'native_output':{'action':[second]}},[first,second])
