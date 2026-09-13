from scripts.probe_six_action_context import variant_context


def test_ablation_does_not_rewrite_current_state_or_invent_empty_history():
    full={'current_controls':[{'value':' Exact  Case! '}],
          'causal_history':[{'issued_value':'previous'}],
          'trained_advice':{'argmax':'SCROLL'},'proposal_feedback':None}
    minimal=variant_context(full,'controls')
    assert 'causal_history' not in minimal and 'trained_advice' not in minimal
    assert minimal['current_controls']==full['current_controls']
    advised=variant_context(full,'controls_advice')
    assert 'causal_history' not in advised and advised['trained_advice']==full['trained_advice']
    complete=variant_context(full,'controls_advice_history')
    assert complete==full
    complete['current_controls'][0]['value']='changed'
    assert full['current_controls'][0]['value']==' Exact  Case! '
