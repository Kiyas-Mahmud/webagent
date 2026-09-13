"""The diagnostic only appends unexecuted model advice; no executor exists."""
import json
from scripts.probe_hybrid_planning_v2 import action_suffix, CASES


def test_plan_preserves_observation_and_exact_model_text():
    context={'schema':'hybrid-semantic-action-v3','current_controls':[{'value':' Exact  Case! '}],
             'causal_history':[],'trained_advice':{'argmax':'NAVIGATE'}}
    original='\nhybrid_action_context: '+json.dumps(context)
    text='TYPE " Exact  Case! "\nThen inspect the page.'
    result=json.loads(action_suffix(original,text).split(': ',1)[1])
    advice=result.pop('tentative_model_plan')
    assert result==context and advice['text']==text and advice['already_executed'] is False
    assert CASES==('enter-text','enter-text-2','click-button-sequence')
