import json
import pytest
from scripts.probe_atomic_action_contract import translate


@pytest.mark.parametrize('kind,value',[('CLICK',None),('TYPE',' Exact  Case! '),('SELECT','one'),
                                    ('SCROLL','down'),('NAVIGATE','http://fixture.local/next'),('PRESS_KEY','TAB')])
def test_translator_preserves_all_six_model_operations_and_arguments(kind,value):
    target='o4:c0' if kind in {'CLICK','TYPE','SELECT'} else None
    result=translate(json.dumps({'operation':kind,'control':target,'argument':value}))
    assert result=={'action_type':kind,'target':target,'value':value,'bbox':None}


@pytest.mark.parametrize('raw',[
    '[{"operation":"TYPE","control":"o1:c0","argument":"a"}]',
    '{"operation":"TYPE","operation":"CLICK","control":null,"argument":null}',
    '{"operation":"TYPE","control":"o1:c0"}',
    '{"operation":"UNKNOWN","control":null,"argument":null}',
])
def test_translator_rejects_ambiguous_or_incomplete_proposals(raw):
    with pytest.raises(ValueError):translate(raw)


def test_one_fence_is_an_envelope_not_an_action_change():
    raw='{"operation":"TYPE","control":"o1:c0","argument":" Exact! "}'
    assert translate('```json\n'+raw+'\n```')==translate(raw)
    with pytest.raises(ValueError):translate('Explanation\n```json\n'+raw+'\n```')
