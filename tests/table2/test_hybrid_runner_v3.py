"""Full version-3 orchestration/audit checks; scripted, zero model inference."""
import json

import pytest

from tests.table2.test_hybrid_runner import engineering_run


@pytest.mark.parametrize('invalid_first,invalid_recovery',[(False,False),(True,False),(False,True)])
def test_v3_production_runner_and_independent_audit(tmp_path,monkeypatch,invalid_first,invalid_recovery):
    root=tmp_path/'run'
    _,rows,audit=engineering_run(root,monkeypatch,interface_version=3,
        invalid_first=invalid_first,invalid_recovery=invalid_recovery)
    assert audit['status']=='PASS',audit['errors']
    assert audit['live_path_verified'] is False
    assert len(rows)==8 and all('error' not in r for r in rows)
    for row in rows:
        assert row['summary']['executor_steps']<=30
        assert row['summary']['recovery_attempts']<=4
        assert row['summary']['model_call_count']<=102
    for path in root.rglob('proposal-*.json'):
        receipt=json.loads(path.read_text())
        context=json.loads(receipt['input_suffix'].removeprefix('\nhybrid_action_context: '))
        assert context['schema']=='hybrid-semantic-action-v3'
        assert 'completed_actions' not in context
        if 'trained_advice' in context:
            assert 'action_class_confidence' in context['trained_advice']
            assert 'grounding_confidence' not in context['trained_advice']
