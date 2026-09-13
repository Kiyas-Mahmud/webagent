"""Final-entrypoint gates and orchestration; all actors are scripted fixtures."""
from copy import deepcopy
import json

import pytest

from scripts import run_table2_hybrid_final as final


def evidence():
    audit = {'status': 'PASS', 'errors': [], 'source_and_assets_verified': True,
             'totals': {'executed_recovery': 2, 'post_recovery_assessments': 2,
                        'memory_queries': 6, 'memory_abstentions': 6},
             'live_path_verified': False}
    analysis = {'all_included_episodes_present': True, 'excluded_invalid_blocks': [],
                'runtime_errors': [], 'completion_difference': 0}
    readiness = {'pre_memory_generation_context_isolation': 'PASS',
                 'bounded_click_continuation_exercised': True, 'memory_writes': 0,
                 'stored_embeddings_regenerated': 0, 'memory_queries': 6,
                 'memory_abstentions': 6, 'memory_generation_exposures': 0}
    return audit, analysis, readiness


def test_verified_abstention_and_null_performance_are_accepted():
    audit, analysis, readiness = evidence()
    result = final.accept_development(audit, analysis, readiness)
    assert result['status'] == 'PASS' and result['memory_generation_exposures'] == 0
    assert result['full_memory_live_path_verified'] is False
    assert audit['live_path_verified'] is False  # never rewrite historical coverage


@pytest.mark.parametrize('part,key,value', [
    (0, 'status', 'FAIL'), (0, 'errors', ['lost admitted memory']),
    (0, 'source_and_assets_verified', False),
    (1, 'all_included_episodes_present', False), (1, 'excluded_invalid_blocks', ['missing']),
    (1, 'runtime_errors', ['retrieval failure']),
    (2, 'pre_memory_generation_context_isolation', 'FAIL'),
    (2, 'bounded_click_continuation_exercised', False), (2, 'memory_writes', 1),
    (2, 'stored_embeddings_regenerated', 1), (2, 'memory_queries', 5),
    (2, 'memory_generation_exposures', 1),
])
def test_broken_evidence_is_not_abstention(part, key, value):
    parts = list(evidence()); parts[part][key] = value
    with pytest.raises(ValueError):
        final.accept_development(*parts)


@pytest.mark.parametrize('key,value', [('memory_queries', 0), ('executed_recovery', 0),
                                      ('post_recovery_assessments', 1)])
def test_missing_live_prerequisites_rejected(key, value):
    parts = evidence(); parts[0]['totals'][key] = value
    with pytest.raises(ValueError):
        final.accept_development(*parts)


def example_plan():
    p = final.protocol()
    old = final.read(p['candidate_development_plan'])
    plan = {'profile': final.PROFILE, 'phase': 'evaluation', 'config': final.final_config(p, old),
            'systems': list(p['systems']), 'runtime_protocol': p['runtime_protocol'],
            'analysis': p['analysis'], 'scoring': p['scoring'], 'episodes_planned': 120,
            'asset_sha256': p['candidate_asset_sha256'],
            'recovery_prompt_path': old['recovery_prompt_path'],
            'protocol_path': str(final.PROTOCOL), 'protocol_sha256': final.PROTOCOL_SHA,
            'blocks': [{**b, 'task_specification': {'task_id': 'miniwob.'+b['task'],
                       'development_partition': False}} for b in p['blocks']]}
    return deepcopy(plan), p, old


def test_config_derivation_retains_agent_and_old_development_guard():
    plan, p, old = example_plan()
    final.verify_structure(plan, p, old)
    assert old['config']['development_only'] is True
    assert plan['config']['hybrid_interface_version'] == 2
    for key in ('development_revision', 'asset_paths', 'model_seed', 'store_manifest_sha256'):
        assert plan['config'][key] == old['config'][key]


@pytest.mark.parametrize('change', ['seed', 'system', 'phase', 'model', 'prompt', 'partition', 'count'])
def test_final_binding_rejects_changes(change):
    plan, p, old = example_plan()
    if change == 'seed': plan['blocks'][0]['stage_reset_seed'] += 1
    elif change == 'system': plan['systems'] = ['H0']
    elif change == 'phase': plan['phase'] = 'development'
    elif change == 'model': plan['config']['model_seed'] = 43
    elif change == 'prompt': plan['recovery_prompt_path'] = 'probe.txt'
    elif change == 'partition': plan['blocks'][0]['task_specification']['development_partition'] = True
    else: plan['episodes_planned'] = 24
    with pytest.raises(ValueError): final.verify_structure(plan, p, old)


def test_scoped_verifier_delegates_and_restores_even_after_failure(tmp_path, monkeypatch):
    historical = tmp_path/'historical'; historical.mkdir()
    (historical/'plan.json').write_text(json.dumps({'profile': 'old'}))
    new = tmp_path/'new'; new.mkdir()
    (new/'plan.json').write_text(json.dumps({'profile': final.PROFILE}))
    old = lambda *a, **kw: 'historical verification'
    monkeypatch.setattr(final.study, 'verify_plan', old)
    def reject(*a, **kw): raise ValueError('broken final evidence')
    monkeypatch.setattr(final, 'verify_final', reject)
    with pytest.raises(ValueError, match='broken final evidence'):
        with final.final_verifier():
            assert final.study.verify_plan(historical) == 'historical verification'
            final.study.verify_plan(new)
    assert final.study.verify_plan is old


def test_final_cli_cannot_launch_a_historical_profile(tmp_path, monkeypatch):
    import sys
    (tmp_path/'plan.json').write_text(json.dumps({'profile': 'miniwob-hybrid-dev-v2'}))
    monkeypatch.setattr(sys, 'argv', ['runner', 'run', '--output', str(tmp_path)])
    with pytest.raises(ValueError, match='only runs/audits'):
        final.main()
    assert not (tmp_path/'.coordinator.lock').exists()


def test_shared_loop_executes_final_partition_with_scripted_actors(tmp_path, monkeypatch):
    from tests.table2.test_hybrid_runner import engineering_run
    original_run = final.study.run
    def final_fixture(out):
        # engineering_run supplies fake actors/browser and an in-memory verifier.
        # Only the final partition/configuration changes for this wiring check.
        plan = final.study.verify_plan(out)
        plan['phase'] = 'evaluation'
        plan['config']['development_only'] = False
        plan['profile'] = plan['config']['profile'] = final.PROFILE
        for block in plan['blocks']:
            block['task_specification']['development_partition'] = False
        (out/'plan.json').write_text(json.dumps(plan))
        original_run(out)
    monkeypatch.setattr(final.study, 'run', final_fixture)
    plan, rows, audit = engineering_run(tmp_path/'fixture', monkeypatch, interface_version=2)
    assert audit['status'] == 'PASS', audit['errors']
    assert plan['phase'] == 'evaluation' and len(rows) == 8
    assert all('error' not in row for row in rows)
    assert audit['live_path_verified'] is False  # scripted, not live model success
