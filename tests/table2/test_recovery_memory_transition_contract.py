"""Memory after a recovery must remain bound to canonical action evidence."""
from dataclasses import replace
import pytest
from web_agent.runtime.contracts import SystemID,validate_episode_foreign_keys
from web_agent.benchmarks.recovery_fixture import run_failure_memory_intervention_smoke
import web_agent.runtime.episode as episode


def test_recovery_memory_transition_is_validated_without_fake_normal_assessment(tmp_path,monkeypatch):
    captured=[]
    def capture(bundle):
        captured.append(bundle)
        return validate_episode_foreign_keys(bundle)
    monkeypatch.setattr(episode,'validate_episode_foreign_keys',capture)
    run_failure_memory_intervention_smoke(tmp_path/'synthetic')
    bundle=next(b for b in captured if b.summary.system_id is SystemID.E3)
    assert bundle.memory_transition_inputs
    transition=bundle.memory_transition_inputs[-1]
    query=replace(bundle.memory_queries[0],failed_action_id=transition.executed_action.action_id,
        post_failure_observation_id=transition.post_observation.observation_id,
        post_failure_observation_sha256=next(o.record_sha256 for o in bundle.observations if o.observation_id==transition.post_observation.observation_id),
        post_action_input_sha256=transition.record_sha256)
    changed=replace(bundle,memory_queries=(query,))
    validate_episode_foreign_keys(changed)
    with pytest.raises(ValueError,match='transition hash differs'):
        validate_episode_foreign_keys(replace(changed,memory_transition_inputs=()))
    bad=replace(transition,execution_result=replace(transition.execution_result,state_changed=not transition.execution_result.state_changed))
    with pytest.raises(ValueError,match='embedded execution differs'):
        validate_episode_foreign_keys(replace(changed,memory_transition_inputs=(bad,)))
