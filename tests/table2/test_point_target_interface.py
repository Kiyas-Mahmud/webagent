from dataclasses import replace
import pytest
from web_agent.runtime.contracts import ActionType,ConcreteAction,PolicyObservation,Observation,ObservationStage,RuntimeTaskView
from web_agent.runtime.recovery.strategies import (build_recovery_target_evidence,build_point_target_evidence,validate_recovery_target_evidence,actions_share_semantic_target,RecoveryResolutionError)
from web_agent.runtime.decision import LoopGuard
from web_agent.runtime.protocol import LoopRule


def action(box=(.1,.2,.3,.2),kind=ActionType.CLICK):
    x,y,w,h=box
    return ConcreteAction(action_id='a',source_decision_id='d',action_type=kind,bbox=box,parameters={'target_x':x+w/2,'target_y':y+h/2,'button':'left','click_count':1})


def observation(controls):
    task=RuntimeTaskView(task_id='task',goal='Click a visible control')
    state={'recovery_target_evidence':build_recovery_target_evidence(task=task,observation_id='obs',compatible_actions=controls),'visible_point_targets':build_point_target_evidence(task=task,observation_id='obs',compatible_actions=controls)}
    obs=PolicyObservation(task_id=task.task_id,goal=task.goal,observation_id='obs',screenshot_sha256='a'*64,screenshot_path=None,width=1280,height=720,url='fixture://point',title='point',current_page_state=state)
    return task,obs


def test_approximate_box_uses_unchanged_actual_click_point():
    task,obs=observation([action()]);predicted=action((.2,.25,.04,.04));before=predicted.to_dict()
    validate_recovery_target_evidence(predicted,task=task,post_failure_observation=obs)
    assert predicted.to_dict()==before


def test_point_outside_control_is_rejected():
    task,obs=observation([action()])
    with pytest.raises(RecoveryResolutionError):validate_recovery_target_evidence(action((.7,.7,.1,.1)),task=task,post_failure_observation=obs)


def test_alternative_boxes_on_same_control_are_same_target():
    task,obs=observation([action()])
    assert actions_share_semantic_target(action((.12,.22,.04,.04)),action((.3,.3,.04,.04)),observation=obs) is True


def test_stale_geometry_and_incompatible_type_fail_closed():
    task,obs=observation([action()]);bad=dict(obs.current_page_state)
    bad['visible_point_targets']={**bad['visible_point_targets'],'observation_id':'stale'}
    with pytest.raises(RecoveryResolutionError,match='binding mismatch'):
        validate_recovery_target_evidence(action(),task=task,post_failure_observation=replace(obs,current_page_state=bad))
    with pytest.raises(RecoveryResolutionError):
        validate_recovery_target_evidence(action(kind=ActionType.TYPE),task=task,post_failure_observation=obs)


def test_overlapping_controls_are_not_guessed():
    task,obs=observation([action(),action((.15,.25,.1,.1))])
    with pytest.raises(RecoveryResolutionError):validate_recovery_target_evidence(action((.17,.27,.02,.02)),task=task,post_failure_observation=obs)


def test_changing_capture_id_does_not_hide_an_unchanged_page_loop():
    guard=LoopGuard(LoopRule());task,obs=observation([action()]);trigger=[]
    for i in range(3):
        state={k:{**v,'observation_id':f'obs-{i}'} for k,v in obs.current_page_state.items()}
        visible=Observation(observation_id=f'obs-{i}',episode_id='episode',stage=ObservationStage.PRE_ACTION,screenshot_sha256='a'*64,page_state=state)
        trigger.append(guard.record(visible,action()))
    assert trigger==[False,False,True]
