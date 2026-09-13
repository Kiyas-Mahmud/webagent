"""Real-browser fixture for overlapping-target resolution; no model inference.

Development-v3 rejected every `click-button-sequence` recovery proposal with
`TARGET_POINT_MATCH_COUNT:2`. This replays the actual MiniWoB reset that
produced those rejections and checks, against the live page, that:

  * each button the model could name now resolves to that exact control,
  * the executed point is hit-tested by the browser to that same DOM element,
  * a point the browser assigns elsewhere is still rejected, and
  * the scripted ONE-then-TWO sequence actually completes in the browser.

This is an engineering fixture. It makes no model call and is not evidence of
model efficacy.
"""
from pathlib import Path
import json
import sys

import gymnasium as gym
import browsergym.miniwob  # noqa: F401
from types import SimpleNamespace

from web_agent.benchmarks.browsergym_webarena import _action_code
from web_agent.benchmarks.miniwob_controls import (
    CONTROL_INTERFACE, CONTROL_JAVASCRIPT, action_point, bind_controls, validate_control_action)
from web_agent.benchmarks.miniwob_support import raw_reward_outcome
from web_agent.runtime.contracts import ActionType, ConcreteAction

# The exact development-v3 click-button-sequence reset.
TASK = 'click-button-sequence'
RESET_SEED = 5528761065506371935

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
env = gym.make('browsergym/miniwob.' + TASK, headless=True, action_mapping=lambda a: a)
obs, _ = env.reset(seed=RESET_SEED % (2**32))
page = env.unwrapped.page
checks = []


def observe(index):
    oid = f'fixture:obs:{index}'
    controls = bind_controls(page.evaluate(CONTROL_JAVASCRIPT), oid)
    view = SimpleNamespace(observation_id=oid, task_id='miniwob.' + TASK,
                           current_page_state={'control_interface': CONTROL_INTERFACE,
                                               'visible_controls': controls})
    return view, controls


def dom_at(point):
    """What the page itself says receives a click at this normalized point.

    Reported with the same identity the control projection uses, so the answer
    is directly comparable to the resolved control's ``source_id``.
    """
    size = page.viewport_size
    return page.evaluate("([x,y]) => { const e = document.elementFromPoint(x,y);"
                         " return e ? String(e.getAttribute('bid') || e.id || e.tagName) : null; }",
                         [point[0]*size['width'], point[1]*size['height']])


def click(control, view):
    point = action_point(control)
    parameters = {'target_x': point[0], 'target_y': point[1], 'target_bbox': control['target_bbox'],
                  'target_control_id': control['control_id'], 'button': 'left', 'click_count': 1}
    resolved = validate_control_action('CLICK', parameters, tuple(control['target_bbox']), view,
                                       control_id=control['control_id'])
    action = ConcreteAction(action_id='fixture:' + control['control_id'], source_decision_id='scripted',
                            action_type=ActionType.CLICK, bbox=tuple(control['target_bbox']),
                            parameters=parameters)
    return resolved, point, action


view, controls = observe(1)
overlapping = [c for c in controls
               if any(other is not c
                      and other['target_bbox'][0] <= c['target_bbox'][0]+c['target_bbox'][2]/2
                      <= other['target_bbox'][0]+other['target_bbox'][2]
                      and other['target_bbox'][1] <= c['target_bbox'][1]+c['target_bbox'][3]/2
                      <= other['target_bbox'][1]+other['target_bbox'][3]
                      for other in controls)]
assert len(overlapping) == 2, ('this reset no longer reproduces the overlap', [c['target_bbox'] for c in controls])

for control in controls:
    resolved, point, _ = click(control, view)
    hit = dom_at(point)
    # The old rule rejected both of these outright.
    try:
        centre = {'target_x': control['target_bbox'][0]+control['target_bbox'][2]/2,
                  'target_y': control['target_bbox'][1]+control['target_bbox'][3]/2,
                  'target_bbox': control['target_bbox'], 'button': 'left', 'click_count': 1}
        validate_control_action('CLICK', centre, tuple(control['target_bbox']), view)
        geometric = 'RESOLVED'
    except ValueError as exc:
        geometric = str(exc)
    assert resolved['control_id'] == control['control_id'], 'named control was not preserved'
    assert hit == control['source_id'], ('browser hit a different element', hit, control['source_id'])
    checks.append({'control_id': control['control_id'], 'text': control['text'],
                   'source_id': control['source_id'], 'target_bbox': control['target_bbox'],
                   'executed_point': list(point), 'browser_element_at_executed_point': hit,
                   'previous_geometric_rule': geometric, 'resolved_control_id': resolved['control_id']})

# A point the browser assigns to another control is still rejected, not repaired.
covered = next(c for c in controls if dom_at((c['target_bbox'][0]+c['target_bbox'][2]/2,
                                              c['target_bbox'][1]+c['target_bbox'][3]/2)) != c['source_id'])
occluded_centre = {'target_x': covered['target_bbox'][0]+covered['target_bbox'][2]/2,
                   'target_y': covered['target_bbox'][1]+covered['target_bbox'][3]/2,
                   'target_bbox': covered['target_bbox'], 'target_control_id': covered['control_id'],
                   'button': 'left', 'click_count': 1}
try:
    validate_control_action('CLICK', occluded_centre, tuple(covered['target_bbox']), view,
                            control_id=covered['control_id'])
    raise AssertionError('an unverified point was accepted')
except ValueError as exc:
    assert 'TARGET_POINT_NOT_BROWSER_VERIFIED' in str(exc), str(exc)
    checks.append({'check': 'occluded_centre_rejected', 'control_id': covered['control_id'],
                   'point': [occluded_centre['target_x'], occluded_centre['target_y']], 'rejection': str(exc)})

# The full scripted sequence must now actually run in the browser.
outcomes = []
for index, text in enumerate(['ONE', 'TWO'], start=2):
    view, controls = observe(index)
    control = next(c for c in controls if c['text'] == text)
    _, point, action = click(control, view)
    _, reward, terminated, truncated, info = env.step(_action_code(action))
    outcomes.append(raw_reward_outcome(reward=reward, terminated=terminated, truncated=truncated, info=info))
    checks.append({'check': 'executed', 'text': text, 'control_id': control['control_id'],
                   'executed_point': list(point), 'outcome': outcomes[-1]})

env.close()
completed = outcomes[-1]['terminated'] and not outcomes[-1]['truncated'] and outcomes[-1]['raw_reward'] == 1.0
assert completed, ('scripted ONE-then-TWO sequence did not complete', outcomes)
(out/'result.json').write_text(json.dumps({
    'status': 'PASS', 'evidence_type': 'SCRIPTED_BROWSER_FIXTURE', 'model_calls': 0,
    'task': TASK, 'reset_seed': RESET_SEED, 'goal': obs['goal'], 'checks': checks}, indent=2)+'\n')
print('PASS: overlapping controls resolve to the named control and the scripted sequence completes')
