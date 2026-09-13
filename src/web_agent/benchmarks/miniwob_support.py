"""Visible custom controls and explicit raw-reward scoring for revised MiniWoB.

Scoring runs outside the observation/policy path. Nothing here reads task
generator variables, target answers or event-handler bodies for action choice.
"""
import math

from web_agent.benchmarks.browsergym_webarena import (
    _CONTROL_SNAPSHOT_JAVASCRIPT, _clean_control_snapshot,
)

CONTROL_INTERFACE = 'visible-pointer-controls-v1'
SCORING_RULE = 'terminated-and-raw-reward-equals-one-v1'

# Preserve native control observations, then add visible custom elements with
# a public link role or a pointer cursor. Descendants of native controls are
# already represented by their owning control and must not become duplicates.
MINIWOB_CONTROL_SNAPSHOT_JAVASCRIPT = r"""
() => {
  const result = (BASE_SNAPSHOT)();
  for (const control of result.controls) {
    control.clickable = true;
    control.link_like = control.tag === 'a' || control.role === 'link';
  }
  const width = Math.max(window.innerWidth, 1);
  const height = Math.max(window.innerHeight, 1);
  const round = value => Math.round(value * 1000000) / 1000000;
  const native = 'a,button,input,textarea,select,[role="button"],[contenteditable="true"]';
  for (const element of document.querySelectorAll('[role="link"],span,div')) {
    if (element.closest(native)) continue;
    const style = window.getComputedStyle(element);
    const role = String(element.getAttribute('role') || '');
    if (role !== 'link' && style.cursor !== 'pointer') continue;
    // Use the innermost visible target, avoiding pointer-cursor wrapper duplicates.
    if (Array.from(element.children).some(child =>
        window.getComputedStyle(child).cursor === 'pointer' &&
        child.getBoundingClientRect().width > 0 && child.getBoundingClientRect().height > 0)) continue;
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0 || style.visibility === 'hidden' ||
        style.display === 'none' || rect.bottom <= 0 || rect.right <= 0 ||
        rect.top >= height || rect.left >= width) continue;
    const left = Math.max(0, rect.left), top = Math.max(0, rect.top);
    const right = Math.min(width, rect.right), bottom = Math.min(height, rect.bottom);
    result.controls.push({
      tag: String(element.tagName).toLowerCase(), role,
      input_type: '',
      name: String(element.getAttribute('aria-label') || element.getAttribute('name') || '').slice(0, 160),
      text: String(element.innerText || element.textContent || '').trim().slice(0, 240),
      bbox: [round(left/width), round(top/height), round((right-left)/width), round((bottom-top)/height)],
      candidate_options: [], destination: '', clickable: true,
      link_like: role === 'link' || style.textDecorationLine.includes('underline')
    });
  }
  result.controls = result.controls.slice(0, 512);
  return result;
}
""".replace('BASE_SNAPSHOT', _CONTROL_SNAPSHOT_JAVASCRIPT)


def clean_miniwob_control(value):
    value = dict(value)
    clickable = value.pop('clickable')
    link_like = value.pop('link_like')
    if type(clickable) is not bool or type(link_like) is not bool:
        raise ValueError('observed control flags must be boolean')
    return {**_clean_control_snapshot(value), 'clickable': clickable, 'link_like': link_like}


def raw_reward_outcome(*, reward, terminated, truncated, info):
    """Capture the reward already returned by task validation; never query the DOM."""
    raw = info.get('task_info', {}).get('RAW_REWARD_GLOBAL')
    if type(raw) not in (int, float) or not math.isfinite(raw) or not -1 <= raw <= 1:
        raise ValueError('MiniWoB raw reward is missing or outside [-1, 1]')
    return {'reward': float(reward), 'raw_reward': float(raw),
            'terminated': bool(terminated), 'truncated': bool(truncated)}


def full_completion(outcomes):
    """Positive partial credit is retained as partial credit, not full completion."""
    for outcome in outcomes:
        raw = outcome.get('raw_reward')
        if type(raw) not in (int, float) or not math.isfinite(raw) or not -1 <= raw <= 1:
            raise ValueError('cannot score completion without a valid raw reward')
    return any(o['terminated'] and not o['truncated'] and o['raw_reward'] == 1.0
               for o in outcomes)
