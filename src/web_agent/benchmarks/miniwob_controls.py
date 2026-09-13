"""Observable MiniWoB controls. No task variables, evaluator state or answers."""
from collections.abc import Mapping
import json

CONTROL_INTERFACE = 'miniwob-observable-controls-v3'
# v2 archives carry no hit-target evidence. They stay replayable and keep their
# original geometric resolution; only v3 observations use the browser hit point.
LEGACY_CONTROL_INTERFACES = ('miniwob-observable-controls-v2',)
OBSERVED_CONTROL_INTERFACES = (CONTROL_INTERFACE,) + LEGACY_CONTROL_INTERFACES
STABLE_TARGET_FIELD = 'target_stable_identity'
STABLE_TARGET_SCHEMA = 'miniwob-stable-target-v1'


def uses_stable_targets(page_state):
    """The bounded hybrid revision alone opts into cross-capture identity."""
    return page_state.get('hybrid_interface_version') == 3 and is_observable_controls(page_state)


def stable_control_identity(control):
    """Exact observed source and presentation evidence, without capture IDs.

    A source ID alone can be reused. Keep every other observed property too,
    including geometry, values, capability flags and the browser hit point.
    Absence of a source ID cannot establish cross-capture control identity.
    """
    from web_agent.runtime.contracts import canonical_sha256
    source_id = control.get('source_id')
    if not isinstance(source_id, str) or not source_id.strip():
        return None
    semantic = {k: v for k, v in control.items()
                if k not in {'control_id', 'observation_id', 'control_interface'}}
    return {'source_id': source_id, 'semantic_sha256': canonical_sha256(semantic)}


def stable_target_receipt(control, observation):
    """Runtime-owned evidence for one uniquely identified current control."""
    from web_agent.runtime.contracts import canonical_sha256
    if not uses_stable_targets(observation.current_page_state):
        return None
    identity = stable_control_identity(control)
    if identity is None:
        return None
    controls = observation.current_page_state.get('visible_controls', ())
    same_source = [c for c in controls if c.get('source_id') == identity['source_id']]
    if len(same_source) != 1 or same_source[0] != control:
        return None
    if control.get('observation_id') != observation.observation_id:
        raise ValueError('STALE_CONTROL_OBSERVATION')
    return {'schema_version': STABLE_TARGET_SCHEMA,
            'task_id': observation.task_id, 'goal_sha256': canonical_sha256(observation.goal),
            'observation_id': observation.observation_id, 'control_id': control['control_id'],
            **identity}


def validate_stable_target_receipt(receipt):
    fields = {'schema_version', 'task_id', 'goal_sha256', 'observation_id',
              'control_id', 'source_id', 'semantic_sha256'}
    if (not isinstance(receipt, Mapping) or set(receipt) != fields
            or receipt.get('schema_version') != STABLE_TARGET_SCHEMA
            or any(not isinstance(v, str) or not v.strip() for v in receipt.values())):
        raise ValueError('INVALID_STABLE_TARGET_RECEIPT')
    for field in ('goal_sha256', 'semantic_sha256'):
        if len(receipt[field]) != 64 or any(c not in '0123456789abcdef' for c in receipt[field]):
            raise ValueError('INVALID_STABLE_TARGET_RECEIPT')
    return {'source_id': receipt['source_id'], 'semantic_sha256': receipt['semantic_sha256']}


def is_observable_controls(page_state):
    return page_state.get('control_interface') in OBSERVED_CONTROL_INTERFACES

# Only DOM presentation/interaction properties are read. Password contents are
# never exported; the actor can remember values it issued itself.
CONTROL_JAVASCRIPT = r'''() => {
  const width = Math.max(innerWidth, 1), height = Math.max(innerHeight, 1);
  const native = 'a,button,input,textarea,select,[role="button"],[contenteditable="true"]';
  const visible = e => { const r=e.getBoundingClientRect(), s=getComputedStyle(e);
    return r.width>0 && r.height>0 && r.bottom>0 && r.right>0 && r.top<height && r.left<width &&
      s.display!=='none' && s.visibility!=='hidden' && s.opacity!=='0'; };
  const text = e => String(e.innerText || e.textContent || '').trim();
  const candidates = Array.from(document.querySelectorAll(native));
  for (const e of document.querySelectorAll('[role="link"],span,div')) {
    if(e.closest(native)) continue;
    if(e.getAttribute('role')!=='link' && getComputedStyle(e).cursor!=='pointer') continue;
    if(Array.from(e.children).some(c=>visible(c) && getComputedStyle(c).cursor==='pointer')) continue;
    candidates.push(e);
  }
  const round = x=>Math.round(x*1000000)/1000000;
  const shown = candidates.filter(visible).slice(0,512);
  const rank = new Map(shown.map((e,i)=>[e,i]));
  // Which listed control actually receives a click at a point, as the browser
  // itself resolves it. Nested candidates resolve to the innermost listed one.
  const owner = (x,y) => { for(let e=document.elementFromPoint(x,y); e; e=e.parentElement)
    if(rank.has(e)) return rank.get(e);
    return -1; };
  // Deterministic, read-only probe of the control's own visible area. It never
  // leaves the control's box and never selects a different control.
  const fractions = [0.5, 0.25, 0.75, 0.1, 0.9];
  const hitPoint = (e,i) => {
    const r=e.getBoundingClientRect();
    const left=Math.max(0,r.left), top=Math.max(0,r.top), right=Math.min(width,r.right), bottom=Math.min(height,r.bottom);
    if(!(right-left>0 && bottom-top>0)) return null;
    for(const fy of fractions) for(const fx of fractions) {
      const x=Math.min(Math.max(left+fx*(right-left), left+0.5), right-0.5);
      const y=Math.min(Math.max(top+fy*(bottom-top), top+0.5), bottom-0.5);
      if(owner(x,y)===i) return [round(x/width), round(y/height)];
    }
    return null;
  };
  return {controls: shown.map((e,control_index)=>{
    const r=e.getBoundingClientRect(), s=getComputedStyle(e), tag=e.tagName.toLowerCase();
    const input_type=String(e.getAttribute('type')||'').toLowerCase();
    const labels=[];
    const add=(value,source)=>{value=String(value||'').trim().slice(0,240);
      if(value && !labels.some(x=>x.value===value && x.source===source)) labels.push({value,source});};
    for(const l of Array.from(e.labels||[])) if(visible(l)) add(text(l),'associated_label');
    for(const id of String(e.getAttribute('aria-labelledby')||'').split(/\s+/)) {
      const l=document.getElementById(id); if(l && visible(l)) add(text(l),'aria_labelledby');
    }
    add(e.getAttribute('aria-label'),'aria_label');
    const parent=e.parentElement;
    if(!labels.length && parent) {
      const peers=Array.from(parent.querySelectorAll('input,textarea,select,[contenteditable="true"]')).filter(visible);
      const ls=Array.from(parent.children).filter(l=>l.tagName==='LABEL' && visible(l));
      if(peers.length===1 && peers[0]===e && ls.length===1 && !ls[0].control) add(text(ls[0]),'unique_sibling_label');
    }
    add(e.getAttribute('placeholder'),'placeholder');
    const buttonValue=tag==='input' && ['button','submit','reset'].includes(input_type);
    const displayText=['input','textarea','select'].includes(tag) ? (buttonValue?String(e.value||''):'') : text(e);
    const editable=tag==='textarea' || tag==='input' || e.isContentEditable;
    const value=editable && input_type!=='password' && !['radio','checkbox'].includes(input_type) ? String(e.value||'').slice(0,240) : null;
    const left=Math.max(0,r.left), top=Math.max(0,r.top), right=Math.min(width,r.right), bottom=Math.min(height,r.bottom);
    return {tag, input_type, role:String(e.getAttribute('role')||''),
      source_id:String(e.getAttribute('bid')||e.id||''), name:String(e.getAttribute('name')||'').slice(0,160),
      text:displayText.slice(0,240), accessible_names:labels, value,
      value_present:editable ? Boolean(e.value || e.innerText) : false,
      target_bbox:[round(left/width),round(top/height),round((right-left)/width),round((bottom-top)/height)],
      candidate_options:tag==='select'?Array.from(e.options).map(o=>o.value||text(o)).slice(0,256):[],
      destination:tag==='a'?String(e.href||''):'', clickable:true,
      link_like:tag==='a'||e.getAttribute('role')==='link'||s.textDecorationLine.includes('underline'),
      contenteditable:Boolean(e.isContentEditable), disabled:Boolean(e.disabled)||e.getAttribute('aria-disabled')==='true',
      readonly:Boolean(e.readOnly)||e.getAttribute('aria-readonly')==='true',
      checked: ['radio','checkbox'].includes(input_type)?Boolean(e.checked):null,
      selected:tag==='select'?Array.from(e.selectedOptions).map(o=>o.value||text(o)):[],
      focused:document.activeElement===e, hit_point:hitPoint(e,control_index)};
  })};
}'''


def supported_actions(control):
    """One capability rule shared by descriptions, resolver and executor."""
    if control.get('disabled') is True:
        return ()
    tag, kind = control.get('tag'), control.get('input_type', '')
    actions = []
    if tag in {'a','button','input','textarea','select'} or control.get('clickable') is True or control.get('role') in {'button','link'} or control.get('contenteditable') is True:
        actions.append('CLICK')
    editable = (tag == 'textarea' or control.get('contenteditable') is True or
                (tag == 'input' and kind not in {'button','checkbox','radio','submit','reset','file','image','hidden','range','color'}))
    if editable and not control.get('readonly'):
        actions.append('TYPE')
    if tag == 'select' and not control.get('readonly'):
        actions.append('SELECT')
    return tuple(actions)


def bind_controls(snapshot, observation_id):
    from urllib.parse import urlsplit
    from web_agent.runtime.contracts import _validate_bbox
    controls = []
    for i, raw in enumerate(snapshot['controls']):
        c = dict(raw)
        _validate_bbox(tuple(c['target_bbox']))
        for field in ('disabled','readonly','focused','value_present','contenteditable','clickable','link_like'):
            if type(c.get(field)) is not bool:
                raise ValueError('invalid visible control flag: '+field)
        if c['checked'] is not None and type(c['checked']) is not bool:
            raise ValueError('invalid checked flag')
        if any(set(l)!={'value','source'} or not all(isinstance(v,str) for v in l.values()) for l in c['accessible_names']):
            raise ValueError('invalid visible label')
        # A v2 snapshot has no hit evidence; it keeps the original geometric path.
        c['hit_point'] = _validated_hit_point(c.get('hit_point'), c['target_bbox'])
        url=urlsplit(c['destination'])
        if url.scheme not in {'http','https'} or not url.hostname or url.username or url.password:
            c['destination']=''
        c.update(control_id=f"o{observation_id.rsplit(':',1)[-1]}:c{i}", observation_id=observation_id,
                 control_interface=CONTROL_INTERFACE, supported_actions=list(supported_actions(c)))
        controls.append(c)
    return controls


def _validated_hit_point(point, box):
    """The browser-reported click point for a control, inside its own box."""
    if point is None:
        return None
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError('invalid control hit point')
    x, y = (float(v) for v in point)
    left, top, width, height = box
    if not (left <= x <= left+width and top <= y <= top+height):
        raise ValueError('control hit point lies outside its own target box')
    return [x, y]


def hit_point(control):
    point = control.get('hit_point')
    return list(point) if isinstance(point, (list, tuple)) and len(point) == 2 else None


def action_point(control):
    """The point an action on this control executes at.

    The browser-verified hit point when one exists, otherwise the box centre
    that this interface has always used. Never a point outside the control.
    """
    point = hit_point(control)
    if point is not None:
        return tuple(point)
    x, y, width, height = control['target_bbox']
    return (x + width/2, y + height/2)


def aliases(control):
    values=[control.get('name',''),control.get('text','')]
    values.extend(x['value'] for x in control.get('accessible_names',()))
    return {' '.join(v.casefold().split()) for v in values if isinstance(v,str) and v.strip()}


# The executed point is interface-owned detail; the model targets controls by
# ID and its prompt context stays exactly as it was under the v2 projection.
_PROMPT_HIDDEN_FIELDS = {'observation_id','source_id','control_interface','hit_point'}
# The model-facing projection did not change when hit evidence was added, so its
# advertised schema must not change either: the observation's own interface
# version would otherwise alter the prompt bytes and move generations.
PROMPT_CONTROL_SCHEMA = 'miniwob-observable-controls-v2'


def prompt_controls(observation):
    # The containing observation binds the list; don't repeat long IDs per control.
    return [{k:v for k,v in c.items() if k not in _PROMPT_HIDDEN_FIELDS}
            for c in observation.current_page_state.get('visible_controls',())]


def control_suffix(observation):
    if not is_observable_controls(observation.current_page_state):
        return ''
    return '\ncurrent_controls: '+json.dumps({'schema':PROMPT_CONTROL_SCHEMA,
        'controls':prompt_controls(observation)},sort_keys=True)


def validate_control_action(action_type, values, bbox, observation, *, control_id=None):
    """Resolve the control this action actually acts on, without changing it.

    ``control_id`` is the control the model itself named. Box containment counts
    cannot decide an overlapping target: the page paints one control over the
    other, so the browser's own hit test at the executed point is the authority.
    Without a named control the original geometric rule is unchanged.
    """
    if not is_observable_controls(observation.current_page_state) or action_type not in {'CLICK','TYPE','SELECT'}:
        return None
    if bbox is None:
        raise ValueError('TARGET_BOX_MISSING')
    x=values.get('target_x',bbox[0]+bbox[2]/2); y=values.get('target_y',bbox[1]+bbox[3]/2)
    controls=observation.current_page_state['visible_controls']
    for c in controls:
        if c['observation_id'] != observation.observation_id:
            raise ValueError('STALE_CONTROL_OBSERVATION')
    if control_id is not None:
        named=[c for c in controls if c['control_id']==control_id]
        if len(named)!=1:
            raise ValueError(f'TARGET_CONTROL_UNRESOLVED:{control_id}')
        point=hit_point(named[0])
        if point is None:
            # Nothing inside this control receives a click: it is covered.
            raise ValueError(f'TARGET_CONTROL_NOT_HIT_TESTABLE:{control_id}')
        if (round(x,6),round(y,6)) != (round(point[0],6),round(point[1],6)):
            raise ValueError(f'TARGET_POINT_NOT_BROWSER_VERIFIED:{control_id}')
        matches=named
    else:
        matches=[c for c in controls
                 if c['target_bbox'][0]<=x<=c['target_bbox'][0]+c['target_bbox'][2]
                 and c['target_bbox'][1]<=y<=c['target_bbox'][1]+c['target_bbox'][3]]
        if len(matches)!=1:
            raise ValueError(f'TARGET_POINT_MATCH_COUNT:{len(matches)}')
    if action_type not in supported_actions(matches[0]):
        raise ValueError(f"ACTION_TARGET_INCOMPATIBLE:{action_type}:{matches[0]['control_id']}")
    if uses_stable_targets(observation.current_page_state) and STABLE_TARGET_FIELD in values:
        receipt = values[STABLE_TARGET_FIELD]
        identity = validate_stable_target_receipt(receipt)
        if (receipt['control_id'] != matches[0]['control_id']
                or receipt['observation_id'] != observation.observation_id
                or stable_control_identity(matches[0]) != identity):
            raise ValueError('STABLE_TARGET_CHANGED_BEFORE_EXECUTION')
        if sum(c.get('source_id') == identity['source_id'] for c in controls) != 1:
            raise ValueError('STABLE_TARGET_AMBIGUOUS_BEFORE_EXECUTION')
    return matches[0]


def compatible_actions(controls):
    from web_agent.runtime.contracts import ActionType, ConcreteAction
    actions=[]
    for i,c in enumerate(controls):
        x,y=action_point(c)
        for kind in supported_actions(c):
            actions.append(ConcreteAction(action_id=f'control-{i}-{kind}',source_decision_id=CONTROL_INTERFACE,
                action_type=ActionType(kind),bbox=tuple(c['target_bbox']),
                parameters={'target_x':x,'target_y':y,'target_bbox':c['target_bbox']}))
        if c.get('destination'):
            actions.append(ConcreteAction(action_id=f'control-{i}-NAVIGATE',source_decision_id=CONTROL_INTERFACE,
                action_type=ActionType.NAVIGATE,parameters={'url':c['destination']}))
    actions.append(ConcreteAction(action_id='viewport-scroll',source_decision_id=CONTROL_INTERFACE,
        action_type=ActionType.SCROLL,parameters={'container':'viewport'}))
    return actions
