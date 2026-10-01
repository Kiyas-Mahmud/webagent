"""Native Browser Use execution on BrowserGym-owned MiniWoB tasks."""
from dataclasses import asdict, dataclass, field
import asyncio
import json
import os
from pathlib import Path
import time

from web_agent.eval.task1.core import write_new
from web_agent.eval.task2.assessment import Observation, ExecutedTransition, describe_action, recovery_advice, recovery_note
from web_agent.eval.task2.native_model import LocalInternVL
from web_agent.eval.task2.elements import element_summary

NATIVE_ACTIONS={'click':'CLICK','input':'TYPE','select_dropdown':'SELECT','scroll':'SCROLL',
                'navigate':'NAVIGATE','send_keys':'PRESS_KEY','go_back':'NAVIGATE'}


class BudgetExhausted(RuntimeError):pass


@dataclass
class Budget:
    max_model_calls: int = 102
    max_executor_requests: int = 30
    seconds: float = 600
    calls: dict = field(default_factory=dict)
    executor_requests: int = 0
    started: float = field(default_factory=time.monotonic)

    def available(self, calls=1):
        return sum(self.calls.values())+calls <= self.max_model_calls and time.monotonic()-self.started < self.seconds

    def charge(self, kind):
        if not self.available():raise BudgetExhausted('Total model/wall-clock budget exhausted')
        self.calls[kind]=self.calls.get(kind,0)+1


def observation(raw):
    return Observation(**{k:raw[k] for k in Observation.__dataclass_fields__})


def recorded_action(native):
    if len(native)!=1:raise ValueError('Expected one native action')
    name,params=next(iter(native.items()))
    if name=='done':return None
    if name not in NATIVE_ACTIONS:raise ValueError('Unsupported native action: '+name)
    target=str(params['index']) if params.get('index') is not None else None
    if name=='click' and target is None:
        target=json.dumps({k:params[k] for k in ('coordinate_x','coordinate_y') if k in params},sort_keys=True)
    value=params.get('text',params.get('url',params.get('keys')))
    if name=='scroll':value=json.dumps(params,sort_keys=True)
    if name=='go_back':value=None
    return dict(action_type=NATIVE_ACTIONS[name],target=target,value=value)


def interacted_element(agent):
    """Observable description of the element the last native action hit (P4 experience key)."""
    try:
        items=agent.history.history[-1].state.interacted_element or []
    except (AttributeError,IndexError):
        return None
    item=items[0] if items else None
    if item is None:return None
    return element_summary(item.to_dict() if hasattr(item,'to_dict') else item)


def interacted_xpath(agent):
    """Stable identity of the element the last native action hit (v3 repeat guard)."""
    try:
        items=agent.history.history[-1].state.interacted_element or []
    except (AttributeError,IndexError):
        return None
    item=items[0] if items else None
    return getattr(item,'x_path',None) if item is not None else None


def same_action(blocked, proposed, xpath):
    """True when a proposal repeats an action the assessor judged failed on this page."""
    if blocked['action_type']!=proposed['action_type']:return False
    kind=proposed['action_type']
    if kind=='SCROLL':
        return ('"down": false' in (blocked['value'] or ''))==('"down": false' in (proposed['value'] or ''))
    if kind in ('NAVIGATE','PRESS_KEY'):
        return (blocked['value'] or '')==(proposed['value'] or '')
    element=blocked['xpath'] is not None and blocked['xpath']==xpath or (
        blocked['xpath'] is None and xpath is None and blocked['target']==proposed['target'])
    if kind=='TYPE':
        return element and (blocked['value'] or '').strip()==(proposed['value'] or '').strip()
    return element


def page_path(url):
    """Blocks are scoped to a page path, so a changed query string (?query=) does not reset them."""
    from urllib.parse import urlsplit
    parts=urlsplit(url or '')
    return (parts.hostname or '')+parts.path.rstrip('/')


def search_query(goal, url):
    """Search text taken from the task goal itself: a quoted phrase, else its content words."""
    import re
    from urllib.parse import urlsplit
    from web_agent.eval.task2.web_worker import STOPWORDS
    quoted=re.search(r'"([^"]+)"',goal)
    if quoted:return quoted.group(1)
    site=set(re.findall(r'[a-z0-9]+',(urlsplit(url or '').hostname or '').lower()))
    words=[w for w in re.findall(r'[A-Za-z0-9]+',goal) if len(w)>2 and w.lower() not in STOPWORDS
           and w.lower() not in site and w.lower() not in {'search','page','pages'}]
    return ' '.join(words[:4])


def search_box(selector_map):
    """Index of a visible site-search input in Browser Use's own element list."""
    import re
    for index,node in selector_map.items():
        if str(getattr(node,'node_name','')).upper() not in ('INPUT','TEXTAREA'):continue
        attributes=getattr(node,'attributes',None) or {}
        text=' '.join(str(attributes.get(k,'')) for k in ('type','name','id','placeholder','aria-label','role','title'))
        if attributes.get('type')=='search' or re.search(r'search|query|(^| )q( |$)',text,re.I):
            return index
    return None


async def create_agent(reset, local_model, output):
    from browser_use import Agent, BrowserSession
    from browser_use.tools.service import Tools
    from browser_use.browser.events import SwitchTabEvent
    tools=Tools()
    excluded=[name for name in tools.registry.registry.actions if name not in {*NATIVE_ACTIONS,'done'}]
    tools=Tools(exclude_actions=excluded)
    browser=BrowserSession(cdp_url=reset['cdp_url'],is_local=False,keep_alive=True,
                           enable_default_extensions=False,headless=True,
                           viewport=reset['viewport'],device_scale_factor=1,
                           allowed_domains=reset.get('allowed_domains',['127.0.0.1','localhost']))
    agent=Agent(task=reset['goal'],llm=local_model,browser_session=browser,tools=tools,
                use_vision=True,use_thinking=False,use_judge=False,
                max_actions_per_step=1,max_failures=5,max_history_items=6,
                message_compaction=False,directly_open_url=False,enable_signal_handler=False,
                file_system_path=str(Path(output)/'agent-files'),
                save_conversation_path=str(Path(output)/'native-conversations'),
                calculate_cost=False,step_timeout=600,llm_timeout=600,
                final_response_after_failure=False)
    try:
        await browser.start()
        tabs=await browser.get_tabs()
        target=next((t for t in tabs if t.url==reset['url']),None)
        if target is None:raise RuntimeError('Benchmark task tab not found')
        await browser.event_bus.dispatch(SwitchTabEvent(target_id=target.target_id))
    except Exception:
        await browser.stop()
        raise
    return agent,browser


async def episode(*, system, block, root, browser_worker, model_worker, settings):
    if system not in ('A','B','C'):raise ValueError('Unknown system')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    episode_id=f"{block['task']}-r{block['repeat_id']}-{system}"
    write_new(root/'started.json',{'episode_id':episode_id,'system':system,'block':block,'settings':settings})
    # Loading assets is setup; recorded per-call latency still includes lazy loads.
    reset=await asyncio.to_thread(browser_worker.call,'reset',episode_id=episode_id,task=block['task'],seed=block['browser_reset_seed'])
    if reset['goal'] != block['goal']:raise RuntimeError('Frozen reset goal mismatch')
    await asyncio.to_thread(model_worker.call,'reset')
    budget=Budget(**settings['budget'])
    llm=LocalInternVL(model_worker,root,budget,settings.get('actor_label'))
    experience=system=='C' and settings.get('memory_mode')=='experience'
    if experience:llm.memory_kind='experience'
    pending=None
    agent,browser=await create_agent(reset,llm,root)
    before=await asyncio.to_thread(browser_worker.call,'observe',episode_id=episode_id)
    write_new(root/'observation-0000.json',before)
    incident=None; incident_attempts=0; recovery_total=0; next_is_recovery=False; incident_diagnosis=None
    counters={'executed':0,'rejected':0,'executed_recoveries':0,'assessments':0,'memory_queries':0,'memory_exposures':0}
    if experience:counters['memory_writes']=0

    async def close_incident(resolved):
        # P4 write: the trained memory head decides whether this incident is worth keeping.
        nonlocal pending
        if pending is None:return
        record=dict(pending['record'],resolved=resolved)
        decision={'incident_id':record['incident_id'],'resolved':resolved,'store_probability':pending['store_probability'],
                  'stored':pending['store_probability']>0.5 and bool(settings['experience_memory'].get('write'))}
        if decision['stored']:
            written=await asyncio.to_thread(model_worker.call,'experience_write',store=settings['experience_memory'],
                                            record=record,vector=pending['vector'])
            decision.update(memory_id=written['memory_id'],store_size=written['store_size'])
            counters['memory_writes']+=1
        write_new(root/f"experience-{record['incident_id']}.json",dict(decision,record=record))
        pending=None
    history=[]; succeeded=[]; stop='budget'; score={'terminated':False,'raw_reward':0.0}; step=0; steps_completed=0
    # v3 (2026-10-01): assessment overlaps the actor's next page capture; confident
    # failures produce a plain-text note with page facts, and the failed action is
    # blocked on this page. The actor still chooses every action.
    v3=system!='A' and settings.get('recovery_mode')=='v3'
    blocked=[]

    async def guard(completion):
        if not blocked:return None
        try:
            proposed=recorded_action(completion.action[0].model_dump(mode='json',exclude_none=True))
        except Exception:
            return None
        if proposed is None:return None
        xpath=None
        if proposed['target'] and proposed['target'].isdigit():
            node=await agent.browser_session.get_dom_element_by_index(int(proposed['target']))
            xpath=getattr(node,'xpath',None) if node is not None else None
        for item in blocked:
            if same_action(item,proposed,xpath):
                return (f"'{item['label']}' was already tried on this page and did not make progress. "
                        'Choose a different action.')
        return None
    if v3:
        llm.guard=guard; llm.guard_regenerations=settings.get('guard_regenerations',2)

    def link_options(facts):
        return [{'label':f"Open the link '{link['text'] or link['href']}' ({link['href']})",
                 'action':{'navigate':{'url':link['href'],'new_tab':False}}} for link in (facts.get('goal_links') or [])[:3]]

    def search_option(box,query):
        # v3.3: searching is a three-step recovery plan: type, submit, open a matching result.
        return {'label':f"Search this site for '{query}' (type it into the search box, then press Enter)",
                'action':{'input':{'index':box,'text':query,'clear':True}},
                'then':{'label':'Press Enter to submit the search','action':{'send_keys':{'keys':'Enter'}},
                        'then':{'resolve':'results'}}}

    async def recovery_options(advice,followup):
        """v3.1+: concrete Browser Use actions offered after a confident failure (the actor picks one)."""
        task=f"Task: {reset['goal']}\n"
        selector_map=await agent.browser_session.get_selector_map()
        query=search_query(reset['goal'],reset['url'])
        if followup and followup.get('resolve')=='search':
            box=search_box(selector_map)
            return (task+'You scrolled to the top to use the search box.',[search_option(box,query)]) if box is not None and query else None
        if followup and followup.get('resolve')=='results':
            facts=await asyncio.to_thread(browser_worker.call,'facts',goal=reset['goal'])
            write_new(root/f'results-facts-{llm.calls:04d}.json',facts)
            links=link_options(facts)
            return (task+'The search was submitted. These links on the results page match the task words.',links) if links else None
        if followup:
            return (task+'Your recovery step typed the search text; it has not been submitted yet.',[followup])
        if not (isinstance(advice,dict) and advice.get('schema')=='task2.recovery_note.v3'):return None
        facts=advice.get('facts') or {}
        # v3.2: all three goal links, as in the note (v3.1 cut IETF's 'About RFCs', ranked third).
        offered=link_options(facts)
        box=search_box(selector_map)
        if box is not None and query:
            offered.append(search_option(box,query))
        elif facts.get('search_boxes') and query:
            # Browser Use lists only elements near the viewport; the header search box may be off-screen.
            offered.append({'label':f"Scroll to the top and search this site for '{query}'",
                            'action':{'scroll':{'down':False,'pages':10.0}},'then':{'resolve':'search'}})
        field=facts.get('field') or {}
        if field.get('value'):
            offered.append({'label':f"Press Enter to submit '{field['value']}'",'action':{'send_keys':{'keys':'Enter'}}})
        if (facts.get('scroll') or {}).get('at_bottom'):
            offered.append({'label':'Scroll back to the top of the page','action':{'scroll':{'down':False,'pages':10.0}}})
        if len(agent.history.history)>1:
            offered.append({'label':'Go back to the previous page','action':{'go_back':{}}})
        allowed=[]
        for option in offered:
            proposed=recorded_action(option['action'])
            xpath=None
            if proposed and proposed['target'] and proposed['target'].isdigit():
                node=selector_map.get(int(proposed['target']))
                xpath=getattr(node,'xpath',None) if node is not None else None
            if proposed and not any(same_action(item,proposed,xpath) for item in blocked):
                allowed.append(option)
        if not allowed:return None
        return (task+advice['text'],allowed[:5])
    if v3 and settings.get('recovery_options'):
        llm.options_provider=recovery_options

    async def assess_step(step,action,element,xpath,before,after,terminal):
        nonlocal incident,incident_attempts,recovery_total,next_is_recovery,incident_diagnosis,pending
        try:
            transition=ExecutedTransition(episode_id=episode_id,task_id=reset.get('task_id','miniwob.'+block['task']),
                action_id=f'a{step}',task_description=reset['goal'],website_domain=reset.get('website_domain','miniwob.local'),
                before=observation(before),after=observation(after),**action,
                phase='recovery_assessment' if next_is_recovery else 'interaction_assessment',
                incident_id=incident if next_is_recovery else None,terminated=terminal)
            budget.charge('assessment')
            assessment=await asyncio.to_thread(model_worker.call,'assess',transition=asdict(transition),image_root=reset['image_root'])
            counters['assessments']+=1
            write_new(root/f'assessment-{step:04d}.json',{'transition':asdict(transition),'assessment':assessment})
            failed=assessment['signals']['outcome_label']=='FAILURE'
            page_changed=before['sha256']!=after['sha256']
            # Recovery trigger gate (2026-09-29): P1 learned "small page change = failure", so a
            # correct TYPE into a search box scored P(FAILURE) 0.55-0.85 in the development run and
            # the advice made the actor retype. Only a confident FAILURE opens recovery; an uncertain
            # one sends no advice and is not listed as succeeded either. The argmax label is unchanged.
            p_fail=assessment.get('outcome_probabilities',{}).get('FAILURE')
            gate=settings.get('recovery_trigger_probability')
            uncertain=failed and transition.phase=='interaction_assessment' and None not in (p_fail,gate) and p_fail<gate
            if uncertain:
                counters['uncertain_failures']=counters.get('uncertain_failures',0)+1
                write_new(root/f'gate-{step:04d}.json',{'p_failure':p_fail,'gate':gate,'recovery_triggered':False})
            if v3 and failed and not uncertain and not (action['action_type']=='SCROLL' and page_changed):
                # A scroll that moved the page is not a dead end even if judged unproductive.
                blocked.append(dict(action,xpath=xpath,url=after.get('url'),
                                    label=describe_action(action['action_type'],element,action['value'])))
                counters['blocked_actions']=counters.get('blocked_actions',0)+1
            if pending and transition.phase=='recovery_assessment':
                pending['record']['recovery_attempts'].append({'action_type':action['action_type'],'element':element,
                    'value':action['value'],'assessed_outcome':assessment['signals']['outcome_label']})
            if not failed:
                # Steps P1 judged successful are named in later advice so they are not redone.
                succeeded.append({'action_type':action['action_type'],'element':element,'value':action['value']})
                await close_incident(resolved=True)
                incident=None;incident_attempts=0;next_is_recovery=False
            elif uncertain:
                pass
            elif not terminal and recovery_total<settings['recovery_attempts_per_episode'] and (incident is None or incident_attempts<settings['recovery_attempts_per_incident']):
                new_incident=incident is None
                if new_incident:
                    incident=f'incident-{step}';incident_attempts=0;incident_diagnosis=assessment['signals']
                next_is_recovery=True
                memory=None
                if experience and budget.available(3):
                    budget.charge('memory_query')
                    memory=await asyncio.to_thread(model_worker.call,'experience',transition=asdict(transition),
                        image_root=reset['image_root'],store=settings['experience_memory'],
                        episode_id=episode_id,controls=after['controls'],url=reset['url'])
                    if new_incident:
                        signals=assessment['signals']
                        pending={'vector':memory['vector'],'store_probability':memory['store_probability'],'record':{
                            'episode_id':episode_id,'incident_id':incident,'task_goal':reset['goal'],'page_url':reset['url'],
                            'failed_action':{'action_type':action['action_type'],'element':element,'value':action['value']},
                            'failure_type':signals.get('failure_type'),'strategy':signals.get('recovery_strategy'),'recovery_attempts':[]}}
                    write_new(root/f'memory-{step:04d}.json',{k:v for k,v in memory.items() if k!='vector'})
                    counters['memory_queries']+=1
                if v3:
                    facts=await asyncio.to_thread(browser_worker.call,'facts',goal=reset['goal'])
                    llm.advice=recovery_note(transition,assessment,element=element,page_changed=page_changed,
                        facts=facts,memory=(memory or {}).get('examples',[]),
                        succeeded=[x for x in succeeded if x['action_type']!='SCROLL'][-3:])
                    llm.memory_context=[]
                    if memory and memory.get('examples'):llm.memory_exposures+=1
                    write_new(root/f'note-{step:04d}.json',llm.advice)
                else:
                    llm.advice=recovery_advice(transition,assessment,failed_element=element,
                        page_changed=page_changed,succeeded=succeeded[-5:])
                    llm.advice['incident_diagnosis']=incident_diagnosis
                    if memory is not None:
                        llm.memory_context=memory['examples']
                    elif system=='C' and not experience and budget.available(3):
                        budget.charge('memory_query')
                        applicability={'post_observation':{'causal_history':history,'current_page_state':{'visible_controls':after['controls']}},
                                       'execution_result':{'status':'executed','error_kind':None},
                                       'executed_action':{'action_type':action['action_type']}}
                        legacy=await asyncio.to_thread(model_worker.call,'memory',transition=asdict(transition),image_root=reset['image_root'],
                            applicability_transition=applicability,excluded_task_ids=settings['excluded_memory_tasks'])
                        write_new(root/f'memory-{step:04d}.json',legacy)
                        counters['memory_queries']+=1
                        llm.memory_context=legacy['examples']
            else:
                await close_incident(resolved=False)
                incident=None;incident_attempts=0;next_is_recovery=False
        except Exception as exc:
            # Browser Use would swallow this inside its step; surface it as infrastructure.
            llm.infrastructure_error=llm.infrastructure_error or f'assessment step {step}: {exc!r}'
            raise

    from browser_use.agent.views import AgentStepInfo
    try:
        for step in range(settings['max_agent_steps']):
            if not budget.available() or budget.executor_requests>=budget.max_executor_requests:break
            if agent.state.consecutive_failures>=5:
                stop='native_max_failures';break
            calls_before=llm.calls
            await agent.step(AgentStepInfo(step_number=step,max_steps=settings['max_agent_steps']))
            steps_completed+=1
            if llm.infrastructure_error:
                raise RuntimeError(llm.infrastructure_error)
            try:
                await llm.ready()
            except Exception:
                raise RuntimeError(llm.infrastructure_error or 'assessment failed')
            previous_advice=llm.advice
            previous_memory=llm.memory_context
            output=agent.state.last_model_output if llm.calls>calls_before else None
            results=agent.state.last_result or []
            score=await asyncio.to_thread(browser_worker.call,'score')
            write_new(root/f'score-{step:04d}.json',score)
            after=await asyncio.to_thread(browser_worker.call,'observe',episode_id=episode_id)
            write_new(root/f'observation-{step+1:04d}.json',after)
            event={'step':step,'actor_calls':llm.calls-calls_before,'native_output':output.model_dump(mode='json',exclude_none=True) if output else None,
                   'native_results':[r.model_dump(mode='json') for r in results],
                   'recovery_proposal':next_is_recovery,'incident_id':incident}
            native=None
            if output is not None:
                if len(output.action)!=1:raise RuntimeError('Non-atomic native output reached executor')
                native=output.action[0].model_dump(mode='json',exclude_none=True)
                proposal=json.loads((root/f'actor-{llm.calls:04d}-parsed.json').read_text())['native_proposal']
                if proposal['action']:
                    event['native_selection']={'actor_request_id':f'actor-{llm.calls:04d}',
                        'proposed_actions':len(proposal['action']),'selected_index':0,
                        'deferred_actions':len(proposal['action'])-1}
                else:
                    # Upstream retried once, then inserted done(success=False); nothing was proposed.
                    event['native_selection']={'actor_request_id':f'actor-{llm.calls:04d}',
                        'proposed_actions':0,'selected_index':None,'deferred_actions':0,'upstream_noop':True}
            action=recorded_action(native) if native else None
            element=interacted_element(agent) if action else None
            xpath=interacted_xpath(agent) if action else None
            if element:event['element']=element
            rejected=not results or any(r.error for r in results)
            if native:
                budget.executor_requests+=1
            if next_is_recovery and llm.calls>calls_before:
                incident_attempts+=1;recovery_total+=1
            if action:
                event.update(action,execution_status='rejected' if rejected else 'executed')
                counters['rejected' if rejected else 'executed']+=1
                if next_is_recovery and not rejected:counters['executed_recoveries']+=1
            write_new(root/f'action-{step:04d}.json',event)
            llm.advice=None; llm.memory_context=[]
            if v3 and page_path(before.get('url'))!=page_path(after.get('url')):
                blocked.clear()
            if v3 and action and rejected:
                # Browser Use could not execute it (e.g. a download link): a deterministic dead end.
                blocked.append(dict(action,xpath=xpath,url=after.get('url'),
                                    label=describe_action(action['action_type'],element,action['value'])+' (it could not be executed)'))
                counters['blocked_actions']=counters.get('blocked_actions',0)+1
            # Independent termination is authoritative, never self-reported success.
            terminal=score['terminated'] or score['invalid_url']
            if action and not rejected and system!='A' and budget.available():
                task=asyncio.create_task(assess_step(step,action,element,xpath,before,after,terminal))
                if v3 and not terminal and not (native and 'done' in native):
                    llm.pending=task
                else:
                    await task
            elif next_is_recovery:
                if incident_attempts>=settings['recovery_attempts_per_incident'] or recovery_total>=settings['recovery_attempts_per_episode']:
                    await close_incident(resolved=False)
                    incident=None;incident_attempts=0;next_is_recovery=False
                else:
                    # Native rejection feedback accompanies the same incident's
                    # advisory context; no new assessment of an unexecuted action.
                    llm.advice=previous_advice
                    llm.memory_context=previous_memory
            if action:
                history.append({'action_type':action['action_type'],'execution_status':'rejected' if rejected else 'executed',
                                'state_changed':before['sha256']!=after['sha256'],'environment_error':rejected})
            before=after
            if terminal:
                stop='environment_terminal';break
            if native and 'done' in native:
                stop='agent_done';break
        await llm.ready()
        await close_incident(resolved=False)
        counters['memory_exposures']=llm.memory_exposures
        if v3:counters.update(guard_blocks=llm.guard_blocks,guard_overrides=llm.guard_overrides,option_choices=llm.option_choices)
        result={'episode_id':episode_id,'system':system,'block':block,'status':'complete',
                'completion':bool(score['terminated'] and not score.get('invalid_url') and score['raw_reward']==1.0),
                'terminal_score':score,'stop_reason':stop,'counters':counters,
                'model_calls':budget.calls,'executor_requests':budget.executor_requests,
                'recovery_attempts':recovery_total,
                'elapsed_seconds':time.monotonic()-budget.started,'agent_steps':steps_completed}
        write_new(root/'result.json',result)
        return result
    except Exception as exc:
        write_new(root/'infrastructure-error.json',{'error_type':type(exc).__name__,'error':str(exc),
                   'model_calls':budget.calls,'executor_requests':budget.executor_requests})
        raise
    finally:
        if llm.pending is not None:llm.pending.cancel()
        await browser.stop()
        await asyncio.to_thread(browser_worker.call,'close')
