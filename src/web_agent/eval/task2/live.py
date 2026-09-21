"""Native Browser Use execution on BrowserGym-owned MiniWoB tasks."""
from dataclasses import asdict, dataclass, field
import asyncio
import json
import os
from pathlib import Path
import time

from web_agent.eval.task1.core import write_new
from web_agent.eval.task2.assessment import Observation, ExecutedTransition, recovery_advice
from web_agent.eval.task2.native_model import LocalInternVL

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
                           allowed_domains=['127.0.0.1','localhost'])
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
    llm=LocalInternVL(model_worker,root,budget)
    agent,browser=await create_agent(reset,llm,root)
    before=await asyncio.to_thread(browser_worker.call,'observe',episode_id=episode_id)
    write_new(root/'observation-0000.json',before)
    incident=None; incident_attempts=0; recovery_total=0; next_is_recovery=False; incident_diagnosis=None
    counters={'executed':0,'rejected':0,'executed_recoveries':0,'assessments':0,'memory_queries':0,'memory_exposures':0}
    history=[]; stop='budget'; score={'terminated':False,'raw_reward':0.0}; step=0; steps_completed=0
    from browser_use.agent.views import AgentStepInfo
    try:
        for step in range(settings['max_agent_steps']):
            if not budget.available() or budget.executor_requests>=budget.max_executor_requests:break
            if agent.state.consecutive_failures>=5:
                stop='native_max_failures';break
            calls_before=llm.calls
            previous_advice=llm.advice
            previous_memory=llm.memory_context
            await agent.step(AgentStepInfo(step_number=step,max_steps=settings['max_agent_steps']))
            steps_completed+=1
            if llm.infrastructure_error:
                raise RuntimeError(llm.infrastructure_error)
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
                event['native_selection']={'actor_request_id':f'actor-{llm.calls:04d}',
                    'proposed_actions':len(proposal['action']),'selected_index':0,
                    'deferred_actions':len(proposal['action'])-1}
            action=recorded_action(native) if native else None
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
            # Independent termination is authoritative, never self-reported success.
            terminal=score['terminated'] or score['invalid_url']
            if action and not rejected and system!='A' and budget.available():
                transition=ExecutedTransition(episode_id=episode_id,task_id='miniwob.'+block['task'],
                    action_id=f'a{step}',task_description=reset['goal'],website_domain='miniwob.local',
                    before=observation(before),after=observation(after),**action,
                    phase='recovery_assessment' if next_is_recovery else 'interaction_assessment',
                    incident_id=incident if next_is_recovery else None,terminated=terminal)
                budget.charge('assessment')
                assessment=await asyncio.to_thread(model_worker.call,'assess',transition=asdict(transition),image_root=reset['image_root'])
                counters['assessments']+=1
                write_new(root/f'assessment-{step:04d}.json',{'transition':asdict(transition),'assessment':assessment})
                failed=assessment['signals']['outcome_label']=='FAILURE'
                if not failed:
                    incident=None;incident_attempts=0;next_is_recovery=False
                elif not terminal and recovery_total<settings['recovery_attempts_per_episode'] and (incident is None or incident_attempts<settings['recovery_attempts_per_incident']):
                    if incident is None:
                        incident=f'incident-{step}';incident_attempts=0;incident_diagnosis=assessment['signals']
                    next_is_recovery=True
                    llm.advice=recovery_advice(transition,assessment)
                    llm.advice['incident_diagnosis']=incident_diagnosis
                    if system=='C' and budget.available(3):
                        budget.charge('memory_query')
                        applicability={'post_observation':{'causal_history':history,'current_page_state':{'visible_controls':after['controls']}},
                                       'execution_result':{'status':'executed','error_kind':None},
                                       'executed_action':{'action_type':action['action_type']}}
                        memory=await asyncio.to_thread(model_worker.call,'memory',transition=asdict(transition),image_root=reset['image_root'],
                            applicability_transition=applicability,excluded_task_ids=settings['excluded_memory_tasks'])
                        write_new(root/f'memory-{step:04d}.json',memory)
                        counters['memory_queries']+=1
                        llm.memory_context=memory['examples']
                else:
                    incident=None;incident_attempts=0;next_is_recovery=False
            elif next_is_recovery:
                if incident_attempts>=settings['recovery_attempts_per_incident'] or recovery_total>=settings['recovery_attempts_per_episode']:
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
        counters['memory_exposures']=llm.memory_exposures
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
        await browser.stop()
        await asyncio.to_thread(browser_worker.call,'close')
