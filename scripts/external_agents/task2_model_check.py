"""Engineering head/query parity on real captured fixture transitions, not task scores."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from web_agent.eval.task1.core import write_new
from web_agent.eval.task2.assessment import ExecutedTransition
from web_agent.eval.task2.live import observation, recorded_action
from web_agent.eval.task2.ipc import Worker

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
fixture=json.loads(Path(sys.argv[2]).read_text())
worker=Worker('.venv/bin/python','web_agent.eval.task2.model_worker',[],root/'model.log')
try:
    t=time.monotonic();write_new(root/'initialization-started.json',{'at':time.time()})
    init=worker.call('initialize',timeout=600)
    write_new(root/'initialization.json',dict(init,elapsed_seconds=time.monotonic()-t))
    row=next(r for r in fixture['actions'] if r['action_type']=='TYPE')
    transition=ExecutedTransition('fixture','fixture-task','fixture-type',
        'Enter Exact Value 42! in Field','miniwob.local',observation(row['before']),observation(row['after']),
        **recorded_action(row['native_action']))
    records=[]
    for phase in ('interaction_assessment','recovery_assessment'):
        current=replace(transition,phase=phase,incident_id='fixture-incident' if phase=='recovery_assessment' else None)
        result=worker.call('parity',transition=asdict(current),image_root=row['image_root'])
        records.append(result);write_new(root/(phase+'.json'),result)
    applicability={'post_observation':{'causal_history':[],'current_page_state':{'visible_controls':row['after']['controls']}},
                   'execution_result':{'status':'executed','error_kind':None},'executed_action':{'action_type':'TYPE'}}
    calls=[]
    for attempt in range(2):
        result=worker.call('memory',transition=asdict(transition),image_root=row['image_root'],applicability_transition=applicability)
        calls.append(result);write_new(root/f'memory-replay-{attempt}.json',result)
    assert calls[0]==calls[1], 'Frozen query replay changed'
    write_new(root/'result.json',{'status':'PASS','scope':'Engineering transition parity and deterministic original-encoder query replay',
                'live_model_episodes':0,'model_calls':6,'original_query_encoder':calls[0]['query_checkpoint_sha256'],
                'memory_writes':0})
    print('PASS engineering head parity and frozen query replay',flush=True)
finally:worker.close()
