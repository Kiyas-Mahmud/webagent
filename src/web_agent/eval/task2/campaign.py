"""Versioned matched Task 2 campaigns. No training or automatic episode retries."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

from web_agent.eval.task1.core import file_hash, write_new
from web_agent.eval.task2.ipc import Worker

ROOT=Path(__file__).resolve().parents[4]
BROWSER_PYTHON='/home/aiub/kiyas/table2-envs/miniwob-feasibility/bin/python'
SETTINGS={'budget':{'max_model_calls':102,'max_executor_requests':30,'seconds':600},
          'max_agent_steps':30,'excluded_memory_tasks':[],
          'recovery_attempts_per_incident':2,'recovery_attempts_per_episode':4,
          'seed':42,'actor_max_new_tokens':512,'actor_do_sample':False,
          'memory_encoder_authorized':True,'native_history_items':6,
          'score':'terminated, not invalid/truncated, raw reward exactly 1.0'}


def browser_worker(root):
    return Worker(BROWSER_PYTHON,'web_agent.eval.task2.browser_worker',[root/'browser'],root/'browser.log',
                  env={'PLAYWRIGHT_BROWSERS_PATH':'/home/aiub/kiyas/table2-inputs/miniwob-browsers'})


def update_status(root, **value):
    temp=root/'status.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(root/'status.json')


def source_paths():
    return sorted(list((ROOT/'src').rglob('*.py')) + list((ROOT/'configs/eval/task2').glob('*.json'))
                  + list((ROOT/'tests/task2').glob('*.py')) + list((ROOT/'scripts/external_agents').glob('task2*.py')))


def freeze(root, phase, development=None):
    from web_agent.eval.table2.miniwob_study import runtime_protocol
    from web_agent.runtime.protocol import RuntimeStage
    from web_agent.memory.miniwob_overlap import MiniWoBOverlapAudit
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    sources={str(p):file_hash(p) for p in source_paths()}
    regression=ROOT/'.task2-assets/task2-regression-v4.json'
    receipt=json.loads(regression.read_text())
    if receipt['status']!='PASS':raise ValueError('Regression PASS required')
    for path,digest in receipt['sources'].items():
        if sources.get(path)!=digest:raise ValueError('Untested producing source: '+path)
    if phase=='evaluation':
        dev=Path(development)
        audit=json.loads((dev/'audit.json').read_text())
        if audit['status']!='PASS':raise ValueError('Development audit required')
        if json.loads((dev/'plan.json').read_text())['sources']!=sources:
            raise ValueError('Implementation changed since development')
    engineer=Path('.task2-assets/engineering-six-actions-v2/result.json')
    if json.loads(engineer.read_text())['status']!='PASS':raise ValueError('Native executor engineering check required')
    parity=Path('.task2-assets/engineering-model-v1/result.json')
    if json.loads(parity.read_text())['status']!='PASS':raise ValueError('Head and query parity check required')
    families=['click-button','enter-text','click-test','click-button-sequence'] if phase=='development' else [
        'click-link','click-option','click-checkboxes','enter-password','login-user','focus-text']
    repeats=1 if phase=='development' else 5
    rng=runtime_protocol('task2-native-'+phase+'-v1').rng_factory()
    training_path=Path('/home/aiub/kiyas/table2-evidence/p4-local-label-memory-v1/memory-items.jsonl')
    training=[json.loads(l) for l in training_path.read_text().splitlines() if l.strip()]
    blocks=[];worker=browser_worker(root)
    try:
        for task in families:
            for repeat in range(repeats):
                seed=rng.seed_for_key(rng.key(task_id='miniwob.'+task,repeat_id=repeat,matched_seed=42,stage=RuntimeStage.RESET,decision_index=0))
                observed=worker.call('reset',episode_id=f'probe-{task}-{repeat}',task=task,seed=seed%(2**32))
                overlap=MiniWoBOverlapAudit(SimpleNamespace(goal=observed['goal'],task_id='miniwob.'+task),training)
                blocks.append({'task':task,'repeat_id':repeat,'stage_reset_seed':seed,
                    'browser_reset_seed':seed%(2**32),'goal':observed['goal'],
                    'eligible':not overlap.record['matching_source_tasks'],'overlap':overlap.record})
    finally:worker.close()
    for path in source_paths():
        target=root/'producing-source'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    # Reuse the already verified immutable Task 1 model/base identities and separate PC01 memory bindings.
    assets=ROOT/'.task1-assets/runs/task1-internvl-dual-v2/freeze.json'
    bindings={str(assets):file_hash(assets),str(training_path):file_hash(training_path),str(engineer.resolve()):file_hash(engineer),
              str(parity.resolve()):file_hash(parity),str(regression):file_hash(regression)}
    identities=json.loads(assets.read_text())['assets']
    model_paths=json.loads((ROOT/'configs/eval/task1/internvl_dual_lab_paths.json').read_text())
    bindings[model_paths['checkpoint']]=identities['checkpoint_sha256']
    for name,digest in identities['internvl_base'].items():
        bindings[str(Path(model_paths['internvl_base'])/name)]=digest
    memory_root=Path('/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1')
    memory_manifest=json.loads((memory_root/'manifest.json').read_text())
    bindings[str(memory_root/'manifest.json')]=file_hash(memory_root/'manifest.json')
    for name,digest in memory_manifest['files'].items():bindings[str(memory_root/name)]=digest
    old=json.loads((ROOT/'configs/eval/table2/miniwob_hybrid_dev_v3.json').read_text())
    bindings[old['asset_paths']['checkpoint']]=memory_manifest['checkpoint_sha256']
    for item in json.loads((Path(old['asset_paths']['export'])/'base_snapshot_manifest.json').read_text())['files']:
        bindings[str(Path(old['asset_paths']['base_snapshot'])/item['path'])]=item['sha256']
    for path in Path(old['asset_paths']['export']).glob('*.json'):bindings[str(path)]=file_hash(path)
    native=ROOT/'.task2-assets/browser-use-env/lib/python3.12/site-packages/browser_use'
    for path in sorted(native.rglob('*')):
        if path.is_file() and path.suffix in ('.py','.md','.js'):bindings[str(path)]=file_hash(path)
    for path,digest in bindings.items():
        if file_hash(path)!=digest:raise ValueError('Asset identity changed: '+path)
    for p in (ROOT/'docs/evidence/task2-start-v1').glob('*.json'):bindings[str(p)]=file_hash(p)
    mini=Path('/home/aiub/kiyas/table2-inputs/miniwob-plusplus/miniwob/html')
    for p in sorted(mini.rglob('*')):
        if p.is_file():bindings[str(p)]=file_hash(p)
    plan={'schema':'task2.native-campaign.v3','phase':phase,'systems':['A','B','C'],
          'settings':SETTINGS,'blocks':blocks,'sources':sources,'bindings':bindings,
          'actor_backend':'InternVL3.5-8B-HF frozen base','assessor_checkpoint':'InternVL full epoch0 seed42',
          'memory_query':'Original PC01 encoder only, user-authorized; frozen 1974-vector store',
          'comparison':'Package B-A; memory package C-B; no native-agent superiority or isolated-head causal claim',
          'analysis':{'primary_contrasts':[['A','B'],['B','C']],'paired_test':'exact two-sided discordant binomial',
                      'multiplicity':'Holm over two contrasts','resamples':10000,'seed':20250831,'unit':'task-reset pair, resample within fixed family'},
          'development':str(Path(development).resolve()) if development else None}
    write_new(root/'plan.json',plan)
    update_status(root,stage='frozen',saved=0,expected=sum(b['eligible'] for b in blocks)*3,model_calls=0)
    return plan


async def run(root):
    from web_agent.eval.task2.live import episode
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    for group in ('sources','bindings'):
        for path,digest in plan[group].items():
            if file_hash(path)!=digest:raise ValueError('Frozen file changed: '+path)
    completed=[]
    for block in plan['blocks']:
        if not block['eligible']:continue
        for system in plan['systems']:
            folder=root/'episodes'/block['task']/str(block['repeat_id'])/system
            if folder.exists():
                if not (folder/'result.json').exists():raise RuntimeError('Preserved unfinished episode requires review: '+str(folder))
                completed.append(json.loads((folder/'result.json').read_text()))
    expected=sum(b['eligible'] for b in plan['blocks'])*3
    model=Worker(ROOT/'.venv/bin/python','web_agent.eval.task2.model_worker',[],root/f'model-{len(completed):04d}.log')
    browser=Worker(BROWSER_PYTHON,'web_agent.eval.task2.browser_worker',[root/'live-browser'],root/f'live-browser-{len(completed):04d}.log',
                  env={'PLAYWRIGHT_BROWSERS_PATH':'/home/aiub/kiyas/table2-inputs/miniwob-browsers'})
    try:
        update_status(root,stage='loading_models',saved=len(completed),expected=expected)
        initialization=await asyncio.to_thread(model.call,'initialize',timeout=600)
        write_new(root/f'initialization-{len(completed):04d}.json',initialization)
        for block in plan['blocks']:
            if not block['eligible']:continue
            for system in plan['systems']:
                folder=root/'episodes'/block['task']/str(block['repeat_id'])/system
                if (folder/'result.json').exists():continue
                update_status(root,stage='running',saved=len(completed),expected=expected,task=block['task'],repeat=block['repeat_id'],system=system)
                result=await episode(system=system,block=block,root=folder,browser_worker=browser,model_worker=model,settings=plan['settings'])
                completed.append(result)
                print(json.dumps({'saved':len(completed),'expected':expected,'episode':result['episode_id'],'completion':result['completion']}),flush=True)
        update_status(root,stage='execution_complete_audit_pending',saved=len(completed),expected=expected)
    except BaseException as exc:
        update_status(root,stage='review_required',saved=len(completed),expected=expected,error=str(exc))
        raise
    finally:browser.close();model.close()


def main():
    os.environ['ANONYMIZED_TELEMETRY']='false'
    os.environ['BROWSER_USE_CONFIG_DIR']=str(ROOT/'.task2-assets/browser-use-config')
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run'])
    parser.add_argument('--out',required=True);parser.add_argument('--phase',choices=['development','evaluation'],default='development')
    parser.add_argument('--development')
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.out,args.phase,args.development)
    else:asyncio.run(run(args.out))


if __name__=='__main__':main()
