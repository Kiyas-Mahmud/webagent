"""Frozen MiniWoB study orchestration, using the existing PC-01 runtime."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'configs/eval/table2/miniwob_interface_v2.json'
PROMPT=ROOT/'configs/eval/table2/miniwob_recovery_prompt_v2.txt'
ENGINEERING=Path('/home/aiub/kiyas/table2-evidence/miniwob-interface-v2-engineering')
EXTRA_SOURCES=[]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sources():
    paths=list((ROOT/'src').rglob('*.py')) + [CONFIG,PROMPT,
        ROOT/'scripts/run_table2_miniwob.py',ROOT/'scripts/check_miniwob_interface_v2.py'] + EXTRA_SOURCES
    return {str(p):sha(p) for p in sorted(paths)}


def verify_assets(config):
    from web_agent.memory.label_backed_store import LabelBackedMemoryStore
    from web_agent.memory.label_experience import LabelExperienceMaterial
    assets=config['asset_paths']; export=Path(assets['export'])
    manifest=json.loads((export/'pc01_export_manifest.json').read_text())
    assert sha(assets['checkpoint'])==config['checkpoint_and_base_bindings']['checkpoint_sha256'],'checkpoint changed'
    bindings={assets['checkpoint']:sha(assets['checkpoint'])}
    for name,record in manifest['files'].items():
        assert sha(export/name)==record['sha256'],('export changed',name)
        bindings[str(export/name)]=record['sha256']
    bindings[str(export/'pc01_export_manifest.json')]=sha(export/'pc01_export_manifest.json')
    for record in json.loads((export/'base_snapshot_manifest.json').read_text())['files']:
        path=Path(assets['base_snapshot'])/record['path']
        assert sha(path)==record['sha256'],('base changed',record['path'])
        bindings[str(path)]=record['sha256']
    for path in (Path(config['full_run_directory'])/'report.json',Path(config['run_contract'])):
        bindings[str(path)]=sha(path)
    store=LabelBackedMemoryStore(assets['memory'])
    assert store.manifest_sha256==config['store_manifest_sha256']
    material=LabelExperienceMaterial(config['memory_material'],store=store,include_source_context=True)
    assert material.sha256==config['memory_material_sha256']
    for row in store._rows:
        from web_agent.memory.label_backed_store import LabelMemoryHit
        material.for_hit(LabelMemoryHit(row['memory_id'],row['source_task'],row['failure_type'],row['strategy'],1.,True))
    for name,h in json.loads((Path(assets['memory'])/'manifest.json').read_text())['files'].items():
        bindings[str(Path(assets['memory'])/name)]=h
    bindings[str(Path(assets['memory'])/'manifest.json')]=store.manifest_sha256
    bindings[config['memory_material']]=material.sha256
    for name,version in config['package_versions'].items():
        assert metadata.version(name)==version,('model package changed',name)
    code='import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in '+repr(list(config['browser_package_versions']))+'}))'
    assert json.loads(subprocess.check_output([config['browser_python'],'-c',code],text=True))==config['browser_package_versions']
    code="import importlib.util,pathlib,json,hashlib; roots=[pathlib.Path(importlib.util.find_spec(n).origin).parent for n in ['browsergym.core','browsergym.miniwob']]; print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in roots for p in root.rglob('*.py')}))"
    bindings.update(json.loads(subprocess.check_output([config['browser_python'],'-c',code],text=True)))
    mini=Path(config['miniwob_source'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=mini,text=True).strip()==config['source_commit']
    assert not subprocess.check_output(['git','diff','HEAD','--','miniwob/html/miniwob'],cwd=mini,text=True)
    if config.get('hybrid_profile'):
        from web_agent.eval.table2.hybrid_study import frozen_scope
        scope = frozen_scope(config)
        bindings[config['development_scope']] = config['development_scope_sha256']
        bindings.update({b['task_html_path']: b['task_html_sha256'] for b in scope['blocks']})
        for key, path in [('report', Path(config['full_run_directory'])/'report.json'),
                          ('run_contract', Path(config['run_contract']))]:
            assert sha(path) == config['checkpoint_and_base_bindings'][key+'_sha256']
    return bindings


def runtime_protocol(campaign_id, *, hybrid=False):
    from web_agent.runtime.protocol import RuntimeProtocol
    return RuntimeProtocol(protocol_id='table2-miniwob-hybrid-v1' if hybrid else 'table2-miniwob-label-memory-v2',campaign_id=campaign_id,
        benchmark_id='miniwob',benchmark_version='0.14.3',provider_id='deterministic_then_frozen_base_fallback',provider_version='v1',
        metadata={'seed_protocol_id':'table2-miniwob-label-memory-v2'} if hybrid else {})


def task_binding(task, block):
    """Keep the task specification distinct from the block's family name."""
    return {**block, 'task_specification': task.to_dict()}


def prepare(out,phase,development=None):
    from web_agent.runtime.protocol import RuntimeStage
    config=json.loads(CONFIG.read_text()); hashes=sources()
    if config.get('development_only') and phase!='development':
        raise ValueError('this profile authorizes development only, not final evaluation')
    bindings=verify_assets(config)
    engineering=json.loads((ENGINEERING/'engineering-checks.json').read_text())
    assert engineering['status']=='PASS'
    for path,h in engineering['tested_sources'].items(): assert sha(path)==h,('tested source changed',path)
    prerequisite={'engineering_checks':str(ENGINEERING/'engineering-checks.json'),
                  'engineering_sha256':sha(ENGINEERING/'engineering-checks.json')}
    if phase=='evaluation':
        if development is None: raise ValueError('evaluation requires the matched development package')
        development=Path(development)
        audit=json.loads((development/'audit.json').read_text())
        assert audit['status']=='PASS' and audit['live_path_verified'], 'development path not verified'
        assert json.loads((development/'plan.json').read_text())['source_sha256']==hashes,'development source differs'
        assert json.loads((development/'analysis.json').read_text())['all_included_episodes_present']
        prerequisite.update(development=str(development),development_audit_sha256=sha(development/'audit.json'))
    tasks=config['development_tasks'] if phase=='development' else config['evaluation_tasks']
    repeats=1 if phase=='development' else 5
    campaign=config['development_campaign_id'] if phase=='development' else config['final_campaign_id']
    protocol=runtime_protocol(campaign, hybrid=config.get('hybrid_profile',False)); rng=protocol.rng_factory(); blocks=[]
    for task in tasks:
        for repeat in range(repeats):
            seed=rng.seed_for_key(rng.key(task_id='miniwob.'+task,repeat_id=repeat,matched_seed=42,stage=RuntimeStage.RESET,decision_index=0))
            blocks.append({'task':task,'repeat_id':repeat,'stage_reset_seed':seed,'browser_reset_seed':seed%(2**32)})
    if config.get('hybrid_profile'):
        from web_agent.eval.table2.hybrid_study import frozen_scope
        scope = frozen_scope(config)
        assert blocks == [{k:b[k] for k in blocks[0]} for b in scope['blocks']]
        blocks = scope['blocks']
    assert len({(b['task'],b['browser_reset_seed']) for b in blocks})==len(blocks)
    plan={'profile':config.get('profile','miniwob-interface-v2'),'phase':phase,'config':config,'blocks':blocks,
          'recovery_prompt_path':str(PROMPT),
          'systems':config['systems'],'episodes_planned':len(blocks)*4,'runtime_protocol':protocol.to_dict(),
          'source_sha256':hashes,'asset_sha256':bindings,'prerequisite':prerequisite,
          'scoring':'terminated and not truncated and raw_reward == 1.0','positive_result_required':False,
          'scope':f'Prespecified resets within {len(tasks)} previously observed families; seed-42 checkpoint; train-label memory',
          'analysis':{'unit':'task-reset pair','contrasts':config.get('primary_contrasts',[['E1','E2'],['E2','E3']]),
                      'secondary_contrasts':config.get('secondary_contrasts',[]),
                      'paired_test':'exact two-sided discordant-pair binomial','multiplicity':'Holm, two contrasts, alpha .05',
                      'interval':'paired bootstrap within fixed task families','confidence':.95,'samples':10000,'seed':20250831},
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan); (out/'plan.sha256').write_text(sha(out/'plan.json')+'\n')
    # Keep the exact producing sources so later repository changes never rewrite history.
    for path in hashes:
        source=Path(path); target=out/'source-snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
    for p in (out/'source-snapshot').rglob('*'):
        if p.is_file(): p.chmod(0o444)
    (out/'plan.json').chmod(0o444);(out/'plan.sha256').chmod(0o444)
    print(json.dumps({'status':'FROZEN','phase':phase,'episodes':plan['episodes_planned'],'output':str(out)}))


def verify_plan(out, *, audit_revision=False):
    assert sha(out/'plan.json')==(out/'plan.sha256').read_text().strip(),'frozen plan changed'
    plan=json.loads((out/'plan.json').read_text())
    prerequisite=plan['prerequisite']
    assert sha(prerequisite['engineering_checks'])==prerequisite['engineering_sha256'],'engineering receipt changed'
    if 'development' in prerequisite:
        assert sha(Path(prerequisite['development'])/'audit.json')==prerequisite['development_audit_sha256'],'development receipt changed'
    # Audit-only revisions replay the producing source snapshot. Execution
    # still requires current source identity and never uses this exception.
    for path,h in {**plan['source_sha256'],**plan['asset_sha256']}.items():
        checked=Path(path)
        if audit_revision and path in plan['source_sha256']:
            checked=out/'source-snapshot'/checked.relative_to(ROOT)
        assert sha(checked)==h,('frozen file changed',path)
    assert verify_assets(plan['config'])==plan['asset_sha256']
    assert runtime_protocol(plan['runtime_protocol']['campaign_id'], hybrid=plan['config'].get('hybrid_profile',False)).to_dict()==plan['runtime_protocol']
    return plan


def run(out):
    plan=verify_plan(out); config=plan['config']
    revision=config.get('development_revision',{})
    if config.get('development_only') and plan['phase']!='development':
        raise ValueError('this profile authorizes development only, not final evaluation')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()
    active=[r for r in active if r.strip() and r.partition(',')[2].strip()!='/usr/libexec/gnome-remote-desktop-daemon']
    if active: raise RuntimeError('Other GPU work is active; preserved: '+repr(active))
    if (out/'completion.json').exists(): raise ValueError('completed run cannot be relaunched')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(ROOT/'src'),
        MINIWOB_URL=(Path(config['miniwob_source'])/'miniwob/html/miniwob').as_uri()+'/',
        PLAYWRIGHT_BROWSERS_PATH=config['browser_binaries'])
    from web_agent.eval.table2.miniwob_model import load_models
    from web_agent.benchmarks.browsergym_miniwob import MiniWoBAdapter
    from web_agent.benchmarks.miniwob_support import full_completion
    from web_agent.runtime.contracts import TaskSpecification,SystemID,HybridSystemID
    from web_agent.runtime.protocol import switches_for
    from web_agent.runtime.policy import CallablePolicyAdapter,PolicyKind,SystemPolicy
    from web_agent.runtime.episode import EpisodeRunner
    from web_agent.runtime.executor import Executor
    from web_agent.runtime.observation import ObservationBuilder
    from web_agent.runtime.event_log import EpisodeEventLogs
    from web_agent.runtime.named_target_policy import NamedTargetRuntime
    from web_agent.runtime.causal_recovery_planner import CausalRecoveryRuntime
    from web_agent.runtime.recovery.controller import RecoveryController,CallableRecoveryActionPlanner
    from web_agent.runtime.action_parameters import concretize
    from web_agent.memory.label_backed_store import LabelBackedMemoryStore
    from web_agent.memory.label_experience import LabelExperienceMaterial
    from web_agent.memory.label_runtime import LabelBackedMemoryAdapter
    from web_agent.memory.miniwob_overlap import MiniWoBOverlapAudit
    bundle,provider,manifest=load_models(config)
    if revision.get('executable_action_selection'):
        # Declared before any episode: the trained policy selects among the
        # action classes this page can execute, from the same registered
        # evidence the generated interfaces already receive. The learned
        # distribution itself is unchanged and still fully logged.
        bundle.selected.action_predictor.__self__.restrict_to_executable_actions=True
    base_runtime=bundle.e0.action_predictor.__self__
    original_generate=base_runtime._generate
    original_generate_with_metadata=getattr(base_runtime,'_generate_with_metadata',None)
    raw_folder=None
    raw_count=0
    def save_generation(raw,kwargs,metadata=None):
        nonlocal raw_count
        raw_count+=1
        record={
            'task_id':kwargs['task'].task_id,'observation_id':kwargs['observation'].observation_id,
            'screenshot_sha256':kwargs['observation'].screenshot_sha256,
            'prompt_sha256':hashlib.sha256(kwargs['prompt'].encode()).hexdigest(),
            'input_suffix':kwargs['suffix'],'raw_response':raw}
        if metadata is not None: record['generation_metadata']=metadata
        write_json(raw_folder/f'raw-parameter-{raw_count:04d}.json',record)
    def capture_parameters(**kwargs):
        raw=original_generate(**kwargs)
        save_generation(raw,kwargs)
        return raw
    base_runtime._generate=capture_parameters
    if config.get('hybrid_interface_version') == 3 and original_generate_with_metadata is not None:
        def capture_metadata(**kwargs):
            raw,metadata=original_generate_with_metadata(**kwargs)
            save_generation(raw,kwargs,metadata)
            return raw,metadata
        base_runtime._generate_with_metadata=capture_metadata
    store=LabelBackedMemoryStore(config['asset_paths']['memory']) if not config.get('hybrid_profile') else None
    material=LabelExperienceMaterial(config['memory_material'],store=store,include_source_context=True) if store else None
    training_items=[json.loads(l) for l in Path(config['memory_material']).read_text().splitlines()]
    protocol=runtime_protocol(plan['runtime_protocol']['campaign_id'], hybrid=config.get('hybrid_profile',False))
    rows=[]
    for block in plan['blocks']:
        task_name=block['task']; repeat=block['repeat_id']; block_root=out/task_name/f'repeat-{repeat}'
        if (block_root/'.started').exists():
            # No incomplete block is ever replayed or partially repaired.
            for path in sorted(block_root.glob('*/result.json')): rows.append(json.loads(path.read_text()))
            if (block_root/'exclusion.json').exists(): rows.append(json.loads((block_root/'exclusion.json').read_text()))
            continue
        block_root.mkdir(parents=True,exist_ok=False); (block_root/'.started').touch(exist_ok=False)
        if config.get('hybrid_profile'):
            task=TaskSpecification.from_dict(block['task_specification'])
        else:
            probe=MiniWoBAdapter(block_root/'goal-probe',browser_python=config['browser_python'])
            try: goal=probe.rpc('reset',seed=block['stage_reset_seed'],task=task_name)['goal']
            finally: probe.close()
            task=TaskSpecification(task_id='miniwob.'+task_name,goal=goal,benchmark_id='miniwob',benchmark_version='0.14.3',start_state_id='seeded-reset',development_partition=plan['phase']=='development')
        write_json(block_root/'task-binding.json',task_binding(task,block))
        overlap=MiniWoBOverlapAudit(task,training_items)
        write_json(block_root/'overlap-audit.json',overlap.record)
        if overlap.record['matching_source_tasks']:
            row={'task':task_name,'repeat_id':repeat,'status':'EXCLUDED_TRAIN_OVERLAP'}
            write_json(block_root/'exclusion.json',row); rows.append(row); continue
        system_type = HybridSystemID if config.get('hybrid_profile') else SystemID
        for system in map(system_type,plan['systems']):
            folder=block_root/system.value
            raw_folder=folder
            raw_count=0
            adapter=MiniWoBAdapter(folder,browser_python=config['browser_python'],
                hybrid_interface_version=config.get('hybrid_interface_version',1))
            eid=f'{protocol.campaign_id}:{system.value}:{task.task_id}:repeat-{repeat}:seed-42'
            selected=system.value not in {'E0','H0'}; backend=bundle.selected if selected else bundle.e0
            if config.get('hybrid_profile'):
                from web_agent.eval.table2.hybrid_study import build_hybrid_components
                policy,memory=build_hybrid_components(system=system,episode_id=eid,folder=folder,
                    bundle=bundle,manifest=manifest,config=config)
            else:
                action_predictor=backend.action_predictor if selected else NamedTargetRuntime(base=backend.action_predictor.__self__,
                    evidence_dir=folder/'named-e0-outputs',allow_json_fence=True,allow_role_target=True,include_control_context=True).predict_action
                policy=CallablePolicyAdapter(policy_id='validation-selected-web-agent' if selected else 'selected-backbone-unadapted',
                    policy_version='v1',kind=PolicyKind.TRAINED if selected else PolicyKind.BASE,
                    checkpoint_sha256=manifest['checkpoint_sha256'] if selected else None,action_predictor=action_predictor,
                    transition_predictor=bundle.selected.transition_predictor if selected else None,
                    recovery_predictor=bundle.selected.recovery_predictor if selected else None)
                memory=LabelBackedMemoryAdapter(store=store,runtime=bundle.selected.action_predictor.__self__,transitions={},recovery_context_material=material,
                    selective_context=revision.get('selective_memory_context',False)) if system is SystemID.E3 else None
            controller=None
            if switches_for(system).recovery_controller:
                causal=CausalRecoveryRuntime(base=bundle.e0.action_predictor.__self__,evidence_dir=folder/'named-recovery-outputs',
                    hybrid_interface_version=config.get('hybrid_interface_version',1) if config.get('hybrid_profile') else 1,
                    action_prompt=Path(plan.get('recovery_prompt_path',PROMPT)).read_text(),allow_json_fence=True,allow_role_target=True,use_step_request_feedback=True,neutral_instruction=True)
                def propose(task,observation,decision,failed_action,rng,*,planner_context=None):
                    proposed=causal.predict_recovery(task,observation,decision,failed_action,rng,planner_context=planner_context)
                    try: parameters=provider.resolve(task,observation,proposed,rng=rng)
                    except ValueError as exc:
                        exc.failure_stage='parameter_resolution'
                        exc.feedback_detail='; '.join(a.source+': '+a.reason for a in getattr(exc,'attempts',()) if a.reason) or str(exc)
                        raise
                    return concretize(action_id=decision.decision_id+':proposal',decision=proposed,parameters=parameters)
                controller=RecoveryController(protocol.budgets,
                    stable_target_identity=revision.get('stable_retry_targets',False),
                    action_planner=CallableRecoveryActionPlanner(
                    planner_id='frozen-qwen-control-interface-v3' if revision else 'frozen-qwen-control-interface-v2',
                    planner_version='v3' if revision else 'v2',callback=propose,accepts_context=True))
            logs=EpisodeEventLogs(folder/'runtime',episode_id=eid,include_memory=switches_for(system).memory_query)
            runner=EpisodeRunner(protocol=protocol,system_policy=SystemPolicy(policy,switches_for(system)),provider=provider,
                executor=Executor(adapter,budgets=protocol.budgets),recovery_controller=controller,memory_adapter=memory,
                duplicate_audit_registry=overlap if memory else None,observation_builder=ObservationBuilder(processor_contract=backend.processor_contract),
                training_processor_contract=backend.processor_contract,event_logs=logs,
                observable_progress_continuation=revision.get('observable_progress_continuation',False),
                hybrid_continuation=revision.get('hybrid_continuation',False),
                hybrid_proposal_repair=revision.get('hybrid_proposal_repair',False),
                stable_retry_targets=revision.get('stable_retry_targets',False))
            row={'task':task_name,'repeat_id':repeat,'system':system.value}
            if config.get('hybrid_profile'):row['profile']=plan['profile']
            try: row['summary']=runner.run(task,repeat_id=repeat,model_seed=42).to_dict()
            except Exception as exc: row.update(error=str(exc),traceback=traceback.format_exc())
            finally:
                try: adapter.close()
                except Exception as exc: row.update(error='browser cleanup failed: '+str(exc),traceback=traceback.format_exc())
            outcomes=json.loads((folder/'browser-outcomes.json').read_text()) if (folder/'browser-outcomes.json').exists() else []
            row.update(environment_outcomes=outcomes,success=full_completion(outcomes))
            write_json(folder/'result.json',row); rows.append(row)
            write_json(out/'results.json',rows)
            write_json(out/'progress.json',{'completed_episodes':sum('system' in r for r in rows),'planned_episodes':plan['episodes_planned'],'latest':row})
            print(json.dumps({'completed':sum('system' in r for r in rows),'task':task_name,'repeat':repeat,'system':system.value,'success':row['success'],'error':row.get('error')}),flush=True)
    write_json(out/'results.json',rows)
    verify_plan(out)
    from web_agent.eval.table2.miniwob_reporting import audit_and_report
    result=audit_and_report(out)
    write_json(out/'completion.json',{'status':'COMPLETE' if result['status']=='PASS' else 'AUDIT_FAILED','episodes':len([r for r in rows if 'system' in r]),'audit_status':result['status']})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['prepare','run','audit'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['development','evaluation'],default='development')
    parser.add_argument('--development',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if args.operation=='prepare': prepare(out,args.phase,args.development)
    elif args.operation=='run':
        import fcntl
        with (out/'.coordinator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(out)
    else:
        from web_agent.eval.table2.miniwob_reporting import audit_and_report
        audit_and_report(out)


if __name__=='__main__': main()
