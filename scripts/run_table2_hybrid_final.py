"""Bind the approved H final protocol to the unchanged episode loop and auditor.

prepare performs reset-only goal/overlap probes; run is a separate operation.
The final verifier is installed only during this entrypoint's execution and
delegates historical plans to their original verifier. No old gate is relaxed.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil

from web_agent.eval.table2 import miniwob_study as study

ROOT = study.ROOT
EVIDENCE = ROOT.parent / 'table2-evidence'
PROTOCOL = ROOT / 'configs/eval/table2/miniwob_hybrid_final_protocol_v1.json'
PROTOCOL_SHA = '0daff965fb94b3766998aa85605ae69d91eacc1d1e71270f12a1ac5310c97b5f'
PROFILE = 'miniwob-hybrid-final-selective-memory-v1'
ORIGINAL_VERIFY = study.verify_plan
SCRIPT = Path(__file__).resolve()
TEST = ROOT / 'tests/table2/test_hybrid_final.py'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def protocol():
    require(study.sha(PROTOCOL) == PROTOCOL_SHA, 'approved protocol changed')
    return read(PROTOCOL)


def accept_development(audit, analysis, readiness):
    """No success/exposure threshold; genuine missing evidence still fails."""
    require(audit['status'] == 'PASS' and not audit['errors'], 'development audit failed')
    require(audit['source_and_assets_verified'], 'development identities not verified')
    require(analysis['all_included_episodes_present'] and
            not analysis['excluded_invalid_blocks'] and not analysis['runtime_errors'],
            'development has missing or invalid episodes')
    require(readiness['pre_memory_generation_context_isolation'] == 'PASS', 'isolation failed')
    require(readiness['bounded_click_continuation_exercised'], 'live continuation missing')
    require(readiness['memory_writes'] == readiness['stored_embeddings_regenerated'] == 0,
            'memory mutated')
    totals = audit['totals']
    require(totals.get('executed_recovery', 0) > 0 and
            totals.get('post_recovery_assessments', 0) == totals['executed_recovery'],
            'live recovery/assessment missing')
    require(totals.get('memory_queries', 0) > 0, 'live retrieval evidence missing')
    for key in ('memory_queries', 'memory_abstentions', 'memory_generation_exposures'):
        require(totals.get(key, 0) == readiness[key], 'memory accounting differs: ' + key)
    require(0 <= totals.get('memory_abstentions', 0) <= totals['memory_queries'],
            'invalid abstention accounting')
    # PASS comes from the independent raw auditor: it replays retrieval/filtering,
    # context delivery, H2 shadow equality and causal isolation for each query.
    return {'status': 'PASS', 'verified_abstention_accepted': True,
            'memory_generation_exposures': totals.get('memory_generation_exposures', 0),
            'full_memory_live_path_verified': audit['live_path_verified'],
            'positive_effect_required': False}


def candidate():
    p = protocol()
    source = Path(p['candidate_development_plan']).parent
    require(study.sha(source/'plan.json') == p['candidate_development_plan_sha256'],
            'candidate plan changed')
    old = ORIGINAL_VERIFY(source)
    require(old['source_sha256'] == p['candidate_source_sha256'] and
            old['asset_sha256'] == p['candidate_asset_sha256'], 'candidate bindings differ')
    return p, source, old


def final_config(p, old):
    config = deepcopy(old['config'])
    config.update(profile=PROFILE, development_only=False, development_tasks=[],
                  evaluation_tasks=list(dict.fromkeys(b['task'] for b in p['blocks'])),
                  final_campaign_id=p['seed_namespace'])
    return config


def sources(p):
    return {**p['candidate_source_sha256'], str(SCRIPT): study.sha(SCRIPT),
            str(TEST): study.sha(TEST), str(PROTOCOL): study.sha(PROTOCOL)}


def check_engineering(path, p):
    receipt = read(path)
    require(receipt['status'] == 'PASS' and receipt['actual_model_inferences'] == 0,
            'final engineering checks missing')
    require(receipt['tested_sources'] == sources(p), 'tested final sources differ')
    for name, digest in receipt['evidence_sha256'].items():
        require(study.sha(name) == digest, 'engineering evidence changed: ' + name)
    audit = read(receipt['development_audit'])
    analysis = read(receipt['development_analysis'])
    readiness = read(receipt['historical_readiness'])
    require(receipt['acceptance'] == accept_development(audit, analysis, readiness),
            'amended acceptance receipt differs')
    return receipt


def browser_environment(config):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONPATH=str(ROOT/'src'), PLAYWRIGHT_BROWSERS_PATH=config['browser_binaries'],
        MINIWOB_URL=(Path(config['miniwob_source'])/'miniwob/html/miniwob').as_uri()+'/')


def prepare(out, engineering):
    from web_agent.benchmarks.browsergym_miniwob import MiniWoBAdapter
    from web_agent.memory.miniwob_overlap import MiniWoBOverlapAudit
    from web_agent.runtime.contracts import TaskSpecification
    p, source, old = candidate()
    check_engineering(engineering, p)
    config = final_config(p, old)
    out.mkdir(parents=True, exist_ok=False)
    study.write_json(out/'preparation-started.json', {'protocol_sha256': PROTOCOL_SHA,
                     'model_calls': 0, 'executor_actions': 0})
    browser_environment(config)
    training = [json.loads(line) for line in Path(config['memory_material']).read_text().splitlines()]
    blocks = []
    for block in p['blocks']:
        require(study.sha(block['task_html_path']) == block['task_html_sha256'], 'task HTML changed')
        probe = MiniWoBAdapter(out/'reset-probes'/block['task']/f"repeat-{block['repeat_id']}",
                              browser_python=config['browser_python'])
        try:
            goal = probe.rpc('reset', seed=block['stage_reset_seed'], task=block['task'])['goal']
        finally:
            probe.close()
        task = TaskSpecification(task_id='miniwob.'+block['task'], goal=goal,
            benchmark_id='miniwob', benchmark_version='0.14.3', start_state_id='seeded-reset',
            development_partition=False)
        overlap = MiniWoBOverlapAudit(task, training)
        bound = {**block, 'task_specification': task.to_dict(), 'overlap_audit': overlap.record,
                 'eligibility': 'EXCLUDED_TRAIN_OVERLAP' if overlap.record['matching_source_tasks'] else 'ELIGIBLE'}
        blocks.append(bound)
        study.write_json(out/'prepared-blocks.json', blocks)
        print(json.dumps({'reset_probes': len(blocks), 'task': block['task'],
                          'repeat': block['repeat_id'], 'eligibility': bound['eligibility']}), flush=True)
    candidate()  # recheck model/source/memory after reset-only preparation
    check_engineering(engineering, p)
    plan = {'profile': PROFILE, 'phase': 'evaluation', 'config': config, 'blocks': blocks,
            'systems': list(p['systems']), 'episodes_planned': p['planned_episodes'],
            'runtime_protocol': p['runtime_protocol'], 'analysis': p['analysis'], 'scope': p['claim'],
            'recovery_prompt_path': old['recovery_prompt_path'], 'scoring': p['scoring'],
            'source_sha256': sources(p), 'asset_sha256': p['candidate_asset_sha256'],
            'protocol_path': str(PROTOCOL), 'protocol_sha256': PROTOCOL_SHA,
            'prerequisite': {'engineering_checks': str(engineering), 'engineering_sha256': study.sha(engineering)},
            'positive_result_required': False, 'memory_exposure_required': False,
            'preparation': {'reset_probes': len(blocks), 'model_calls': 0, 'executor_actions': 0,
                            'eligible_episodes': 4*sum(b['eligibility']=='ELIGIBLE' for b in blocks)},
            'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    study.write_json(out/'plan.json', plan)
    (out/'plan.sha256').write_text(study.sha(out/'plan.json')+'\n')
    for name in plan['source_sha256']:
        target = out/'source-snapshot'/Path(name).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
        target.chmod(0o444)
    (out/'plan.json').chmod(0o444)
    (out/'plan.sha256').chmod(0o444)
    verify_final(out)
    print(json.dumps({'status': 'FROZEN', **plan['preparation'], 'output': str(out)}))


def verify_structure(plan, p, old):
    require(plan['profile'] == PROFILE and plan['phase'] == 'evaluation', 'wrong final profile/phase')
    require(plan['config'] == final_config(p, old), 'frozen agent configuration changed')
    for key, expected in (('systems', list(p['systems'])), ('runtime_protocol', p['runtime_protocol']),
                          ('analysis', p['analysis']), ('scoring', p['scoring']),
                          ('episodes_planned', p['planned_episodes']), ('asset_sha256', p['candidate_asset_sha256']),
                          ('recovery_prompt_path', old['recovery_prompt_path'])):
        require(plan[key] == expected, 'final binding differs: ' + key)
    require(plan['protocol_path'] == str(PROTOCOL) and plan['protocol_sha256'] == PROTOCOL_SHA,
            'wrong protocol binding')
    require(len(plan['blocks']) == len(p['blocks']), 'final block count changed')
    for actual, expected in zip(plan['blocks'], p['blocks']):
        require({k:actual.get(k) for k in expected} == expected, 'task/reset binding changed')
        task = actual['task_specification']
        require(task['task_id'] == 'miniwob.'+expected['task'] and
                task['development_partition'] is False, 'wrong task binding')


def verify_final(out, *, audit_revision=False):
    out = Path(out)
    require(study.sha(out/'plan.json') == (out/'plan.sha256').read_text().strip(), 'final plan changed')
    plan = read(out/'plan.json')
    p, _, old = candidate()
    verify_structure(plan, p, old)
    require(plan['source_sha256'] == sources(p), 'final producing sources changed')
    for name, digest in plan['source_sha256'].items():
        require(study.sha(out/'source-snapshot'/Path(name).relative_to(ROOT)) == digest,
                'final source snapshot changed')
    prerequisite = plan['prerequisite']
    require(study.sha(prerequisite['engineering_checks']) == prerequisite['engineering_sha256'],
            'engineering receipt changed')
    check_engineering(prerequisite['engineering_checks'], p)
    # Eligibility is independently recomputed from the frozen material. The
    # unchanged adapter also verifies the bound goal on every actual reset.
    from web_agent.memory.miniwob_overlap import MiniWoBOverlapAudit
    from web_agent.runtime.contracts import TaskSpecification
    training = [json.loads(line) for line in Path(plan['config']['memory_material']).read_text().splitlines()]
    for b in plan['blocks']:
        require(study.sha(b['task_html_path']) == b['task_html_sha256'], 'task HTML changed')
        audit = MiniWoBOverlapAudit(TaskSpecification.from_dict(b['task_specification']), training).record
        require(audit == b['overlap_audit'], 'bound overlap decision changed')
        expected = 'EXCLUDED_TRAIN_OVERLAP' if audit['matching_source_tasks'] else 'ELIGIBLE'
        require(b['eligibility'] == expected, 'bound eligibility changed')
    return plan


@contextmanager
def final_verifier():
    previous = study.verify_plan
    def dispatch(out, **kwargs):
        if read(Path(out)/'plan.json').get('profile') == PROFILE:
            return verify_final(out, **kwargs)
        return previous(out, **kwargs)
    study.verify_plan = dispatch
    try:
        yield
    finally:
        study.verify_plan = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'verify', 'run', 'audit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.operation == 'prepare':
        require(args.engineering is not None, 'prepare requires engineering evidence')
        prepare(out, args.engineering.resolve())
    elif args.operation == 'verify':
        plan = verify_final(out)
        print(json.dumps({'status': 'PASS', 'planned_episodes': plan['episodes_planned'],
                          'memory_exposure_required': False}))
    else:
        require(read(out/'plan.json').get('profile') == PROFILE,
                'this entrypoint only runs/audits its bound final profile')
        with (out/'.coordinator.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with final_verifier():
                if args.operation == 'run':
                    study.run(out)
                else:
                    from web_agent.eval.table2.miniwob_reporting import audit_and_report
                    audit_and_report(out)


if __name__ == '__main__':
    main()
