"""Config-driven Task 2 campaigns: headroom screen, development, evaluation.

Reuses the audited native episode loop unchanged. The protocol config fixes the
backend, budgets, screened families and the family-selection rule before any
outcome is observed. No training, no automatic episode retries.
"""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from web_agent.eval.task1.core import file_hash, write_new
from web_agent.eval.task2.campaign import BROWSER_PYTHON, ROOT, browser_worker, source_paths
from web_agent.eval.task2.ipc import Worker

PHASES = ('screen', 'development', 'evaluation')
REGRESSION = ROOT/'.task2-assets/task2-regression-qwen25-v5.json'
ENGINEERING = {'executor': ROOT/'.task2-assets/engineering-six-actions-v2/result.json',
               'model': ROOT/'.task2-assets/engineering-model-qwen25-v1/result.json'}
MEMORY_ITEMS = Path('/home/aiub/kiyas/table2-evidence/p4-local-label-memory-v1/memory-items.jsonl')
MEMORY_ROOT = Path('/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1')
MINIWOB_HTML = Path('/home/aiub/kiyas/table2-inputs/miniwob-plusplus/miniwob/html')


def update_status(root, **value):
    value['pid'] = os.getpid()
    temp = root/'status.tmp'
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(root/'status.json')


def select_families(audit, rule):
    """Apply the frozen headroom rule to a completed A-only screen audit."""
    rates = {row['task']: row for row in audit['per_family'] if row['system'] == 'A'}
    for row in rates.values():
        if not row['episodes']:
            raise ValueError('Unscreened family: '+row['task'])
    # Some but not all eligible screened resets completed (1-4 of 5 with no exclusions).
    kept = sorted((t for t, r in rates.items() if 0 < r['completed'] < r['episodes']),
                  key=lambda t: (abs(rates[t]['completed']/rates[t]['episodes']-0.5), t))[:rule['max_families']]
    if len(kept) < rule['min_families']:
        kept += sorted(t for t, r in rates.items() if r['completed'] == 0)[:rule['min_families']-len(kept)]
    return sorted(kept), {t: rates[t]['completed'] for t in sorted(rates)}


def bindings():
    from web_agent.eval.task1.qwen25_backends import QWEN25_CHECKPOINT_HASH
    paths = json.loads((ROOT/'configs/eval/task1/qwen25_dual_lab_paths.json').read_text())
    identities = json.loads((ROOT/'.task1-assets/runs/task1-qwen25-dual-v1/freeze.json').read_text())['assets']
    if identities['checkpoint_sha256'] != QWEN25_CHECKPOINT_HASH:
        raise ValueError('Task 1 Qwen2.5 identity is not the selected checkpoint')
    bound = {paths['checkpoint']: QWEN25_CHECKPOINT_HASH, str(REGRESSION): file_hash(REGRESSION),
             str(MEMORY_ITEMS): file_hash(MEMORY_ITEMS)}
    for name, digest in identities['qwen25_base'].items():
        bound[str(Path(paths['qwen25_base'])/name)] = digest
    for path in ENGINEERING.values():
        bound[str(path)] = file_hash(path)
    manifest = json.loads((MEMORY_ROOT/'manifest.json').read_text())
    bound[str(MEMORY_ROOT/'manifest.json')] = file_hash(MEMORY_ROOT/'manifest.json')
    for name, digest in manifest['files'].items():
        bound[str(MEMORY_ROOT/name)] = digest
    old = json.loads((ROOT/'configs/eval/table2/miniwob_hybrid_dev_v3.json').read_text())
    bound[old['asset_paths']['checkpoint']] = manifest['checkpoint_sha256']
    for item in json.loads((Path(old['asset_paths']['export'])/'base_snapshot_manifest.json').read_text())['files']:
        bound[str(Path(old['asset_paths']['base_snapshot'])/item['path'])] = item['sha256']
    for path in Path(old['asset_paths']['export']).glob('*.json'):
        bound[str(path)] = file_hash(path)
    native = ROOT/'.task2-assets/browser-use-env/lib/python3.12/site-packages/browser_use'
    for path in sorted(native.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.md', '.js'):
            bound[str(path)] = file_hash(path)
    for path, digest in bound.items():
        if file_hash(path) != digest:
            raise ValueError('Asset identity changed: '+path)
    for path in sorted(MINIWOB_HTML.rglob('*')):
        if path.is_file():
            bound[str(path)] = file_hash(path)
    return bound


def blocks_for(root, protocol, phase, families, repeats):
    from web_agent.eval.table2.miniwob_study import runtime_protocol
    from web_agent.memory.miniwob_overlap import MiniWoBOverlapAudit
    from web_agent.runtime.protocol import RuntimeStage
    rng = runtime_protocol(f'{protocol}-{phase}').rng_factory()
    training = [json.loads(line) for line in MEMORY_ITEMS.read_text().splitlines() if line.strip()]
    blocks = []; worker = browser_worker(root)
    try:
        for task in families:
            for repeat in range(repeats):
                seed = rng.seed_for_key(rng.key(task_id='miniwob.'+task, repeat_id=repeat, matched_seed=42,
                                                stage=RuntimeStage.RESET, decision_index=0))
                observed = worker.call('reset', episode_id=f'probe-{task}-{repeat}', task=task, seed=seed % (2**32))
                overlap = MiniWoBOverlapAudit(SimpleNamespace(goal=observed['goal'], task_id='miniwob.'+task), training)
                blocks.append({'task': task, 'repeat_id': repeat, 'stage_reset_seed': seed,
                               'browser_reset_seed': seed % (2**32), 'goal': observed['goal'],
                               'eligible': not overlap.record['matching_source_tasks'], 'overlap': overlap.record})
    finally:
        worker.close()
    return blocks


def require_audited(folder, sources, label):
    folder = Path(folder).resolve()
    audit = json.loads((folder/'audit.json').read_text())
    plan = json.loads((folder/'plan.json').read_text())
    if audit['status'] != 'PASS':
        raise ValueError(label+' audit PASS required')
    if plan['sources'] != sources:
        raise ValueError('Implementation changed since '+label)
    return folder, plan, audit


def freeze(root, phase, protocol_path, previous=None):
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    config = json.loads(protocol_path.read_text())
    sources = {str(p): file_hash(p) for p in source_paths()}
    receipt = json.loads(REGRESSION.read_text())
    if receipt['status'] != 'PASS':
        raise ValueError('Regression PASS required')
    for path, digest in receipt['sources'].items():
        if sources.get(path) != digest:
            raise ValueError('Untested producing source: '+path)
    for name, path in ENGINEERING.items():
        if json.loads(path.read_text())['status'] != 'PASS':
            raise ValueError('Engineering check required: '+name)
    earlier_seeds = set(); selection = None
    if phase == 'screen':
        families, repeats = config['screen']['families'], config['screen']['repeats']
    elif phase == 'development':
        folder, plan, audit = require_audited(previous, sources, 'screen')
        if plan['phase'] != 'screen':
            raise ValueError('Development follows the headroom screen')
        families, screened = select_families(audit, config['selection_rule'])
        selection = {'screen': str(folder), 'screen_audit_sha256': file_hash(folder/'audit.json'),
                     'screened_A_completions': screened, 'selected': families}
        repeats = config['development']['repeats']
        earlier_seeds = {b['browser_reset_seed'] for b in plan['blocks']}
    else:
        folder, plan, audit = require_audited(previous, sources, 'development')
        if plan['phase'] != 'development':
            raise ValueError('Evaluation follows development')
        families = plan['selection']['selected']
        selection = dict(plan['selection'], development=str(folder),
                         development_audit_sha256=file_hash(folder/'audit.json'))
        repeats = math.ceil(config['evaluation']['target_pairs_per_contrast']/len(families))
        earlier_seeds = {b['browser_reset_seed'] for b in plan['blocks']}
        screen_plan = json.loads((Path(plan['selection']['screen'])/'plan.json').read_text())
        earlier_seeds |= {b['browser_reset_seed'] for b in screen_plan['blocks']}
    root.mkdir(parents=True, exist_ok=False)
    blocks = blocks_for(root, config['protocol'], phase, families, repeats)
    if earlier_seeds & {b['browser_reset_seed'] for b in blocks}:
        raise ValueError('Reset seeds must be disjoint from earlier phases')
    for path in source_paths():
        target = root/'producing-source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    bound = bindings()
    bound[str(protocol_path)] = file_hash(protocol_path)
    systems = config[phase]['systems']
    plan = {'schema': 'task2.native-campaign.v4', 'protocol': config['protocol'], 'phase': phase,
            'systems': systems, 'backend': config['backend'], 'actor_backend': config['actor_backend'],
            'actor': config['actor'], 'assessor': config['assessor'], 'memory_query': config['memory_query'],
            'settings': dict(config['settings'], actor_label=config['actor_label']),
            'families': families, 'repeats': repeats, 'selection': selection, 'blocks': blocks,
            'sources': sources, 'bindings': bound, 'analysis': config['analysis'],
            'comparison': 'Package B-A; memory package C-B; no native-agent superiority or isolated-head causal claim'}
    write_new(root/'plan.json', plan)
    update_status(root, stage='frozen', saved=0, expected=sum(b['eligible'] for b in blocks)*len(systems))
    return plan


async def run(root):
    from web_agent.eval.task2.live import episode
    root = Path(root).resolve(); plan = json.loads((root/'plan.json').read_text())
    for group in ('sources', 'bindings'):
        for path, digest in plan[group].items():
            if file_hash(path) != digest:
                raise ValueError('Frozen file changed: '+path)
    completed = []
    for block in plan['blocks']:
        if not block['eligible']:continue
        for system in plan['systems']:
            folder = root/'episodes'/block['task']/str(block['repeat_id'])/system
            if folder.exists():
                if not (folder/'result.json').exists():
                    raise RuntimeError('Preserved unfinished episode requires review: '+str(folder))
                completed.append(json.loads((folder/'result.json').read_text()))
    expected = sum(b['eligible'] for b in plan['blocks'])*len(plan['systems'])
    model = Worker(ROOT/'.venv/bin/python', 'web_agent.eval.task2.model_worker', [plan['backend']],
                   root/f'model-{len(completed):04d}.log')
    browser = Worker(BROWSER_PYTHON, 'web_agent.eval.task2.browser_worker', [root/'live-browser'],
                     root/f'live-browser-{len(completed):04d}.log',
                     env={'PLAYWRIGHT_BROWSERS_PATH': '/home/aiub/kiyas/table2-inputs/miniwob-browsers'})
    try:
        update_status(root, stage='loading_models', saved=len(completed), expected=expected)
        initialization = await asyncio.to_thread(model.call, 'initialize', timeout=1200)
        write_new(root/f'initialization-{len(completed):04d}.json', initialization)
        for block in plan['blocks']:
            if not block['eligible']:continue
            for system in plan['systems']:
                folder = root/'episodes'/block['task']/str(block['repeat_id'])/system
                if (folder/'result.json').exists():continue
                update_status(root, stage='running', saved=len(completed), expected=expected,
                              task=block['task'], repeat=block['repeat_id'], system=system)
                result = await episode(system=system, block=block, root=folder, browser_worker=browser,
                                       model_worker=model, settings=plan['settings'])
                completed.append(result)
                print(json.dumps({'saved': len(completed), 'expected': expected, 'episode': result['episode_id'],
                                  'completion': result['completion']}), flush=True)
        update_status(root, stage='execution_complete_audit_pending', saved=len(completed), expected=expected)
    except BaseException as exc:
        update_status(root, stage='review_required', saved=len(completed), expected=expected, error=str(exc))
        raise
    finally:
        browser.close(); model.close()


def main():
    os.environ['ANONYMIZED_TELEMETRY'] = 'false'
    os.environ['BROWSER_USE_CONFIG_DIR'] = str(ROOT/'.task2-assets/browser-use-config')
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'run'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--protocol', default=str(ROOT/'configs/eval/task2/qwen25_v1.json'))
    parser.add_argument('--previous', help='audited screen (for development) or development (for evaluation)')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.out, args.phase, args.protocol, args.previous)
    else:
        asyncio.run(run(args.out))


if __name__ == '__main__':
    main()
