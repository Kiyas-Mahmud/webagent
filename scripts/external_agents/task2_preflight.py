"""Read-only Task 2 asset/contract preflight. Does not install, load models or run agents."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess

from web_agent.eval.task1.core import CHECKPOINT_HASH, file_hash, write_new
from web_agent.eval.task2.memory_contract import verify_memory_query_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    config_path = Path('configs/eval/task2/development_candidate_v1.json')
    config = json.loads(config_path.read_text())
    assets = json.loads(Path('configs/eval/task1/internvl_dual_lab_paths.json').read_text())
    checkpoint_hash = file_hash(assets['checkpoint'])
    if checkpoint_hash != CHECKPOINT_HASH:
        raise ValueError('Selected InternVL checkpoint changed')
    final_manifest = Path('.task1-assets/runs/task1-internvl-dual-v2/final-evidence-manifest-20260916.json')
    preserved = json.loads(final_manifest.read_text())
    for path, digest in preserved['files'].items():
        if file_hash(path) != digest:
            raise ValueError('Task 1 evidence changed: ' + path)
    store = Path(config['memory']['store_directory'])
    manifest_hash = file_hash(store / 'manifest.json')
    store_check = verify_memory_query_identity(
        store, expected_manifest_sha256=manifest_hash,
        query_checkpoint_sha256=config['memory']['required_query_checkpoint_sha256'],
        query_dimension=768)
    try:
        verify_memory_query_identity(store, expected_manifest_sha256=manifest_hash,
                                     query_checkpoint_sha256=CHECKPOINT_HASH, query_dimension=768)
    except ValueError as exc:
        if 'MEMORY_EMBEDDING_SPACE_MISMATCH' not in str(exc):
            raise
        mismatch = str(exc)
    else:
        raise AssertionError('Unexpected InternVL compatibility with the historical memory store')
    packages = {}
    for name in ('browser-use', 'transformers', 'peft', 'torch', 'playwright'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    browser_python = Path('/home/aiub/kiyas/table2-envs/miniwob-feasibility/bin/python')
    browser = subprocess.run([str(browser_python), '-c',
        'from importlib.metadata import version; import json; print(json.dumps({n:version(n) for n in ("browsergym-miniwob","playwright","gymnasium")}))'],
        capture_output=True, text=True, timeout=30, check=True)
    agent_python = Path(config['agent_candidate']['isolated_python'])
    agent = subprocess.run([str(agent_python), '-c',
        'from importlib.metadata import version; print(version("browser-use"))'],
        capture_output=True, text=True, timeout=30, check=True)
    paths = [config_path, Path(__file__)] + sorted(Path('src/web_agent/eval/task2').glob('*.py')) + sorted(Path('tests/task2').glob('*.py'))
    result = dict(schema='task2.preflight.v1', status='ENGINEERING_START_ONLY',
                  live_ready=False, final_evaluation_ready=False, model_calls=0, browser_episodes=0,
                  checkpoint_sha256=checkpoint_hash, model_environment=packages,
                  browser_environment=json.loads(browser.stdout),
                  isolated_agent_environment={'python':str(agent_python), 'browser_use':agent.stdout.strip()},
                  host_running_jobs_verified=False,
                  memory_manifest_sha256=manifest_hash, original_query_asset_check=store_check,
                  internvl_direct_query_rejection=mismatch,
                  task1_files_verified_unchanged=len(preserved['files']),
                  task1_final_manifest_sha256=file_hash(final_manifest),
                  source_hashes={str(p):file_hash(p) for p in paths},
                  pending=['Local InternVL actor adapter and atomic action execution',
                           'Live head parity including strategy and recovery transitions',
                           'Incident, budget and continuation controller',
                           'Memory query route decision and live query parity',
                           'Matched development protocol and host job/GPU check'])
    write_new(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
