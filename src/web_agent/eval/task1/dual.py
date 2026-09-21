"""Versioned InternVL-only runner. Historical Qwen CLI remains available separately."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from .profiles import PROFILE, ROWS, ROW_IDS, CONTRASTS, METHODS, profile_spec
from .core import (read, write_new, file_hash, sha, encode, load_prepared, request_key,
                   execute, SOURCE_HASH, ROOT)
from .reporting import collect_records, score, markdown


def update_status(root, stage, **extra):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    value = {'profile': PROFILE, 'stage': stage,
             'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    temp = root/'status.tmp'
    with temp.open('wb') as f:
        f.write(encode(value)); f.flush(); os.fsync(f.fileno())
    temp.replace(root/'status.json')
    print(json.dumps(value), flush=True)


def current_binding(config, prepared, requests):
    from .cli import binding
    from .backends import internvl_generation_config
    result = binding(config, prepared, requests)
    result['profile'] = profile_spec()
    result['effective_generation'] = internvl_generation_config(config).to_dict()
    result['numerical_contract'] = {'load_in_4bit': True, 'quant_type': 'nf4', 'double_quant': True,
                                    'compute_dtype': 'bfloat16', 'attention': 'sdpa',
                                    'crop_to_patches': False, 'image_size': [448,448], 'nonquantized_parameters': 'float32'}
    return result


def existing_records(root, requests, identity):
    """Do not silently continue a run containing a previously failed attempt."""
    root = Path(root); blocked = []
    records = {s: collect_records(root, s, requests, identity) for s in ROW_IDS}
    for system in ROW_IDS:
        folder = root/system
        for r in requests:
            key = request_key(r); row = records[system].get(key)
            if row and row['status'] in ('infrastructure_error', 'interrupted'):
                blocked.append(system+'/'+key)
            elif row is None and (folder/(key+'.started.json')).exists():
                write_new(folder/(key+'.json'), {'request_key': key, 'case_id': r['case_id'],
                    'task_id': r['task_id'], 'phase': r['phase'], 'system': system,
                    'status': 'interrupted', 'predictions': {}, 'model_calls': None, 'latency_seconds': None})
                blocked.append(system+'/'+key)
    if blocked:
        raise RuntimeError('Review required for preserved attempts: '+','.join(blocked))
    return records


def check_record_routes(records):
    from .core import parse_response
    from .profiles import row_spec
    for system, data in records.items():
        spec = row_spec(system)
        for r in data.values():
            if r['status'] in ('interrupted', 'infrastructure_error'):
                raise ValueError('Cannot certify failed execution')
            if (r.get('row_id'),r.get('weight_variant'),r.get('method'),r.get('output_mode'),r.get('backend')) != (
                    spec.id,spec.weight_variant,spec.method,spec.output_mode,'InternVL3.5-8B-HF'):
                raise ValueError('Record backend identity mismatch')
            if spec.output_mode == 'generation':
                p = parse_response(r['raw_response'],r['phase'])
                if p['status'] != r['status'] or p['predictions'] != r['predictions']:
                    raise ValueError('Raw output replay mismatch')
                if r['generated_token_count'] != len(r['generated_tokens']):
                    raise ValueError('Token telemetry mismatch')
                if r['limit_hit'] != (len(r['generated_tokens']) == 128):
                    raise ValueError('Token limit telemetry mismatch')
                e = r['adapter_evidence']
                if spec.weight_variant == 'base':
                    if e['configured_layers'] or e['invoked_layers'] or e['active_adapter'] is not None:
                        raise ValueError('Base contains trained adapters')
                elif not (e['configured_layers'] > 0 and e['invoked_layers'] > 0
                          and e['forward_calls'] > 0 and e['active_adapter'] == 'default'
                          and e['restoration_verified']):
                    raise ValueError('Trained adapter execution not established')


def check_generation_parity(records, requests, identity):
    expected_gen = sha(encode(identity['effective_generation']))
    for method in METHODS:
        for r in requests:
            key = request_key(r)
            a = records[method+'_base'][key]; b = records[method+'_trained'][key]
            for field in ('input_tensor_hashes','prompt','serialized_chat','base_parameter_signature_sha256'):
                if a[field] != b[field]:
                    raise ValueError('Base/trained input parity mismatch: '+field)
            if a['generation_config_sha256'] != expected_gen or b['generation_config_sha256'] != expected_gen:
                raise ValueError('Generation settings differ')


def run_rows(config, prepared, source, output, requests, identity, development=False):
    from .backends import InternVLGenerator, InternVLRowBackend
    import torch
    root = Path(output)
    existing = existing_records(root, requests, identity)
    for variant in ('base', 'trained'):
        group = [r for r in ROWS if r.weight_variant == variant]
        if not development and all(len(existing[r.id]) == len(requests) for r in group):
            continue
        update_status(root, 'loading', weight_variant=variant,
                      expected_records=len(requests)*len(ROWS))
        generator = InternVLGenerator(config, variant)
        signature_path = root/(variant+'-parameter-signature.json')
        if signature_path.exists():
            if read(signature_path) != generator.base_signature: raise ValueError('Base parameter signature changed')
        else: write_new(signature_path,generator.base_signature)
        if generator.generation_config.to_dict() != identity['effective_generation']:
            raise ValueError('Loaded generation configuration differs from binding')
        try:
            for row in group:
                torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                update_status(root, 'running', system=row.id, method=row.method,
                              weight_variant=variant, output_mode=row.output_mode,
                              expected_records=len(requests)*len(ROWS))
                execute(requests, row.id, InternVLRowBackend(generator,row), root/row.id, identity)
            if development and variant == 'trained':
                update_status(root, 'head_parity', weight_variant=variant)
                parity = generator.owner.parity(requests, read(source))
                if (root/'parity.json').exists():
                    if read(root/'parity.json') != parity: raise ValueError('Saved parity differs')
                else: write_new(root/'parity.json',parity)
        finally:
            del generator; gc.collect(); torch.cuda.empty_cache()
    records = existing_records(root, requests, identity)
    if any(len(records[s]) != len(requests) for s in ROW_IDS):
        raise ValueError('Missing records')
    check_record_routes(records)
    check_generation_parity(records,requests,identity)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight','check','freeze','run','score','monitor'])
    parser.add_argument('--profile', required=True, choices=[PROFILE])
    parser.add_argument('--prepared', default=str(ROOT/'docs/evidence/task1-assessment-prepared-v1'))
    parser.add_argument('--config', default=str(ROOT/'configs/eval/task1/internvl_dual_lab_paths.json'))
    parser.add_argument('--out')
    parser.add_argument('--source')
    parser.add_argument('--development-run')
    parser.add_argument('--freeze')
    parser.add_argument('--run')
    parser.add_argument('--csv-out')
    args = parser.parse_args(argv)
    if args.command == 'monitor':
        from .dual_reporting import monitor
        if not args.run: parser.error('--run required')
        monitor(args.run); return
    config = read(args.config)
    if set(config) != {'checkpoint','internvl_base','image_root'}:
        raise ValueError('InternVL-only profile requires checkpoint, internvl_base, image_root; no Qwen backend')
    pilot = load_prepared(args.prepared); dev = load_prepared(args.prepared,True)
    identity = current_binding(config,args.prepared,pilot)
    if not args.out: parser.error('--out required')
    out = Path(args.out)
    if args.command in ('run','score'):
        if not args.freeze: parser.error('--freeze required')
        frozen = read(args.freeze)
        if frozen['binding'] != identity: raise ValueError('Frozen profile mismatch')
    if args.command == 'score':
        from .dual_reporting import verified_report
        if not args.run or not args.csv_out: parser.error('--run and --csv-out required')
        records = existing_records(args.run,pilot,identity)
        if any(len(records[s]) != len(pilot) for s in ROW_IDS): raise ValueError('Missing evaluation records')
        check_record_routes(records); check_generation_parity(records,pilot,identity)
        result = score(pilot,read(Path(args.prepared)/'scoring-only.json'),records,
                       systems=ROW_IDS,contrasts=CONTRASTS)
        verified_report(result,records,pilot,read(Path(args.prepared)/'scoring-only.json'),
                        out,Path(args.csv_out),identity)
        update_status(args.run,'finished',report=str(out),records=2520,report_status=result['status'])
        return
    from .cli import preflight
    from .backends import INTERNVL_REVISION
    flight = preflight(config,dev+pilot,model_revisions=[('internvl_base',INTERNVL_REVISION)])
    if args.command == 'preflight':
        write_new(out,flight); print(flight['status']); return
    if flight['status'] != 'PASS': raise RuntimeError('Preflight blocked: '+json.dumps(flight['issues']))
    if args.command == 'freeze':
        if not args.development_run: parser.error('--development-run required')
        root = Path(args.development_run); receipt = read(root/'receipt.json')
        if receipt['binding'] != identity or receipt['assets'] != flight['assets'] or receipt['parity'] != 'PASS':
            raise ValueError('Development receipt mismatch')
        for rel,digest in receipt['evidence'].items():
            if file_hash(root/rel) != digest: raise ValueError('Development evidence changed')
        # Keep a complete producing snapshot before inference, not only git HEAD.
        snapshot = out.parent/'producing-source'; snapshot.mkdir(parents=True,exist_ok=False)
        import shutil
        for rel,digest in identity['code'].items():
            if file_hash(ROOT/rel) != digest: raise ValueError('Sources changed while freezing')
            dest = snapshot/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,dest)
        write_new(out,{'binding':identity,'assets':flight['assets'],'status':'frozen',
                       'development_receipt_sha256':file_hash(root/'receipt.json')})
        print('Frozen InternVL dual profile'); return
    if args.command == 'check':
        if not args.source or file_hash(args.source) != SOURCE_HASH: raise ValueError('Wrong development parity source')
    elif frozen['assets'] != flight['assets']:
        raise ValueError('Frozen assets mismatch')
    try:
        records = run_rows(config,args.prepared,args.source,out,dev if args.command=='check' else pilot,
                           identity,development=args.command=='check')
        if args.command == 'check':
            from collections import Counter
            evidence = {str(p.relative_to(out)):file_hash(p) for p in out.rglob('*.json')
                        if p.name not in ('status.json','receipt.json')}
            write_new(out/'receipt.json',{'binding':identity,'assets':flight['assets'],'parity':'PASS',
                       'input_parity':'PASS','adapter_execution':'PASS','records':126,
                       'status_counts':{s:dict(Counter(r['status'] for r in data.values())) for s,data in records.items()},
                       'evidence':evidence})
        update_status(out,'development_pass' if args.command=='check' else 'inference_complete',
                      records=sum(map(len,records.values())))
    except Exception as exc:
        update_status(out,'stopped',error=str(exc)); raise

if __name__ == '__main__':
    main()
