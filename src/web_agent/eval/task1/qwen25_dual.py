"""Versioned runner for the selected Qwen2.5-VL-7B Task 1 follow-up."""
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .core import (
    ROOT,
    SOURCE_HASH,
    encode,
    execute,
    file_hash,
    load_prepared,
    read,
    request_key,
    sha,
    write_new,
)
from .qwen25_profiles import CONTRASTS, METHODS, PROFILE, ROWS, ROW_IDS, profile_spec
from .reporting import collect_records, score


def update_status(root, stage, **extra):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    value = {
        'profile': PROFILE,
        'stage': stage,
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temporary = root / 'status.tmp'
    with temporary.open('wb') as handle:
        handle.write(encode(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(root / 'status.json')
    print(json.dumps(value), flush=True)


def current_binding(config, prepared, requests):
    from .cli import binding
    from .qwen25_backends import qwen25_generation_config

    result = binding(config, prepared, requests)
    result['profile'] = profile_spec()
    result['effective_generation'] = qwen25_generation_config(config).to_dict()
    result['numerical_contract'] = {
        'family': 'qwen2_vl',
        'load_in_4bit': True,
        'quant_type': 'nf4',
        'double_quant': True,
        'compute_dtype': 'bfloat16',
        'attention': 'sdpa',
        'min_pixels': 50176,
        'max_pixels': 200704,
        'dynamic_image_grid': True,
        'nonquantized_parameters': 'float32',
    }
    return result


def existing_records(root, requests, identity):
    """Preserve and block any earlier failed or interrupted attempt."""
    root = Path(root)
    blocked = []
    records = {
        system: collect_records(root, system, requests, identity)
        for system in ROW_IDS
    }
    for system in ROW_IDS:
        folder = root / system
        for request in requests:
            key = request_key(request)
            row = records[system].get(key)
            if row and row['status'] in ('infrastructure_error', 'interrupted'):
                blocked.append(system + '/' + key)
            elif row is None and (folder / (key + '.started.json')).exists():
                write_new(folder / (key + '.json'), {
                    'request_key': key,
                    'case_id': request['case_id'],
                    'task_id': request['task_id'],
                    'phase': request['phase'],
                    'system': system,
                    'status': 'interrupted',
                    'predictions': {},
                    'model_calls': None,
                    'latency_seconds': None,
                })
                blocked.append(system + '/' + key)
    if blocked:
        raise RuntimeError('Review required for preserved attempts: ' + ','.join(blocked))
    return records


def check_record_routes(records):
    from .core import parse_response
    from .qwen25_backends import QWEN25_BACKEND
    from .qwen25_profiles import row_spec

    for system, data in records.items():
        spec = row_spec(system)
        for record in data.values():
            if record['status'] in ('interrupted', 'infrastructure_error'):
                raise ValueError('Cannot certify failed execution')
            actual = (
                record.get('row_id'),
                record.get('weight_variant'),
                record.get('method'),
                record.get('output_mode'),
                record.get('backend'),
            )
            expected = (
                spec.id,
                spec.weight_variant,
                spec.method,
                spec.output_mode,
                QWEN25_BACKEND,
            )
            if actual != expected:
                raise ValueError('Qwen2.5 record backend identity mismatch')
            if spec.output_mode == 'generation':
                parsed = parse_response(record['raw_response'], record['phase'])
                if parsed['status'] != record['status'] or parsed['predictions'] != record['predictions']:
                    raise ValueError('Raw Qwen2.5 output replay mismatch')
                if record['generated_token_count'] != len(record['generated_tokens']):
                    raise ValueError('Qwen2.5 token telemetry mismatch')
                if record['limit_hit'] != (len(record['generated_tokens']) == 128):
                    raise ValueError('Qwen2.5 token-limit telemetry mismatch')
                evidence = record['adapter_evidence']
                if spec.weight_variant == 'base':
                    if (evidence['configured_layers'] or evidence['invoked_layers']
                            or evidence['active_adapter'] is not None):
                        raise ValueError('Qwen2.5 base route contains trained adapters')
                elif not (
                    evidence['configured_layers'] > 0
                    and evidence['invoked_layers'] > 0
                    and evidence['forward_calls'] > 0
                    and evidence['active_adapter'] == 'default'
                    and evidence['restoration_verified']
                ):
                    raise ValueError('Qwen2.5 trained adapter execution not established')


def check_generation_parity(records, requests, identity):
    expected_generation = sha(encode(identity['effective_generation']))
    for method in METHODS:
        for request in requests:
            key = request_key(request)
            base = records[method + '_base'][key]
            trained = records[method + '_trained'][key]
            for field in (
                'input_tensor_hashes',
                'prompt',
                'serialized_chat',
                'base_parameter_signature_sha256',
            ):
                if base[field] != trained[field]:
                    raise ValueError('Qwen2.5 base/trained input parity mismatch: ' + field)
            if (base['generation_config_sha256'] != expected_generation
                    or trained['generation_config_sha256'] != expected_generation):
                raise ValueError('Qwen2.5 generation settings differ')


def run_rows(config, source, output, requests, identity, development=False):
    import torch
    from .qwen25_backends import Qwen25Generator, Qwen25RowBackend

    root = Path(output)
    existing = existing_records(root, requests, identity)
    for variant in ('base', 'trained'):
        group = [row for row in ROWS if row.weight_variant == variant]
        if not development and all(len(existing[row.id]) == len(requests) for row in group):
            continue
        update_status(
            root, 'loading', weight_variant=variant,
            expected_records=len(requests) * len(ROWS),
        )
        generator = Qwen25Generator(config, variant)
        signature_path = root / (variant + '-parameter-signature.json')
        if signature_path.exists():
            if read(signature_path) != generator.base_signature:
                raise ValueError('Qwen2.5 base parameter signature changed')
        else:
            write_new(signature_path, generator.base_signature)
        if generator.generation_config.to_dict() != identity['effective_generation']:
            raise ValueError('Loaded Qwen2.5 generation configuration differs from binding')
        try:
            for row in group:
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                update_status(
                    root, 'running', system=row.id, method=row.method,
                    weight_variant=variant, output_mode=row.output_mode,
                    expected_records=len(requests) * len(ROWS),
                )
                execute(
                    requests, row.id, Qwen25RowBackend(generator, row),
                    root / row.id, identity,
                )
            if development and variant == 'trained':
                update_status(root, 'head_parity', weight_variant=variant)
                parity = generator.owner.parity(requests, read(source))
                if (root / 'parity.json').exists():
                    if read(root / 'parity.json') != parity:
                        raise ValueError('Saved Qwen2.5 parity differs')
                else:
                    write_new(root / 'parity.json', parity)
        finally:
            del generator
            gc.collect()
            torch.cuda.empty_cache()
    records = existing_records(root, requests, identity)
    if any(len(records[system]) != len(requests) for system in ROW_IDS):
        raise ValueError('Missing Qwen2.5 records')
    check_record_routes(records)
    check_generation_parity(records, requests, identity)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'check', 'freeze', 'run', 'score', 'monitor'])
    parser.add_argument('--profile', required=True, choices=[PROFILE])
    parser.add_argument('--prepared', default=str(ROOT / 'docs/evidence/task1-assessment-prepared-v1'))
    parser.add_argument('--config', default=str(ROOT / 'configs/eval/task1/qwen25_dual_lab_paths.json'))
    parser.add_argument('--out')
    parser.add_argument('--source')
    parser.add_argument('--development-run')
    parser.add_argument('--freeze')
    parser.add_argument('--run')
    parser.add_argument('--csv-out')
    args = parser.parse_args(argv)
    if args.command == 'monitor':
        from .qwen25_reporting import monitor
        if not args.run:
            parser.error('--run required')
        monitor(args.run)
        return
    config = read(args.config)
    if set(config) != {'checkpoint', 'qwen25_base', 'image_root'}:
        raise ValueError('Qwen2.5 profile requires checkpoint, qwen25_base and image_root only')
    pilot = load_prepared(args.prepared)
    development = load_prepared(args.prepared, True)
    identity = current_binding(config, args.prepared, pilot)
    if not args.out:
        parser.error('--out required')
    output = Path(args.out)
    if args.command in ('run', 'score'):
        if not args.freeze:
            parser.error('--freeze required')
        frozen = read(args.freeze)
        if frozen['binding'] != identity:
            raise ValueError('Frozen Qwen2.5 profile mismatch')
    if args.command == 'score':
        from .qwen25_reporting import verified_report
        if not args.run or not args.csv_out:
            parser.error('--run and --csv-out required')
        records = existing_records(args.run, pilot, identity)
        if any(len(records[system]) != len(pilot) for system in ROW_IDS):
            raise ValueError('Missing Qwen2.5 evaluation records')
        check_record_routes(records)
        check_generation_parity(records, pilot, identity)
        targets = read(Path(args.prepared) / 'scoring-only.json')
        result = score(
            pilot, targets, records, systems=ROW_IDS, contrasts=CONTRASTS,
        )
        verified_report(
            result, records, pilot, targets, output, Path(args.csv_out), identity,
        )
        update_status(
            args.run, 'finished', report=str(output), records=2520,
            report_status=result['status'],
        )
        return
    from .cli import preflight
    from .qwen25_backends import QWEN25_CHECKPOINT_HASH, QWEN25_REVISION

    flight = preflight(
        config,
        development + pilot,
        model_revisions=[('qwen25_base', QWEN25_REVISION)],
        checkpoint_hash=QWEN25_CHECKPOINT_HASH,
        checkpoint_stage='qwen25_checkpoint',
    )
    if args.command == 'preflight':
        write_new(output, flight)
        print(flight['status'])
        return
    if flight['status'] != 'PASS':
        raise RuntimeError('Qwen2.5 preflight blocked: ' + json.dumps(flight['issues']))
    if args.command == 'freeze':
        if not args.development_run:
            parser.error('--development-run required')
        root = Path(args.development_run)
        receipt = read(root / 'receipt.json')
        if (receipt['binding'] != identity or receipt['assets'] != flight['assets']
                or receipt['parity'] != 'PASS'):
            raise ValueError('Qwen2.5 development receipt mismatch')
        for relative, digest in receipt['evidence'].items():
            if file_hash(root / relative) != digest:
                raise ValueError('Qwen2.5 development evidence changed')
        snapshot = output.parent / 'producing-source'
        snapshot.mkdir(parents=True, exist_ok=False)
        import shutil
        for relative, digest in identity['code'].items():
            if file_hash(ROOT / relative) != digest:
                raise ValueError('Sources changed while freezing Qwen2.5 profile')
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        write_new(output, {
            'binding': identity,
            'assets': flight['assets'],
            'status': 'frozen',
            'development_receipt_sha256': file_hash(root / 'receipt.json'),
        })
        print('Frozen Qwen2.5 dual profile')
        return
    if args.command == 'check':
        if not args.source or file_hash(args.source) != SOURCE_HASH:
            raise ValueError('Wrong Qwen2.5 development parity source')
    elif frozen['assets'] != flight['assets']:
        raise ValueError('Frozen Qwen2.5 assets mismatch')
    try:
        records = run_rows(
            config,
            args.source,
            output,
            development if args.command == 'check' else pilot,
            identity,
            development=args.command == 'check',
        )
        if args.command == 'check':
            from collections import Counter
            evidence = {
                str(path.relative_to(output)): file_hash(path)
                for path in output.rglob('*.json')
                if path.name not in ('status.json', 'receipt.json')
            }
            write_new(output / 'receipt.json', {
                'binding': identity,
                'assets': flight['assets'],
                'parity': 'PASS',
                'input_parity': 'PASS',
                'adapter_execution': 'PASS',
                'records': 126,
                'status_counts': {
                    system: dict(Counter(row['status'] for row in data.values()))
                    for system, data in records.items()
                },
                'evidence': evidence,
            })
        update_status(
            output,
            'development_pass' if args.command == 'check' else 'inference_complete',
            records=sum(map(len, records.values())),
            expected_records=len(development if args.command == 'check' else pilot) * len(ROWS),
        )
    except Exception as exc:
        update_status(output, 'stopped', error=str(exc))
        raise


if __name__ == '__main__':
    main()
