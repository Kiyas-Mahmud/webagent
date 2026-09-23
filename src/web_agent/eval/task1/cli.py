"""Usage: python -m web_agent.eval.task1.cli --help. No automatic test-split reads."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from .core import (ROOT,SYSTEMS,CHECKPOINT_HASH,SOURCE_HASH,read,write_new,file_hash,sha,encode,
                   prepare,load_prepared,execute,request_key)
from .backends import QWEN_REVISION,INTERNVL_REVISION,GENERATION,require_local_images

def code_identity():
    paths=list((ROOT/'src/web_agent').rglob('*.py'))+list((ROOT/'configs/eval/task1').rglob('*'))
    paths+=[ROOT/'scripts/external_agents/task1.py']
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths) if p.is_file() and '__pycache__' not in p.parts}

def environment():
    names=('torch','transformers','peft','bitsandbytes','numpy','scikit-learn','Pillow','accelerate')
    result={'python':platform.python_version(),'architecture':platform.machine()}
    for name in names:
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]=None
    return result

def tree_identity(folder):
    root=Path(folder)
    if not root.is_dir(): raise FileNotFoundError(str(root))
    files={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and '.cache' not in p.parts}
    if not any(k.endswith('.safetensors') for k in files): raise ValueError('Missing model weights: '+str(root))
    if 'config.json' not in files: raise ValueError('Missing base config')
    return files

def preflight(config, requests, model_revisions=None, checkpoint_hash=CHECKPOINT_HASH,
              checkpoint_stage='pc03_checkpoint'):
    issues=[]; assets={}
    checkpoint=Path(config['checkpoint'])
    if not checkpoint.is_file(): issues.append({'stage':checkpoint_stage,'reason':'Selected checkpoint file unavailable','path':str(checkpoint)})
    elif file_hash(checkpoint)!=checkpoint_hash: issues.append({'stage':checkpoint_stage,'reason':'SHA-256 mismatch'})
    else: assets['checkpoint_sha256']=checkpoint_hash
    for key,revision in (model_revisions if model_revisions is not None else [('qwen_base',QWEN_REVISION),('internvl_base',INTERNVL_REVISION)]):
        path=Path(config[key])
        try:
            if path.name!=revision: raise ValueError('Expected snapshot directory named '+revision)
            assets[key]=tree_identity(path)
        except (ValueError,FileNotFoundError) as exc: issues.append({'stage':key,'reason':str(exc)})
    # The user-authorized local mount is explicit in the frozen configuration.
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Unchanged model requires a CUDA device with BF16 support')
        assets['gpu']={'name':torch.cuda.get_device_name(0),
                       'capability':list(torch.cuda.get_device_capability(0)),
                       'cuda':torch.version.cuda}
    except Exception as exc:
        issues.append({'stage':'cuda_compatibility','reason':str(exc)})
    try:
        require_local_images(config['image_root'])
        from PIL import Image
        root=Path(config['image_root']).resolve(); images={}
        for r in requests:
            for key in ('before_image','after_image'):
                rel=r[key]
                if rel in images: continue
                p=(root/rel).resolve()
                if not p.is_relative_to(root): raise ValueError('Image outside root')
                with Image.open(p) as im: im.verify()
                images[rel]=file_hash(p)
        assets['images']=images
    except Exception as exc: issues.append({'stage':'image_assets','reason':str(exc)})
    upstream=read(ROOT/'configs/eval/task1/upstream/manifest.json')
    for name,spec in upstream.items():
        if file_hash(ROOT/spec['file'])!=spec['sha256']: issues.append({'stage':'upstream_sources','reason':name+' source changed'})
    env=environment()
    for name,value in env.items():
        if value is None:
            issues.append({'stage':'installed_dependencies','reason':'Missing '+name})
    return {'status':'BLOCKED' if issues else 'PASS','issues':issues,'assets':assets,'environment':env}

def binding(config, prepared, requests):
    return {'config':config,'prepared_manifest_sha256':file_hash(Path(prepared)/'manifest.json'),
            'code':code_identity(),'environment':environment(),'generation':GENERATION,
            'request_set_sha256':sha(encode(requests))}

def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    profile = None
    for index, value in enumerate(argv):
        if value == '--profile' and index + 1 < len(argv):
            profile = argv[index + 1]
        elif value.startswith('--profile='):
            profile = value.split('=', 1)[1]
    if profile is not None:
        if profile == 'internvl_dual_v2':
            from .dual import main as dual_main
            return dual_main(argv)
        if profile == 'qwen25_dual_v1':
            from .qwen25_dual import main as qwen25_main
            return qwen25_main(argv)
        raise ValueError('Unknown Task 1 profile: ' + profile)
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--source',required=True); prep.add_argument('--out',required=True)
    prep.add_argument('--selection',default=str(ROOT/'docs/evidence/task1-mini-validation-240-v1'))
    for name in ('preflight','check','freeze','run','score'):
        s=sub.add_parser(name); s.add_argument('--prepared',required=True); s.add_argument('--config',required=True); s.add_argument('--out',required=True)
        if name=='check': s.add_argument('--source',required=True)
        if name=='freeze': s.add_argument('--development-run',required=True)
        if name=='run': s.add_argument('--system',choices=SYSTEMS,required=True); s.add_argument('--freeze',required=True)
        if name=='score': s.add_argument('--run',required=True); s.add_argument('--freeze',required=True)
    args=p.parse_args(argv)
    if args.command=='prepare':
        print(json.dumps(prepare(args.source,args.selection,args.out),indent=2)); return
    config=read(args.config)
    if set(config)!={'checkpoint','qwen_base','internvl_base','image_root'}: raise ValueError('Unexpected runtime configuration')
    dev=load_prepared(args.prepared,True); pilot=load_prepared(args.prepared)
    current=binding(config,args.prepared,pilot)
    out=Path(args.out)
    if args.command=='score':
        from .reporting import collect_records,score,markdown
        frozen=read(args.freeze)
        if frozen['binding']!=current: raise ValueError('Scoring source/environment differs from frozen profile')
        records={s:collect_records(args.run,s,pilot,current) for s in SYSTEMS}
        report=score(pilot,read(Path(args.prepared)/'scoring-only.json'),records)
        out.mkdir(parents=True,exist_ok=False)
        write_new(out/'results.json',report)
        with (out/'results.md').open('x') as f:f.write(markdown(report))
        print(report['status']); return
    flight=preflight(config,dev+pilot)
    if args.command=='preflight':
        write_new(out,flight); print(json.dumps({'status':flight['status'],'issues':flight['issues']},indent=2)); return
    if flight['status']!='PASS':
        raise RuntimeError('Preflight blocked: '+json.dumps(flight['issues']))
    if args.command=='check':
        if file_hash(args.source)!=SOURCE_HASH: raise ValueError('Wrong parity validation source')
        from .backends import QwenAssessor,PC03Assessor
        import gc,torch
        for system in SYSTEMS:
            backend=PC03Assessor(config) if system=='pc03' else QwenAssessor(system,config)
            execute(dev,system,backend,out/system,current)
            if system=='pc03':
                parity=backend.parity(dev,read(args.source))
                if not (out/'parity.json').exists(): write_new(out/'parity.json',parity)
            del backend; gc.collect(); torch.cuda.empty_cache()
        from .reporting import collect_records
        rows={s:collect_records(out,s,dev,current) for s in SYSTEMS}
        if any(len(rows[s])!=len(dev) or any(r['status'] not in ('valid','abstention','parse_error') for r in rows[s].values()) for s in SYSTEMS):
            raise ValueError('Development path incomplete')
        # Receipt certifies execution/parity, not accuracy; original raw outputs remain.
        write_new(out/'receipt.json',{'binding':current,'assets':flight['assets'],'parity':'PASS',
                  'records_per_system':len(dev),'status_counts':{s:dict(__import__('collections').Counter(r['status'] for r in rows[s].values())) for s in SYSTEMS},
                  'evidence':{str(p.relative_to(out)):file_hash(p) for p in out.rglob('*.json')}})
        print('Development execution and PC-03 parity PASS'); return
    if args.command=='freeze':
        root=Path(args.development_run); receipt=read(root/'receipt.json')
        if receipt['binding']!=current or receipt['assets']!=flight['assets'] or receipt['parity']!='PASS':
            raise ValueError('Development receipt does not match current profile/assets')
        for path,digest in receipt['evidence'].items():
            if file_hash(root/path)!=digest: raise ValueError('Development evidence changed')
        write_new(out,{'binding':current,'assets':flight['assets'],
                       'development_receipt_sha256':file_hash(root/'receipt.json'),'status':'frozen'})
        print('Frozen; no comparative inference launched'); return
    if args.command=='run':
        frozen=read(args.freeze)
        if frozen['binding']!=current or frozen['assets']!=flight['assets']: raise ValueError('Frozen profile mismatch')
        from .backends import QwenAssessor,PC03Assessor
        backend=PC03Assessor(config) if args.system=='pc03' else QwenAssessor(args.system,config)
        execute(pilot,args.system,backend,out/args.system,current)
        print('System pass finished; score separately')

if __name__=='__main__': main()
