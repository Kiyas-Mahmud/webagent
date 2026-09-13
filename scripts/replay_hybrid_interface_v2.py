"""Read-only saved-response replay; no model calls or browser execution.

Reports parser/target resolution only, never hypothetical task completions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from web_agent.runtime.contracts import PolicyObservation
from web_agent.runtime.hybrid_interface import parse_action, semantic_context
from web_agent.runtime.policy import ActionParseError


def replay(root, out):
    out.mkdir(parents=True, exist_ok=False)
    cases=[]; counts=Counter(); bindings={}
    for path in sorted(root.glob('*/repeat-*/H*/hybrid-normal-outputs/proposal-*.json')):
        data=path.read_bytes();bindings[str(path)]=hashlib.sha256(data).hexdigest()
        receipt=json.loads(data);view=PolicyObservation.from_dict(receipt['observation'])
        assert view.record_sha256==receipt['observation_sha256']
        case={'receipt':str(path),'previous_status':receipt['status']}
        try:
            parsed,resolution=parse_action(receipt['raw_response'],view)
        except ActionParseError as exc:
            assert receipt['status']!='RESOLVED', 'Previously valid response regressed'
            case.update(status='REJECTED',error=str(exc),stage=getattr(exc,'failure_stage','parsing'))
            counts['still_rejected']+=1
        else:
            case.update(status='RESOLVED',proposal=parsed,resolution=resolution)
            if receipt['status']=='RESOLVED':
                assert parsed==receipt['resolved_proposal'], 'Previously valid proposal changed'
                counts['previously_resolved_unchanged']+=1
            else:
                counts['newly_resolvable']+=1
        cases.append(case)
    parity=[]
    for block in sorted(root.glob('*/repeat-*')):
        files=[block/s/'hybrid-normal-outputs/proposal-0001.json' for s in ('H1','H2','H3')]
        if not all(p.exists() for p in files):continue
        records=[json.loads(p.read_text()) for p in files]
        projections=[semantic_context(json.loads(r['input_suffix'].split(': ',1)[1]),episode_id=r['episode_id']) for r in records]
        assert projections[0]==projections[1]==projections[2],str(block)
        assert len({r['observation']['screenshot_sha256'] for r in records})==1
        assert len({r['task']['goal'] for r in records})==1
        parity.append({'block':str(block),'semantic_context_equal':True,'screenshot_equal':True,'goal_equal':True})
    report={'status':'PASS','evidence_type':'SAVED_RESPONSE_ENGINEERING_REPLAY',
        'actual_model_calls':0,'browser_actions':0,'live_episodes':0,'responses':len(cases),
        'counts':dict(counts),'paired_initial_contexts':parity,
        'claim':'Resolution of saved responses is not a browser result or a test of the new prompt.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (out/'input-sha256.json').write_text(json.dumps(bindings,indent=2)+'\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();replay(args.input,args.output)
