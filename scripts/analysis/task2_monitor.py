"""Read-only Task 2 progress; never initializes a model or changes run records."""
import argparse
import json
from pathlib import Path
from collections import Counter

p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
r=Path(args.run)
status=json.loads((r/'status.json').read_text())
completed=[]
for f in sorted((r/'episodes').glob('*/*/*/result.json')):
    completed.append(json.loads(f.read_text()))
rows={}
for s in ('A','B','C'):
    selected=[x for x in completed if x['system']==s]
    rows[s]={'saved':len(selected),'completed':sum(x['completion'] for x in selected),
             'model_calls':sum(sum(x['model_calls'].values()) for x in selected)}
status['systems']=rows
status['saved_raw_generations']=len(list((r/'episodes').glob('*/*/*/actor-*-raw.json')))
status['infrastructure_errors']=[str(x) for x in (r/'episodes').glob('*/*/*/infrastructure-error.json')]
status['stage_note']='Counts are provisional until independent audit; engineering checks are not live successes.'
print(json.dumps(status,indent=2))
