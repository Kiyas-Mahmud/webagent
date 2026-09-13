"""Seven frozen development inputs; one instruction deletion, no browser actions."""
from pathlib import Path
import hashlib
import json
import sys

from PIL import Image
from web_agent.runtime.contracts import PolicyObservation, RuntimeTaskView
from web_agent.runtime.named_target_policy import resolve_named_target
from web_agent.runtime.policy import ActionParseError


def main():
    out = Path(sys.argv[1])
    source = out / 'baseline' / 'run.py'
    namespace = {}
    # Reuse the authenticated loader declarations, stopping before any browser code.
    prefix = source.read_text().split('import subprocess,threading,datetime')[0]
    exec(compile(prefix, str(source), 'exec'), namespace)
    runtime = namespace['bundle'].e0.action_predictor.__self__
    old = Path('/home/aiub/kiyas/table2-evidence/miniwob-completion-development-v1')
    prompt = Path('/home/aiub/kiyas/table2-evidence/miniwob-feasibility/recovery-planner-v2/prompt.txt').read_text()
    sentence = 'Use TYPE for entering text, preserving the exact requested text in value. '
    results = []
    cases = [(t, p) for t in json.loads((old/'plan.json').read_text())['tasks']
             for p in sorted((old/t/'E2'/'named-recovery-outputs').glob('output-*.json'))]
    assert len(cases) == 7
    target = out/'prompt-probe'; target.mkdir(exist_ok=False)
    for task_name, path in cases:
        saved = json.loads(path.read_text())
        suffix = saved['input_suffix']
        assert suffix.count(sentence) == 1
        assert hashlib.sha256(prompt.encode()).hexdigest() == saved['prompt_sha256']
        task = RuntimeTaskView(task_id=saved['task_id'], goal=json.loads((old/task_name/'task-binding.json').read_text())['task']['goal'])
        screenshot = old/task_name/'E2'/f"observation-{saved['observation_id'].rsplit(':',1)[1]}.png"
        assert hashlib.sha256(screenshot.read_bytes()).hexdigest() == saved['screenshot_sha256']
        with Image.open(screenshot) as im: width, height = im.size
        context = json.loads(suffix.split('\ncausal_recovery_context: ')[1].split('\nnext_action_request: ')[0])
        observation = PolicyObservation(task_id=task.task_id, goal=task.goal,
            observation_id=saved['observation_id'], screenshot_sha256=saved['screenshot_sha256'],
            screenshot_path=str(screenshot), width=width, height=height,
            url='file:///home/aiub/kiyas/table2-inputs/miniwob-plusplus/miniwob/html/miniwob/'+task_name+'.html',
            title='', current_page_state={'visible_controls':context['current_visible_controls']})
        changed = suffix.replace(sentence, '')
        raw = runtime._generate(prompt=prompt, task=task, observation=observation, suffix=changed)
        row = dict(task=task_name, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   original=saved, changed_suffix=changed, raw_response=raw, diagnostic_only=True)
        try:
            row['resolved'], row['resolution'] = resolve_named_target(raw, observation, allow_json_fence=True, allow_role_target=True)
            row['status'] = 'RESOLVED'
        except ActionParseError as exc:
            row.update(status='REJECTED', error=str(exc))
        results.append(row)
        (target/f'probe-{len(results):02d}.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps({'case':len(results),'task':task_name,'old_status':saved['status'],'new_status':row['status'],'response':raw}),flush=True)
    (target/'results.json').write_text(json.dumps({'status':'COMPLETE','cases':results,'live_episodes':0,
        'changed_factor':sentence,'old_rejected':sum(r['original']['status']=='REJECTED' for r in results),
        'new_rejected':sum(r['status']=='REJECTED' for r in results)},indent=2)+'\n')


if __name__ == '__main__': main()
