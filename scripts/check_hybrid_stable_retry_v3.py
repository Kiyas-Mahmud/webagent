"""Bounded real-browser engineering check using an authored local HTML fixture.

The production MiniWoB adapter, worker, parameter resolver, recovery controller,
executor and action serializer run unchanged. Only gym.make is substituted in
the worker with an authored fixture environment backed by real Playwright.
No MiniWoB task, model, GPU, dataset, Gold image or archived evidence is loaded.
The first request is deliberately rejected before browser dispatch in each case.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
FIXTURE_GOAL = 'Engineering fixture: enter the issued value into Person.'
ISSUED_VALUE = '  Exact Case "value"  '
HTML = '''<!doctype html><html><body style="margin:40px">
<label for="person">Person</label><input id="person" name="person" type="text">
<label for="other">Other</label><input id="other" name="other" type="text">
</body></html>'''


def worker(folder: Path) -> None:
    import gymnasium as gym
    import numpy as np
    from PIL import Image
    from playwright.sync_api import sync_playwright
    import browsergym.miniwob  # noqa: F401; production worker imports this module.
    from web_agent.benchmarks import miniwob_worker

    class FixtureEnvironment:
        def __init__(self):
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page(viewport={'width': 900, 'height': 400})
            self.unwrapped = self

        def reset(self, *, seed):
            del seed
            self.page.set_content(HTML)
            return self._get_obs(), {}

        def _get_obs(self):
            with Image.open(io.BytesIO(self.page.screenshot())) as screenshot:
                pixels = np.array(screenshot.convert('RGB'))
            return {'goal': FIXTURE_GOAL, 'screenshot': pixels, 'last_action_error': ''}

        def step(self, code):
            exec(code, {'page': self.page})
            # A fixed fixture placeholder, never a task score or success signal.
            return self._get_obs(), 0.0, False, False, {'task_info': {'RAW_REWARD_GLOBAL': 0.0}}

        def close(self):
            self.browser.close()
            self.playwright.stop()

    def fixture_make(name, **kwargs):
        del kwargs
        if name != 'browsergym/miniwob.fixture-stable-retry-v3':
            raise ValueError('engineering harness only permits its authored fixture')
        return FixtureEnvironment()

    gym.make = fixture_make
    sys.argv = [sys.argv[0], str(folder)]
    miniwob_worker.main()


def check(browser_python: str, browser_binaries: str, output: Path) -> dict:
    from web_agent.benchmarks.browsergym_miniwob import MiniWoBAdapter
    from web_agent.runtime.action_parameters import DeterministicParameterProvider, concretize
    from web_agent.runtime.contracts import (
        ActionType, ConcreteAction, ExecutionStatus, PreActionDecision,
        RecoveryDecision, RecoveryStrategy, TaskSpecification, probability_map,
    )
    from web_agent.runtime.event_log import _payload
    from web_agent.runtime.executor import Executor
    from web_agent.runtime.observation import ObservationBuilder
    from web_agent.runtime.protocol import RuntimeBudgets
    from web_agent.runtime.recovery.controller import RecoveryController

    class FixtureAdapter(MiniWoBAdapter):
        def __init__(self, folder):
            self.hybrid_interface_version = 3
            self.folder = folder
            folder.mkdir()
            self.lock = threading.Lock()
            self.i = 0
            self.last = None
            self.current = None
            self.reject_next_execution = True
            self.actual_action_dispatches = 0
            self.retry_worker_requests = 0
            self.log = (folder / 'browser.log').open('x')
            self.proc = subprocess.Popen(
                [browser_python, str(Path(__file__).resolve()), '--worker', str(folder)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                env={**os.environ, 'PLAYWRIGHT_BROWSERS_PATH': browser_binaries,
                     'PYTHONDONTWRITEBYTECODE': '1'})

        def rpc(self, op, **kwargs):
            if op == 'execute' and kwargs.get('expected_control') is not None:
                if self.reject_next_execution:
                    self.reject_next_execution = False
                    return {'action_error': 'CONTROLLED_PRE_DISPATCH_REJECTION', 'rejected': True}
                self.retry_worker_requests += 1
                response = super().rpc(op, **kwargs)
                if not response['rejected']:
                    self.actual_action_dispatches += 1
                return response
            return super().rpc(op, **kwargs)

    # Mutations are authored fixture setup between the failed request and its
    # current observation; none is a recovery action or a task performance claim.
    mutations = {
        'unchanged': None,
        'changed_before_dispatch': None,
        'changed': "page.locator('#person').evaluate(\"e => e.name = 'changed'\")",
        'removed': "page.locator('#person').evaluate('e => e.remove()')",
        'replaced_source': "page.locator('#person').evaluate(\"e => {const n=e.cloneNode(); n.id='replacement'; e.replaceWith(n)}\")",
        'occluded': "page.evaluate(\"() => {const e=document.createElement('div'); e.style='position:fixed;inset:0;background:white;z-index:999'; document.body.append(e)}\")",
        'disabled': "page.locator('#person').evaluate('e => e.disabled=true')",
        'duplicate_source': "page.locator('#person').evaluate('e => e.after(e.cloneNode())')",
        'indistinguishable_replacement': "page.locator('#person').evaluate('e => e.replaceWith(e.cloneNode())')",
    }
    results = []
    task = TaskSpecification(
        task_id='miniwob.fixture-stable-retry-v3', goal=FIXTURE_GOAL,
        benchmark_id='miniwob', benchmark_version='0.14.3',
        start_state_id='authored-local-html', development_partition=True)
    for case, mutation in mutations.items():
        adapter = FixtureAdapter(output / case)
        executor = Executor(adapter, budgets=RuntimeBudgets())
        try:
            before = executor.reset(task, episode_id='engineering-' + case, seed=0)
            view = ObservationBuilder().pre_action(task, before)
            selected = next(c for c in view.current_page_state['visible_controls'] if c['source_id'] == 'person')
            decision = PreActionDecision(
                decision_id='issued:' + case, observation_id=view.observation_id,
                action_type=ActionType.TYPE,
                action_probabilities=probability_map([a.value for a in ActionType], 'TYPE'),
                bbox=tuple(selected['target_bbox']), grounding_confidence=0.0, confidence_before=0.0,
                input_observation_ids=(view.observation_id,), policy_id='authored-engineering', policy_version='v3',
                parameter_hints={'target': selected['control_id'], 'text': ISSUED_VALUE})
            parameters = DeterministicParameterProvider().resolve(task, view, decision, rng=random.Random(0))
            failed = concretize(action_id='failed:' + case, decision=decision, parameters=parameters)
            original_hash = failed.record_sha256
            failure = executor.execute(failed)
            assert failure.status is ExecutionStatus.REJECTED and not failure.state_changed
            assert adapter.actual_action_dispatches == 0
            if mutation:
                response = adapter.rpc('execute', code=mutation, expected_control=None)
                assert response == {'action_error': '', 'rejected': False}
            current = executor.observe_after(failed)
            current_view = ObservationBuilder().pre_action(task, current)
            controller = RecoveryController(RuntimeBudgets(), stable_target_identity=True)
            plan = controller.begin_attempt(
                RecoveryDecision(decision_id='retry:' + case, incident_id='incident:' + case,
                                 strategy=RecoveryStrategy.RETRY, trigger_sources=('executor',),
                                 diagnosis='CONTROLLED_PRE_DISPATCH_REJECTION'),
                failed, task=task, post_failure_observation=current_view)
            expected_ready = case in {'unchanged', 'indistinguishable_replacement', 'changed_before_dispatch'}
            assert plan.resolution_status == ('READY' if expected_ready else 'REJECTED'), plan.rejection_reason
            value_preserved = False
            retry_execution_status = None
            retry_execution_message = ''
            if expected_ready:
                issued = plan.actions[0]
                assert ConcreteAction.from_dict(json.loads(issued.to_json())).record_sha256 == issued.record_sha256
                assert issued.action_type is ActionType.TYPE and issued.parameters['text'] == ISSUED_VALUE
                assert issued.parameters['target_control_id'] != failed.parameters['target_control_id']
                assert issued.bbox == failed.bbox
                if case == 'changed_before_dispatch':
                    adapter.rpc('execute', code="page.locator('#person').evaluate(\"e => e.value='concurrent edit'\")",
                                expected_control=None)
                execution = executor.execute(issued)
                retry_execution_status = execution.status.value
                retry_execution_message = execution.message
                if case == 'changed_before_dispatch':
                    assert execution.status is ExecutionStatus.REJECTED
                    assert 'STABLE_TARGET_CHANGED_BEFORE_EXECUTION' in execution.message
                    assert adapter.actual_action_dispatches == 0
                else:
                    assert execution.status is ExecutionStatus.EXECUTED, execution.message
                final = executor.observe_after(issued, recovery=True)
                controls = final.page_state['visible_controls']
                person = next(c for c in controls if c['source_id'] == 'person')
                other = next(c for c in controls if c['source_id'] == 'other')
                value_preserved = person['value'] == ISSUED_VALUE and other['value'] == ''
                if case == 'changed_before_dispatch':
                    assert person['value'] == 'concurrent edit' and other['value'] == ''
                else:
                    assert value_preserved
                logged = _payload(issued)
                assert logged['parameters']['text']['sha256'] == hashlib.sha256(ISSUED_VALUE.encode()).hexdigest()
                assert ISSUED_VALUE not in json.dumps(logged)
                (adapter.folder / 'issued-action-redacted.json').write_text(json.dumps(logged, indent=2) + '\n')
            assert failed.record_sha256 == original_hash
            results.append({'case': case, 'retry_status': plan.resolution_status,
                            'rejection_reason': plan.rejection_reason,
                            'retry_execution_status': retry_execution_status,
                            'retry_execution_message': retry_execution_message,
                            'recovery_attempts': controller.episode_attempts,
                            'executor_requests': executor.steps_used,
                            'initial_request_browser_dispatches': 0,
                            'retry_browser_dispatches': adapter.actual_action_dispatches,
                            'retry_worker_requests': adapter.retry_worker_requests,
                            'exact_value_in_selected_control_and_decoy_unchanged': value_preserved,
                            'fixture_mutations': int(mutation is not None or case == 'changed_before_dispatch')})
        finally:
            executor.close()
    report = {
        'schema': 'hybrid-stable-retry-v3-browser-engineering-v1',
        'scope': 'Authored local HTML, real Playwright, production adapter/worker/executor; no task performance evidence.',
        'actual_model_inferences': 0,
        'actual_browser_recovery_actions': sum(r['retry_browser_dispatches'] for r in results),
        'identity_limit': 'Replacing a DOM node with an indistinguishable clone that reuses source_id and every exported property is observationally indistinguishable; it remains eligible. No DOM-instance identity is claimed.',
        'cases': results,
    }
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--browser-python', default='/home/aiub/kiyas/table2-envs/miniwob-feasibility/bin/python')
    parser.add_argument('--browser-binaries', default='/home/aiub/kiyas/table2-inputs/miniwob-browsers')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    output = args.output
    if output is None:
        output = Path(tempfile.mkdtemp(prefix='hybrid-v3-stable-browser-'))
    else:
        output.mkdir(parents=True, exist_ok=False)
    report = check(args.browser_python, args.browser_binaries, output)
    print(json.dumps({'output': str(output), 'cases': len(report['cases']),
                      'actual_browser_recovery_actions': report['actual_browser_recovery_actions'],
                      'actual_model_inferences': 0}))


if __name__ == '__main__':
    main()
