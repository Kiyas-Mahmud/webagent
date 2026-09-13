"""CPU scripted callbacks through the real H actor, episode and recovery loop.

These exercise contracts/accounting only; there is no model or GPU inference.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from web_agent.benchmarks.base import AdapterExecution, BenchmarkAdapter
from web_agent.benchmarks.miniwob_controls import compatible_actions
from web_agent.runtime.action_parameters import build_registered_hybrid_parameter_provider
from web_agent.runtime.contracts import (
    ActionType, ConcreteAction, ExecutionStatus, HybridSystemID, Observation,
    ObservationStage, OpaqueTerminalSignal, RecoveryAssessment, RecoveryStrategy,
    TaskSpecification, TerminalReason, TransitionAssessment, canonical_sha256,
)
from web_agent.runtime.episode import EpisodeRunner
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import Executor
from web_agent.runtime.hybrid_action_policy import HybridActionPolicy
from web_agent.runtime.policy import ActionParseError, CallablePolicyAdapter, PolicyKind, SystemPolicy
from web_agent.runtime.protocol import RuntimeProtocol, switches_for
from web_agent.runtime.recovery.controller import CallableRecoveryActionPlanner, RecoveryController
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence
from tests.table2.test_development_v3 import observation
from tests.table2.test_hybrid_agent import advice


def run_case(root, *, system='H2', responses=('not JSON',), enabled=True,
             repair_plan=False, terminal_after=1, learned_abort=False):
    calls = []; prompts = []; assessed = []; recovered = []; rejected = []; executed = []

    class Page(BenchmarkAdapter):
        benchmark_id = 'fixture'; benchmark_version = 'v1'

        def reset(self, task, *, episode_id, seed):
            self.task = task; self.episode_id = episode_id; self.capture = 0; self.value = ''
            return self.observe(stage=ObservationStage.RESET)

        def observe(self, *, stage, prior_action_id=None):
            self.capture += 1
            oid = f'{self.episode_id}:obs:{self.capture}'
            state = observation(self.value, oid=oid).current_page_state
            state['visible_controls'][0]['hit_point'] = [.25, .25]
            state['hybrid_interface_version'] = 3 if enabled else 2
            state['recovery_target_evidence'] = build_recovery_target_evidence(
                task=self.task, observation_id=oid,
                compatible_actions=compatible_actions(state['visible_controls']))
            return Observation(observation_id=oid, episode_id=self.episode_id,
                stage=stage, prior_action_id=prior_action_id,
                screenshot_sha256=canonical_sha256({'value': self.value}), screenshot_path=None,
                width=1000, height=600, url='http://fixture/', title='', page_state=state)

        def execute(self, action):
            executed.append(action)
            if action.action_type is ActionType.TYPE:
                self.value = action.parameters['text']
            return AdapterExecution(status=ExecutionStatus.EXECUTED, state_changed=True)

        def register_rejected_action(self, action, execution):
            rejected.append((action, execution))

        def terminal_signal(self, task, binding=None):
            return OpaqueTerminalSignal(event_id=f'signal:{self.capture}', token_sha256='f' * 64,
                terminate=terminal_after is not None and len(executed) >= terminal_after)

        def close(self):
            pass

    def generate(**kwargs):
        prompts.append(kwargs)
        calls.append('generation')
        raw = responses[min(len(prompts) - 1, len(responses) - 1)]
        return raw(kwargs['observation']) if callable(raw) else raw

    def predict(task, obs, rng):
        calls.append('advice')
        return replace(advice(obs), policy_id='fixture', decision_id='trained:' + str(len(calls)))

    def diagnose(task, transition, rng):
        calls.append('P1'); assessed.append(transition)
        strategy = RecoveryStrategy.ABORT if learned_abort else RecoveryStrategy.REPLAN
        return TransitionAssessment(assessment_id='p1:' + transition.executed_action.action_id,
            pre_observation_id=transition.pre_observation.observation_id,
            post_observation_id=transition.post_observation.observation_id,
            executed_action_id=transition.executed_action.action_id,
            predicted_failure=True, failure_probability=1., failure_type='ELEMENT_NOT_FOUND',
            failure_type_probabilities={'ELEMENT_NOT_FOUND': 1.}, needs_recovery=True,
            needs_recovery_probability=1., recovery_strategy=strategy,
            recovery_probabilities={strategy.value: 1.})

    def assess(task, transition, rng):
        calls.append('recovery_assessment'); recovered.append(transition)
        return RecoveryAssessment(assessment_id='assessment:' + transition.attempt_id,
            incident_id=transition.incident_id, attempt_id=transition.attempt_id,
            pre_recovery_observation_id=transition.pre_recovery_observation.observation_id,
            post_recovery_observation_id=transition.post_recovery_observation.observation_id,
            recovery_action_ids=tuple(action.action_id for action in transition.recovery_actions),
            predicted_failure_resolved=False, predicted_resolution_probability=0.,
            predicted_progress=False, predicted_progress_probability=0.)

    def plan(task, obs, decision, failed, rng):
        calls.append('planner')
        if not repair_plan:
            raise ActionParseError('scripted rejected recovery proposal')
        return ConcreteAction(action_id=decision.decision_id + ':planned',
            source_decision_id=decision.decision_id, action_type=ActionType.TYPE,
            bbox=(.1, .2, .3, .1), parameters={'target_x': .25, 'target_y': .25,
                'target_bbox': [.1, .2, .3, .1], 'text': 'Prepared By Model'})

    protocol = RuntimeProtocol(protocol_id='hybrid-failure-v3-fixture',
        campaign_id='hybrid-failure-v3-fixture', benchmark_id='fixture', benchmark_version='v1',
        provider_id='hybrid-parameter-provider', provider_version='v1')
    episode_id = f'{protocol.campaign_id}:{system}:task:repeat-0:seed-42'
    trained = CallablePolicyAdapter(policy_id='fixture', policy_version='v1',
        kind=PolicyKind.TRAINED, checkpoint_sha256='a' * 64,
        action_predictor=predict, transition_predictor=diagnose, recovery_predictor=assess)
    actor = HybridActionPolicy(system=system, episode_id=episode_id,
        base=SimpleNamespace(_generate=generate), evidence_dir=root / 'proposals',
        trained_policy=None if system == 'H0' else trained, interface_version=3 if enabled else 2)

    def fallback(*args, **kwargs):
        raise AssertionError('target rejections and complete proposals need no fallback')

    provider = build_registered_hybrid_parameter_provider(fallback_resolver=fallback,
        fallback_policy_id='unused', fallback_policy_version='v1', prompt_text='fixture',
        decoding_parameters={})
    controller = RecoveryController(protocol.budgets, stable_target_identity=enabled,
        action_planner=CallableRecoveryActionPlanner(planner_id='fixture',
            planner_version='v1', callback=plan)) if system == 'H2' else None
    runner = EpisodeRunner(protocol=protocol,
        system_policy=SystemPolicy(actor, switches_for(HybridSystemID(system))),
        provider=provider, executor=Executor(Page(), budgets=protocol.budgets),
        recovery_controller=controller, hybrid_continuation=True,
        hybrid_proposal_repair=enabled, stable_retry_targets=enabled,
        event_logs=EpisodeEventLogs(root / 'events', episode_id=episode_id, include_memory=False))
    task = TaskSpecification(task_id='task', goal='Use current visible controls',
        benchmark_id='fixture', benchmark_version='v1', start_state_id='reset')
    summary = runner.run(task, repeat_id=0, model_seed=42)
    rows = [json.loads(line) for path in (root / 'events').glob('*.jsonl')
            for line in path.read_text().splitlines()]
    return SimpleNamespace(summary=summary, calls=calls, prompts=prompts, assessed=assessed,
        recovered=recovered, rejected=rejected, executed=executed, rows=rows, runner=runner)


def target_error(kind='TYPE', value=' Exact  Case! ', target='missing field'):
    return json.dumps({'action_type': kind, 'target': target, 'bbox': None, 'value': value})


def valid_action(obs):
    return json.dumps({'action_type': 'TYPE',
        'target': obs.current_page_state['visible_controls'][0]['control_id'],
        'bbox': None, 'value': ' Model Chosen Preparation! '})


@pytest.mark.parametrize('system', ['H0', 'H1', 'H2'])
def test_repeated_malformed_json_gets_one_charged_correction_and_no_fake_action(tmp_path, system):
    result = run_case(tmp_path, system=system)
    summary = result.summary
    assert summary.terminal_reason is TerminalReason.LOOP
    assert summary.executor_steps == summary.normal_actions == 2
    assert summary.model_call_count == len(result.calls) == (2 if system == 'H0' else 4)
    assert not result.assessed and not result.executed and not result.rejected
    assert summary.failure_incidents == summary.recovery_attempts == 0
    assert result.prompts[0]['observation'].record_sha256 == result.prompts[1]['observation'].record_sha256
    first, second = [json.loads(p['suffix'].split(': ', 1)[1]) for p in result.prompts]
    assert first['proposal_feedback'] is None
    assert second['proposal_feedback']['stage'] == 'parsing'
    requests = [row['payload'] for row in result.rows if row['event_type'] == 'pre_action_parse_rejection']
    assert len(requests) == 2
    assert all(r['decision'] is None and r['action'] is None for r in requests)
    assert any(row['event_type'] == 'episode_contract_validation' for row in result.rows)


@pytest.mark.parametrize('responses', [('not JSON', '{}', '{"action_type":"CLICK"}'),
                                      (target_error(), 'not JSON', '{}')])
def test_changing_diagnostics_does_not_refresh_same_page_correction_budget(tmp_path, responses):
    result = run_case(tmp_path, responses=responses)
    assert result.summary.terminal_reason is TerminalReason.POLICY_ERROR
    assert len(result.prompts) == result.summary.normal_actions == 2
    assert result.summary.model_call_count == len(result.calls)
    assert result.summary.recovery_attempts <= 2
    assert not result.executed


@pytest.mark.parametrize('system', ['H0', 'H1', 'H2'])
def test_valid_correction_executes_exact_model_chosen_preparation(tmp_path, system):
    result = run_case(tmp_path, system=system, responses=('{}', valid_action))
    assert result.summary.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert result.summary.normal_actions == result.summary.executor_steps == 2
    assert result.summary.model_call_count == len(result.calls)
    assert len(result.executed) == 1
    assert result.executed[0].action_type is ActionType.TYPE
    assert result.executed[0].parameters['text'] == ' Model Chosen Preparation! '
    assert any(row['event_type'] == 'hybrid_proposal_repair' and
        row['payload']['outcome'] == 'corrected' for row in result.rows)


@pytest.mark.parametrize('kind,value', [('TYPE', ' Exact  Case! '), ('CLICK', ' Exact Label ')])
def test_known_target_failure_reaches_p1_with_original_action_and_actual_unchanged_page(tmp_path, kind, value):
    result = run_case(tmp_path, responses=(target_error(kind, value),))
    assert result.summary.terminal_reason is TerminalReason.LOOP
    assert result.summary.executor_steps == result.summary.normal_actions == 2
    assert result.summary.failure_incidents == 1
    assert result.summary.recovery_attempts == 2
    assert result.summary.model_call_count == len(result.calls) == 8
    assert len(result.assessed) == len(result.rejected) == 2
    assert not result.executed and not result.recovered
    for transition in result.assessed:
        assert transition.executed_action.action_type is ActionType(kind)
        assert transition.executed_action.parameters['target'] == 'missing field'
        assert transition.executed_action.parameters['value'] == value
        if kind == 'TYPE':
            assert transition.executed_action.parameters['text'] == value
        assert transition.executed_action.bbox is None
        assert transition.execution_result.status is ExecutionStatus.REJECTED
        assert transition.execution_result.evidence.controller_command is None
        assert transition.execution_result.state_changed is False
        assert transition.pre_observation.screenshot_sha256 == transition.post_observation.screenshot_sha256
        assert transition.pre_observation.observation_id != transition.post_observation.observation_id
        assert not transition.post_observation.current_page_state['visible_controls'][0]['value']
    raw = [json.loads(path.read_text()) for path in (tmp_path / 'hybrid-action-records').glob('*.json')]
    assert len(raw) == 2
    assert all(record['proposal_rejection']['proposed_action'] == json.loads(target_error(kind, value)) for record in raw)
    assert not any(row['event_type'] == 'pre_action_parse_rejection' for row in result.rows)


def test_known_rejection_can_replan_and_still_assesses_terminal_recovery(tmp_path):
    result = run_case(tmp_path, responses=(target_error(),), repair_plan=True)
    assert result.summary.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL
    assert result.summary.normal_actions == result.summary.recovery_actions == 1
    assert result.summary.recovery_attempts == len(result.recovered) == 1
    assert len(result.assessed) == 1
    assert result.executed[0].action_type is ActionType.TYPE
    assert result.executed[0].parameters['text'] == 'Prepared By Model'
    assert result.summary.model_call_count == len(result.calls) == 5
    assert not result.recovered[0].recovery_actions[0].action_id == result.rejected[0][0].action_id


def test_format_repair_can_succeed_while_target_is_still_rejected(tmp_path):
    result = run_case(tmp_path, responses=('{}', target_error()))
    assert result.summary.terminal_reason is TerminalReason.POLICY_ERROR
    assert result.summary.normal_actions == 2
    assert len(result.assessed) == len(result.rejected) == 1
    assert not result.executed and result.summary.recovery_attempts == 0
    assert any(row['event_type'] == 'hybrid_proposal_repair' and
        row['payload']['outcome'] == 'corrected' for row in result.rows)


def test_learned_abort_remains_terminal_after_known_target_rejection(tmp_path):
    result = run_case(tmp_path, responses=(target_error(),), learned_abort=True)
    assert result.summary.terminal_reason is TerminalReason.ABORT
    assert result.summary.normal_actions == len(result.assessed) == 1
    assert not result.executed


def test_default_off_preserves_historical_thirty_request_behavior(tmp_path):
    result = run_case(tmp_path, enabled=False)
    assert result.summary.terminal_reason is TerminalReason.ACTION_BUDGET_EXHAUSTED
    assert result.summary.normal_actions == result.summary.executor_steps == 30
    assert result.summary.model_call_count == len(result.calls) == 60
    assert not result.executed and not result.assessed
    assert not any(row['event_type'] == 'hybrid_proposal_repair' for row in result.rows)


def test_full_study_typed_rejections_reach_read_only_h3_memory_and_replay(tmp_path, monkeypatch):
    """Eight matched scripted episodes; the real frozen label store is read only."""
    from collections import Counter, defaultdict
    import hashlib
    from pathlib import Path

    from tests.table2.test_hybrid_runner import (
        CONFIG, ScriptedBase, ScriptedPage, ScriptedTrained, engineering_run,
    )
    from web_agent.memory.label_backed_store import LabelBackedMemoryStore
    from web_agent.memory.label_runtime import LabelBackedMemoryAdapter

    config = json.loads(CONFIG.read_text())
    store_root = Path(config['asset_paths']['memory'])
    manifest = json.loads((store_root / 'manifest.json').read_text())
    protected = [store_root / 'manifest.json', Path(config['memory_material'])]
    protected.extend(store_root / name for name in manifest['files'])
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    actual_calls = defaultdict(Counter)
    memory_calls = []
    registered = []
    raw_proposal = target_error('TYPE', ' Exact  Case! ', ' Missing Model Target ')

    def episode_for(obs):
        return obs.observation_id.rsplit(':obs:', 1)[0]

    def generate(self, *, prompt, task, observation, suffix):
        recovery = suffix.startswith('\ncausal_recovery_context:')
        actual_calls[episode_for(observation)]['recovery_generator' if recovery else 'normal_generator'] += 1
        return '{}' if recovery else raw_proposal

    original_action = ScriptedTrained.action
    original_diagnose = ScriptedTrained.diagnose
    original_init = ScriptedTrained.__init__

    def predict(self, task, obs, rng):
        actual_calls[episode_for(obs)]['advice'] += 1
        return original_action(self, task, obs, rng)

    def diagnose(self, task, transition, rng):
        actual_calls[episode_for(transition.post_observation)]['P1'] += 1
        assert transition.execution_result.status is ExecutionStatus.REJECTED
        assert transition.executed_action.parameters['target'] == ' Missing Model Target '
        assert transition.executed_action.parameters['text'] == ' Exact  Case! '
        assert transition.pre_observation.screenshot_sha256 == transition.post_observation.screenshot_sha256
        return original_diagnose(self, task, transition, rng)

    def init_trained(self, config):
        original_init(self, config)
        original_embedding = self.model.memory_embedding
        self.batch.transition = lambda task, transition: {'episode_id': episode_for(transition.post_observation)}

        def embedding(batch):
            actual_calls[batch['episode_id']]['memory_embedding'] += 1
            return original_embedding(batch)

        self.model.memory_embedding = embedding

    original_memory_apply = LabelBackedMemoryAdapter.apply_post_failure

    def apply_memory(self, **kwargs):
        assert type(self.store) is LabelBackedMemoryStore
        assert self.store.frozen and not self.store.write_enabled
        assert self.reader.frozen and not self.reader.write_enabled
        assert not self.store._vectors.flags.writeable
        memory_calls.append(kwargs['query'])
        return original_memory_apply(self, **kwargs)

    class RejectionOnlyPage(ScriptedPage):
        def reset(self, task, *, episode_id, seed):
            (self.folder / 'browser-outcomes.json').write_text('[]\n')
            return super().reset(task, episode_id=episode_id, seed=seed)

        def register_rejected_action(self, action, execution):
            registered.append((action, execution))
            return super().register_rejected_action(action, execution)

        def execute(self, action):
            raise AssertionError('rejected targets and malformed recovery never dispatch a browser command')

    monkeypatch.setattr(ScriptedBase, '_generate', generate)
    monkeypatch.setattr(ScriptedTrained, '__init__', init_trained)
    monkeypatch.setattr(ScriptedTrained, 'action', predict)
    monkeypatch.setattr(ScriptedTrained, 'diagnose', diagnose)
    monkeypatch.setattr(LabelBackedMemoryAdapter, 'apply_post_failure', apply_memory)
    root = tmp_path / 'full-study'
    _, rows, audit = engineering_run(root, monkeypatch, interface_version=3, page_class=RejectionOnlyPage)
    assert audit['status'] == 'PASS', audit['errors']
    assert len(rows) == 8 and {row['system'] for row in rows} == {'H0', 'H1', 'H2', 'H3'}
    assert len(memory_calls) == 4 and all(':H3:' in query.episode_id for query in memory_calls)
    assert len(registered) == 16
    for row in rows:
        assert 'error' not in row, row
        summary = row['summary']
        system = row['system']
        expected = Counter(normal_generator=2)
        if system != 'H0':
            expected.update(advice=2)
        if system in {'H2', 'H3'}:
            expected.update(P1=2, recovery_generator=2)
        if system == 'H3':
            expected.update(memory_embedding=2)
        assert actual_calls[summary['episode_id']] == expected
        assert summary['model_call_count'] == sum(expected.values())
        assert summary['normal_actions'] == summary['executor_steps'] == 2
        assert summary['recovery_actions'] == 0
        assert summary['recovery_attempts'] == (2 if system in {'H2', 'H3'} else 0)
        assert summary['memory_queries'] == (2 if system == 'H3' else 0)
        assert summary['terminal_reason'] == 'loop'
        assert not row['success'] and row['environment_outcomes'] == []
        folder = root / row['task'] / 'repeat-0' / system
        records = [json.loads(path.read_text()) for path in (folder / 'hybrid-action-records').glob('*.json')]
        assert len(records) == 2
        for record in records:
            assert record['proposal_rejection']['proposed_action'] == json.loads(raw_proposal)
            assert record['action']['action_type'] == 'TYPE'
            assert record['action']['parameters']['value'] == ' Exact  Case! '
            assert record['execution']['status'] == 'rejected'
            assert record['execution']['evidence']['controller_command'] is None
        runtime_rows = [json.loads(line) for path in (folder / 'runtime').glob('*.jsonl')
                        for line in path.read_text().splitlines()]
        assert not any(event['event_type'] == 'pre_action_parse_rejection' for event in runtime_rows)
        assert any(event['event_type'] == 'episode_contract_validation' for event in runtime_rows)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    assert after == before
