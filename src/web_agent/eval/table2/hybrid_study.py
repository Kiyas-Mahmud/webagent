"""Thin hybrid profile bindings; model loaders, executor and recovery are shared."""
import json
from pathlib import Path

from web_agent.runtime.contracts import HybridSystemID


def frozen_scope(config):
    from web_agent.eval.table2.miniwob_study import sha, runtime_protocol
    from web_agent.runtime.protocol import RuntimeStage
    path = Path(config['development_scope'])
    assert sha(path) == config['development_scope_sha256'], 'hybrid scope changed'
    scope = json.loads(path.read_text())
    assert config['development_only'] and config['profile'] == scope['profile']
    assert config['systems'] == list(scope['systems']) == [s.value for s in HybridSystemID]
    assert config['development_tasks'] == [b['task'] for b in scope['blocks']]
    assert config['model_seed'] == scope['model_seed'] == 42
    assert config['development_campaign_id'] == scope['seed_namespace']
    assert not config['development_revision'].get('executable_action_selection', False)
    assert config['development_revision']['hybrid_continuation'] is True
    assert config['development_revision']['selective_memory_context'] is True
    assert config['store_manifest_sha256'] == scope['memory_manifest_sha256']
    assert config['memory_material_sha256'] == scope['memory_material_sha256']
    if config.get('hybrid_interface_version') == 3:
        assert scope['hybrid_interface_version'] == 3
        assert config['development_revision']['hybrid_proposal_repair'] is True
        assert config['development_revision']['stable_retry_targets'] is True
        assert scope['memory_applicability_policy'] == 'strict selective context unchanged from v2'
    protocol = runtime_protocol(scope['seed_namespace'], hybrid=True)
    assert protocol.budgets.to_dict() == scope['budgets']
    rng = protocol.rng_factory()
    for block in scope['blocks']:
        seed = rng.seed_for_key(rng.key(task_id='miniwob.' + block['task'],
            repeat_id=block['repeat_id'], matched_seed=42, stage=RuntimeStage.RESET, decision_index=0))
        assert seed == block['stage_reset_seed'] and seed % 2**32 == block['browser_reset_seed']
        assert sha(block['task_html_path']) == block['task_html_sha256']
    return scope


def build_hybrid_components(*, system, episode_id, folder, bundle, manifest, config):
    """Single production factory also exercised by zero-inference integration checks."""
    from web_agent.runtime.hybrid_action_policy import HybridActionPolicy
    from web_agent.runtime.policy import CallablePolicyAdapter, PolicyKind
    from web_agent.memory.hybrid_runtime import build_hybrid_memory
    system = HybridSystemID(system)
    trained = None
    if system is not HybridSystemID.H0:
        trained = CallablePolicyAdapter(policy_id='validation-selected-web-agent', policy_version='v1',
            kind=PolicyKind.TRAINED, checkpoint_sha256=manifest['checkpoint_sha256'],
            action_predictor=bundle.selected.action_predictor,
            transition_predictor=bundle.selected.transition_predictor,
            recovery_predictor=bundle.selected.recovery_predictor)
    policy = HybridActionPolicy(system=system.value, episode_id=episode_id,
        base=bundle.e0.action_predictor.__self__, evidence_dir=folder/'hybrid-normal-outputs',
        trained_policy=trained, interface_version=config.get('hybrid_interface_version', 1))
    # The factory returns before accessing runtime/files for H0–H2.
    memory = build_hybrid_memory(system=system.value, episode_id=episode_id,
        runtime=bundle.selected.action_predictor.__self__ if system is HybridSystemID.H3 else None,
        store_directory=config['asset_paths']['memory'], material_path=config['memory_material'],
        evidence_dir=folder/'hybrid-memory-outputs',
        expected_manifest_sha256=config['store_manifest_sha256'],
        expected_material_sha256=config['memory_material_sha256'])
    return policy, memory
