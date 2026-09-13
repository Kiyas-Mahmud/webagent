"""Necessary conditions for advisory context, never an action selector."""
from web_agent.benchmarks.miniwob_controls import supported_actions

VERSION = 'observable-memory-applicability-v1'


def context_exclusion(example, transition):
    post = transition['post_observation']
    history = post.get('causal_history', ())
    strategy = example['recorded_strategy']
    if strategy == 'BACKTRACK' and not any(
        h['action_type'] == 'NAVIGATE' and h['execution_status'] == 'executed'
        and h['state_changed'] and not h['environment_error'] for h in history
    ):
        return 'BACKTRACK_NO_EXECUTED_IN_EPISODE_NAVIGATION'
    if strategy in {'RETRY', 'ALTERNATIVE_TARGET'} and (
        transition['execution_result']['status'] == 'rejected'
        and transition['execution_result']['error_kind'] == 'parameter_resolution_rejected'
        and example['recovery_action'] == transition['executed_action']['action_type']
    ):
        return 'SAME_UNRESOLVED_ACTION'
    kind = example['recovery_action']
    if kind in {'TYPE', 'SELECT', 'NAVIGATE', 'PRESS_KEY', 'SCROLL'} and not example['recovery_action_value']:
        return 'CORRECTIVE_ARGUMENT_UNAVAILABLE'
    controls = post.get('current_page_state', {}).get('visible_controls', ())
    if kind in {'CLICK', 'TYPE', 'SELECT'} and not any(kind in supported_actions(c) for c in controls):
        return 'NO_CURRENT_COMPATIBLE_CONTROL'
    if kind == 'NAVIGATE' and not any(c.get('destination') == example['recovery_action_value'] for c in controls):
        return 'DESTINATION_NOT_CURRENTLY_OBSERVED'
    return None
