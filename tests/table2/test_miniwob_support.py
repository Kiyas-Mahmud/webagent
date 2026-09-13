"""Scoring and observable-control unit checks; not benchmark completion evidence."""
import pytest

from web_agent.benchmarks.miniwob_support import full_completion, raw_reward_outcome


@pytest.mark.parametrize('raw,terminated,truncated,expected', [
    (1.0, True, False, True), (.8, True, False, False),
    (0.0, True, False, False), (-1.0, True, False, False),
    (1.0, False, False, False), (1.0, True, True, False),
])
def test_full_completion_requires_untruncated_termination_and_full_raw_credit(raw, terminated, truncated, expected):
    outcome = raw_reward_outcome(reward=float(raw > 0), terminated=terminated,
                                 truncated=truncated, info={'task_info': {'RAW_REWARD_GLOBAL': raw}})
    assert full_completion([outcome]) is expected
    assert outcome['raw_reward'] == raw


@pytest.mark.parametrize('raw', [None, True, float('nan'), float('inf'), -2, 2])
def test_missing_or_invalid_raw_reward_never_falls_back_to_binary_reward(raw):
    with pytest.raises(ValueError):
        raw_reward_outcome(reward=1, terminated=True, truncated=False,
                           info={'task_info': {'RAW_REWARD_GLOBAL': raw}})
    with pytest.raises(ValueError):
        full_completion([{'reward': 1, 'raw_reward': raw, 'terminated': True, 'truncated': False}])


def test_empty_or_partial_outcomes_do_not_count_as_full_success():
    assert not full_completion([])
    assert not full_completion([{'raw_reward': .5, 'terminated': True, 'truncated': False}])
