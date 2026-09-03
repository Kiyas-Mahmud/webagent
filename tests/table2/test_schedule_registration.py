from __future__ import annotations

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.schedule import (
    REGISTERED_MATCHED_MODEL_SEEDS,
    build_paired_schedule,
    validate_schedule,
)


def test_table2_schedule_registers_seed_42_only():
    schedule = build_paired_schedule([{"task_id": "public-dev-0"}])

    assert REGISTERED_MATCHED_MODEL_SEEDS == (42,)
    assert len(schedule) == 1
    assert schedule[0]["matched_model_seed"] == 42


@pytest.mark.parametrize("unregistered", [(43,), (42, 43, 44), (), (True,)])
def test_table2_schedule_rejects_unregistered_model_seeds(unregistered):
    with pytest.raises(ValueError, match="seed 42 only"):
        build_paired_schedule(
            [{"task_id": "public-dev-0"}], model_seeds=unregistered
        )


def test_schedule_validation_rejects_a_seed_43_row():
    row = build_paired_schedule([{"task_id": "public-dev-0"}])[0]
    row["matched_model_seed"] = 43

    with pytest.raises(SchemaError, match="seed 42 only"):
        validate_schedule([row])
