from __future__ import annotations

import random

import pytest

from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    PolicyObservation,
    RecoveryDecision,
    RecoveryStrategy,
    TaskSpecification,
)
from web_agent.runtime.protocol import REGISTERED_BUDGETS
from web_agent.runtime.recovery.controller import (
    CallableRecoveryActionPlanner,
    RecoveryBudgetExceeded,
    RecoveryController,
)
from web_agent.runtime.recovery.strategies import (
    RECOVERY_TARGET_EVIDENCE_KEY,
    build_recovery_target_evidence,
)


SHA = "a" * 64


def _task() -> TaskSpecification:
    return TaskSpecification(
        task_id="task-1",
        goal="interact with the visible target",
        benchmark_id="fixture",
        benchmark_version="v1",
        start_state_id="s0",
    )


def _post_failure_observation(
    *,
    task_id: str = "task-1",
    goal: str = "interact with the visible target",
    current_page_state=None,
    compatible_actions: tuple[ConcreteAction, ...] | None = None,
) -> PolicyObservation:
    observation_id = "post-failure-observation"
    if compatible_actions is None:
        planned = _decision(RecoveryStrategy.REPLAN).planned_action
        assert planned is not None
        compatible_actions = (_failed_action(), planned)
    page_state = dict(current_page_state or {})
    page_state[RECOVERY_TARGET_EVIDENCE_KEY] = build_recovery_target_evidence(
        task=TaskSpecification(
            task_id=task_id,
            goal=goal,
            benchmark_id="fixture",
            benchmark_version="v1",
            start_state_id="s0",
        ),
        observation_id=observation_id,
        compatible_actions=compatible_actions,
    )
    return PolicyObservation(
        task_id=task_id,
        goal=goal,
        observation_id=observation_id,
        screenshot_sha256=SHA,
        screenshot_path=None,
        width=1280,
        height=720,
        url="fixture://post-failure",
        title="post failure",
        current_page_state=page_state,
    )


def _failed_action() -> ConcreteAction:
    return ConcreteAction(
        action_id="failed-action",
        source_decision_id="policy-decision",
        action_type=ActionType.CLICK,
        parameters={
            "target_x": 0.2,
            "target_y": 0.2,
            "target_bbox": [0.1, 0.1, 0.2, 0.2],
            "button": "left",
            "click_count": 1,
        },
        bbox=(0.1, 0.1, 0.2, 0.2),
    )


def _select_action(
    *,
    action_id: str,
    bbox: tuple[float, float, float, float],
    option: str,
    candidate_options: list[str],
) -> ConcreteAction:
    x, y, width, height = bbox
    return ConcreteAction(
        action_id=action_id,
        source_decision_id="policy-decision",
        action_type=ActionType.SELECT,
        parameters={
            "target_x": x + width / 2.0,
            "target_y": y + height / 2.0,
            "target_bbox": list(bbox),
            "option": option,
            "candidate_options": list(candidate_options),
        },
        bbox=bbox,
    )


def _observable_select_control(
    bbox: tuple[float, float, float, float],
    candidate_options: list[str],
):
    return {
        "observable_select_controls": [
            {
                "target_bbox": list(bbox),
                "candidate_options": list(candidate_options),
            }
        ]
    }


def _decision(strategy: RecoveryStrategy, incident: str = "incident-1") -> RecoveryDecision:
    planned = None
    if strategy in {RecoveryStrategy.REPLAN, RecoveryStrategy.ALTERNATIVE_TARGET}:
        planned = ConcreteAction(
            action_id="planned",
            source_decision_id="policy-decision",
            action_type=ActionType.CLICK,
            parameters={
                "target_x": 0.8,
                "target_y": 0.8,
                "target_bbox": [0.7, 0.7, 0.2, 0.2],
                "button": "left",
                "click_count": 1,
            },
            bbox=(0.7, 0.7, 0.2, 0.2),
        )
    return RecoveryDecision(
        decision_id=f"decision-{incident}-{strategy.value}",
        incident_id=incident,
        strategy=strategy,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
        planned_action=planned,
    )


@pytest.mark.parametrize(
    "strategy",
    [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.REPLAN,
        RecoveryStrategy.BACKTRACK,
        RecoveryStrategy.ALTERNATIVE_TARGET,
    ],
)
def test_every_executable_recovery_strategy_is_bounded_to_one_action(strategy):
    controller = RecoveryController(REGISTERED_BUDGETS)
    plan = controller.begin_attempt(
        _decision(strategy),
        _failed_action(),
        task=_task(),
        post_failure_observation=_post_failure_observation(),
    )
    assert plan.resolution_status == "READY"
    assert len(plan.actions) == 1
    assert controller.episode_attempts == 1


def test_abort_is_one_attempt_zero_browser_actions_and_never_a_success_label():
    controller = RecoveryController(REGISTERED_BUDGETS)
    plan = controller.begin_attempt(_decision(RecoveryStrategy.ABORT), _failed_action())
    attempt = controller.finish_attempt(plan)
    assert controller.episode_attempts == 1
    assert plan.resolution_status == "ABORT"
    assert plan.actions == ()
    assert attempt.action_ids == ()
    assert not hasattr(attempt, "verified_success")


def test_resolve_exposes_the_registered_recovery_entrypoint():
    controller = RecoveryController(REGISTERED_BUDGETS)
    plan = controller.resolve(
        _decision(RecoveryStrategy.RETRY),
        _failed_action(),
        task=_task(),
        post_failure_observation=_post_failure_observation(),
    )

    assert plan.resolution_status == "READY"
    assert len(plan.actions) == 1
    assert controller.episode_attempts == 1


def test_recovery_controller_enforces_incident_and_episode_limits():
    controller = RecoveryController(REGISTERED_BUDGETS)
    failed = _failed_action()
    current = _post_failure_observation()
    task = _task()
    controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "i1"),
        failed,
        task=task,
        post_failure_observation=current,
    )
    controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "i1"),
        failed,
        task=task,
        post_failure_observation=current,
    )
    with pytest.raises(RecoveryBudgetExceeded):
        controller.begin_attempt(
            _decision(RecoveryStrategy.RETRY, "i1"),
            failed,
            task=task,
            post_failure_observation=current,
        )

    controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "i2"),
        failed,
        task=task,
        post_failure_observation=current,
    )
    controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "i3"),
        failed,
        task=task,
        post_failure_observation=current,
    )
    with pytest.raises(RecoveryBudgetExceeded):
        controller.begin_attempt(
            _decision(RecoveryStrategy.RETRY, "i4"),
            failed,
            task=task,
            post_failure_observation=current,
        )
    assert controller.episode_attempts == 4


def test_retry_without_current_oracle_blind_inputs_fails_closed_and_consumes_attempt():
    controller = RecoveryController(REGISTERED_BUDGETS)

    plan = controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY),
        _failed_action(),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "current task and post-failure observation" in plan.rejection_reason
    assert controller.episode_attempts == 1


def test_retry_rejects_syntactically_valid_target_absent_from_current_state():
    failed = _failed_action()
    controller = RecoveryController(REGISTERED_BUDGETS)

    plan = controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY),
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            compatible_actions=(),
        ),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "not visibly present" in plan.rejection_reason


def test_retry_accepts_syntactically_valid_target_registered_visible_and_compatible():
    failed = _failed_action()
    plan = RecoveryController(REGISTERED_BUDGETS).begin_attempt(
        _decision(RecoveryStrategy.RETRY),
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            compatible_actions=(failed,),
        ),
    )

    assert plan.resolution_status == "READY"
    assert len(plan.actions) == 1


def test_retry_rejects_foreign_or_oracle_bearing_post_failure_observation():
    controller = RecoveryController(REGISTERED_BUDGETS)
    foreign = controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "foreign"),
        _failed_action(),
        task=_task(),
        post_failure_observation=_post_failure_observation(task_id="another-task"),
    )
    leaking = controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "leaking"),
        _failed_action(),
        task=_task(),
        post_failure_observation=_post_failure_observation(
            current_page_state={"oracle_success": True}
        ),
    )

    assert foreign.resolution_status == "REJECTED"
    assert leaking.resolution_status == "REJECTED"
    assert foreign.actions == leaking.actions == ()


def test_alternative_target_rejects_same_semantic_target_with_other_click_count():
    failed = _failed_action()
    same_target = ConcreteAction(
        action_id="planned-same-target",
        source_decision_id="policy-decision",
        action_type=ActionType.CLICK,
        parameters={**failed.parameters, "click_count": 2},
        bbox=failed.bbox,
    )
    decision = RecoveryDecision(
        decision_id="alternative-same-target",
        incident_id="incident-same-target",
        strategy=RecoveryStrategy.ALTERNATIVE_TARGET,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
        planned_action=same_target,
    )

    plan = RecoveryController(REGISTERED_BUDGETS).begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "failed semantic target" in plan.rejection_reason


def test_alternative_target_planner_rejects_same_target_with_changed_parameters():
    failed = _failed_action()

    def same_target_planner(task, observation, decision, failed_action, rng):
        del task, observation, decision, rng
        return ConcreteAction(
            action_id="planner-same-target",
            source_decision_id="policy-decision",
            action_type=ActionType.CLICK,
            parameters={**failed_action.parameters, "click_count": 3},
            bbox=failed_action.bbox,
        )

    controller = RecoveryController(
        REGISTERED_BUDGETS,
        action_planner=CallableRecoveryActionPlanner(
            planner_id="frozen-planner",
            planner_version="v1",
            callback=same_target_planner,
        ),
    )
    decision = RecoveryDecision(
        decision_id="planned-alternative",
        incident_id="planner-incident",
        strategy=RecoveryStrategy.ALTERNATIVE_TARGET,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
    )

    plan = controller.begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(),
        rng=random.Random(7),
    )
    assert plan.resolution_status == "REJECTED"
    assert "failed semantic target" in plan.rejection_reason
    assert controller.episode_attempts == 1


def test_alternative_target_rejects_different_but_unregistered_target():
    failed = _failed_action()
    decision = _decision(RecoveryStrategy.ALTERNATIVE_TARGET, "unregistered")
    plan = RecoveryController(REGISTERED_BUDGETS).begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            compatible_actions=(failed,),
        ),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "not visibly present" in plan.rejection_reason


def test_replan_rejects_visible_target_not_compatible_with_planned_action_type():
    failed = _failed_action()
    planned = _decision(RecoveryStrategy.REPLAN, "incompatible").planned_action
    assert planned is not None
    same_target_wrong_type = ConcreteAction(
        action_id="same-target-type-only",
        source_decision_id="policy-decision",
        action_type=ActionType.TYPE,
        parameters={
            "target_x": planned.parameters["target_x"],
            "target_y": planned.parameters["target_y"],
            "target_bbox": planned.parameters["target_bbox"],
            "text": "visible text",
        },
        bbox=planned.bbox,
    )
    decision = RecoveryDecision(
        decision_id="replan-incompatible-action-type",
        incident_id="replan-incompatible-action-type",
        strategy=RecoveryStrategy.REPLAN,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
        planned_action=planned,
    )

    plan = RecoveryController(REGISTERED_BUDGETS).begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            compatible_actions=(same_target_wrong_type,),
        ),
    )

    assert plan.resolution_status == "REJECTED"
    assert "not registered task-compatible" in plan.rejection_reason


def test_replan_rejects_invalid_planned_parameters_and_consumes_attempt():
    invalid_action = ConcreteAction(
        action_id="invalid-replan-action",
        source_decision_id="policy-decision",
        action_type=ActionType.CLICK,
        parameters={"target_x": 0.8, "target_y": 0.8},
        bbox=(0.7, 0.7, 0.2, 0.2),
    )
    decision = RecoveryDecision(
        decision_id="invalid-replan",
        incident_id="invalid-replan-incident",
        strategy=RecoveryStrategy.REPLAN,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
        planned_action=invalid_action,
    )
    controller = RecoveryController(REGISTERED_BUDGETS)

    plan = controller.begin_attempt(
        decision,
        _failed_action(),
        task=_task(),
        post_failure_observation=_post_failure_observation(),
    )

    assert plan.resolution_status == "REJECTED"
    assert "registered parameter validation" in plan.rejection_reason
    assert controller.episode_attempts == 1


@pytest.mark.parametrize(
    ("option", "provider_candidates", "observed_candidates"),
    [
        (
            "visible-option",
            ["visible-option", "provider-only"],
            ["visible-option", "observed-only"],
        ),
        (
            "hallucinated-option",
            ["visible-option", "hallucinated-option"],
            ["visible-option"],
        ),
    ],
)
def test_retry_select_rejects_candidates_or_choice_absent_from_current_control(
    option,
    provider_candidates,
    observed_candidates,
):
    bbox = (0.1, 0.1, 0.2, 0.2)
    failed = _select_action(
        action_id="failed-select",
        bbox=bbox,
        option=option,
        candidate_options=provider_candidates,
    )
    controller = RecoveryController(REGISTERED_BUDGETS)

    plan = controller.begin_attempt(
        _decision(RecoveryStrategy.RETRY, "retry-select-evidence"),
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            current_page_state=_observable_select_control(
                bbox,
                observed_candidates,
            ),
            compatible_actions=(failed,),
        ),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "grounded observable control" in plan.rejection_reason
    assert controller.episode_attempts == 1


@pytest.mark.parametrize(
    "strategy",
    [RecoveryStrategy.REPLAN, RecoveryStrategy.ALTERNATIVE_TARGET],
)
def test_embedded_planned_select_rejects_unobserved_candidate_evidence(strategy):
    failed = _failed_action()
    planned_bbox = (0.7, 0.7, 0.2, 0.2)
    planned = _select_action(
        action_id="planned-select",
        bbox=planned_bbox,
        option="planner-only",
        candidate_options=["planner-only"],
    )
    decision = RecoveryDecision(
        decision_id=f"{strategy.value.lower()}-select-evidence",
        incident_id=f"{strategy.value.lower()}-select-evidence",
        strategy=strategy,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
        planned_action=planned,
    )
    controller = RecoveryController(REGISTERED_BUDGETS)

    plan = controller.begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            current_page_state=_observable_select_control(
                planned_bbox,
                ["visible-option"],
            ),
            compatible_actions=(failed, planned),
        ),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "grounded observable control" in plan.rejection_reason
    assert controller.episode_attempts == 1


@pytest.mark.parametrize(
    "strategy",
    [RecoveryStrategy.REPLAN, RecoveryStrategy.ALTERNATIVE_TARGET],
)
def test_callable_planner_select_rejects_unobserved_candidate_evidence(strategy):
    failed = _failed_action()
    planned_bbox = (0.7, 0.7, 0.2, 0.2)
    planned = _select_action(
        action_id="planner-select",
        bbox=planned_bbox,
        option="planner-only",
        candidate_options=["planner-only"],
    )

    def planner(task, observation, decision, failed_action, rng):
        del task, observation, decision, failed_action, rng
        return planned

    controller = RecoveryController(
        REGISTERED_BUDGETS,
        action_planner=CallableRecoveryActionPlanner(
            planner_id="select-evidence-planner",
            planner_version="v1",
            callback=planner,
        ),
    )
    decision = RecoveryDecision(
        decision_id=f"planner-{strategy.value.lower()}-select-evidence",
        incident_id=f"planner-{strategy.value.lower()}-select-evidence",
        strategy=strategy,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
    )

    plan = controller.begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            current_page_state=_observable_select_control(
                planned_bbox,
                ["visible-option"],
            ),
            compatible_actions=(failed, planned),
        ),
        rng=random.Random(11),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "grounded observable control" in plan.rejection_reason
    assert controller.episode_attempts == 1


@pytest.mark.parametrize(
    "strategy",
    [
        RecoveryStrategy.RETRY,
        RecoveryStrategy.REPLAN,
        RecoveryStrategy.ALTERNATIVE_TARGET,
    ],
)
def test_recovery_select_accepts_exact_current_control_evidence(strategy):
    failed = _select_action(
        action_id="failed-select",
        bbox=(0.1, 0.1, 0.2, 0.2),
        option="visible-option",
        candidate_options=["visible-option", "other-option"],
    )
    planned = None
    active = failed
    if strategy in {RecoveryStrategy.REPLAN, RecoveryStrategy.ALTERNATIVE_TARGET}:
        active = _select_action(
            action_id="planned-select",
            bbox=(0.7, 0.7, 0.2, 0.2),
            option="visible-option",
            candidate_options=["visible-option", "other-option"],
        )
        planned = active
    decision = RecoveryDecision(
        decision_id=f"valid-{strategy.value.lower()}-select",
        incident_id=f"valid-{strategy.value.lower()}-select",
        strategy=strategy,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
        planned_action=planned,
    )

    plan = RecoveryController(REGISTERED_BUDGETS).begin_attempt(
        decision,
        failed,
        task=_task(),
        post_failure_observation=_post_failure_observation(
            current_page_state=_observable_select_control(
                active.bbox,
                ["visible-option", "other-option"],
            ),
            compatible_actions=(failed, active),
        ),
    )

    assert plan.resolution_status == "READY"
    assert len(plan.actions) == 1


def test_unexpected_planner_failure_is_a_logged_rejectable_attempt_contract():
    secret = "PRIVATE_PAGE_TEXT_MUST_NOT_ENTER_LOGS"

    def planner(task, observation, decision, failed_action, rng):
        del task, observation, decision, failed_action, rng
        raise RuntimeError(secret)

    controller = RecoveryController(
        REGISTERED_BUDGETS,
        action_planner=CallableRecoveryActionPlanner(
            planner_id="failing-planner",
            planner_version="v1",
            callback=planner,
        ),
    )
    decision = RecoveryDecision(
        decision_id="failing-planner-decision",
        incident_id="failing-planner-incident",
        strategy=RecoveryStrategy.REPLAN,
        trigger_sources=("policy",),
        diagnosis="NO_EFFECT",
    )

    plan = controller.begin_attempt(
        decision,
        _failed_action(),
        task=_task(),
        post_failure_observation=_post_failure_observation(),
        rng=random.Random(17),
    )

    assert plan.resolution_status == "REJECTED"
    assert plan.actions == ()
    assert "RuntimeError" in plan.rejection_reason
    assert "error_sha256=" in plan.rejection_reason
    assert secret not in plan.rejection_reason
    assert controller.episode_attempts == 1
    assert controller.incident_attempts(decision.incident_id) == 1
    attempt = controller.finish_attempt(plan)
    assert attempt.completed is False
    assert attempt.action_ids == ()
