from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from web_agent.benchmarks.base import BenchmarkUnavailableError
from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymEpisodeAbortReceipt,
    BrowserGymRuntimeConfiguration,
    BrowserGymTaskStateResetReceipt,
    BrowserGymWebArenaError,
    FrozenBrowserGymTaskStateResetter,
    OracleBlindWebArenaTask,
    PinnedBrowserGymAPI,
    _BrowserGymEpisodeCallbacks,
    _action_code,
    callable_source_sha256,
    load_pinned_browsergym_api,
    module_source_sha256,
)
from web_agent.eval.table2.sealed_page_broker import (
    create_one_way_sealed_page_broker,
)
from web_agent.eval.table2.webarena_preflight import PINNED_WEBARENA_PACKAGES
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ObservationStage,
    RuntimeGeolocation,
    RuntimeStartState,
    TaskSpecification,
    canonical_sha256,
)


SERVICE_URLS = {
    "WA_SHOPPING": "http://shopping.test",
    "WA_SHOPPING_ADMIN": "http://shopping-admin.test",
    "WA_REDDIT": "http://reddit.test",
    "WA_GITLAB": "http://gitlab.test",
    "WA_WIKIPEDIA": "http://wikipedia.test",
    "WA_MAP": "http://map.test",
    "WA_HOMEPAGE": "http://homepage.test",
}


class _FakeMouse:
    pass


class _FakeKeyboard:
    pass


class _FakeContext:
    def __init__(self) -> None:
        self.pages: list[_FakePage] = []

    def new_page(self):
        page = _FakePage(context=self)
        self.pages.append(page)
        return page


class _FakePage:
    def __init__(self, *, context: _FakeContext | None = None) -> None:
        self.context = context or _FakeContext()
        if self not in self.context.pages:
            self.context.pages.append(self)
        self.url = "about:blank"
        self.viewport_size = {"width": 1280, "height": 720}
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.waits: list[tuple[str, int]] = []

    def goto(self, url: str, **_: object) -> None:
        self.url = url

    def bring_to_front(self) -> None:
        return None

    def title(self) -> str:
        return "Causal page"

    def content(self) -> str:
        return f"<html><body>{self.url}</body></html>"

    def screenshot(self, *, type: str) -> bytes:
        assert type == "png"
        return b"\x89PNG\r\n\x1a\nfixture"

    def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        self.waits.append((state, timeout))

    def evaluate(self, source: str, *args: object):
        del source, args
        return {
            "visible_text": "Choose a project",
            "controls": [
                {
                    "tag": "select",
                    "role": "",
                    "input_type": "",
                    "name": "Project",
                    "text": "Alpha Beta",
                    "bbox": [0.1, 0.2, 0.3, 0.1],
                    "candidate_options": ["alpha", "beta"],
                    "destination": "",
                },
                {
                    "tag": "a",
                    "role": "",
                    "input_type": "",
                    "name": "Issues",
                    "text": "Issues",
                    "bbox": [0.5, 0.2, 0.2, 0.1],
                    "candidate_options": [],
                    "destination": "http://gitlab.test/issues",
                },
            ],
        }


class _FakeBrowserEnv:
    last_kwargs: dict | None = None
    instances: list["_FakeBrowserEnv"] = []

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs
        type(self).instances.append(self)
        self.kwargs = kwargs
        self.page = _FakePage()
        self.close_calls = 0
        self.pre_step_calls = 0
        self.post_validate_values: list[bool] = []
        self.last_action = ""
        self.last_action_error = ""

    def reset(self, *, seed: int):
        task = self.kwargs["task_entrypoint"](
            seed=seed,
            **self.kwargs["task_kwargs"],
        )
        assert type(task) is OracleBlindWebArenaTask
        task.setup(self.page)
        self.task = task
        return {"causal": True}, {"must_not_be_read": object()}

    def pre_step(self):
        self.pre_step_calls += 1
        return {}, lambda value: None, lambda value: None

    def post_step(self, info, *, validate: bool):
        del info
        self.post_validate_values.append(validate)
        return {"causal": True}, object(), object(), object(), object()

    def close(self) -> None:
        self.close_calls += 1


EXECUTED_CODE: list[str] = []


def _fake_execute(code, page, *, send_message_to_user, report_infeasible_instructions):
    del page, send_message_to_user, report_infeasible_instructions
    EXECUTED_CODE.append(code)


def _reset_callback(
    task_id: str,
    start_state: RuntimeStartState,
    seed: int,
) -> BrowserGymTaskStateResetReceipt:
    return BrowserGymTaskStateResetReceipt(
        resetter_id="fixture-service-reset",
        resetter_version="v1",
        resetter_source_sha256=callable_source_sha256(_reset_callback),
        task_id=task_id,
        start_state_sha256=start_state.start_state_sha256,
        reset_seed=seed,
        require_reset=start_state.require_reset,
        reset_performed=start_state.require_reset,
        service_url_map_sha256=canonical_sha256(SERVICE_URLS),
        service_state_commitments={
            site: canonical_sha256({"site": site, "state": "clean"})
            for site in start_state.sites
        },
    )


def _task() -> TaskSpecification:
    start = RuntimeStartState(
        sites=("gitlab",),
        start_url="http://gitlab.test/project",
        require_login=True,
        storage_state=".auth/gitlab_state.json",
        geolocation=RuntimeGeolocation(latitude=23.81, longitude=90.41),
        require_reset=True,
    )
    return TaskSpecification(
        task_id="webarena.44",
        goal="Open the visible project issues",
        benchmark_id="webarena",
        benchmark_version="0.14.3",
        start_state_id=start.start_state_sha256,
        site="gitlab",
        start_url=start.start_url,
        runtime_start_state=start,
    )


def _callbacks(tmp_path: Path):
    auth = tmp_path / ".auth"
    auth.mkdir()
    (auth / "gitlab_state.json").write_text(
        '{"cookies": [], "origins": []}',
        encoding="utf-8",
    )
    runtime, _ = create_one_way_sealed_page_broker()
    api = PinnedBrowserGymAPI(
        browser_env_class=_FakeBrowserEnv,
        execute_python_code=_fake_execute,
        package_versions=PINNED_WEBARENA_PACKAGES,
        production_loader=False,
    )
    resetter = FrozenBrowserGymTaskStateResetter(
        resetter_id="fixture-service-reset",
        resetter_version="v1",
        source_sha256=callable_source_sha256(_reset_callback),
        service_url_map_sha256=canonical_sha256(SERVICE_URLS),
        callback=_reset_callback,
    )
    return _BrowserGymEpisodeCallbacks(
        task=_task(),
        api=api,
        configuration=BrowserGymRuntimeConfiguration(),
        service_url_map=SERVICE_URLS,
        credential_bundle_root=tmp_path,
        task_state_resetter=resetter,
        runtime_page_publisher=runtime,
        boundary_source_sha256=module_source_sha256(),
    )


def _action(action_type: ActionType) -> ConcreteAction:
    base = {
        "target_x": 0.2,
        "target_y": 0.25,
        "target_bbox": [0.1, 0.2, 0.3, 0.1],
    }
    parameters = {
        ActionType.CLICK: {**base, "button": "left", "click_count": 1},
        ActionType.TYPE: {**base, "text": "safe text"},
        ActionType.SELECT: {
            **base,
            "option": "alpha",
            "candidate_options": ["alpha", "beta"],
        },
        ActionType.SCROLL: {
            "direction": "down",
            "amount": 0.5,
            "container": "viewport",
        },
        ActionType.NAVIGATE: {"url": "http://gitlab.test/issues"},
        ActionType.PRESS_KEY: {"key": "ENTER"},
    }[action_type]
    return ConcreteAction(
        action_id=f"action-{action_type.value}",
        source_decision_id="decision-1",
        action_type=action_type,
        parameters=parameters,
        bbox=(0.1, 0.2, 0.3, 0.1) if action_type in {
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.SELECT,
        } else None,
    )


def test_pinned_loader_fails_closed_when_optional_stack_is_absent(monkeypatch) -> None:
    def missing(distribution: str) -> str:
        raise __import__("importlib").metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(
        "web_agent.benchmarks.browsergym_webarena.metadata.version",
        missing,
    )
    with pytest.raises(BenchmarkUnavailableError, match="unavailable"):
        load_pinned_browsergym_api()


def test_six_actions_compile_without_answer_or_stop_channel() -> None:
    for action_type in ActionType:
        code = _action_code(_action(action_type))
        compile(code, f"<{action_type.value}>", "exec")
        assert "send_message_to_user" not in code
        assert "report_infeasible" not in code


def test_validation_disabled_wrapper_applies_six_fields_and_emits_causal_evidence(
    tmp_path: Path,
) -> None:
    callbacks = _callbacks(tmp_path)
    environment = callbacks.environment_factory(callbacks.task, 42)
    raw = environment.reset(seed=42)
    settled = callbacks.settle(
        environment,
        raw,
        ObservationStage.RESET,
        3.0,
        True,
    )
    observation = callbacks.observation_mapper(
        settled,
        "campaign:E2:webarena.44:repeat-0:seed-42",
        ObservationStage.RESET,
        None,
    )
    assert observation.page_settled is True
    assert observation.page_state["observable_select_controls"] == [
        {
            "target_bbox": [0.1, 0.2, 0.3, 0.1],
            "candidate_options": ["alpha", "beta"],
        }
    ]
    evidence = observation.page_state["recovery_target_evidence"]
    assert evidence["observation_id"] == observation.observation_id
    assert evidence["registered_visible_targets"]
    assert callbacks.task.runtime_start_state is not None
    assert environment.reset_receipt.start_state_sha256 == (
        callbacks.task.runtime_start_state.start_state_sha256
    )
    kwargs = _FakeBrowserEnv.last_kwargs
    assert kwargs is not None
    assert kwargs["task_entrypoint"] is OracleBlindWebArenaTask
    assert kwargs["action_mapping"] is None
    assert kwargs["pw_context_kwargs"]["storage_state"].endswith(
        ".auth/gitlab_state.json"
    )
    assert kwargs["pw_context_kwargs"]["geolocation"] == {
        "latitude": 23.81,
        "longitude": 90.41,
    }

    environment.step(_action_code(_action(ActionType.CLICK)))
    browser = _FakeBrowserEnv.instances[-1]
    assert browser.pre_step_calls == 1
    assert browser.post_validate_values == [False]
    assert browser.close_calls == 0
    environment.close()
    assert browser.close_calls == 0
    receipt = callbacks.abort_episode(
        "campaign:E2:webarena.44:repeat-0:seed-42",
        "webarena.44",
    )
    assert type(receipt) is BrowserGymEpisodeAbortReceipt
    assert receipt.outcome == "BROKER_ABORTED"
    assert browser.close_calls == 1


def test_custom_task_validation_is_a_tripwire() -> None:
    task = _task()
    assert task.runtime_start_state is not None
    runtime_task = OracleBlindWebArenaTask(
        42,
        task=task,
        start_state=task.runtime_start_state,
        configuration=BrowserGymRuntimeConfiguration(),
    )
    with pytest.raises(BrowserGymWebArenaError, match="forbidden"):
        runtime_task.validate(object(), [])


def test_missing_authenticated_storage_fails_before_browser_creation(
    tmp_path: Path,
) -> None:
    callbacks = _callbacks(tmp_path)
    (tmp_path / ".auth" / "gitlab_state.json").unlink()
    environment = callbacks.environment_factory(callbacks.task, 42)
    with pytest.raises(BrowserGymWebArenaError, match="unavailable"):
        environment.reset(seed=42)
    receipt = callbacks.abort_episode("never-published", "webarena.44")
    assert receipt.outcome == "NO_BROWSER_CREATED"
