from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from web_agent.benchmarks import browsergym_webarena as browsergym_module
from web_agent.benchmarks.base import BenchmarkUnavailableError
from web_agent.benchmarks.browsergym_webarena import (
    BrowserGymWebArenaRuntimeFactory,
    BrowserGymEpisodeAbortReceipt,
    BrowserGymRuntimeConfiguration,
    BrowserGymTaskStateResetReceipt,
    BrowserGymWebArenaError,
    FrozenBrowserGymTaskStateResetter,
    OracleBlindWebArenaTask,
    PinnedBrowserGymAPI,
    _BrowserGymEpisodeCallbacks,
    _ProcessBrokerBrowserGymWebArenaAdapter,
    _action_code,
    _runtime_visible_page_url,
    callable_source_sha256,
    load_pinned_browsergym_api,
    module_source_sha256,
)
from web_agent.benchmarks.webarena import WebArenaAdapter
from web_agent.eval.table2.process_broker_protocol import (
    registered_browser_error_observation_url,
    validate_runtime_result,
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


def _callbacks(tmp_path: Path, *, execute_python_code=_fake_execute):
    auth = tmp_path / ".auth"
    auth.mkdir()
    (auth / "gitlab_state.json").write_text(
        '{"cookies": [], "origins": []}',
        encoding="utf-8",
    )
    runtime, _ = create_one_way_sealed_page_broker()
    api = PinnedBrowserGymAPI(
        browser_env_class=_FakeBrowserEnv,
        execute_python_code=execute_python_code,
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


def test_only_registered_observable_browser_errors_gain_a_runtime_url() -> None:
    assert (
        _runtime_visible_page_url(
            "https://gitlab.test/issues",
            browser_error_kind=None,
        )
        == "https://gitlab.test/issues"
    )
    expected = registered_browser_error_observation_url("TIMEOUT_ERROR")
    assert (
        _runtime_visible_page_url(
            "chrome-error://chromewebdata/",
            browser_error_kind="TIMEOUT_ERROR",
        )
        == expected
    )
    assert (
        _runtime_visible_page_url(
            "about:blank",
            browser_error_kind="TIMEOUT_ERROR",
        )
        == expected
    )
    for url, error_kind in (
        ("chrome-error://chromewebdata/", None),
        ("chrome-error://chromewebdata:9/", "TIMEOUT_ERROR"),
        ("CHROME-ERROR://CHROMEWEBDATA/", "TIMEOUT_ERROR"),
        ("chrome-error://chromewebdata", "TIMEOUT_ERROR"),
        ("chrome-error://chromewebdata/?detail=x", "TIMEOUT_ERROR"),
        ("chrome-error://user@chromewebdata/", "TIMEOUT_ERROR"),
        ("chrome://settings", "TIMEOUT_ERROR"),
        ("about:blank", None),
    ):
        with pytest.raises(BrowserGymWebArenaError, match="registered causal error"):
            _runtime_visible_page_url(url, browser_error_kind=error_kind)
    with pytest.raises(BrowserGymWebArenaError, match="safe absolute data"):
        _runtime_visible_page_url(
            object(),
            browser_error_kind="TIMEOUT_ERROR",
        )


def test_browser_error_observation_survives_browsergym_to_process_broker(
    tmp_path: Path,
) -> None:
    callbacks = _callbacks(tmp_path)
    environment = callbacks.environment_factory(callbacks.task, 42)
    environment.reset(seed=42)
    browser = _FakeBrowserEnv.instances[-1]
    browser.page.url = "chrome-error://chromewebdata/"
    environment._last_action_error_kind = "TIMEOUT_ERROR"
    raw = environment._capture(page_settled=True)
    episode_id = "campaign:E2:webarena.44:repeat-0:seed-42"
    observation = callbacks.observation_mapper(
        raw,
        episode_id,
        ObservationStage.POST_ACTION,
        "action-1",
    )
    expected_url = registered_browser_error_observation_url("TIMEOUT_ERROR")
    assert observation.url == expected_url
    assert observation.environment_error is True
    assert observation.page_state["browser_error_kind"] == "TIMEOUT_ERROR"
    value = {"observation": observation.to_dict()}
    assert validate_runtime_result(
        "runtime_observe",
        value,
        request_payload={
            "episode_id": episode_id,
            "task_id": "webarena.44",
            "stage": "post_action",
            "prior_action_id": "action-1",
        },
    ) == value
    callbacks.abort_episode(episode_id, "webarena.44")


def test_executed_browser_failure_reaches_post_action_observation(
    tmp_path: Path,
) -> None:
    def execute_with_timeout(
        code,
        page,
        *,
        send_message_to_user,
        report_infeasible_instructions,
    ) -> None:
        del code, send_message_to_user, report_infeasible_instructions
        page.url = "chrome-error://chromewebdata/"
        raise TimeoutError("fixture browser timeout")

    callbacks = _callbacks(
        tmp_path,
        execute_python_code=execute_with_timeout,
    )
    environment = callbacks.environment_factory(callbacks.task, 42)
    environment.reset(seed=42)
    raw, _, _, _, _ = environment.step(_action_code(_action(ActionType.CLICK)))
    assert raw.environment_error is True
    assert raw.browser_error_kind == "TIMEOUTERROR"
    settled = callbacks.settle(
        environment,
        raw,
        ObservationStage.POST_ACTION,
        3.0,
        True,
    )
    episode_id = "campaign:E2:webarena.44:repeat-0:seed-42"
    observation = callbacks.observation_mapper(
        settled,
        episode_id,
        ObservationStage.POST_ACTION,
        "action-1",
    )
    assert observation.url == registered_browser_error_observation_url(
        "TIMEOUTERROR"
    )
    assert observation.environment_error is True
    assert observation.page_state["browser_error_kind"] == "TIMEOUTERROR"
    callbacks.abort_episode(episode_id, "webarena.44")


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


def test_process_broker_adapter_close_defers_browser_until_control_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks = _callbacks(tmp_path)
    runtime_dir = tmp_path / "episode-runtime"
    runtime_dir.mkdir()
    factory = SimpleNamespace(
        benchmark_version="0.14.3",
        environment_adapter_id="fixture-child-browsergym",
        environment_adapter_version="v1",
        environment_state_digester_id="fixture-page-state",
        environment_state_digester_version="v1",
        infrastructure_fault_classifier=None,
        reset_state_attester=None,
        action_safety_policy=None,
        manual_rescue_guard=None,
        page_settle_policy_id="fixture-settle",
        configuration=BrowserGymRuntimeConfiguration(),
    )
    # This test supplies deterministic callbacks directly; dependency/policy
    # completeness is covered by BrowserGymWebArenaRuntimeFactory construction.
    monkeypatch.setattr(WebArenaAdapter, "_require_contract", lambda _self: None)
    adapter = _ProcessBrokerBrowserGymWebArenaAdapter(
        task=callbacks.task,
        callbacks=callbacks,
        runtime_factory=factory,
        episode_runtime_dir=runtime_dir,
    )
    episode_id = "campaign:E2:webarena.44:repeat-0:seed-42"
    adapter.reset(callbacks.task, episode_id=episode_id, seed=42)
    browser = _FakeBrowserEnv.instances[-1]
    adapter.close()
    assert browser.close_calls == 0

    # A sealed finalizer can still read the exact state after runtime close.
    committed_before_cleanup = adapter.process_broker_environment_state_sha256()
    assert len(committed_before_cleanup) == 64
    assert browser.page.content()

    receipt = adapter.process_broker_close_browser()
    assert receipt is not None
    assert receipt.outcome == "BROKER_ABORTED"
    assert browser.close_calls == 1
    with pytest.raises(BrowserGymWebArenaError, match="exactly once"):
        adapter.process_broker_close_browser()


def test_exported_process_broker_adapter_entrypoint_is_exact_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(browsergym_module.create_environment_adapter)
    assert tuple(signature.parameters) == ("task", "episode_runtime_dir")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    runtime_dir = tmp_path / "episode-runtime"
    runtime_dir.mkdir()
    with pytest.raises(BrowserGymWebArenaError, match="bootstrap is not registered"):
        browsergym_module.create_environment_adapter(
            task=_task(),
            episode_runtime_dir=runtime_dir,
        )

    exact_factory = object.__new__(BrowserGymWebArenaRuntimeFactory)
    expected = object.__new__(WebArenaAdapter)
    monkeypatch.setattr(
        browsergym_module,
        "_load_process_broker_browser_runtime_factory",
        lambda: exact_factory,
    )
    monkeypatch.setattr(
        BrowserGymWebArenaRuntimeFactory,
        "create_process_broker_environment_adapter",
        lambda _self, *, task, episode_runtime_dir: expected,
    )
    assert browsergym_module.create_environment_adapter(
        task=_task(),
        episode_runtime_dir=runtime_dir,
    ) is expected
