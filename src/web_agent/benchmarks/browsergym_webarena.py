"""Pinned BrowserGym execution boundary for the Table 2 WebArena pilot.

This module is intentionally dependency-lazy.  Importing it does not import
BrowserGym, Playwright, Gymnasium, or WebArena.  The production loader accepts
only the preregistered package versions and the runtime wrapper never creates a
``GenericWebArenaTask``.  It uses a small oracle-blind task whose ``validate``
method is a fail-closed tripwire, and executes every action through the exact
``BrowserEnv.pre_step`` / ``execute_python_code`` /
``BrowserEnv.post_step(validate=False)`` path.

The live page is published to the one-way sealed broker after all six upstream
start-state fields have been applied.  Ordinary adapter close is deferred by
that broker; explicit abort and sealed finalization are the only cleanup paths
after publication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import functools
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import urlsplit

from web_agent.benchmarks.base import (
    BenchmarkStateError,
    BenchmarkUnavailableError,
)
from web_agent.benchmarks.webarena import (
    FrozenWebArenaActionSafetyPolicy,
    FrozenWebArenaEnvironmentStateDigester,
    FrozenWebArenaInfrastructureFaultClassifier,
    FrozenWebArenaManualRescueGuard,
    FrozenWebArenaPageSettlePolicy,
    WebArenaAdapter,
    WebArenaManualRescueCheck,
    WebArenaManualRescueReceipt,
)
from web_agent.runtime.action_parameters import validate_action_parameters
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    JsonValue,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    RuntimeStartState,
    TaskSpecification,
    VersionedRecord,
    canonical_sha256,
    detached_record_copy,
)
from web_agent.runtime.observation import assert_oracle_blind_mapping
from web_agent.runtime.recovery.strategies import build_recovery_target_evidence
from web_agent.runtime.state_reset import FrozenWebArenaResetStateAttester

from web_agent.eval.table2.live_deployment import (
    REGISTERED_BROWSERGYM_VERSION,
    REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH,
)
from web_agent.eval.table2.common import canonical_json_bytes
from web_agent.eval.table2.process_broker_protocol import (
    ProcessBrokerProtocolError,
    registered_browser_error_observation_url,
)
from web_agent.eval.table2.sealed_page_broker import (
    BrowserGymStartStateApplicationReceipt,
    BrowserGymValidationDisabledBoundary,
    PageSessionAbortReceipt,
    RuntimePagePublisher,
)
from web_agent.eval.table2.webarena_preflight import PINNED_WEBARENA_PACKAGES
from web_agent.eval.table2.webarena_preflight_binding import (
    ValidatedDeploymentPreflight,
)


CAUSAL_OBSERVATION_SCHEMA = "table2-browsergym-causal-observation-v1"
TASK_STATE_RESET_SCHEMA = "table2-browsergym-task-state-reset-v1"
EPISODE_ABORT_SCHEMA = "table2-browsergym-episode-abort-v1"
PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_SCHEMA_VERSION = (
    "table2-process-broker-manual-rescue-sidecar-v1"
)
PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_RECORD_TYPE = (
    "ProcessBrokerManualRescueEvidence"
)
PROCESS_BROKER_MANUAL_RESCUE_IDENTITY_SCHEMA_VERSION = (
    "table2-process-broker-manual-rescue-sidecar-identity-v1"
)
PROCESS_BROKER_MANUAL_RESCUE_IDENTITY_RECORD_TYPE = (
    "ProcessBrokerManualRescueSidecarIdentity"
)
PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_FILENAME = (
    "manual_rescue_guard.child.jsonl"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_SCHEMA_VERSION = (
    "table2-process-broker-sealed-callback-state-guard-v1"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_RECORD_TYPE = (
    "ProcessBrokerSealedCallbackStateGuard"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION = (
    "table2-process-broker-sealed-callback-state-guard-sidecar-identity-v1"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE = (
    "ProcessBrokerSealedCallbackStateGuardSidecarIdentity"
)
PROCESS_BROKER_SEALED_CALLBACK_GUARD_SIDECAR_FILENAME = (
    "sealed_callback_state_guard.child.jsonl"
)
REGISTERED_BROWSERGYM_DEPENDENCY_MODULE = "browsergym"
REGISTERED_ACTION_TYPES = tuple(ActionType)
REGISTERED_SERVICE_KEYS = frozenset(
    {
        "WA_SHOPPING",
        "WA_SHOPPING_ADMIN",
        "WA_REDDIT",
        "WA_GITLAB",
        "WA_WIKIPEDIA",
        "WA_MAP",
        "WA_HOMEPAGE",
    }
)
SITE_TO_SERVICE_KEY = {
    "shopping": "WA_SHOPPING",
    "shopping_admin": "WA_SHOPPING_ADMIN",
    "reddit": "WA_REDDIT",
    "gitlab": "WA_GITLAB",
    "wikipedia": "WA_WIKIPEDIA",
    "map": "WA_MAP",
    "homepage": "WA_HOMEPAGE",
}
REGISTERED_ABORT_OUTCOMES = frozenset(
    {
        "NO_BROWSER_CREATED",
        "CLEANED_BEFORE_PUBLICATION",
        "BROKER_ABORTED",
        "ALREADY_CLEANED_BY_SEALED_BROKER",
    }
)


class BrowserGymWebArenaError(BenchmarkStateError):
    """The pinned live wrapper or one of its evidence gates failed."""


def _require_sha256(value: object, *, field_name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _source_file(callback: Callable[..., Any]) -> Path:
    target: Any = callback.func if isinstance(callback, functools.partial) else callback
    if inspect.ismethod(target):
        target = target.__func__
    elif not (inspect.isfunction(target) or inspect.isclass(target)):
        target = getattr(type(target), "__call__", None)
    if target is None:
        raise ValueError("cannot resolve callback source")
    source_name = inspect.getsourcefile(target) or inspect.getfile(target)
    source = Path(source_name).absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError("callback source must be a regular non-symlink file")
    return source.resolve()


def callable_source_sha256(callback: Callable[..., Any]) -> str:
    """Return the byte identity of one concrete source-backed callback."""

    return hashlib.sha256(_source_file(callback).read_bytes()).hexdigest()


def module_source_sha256() -> str:
    """Identity used by the frozen validation-disabled boundary receipt."""

    source = Path(__file__).absolute()
    if source.is_symlink() or not source.is_file():
        raise BrowserGymWebArenaError(
            "BrowserGym wrapper source must be a regular non-symlink file"
        )
    return hashlib.sha256(source.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PinnedBrowserGymAPI:
    """The two BrowserGym callables used by the wrapper, loaded lazily."""

    browser_env_class: type
    execute_python_code: Callable[..., Any]
    package_versions: Mapping[str, str]
    production_loader: bool

    def __post_init__(self) -> None:
        if not isinstance(self.browser_env_class, type):
            raise TypeError("BrowserGym API requires the BrowserEnv class")
        if not callable(self.execute_python_code):
            raise TypeError("BrowserGym API requires execute_python_code")
        if dict(self.package_versions) != dict(PINNED_WEBARENA_PACKAGES):
            raise ValueError("BrowserGym API package versions differ from registration")
        if type(self.production_loader) is not bool:
            raise TypeError("BrowserGym API production_loader must be boolean")


def load_pinned_browsergym_api() -> PinnedBrowserGymAPI:
    """Import the exact audited API only after the live gate requests it."""

    installed: dict[str, str] = {}
    for distribution, expected in PINNED_WEBARENA_PACKAGES.items():
        try:
            installed[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise BenchmarkUnavailableError(
                f"required pinned dependency is unavailable: {distribution}=={expected}"
            ) from exc
        if installed[distribution] != expected:
            raise BenchmarkUnavailableError(
                f"pinned dependency mismatch for {distribution}: "
                f"{installed[distribution]!r} != {expected!r}"
            )
    try:
        env_module = importlib.import_module("browsergym.core.env")
        action_module = importlib.import_module("browsergym.core.action.base")
        browser_env_class = getattr(env_module, "BrowserEnv")
        execute_python_code = getattr(action_module, "execute_python_code")
    except (ImportError, AttributeError) as exc:
        raise BenchmarkUnavailableError(
            "pinned BrowserGym API could not be imported exactly"
        ) from exc
    if (
        not isinstance(browser_env_class, type)
        or browser_env_class.__module__ != "browsergym.core.env"
        or browser_env_class.__name__ != "BrowserEnv"
        or not callable(execute_python_code)
        or getattr(execute_python_code, "__module__", "")
        != "browsergym.core.action.base"
        or getattr(execute_python_code, "__name__", "") != "execute_python_code"
    ):
        raise BenchmarkUnavailableError("pinned BrowserGym API identity changed")
    return PinnedBrowserGymAPI(
        browser_env_class=browser_env_class,
        execute_python_code=execute_python_code,
        package_versions=installed,
        production_loader=True,
    )


@dataclass(frozen=True, slots=True)
class BrowserGymTaskStateResetReceipt(VersionedRecord):
    """Hashes-only service/reset evidence produced before browser creation."""

    schema_version = TASK_STATE_RESET_SCHEMA

    resetter_id: str
    resetter_version: str
    resetter_source_sha256: str
    task_id: str
    start_state_sha256: str
    reset_seed: int
    require_reset: bool
    reset_performed: bool
    service_url_map_sha256: str
    service_state_commitments: Mapping[str, str]
    credential_material_returned: bool = False
    oracle_labels_observed: bool = False
    locked_test_rows_read: int = 0

    def __post_init__(self) -> None:
        for name in ("resetter_id", "resetter_version", "task_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"task-state reset receipt requires {name}")
        for name in (
            "resetter_source_sha256",
            "start_state_sha256",
            "service_url_map_sha256",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        if type(self.reset_seed) is not int or self.reset_seed < 0:
            raise ValueError("task-state reset seed must be a nonnegative integer")
        if type(self.require_reset) is not bool or type(self.reset_performed) is not bool:
            raise TypeError("task-state reset flags must be exact booleans")
        if self.reset_performed is not self.require_reset:
            raise ValueError("task-state reset execution contradicts require_reset")
        if not isinstance(self.service_state_commitments, Mapping) or not self.service_state_commitments:
            raise ValueError("task-state reset requires service-state commitments")
        for site, digest in self.service_state_commitments.items():
            if type(site) is not str or not site.strip():
                raise ValueError("service-state commitment has an invalid site")
            _require_sha256(digest, field_name=f"service commitment {site}")
        if (
            self.credential_material_returned is not False
            or self.oracle_labels_observed is not False
            or type(self.locked_test_rows_read) is not int
            or self.locked_test_rows_read != 0
        ):
            raise ValueError("task-state reset receipt crossed a protected boundary")


TaskStateResetCallable = Callable[
    [str, RuntimeStartState, int], BrowserGymTaskStateResetReceipt
]


@dataclass(frozen=True, slots=True)
class FrozenBrowserGymTaskStateResetter:
    """Source-bound deployment callback for the six-field reset contract."""

    resetter_id: str
    resetter_version: str
    source_sha256: str
    service_url_map_sha256: str
    callback: TaskStateResetCallable
    frozen: bool = True

    def __post_init__(self) -> None:
        if not self.resetter_id.strip() or not self.resetter_version.strip():
            raise ValueError("task-state resetter identity/version are required")
        _require_sha256(self.source_sha256, field_name="task-state resetter source")
        _require_sha256(
            self.service_url_map_sha256,
            field_name="task-state resetter service URL map",
        )
        if not callable(self.callback):
            raise TypeError("task-state resetter callback must be callable")
        if callable_source_sha256(self.callback) != self.source_sha256:
            raise ValueError("task-state resetter callback differs from its source hash")
        if self.frozen is not True:
            raise ValueError("task-state resetter must be frozen")

    def reset(
        self,
        task_id: str,
        start_state: RuntimeStartState,
        seed: int,
    ) -> BrowserGymTaskStateResetReceipt:
        if type(start_state) is not RuntimeStartState:
            raise TypeError("task-state reset requires RuntimeStartState")
        protected_sha256 = start_state.record_sha256
        detached = detached_record_copy(start_state)
        callback_error: BaseException | None = None
        result: object | None = None
        try:
            result = self.callback(task_id, detached, seed)
        except BaseException as exc:
            callback_error = exc
        if (
            start_state.record_sha256 != protected_sha256
            or detached.record_sha256 != protected_sha256
        ):
            mutation = BrowserGymWebArenaError(
                "task-state resetter mutated protected start-state input"
            )
            if callback_error is not None:
                raise mutation from callback_error
            raise mutation
        if callback_error is not None:
            raise callback_error
        if type(result) is not BrowserGymTaskStateResetReceipt:
            raise TypeError("task-state resetter returned the wrong receipt type")
        expected = {
            "resetter_id": self.resetter_id,
            "resetter_version": self.resetter_version,
            "resetter_source_sha256": self.source_sha256,
            "task_id": task_id,
            "start_state_sha256": start_state.start_state_sha256,
            "reset_seed": seed,
            "require_reset": start_state.require_reset,
            "service_url_map_sha256": self.service_url_map_sha256,
        }
        for field_name, value in expected.items():
            if getattr(result, field_name) != value:
                raise ValueError(
                    f"task-state reset receipt differs from request: {field_name}"
                )
        if set(result.service_state_commitments) != set(start_state.sites):
            raise ValueError("task-state reset commitments do not cover task sites exactly")
        return result


@dataclass(frozen=True, slots=True)
class BrowserGymEpisodeAbortReceipt(VersionedRecord):
    """One typed orchestration receipt for exceptional live-browser cleanup."""

    schema_version = EPISODE_ABORT_SCHEMA

    episode_id: str
    task_id: str
    outcome: str
    underlying_browser_created: bool
    underlying_browser_close_called: bool
    broker_abort_receipt_sha256: str | None = None
    live_page_returned: bool = False

    def __post_init__(self) -> None:
        for name in ("episode_id", "task_id"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"browser abort receipt requires {name}")
        if self.outcome not in REGISTERED_ABORT_OUTCOMES:
            raise ValueError("browser abort receipt outcome is not registered")
        if (
            type(self.underlying_browser_created) is not bool
            or type(self.underlying_browser_close_called) is not bool
        ):
            raise TypeError("browser abort receipt flags must be exact booleans")
        expected_close = self.outcome != "NO_BROWSER_CREATED"
        if self.underlying_browser_created is not expected_close:
            raise ValueError("browser abort creation flag contradicts outcome")
        if self.underlying_browser_close_called is not expected_close:
            raise ValueError("browser abort cleanup flag contradicts outcome")
        if self.outcome == "BROKER_ABORTED":
            _require_sha256(
                self.broker_abort_receipt_sha256,
                field_name="broker abort receipt",
            )
        elif self.broker_abort_receipt_sha256 is not None:
            raise ValueError("non-broker abort cannot cite a broker abort receipt")
        if self.live_page_returned is not False:
            raise ValueError("browser abort receipt cannot return the live page")


@dataclass(frozen=True, slots=True)
class BrowserGymRuntimeConfiguration(VersionedRecord):
    """Frozen browser/controller choices that affect task difficulty."""

    viewport_width: int = 1280
    viewport_height: int = 720
    headless: bool = True
    slow_mo_ms: int = 0
    playwright_timeout_ms: int = 10_000
    locale: str | None = None
    timezone_id: str | None = None
    tags_to_mark: str = "standard_html"
    pre_observation_delay_seconds: float = 0.5
    network_idle_required: bool = True
    page_settle_timeout_seconds: float = 10.0
    video_recording_enabled: bool = False
    trace_recording_enabled: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.viewport_width) is not int
            or type(self.viewport_height) is not int
            or self.viewport_width <= 0
            or self.viewport_height <= 0
        ):
            raise ValueError("BrowserGym viewport must use positive exact integers")
        for name in ("slow_mo_ms", "playwright_timeout_ms"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"BrowserGym {name} must be nonnegative")
        if self.playwright_timeout_ms == 0:
            raise ValueError("BrowserGym Playwright timeout must be positive")
        if self.headless is not True:
            raise ValueError("production BrowserGym controller must be headless")
        if self.tags_to_mark not in {"all", "standard_html"}:
            raise ValueError("BrowserGym tags_to_mark is not registered")
        for name in (
            "pre_observation_delay_seconds",
            "page_settle_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"BrowserGym {name} must be positive and finite")
        if type(self.network_idle_required) is not bool:
            raise TypeError("BrowserGym network-idle setting must be boolean")
        if self.video_recording_enabled or self.trace_recording_enabled:
            raise ValueError("raw BrowserGym video/trace capture is forbidden")


@dataclass(frozen=True, slots=True)
class BrowserGymCausalRawObservation:
    """Internal current-page snapshot; it contains no live page capability."""

    task_id: str
    task_goal: str
    sequence: int
    url: str
    title: str
    width: int
    height: int
    screenshot_png: bytes
    browser_state: Mapping[str, JsonValue]
    page_settled: bool
    environment_error: bool
    browser_error_kind: str | None
    snapshot_sha256: str

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id.strip():
            raise ValueError("causal snapshot requires task_id")
        if type(self.task_goal) is not str or not self.task_goal.strip():
            raise ValueError("causal snapshot requires task_goal")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise ValueError("causal snapshot sequence must be positive")
        if type(self.width) is not int or type(self.height) is not int:
            raise TypeError("causal snapshot viewport must use exact integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("causal snapshot viewport must be positive")
        if type(self.screenshot_png) is not bytes or not self.screenshot_png:
            raise ValueError("causal snapshot requires PNG bytes")
        if not isinstance(self.browser_state, Mapping):
            raise TypeError("causal snapshot browser state must be a mapping")
        assert_oracle_blind_mapping(
            self.browser_state,
            location="browsergym_causal_raw_observation",
        )
        if type(self.page_settled) is not bool or type(self.environment_error) is not bool:
            raise TypeError("causal snapshot state flags must be exact booleans")
        if self.browser_error_kind is not None and (
            type(self.browser_error_kind) is not str
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", self.browser_error_kind)
        ):
            raise ValueError("causal snapshot browser error kind is invalid")
        if self.environment_error is not (self.browser_error_kind is not None):
            raise ValueError(
                "causal snapshot browser error flag and kind must agree"
            )
        _require_sha256(self.snapshot_sha256, field_name="causal snapshot")


class OracleBlindWebArenaTask:
    """Duck-typed BrowserGym task with no evaluator construction or validation."""

    def __init__(
        self,
        seed: int,
        *,
        task: TaskSpecification,
        start_state: RuntimeStartState,
        configuration: BrowserGymRuntimeConfiguration,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("oracle-blind BrowserGym task seed is invalid")
        if type(task) is not TaskSpecification or type(start_state) is not RuntimeStartState:
            raise TypeError("oracle-blind BrowserGym task has invalid contracts")
        if task.runtime_start_state != start_state:
            raise ValueError("oracle-blind BrowserGym task start-state mismatch")
        self.task = task
        self.start_state = start_state
        self.viewport = {
            "width": configuration.viewport_width,
            "height": configuration.viewport_height,
        }
        self.slow_mo = configuration.slow_mo_ms
        self.timeout = configuration.playwright_timeout_ms
        self.locale = configuration.locale
        self.timezone_id = configuration.timezone_id

    @classmethod
    def get_task_id(cls) -> str:
        return "table2-oracle-blind-webarena"

    def setup(self, page: Any) -> tuple[str, dict[str, Any]]:
        urls = self.start_state.start_url.split(" |AND| ")
        active = page
        for index, url in enumerate(urls):
            if index:
                active = page.context.new_page()
            active.goto(url, wait_until="domcontentloaded")
        if hasattr(active, "bring_to_front"):
            active.bring_to_front()
        return self.task.goal, {}

    def validate(self, page: Any, chat_messages: Any) -> tuple[Any, ...]:
        del page, chat_messages
        raise BrowserGymWebArenaError(
            "runtime task validation/evaluator invocation is forbidden"
        )

    def teardown(self) -> None:
        return None


_CONTROL_SNAPSHOT_JAVASCRIPT = r"""
() => {
  const round = value => Math.round(value * 1000000) / 1000000;
  const width = Math.max(window.innerWidth, 1);
  const height = Math.max(window.innerHeight, 1);
  const candidates = Array.from(document.querySelectorAll(
    'a,button,input,textarea,select,[role="button"],[contenteditable="true"]'
  ));
  const controls = [];
  for (const element of candidates) {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    if (rect.width <= 0 || rect.height <= 0 || style.visibility === 'hidden' ||
        style.display === 'none' || rect.bottom <= 0 || rect.right <= 0 ||
        rect.top >= height || rect.left >= width) continue;
    const left = Math.max(0, rect.left);
    const top = Math.max(0, rect.top);
    const right = Math.min(width, rect.right);
    const bottom = Math.min(height, rect.bottom);
    const bbox = [
      round(left / width), round(top / height),
      round((right - left) / width), round((bottom - top) / height)
    ];
    const options = element.tagName === 'SELECT'
      ? Array.from(element.options).map(option => option.value || option.textContent || '')
      : [];
    controls.push({
      tag: String(element.tagName || '').toLowerCase(),
      role: String(element.getAttribute('role') || ''),
      input_type: String(element.getAttribute('type') || ''),
      name: String(element.getAttribute('aria-label') || element.getAttribute('name') || '').slice(0, 160),
      text: String(element.innerText || element.value || element.textContent || '').trim().slice(0, 240),
      bbox,
      candidate_options: options.slice(0, 256),
      destination: element.tagName === 'A' ? String(element.href || '') : ''
    });
  }
  return {
    visible_text: String(document.body ? document.body.innerText : '').slice(0, 8192),
    controls: controls.slice(0, 512)
  };
}
"""


def _safe_origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise BrowserGymWebArenaError("WebArena URL is not a credential-free HTTP(S) URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BrowserGymWebArenaError("WebArena URL contains an invalid port") from exc
    return parsed.scheme.lower(), parsed.hostname.lower(), port


def _runtime_visible_page_url(
    url: object,
    *,
    browser_error_kind: str | None,
) -> str:
    """Map only an observed Chromium error document to the typed runtime sentinel."""

    if (
        type(url) is not str
        or not url
        or url != url.strip()
        or "\\" in url
        or any(character.isspace() or ord(character) < 32 for character in url)
    ):
        raise BrowserGymWebArenaError("BrowserGym page URL is not safe absolute data")
    if url in {"about:blank", "chrome-error://chromewebdata/"}:
        if browser_error_kind is None:
            raise BrowserGymWebArenaError(
                "BrowserGym internal page URL is not a registered causal error observation"
            )
        try:
            return registered_browser_error_observation_url(browser_error_kind)
        except ProcessBrokerProtocolError as exc:
            raise BrowserGymWebArenaError(
                "BrowserGym browser error kind cannot form a runtime URL"
            ) from exc
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise BrowserGymWebArenaError("BrowserGym page URL is not parseable") from exc
    if parsed.scheme.casefold() in {"http", "https"}:
        _safe_origin(url)
        if "@" in parsed.netloc or port == 0:
            raise BrowserGymWebArenaError(
                "BrowserGym page URL is outside the registered HTTP(S) schema"
            )
        return url
    raise BrowserGymWebArenaError(
        "BrowserGym internal page URL is not a registered causal error observation"
    )


def _validate_start_state_services(
    start_state: RuntimeStartState,
    service_url_map: Mapping[str, str],
) -> None:
    if set(service_url_map) != REGISTERED_SERVICE_KEYS:
        raise BrowserGymWebArenaError("WebArena service map does not cover seven origins")
    service_origins = {
        key: _safe_origin(url) for key, url in service_url_map.items()
    }
    for site in start_state.sites:
        key = SITE_TO_SERVICE_KEY.get(site)
        if key is None or key not in service_origins:
            raise BrowserGymWebArenaError(f"WebArena task names unknown site {site!r}")
    allowed = set(service_origins.values())
    for url in start_state.start_url.split(" |AND| "):
        if _safe_origin(url) not in allowed:
            raise BrowserGymWebArenaError(
                "WebArena start URL is outside the measured seven-service map"
            )


def _credential_storage_path(
    start_state: RuntimeStartState,
    credential_bundle_root: Path,
) -> tuple[Path | None, str | None]:
    reference = start_state.storage_state
    if start_state.require_login and reference is None:
        raise BrowserGymWebArenaError(
            "authenticated WebArena start state lacks storage_state"
        )
    if reference is None:
        return None, None
    unresolved = credential_bundle_root / reference
    if unresolved.is_symlink():
        raise BrowserGymWebArenaError("credential storage state must not be a symlink")
    path = unresolved.resolve()
    if credential_bundle_root != path and credential_bundle_root not in path.parents:
        raise BrowserGymWebArenaError("credential storage state escaped its frozen root")
    try:
        metadata_value = path.lstat()
    except OSError as exc:
        raise BrowserGymWebArenaError("credential storage state is unavailable") from exc
    if not stat.S_ISREG(metadata_value.st_mode) or metadata_value.st_size <= 0:
        raise BrowserGymWebArenaError("credential storage state must be a nonempty file")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrowserGymWebArenaError("credential storage state is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise BrowserGymWebArenaError("credential storage state must be a JSON object")
    # Do not retain or return the parsed credential material.
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_normalized_bbox(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise BrowserGymWebArenaError("visible control bbox must be a JSON list")
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
        ):
            raise BrowserGymWebArenaError("visible control bbox is invalid")
        result.append(float(item))
    x, y, width, height = result
    if width <= 0.0 or height <= 0.0 or x + width > 1.000001 or y + height > 1.000001:
        raise BrowserGymWebArenaError("visible control bbox falls outside viewport")
    # JS rounding can exceed the boundary by one micro-unit; clamp only that
    # measured edge without changing the target centre materially.
    result[2] = min(width, 1.0 - x)
    result[3] = min(height, 1.0 - y)
    return result


def _clean_control_snapshot(value: object) -> dict[str, JsonValue]:
    expected = {
        "tag",
        "role",
        "input_type",
        "name",
        "text",
        "bbox",
        "candidate_options",
        "destination",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BrowserGymWebArenaError("visible control schema changed")
    for field_name in ("tag", "role", "input_type", "name", "text", "destination"):
        if type(value[field_name]) is not str:
            raise BrowserGymWebArenaError(f"visible control {field_name} must be text")
    options = value["candidate_options"]
    if not isinstance(options, list) or any(
        type(option) not in {str, int, float} or (
            isinstance(option, float) and not math.isfinite(option)
        )
        for option in options
    ):
        raise BrowserGymWebArenaError("visible SELECT options are not JSON scalars")
    destination = str(value["destination"])
    if destination:
        try:
            _safe_origin(destination)
        except BrowserGymWebArenaError:
            destination = ""
    return {
        "tag": str(value["tag"]).lower()[:32],
        "role": str(value["role"])[:64],
        "input_type": str(value["input_type"]).lower()[:32],
        "name": str(value["name"])[:160],
        "text": str(value["text"])[:240],
        "target_bbox": _finite_normalized_bbox(value["bbox"]),
        "candidate_options": list(options[:256]),
        "destination": destination,
    }


def _error_kind(error: BaseException) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", type(error).__name__).upper().strip("_")
    return (name or "BROWSER_ACTION_ERROR")[:96]


def _action_code(action: ConcreteAction) -> str:
    if type(action) is not ConcreteAction or action.action_type not in REGISTERED_ACTION_TYPES:
        raise BrowserGymWebArenaError("browser action is outside the six-class contract")
    validate_action_parameters(action.action_type, action.parameters, action.bbox)
    values = action.parameters
    if action.action_type in {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT}:
        x = float(values["target_x"])
        y = float(values["target_y"])
        prefix = (
            "viewport = page.viewport_size\n"
            "if viewport is None:\n"
            "    raise RuntimeError('active page has no viewport')\n"
            f"target_x = viewport['width'] * {x!r}\n"
            f"target_y = viewport['height'] * {y!r}\n"
        )
        if action.action_type is ActionType.CLICK:
            return prefix + (
                "page.mouse.click(target_x, target_y, "
                f"button={json.dumps(values['button'])}, "
                f"click_count={int(values['click_count'])})"
            )
        if action.action_type is ActionType.TYPE:
            return prefix + (
                "page.mouse.click(target_x, target_y)\n"
                "page.keyboard.press('ControlOrMeta+A')\n"
                f"page.keyboard.insert_text({json.dumps(values['text'])})"
            )
        option_json = json.dumps(values["option"], ensure_ascii=False)
        return prefix + (
            "selected = page.evaluate(\"\"\"([x, y, requested]) => {\n"
            "  const element = document.elementFromPoint(x, y);\n"
            "  if (!element || element.tagName !== 'SELECT') return false;\n"
            "  const requestedText = String(requested);\n"
            "  const option = Array.from(element.options).find(item =>\n"
            "    item.value === requestedText || String(item.textContent || '').trim() === requestedText\n"
            "  );\n"
            "  if (!option) return false;\n"
            "  element.value = option.value;\n"
            "  element.dispatchEvent(new Event('input', {bubbles: true}));\n"
            "  element.dispatchEvent(new Event('change', {bubbles: true}));\n"
            "  return true;\n"
            "}\"\"\", [target_x, target_y, " + option_json + "])\n"
            "if selected is not True:\n"
            "    raise RuntimeError('grounded SELECT control/option is unavailable')"
        )
    if action.action_type is ActionType.SCROLL:
        signs = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        dx_sign, dy_sign = signs[str(values["direction"])]
        amount = float(values["amount"])
        return (
            "viewport = page.viewport_size\n"
            "if viewport is None:\n"
            "    raise RuntimeError('active page has no viewport')\n"
            f"page.mouse.wheel(viewport['width'] * {dx_sign * amount!r}, "
            f"viewport['height'] * {dy_sign * amount!r})"
        )
    if action.action_type is ActionType.NAVIGATE:
        return (
            f"page.goto({json.dumps(values['url'])}, "
            "wait_until='domcontentloaded')"
        )
    key = {
        "ENTER": "Enter",
        "TAB": "Tab",
        "ESCAPE": "Escape",
        "ARROWDOWN": "ArrowDown",
        "ARROWUP": "ArrowUp",
        "SPACE": "Space",
        "ALT+LEFT": "Alt+ArrowLeft",
    }[str(values["key"])]
    return f"page.keyboard.press({json.dumps(key)})"


def _recovery_compatible_actions(
    controls: list[dict[str, JsonValue]],
) -> list[ConcreteAction]:
    actions: list[ConcreteAction] = []
    for index, control in enumerate(controls):
        raw_bbox = control["target_bbox"]
        assert isinstance(raw_bbox, list)
        bbox = tuple(float(item) for item in raw_bbox)
        x, y, width, height = bbox
        base = {
            "target_x": x + width / 2.0,
            "target_y": y + height / 2.0,
            "target_bbox": list(bbox),
        }
        tag = str(control["tag"])
        input_type = str(control["input_type"])
        types = [ActionType.CLICK]
        if tag in {"input", "textarea"} and input_type not in {
            "button", "checkbox", "radio", "submit", "reset", "file", "image",
        }:
            types.append(ActionType.TYPE)
        if tag == "select":
            types.append(ActionType.SELECT)
        for action_type in types:
            actions.append(
                ConcreteAction(
                    action_id=f"visible-control-{index}-{action_type.value}",
                    source_decision_id="browsergym-visible-target-evidence",
                    action_type=action_type,
                    parameters=base,
                    bbox=bbox,
                )
            )
        destination = control["destination"]
        if isinstance(destination, str) and destination:
            actions.append(
                ConcreteAction(
                    action_id=f"visible-destination-{index}",
                    source_decision_id="browsergym-visible-target-evidence",
                    action_type=ActionType.NAVIGATE,
                    parameters={"url": destination},
                )
            )
    actions.append(
        ConcreteAction(
            action_id="visible-viewport-scroll",
            source_decision_id="browsergym-visible-target-evidence",
            action_type=ActionType.SCROLL,
            parameters={"container": "viewport"},
        )
    )
    return actions


class ValidationDisabledBrowserGymEnvironment:
    """Environment object consumed by :class:`WebArenaAdapter`."""

    def __init__(
        self,
        *,
        task: TaskSpecification,
        seed: int,
        api: PinnedBrowserGymAPI,
        configuration: BrowserGymRuntimeConfiguration,
        service_url_map: Mapping[str, str],
        credential_bundle_root: Path,
        task_state_resetter: FrozenBrowserGymTaskStateResetter,
        runtime_page_publisher: RuntimePagePublisher,
        boundary_source_sha256: str,
    ) -> None:
        if type(task) is not TaskSpecification or task.runtime_start_state is None:
            raise BrowserGymWebArenaError(
                "BrowserGym environment requires a typed six-field start state"
            )
        if type(api) is not PinnedBrowserGymAPI:
            raise TypeError("BrowserGym environment requires the pinned API contract")
        if type(runtime_page_publisher) is not RuntimePagePublisher:
            raise TypeError("BrowserGym environment requires runtime-only page publisher")
        if type(seed) is not int or seed < 0:
            raise ValueError("BrowserGym environment seed must be nonnegative")
        self.task = task
        self.start_state = task.runtime_start_state
        self.expected_seed = seed
        self.api = api
        self.configuration = configuration
        self.service_url_map = dict(service_url_map)
        self.credential_bundle_root = credential_bundle_root.resolve()
        self.task_state_resetter = task_state_resetter
        self.publisher = runtime_page_publisher
        self.boundary_source_sha256 = _require_sha256(
            boundary_source_sha256,
            field_name="validation-disabled boundary source",
        )
        self.browser_env: Any | None = None
        self.reset_receipt: BrowserGymTaskStateResetReceipt | None = None
        self.storage_state_sha256: str | None = None
        self._sequence = 0
        self._last_action_error_kind: str | None = None
        self._published_episode_id: str | None = None
        self._session_id: str | None = None
        self._deferred_close_requested = False
        self._underlying_closed = False
        self._closed_page_state_sha256: str | None = None
        self._underlying_created = False
        self._cleanup_origin: str | None = None
        self._abort_requested = False

    def _close_underlying(self) -> None:
        if self._underlying_closed:
            raise BrowserGymWebArenaError(
                "underlying BrowserGym environment close may run exactly once"
            )
        self._underlying_closed = True
        self._cleanup_origin = "sealed_broker"
        environment = self.browser_env
        digest_error: BaseException | None = None
        if environment is not None:
            try:
                self._closed_page_state_sha256 = _page_state_digest(
                    environment.page
                )
            except BaseException as exc:
                digest_error = exc
        self.browser_env = None
        try:
            if environment is not None:
                result = environment.close()
                if result is not None:
                    raise TypeError("BrowserGym close must return None")
        except BaseException as close_error:
            if digest_error is not None:
                raise BrowserGymWebArenaError(
                    "BrowserGym page digest and close both failed"
                ) from close_error
            raise
        if digest_error is not None:
            raise BrowserGymWebArenaError(
                "BrowserGym final page state could not be committed before close"
            ) from digest_error

    def reset(self, *, seed: int) -> BrowserGymCausalRawObservation:
        if seed != self.expected_seed:
            raise BrowserGymWebArenaError("BrowserGym reset seed changed after factory binding")
        if self.browser_env is not None or self._underlying_created:
            raise BrowserGymWebArenaError("BrowserGym episode may reset exactly once")
        _validate_start_state_services(self.start_state, self.service_url_map)
        storage_path, storage_sha256 = _credential_storage_path(
            self.start_state,
            self.credential_bundle_root,
        )
        self.storage_state_sha256 = storage_sha256
        self.reset_receipt = self.task_state_resetter.reset(
            self.task.task_id,
            self.start_state,
            seed,
        )
        context_kwargs: dict[str, Any] = {}
        if storage_path is not None:
            context_kwargs["storage_state"] = str(storage_path)
        if self.start_state.geolocation is not None:
            context_kwargs["geolocation"] = self.start_state.geolocation.to_webarena_mapping()
        try:
            self.browser_env = self.api.browser_env_class(
                task_entrypoint=OracleBlindWebArenaTask,
                task_kwargs={
                    "task": self.task,
                    "start_state": self.start_state,
                    "configuration": self.configuration,
                },
                viewport={
                    "width": self.configuration.viewport_width,
                    "height": self.configuration.viewport_height,
                },
                slow_mo=self.configuration.slow_mo_ms,
                timeout=self.configuration.playwright_timeout_ms,
                locale=self.configuration.locale,
                timezone_id=self.configuration.timezone_id,
                tags_to_mark=self.configuration.tags_to_mark,
                headless=True,
                wait_for_user_message=False,
                terminate_on_infeasible=False,
                resizeable_window=False,
                record_video_dir=None,
                pw_context_kwargs=context_kwargs,
                action_mapping=None,
                use_raw_page_output=False,
                pre_observation_delay=(
                    self.configuration.pre_observation_delay_seconds
                ),
            )
            self._underlying_created = True
            reset_result = self.browser_env.reset(seed=seed)
            if type(reset_result) is not tuple or len(reset_result) != 2:
                raise BrowserGymWebArenaError(
                    "pinned BrowserGym reset must return (observation, info)"
                )
            if reset_result[0] is None:
                raise BrowserGymWebArenaError("BrowserGym reset returned no observation")
            return self._capture(page_settled=False)
        except BaseException:
            if self._underlying_created and not self._underlying_closed:
                self._cleanup_origin = "before_publication"
                try:
                    self._close_underlying()
                finally:
                    self._cleanup_origin = "before_publication"
            raise

    def _require_live_page(self) -> Any:
        if self.browser_env is None or self._underlying_closed:
            raise BrowserGymWebArenaError("BrowserGym live page is unavailable")
        page = getattr(self.browser_env, "page", None)
        if page is None:
            raise BrowserGymWebArenaError("BrowserGym environment has no active page")
        return page

    def _capture(self, *, page_settled: bool) -> BrowserGymCausalRawObservation:
        page = self._require_live_page()
        raw = page.evaluate(_CONTROL_SNAPSHOT_JAVASCRIPT)
        if not isinstance(raw, Mapping) or set(raw) != {"visible_text", "controls"}:
            raise BrowserGymWebArenaError("BrowserGym causal DOM snapshot schema changed")
        if type(raw["visible_text"]) is not str or not isinstance(raw["controls"], list):
            raise BrowserGymWebArenaError("BrowserGym causal DOM snapshot is malformed")
        controls = [_clean_control_snapshot(value) for value in raw["controls"]]
        page_state: dict[str, JsonValue] = {
            "schema_version": CAUSAL_OBSERVATION_SCHEMA,
            "visible_text": str(raw["visible_text"])[:8192],
            "visible_controls": controls,
            "has_browser_error": self._last_action_error_kind is not None,
            "browser_error_kind": self._last_action_error_kind,
        }
        assert_oracle_blind_mapping(page_state, location="browsergym_page_state")
        screenshot = page.screenshot(type="png")
        if not isinstance(screenshot, (bytes, bytearray)) or not screenshot:
            raise BrowserGymWebArenaError("BrowserGym page returned no PNG screenshot")
        viewport = getattr(page, "viewport_size", None)
        if not isinstance(viewport, Mapping):
            raise BrowserGymWebArenaError("BrowserGym page has no frozen viewport")
        width = viewport.get("width")
        height = viewport.get("height")
        if type(width) is not int or type(height) is not int:
            raise BrowserGymWebArenaError("BrowserGym viewport dimensions changed type")
        runtime_url = _runtime_visible_page_url(
            getattr(page, "url", ""),
            browser_error_kind=self._last_action_error_kind,
        )
        self._sequence += 1
        state_identity = {
            "task_id": self.task.task_id,
            "sequence": self._sequence,
            "url": runtime_url,
            "title": str(page.title()),
            "width": width,
            "height": height,
            "screenshot_sha256": hashlib.sha256(bytes(screenshot)).hexdigest(),
            "browser_state": page_state,
            "page_settled": page_settled,
            "environment_error": self._last_action_error_kind is not None,
        }
        return BrowserGymCausalRawObservation(
            task_id=self.task.task_id,
            task_goal=self.task.goal,
            sequence=self._sequence,
            url=state_identity["url"],
            title=state_identity["title"],
            width=width,
            height=height,
            screenshot_png=bytes(screenshot),
            browser_state=page_state,
            page_settled=page_settled,
            environment_error=self._last_action_error_kind is not None,
            browser_error_kind=self._last_action_error_kind,
            snapshot_sha256=canonical_sha256(state_identity),
        )

    def settle(
        self,
        raw_observation: BrowserGymCausalRawObservation,
        stage: ObservationStage,
        timeout_seconds: float,
        network_idle_required: bool,
    ) -> BrowserGymCausalRawObservation:
        if type(raw_observation) is not BrowserGymCausalRawObservation:
            raise BrowserGymWebArenaError("page settle received an unknown observation")
        if type(stage) is not ObservationStage:
            raise TypeError("page settle stage must use ObservationStage")
        page = self._require_live_page()
        timeout_ms = int(float(timeout_seconds) * 1000.0)
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        if network_idle_required:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        return self._capture(page_settled=True)

    def step(self, python_code: str) -> tuple[Any, Any, Any, Any, Any]:
        if type(python_code) is not str or not python_code.strip():
            raise BrowserGymWebArenaError("BrowserGym native action must be Python code")
        if self.browser_env is None or self._underlying_closed:
            raise BrowserGymWebArenaError("BrowserGym step lacks a live environment")
        environment = self.browser_env
        environment.last_action = python_code
        info, send_message, report_infeasible = environment.pre_step()
        try:
            self.api.execute_python_code(
                python_code,
                environment.page,
                send_message_to_user=send_message,
                report_infeasible_instructions=report_infeasible,
            )
            environment.last_action_error = ""
            self._last_action_error_kind = None
        except BaseException as exc:
            environment.last_action_error = f"{type(exc).__name__}: action execution failed"
            self._last_action_error_kind = _error_kind(exc)
        post_result = environment.post_step(info, validate=False)
        if type(post_result) is not tuple or len(post_result) != 5:
            raise BrowserGymWebArenaError(
                "pinned BrowserGym post_step must return its five-slot contract"
            )
        if post_result[0] is None:
            raise BrowserGymWebArenaError("BrowserGym post_step returned no observation")
        snapshot = self._capture(page_settled=False)
        # Never forward or inspect post_result[1:]: even validation-disabled
        # defaults are kept outside the runtime callback graph.
        return snapshot, None, None, None, None

    def publish_for_observation(self, episode_id: str) -> None:
        if self._published_episode_id is not None:
            if self._published_episode_id != episode_id:
                raise BrowserGymWebArenaError("live page was rebound to another episode")
            return
        if self.reset_receipt is None:
            raise BrowserGymWebArenaError("live page cannot publish before task-state reset")
        page = self._require_live_page()
        session_id = canonical_sha256(
            {
                "contract": "table2-live-page-session-v1",
                "episode_id": episode_id,
                "task_id": self.task.task_id,
                "start_state_sha256": self.start_state.start_state_sha256,
            }
        )
        boundary = BrowserGymValidationDisabledBoundary(
            boundary_id="pc01-browsergym-validation-disabled",
            boundary_version="v1",
            source_sha256=self.boundary_source_sha256,
        )
        start_receipt = BrowserGymStartStateApplicationReceipt(
            episode_id=episode_id,
            task_id=self.task.task_id,
            runtime_start_state=self.start_state,
            applied_start_state_sha256=self.start_state.start_state_sha256,
            applier_id="pc01-six-field-browsergym-start-state",
            applier_version="v1",
            applier_source_sha256=self.boundary_source_sha256,
        )
        self.publisher.publish(
            session_id=session_id,
            episode_id=episode_id,
            task_id=self.task.task_id,
            page=page,
            page_state_digest=_page_state_digest,
            close_live_page=self._close_underlying,
            validation_boundary=boundary,
            start_state_application=start_receipt,
        )
        self._session_id = session_id
        self._published_episode_id = episode_id

    def close(self) -> None:
        if self._underlying_closed:
            return None
        if self._published_episode_id is None or self._session_id is None:
            self._cleanup_origin = "before_publication"
            self._close_underlying()
            self._cleanup_origin = "before_publication"
            return None
        if self._deferred_close_requested:
            raise BrowserGymWebArenaError("BrowserGym adapter close requested twice")
        self.publisher.close(
            session_id=self._session_id,
            episode_id=self._published_episode_id,
            task_id=self.task.task_id,
        )
        self._deferred_close_requested = True
        return None

    def abort(self, *, episode_id: str, task_id: str) -> BrowserGymEpisodeAbortReceipt:
        if self._abort_requested:
            raise BrowserGymWebArenaError("BrowserGym orchestration abort requested twice")
        self._abort_requested = True
        if episode_id != (self._published_episode_id or episode_id) or task_id != self.task.task_id:
            raise BrowserGymWebArenaError("BrowserGym abort identity mismatch")
        if not self._underlying_created:
            return BrowserGymEpisodeAbortReceipt(
                episode_id=episode_id,
                task_id=task_id,
                outcome="NO_BROWSER_CREATED",
                underlying_browser_created=False,
                underlying_browser_close_called=False,
            )
        if self._underlying_closed:
            outcome = (
                "CLEANED_BEFORE_PUBLICATION"
                if self._cleanup_origin == "before_publication"
                else "ALREADY_CLEANED_BY_SEALED_BROKER"
            )
            return BrowserGymEpisodeAbortReceipt(
                episode_id=episode_id,
                task_id=task_id,
                outcome=outcome,
                underlying_browser_created=True,
                underlying_browser_close_called=True,
            )
        if self._session_id is None or self._published_episode_id is None:
            self._cleanup_origin = "before_publication"
            self._close_underlying()
            self._cleanup_origin = "before_publication"
            return BrowserGymEpisodeAbortReceipt(
                episode_id=episode_id,
                task_id=task_id,
                outcome="CLEANED_BEFORE_PUBLICATION",
                underlying_browser_created=True,
                underlying_browser_close_called=True,
            )
        broker_receipt = self.publisher.abort(
            session_id=self._session_id,
            episode_id=self._published_episode_id,
            task_id=self.task.task_id,
        )
        if type(broker_receipt) is not PageSessionAbortReceipt:
            raise BrowserGymWebArenaError("sealed broker returned wrong abort receipt")
        return BrowserGymEpisodeAbortReceipt(
            episode_id=episode_id,
            task_id=task_id,
            outcome="BROKER_ABORTED",
            underlying_browser_created=True,
            underlying_browser_close_called=True,
            broker_abort_receipt_sha256=broker_receipt.record_sha256,
        )

    def page_state_sha256(self) -> str:
        if self._underlying_closed:
            if self._closed_page_state_sha256 is None:
                raise BrowserGymWebArenaError(
                    "closed BrowserGym page lacks its final state commitment"
                )
            return self._closed_page_state_sha256
        return _page_state_digest(self._require_live_page())


def _page_state_digest(page: Any) -> str:
    content = page.content()
    context = getattr(page, "context", None)
    pages = getattr(context, "pages", ()) if context is not None else ()
    return canonical_sha256(
        {
            "url": str(getattr(page, "url", "")),
            "title": str(page.title()),
            "content_sha256": hashlib.sha256(str(content).encode("utf-8")).hexdigest(),
            "open_page_urls": [str(getattr(item, "url", "")) for item in pages],
        }
    )


class _ProcessBrokerManualRescueSidecar:
    """Single-writer append-only guard evidence owned by the browser child."""

    __slots__ = (
        "_closed",
        "_count",
        "_descriptor",
        "_file_identity",
        "_last_receipt_sha256",
        "_path",
        "_tail_sha256",
    )

    def __init__(self, episode_runtime_dir: Path) -> None:
        runtime_dir = Path(episode_runtime_dir).absolute()
        if (
            not runtime_dir.is_dir()
            or runtime_dir.is_symlink()
            or runtime_dir.resolve(strict=True) != runtime_dir
        ):
            raise BrowserGymWebArenaError(
                "process-broker episode runtime directory is not canonical"
            )
        path = runtime_dir / PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_FILENAME
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise BrowserGymWebArenaError(
                "child manual-rescue sidecar must be a fresh file"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise BrowserGymWebArenaError(
                "child manual-rescue sidecar is not a single-link regular file"
            )
        self._path = path
        self._descriptor = descriptor
        self._file_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
        )
        self._count = 0
        self._tail_sha256: str | None = None
        self._last_receipt_sha256 = "0" * 64
        self._closed = False

    def _validate_open_file(self) -> None:
        if self._closed:
            raise BrowserGymWebArenaError(
                "child manual-rescue sidecar is already closed"
            )
        try:
            opened = os.fstat(self._descriptor)
            rebound = self._path.lstat()
        except OSError as exc:
            raise BrowserGymWebArenaError(
                "child manual-rescue sidecar identity is unavailable"
            ) from exc
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
        )
        rebound_identity = (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_mode,
            rebound.st_nlink,
        )
        if (
            opened_identity != self._file_identity
            or rebound_identity != self._file_identity
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise BrowserGymWebArenaError(
                "child manual-rescue sidecar pathname changed"
            )

    def append(
        self,
        check: WebArenaManualRescueCheck,
        receipt: WebArenaManualRescueReceipt,
    ) -> None:
        self._validate_open_file()
        if type(check) is not WebArenaManualRescueCheck or type(receipt) is not (
            WebArenaManualRescueReceipt
        ):
            raise BrowserGymWebArenaError(
                "child manual-rescue sidecar requires typed check and receipt"
            )
        expected_check_index = self._count + 1
        if (
            check.check_index != expected_check_index
            or receipt.check_index != expected_check_index
            or check.previous_receipt_sha256 != self._last_receipt_sha256
            or receipt.check_sha256 != check.record_sha256
            or receipt.stage != check.stage
            or receipt.guard_id != check.guard_id
            or receipt.guard_version != check.guard_version
            or receipt.evidence_mode != check.evidence_mode
        ):
            raise BrowserGymWebArenaError(
                "child manual-rescue evidence is out of order or misbound"
            )
        body = {
            "schema_version": PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_SCHEMA_VERSION,
            "record_type": PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_RECORD_TYPE,
            "record_index": expected_check_index,
            "previous_record_sha256": self._tail_sha256 or ("0" * 64),
            "check": check.to_dict(),
            "check_sha256": check.record_sha256,
            "receipt": receipt.to_dict(),
            "receipt_sha256": receipt.record_sha256,
        }
        record_sha256 = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        encoded = canonical_json_bytes(
            {**body, "record_sha256": record_sha256}
        ) + b"\n"
        view = memoryview(encoded)
        while view:
            written = os.write(self._descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file write contract
                raise BrowserGymWebArenaError(
                    "child manual-rescue sidecar append made no progress"
                )
            view = view[written:]
        os.fsync(self._descriptor)
        self._count = expected_check_index
        self._tail_sha256 = record_sha256
        self._last_receipt_sha256 = receipt.record_sha256

    def identity(self) -> dict[str, Any]:
        self._validate_open_file()
        os.fsync(self._descriptor)
        current_offset = os.lseek(self._descriptor, 0, os.SEEK_CUR)
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        try:
            while True:
                block = os.read(self._descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        finally:
            os.lseek(self._descriptor, current_offset, os.SEEK_SET)
        self._validate_open_file()
        return {
            "schema_version": (
                PROCESS_BROKER_MANUAL_RESCUE_IDENTITY_SCHEMA_VERSION
            ),
            "record_type": PROCESS_BROKER_MANUAL_RESCUE_IDENTITY_RECORD_TYPE,
            "relative_path": PROCESS_BROKER_MANUAL_RESCUE_SIDECAR_FILENAME,
            "record_count": self._count,
            "tail_sha256": self._tail_sha256,
            "content_sha256": digest.hexdigest(),
        }

    def close(self) -> None:
        if self._closed:
            return None
        self._validate_open_file()
        os.fsync(self._descriptor)
        os.close(self._descriptor)
        self._closed = True
        return None


class _ProcessBrokerSealedCallbackGuardSidecar:
    """Append-only before/after digest evidence for child sealed callbacks."""

    __slots__ = (
        "_closed",
        "_count",
        "_descriptor",
        "_file_identity",
        "_path",
        "_tail_sha256",
    )

    def __init__(self, episode_runtime_dir: Path) -> None:
        runtime_dir = Path(episode_runtime_dir).absolute()
        if (
            not runtime_dir.is_dir()
            or runtime_dir.is_symlink()
            or runtime_dir.resolve(strict=True) != runtime_dir
        ):
            raise BrowserGymWebArenaError(
                "process-broker episode runtime directory is not canonical"
            )
        path = runtime_dir / PROCESS_BROKER_SEALED_CALLBACK_GUARD_SIDECAR_FILENAME
        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise BrowserGymWebArenaError(
                "child sealed-callback guard sidecar must be a fresh file"
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise BrowserGymWebArenaError(
                "child sealed-callback guard sidecar is not a single-link file"
            )
        self._path = path
        self._descriptor = descriptor
        self._file_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
        )
        self._count = 0
        self._tail_sha256: str | None = None
        self._closed = False

    def _validate_open_file(self) -> None:
        if self._closed:
            raise BrowserGymWebArenaError(
                "child sealed-callback guard sidecar is already closed"
            )
        try:
            opened = os.fstat(self._descriptor)
            rebound = self._path.lstat()
        except OSError as exc:
            raise BrowserGymWebArenaError(
                "child sealed-callback guard sidecar identity is unavailable"
            ) from exc
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
        )
        rebound_identity = (
            rebound.st_dev,
            rebound.st_ino,
            rebound.st_mode,
            rebound.st_nlink,
        )
        if (
            opened_identity != self._file_identity
            or rebound_identity != self._file_identity
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise BrowserGymWebArenaError(
                "child sealed-callback guard sidecar pathname changed"
            )

    def append(
        self,
        *,
        callback_kind: str,
        environment_state_sha256_before: str,
        environment_state_sha256_after: str,
        digester_id: str,
        digester_version: str,
    ) -> None:
        self._validate_open_file()
        if callback_kind not in {"sealed_transition", "sealed_finalizer"}:
            raise BrowserGymWebArenaError(
                "sealed callback guard kind is unregistered"
            )
        for label, value in (
            ("before", environment_state_sha256_before),
            ("after", environment_state_sha256_after),
        ):
            _require_sha256(value, field_name=f"sealed callback guard {label}")
        if environment_state_sha256_before != environment_state_sha256_after:
            raise BrowserGymWebArenaError(
                "mutating sealed callback cannot be recorded as unchanged"
            )
        if (
            type(digester_id) is not str
            or not digester_id.strip()
            or type(digester_version) is not str
            or not digester_version.strip()
        ):
            raise BrowserGymWebArenaError(
                "sealed callback guard lacks the registered digester identity"
            )
        next_index = self._count + 1
        body = {
            "schema_version": PROCESS_BROKER_SEALED_CALLBACK_GUARD_SCHEMA_VERSION,
            "record_type": PROCESS_BROKER_SEALED_CALLBACK_GUARD_RECORD_TYPE,
            "callback_guard_index": next_index,
            "previous_record_sha256": self._tail_sha256 or ("0" * 64),
            "callback_kind": callback_kind,
            "environment_state_digester_id": digester_id,
            "environment_state_digester_version": digester_version,
            "environment_state_sha256_before": environment_state_sha256_before,
            "environment_state_sha256_after": environment_state_sha256_after,
            "environment_state_unchanged": True,
        }
        record_sha256 = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        encoded = canonical_json_bytes(
            {**body, "record_sha256": record_sha256}
        ) + b"\n"
        view = memoryview(encoded)
        while view:
            written = os.write(self._descriptor, view)
            if written <= 0:  # pragma: no cover - regular-file write contract
                raise BrowserGymWebArenaError(
                    "child sealed-callback guard append made no progress"
                )
            view = view[written:]
        os.fsync(self._descriptor)
        self._count = next_index
        self._tail_sha256 = record_sha256

    def identity(self) -> dict[str, Any]:
        self._validate_open_file()
        os.fsync(self._descriptor)
        current_offset = os.lseek(self._descriptor, 0, os.SEEK_CUR)
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        try:
            while True:
                block = os.read(self._descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        finally:
            os.lseek(self._descriptor, current_offset, os.SEEK_SET)
        self._validate_open_file()
        return {
            "schema_version": (
                PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_SCHEMA_VERSION
            ),
            "record_type": (
                PROCESS_BROKER_SEALED_CALLBACK_GUARD_IDENTITY_RECORD_TYPE
            ),
            "relative_path": PROCESS_BROKER_SEALED_CALLBACK_GUARD_SIDECAR_FILENAME,
            "record_count": self._count,
            "tail_sha256": self._tail_sha256,
            "content_sha256": digest.hexdigest(),
        }

    def close(self) -> None:
        if self._closed:
            return None
        self._validate_open_file()
        os.fsync(self._descriptor)
        os.close(self._descriptor)
        self._closed = True
        return None


class _ProcessBrokerTerminalSignalBridge:
    """Evaluator-free browser callback populated only by the sealed entrypoint."""

    __slots__ = ("_callback",)

    def __init__(self) -> None:
        self._callback: Callable[..., OpaqueTerminalSignal] | None = None

    def bind(self, callback: Callable[..., OpaqueTerminalSignal]) -> None:
        if not callable(callback):
            raise TypeError("process-broker terminal mapper must be callable")
        if self._callback is None:
            self._callback = callback
        elif self._callback is not callback:
            raise BrowserGymWebArenaError(
                "process-broker sealed evaluator changed within one episode"
            )

    def __call__(self, *args: Any, **kwargs: Any) -> OpaqueTerminalSignal:
        callback = self._callback
        if callback is None:
            raise BrowserGymWebArenaError(
                "process-broker sealed evaluator is not bound"
            )
        return callback(*args, **kwargs)


class _ProcessBrokerBrowserGymWebArenaAdapter(WebArenaAdapter):
    """Child-only adapter retaining the page until sealed finalization."""

    def __init__(
        self,
        *,
        task: TaskSpecification,
        callbacks: "_BrowserGymEpisodeCallbacks",
        runtime_factory: "BrowserGymWebArenaRuntimeFactory",
        episode_runtime_dir: Path,
    ) -> None:
        self._process_broker_sidecar = _ProcessBrokerManualRescueSidecar(
            episode_runtime_dir
        )
        self._process_broker_terminal_bridge = _ProcessBrokerTerminalSignalBridge()
        self._process_broker_abort_episode = callbacks.abort_episode
        self._process_broker_sealed_evaluator: Any | None = None
        self._process_broker_sealed_writer: Any | None = None
        self._process_broker_shutdown_called = False
        try:
            super().__init__(
                benchmark_version=runtime_factory.benchmark_version,
                adapter_id=runtime_factory.environment_adapter_id,
                adapter_version=runtime_factory.environment_adapter_version,
                dependency_module=REGISTERED_BROWSERGYM_DEPENDENCY_MODULE,
                environment_factory=callbacks.environment_factory,
                observation_mapper=callbacks.observation_mapper,
                action_mapper=callbacks.action_mapper,
                screenshot_bytes_provider=callbacks.screenshot_bytes_provider,
                episode_runtime_dir=episode_runtime_dir,
                terminal_signal_mapper=self._process_broker_terminal_bridge,
                environment_state_digester=(
                    FrozenWebArenaEnvironmentStateDigester(
                        digester_id=runtime_factory.environment_state_digester_id,
                        digester_version=(
                            runtime_factory.environment_state_digester_version
                        ),
                        benchmark_version=runtime_factory.benchmark_version,
                        callback=callbacks.environment_state_digest,
                    )
                ),
                infrastructure_fault_classifier=(
                    runtime_factory.infrastructure_fault_classifier
                ),
                reset_state_attester=runtime_factory.reset_state_attester,
                require_reset_state_receipt=True,
                page_settle_policy=FrozenWebArenaPageSettlePolicy(
                    policy_id=runtime_factory.page_settle_policy_id,
                    benchmark_version=runtime_factory.benchmark_version,
                    network_idle_required=(
                        runtime_factory.configuration.network_idle_required
                    ),
                    settle_timeout_seconds=(
                        runtime_factory.configuration.page_settle_timeout_seconds
                    ),
                    callback=callbacks.settle,
                ),
                require_page_settle_policy=True,
                action_safety_policy=runtime_factory.action_safety_policy,
                require_action_safety_policy=True,
                manual_rescue_guard=runtime_factory.manual_rescue_guard,
                manual_rescue_evidence_sink=self._process_broker_sidecar.append,
                require_manual_rescue_guard=True,
            )
        except BaseException:
            self._process_broker_sidecar.close()
            raise
        if task.record_sha256 != callbacks.task.record_sha256:
            self._process_broker_sidecar.close()
            raise BrowserGymWebArenaError(
                "process-broker adapter task changed during construction"
            )

    def process_broker_bind_terminal_signal_mapper(
        self,
        callback: Callable[..., OpaqueTerminalSignal],
    ) -> None:
        self._process_broker_terminal_bridge.bind(callback)

    def process_broker_bind_sealed_evaluator(
        self,
        *,
        evaluator: Any,
        evidence_writer: Any,
    ) -> None:
        """Bind evaluator-owned callbacks without importing evaluator code here."""

        terminal_mapper = getattr(evaluator, "terminal_signal_mapper", None)
        finalizer = getattr(evaluator, "finalize_episode_evidence", None)
        if not callable(terminal_mapper) or not callable(finalizer):
            raise BrowserGymWebArenaError(
                "process-broker sealed evaluator binding is malformed"
            )
        if self._process_broker_sealed_evaluator is None:
            self._process_broker_sealed_evaluator = evaluator
            self._process_broker_sealed_writer = evidence_writer
            self._process_broker_terminal_bridge.bind(terminal_mapper)
            return None
        if (
            self._process_broker_sealed_evaluator is not evaluator
            or self._process_broker_sealed_writer is not evidence_writer
        ):
            raise BrowserGymWebArenaError(
                "process-broker sealed evaluator changed within one episode"
            )
        return None

    def process_broker_sealed_evaluator(self, *, evidence_writer: Any) -> Any | None:
        if (
            self._process_broker_sealed_evaluator is not None
            and self._process_broker_sealed_writer is not evidence_writer
        ):
            raise BrowserGymWebArenaError(
                "process-broker sealed evaluator writer identity changed"
            )
        return self._process_broker_sealed_evaluator

    def process_broker_environment_state_sha256(self) -> str:
        environment = self._environment
        digester = self._environment_state_digester
        if environment is None or digester is None:
            raise BrowserGymWebArenaError(
                "process-broker adapter lacks a created browser environment"
            )
        return digester.digest(environment)

    def process_broker_manual_rescue_sidecar_identity(self) -> dict[str, Any]:
        return self._process_broker_sidecar.identity()

    def close(self) -> None:
        """Request sealed close but retain the state-digest capability."""

        if self._closed:
            return None
        environment = self._environment
        if environment is not None and hasattr(environment, "close"):
            result = environment.close()
            if result is not None:
                raise BrowserGymWebArenaError(
                    "process-broker BrowserGym close returned data"
                )
        self._closed = True
        return None

    def process_broker_close_browser(self) -> BrowserGymEpisodeAbortReceipt | None:
        if self._process_broker_shutdown_called:
            raise BrowserGymWebArenaError(
                "process-broker browser shutdown may run exactly once"
            )
        self._process_broker_shutdown_called = True
        episode_id = self._episode_id
        task = self._task
        try:
            if episode_id is None or task is None:
                return None
            receipt = self._process_broker_abort_episode(episode_id, task.task_id)
            if type(receipt) is not BrowserGymEpisodeAbortReceipt:
                raise BrowserGymWebArenaError(
                    "process-broker browser abort returned the wrong receipt"
                )
            return receipt
        finally:
            self._process_broker_sidecar.close()


@dataclass(slots=True)
class _BrowserGymEpisodeCallbacks:
    task: TaskSpecification
    api: PinnedBrowserGymAPI
    configuration: BrowserGymRuntimeConfiguration
    service_url_map: Mapping[str, str]
    credential_bundle_root: Path
    task_state_resetter: FrozenBrowserGymTaskStateResetter
    runtime_page_publisher: RuntimePagePublisher
    boundary_source_sha256: str
    environment: ValidationDisabledBrowserGymEnvironment | None = None

    def environment_factory(self, task: TaskSpecification, seed: int) -> Any:
        if type(task) is not TaskSpecification or task.record_sha256 != self.task.record_sha256:
            raise BrowserGymWebArenaError("BrowserGym factory task identity changed")
        if self.environment is not None:
            raise BrowserGymWebArenaError("BrowserGym episode environment already exists")
        self.environment = ValidationDisabledBrowserGymEnvironment(
            task=task,
            seed=seed,
            api=self.api,
            configuration=self.configuration,
            service_url_map=self.service_url_map,
            credential_bundle_root=self.credential_bundle_root,
            task_state_resetter=self.task_state_resetter,
            runtime_page_publisher=self.runtime_page_publisher,
            boundary_source_sha256=self.boundary_source_sha256,
        )
        return self.environment

    def observation_mapper(
        self,
        raw: Any,
        episode_id: str,
        stage: ObservationStage,
        prior_action_id: str | None,
    ) -> Observation:
        if type(raw) is not BrowserGymCausalRawObservation:
            raise BrowserGymWebArenaError("BrowserGym mapper received unknown observation")
        if raw.task_id != self.task.task_id or raw.task_goal != self.task.goal:
            raise BrowserGymWebArenaError("BrowserGym observation belongs to another task")
        if self.environment is None:
            raise BrowserGymWebArenaError("BrowserGym mapper lacks its environment")
        self.environment.publish_for_observation(episode_id)
        observation_id = (
            f"{episode_id}:{stage.value}:{raw.sequence}:{raw.snapshot_sha256[:16]}"
        )
        browser_state = json.loads(
            json.dumps(dict(raw.browser_state), ensure_ascii=False, allow_nan=False)
        )
        controls = browser_state["visible_controls"]
        if not isinstance(controls, list):
            raise BrowserGymWebArenaError("visible controls lost JSON-array form")
        select_controls = [
            {
                "target_bbox": list(control["target_bbox"]),
                "candidate_options": list(control["candidate_options"]),
            }
            for control in controls
            if control.get("tag") == "select" and control.get("candidate_options")
        ]
        browser_state["observable_select_controls"] = select_controls
        browser_state["recovery_target_evidence"] = build_recovery_target_evidence(
            task=self.task,
            observation_id=observation_id,
            compatible_actions=_recovery_compatible_actions(controls),
        )
        assert_oracle_blind_mapping(browser_state, location="browsergym_observation_mapper")
        return Observation(
            observation_id=observation_id,
            episode_id=episode_id,
            stage=stage,
            screenshot_sha256=hashlib.sha256(raw.screenshot_png).hexdigest(),
            width=raw.width,
            height=raw.height,
            url=raw.url,
            title=raw.title,
            page_state=browser_state,
            page_settled=raw.page_settled,
            environment_error=raw.environment_error,
            prior_action_id=prior_action_id,
        )

    def action_mapper(self, action: ConcreteAction) -> str:
        return _action_code(action)

    def screenshot_bytes_provider(
        self,
        raw: Any,
        observation: Observation,
    ) -> bytes:
        if type(raw) is not BrowserGymCausalRawObservation:
            raise BrowserGymWebArenaError("screenshot provider received unknown observation")
        digest = hashlib.sha256(raw.screenshot_png).hexdigest()
        if observation.screenshot_sha256 != digest:
            raise BrowserGymWebArenaError("screenshot observation binding changed")
        return raw.screenshot_png

    def environment_state_digest(self, environment: Any) -> str:
        if environment is not self.environment or not isinstance(
            environment, ValidationDisabledBrowserGymEnvironment
        ):
            raise BrowserGymWebArenaError("state digester received another environment")
        return environment.page_state_sha256()

    def settle(
        self,
        environment: Any,
        raw: Any,
        stage: ObservationStage,
        timeout_seconds: float,
        network_idle_required: bool,
    ) -> BrowserGymCausalRawObservation:
        if environment is not self.environment or not isinstance(
            environment, ValidationDisabledBrowserGymEnvironment
        ):
            raise BrowserGymWebArenaError("settle callback received another environment")
        return environment.settle(
            raw,
            stage,
            timeout_seconds,
            network_idle_required,
        )

    def abort_episode(
        self,
        episode_id: str,
        task_id: str,
    ) -> BrowserGymEpisodeAbortReceipt:
        if self.environment is None:
            return BrowserGymEpisodeAbortReceipt(
                episode_id=episode_id,
                task_id=task_id,
                outcome="NO_BROWSER_CREATED",
                underlying_browser_created=False,
                underlying_browser_close_called=False,
            )
        return self.environment.abort(episode_id=episode_id, task_id=task_id)


@dataclass(frozen=True, slots=True)
class BrowserGymWebArenaRuntimeFactory:
    """Build one exact ``WebArenaRuntimeBinding`` from measured prerequisites."""

    validated_preflight: ValidatedDeploymentPreflight
    credential_bundle_root: Path
    task_state_resetter: FrozenBrowserGymTaskStateResetter
    runtime_page_publisher: RuntimePagePublisher
    configuration: BrowserGymRuntimeConfiguration
    benchmark_version: str
    environment_adapter_id: str
    environment_adapter_version: str
    environment_state_digester_id: str
    environment_state_digester_version: str
    page_settle_policy_id: str
    infrastructure_fault_classifier: FrozenWebArenaInfrastructureFaultClassifier
    reset_state_attester: FrozenWebArenaResetStateAttester
    action_safety_policy: FrozenWebArenaActionSafetyPolicy
    manual_rescue_guard: FrozenWebArenaManualRescueGuard
    wrapper_source_sha256: str
    api_loader: Callable[[], PinnedBrowserGymAPI] = field(
        default=load_pinned_browsergym_api,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.validated_preflight) is not ValidatedDeploymentPreflight:
            raise TypeError("production BrowserGym factory requires validated preflight")
        if set(self.validated_preflight.service_url_map) != REGISTERED_SERVICE_KEYS:
            raise ValueError("validated BrowserGym preflight lacks seven services")
        if self.validated_preflight.evidence.get("status") != "PASS":
            raise ValueError("BrowserGym preflight did not pass")
        root = Path(self.credential_bundle_root).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("credential bundle root must be an existing non-symlink directory")
        object.__setattr__(self, "credential_bundle_root", root.resolve())
        if type(self.task_state_resetter) is not FrozenBrowserGymTaskStateResetter:
            raise TypeError("BrowserGym factory requires frozen task-state resetter")
        expected_url_map_sha256 = canonical_sha256(
            dict(self.validated_preflight.service_url_map)
        )
        if self.task_state_resetter.service_url_map_sha256 != expected_url_map_sha256:
            raise ValueError("task-state resetter uses another service URL map")
        if type(self.runtime_page_publisher) is not RuntimePagePublisher:
            raise TypeError("BrowserGym factory requires runtime-only page publisher")
        if type(self.configuration) is not BrowserGymRuntimeConfiguration:
            raise TypeError("BrowserGym factory requires frozen runtime configuration")
        if self.benchmark_version != REGISTERED_BROWSERGYM_VERSION:
            raise ValueError("BrowserGym benchmark version differs from registration")
        for name in (
            "environment_adapter_id",
            "environment_adapter_version",
            "environment_state_digester_id",
            "environment_state_digester_version",
            "page_settle_policy_id",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"BrowserGym factory requires {name}")
        for typed, expected in (
            (self.infrastructure_fault_classifier, FrozenWebArenaInfrastructureFaultClassifier),
            (self.reset_state_attester, FrozenWebArenaResetStateAttester),
            (self.action_safety_policy, FrozenWebArenaActionSafetyPolicy),
            (self.manual_rescue_guard, FrozenWebArenaManualRescueGuard),
        ):
            if type(typed) is not expected:
                raise TypeError("BrowserGym factory received a wrong policy contract")
        for policy in (
            self.infrastructure_fault_classifier,
            self.action_safety_policy,
            self.manual_rescue_guard,
        ):
            if policy.benchmark_version != self.benchmark_version:
                raise ValueError("BrowserGym policy benchmark version changed")
        _require_sha256(self.wrapper_source_sha256, field_name="BrowserGym wrapper source")
        if self.wrapper_source_sha256 != module_source_sha256():
            raise ValueError("BrowserGym wrapper differs from its frozen source hash")
        if self.api_loader is not load_pinned_browsergym_api:
            raise ValueError("production BrowserGym factory forbids alternate API loaders")

    def _episode_callbacks(
        self,
        task: TaskSpecification,
    ) -> _BrowserGymEpisodeCallbacks:
        if type(task) is not TaskSpecification or task.runtime_start_state is None:
            raise BrowserGymWebArenaError("BrowserGym runtime task lacks six-field state")
        if task.benchmark_id.lower() != "webarena":
            raise BrowserGymWebArenaError("BrowserGym runtime received non-WebArena task")
        if task.benchmark_version != self.benchmark_version:
            raise BrowserGymWebArenaError("BrowserGym runtime benchmark version changed")
        api = self.api_loader()
        if api.production_loader is not True:
            raise BrowserGymWebArenaError("production BrowserGym API was not source-loaded")
        return _BrowserGymEpisodeCallbacks(
            task=task,
            api=api,
            configuration=self.configuration,
            service_url_map=dict(self.validated_preflight.service_url_map),
            credential_bundle_root=self.credential_bundle_root,
            task_state_resetter=self.task_state_resetter,
            runtime_page_publisher=self.runtime_page_publisher,
            boundary_source_sha256=self.wrapper_source_sha256,
        )

    def __call__(self, task: TaskSpecification) -> Any:
        callbacks = self._episode_callbacks(task)
        # Local import preserves the dependency direction: importing benchmark
        # adapters never imports the production campaign runner.
        from web_agent.eval.table2.production_runner import WebArenaRuntimeBinding

        return WebArenaRuntimeBinding(
            benchmark_version=self.benchmark_version,
            environment_adapter_id=self.environment_adapter_id,
            environment_adapter_version=self.environment_adapter_version,
            dependency_module=REGISTERED_BROWSERGYM_DEPENDENCY_MODULE,
            environment_factory=callbacks.environment_factory,
            observation_mapper=callbacks.observation_mapper,
            action_mapper=callbacks.action_mapper,
            screenshot_bytes_provider=callbacks.screenshot_bytes_provider,
            environment_state_digester=FrozenWebArenaEnvironmentStateDigester(
                digester_id=self.environment_state_digester_id,
                digester_version=self.environment_state_digester_version,
                benchmark_version=self.benchmark_version,
                callback=callbacks.environment_state_digest,
            ),
            infrastructure_fault_classifier=self.infrastructure_fault_classifier,
            reset_state_attester=self.reset_state_attester,
            page_settle_policy=FrozenWebArenaPageSettlePolicy(
                policy_id=self.page_settle_policy_id,
                benchmark_version=self.benchmark_version,
                network_idle_required=self.configuration.network_idle_required,
                settle_timeout_seconds=self.configuration.page_settle_timeout_seconds,
                callback=callbacks.settle,
            ),
            action_safety_policy=self.action_safety_policy,
            manual_rescue_guard=self.manual_rescue_guard,
            abort_episode=callbacks.abort_episode,
            frozen=True,
        )

    def create_process_broker_environment_adapter(
        self,
        *,
        task: TaskSpecification,
        episode_runtime_dir: Path,
    ) -> WebArenaAdapter:
        """Build the evaluator-free child adapter used by the process broker."""

        callbacks = self._episode_callbacks(task)
        return _ProcessBrokerBrowserGymWebArenaAdapter(
            task=task,
            callbacks=callbacks,
            runtime_factory=self,
            episode_runtime_dir=Path(episode_runtime_dir),
        )


def _load_process_broker_browser_runtime_factory() -> BrowserGymWebArenaRuntimeFactory:
    """Fail closed until an operator supplies an externally validated bootstrap.

    Credential and service discovery is deliberately not implemented here.  A
    deployment integration may replace this narrow loader only before the
    source-attested entrypoint is invoked in its isolated child.
    """

    raise BrowserGymWebArenaError(
        "process-broker BrowserGym runtime bootstrap is not registered; "
        "validated preflight, credentials, resetter, policies, and page "
        "publisher must be supplied by the deployment integration"
    )


def create_environment_adapter(
    *,
    task: TaskSpecification,
    episode_runtime_dir: Path,
) -> WebArenaAdapter:
    """Source-attested child entrypoint with the broker's exact call shape."""

    factory = _load_process_broker_browser_runtime_factory()
    if type(factory) is not BrowserGymWebArenaRuntimeFactory:
        raise BrowserGymWebArenaError(
            "process-broker bootstrap returned the wrong BrowserGym factory type"
        )
    adapter = factory.create_process_broker_environment_adapter(
        task=task,
        episode_runtime_dir=episode_runtime_dir,
    )
    if not isinstance(adapter, WebArenaAdapter):
        raise BrowserGymWebArenaError(
            "process-broker bootstrap returned a non-WebArena adapter"
        )
    return adapter


assert REGISTERED_VALIDATION_DISABLED_EXECUTION_PATH == (
    "BrowserEnv.pre_step+execute_python_code+post_step(validate=False)"
)
