"""Measured host-compatibility gate for the pinned Table 2 WebArena stack."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from importlib import metadata
import platform
from pathlib import Path
import re
import sys
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from .common import SchemaError, read_json, sha256_bytes, sha256_file, sha256_json
from .webarena_export import (
    PINNED_BROWSERGYM_WEBARENA_VERSION,
    PINNED_LIBWEBARENA_VERSION,
    _registered_url_map,
)


PINNED_WEBARENA_PACKAGES = {
    "browsergym-core": PINNED_BROWSERGYM_WEBARENA_VERSION,
    "browsergym-webarena": PINNED_BROWSERGYM_WEBARENA_VERSION,
    "gymnasium": "1.0.0",
    "libwebarena": PINNED_LIBWEBARENA_VERSION,
    "playwright": "1.44.0",
}
PREFLIGHT_SCHEMA_VERSION = "table2-webarena-host-preflight-v1"
PREFLIGHT_RECORD_TYPE = "WebArenaHostCompatibilityPreflight"
PREFLIGHT_EVIDENCE_LABEL = "PRE_CAMPAIGN_COMPATIBILITY_ONLY"
PREFLIGHT_DEPLOYMENT_PASS = "SINGLE_DGX_COMPATIBLE"
PINNED_WEBARENA_SERVICE_URL_KEYS = frozenset(
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

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PackageVersionGetter = Callable[[str], str]
BrowserProbe = Callable[[], Mapping[str, Any]]
ServiceProbe = Callable[[str], Mapping[str, Any]]
LiveResetProbe = Callable[[int], Mapping[str, Any]]


def _validate_registered_task_indices(
    value: Sequence[int] | None,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    indices = tuple(value)
    if (
        len(indices) != 50
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise SchemaError(
            "preflight registered task indices must be exactly 50 unique "
            "nonnegative integers in tracked order"
        )
    return indices


def _probe_passed(value: Mapping[str, Any]) -> bool:
    return value.get("status") == "PASS"


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected, key=str)
        raise SchemaError(
            f"{context} fields are not the registered closure "
            f"(missing={missing}, extra={extra})"
        )


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise SchemaError(f"{context} must be one lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context} must be a nonempty string")
    return value


def registered_service_url_map(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate the seven BrowserGym deployment origins, not task tokens."""

    return _registered_url_map(
        value, required_tokens=PINNED_WEBARENA_SERVICE_URL_KEYS
    )


def load_service_url_map(path: str | Path) -> dict[str, str]:
    """Load a non-symlinked seven-origin BrowserGym deployment map."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_symlink():
        raise SchemaError("WebArena service URL-map file must not be a symlink")
    value = read_json(source)
    nested = value.get("service_url_map")
    if nested is not None and set(value) != {"service_url_map"}:
        raise SchemaError(
            "wrapped WebArena service URL-map may contain only service_url_map"
        )
    mapping = nested if nested is not None else value
    if not isinstance(mapping, Mapping):
        raise SchemaError("WebArena service URL-map must contain a JSON mapping")
    return registered_service_url_map(mapping)


def validate_webarena_host_preflight(
    evidence: Mapping[str, Any],
    *,
    service_url_map: Mapping[str, Any],
    expected_live_reset_task_index: int | None = None,
    registered_task_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate campaign-eligible host evidence without trusting PASS labels.

    The URL map is required so the aggregate map digest and every redacted
    service-origin digest are recomputed from the campaign's actual deployment
    configuration. A successful return proves host compatibility only; it is
    not task-result evidence and never authorizes evaluator access.
    """

    if not isinstance(evidence, Mapping):
        raise SchemaError("WebArena host preflight evidence must be a mapping")
    urls = registered_service_url_map(service_url_map)
    _require_exact_fields(
        evidence,
        {
            "schema_version",
            "record_type",
            "status",
            "campaign_eligible",
            "deployment_decision",
            "evidence_label",
            "paper_table_status",
            "host",
            "service_url_map_sha256",
            "checks",
            "package_check",
            "browser_check",
            "service_checks",
            "live_reset_check",
        },
        context="WebArena host preflight",
    )
    registered_scalars = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "record_type": PREFLIGHT_RECORD_TYPE,
        "status": "PASS",
        "campaign_eligible": True,
        "deployment_decision": PREFLIGHT_DEPLOYMENT_PASS,
        "evidence_label": PREFLIGHT_EVIDENCE_LABEL,
        "paper_table_status": "N/R",
        "service_url_map_sha256": sha256_json(urls),
    }
    for field, expected in registered_scalars.items():
        actual = evidence.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise SchemaError(
                f"WebArena host preflight {field} is not campaign-eligible: "
                f"{actual!r}"
            )

    host = evidence.get("host")
    if not isinstance(host, Mapping):
        raise SchemaError("WebArena host preflight host identity must be a mapping")
    _require_exact_fields(
        host,
        {
            "system",
            "release",
            "machine",
            "python_version",
            "python_executable_sha256",
        },
        context="preflight host identity",
    )
    for field in ("system", "release", "machine", "python_version"):
        _require_nonempty_string(host.get(field), context=f"preflight host {field}")
    _require_sha256(
        host.get("python_executable_sha256"),
        context="preflight host python_executable_sha256",
    )

    checks = evidence.get("checks")
    if not isinstance(checks, Mapping):
        raise SchemaError("WebArena host preflight checks must be a mapping")
    expected_checks = {
        "packages": True,
        "chromium": True,
        "services": True,
        "live_reset": True,
    }
    _require_exact_fields(checks, set(expected_checks), context="preflight checks")
    for field in expected_checks:
        if type(checks.get(field)) is not bool or checks.get(field) is not True:
            raise SchemaError(f"WebArena host preflight check {field} did not pass")

    package_check = evidence.get("package_check")
    if not isinstance(package_check, Mapping):
        raise SchemaError("preflight package_check must be a mapping")
    _require_exact_fields(
        package_check, {"status", "packages"}, context="preflight package_check"
    )
    if package_check.get("status") != "PASS":
        raise SchemaError("preflight package check did not pass")
    package_rows = package_check.get("packages")
    if not isinstance(package_rows, list) or len(package_rows) != len(
        PINNED_WEBARENA_PACKAGES
    ):
        raise SchemaError("preflight package rows do not cover the pinned stack")
    by_distribution: dict[str, Mapping[str, Any]] = {}
    for row in package_rows:
        if not isinstance(row, Mapping):
            raise SchemaError("preflight package row must be a mapping")
        _require_exact_fields(
            row,
            {"distribution", "expected_version", "actual_version", "status"},
            context="preflight package row",
        )
        distribution = row.get("distribution")
        if not isinstance(distribution, str) or distribution in by_distribution:
            raise SchemaError("preflight package distributions must be unique strings")
        by_distribution[distribution] = row
    if set(by_distribution) != set(PINNED_WEBARENA_PACKAGES):
        raise SchemaError("preflight package identities differ from the pinned stack")
    for distribution, expected_version in PINNED_WEBARENA_PACKAGES.items():
        row = by_distribution[distribution]
        if (
            row.get("status") != "PASS"
            or row.get("expected_version") != expected_version
            or row.get("actual_version") != expected_version
        ):
            raise SchemaError(
                f"preflight package {distribution} does not match pinned version "
                f"{expected_version}"
            )

    browser = evidence.get("browser_check")
    if not isinstance(browser, Mapping):
        raise SchemaError("preflight browser_check must be a mapping")
    _require_exact_fields(
        browser,
        {
            "status",
            "browser",
            "browser_version",
            "viewport",
            "device_scale_factor",
            "screenshot_sha256",
        },
        context="preflight browser_check",
    )
    if browser.get("status") != "PASS" or browser.get("browser") != "chromium":
        raise SchemaError("preflight did not pass with Chromium")
    _require_nonempty_string(
        browser.get("browser_version"), context="preflight Chromium version"
    )
    if browser.get("viewport") != {"width": 1280, "height": 720}:
        raise SchemaError("preflight Chromium viewport differs from 1280x720")
    if type(browser.get("device_scale_factor")) is not int or browser.get(
        "device_scale_factor"
    ) != 1:
        raise SchemaError("preflight Chromium device scale factor must equal 1")
    _require_sha256(
        browser.get("screenshot_sha256"), context="preflight screenshot_sha256"
    )

    service_rows = evidence.get("service_checks")
    if not isinstance(service_rows, list) or len(service_rows) != len(urls):
        raise SchemaError("preflight service checks do not cover every registered site")
    by_token: dict[str, Mapping[str, Any]] = {}
    for row in service_rows:
        if not isinstance(row, Mapping):
            raise SchemaError("preflight service row must be a mapping")
        _require_exact_fields(
            row,
            {"status", "http_status", "service_key", "origin_sha256"},
            context="preflight service row",
        )
        token = row.get("service_key")
        if not isinstance(token, str) or token in by_token:
            raise SchemaError("preflight service tokens must be unique strings")
        by_token[token] = row
    if set(by_token) != set(urls):
        raise SchemaError("preflight service identities differ from the URL map")
    for token, url in urls.items():
        row = by_token[token]
        status_code = row.get("http_status")
        if (
            row.get("status") != "PASS"
            or type(status_code) is not int
            or not 200 <= status_code < 400
            or row.get("origin_sha256") != sha256_json(url)
        ):
            raise SchemaError(f"preflight service check is invalid for {token}")
        _require_sha256(
            row.get("origin_sha256"),
            context=f"preflight service {token} origin_sha256",
        )

    live_reset = evidence.get("live_reset_check")
    if not isinstance(live_reset, Mapping):
        raise SchemaError("preflight live_reset_check must be a mapping")
    _require_exact_fields(
        live_reset,
        {
            "status",
            "task_index",
            "seed",
            "goal_sha256",
            "current_url_sha256",
            "screenshot_shape",
            "observation_keys",
            "action_taken",
            "reward_read",
            "evaluator_output_read",
        },
        context="preflight live_reset_check",
    )
    registered_indices = _validate_registered_task_indices(
        registered_task_indices
    )
    task_index = live_reset.get("task_index")
    if type(task_index) is not int or task_index < 0:
        raise SchemaError("preflight live-reset task index is invalid")
    if registered_indices is not None and task_index not in registered_indices:
        raise SchemaError(
            "preflight live-reset task is absent from the tracked public registry"
        )
    # Preserve the original standalone 0--49 guard for callers that supply
    # neither an exact registry nor an externally bound expected task.  Active
    # campaign paths always supply one of those authorities.
    if (
        registered_indices is None
        and expected_live_reset_task_index is None
        and task_index not in range(50)
    ):
        raise SchemaError("preflight live-reset task requires registry authority")
    if (
        expected_live_reset_task_index is not None
        and task_index != expected_live_reset_task_index
    ):
        raise SchemaError("preflight live-reset task differs from the requested task")
    if (
        live_reset.get("status") != "PASS"
        or type(live_reset.get("seed")) is not int
        or live_reset.get("seed") != 42
    ):
        raise SchemaError("preflight live reset did not pass with registered seed 42")
    _require_sha256(live_reset.get("goal_sha256"), context="preflight goal_sha256")
    _require_sha256(
        live_reset.get("current_url_sha256"), context="preflight current_url_sha256"
    )
    shape = live_reset.get("screenshot_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
    ):
        raise SchemaError(
            "preflight live reset must contain a three-dimensional screenshot"
        )
    observation_keys = live_reset.get("observation_keys")
    if (
        not isinstance(observation_keys, list)
        or not observation_keys
        or any(not isinstance(key, str) or not key for key in observation_keys)
        or observation_keys != sorted(set(observation_keys))
        or not {"goal", "url", "screenshot"}.issubset(observation_keys)
    ):
        raise SchemaError("preflight live reset observation keys are invalid")
    for field in ("action_taken", "reward_read", "evaluator_output_read"):
        if (
            type(live_reset.get(field)) is not bool
            or live_reset.get(field) is not False
        ):
            raise SchemaError(f"preflight live reset violates no-{field} boundary")

    # Return detached evidence so callers cannot mutate a validated nested view.
    return deepcopy(dict(evidence))


def inspect_pinned_packages(
    version_getter: PackageVersionGetter = metadata.version,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for distribution, expected in sorted(PINNED_WEBARENA_PACKAGES.items()):
        try:
            actual = str(version_getter(distribution))
        except metadata.PackageNotFoundError:
            actual = None
        except Exception as exc:  # metadata backends can fail independently
            rows.append(
                {
                    "distribution": distribution,
                    "expected_version": expected,
                    "actual_version": None,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                }
            )
            continue
        rows.append(
            {
                "distribution": distribution,
                "expected_version": expected,
                "actual_version": actual,
                "status": "PASS" if actual == expected else "FAIL",
            }
        )
    return {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "packages": rows,
    }


def probe_chromium() -> dict[str, Any]:
    """Launch the pinned Playwright Chromium and capture a local-only screenshot."""

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    device_scale_factor=1,
                )
                page = context.new_page()
                page.set_content(
                    "<!doctype html><title>table2-preflight</title><p>ok</p>"
                )
                screenshot = page.screenshot(type="png")
                title = page.title()
                version = str(browser.version)
                context.close()
            finally:
                browser.close()
    except Exception as exc:
        return {"status": "FAIL", "error_type": type(exc).__name__}
    return {
        "status": "PASS" if title == "table2-preflight" and screenshot else "FAIL",
        "browser": "chromium",
        "browser_version": version,
        "viewport": {"width": 1280, "height": 720},
        "device_scale_factor": 1,
        "screenshot_sha256": sha256_bytes(bytes(screenshot)),
    }


def probe_service(base_url: str, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Check one WebArena origin without recording its URL or response content."""

    request = url_request.Request(
        base_url,
        method="GET",
        headers={"User-Agent": "webagent-table2-preflight/1"},
    )
    try:
        with url_request.urlopen(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
            response.read(1)
    except url_error.HTTPError as exc:
        status_code = int(exc.code)
    except Exception as exc:
        return {"status": "FAIL", "error_type": type(exc).__name__}
    return {
        "status": "PASS" if 200 <= status_code < 400 else "FAIL",
        "http_status": status_code,
    }


def probe_live_reset(task_index: int = 0) -> dict[str, Any]:
    """Reset one public task and close it without taking an action or reading reward."""

    if type(task_index) is not int or task_index < 0:
        raise SchemaError("live-reset preflight task index must be nonnegative")
    environment = None
    try:
        # Importing the benchmark registers its exact Gymnasium task IDs.
        import browsergym.webarena  # noqa: F401
        import gymnasium as gym

        environment = gym.make(
            f"browsergym/webarena.{task_index}",
            headless=True,
            viewport={"width": 1280, "height": 720},
        )
        observation, _information = environment.reset(seed=42)
        if not isinstance(observation, Mapping):
            raise TypeError("WebArena reset returned a non-mapping observation")
        goal = str(observation.get("goal") or "")
        current_url = str(observation.get("url") or "")
        screenshot = observation.get("screenshot")
        screenshot_shape = list(getattr(screenshot, "shape", ()))
        if not goal or not current_url or len(screenshot_shape) != 3:
            raise ValueError("WebArena reset lacked goal, URL, or screenshot")
        result = {
            "status": "PASS",
            "task_index": task_index,
            "seed": 42,
            "goal_sha256": sha256_bytes(goal.encode("utf-8")),
            "current_url_sha256": sha256_bytes(current_url.encode("utf-8")),
            "screenshot_shape": screenshot_shape,
            "observation_keys": sorted(str(key) for key in observation),
            "action_taken": False,
            "reward_read": False,
            "evaluator_output_read": False,
        }
    except Exception as exc:
        result = {"status": "FAIL", "error_type": type(exc).__name__}
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                if result.get("status") == "PASS":
                    result = {"status": "FAIL", "error_type": "EnvironmentCloseError"}
    return result


def run_webarena_host_preflight(
    *,
    service_url_map: Mapping[str, Any],
    version_getter: PackageVersionGetter = metadata.version,
    browser_probe: BrowserProbe = probe_chromium,
    service_probe: ServiceProbe = probe_service,
    live_reset_probe: LiveResetProbe = probe_live_reset,
    run_live_reset: bool = True,
    live_reset_task_index: int = 0,
    registered_task_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure whether one host can own both browser and model campaign planes."""

    urls = registered_service_url_map(service_url_map)
    registered_indices = _validate_registered_task_indices(
        registered_task_indices
    )
    if type(live_reset_task_index) is not int or live_reset_task_index < 0:
        raise SchemaError("live-reset preflight task index must be nonnegative")
    if (
        registered_indices is not None
        and live_reset_task_index not in registered_indices
    ):
        raise SchemaError(
            "live-reset preflight task is absent from the tracked public registry"
        )
    if registered_indices is None and live_reset_task_index not in range(50):
        raise SchemaError("live-reset preflight task requires registry authority")
    packages = inspect_pinned_packages(version_getter)
    browser = dict(browser_probe())
    if "status" not in browser:
        raise SchemaError("browser compatibility probe returned no status")

    services: list[dict[str, Any]] = []
    for token, url in sorted(urls.items()):
        result = dict(service_probe(url))
        if "status" not in result:
            raise SchemaError(f"service probe for {token} returned no status")
        # Bind evidence to the measured origin without writing deployment URLs.
        result.update({"service_key": token, "origin_sha256": sha256_json(url)})
        services.append(result)

    if run_live_reset:
        live_reset = dict(live_reset_probe(live_reset_task_index))
        if "status" not in live_reset:
            raise SchemaError("live-reset compatibility probe returned no status")
    else:
        live_reset = {
            "status": "NOT_RUN",
            "task_index": live_reset_task_index,
            "campaign_eligible": False,
        }

    checks = {
        "packages": _probe_passed(packages),
        "chromium": _probe_passed(browser),
        "services": all(_probe_passed(row) for row in services),
        "live_reset": _probe_passed(live_reset),
    }
    infrastructure_pass = all(
        checks[name] for name in ("packages", "chromium", "services")
    )
    all_pass = infrastructure_pass and checks["live_reset"]
    if all_pass:
        status = "PASS"
        deployment = "SINGLE_DGX_COMPATIBLE"
    elif not infrastructure_pass:
        status = "FAIL"
        deployment = "SPLIT_LOCAL_BROWSER_DGX_INFERENCE_REQUIRED"
    elif not run_live_reset:
        status = "INCOMPLETE"
        deployment = "UNDETERMINED_LIVE_RESET_REQUIRED"
    else:
        status = "FAIL"
        deployment = "SPLIT_LOCAL_BROWSER_DGX_INFERENCE_REQUIRED"
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "record_type": PREFLIGHT_RECORD_TYPE,
        "status": status,
        "campaign_eligible": all_pass,
        "deployment_decision": deployment,
        "evidence_label": PREFLIGHT_EVIDENCE_LABEL,
        "paper_table_status": "N/R",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable_sha256": sha256_file(sys.executable),
        },
        "service_url_map_sha256": sha256_json(urls),
        "checks": checks,
        "package_check": packages,
        "browser_check": browser,
        "service_checks": services,
        "live_reset_check": live_reset,
    }
    # A PASS report is a campaign authorization input. Re-derive every
    # registered claim rather than trusting injected probe status strings.
    if all_pass:
        validate_webarena_host_preflight(
            report,
            service_url_map=urls,
            expected_live_reset_task_index=live_reset_task_index,
            registered_task_indices=registered_indices,
        )
    return report
