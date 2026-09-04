#!/usr/bin/env python3
"""Generate the Table 2 page-state parity fixture from pinned WebArena.

This generator is deliberately independent of ``web_agent``.  It executes the
evaluator classes shipped in the exact pinned ``libwebarena==0.0.4`` wheel
against a real Chromium page, after proving that the imported evaluator source
has the same bytes as the wheel member.  The checked-in JSON is therefore a
replayable local compatibility reference, not a hand-authored expected-result
claim and not an external review of the Table 2 compatibility port.

Example (inside the pinned BrowserGym/WebArena environment)::

    python generate_webarena_page_state_reference.py \
      --wheel /path/to/libwebarena-0.0.4-py3-none-any.whl \
      --output webarena_page_state_reference.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib import metadata
import inspect
import json
import os
import platform
from pathlib import Path
import tempfile
from typing import Any
import zipfile


PINNED_LIBWEBARENA_VERSION = "0.0.4"
PINNED_PLAYWRIGHT_VERSION = "1.44.0"
PINNED_WHEEL_SHA256 = (
    "9ebee3b4371502c4f0f7e727a72e5846235d6750d420db9a3b8a168107654feb"
)
PINNED_MEMBERS = {
    "webarena/test.raw.json": (
        "7b50386fd69163dbc05d615d834df4c6ed2c35596e97a1b10d17451c02537652"
    ),
    "webarena/evaluation_harness/evaluators.py": (
        "eaa0532bcc97576b86fe3c666a0e538a8b0c41e5fa425edb6db2b32fb05a8186"
    ),
    "webarena/evaluation_harness/helper_functions.py": (
        "7ac7b1b7095aab758d93c3aa8a14769bb1fa5c8a94a3454066be624747834398"
    ),
}
SOURCE_TASK_INDICES = (369, 676, 704, 758)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _verified_wheel(wheel_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    wheel_bytes = wheel_path.read_bytes()
    actual_wheel_sha256 = _sha256_bytes(wheel_bytes)
    if actual_wheel_sha256 != PINNED_WHEEL_SHA256:
        raise RuntimeError(
            "libwebarena wheel identity mismatch: "
            f"expected {PINNED_WHEEL_SHA256}, observed {actual_wheel_sha256}"
        )
    with zipfile.ZipFile(wheel_path) as archive:
        member_hashes: dict[str, str] = {}
        member_bytes: dict[str, bytes] = {}
        for name, expected_sha256 in PINNED_MEMBERS.items():
            if archive.namelist().count(name) != 1:
                raise RuntimeError(f"wheel must contain exactly one {name}")
            payload = archive.read(name)
            actual_sha256 = _sha256_bytes(payload)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"pinned wheel member mismatch for {name}: "
                    f"expected {expected_sha256}, observed {actual_sha256}"
                )
            member_hashes[name] = actual_sha256
            member_bytes[name] = payload
    task_rows = json.loads(member_bytes["webarena/test.raw.json"])
    if not isinstance(task_rows, list) or len(task_rows) <= max(SOURCE_TASK_INDICES):
        raise RuntimeError("pinned WebArena task corpus has an unexpected shape")
    return task_rows, member_hashes


def _load_upstream() -> tuple[Any, Any, Any, Any, Path]:
    # libwebarena imports its environment URL table even for page-only rules.
    # Fixed non-routable examples satisfy that import guard without reading any
    # campaign credential or live service configuration.
    for key in (
        "REDDIT",
        "SHOPPING",
        "SHOPPING_ADMIN",
        "GITLAB",
        "WIKIPEDIA",
        "MAP",
        "HOMEPAGE",
    ):
        os.environ[key] = f"https://{key.casefold().replace('_', '-')}.invalid"

    from playwright.sync_api import sync_playwright
    from webarena.evaluation_harness import evaluators
    from webarena.evaluation_harness.helper_functions import PseudoPage

    evaluator_path = Path(inspect.getfile(evaluators)).resolve()
    actual_source_sha256 = _sha256_bytes(evaluator_path.read_bytes())
    expected_source_sha256 = PINNED_MEMBERS[
        "webarena/evaluation_harness/evaluators.py"
    ]
    if actual_source_sha256 != expected_source_sha256:
        raise RuntimeError(
            "imported WebArena evaluator source does not match the pinned wheel: "
            f"expected {expected_source_sha256}, observed {actual_source_sha256}"
        )
    if metadata.version("libwebarena") != PINNED_LIBWEBARENA_VERSION:
        raise RuntimeError("installed libwebarena version is not pinned")
    if metadata.version("playwright") != PINNED_PLAYWRIGHT_VERSION:
        raise RuntimeError("installed Playwright version is not pinned")
    return (
        evaluators,
        PseudoPage,
        sync_playwright,
        metadata,
        evaluator_path,
    )


def _resolved_copy(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolved_copy(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolved_copy(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for token, replacement in replacements.items():
            result = result.replace(token, replacement)
        return result
    return value


def _task_source(
    task_rows: list[dict[str, Any]],
    index: int,
    resolved_eval: dict[str, Any],
    *,
    note: str,
) -> dict[str, Any]:
    task_row = task_rows[index]
    return {
        "kind": "exact_pinned_task_eval",
        "task_index": index,
        "task_record_sha256": _canonical_sha256(task_row),
        "upstream_eval_config_sha256": _canonical_sha256(task_row["eval"]),
        "resolved_eval_config_sha256": _canonical_sha256(resolved_eval),
        "note": note,
    }


def _cases(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_369 = deepcopy(task_rows[369]["eval"])
    task_676 = _resolved_copy(
        task_rows[676]["eval"],
        {"__SHOPPING_ADMIN__": "https://admin.test"},
    )
    task_704 = _resolved_copy(
        task_rows[704]["eval"],
        {"__SHOPPING_ADMIN__": "https://admin.test"},
    )
    task_758 = deepcopy(task_rows[758]["eval"])
    return [
        {
            "case_id": "url_scheme_trailing_slash_and_extra_query",
            "source": {"kind": "synthetic_edge_case_specification_v1"},
            "config": {
                "eval_types": ["url_match"],
                "reference_answers": None,
                "reference_url": "http://shop.test/catalog/items/?q=usb+wifi",
                "program_html": [],
                "url_note": "GOLD in PRED",
            },
            "page": {
                "url": "https://shop.test/catalog/items/?q=usb+wifi&sort=price",
                "content": "<html></html>",
                "dom": {},
            },
        },
        {
            "case_id": "url_or_reference_query_union",
            "source": {"kind": "synthetic_edge_case_specification_v1"},
            "config": {
                "eval_types": ["url_match"],
                "reference_answers": None,
                "reference_url": (
                    "https://map.test/search?query=parking |OR| "
                    "https://map.test/search?query=garage"
                ),
                "program_html": [],
                "url_note": "GOLD in PRED",
            },
            "page": {
                "url": "https://map.test/search?query=garage",
                "content": "<html></html>",
                "dom": {},
            },
        },
        {
            "case_id": "url_missing_required_query_value",
            "source": {"kind": "synthetic_edge_case_specification_v1"},
            "config": {
                "eval_types": ["url_match"],
                "reference_answers": None,
                "reference_url": "https://shop.test/search?q=chairs&order=price",
                "program_html": [],
                "url_note": "GOLD in PRED",
            },
            "page": {
                "url": "https://shop.test/search?q=chairs&order=name",
                "content": "<html></html>",
                "dom": {},
            },
        },
        {
            "case_id": "url_base_substring_rule",
            "source": {"kind": "synthetic_edge_case_specification_v1"},
            "config": {
                "eval_types": ["url_match"],
                "reference_answers": None,
                "reference_url": "https://shop.test/catalog",
                "program_html": [],
                "url_note": "GOLD in PRED",
            },
            "page": {
                "url": "https://shop.test/catalogue/items",
                "content": "<html></html>",
                "dom": {},
            },
        },
        {
            "case_id": "exact_pinned_task_758_program_html",
            "source": _task_source(
                task_rows,
                758,
                task_758,
                note=(
                    "Includes the exact upstream route_from/route_to locators; "
                    "Chromium accepts their omitted closing square bracket."
                ),
            ),
            "config": task_758,
            "page": {
                "url": "https://map.test/directions",
                "content": (
                    "<div id='content'><select class='routing_engines'>"
                    "<option>car</option><option selected>walk</option>"
                    "</select><input name='route_from' value='New York, New York'>"
                    "<input name='route_to' "
                    "value='Portland, Cumberland County, Maine'></div>"
                ),
                "dom": {
                    "div#content select.routing_engines|selectedIndex": 1,
                    "[name=\"route_from\"|value": "New York, New York",
                    "[name=\"route_to\"|value": (
                        "Portland, Cumberland County, Maine"
                    ),
                },
            },
        },
        {
            "case_id": "exact_pinned_task_758_partial_failure",
            "source": _task_source(
                task_rows,
                758,
                task_758,
                note="Exact task evaluator with one deliberately unsatisfied target.",
            ),
            "config": deepcopy(task_758),
            "page": {
                "url": "https://map.test/directions",
                "content": (
                    "<div id='content'><select class='routing_engines'>"
                    "<option>car</option><option selected>walk</option>"
                    "</select><input name='route_from' value='New York, New York'>"
                    "<input name='route_to' value='Boston, Massachusetts'></div>"
                ),
                "dom": {
                    "div#content select.routing_engines|selectedIndex": 1,
                    "[name=\"route_from\"|value": "New York, New York",
                    "[name=\"route_to\"|value": "Boston, Massachusetts",
                },
            },
        },
        {
            "case_id": "exact_pinned_task_369_sidebar_outer_text",
            "source": _task_source(
                task_rows,
                369,
                task_369,
                note="Exact upstream page-state evaluator configuration.",
            ),
            "config": task_369,
            "page": {
                "url": "https://map.test/place/369",
                "content": "<aside id='sidebar_content'>Carnegie Music Hall</aside>",
                "dom": {
                    "[id=\"sidebar_content\"|outerText": "Carnegie Music Hall"
                },
            },
        },
        {
            "case_id": "resolved_pinned_task_676_url_and_filter",
            "source": _task_source(
                task_rows,
                676,
                task_676,
                note=(
                    "Exact upstream evaluator after deterministic replacement of "
                    "__SHOPPING_ADMIN__ with a credential-free fixture origin."
                ),
            ),
            "config": task_676,
            "page": {
                "url": "https://admin.test/sales/order/123",
                "content": (
                    "<div class='admin__data-grid-filters-current'>"
                    "Suspected Fraud</div>"
                ),
                "dom": {
                    "div.admin__data-grid-filters-current|outerText": (
                        "Suspected Fraud"
                    )
                },
            },
        },
        {
            "case_id": "resolved_pinned_task_704_report_dates",
            "source": _task_source(
                task_rows,
                704,
                task_704,
                note=(
                    "Exact upstream evaluator after deterministic replacement of "
                    "__SHOPPING_ADMIN__ with a credential-free fixture origin."
                ),
            ),
            "config": task_704,
            "page": {
                "url": "https://admin.test/reports/report_sales/sales",
                "content": (
                    "<input id='sales_report_from' value='2/1/23'>"
                    "<input id='sales_report_to' value='2/28/23'>"
                ),
                "dom": {
                    "[id=\"sales_report_from\"|value": "2/1/23",
                    "[id=\"sales_report_to\"|value": "2/28/23",
                },
            },
        },
        {
            "case_id": "html_or_must_include",
            "source": {"kind": "synthetic_edge_case_specification_v1"},
            "config": {
                "eval_types": ["program_html"],
                "reference_answers": None,
                "reference_url": None,
                "program_html": [
                    {
                        "url": "last",
                        "locator": (
                            "document.querySelector(\"main\").outerText"
                        ),
                        "required_contents": {
                            "must_include": ["Portland |OR| Augusta", "Maine"]
                        },
                    }
                ],
            },
            "page": {
                "url": "https://map.test/place",
                "content": "<main>Portland, Cumberland County, Maine</main>",
                "dom": {
                    "main|outerText": "Portland, Cumberland County, Maine"
                },
            },
        },
        {
            "case_id": "html_full_page_unescape_and_clean_quotes",
            "source": {"kind": "synthetic_edge_case_specification_v1"},
            "config": {
                "eval_types": ["program_html"],
                "reference_answers": None,
                "reference_url": "",
                "program_html": [
                    {
                        "url": "last",
                        "locator": "",
                        "required_contents": {"must_include": ["Tom & Jerry"]},
                    }
                ],
            },
            "page": {
                "url": "https://shop.test/item",
                "content": "<main>'Tom &amp; Jerry'</main>",
                "dom": {},
            },
        },
        {
            "case_id": "html_invalid_css_scores_false_upstream",
            "source": {
                "kind": "synthetic_edge_case_specification_v1",
                "note": (
                    "Pinned upstream catches the querySelector exception and "
                    "scores an empty selected value."
                ),
            },
            "config": {
                "eval_types": ["program_html"],
                "reference_answers": None,
                "reference_url": None,
                "program_html": [
                    {
                        "url": "last",
                        "locator": "document.querySelector(\"??\").outerText",
                        "required_contents": {"must_include": ["never"]},
                    }
                ],
            },
            "page": {
                "url": "https://shop.test/item",
                "content": "<main>never</main>",
                "dom": {},
            },
        },
    ]


def _write_config(config: dict[str, Any], directory: Path, name: str) -> Path:
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps({"intent": "local parity fixture", "eval": config}),
        encoding="utf-8",
    )
    return path


def _official_result(
    *,
    case: dict[str, Any],
    page: Any,
    evaluators: Any,
    pseudo_page_type: Any,
    config_directory: Path,
) -> tuple[list[bool], float]:
    config = case["config"]
    pseudo_page = pseudo_page_type(page, case["page"]["url"])
    before = (page.url, page.content())
    vector: list[bool] = []
    criterion_number = 0
    for eval_type in config["eval_types"]:
        if eval_type == "url_match":
            criterion_config = deepcopy(config)
            criterion_config["eval_types"] = ["url_match"]
            criterion_config["program_html"] = []
            path = _write_config(
                criterion_config,
                config_directory,
                f"{case['case_id']}-{criterion_number}",
            )
            score = evaluators.URLEvaluator()([], path, pseudo_page, None)
            vector.append(score == 1.0)
            criterion_number += 1
        elif eval_type == "program_html":
            for target in config["program_html"]:
                criterion_config = deepcopy(config)
                criterion_config["eval_types"] = ["program_html"]
                criterion_config["program_html"] = [deepcopy(target)]
                criterion_config.pop("url_note", None)
                path = _write_config(
                    criterion_config,
                    config_directory,
                    f"{case['case_id']}-{criterion_number}",
                )
                score = evaluators.HTMLContentEvaluator()(
                    [], path, pseudo_page, None
                )
                vector.append(score == 1.0)
                criterion_number += 1
        else:  # pragma: no cover - generator case invariant
            raise RuntimeError(f"unsupported generator mechanism: {eval_type}")

    full_path = _write_config(
        config,
        config_directory,
        f"{case['case_id']}-combined",
    )
    combined_score = evaluators.evaluator_router(full_path)(
        [], full_path, pseudo_page, None
    )
    after = (page.url, page.content())
    if after != before:
        raise RuntimeError("supported upstream evaluator unexpectedly mutated the page")
    if (combined_score == 1.0) != all(vector):
        raise RuntimeError("criterion vector does not reproduce upstream combined score")
    return vector, float(combined_score)


def generate(wheel_path: Path) -> dict[str, Any]:
    task_rows, member_hashes = _verified_wheel(wheel_path)
    evaluators, pseudo_page_type, sync_playwright, _, evaluator_path = (
        _load_upstream()
    )
    cases = _cases(task_rows)
    with sync_playwright() as playwright:
        chromium_executable = Path(playwright.chromium.executable_path).resolve()
        if not chromium_executable.is_file():
            raise RuntimeError("pinned Playwright Chromium executable is absent")
        chromium_executable_sha256 = _sha256_bytes(
            chromium_executable.read_bytes()
        )
        browser = playwright.chromium.launch(headless=True)
        try:
            browser_version = browser.version
            page = browser.new_page()
            with tempfile.TemporaryDirectory(prefix="table2-page-reference-") as raw:
                config_directory = Path(raw)
                for case in cases:
                    page.set_content(case["page"]["content"])
                    vector, combined_score = _official_result(
                        case=case,
                        page=page,
                        evaluators=evaluators,
                        pseudo_page_type=pseudo_page_type,
                        config_directory=config_directory,
                    )
                    case["expected_vector"] = vector
                    case["expected_combined_score"] = combined_score
        finally:
            browser.close()

    fixture: dict[str, Any] = {
        "schema_version": "table2-webarena-page-state-reference-v1",
        "classification": "generated_pinned_upstream_executable_reference",
        "implementation_classification": (
            "reviewed_compatibility_port_pending_external_review"
        ),
        "reference_role": (
            "local_compatibility_parity_only_not_external_review"
        ),
        "provenance": {
            "generator_source_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "python_version": platform.python_version(),
            "libwebarena_version": metadata.version("libwebarena"),
            "playwright_version": metadata.version("playwright"),
            "chromium_version": browser_version,
            "chromium_executable_name": chromium_executable.name,
            "chromium_executable_sha256": chromium_executable_sha256,
            "wheel_sha256": PINNED_WHEEL_SHA256,
            "wheel_member_sha256": member_hashes,
            "installed_evaluator_source_matches_wheel": True,
            "imported_evaluator_path_name": evaluator_path.name,
            "imported_evaluator_source_sha256": _sha256_bytes(
                evaluator_path.read_bytes()
            ),
        },
        "cases": cases,
    }
    fixture["payload_sha256"] = _canonical_sha256(fixture)
    return fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = generate(args.wheel.resolve())
    args.output.write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
