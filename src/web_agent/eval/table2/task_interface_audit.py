"""Audit WebArena evaluator compatibility with the fixed PC-01 action interface.

The selected PC-01 policy can issue six concrete browser actions.  It cannot
emit WebArena's ``STOP`` action or attach an assistant answer to that action.
Consequently, official ``string_match`` tasks cannot be scored faithfully by
this runtime, while evaluator configurations composed solely of the known
page-state evaluators can be.  This module records that fact; it never changes
an evaluator, remaps an action, or removes a task.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from web_agent.runtime.contracts import ActionType

from .common import SchemaError, atomic_write_json, require_keys, sha256_json
from .webarena_export import (
    PUBLIC_PILOT_EXPORT_RECORD_TYPE,
    PUBLIC_PILOT_EXPORT_SCHEMA_VERSION,
)


TASK_INTERFACE_AUDIT_SCHEMA_VERSION = (
    "table2-webarena-task-interface-audit-v1"
)
TASK_INTERFACE_AUDIT_RECORD_TYPE = "WebArenaTaskActionInterfaceAudit"
TASK_INTERFACE_AUDIT_RELATIVE_PATH = "webarena_task_interface_audit.json"

PC01_ACTION_INTERFACE_ID = "pc01-p3-six-browser-actions-v1"
PC01_BROWSER_ACTIONS = (
    "CLICK",
    "TYPE",
    "SELECT",
    "SCROLL",
    "NAVIGATE",
    "PRESS_KEY",
)
ASSISTANT_ANSWER_EVALUATOR = "string_match"
PAGE_STATE_EVALUATORS = frozenset({"url_match", "program_html"})

_ANSWER_REQUIRED = "ASSISTANT_ANSWER_REQUIRED"
_PAGE_STATE_ONLY = "PAGE_STATE_ONLY"
_UNSUPPORTED = "UNSUPPORTED_EVALUATOR_TYPE"


def _action_interface_contract() -> dict[str, Any]:
    runtime_actions = tuple(action.value for action in ActionType)
    if runtime_actions != PC01_BROWSER_ACTIONS:
        raise SchemaError(
            "runtime ActionType no longer matches the registered PC-01 six-action "
            "interface; a new compatibility protocol is required"
        )
    return {
        "interface_id": PC01_ACTION_INTERFACE_ID,
        "action_classes": list(PC01_BROWSER_ACTIONS),
        "action_class_count": len(PC01_BROWSER_ACTIONS),
        "browser_actions_only": True,
        "stop_action_supported": False,
        "assistant_answer_submission_supported": False,
        "evaluator_remapping_permitted": False,
        "task_dropping_permitted": False,
    }


def _export_rows(task_export: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    require_keys(
        task_export,
        (
            "schema_version",
            "record_type",
            "snapshot_id",
            "benchmark",
            "benchmark_version",
            "task_definition_version",
            "source",
            "site_url_map_sha256",
            "resolved_task_set_sha256",
            "required_task_count",
            "tasks",
        ),
        context="WebArena task export",
    )
    if task_export.get("schema_version") != PUBLIC_PILOT_EXPORT_SCHEMA_VERSION:
        raise SchemaError("task-interface audit requires the registered task-export schema")
    if task_export.get("record_type") != PUBLIC_PILOT_EXPORT_RECORD_TYPE:
        raise SchemaError("task-interface audit requires the registered task-export type")
    if task_export.get("benchmark") != "webarena":
        raise SchemaError("task-interface audit supports WebArena task exports only")
    source = task_export.get("source")
    if not isinstance(source, Mapping) or not isinstance(
        source.get("task_source_sha256"), str
    ):
        raise SchemaError("task-interface audit requires authenticated task-source identity")
    rows = task_export.get("tasks")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise SchemaError("task-interface audit requires an array of task objects")
    if task_export.get("required_task_count") != 50 or len(rows) != 50:
        raise SchemaError("task-interface audit requires the exact public 0--49 task set")
    indices = [row.get("upstream_index") for row in rows]
    if indices != list(range(50)):
        raise SchemaError("task-interface audit requires tasks in exact index 0--49 order")
    if task_export.get("resolved_task_set_sha256") != sha256_json(rows):
        raise SchemaError("task-interface audit task-set hash does not match task rows")
    return rows


def _task_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    upstream_index = row.get("upstream_index")
    context = f"WebArena task {upstream_index}"
    require_keys(
        row,
        ("task_id", "upstream_index", "benchmark_task_id", "task_config", "evaluator"),
        context=context,
    )
    task_config = row.get("task_config")
    evaluator = row.get("evaluator")
    if not isinstance(task_config, Mapping) or not isinstance(evaluator, Mapping):
        raise SchemaError(f"{context} lacks task/evaluator mappings")
    evaluator_config = evaluator.get("config")
    task_eval = task_config.get("eval")
    if not isinstance(evaluator_config, Mapping) or not isinstance(task_eval, Mapping):
        raise SchemaError(f"{context} lacks official evaluator configuration")
    if dict(evaluator_config) != dict(task_eval):
        raise SchemaError(f"{context} evaluator config differs from task source")

    raw_eval_types = evaluator_config.get("eval_types")
    if (
        not isinstance(raw_eval_types, list)
        or not raw_eval_types
        or not all(isinstance(value, str) and value.strip() for value in raw_eval_types)
    ):
        raise SchemaError(f"{context} eval_types must be a non-empty string array")
    eval_types = [value.strip() for value in raw_eval_types]
    if len(eval_types) != len(set(eval_types)):
        raise SchemaError(f"{context} eval_types must not contain duplicates")

    answer_required = ASSISTANT_ANSWER_EVALUATOR in eval_types
    unknown = sorted(
        set(eval_types) - PAGE_STATE_EVALUATORS - {ASSISTANT_ANSWER_EVALUATOR}
    )
    reasons: list[str] = []
    if answer_required:
        reasons.append("ASSISTANT_ANSWER_SUBMISSION_UNSUPPORTED")
    if unknown:
        reasons.append("UNREGISTERED_EVALUATOR_TYPE")

    reference_answers = evaluator_config.get("reference_answers")
    answer_modes: list[str] = []
    if answer_required:
        if not isinstance(reference_answers, Mapping) or not reference_answers:
            reasons.append("MALFORMED_STRING_MATCH_REFERENCE_ANSWERS")
        else:
            answer_modes = sorted(str(key) for key in reference_answers)

    if answer_required:
        evaluator_mode = _ANSWER_REQUIRED
    elif unknown:
        evaluator_mode = _UNSUPPORTED
    else:
        evaluator_mode = _PAGE_STATE_ONLY
    compatible = not reasons and evaluator_mode == _PAGE_STATE_ONLY

    return {
        "task_id": str(row["task_id"]),
        "upstream_index": upstream_index,
        "benchmark_task_id": str(row["benchmark_task_id"]),
        "eval_types": eval_types,
        "evaluator_mode": evaluator_mode,
        "assistant_answer_required": answer_required,
        "answer_matching_modes": answer_modes,
        "compatible": compatible,
        "incompatibility_reasons": reasons,
    }


def build_webarena_task_interface_audit(
    task_export: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic per-task compatibility report.

    ``FAIL`` is a valid audit outcome.  It means the task set cannot enter the
    evaluation handoff under the current, fixed action interface.
    """

    if not isinstance(task_export, Mapping):
        raise SchemaError("task-interface audit input must be a JSON object")
    rows = _export_rows(task_export)
    task_results = [_task_audit(row) for row in rows]
    evaluator_type_counts: Counter[str] = Counter()
    answer_mode_counts: Counter[str] = Counter()
    for result in task_results:
        evaluator_type_counts.update(result["eval_types"])
        answer_mode_counts.update(result["answer_matching_modes"])

    compatible_count = sum(bool(row["compatible"]) for row in task_results)
    answer_required_count = sum(
        bool(row["assistant_answer_required"]) for row in task_results
    )
    unsupported_count = sum(
        "UNREGISTERED_EVALUATOR_TYPE" in row["incompatibility_reasons"]
        for row in task_results
    )
    fuzzy_count = sum(
        "fuzzy_match" in row["answer_matching_modes"] for row in task_results
    )
    incompatible_count = len(task_results) - compatible_count

    return {
        "schema_version": TASK_INTERFACE_AUDIT_SCHEMA_VERSION,
        "record_type": TASK_INTERFACE_AUDIT_RECORD_TYPE,
        "audit_scope": "EXACT_RESOLVED_WEBARENA_PUBLIC_INDICES_0_49",
        "status": "PASS" if incompatible_count == 0 else "FAIL",
        "handoff_eligible": incompatible_count == 0,
        "action_interface": _action_interface_contract(),
        "task_export_schema_version": task_export["schema_version"],
        "task_export_record_type": task_export["record_type"],
        "task_export_snapshot_id": task_export["snapshot_id"],
        "task_export_content_sha256": sha256_json(task_export),
        "resolved_task_set_sha256": task_export["resolved_task_set_sha256"],
        "task_source_sha256": task_export["source"]["task_source_sha256"],
        "site_url_map_sha256": task_export["site_url_map_sha256"],
        "task_count": len(task_results),
        "compatible_task_count": compatible_count,
        "incompatible_task_count": incompatible_count,
        "assistant_answer_required_task_count": answer_required_count,
        "page_state_only_task_count": sum(
            row["evaluator_mode"] == _PAGE_STATE_ONLY for row in task_results
        ),
        "unsupported_evaluator_task_count": unsupported_count,
        "fuzzy_string_match_task_count": fuzzy_count,
        "evaluator_type_counts": dict(sorted(evaluator_type_counts.items())),
        "answer_matching_mode_counts": dict(sorted(answer_mode_counts.items())),
        "tasks": task_results,
    }


def validate_webarena_task_interface_audit(
    value: Mapping[str, Any],
    *,
    task_export: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute an audit and require exact equality with submitted evidence."""

    if not isinstance(value, Mapping):
        raise SchemaError("WebArena task-interface audit must be a JSON object")
    expected = build_webarena_task_interface_audit(task_export)
    if dict(value) != expected:
        raise SchemaError(
            "WebArena task-interface audit differs from exact recomputation over "
            "the resolved task export"
        )
    return expected


def require_webarena_task_interface_compatible(
    audit: Mapping[str, Any],
) -> None:
    """Fail closed when any task needs an unsupported evaluator interaction."""

    if audit.get("status") != "PASS" or audit.get("handoff_eligible") is not True:
        incompatible = audit.get("incompatible_task_count")
        answer_required = audit.get("assistant_answer_required_task_count")
        raise SchemaError(
            "WebArena task/action-interface audit rejects evaluation handoff: "
            f"{incompatible} incompatible task(s), including {answer_required} "
            "task(s) requiring unsupported assistant-answer submission; do not "
            "remap evaluators or drop tasks"
        )


def write_webarena_task_interface_audit(
    output_path: str | Path,
    *,
    task_export: Mapping[str, Any],
) -> Path:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite WebArena task-interface audit: {destination}"
        )
    return atomic_write_json(
        destination, build_webarena_task_interface_audit(task_export)
    )
