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

from .common import (
    SchemaError,
    atomic_write_json,
    require_keys,
    sha256_file,
    sha256_json,
)
from .webarena_export import (
    PUBLIC_PILOT_EXPORT_RECORD_TYPE,
    PUBLIC_PILOT_EXPORT_SCHEMA_VERSION,
)
from .public_task_registry import PUBLIC_DEVELOPMENT_TASK_COUNT


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
_AUDIT_SCOPE = "EXACT_ORDERED_TRACKED_WEBARENA_PUBLIC_DEVELOPMENT_REGISTRY"
PAGE_STATE_COMPILE_AUTHORITY_SCHEMA_VERSION = (
    "table2-webarena-page-state-compile-authority-v1"
)
PAGE_STATE_COMPILE_AUTHORITY_RECORD_TYPE = (
    "WebArenaPageStateEvaluatorCompileAuthority"
)
PAGE_STATE_COMPILE_CONTRACT = (
    "strict-read-only-page-state-config-compiler-exact-50-v1"
)


def _validate_ordered_task_identities(
    rows: list[Mapping[str, Any]], *, context: str
) -> None:
    """Reject incomplete or ambiguous order without assuming contiguous IDs."""

    if len(rows) != PUBLIC_DEVELOPMENT_TASK_COUNT:
        raise SchemaError(f"{context} requires exactly 50 registered tasks")
    indices: set[int] = set()
    task_ids: set[str] = set()
    benchmark_task_ids: set[str] = set()
    for position, row in enumerate(rows):
        upstream_index = row.get("upstream_index")
        task_id = row.get("task_id")
        benchmark_task_id = row.get("benchmark_task_id")
        if type(upstream_index) is not int or upstream_index < 0:
            raise SchemaError(
                f"{context} task {position} has an invalid upstream index"
            )
        if not isinstance(task_id, str) or not task_id.strip():
            raise SchemaError(f"{context} task {position} lacks a task ID")
        if not isinstance(benchmark_task_id, str) or not benchmark_task_id.strip():
            raise SchemaError(
                f"{context} task {position} lacks a benchmark task identity"
            )
        if (
            upstream_index in indices
            or task_id in task_ids
            or benchmark_task_id in benchmark_task_ids
        ):
            raise SchemaError(f"{context} repeats a stable task identity")
        indices.add(upstream_index)
        task_ids.add(task_id)
        benchmark_task_ids.add(benchmark_task_id)


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
    if task_export.get("required_task_count") != PUBLIC_DEVELOPMENT_TASK_COUNT:
        raise SchemaError(
            "task-interface audit requires exactly 50 registered public tasks"
        )
    _validate_ordered_task_identities(rows, context="task-interface audit")
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


def _compile_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the evaluator-relevant task bytes shared by export/snapshot rows."""

    return {
        key: row[key]
        for key in (
            "task_id",
            "upstream_index",
            "benchmark_task_id",
            "benchmark_task_version",
            "instruction",
            "start_state",
            "task_config",
            "evaluator",
        )
    }


def build_page_state_compile_authority(
    task_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compile the exact 50 page-state configs and bind compiler source bytes.

    This is the identity-neutral handoff binding.  When the rows use the
    repository's current pending evaluator identity, it also embeds the
    implementation-specific ``build_page_state_evaluator_compile_report``
    output and checks that both reports agree.  That nested report remains a
    local parity/review input, while this outer report binds the exact task
    projection across export, handoff, and freeze.  Neither report grants
    campaign readiness; an independently authored review receipt is separate
    and mandatory.
    """

    if not isinstance(task_rows, list) or not all(
        isinstance(row, Mapping) for row in task_rows
    ):
        raise SchemaError("page-state compile authority requires task mappings")
    _validate_ordered_task_identities(
        task_rows, context="page-state compile authority"
    )

    # Kept local to avoid making the evaluator/runtime dependency part of the
    # module import graph.  The source digest below makes the exact compiler
    # bytes part of the frozen scientific authority.
    from . import webarena_page_state_evaluator as compiler_module

    compiler_path = Path(__file__).resolve().with_name(
        "webarena_page_state_evaluator.py"
    )
    if Path(compiler_module.__file__).resolve() != compiler_path:
        raise SchemaError(
            "page-state compiler import does not resolve to the sibling "
            "attested source module"
        )
    compiled_rows: list[dict[str, Any]] = []
    evaluator_identities: set[tuple[str, str]] = set()
    criterion_count = 0
    projections: list[dict[str, Any]] = []
    for position, raw_row in enumerate(task_rows):
        row = dict(raw_row)
        audit_row = _task_audit(row)
        if not audit_row["compatible"]:
            raise SchemaError(
                "page-state compile authority cannot compile an interface-"
                f"incompatible task at position {position}"
            )
        evaluator = row.get("evaluator")
        task_config = row.get("task_config")
        if not isinstance(evaluator, Mapping) or not isinstance(
            task_config, Mapping
        ):
            raise SchemaError(
                f"page-state compile task {position} lacks evaluator content"
            )
        config = evaluator.get("config")
        if not isinstance(config, Mapping) or task_config.get("eval") != config:
            raise SchemaError(
                f"page-state compile task {position} evaluator config differs"
            )
        evaluator_id = evaluator.get("evaluator_id")
        evaluator_version = evaluator.get("evaluator_version")
        if (
            not isinstance(evaluator_id, str)
            or not evaluator_id.strip()
            or not isinstance(evaluator_version, str)
            or not evaluator_version.strip()
        ):
            raise SchemaError(
                f"page-state compile task {position} lacks evaluator identity"
            )
        evaluator_identities.add((evaluator_id, evaluator_version))
        try:
            compiled = compiler_module.compile_page_state_evaluator_config(config)
        except compiler_module.WebArenaPageStateEvaluatorError as exc:
            raise SchemaError(
                "page-state evaluator rejected exact task config at position "
                f"{position}: {exc}"
            ) from exc
        projection = _compile_projection(row)
        projections.append(projection)
        criterion_count += compiled.criterion_count
        compiled_rows.append(
            {
                "position": position,
                "task_id": str(row["task_id"]),
                "upstream_index": row["upstream_index"],
                "benchmark_task_id": str(row["benchmark_task_id"]),
                "task_projection_sha256": sha256_json(projection),
                "evaluator_config_sha256": compiled.config_sha256,
                "eval_types": list(compiled.eval_types),
                "criterion_count": compiled.criterion_count,
            }
        )
    if len(evaluator_identities) != 1:
        raise SchemaError(
            "page-state compile authority requires one evaluator identity"
        )
    evaluator_id, evaluator_version = next(iter(evaluator_identities))
    implementation_report: dict[str, Any] | None = None
    implementation_report_sha256: str | None = None
    implementation_report_status = "NOT_APPLICABLE_DIFFERENT_EVALUATOR_IDENTITY"
    if (
        evaluator_id == compiler_module.EVALUATOR_ID
        and evaluator_version == compiler_module.EVALUATOR_VERSION
    ):
        implementation_report = (
            compiler_module.build_page_state_evaluator_compile_report(projections)
        )
        compiler_module.validate_page_state_evaluator_compile_report(
            implementation_report,
            task_rows=projections,
        )
        implementation_rows = implementation_report.get("tasks")
        if not isinstance(implementation_rows, list) or len(
            implementation_rows
        ) != len(compiled_rows):
            raise SchemaError(
                "implementation-specific page-state compile report has an "
                "invalid task count"
            )
        for generic, specific in zip(
            compiled_rows, implementation_rows, strict=True
        ):
            for field in (
                "position",
                "task_id",
                "upstream_index",
                "benchmark_task_id",
                "evaluator_config_sha256",
                "eval_types",
                "criterion_count",
            ):
                if generic[field] != specific.get(field):
                    raise SchemaError(
                        "generic and implementation-specific compile reports "
                        f"disagree at {generic['task_id']}: {field}"
                    )
        implementation_report_sha256 = sha256_json(implementation_report)
        implementation_report_status = "BOUND_PENDING_EXTERNAL_REVIEW_INPUT"
    payload = {
        "schema_version": PAGE_STATE_COMPILE_AUTHORITY_SCHEMA_VERSION,
        "record_type": PAGE_STATE_COMPILE_AUTHORITY_RECORD_TYPE,
        "compile_contract": PAGE_STATE_COMPILE_CONTRACT,
        "status": "COMPILED",
        "all_exact_task_configs_compiled": True,
        "external_independent_review_required": True,
        "campaign_authority": False,
        "task_count": len(compiled_rows),
        "criterion_count": criterion_count,
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "compiler_source_relative_path": (
            "src/web_agent/eval/table2/webarena_page_state_evaluator.py"
        ),
        "compiler_source_sha256": sha256_file(compiler_path),
        "exact_task_projection_sha256": sha256_json(projections),
        "implementation_specific_report_status": implementation_report_status,
        "implementation_specific_compile_report": implementation_report,
        "implementation_specific_compile_report_content_sha256": (
            implementation_report_sha256
        ),
        "tasks": compiled_rows,
    }
    payload["report_sha256"] = sha256_json(payload)
    return payload


def validate_page_state_compile_authority(
    value: Mapping[str, Any],
    *,
    task_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompile every exact row and require byte-equivalent authority."""

    if not isinstance(value, Mapping):
        raise SchemaError("page-state compile authority must be a mapping")
    expected = build_page_state_compile_authority(task_rows)
    if dict(value) != expected:
        raise SchemaError(
            "page-state compile authority differs from exact 50-task "
            "recompilation"
        )
    return expected


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
        "audit_scope": _AUDIT_SCOPE,
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


def validate_resolved_task_interface_binding(
    resolved_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute interface compatibility from a frozen resolved snapshot.

    This is the campaign-freeze side of the task-source handoff. It prevents a
    lower-level freezer caller from omitting, replacing, or falsifying the
    six-action compatibility result while presenting an otherwise consistent
    task snapshot.
    """

    if not isinstance(resolved_snapshot, Mapping):
        raise SchemaError("resolved task snapshot must be a mapping")
    require_keys(
        resolved_snapshot,
        (
            "snapshot_id",
            "upstream_export_schema_version",
            "upstream_export_record_type",
            "upstream_export_content_sha256",
            "upstream_task_source",
            "site_url_map_sha256",
            "resolved_task_set_sha256",
            "task_action_interface_audit",
            "task_action_interface_audit_content_sha256",
            "page_state_evaluator_compile_report",
            "page_state_evaluator_compile_report_content_sha256",
            "tasks",
        ),
        context="resolved task interface binding",
    )
    rows = resolved_snapshot.get("tasks")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise SchemaError("resolved task interface binding requires task mappings")
    _validate_ordered_task_identities(
        rows, context="resolved task interface binding"
    )
    audit = resolved_snapshot.get("task_action_interface_audit")
    if not isinstance(audit, Mapping):
        raise SchemaError("resolved task snapshot lacks its interface audit")

    task_results = [_task_audit(row) for row in rows]
    evaluator_type_counts: Counter[str] = Counter()
    answer_mode_counts: Counter[str] = Counter()
    for result in task_results:
        evaluator_type_counts.update(result["eval_types"])
        answer_mode_counts.update(result["answer_matching_modes"])
    compatible_count = sum(bool(row["compatible"]) for row in task_results)
    incompatible_count = len(task_results) - compatible_count
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
    source = resolved_snapshot.get("upstream_task_source")
    if not isinstance(source, Mapping):
        raise SchemaError("resolved task snapshot lacks upstream task-source identity")
    expected = {
        "schema_version": TASK_INTERFACE_AUDIT_SCHEMA_VERSION,
        "record_type": TASK_INTERFACE_AUDIT_RECORD_TYPE,
        "audit_scope": _AUDIT_SCOPE,
        "status": "PASS" if incompatible_count == 0 else "FAIL",
        "handoff_eligible": incompatible_count == 0,
        "action_interface": _action_interface_contract(),
        "task_export_schema_version": resolved_snapshot[
            "upstream_export_schema_version"
        ],
        "task_export_record_type": resolved_snapshot[
            "upstream_export_record_type"
        ],
        "task_export_snapshot_id": resolved_snapshot["snapshot_id"],
        "task_export_content_sha256": resolved_snapshot[
            "upstream_export_content_sha256"
        ],
        "resolved_task_set_sha256": resolved_snapshot[
            "resolved_task_set_sha256"
        ],
        "task_source_sha256": source.get("task_source_sha256"),
        "site_url_map_sha256": resolved_snapshot["site_url_map_sha256"],
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
    if dict(audit) != expected:
        raise SchemaError(
            "resolved task-interface audit differs from exact task-row recomputation"
        )
    if resolved_snapshot.get("task_action_interface_audit_content_sha256") != sha256_json(
        audit
    ):
        raise SchemaError("resolved task-interface audit content hash differs")
    compile_report = resolved_snapshot.get(
        "page_state_evaluator_compile_report"
    )
    compile_report_sha256 = resolved_snapshot.get(
        "page_state_evaluator_compile_report_content_sha256"
    )
    if expected["handoff_eligible"]:
        if not isinstance(compile_report, Mapping):
            raise SchemaError(
                "resolved task snapshot lacks its page-state compile report"
            )
        validate_page_state_compile_authority(
            compile_report,
            task_rows=[dict(row) for row in rows],
        )
        if compile_report_sha256 != sha256_json(compile_report):
            raise SchemaError(
                "resolved page-state compile report content hash differs"
            )
    elif compile_report is not None or compile_report_sha256 is not None:
        raise SchemaError(
            "interface-incompatible resolved tasks cannot claim a page-state "
            "compile report"
        )
    return dict(audit)


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
