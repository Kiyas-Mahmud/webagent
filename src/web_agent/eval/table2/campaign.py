"""Campaign orchestration around a pluggable, frozen runtime runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import inspect
import json
import math
from pathlib import Path
import traceback
import hashlib
from time import perf_counter
from typing import Any

from .package_validator import (
    append_campaign_ledger_event,
    load_yaml,
    validate_campaign,
)
from .common import (
    SCHEMA_VERSION,
    SchemaError,
    Table2Error,
    as_mapping,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    strict_bool,
)
from .execution_guard import (
    InfrastructureInvalidError,
    verify_runner_before_execution,
)
from .outcome_semantics import (
    OPAQUE_VERIFIER_TERMINAL,
    normalize_runtime_terminal_reason,
)
from .schedule import (
    SYSTEM_IDS,
    discover_block_attempt_directories,
    resolve_block_attempts,
    validate_schedule,
)
from .sealed_verifier import (
    SealedVerifierSink,
    SealedVerifierWriter,
    assert_no_verifier_evidence,
)


def load_entrypoint(specification: str) -> Any:
    """Load ``module:attribute`` without imposing a runtime package import."""

    if ":" not in specification:
        raise ValueError("runner entrypoint must use module:attribute syntax")
    module_name, attribute_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)


def run_campaign(
    campaign_dir: str | Path,
    *,
    runner: Any,
    runner_entrypoint: str | None = None,
    maximum_blocks: int | None = None,
) -> dict[str, Any]:
    return CampaignRunner(
        campaign_dir=campaign_dir,
        runner=runner,
        runner_entrypoint=runner_entrypoint,
        maximum_blocks=maximum_blocks,
    ).run()


class CampaignRunner:
    """Public whole-block orchestrator for the frozen Table 2 campaign."""

    def __init__(
        self,
        *,
        campaign_dir: str | Path,
        runner: Any,
        runner_entrypoint: str | None = None,
        maximum_blocks: int | None = None,
    ) -> None:
        self.root = Path(campaign_dir).resolve()
        self.runner = runner
        self.runner_entrypoint = runner_entrypoint
        self.maximum_blocks = maximum_blocks
        if maximum_blocks is not None and maximum_blocks <= 0:
            raise ValueError("maximum_blocks must be positive")
        preflight = validate_campaign(
            self.root,
            require_complete=False,
            require_aggregates=False,
        )
        if not preflight.passed:
            raise Table2Error(
                "campaign preflight validation failed: " + "; ".join(preflight.errors)
            )
        self.manifest = read_json(self.root / "campaign_manifest.json")
        verify_runner_before_execution(
            self.root,
            runner=self.runner,
            runner_entrypoint=self.runner_entrypoint,
        )
        self.schedule = read_jsonl(self.root / "schedule" / "schedule.jsonl")
        validate_schedule(self.schedule)
        self.tasks = _load_frozen_tasks(self.root)
        self.protocol = load_yaml(self.root / "frozen" / "protocol.yaml")
        self.systems = {
            system_id: load_yaml(
                self.root / "frozen" / "systems" / f"{system_id.lower()}.yaml"
            )
            for system_id in SYSTEM_IDS
        }

    def run_block(self, block: str | Mapping[str, Any]) -> dict[str, Any]:
        """Run or resume one complete matched E0--E3 block."""

        verify_runner_before_execution(
            self.root,
            runner=self.runner,
            runner_entrypoint=self.runner_entrypoint,
        )

        if isinstance(block, str):
            matches = [row for row in self.schedule if str(row["block_id"]) == block]
            if len(matches) != 1:
                raise SchemaError(f"unknown or duplicate scheduled block ID: {block}")
            schedule_row = matches[0]
        else:
            supplied = dict(block)
            matches = [
                row
                for row in self.schedule
                if str(row["block_id"]) == str(supplied.get("block_id"))
            ]
            if len(matches) != 1:
                raise SchemaError("run_block received a row outside the frozen schedule")
            schedule_row = matches[0]
            for key, value in supplied.items():
                if key in schedule_row and schedule_row[key] != value:
                    raise SchemaError(
                        f"run_block cannot override frozen schedule field {key!r}"
                    )
        task_id = str(schedule_row["task_id"])
        if task_id not in self.tasks:
            raise SchemaError(f"scheduled task is absent from frozen manifest: {task_id}")
        return _run_block(
            self.root,
            schedule_row,
            task=self.tasks[task_id],
            protocol=self.protocol,
            systems=self.systems,
            runner=self.runner,
        )

    def run(self) -> dict[str, Any]:
        selected_rows = (
            self.schedule
            if self.maximum_blocks is None
            else self.schedule[: self.maximum_blocks]
        )
        resolutions = [self.run_block(row) for row in selected_rows]
        all_terminal = len(resolutions) == len(self.schedule) and all(
            row["status"] in {"INCLUDED", "EXCLUDED_INFRASTRUCTURE"}
            for row in resolutions
        )
        completion = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.manifest["campaign_id"],
            "status": "COMPLETE" if all_terminal else "INCOMPLETE",
            "scheduled_block_count": len(self.schedule),
            "processed_block_count": len(resolutions),
            "included_block_count": sum(row["status"] == "INCLUDED" for row in resolutions),
            "infrastructure_excluded_block_count": sum(
                row["status"] == "EXCLUDED_INFRASTRUCTURE" for row in resolutions
            ),
            "publication_status": (
                "PILOT_ONLY"
                if self.manifest.get("evidence_label") == "PILOT_ONLY"
                else "N/R"
            ),
        }
        atomic_write_json(self.root / "completion.json", completion)
        return completion


def _run_block(
    root: Path,
    schedule_row: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    protocol: Mapping[str, Any],
    systems: Mapping[str, Mapping[str, Any]],
    runner: Any,
) -> dict[str, Any]:
    base = _block_base(root, schedule_row)
    campaign_manifest = read_json(root / "campaign_manifest.json")
    campaign_id = str(campaign_manifest["campaign_id"])
    campaign_mode = str(campaign_manifest.get("campaign_mode", "evaluation"))
    base.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_attempts(base, int(schedule_row["max_block_attempts"]))
    resolution = resolve_block_attempts(schedule_row, existing)
    if resolution["status"] in {"INCLUDED", "EXCLUDED_INFRASTRUCTURE"}:
        atomic_write_json(base / "resolution.json", resolution)
        return resolution
    if resolution["status"] == "INTERRUPTED_UNAUTHORIZED":
        raise Table2Error(
            "partial block lacks typed preregistered infrastructure evidence; "
            "it cannot consume a rerun"
        )

    start = len(existing)
    maximum = int(schedule_row["max_block_attempts"])
    for attempt_id in range(start, maximum):
        rerun = base / f"rerun_{attempt_id}"
        if rerun.exists() and any(rerun.iterdir()):
            raise Table2Error(f"refusing to overwrite partial paired attempt: {rerun}")
        status_rows = {
            system_id: {
                "launched": False,
                "completed": False,
                "infrastructure_invalid": False,
                "episode_id": None,
                "error": None,
                "error_hash": None,
                "fatal_noninfrastructure_error": False,
                "infrastructure_reason": None,
                "infrastructure_evidence_sha256": None,
            }
            for system_id in SYSTEM_IDS
        }
        block_manifest = {
            "schema_version": SCHEMA_VERSION,
            "block_id": schedule_row["block_id"],
            "attempt_id": attempt_id,
            "systems": status_rows,
        }
        rerun.mkdir(parents=True, exist_ok=False)
        for system_id in SYSTEM_IDS:
            (rerun / system_id / "runtime").mkdir(parents=True)
            (rerun / system_id / "sealed").mkdir(parents=True)
        atomic_write_json(rerun / "block_manifest.json", block_manifest)

        block_invalid = False
        for system_id in schedule_row["execution_order"]:
            if block_invalid:
                break
            package = rerun / system_id
            runtime_dir = package / "runtime"
            episode_id = (
                f"{campaign_id}:{system_id}:{schedule_row['task_id']}:"
                f"repeat-{int(schedule_row['repeat_id'])}:"
                f"seed-{int(schedule_row['matched_model_seed'])}"
            )
            event_logs = _initialize_runtime_package(
                root,
                runtime_dir,
                schedule_row=schedule_row,
                attempt_id=attempt_id,
                system_id=system_id,
                episode_id=episode_id,
            )
            sink = SealedVerifierSink(
                root,
                block_id=str(schedule_row["block_id"]),
                attempt_id=attempt_id,
                system_id=system_id,
                episode_id=episode_id,
                matched_seed=int(schedule_row["matched_model_seed"]),
                task_id=str(schedule_row["task_id"]),
                repeat_id=int(schedule_row["repeat_id"]),
            )
            status_rows[system_id]["launched"] = True
            status_rows[system_id]["episode_id"] = episode_id
            append_campaign_ledger_event(
                root,
                ledger_type="access",
                event_type="episode_task_load",
                payload={
                    "block_id": str(schedule_row["block_id"]),
                    "task_id": str(schedule_row["task_id"]),
                    "task_partition": str(schedule_row["task_partition"]),
                    "system_id": system_id,
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "locked_test_content": (
                        read_json(root / "campaign_manifest.json").get("evidence_label")
                        != "PILOT_ONLY"
                    ),
                },
            )
            atomic_write_json(rerun / "block_manifest.json", block_manifest)
            physical_episode_started = perf_counter()
            try:
                result = _invoke_runner(
                    runner,
                    task=dict(task),
                    repeat_id=int(schedule_row["repeat_id"]),
                    model_seed=int(schedule_row["matched_model_seed"]),
                    system_id=system_id,
                    system_config=dict(systems[system_id]),
                    protocol=dict(protocol),
                    stage_seeds=dict(schedule_row["stage_seeds"]),
                    runtime_dir=runtime_dir,
                    event_logs=event_logs,
                    verifier_writer=SealedVerifierWriter(sink),
                    episode_id=episode_id,
                    block_id=str(schedule_row["block_id"]),
                    rerun_id=attempt_id,
                )
                _ensure_final_verifier_receipt(
                    runner,
                    result=result,
                    verifier_sink=sink,
                    event_logs=event_logs,
                    task=dict(task),
                    system_id=system_id,
                    episode_id=episode_id,
                    runtime_dir=runtime_dir,
                )
                summary = _normalize_episode_summary(
                    result,
                    episode_id=episode_id,
                    runtime_dir=runtime_dir,
                    system_id=system_id,
                    task_id=str(schedule_row["task_id"]),
                    repeat_id=int(schedule_row["repeat_id"]),
                    model_seed=int(schedule_row["matched_model_seed"]),
                    require_efficiency=(
                        read_json(root / "campaign_manifest.json").get("campaign_mode")
                        != "smoke"
                    ),
                )
                assert_no_verifier_evidence(summary, context=f"runner result {episode_id}")
            except Exception as exc:  # Preserve the complete failed attempt; never overwrite it.
                if not isinstance(exc, InfrastructureInvalidError):
                    status_rows[system_id].update(
                        {
                            "fatal_noninfrastructure_error": True,
                            "error": type(exc).__name__,
                            "error_hash": hashlib.sha256(
                                str(exc).encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                    atomic_write_json(rerun / "block_manifest.json", block_manifest)
                    exception_path = package / "sealed" / "runtime_exception.txt"
                    exception_path.write_text(
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                        encoding="utf-8",
                    )
                    exception_path.chmod(0o600)
                    raise
                infrastructure_evidence = exc.evidence_record(
                    campaign_mode=campaign_mode,
                    block_id=str(schedule_row["block_id"]),
                    attempt_id=attempt_id,
                    system_id=system_id,
                    episode_id=episode_id,
                )
                infrastructure_path = package / "sealed" / "infrastructure_invalid.json"
                atomic_write_json(infrastructure_path, infrastructure_evidence, mode=0o600)
                infrastructure_hash = sha256_file(infrastructure_path)
                summary = _infrastructure_partial_summary(
                    exc,
                    runtime_dir=runtime_dir,
                    episode_id=episode_id,
                    system_id=system_id,
                    task_id=str(schedule_row["task_id"]),
                    repeat_id=int(schedule_row["repeat_id"]),
                    model_seed=int(schedule_row["matched_model_seed"]),
                    physical_wall_clock_seconds=(
                        perf_counter() - physical_episode_started
                    ),
                    infrastructure_hash=infrastructure_hash,
                )
                exception_path = package / "sealed" / "runtime_exception.txt"
                exception_path.write_text(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    encoding="utf-8",
                )
                exception_path.chmod(0o600)
            atomic_write_json(runtime_dir / "episode_summary.json", summary)
            status_rows[system_id].update(
                {
                    "completed": bool(summary["completed"]),
                    "infrastructure_invalid": bool(summary["infrastructure_invalid"]),
                    "error": summary.get("runtime_error_type"),
                    "infrastructure_reason": summary.get("infrastructure_reason"),
                    "infrastructure_evidence_sha256": summary.get(
                        "infrastructure_evidence_sha256"
                    ),
                }
            )
            block_invalid = block_invalid or bool(summary["infrastructure_invalid"])
            _write_runtime_artifact_hashes(runtime_dir)
            atomic_write_json(rerun / "block_manifest.json", block_manifest)

        existing.append(block_manifest)
        resolution = resolve_block_attempts(schedule_row, existing)
        atomic_write_json(base / "resolution.json", resolution)
        if resolution["status"] == "INCLUDED":
            return resolution
    return resolution


def _invoke_runner(runner: Any, **context: Any) -> Any:
    target: Callable[..., Any]
    if hasattr(runner, "run") and callable(runner.run):
        target = runner.run
    elif callable(runner):
        target = runner
    else:
        raise TypeError("runner must be callable or expose .run()")
    signature = inspect.signature(target)
    if len(signature.parameters) == 1:
        sole = next(iter(signature.parameters.values()))
        if sole.name in {"request", "context"}:
            return target(dict(context))
    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = context if has_kwargs else {
        name: context[name] for name in signature.parameters if name in context
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and name not in kwargs
    ]
    if missing:
        raise TypeError(f"runner requires unsupported parameters: {missing}")
    return target(**kwargs)


def _ensure_final_verifier_receipt(
    runner: Any,
    *,
    result: Any,
    verifier_sink: SealedVerifierSink,
    event_logs: Any,
    **context: Any,
) -> None:
    """Finalize non-success terminal paths without exposing verifier truth.

    A benchmark/runtime wrapper may seal the final record during its last
    terminal check.  Otherwise it must expose ``finalize_episode``: that method
    owns the independent evaluator, writes through a write-only verifier
    capability, and returns only an OpaqueTerminalSignal. Campaign
    orchestration retains the readable sink and records that opaque receipt
    after all agent decisions/actions are finished.
    """

    if verifier_sink.has_episode_final:
        return
    finalize = getattr(runner, "finalize_episode", None)
    if not callable(finalize):
        raise SchemaError(
            "runner ended without sealed episode_final; provide a benchmark wrapper "
            "with finalize_episode(...)->OpaqueTerminalSignal for timeout/loop/ABORT paths"
        )
    available = {
        **context,
        "result": result,
        "episode_summary": result,
        "verifier_writer": SealedVerifierWriter(verifier_sink),
    }
    signature = inspect.signature(finalize)
    has_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = available if has_kwargs else {
        name: available[name] for name in signature.parameters if name in available
    }
    signal = finalize(**kwargs)
    if not verifier_sink.has_episode_final:
        raise SchemaError("runner finalize_episode did not write sealed episode_final")
    try:
        signal_values = as_mapping(signal)
    except TypeError as exc:
        raise SchemaError("runner finalize_episode did not return an opaque signal") from exc
    if set(signal_values).intersection({"task_success", "oracle_success", "oracle_progress"}):
        raise SchemaError("runner finalize_episode leaked verifier truth")
    if event_logs is None:
        raise SchemaError("final opaque verifier receipt requires runtime event logs")
    event_logs.append("terminal_signals", "episode_final_receipt", signal)


def _normalize_episode_summary(
    result: Any,
    *,
    episode_id: str,
    runtime_dir: Path,
    system_id: str,
    task_id: str,
    repeat_id: int,
    model_seed: int,
    require_efficiency: bool,
) -> dict[str, Any]:
    row = as_mapping(result)
    if "episode_id" in row and str(row["episode_id"]) != episode_id:
        raise SchemaError("runner returned an episode ID outside its frozen package")
    expected_identity = {
        "system_id": system_id,
        "task_id": task_id,
        "repeat_id": repeat_id,
        "model_seed": model_seed,
    }
    for key, expected in expected_identity.items():
        if key in row:
            actual = getattr(row[key], "value", row[key])
            if str(actual) != str(expected):
                raise SchemaError(f"runner returned a mismatched {key}")
    aliases = {
        "step_count": (
            "step_count",
            "executor_steps",
            "total_browser_actions",
            "browser_action_count",
        ),
        "executed_action_count": (
            "executed_action_count",
            "executed_actions",
            "executor_steps",
        ),
        "rejected_action_count": ("rejected_action_count", "rejected_actions"),
        "recovery_action_count": ("recovery_action_count", "recovery_actions"),
        "recovery_attempt_count": ("recovery_attempt_count", "recovery_attempts"),
        "model_call_count": ("model_call_count", "model_calls"),
        "task_wall_clock_seconds": (
            "task_wall_clock_seconds",
            "elapsed_seconds",
            "wall_time",
            "wall_clock_seconds",
        ),
        "input_token_count": (
            "input_token_count",
            "prompt_token_count",
            "input_tokens",
            "prompt_tokens",
        ),
        "output_token_count": (
            "output_token_count",
            "output_tokens",
            "completion_token_count",
            "completion_tokens",
        ),
        "model_parameter_count": ("model_parameter_count", "model_parameters"),
        "trainable_parameter_count": (
            "trainable_parameter_count",
            "trainable_parameters",
        ),
        "peak_gpu_memory_mb": ("peak_gpu_memory_mb", "peak_gpu_mb"),
        "peak_system_memory_mb": ("peak_system_memory_mb", "peak_ram_mb"),
        "memory_index_size": ("memory_index_size", "index_size"),
        "training_gpu_hours": ("training_gpu_hours",),
    }
    normalized = dict(row)
    for destination, sources in aliases.items():
        if destination not in normalized:
            normalized[destination] = next(
                (row[source] for source in sources if source in row),
                None
                if destination
                in {
                    "model_call_count",
                    "task_wall_clock_seconds",
                    "input_token_count",
                    "output_token_count",
                    "model_parameter_count",
                    "trainable_parameter_count",
                    "peak_gpu_memory_mb",
                    "peak_system_memory_mb",
                    "memory_index_size",
                    "training_gpu_hours",
                }
                else 0,
            )
    completed = _optional_strict_bool(row, "completed", default=True)
    infrastructure_invalid = _optional_strict_bool(
        row, "infrastructure_invalid", default=False
    )
    environment_failure = _optional_strict_bool(
        row, "environment_failure", default=False
    )
    reset_already_success = _optional_strict_bool(
        row, "reset_already_success", default=False
    )
    valid_for_primary = _optional_strict_bool(
        row, "valid_for_primary", default=True
    )
    if (
        infrastructure_invalid
        or environment_failure
        or reset_already_success
        or not valid_for_primary
    ):
        raise SchemaError(
            "runner summary booleans cannot authorize a whole-block rerun; "
            "the benchmark adapter must raise InfrastructureInvalidError with "
            "a preregistered reason and adapter evidence"
        )
    raw_runtime_terminal_reason = row.get(
        "runtime_terminal_reason",
        row.get("terminal_reason"),
    )
    if raw_runtime_terminal_reason is None:
        if require_efficiency:
            raise SchemaError(
                "evaluation runtime summary lacks runtime_terminal_reason"
            )
        # Legacy engineering-smoke runners terminate through their synthetic
        # sealed finalizer but predate the explicit runtime/evaluator split.
        # This compatibility default is forbidden for evaluation campaigns.
        runtime_terminal_reason = OPAQUE_VERIFIER_TERMINAL
    else:
        runtime_terminal_reason = normalize_runtime_terminal_reason(
            raw_runtime_terminal_reason,
            context=f"runtime summary {episode_id}",
        )
    normalized.update(
        {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            **expected_identity,
            "completed": completed,
            "infrastructure_invalid": False,
            "runtime_terminal_reason": runtime_terminal_reason,
        }
    )
    for key in (
        "step_count",
        "executed_action_count",
        "rejected_action_count",
        "recovery_action_count",
        "recovery_attempt_count",
    ):
        if type(normalized[key]) is not int:
            raise SchemaError(f"runtime summary {key} must be an exact integer")
        if normalized[key] < 0:
            raise SchemaError(f"runtime summary {key} cannot be negative")
    if normalized["model_call_count"] is not None:
        if type(normalized["model_call_count"]) is not int:
            raise SchemaError("runtime summary model_call_count must be an exact integer")
        if normalized["model_call_count"] < 0:
            raise SchemaError("runtime summary model_call_count cannot be negative")
    elif require_efficiency:
        raise SchemaError("evaluation runtime summary lacks measured model_call_count")
    if normalized["task_wall_clock_seconds"] is not None:
        if isinstance(normalized["task_wall_clock_seconds"], bool):
            raise SchemaError("runtime task wall-clock cannot be boolean")
        normalized["task_wall_clock_seconds"] = float(
            normalized["task_wall_clock_seconds"]
        )
        if (
            not math.isfinite(normalized["task_wall_clock_seconds"])
            or normalized["task_wall_clock_seconds"] < 0
        ):
            raise SchemaError("runtime task wall-clock must be finite and nonnegative")
    elif require_efficiency:
        raise SchemaError("evaluation runtime summary lacks measured task wall-clock")
    for key in (
        "input_token_count",
        "output_token_count",
        "model_parameter_count",
        "trainable_parameter_count",
        "memory_index_size",
    ):
        value = normalized[key]
        if value is not None and (type(value) is not int or value < 0):
            raise SchemaError(f"runtime summary {key} must be an exact nonnegative integer")
    for key in (
        "peak_gpu_memory_mb",
        "peak_system_memory_mb",
        "training_gpu_hours",
    ):
        value = normalized[key]
        if value is None:
            continue
        if isinstance(value, bool):
            raise SchemaError(f"runtime summary {key} cannot be boolean")
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"runtime summary {key} must be numeric") from exc
        if not math.isfinite(normalized[key]) or normalized[key] < 0:
            raise SchemaError(f"runtime summary {key} must be finite and nonnegative")
    action_records = read_jsonl(runtime_dir / "actions.jsonl")
    if action_records:
        statuses: list[str] = []
        for event in action_records:
            payload = _runtime_event_payload(event)
            execution = payload.get("execution", {}) if isinstance(payload, Mapping) else {}
            status = execution.get("status", "") if isinstance(execution, Mapping) else ""
            normalized_status = str(getattr(status, "value", status)).upper()
            if normalized_status not in {"EXECUTED", "REJECTED", "ERROR"}:
                raise SchemaError("runtime action log has an unknown execution status")
            statuses.append(normalized_status)
        normalized["step_count"] = len(action_records)
        normalized["executed_action_count"] = sum(status == "EXECUTED" for status in statuses)
        normalized["rejected_action_count"] = len(statuses) - normalized["executed_action_count"]
        normalized["recovery_action_count"] = sum(
            str(event.get("event_type", "")) == "recovery_action"
            for event in action_records
        )
    elif normalized["step_count"]:
        raise SchemaError("runtime summary reports steps without hashed action events")

    latency_sources = _runtime_latency_totals(runtime_dir)
    for key, value in latency_sources.items():
        if key in normalized:
            supplied = normalized[key]
            if (
                isinstance(supplied, bool)
                or not isinstance(supplied, (int, float))
                or not math.isfinite(float(supplied))
                or not math.isclose(
                    float(supplied), value, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                raise SchemaError(
                    f"runtime summary {key} differs from append-only latency evidence"
                )
        normalized[key] = value
    if normalized["step_count"] != (
        normalized["executed_action_count"] + normalized["rejected_action_count"]
    ):
        raise SchemaError(
            "step_count must equal executed plus rejected budget-consuming requests"
        )
    if normalized["recovery_action_count"] > normalized["step_count"]:
        raise SchemaError("recovery actions are a subset of steps, not an extra total")
    return normalized


def _infrastructure_partial_summary(
    error: InfrastructureInvalidError,
    *,
    runtime_dir: Path,
    episode_id: str,
    system_id: str,
    task_id: str,
    repeat_id: int,
    model_seed: int,
    physical_wall_clock_seconds: float,
    infrastructure_hash: str,
) -> dict[str, Any]:
    """Preserve exact work consumed before a typed infrastructure fault.

    A late controller disconnect must not turn a partially executed episode
    into an all-zero run.  Canonical append-only logs are the accounting
    authority; an EpisodeRunner partial summary, when present, is an
    additional reconciliation receipt rather than a trusted replacement.
    """

    action_records = read_jsonl(runtime_dir / "actions.jsonl")
    statuses: list[str] = []
    for record in action_records:
        payload = _runtime_event_payload(record)
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise SchemaError(
                "infrastructure-interrupted action lacks execution evidence"
            )
        status = str(getattr(execution.get("status"), "value", execution.get("status")))
        status = status.upper()
        if status not in {"EXECUTED", "REJECTED", "ERROR"}:
            raise SchemaError(
                "infrastructure-interrupted action has unknown execution status"
            )
        statuses.append(status)
    recovery_action_count = sum(
        str(record.get("event_type", "")) == "recovery_action"
        for record in action_records
    )
    recovery_attempt_count = sum(
        str(record.get("event_type", "")) == "recovery_attempt"
        for record in read_jsonl(runtime_dir / "recoveries.jsonl")
    )
    memory_path = runtime_dir / "memory_queries.jsonl"
    memory_query_count = (
        sum(
            str(record.get("event_type", "")) == "post_failure_query"
            for record in read_jsonl(memory_path)
        )
        if memory_path.exists()
        else 0
    )
    partial = getattr(error, "partial_episode_summary", None)
    if partial is not None:
        if not isinstance(partial, Mapping):
            raise SchemaError("infrastructure partial summary is not an object")
        assert_no_verifier_evidence(
            partial,
            context=f"infrastructure partial summary {episode_id}",
        )
        expected = {
            "episode_id": episode_id,
            "system_id": system_id,
            "task_id": task_id,
            "repeat_id": repeat_id,
            "model_seed": model_seed,
        }
        for field, value in expected.items():
            actual = getattr(partial.get(field), "value", partial.get(field))
            if str(actual) != str(value):
                raise SchemaError(
                    f"infrastructure partial summary changed {field}"
                )
        reconciliations = {
            "executor_steps": len(action_records),
            "recovery_actions": recovery_action_count,
            "recovery_attempts": recovery_attempt_count,
            "memory_queries": memory_query_count,
        }
        for field, expected_value in reconciliations.items():
            if type(partial.get(field)) is not int or partial[field] != expected_value:
                raise SchemaError(
                    f"infrastructure partial summary {field} differs from logs"
                )
        partial_sha256: str | None = sha256_json(dict(partial))
        failure_incidents = partial.get("failure_incidents")
        memory_interventions = partial.get("memory_interventions")
    else:
        partial_sha256 = None
        failure_incidents = None
        memory_interventions = None

    if (
        isinstance(physical_wall_clock_seconds, bool)
        or not math.isfinite(physical_wall_clock_seconds)
        or physical_wall_clock_seconds < 0
    ):
        raise SchemaError("infrastructure partial wall clock is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "system_id": system_id,
        "task_id": task_id,
        "repeat_id": repeat_id,
        "model_seed": model_seed,
        "completed": False,
        "infrastructure_invalid": True,
        "valid_for_primary": False,
        "step_count": len(action_records),
        "executed_action_count": sum(status == "EXECUTED" for status in statuses),
        "rejected_action_count": sum(status != "EXECUTED" for status in statuses),
        "recovery_action_count": recovery_action_count,
        "recovery_attempt_count": recovery_attempt_count,
        "failure_incident_count": failure_incidents,
        "memory_query_count": memory_query_count,
        "memory_intervention_count": memory_interventions,
        "model_call_count": None,
        "task_wall_clock_seconds": float(physical_wall_clock_seconds),
        "partial_episode_summary_sha256": partial_sha256,
        "runtime_error_type": type(error).__name__,
        "runtime_error_hash": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "infrastructure_reason": error.reason_code,
        "infrastructure_evidence_sha256": infrastructure_hash,
    }


def _initialize_runtime_package(
    campaign_root: Path,
    runtime_dir: Path,
    *,
    schedule_row: Mapping[str, Any],
    attempt_id: int,
    system_id: str,
    episode_id: str,
) -> Any:
    campaign_manifest = read_json(campaign_root / "campaign_manifest.json")
    system_config = load_yaml(
        campaign_root / "frozen" / "systems" / f"{system_id.lower()}.yaml"
    )
    model_manifest = campaign_root / "frozen" / "models" / f"seed_{int(schedule_row['matched_model_seed'])}.json"
    memory_manifest = (
        campaign_root
        / "memory"
        / f"seed_{int(schedule_row['matched_model_seed'])}"
        / "manifest.json"
    )
    duplicate_audit_manifest = (
        campaign_root / "frozen" / "benchmark" / "duplicate_audit_manifest.json"
    )
    rng_provenance = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256_stage_keyed_v1",
        "protocol_id": read_json(campaign_root / "campaign_manifest.json")["protocol_id"],
        "campaign_id": read_json(campaign_root / "campaign_manifest.json")["campaign_id"],
        "campaign_seed": read_json(campaign_root / "campaign_manifest.json")["campaign_seed"],
        "task_id": schedule_row["task_id"],
        "repeat_id": schedule_row["repeat_id"],
        "matched_seed": schedule_row["matched_model_seed"],
        "anchor_decision_index": schedule_row["stage_seed_decision_index"],
        "stage_seed_anchors": schedule_row["stage_seeds"],
        "dynamic_key_fields": [
            "campaign_id",
            "task_id",
            "repeat_id",
            "matched_seed",
            "stage",
            "decision_index",
            "incident_index",
            "attempt_index",
            "stream",
        ],
        "namespace_fields": ["protocol_id", "campaign_seed"],
        "system_id_in_seed_key": False,
        "rerun_id_in_seed_key": False,
    }
    atomic_write_json(runtime_dir / "rng_provenance.json", rng_provenance)
    atomic_write_json(
        runtime_dir / "episode_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "block_id": schedule_row["block_id"],
            "attempt_id": attempt_id,
            "system_id": system_id,
            "task_id": schedule_row["task_id"],
            "matched_model_seed": schedule_row["matched_model_seed"],
            "repeat_id": schedule_row["repeat_id"],
            "stage_seeds": schedule_row["stage_seeds"],
            "rng_provenance_sha256": sha256_file(runtime_dir / "rng_provenance.json"),
            "campaign_manifest_sha256": sha256_file(campaign_root / "campaign_manifest.json"),
            "runner_identity_scope": campaign_manifest["runner_identity_scope"],
            "runner_entrypoint": campaign_manifest.get("runner_entrypoint"),
            "runner_attestation_sha256": campaign_manifest.get(
                "runner_attestation_sha256"
            ),
            "frozen_artifact_hashes_sha256": sha256_file(campaign_root / "artifact_hashes.json"),
            "protocol_sha256": sha256_file(campaign_root / "frozen" / "protocol.yaml"),
            "environment_manifest_relative_path": "frozen/environment.json",
            "environment_manifest_sha256": sha256_file(
                campaign_root / "frozen" / "environment.json"
            ),
            "system_overlay_sha256": sha256_file(
                campaign_root / "frozen" / "systems" / f"{system_id.lower()}.yaml"
            ),
            "parameter_provider_prompt_sha256": sha256_file(
                campaign_root / "frozen" / "prompts" / "parameter_provider_v1.txt"
            ),
            "resolved_features": system_config["features"],
            "policy_source": system_config["policy_source"],
            "model_manifest_relative_path": (
                str(model_manifest.relative_to(campaign_root))
                if model_manifest.exists()
                else None
            ),
            "model_manifest_sha256": sha256_file(model_manifest) if model_manifest.exists() else None,
            "memory_store_relative_path": (
                str(memory_manifest.parent.relative_to(campaign_root))
                if system_id == "E3" and memory_manifest.exists()
                else None
            ),
            "memory_manifest_sha256": (
                sha256_file(memory_manifest) if system_id == "E3" and memory_manifest.exists() else None
            ),
            "duplicate_audit_relative_path": (
                str(duplicate_audit_manifest.relative_to(campaign_root))
                if system_id == "E3"
                else None
            ),
            "duplicate_audit_manifest_sha256": (
                sha256_file(duplicate_audit_manifest) if system_id == "E3" else None
            ),
        },
    )
    try:
        from web_agent.runtime.event_log import EpisodeEventLogs

        return EpisodeEventLogs(
            runtime_dir,
            episode_id=episode_id,
            include_memory=system_id == "E3",
            system_id=system_id,
            task_id=str(schedule_row["task_id"]),
            repeat_id=int(schedule_row["repeat_id"]),
            matched_seed=int(schedule_row["matched_model_seed"]),
        )
    except ImportError:  # Detached artifact tools can still initialize placeholders.
        for filename in (
            "actions.jsonl",
            "transitions.jsonl",
            "recoveries.jsonl",
            "environment_events.jsonl",
            "terminal_signals.jsonl",
        ):
            (runtime_dir / filename).touch(exist_ok=False)
        if system_id == "E3":
            (runtime_dir / "memory_queries.jsonl").touch(exist_ok=False)
        return None


def _write_runtime_artifact_hashes(runtime_dir: Path) -> None:
    entries = list(runtime_dir.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise SchemaError(
            f"runtime artifact package contains a forbidden symlink: {runtime_dir}"
        )
    files = {
        str(path.relative_to(runtime_dir)): sha256_file(path)
        for path in sorted(entries)
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    atomic_write_json(
        runtime_dir / "artifact_hashes.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "files": files,
        },
    )


def _runtime_event_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload", record)
    if not isinstance(payload, Mapping):
        raise SchemaError("runtime event payload must be an object")
    return dict(payload)


def _latency(value: Any) -> float:
    if not isinstance(value, Mapping):
        return 0.0
    try:
        raw = value.get("latency_ms", 0.0)
        if isinstance(raw, bool):
            raise SchemaError("runtime latency cannot be boolean")
        latency = float(raw)
    except (TypeError, ValueError) as exc:
        raise SchemaError("runtime latency must be numeric") from exc
    if not math.isfinite(latency) or latency < 0:
        raise SchemaError("runtime latency must be finite and nonnegative")
    return latency


def _runtime_latency_totals(runtime_dir: Path) -> dict[str, float]:
    decision = 0.0
    provider = 0.0
    recovery = 0.0
    retrieval = 0.0
    for event in read_jsonl(runtime_dir / "actions.jsonl"):
        payload = _runtime_event_payload(event)
        decision += _latency(payload.get("decision"))
        # The typed trace covers deterministic extraction, the optional frozen
        # fallback, and wrapper overhead.  It is present for rejected requests
        # too, so provider cost is not biased downward by counting successes
        # only.  Legacy/smoke records retain the ActionParameters fallback.
        provider += _latency(
            payload.get("parameter_resolution", payload.get("parameters"))
        )
    for event in read_jsonl(runtime_dir / "transitions.jsonl"):
        payload = _runtime_event_payload(event)
        decision += _latency(payload.get("assessment"))
    for event in read_jsonl(runtime_dir / "recoveries.jsonl"):
        payload = _runtime_event_payload(event)
        recovery += _latency(payload.get("predicted_assessment"))
    memory_path = runtime_dir / "memory_queries.jsonl"
    if memory_path.exists():
        for event in read_jsonl(memory_path):
            if str(event.get("event_type", "")) != "post_failure_query":
                continue
            payload = _runtime_event_payload(event)
            retrieval += _latency(payload.get("query_result", payload))
    return {
        "decision_latency_ms": decision,
        "provider_latency_ms": provider,
        "recovery_latency_ms": recovery,
        "retrieval_latency_ms": retrieval,
    }


def _optional_strict_bool(
    row: Mapping[str, Any], key: str, *, default: bool
) -> bool:
    if key not in row:
        return default
    return strict_bool(row[key], context=f"episode_summary.{key}")


def _is_infrastructure_exception(exc: Exception) -> bool:
    """Only the typed, evidence-bearing adapter contract may consume a rerun."""

    return isinstance(exc, InfrastructureInvalidError)


def _read_existing_attempts(base: Path, maximum: int) -> list[dict[str, Any]]:
    return [
        read_json(directory / "block_manifest.json")
        for _, directory in discover_block_attempt_directories(
            base,
            maximum=maximum,
        )
    ]


def _load_frozen_tasks(root: Path) -> dict[str, dict[str, Any]]:
    candidates = list((root / "frozen").glob("task_manifest.*"))
    if len(candidates) != 1:
        raise SchemaError("frozen campaign must contain exactly one task_manifest.*")
    suffix = candidates[0].suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(candidates[0])
    elif suffix == ".csv":
        import csv

        with candidates[0].open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        value = json.loads(candidates[0].read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("tasks", value.get("records", []))
    if not isinstance(rows, list):
        raise SchemaError("frozen task manifest has no task records")
    tasks = {
        str(row["task_id"]): {**dict(row), "task_partition": "normal"}
        for row in rows
    }
    recovery_path = root / "frozen" / "benchmark" / "recovery_scenarios.json"
    if recovery_path.exists():
        payload = read_json(recovery_path)
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list) or not all(
            isinstance(row, Mapping) for row in scenarios
        ):
            raise SchemaError("frozen recovery scenario manifest is malformed")
        for row in scenarios:
            scenario_id = str(row.get("scenario_id", ""))
            if not scenario_id:
                raise SchemaError("recovery scenario lacks scenario_id")
            if scenario_id in tasks:
                raise SchemaError(f"normal/recovery task ID collision: {scenario_id}")
            tasks[scenario_id] = {
                **dict(row),
                "task_id": scenario_id,
                "task_partition": "recovery_diagnostic",
            }
    return tasks


def _block_base(root: Path, row: Mapping[str, Any]) -> Path:
    return (
        root
        / "paired_blocks"
        / f"seed_{int(row['matched_model_seed'])}"
        / _path_id(str(row["task_id"]))
        / f"repeat_{int(row['repeat_id'])}"
    )


def _path_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise SchemaError(f"unsafe empty artifact path ID: {value!r}")
    return cleaned[:160]
