"""Cross-system causal-trace and sealed-receipt validation.

These checks operate only on frozen artifacts after execution.  They never
return sealed evidence to the runtime and never infer missing events.  The
module is intentionally isolated so campaign/package validation can invoke it
without coupling policy code to the evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re
import stat
from typing import Any

from .common import SchemaError, read_json, read_jsonl, sha256_json
from .sealed_verifier import verify_sealed_stream


_OBSERVATION_EVENT_TYPES = {
    "reset",
    "post_action_observation",
    "post_recovery_observation",
}
_ACTION_EVENT_TYPES = {"normal_action", "recovery_action"}
_RUNTIME_LINK_KEYS = {
    "event_id",
    "episode_id",
    "system_id",
    "observation_id",
    "prior_action_id",
    "decision_id",
    "source_decision_id",
    "input_observation_ids",
    "action_id",
    "action_ids",
    "executed_action_id",
    "pre_observation_id",
    "post_observation_id",
    "assessment_id",
    "incident_id",
    "attempt_id",
    "recovery_attempt_id",
    "predicted_assessment_id",
    "failure_incident_id",
    "query_id",
    "post_failure_observation_id",
    "failed_action_id",
}
_VOLATILE_KEYS = {
    "schema_version",
    "record_type",
    "record_hash",
    "previous_record_hash",
    "timestamp_utc",
    "started_at_utc",
    "ended_at_utc",
    "sequence",
    "stream",
    "latency_ms",
    "elapsed_seconds",
    "wall_clock_seconds",
    "screenshot_path",
    # These hashes commit records containing normalized-away runtime IDs.  The
    # underlying semantic values remain in the compared payloads.
    "action_sha256",
    "observation_record_sha256",
    "post_observation_sha256",
    "logged_observation_sha256",
    "shadow_decision_sha256",
    "decision_sha256",
    # This hash binds a typed safety record containing normalized-away action
    # IDs; the receipt's semantic policy/version/result fields remain.
    "source_record_sha256",
}


def validate_runtime_screenshot_artifacts(
    runtime_dir: str | Path,
) -> dict[str, Any]:
    """Verify every logged WebArena screenshot against its owned byte artifact."""

    runtime = Path(runtime_dir).resolve()
    screenshot_root = runtime / "screenshots"
    if screenshot_root.is_symlink() or not screenshot_root.is_dir():
        raise SchemaError(
            "production WebArena runtime lacks a non-symlink screenshots directory"
        )
    records = [
        row
        for row in read_jsonl(runtime / "environment_events.jsonl")
        if str(row.get("event_type", "")) in _OBSERVATION_EVENT_TYPES
    ]
    if not records:
        raise SchemaError("production WebArena runtime contains no observations")
    artifact_manifest = read_json(runtime / "artifact_hashes.json")
    artifact_files = artifact_manifest.get("files")
    if not isinstance(artifact_files, Mapping):
        raise SchemaError("runtime artifact hash manifest has no file mapping")

    referenced: set[Path] = set()
    observation_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        observation = _event_payload(record)
        observation_id = _required_text(
            observation.get("observation_id"),
            context=f"screenshot observation[{index}] ID",
        )
        if observation_id in observation_ids:
            raise SchemaError("runtime screenshot observations reuse an observation ID")
        observation_ids.add(observation_id)
        digest = _required_sha256(
            observation.get("screenshot_sha256"),
            context=f"screenshot observation[{index}]",
        )
        raw_path = _required_text(
            observation.get("screenshot_path"),
            context=f"screenshot observation[{index}] path",
        )
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise SchemaError("production screenshot path must be adapter-owned absolute path")
        if candidate.is_symlink():
            raise SchemaError("production screenshot artifact must not be a symlink")
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise SchemaError("production screenshot artifact is absent") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise SchemaError("production screenshot artifact is not a regular file")
        if metadata.st_nlink != 1:
            raise SchemaError("production screenshot artifact must not be hard-linked")
        resolved = candidate.resolve()
        if resolved.parent != screenshot_root:
            raise SchemaError("production screenshot path escapes its runtime directory")
        if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise SchemaError("production screenshot artifact must be read-only")
        if resolved in referenced:
            raise SchemaError("production observations share one mutable screenshot path")
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != digest:
            raise SchemaError("production screenshot byte SHA-256 mismatch")
        expected_name = rf"{index:06d}-{re.escape(digest)}\.png"
        if re.fullmatch(expected_name, resolved.name) is None:
            raise SchemaError(
                "production screenshot filename is not per-observation content-addressed"
            )
        relative = str(resolved.relative_to(runtime))
        if artifact_files.get(relative) != digest:
            raise SchemaError(
                "production screenshot is absent from the runtime artifact hash closure"
            )
        referenced.add(resolved)

    entries = list(screenshot_root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise SchemaError("production screenshot directory contains a non-regular entry")
    if set(entries) != referenced:
        raise SchemaError(
            "production screenshot directory differs from logged observation closure"
        )
    return {
        "status": "PASS",
        "observation_count": len(records),
        "screenshot_count": len(referenced),
    }


def validate_included_block_causal_trace(
    rerun_dir: str | Path,
    *,
    state_fingerprint_fields: Sequence[str],
    require_trained_first_pre_action: bool = True,
) -> dict[str, Any]:
    """Validate paired reset state and the E2/E3 causal trace."""

    root = Path(rerun_dir)
    reset_result = validate_paired_reset_fingerprints(
        root,
        state_fingerprint_fields=state_fingerprint_fields,
    )
    first_decision_result = (
        validate_e1_e2_e3_first_pre_action_equivalence(root)
        if require_trained_first_pre_action
        else None
    )
    causal_result = validate_e2_e3_causal_trace(
        root / "E2" / "runtime",
        root / "E3" / "runtime",
    )
    return {
        **causal_result,
        "paired_reset_fingerprint": reset_result["state_fingerprint"],
        "paired_reset_system_count": reset_result["system_count"],
        "trained_first_pre_action_sha256": (
            first_decision_result["normalized_output_sha256"]
            if first_decision_result is not None
            else None
        ),
        "trained_first_pre_action_system_count": (
            first_decision_result["system_count"]
            if first_decision_result is not None
            else 0
        ),
    }


def validate_e1_e2_e3_first_pre_action_equivalence(
    rerun_dir: str | Path,
) -> dict[str, Any]:
    """Prove the first trained pre-action output is identical in E1--E3.

    Included blocks already prove an identical normalized reset observation and
    validate every stage-keyed RNG use.  This check closes the remaining
    artifact-level link by comparing the first trained-policy result after
    removing only runtime-generated IDs, paths, timestamps, and latency.  It
    accepts either a normal decision or the registered charged parser-rejection
    outcome, but all three trained systems must produce the same kind and
    semantic output.
    """

    root = Path(rerun_dir)
    normalized: dict[str, dict[str, Any]] = {}
    for system_id in ("E1", "E2", "E3"):
        records = read_jsonl(root / system_id / "runtime" / "actions.jsonl")
        first = next(
            (
                row
                for row in records
                if str(row.get("event_type", ""))
                in {"normal_action", "pre_action_parse_rejection"}
            ),
            None,
        )
        if first is None:
            raise SchemaError(
                f"{system_id} included episode lacks a first pre-action outcome"
            )
        event_type = str(first.get("event_type", ""))
        payload = _event_payload(first)
        if event_type == "normal_action":
            decision = payload.get("decision")
            parameters = payload.get("parameters")
            parameter_resolution = payload.get("parameter_resolution")
            attempts = payload.get("resolution_attempts")
            if not isinstance(decision, Mapping):
                raise SchemaError(f"{system_id} first normal action lacks a decision")
            if parameters is not None and not isinstance(parameters, Mapping):
                raise SchemaError(
                    f"{system_id} first normal action has malformed parameters"
                )
            if parameter_resolution is not None and not isinstance(
                parameter_resolution, Mapping
            ):
                raise SchemaError(
                    f"{system_id} first normal action has malformed provider trace"
                )
            if not isinstance(attempts, list):
                raise SchemaError(
                    f"{system_id} first normal action lacks provider attempts"
                )
            output = {
                "decision": decision,
                "parameters": parameters,
                "parameter_error": payload.get("parameter_error"),
                "parameter_resolution": parameter_resolution,
                "resolution_attempts": attempts,
            }
        else:
            output = payload.get("rejection")
            if not isinstance(output, Mapping):
                raise SchemaError(
                    f"{system_id} first parser rejection lacks typed evidence"
                )
        normalized[system_id] = {
            "event_type": event_type,
            "output": _normalize_semantics(output),
        }

    reference = normalized["E1"]
    if any(normalized[system_id] != reference for system_id in ("E2", "E3")):
        raise SchemaError(
            "included block E1/E2/E3 first trained pre-action outputs differ"
        )
    return {
        "status": "PASS",
        "normalized_output_sha256": sha256_json(reference),
        "system_count": 3,
        "event_type": reference["event_type"],
    }


def validate_paired_reset_fingerprints(
    rerun_dir: str | Path,
    *,
    state_fingerprint_fields: Sequence[str],
) -> dict[str, Any]:
    """Require one identical normalized first reset state across E0--E3.

    The fingerprint uses only the fields frozen in
    ``loop_rule.state_fingerprint_fields`` and the same canonical-SHA256
    construction as the runtime loop guard.  Runtime IDs, timestamps, paths,
    and system labels therefore cannot make otherwise matched resets differ or
    conceal a state difference.
    """

    fields = tuple(state_fingerprint_fields)
    if (
        not fields
        or len(set(fields)) != len(fields)
        or any(type(field) is not str or not field for field in fields)
    ):
        raise SchemaError(
            "paired reset validation requires unique nonempty registered "
            "state_fingerprint_fields"
        )

    root = Path(rerun_dir)
    fingerprints: dict[str, str] = {}
    for system_id in ("E0", "E1", "E2", "E3"):
        path = root / system_id / "runtime" / "environment_events.jsonl"
        observations = [
            row
            for row in read_jsonl(path)
            if str(row.get("event_type", "")) in _OBSERVATION_EVENT_TYPES
        ]
        if not observations or str(observations[0].get("event_type", "")) != "reset":
            raise SchemaError(
                f"{system_id} first observation evidence is not the reset observation"
            )
        resets = [
            row
            for row in observations
            if str(row.get("event_type", "")) == "reset"
        ]
        if len(resets) != 1:
            raise SchemaError(
                f"{system_id} must contain exactly one reset observation, "
                f"found {len(resets)}"
            )
        reset = _event_payload(resets[0])
        # Recheck the committed first-reset record before trusting its fields;
        # individual package validation also binds this digest to the sealed
        # after-reset receipt.
        _observation_record_sha256(
            reset,
            context=f"{system_id} reset observation",
        )
        missing = [field for field in fields if field not in reset]
        if missing:
            raise SchemaError(
                f"{system_id} reset observation lacks registered fingerprint "
                f"fields {missing}"
            )
        fingerprints[system_id] = sha256_json(
            {field: reset[field] for field in fields}
        )

    if len(set(fingerprints.values())) != 1:
        raise SchemaError(
            "included block E0--E3 normalized reset state fingerprints differ: "
            + ", ".join(
                f"{system_id}={fingerprints[system_id]}"
                for system_id in ("E0", "E1", "E2", "E3")
            )
        )
    return {
        "status": "PASS",
        "state_fingerprint": fingerprints["E0"],
        "state_fingerprint_fields": list(fields),
        "system_count": 4,
    }


def validate_e2_e3_causal_trace(
    e2_runtime_dir: str | Path,
    e3_runtime_dir: str | Path,
) -> dict[str, Any]:
    """Prove E2/E3 equality until the first admitted E3 intervention.

    Runtime IDs, timestamps, hash-chain fields, and measured latency are
    normalized away.  Observable state, model probabilities/confidences,
    selected actions and parameters, executor outcomes, transition diagnoses,
    no-memory shadows, and recovery plans/assessments remain semantic and are
    compared exactly.  When every E3 query abstains, the complete causal trace
    must be equal.
    """

    e2_root = Path(e2_runtime_dir)
    e3_root = Path(e3_runtime_dir)
    e2 = _load_causal_streams(e2_root, require_memory=False)
    e3 = _load_causal_streams(e3_root, require_memory=True)

    shadows, queries = _parse_e3_memory_events(e3["memory"])
    e2_plans = _recovery_plan_records(e2["recoveries"])
    e3_plans = _recovery_plan_records(e3["recoveries"])
    if len(shadows) != len(queries):
        raise SchemaError("E3 memory stream lacks one shadow per query")
    if len(queries) != len(e3_plans):
        raise SchemaError("E3 memory queries do not cover recovery plans one-to-one")

    first_admitted: int | None = None
    for index, (shadow_record, query_record) in enumerate(zip(shadows, queries)):
        admitted = _exact_bool(
            _query_result(query_record).get("admitted"),
            context=f"E3 query[{index}].admitted",
        )
        if first_admitted is None and admitted:
            first_admitted = index
        if first_admitted is None or index <= first_admitted:
            if index >= len(e2_plans):
                raise SchemaError("E2 lacks the recovery shadow paired to E3")
            _validate_paired_shadow(
                index,
                shadow_record=shadow_record,
                query_record=query_record,
                e2_plan=e2_plans[index][1],
                e3_plan=e3_plans[index][1],
                admitted=admitted,
            )
        if first_admitted == index:
            break

    if first_admitted is None:
        if len(e2_plans) != len(e3_plans):
            raise SchemaError("E2/E3 recovery-plan counts differ without intervention")
        for name in ("actions", "observations", "transitions", "recoveries"):
            _assert_semantic_equal(
                name,
                e2[name],
                e3[name],
            )
        return {
            "status": "PASS",
            "comparison_scope": "FULL_TRACE_E3_ABSTAINED",
            "first_admitted_query_index": None,
            "e3_query_count": len(queries),
        }

    query_payload = _event_payload(queries[first_admitted])
    query = query_payload.get("query")
    if not isinstance(query, Mapping):
        raise SchemaError(
            "admitted E3 query lacks causal failed-action/post-observation binding"
        )
    failed_action_id = _required_text(
        query.get("failed_action_id"),
        context="admitted E3 query failed_action_id",
    )
    post_observation_id = _required_text(
        query.get("post_failure_observation_id"),
        context="admitted E3 query post_failure_observation_id",
    )
    action_stop = _find_action_index(e3["actions"], failed_action_id)
    observation_stop = _find_observation_index(
        e3["observations"], post_observation_id
    )
    transition_stop = _find_transition_index(
        e3["transitions"], failed_action_id
    )
    e2_plan_stop = e2_plans[first_admitted][0]
    e3_plan_stop = e3_plans[first_admitted][0]
    _assert_semantic_equal(
        "actions before first admitted intervention",
        e2["actions"][: action_stop + 1],
        e3["actions"][: action_stop + 1],
    )
    _assert_semantic_equal(
        "observations before first admitted intervention",
        e2["observations"][: observation_stop + 1],
        e3["observations"][: observation_stop + 1],
    )
    _assert_semantic_equal(
        "transitions before first admitted intervention",
        e2["transitions"][: transition_stop + 1],
        e3["transitions"][: transition_stop + 1],
    )
    _assert_semantic_equal(
        "recoveries before first admitted intervention",
        e2["recoveries"][:e2_plan_stop],
        e3["recoveries"][:e3_plan_stop],
    )
    return {
        "status": "PASS",
        "comparison_scope": "PREFIX_THROUGH_FIRST_ADMITTED_INTERVENTION",
        "first_admitted_query_index": first_admitted,
        "e3_query_count": len(queries),
    }


def validate_ordered_verifier_receipts(
    runtime_dir: str | Path,
    sealed_dir: str | Path,
    *,
    episode_id: str,
) -> dict[str, Any]:
    """Require ordered reset/action receipts plus one final sealed record.

    There must be one HMAC-sealed receipt after reset and one after every logged
    normal or recovery action.  Each receipt is joined to the exact causal
    action and post-action observation by both ID and digest.  Runtime opaque
    receipts and sealed events must have identical order, token, and stop bit.
    """

    runtime = Path(runtime_dir)
    sealed = Path(sealed_dir)
    actions = [
        row
        for row in read_jsonl(runtime / "actions.jsonl")
        if str(row.get("event_type", "")) in _ACTION_EVENT_TYPES
    ]
    observations = [
        row
        for row in read_jsonl(runtime / "environment_events.jsonl")
        if str(row.get("event_type", "")) in _OBSERVATION_EVENT_TYPES
    ]
    terminal = read_jsonl(runtime / "terminal_signals.jsonl")
    expected = _expected_receipt_bindings(actions, observations)

    runtime_bound = [
        row
        for row in terminal
        if str(row.get("event_type", "")) != "episode_final_receipt"
    ]
    runtime_finals = [
        row
        for row in terminal
        if str(row.get("event_type", "")) == "episode_final_receipt"
    ]
    if len(runtime_finals) != 1 or terminal[-1:] != runtime_finals:
        raise SchemaError("runtime must end with exactly one episode_final_receipt")
    if len(runtime_bound) != len(expected):
        raise SchemaError(
            "runtime verifier receipts are not reset-plus-one-per-action"
        )

    verifier_path = sealed / _path_id(episode_id) / "verifier_events.jsonl"
    sealed_records = verify_sealed_stream(verifier_path)
    sealed_finals = [
        row for row in sealed_records if str(row.get("event_kind", "")) == "episode_final"
    ]
    if len(sealed_finals) != 1 or sealed_records[-1:] != sealed_finals:
        raise SchemaError("sealed stream must end with exactly one episode_final")
    sealed_bound = sealed_records[:-1]
    if len(sealed_bound) != len(expected):
        raise SchemaError("sealed verifier evidence is final-only or misses an action")
    if len(terminal) != len(sealed_records):
        raise SchemaError("runtime/sealed verifier receipt counts differ")

    for index, (expected_binding, runtime_record, sealed_record) in enumerate(
        zip(expected, runtime_bound, sealed_bound)
    ):
        runtime_payload = _event_payload(runtime_record)
        runtime_binding = _normalize_binding(runtime_payload.get("receipt_binding"))
        sealed_evidence = sealed_record.get("evidence")
        if not isinstance(sealed_evidence, Mapping):
            raise SchemaError(f"sealed receipt[{index}] evidence is not an object")
        sealed_binding = _normalize_binding(sealed_evidence.get("runtime_binding"))
        if runtime_binding != expected_binding or sealed_binding != expected_binding:
            raise SchemaError(
                f"verifier receipt[{index}] is not bound to its causal action/observation"
            )
        if str(runtime_record.get("event_type", "")) != expected_binding["receipt_kind"]:
            raise SchemaError(f"runtime verifier receipt[{index}] kind is out of order")
        if str(sealed_record.get("event_kind", "")) != expected_binding["receipt_kind"]:
            raise SchemaError(f"sealed verifier receipt[{index}] kind is out of order")

    for index, (runtime_record, sealed_record) in enumerate(
        zip(terminal, sealed_records)
    ):
        payload = _event_payload(runtime_record)
        if str(payload.get("event_id", "")) != str(sealed_record.get("event_id", "")):
            raise SchemaError(f"runtime/sealed verifier event order differs at {index}")
        if payload.get("token_sha256") != sealed_record.get("opaque_token_sha256"):
            raise SchemaError(f"runtime/sealed verifier token differs at {index}")
        if _exact_bool(payload.get("terminate"), context="runtime terminate") != _exact_bool(
            sealed_record.get("should_terminate"), context="sealed terminate"
        ):
            raise SchemaError(f"runtime/sealed verifier terminal bit differs at {index}")
    if "runtime_binding" in sealed_finals[0].get("evidence", {}):
        raise SchemaError("episode_final cannot substitute for a transition receipt")
    return {
        "status": "PASS",
        "bound_receipt_count": len(expected),
        "action_receipt_count": len(actions),
        "sealed_event_count": len(sealed_records),
    }


def _load_causal_streams(root: Path, *, require_memory: bool) -> dict[str, list[dict[str, Any]]]:
    required = {
        "actions": "actions.jsonl",
        "observations": "environment_events.jsonl",
        "transitions": "transitions.jsonl",
        "recoveries": "recoveries.jsonl",
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for name, filename in required.items():
        path = root / filename
        if not path.is_file():
            raise SchemaError(f"causal trace is missing {path}")
        rows = read_jsonl(path)
        if name == "observations":
            rows = [
                row
                for row in rows
                if str(row.get("event_type", "")) in _OBSERVATION_EVENT_TYPES
            ]
        output[name] = rows
    memory_path = root / "memory_queries.jsonl"
    if require_memory and not memory_path.is_file():
        raise SchemaError("E3 causal trace lacks memory_queries.jsonl")
    output["memory"] = read_jsonl(memory_path) if memory_path.is_file() else []
    return output


def _parse_e3_memory_events(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shadows: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    expect_shadow = True
    for row in rows:
        event_type = str(row.get("event_type", ""))
        if expect_shadow and event_type == "no_memory_shadow_before_retrieval":
            shadows.append(dict(row))
            expect_shadow = False
        elif not expect_shadow and event_type == "post_failure_query":
            queries.append(dict(row))
            expect_shadow = True
        else:
            raise SchemaError("E3 memory shadow/query order is not strictly alternating")
    if not expect_shadow:
        raise SchemaError("E3 memory stream ends with an unused shadow")
    return shadows, queries


def _recovery_plan_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, dict(row))
        for index, row in enumerate(rows)
        if str(row.get("event_type", "")) == "recovery_plan"
    ]


def _validate_paired_shadow(
    index: int,
    *,
    shadow_record: Mapping[str, Any],
    query_record: Mapping[str, Any],
    e2_plan: Mapping[str, Any],
    e3_plan: Mapping[str, Any],
    admitted: bool,
) -> None:
    shadow = _event_payload(shadow_record)
    query_payload = _event_payload(query_record)
    query_shadow = query_payload.get("shadow_decision")
    e2_payload = _event_payload(e2_plan)
    e3_payload = _event_payload(e3_plan)
    for context, candidate in (
        ("query shadow", query_shadow),
        ("E2 recovery shadow", e2_payload.get("shadow_decision")),
        ("E2 no-memory final", e2_payload.get("final_decision")),
        ("E3 recovery shadow", e3_payload.get("shadow_decision")),
    ):
        if _normalize_semantics(candidate) != _normalize_semantics(shadow):
            raise SchemaError(f"E2/E3 shadow mismatch at query {index}: {context}")
    query_final = query_payload.get("final_decision")
    if not admitted:
        if _normalize_semantics(query_final) != _normalize_semantics(shadow):
            raise SchemaError(f"abstained E3 query {index} changed its decision")
        if _normalize_semantics(e3_payload) != _normalize_semantics(e2_payload):
            raise SchemaError(f"E2/E3 recovery plan diverged after abstained query {index}")


def _query_result(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _event_payload(record)
    nested = payload.get("query_result")
    if not isinstance(nested, Mapping):
        raise SchemaError("E3 post_failure_query lacks query_result")
    return nested


def _assert_semantic_equal(
    context: str,
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> None:
    left_values = [_normalize_event(row) for row in left]
    right_values = [_normalize_event(row) for row in right]
    if left_values != right_values:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(left_values, right_values))
                if pair[0] != pair[1]
            ),
            min(len(left_values), len(right_values)),
        )
        raise SchemaError(
            f"E2/E3 semantic {context} differ at record {mismatch} "
            f"(counts {len(left_values)} != {len(right_values)})"
        )


def _normalize_event(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_type": str(record.get("event_type", "")),
        "payload": _normalize_semantics(_event_payload(record)),
    }


def _normalize_semantics(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(nested_key): _normalize_semantics(item, key=str(nested_key))
            for nested_key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(nested_key) not in _RUNTIME_LINK_KEYS
            and str(nested_key) not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_semantics(item, key=key) for item in value]
    return value


def _find_action_index(rows: Sequence[Mapping[str, Any]], action_id: str) -> int:
    matches = [
        index
        for index, row in enumerate(rows)
        if isinstance(_event_payload(row).get("action"), Mapping)
        and str(_event_payload(row)["action"].get("action_id", "")) == action_id
    ]
    return _single_index(matches, context="admitted-query failed action")


def _find_observation_index(
    rows: Sequence[Mapping[str, Any]], observation_id: str
) -> int:
    matches = [
        index
        for index, row in enumerate(rows)
        if str(_event_payload(row).get("observation_id", "")) == observation_id
    ]
    return _single_index(matches, context="admitted-query post-failure observation")


def _find_transition_index(rows: Sequence[Mapping[str, Any]], action_id: str) -> int:
    matches: list[int] = []
    for index, row in enumerate(rows):
        payload = _event_payload(row)
        transition = payload.get("input")
        action = transition.get("executed_action") if isinstance(transition, Mapping) else None
        if isinstance(action, Mapping) and str(action.get("action_id", "")) == action_id:
            matches.append(index)
    return _single_index(matches, context="admitted-query transition")


def _single_index(matches: Sequence[int], *, context: str) -> int:
    if len(matches) != 1:
        raise SchemaError(f"{context} must occur exactly once, found {len(matches)}")
    return int(matches[0])


def _expected_receipt_bindings(
    actions: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    resets = [row for row in observations if str(row.get("event_type", "")) == "reset"]
    if len(resets) != 1:
        raise SchemaError("runtime receipt validation requires exactly one reset observation")
    reset = _event_payload(resets[0])
    expected = [
        {
            "receipt_kind": "after_reset",
            "observation_id": _required_text(
                reset.get("observation_id"), context="reset observation_id"
            ),
            "observation_sha256": _observation_record_sha256(
                reset,
                context="reset observation",
            ),
            "action_id": None,
            "action_sha256": None,
        }
    ]
    post_observations = [
        row for row in observations if str(row.get("event_type", "")) != "reset"
    ]
    used_observations: set[int] = set()
    for action_index, action_record in enumerate(actions):
        event_type = str(action_record.get("event_type", ""))
        action_payload = _event_payload(action_record)
        action = action_payload.get("action")
        if not isinstance(action, Mapping):
            raise SchemaError(f"action[{action_index}] lacks action object")
        action_id = _required_text(
            action.get("action_id"), context=f"action[{action_index}].action_id"
        )
        matches = [
            (index, row)
            for index, row in enumerate(post_observations)
            if str(_event_payload(row).get("prior_action_id", "")) == action_id
        ]
        if len(matches) != 1:
            raise SchemaError(
                f"action {action_id} must have exactly one causal post observation"
            )
        observation_index, observation_record = matches[0]
        if observation_index in used_observations:
            raise SchemaError("one post observation is bound to multiple actions")
        used_observations.add(observation_index)
        expected_observation_type = (
            "post_action_observation"
            if event_type == "normal_action"
            else "post_recovery_observation"
        )
        if str(observation_record.get("event_type", "")) != expected_observation_type:
            raise SchemaError("normal/recovery action uses the wrong observation stage")
        observation = _event_payload(observation_record)
        expected.append(
            {
                "receipt_kind": (
                    "after_normal_action"
                    if event_type == "normal_action"
                    else "after_recovery_action"
                ),
                "observation_id": _required_text(
                    observation.get("observation_id"),
                    context=f"action[{action_index}] observation_id",
                ),
                "observation_sha256": _observation_record_sha256(
                    observation,
                    context=f"action[{action_index}] observation",
                ),
                "action_id": action_id,
                "action_sha256": _required_sha256(
                    action_payload.get("action_sha256"),
                    context=f"action[{action_index}]",
                ),
            }
        )
    if len(used_observations) != len(post_observations):
        raise SchemaError("runtime has post observations without a logged action")
    return expected


def _normalize_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError("verifier receipt lacks runtime_binding object")
    required = {
        "receipt_kind",
        "observation_id",
        "observation_sha256",
        "action_id",
        "action_sha256",
    }
    missing = required - set(value)
    if missing:
        raise SchemaError(f"verifier runtime_binding misses {sorted(missing)}")
    return {key: value[key] for key in sorted(required)}


def _observation_record_sha256(
    observation: Mapping[str, Any],
    *,
    context: str,
) -> str:
    """Validate/use the pre-redaction digest of the full Observation record."""

    claimed = _required_sha256(
        observation.get("observation_record_sha256"),
        context=f"{context} record",
    )
    record = dict(observation)
    record.pop("observation_record_sha256", None)
    logged_claim = _required_sha256(
        record.pop("logged_observation_sha256", None),
        context=f"{context} logged projection",
    )
    if sha256_json(record) != logged_claim:
        raise SchemaError(f"{context} complete record SHA-256 mismatch")
    # Non-sensitive observations are fully reproducible from the public log,
    # so reject a screenshot-only or otherwise forged commitment immediately.
    # When a field was intentionally redacted, the pre-redaction digest remains
    # the only safe join key and is still protected by the event/artifact hash
    # chains and the sealed receipt.
    if not _contains_redaction(record) and sha256_json(record) != claimed:
        raise SchemaError(f"{context} complete record SHA-256 mismatch")
    return claimed


def _contains_redaction(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("redacted") is True and set(value).issuperset(
            {"redacted", "sha256"}
        ):
            return True
        return any(_contains_redaction(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_redaction(item) for item in value)
    return False


def _event_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("payload", record)
    if not isinstance(value, Mapping):
        raise SchemaError("runtime event payload must be an object")
    return dict(value)


def _required_text(value: Any, *, context: str) -> str:
    text = str(value or "")
    if not text.strip():
        raise SchemaError(f"{context} must be nonempty text")
    return text


def _required_sha256(value: Any, *, context: str) -> str:
    text = _required_text(value, context=f"{context} SHA-256")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise SchemaError(f"{context} must be lowercase hexadecimal SHA-256")
    return text


def _exact_bool(value: Any, *, context: str) -> bool:
    if type(value) is not bool:
        raise SchemaError(f"{context} must be an exact boolean")
    return value


def _path_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    ).strip("-")
    if not cleaned:
        raise SchemaError("episode_id cannot map to an empty sealed path")
    return cleaned[:160]
