from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.production_runner import (
    ProductionRunnerError,
    ProductionTable2Runner,
    SeedRuntimeBinding,
)
from web_agent.eval.table2.state_isolation_validation import (
    validate_episode_state_reset,
    validate_included_block_state_isolation,
    validate_webarena_reset_state_receipt,
)
from web_agent.runtime.contracts import SystemID, canonical_sha256
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.state_reset import (
    REGISTERED_STATEFUL_BACKEND_ROLES,
    CallableEpisodeStateResetter,
    EpisodeStateResetRequest,
    FrozenWebArenaResetStateAttester,
    PreBrowserSetupEvidence,
    REGISTERED_WEBARENA_RESET_COMMITMENTS,
    WebArenaResetStateRequest,
    evidence_from_backend_digests,
    webarena_reset_receipt_from_commitments,
)


RESET_SEED = 731_921
RESETTER_SOURCE_SHA256 = "a" * 64
START_STATE_SHA256 = "c" * 64


class _HostileStatefulBackends:
    """A shared backend whose first output changes unless every run resets it."""

    def __init__(self) -> None:
        self.hidden_state = "contaminated-before-first-episode"

    def reset(self, seed: int) -> None:
        self.hidden_state = canonical_sha256({"registered_reset_seed": seed})

    def backend_digests(self) -> dict[str, str]:
        return {
            role: canonical_sha256(
                {"backend_role": role, "initial_state": self.hidden_state}
            )
            for role in REGISTERED_STATEFUL_BACKEND_ROLES
        }

    def first_pre_action_output(self, system_id: SystemID) -> str:
        result = canonical_sha256(
            {
                "system_id": system_id.value,
                "hidden_state": self.hidden_state,
            }
        )
        self.hidden_state = canonical_sha256(
            {"prior": self.hidden_state, "contaminated_by": system_id.value}
        )
        return result


def _resetter(backends: _HostileStatefulBackends) -> CallableEpisodeStateResetter:
    def reset(request: EpisodeStateResetRequest):
        backends.reset(request.reset_stage_seed)
        return evidence_from_backend_digests(
            request,
            resetter_id="fixture-source-attested-resetter",
            resetter_version="v1",
            resetter_source_sha256=RESETTER_SOURCE_SHA256,
            backend_state_sha256=backends.backend_digests(),
        )

    return CallableEpisodeStateResetter(
        resetter_id="fixture-source-attested-resetter",
        resetter_version="v1",
        source_sha256=RESETTER_SOURCE_SHA256,
        callback=reset,
        digest_callback=backends.backend_digests,
    )


def _episode_logs(root: Path, system_id: SystemID) -> EpisodeEventLogs:
    episode_id = f"state-canary:{system_id.value}:task-1:repeat-0:seed-42"
    return EpisodeEventLogs(
        root / system_id.value,
        episode_id=episode_id,
        include_memory=system_id is SystemID.E3,
        system_id=system_id.value,
        task_id="task-1",
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2026-08-31T00:00:00+00:00",
    )


def _run_hostile_order(
    root: Path, order: tuple[SystemID, ...]
) -> tuple[dict[str, str], dict[str, str]]:
    backends = _HostileStatefulBackends()
    resetter = _resetter(backends)
    runner = object.__new__(ProductionTable2Runner)
    runner.manifest = {"campaign_id": "state-canary"}
    state = SimpleNamespace(
        binding=SimpleNamespace(episode_state_resetter=resetter)
    )
    outputs: dict[str, str] = {}
    initial_digests: dict[str, str] = {}
    for system_id in order:
        logs = _episode_logs(root, system_id)
        evidence = runner._reset_episode_backends(
            state=state,
            system_id=system_id,
            repeat_id=0,
            model_seed=42,
            stage_seeds={"reset": RESET_SEED},
            event_logs=logs,
        )
        initial_digests[system_id.value] = evidence.initial_state_sha256
        outputs[system_id.value] = backends.first_pre_action_output(system_id)
    return outputs, initial_digests


def _append_reset(
    logs: EpisodeEventLogs,
    *,
    system_id: SystemID,
    reset_seed: int = RESET_SEED,
    contamination: str = "clean",
) -> None:
    request = EpisodeStateResetRequest(
        episode_id=logs.episode_id,
        system_id=system_id,
        reset_stage_seed=reset_seed,
    )
    backend_digests = {
        role: canonical_sha256(
            {"role": role, "seed": reset_seed, "state": contamination}
        )
        for role in REGISTERED_STATEFUL_BACKEND_ROLES
    }
    evidence = evidence_from_backend_digests(
        request,
        resetter_id="fixture-source-attested-resetter",
        resetter_version="v1",
        resetter_source_sha256=RESETTER_SOURCE_SHA256,
        backend_state_sha256=backend_digests,
    )
    logs.append(
        "environment_events",
        "episode_state_reset",
        {
            "request": request.to_dict(),
            "request_sha256": request.record_sha256,
            "evidence": evidence.to_dict(),
            "evidence_sha256": evidence.record_sha256,
        },
    )


def _append_webarena_reset_receipt(
    logs: EpisodeEventLogs,
    *,
    system_id: SystemID,
    contamination: str = "clean",
) -> None:
    setup = PreBrowserSetupEvidence(
        episode_id=logs.episode_id,
        system_id=system_id,
        timeout_seconds=120.0,
        elapsed_seconds=1.25,
    )
    logs.append(
        "environment_events",
        "pre_browser_setup_boundary",
        {
            "evidence": setup.to_dict(),
            "evidence_sha256": setup.record_sha256,
        },
    )
    request = WebArenaResetStateRequest(
        episode_id=logs.episode_id,
        task_id="task-1",
        benchmark_version="0.14.3",
        start_state_id=START_STATE_SHA256,
        reset_stage_seed=RESET_SEED,
    )
    receipt = webarena_reset_receipt_from_commitments(
        request,
        attester_id="fixture-webarena-reset-attester",
        attester_version="v1",
        attester_source_sha256=RESETTER_SOURCE_SHA256,
        commitments_sha256={
            role: canonical_sha256(
                {"role": role, "reset_state": contamination}
            )
            for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
        },
    )
    logs.append(
        "environment_events",
        "webarena_reset_state_receipt",
        {
            "receipt": receipt.to_dict(),
            "receipt_sha256": receipt.record_sha256,
        },
    )


def test_reset_capability_has_no_task_or_oracle_input_and_covers_all_backends() -> None:
    assert {item.name for item in fields(EpisodeStateResetRequest)} == {
        "episode_id",
        "system_id",
        "reset_stage_seed",
    }
    assert set(REGISTERED_STATEFUL_BACKEND_ROLES) == {
        "selected_checkpoint_policy",
        "selected_backbone_policy",
        "parameter_provider",
        "recovery_planner",
        "post_failure_embedding",
    }
    assert "episode_state_resetter" in {
        item.name for item in fields(SeedRuntimeBinding)
    }
    assert {item.name for item in fields(WebArenaResetStateRequest)} == {
        "episode_id",
        "task_id",
        "benchmark_version",
        "start_state_id",
        "reset_stage_seed",
    }
    assert set(REGISTERED_WEBARENA_RESET_COMMITMENTS) == {
        "service",
        "account",
        "database",
        "start_state",
    }


def test_reset_state_attester_receipt_is_typed_hash_only_and_source_bound() -> None:
    observed: list[WebArenaResetStateRequest] = []

    def attest(request: WebArenaResetStateRequest):
        observed.append(request)
        return webarena_reset_receipt_from_commitments(
            request,
            attester_id="fixture-webarena-reset-attester",
            attester_version="v1",
            attester_source_sha256=RESETTER_SOURCE_SHA256,
            commitments_sha256={
                role: canonical_sha256({"role": role, "state": "clean"})
                for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
            },
        )

    attester = FrozenWebArenaResetStateAttester(
        attester_id="fixture-webarena-reset-attester",
        attester_version="v1",
        source_sha256=RESETTER_SOURCE_SHA256,
        callback=attest,
    )
    request = WebArenaResetStateRequest(
        episode_id="state-canary:E0:task-1:repeat-0:seed-42",
        task_id="task-1",
        benchmark_version="0.14.3",
        start_state_id=START_STATE_SHA256,
        reset_stage_seed=RESET_SEED,
    )
    receipt = attester.attest(request)

    assert observed == [request]
    assert receipt.oracle_labels_observed is False
    assert set(receipt.commitments_sha256) == set(
        REGISTERED_WEBARENA_RESET_COMMITMENTS
    )
    assert not any(
        key in receipt.to_dict()
        for key in ("success", "failure", "progress", "reward", "reference_trajectory")
    )


def test_hostile_backend_first_output_is_independent_of_schedule_order(
    tmp_path: Path,
) -> None:
    canonical_order = (SystemID.E0, SystemID.E1, SystemID.E2, SystemID.E3)
    hostile_order = (SystemID.E3, SystemID.E1, SystemID.E0, SystemID.E2)
    canonical_outputs, canonical_digests = _run_hostile_order(
        tmp_path / "canonical", canonical_order
    )
    hostile_outputs, hostile_digests = _run_hostile_order(
        tmp_path / "hostile", hostile_order
    )

    assert hostile_outputs == canonical_outputs
    assert len(set(canonical_digests.values())) == 1
    assert hostile_digests == canonical_digests


def test_runtime_reset_is_first_and_cannot_repeat_on_one_event_stream(
    tmp_path: Path,
) -> None:
    backends = _HostileStatefulBackends()
    runner = object.__new__(ProductionTable2Runner)
    runner.manifest = {"campaign_id": "state-canary"}
    state = SimpleNamespace(
        binding=SimpleNamespace(episode_state_resetter=_resetter(backends))
    )
    logs = _episode_logs(tmp_path, SystemID.E0)
    runner._reset_episode_backends(
        state=state,
        system_id=SystemID.E0,
        repeat_id=0,
        model_seed=42,
        stage_seeds={"reset": RESET_SEED},
        event_logs=logs,
    )
    with pytest.raises(ProductionRunnerError, match="first environment event"):
        runner._reset_episode_backends(
            state=state,
            system_id=SystemID.E0,
            repeat_id=0,
            model_seed=42,
            stage_seeds={"reset": RESET_SEED},
            event_logs=logs,
        )


def test_package_reset_validator_rejects_missing_repeated_and_mismatched(
    tmp_path: Path,
) -> None:
    missing = _episode_logs(tmp_path / "missing", SystemID.E0)
    missing.append("environment_events", "rng_seed_plan", {"seed": RESET_SEED})
    with pytest.raises(SchemaError, match="exactly one"):
        validate_episode_state_reset(
            tmp_path / "missing" / "E0",
            expected_episode_id=missing.episode_id,
            expected_system_id="E0",
            expected_reset_stage_seed=RESET_SEED,
            attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
        )

    repeated = _episode_logs(tmp_path / "repeated", SystemID.E0)
    _append_reset(repeated, system_id=SystemID.E0)
    _append_reset(repeated, system_id=SystemID.E0)
    with pytest.raises(SchemaError, match="exactly one"):
        validate_episode_state_reset(
            tmp_path / "repeated" / "E0",
            expected_episode_id=repeated.episode_id,
            expected_system_id="E0",
            expected_reset_stage_seed=RESET_SEED,
            attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
        )

    mismatched = _episode_logs(tmp_path / "mismatched", SystemID.E0)
    _append_reset(mismatched, system_id=SystemID.E1)
    with pytest.raises(SchemaError, match="scheduled identity"):
        validate_episode_state_reset(
            tmp_path / "mismatched" / "E0",
            expected_episode_id=mismatched.episode_id,
            expected_system_id="E0",
            expected_reset_stage_seed=RESET_SEED,
            attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
        )

    unattested = _episode_logs(tmp_path / "unattested", SystemID.E0)
    _append_reset(unattested, system_id=SystemID.E0)
    with pytest.raises(SchemaError, match="source.*attestation"):
        validate_episode_state_reset(
            tmp_path / "unattested" / "E0",
            expected_episode_id=unattested.episode_id,
            expected_system_id="E0",
            expected_reset_stage_seed=RESET_SEED,
            attested_source_sha256=frozenset({"b" * 64}),
        )

    corrupt = _episode_logs(tmp_path / "corrupt", SystemID.E0)
    corrupt_request = EpisodeStateResetRequest(
        episode_id=corrupt.episode_id,
        system_id=SystemID.E0,
        reset_stage_seed=RESET_SEED,
    )
    corrupt_evidence = evidence_from_backend_digests(
        corrupt_request,
        resetter_id="fixture-source-attested-resetter",
        resetter_version="v1",
        resetter_source_sha256=RESETTER_SOURCE_SHA256,
        backend_state_sha256={
            role: canonical_sha256({"role": role, "state": "clean"})
            for role in REGISTERED_STATEFUL_BACKEND_ROLES
        },
    ).to_dict()
    corrupt_evidence["initial_state_sha256"] = "f" * 64
    corrupt.append(
        "environment_events",
        "episode_state_reset",
        {
            "request": corrupt_request.to_dict(),
            "request_sha256": corrupt_request.record_sha256,
            "evidence": corrupt_evidence,
            "evidence_sha256": canonical_sha256(corrupt_evidence),
        },
    )
    with pytest.raises(SchemaError, match="initial state digest"):
        validate_episode_state_reset(
            tmp_path / "corrupt" / "E0",
            expected_episode_id=corrupt.episode_id,
            expected_system_id="E0",
            expected_reset_stage_seed=RESET_SEED,
            attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
        )


def test_included_block_rejects_cross_system_initial_state_contamination(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "rerun_0"
    for system_id in SystemID:
        runtime = attempt / system_id.value / "runtime"
        logs = EpisodeEventLogs(
            runtime,
            episode_id=(
                f"state-canary:{system_id.value}:task-1:repeat-0:seed-42"
            ),
            include_memory=system_id is SystemID.E3,
            system_id=system_id.value,
            task_id="task-1",
            repeat_id=0,
            matched_seed=42,
        )
        _append_reset(
            logs,
            system_id=system_id,
            contamination="leaked-E0-state" if system_id is SystemID.E2 else "clean",
        )
        (runtime / "episode_manifest.json").write_text(
            '{"episode_id":"' + logs.episode_id + '"}\n', encoding="utf-8"
        )

    with pytest.raises(SchemaError, match="cross-system/schedule-order"):
        validate_included_block_state_isolation(
            attempt,
            schedule_row={"stage_seeds": {"reset": RESET_SEED}},
            attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
        )


def test_hidden_webarena_receipt_is_required_attested_and_matched(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "single" / "E0"
    logs = _episode_logs(tmp_path / "single", SystemID.E0)
    _append_reset(logs, system_id=SystemID.E0)
    _append_webarena_reset_receipt(logs, system_id=SystemID.E0)
    receipt = validate_webarena_reset_state_receipt(
        runtime,
        expected_episode_id=logs.episode_id,
        expected_system_id="E0",
        expected_task_id="task-1",
        expected_benchmark_version="0.14.3",
        expected_start_state_id=START_STATE_SHA256,
        expected_reset_stage_seed=RESET_SEED,
        attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
    )
    assert receipt.reset_applied is True
    with pytest.raises(SchemaError, match="attester source"):
        validate_webarena_reset_state_receipt(
            runtime,
            expected_episode_id=logs.episode_id,
            expected_system_id="E0",
            expected_task_id="task-1",
            expected_benchmark_version="0.14.3",
            expected_start_state_id=START_STATE_SHA256,
            expected_reset_stage_seed=RESET_SEED,
            attested_source_sha256=frozenset({"f" * 64}),
        )

    attempt = tmp_path / "paired" / "rerun_0"
    for system_id in SystemID:
        runtime_dir = attempt / system_id.value / "runtime"
        paired_logs = EpisodeEventLogs(
            runtime_dir,
            episode_id=(
                f"state-canary:{system_id.value}:task-1:repeat-0:seed-42"
            ),
            include_memory=system_id is SystemID.E3,
            system_id=system_id.value,
            task_id="task-1",
            repeat_id=0,
            matched_seed=42,
        )
        _append_reset(paired_logs, system_id=system_id)
        _append_webarena_reset_receipt(
            paired_logs,
            system_id=system_id,
            contamination="different-database" if system_id is SystemID.E2 else "clean",
        )
        (runtime_dir / "episode_manifest.json").write_text(
            '{"episode_id":"' + paired_logs.episode_id + '"}\n',
            encoding="utf-8",
        )

    with pytest.raises(SchemaError, match="hidden WebArena reset state differs"):
        validate_included_block_state_isolation(
            attempt,
            schedule_row={
                "task_id": "task-1",
                "task_partition": "normal",
                "stage_seeds": {"reset": RESET_SEED},
            },
            attested_source_sha256=frozenset({RESETTER_SOURCE_SHA256}),
        )
