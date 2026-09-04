"""Deterministic engineering-only backend for isolated-broker tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from web_agent.benchmarks.base import AdapterExecution
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionEvidence,
    ExecutionStatus,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    VerifierReceiptBinding,
)
from web_agent.runtime.state_reset import (
    REGISTERED_WEBARENA_RESET_COMMITMENTS,
    WebArenaResetStateReceipt,
    webarena_reset_state_digest,
)
from web_agent.eval.table2.common import sha256_json


_FIXTURE_SHA256 = "a" * 64
_FIXTURE_TIMESTAMP = "2026-01-01T00:00:00+00:00"


class _RawFixturePage:
    def __init__(self) -> None:
        self.url = "https://fixture.invalid/start"
        self.actions = 0
        self.closed = False


class FixtureSealedBackend:
    """Owns the raw fixture page; no runtime message returns this object."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("schema_version") != (
            "table2-process-broker-fixture-backend-v1"
        ):
            raise ValueError("fixture backend config version differs")
        self._raw_page = _RawFixturePage()
        self._terminal_events = 0
        screenshot_root = config.get("screenshot_root")
        self._screenshot_root = (
            Path(str(screenshot_root)) if screenshot_root is not None else None
        )
        self._observation_count = 0

    def runtime_reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        episode_id = str(payload["episode_id"])
        task_id = str(payload["task_id"])
        benchmark_version = str(payload["benchmark_version"])
        start_state_id = str(payload["start_state_id"])
        reset_stage_seed = int(payload["reset_stage_seed"])
        commitments = {
            role: _FIXTURE_SHA256
            for role in REGISTERED_WEBARENA_RESET_COMMITMENTS
        }
        reset_state_sha256 = webarena_reset_state_digest(
            attester_id="fixture-reset-attester",
            attester_version="1.0",
            attester_source_sha256=_FIXTURE_SHA256,
            task_id=task_id,
            benchmark_version=benchmark_version,
            start_state_id=start_state_id,
            reset_stage_seed=reset_stage_seed,
            commitments_sha256=commitments,
        )
        observation = self.runtime_observe(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "stage": "reset",
                "prior_action_id": None,
            }
        )["observation"]
        receipt = WebArenaResetStateReceipt(
            episode_id=episode_id,
            task_id=task_id,
            benchmark_version=benchmark_version,
            start_state_id=start_state_id,
            reset_stage_seed=reset_stage_seed,
            attester_id="fixture-reset-attester",
            attester_version="1.0",
            attester_source_sha256=_FIXTURE_SHA256,
            commitments_sha256=commitments,
            reset_state_sha256=reset_state_sha256,
            reset_applied=True,
            oracle_labels_observed=False,
        )
        return {
            "observation": observation,
            "reset_state_receipt": receipt.to_dict(),
        }

    def runtime_observe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        episode_id = str(payload["episode_id"])
        task_id = str(payload["task_id"])
        stage = ObservationStage(str(payload["stage"]))
        prior_action_id = payload["prior_action_id"]
        observation_id = (
            f"{episode_id}:{stage.value}:{self._raw_page.actions + 1}"
        )
        observation = Observation(
                observation_id=observation_id,
                episode_id=episode_id,
                stage=stage,
                screenshot_sha256=_FIXTURE_SHA256,
                screenshot_path=None,
                width=1280,
                height=720,
                url=self._raw_page.url,
                title="Fixture",
                page_state={
                    "schema_version": "table2-browsergym-causal-observation-v1",
                    "visible_text": (
                        f"fixture action count {self._raw_page.actions}"
                    ),
                    "visible_controls": [],
                    "has_browser_error": False,
                    "browser_error_kind": None,
                    "observable_select_controls": [],
                    "recovery_target_evidence": {
                        "schema_version": "oracle-blind-visible-targets-v1",
                        "observation_id": observation_id,
                        "task_id": task_id,
                        "task_goal_sha256": _FIXTURE_SHA256,
                        "registered_visible_targets": [],
                    },
                },
                page_settled=True,
                environment_error=False,
                prior_action_id=(
                    str(prior_action_id) if prior_action_id is not None else None
                ),
            ).to_dict()
        if self._screenshot_root is not None:
            self._observation_count += 1
            screenshot_bytes = (
                b"\x89PNG\r\n\x1a\nfixture-" + observation_id.encode("utf-8")
            )
            digest = hashlib.sha256(screenshot_bytes).hexdigest()
            screenshot_path = self._screenshot_root / (
                f"{self._observation_count:06d}-{digest}.png"
            )
            screenshot_path.write_bytes(screenshot_bytes)
            screenshot_path.chmod(0o400)
            observation["screenshot_sha256"] = digest
            observation["screenshot_path"] = str(screenshot_path)
        return {"observation": observation}

    def runtime_execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action_value = payload.get("action")
        if not isinstance(action_value, Mapping):
            raise ValueError("fixture execution requires an action object")
        action = ConcreteAction.from_dict(action_value)
        if type(action) is not ConcreteAction:
            raise ValueError("fixture execution requires ConcreteAction")
        if action.action_id == "late-unregistered-import":
            # The worker must enforce its source closure after each backend
            # operation, not only while publishing readiness.
            from web_agent.runtime import action_parameters as _late  # noqa: F401
        self._raw_page.actions += 1
        if action.action_type is ActionType.NAVIGATE:
            self._raw_page.url = str(action.parameters["url"])
        if action.action_id == "aliased-result":
            return {"execution": {"score": 1}}
        return {
            "execution": AdapterExecution(
                status=ExecutionStatus.EXECUTED,
                state_changed=True,
                environment_error=False,
                error_kind=None,
                message="browser request completed",
                internal_retry_count=0,
                latency_ms=0.0,
                evidence=ExecutionEvidence(
                    action_id=action.action_id,
                    started_at_utc=_FIXTURE_TIMESTAMP,
                    ended_at_utc=_FIXTURE_TIMESTAMP,
                    status=ExecutionStatus.EXECUTED,
                ),
            ).to_dict()
        }

    def runtime_terminal(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        binding = VerifierReceiptBinding.from_dict(payload["receipt_binding"])
        self._terminal_events += 1
        signal = OpaqueTerminalSignal(
            event_id=f"fixture-terminal-{self._terminal_events}",
            token_sha256=sha256_json(binding.to_dict()),
            terminate=self._raw_page.actions >= 1,
        )
        return {"opaque_terminal_signal": signal.to_dict()}

    def runtime_close(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        del payload
        self._raw_page.closed = True
        return {"closed": True}

    def shutdown(self) -> None:
        self._raw_page.closed = True


def create_backend(config: Mapping[str, Any]) -> FixtureSealedBackend:
    return FixtureSealedBackend(config)


def create_failing_backend(config: Mapping[str, Any]) -> FixtureSealedBackend:
    del config
    raise RuntimeError("fixture startup failure")


def create_unregistered_import_backend(
    config: Mapping[str, Any],
) -> FixtureSealedBackend:
    # Deliberately expands the worker's repository-local import graph.  The
    # readiness closure test requires the broker to reject this factory before
    # publishing READY because this source is not in the launch manifest.
    from web_agent.runtime import action_parameters as _unregistered  # noqa: F401

    return FixtureSealedBackend(config)
