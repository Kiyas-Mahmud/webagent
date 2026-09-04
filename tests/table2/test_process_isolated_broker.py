from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import json
import secrets
import socket
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from web_agent.benchmarks.base import AdapterExecution
from web_agent.eval.table2.common import (
    SchemaError,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.dependency_lock import (
    build_semantic_dependency_lock,
    validate_semantic_dependency_lock,
)
from web_agent.eval.table2.execution_guard import (
    PC01_PROCESS_BROKER_SOURCE_PATHS,
    assert_pc01_page_broker_production_authorized,
    process_isolated_pc01_page_broker_security_binding,
    validate_dependency_lock_for_environment,
    validate_pc01_page_broker_security_binding,
)
from web_agent.eval.table2.process_broker import (
    PROCESS_BROKER_SOURCE_PATHS,
    ProcessIsolatedBroker,
    _verify_parent_imported_sources,
    _validated_readiness_envelope,
    _validate_shutdown_response,
)
from web_agent.eval.table2.process_broker_protocol import (
    POLICY_SCREENSHOT_TRANSPORT_CONTRACT,
    PROCESS_BROKER_INNER_SCHEMA_REGISTRY,
    PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS,
    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256,
    PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION,
    PROCESS_BROKER_PROTOCOL_VERSION,
    RUNTIME_INNER_SCHEMA_PATHS,
    authenticated_envelope,
    expected_verifier_receipt_binding,
    receive_frame,
    send_frame,
    validate_runtime_request_payload,
    validate_runtime_result,
    validated_policy_screenshot_root,
    verify_authenticated_envelope,
)
from web_agent.eval.table2.process_broker_runtime import (
    ProcessIsolatedRuntimeClient,
)
from web_agent.eval.table2.split_deployment_preflight import SINGLE_HOST_TOPOLOGY
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_PACKAGES,
    run_webarena_host_preflight,
)
from web_agent.eval.table2.webarena_preflight_binding import (
    PREFLIGHT_BINDING_FIELD,
    build_deployment_preflight_binding,
)
from web_agent.runtime.contracts import (
    ActionType,
    ConcreteAction,
    ExecutionEvidence,
    ExecutionStatus,
    Observation,
    ObservationStage,
    VerifierReceiptBinding,
)


ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64
URLS = {
    "WA_SHOPPING": "https://shopping.invalid",
    "WA_SHOPPING_ADMIN": "https://admin.invalid",
    "WA_REDDIT": "https://reddit.invalid",
    "WA_GITLAB": "https://gitlab.invalid",
    "WA_WIKIPEDIA": "https://wikipedia.invalid",
    "WA_MAP": "https://map.invalid",
    "WA_HOMEPAGE": "https://home.invalid",
}
TIMESTAMP = "2026-01-01T00:00:00+00:00"
FIXTURE_BACKEND_CONFIG = {
    "schema_version": "table2-process-broker-fixture-backend-v1"
}


def _navigate_action(
    *,
    action_id: str = "action-1",
    url: str = "https://fixture.invalid/done",
    recovery_attempt_id: str | None = None,
) -> dict:
    return ConcreteAction(
        action_id=action_id,
        source_decision_id="decision-1",
        action_type=ActionType.NAVIGATE,
        parameters={"url": url},
        recovery_attempt_id=recovery_attempt_id,
    ).to_dict()


def _observation(
    *,
    episode_id: str = "episode-1",
    task_id: str = "task-1",
    visible_text: str = "fixture page",
    stage: ObservationStage = ObservationStage.POST_ACTION,
    prior_action_id: str | None = "action-1",
) -> dict:
    observation_id = f"{episode_id}:{stage.value}:1"
    return Observation(
        observation_id=observation_id,
        episode_id=episode_id,
        stage=stage,
        screenshot_sha256=SHA,
        screenshot_path=None,
        width=1280,
        height=720,
        url="https://fixture.invalid/start",
        title="Fixture",
        page_state={
            "schema_version": "table2-browsergym-causal-observation-v1",
            "visible_text": visible_text,
            "visible_controls": [],
            "has_browser_error": False,
            "browser_error_kind": None,
            "observable_select_controls": [],
            "recovery_target_evidence": {
                "schema_version": "oracle-blind-visible-targets-v1",
                "observation_id": observation_id,
                "task_id": task_id,
                "task_goal_sha256": SHA,
                "registered_visible_targets": [],
            },
        },
        page_settled=True,
        environment_error=False,
        prior_action_id=prior_action_id,
    ).to_dict()


def _observe_request(
    *,
    episode_id: str = "episode-1",
    task_id: str = "task-1",
    stage: str = "post_action",
    prior_action_id: str | None = "action-1",
) -> dict:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "stage": stage,
        "prior_action_id": prior_action_id,
    }


def _reset(client: ProcessIsolatedRuntimeClient) -> dict:
    observation = client.reset(
        episode_id="episode-1",
        task_id="task-1",
        benchmark_version="webarena-fixture-v1",
        start_state_id=SHA,
        reset_stage_seed=42,
    )
    signal = client.terminal_signal(
        episode_id="episode-1",
        task_id="task-1",
        receipt_binding=expected_verifier_receipt_binding(
            observation=observation,
            action=None,
        ),
    )
    assert signal.terminate is False
    return observation


def _post_action(
    client: ProcessIsolatedRuntimeClient, *, action_id: str = "action-1"
) -> dict:
    return client.observe(
        **_observe_request(stage="post_action", prior_action_id=action_id)
    )


def _terminal_after(
    client: ProcessIsolatedRuntimeClient,
    *,
    observation: dict,
    action: dict,
):
    return client.terminal_signal(
        episode_id="episode-1",
        task_id="task-1",
        receipt_binding=expected_verifier_receipt_binding(
            observation=observation,
            action=action,
        ),
    )


def _execution(*, action_id: str = "action-1") -> dict:
    return AdapterExecution(
        status=ExecutionStatus.EXECUTED,
        state_changed=True,
        environment_error=False,
        error_kind=None,
        message="browser request completed",
        internal_retry_count=0,
        latency_ms=1.0,
        evidence=ExecutionEvidence(
            action_id=action_id,
            started_at_utc=TIMESTAMP,
            ended_at_utc=TIMESTAMP,
            status=ExecutionStatus.EXECUTED,
        ),
    ).to_dict()


def _broker(
    *,
    policy_screenshot_root: Path | None = None,
    backend_screenshot_root: Path | None = None,
) -> ProcessIsolatedBroker:
    backend_config = dict(FIXTURE_BACKEND_CONFIG)
    if backend_screenshot_root is not None:
        backend_config["screenshot_root"] = str(backend_screenshot_root)
    return ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=(
            "web_agent.eval.table2.process_broker_fixture_backend:create_backend"
        ),
        backend_source_relative_path=(
            "src/web_agent/eval/table2/process_broker_fixture_backend.py"
        ),
        sealed_backend_config=backend_config,
        policy_screenshot_root=policy_screenshot_root,
    )


def test_process_broker_has_distinct_processes_exact_outer_envelopes_and_cleanup() -> None:
    backend_module = "web_agent.eval.table2.process_broker_fixture_backend"
    evaluator_module = "web_agent.eval.table2.webarena_page_state_evaluator"
    preexisting_evaluator_module = sys.modules.get(evaluator_module)
    sys.modules.pop(backend_module, None)
    sys.modules.pop("web_agent.eval.table2.process_broker_worker", None)
    broker = _broker()
    with broker:
        receipt = broker.receipt
        launch_receipt_sha256 = sha256_json(receipt.to_dict())
        with pytest.raises(TypeError):
            receipt.source_files[0]["sha256"] = "b" * 64
        assert receipt.runtime_pid != receipt.sealed_evaluator_pid
        assert receipt.sealed_evaluator_pid == broker._process.pid
        assert receipt.external_deployment_authority is False
        assert receipt.outer_envelope_fields_exact is True
        assert receipt.operation_specific_inner_schemas_registered is True
        assert receipt.loaded_source_closure_enforced is True
        assert receipt.single_episode_task_session_enforced is True
        assert receipt.observation_stage_prior_action_bound is True
        assert receipt.runtime_value_provenance_attested is False
        assert receipt.operation_specific_inner_schema_paths == (
            RUNTIME_INNER_SCHEMA_PATHS
        )
        assert (
            receipt.inner_schema_registry_version
            == PROCESS_BROKER_INNER_SCHEMA_REGISTRY_VERSION
        )
        assert (
            receipt.inner_schema_registry_sha256
            == PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256
        )
        assert receipt.future_promotion_requirements == (
            PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
        )
        assert {
            "src/web_agent/__init__.py",
            "src/web_agent/eval/__init__.py",
            "src/web_agent/eval/table2/__init__.py",
            "src/web_agent/eval/table2/common.py",
        }.issubset({row["relative_path"] for row in receipt.source_files})
        assert backend_module not in sys.modules
        assert "web_agent.eval.table2.process_broker_worker" not in sys.modules
        if preexisting_evaluator_module is None:
            assert evaluator_module not in sys.modules
        else:
            assert sys.modules.get(evaluator_module) is preexisting_evaluator_module
        client = broker.runtime_client()
        assert type(client) is ProcessIsolatedRuntimeClient
        assert not hasattr(client, "evaluate")
        assert not hasattr(client, "raw_page")
        observation = _reset(client)
        assert observation["record_type"] == "Observation"
        assert observation["episode_id"] == "episode-1"
        assert observation["url"] == "https://fixture.invalid/start"
        action = _navigate_action()
        client.execute(
            episode_id="episode-1",
            task_id="task-1",
            action=action,
        )
        post = _post_action(client)
        assert _terminal_after(
            client, observation=post, action=action
        ).terminate
        assert client.close(episode_id="episode-1", task_id="task-1") == {
            "closed": True
        }
    assert broker.cleaned
    cleanup = broker.cleanup_receipt
    assert cleanup.launch_receipt_sha256 == launch_receipt_sha256
    assert cleanup.sealed_evaluator_pid == receipt.sealed_evaluator_pid
    assert cleanup.source_set_sha256 == receipt.source_set_sha256
    assert cleanup.external_deployment_authority is False


def test_process_broker_transports_root_confined_content_addressed_screenshots(
    tmp_path: Path,
) -> None:
    screenshot_root = tmp_path / "screenshots"
    screenshot_root.mkdir()
    broker = _broker(
        policy_screenshot_root=screenshot_root,
        backend_screenshot_root=screenshot_root,
    )
    with broker:
        receipt = broker.receipt
        assert receipt.policy_screenshot_transport_contract == (
            POLICY_SCREENSHOT_TRANSPORT_CONTRACT
        )
        assert receipt.policy_screenshot_root_identity_sha256 == hashlib.sha256(
            str(screenshot_root).encode("utf-8")
        ).hexdigest()
        client = broker.runtime_client()
        reset_observation = _reset(client)
        reset_path = Path(reset_observation["screenshot_path"])
        assert reset_path.parent == screenshot_root
        assert reset_path.name == (
            f"000001-{reset_observation['screenshot_sha256']}.png"
        )
        assert sha256_file(reset_path) == reset_observation["screenshot_sha256"]
        assert reset_path.stat().st_mode & 0o222 == 0

        action = _navigate_action()
        client.execute(
            episode_id="episode-1", task_id="task-1", action=action
        )
        post = _post_action(client)
        assert Path(post["screenshot_path"]).parent == screenshot_root
        assert _terminal_after(
            client, observation=post, action=action
        ).terminate is True
        client.close(episode_id="episode-1", task_id="task-1")


def test_process_broker_rejects_screenshot_from_another_registered_root(
    tmp_path: Path,
) -> None:
    policy_root = tmp_path / "policy" / "screenshots"
    backend_root = tmp_path / "backend" / "screenshots"
    policy_root.mkdir(parents=True)
    backend_root.mkdir(parents=True)
    with _broker(
        policy_screenshot_root=policy_root,
        backend_screenshot_root=backend_root,
    ) as broker:
        client = broker.runtime_client()
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            client.reset(
                episode_id="episode-1",
                task_id="task-1",
                benchmark_version="webarena-fixture-v1",
                start_state_id=SHA,
                reset_stage_seed=42,
            )


def test_policy_screenshot_root_rejects_symlink_and_noncanonical_directory(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "actual" / "screenshots"
    actual.mkdir(parents=True)
    linked = tmp_path / "screenshots"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(Exception, match="non-symlink"):
        validated_policy_screenshot_root(linked)
    with pytest.raises(Exception, match="screenshots directory"):
        validated_policy_screenshot_root(tmp_path / "actual")

    with pytest.raises(Exception, match="requires a screenshot"):
        validate_runtime_result(
            "runtime_observe",
            {"observation": _observation()},
            request_payload=_observe_request(),
            policy_screenshot_root=actual,
        )


def test_policy_screenshot_artifact_rejects_writable_hardlinked_and_symlinked_files(
    tmp_path: Path,
) -> None:
    screenshot_root = tmp_path / "screenshots"
    screenshot_root.mkdir()
    screenshot_bytes = b"\x89PNG\r\n\x1a\nfixture"
    digest = hashlib.sha256(screenshot_bytes).hexdigest()
    request = _observe_request()

    def value_for(path: Path) -> dict:
        observation = _observation()
        observation["screenshot_sha256"] = digest
        observation["screenshot_path"] = str(path)
        return {"observation": observation}

    valid_path = screenshot_root / f"000001-{digest}.png"
    valid_path.write_bytes(screenshot_bytes)
    valid_path.chmod(0o400)
    validate_runtime_result(
        "runtime_observe",
        value_for(valid_path),
        request_payload=request,
        policy_screenshot_root=screenshot_root,
    )

    valid_path.chmod(0o600)
    with pytest.raises(Exception, match="immutable root contract"):
        validate_runtime_result(
            "runtime_observe",
            value_for(valid_path),
            request_payload=request,
            policy_screenshot_root=screenshot_root,
        )
    valid_path.unlink()

    outside = tmp_path / "outside.png"
    outside.write_bytes(screenshot_bytes)
    hardlink = screenshot_root / f"000002-{digest}.png"
    hardlink.hardlink_to(outside)
    hardlink.chmod(0o400)
    with pytest.raises(Exception, match="immutable root contract"):
        validate_runtime_result(
            "runtime_observe",
            value_for(hardlink),
            request_payload=request,
            policy_screenshot_root=screenshot_root,
        )
    hardlink.unlink()
    outside.chmod(0o600)

    symlink = screenshot_root / f"000003-{digest}.png"
    symlink.symlink_to(outside)
    with pytest.raises(Exception, match="immutable root contract"):
        validate_runtime_result(
            "runtime_observe",
            value_for(symlink),
            request_payload=request,
            policy_screenshot_root=screenshot_root,
        )


def test_runtime_cannot_request_evaluator_operation_or_send_raw_page() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        private_request = getattr(
            client, "_ProcessIsolatedRuntimeClient__request"
        )
        with pytest.raises(Exception, match="operation is forbidden"):
            private_request("evaluate_final", {})
        with pytest.raises(Exception, match="forbidden named keys"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action={"raw_page": "forbidden"},
            )


@pytest.mark.parametrize(
    "alias", ["score", "success", "judgment", "oracle", "reward", "evaluator"]
)
def test_runtime_rejects_sensitive_aliases_before_consuming_sequence(alias: str) -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        with pytest.raises(Exception, match="forbidden named keys"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action={alias: "forbidden"},
            )
        client.execute(
            episode_id="episode-1",
            task_id="task-1",
            action=_navigate_action(),
        )
        assert _post_action(client)["record_type"] == "Observation"


def test_server_rejects_extra_request_field_without_sequence_desync() -> None:
    with _broker() as broker:
        key = broker._runtime_key
        body = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "runtime",
            "session_id": broker._session_id,
            "sequence": 0,
            "nonce": secrets.token_hex(32),
            "operation": "runtime_observe",
            "payload": {
                "episode_id": "episode-1",
                "task_id": "task-1",
                "stage": "reset",
                "prior_action_id": None,
                "extra": True,
            },
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(broker._endpoint))
            send_frame(
                connection, authenticated_envelope(body, authentication_key=key)
            )
            rejected = verify_authenticated_envelope(
                receive_frame(connection), authentication_key=key
            )
        assert rejected["status"] == "REJECTED"

        # The invalid authenticated frame did not consume server sequence zero.
        client = broker.runtime_client()
        assert _reset(client)["record_type"] == "Observation"


@pytest.mark.skipif(
    not hasattr(socket, "SO_PEERCRED"), reason="requires Linux peer credentials"
)
def test_server_rejects_authenticated_key_from_wrong_peer_pid() -> None:
    with _broker() as broker:
        child = r"""
import json,secrets,socket,sys
from web_agent.eval.table2.process_broker_protocol import (
    PROCESS_BROKER_PROTOCOL_VERSION, authenticated_envelope,
    receive_frame, send_frame, verify_authenticated_envelope,
)
config=json.loads(sys.stdin.read())
key=bytes.fromhex(config["key_hex"])
body={
    "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
    "role": "runtime",
    "session_id": config["session_id"],
    "sequence": 0,
    "nonce": secrets.token_hex(32),
    "operation": "runtime_observe",
    "payload": {
        "episode_id": "episode-1",
        "task_id": "task-1",
        "stage": "reset",
        "prior_action_id": None,
    },
}
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(config["endpoint"])
    send_frame(connection, authenticated_envelope(body, authentication_key=key))
    response=verify_authenticated_envelope(
        receive_frame(connection), authentication_key=key
    )
print(response.get("status", "ERROR"))
"""
        completed = subprocess.run(
            [sys.executable, "-c", child],
            input=json.dumps(
                {
                    "endpoint": str(broker._endpoint),
                    "key_hex": broker._runtime_key.hex(),
                    "session_id": broker._session_id,
                }
            ),
            text=True,
            capture_output=True,
            check=True,
        )
        assert completed.stdout.strip() == "REJECTED"

        # A rejected peer does not consume the authorized parent's sequence.
        client = broker.runtime_client()
        assert _reset(client)["record_type"] == "Observation"


def test_server_rejects_sensitive_result_and_fails_session_closed() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action=_navigate_action(
                    action_id="aliased-result",
                    url="https://fixture.invalid/aliased-result",
                ),
            )
        with pytest.raises(Exception, match="failed closed"):
            client.terminal_signal(
                episode_id="episode-1",
                task_id="task-1",
                receipt_binding=VerifierReceiptBinding(
                    receipt_kind="after_reset",
                    observation_id="unavailable",
                    observation_sha256=SHA,
                ),
            )
        assert client.close(episode_id="episode-1", task_id="task-1") == {
            "closed": True
        }


def test_broker_binds_one_episode_task_and_pending_action_state() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        reset_observation = _reset(client)
        with pytest.raises(Exception, match="episode/task identity differs"):
            client.terminal_signal(
                episode_id="episode-2",
                task_id="task-1",
                receipt_binding=expected_verifier_receipt_binding(
                    observation=reset_observation,
                    action=None,
                ),
            )

        action = _navigate_action()
        client.execute(
            episode_id="episode-1",
            task_id="task-1",
            action=action,
        )
        with pytest.raises(Exception, match="awaits its post observation"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action=_navigate_action(action_id="action-2"),
            )
        with pytest.raises(Exception, match="differs from pending action"):
            client.observe(
                **_observe_request(
                    stage="post_action", prior_action_id="another-action"
                )
            )
        post = _post_action(client)
        assert _terminal_after(
            client, observation=post, action=action
        ).terminate
        client.close(episode_id="episode-1", task_id="task-1")
        with pytest.raises(Exception, match="session is closed"):
            client.terminal_signal(
                episode_id="episode-1",
                task_id="task-1",
                receipt_binding=expected_verifier_receipt_binding(
                    observation=post,
                    action=action,
                ),
            )


def test_runtime_close_rejects_missing_after_reset_terminal_receipt() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        client.reset(
            episode_id="episode-1",
            task_id="task-1",
            benchmark_version="webarena-fixture-v1",
            start_state_id=SHA,
            reset_stage_seed=42,
        )
        with pytest.raises(Exception, match="pending terminal receipt"):
            client.close(episode_id="episode-1", task_id="task-1")
        private_request = getattr(
            client, "_ProcessIsolatedRuntimeClient__request"
        )
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            private_request(
                "runtime_close",
                {"episode_id": "episode-1", "task_id": "task-1"},
            )


def test_runtime_close_rejects_pending_post_action_observation() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        client.execute(
            episode_id="episode-1",
            task_id="task-1",
            action=_navigate_action(),
        )
        with pytest.raises(Exception, match="awaits its post observation"):
            client.close(episode_id="episode-1", task_id="task-1")
        private_request = getattr(
            client, "_ProcessIsolatedRuntimeClient__request"
        )
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            private_request(
                "runtime_close",
                {"episode_id": "episode-1", "task_id": "task-1"},
            )


def test_runtime_close_accepts_receipt_complete_nonterminal_local_stop() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        assert client.close(episode_id="episode-1", task_id="task-1") == {
            "closed": True
        }


def test_terminal_receipt_must_match_causal_observation_and_cannot_repeat() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        observation = client.reset(
            episode_id="episode-1",
            task_id="task-1",
            benchmark_version="webarena-fixture-v1",
            start_state_id=SHA,
            reset_stage_seed=42,
        )
        binding = expected_verifier_receipt_binding(
            observation=observation, action=None
        )
        wrong = dict(binding)
        wrong["observation_sha256"] = "b" * 64
        with pytest.raises(Exception, match="differs from causal state"):
            client.terminal_signal(
                episode_id="episode-1",
                task_id="task-1",
                receipt_binding=wrong,
            )
        signal = client.terminal_signal(
            episode_id="episode-1",
            task_id="task-1",
            receipt_binding=binding,
        )
        assert signal.terminate is False
        with pytest.raises(Exception, match="duplicate or out of order"):
            client.terminal_signal(
                episode_id="episode-1",
                task_id="task-1",
                receipt_binding=binding,
            )
        client.close(episode_id="episode-1", task_id="task-1")


def test_runtime_client_deep_snapshots_action_and_observation_state() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        reset_observation = client.reset(
            episode_id="episode-1",
            task_id="task-1",
            benchmark_version="webarena-fixture-v1",
            start_state_id=SHA,
            reset_stage_seed=42,
        )
        original_reset = copy.deepcopy(reset_observation)
        reset_observation["page_state"]["visible_text"] = "caller mutation"
        assert client.terminal_signal(
            episode_id="episode-1",
            task_id="task-1",
            receipt_binding=expected_verifier_receipt_binding(
                observation=original_reset, action=None
            ),
        ).terminate is False

        action = _navigate_action()
        original_action = copy.deepcopy(action)
        client.execute(
            episode_id="episode-1", task_id="task-1", action=action
        )
        action["parameters"]["url"] = "https://caller-mutation.invalid"
        post = _post_action(client)
        original_post = copy.deepcopy(post)
        post["page_state"]["visible_text"] = "caller mutation"
        assert client.terminal_signal(
            episode_id="episode-1",
            task_id="task-1",
            receipt_binding=expected_verifier_receipt_binding(
                observation=original_post,
                action=original_action,
            ),
        ).terminate is True
        client.close(episode_id="episode-1", task_id="task-1")


def test_worker_independently_rejects_wrong_terminal_receipt_hash() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        observation = client.reset(
            episode_id="episode-1",
            task_id="task-1",
            benchmark_version="webarena-fixture-v1",
            start_state_id=SHA,
            reset_stage_seed=42,
        )
        wrong = expected_verifier_receipt_binding(
            observation=observation, action=None
        )
        wrong["observation_sha256"] = "b" * 64
        private_request = getattr(
            client, "_ProcessIsolatedRuntimeClient__request"
        )
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            private_request(
                "runtime_terminal",
                {
                    "episode_id": "episode-1",
                    "task_id": "task-1",
                    "receipt_binding": wrong,
                },
            )


def test_broker_requires_recovery_stage_for_recovery_action() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        client.execute(
            episode_id="episode-1",
            task_id="task-1",
            action=_navigate_action(
                action_id="recovery-action-1",
                recovery_attempt_id="recovery-attempt-1",
            ),
        )
        with pytest.raises(Exception, match="differs from pending action"):
            client.observe(
                **_observe_request(
                    stage="post_action", prior_action_id="recovery-action-1"
                )
            )
        observation = client.observe(
            **_observe_request(
                stage="post_recovery", prior_action_id="recovery-action-1"
            )
        )
        assert observation["stage"] == "post_recovery"


def test_worker_rejects_cross_identity_even_via_private_request() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        reset_observation = _reset(client)
        private_request = getattr(
            client, "_ProcessIsolatedRuntimeClient__request"
        )
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            private_request(
                "runtime_terminal",
                {
                    "episode_id": "episode-2",
                    "task_id": "task-2",
                    "receipt_binding": expected_verifier_receipt_binding(
                        observation=reset_observation,
                        action=None,
                    ),
                },
            )


def test_worker_readiness_rejects_unregistered_transitive_source_import() -> None:
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=(
            "web_agent.eval.table2.process_broker_fixture_backend:"
            "create_unregistered_import_backend"
        ),
        backend_source_relative_path=(
            "src/web_agent/eval/table2/process_broker_fixture_backend.py"
        ),
        sealed_backend_config=FIXTURE_BACKEND_CONFIG,
        startup_timeout_seconds=1.0,
    )
    with pytest.raises(Exception):
        broker.start()
    assert broker.cleaned


def test_worker_rejects_unregistered_import_loaded_during_operation() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        _reset(client)
        with pytest.raises(Exception, match="REGISTERED_REQUEST_REJECTED"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action=_navigate_action(action_id="late-unregistered-import"),
            )


def test_result_schema_rejects_extra_fields_and_sensitive_aliases() -> None:
    with pytest.raises(Exception, match="fields differ"):
        validate_runtime_result(
            "runtime_terminal", {"opaque_terminal": False, "extra": False}
        )
    for alias in ("score", "success", "judgment", "oracle", "reward", "evaluator"):
        with pytest.raises(Exception, match="forbidden named keys"):
            validate_runtime_result(
                "runtime_observe", {"observation": {alias: "forbidden"}}
            )


def test_action_inner_schema_rejects_root_parameter_and_type_drift() -> None:
    payload = {
        "episode_id": "episode-1",
        "task_id": "task-1",
        "action": _navigate_action(),
    }
    assert validate_runtime_request_payload("runtime_execute", payload) == payload

    mutations = []
    extra_root = copy.deepcopy(payload)
    extra_root["action"]["extra"] = True
    mutations.append(extra_root)
    wrong_record = copy.deepcopy(payload)
    wrong_record["action"]["record_type"] = "Observation"
    mutations.append(wrong_record)
    extra_parameter = copy.deepcopy(payload)
    extra_parameter["action"]["parameters"]["method"] = "POST"
    mutations.append(extra_parameter)
    unknown_action = copy.deepcopy(payload)
    unknown_action["action"]["action_type"] = "ANSWER"
    mutations.append(unknown_action)
    non_boolean = copy.deepcopy(payload)
    non_boolean["action"]["destructive"] = 0
    mutations.append(non_boolean)

    for mutated in mutations:
        with pytest.raises(Exception):
            validate_runtime_request_payload("runtime_execute", mutated)


def test_action_inner_schema_rejects_destructive_and_bad_recovery_flags() -> None:
    for recovery_attempt_id, destructive in ((None, True), (7, False), ("", False)):
        action = _navigate_action()
        action["recovery_attempt_id"] = recovery_attempt_id
        action["destructive"] = destructive
        with pytest.raises(Exception):
            validate_runtime_request_payload(
                "runtime_execute",
                {
                    "episode_id": "episode-1",
                    "task_id": "task-1",
                    "action": action,
                },
            )


def test_observe_request_schema_requires_exact_temporal_binding() -> None:
    assert validate_runtime_request_payload(
        "runtime_observe", _observe_request()
    ) == _observe_request()
    assert validate_runtime_request_payload(
        "runtime_observe",
        _observe_request(stage="post_action", prior_action_id="action-1"),
    ) == _observe_request(stage="post_action", prior_action_id="action-1")
    for invalid in (
        {"episode_id": "episode-1", "task_id": "task-1"},
        _observe_request(stage="unknown"),
        _observe_request(stage="reset", prior_action_id="action-1"),
        _observe_request(stage="post_action", prior_action_id=None),
    ):
        with pytest.raises(Exception):
            validate_runtime_request_payload("runtime_observe", invalid)


@pytest.mark.parametrize(
    "action",
    [
        ConcreteAction(
            action_id="click",
            source_decision_id="decision",
            action_type=ActionType.CLICK,
            parameters={
                "target_x": 0.2,
                "target_y": 0.2,
                "target_bbox": [0.1, 0.1, 0.2, 0.2],
                "button": "left",
                "click_count": 1,
            },
            bbox=(0.1, 0.1, 0.2, 0.2),
        ),
        ConcreteAction(
            action_id="type",
            source_decision_id="decision",
            action_type=ActionType.TYPE,
            parameters={
                "target_x": 0.2,
                "target_y": 0.2,
                "target_bbox": [0.1, 0.1, 0.2, 0.2],
                "text": "causal input",
            },
            bbox=(0.1, 0.1, 0.2, 0.2),
        ),
        ConcreteAction(
            action_id="select",
            source_decision_id="decision",
            action_type=ActionType.SELECT,
            parameters={
                "target_x": 0.2,
                "target_y": 0.2,
                "target_bbox": [0.1, 0.1, 0.2, 0.2],
                "option": "A",
                "candidate_options": ["A", "B"],
            },
            bbox=(0.1, 0.1, 0.2, 0.2),
        ),
        ConcreteAction(
            action_id="scroll",
            source_decision_id="decision",
            action_type=ActionType.SCROLL,
            parameters={
                "direction": "down",
                "amount": 0.5,
                "container": "viewport",
            },
        ),
        ConcreteAction(
            action_id="navigate",
            source_decision_id="decision",
            action_type=ActionType.NAVIGATE,
            parameters={"url": "https://fixture.invalid/next"},
        ),
        ConcreteAction(
            action_id="key",
            source_decision_id="decision",
            action_type=ActionType.PRESS_KEY,
            parameters={"key": "ENTER"},
        ),
    ],
)
def test_action_inner_schema_accepts_all_six_registered_actions(
    action: ConcreteAction,
) -> None:
    payload = {
        "episode_id": "episode-1",
        "task_id": "task-1",
        "action": action.to_dict(),
    }
    assert validate_runtime_request_payload("runtime_execute", payload) == payload


def test_live_broker_action_schema_is_canonical_for_live_values_and_rejects_fixture_url() -> None:
    # The process broker is a live-browser boundary: all six classes reuse the
    # canonical field/value rules, while the fixture-only navigation scheme is
    # deliberately outside this production subset.
    whitespace_type = ConcreteAction(
        action_id="type-spaces",
        source_decision_id="decision",
        action_type=ActionType.TYPE,
        parameters={
            "target_x": 0.2,
            "target_y": 0.2,
            "target_bbox": [0.1, 0.1, 0.2, 0.2],
            "text": "   ",
        },
        bbox=(0.1, 0.1, 0.2, 0.2),
    )
    validate_runtime_request_payload(
        "runtime_execute",
        {
            "episode_id": "episode-1",
            "task_id": "task-1",
            "action": whitespace_type.to_dict(),
        },
    )
    with pytest.raises(Exception, match="outside the registered URL schema"):
        validate_runtime_request_payload(
            "runtime_execute",
            {
                "episode_id": "episode-1",
                "task_id": "task-1",
                "action": _navigate_action(url="fixture://task/done"),
            },
        )


def test_observation_inner_schema_rejects_arbitrary_and_cross_bound_values() -> None:
    request = _observe_request()
    value = {"observation": _observation()}
    assert (
        validate_runtime_result(
            "runtime_observe", value, request_payload=request
        )
        == value
    )

    mutations = []
    arbitrary_root = copy.deepcopy(value)
    arbitrary_root["observation"]["content"] = "unregistered"
    mutations.append(arbitrary_root)
    arbitrary_page_state = copy.deepcopy(value)
    arbitrary_page_state["observation"]["page_state"]["content"] = "unregistered"
    mutations.append(arbitrary_page_state)
    conflicting_error = copy.deepcopy(value)
    conflicting_error["observation"]["page_state"]["has_browser_error"] = True
    mutations.append(conflicting_error)
    wrong_observation = copy.deepcopy(value)
    wrong_observation["observation"]["page_state"]["recovery_target_evidence"][
        "observation_id"
    ] = "another-observation"
    mutations.append(wrong_observation)

    for mutated in mutations:
        with pytest.raises(Exception):
            validate_runtime_result(
                "runtime_observe", mutated, request_payload=request
            )

    with pytest.raises(Exception, match="another episode/task"):
        validate_runtime_result(
            "runtime_observe",
            value,
            request_payload=_observe_request(episode_id="episode-2"),
        )

    bad_digest = copy.deepcopy(value)
    bad_digest["observation"]["screenshot_sha256"] = "X" * 64
    with pytest.raises(Exception, match="lowercase SHA-256"):
        validate_runtime_result(
            "runtime_observe", bad_digest, request_payload=request
        )

    path_channel = copy.deepcopy(value)
    path_channel["observation"]["screenshot_path"] = "/tmp/unregistered.png"
    with pytest.raises(Exception, match="registered path root"):
        validate_runtime_result(
            "runtime_observe", path_channel, request_payload=request
        )

    unsafe_url = copy.deepcopy(value)
    unsafe_url["observation"]["url"] = "data:text/html,unregistered"
    with pytest.raises(Exception, match="registered URL schema"):
        validate_runtime_result(
            "runtime_observe", unsafe_url, request_payload=request
        )

    wrong_stage = {
        "observation": _observation(
            stage=ObservationStage.POST_RECOVERY,
            prior_action_id="action-1",
        )
    }
    with pytest.raises(Exception, match="stage/prior action differs"):
        validate_runtime_result(
            "runtime_observe", wrong_stage, request_payload=request
        )

    with pytest.raises(Exception, match="requires its validated request binding"):
        validate_runtime_result("runtime_observe", value)

    boolean_dimensions = copy.deepcopy(value)
    boolean_dimensions["observation"]["width"] = True
    with pytest.raises(Exception):
        validate_runtime_result(
            "runtime_observe", boolean_dimensions, request_payload=request
        )


def test_observation_inner_schema_binds_select_projection_to_visible_controls() -> None:
    observation = _observation()
    control = {
        "tag": "select",
        "role": "",
        "input_type": "",
        "name": "choice",
        "text": "A B",
        "target_bbox": [0.1, 0.1, 0.2, 0.2],
        "candidate_options": ["A", "B"],
        "destination": "",
    }
    observation["page_state"]["visible_controls"] = [control]
    observation["page_state"]["observable_select_controls"] = [
        {
            "target_bbox": [0.1, 0.1, 0.2, 0.2],
            "candidate_options": ["A", "B"],
        }
    ]
    value = {"observation": observation}
    request = _observe_request()
    assert (
        validate_runtime_result(
            "runtime_observe", value, request_payload=request
        )
        == value
    )

    mismatched = copy.deepcopy(value)
    mismatched["observation"]["page_state"]["observable_select_controls"][0][
        "candidate_options"
    ] = ["A"]
    with pytest.raises(Exception, match="differ from visible controls"):
        validate_runtime_result(
            "runtime_observe", mismatched, request_payload=request
        )


def test_execution_inner_schema_requires_exact_action_bound_evidence() -> None:
    request = validate_runtime_request_payload(
        "runtime_execute",
        {
            "episode_id": "episode-1",
            "task_id": "task-1",
            "action": _navigate_action(),
        },
    )
    value = {"execution": _execution()}
    assert (
        validate_runtime_result(
            "runtime_execute", value, request_payload=request
        )
        == value
    )

    mutations = []
    arbitrary_root = copy.deepcopy(value)
    arbitrary_root["execution"]["accepted"] = True
    mutations.append(arbitrary_root)
    no_evidence = copy.deepcopy(value)
    no_evidence["execution"]["evidence"] = None
    mutations.append(no_evidence)
    wrong_action = copy.deepcopy(value)
    wrong_action["execution"]["evidence"]["action_id"] = "action-2"
    mutations.append(wrong_action)
    infinite_latency = copy.deepcopy(value)
    infinite_latency["execution"]["latency_ms"] = float("inf")
    mutations.append(infinite_latency)

    for mutated in mutations:
        with pytest.raises(Exception):
            validate_runtime_result(
                "runtime_execute", mutated, request_payload=request
            )

    boolean_retry_count = copy.deepcopy(value)
    boolean_retry_count["execution"]["internal_retry_count"] = False
    with pytest.raises(Exception):
        validate_runtime_result(
            "runtime_execute", boolean_retry_count, request_payload=request
        )


def test_neutral_key_values_are_accepted_without_semantic_provenance_claim() -> None:
    # The positive schema closes arbitrary keys, but visible text is still a
    # scalar whose origin cannot be established by local schema validation.
    value = {
        "observation": _observation(
            visible_text="synthetic raw-like DOM or evaluator-derived value"
        )
    }
    request = _observe_request()
    assert (
        validate_runtime_result(
            "runtime_observe", value, request_payload=request
        )
        == value
    )

    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in PROCESS_BROKER_SOURCE_PATHS
    }
    binding = process_isolated_pc01_page_broker_security_binding(
        source_hashes=source_hashes
    )
    assert binding["operation_specific_inner_schemas_registered"] is True
    assert binding["loaded_source_closure_enforced"] is True
    assert binding["single_episode_task_session_enforced"] is True
    assert binding["observation_stage_prior_action_bound"] is True
    assert binding["runtime_value_provenance_attested"] is False
    assert binding["operation_specific_inner_schema_paths"] == list(
        RUNTIME_INNER_SCHEMA_PATHS
    )
    assert (
        binding["inner_schema_registry_sha256"]
        == PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256
    )
    assert binding["future_promotion_requirements"] == list(
        PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        (b'{"a":1,"a":2}', "duplicate JSON key"),
        (b'{"b":2, "a":1}', "canonical JSON bytes"),
        (b'{"value":NaN}', "canonical JSON"),
    ],
)
def test_receive_frame_rejects_ambiguous_or_noncanonical_json(
    payload: bytes, message: str
) -> None:
    receiver, sender = socket.socketpair()
    try:
        sender.sendall(struct.pack("!I", len(payload)) + payload)
        with pytest.raises(Exception, match=message):
            receive_frame(receiver)
    finally:
        receiver.close()
        sender.close()


def test_runtime_client_poisoned_after_ambiguous_transport_failure(
    tmp_path: Path,
) -> None:
    client = ProcessIsolatedRuntimeClient(
        endpoint=tmp_path / "missing.sock",
        authentication_key=secrets.token_bytes(32),
        session_id=secrets.token_hex(32),
    )
    kwargs = {
        "episode_id": "episode-1",
        "task_id": "task-1",
        "benchmark_version": "webarena-fixture-v1",
        "start_state_id": SHA,
        "reset_stage_seed": 42,
    }
    with pytest.raises(OSError):
        client.reset(**kwargs)
    with pytest.raises(Exception, match="failed closed"):
        client.reset(**kwargs)


@pytest.mark.parametrize(
    "field, value",
    [
        ("benchmark_version", 7),
        ("benchmark_version", " webarena-fixture-v1"),
        ("start_state_id", int("1" * 64)),
        ("start_state_id", "A" * 64),
        ("reset_stage_seed", True),
        ("reset_stage_seed", -1),
        ("reset_stage_seed", 2**63),
    ],
)
def test_reset_request_rejects_type_or_identity_drift_before_backend(
    field: str, value: object
) -> None:
    payload = {
        "episode_id": "episode-1",
        "task_id": "task-1",
        "benchmark_version": "webarena-fixture-v1",
        "start_state_id": SHA,
        "reset_stage_seed": 42,
    }
    payload[field] = value
    with pytest.raises(Exception):
        validate_runtime_request_payload("runtime_reset", payload)


def test_process_broker_static_contract_copies_remain_exactly_aligned() -> None:
    assert PROCESS_BROKER_SOURCE_PATHS == PC01_PROCESS_BROKER_SOURCE_PATHS

    module_name = "_table2_evaluation_bootstrap_contract_test"
    script_path = ROOT / "scripts/run_table2_evaluation.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        assert tuple(module.PC01_PROCESS_BROKER_SOURCE_PATHS) == (
            PROCESS_BROKER_SOURCE_PATHS
        )
        assert tuple(module.PC01_PROCESS_BROKER_INNER_SCHEMA_PATHS) == (
            RUNTIME_INNER_SCHEMA_PATHS
        )
        assert module.PC01_PROCESS_BROKER_INNER_SCHEMA_REGISTRY == list(
            PROCESS_BROKER_INNER_SCHEMA_REGISTRY
        )
        assert module.PC01_PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256 == (
            PROCESS_BROKER_INNER_SCHEMA_REGISTRY_SHA256
        )
    finally:
        sys.modules.pop(module_name, None)


def test_start_failure_always_cleans_process_socket_and_tempdir() -> None:
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=(
            "web_agent.eval.table2.process_broker_fixture_backend:create_failing_backend"
        ),
        backend_source_relative_path=(
            "src/web_agent/eval/table2/process_broker_fixture_backend.py"
        ),
        sealed_backend_config=FIXTURE_BACKEND_CONFIG,
        startup_timeout_seconds=1.0,
    )
    with pytest.raises(Exception):
        broker.start()
    assert broker.cleaned
    assert not broker._endpoint.exists()


def test_parent_imported_module_path_must_match_source_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _broker()
    rows = broker._source_rows()
    monkeypatch.setitem(
        sys.modules,
        "web_agent.eval.table2.process_broker_runtime",
        SimpleNamespace(__file__=str(ROOT / "src/web_agent/eval/table2/common.py")),
    )
    with pytest.raises(SchemaError, match="imported parent broker module identity differs"):
        _verify_parent_imported_sources(ROOT, rows)
    broker.stop(force=True)


def test_readiness_requires_authentication_exact_child_pid_and_fields() -> None:
    key = secrets.token_bytes(32)
    body = {
        "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
        "role": "sealed_readiness",
        "session_id": "session-1",
        "startup_nonce": "nonce-1",
        "status": "READY",
        "evaluator_pid": 123,
        "endpoint_mode": "0o600",
        "sealed_backend_config_sha256": SHA,
        "policy_screenshot_root_identity_sha256": None,
    }
    envelope = authenticated_envelope(body, authentication_key=key)
    assert (
        _validated_readiness_envelope(
            envelope,
            control_key=key,
            session_id="session-1",
            startup_nonce="nonce-1",
            child_pid=123,
            sealed_backend_config_sha256=SHA,
            policy_screenshot_root_identity_sha256=None,
        )
        == 123
    )
    with pytest.raises(Exception, match="authentication failed"):
        _validated_readiness_envelope(
            envelope,
            control_key=secrets.token_bytes(32),
            session_id="session-1",
            startup_nonce="nonce-1",
            child_pid=123,
            sealed_backend_config_sha256=SHA,
            policy_screenshot_root_identity_sha256=None,
        )
    with pytest.raises(Exception, match="readiness failed"):
        _validated_readiness_envelope(
            envelope,
            control_key=key,
            session_id="session-1",
            startup_nonce="nonce-1",
            child_pid=124,
            sealed_backend_config_sha256=SHA,
            policy_screenshot_root_identity_sha256=None,
        )
    extra = authenticated_envelope(
        {**body, "extra": True}, authentication_key=key
    )
    with pytest.raises(Exception, match="fields differ"):
        _validated_readiness_envelope(
            extra,
            control_key=key,
            session_id="session-1",
            startup_nonce="nonce-1",
            child_pid=123,
            sealed_backend_config_sha256=SHA,
            policy_screenshot_root_identity_sha256=None,
        )


def test_shutdown_response_is_bound_to_role_session_sequence_and_nonce() -> None:
    key = secrets.token_bytes(32)
    body = {
        "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
        "role": "sealed_control_response",
        "session_id": "session-1",
        "sequence": 0,
        "nonce": "nonce-1",
        "status": "PASS",
        "result": {"shutdown": True},
    }
    envelope = authenticated_envelope(body, authentication_key=key)
    _validate_shutdown_response(
        envelope,
        control_key=key,
        session_id="session-1",
        sequence=0,
        nonce="nonce-1",
    )
    for changed in (
        {**body, "role": "sealed_runtime_response"},
        {**body, "session_id": "other"},
        {**body, "sequence": 1},
        {**body, "nonce": "other"},
        {**body, "extra": True},
    ):
        with pytest.raises(Exception, match="shutdown was rejected"):
            _validate_shutdown_response(
                authenticated_envelope(changed, authentication_key=key),
                control_key=key,
                session_id="session-1",
                sequence=0,
                nonce="nonce-1",
            )


def test_server_rejects_authenticated_evaluator_name_on_runtime_channel() -> None:
    with _broker() as broker:
        # This deliberately bypasses the public runtime client.  Authentication
        # succeeds, but the worker's independent operation allow-list rejects
        # an evaluator request without returning evaluator data.
        key = getattr(broker, "_runtime_key")
        session_id = getattr(broker, "_session_id")
        endpoint = getattr(broker, "_endpoint")
        nonce = secrets.token_hex(32)
        body = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "runtime",
            "session_id": session_id,
            "sequence": 0,
            "nonce": nonce,
            "operation": "evaluate_final",
            "payload": {},
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(endpoint))
            send_frame(
                connection, authenticated_envelope(body, authentication_key=key)
            )
            response = verify_authenticated_envelope(
                receive_frame(connection), authentication_key=key
            )
        assert response["status"] == "REJECTED"
        assert response["result"] == {}


def test_server_rejects_wrong_authentication_and_replayed_message() -> None:
    with _broker() as broker:
        key = getattr(broker, "_runtime_key")
        wrong_key = secrets.token_bytes(32)
        session_id = getattr(broker, "_session_id")
        endpoint = getattr(broker, "_endpoint")
        body = {
            "protocol_version": PROCESS_BROKER_PROTOCOL_VERSION,
            "role": "runtime",
            "session_id": session_id,
            "sequence": 0,
            "nonce": secrets.token_hex(32),
            "operation": "runtime_reset",
            "payload": {
                "episode_id": "episode-1",
                "task_id": "task-1",
                "benchmark_version": "webarena-fixture-v1",
                "start_state_id": SHA,
                "reset_stage_seed": 42,
            },
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(endpoint))
            send_frame(
                connection,
                authenticated_envelope(body, authentication_key=wrong_key),
            )
            rejected = receive_frame(connection)
        with pytest.raises(Exception, match="authentication failed"):
            verify_authenticated_envelope(rejected, authentication_key=wrong_key)

        envelope = authenticated_envelope(body, authentication_key=key)
        responses = []
        for _ in range(2):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(endpoint))
                send_frame(connection, envelope)
                responses.append(
                    verify_authenticated_envelope(
                        receive_frame(connection), authentication_key=key
                    )
                )
        assert responses[0]["status"] == "PASS"
        assert responses[1]["status"] == "REJECTED"
        assert responses[1]["result"] == {}


def test_source_authority_stays_fail_closed_without_external_receipt() -> None:
    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in PROCESS_BROKER_SOURCE_PATHS
    }
    binding = process_isolated_pc01_page_broker_security_binding(
        source_hashes=source_hashes
    )
    assert (
        validate_pc01_page_broker_security_binding(
            binding, source_hashes=source_hashes
        )
        == binding
    )
    with pytest.raises(SchemaError, match="not external deployment authority"):
        assert_pc01_page_broker_production_authorized(binding)
    forged = dict(binding)
    forged["production_dispatch_authorized"] = True
    with pytest.raises(SchemaError):
        validate_pc01_page_broker_security_binding(forged)


def _preflight() -> dict:
    return run_webarena_host_preflight(
        service_url_map=URLS,
        version_getter=lambda distribution: PINNED_WEBARENA_PACKAGES[distribution],
        browser_probe=lambda: {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture-chromium-1",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
            "screenshot_sha256": SHA,
        },
        service_probe=lambda _url: {"status": "PASS", "http_status": 200},
        live_reset_probe=lambda index: {
            "status": "PASS",
            "task_index": index,
            "seed": 42,
            "goal_sha256": SHA,
            "current_url_sha256": SHA,
            "screenshot_shape": [720, 1280, 3],
            "observation_keys": ["goal", "screenshot", "url"],
            "action_taken": False,
            "reward_read": False,
            "evaluator_output_read": False,
        },
    )


def _environment() -> dict:
    return {
        "benchmark": "webarena",
        "benchmark_version": "fixture-benchmark-1",
        "benchmark_revision": "fixture-revision-1",
        "operating_system": "fixture-linux-1",
        "browser": "chromium",
        "browser_version": "fixture-chromium-1",
        "playwright_version": PINNED_WEBARENA_PACKAGES["playwright"],
        "controller_id": "fixture-controller",
        "controller_version": "fixture-controller-1",
        "environment_adapter_id": "fixture-adapter",
        "environment_adapter_version": "fixture-adapter-1",
        "container_digest": "sha256:fixture-container",
    }


def test_semantic_dependency_lock_cross_checks_measured_values(tmp_path: Path) -> None:
    environment = _environment()
    preflight = _preflight()
    lock = build_semantic_dependency_lock(
        environment=environment,
        deployment_preflight=preflight,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
    )
    assert (
        validate_semantic_dependency_lock(
            lock,
            environment=environment,
            deployment_preflight=preflight,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
        )
        == lock
    )
    changed = dict(environment)
    changed["browser_version"] = "different"
    with pytest.raises(SchemaError, match="Chromium version differs"):
        validate_semantic_dependency_lock(
            lock,
            environment=changed,
            deployment_preflight=preflight,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
        )

    preflight_path = atomic_write_json(tmp_path / "webarena_deployment_preflight.json", preflight)
    url_path = atomic_write_json(tmp_path / "webarena_service_url_map.json", URLS)
    environment[PREFLIGHT_BINDING_FIELD] = build_deployment_preflight_binding(
        evidence_path=preflight_path,
        service_url_map_path=url_path,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        expected_live_reset_task_index=0,
    )
    lock_path = atomic_write_json(tmp_path / "dependency.lock", lock)
    environment["dependency_lock_relative_path"] = "dependency.lock"
    environment["dependency_lock_sha256"] = sha256_file(lock_path)
    environment_path = atomic_write_json(tmp_path / "environment.json", environment)
    assert validate_dependency_lock_for_environment(environment_path) == lock_path

    opaque = atomic_write_json(tmp_path / "dependency.lock", {"hash_only": True})
    environment["dependency_lock_sha256"] = sha256_file(opaque)
    atomic_write_json(environment_path, environment)
    with pytest.raises(SchemaError, match="semantic dependency-lock fields"):
        validate_dependency_lock_for_environment(environment_path)
