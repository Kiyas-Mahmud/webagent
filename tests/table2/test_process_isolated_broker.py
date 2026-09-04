from __future__ import annotations

from pathlib import Path
import json
import secrets
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from web_agent.eval.table2.common import SchemaError, atomic_write_json, sha256_file
from web_agent.eval.table2.dependency_lock import (
    build_semantic_dependency_lock,
    validate_semantic_dependency_lock,
)
from web_agent.eval.table2.execution_guard import (
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
    ARBITRARY_RUNTIME_MAPPING_PATHS,
    PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS,
    PROCESS_BROKER_PROTOCOL_VERSION,
    authenticated_envelope,
    receive_frame,
    send_frame,
    validate_runtime_result,
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


def _broker() -> ProcessIsolatedBroker:
    return ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=(
            "web_agent.eval.table2.process_broker_fixture_backend:create_backend"
        ),
        backend_source_relative_path=(
            "src/web_agent/eval/table2/process_broker_fixture_backend.py"
        ),
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
        assert receipt.runtime_pid != receipt.sealed_evaluator_pid
        assert receipt.sealed_evaluator_pid == broker._process.pid
        assert receipt.external_deployment_authority is False
        assert receipt.outer_envelope_fields_exact is True
        assert receipt.operation_specific_inner_schemas_registered is False
        assert receipt.runtime_value_provenance_attested is False
        assert receipt.arbitrary_nested_mapping_paths == (
            ARBITRARY_RUNTIME_MAPPING_PATHS
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
        observation = client.observe(episode_id="episode-1", task_id="task-1")
        assert observation == {
            "action_count": 0,
            "url": "https://fixture.invalid/start",
        }
        client.execute(
            episode_id="episode-1",
            task_id="task-1",
            action={"url": "https://fixture.invalid/done"},
        )
        assert client.terminal_signal(episode_id="episode-1", task_id="task-1")
        assert client.close(episode_id="episode-1", task_id="task-1") == {
            "closed": True
        }
    assert broker.cleaned
    cleanup = broker.cleanup_receipt
    assert cleanup.sealed_evaluator_pid == receipt.sealed_evaluator_pid
    assert cleanup.source_set_sha256 == receipt.source_set_sha256
    assert cleanup.external_deployment_authority is False


def test_runtime_cannot_request_evaluator_operation_or_send_raw_page() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
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
        with pytest.raises(Exception, match="forbidden named keys"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action={alias: "forbidden"},
            )
        assert client.observe(episode_id="episode-1", task_id="task-1")[
            "action_count"
        ] == 0


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
        assert client.observe(episode_id="episode-1", task_id="task-1")[
            "action_count"
        ] == 0


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
    "payload": {"episode_id": "episode-1", "task_id": "task-1"},
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
        assert client.observe(episode_id="episode-1", task_id="task-1")[
            "action_count"
        ] == 0


def test_server_rejects_sensitive_result_and_remains_sequence_synchronized() -> None:
    with _broker() as broker:
        client = broker.runtime_client()
        with pytest.raises(Exception, match="request rejected"):
            client.execute(
                episode_id="episode-1",
                task_id="task-1",
                action={"url": "https://fixture.invalid/aliased-result"},
            )
        assert client.observe(episode_id="episode-1", task_id="task-1")[
            "action_count"
        ] == 1


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


def test_neutral_key_values_are_accepted_without_semantic_provenance_claim() -> None:
    # This synthetic value could represent raw DOM, evaluator-derived content,
    # or ordinary observation text. The current generic mapping cannot tell.
    value = {
        "observation": {
            "content": "synthetic raw-like DOM or evaluator-derived value"
        }
    }
    assert validate_runtime_result("runtime_observe", value) == value

    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in PROCESS_BROKER_SOURCE_PATHS
    }
    binding = process_isolated_pc01_page_broker_security_binding(
        source_hashes=source_hashes
    )
    assert binding["operation_specific_inner_schemas_registered"] is False
    assert binding["runtime_value_provenance_attested"] is False
    assert binding["arbitrary_nested_mapping_paths"] == list(
        ARBITRARY_RUNTIME_MAPPING_PATHS
    )
    assert binding["future_promotion_requirements"] == list(
        PROCESS_BROKER_FUTURE_PROMOTION_REQUIREMENTS
    )


def test_start_failure_always_cleans_process_socket_and_tempdir() -> None:
    broker = ProcessIsolatedBroker(
        repository_root=ROOT,
        backend_entrypoint=(
            "web_agent.eval.table2.process_broker_fixture_backend:create_failing_backend"
        ),
        backend_source_relative_path=(
            "src/web_agent/eval/table2/process_broker_fixture_backend.py"
        ),
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
    }
    envelope = authenticated_envelope(body, authentication_key=key)
    assert (
        _validated_readiness_envelope(
            envelope,
            control_key=key,
            session_id="session-1",
            startup_nonce="nonce-1",
            child_pid=123,
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
        )
    with pytest.raises(Exception, match="readiness failed"):
        _validated_readiness_envelope(
            envelope,
            control_key=key,
            session_id="session-1",
            startup_nonce="nonce-1",
            child_pid=124,
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
            "operation": "runtime_observe",
            "payload": {"episode_id": "episode-1", "task_id": "task-1"},
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
