from __future__ import annotations

from copy import deepcopy

import pytest

from web_agent.eval.table2.common import SchemaError, sha256_json
from web_agent.eval.table2.split_deployment_preflight import (
    BRIDGE_REQUEST_SCHEMA,
    BRIDGE_RESPONSE_SCHEMA,
    PC01_BACKBONE_ID,
    PC01_BACKBONE_REVISION,
    PC01_BASE_SNAPSHOT_SHA256,
    PC01_CHECKPOINT_SHA256,
    PC01_PROCESSOR_CONTRACT_SHA256,
    PC01_RESOLVED_CONFIG_SHA256,
    SINGLE_HOST_TOPOLOGY,
    SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION,
    SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION,
    SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION,
    SPLIT_BRIDGE_PROTOCOL_ID,
    SPLIT_BRIDGE_PROTOCOL_VERSION,
    SPLIT_HOST_TOPOLOGY,
    build_dgx_model_runtime_identity,
    build_split_deployment_preflight,
    validate_split_deployment_preflight,
    validate_webarena_deployment_preflight,
)
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_PACKAGES,
    run_webarena_host_preflight,
)


URL_MAP = {
    "WA_GITLAB": "http://gitlab.private.example",
    "WA_MAP": "http://map.private.example",
    "WA_REDDIT": "http://reddit.private.example",
    "WA_SHOPPING": "http://shopping.private.example",
    "WA_SHOPPING_ADMIN": "http://admin.private.example",
    "WA_WIKIPEDIA": "http://wikipedia.private.example",
    "WA_HOMEPAGE": "http://homepage.private.example",
}


def _local_browser_pass() -> dict:
    return run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=lambda name: PINNED_WEBARENA_PACKAGES[name],
        browser_probe=lambda: {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture-chromium",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
            "screenshot_sha256": "a" * 64,
        },
        service_probe=lambda _url: {"status": "PASS", "http_status": 200},
        live_reset_probe=lambda index: {
            "status": "PASS",
            "task_index": index,
            "seed": 42,
            "goal_sha256": "b" * 64,
            "current_url_sha256": "c" * 64,
            "screenshot_shape": [720, 1280, 3],
            "observation_keys": ["goal", "screenshot", "url"],
            "action_taken": False,
            "reward_read": False,
            "evaluator_output_read": False,
        },
    )


def _dgx_identity() -> dict:
    dependency_identity = {
        "host": {
            "system": "Linux",
            "release": "fixture-dgx-release",
            "machine": "aarch64",
            "python_version": "3.12.3",
            "python_executable_sha256": "0" * 64,
        },
        "packages": [
            {"distribution": "accelerate", "version": "1.0.0"},
            {"distribution": "bitsandbytes", "version": "0.49.0"},
            {"distribution": "peft", "version": "0.18.0"},
            {"distribution": "torch", "version": "2.13.0"},
            {"distribution": "transformers", "version": "4.57.6"},
        ],
    }
    return build_dgx_model_runtime_identity(
        host_identity_sha256=sha256_json(dependency_identity["host"]),
        dependency_identity=dependency_identity,
        runtime_identity={
            "backbone_id": PC01_BACKBONE_ID,
            "backbone_revision": PC01_BACKBONE_REVISION,
            "base_snapshot_sha256": PC01_BASE_SNAPSHOT_SHA256,
            "checkpoint_sha256": PC01_CHECKPOINT_SHA256,
            "resolved_config_sha256": PC01_RESOLVED_CONFIG_SHA256,
            "processor_contract_sha256": PC01_PROCESSOR_CONTRACT_SHA256,
        },
        runtime_source_files=[
            {"relative_path": "src/runtime.py", "sha256": "1" * 64}
        ],
        runtime_environment={
            "python_version": "3.12.3",
            "python_executable_sha256": "0" * 64,
            "torch_version": "2.13.0",
            "transformers_version": "4.57.6",
            "cuda_available": True,
            "cuda_runtime_version": "13.0",
            "device_type": "cuda",
            "device_name": "NVIDIA GB10",
            "device_count": 1,
            "container_digest": "sha256:" + "d" * 64,
        },
    )


def _bridge_identity(local: dict, dgx: dict) -> dict:
    sources = [
        {"relative_path": "src/bridge/client.py", "sha256": "2" * 64},
        {"relative_path": "src/bridge/server.py", "sha256": "3" * 64},
    ]
    return {
        "schema_version": SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION,
        "record_type": "SplitInferenceBridgeCompatibilityIdentity",
        "bridge_id": "table2-split-bridge",
        "bridge_version": "fixture-v1",
        "protocol_id": SPLIT_BRIDGE_PROTOCOL_ID,
        "protocol_version": SPLIT_BRIDGE_PROTOCOL_VERSION,
        "source_files": sources,
        "source_set_sha256": sha256_json(sources),
        "browser_host_identity_sha256": sha256_json(local["host"]),
        "browser_endpoint_identity_sha256": "4" * 64,
        "dgx_endpoint_identity_sha256": "5" * 64,
        "dgx_model_runtime_identity_sha256": sha256_json(dgx),
        "transport_identity_sha256": "6" * 64,
        "request_schema_sha256": sha256_json(BRIDGE_REQUEST_SCHEMA),
        "response_schema_sha256": sha256_json(BRIDGE_RESPONSE_SCHEMA),
        "reward_envelope_fields_permitted": False,
        "oracle_envelope_fields_permitted": False,
        "evaluator_envelope_fields_permitted": False,
        "payload_content_attested": False,
        "endpoint_origin_attested": False,
    }


def _exchanges() -> list[dict]:
    return [
        {
            "request": {
                "schema_version": SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION,
                "request_id": "request-0",
                "operation": "pre_action_prediction",
                "payload_sha256": "9" * 64,
            },
            "response": {
                "schema_version": SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION,
                "request_id": "request-0",
                "operation": "pre_action_prediction",
                "payload_sha256": "a" * 64,
            },
        },
        {
            "request": {
                "schema_version": SPLIT_BRIDGE_REQUEST_SCHEMA_VERSION,
                "request_id": "request-1",
                "operation": "post_action_diagnosis",
                "payload_sha256": "b" * 64,
            },
            "response": {
                "schema_version": SPLIT_BRIDGE_RESPONSE_SCHEMA_VERSION,
                "request_id": "request-1",
                "operation": "post_action_diagnosis",
                "payload_sha256": "c" * 64,
            },
        },
    ]


def _valid_split() -> tuple[dict, dict, dict, dict]:
    local = _local_browser_pass()
    dgx = _dgx_identity()
    bridge = _bridge_identity(local, dgx)
    evidence = build_split_deployment_preflight(
        local_browser_preflight=local,
        service_url_map=URL_MAP,
        dgx_model_runtime_identity=dgx,
        bridge_identity=bridge,
        exchanges=_exchanges(),
        expected_live_reset_task_index=0,
    )
    return evidence, local, dgx, bridge


def _validate(evidence: dict, dgx: dict, bridge: dict) -> dict:
    return validate_split_deployment_preflight(
        evidence,
        service_url_map=URL_MAP,
        expected_live_reset_task_index=0,
        expected_dgx_model_runtime_identity=dgx,
        expected_bridge_identity=bridge,
    )


def test_split_artifact_records_compatibility_but_remains_dispatch_blocked() -> None:
    evidence, _local, dgx, bridge = _valid_split()

    validated = _validate(evidence, dgx, bridge)

    assert validated == evidence
    assert validated is not evidence
    assert evidence["status"] == "PASS"
    assert evidence["campaign_eligible"] is False
    assert evidence["dispatch_authorized"] is False
    assert evidence["dispatch_blocker"].startswith("BLOCKED_DGX_")
    assert evidence["failed_dgx_routing_report_used"] is False
    assert evidence["deployment_topology"] == SPLIT_HOST_TOPOLOGY
    assert evidence["bridge_transcript"]["entry_count"] == 2
    assert evidence["bridge_identity"]["payload_content_attested"] is False
    assert evidence["bridge_identity"]["endpoint_origin_attested"] is False
    assert "NOT_PAYLOAD_CONTENT_OR_ORIGIN_PROOF" in evidence[
        "bridge_transcript"
    ]["claim_scope"]
    assert evidence["bridge_transcript"]["forbidden_envelope_field_totals"] == {
        "reward": 0,
        "oracle": 0,
        "evaluator": 0,
    }


def test_dispatcher_validates_both_registered_topologies() -> None:
    split, local, dgx, bridge = _valid_split()
    assert validate_webarena_deployment_preflight(
        local,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        service_url_map=URL_MAP,
    ) == local
    assert validate_webarena_deployment_preflight(
        split,
        deployment_topology=SPLIT_HOST_TOPOLOGY,
        service_url_map=URL_MAP,
        expected_dgx_model_runtime_identity=dgx,
        expected_bridge_identity=bridge,
    ) == split


def test_failed_dgx_routing_report_cannot_authorize_split() -> None:
    failed = run_webarena_host_preflight(
        service_url_map=URL_MAP,
        version_getter=lambda name: PINNED_WEBARENA_PACKAGES[name],
        browser_probe=lambda: {"status": "FAIL", "error_type": "Unsupported"},
        service_probe=lambda _url: {"status": "PASS", "http_status": 200},
        live_reset_probe=lambda _index: {"status": "FAIL"},
    )
    dgx = _dgx_identity()
    bridge = _bridge_identity(_local_browser_pass(), dgx)

    with pytest.raises(SchemaError, match="status"):
        build_split_deployment_preflight(
            local_browser_preflight=failed,
            service_url_map=URL_MAP,
            dgx_model_runtime_identity=dgx,
            bridge_identity=bridge,
            exchanges=_exchanges(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "reward",
        "score",
        "oracle",
        "success",
        "task_outcome",
        "judgment",
        "label",
        "verifier_output",
        "evaluator_output",
    ],
)
@pytest.mark.parametrize("direction", ["request", "response"])
def test_generator_rejects_forbidden_fields_in_either_direction(
    field: str, direction: str
) -> None:
    local = _local_browser_pass()
    dgx = _dgx_identity()
    bridge = _bridge_identity(local, dgx)
    exchanges = _exchanges()
    exchanges[0][direction][field] = 0

    with pytest.raises(SchemaError, match="must be exactly zero"):
        build_split_deployment_preflight(
            local_browser_preflight=local,
            service_url_map=URL_MAP,
            dgx_model_runtime_identity=dgx,
            bridge_identity=bridge,
            exchanges=exchanges,
        )


def test_split_compatibility_validation_requires_both_pinned_identities() -> None:
    evidence, _local, dgx, bridge = _valid_split()

    with pytest.raises(SchemaError, match="expected DGX"):
        validate_webarena_deployment_preflight(
            evidence,
            deployment_topology=SPLIT_HOST_TOPOLOGY,
            service_url_map=URL_MAP,
            expected_bridge_identity=bridge,
        )
    with pytest.raises(SchemaError, match="expected bridge"):
        validate_webarena_deployment_preflight(
            evidence,
            deployment_topology=SPLIT_HOST_TOPOLOGY,
            service_url_map=URL_MAP,
            expected_dgx_model_runtime_identity=dgx,
        )


def test_split_validation_rejects_a_different_expected_dgx_runtime() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    changed = deepcopy(dgx)
    changed["runtime_identity"]["checkpoint_sha256"] = "0" * 64
    changed["runtime_identity_sha256"] = sha256_json(changed["runtime_identity"])

    with pytest.raises(SchemaError, match="frozen expectation"):
        _validate(evidence, changed, bridge)


def test_split_rejects_missing_or_rehashed_fabricated_dgx_inventory() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    missing = deepcopy(evidence)
    missing["dgx_model_runtime_identity"]["dependency_identity"]["packages"] = []
    missing["dgx_model_runtime_identity"]["dependency_identity_sha256"] = sha256_json(
        missing["dgx_model_runtime_identity"]["dependency_identity"]
    )
    missing["dgx_model_runtime_identity_sha256"] = sha256_json(
        missing["dgx_model_runtime_identity"]
    )
    with pytest.raises(SchemaError, match="packages must be a nonempty"):
        _validate(missing, dgx, bridge)

    fabricated = deepcopy(evidence)
    fabricated_host = fabricated["dgx_model_runtime_identity"][
        "dependency_identity"
    ]["host"]
    fabricated_host["machine"] = "x86_64"
    fabricated["dgx_model_runtime_identity"]["dependency_identity_sha256"] = (
        sha256_json(fabricated["dgx_model_runtime_identity"]["dependency_identity"])
    )
    fabricated["dgx_model_runtime_identity"]["host_identity_sha256"] = sha256_json(
        fabricated_host
    )
    fabricated["dgx_model_runtime_identity_sha256"] = sha256_json(
        fabricated["dgx_model_runtime_identity"]
    )
    with pytest.raises(SchemaError, match="frozen expectation"):
        _validate(fabricated, dgx, bridge)


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_seed", 43), ("evaluation_mode", False), ("weights_mutated", True)],
)
def test_split_rejects_unsafe_dgx_runtime_claims(field: str, value: object) -> None:
    evidence, _local, dgx, bridge = _valid_split()
    tampered = deepcopy(evidence)
    tampered["dgx_model_runtime_identity"][field] = value
    tampered["dgx_model_runtime_identity_sha256"] = sha256_json(
        tampered["dgx_model_runtime_identity"]
    )

    with pytest.raises(SchemaError, match=field):
        _validate(tampered, dgx, bridge)


def test_split_validation_rejects_a_different_expected_bridge() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    changed = dict(bridge)
    changed["dgx_endpoint_identity_sha256"] = "0" * 64

    with pytest.raises(SchemaError, match="frozen expected identity"):
        _validate(evidence, dgx, changed)


def test_split_validation_rejects_bridge_protocol_or_source_drift() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    protocol_drift = deepcopy(evidence)
    protocol_drift["bridge_identity"]["protocol_version"] = "v999"
    protocol_drift["bridge_identity_sha256"] = sha256_json(
        protocol_drift["bridge_identity"]
    )
    with pytest.raises(SchemaError, match="protocol_version"):
        _validate(protocol_drift, dgx, bridge)

    source_drift = deepcopy(evidence)
    source_drift["bridge_identity"]["source_files"][0]["sha256"] = "0" * 64
    source_drift["bridge_identity_sha256"] = sha256_json(
        source_drift["bridge_identity"]
    )
    with pytest.raises(SchemaError, match="source-set hash"):
        _validate(source_drift, dgx, bridge)


def test_split_validation_rejects_hash_chain_tampering() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    tampered = deepcopy(evidence)
    tampered["bridge_transcript"]["entries"][1]["previous_entry_sha256"] = (
        "0" * 64
    )
    tampered["bridge_transcript_sha256"] = sha256_json(
        tampered["bridge_transcript"]
    )

    with pytest.raises(SchemaError, match="previous-entry hash"):
        _validate(tampered, dgx, bridge)


def test_split_validation_rejects_nonzero_forbidden_claim_after_rehash() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    tampered = deepcopy(evidence)
    entry = tampered["bridge_transcript"]["entries"][0]
    entry["request_forbidden_envelope_field_counts"]["oracle"] = 1
    unhashed = {key: value for key, value in entry.items() if key != "entry_sha256"}
    entry["entry_sha256"] = sha256_json(unhashed)
    tampered["bridge_transcript_sha256"] = sha256_json(
        tampered["bridge_transcript"]
    )

    with pytest.raises(SchemaError, match="oracle must be exactly zero"):
        _validate(tampered, dgx, bridge)


def test_failed_routing_report_cannot_be_added_to_authorization_closure() -> None:
    evidence, _local, dgx, bridge = _valid_split()
    evidence["failed_dgx_routing_report"] = {"status": "FAIL"}

    with pytest.raises(SchemaError, match="registered closure"):
        _validate(evidence, dgx, bridge)


def test_bridge_envelopes_require_registered_schema_and_response_binding() -> None:
    local = _local_browser_pass()
    dgx = _dgx_identity()
    bridge = _bridge_identity(local, dgx)

    schema_drift = _exchanges()
    schema_drift[0]["request"]["schema_version"] = "invented"
    with pytest.raises(SchemaError, match="schema version"):
        build_split_deployment_preflight(
            local_browser_preflight=local,
            service_url_map=URL_MAP,
            dgx_model_runtime_identity=dgx,
            bridge_identity=bridge,
            exchanges=schema_drift,
        )

    response_drift = _exchanges()
    response_drift[0]["response"]["request_id"] = "different-request"
    with pytest.raises(SchemaError, match="bound to request ID and operation"):
        build_split_deployment_preflight(
            local_browser_preflight=local,
            service_url_map=URL_MAP,
            dgx_model_runtime_identity=dgx,
            bridge_identity=bridge,
            exchanges=response_drift,
        )


def test_bridge_rejects_recursive_oracle_alias_and_unregistered_schema_hash() -> None:
    local = _local_browser_pass()
    dgx = _dgx_identity()
    bridge = _bridge_identity(local, dgx)
    nested_alias = _exchanges()
    nested_alias[0]["request"]["metadata"] = {
        "task_success_judgement": True
    }
    with pytest.raises(SchemaError, match="must be exactly zero"):
        build_split_deployment_preflight(
            local_browser_preflight=local,
            service_url_map=URL_MAP,
            dgx_model_runtime_identity=dgx,
            bridge_identity=bridge,
            exchanges=nested_alias,
        )

    schema_hash_drift = deepcopy(bridge)
    schema_hash_drift["request_schema_sha256"] = "f" * 64
    with pytest.raises(SchemaError, match="request schema hash"):
        build_split_deployment_preflight(
            local_browser_preflight=local,
            service_url_map=URL_MAP,
            dgx_model_runtime_identity=dgx,
            bridge_identity=schema_hash_drift,
            exchanges=_exchanges(),
        )


@pytest.mark.parametrize(
    ("nested_field", "derived_field", "error"),
    [
        ("runtime_source_files", "runtime_source_set_sha256", "source-set hash"),
        ("runtime_environment", "runtime_environment_sha256", "environment hash"),
    ],
)
def test_dgx_runtime_derived_hashes_reject_nested_tampering(
    nested_field: str, derived_field: str, error: str
) -> None:
    evidence, _local, dgx, bridge = _valid_split()
    tampered = deepcopy(evidence)
    nested = tampered["dgx_model_runtime_identity"][nested_field]
    if nested_field == "runtime_source_files":
        nested[0]["sha256"] = "e" * 64
    else:
        nested["device_name"] = "different-device"
    # Rehash the outer record but deliberately leave the derived nested binding
    # unchanged; validation must not treat a caller-supplied outer hash as proof.
    tampered["dgx_model_runtime_identity_sha256"] = sha256_json(
        tampered["dgx_model_runtime_identity"]
    )
    assert tampered["dgx_model_runtime_identity"][derived_field] == dgx[derived_field]
    with pytest.raises(SchemaError, match=error):
        _validate(tampered, dgx, bridge)


def test_unknown_topology_is_rejected() -> None:
    evidence, _local, _dgx, _bridge = _valid_split()

    with pytest.raises(SchemaError, match="unregistered WebArena deployment topology"):
        validate_webarena_deployment_preflight(
            evidence,
            deployment_topology="AUTO",
            service_url_map=URL_MAP,
        )
