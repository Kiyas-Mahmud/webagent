from __future__ import annotations

import json

import pytest

from web_agent.eval.table2.common import SchemaError, Table2Error
from web_agent.eval.table2.handoff import (
    _campaign_is_pilot,
    _require_external_tree_disjoint,
    prepare_handoff,
    validate_handoff_input_field_closure,
)
from web_agent.eval.table2.package_validator import (
    _validate_protocol_access_boundary,
)
from web_agent.eval.table2.split_deployment_preflight import (
    SINGLE_HOST_TOPOLOGY,
    SPLIT_HOST_TOPOLOGY,
)


def _single_pc01_input() -> dict[str, object]:
    return {
        "schema_version": "table2-handoff-input-v1",
        "campaign_config": "campaign.yaml",
        "dependency_lock": "dependency.lock.json",
        "environment": {},
        "evaluator": {},
        "resolved_task_export": "tasks.json",
        "webarena_task_interface_audit": "task-audit.json",
        "webarena_task_source": "raw-webarena-tasks.json",
        "authorized_raw_webarena_task_source_sha256": "a" * 64,
        "webarena_site_url_map": "task-urls.json",
        "webarena_service_url_map": "service-urls.json",
        "webarena_host_preflight": "preflight.json",
        "webarena_deployment_topology": SINGLE_HOST_TOPOLOGY,
        "pc01_live_deployment_manifest": "live-deployment.json",
        "pc01_live_deployment_evidence_root": "live-evidence",
        "duplicate_audit": "duplicate-audit.json",
        "joint_duplicate_assignment_package": "joint-assignment",
        "p4_preparation_package": "p4-preparation",
        "joint_duplicate_provenance_manifest": "joint-provenance.json",
        "selection_evidence": {},
        "pc01_checkpoint_compatibility_receipt": "pc01-receipt.json",
        "models": [],
        "memory_manifests": [],
        "runner": {},
    }


def test_handoff_input_exact_single_pc01_fixture_is_closed() -> None:
    value = _single_pc01_input()

    assert validate_handoff_input_field_closure(
        value,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        selection_mode="pc01_provisional",
    ) == value


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.__setitem__("invented_authority", True), "unknown"),
        (lambda value: value.pop("p4_preparation_package"), "missing"),
    ],
)
def test_handoff_input_rejects_unknown_or_missing_top_level_fields(
    mutation, expected: str
) -> None:
    value = _single_pc01_input()
    mutation(value)

    with pytest.raises(SchemaError, match=expected):
        validate_handoff_input_field_closure(
            value,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
            selection_mode="pc01_provisional",
        )


def test_handoff_input_enforces_topology_conditional_fields() -> None:
    value = _single_pc01_input()
    value["webarena_deployment_topology"] = SPLIT_HOST_TOPOLOGY
    with pytest.raises(SchemaError, match="expected_bridge_identity"):
        validate_handoff_input_field_closure(
            value,
            deployment_topology=SPLIT_HOST_TOPOLOGY,
            selection_mode="pc01_provisional",
        )

    value["expected_dgx_model_runtime_identity"] = {"host": "dgx"}
    value["expected_bridge_identity"] = {"bridge": "local-dgx"}
    assert validate_handoff_input_field_closure(
        value,
        deployment_topology=SPLIT_HOST_TOPOLOGY,
        selection_mode="pc01_provisional",
    ) == value

    value["webarena_deployment_topology"] = SINGLE_HOST_TOPOLOGY
    with pytest.raises(SchemaError, match="out_of_scope"):
        validate_handoff_input_field_closure(
            value,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
            selection_mode="pc01_provisional",
        )


@pytest.mark.parametrize(
    ("task_source", "requires_raw_authority"),
    [
        ("raw-webarena-tasks.json", True),
        ("raw-webarena-tasks", True),
        ("libwebarena-0.0.4.whl", False),
        ("libwebarena-0.0.4.zip", False),
        ("LIBWEBARENA-0.0.4.WHL", False),
        ("LIBWEBARENA-0.0.4.ZIP", False),
    ],
)
def test_handoff_input_task_source_authority_matrix(
    task_source: str,
    requires_raw_authority: bool,
) -> None:
    value = _single_pc01_input()
    value["webarena_task_source"] = task_source
    if not requires_raw_authority:
        value.pop("authorized_raw_webarena_task_source_sha256")

    assert validate_handoff_input_field_closure(
        value,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        selection_mode="pc01_provisional",
    ) == value

    if requires_raw_authority:
        value.pop("authorized_raw_webarena_task_source_sha256")
    else:
        value["authorized_raw_webarena_task_source_sha256"] = "a" * 64
    with pytest.raises(
        SchemaError, match="authorized_raw_webarena_task_source_sha256"
    ):
        validate_handoff_input_field_closure(
            value,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
            selection_mode="pc01_provisional",
        )


@pytest.mark.parametrize(
    "task_source",
    [
        " libwebarena-0.0.4.whl",
        "libwebarena-0.0.4.whl ",
        "\tlibwebarena-0.0.4.zip\n",
    ],
)
def test_handoff_input_rejects_ambiguous_task_source_whitespace(
    task_source: str,
) -> None:
    value = _single_pc01_input()
    value["webarena_task_source"] = task_source

    with pytest.raises(SchemaError, match="surrounding whitespace"):
        validate_handoff_input_field_closure(
            value,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
            selection_mode="pc01_provisional",
        )


def test_handoff_input_enforces_selection_conditional_receipt() -> None:
    value = _single_pc01_input()
    with pytest.raises(SchemaError, match="pc01_checkpoint_compatibility_receipt"):
        validate_handoff_input_field_closure(
            value,
            deployment_topology=SINGLE_HOST_TOPOLOGY,
            selection_mode="three_candidate_final",
        )

    value.pop("pc01_checkpoint_compatibility_receipt")
    assert validate_handoff_input_field_closure(
        value,
        deployment_topology=SINGLE_HOST_TOPOLOGY,
        selection_mode="three_candidate_final",
    ) == value


def test_checked_final_template_cannot_authorize_handoff() -> None:
    protocol = {
        "protocol_status": "AWAITING_MODEL_PROMOTION",
        "evidence_label": "FINAL_TEMPLATE_ONLY",
        "benchmark": {"allow_locked_reads": False},
        "manual_rescue": {"policy": "forbidden"},
    }
    campaign = {
        "campaign_kind": "locked_final",
        "campaign_mode": "evaluation",
        "evidence_label": "FINAL_LOCKED",
        "manual_rescue": "forbidden",
    }

    with pytest.raises(SchemaError, match="cannot authorize a campaign"):
        _validate_protocol_access_boundary(
            protocol,
            campaign,
            pilot_only=False,
        )


def test_handoff_external_evidence_must_be_bidirectionally_tree_disjoint(
    tmp_path,
) -> None:
    repository = tmp_path / "source" / "repository"
    repository.mkdir(parents=True)
    external = tmp_path / "measured" / "evidence.json"
    external.parent.mkdir()
    external.write_text("{}", encoding="utf-8")
    assert _require_external_tree_disjoint(
        external,
        repository_root=repository,
        field="evidence",
    ) == external.resolve()

    for unsafe in (repository, repository / "inside", repository.parent):
        if not unsafe.exists():
            unsafe.mkdir()
        with pytest.raises(SchemaError, match="tree-disjoint"):
            _require_external_tree_disjoint(
                unsafe,
                repository_root=repository,
                field="evidence",
            )


def test_handoff_external_evidence_rejects_symlink_ancestry(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    measured = tmp_path / "measured"
    measured.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(measured, target_is_directory=True)
    evidence = linked / "evidence.json"
    (measured / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SchemaError, match="symlink ancestry"):
        _require_external_tree_disjoint(
            evidence,
            repository_root=repository,
            field="evidence",
        )


def test_handoff_external_evidence_rejects_hard_link_to_repository(
    tmp_path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    tracked = repository / "authority.json"
    tracked.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "external-authority.json"
    external.hardlink_to(tracked)

    with pytest.raises(SchemaError, match="hard-linked"):
        _require_external_tree_disjoint(
            external,
            repository_root=repository,
            field="evidence",
        )


def test_prepare_handoff_rejects_output_inside_live_evidence_before_write(
    tmp_path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    authority = tmp_path / "external-authority"
    evidence_root = authority / "evidence"
    evidence_root.mkdir(parents=True)
    manifest = authority / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    value = _single_pc01_input()
    value["pc01_live_deployment_manifest"] = str(manifest)
    value["pc01_live_deployment_evidence_root"] = str(evidence_root)
    spec = tmp_path / "handoff-input.json"
    spec.write_text(json.dumps(value), encoding="utf-8")
    forbidden_output = evidence_root / "new-handoff"

    with pytest.raises(Table2Error, match="tree-disjoint"):
        prepare_handoff(
            spec_path=spec,
            output_dir=forbidden_output,
            repository_root=repository,
        )

    assert not forbidden_output.exists()


def test_prepare_handoff_rejects_duplicate_authority_key_before_write(
    tmp_path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    spec = tmp_path / "duplicate-input.json"
    spec.write_text(
        '{"schema_version":"table2-handoff-input-v1",'
        '"campaign_config":"first.yaml",'
        '"campaign_config":"second.yaml"}',
        encoding="utf-8",
    )
    output = tmp_path / "handoff-output"

    with pytest.raises(SchemaError, match="duplicate JSON key"):
        prepare_handoff(
            spec_path=spec,
            output_dir=output,
            repository_root=repository,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("campaign_kind", "evidence_label", "expected"),
    [
        ("engineering_pilot", "PILOT_ONLY", True),
        ("locked_final", "FINAL_LOCKED", False),
    ],
)
def test_handoff_campaign_profile_requires_exact_registered_pair(
    campaign_kind: str,
    evidence_label: str,
    expected: bool,
) -> None:
    assert _campaign_is_pilot(
        {
            "campaign_kind": campaign_kind,
            "evidence_label": evidence_label,
        }
    ) is expected


@pytest.mark.parametrize(
    ("campaign_kind", "evidence_label"),
    [
        ("engineering_pilot", "FINAL_LOCKED"),
        ("locked_final", "PILOT_ONLY"),
        ("other", "PILOT_ONLY"),
    ],
)
def test_handoff_campaign_profile_rejects_mixed_or_unknown_pair(
    campaign_kind: str,
    evidence_label: str,
) -> None:
    with pytest.raises(SchemaError, match="profile is not registered"):
        _campaign_is_pilot(
            {
                "campaign_kind": campaign_kind,
                "evidence_label": evidence_label,
            }
        )
