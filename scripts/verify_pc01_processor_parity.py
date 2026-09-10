"""Write a no-model-weight PC-01 training/runtime processor parity receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import canonical_json_bytes, sha256_bytes, sha256_file
from web_agent.eval.table2.pc01_artifacts import (
    PC01ArtifactError,
    build_pc01_base_snapshot_manifest,
    load_pc01_base_snapshot_manifest,
    load_pc01_resolved_config,
    load_pc01_training_action_value_evidence,
    load_pc01_training_environment,
    load_pinned_local_processor,
    load_processor_contract,
    validate_pc01_training_sources,
)
from web_agent.eval.table2.pc01_processor_parity import (
    build_pc01_processor_parity_receipt,
    validate_pc01_processor_parity_receipt,
    write_pc01_processor_parity_receipt,
)
from web_agent.runtime.observation import validate_processor_parity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--processor-contract", required=True, type=Path)
    parser.add_argument("--processor-source", required=True, type=Path)
    parser.add_argument("--base-snapshot-manifest", required=True, type=Path)
    parser.add_argument("--training-environment", required=True, type=Path)
    parser.add_argument(
        "--training-action-value-evidence",
        required=True,
        type=Path,
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pc01_resolved_config(args.resolved_config)
    training_environment_sha256 = sha256_file(args.training_environment)
    training_environment = load_pc01_training_environment(
        args.training_environment,
        run_contract_path=args.run_contract,
    )
    training_environment_record_sha256 = sha256_bytes(
        canonical_json_bytes(training_environment)
    )
    training_source_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(validate_pc01_training_sources())
    )
    training_action_value_evidence = load_pc01_training_action_value_evidence(
        args.training_action_value_evidence,
        report_path=args.report,
        run_contract_path=args.run_contract,
    )
    del training_action_value_evidence
    training_action_value_evidence_sha256 = sha256_file(
        args.training_action_value_evidence
    )
    registered_base_manifest = load_pc01_base_snapshot_manifest(
        args.base_snapshot_manifest
    )
    current_base_manifest = build_pc01_base_snapshot_manifest(
        args.processor_source
    )
    if current_base_manifest != registered_base_manifest:
        raise PC01ArtifactError(
            "current base snapshot bytes differ from the supplied manifest"
        )
    expected_contract = load_processor_contract(args.processor_contract)
    loaded = load_pinned_local_processor(
        config,
        args.processor_source,
        training_environment_sha256=training_environment_sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_action_value_evidence_sha256=(
            training_action_value_evidence_sha256
        ),
    )
    validate_processor_parity(expected_contract, loaded.contract)

    base_manifest = registered_base_manifest
    base_files = {item["path"]: item for item in base_manifest["files"]}
    for processor_file in loaded.artifact_manifest["files"]:
        base_file = base_files.get(processor_file["path"])
        if base_file != processor_file:
            raise PC01ArtifactError(
                "base snapshot manifest and current processor artifact differ at "
                f"{processor_file['path']!r}"
            )

    processor_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(loaded.artifact_manifest)
    )
    receipt = build_pc01_processor_parity_receipt(
        config=config,
        processor=loaded.processor,
        processor_source=args.processor_source,
        processor_contract=loaded.contract,
        implementation_identity=loaded.implementation_identity,
        processor_artifact_manifest_sha256=processor_manifest_sha256,
        base_snapshot_directory_payload_sha256=base_manifest[
            "directory_payload_sha256"
        ],
        training_environment_sha256=training_environment_sha256,
        training_environment_record_sha256=training_environment_record_sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_action_value_evidence_sha256=(
            training_action_value_evidence_sha256
        ),
        run_contract_sha256=sha256_file(args.run_contract),
    )
    receipt = validate_pc01_processor_parity_receipt(
        receipt,
        resolved_config_record_sha256=sha256_bytes(canonical_json_bytes(config)),
        processor_contract_sha256=loaded.contract.record_sha256,
        processor_artifact_manifest_sha256=processor_manifest_sha256,
        base_snapshot_directory_payload_sha256=base_manifest[
            "directory_payload_sha256"
        ],
        training_environment_sha256=training_environment_sha256,
        training_environment_record_sha256=training_environment_record_sha256,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_action_value_evidence_sha256=(
            training_action_value_evidence_sha256
        ),
        run_contract_sha256=sha256_file(args.run_contract),
    )
    destination = write_pc01_processor_parity_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "output": str(destination.resolve()),
                "receipt_sha256": sha256_file(destination),
                "parity_verified": True,
                "model_weights_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
