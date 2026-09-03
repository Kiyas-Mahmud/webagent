"""Recompute and bind registered Gold v2.8 model-selection evidence.

The Table 2 handoff must not accept a model manifest merely because it says
``validation_only``.  The provisional profile independently replays PC-01's
within-run epoch gate and binds the registered epoch-6 bytes.  The later final
profile replays all three immutable seed-42 reports and supplied comparison
outputs.  Evidence from one profile cannot authorize the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import yaml

from web_agent.train.selection import (
    ALL_GATES_THEN_OUTCOME_RULE,
    controlled_quality_gates,
    require_selected_checkpoint,
)
from web_agent.config import load_config

from .common import (
    SchemaError,
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
)
from .pc01_artifacts import (
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_MODEL_ID,
)


SELECTION_EVIDENCE_SCHEMA_VERSION = "table2-validation-selection-evidence-v3"
PC01_PROVISIONAL_SELECTION_MODE = "pc01_provisional"
THREE_CANDIDATE_FINAL_SELECTION_MODE = "three_candidate_final"
PC01_EXPECTED_SELECTED_EPOCH = 6
CANDIDATE_MODEL_IDS = (
    "qwen2vl_2b_gold_v2_8_dgx",
    "qwen25vl_7b_gold_v2_8_dgx",
    "internvl35_8b_gold_v2_8_dgx",
)
COMPARISON_RULE = (
    "all_quality_gates_then_outcome_mcc; ties: recovery_outcome_mcc, "
    "action_macro_f1, lower_outcome_ece, model_id"
)
EXPECTED_TRAIN_ROWS = 24_107
EXPECTED_VALIDATION_ROWS = 7_861
EXPECTED_SUPPLEMENT_ROWS = 194
EXPECTED_MAXIMUM_EPOCHS = 10
EXPECTED_CANDIDATE_SEED = 42


def _selection_policy(selection_mode: str) -> dict[str, Any]:
    if selection_mode == PC01_PROVISIONAL_SELECTION_MODE:
        return {
            "candidate_model_ids": (PC01_MODEL_ID,),
            "selection_status": "PROVISIONAL_PC01_PILOT",
            "evidence_label": "PILOT_ONLY",
            "provisional": True,
            "final_campaign_eligible": False,
            "comparison_performed": False,
        }
    if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE:
        return {
            "candidate_model_ids": CANDIDATE_MODEL_IDS,
            "selection_status": "FINAL_THREE_CANDIDATE",
            "evidence_label": "FINAL_SELECTION",
            "provisional": False,
            "final_campaign_eligible": True,
            "comparison_performed": True,
        }
    raise SchemaError(f"unregistered selection evidence mode: {selection_mode!r}")


def stage_selection_evidence(
    *,
    selection_spec: object,
    spec_path: Path,
    output_dir: Path,
    repository_root: Path,
    selected_model_manifest: Mapping[str, Any],
    selection_mode: str = THREE_CANDIDATE_FINAL_SELECTION_MODE,
) -> Path:
    """Validate external selection artifacts and stage an immutable package."""

    policy = _selection_policy(selection_mode)
    candidate_model_ids = tuple(policy["candidate_model_ids"])
    if not isinstance(selection_spec, Mapping):
        raise SchemaError("handoff input requires selection_evidence mapping")
    declared_mode = selection_spec.get("mode")
    if declared_mode is not None and declared_mode != selection_mode:
        raise SchemaError(
            "selection_evidence.mode differs from the frozen protocol selection mode"
        )
    candidate_specs = selection_spec.get("candidate_runs")
    if not isinstance(candidate_specs, Sequence) or isinstance(
        candidate_specs, (str, bytes)
    ):
        raise SchemaError("selection_evidence.candidate_runs must be an array")
    if len(candidate_specs) != len(candidate_model_ids):
        raise SchemaError(
            f"{selection_mode} selection evidence requires exactly "
            f"{len(candidate_model_ids)} registered candidate run(s)"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []
    git_commits: set[str] = set()
    seen: set[str] = set()

    for index, raw in enumerate(candidate_specs):
        if not isinstance(raw, Mapping):
            raise SchemaError(f"selection candidate_runs[{index}] must be an object")
        model_id = str(raw.get("model_id") or "")
        if model_id not in candidate_model_ids or model_id in seen:
            raise SchemaError("selection candidates are missing, duplicated, or unregistered")
        seen.add(model_id)
        contract_source = _input_path(
            spec_path,
            raw.get("run_contract"),
            field=f"selection_evidence.candidate_runs[{index}].run_contract",
        )
        report_source = _input_path(
            spec_path,
            raw.get("full_report"),
            field=f"selection_evidence.candidate_runs[{index}].full_report",
        )
        resolved_config_source = _input_path(
            spec_path,
            raw.get("resolved_config"),
            field=(
                f"selection_evidence.candidate_runs[{index}].resolved_config"
            ),
        )
        checkpoint_source = _input_path(
            spec_path,
            raw.get("selected_checkpoint"),
            field=(
                f"selection_evidence.candidate_runs[{index}].selected_checkpoint"
            ),
        )
        contract = read_json(contract_source)
        report = read_json(report_source)
        resolved_config = read_json(resolved_config_source)
        config_relative = _validate_run_contract(
            contract,
            model_id=model_id,
            repository_root=repository_root,
        )
        _validate_resolved_config_identity(
            resolved_config,
            contract=contract,
            model_id=model_id,
        )
        candidate_row = _recompute_candidate(
            report,
            model_id=model_id,
            contract=contract,
        )
        if sha256_json(resolved_config) != candidate_row["resolved_config_sha256"]:
            raise SchemaError(
                f"{model_id} resolved-config artifact differs from its full report"
            )
        actual_checkpoint_sha256 = sha256_file(checkpoint_source)
        if actual_checkpoint_sha256 != candidate_row["checkpoint_sha256"]:
            raise SchemaError(
                f"{model_id} selected checkpoint bytes differ from its full report"
            )
        git_commits.add(str(contract["git_commit"]))

        candidate_dir = output_dir / "candidates" / model_id
        contract_destination = candidate_dir / "run_contract.json"
        report_destination = candidate_dir / "full_report.json"
        resolved_config_destination = candidate_dir / "resolved_config.json"
        config_source = (repository_root / config_relative).resolve()
        config_destination = candidate_dir / "registered_config.yaml"
        checkpoint_identity_destination = (
            candidate_dir / "selected_checkpoint_identity.json"
        )
        for source, destination in (
            (contract_source, contract_destination),
            (report_source, report_destination),
            (resolved_config_source, resolved_config_destination),
            (config_source, config_destination),
        ):
            _copy_exact(source, destination)
            artifacts.append(
                {
                    "path": destination.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(destination),
                }
            )
        checkpoint_identity = {
            "schema_version": "table2-selection-checkpoint-identity-v1",
            "model_id": model_id,
            "selected_epoch": candidate_row["selected_epoch"],
            "checkpoint_sha256": actual_checkpoint_sha256,
            "checkpoint_bytes": checkpoint_source.stat().st_size,
            "checkpoint_file_count": 1,
        }
        atomic_write_json(checkpoint_identity_destination, checkpoint_identity)
        artifacts.append(
            {
                "path": checkpoint_identity_destination.relative_to(
                    output_dir
                ).as_posix(),
                "sha256": sha256_file(checkpoint_identity_destination),
            }
        )
        rows.append(
            {
                **candidate_row,
                "run_contract_path": contract_destination.relative_to(
                    output_dir
                ).as_posix(),
                "run_contract_sha256": sha256_file(contract_destination),
                "full_report_path": report_destination.relative_to(
                    output_dir
                ).as_posix(),
                "full_report_sha256": sha256_file(report_destination),
                "resolved_config_path": resolved_config_destination.relative_to(
                    output_dir
                ).as_posix(),
                "resolved_config_artifact_sha256": sha256_file(
                    resolved_config_destination
                ),
                "checkpoint_identity_path": (
                    checkpoint_identity_destination.relative_to(output_dir).as_posix()
                ),
                "checkpoint_identity_sha256": sha256_file(
                    checkpoint_identity_destination
                ),
                "registered_config_path": config_destination.relative_to(
                    output_dir
                ).as_posix(),
                "registered_config_sha256": sha256_file(config_destination),
                "recomputed_quality_sha256": sha256_json(
                    candidate_row["recomputed_quality"]
                ),
            }
        )

    if seen != set(candidate_model_ids):
        raise SchemaError(
            f"selection evidence does not cover the registered {selection_mode} candidates"
        )
    if len(git_commits) != 1:
        raise SchemaError("candidate runs used different Git commits")
    if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE:
        ranked = _rank_candidates(rows)
        decision = _decision(ranked, git_commit=next(iter(git_commits)))
        comparison_json_source = _input_path(
            spec_path,
            selection_spec.get("comparison_json"),
            field="selection_evidence.comparison_json",
        )
        comparison_csv_source = _input_path(
            spec_path,
            selection_spec.get("comparison_csv"),
            field="selection_evidence.comparison_csv",
        )
        _validate_supplied_comparison_json(read_json(comparison_json_source), decision)
        _validate_supplied_comparison_csv(comparison_csv_source, ranked)
        for source, name in (
            (comparison_json_source, "three_model_selection.json"),
            (comparison_csv_source, "three_model_validation_comparison.csv"),
        ):
            destination = output_dir / "comparison" / name
            _copy_exact(source, destination)
            artifacts.append(
                {
                    "path": destination.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(destination),
                }
            )
        selected = ranked[0]
    else:
        if "comparison_json" in selection_spec or "comparison_csv" in selection_spec:
            raise SchemaError(
                "pc01_provisional evidence must not include comparison JSON or CSV"
            )
        selected = rows[0]
        _validate_pc01_provisional_candidate(selected)
        ranked = [selected]
        decision = _provisional_decision(
            selected, git_commit=next(iter(git_commits))
        )
    _bind_selected_model_manifest(selected_model_manifest, selected)
    compare_source = repository_root / "scripts" / "compare_full_models.py"
    selector_source = repository_root / "src" / "web_agent" / "train" / "selection.py"
    required_sources = (
        (compare_source, selector_source)
        if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE
        else (selector_source,)
    )
    for source in required_sources:
        if not source.is_file() or source.is_symlink():
            raise SchemaError(f"registered selection implementation is missing: {source}")

    manifest = {
        "schema_version": SELECTION_EVIDENCE_SCHEMA_VERSION,
        "selection_mode": selection_mode,
        "selection_status": policy["selection_status"],
        "evidence_label": policy["evidence_label"],
        "provisional": policy["provisional"],
        "final_campaign_eligible": policy["final_campaign_eligible"],
        "comparison_performed": policy["comparison_performed"],
        "selection_scope": "validation_only",
        "checkpoint_selection": "validation_only",
        "epoch_selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
        "comparison_rule": (
            COMPARISON_RULE
            if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE
            else None
        ),
        "candidate_seed": EXPECTED_CANDIDATE_SEED,
        "candidate_model_ids": list(candidate_model_ids),
        "git_commit": next(iter(git_commits)),
        "selected_model_id": selected["model_id"],
        "selected_epoch": selected["selected_epoch"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "recomputed_decision": decision,
        "candidates": sorted(rows, key=lambda row: str(row["model_id"])),
        "selection_implementation": {
            "relative_path": "src/web_agent/train/selection.py",
            "sha256": sha256_file(selector_source),
        },
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
    }
    if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE:
        manifest["comparison_script"] = {
            "relative_path": "scripts/compare_full_models.py",
            "sha256": sha256_file(compare_source),
        }
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    validate_selection_evidence(
        manifest_path,
        repository_root=repository_root,
        selected_model_manifest=selected_model_manifest,
        expected_selection_mode=selection_mode,
    )
    return manifest_path


def validate_selection_evidence(
    manifest_path: str | Path,
    *,
    repository_root: Path | None = None,
    selected_model_manifest: Mapping[str, Any] | None = None,
    selected_model_manifests: Mapping[int, Mapping[str, Any]] | None = None,
    expected_model_seeds: Sequence[int] | None = None,
    expected_selection_mode: str | None = None,
) -> dict[str, Any]:
    """Replay a staged selection package from its frozen source artifacts."""

    path = Path(manifest_path).resolve()
    root = path.parent
    manifest = read_json(path)
    if manifest.get("schema_version") != SELECTION_EVIDENCE_SCHEMA_VERSION:
        raise SchemaError("validation-selection evidence schema is not registered")
    selection_mode = str(manifest.get("selection_mode") or "")
    policy = _selection_policy(selection_mode)
    if (
        expected_selection_mode is not None
        and selection_mode != expected_selection_mode
    ):
        raise SchemaError(
            "selection evidence mode differs from the frozen protocol selection mode"
        )
    candidate_model_ids = tuple(policy["candidate_model_ids"])
    expected_constants = {
        "selection_mode": selection_mode,
        "selection_status": policy["selection_status"],
        "evidence_label": policy["evidence_label"],
        "provisional": policy["provisional"],
        "final_campaign_eligible": policy["final_campaign_eligible"],
        "comparison_performed": policy["comparison_performed"],
        "selection_scope": "validation_only",
        "checkpoint_selection": "validation_only",
        "epoch_selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
        "comparison_rule": (
            COMPARISON_RULE
            if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE
            else None
        ),
        "candidate_seed": EXPECTED_CANDIDATE_SEED,
        "candidate_model_ids": list(candidate_model_ids),
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    for key, expected in expected_constants.items():
        if manifest.get(key) != expected:
            raise SchemaError(f"validation-selection evidence changed {key}")
    descriptors = manifest.get("artifacts")
    expected_artifact_count = len(candidate_model_ids) * 5 + (
        2 if policy["comparison_performed"] else 0
    )
    if (
        not isinstance(descriptors, list)
        or len(descriptors) != expected_artifact_count
    ):
        raise SchemaError("validation-selection evidence artifact closure is incomplete")
    descriptor_by_path: dict[str, str] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise SchemaError("selection artifact descriptor is malformed")
        relative = _safe_relative(str(descriptor.get("path") or ""))
        if relative in descriptor_by_path:
            raise SchemaError("selection evidence repeats an artifact path")
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file() or source.is_symlink():
            raise SchemaError("selection evidence artifact is absent or unsafe")
        actual = sha256_file(source)
        if descriptor.get("sha256") != actual:
            raise SchemaError(f"selection evidence artifact hash mismatch: {relative}")
        descriptor_by_path[relative] = actual

    candidate_rows = manifest.get("candidates")
    if (
        not isinstance(candidate_rows, list)
        or len(candidate_rows) != len(candidate_model_ids)
    ):
        raise SchemaError(
            f"{selection_mode} evidence requires exactly "
            f"{len(candidate_model_ids)} candidate record(s)"
        )
    recomputed: list[dict[str, Any]] = []
    commits: set[str] = set()
    for stored in candidate_rows:
        if not isinstance(stored, Mapping):
            raise SchemaError("selection candidate record is malformed")
        model_id = str(stored.get("model_id") or "")
        contract_relative = _safe_relative(str(stored.get("run_contract_path") or ""))
        report_relative = _safe_relative(str(stored.get("full_report_path") or ""))
        resolved_config_relative = _safe_relative(
            str(stored.get("resolved_config_path") or "")
        )
        checkpoint_identity_relative = _safe_relative(
            str(stored.get("checkpoint_identity_path") or "")
        )
        config_relative = _safe_relative(
            str(stored.get("registered_config_path") or "")
        )
        for relative, hash_field in (
            (contract_relative, "run_contract_sha256"),
            (report_relative, "full_report_sha256"),
            (
                resolved_config_relative,
                "resolved_config_artifact_sha256",
            ),
            (
                checkpoint_identity_relative,
                "checkpoint_identity_sha256",
            ),
            (config_relative, "registered_config_sha256"),
        ):
            if descriptor_by_path.get(relative) != stored.get(hash_field):
                raise SchemaError("selection candidate artifact binding mismatch")
        contract = read_json(root / contract_relative)
        report = read_json(root / report_relative)
        resolved_config = read_json(root / resolved_config_relative)
        checkpoint_identity = read_json(root / checkpoint_identity_relative)
        _validate_staged_run_contract(contract, model_id=model_id)
        _validate_staged_registered_config(
            root / config_relative,
            contract=contract,
            model_id=model_id,
        )
        _validate_resolved_config_identity(
            resolved_config,
            contract=contract,
            model_id=model_id,
        )
        row = _recompute_candidate(report, model_id=model_id, contract=contract)
        if sha256_json(resolved_config) != row["resolved_config_sha256"]:
            raise SchemaError(
                f"{model_id} staged resolved config differs from the report hash"
            )
        _validate_checkpoint_identity(
            checkpoint_identity,
            model_id=model_id,
            selected_epoch=int(row["selected_epoch"]),
            checkpoint_sha256=str(row["checkpoint_sha256"]),
        )
        commits.add(str(contract["git_commit"]))
        expected_projection = {
            key: value
            for key, value in stored.items()
            if key
            not in {
                "run_contract_path",
                "run_contract_sha256",
                "full_report_path",
                "full_report_sha256",
                "resolved_config_path",
                "resolved_config_artifact_sha256",
                "checkpoint_identity_path",
                "checkpoint_identity_sha256",
                "registered_config_path",
                "registered_config_sha256",
                "recomputed_quality_sha256",
            }
        }
        if expected_projection != row:
            raise SchemaError(f"stored selection result differs from replay for {model_id}")
        if stored.get("recomputed_quality_sha256") != sha256_json(
            row["recomputed_quality"]
        ):
            raise SchemaError("stored selection quality hash differs from replay")
        recomputed.append({**dict(stored), **row})
    if {row["model_id"] for row in recomputed} != set(candidate_model_ids):
        raise SchemaError("selection candidate set differs from registration")
    if len(commits) != 1 or manifest.get("git_commit") != next(iter(commits)):
        raise SchemaError("selection evidence Git commit binding is inconsistent")
    if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE:
        ranked = _rank_candidates(recomputed)
        decision = _decision(ranked, git_commit=next(iter(commits)))
    else:
        selected = recomputed[0]
        _validate_pc01_provisional_candidate(selected)
        ranked = [selected]
        decision = _provisional_decision(
            selected, git_commit=next(iter(commits))
        )
    if manifest.get("recomputed_decision") != decision:
        raise SchemaError("stored comparison decision differs from replay")
    if (
        manifest.get("selected_model_id") != ranked[0]["model_id"]
        or manifest.get("selected_epoch") != ranked[0]["selected_epoch"]
        or manifest.get("selected_checkpoint_sha256")
        != ranked[0]["checkpoint_sha256"]
    ):
        raise SchemaError("selection manifest winner differs from replay")
    comparison_root = root / "comparison"
    if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE:
        comparison_json = comparison_root / "three_model_selection.json"
        comparison_csv = comparison_root / "three_model_validation_comparison.csv"
        _validate_supplied_comparison_json(read_json(comparison_json), decision)
        _validate_supplied_comparison_csv(comparison_csv, ranked)
    elif comparison_root.exists():
        raise SchemaError(
            "pc01_provisional evidence must not contain a comparison directory"
        )
    if selected_model_manifest is not None and selected_model_manifests is not None:
        raise SchemaError(
            "selection validation accepts either one legacy model manifest or "
            "the exact campaign model-manifest mapping, not both"
        )
    if selected_model_manifest is not None:
        if expected_model_seeds is not None:
            raise SchemaError(
                "expected_model_seeds requires the campaign model-manifest mapping"
            )
        _bind_selected_model_manifest(selected_model_manifest, ranked[0])
    if selected_model_manifests is not None:
        _bind_selected_model_manifests(
            selected_model_manifests,
            ranked[0],
            expected_model_seeds=expected_model_seeds,
        )
    elif expected_model_seeds is not None:
        raise SchemaError(
            "expected_model_seeds requires the campaign model-manifest mapping"
        )
    if repository_root is not None:
        source_fields = ["selection_implementation"]
        if selection_mode == THREE_CANDIDATE_FINAL_SELECTION_MODE:
            source_fields.insert(0, "comparison_script")
        elif "comparison_script" in manifest:
            raise SchemaError(
                "pc01_provisional evidence must not claim a comparison script"
            )
        for field in source_fields:
            identity = manifest.get(field)
            if not isinstance(identity, Mapping):
                raise SchemaError(f"selection evidence lacks {field}")
            relative = _safe_relative(str(identity.get("relative_path") or ""))
            source = (repository_root.resolve() / relative).resolve()
            if (
                repository_root.resolve() not in source.parents
                or not source.is_file()
                or sha256_file(source) != identity.get("sha256")
            ):
                raise SchemaError(f"registered {field} source differs from evidence")
    return manifest


def _validate_run_contract(
    contract: Mapping[str, Any], *, model_id: str, repository_root: Path
) -> str:
    _validate_staged_run_contract(contract, model_id=model_id)
    candidate = contract["candidate"]
    config_relative = _safe_relative(str(candidate.get("config") or ""))
    expected = f"configs/backbones/{model_id}.yaml"
    if config_relative != expected:
        raise SchemaError(f"{model_id} used a nonregistered backbone configuration")
    config_path = (repository_root.resolve() / config_relative).resolve()
    if repository_root.resolve() not in config_path.parents or not config_path.is_file():
        raise SchemaError(f"registered candidate config is missing: {config_relative}")
    try:
        registered_config = load_config(config_path)
    except (OSError, TypeError, ValueError) as exc:
        raise SchemaError(f"cannot resolve registered config for {model_id}") from exc
    if (
        contract.get("config_name") != registered_config.get("name")
        or contract.get("model_revision")
        != registered_config.get("backbone", {}).get("revision")
    ):
        raise SchemaError(
            f"{model_id} run contract differs from the registered config identity"
        )
    return config_relative


def _validate_staged_run_contract(
    contract: Mapping[str, Any], *, model_id: str
) -> None:
    candidate = contract.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("model_id") != model_id:
        raise SchemaError(f"run contract identifies another candidate: {model_id}")
    expected_config = f"configs/backbones/{model_id}.yaml"
    if candidate.get("config") != expected_config:
        raise SchemaError(f"{model_id} run contract changed its registered config path")
    exact = {
        "seed": EXPECTED_CANDIDATE_SEED,
        "maximum_full_epochs": EXPECTED_MAXIMUM_EPOCHS,
        "effective_batch_size": 32,
        "checkpoint_selection": ALL_GATES_THEN_OUTCOME_RULE,
        "checkpoint_selection_source": "original_gold_validation_only",
        "supplement_validation_selects_checkpoint": False,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "original_validation_rows": EXPECTED_VALIDATION_ROWS,
        "supplement_validation_rows": EXPECTED_SUPPLEMENT_ROWS,
        "test_rows_read": 0,
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise SchemaError(f"{model_id} run contract changed {key}")
    commit = str(contract.get("git_commit") or "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SchemaError(f"{model_id} run contract lacks a full Git commit")


def _validate_staged_registered_config(
    config_path: Path,
    *,
    contract: Mapping[str, Any],
    model_id: str,
) -> None:
    """Re-resolve the copied config instead of trusting stored descriptors."""

    if not config_path.is_file() or config_path.is_symlink():
        raise SchemaError(f"staged registered config is missing or unsafe: {model_id}")
    try:
        registered_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SchemaError(f"cannot resolve staged registered config for {model_id}") from exc
    if not isinstance(registered_config, Mapping):
        raise SchemaError(f"staged registered config is malformed: {model_id}")
    if (
        contract.get("config_name") != registered_config.get("name")
        or contract.get("model_revision")
        != registered_config.get("backbone", {}).get("revision")
    ):
        raise SchemaError(
            f"{model_id} staged config differs from the run-contract identity"
        )


def _validate_resolved_config_identity(
    resolved_config: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    model_id: str,
) -> None:
    """Bind a full-run resolved configuration to its registered candidate."""

    expected_name = f"{contract.get('config_name')}_FULL_SEED{EXPECTED_CANDIDATE_SEED}"
    backbone = resolved_config.get("backbone")
    train = resolved_config.get("train")
    optim = resolved_config.get("optim")
    if (
        resolved_config.get("name") != expected_name
        or resolved_config.get("seeds") != [EXPECTED_CANDIDATE_SEED]
        or not isinstance(backbone, Mapping)
        or backbone.get("revision") != contract.get("model_revision")
        or not isinstance(train, Mapping)
        or train.get("epochs") != EXPECTED_MAXIMUM_EPOCHS
        or train.get("quality_selection_rule") != ALL_GATES_THEN_OUTCOME_RULE
        or not isinstance(optim, Mapping)
    ):
        raise SchemaError(
            f"{model_id} resolved configuration differs from its run contract"
        )
    try:
        effective_batch = int(optim["batch_size"]) * int(optim["grad_accum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"{model_id} resolved configuration lacks batch controls"
        ) from exc
    if effective_batch != 32:
        raise SchemaError(
            f"{model_id} resolved configuration changed effective batch size"
        )


def _validate_checkpoint_identity(
    identity: Mapping[str, Any],
    *,
    model_id: str,
    selected_epoch: int,
    checkpoint_sha256: str,
) -> None:
    expected = {
        "schema_version": "table2-selection-checkpoint-identity-v1",
        "model_id": model_id,
        "selected_epoch": selected_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_file_count": 1,
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise SchemaError(
                f"{model_id} checkpoint identity differs at {field}"
            )
    if (
        type(identity.get("checkpoint_bytes")) is not int
        or int(identity["checkpoint_bytes"]) <= 0
    ):
        raise SchemaError(f"{model_id} checkpoint identity lacks a byte count")


def _recompute_candidate(
    report: Mapping[str, Any], *, model_id: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    exact = {
        "seed": EXPECTED_CANDIDATE_SEED,
        "requested_epochs": EXPECTED_MAXIMUM_EPOCHS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "val_rows": EXPECTED_VALIDATION_ROWS,
        "test_rows_read": 0,
        "selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
    }
    for key, expected in exact.items():
        if report.get(key) != expected:
            raise SchemaError(f"{model_id} full report changed {key}")
    source_validation = report.get("source_validation")
    if not isinstance(source_validation, Mapping):
        raise SchemaError(f"{model_id} report lacks source-separated validation")
    supplement = source_validation.get("supplement_retry_abort")
    primary = source_validation.get("primary_original_gold")
    if (
        not isinstance(supplement, Mapping)
        or supplement.get("rows") != EXPECTED_SUPPLEMENT_ROWS
        or not isinstance(primary, Mapping)
        or primary.get("rows") != EXPECTED_VALIDATION_ROWS
        or primary.get("checkpoint_selection_source") is not True
    ):
        raise SchemaError(f"{model_id} report changed validation-source boundaries")
    control = report.get("experiment_control")
    if not isinstance(control, Mapping):
        raise SchemaError(f"{model_id} report lacks experiment control evidence")
    if (
        control.get("checkpoint_selection_source")
        != "original_gold_validation_only"
        or control.get("supplement_validation_selects_checkpoint") is not False
        or control.get("locked_test_read") is not False
    ):
        raise SchemaError(f"{model_id} report violated validation-only selection")
    resolved_config_sha256 = str(control.get("config_sha256") or "")
    if not _is_sha256(resolved_config_sha256):
        raise SchemaError(f"{model_id} report lacks its resolved-config hash")
    quality = controlled_quality_gates(report, ALL_GATES_THEN_OUTCOME_RULE)
    checkpoint = require_selected_checkpoint(report, quality)
    if quality.get("status") != "PASS" or quality.get("selected_epoch_is_eligible") is not True:
        raise SchemaError(f"{model_id} has no all-gate-eligible selected epoch")
    selected_epoch = int(quality["selected_epoch"])
    if (
        report.get("status") != "PASS"
        or int(report.get("selected_epoch", -1)) != selected_epoch
        or str(report.get("best_checkpoint") or "") != checkpoint
    ):
        raise SchemaError(f"{model_id} reported selection differs from replay")
    reported_quality = report.get("quality_gates")
    if reported_quality != quality:
        raise SchemaError(f"{model_id} stored quality-gate report differs from replay")
    checkpoint_sha = str(report.get("selected_checkpoint_sha256") or "")
    if not _is_sha256(checkpoint_sha):
        raise SchemaError(f"{model_id} selected checkpoint hash is invalid")
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise SchemaError(f"{model_id} full report lacks epoch history")
    selected_rows: list[Mapping[str, Any]] = []
    for index, value in enumerate(history):
        if not isinstance(value, Mapping):
            raise SchemaError(f"{model_id} history[{index}] is not an object")
        try:
            epoch = int(value["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"{model_id} history[{index}] lacks a valid epoch") from exc
        if epoch == selected_epoch:
            selected_rows.append(value)
    if len(selected_rows) != 1:
        raise SchemaError(
            f"{model_id} report must contain exactly one selected-epoch row"
        )
    metrics = selected_rows[0]
    required_metrics = (
        "outcome_mcc",
        "failure_macro_f1",
        "failtype_macro_f1",
        "action_macro_f1",
        "needs_recovery_macro_f1",
        "strategy_attempted_macro_f1",
        "recovery_outcome_mcc",
        "memory_mcc",
        "bbox_mean_iou",
        "bbox_recall_iou50",
        "outcome_ece",
    )
    selected_metrics: dict[str, float] = {}
    for name in required_metrics:
        try:
            selected_metrics[name] = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError(f"{model_id} selected metrics lack {name}") from exc
        value = selected_metrics[name]
        if not math.isfinite(value):
            raise SchemaError(f"{model_id} selected metric {name} is not finite")
        lower = -1.0 if name.endswith("_mcc") else 0.0
        if not lower <= value <= 1.0:
            raise SchemaError(
                f"{model_id} selected metric {name} is outside [{lower}, 1]"
            )
    return {
        "model_id": model_id,
        "seed": EXPECTED_CANDIDATE_SEED,
        "selected_epoch": selected_epoch,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config_sha256": resolved_config_sha256,
        **selected_metrics,
        "recomputed_quality": quality,
        "run_git_commit": str(contract["git_commit"]),
    }


def _rank_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["outcome_mcc"]),
            -float(row["recovery_outcome_mcc"]),
            -float(row["action_macro_f1"]),
            float(row["outcome_ece"]),
            str(row["model_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["validation_rank"] = rank
        row["selected_candidate"] = rank == 1
    return ranked


def _comparison_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_id",
        "seed",
        "selected_epoch",
        "checkpoint",
        "checkpoint_sha256",
        "outcome_mcc",
        "failure_macro_f1",
        "failtype_macro_f1",
        "action_macro_f1",
        "needs_recovery_macro_f1",
        "strategy_attempted_macro_f1",
        "recovery_outcome_mcc",
        "memory_mcc",
        "bbox_mean_iou",
        "bbox_recall_iou50",
        "outcome_ece",
        "validation_rank",
        "selected_candidate",
    )
    return {key: row[key] for key in keys}


def _validate_pc01_provisional_candidate(candidate: Mapping[str, Any]) -> None:
    """Bind the pilot to the registered PC-01 epoch-6 artifact identity."""

    expected = {
        "model_id": PC01_MODEL_ID,
        "seed": EXPECTED_CANDIDATE_SEED,
        "selected_epoch": PC01_EXPECTED_SELECTED_EPOCH,
        "checkpoint_sha256": PC01_EXPECTED_CHECKPOINT_SHA256,
        "resolved_config_sha256": PC01_EXPECTED_CONFIG_SHA256,
    }
    for field, value in expected.items():
        if candidate.get(field) != value:
            raise SchemaError(
                f"pc01_provisional evidence differs from registered PC-01 {field}"
            )


def _provisional_decision(
    candidate: Mapping[str, Any], *, git_commit: str
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "selection_mode": PC01_PROVISIONAL_SELECTION_MODE,
        "selection_scope": "validation_only",
        "selection_rule": ALL_GATES_THEN_OUTCOME_RULE,
        "git_commit": git_commit,
        "selected_model_id": candidate["model_id"],
        "selected_epoch": candidate["selected_epoch"],
        "selected_checkpoint_sha256": candidate["checkpoint_sha256"],
        "resolved_config_sha256": candidate["resolved_config_sha256"],
        "test_rows_read": 0,
        "locked_test_rows_read": 0,
        "provisional": True,
        "evidence_label": "PILOT_ONLY",
        "final_campaign_eligible": False,
        "next_action": (
            "Use PC-01 epoch 6, seed 42 for PILOT_ONLY engineering evidence. "
            "After PC-02 and PC-03 finish, replay the registered validation-only "
            "three-candidate comparison at seed 42 before final promotion."
        ),
    }


def _decision(rows: Sequence[Mapping[str, Any]], *, git_commit: str) -> dict[str, Any]:
    return {
        "status": "PASS",
        "selection_scope": "validation_only",
        "selection_rule": COMPARISON_RULE,
        "git_commit": git_commit,
        "selected_model_id": rows[0]["model_id"],
        "models": [_comparison_projection(row) for row in rows],
        "test_rows_read": 0,
        "next_action": (
            "Promote the validation-selected seed-42 checkpoint to the frozen final "
            "Table 2 campaign; do not add model seeds 43 or 44."
        ),
    }


def _validate_supplied_comparison_json(
    supplied: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if supplied != expected:
        raise SchemaError(
            "supplied three-model selection JSON differs from an independent replay"
        )


def _validate_supplied_comparison_csv(
    path: Path, expected_rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = [_comparison_projection(row) for row in expected_rows]
    if len(rows) != len(expected):
        raise SchemaError("three-model comparison CSV has the wrong row count")
    for actual, wanted in zip(rows, expected, strict=True):
        if set(actual) != set(wanted):
            raise SchemaError("three-model comparison CSV columns differ from registration")
        for key, value in wanted.items():
            text = str(actual[key])
            if isinstance(value, bool):
                valid = text == str(value)
            elif isinstance(value, int):
                valid = text == str(value)
            elif isinstance(value, float):
                try:
                    valid = float(text) == value
                except ValueError:
                    valid = False
            else:
                valid = text == str(value)
            if not valid:
                raise SchemaError(
                    f"three-model comparison CSV differs at {wanted['model_id']}.{key}"
                )


def _bind_selected_model_manifest(
    model_manifest: Mapping[str, Any], winner: Mapping[str, Any]
) -> None:
    if int(model_manifest.get("model_seed", -1)) != EXPECTED_CANDIDATE_SEED:
        raise SchemaError("Table 2 selected model manifest must use seed 42")
    if model_manifest.get("selected_model_id") != winner["model_id"]:
        raise SchemaError("Table 2 model ID differs from validation-selected winner")
    if int(model_manifest.get("selected_epoch", -1)) != int(winner["selected_epoch"]):
        raise SchemaError("Table 2 checkpoint epoch differs from validation selection")
    if model_manifest.get("selected_checkpoint_sha256") != winner["checkpoint_sha256"]:
        raise SchemaError("Table 2 checkpoint bytes differ from validation selection")
    if (
        model_manifest.get("resolved_config_record_sha256")
        != winner["resolved_config_sha256"]
    ):
        raise SchemaError(
            "Table 2 resolved configuration differs from the validation-selected run"
        )
    if (
        model_manifest.get("selection_scope") != "validation_only"
        or model_manifest.get("checkpoint_selection") != "validation_only"
        or int(model_manifest.get("test_rows_read", -1)) != 0
        or int(model_manifest.get("locked_test_rows_read", -1)) != 0
    ):
        raise SchemaError("Table 2 model manifest violates validation-only selection")


def _bind_selected_model_manifests(
    model_manifests: Mapping[int, Mapping[str, Any]],
    winner: Mapping[str, Any],
    *,
    expected_model_seeds: Sequence[int] | None,
) -> None:
    """Bind the single registered seed-42 Table 2 model manifest."""

    if not isinstance(model_manifests, Mapping) or not model_manifests:
        raise SchemaError("selection evidence requires a nonempty model manifest mapping")
    normalized: dict[int, Mapping[str, Any]] = {}
    for raw_seed, payload in model_manifests.items():
        if type(raw_seed) is not int or raw_seed < 0:
            raise SchemaError("campaign model-manifest keys must be nonnegative integers")
        if not isinstance(payload, Mapping):
            raise SchemaError(f"campaign model manifest for seed {raw_seed} is malformed")
        if type(payload.get("model_seed")) is not int or int(payload["model_seed"]) != raw_seed:
            raise SchemaError(
                f"campaign model-manifest key/payload seed mismatch for seed {raw_seed}"
            )
        normalized[raw_seed] = payload

    if expected_model_seeds is not None:
        expected: list[int] = []
        for raw_seed in expected_model_seeds:
            if type(raw_seed) is not int or raw_seed < 0:
                raise SchemaError("expected campaign model seeds must be nonnegative integers")
            expected.append(raw_seed)
        if not expected or len(expected) != len(set(expected)):
            raise SchemaError("expected campaign model seeds must be nonempty and unique")
        if set(normalized) != set(expected):
            raise SchemaError(
                "selection evidence model manifests must cover the exact campaign "
                f"matched-seed set; expected={sorted(expected)}, "
                f"actual={sorted(normalized)}"
            )

    if set(normalized) != {EXPECTED_CANDIDATE_SEED}:
        raise SchemaError(
            "Table 2 selection evidence authorizes exactly model seed 42; "
            "seeds 43 and 44 are not registered"
        )

    for seed, model_manifest in sorted(normalized.items()):
        if model_manifest.get("selected_model_id") != winner["model_id"]:
            raise SchemaError(
                f"Table 2 seed {seed} model ID differs from validation-selected winner"
            )
        if (
            model_manifest.get("selection_scope") != "validation_only"
            or model_manifest.get("checkpoint_selection") != "validation_only"
            or type(model_manifest.get("validation_rows_read")) is not int
            or int(model_manifest["validation_rows_read"]) <= 0
            or type(model_manifest.get("test_rows_read")) is not int
            or int(model_manifest["test_rows_read"]) != 0
            or type(model_manifest.get("locked_test_rows_read")) is not int
            or int(model_manifest["locked_test_rows_read"]) != 0
        ):
            raise SchemaError(
                f"Table 2 seed {seed} model manifest violates validation-only selection"
            )
        if type(model_manifest.get("selected_epoch")) is not int or int(
            model_manifest["selected_epoch"]
        ) < 0:
            raise SchemaError(
                f"Table 2 seed {seed} model manifest lacks a selected epoch"
            )
        for field in (
            "selected_checkpoint_sha256",
            "resolved_config_sha256",
            "resolved_config_record_sha256",
        ):
            if not _is_sha256(model_manifest.get(field)):
                raise SchemaError(
                    f"Table 2 seed {seed} model manifest lacks {field}"
                )

    # Seed 42 is the sole model seed and must bind the exact selected epoch,
    # checkpoint bytes, and resolved configuration.
    _bind_selected_model_manifest(normalized[EXPECTED_CANDIDATE_SEED], winner)


def _input_path(spec_path: Path, value: object, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise SchemaError(f"handoff input requires {field}")
    candidate = Path(text)
    path = (candidate if candidate.is_absolute() else spec_path.parent / candidate).resolve()
    if not path.is_file() or path.is_symlink():
        raise SchemaError(f"selection evidence input is missing or unsafe: {field}")
    return path


def _copy_exact(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SchemaError(f"selection evidence source is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise SchemaError("selection evidence copy changed bytes")


def _safe_relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise SchemaError("selection evidence contains an unsafe relative path")
    return path.as_posix()


def _is_sha256(value: object) -> bool:
    if (
        not isinstance(value, str)
        or value != value.lower()
        or len(value) != 64
    ):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
