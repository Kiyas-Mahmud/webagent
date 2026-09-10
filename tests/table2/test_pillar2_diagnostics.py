from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import threading

import pytest

from web_agent.eval.table2.common import (
    SchemaError,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from web_agent.eval.table2.pillar2_diagnostics import (
    CANONICAL_INPUT_PROVENANCE_STATUS,
    DiagnosticBackendIdentity,
    DiagnosticCondition,
    DiagnosticExample,
    DiagnosticTargets,
    DiagnosticTextState,
    ENGINEERING_EVIDENCE_ROLE,
    ImageArtifact,
    INPUT_PROVENANCE_BLOCKED_PROMOTION,
    NEUTRAL_TEXT_ID,
    PILLAR2_EVIDENCE_ROLE,
    PAPER_TABLE_STATUS,
    Pillar2DiagnosticError,
    Pillar2DiagnosticInput,
    PostDiagnosticPrediction,
    PostDiagnosticView,
    PreDiagnosticPrediction,
    PreDiagnosticView,
    _canonical_pc01_error,
    run_pillar2_diagnostics,
    validate_pillar2_diagnostic_report,
    validate_pillar2_diagnostic_package,
    write_pillar2_diagnostic_package,
)
from web_agent.eval.table2.pc01_pillar2_diagnostics import (
    PC01_EXPECTED_CHECKPOINT_SHA256,
    PC01_EXPECTED_CONFIG_SHA256,
    PC01_EXPECTED_PROCESSOR_CONTRACT_SHA256,
    PC01_MODEL_ID,
    PC01_MODEL_REVISION,
    PC01_PILLAR2_PREDICTOR_ID,
    PC01_PILLAR2_PREDICTOR_MODULE,
    PC01_PILLAR2_PREDICTOR_QUALNAME,
    PC01_PILLAR2_PREDICTOR_VERSION,
    PC01Pillar2DiagnosticPredictor,
    canonical_pc01_backend_error,
    pc01_pillar2_predictor_source_sha256,
)
from scripts.run_table2_pillar2_diagnostics import (
    _verify_clean_repository_before_import,
    _verify_committed_source_file,
)


SHA = "a" * 64
COMMIT = "b" * 40
NEUTRAL_TEXT = "[REGISTERED_NEUTRAL_TEXT_STATE_V1]"


def create_test_predictor():  # pragma: no cover - identity fixture for CLI contract
    raise AssertionError("test-only factory must not be called")


class DeterministicP2Predictor:
    def __init__(
        self,
        identity: DiagnosticBackendIdentity,
        *,
        contaminate_pre_after_post: bool = False,
    ) -> None:
        self._identity = identity
        self.contaminate_pre_after_post = contaminate_pre_after_post
        self.post_called = False
        self.pre_views: list[PreDiagnosticView] = []
        self.post_views: list[PostDiagnosticView] = []

    @property
    def diagnostic_identity(self) -> DiagnosticBackendIdentity:
        return self._identity

    def predict_pre(
        self, view: PreDiagnosticView, *, seed: int
    ) -> PreDiagnosticPrediction:
        self.pre_views.append(view)
        salt = int(view.image.sha256[:2], 16) + (seed % 3)
        if view.text_state.domain == "neutral.invalid":
            salt += 7
        if self.contaminate_pre_after_post and self.post_called:
            salt += 31
        chosen = salt % 6
        probabilities = {label: 0.04 for label in (
            "CLICK", "TYPE", "SELECT", "SCROLL", "NAVIGATE", "PRESS_KEY"
        )}
        probabilities[list(probabilities)[chosen]] = 0.8
        return PreDiagnosticPrediction(
            action_probabilities=probabilities,
            bbox=(0.1 + chosen * 0.01, 0.2, 0.2, 0.2),
            confidence_before=0.75,
        )

    def predict_post(
        self, view: PostDiagnosticView, *, seed: int
    ) -> PostDiagnosticPrediction:
        self.post_called = True
        self.post_views.append(view)
        action_index = {
            None: 0,
            "CLICK": 1,
            "TYPE": 2,
            "SELECT": 3,
            "SCROLL": 4,
            "NAVIGATE": 5,
            "PRESS_KEY": 6,
        }[view.executed_action]
        salt = (int(view.post_image.sha256[:2], 16) + action_index + seed) % 10
        if view.text_state.domain == "neutral.invalid":
            salt = (salt + 3) % 10
        failure_probability = salt / 10
        failure_labels = ("NONE", "PERCEPTION_ERROR", "ACTION_MISMATCH", "LOOP_DETECTED")
        failure_index = salt % len(failure_labels)
        failure_distribution = {label: 0.1 for label in failure_labels}
        failure_distribution[failure_labels[failure_index]] = 0.7
        recovery_labels = (
            "NONE", "RETRY", "REPLAN", "BACKTRACK", "ALTERNATIVE_TARGET", "ABORT"
        )
        recovery_index = salt % len(recovery_labels)
        recovery_distribution = {label: 0.04 for label in recovery_labels}
        recovery_distribution[recovery_labels[recovery_index]] = 0.8
        return PostDiagnosticPrediction(
            failure_probability=failure_probability,
            failure_type_probabilities=failure_distribution,
            needs_recovery_probability=(10 - salt) / 10,
            recovery_probabilities=recovery_distribution,
        )


def _write(root: Path, name: str, payload: bytes) -> ImageArtifact:
    path = root / name
    path.write_bytes(payload)
    return ImageArtifact(
        artifact_id=name,
        relative_path=name,
        sha256=sha256_bytes(payload),
        width=16,
        height=12,
    )


def _identity() -> DiagnosticBackendIdentity:
    source_hash = sha256_file(Path(__file__))
    return DiagnosticBackendIdentity(
        predictor_id="deterministic-p2-fixture",
        predictor_version="v1",
        predictor_module=__name__,
        predictor_qualname="DeterministicP2Predictor",
        factory_entrypoint=f"{__name__}:create_test_predictor",
        model_id="deterministic-fixture-model",
        model_revision="fixture-v1",
        checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        resolved_config_record_sha256="6" * 64,
        processor_contract_sha256="3" * 64,
        factory_source_sha256=source_hash,
        predictor_source_sha256=source_hash,
        repository_commit=COMMIT,
        model_seed=42,
    )


def _target(action: str, failure_type: str, recovery: str) -> DiagnosticTargets:
    return DiagnosticTargets(
        action_type=action,
        action_bbox=(0.1, 0.2, 0.2, 0.2),
        failure=failure_type != "NONE",
        failure_type=failure_type,
        needs_recovery=recovery != "NONE",
        recovery_strategy=recovery,
    )


def _diagnostic(tmp_path: Path) -> tuple[Pillar2DiagnosticInput, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    neutral = _write(image_root, "neutral.bin", b"neutral-image")
    rows = (
        DiagnosticExample(
            example_id="example-a",
            task_id="task-a",
            pre_image=_write(image_root, "a-pre.bin", b"a-pre-image"),
            post_image=_write(image_root, "a-post.bin", b"a-post-image"),
            text_state=DiagnosticTextState(
                task_text="click the visible account link",
                domain="gitlab.local",
                current_url="http://gitlab.local/a",
                title="Account A",
                page_state={"visible_text": "Account A"},
            ),
            executed_action="CLICK",
            targets=_target("CLICK", "ACTION_MISMATCH", "REPLAN"),
            source_record_sha256="4" * 64,
        ),
        DiagnosticExample(
            example_id="example-b",
            task_id="task-b",
            pre_image=_write(image_root, "b-pre.bin", b"b-pre-image"),
            post_image=_write(image_root, "b-post.bin", b"b-post-image"),
            text_state=DiagnosticTextState(
                task_text="type into the visible search field",
                domain="gitlab.local",
                current_url="http://gitlab.local/b",
                title="Account B",
                page_state={"visible_text": "Search"},
            ),
            executed_action="TYPE",
            targets=_target("TYPE", "NONE", "NONE"),
            source_record_sha256="5" * 64,
        ),
    )
    diagnostic = Pillar2DiagnosticInput(
        diagnostic_id="p2-fixture-v1",
        campaign_id="table2-pc01-pilot-v1",
        source_partition="public_development",
        registered_seed=42,
        bootstrap_samples=200,
        bootstrap_confidence=0.95,
        bootstrap_seed=20250831,
        neutral_text_id=NEUTRAL_TEXT_ID,
        neutral_text=NEUTRAL_TEXT,
        neutral_images=(neutral,),
        examples=rows,
        backend_identity=_identity(),
        actions_frozen_before_diagnostics=True,
        runtime_outputs_consumed=False,
        affects_primary_table2=False,
        locked_test_rows_read=0,
        evidence_role=ENGINEERING_EVIDENCE_ROLE,
    )
    return diagnostic, image_root


def _run_pillar2_fixture(diagnostic, *, image_root, predictor):
    return run_pillar2_diagnostics(
        diagnostic,
        image_root=image_root,
        predictor=predictor,
        allow_uncommitted_engineering=True,
    )


def test_runs_all_registered_controls_without_runtime_outputs(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    predictor = DeterministicP2Predictor(diagnostic.backend_identity)
    report = _run_pillar2_fixture(
        diagnostic, image_root=image_root, predictor=predictor
    )
    validate_pillar2_diagnostic_report(report)

    assert report["evidence_role"] == ENGINEERING_EVIDENCE_ROLE
    assert report["promotion_status"] == "UNPROMOTABLE"
    assert report["paper_table_status"] == PAPER_TABLE_STATUS
    assert report["claim_scope"] == {
        "modality": "INFERENCE_TIME_SENSITIVITY_NOT_TRAINING_ABLATION",
        "causal_controls": "INPUT_CONDITIONING_DIAGNOSTIC_NOT_CAUSAL_PROOF",
        "operational_recovery": "NOT_MEASURED",
        "table2_row_or_contrast": "NONE",
    }
    assert report["browser_actions_executed"] == 0
    assert report["recovery_triggers_emitted"] == 0
    assert report["memory_queries_executed"] == 0
    assert report["memory_writes_executed"] == 0
    assert len(report["records"]) == 2 * (3 + 6)
    assert report["temporal_replay"]["count"] == 2
    assert all(row["identical"] for row in report["temporal_replay"]["records"])

    # Each predictor-visible type structurally omits targets/oracle truth.
    assert len(predictor.pre_views) == 2 * 4  # full + two controls + replay
    assert len(predictor.post_views) == 2 * 6
    assert all(not hasattr(view, "targets") for view in predictor.pre_views)
    assert all(not hasattr(view, "post_image") for view in predictor.pre_views)
    assert all(not hasattr(view, "executed_action") for view in predictor.pre_views)
    assert all(not hasattr(view, "targets") for view in predictor.post_views)


def test_each_control_changes_only_its_registered_input_factor(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    predictor = DeterministicP2Predictor(diagnostic.backend_identity)
    report = _run_pillar2_fixture(
        diagnostic, image_root=image_root, predictor=predictor
    )
    records = {
        (row["example_id"], row["phase"], row["condition"]): row
        for row in report["records"]
    }
    for example in diagnostic.examples:
        removed = next(
            view
            for view in predictor.post_views
            if view.example_id == example.example_id and view.executed_action is None
        )
        assert removed.pre_image.sha256 == example.pre_image.sha256
        assert removed.post_image.sha256 == example.post_image.sha256
        assert removed.text_state == example.text_state

        swapped_record = records[
            (example.example_id, "post", DiagnosticCondition.POST_STATE_SWAPPED.value)
        ]
        swapped = next(
            view
            for view in predictor.post_views
            if view.example_id == example.example_id
            and view.post_image.sha256
            != example.post_image.sha256
            and view.pre_image.sha256 == example.pre_image.sha256
        )
        assert swapped.executed_action == example.executed_action
        assert swapped.text_state == example.text_state
        assert swapped_record["donor"]["example_id"] != example.example_id

        mismatch_record = records[
            (example.example_id, "post", DiagnosticCondition.ACTION_POST_MISMATCH.value)
        ]
        mismatch = next(
            view
            for view in predictor.post_views
            if view.example_id == example.example_id
            and view.post_image.sha256 == example.post_image.sha256
            and view.executed_action not in {example.executed_action, None}
        )
        assert mismatch.pre_image.sha256 == example.pre_image.sha256
        assert mismatch.text_state == example.text_state
        assert mismatch_record["donor"]["example_id"] != example.example_id


def test_pre_action_replay_fails_on_post_stage_contamination(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    predictor = DeterministicP2Predictor(
        diagnostic.backend_identity, contaminate_pre_after_post=True
    )
    with pytest.raises(Pillar2DiagnosticError, match="temporal replay changed"):
        _run_pillar2_fixture(
            diagnostic, image_root=image_root, predictor=predictor
        )


def test_offline_targets_cannot_change_any_inference_prediction(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    first = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )
    changed_examples = tuple(
        replace(
            example,
            targets=_target("PRESS_KEY", "LOOP_DETECTED", "ABORT"),
        )
        for example in diagnostic.examples
    )
    changed = replace(diagnostic, examples=changed_examples)
    second = _run_pillar2_fixture(
        changed,
        image_root=image_root,
        predictor=DeterministicP2Predictor(changed.backend_identity),
    )
    first_predictions = [
        (row["example_id"], row["phase"], row["condition"], row["prediction_sha256"])
        for row in first["records"]
    ]
    second_predictions = [
        (row["example_id"], row["phase"], row["condition"], row["prediction_sha256"])
        for row in second["records"]
    ]
    assert first_predictions == second_predictions
    assert first["input_manifest_sha256"] != second["input_manifest_sha256"]


def test_oracle_fields_and_locked_access_fail_before_inference(tmp_path: Path):
    diagnostic, _ = _diagnostic(tmp_path)
    with pytest.raises(Exception, match="oracle-blind"):
        DiagnosticTextState(
            task_text="task",
            domain="example.invalid",
            current_url="https://example.invalid/",
            title="title",
            page_state={"oracle_success": True},
        )
    with pytest.raises(Exception, match="zero locked-test"):
        replace(diagnostic, locked_test_rows_read=1)
    with pytest.raises(Exception, match="after actions are frozen"):
        replace(diagnostic, actions_frozen_before_diagnostics=False)
    with pytest.raises(Exception, match="may not feed outputs"):
        replace(diagnostic, runtime_outputs_consumed=True)


def test_image_and_predictor_source_hashes_are_revalidated(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    tampered = image_root / diagnostic.examples[0].pre_image.relative_path
    tampered.write_bytes(b"changed")
    with pytest.raises(Pillar2DiagnosticError, match="image hash mismatch"):
        _run_pillar2_fixture(
            diagnostic,
            image_root=image_root,
            predictor=DeterministicP2Predictor(diagnostic.backend_identity),
        )
    with pytest.raises(Pillar2DiagnosticError, match="source hash mismatch"):
        _run_pillar2_fixture(
            replace(
                diagnostic,
                backend_identity=replace(
                    diagnostic.backend_identity,
                    predictor_source_sha256="9" * 64,
                ),
            ),
            image_root=image_root,
            predictor=DeterministicP2Predictor(
                replace(
                    diagnostic.backend_identity,
                    predictor_source_sha256="9" * 64,
                )
            ),
        )


def test_package_is_hash_bound_and_cannot_be_relabelled(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    report = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )
    output = tmp_path / "package"
    package = write_pillar2_diagnostic_package(
        output, diagnostic=diagnostic, report=report
    )
    assert package["report_sha256"] == sha256_file(
        output / "pillar2_diagnostic_report.json"
    )
    assert (output / "pillar2_diagnostic_report.json").stat().st_mode & 0o222 == 0
    validation = validate_pillar2_diagnostic_package(output)
    assert validation["status"] == "PASS"
    with pytest.raises(Pillar2DiagnosticError, match="not empty"):
        write_pillar2_diagnostic_package(
            output, diagnostic=diagnostic, report=report
        )
    relabelled = dict(report)
    relabelled["paper_table_status"] = "FINAL"
    with pytest.raises(Exception, match="paper_table_status"):
        validate_pillar2_diagnostic_report(relabelled)
    report_path = output / "pillar2_diagnostic_report.json"
    report_path.chmod(0o644)
    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="report hash mismatch"):
        validate_pillar2_diagnostic_package(output)


def test_donor_choice_and_output_are_deterministic(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    first = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )
    second = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )
    assert sha256_json(first) == sha256_json(second)


def _summary_statistics(report):
    summary = report["summary"]
    for family_name in (
        "pre_action_modality_sensitivity",
        "post_action_controls",
    ):
        for condition in summary[family_name].values():
            for metric_name, metric in condition.items():
                if metric_name != "example_count":
                    yield metric_name, metric


def test_every_diagnostic_rate_and_mean_has_clustered_95_ci_evidence(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    report = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )

    common = {
        "metric_kind",
        "estimate",
        "ci_low",
        "ci_high",
        "ci95_low",
        "ci95_high",
        "confidence",
        "interval_method",
        "inference_unit",
        "cluster_unit",
        "contribution_unit",
        "n_task_clusters",
        "bootstrap_samples",
        "bootstrap_seed",
        "valid_bootstrap_samples",
        "interval_status",
        "reason",
    }
    statistics = list(_summary_statistics(report))
    assert statistics
    for _, metric in statistics:
        count_fields = (
            {"numerator", "denominator"}
            if metric["metric_kind"] == "rate"
            else {"total", "count"}
        )
        assert set(metric) == common | count_fields
        assert metric["confidence"] == 0.95
        assert metric["interval_method"] == "percentile_task_cluster_bootstrap"
        assert metric["inference_unit"] == "task_id"
        assert metric["cluster_unit"] == "task_id"
        assert metric["contribution_unit"] == "diagnostic_example"
        assert metric["n_task_clusters"] == 2
        assert metric["bootstrap_samples"] == 200
        assert metric["valid_bootstrap_samples"] == 200
        assert metric["interval_status"] == "ESTIMATED"
        assert metric["reason"] is None
        assert metric["ci_low"] == metric["ci95_low"]
        assert metric["ci_high"] == metric["ci95_high"]


def test_summary_interval_and_schema_tampering_fail_validation(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    report = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )
    changed_interval = deepcopy(report)
    metric = changed_interval["summary"]["pre_action_modality_sensitivity"][
        "full"
    ]["action_accuracy"]
    metric["ci_low"] = 0.123456
    with pytest.raises(Exception, match="95% CI|not reproducible"):
        validate_pillar2_diagnostic_report(changed_interval)

    missing_reason = deepcopy(report)
    del missing_reason["summary"]["post_action_controls"]["full"][
        "outcome_accuracy"
    ]["reason"]
    with pytest.raises(Exception, match="fields mismatch"):
        validate_pillar2_diagnostic_report(missing_reason)


def test_diagnostic_intervals_state_why_ci_is_not_estimable(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    one_task_examples = tuple(
        replace(
            example,
            task_id="one-task-cluster",
            targets=replace(example.targets, action_bbox=None),
        )
        for example in diagnostic.examples
    )
    one_task = replace(diagnostic, examples=one_task_examples)
    report = _run_pillar2_fixture(
        one_task,
        image_root=image_root,
        predictor=DeterministicP2Predictor(one_task.backend_identity),
    )
    action_rate = report["summary"]["pre_action_modality_sensitivity"]["full"][
        "action_accuracy"
    ]
    assert action_rate["denominator"] == 2
    assert action_rate["n_task_clusters"] == 1
    assert action_rate["ci95_low"] is None
    assert action_rate["interval_status"] == (
        "NOT_ESTIMABLE_FEWER_THAN_TWO_TASK_CLUSTERS"
    )
    assert action_rate["reason"] == "fewer than two unique task clusters"

    bbox_mean = report["summary"]["pre_action_modality_sensitivity"]["full"][
        "bbox_iou_mean"
    ]
    assert bbox_mean["total"] == 0.0
    assert bbox_mean["count"] == 0
    assert bbox_mean["estimate"] is None
    assert bbox_mean["n_task_clusters"] == 0
    assert bbox_mean["interval_status"] == "NOT_APPLICABLE"
    assert bbox_mean["reason"] == "no eligible diagnostic examples"
    validate_pillar2_diagnostic_report(report)


def test_generic_predictor_cannot_claim_pc01_companion_evidence(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    relabelled = replace(diagnostic, evidence_role=PILLAR2_EVIDENCE_ROLE)
    with pytest.raises(Pillar2DiagnosticError, match="evidence role is inconsistent"):
        run_pillar2_diagnostics(
            relabelled,
            image_root=image_root,
            predictor=DeterministicP2Predictor(relabelled.backend_identity),
            allow_uncommitted_engineering=True,
        )
    with pytest.raises(Pillar2DiagnosticError, match="clean repository verification"):
        run_pillar2_diagnostics(
            diagnostic,
            image_root=image_root,
            predictor=DeterministicP2Predictor(diagnostic.backend_identity),
        )
    generic_report = _run_pillar2_fixture(
        diagnostic,
        image_root=image_root,
        predictor=DeterministicP2Predictor(diagnostic.backend_identity),
    )
    relabelled_report = dict(generic_report)
    relabelled_report["evidence_role"] = PILLAR2_EVIDENCE_ROLE
    relabelled_report["promotion_status"] = "CANONICAL_PC01_COMPANION_EVIDENCE"
    with pytest.raises(Exception, match="not canonical PC-01"):
        validate_pillar2_diagnostic_report(relabelled_report)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository, text=True
    ).strip()


def test_preimport_git_gate_requires_exact_clean_commit_and_tracked_source(
    tmp_path: Path,
):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "P2 Fixture")
    _git(repository, "config", "user.email", "p2@example.invalid")
    source = repository / "predictor.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "predictor.py")
    _git(repository, "commit", "-q", "-m", "fixture")
    commit = _git(repository, "rev-parse", "HEAD")

    receipt = _verify_clean_repository_before_import(
        repository, expected_commit=commit
    )
    assert receipt["repository_commit"] == commit
    assert _verify_committed_source_file(
        repository, source, expected_commit=commit
    ) == "predictor.py"
    with pytest.raises(RuntimeError, match="commit differs"):
        _verify_clean_repository_before_import(
            repository, expected_commit="0" * 40
        )

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git checkout"):
        _verify_clean_repository_before_import(repository, expected_commit=commit)
    with pytest.raises(RuntimeError, match="differs from registered commit bytes"):
        _verify_committed_source_file(repository, source, expected_commit=commit)


def test_cli_imports_project_code_only_after_clean_git_gate():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_table2_pillar2_diagnostics.py"
    )
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            assert not (statement.module or "").startswith("web_agent")
        elif isinstance(statement, ast.Import):
            assert all(not alias.name.startswith("web_agent") for alias in statement.names)
    assert source.index(
        "starting_git = _verify_clean_repository_before_import"
    ) < source.index(
        'diagnostics = importlib.import_module('
    ) < source.index(
        "factory = _load_factory("
    )
    assert source.index("report = diagnostics.run_pillar2_diagnostics(") < source.index(
        "ending_git = _verify_clean_repository_before_import"
    )


def test_cli_help_discloses_input_provenance_blocked_status() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_table2_pillar2_diagnostics.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert INPUT_PROVENANCE_BLOCKED_PROMOTION in result.stdout


class _FakeBatchBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        import torch

        return {"input_ids": torch.tensor([[len(self.calls)]])}


class _FakePC01Runtime:
    def __init__(self, identity: DiagnosticBackendIdentity) -> None:
        import torch

        self.torch = torch
        self.checkpoint_sha256 = identity.checkpoint_sha256
        self.processor_contract = SimpleNamespace(
            record_sha256=identity.processor_contract_sha256
        )
        self.batch = _FakeBatchBuilder()
        self.lock = threading.Lock()
        self.model = self
        self.action_calls: list[Any] = []

    def _assert_eval(self) -> None:
        return None

    def predict_action(self, task, observation, rng):
        self.action_calls.append((task, observation, rng))
        return SimpleNamespace(
            action_probabilities={
                "CLICK": 0.8,
                "TYPE": 0.04,
                "SELECT": 0.04,
                "SCROLL": 0.04,
                "NAVIGATE": 0.04,
                "PRESS_KEY": 0.04,
            },
            bbox=(0.1, 0.2, 0.2, 0.2),
            confidence_before=0.75,
        )

    def __call__(self, batch):
        del batch
        torch = self.torch
        return {
            "outcome": torch.tensor([[0.25, 0.75]]).log(),
            "failure_type": torch.tensor([[0.1, 0.2, 0.6, 0.1]]).log(),
            "recovery": torch.tensor([[0.1, 0.1, 0.4, 0.1, 0.2, 0.1]]).log(),
            "needs_recovery": torch.tensor([[1.0]]),
        }


def _canonical_pc01_identity(
    base: DiagnosticBackendIdentity,
) -> DiagnosticBackendIdentity:
    return replace(
        base,
        predictor_id=PC01_PILLAR2_PREDICTOR_ID,
        predictor_version=PC01_PILLAR2_PREDICTOR_VERSION,
        predictor_module=PC01_PILLAR2_PREDICTOR_MODULE,
        predictor_qualname=PC01_PILLAR2_PREDICTOR_QUALNAME,
        model_id=PC01_MODEL_ID,
        model_revision=PC01_MODEL_REVISION,
        checkpoint_sha256=PC01_EXPECTED_CHECKPOINT_SHA256,
        resolved_config_sha256=PC01_EXPECTED_CONFIG_SHA256,
        resolved_config_record_sha256=PC01_EXPECTED_CONFIG_SHA256,
        processor_contract_sha256=PC01_EXPECTED_PROCESSOR_CONTRACT_SHA256,
        predictor_source_sha256=pc01_pillar2_predictor_source_sha256(),
        model_seed=42,
    )


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("predictor_module", "lookalike.predictor"),
        ("predictor_qualname", "LookalikePC01Predictor"),
        ("predictor_source_sha256", "9" * 64),
        ("model_id", "lookalike-model"),
        ("model_revision", "0" * 40),
        ("checkpoint_sha256", "8" * 64),
        ("resolved_config_sha256", "7" * 64),
        ("resolved_config_record_sha256", "6" * 64),
        ("processor_contract_sha256", "5" * 64),
        ("model_seed", 43),
    ),
)
def test_canonical_pc01_identity_rejects_every_changed_registered_binding(
    field: str,
    changed: Any,
):
    identity = _canonical_pc01_identity(_identity())
    assert canonical_pc01_backend_error(identity) is None
    assert canonical_pc01_backend_error(replace(identity, **{field: changed}))
    assert "exact predictor class" in canonical_pc01_backend_error(
        identity,
        predictor=DeterministicP2Predictor(identity),
    )


def test_pc01_adapter_uses_exact_pre_and_action_conditioned_post_streams(tmp_path: Path):
    diagnostic, image_root = _diagnostic(tmp_path)
    identity = _canonical_pc01_identity(diagnostic.backend_identity)
    runtime = _FakePC01Runtime(identity)
    predictor = PC01Pillar2DiagnosticPredictor(runtime=runtime, identity=identity)
    example = diagnostic.examples[0]
    pre_image = example.pre_image.resolve(image_root)
    post_image = example.post_image.resolve(image_root)
    pre_view = PreDiagnosticView(
        example.example_id,
        example.task_id,
        pre_image,
        example.text_state,
    )
    pre_prediction = predictor.predict_pre(pre_view, seed=42)
    assert max(
        pre_prediction.action_probabilities,
        key=pre_prediction.action_probabilities.get,
    ) == "CLICK"
    assert runtime.action_calls[0][0].goal == example.text_state.task_text
    assert runtime.action_calls[0][1].screenshot_sha256 == pre_image.sha256

    post_view = PostDiagnosticView(
        example.example_id,
        example.task_id,
        pre_image,
        post_image,
        example.text_state,
        None,
    )
    post_prediction = predictor.predict_post(post_view, seed=42)
    assert post_prediction.failure_probability == pytest.approx(0.75)
    assert [call["phase"] for call in runtime.batch.calls] == ["pre", "post"]
    assert runtime.batch.calls[1]["executed_action"] == ""
    assert runtime.batch.calls[1]["action_value"] == ""
    assert len(runtime.batch.calls[1]["observations"]) == 2


def _clean_repository_receipt(
    repository_root: Path, identity: DiagnosticBackendIdentity
) -> dict[str, Any]:
    return {
        "status": "CLEAN_VERIFIED",
        "repository_root_sha256": sha256_bytes(
            str(repository_root.resolve()).encode("utf-8")
        ),
        "repository_commit": identity.repository_commit,
        "git_status_porcelain_sha256": sha256_bytes(b""),
    }


def test_exact_pc01_runs_only_as_clean_input_provenance_blocked_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic, image_root = _diagnostic(tmp_path)
    identity = _canonical_pc01_identity(diagnostic.backend_identity)
    diagnostic = replace(
        diagnostic,
        backend_identity=identity,
        bootstrap_samples=10_000,
        evidence_role=ENGINEERING_EVIDENCE_ROLE,
    )
    predictor = PC01Pillar2DiagnosticPredictor(
        runtime=_FakePC01Runtime(identity), identity=identity
    )
    repository_root = tmp_path / "clean-repository"
    repository_root.mkdir()
    clean_receipt = _clean_repository_receipt(repository_root, identity)

    monkeypatch.setattr(
        "web_agent.eval.table2.pillar2_diagnostics.verify_clean_repository",
        lambda root, *, expected_commit: dict(clean_receipt),
    )
    monkeypatch.setattr(
        "web_agent.eval.table2.pillar2_diagnostics._verify_committed_repository_source",
        lambda root, source, *, expected_commit: "src/exact-pc01-p2.py",
    )

    report = run_pillar2_diagnostics(
        diagnostic,
        image_root=image_root,
        predictor=predictor,
        repository_root=repository_root,
    )
    assert report["evidence_role"] == ENGINEERING_EVIDENCE_ROLE
    assert report["promotion_status"] == INPUT_PROVENANCE_BLOCKED_PROMOTION
    assert report["repository_verification"] == clean_receipt
    assert report["paper_table_status"] == "N/R"
    validate_pillar2_diagnostic_report(report)

    package_dir = tmp_path / "p2-exact-engineering-package"
    written = write_pillar2_diagnostic_package(
        package_dir, diagnostic=diagnostic, report=report
    )
    assert written["promotion_status"] == INPUT_PROVENANCE_BLOCKED_PROMOTION
    validated = validate_pillar2_diagnostic_package(package_dir)
    assert validated["status"] == "PASS"
    assert validated["paper_table_status"] == "N/R"


def test_exact_pc01_input_provenance_block_cannot_use_uncommitted_escape(
    tmp_path: Path,
) -> None:
    diagnostic, image_root = _diagnostic(tmp_path)
    identity = _canonical_pc01_identity(diagnostic.backend_identity)
    diagnostic = replace(
        diagnostic,
        backend_identity=identity,
        bootstrap_samples=10_000,
        evidence_role=ENGINEERING_EVIDENCE_ROLE,
    )
    predictor = PC01Pillar2DiagnosticPredictor(
        runtime=_FakePC01Runtime(identity), identity=identity
    )

    with pytest.raises(
        Pillar2DiagnosticError,
        match="exact PC-01 P2 diagnostics.*clean repository verification",
    ):
        run_pillar2_diagnostics(
            diagnostic,
            image_root=image_root,
            predictor=predictor,
            allow_uncommitted_engineering=True,
        )


def test_exact_pc01_blocked_report_cannot_hide_or_fabricate_source_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic, image_root = _diagnostic(tmp_path)
    identity = _canonical_pc01_identity(diagnostic.backend_identity)
    diagnostic = replace(
        diagnostic,
        backend_identity=identity,
        bootstrap_samples=10_000,
        evidence_role=ENGINEERING_EVIDENCE_ROLE,
    )
    predictor = PC01Pillar2DiagnosticPredictor(
        runtime=_FakePC01Runtime(identity), identity=identity
    )
    repository_root = tmp_path / "clean-repository"
    repository_root.mkdir()
    clean_receipt = _clean_repository_receipt(repository_root, identity)
    monkeypatch.setattr(
        "web_agent.eval.table2.pillar2_diagnostics.verify_clean_repository",
        lambda root, *, expected_commit: dict(clean_receipt),
    )
    monkeypatch.setattr(
        "web_agent.eval.table2.pillar2_diagnostics._verify_committed_repository_source",
        lambda root, source, *, expected_commit: "src/exact-pc01-p2.py",
    )
    report = run_pillar2_diagnostics(
        diagnostic,
        image_root=image_root,
        predictor=predictor,
        repository_root=repository_root,
    )

    hidden = deepcopy(report)
    hidden["promotion_status"] = "UNPROMOTABLE"
    with pytest.raises(SchemaError, match="must remain input-provenance blocked"):
        validate_pillar2_diagnostic_report(hidden)

    uncommitted = deepcopy(report)
    uncommitted["repository_verification"] = {
        "status": "ENGINEERING_UNCOMMITTED_UNPROMOTABLE",
        "repository_root_sha256": None,
        "repository_commit": identity.repository_commit,
        "git_status_porcelain_sha256": None,
    }
    with pytest.raises(SchemaError, match="lacks clean Git verification"):
        validate_pillar2_diagnostic_report(uncommitted)

    generic_workspace = tmp_path / "generic"
    generic_workspace.mkdir()
    generic, generic_root = _diagnostic(generic_workspace)
    generic_report = _run_pillar2_fixture(
        generic,
        image_root=generic_root,
        predictor=DeterministicP2Predictor(generic.backend_identity),
    )
    fabricated = deepcopy(generic_report)
    fabricated["promotion_status"] = INPUT_PROVENANCE_BLOCKED_PROMOTION
    with pytest.raises(SchemaError, match="generic P2 diagnostics"):
        validate_pillar2_diagnostic_report(fabricated)

    assert CANONICAL_INPUT_PROVENANCE_STATUS in str(
        _canonical_pc01_error(identity)
    )
