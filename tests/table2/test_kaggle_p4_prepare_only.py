from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import web_agent.memory.kaggle_prepare_only as kaggle_prepare
from web_agent.memory.joint_duplicate_audit import (
    JointDuplicateAuditError,
    _preparation_execution_binding,
)
from web_agent.memory.kaggle_prepare_only import (
    EXECUTED_SOURCE_RELATIVE_PATHS,
    EXPECTED_DATASETS,
    KaggleP4PrepareOnlyError,
    SOURCE_ARCHIVE_EVIDENCE_ROLE,
    load_prepare_only_config,
    resolve_dataset_mount,
    resolve_registered_dataset_mounts,
    run_prepare_only,
    validate_compact_preparation_outputs,
    validate_prepare_only_execution_receipt,
)
from web_agent.memory.manifest import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY_ROOT / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"


def _write(path: Path, value: bytes = b"[]") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _mounts(root: Path, *, legacy: bool = False, deep: bool = False) -> Path:
    prefix = root / "datasets" / "kiyasmahmud" if legacy else root
    gold = prefix / "web-gold-40k"
    retry = prefix / "gold-40k-retry"
    gold_data = gold / "final_data_set_40k" if deep else gold
    retry_data = (
        retry / "web_gold_40k_retry_abort_supplement_v2_kaggle"
        if deep
        else retry
    )
    _write(gold_data / "split_train.json")
    _write(retry_data / "data" / "supplement_train.json")
    return root


def _clean_source_checkout(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "clean-source"
    relative_paths = {
        *EXECUTED_SOURCE_RELATIVE_PATHS,
        "configs/eval/table2/kaggle_p4_prepare_only_v1.json",
        "configs/eval/table2/p4_source_authority_v1.json",
    }
    for relative in sorted(relative_paths):
        source = REPOSITORY_ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "fixture@test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    return repository, commit


def test_registered_config_declares_exact_version_one_datasets():
    payload = load_prepare_only_config(CONFIG)
    assert payload["mode"] == "P4_PREPARE_ONLY"
    assert payload["paper_table_status"] == "N/R"
    assert [row["slug"] for row in payload["datasets"]] == [
        "kiyasmahmud/web-gold-40k",
        "kiyasmahmud/gold-40k-retry",
    ]
    assert [row["kaggle_dataset_version"] for row in payload["datasets"]] == [1, 1]
    assert payload["allowed_operations"] == [
        "audit-candidates",
        "validate-preparation",
    ]


def test_kaggle_metadata_is_private_cpu_offline_and_has_only_registered_sources():
    metadata = json.loads(
        (
            REPOSITORY_ROOT
            / "kaggle/table2_p4_prepare_only/kernel-metadata.template.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["dataset_sources"] == [
        "kiyasmahmud/web-gold-40k",
        "kiyasmahmud/gold-40k-retry",
    ]
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["is_private"] is True
    assert metadata["competition_sources"] == []
    assert metadata["kernel_sources"] == []


def test_config_is_semantically_frozen(tmp_path: Path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["datasets"][0]["kaggle_dataset_version"] = 2
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KaggleP4PrepareOnlyError, match="exact contract"):
        load_prepare_only_config(changed)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("deep", [False, True])
def test_mount_resolution_supports_only_known_kaggle_and_data_layouts(
    tmp_path: Path,
    legacy: bool,
    deep: bool,
):
    input_root = _mounts(tmp_path / "input", legacy=legacy, deep=deep)
    resolved = resolve_registered_dataset_mounts(input_root)
    assert [row.spec.role for row in resolved] == [
        "original_gold",
        "retry_abort_supplement_v2",
    ]
    assert all(row.spec.kaggle_dataset_version == 1 for row in resolved)
    assert resolved[0].train_json.name == "split_train.json"
    assert resolved[1].train_json.name == "supplement_train.json"


def test_mount_resolution_rejects_zero_and_multiple_mounts(tmp_path: Path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    with pytest.raises(KaggleP4PrepareOnlyError, match="exactly one"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])

    _write(input_root / "web-gold-40k" / "split_train.json")
    _write(
        input_root
        / "datasets"
        / "kiyasmahmud"
        / "web-gold-40k"
        / "split_train.json"
    )
    with pytest.raises(KaggleP4PrepareOnlyError, match="exactly one"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])


def test_mount_resolution_rejects_unregistered_train_root(tmp_path: Path):
    input_root = tmp_path / "input"
    _write(input_root / "web-gold-40k" / "unregistered" / "split_train.json")
    with pytest.raises(KaggleP4PrepareOnlyError, match="unexpected root"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])


def test_mount_resolution_rejects_unregistered_mount_shape(tmp_path: Path):
    input_root = tmp_path / "input"
    _write(input_root / "unexpected-owner" / "web-gold-40k" / "split_train.json")
    with pytest.raises(KaggleP4PrepareOnlyError, match="unexpected mount root"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])


def test_mount_resolution_rejects_multiple_known_data_roots(tmp_path: Path):
    input_root = tmp_path / "input"
    mount = input_root / "web-gold-40k"
    _write(mount / "split_train.json")
    _write(mount / "final_data_set_40k" / "split_train.json")
    with pytest.raises(KaggleP4PrepareOnlyError, match="exactly one"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])


def test_mount_resolution_rejects_symlink_components(tmp_path: Path):
    input_root = tmp_path / "input"
    real = tmp_path / "real-gold"
    _write(real / "split_train.json")
    input_root.mkdir()
    (input_root / "web-gold-40k").symlink_to(real, target_is_directory=True)
    with pytest.raises(KaggleP4PrepareOnlyError, match="symlink"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])


def test_mount_resolution_rejects_symlinked_train_json(tmp_path: Path):
    input_root = tmp_path / "input"
    mount = input_root / "web-gold-40k"
    mount.mkdir(parents=True)
    actual = _write(tmp_path / "outside" / "split_train.json")
    (mount / "split_train.json").symlink_to(actual)
    with pytest.raises(KaggleP4PrepareOnlyError, match="symlink"):
        resolve_dataset_mount(input_root, EXPECTED_DATASETS[0])


@dataclass(frozen=True)
class _Package:
    status: str = "REVIEW_REQUIRED"


def _fake_package(path: Path) -> None:
    path.mkdir(parents=True)
    for name in (
        "candidate_audit.json",
        "preparation_manifest.json",
        "read_ledger.json",
        "source_authority.json",
    ):
        _write(path / name, b"{}\n")
    _write(path / "review_queue.jsonl", b"")
    _write(path / "preparation_manifest.sha256", b"a" * 64 + b"\n")


def test_compact_output_allowlist_rejects_nested_or_extra_artifacts(tmp_path: Path):
    package = tmp_path / "preparation"
    _fake_package(package)
    _write(package / "embeddings.npy", b"not permitted")
    with pytest.raises(KaggleP4PrepareOnlyError, match="allowlist"):
        validate_compact_preparation_outputs(package)


def test_runner_invokes_only_authenticated_prepare_then_validation_and_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_root = _mounts(tmp_path / "input", legacy=True, deep=True)
    repository, source_commit = _clean_source_checkout(tmp_path)
    calls: list[tuple[str, dict | str]] = []

    def prepare(**kwargs):
        calls.append(("audit-candidates", kwargs))
        _fake_package(Path(kwargs["output_dir"]))
        return _Package()

    def validate(path):
        calls.append(("validate-preparation", str(path)))
        return _Package()

    source_archive = _write(tmp_path / "source.tar", b"attested source")
    monkeypatch.setattr(kaggle_prepare, "prepare_p4_candidate_audit", prepare)
    monkeypatch.setattr(kaggle_prepare, "validate_p4_preparation_package", validate)
    result = run_prepare_only(
        repository_root=repository,
        config_path=(
            repository / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
        ),
        input_root=input_root,
        output_root=tmp_path / "output",
        argv=["--fixture"],
        source_commit=source_commit,
        source_archive=source_archive,
    )

    assert result.status == "REVIEW_REQUIRED"
    assert [row[0] for row in calls] == [
        "audit-candidates",
        "validate-preparation",
    ]
    prepare_args = calls[0][1]
    assert isinstance(prepare_args, dict)
    assert prepare_args["dataset_id"] == "web-gold-v2.8"
    assert prepare_args["dataset_version"].startswith("pc01-train-")
    assert Path(prepare_args["gold_train_json"]).name == "split_train.json"
    assert Path(prepare_args["supplement_train_json"]).name == "supplement_train.json"

    receipt = json.loads(
        (result.output_root / "execution_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "REVIEW_REQUIRED"
    assert receipt["paper_table_status"] == "N/R"
    assert receipt["stage_status"]["joint-duplicate-audit"] == "FORBIDDEN_NOT_RUN"
    assert receipt["scientific_nonclaims"] == {
        "eligible_memory_items_claimed": 0,
        "independent_review_performed": False,
        "joint_duplicate_audit_performed": False,
        "memory_store_built": False,
        "provenance_manifest_created": False,
        "table2_value_created": False,
    }
    assert receipt["zero_read_claims"]["scope"].startswith("APPLICATION_LEVEL")
    assert [row["declared_kaggle_dataset_version"] for row in receipt["datasets"]] == [
        1,
        1,
    ]
    assert receipt["source"]["source_archive"]["sha256"] == sha256_file(
        source_archive
    )
    assert receipt["source"]["source_archive"]["evidence_role"] == (
        SOURCE_ARCHIVE_EVIDENCE_ROLE
    )
    assert receipt["source"]["source_commit_supplied"] == source_commit
    assert receipt["source"]["repository_clean"] is True
    assert [
        row["relative_path"] for row in receipt["source"]["executed_source_files"]
    ] == list(EXECUTED_SOURCE_RELATIVE_PATHS)
    assert set(receipt["outputs"]["files"]) == {
        "candidate_audit.json",
        "preparation_manifest.json",
        "preparation_manifest.sha256",
        "read_ledger.json",
        "review_queue.jsonl",
        "source_authority.json",
    }
    assert (result.output_root / "execution_receipt.sha256").read_text().strip() == (
        sha256_file(result.output_root / "execution_receipt.json")
    )
    validated_receipt = validate_prepare_only_execution_receipt(
        preparation_root=result.output_root / "preparation",
        repository_root=repository,
    )
    assert validated_receipt["execution_receipt_sha256"] == sha256_file(
        result.output_root / "execution_receipt.json"
    )


def test_failed_in_boundary_run_keeps_only_fail_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_root = _mounts(tmp_path / "input")
    repository, source_commit = _clean_source_checkout(tmp_path)

    def fail_prepare(**_):
        raise RuntimeError("fixture preparation failure")

    monkeypatch.setattr(kaggle_prepare, "prepare_p4_candidate_audit", fail_prepare)
    result = run_prepare_only(
        repository_root=repository,
        config_path=(
            repository / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
        ),
        input_root=input_root,
        output_root=tmp_path / "failed-output",
        argv=[],
        source_commit=source_commit,
    )
    assert result.status == "FAIL"
    assert result.exit_code == 2
    assert sorted(path.name for path in result.output_root.iterdir()) == [
        "execution_receipt.json",
        "execution_receipt.sha256",
    ]
    assert result.receipt["error"]["message"] == "fixture preparation failure"
    assert result.receipt["stage_status"]["memory-store-build"] == "FORBIDDEN_NOT_RUN"


def test_runner_rejects_missing_source_commit_with_fail_receipt(tmp_path: Path):
    input_root = _mounts(tmp_path / "input")
    repository, _ = _clean_source_checkout(tmp_path)
    result = run_prepare_only(
        repository_root=repository,
        config_path=(
            repository / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
        ),
        input_root=input_root,
        output_root=tmp_path / "missing-commit-output",
        argv=[],
    )
    assert result.status == "FAIL"
    assert "requires --source-commit" in result.receipt["error"]["message"]


def test_runner_rejects_dirty_source_checkout(tmp_path: Path):
    input_root = _mounts(tmp_path / "input")
    repository, source_commit = _clean_source_checkout(tmp_path)
    executed_source = repository / EXECUTED_SOURCE_RELATIVE_PATHS[-1]
    executed_source.write_text(
        executed_source.read_text(encoding="utf-8") + "\n# dirty fixture\n",
        encoding="utf-8",
    )
    result = run_prepare_only(
        repository_root=repository,
        config_path=(
            repository / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
        ),
        input_root=input_root,
        output_root=tmp_path / "dirty-source-output",
        argv=[],
        source_commit=source_commit,
    )
    assert result.status == "FAIL"
    assert "clean Git checkout" in result.receipt["error"]["message"]


def test_runner_rechecks_source_after_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_root = _mounts(tmp_path / "input")
    repository, source_commit = _clean_source_checkout(tmp_path)

    def mutate_source_then_prepare(**kwargs):
        executed_source = repository / EXECUTED_SOURCE_RELATIVE_PATHS[-1]
        executed_source.write_text(
            executed_source.read_text(encoding="utf-8") + "\n# changed in run\n",
            encoding="utf-8",
        )
        _fake_package(Path(kwargs["output_dir"]))
        return _Package()

    monkeypatch.setattr(
        kaggle_prepare,
        "prepare_p4_candidate_audit",
        mutate_source_then_prepare,
    )
    monkeypatch.setattr(
        kaggle_prepare,
        "validate_p4_preparation_package",
        lambda _: _Package(),
    )
    result = run_prepare_only(
        repository_root=repository,
        config_path=(
            repository / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
        ),
        input_root=input_root,
        output_root=tmp_path / "changed-during-run-output",
        argv=[],
        source_commit=source_commit,
    )
    assert result.status == "FAIL"
    assert "dirty" in result.receipt["error"]["message"]
    assert sorted(path.name for path in result.output_root.iterdir()) == [
        "execution_receipt.json",
        "execution_receipt.sha256",
    ]


def test_registered_receipt_replay_checks_dirty_dot_git_file_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linked worktrees must not bypass clean-HEAD receipt authentication."""

    repository, source_commit = _clean_source_checkout(tmp_path)
    worktree = tmp_path / "linked-worktree"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "worktree",
            "add",
            "--detach",
            str(worktree),
            source_commit,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (worktree / ".git").is_file()

    input_root = _mounts(tmp_path / "worktree-input")

    def prepare(**kwargs):
        _fake_package(Path(kwargs["output_dir"]))
        return _Package()

    monkeypatch.setattr(kaggle_prepare, "prepare_p4_candidate_audit", prepare)
    monkeypatch.setattr(
        kaggle_prepare,
        "validate_p4_preparation_package",
        lambda _: _Package(),
    )
    result = run_prepare_only(
        repository_root=worktree,
        config_path=(
            worktree / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"
        ),
        input_root=input_root,
        output_root=tmp_path / "worktree-output",
        argv=["--worktree-fixture"],
        source_commit=source_commit,
    )
    assert result.status == "REVIEW_REQUIRED"

    binding = _preparation_execution_binding(
        preparation_root=result.output_root / "preparation",
        source_authority_path=(
            worktree / "configs/eval/table2/p4_source_authority_v1.json"
        ),
        repository_root=worktree,
    )
    assert binding["preparation_source_commit"] == source_commit

    (worktree / "untracked-after-receipt.txt").write_text(
        "must invalidate registered evidence\n",
        encoding="utf-8",
    )
    with pytest.raises(
        JointDuplicateAuditError,
        match="checkout is missing, dirty, or at another commit",
    ):
        _preparation_execution_binding(
            preparation_root=result.output_root / "preparation",
            source_authority_path=(
                worktree / "configs/eval/table2/p4_source_authority_v1.json"
            ),
            repository_root=worktree,
        )
