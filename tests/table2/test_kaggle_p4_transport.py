from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import ModuleType

import pytest

import web_agent.memory.kaggle_transport_staging as transport_staging
from web_agent.memory.kaggle_prepare_only import EXECUTED_SOURCE_RELATIVE_PATHS
from web_agent.memory.kaggle_transport_staging import (
    BOOTSTRAP_RELATIVE_PATH,
    BUNDLE_NAME,
    REGISTERED_DATASET_SOURCES,
    TRANSPORT_MANIFEST_NAME,
    KaggleTransportStagingError,
    stage_kaggle_transport,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_ID = "fixture-owner/table2-p4-source-transport-v1"
KERNEL_ID = "fixture-owner/table2-p4-prepare-only-v1"


def test_local_stager_git_subprocess_uses_fixed_credential_free_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("KAGGLE_KEY", "must-not-reach-local-git")
    monkeypatch.setenv("TABLE2_TEST_SECRET", "must-not-reach-local-git")
    monkeypatch.setattr(transport_staging.subprocess, "run", fake_run)
    transport_staging._git(tmp_path, "status", "--porcelain")
    environment = observed["kwargs"]["env"]
    assert "KAGGLE_KEY" not in environment
    assert "TABLE2_TEST_SECRET" not in environment
    assert set(environment) == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "LANG",
        "LC_ALL",
        "PATH",
        "TZ",
    }
    assert Path(observed["command"][0]).is_absolute()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "fixture-repository"
    relative_paths = {
        *EXECUTED_SOURCE_RELATIVE_PATHS,
        "configs/eval/table2/kaggle_p4_prepare_only_v1.json",
        "configs/eval/table2/p4_source_authority_v1.json",
    }
    for relative in sorted(relative_paths):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / relative, destination)
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


def _stage(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    repository, commit = _fixture_repository(tmp_path)
    source_stage = tmp_path / "source-stage"
    kernel_stage = tmp_path / "kernel-stage"
    staged = stage_kaggle_transport(
        repository_root=repository,
        source_dataset_id=SOURCE_DATASET_ID,
        source_dataset_version=1,
        kernel_id=KERNEL_ID,
        source_output_dir=source_stage,
        kernel_output_dir=kernel_stage,
    )
    assert staged.repository_commit == commit
    input_root = tmp_path / "input"
    mount = input_root / SOURCE_DATASET_ID.split("/", 1)[1]
    shutil.copytree(source_stage, mount)
    working_root = tmp_path / "working"
    working_root.mkdir()
    return kernel_stage, input_root, mount, working_root, commit


def _load_bootstrap(script: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"table2_kaggle_bootstrap_{id(script)}", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rehash_not_needed(manifest: Path, mutate) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(manifest, payload)


def test_stager_generates_exact_private_metadata_from_explicit_ids(tmp_path: Path):
    repository, commit = _fixture_repository(tmp_path)
    source_stage = tmp_path / "source-stage"
    kernel_stage = tmp_path / "kernel-stage"
    result = stage_kaggle_transport(
        repository_root=repository,
        source_dataset_id=SOURCE_DATASET_ID,
        source_dataset_version=1,
        kernel_id=KERNEL_ID,
        source_output_dir=source_stage,
        kernel_output_dir=kernel_stage,
    )

    assert result.repository_commit == commit
    assert {path.name for path in kernel_stage.iterdir()} == {
        "run.py",
        "kernel-metadata.json",
    }
    kernel_metadata = json.loads(
        (kernel_stage / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert kernel_metadata["id"] == KERNEL_ID
    assert kernel_metadata["code_file"] == "run.py"
    assert kernel_metadata["is_private"] is True
    assert kernel_metadata["enable_gpu"] is False
    assert kernel_metadata["enable_tpu"] is False
    assert kernel_metadata["enable_internet"] is False
    assert kernel_metadata["dataset_sources"] == [
        *REGISTERED_DATASET_SOURCES,
        SOURCE_DATASET_ID,
    ]
    assert kernel_metadata["competition_sources"] == []
    assert kernel_metadata["kernel_sources"] == []
    assert kernel_metadata["model_sources"] == []
    dataset_metadata = json.loads(
        (source_stage / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    assert dataset_metadata["id"] == SOURCE_DATASET_ID
    private_policy = json.loads(
        (source_stage / "private-upload-policy.json").read_text(encoding="utf-8")
    )
    assert private_policy["required_visibility"] == "PRIVATE"
    assert private_policy["remote_action_performed_by_stager"] is False
    manifest = json.loads(
        (source_stage / TRANSPORT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["repository_commit"] == commit
    assert manifest["source_dataset_id"] == SOURCE_DATASET_ID
    assert manifest["source_dataset_version"] == 1
    assert manifest["bundle"]["relative_path"] == BUNDLE_NAME
    assert manifest["evidence_role"].endswith("NOT_SCIENTIFIC_SOURCE_AUTHORITY")


@pytest.mark.parametrize(
    "source_id,kernel_id",
    [
        ("REPLACE_WITH_USERNAME/source", KERNEL_ID),
        (SOURCE_DATASET_ID, "owner/kernel;touch-pwned"),
        ("owner/../escape", KERNEL_ID),
        ("owner/unregistered-source", KERNEL_ID),
        (SOURCE_DATASET_ID, "owner/unregistered-kernel"),
    ],
)
def test_stager_rejects_placeholders_and_shell_injection_ids(
    tmp_path: Path, source_id: str, kernel_id: str
):
    repository, _ = _fixture_repository(tmp_path)
    with pytest.raises(
        KaggleTransportStagingError, match="Kaggle ID|placeholder|slug must be exactly"
    ):
        stage_kaggle_transport(
            repository_root=repository,
            source_dataset_id=source_id,
            source_dataset_version=1,
            kernel_id=kernel_id,
            source_output_dir=tmp_path / "source-stage",
            kernel_output_dir=tmp_path / "kernel-stage",
        )
    assert not (tmp_path / "pwned").exists()


def test_stager_refuses_existing_output_and_non_v1_transport(tmp_path: Path):
    repository, _ = _fixture_repository(tmp_path)
    existing = tmp_path / "existing-source"
    existing.mkdir()
    with pytest.raises(KaggleTransportStagingError, match="overwrite"):
        stage_kaggle_transport(
            repository_root=repository,
            source_dataset_id=SOURCE_DATASET_ID,
            source_dataset_version=1,
            kernel_id=KERNEL_ID,
            source_output_dir=existing,
            kernel_output_dir=tmp_path / "kernel-stage",
        )
    with pytest.raises(KaggleTransportStagingError, match="explicitly supplied as 1"):
        stage_kaggle_transport(
            repository_root=repository,
            source_dataset_id=SOURCE_DATASET_ID,
            source_dataset_version=2,
            kernel_id=KERNEL_ID,
            source_output_dir=tmp_path / "source-stage",
            kernel_output_dir=tmp_path / "kernel-stage-2",
        )


def test_no_argument_bootstrap_clones_bundle_with_no_local_and_builds_exact_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel, input_root, mount, working, commit = _stage(tmp_path)
    bootstrap = _load_bootstrap(kernel / "run.py")
    calls: list[tuple[list[str], dict]] = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("KAGGLE_KEY", "must-not-reach-child")
    monkeypatch.setenv("TABLE2_TEST_SECRET", "must-not-reach-child")
    assert bootstrap.run_no_argument_bootstrap(
        input_root=input_root,
        working_root=working,
        script_path=kernel / "run.py",
        command_runner=command_runner,
    ) == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    checkout = working / bootstrap.CHECKOUT_RELATIVE_PATH
    assert command == [
        os.sys.executable,
        "-B",
        str(checkout / "scripts/run_table2_p4_kaggle_prepare_only.py"),
        "--repository-root",
        str(checkout),
        "--config",
        str(checkout / "configs/eval/table2/kaggle_p4_prepare_only_v1.json"),
        "--input-root",
        str(input_root),
        "--output-root",
        str(working / bootstrap.OUTPUT_RELATIVE_PATH),
        "--source-commit",
        commit,
        "--source-bundle",
        str(mount / BUNDLE_NAME),
    ]
    assert kwargs["check"] is False
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["PYTHONHASHSEED"] == "0"
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert kwargs["env"]["PYTHONPATH"] == str(checkout / "src")
    assert "KAGGLE_KEY" not in kwargs["env"]
    assert "TABLE2_TEST_SECRET" not in kwargs["env"]
    assert set(kwargs["env"]) == {
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TZ",
    }
    assert kwargs["env"]["PATH"] == os.defpath
    assert subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip() == commit
    assert subprocess.run(
        ["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
        check=False,
    ).returncode == 1
    assert subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain"], text=True
    ) == ""
    assert bootstrap.main(["--repository-root", "/tmp/injected"]) == 2


def test_bootstrap_git_environment_is_fixed_and_credential_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    kernel, input_root, _, working, _ = _stage(tmp_path)
    bootstrap = _load_bootstrap(kernel / "run.py")
    monkeypatch.setenv("KAGGLE_KEY", "must-not-reach-git")
    monkeypatch.setenv("TABLE2_TEST_SECRET", "must-not-reach-git")
    environment = bootstrap._safe_git_environment()
    assert "KAGGLE_KEY" not in environment
    assert "TABLE2_TEST_SECRET" not in environment
    assert set(environment) == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "LANG",
        "LC_ALL",
        "PATH",
        "TZ",
    }
    assert bootstrap.run_no_argument_bootstrap(
        input_root=input_root,
        working_root=working,
        script_path=kernel / "run.py",
        command_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    ) == 0


def test_generated_no_argument_script_reaches_authenticated_inner_runner(
    tmp_path: Path,
):
    kernel, input_root, _, working, _ = _stage(tmp_path)
    bootstrap = _load_bootstrap(kernel / "run.py")
    # The scientific Gold inputs are intentionally absent. Reaching their
    # fail-closed mount check proves the generated no-argument script found,
    # cloned, imported, and invoked the committed registered runner.
    assert bootstrap.run_no_argument_bootstrap(
        input_root=input_root,
        working_root=working,
        script_path=kernel / "run.py",
    ) == 2
    receipt_path = working / "table2-p4-prepare-only-v1/execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "table2-p4-kaggle-prepare-receipt-v3"
    assert receipt["source"]["repository_clean"] is True
    assert receipt["source"]["source_transport"]["format"] == "git_bundle"
    assert "exactly one permitted Kaggle mount" in receipt["error"]["message"]


def test_bootstrap_rejects_duplicate_manifest(tmp_path: Path):
    kernel, input_root, mount, working, _ = _stage(tmp_path)
    duplicate = mount / "nested" / TRANSPORT_MANIFEST_NAME
    duplicate.parent.mkdir()
    shutil.copy2(mount / TRANSPORT_MANIFEST_NAME, duplicate)
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="exactly one versioned"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )


def test_bootstrap_rejects_duplicate_bundle(tmp_path: Path):
    kernel, input_root, mount, working, _ = _stage(tmp_path)
    shutil.copy2(mount / BUNDLE_NAME, mount / "second.bundle")
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="exactly one Git bundle"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )


def test_bootstrap_rejects_bundle_path_escape_even_with_rewritten_manifest(
    tmp_path: Path,
):
    kernel, input_root, mount, working, _ = _stage(tmp_path)
    manifest = mount / TRANSPORT_MANIFEST_NAME
    _rehash_not_needed(
        manifest,
        lambda payload: payload["bundle"].__setitem__("relative_path", "../escape.bundle"),
    )
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="safe relative path"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )


def test_bootstrap_rejects_unregistered_transport_version(tmp_path: Path):
    kernel, input_root, mount, working, _ = _stage(tmp_path)
    manifest = mount / TRANSPORT_MANIFEST_NAME
    _rehash_not_needed(
        manifest,
        lambda payload: payload.__setitem__("source_dataset_version", 2),
    )
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="exactly 1"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_bootstrap_rejects_linked_bundle(tmp_path: Path, link_kind: str):
    kernel, input_root, mount, working, _ = _stage(tmp_path)
    bundle = mount / BUNDLE_NAME
    outside = tmp_path / "outside.bundle"
    shutil.copy2(bundle, outside)
    bundle.unlink()
    if link_kind == "symlink":
        bundle.symlink_to(outside)
    else:
        os.link(outside, bundle)
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="symlink|hard-linked"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )


def test_bootstrap_rejects_bundle_hash_and_commit_mismatch(tmp_path: Path):
    kernel, input_root, mount, working, _ = _stage(tmp_path)
    bundle = mount / BUNDLE_NAME
    data = bytearray(bundle.read_bytes())
    data[len(data) // 2] ^= 1
    bundle.write_bytes(data)
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="SHA-256 mismatch"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )

    kernel2, input2, mount2, working2, _ = _stage(tmp_path / "second")
    manifest2 = mount2 / TRANSPORT_MANIFEST_NAME
    _rehash_not_needed(
        manifest2,
        lambda payload: payload.__setitem__("repository_commit", "0" * 40),
    )
    bootstrap2 = _load_bootstrap(kernel2 / "run.py")
    with pytest.raises(bootstrap2.BootstrapError, match="exactly the registered commit"):
        bootstrap2.build_prepare_command(
            input_root=input2,
            working_root=working2,
            script_path=kernel2 / "run.py",
        )


@pytest.mark.parametrize(
    "relative",
    ["table2-p4-prepare-only-v1", "webagent-table2-p4-source-v1"],
)
def test_bootstrap_rejects_existing_output_without_overwrite(
    tmp_path: Path, relative: str
):
    kernel, input_root, _, working, _ = _stage(tmp_path)
    (working / relative).mkdir()
    bootstrap = _load_bootstrap(kernel / "run.py")
    with pytest.raises(bootstrap.BootstrapError, match="refusing to overwrite"):
        bootstrap.build_prepare_command(
            input_root=input_root,
            working_root=working,
            script_path=kernel / "run.py",
        )


def test_materialized_checkout_rejects_dirty_and_wrong_head(tmp_path: Path):
    kernel, input_root, _, working, commit = _stage(tmp_path)
    bootstrap = _load_bootstrap(kernel / "run.py")
    bootstrap.build_prepare_command(
        input_root=input_root,
        working_root=working,
        script_path=kernel / "run.py",
    )
    checkout = working / bootstrap.CHECKOUT_RELATIVE_PATH
    (checkout / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(bootstrap.BootstrapError, match="dirty"):
        bootstrap._verify_materialized_checkout(checkout, commit)
    (checkout / "untracked.txt").unlink()
    with pytest.raises(bootstrap.BootstrapError, match="differs"):
        bootstrap._verify_materialized_checkout(checkout, "0" * 40)
