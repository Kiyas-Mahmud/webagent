from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import web_agent.memory.kaggle_prepare_only as kaggle_prepare


def _downloaded_output(tmp_path: Path) -> Path:
    output = tmp_path / "table2-p4-prepare-only-v1"
    (output / "preparation").mkdir(parents=True)
    (output / "execution_receipt.json").write_text("{}\n", encoding="utf-8")
    (output / "execution_receipt.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    return output


def test_downloaded_output_cli_delegates_to_strict_production_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)
    observed: dict[str, object] = {}

    def validate(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {
            "execution_receipt_sha256": "a" * 64,
            "receipt_core_sha256": "b" * 64,
            "executed_source_set_sha256": "c" * 64,
            "source_commit": "d" * 40,
        }

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        validate,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 0
    assert observed == {
        "preparation_root": output / "preparation",
        "repository_root": repository,
        "require_clean_git_checkout": True,
    }
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS"
    assert result["package_status"] == "REVIEW_REQUIRED"
    assert result["paper_table_status"] == "N/R"
    assert result["source_commit"] == "d" * 40


def test_downloaded_output_cli_fails_closed_on_production_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)

    def reject(**_: object) -> dict[str, str]:
        raise kaggle_prepare.KaggleP4PrepareOnlyError("receipt sidecar mismatch")

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        reject,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "error": "receipt sidecar mismatch",
        "paper_table_status": "N/R",
        "status": "FAIL",
        "validation_scope": "DOWNLOADED_PREPARE_ONLY_RECEIPT_AND_OUTPUT_BYTES",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "output",
        "preparation",
        "execution_receipt.json",
        "execution_receipt.sha256",
    ],
)
def test_downloaded_output_cli_rejects_incomplete_layout_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = tmp_path / "table2-p4-prepare-only-v1"
    if missing != "output":
        output = _downloaded_output(tmp_path)
        target = output / missing
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()

    def unexpected(**_: object) -> dict[str, str]:
        raise AssertionError("production validator must not receive an unsafe layout")

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        unexpected,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert result["paper_table_status"] == "N/R"


@pytest.mark.parametrize("extra_kind", ["file", "empty_directory"])
def test_downloaded_output_cli_rejects_extra_outer_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_kind: str,
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)
    extra = output / "unregistered"
    if extra_kind == "file":
        extra.write_text("not registered\n", encoding="utf-8")
    else:
        extra.mkdir()

    def unexpected(**_: object) -> dict[str, str]:
        raise AssertionError("inner validator must not receive a mixed outer package")

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        unexpected,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "exact allowlist" in result["error"]


def test_downloaded_output_cli_rejects_symlinked_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)
    (output / "preparation").rmdir()
    real_preparation = tmp_path / "real-preparation"
    real_preparation.mkdir()
    (output / "preparation").symlink_to(real_preparation, target_is_directory=True)

    def unexpected(**_: object) -> dict[str, str]:
        raise AssertionError("inner validator must not receive a symlinked package")

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        unexpected,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "symlink" in result["error"]


def test_downloaded_output_cli_rejects_symlinked_outer_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    real_output = _downloaded_output(tmp_path)
    output = tmp_path / "download-alias"
    output.symlink_to(real_output, target_is_directory=True)

    def unexpected(**_: object) -> dict[str, str]:
        raise AssertionError("inner validator must not receive a symlinked package")

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        unexpected,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "symlink" in result["error"]


@pytest.mark.parametrize(
    "filename", ["execution_receipt.json", "execution_receipt.sha256"]
)
def test_downloaded_output_cli_rejects_symlinked_receipt_or_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filename: str,
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)
    target = output / filename
    target.unlink()
    external = tmp_path / f"external-{filename}"
    external.write_text("{}\n", encoding="utf-8")
    target.symlink_to(external)

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        lambda **_: pytest.fail("unsafe outer package reached inner validator"),
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "symlink" in result["error"]


def test_downloaded_output_cli_rejects_nonregular_outer_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)
    receipt = output / "execution_receipt.json"
    receipt.unlink()
    os.mkfifo(receipt)

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        lambda **_: pytest.fail("unsafe outer package reached inner validator"),
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "regular file" in result["error"]


@pytest.mark.parametrize(
    "filename", ["execution_receipt.json", "execution_receipt.sha256"]
)
def test_downloaded_output_cli_rejects_hard_linked_receipt_or_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filename: str,
) -> None:
    repository = tmp_path / "clean-source"
    repository.mkdir()
    output = _downloaded_output(tmp_path)
    target = output / filename
    target.unlink()
    linked_source = tmp_path / f"linked-{filename}"
    linked_source.write_text("{}\n", encoding="utf-8")
    os.link(linked_source, target)

    def unexpected(**_: object) -> dict[str, str]:
        raise AssertionError("inner validator must not receive a hard-linked package")

    monkeypatch.setattr(
        kaggle_prepare,
        "validate_prepare_only_execution_receipt",
        unexpected,
    )
    exit_code = kaggle_prepare.validate_downloaded_output_main(
        [
            "--repository-root",
            str(repository),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FAIL"
    assert "hard-linked" in result["error"]
