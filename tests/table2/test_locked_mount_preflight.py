from __future__ import annotations

import os
from pathlib import Path

import pytest

from web_agent.eval.table2.locked_mount_preflight import (
    LockedMountPreflightError,
    assert_locked_mount_inaccessible,
    assert_production_locked_mount_preflight,
    inspect_locked_mount,
    validate_locked_mount_attestation,
)
from web_agent.eval.table2.production_runner import (
    ProductionRunnerError,
    ProductionTable2Runner,
)


def _protocol(path: str = "locked-benchmark") -> dict:
    return {
        "benchmark": {
            "locked_mount": path,
            "allow_locked_reads": False,
        }
    }


def _environment(root: Path, protocol: dict) -> dict:
    return {
        "locked_mount_preflight": inspect_locked_mount(root, protocol),
    }


def test_absent_unmounted_unreadable_path_passes_live_and_frozen_checks(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    environment = _environment(tmp_path, protocol)

    assert validate_locked_mount_attestation(environment, protocol)[
        "path_exists"
    ] is False
    state = assert_production_locked_mount_preflight(
        repository_root=tmp_path,
        protocol=protocol,
        environment=environment,
    )
    assert state["is_mounted"] is state["readable"] is False


def test_existing_locked_path_fails_even_when_it_is_an_empty_directory(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    (tmp_path / "locked-benchmark").mkdir()

    with pytest.raises(LockedMountPreflightError, match="path_exists"):
        assert_locked_mount_inaccessible(tmp_path, protocol)


def test_dangling_symlink_at_locked_path_is_treated_as_existing(
    tmp_path: Path,
) -> None:
    (tmp_path / "locked-benchmark").symlink_to(tmp_path / "missing-target")
    with pytest.raises(LockedMountPreflightError, match="path_exists"):
        assert_locked_mount_inaccessible(tmp_path, _protocol())


@pytest.mark.parametrize(
    ("probe", "message"),
    [("ismount", "is_mounted"), ("access", "readable")],
)
def test_each_independent_live_probe_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: str,
    message: str,
) -> None:
    protocol = _protocol()
    if probe == "ismount":
        monkeypatch.setattr(os.path, "ismount", lambda _: True)
    else:
        monkeypatch.setattr(os, "access", lambda *_: True)

    with pytest.raises(LockedMountPreflightError, match=message):
        assert_locked_mount_inaccessible(tmp_path, protocol)


def test_probe_error_fails_closed_instead_of_assuming_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os.path,
        "ismount",
        lambda _: (_ for _ in ()).throw(OSError("probe denied")),
    )
    with pytest.raises(LockedMountPreflightError, match="could not be inspected"):
        assert_locked_mount_inaccessible(tmp_path, _protocol())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configured_path", "some-other-path"),
        ("path_exists", True),
        ("is_mounted", True),
        ("readable", True),
        ("readable", 0),
    ],
)
def test_frozen_attestation_rejects_mismatch_placeholder_or_nonexact_bool(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    protocol = _protocol()
    environment = _environment(tmp_path, protocol)
    environment["locked_mount_preflight"][field] = value

    with pytest.raises(LockedMountPreflightError, match="attestation"):
        validate_locked_mount_attestation(environment, protocol)


def test_locked_path_must_be_repository_relative_and_contained(tmp_path: Path) -> None:
    for unsafe in ("../locked", str(tmp_path / "absolute-locked")):
        with pytest.raises(LockedMountPreflightError, match="repository-relative"):
            assert_locked_mount_inaccessible(tmp_path, _protocol(unsafe))


def test_production_runner_rechecks_live_state_after_frozen_attestation(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    runner = object.__new__(ProductionTable2Runner)
    runner.repository_root = tmp_path
    runner.protocol_mapping = protocol
    runner.environment = _environment(tmp_path, protocol)

    # The frozen state was safe, but the path appeared before an episode.
    (tmp_path / "locked-benchmark").mkdir()
    with pytest.raises(ProductionRunnerError, match="locked-mount preflight"):
        runner._assert_locked_mount_absent()
