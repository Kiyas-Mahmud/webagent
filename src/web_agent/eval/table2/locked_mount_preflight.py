"""Fail-closed locked-benchmark absence checks for the development pilot."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any


LOCKED_MOUNT_ATTESTATION_FIELD = "locked_mount_preflight"
LOCKED_MOUNT_ATTESTATION_SCHEMA = "table2-locked-mount-preflight-v1"
LOCKED_MOUNT_CHECK_ALGORITHM = "lexists_ismount_access_r_ok_v1"


class LockedMountPreflightError(RuntimeError):
    """The development-pilot runtime can access its locked benchmark mount."""


def configured_locked_mount(protocol: Mapping[str, Any]) -> str:
    benchmark = protocol.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise LockedMountPreflightError(
            "frozen protocol lacks its benchmark locked-mount boundary"
        )
    value = benchmark.get("locked_mount")
    if not isinstance(value, str) or not value.strip():
        raise LockedMountPreflightError(
            "frozen protocol must name a nonempty locked benchmark mount"
        )
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise LockedMountPreflightError(
            "development-pilot locked_mount must be repository-relative and contained"
        )
    return path.as_posix()


def resolve_locked_mount(
    repository_root: str | Path,
    protocol: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    configured = configured_locked_mount(protocol)
    # Keep the lexical path for the probes: resolving a dangling final symlink
    # would hide the directory entry from lexists(). ``..`` and absolute paths
    # were already rejected above, so this remains contained by construction.
    resolved = (root / configured).absolute()
    if resolved == root or root not in resolved.parents:
        raise LockedMountPreflightError(
            "development-pilot locked mount resolves outside the repository boundary"
        )
    return resolved


def inspect_locked_mount(
    repository_root: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the three independent access facts used by the frozen attestation."""

    path = resolve_locked_mount(repository_root, protocol)
    try:
        path_exists = bool(os.path.lexists(path))
        is_mounted = bool(os.path.ismount(path))
        readable = bool(os.access(path, os.R_OK))
    except OSError as exc:
        raise LockedMountPreflightError(
            "locked benchmark mount state could not be inspected safely"
        ) from exc
    return {
        "schema_version": LOCKED_MOUNT_ATTESTATION_SCHEMA,
        "configured_path": configured_locked_mount(protocol),
        "check_algorithm": LOCKED_MOUNT_CHECK_ALGORITHM,
        # lexists also rejects a dangling symlink placed at the registered path.
        "path_exists": path_exists,
        "is_mounted": is_mounted,
        "readable": readable,
    }


def assert_locked_mount_inaccessible(
    repository_root: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail unless the registered locked path is absent, unmounted, and unreadable."""

    state = inspect_locked_mount(repository_root, protocol)
    unsafe = [
        name
        for name in ("path_exists", "is_mounted", "readable")
        if state[name] is True
    ]
    if unsafe:
        raise LockedMountPreflightError(
            "locked benchmark mount is accessible during the development pilot: "
            f"{state['configured_path']} ({', '.join(unsafe)})"
        )
    return state


def validate_locked_mount_attestation(
    environment: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen environment's claim without trusting it as a live check."""

    value = environment.get(LOCKED_MOUNT_ATTESTATION_FIELD)
    if not isinstance(value, Mapping):
        raise LockedMountPreflightError(
            "frozen environment lacks locked_mount_preflight attestation"
        )
    expected = {
        "schema_version": LOCKED_MOUNT_ATTESTATION_SCHEMA,
        "configured_path": configured_locked_mount(protocol),
        "check_algorithm": LOCKED_MOUNT_CHECK_ALGORITHM,
        "path_exists": False,
        "is_mounted": False,
        "readable": False,
    }
    if set(value) != set(expected):
        raise LockedMountPreflightError(
            "frozen locked-mount attestation fields are not the registered closure"
        )
    for key, registered in expected.items():
        actual = value.get(key)
        if type(actual) is not type(registered) or actual != registered:
            raise LockedMountPreflightError(
                "frozen locked-mount attestation is unsafe or mismatched: "
                f"{key}={actual!r}"
            )
    return dict(value)


def assert_production_locked_mount_preflight(
    *,
    repository_root: str | Path,
    protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Require both a frozen absence attestation and a fresh live absence check."""

    validate_locked_mount_attestation(environment, protocol)
    return assert_locked_mount_inaccessible(repository_root, protocol)
