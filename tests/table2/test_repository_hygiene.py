"""Git-index-only checks for Table 2 raw and secret artifacts."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_ROOTS = (
    "artifacts/",
    "outputs/",
    "locked_benchmark_mount/",
    "browser_profiles/",
    "captures/",
    "private/",
    "playwright/.auth/",
    "playwright-report/",
    "test-results/",
)
FORBIDDEN_RAW_SUFFIXES = (
    ".har",
    ".webm",
    ".mp4",
    ".mkv",
    ".trace",
    ".trace.zip",
    ".faiss",
    ".index",
    ".embeddings.npy",
)
FORBIDDEN_RUNTIME_DIRECTORIES = (
    "/runtime/screenshots/",
)
FORBIDDEN_SECRET_NAMES = (
    "cookies*.json",
    "credentials*.json",
    "storage_state*.json",
)


def _tracked_paths() -> tuple[str, ...]:
    """Read only path names registered in Git's index, never file contents."""

    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        raw.decode("utf-8", errors="surrogateescape")
        for raw in result.stdout.split(b"\0")
        if raw
    )


def _hygiene_violations(paths: tuple[str, ...]) -> tuple[str, ...]:
    violations: list[str] = []
    for path in paths:
        normalized = PurePosixPath(path).as_posix().lstrip("./")
        lowered = normalized.casefold()
        basename = PurePosixPath(lowered).name
        forbidden = (
            any(lowered.startswith(root) for root in FORBIDDEN_ROOTS)
            or any(lowered.endswith(suffix) for suffix in FORBIDDEN_RAW_SUFFIXES)
            or any(
                segment in f"/{lowered}/"
                for segment in FORBIDDEN_RUNTIME_DIRECTORIES
            )
            or any(fnmatch(basename, pattern) for pattern in FORBIDDEN_SECRET_NAMES)
        )
        if forbidden:
            violations.append(normalized)
    return tuple(sorted(violations))


def test_git_index_contains_no_table2_raw_locked_or_browser_secret_artifacts() -> None:
    violations = _hygiene_violations(_tracked_paths())
    assert not violations, (
        "Git tracks forbidden Table 2 raw, locked, or browser-secret artifacts: "
        f"{list(violations)}"
    )


def test_hygiene_policy_covers_each_forbidden_artifact_class() -> None:
    canaries = (
        "artifacts/table2/campaign/actions.jsonl",
        "locked_benchmark_mount/task.json",
        "browser_profiles/default/state.sqlite",
        "playwright/.auth/session.json",
        "captures/episode.har",
        "captures/browser.trace",
        "captures/browser.trace.zip",
        "campaign/E3/runtime/screenshots/000001-state.png",
        "memory/seed_42.faiss",
        "memory/seed_42.embeddings.npy",
        "private/cookies-user.json",
        "private/credentials-lab.json",
        "private/storage_state-seed42.json",
        "private/secrets.json",
        "private/cookies.sqlite",
        "captures/screenshot.png",
    )
    assert _hygiene_violations(canaries) == tuple(sorted(canaries))
