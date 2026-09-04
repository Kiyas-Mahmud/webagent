"""Git-index-only checks for Table 2 raw and secret artifacts."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
import re
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
    ".env",
    ".env.*",
    "cookies*.json",
    "credentials*.json",
    "storage_state*.json",
    "*token*.json",
    "*secret*.json",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
)
ALLOWED_SECRET_TEMPLATE_NAMES = frozenset({".env.example", ".env.template"})
MAX_SECRET_SCAN_BYTES = 5 * 1024 * 1024
HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,255}(?![A-Za-z0-9])"),
    re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{34,}(?![A-Za-z0-9])"),
    re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{32,}(?![A-Za-z0-9])"),
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
        normalized = PurePosixPath(path).as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        lowered = normalized.casefold()
        basename = PurePosixPath(lowered).name
        forbidden = (
            any(lowered.startswith(root) for root in FORBIDDEN_ROOTS)
            or any(lowered.endswith(suffix) for suffix in FORBIDDEN_RAW_SUFFIXES)
            or any(
                segment in f"/{lowered}/"
                for segment in FORBIDDEN_RUNTIME_DIRECTORIES
            )
            or (
                basename not in ALLOWED_SECRET_TEMPLATE_NAMES
                and any(
                    fnmatch(basename, pattern)
                    for pattern in FORBIDDEN_SECRET_NAMES
                )
            )
        )
        if forbidden:
            violations.append(normalized)
    return tuple(sorted(violations))


def _secret_content_violations(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Scan only small tracked files for unmistakable credential material."""

    violations: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > MAX_SECRET_SCAN_BYTES:
            continue
        payload = path.read_bytes()
        if any(pattern.search(payload) for pattern in HIGH_CONFIDENCE_SECRET_PATTERNS):
            violations.append(PurePosixPath(relative).as_posix())
    return tuple(sorted(violations))


def test_git_index_contains_no_table2_raw_locked_or_browser_secret_artifacts() -> None:
    violations = _hygiene_violations(_tracked_paths())
    assert not violations, (
        "Git tracks forbidden Table 2 raw, locked, or browser-secret artifacts: "
        f"{list(violations)}"
    )


def test_git_index_contains_no_high_confidence_secret_material() -> None:
    violations = _secret_content_violations(_tracked_paths())
    assert not violations, (
        "Git tracks files containing high-confidence credential material: "
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
        ".env.production",
        "config/service-token.json",
        "config/client_secret.json",
        "keys/model-plane.pem",
        "keys/id_ed25519",
    )
    assert _hygiene_violations(canaries) == tuple(sorted(canaries))


def test_hygiene_allows_only_explicit_environment_templates() -> None:
    assert _hygiene_violations((".env.example", ".env.template")) == ()
    assert _hygiene_violations((".env", ".env.local")) == (".env", ".env.local")
