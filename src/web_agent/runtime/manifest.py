"""Immutable campaign/runtime manifest helpers and artifact verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping

from web_agent.runtime.contracts import (
    JsonValue,
    SystemID,
    VersionedRecord,
    canonical_json,
    canonical_sha256,
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash relative paths and bytes for a deterministic directory identity."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactReference(VersionedRecord):
    role: str
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.role or not self.path:
            raise ValueError("artifact role and path must be non-empty")
        if len(self.sha256) != 64:
            raise ValueError("artifact SHA-256 must contain 64 hex characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("artifact SHA-256 is not hexadecimal") from exc
        if self.size_bytes < 0:
            raise ValueError("artifact size cannot be negative")

    @classmethod
    def from_file(cls, role: str, path: str | Path) -> "ArtifactReference":
        source = Path(path)
        return cls(
            role=role,
            path=str(source),
            sha256=sha256_file(source),
            size_bytes=source.stat().st_size,
        )

    def verify(self) -> None:
        source = Path(self.path)
        if not source.is_file():
            raise FileNotFoundError(f"missing {self.role} artifact: {source}")
        if source.stat().st_size != self.size_bytes:
            raise ValueError(f"{self.role} artifact size changed: {source}")
        actual = sha256_file(source)
        if actual != self.sha256:
            raise ValueError(
                f"{self.role} artifact hash mismatch: {actual} != {self.sha256}"
            )


@dataclass(frozen=True, slots=True)
class ResolvedSystemManifest(VersionedRecord):
    system_id: SystemID
    overlay_sha256: str
    policy_source: str
    policy_id: str
    policy_version: str
    checkpoint_sha256: str | None
    memory_index_sha256: str | None
    resolved_switches: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class CampaignManifest(VersionedRecord):
    campaign_id: str
    protocol_id: str
    protocol_sha256: str
    code_commit: str
    task_manifest: ArtifactReference
    oracle_manifest: ArtifactReference
    environment_manifest: ArtifactReference
    provider_prompt: ArtifactReference
    systems: tuple[ResolvedSystemManifest, ...]
    locked_test_opened: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.protocol_id or not self.code_commit:
            raise ValueError("campaign/protocol/code identities are required")
        if len(self.protocol_sha256) != 64:
            raise ValueError("protocol SHA-256 must contain 64 characters")
        if tuple(item.system_id for item in self.systems) != tuple(SystemID):
            raise ValueError("campaign systems must appear exactly as E0, E1, E2, E3")

    def verify_artifacts(self) -> None:
        for artifact in (
            self.task_manifest,
            self.oracle_manifest,
            self.environment_manifest,
            self.provider_prompt,
        ):
            artifact.verify()

    @property
    def campaign_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuntimeEpisodeManifest(VersionedRecord):
    campaign_id: str
    campaign_sha256: str
    protocol_id: str
    system_id: SystemID
    task_id: str
    repeat_id: int
    matched_seed: int
    rerun_ordinal: int
    event_files: Mapping[str, str]
    runtime_summary_path: str

    def __post_init__(self) -> None:
        if min(self.repeat_id, self.rerun_ordinal) < 0:
            raise ValueError("repeat/rerun ordinals cannot be negative")
        if len(self.campaign_sha256) != 64:
            raise ValueError("episode campaign hash must contain 64 characters")


def write_manifest(
    path: str | Path,
    record: VersionedRecord,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(record.to_json(indent=2) + "\n", encoding="utf-8")
    return destination


def load_manifest_payload(path: str | Path) -> Mapping[str, JsonValue]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root is not a mapping")
    if payload.get("schema_version") != VersionedRecord.schema_version:
        raise ValueError("manifest schema version mismatch")
    return payload
