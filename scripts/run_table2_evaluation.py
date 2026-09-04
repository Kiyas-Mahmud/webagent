"""Run or resume frozen paired E0--E3 Table 2 blocks."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


EVALUATION_RUNNER_SCOPE = "FROZEN_EVALUATION_RUNNER"
FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH = "dependency.lock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument(
        "--runner",
        required=True,
        help=(
            "module:attribute for a callable/runtime object; the callable may "
            "accept the complete request mapping or supported keyword arguments"
        ),
    )
    parser.add_argument(
        "--runner-factory",
        action="store_true",
        help="call the selected entrypoint once with campaign_dir before execution",
    )
    parser.add_argument("--block-id", default=None, help="run/resume only one frozen block")
    parser.add_argument("--maximum-blocks", type=int, default=None)
    parser.add_argument(
        "--live-readiness-probe-for",
        type=Path,
        default=None,
        help=(
            "run exactly one explicit normal E0--E3 block in this isolated "
            "campaign and write a non-scored readiness receipt for the supplied "
            "otherwise-unstarted PILOT_ONLY target campaign"
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"evaluation bootstrap expected a JSON object: {path}")
    return value


def _bootstrap_verify_evaluation_source(
    campaign_dir: Path,
    *,
    runner_entrypoint: str,
) -> None:
    """Use only the standard library before importing the evaluation package."""

    campaign_root = campaign_dir.resolve()
    manifest = _read_mapping(campaign_root / "campaign_manifest.json")
    if manifest.get("campaign_mode") == "smoke":
        return
    if manifest.get("runner_identity_scope") != EVALUATION_RUNNER_SCOPE:
        raise RuntimeError("evaluation bootstrap lacks a frozen runner identity")
    attestation_path = campaign_root / "frozen" / "runner_attestation.json"
    if _sha256_file(attestation_path) != manifest.get("runner_attestation_sha256"):
        raise RuntimeError("evaluation bootstrap runner attestation hash mismatch")
    attestation = _read_mapping(attestation_path)
    if runner_entrypoint != attestation.get("runner_entrypoint"):
        raise RuntimeError("evaluation bootstrap runner entrypoint mismatch")

    repository_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("evaluation bootstrap requires a Git checkout") from exc
    if status:
        raise RuntimeError("evaluation bootstrap requires a clean Git checkout")
    if (
        not commit
        or commit != manifest.get("repository_commit")
        or commit != attestation.get("repository_commit")
    ):
        raise RuntimeError("evaluation bootstrap Git commit differs from campaign")

    rows = attestation.get("source_files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("evaluation bootstrap source attestation is malformed")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("evaluation bootstrap source row is malformed")
        relative_text = str(row.get("relative_path", ""))
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise RuntimeError("evaluation bootstrap source path is unsafe or duplicate")
        seen.add(relative_text)
        live = (repository_root / relative).resolve()
        frozen = (campaign_root / "frozen" / "runner_source" / relative).resolve()
        expected = row.get("sha256")
        if (
            repository_root not in live.parents
            or campaign_root not in frozen.parents
            or not live.is_file()
            or not frozen.is_file()
            or _sha256_file(live) != expected
            or _sha256_file(frozen) != expected
        ):
            raise RuntimeError(
                f"evaluation bootstrap source identity mismatch: {relative_text}"
            )
    own_relative = Path(__file__).resolve().relative_to(repository_root).as_posix()
    if own_relative not in seen:
        raise RuntimeError("evaluation CLI source is absent from runner attestation")

    environment = _read_mapping(campaign_root / "frozen" / "environment.json")
    if (
        environment.get("dependency_lock_relative_path")
        != FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH
    ):
        raise RuntimeError("evaluation bootstrap dependency-lock path is not frozen")
    dependency_lock = campaign_root / "frozen" / FROZEN_DEPENDENCY_LOCK_RELATIVE_PATH
    if (
        dependency_lock.is_symlink()
        or not dependency_lock.is_file()
        or _sha256_file(dependency_lock) != environment.get("dependency_lock_sha256")
    ):
        raise RuntimeError("evaluation bootstrap dependency-lock hash mismatch")


def main() -> None:
    args = parse_args()
    if args.maximum_blocks is not None and args.maximum_blocks <= 0:
        raise ValueError("--maximum-blocks must be positive")
    if args.live_readiness_probe_for is not None:
        if args.block_id is None:
            raise ValueError("--live-readiness-probe-for requires --block-id")
        if args.maximum_blocks is not None:
            raise ValueError(
                "--live-readiness-probe-for cannot be combined with --maximum-blocks"
            )
    _bootstrap_verify_evaluation_source(
        args.campaign_dir,
        runner_entrypoint=args.runner,
    )
    from web_agent.eval.table2.campaign import CampaignRunner, load_entrypoint

    entrypoint = load_entrypoint(args.runner)
    if args.runner_factory:
        entrypoint = entrypoint(campaign_dir=args.campaign_dir)
    elif inspect.isclass(entrypoint):
        entrypoint = entrypoint()
    campaign = CampaignRunner(
        campaign_dir=args.campaign_dir,
        runner=entrypoint,
        runner_entrypoint=args.runner,
        maximum_blocks=args.maximum_blocks,
        live_readiness_probe_target=args.live_readiness_probe_for,
    )
    result = campaign.run_block(args.block_id) if args.block_id else campaign.run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
