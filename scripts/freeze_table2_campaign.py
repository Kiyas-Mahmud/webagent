"""Freeze and hash every input to one Table 2 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.package_validator import freeze_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", required=True, type=Path)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--model-manifest", action="append", default=[], type=Path)
    parser.add_argument("--memory-manifest", action="append", default=[], type=Path)
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=None,
        help="frozen browser/container/hardware manifest (required outside smoke mode)",
    )
    parser.add_argument(
        "--runner-attestation",
        type=Path,
        default=None,
        help=(
            "frozen evaluation-runner/source/loaded-model attestation "
            "(required outside ENGINEERING_SMOKE_ONLY mode)"
        ),
    )
    parser.add_argument(
        "--resolved-task-snapshot",
        type=Path,
        default=None,
        help=(
            "content-complete WebArena task export bound to the tracked 0--49 "
            "registry (required outside ENGINEERING_SMOKE_ONLY mode)"
        ),
    )
    parser.add_argument(
        "--allow-dirty-pilot",
        action="store_true",
        help="permit an explicitly PILOT_ONLY freeze from a dirty worktree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = freeze_campaign(
        repository_root=args.repository_root,
        campaign_config_path=args.campaign_config,
        campaign_dir=args.campaign_dir,
        campaign_id=args.campaign_id,
        model_manifest_paths=args.model_manifest,
        memory_manifest_paths=args.memory_manifest,
        environment_manifest_path=args.environment_manifest,
        runner_attestation_path=args.runner_attestation,
        resolved_task_snapshot_path=args.resolved_task_snapshot,
        allow_dirty_pilot=args.allow_dirty_pilot,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
