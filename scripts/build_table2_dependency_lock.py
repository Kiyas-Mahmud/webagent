#!/usr/bin/env python3
"""Build or validate a semantic Table 2 dependency lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from web_agent.eval.table2.common import read_json
from web_agent.eval.table2.dependency_lock import (
    BROWSER_HOST_ROLE,
    DGX_HOST_ROLE,
    SINGLE_HOST_ROLE,
    read_and_validate_semantic_dependency_lock,
    validate_current_host_against_semantic_dependency_lock,
    write_semantic_dependency_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--deployment-preflight", type=Path, required=True)
    parser.add_argument("--deployment-topology", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--remeasure-host-role",
        choices=(SINGLE_HOST_ROLE, BROWSER_HOST_ROLE, DGX_HOST_ROLE),
        help=(
            "during validation, remeasure this physical host's Python/platform/"
            "package identity against its frozen topology role"
        ),
    )
    parser.add_argument(
        "--supplied-dgx-runtime-identity",
        type=Path,
        help=(
            "caller-supplied v2 DGX model-runtime identity to compare; this is "
            "not an independent runtime measurement"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = {
        "environment": read_json(args.environment),
        "deployment_preflight": read_json(args.deployment_preflight),
        "deployment_topology": args.deployment_topology,
    }
    if args.validate:
        lock = read_and_validate_semantic_dependency_lock(args.output, **kwargs)
        if args.supplied_dgx_runtime_identity and (
            args.remeasure_host_role != DGX_HOST_ROLE
        ):
            raise RuntimeError(
                "--supplied-dgx-runtime-identity requires --remeasure-host-role dgx_host"
            )
        if args.remeasure_host_role:
            supplied_runtime = (
                read_json(args.supplied_dgx_runtime_identity)
                if args.supplied_dgx_runtime_identity is not None
                else None
            )
            validate_current_host_against_semantic_dependency_lock(
                lock,
                host_role=args.remeasure_host_role,
                supplied_runtime_identity=supplied_runtime,
            )
    else:
        if args.remeasure_host_role or args.supplied_dgx_runtime_identity:
            raise RuntimeError(
                "--remeasure-host-role/--supplied-dgx-runtime-identity require --validate"
            )
        if args.output.exists():
            raise RuntimeError(f"refusing to overwrite dependency lock: {args.output}")
        write_semantic_dependency_lock(args.output, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
