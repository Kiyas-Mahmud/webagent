"""Authenticate and export PC-01 artifacts for the Table 2 handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.pc01_artifacts import (
    export_pc01_artifacts,
    export_pc01_checkpoint_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--processor-source",
        type=Path,
        help="materialized local pinned Qwen2-VL base snapshot (required for full export)",
    )
    parser.add_argument(
        "--training-environment",
        type=Path,
        help="checked-in PC-01 training environment.json (required for full export)",
    )
    parser.add_argument(
        "--training-action-value-evidence",
        type=Path,
        help="canonical train-only action-value audit receipt (required for full export)",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="export authenticated checkpoint config without requiring processor/base weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_only:
        if args.processor_source is not None:
            raise ValueError("--processor-source is not used with --config-only")
        if args.training_environment is not None:
            raise ValueError("--training-environment is not used with --config-only")
        if args.training_action_value_evidence is not None:
            raise ValueError(
                "--training-action-value-evidence is not used with --config-only"
            )
        result = export_pc01_checkpoint_identity(
            checkpoint_path=args.checkpoint,
            report_path=args.report,
            run_contract_path=args.run_contract,
            output_dir=args.output_dir,
        )
    else:
        if args.processor_source is None:
            raise ValueError("full PC-01 export requires --processor-source")
        if args.training_environment is None:
            raise ValueError("full PC-01 export requires --training-environment")
        if args.training_action_value_evidence is None:
            raise ValueError(
                "full PC-01 export requires --training-action-value-evidence"
            )
        result = export_pc01_artifacts(
            checkpoint_path=args.checkpoint,
            report_path=args.report,
            run_contract_path=args.run_contract,
            processor_source=args.processor_source,
            training_environment_path=args.training_environment,
            training_action_value_evidence_path=(
                args.training_action_value_evidence
            ),
            output_dir=args.output_dir,
        )
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "checkpoint_sha256": result.checkpoint_sha256,
        "config_sha256": result.config_sha256,
        "mode": "config_only" if args.config_only else "full",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
