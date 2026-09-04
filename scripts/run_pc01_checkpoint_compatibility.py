"""Run the real DGX CUDA compatibility gate for PC-01 Table 2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.common import sha256_file
from web_agent.eval.table2.pc01_checkpoint_compatibility import (
    run_pc01_checkpoint_compatibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-snapshot", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-contract", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--expected-source-commit",
        required=True,
        help="clean 40-character Git commit containing this gate/runtime",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_pc01_checkpoint_compatibility(
        export_dir=args.export_dir,
        checkpoint_path=args.checkpoint,
        base_snapshot_path=args.base_snapshot,
        report_path=args.report,
        run_contract_path=args.run_contract,
        repository_root=args.repository_root,
        expected_source_commit=args.expected_source_commit,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "PROVISIONAL_ENGINEERING_PILOT_COMPATIBILITY_ONLY",
                "paper_table_status": "N/R",
                "receipt": str(output.resolve()),
                "receipt_sha256": sha256_file(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
