"""Write train-only evidence for PC-01 action-value prompt semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.pc01_artifacts import (
    build_pc01_training_action_value_evidence,
    write_pc01_training_action_value_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-train", type=Path, required=True)
    parser.add_argument("--supplement-train", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = build_pc01_training_action_value_evidence(
        primary_train_path=args.primary_train,
        supplement_train_path=args.supplement_train,
        report_path=args.report,
        run_contract_path=args.run_contract,
    )
    output = write_pc01_training_action_value_evidence(args.output, evidence)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "combined_train_rows": evidence["combined_train_rows"],
                "action_value_mode": evidence["action_value_mode"],
                "validation_rows_read": evidence["validation_rows_read"],
                "test_rows_read": evidence["test_rows_read"],
                "locked_test_rows_read": evidence["locked_test_rows_read"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
