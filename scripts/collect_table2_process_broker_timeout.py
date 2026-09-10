"""Collect one frozen, non-authorizing Table 2 process-broker timeout block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.eval.table2.process_broker_timeout_collector import (
    collect_from_frozen_timeout_collector_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collector-input",
        required=True,
        type=Path,
        help="absolute path to the read-only externally frozen collector input",
    )
    parser.add_argument(
        "--collector-input-sha256",
        required=True,
        help="externally supplied canonical JSON SHA-256 for --collector-input",
    )
    parser.add_argument(
        "--safe-probe-manifest",
        required=True,
        type=Path,
        help="absolute path to the read-only approved safe-probe manifest",
    )
    parser.add_argument(
        "--harness-source-receipt",
        required=True,
        type=Path,
        help="absolute path to the read-only measurement-harness source receipt",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        type=Path,
        help="new absolute path; existing paths are always refused",
    )
    parser.add_argument(
        "--repository-root",
        required=True,
        type=Path,
        help="absolute source root bound by the harness source receipt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_from_frozen_timeout_collector_input(
        repository_root=args.repository_root,
        output_directory=args.output_directory,
        collector_input_path=args.collector_input,
        expected_collector_input_sha256=args.collector_input_sha256,
        safe_probe_manifest_path=args.safe_probe_manifest,
        measurement_harness_source_receipt_path=args.harness_source_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
