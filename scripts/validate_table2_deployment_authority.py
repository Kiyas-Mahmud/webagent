"""Issue or consume Table 2 deployment challenges using the pinned registry.

The tool verifies public-key signatures only. It has no signing operation and
accepts neither alternate registry bytes nor key material. The local ledger can
serialize use within one file, but copying or restoring that file defeats global
replay detection; dispatch therefore requires a separately maintained external
anchor that this phase does not implement. Because the current source-pinned
registry is empty, issue and consume commands intentionally fail until an
independent lab authority is registered by a reviewed source change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.deployment_authority import (
    AUTHORITY_PHASES,
    BLOCK_CLOSE_PHASE,
    BLOCK_OPEN_PHASE,
    EMPTY_REGISTRY_STATUS,
    LOCAL_LEDGER_CLAIM_SCOPE,
    PRODUCTION_AUTHORITY_REGISTRY_FILE_SHA256,
    TRANSCRIPT_CHANNELS,
    consume_production_receipt,
    issue_production_challenge,
    load_production_authority_registry,
    read_strict_json,
    read_runtime_transcript_jsonl,
    reconstruct_transcript_root,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "registry-status",
        help="show the fixed registry status without authorizing execution",
    )

    issue = commands.add_parser(
        "issue-challenge",
        help="append one random challenge for a key already in the fixed registry",
    )
    issue.add_argument("--ledger", required=True, type=Path)
    issue.add_argument("--authority-id", required=True)
    issue.add_argument("--key-id", required=True)
    issue.add_argument("--phase", required=True, choices=AUTHORITY_PHASES)
    issue.add_argument("--campaign-id", required=True)
    issue.add_argument("--handoff-manifest-sha256", required=True)
    issue.add_argument("--runner-attestation-sha256", required=True)
    issue.add_argument("--semantic-dependency-lock-sha256", required=True)
    issue.add_argument("--block-id")
    issue.add_argument("--rerun-id", type=int)
    issue.add_argument("--previous-receipt-sha256")
    issue.add_argument("--ttl-seconds", type=int, default=300)

    consume = commands.add_parser(
        "consume-receipt",
        help="verify and atomically consume one signed receipt",
    )
    consume.add_argument("--ledger", required=True, type=Path)
    consume.add_argument("--challenge-sha256", required=True)
    consume.add_argument("--receipt", required=True, type=Path)

    transcript = commands.add_parser(
        "reconstruct-transcript",
        help="reconstruct a local transcript root without claiming provenance",
    )
    transcript.add_argument("--transcript-jsonl", required=True, type=Path)
    transcript.add_argument("--campaign-id", required=True)
    transcript.add_argument("--block-id", required=True)
    transcript.add_argument("--rerun-id", required=True, type=int)
    transcript.add_argument("--dgx-service-session-id", required=True)
    transcript.add_argument(
        "--channel", required=True, choices=sorted(TRANSCRIPT_CHANNELS)
    )
    return parser.parse_args(argv)


def _read_transcript(path: Path) -> list[dict[str, Any]]:
    return read_runtime_transcript_jsonl(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "registry-status":
        registry = load_production_authority_registry()
        return {
            "status": registry["status"],
            "registry_id": registry["registry_id"],
            "registry_file_sha256": PRODUCTION_AUTHORITY_REGISTRY_FILE_SHA256,
            "registered_authority_count": len(registry["authorities"]),
            "production_dispatch_authorized": False,
            "reason": (
                "NO_INDEPENDENT_AUTHORITY_REGISTERED"
                if registry["status"] == EMPTY_REGISTRY_STATUS
                else "SIGNED_RUNTIME_RECEIPTS_STILL_REQUIRED"
            ),
        }
    if args.command == "issue-challenge":
        block_phase = args.phase in {BLOCK_OPEN_PHASE, BLOCK_CLOSE_PHASE}
        if block_phase != (args.block_id is not None and args.rerun_id is not None):
            raise SchemaError(
                "block ID and rerun ID are required only for physical-block phases"
            )
        return issue_production_challenge(
            args.ledger,
            authority_id=args.authority_id,
            key_id=args.key_id,
            phase=args.phase,
            campaign_id=args.campaign_id,
            handoff_manifest_sha256=args.handoff_manifest_sha256,
            runner_attestation_sha256=args.runner_attestation_sha256,
            semantic_dependency_lock_sha256=(
                args.semantic_dependency_lock_sha256
            ),
            block_id=args.block_id,
            rerun_id=args.rerun_id,
            previous_receipt_sha256=args.previous_receipt_sha256,
            ttl_seconds=args.ttl_seconds,
        )
    if args.command == "consume-receipt":
        envelope = read_strict_json(args.receipt)
        verified = consume_production_receipt(
            args.ledger,
            challenge_sha256=args.challenge_sha256,
            envelope_value=envelope,
        )
        return {
            "status": "AUTHENTICATED_AND_LOCALLY_CONSUMED_NOT_DISPATCH_AUTHORITY",
            "claim_scope": LOCAL_LEDGER_CLAIM_SCOPE,
            "external_global_replay_anchor_present": False,
            "production_dispatch_authorized": False,
            "receipt": verified.to_dict(),
        }
    if args.command == "reconstruct-transcript":
        return reconstruct_transcript_root(
            _read_transcript(args.transcript_jsonl),
            campaign_id=args.campaign_id,
            block_id=args.block_id,
            rerun_id=args.rerun_id,
            dgx_service_session_id=args.dgx_service_session_id,
            expected_channel=args.channel,
        )
    raise SchemaError("deployment-authority command is unregistered")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, SchemaError, ValueError) as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(error)},
                sort_keys=True,
                allow_nan=False,
            )
        )
        raise SystemExit(1) from error
