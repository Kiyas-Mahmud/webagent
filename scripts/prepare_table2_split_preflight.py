"""Build or validate a frozen Table 2 split-deployment preflight artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from web_agent.eval.table2.common import (
    SchemaError,
    atomic_write_json,
    read_json,
    sha256_json,
)
from web_agent.eval.table2.split_deployment_preflight import (
    build_split_deployment_preflight,
    validate_split_deployment_preflight,
)
from web_agent.eval.table2.public_task_registry import (
    load_public_development_task_registry,
)
from web_agent.eval.table2.webarena_preflight import load_service_url_map


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _authority_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--service-url-map",
        type=Path,
        required=True,
        help="JSON mapping of all seven BrowserGym WA_* service origins",
    )
    parser.add_argument(
        "--expected-dgx-runtime-identity",
        type=Path,
        required=True,
        help="separate frozen expected DGX runtime-identity JSON",
    )
    parser.add_argument(
        "--expected-bridge-identity",
        type=Path,
        required=True,
        help="separate frozen expected oracle-blind bridge-identity JSON",
    )
    parser.add_argument(
        "--task-registry",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/table2/pilot/task_manifest.json",
        help="tracked ordered 50-task public-development registry",
    )
    parser.add_argument(
        "--live-reset-task-index",
        type=int,
        help="must equal the first upstream index in the tracked task registry",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="build a redacted artifact from explicit measured JSON inputs",
    )
    build.add_argument("--local-browser-preflight", type=Path, required=True)
    build.add_argument("--dgx-runtime-identity", type=Path, required=True)
    build.add_argument("--bridge-identity", type=Path, required=True)
    build.add_argument(
        "--exchanges",
        type=Path,
        required=True,
        help=(
            "JSON array of real bridge request/response objects; "
            "payloads are not emitted"
        ),
    )
    build.add_argument("--output", type=Path, required=True)
    _authority_arguments(build)

    validate = commands.add_parser(
        "validate",
        help="revalidate an existing artifact against separate frozen authorities",
    )
    validate.add_argument("--artifact", type=Path, required=True)
    _authority_arguments(validate)
    return parser.parse_args(argv)


def _read_exchange_array(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read bridge exchanges JSON: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise SchemaError("bridge exchanges JSON must be an array of objects")
    return value


def _require_separate_paths(actual: Path, expected: Path, *, label: str) -> None:
    if actual.resolve() == expected.resolve():
        raise SchemaError(
            f"{label} input and expected authority must be separate JSON files"
        )


def _summary(value: dict[str, Any], *, artifact: Path) -> dict[str, Any]:
    transcript = value["bridge_transcript"]
    return {
        "status": value["status"],
        "campaign_eligible": value["campaign_eligible"],
        "deployment_topology": value["deployment_topology"],
        "evidence_label": value["evidence_label"],
        "paper_table_status": value["paper_table_status"],
        "artifact": str(artifact.resolve()),
        "artifact_content_sha256": sha256_json(value),
        "local_browser_preflight_sha256": value[
            "local_browser_preflight_sha256"
        ],
        "dgx_model_runtime_identity_sha256": value[
            "dgx_model_runtime_identity_sha256"
        ],
        "bridge_identity_sha256": value["bridge_identity_sha256"],
        "bridge_transcript_sha256": value["bridge_transcript_sha256"],
        "bridge_exchange_count": transcript["entry_count"],
        "forbidden_field_totals": transcript["forbidden_field_totals"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build" and args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite split preflight evidence: {args.output}"
        )
    service_urls = load_service_url_map(args.service_url_map)
    task_registry = load_public_development_task_registry(args.task_registry)
    first_registered_index = task_registry.ordered_upstream_indices[0]
    live_reset_task_index = (
        first_registered_index
        if args.live_reset_task_index is None
        else args.live_reset_task_index
    )
    if live_reset_task_index != first_registered_index:
        raise SchemaError(
            "split preflight must bind the first task in exact tracked registry order"
        )
    expected_dgx = read_json(args.expected_dgx_runtime_identity)
    expected_bridge = read_json(args.expected_bridge_identity)

    if args.command == "build":
        _require_separate_paths(
            args.dgx_runtime_identity,
            args.expected_dgx_runtime_identity,
            label="DGX runtime identity",
        )
        _require_separate_paths(
            args.bridge_identity,
            args.expected_bridge_identity,
            label="bridge identity",
        )
        value = build_split_deployment_preflight(
            local_browser_preflight=read_json(args.local_browser_preflight),
            service_url_map=service_urls,
            dgx_model_runtime_identity=read_json(args.dgx_runtime_identity),
            bridge_identity=read_json(args.bridge_identity),
            exchanges=_read_exchange_array(args.exchanges),
            expected_live_reset_task_index=live_reset_task_index,
        )
        value = validate_split_deployment_preflight(
            value,
            service_url_map=service_urls,
            expected_live_reset_task_index=live_reset_task_index,
            expected_dgx_model_runtime_identity=expected_dgx,
            expected_bridge_identity=expected_bridge,
        )
        atomic_write_json(args.output, value, mode=0o444)
        artifact = args.output
    else:
        _require_separate_paths(
            args.artifact,
            args.expected_dgx_runtime_identity,
            label="split artifact and DGX authority",
        )
        _require_separate_paths(
            args.artifact,
            args.expected_bridge_identity,
            label="split artifact and bridge authority",
        )
        artifact = args.artifact
        value = validate_split_deployment_preflight(
            read_json(artifact),
            service_url_map=service_urls,
            expected_live_reset_task_index=live_reset_task_index,
            expected_dgx_model_runtime_identity=expected_dgx,
            expected_bridge_identity=expected_bridge,
        )
    print(json.dumps(_summary(value, artifact=artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
