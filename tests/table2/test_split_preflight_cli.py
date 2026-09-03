from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

from web_agent.eval.table2.common import sha256_json
from web_agent.eval.table2.split_deployment_preflight import (
    SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION,
    SPLIT_BRIDGE_PROTOCOL_ID,
    SPLIT_BRIDGE_PROTOCOL_VERSION,
    build_dgx_model_runtime_identity,
)
from web_agent.eval.table2.webarena_preflight import (
    PINNED_WEBARENA_PACKAGES,
    run_webarena_host_preflight,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts/prepare_table2_split_preflight.py"
SERVICE_URL_MAP = {
    "WA_GITLAB": "http://gitlab.private.example",
    "WA_MAP": "http://map.private.example",
    "WA_REDDIT": "http://reddit.private.example",
    "WA_SHOPPING": "http://shopping.private.example",
    "WA_SHOPPING_ADMIN": "http://admin.private.example",
    "WA_WIKIPEDIA": "http://wikipedia.private.example",
    "WA_HOMEPAGE": "http://homepage.private.example",
}


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _local_pass() -> dict:
    return run_webarena_host_preflight(
        service_url_map=SERVICE_URL_MAP,
        version_getter=lambda name: PINNED_WEBARENA_PACKAGES[name],
        browser_probe=lambda: {
            "status": "PASS",
            "browser": "chromium",
            "browser_version": "fixture",
            "viewport": {"width": 1280, "height": 720},
            "device_scale_factor": 1,
            "screenshot_sha256": "a" * 64,
        },
        service_probe=lambda _url: {"status": "PASS", "http_status": 200},
        live_reset_probe=lambda index: {
            "status": "PASS",
            "task_index": index,
            "seed": 42,
            "goal_sha256": "b" * 64,
            "current_url_sha256": "c" * 64,
            "screenshot_shape": [720, 1280, 3],
            "observation_keys": ["goal", "screenshot", "url"],
            "action_taken": False,
            "reward_read": False,
            "evaluator_output_read": False,
        },
    )


def _inputs(root: Path) -> dict[str, Path]:
    local = _local_pass()
    dgx = build_dgx_model_runtime_identity(
        host_identity_sha256="d" * 64,
        runtime_identity={
            "checkpoint_sha256": "e" * 64,
            "processor_contract_sha256": "f" * 64,
        },
        runtime_source_set_sha256="1" * 64,
        runtime_environment_sha256="2" * 64,
    )
    sources = [{"relative_path": "bridge/server.py", "sha256": "3" * 64}]
    bridge = {
        "schema_version": SPLIT_BRIDGE_IDENTITY_SCHEMA_VERSION,
        "record_type": "OracleBlindInferenceBridgeIdentity",
        "bridge_id": "fixture-bridge",
        "bridge_version": "v1",
        "protocol_id": SPLIT_BRIDGE_PROTOCOL_ID,
        "protocol_version": SPLIT_BRIDGE_PROTOCOL_VERSION,
        "source_files": sources,
        "source_set_sha256": sha256_json(sources),
        "browser_host_identity_sha256": sha256_json(local["host"]),
        "browser_endpoint_identity_sha256": "4" * 64,
        "dgx_endpoint_identity_sha256": "5" * 64,
        "dgx_model_runtime_identity_sha256": sha256_json(dgx),
        "transport_identity_sha256": "6" * 64,
        "request_schema_sha256": "7" * 64,
        "response_schema_sha256": "8" * 64,
        "reward_fields_permitted": False,
        "oracle_fields_permitted": False,
        "evaluator_fields_permitted": False,
    }
    exchanges = [
        {
            "request": {"request_id": "preflight-0", "input_sha256": "9" * 64},
            "response": {"request_id": "preflight-0", "output_sha256": "a" * 64},
        }
    ]
    return {
        "service_map": _write_json(root / "service-map.json", SERVICE_URL_MAP),
        "local": _write_json(root / "local-pass.json", local),
        "dgx": _write_json(root / "input/dgx.json", dgx),
        "expected_dgx": _write_json(root / "expected/dgx.json", dgx),
        "bridge": _write_json(root / "input/bridge.json", bridge),
        "expected_bridge": _write_json(root / "expected/bridge.json", bridge),
        "exchanges": _write_json(root / "exchanges.json", exchanges),
    }


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{REPOSITORY_ROOT / 'src'}{os.pathsep}{current}"
        if current
        else str(REPOSITORY_ROOT / "src")
    )
    return environment


def _build_command(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "build",
        "--local-browser-preflight",
        str(paths["local"]),
        "--service-url-map",
        str(paths["service_map"]),
        "--dgx-runtime-identity",
        str(paths["dgx"]),
        "--expected-dgx-runtime-identity",
        str(paths["expected_dgx"]),
        "--bridge-identity",
        str(paths["bridge"]),
        "--expected-bridge-identity",
        str(paths["expected_bridge"]),
        "--exchanges",
        str(paths["exchanges"]),
        "--output",
        str(output),
    ]


def test_split_preflight_cli_help_exposes_build_and_validation_contracts() -> None:
    top = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    build = subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--help"],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert top.returncode == 0
    assert "build" in top.stdout and "validate" in top.stdout
    assert build.returncode == 0
    assert "--service-url-map" in build.stdout
    assert "--expected-dgx-runtime-identity" in build.stdout
    assert "--expected-bridge-identity" in build.stdout


def test_cli_builds_redacted_read_only_artifact_then_revalidates_it(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    output = tmp_path / "split-preflight.json"
    built = subprocess.run(
        _build_command(paths, output),
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert built.returncode == 0, built.stderr
    summary = json.loads(built.stdout)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["bridge_exchange_count"] == 1
    assert summary["forbidden_field_totals"] == {
        "reward": 0,
        "oracle": 0,
        "evaluator": 0,
    }
    assert artifact["campaign_eligible"] is True
    assert "preflight-0" not in output.read_text(encoding="utf-8")
    assert output.stat().st_mode & stat.S_IWUSR == 0

    validated = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate",
            "--artifact",
            str(output),
            "--service-url-map",
            str(paths["service_map"]),
            "--expected-dgx-runtime-identity",
            str(paths["expected_dgx"]),
            "--expected-bridge-identity",
            str(paths["expected_bridge"]),
        ],
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["artifact_content_sha256"] == sha256_json(
        artifact
    )


def test_cli_refuses_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    output = tmp_path / "split-preflight.json"
    output.write_text("do-not-overwrite\n", encoding="utf-8")

    result = subprocess.run(
        _build_command(paths, output),
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert output.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_cli_requires_expected_identity_files_separate_from_inputs(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    paths["expected_dgx"] = paths["dgx"]
    output = tmp_path / "split-preflight.json"

    result = subprocess.run(
        _build_command(paths, output),
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must be separate JSON files" in result.stderr
    assert not output.exists()


def test_cli_rejects_expected_identity_drift_before_writing(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    expected = json.loads(paths["expected_dgx"].read_text(encoding="utf-8"))
    expected["runtime_environment_sha256"] = "0" * 64
    _write_json(paths["expected_dgx"], expected)
    output = tmp_path / "split-preflight.json"

    result = subprocess.run(
        _build_command(paths, output),
        cwd=REPOSITORY_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "frozen expectation" in result.stderr
    assert not output.exists()
