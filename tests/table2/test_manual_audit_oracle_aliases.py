from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.eval.table2.common import SchemaError
from web_agent.eval.table2.summary import _write_manual_audit_reviewer_packet


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _packet_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = (tmp_path / "alias-packet").resolve()
    _write_json(
        root / "campaign_manifest.json",
        {"campaign_id": "packet-campaign", "campaign_mode": "evaluation"},
    )
    _write_json(
        root / "frozen/task_manifest.json",
        {
            "tasks": [
                {
                    "task_id": "webarena.1",
                    "instruction": "Perform the observable task.",
                    "start_state": {
                        "sites": ["site"],
                        "start_url": "https://site.example/",
                        "require_login": False,
                        "storage_state": None,
                        "geolocation": None,
                        "require_reset": False,
                    },
                }
            ]
        },
    )
    runtime = root / "paired_blocks/block/rerun_0/E1/runtime"
    _write_json(runtime / "episode_summary.json", {"episode_id": "episode-1"})
    reviewer_root = root / "manual_audit/reviewer_packets"
    reviewer_root.mkdir(parents=True)
    return root, runtime, reviewer_root


@pytest.mark.parametrize(
    ("filename", "payload"),
    (
        (
            "nested.json",
            {"visible": [{"metadata": {"Official-Task.Success": True}}]},
        ),
        (
            "nested.jsonl",
            {"visible": [{"metadata": {"postHoc-Relevance.Label": ["m-1"]}}]},
        ),
    ),
)
def test_manual_audit_packet_rejects_nested_normalized_oracle_aliases(
    tmp_path: Path,
    filename: str,
    payload: object,
) -> None:
    root, runtime, reviewer_root = _packet_fixture(tmp_path)
    path = runtime / filename
    if path.suffix == ".jsonl":
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        _write_json(path, payload)

    with pytest.raises(SchemaError, match="leaks sealed verifier keys"):
        _write_manual_audit_reviewer_packet(
            root=root,
            reviewer_packet_root=reviewer_root,
            campaign_id="packet-campaign",
            audit_id="audit-01",
            primary_source=runtime,
            primary_episode_id="episode-1",
            primary_system_id="E1",
            task_id="webarena.1",
            paired_e2_source=None,
            paired_e2_episode_id=None,
        )
