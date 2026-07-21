"""Replay the v2.7 checkpoint selector on a completed v2.6 mini report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from web_agent.train.selection import replay_selection_report  # noqa: E402
from web_agent.utils.results import save_mini_result_csv  # noqa: E402


def _report_from_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if not isinstance(text, str):
                continue
            for offset, character in enumerate(text):
                if character != "{":
                    continue
                try:
                    payload, _ = decoder.raw_decode(text[offset:])
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(payload, dict)
                    and isinstance(payload.get("history"), list)
                    and isinstance(payload.get("epoch_checkpoints"), dict)
                    and payload.get("train_rows") == 5_000
                    and payload.get("val_rows") == 500
                ):
                    candidates.append(payload)
    if not candidates:
        raise ValueError(f"no completed 5k mini report found in notebook: {path}")
    return candidates[-1]


def load_source_report(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".ipynb":
        return _report_from_notebook(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source report must contain a JSON object: {path}")
    return payload


def _resolve_checkpoint(
    recorded_path: str,
    *,
    source_path: Path,
    checkpoint_root: Path | None,
) -> Path | None:
    recorded = Path(recorded_path)
    direct_candidates = [recorded, source_path.parent / recorded]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()

    roots = [root for root in (checkpoint_root, source_path.parent) if root]
    matches: list[Path] = []
    for root in roots:
        if root.is_dir():
            matches.extend(path.resolve() for path in root.rglob(recorded.name))
    unique_matches = sorted(set(matches))
    if len(unique_matches) > 1:
        raise ValueError(
            f"multiple files match selected checkpoint {recorded.name}: {unique_matches}"
        )
    return unique_matches[0] if unique_matches else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()

    source_report = load_source_report(args.source)
    replay = replay_selection_report(source_report)
    selected_epoch = int(replay["selected_epoch"])
    selected_checkpoint = str(replay["best_checkpoint"])
    checkpoint_file = _resolve_checkpoint(
        selected_checkpoint,
        source_path=args.source,
        checkpoint_root=args.checkpoint_root,
    )
    expected_marker = f"best_e{selected_epoch}_"
    filename_matches_epoch = expected_marker in Path(selected_checkpoint).name
    if checkpoint_file is not None:
        filename_matches_epoch = filename_matches_epoch and (
            expected_marker in checkpoint_file.name
        )
    replay["selected_checkpoint_artifact"] = {
        "status": "VERIFIED" if checkpoint_file is not None else "NOT_MOUNTED",
        "recorded_path": selected_checkpoint,
        "resolved_path": str(checkpoint_file) if checkpoint_file else None,
        "filename_matches_selected_epoch": filename_matches_epoch,
        "size_bytes": checkpoint_file.stat().st_size if checkpoint_file else None,
        "sha256": _sha256(checkpoint_file) if checkpoint_file else None,
        "prediction_roundtrip_rerun": False,
    }
    if not filename_matches_epoch:
        raise ValueError(
            "selected checkpoint filename does not identify the selected epoch: "
            f"epoch={selected_epoch}, checkpoint={selected_checkpoint}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    save_mini_result_csv(replay, args.csv)
    print(
        json.dumps(
            {
                "status": replay["quality_gates"]["status"],
                "selection_rule": replay["selection_rule"],
                "eligible_epochs": replay["quality_gates"]["eligible_epochs"],
                "selected_epoch": selected_epoch,
                "selected_outcome_mcc": replay["best_metric"],
                "selected_checkpoint": selected_checkpoint,
                "checkpoint_artifact": replay["selected_checkpoint_artifact"]["status"],
                "test_rows_read": replay["test_rows_read"],
                "report": str(args.report),
                "csv": str(args.csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
