"""Link the dataset into ./data and sanity-check it (run once after clone).

Local (Windows/Linux): make ./data point at ../FinalData (sibling of code/).
Kaggle: pass --source /kaggle/input/<dataset-name> instead.

Verifies the 3 split JSONs + images/ exist, prints record counts, and confirms
UTF-8 loads. Does NOT modify the dataset. Idempotent.

    python scripts/setup_data.py
    python scripts/setup_data.py --source /kaggle/input/web-agent-data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
# Kaggle is where training runs; local ../FinalData is a review-only copy.
KAGGLE_SOURCE = Path("/kaggle/input/datasets/kiyasmahmud/thesisdata/FinalData")
LOCAL_SOURCE = CODE_ROOT.parent / "FinalData"          # e:/University/thesis/FinalData
DEFAULT_SOURCE = KAGGLE_SOURCE if KAGGLE_SOURCE.exists() else LOCAL_SOURCE
LINK = CODE_ROOT / "data"                              # ./data
REQUIRED = ("split_train.json", "split_val.json", "split_test.json", "images")


def _check_source(source: Path) -> None:
    missing = [name for name in REQUIRED if not (source / name).exists()]
    if missing:
        raise SystemExit(f"Source {source} missing: {missing}")


def _make_link(source: Path) -> str:
    """Point ./data at source. Returns a human description of what happened."""
    if LINK.exists() or LINK.is_symlink():
        # Already linked/created — leave it (idempotent).
        return f"./data already exists -> {os.path.realpath(LINK)}"
    try:
        os.symlink(source, LINK, target_is_directory=True)
        return f"symlinked ./data -> {source}"
    except (OSError, NotImplementedError) as e:
        # Windows without Developer Mode/admin: symlink denied. Fall back to a note.
        return (f"could NOT create symlink ({e.__class__.__name__}). "
                f"Set data.root to an absolute path in your config, e.g.\n"
                f"    data:\n      root: {source.as_posix()}")


def _counts(source: Path) -> None:
    for name in ("split_train.json", "split_val.json", "split_test.json"):
        with open(source / name, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {name:20} {len(data):>7,} records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="dataset folder to link")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    print(f"Source: {source}")
    _check_source(source)
    print(_make_link(source))
    print("Record counts:")
    _counts(source)
    print("OK. Next: python -m web_agent.data.verify_dataset --data data")


if __name__ == "__main__":
    main()
