"""Kaggle code-file bootstrap for the Table 2 P4_PREPARE_ONLY runner."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _repository_root(arguments: list[str]) -> Path:
    if "--repository-root" in arguments:
        index = arguments.index("--repository-root")
        if index + 1 >= len(arguments):
            raise SystemExit("--repository-root requires a value")
        return Path(arguments[index + 1]).resolve()
    configured = os.environ.get("TABLE2_REPOSITORY_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2]


arguments = sys.argv[1:]
repository = _repository_root(arguments)
source_root = repository / "src"
if not (source_root / "web_agent" / "memory" / "kaggle_prepare_only.py").is_file():
    raise SystemExit(
        "Exact Table 2 repository source is unavailable. Supply "
        "--repository-root or TABLE2_REPOSITORY_ROOT; no source is downloaded."
    )
sys.path.insert(0, str(source_root))

from web_agent.memory.kaggle_prepare_only import main  # noqa: E402


if "--repository-root" not in arguments:
    arguments = ["--repository-root", str(repository), *arguments]
raise SystemExit(main(arguments))
