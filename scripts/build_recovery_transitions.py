"""Audit recovery classes and export causal transition manifests without images.

Example (Kaggle):
  python scripts/build_recovery_transitions.py \
    --data-root /kaggle/input/datasets/kiyasmahmud/web-gold-40k \
    --output-dir /kaggle/working/recovery_transition_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from web_agent.data.recovery_transitions import transition_manifest


def resolve_data_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "split_train.json").is_file():
        return root
    candidates = sorted(root.rglob("split_train.json"))
    compatible = [
        candidate.parent for candidate in candidates
        if (candidate.parent / "split_val.json").is_file()
        and (candidate.parent / "split_test.json").is_file()
    ]
    if len(compatible) != 1:
        raise FileNotFoundError(
            f"expected exactly one gold split directory under {root}; found {compatible}"
        )
    return compatible[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--output-dir", default="/kaggle/working/recovery_transition_v2",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "schema_version": "recovery-transition-v2",
        "data_root": str(root),
        "splits": {},
    }
    for split in args.splits:
        source = root / f"split_{split}.json"
        records = json.loads(source.read_text(encoding="utf-8"))
        manifest = transition_manifest(records, split)
        combined["splits"][split] = {
            "transition_report": manifest["transition_report"],
            "class_audit": manifest["class_audit"],
        }
        if not args.audit_only:
            destination = output_dir / f"recovery_transitions_{split}.json"
            destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"wrote {destination} ({len(manifest['transitions'])} transitions)")

    report_path = output_dir / "recovery_class_audit.json"
    report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))
    print("audit saved:", report_path)


if __name__ == "__main__":
    main()
