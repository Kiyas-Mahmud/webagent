"""Preflight a cross-machine resume of a reviewed-Gold comparison candidate.

Run this on the NEW machine, from the repository root, before relaunching the
candidate cell in ``notebooks/dgx_three_model_comparison.ipynb``::

    PYTHONPATH=src .venv/bin/python scripts/verify_resume.py \
        --model-id qwen25vl_7b_gold_v2_8_dgx

It reports every condition that would make the run restart from scratch or
fail closed instead of resuming at the exact next physical batch.  It reads
only the run's own evidence and the checkpoint; it never opens the locked test
split and never writes to the run directory.

The checks that matter, and why they are not obvious:

* ``training_signature`` excludes only the *top-level* ``data`` keys
  ``num_workers`` and ``root``.  ``data.recovery_supplement.root`` is nested, so
  its absolute path is part of the signature and must be identical on the new
  machine.
* ``run_comparison_candidate.py`` compares the whole ``run_contract.json``,
  including ``git_commit``, so the checkout must sit on the contract's commit.
* ``gold_full.py`` renames the config to ``{name}_FULL_SEED{seed}`` before
  training, so a signature check must apply the same rename or it reports a
  false ``name`` mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

DEFAULT_REPO = Path("/home/aiub/kiyas/webagent")
DEFAULT_WORKSPACE = Path("/home/aiub/kiyas/webagent_comparison")
DEFAULT_DATA = Path("/home/aiub/kiyas/webagent_full/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="qwen25vl_7b_gold_v2_8_dgx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


class Report:
    """Collect pass/fail lines so every check runs before the exit code."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, message: str, detail: str = "") -> bool:
        print(("  OK   " if ok else "  FAIL ") + message + (f"  [{detail}]" if detail else ""))
        if not ok:
            self.failures.append(message)
        return ok


def main() -> int:
    args = parse_args()
    report = Report()
    run_root = (
        args.workspace_root / "outputs" / "model_comparison"
        / args.model_id / f"seed_{args.seed}"
    )

    print("== 1. the exact absolute paths exist ==")
    for path in (
        args.repo_root / ".git",
        args.data_home / "original",
        args.data_home / "supplement",
        run_root,
    ):
        report.check(path.exists(), str(path))
    if report.failures:
        print("\nRESUME PREFLIGHT: FAIL (paths missing; nothing else can be checked)")
        return 1

    print("== 2. git commit matches the frozen run contract ==")
    contract = json.loads((run_root / "run_contract.json").read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    report.check(
        head == contract["git_commit"],
        "HEAD == run_contract.git_commit",
        f"HEAD={head[:9]} contract={contract['git_commit'][:9]}",
    )

    print("== 3. prior gate evidence present and PASS (else smoke/mini re-run) ==")
    for name, path in (
        ("compatibility", run_root / "model_compatibility_report.json"),
        ("mini", run_root / "mini" / "report.json"),
    ):
        passed = (
            path.is_file()
            and json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
        )
        report.check(passed, f"{name} report PASS", str(path))
    report.check(
        (run_root / "full" / "epoch_metrics.csv").is_file(),
        "full/epoch_metrics.csv copied alongside last.ckpt",
    )

    print("== 4. supplement root resolves to the string stored in the checkpoint ==")
    sys.path.insert(0, str(args.repo_root / "src"))
    import torch

    from web_agent.config import load_config
    from web_agent.data.recovery_supplement import resolve_supplement_root
    from web_agent.train.resume import training_signature

    supplement = str(resolve_supplement_root(args.data_home / "supplement"))
    checkpoint_path = (
        run_root / "full" / "checkpoints"
        / f"{contract['config_name']}_FULL_SEED{args.seed}" / "last.ckpt"
    )
    if not report.check(checkpoint_path.is_file(), "last.ckpt present", str(checkpoint_path)):
        print("\nRESUME PREFLIGHT: FAIL (no checkpoint to resume)")
        return 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_supplement = checkpoint["config"]["data"]["recovery_supplement"]["root"]
    report.check(
        supplement == saved_supplement,
        "resolved supplement root == checkpoint value",
        f"new={supplement} saved={saved_supplement}",
    )

    print("== 5. full training signature matches (this is the real resume gate) ==")
    cfg = load_config(f"configs/backbones/{args.model_id}.yaml")
    # gold_full.py renames the config for the full stage; mirror it exactly.
    cfg["name"] = f"{cfg['name']}_FULL_SEED{args.seed}"
    cfg["data"]["root"] = str(
        next((args.data_home / "original").rglob("split_train.json")).parent
    )
    cfg["data"].setdefault("recovery_supplement", {}).update(
        {"enabled": True, "root": supplement, "include_in_primary_validation": False}
    )
    saved_signature = training_signature(checkpoint["config"])
    expected_signature = training_signature(cfg)
    differing = [
        key for key in expected_signature
        if saved_signature.get(key) != expected_signature.get(key)
    ]
    report.check(not differing, "training_signature identical", f"differing={differing}")

    print("== 6. resume position and runtime ==")
    early = checkpoint["early_stop_state"]
    print(
        f"  epoch={checkpoint['epoch']} epoch_complete={checkpoint['epoch_complete']} "
        f"next_batch={checkpoint['batch_in_epoch']} global_step={checkpoint['step']} "
        f"bad_epochs={early['bad_epochs']} best_metric={early['best_metric']:.4f}"
    )
    report.check(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA available and bf16 supported (the 7B/8B profiles require bf16)",
    )

    print()
    if report.failures:
        print(f"RESUME PREFLIGHT: FAIL ({len(report.failures)})")
        return 1
    print("RESUME PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
