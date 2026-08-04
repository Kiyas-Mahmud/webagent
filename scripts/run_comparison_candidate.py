"""Run one reviewed-Gold model candidate with fail-closed staged evidence.

This is the process launched by one assigned cell on one lab PC.  It keeps each
model in an isolated directory, runs the cheap model/data compatibility gate,
requires a controlled 5k mini for new backbones, and only then starts or resumes
the full 24,107-row run.  The locked test split is never opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

from web_agent.config import load_config


@dataclass(frozen=True)
class Candidate:
    model_id: str
    config: str
    label: str
    accepted_prior_mini: bool
    prior_mini_evidence: str


CANDIDATES = {
    candidate.model_id: candidate
    for candidate in (
        Candidate(
            model_id="qwen2vl_2b_gold_v2_8_dgx",
            config="configs/backbones/qwen2vl_2b_gold_v2_8_dgx.yaml",
            label="Qwen2-VL-2B",
            accepted_prior_mini=True,
            prior_mini_evidence=(
                "Reviewed v2.8 mini epochs 0-3; epoch 3 passed all eight "
                "registered quality gates. Epoch 4 was compute-interrupted."
            ),
        ),
        Candidate(
            model_id="qwen25vl_7b_gold_v2_8_dgx",
            config="configs/backbones/qwen25vl_7b_gold_v2_8_dgx.yaml",
            label="Qwen2.5-VL-7B",
            accepted_prior_mini=False,
            prior_mini_evidence="none; this backbone must pass its own mini",
        ),
        Candidate(
            model_id="internvl35_8b_gold_v2_8_dgx",
            config="configs/backbones/internvl35_8b_gold_v2_8_dgx.yaml",
            label="InternVL3.5-8B-HF",
            accepted_prior_mini=False,
            prior_mini_evidence="none; this backbone must pass its own mini",
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("auto", "mini", "full"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mini-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def read_pass(path: Path, *, stage: str) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        return False
    if int(report.get("test_rows_read", -1)) != 0:
        raise AssertionError(f"{stage} report accessed locked test rows: {path}")
    return True


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    candidate = CANDIDATES[args.model_id]
    cfg = load_config(candidate.config)
    if args.epochs <= 0 or args.mini_epochs <= 0:
        raise ValueError("epoch counts must be positive")
    if args.epochs != int(cfg["train"]["epochs"]):
        raise ValueError(
            f"--epochs must match the registered config ({cfg['train']['epochs']})"
        )
    if cfg["optim"]["batch_size"] * cfg["optim"]["grad_accum"] != 32:
        raise AssertionError("every candidate must preserve effective batch size 32")
    if cfg["train"]["quality_selection_rule"] != "all_gates_then_outcome_mcc":
        raise AssertionError("candidate changed the registered selection rule")
    if not cfg["data"].get("causal_routing"):
        raise AssertionError("candidate disabled causal pre/post routing")
    if not cfg["data"].get("recovery_transitions"):
        raise AssertionError("candidate disabled causal recovery transitions")

    run_root = (
        args.workspace_root / "outputs" / "model_comparison"
        / args.model_id / f"seed_{args.seed}"
    )
    mini_root = run_root / "mini"
    full_root = run_root / "full"
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = run_root / "run_contract.json"
    contract = {
        "candidate": asdict(candidate),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_name": cfg["name"],
        "model_revision": cfg["backbone"].get("revision", "unfixed legacy model"),
        "seed": args.seed,
        "maximum_full_epochs": args.epochs,
        "controlled_mini_epochs": args.mini_epochs,
        "effective_batch_size": 32,
        "checkpoint_selection": "all_gates_then_outcome_mcc",
        "checkpoint_selection_source": "original_gold_validation_only",
        "supplement_validation_selects_checkpoint": False,
        "train_rows": 24_107,
        "original_validation_rows": 7_861,
        "supplement_validation_rows": 194,
        "test_rows_read": 0,
    }
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise AssertionError("existing run contract differs; use a new run directory")
    else:
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    common = [
        sys.executable,
        "scripts/run_gold.py",
        "--config", candidate.config,
        "--data-root", str(args.data_root),
        "--supplement-root", str(args.supplement_root),
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
    ]

    smoke_report = run_root / "model_compatibility_report.json"
    if not read_pass(smoke_report, stage="compatibility"):
        if smoke_report.exists():
            raise AssertionError(
                "an existing compatibility report failed; preserve it and use "
                "a new code commit/run directory after fixing the model path"
            )
        run([
            *common,
            "--stage", "smoke",
            "--smoke-report-json", str(smoke_report),
        ])
        if not read_pass(smoke_report, stage="compatibility"):
            raise AssertionError("model compatibility gate did not pass")

    mini_report = mini_root / "report.json"
    mini_required = not candidate.accepted_prior_mini
    if mini_required and not read_pass(mini_report, stage="mini"):
        if mini_report.exists():
            raise AssertionError(
                "an existing controlled mini failed; do not overwrite its "
                "evidence—diagnose it before starting another run"
            )
        mini_root.mkdir(parents=True, exist_ok=True)
        run([
            *common,
            "--stage", "mini",
            "--train-rows", "5000",
            "--val-rows", "500",
            "--epochs", str(args.mini_epochs),
            "--checkpoint-root", str(mini_root / "checkpoints"),
            "--result-csv", str(mini_root / "epoch_metrics.csv"),
            "--diagnostics-json", str(mini_root / "diagnostics.json"),
            "--report-json", str(mini_report),
            "--source-validation-csv", str(mini_root / "source_validation.csv"),
        ])
        if not read_pass(mini_report, stage="mini"):
            raise AssertionError(
                "controlled mini failed; full training remains blocked"
            )

    if args.phase == "mini":
        print("Candidate mini evidence is complete; full stage was not requested.")
        return
    if args.phase == "full" and mini_required and not read_pass(
        mini_report, stage="mini"
    ):
        raise AssertionError("full phase requires this candidate's passed mini")

    full_root.mkdir(parents=True, exist_ok=True)
    full_report = full_root / "report.json"
    if read_pass(full_report, stage="full"):
        print("Full run already passed:", full_report)
        return
    model_checkpoint_dir = (
        full_root / "checkpoints" / f"{cfg['name']}_FULL_SEED{args.seed}"
    )
    last_checkpoint = model_checkpoint_dir / "last.ckpt"
    metrics_csv = full_root / "epoch_metrics.csv"
    if metrics_csv.exists() and not last_checkpoint.is_file():
        raise AssertionError(
            "full metrics exist without last.ckpt; restore the complete run "
            "directory or choose a new seed"
        )
    command = [
        *common,
        "--stage", "full",
        "--epochs", str(args.epochs),
        "--checkpoint-every-steps", str(cfg["train"]["checkpoint_every_steps"]),
        "--checkpoint-root", str(full_root / "checkpoints"),
        "--result-csv", str(metrics_csv),
        "--diagnostics-json", str(full_root / "diagnostics.json"),
        "--report-json", str(full_report),
        "--source-validation-csv", str(full_root / "source_validation.csv"),
    ]
    if last_checkpoint.is_file():
        command.extend(["--resume-checkpoint", str(last_checkpoint)])
        print("Resuming exact next batch from:", last_checkpoint, flush=True)
    run(command)
    if not read_pass(full_report, stage="full"):
        raise AssertionError(
            "full training completed but no epoch passed every registered gate"
        )
    print("FULL CANDIDATE PASSED:", full_report)


if __name__ == "__main__":
    main()
