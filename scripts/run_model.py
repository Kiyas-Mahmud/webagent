"""Unified CLI entry — dispatch any model to any stage (test-first workflow).

    python scripts/run_model.py --config configs/backbones/y1_siglip_roberta.yaml --stage smoke
    stages: smoke | mini | full | eval

The notebook (notebooks/kaggle_train.ipynb) calls the same dispatch so logic
stays in the package, not in cells.
"""

from __future__ import annotations

import argparse

from web_agent.config import load_config
from web_agent.utils.seed import set_seed

STAGES = ("smoke", "mini", "full", "eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a model YAML")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--checkpoint", default=None, help="ckpt path for --stage eval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seeds", [42])[0])

    if args.stage == "smoke":
        from web_agent.train import smoke_test
        smoke_test.run(cfg)
    elif args.stage == "mini":
        from web_agent.train import mini_train
        mini_train.run(cfg)
    elif args.stage == "full":
        from web_agent.train import train
        train.run(cfg)
    elif args.stage == "eval":
        from web_agent.eval import evaluate
        evaluate.run(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
