"""Gold-dataset runner (separate from the synthetic notebook/pipeline).

    python scripts/run_gold.py --stage smoke
    python scripts/run_gold.py --stage train --data-root /kaggle/input/.../gold --epochs 5
    python scripts/run_gold.py --stage eval  --checkpoint checkpoints/Y_QWEN2VL_2B_GOLD/best_*.ckpt

Reuses the shared model / loss / Trainer / metrics unchanged. Honest input =
state_before + state_after + task_description. confidence/memory/recovery_outcome
heads are disabled in the gold config (no gold labels).
"""

from __future__ import annotations

import argparse

import torch

from web_agent.config import load_config
from web_agent.utils.seed import set_seed


def build(cfg):
    from transformers import AutoProcessor
    from web_agent.models.model import WebAgentModel
    from web_agent.models.loss import CombinedLoss
    from web_agent.data.gold_dataloader import load_gold_split, gold_class_weights

    bb = cfg["backbone"]
    processor = AutoProcessor.from_pretrained(
        bb["vlm_model"], min_pixels=bb["min_pixels"], max_pixels=bb["max_pixels"],
    )
    model = WebAgentModel(cfg)
    device = "cuda"
    for m in (model.adapter, model.failure_head, model.action_head,
              model.memory_head, model.recovery_outcome_head):
        m.to(device)

    train_recs = load_gold_split(cfg, "train")
    aw, fw, ow = gold_class_weights(train_recs)
    print("action w:", [round(x, 2) for x in aw.tolist()])
    print("failtype w:", [round(x, 2) for x in fw.tolist()])
    print("outcome w:", [round(x, 2) for x in ow.tolist()], "(capped)")
    loss_fn = CombinedLoss(
        cfg, action_class_weights=aw.to(device), failtype_class_weights=fw.to(device),
        outcome_class_weights=ow.to(device),
    ).to(device)
    return processor, model, loss_fn, train_recs, device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backbones/qwen2vl_2b_gold.yaml")
    ap.add_argument("--stage", required=True, choices=("smoke", "train", "eval"))
    ap.add_argument("--data-root", default=None, help="override cfg.data.root (Kaggle path)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--train-rows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    set_seed(cfg.get("seeds", [42])[0])

    from web_agent.data.gold_dataloader import build_gold_dataloader, load_gold_split
    from web_agent.train.trainer import Trainer, collect_predictions, compute_metrics

    processor, model, loss_fn, train_recs, device = build(cfg)

    if args.stage == "smoke":
        loader = build_gold_dataloader(cfg, "train", processor, records=train_recs,
                                       limit=16, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        model.train()
        with torch.autocast("cuda", dtype=torch.float16):
            preds = model(batch)
            terms = loss_fn(preds, {k: (v.to(device) if torch.is_tensor(v) else v)
                                    for k, v in batch.items()})
        print("loss terms:", {k: round(float(v.detach()), 4) for k, v in terms.items()})
        assert torch.isfinite(terms["total"]), "loss NaN/inf - STOP"
        for k in ("confidence", "memory", "recovery_outcome"):
            assert float(terms[k].detach()) * cfg["loss"][_w(k)] == 0.0, f"{k} not disabled"
        print("SMOKE PASS (disabled heads contribute 0)")
        return

    if args.stage == "train":
        train_loader = build_gold_dataloader(cfg, "train", processor, records=train_recs,
                                             limit=args.train_rows, shuffle=True, num_workers=4)
        val_loader = build_gold_dataloader(cfg, "val", processor, shuffle=False, num_workers=4)
        trainer = Trainer(model, loss_fn, cfg, train_loader, val_loader, train_sampler=None)
        if args.checkpoint:
            _load(model, args.checkpoint)   # optional synthetic-pretrained start
        print("training done:", trainer.fit())

    # final eval on the gold TEST split (both train and eval stages run this)
    test_loader = build_gold_dataloader(cfg, "test", processor, shuffle=False, num_workers=4)
    if args.stage == "eval" and args.checkpoint:
        _load(model, args.checkpoint)
    preds = collect_predictions(model, test_loader, device)
    m = compute_metrics(preds)
    print("GOLD TEST:", {k: round(v, 4) for k, v in m.items()})
    print("  (ignore memory_acc / recovery_outcome_acc / ece — heads disabled on gold)")


def _w(term_key: str) -> str:
    return {"confidence": "confidence", "memory": "memory_flag",
            "recovery_outcome": "recovery_outcome"}[term_key]


def _load(model, path: str) -> None:
    from peft import set_peft_model_state_dict
    ck = torch.load(path, map_location="cuda")
    set_peft_model_state_dict(model.encoder.model, ck["lora"])
    model.adapter.load_state_dict(ck["adapter"])
    model.failure_head.load_state_dict(ck["failure"])
    model.action_head.load_state_dict(ck["action"])
    model.memory_head.load_state_dict(ck["memory"])
    model.recovery_outcome_head.load_state_dict(ck["recovery_outcome"])
    print("loaded checkpoint:", path)


if __name__ == "__main__":
    main()
