"""Run logger wrapper (W&B / tensorboard / none) — IMPLEMENTATION_PLAN 0.3.

Logs every run config (model, lr, batch, seed) so every paper number is traceable,
plus per-step loss terms and per-epoch val metrics. Degrades gracefully: if the
chosen backend isn't installed, it falls back to console prints (never crashes a
training run over a logging dependency).

    logger = RunLogger(backend="wandb", config=cfg, run_name="Y1_seed42")
    logger.log({"loss/total": 0.83}, step=10)
    logger.finish()
"""

from __future__ import annotations

from typing import Any


class RunLogger:
    def __init__(
        self,
        backend: str = "wandb",
        config: dict | None = None,
        run_name: str | None = None,
        project: str = "failure-aware-web-agent",
    ):
        self.backend = (backend or "none").lower()
        self.run_name = run_name
        self._wandb = None
        self._tb = None

        if self.backend == "wandb":
            try:
                import wandb  # type: ignore

                self._wandb = wandb
                wandb.init(project=project, name=run_name, config=config or {})
            except Exception as e:  # not installed / offline / login missing
                print(f"[RunLogger] wandb unavailable ({e.__class__.__name__}); using console.")
                self.backend = "none"

        elif self.backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore

                self._tb = SummaryWriter(comment=f"_{run_name}" if run_name else "")
            except Exception as e:
                print(f"[RunLogger] tensorboard unavailable ({e.__class__.__name__}); using console.")
                self.backend = "none"

        if self.backend == "none":
            print(f"[RunLogger] console mode | run={run_name} | config keys={list((config or {}).keys())}")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        elif self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, global_step=step)
        else:
            prefix = f"step {step}: " if step is not None else ""
            print(prefix + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                      for k, v in metrics.items()))

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
        elif self._tb is not None:
            self._tb.close()
