"""Recovery v2.9 full-run wiring.

Design note -- why scoped injection instead of a forked ``run_gold_full``
-----------------------------------------------------------------------
v2.9 changes exactly two constructions inside ``run_gold_full``:

  * the action head becomes ``ActionHeadV29`` (adds the sub-cell centre offset);
  * the trainer becomes ``TrainerV29`` (step warmup, min-epoch early-stop floor).

Everything else -- resume validation, split hashing, the eight registered
quality gates, checkpoint round-trip, supplement reporting, bbox geometry audit
-- must stay bit-identical to v2.8, because those gates are what make the run
reportable.  Copying ~200 lines of gate logic into a v2.9 fork would create two
copies that can silently drift; a reviewer could then no longer tell whether a
v2.9 result passed the same gates as v2.8.

So this module rebinds the two names inside the ``gold_full`` module namespace
for the duration of one call, then restores them.  The rebinding is explicit,
scoped by a context manager, asserted on entry and exit, and covered by
``tests/test_v2_9.py::test_injection_restores_v2_8_bindings``.  No file on disk
is modified.

The locked test split is never read here; that guarantee lives in the v2.8 code
this delegates to.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from web_agent.models.heads import ActionHead
from web_agent.models.heads_v2_9 import ActionHeadV29
from web_agent.train import gold_full, gold_stages
from web_agent.train.trainer import Trainer
from web_agent.train.trainer_v2_9 import TrainerV29


def upgrade_action_head(model, *, centre_offset_max: float | None = None) -> None:
    """Replace ``model.action_head`` in place, carrying every v2.8 tensor over.

    ``build_gold_components`` calls ``initialize_bbox_size_prior`` on the head it
    built, so the state dict is copied rather than the head rebuilt from config.
    ``model.head_parameters()`` reads ``self.action_head`` dynamically, so the new
    parameters are picked up as long as this runs before the optimiser is built.
    """
    if isinstance(model.action_head, ActionHeadV29):
        raise RuntimeError("action head has already been upgraded to v2.9")
    kwargs: dict[str, Any] = {}
    if centre_offset_max is not None:
        kwargs["centre_offset_max"] = centre_offset_max
    model.action_head = ActionHeadV29.from_v2_8(model.action_head, **kwargs)
    print(
        "v2.9 action head installed (zero-initialised centre offset, "
        f"max={model.action_head.centre_offset_max})",
        flush=True,
    )


@contextlib.contextmanager
def v2_9_bindings(*, centre_offset_max: float | None = None):
    """Temporarily point the training modules at the v2.9 trainer and action head.

    Both ``gold_full`` and ``gold_stages`` are patched.  ``build_gold_components``
    is *defined* in ``gold_stages`` and *imported into* ``gold_full``, so those are
    two independent name bindings: patching only one would leave the other stage
    silently running the v2.8 head.  ``gold_stages`` is what ``run_gold_smoke``
    resolves, and the smoke is the only cheap pre-flight this run gets.
    """
    original_trainer = gold_full.Trainer
    original_full_builder = gold_full.build_gold_components
    original_stage_builder = gold_stages.build_gold_components
    if original_trainer is not Trainer:
        raise RuntimeError("gold_full.Trainer is not the v2.8 trainer; refusing to patch")

    def wrap(inner):
        def build_and_upgrade(*args, **kwargs):
            components = inner(*args, **kwargs)
            upgrade_action_head(components.model, centre_offset_max=centre_offset_max)
            return components
        return build_and_upgrade

    gold_full.Trainer = TrainerV29
    gold_full.build_gold_components = wrap(original_full_builder)
    gold_stages.build_gold_components = wrap(original_stage_builder)
    try:
        yield
    finally:
        gold_full.Trainer = original_trainer
        gold_full.build_gold_components = original_full_builder
        gold_stages.build_gold_components = original_stage_builder
    if (
        gold_full.Trainer is not Trainer
        or gold_full.build_gold_components is not original_full_builder
        or gold_stages.build_gold_components is not original_stage_builder
    ):
        raise RuntimeError("failed to restore v2.8 bindings")


def run_gold_smoke_v2_9(cfg: dict, *, processor=None, rows: int = 16, seed: int = 42) -> dict:
    """Run the 16-row fail-closed smoke with the v2.9 head installed.

    Without this, ``run_gold.py --stage smoke --config <v2.9 yaml>`` would verify
    the v2.9 *config* while forwarding and backwarding through the v2.8 *head* --
    the one cheap pre-flight before a multi-day run would test nothing new.
    """
    centre_offset_max = cfg.get("model", {}).get("centre_offset_max")
    with v2_9_bindings(centre_offset_max=centre_offset_max):
        return gold_stages.run_gold_smoke(cfg, processor=processor, rows=rows, seed=seed)


def run_gold_full_v2_9(
    cfg: dict,
    *,
    processor=None,
    epochs: int,
    seed: int = 42,
    checkpoint_root: str | Path,
    metrics_csv: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run the v2.8 full pipeline with the two v2.9 components substituted."""
    if str(cfg["loss"].get("dynamic_weighting")) != "fixed":
        raise ValueError(
            "v2.9 requires loss.dynamic_weighting='fixed' so the total loss is a "
            "sum of non-negative weighted terms"
        )
    centre_offset_max = cfg.get("model", {}).get("centre_offset_max")
    with v2_9_bindings(centre_offset_max=centre_offset_max):
        result = gold_full.run_gold_full(
            cfg,
            processor=processor,
            epochs=epochs,
            seed=seed,
            checkpoint_root=checkpoint_root,
            metrics_csv=metrics_csv,
            resume_checkpoint=resume_checkpoint,
        )
    result["v2_9"] = {
        "dynamic_weighting": "fixed",
        "centre_offset_max": centre_offset_max,
        "warmup": "absolute train.warmup_steps",
        "min_epochs_before_early_stop": int(cfg["train"].get("min_epochs", 0)),
    }
    return result
