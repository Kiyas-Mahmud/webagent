# learn/ — How this project works, phase by phase

> **HISTORICAL TUTORIAL — DO NOT EXECUTE AS THE CURRENT RESEARCH PROTOCOL.**
> These lessons describe earlier corpus, action-space, routing, and evaluation
> stages and intentionally preserve that history. They are not authority for
> training, checkpoint selection, locked-test access, the six-action runtime,
> or Table 2. Use `docs/AGENT.md`, `docs/DGX_THREE_MODEL_COMPARISON.md`, and
> `docs/TABLE2_END_TO_END_RUNTIME_AND_POST_TRAINING_PLAN.md` instead.

This folder teaches you the **techniques, code structure, algorithms, and key
functions** behind every part of the project. One file per phase. Read in order.

Each phase file has the same five sections so you always know where to look:

1. **Learning topics** — the concepts to understand (look these up if new).
2. **Why this phase exists** — the problem it solves.
3. **Code structure** — which files do what, and how they connect.
4. **How it works (algorithm)** — step by step, in plain language.
5. **Key functions** — the important functions, what they take and return.

## Index

| Phase | File | What you learn |
|-------|------|----------------|
| 0 | [phase0_setup.md](phase0_setup.md) | Python packages, `src` layout, `sys.path`, YAML config inheritance, reproducible seeds |
| 1 | [phase1_data_verification.md](phase1_data_verification.md) | Data validation, assertions, distributions, label-logic consistency |
| 2 | [phase2_dataloader.md](phase2_dataloader.md) | PyTorch `Dataset`/`DataLoader`, tokenization, image preprocessing, bbox normalization, stratified sampling, tensor shapes & dtypes |
| 3+ | [roadmap_phase3plus.md](roadmap_phase3plus.md) | Preview: encoders, cross-attention fusion, task heads, combined loss, training loop |
| 5 | [phase5_qlora_training.md](phase5_qlora_training.md) | QLoRA, peft, param groups, 10-term loss, contrastive, calibration, the Trainer |

## The 30-second mental model

```
docs/      = the rules (what to build, why)
configs/   = the knobs (one YAML per model)
src/web_agent/ = the engine (reusable code)
notebooks/ = the cockpit (run + watch on Kaggle)
learn/     = the textbook (this folder)
```

A screenshot + a task sentence go in. The model must say: did this step fail,
what kind of failure, what action to take, where to click, should we remember it,
how to recover. Everything in `src/` exists to make that one prediction, trained
on 70,965 labeled web-interaction steps.

## How to study this

1. Open the phase file.
2. Read sections 1-2 (topics + why).
3. Open the real source file it points to, side by side.
4. Match each "Key function" to the code.
5. Run the Kaggle notebook cell for that phase and watch the output.
