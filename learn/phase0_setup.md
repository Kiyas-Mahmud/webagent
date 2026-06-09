# Phase 0 — Project Setup & Foundations

## 1. Learning topics

- **Python package** — a folder with `__init__.py` files that Python can `import`.
- **`src` layout** — putting the package under `src/` so tests/imports use the
  installed package, not accidental local files.
- **`sys.path`** — the list of folders Python searches when you `import` something.
- **YAML config + inheritance** — storing settings in `.yaml` files; a child file
  overrides a base file.
- **Deterministic seeds** — making random operations repeatable for science.
- **Editable install vs `sys.path`** — two ways to make a package importable, and
  why one broke on Kaggle.

## 2. Why this phase exists

Before any model, you need: a place for code that 19 different models can reuse,
a way to pick settings without editing code, and guaranteed reproducibility
(Q1 journals require it). Phase 0 builds that skeleton.

## 3. Code structure

```
pyproject.toml          declares the package "web_agent" and its dependencies
configs/base.yaml       shared defaults for ALL models
configs/backbones/      one YAML per model, each "extends: base.yaml"
src/web_agent/
  __init__.py           marks the package (version lives here)
  config.py             loads YAML and merges base + override
  labels.py             the integer label maps (single source of truth)
  utils/seed.py         set_seed() for reproducibility
```

### Why `src/web_agent/` and not just `web_agent/`?
The `src` layout forces you to import the *installed* package. It stops a common
bug where your code accidentally imports a half-finished local folder instead of
the real package. `pyproject.toml` tells the packager "the code is under `src`":

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

## 4. How it works (algorithm)

### Config loading — `config.py`
```
load_config("configs/backbones/y1_siglip_roberta.yaml"):
  1. open the file, parse YAML into a dict
  2. if it has  extends: base.yaml
        load base.yaml first (recursively)
        deep-merge this file ON TOP of base
  3. return the merged dict
```
"Deep merge" means nested dicts combine key-by-key, and the child wins on
conflicts. So `base.yaml` sets `loss.outcome: 0.25` and Y1 keeps it, but Y1
overrides `fusion.type` and `seeds`.

### Reproducibility — `seed.py`
```
set_seed(42):
  seed python's random, numpy, and torch (CPU + GPU)
  set PYTHONHASHSEED
  turn on cudnn deterministic mode
```
Call it once at the top of every run so the same code + same seed = same numbers.

### The Kaggle import lesson (important real bug)
We first tried `pip install -e .` (an "editable install"). On Kaggle this
registered a **meta-path finder** (PEP 660) that hid sub-packages, causing
`No module named 'web_agent.data'`. The fix was simpler: add the source folder
to `sys.path` directly.

```python
sys.path.insert(0, "/kaggle/working/webagent/src")  # now `import web_agent` works
```
`sys.path` is just a list; whatever folder is on it, Python can import from it.

> Separate, sneakier bug we also hit: `.gitignore` had `data/`, which matched
> **every** folder named `data` — including `src/web_agent/data/`. That package
> never got committed, so Kaggle's clone couldn't import it. Lesson: anchor
> ignore rules with a leading slash (`/data/`) so they only match the repo root.

## 5. Key functions

| Function | File | In → Out |
|----------|------|----------|
| `load_config(path)` | [config.py](../src/web_agent/config.py) | yaml path → merged settings dict |
| `_deep_merge(base, override)` | [config.py](../src/web_agent/config.py) | two dicts → one merged dict |
| `set_seed(seed=42)` | [utils/seed.py](../src/web_agent/utils/seed.py) | an int → (side effect: global determinism) |

### The label maps — [labels.py](../src/web_agent/labels.py)
Not a function, but the most important data in the project. It turns strings into
the integers the model trains on:
```python
EXECUTION_OUTCOME = {"SUCCESS": 0, "FAILURE": 1}
ACTION_TYPE       = {"CLICK": 0, "TYPE": 1, "SELECT": 2, "SCROLL": 3, "NAVIGATE": 4}
```
Every part of the code imports these from one file, so an encoding can never
drift out of sync — the #1 cause of "loss won't go down" bugs.
