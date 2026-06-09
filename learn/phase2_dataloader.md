# Phase 2 — Dataset & DataLoader

## 1. Learning topics

- **PyTorch `Dataset`** — a class that turns "record #i" into tensors.
- **PyTorch `DataLoader`** — batches many records and feeds them to the model.
- **Tensor** — an n-dimensional array (the data type GPUs work on).
- **Tokenization** — turning a sentence into integer IDs a text model understands.
- **Image preprocessing** — resize + normalize an image into a tensor.
- **Normalization** — scaling numbers into a fixed range (here bbox → [0,1]).
- **Masking** — a 0/1 flag telling the loss to ignore missing values.
- **Stratified sampling** — taking a small subset that keeps class proportions.
- **dtypes** — why labels are `int64` but regression targets are `float32`.

## 2. Why this phase exists

The model cannot read JSON, JPEGs, or strings. It only eats **tensors of numbers**.
Phase 2 is the translator: one messy record → a clean dict of tensors with the
right shapes and types. Everything downstream depends on this being exact.

## 3. Code structure

```
src/web_agent/data/
  dataset.py     class WebAgentDataset  (one record -> tensors)
  dataloader.py  load_split, filter_records, stratified_subsample, build_dataloader
```

Design choice (backbone-agnostic): the **image processor and text tokenizer are
passed in**, not hardcoded. The same `Dataset` works for SigLIP, CLIP, or a VLM —
you just inject a different processor. This is why one Dataset serves all models.

## 4. How it works (algorithm)

### `WebAgentDataset.__getitem__(i)` — the core translator
```
take record i:
  IMAGE
    open state_before image (UTF-8 path), convert to RGB
    processor(image) -> pixel_values tensor [3, 256, 256]   (resize + normalize)

  TEXT
    join "task_description </s> action_target_desc </s> website_domain"
    tokenizer(text, max_length=128, pad/truncate)
        -> input_ids [128], attention_mask [128]

  BBOX (where to click)
    if bbox exists:
        normalize x,y,width,height by the image's own width/height -> [0,1]
        mask = 1
    else:
        bbox = zeros(4), mask = 0      # loss will skip it

  LABELS (encode strings -> ints via labels.py)
    label_outcome    = EXECUTION_OUTCOME[outcome]     # 0/1
    label_failtype   = FAILURE_TYPE[failure_type]     # 0..3
    label_action     = ACTION_TYPE[action_type]       # 0..4
    label_recovery   = RECOVERY_STRATEGY[recovery]    # 0..5
    label_memory     = float(memory_update_flag)      # 0.0/1.0
    label_confidence = float(agent_confidence_before) # 0..1

  return dict of all these tensors
```

### Why two different dtypes (a key concept)
```
classification labels  -> int64, shape (B,)      e.g. label_outcome
regression/BCE targets -> float32, shape (B, 1)  e.g. label_confidence, bbox
```
- **CrossEntropyLoss** (for "which class?") needs an integer index per row.
- **MSELoss / BCELoss** (for "what value?" / "yes-no probability") need floats.
Getting this wrong is a classic crash. Our cell-3 output confirmed it:
`label_outcome (8,) int64` vs `label_confidence (8,1) float32`. Correct.

### `build_dataloader(...)` — assembling a batch feeder
```
build_dataloader(cfg, mode, records, processor, tokenizer, limit, batch_size):
  rows = filter_records(records, mode)        # apply one of the 5 mode filters
  if limit:  rows = stratified_subsample(rows, limit)   # smoke=16 / mini=5000
  ds = WebAgentDataset(rows, cfg, processor, tokenizer)
  return DataLoader(ds, batch_size, shuffle, num_workers=2, pin_memory=True)
```

The DataLoader automatically **stacks** N single records into a batch:
`pixel_values [3,256,256]` × 8 records → `[8, 3, 256, 256]`. That extra first
dimension (the "8") is the batch dimension every model expects.

### The 5 mode filters — `filter_records`
Same JSON, different rows, chosen by a string (never new files):
```
full_labels -> split == "train"                         (all training rows)
visual      -> train and not borrowed_image             (clean images only)
eval_labels -> test_task/website/domain                 (all eval rows)
eval_visual -> eval and not borrowed_image              (clean eval images)
bbox        -> action_target_bbox is not None           (rows with a click box)
```

### Stratified subsample — `stratified_subsample`
For quick test runs we want 5,000 rows that still contain rare classes
(LOOP_DETECTED is only 7.4%). The algorithm keeps each class proportional:
```
group rows by failure_type
for each group:
    take round(n * group_size / total) rows at random (seeded)
shuffle, return n rows
```
A naive `rows[:5000]` could miss a whole class; stratifying guarantees coverage.

## 5. Key functions

| Function | File | In → Out |
|----------|------|----------|
| `WebAgentDataset.__getitem__(i)` | [dataset.py](../src/web_agent/data/dataset.py) | index → dict of tensors |
| `WebAgentDataset._bbox(rec, w, h)` | dataset.py | record + image size → (bbox[4], mask[1]) |
| `WebAgentDataset._text_tokens(rec)` | dataset.py | record → (input_ids[128], attention_mask[128]) |
| `load_split(cfg, which)` | [dataloader.py](../src/web_agent/data/dataloader.py) | "train" → list of records (UTF-8) |
| `filter_records(records, mode)` | dataloader.py | rows + mode → filtered rows |
| `stratified_subsample(rows, n)` | dataloader.py | rows + n → balanced subset |
| `build_dataloader(...)` | dataloader.py | everything → a `DataLoader` |

### What "correct output" looked like (real Kaggle run)
```
pixel_values  (8, 3, 256, 256) float32     ← 8 images, 3 colors, 256×256
input_ids     (8, 128) int64               ← 8 sentences, 128 tokens each
bbox          (8, 4) float32               ← 8 click boxes, normalized
label_outcome (8,) int64                   ← 8 class indices

encoded label_outcome : [1, 0, 1, 1]   → FAILURE, SUCCESS, FAILURE, FAILURE
encoded label_action  : [0, 0, 1, 1]   → CLICK, CLICK, TYPE, TYPE
```
Shapes, dtypes, and label encodings all correct → the translator works.
