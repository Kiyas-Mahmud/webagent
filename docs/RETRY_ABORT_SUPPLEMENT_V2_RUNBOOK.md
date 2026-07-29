# RETRY/ABORT Supplement v2 Runbook

## Purpose

This gate integrates the reviewed supplement without modifying Web-Gold-40K.
Supplement rows supervise only:

- recovery strategy (`RETRY` or `ABORT`);
- recovery success.

Outcome, failure type, action, bbox, confidence, memory, calibration, and
contrastive losses are masked for supplement rows.

## Kaggle inputs

Attach:

1. the existing `kiyasmahmud/web-gold-40k` dataset;
2. `kiyasmahmud/gold-40k-retry`.

Expected Kaggle mount paths:

```text
/kaggle/input/datasets/kiyasmahmud/web-gold-40k
/kaggle/input/datasets/kiyasmahmud/gold-40k-retry
```

Kaggle may expose the second dataset as already-extracted files instead of
retaining `web_gold_40k_retry_abort_supplement_v2_kaggle.zip`. Both layouts are
supported. When the ZIP is retained, its archive SHA-256 is checked. When Kaggle
extracts it, `SHA256SUMS.txt` verifies the mounted package files instead.

Use a Tesla T4. Do not use CPU or P100 for this controlled gate.

## Notebook to run now

Run all cells in:

```text
notebooks/kaggle_gold_retry_abort_supplement_v2_sanity.ipynb
```

Do not edit or run `notebooks/kaggle_gold.ipynb` for this gate.

The notebook:

1. rejects non-T4 hardware;
2. pulls the `Code` branch;
3. locates either the extracted supplement or retained ZIP;
4. verifies the ZIP hash when available and always verifies package checksums;
5. validates 608 train rows, 194 validation rows, and 1,604 images;
6. proves no supplement test split is read;
7. audits 24,107 combined training rows;
8. keeps the 7,861-row original validation source as checkpoint-selection data;
9. keeps all 194 supplement validation rows separately available;
10. runs one forward/backward batch and checks that only recovery output heads
    receive direct supplement gradients.

## Required PASS outputs

Download these files from `/kaggle/working`:

```text
retry_abort_supplement_v2_validation_report.json
retry_abort_supplement_v2_multisource_report.json
retry_abort_supplement_v2_smoke_report.json
retry_abort_supplement_v2_sanity_summary.json
```

All three reports and the final summary must say `PASS`. If any report fails,
do not start the controlled mini.

## After PASS

The next experiment is a five-epoch, 5,000-row v2.8 mini. Its 5,000 training
rows include every accepted supplement-train row once and fill the remainder
from original Gold using deterministic joint stratification. It selects the
checkpoint only on original Gold validation and then reports RETRY/ABORT
strategy and recovery-success metrics on the complete supplement validation
source separately.

The locked test split remains unavailable until the final frozen evaluation.
