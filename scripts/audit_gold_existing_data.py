"""Create non-destructive bbox and weak-class review packages on Kaggle.

The attached Web-Gold-40K ZIP is streamed in place.  The script reads only the
train and validation JSON/images, writes small CSV/JSON/PNG artifacts under
``/kaggle/working``, and never edits or extracts the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from validate_kaggle_gold import (
    ZipDataset,
    find_data_source,
    load_records,
    open_binary,
    resolve_image,
)
from web_agent.data.improvement_audit import (
    build_bbox_review_queue,
    build_weak_class_review_queue,
    class_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/input/datasets/kiyasmahmud/web-gold-40k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/gold_existing_data_improvement"),
    )
    parser.add_argument("--review-tasks-per-class", type=int, default=100)
    parser.add_argument("--minimum-val-support", type=int, default=100)
    parser.add_argument("--montage-rows-per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


class ImageReader:
    """Cache locators/sizes while leaving image bytes inside the source ZIP."""

    def __init__(self, source):
        self.source = source
        self.locators: dict[str, object | None] = {}
        self.sizes: dict[str, tuple[tuple[int, int] | None, str]] = {}

    def locator(self, reference: str):
        if reference not in self.locators:
            self.locators[reference] = resolve_image(self.source, reference)
        return self.locators[reference]

    def info(self, reference: str) -> tuple[tuple[int, int] | None, str]:
        if reference in self.sizes:
            return self.sizes[reference]
        if not reference:
            result = (None, "missing_state_before_path")
        else:
            locator = self.locator(reference)
            if locator is None:
                result = (None, "unreadable_state_before_image")
            else:
                try:
                    with open_binary(self.source, locator) as handle:
                        with Image.open(handle) as image:
                            result = (image.size, "")
                except (FileNotFoundError, OSError, ValueError):
                    result = (None, "unreadable_state_before_image")
        self.sizes[reference] = result
        return result

    def image(self, reference: str) -> Image.Image | None:
        locator = self.locator(reference)
        if locator is None:
            return None
        try:
            with open_binary(self.source, locator) as handle:
                payload = handle.read()
            with Image.open(io.BytesIO(payload)) as image:
                return image.convert("RGB")
        except (FileNotFoundError, OSError, ValueError):
            return None


def _ellipsize(text: object, limit: int = 68) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _fit_image(image: Image.Image, width: int, height: int) -> tuple[Image.Image, float]:
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas, scale


def bbox_montage(
    rows: list[dict],
    reader: ImageReader,
    destination: Path,
    *,
    limit: int,
) -> int:
    chosen = sorted(
        rows,
        key=lambda row: (
            row["geometry_reasons"],
            row["split"],
            row["sample_id"],
        ),
    )[:limit]
    if not chosen:
        return 0
    tile_width, tile_height = 480, 335
    columns = 2
    output = Image.new(
        "RGB",
        (columns * tile_width, math.ceil(len(chosen) / columns) * tile_height),
        "white",
    )
    font = ImageFont.load_default()
    for index, row in enumerate(chosen):
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        draw = ImageDraw.Draw(tile)
        source = reader.image(str(row["state_before"]))
        if source is None:
            draw.text((8, 8), "IMAGE UNREADABLE", fill="red", font=font)
        else:
            image, scale = _fit_image(source, tile_width, 270)
            tile.paste(image, (0, 0))
            draw = ImageDraw.Draw(tile)
            offset_x = (tile_width - round(source.width * scale)) // 2
            offset_y = (270 - round(source.height * scale)) // 2
            try:
                x = float(row["original_x"])
                y = float(row["original_y"])
                width = float(row["original_width"])
                height = float(row["original_height"])
                draw.rectangle(
                    (
                        offset_x + x * scale,
                        offset_y + y * scale,
                        offset_x + (x + width) * scale,
                        offset_y + (y + height) * scale,
                    ),
                    outline="red",
                    width=3,
                )
            except (TypeError, ValueError):
                pass
        draw.text(
            (6, 275),
            _ellipsize(f"{row['split']} {row['sample_id']}"),
            fill="black",
            font=font,
        )
        draw.text(
            (6, 292),
            _ellipsize(row["geometry_reasons"]),
            fill="red",
            font=font,
        )
        draw.text(
            (6, 309),
            _ellipsize(f"bbox={row['original_bbox']}"),
            fill="black",
            font=font,
        )
        output.paste(
            tile,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    output.save(destination)
    return len(chosen)


def weak_class_montages(
    rows: list[dict],
    reader: ImageReader,
    output_dir: Path,
    *,
    per_category: int,
) -> dict[str, int]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row["is_focus_row"]:
            continue
        for category in str(row["row_focus_targets"]).split("|"):
            if category:
                by_category[category].append(row)

    counts: dict[str, int] = {}
    font = ImageFont.load_default()
    for category, candidates in sorted(by_category.items()):
        chosen = sorted(
            candidates,
            key=lambda row: (row["split"], row["task_id"], row["step_index"]),
        )[:per_category]
        if not chosen:
            continue
        tile_width, tile_height = 800, 285
        output = Image.new(
            "RGB",
            (tile_width, len(chosen) * tile_height),
            "white",
        )
        for index, row in enumerate(chosen):
            tile = Image.new("RGB", (tile_width, tile_height), "white")
            before = reader.image(str(row["state_before"]))
            after = reader.image(str(row["state_after"]))
            if before is not None:
                before_tile, _ = _fit_image(before, 395, 220)
                tile.paste(before_tile, (0, 0))
            if after is not None:
                after_tile, _ = _fit_image(after, 395, 220)
                tile.paste(after_tile, (405, 0))
            draw = ImageDraw.Draw(tile)
            draw.text((6, 224), "BEFORE", fill="black", font=font)
            draw.text((411, 224), "AFTER", fill="black", font=font)
            draw.text(
                (6, 242),
                _ellipsize(
                    f"{row['split']} {row['sample_id']} action={row['action_type']} "
                    f"failure={row['failure_type_4']} recovery={row['recovery_strategy']}",
                    110,
                ),
                fill="black",
                font=font,
            )
            draw.text(
                (6, 260),
                _ellipsize(row["task_description"], 110),
                fill="black",
                font=font,
            )
            output.paste(tile, (0, index * tile_height))
        safe_name = category.lower().replace(":", "_")
        output.save(output_dir / f"review_montage_{safe_name}.png")
        counts[category] = len(chosen)
    return counts


def readme_text(report: dict) -> str:
    bbox_total = sum(
        item["invalid_bbox_rows"] for item in report["bbox_audit"].values()
    )
    collection = report["class_coverage"]["targeted_new_collection_required"]
    collection_text = ", ".join(collection) if collection else "none"
    return f"""# Existing-data improvement review package

This package was generated without modifying or extracting the source dataset.
The locked test split was not read (`test_rows_read = 0`).

## BBox decision

- Invalid train/validation bbox rows: **{bbox_total}**
- Runtime disposition: keep each record for non-localization heads and set only
  `bbox_mask = 0`.
- Do not copy coordinates from the montage, clamp them, or invent corrections.
- Fill proposed bbox columns only when the replay/source screenshot proves the
  replacement and the second reviewer confirms it.

## Weak classes

- Review complete trajectories in `weak_class_review_queue.csv`.
- `A+B` assignments are the double-reviewed agreement subset.
- Preserve both reviewers' original decisions.
- Zero-support recovery classes requiring targeted real collection if claimed:
  **{collection_text}**.

## Files

- `bbox_review_queue.csv`: every invalid bbox and blank correction fields.
- `bbox_mask_manifest.json`: immutable mask evidence for current training.
- `weak_class_review_queue.csv`: complete trajectories selected for weak-class review.
- `dataset_improvement_audit.json`: counts, coverage, and generation settings.
- `review_montage_*.png`: visual aids; replay evidence remains authoritative.

After review, do not edit split JSON files directly. Apply approved changes to
the source/export ledger, regenerate splits, and rerun the full validator.
"""


def main() -> int:
    args = parse_args()
    if args.review_tasks_per_class <= 0:
        raise ValueError("--review-tasks-per-class must be positive")
    if args.montage_rows_per_category <= 0:
        raise ValueError("--montage-rows-per-category must be positive")

    source = find_data_source(args.data_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = ImageReader(source)

    records_by_split = {
        "train": load_records(source, "split_train.json"),
        "val": load_records(source, "split_val.json"),
    }
    bbox_rows: list[dict] = []
    weak_rows: list[dict] = []
    bbox_reports: dict[str, dict] = {}
    weak_reports: dict[str, dict] = {}
    for split, records in records_by_split.items():
        split_bbox_rows, bbox_report = build_bbox_review_queue(
            records,
            split,
            reader.info,
            seed=args.seed,
        )
        split_weak_rows, weak_report = build_weak_class_review_queue(
            records,
            split,
            max_tasks_per_class=args.review_tasks_per_class,
            seed=args.seed,
        )
        bbox_rows.extend(split_bbox_rows)
        weak_rows.extend(split_weak_rows)
        bbox_reports[split] = bbox_report
        weak_reports[split] = weak_report

    write_csv(output_dir / "bbox_review_queue.csv", bbox_rows)
    write_csv(output_dir / "weak_class_review_queue.csv", weak_rows)
    mask_manifest = {
        "schema_version": "bbox-mask-manifest-v1",
        "definition": (
            "Mask only localization loss for invalid geometry; preserve the "
            "record for every other supervised task."
        ),
        "source_records_mutated": False,
        "test_rows_read": 0,
        "rows": [
            {
                "split": row["split"],
                "sample_id": row["sample_id"],
                "bbox_mask": 0,
                "reasons": row["geometry_reasons"].split("|"),
            }
            for row in bbox_rows
        ],
    }
    (output_dir / "bbox_mask_manifest.json").write_text(
        json.dumps(mask_manifest, indent=2),
        encoding="utf-8",
    )

    montage_counts = {
        "bbox": bbox_montage(
            bbox_rows,
            reader,
            output_dir / "bbox_invalid_review_montage.png",
            limit=args.montage_rows_per_category * 2,
        ),
        "weak_classes": weak_class_montages(
            weak_rows,
            reader,
            output_dir,
            per_category=args.montage_rows_per_category,
        ),
    }
    source_description = (
        source.description if isinstance(source, ZipDataset) else str(source)
    )
    report = {
        "schema_version": "gold-existing-data-improvement-v1",
        "status": "PASS",
        "source": source_description,
        "source_mode": "zip_stream" if isinstance(source, ZipDataset) else "directory",
        "source_records_mutated": False,
        "source_images_copied_or_extracted": False,
        "splits_read": ["train", "val"],
        "test_rows_read": 0,
        "settings": {
            "seed": args.seed,
            "review_tasks_per_class": args.review_tasks_per_class,
            "minimum_validation_support": args.minimum_val_support,
            "montage_rows_per_category": args.montage_rows_per_category,
        },
        "bbox_audit": bbox_reports,
        "weak_class_review": weak_reports,
        "class_coverage": class_coverage(
            records_by_split,
            minimum_validation_support=args.minimum_val_support,
        ),
        "outputs": {
            "bbox_review_queue_rows": len(bbox_rows),
            "weak_class_review_queue_rows": len(weak_rows),
            "bbox_mask_manifest_rows": len(mask_manifest["rows"]),
            "montage_rows": montage_counts,
        },
    }
    report_path = output_dir / "dataset_improvement_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(readme_text(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("Review package:", output_dir)
    if isinstance(source, ZipDataset):
        source.archive.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
