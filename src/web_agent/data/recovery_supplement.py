"""Validated, source-aware loading for the RETRY/ABORT supplement v2.

The supplement stores one complete causal recovery transition per row:

    state_before (failure state)
      -> labels.action_type/value (executed recovery)
      -> state_after (post-recovery state)
      -> labels.recovery_success

Only recovery-strategy and recovery-outcome losses are active for these rows.
Private ``meta`` keys added here are runtime annotations; source JSON is never
mutated or overwritten.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "train": {"rows": 608, "RETRY": 313, "ABORT": 295},
    "val": {"rows": 194, "RETRY": 102, "ABORT": 92},
}
EXPECTED_IMAGES = 1_604
EXPECTED_ARCHIVE_SHA256 = (
    "c82fd5cb3567ffcbe7c3bf664a3c0aa9b0f4a408f733d321366e1343798b8c02"
)
SUPPLEMENT_SOURCE = "retry_abort_supplement_v2"

RECOVERY_ONLY_LOSS_MASKS = {
    "outcome": 0.0,
    "failure_type": 0.0,
    "action": 0.0,
    "recovery": 1.0,
    "needs_recovery": 0.0,
    "bbox": 0.0,
    "memory": 0.0,
    "confidence": 0.0,
    "calibration": 0.0,
    "contrastive": 0.0,
    "recovery_outcome": 1.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_supplement_root(path: str | Path) -> Path:
    """Resolve a directory containing ``data/supplement_train.json``."""
    root = Path(path).resolve()
    direct = root / "data" / "supplement_train.json"
    if direct.is_file():
        return root
    matches = sorted(root.rglob("data/supplement_train.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one data/supplement_train.json under "
            f"{root}; found {matches}"
        )
    return matches[0].parent.parent.resolve()


def verify_checksum_manifest(root: str | Path) -> dict[str, Any]:
    """Verify every regular file listed in ``SHA256SUMS.txt``."""
    resolved = resolve_supplement_root(root)
    manifest = resolved / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {manifest}")
    checked: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            failures.append({"entry": line, "reason": "invalid checksum line"})
            continue
        expected, relative = parts
        relative = relative.lstrip("*").replace("\\", "/")
        if relative.startswith(f"{resolved.name}/"):
            relative = relative.removeprefix(f"{resolved.name}/")
        candidate = (resolved / relative).resolve()
        try:
            candidate.relative_to(resolved)
        except ValueError:
            failures.append({"entry": line, "reason": "path escapes root"})
            continue
        if not candidate.is_file():
            failures.append({"path": relative, "reason": "missing file"})
            continue
        actual = _sha256(candidate)
        checked.append({"path": relative, "sha256": actual})
        if actual.lower() != expected.lower():
            failures.append({
                "path": relative,
                "reason": "sha256 mismatch",
                "expected": expected.lower(),
                "actual": actual.lower(),
            })
    return {
        "status": "PASS" if checked and not failures else "FAIL",
        "manifest": str(manifest),
        "checked_files": len(checked),
        "failures": failures,
    }


def _load_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return payload


def _private_annotate(record: dict, root: Path) -> dict:
    copied = deepcopy(record)
    if not isinstance(copied.get("inputs"), dict):
        raise ValueError("Supplement row is missing inputs")
    if not isinstance(copied.get("labels"), dict):
        raise ValueError("Supplement row is missing labels")
    meta = copied.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Supplement row meta must be an object")
    meta["_data_root"] = str(root)
    meta["_source_dataset"] = SUPPLEMENT_SOURCE
    meta["_direct_recovery_transition"] = True
    meta["_strategy_source_pre"] = True
    meta["_loss_masks"] = dict(RECOVERY_ONLY_LOSS_MASKS)
    return copied


def load_supplement_split(
    root: str | Path,
    which: str,
    *,
    annotate: bool = True,
) -> list[dict]:
    """Load ``train`` or ``val`` rows, optionally with private runtime metadata."""
    if which not in {"train", "val"}:
        raise ValueError("The supplement contains train and val only")
    resolved = resolve_supplement_root(root)
    rows = _load_json(resolved / "data" / f"supplement_{which}.json")
    if annotate:
        return [_private_annotate(record, resolved) for record in rows]
    return rows


def _identity_sets(rows: list[dict]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for key in ("task_id", "trajectory_id", "session_id", "site_group"):
        result[key] = {
            str(record.get("meta", {}).get(key))
            for record in rows
            if record.get("meta", {}).get(key) not in (None, "")
        }
    return result


def validate_supplement(root: str | Path) -> dict[str, Any]:
    """Validate the accepted v2 package without reading any Gold test data."""
    resolved = resolve_supplement_root(root)
    checksum_report = verify_checksum_manifest(resolved)
    split_rows = {
        split: load_supplement_split(resolved, split, annotate=False)
        for split in ("train", "val")
    }
    errors: list[str] = []
    split_reports: dict[str, dict[str, Any]] = {}
    referenced_images: set[Path] = set()

    for split, rows in split_rows.items():
        strategy_counts: Counter[str] = Counter()
        outcome_counts: dict[str, Counter[str]] = {
            "RETRY": Counter(),
            "ABORT": Counter(),
        }
        review_counts: Counter[str] = Counter()
        sample_ids: list[str] = []
        missing_images: list[str] = []
        invalid_rows: list[str] = []
        for index, record in enumerate(rows):
            inputs = record.get("inputs", {})
            labels = record.get("labels", {})
            meta = record.get("meta", {})
            sample_id = str(meta.get("sample_id") or f"{split}:{index}")
            sample_ids.append(sample_id)
            strategy = str(labels.get("recovery_strategy"))
            strategy_counts[strategy] += 1
            outcome_counts.setdefault(strategy, Counter())[
                str(bool(labels.get("recovery_success")))
            ] += 1
            review_counts[str(meta.get("review_status"))] += 1
            if strategy not in {"RETRY", "ABORT"}:
                invalid_rows.append(f"{sample_id}: strategy={strategy}")
            if labels.get("recovery_success") not in {True, False}:
                invalid_rows.append(f"{sample_id}: invalid recovery_success")
            if labels.get("recovery_attempted") is not True:
                invalid_rows.append(f"{sample_id}: recovery_attempted is not true")
            for field in ("state_before", "state_after"):
                relative = inputs.get(field)
                if not isinstance(relative, str) or not relative:
                    invalid_rows.append(f"{sample_id}: missing {field}")
                    continue
                candidate = (resolved / relative).resolve()
                try:
                    candidate.relative_to(resolved)
                except ValueError:
                    invalid_rows.append(f"{sample_id}: {field} escapes root")
                    continue
                referenced_images.add(candidate)
                if not candidate.is_file():
                    missing_images.append(relative)

        expected = EXPECTED_COUNTS[split]
        if len(rows) != expected["rows"]:
            errors.append(
                f"{split} rows: expected {expected['rows']}, found {len(rows)}"
            )
        for strategy in ("RETRY", "ABORT"):
            if strategy_counts[strategy] != expected[strategy]:
                errors.append(
                    f"{split} {strategy}: expected {expected[strategy]}, "
                    f"found {strategy_counts[strategy]}"
                )
            if set(outcome_counts[strategy]) != {"True", "False"}:
                errors.append(
                    f"{split} {strategy} does not contain both recovery outcomes"
                )
        if review_counts != Counter({"approved": len(rows)}):
            errors.append(f"{split} contains non-approved rows: {dict(review_counts)}")
        if len(sample_ids) != len(set(sample_ids)):
            errors.append(f"{split} contains duplicate sample_id values")
        if missing_images:
            errors.append(f"{split} has {len(missing_images)} missing image references")
        if invalid_rows:
            errors.append(f"{split} has {len(invalid_rows)} invalid rows")
        split_reports[split] = {
            "rows": len(rows),
            "recovery_strategy": dict(strategy_counts),
            "recovery_outcome": {
                name: dict(counts) for name, counts in outcome_counts.items()
            },
            "review_status": dict(review_counts),
            "missing_images": len(missing_images),
            "invalid_rows": invalid_rows[:50],
        }

    train_ids = _identity_sets(split_rows["train"])
    val_ids = _identity_sets(split_rows["val"])
    overlaps = {
        key: sorted(train_ids[key] & val_ids[key])
        for key in train_ids
    }
    for key, values in overlaps.items():
        if values:
            errors.append(f"train/val {key} overlap: {len(values)}")
    image_pairs = {}
    for split, rows in split_rows.items():
        image_pairs[split] = {
            (
                str(record.get("inputs", {}).get("state_before") or ""),
                str(record.get("inputs", {}).get("state_after") or ""),
            )
            for record in rows
        }
    image_pair_overlap = image_pairs["train"] & image_pairs["val"]
    if image_pair_overlap:
        errors.append(
            f"train/val exact image-pair overlap: {len(image_pair_overlap)}"
        )
    if len(referenced_images) != EXPECTED_IMAGES:
        errors.append(
            f"unique referenced images: expected {EXPECTED_IMAGES}, "
            f"found {len(referenced_images)}"
        )
    if checksum_report["status"] != "PASS":
        errors.append("SHA256SUMS verification failed")

    return {
        "status": "PASS" if not errors else "FAIL",
        "schema": "retry-abort-supplement-v2",
        "root": str(resolved),
        "test_rows_read": 0,
        "source_records_mutated": False,
        "loss_masks": dict(RECOVERY_ONLY_LOSS_MASKS),
        "splits": split_reports,
        "unique_referenced_images": len(referenced_images),
        "train_validation_overlap": {
            **{key: len(values) for key, values in overlaps.items()},
            "exact_image_pair": len(image_pair_overlap),
        },
        "checksum_manifest": checksum_report,
        "errors": errors,
    }
