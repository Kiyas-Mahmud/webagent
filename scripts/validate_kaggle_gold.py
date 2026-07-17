"""Validate the Web Gold 40K Kaggle dataset without downloading it locally.

Run this inside a Kaggle Notebook after attaching kiyasmahmud/web-gold-40k.
Only small JSON reports are written, and only under /kaggle/working.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import random
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image


EXPECTED_SPLIT_ROWS = {"train": 23_499, "val": 7_861, "test": 7_855}
EXPECTED_TOTAL_ROWS = 39_215
EXPECTED_SPLIT_TASKS = {"train": 17_981, "val": 6_003, "test": 5_994}
EXPECTED_SPLIT_DOMAINS = {"train": 305, "val": 101, "test": 103}
EXPECTED_OUTCOME_COUNTS = {"SUCCESS": 17_136, "FAILURE": 22_079}
EXPECTED_FAILURE_COUNTS = {
    "NONE": 17_136,
    "ACTION_MISMATCH": 10_832,
    "PERCEPTION_ERROR": 10_053,
    "LOOP_DETECTED": 1_194,
}
EXPECTED_ACTION_COUNTS = {
    "CLICK": 6_427,
    "NAVIGATE": 7_444,
    "TYPE": 6_469,
    "SELECT": 6_465,
    "SCROLL": 6_463,
    "PRESS_KEY": 5_947,
}
EXPECTED_INPUT_KEYS = {
    "state_before",
    "state_after",
    "task_description",
    "website_domain",
}
REQUIRED_LABEL_KEYS = {
    "outcome_label",
    "failure_type_4",
    "agent_confidence_before",
    "action_type",
    "action_coordinates",
    "action_target_bbox",
    "recovery_attempted",
    "recovery_strategy",
    "recovery_success",
    "memory_update_flag",
}
EXPECTED_OUTCOMES = {"SUCCESS", "FAILURE"}
EXPECTED_FAILURE_TYPES = {
    "NONE",
    "ACTION_MISMATCH",
    "PERCEPTION_ERROR",
    "LOOP_DETECTED",
}
EXPECTED_ACTIONS = {
    "CLICK",
    "NAVIGATE",
    "TYPE",
    "SELECT",
    "SCROLL",
    "PRESS_KEY",
}
SPLIT_FILES = {
    "train": "split_train.json",
    "val": "split_val.json",
    "test": "split_test.json",
}


@dataclass
class Gate:
    name: str
    status: str
    detail: str


class Audit:
    def __init__(self) -> None:
        self.gates: list[Gate] = []

    def add(self, name: str, passed: bool, detail: str, severity: str = "FAIL") -> None:
        status = "PASS" if passed else severity
        self.gates.append(Gate(name, status, detail))
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
        print(f"{icon} {name}: {detail}")

    def skip(self, name: str, detail: str) -> None:
        self.gates.append(Gate(name, "SKIP", detail))
        print(f"[SKIP] {name}: {detail}")

    def summary(self) -> Counter:
        return Counter(g.status for g in self.gates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/kaggle/working/web_gold_40k_validation_report.json"),
    )
    parser.add_argument("--decode-sample", type=int, default=600)
    parser.add_argument("--full-image-hash", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--adult-domain-policy-confirmed", action="store_true")
    return parser.parse_args()


def normalise_member(name: str) -> str | None:
    cleaned = name.replace("\\", "/")
    path = PurePosixPath(cleaned)
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = [part for part in path.parts if part not in {"", "."}]
    return "/".join(parts)


class ZipDataset:
    """Read a nested gold dataset directly from one ZIP without extracting it."""

    def __init__(self, archive_path: Path):
        self.archive_path = archive_path.resolve()
        self.archive = zipfile.ZipFile(self.archive_path)
        self.member_map: dict[str, str] = {}
        self.duplicate_members: list[str] = []
        for info in self.archive.infolist():
            if info.is_dir():
                continue
            normalised = normalise_member(info.filename)
            if normalised is None:
                continue
            if normalised in self.member_map:
                self.duplicate_members.append(normalised)
            self.member_map[normalised] = info.filename

        prefixes = []
        for member in self.member_map:
            if PurePosixPath(member).name != SPLIT_FILES["train"]:
                continue
            prefix = posixpath.dirname(member)
            if all(self._join(prefix, filename) in self.member_map for filename in SPLIT_FILES.values()):
                prefixes.append(prefix)
        if not prefixes:
            self.archive.close()
            raise FileNotFoundError(
                f"{self.archive_path.name} does not contain sibling train/val/test split JSON files"
            )
        prefixes.sort(key=lambda value: (len(PurePosixPath(value).parts), value))
        self.prefix = prefixes[0]

    @staticmethod
    def _join(prefix: str, child: str) -> str:
        return posixpath.join(prefix, child) if prefix else child

    @property
    def description(self) -> str:
        suffix = f"!/{self.prefix}" if self.prefix else "!/"
        return f"{self.archive_path}{suffix}"

    def split_member(self, filename: str) -> str:
        return self._join(self.prefix, filename)

    def load_records(self, filename: str) -> list[dict[str, Any]]:
        member = self.split_member(filename)
        with self.archive.open(self.member_map[member], "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as text:
                return records_from_payload(json.load(text), member)

    def resolve_image(self, reference: Any) -> str | None:
        if not isinstance(reference, str) or not reference.strip():
            return None
        rel = normalise_member(reference)
        if rel is None:
            return None
        candidates = [self._join(self.prefix, rel), rel]
        if not rel.lower().startswith("images/"):
            candidates.insert(1, self._join(self.prefix, f"images/{rel}"))
        for candidate in candidates:
            if candidate in self.member_map:
                return candidate
        return None

    def open_binary(self, member: str):
        return self.archive.open(self.member_map[member], "r")


DataSource = Path | ZipDataset
Locator = Path | str


def zip_candidates(root: Path) -> list[Path]:
    candidates = []
    for pattern in ("*.zip", "*/*.zip", "*/*/*.zip"):
        candidates.extend(root.glob(pattern))
    return sorted(set(path.resolve() for path in candidates if path.is_file()))


def open_zip_candidate(path: Path) -> ZipDataset | None:
    try:
        return ZipDataset(path)
    except (FileNotFoundError, zipfile.BadZipFile):
        return None


def find_data_source(explicit: Path | None) -> DataSource:
    if explicit is not None:
        source_path = explicit.resolve()
        if source_path.is_file():
            if source_path.suffix.lower() != ".zip":
                raise ValueError(f"Dataset file must be a ZIP archive: {source_path}")
            return ZipDataset(source_path)
        if not source_path.is_dir():
            raise FileNotFoundError(source_path)
        if all((source_path / name).is_file() for name in SPLIT_FILES.values()):
            return source_path
        for archive_path in zip_candidates(source_path):
            archive = open_zip_candidate(archive_path)
            if archive is not None:
                return archive
        raise FileNotFoundError(
            f"No sibling split JSON files or compatible ZIP archive found under {source_path}"
        )

    input_root = Path("/kaggle/input")
    if not input_root.is_dir():
        raise RuntimeError("Run this on Kaggle and attach kiyasmahmud/web-gold-40k first.")

    candidates = []
    train_files = list(input_root.glob(f"*/{SPLIT_FILES['train']}"))
    train_files += list(input_root.glob(f"*/*/{SPLIT_FILES['train']}"))
    if not train_files:
        train_files = list(input_root.rglob(SPLIT_FILES["train"]))
    for train_file in train_files:
        root = train_file.parent
        if all((root / name).is_file() for name in SPLIT_FILES.values()):
            candidates.append(root)
    if candidates:
        candidates.sort(key=lambda p: ("web-gold-40k" not in str(p).lower(), len(p.parts)))
        if len(candidates) > 1:
            print("Candidate folders:", *candidates, sep="\n  - ")
            print("Using:", candidates[0])
        return candidates[0].resolve()

    archives = zip_candidates(input_root)
    archives.sort(key=lambda p: ("web-gold-40k" not in str(p).lower(), len(p.parts), str(p)))
    for archive_path in archives:
        archive = open_zip_candidate(archive_path)
        if archive is not None:
            print("Using ZIP archive:", archive.description)
            return archive
    raise FileNotFoundError(
        "No folder or ZIP containing split_train.json, split_val.json, and split_test.json "
        "was found under /kaggle/input. Attach kiyasmahmud/web-gold-40k."
    )


def records_from_payload(payload: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("records", "data", "rows", "samples"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        else:
            raise TypeError(f"{name} is an object but has no records/data/rows/samples list")
    else:
        raise TypeError(f"{name} must contain a JSON list or record container")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{name} contains non-object rows")
    return rows


def load_records(source: DataSource, filename: str) -> list[dict[str, Any]]:
    if isinstance(source, ZipDataset):
        return source.load_records(filename)
    path = source / filename
    with path.open(encoding="utf-8") as handle:
        return records_from_payload(json.load(handle), path.name)


def meta_value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    meta = row.get("meta") or {}
    for key in keys:
        value = meta.get(key)
        if value is not None and value != "":
            return value
    return None


def select_meta_key(rows: list[dict[str, Any]], candidates: list[str]) -> str | None:
    if not rows:
        return None
    scores = Counter()
    for row in rows:
        meta = row.get("meta") or {}
        for key in candidates:
            if meta.get(key) not in (None, ""):
                scores[key] += 1
    for key in candidates:
        if scores[key] == len(rows):
            return key
    return scores.most_common(1)[0][0] if scores else None


def counter_text(counter: Counter) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items(), key=str))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def safe_image_path(root: Path, reference: Any) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    rel = Path(reference.replace("\\", "/"))
    candidates = [root / rel]
    if not str(rel).lower().startswith("images/"):
        candidates.append(root / "images" / rel)
    root_resolved = root.resolve()
    first_safe: Path | None = None
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        if first_safe is None:
            first_safe = resolved
        if resolved.is_file():
            return resolved
    return first_safe


def resolve_image(source: DataSource, reference: Any) -> Locator | None:
    if isinstance(source, ZipDataset):
        return source.resolve_image(reference)
    return safe_image_path(source, reference)


def open_binary(source: DataSource, locator: Locator):
    if isinstance(source, ZipDataset):
        if not isinstance(locator, str):
            raise TypeError("ZIP member locator must be a string")
        return source.open_binary(locator)
    if not isinstance(locator, Path):
        raise TypeError("Directory locator must be a Path")
    return locator.open("rb")


def sha256(source: DataSource, locator: Locator) -> str:
    digest = hashlib.sha256()
    with open_binary(source, locator) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(source: DataSource, locator: Locator) -> int:
    """Return a 64-bit dHash for lightweight cross-split near-duplicate screening."""
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with open_binary(source, locator) as handle:
        with Image.open(handle) as image:
            resized = image.convert("L").resize((9, 8), resampling)
            if hasattr(resized, "get_flattened_data"):
                pixels = list(resized.get_flattened_data())
            else:
                pixels = list(resized.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def near_hash_cross_split_pairs(
    values: list[tuple[int, str, Locator]], max_distance: int = 3
) -> tuple[int, list[dict[str, Any]]]:
    """Find cross-split dHash neighbours using four 16-bit LSH bands.

    With a Hamming threshold of three, at least one of four bands must match exactly.
    This avoids an all-pairs comparison while preserving all candidates at this threshold.
    """
    band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
    prior: list[tuple[int, str, Locator]] = []
    count = 0
    examples: list[dict[str, Any]] = []
    for current_hash, current_split, current_path in values:
        candidate_indices: set[int] = set()
        for band in range(4):
            band_value = (current_hash >> (band * 16)) & 0xFFFF
            candidate_indices.update(band_index[(band, band_value)])
        for index in candidate_indices:
            other_hash, other_split, other_path = prior[index]
            if current_split == other_split:
                continue
            distance = (current_hash ^ other_hash).bit_count()
            if distance <= max_distance:
                count += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "distance": distance,
                            "left_split": other_split,
                            "left_path": str(other_path),
                            "right_split": current_split,
                            "right_path": str(current_path),
                        }
                    )
        new_index = len(prior)
        prior.append((current_hash, current_split, current_path))
        for band in range(4):
            band_value = (current_hash >> (band * 16)) & 0xFFFF
            band_index[(band, band_value)].append(new_index)
    return count, examples


def audit_schema(audit: Audit, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in splits.values() for row in rows]
    bad_top = 0
    bad_inputs = 0
    missing_labels = Counter()
    missing_review = 0
    review_status = Counter()

    for row in all_rows:
        if not all(isinstance(row.get(key), dict) for key in ("inputs", "labels", "meta")):
            bad_top += 1
            continue
        input_keys = set(row["inputs"])
        if input_keys != EXPECTED_INPUT_KEYS:
            bad_inputs += 1
        missing_labels.update(REQUIRED_LABEL_KEYS - set(row["labels"]))
        status = row["meta"].get("review_status")
        if status in (None, ""):
            missing_review += 1
        else:
            review_status[str(status).lower()] += 1

    audit.add("nested schema", bad_top == 0, f"bad rows={bad_top}/{len(all_rows)}")
    audit.add(
        "model input allow-list",
        bad_inputs == 0,
        f"rows whose input keys are not exactly {sorted(EXPECTED_INPUT_KEYS)}: {bad_inputs}",
    )
    audit.add(
        "required label fields",
        not missing_labels,
        "all present" if not missing_labels else counter_text(missing_labels),
    )
    audit.add("review_status present", missing_review == 0, f"missing={missing_review}")
    approved = review_status.get("approved", 0)
    audit.add(
        "approved-only publication gate",
        approved == len(all_rows),
        f"{counter_text(review_status)}; approved={approved}/{len(all_rows)}",
    )
    return {"review_status": dict(review_status)}


def audit_counts_and_labels(
    audit: Audit, splits: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    all_rows = [row for rows in splits.values() for row in rows]
    split_counts = {name: len(rows) for name, rows in splits.items()}
    audit.add(
        "reported split row counts",
        split_counts == EXPECTED_SPLIT_ROWS,
        f"observed={split_counts}; expected={EXPECTED_SPLIT_ROWS}",
    )
    audit.add(
        "reported total row count",
        len(all_rows) == EXPECTED_TOTAL_ROWS,
        f"observed={len(all_rows)}; expected={EXPECTED_TOTAL_ROWS}",
    )

    outcomes = Counter(row.get("labels", {}).get("outcome_label") for row in all_rows)
    failures = Counter(row.get("labels", {}).get("failure_type_4") for row in all_rows)
    actions = Counter(row.get("labels", {}).get("action_type") for row in all_rows)
    recoveries = Counter(row.get("labels", {}).get("recovery_success") for row in all_rows)
    attempted = sum(bool(row.get("labels", {}).get("recovery_attempted")) for row in all_rows)
    memories = Counter(row.get("labels", {}).get("memory_update_flag") for row in all_rows)

    audit.add("outcome label set", set(outcomes) == EXPECTED_OUTCOMES, counter_text(outcomes))
    audit.add(
        "failure label set", set(failures) == EXPECTED_FAILURE_TYPES, counter_text(failures)
    )
    audit.add("six action classes", set(actions) == EXPECTED_ACTIONS, counter_text(actions))
    audit.add(
        "handoff outcome counts",
        dict(outcomes) == EXPECTED_OUTCOME_COUNTS,
        f"observed={dict(outcomes)}; expected={EXPECTED_OUTCOME_COUNTS}",
    )
    audit.add(
        "handoff failure counts",
        dict(failures) == EXPECTED_FAILURE_COUNTS,
        f"observed={dict(failures)}; expected={EXPECTED_FAILURE_COUNTS}",
    )
    audit.add(
        "handoff action counts",
        dict(actions) == EXPECTED_ACTION_COUNTS,
        f"observed={dict(actions)}; expected={EXPECTED_ACTION_COUNTS}",
    )
    audit.add(
        "handoff recovery counts",
        attempted == 9_237 and recoveries.get(True, 0) == 3_153 and recoveries.get(False, 0) == 6_084,
        f"attempted={attempted}; success={recoveries.get(True, 0)}; "
        f"failure={recoveries.get(False, 0)}; expected=9237/3153/6084",
    )

    success_share = outcomes.get("SUCCESS", 0) / max(1, len(all_rows))
    audit.add(
        "outcome balance target",
        0.45 <= success_share <= 0.55,
        f"SUCCESS share={success_share:.2%}; target=45%-55%",
        severity="WARN",
    )
    largest_action = max(actions.values(), default=0) / max(1, len(all_rows))
    audit.add(
        "action dominance cap",
        largest_action <= 0.40,
        f"largest action share={largest_action:.2%}; cap=40%",
    )

    logic_errors = Counter()
    for row in all_rows:
        labels = row.get("labels", {})
        outcome = labels.get("outcome_label")
        failure = labels.get("failure_type_4")
        if outcome == "SUCCESS" and failure != "NONE":
            logic_errors["success_non_none_failure"] += 1
        if outcome == "FAILURE" and failure == "NONE":
            logic_errors["failure_none_type"] += 1
        if labels.get("recovery_attempted") and labels.get("recovery_success") is None:
            logic_errors["attempted_missing_result"] += 1
        if not labels.get("recovery_attempted") and labels.get("recovery_success") is not None:
            logic_errors["unattempted_has_result"] += 1
    audit.add(
        "label logic consistency",
        not logic_errors,
        "all consistent" if not logic_errors else counter_text(logic_errors),
    )

    return {
        "split_rows": split_counts,
        "total_rows": len(all_rows),
        "outcomes": dict(outcomes),
        "failure_types": dict(failures),
        "actions": dict(actions),
        "recovery_success": {str(k): v for k, v in recoveries.items()},
        "recovery_attempted": attempted,
        "memory_update_flag": {str(k): v for k, v in memories.items()},
    }


def audit_identifiers_and_splits(
    audit: Audit, splits: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    all_rows = [row for rows in splits.values() for row in rows]
    sample_key = select_meta_key(all_rows, ["sample_id", "row_id", "id"])
    task_key = select_meta_key(
        all_rows, ["trajectory_id", "task_id", "original_task_id", "episode_id"]
    )
    audit.add("sample identifier available", sample_key is not None, f"key={sample_key}")
    audit.add("task/trajectory identifier available", task_key is not None, f"key={task_key}")

    if sample_key:
        values = [row.get("meta", {}).get(sample_key) for row in all_rows]
        duplicates = len(values) - len(set(values))
        audit.add("unique sample identifiers", duplicates == 0, f"duplicates={duplicates}")

    domains = {
        split: {row.get("inputs", {}).get("website_domain") for row in rows}
        for split, rows in splits.items()
    }
    domain_overlap = {
        "train_val": len(domains["train"] & domains["val"]),
        "train_test": len(domains["train"] & domains["test"]),
        "val_test": len(domains["val"] & domains["test"]),
    }
    audit.add(
        "domain-disjoint splits",
        sum(domain_overlap.values()) == 0,
        f"overlap={domain_overlap}; counts={dict((k, len(v)) for k, v in domains.items())}",
    )
    domain_counts = {key: len(value) for key, value in domains.items()}
    audit.add(
        "handoff domain counts",
        domain_counts == EXPECTED_SPLIT_DOMAINS,
        f"observed={domain_counts}; expected={EXPECTED_SPLIT_DOMAINS}",
    )

    task_overlap: dict[str, int] | None = None
    if task_key:
        tasks = {
            split: {row.get("meta", {}).get(task_key) for row in rows}
            for split, rows in splits.items()
        }
        task_overlap = {
            "train_val": len(tasks["train"] & tasks["val"]),
            "train_test": len(tasks["train"] & tasks["test"]),
            "val_test": len(tasks["val"] & tasks["test"]),
        }
        audit.add(
            "task/trajectory-disjoint splits",
            sum(task_overlap.values()) == 0,
            f"key={task_key}; overlap={task_overlap}; "
            f"counts={dict((k, len(v)) for k, v in tasks.items())}",
        )
        task_counts = {key: len(value) for key, value in tasks.items()}
        audit.add(
            "handoff task/trajectory counts",
            task_counts == EXPECTED_SPLIT_TASKS,
            f"observed={task_counts}; expected={EXPECTED_SPLIT_TASKS}",
        )

    signatures: dict[str, set[str]] = defaultdict(set)
    signature_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            key = canonical(row.get("inputs"))
            signatures[key].add(canonical(row.get("labels")))
            signature_splits[key].add(split)
    cross_split_inputs = sum(len(parts) > 1 for parts in signature_splits.values())
    conflicts = sum(len(labels) > 1 for labels in signatures.values())
    audit.add("exact full-input split overlap", cross_split_inputs == 0, f"groups={cross_split_inputs}")
    audit.add("exact-input label conflicts", conflicts == 0, f"groups={conflicts}")

    return {
        "sample_id_key": sample_key,
        "task_id_key": task_key,
        "domain_overlap": domain_overlap,
        "task_overlap": task_overlap,
        "exact_input_cross_split_groups": cross_split_inputs,
        "exact_input_conflict_groups": conflicts,
    }


def audit_images(
    audit: Audit,
    source: DataSource,
    splits: dict[str, list[dict[str, Any]]],
    decode_sample: int,
    full_hash: bool,
) -> dict[str, Any]:
    references: dict[str, list[tuple[str, Locator | None]]] = defaultdict(list)
    missing: list[str] = []
    path_splits: dict[Locator, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            inputs = row.get("inputs", {})
            for field in ("state_before", "state_after"):
                ref = inputs.get(field)
                path = resolve_image(source, ref)
                references[split].append((str(ref), path))
                exists = path is not None and (
                    isinstance(source, ZipDataset) or (isinstance(path, Path) and path.is_file())
                )
                if not exists:
                    missing.append(f"{split}:{field}:{ref}")
                else:
                    assert path is not None
                    path_splits[path].add(split)

    audit.add(
        "all image references exist",
        not missing,
        f"references={sum(map(len, references.values()))}; missing={len(missing)}; "
        f"examples={missing[:5]}",
    )
    cross_split_paths = sum(len(parts) > 1 for parts in path_splits.values())
    audit.add("image-path split overlap", cross_split_paths == 0, f"paths={cross_split_paths}")

    valid_paths = sorted(path_splits)
    rng = random.Random(42)
    sampled = rng.sample(valid_paths, min(max(0, decode_sample), len(valid_paths)))
    corrupt = []
    modes = Counter()
    sizes = Counter()
    for path in sampled:
        try:
            with open_binary(source, path) as handle:
                with Image.open(handle) as image:
                    image.verify()
            with open_binary(source, path) as handle:
                with Image.open(handle) as image:
                    modes[image.mode] += 1
                    sizes[f"{image.width}x{image.height}"] += 1
        except Exception as exc:  # noqa: BLE001 - report corrupt data, do not hide it
            corrupt.append(f"{path}: {exc}")
    audit.add(
        "sampled image decode",
        not corrupt,
        f"decoded={len(sampled)}; corrupt={len(corrupt)}; examples={corrupt[:3]}",
    )

    hash_overlap = None
    near_hash_overlap = None
    near_hash_examples: list[dict[str, Any]] = []
    if full_hash and not missing:
        digest_splits: dict[str, set[str]] = defaultdict(set)
        perceptual_values: list[tuple[int, str, Locator]] = []
        for index, path in enumerate(valid_paths, start=1):
            digest_splits[sha256(source, path)].update(path_splits[path])
            path_split = sorted(path_splits[path])[0]
            perceptual_values.append((difference_hash(source, path), path_split, path))
            if index % 5000 == 0:
                print(f"Content-hashed {index:,}/{len(valid_paths):,} unique images")
        hash_overlap = sum(len(parts) > 1 for parts in digest_splits.values())
        audit.add("exact image-content split overlap", hash_overlap == 0, f"hashes={hash_overlap}")
        near_hash_overlap, near_hash_examples = near_hash_cross_split_pairs(perceptual_values)
        audit.add(
            "near-image perceptual split overlap",
            near_hash_overlap == 0,
            f"cross-split dHash pairs within Hamming distance <=3: {near_hash_overlap}; "
            f"examples={near_hash_examples[:3]}",
        )
    else:
        audit.skip(
            "exact image-content split overlap",
            "rerun with --full-image-hash for the final publication audit",
        )
        audit.skip(
            "near-image perceptual split overlap",
            "rerun with --full-image-hash for the dHash Hamming-distance audit",
        )

    return {
        "references": sum(map(len, references.values())),
        "unique_existing_paths": len(valid_paths),
        "missing": len(missing),
        "cross_split_paths": cross_split_paths,
        "decoded_sample": len(sampled),
        "corrupt_sample": len(corrupt),
        "sample_modes": dict(modes),
        "sample_sizes": dict(sizes),
        "cross_split_sha256_hashes": hash_overlap,
        "near_image_dhash_pairs": near_hash_overlap,
        "near_image_dhash_examples": near_hash_examples,
    }


def audit_shortcut_baselines(
    audit: Audit, splits: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        audit.add("shortcut baseline dependencies", False, str(exc))
        return {}

    def values(rows: list[dict[str, Any]], input_key: str | None, label_key: str):
        x = []
        y = []
        for row in rows:
            label = row.get("labels", {}).get(label_key)
            value = row.get("inputs", {}).get(input_key) if input_key else None
            if label is not None and (input_key is None or value not in (None, "")):
                x.append(value)
                y.append(label)
        return x, y

    train_text, train_action = values(splits["train"], "task_description", "action_type")
    test_text, test_action = values(splits["test"], "task_description", "action_type")
    text_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True),
        LinearSVC(class_weight="balanced"),
    )
    text_model.fit(train_text, train_action)
    action_pred = text_model.predict(test_text)
    action_acc = float(accuracy_score(test_action, action_pred))
    action_f1 = float(f1_score(test_action, action_pred, average="macro"))
    majority = max(Counter(test_action).values()) / len(test_action)
    audit.add(
        "task-text-only action blocker",
        action_acc < 0.90,
        f"test accuracy={action_acc:.4f}; macro-F1={action_f1:.4f}; "
        f"majority accuracy={majority:.4f}; blocker=0.90",
    )
    audit.add(
        "task-text-only action warning",
        action_acc <= majority + 0.20,
        f"improvement over majority={action_acc - majority:.4f}; warning threshold=+0.20",
        severity="WARN",
    )

    train_conf = []
    train_outcome = []
    test_conf = []
    test_outcome = []
    for split, target_x, target_y in (
        ("train", train_conf, train_outcome),
        ("test", test_conf, test_outcome),
    ):
        for row in splits[split]:
            labels = row.get("labels", {})
            confidence = labels.get("agent_confidence_before")
            outcome = labels.get("outcome_label")
            if confidence is not None and outcome in EXPECTED_OUTCOMES:
                target_x.append([float(confidence)])
                target_y.append(int(outcome == "SUCCESS"))
    conf_model = LogisticRegression(class_weight="balanced", random_state=42)
    conf_model.fit(train_conf, train_outcome)
    conf_pred = conf_model.predict(test_conf)
    conf_prob = conf_model.predict_proba(test_conf)[:, 1]
    conf_mcc = float(matthews_corrcoef(test_outcome, conf_pred))
    conf_auc = float(roc_auc_score(test_outcome, conf_prob))
    audit.add(
        "confidence-only outcome blocker",
        abs(conf_mcc) < 0.90,
        f"test MCC={conf_mcc:.4f}; ROC-AUC={conf_auc:.4f}; blocker=|MCC|>=0.90",
    )
    audit.add(
        "confidence-only outcome warning",
        abs(conf_mcc) < 0.70,
        f"test MCC={conf_mcc:.4f}; warning threshold=|MCC|>=0.70",
        severity="WARN",
    )
    return {
        "task_text_action": {
            "test_accuracy": action_acc,
            "test_macro_f1": action_f1,
            "majority_accuracy": majority,
        },
        "confidence_outcome": {"test_mcc": conf_mcc, "test_roc_auc": conf_auc},
    }


def main() -> int:
    args = parse_args()
    source = find_data_source(args.data_root)
    source_description = source.description if isinstance(source, ZipDataset) else str(source)
    print("DATA_SOURCE =", source_description)
    print("No KaggleHub download or ZIP extraction API is used. Files are read in place.")

    splits = {name: load_records(source, filename) for name, filename in SPLIT_FILES.items()}
    audit = Audit()
    if isinstance(source, ZipDataset):
        audit.add(
            "unique safe ZIP members",
            not source.duplicate_members,
            f"members={len(source.member_map)}; duplicate normalised names="
            f"{len(source.duplicate_members)}; prefix={source.prefix or '/'}",
        )
    report: dict[str, Any] = {
        "dataset_source": source_description,
        "source_type": "zip" if isinstance(source, ZipDataset) else "directory",
        "schema": audit_schema(audit, splits),
        "counts_and_labels": audit_counts_and_labels(audit, splits),
        "identifiers_and_splits": audit_identifiers_and_splits(audit, splits),
        "images": audit_images(
            audit,
            source,
            splits,
            decode_sample=args.decode_sample,
            full_hash=args.full_image_hash,
        ),
    }
    if args.adult_domain_policy_confirmed:
        audit.add(
            "adult-domain thesis protocol",
            True,
            "confirmed by --adult-domain-policy-confirmed; document the inclusion or quarantine decision",
        )
    else:
        audit.skip(
            "adult-domain thesis protocol",
            "the handoff reports 1,354 rows from 16 adult domains; confirm inclusion or quarantine in the thesis protocol",
        )
    if args.skip_baselines:
        audit.skip("shortcut baselines", "disabled by --skip-baselines")
        report["shortcut_baselines"] = None
    else:
        report["shortcut_baselines"] = audit_shortcut_baselines(audit, splits)

    counts = audit.summary()
    report["gate_counts"] = dict(counts)
    report["gates"] = [asdict(gate) for gate in audit.gates]
    report["publication_ready"] = counts.get("FAIL", 0) == 0 and counts.get("SKIP", 0) == 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FINAL SUMMARY ===")
    print(dict(counts))
    if report["publication_ready"]:
        print("PUBLICATION DATA GATES: PASS")
    else:
        print("PUBLICATION DATA GATES: NOT YET PASSING")
        for gate in audit.gates:
            if gate.status in {"FAIL", "SKIP"}:
                print(f"  - {gate.status}: {gate.name}: {gate.detail}")
    print("Report saved to:", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
