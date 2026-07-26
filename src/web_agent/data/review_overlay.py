"""Apply a passed human-review reconciliation as an in-memory data overlay.

The original split JSON files remain untouched. Every proposed value is checked
against the corresponding original value before it is applied, so a ledger
cannot silently target the wrong dataset version.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROPOSAL_TO_LABEL = {
    "proposed_bbox": ("original_bbox", "action_target_bbox"),
    "proposed_action_type": ("original_action_type", "action_type"),
    "proposed_outcome_label": ("original_outcome_label", "outcome_label"),
    "proposed_failure_type": ("original_failure_type", "failure_type_4"),
    "proposed_recovery_attempted": (
        "original_recovery_attempted",
        "recovery_attempted",
    ),
    "proposed_recovery_strategy": (
        "original_recovery_strategy",
        "recovery_strategy",
    ),
    "proposed_recovery_success": (
        "original_recovery_success",
        "recovery_success",
    ),
}
EXCLUDED_DECISIONS = {"reject_recollect", "quarantine_policy"}
RETAINED_DECISIONS = {
    "approve_no_change",
    "approve_after_correction",
    "mask_bbox_no_evidence",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_view(record: dict) -> tuple[dict, dict, dict]:
    if "inputs" in record and "labels" in record:
        return record["inputs"], record["labels"], record.get("meta", {})
    return record, record, record


def _sample_identity(record: dict, position: int) -> tuple[str, str, str]:
    _, _, meta = _row_view(record)
    sample_id = str(meta.get("sample_id") or "")
    task_id = str(meta.get("task_id") or meta.get("original_task_id") or "")
    step = meta.get("step_index", meta.get("step", position))
    if not sample_id or not task_id:
        raise ValueError(
            f"record at position {position} lacks sample_id or task_id"
        )
    return sample_id, task_id, str(step)


def _boolean(value: object, *, allow_null: bool = False) -> bool | None:
    if value is None and allow_null:
        return None
    if isinstance(value, bool):
        return value
    cleaned = str(value).strip().lower()
    if allow_null and cleaned in {"", "null", "none"}:
        return None
    if cleaned == "true":
        return True
    if cleaned == "false":
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _bbox(value: object) -> dict[str, float] | None:
    if value in (None, "", "null"):
        return None
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError(f"bbox must be an object, got {parsed!r}")
    try:
        result = {
            name: float(parsed[name])
            for name in ("x", "y", "width", "height")
        }
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"invalid bbox object {parsed!r}") from None
    if not all(math.isfinite(number) for number in result.values()):
        raise ValueError(f"bbox contains non-finite values: {parsed!r}")
    return result


def _normalised(value: object, label_field: str) -> object:
    if label_field == "action_target_bbox":
        return _bbox(value)
    if label_field == "recovery_attempted":
        return _boolean(value)
    if label_field == "recovery_success":
        return _boolean(value, allow_null=True)
    return str(value if value is not None else "")


def _validate_labels(labels: dict, sample_id: str) -> None:
    outcome = labels.get("outcome_label")
    failure = labels.get("failure_type_4")
    if (
        outcome == "SUCCESS"
        and failure != "NONE"
    ) or (
        outcome == "FAILURE"
        and failure == "NONE"
    ):
        raise ValueError(
            f"{sample_id} has inconsistent outcome/failure labels after overlay"
        )

    attempted_value = labels.get("recovery_attempted")
    success = labels.get("recovery_success")
    attempted = (
        _boolean(attempted_value)
        if attempted_value is not None
        else success is not None
    )
    strategy = labels.get("recovery_strategy")
    if not attempted and (strategy != "NONE" or success is not None):
        raise ValueError(
            f"{sample_id} has inconsistent non-attempted recovery labels"
        )
    if attempted and (
        strategy in {None, "", "NONE"} or not isinstance(success, bool)
    ):
        raise ValueError(
            f"{sample_id} has incomplete attempted-recovery labels"
        )


@dataclass(frozen=True)
class ReviewOverlay:
    root: Path
    report: dict
    corrections: tuple[dict[str, str], ...]
    dispositions: tuple[dict[str, str], ...]

    @classmethod
    def load(cls, root: str | Path) -> "ReviewOverlay":
        source = Path(root).resolve()
        report_path = source / "review_reconciliation_report.json"
        correction_path = source / "approved_corrections.csv"
        disposition_path = source / "finalized_dispositions.csv"
        for path in (report_path, correction_path, disposition_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        required_report = {
            "status": "PASS",
            "controlled_mini_permitted": True,
            "publication_ready": False,
            "test_rows_read": 0,
            "source_records_mutated": False,
        }
        mismatches = {
            key: {"expected": expected, "actual": report.get(key)}
            for key, expected in required_report.items()
            if report.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"reconciliation report is not an eligible overlay: {mismatches}"
            )

        for filename, path in (
            ("approved_corrections.csv", correction_path),
            ("finalized_dispositions.csv", disposition_path),
        ):
            expected = (
                report.get("outputs", {})
                .get(filename, {})
                .get("sha256")
            )
            actual = _sha256(path)
            if not expected or actual != expected:
                raise ValueError(
                    f"{filename} SHA-256 does not match reconciliation report"
                )

        corrections = tuple(_read_csv(correction_path))
        dispositions = tuple(_read_csv(disposition_path))
        if len(corrections) != report.get("approved_corrections"):
            raise ValueError("approved correction count does not match report")
        if len(dispositions) != report.get("finalized_dispositions"):
            raise ValueError("finalized disposition count does not match report")
        return cls(source, report, corrections, dispositions)

    def provenance(self) -> dict:
        return {
            "enabled": True,
            "reconciliation_report_sha256": _sha256(
                self.root / "review_reconciliation_report.json"
            ),
            "status": self.report["status"],
            "publication_ready": False,
            "test_rows_read": 0,
            "approved_corrections": len(self.corrections),
            "finalized_dispositions": len(self.dispositions),
            "decision_targets": self.report.get("decision_targets", {}),
            "artifact_sha256": {
                filename: self.report["outputs"][filename]["sha256"]
                for filename in (
                    "approved_corrections.csv",
                    "finalized_dispositions.csv",
                )
            },
        }

    def apply(
        self,
        records: Sequence[dict],
        *,
        split: str,
    ) -> tuple[list[dict], dict]:
        if split not in {"train", "val"}:
            raise ValueError("review overlay may be applied only to train or val")

        record_map: dict[str, tuple[int, dict, str, str]] = {}
        for position, record in enumerate(records):
            sample_id, task_id, step = _sample_identity(record, position)
            if sample_id in record_map:
                raise ValueError(f"duplicate sample_id in {split}: {sample_id}")
            record_map[sample_id] = (position, record, task_id, step)

        dispositions: dict[tuple[str, str], dict[str, str]] = {}
        for row in self.dispositions:
            row_split = str(row.get("split") or "")
            if row_split not in {"train", "val"}:
                raise ValueError(
                    f"disposition targets forbidden split {row_split!r}"
                )
            key = (str(row.get("queue_kind") or ""), str(row.get("sample_id") or ""))
            if key in dispositions:
                raise ValueError(f"duplicate disposition: {key}")
            decision = str(row.get("final_review_decision") or "")
            if decision not in RETAINED_DECISIONS | EXCLUDED_DECISIONS:
                raise ValueError(f"unresolved disposition {decision!r} for {key}")
            dispositions[key] = row

        corrections_by_sample: dict[str, list[dict[str, str]]] = {}
        for correction in self.corrections:
            row_split = str(correction.get("split") or "")
            if row_split not in {"train", "val"}:
                raise ValueError(
                    f"correction targets forbidden split {row_split!r}"
                )
            queue_kind = str(correction.get("queue_kind") or "")
            sample_id = str(correction.get("sample_id") or "")
            disposition = dispositions.get((queue_kind, sample_id))
            if (
                disposition is None
                or disposition.get("final_review_decision")
                != "approve_after_correction"
            ):
                raise ValueError(
                    f"correction lacks approved disposition: {queue_kind}:{sample_id}"
                )
            corrections_by_sample.setdefault(sample_id, []).append(correction)

        excluded = {
            str(row["sample_id"])
            for row in self.dispositions
            if row.get("split") == split
            and row.get("final_review_decision") in EXCLUDED_DECISIONS
        }
        missing_targets = sorted(
            sample_id
            for sample_id in (
                {
                    str(row["sample_id"])
                    for row in self.dispositions
                    if row.get("split") == split
                }
                | {
                    sample_id
                    for sample_id, rows in corrections_by_sample.items()
                    if any(row.get("split") == split for row in rows)
                }
            )
            if sample_id not in record_map
        )
        if missing_targets:
            raise ValueError(
                f"overlay targets missing from {split}: {missing_targets[:10]}"
            )

        output: list[dict] = []
        applied_fields: dict[str, int] = {}
        corrected_samples = 0
        for position, record in enumerate(records):
            sample_id, task_id, step = _sample_identity(record, position)
            if sample_id in excluded:
                continue
            applicable = [
                row
                for row in corrections_by_sample.get(sample_id, [])
                if row.get("split") == split
            ]
            if not applicable:
                output.append(record)
                continue

            updated = copy.deepcopy(record)
            _, labels, meta = _row_view(updated)
            changed: dict[str, object] = {}
            for correction in applicable:
                if (
                    str(correction.get("task_id") or "") != task_id
                    or str(correction.get("step_index") or "") != step
                ):
                    raise ValueError(
                        f"correction identity mismatch for {split}:{sample_id}"
                    )
                for proposal_field, (
                    original_field,
                    label_field,
                ) in PROPOSAL_TO_LABEL.items():
                    proposed_raw = correction.get(proposal_field)
                    if proposed_raw in (None, ""):
                        continue
                    proposed = _normalised(proposed_raw, label_field)
                    expected_original = _normalised(
                        correction.get(original_field),
                        label_field,
                    )
                    actual_original = _normalised(
                        labels.get(label_field),
                        label_field,
                    )
                    if actual_original != expected_original:
                        raise ValueError(
                            f"{split}:{sample_id} {label_field} original value "
                            "does not match correction ledger"
                        )
                    if (
                        label_field in changed
                        and changed[label_field] != proposed
                    ):
                        raise ValueError(
                            f"conflicting corrections for {split}:{sample_id} "
                            f"field {label_field}"
                        )
                    changed[label_field] = proposed

            if not changed:
                raise ValueError(
                    f"approved correction for {split}:{sample_id} changes no fields"
                )
            labels.update(changed)
            _validate_labels(labels, sample_id)
            meta["targeted_review_overlay"] = {
                "status": "corrected",
                "fields": sorted(changed),
            }
            for field in changed:
                applied_fields[field] = applied_fields.get(field, 0) + 1
            corrected_samples += 1
            output.append(updated)

        report = {
            "split": split,
            "source_rows": len(records),
            "output_rows": len(output),
            "excluded_rows": len(records) - len(output),
            "corrected_samples": corrected_samples,
            "applied_field_counts": dict(sorted(applied_fields.items())),
            "test_rows_read": 0,
            "source_records_mutated": False,
        }
        return output, report
