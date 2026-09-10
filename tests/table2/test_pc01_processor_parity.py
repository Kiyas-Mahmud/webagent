from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from web_agent.eval.table2.common import canonical_json_bytes
from web_agent.eval.table2.pc01_artifacts import (
    PC01_IMAGE_PROCESSOR_CLASS,
    PC01_MODEL_REVISION,
    PC01_PROCESSOR_CLASS,
    PC01_TRANSFORMERS_VERSION,
    POST_ACTION_FIELD_MAPPING,
    PRE_ACTION_FIELD_MAPPING,
    registered_pc01_training_source_manifest,
)
from web_agent.eval.table2.pc01_processor_parity import (
    PC01ProcessorParityError,
    build_pc01_processor_parity_receipt,
    current_pc01_processor_parity_source_identity,
    validate_pc01_processor_parity_receipt,
)
from web_agent.runtime.observation import ProcessorParityContract


@pytest.fixture(autouse=True)
def _registered_transformers_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic parity tests independent of optional ML installation."""

    monkeypatch.setattr(
        "web_agent.eval.table2.pc01_processor_parity.importlib_metadata.version",
        lambda package: (
            PC01_TRANSFORMERS_VERSION
            if package == "transformers"
            else pytest.fail(f"unexpected package version lookup: {package}")
        ),
    )


class _ImageProcessor:
    min_pixels = 50_176
    max_pixels = 200_704


class _DeterministicProcessor:
    image_processor = _ImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return json.dumps(messages, sort_keys=True, separators=(",", ":"))

    def __call__(self, *, text, images, return_tensors):
        assert return_tensors == "pt"
        image_payloads = [image.tobytes() for image in images]
        digest = hashlib.sha256(
            text[0].encode("utf-8") + b"\0" + b"\0".join(image_payloads)
        ).digest()
        return {
            "input_ids": torch.tensor([list(digest[:16])], dtype=torch.long),
            "attention_mask": torch.ones((1, 16), dtype=torch.long),
            "pixel_values": torch.tensor(
                [
                    [float(payload[0]), float(payload[1]), float(payload[2])]
                    for payload in image_payloads
                ],
                dtype=torch.float32,
            ),
            "image_grid_thw": torch.tensor(
                [[1, image.height, image.width] for image in images],
                dtype=torch.long,
            ),
        }


class _DriftingProcessor(_DeterministicProcessor):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        encoded = super().__call__(**kwargs)
        if self.calls > 3:
            encoded["input_ids"] = encoded["input_ids"].clone()
            encoded["input_ids"][0, 0] += 1
        return encoded


_ImageProcessor.__module__, _ImageProcessor.__qualname__ = (
    PC01_IMAGE_PROCESSOR_CLASS.rsplit(".", 1)
)
for _processor_type in (_DeterministicProcessor, _DriftingProcessor):
    _processor_type.__module__, _processor_type.__qualname__ = (
        PC01_PROCESSOR_CLASS.rsplit(".", 1)
    )


def _config() -> dict:
    return {
        "name": "Y_QWEN2VL_2B_GOLD_V2_8_DGX_FULL_SEED42",
        "backbone": {
            "path": "vlm",
            "family": "qwen2_vl",
            "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
            "revision": PC01_MODEL_REVISION,
            "trust_remote_code": False,
            "min_pixels": 50_176,
            "max_pixels": 200_704,
        },
    }


def _contract() -> ProcessorParityContract:
    return ProcessorParityContract(
        processor_class=PC01_PROCESSOR_CLASS,
        processor_revision=PC01_MODEL_REVISION,
        processor_config_sha256="a" * 64,
        pre_action_field_mapping=PRE_ACTION_FIELD_MAPPING,
        post_action_field_mapping=POST_ACTION_FIELD_MAPPING,
    )


def _implementation() -> dict:
    return {
        "transformers_version": PC01_TRANSFORMERS_VERSION,
        "processor_class": PC01_PROCESSOR_CLASS,
        "image_processor_class": PC01_IMAGE_PROCESSOR_CLASS,
        "use_fast": True,
    }


def _build(processor) -> dict:
    training_source_manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(registered_pc01_training_source_manifest())
    ).hexdigest()
    return build_pc01_processor_parity_receipt(
        config=_config(),
        processor=processor,
        processor_source="/test-only/local-snapshot",
        processor_contract=_contract(),
        implementation_identity=_implementation(),
        processor_artifact_manifest_sha256="b" * 64,
        base_snapshot_directory_payload_sha256="c" * 64,
        training_environment_sha256="d" * 64,
        training_environment_record_sha256="f" * 64,
        training_source_manifest_sha256=training_source_manifest_sha256,
        training_action_value_evidence_sha256="1" * 64,
        run_contract_sha256="e" * 64,
        training_processor_loader=lambda _config, _source: _DeterministicProcessor(),
    )


def _validate(receipt: dict) -> dict:
    return validate_pc01_processor_parity_receipt(
        receipt,
        resolved_config_record_sha256=receipt[
            "resolved_config_record_sha256"
        ],
        processor_contract_sha256=receipt["processor_contract_sha256"],
        processor_artifact_manifest_sha256=receipt[
            "processor_artifact_manifest_sha256"
        ],
        base_snapshot_directory_payload_sha256=receipt[
            "base_snapshot_directory_payload_sha256"
        ],
        training_environment_sha256=receipt["training_environment_sha256"],
        training_environment_record_sha256=receipt[
            "training_environment_record_sha256"
        ],
        training_source_manifest_sha256=receipt[
            "training_source_manifest_sha256"
        ],
        training_action_value_evidence_sha256=receipt[
            "training_action_value_evidence_sha256"
        ],
        run_contract_sha256=receipt["run_contract_sha256"],
    )


def test_processor_only_receipt_is_deterministic_and_covers_all_causal_streams() -> None:
    first = _build(_DeterministicProcessor())
    second = _build(_DeterministicProcessor())
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["parity_verified"] is True
    assert first["model_weights_loaded"] is False
    assert first["model_forward_executed"] is False
    assert first["network_access_used"] is False
    assert tuple(first["streams"]) == ("pre", "post", "recovery")
    assert set(first["action_class_matrix"]) == {
        "CLICK",
        "TYPE",
        "SELECT",
        "SCROLL",
        "NAVIGATE",
        "PRESS_KEY",
    }
    assert all(
        row["processor_action_value_omitted"] is True
        for row in first["action_class_matrix"].values()
    )
    for stream in first["streams"].values():
        assert stream["training_bundle_sha256"] == stream["runtime_bundle_sha256"]
        assert set(stream["tensors"]) == {
            "attention_mask",
            "image_counts",
            "image_grid_thw",
            "input_ids",
            "pixel_values",
        }


def test_processor_only_receipt_fails_on_runtime_tensor_drift() -> None:
    with pytest.raises(PC01ProcessorParityError, match="tensor bytes differ"):
        _build(_DriftingProcessor())


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("callable", "attacker.rebound_runtime_stream"),
        ("source_sha256", "0" * 64),
    ),
)
def test_parity_validator_rejects_rebound_live_source_identity(
    field: str,
    replacement: str,
) -> None:
    receipt = deepcopy(_build(_DeterministicProcessor()))
    assert receipt["source_identity"] == (
        current_pc01_processor_parity_source_identity()
    )
    receipt["source_identity"]["runtime_stream"][field] = replacement

    with pytest.raises(
        PC01ProcessorParityError,
        match="live registered callables",
    ):
        _validate(receipt)
