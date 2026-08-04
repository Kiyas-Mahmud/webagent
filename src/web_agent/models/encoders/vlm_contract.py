"""Runtime contracts for the supported unified vision-language families.

The Gold dataset and trainer are shared across backbones, but Hugging Face VLM
processors do not emit identical tensors.  Keeping those differences in this
small registry prevents model-name conditionals from leaking through the data,
collation, and encoder layers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VLMContract:
    """Processor/model tensor contract for one VLM family."""

    family: str
    required_input_keys: tuple[str, ...]
    optional_input_keys: tuple[str, ...]
    sequence_input_keys: tuple[str, ...]
    concat_input_keys: tuple[str, ...]
    coordinate_mode: str
    processor_uses_pixel_bounds: bool
    requires_single_patch_images: bool

    @property
    def model_input_keys(self) -> tuple[str, ...]:
        return self.required_input_keys + self.optional_input_keys


_CONTRACTS = {
    "qwen2_vl": VLMContract(
        family="qwen2_vl",
        required_input_keys=(
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        ),
        optional_input_keys=("mm_token_type_ids",),
        sequence_input_keys=(
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
        ),
        concat_input_keys=("pixel_values", "image_grid_thw"),
        coordinate_mode="qwen_grid",
        processor_uses_pixel_bounds=True,
        requires_single_patch_images=False,
    ),
    "internvl_hf": VLMContract(
        family="internvl_hf",
        required_input_keys=("input_ids", "attention_mask", "pixel_values"),
        optional_input_keys=("image_sizes",),
        sequence_input_keys=("input_ids", "attention_mask"),
        concat_input_keys=("pixel_values", "image_sizes"),
        coordinate_mode="fixed_square",
        processor_uses_pixel_bounds=False,
        requires_single_patch_images=True,
    ),
}


def infer_vlm_family(backbone: dict) -> str:
    """Return the explicit family, with a safe legacy Qwen inference fallback."""
    configured = backbone.get("family")
    if configured:
        return str(configured)
    model_name = str(backbone.get("vlm_model", "")).lower()
    if "qwen" in model_name and "vl" in model_name:
        return "qwen2_vl"
    raise ValueError(
        "backbone.family is required for non-Qwen VLMs; supported values are "
        f"{sorted(_CONTRACTS)}"
    )


def get_vlm_contract(cfg: dict) -> VLMContract:
    """Resolve and validate the configured unified-VLM input contract."""
    if cfg.get("backbone", {}).get("path") != "vlm":
        raise ValueError("a VLM contract requires backbone.path='vlm'")
    family = infer_vlm_family(cfg["backbone"])
    try:
        return _CONTRACTS[family]
    except KeyError as exc:
        raise ValueError(
            f"unsupported backbone.family {family!r}; expected one of "
            f"{sorted(_CONTRACTS)}"
        ) from exc
