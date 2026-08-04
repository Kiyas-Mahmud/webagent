"""Coordinate utilities for spatial-token grounding.

Qwen exposes image tokens in raster order after spatial merging.  These helpers
attach explicit normalized patch centres to those tokens and convert a learned
attention distribution into a differentiable image coordinate.
"""

from __future__ import annotations

import torch


def normalized_spatial_coordinates(
    spatial_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Return ``[batch, sequence, 2]`` normalized ``(x, y)`` patch centres.

    ``image_grid_thw`` is concatenated by the collator. Each stream has the same
    number of images per row (one pre-action image, two post/recovery images),
    so its first dimension can be divided evenly across the batch.
    """
    if spatial_mask.ndim != 2:
        raise ValueError("spatial_mask must have shape [batch, sequence]")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [images, 3]")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    batch_size, sequence_length = spatial_mask.shape
    if batch_size == 0 or image_grid_thw.shape[0] % batch_size:
        raise ValueError("image grids must divide evenly across the batch")
    images_per_row = image_grid_thw.shape[0] // batch_size
    coordinates = torch.zeros(
        (batch_size, sequence_length, 2),
        dtype=torch.float32,
        device=spatial_mask.device,
    )

    for row in range(batch_size):
        row_coordinates = []
        start = row * images_per_row
        for grid in image_grid_thw[start:start + images_per_row]:
            time, height, width = (int(value) for value in grid.tolist())
            if time <= 0 or height <= 0 or width <= 0:
                raise ValueError(f"invalid image grid: {(time, height, width)!r}")
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError(
                    "image grid dimensions must be divisible by spatial_merge_size"
                )
            merged_height = height // spatial_merge_size
            merged_width = width // spatial_merge_size
            y = (
                torch.arange(merged_height, device=spatial_mask.device).float()
                + 0.5
            ) / merged_height
            x = (
                torch.arange(merged_width, device=spatial_mask.device).float()
                + 0.5
            ) / merged_width
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid_coordinates = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            row_coordinates.append(grid_coordinates.repeat(time, 1))

        row_coordinates = torch.cat(row_coordinates, dim=0)
        token_positions = spatial_mask[row].bool()
        token_count = int(token_positions.sum())
        if row_coordinates.shape[0] != token_count:
            raise ValueError(
                "Qwen spatial-token/grid mismatch: "
                f"grid produced {row_coordinates.shape[0]} coordinates but "
                f"the token mask selected {token_count} positions"
            )
        coordinates[row, token_positions] = row_coordinates
    return coordinates


def normalized_fixed_square_coordinates(
    spatial_mask: torch.Tensor,
    image_counts: torch.Tensor,
    *,
    tokens_per_image: int,
) -> torch.Tensor:
    """Return raster coordinates for fixed-square image-token backbones.

    Hugging Face InternVL represents every image with a fixed number of image
    placeholder tokens.  The processor may concatenate one pre-action image or
    two post/recovery images, so the collator preserves the per-row image count
    and this function repeats the same normalized square grid for each image.
    """
    if spatial_mask.ndim != 2:
        raise ValueError("spatial_mask must have shape [batch, sequence]")
    if image_counts.ndim not in {1, 2}:
        raise ValueError("image_counts must have shape [batch] or [batch, 1]")
    counts = image_counts.reshape(-1).long()
    if counts.shape[0] != spatial_mask.shape[0]:
        raise ValueError("image_counts must contain one value per batch row")
    side = int(tokens_per_image ** 0.5)
    if side <= 0 or side * side != tokens_per_image:
        raise ValueError("tokens_per_image must be a positive perfect square")

    y = (torch.arange(side, device=spatial_mask.device).float() + 0.5) / side
    x = (torch.arange(side, device=spatial_mask.device).float() + 0.5) / side
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image_coordinates = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    coordinates = torch.zeros(
        (*spatial_mask.shape, 2),
        dtype=torch.float32,
        device=spatial_mask.device,
    )
    for row, image_count in enumerate(counts.tolist()):
        if image_count <= 0:
            raise ValueError("every multimodal row must contain at least one image")
        row_coordinates = image_coordinates.repeat(image_count, 1)
        token_positions = spatial_mask[row].bool()
        token_count = int(token_positions.sum())
        if token_count != row_coordinates.shape[0]:
            raise ValueError(
                "fixed-grid spatial-token mismatch: "
                f"expected {row_coordinates.shape[0]} tokens for {image_count} "
                f"images but selected {token_count}"
            )
        coordinates[row, token_positions] = row_coordinates
    return coordinates


def spatial_soft_argmax(
    attention: torch.Tensor,
    coordinates: torch.Tensor,
    spatial_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert spatial attention into a normalized centre and entropy.

    The weights are renormalized after masking so attention dropout cannot move
    the centre outside the convex hull of valid patch coordinates.
    """
    if attention.shape != spatial_mask.shape:
        raise ValueError("attention and spatial_mask must have identical shapes")
    if coordinates.shape != (*attention.shape, 2):
        raise ValueError("coordinates must have shape [batch, sequence, 2]")
    weights = attention.float().masked_fill(~spatial_mask.bool(), 0.0).clamp_min(0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    centre = (weights.unsqueeze(-1) * coordinates.float()).sum(dim=1)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    return centre.clamp(0.0, 1.0), entropy
