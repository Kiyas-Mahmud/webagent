"""Bounding-box geometry shared by the head, loss, and diagnostic stages.

Dataset labels use normalized top-left ``xywh``.  Recovery-v2.2 predicts the
DETR-style centre representation ``cxcywh`` internally, applies coordinate L1
there, and converts to corners for GIoU.  Public model predictions remain
top-left ``xywh`` so existing metrics and saved-result schemas stay stable.

Recovery-v2.4 keeps the centre path unchanged but predicts width/height in log
space.  Its direct log-size loss is intentionally computed from the unclamped
logits: unlike a saturated sigmoid, that path keeps a useful gradient when a
decoded size is extremely small.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert top-left ``[x, y, width, height]`` to centre ``cxcywh``."""
    xy = boxes[..., :2]
    wh = boxes[..., 2:]
    return torch.cat([xy + 0.5 * wh, wh], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert centre ``cxcywh`` to un-clipped corner ``xyxy``."""
    center = boxes[..., :2]
    half_size = 0.5 * boxes[..., 2:]
    return torch.cat([center - half_size, center + half_size], dim=-1)


def cxcywh_to_bounded_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Decode centre boxes to valid normalized top-left ``xywh`` for metrics."""
    corners = cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
    xy = corners[..., :2]
    wh = (corners[..., 2:] - xy).clamp(min=0.0, max=1.0)
    return torch.cat([xy, wh], dim=-1)


def generalized_iou_cxcywh(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-row generalized IoU for normalized centre-format boxes."""
    pred = cxcywh_to_xyxy(prediction.float())
    true = cxcywh_to_xyxy(target.float())
    inter_xy1 = torch.maximum(pred[..., :2], true[..., :2])
    inter_xy2 = torch.minimum(pred[..., 2:], true[..., 2:])
    inter_wh = (inter_xy2 - inter_xy1).clamp(min=0.0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]

    pred_wh = (pred[..., 2:] - pred[..., :2]).clamp(min=0.0)
    true_wh = (true[..., 2:] - true[..., :2]).clamp(min=0.0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    union = (pred_area + true_area - intersection).clamp(min=1e-7)
    iou = intersection / union

    enclosing_xy1 = torch.minimum(pred[..., :2], true[..., :2])
    enclosing_xy2 = torch.maximum(pred[..., 2:], true[..., 2:])
    enclosing_wh = (enclosing_xy2 - enclosing_xy1).clamp(min=0.0)
    enclosing = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


def detr_bbox_loss_terms(
    prediction_cxcywh: torch.Tensor,
    target_xywh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DETR-style coordinate L1 and GIoU losses without coefficients.

    L1 is summed across the four coordinates and averaged across rows, matching
    the normalization used by DETR for one matched box per example.  Keeping the
    two terms separate makes their relative scale visible in every report.
    """
    target_cxcywh = xywh_to_cxcywh(target_xywh.float())
    prediction_cxcywh = prediction_cxcywh.float()
    l1 = F.l1_loss(
        prediction_cxcywh,
        target_cxcywh,
        reduction="none",
    ).sum(dim=-1).mean()
    giou = (1.0 - generalized_iou_cxcywh(
        prediction_cxcywh,
        target_cxcywh,
    )).mean()
    return l1, giou


def detr_bbox_log_size_loss_terms(
    prediction_cxcywh: torch.Tensor,
    prediction_log_wh: torch.Tensor,
    target_xywh: torch.Tensor,
    *,
    minimum_target_size: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centre L1, log-size SmoothL1, and GIoU losses.

    The target boxes have already passed the strict geometry audit, so their
    sizes are positive.  ``clamp_min`` is only a final numerical guard before
    ``log``.  Crucially, the size loss reads the *unclamped* predicted log-size;
    therefore its gradient cannot disappear merely because the public decoded
    width or height is very small.
    """
    if prediction_log_wh.shape != prediction_cxcywh[..., 2:].shape:
        raise ValueError(
            "prediction_log_wh must match the two cxcywh size coordinates"
        )
    target_cxcywh = xywh_to_cxcywh(target_xywh.float())
    prediction_cxcywh = prediction_cxcywh.float()
    prediction_log_wh = prediction_log_wh.float()
    centre_l1 = F.l1_loss(
        prediction_cxcywh[..., :2],
        target_cxcywh[..., :2],
        reduction="none",
    ).sum(dim=-1).mean()
    target_log_wh = torch.log(
        target_cxcywh[..., 2:].clamp_min(float(minimum_target_size))
    )
    log_size_smooth_l1 = F.smooth_l1_loss(
        prediction_log_wh,
        target_log_wh,
        reduction="none",
    ).sum(dim=-1).mean()
    giou = (1.0 - generalized_iou_cxcywh(
        prediction_cxcywh,
        target_cxcywh,
    )).mean()
    return centre_l1, log_size_smooth_l1, giou


def bbox_attention_target_distribution(
    spatial_coords: torch.Tensor,
    spatial_mask: torch.Tensor,
    target_xywh: torch.Tensor,
    *,
    minimum_scale: float = 1e-4,
) -> torch.Tensor:
    """Build a soft target over patches from each supervised target box.

    The Gaussian is centred on the target box and uses half its width/height as
    the axis scales. This makes small controls select their nearest patch while
    larger elements supervise a broader region. Invalid/padded tokens receive
    exactly zero probability.
    """
    if spatial_coords.ndim != 3 or spatial_coords.shape[-1] != 2:
        raise ValueError("spatial_coords must have shape [batch, sequence, 2]")
    if spatial_mask.shape != spatial_coords.shape[:2]:
        raise ValueError("spatial_mask must match the coordinate batch/sequence")
    if target_xywh.shape != (spatial_coords.shape[0], 4):
        raise ValueError("target_xywh must have shape [batch, 4]")
    if not spatial_mask.bool().any(dim=1).all().item():
        raise ValueError("every row needs at least one valid spatial token")

    target_cxcywh = xywh_to_cxcywh(target_xywh.float())
    centre = target_cxcywh[:, None, :2]
    scale = (0.5 * target_cxcywh[:, None, 2:]).clamp_min(
        float(minimum_scale)
    )
    standardized = (spatial_coords.float() - centre) / scale
    logits = -0.5 * standardized.square().sum(dim=-1)
    logits = logits.masked_fill(~spatial_mask.bool(), float("-inf"))
    return torch.softmax(logits, dim=-1)


def bbox_attention_kl_loss(
    predicted_attention: torch.Tensor,
    spatial_coords: torch.Tensor,
    spatial_mask: torch.Tensor,
    target_xywh: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalized KL(target || predicted) for direct patch supervision.

    Dividing each row by ``log(valid_patch_count)`` keeps this auxiliary term on
    a comparable scale across Qwen's dynamic image resolutions. The prediction
    is renormalized after masking, so only real image tokens participate.
    """
    if predicted_attention.shape != spatial_mask.shape:
        raise ValueError("predicted attention and spatial_mask must match")
    mask = spatial_mask.bool()
    prediction = predicted_attention.float().masked_fill(~mask, 0.0)
    prediction = prediction.clamp_min(0.0)
    prediction = prediction / prediction.sum(dim=-1, keepdim=True).clamp_min(
        float(epsilon)
    )
    target = bbox_attention_target_distribution(
        spatial_coords,
        mask,
        target_xywh,
    )
    per_row = (
        target
        * (
            torch.log(target + float(epsilon))
            - torch.log(prediction + float(epsilon))
        )
    ).sum(dim=-1)
    normalizer = torch.log(mask.sum(dim=-1).float()).clamp_min(1.0)
    return (per_row / normalizer).mean()
