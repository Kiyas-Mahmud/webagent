"""Bounding-box geometry shared by the head, loss, and diagnostic stages.

Dataset labels use normalized top-left ``xywh``.  Recovery-v2.2 predicts the
DETR-style centre representation ``cxcywh`` internally, applies coordinate L1
there, and converts to corners for GIoU.  Public model predictions remain
top-left ``xywh`` so existing metrics and saved-result schemas stay stable.
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
