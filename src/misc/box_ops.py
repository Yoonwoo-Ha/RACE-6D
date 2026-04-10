"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision
from torch import Tensor 
from typing import List, Tuple


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


# elementwise
def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Args:
        boxes1, [N, 4]
        boxes2, [N, 4]
    Returns:
        iou, [N, ]
        union, [N, ]
    """
    area1 = torchvision.ops.box_area(boxes1) # [N, ]
    area2 = torchvision.ops.box_area(boxes2) # [N, ]
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N, ]
    union = area1 + area2 - inter
    iou = inter / union
    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Args:
        boxes1, [N, 4] with [x1, y1, x2, y2]
        boxes2, [N, 4] with [x1, y1, x2, y2]
    Returns:
        giou, [N, ]
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2]) # [N, 2]
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:]) # [N, 2]
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    area = wh[:, 0] * wh[:, 1]
    return iou - (area - union) / area


def check_point_inside_box(points: Tensor, boxes: Tensor, eps=1e-9) -> Tensor:
    """
    Args:
        points, [K, 2], (x, y)
        boxes, [N, 4], (x1, y1, y2, y2)
    Returns:
        Tensor (bool), [K, N]
    """
    x, y = [p.unsqueeze(-1) for p in points.unbind(-1)]
    x1, y1, x2, y2 = [x.unsqueeze(0) for x in boxes.unbind(-1)]

    l = x - x1
    t = y - y1 
    r = x2 - x
    b = y2 - y
    
    ltrb = torch.stack([l, t, r, b], dim=-1)
    mask = ltrb.min(dim=-1).values > eps

    return mask


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    """
    Args:
        boxes, [N, 4], (x1, y1, x2, y2)
        points, [N, 2], (x, y)
    Returns:
        Tensor (N, 4), (l, t, r, b)
    """
    x1y1, x2y2 = torch.split(boxes, 2, dim=-1)
    lt = points - x1y1
    rb = x2y2 - points
    return torch.concat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    """
    Args:
        points (Tensor), [N, 2], (x, y)
        distances (Tensor), [N, 4], (l, t, r, b)
    Returns:
        boxes (Tensor),  (N, 4), (x1, y1, x2, y2)
    """
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = -lt + points
    x2y2 = rb + points
    boxes = torch.concat([x1y1, x2y2], dim=-1)
    return boxes

def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from center (cx, cy, w, h) to corner (x1, y1, x2, y2) format.
    
    Args:
        x (torch.Tensor): Tensor of shape (..., 4), where the last dimension contains (cx, cy, w, h).
    
    Returns:
        torch.Tensor: Tensor of shape (..., 4), where the last dimension contains (x1, y1, x2, y2).
    """
    # Unbind input tensor into components
    x_c, y_c, w, h = x.unbind(-1)

    # Ensure width and height are non-negative (clamp to prevent invalid values)
    w = w.clamp(min=0)
    h = h.clamp(min=0)
    
    # Compute corner coordinates
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h

    # Stack results back together
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_mxmywh_to_xyxy(x):
    x_m, y_m, w, h = x.unbind(-1)
    b = [x_m, y_m,
         x_m + w, y_m + h]
    return torch.stack(b, dim=-1)

def box_xyxy_to_mxmywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_mxmywh_to_cxcywh(x):
    x_m, y_m, w, h = x.unbind(-1)
    b = [x_m + 0.5 * w, y_m + 0.5 * h,
         w, h]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_mxmywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h,
         w, h]
    return torch.stack(b, dim=-1)