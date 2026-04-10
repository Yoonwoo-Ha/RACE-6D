"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.
---------------------------------------------------------------------
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import torch
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
import torchvision

from ...core import register


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """Generalized IoU from https://giou.stanford.edu/

    Args:
        boxes1: [N, 4] xyxy
        boxes2: [N, 4] xyxy (same N, pairwise)

    Returns:
        giou: [N, N] generalized iou matrix
    """
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


@register()
class HungarianMatcher(torch.nn.Module):
    """
    GPU-Util Hungarian Matcher for RACE-6D (bbox mode).

    Uses classification + bbox L1 + GIoU costs for matching.
    """
    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0, iou_threshold=0.0):
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict.get('cost_bbox', 5.0)
        self.cost_giou = weight_dict.get('cost_giou', 2.0)

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        # ---- Classification probabilities ----
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))   # [B*Nq, C]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Nq, C]

        # pred_boxes: [B, Nq, 4] (cxcywh, normalized)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Nq, 4]

        # Concatenated GT
        tgt_ids = torch.cat([v["labels"] for v in targets])  # [B*Nt]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [B*Nt, 4]

        # ---- Classification cost ----
        if self.use_focal_loss:
            prob = out_prob[:, tgt_ids]
            neg_cost = (1 - self.alpha) * (prob ** self.gamma) * (-(1 - prob + 1e-8).log())
            pos_cost = self.alpha * ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
            cost_class = pos_cost - neg_cost  # [B*Nq, B*Nt]
        else:
            cost_class = -out_prob[:, tgt_ids]  # [B*Nq, B*Nt]

        # ---- BBox L1 cost ----
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*Nq, B*Nt]

        # ---- GIoU cost ----
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)  # [B*Nq, B*Nt]

        # ---- Total cost ----
        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou)  # [B*Nq, B*Nt]

        # ---- Group-wise split and solve assignment ----
        sizes = [len(v["labels"]) for v in targets]
        C = C.view(bs, num_queries, -1)  # [B, G*N, total_targets]

        g_num_queries = num_queries // group_detr
        C_groups = C.split(g_num_queries, dim=1)  # G개의 [B, N, total_targets]

        all_indices = None
        for g_i in range(group_detr):
            C_g = C_groups[g_i]  # [B, N, total_targets]
            C_g_split = C_g.split(sizes, -1)

            batch_costs = []
            for i in range(bs):
                cost_matrix = C_g_split[i][i]  # [N, size_i]
                batch_costs.append(cost_matrix)

            max_size = max(sizes) if sizes else 1
            padded_costs = []
            for i, cost_matrix in enumerate(batch_costs):
                num_q, size_i = cost_matrix.shape
                pad_cols = max_size - size_i
                if pad_cols > 0:
                    padded = torch.cat(
                        [cost_matrix, torch.full((num_q, pad_cols), 1e9, device=device)], dim=1
                    )
                else:
                    padded = cost_matrix
                padded_costs.append(padded)

            batch_C = torch.stack(padded_costs, dim=0)  # [bs, N, max_size]

            assignments = batch_linear_assignment(batch_C)
            row_indices, col_indices = assignment_to_indices(assignments)

            indices_g = []
            for i, size in enumerate(sizes):
                valid_mask = col_indices[i] < size
                rows = row_indices[i][valid_mask]
                cols = col_indices[i][valid_mask]

                # IoU threshold filtering
                if self.iou_threshold > 0 and len(rows) > 0:
                    pred_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"][i][rows + g_num_queries * g_i])
                    tgt_xyxy = box_cxcywh_to_xyxy(targets[i]["boxes"][cols])
                    iou_matrix = torchvision.ops.box_iou(pred_xyxy, tgt_xyxy)
                    ious = torch.diag(iou_matrix)
                    keep = ious >= self.iou_threshold
                    rows = rows[keep]
                    cols = cols[keep]

                # query index에 그룹 offset 추가
                rows = rows + g_num_queries * g_i
                indices_g.append((rows, cols))

            if all_indices is None:
                all_indices = indices_g
            else:
                all_indices = [
                    (torch.cat([prev[0], cur[0]]), torch.cat([prev[1], cur[1]]))
                    for prev, cur in zip(all_indices, indices_g)
                ]

        return {'indices': all_indices}
