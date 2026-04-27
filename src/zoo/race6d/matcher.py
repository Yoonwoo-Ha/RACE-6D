"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.
---------------------------------------------------------------------
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import math
import torch
import torch.nn.functional as F
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
import torchvision
from pytorch3d.transforms import rotation_6d_to_matrix

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
    GPU-Util Hungarian Matcher for RACE-6D.

    Costs: classification + bbox L1 + GIoU [+ log-tz + geodesic rotation]

    cost_tz and cost_rot default to 0 (disabled). Enable via weight_dict.
    sym_cache is injected by the criterion via set_pose_source() after init.
    """
    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox  = weight_dict.get('cost_bbox', 5.0)
        self.cost_giou  = weight_dict.get('cost_giou', 2.0)
        self.cost_tz    = weight_dict.get('cost_tz',   0.0)
        self.cost_rot   = weight_dict.get('cost_rot',  0.0)
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        # Populated via set_pose_source() called by criterion after creation
        self._sym_cache: dict = {}

    def set_pose_source(self, sym_cache=None, **kwargs):
        """Receive symmetry rotations from criterion's pose data.

        Called automatically by RACE6DCriterion_addr.set_pose_source() if this
        method exists. Extra kwargs are accepted for forward-compatibility.
        """
        if sym_cache is not None:
            self._sym_cache = sym_cache

    def _geodesic_cost(self, out_R, tgt_R_all, tgt_labels_all, device):
        """Symmetry-aware geodesic cost matrix.

        out_R:          [B*Q, 3, 3]
        tgt_R_all:      [sum_T, 3, 3]
        tgt_labels_all: list[int] length sum_T

        Returns [B*Q, sum_T] ∈ [0, 1]  (geodesic / π)
        """
        sum_T = tgt_R_all.shape[0]
        identity = torch.eye(3, device=device)

        # Per-target symmetry rotation list
        sym_list = []
        for lbl in tgt_labels_all:
            s = self._sym_cache.get(int(lbl))
            sym_list.append(s.to(device) if s is not None else identity.unsqueeze(0))

        S_max = max(s.shape[0] for s in sym_list)

        # Padded sym_Rs [sum_T, S_max, 3, 3] — extra slots stay as identity
        # (identity = valid "no-symmetry" rotation, so min over them is correct)
        sym_padded = identity.unsqueeze(0).unsqueeze(0).expand(sum_T, S_max, 3, 3).clone()
        for i, s in enumerate(sym_list):
            sym_padded[i, :s.shape[0]] = s

        # [sum_T, S_max, 3, 3]
        R_gt_sym = tgt_R_all[:, None] @ sym_padded

        # [B*Q, sum_T, S_max, 3, 3]
        R_diff = out_R[:, None, None] @ R_gt_sym[None].mT

        trace    = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        angles   = ((trace - 1) / 2).clamp(-1.0, 1.0).acos()  # [B*Q, sum_T, S_max]
        return angles.min(dim=-1).values / math.pi              # [B*Q, sum_T]

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        # ---- Classification probabilities ----
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))   # [B*Nq, C]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Nq, C]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Nq, 4]

        tgt_ids  = torch.cat([v["labels"] for v in targets])  # [B*Nt]
        tgt_bbox = torch.cat([v["boxes"]  for v in targets])  # [B*Nt, 4]

        # ---- Classification cost ----
        if self.use_focal_loss:
            prob = out_prob[:, tgt_ids]
            neg_cost = (1 - self.alpha) * (prob ** self.gamma) * (-(1 - prob + 1e-8).log())
            pos_cost = self.alpha * ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
            cost_class = pos_cost - neg_cost  # [B*Nq, B*Nt]
        else:
            cost_class = -out_prob[:, tgt_ids]

        # ---- BBox L1 cost ----
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*Nq, B*Nt]

        # ---- GIoU cost ----
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )  # [B*Nq, B*Nt]

        # ---- log-space tz cost (optional) ----
        cost_tz = None
        if self.cost_tz > 0 and 'pred_translations' in outputs:
            pred_log_tz = outputs['pred_translations'][..., 2].flatten(0, 1)  # [B*Nq]
            gt_tz_mm    = torch.cat([v['poses'][:, 2] for v in targets])       # [B*Nt]
            gt_log_tz   = torch.log(torch.clamp(gt_tz_mm / 1000.0, min=1e-3))
            cost_tz = torch.cdist(
                pred_log_tz.unsqueeze(-1), gt_log_tz.unsqueeze(-1), p=1
            )  # [B*Nq, B*Nt]

        # ---- Geodesic rotation cost (optional, symmetry-aware) ----
        cost_rot = None
        if self.cost_rot > 0 and 'pred_rotations' in outputs:
            out_R = rotation_6d_to_matrix(
                outputs['pred_rotations'].flatten(0, 1)  # [B*Q, 6]
            )  # [B*Q, 3, 3]
            tgt_R_all = torch.cat(
                [v['poses'][:, 3:].reshape(-1, 3, 3) for v in targets]
            ).to(device)  # [sum_T, 3, 3]
            tgt_labels_all = [int(l) for v in targets for l in v['labels'].tolist()]
            cost_rot = self._geodesic_cost(out_R, tgt_R_all, tgt_labels_all, device)
            # [B*Nq, B*Nt]

        # ---- Total cost ----
        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou)
        if cost_tz is not None:
            C = C + self.cost_tz * cost_tz
        if cost_rot is not None:
            C = C + self.cost_rot * cost_rot

        # ---- Batched Hungarian assignment ----
        sizes = [len(v["labels"]) for v in targets]
        C = C.view(bs, num_queries, -1)  # [B, Nq, total_targets]
        C_split = C.split(sizes, -1)

        batch_costs = [C_split[i][i] for i in range(bs)]  # [Nq, size_i] per batch

        max_size = max(sizes) if sizes else 1
        padded_costs = []
        for cost_matrix in batch_costs:
            num_q, size_i = cost_matrix.shape
            pad_cols = max_size - size_i
            if pad_cols > 0:
                padded = torch.cat(
                    [cost_matrix, torch.full((num_q, pad_cols), 1e9, device=device)], dim=1
                )
            else:
                padded = cost_matrix
            padded_costs.append(padded)

        batch_C = torch.stack(padded_costs, dim=0)  # [bs, Nq, max_size]
        assignments = batch_linear_assignment(batch_C)
        row_indices, col_indices = assignment_to_indices(assignments)

        all_indices = []
        for i, size in enumerate(sizes):
            valid_mask = col_indices[i] < size
            rows = row_indices[i][valid_mask]
            cols = col_indices[i][valid_mask]

            # indices: 필터 없음 (cls/bbox bootstrap용 — VFL이 IoU=0 자동 처리)
            all_indices.append((rows, cols))

        return {'indices': all_indices}
