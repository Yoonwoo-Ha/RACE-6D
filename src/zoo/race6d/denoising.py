"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
Pose-aware denoising training for RACE6D.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += (rand_sign * rand_part * diff)
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        # FIXME, RT-DETR do not have this
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


# ======================== Pose-Aware Denoising ========================

def _enc_trans_to_3d(trans_pred, cam_K, bbox, img_w, img_h):
    """Convert encoder bbox-relative (rx, ry, log_tz) to 3D translation in mm."""
    rx, ry, log_tz = trans_pred[:, 0], trans_pred[:, 1], trans_pred[:, 2]
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    px, py = cam_K[0, 2], cam_K[1, 2]
    cx_pix = bbox[:, 0] * img_w
    cy_pix = bbox[:, 1] * img_h
    w_pix = bbox[:, 2] * img_w
    h_pix = bbox[:, 3] * img_h
    tz = torch.exp(log_tz) * 1000.0  # mm
    tx = ((rx * w_pix + cx_pix - px) * tz) / fx
    ty = ((ry * h_pix + cy_pix - py) * tz) / fy
    return torch.stack([tx, ty, tz], dim=-1)  # [N, 3] mm


def _compute_add_cost(enc_R, enc_t, gt_R, gt_t, model_points):
    """ADD cost: [N_enc] — ADD distance between each encoder prediction and a single GT."""
    N = enc_R.shape[0]
    # Transform by prediction: [N, P, 3]
    pts_pred = torch.matmul(enc_R, model_points.T).transpose(-2, -1) + enc_t[:, None, :]
    # Transform by GT: [P, 3]
    pts_gt = (gt_R @ model_points.T).T + gt_t
    return torch.norm(pts_pred - pts_gt[None], dim=-1).mean(dim=-1)  # [N]


def _compute_adds_cost(enc_R, enc_t, gt_R, gt_t, model_points):
    """ADD-S cost: [N_enc] — nearest-neighbor based, for symmetric objects."""
    N = enc_R.shape[0]
    P = model_points.shape[0]
    pts_pred = torch.matmul(enc_R, model_points.T).transpose(-2, -1) + enc_t[:, None, :]  # [N, P, 3]
    pts_gt = (gt_R @ model_points.T).T + gt_t  # [P, 3]
    # nearest neighbor: find the closest GT point for each predicted point
    dist_matrix = torch.cdist(pts_pred, pts_gt[None].expand(N, -1, -1))  # [N, P, P]
    return dist_matrix.min(dim=-1).values.mean(dim=-1)  # [N]


def _get_symmetry_type(label_id, models_info):
    """Determine object symmetry type."""
    obj_info = models_info.get(label_id)
    if obj_info is None:
        return 'asymmetric'
    if 'symmetries_continuous' in obj_info:
        return 'symmetric'
    if 'symmetries_discrete' in obj_info and len(obj_info['symmetries_discrete']) > 0:
        return 'symmetric'
    return 'asymmetric'


@torch.no_grad()
@torch.amp.autocast('cuda', enabled=False)
def _match_enc_to_gt(targets, enc_bboxes, enc_trans, enc_rots,
                     points_3d_cache, models_info, mscoco_label2category,
                     img_w, img_h):
    """Match encoder predictions to GT using ADD/ADD-R cost.

    Returns:
        matched_enc_indices: list[Tensor] — per-batch [N_gt] encoder index matched to each GT
    """
    bs = len(targets)
    device = enc_bboxes.device
    matched_enc_indices = []

    for b in range(bs):
        gt_labels = targets[b]['labels']  # [N_gt]
        N_gt = len(gt_labels)

        if N_gt == 0:
            matched_enc_indices.append(torch.zeros(0, dtype=torch.long, device=device))
            continue

        gt_poses = targets[b]['poses']  # [N_gt, 12]
        gt_R = gt_poses[:, 3:].reshape(-1, 3, 3)
        gt_t = gt_poses[:, :3]  # mm

        cam_K = targets[b]['cam_K'][0].reshape(3, 3)
        enc_t_3d = _enc_trans_to_3d(enc_trans[b], cam_K, enc_bboxes[b], img_w, img_h)  # [N_enc, 3] mm
        enc_R_b = enc_rots[b]  # [N_enc, 3, 3]
        N_enc = enc_R_b.shape[0]

        # Cost matrix [N_enc, N_gt]
        cost = torch.full([N_enc, N_gt], 1e6, device=device)

        for j in range(N_gt):
            label = int(gt_labels[j].item())
            pts = points_3d_cache.get(label)
            if pts is None:
                continue

            sym_type = _get_symmetry_type(label, models_info)
            if sym_type == 'asymmetric':
                cost[:, j] = _compute_add_cost(enc_R_b, enc_t_3d, gt_R[j], gt_t[j], pts)
            else:
                cost[:, j] = _compute_adds_cost(enc_R_b, enc_t_3d, gt_R[j], gt_t[j], pts)

        # Hungarian matching (scipy, small matrix)
        cost_np = cost.cpu().float().numpy()
        cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=1e6)
        row_ind, col_ind = scipy_linear_sum_assignment(cost_np)

        # col_ind → GT index, row_ind → matched encoder index
        # sort results in GT order
        enc_idx_for_gt = torch.zeros(N_gt, dtype=torch.long, device=device)
        matched_set = set()
        for r, c in zip(row_ind, col_ind):
            if c < N_gt:
                enc_idx_for_gt[c] = r
                matched_set.add(c)

        # unmatched GT falls back to the nearest encoder prediction
        for j in range(N_gt):
            if j not in matched_set:
                enc_idx_for_gt[j] = cost[:, j].argmin()

        matched_enc_indices.append(enc_idx_for_gt)

    return matched_enc_indices


def get_pose_denoising_training_group(
    targets,
    enc_topk_bboxes,    # [B, N, 4] — sigmoid bboxes
    enc_topk_kpts,      # [B, N, K*2] — keypoint predictions
    enc_topk_trans,     # [B, N, 3] — (rx, ry, log_tz)
    enc_topk_rots,      # [B, N, 3, 3] — rotation matrices
    points_3d_cache,
    models_info,
    mscoco_label2category,
    num_classes,
    num_queries,        # number of normal queries (determines attn_mask size)
    class_embed,        # nn.Embedding(num_classes+1, hidden_dim)
    img_w=640,
    img_h=480,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    """Pose-aware contrastive denoising training.

    Match encoder predictions to GT via ADD/ADD-R cost and use
    the matched encoder pose references (kpt, trans, rot) for DN queries.
    Apply standard CDN-style noise to bbox/cls.

    Returns:
        dn_logits:      [B, num_dn, C] — class embedding (rtdetrv2 style)
        dn_bbox_unact:  [B, num_dn, 4] — noised GT bbox (inverse sigmoid)
        dn_kpt:         [B, num_dn, K*2] — encoder kpt predictions
        dn_trans:       [B, num_dn, 3] — encoder trans predictions
        dn_rot:         [B, num_dn, 3, 3] — encoder rot predictions
        attn_mask:      [num_dn + num_queries, num_dn + num_queries]
        dn_meta:        dict with positive indices and split info
    """
    if num_denoising <= 0:
        return None, None, None, None, None, None, None

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    bs = len(num_gts)
    max_gt_num = max(num_gts)
    num_kpt_dim = enc_topk_kpts.shape[-1]

    if max_gt_num == 0:
        return None, None, None, None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    num_denoising = int(max_gt_num * 2 * num_group)

    # === 1. ADD/ADD-R matching: find the closest encoder prediction for each GT ===
    matched_enc_indices = _match_enc_to_gt(
        targets, enc_topk_bboxes, enc_topk_trans, enc_topk_rots,
        points_3d_cache, models_info, mscoco_label2category,
        img_w, img_h)

    # === 2. Pad GT + matched encoder features ===
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)
    # Encoder pose references for matched predictions
    input_enc_kpt = torch.zeros([bs, max_gt_num, num_kpt_dim], device=device)
    input_enc_trans = torch.zeros([bs, max_gt_num, 3], device=device)
    input_enc_rot = torch.eye(3, device=device).unsqueeze(0).expand(bs, max_gt_num, -1, -1).clone()

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = True

            idx = matched_enc_indices[i]  # [num_gt]
            input_enc_kpt[i, :num_gt] = enc_topk_kpts[i][idx]
            input_enc_trans[i, :num_gt] = enc_topk_trans[i][idx]
            input_enc_rot[i, :num_gt] = enc_topk_rots[i][idx]

    # === 3. CDN structure: num_group positive + num_group negative queries per GT ===
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    input_enc_kpt = input_enc_kpt.tile([1, 2 * num_group, 1])
    input_enc_trans = input_enc_trans.tile([1, 2 * num_group, 1])
    input_enc_rot = input_enc_rot.tile([1, 2 * num_group, 1, 1])

    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask

    # positive index
    positive_gt_mask_flat = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask_flat)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    # === 4. Label noise (same as standard CDN) ===
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # === 5. Box noise (same as standard CDN) ===
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += (rand_sign * rand_part * diff)
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1

    input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    # Content: class embedding (standard CDN style)
    dn_class_logits = class_embed(input_query_class)

    # === 6. Attention mask: DN isolation + Group DETR isolation ===
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)

    # Normal queries cannot see DN queries
    attn_mask[num_denoising:, :num_denoising] = True

    # DN groups cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }

    return (dn_class_logits, input_query_bbox_unact,
            input_enc_kpt, input_enc_trans, input_enc_rot,
            attn_mask, dn_meta)
