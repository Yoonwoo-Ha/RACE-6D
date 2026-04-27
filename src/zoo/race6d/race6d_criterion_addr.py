"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision
import copy
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register

@register()
class RACE6DCriterion_addr(nn.Module):
    """ DETR의 손실을 계산하는 클래스.
    과정:
        1) 모델의 출력과 정답 박스 간 헝가리안 매칭을 계산
        2) 매칭된 각 쌍에 대해 감독 학습(클래스와 박스 모두)
    """
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        gamma_pos=None,
        gamma_neg=None,
        num_classes=21,
        boxes_weight_format=None,
        share_matched_indices=False,
        enc_weight_dict=None,
        pose_quality_scale=0.2,
        **kwargs):
        """
        Parameters:
            matcher: 타겟과 제안 간 매칭을 계산할 수 있는 모듈
            num_classes: 객체 카테고리 수
            weight_dict: 손실 이름을 키로, 상대적 가중치를 값으로 하는 딕셔너리
            losses: 적용할 모든 손실의 리스트
            boxes_weight_format: 박스 가중치의 형식
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.enc_weight_dict = enc_weight_dict or weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        # Pose-aware score target (boxes_weight_format='iou_pose')
        #   values = ADD-R/diameter based pose_quality in [0, 1] (bbox IoU 미반영)
        self.pose_quality_scale = float(pose_quality_scale)
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        # MAL/VFL용: positive target은 gamma_pos, negative weight는 gamma_neg
        # 미지정 시 기존 gamma로 fallback (backward compat)
        self.gamma_pos = gamma_pos if gamma_pos is not None else gamma
        self.gamma_neg = gamma_neg if gamma_neg is not None else gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = 0

        # Pose data placeholders — populated by set_pose_source(decoder) in solver
        self.keypoints_3d = None
        self.edges_mask = None
        self.models_info = {}
        self.sym_cache = {}
        self.points_3d_cache = {}
        self.diameter_cache = {}
        self.mscoco_category2label = {}
        self.mscoco_label2category = {}
        self.valid_label_ids = []

    def set_pose_source(self, decoder):
        """Decoder의 pose buffer를 참조하여 criterion 데이터 교체.
        Solver에서 model/criterion 생성 후 호출."""
        self.keypoints_3d = decoder.keypoints_3d
        self.edges_mask = decoder.edges_mask
        self.sym_cache = decoder.get_sym_cache()
        self.points_3d_cache = decoder.get_points_3d_cache()
        self.diameter_cache = decoder.get_diameter_cache()
        self.mscoco_category2label = decoder.mscoco_category2label
        self.mscoco_label2category = decoder.mscoco_label2category
        self.valid_label_ids = list(decoder.mscoco_label2category.keys())
        self.models_info = decoder.models_info
        self.num_classes = len(self.mscoco_label2category)

        # Propagate pose data to matcher so cost_keypoint (sym-aware kpt L1) can be computed.
        # If matcher doesn't support it, this is a no-op.
        if hasattr(self.matcher, 'set_pose_source') and callable(self.matcher.set_pose_source):
            self.matcher.set_pose_source(
                keypoints_3d=self.keypoints_3d,
                sym_cache=self.sym_cache,
                mscoco_label2category=self.mscoco_label2category,
            )

    def _get_orig_size(self, targets):
        """targets에서 원본 이미지 크기 (w, h) 반환.

        Criterion은 학습 전용이므로 targets에는 항상 orig_size가 있어야 한다.
        """
        orig_size = targets[0]['orig_size']  # [W, H]
        return float(orig_size[0]), float(orig_size[1])

    def get_symmetry_type(self, label_id: int) -> str:
        """
        객체의 대칭성 타입을 판별

        반환 string은 내부에서 오직 `== 'asymmetric'` 이진 분기에만 사용됨.
        실제 대칭 회전 행렬은 decoder가 만든 `sym_cache[label_id]`에 임의 축 포함
        Rodrigues로 정확히 계산돼 있으므로, 여기선 "대칭 여부"만 판단하면 됨.

        Args:
            label_id: 리맵된 레이블 ID (0-index)

        Returns:
            str: 'asymmetric', 'continuous_symmetric', 'discrete_symmetric'
        """
        obj_info = self.models_info.get(label_id)
        if obj_info is None:
            return 'asymmetric'

        if 'symmetries_continuous' in obj_info and len(obj_info['symmetries_continuous']) > 0:
            return 'continuous_symmetric'

        if 'symmetries_discrete' in obj_info and len(obj_info['symmetries_discrete']) > 0:
            return 'discrete_symmetric'

        return 'asymmetric'
    
    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).clamp(min=0).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=self.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma_neg) * (1 - target) + target_score

        if target_score.shape[2] == 1:
            print(src_logits.shape, target_score.shape, weight.shape)

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        pred_classes = src_logits[idx].argmax(dim=-1)
        cls_error = (pred_classes != target_classes_o).float().mean() if len(target_classes_o) > 0 else torch.tensor(0.0, device=self.device)

        return {'loss_vfl': loss, 'metric_cls_error': cls_error}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """Matchability-Aware Loss (MAL) with IoU score target."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # IoU 계산
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).clamp(min=0).detach()

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=self.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target
        target_score = target_score.pow(self.gamma_pos)  # positive target에 gamma_pos 적용

        pred_score = torch.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma_neg) * (1 - target) + target  # negative weight에 gamma_neg

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        pred_classes = src_logits[idx].argmax(dim=-1)
        cls_error = (pred_classes != target_classes_o).float().mean() if len(target_classes_o) > 0 else torch.tensor(0.0, device=self.device)

        return {'loss_mal': loss, 'metric_cls_error': cls_error}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # Per-component bbox error in pixels (orig_size 기준)
        orig_w, orig_h = self._get_orig_size(targets)
        if len(src_boxes) > 0:
            diff = (src_boxes - target_boxes).abs()
            losses['metric_bbox_cx_error'] = (diff[:, 0].mean() * orig_w).detach()
            losses['metric_bbox_cy_error'] = (diff[:, 1].mean() * orig_h).detach()
            losses['metric_bbox_w_error'] = (diff[:, 2].mean() * orig_w).detach()
            losses['metric_bbox_h_error'] = (diff[:, 3].mean() * orig_h).detach()
        else:
            zero = torch.tensor(0.0, device=src_boxes.device)
            losses['metric_bbox_cx_error'] = zero
            losses['metric_bbox_cy_error'] = zero
            losses['metric_bbox_w_error'] = zero
            losses['metric_bbox_h_error'] = zero
        return losses
    
    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        assert 'pred_keypoints' in outputs
        device = self.device
        img_w, img_h = self._get_orig_size(targets)

        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_keypoints'][idx].float()  # [N, K*2] bbox-relative, FP32 강제

        if src_kpts.dim() == 2 and src_kpts.size(-1) == 64:
            src_kpts = src_kpts.reshape(-1, 32, 2)  # [N, 32, 2] bbox-relative
        else:
            raise ValueError(f"Unexpected pred_keypoints shape: {src_kpts.shape}")

        N = src_kpts.shape[0]
        if N == 0:
            return {
                'loss_keypoints': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_cr': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_oks': torch.tensor(0.0, device=device, requires_grad=True),
                'metric_kpt_error': torch.tensor(0.0, device=device),
            }

        # GT 데이터 수집
        target_poses  = torch.cat([t['poses'][j]  for t, (_, j) in zip(targets, indices)], dim=0)  # [N,12]
        target_labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N]

        # GT R, t 복원 (raw mm)
        R_gt = target_poses[:, 3:].reshape(-1, 3, 3)  # [N,3,3]

        # cam_K from targets (per-instance)
        cam_K = torch.cat([t['cam_K'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        cam_K = cam_K.reshape(-1, 3, 3)  # [N,3,3] per-instance

        t_gt = self._c2t_gt(target_poses[:, :3], cam_K)  # [N,3] - (tx, ty, tz) in mm

        # pred bbox-relative → pixel (GT boxes 사용)
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N,4]
        pred_pix = self._bbox_relative_to_pixel(src_kpts, target_boxes, img_w, img_h)  # [N, 32, 2]

        # Bbox area (픽셀 단위) — GT boxes for OKS scale
        bbox_area_pix = (target_boxes[:, 2] * img_w) * (target_boxes[:, 3] * img_h)  # [N]

        # 클래스별로 계산
        unique_cls = torch.unique(target_labels)
        per_inst_l1_losses = []
        per_inst_cr_losses = []
        per_inst_oks_losses = []
        per_inst_kpt_pix_errors = []

        for cls_id in unique_cls:
            label_id = int(cls_id.item())
            if label_id not in self.mscoco_label2category:
                continue

            category_id = self.mscoco_label2category[label_id]

            mask = (target_labels == cls_id)
            M = int(mask.sum().item())
            if M == 0:
                continue

            # 클래스 데이터 추출
            Rg = R_gt[mask]                     # [M,3,3]
            tg = t_gt[mask]                     # [M,3]
            bbox_area = bbox_area_pix[mask]     # [M] - 픽셀 area
            pred = pred_pix[mask]               # [M,32,2] - 픽셀 좌표
            Kg = cam_K[mask]                    # [M,3,3] - per-instance intrinsics
            fx = Kg[:, 0, 0][:, None, None]     # [M,1,1]
            fy = Kg[:, 1, 1][:, None, None]
            px = Kg[:, 0, 2][:, None, None]
            py = Kg[:, 1, 2][:, None, None]

            # 3D 키포인트 32개 로드
            kp3d_cls = self.keypoints_3d[category_id - 1].to(device)  # [32,3]
            kp3d = kp3d_cls.unsqueeze(0).expand(M, -1, -1)              # [M,32,3]

            # 캐시에서 대칭 변환 가져오기
            sym_Rs = self.sym_cache.get(label_id)
            if sym_Rs is None:
                sym_Rs = torch.eye(3, device=device).unsqueeze(0)
            S = sym_Rs.shape[0]

            # === GT 키포인트 생성 (모든 대칭) ===
            # 1. 등가 포즈 생성
            R_prime = torch.matmul(Rg.unsqueeze(1), sym_Rs.unsqueeze(0))  # [M,S,3,3]
            t_prime = tg.unsqueeze(1)  # [M,1,3]

            # 2. 32개 키포인트를 3D → 2D 투영 (픽셀)
            X_cam = torch.einsum('msij, mkj -> mski', R_prime, kp3d) + t_prime.unsqueeze(2)  # [M,S,32,3]
            # NOTE: clamp는 mm 단위. 1mm 이하로 내려가는 물리적 상황은 없음.
            # 1e-6은 fp16 subnormal 영역이라 AMP 하에서 underflow 위험.
            Z = X_cam[..., 2].clamp_min(1.0)  # [M,S,32]
            u_pix = fx * (X_cam[..., 0] / Z) + px  # [M,S,32]
            v_pix = fy * (X_cam[..., 1] / Z) + py  # [M,S,32]

            tgt_pix = torch.stack([u_pix, v_pix], dim=-1)  # [M,S,32,2] - 픽셀 좌표

            # === 예측 키포인트 확장 ===
            pred_exp = pred.unsqueeze(1).expand(-1, S, -1, -1)  # [M,S,32,2] 픽셀

            # === 1. L1 Loss (image-normalized space) ===
            diff = pred_exp - tgt_pix  # [M,S,32,2] 픽셀 단위
            diff_img = diff.clone()
            diff_img[..., 0] = diff[..., 0] / img_w
            diff_img[..., 1] = diff[..., 1] / img_h
            l1_per_point = torch.abs(diff_img).sum(dim=-1)  # [M,S,32]
            l1_per_symmetry = l1_per_point.sum(dim=-1)  # [M,S]

            # === 2. OKS Loss (픽셀 공간에서) ===
            s_pix = torch.sqrt(bbox_area)  # [M] - 픽셀 스케일

            ki = 0.1
            squared_dist_pix = torch.sum((pred_exp - tgt_pix) ** 2, dim=-1)  # [M,S,32]

            s_expanded = s_pix.unsqueeze(1).unsqueeze(2).expand(-1, S, 32)  # [M,S,32]
            exp_term = torch.exp(-squared_dist_pix / (2 * s_expanded ** 2 * ki ** 2))  # [M,S,32]
            oks_per_symmetry = 1.0 - exp_term.sum(dim=-1) / 32.0  # [M,S]

            # === 3. 최적 대칭 선택 (pose loss와 일관) ===
            batch_indices = torch.arange(M, device=device)
            if hasattr(self, '_cached_sym_indices') and self._cached_sym_indices is not None:
                # pose loss에서 결정된 symmetry index 사용 (ADD-R 기준)
                min_indices = self._cached_sym_indices[mask]
                min_oks_values = oks_per_symmetry[batch_indices, min_indices]
            else:
                # fallback: OKS 기준 자체 선택
                min_oks_values, min_indices = oks_per_symmetry.min(dim=1)  # [M]

            selected_l1_values = l1_per_symmetry[batch_indices, min_indices]  # [M]

            # === 4. Cross-ratio 계산 (픽셀 좌표) ===
            selected_tgt_pix = tgt_pix[batch_indices, min_indices, :, :]  # [M,32,2]
            pred_pix_cls = pred_pix[mask]  # [M,32,2] 픽셀

            cr_losses = self.compute_cross_ratio_loss(
                pred_pix_cls.unsqueeze(1),      # [M,1,32,2] 픽셀
                selected_tgt_pix.unsqueeze(1),   # [M,1,32,2] 픽셀
                label_id
            )

            # 5. 결과 저장
            per_inst_l1_losses.append(selected_l1_values)
            per_inst_oks_losses.append(min_oks_values)
            per_inst_cr_losses.append(cr_losses)

            # Pixel error for metric
            diff_pix = pred_exp - tgt_pix  # [M,S,32,2] 픽셀 단위
            l1_pix_per_point = torch.abs(diff_pix).sum(dim=-1)  # [M,S,32]
            l1_pix_per_symmetry = l1_pix_per_point.mean(dim=-1)  # [M,S]
            selected_pix_error = l1_pix_per_symmetry[batch_indices, min_indices]  # [M]
            per_inst_kpt_pix_errors.append(selected_pix_error)

        # 최종 loss 계산
        if len(per_inst_l1_losses) > 0:
            per_inst_l1_losses = torch.cat(per_inst_l1_losses, dim=0)
            loss_keypoints = per_inst_l1_losses.sum() / max(num_boxes, 1.0)
        else:
            loss_keypoints = torch.tensor(0.0, device=device, requires_grad=True)

        if len(per_inst_oks_losses) > 0:
            per_inst_oks_losses = torch.cat(per_inst_oks_losses, dim=0)
            loss_oks = per_inst_oks_losses.sum() / max(num_boxes, 1.0)
        else:
            loss_oks = torch.tensor(0.0, device=device, requires_grad=True)

        if len(per_inst_cr_losses) > 0:
            per_inst_cr_losses = torch.cat(per_inst_cr_losses, dim=0)
            loss_cr = per_inst_cr_losses.sum() / max(num_boxes, 1.0)
        else:
            loss_cr = torch.tensor(0.0, device=device, requires_grad=True)

        # Keypoint error in pixels (metric)
        if len(per_inst_kpt_pix_errors) > 0:
            kpt_error = torch.cat(per_inst_kpt_pix_errors).mean()
        else:
            kpt_error = torch.tensor(0.0, device=device)

        return {
            'loss_keypoints': loss_keypoints,
            'loss_oks': loss_oks,
            'loss_cr': loss_cr,
            'metric_kpt_error': kpt_error.detach(),
        }

    def compute_cross_ratio_loss(self, pred_keypoints, target_keypoints, label_id):
        """
        완전히 벡터화된 Cross-ratio loss (픽셀 좌표)
        
        Args:
            pred_keypoints: [M, 1, 32, 2] - 픽셀 좌표
            target_keypoints: [M, 1, 32, 2] - 픽셀 좌표
            label_id: 리맵된 레이블 ID
        """
        M = pred_keypoints.shape[0]
        device = pred_keypoints.device
        
        category_id = self.mscoco_label2category.get(label_id)
        if category_id is None:
            return torch.zeros(M, device=device)

        edges_key = str(category_id)
        if edges_key not in self.edges_mask:
            return torch.zeros(M, device=device)
        
        edges = self.edges_mask[edges_key]
        num_edges = len(edges)
        
        if num_edges == 0:
            return torch.zeros(M, device=device)
        
        # ===== 벡터화: 모든 edge를 한 번에 처리 =====
        edges_tensor = torch.tensor(edges, dtype=torch.long, device=device)  # [num_edges, 4]
        
        # Squeeze + FP32 강제 (AMP FP16에서 epsilon underflow 방지)
        pred_kpts = pred_keypoints.squeeze(1).float()  # [M, 32, 2]
        tgt_kpts = target_keypoints.squeeze(1).float()  # [M, 32, 2]
        
        # 모든 edge의 4개 점을 한 번에 인덱싱: [M, num_edges, 4, 2]
        pred_points = pred_kpts[:, edges_tensor]
        tgt_points = tgt_kpts[:, edges_tensor]
        
        # 4개 점 분리: A, B, C, D
        pred_A = pred_points[:, :, 0]  # [M, num_edges, 2]
        pred_B = pred_points[:, :, 1]
        pred_C = pred_points[:, :, 2]
        pred_D = pred_points[:, :, 3]
        
        tgt_A = tgt_points[:, :, 0]
        tgt_B = tgt_points[:, :, 1]
        tgt_C = tgt_points[:, :, 2]
        tgt_D = tgt_points[:, :, 3]
        
        # Cross-ratio 계산 (벡터화) - 모든 edge 사용
        pred_CA = torch.norm(pred_C - pred_A, dim=-1)
        pred_DB = torch.norm(pred_D - pred_B, dim=-1)
        pred_CB = torch.norm(pred_C - pred_B, dim=-1)
        pred_DA = torch.norm(pred_D - pred_A, dim=-1)
        pred_denominator = pred_CB * pred_DA + 1e-8
        pred_cr = (pred_CA * pred_DB) / pred_denominator
        pred_cr = torch.clamp(pred_cr, min=1e-6, max=100.0)
        
        tgt_CA = torch.norm(tgt_C - tgt_A, dim=-1)
        tgt_DB = torch.norm(tgt_D - tgt_B, dim=-1)
        tgt_CB = torch.norm(tgt_C - tgt_B, dim=-1)
        tgt_DA = torch.norm(tgt_D - tgt_A, dim=-1)
        tgt_denominator = tgt_CB * tgt_DA + 1e-8
        tgt_cr = (tgt_CA * tgt_DB) / tgt_denominator
        
        # SmoothL1 loss - 모든 edge 사용
        smooth_l1 = F.smooth_l1_loss(pred_cr, tgt_cr, reduction='none', beta=0.1)
        
        # 각 인스턴스별 평균
        loss_per_instance = smooth_l1.mean(dim=1)
        
        return loss_per_instance
    
    def _c2t_pred(self, translation, cam_K, bbox_info, img_w, img_h):
        """Convert bbox-relative (rx, ry, log_tz) to 3D translation (mm).
        translation: [N, 3] or [3]  — (rx, ry, log_tz)  bbox-relative offset
        cam_K: [3, 3] or [N, 3, 3] camera intrinsics (per-instance 지원)
        bbox_info: [N, 4] normalized [cx, cy, w, h]
        img_w, img_h: 원본 이미지 크기 (caller가 반드시 전달, cam_K 기준과 일치해야 함)
        """

        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        rx, ry, log_tz = translation[:, 0], translation[:, 1], translation[:, 2]

        if cam_K.dim() == 3:
            # Per-instance [N, 3, 3]
            fx = cam_K[:, 0, 0]
            fy = cam_K[:, 1, 1]
            px = cam_K[:, 0, 2]
            py = cam_K[:, 1, 2]
        else:
            # Shared [3, 3]
            fx = cam_K[0, 0]
            fy = cam_K[1, 1]
            px = cam_K[0, 2]
            py = cam_K[1, 2]

        if bbox_info.dim() == 1:
            bbox_info = bbox_info.unsqueeze(0)

        cxbbox = bbox_info[:, 0] * img_w
        cybbox = bbox_info[:, 1] * img_h
        wbbox = bbox_info[:, 2] * img_w
        hbbox = bbox_info[:, 3] * img_h

        tz = torch.exp(log_tz) * 1000.0  # mm
        tx = ((rx * wbbox + cxbbox - px) * tz) / fx
        ty = ((ry * hbbox + cybbox - py) * tz) / fy

        result = torch.stack([tx, ty, tz], dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result
    
    def _c2t_gt(self, translation, cam_K, bbox_info=None):
        """Return raw mm translation directly (no ConvertPose in pipeline).
        translation: [N, 3] or [3]  — (tx_mm, ty_mm, tz_mm)
        cam_K: unused (kept for API compatibility)
        bbox_info: unused (kept for API compatibility)
        """
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        tx, ty, tz = translation[:, 0], translation[:, 1], translation[:, 2]
        result = torch.stack([tx, ty, tz], dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _bbox_relative_to_pixel(self, keypoints_relative, bbox_info, img_w, img_h):
        """Bbox 중심 상대 좌표를 픽셀 절대 좌표로 변환.
        keypoints_relative: [N, K, 2] bbox-relative (w,h로 정규화된 offset)
        bbox_info: [N, 4] normalized [cx, cy, w, h]
        Returns: [N, K, 2] pixel coords
        """
        if keypoints_relative.dim() == 2:
            keypoints_relative = keypoints_relative.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if bbox_info.dim() == 1:
            bbox_info = bbox_info.unsqueeze(0)

        cx_pix = bbox_info[:, 0] * img_w  # [N]
        cy_pix = bbox_info[:, 1] * img_h  # [N]
        w_pix = bbox_info[:, 2] * img_w   # [N]
        h_pix = bbox_info[:, 3] * img_h   # [N]

        kpts_x_pix = keypoints_relative[..., 0] * w_pix.unsqueeze(-1) + cx_pix.unsqueeze(-1)  # [N, K]
        kpts_y_pix = keypoints_relative[..., 1] * h_pix.unsqueeze(-1) + cy_pix.unsqueeze(-1)  # [N, K]

        result = torch.stack([kpts_x_pix, kpts_y_pix], dim=-1)  # [N, K, 2]

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def loss_addr(self, outputs, targets, indices, num_boxes):
        """
        ADD / ADD-R loss (m 단위, REF6D 방식)
        - 비대칭: ADD loss
        - 대칭: ADD-R loss
        """
        assert 'pred_rotations' in outputs
        assert 'pred_translations' in outputs

        idx = self._get_src_permutation_idx(indices)

        # Targets
        target_poses = torch.cat([
            t['poses'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Empty-match short-circuit — must come BEFORE any per-instance indexing
        # (e.g., cam_K[0]) which would fail on a zero-length tensor. This path
        # is hit when Hungarian matching returns no valid pairs (early training
        # with strict iou_threshold, or degenerate batches).
        if len(target_classes) == 0:
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            zero_m = torch.tensor(0.0, device=self.device)
            return {
                'loss_pose': zero,
                'metric_tx_error': zero_m, 'metric_ty_error': zero_m,
                'metric_tz_error': zero_m, 'metric_rot_error': zero_m,
            }

        target_rotations = target_poses[:, 3:].reshape(-1, 3, 3)

        # cam_K from targets (per-instance)
        cam_K = torch.cat([t['cam_K'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        cam_K = cam_K.reshape(-1, 3, 3)  # [N,3,3] per-instance

        target_translations = self._c2t_gt(target_poses[:, :3], cam_K) / 1000.0  # mm → m

        # Predictions (bbox-relative → m)
        orig_w, orig_h = self._get_orig_size(targets)
        src_rotations = outputs['pred_rotations'][idx].reshape(-1, 3, 3)
        src_boxes = outputs['pred_boxes'][idx].detach()
        src_translations = self._c2t_pred(outputs['pred_translations'][idx], cam_K, src_boxes, orig_w, orig_h) / 1000.0  # mm → m

        tx_error_mm = (src_translations[:, 0] - target_translations[:, 0]).abs().mean() * 1000.0
        ty_error_mm = (src_translations[:, 1] - target_translations[:, 1]).abs().mean() * 1000.0
        tz_error_mm = (src_translations[:, 2] - target_translations[:, 2]).abs().mean() * 1000.0

        all_losses = []
        all_rot_errors = []
        # per-instance best symmetry index 수집 (keypoint loss와 공유)
        all_sym_indices = torch.zeros(len(target_classes), dtype=torch.long, device=self.device)

        for cls_id in torch.unique(target_classes):
            cls_id_val = int(cls_id.item())
            if cls_id_val not in self.mscoco_label2category:
                continue

            cls_mask = (target_classes == cls_id)

            cls_R_pred = src_rotations[cls_mask]
            cls_t_pred = src_translations[cls_mask]
            cls_R_gt = target_rotations[cls_mask]
            cls_t_gt = target_translations[cls_mask]

            model_points = self.points_3d_cache.get(cls_id_val)
            if model_points is None:
                continue
            model_points_m = model_points / 1000.0  # mm → m

            diameter_m = self.diameter_cache.get(cls_id_val, 1.0) / 1000.0  # mm → m

            symmetry_type = self.get_symmetry_type(cls_id_val)
            sym_Rs = self.sym_cache.get(cls_id_val)
            if sym_Rs is None:
                sym_Rs = torch.eye(3, device=self.device).unsqueeze(0)

            if symmetry_type == 'asymmetric':
                loss, best_sym_idx = self._compute_add_loss(
                    cls_R_pred, cls_t_pred,
                    cls_R_gt, cls_t_gt,
                    model_points_m,
                )
            else:
                loss, best_sym_idx = self._compute_addr_loss(
                    cls_R_pred, cls_t_pred,
                    cls_R_gt, cls_t_gt,
                    sym_Rs, model_points_m,
                )
            loss = loss / diameter_m

            all_losses.append(loss)
            all_sym_indices[cls_mask] = best_sym_idx

            if symmetry_type == 'asymmetric':
                rot_error = self._compute_rotation_error(cls_R_pred, cls_R_gt)
            else:
                rot_error = self._compute_symmetric_rotation_error(cls_R_pred, cls_R_gt, sym_Rs)
            all_rot_errors.append(rot_error)

        # keypoint loss에서 사용할 수 있도록 캐싱
        self._cached_sym_indices = all_sym_indices

        if len(all_losses) == 0:
            return {
                'loss_pose': torch.tensor(0.0, device=self.device, requires_grad=True),
                'metric_tx_error': tx_error_mm.detach(),
                'metric_ty_error': ty_error_mm.detach(),
                'metric_tz_error': tz_error_mm.detach(),
                'metric_rot_error': torch.tensor(0.0, device=self.device),
            }

        final_loss = torch.cat(all_losses, dim=0).sum() / max(num_boxes, 1.0)
        rot_error_deg = torch.cat(all_rot_errors, dim=0).mean()

        return {
            'loss_pose': final_loss,
            'metric_tx_error': tx_error_mm.detach(),
            'metric_ty_error': ty_error_mm.detach(),
            'metric_tz_error': tz_error_mm.detach(),
            'metric_rot_error': rot_error_deg.detach(),
        }
    
    def _compute_add_loss(self, R_pred, t_pred, R_gt, t_gt, model_points):
        """
        비대칭 객체용 ADD loss

        Args:
            R_pred: [M, 3, 3]
            t_pred: [M, 3]
            R_gt: [M, 3, 3]
            t_gt: [M, 3]
            model_points: [P, 3]

        Returns:
            losses: [M] — ADD distance
            best_sym_idx: [M] — 항상 0 (비대칭이므로 대칭 index 없음)
        """
        points_pred = torch.matmul(
            R_pred, model_points.T
        ).transpose(-2, -1) + t_pred[:, None, :]  # [M, P, 3]

        points_gt = torch.matmul(
            R_gt, model_points.T
        ).transpose(-2, -1) + t_gt[:, None, :]  # [M, P, 3]

        avg_distances = torch.norm(points_pred - points_gt, dim=-1).mean(dim=-1)  # [M]

        best_sym_idx = torch.zeros(R_pred.shape[0], dtype=torch.long, device=R_pred.device)
        return avg_distances, best_sym_idx

    def _compute_addr_loss(self, R_pred, t_pred, R_gt, t_gt, sym_Rs, model_points):
        """
        대칭 객체용 ADD-R loss

        Args:
            R_pred: [M, 3, 3]
            t_pred: [M, 3]
            R_gt: [M, 3, 3]
            t_gt: [M, 3]
            sym_Rs: [S, 3, 3]
            model_points: [P, 3]

        Returns:
            losses: [M] — ADD-R distance
            best_sym_idx: [M]
        """
        points_pred = torch.matmul(
            R_pred, model_points.T
        ).transpose(-2, -1) + t_pred[:, None, :]  # [M, P, 3]

        R_gt_sym = torch.matmul(
            R_gt[:, None, :, :],
            sym_Rs[None, :, :, :]
        )  # [M, S, 3, 3]

        points_gt_sym = torch.matmul(
            R_gt_sym, model_points.T[None, :, :]
        ).transpose(-2, -1) + t_gt[:, None, None, :]  # [M, S, P, 3]

        distances = torch.norm(
            points_pred[:, None, :, :] - points_gt_sym,
            dim=-1
        )  # [M, S, P]

        avg_distances = distances.mean(dim=-1)  # [M, S]
        min_distances, best_sym_idx = avg_distances.min(dim=-1)  # [M], [M]

        return min_distances, best_sym_idx
    
    def _compute_rotation_error(self, R_pred, R_gt):
        R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
        trace_vals = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = ((trace_vals - 1.0) / 2.0).clamp(-1.0, 1.0)
        angle_deg = torch.acos(cos_angle) * (180.0 / torch.pi)
        return angle_deg
    
    def _compute_symmetric_rotation_error(self, R_pred, R_gt, sym_Rs):
        M = R_pred.shape[0]
        S = sym_Rs.shape[0]
        
        R_gt_sym = torch.matmul(R_gt[:, None, :, :], sym_Rs[None, :, :, :])  # [M, S, 3, 3]
        
        R_diff = torch.matmul(
            R_pred[:, None, :, :],
            R_gt_sym.transpose(-1, -2)
        )  # [M, S, 3, 3]
        
        trace_vals = R_diff[:, :, 0, 0] + R_diff[:, :, 1, 1] + R_diff[:, :, 2, 2]  # [M, S]
        cos_angle = ((trace_vals - 1.0) / 2.0).clamp(-1.0, 1.0)
        angle_rad = torch.acos(cos_angle)  # [M, S]
        
        min_angle_rad, _ = angle_rad.min(dim=-1)  # [M]
        angle_deg = min_angle_rad * (180.0 / torch.pi)
        
        return angle_deg
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'pose': self.loss_addr,
            'keypoint': self.loss_keypoints,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # sym index 캐시 초기화 (pose → keypoint 순서로 호출 시 공유)
        self._cached_sym_indices = None

        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Main losses
        losses = {}
        for loss in self.losses:
            idx_use, nb_use = indices, num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, idx_use)
            l_dict = self.get_loss(loss, outputs, targets, idx_use, nb_use, **meta)
            l_dict_weighted = {}
            for k in l_dict:
                if k.startswith('metric_'):
                    l_dict_weighted[k] = l_dict[k].detach()
                elif k in self.weight_dict:
                    l_dict_weighted[k] = l_dict[k] * self.weight_dict[k]
            losses.update(l_dict_weighted)

        # Auxiliary losses
        # - share_matched_indices=True  : main indices 재사용
        # - share_matched_indices=False : layer 마다 새 매칭
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.share_matched_indices:
                    indices_aux = indices
                    num_boxes_aux = num_boxes
                else:
                    matched = self.matcher(aux_outputs, targets)
                    indices_aux = matched['indices']
                    num_boxes_aux = num_boxes
                for loss in self.losses:
                    idx_use, nb_use = indices_aux, num_boxes_aux
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, idx_use)
                    l_dict = self.get_loss(loss, aux_outputs, targets, idx_use, nb_use, **meta)
                    l_dict_kept = {}
                    for k in l_dict:
                        if k.startswith('metric_'):
                            l_dict_kept[k] = l_dict[k].detach()
                        elif k in self.weight_dict:
                            l_dict_kept[k] = l_dict[k] * self.weight_dict[k]
                    l_dict_kept = {k + f'_aux_{i}': v for k, v in l_dict_kept.items()}
                    losses.update(l_dict_kept)

        # Encoder auxiliary losses
        # enc 는 항상 자체 매칭 필요 (main decoder 와 다른 branch)
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices_enc = matched['indices']
                num_boxes_enc = num_boxes
                for loss in self.losses:
                    idx_use, nb_use = indices_enc, num_boxes_enc
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, idx_use)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, idx_use, nb_use, **meta)
                    l_dict_kept = {}
                    for k in l_dict:
                        if k.startswith('metric_'):
                            l_dict_kept[k] = l_dict[k].detach()
                        elif k in self.enc_weight_dict:
                            l_dict_kept[k] = l_dict[k] * self.enc_weight_dict[k]
                    l_dict_kept = {k + f'_enc_{i}': v for k, v in l_dict_kept.items()}
                    losses.update(l_dict_kept)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # Denoising losses (cls + bbox only)
        if 'dn_aux_outputs' in outputs and 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
            indices = self.get_cdn_matched_indices(dn_meta, targets)
            # DN은 자체 정규화 필요 → raw GT count 재계산
            raw_num_boxes = sum(len(t["labels"]) for t in targets)
            raw_num_boxes = torch.as_tensor([raw_num_boxes], dtype=torch.float, device=self.device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(raw_num_boxes)
            raw_num_boxes = torch.clamp(raw_num_boxes / get_world_size(), min=1).item()
            dn_num_boxes = raw_num_boxes * dn_meta['dn_num_group']

            # cls + bbox losses만 (pose, keypoint 제외)
            dn_losses = [l for l in self.losses if l in ('vfl', 'mal', 'boxes')]

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in dn_losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format in ('iou', 'iou_pose'):
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou).clamp(min=0)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError(f'unknown boxes_weight_format: {self.boxes_weight_format}')

        # Default values for score target (VFL/MAL): IoU only
        values = iou

        # Pose-aware score target: values = ADD-R based pose_quality (bbox 배제)
        if self.boxes_weight_format == 'iou_pose' and self._has_pose_info_available():
            with torch.no_grad():
                values = self._compute_pose_quality(outputs, targets, indices)  # [M] in [0,1]

        if loss in ('boxes',):
            meta = {'boxes_weight': iou}   # bbox loss is always weighted by raw IoU
        elif loss in ('vfl', 'mal'):
            meta = {'values': values}
        else:
            meta = {}
        return meta

    def _has_pose_info_available(self):
        """Pose quality requires sym_cache + points_3d_cache + label mapping."""
        return (bool(self.sym_cache) and bool(self.points_3d_cache)
                and bool(self.mscoco_label2category))

    def _compute_pose_quality(self, outputs, targets, indices):
        """Compute per-matched-query ADD-R based pose quality in [0, 1].

        quality = clamp(1 - ADD-R/diameter / pose_quality_scale, 0, 1)   # BOP-style
        Sym-aware (min over symmetry orbit).
        """
        idx = self._get_src_permutation_idx(indices)
        pred_R = outputs['pred_rotations'][idx].reshape(-1, 3, 3).detach()
        target_poses = torch.cat([t['poses'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)], dim=0)
        R_gt = target_poses[:, 3:12].reshape(-1, 3, 3)
        M = pred_R.shape[0]
        if M == 0:
            return torch.zeros(0, device=self.device)

        cam_K = torch.cat([t['cam_K'][j] for t, (_, j) in zip(targets, indices)], dim=0).reshape(-1, 3, 3)
        t_gt_mm = self._c2t_gt(target_poses[:, :3], cam_K)
        orig_w, orig_h = self._get_orig_size(targets)
        pred_boxes = outputs['pred_boxes'][idx].detach()
        t_pred_mm = self._c2t_pred(
            outputs['pred_translations'][idx].detach(), cam_K, pred_boxes, orig_w, orig_h)

        qualities = torch.zeros(M, device=self.device)
        # Group by class (one iter per unique class, not per instance) to amortize
        # dict lookups and eliminate per-instance .item() syncs. Inside each class
        # group we vectorize over M_cls instances × S symmetries.
        unique_labels = torch.unique(target_labels)
        for cls_id in unique_labels:
            lab = int(cls_id.item())  # 1 sync per class (not per instance)
            mask = (target_labels == cls_id)
            sR = self.sym_cache.get(lab)
            if sR is None or sR.shape[0] == 0:
                sR = torch.eye(3, device=self.device).unsqueeze(0)
            model_pts = self.points_3d_cache.get(lab)
            if model_pts is None:
                continue
            model_pts = model_pts.to(self.device)
            diameter = float(self.diameter_cache.get(lab, 100.0))
            R_gt_cls = R_gt[mask]                                        # [M_cls, 3, 3]
            pred_R_cls = pred_R[mask]                                    # [M_cls, 3, 3]
            t_gt_cls = t_gt_mm[mask]                                     # [M_cls, 3]
            t_pred_cls = t_pred_mm[mask]                                 # [M_cls, 3]
            # ADD-R: [M_cls, S] distances, then min over S
            R_gs = torch.matmul(R_gt_cls.unsqueeze(1), sR.unsqueeze(0))  # [M_cls, S, 3, 3]
            # pts_pred [M_cls, P, 3]
            pts_pred = torch.einsum('mij,pj->mpi', pred_R_cls, model_pts) + t_pred_cls.unsqueeze(1)
            # pts_gs  [M_cls, S, P, 3]
            pts_gs = torch.einsum('msij,pj->mspi', R_gs, model_pts) + t_gt_cls.unsqueeze(1).unsqueeze(1)
            dist = (pts_pred.unsqueeze(1) - pts_gs).norm(dim=-1).mean(dim=-1)  # [M_cls, S]
            addr_rel = dist.min(dim=-1).values / max(diameter, 1e-3)           # [M_cls]
            qualities[mask] = torch.clamp(1.0 - addr_rel / self.pose_quality_scale, 0.0, 1.0)

        return qualities

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """CDN denoising의 positive query와 GT를 매칭."""
        dn_positive_idx = dn_meta["dn_positive_idx"]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((
                    torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices
