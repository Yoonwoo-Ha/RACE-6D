"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import os
import yaml

from ...core import register


__all__ = ['RACE6DPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RACE6DPostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'category_file',
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        vis_enc=True,
        category_file=None,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        self.vis_enc = vis_enc

        # Load category mapping from YAML
        if category_file is not None and os.path.exists(category_file):
            with open(category_file, 'r') as f:
                category_config = yaml.safe_load(f)
                mscoco_category2name = category_config['category2name']

        self.mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
        self.mscoco_label2category = {v: k for k, v in self.mscoco_category2label.items()}

        # Vectorized label→category lookup table (avoids Python loop + .item() GPU→CPU sync)
        if self.mscoco_label2category:
            max_label = max(self.mscoco_label2category.keys())
            lut = torch.zeros(max_label + 1, dtype=torch.long)
            for label, cat in self.mscoco_label2category.items():
                lut[label] = cat
            self.register_buffer('_label2cat_lut', lut)

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs['pred_logits']
        boxes = outputs['pred_boxes']  # [B, N, 4] cxcywh normalized

        # Top-K selection
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores_flat = scores.flatten(1)
            topk_scores, topk_flat_indices = torch.topk(scores_flat, self.num_top_queries, dim=-1)
            labels = mod(topk_flat_indices, self.num_classes)
            query_indices = topk_flat_indices // self.num_classes

            final_logits = logits.gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            )
            final_boxes = boxes.gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1])
            )
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores_max, labels = scores.max(dim=-1)
            topk_scores, query_indices = torch.topk(
                scores_max, min(self.num_top_queries, scores_max.shape[1]), dim=-1
            )
            labels = torch.gather(labels, dim=1, index=query_indices)

            final_logits = logits.gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            )
            final_boxes = boxes.gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1])
            )

        # Convert boxes: cxcywh normalized -> xyxy pixel
        boxes_xyxy = torchvision.ops.box_convert(final_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        scale = orig_target_sizes.unsqueeze(1).repeat(1, 1, 2)  # [B, 1, 4] = [W,H,W,H]
        boxes_pixel = boxes_xyxy * scale

        # Gather rotation and translation if available
        has_pose = 'pred_rotations' in outputs and 'pred_translations' in outputs
        if has_pose:
            pred_rot = outputs['pred_rotations']
            if pred_rot.dim() == 3 and pred_rot.shape[-1] == 9:
                rot_gathered = pred_rot.gather(
                    dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, 9))
                rot_matrices = rot_gathered.reshape(-1, self.num_top_queries, 3, 3)
            else:
                rot_matrices = pred_rot.gather(
                    dim=1, index=query_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))

            translations = outputs['pred_translations'].gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, 3)
            )

        # Gather keypoints if available
        has_keypoints = 'pred_keypoints' in outputs
        if has_keypoints:
            kpt_dim = outputs['pred_keypoints'].shape[-1]
            keypoints = outputs['pred_keypoints'].gather(
                dim=1, index=query_indices.unsqueeze(-1).expand(-1, -1, kpt_dim)
            )

        # Remap categories
        if self.remap_mscoco_category:
            labels = self._label2cat_lut[labels]

        # Deploy: minimal output, no dict construction
        if self.deploy_mode:
            if has_pose:
                return labels, boxes_pixel, topk_scores, rot_matrices, translations
            return labels, boxes_pixel, topk_scores

        # vis_enc=False: skip aux/enc gather, return batch tensors directly
        if not self.vis_enc:
            results = []
            for b in range(labels.shape[0]):
                result = dict(
                    labels=labels[b], boxes=boxes_pixel[b],
                    scores=topk_scores[b], logits=final_logits[b],
                )
                if has_pose:
                    result['rotations'] = rot_matrices[b]
                    result['translations'] = translations[b]
                if has_keypoints:
                    result['keypoints'] = keypoints[b]
                results.append(result)
            return results

        # vis_enc=True: also gather aux/enc outputs
        aux_outputs_gathered = []
        enc_outputs_gathered = []
        if 'aux_outputs' in outputs:
            for aux in outputs['aux_outputs']:
                aux_outputs_gathered.append(self._gather_aux_outputs(
                    aux, query_indices, orig_target_sizes))
        if 'enc_aux_outputs' in outputs:
            for enc_aux in outputs['enc_aux_outputs']:
                enc_outputs_gathered.append(self._gather_aux_outputs(
                    enc_aux, query_indices, orig_target_sizes))

        results = []
        for b in range(labels.shape[0]):
            result = dict(
                labels=labels[b], boxes=boxes_pixel[b],
                scores=topk_scores[b], logits=final_logits[b],
            )
            if has_pose:
                result['rotations'] = rot_matrices[b]
                result['translations'] = translations[b]
            if has_keypoints:
                result['keypoints'] = keypoints[b]
            result['aux_outputs'] = [a[b] for a in aux_outputs_gathered]
            result['enc_outputs'] = [e[b] for e in enc_outputs_gathered]

            results.append(result)

        return results

    def _gather_aux_outputs(self, layer_output, query_indices, orig_target_sizes):
        """Gather aux/enc layer outputs using final layer's query indices."""
        gathered = {}

        # Boxes
        if 'pred_boxes' in layer_output:
            layer_boxes = layer_output['pred_boxes']
            gathered_boxes = layer_boxes.gather(
                dim=1, index=query_indices.unsqueeze(-1).repeat(1, 1, layer_boxes.shape[-1])
            )
            # Convert to pixel xyxy
            boxes_xyxy = torchvision.ops.box_convert(gathered_boxes, in_fmt='cxcywh', out_fmt='xyxy')
            scale = orig_target_sizes.repeat(1, 2).unsqueeze(1)
            gathered['boxes'] = boxes_xyxy * scale
        else:
            B, N = query_indices.shape
            gathered['boxes'] = query_indices.new_zeros(B, N, 4).float()

        # Logits & scores
        layer_logits = layer_output['pred_logits']
        logits_gathered = layer_logits.gather(
            dim=1, index=query_indices.unsqueeze(-1).repeat(1, 1, layer_logits.shape[-1])
        )

        if self.use_focal_loss:
            scores_all = F.sigmoid(layer_logits)
            scores_gathered = scores_all.gather(
                dim=1, index=query_indices.unsqueeze(-1).repeat(1, 1, scores_all.shape[-1])
            )
            layer_scores, layer_labels = scores_gathered.max(dim=-1)
        else:
            scores_all = F.softmax(layer_logits, dim=-1)[:, :, :-1]
            scores_all_max, labels_all = scores_all.max(dim=-1)
            layer_labels = torch.gather(labels_all, dim=1, index=query_indices)
            layer_scores = torch.gather(scores_all_max, dim=1, index=query_indices)

        if self.remap_mscoco_category:
            layer_labels = self._label2cat_lut[layer_labels]

        gathered['labels'] = layer_labels
        gathered['scores'] = layer_scores
        gathered['logits'] = logits_gathered

        # Return per-batch dicts
        B = query_indices.shape[0]
        results = []
        for b in range(B):
            results.append({k: v[b] for k, v in gathered.items()})
        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
