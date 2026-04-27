"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import math
import copy
import functools
import os
import json
from collections import OrderedDict
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from pytorch3d.transforms import rotation_6d_to_matrix
from pytorch3d.ops import sample_farthest_points
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob, depth_ratio_weighting_focus_center, distance2depth

from .denoising import get_pose_denoising_training_group
from ...core import register

__all__ = ['RACE6DTransformer_DQE']

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).view(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            # reference_points: [B, Nq, n_levels, 2]
            # sampling_offsets: [B, Nq, heads, total_points, 2]
            # Build per-point normalizer and ref point based on level assignment
            offset_normalizer = torch.tensor(value_spatial_shapes,
                                             device=query.device, dtype=query.dtype)
            offset_normalizer = offset_normalizer.flip([1])  # [n_levels, 2] → (W, H)

            # Expand normalizer per point: each point belongs to a level
            norm_per_point = torch.cat([
                offset_normalizer[lvl:lvl+1].expand(n, -1)
                for lvl, n in enumerate(self.num_points_list)
            ], dim=0)  # [total_points, 2]
            norm_per_point = norm_per_point.reshape(1, 1, 1, -1, 2)

            # Expand ref points per point: each point uses its level's ref
            ref_per_point = torch.cat([
                reference_points[:, :, lvl:lvl+1, :].expand(-1, -1, n, -1)
                for lvl, n in enumerate(self.num_points_list)
            ], dim=2)  # [B, Nq, total_points, 2]
            ref_per_point = ref_per_point.unsqueeze(2)  # [B, Nq, 1, total_points, 2]

            sampling_locations = ref_per_point + sampling_offsets / norm_per_point
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output

class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        b,n,c = x1.shape
        gate_input = torch.cat([x1, x2], dim=2)
        gates = torch.sigmoid(self.gate(gate_input))
        #gate1, gate2 = gates.chunk(2, dim=-1)
        gate1 = gates[:,:,0:int(c)]
        gate2 = gates[:,:,int(c):int(2*c)]

        return self.norm((gate1 * x1 + gate2 * x2))

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default'):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        # self.norm2 = nn.LayerNorm(d_model)

        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention (rf-detr style split-merge for group isolation)
        q = k = self.with_pos_embed(target, query_pos_embed)
        bs = target.shape[0]

        if attn_mask is not None:
            # DN active: attn_mask isolates DN queries from normal queries
            target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        else:
            target2, _ = self.self_attn(q, k, value=target)

        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        # target = target + self.dropout2(target2)
        # target = self.norm2(target)
        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target
    
class Integral(nn.Module):
    """Integral module for distribution to coordinate conversion"""
    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        """
        Args:
            x: [B, num_queries, num_kpts*2*(reg_max+1)]
            project: weighting function
        Returns:
            coordinates: [B, num_queries, num_kpts*2]
        """
        b, n, c = x.shape
        # Reshape and apply softmax
        X = x.reshape(b*n, -1, self.reg_max+1)
        P = F.softmax(X, dim=-1)                      # [BN, M, 33]
        coords = (P @ project.to(P)).view(b, n, -1)   # [B,N,M]
        return coords
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1, reg_max=32, r_min=0.5, r_max=2.0, sharp=2.0, reg_scale=1.0):
        super(TransformerDecoder, self).__init__()
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.integral = Integral(reg_max=reg_max)
        project = depth_ratio_weighting_focus_center(reg_max, r_min, r_max, sharp, r_min.device)  # [reg_max+1, 1]
        self.register_buffer('project_buf', project.view(-1,1))  # [reg_max+1,1]
        self.reg_scale = reg_scale
    
    def dec_combine_rotation(self, rotation, rotation_ref):
        with torch.amp.autocast('cuda', enabled=False):
            rotation_so3 = rotation_6d_to_matrix(rotation.float())
            combined_rot_so3 = torch.matmul(rotation_so3, rotation_ref.float())
        return combined_rot_so3

    def forward(self,
                target,
                ref_points_unact,
                ref_kpts,
                ref_trans,
                ref_rot,
                memory,
                memory_spatial_shapes,
                score_head,
                bbox_head,
                kpt_head,
                trans_xy_head,
                reg_head,
                rot_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                return_intermediate=False):
        
        dec_out_logits = []
        dec_out_bboxes = []
        dec_out_kpts = []
        dec_out_trans = []
        dec_out_rots = []

        ref_points_detach = torch.sigmoid(ref_points_unact)
        ref_kpts_detach = ref_kpts
        pred_depth_initial = pred_depth_detach = ref_trans[..., -1:]  # [B, N, 1]
        ref_trans_xy = ref_trans_xy_detach = ref_trans[..., :2] # [B, N, 2]
        ref_rot_detach = ref_rot # [B, N, 3, 3]
        output = target
        bs, num_q, _ = target.shape
        output_detach = pred_depth_undetach = 0
        
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2) # [B,N,1,4]
            query_pos_embed = query_pos_head(torch.concat([ref_points_detach, pred_depth_detach], dim=-1))  # [B,N,H]

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            trans_xy_embed = trans_xy_head[i](output)
            inter_ref_trans_xy = trans_xy_embed + ref_trans_xy_detach  # [B,N,2]
            # Keypoints: bbox-relative (raw additive refinement)
            kpt_embed = kpt_head[i](output)
            inter_ref_kpts = kpt_embed + ref_kpts_detach  # [B,N,64]
            with torch.amp.autocast('cuda', enabled=False):
                rot_input = torch.cat([output.float(), inter_ref_kpts.detach().float()], dim=-1)
                rot_embed = rot_head[i](rot_input)  # [B,N,6] float32
                inter_ref_rot = self.dec_combine_rotation(rot_embed, ref_rot_detach) # [B,N,3,3]

            scores = score_head[i](output)
            pred_depth_dist = reg_head[i](output + output_detach) + pred_depth_undetach  # [B,N,reg_max+1]
            inter_ref_depth = distance2depth(pred_depth_initial, self.integral(pred_depth_dist, self.project_buf).reshape(bs, num_q, 1), self.reg_scale) # [B,N,1]

            inter_ref_trans = torch.cat([inter_ref_trans_xy, inter_ref_depth], dim=-1)  # [B,N,3]

            if self.training or return_intermediate:
                dec_out_logits.append(scores)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    dec_out_kpts.append(inter_ref_kpts)
                    dec_out_trans.append(inter_ref_trans)
                    dec_out_rots.append(inter_ref_rot.flatten(-2))
                else:
                    dec_out_bboxes.append(torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
                    dec_out_kpts.append(kpt_embed + ref_kpts)
                    dec_out_trans.append(torch.cat([trans_xy_embed + ref_trans_xy, inter_ref_depth], dim=-1))
                    dec_out_rots.append(self.dec_combine_rotation(rot_embed, ref_rot).flatten(-2))

            elif i == self.eval_idx:
                dec_out_logits.append(scores)
                dec_out_kpts.append(inter_ref_kpts)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_trans.append(inter_ref_trans)
                dec_out_rots.append(inter_ref_rot.flatten(-2))
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
            ref_kpts = inter_ref_kpts
            ref_kpts_detach = inter_ref_kpts.detach()
            ref_trans_xy = inter_ref_trans_xy
            ref_trans_xy_detach = inter_ref_trans_xy.detach()
            ref_rot = inter_ref_rot
            ref_rot_detach = inter_ref_rot.detach()
            output_detach = output.detach()
            pred_depth_undetach = pred_depth_dist
            pred_depth_detach = inter_ref_depth.detach()

        return torch.stack(dec_out_logits), torch.stack(dec_out_bboxes), torch.stack(dec_out_kpts), torch.stack(dec_out_trans), torch.stack(dec_out_rots)


@register()
class RACE6DTransformer_DQE(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size', 'coco_path', 'category_file']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="silu",
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 vis_enc=False,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=1.0,
                 num_keypoints=32,
                 mlp_act='silu',
                 coco_path=None,
                 category_file=None,
                 max_sym_disc_step=0.01,
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 r_min=0.5,
                 r_max=2.0,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        # eval_spatial_size: 실제 모델 입력 크기 (anchors/pos embed 용)
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.vis_enc = vis_enc
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.num_keypoints = num_keypoints

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.r_min = nn.Parameter(torch.tensor(r_min), requires_grad=False)
        self.r_max = nn.Parameter(torch.tensor(r_max), requires_grad=False)
        self.sharp = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_layers, eval_idx, reg_max, self.r_min, self.r_max, self.sharp, reg_scale)

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4 + 1, 2 * hidden_dim, hidden_dim, 2) # (xywh + tz_m)

        # Encoder heads (single shared branch)
        enc_score_cls = 1 if query_select_method == 'agnostic' else num_classes
        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim)),
        ]))
        self.enc_score_head = nn.Linear(hidden_dim, enc_score_cls)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, mlp_act)
        self.enc_kpt_head = MLP(hidden_dim, hidden_dim, 2 * num_keypoints, 3, mlp_act)
        self.enc_trans_head = MLP(hidden_dim, hidden_dim, 3, 3, mlp_act)
        self.enc_rot_head = MLP(hidden_dim + 2 * num_keypoints, hidden_dim, 6, 3, mlp_act)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3, mlp_act) for _ in range(num_layers) # (cx, cy, w, h)
        ])
        self.dec_kpt_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2 * num_keypoints, 3, mlp_act) for _ in range(num_layers) # rx, ry
        ])
        self.dec_trans_xy_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2, 3, mlp_act) for _ in range(num_layers) # rx, ry
        ])
        self.dec_reg_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, reg_max + 1 , 3, mlp_act) for _ in range(num_layers) # 33 bins for depth
        ])
        self.dec_rot_head = nn.ModuleList([
            MLP(hidden_dim + 2 * num_keypoints, hidden_dim, 6, 3, mlp_act) for _ in range(num_layers)
        ]) # rotation-6d

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors_bbox, valid_mask = self._generate_anchors_bbox()
            self.register_buffer('anchors_bbox', anchors_bbox)
            self.register_buffer('valid_mask', valid_mask)

        # pose info (buffer로 저장 → checkpoint에 포함)
        self._has_pose_info = coco_path is not None
        if self._has_pose_info:
            self._load_pose_info(coco_path, category_file, max_sym_disc_step)

        self._reset_parameters()


    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        identity_r6d = torch.tensor([1., 0., 0., 0., 1., 0.])

        # Encoder heads 초기화
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.enc_kpt_head.layers[-1].weight, 0)
        init.constant_(self.enc_kpt_head.layers[-1].bias, 0)
        init.constant_(self.enc_trans_head.layers[-1].weight, 0)
        init.constant_(self.enc_trans_head.layers[-1].bias, 0)
        init.constant_(self.enc_rot_head.layers[-1].weight, 0)
        self.enc_rot_head.layers[-1].bias.data.copy_(identity_r6d + torch.randn(6) * 0.01)

        # Decoder heads 초기화 (헤드별 특성에 맞게 설정)
        for _cls, _box, _kpt, _trans, _reg, _rot in zip(self.dec_score_head, self.dec_bbox_head, self.dec_kpt_head, self.dec_trans_xy_head, self.dec_reg_head, self.dec_rot_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_box.layers[-1].weight, 0)
            init.constant_(_box.layers[-1].bias, 0)
            init.constant_(_kpt.layers[-1].weight, 0)
            init.constant_(_kpt.layers[-1].bias, 0)
            init.constant_(_trans.layers[-1].weight, 0)
            init.constant_(_trans.layers[-1].bias, 0)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
            init.constant_(_rot.layers[-1].weight, 0)
            _rot.layers[-1].bias.data.copy_(identity_r6d + torch.randn(6) * 0.01)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    # ==================== Pose Info Loading ====================

    def _load_pose_info(self, coco_path, category_file, max_sym_disc_step=0.01):
        """Pose 관련 데이터를 로드하고 register_buffer로 저장"""
        import yaml
        import open3d as o3d

        coco_path = os.path.expanduser(coco_path)

        # Category mapping
        mscoco_category2name = {i: str(i) for i in range(1, self.num_classes + 1)}
        if category_file and os.path.exists(category_file):
            with open(category_file, 'r') as f:
                category_config = yaml.safe_load(f) or {}
            mscoco_category2name = category_config.get('category2name', mscoco_category2name)
        mscoco_category2name = {int(k): v for k, v in mscoco_category2name.items()}

        self.mscoco_category2label = {k: idx for idx, k in enumerate(mscoco_category2name.keys())}
        self.mscoco_label2category = {v: k for k, v in self.mscoco_category2label.items()}
        valid_label_ids = list(self.mscoco_label2category.keys())

        # 1. keypoints_3d
        keypoints_3d, edges_mask = self._load_keypoints_3d_and_edges(coco_path)
        self.register_buffer('keypoints_3d', keypoints_3d)
        self.edges_mask = edges_mask

        # 3. models_info 로드
        models_info_path = os.path.join(coco_path, 'models', 'models_info.json')
        if os.path.exists(models_info_path):
            with open(models_info_path, 'r') as f:
                original_models_info = json.load(f)
        else:
            print(f'File not found: {models_info_path}')
            original_models_info = {}

        self.models_info = {}

        # 4. Per-class data 수집 (임시 dict → 나중에 padded tensor로 변환)
        C = len(valid_label_ids)
        num_model_points = 1500

        diameters_list = []
        points_3d_list = []
        sym_list = []

        for label_id in valid_label_ids:
            category_id = self.mscoco_label2category[label_id]
            category_key = str(category_id)

            if category_key not in original_models_info:
                print(f"Category {category_key} not found in {models_info_path}")
                diameters_list.append(0.0)
                points_3d_list.append(torch.zeros(num_model_points, 3))
                sym_list.append(torch.eye(3).unsqueeze(0))
                continue

            obj_info = original_models_info[category_key]
            self.models_info[label_id] = obj_info

            # Diameter
            diameter_mm = obj_info.get('diameter', 0)
            diameters_list.append(diameter_mm)

            # 3D model points (FPS sampled)
            model_path = os.path.join(coco_path, 'models', f'obj_{category_id:06d}.ply')
            if os.path.exists(model_path):
                points = self._load_model_points(o3d, model_path, num_model_points)
                points_3d_list.append(points)
            else:
                print(f'Model file not found: {model_path}')
                points_3d_list.append(torch.zeros(num_model_points, 3))

            # Symmetry transforms
            sym_mats = self._build_symmetry_matrices(obj_info, max_sym_disc_step)
            sym_list.append(torch.stack(sym_mats, 0))  # [S_i, 3, 3]

        # 5. Padded tensors로 변환 + register_buffer
        self.register_buffer('diameters', torch.tensor(diameters_list, dtype=torch.float32))  # [C]
        self.register_buffer('points_3d', torch.stack(points_3d_list, 0))  # [C, 1500, 3]

        # sym_rotations: class별 대칭 수가 다르므로 padding
        max_S = max(s.shape[0] for s in sym_list)
        sym_padded = torch.zeros(C, max_S, 3, 3)
        sym_counts = torch.zeros(C, dtype=torch.long)
        for i, s in enumerate(sym_list):
            S_i = s.shape[0]
            sym_padded[i, :S_i] = s
            sym_counts[i] = S_i
        self.register_buffer('sym_rotations', sym_padded)  # [C, max_S, 3, 3]
        self.register_buffer('sym_counts', sym_counts)      # [C]

    @staticmethod
    def _load_cam_K(coco_path):
        """scene_camera.json에서 cam_K 로드"""
        test_path = os.path.join(coco_path, 'test')
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test path not found: {test_path}")
        test_folders = sorted([f for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))])
        if len(test_folders) == 0:
            raise FileNotFoundError(f"No folders found in {test_path}")
        scene_camera_path = os.path.join(test_path, test_folders[0], 'scene_camera.json')
        with open(scene_camera_path, 'r') as f:
            scene_camera = json.load(f)
        first_key = list(scene_camera.keys())[0]
        cam_K_list = scene_camera[first_key]['cam_K']
        return torch.tensor(cam_K_list, dtype=torch.float32)

    @staticmethod
    def _load_model_points(o3d, file_path, num_samples=1500):
        """PLY 파일에서 3D points 로드 + FPS 샘플링"""
        pcd = o3d.io.read_point_cloud(file_path)
        points = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        if len(points) == 0:
            return torch.zeros(num_samples, 3)
        if len(points) >= num_samples:
            sampled, _ = sample_farthest_points(points.unsqueeze(0), K=num_samples)
            return sampled.squeeze(0)
        else:
            sampled = points
            while len(sampled) < num_samples:
                remaining = num_samples - len(sampled)
                sampled = torch.cat([sampled, points[:remaining]], dim=0)
            return sampled[:num_samples]

    @staticmethod
    def _load_keypoints_3d_and_edges(coco_path):
        """JSON 파일에서 32개 3D 키포인트와 edges_mask 로드"""
        keypoints_path = os.path.join(coco_path, 'cached_keypoints_3d_edge.json')
        with open(keypoints_path, 'r') as f:
            data = json.load(f)
        keypoints_3d_data = data['keypoints_3d']
        edges_mask_data = data['edges_mask']
        max_class_id = max(int(k) for k in keypoints_3d_data.keys())
        keypoints_3d = torch.zeros(max_class_id + 1, 32, 3)
        for class_id_str, keypoints in keypoints_3d_data.items():
            class_id = int(class_id_str)
            keypoints_3d[class_id - 1] = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints_3d, edges_mask_data

    @staticmethod
    def _build_symmetry_matrices(obj_info, max_sym_disc_step=0.01):
        """models_info에서 대칭 변환 행렬 리스트 생성.
        BOP toolkit 표준: max_sym_disc_step=0.01 (diameter 대비 비율)
        → discrete_steps = ceil(π / 0.01) = 315, angular_step ≈ 1.14°
        """
        sym_mats = [torch.eye(3)]  # identity 항상 포함

        # 이산 대칭
        if 'symmetries_discrete' in obj_info:
            for sym in obj_info['symmetries_discrete']:
                sym_mat = torch.tensor(sym, dtype=torch.float32).reshape(4, 4)
                sym_mats.append(sym_mat[:3, :3])

        # 연속 대칭 (BOP toolkit 공식: discrete_steps = ceil(π / max_sym_disc_step))
        if 'symmetries_continuous' in obj_info and max_sym_disc_step is not None:
            for sym in obj_info['symmetries_continuous']:
                axis = np.array(sym['axis'], dtype=np.float64)
                axis_norm = axis / np.linalg.norm(axis)
                num_steps = int(np.ceil(np.pi / max_sym_disc_step))
                actual_step = 2.0 * np.pi / num_steps
                K = np.array([
                    [0, -axis_norm[2], axis_norm[1]],
                    [axis_norm[2], 0, -axis_norm[0]],
                    [-axis_norm[1], axis_norm[0], 0]
                ])
                for i in range(1, num_steps):
                    angle = i * actual_step
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    sym_mats.append(torch.tensor(R, dtype=torch.float32))

        return sym_mats

    # Dict conversion helpers (criterion 등 외부에서 사용)
    def get_sym_cache(self) -> Dict[int, torch.Tensor]:
        """buffer → {label_id: [S, 3, 3]} dict 변환"""
        result = {}
        valid_label_ids = list(self.mscoco_label2category.keys())
        for i, label_id in enumerate(valid_label_ids):
            S = int(self.sym_counts[i].item())
            result[label_id] = self.sym_rotations[i, :S]  # [S, 3, 3]
        return result

    def get_points_3d_cache(self) -> Dict[int, torch.Tensor]:
        """buffer → {label_id: [P, 3]} dict 변환"""
        result = {}
        valid_label_ids = list(self.mscoco_label2category.keys())
        for i, label_id in enumerate(valid_label_ids):
            result[label_id] = self.points_3d[i]  # [1500, 3]
        return result

    def get_diameter_cache(self) -> Dict[int, float]:
        """buffer → {label_id: float} dict 변환"""
        result = {}
        valid_label_ids = list(self.mscoco_label2category.keys())
        for i, label_id in enumerate(valid_label_ids):
            result[label_id] = self.diameters[i].item()
        return result

    # ==========================================================

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        """최적화된 버전 - 배치 처리"""
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        feat_flattens = []
        spatial_shapes = []
        
        for feat in proj_feats:
            B, C, H, W = feat.shape
            # contiguous view 사용으로 메모리 복사 최소화
            feat_flat = feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
            feat_flattens.append(feat_flat)
            spatial_shapes.append([H, W])
        
        # 미리 총 길이 계산하여 메모리 할당 최적화
        total_length = sum(f.shape[1] for f in feat_flattens)
        B, C = feat_flattens[0].shape[0], feat_flattens[0].shape[2]
        
        feat_flatten = torch.empty(B, total_length, C, 
                                device=feat_flattens[0].device, 
                                dtype=feat_flattens[0].dtype)
        
        start_idx = 0
        for feat_flat in feat_flattens:
            length = feat_flat.shape[1]
            feat_flatten[:, start_idx:start_idx+length] = feat_flat
            start_idx += length
        
        return feat_flatten, spatial_shapes

    def _generate_anchors_bbox(self, spatial_shapes=None, grid_size=0.05, 
                                    dtype=torch.float32, device='cpu'):
        """최적화된 앵커 생성"""
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])
        all_anchors = []
        
        for lvl, (h, w) in enumerate(spatial_shapes):
            # 더 효율적인 그리드 생성
            y_coords = torch.arange(h, dtype=dtype, device=device)
            x_coords = torch.arange(w, dtype=dtype, device=device)
            
            # meshgrid 최적화
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # 한번에 정규화
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy_norm = (grid_xy + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
            
            # 한번에 wh 계산
            wh = torch.full_like(grid_xy_norm, grid_size * (2.0 ** lvl))
            
            # 한번에 concat
            lvl_anchors = torch.cat([grid_xy_norm, wh], dim=-1).view(1, h * w, 4)
            all_anchors.append(lvl_anchors)

        anchors = torch.cat(all_anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors.masked_fill_(~valid_mask, float('inf'))

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors_bbox, valid_mask = self._generate_anchors_bbox(spatial_shapes, device=memory.device)
        else:
            anchors_bbox = self.anchors_bbox
            valid_mask = self.valid_mask

        memory = valid_mask.to(memory.dtype) * memory

        # Encoder projection + top-K query selection
        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits, self.num_queries, anchors=anchors_bbox)

        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
        enc_topk_bboxes = torch.sigmoid(enc_topk_bbox_unact)
        enc_topk_trans = self.enc_trans_head(enc_topk_memory)
        enc_topk_keypoints = self.enc_kpt_head(enc_topk_memory)
        with torch.amp.autocast('cuda', enabled=False):
            enc_rot_input = torch.cat([enc_topk_memory.float(), enc_topk_keypoints.detach().float()], dim=-1)
            enc_topk_rots = rotation_6d_to_matrix(self.enc_rot_head(enc_rot_input))

        enc_topk_bboxes_list, enc_topk_kpts_list, enc_topk_trans_list, enc_topk_rots_list, enc_topk_logits_list = [], [], [], [], []
        if self.vis_enc:
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_kpts_list.append(enc_topk_keypoints)
            enc_topk_logits_list.append(enc_topk_logits)
            enc_topk_trans_list.append(enc_topk_trans.float())
            enc_topk_rots_list.append(enc_topk_rots.float().flatten(-2))

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        enc_topk_keypoints = enc_topk_keypoints.detach()
        enc_topk_trans = enc_topk_trans.detach()
        enc_topk_rots = enc_topk_rots.detach()

        return (content, enc_topk_bbox_unact, enc_topk_keypoints, enc_topk_trans, enc_topk_rots,
                enc_topk_bboxes_list, enc_topk_kpts_list, enc_topk_trans_list, enc_topk_rots_list, enc_topk_logits_list)

    def _select_topk(self, memory, outputs_logits, topk, anchors=None):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        B = memory.shape[0]

        # Gather: memory + logits + anchors (bbox computed after topk)
        parts = [memory, outputs_logits]
        if anchors is not None:
            parts.append(anchors.expand(B, -1, -1))
        all_outputs = torch.cat(parts, dim=-1)

        total_dim = all_outputs.shape[-1]
        topk_all = all_outputs.gather(dim=1,
            index=topk_ind.unsqueeze(-1).expand(-1, -1, total_dim))

        hidden_dim = memory.shape[-1]
        num_classes = outputs_logits.shape[-1]

        start_idx = 0
        topk_memory = topk_all[:, :, start_idx:start_idx+hidden_dim]
        start_idx += hidden_dim

        topk_logits = topk_all[:, :, start_idx:start_idx+num_classes]
        start_idx += num_classes

        if anchors is not None:
            anchor_dim = anchors.shape[-1]
            topk_anchors = topk_all[:, :, start_idx:start_idx+anchor_dim]
            return topk_memory, topk_logits, topk_anchors

        return topk_memory, topk_logits
    
    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        (init_ref_contents, init_ref_points_unact, init_ref_keypoints, init_ref_trans, init_ref_rots,
         enc_topk_bboxes_list, enc_topk_kpts_list, enc_topk_trans_list, enc_topk_rots_list, enc_topk_logits_list) = self._get_decoder_input(memory, spatial_shapes)

        # === Pose-aware Denoising ===
        attn_mask = None
        dn_meta = None
        if self.training and self.num_denoising > 0 and targets is not None and self._has_pose_info:
            num_queries_total = init_ref_contents.shape[1]

            enc_bbox = torch.sigmoid(init_ref_points_unact).detach()
            enc_kpt = init_ref_keypoints.detach()
            enc_trans = init_ref_trans.detach()
            enc_rot = init_ref_rots.detach()

            # cam_K is calibrated in original image coordinates, so pixel-level
            # conversions inside DN training (_enc_trans_to_3d) must use each
            # sample's pre-resize size. DN is training-only, so targets always
            # carry per-image orig_size.
            _orig = targets[0]['orig_size']  # BOP convention: [W, H]
            img_w, img_h = int(_orig[0].item()), int(_orig[1].item())

            dn_result = get_pose_denoising_training_group(
                targets,
                enc_bbox, enc_kpt, enc_trans, enc_rot,
                self.get_points_3d_cache(), self.models_info, self.mscoco_label2category,
                self.num_classes, num_queries_total,
                self.denoising_class_embed,
                img_w=img_w, img_h=img_h,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )

            if dn_result[0] is not None:
                dn_logits, dn_bbox_unact, dn_kpt, dn_trans, dn_rot, attn_mask, dn_meta = dn_result

                # DN content = class embedding (rtdetrv2 방식)
                init_ref_contents = torch.cat([dn_logits, init_ref_contents], dim=1)
                init_ref_points_unact = torch.cat([dn_bbox_unact, init_ref_points_unact], dim=1)
                init_ref_keypoints = torch.cat([dn_kpt, init_ref_keypoints], dim=1)
                init_ref_trans = torch.cat([dn_trans, init_ref_trans], dim=1)
                init_ref_rots = torch.cat([dn_rot, init_ref_rots], dim=1)

        # decoder
        out_logits, out_bboxes, out_kpts_raw, out_trans_raw, out_rots_raw = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            init_ref_keypoints,
            init_ref_trans,
            init_ref_rots,
            memory,
            spatial_shapes,
            self.dec_score_head,
            self.dec_bbox_head,
            self.dec_kpt_head,
            self.dec_trans_xy_head,
            self.dec_reg_head,
            self.dec_rot_head,
            self.query_pos_head,
            attn_mask=attn_mask,
            return_intermediate=self.vis_enc)

        # === Split DN outputs from normal outputs ===
        if self.training and dn_meta is not None:
            dn_num_split = dn_meta['dn_num_split']
            dn_out_logits, out_logits = torch.split(out_logits, dn_num_split, dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_num_split, dim=2)
            dn_out_kpts_raw, out_kpts_raw = torch.split(out_kpts_raw, dn_num_split, dim=2)
            dn_out_trans_raw, out_trans_raw = torch.split(out_trans_raw, dn_num_split, dim=2)
            dn_out_rots_raw, out_rots_raw = torch.split(out_rots_raw, dn_num_split, dim=2)

        # === Pose Processing ===
        out_kpts = out_kpts_raw.float()
        out_trans = out_trans_raw.float()
        out_rots = out_rots_raw.float()

        if self.vis_enc:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_keypoints': out_kpts[-1], 'pred_translations': out_trans[-1], 'pred_rotations': out_rots[-1]}
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_kpts[:-1], out_trans[:-1], out_rots[:-1])
            # Encoder outputs also use groups: [B, G*N, ...] (rf-detr style)
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list, enc_topk_kpts_list, enc_topk_trans_list, enc_topk_rots_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_keypoints': out_kpts[-1], 'pred_translations': out_trans[-1], 'pred_rotations': out_rots[-1]}

        # DN outputs
        if self.training and dn_meta is not None:
            out['dn_aux_outputs'] = self._set_aux_loss(
                dn_out_logits, dn_out_bboxes,
                dn_out_kpts_raw.float(), dn_out_trans_raw.float(), dn_out_rots_raw.float())
            out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_keypoints, outputs_translations, outputs_rotations):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_keypoints': c, 'pred_translations': d, 'pred_rotations': e}
                for a, b, c, d, e in zip(outputs_class, outputs_coord, outputs_keypoints, outputs_translations, outputs_rotations)]