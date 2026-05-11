"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py
----------------------------------------------------------------------
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
----------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.

"""

import sys
import math
from typing import Iterable, Dict, List

import torch
import numpy as np
from sklearn.metrics import auc
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
import torch.nn.functional as F

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

import torch
import numpy as np


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    lr_scheduler = kwargs.get("lr_scheduler", None)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        # Extract per-image cam_K from targets
        cam_K = None
        if targets and 'cam_K' in targets[0] and len(targets[0]['cam_K']) > 0:
            cam_K = torch.stack([t['cam_K'][0].reshape(3, 3) for t in targets])  # [B, 3, 3]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, cam_K=cam_K, targets=targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(v for k, v in loss_dict.items() if not k.startswith("metric_"))
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, cam_K=cam_K, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(
                v for k, v in loss_dict.items() if not k.startswith("metric_")
            )
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

        # ema
        if ema is not None:
            ema.update(model)

        # LR scheduling: iter-based (FlatCosineAnnealingLR) or legacy warmup
        if lr_scheduler is not None and getattr(lr_scheduler, '_iter_based', False):
            lr_scheduler.step(global_step, optimizer)
        elif lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        metrics_reduced = {
            k: v for k, v in loss_dict_reduced.items() if k.startswith("metric_")
        }
        losses_reduced = {
            k: v for k, v in loss_dict_reduced.items() if not k.startswith("metric_")
        }
        loss_value = sum(losses_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(losses_reduced)
            sys.exit(1)

        metric_logger.update(**metrics_reduced)
        metric_logger.update(loss=loss_value, **losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in losses_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)
            for k, v in metrics_reduced.items():
                writer.add_scalar(f"Metric/{k}", v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
):
    """Evaluation function — runs criterion loss and COCO eval only."""
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    mscoco_label2category = getattr(criterion, "mscoco_label2category", {})

    if coco_evaluator is not None and hasattr(coco_evaluator, "set_label_mapping"):
        coco_evaluator.set_label_mapping(mscoco_label2category)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    iou_types = coco_evaluator.iou_types

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        cam_K = None
        if targets and 'cam_K' in targets[0] and len(targets[0]['cam_K']) > 0:
            cam_K = torch.stack([t['cam_K'][0].reshape(3, 3) for t in targets])
        outputs = model(samples, cam_K=cam_K)
        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        metrics_reduced_eval = {k: v for k, v in loss_dict_reduced.items() if k.startswith("metric_")}
        losses_reduced_eval = {k: v for k, v in loss_dict_reduced.items() if not k.startswith("metric_")}
        loss_value = sum(losses_reduced_eval.values())
        metric_logger.update(**metrics_reduced_eval)
        metric_logger.update(loss=loss_value, **losses_reduced_eval)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator