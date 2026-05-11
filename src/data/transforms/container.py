"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from PIL import Image as PILImage

import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

from ._transforms import EmptyTransform, Image, BoundingBoxes, Mask, Pose
from ...core import register, GLOBAL_CONFIG


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop("type")
                    transfom = getattr(
                        GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"]
                    )(**op)
                    transforms.append(transfom)
                    op["type"] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError("")
        else:
            transforms = [
                EmptyTransform(),
            ]

        super().__init__(transforms=transforms)

        if policy is None:
            policy = {"name": "default"}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy["name"])(*inputs)

    def get_forward(self, name):
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
            "stop_sample": self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def clone_image_target(self, sample):
        """
        (image, target, dataset) 형태의 sample에서, dataset 복제는 제외하고
        image와 target(dict)만 복제하는 헬퍼 함수.
        torchvision.datapoints (Image, BoundingBox, Mask 등)이 들어있는 경우
        copy.deepcopy() 대신 각각 .clone() 등을 사용해 복제한다.
        """
        image, target, dataset = sample

        # ─────────────────────────────────────────────────────────────
        # 1) image 복제
        #   - PIL.Image : image.copy()
        #   - Tensor or Datapoint : .clone() 사용
        # ─────────────────────────────────────────────────────────────
        if isinstance(image, Image):
            # torchvision.datapoints.Image => .clone()
            cloned_image = image.clone()
        elif isinstance(image, torch.Tensor):
            # 일반 tensor라면 .clone()
            cloned_image = image.clone()
        elif isinstance(image, PILImage.Image):
            # PIL.Image 계열이면 copy()
            cloned_image = image.copy()
        else:
            # 그 외엔 상황에 따라 deepcopy 시도 (필요 시 예외처리)
            cloned_image = copy.deepcopy(image)

        # ─────────────────────────────────────────────────────────────
        # 2) target 복제
        #   - target 내부에 Tensor, Datapoint, PIL.Image 등이 섞여 있을 수 있음
        #   - dict 항목별로 분기 처리
        # ─────────────────────────────────────────────────────────────
        cloned_target = {}
        for k, v in target.items():
            # case 1) Datapoint (BoundingBox, Mask, Image, Video...)
            if isinstance(v, (BoundingBoxes, Mask, Pose, Image)):
                cloned_target[k] = v.clone()
            # case 2) 일반 Tensor
            elif isinstance(v, torch.Tensor):
                cloned_target[k] = v.clone()
            # case 3) PIL.Image
            elif hasattr(v, "copy"):
                cloned_target[k] = v.copy()
            else:
                # 나머지는 copy.deepcopy 시도 or 그냥 그대로 참조
                # 필요에 따라 골라 사용
                cloned_target[k] = copy.deepcopy(v)

        # dataset은 그대로 참조만 유지
        return cloned_image, cloned_target

    # Transforms that may remove or invalidate masks and need rollback protection.
    # Pixel-only augmentations (ColorJitter, GaussianNoise, etc.) never touch masks,
    # so cloning before/after them is pure overhead (~40ms/sample with 20 transforms).
    _SPATIAL_TRANSFORMS = frozenset(
        {
            "CopyPasteSingleClass",
            "PoseAugmentation",
            "RandomCoarseDropout",
            "RandomObjectOcclusion",
            "Mosaic",
            "RandomRotateExpand",
        }
    )

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        _, _, dataset = sample

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]

        # epoch이 리스트면 [start, stop] — start 전이거나 stop 이후면 끔
        if isinstance(policy_epoch, (list, tuple)) and len(policy_epoch) == 2:
            start_epoch, stop_epoch = policy_epoch
            aug_active = start_epoch <= cur_epoch < stop_epoch
        else:
            # 기존 방식: epoch 이후에 끔
            start_epoch, stop_epoch = 0, policy_epoch
            aug_active = cur_epoch < stop_epoch

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and not aug_active:
                pass
            else:
                t_name = type(transform).__name__
                needs_guard = t_name in self._SPATIAL_TRANSFORMS

                if needs_guard:
                    prev_image, prev_target = self.clone_image_target(sample)
                    prev_sample = (prev_image, prev_target, dataset)

                sample = transform(sample)

                if needs_guard:
                    target = sample[1]
                    cam_K = target.get("cam_K")
                    cam_K_empty = cam_K is not None and hasattr(cam_K, "__len__") and len(cam_K) == 0

                    # Pick the primary validity key from the dataset's return_masks
                    # setting: True → check masks (original behavior), False → check
                    # boxes. This matches what the dataset is actually tracking.
                    if getattr(dataset, "return_masks", True):
                        masks = target.get("masks")
                        primary_empty = masks is None or (hasattr(masks, "__len__") and len(masks) == 0)
                        primary_name = "masks"
                    else:
                        boxes = target.get("boxes")
                        primary_empty = boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0)
                        primary_name = "boxes"

                    if primary_empty or cam_K_empty:
                        reason = primary_name if primary_empty else "cam_K"
                        print(
                            f"Invalid target detected after {t_name} ({reason} empty), rolling back"
                        )
                        sample = prev_sample
        return sample

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_sample = self.policy["sample"]

        for transform in self.transforms:
            if (
                type(transform).__name__ in policy_ops
                and self.global_samples >= policy_sample
            ):
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
