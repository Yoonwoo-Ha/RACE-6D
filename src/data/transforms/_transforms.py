"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image as PILImage

import numpy as np
import cv2
import os
import json

from typing import Any, Dict, List, Optional, Tuple, Union
from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes, Pose, BoundingBoxFormat
from .._misc import SanitizeBoundingBoxes
from ...core import register, GLOBAL_CONFIG


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


class AugmentationProbabilities:
    """각 증강 기법별 확률 설정 (GDRNPP 스타일)

    GDRNPP BOP2022 설정 기반:
    - base_probability = 0.8
    - 실제 적용 확률 = base_probability * adjustment
    """

    def __init__(self):
        # GDRNPP Sometimes() 확률값을 그대로 사용 (base 0.8과 곱해짐)
        self.brightness = 0.5  # Sometimes(0.5, EnhanceBrightness)
        self.contrast = 0.3  # Sometimes(0.3, EnhanceContrast)
        self.linear_contrast = 0.5  # Sometimes(0.5, LinearContrast)
        self.color_jitter = 0.3  # Sometimes(0.3, EnhanceColor) - saturation
        self.hsv_adjust = 0.3  # 추가 HSV 조정
        self.sharpen = 0.3  # Sometimes(0.3, EnhanceSharpness)
        self.motion_blur = 0.3  # motion blur
        self.gaussian_blur = 0.4  # Sometimes(0.4, GaussianBlur)
        self.gaussian_noise = 0.1  # Sometimes(0.1, AdditiveGaussianNoise)
        self.additional_noise = 0.1  # 추가 노이즈
        self.grayscale = 0.5  # Sometimes(0.5, Grayscale)
        self.random_background = 0.4  # 별도 고정값 (기본 확률 무시)
        self.coarse_dropout = 0.5  # Sometimes(0.5, CoarseDropout)
        self.rotate_expand = 0.3  # rotate expand
        self.object_occlusion = 0.3  # object occlusion
        self.add_value = 0.5  # Sometimes(0.5, Add)
        self.multiply = 0.5  # Sometimes(0.5, Multiply)


class AugmentationManager:
    def __init__(self, base_probability: float = 0.8):
        """
        Args:
            base_probability: 기본 적용 확률 (기본값: 0.8)
        """
        self.base_probability = max(0.1, min(1.0, base_probability))
        self.probs = AugmentationProbabilities()

    def get_probability(self, transform_name: str) -> float:
        """변환 기법별 실제 적용 확률 계산"""
        if hasattr(self.probs, transform_name):
            adjustment = getattr(self.probs, transform_name)

            # 배경 교체는 별도 고정값 사용
            if transform_name == "random_background":
                return adjustment

            # 나머지는 기본 확률에 조절값 적용
            final_prob = self.base_probability * adjustment
            return max(0.0, min(1.0, final_prob))

        # 정의되지 않은 변환은 기본 확률 사용
        return self.base_probability


class BackgroundColors:
    STUDIO = {
        "pure_white": [255, 255, 255],
        "white_smoke": [245, 245, 245],
        "light_gray": [211, 211, 211],
        "neutral_gray": [128, 128, 128],
        "dark_gray": [64, 64, 64],
        "pure_black": [0, 0, 0],
    }

    CHROMA = {
        "chroma_green": [0, 255, 0],
        "chroma_blue": [0, 255, 255],
    }

    INDUSTRIAL = {
        "industrial_gray": [169, 169, 169],
        "cool_gray": [144, 144, 192],
        "warm_gray": [192, 144, 144],
        "blue_gray": [104, 120, 140],
    }

    NATURAL = {
        "sky_blue": [135, 206, 235],
        "cloud_white": [236, 236, 236],
        "sand_beige": [210, 180, 140],
        "earth_brown": [139, 69, 19],
        "forest_green": [34, 139, 34],
    }

    INDOOR = {
        "wall_cream": [255, 253, 208],
        "floor_brown": [139, 115, 85],
        "ceiling_white": [248, 248, 255],
        "room_beige": [245, 245, 220],
    }


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PILImage.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        canvas_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(
                inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower()
            )
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), canvas_size=canvas_size
            )
        if self.normalize:
            inpt = inpt / torch.tensor(canvas_size[::-1]).tile(2)[None]
        return inpt


@register()
class ConvertPose(nn.Module):
    """
    Pose를 정규화된 좌표로 변환 (target['cam_K']에서 cam_K 가져옴)
    bbox 기반 상대 좌표 방식 사용
    """

    def __init__(self, normalize=False, coco_path=None) -> None:
        super().__init__()
        self.normalize = normalize
        # coco_path는 기존 config 호환성을 위해 받지만 사용하지 않음
        # cam_K는 target['cam_K']에서 가져옴

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        if isinstance(inputs, tuple):
            image, target = inputs[0], inputs[1]
            dataset_info = inputs[2] if len(inputs) > 2 else None
        else:
            target, image, dataset_info = inputs, None, None

        # boxes 정보 확인
        if "boxes" not in target:
            raise ValueError(
                "Bounding box information is required for ConvertPose transform"
            )

        # poses 정보 확인
        if "poses" not in target:
            raise ValueError("Pose information is required for ConvertPose transform")

        # cam_K from target (per-annotation)
        if "cam_K" not in target:
            raise ValueError("cam_K is not available in target")

        boxes = (
            target["boxes"].data
            if hasattr(target["boxes"], "data")
            else target["boxes"]
        )
        poses = (
            target["poses"].data
            if isinstance(target["poses"], Pose)
            else target["poses"]
        )
        cam_K = target["cam_K"][0]  # [fx, 0, cx, 0, fy, cy, 0, 0, 1]

        # 좌표 변환 (정규화)
        converted_poses = self._convert_poses(poses, boxes, cam_K)

        # 새로운 타겟 생성
        new_target = target.copy()
        new_target["poses"] = Pose(converted_poses)

        return (
            (image, new_target, dataset_info)
            if (image is not None and dataset_info is not None)
            else (image, new_target)
            if image is not None
            else new_target
        )

    def _convert_poses(self, poses, bbox_data, cam_K):
        """bbox 기반 상대 좌표 변환"""
        if isinstance(poses, torch.Tensor):
            device = poses.device
            # cam_K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            fx = cam_K[0]
            fy = cam_K[4]
            px = cam_K[2]
            py = cam_K[5]

            poses = poses.reshape(-1, 12)

            # bbox 데이터를 올바르게 처리
            if isinstance(bbox_data, torch.Tensor):
                bbox_tensor = bbox_data.to(device).reshape(-1, 4)
            else:
                bbox_tensor = torch.tensor(bbox_data, device=device).reshape(-1, 4)

            # Extract translation components
            tx = poses[:, 0]  # mm 단위
            ty = poses[:, 1]  # mm 단위
            tz = poses[:, 2]  # mm 단위

            # Calculate bounding box size
            # bbox format: [x1, y1, x2, y2] (정규화 안된 좌표값)
            x1 = bbox_tensor[:, 0]
            y1 = bbox_tensor[:, 1]
            x2 = bbox_tensor[:, 2]
            y2 = bbox_tensor[:, 3]
            wbbox = x2 - x1
            hbbox = y2 - y1
            cxbbox = (x1 + x2) / 2
            cybbox = (y1 + y2) / 2

            # Calculate relative translation coordinates
            rx = (px + (fx * tx / tz) - cxbbox) / wbbox  # 정규화된 좌표값
            ry = (py + (fy * ty / tz) - cybbox) / hbbox  # 정규화된 좌표값
            rz = tz / 1000  # 정규화된 깊이 m단위

            # Rotation (9D flattened 3x3)
            new_rot = poses[:, 3:12]

            new_tran = torch.stack([rx, ry, rz], dim=1)
            new_poses = torch.cat([new_tran, new_rot], dim=1)

        elif isinstance(poses, np.ndarray):
            cam_K_np = np.array(cam_K) if not isinstance(cam_K, np.ndarray) else cam_K
            fx = cam_K_np[0]
            fy = cam_K_np[4]
            px = cam_K_np[2]
            py = cam_K_np[5]

            poses = poses.reshape(-1, 12)

            # bbox 데이터를 올바르게 처리
            if isinstance(bbox_data, np.ndarray):
                bbox_array = bbox_data.reshape(-1, 4)
            elif isinstance(bbox_data, torch.Tensor):
                bbox_array = bbox_data.numpy().reshape(-1, 4)
            else:
                bbox_array = np.array(bbox_data).reshape(-1, 4)

            # Extract translation components
            tx = poses[:, 0]  # mm 단위
            ty = poses[:, 1]  # mm 단위
            tz = poses[:, 2]  # mm 단위

            # Calculate bounding box size
            # bbox format: [x1, y1, x2, y2] (정규화 안된 좌표값)
            x1 = bbox_array[:, 0]
            y1 = bbox_array[:, 1]
            x2 = bbox_array[:, 2]
            y2 = bbox_array[:, 3]
            wbbox = x2 - x1
            hbbox = y2 - y1
            cxbbox = (x1 + x2) / 2
            cybbox = (y1 + y2) / 2

            # Calculate relative translation coordinates
            rx = (px + (fx * tx / tz) - cxbbox) / wbbox
            ry = (py + (fy * ty / tz) - cybbox) / hbbox
            rz = tz / 1000  # 정규화된 깊이 m단위

            # Rotation (9D flattened 3x3)
            new_rot = poses[:, 3:12]

            new_tran = np.stack([rx, ry, rz], axis=1)
            new_poses = np.concatenate([new_tran, new_rot], axis=1)

        else:
            raise TypeError("Unsupported type for pose transformation")

        return new_poses


@register()
class NormalizePose(nn.Module):
    def __init__(self, normalize=False) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, *inputs):
        # 입력 구조 파싱
        if len(inputs) == 1:
            inputs = inputs[0]

        # 튜플 형태의 입력에서 데이터 추출
        if isinstance(inputs, tuple):
            image = inputs[0]
            target = inputs[1]
            dataset_info = inputs[2] if len(inputs) > 2 else None
        else:
            # 단일 타겟인 경우
            target = inputs
            image = None
            dataset_info = None

        # poses가 없으면 그대로 반환
        if not isinstance(target, dict) or "poses" not in target:
            return inputs

        # pose 데이터 추출
        poses = target["poses"]
        if isinstance(poses, Pose):
            poses = poses.data

        # pose 변환 수행
        converted_poses = self._normalize_poses(poses)

        # 새로운 target 생성
        new_target = target.copy()
        new_target["poses"] = Pose(converted_poses)

        # 결과 반환
        if image is not None:
            if dataset_info is not None:
                return (image, new_target, dataset_info)
            else:
                return (image, new_target)
        else:
            return new_target

    def _normalize_poses(self, poses):
        """실제 pose 변환 로직"""
        if isinstance(poses, torch.Tensor):
            poses = poses.reshape(-1, 12)

            # Extract translation components
            tx = poses[:, 0]  # mm 단위
            ty = poses[:, 1]  # mm 단위
            tz = poses[:, 2]  # mm 단위

            tx_m = tx / 1000
            ty_m = ty / 1000
            tz_m = tz / 1000

            # Process rotation (unchanged - still r6d)
            R = poses[:, 3:12].reshape(-1, 3, 3)
            r1 = R[:, :, 0]  # column 1
            r2 = R[:, :, 1]  # column 2
            new_rot = torch.cat([r1, r2], dim=1)

            new_tran = torch.stack([tx_m, ty_m, tz_m], dim=1)
            new_poses = torch.cat([new_tran, new_rot], dim=1)

        elif isinstance(poses, np.ndarray):
            poses = poses.reshape(-1, 12)

            # Extract translation components
            tx = poses[:, 0]  # mm 단위
            ty = poses[:, 1]  # mm 단위
            tz = poses[:, 2]  # mm 단위

            tx_m = tx / 1000
            ty_m = ty / 1000
            tz_m = tz / 1000

            # Calculate relative translation coordinates
            # Process rotation (unchanged - still r6d)
            R = poses[:, 3:12].reshape(-1, 3, 3)
            r1 = R[:, :, 0]  # column 1
            r2 = R[:, :, 1]  # column 2
            new_rot = np.concatenate([r1, r2], axis=1)

            new_tran = np.stack([tx_m, ty_m, tz_m], axis=1)
            new_poses = np.concatenate([new_tran, new_rot], axis=1)

        else:
            raise TypeError("Unsupported type for pose transformation")

        return new_poses


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PILImage.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()
        if self.scale:
            inpt = inpt / 255.0
        inpt = Image(inpt)
        return inpt


@register()
class ColorJitter(T.ColorJitter):
    """GDRNPP 스타일 Color/Saturation 조정 (EnhanceColor 대응)

    GDRNPP: Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.)))
    factor 0 = 흑백, 1 = 원본, >1 = 채도 증가
    """

    def __init__(self, saturation=(0.5, 3.0), hue=0):
        # GDRNPP: pillike.EnhanceColor(factor=(0., 20.))
        # factor 0 = 흑백, 1 = 원본, >1 = 채도 증가 (hue=0: GDRNPP에는 hue 조정 없음)
        super().__init__(brightness=0, contrast=0, saturation=saturation, hue=hue)
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("color_jitter")

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class RandomBrightness(T.Transform):
    """GDRNPP 스타일 Brightness 조정

    GDRNPP: Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.)))
    factor 0 = 검정, 1 = 원본, >1 = 밝게
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(self, factor_range: Tuple[float, float] = (0.4, 2.5)) -> None:
        super().__init__()
        self.factor_range = factor_range
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("brightness")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        factor = float(
            torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
        )
        return {"apply": True, "factor": factor}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Brightness(inpt)
            return enhancer.enhance(params["factor"])
        return inpt


@register()
class RandomContrast(T.Transform):
    """GDRNPP 스타일 Contrast 조정

    GDRNPP: Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.)))
    factor 0 = 회색, 1 = 원본, >1 = 대비 증가
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(self, factor_range: Tuple[float, float] = (0.5, 3.0)) -> None:
        super().__init__()
        self.factor_range = factor_range
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("contrast")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        factor = float(
            torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
        )
        return {"apply": True, "factor": factor}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Contrast(inpt)
            return enhancer.enhance(params["factor"])
        return inpt


@register()
class RandomLinearContrast(T.Transform):
    """GDRNPP 스타일 Linear Contrast 조정

    GDRNPP: Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3))
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(
        self,
        factor_range: Tuple[float, float] = (0.5, 2.2),
        per_channel_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.factor_range = factor_range
        self.per_channel_prob = per_channel_prob
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("linear_contrast")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        per_channel = torch.rand(1) < self.per_channel_prob
        if per_channel:
            factors = [
                float(
                    torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
                )
                for _ in range(3)
            ]
        else:
            f = float(
                torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
            )
            factors = [f, f, f]
        return {"apply": True, "factors": factors, "per_channel": per_channel}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            img_array = np.array(inpt, dtype=np.float32)
            mean = 128.0
            for c in range(3):
                img_array[:, :, c] = (img_array[:, :, c] - mean) * params["factors"][
                    c
                ] + mean
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return PILImage.fromarray(img_array)
        return inpt


@register()
class RandomGrayscale(T.Transform):
    """GDRNPP 스타일 Grayscale 변환

    GDRNPP: Sometimes(0.5, Grayscale(alpha=(0.0, 1.0)))
    alpha 0 = 원본, 1 = 완전 흑백, 중간값 = 블렌딩
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(self, alpha_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        self.alpha_range = alpha_range
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("grayscale")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        alpha = float(torch.empty(1).uniform_(self.alpha_range[0], self.alpha_range[1]))
        return {"apply": True, "alpha": alpha}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            img_array = np.array(inpt, dtype=np.float32)
            gray = np.dot(img_array[:, :, :3], [0.2989, 0.5870, 0.1140])
            gray_3ch = np.stack([gray, gray, gray], axis=2)
            alpha = params["alpha"]
            blended = img_array * (1.0 - alpha) + gray_3ch * alpha
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            return PILImage.fromarray(blended)
        return inpt


@register()
class RandomHSVAdjust(T.ColorJitter):
    def __init__(
        self,
        saturation_range: Tuple[float, float] = (1.25, 1.45),
        value_range: Tuple[float, float] = (1.15, 1.35),
    ) -> None:
        super().__init__(
            brightness=value_range, contrast=None, saturation=saturation_range, hue=None
        )
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("hsv_adjust")

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class RandomSharpen(T.Transform):
    """GDRNPP 스타일 Sharpness 조정

    GDRNPP: Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.)))
    factor 0 = 흐릿, 1 = 원본, >1 = 선명
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(self, factor_range: Tuple[float, float] = (0.0, 20.0)) -> None:
        super().__init__()
        self.factor_range = factor_range
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("sharpen")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        factor = float(
            torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
        )
        return {"apply": True, "factor": factor}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            from PIL import ImageEnhance

            enhancer = ImageEnhance.Sharpness(inpt)
            return enhancer.enhance(params["factor"])
        return inpt


@register()
class RandomGaussianBlur(nn.Module):
    """GDRNPP 스타일 Gaussian Blur

    GDRNPP: Sometimes(0.4, GaussianBlur((0., 3.)))
    """

    def __init__(
        self,
        kernel_sizes: List[int] = [3, 5, 7],
        sigma_range: Tuple[float, float] = (0.0, 3.0),
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("gaussian_blur")

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        # 랜덤 커널 사이즈 선택
        kernel_size = self.kernel_sizes[
            torch.randint(0, len(self.kernel_sizes), (1,)).item()
        ]
        sigma = float(torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]))
        if sigma < 0.1:
            # sigma가 너무 작으면 blur 효과 없음, 원본 반환
            return inputs if len(inputs) > 1 else inputs[0]
        blur = T.GaussianBlur(
            kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma)
        )
        return blur(*inputs)


@register()
class RandomAdd(T.Transform):
    """GDRNPP 스타일 Add (밝기 shift)

    GDRNPP: Sometimes(0.5, Add((-25, 25), per_channel=0.3))
    모든 픽셀에 동일한 값을 더함
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(
        self, value_range: Tuple[int, int] = (-25, 25), per_channel_prob: float = 0.3
    ) -> None:
        super().__init__()
        self.value_range = value_range
        self.per_channel_prob = per_channel_prob
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("add_value")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        per_channel = torch.rand(1) < self.per_channel_prob
        if per_channel:
            values = [
                int(
                    torch.randint(
                        self.value_range[0], self.value_range[1] + 1, (1,)
                    ).item()
                )
                for _ in range(3)
            ]
        else:
            v = int(
                torch.randint(self.value_range[0], self.value_range[1] + 1, (1,)).item()
            )
            values = [v, v, v]
        return {"apply": True, "values": values}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            img_array = np.array(inpt, dtype=np.float32)
            for c in range(3):
                img_array[:, :, c] += params["values"][c]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return PILImage.fromarray(img_array)
        return inpt


@register()
class RandomMultiply(T.Transform):
    """GDRNPP 스타일 Multiply (대비/밝기 스케일링)

    GDRNPP: Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5))
    모든 픽셀에 동일한 값을 곱함
    """

    _transformed_types = (PILImage.Image, Image)

    def __init__(
        self,
        factor_range: Tuple[float, float] = (0.6, 1.4),
        per_channel_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.factor_range = factor_range
        self.per_channel_prob = per_channel_prob
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("multiply")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        per_channel = torch.rand(1) < self.per_channel_prob
        if per_channel:
            factors = [
                float(
                    torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
                )
                for _ in range(3)
            ]
        else:
            f = float(
                torch.empty(1).uniform_(self.factor_range[0], self.factor_range[1])
            )
            factors = [f, f, f]
        return {"apply": True, "factors": factors}

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            img_array = np.array(inpt, dtype=np.float32)
            for c in range(3):
                img_array[:, :, c] *= params["factors"][c]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            return PILImage.fromarray(img_array)
        return inpt



class _LegacyRandomGaussianBlur(nn.Module):
    """이전 버전과의 호환성을 위한 레거시 클래스"""

    def __init__(
        self,
        kernel_sizes: List[int] = [3, 5],
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        strong_blur_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range
        self.strong_blur_prob = strong_blur_prob
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("gaussian_blur")

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) < self.strong_blur_prob and len(self.kernel_sizes) > 1:
            kernel_size = self.kernel_sizes[-1]
        else:
            kernel_size = self.kernel_sizes[0]
        blur = T.GaussianBlur(
            kernel_size=(kernel_size, kernel_size), sigma=self.sigma_range
        )
        return blur(*inputs)


@register()
class RandomCoarseDropout(T.Transform):
    """imgaug의 CoarseDropout과 동일한 방식으로 구현 - 낮은 해상도에서 마스크 생성 후 업샘플링"""

    _transformed_types = (
        PILImage.Image,
        Image,
    )

    def __init__(
        self,
        p: Union[float, Tuple[float, float]] = 0.2,
        size_percent: Union[float, Tuple[float, float]] = 0.05,
        per_channel: bool = False,
        min_size: int = 3,
    ) -> None:
        super().__init__()

        # p 파라미터 처리 (드롭아웃 확률)
        if isinstance(p, (int, float)):
            self.p_range = (p, p)
        else:
            self.p_range = p

        # size_percent 처리 (낮은 해상도 크기)
        if isinstance(size_percent, (int, float)):
            self.size_percent_range = (size_percent, size_percent)
        else:
            self.size_percent_range = size_percent

        self.per_channel = per_channel
        self.min_size = min_size

        self.aug_manager = AugmentationManager()
        self.p_apply = self.aug_manager.get_probability("coarse_dropout")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p_apply:
            return {"apply_transform": False}

        # 이미지 크기 가져오기
        img = flat_inputs[0]
        if isinstance(img, PILImage.Image):
            img_width, img_height = img.size
            channels = 3  # RGB 가정
        elif isinstance(img, Image):
            channels, img_height, img_width = img.shape
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # 드롭아웃 확률 샘플링
        p_dropout = float(torch.empty(1).uniform_(self.p_range[0], self.p_range[1]))

        # 낮은 해상도 크기 샘플링 (imgaug의 핵심!)
        size_percent = float(
            torch.empty(1).uniform_(
                self.size_percent_range[0], self.size_percent_range[1]
            )
        )

        # 낮은 해상도 크기 계산
        low_height = max(self.min_size, int(img_height * size_percent))
        low_width = max(self.min_size, int(img_width * size_percent))

        return {
            "apply_transform": True,
            "img_height": img_height,
            "img_width": img_width,
            "channels": channels,
            "p_dropout": p_dropout,
            "low_height": low_height,
            "low_width": low_width,
        }

    def _apply_coarse_dropout(
        self, img: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """imgaug 스타일의 CoarseDropout 적용 - FromLowerResolution 방식"""
        if isinstance(img, Image):
            img = F.to_pil_image(img)
        if isinstance(img, PILImage.Image):
            img = np.array(img)

        img_height = params["img_height"]
        img_width = params["img_width"]
        channels = params["channels"]
        p_dropout = params["p_dropout"]
        low_height = params["low_height"]
        low_width = params["low_width"]

        img_result = img.copy().astype(np.float32)

        if self.per_channel and len(img.shape) == 3:
            # 채널별로 독립적인 마스크 생성
            for c in range(channels):
                # 1. 낮은 해상도에서 드롭아웃 마스크 생성
                low_mask = (
                    torch.rand(low_height, low_width) >= p_dropout
                )  # keep mask (1=keep, 0=drop)
                low_mask = low_mask.float().numpy()

                # 2. 마스크를 원본 크기로 업샘플링 (nearest neighbor)
                mask = cv2.resize(
                    low_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
                )

                # 3. 마스크 적용
                img_result[:, :, c] *= mask
        else:
            # 전체 이미지에 동일한 마스크 적용
            # 1. 낮은 해상도에서 드롭아웃 마스크 생성
            low_mask = torch.rand(low_height, low_width) >= p_dropout  # keep mask
            low_mask = low_mask.float().numpy()

            # 2. 마스크를 원본 크기로 업샘플링
            mask = cv2.resize(
                low_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
            )

            # 3. 마스크 적용
            if len(img.shape) == 3:
                mask = mask[:, :, np.newaxis]  # 채널 차원 추가
            img_result *= mask

        # 결과를 uint8로 변환
        img_result = np.clip(img_result, 0, 255).astype(np.uint8)
        return PILImage.fromarray(img_result)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply_transform"]:
            return inpt

        if isinstance(inpt, (PILImage.Image, Image)):
            return self._apply_coarse_dropout(inpt, params)
        return inpt


@register()
class RandomObjectOcclusion(T.Transform):
    """객체 중심적 occlusion augmentation - bbox 영역에 1-3개의 랜덤 색상 블록으로 가림"""

    _transformed_types = (
        PILImage.Image,
        Image,
    )

    def __init__(
        self,
        occlusion_ratio_range: Tuple[float, float] = (0.05, 0.1),
        num_blocks_range: Tuple[int, int] = (1, 2),
        bbox_margin_ratio: float = 0.1,
        min_bbox_size: int = 20,
        max_objects_per_image: int = 20,
        color_mode: str = "random",  # 'random', 'natural'
        augmentation_strength: float = 0.7,
    ) -> None:
        super().__init__()

        self.occlusion_ratio_range = occlusion_ratio_range
        self.num_blocks_range = num_blocks_range
        self.bbox_margin_ratio = bbox_margin_ratio
        self.min_bbox_size = min_bbox_size
        self.max_objects_per_image = max_objects_per_image
        self.color_mode = color_mode
        self.aug_manager = AugmentationManager(augmentation_strength)
        self.p = self.aug_manager.get_probability("rotate_expand")

    def _generate_random_color(self) -> Tuple[int, int, int]:
        """랜덤 색상 생성"""
        if self.color_mode == "random":
            return tuple(torch.randint(30, 225, (3,)).tolist())
        elif self.color_mode == "natural":
            # 자연스러운 색상들
            colors = [
                [139, 115, 85],  # 갈색
                [128, 128, 128],  # 회색
                [210, 180, 140],  # 베이지
                [169, 169, 169],  # 밝은 회색
                [160, 82, 45],  # 안장 갈색
                [105, 105, 105],  # 어두운 회색
            ]
            base_color = colors[torch.randint(0, len(colors), (1,)).item()]
            # 약간의 변동성 추가
            variation = torch.randint(-15, 16, (3,))
            final_color = [
                max(30, min(225, base_color[i] + variation[i].item())) for i in range(3)
            ]
            return tuple(final_color)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """변환 파라미터 생성"""
        # flat_inputs에서 이미지 추출 (첫 번째 요소가 이미지)
        img = flat_inputs[0]

        if isinstance(img, PILImage.Image):
            img_w, img_h = img.size
        elif isinstance(img, Image):
            _, img_h, img_w = img.shape
        else:
            return {"apply_transform": False}

        return {
            "apply_transform": True,
            "img_size": (img_w, img_h),
            "occlusion_blocks": [],  # 실제 블록들은 forward에서 생성
        }

    def _expand_bbox(
        self, bbox: torch.Tensor, img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """bbox에 margin 추가"""
        img_w, img_h = img_size
        x1, y1, x2, y2 = bbox

        w, h = x2 - x1, y2 - y1
        margin_w = w * self.bbox_margin_ratio
        margin_h = h * self.bbox_margin_ratio

        new_x1 = max(0, x1 - margin_w)
        new_y1 = max(0, y1 - margin_h)
        new_x2 = min(img_w, x2 + margin_w)
        new_y2 = min(img_h, y2 + margin_h)

        return torch.tensor([new_x1, new_y1, new_x2, new_y2])

    def _create_blocks_for_bbox(
        self, bbox: torch.Tensor, img_size: Tuple[int, int]
    ) -> List[Dict]:
        """하나의 bbox에 대해 occlusion 블록들 생성"""
        expanded_bbox = self._expand_bbox(bbox, img_size)

        bbox_w = expanded_bbox[2] - expanded_bbox[0]
        bbox_h = expanded_bbox[3] - expanded_bbox[1]
        bbox_area = bbox_w * bbox_h

        # 가림 비율과 블록 수 결정
        occlusion_ratio = (
            torch.empty(1)
            .uniform_(self.occlusion_ratio_range[0], self.occlusion_ratio_range[1])
            .item()
        )

        num_blocks = torch.randint(
            self.num_blocks_range[0], self.num_blocks_range[1] + 1, (1,)
        ).item()

        total_occlusion_area = bbox_area * occlusion_ratio
        area_per_block = total_occlusion_area / num_blocks

        blocks = []
        for _ in range(num_blocks):
            # 블록 크기 결정
            block_size = int(np.sqrt(area_per_block))
            aspect_ratio = torch.empty(1).uniform_(0.7, 1.4).item()

            if aspect_ratio >= 1.0:
                block_w = int(block_size * aspect_ratio)
                block_h = block_size
            else:
                block_w = block_size
                block_h = int(block_size / aspect_ratio)

            # bbox 영역을 벗어나지 않도록 조정
            block_w = min(int(block_w), int(bbox_w) - 2)
            block_h = min(int(block_h), int(bbox_h) - 2)
            block_w = max(block_w, 10)
            block_h = max(block_h, 10)

            # 블록 위치 결정
            max_x = int(bbox_w - block_w)
            max_y = int(bbox_h - block_h)

            if max_x <= 0 or max_y <= 0:
                continue

            rel_x = torch.randint(0, max_x, (1,)).item()
            rel_y = torch.randint(0, max_y, (1,)).item()

            abs_x = int(expanded_bbox[0]) + rel_x
            abs_y = int(expanded_bbox[1]) + rel_y

            blocks.append(
                {
                    "x1": abs_x,
                    "y1": abs_y,
                    "x2": abs_x + block_w,
                    "y2": abs_y + block_h,
                    "color": self._generate_random_color(),
                }
            )

        return blocks

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """이미지에 occlusion 블록들 적용"""
        if not params["apply_transform"] or len(params["occlusion_blocks"]) == 0:
            return inpt

        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        if isinstance(inpt, PILImage.Image):
            img_array = np.array(inpt)
        else:
            return inpt

        img_result = img_array.copy()

        # 각 블록 그리기
        for block in params["occlusion_blocks"]:
            x1, y1, x2, y2 = block["x1"], block["y1"], block["x2"], block["y2"]
            color = block["color"]

            # 좌표 범위 확인
            img_h, img_w = img_result.shape[:2]
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            # 블록 그리기
            if len(img_result.shape) == 3:
                img_result[y1:y2, x1:x2] = color
            else:
                gray_val = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                img_result[y1:y2, x1:x2] = gray_val

        return PILImage.fromarray(img_result.astype(np.uint8))

    def forward(self, *inputs):
        """메인 forward 함수"""
        # 입력 파싱
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            input_tuple = inputs[0]
            image = input_tuple[0]
            target = input_tuple[1] if len(input_tuple) > 1 else {}
        else:
            return inputs if len(inputs) > 1 else inputs[0]

        # bbox 정보 확인
        if not isinstance(target, dict) or "boxes" not in target:
            return inputs if len(inputs) > 1 else inputs[0]

        # bbox 데이터 추출
        boxes = target["boxes"]
        if hasattr(boxes, "data"):
            bbox_data = boxes.data
        else:
            bbox_data = boxes

        if len(bbox_data) == 0:
            return inputs if len(inputs) > 1 else inputs[0]

        # 이미지 크기
        if isinstance(image, PILImage.Image):
            img_w, img_h = image.size
        elif isinstance(image, Image):
            _, img_h, img_w = image.shape
        else:
            return inputs if len(inputs) > 1 else inputs[0]

        # 각 물체마다 독립적으로 확률 적용
        selected_bboxes = []
        for i, bbox in enumerate(bbox_data):
            # bbox 크기 체크
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            # 너무 작은 bbox는 스킵
            if bbox_w < self.min_bbox_size or bbox_h < self.min_bbox_size:
                continue

            # 각 물체마다 독립적으로 확률 판정
            if torch.rand(1) < self.p:
                selected_bboxes.append(bbox)

        # 최대 개수 제한
        if len(selected_bboxes) > self.max_objects_per_image:
            indices = torch.randperm(len(selected_bboxes))[: self.max_objects_per_image]
            selected_bboxes = [selected_bboxes[i] for i in indices]

        if len(selected_bboxes) == 0:
            return inputs if len(inputs) > 1 else inputs[0]

        # 선택된 bbox들에 대해 블록 생성
        all_blocks = []
        for bbox in selected_bboxes:
            blocks = self._create_blocks_for_bbox(bbox, (img_w, img_h))
            all_blocks.extend(blocks)

        if len(all_blocks) == 0:
            return inputs if len(inputs) > 1 else inputs[0]

        # 파라미터 업데이트
        params = self.make_params([image])
        params["occlusion_blocks"] = all_blocks

        # 이미지 변환
        transformed_image = self.transform(image, params)

        # 결과 반환
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            input_tuple = inputs[0]
            return (transformed_image,) + input_tuple[1:]
        else:
            return (
                transformed_image
                if len(inputs) == 1
                else (transformed_image,) + inputs[1:]
            )


@register()
class RandomBackgroundWithPresets(nn.Module):
    def __init__(
        self,
        background_dir: Optional[str] = None,
        preset_categories: Optional[List[str]] = None,
        random_color_prob: float = 0.0,  # 단색 배경 비활성화
        blur_edge: bool = True,
        random_noise: bool = True,
        p: float = None,
    ) -> None:
        super().__init__()
        self.background_dir = (
            os.path.expanduser(background_dir) if background_dir else None
        )
        self.random_color_prob = random_color_prob
        self.blur_edge = blur_edge
        self.random_noise = random_noise
        self.aug_manager = AugmentationManager()
        if p is not None:
            self.p = p
        else:
            self.p = self.aug_manager.get_probability("random_background")

        # 프리셋 카테고리 설정
        self.preset_categories = preset_categories or ["STUDIO", "NATURAL", "INDOOR"]
        self.bg_colors = BackgroundColors()
        self.available_colors = self._gather_colors()

        # 배경 이미지 로드
        if background_dir:
            self.background_images = self._load_background_images()
        else:
            self.background_images = None

        self._cached_image = None
        self._cached_params = None

    def _gather_colors(self) -> Dict[str, List[int]]:
        """선택된 카테고리의 모든 색상 수집"""
        colors = {}
        for category in self.preset_categories:
            if hasattr(self.bg_colors, category):
                colors.update(getattr(self.bg_colors, category))
        return colors

    def _load_background_images(self) -> List[str]:
        """배경 이미지 디렉토리에서 이미지 파일 리스트 로드"""
        valid_extensions = [".jpg", ".jpeg", ".png"]
        background_files = []

        if os.path.exists(self.background_dir):
            for file in os.listdir(self.background_dir):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    background_files.append(os.path.join(self.background_dir, file))

        return background_files

    def make_params(self, image) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply_transform": False}

        # 이미지 크기 계산
        if isinstance(image, PILImage.Image):
            w, h = image.size
            shape = (h, w, 3)
        elif isinstance(image, Image):
            c, h, w = image.shape
            shape = (h, w, 3)

        # 배경 생성 방식 결정
        use_color = (
            torch.rand(1) < self.random_color_prob
        ) or not self.background_images

        params = {
            "apply_transform": True,
            "shape": shape,
            "use_color": use_color,
        }

        if use_color:
            # 색상 선택 및 변동성 추가
            color_name = torch.randint(0, len(self.available_colors), (1,)).item()
            color_value = list(self.available_colors.values())[color_name]

            # 순수 흰색/검정이 아닌 경우 변동성 추가
            if color_value not in ([255, 255, 255], [0, 0, 0]):
                variation = torch.randint(-10, 11, (3,))
                color_value = np.clip(np.array(color_value) + variation.numpy(), 0, 255)

            params["color_value"] = color_value
        else:
            # 배경 이미지 선택
            bg_idx = torch.randint(0, len(self.background_images), (1,)).item()
            params["background_path"] = self.background_images[bg_idx]

        return params

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """마스크 전처리 및 경계선 블러 처리"""
        if len(mask.shape) == 3:
            mask = mask.max(dim=0)[0]

        mask = (mask > 0.5).float()

        if self.blur_edge:
            # numpy로 변환하여 가우시안 블러 적용
            mask_np = mask.numpy()
            mask_np = cv2.GaussianBlur(mask_np, (7, 7), 1.5)
            mask = torch.from_numpy(np.clip(mask_np, 0, 1))

        return mask.unsqueeze(0).repeat(3, 1, 1)

    def _add_noise(self, image: torch.Tensor) -> torch.Tensor:
        """이미지에 노이즈 추가"""
        noise = torch.randn_like(image) * 10
        noisy_img = image + noise
        return torch.clamp(noisy_img, 0, 255)

    def _transform_single(self, image, mask, params: Dict[str, Any]) -> torch.Tensor:
        if not params["apply_transform"]:
            return image

        # 이미지를 텐서로 변환
        if isinstance(image, PILImage.Image):
            image_tensor = F.pil_to_tensor(image).float()
        else:
            image_tensor = image.float()

        # 마스크 준비
        mask_tensor = self._prepare_mask(mask)

        # 배경 생성
        if params["use_color"]:
            background = (
                torch.tensor(params["color_value"], dtype=torch.float32)
                .view(3, 1, 1)
                .repeat(1, *params["shape"][:2])
            )
        else:
            bg_img = cv2.imread(params["background_path"])
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            bg_img = cv2.resize(bg_img, (params["shape"][1], params["shape"][0]))
            background = torch.from_numpy(bg_img).permute(2, 0, 1).float()

        if self.random_noise:
            background = self._add_noise(background)

        # 이미지 합성
        result = image_tensor * mask_tensor + background * (1 - mask_tensor)
        result = result.clamp(0, 255).byte()

        return F.to_pil_image(result)

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        image = inputs[0]
        target = inputs[1]

        # Get transformation parameters
        params = self.make_params(image)

        if not params["apply_transform"]:
            return inputs

        # Transform image with masks
        if "masks" in target:
            transformed_image = self._transform_single(image, target["masks"], params)
            transformed_inputs = list(inputs)
            transformed_inputs[0] = transformed_image
            return tuple(transformed_inputs)

        return inputs


@register()
class RandomMotionBlur(T.Transform):
    _transformed_types = (
        PILImage.Image,
        Image,
    )

    def __init__(
        self, degree: Tuple[int, int] = (0, 360), length: Tuple[int, int] = (1, 10)
    ) -> None:
        super().__init__()
        self.degree_range = self._check_range(degree, "degree")
        self.length_range = self._check_range(length, "length")
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("motion_blur")

    def _check_range(self, value: Tuple[int, int], name: str) -> Tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} should be a tuple of length 2")
        if value[0] > value[1]:
            raise ValueError(f"{name} values should be in ascending order")
        return (int(value[0]), int(value[1]))

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply_transform": False}
        angle = float(
            torch.empty(1).uniform_(self.degree_range[0], self.degree_range[1])
        )
        length = int(
            torch.empty(1).uniform_(self.length_range[0], self.length_range[1])
        )
        kernel = self._create_motion_kernel(angle, length)
        return {
            "apply_transform": True,
            "kernel": kernel,
            "angle": angle,
            "length": length,
        }

    def _create_motion_kernel(self, angle: float, length: int) -> np.ndarray:
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        kernel_size = int(max(abs(dx), abs(dy)) * length * 2)
        if kernel_size < 3:
            kernel_size = 3
        kernel = np.zeros((kernel_size, kernel_size))
        center_x = kernel_size // 2
        center_y = kernel_size // 2
        end_x = int(center_x + dx * length)
        end_y = int(center_y + dy * length)
        cv2.line(kernel, (center_x, center_y), (end_x, end_y), 1.0, thickness=1)
        kernel_sum = kernel.sum()
        if kernel_sum == 0:
            kernel[center_x, center_y] = 1.0
        else:
            kernel = kernel / kernel_sum
        return kernel

    def _apply_motion_blur(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        if isinstance(img, Image):
            img = F.to_pil_image(img)
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        img_blurred = cv2.filter2D(img, -1, kernel)
        result = np.clip(img_blurred, 0, 255).astype(np.uint8)
        return PILImage.fromarray(result)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply_transform"]:
            return inpt
        if isinstance(inpt, (PILImage.Image, Image)):
            return self._apply_motion_blur(inpt, params["kernel"])
        return inpt


@register()
class RandomGaussianNoise(T.Transform):
    _transformed_types = (
        PILImage.Image,
        Image,
    )

    def __init__(
        self,
        scale: float = 10.0,
        per_channel: bool = True,
    ) -> None:
        """GDRNPP: Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True))

        Args:
            scale: 노이즈 표준편차 (GDRNPP 기본값: 10)
            per_channel: 채널별 독립 노이즈 (GDRNPP 기본값: True)
        """
        super().__init__()
        self.scale = scale
        self.per_channel = per_channel
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("gaussian_noise")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        return {"apply": True, "scale": self.scale}

    def _apply_gaussian_noise(self, img: np.ndarray, scale: float) -> np.ndarray:
        if isinstance(img, Image):
            img = F.to_pil_image(img)
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        noise = np.random.normal(0, scale, img.shape)
        noisy_img = img.astype(np.float32) + noise
        result = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return PILImage.fromarray(result)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params.get("apply", False):
            return inpt
        if isinstance(inpt, (PILImage.Image, Image)):
            return self._apply_gaussian_noise(inpt, params["scale"])
        return inpt


@register()
class RandomAdditionalNoise(T.Transform):
    _transformed_types = (
        PILImage.Image,
        Image,
    )

    def __init__(
        self, mean: float = 0.0, scale_range: Tuple[float, float] = (7.0, 10.0)
    ) -> None:
        super().__init__()
        self.mean = mean
        self.scale_range = self._check_range(scale_range, "scale_range")
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("additional_noise")

    def _check_range(
        self, value: Tuple[float, float], name: str
    ) -> Tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} should be a tuple of length 2")
        if not 0 <= value[0] <= value[1]:
            raise ValueError(
                f"{name} values should be non-negative and in ascending order"
            )
        return (float(value[0]), float(value[1]))

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply_transform": False}
        scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]))
        return {"apply_transform": True, "scale": scale}

    def _apply_additional_noise(self, img: np.ndarray, scale: float) -> np.ndarray:
        if isinstance(img, Image):
            img = F.to_pil_image(img)
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        noise = np.random.normal(self.mean, scale, img.shape)
        noisy_img = img + noise
        result = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return PILImage.fromarray(result)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply_transform"]:
            return inpt
        if isinstance(inpt, (PILImage.Image, Image)):
            return self._apply_additional_noise(np.array(inpt), params["scale"])
        return inpt


@register()
class RandomSelect(nn.Module):
    """두 가지 변환 중 하나를 확률적으로 선택"""

    def __init__(
        self, transform1: Dict[str, Any], transform2: Dict[str, Any], p: float = 0.5
    ) -> None:
        super().__init__()
        # transform 문자열을 실제 transform 객체로 변환
        if isinstance(transform1, dict):
            name1 = transform1.pop("type")
            self.transform1 = getattr(
                GLOBAL_CONFIG[name1]["_pymodule"], GLOBAL_CONFIG[name1]["_name"]
            )(**transform1)
            transform1["type"] = name1

        if isinstance(transform2, dict):
            name2 = transform2.pop("type")
            self.transform2 = getattr(
                GLOBAL_CONFIG[name2]["_pymodule"], GLOBAL_CONFIG[name2]["_name"]
            )(**transform2)
            transform2["type"] = name2

        self.p = p

    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1) < self.p:
            return self.transform1(*inputs)
        return self.transform2(*inputs)


@register()
class ZoomPoseAugmentation(nn.Module):
    """
    cam_K 고정 상태에서 이미지 zoom + pose tz 조정

    Zoom은 principal point (cx, cy)를 중심으로 수행됨.

    수학적 원리:
        투영: u = fx * tx/tz + cx
        Zoom 후: u' = cx + (u - cx) * zoom_factor

        이를 만족시키려면:
        - tz_new = tz / zoom_factor
        - tx, ty는 그대로 유지

    - zoom_factor > 1: 확대 (물체가 가까워 보임) → tz 감소
    - zoom_factor < 1: 축소 (물체가 멀어 보임) → tz 증가

    ConvertPose 이전에 적용해야 함 (원시 mm 단위 pose 사용)
    """

    def __init__(
        self,
        zoom_factor: float = 1.0,
        zoom_range: Optional[Tuple[float, float]] = None,
        p: float = 1.0,
        fill_value: Union[int, Tuple[int, int, int]] = 114,
        background_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            zoom_factor: 고정 zoom factor (zoom_range가 None일 때 사용)
            zoom_range: 랜덤 zoom factor 범위 (min, max)
            p: 적용 확률
            fill_value: zoom out 시 padding 색상 (background_dir=None일 때만 사용)
            background_dir: 주어지면 zoom-out 여백을 해당 디렉터리의 random
                natural image로 채움.
        """
        super().__init__()
        self.zoom_factor = zoom_factor
        self.zoom_range = zoom_range
        self.p = p
        self.fill_value = (
            fill_value
            if isinstance(fill_value, tuple)
            else (fill_value, fill_value, fill_value)
        )
        self._background_dir = (
            os.path.expanduser(background_dir) if background_dir else None
        )
        self._background_images: Optional[List[str]] = None


    def _load_backgrounds(self) -> List[str]:
        from pathlib import Path
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if not self._background_dir:
            return []
        return [
            str(p) for p in Path(self._background_dir).iterdir()
            if p.suffix.lower() in exts
        ]

    def _sample_background(self, h: int, w: int) -> Optional[np.ndarray]:
        if self._background_dir is None:
            return None
        if self._background_images is None:
            self._background_images = self._load_backgrounds()
        if not self._background_images:
            return None
        bg_path = self._background_images[np.random.randint(len(self._background_images))]
        bg_img = PILImage.open(bg_path).convert("RGB")
        bg_np = np.array(bg_img.resize((w, h)))
        return bg_np
    def _get_zoom_factor(self) -> float:
        """zoom factor 결정"""
        if self.zoom_range is not None:
            return float(
                torch.empty(1).uniform_(self.zoom_range[0], self.zoom_range[1])
            )
        return self.zoom_factor

    def _zoom_in_image(
        self, img: np.ndarray, zoom_factor: float, cx: float, cy: float
    ) -> np.ndarray:
        h, w = img.shape[:2]

        crop_w = w / zoom_factor
        crop_h = h / zoom_factor

        crop_x1 = cx * (1 - 1 / zoom_factor)
        crop_y1 = cy * (1 - 1 / zoom_factor)
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h

        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - w)
        pad_bottom = max(0, crop_y2 - h)

        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(w, crop_x2)
        crop_y2 = min(h, crop_y2)

        cropped = img[int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                int(pad_top),
                int(pad_bottom),
                int(pad_left),
                int(pad_right),
                cv2.BORDER_CONSTANT,
                value=self.fill_value,
            )
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def _zoom_out_image(
        self, img: np.ndarray, zoom_factor: float, cx: float, cy: float
    ) -> np.ndarray:
        """
        Zoom out: 이미지 축소 후 principal point가 중심에 오도록 padding
        """
        h, w = img.shape[:2]

        # 축소된 크기
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        # resize (축소)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # principal point가 원래 위치에 오도록 padding 계산
        # 축소 후 cx_new = cx * zoom_factor
        # padding 후 cx가 원래 위치에 있으려면:
        # pad_left + cx * zoom_factor = cx
        # pad_left = cx * (1 - zoom_factor)
        pad_left = int(cx * (1 - zoom_factor))
        pad_top = int(cy * (1 - zoom_factor))
        pad_right = w - new_w - pad_left
        pad_bottom = h - new_h - pad_top

        # 음수 padding 처리 (이미지가 canvas보다 클 때)
        if pad_left < 0:
            resized = resized[:, -pad_left:]
            pad_left = 0
        if pad_top < 0:
            resized = resized[-pad_top:, :]
            pad_top = 0
        if pad_right < 0:
            resized = resized[:, :pad_right]
            pad_right = 0
        if pad_bottom < 0:
            resized = resized[:pad_bottom, :]
            pad_bottom = 0

        # Fill strategy: background image (RGB only) or constant fill
        is_rgb = img.ndim == 3 and img.shape[2] == 3
        bg = self._sample_background(h, w) if is_rgb else None
        if bg is not None:
            rh, rw = resized.shape[:2]
            rh = min(rh, h - pad_top)
            rw = min(rw, w - pad_left)
            canvas = bg.copy()
            if rh > 0 and rw > 0:
                canvas[pad_top:pad_top + rh, pad_left:pad_left + rw] = resized[:rh, :rw]
            zoomed = canvas
        else:
            fill_for_pad = 0 if not is_rgb else self.fill_value
            zoomed = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=fill_for_pad,
            )

        # 크기 보정 (rounding으로 인한 1-2픽셀 차이)
        if zoomed.shape[0] != h or zoomed.shape[1] != w:
            zoomed = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed

    def _zoom_image(
        self, image: Any, zoom_factor: float, cx: float, cy: float
    ) -> PILImage.Image:
        """이미지 zoom 적용"""
        if isinstance(image, PILImage.Image):
            img = np.array(image)
        elif isinstance(image, Image):
            img = image.permute(1, 2, 0).numpy()
        else:
            img = np.array(image)

        if zoom_factor > 1:
            zoomed = self._zoom_in_image(img, zoom_factor, cx, cy)
        elif zoom_factor < 1:
            zoomed = self._zoom_out_image(img, zoom_factor, cx, cy)
        else:
            zoomed = img

        return PILImage.fromarray(zoomed.astype(np.uint8))

    def _zoom_poses(self, poses: Any, zoom_factor: float) -> Any:
        """
        Pose translation 조정 (tz만 변경)

        수학적 유도:
        u = fx * tx/tz + cx
        zoom 후: u' = cx + (u - cx) * zoom_factor
                   = cx + (fx * tx/tz) * zoom_factor
                   = fx * (tx * zoom_factor / tz) + cx
                   = fx * tx / (tz / zoom_factor) + cx

        따라서: tz_new = tz / zoom_factor, tx와 ty는 그대로
        """
        if isinstance(poses, torch.Tensor):
            poses = poses.clone()
            poses_reshaped = poses.reshape(-1, 12)
            poses_reshaped[:, 2] = poses_reshaped[:, 2] / zoom_factor  # tz만 조정
            return poses_reshaped.reshape(poses.shape)
        elif isinstance(poses, np.ndarray):
            poses = poses.copy()
            poses_reshaped = poses.reshape(-1, 12)
            poses_reshaped[:, 2] = poses_reshaped[:, 2] / zoom_factor  # tz만 조정
            return poses_reshaped.reshape(poses.shape)
        else:
            return poses

    def _zoom_boxes(
        self,
        boxes: Any,
        zoom_factor: float,
        cx: float,
        cy: float,
        img_w: int,
        img_h: int,
    ) -> Tuple[Any, torch.Tensor]:
        """
        BBox 좌표 변환 (principal point 기준)

        x_new = cx + (x - cx) * zoom_factor
        y_new = cy + (y - cy) * zoom_factor
        """
        if hasattr(boxes, "data"):
            boxes_data = boxes.data.clone()
            box_format = boxes.format
        else:
            boxes_data = (
                boxes.clone()
                if isinstance(boxes, torch.Tensor)
                else torch.tensor(boxes)
            )
            box_format = BoundingBoxFormat.XYXY

        # 좌표 변환
        boxes_data[:, 0] = cx + (boxes_data[:, 0] - cx) * zoom_factor  # x1
        boxes_data[:, 1] = cy + (boxes_data[:, 1] - cy) * zoom_factor  # y1
        boxes_data[:, 2] = cx + (boxes_data[:, 2] - cx) * zoom_factor  # x2
        boxes_data[:, 3] = cy + (boxes_data[:, 3] - cy) * zoom_factor  # y2

        # 이미지 경계로 clip
        boxes_data[:, 0] = boxes_data[:, 0].clamp(0, img_w)
        boxes_data[:, 1] = boxes_data[:, 1].clamp(0, img_h)
        boxes_data[:, 2] = boxes_data[:, 2].clamp(0, img_w)
        boxes_data[:, 3] = boxes_data[:, 3].clamp(0, img_h)

        new_boxes = BoundingBoxes(
            boxes_data, format=box_format, canvas_size=(img_h, img_w)
        )
        return new_boxes

    def _zoom_masks(
        self,
        masks: Any,
        zoom_factor: float,
        cx: float,
        cy: float,
        img_w: int,
        img_h: int,
    ) -> Any:
        """Mask zoom (이미지와 동일한 방식)"""
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)

        zoomed_masks = []
        for mask in masks_np:
            mask_uint8 = (
                (mask * 255).astype(np.uint8)
                if mask.max() <= 1
                else mask.astype(np.uint8)
            )

            if zoom_factor > 1:
                zoomed = self._zoom_in_image(mask_uint8, zoom_factor, cx, cy)
            elif zoom_factor < 1:
                zoomed = self._zoom_out_image(mask_uint8, zoom_factor, cx, cy)
            else:
                zoomed = mask_uint8

            # 이진화 (threshold)
            zoomed = (zoomed > 127).astype(np.uint8)
            zoomed_masks.append(zoomed)

        return torch.from_numpy(np.array(zoomed_masks, dtype=np.uint8))

    def forward(self, *inputs):
        """
        메인 forward 함수

        입력: (image, target) 또는 (image, target, dataset_info)
        """
        # 확률 체크
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # 입력 파싱
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            input_tuple = inputs[0]
        else:
            input_tuple = inputs

        image = input_tuple[0]
        target = input_tuple[1] if len(input_tuple) > 1 else {}
        dataset_info = input_tuple[2] if len(input_tuple) > 2 else None

        # 이미지 크기 확인
        if isinstance(image, PILImage.Image):
            img_w, img_h = image.size
        elif isinstance(image, Image):
            _, img_h, img_w = image.shape
        else:
            return inputs if len(inputs) > 1 else inputs[0]

        # cam_K에서 principal point 가져오기
        if "cam_K" in target:
            cam_K = target["cam_K"]
            if isinstance(cam_K, torch.Tensor):
                cam_K = cam_K.cpu().numpy()
            if len(cam_K.shape) > 1:
                cam_K = cam_K[0]  # 첫 번째 인스턴스의 cam_K 사용
            cx, cy = float(cam_K[2]), float(cam_K[5])
        else:
            # 기본값: 이미지 중심
            cx, cy = img_w / 2, img_h / 2

        # zoom factor 결정
        zoom_factor = self._get_zoom_factor()

        if zoom_factor == 1.0:
            return inputs if len(inputs) > 1 else inputs[0]

        # 이미지 변환
        transformed_image = self._zoom_image(image, zoom_factor, cx, cy)

        # target 변환 (필터링/visibility 재계산은 후속 FilterSmallBoxLowVis가 담당)
        transformed_target = target.copy()

        if "poses" in target:
            poses = target["poses"]
            if isinstance(poses, Pose):
                transformed_target["poses"] = Pose(
                    self._zoom_poses(poses.data, zoom_factor)
                )
            else:
                transformed_target["poses"] = self._zoom_poses(poses, zoom_factor)

        if "boxes" in target:
            transformed_target["boxes"] = self._zoom_boxes(
                target["boxes"], zoom_factor, cx, cy, img_w, img_h
            )

        if "masks" in target:
            transformed_target["masks"] = self._zoom_masks(
                target["masks"], zoom_factor, cx, cy, img_w, img_h
            )

        if "full_masks" in target:
            transformed_target["full_masks"] = self._zoom_masks(
                target["full_masks"], zoom_factor, cx, cy, img_w, img_h
            )

        # 결과 반환
        if dataset_info is not None:
            return (transformed_image, transformed_target, dataset_info)
        elif len(input_tuple) > 1:
            return (transformed_image, transformed_target)
        else:
            return transformed_image


# ============================================================================
# Shared helpers for depth-ordered compositing (CopyPaste, RandomTransformAug)
# ============================================================================

def _gaussian_blur_edge(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """마스크 엣지에 Gaussian blur 적용 (블렌딩용)"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)


def _extract_objects(image, target):
    """이미지에서 물체들을 visib_mask 기반으로 추출.

    Returns:
        list of dict: 각 물체 정보 (pixels, visib_mask, full_mask, tz, pose, label, box, cam_K)
    """
    objects = []
    masks = target.get("masks")
    poses = target.get("poses")
    labels = target.get("labels")
    cam_K = target.get("cam_K")
    full_masks = target.get("full_masks")
    boxes = target.get("boxes")

    if masks is None or poses is None or labels is None:
        return objects

    img_np = np.array(image)
    masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)
    poses_data = poses.data if isinstance(poses, Pose) else poses
    box_data = boxes.data if hasattr(boxes, "data") else boxes if boxes is not None else None

    for i in range(len(masks_np)):
        visib_mask = masks_np[i]
        if visib_mask.sum() == 0:
            continue

        pose = poses_data[i]
        tz = float(pose[2].item() if isinstance(pose[2], torch.Tensor) else pose[2])

        pixels = img_np.copy()
        pixels[visib_mask == 0] = 0

        obj = {
            "pixels": pixels,
            "visib_mask": visib_mask.astype(np.uint8),
            "tz": tz,
            "pose": pose.clone() if isinstance(pose, torch.Tensor) else torch.tensor(pose),
            "label": labels[i:i+1].clone(),
            "box": box_data[i:i+1].clone() if box_data is not None else None,
        }

        if cam_K is not None:
            obj["cam_K"] = cam_K[i:i+1].clone() if len(cam_K.shape) > 1 else cam_K.unsqueeze(0).clone()

        if full_masks is not None:
            fm = full_masks[i]
            fm_np = fm.numpy() if torch.is_tensor(fm) else np.array(fm)
            obj["full_mask"] = fm_np.astype(np.uint8)
        else:
            obj["full_mask"] = visib_mask.astype(np.uint8)

        objects.append(obj)

    return objects


def _depth_composite(objects, background, H, W, edge_blur_kernel=5):
    """tz 기준 depth ordering으로 캔버스에 합성 (objects는 tz 내림차순 정렬 완료).

    Returns:
        canvas: H x W x 3 합성 이미지
        occupancy: H x W int array (각 픽셀을 소유하는 물체 index, -1=배경)
    """
    canvas = background.copy()
    occupancy = np.full((H, W), -1, dtype=np.int32)

    for i, obj in enumerate(objects):
        mask = obj["visib_mask"]
        pixels = obj["pixels"]

        if edge_blur_kernel > 0:
            alpha = _gaussian_blur_edge(mask, edge_blur_kernel)
            alpha = np.clip(alpha, 0, 1)
            alpha_3ch = alpha[:, :, np.newaxis]
            canvas = (canvas * (1 - alpha_3ch) + pixels * alpha_3ch).astype(np.uint8)
            occupancy[mask > 0] = i
        else:
            canvas[mask > 0] = pixels[mask > 0]
            occupancy[mask > 0] = i

    return canvas, occupancy


def _visibility_filter(objects, occupancy, min_visibility=0.1, min_modal_size=5):
    """최종 visibility 계산 및 필터링.

    Returns:
        keep_indices: 유지할 물체의 인덱스 리스트
        final_masks: 각 물체의 최종 visible mask
    """
    keep_indices = []
    final_masks = []

    for i, obj in enumerate(objects):
        final_mask = (occupancy == i).astype(np.uint8)
        full_area = obj["full_mask"].sum()
        if full_area == 0:
            continue
        visibility = final_mask.sum() / full_area
        if visibility < min_visibility:
            continue

        # modal bbox 최소 크기 체크
        vys, vxs = np.where(final_mask > 0)
        if len(vys) == 0:
            continue
        if (vxs.max() - vxs.min()) < min_modal_size or (vys.max() - vys.min()) < min_modal_size:
            continue

        keep_indices.append(i)
        final_masks.append(final_mask)

    return keep_indices, final_masks


def _build_target_from_objects(objects, keep_indices, final_masks, H, W):
    """필터링된 물체들로 target dict 생성."""
    kept = [objects[i] for i in keep_indices]
    if not kept:
        return None

    target = {}
    target["masks"] = Mask(torch.from_numpy(np.stack(final_masks, axis=0)))
    target["full_masks"] = Mask(torch.from_numpy(
        np.stack([objects[i]["full_mask"] for i in keep_indices], axis=0)
    ))
    target["poses"] = Pose(torch.stack([o["pose"] for o in kept], dim=0))
    target["labels"] = torch.cat([o["label"] for o in kept], dim=0)

    if "cam_K" in kept[0]:
        target["cam_K"] = torch.cat([o["cam_K"] for o in kept], dim=0)

    boxes_list = [o["box"] for o in kept if o["box"] is not None]
    if boxes_list:
        target["boxes"] = BoundingBoxes(
            torch.cat(boxes_list, dim=0),
            format=BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        )
    else:
        return None

    target["area"] = torch.tensor([float(fm.sum()) for fm in final_masks], dtype=torch.float32)
    target["iscrowd"] = torch.zeros(len(kept), dtype=torch.int64)

    # visibility 재계산: visib_mask_area / full_mask_area
    full_masks_np = np.stack([objects[i]["full_mask"] for i in keep_indices], axis=0)
    visib_areas = np.array([float(fm.sum()) for fm in final_masks])
    full_areas = np.array([float(fm.sum()) for fm in full_masks_np])
    vis = np.divide(visib_areas, full_areas, out=np.zeros_like(visib_areas), where=full_areas > 0)
    target["visibility"] = torch.tensor(vis, dtype=torch.float32)

    return target


def _load_background_image(background_images, H, W):
    """배경 이미지 로드 (이미지 목록 또는 랜덤 단색)."""
    import random as _rng
    if background_images:
        bg_path = _rng.choice(background_images)
        bg_img = cv2.imread(bg_path)
        if bg_img is not None:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            return cv2.resize(bg_img, (W, H))
    color = [_rng.randint(0, 255) for _ in range(3)]
    return np.full((H, W, 3), color, dtype=np.uint8)


def _load_background_list(background_dir):
    """디렉토리에서 배경 이미지 경로 목록 로드."""
    from pathlib import Path
    if not background_dir:
        return []
    bg_path = Path(background_dir)
    if not bg_path.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return [str(p) for p in bg_path.iterdir() if p.suffix.lower() in exts]


# ============================================================================
# CopyPaste — depth-ordered two-image compositing
# ============================================================================

@register()
class CopyPaste(nn.Module):
    """
    Depth-ordered two-image copy-paste augmentation.

    같은 데이터셋에서 랜덤 이미지 B를 로드, A+B 물체를 tz 기반 depth ordering으로 합성.
    배경은 A의 비물체 영역 그대로 유지.

    Args:
        p: 적용 확률
        min_visibility: 최종 visibility 필터 임계값
        edge_blur_kernel: 엣지 블렌딩 커널 크기
        max_objects_from_b: B에서 가져올 최대 물체 수 (None이면 전부)
        min_objects_threshold: A의 물체 수가 이 값 이하일 때만 적용 (None이면 항상)
        max_total_objects: 합산 최대 물체 수 (None이면 무제한). B에서 가져올 개수를 자동 조절.
    """

    def __init__(self, p=0.5, min_visibility=0.1, edge_blur_kernel=5,
                 max_objects_from_b=None, min_objects_threshold=None, max_total_objects=None):
        super().__init__()
        self.p = p
        self.min_visibility = min_visibility
        self.edge_blur_kernel = edge_blur_kernel
        self.max_objects_from_b = max_objects_from_b
        self.min_objects_threshold = min_objects_threshold
        self.max_total_objects = max_total_objects

    def forward(self, *inputs):
        import random
        input_tuple = inputs[0] if len(inputs) == 1 else inputs

        if len(input_tuple) >= 3:
            image_a, target_a, dataset = input_tuple[0], input_tuple[1], input_tuple[2]
        else:
            return input_tuple

        if random.random() > self.p:
            return input_tuple
        if "masks" not in target_a or "poses" not in target_a:
            return input_tuple
        if isinstance(image_a, torch.Tensor):
            return input_tuple

        # 물체 수가 threshold 이하일 때만 적용
        if self.min_objects_threshold is not None:
            n_objects = len(target_a.get("labels", []))
            if n_objects > self.min_objects_threshold:
                return input_tuple

        try:
            idx_b = random.randint(0, len(dataset) - 1)
            image_b, target_b = dataset.load_item(idx_b)
        except Exception:
            return input_tuple

        if image_a.size != image_b.size:
            return input_tuple

        W, H = image_a.size

        objects_a = _extract_objects(image_a, target_a)
        objects_b = _extract_objects(image_b, target_b)

        if not objects_a and not objects_b:
            return input_tuple

        # B에서 가져올 물체 수 제한
        max_from_b = len(objects_b)
        if self.max_total_objects is not None:
            max_from_b = min(max_from_b, max(0, self.max_total_objects - len(objects_a)))
        if self.max_objects_from_b is not None:
            max_from_b = min(max_from_b, self.max_objects_from_b)
        if max_from_b <= 0:
            return input_tuple
        if len(objects_b) > max_from_b:
            objects_b = random.sample(objects_b, max_from_b)

        all_objects = objects_a + objects_b
        all_objects.sort(key=lambda x: -x["tz"])

        background = np.array(image_a).copy()
        canvas, occupancy = _depth_composite(all_objects, background, H, W, self.edge_blur_kernel)
        keep_indices, final_masks = _visibility_filter(all_objects, occupancy, self.min_visibility)

        if not keep_indices:
            return input_tuple

        merged = _build_target_from_objects(all_objects, keep_indices, final_masks, H, W)
        if merged is None:
            return input_tuple

        for k in target_a:
            if k not in merged:
                merged[k] = target_a[k]

        return (PILImage.fromarray(canvas), merged, dataset)


# ============================================================================
# RandomTransformAug — per-object Z축 회전 + VOC 배경 + depth ordering
# ============================================================================

@register()
class RandomTransformAug(nn.Module):
    """
    Per-object Z축 회전 + depth-ordered compositing + 배경 교체.

    각 물체를 visib_mask로 분리 → 개별 회전(±range, 확률적 180° 플립)
    → tz 기반 depth ordering으로 재합성 → 배경 교체.

    Args:
        principal_rotation_range: 회전 범위 (도)
        flip_prob: 180도 추가 회전 확률
        p: 적용 확률
        background_dir: 배경 이미지 디렉토리 (None이면 랜덤 단색)
        min_visibility: visibility 필터 임계값
        edge_blur_kernel: 엣지 블렌딩 커널 크기
        max_angle: legacy 파라미터 (principal_rotation_range로 변환)
    """

    def __init__(
        self,
        principal_rotation_range: float = 15.0,
        flip_prob: float = 0.3,
        p: float = 0.98,
        background_dir: Optional[str] = None,
        min_visibility: float = 0.1,
        edge_blur_kernel: int = 5,
        # Legacy parameters
        max_angle: Optional[float] = None,
        principal_rotation_flip: bool = False,
        fallback_for_renderer: bool = False,
        use_amodal_bbox: bool = True,
    ) -> None:
        super().__init__()
        if max_angle is not None and principal_rotation_range == 15.0:
            self.rotation_range = max_angle
        else:
            self.rotation_range = principal_rotation_range
        self.flip_prob = flip_prob
        self.p = p
        self.min_visibility = min_visibility
        self.edge_blur_kernel = edge_blur_kernel
        self.fallback_for_renderer = fallback_for_renderer

        bg_dir = os.path.expanduser(background_dir) if background_dir else None
        self._background_images = _load_background_list(bg_dir)

    def _rotate_object(self, obj, angle, H, W, cx, cy):
        """단일 물체를 principal point (cx, cy) 기준으로 Z축 회전.

        카메라 광축 회전과 동일: pixels는 (cx,cy) 기준 회전,
        pose는 R_new = Rz @ R, t_new = Rz @ t.

        Returns:
            rotated obj dict 또는 None (visibility 실패)
        """
        visib_mask = obj["visib_mask"]
        full_mask = obj["full_mask"]
        pixels = obj["pixels"]

        # principal point 기준 회전 행렬
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # 픽셀 & 마스크 회전
        rot_pixels = cv2.warpAffine(pixels, rot_mat, (W, H), borderValue=(0, 0, 0))
        rot_visib = cv2.warpAffine(visib_mask, rot_mat, (W, H), flags=cv2.INTER_NEAREST)
        rot_full = cv2.warpAffine(full_mask, rot_mat, (W, H), flags=cv2.INTER_NEAREST)

        # visibility 체크
        full_area = rot_full.sum()
        if full_area == 0:
            return None
        vis_area = rot_visib.sum()
        if vis_area / full_area < self.min_visibility:
            return None

        # modal bbox (visib_mask 기반) 최소 크기 체크
        vys, vxs = np.where(rot_visib > 0)
        if len(vys) == 0:
            return None
        if (int(vxs.max()) - int(vxs.min())) < 5 or (int(vys.max()) - int(vys.min())) < 5:
            return None

        # amodal bbox (full_mask 기반)
        fys, fxs = np.where(rot_full > 0)
        if len(fys) == 0:
            return None
        new_box = torch.tensor([[fxs.min(), fys.min(), fxs.max(), fys.max()]], dtype=torch.float32)

        # pose 업데이트: R_new = Rz @ R, t_new = Rz @ t
        pose = obj["pose"].clone()
        Rz = cv2.Rodrigues(np.array([0, 0, np.deg2rad(-angle)]))[0].astype(np.float32)

        t_old = pose[:3].numpy()
        R_old = pose[3:].reshape(3, 3).numpy()
        t_new = Rz @ t_old
        R_new = Rz @ R_old

        pose[:3] = torch.from_numpy(t_new).float()
        pose[3:] = torch.from_numpy(R_new.flatten()).float()

        new_obj = {
            "pixels": rot_pixels,
            "visib_mask": rot_visib.astype(np.uint8),
            "full_mask": rot_full.astype(np.uint8),
            "tz": float(t_new[2]),  # tz는 변하지 않지만 일관성을 위해
            "pose": pose,
            "label": obj["label"].clone(),
            "box": new_box,
        }
        if "cam_K" in obj:
            new_obj["cam_K"] = obj["cam_K"].clone()

        return new_obj

    def forward(self, *inputs):
        import random

        if self.fallback_for_renderer:
            input_tuple = inputs[0] if len(inputs) == 1 else inputs
            if input_tuple[1].get("_renderer_aug_applied", False):
                return inputs if len(inputs) > 1 else inputs[0]

        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        input_tuple = inputs[0] if len(inputs) == 1 else inputs
        image = input_tuple[0]
        target = input_tuple[1]
        dataset_info = input_tuple[2] if len(input_tuple) > 2 else None

        if "masks" not in target or "poses" not in target:
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        if isinstance(image, torch.Tensor):
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        W, H = image.size

        # cam_K에서 principal point 추출
        cam_K = target.get("cam_K")
        if cam_K is not None:
            cam_K_np = cam_K[0].numpy() if len(cam_K.shape) > 1 else cam_K.numpy()
            cx, cy = float(cam_K_np[2]), float(cam_K_np[5])
        else:
            cx, cy = W / 2.0, H / 2.0

        # 물체 추출
        objects = _extract_objects(image, target)
        if not objects:
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        # Per-object 회전 (principal point 기준)
        rotated_objects = []
        for obj in objects:
            # 회전 각도: ±range + 확률적 180° 플립
            delta = np.random.uniform(-self.rotation_range, self.rotation_range)
            if random.random() < self.flip_prob:
                delta += 180.0
            angle = delta

            rot_obj = self._rotate_object(obj, angle, H, W, cx, cy)
            if rot_obj is not None:
                rotated_objects.append(rot_obj)
            else:
                # 회전 후 visibility 실패 → 원본 유지
                rotated_objects.append(obj)

        # tz 기준 depth ordering
        rotated_objects.sort(key=lambda x: -x["tz"])

        # 배경 이미지
        background = _load_background_image(self._background_images, H, W)

        # Depth-ordered compositing
        canvas, occupancy = _depth_composite(rotated_objects, background, H, W, self.edge_blur_kernel)
        keep_indices, final_masks = _visibility_filter(rotated_objects, occupancy, self.min_visibility)

        if not keep_indices:
            # 모든 물체가 사라지면 원본 반환
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        merged = _build_target_from_objects(rotated_objects, keep_indices, final_masks, H, W)
        if merged is None:
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        # 원본 target에서 추가 필드 복사
        for k in target:
            if k not in merged:
                merged[k] = target[k]

        merged["_geometric_aug_applied"] = True
        result_image = PILImage.fromarray(canvas)

        if dataset_info is not None:
            return (result_image, merged, dataset_info)
        return (result_image, merged)


@register()
class FilterSmallBoxLowVis(nn.Module):
    """Modal bbox 최소 크기 + dynamic visibility 필터.

    Augmentation 이후 위치에 두면 zoom/crop/dropout으로 가려진 결과까지 반영해
    필터링된다. Visibility는 다음 우선순위로 계산:

      1) **Dynamic** (권장): PLY 모델에서 FPS 샘플한 점들을 현재 pose+cam_K로
         이미지에 투영하여 visible mask 안에 들어간 비율. coco_path /
         category_file이 주어졌을 때 활성화.
      2) Amodal mask 면적 비율 (`amodal_masks` / `full_masks`).
      3) Annotation의 `visibility` 필드 (BOP 정적 값).

    `coco_path`/`category_file`이 비어있거나 PLY 로딩 실패 시 자동으로 (2)→(3)
    fallback. 따라서 ignore=True 분기를 dataset에서 제거해도 동등한 supervision
    필터링이 학습 단에서 보장된다.
    """

    def __init__(
        self,
        min_size: int = 5,
        min_visib: float = 0.0,
        coco_path: Optional[str] = None,
        category_file: Optional[str] = None,
        num_fps_points: int = 1000,
    ):
        super().__init__()
        self.min_size = min_size
        self.min_visib = min_visib
        self.num_fps_points = int(num_fps_points)

        # {label_id (post-remap, 0-indexed): np.ndarray [N, 3] in mm}
        self._fps_cache: Dict[int, np.ndarray] = {}
        # {label_id: diameter in mm} — for tz<d/2 pre-filter (object intersects
        # near plane → some FPS pts have Z<0, projection unstable).
        self._diameter_cache: Dict[int, float] = {}
        if coco_path and category_file:
            self._load_fps_models(coco_path, category_file, self.num_fps_points)
            self._load_diameters(coco_path, category_file)

    def _load_fps_models(self, coco_path: str, category_file: str, num_pts: int) -> None:
        try:
            import yaml
            import hashlib
        except Exception as exc:
            print(f"[FilterSmallBoxLowVis] dynamic vis disabled (import failed: {exc})")
            return

        coco_path_exp = os.path.expanduser(coco_path)
        cat_path = os.path.expanduser(category_file)
        models_dir = os.path.join(coco_path_exp, "models")
        if not (os.path.isdir(models_dir) and os.path.isfile(cat_path)):
            print("[FilterSmallBoxLowVis] dynamic vis disabled (missing models dir or category file)")
            return

        # Disk cache: key on (models_dir abs path, category_file abs path, num_pts).
        # PLY 파일 mtime까지 합쳐 변경 감지.
        ply_paths = sorted(
            [os.path.join(models_dir, fn) for fn in os.listdir(models_dir) if fn.endswith(".ply")]
        )
        ply_mtimes = ",".join(f"{os.path.basename(p)}:{os.path.getmtime(p):.0f}" for p in ply_paths)
        key = f"{os.path.abspath(models_dir)}|{os.path.abspath(cat_path)}|{num_pts}|{ply_mtimes}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        cache_dir = os.path.expanduser("~/.cache/race6d/fps_models")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"fps_{digest}.pt")

        if os.path.isfile(cache_path):
            try:
                self._fps_cache = torch.load(cache_path, weights_only=False)
                print(
                    f"[FilterSmallBoxLowVis] dynamic vis enabled: loaded {len(self._fps_cache)} "
                    f"models from cache ({cache_path})"
                )
                return
            except Exception as exc:
                print(f"[FilterSmallBoxLowVis] cache load failed ({exc}), recomputing...")

        try:
            import open3d as o3d  # type: ignore[import-not-found]
            from pytorch3d.ops import sample_farthest_points  # type: ignore[import-not-found]
        except Exception as exc:
            print(f"[FilterSmallBoxLowVis] dynamic vis disabled (import failed: {exc})")
            return

        with open(cat_path, "r") as f:
            cat = yaml.safe_load(f).get("category2name", {})

        loaded = 0
        for label_id, cat_id in enumerate(cat.keys()):
            ply_path = os.path.join(models_dir, f"obj_{int(cat_id):06d}.ply")
            if not os.path.isfile(ply_path):
                continue
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                pts = np.asarray(pcd.points, dtype=np.float32)
                if pts.shape[0] == 0:
                    continue
                if pts.shape[0] > num_pts:
                    pts_t = torch.from_numpy(pts).unsqueeze(0)
                    sampled, _ = sample_farthest_points(pts_t, K=num_pts)
                    pts = sampled.squeeze(0).numpy().astype(np.float32)
                self._fps_cache[label_id] = pts
                loaded += 1
            except Exception as exc:
                print(f"[FilterSmallBoxLowVis] failed to load {ply_path}: {exc}")

        if loaded > 0:
            try:
                torch.save(self._fps_cache, cache_path)
                print(
                    f"[FilterSmallBoxLowVis] dynamic vis enabled: {loaded} models, "
                    f"{num_pts} FPS pts each (cached → {cache_path})"
                )
            except Exception as exc:
                print(f"[FilterSmallBoxLowVis] cache save failed ({exc}), continuing without cache")

    def _load_diameters(self, coco_path: str, category_file: str) -> None:
        """Load per-class diameter (mm) from BOP `models_info.json`."""
        try:
            import yaml, json
        except Exception:
            return
        coco_path_exp = os.path.expanduser(coco_path)
        cat_path = os.path.expanduser(category_file)
        info_path = os.path.join(coco_path_exp, "models", "models_info.json")
        if not (os.path.isfile(cat_path) and os.path.isfile(info_path)):
            return
        with open(cat_path, "r") as f:
            cat = yaml.safe_load(f).get("category2name", {})
        with open(info_path, "r") as f:
            info = json.load(f)
        for label_id, cat_id in enumerate(cat.keys()):
            entry = info.get(str(int(cat_id))) or info.get(int(cat_id))
            if entry and "diameter" in entry:
                self._diameter_cache[label_id] = float(entry["diameter"])

    def _compute_dynamic_visib(
        self, target: Dict[str, Any], masks_np: np.ndarray
    ) -> Optional[np.ndarray]:
        """PLY 점을 pose+K로 투영해서 visible mask 안에 든 비율을 instance별로 계산.

        반환 [N]: 모델이 로드된 instance는 [0,1] 비율, 안 된 instance는 NaN.
        """
        if not self._fps_cache:
            return None

        poses = target.get("poses")
        cam_K = target.get("cam_K")
        labels = target.get("labels")
        if poses is None or cam_K is None or labels is None:
            return None

        poses_data = poses.data if hasattr(poses, "data") else poses
        poses_np = (
            poses_data.numpy() if torch.is_tensor(poses_data) else np.asarray(poses_data)
        )
        cam_K_np = cam_K.numpy() if torch.is_tensor(cam_K) else np.asarray(cam_K)
        labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)

        N = masks_np.shape[0]
        H, W = masks_np.shape[1], masks_np.shape[2]
        vis = np.full(N, np.nan, dtype=np.float32)

        for i in range(N):
            label_id = int(labels_np[i])
            pts = self._fps_cache.get(label_id)
            if pts is None:
                continue  # NaN → fallback path takes over

            pose_i = poses_np[i].reshape(-1)
            t = pose_i[:3].astype(np.float32)

            # Pre-filter: tz < diameter/2 means object intersects camera near plane.
            # Some FPS pts will have Z<0 → projection ill-defined. Mark vis=0 (filter).
            diameter = self._diameter_cache.get(label_id)
            if diameter is not None and float(t[2]) < 0.5 * diameter:
                vis[i] = 0.0
                continue

            R = pose_i[3:12].reshape(3, 3).astype(np.float32)
            K = (
                cam_K_np[i].reshape(3, 3) if cam_K_np.ndim > 1 else cam_K_np.reshape(3, 3)
            ).astype(np.float32)

            X_cam = pts @ R.T + t  # [N_pts, 3] in mm
            Z = X_cam[:, 2]
            valid = Z > 1e-3
            if not valid.any():
                vis[i] = 0.0
                continue

            # Z 1mm 하한: tz<d/2 케이스를 위에서 걸렀으므로 Z>0 가 보장되지만
            # 수치 안전을 위해 1mm 클램프 (int32 cast 안전 범위 보장).
            Z_safe = np.maximum(Z, 1.0)
            u = K[0, 0] * X_cam[:, 0] / Z_safe + K[0, 2]
            v = K[1, 1] * X_cam[:, 1] / Z_safe + K[1, 2]
            ui = np.floor(u + 0.5).astype(np.int32)
            vi = np.floor(v + 0.5).astype(np.int32)
            in_img = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if not in_img.any():
                vis[i] = 0.0
                continue

            mask = masks_np[i] > 0
            n_inside = int(mask[vi[in_img], ui[in_img]].sum())
            vis[i] = n_inside / max(pts.shape[0], 1)

        return vis

    def forward(self, *inputs):
        input_tuple = inputs[0] if len(inputs) == 1 else inputs
        image = input_tuple[0]
        target = input_tuple[1]
        dataset_info = input_tuple[2] if len(input_tuple) > 2 else None

        masks = target.get("masks")
        if masks is None:
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)
        N = masks_np.shape[0]
        H, W = masks_np.shape[1], masks_np.shape[2]

        # Vectorized bbox-size check.
        # `np.any` works directly on integer masks (0 → False, nonzero → True),
        # so we avoid the [N,H,W] boolean materialization that `masks_np > 0` does.
        y_any = masks_np.any(axis=2)                  # [N, H]
        x_any = masks_np.any(axis=1)                  # [N, W]
        has_any = y_any.any(axis=1)                   # [N]

        # First / last True column via argmax. argmax on all-False rows returns 0,
        # which is harmless because `has_any` masks those instances to width=height=0.
        x_min = x_any.argmax(axis=1)
        x_max = (W - 1) - x_any[:, ::-1].argmax(axis=1)
        y_min = y_any.argmax(axis=1)
        y_max = (H - 1) - y_any[:, ::-1].argmax(axis=1)
        widths = np.where(has_any, x_max - x_min, 0)
        heights = np.where(has_any, y_max - y_min, 0)

        size_ok = (widths >= self.min_size) & (heights >= self.min_size) & has_any

        # Visibility: dynamic (PLY 투영) 1순위, 실패한 instance는 ann visibility로 fallback.
        dyn_vis = self._compute_dynamic_visib(target, masks_np) if self.min_visib > 0 else None

        ann_visib = None
        if self.min_visib > 0:
            v = target.get("visibility")
            if v is not None:
                ann_visib = v.numpy() if torch.is_tensor(v) else np.asarray(v)

        if self.min_visib > 0:
            visib_fract = np.full(N, np.nan, dtype=np.float32)
            if dyn_vis is not None:
                visib_fract = np.where(np.isnan(dyn_vis), visib_fract, dyn_vis)
            if ann_visib is not None:
                use_ann = np.isnan(visib_fract) & (np.arange(N) < len(ann_visib))
                if use_ann.any():
                    visib_fract = np.where(use_ann, ann_visib[:N], visib_fract)
            # NaN remaining → no source available, treat as visible (no filter)
            visib_fract = np.where(np.isnan(visib_fract), 1.0, visib_fract)
            visib_ok = visib_fract >= self.min_visib
        else:
            visib_ok = np.ones(N, dtype=bool)

        keep = torch.tensor(size_ok & visib_ok, dtype=torch.bool)

        if keep.all():
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        if not keep.any():
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        # Explicit whitelist of per-object fields so we never collide with
        # fixed-shape metadata like orig_size ([W, H], shape [2]) when a sample
        # happens to have the same number of GTs as the metadata length.
        per_object_keys = {
            "boxes", "masks", "full_masks", "amodal_masks",
            "labels", "poses", "cam_K", "visibility",
            "area", "iscrowd",
        }
        filtered = {}
        for k, v in target.items():
            if (
                k in per_object_keys
                and isinstance(v, (torch.Tensor, BoundingBoxes, Mask, Pose))
                and v.shape[0] == len(keep)
            ):
                if isinstance(v, BoundingBoxes):
                    filtered[k] = BoundingBoxes(v.data[keep], format=v.format, canvas_size=v.canvas_size)
                elif isinstance(v, Mask):
                    filtered[k] = Mask(v[keep])
                elif isinstance(v, Pose):
                    filtered[k] = Pose(v.data[keep])
                else:
                    filtered[k] = v[keep]
            else:
                filtered[k] = v

        if dataset_info is not None:
            return (image, filtered, dataset_info)
        return (image, filtered)

@register()
class BackgroundReplacement:
    """배경만 교체하는 경량 augmentation.

    RendererAugmentation, RandomTransformAug 모두 적용되지 않은 이미지에만 적용.
    """

    def __init__(
        self,
        background_dir: str,
        p: float = 0.5,
    ):
        self.p = p
        self._background_dir = os.path.expanduser(background_dir)
        self._background_images: Optional[List[str]] = None

    def _load_backgrounds(self) -> List[str]:
        from pathlib import Path
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [
            str(p) for p in Path(self._background_dir).iterdir()
            if p.suffix.lower() in exts
        ]

    def __call__(self, *inputs):
        input_tuple = inputs[0] if len(inputs) == 1 else inputs
        image = input_tuple[0]
        target = input_tuple[1]
        dataset_info = input_tuple[2] if len(input_tuple) > 2 else None

        # 기하학적 augmentation이 이미 적용된 경우 스킵
        if target.get("_renderer_aug_applied", False):
            return inputs if len(inputs) > 1 else inputs[0]
        if target.get("_geometric_aug_applied", False):
            return inputs if len(inputs) > 1 else inputs[0]

        if np.random.rand() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # Lazy init
        if self._background_images is None:
            self._background_images = self._load_backgrounds()
        if not self._background_images:
            return inputs if len(inputs) > 1 else inputs[0]

        masks = target.get("masks")
        if masks is None:
            return inputs if len(inputs) > 1 else inputs[0]

        masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)

        # 합산 마스크 (어떤 인스턴스라도 있는 픽셀)
        if masks_np.ndim == 3:
            combined = np.any(masks_np > 0, axis=0)
        else:
            combined = masks_np > 0

        # 배경 이미지 로드 및 리사이즈
        bg_path = self._background_images[np.random.randint(len(self._background_images))]
        bg_img = PILImage.open(bg_path).convert("RGB")

        image_np = np.array(image)
        bg_np = np.array(bg_img.resize((image_np.shape[1], image_np.shape[0])))

        # 배경 교체: 마스크 밖 = 새 배경
        result = image_np.copy()
        result[~combined] = bg_np[~combined]

        result_image = PILImage.fromarray(result)
        target["_bg_replaced"] = True

        if dataset_info is not None:
            return (result_image, target, dataset_info)
        return (result_image, target)


def _depth_aug_image_border_mask(H: int, W: int, margin: int) -> np.ndarray:
    bm = np.zeros((H, W), dtype=bool)
    if margin > 0:
        bm[:margin, :] = True
        bm[-margin:, :] = True
        bm[:, :margin] = True
        bm[:, -margin:] = True
    return bm


def _depth_aug_stamp_ellipse(depth, valid_region, yy, xx, cy, cx,
                             target_px, rx_ratio_range, rng):
    """Stamp a random-orientation elliptical cluster (set to 0) within valid_region.

    Uses bbox-local grid instead of full H*W allocations: since max target_px
    ≈ 800 → base_r ≈ 16, the working bbox is typically ~32*32 = 1024 pixels
    vs 576*768 = 442368 pixels. ~400x fewer ops per stamp.

    `yy`, `xx` args kept for backwards compat but unused.
    """
    rx_ratio = rng.uniform(*rx_ratio_range)
    base_r = np.sqrt(max(target_px, 1) / np.pi)
    rx = base_r * rx_ratio
    ry = base_r / rx_ratio
    theta = rng.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Axis-aligned bbox half-widths of the rotated ellipse
    bx = int(np.ceil(np.sqrt((rx * cos_t) ** 2 + (ry * sin_t) ** 2))) + 1
    by = int(np.ceil(np.sqrt((rx * sin_t) ** 2 + (ry * cos_t) ** 2))) + 1
    H, W = depth.shape
    y0 = max(0, cy - by); y1 = min(H, cy + by + 1)
    x0 = max(0, cx - bx); x1 = min(W, cx + bx + 1)
    if y1 <= y0 or x1 <= x0:
        return
    ly = np.arange(y0 - cy, y1 - cy, dtype=np.float32)[:, None]
    lx = np.arange(x0 - cx, x1 - cx, dtype=np.float32)[None, :]
    dy_r = ly * cos_t - lx * sin_t
    dx_r = ly * sin_t + lx * cos_t
    local_ellipse = ((dy_r / ry) ** 2 + (dx_r / rx) ** 2) <= 1.0
    local_region = local_ellipse & valid_region[y0:y1, x0:x1]
    patch = depth[y0:y1, x0:x1]
    patch[local_region] = 0.0


@register()
class DepthAugment(nn.Module):
    """Sim-to-real depth augmentation for BOP RGBD pose estimation.

    TLESS test real sensor data 분석 기반 증강:
    - Object mask boundary에 hole 3x 편향 → clustered_hole_gen (edge_bias)
    - 중간 거리 scene 영역 (table/floor) 에도 hole → bg_clustered_hole_gen
    - Far background (Z_MAX 근처)는 제외
    - Image 테두리에는 hole 생성 안 함
    - Edge wobble 효과 → wavy_boundary_warp (coherent displacement)
    - Flying pixel 효과 → boundary_speckle (per-pixel neighbor swap)
    - GDRNPP 3종도 유지 (fill / gaussian noise)

    입력: dataset의 `_load_depth_tensor`가 반환한 [0, 1] 정규화 depth tensor.
    출력: target['_depth_tensor']에 증강된 depth 저장 → __getitem__이 concat 시 사용.

    모든 파라미터는 [0, 1] 스케일 기준 (Z_MAX_MM=2000 가정).
    예: fill_std=0.05 ≈ 100mm, noise_base_std=0.001 ≈ 2mm.
    """

    def __init__(
        self,
        # (1) Zero point fill (기존 hole을 작은 noise로 채움, GDRNPP)
        fill_prob: float = 1.0,
        fill_std: float = 0.05,
        # (2) Wavy boundary warp (mild + tight: 촘촘한 웨이브)
        wavy_prob: float = 0.7,
        wavy_amplitude: float = 1.5,
        wavy_smoothness: float = 5.0,
        wavy_edge_decay: float = 4.0,
        wavy_image_edge_margin: int = 5,
        # (3) Boundary speckle (per-pixel swap, flying pixel)
        speckle_prob: float = 0.6,
        speckle_thickness: int = 1,
        speckle_swap_prob: float = 0.4,
        # (4) Object clustered holes (visibility-aware: TLESS real sensor 실측)
        # 낮은 visibility (occluded sliver) 일수록 real sensor는 depth hole이 더 많음
        # (Pearson corr = -0.59, 100vs100 분석 기반).
        obj_hole_prob: float = 0.7,
        obj_n_clusters_min: int = 1,
        obj_n_clusters_max: int = 5,
        obj_cluster_size_min: int = 5,
        obj_cluster_size_max: int = 200,
        obj_edge_bias: float = 0.7,
        obj_image_edge_margin: int = 5,
        # (5) Background clustered holes (scene 영역, far bg 제외)
        bg_hole_prob: float = 0.7,
        bg_n_clusters_min: int = 1,
        bg_n_clusters_max: int = 4,
        bg_cluster_size_min: int = 50,
        bg_cluster_size_max: int = 800,
        bg_image_edge_margin: int = 10,
        bg_far_depth_threshold: float = 0.95,
        # (6) Gaussian noise (valid pixel 전역)
        noise_prob: float = 0.6,
        noise_base_std: float = 0.001,
        noise_level_max: float = 0.0025,  # iter-random upper bound (GDRNPP style)
    ):
        super().__init__()
        # (1)
        self.fill_prob = float(fill_prob)
        self.fill_std = float(fill_std)
        # (2)
        self.wavy_prob = float(wavy_prob)
        self.wavy_amplitude = float(wavy_amplitude)
        self.wavy_smoothness = float(wavy_smoothness)
        self.wavy_edge_decay = float(wavy_edge_decay)
        self.wavy_image_edge_margin = int(wavy_image_edge_margin)
        # (3)
        self.speckle_prob = float(speckle_prob)
        self.speckle_thickness = int(speckle_thickness)
        self.speckle_swap_prob = float(speckle_swap_prob)
        # (4)
        self.obj_hole_prob = float(obj_hole_prob)
        self.obj_n_clusters_range = (int(obj_n_clusters_min), int(obj_n_clusters_max))
        self.obj_cluster_size_range = (int(obj_cluster_size_min), int(obj_cluster_size_max))
        self.obj_edge_bias = float(obj_edge_bias)
        self.obj_image_edge_margin = int(obj_image_edge_margin)
        # (5)
        self.bg_hole_prob = float(bg_hole_prob)
        self.bg_n_clusters_range = (int(bg_n_clusters_min), int(bg_n_clusters_max))
        self.bg_cluster_size_range = (int(bg_cluster_size_min), int(bg_cluster_size_max))
        self.bg_image_edge_margin = int(bg_image_edge_margin)
        self.bg_far_depth_threshold = float(bg_far_depth_threshold)
        # (6)
        self.noise_prob = float(noise_prob)
        self.noise_base_std = float(noise_base_std)
        self.noise_level_max = float(noise_level_max)

    # ------- helpers -------

    def _extract_masks(self, target, target_H, target_W):
        """target['masks']를 depth 해상도에 맞춘 list of bool ndarray로 변환."""
        masks = target.get("masks")
        if masks is None:
            return []
        if isinstance(masks, torch.Tensor):
            t = masks
        else:
            try:
                t = torch.as_tensor(np.asarray(masks))
            except Exception:
                return []
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 0:
            return []
        # Resize masks to depth spatial size if mismatch
        if tuple(t.shape[-2:]) != (target_H, target_W):
            t_f = t.float()
            if t_f.ndim == 3:
                t_f = t_f.unsqueeze(0)  # [1, N, H, W]
            t_r = torch.nn.functional.interpolate(
                t_f, size=(target_H, target_W), mode="nearest"
            ).squeeze(0)
        else:
            t_r = t.float()
        t_r = (t_r > 0.5).cpu().numpy()
        return [t_r[i] for i in range(t_r.shape[0])]

    def _wavy_warp(self, depth, masks, rng):
        """Coherent wavy warp at mask boundaries.

        Uses cv2.GaussianBlur (C, multithreaded) instead of scipy.gaussian_filter,
        and cv2.distanceTransform instead of scipy.distance_transform_edt.
        cv2.remap replaces scipy.map_coordinates.
        """
        H, W = depth.shape
        raw_dy = rng.randn(H, W).astype(np.float32)
        raw_dx = rng.randn(H, W).astype(np.float32)
        # ksize=0 → derived from sigma; sigma is in pixels, matches scipy's sigma
        sigma = float(self.wavy_smoothness)
        dy_field = cv2.GaussianBlur(raw_dy, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        dx_field = cv2.GaussianBlur(raw_dx, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
        dy_std = float(dy_field.std())
        dx_std = float(dx_field.std())
        if dy_std > 1e-9:
            dy_field *= self.wavy_amplitude / dy_std
        if dx_std > 1e-9:
            dx_field *= self.wavy_amplitude / dx_std
        edge_weight = np.zeros((H, W), dtype=np.float32)
        if masks:
            any_mask = np.zeros((H, W), dtype=np.uint8)
            for m in masks:
                any_mask |= m.astype(np.uint8)
            dist_in = cv2.distanceTransform(any_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            dist_out = cv2.distanceTransform(1 - any_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            signed_dist = np.minimum(dist_in, dist_out)
            edge_weight = np.exp(-signed_dist / self.wavy_edge_decay).astype(np.float32)
        if self.wavy_image_edge_margin > 0:
            mg = self.wavy_image_edge_margin
            edge_weight[:mg, :] = 0
            edge_weight[-mg:, :] = 0
            edge_weight[:, :mg] = 0
            edge_weight[:, -mg:] = 0
        dy_field *= edge_weight
        dx_field *= edge_weight
        # cv2.remap takes (map_x, map_y) in float32
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        map_x = (xx + dx_field).astype(np.float32)
        map_y = (yy + dy_field).astype(np.float32)
        return cv2.remap(
            depth, map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)

    def _boundary_speckle(self, depth, masks, rng):
        """Vectorized boundary speckle swap.

        Build union mask once, compute ring via cv2 morphology (C), then do
        a single fancy-indexed swap. Semantically equivalent to the old
        per-mask+per-pixel loop: old code may swap overlapping ring pixels
        twice, this does each ring pixel at most once, but total swap count
        (`speckle_swap_prob * |ring|`) is unchanged.
        """
        if not masks:
            return depth
        out = depth.copy()
        H, W = depth.shape
        t = self.speckle_thickness
        union = np.zeros((H, W), dtype=np.uint8)
        for m in masks:
            if m.sum() >= 20:
                union |= m.astype(np.uint8)
        if union.sum() == 0:
            return out
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(union, kernel, iterations=t)
        eroded = cv2.erode(union, kernel, iterations=t)
        ring = dilated.astype(bool) & ~eroded.astype(bool)
        ring_ys, ring_xs = np.where(ring)
        if len(ring_ys) == 0:
            return out
        n_swap = int(self.speckle_swap_prob * len(ring_ys))
        if n_swap == 0:
            return out
        idx = rng.choice(len(ring_ys), size=n_swap, replace=False)
        ys = ring_ys[idx]
        xs = ring_xs[idx]
        dys = rng.randint(-t - 1, t + 2, size=n_swap)
        dxs = rng.randint(-t - 1, t + 2, size=n_swap)
        nys = np.clip(ys + dys, 0, H - 1)
        nxs = np.clip(xs + dxs, 0, W - 1)
        out[ys, xs] = depth[nys, nxs]
        return out

    @staticmethod
    def _target_hole_frac(vis, rng):
        """Empirical hole fraction as a function of visibility (TLESS real).

        실측 (1000 val images):
          vis 0.1-0.2: mean 0.59, fully_hole 22.4%
          vis 0.2-0.4: mean 0.34, fully_hole  6.7%
          vis 0.4-0.6: mean 0.13, fully_hole  1.7%
          vis 0.6-0.8: mean 0.05, fully_hole  0.2%
          vis 0.8-1.0: mean 0.02, fully_hole  0.0%
        """
        if vis < 0.2:
            if rng.rand() < 0.22:
                return 1.0
            return rng.uniform(0.3, 0.9)
        if vis < 0.4:
            if rng.rand() < 0.07:
                return 1.0
            return rng.uniform(0.1, 0.6)
        if vis < 0.6:
            return rng.uniform(0.02, 0.3)
        if vis < 0.8:
            return rng.uniform(0.01, 0.15)
        return rng.uniform(0.0, 0.05)

    def _obj_clustered_holes(self, depth, masks, visibilities, rng):
        """Per-mask clustered hole stamping, visibility-aware.

        Real TLESS sensor 실측을 기반으로:
          - 낮은 visibility (heavily occluded sliver) 물체는 depth hole이 많음
          - visibility가 높으면 거의 온전한 depth
        target_hole_frac(visibility)로 mask당 목표 hole 비율을 결정하고,
        cluster 기반 stamping으로 그 양만큼 hole을 채움.

        vis < 0.2 의 22%는 전체가 hole이라 valid_mask 전체를 0으로 short-circuit.
        """
        out = depth.copy()
        H, W = depth.shape
        yy, xx = np.mgrid[:H, :W]
        border_mask = _depth_aug_image_border_mask(H, W, self.obj_image_edge_margin)
        not_border = (~border_mask).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        size_lo, size_hi = self.obj_cluster_size_range
        for idx, mask in enumerate(masks):
            vu8 = mask.astype(np.uint8) & not_border
            mask_area = int(vu8.sum())
            if mask_area < 20:
                continue
            vis = float(visibilities[idx]) if idx < len(visibilities) else 1.0

            target_frac = self._target_hole_frac(vis, rng)
            # Full hole shortcut: 전체 visible mask를 통째로 0으로
            if target_frac >= 0.999:
                out[vu8.astype(bool)] = 0.0
                continue
            target_px = int(round(mask_area * target_frac))
            if target_px < size_lo:
                continue

            valid_mask = vu8.astype(bool)
            interior_u8 = cv2.erode(vu8, kernel, iterations=3)
            boundary = valid_mask & ~interior_u8.astype(bool)
            mask_ys, mask_xs = np.where(valid_mask)
            boundary_ys, boundary_xs = np.where(boundary)

            # Budget-based stamping: 누적 hole이 target_px에 도달할 때까지 cluster 생성.
            # cluster 크기는 원래 범위 유지 (실측 cluster shape과 일치).
            budget = target_px
            max_stamps = max(self.obj_n_clusters_range[1] * 4, 8)  # safety cap
            stamps = 0
            while budget > size_lo and stamps < max_stamps:
                if rng.rand() < self.obj_edge_bias and len(boundary_ys) > 0:
                    i = rng.randint(len(boundary_ys))
                    cy, cx = int(boundary_ys[i]), int(boundary_xs[i])
                else:
                    i = rng.randint(len(mask_ys))
                    cy, cx = int(mask_ys[i]), int(mask_xs[i])
                this_px = min(budget, rng.randint(size_lo, size_hi + 1))
                _depth_aug_stamp_ellipse(out, valid_mask, yy, xx, cy, cx,
                                         this_px, (0.5, 2.0), rng)
                budget -= this_px
                stamps += 1
        return out

    def _bg_clustered_holes(self, depth, masks, rng):
        out = depth.copy()
        H, W = depth.shape
        yy, xx = np.mgrid[:H, :W]
        obj_union = np.zeros((H, W), dtype=bool)
        for m in masks:
            obj_union |= m
        border_mask = _depth_aug_image_border_mask(H, W, self.bg_image_edge_margin)
        far_mask = depth >= self.bg_far_depth_threshold
        valid_bg = (~obj_union) & (~border_mask) & (~far_mask) & (depth > 0)
        if valid_bg.sum() < 100:
            return out
        valid_ys, valid_xs = np.where(valid_bg)
        n_clusters = rng.randint(self.bg_n_clusters_range[0],
                                 self.bg_n_clusters_range[1] + 1)
        for _ in range(n_clusters):
            i = rng.randint(len(valid_ys))
            cy, cx = int(valid_ys[i]), int(valid_xs[i])
            target_px = rng.randint(self.bg_cluster_size_range[0],
                                    self.bg_cluster_size_range[1] + 1)
            _depth_aug_stamp_ellipse(out, valid_bg, yy, xx, cy, cx,
                                     target_px, (0.5, 2.0), rng)
        return out

    # ------- main -------

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            inputs = inputs[0]
        image = inputs[0]
        target = inputs[1] if len(inputs) > 1 else {}
        dataset_info = inputs[2] if len(inputs) > 2 else None

        if not isinstance(target, dict) or "_depth_path" not in target:
            return inputs if len(inputs) > 1 else inputs[0]
        if dataset_info is None or not hasattr(dataset_info, "_load_depth_tensor"):
            return inputs if len(inputs) > 1 else inputs[0]

        depth_t = dataset_info._load_depth_tensor(target["_depth_path"])
        if depth_t is None:
            return inputs if len(inputs) > 1 else inputs[0]

        depth = depth_t.squeeze(0).numpy().astype(np.float32)  # [H, W] in [0, 1]

        # Resize depth to match the transform-target resolution (after Resize
        # transform) so downstream _extract_masks doesn't need to re-resize
        # masks. Target resolution is inferred from masks (preferred — already
        # at the Resize target) or image shape as fallback.
        target_hw = None
        masks_obj = target.get("masks")
        if masks_obj is not None and hasattr(masks_obj, "shape") and len(masks_obj.shape) >= 2:
            target_hw = (int(masks_obj.shape[-2]), int(masks_obj.shape[-1]))
        elif hasattr(image, "size"):  # PIL.Image: .size = (W, H)
            target_hw = (int(image.size[1]), int(image.size[0]))
        elif hasattr(image, "shape") and len(image.shape) >= 2:
            target_hw = (int(image.shape[-2]), int(image.shape[-1]))
        if target_hw is not None and depth.shape != target_hw:
            depth = cv2.resize(
                depth,
                (target_hw[1], target_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        H, W = depth.shape
        rng = np.random  # use global numpy random state

        # Masks are already at (H, W); _extract_masks will just convert to bool.
        masks = self._extract_masks(target, H, W)
        # Per-mask visibility (annotation value). Default 1.0 if missing.
        vis_t = target.get("visibility")
        if vis_t is None:
            visibilities = [1.0] * len(masks)
        else:
            if isinstance(vis_t, torch.Tensor):
                visibilities = vis_t.detach().cpu().tolist()
            else:
                visibilities = list(np.asarray(vis_t).tolist())
            # Pad/truncate to match masks (safety against len mismatch)
            if len(visibilities) < len(masks):
                visibilities = visibilities + [1.0] * (len(masks) - len(visibilities))

        # (1) Zero point fill — 기존 hole을 N(0, fill_std)로 채움
        if rng.rand() < self.fill_prob:
            hole_idx = depth == 0
            if hole_idx.any():
                fill = rng.normal(0.0, self.fill_std, size=int(hole_idx.sum())).astype(np.float32)
                depth = depth.copy()
                depth[hole_idx] = fill

        # (2) Wavy boundary warp (mask가 있을 때만 의미 있음)
        if masks and rng.rand() < self.wavy_prob:
            depth = self._wavy_warp(depth, masks, rng)

        # (3) Boundary speckle
        if masks and rng.rand() < self.speckle_prob:
            depth = self._boundary_speckle(depth, masks, rng)

        # (4) Object mask clustered holes (visibility-aware)
        if masks and rng.rand() < self.obj_hole_prob:
            depth = self._obj_clustered_holes(depth, masks, visibilities, rng)

        # (5) Background clustered holes
        if rng.rand() < self.bg_hole_prob:
            depth = self._bg_clustered_holes(depth, masks, rng)

        # (6) Gaussian noise on valid pixels
        if rng.rand() < self.noise_prob:
            valid = depth > 0
            level = float(rng.uniform(self.noise_base_std, self.noise_level_max))
            if level > 0 and valid.any():
                noise = (rng.randn(*depth.shape) * level).astype(np.float32)
                depth = np.where(valid, depth + noise, depth).astype(np.float32)

        depth = np.clip(depth, 0.0, 1.0).astype(np.float32)
        target["_depth_tensor"] = torch.from_numpy(depth).unsqueeze(0).float()

        if len(inputs) > 2:
            return (image, target, dataset_info)
        return (image, target)


# ============================================================================
# RendererAugmentation - PyTorch3D 기반 R|t 증강
# ============================================================================

# PyRender imports (EGL backend for headless rendering)
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # Must be set before pyrender import

try:
    import pyrender

    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

_OPEN3D_AVAILABLE = (
    False  # Open3D removed: Embree deadlocks under fork()-based DataLoader
)
