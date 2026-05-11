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

from PIL import Image as PILImage, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated cc_textures JPEGs

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
@register()
class Resize(T.Resize):
    @staticmethod
    def _get_hw(img):
        if hasattr(img, 'shape'):
            return img.shape[-2], img.shape[-1]
        return img.height, img.width

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        orig_h, orig_w = self._get_hw(sample[0])

        out = super().forward(*inputs)
        out_sample = out if isinstance(out, (list, tuple)) else (out,)
        new_h, new_w = self._get_hw(out_sample[0])

        target = out_sample[1]
        if "px_count_all" in target:
            area_ratio = (new_h * new_w) / (orig_h * orig_w)
            target["px_count_all"] = target["px_count_all"] * area_ratio

        return out

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
        import os
        if os.environ.get("AUG_FORCE_P_ONE", "0") == "1":
            return 1.0
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

    ITODD = {
        "itodd_dark_25": [25, 25, 25],
        "itodd_dark_32": [32, 32, 32],
        "itodd_dark_39": [39, 39, 39],
        "itodd_dark_48": [48, 48, 48],
        "itodd_dark_58": [58, 58, 58],
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

    def __init__(self, alpha_range: Tuple[float, float] = (0.0, 1.0), p: float = None) -> None:
        super().__init__()
        self.alpha_range = alpha_range
        self.aug_manager = AugmentationManager()
        self.p = p if p is not None else self.aug_manager.get_probability("grayscale")

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
            alpha = params["alpha"]
            arr = np.array(inpt)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if alpha >= 1.0:
                gray_3ch = cv2.merge([gray, gray, gray])
                return PILImage.fromarray(gray_3ch)
            gray_3ch = cv2.merge([gray, gray, gray])
            blended = cv2.addWeighted(arr, 1.0 - alpha, gray_3ch, alpha, 0)
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
        p: float = None,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range
        self.aug_manager = AugmentationManager()
        self.p = p if p is not None else self.aug_manager.get_probability("gaussian_blur")

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        kernel_size = self.kernel_sizes[
            torch.randint(0, len(self.kernel_sizes), (1,)).item()
        ]
        sigma = float(torch.empty(1).uniform_(self.sigma_range[0], self.sigma_range[1]))
        if sigma < 0.1:
            return inputs if len(inputs) > 1 else inputs[0]

        if len(inputs) > 1:
            image, rest = inputs[0], inputs[1:]
        else:
            image, rest = inputs[0], ()

        if isinstance(image, Image):
            image = F.to_pil_image(image)
        if isinstance(image, PILImage.Image):
            img_np = np.array(image, dtype=np.float32)
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
            image = PILImage.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

        return (image, *rest) if rest else image


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
        p: float = None,
        drop_prob: Union[float, Tuple[float, float]] = 0.2,
        size_percent: Union[float, Tuple[float, float]] = 0.05,
        per_channel: bool = False,
        min_size: int = 3,
        dilation: int = 0,
    ) -> None:
        super().__init__()

        # drop_prob: 픽셀별 드롭아웃 확률
        if isinstance(drop_prob, (int, float)):
            self.p_range = (drop_prob, drop_prob)
        else:
            self.p_range = drop_prob

        # size_percent 처리 (낮은 해상도 크기)
        if isinstance(size_percent, (int, float)):
            self.size_percent_range = (size_percent, size_percent)
        else:
            self.size_percent_range = size_percent

        self.per_channel = per_channel
        self.min_size = min_size
        self.dilation = dilation

        self.aug_manager = AugmentationManager()
        self.p_apply = p if p is not None else self.aug_manager.get_probability("coarse_dropout")

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
            for c in range(channels):
                low_mask = (
                    torch.rand(low_height, low_width) >= p_dropout
                ).float().numpy()

                if self.dilation > 0:
                    kernel = np.ones((self.dilation, self.dilation), np.uint8)
                    low_mask = cv2.erode(low_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)

                mask = cv2.resize(
                    low_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
                )
                img_result[:, :, c] *= mask
        else:
            low_mask = (torch.rand(low_height, low_width) >= p_dropout).float().numpy()

            if self.dilation > 0:
                kernel = np.ones((self.dilation, self.dilation), np.uint8)
                low_mask = cv2.erode(low_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)

            mask = cv2.resize(
                low_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
            )

            if len(img.shape) == 3:
                mask = mask[:, :, np.newaxis]
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
class RandomISPSimulation(T.Transform):
    """Camera ISP pipeline simulation: Debayering artifacts + Unsharp masking + CLAHE.

    PBR 렌더링에는 없는 실제 카메라 ISP 처리를 시뮬레이션하여 sim-to-real gap 축소.
    SurfEmb (CVPR 2022) 구현 기반.
    """

    _transformed_types = (
        PILImage.Image,
        Image,
    )

    def __init__(
        self,
        p: float = 0.5,
        debayer: bool = True,
        unsharp: bool = True,
        clahe: bool = True,
        unsharp_k_limits: Tuple[int, int] = (3, 7),
        unsharp_strength: Tuple[float, float] = (0.0, 2.0),
        clahe_clip_limit: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.p = p
        self.debayer = debayer
        self.unsharp = unsharp
        self.clahe = clahe
        self.unsharp_k_limits = unsharp_k_limits
        self.unsharp_strength = unsharp_strength
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid = clahe_grid

    def _apply_debayer(self, img: np.ndarray) -> np.ndarray:
        channel_idxs = np.random.permutation(3)
        channel_idxs_inv = np.empty(3, dtype=int)
        channel_idxs_inv[channel_idxs] = 0, 1, 2

        bayer = np.zeros(img.shape[:2], dtype=img.dtype)
        bayer[::2, ::2] = img[::2, ::2, channel_idxs[2]]
        bayer[1::2, ::2] = img[1::2, ::2, channel_idxs[1]]
        bayer[::2, 1::2] = img[::2, 1::2, channel_idxs[1]]
        bayer[1::2, 1::2] = img[1::2, 1::2, channel_idxs[0]]

        method = np.random.choice((cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_BG2BGR_EA))
        return cv2.cvtColor(bayer, method)[..., channel_idxs_inv]

    def _apply_unsharp(self, img: np.ndarray) -> np.ndarray:
        k = np.random.randint(
            self.unsharp_k_limits[0] // 2, self.unsharp_k_limits[1] // 2 + 1
        ) * 2 + 1
        s = k / 3
        blur = cv2.GaussianBlur(img, (k, k), s)
        strength = np.random.uniform(*self.unsharp_strength)
        return cv2.addWeighted(img, 1 + strength, blur, -strength, 0)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        cl = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid
        )
        lab[:, :, 0] = cl.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return {
            "apply_transform": torch.rand(1).item() < self.p,
            "do_debayer": self.debayer and torch.rand(1).item() < 0.5,
            "do_unsharp": self.unsharp and torch.rand(1).item() < 0.5,
            "do_clahe": self.clahe and torch.rand(1).item() < 0.5,
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply_transform"]:
            return inpt
        if not isinstance(inpt, (PILImage.Image, Image)):
            return inpt

        if isinstance(inpt, Image):
            inpt = F.to_pil_image(inpt)
        img = np.array(inpt)

        if params["do_debayer"]:
            img = self._apply_debayer(img)
        if params["do_unsharp"]:
            img = self._apply_unsharp(img)
        if params["do_clahe"]:
            img = self._apply_clahe(img)

        return PILImage.fromarray(img)


@register()
class PerClassColorDiversification(nn.Module):
    """Per-class HSV color diversification on visib_mask regions.

    PBR 텍스처의 색감 오버피팅을 방지하기 위해 클래스별로 독립적인
    HSV 변환을 적용. 텍스처 구조(edge, gradient)는 보존하고 색상만 변경.
    """

    def __init__(
        self,
        hue_range: float = 30.0,
        saturation_range: Tuple[float, float] = (0.7, 1.4),
        value_range: Tuple[float, float] = (0.8, 1.2),
        p: float = None,
    ) -> None:
        super().__init__()
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range
        self.aug_manager = AugmentationManager()
        self.p = p if p is not None else 0.5

    def _random_hsv_params(self) -> Tuple[float, float, float]:
        hue_shift = float(torch.empty(1).uniform_(-self.hue_range, self.hue_range))
        sat_scale = float(torch.empty(1).uniform_(*self.saturation_range))
        val_scale = float(torch.empty(1).uniform_(*self.value_range))
        return hue_shift, sat_scale, val_scale

    def _apply_hsv_shift(
        self, img_hsv: np.ndarray, mask: np.ndarray,
        hue_shift: float, sat_scale: float, val_scale: float,
    ) -> np.ndarray:
        mask_bool = mask > 0.5
        if not mask_bool.any():
            return img_hsv

        h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h[mask_bool] = (h[mask_bool].astype(np.float32) + hue_shift) % 180
        s[mask_bool] = np.clip(s[mask_bool].astype(np.float32) * sat_scale, 0, 255)
        v[mask_bool] = np.clip(v[mask_bool].astype(np.float32) * val_scale, 0, 255)
        img_hsv[:, :, 0] = h.astype(np.uint8)
        img_hsv[:, :, 1] = s.astype(np.uint8)
        img_hsv[:, :, 2] = v.astype(np.uint8)
        return img_hsv

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target = inputs[0], inputs[1]

        if torch.rand(1) >= self.p:
            return inputs

        if "masks" not in target or "labels" not in target:
            return inputs

        masks = target["masks"]
        if hasattr(masks, "data"):
            masks = masks.data
        labels = target["labels"]

        if len(masks) == 0:
            return inputs

        if isinstance(image, Image):
            image = F.to_pil_image(image)
        img_np = np.array(image)
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        unique_labels = torch.unique(labels)
        class_hsv_params = {int(lbl): self._random_hsv_params() for lbl in unique_labels}

        for i in range(len(masks)):
            lbl = int(labels[i])
            mask_np = masks[i].numpy() if isinstance(masks[i], torch.Tensor) else masks[i]
            if len(mask_np.shape) == 3:
                mask_np = mask_np.max(axis=0)
            hue_shift, sat_scale, val_scale = class_hsv_params[lbl]
            img_hsv = self._apply_hsv_shift(img_hsv, mask_np, hue_shift, sat_scale, val_scale)

        img_result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        result_image = PILImage.fromarray(img_result)

        out = list(inputs)
        out[0] = result_image
        return tuple(out)


@register()
class MaskRegionDegradation(nn.Module):
    """visib_mask 영역에만 blur + noise를 적용하여 PBR 렌더링의 과도한 선명함을 완화."""

    def __init__(
        self,
        blur_kernel: int = 7,
        blur_sigma_range: Tuple[float, float] = (2.0, 3.0),
        noise_scale_range: Tuple[float, float] = (5.0, 10.0),
        p: float = 0.9,
    ) -> None:
        super().__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range
        self.noise_scale_range = noise_scale_range
        self.p = p

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target = inputs[0], inputs[1]

        if torch.rand(1) >= self.p:
            return inputs

        if "masks" not in target:
            return inputs

        masks = target["masks"]
        if hasattr(masks, "data"):
            masks = masks.data
        if len(masks) == 0:
            return inputs

        if isinstance(image, Image):
            image = F.to_pil_image(image)
        img_np = np.array(image, dtype=np.float32)

        # union of all visib masks
        masks_np = masks.numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)
        union_mask = (masks_np.max(axis=0) > 0.5).astype(np.float32)

        # blur
        sigma = float(torch.empty(1).uniform_(*self.blur_sigma_range))
        blurred = cv2.GaussianBlur(img_np, (self.blur_kernel, self.blur_kernel), sigma)

        # noise
        noise_scale = float(torch.empty(1).uniform_(*self.noise_scale_range))
        noise = np.random.normal(0, noise_scale, img_np.shape).astype(np.float32)

        # apply only on mask region
        union_3ch = union_mask[:, :, None]
        degraded = blurred + noise
        result = img_np * (1.0 - union_3ch) + degraded * union_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)

        out = list(inputs)
        out[0] = PILImage.fromarray(result)
        return tuple(out)


@register()
class GlobalDegradation(nn.Module):
    """전체 이미지에 blur + noise를 한번에 적용. cv2 기반."""

    def __init__(
        self,
        blur_kernel: int = 7,
        blur_sigma_range: Tuple[float, float] = (2.0, 3.0),
        noise_scale_range: Tuple[float, float] = (5.0, 10.0),
        p: float = 0.9,
    ) -> None:
        super().__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma_range = blur_sigma_range
        self.noise_scale_range = noise_scale_range
        self.p = p

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        image = inputs[0]

        if torch.rand(1) >= self.p:
            return inputs

        if isinstance(image, Image):
            image = F.to_pil_image(image)
        img_np = np.array(image, dtype=np.float32)

        sigma = float(torch.empty(1).uniform_(*self.blur_sigma_range))
        blurred = cv2.GaussianBlur(img_np, (self.blur_kernel, self.blur_kernel), sigma)

        noise_scale = float(torch.empty(1).uniform_(*self.noise_scale_range))
        noise = np.random.normal(0, noise_scale, img_np.shape).astype(np.float32)

        result = np.clip(blurred + noise, 0, 255).astype(np.uint8)

        out = list(inputs)
        out[0] = PILImage.fromarray(result)
        return tuple(out)


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
        scale_range: Tuple[float, float] = None,
        per_channel: bool = True,
        p: float = None,
    ) -> None:
        """Args:
            scale: 고정 노이즈 표준편차 (scale_range가 없을 때 사용)
            scale_range: [min, max] 범위에서 랜덤 샘플링 (scale보다 우선)
            per_channel: 채널별 독립 노이즈
            p: 적용 확률 override
        """
        super().__init__()
        self.scale = scale
        self.scale_range = scale_range
        self.per_channel = per_channel
        self.aug_manager = AugmentationManager()
        self.p = p if p is not None else self.aug_manager.get_probability("gaussian_noise")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1) >= self.p:
            return {"apply": False}
        if self.scale_range is not None:
            s = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]))
        else:
            s = self.scale
        return {"apply": True, "scale": s}

    def _apply_gaussian_noise(self, img: np.ndarray, scale: float) -> np.ndarray:
        if isinstance(img, Image):
            img = F.to_pil_image(img)
        if isinstance(img, PILImage.Image):
            img = np.array(img)
        noise = np.empty(img.shape, dtype=np.int16)
        n_ch = img.shape[2] if img.ndim == 3 else 1
        mean = tuple([0] * n_ch)
        std = tuple([scale] * n_ch)
        cv2.randn(noise, mean, std)
        result = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
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
        self,
        mean: float = 0.0,
        scale_range: Tuple[float, float] = (7.0, 10.0),
        p: float = None,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.scale_range = self._check_range(scale_range, "scale_range")
        self.aug_manager = AugmentationManager()
        self.p = p if p is not None else self.aug_manager.get_probability("additional_noise")

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

# ============================================================================
# Shared helpers for depth-ordered compositing and geometric pose augmentation.
#
# Consumers: CopyPasteSingleClass, FillSingleClass, PoseAugmentation.
#
# object dict contract (from `_extract_objects` / `_extract_object_at`):
#   - pixels        : H×W×3 uint8       — image masked by visib_mask (else 0)
#   - visib_mask    : H×W uint8         — modal / visible silhouette
#   - full_mask     : H×W uint8         — amodal silhouette (REQUIRED for
#                                         geometric pose aug; helper falls
#                                         back to visib_mask, but
#                                         PoseAugmentation rejects targets
#                                         without `full_masks` upfront)
#   - tz            : float (mm)        — pose[2]
#   - pose          : tensor[12]        — flat (tx, ty, tz, R[3,3] row-major)
#   - label         : tensor[1] int64   — class id
#   - box           : tensor[1, 4] xyxy — modal bbox or None
#   - cam_K         : tensor[1, 9]      — flat intrinsics or absent
#   - px_count_all  : float             — BOP amodal pixel count (3× canvas)
# ============================================================================

def _gaussian_blur_edge(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """마스크 엣지에 Gaussian blur 적용 (블렌딩용)"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)


def _extract_objects(image, target):
    """이미지에서 물체들을 visib_mask 기반으로 추출.

    Returns:
        list of dict: 각 물체 정보 (pixels, visib_mask, full_mask, px_count_all, tz, pose, label, box, cam_K)
    """
    objects = []
    masks = target.get("masks")
    poses = target.get("poses")
    labels = target.get("labels")
    cam_K = target.get("cam_K")
    full_masks = target.get("full_masks")
    boxes = target.get("boxes")
    px_count_all = target.get("px_count_all")

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

        if px_count_all is not None:
            val = px_count_all[i]
            obj["px_count_all"] = float(val.item() if isinstance(val, torch.Tensor) else val)
        else:
            obj["px_count_all"] = float(visib_mask.sum())

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


def _extract_object_at(image, target, idx):
    """Single-instance variant of `_extract_objects`. Byte-identical to
    `_extract_objects(image, target)[idx]` but skips the other instances —
    used by CopyPasteSingleClass where only one candidate per image is
    needed (saves ~11x time on ITODD's 9.8-instance/image average).
    """
    masks = target.get("masks")
    poses = target.get("poses")
    labels = target.get("labels")
    cam_K = target.get("cam_K")
    full_masks = target.get("full_masks")
    boxes = target.get("boxes")
    px_count_all = target.get("px_count_all")

    if masks is None or poses is None or labels is None:
        return None

    masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)
    if idx >= len(masks_np):
        return None
    visib_mask = masks_np[idx]
    if visib_mask.sum() == 0:
        return None

    img_np = np.array(image)
    poses_data = poses.data if isinstance(poses, Pose) else poses
    box_data = boxes.data if hasattr(boxes, "data") else (
        boxes if boxes is not None else None
    )

    pose = poses_data[idx]
    tz = float(pose[2].item() if isinstance(pose[2], torch.Tensor) else pose[2])

    pixels = img_np.copy()
    pixels[visib_mask == 0] = 0

    obj = {
        "pixels": pixels,
        "visib_mask": visib_mask.astype(np.uint8),
        "tz": tz,
        "pose": pose.clone() if isinstance(pose, torch.Tensor) else torch.tensor(pose),
        "label": labels[idx:idx + 1].clone(),
        "box": box_data[idx:idx + 1].clone() if box_data is not None else None,
    }

    if px_count_all is not None:
        val = px_count_all[idx]
        obj["px_count_all"] = float(val.item() if isinstance(val, torch.Tensor) else val)
    else:
        obj["px_count_all"] = float(visib_mask.sum())

    if cam_K is not None:
        obj["cam_K"] = (
            cam_K[idx:idx + 1].clone()
            if len(cam_K.shape) > 1
            else cam_K.unsqueeze(0).clone()
        )

    if full_masks is not None:
        fm = full_masks[idx]
        fm_np = fm.numpy() if torch.is_tensor(fm) else np.array(fm)
        obj["full_mask"] = fm_np.astype(np.uint8)
    else:
        obj["full_mask"] = visib_mask.astype(np.uint8)

    return obj


def _resize_sample(image, target, target_size):
    """Resize PIL image + image-plane fields of `target` to `target_size=(W, H)`.

    Scales: masks/full_masks (NEAREST), boxes (xyxy pixel), px_count_all (area).
    Pose, cam_K, labels are unchanged (metric / original-frame data).

    CopyPasteSingleClass uses this when running AFTER Resize: candidates loaded
    via dataset.load_item come back at raw image size, this brings them down to
    the post-Resize size before composite.
    """
    W_new, H_new = target_size
    W_old, H_old = image.size
    if (W_new, H_new) == (W_old, H_old):
        return image, target

    sx = W_new / W_old
    sy = H_new / H_old

    image = image.resize((W_new, H_new), PILImage.BILINEAR)

    masks = target.get("masks")
    if masks is not None:
        m_data = masks.data if hasattr(masks, "data") else masks
        m_np = m_data.numpy() if torch.is_tensor(m_data) else np.asarray(m_data)
        if len(m_np) > 0:
            resized = np.stack([
                cv2.resize(mi.astype(np.uint8), (W_new, H_new),
                           interpolation=cv2.INTER_NEAREST)
                for mi in m_np
            ])
        else:
            resized = np.zeros((0, H_new, W_new), dtype=np.uint8)
        target["masks"] = Mask(torch.from_numpy(resized))

    full_masks = target.get("full_masks")
    if full_masks is not None:
        fm_data = full_masks.data if hasattr(full_masks, "data") else full_masks
        fm_np = fm_data.numpy() if torch.is_tensor(fm_data) else np.asarray(fm_data)
        if len(fm_np) > 0:
            resized = np.stack([
                cv2.resize(fmi.astype(np.uint8), (W_new, H_new),
                           interpolation=cv2.INTER_NEAREST)
                for fmi in fm_np
            ])
        else:
            resized = np.zeros((0, H_new, W_new), dtype=np.uint8)
        target["full_masks"] = Mask(torch.from_numpy(resized))

    boxes = target.get("boxes")
    if boxes is not None:
        b_data = boxes.data if hasattr(boxes, "data") else boxes
        scaled = b_data.clone().float()
        scaled[:, 0] *= sx
        scaled[:, 2] *= sx
        scaled[:, 1] *= sy
        scaled[:, 3] *= sy
        target["boxes"] = BoundingBoxes(
            scaled, format=BoundingBoxFormat.XYXY, canvas_size=(H_new, W_new)
        )

    if target.get("px_count_all") is not None:
        target["px_count_all"] = target["px_count_all"] * (sx * sy)

    return image, target


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
        pxa = obj.get("px_count_all", 0)
        denom = pxa if pxa > 0 else obj["full_mask"].sum()
        if denom == 0:
            continue
        visibility = final_mask.sum() / denom
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

    pxa = torch.tensor([o["px_count_all"] for o in kept], dtype=torch.float32)
    target["px_count_all"] = pxa

    visib_areas = torch.tensor([float(fm.sum()) for fm in final_masks], dtype=torch.float32)
    target["visibility"] = torch.where(
        pxa > 1e-6, visib_areas / pxa.clamp(min=1.0), torch.zeros_like(pxa)
    ).clamp(0.0, 1.0)

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
# Geometric pose augmentation: principal-point rotation + zoom
#
#   - PoseAugmentation     : scene / instance modes via `per_instance` switch
#   - `_augment_object_pose`: per-object affine kernel used by instance mode
# ============================================================================

def _augment_object_pose(obj, cx, cy, angle_deg, scale, H, W):
    """인스턴스 하나에 principal point 기준 회전 + zoom 적용 (in-place).

    cv2 CCW angle on y-down image → camera Z-axis rotation by -angle.
    zoom scale s → tz_new = tz / s (투영 크기 s배 확대).
    """
    if abs(angle_deg) < 0.1 and abs(scale - 1.0) < 1e-4:
        return

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    obj["pixels"] = cv2.warpAffine(
        obj["pixels"], M, (W, H),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    obj["visib_mask"] = cv2.warpAffine(
        obj["visib_mask"], M, (W, H),
        flags=cv2.INTER_NEAREST, borderValue=0)
    obj["full_mask"] = cv2.warpAffine(
        obj["full_mask"], M, (W, H),
        flags=cv2.INTER_NEAREST, borderValue=0)

    angle_rad = np.deg2rad(angle_deg)
    c, s_val = np.cos(-angle_rad), np.sin(-angle_rad)
    Rz = np.array([[c, -s_val, 0], [s_val, c, 0], [0, 0, 1]], dtype=np.float32)

    pose = obj["pose"]
    t_old = pose[:3].numpy().astype(np.float32)
    R_old = pose[3:].numpy().reshape(3, 3).astype(np.float32)

    t_new = Rz @ t_old
    t_new[2] /= scale
    R_new = Rz @ R_old

    obj["pose"] = torch.from_numpy(np.concatenate([t_new, R_new.reshape(9)])).float()
    obj["tz"] = float(t_new[2])

    if "px_count_all" in obj:
        obj["px_count_all"] *= (scale * scale)

    ys, xs = np.nonzero(obj["full_mask"] > 0)
    if len(xs) > 0:
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        obj["box"] = torch.from_numpy(box).unsqueeze(0)


@register()
class PoseAugmentation(nn.Module):
    """Unified geometric pose augmentation (rotation + zoom around principal point).

    Modes
    -----
    per_instance=False (scene mode)
        모든 instance에 동일 (angle, scale) affine 적용. cam_K cx,cy 기준.
        rotation_range=0 → zoom only / zoom_range=(1,1) → rotation only / 둘 다
        지정 시 결합 변환.

    per_instance=True (instance mode)
        각 object별로 독립 (angle, scale) 샘플 → cv2 affine warp →
        depth-ordered composite. revert_visibility 미만으로 가려진 object는
        backup으로 rollback.

    Math (cv2 image y-down → BOP camera frame, x=right, y=down, z=forward):
        M       = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
        Rz_3d   = Rodrigues([0, 0, deg2rad(-angle_deg)])
        t_new   = Rz_3d @ t_old; t_new[2] /= scale
        R_new   = Rz_3d @ R_old
        cam_K   = unchanged (principal-point centered transform)
        px_all  = px_all * scale²

    full_masks REQUIREMENT
    ----------------------
    Geometric warp 후 amodal bbox는 warped `full_masks`에서 nonzero tight box로
    재계산됨. 따라서 target에 `full_masks`가 없으면 `RuntimeError` 발생 (fail-fast).
    visib_mask로의 silent fallback은 amodal 학습 시그널을 손상시키므로 금지.

    Args
    ----
        per_instance     : False = scene-wide / True = per-object
        rotation_range   : ±deg uniform sample. 0 → rotation off
        zoom_range       : (min, max) uniform sample. (1.0, 1.0) → zoom off
        p                : 적용 확률
        background_dir   : warp 후 빈 영역(scene mode) / canvas(instance mode)
                           채움용 image 디렉토리. None이면 fill_value 단색
        fill_value       : background_dir 없을 때 단색 padding
        revert_visibility: instance mode에서 변환 후 visibility가 이 값 미만이면 변환 취소
        edge_blur_kernel : instance mode composite 경계 블렌딩 커널 (0=hard)
        p_rot180         : 추가 180° flip 확률
    """

    def __init__(
        self,
        per_instance: bool = False,
        rotation_range: float = 0.0,
        zoom_range: Optional[Tuple[float, float]] = None,
        p: float = 0.5,
        background_dir: Optional[str] = None,
        fill_value: Union[int, Tuple[int, int, int]] = 114,
        revert_visibility: float = 0.7,
        edge_blur_kernel: int = 5,
        p_rot180: float = 0.0,
    ) -> None:
        super().__init__()
        self.per_instance = bool(per_instance)
        self.rotation_range = float(rotation_range)
        if zoom_range is None:
            self.zoom_range = (1.0, 1.0)
        else:
            self.zoom_range = (float(zoom_range[0]), float(zoom_range[1]))
        self.p = float(p)
        self.fill_value = (
            fill_value if isinstance(fill_value, (tuple, list))
            else (int(fill_value), int(fill_value), int(fill_value))
        )
        bg_dir = os.path.expanduser(background_dir) if background_dir else None
        self._background_images = _load_background_list(bg_dir) if bg_dir else []
        self.revert_visibility = float(revert_visibility)
        self.edge_blur_kernel = int(edge_blur_kernel)
        self.p_rot180 = float(p_rot180)

    # ---- common helpers --------------------------------------------------

    def _sample_params(self) -> Tuple[float, float]:
        angle = (
            float(np.random.uniform(-self.rotation_range, self.rotation_range))
            if self.rotation_range > 0
            else 0.0
        )
        if self.p_rot180 > 0.0 and np.random.rand() < self.p_rot180:
            angle += 180.0
        if self.zoom_range[0] == 1.0 and self.zoom_range[1] == 1.0:
            scale = 1.0
        else:
            scale = float(np.random.uniform(self.zoom_range[0], self.zoom_range[1]))
        return angle, scale

    @staticmethod
    def _principal_point(target: dict, W: int, H: int) -> Tuple[float, float]:
        cam_K = target.get("cam_K")
        if cam_K is not None and isinstance(cam_K, torch.Tensor):
            ck = cam_K[0].numpy() if cam_K.dim() > 1 else cam_K.numpy()
            return float(ck[2]), float(ck[5])
        return W / 2.0, H / 2.0

    @staticmethod
    def _pose_rz(angle_deg: float) -> torch.Tensor:
        """cv2 CCW(y-down image) angle → BOP camera +z rotation by -angle."""
        a = np.deg2rad(-angle_deg)
        c, s = np.cos(a), np.sin(a)
        return torch.from_numpy(
            np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        )

    def _fill_blank(self, warped_img: np.ndarray, valid_mask: np.ndarray,
                    H: int, W: int) -> np.ndarray:
        """Warp 후 invalid 영역을 background_dir 이미지 / fill_value 단색으로 채움."""
        if valid_mask.all():
            return warped_img
        if self._background_images:
            bg_img = _load_background_image(self._background_images, H, W)
        else:
            bg_img = np.full((H, W, 3), self.fill_value, dtype=np.uint8)
        return np.where(valid_mask[..., None], warped_img, bg_img)

    @staticmethod
    def _warp_masks_3d(masks_np: np.ndarray, M: np.ndarray,
                       H: int, W: int) -> np.ndarray:
        """[N, H, W] uint8 mask 스택 → 동일한 affine으로 warp."""
        return np.stack(
            [
                cv2.warpAffine(
                    m.astype(np.uint8), M, (W, H),
                    flags=cv2.INTER_NEAREST, borderValue=0,
                )
                for m in masks_np
            ],
            axis=0,
        ).astype(np.uint8)

    @staticmethod
    def _amodal_boxes_from_full(full_arr: np.ndarray) -> np.ndarray:
        """[N, H, W] full_mask → [N, 4] xyxy tight bbox (empty → (0,0,0,0))."""
        N = full_arr.shape[0]
        boxes = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            ys, xs = np.nonzero(full_arr[i] > 0)
            if len(xs) == 0:
                continue
            boxes[i, 0] = float(xs.min())
            boxes[i, 1] = float(ys.min())
            boxes[i, 2] = float(xs.max() + 1)
            boxes[i, 3] = float(ys.max() + 1)
        return boxes

    @staticmethod
    def _return(image, target, dataset_info):
        if dataset_info is not None:
            return (image, target, dataset_info)
        return (image, target)

    # ---- forward ---------------------------------------------------------

    def forward(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            input_tuple = inputs[0]
        else:
            input_tuple = inputs

        if not isinstance(input_tuple, tuple) or len(input_tuple) < 2:
            return inputs if len(inputs) > 1 else inputs[0]

        image = input_tuple[0]
        target = input_tuple[1]
        dataset_info = input_tuple[2] if len(input_tuple) > 2 else None

        if not isinstance(image, PILImage.Image):
            return self._return(image, target, dataset_info)
        if "masks" not in target or "poses" not in target:
            return self._return(image, target, dataset_info)

        # fail-fast: full_masks REQUIRED for amodal bbox recomputation
        if "full_masks" not in target or target.get("full_masks") is None:
            raise RuntimeError(
                "PoseAugmentation requires `full_masks` in target for amodal "
                "bbox recomputation after geometric warp. Configure the "
                "dataset to emit amodal masks (e.g. `return_masks: True` with "
                "amodal annotations) before applying this transform."
            )

        if torch.rand(1).item() >= self.p:
            return self._return(image, target, dataset_info)

        if self.per_instance:
            return self._apply_instance(image, target, dataset_info)
        return self._apply_scene(image, target, dataset_info)

    # ---- scene mode ------------------------------------------------------

    def _apply_scene(self, image, target, dataset_info):
        W, H = image.size
        cx, cy = self._principal_point(target, W, H)

        angle, scale = self._sample_params()
        if abs(angle) < 1e-3 and abs(scale - 1.0) < 1e-4:
            return self._return(image, target, dataset_info)

        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

        # 1) Image warp + blank fill
        img_np = np.array(image)
        warped_img = cv2.warpAffine(
            img_np, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        valid_mask = (
            cv2.warpAffine(
                np.full((H, W), 255, dtype=np.uint8), M, (W, H),
                flags=cv2.INTER_NEAREST, borderValue=0,
            )
            > 127
        )
        warped_img = self._fill_blank(warped_img, valid_mask, H, W)

        # 2) Mask warp (visib + full)
        new_target = dict(target)
        masks = target["masks"]
        masks_np = masks.numpy() if torch.is_tensor(masks) else np.asarray(masks)
        warped_masks = self._warp_masks_3d(masks_np, M, H, W)
        masks_t = torch.from_numpy(warped_masks)
        new_target["masks"] = Mask(masks_t) if isinstance(masks, Mask) else masks_t

        full_masks = target["full_masks"]
        fm_np = full_masks.numpy() if torch.is_tensor(full_masks) else np.asarray(full_masks)
        warped_full = self._warp_masks_3d(fm_np, M, H, W)
        full_t = torch.from_numpy(warped_full)
        new_target["full_masks"] = (
            Mask(full_t) if isinstance(full_masks, Mask) else full_t
        )

        # 3) Pose transform: Rz @ t, tz /= scale, R_new = Rz @ R
        Rz = self._pose_rz(angle)
        poses = target["poses"]
        poses_data = poses.data if hasattr(poses, "data") else poses
        poses_data = poses_data.clone()
        N = poses_data.shape[0]
        t_old = poses_data[:, :3]
        R_old = poses_data[:, 3:].reshape(N, 3, 3)
        t_new = (Rz @ t_old.unsqueeze(-1)).squeeze(-1)
        if abs(scale - 1.0) > 1e-4:
            t_new = t_new.clone()
            t_new[:, 2] = t_new[:, 2] / scale
        R_new = Rz @ R_old
        new_poses_data = torch.cat([t_new, R_new.reshape(N, 9)], dim=-1)
        new_target["poses"] = (
            Pose(new_poses_data) if isinstance(poses, Pose) else new_poses_data
        )

        # 4) Bbox 재계산 from warped full_masks (amodal)
        new_boxes = self._amodal_boxes_from_full(warped_full)
        new_target["boxes"] = BoundingBoxes(
            torch.from_numpy(new_boxes),
            format=BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        )

        # 5) px_count_all (rotation 보존, zoom z²) + visibility = visib/px_all
        if "px_count_all" in target:
            scale_factor = float(scale) ** 2
            pxa = target["px_count_all"]
            if isinstance(pxa, torch.Tensor):
                new_pxa = pxa.float() * scale_factor
            else:
                new_pxa = torch.from_numpy(
                    np.asarray(pxa, dtype=np.float32) * scale_factor
                )
            new_target["px_count_all"] = new_pxa

            visib_areas = masks_t.reshape(N, -1).sum(dim=1).float()
            new_target["visibility"] = torch.where(
                new_pxa > 1e-6,
                visib_areas / new_pxa.clamp(min=1.0),
                torch.zeros_like(new_pxa),
            ).clamp(0.0, 1.0)

        result_image = PILImage.fromarray(warped_img.astype(np.uint8))
        return self._return(result_image, new_target, dataset_info)

    # ---- instance mode ---------------------------------------------------

    def _apply_instance(self, image, target, dataset_info):
        import random
        import copy

        W, H = image.size
        cx, cy = self._principal_point(target, W, H)

        objects = _extract_objects(image, target)
        if not objects:
            return self._return(image, target, dataset_info)

        # background = non-object pixel median (단색 패치 회피)
        img_np = np.array(image)
        all_mask = np.zeros((H, W), dtype=bool)
        for obj in objects:
            all_mask |= (obj["visib_mask"] > 0)
        non_obj = ~all_mask
        if non_obj.any():
            median_color = np.median(img_np[non_obj], axis=0).astype(np.uint8)
        else:
            median_color = np.array([128, 128, 128], dtype=np.uint8)
        background = img_np.copy()
        background[all_mask] = median_color

        for i, obj in enumerate(objects):
            pxa = obj["px_count_all"]
            if pxa <= 0:
                continue
            backup = copy.deepcopy(obj)
            angle, scale = self._sample_params()
            _augment_object_pose(obj, cx, cy, angle, scale, H, W)
            denom = max(obj["px_count_all"], 1.0)
            vis = float(obj["visib_mask"].sum()) / denom
            if vis < self.revert_visibility:
                objects[i] = backup

        objects.sort(key=lambda x: -x["tz"])
        canvas, occupancy = _depth_composite(
            objects, background, H, W, self.edge_blur_kernel
        )

        keep_indices = []
        final_masks = []
        for i in range(len(objects)):
            fm = (occupancy == i).astype(np.uint8)
            if fm.sum() > 0:
                keep_indices.append(i)
                final_masks.append(fm)

        if not keep_indices:
            return self._return(image, target, dataset_info)

        merged = _build_target_from_objects(objects, keep_indices, final_masks, H, W)
        if merged is None:
            return self._return(image, target, dataset_info)

        # Amodal bbox 재계산: each kept obj's warped full_mask
        kept = [objects[i] for i in keep_indices]
        N = len(kept)
        full_arr = np.stack([o["full_mask"] for o in kept], axis=0)
        new_boxes = self._amodal_boxes_from_full(full_arr)
        merged["boxes"] = BoundingBoxes(
            torch.from_numpy(new_boxes),
            format=BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        )

        for k in target:
            if k not in merged:
                merged[k] = target[k]

        result = PILImage.fromarray(canvas)
        return self._return(result, merged, dataset_info)

# ============================================================================
# FillSingleClass — single-class scene augmentation for ITODD-like test
# ============================================================================

@register()
class FillSingleClass(nn.Module):
    """Single-class scene augmentation by duplicating one high-visibility object.

    현재 이미지에서 visibility가 가장 높은 물체 1개를 선택,
    n_target개로 복제하여 각각 랜덤 rotation/zoom 적용 후 depth-ordered composite.
    배경 교체는 별도 증강(RandomBackgroundWithPresets)에서 처리.

    Args:
        min_total: 최종 물체 수 하한
        max_total: 최종 물체 수 상한
        min_visibility: composite 후 visibility 하한
        rotation_range: ±도 범위
        p_rot180: 180° 추가 회전 확률
        zoom_min: zoom 하한
        zoom_max: zoom 상한
        edge_blur_kernel: 엣지 블렌딩 커널 크기
    """

    def __init__(self, min_total=1, max_total=5,
                 min_visibility=0.7,
                 rotation_range=30.0, p_rot180=0.5,
                 zoom_min=0.9, zoom_max=1.1,
                 edge_blur_kernel=5,
                 max_attempts=30):
        super().__init__()
        self.min_total = min_total
        self.max_total = max_total
        self.min_visibility = min_visibility
        self.rotation_range = float(rotation_range)
        self.p_rot180 = float(p_rot180)
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.edge_blur_kernel = int(edge_blur_kernel)
        self.max_attempts = int(max_attempts)

    def _select_source(self, target):
        """Lightweight eligible object selection (no pixel extraction)."""
        masks = target["masks"]
        masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)
        pxa_all = target.get("px_count_all")

        eligible = []
        for i in range(len(masks_np)):
            visib_area = float(masks_np[i].sum())
            pxa = float(pxa_all[i]) if pxa_all is not None else visib_area
            if pxa > 0 and visib_area / pxa >= self.min_visibility:
                eligible.append(i)
        return eligible

    def forward(self, *inputs):
        import random
        input_tuple = inputs[0] if len(inputs) == 1 else inputs

        if len(input_tuple) >= 3:
            image, target, dataset = input_tuple[0], input_tuple[1], input_tuple[2]
        else:
            return input_tuple

        if "masks" not in target or "poses" not in target:
            return input_tuple
        if isinstance(image, torch.Tensor):
            return input_tuple

        eligible = self._select_source(target)
        if not eligible:
            return input_tuple
        src_idx = random.choice(eligible)

        W, H = image.size
        img_np = np.array(image)

        masks = target["masks"]
        masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)
        src_visib = masks_np[src_idx].astype(np.uint8)
        src_pixels = img_np.copy()
        src_pixels[src_visib == 0] = 0

        full_masks = target.get("full_masks")
        fm_np = full_masks.numpy() if torch.is_tensor(full_masks) else np.array(full_masks)
        src_full = fm_np[src_idx].astype(np.uint8) if full_masks is not None else src_visib.copy()

        poses_data = target["poses"].data if hasattr(target["poses"], 'data') else target["poses"]
        src_t = poses_data[src_idx, :3].numpy().astype(np.float32)
        src_R = poses_data[src_idx, 3:].numpy().reshape(3, 3).astype(np.float32)
        src_label = target["labels"][src_idx:src_idx+1].clone()

        pxa_all = target.get("px_count_all")
        src_pxa = float(pxa_all[src_idx]) if pxa_all is not None else float(src_visib.sum())

        cam_K = target.get("cam_K")
        if cam_K is not None:
            cam_K_np = cam_K[0].numpy() if cam_K.dim() > 1 else cam_K.numpy()
            # image-plane 회전·zoom 중심은 현재 image (W, H) 좌표계의 진짜 주점이어야 함.
            # 원본 cam_K(=orig_size 기준)에 image scale ratio를 곱해서 cx, cy로만 사용.
            # src_cam_K는 원본 그대로 보관 — pose 라벨/criterion/추론 좌표계는 원본 K 유지.
            # FILLSINGLE_DISABLE_K_CORRECT=1 환경변수로 보정 비활성화 (디버그/비교용)
            if os.environ.get("FILLSINGLE_DISABLE_K_CORRECT") == "1":
                sx, sy = 1.0, 1.0
            else:
                orig_size = target.get("orig_size")
                if orig_size is not None:
                    sx = W / float(orig_size[0])
                    sy = H / float(orig_size[1])
                else:
                    sx, sy = 1.0, 1.0
            cx = float(cam_K_np[2]) * sx
            cy = float(cam_K_np[5]) * sy
            src_cam_K = cam_K[src_idx:src_idx+1].clone() if cam_K.dim() > 1 else cam_K.unsqueeze(0).clone()
        else:
            cx, cy = W / 2.0, H / 2.0
            src_cam_K = None

        boxes = target.get("boxes")
        src_box = boxes.data[src_idx:src_idx+1].clone() if boxes is not None else None

        n_target = random.randint(self.min_total, self.max_total)

        # Phase 0: 소스 물체 자체를 첫 번째 후보로 확보 (실패 방지)
        base_candidate = (float(src_visib.sum()) / max(src_pxa, 1),
                          src_visib, 0.0, 1.0,
                          np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))

        # Phase 1: visib_mask만 warp → 가시성 체크 (cheap)
        light_candidates = []
        for _ in range(self.max_attempts):
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            if random.random() < self.p_rot180:
                angle += 180.0
            scale = random.uniform(self.zoom_min, self.zoom_max)

            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            new_visib = cv2.warpAffine(src_visib, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

            new_pxa = src_pxa * scale * scale
            if new_pxa > 0 and new_visib.sum() / new_pxa < self.min_visibility:
                continue

            vis_ratio = float(new_visib.sum()) / max(new_pxa, 1)
            light_candidates.append((vis_ratio, new_visib, angle, scale, M))

        # Phase 2: greedy 비겹침 선택 (mask boolean만)
        light_candidates.append(base_candidate)
        light_candidates.sort(key=lambda x: -x[0])
        chosen = []
        occupancy = np.zeros((H, W), dtype=bool)
        for vis_ratio, new_visib, angle, scale, M in light_candidates:
            if len(chosen) >= n_target:
                break
            mask = new_visib > 0
            visible = float(mask.sum()) - float((mask & occupancy).sum())
            new_pxa = src_pxa * scale * scale
            if new_pxa > 0 and visible / new_pxa < self.min_visibility:
                continue
            chosen.append((new_visib, angle, scale, M))
            occupancy |= mask

        # Phase 3: 선택된 것만 pixels + full_mask warp + pose 계산 (expensive)
        candidates = []
        for new_visib, angle, scale, M in chosen:
            new_pixels = cv2.warpAffine(src_pixels, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            new_full = cv2.warpAffine(src_full, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            new_pxa = src_pxa * scale * scale

            angle_rad = np.deg2rad(angle)
            c, s = np.cos(-angle_rad), np.sin(-angle_rad)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            t_new = Rz @ src_t
            t_new[2] /= scale
            R_new = Rz @ src_R

            ys, xs = np.nonzero(new_full > 0)
            if len(xs) == 0:
                continue
            new_box = torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)

            candidates.append({
                "pixels": new_pixels,
                "visib_mask": new_visib,
                "full_mask": new_full,
                "px_count_all": new_pxa,
                "pose": torch.from_numpy(np.concatenate([t_new, R_new.reshape(9)])).float(),
                "tz": float(t_new[2]),
                "label": src_label.clone(),
                "box": new_box,
                "cam_K": src_cam_K.clone() if src_cam_K is not None else None,
            })

        if not candidates:
            return input_tuple

        candidates.sort(key=lambda x: -x["tz"])

        background = np.zeros((H, W, 3), dtype=np.uint8)
        canvas, occ_map = _depth_composite(candidates, background, H, W, self.edge_blur_kernel)
        keep_indices, final_masks = _visibility_filter(candidates, occ_map, self.min_visibility)

        if not keep_indices:
            return input_tuple

        merged = _build_target_from_objects(candidates, keep_indices, final_masks, H, W)
        if merged is None:
            return input_tuple

        for k in target:
            if k not in merged:
                merged[k] = target[k]

        return (PILImage.fromarray(canvas), merged, dataset)
@register()
class FilterSmallBoxLowVis(nn.Module):
    """Modal bbox 최소 크기 + visibility 필터 (BOP visib_fract 기준).

    `target['visibility']`를 그대로 사용. 기하 증강(PoseAugmentation 등)이
    각자 visibility를 재계산해주는 책임을 가지므로 filter는 단순히 임계값
    비교만 수행. coco_dataset.py가 어노테이션의 BOP visib_fract를 그대로
    `target['visibility']`에 저장 → BOP eval과 정합.

    visibility가 target에 없으면 1.0으로 가정 (필터 안 함).
    """

    def __init__(
        self,
        min_size: int = 5,
        min_visib: float = 0.0,
        **deprecated_kwargs,
    ):
        super().__init__()
        self.min_size = int(min_size)
        self.min_visib = float(min_visib)
        if deprecated_kwargs:
            ignored = ", ".join(deprecated_kwargs.keys())
            print(
                f"[FilterSmallBoxLowVis] deprecated kwargs ignored: {ignored}. "
                "Filter now relies on target['visibility'] (updated by aug ops)."
            )

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
        if masks_np.ndim != 3 or masks_np.shape[0] == 0:
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)
        N, H, W = masks_np.shape

        # Vectorized bbox-size check on visible mask.
        y_any = masks_np.any(axis=2)
        x_any = masks_np.any(axis=1)
        has_any = y_any.any(axis=1)
        x_min = x_any.argmax(axis=1)
        x_max = (W - 1) - x_any[:, ::-1].argmax(axis=1)
        y_min = y_any.argmax(axis=1)
        y_max = (H - 1) - y_any[:, ::-1].argmax(axis=1)
        widths = np.where(has_any, x_max - x_min, 0)
        heights = np.where(has_any, y_max - y_min, 0)
        size_ok = (widths >= self.min_size) & (heights >= self.min_size) & has_any

        # Visibility check (target['visibility'] 직접 사용; 기하 증강이 갱신).
        if self.min_visib > 0:
            v = target.get("visibility")
            if v is None:
                visib_ok = np.ones(N, dtype=bool)
            else:
                arr = v.numpy() if torch.is_tensor(v) else np.asarray(v)
                visib = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
                visib_ok = visib >= self.min_visib
        else:
            visib_ok = np.ones(N, dtype=bool)

        keep = torch.tensor(size_ok & visib_ok, dtype=torch.bool)

        if keep.all():
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        # 모두 fail이면 target 유지 (criterion이 비어있는 상태 잘 처리하도록).
        if not keep.any():
            if dataset_info is not None:
                return (image, target, dataset_info)
            return (image, target)

        per_object_keys = {
            "boxes", "masks", "full_masks", "amodal_masks",
            "labels", "poses", "cam_K", "visibility",
            "area", "iscrowd", "px_count_all",
        }
        filtered = {}
        for k, v in target.items():
            if (
                k in per_object_keys
                and isinstance(v, (torch.Tensor, BoundingBoxes, Mask, Pose))
                and v.shape[0] == len(keep)
            ):
                if isinstance(v, BoundingBoxes):
                    filtered[k] = BoundingBoxes(
                        v.data[keep], format=v.format, canvas_size=v.canvas_size
                    )
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


# ============================================================================
# CopyPasteSingleClass — cross-image same-class CopyPaste with VOC background
# ============================================================================
# 1) 현재 image A에서 한 class 선택 (visibility >= min_visibility 필터 통과한
#    instance 중 1개 = source).
# 2) 다른 image들에서 동일 class instance candidate K개 수집.
# 3) candidate를 visibility 내림차순으로 정렬한 뒤 source 및 이미 선택된
#    instance의 visib_mask와 겹침이 적은 순서로 source 외 2~4개 greedy 선택.
# 4) source + 추가 N개를 VOC 배경 위에 depth-ordered composite. 각 instance의
#    pose/cam_K는 원본 그대로 (회전·zoom 변환 없음 — cross-image라 자연 viewpoint).
# 5) 결과: 동일 class 3~5개 image. val(test_eval) single-class scene 분포 매칭.

@register()
class CopyPasteSingleClass(nn.Module):
    """Cross-image same-class CopyPaste with VOC background.

    Args:
        voc_root: VOC2012 JPEGImages (또는 임의 background) 디렉토리.
            None이면 random 단색 fallback.
        p: 적용 확률.
        min_extra, max_extra: source 외 추가할 instance 수 (기본 2~4 → 총 3~5).
        candidate_pool_size: 다른 image에서 미리 모을 same-class 후보 수.
        min_visibility: source/candidate 필터 + composite 후 visibility 필터.
        max_attempts: candidate 수집 시 최대 image sampling 시도.
        edge_blur_kernel: composite 엣지 블렌딩 커널 (0=hard edge).
    """

    def __init__(
        self,
        voc_root: str = None,
        p: float = 0.5,
        min_extra: int = 2,
        max_extra: int = 4,
        candidate_pool_size: int = 16,
        min_visibility: float = 0.1,
        max_bbox_iou: float = 1.0,    # bbox IoU 임계값. >이면 reject. 1.0=disabled
        max_attempts: int = 30,
        edge_blur_kernel: int = 0,
        cc_textures_dir: str = None,
        cc_industrial_only: bool = True,
        cc_bg_prob: float = 1.0,
        cc_cache_dir: str = None,
    ):
        super().__init__()
        self.voc_root = os.path.expanduser(voc_root) if voc_root else None
        self.p = float(p)
        self.min_extra = int(min_extra)
        self.max_extra = int(max_extra)
        self.candidate_pool_size = int(candidate_pool_size)
        self.min_visibility = float(min_visibility)
        self.max_bbox_iou = float(max_bbox_iou)
        self.max_attempts = int(max_attempts)
        self.edge_blur_kernel = int(edge_blur_kernel)
        self._voc_paths = None
        self.cc_textures_dir = cc_textures_dir
        self.cc_industrial_only = bool(cc_industrial_only)
        self.cc_bg_prob = float(cc_bg_prob)
        self._cc_paths = None
        # mmap cache
        self.cc_cache_dir = os.path.expanduser(cc_cache_dir) if cc_cache_dir else None
        self._cc_cache_color = None
        self._cc_cache_normal = None
        self._cc_cache_rough = None
        self._cc_cache_size = 0

    def _voc_list(self):
        if self._voc_paths is None:
            self._voc_paths = (
                _load_background_list(self.voc_root) if self.voc_root else []
            )
        return self._voc_paths

    def _cc_texture_list(self):
        if self._cc_paths is None:
            if not self.cc_textures_dir:
                self._cc_paths = []
                return self._cc_paths
            root = os.path.expanduser(self.cc_textures_dir)
            if not os.path.isdir(root):
                self._cc_paths = []
                return self._cc_paths
            industrial_prefixes = (
                'Metal', 'DiamondPlate', 'SheetMetal', 'CorrugatedSteel',
                'MetalPlates', 'MetalWalkway', 'PaintedMetal',
                'Concrete', 'ConcreteB', 'ConcreteC', 'ConcreteD',
                'Rust', 'Tiles', 'RoofingTiles',
            )
            all_dirs = sorted(os.listdir(root))
            if self.cc_industrial_only:
                keep = []
                for d in all_dirs:
                    stem = ''.join(c for c in d if not c.isdigit())
                    if stem in industrial_prefixes:
                        keep.append(d)
                paths = [os.path.join(root, d) for d in keep]
            else:
                paths = [os.path.join(root, d) for d in all_dirs
                         if os.path.isdir(os.path.join(root, d))]
            valid = []
            for p in paths:
                if not os.path.isdir(p):
                    continue
                color_files = [f for f in os.listdir(p) if f.endswith('_Color.jpg')]
                if color_files:
                    valid.append(p)
            self._cc_paths = valid
        return self._cc_paths

    def _cc_cache_load(self) -> bool:
        """Lazy-load mmap cache. Returns True if cache is available."""
        if self._cc_cache_color is not None:
            return True
        if not self.cc_cache_dir:
            return False
        color_path = os.path.join(self.cc_cache_dir, 'cc_cache_color.npy')
        if not os.path.exists(color_path):
            return False
        try:
            self._cc_cache_color = np.load(color_path, mmap_mode='r')
            self._cc_cache_normal = np.load(
                os.path.join(self.cc_cache_dir, 'cc_cache_normal.npy'), mmap_mode='r'
            )
            self._cc_cache_rough = np.load(
                os.path.join(self.cc_cache_dir, 'cc_cache_rough.npy'), mmap_mode='r'
            )
            self._cc_cache_size = self._cc_cache_color.shape[0]
            return True
        except Exception:
            self._cc_cache_color = None
            return False

    def _synth_cc_lite_bg(self, H, W):
        """V_cc_lite: PBR-lite shaded cc_texture + sigma=1 blur + brightness shift.

        Uses mmap cache (cc_cache_dir) when available for near-zero JPEG decode
        overhead. Falls back to per-call JPEG loading when cache is absent.
        """
        import random as _rng
        from scipy.ndimage import gaussian_filter

        if self._cc_cache_load():
            # --- Fast path: mmap cache ---
            tex_id = _rng.randrange(self._cc_cache_size)
            color_full = self._cc_cache_color[tex_id]   # (2048, 2048) uint8
            normal_full = self._cc_cache_normal[tex_id]  # (2048, 2048, 3) uint8
            rough_full = self._cc_cache_rough[tex_id]    # (2048, 2048) uint8

            src_h, src_w = color_full.shape
            if src_w > W and src_h > H:
                x = _rng.randint(0, src_w - W)
                y = _rng.randint(0, src_h - H)
                # np.asarray copies the mmap slice into a contiguous array
                color_arr = np.asarray(color_full[y:y + H, x:x + W]).astype(np.float32)
                normal_crop = np.asarray(normal_full[y:y + H, x:x + W])
                rough_arr = np.asarray(rough_full[y:y + H, x:x + W]).astype(np.float32) / 255.0
            else:
                # Resize fallback (rare: only if cache was built at smaller size)
                color_arr = np.asarray(
                    PILImage.fromarray(np.asarray(color_full)).resize((W, H), PILImage.BILINEAR)
                ).astype(np.float32)
                normal_crop = np.asarray(
                    PILImage.fromarray(np.asarray(normal_full)).resize((W, H), PILImage.BILINEAR)
                )
                rough_arr = np.asarray(
                    PILImage.fromarray(np.asarray(rough_full)).resize((W, H), PILImage.BILINEAR)
                ).astype(np.float32) / 255.0

            n = (normal_crop.astype(np.float32) / 127.5) - 1.0
            n_norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
            n = n / n_norm
            rough = rough_arr

        else:
            # --- Slow path: per-call JPEG decode (backward compat) ---
            paths = self._cc_texture_list()
            if not paths:
                return self._synth_gray_bg(H, W)

            tex_dir = _rng.choice(paths)
            files = os.listdir(tex_dir)
            color_file = next((f for f in files if f.endswith('_Color.jpg')), None)
            if not color_file:
                return self._synth_gray_bg(H, W)

            color_img = PILImage.open(os.path.join(tex_dir, color_file)).convert('L')

            normal_file = next((f for f in files if 'NormalGL' in f), None)
            normal_img = (
                PILImage.open(os.path.join(tex_dir, normal_file)).convert('RGB')
                if normal_file else None
            )

            rough_file = next((f for f in files if 'Roughness' in f), None)
            rough_img = (
                PILImage.open(os.path.join(tex_dir, rough_file)).convert('L')
                if rough_file else None
            )

            src_w, src_h = color_img.size
            if src_w > W and src_h > H:
                x = _rng.randint(0, src_w - W)
                y = _rng.randint(0, src_h - H)
                color_img = color_img.crop((x, y, x + W, y + H))
                if normal_img:
                    normal_img = normal_img.crop((x, y, x + W, y + H))
                if rough_img:
                    rough_img = rough_img.crop((x, y, x + W, y + H))
            else:
                color_img = color_img.resize((W, H), PILImage.BILINEAR)
                if normal_img:
                    normal_img = normal_img.resize((W, H), PILImage.BILINEAR)
                if rough_img:
                    rough_img = rough_img.resize((W, H), PILImage.BILINEAR)

            color_arr = np.array(color_img).astype(np.float32)

            if normal_img:
                n = (np.array(normal_img).astype(np.float32) / 127.5) - 1.0
                n_norm = np.linalg.norm(n, axis=2, keepdims=True) + 1e-6
                n = n / n_norm
            else:
                n = np.zeros((H, W, 3), dtype=np.float32)
                n[..., 2] = 1.0

            if rough_img:
                rough = np.array(rough_img).astype(np.float32) / 255.0
            else:
                rough = np.full((H, W), 0.5, dtype=np.float32)

        # --- Shared PBR-lite shading (identical for both paths) ---
        light_dir = np.array([
            _rng.uniform(-0.5, 0.5),
            _rng.uniform(-0.5, 0.5),
            _rng.uniform(0.5, 1.0),
        ], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)

        diffuse = np.clip(np.dot(n, light_dir), 0, 1)
        spec_strength = 1.0 - rough
        specular = spec_strength * np.power(diffuse, 16)

        ambient = _rng.uniform(0.15, 0.35)
        diff_w = _rng.uniform(0.5, 0.7)
        spec_w = _rng.uniform(0.1, 0.3)

        color_n = color_arr / 255.0
        shaded = (ambient + diff_w * diffuse) * color_n + spec_w * specular
        shaded = np.clip(shaded * 255, 0, 255)

        shaded = gaussian_filter(shaded, sigma=1.0)

        target_mean = _rng.uniform(30, 50)
        shift = target_mean - shaded.mean()
        shaded = np.clip(shaded + shift, 5, 90)

        bg = np.stack([shaded, shaded, shaded], axis=-1).astype(np.uint8)
        return bg

    @staticmethod
    def _synth_gray_bg(H, W):
        """ITODD test_real-style BG with controlled diversity.

        Stats: mean ~40 (TEST matched), range 3-83 (slightly wider than TEST).
          - 90% grayscale (R=G=B), 10% per-channel colored
          - base brightness: uniform [10, 70] (test_real mean 40 위주)
          - gradient mode (random):
              50% linear (random 2D direction, ±20)
              30% radial (random center, distance-based, ±25)
              20% spotlight (random gaussian peak, ±30)
          - per-pixel Gaussian noise: N(0, 12) — test_real std matched
        """
        import random as _rng
        # Base color
        if _rng.random() < 0.1:
            base = np.random.uniform(10, 70, size=3).astype(np.float32)   # colored
        else:
            base = np.full(3, np.random.uniform(10, 70), dtype=np.float32)  # grayscale
        bg = np.full((H, W, 3), base, dtype=np.float32)
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij"
        )
        # Gradient mode
        mode = _rng.random()
        if mode < 0.5:
            # Linear: random direction
            ax = float(np.random.uniform(-20, 20))
            ay = float(np.random.uniform(-20, 20))
            bg += (yy * ay + xx * ax)[..., None]
        elif mode < 0.8:
            # Radial: random center, distance-based (center-out or vignette)
            cy = float(np.random.uniform(-0.5, 0.5))
            cx = float(np.random.uniform(-0.5, 0.5))
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            amp = float(np.random.uniform(-25, 25))
            bg += (r * amp)[..., None]
        else:
            # Spotlight: random position gaussian peak
            cy = float(np.random.uniform(-0.7, 0.7))
            cx = float(np.random.uniform(-0.7, 0.7))
            sigma = float(np.random.uniform(0.3, 0.8))
            spot = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            amp = float(np.random.uniform(-30, 30))
            bg += (spot * amp)[..., None]
        # Per-pixel noise
        bg += np.random.normal(0, 12, size=(H, W, 3))
        return np.clip(bg, 0, 255).astype(np.uint8)

    def forward(self, *inputs):
        import random
        sample = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(sample, (list, tuple)) or len(sample) < 3:
            return sample
        image, target, dataset = sample[0], sample[1], sample[2]

        if random.random() > self.p:
            return sample
        if not isinstance(image, PILImage.Image):
            return sample
        if "masks" not in target or "labels" not in target or "poses" not in target:
            return sample

        masks = target["masks"]
        masks_np = masks.numpy() if torch.is_tensor(masks) else np.array(masks)
        labels = target["labels"]
        pxa_all = target.get("px_count_all")

        # (1) source 후보 (visibility 통과)
        eligible = []
        for i in range(len(masks_np)):
            visib_area = float(masks_np[i].sum())
            pxa = float(pxa_all[i]) if pxa_all is not None else visib_area
            if pxa > 0 and visib_area / pxa >= self.min_visibility:
                eligible.append(i)
        if not eligible:
            return sample

        src_idx = random.choice(eligible)
        src_class = int(labels[src_idx].item())

        source = _extract_object_at(image, target, src_idx)
        if source is None:
            return sample

        W, H = image.size

        # (2) 다른 image에서 same-class candidate 수집
        # dataset.class_to_image_idx cache가 있으면 random retry 없이 O(1) lookup,
        # 없으면 fallback으로 random idx + retry. label→cat_id 변환 후 lookup.
        cache = getattr(dataset, "class_to_image_idx", None)
        if cache is not None and getattr(dataset, "mscoco_label2category", None):
            src_cat_id = dataset.mscoco_label2category.get(src_class, src_class)
        else:
            src_cat_id = src_class
        pool = cache.get(src_cat_id) if cache else None

        # cache 기반 lookup일 때만 load_item light-mode 사용 가능
        # (cache의 image는 같은 class ann이 보장되어 light-mode가 거의 즉시 매치).
        use_light_mode = pool is not None and cache is not None

        candidates = []
        attempts = 0
        while (
            len(candidates) < self.candidate_pool_size
            and attempts < self.max_attempts
        ):
            attempts += 1
            try:
                if pool:
                    other_idx = random.choice(pool)
                else:
                    other_idx = random.randint(0, len(dataset) - 1)
                if use_light_mode:
                    other_img, other_target = dataset.load_item(
                        other_idx,
                        target_class_label=src_class,
                        min_visibility=self.min_visibility,
                        draft_size=image.size,   # JPEG draft (raw 1280×960 → image.size)
                    )
                else:
                    other_img, other_target = dataset.load_item(
                        other_idx, draft_size=image.size,
                    )
            except Exception:
                continue
            if other_img is None or other_target is None:
                continue
            if not isinstance(other_img, PILImage.Image):
                continue
            # candidate가 raw size (1280x960) 이면 현재 image (post-Resize) 크기로 맞춤.
            # pose/cam_K는 metric/original frame이라 그대로, image-plane만 비례 scale.
            if other_img.size != image.size:
                other_img, other_target = _resize_sample(
                    other_img, other_target, image.size
                )
            o_labels = other_target.get("labels")
            o_masks = other_target.get("masks")
            if o_labels is None or o_masks is None:
                continue
            o_masks_np = (
                o_masks.numpy() if torch.is_tensor(o_masks) else np.array(o_masks)
            )
            o_pxa = other_target.get("px_count_all")

            if use_light_mode:
                # light-mode: target이 이미 src_class 1개 instance만 가짐
                same_idx_list = list(range(len(o_labels)))
            else:
                same_idx_list = [
                    i for i in range(len(o_labels))
                    if int(o_labels[i].item()) == src_class
                ]
                if not same_idx_list:
                    continue

            for i in same_idx_list:
                vis_area = float(o_masks_np[i].sum())
                pxa = float(o_pxa[i]) if o_pxa is not None else vis_area
                if pxa <= 0:
                    continue
                visib = vis_area / pxa
                if visib < self.min_visibility:
                    continue
                obj = _extract_object_at(other_img, other_target, i)
                if obj is None:
                    continue
                obj["_visib"] = visib
                candidates.append(obj)
                if len(candidates) >= self.candidate_pool_size:
                    break

        if not candidates:
            return sample

        # (3) visibility ↓ 정렬 → greedy 비겹침 (source 외 2~4)
        candidates.sort(key=lambda x: -x.get("_visib", 0.0))
        n_extra = random.randint(self.min_extra, self.max_extra)

        chosen = [source]
        occupancy = (source["visib_mask"] > 0).astype(bool).copy()

        def _bbox_iou(b1, b2):
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
            inter = iw * ih
            a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
            a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
            return inter / max(a1 + a2 - inter, 1e-6)

        for cand in candidates:
            if len(chosen) - 1 >= n_extra:
                break
            mask = cand["visib_mask"].astype(bool)
            visible = float((mask & ~occupancy).sum())
            cand_pxa = float(cand.get("px_count_all", float(mask.sum())))
            if cand_pxa <= 0:
                continue
            if visible / cand_pxa < self.min_visibility:
                continue
            # bbox IoU filter (test_eval은 거의 isolated; cluttered overlap 학습 분포 외)
            if self.max_bbox_iou < 1.0 and cand.get("box") is not None:
                cb = cand["box"]
                cb_arr = cb.numpy()[0] if hasattr(cb, "numpy") else np.asarray(cb)[0]
                rejected = False
                for c2 in chosen:
                    if c2.get("box") is None:
                        continue
                    eb = c2["box"]
                    eb_arr = eb.numpy()[0] if hasattr(eb, "numpy") else np.asarray(eb)[0]
                    if _bbox_iou(cb_arr, eb_arr) > self.max_bbox_iou:
                        rejected = True
                        break
                if rejected:
                    continue
            chosen.append(cand)
            occupancy |= mask

        if len(chosen) <= 1:
            return sample

        # (4) depth-ordered composite. voc_root에 이미지 있으면 VOC 사용,
        # cc_textures_dir이 있으면 V_cc_lite PBR-lite 배경 사용,
        # 없으면 test_real-style 단색 회색 + 노이즈 + gradient (학습 distribution
        # 을 test 회색 배경에 맞춤; FP background-as-object 케이스 완화).
        chosen.sort(key=lambda x: -x["tz"])
        voc_paths = self._voc_list()
        cc_paths = self._cc_texture_list()
        if voc_paths:
            background = _load_background_image(voc_paths, H, W)
        elif cc_paths and random.random() < self.cc_bg_prob:
            background = self._synth_cc_lite_bg(H, W)
        else:
            background = self._synth_gray_bg(H, W)

        canvas, occ_map = _depth_composite(
            chosen, background, H, W, self.edge_blur_kernel
        )
        keep_indices, final_masks = _visibility_filter(
            chosen, occ_map, self.min_visibility
        )
        if not keep_indices:
            return sample

        merged = _build_target_from_objects(chosen, keep_indices, final_masks, H, W)
        if merged is None:
            return sample

        for k in target:
            if k not in merged:
                merged[k] = target[k]

        return (PILImage.fromarray(canvas), merged, dataset)


# ============================================================================
# GDRN/yolox-style imgaug photometric augmentation
# ============================================================================
# Mirrors `gdrnpp_bop2022/det/yolox/data/datasets/mosaicdetection.py::_get_color_augmentor`
# (aug_type="code") so the COLOR_AUG_CODE string from gdrnpp configs can be
# evaluated as-is. Operates on PIL RGB images and is opt-in via config:
#     - {type: GDRNPhotoAug, p: 0.8}
#     - {type: GDRNGrayscale, p: 0.5, alpha_range: [0.0, 1.0]}
# Other RACE6D photometric ops (RandomGrayscale, ColorJitter, etc.) are
# untouched, so existing per-dataset configs keep working.

# gdrnpp/yolox itodd `COLOR_AUG_CODE` 와 골격 동일하나 4개 enhance op의 극단
# 강도(Sharpness 50배, Contrast 50배 / 0.2배, Brightness 6배 / 0.1배, Color 20배)를
# 좁힌 ITODD-tuned default. 학습 시 일부 sample이 saturate되어 물체 식별 불가가
# 되는 현상을 방지. wide variation은 유지. Invert는 그대로 둠. 원본 gdrn 강도가
# 필요하면 GDRNPhotoAug(code=...) 로 직접 지정 가능.
_GDRN_DEFAULT_AUG_CODE = (
    "Sequential(["
    "Sometimes(0.5, CoarseDropout(p=0.2, size_percent=0.05)),"
    "Sometimes(0.4, GaussianBlur((0., 3.))),"
    "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 10.))),"
    "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.7, 10.))),"
    "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.7, 4.0))),"
    "Sometimes(0.3, pillike.EnhanceColor(factor=(0.3, 5.))),"
    "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
    "Sometimes(0.3, Invert(0.2, per_channel=True)),"
    "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
    "Sometimes(0.5, Multiply((0.6, 1.4))),"
    "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
    "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
    "], random_order=True)"
)


def _build_imgaug_from_code(code: str):
    """eval() helper that exposes the same imgaug namespace as gdrnpp."""
    import imgaug.augmenters as iaa  # noqa: F401
    from imgaug.augmenters import (  # noqa: F401
        Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
        Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
        Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
        GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
        DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
        MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
        Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast,
    )
    try:
        from imgaug.augmenters import Canny  # noqa: F401
    except Exception:
        pass
    return eval(code)


@register()
class GDRNPhotoAug(nn.Module):
    """yolox/gdrn `COLOR_AUG_CODE` photometric block, applied via imgaug.

    Mirrors `aug_wrapper` in gdrnpp_bop2022 (yolox detection): the inner
    Sequential is wrapped by an outer Bernoulli with probability `p`
    (= COLOR_AUG_PROB, default 0.8). The inner block contains its own
    Sometimes(...) gates exactly as in the upstream config.

    Args:
        p: outer apply probability (matches COLOR_AUG_PROB).
        code: imgaug code string. Default = yolox itodd block (no Grayscale).
    """

    def __init__(self, p: float = 0.8, code: str = None):
        super().__init__()
        self.p = float(p)
        self.code = code if code is not None else _GDRN_DEFAULT_AUG_CODE
        self._augmentor = None

    def _ensure_built(self):
        if self._augmentor is None:
            self._augmentor = _build_imgaug_from_code(self.code)
        return self._augmentor

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(sample, (list, tuple)) or len(sample) < 1:
            return sample
        image = sample[0]
        if not isinstance(image, PILImage.Image):
            return sample
        if torch.rand(1).item() >= self.p:
            return sample
        aug = self._ensure_built()
        img_np = np.array(image)
        img_aug = aug.augment_image(img_np)
        new_image = PILImage.fromarray(img_aug)
        return (new_image,) + tuple(sample[1:])


@register()
class GDRNGrayscale(nn.Module):
    """imgaug Grayscale matching gdrnpp pose configs:
        Sometimes(p, Grayscale(alpha=alpha_range))
    Default p=0.5, alpha_range=(0.0, 1.0) — partial grayscale blending,
    same as `gdrn/itodd_pbr/...itodd.py` line 31.
    """

    def __init__(self, p: float = 0.5, alpha_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.p = float(p)
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self._aug = None

    def _ensure_built(self):
        if self._aug is None:
            from imgaug.augmenters import Grayscale
            self._aug = Grayscale(alpha=self.alpha_range)
        return self._aug

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(sample, (list, tuple)) or len(sample) < 1:
            return sample
        image = sample[0]
        if not isinstance(image, PILImage.Image):
            return sample
        if torch.rand(1).item() >= self.p:
            return sample
        aug = self._ensure_built()
        img_np = np.array(image)
        img_aug = aug.augment_image(img_np)
        new_image = PILImage.fromarray(img_aug)
        return (new_image,) + tuple(sample[1:])
