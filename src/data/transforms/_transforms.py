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
    """Per-augmentation probability settings (GDRNPP style).

    Based on GDRNPP BOP2022 configuration:
    - base_probability = 0.8
    - effective apply probability = base_probability * adjustment
    """

    def __init__(self):
        # GDRNPP Sometimes() probability values used as-is (multiplied by base 0.8)
        self.brightness = 0.5  # Sometimes(0.5, EnhanceBrightness)
        self.contrast = 0.3  # Sometimes(0.3, EnhanceContrast)
        self.linear_contrast = 0.5  # Sometimes(0.5, LinearContrast)
        self.color_jitter = 0.3  # Sometimes(0.3, EnhanceColor) - saturation
        self.hsv_adjust = 0.3  # additional HSV adjustment
        self.sharpen = 0.3  # Sometimes(0.3, EnhanceSharpness)
        self.motion_blur = 0.3  # motion blur
        self.gaussian_blur = 0.4  # Sometimes(0.4, GaussianBlur)
        self.gaussian_noise = 0.1  # Sometimes(0.1, AdditiveGaussianNoise)
        self.additional_noise = 0.1  # additional noise
        self.grayscale = 0.5  # Sometimes(0.5, Grayscale)
        self.random_background = 0.4  # separate fixed value (ignores base probability)
        self.coarse_dropout = 0.5  # Sometimes(0.5, CoarseDropout)
        self.rotate_expand = 0.3  # rotate expand
        self.object_occlusion = 0.3  # object occlusion
        self.add_value = 0.5  # Sometimes(0.5, Add)
        self.multiply = 0.5  # Sometimes(0.5, Multiply)


class AugmentationManager:
    def __init__(self, base_probability: float = 0.8):
        """
        Args:
            base_probability: base apply probability (default: 0.8)
        """
        self.base_probability = max(0.1, min(1.0, base_probability))
        self.probs = AugmentationProbabilities()

    def get_probability(self, transform_name: str) -> float:
        """Compute the effective apply probability for a given transform."""
        import os
        if os.environ.get("AUG_FORCE_P_ONE", "0") == "1":
            return 1.0
        if hasattr(self.probs, transform_name):
            adjustment = getattr(self.probs, transform_name)

            # Background replacement uses its own fixed value
            if transform_name == "random_background":
                return adjustment

            # All others: apply adjustment to base probability
            final_prob = self.base_probability * adjustment
            return max(0.0, min(1.0, final_prob))

        # Undefined transforms fall back to base probability
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
    Convert pose to normalized coordinates (cam_K fetched from target['cam_K']).
    Uses bbox-relative coordinate encoding.
    """

    def __init__(self, normalize=False, coco_path=None) -> None:
        super().__init__()
        self.normalize = normalize
        # coco_path is accepted for config backward compatibility but is not used.
        # cam_K is read from target['cam_K'].

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        if isinstance(inputs, tuple):
            image, target = inputs[0], inputs[1]
            dataset_info = inputs[2] if len(inputs) > 2 else None
        else:
            target, image, dataset_info = inputs, None, None

        # Check bounding box information
        if "boxes" not in target:
            raise ValueError(
                "Bounding box information is required for ConvertPose transform"
            )

        # Check pose information
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

        # Coordinate conversion (normalization)
        converted_poses = self._convert_poses(poses, boxes, cam_K)

        # Build new target
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
        """Convert poses to bbox-relative coordinates."""
        if isinstance(poses, torch.Tensor):
            device = poses.device
            # cam_K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            fx = cam_K[0]
            fy = cam_K[4]
            px = cam_K[2]
            py = cam_K[5]

            poses = poses.reshape(-1, 12)

            # Handle bbox data correctly
            if isinstance(bbox_data, torch.Tensor):
                bbox_tensor = bbox_data.to(device).reshape(-1, 4)
            else:
                bbox_tensor = torch.tensor(bbox_data, device=device).reshape(-1, 4)

            # Extract translation components
            tx = poses[:, 0]  # in mm
            ty = poses[:, 1]  # in mm
            tz = poses[:, 2]  # in mm

            # Calculate bounding box size
            # bbox format: [x1, y1, x2, y2] (unnormalized pixel coordinates)
            x1 = bbox_tensor[:, 0]
            y1 = bbox_tensor[:, 1]
            x2 = bbox_tensor[:, 2]
            y2 = bbox_tensor[:, 3]
            wbbox = x2 - x1
            hbbox = y2 - y1
            cxbbox = (x1 + x2) / 2
            cybbox = (y1 + y2) / 2

            # Calculate relative translation coordinates
            rx = (px + (fx * tx / tz) - cxbbox) / wbbox  # normalized coordinate
            ry = (py + (fy * ty / tz) - cybbox) / hbbox  # normalized coordinate
            rz = tz / 1000  # normalized depth in meters

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

            # Handle bbox data correctly
            if isinstance(bbox_data, np.ndarray):
                bbox_array = bbox_data.reshape(-1, 4)
            elif isinstance(bbox_data, torch.Tensor):
                bbox_array = bbox_data.numpy().reshape(-1, 4)
            else:
                bbox_array = np.array(bbox_data).reshape(-1, 4)

            # Extract translation components
            tx = poses[:, 0]  # in mm
            ty = poses[:, 1]  # in mm
            tz = poses[:, 2]  # in mm

            # Calculate bounding box size
            # bbox format: [x1, y1, x2, y2] (unnormalized pixel coordinates)
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
            rz = tz / 1000  # normalized depth in meters

            # Rotation (9D flattened 3x3)
            new_rot = poses[:, 3:12]

            new_tran = np.stack([rx, ry, rz], axis=1)
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
    """GDRNPP-style color/saturation adjustment (maps to EnhanceColor).

    GDRNPP: Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.)))
    factor 0 = grayscale, 1 = original, >1 = increased saturation
    """

    def __init__(self, saturation=(0.5, 3.0), hue=0):
        # GDRNPP: pillike.EnhanceColor(factor=(0., 20.))
        # factor 0 = grayscale, 1 = original, >1 = increased saturation (hue=0: no hue adjustment in GDRNPP)
        super().__init__(brightness=0, contrast=0, saturation=saturation, hue=hue)
        self.aug_manager = AugmentationManager()
        self.p = self.aug_manager.get_probability("color_jitter")

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


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
    """GDRNPP-style sharpness adjustment.

    GDRNPP: Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.)))
    factor 0 = blurry, 1 = original, >1 = sharp
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
    """GDRNPP-style Gaussian blur.

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
class RandomCoarseDropout(T.Transform):
    """Implements CoarseDropout identical to imgaug - generates a mask at low resolution then upsamples."""

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

        # drop_prob: per-pixel dropout probability
        if isinstance(drop_prob, (int, float)):
            self.p_range = (drop_prob, drop_prob)
        else:
            self.p_range = drop_prob

        # size_percent: controls low-resolution mask size
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

        # Get image dimensions
        img = flat_inputs[0]
        if isinstance(img, PILImage.Image):
            img_width, img_height = img.size
            channels = 3  # assume RGB
        elif isinstance(img, Image):
            channels, img_height, img_width = img.shape
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        # Sample dropout probability
        p_dropout = float(torch.empty(1).uniform_(self.p_range[0], self.p_range[1]))

        # Sample low-resolution size (key to the imgaug approach)
        size_percent = float(
            torch.empty(1).uniform_(
                self.size_percent_range[0], self.size_percent_range[1]
            )
        )

        # Compute low-resolution dimensions
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
        """Apply CoarseDropout in imgaug style - FromLowerResolution approach."""
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

        # Convert result to uint8
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

    Simulates real camera ISP processing absent from PBR rendering to reduce the sim-to-real gap.
    Based on the SurfEmb (CVPR 2022) implementation.
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
            scale: fixed noise standard deviation (used when scale_range is absent)
            scale_range: randomly sampled from [min, max] (takes priority over scale)
            per_channel: independent noise per channel
            p: apply probability override
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
    """Apply Gaussian blur to mask edges (for blending)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)


def _extract_objects(image, target):
    """Extract objects from the image using visib_mask.

    Returns:
        list of dict: per-object info (pixels, visib_mask, full_mask, px_count_all, tz, pose, label, box, cam_K)
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
    """Composite objects onto canvas in depth order by tz (objects must be sorted descending by tz).

    Returns:
        canvas: H x W x 3 composited image
        occupancy: H x W int array (index of the object owning each pixel, -1 = background)
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
    """Compute final visibility and filter objects below the threshold.

    Returns:
        keep_indices: list of indices of objects to keep
        final_masks: final visible mask for each kept object
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

        # Check minimum modal bbox size
        vys, vxs = np.where(final_mask > 0)
        if len(vys) == 0:
            continue
        if (vxs.max() - vxs.min()) < min_modal_size or (vys.max() - vys.min()) < min_modal_size:
            continue

        keep_indices.append(i)
        final_masks.append(final_mask)

    return keep_indices, final_masks


def _build_target_from_objects(objects, keep_indices, final_masks, H, W):
    """Build a target dict from the filtered set of objects."""
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


# ---------------------------------------------------------------------------
# ROI variants of the CopyPaste hot path. Used exclusively by
# CopyPasteSingleClass; the full-canvas helpers above remain in use by
# PoseAugmentation (instance mode), which mutates obj["pixels"] via
# cv2.warpAffine and therefore requires full H×W storage.
#
# Schema differences vs the full-canvas extractor:
#   - pixels_roi : (h, w, 3) uint8 — cropped to mask bbox, then mask-zeroed
#   - mask_roi   : (h, w)    uint8 — cropped visibility mask
#   - bbox_roi   : (y0, y1, x0, x1)
#   - visib_mask : (H, W)    uint8 — kept full-res for occupancy/IoU queries
#   - full_mask  : (H, W)    uint8 — kept full-res for downstream targets
# All other fields (tz, pose, label, box, cam_K, px_count_all) match
# _extract_object_at exactly, so _build_target_from_objects works unchanged.
# ---------------------------------------------------------------------------
def _extract_object_at_roi(image, target, idx):
    """ROI variant of `_extract_object_at` — pixels are stored only inside the
    object's tight bbox to avoid full H×W per-object copies in composite.

    Composite output is byte-identical to the full-canvas path.
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

    ys, xs = np.where(visib_mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    mask_roi = visib_mask[y0:y1, x0:x1].astype(np.uint8)

    img_np = np.array(image)
    pixels_roi = img_np[y0:y1, x0:x1].copy()
    pixels_roi[mask_roi == 0] = 0

    poses_data = poses.data if isinstance(poses, Pose) else poses
    box_data = boxes.data if hasattr(boxes, "data") else (
        boxes if boxes is not None else None
    )
    pose = poses_data[idx]
    tz = float(pose[2].item() if isinstance(pose[2], torch.Tensor) else pose[2])

    obj = {
        "pixels_roi": pixels_roi,
        "mask_roi": mask_roi,
        "bbox_roi": (y0, y1, x0, x1),
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


def _depth_composite_roi(objects, background, H, W, edge_blur_kernel=0):
    """ROI variant of `_depth_composite`. Touches canvas only inside each
    object's bbox_roi. Byte-identical to the full-canvas path (verified by
    scripts/bench_copypaste_roi.py for edge_blur in {0, 5}).
    """
    canvas = background.copy()
    occupancy = np.full((H, W), -1, dtype=np.int32)
    for i, obj in enumerate(objects):
        y0, y1, x0, x1 = obj["bbox_roi"]
        m = obj["mask_roi"]
        px = obj["pixels_roi"]
        sub_canvas = canvas[y0:y1, x0:x1]
        if edge_blur_kernel > 0:
            alpha = _gaussian_blur_edge(m, edge_blur_kernel)
            alpha = np.clip(alpha, 0, 1)
            a3 = alpha[..., None]
            canvas[y0:y1, x0:x1] = (sub_canvas * (1 - a3) + px * a3).astype(np.uint8)
            occupancy[y0:y1, x0:x1][m > 0] = i
        else:
            sub_canvas[m > 0] = px[m > 0]
            occupancy[y0:y1, x0:x1][m > 0] = i
    return canvas, occupancy


def _visibility_filter_roi(objects, occupancy, H, W, min_visibility=0.1, min_modal_size=5):
    """ROI variant of `_visibility_filter`. Reads occupancy only inside each
    object's bbox_roi but returns full H×W final masks (so downstream
    `_build_target_from_objects` is unchanged).
    """
    keep_indices = []
    final_masks = []
    for i, obj in enumerate(objects):
        y0, y1, x0, x1 = obj["bbox_roi"]
        sub_fm = (occupancy[y0:y1, x0:x1] == i).astype(np.uint8)
        pxa = obj.get("px_count_all", 0)
        denom = pxa if pxa > 0 else obj["full_mask"].sum()
        if denom == 0:
            continue
        if sub_fm.sum() / denom < min_visibility:
            continue
        vys, vxs = np.where(sub_fm > 0)
        if len(vys) == 0:
            continue
        if (vxs.max() - vxs.min()) < min_modal_size or (vys.max() - vys.min()) < min_modal_size:
            continue
        full = np.zeros((H, W), dtype=np.uint8)
        full[y0:y1, x0:x1] = sub_fm
        keep_indices.append(i)
        final_masks.append(full)
    return keep_indices, final_masks


def _load_background_image(background_images, H, W):
    """Load a background image (from image list or random solid color)."""
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
    """Load a list of background image paths from a directory."""
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
    """Apply rotation + zoom around the principal point to one instance (in-place).

    cv2 CCW angle on y-down image → camera Z-axis rotation by -angle.
    zoom scale s → tz_new = tz / s (projected size enlarged by s).
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
        Applies the same (angle, scale) affine to all instances, centered on cam_K cx,cy.
        rotation_range=0 → zoom only / zoom_range=(1,1) → rotation only / both
        specified → combined transform.

    per_instance=True (instance mode)
        Samples an independent (angle, scale) per object → cv2 affine warp →
        depth-ordered composite. Objects occluded below revert_visibility are
        rolled back to their backup.

    Math (cv2 image y-down → BOP camera frame, x=right, y=down, z=forward):
        M       = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
        Rz_3d   = Rodrigues([0, 0, deg2rad(-angle_deg)])
        t_new   = Rz_3d @ t_old; t_new[2] /= scale
        R_new   = Rz_3d @ R_old
        cam_K   = unchanged (principal-point centered transform)
        px_all  = px_all * scale²

    full_masks REQUIREMENT
    ----------------------
    After geometric warp, amodal bboxes are recomputed as the nonzero tight box
    of warped `full_masks`. Targets without `full_masks` raise `RuntimeError`
    (fail-fast). Silent fallback to visib_mask is prohibited as it corrupts
    amodal training signal.

    Args
    ----
        per_instance     : False = scene-wide / True = per-object
        rotation_range   : ±deg uniform sample. 0 → rotation off
        zoom_range       : (min, max) uniform sample. (1.0, 1.0) → zoom off
        p                : apply probability
        background_dir   : directory of images used to fill blank areas after warp
                           (scene mode) / canvas (instance mode). None → solid fill_value
        fill_value       : solid-color padding when background_dir is absent
        revert_visibility: in instance mode, revert transform if post-warp visibility
                           falls below this threshold
        edge_blur_kernel : blending kernel for composite edges in instance mode (0=hard)
        p_rot180         : additional 180° flip probability
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
        """Fill blank areas after warp using a background_dir image or solid fill_value color."""
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
        """Warp a [N, H, W] uint8 mask stack with the same affine."""
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

        # 4) Recompute bboxes from warped full_masks (amodal)
        new_boxes = self._amodal_boxes_from_full(warped_full)
        new_target["boxes"] = BoundingBoxes(
            torch.from_numpy(new_boxes),
            format=BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        )

        # 5) px_count_all (preserved under rotation, scaled by zoom²) + visibility = visib/px_all
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

        # background = median color of non-object pixels (avoids solid-color patches)
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

        # Recompute amodal bbox from each kept object's warped full_mask
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


@register()
class FilterSmallBoxLowVis(nn.Module):
    """Filter by minimum modal bbox size and visibility (based on BOP visib_fract).

    Reads `target['visibility']` directly. Geometric augmentation ops (e.g.
    PoseAugmentation) are responsible for recomputing visibility; this filter
    simply compares against the threshold. coco_dataset.py stores the BOP
    visib_fract annotation directly in `target['visibility']` for BOP-eval
    consistency.

    If visibility is absent from target, assumes 1.0 (no filtering).
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

        # Visibility check (reads target['visibility'] directly; updated by geometric augmentation).
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

        # If all fail, keep original target (so criterion can handle empty batches gracefully).
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
    """Lightweight augmentation that replaces only the background.

    Applied only when neither RendererAugmentation nor RandomTransformAug has been applied.
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

        # Skip if geometric augmentation has already been applied
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

        # Union mask (pixels belonging to any instance)
        if masks_np.ndim == 3:
            combined = np.any(masks_np > 0, axis=0)
        else:
            combined = masks_np > 0

        # Load and resize background image
        bg_path = self._background_images[np.random.randint(len(self._background_images))]
        bg_img = PILImage.open(bg_path).convert("RGB")

        image_np = np.array(image)
        bg_np = np.array(bg_img.resize((image_np.shape[1], image_np.shape[0])))

        # Replace background: pixels outside the mask get the new background
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

    Based on analysis of TLESS test real sensor data:
    - Holes biased 3x toward object mask boundaries → clustered_hole_gen (edge_bias)
    - Holes also generated in mid-range scene regions (table/floor) → bg_clustered_hole_gen
    - Far background (near Z_MAX) excluded
    - No holes generated at image borders
    - Edge wobble effect → wavy_boundary_warp (coherent displacement)
    - Flying pixel effect → boundary_speckle (per-pixel neighbor swap)
    - Three GDRNPP ops retained (fill / gaussian noise)

    Input: [0, 1] normalized depth tensor returned by dataset's `_load_depth_tensor`.
    Output: augmented depth stored in target['_depth_tensor'] → used by __getitem__ for concat.

    All parameters are in [0, 1] scale (assumes Z_MAX_MM=2000).
    Example: fill_std=0.05 ≈ 100mm, noise_base_std=0.001 ≈ 2mm.
    """

    def __init__(
        self,
        # (1) Zero point fill (fills existing holes with small noise, GDRNPP)
        fill_prob: float = 1.0,
        fill_std: float = 0.05,
        # (2) Wavy boundary warp (mild + tight: dense waves)
        wavy_prob: float = 0.7,
        wavy_amplitude: float = 1.5,
        wavy_smoothness: float = 5.0,
        wavy_edge_decay: float = 4.0,
        wavy_image_edge_margin: int = 5,
        # (3) Boundary speckle (per-pixel swap, flying pixel)
        speckle_prob: float = 0.6,
        speckle_thickness: int = 1,
        speckle_swap_prob: float = 0.4,
        # (4) Object clustered holes (visibility-aware: measured from TLESS real sensor)
        # Lower visibility (heavily occluded) → more depth holes in real sensor
        # (Pearson corr = -0.59, based on 100-vs-100 analysis).
        obj_hole_prob: float = 0.7,
        obj_n_clusters_min: int = 1,
        obj_n_clusters_max: int = 5,
        obj_cluster_size_min: int = 5,
        obj_cluster_size_max: int = 200,
        obj_edge_bias: float = 0.7,
        obj_image_edge_margin: int = 5,
        # (5) Background clustered holes (scene region, excluding far background)
        bg_hole_prob: float = 0.7,
        bg_n_clusters_min: int = 1,
        bg_n_clusters_max: int = 4,
        bg_cluster_size_min: int = 50,
        bg_cluster_size_max: int = 800,
        bg_image_edge_margin: int = 10,
        bg_far_depth_threshold: float = 0.95,
        # (6) Gaussian noise (applied to all valid pixels)
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
        """Convert target['masks'] to a list of bool ndarrays at depth resolution."""
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

        Measured on 1000 val images:
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

        Based on real TLESS sensor measurements:
          - Low-visibility (heavily occluded) objects have many depth holes
          - High-visibility objects have mostly intact depth
        Uses target_hole_frac(visibility) to determine the target hole fraction
        per mask, then stamps that amount using cluster-based stamping.

        22% of vis < 0.2 cases are fully holed → short-circuit: set entire valid_mask to 0.
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
            # Full hole shortcut: zero out the entire visible mask
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

            # Budget-based stamping: keep generating clusters until cumulative holes reach target_px.
            # Cluster size stays within the original range (matches measured cluster shapes).
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

        # (1) Zero point fill — fill existing holes with N(0, fill_std)
        if rng.rand() < self.fill_prob:
            hole_idx = depth == 0
            if hole_idx.any():
                fill = rng.normal(0.0, self.fill_std, size=int(hole_idx.sum())).astype(np.float32)
                depth = depth.copy()
                depth[hole_idx] = fill

        # (2) Wavy boundary warp (only meaningful when masks are present)
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
# RendererAugmentation - PyTorch3D-based R|t augmentation
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
# 1) Select one class from the current image A (source = one instance that passes
#    the visibility >= min_visibility filter).
# 2) Collect K same-class instance candidates from other images.
# 3) Sort candidates by visibility descending, then greedily select 2-4 with least
#    overlap with the source and already-selected instances' visib_masks.
# 4) Composite source + N extras depth-ordered on a VOC background. Each instance's
#    pose/cam_K is kept as-is (no rotation/zoom — natural viewpoints across images).
# 5) Result: image with 3-5 instances of the same class, matching val/test_eval
#    single-class scene distribution.

@register()
class CopyPasteSingleClass(nn.Module):
    """Cross-image same-class CopyPaste with VOC background.

    Args:
        voc_root: VOC2012 JPEGImages (or arbitrary background) directory.
            None falls back to a random solid color.
        p: apply probability.
        min_extra, max_extra: number of instances to add beyond the source (default 2~4 → total 3~5).
        candidate_pool_size: number of same-class candidates to collect from other images.
        min_visibility: visibility filter for source/candidates and after compositing.
        max_attempts: maximum image sampling attempts during candidate collection.
        edge_blur_kernel: blending kernel for composite edges (0=hard edge).
    """

    def __init__(
        self,
        voc_root: str = None,
        p: float = 0.5,
        min_extra: int = 2,
        max_extra: int = 4,
        candidate_pool_size: int = 16,
        min_visibility: float = 0.1,
        max_bbox_iou: float = 1.0,    # bbox IoU threshold. >this rejects. 1.0=disabled
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

        shaded = cv2.GaussianBlur(np.ascontiguousarray(shaded, dtype=np.float32), (0, 0), sigmaX=1.0)

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
          - base brightness: uniform [10, 70] (centered around test_real mean 40)
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

        # (1) Source candidates (visibility filter)
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

        source = _extract_object_at_roi(image, target, src_idx)
        if source is None:
            return sample

        W, H = image.size

        # (2) Collect same-class candidates from other images.
        # If dataset.class_to_image_idx cache is available, use O(1) lookup without retries;
        # otherwise fall back to random idx + retry. Convert label → cat_id before lookup.
        cache = getattr(dataset, "class_to_image_idx", None)
        if cache is not None and getattr(dataset, "mscoco_label2category", None):
            src_cat_id = dataset.mscoco_label2category.get(src_class, src_class)
        else:
            src_cat_id = src_class
        pool = cache.get(src_cat_id) if cache else None

        # light-mode for load_item is only viable with cache-based lookup
        # (cache images are guaranteed to have the same class annotation → near-instant match).
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
            # If candidate is at raw size (1280x960), resize to current image (post-Resize) size.
            # pose/cam_K stays in metric/original frame; only image-plane fields are scaled.
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
                # light-mode: target already contains only one src_class instance
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
                obj = _extract_object_at_roi(other_img, other_target, i)
                if obj is None:
                    continue
                obj["_visib"] = visib
                candidates.append(obj)
                if len(candidates) >= self.candidate_pool_size:
                    break

        if not candidates:
            return sample

        # (3) Sort by visibility descending → greedy non-overlapping selection (2~4 beyond source)
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
            # bbox IoU filter (test_eval scenes are nearly isolated; cluttered overlap is out of training distribution)
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

        # (4) Depth-ordered composite. Use VOC if voc_root has images,
        # V_cc_lite PBR-lite background if cc_textures_dir is set,
        # otherwise test_real-style solid gray + noise + gradient (aligns training
        # distribution to test gray background; mitigates FP background-as-object cases).
        chosen.sort(key=lambda x: -x["tz"])
        voc_paths = self._voc_list()
        cc_paths = self._cc_texture_list()
        if voc_paths:
            background = _load_background_image(voc_paths, H, W)
        elif cc_paths and random.random() < self.cc_bg_prob:
            background = self._synth_cc_lite_bg(H, W)
        else:
            background = self._synth_gray_bg(H, W)

        canvas, occ_map = _depth_composite_roi(
            chosen, background, H, W, self.edge_blur_kernel
        )
        keep_indices, final_masks = _visibility_filter_roi(
            chosen, occ_map, H, W, self.min_visibility
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

# Structurally identical to gdrnpp/yolox itodd `COLOR_AUG_CODE`, but with the
# extreme intensities of the 4 enhance ops narrowed (Sharpness 50x, Contrast 50x/0.2x,
# Brightness 6x/0.1x, Color 20x) → ITODD-tuned default. Prevents some training
# samples from becoming so saturated that the object is unrecognizable, while
# keeping wide variation. Invert is left unchanged. Use GDRNPhotoAug(code=...)
# to specify the original gdrn intensities directly.
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


# =====================================================================
# Unused augmentations — kept for reference / future experiments only.
# Not referenced by any YAML config in configs/race6d/. Do not remove
# without checking configs and external usage.
# =====================================================================


# (unused) Pad image/target to a fixed spatial size (torchvision T.Pad subclass).
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



# (unused) Multiply pixel brightness by a random factor.
@register()
class RandomBrightness(T.Transform):
    """GDRNPP-style brightness adjustment.

    GDRNPP: Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.)))
    factor 0 = black, 1 = original, >1 = brighter
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



# (unused) Scale contrast around per-channel mean by a random factor.
@register()
class RandomContrast(T.Transform):
    """GDRNPP-style contrast adjustment.

    GDRNPP: Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.)))
    factor 0 = gray, 1 = original, >1 = increased contrast
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



# (unused) Linear contrast scaling with per-channel option (imgaug-style).
@register()
class RandomLinearContrast(T.Transform):
    """GDRNPP-style linear contrast adjustment.

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



# (unused) Blend RGB with grayscale by a random alpha.
@register()
class RandomGrayscale(T.Transform):
    """GDRNPP-style grayscale conversion.

    GDRNPP: Sometimes(0.5, Grayscale(alpha=(0.0, 1.0)))
    alpha 0 = original, 1 = fully grayscale, intermediate = blended
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



# (unused) Add a random offset to pixel values (additive jitter).
@register()
class RandomAdd(T.Transform):
    """GDRNPP-style additive brightness shift.

    GDRNPP: Sometimes(0.5, Add((-25, 25), per_channel=0.3))
    Adds the same value to all pixels.
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



# (unused) Multiply pixel values by a random factor (multiplicative jitter).
@register()
class RandomMultiply(T.Transform):
    """GDRNPP-style multiplicative contrast/brightness scaling.

    GDRNPP: Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5))
    Multiplies all pixels by the same factor.
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



