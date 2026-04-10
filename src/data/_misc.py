"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
from torch import Tensor

if importlib.metadata.version('torchvision') == '0.15.1':
    import torchvision
    torchvision.disable_beta_transforms_warning()
    
    import torch
    from typing import Optional, Union, Any
    from torchvision.datapoints._datapoint import Datapoint
    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    import torch
    from typing import Optional, Union, Any, Dict, List
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video, TVTensor)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')

class Pose(TVTensor):
    """TVTensor subclass for 3D pose data: (N, 9) or (N, 12) or (N, 6) [R | t]"""
    @staticmethod
    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> "Pose":
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        if requires_grad:
            tensor.requires_grad_(True)
        if tensor.dim() != 2 or tensor.shape[1] not in (6, 9, 12):
            raise ValueError(f"Expected tensor with shape (N, 9) or (N, 12), got {tensor.shape}")
        return tensor.as_subclass(cls)
        
    @classmethod
    def wrap_like(cls, other: "Pose", tensor: Tensor) -> "Pose":
        return tensor.as_subclass(cls)
    
class Keypoints(TVTensor):
    """TVTensor subclass for 2D keypoints: (N, 64) = 32 keypoints (x, y)"""
    @staticmethod
    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> "Keypoints":
        tensor = torch.as_tensor(data, dtype=dtype, device=device)
        if requires_grad:
            tensor.requires_grad_(True)
        if tensor.dim() != 2 or tensor.shape[1] != 64:
            raise ValueError(f"Expected tensor with shape (N, 64), got {tensor.shape}")
        return tensor.as_subclass(cls)
        
    @classmethod
    def wrap_like(cls, other: "Keypoints", tensor: Tensor) -> "Keypoints":
        return tensor.as_subclass(cls)

def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', canvas_size=None) -> Tensor:
    """
    Convert plain tensor to appropriate tv_tensor-style object
    """
    assert key in ('boxes', 'masks', 'poses', 'keypoints'), \
        "Only support 'boxes', 'masks', 'poses', 'keypoints'"

    if key == 'boxes':
        box_format_enum = getattr(BoundingBoxFormat, box_format.upper())
        kwargs = dict(zip(_boxes_keys, [box_format_enum, canvas_size]))
        return BoundingBoxes(tensor, **kwargs)

    if key == 'masks':
        return Mask(tensor)

    elif key == 'poses':
        return Pose(tensor)
    
    elif key == 'keypoints':
        return Keypoints(tensor)