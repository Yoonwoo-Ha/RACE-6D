"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
-----------------------------------------------------------------------
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
-----------------------------------------------------------------------
Copyright(c) 2026 Yoonwoo-Ha. All Rights Reserved.
"""

import torch

import torchvision

torchvision.disable_beta_transforms_warning()

from PIL import Image
from pycocotools import mask as coco_mask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor, Pose
from ...core import register

import numpy as np
import cv2
import copy
import os
import yaml
import json

__all__ = ["CocoDetection"]


@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = [
        "transforms",
    ]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        return_full_masks=True,  # geometric aug 없으면 False로 두면 amodal polygon decode skip
        remap_mscoco_category=False,
        return_depth=False,
        category_file=None,
        coco_path=None,
        need_aligned=False,
        depth_scale=10000,       # legacy, unused after the mm-based normalization
        depth_z_max_mm=2000.0,   # depth clip 상한 (mm). 이 값을 1.0으로 정규화
    ):
        self.img_folder = os.path.expanduser(img_folder)
        self.ann_file = os.path.expanduser(ann_file)
        super(CocoDetection, self).__init__(self.img_folder, self.ann_file)
        self._transforms = transforms
        self.return_masks = return_masks
        self.return_full_masks = return_full_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.return_depth = return_depth
        self.depth_scale = depth_scale
        self.depth_z_max_mm = float(depth_z_max_mm)
        self.mscoco_category2name = None
        self.coco_path = os.path.expanduser(coco_path)
        self.need_aligned = need_aligned

        # Paligned 값 정의 (YCB-Video 전용)
        # 원본 category_id 기준 (remap 전)
        self.paligned_values = {
            19: [10.4796698, -5.41739619, -1.23077576],  # 원본 category_id
            20: [-8.82785585, -10.93032056, 0.09932552],  # 원본 category_id
        }

        # Load category mapping from YAML
        if category_file is not None and os.path.exists(category_file):
            with open(category_file, "r") as f:
                category_config = yaml.safe_load(f)
                self.mscoco_category2name = category_config["category2name"]

        # Generate mapping dictionaries
        if self.mscoco_category2name is not None:
            self.mscoco_category2label = {
                k: i for i, k in enumerate(self.mscoco_category2name.keys())
            }
            self.mscoco_label2category = {
                v: k for k, v in self.mscoco_category2label.items()
            }
        else:
            self.mscoco_category2label = {}
            self.mscoco_label2category = {}

        self.prepare = ConvertCocoPolysToMask(return_masks, return_full_masks)

        # BOP depth_scale: annotation 내 depth_scale 필드에서 읽기
        # PBR=0.1 (px*0.1=mm), Real=1.0 (px*1.0=mm)
        self._bop_depth_scale = 1.0
        if self.return_depth:
            ann_ids = self.coco.getAnnIds()
            if ann_ids:
                first_ann = self.coco.loadAnns(ann_ids[:1])[0]
                self._bop_depth_scale = first_ann.get("depth_scale", 1.0)

        # 유효한 이미지 필터링 (Paligned 변환 후)
        self._filter_valid_images()

        # category_id -> [valid dataset_idx, ...] cache.
        # _filter_valid_images 직후에 build해서 cache가 valid image만 포함하도록.
        # ann-level filter (ignore=True / iscrowd=1)도 반영.
        self._build_class_to_image_idx_cache()

        # img_folder 기반으로 depth_root 자동 결정
        self.depth_root = None
        if self.return_depth:
            root = os.path.basename(os.path.normpath(self.img_folder))
            parent = os.path.dirname(os.path.normpath(self.img_folder))
            if root.endswith("2017"):
                self.depth_root = os.path.join(parent, root + "_depth")
            else:
                self.depth_root = self.img_folder + "_depth"

    def __getitem__(self, idx):
        img, target = self.load_item(idx)

        depth_path = None
        if self.return_depth:
            rel = self.coco.loadImgs(self.ids[idx])[0]["file_name"]
            depth_path = self._depth_path_from_rel(rel)

        if self.return_depth:
            target["_depth_path"] = depth_path
            target["_depth_scale"] = self.depth_scale

        # Transform 파이프라인 (PIL Image 상태)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)

        # RGBD model input (return_depth=True인 경우만)
        if self.return_depth:
            # DepthAugment 등 transform이 미리 로드+증강한 경우 그 결과를 우선 사용
            if "_depth_tensor" in target:
                depth = target["_depth_tensor"]
            elif "_depth_path" in target:
                depth = self._load_depth_tensor(
                    target["_depth_path"], target.get("_depth_scale", self.depth_scale)
                )
            else:
                depth = None

            target.pop("_depth_path", None)
            target.pop("_depth_scale", None)
            target.pop("_depth_tensor", None)

            if depth is not None:
                # depth는 원본 해상도로 로드되므로 transform된 img 크기에 맞춰 resize.
                # nearest mode로 경계에서 가짜 depth 값이 생기지 않도록 함.
                if depth.shape[-2:] != img.shape[-2:]:
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0),  # [1, 1, H, W]
                        size=img.shape[-2:],
                        mode="nearest",
                    ).squeeze(0)  # [1, H, W]
                img = torch.cat([img, depth], dim=0)  # [4, H, W]

        # _depth_tensor 정리 (RendererAugmentation 등 transform 내부에서 생성될 수 있음)
        target.pop("_depth_tensor", None)

        return img, target

    def _apply_paligned_transform(self, poses, class_ids):
        """
        클래스 ID가 19, 20 (원본 category_id)인 객체에 대해 Paligned 변환 적용

        Args:
            poses: pose array
            class_ids: category_id array
        """
        if not self.need_aligned or class_ids is None:
            return poses

        paligned_dict = self.paligned_values

        if isinstance(poses, torch.Tensor):
            poses = poses.reshape(-1, 12).clone()
            device = poses.device

            for i, class_id in enumerate(class_ids):
                class_id_val = int(
                    class_id.item() if isinstance(class_id, torch.Tensor) else class_id
                )

                if class_id_val in paligned_dict:
                    tgt = poses[i, :3]
                    R = poses[i, 3:12].reshape(3, 3)

                    Paligned = torch.tensor(
                        paligned_dict[class_id_val], device=device, dtype=torch.float32
                    )

                    tgt_new = tgt + torch.matmul(R, Paligned)
                    poses[i, :3] = tgt_new

        elif isinstance(poses, np.ndarray):
            poses = poses.reshape(-1, 12).copy()

            for i, class_id in enumerate(class_ids):
                class_id_val = int(
                    class_id.item() if hasattr(class_id, "item") else class_id
                )

                if class_id_val in paligned_dict:
                    tgt = poses[i, :3]
                    R = poses[i, 3:12].reshape(3, 3)

                    Paligned = np.array(paligned_dict[class_id_val], dtype=np.float32)
                    tgt_new = tgt + np.dot(R, Paligned)
                    poses[i, :3] = tgt_new

        return poses

    def _filter_valid_images(self):
        """
        Paligned 변환 후 필터링(ignore, bbox 크기, tz>0)을 모두 통과하는
        annotation이 하나라도 있는 이미지만 유지
        """
        valid_ids = []

        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if not ann_ids:
                continue

            anns = self.coco.loadAnns(ann_ids)

            # 이미지 크기 가져오기
            img_info = self.coco.loadImgs(img_id)[0]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # 1. ignore=True / iscrowd=1 ann은 학습/평가 대상 아님 (BOP convention).
            #    이런 ann은 보통 객체가 카메라에 비현실적으로 가까워 (tz < diameter/2)
            #    Z<0 점이 발생하는 합성 잡음. 데이터 진입 단에서 미리 제외.
            #    visibility<0.1 필터는 FilterSmallBoxLowVis transform이 담당.
            valid_anns = [
                ann for ann in anns
                if not ann.get("ignore", False) and ann.get("iscrowd", 0) == 0
            ]

            if len(valid_anns) == 0:
                continue

            # 2. Paligned 변환 적용 (원본 category_id 사용)
            # 주의: 이 변환은 필터링 목적이며, 원본 COCO annotation은 수정하지 않음
            if self.need_aligned:
                poses = np.array(
                    [ann.get("pose", [0] * 12) for ann in valid_anns]
                ).reshape(-1, 12)
                class_ids = np.array([ann.get("category_id", 0) for ann in valid_anns])
                poses_transformed = self._apply_paligned_transform(poses, class_ids)
            else:
                poses_transformed = np.array(
                    [ann.get("pose", [0] * 12) for ann in valid_anns]
                ).reshape(-1, 12)

            # 3. tz > 0 체크 (visibility 필터 제거)
            final_valid_anns = []
            for idx, ann in enumerate(valid_anns):
                # tz > 0 체크
                pose = poses_transformed[idx]
                if len(pose) < 12:
                    continue

                tz = pose[2]
                eps = 1e-6
                if tz <= eps or not np.isfinite(tz):
                    continue

                final_valid_anns.append(ann)

            if len(final_valid_anns) > 0:
                valid_ids.append(img_id)

        original_count = len(self.ids)
        self.ids = valid_ids
        filtered_count = original_count - len(self.ids)

        print(f"Filtered out {filtered_count} images with no valid annotations")
        print(f"  (after ignore/bbox/coordinate filters with Paligned transform)")
        print(f"Remaining images: {len(self.ids)}")

    def _build_class_to_image_idx_cache(self):
        """category_id -> [valid dataset_idx, ...] 매핑 cache.

        _filter_valid_images 와 동일한 ann-level filter (ignore=False, iscrowd=0,
        tz>0) 를 그대로 적용. 즉 cache 안의 image_idx는 해당 cat_id의 유효한
        ann 이 최소 1개 보장. CopyPasteSingleClass 같은 same-class fetch 가
        random retry 없이 O(1) lookup으로 동작하기 위함.
        """
        img_id_to_idx = {iid: i for i, iid in enumerate(self.ids)}
        cat_to_idx = {cat_id: [] for cat_id in self.coco.getCatIds()}

        for cat_id in list(cat_to_idx.keys()):
            for iid in self.coco.getImgIds(catIds=[cat_id]):
                if iid not in img_id_to_idx:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=iid, catIds=[cat_id])
                for ann in self.coco.loadAnns(ann_ids):
                    if ann.get("ignore", False) or ann.get("iscrowd", 0) != 0:
                        continue
                    pose = ann.get("pose", [0] * 12)
                    if len(pose) < 12:
                        continue
                    tz = pose[2]
                    if tz <= 1e-6 or not np.isfinite(tz):
                        continue
                    cat_to_idx[cat_id].append(img_id_to_idx[iid])
                    break

        self.class_to_image_idx = {k: v for k, v in cat_to_idx.items() if v}
        total_entries = sum(len(v) for v in self.class_to_image_idx.values())
        print(
            f"class_to_image_idx cache: {len(self.class_to_image_idx)} classes, "
            f"{total_entries} entries"
        )

    def _safe_deep_copy(self, obj):
        if isinstance(obj, Pose):
            return Pose(obj.clone())
        elif isinstance(obj, torch.Tensor):
            return obj.clone()
        elif isinstance(obj, dict):
            return {k: self._safe_deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._safe_deep_copy(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self._safe_deep_copy(x) for x in obj)
        else:
            try:
                return copy.deepcopy(obj)
            except:
                return obj

    def load_item(self, idx, target_class_label=None, min_visibility=0.0, draft_size=None):
        """Load (image, target) for `idx`.

        Args:
            target_class_label: light-mode (single ann of given class).
            draft_size: (W, H) — JPEG draft mode 사용. PIL이 reduced size로 빠르게 decode.
                JPEG에서만 효과 있고, PNG는 no-op (PIL이 무시). bbox/area는 비례 scale.
                cam_K, segmentation은 그대로 (Resize transform이 cam_K를 처리 안 하는
                정상 path 동작과 동등; segmentation은 RLE size mismatch 시
                convert_coco_poly_to_mask가 자동 align).
        """
        image_id = self.ids[idx]
        if draft_size is not None:
            # Custom load with draft (super().__getitem__ 우회)
            from PIL import Image as PILImage
            file_name = self.coco.loadImgs(image_id)[0]['file_name']
            path = os.path.join(self.img_folder, file_name)
            image = PILImage.open(path)
            raw_W, raw_H = image.size
            image.draft('RGB', draft_size)
            image = image.convert('RGB')
            new_W, new_H = image.size
            sx = new_W / raw_W
            sy = new_H / raw_H
            # Annotation: bbox/area만 scale (raw → image scale).
            # cam_K, segmentation, pose 등은 그대로 (정상 path와 동일).
            raw_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
            if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
                anns = []
                for a in raw_anns:
                    ac = dict(a)
                    if 'bbox' in a and a['bbox']:
                        x, y, w, h = a['bbox']
                        ac['bbox'] = [x * sx, y * sy, w * sx, h * sy]
                    if 'area' in a:
                        ac['area'] = a['area'] * sx * sy
                    anns.append(ac)
            else:
                anns = raw_anns
        else:
            image, anns = super(CocoDetection, self).__getitem__(idx)

        if target_class_label is not None:
            selected = None
            for a in anns:
                if a.get("ignore", False) or a.get("iscrowd", 0) != 0:
                    continue
                cat_id = a.get("category_id", 0)
                if self.remap_mscoco_category and self.mscoco_category2label:
                    label = self.mscoco_category2label.get(cat_id, cat_id)
                else:
                    label = cat_id
                if label != target_class_label:
                    continue
                if a.get("visibility", 1.0) < min_visibility:
                    continue
                selected = a
                break
            if selected is None:
                return None, None
            anns = [selected]

        target = {"image_id": image_id, "annotations": anns}

        # cam_K, need_aligned, paligned_values를 ConvertCocoPolysToMask에 전달
        if self.remap_mscoco_category:
            image, target = self.prepare(
                image,
                target,
                category2label=self.mscoco_category2label,
                need_aligned=self.need_aligned,
                paligned_values=self.paligned_values,
            )
        else:
            image, target = self.prepare(
                image,
                target,
                need_aligned=self.need_aligned,
                paligned_values=self.paligned_values,
            )

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(
                target["boxes"], key="boxes", canvas_size=image.size[::-1]
            )

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")
            if "full_masks" in target:
                target["full_masks"] = convert_to_tv_tensor(
                    target["full_masks"], key="masks"
                )

        if "poses" in target:
            target["poses"] = convert_to_tv_tensor(target["poses"], key="poses")

        return image, target

    def _load_depth_tensor(self, dpath, depth_scale=None):
        """BOP depth_scale을 사용해서 mm로 변환 후 [0, 1] 정규화.

        Flow: raw pixel --(× self._bop_depth_scale)--> mm
                       --(clip to [0, depth_z_max_mm])--> mm (clipped)
                       --(÷ depth_z_max_mm)--> [0, 1]

        - Train/test 모두 BOP ann의 `depth_scale` 필드(PBR=0.1, Real=1.0)를 사용해 실제 mm로 변환
        - Z_MAX_MM 상한으로 clip하여 PBR 원거리 배경(~6m)이 test sensor range 밖으로 흡수
        - 최종 output은 RGB channel과 동일한 [0, 1] range → 4ch concat 시 gradient 균형
        - hole (raw=0)은 정규화 후에도 0 유지 (sentinel)
        """
        if not os.path.exists(dpath):
            return None
        depth_pil = Image.open(dpath)
        if depth_pil.mode in ("I;16", "I;16B", "I;16L", "I"):
            d_raw = np.array(depth_pil, dtype=np.uint16).astype(np.float32)
        else:
            d_raw = np.array(depth_pil.convert("L"), dtype=np.uint8).astype(np.float32)

        # 1) BOP depth_scale로 실제 mm 변환
        d_mm = d_raw * float(self._bop_depth_scale)

        # 2) 물리적 상한으로 clip (배경 saturation 정렬)
        z_max = float(self.depth_z_max_mm)
        d_mm = np.clip(d_mm, 0.0, z_max)

        # 3) [0, 1] 정규화
        d_norm = d_mm / z_max

        # 4) hole 처리: raw가 0인 자리는 정규화 후에도 0 (sentinel)
        valid = np.isfinite(d_raw) & (d_raw > 0)
        d_norm = np.where(valid, d_norm, 0.0).astype(np.float32)

        return torch.from_numpy(d_norm).unsqueeze(0).float()

    def _read_depth_tensor(self, dpath, unit="auto", return_mask=False):
        if not os.path.exists(dpath):
            return (None, None) if return_mask else None

        pil = Image.open(dpath)
        # 16-bit 보존
        if pil.mode in ("I;16", "I;16B", "I;16L", "I"):
            d = np.array(pil, dtype=np.uint16).astype(np.float32)
        else:
            # 8-bit밖에 없을 때만 L 사용
            d = np.array(pil.convert("L"), dtype=np.uint8).astype(np.float32)

        # 단위 통일: depth_scale 사용 (config에서 설정)
        if unit == "mm":
            d = d / self.depth_scale
        elif unit == "auto" and d.max() > 100.0:
            d = d / self.depth_scale

        valid = np.isfinite(d) & (d > 0)
        d[~valid] = 0.0  # invalid를 0으로
        depth = torch.from_numpy(d).unsqueeze(0).float()  # [1,H,W]

        if return_mask:
            mask = torch.from_numpy(valid.astype(np.uint8)).unsqueeze(0)  # [1,H,W]
            return depth, mask
        return depth

    def _depth_path_from_rel(self, rel_path, depth_suffix=".png"):
        # coco file_name 보존, 확장자만 교체
        stem = os.path.splitext(rel_path)[0]
        return os.path.join(self.depth_root, stem + depth_suffix)

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        s += f" return_full_masks: {self.return_full_masks}\n"
        s += f" need_aligned: {self.need_aligned}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        if hasattr(self, "_preset") and self._preset is not None:
            s += f" preset:\n   {repr(self._preset)}"
        return s

    @property
    def categories(
        self,
    ):
        return self.coco.dataset["categories"]

    @property
    def category2name(
        self,
    ):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(
        self,
    ):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(
        self,
    ):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    """polygon → (h, w) mask. compressed RLE은 자체 size로 decode되므로
    image size와 다를 때(load_item draft_size 등) 자동으로 (h, w)로 resize.
    정상 path(image size = RLE size)에서는 분기 skip → no-op."""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # compressed RLE은 frPyObjects가 (h, w)를 무시하고 자체 size로 decode.
        # image와 mismatch 시 image size에 맞춰 resize (binary → INTER_NEAREST).
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            if mask.ndim == 2:
                mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_full_masks=True):
        self.return_masks = return_masks
        self.return_full_masks = return_full_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        # NOTE: ignore=True / iscrowd / visibility<0.1 필터는 제거됨.
        # 가려짐이 심한 GT도 training supervision과 val target에 포함되어,
        # (1) train supervision gap 해소 (confident-wrong-class 문제 완화)
        # (2) val target dict이 COCO eval GT(raw ann_file)와 일치하도록 함.
        # pose 유효성(tz>0)만 남겨서 물리적으로 불가능한 ann만 제외.

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 2. Paligned 변환 적용 (원본 category_id 사용)
        poses = [obj["pose"] for obj in anno]
        poses = torch.as_tensor(poses, dtype=torch.float32).reshape(-1, 12)

        # cam_K per annotation (RandomTransformAug 등에서 사용)
        cam_Ks = [obj["cam_K"] for obj in anno]
        cam_Ks = torch.as_tensor(cam_Ks, dtype=torch.float32).reshape(-1, 9)

        need_aligned = kwargs.get("need_aligned", False)
        paligned_values = kwargs.get("paligned_values", None)

        if need_aligned and paligned_values is not None:
            category_ids = [obj["category_id"] for obj in anno]
            poses = self._apply_paligned_transform(poses, category_ids, paligned_values)

        # 3. category2label 매핑
        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

            # full_masks: amodal silhouette. PoseAugmentation 등 geometric aug에서
            # warp 후 amodal bbox 재계산용. 비활성화 가능 (return_full_masks=False)
            # — geometric aug 없는 config는 polygon decode skip → I/O 단축.
            full_masks = None
            if (
                self.return_full_masks
                and len(anno) > 0
                and ("full_masks" in anno[0] or "full_segmentation" in anno[0])
            ):
                full_segmentations = [
                    obj.get("full_masks", obj.get("full_segmentation", obj["segmentation"]))
                    for obj in anno
                ]
                full_masks = convert_coco_poly_to_mask(full_segmentations, h, w)

        # 4. visibility 처리 + tz > 0 체크
        # annotation에 visibility 있으면 그대로, 없으면 mask 기반으로 계산
        raw_visibility = []
        for i, obj in enumerate(anno):
            vis = obj.get("visibility", None)
            if vis is not None:
                raw_visibility.append(float(vis))
            else:
                if self.return_masks and full_masks is not None:
                    vis_area = float(masks[i].sum())
                    amodal_area = float(full_masks[i].sum())
                    vis = vis_area / max(amodal_area, 1.0)
                else:
                    vis = 1.0
                raw_visibility.append(vis)
        visibility = torch.tensor(raw_visibility, dtype=torch.float32)

        # px_count_all: BOP 3× canvas full silhouette 픽셀 수 복원.
        # visib_fract = px_count_visib / px_count_all 이므로
        #   px_count_all = visib_mask_area / visib_fract
        # 이 값은 truncation+occlusion 모두 깎인 분모. 기하 증강 (ZoomPose 등)
        # 적용 시 area scale factor (z²)로 갱신하면 post-aug에서도 BOP 정의에
        # 맞는 visibility 재계산 가능. visibility=0/missing은 fallback으로
        # visib_area를 그대로 (filter가 0/0=0 처리해 자연 drop).
        if self.return_masks:
            visib_areas = masks.reshape(masks.shape[0], -1).sum(dim=1).float()
        else:
            bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            visib_areas = (bw * bh).float()
        safe_vis = visibility.clamp(min=1e-6)
        px_count_all = torch.where(
            visibility > 1e-6,
            visib_areas / safe_vis,
            visib_areas,
        )

        # ignore=True / iscrowd=1 ann 제외 (BOP convention: 평가 대상 아님 +
        # 보통 tz<diameter/2 인 합성 잡음 → projection에서 Z<0 발생).
        not_ignore = torch.tensor(
            [not obj.get("ignore", False) for obj in anno], dtype=torch.bool
        )
        not_crowd = torch.tensor(
            [obj.get("iscrowd", 0) == 0 for obj in anno], dtype=torch.bool
        )
        eps = 1e-6
        keep = (
            (poses[:, 2] > eps) & torch.isfinite(poses[:, 2]) & not_ignore & not_crowd
        )

        boxes = boxes[keep]
        poses = poses[keep]
        labels = labels[keep]
        cam_Ks = cam_Ks[keep]
        if self.return_masks:
            masks = masks[keep]
            if full_masks is not None:
                full_masks = full_masks[keep]

        target = {}
        target["boxes"] = boxes
        target["poses"] = poses
        target["labels"] = labels
        target["cam_K"] = cam_Ks
        target["visibility"] = visibility[keep]
        target["px_count_all"] = px_count_all[keep]
        if self.return_masks:
            target["masks"] = masks
            if full_masks is not None:
                target["full_masks"] = full_masks
        target["image_id"] = image_id

        # for conversion to coco api - keep 필터링 적용
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(w), int(h)])

        return image, target

    def _apply_paligned_transform(self, poses, category_ids, paligned_values):
        """
        Paligned 변환 적용 (원본 category_id 기준: 19, 20)

        Args:
            poses: torch.Tensor [N, 12]
            category_ids: list of category_id (원본)
            paligned_values: dict {category_id: [dx, dy, dz]}
        """
        poses = poses.clone()

        for i, cat_id in enumerate(category_ids):
            if cat_id in paligned_values:
                tgt = poses[i, :3]
                R = poses[i, 3:12].reshape(3, 3)

                Paligned = torch.tensor(
                    paligned_values[cat_id], device=poses.device, dtype=torch.float32
                )

                tgt_new = tgt + torch.matmul(R, Paligned)
                poses[i, :3] = tgt_new

        return poses


