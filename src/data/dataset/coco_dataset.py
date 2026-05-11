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
        return_full_masks=True,  # set False to skip amodal polygon decode when no geometric aug is used
        remap_mscoco_category=False,
        return_depth=False,
        category_file=None,
        coco_path=None,
        need_aligned=False,
        depth_scale=10000,       # legacy, unused after the mm-based normalization
        depth_z_max_mm=2000.0,   # depth clip upper bound (mm); normalized to 1.0
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

        # Paligned value definitions (YCB-Video only)
        # Based on the original category_id (before remap)
        self.paligned_values = {
            19: [10.4796698, -5.41739619, -1.23077576],  # original category_id
            20: [-8.82785585, -10.93032056, 0.09932552],  # original category_id
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

        # BOP depth_scale: read from the depth_scale field in the annotation
        # PBR=0.1 (px*0.1=mm), Real=1.0 (px*1.0=mm)
        self._bop_depth_scale = 1.0
        if self.return_depth:
            ann_ids = self.coco.getAnnIds()
            if ann_ids:
                first_ann = self.coco.loadAnns(ann_ids[:1])[0]
                self._bop_depth_scale = first_ann.get("depth_scale", 1.0)

        # Filter valid images (after the Paligned transform)
        self._filter_valid_images()

        # category_id -> [valid dataset_idx, ...] cache.
        # Built right after _filter_valid_images so the cache contains valid images only.
        # Also reflects ann-level filters (ignore=True / iscrowd=1).
        self._build_class_to_image_idx_cache()

        # Auto-determine depth_root from img_folder
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

        # Transform pipeline (PIL Image state)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)

        # RGBD model input (only when return_depth=True)
        if self.return_depth:
            # If a transform such as DepthAugment already loaded and augmented depth, use that result first
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
                # depth is loaded at the original resolution, so resize to match the transformed img size.
                # Use nearest mode to avoid spurious depth values at boundaries.
                if depth.shape[-2:] != img.shape[-2:]:
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0),  # [1, 1, H, W]
                        size=img.shape[-2:],
                        mode="nearest",
                    ).squeeze(0)  # [1, H, W]
                img = torch.cat([img, depth], dim=0)  # [4, H, W]

        # Clean up _depth_tensor (may be created inside transforms such as RendererAugmentation)
        target.pop("_depth_tensor", None)

        return img, target

    def _apply_paligned_transform(self, poses, class_ids):
        """
        Apply the Paligned transform to objects with class ID 19 or 20 (original category_id).

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
        Keep only images that have at least one annotation passing all post-Paligned filters
        (ignore, bbox size, tz>0).
        """
        valid_ids = []

        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if not ann_ids:
                continue

            anns = self.coco.loadAnns(ann_ids)

            # Fetch image dimensions
            img_info = self.coco.loadImgs(img_id)[0]
            img_width = img_info["width"]
            img_height = img_info["height"]

            # 1. Annotations with ignore=True / iscrowd=1 are not training/eval targets (BOP convention).
            #    These usually occur when an object is unrealistically close to the camera (tz < diameter/2)
            #    and produce Z<0 points in synthetic noise. Exclude at the data entry stage.
            #    The visibility<0.1 filter is handled by the FilterSmallBoxLowVis transform.
            valid_anns = [
                ann for ann in anns
                if not ann.get("ignore", False) and ann.get("iscrowd", 0) == 0
            ]

            if len(valid_anns) == 0:
                continue

            # 2. Apply Paligned transform (using the original category_id)
            # Note: this transform is for filtering only; the original COCO annotations are not modified
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

            # 3. tz > 0 check (visibility filter removed)
            final_valid_anns = []
            for idx, ann in enumerate(valid_anns):
                # tz > 0 check
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
        """category_id -> [valid dataset_idx, ...] mapping cache.

        Applies the same ann-level filters as _filter_valid_images (ignore=False,
        iscrowd=0, tz>0). Each image_idx in the cache is guaranteed to have at least
        one valid annotation for that cat_id, enabling O(1) lookup for same-class
        fetches (e.g. CopyPasteSingleClass) without random retries.
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
            draft_size: (W, H) — use JPEG draft mode. PIL decodes quickly at reduced size.
                Only effective for JPEG; PNG is a no-op (PIL ignores it). bbox/area are scaled proportionally.
                cam_K and segmentation are left as-is (equivalent to the normal path where Resize does not
                touch cam_K; on RLE size mismatch, convert_coco_poly_to_mask auto-aligns the segmentation).
        """
        image_id = self.ids[idx]
        if draft_size is not None:
            # Custom load with draft (bypasses super().__getitem__)
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
            # Annotation: only scale bbox/area (raw -> image scale).
            # cam_K, segmentation, pose, etc. are left as-is (same as the normal path).
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

        # Pass cam_K, need_aligned, paligned_values to ConvertCocoPolysToMask
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
        """Convert to mm using the BOP depth_scale, then normalize to [0, 1].

        Flow: raw pixel --(x self._bop_depth_scale)--> mm
                       --(clip to [0, depth_z_max_mm])--> mm (clipped)
                       --(/ depth_z_max_mm)--> [0, 1]

        - Both train and test use the BOP annotation's `depth_scale` field (PBR=0.1, Real=1.0) to convert to actual mm.
        - Clip at Z_MAX_MM upper bound so PBR far backgrounds (~6m) are absorbed outside the test sensor range.
        - Final output is in the same [0, 1] range as RGB channels -> balanced gradients when concatenated as 4ch.
        - Holes (raw=0) remain 0 after normalization (sentinel).
        """
        if not os.path.exists(dpath):
            return None
        depth_pil = Image.open(dpath)
        if depth_pil.mode in ("I;16", "I;16B", "I;16L", "I"):
            d_raw = np.array(depth_pil, dtype=np.uint16).astype(np.float32)
        else:
            d_raw = np.array(depth_pil.convert("L"), dtype=np.uint8).astype(np.float32)

        # 1) Convert to actual mm via BOP depth_scale
        d_mm = d_raw * float(self._bop_depth_scale)

        # 2) Clip at the physical upper bound (align background saturation)
        z_max = float(self.depth_z_max_mm)
        d_mm = np.clip(d_mm, 0.0, z_max)

        # 3) Normalize to [0, 1]
        d_norm = d_mm / z_max

        # 4) Hole handling: pixels where raw==0 remain 0 after normalization (sentinel)
        valid = np.isfinite(d_raw) & (d_raw > 0)
        d_norm = np.where(valid, d_norm, 0.0).astype(np.float32)

        return torch.from_numpy(d_norm).unsqueeze(0).float()

    def _read_depth_tensor(self, dpath, unit="auto", return_mask=False):
        if not os.path.exists(dpath):
            return (None, None) if return_mask else None

        pil = Image.open(dpath)
        # Preserve 16-bit
        if pil.mode in ("I;16", "I;16B", "I;16L", "I"):
            d = np.array(pil, dtype=np.uint16).astype(np.float32)
        else:
            # Use L only when 8-bit is all that's available
            d = np.array(pil.convert("L"), dtype=np.uint8).astype(np.float32)

        # Unify units: use depth_scale (set in the config)
        if unit == "mm":
            d = d / self.depth_scale
        elif unit == "auto" and d.max() > 100.0:
            d = d / self.depth_scale

        valid = np.isfinite(d) & (d > 0)
        d[~valid] = 0.0  # set invalid pixels to 0
        depth = torch.from_numpy(d).unsqueeze(0).float()  # [1,H,W]

        if return_mask:
            mask = torch.from_numpy(valid.astype(np.uint8)).unsqueeze(0)  # [1,H,W]
            return depth, mask
        return depth

    def _depth_path_from_rel(self, rel_path, depth_suffix=".png"):
        # Preserve coco file_name, swap only the extension
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
    """polygon -> (h, w) mask. compressed RLE decodes at its own size, so when it
    differs from the image size (e.g. load_item draft_size) it is automatically
    resized to (h, w). On the normal path (image size == RLE size), the branch
    is skipped -> no-op."""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # compressed RLE: frPyObjects ignores (h, w) and decodes at its own size.
        # On image mismatch, resize to image size (binary -> INTER_NEAREST).
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

        # NOTE: ignore=True / iscrowd / visibility<0.1 filters are removed here.
        # Heavily occluded GTs are included in both training supervision and val targets so that:
        # (1) the training supervision gap is closed (mitigating the confident-wrong-class problem)
        # (2) the val target dict matches COCO eval GT (raw ann_file).
        # Only pose validity (tz>0) is kept, excluding physically impossible annotations.

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 2. Apply Paligned transform (using the original category_id)
        poses = [obj["pose"] for obj in anno]
        poses = torch.as_tensor(poses, dtype=torch.float32).reshape(-1, 12)

        # cam_K per annotation (used by RandomTransformAug, etc.)
        cam_Ks = [obj["cam_K"] for obj in anno]
        cam_Ks = torch.as_tensor(cam_Ks, dtype=torch.float32).reshape(-1, 9)

        need_aligned = kwargs.get("need_aligned", False)
        paligned_values = kwargs.get("paligned_values", None)

        if need_aligned and paligned_values is not None:
            category_ids = [obj["category_id"] for obj in anno]
            poses = self._apply_paligned_transform(poses, category_ids, paligned_values)

        # 3. category2label mapping
        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

            # full_masks: amodal silhouette. Used to recompute amodal bbox after
            # warping in geometric augs like PoseAugmentation. Can be disabled
            # (return_full_masks=False) — configs without geometric aug skip
            # polygon decode -> I/O savings.
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

        # 4. visibility handling + tz > 0 check
        # Use the annotation's visibility if present; otherwise compute from masks.
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

        # px_count_all: recover the BOP 3x canvas full-silhouette pixel count.
        # Since visib_fract = px_count_visib / px_count_all,
        #   px_count_all = visib_mask_area / visib_fract.
        # This is the denominator after both truncation and occlusion. Updating
        # with the area scale factor (z^2) under geometric augs (e.g. ZoomPose)
        # allows post-aug visibility to be recomputed under the BOP definition.
        # visibility=0/missing falls back to visib_area (the filter treats 0/0=0 and drops naturally).
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

        # Exclude ignore=True / iscrowd=1 annotations (BOP convention: not eval
        # targets + typically tz<diameter/2 synthetic noise -> Z<0 in projection).
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

        # for conversion to coco api - apply the keep filter
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
        Apply the Paligned transform (based on original category_id: 19, 20).

        Args:
            poses: torch.Tensor [N, 12]
            category_ids: list of original category_id
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


