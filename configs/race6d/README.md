# RACE-6D Configuration Reference

This directory contains all training, optimization, and augmentation configurations for RACE-6D. Configurations compose via `__include__` directives, and every component is wired through the `@register()` registry described in `src/core/workspace.py`.

```
configs/race6d/
├── include/                 # shared building blocks
│   ├── dataloader.yml       # train / val pipeline + augmentation policy
│   ├── optimizer.yml        # AdamW + flat-cosine LR + EMA
│   ├── race6d_r50vd_best.yml  # default model architecture
│   ├── *2coco_category.yml  # per-dataset COCO category remap
└── r50vd/                   # per-dataset configs (LM-O, YCB-V, T-LESS, TUD-L, HB, IC-BIN, ITODD)
```

The pipeline below reflects the **currently active augmentations** in `configs/race6d/include/dataloader.yml` and any per-dataset overrides in `r50vd/*.yml`. All augmentation classes are defined in `src/data/transforms/_transforms.py`.

> **Note**
> These settings are a **working baseline**, not a tuned optimum. Probability, strength range, ordering, and the set of augmentations themselves are tunable knobs. Combining and tuning them is expected to yield further accuracy gains.

## Photometric / color

| Transform | What it does |
|-----------|--------------|
| **ColorJitter** | GDRNPP-style saturation jitter (default range 0.5–3.0); optional hue shift. |
| **RandomHSVAdjust** | Joint random brightness (V) and saturation (S) scaling in HSV space. |
| **RandomSharpen** | Sharpness enhancement with a random factor in [0, 20] (factor < 1 blurs, > 1 sharpens). |
| **RandomBrightness**, **RandomContrast**, **RandomLinearContrast**, **RandomAdd**, **RandomMultiply**, **RandomInvert** | GDRNPP-style color augmentation (ITODD). |
| **RandomGrayscale** | Partial grayscale blending — mixes the original image with its grayscale version by a random `alpha ∈ [0, 1]` (ITODD, whose test images are grayscale). |

## Blur / noise

| Transform | What it does |
|-----------|--------------|
| **RandomMotionBlur** | Linear motion blur with random angle (0–360°) and kernel length (1–10 px). |
| **RandomGaussianBlur** | Gaussian blur with random kernel size ∈ {3, 5, 7} and sigma ∈ [0, 3]. |
| **RandomGaussianNoise** | Per-channel additive Gaussian noise; std fixed or sampled from a range. |
| **RandomAdditionalNoise** | A second additive Gaussian noise pass with an independently tuned std range, stacked on top of `RandomGaussianNoise`. |

## Dropout / occlusion

| Transform | What it does |
|-----------|--------------|
| **RandomCoarseDropout** | Drops coarse rectangular patches (imgaug-style): generates a low-resolution dropout mask and upsamples it to image resolution. |

## Depth (RGB-D configs only)

| Transform | What it does |
|-----------|--------------|
| **DepthAugment** | Sim-to-real depth augmentation on the [0, 1] normalized depth tensor: hole generators (object/scene clusters, depth-step shadows, dark-surface and surface-region holes, image-border bands), edge wobble (coherent boundary displacement), flying-pixel speckle, plus GDRNPP-style hole fill and Gaussian noise. |

### DepthAugment presets

Per-dataset settings fitted to the hole statistics of the real test depth (holes on objects, background, dark and bright surfaces, depth steps and image borders). Put the line of your dataset in the train `ops`, after any geometric augmentation and before the photometric ones.

```yaml
# LM-O
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, edge_hole_prob: 1.0, edge_far_side: True, edge_keep_frac: [0.186, 0.388], edge_width_median: 8.4, edge_ragged: 3.0, edge_ragged_sigma: 3.0, region_hole_prob: 1.0, region_frac: [0.01, 0.04], region_sigma: 5.0, region_object_bias: 0.5, border_hole_prob: 1.0, border_left: [3, 15], border_top: [3, 7], border_jitter: 4, border_smooth: 15}
# T-LESS
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, edge_hole_prob: 1.0, edge_far_side: True, edge_keep_frac: [0.08, 0.321], edge_width_median: 5.6, edge_ragged: 2.0, edge_ragged_sigma: 2.0, region_hole_prob: 0.7, region_frac: [0.002, 0.01], region_sigma: 6.0, region_dark_bias: 4.0}
# TUD-L
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, edge_hole_prob: 1.0, edge_far_side: True, edge_keep_frac: [0.08, 0.22], edge_width_median: 6.0, edge_width_sigma: 0.8, edge_width_max: 30.0, edge_ragged: 4.0, edge_ragged_sigma: 4.0, region_hole_prob: 1.0, region_frac: [0.1, 0.18], region_sigma: 10.0, region_object_bias: 0.1, region_dark_bias: 4.0, region_bright_bias: 1.0, border_hole_prob: 1.0, border_left: [19, 24], border_right: [35, 45], border_top: [35, 45], border_bottom: [0, 45], border_jitter: 2, border_smooth: 20}
# YCB-V
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, edge_hole_prob: 1.0, edge_far_side: True, edge_keep_frac: [0.5, 0.8], edge_width_median: 8.0, edge_width_sigma: 0.9, edge_width_max: 80.0, edge_ragged: 4.0, edge_ragged_sigma: 4.0, region_hole_prob: 1.0, region_frac: [0.05, 0.14], region_sigma: 12.0, region_object_bias: 0.6, region_dark_bias: 5.0, region_bright_bias: 2.0}
# HB
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, edge_hole_prob: 1.0, edge_far_side: True, edge_keep_frac: [0.154, 0.41], edge_width_median: 7.0, edge_ragged: 3.0, edge_ragged_sigma: 3.0, region_hole_prob: 1.0, region_frac: [0.005, 0.03], region_sigma: 6.0, region_dark_bias: 2.0, border_hole_prob: 1.0, border_left: [6, 14], border_right: [28, 36], border_top: [9, 13], border_bottom: [5, 9], border_jitter: 2, border_smooth: 20}
# IC-BIN
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, edge_hole_prob: 1.0, edge_far_side: True, edge_keep_frac: [0.15, 0.338], edge_width_median: 8.0, edge_ragged: 3.0, edge_ragged_sigma: 3.0, region_hole_prob: 1.0, region_frac: [0.007, 0.035], region_sigma: 8.0, region_object_bias: 0.3, region_dark_bias: 3.0, border_hole_prob: 1.0, border_left: [0, 4], border_right: [38, 48], border_top: [11, 15], border_bottom: [9, 13], border_jitter: 2, border_smooth: 20}
# ITODD
- {type: DepthAugment, fill_prob: 0.0, obj_hole_prob: 0.0, bg_hole_prob: 0.0, region_hole_prob: 1.0, region_frac: [0.08, 0.21], region_sigma: 12.0, region_object_bias: 2.5, region_dark_bias: 15.0, region_bright_bias: 5.0, border_hole_prob: 1.0, border_left: [1, 4], border_right: [40, 46], border_top: [0, 45], border_bottom: [3, 7], border_jitter: 1, border_smooth: 30}
```

These presets match the hole statistics only; their effect on training is not validated yet.

## Background / composition

| Transform | What it does |
|-----------|--------------|
| **BackgroundReplacement** | Replaces the image background with a random image from `background_dir` (e.g. VOC2012 `JPEGImages`). |
| **CopyPasteSingleClass** | Cross-image same-class copy-paste with a VOC background. Adds 2–10 extra instances per image, filtered by visibility and bbox-IoU, with optional edge blending. |
| **PasteSurfaces** | RGB-D: stands large panels and boxes (monitor / wall-like surfaces) on the floor plane fitted to the depth, ray-cast into depth and RGB (texture from `background_dir`); occluded masks and visibility are updated. Place it first in `ops` (LM-O RGB-D). |

## Geometric pose augmentation

| Transform | What it does |
|-----------|--------------|
| **PoseAugmentation** | Unified affine warp around the camera principal point. In *scene mode* all objects share one `(angle, scale)`; in *instance mode* each object samples its own and the warped instances are composited in depth order, with rollback for any object whose visibility drops below threshold. Updates `R`, `t`, and 2D keypoints consistently. With depth (RGB-D), scene mode warps the depth map the same way (a roll keeps every z, a zoom by `scale` divides it by `scale`; blank areas become holes); instance mode does not support depth. |

## Filtering

| Transform | What it does |
|-----------|--------------|
| **FilterSmallBoxLowVis** | Removes target annotations whose modal bbox is below `min_size` pixels or whose BOP `visib_fract` is below `min_visib`. Keeps target arrays consistent with the BOP eval protocol. |

## Preprocessing / dtype

| Transform | What it does |
|-----------|--------------|
| **Resize** | Resize image (and 2D targets) to the model input size (480 × 640). |
| **ConvertPose** | Encodes the GT translation relative to the box (`rx, ry, log tz`) with each annotation's `cam_K`. |
| **ConvertPILImage** | Converts PIL → float32 tensor with optional [0, 1] scaling. |
| **ConvertBoxes** | Converts boxes to `cxcywh` and normalizes by image size. |

## Augmentation scheduling

`configs/race6d/include/dataloader.yml` declares a `stop_epoch` policy that turns the photometric and dropout
augmentations off from epoch 71. The dataset configs override it with `policy: {name: default}`, so every augmentation
stays on until the last epoch. The T-LESS RGB-D config keeps the policy of its released checkpoint (`stop_epoch`,
photometric, dropout and `DepthAugment` on for epochs 0–25).

## Tuning surface

The current config is intentionally conservative. Promising tuning directions:

- **Per-transform application probability** — defaults come from `AugmentationManager` in `_transforms.py`; per-dataset sweeps usually move the needle.
- **Strength ranges** — `RandomMotionBlur.length`, `RandomGaussianNoise.scale`, `RandomCoarseDropout.drop_prob`, `PoseAugmentation.zoom_range`, etc.
- **Copy-paste density** — `min_extra` / `max_extra` and `min_visibility` thresholds.
- **Per-dataset selection** — not every augmentation helps every dataset (e.g. `DepthAugment` is only enabled for RGB-D configs).

## See also

- `src/data/transforms/_transforms.py` — augmentation class implementations
- `src/data/transforms/container.py` — `Compose`, `stop_epoch` policy, and spatial-transform mask-rollback logic
- `configs/race6d/include/optimizer.yml` — AdamW + flat-cosine LR + EMA defaults
- `configs/race6d/include/race6d_r50vd_best.yml` — default model architecture
