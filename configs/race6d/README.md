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
> These settings are a **working baseline**, not a tuned optimum. Probability, strength range, ordering, and the set of augmentations themselves are tunable knobs. Combining and tuning them — or wiring in the additional implementations under the *Unused augmentations* block at the bottom of `_transforms.py` — is expected to yield further accuracy gains.

## Photometric / color

| Transform | What it does |
|-----------|--------------|
| **ColorJitter** | GDRNPP-style saturation jitter (default range 0.5–3.0); optional hue shift. |
| **RandomHSVAdjust** | Joint random brightness (V) and saturation (S) scaling in HSV space. |
| **RandomSharpen** | Sharpness enhancement with a random factor in [0, 20] (factor < 1 blurs, > 1 sharpens). |
| **GDRNPhotoAug** | Wraps the `yolox/gdrn` `COLOR_AUG_CODE` photometric block via `imgaug` (matches gdrnpp_bop2022 defaults). |
| **GDRNGrayscale** | Partial grayscale blending — mixes the original image with its grayscale version by a random `alpha ∈ [0, 1]`. |

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

## Sensor / ISP simulation

| Transform | What it does |
|-----------|--------------|
| **RandomISPSimulation** | Simulates a real-camera ISP pipeline — Bayer-CFA demosaicing artifacts, unsharp masking, and CLAHE — to narrow the sim-to-real gap on PBR renders (SurfEmb-style). |

## Depth (RGB-D configs only)

| Transform | What it does |
|-----------|--------------|
| **DepthAugment** | Sim-to-real depth augmentation on the [0, 1] normalized depth tensor: object-boundary clustered holes (edge-biased), mid-distance scene-region holes, edge wobble (coherent boundary displacement), flying-pixel speckle, plus GDRNPP-style hole fill and Gaussian noise. |

## Background / composition

| Transform | What it does |
|-----------|--------------|
| **BackgroundReplacement** | Replaces the image background with a random image from `background_dir` (e.g. VOC2012 `JPEGImages`). |
| **CopyPasteSingleClass** | Cross-image same-class copy-paste with a VOC background. Adds 2–10 extra instances per image, filtered by visibility and bbox-IoU, with optional edge blending. |

## Geometric pose augmentation

| Transform | What it does |
|-----------|--------------|
| **PoseAugmentation** | Unified affine warp around the camera principal point. In *scene mode* all objects share one `(angle, scale)`; in *instance mode* each object samples its own and the warped instances are composited in depth order, with rollback for any object whose visibility drops below threshold. Updates `R`, `t`, and 2D keypoints consistently. |

## Filtering

| Transform | What it does |
|-----------|--------------|
| **FilterSmallBoxLowVis** | Removes target annotations whose modal bbox is below `min_size` pixels or whose BOP `visib_fract` is below `min_visib`. Keeps target arrays consistent with the BOP eval protocol. |

## Preprocessing / dtype

| Transform | What it does |
|-----------|--------------|
| **Resize** | Resize image (and 2D targets) to the model input size (480 × 640). |
| **ConvertPose** | Loads per-dataset object meshes from `coco_path/models/` and packs pose targets (R, t, sampled 3D model points) into the target dict. |
| **ConvertPILImage** | Converts PIL → float32 tensor with optional [0, 1] scaling. |
| **ConvertBoxes** | Converts boxes to `cxcywh` and normalizes by image size. |

## Augmentation scheduling

Augmentations are scheduled via the `stop_epoch` policy declared in `configs/race6d/include/dataloader.yml`:

```yaml
policy:
  name: stop_epoch
  epoch: 71  # epoch in [71, ∞) disables the listed `ops`
  ops:
    - ColorJitter
    - RandomHSVAdjust
    - RandomSharpen
    - RandomMotionBlur
    - RandomGaussianBlur
    - RandomGaussianNoise
    - RandomAdditionalNoise
    - RandomCoarseDropout
    - RandomRotateExpand
```

The photometric and dropout augmentations turn off in the final stage of training so the model can fit clean data near convergence. Geometric augmentations (`PoseAugmentation`, `CopyPasteSingleClass`), `BackgroundReplacement`, and `DepthAugment` remain active throughout training.

## Tuning surface

The current config is intentionally conservative. Promising tuning directions:

- **Per-transform application probability** — defaults come from `AugmentationManager` in `_transforms.py`; per-dataset sweeps usually move the needle.
- **Strength ranges** — `RandomMotionBlur.length`, `RandomGaussianNoise.scale`, `RandomCoarseDropout.drop_prob`, `PoseAugmentation.zoom_range`, etc.
- **`stop_epoch` boundary** — currently 71; some datasets benefit from a longer photometric tail.
- **Copy-paste density** — `min_extra` / `max_extra` and `min_visibility` thresholds.
- **Per-dataset selection** — not every augmentation helps every dataset (e.g. `DepthAugment` is only enabled for RGB-D configs; ISP simulation is more impactful on PBR-heavy datasets).

In addition, `src/data/transforms/_transforms.py` ships several **registered but currently unused** augmentations under the *Unused augmentations* block at the bottom of the file:

- `PadToSize`
- `RandomBrightness`, `RandomContrast`, `RandomLinearContrast`, `RandomGrayscale`
- `RandomAdd`, `RandomMultiply`

These are kept as reference implementations and can be wired into any config via `{type: <Name>, ...}` in the `ops` list. Mixing them into the active pipeline — or replacing one of the active transforms with a tuned alternative — is a straightforward path to further improvements.

## See also

- `src/data/transforms/_transforms.py` — augmentation class implementations and `(unused)` reference block
- `src/data/transforms/container.py` — `Compose`, `stop_epoch` policy, and spatial-transform mask-rollback logic
- `configs/race6d/include/optimizer.yml` — AdamW + flat-cosine LR + EMA defaults
- `configs/race6d/include/race6d_r50vd_best.yml` — default model architecture
