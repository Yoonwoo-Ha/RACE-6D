# Test-time depth refinement

An RGB model estimates an object's distance from its apparent size, and distance is its largest pose error. When a
depth image is available at test time, `RACE6DPostProcessor` can correct that distance without retraining. Each
selected prediction is moved along its viewing ray until the visible side of its 3D model agrees with the measured
depth. Rotation and 2D position stay as predicted.

The refinement is off by default. No config in `configs/` enables it.

## How it works

For every prediction with score ≥ `score_thr`, and for `iters` iterations:

1. Pose `num_points` model points (sampled on the mesh surface) with the predicted pose and project them.
2. Keep the visible side. On a grid of `cell` px only the points nearest to the camera are kept. The grid is a
   128 × 128-cell patch over the object, so objects wider than 384 px use proportionally larger cells.
3. Compare each kept point's depth with the measured depth at its pixel. Pairs more than `max_diff_mm` apart are
   dropped (occluders in front, sensor holes, background).
4. Scale the translation by `(t_z + median difference) / t_z`. Predictions with fewer than 15 usable points keep their
   translation.

The code is in `depth_refine.py` (`refine_translations_with_depth`). It is one PyTorch implementation for CPU and GPU.

## Configuration

```yaml
RACE6DPostProcessor:
  depth_refine:
    models_path: ~/Downloads/lmo/models_eval   # obj_XXXXXX.ply in the original BOP model frame
    score_thr: 0.25
    device: auto
    compile: True

val_dataloader:
  dataset:
    return_depth_mm: True      # target["depth_mm"]: raw depth in mm at the original image resolution
```

| Key | Default | Meaning |
|-----|---------|---------|
| `models_path` | (required) | Folder with the BOP meshes in the original model frame, e.g. `models_eval/`. The refinement runs after `aligned_offsets` are reversed, so use the original meshes for YCB-V too. |
| `score_thr` | `0.25` | Only predictions scoring at least this much are refined; the others are returned unchanged. |
| `num_points` | `3000` | Model points per object. 1500 points lose 0.003–0.007 AR. |
| `cell` | `3` | Grid cell in px for the visibility test. |
| `max_diff_mm` | `50` | Measured-vs-predicted depth gap above which a point is ignored. |
| `iters` | `2` | Refinement iterations. |
| `device` | `auto` | `auto` runs where the network outputs are (GPU inference → GPU, CPU or NPU host → CPU), whatever the batch size. `cpu` / `cuda` force a device; the result is returned on the original one. |
| `compile` | `False` | `torch.compile` the GPU core (first call compiles). Ignored on CPU. |

On GPU, all predictions are processed with fixed shapes and no host synchronization, and the score threshold only
masks the update. On CPU, only predictions above the threshold are processed.

### Calling the postprocessor directly

```python
results = postprocessor(outputs, orig_target_sizes, cam_K=K, depth=depth_mm)
```

`depth_mm` is `[B, H, W]` in millimeters at the original image resolution (the resolution of `cam_K`), with 0 for
missing measurements. Convert raw BOP depth with the image's `depth_scale` from `scene_camera.json`. Without `cam_K`
or `depth`, the postprocessor works as before.

With `bop_eval` enabled (see [src/solver/README.md](../../solver/README.md)), the per-epoch BOP19 AR includes the
refinement.

## Results

Released checkpoints, BOP19 test targets, local BOP toolkit evaluation (not submitted to the BOP server).
Settings: `score_thr` 0.25, 3000 points.

| Dataset | Model | AR | AR with refinement |
|---------|-------|---:|-------------------:|
| LM-O | RGB | 0.683 | 0.733 |
| YCB-V | RGB | 0.782 | 0.831 |
| T-LESS | RGB | 0.680 | 0.727 |
| TUD-L | RGB | 0.802 | 0.895 |
| IC-BIN | RGB | 0.597 | 0.656 |
| YCB-V | RGB-D | 0.836 | 0.847 |
| T-LESS | RGB-D | 0.755 | 0.735 |

HB is not listed because its test ground truth is not public.

The gain comes from the distance: VSD and MSSD rise, while MSPD (a 2D projection error) is unchanged. For RGB-D models
the network already uses depth, and the refinement adds little or loses accuracy (T-LESS). Use it with RGB models.
Results that use depth count as RGB-D in the BOP benchmark.

## Latency

RTX 5090, FP32. Times include GPU transfer of the image and depth, the network, the postprocessor and the results
copied to the CPU. The added time is the median of paired runs with and without depth on the same batches. Batch-1
runs are CPU-launch-bound and jitter by about ±1 ms between runs.

| Dataset (image size) | Batch 1 FPS | Batch 8 FPS | Batch 32 FPS |
|----------------------|-------------|-------------|--------------|
| LM-O (640 × 480) | 90.8 → 86.7 (+0.53 ms) | 328.6 → 321.4 (+0.07 ms) | 329.5 → 322.8 (+0.06 ms) |
| TUD-L (640 × 480) | 96.2 → 91.1 (+0.58 ms) | 326.9 → 319.9 (+0.07 ms) | 328.7 → 324.3 (+0.04 ms) |
| IC-BIN (640 × 480) | 98.8 → 93.9 (+0.53 ms) | 325.6 → 316.3 (+0.09 ms) | 324.6 → 312.5 (+0.12 ms) |
| ITODD (1280 × 960) | 85.5 → 80.5 (+0.72 ms) | 325.1 → 305.5 (+0.20 ms) | 323.1 → 300.8 (+0.23 ms) |

On CPU (i7-14700K, 8 threads) the refinement takes about 0.5 ms for one refined object and about 2 ms for 8–18
refined objects per 640 × 480 image.

## Requirements and limits

- The depth image must be registered to the RGB image and share its intrinsics, as in the BOP datasets.
- The meshes must be in the frame of the predicted poses (the original BOP frame).
- Only the distance is corrected. Rotation errors remain.
