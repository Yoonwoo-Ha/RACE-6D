# Data preparation helpers

| File | Purpose |
|------|---------|
| `compute_symmetry_centers.py` | Lists the objects whose symmetry axis is off the model origin and computes the shift P |
| `process_models.py` | Moves the meshes by `-P` and, with `--keypoints`, the keypoint cache; the P values are in `PALIGNED_VALUES` |
| `test_kpt.ipynb` | Builds and visualizes the 3D keypoint cache (`cached_keypoints_3d_edge.json`), one cell per dataset |
| `fixcamK.py` | Converts the YCB-V real training poses to a single camera matrix |

## Objects with an off-origin symmetry axis

A BOP symmetry maps a model point `x` to `R x + t`. RACE-6D uses only the rotation `R` of each symmetry, so an object
whose symmetry axis does not pass through the model origin (`t != 0`) is trained in a model frame moved to a point `P`
on that axis. In that frame its symmetries are pure rotations.

| Dataset | Object | P (mm) |
|---------|--------|--------|
| YCB-V | 19 | [10.19792, -3.530759, -1.122307] |
| YCB-V | 20 | [-3.871881, -10.338984, 0.088532] |
| ITODD | 2 | [0.0, 5.344601, 1.363572] |

The other BOP datasets used here (LM-O, TUD-L, HB, IC-BIN) have no such objects. T-LESS objects 25 and 26 also have an
off-origin axis (P = [-7, 0, z]), but only their body is symmetric: a 14 mm protrusion on one end has no counterpart.
They are trained as asymmetric objects instead (`symmetry_override: {25: none, 26: none}` in the T-LESS configs); BOP
evaluation keeps their official symmetry.

The values are computed with `compute_symmetry_centers.py` (the point of the axis nearest to the surface centroid) and
used by the YCB-V and ITODD configs. Once a model is trained, its P must not change: the network predicts the
translation of P and the postprocessor subtracts `R @ P` with the same values.

- The released YCB-V checkpoints were trained with other points of the same axes, 19: [10.4796698, -5.41739619,
  -1.23077576] and 20: [-8.82785585, -10.93032056, 0.09932552]. The `*_bop.yml` configs set these values in
  `aligned_offsets` for inference with them.
- A model trained before this change in the original model frame (earlier ITODD runs) must be evaluated without
  `aligned_offsets`.

### Preparing a dataset

1. Find the objects and P (only needed for a new dataset):

   ```bash
   python sub/compute_symmetry_centers.py --models_path ~/Downloads/bop_eval/itodd/models_eval
   ```

2. Move the meshes in `models/` and `models_eval/`:

   ```bash
   python sub/process_models.py itodd ~/Downloads/itodd/models
   python sub/process_models.py itodd ~/Downloads/itodd/models_eval
   ```

3. Build the keypoint cache from the moved `models_eval/` with `test_kpt.ipynb` (see below), or move the existing
   cache with `--keypoints ~/Downloads/itodd/cached_keypoints_3d_edge.json` in step 2.

4. Keep the original poses in `scene_gt.json` and the COCO annotations. The dataset moves them when it loads them
   (`need_aligned: True` with `paligned_values` in the configs).

5. Keep the meshes used for BOP scoring (`~/Downloads/bop_eval/<dataset>/`) in the original frame. With `cam_K`, the
   postprocessor returns poses in the original frame (`t - R @ P`, `aligned_offsets` in the configs).

`process_models.py` keeps the original files as `*.backup_paligned` and skips files that already have a backup, so
running it again does not move anything twice. It changes only the vertex coordinates; the PLY format and the other
properties are kept.

## 3D keypoint cache

`test_kpt.ipynb` builds `<dataset>/cached_keypoints_3d_edge.json`: 32 keypoints per object (8 box corners and 2
points on each of the 12 box edges) and the 12 edges (`edges_mask`). Run the first cell, then the cell of the dataset;
the cell visualizes the keypoints, and the commented `save_keypoints_to_json` line writes the cache.

- The box is the outermost box of the surface points: the minimum-area rectangle of the x-y hull and the z range.
- For a symmetric object, the same box is also built in a symmetry frame: z is the symmetry axis (with several axes,
  the axis of the highest order, and x a perpendicular 180-degree axis, with the box aligned to them, e.g. the YCB-V wood
  block). It is used when its keypoints match better after the symmetry rotations (the plot shows `axis box` and the
  axis as a dashed line); otherwise the model-frame box is kept (`model box`). `sym err` is the largest keypoint distance after a
  symmetry rotation. `AXIS_BOX = False` keeps the model-frame box for all objects.
- `PLY_DIR` must hold the meshes in the training frame (`<dataset>/models_eval`, moved by `-P` for the objects in
  `PALIGNED_VALUES`), so the keypoints are in the same frame.
- Point sampling is seeded, so the cache is reproducible.

A rectangular box cannot match a 3-, 5-, 6-, 12-, 18- or 23-fold symmetry (ITODD 2, 8, 14, 17, 23 and 25). Released
checkpoints store the keypoints they were trained with, so a new cache only affects new training.

## YCB-V real training data: one camera matrix

The YCB-V real images were recorded with two cameras: scenes 0–59 use one camera matrix and scenes 60–91 another.
`fixcamK.py` rewrites the poses of the scenes with the other matrix (`scene_gt.json`, `scene_camera.json` in
`train_real/`) so that they project to the same pixels with a single matrix, without changing the images: it projects
model points with the original matrix and solves PnP with the target one.

The released YCB-V checkpoints were trained on `train_real/` converted this way (scenes 60–91 to the matrix of
scenes 0–59). Set `BASE_DIR` and `MODELS_DIR` at the bottom of the script; it asks for confirmation and keeps the
original files as backups.
