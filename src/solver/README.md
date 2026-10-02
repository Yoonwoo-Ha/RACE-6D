# BOP19 evaluation during training

`PoseSolver` can score every epoch, and `--test-only`, with the official BOP19 pose metrics
(VSD, MSSD, MSPD and their mean, AR). It runs `scripts/eval_bop19_pose.py` from the
[BOP toolkit](https://github.com/thodan/bop_toolkit) on the predictions of the validation set.

## How it works

1. `pose_engine.evaluate` calls the postprocessor with the per-image intrinsics `cam_K`. The
   postprocessor then returns BOP poses. Translations are in millimeters and rotations are
   egocentric (`rot_repr: allo` predictions are converted). Poses are in the original BOP model
   frame: `aligned_offsets` are reversed, for example for YCB-V objects 19 and 20.
2. `BOPPoseEvaluator` (`bop_evaluator.py`) takes `scene_id` and `original_filename` from each COCO
   image entry. It maps every prediction to `(scene_id, im_id, obj_id, score, R, t)`; labels are
   mapped to BOP object ids through the category file.
3. After the validation pass, rank 0 writes one BOP CSV with time `0.0` and runs the toolkit.
   It reads `scores_bop19.json` and adds the scores to the epoch statistics.

The validation COCO set must contain the BOP19 test images. For YCB-V, that is the 900 images
listed in `test_targets_bop19.json`.

## Configuration

```yaml
bop_eval:
  enabled: true
  dataset: ycbv                            # BOP dataset name
  split: test
  dataset_root: ~/Downloads/bop_eval/ycbv  # original BOP dataset folder
  toolkit_path: ~/Downloads/bop_toolkit    # bop_toolkit clone
  renderer_path: ~/Downloads/bop_renderer/build  # built bop_renderer (renderer_type: cpp)
  renderer_type: cpp                       # cpp | vispy | python
  targets_filename: test_targets_bop19.json
  num_workers: 8                           # toolkit processes
  checkpoint_topk: 5                       # checkpoints kept by BOP AR (and by COCO mAP)
  # method: race6deval                     # CSV name prefix (default)
```

Relative paths are resolved from the repository root, and `~` is expanded.

## Path setup

- **`dataset_root`:** an original BOP dataset folder. It must contain the test split, `models_eval/`
  and the targets file. The toolkit uses its parent folder as `BOP_PATH`, so the folder name
  must equal `dataset`.
  - Use the original BOP meshes here, not meshes shifted for training. The predictions are already
    mapped back to the original model frame.
  - For YCB-V, keep a separate copy if your training data uses the shifted meshes of objects 19 and 20.
- **`toolkit_path`:** a clone of the BOP toolkit, either the official one or the
  [RACE6D fork](https://github.com/Yoonwoo-Ha/bop_toolkit). It is added to `PYTHONPATH`, and the
  evaluation script runs from this folder.
- **`renderer_path`:** the build folder of [bop_renderer](https://github.com/thodan/bop_renderer),
  which contains `bop_renderer*.so`. It is needed for `renderer_type: cpp` and is added to
  `PYTHONPATH`. With `vispy`, no build is needed, but VSD is slower and needs an OpenGL context.

## Outputs (in `output_dir`)

| Path | Content |
|------|---------|
| `log.txt` | `test_bop_vsd`, `test_bop_mssd`, `test_bop_mspd` and `test_bop_ar` per epoch |
| `bop_results/<method>_<dataset>-<split>.csv` | Predictions of the latest evaluation |
| `bop_eval_results/` | Toolkit error and score files of the latest evaluation |
| `bop_scores/epochNNNN.json` | Toolkit scores of every epoch |
| `bop_eval_latest.log` | Toolkit console output |
| `ckpNNNN_topR_SCORE.pth`, `bop_top5.json` | Top-k checkpoints by BOP AR |
| `ckpNNNN_map_topR_SCORE.pth`, `map_top5.json` | Top-k checkpoints by COCO mAP |

When `bop_eval` is enabled, `last.pth` and these top-k checkpoints replace the periodic
`checkpointNNNN.pth` files.

## Usage

The training command does not change. With `bop_eval` in the config, each epoch ends with a
BOP19 evaluation. For YCB-V this takes about two minutes on the 900 test images with 8 workers.

```bash
python tools/train.py -c path/to/config_with_bop_eval.yml --test-only -r path/to/checkpoint.pth
```

Use the per-epoch AR to monitor training; selecting a checkpoint by it selects on the test set.
The CSV time column is `0.0`, so measure run time separately for a BOP submission.
