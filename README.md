<p align="center">
  <h1 align="center"> <ins>RACE-6D</ins> 🎯<br>Real-time Accurate Coarse-to-finE object 6D Pose Transformer</h1>
  <p align="center">
    A transformer-based framework for 6D object pose estimation, extending RT-DETR with parallel pose heads to predict 3D rotation and translation of known objects from RGB / RGB-D images.
  </p>
  <div align="center">

  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python)](https://www.python.org/)
  [![BOP](https://img.shields.io/badge/Benchmark-BOP-orange)](https://bop.felk.cvut.cz/)
  [![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

  </div>
</p>

RACE6D is a 6D object pose estimation framework built on PyTorch. It extends **RT-DETR** (Real-Time Detection Transformer) with parallel pose-estimation heads (rotation, translation, keypoints, visibility) and is designed for the **BOP benchmark** datasets: LMO, YCBV, T-LESS, TUDL, HB, IC-BIN, and ITODD.

## 🔍 Overview

RACE6D treats 6D pose estimation as a **set-prediction problem**. Object queries are refined through a **DQE (Dynamic Query Enhancement) decoder**, and each query predicts a full pose — class, 2D box, 6D continuous rotation, depth, and keypoints — in parallel. Hungarian matching assigns predictions to ground truth during training, and pose supervision uses an ADD-S loss on sampled 3D model points.

## ✨ Highlights

- **End-to-end transformer** for joint detection + 6D pose estimation
- **DQE decoder** with 3 refinement layers for progressive pose refinement
- **6D continuous rotation** representation ([Zhou et al., CVPR 2019](https://arxiv.org/abs/1812.07035))
- **ADD-S loss** on sampled 3D model points with symmetry-aware matching
- **Registry + YAML-driven** architecture — swap backbones, encoders, or heads with a single config line
- **RGB and RGB-D** modalities supported

## 📦 Installation

### Dependencies

**Tested environment**: Python 3.10, PyTorch 2.7.0 + CUDA 12.8

Minimum requirements:
- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) — 3D rotation utilities (`rotation_6d_to_matrix`)
- [Open3D](http://www.open3d.org/) — 3D model loading
- [`torch_linear_assignment`](https://github.com/ivan-chai/torch-linear-assignment) — GPU-accelerated Hungarian matching

```bash
git clone https://github.com/Yoonwoo-Ha/RACE-6D.git && cd RACE-6D

# 1) Standard pip packages (numpy, scipy, opencv, tensorboard, ...)
pip install -r requirements.txt

# 2) Extra packages that need custom installation
pip install open3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install git+https://github.com/ivan-chai/torch-linear-assignment.git
```

> **Note**: PyTorch3D installation can be version-sensitive. If the `pip install` from GitHub fails, follow the [official PyTorch3D install guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) that matches your PyTorch / CUDA combination.

### Datasets

Download the BOP datasets you want to train on from the [BOP benchmark website](https://bop.felk.cvut.cz/datasets/):

- [LM-O](https://bop.felk.cvut.cz/datasets/#LM-O)
- [YCB-V](https://bop.felk.cvut.cz/datasets/#YCB-V)
- [T-LESS](https://bop.felk.cvut.cz/datasets/#T-LESS)
- [TUD-L](https://bop.felk.cvut.cz/datasets/#TUD-L)
- [HB](https://bop.felk.cvut.cz/datasets/#HB)
- [IC-BIN](https://bop.felk.cvut.cz/datasets/#IC-BIN)
- [ITODD](https://bop.felk.cvut.cz/datasets/#ITODD)

Update the dataset paths in `configs/race6d/r50vd/race6d_r50vd_{dataset}_rgb.yml` to match your local layout. Each dataset directory must contain the `models/` folder (3D CAD models are loaded by the criterion at initialization).

## 🚀 Getting Started

### 1. Train

Single-GPU:

```bash
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml --use-amp
```

Multi-GPU distributed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port=8989 \
    tools/train.py -c configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml --use-amp
```

Resume or fine-tune:

```bash
# Resume
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml \
    -r output/race6d_r50vd_lmo_rgb/last.pth

# Fine-tune from pretrained weights
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml \
    -t path/to/pretrained.pth
```

Override config values from the CLI:

```bash
python tools/train.py -c config.yml -u key1=value1 key2=value2
```

### 2. Evaluate

```bash
python tools/train.py -c configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml \
    --test-only -r path/to/checkpoint.pth
```

### 3. TensorBoard

```bash
tensorboard --logdir=output/race6d_r50vd_lmo_rgb/summary/ --port=8989
```

### 4. Export & profile

```bash
python tools/export_onnx.py -c config.yml -r checkpoint.pth --check
python tools/run_profile.py -c config.yml
```

## 🧠 Architecture

```
Image [B, 3, H, W]
      │
      ▼
 PResNet backbone          (multi-scale features, strides 8 / 16 / 32)
      │
      ▼
 Hybrid Encoder            (multi-scale fusion → 256-d features)
      │
      ▼
 RACE6D Transformer-DQE    (3 refinement layers, object queries)
      │
      ├── class logits
      ├── 2D boxes
      ├── 6D rotation      → rotation_6d_to_matrix
      ├── depth / translation
      ├── keypoints
      └── visibility
```

The model is assembled declaratively through a **registry + factory + dependency-injection** system in `src/core/workspace.py`. Components are registered with `@register()`, sub-components are wired via `__inject__`, and the entire pipeline — model, optimizer, data — is defined in YAML.

### Configuration composition

YAML configs use `__include__` for hierarchical composition:

```
race6d_r50vd_lmo_rgb.yml
├── __include__: ../../dataset/coco_detection.yml   # base dataset
├── __include__: ../../runtime.yml                   # runtime settings
├── __include__: ../include/dataloader.yml           # augmentation pipeline
├── __include__: ../include/optimizer.yml            # AdamW, LR schedule, EMA
├── __include__: ../include/race6d_r50vd_best.yml    # model architecture
└── Local overrides (num_classes, paths, loss weights, ...)
```

## 🗃️ Code Structure

```graphql
├── configs/race6d/
│   ├── include/                      # Shared config fragments (model, optimizer, dataloader)
│   └── r50vd/                        # Per-dataset configs (LMO, YCBV, T-LESS, TUD-L, HB, IC-BIN, ITODD)
├── src/
│   ├── core/workspace.py             # Registry, factory, DI
│   ├── core/yaml_config.py           # Lazy YAML config loader
│   ├── nn/backbone/                  # PResNet and variants
│   ├── zoo/race6d/
│   │   ├── race6d.py                 # Main model (composition)
│   │   ├── hybrid_encoder.py         # Multi-scale encoder
│   │   ├── race6d_decoder_dqe.py     # DQE decoder (core research)
│   │   ├── race6d_criterion_addr.py  # ADD-S loss and matcher supervision
│   │   ├── race6d_postprocessor.py   # Pose decoding at inference
│   │   ├── matcher.py                # Hungarian matcher
│   │   └── denoising.py              # Query denoising
│   ├── solver/
│   │   ├── pose_solver.py            # PoseSolver (main task)
│   │   └── pose_engine.py            # Training loop
│   └── data/
│       ├── dataset/coco_dataset.py   # COCO-format with pose annotations
│       └── transforms/_transforms.py # Pose-specific augmentation
├── tools/
│   ├── train.py                      # Main entry point
│   ├── export_onnx.py                # ONNX export
│   └── run_profile.py                # Profiling
└── output/                           # Checkpoints, logs, TensorBoard summaries
```

## 🧩 Task Dispatch

`tools/train.py` reads `task` from the YAML config and dispatches to the appropriate solver:

| `task` value       | Solver           | Purpose                  |
|--------------------|------------------|--------------------------|
| `pose_estimation`  | `PoseSolver`     | Main 6D pose task        |
| `kpt_estimation`   | `KptSolver`      | Keypoint-only training   |
| `classification`   | `ClasSolver`     | Classification baselines |

## ⚙️ Deploy Mode

For inference, call `model.deploy()` to fuse `BatchNorm` into `Conv` layers (RepVGG-style reparameterization in `HybridEncoder`). Always call this before ONNX export or benchmarking.

```python
model.eval()
model.deploy()
```

## 📁 Supported Datasets

| Dataset | Modality | Config |
|---------|----------|--------|
| LM-O    | RGB      | `configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml`   |
| YCB-V   | RGB / RGB-D | `configs/race6d/r50vd/race6d_r50vd_ycbv_rgb.yml`, `..._rgbd.yml` |
| T-LESS  | RGB / RGB-D | `configs/race6d/r50vd/race6d_r50vd_tless_rgb.yml`, `..._rgbd.yml` |
| TUD-L   | RGB      | `configs/race6d/r50vd/race6d_r50vd_tudl_rgb.yml`  |
| HB      | RGB      | `configs/race6d/r50vd/race6d_r50vd_hb_rgb.yml`    |
| IC-BIN  | RGB      | `configs/race6d/r50vd/race6d_r50vd_icbin_rgb.yml` |
| ITODD   | RGB      | `configs/race6d/r50vd/race6d_r50vd_itodd_rgb.yml` |

## 🙏 Acknowledgements

RACE6D builds on ideas and code from:

- [RT-DETR / RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) — real-time detection transformer backbone
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) — 3D geometry and rotation utilities
- [BOP Toolkit](https://github.com/thodan/bop_toolkit) — benchmark evaluation
- [Zhou et al., CVPR 2019](https://arxiv.org/abs/1812.07035) — 6D continuous rotation representation

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{ha2026race6d,
  title     = {RACE-6D: Real-time Accurate Coarse-to-finE object 6D Pose Transformer},
  author    = {Ha, Yoonwoo and Moon, Hyungpil},
  booktitle = {CVPR 2026 (Findings)},
  year      = {2026}
}
```

## 📬 Contact

For questions, feedback, or collaboration, please open an issue on GitHub or contact the maintainer via the repository page.
