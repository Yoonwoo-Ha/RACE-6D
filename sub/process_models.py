#!/usr/bin/env python3
"""
Move the PLY models of objects whose symmetry axis is off the model origin by -P (v_new = v - P).

P is a point on the symmetry axis (sub/compute_symmetry_centers.py), so the BOP symmetries of the moved model are pure
rotations. The original files are kept as obj_XXXXXX.ply.backup_paligned; a model that already has a backup is
skipped, so running the script twice does not move it twice.

The poses are moved on the fly (t_new = t + R P, dataset need_aligned / paligned_values in the configs); the COCO
annotations keep the original poses. The postprocessor reverses the shift with
`aligned_offsets`, so these values must not change once a model is trained with them.

Usage:
    python sub/process_models.py ycbv ~/Downloads/ycbv/models
    python sub/process_models.py itodd ~/Downloads/itodd/models --keypoints ~/Downloads/itodd/cached_keypoints_3d_edge.json
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData

PALIGNED_VALUES = {
    'ycbv': {
        19: [10.19792, -3.530759, -1.122307],
        20: [-3.871881, -10.338984, 0.088532],
    },
    'itodd': {
        2: [0.0, 5.344601, 1.363572],
    },
}


def transform_ply(ply_path, translation):
    """Return the PLY with its vertices moved by -translation; every other property and the format are kept."""
    ply = PlyData.read(str(ply_path), mmap=False)
    vertex = ply['vertex'].data
    before = np.stack([vertex['x'], vertex['y'], vertex['z']], 1)
    for axis, value in zip('xyz', translation):
        vertex[axis] = vertex[axis] - value
    after = np.stack([vertex['x'], vertex['y'], vertex['z']], 1)
    print(f'  Bounds before: {before.min(0).round(3).tolist()} {before.max(0).round(3).tolist()}')
    print(f'  Bounds after:  {after.min(0).round(3).tolist()} {after.max(0).round(3).tolist()}')
    return ply


def process_ply_models(dataset, models_path):
    models_path = Path(models_path).expanduser()
    for obj_id, translation in PALIGNED_VALUES[dataset].items():
        ply_path = models_path / f'obj_{obj_id:06d}.ply'
        backup_path = models_path / f'obj_{obj_id:06d}.ply.backup_paligned'
        if not ply_path.exists():
            print(f'{ply_path.name}: not found, skipped')
            continue
        if backup_path.exists():
            print(f'{ply_path.name}: {backup_path.name} exists (already moved), skipped')
            continue
        print(f'{ply_path.name}: moving by -{translation}')
        shutil.copy2(ply_path, backup_path)
        transform_ply(ply_path, translation).write(str(ply_path))


def process_keypoints(dataset, keypoints_path):
    """The cached box keypoints are translation-equivariant, so they move with the model."""
    keypoints_path = Path(keypoints_path).expanduser()
    backup_path = keypoints_path.with_name(keypoints_path.name + '.backup_paligned')
    if backup_path.exists():
        print(f'{keypoints_path.name}: {backup_path.name} exists (already moved), skipped')
        return
    data = json.loads(keypoints_path.read_text())
    shutil.copy2(keypoints_path, backup_path)
    for obj_id, translation in PALIGNED_VALUES[dataset].items():
        keypoints = np.asarray(data['keypoints_3d'][str(obj_id)]) - np.asarray(translation)
        data['keypoints_3d'][str(obj_id)] = keypoints.tolist()
        print(f'{keypoints_path.name}: obj {obj_id} keypoints moved by -{translation}')
    keypoints_path.write_text(json.dumps(data, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=sorted(PALIGNED_VALUES))
    parser.add_argument('models_path', help='folder with obj_XXXXXX.ply, e.g. ~/Downloads/itodd/models')
    parser.add_argument('--keypoints', help='cached_keypoints_3d_edge.json to move as well')
    args = parser.parse_args()

    process_ply_models(args.dataset, args.models_path)
    if args.keypoints:
        process_keypoints(args.dataset, args.keypoints)
