#!/usr/bin/env python3
"""
Find symmetric objects whose symmetry axis does not pass through the model origin, and the shift P that fixes it.

A BOP symmetry maps x -> R x + t. Training only uses the rotation part R (decoder _build_symmetry_matrices), so a
symmetry with t != 0 is applied about the wrong axis. Every point P with (I - R) P = t lies on the symmetry axis;
moving the model by -P (vertices v - P, poses t_gt + R_gt P) turns the symmetry into a pure rotation. Any point of the
axis works; this script takes the axis point nearest to the surface centroid.

For continuous symmetries the axis passes through the given offset. Objects with several discrete symmetries use
all of them in one least-squares solve.

Usage:
    python sub/compute_symmetry_centers.py --models_path ~/Downloads/bop_eval/tless/models_eval
    python sub/compute_symmetry_centers.py --models_path ~/Downloads/bop_eval/tless/models_eval --output tless_centers.json

The output JSON ({obj_id: [Px, Py, Pz]} in mm, only objects above --min_shift) gives the values of PALIGNED_VALUES in
sub/process_models.py and of paligned_values / aligned_offsets in the configs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def surface_centroid(ply_path, n=20000, seed=0):
    mesh = trimesh.load(str(ply_path))
    if hasattr(mesh, 'faces') and len(mesh.faces):
        points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        return points.mean(axis=0)
    return np.asarray(mesh.vertices).mean(axis=0)


def symmetry_axis(info):
    """Return (point on the axis, unit axis direction, max residual in mm) or None for asymmetric objects."""
    if info.get('symmetries_continuous'):
        sym = info['symmetries_continuous'][0]
        axis = np.asarray(sym['axis'], dtype=np.float64)
        return np.asarray(sym['offset'], dtype=np.float64), axis / np.linalg.norm(axis), 0.0
    if info.get('symmetries_discrete'):
        mats = [np.asarray(s, dtype=np.float64).reshape(4, 4) for s in info['symmetries_discrete']]
        A = np.concatenate([np.eye(3) - m[:3, :3] for m in mats])
        b = np.concatenate([m[:3, 3] for m in mats])
        point = np.linalg.lstsq(A, b, rcond=None)[0]
        _, sv, vt = np.linalg.svd(A)
        axis = vt[-1] if sv[-1] < 1e-6 * sv[0] else None
        return point, axis, float(np.abs(A @ point - b).max())
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models_path', required=True, help='BOP models folder (models_info.json + obj_XXXXXX.ply)')
    parser.add_argument('--min_shift', type=float, default=0.02,
                        help='report objects whose symmetry axis is farther than this fraction of the diameter from the origin')
    parser.add_argument('--output', help='write {obj_id: P} for the reported objects to this JSON file')
    args = parser.parse_args()

    models_path = Path(args.models_path).expanduser()
    models_info = json.loads((models_path / 'models_info.json').read_text())

    centers = {}
    for obj_id in sorted(models_info, key=int):
        info = models_info[obj_id]
        found = symmetry_axis(info)
        if found is None:
            continue
        point, axis, residual = found
        kind = 'continuous' if info.get('symmetries_continuous') else f"discrete x{len(info['symmetries_discrete']) + 1}"
        if axis is not None:
            centroid = surface_centroid(models_path / f'obj_{int(obj_id):06d}.ply')
            point = point + axis * ((centroid - point) @ axis)
            dist = np.linalg.norm(point - axis * (point @ axis))  # distance of the axis from the origin
        else:
            dist = np.linalg.norm(point)  # the symmetries fix a single point
        ratio = dist / info['diameter']
        flag = ratio > args.min_shift
        print(f"obj {int(obj_id):3d} {kind:14s} axis {np.round(axis, 3) if axis is not None else '-'} "
              f"P {np.round(point, 3)} | axis {dist:6.2f} mm ({ratio:.3f} x diam) from the origin | "
              f"residual {residual:.4f} mm{'  <-- shift' if flag else ''}")
        if flag:
            centers[str(int(obj_id))] = [round(float(v), 6) for v in point]

    print(f'\n{len(centers)} object(s) above {args.min_shift:.0%} of the diameter: {sorted(centers, key=int)}')
    if args.output:
        Path(args.output).write_text(json.dumps(centers, indent=2) + '\n')
        print(f'Written: {args.output}')


if __name__ == '__main__':
    main()
