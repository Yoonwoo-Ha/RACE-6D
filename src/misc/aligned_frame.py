"""Model-frame shift P of objects whose symmetry axis is off the model origin.

Meshes, GT poses and the keypoint cache of bop_toolkit stay in the original BOP model frame. Training moves
these objects to v - P (poses t + R @ P, keypoints kp - P) and the postprocessor returns t - R @ P.
"""

import json
import os

P_VALUES_FILE = 'p_values.json'


def resolve_aligned_offsets(aligned_offsets=None, coco_path=None):
    """Return {category_id: [x, y, z] mm} for the objects with a nonzero P.

    aligned_offsets: a dict sets P explicitly (e.g. the P a released checkpoint was trained with); {} or 'none'
        trains every object in the original frame; None or 'auto' reads {coco_path}/p_values.json, written by
        bop_toolkit (scripts/prepare_race6d_models.py --install_to), or no P when the file is missing.
    """
    if isinstance(aligned_offsets, str):
        mode = aligned_offsets.lower()
        if mode == 'none':
            return {}
        if mode != 'auto':
            raise ValueError(f"aligned_offsets must be a dict, 'auto' or 'none', got {aligned_offsets!r}")
        aligned_offsets = None
    if aligned_offsets is None:
        if not coco_path:
            return {}
        path = os.path.join(os.path.expanduser(coco_path), P_VALUES_FILE)
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            aligned_offsets = json.load(f)

    offsets = {}
    for cat, p in aligned_offsets.items():
        p = [float(v) for v in p]
        if len(p) != 3:
            raise ValueError(f'P of category {cat} must have 3 values, got {p}')
        if any(p):
            offsets[int(cat)] = p
    return offsets
