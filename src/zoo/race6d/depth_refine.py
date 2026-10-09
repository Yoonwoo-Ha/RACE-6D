"""
Test-time depth refinement of predicted translations.

The network's pose is kept except for its distance: every selected prediction is moved along its viewing ray so that
the visible side of its model points agrees with the measured depth. Rotation and 2D position are unchanged.
"""

import numpy as np
import torch


__all__ = ['refine_translations_with_depth', 'sample_model_points']


def sample_model_points(ply_path, num_points=3000, seed=0):
    """Area-weighted uniform samples on a mesh surface (vertices if the file has no faces), [num_points, 3] float32."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(ply_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if not mesh.has_triangles():
        return vertices[rng.choice(len(vertices), num_points, replace=len(vertices) < num_points)].astype(np.float32)
    tri = vertices[np.asarray(mesh.triangles)]                      # [F, 3, 3]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    face = rng.choice(len(tri), num_points, p=area / area.sum())
    a, b = rng.random((2, num_points, 1))
    flip = (a + b) > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    t = tri[face]
    return (t[:, 0] + a * (t[:, 1] - t[:, 0]) + b * (t[:, 2] - t[:, 0])).astype(np.float32)


def _refine_chunk(points, labels, R, t, K, depth, image_index, valid, cell: int, window: int, max_diff_mm: float,
                  front_tol_mm: float, iters: int, min_points: int):
    """Fixed-shape core (no host sync, so it can be compiled and captured in a CUDA graph)."""
    P = points[labels].float()                                    # [M, N, 3]
    R, t, K, depth = R.float(), t.float(), K.float(), depth.float()
    image_index = image_index.long()
    H, W = depth.shape[-2:]
    M = t.shape[0]
    fx, fy = K[:, 0, 0, None], K[:, 1, 1, None]
    cx, cy = K[:, 0, 2, None], K[:, 1, 2, None]
    det = torch.arange(M, device=t.device)[:, None]

    for _ in range(iters):
        X = P @ R.transpose(1, 2) + t[:, None, :]                  # [M, N, 3]
        z = X[..., 2]
        zs = z.clamp(min=1e-3)
        u = torch.round(X[..., 0] / zs * fx + cx).long()
        v = torch.round(X[..., 1] / zs * fy + cy).long()
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        u, v = u.clamp(0, W - 1), v.clamp(0, H - 1)

        # Visible side: nearest point per (prediction, grid cell). Each prediction gets a window x window grid
        # anchored at its projection; objects wider than window * cell px use proportionally larger cells.
        u0 = torch.where(inside, u, W).amin(dim=1, keepdim=True)
        v0 = torch.where(inside, v, H).amin(dim=1, keepdim=True)
        extent = torch.maximum(torch.where(inside, u, -1).amax(dim=1, keepdim=True) - u0,
                               torch.where(inside, v, -1).amax(dim=1, keepdim=True) - v0) + 1
        c = torch.clamp((extent + window - 1) // window, min=cell)
        gu = ((u - u0) // c).clamp(0, window - 1)
        gv = ((v - v0) // c).clamp(0, window - 1)
        slot = det * (window * window) + gv * window + gu
        nearest = torch.full((M * window * window,), float('inf'), device=z.device)
        nearest = nearest.scatter_reduce(0, slot.flatten(), torch.where(inside, z, torch.inf).flatten(), 'amin')
        front = inside & (z <= nearest[slot] + front_tol_mm)

        measured = depth[image_index[:, None], v, u]               # [M, N]
        diff = measured - z
        use = front & (measured > 0) & (diff.abs() < max_diff_mm)

        # Lower median of the usable differences (unusable ones sorted to the end)
        n = use.sum(dim=1)
        ordered = torch.where(use, diff, torch.inf).sort(dim=1).values
        shift = ordered.gather(1, ((n - 1).clamp(min=0) // 2)[:, None]).squeeze(1)
        ok = (n >= min_points) & valid
        tz = t[:, 2]
        factor = torch.where(ok, (tz + torch.where(ok, shift, torch.zeros_like(shift))) / tz, torch.ones_like(tz))
        t = t * factor[:, None]
    return t


_compiled = None


@torch.no_grad()
def refine_translations_with_depth(
    rotations,
    translations,
    labels,
    cam_K,
    depth_mm,
    points,
    image_index,
    valid=None,
    cell=3,
    window=128,
    max_diff_mm=50.0,
    front_tol_mm=5.0,
    iters=2,
    min_points=15,
    max_cells=1 << 24,
    compile=False,
    device='auto',
):
    """Shift each selected translation along its ray by the median (measured - predicted) depth of its visible points.

    Per prediction and iteration: the model points are posed and projected; on a grid of `cell` px only the points
    nearest to the camera are kept (the visible side; the grid is a `window` x `window` patch over the projection, so
    objects wider than window * cell px use larger cells); their depth is compared with `depth_mm` at their pixels, pairs
    more than `max_diff_mm` apart (occluders, holes, background) are dropped, and the translation is scaled by
    (t_z + median difference) / t_z. Predictions with fewer than `min_points` usable points keep their translation.

    Args:
        rotations: [M, 3, 3] egocentric rotations (model -> camera).
        translations: [M, 3] translations in mm, in the frame of `points`.
        labels: [M] long, row of `points` for each prediction.
        cam_K: [M, 3, 3] intrinsics of each prediction's image.
        depth_mm: [B, H, W] measured depth in mm (0 = no measurement), any dtype.
        points: [C, N, 3] model points in mm.
        image_index: [M] long, image (row of `depth_mm`) of each prediction.
        valid: optional [M] bool, predictions to refine (e.g. score >= threshold); the others are returned unchanged.
            On GPU all M predictions are processed with fixed shapes (no host sync) and `valid` only masks the update;
            on CPU only the valid ones are processed.
        max_cells: bound of the visibility buffer (predictions x window^2 cells) per chunk of predictions.
        compile: run the core through torch.compile on GPU (first call compiles; recompiles when shapes change);
            ignored on CPU.
        device: 'auto' runs where `translations` are; 'cpu' or 'cuda' moves the inputs there and the result back.

    Returns:
        [M, 3] refined translations (same dtype as `translations`).
    """
    M = translations.shape[0]
    if M == 0:
        return translations.clone()
    if valid is None:
        valid = torch.ones(M, dtype=torch.bool, device=translations.device)

    home = translations.device
    run_on = home if device in (None, 'auto') else torch.device(device)
    if run_on != home:
        rotations, translations, labels, cam_K, depth_mm, points, image_index, valid = (
            x.to(run_on) for x in (rotations, translations, labels, cam_K, depth_mm, points, image_index, valid))
    options = dict(cell=cell, window=window, max_diff_mm=max_diff_mm, front_tol_mm=front_tol_mm, iters=iters,
                   min_points=min_points, max_cells=max_cells)

    if run_on.type == 'cuda':
        out = _refine_fixed(rotations, translations, labels, cam_K, depth_mm, points, image_index, valid,
                            compile=compile, **options)
    else:
        # CPU: skip the predictions that are not refined (most of them below the score threshold)
        out = translations.clone()
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = _refine_fixed(rotations[idx], translations[idx], labels[idx], cam_K[idx], depth_mm, points,
                                     image_index[idx], valid[idx], compile=False, **options)
    return out.to(home)


def _refine_fixed(rotations, translations, labels, cam_K, depth_mm, points, image_index, valid, cell, window,
                  max_diff_mm, front_tol_mm, iters, min_points, max_cells, compile):
    """All predictions with fixed shapes, in chunks bounded by `max_cells`."""
    M = translations.shape[0]
    chunk = max(1, min(M, max_cells // (window * window)))
    pad = (-M) % chunk
    R, t, K = rotations, translations, cam_K
    if pad:
        R, t, K, labels, image_index = (torch.cat([x, x[:1].expand(pad, *x.shape[1:])])
                                        for x in (R, t, K, labels, image_index))
        valid = torch.cat([valid, valid.new_zeros(pad)])

    fn = _refine_chunk
    if compile:
        global _compiled
        if _compiled is None:
            _compiled = torch.compile(_refine_chunk, dynamic=False)
        fn = _compiled
    args = (cell, window, float(max_diff_mm), float(front_tol_mm), iters, min_points)
    if chunk == M + pad:
        out = fn(points, labels, R, t, K, depth_mm, image_index, valid, *args)
    else:
        out = torch.cat([
            fn(points, labels[i:i + chunk], R[i:i + chunk], t[i:i + chunk], K[i:i + chunk], depth_mm,
               image_index[i:i + chunk], valid[i:i + chunk], *args)
            for i in range(0, M + pad, chunk)
        ])
    return out[:M].to(translations.dtype)
