"""Marching-cubes wrapper for the molecular-surface generator.

We delegate the actual marching-cubes pass to :func:`skimage.measure.
marching_cubes`, which is a fast, well-tested Lewiner et al. (2003)
implementation that handles the 15 ambiguous-cell cases correctly
via an extended triangle table.

This module just:

1. Builds the correct 3D scalar field from the EDT-populated voxel
   grid (``vpBits`` + ``vpDistance`` produced by ``surface.py``).
2. Picks the iso-value appropriate to the requested surface type:

     - ``SURFACE_VDW``  (1): ``inside`` mask (0 outside, 1 inside
       VdW spheres). Iso = 0.5.
     - ``SURFACE_SAS``  (3): same, but the inside mask is the
       probe-expanded one. Iso = 0.5.
     - ``SURFACE_MS``   (2): the squared-distance field. Iso =
       ``probe_radius²``.

3. Calls ``skimage.measure.marching_cubes`` with the correct spacing
   (``1 / scale_factor`` Å per voxel step).
4. Returns vertices in Angstroms and triangle faces in the
   original vertex index order.

Skimage's marching_cubes uses a Lewiner table so winding order and
normal direction are consistent across all 256 cube configurations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def marching_cubes(
    bits: np.ndarray,
    dist: np.ndarray,
    *,
    stype: int,
    probe_radius: float,
    nX: int,
    nY: int,
    nZ: int,
    scale: float,
    origin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run marching cubes on the EDT-populated voxel grid.

    Parameters
    ----------
    bits : (nX, nY, nZ) float — for VDW/SAS this is the inside mask
           (0 outside, 1 inside). Ignored for MS.
    dist : (nX, nY, nZ) float — distance (Å) from each voxel to the
           nearest boundary voxel. Used for MS (iso = probe_radius).
    stype : surface type (1=VDW, 3=SAS, 2=MS).
    probe_radius : in Angstroms.
    nX, nY, nZ : grid dimensions.
    scale : grid resolution = 1 / scale Å (e.g. scale=2 → 0.5 Å voxels).
    origin : (3,) world-coordinate origin of voxel (0, 0, 0).

    Returns
    -------
    vertices : (V, 3) float — world positions in Angstroms.
    faces    : (F, 3) int — triangle vertex indices.
    """
    from skimage.measure import marching_cubes as sk_mc

    spacing = (1.0 / scale, 1.0 / scale, 1.0 / scale)

    if stype == 2:
        # MS: iso-surface at d == probe_radius on the distance field.
        # Voxels with d < probe are "inside the SES" (closer to
        # boundary than the probe), voxels with d > probe are
        # "outside the SES".
        #
        # `dist` comes pre-shaped as (nX, nY, nZ); just use it.
        # The caller (surface.py) has already bumped outside-the-mask
        # voxels to > probe, so the iso-surface sits exactly on the
        # boundary.
        field = np.asarray(dist, dtype=np.float64)
        if field.ndim == 1:
            field = field.reshape(nX, nY, nZ)
        iso = float(probe_radius)
    else:
        # VDW / SAS: iso = 0.5 on the inside mask.
        field = np.asarray(bits, dtype=np.float64)
        if field.ndim == 1:
            field = field.reshape(nX, nY, nZ)
        iso = 0.5

    # skimage's marching_cubes returns (verts, faces, normals, values)
    # where verts are in voxel-index coordinates; multiply by spacing
    # and add origin to convert to world coordinates.
    try:
        out = sk_mc(field, level=iso, spacing=spacing,
                    allow_degenerate=False)
    except (ValueError, RuntimeError):
        # No surface found (entire volume is one side of iso).
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.int64))
    verts, faces, _normals, _values = out
    # Translate from voxel-index space to world space.
    verts = verts + np.asarray(origin, dtype=np.float64)
    return verts.astype(np.float64), faces.astype(np.int64)