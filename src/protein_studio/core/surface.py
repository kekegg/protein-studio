"""EDTSurf-style molecular surface generation.

A Python port of 3Dmol.js's :class:`ProteinSurface4` algorithm (Xu
& Zhang 2009). Generates the solvent-excluded surface (SES, aka
molecular surface) of a molecule as a triangle mesh by:

1. Voxelizing the molecule's atoms into a 3D grid (each voxel
   records whether it is inside the union of VdW+probe spheres, and
   which atom it belongs to).
2. Marking the boundary voxels (inside, but with at least one
   26-neighbour that is outside).
3. Running a Euclidean distance transform from the boundary voxels
   so every inside voxel knows its distance to the nearest outside
   voxel.
4. Running marching cubes at iso-value ``probe_radius`` on the
   distance field — three surface types are supported:

   - **VDW** (1): boundary of the union of VdW spheres (no probe)
   - **SAS** (3): boundary of VdW + probe — probe-rolled surface
   - **MS/SES** (2): iso-surface at ``d = probe_radius`` on the
     distance field — the classic solvent-excluded surface with
     concave re-entrant pockets

Each generated vertex is tagged with the source atom it belongs to
so the caller can colour the surface by any per-atom property
(element, residue, partial charge → ESP, etc.).

We delegate the actual marching-cubes pass to
:func:`skimage.measure.marching_cubes` (Lewiner et al. 2003) which
handles the 15 ambiguous-cell cases correctly.

Resolution: ``scale_factor = 2`` gives 0.5 Å voxels — same as
3Dmol.js. Finer is possible but rarely visible; coarser is faster
but stair-step artifacts appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ============================================================== VdW radius table
#
# Bondi (1964) values, the same set used by 3Dmol.js / PyMOL / NAMD.
VDW_RADII: dict[str, float] = {
    "H":  1.20, "HE": 1.40,
    "LI": 1.82, "BE": 1.53, "B": 1.92, "C": 1.70,
    "N":  1.55, "O":  1.52, "F":  1.47, "NE": 1.54,
    "NA": 2.27, "MG": 1.73, "AL": 1.84, "SI": 2.10, "P": 1.80,
    "S":  1.80, "CL": 1.75, "AR": 1.88,
    "K":  2.75, "CA": 1.97, "FE": 1.94, "CU": 1.40, "ZN": 1.39,
    "BR": 1.85, "I":  1.98, "X":  2.00,
}

# Surface type identifiers — match 3Dmol.js SurfaceType enum.
SURFACE_VDW = 1
SURFACE_SAS = 3
SURFACE_MS  = 2


@dataclass
class SESResult:
    """Output of :func:`generate_ses_surface`.

    Attributes
    ----------
    vertices : (N, 3) float array — vertex positions in Angstroms.
    faces    : (F, 3) int array — triangle vertex indices.
    atom_id  : (N,) int array — index of the nearest atom for each
               vertex. Allows downstream colouring by any atom
               property (element, residue, partialCharge, …).
    """
    vertices: np.ndarray
    faces: np.ndarray
    atom_id: np.ndarray


def _coords_to_voxel(coords: np.ndarray, scale: float, ptran: np.ndarray) -> np.ndarray:
    """Convert world coords → voxel indices (float)."""
    return 0.5 + scale * (coords + ptran)


def _assign_atoms(verts: np.ndarray, coords: np.ndarray,
                  atom_indices: np.ndarray) -> np.ndarray:
    """For each surface vertex, find the source-atom index."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, idx = tree.query(verts, k=1)
    return atom_indices[idx]


def generate_ses_surface(
    coords: np.ndarray,
    elements: List[str],
    *,
    probe_radius: float = 1.4,
    surface_type: int = SURFACE_MS,
    scale_factor: float = 2.0,
    atom_indices: Optional[np.ndarray] = None,
) -> SESResult:
    """Build a molecular surface mesh (SES, SAS, or VDW).

    Parameters
    ----------
    coords : (N, 3) array of atomic positions in Angstroms.
    elements : N element symbols.
    probe_radius : solvent probe radius in Å (default 1.4).
    surface_type : one of ``SURFACE_MS``, ``SURFACE_SAS``, ``SURFACE_VDW``.
    scale_factor : grid resolution = 1/scale Å. Default 2 (0.5 Å voxels);
                   raise to 3 (0.33 Å) for finer detail, lower to 1 for
                   fast preview.
    atom_indices : optional (N,) array of "source" atom indices. When
                   the caller passes a subset of atoms (e.g. one chain
                   of a multi-chain PDB), set this so the per-vertex
                   ``atom_id`` field maps back to the original atom
                   numbering rather than the subset index. Defaults to
                   ``arange(N)``.

    Returns
    -------
    :class:`SESResult` with the triangle mesh and per-vertex atom id.
    """
    from scipy.ndimage import distance_transform_edt

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n == 0:
        return SESResult(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            atom_id=np.zeros((0,), dtype=np.int64),
        )
    if atom_indices is None:
        atom_indices = np.arange(n, dtype=np.int64)
    atom_indices = np.asarray(atom_indices, dtype=np.int64)
    scale = float(scale_factor)
    probe = float(probe_radius)

    # ---- 1) Grid setup ----
    margin = (1.0 / scale) * 5.5
    mins = coords.min(axis=0) - (probe + margin)
    maxs = coords.max(axis=0) + (probe + margin)
    pmin = np.floor(mins * scale) / scale
    pmax = np.ceil(maxs * scale) / scale
    ptran = -pmin

    nX = int(np.ceil(scale * (pmax[0] - pmin[0]))) + 1
    nY = int(np.ceil(scale * (pmax[1] - pmin[1]))) + 1
    nZ = int(np.ceil(scale * (pmax[2] - pmin[2]))) + 1
    n_voxels = nX * nY * nZ
    if n_voxels > 5_000_000:
        raise MemoryError(
            f"Grid too large ({n_voxels:,} voxels). "
            f"Reduce scale_factor (currently {scale}) or load a smaller molecule."
        )

    # ---- 2) Per-atom expanded radius (depends on surface type) ----
    if surface_type == SURFACE_VDW:
        # VDW: pure van der Waals radius — no probe expansion
        per_atom_radius = np.array([
            VDW_RADII.get(e.upper(), VDW_RADII["X"]) for e in elements
        ], dtype=np.float64)
    else:
        # SAS / MS: VdW + probe
        per_atom_radius = np.array([
            VDW_RADII.get(e.upper(), VDW_RADII["X"]) + probe
            for e in elements
        ], dtype=np.float64)

    # ---- 3) Voxelize: KDTree nearest-atom query, mark inside/outside ----
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    xs = pmin[0] + np.arange(nX) / scale
    ys = pmin[1] + np.arange(nY) / scale
    zs = pmin[2] + np.arange(nZ) / scale
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    flat_pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    dist_to_nearest, nearest_idx = tree.query(flat_pts, k=1)
    cutoff = per_atom_radius[nearest_idx]
    inside_mask = (dist_to_nearest <= cutoff).reshape(nX, nY, nZ)
    # Map each inside voxel to its owning atom (in source-atom indices)
    nearest_3d = nearest_idx.reshape(nX, nY, nZ)
    voxel_serial = np.where(inside_mask, nearest_3d, -1).astype(np.int32)
    inside_vox_serial = atom_indices[voxel_serial]  # (nX, nY, nZ) int32

    # ---- 4) Boundary = inside voxels with at least one 26-neighbour outside ----
    # Use scipy's binary_erosion on the *complement* mask to find
    # boundary voxels in one vectorized pass: a voxel is boundary iff
    # it is inside AND at least one of its 26 neighbours is outside.
    outside_mask = ~inside_mask
    # Pixels that are inside AND not "deeply interior". A pixel is
    # interior iff all its 26 neighbours are also inside. Equivalently:
    # boundary = inside AND (erode(inside) is False at this pixel).
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(inside_mask, border_value=False)
    boundary_mask = inside_mask & ~eroded

    if not boundary_mask.any():
        # Degenerate case: too few voxels to form a surface.
        return SESResult(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            atom_id=np.zeros((0,), dtype=np.int64),
        )

    # ---- 5) Euclidean distance transform on the inside mask ----
    # scipy.ndimage.distance_transform_edt returns, for each pixel,
    # the Euclidean distance to the nearest *zero* pixel. We pass
    # `inside_mask` (inside=1, outside=0), so:
    #   - For inside voxels (1): distance to nearest 0 = distance to
    #     nearest outside voxel — exactly what we want for SES.
    #   - For outside voxels (0): distance to nearest 0 = 0 (they
    #     ARE zero pixels).
    d_inside = distance_transform_edt(inside_mask, sampling=1.0 / scale)

    # ---- 6) Marching cubes ----
    from .marching_cubes import marching_cubes
    if surface_type == SURFACE_MS:
        # SES: iso = probe_radius on the distance field.
        # Outside-the-mask voxels were set to 0 by EDT — bump them
        # so they're on the "outside the SES" side of the iso.
        d_field = d_inside.copy()
        d_field[outside_mask] = probe + 1.0  # outside the SES
        # Inside voxels already have distance-to-outside; d < probe
        # means they're closer than the probe (inside SES); d > probe
        # means they're far enough (outside SES).
        field = d_field.astype(np.float64)
    else:
        # VDW / SAS: iso = 0.5 on the inside mask.
        field = inside_mask.astype(np.float64)

    verts, faces = marching_cubes(
        field, d_inside,
        stype=surface_type,
        probe_radius=probe,
        nX=nX, nY=nY, nZ=nZ,
        scale=scale,
        origin=pmin,
    )

    if len(verts) == 0:
        return SESResult(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            atom_id=np.zeros((0,), dtype=np.int64),
        )

    # ---- 7) Per-vertex atom id ----
    atom_id = _assign_atoms(verts, coords, atom_indices)
    return SESResult(
        vertices=verts.astype(np.float64),
        faces=faces.astype(np.int64),
        atom_id=atom_id,
    )