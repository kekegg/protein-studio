"""Convert :class:`Molecule` data into renderable meshes.

Each function returns a :class:`pyvista.PolyData` (or a small dict of them)
that the 3D viewport can display. The implementations are deliberately
simple — atom spheres + bonds for ball-and-stick, splines through CA atoms
for cartoon/ribbon, splines + tube for trace, and a per-atom colored
glyph cloud for point cloud.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .colors import ColorScheme
from .molecule import Atom, Molecule, Residue


# ============================================================ internal helpers


def _element_radii() -> Dict[str, float]:
    """Approximate van-der-Waals radii used for ball rendering (A/V)."""
    return {
        "H":  0.25, "C":  0.77, "N":  0.75, "O":  0.73, "S":  1.02,
        "P":  1.06, "F":  0.71, "CL": 0.99, "BR": 1.14, "I":  1.33,
        "FE": 1.94, "MG": 1.73, "CA": 1.97, "ZN": 1.39, "NA": 1.54,
        "K":  2.31, "CU": 1.40, "MN": 1.39,
    }


_ELEM_RADII = _element_radii()
_BOND_CUTOFF_SQ = 1.9 * 1.9  # A/V, squared; conservative for backbone + sidechain


def _build_bonds(atoms: List[Atom]) -> List[Tuple[int, int]]:
    """Build covalent bonds from C-alpha + sidechain distance heuristics."""
    n = len(atoms)
    coords = np.array([a.coord for a in atoms], dtype=float)
    if n < 2:
        return []
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        # Cheap O(n) bounding-box pruner: only check within 2.0 A/V window
        diffs = coords[i + 1:] - coords[i]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        js = np.where(d2 < _BOND_CUTOFF_SQ)[0]
        for jrel in js:
            j = i + 1 + int(jrel)
            a, b = atoms[i], atoms[j]
            # Skip 1-3 pairs (no peptide bond to neighbor's neighbor)
            if (
                a.residue is not None
                and b.residue is not None
                and a.residue is b.residue
                and abs(a.index - b.index) > 4
            ):
                continue
            bonds.append((i, j))
    return bonds


def _colors_for(atoms: List[Atom], scheme: ColorScheme) -> np.ndarray:
    out = np.zeros((len(atoms), 3), dtype=float)
    for i, a in enumerate(atoms):
        out[i] = scheme.color(a)
    return out


def _safe_import_pyvista():
    """Defer pyvista import so the module is importable without rendering deps."""
    import pyvista as pv
    return pv


# =============================================================== representations


def ball_and_stick(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    bond_radius: float = 0.20,
    include_hetero: bool = True,
) -> "object":
    """Return a dict with two :class:`pyvista.PolyData` meshes:
    ``"atoms"`` and ``"bonds"`` (both colored by *scheme*)."""
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("Element", "element")
    atoms = [
        a for a in mol.atoms
        if (include_hetero or not a.is_hetero)
    ]
    if not atoms:
        return {"atoms": pv.PolyData(), "bonds": pv.PolyData()}

    coords = np.array([a.coord for a in atoms], dtype=float)
    colors = _colors_for(atoms, scheme)
    radii = np.array([_ELEM_RADII.get(a.element, 0.65) for a in atoms], dtype=float)

    atom_mesh = pv.PolyData(coords)
    atom_mesh.point_data["colors"] = colors
    atom_mesh.point_data["radii"] = radii
    # Glyph to spheres — pre-color the geometry so colors survive
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12, radius=1.0)
    atoms_glyph = atom_mesh.glyph(scale="radii", factor=1.0, geom=sphere)
    # PyVista glyph doesn't preserve source point data by default; we
    # re-inject colors by repeating each atom's color N-cells-per-sphere.
    try:
        cell_counts = atoms_glyph.compute_cell_sizes(length=False, area=False, volume=False)
        # vtkOriginalCellIds holds the source-point index of every cell
        src_ids = atoms_glyph.cell_data.get("vtkOriginalCellIds")
        if src_ids is None:
            # fallback: equal split
            n_cells_per_atom = atoms_glyph.n_cells // max(1, len(atoms))
            src_ids = np.repeat(np.arange(len(atoms)), n_cells_per_atom)
            src_ids = src_ids[: atoms_glyph.n_cells]
        atoms_glyph.cell_data["colors"] = colors[src_ids]
    except Exception:
        pass

    bonds = _build_bonds(atoms)
    if not bonds:
        bond_mesh = pv.PolyData()
    else:
        lines = []
        for i, j in bonds:
            lines.extend([2, i, j])
        bond_mesh = pv.PolyData(coords, lines=np.array(lines, dtype=int))
        bond_colors = np.array([colors[i] for i, _ in bonds], dtype=float)
        bond_mesh.cell_data["colors"] = bond_colors
        bond_mesh_tube = bond_mesh.tube(radius=bond_radius, n_sides=8)
        # After tube(), point counts change — keep colors as cell data
        bond_mesh_tube.cell_data["colors"] = np.tile(
            bond_colors.mean(axis=0), (bond_mesh_tube.n_cells, 1)
        )
        bond_mesh = bond_mesh_tube

    return {"atoms": atoms_glyph, "bonds": bond_mesh}


def _catmull_rom(coords: np.ndarray, n_per_seg: int = 8, closed: bool = False) -> np.ndarray:
    """Smooth spline through points (Catmull-Rom-ish)."""
    n = len(coords)
    if n < 2:
        return coords.copy()
    if n == 2:
        return np.linspace(coords[0], coords[1], n_per_seg)
    pts = np.vstack([coords, coords[-1:]]) if not closed else coords
    out = []
    for i in range(n - 1):
        p0 = coords[i - 1] if i > 0 else coords[i]
        p1 = coords[i]
        p2 = coords[i + 1]
        p3 = coords[i + 2] if i + 2 < n else coords[i + 1]
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            v = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            out.append(v)
    out.append(coords[-1])
    return np.array(out)


def _ca_per_residue(mol: Molecule) -> List[List[np.ndarray]]:
    """Group CA atoms per chain."""
    out: List[List[np.ndarray]] = []
    for chain in mol.chains:
        pts = []
        for res in chain.residues:
            for a in res.atoms:
                if a.name == "CA":
                    pts.append(a.coord)
                    break
        if pts:
            out.append(pts)
    return out


def cartoon(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    radius: float = 0.35,
) -> "object":
    """Smooth tube through CA atoms per chain (Blender 'cartoon' approximation)."""
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("Secondary", "secondary")
    chains_ca = _ca_per_residue(mol)
    if not chains_ca:
        return pv.PolyData()

    # Build per-chain per-residue proxy-attribute lists aligned with the
    # CA order. For "secondary" we hand back SS strings; for everything
    # else we hand back the chain id so ColorScheme can pick.
    chain_key_per_ca: list[list[tuple[str, str]]] = []
    for chain_pts, chain in zip(chains_ca, mol.chains):
        keys: list[tuple[str, str]] = []
        for res in chain.residues:
            if any(a.name == "CA" for a in res.atoms):
                keys.append((res.secondary_structure, chain.chain_id))
        if len(keys) != len(chain_pts):
            if not keys:
                keys = [("unknown", chain.chain_id)] * len(chain_pts)
            else:
                idx = (np.arange(len(chain_pts)) * (len(keys) - 1)
                       / max(1, len(chain_pts) - 1)).astype(int)
                keys = [keys[i] for i in idx]
        chain_key_per_ca.append(keys)

    splines: List[np.ndarray] = []
    seg_colors: List[np.ndarray] = []
    for chain_pts, keys, chain in zip(chains_ca, chain_key_per_ca, mol.chains):
        coords = np.array(chain_pts, dtype=float)
        if len(coords) < 2:
            continue
        spline = _catmull_rom(coords, n_per_seg=8)
        splines.append(spline)
        n_src = len(coords)
        n_out = len(spline)
        idx = (np.arange(n_out) * (n_src - 1) / max(1, n_out - 1)).astype(int)
        n_seg = n_out - 1
        seg_idx = idx[:n_seg]

        # Pull per-residue B-factor (pLDDT) directly so schemes like
        # AlphaFold can color by residue-level bfactor.
        ca_atoms_by_residue: list[float] = []
        for res in chain.residues:
            for a in res.atoms:
                if a.name == "CA":
                    ca_atoms_by_residue.append(float(a.bfactor))
                    break
            else:
                ca_atoms_by_residue.append(0.0)

        chain_seg_colors = np.array(
            [scheme.color(_ProxyAtomForKey(keys[i], scheme, bfactor=ca_atoms_by_residue[i]))
             for i in seg_idx],
            dtype=float,
        )
        seg_colors.append(chain_seg_colors)

    if not splines:
        return pv.PolyData()

    all_pts = np.vstack(splines)
    n_each = [len(s) for s in splines]
    seg_color_arr = np.vstack(seg_colors)
    lines = []
    base = 0
    for n in n_each:
        for _i in range(n - 1):
            lines.extend([2, base, base + 1])
            base += 1
        base += 1
    poly = pv.PolyData(all_pts, lines=np.array(lines, dtype=int))
    poly.cell_data["colors"] = seg_color_arr
    poly_tube = poly.tube(radius=radius, n_sides=10)
    surf = poly_tube.extract_surface(algorithm="dataset_surface")
    try:
        surf.cell_data["colors"] = _propagate_cell_colors(
            poly_tube, seg_color_arr, source_splines=splines
        )
    except Exception:
        pass
    return surf


class _ProxyAtomForKey:
    """Shim that exposes the right attribute for any ColorScheme.key."""
    __slots__ = ("ss", "chain_id", "bfactor")

    def __init__(self, key_pair, scheme, bfactor: float = 0.0):
        self.ss, self.chain_id = key_pair
        self.bfactor = bfactor

    @property
    def secondary_structure(self) -> str:
        return self.ss

    @property
    def element(self) -> str:
        return "C"

    @property
    def residue_name(self) -> str:
        return ""

    @property
    def chain_id_prop(self) -> str:
        return self.chain_id


def _propagate_cell_colors(tube_mesh, src_seg_colors: np.ndarray,
                          source_splines=None) -> np.ndarray:
    """For a tube mesh whose cells come from sweeping source line segments,
    assign each tube cell the color of its nearest source segment by
    comparing cell centroids.
    """
    n_cells = tube_mesh.n_cells
    if n_cells == 0:
        return src_seg_colors
    if not source_splines:
        return np.tile(src_seg_colors.mean(axis=0), (n_cells, 1))

    try:
        centers = tube_mesh.cell_centers().points
    except Exception:
        centers = tube_mesh.points

    seg_mids = []
    seg_color_idx = []
    seg_idx = 0
    for spline in source_splines:
        for k in range(len(spline) - 1):
            seg_mids.append(0.5 * (spline[k] + spline[k + 1]))
            seg_color_idx.append(seg_idx)
            seg_idx += 1
    seg_mids = np.asarray(seg_mids)
    seg_color_idx = np.asarray(seg_color_idx)

    out = np.zeros((n_cells, 3), dtype=float)
    chunk = 5000
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        diffs = centers[start:end, None, :] - seg_mids[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
        nearest = np.argmin(d2, axis=1)
        out[start:end] = src_seg_colors[seg_color_idx[nearest]]
    return out


class _ProxyAtom:
    """Tiny shim so ColorScheme can compute a color for a residue SS bucket."""
    __slots__ = ("ss", "scheme")

    def __init__(self, ss: str, scheme: ColorScheme):
        self.ss = ss
        self.scheme = scheme

    @property
    def secondary_structure(self) -> str:
        return self.ss

    @property
    def element(self) -> str:
        return "C"

    @property
    def residue_name(self) -> str:
        return ""

    @property
    def chain_id(self) -> str:
        return ""


# =====================================================================
# SS-aware ribbon (Blender/3Dmol-style)
# =====================================================================
# Builds a proper ribbon mesh where each residue's secondary structure
# gets a distinct geometry:
#   * helix   — a smooth wide cylinder segment per residue
#   * sheet   — a flat rectangular strip per residue + arrowhead at C-end
#   * loop    — a thin tube, same as the cartoon path
#
# This is the standard representation used by PyMOL / ChimeraX / 3Dmol.js.

@dataclass
class _ResFrame:
    """Per-residue local frame used by the SS-aware ribbon builder."""
    residue: Any
    ca: np.ndarray            # (3,) Cα coordinate
    axis: np.ndarray          # (3,) chain-direction unit vector
    up: np.ndarray            # (3,) perpendicular unit vector
    # Optional BB atoms used for axis estimation (may be None if missing).
    n_atom: Any = None
    c_atom: Any = None


def _slerp_unit(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical-linear interpolation between two unit vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.clip(a.dot(b), -1.0, 1.0)
    if dot > 0.9995:
        out = (1 - t) * a + t * b
    elif dot < -0.9995:
        # Antiparallel — pick any perpendicular axis to rotate around
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        out = _rotate_axis_angle(a, axis, t * np.pi)
    else:
        omega = np.arccos(dot)
        sin_o = np.sin(omega)
        out = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / sin_o
    nrm = np.linalg.norm(out)
    return out / nrm if nrm > 1e-9 else a


def _rotate_axis_angle(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation of *v* around unit *axis* by *theta* radians."""
    axis = axis / np.linalg.norm(axis)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return (v * cos_t + np.cross(axis, v) * sin_t +
            axis * axis.dot(v) * (1 - cos_t))


def _local_frames(chain_residues: List[Any]) -> List[_ResFrame]:
    """Compute one local frame per residue along a chain.

    Strategy:
      1. CA positions are required (no CA → residue dropped).
      2. Per-residue *axis* is the average of (prev_CA → CA) and
         (CA → next_CA), normalized. Falls back to a single tangent
         at chain ends.
      3. *up* is a stable perpendicular computed by Gram–Schmidt against
         the world-up vector and then propagated through the chain so
         adjacent residues don't flip (the classic ribbon "twist"
         artifact).
    """
    frames: List[_ResFrame] = []
    cas: List[np.ndarray] = []
    for res in chain_residues:
        ca = res.atom("CA")
        if ca is None:
            continue
        cas.append(np.asarray(ca.coord, dtype=float))
        frames.append(_ResFrame(
            residue=res, ca=cas[-1], axis=np.zeros(3), up=np.zeros(3),
            n_atom=res.atom("N"), c_atom=res.atom("C"),
        ))
    if not frames:
        return frames
    n = len(frames)
    # Per-residue tangent
    for i in range(n):
        if n == 1:
            t = np.array([0.0, 0.0, 1.0])
        elif i == 0:
            t = cas[1] - cas[0]
        elif i == n - 1:
            t = cas[-1] - cas[-2]
        else:
            t = 0.5 * ((cas[i + 1] - cas[i]) + (cas[i] - cas[i - 1]))
        norm = np.linalg.norm(t)
        frames[i].axis = t / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    # Per-residue perpendicular with twist propagation (parallel-transport)
    world_up = np.array([0.0, 1.0, 0.0])
    prev_up = None
    for i in range(n):
        axis = frames[i].axis
        if prev_up is None:
            # Pick the world vector least aligned with the axis
            cand = world_up if abs(axis.dot(world_up)) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = cand - axis * axis.dot(cand)
        else:
            # Remove any component parallel to the new axis — this keeps
            # the perpendicular continuous across residues (no twist).
            up = prev_up - axis * axis.dot(prev_up)
        nrm = np.linalg.norm(up)
        if nrm < 1e-9:
            # Degenerate; pick any perpendicular
            cand = world_up if abs(axis.dot(world_up)) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = cand - axis * axis.dot(cand)
            nrm = np.linalg.norm(up)
        up = up / nrm
        frames[i].up = up
        prev_up = up
    return frames


def _merge_meshes(meshes: List[Any]) -> Any:
    """Concatenate a list of PolyData / dict meshes into one PolyData.

    Each sub-mesh's ``cell_data["colors"]`` is preserved via a
    per-input offset. Returns an empty PolyData if no inputs.
    """
    pv = _safe_import_pyvista()
    if not meshes:
        return pv.PolyData()
    # Flatten any dicts
    flat = []
    for m in meshes:
        if isinstance(m, dict):
            for sub in m.values():
                flat.append(sub)
        else:
            flat.append(m)
    flat = [m for m in flat if m is not None and m.n_points > 0]
    if not flat:
        return pv.PolyData()
    if len(flat) == 1:
        return flat[0]
    out = flat[0].copy()
    for m in flat[1:]:
        out = out.merge(m)
    return out


def _helix_segment(frames: List[_ResFrame], radius: float,
                    color: tuple[float, float, float],
                    subdiv: int = 3) -> Any:
    """Build a single connected helix cylinder from a run of helix frames.

    The cylinder runs along the helix axis with width = ``radius`` and
    is built by sampling cross-section rings at every residue plus
    ``subdiv`` interpolated rings between residues, then connecting them
    with quads.
    """
    pv = _safe_import_pyvista()
    if not frames:
        return pv.PolyData()
    if len(frames) < 2:
        return pv.PolyData()

    n_sides = 12
    # Build a denser set of frames by Catmull-Rom-interpolating the
    # per-residue (position, axis, up) tuples. subdiv=3 puts 2 extra
    # rings between every pair of residue rings.
    dense: List[_ResFrame] = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        for s in range(subdiv):
            t = s / subdiv
            dense.append(_ResFrame(
                residue=a.residue,
                ca=(1 - t) * a.ca + t * b.ca,
                axis=_slerp_unit(a.axis, b.axis, t),
                up=_slerp_unit(a.up, b.up, t),
            ))
    dense.append(frames[-1])

    ring_indices: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    total_pts = 0
    for f in dense:
        ring = np.zeros((n_sides, 3))
        side = np.cross(f.axis, f.up)
        s = np.linalg.norm(side)
        if s < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
            side = side - f.axis * f.axis.dot(side)
            s = np.linalg.norm(side)
            if s < 1e-6:
                side = np.array([0.0, 1.0, 0.0])
                side = side - f.axis * f.axis.dot(side)
                s = np.linalg.norm(side)
        side = side / s
        up = np.cross(side, f.axis)
        for k in range(n_sides):
            theta = 2 * np.pi * k / n_sides
            ring[k] = f.ca + radius * (np.cos(theta) * up + np.sin(theta) * side)
        ring_indices.append(np.arange(total_pts, total_pts + n_sides))
        positions.append(ring)
        total_pts += n_sides
    all_pts = np.vstack(positions)
    # Quad connectivity between ring[i] and ring[i+1]
    quads: List[List[int]] = []
    for i in range(len(dense) - 1):
        a = ring_indices[i]
        b = ring_indices[i + 1]
        for k in range(n_sides):
            kn = (k + 1) % n_sides
            quads.append([a[k], a[kn], b[kn], b[k]])
    # Cap the ends so the cylinder is closed
    # Start cap: fan from a virtual center at the first ring
    center_first = len(all_pts)
    all_pts = np.vstack([all_pts, frames[0].ca[None, :]])
    cap_start: List[List[int]] = []
    for k in range(n_sides):
        kn = (k + 1) % n_sides
        cap_start.append([center_first, ring_indices[0][kn], ring_indices[0][k]])
    # End cap
    center_last = len(all_pts)
    all_pts = np.vstack([all_pts, frames[-1].ca[None, :]])
    cap_end: List[List[int]] = []
    for k in range(n_sides):
        kn = (k + 1) % n_sides
        cap_end.append([center_last, ring_indices[-1][k], ring_indices[-1][kn]])
    # Build faces (two triangles per quad + cap triangles)
    faces: List[List[int]] = []
    for q in quads:
        faces.append([3, q[0], q[1], q[2]])
        faces.append([3, q[0], q[2], q[3]])
    for tri in cap_start + cap_end:
        faces.append([3, tri[0], tri[1], tri[2]])
    faces_flat = np.hstack(faces).astype(np.int64) if faces else np.array([], dtype=np.int64)
    mesh = pv.PolyData(all_pts, faces=faces_flat)
    mesh.cell_data["colors"] = np.tile(color, (mesh.n_cells, 1))
    return mesh


def _sheet_segment(frames: List[_ResFrame],
                    width: float, thickness: float,
                    color: tuple[float, float, float],
                    arrowhead: bool = True,
                    subdiv: int = 2) -> Any:
    """Build a flat sheet strip with an arrowhead at the C-terminal end.

    The strip is a rectangular ribbon perpendicular to the chain axis.
    Each residue contributes ``subdiv`` quads (a thin slice of the
    sheet). An optional arrowhead (two triangles) widens the C-terminal
    end.
    """
    pv = _safe_import_pyvista()
    if not frames:
        return pv.PolyData()
    if len(frames) < 2:
        return pv.PolyData()

    # Build a denser set of frames so the strip is smooth (no kinks
    # at every residue).
    dense: List[_ResFrame] = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        for s in range(subdiv):
            t = s / subdiv
            dense.append(_ResFrame(
                residue=a.residue,
                ca=(1 - t) * a.ca + t * b.ca,
                axis=_slerp_unit(a.axis, b.axis, t),
                up=_slerp_unit(a.up, b.up, t),
            ))
    dense.append(frames[-1])

    # Centerline goes through the CA positions; the strip extends to
    # either side perpendicular to both the chain axis and the `up`
    # direction (so it lies in the plane formed by axis × up).
    pts: List[np.ndarray] = []
    left_idx: List[int] = []
    right_idx: List[int] = []
    for f in dense:
        # side = perpendicular to axis in the (axis, up) plane
        side = np.cross(f.axis, f.up)
        s = np.linalg.norm(side)
        if s < 1e-6:
            side = np.array([1.0, 0.0, 0.0])
            side = side - f.axis * f.axis.dot(side)
            s = np.linalg.norm(side)
        side = side / s
        left = f.ca - side * (width * 0.5)
        right = f.ca + side * (width * 0.5)
        left_idx.append(len(pts))
        pts.append(left)
        right_idx.append(len(pts))
        pts.append(right)

    quads: List[List[int]] = []
    for i in range(len(frames) - 1):
        quads.append([left_idx[i], left_idx[i + 1],
                     right_idx[i + 1], right_idx[i]])

    # Arrowhead at C-terminal end — extend the strip wider for ~1.5 residues
    if arrowhead and len(frames) >= 1:
        last = frames[-1]
        prev = frames[-2]
        # Arrow direction is the chain axis (continuing direction)
        axis = last.axis
        # Recompute side perpendicular to axis (use last frame's up)
        side = np.cross(axis, last.up)
        s = np.linalg.norm(side)
        side = side / s if s > 1e-6 else np.array([1.0, 0.0, 0.0])
        arrow_tip = last.ca + axis * (width * 0.8)
        # Two extra points forming the wide base of the arrowhead
        wide_left  = last.ca - side * (width * 1.2)
        wide_right = last.ca + side * (width * 1.2)
        # If the arrow is on a strand of length 1 (degenerate prev==last),
        # pull the base slightly back to avoid a zero-area triangle.
        if np.allclose(last.ca, prev.ca):
            base = last.ca - axis * (width * 0.4)
            wide_left = base - side * (width * 1.2)
            wide_right = base + side * (width * 1.2)
        wl_idx = len(pts); pts.append(wide_left)
        wr_idx = len(pts); pts.append(wide_right)
        tip_idx = len(pts); pts.append(arrow_tip)
        # Two triangles for the arrowhead
        quads.append([right_idx[-1], wr_idx, tip_idx, tip_idx])
        quads.append([tip_idx, wl_idx, left_idx[-1], left_idx[-1]])

    all_pts = np.array(pts)
    faces: List[List[int]] = []
    for q in quads:
        if q[2] == q[3]:
            faces.append([3, q[0], q[1], q[2]])
        else:
            faces.append([3, q[0], q[1], q[2]])
            faces.append([3, q[0], q[2], q[3]])
    faces_flat = np.hstack(faces).astype(np.int64) if faces else np.array([], dtype=np.int64)
    mesh = pv.PolyData(all_pts, faces=faces_flat)
    # Extrude thickness slightly so the strip has a tiny depth and
    # catches a bit of Phong shading. Duplicated verts with -thickness
    # offset along the up direction give a closed 2-sided ribbon.
    if thickness > 0 and mesh.n_points > 0:
        pts2 = mesh.points.copy()
        # Find the `up` direction for the sheet plane (use last frame's up)
        up = frames[-1].up
        # Determine which side each point sits on so the offset goes
        # inward. Use the ribbon centerline projection.
        # Simpler: offset all points by ±thickness along up, then weld.
        top = mesh.copy()
        bot = mesh.copy()
        bot.points = bot.points - up * thickness
        mesh = mesh.merge(bot)
        # Close the edges with side quads
        # (left edge = top_left vs bot_left for each original sample;
        #  top/bot share the same point indices, so we need to offset)
        # Skip edge closure for speed; the strip is visually fine as an
        # open ribbon on real proteins (matches 3Dmol.js default).
    mesh.cell_data["colors"] = np.tile(color, (mesh.n_cells, 1))
    return mesh


def _loop_segment(frames: List[_ResFrame], radius: float,
                   color: tuple[float, float, float]) -> Any:
    """Build a smooth thin tube through a run of loop frames (== cartoon).

    For a single-residue "run" (often a lone unknown-SS residue
    wedged between helices), emit nothing rather than a degenerate
    zero-length cylinder which can render as a floating disc.
    """
    if len(frames) < 2:
        pv = _safe_import_pyvista()
        return pv.PolyData()
    pv = _safe_import_pyvista()
    coords = np.array([f.ca for f in frames])
    # Build a polyline through the CA positions and tube it
    poly = pv.PolyData(coords)
    lines = np.hstack([[len(coords)], np.arange(len(coords))]).astype(np.int64)
    poly.lines = lines
    tubed = poly.tube(radius=radius, n_sides=8)
    tubed.cell_data["colors"] = np.tile(color, (tubed.n_cells, 1))
    return tubed


def _group_runs(frames: List[_ResFrame]) -> List[tuple[str, List[_ResFrame]]]:
    """Group consecutive frames sharing the same SS into runs."""
    runs: List[tuple[str, List[_ResFrame]]] = []
    cur_kind: Optional[str] = None
    cur: List[_ResFrame] = []
    for f in frames:
        ss = (f.residue.secondary_structure or "loop").lower()
        kind = "helix" if ss == "helix" else ("sheet" if ss == "sheet" else "loop")
        if kind != cur_kind:
            if cur:
                runs.append((cur_kind or "loop", cur))
            cur_kind = kind
            cur = [f]
        else:
            cur.append(f)
    if cur:
        runs.append((cur_kind or "loop", cur))
    return runs


def ribbon(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    helix_sheet_width: float = 1.3,
    coil_width: float = 0.5,
    thickness: float = 0.4,
    quality: int = 8,
) -> "object":
    """SS-aware ribbon: a flat twisted strip following the chain.

    Implements the 3Dmol.js / PyMOL "ribbon" representation. For every
    residue, ``quality`` cross-section points are laid out perpendicular
    to the chain direction spanning a width of ``helix_sheet_width``
    (for helix/sheet) or ``coil_width`` (for coil). The strip is then
    triangulated between consecutive residues, parallel-transported to
    avoid twist, and extruded slightly along the binormal so it has
    volume and catches Phong shading. Beta-sheet strands widen at the
    C-terminal end into an arrowhead by adding a parallel "arrow"
    cross-section row at twice the regular width at the last residue.

    Result: the protein backbone is rendered as a single continuous
    ribbon that twists naturally with the chain — never as a uniform
    cylinder or as disconnected disks.
    """
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("secondary", "secondary")
    num_cross = quality  # number of points across the ribbon width

    chain_meshes: List[Any] = []
    for chain in mol.chains:
        # Per-residue backbone atoms (skip residues without CA)
        backbone: List[Any] = []
        for res in chain.residues:
            ca = res.atom("CA")
            if ca is not None:
                backbone.append(ca)
        if len(backbone) < 2:
            continue

        positions = np.array([a.coord for a in backbone], dtype=float)
        n = len(positions)
        ss_list = [(a.residue.secondary_structure or "loop").lower()
                    for a in backbone]

        # Tangent per residue (avg of prev→CA and CA→next, normalized)
        tangents = np.zeros_like(positions)
        for i in range(n):
            if n == 1:
                t = np.array([0.0, 0.0, 1.0])
            elif i == 0:
                t = positions[1] - positions[0]
            elif i == n - 1:
                t = positions[-1] - positions[-2]
            else:
                t = 0.5 * ((positions[i + 1] - positions[i]) +
                           (positions[i] - positions[i - 1]))
            nrm = np.linalg.norm(t)
            tangents[i] = t / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])

        # Side vector (perpendicular to tangent, parallel-transported to
        # avoid twist between residues)
        world_up = np.array([0.0, 1.0, 0.0])
        sides = np.zeros_like(positions)
        prev_side = None
        for i in range(n):
            axis = tangents[i]
            if prev_side is None:
                cand = world_up if abs(axis.dot(world_up)) < 0.9 \
                    else np.array([1.0, 0.0, 0.0])
                side = cand - axis * axis.dot(cand)
            else:
                side = prev_side - axis * axis.dot(prev_side)
            nrm = np.linalg.norm(side)
            if nrm < 1e-6:
                cand = world_up if abs(axis.dot(world_up)) < 0.9 \
                    else np.array([1.0, 0.0, 0.0])
                side = cand - axis * axis.dot(cand)
                nrm = np.linalg.norm(side)
            sides[i] = side / nrm
            prev_side = sides[i]

        # Width per residue (helix/sheet wide, coil narrow)
        widths = np.array([
            helix_sheet_width if ss in ("helix", "sheet") else coil_width
            for ss in ss_list
        ], dtype=float)

        # Mark sheet arrowhead: last two residues of each sheet run
        # become "arrow start" + "arrow end" so the strip widens at
        # the C-terminus.
        arrow_residue_idx: set[int] = set()
        for k in range(n):
            if ss_list[k] != "sheet":
                continue
            nxt = ss_list[k + 1] if k + 1 < n else "loop"
            if nxt != "sheet":
                # End of a sheet run: mark last two residues (or last
                # one if the run has just one residue).
                arrow_residue_idx.add(k)
                if k > 0 and ss_list[k - 1] == "sheet":
                    arrow_residue_idx.add(k - 1)

        # Binormal = side × tangent (perpendicular to both, gives the
        # direction along which we extrude for thickness).
        binormals = np.cross(sides, tangents)
        for i in range(n):
            nrm = np.linalg.norm(binormals[i])
            binormals[i] = binormals[i] / nrm if nrm > 1e-9 \
                else np.array([0.0, 0.0, 1.0])

        # Build the 2D grid of strip vertices: pts[cross, length].
        # pts[j, i] = position[i] + delta_j * width[i] * side[i]
        #             ± thickness/2 * binormal[i]
        # We extrude both "above" and "below" the ribbon plane so it
        # has visible volume.
        pts = np.zeros((num_cross, n, 3))
        for i in range(n):
            w = widths[i]
            if i in arrow_residue_idx:
                w *= 2.0  # widen the strip at the arrowhead
            for j in range(num_cross):
                delta = -1.0 + j * 2.0 / (num_cross - 1)
                pts[j, i] = (positions[i]
                              + delta * w * sides[i]
                              + (0.5 - delta * 0.0) * 0.0)  # flat sheet
        # Flatten the top & bottom extrusion: simply duplicate every
        # cross-section vertex with ±thickness/2 along the binormal.
        # Build a 2*num_cross x n grid (top + bottom).
        full = np.zeros((num_cross * 2, n, 3))
        for i in range(n):
            for j in range(num_cross):
                p = pts[j, i]
                full[j, i] = p + binormals[i] * thickness * 0.5
                full[j + num_cross, i] = p - binormals[i] * thickness * 0.5

        # Per-vertex color (one color per residue, smoothed)
        cols = np.zeros((full.shape[0], n, 3))
        for i in range(n):
            c = scheme.color(backbone[i])
            for j in range(full.shape[0]):
                cols[j, i] = c

        # Build the strip mesh: triangles connecting consecutive
        # cross-section vertices across consecutive residues, plus
        # side strips closing the top and bottom edges of the ribbon.
        verts_list = []
        faces_list = []
        # Index helper: vertex index in the flat array
        def vid(jc: int, i: int) -> int:
            return jc * n + i

        # Flatten verts to a (V, 3) array
        for jc in range(full.shape[0]):
            for i in range(n):
                verts_list.append(full[jc, i])
        # Triangles between cross-section row jc and jc+1 (lateral)
        for jc in range(full.shape[0] - 1):
            for i in range(n - 1):
                a = vid(jc, i)
                b = vid(jc + 1, i)
                c = vid(jc + 1, i + 1)
                d = vid(jc, i + 1)
                faces_list.append([3, a, b, c])
                faces_list.append([3, a, c, d])
        # Close the chain ends with end-caps (triangle fans)
        for jc in range(full.shape[0] - 1):
            i = 0
            a = vid(jc, 0)
            b = vid(jc + 1, 0)
            c = vid(jc + 1, 1)
            d = vid(jc, 1)
            # Already covered above by lateral loop for i=0..1; skip.
            break
        # Closing strip edges: top edge (jc=0 ↔ jc=last) connect across
        # residues to give a side face
        top = full.shape[0] - 1
        for i in range(n - 1):
            a = vid(0, i)
            b = vid(0, i + 1)
            c = vid(top, i + 1)
            d = vid(top, i)
            faces_list.append([3, a, b, c])
            faces_list.append([3, a, c, d])

        if not faces_list:
            continue

        verts_arr = np.array(verts_list, dtype=float)
        faces_arr = np.hstack(faces_list).astype(np.int64)
        mesh = pv.PolyData(verts_arr, faces=faces_arr)
        # Assign per-cell color (interpolated between cross-section rows)
        n_cells = mesh.n_cells
        # Walk face list and assign color from cols[][] using the
        # average across the cell's vertices (approximation: use
        # the residue index of the first vertex).
        cell_colors = np.zeros((n_cells, 3))
        # Re-walk to assign colors based on residue i (use middle of
        # the quad)
        for ci, face in enumerate(faces_list):
            # face is [n, v0, v1, ..., v_{n-1}]; we only have triangles
            verts = face[1:]
            # Each vert = jc * n + i, so i = vert % n
            i_avg = sum((v % n) for v in verts) / len(verts)
            jc_avg = sum((v // n) for v in verts) / len(verts)
            i_idx = int(round(i_avg))
            i_idx = max(0, min(n - 1, i_idx))
            jc_idx = int(round(jc_avg))
            jc_idx = max(0, min(full.shape[0] - 1, jc_idx))
            cell_colors[ci] = cols[jc_idx, i_idx]
        mesh.cell_data["colors"] = cell_colors
        chain_meshes.append(mesh)

    if not chain_meshes:
        return pv.PolyData()
    return _merge_meshes(chain_meshes)


def surface(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    probe_radius: float = 1.4,
    resolution: int = 12,
) -> "object":
    """Generate a per-atom sphere-union surface using PyVista's filter.

    This is a coarse molecular surface (sum of VdW spheres expanded by
    *probe_radius*) — fast but not as smooth as a real solvent-excluded
    surface.
    """
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("Element", "element")
    atoms = mol.atoms
    if not atoms:
        return pv.PolyData()
    coords = np.array([a.coord for a in atoms], dtype=float)
    colors = _colors_for(atoms, scheme)
    radii = np.array([_ELEM_RADII.get(a.element, 0.65) + probe_radius for a in atoms])

    cloud = pv.PolyData(coords)
    cloud.point_data["colors"] = colors
    cloud.point_data["radii"] = radii
    sphere = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution, radius=1.0)
    glyph = cloud.glyph(scale="radii", factor=1.0, geom=sphere)
    try:
        surf = glyph.extract_surface(algorithm="dataset_surface")
        try:
            surf.cell_data["colors"] = surf.points_to_point_data()["colors"]
        except Exception:
            pass
    except Exception:
        surf = glyph
    return surf


def point_cloud(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    point_size: float = 5.0,
) -> "object":
    """Pure point cloud (useful for very large structures)."""
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("Element", "element")
    coords = np.array([a.coord for a in mol.atoms], dtype=float)
    if len(coords) == 0:
        return pv.PolyData()
    cloud = pv.PolyData(coords)
    cloud.point_data["colors"] = _colors_for(mol.atoms, scheme)
    return cloud


def esp_surface(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    probe_radius: float = 1.4,
    n_points: int = 4000,
) -> "object":
    """Molecular surface colored by approximate electrostatic potential.

    Implementation note: a full Coulomb sum over all atoms tends to
    paint the whole surface the same color (because backbone O dominate
    the cumulative potential). Instead, we colour each surface sphere
    by a *local* ESP — the sum over the 12 nearest atoms — which
    highlights charged patches and produces a visually rich map.
    """
    from .charges import assign_partial_charges, esp_to_rgb
    pv = _safe_import_pyvista()

    atoms = [a for a in mol.atoms if not a.is_hetero]
    if not atoms:
        return pv.PolyData()
    coords = np.array([a.coord for a in atoms], dtype=float)
    q = assign_partial_charges(mol)
    q = q[np.array([not a.is_hetero for a in mol.atoms])]

    # For each atom, sum partial charges of K nearest neighbours
    K = 12
    n = len(coords)
    if n <= K:
        per_atom_pot = q.copy()
    else:
        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        nearest = np.argpartition(dist2, K, axis=1)[:, :K]
        per_atom_pot = np.array([q[nearest[i]].sum() for i in range(n)])

    # Map to colour via tanh to keep range bounded
    per_atom_pot = np.tanh(per_atom_pot / 0.6)
    per_atom_colors = esp_to_rgb(per_atom_pot)

    # Render as sphere union
    all_coords = np.array([a.coord for a in mol.atoms], dtype=float)
    all_colors = np.zeros((len(mol.atoms), 3), dtype=float)
    all_pot = np.zeros(len(mol.atoms), dtype=float)
    # hetero atoms → neutral grey
    hetero_mask = np.array([a.is_hetero for a in mol.atoms])
    protein_mask = ~hetero_mask
    protein_indices = np.where(protein_mask)[0]
    for i, idx in enumerate(protein_indices):
        all_colors[idx] = per_atom_colors[i]
        all_pot[idx] = per_atom_pot[i]
    all_colors[hetero_mask] = (0.85, 0.85, 0.85)

    radii = np.array([_ELEM_RADII.get(a.element, 0.65) + probe_radius
                      for a in mol.atoms])
    cloud = pv.PolyData(all_coords)
    cloud.point_data["colors"] = all_colors
    cloud.point_data["potential"] = all_pot
    cloud.point_data["radii"] = radii
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12, radius=1.0)
    glyph = cloud.glyph(scale="radii", factor=1.0, geom=sphere)
    try:
        src_ids = glyph.cell_data.get("vtkOriginalPointIds")
        if src_ids is None:
            n_cells_per_atom = max(1, glyph.n_cells // max(1, len(mol.atoms)))
            src_ids = np.repeat(np.arange(len(mol.atoms)), n_cells_per_atom)
            src_ids = src_ids[: glyph.n_cells]
        glyph.cell_data["colors"] = all_colors[src_ids]
    except Exception:
        pass
    return glyph


def hbonds_representation(mol: Molecule, **kwargs) -> "object":
    """Render hbonds as dashed-yellow lines between donor and acceptor."""
    from .hbonds import detect_hbonds, hbonds_as_lines
    pv = _safe_import_pyvista()
    hbs = detect_hbonds(mol, **kwargs)
    coords, lines = hbonds_as_lines(hbs)
    if len(coords) == 0:
        return pv.PolyData()
    poly = pv.PolyData(coords, lines=lines)
    color = np.tile(np.array([0.95, 0.85, 0.20], dtype=float), (len(hbs), 1))
    poly.cell_data["colors"] = color
    return poly


# Public registry — name -> renderer
REPRESENTATIONS: Dict[str, callable] = {
    "ball_and_stick": ball_and_stick,
    "cartoon":        cartoon,
    "ribbon":         ribbon,
    "surface":        surface,
    "points":         point_cloud,
    "esp_surface":    esp_surface,
    "hbonds":         hbonds_representation,
}


def available_representations() -> List[str]:
    return list(REPRESENTATIONS.keys())


def build_mesh(obj, scheme):
    """Render a SceneObject into a PyVista mesh using its representation.

    Centralized so callers (GUI viewport, headless API, tests) don't
    have to know about REPRESENTATIONS dict or signature differences
    between rep functions.
    """
    renderer = REPRESENTATIONS.get(obj.representation)
    if renderer is None:
        raise ValueError(f"Unknown representation: {obj.representation!r}")
    mesh = renderer(obj.molecule, scheme=scheme)
    return _apply_obj_transform(mesh, obj)


def _apply_obj_transform(mesh, obj):
    """Apply a SceneObject's transform to a mesh (or dict of meshes)."""
    import numpy as np
    M = obj.transform_matrix()
    if M is None:
        return mesh
    if isinstance(mesh, dict):
        return {k: _apply_M(v, M) for k, v in mesh.items()}
    return _apply_M(mesh, M)


def _apply_M(mesh, M):
    """Apply a 4x4 transform matrix M to a PyVista mesh in-place."""
    import numpy as np
    pts = mesh.points
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    mesh.points = (M @ pts_h.T).T[:, :3]
    return mesh