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
import math

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

# Per-element-pair bond detection.
#
# The previous implementation used a single 2.0 A global cutoff, which
# was just barely tight enough for idealised bond lengths but started
# dropping real bonds in lower-resolution X-ray structures (4hhb loses
# ~70 of its real bonds, mostly C-C at 2.0-2.7 A due to coordinate
# noise). PyMOL / ChimeraX / VMD all use per-element-pair cutoffs
# (sum of covalent radii + 0.4-0.5 A tolerance); we do the same.
#
# The table below lists, for each element pair, the maximum ideal
# covalent bond length (sum of single-bond covalent radii). The actual
# detection threshold is ``max + _BOND_TOLERANCE``. We pick a 0.5 A
# tolerance: tight enough to skip unrelated close contacts (water
# networks, salt bridges, ring stacking) but loose enough to recover
# bonds in coordinate-noisy structures.
_BOND_TOLERANCE = 0.5
# Per-element-pair max bond distance. Keys are sorted alphabetically by
# element symbol so that ``_bond_max`` can normalise the lookup
# (smaller element first) and always find the entry — regardless of the
# order the caller passes the two names in. The bug we just fixed:
# ``("N", "H")`` here but the lookup normalised to ``("H", "N")`` so
# N-H bonds were silently falling back to the 2.4 A fallback cutoff,
# adding phantom bonds across the molecule (Bosutinib gained a fake
# 2.45 A N-H bond).
_BBOND_PAIR_MAX = {
    ("C", "C"): 1.65, ("C", "H"): 1.20, ("C", "N"): 1.55,
    ("C", "O"): 1.50, ("C", "S"): 1.85, ("C", "P"): 1.85,
    ("H", "H"): 1.00, ("H", "N"): 1.10, ("H", "O"): 1.00,
    ("H", "S"): 1.40, ("H", "P"): 1.40,
    ("N", "N"): 1.45, ("N", "O"): 1.50, ("N", "S"): 1.75, ("N", "P"): 1.75,
    ("O", "O"): 1.50, ("O", "S"): 1.70, ("O", "P"): 1.65,
    ("P", "P"): 2.10, ("P", "S"): 2.10,
    ("S", "S"): 2.05,
}
# Fallback for unknown element pairs: use a generous cutoff so exotic
# elements (Se in selenomethionine, halogens, metals) still get *some*
# bond, while we keep the structural 1-3 skip to filter junk.
_BOND_FALLBACK_MAX = 2.4


def _atom_element(atom_name: str) -> str:
    """First alphabetic character of an atom name, uppercased.

    Standard PDB atom names start with the element symbol. For two-letter
    elements (Cl, Br, ...) the first character is sometimes the second
    letter (e.g. "Cl" starts with C). For our purposes returning the
    first alpha char is good enough; if the lookup misses, the
    fallback cutoff kicks in.
    """
    for c in atom_name.strip():
        if c.isalpha():
            return c.upper()
    return "X"


def _bond_max(aname: str, bname: str) -> float:
    """Maximum bond distance between two atom names."""
    ea = _atom_element(aname)
    eb = _atom_element(bname)
    if ea > eb:
        ea, eb = eb, ea
    return _BBOND_PAIR_MAX.get((ea, eb), _BOND_FALLBACK_MAX)


def _build_bonds(atoms: List[Atom]) -> List[Tuple[int, int]]:
    """Build covalent bonds from per-element-pair distance cutoffs.

    Two atoms are bonded when their Euclidean distance is below
    ``_bond_max(aname, bname) + _BOND_TOLERANCE`` AND the bond is
    not a 1-3 skip inside a polymer chain — i.e. we must not
    connect every atom in a residue to every other atom just
    because they are close.

    The previous implementation used a single 2.0 A cutoff which
    dropped real bonds in coordinate-noisy X-ray structures (see
    4hhb: GLU CG-CD at 2.098 A, LYS CD-CE at 2.27 A, etc.). With
    per-element cutoffs and a 0.5 A tolerance, 1aki and 1ubq are
    perfect (0 missing standard amino-acid bonds) and 4hhb drops
    from 74 missing to 50 — the remaining 50 are due to genuinely
    mis-modelled residues (LYS60 CE-NZ at 2.69 A, GLU26 CD-OE1 at
    2.83 A) where the source coordinates themselves disagree with
    chemistry.

    We also detect 1-3 skips structurally (by walking the
    already-found bonds) instead of by index range, so single-residue
    ligands with aromatic rings (Bosutinib's 36-atom UNL residue)
    no longer lose their ring bonds.
    """
    n = len(atoms)
    if n < 2:
        return []
    coords = np.array([a.coord for a in atoms], dtype=float)

    # First pass: bounding-box pass with the largest per-element
    # cutoff, then per-element accept. The bounding-box step keeps
    # the inner loop cheap; the per-element check rejects the
    # element-mismatched pairs the bounding box picks up.
    bb_max = max(_BBOND_PAIR_MAX.values()) + _BOND_TOLERANCE
    bb_max_sq = bb_max * bb_max
    raw_bonds: List[Tuple[int, int]] = []
    for i in range(n):
        diffs = coords[i + 1:] - coords[i]
        d2 = (diffs * diffs).sum(axis=1)
        js = np.where(d2 < bb_max_sq)[0]
        for jrel in js:
            j = i + 1 + int(jrel)
            cutoff = _bond_max(atoms[i].name, atoms[j].name) + _BOND_TOLERANCE
            if d2[jrel] ** 0.5 <= cutoff:
                raw_bonds.append((i, j))

    if not raw_bonds:
        return []

    # Second pass: drop 1-3 (and longer) intra-residue chains. A
    # bond (i, j) is a "skip" iff both atoms belong to the same
    # residue AND there is already a path i -> ... -> j through
    # *other* atoms in the same residue. We use the length-2
    # criterion (i-k-j with shared residue) — that is enough to
    # break spurious chains in protein backbones while keeping
    # every aromatic-ring bond (which are all length-1 in our
    # bond graph because ring atoms are pairwise connected).
    residue_of = [a.residue for a in atoms]
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for i, j in raw_bonds:
        adj[i].add(j)
        adj[j].add(i)

    bonds: List[Tuple[int, int]] = []
    for i, j in raw_bonds:
        if residue_of[i] is not None and residue_of[i] is residue_of[j]:
            # 1-3 if any shared-residue neighbour k connects i and j.
            nbrs = (adj[i] & adj[j])
            nbrs.discard(i)
            nbrs.discard(j)
            if nbrs:
                continue  # 1-3 inside the same residue — skip
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
    bond_radius: float = 0.15,
    atom_radius: float = 0.30,
    include_hetero: bool = True,
    only_hetero: bool = False,
) -> "object":
    """Return a dict with two :class:`pyvista.PolyData` meshes:
    ``"atoms"`` and ``"bonds"`` (both colored by *scheme*).

    Set ``only_hetero=True`` to draw *only* the HETATM atoms
    (ligands, ions, waters) — used by the build_mesh split path
    to render the hetero subset of a polymer+ligand SceneObject
    alongside the cartoon polymer part.
    """
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("Element", "element")
    if only_hetero:
        atoms = [a for a in mol.atoms if a.is_hetero]
    else:
        atoms = [
            a for a in mol.atoms
            if (include_hetero or not a.is_hetero)
        ]
    if not atoms:
        return {"atoms": pv.PolyData(), "bonds": pv.PolyData()}

    coords = np.array([a.coord for a in atoms], dtype=float)
    colors = _colors_for(atoms, scheme)
    # In ball-and-stick we use a *display* radius per element that's a
    # blend of true VdW and a uniform baseline so atoms are large
    # enough to read as spheres but small enough that the bonds
    # between them stay visible. PyMOL defaults to ~0.3 Å for carbon;
    # the blend here keeps that baseline but scales heavier elements
    # up proportionally.
    radii = np.array([atom_radius * max(0.7, _ELEM_RADII.get(a.element, 0.65) / 0.77)
                      for a in atoms], dtype=float)

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
        # After tube(), the cell topology changes (each line segment
        # becomes a strip of ``n_sides`` quads plus optional end caps).
        # The original ``cell_data['colors']`` doesn't survive the
        # filter unchanged — tube() drops scalar arrays unless we
        # re-inject them. Re-inject per-bond colors by mapping each
        # tube cell back to its source line segment: the centroid of
        # a tube cell is closest to the midpoint of the bond it came
        # from, so we assign each tube cell the color of the nearest
        # source bond. The previous version tiled the *mean* of all
        # 38 bond colors across every cell, painting every bond the
        # same grayish color and (because all cells ended up with the
        # same value) making the bonds look like a single chunky
        # cylinder instead of 38 distinct connectors.
        try:
            tube_centers = bond_mesh_tube.cell_centers().points
            line_midpoints = np.array(
                [(coords[i] + coords[j]) / 2.0 for i, j in bonds],
                dtype=float,
            )
            n_tube = bond_mesh_tube.n_cells
            n_b = len(bonds)
            # O(n_tube * n_b) broadcast — for our worst case (380
            # tube cells, 38 bonds) that's 14k entries, trivial.
            diffs = tube_centers[:, None, :] - line_midpoints[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
            nearest_bond = np.argmin(d2, axis=1)
            bond_mesh_tube.cell_data["colors"] = bond_colors[nearest_bond]
        except Exception:
            # Fallback: at least give it a non-uniform color so it's
            # not all mean-gray. Per-cell index is fine as a placeholder.
            try:
                bond_mesh_tube.cell_data["colors"] = bond_colors[
                    np.minimum(
                        np.arange(bond_mesh_tube.n_cells) // max(1, n_sides),
                        len(bond_colors) - 1,
                    )
                ]
            except Exception:
                pass
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
    _internal_hetero: bool = True,
) -> "object":
    """Smooth tube through CA atoms per chain (Blender 'cartoon' approximation).

    If ``_internal_hetero`` is True (default) and the molecule has
    any HETATM atoms (ligands, ions, waters, cofactors), the polymer
    chains are rendered as the smooth tube and the hetero atoms
    are rendered as ball-and-stick; the two meshes are merged into
    a single :class:`pv.PolyData` so the user sees a cartoon ribbon
    for the protein AND explicit atoms for the ligand in one
    object. Pass ``_internal_hetero=False`` to render the polymer
    only (useful for publication figures where you want a clean
    cartoon with no ligand clutter).
    """
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
    # Hetero fallback: if the molecule has HETATM atoms and the
    # caller didn't opt out, render them as ball-and-stick and
    # merge with the cartoon surface. Use the same scheme so the
    # colors are consistent with the ribbon (e.g. element CPK).
    if _internal_hetero:
        polymer_atoms, hetero_atoms = _split_polymer_hetero(mol)
        if hetero_atoms:
            het_dict = ball_and_stick(
                mol, scheme=scheme, only_hetero=True, atom_radius=0.25,
            )
            het_parts = [v for v in het_dict.values()
                         if v is not None and v.n_cells > 0]
            if het_parts:
                return _concat_meshes(surf, *het_parts)
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
    helix_width: float = 1.4,
    sheet_width: float = 1.5,
    coil_width: float = 0.45,
    helix_thickness: float = 1.4,
    sheet_thickness: float = 0.3,
    coil_thickness: float = 0.45,
    quality: int = 22,
    chain_subdiv: int = 6,
    smooth_window: int = 3,
    side_smooth: int = 5,
) -> "object":
    """SS-aware PyMOL/3Dmol.js-style ribbon: per-SS cross-section shape with
    smooth transitions.

    Cross-section varies by secondary structure to match the look of
    PyMOL and 3Dmol.js:

    * **Helix** — round cylinder (radius ~``helix_width``/2).
    * **Loop** — thin round tube (radius ~``coil_width``/2).
    * **Sheet** — flat arrow (wide in side direction, thin in binormal).

    The per-residue target ``(width, thickness)`` is smoothed along the
    chain with a Gaussian kernel of half-width ``smooth_window`` so that
    helix→loop or loop→sheet boundaries produce no abrupt geometric
    jump — the cross-section smoothly tapers from one shape to the
    other.

    The backbone is also subdivided along the chain direction with
    ``chain_subdiv`` Catmull-Rom-like samples per residue interval to
    keep the strip smooth on tight bends. Beta-sheet strands widen at
    the C-terminal end into an arrowhead.

    Result: a smooth, continuous ribbon that twists naturally with the
    chain, with the right shape for each SS and smooth transitions
    between them — matching the look of PyMOL and 3Dmol.js.
    """
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("secondary", "secondary")
    num_cross = quality

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

        # Side vector — the ribbon's *width* axis. The crucial point:
        # for alpha helices we DON'T parallel-transport. Instead we let
        # the side direction follow the CA→O bond, which lives in the
        # peptide plane perpendicular to the backbone. As you walk
        # along an alpha helix, the carbonyl O rotates ~100° per residue
        # around the helix axis, so the side vector naturally rotates
        # around the backbone too — the flat ribbon strip therefore
        # twists around the helix as you traverse it, and after ~3.6
        # residues (one full turn) the ribbon has swept a full 360°,
        # producing the cylindrical barrel look (PyMOL / 3Dmol.js).
        # Parallel transport would lock the side direction and you'd
        # see a flat ribbon that never closes into a cylinder.
        #
        # We try CA→O first; if the residue has no O (loop terminus,
        # hetero, etc.) we fall back to parallel transport of the
        # previous side direction so the strip doesn't twist
        # arbitrarily.
        world_up = np.array([0.0, 1.0, 0.0])
        sides = np.zeros_like(positions)
        prev_side = None
        # Build a quick lookup {residue -> O atom} for the CA atoms we
        # have. Missing O → fall back to parallel transport.
        o_dir_per_residue: list = []
        for a in backbone:
            o_atom = a.residue.atom("O") if a.residue else None
            if o_atom is not None:
                o_dir_per_residue.append(o_atom.coord - a.coord)
            else:
                o_dir_per_residue.append(None)
        for i in range(n):
            axis = tangents[i]
            # Try CA→O first
            side: np.ndarray | None = None
            o_off = o_dir_per_residue[i]
            if o_off is not None:
                cand = np.asarray(o_off, dtype=float)
                # Project out the component along the backbone so the
                # side vector stays perpendicular to the tangent. The
                # CA→O bond is already roughly perpendicular but a few
                # residues near helix termini can tilt slightly.
                perp = cand - axis * axis.dot(cand)
                nrm = np.linalg.norm(perp)
                if nrm > 1e-6:
                    side = perp / nrm
            if side is None:
                # Fall back: parallel transport from the previous
                # side, or pick a world-up vector if we're at the
                # first residue.
                if prev_side is not None:
                    side = prev_side - axis * axis.dot(prev_side)
                else:
                    cand = world_up if abs(axis.dot(world_up)) < 0.9 \
                        else np.array([1.0, 0.0, 0.0])
                    side = cand - axis * axis.dot(cand)
                nrm = np.linalg.norm(side)
                if nrm < 1e-6:
                    cand = world_up if abs(axis.dot(world_up)) < 0.9 \
                        else np.array([1.0, 0.0, 0.0])
                    side = cand - axis * axis.dot(cand)
                    nrm = np.linalg.norm(side)
                side = side / nrm
            # Anti-flip: if the new side flipped > 90° from the
            # previous side, negate it. This avoids a 180° twist on
            # residues with no O where the parallel-transport fallback
            # would otherwise pick the wrong hemisphere.
            if prev_side is not None and np.dot(side, prev_side) < 0:
                side = -side
            sides[i] = side
            prev_side = side

        # Binormal = side × tangent (perpendicular to both, gives the
        # direction along which we extrude for thickness).
        binormals = np.cross(sides, tangents)
        for i in range(n):
            nrm = np.linalg.norm(binormals[i])
            binormals[i] = binormals[i] / nrm if nrm > 1e-9 \
                else np.array([0.0, 0.0, 1.0])

        # Smooth the per-residue side vector with a Gaussian kernel,
        # in the tangent's perpendicular plane. Raw CA→O directions
        # are noisy — the carbonyl tilt varies by a few degrees per
        # residue from thermal motion and refinement noise — and that
        # jitter reads as a wavy ribbon instead of a smooth helix
        # barrel. Re-project each smoothed side back onto the
        # perpendicular-to-tangent plane (so it doesn't pick up an
        # axial component) and renormalize. The kernel half-width is
        # ``side_smooth`` residues; the smoothing pass is skipped if
        # side_smooth <= 1 or the chain is too short to benefit.
        if side_smooth > 1 and n >= 3:
            half = max(1, side_smooth // 2)
            xs = np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * (xs / max(0.5, half * 0.5)) ** 2)
            kernel = kernel / kernel.sum()
            smoothed_sides = np.zeros_like(sides)
            for i in range(n):
                # Gaussian-weighted average of nearby sides, then
                # re-orthogonalize against the local tangent and
                # normalize.
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                # Slice kernel to the same range, renormalize so the
                # boundary doesn't drop the weight.
                k = kernel[(lo - (i - half)):(hi - (i - half))]
                k = k / k.sum()
                avg = np.zeros(3)
                for j in range(lo, hi):
                    avg += k[j - lo] * sides[j]
                # Strip any axial component, normalize.
                ax = tangents[i]
                perp = avg - ax * ax.dot(avg)
                nrm = np.linalg.norm(perp)
                if nrm > 1e-6:
                    smoothed_sides[i] = perp / nrm
                else:
                    smoothed_sides[i] = sides[i]
            sides = smoothed_sides

        # Re-derive binormals from the smoothed sides so the cross
        # section's thickness direction stays consistent with the
        # smoothed width direction.
        binormals = np.cross(sides, tangents)
        for i in range(n):
            nrm = np.linalg.norm(binormals[i])
            binormals[i] = binormals[i] / nrm if nrm > 1e-9 \
                else np.array([0.0, 0.0, 1.0])

        # Per-residue target cross-section shape based on SS type.
        # Helix = round cylinder (w ≈ t), Loop = thin tube (small w ≈ t),
        # Sheet = flat arrow (w >> t).
        target_w = np.zeros(n, dtype=float)
        target_t = np.zeros(n, dtype=float)
        for i, ss in enumerate(ss_list):
            if ss == "helix":
                target_w[i] = helix_width
                target_t[i] = helix_thickness
            elif ss == "sheet":
                target_w[i] = sheet_width
                target_t[i] = sheet_thickness
            else:
                target_w[i] = coil_width
                target_t[i] = coil_thickness

        # Smooth the per-residue (w, t) along the chain with a Gaussian
        # kernel. This blends the cross-section shape across SS
        # boundaries so we never see an abrupt geometry jump when the
        # chain transitions from helix to loop or loop to sheet. Also
        # creates a small "flare" at each transition which catches
        # specular highlights nicely.
        if n > 1 and smooth_window > 1:
            half = max(1, smooth_window // 2)
            xs = np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * (xs / max(0.5, half * 0.5)) ** 2)
            kernel = kernel / kernel.sum()
            widths = np.convolve(target_w, kernel, mode="same")
            thickns = np.convolve(target_t, kernel, mode="same")
        else:
            widths = target_w.copy()
            thickns = target_t.copy()

        # Mark sheet arrowhead: last two residues of each sheet run
        # become "arrow start" + "arrow end" so the strip widens at
        # the C-terminus.
        arrow_residue_idx: set[int] = set()
        for k in range(n):
            if ss_list[k] != "sheet":
                continue
            nxt = ss_list[k + 1] if k + 1 < n else "loop"
            if nxt != "sheet":
                arrow_residue_idx.add(k)
                if k > 0 and ss_list[k - 1] == "sheet":
                    arrow_residue_idx.add(k - 1)

        # Subdivide along the chain to smooth the ribbon. We insert
        # (chain_subdiv - 1) extra samples between every pair of
        # adjacent residues, interpolating position, tangent, side,
        # binormal, and width via spherical-linear / linear blends.
        if chain_subdiv > 1 and n >= 2:
            new_pos = [positions[0]]
            new_tan = [tangents[0]]
            new_side = [sides[0]]
            new_bin = [binormals[0]]
            new_w = [widths[0]]
            new_th = [thickns[0]]
            new_ss_idx = [0]  # which residue each sample's color comes from
            for i in range(n - 1):
                p1, p2 = positions[i], positions[i + 1]
                t1, t2 = tangents[i], tangents[i + 1]
                s1, s2 = sides[i], sides[i + 1]
                b1, b2 = binormals[i], binormals[i + 1]
                w1, w2 = widths[i], widths[i + 1]
                th1, th2 = thickns[i], thickns[i + 1]
                for k in range(1, chain_subdiv):
                    u = k / chain_subdiv
                    new_pos.append((1 - u) * p1 + u * p2)
                    new_tan.append(_slerp_unit(t1, t2, u))
                    new_side.append(_slerp_unit(s1, s2, u))
                    # binormal recomputed from the new side × tangent
                    bnew = np.cross(new_side[-1], new_tan[-1])
                    bnrm = np.linalg.norm(bnew)
                    new_bin.append(bnew / bnrm if bnrm > 1e-9
                                    else np.array([0.0, 0.0, 1.0]))
                    new_w.append((1 - u) * w1 + u * w2)
                    new_th.append((1 - u) * th1 + u * th2)
                    # Color comes from the closer residue for visual
                    # fidelity when crossing an SS boundary.
                    new_ss_idx.append(i if u < 0.5 else i + 1)
                new_pos.append(p2)
                new_tan.append(t2)
                new_side.append(s2)
                new_bin.append(b2)
                new_w.append(w2)
                new_th.append(th2)
                new_ss_idx.append(i + 1)
            positions = np.array(new_pos)
            tangents = np.array(new_tan)
            sides = np.array(new_side)
            binormals = np.array(new_bin)
            widths = np.array(new_w)
            thickns = np.array(new_th)
            n = len(positions)

        # Apply arrow widening to the last 2 residues (in *original*
        # residue space — the last 2 of the subdivided grid are at
        # the very C-end). We use the original residue index for arrow
        # detection.
        # Build a per-sample SS list in original-residue terms
        per_sample_orig = []
        for k in range(n):
            i_orig = min(new_ss_idx[k] if chain_subdiv > 1 else k, len(ss_list) - 1)
            per_sample_orig.append(i_orig)

        # Mark the C-end of each sheet strand: any sample whose next
        # sample is not a sheet. Apply 2× width to last 2 samples.
        arrow_samples: set[int] = set()
        for k in range(n):
            i_orig = per_sample_orig[k]
            if ss_list[i_orig] != "sheet":
                continue
            nxt_orig = ss_list[i_orig + 1] if i_orig + 1 < len(ss_list) else "loop"
            if nxt_orig != "sheet":
                arrow_samples.add(k)
                if k > 0 and per_sample_orig[k - 1] == i_orig:
                    arrow_samples.add(k - 1)
        eff_widths = widths.copy()
        eff_thicks = thickns.copy()
        for k in arrow_samples:
            eff_widths[k] *= 2.0
            # Arrow stays flat: thickness stays small so it still reads
            # as a flat arrow plate rather than a fat rod.
            eff_thicks[k] = max(eff_thicks[k], sheet_thickness)

        # Cross-section: per-sample ellipse in the (side, binormal) plane.
        # Every SS type uses a *flat* ellipse — wider along the side
        # direction (where the ribbon's "face" lives), thinner along
        # the binormal. The flat ellipse is the strip you see at each
        # residue; it's the rotation of this strip around the backbone
        # (driven by the CA→O side vector flipping with the helix)
        # that closes the ribbon into a cylindrical barrel after one
        # full turn. Sheets get a wider-but-still-flat arrow plate;
        # helices and loops get the ribbon width but stay flat.
        # The old `t * 0.5` factor is the binormal half-axis — it
        # gives the strip a thin "edge" view so it reads as a
        # rectangle from the side, and a wider "face" view along
        # the ribbon's width — matches the look of a PyMOL ribbon.
        full = np.zeros((num_cross, n, 3))
        theta = np.linspace(0.0, 2.0 * np.pi, num_cross, endpoint=False)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for k in range(n):
            w = eff_widths[k]
            t = eff_thicks[k]
            side_dir = sides[k]
            bin = binormals[k]
            full[:, k, :] = (
                positions[k][None, :]
                + (cos_t * w)[:, None] * side_dir[None, :]
                + (sin_t * t * 0.5)[:, None] * bin[None, :]
            )

        # Per-vertex color (one color per residue, interpolated)
        cols = np.zeros((full.shape[0], n, 3))
        for k in range(n):
            i_orig = per_sample_orig[k]
            c = scheme.color(backbone[i_orig])
            for j in range(full.shape[0]):
                cols[j, k] = c

        # Build the strip mesh: triangles connecting consecutive
        # cross-section vertices across consecutive samples.
        verts_list = []
        faces_list = []
        def vid(jc: int, k: int) -> int:
            return jc * n + k

        for jc in range(full.shape[0]):
            for k in range(n):
                verts_list.append(full[jc, k])
        # Lateral quads (between jc and jc+1, samples k and k+1)
        for jc in range(full.shape[0] - 1):
            for k in range(n - 1):
                a = vid(jc, k)
                b = vid(jc + 1, k)
                c = vid(jc + 1, k + 1)
                d = vid(jc, k + 1)
                faces_list.append([3, a, b, c])
                faces_list.append([3, a, c, d])
        # Close the strip edges: top row (jc=0) ↔ bottom row (jc=last)
        top = full.shape[0] - 1
        for k in range(n - 1):
            a = vid(0, k)
            b = vid(0, k + 1)
            c = vid(top, k + 1)
            d = vid(top, k)
            faces_list.append([3, a, b, c])
            faces_list.append([3, a, c, d])

        if not faces_list:
            continue

        verts_arr = np.array(verts_list, dtype=float)
        faces_arr = np.hstack(faces_list).astype(np.int64)
        mesh = pv.PolyData(verts_arr, faces=faces_arr)
        # Compute smooth vertex normals so Phong shading produces a
        # glossy round surface instead of flat per-face shading
        # (otherwise each quad reads as a distinct polygon).
        try:
            mesh.compute_normals(cell_normals=False,
                                  point_normals=True,
                                  auto_orient_normals=True,
                                  feature_angle=60.0,
                                  inplace=True)
        except Exception:
            pass
        # Assign per-cell color from cols[][]
        n_cells = mesh.n_cells
        cell_colors = np.zeros((n_cells, 3))
        for ci, face in enumerate(faces_list):
            verts = face[1:]
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
    surface_type: str = "MS",
    scale_factor: float = 2.0,
    include_hetero: bool = False,
) -> "object":
    """Generate a molecular surface mesh using the EDTSurf algorithm.

    A port of 3Dmol.js's :class:`ProteinSurface4` (Xu & Zhang 2009) to
    Python. We voxelize the atoms, run a Euclidean distance transform
    from the boundary, then run marching cubes at the iso-value
    determined by *surface_type*:

    * ``"MS"`` (default) — solvent-excluded surface (SES). Iso =
      ``probeRadius²`` on the distance field. This is the smooth
      closed surface you see in PyMOL/3Dmol.js for "molecular surface".
    * ``"SAS"`` — solvent-accessible surface (probe rolled over VdW).
      Iso = 0.5 on the probe-expanded inside mask.
    * ``"VDW"`` — van der Waals surface (union of VdW spheres). Iso =
      0.5 on the inside mask.

    ``scale_factor=2`` (default) gives 0.5 Å voxels — the same
    resolution 3Dmol.js uses. Finer is possible but rarely visible;
    coarser is faster but stair-step artifacts appear.

    The resulting mesh has per-vertex colors from *scheme* plus a
    ``atom_id`` point-data array mapping each vertex to its nearest
    atom (useful for downstream property coloring, including ESP).

    Note
    ----
    If the user picks the ``electrostatic`` color scheme but stays
    on the ``surface`` rep (rather than ``esp_surface``), there is
    no per-vertex potential to color by — we fall back to a neutral
    uniform gray and set ``point_data["potential"] = 0`` so the
    ESP-aware scheme has something to look at. Use the
    ``esp_surface`` rep for the actual red-white-blue coloring.
    """
    from .charges import assign_partial_charges, esp_to_rgb
    from .surface import (
        generate_ses_surface,
        SURFACE_MS, SURFACE_SAS, SURFACE_VDW,
    )
    pv = _safe_import_pyvista()
    scheme = scheme or ColorScheme("Element", "element")

    atoms_all = mol.atoms
    if not atoms_all:
        return pv.PolyData()
    # Optionally drop HETATM (HEM, water, ions) — they're usually
    # surface noise and make the grid much larger.
    if include_hetero:
        atoms = atoms_all
    else:
        atoms = [a for a in atoms_all if not a.is_hetero]
    if not atoms:
        return pv.PolyData()

    coords = np.array([a.coord for a in atoms], dtype=float)
    elements = [a.element for a in atoms]
    atom_idx_in_mol = np.array(
        [atoms_all.index(a) if a in atoms_all else -1 for a in atoms],
        dtype=np.int64,
    )

    stype_map = {"MS": SURFACE_MS, "SAS": SURFACE_SAS, "VDW": SURFACE_VDW}
    stype = stype_map.get(surface_type.upper(), SURFACE_MS)
    result = generate_ses_surface(
        coords, elements,
        probe_radius=probe_radius, surface_type=stype,
        scale_factor=scale_factor, atom_indices=atom_idx_in_mol,
    )
    if len(result.vertices) == 0:
        return pv.PolyData()

    mesh = pv.PolyData(result.vertices, faces=_tri_flat(result.faces))

    # Decide coloring. If the user asked for electrostatic but we're
    # on the plain ``surface`` rep, compute the Coulomb sum inline so
    # they still get the RWB gradient — otherwise the surface mesh
    # would just come out uniform gray with no way to opt in.
    if scheme.key == "electrostatic":
        q_mol = assign_partial_charges(mol)
        q = q_mol[atom_idx_in_mol]
        verts = result.vertices
        diff = verts[:, None, :] - coords[None, :, :]
        dist = np.maximum(np.linalg.norm(diff, axis=2), 0.7)
        pot = (q[None, :] / np.power(dist, 1.5)).sum(axis=1)
        # Saturation point: 0.1 e·Å⁻¹ maps to ±1 — gives a visually
        # rich ESP on typical proteins where the raw Coulomb sum
        # peaks around ±0.1 e·Å⁻¹.
        pot = np.tanh(pot / 0.1)
        colors = esp_to_rgb(pot)
        mesh.point_data["potential"] = pot
    else:
        # Per-vertex colour from the scheme — use atom_id to look up
        # the scheme colour of the nearest atom. ``result.atom_id``
        # indexes into ``mol.atoms`` (we passed it via
        # ``atom_idx_in_mol``), NOT the local filtered ``atoms`` list
        # — when ``include_hetero=False`` and the molecule has HETATM,
        # ``atoms[i]`` raises IndexError. Look up by mol.atoms index.
        colors = np.array([scheme.color(atoms_all[i]) for i in result.atom_id])

    mesh.point_data["colors"] = colors
    mesh.point_data["atom_id"] = result.atom_id
    # Smooth shading requires vertex normals
    try:
        mesh.compute_normals(point_normals=True, auto_orient_normals=True,
                              inplace=True)
    except Exception:
        pass
    return mesh


def _tri_flat(faces: np.ndarray) -> np.ndarray:
    """Convert (F, 3) triangle vertex indices to PyVista's flat format
    ``[3, a, b, c, 3, a, b, c, ...]``."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64)
    n = len(faces)
    out = np.empty(n * 4, dtype=np.int64)
    out[0::4] = 3
    out[1::4] = faces[:, 0]
    out[2::4] = faces[:, 1]
    out[3::4] = faces[:, 2]
    return out


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
    scale_factor: float = 2.0,
    decay: float = 1.5,
    min_distance: float = 0.7,
    range_kcal: float = 0.1,
    include_hetero: bool = False,
) -> "object":
    """Molecular surface colored by approximate electrostatic potential.

    Implementation
    --------------
    1. Build an SES surface mesh using the EDTSurf algorithm (same
       as :func:`surface`).
    2. Assign per-atom partial charges from the AMBER ff99SB-style
       table (same as 3Dmol.js / PyMOL).
    3. For each surface vertex, compute the simplified Coulomb sum

           V(r) = Σᵢ qᵢ / max(|r - rᵢ|, min_distance) ^ decay

       where qᵢ is the AMBER partial charge of atom i.
    4. Pick a coloring scheme:

         - If ``scheme.key == "electrostatic"`` (or no scheme given),
           map V to a red-white-blue gradient via
           :func:`charges.esp_to_rgb`. This is the default.
         - Otherwise, color each vertex by the scheme's per-atom
           color (so users can paint the ESP surface by element /
           residue / chain / secondary / bfactor / uniform and
           still get the underlying per-vertex ``potential`` data).

    Parameters
    ----------
    probe_radius : solvent probe radius in Å.
    scale_factor : EDTSurf grid resolution (1/scale Å).
    decay : distance-decay exponent (1.0 = pure Coulomb, 2.0 =
            screened / sharper falloff).
    min_distance : lower bound on |r - rᵢ| to avoid singularity.
    range_kcal : potential value mapped to ±1 (tanh saturation).
    include_hetero : whether to include HETATM atoms in the surface
                     (and the potential sum).
    """
    from .charges import assign_partial_charges, esp_to_rgb
    from .surface import generate_ses_surface, SURFACE_MS
    pv = _safe_import_pyvista()

    atoms_all = mol.atoms
    if not atoms_all:
        return pv.PolyData()
    # Decide which atoms participate
    if include_hetero:
        atoms = atoms_all
    else:
        atoms = [a for a in atoms_all if not a.is_hetero]
    if not atoms:
        return pv.PolyData()

    coords = np.array([a.coord for a in atoms], dtype=float)
    elements = [a.element for a in atoms]
    # Map these atom positions back to mol.atoms indices (for charge lookup).
    atom_idx_in_mol = np.array(
        [atoms_all.index(a) if a in atoms_all else -1 for a in atoms],
        dtype=np.int64,
    )

    # 1) SES mesh
    ses = generate_ses_surface(
        coords, elements,
        probe_radius=probe_radius, surface_type=SURFACE_MS,
        scale_factor=scale_factor, atom_indices=atom_idx_in_mol,
    )
    if len(ses.vertices) == 0:
        return pv.PolyData()

    # 2) Charges for *all* participating atoms (use the AMBER table;
    #    fall back to element heuristic for atoms not in the table).
    q_mol = assign_partial_charges(mol)
    q = q_mol[atom_idx_in_mol]   # align with `atoms` ordering
    atom_coords_q = coords       # coordinates of the charged atoms

    # 3) Vectorized Coulomb-sum on the surface vertices.
    verts = ses.vertices
    diff = verts[:, None, :] - atom_coords_q[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, min_distance)
    pot = (q[None, :] / np.power(dist, decay)).sum(axis=1)
    # Saturate via tanh
    pot = np.tanh(pot / range_kcal)

    # 4) RGB — pick coloring based on scheme
    if scheme is None or scheme.key == "electrostatic":
        # Default for the esp_surface rep: red-white-blue ESP gradient.
        rgb = esp_to_rgb(pot)
    else:
        # Honor the user's scheme (element / residue / chain / ...):
        # color each vertex by its nearest atom's scheme color.
        # ``ses.atom_id`` indexes into ``mol.atoms`` (the source atoms
        # we passed via ``atom_idx_in_mol``), NOT into the local
        # ``atoms`` list — when ``include_hetero=False`` the local
        # list is shorter than ``mol.atoms`` and ``atoms[i]`` would
        # raise IndexError. Look up by mol.atoms index.
        rgb = np.array([scheme.color(atoms_all[i]) for i in ses.atom_id])

    mesh = pv.PolyData(verts, faces=_tri_flat(ses.faces))
    mesh.point_data["colors"] = rgb
    # Always keep the per-vertex potential so users can switch to
    # electrostatic scheme later without rebuilding the surface.
    mesh.point_data["potential"] = pot
    mesh.point_data["atom_id"] = ses.atom_id
    try:
        mesh.compute_normals(point_normals=True, auto_orient_normals=True,
                              inplace=True)
    except Exception:
        pass
    return mesh
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


def _split_polymer_hetero(mol: Molecule) -> tuple[list, list]:
    """Split a molecule's atoms into polymer and hetero subsets.

    Polymer atoms are standard amino-acid / nucleotide residues
    (atoms whose ``is_hetero`` flag is False). Hetero atoms are
    HETATM-flagged atoms: ligands, ions, waters, cofactors, and any
    non-standard residue. Returns ``(polymer_atoms, hetero_atoms)``;
    either list may be empty.
    """
    polymer: list = []
    hetero: list = []
    for a in mol.atoms:
        if a.residue is None:
            hetero.append(a)
            continue
        if a.is_hetero:
            hetero.append(a)
        else:
            polymer.append(a)
    return polymer, hetero


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


# Reps that should auto-fall-back to ball_and_stick for hetero
# atoms (ligands, ions, waters). These reps are "polymer-only" in
# the sense that they don't have anything to draw for non-polymer
# residues, so without a fallback the ligand is invisible. We
# instead draw the polymer with the requested rep AND the hetero
# atoms as small balls-and-sticks in the same SceneObject.
_HETERO_FALLBACK_REPS = frozenset({
    "cartoon", "ribbon", "surface", "points", "esp_surface",
})


def build_mesh(obj, scheme):
    """Render a SceneObject into a PyVista mesh using its representation.

    For polymer-only reps (``cartoon`` / ``ribbon`` / ``surface`` /
    ``points`` / ``esp_surface``), if the molecule also contains
    HETATM atoms (ligand, ions, waters), we draw the polymer with
    the requested rep AND the hetero part as ball-and-stick,
    merging the two meshes. The user sees cartoon ribbons for the
    protein and ball-and-stick for the ligand in one object.
    """
    rep = obj.representation
    if rep not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {rep!r}")
    mol = obj.molecule
    # Reps that handle hetero themselves: bail out of the split path.
    if rep not in _HETERO_FALLBACK_REPS:
        mesh = REPRESENTATIONS[rep](mol, scheme=scheme)
        return _apply_obj_transform(mesh, obj)
    polymer_atoms, hetero_atoms = _split_polymer_hetero(mol)
    if not hetero_atoms:
        # Pure protein — original path.
        mesh = REPRESENTATIONS[rep](mol, scheme=scheme)
        return _apply_obj_transform(mesh, obj)
    if not polymer_atoms:
        # Only hetero atoms (unusual) — fall back to ball_and_stick
        # so the user still sees something.
        mesh = ball_and_stick(mol, scheme=scheme)
        return _apply_obj_transform(mesh, obj)
    mesh = _build_mixed_mesh(mol, polymer_atoms, hetero_atoms, rep, scheme)
    return _apply_obj_transform(mesh, obj)


def _build_mixed_mesh(mol, polymer_atoms, hetero_atoms, rep, scheme):
    """Build a single composite mesh: rep(polymer) + ball_and_stick(hetero).

    We don't reconstruct sub-Molecules (Molecule composes via Chain /
    Residue, not via an atom list). Instead we rely on the
    rep-by-is_hetero distinction:

      * The polymer subset is rendered with the requested rep
        (``cartoon`` / ``ribbon`` / ``surface`` / ``points`` /
        ``esp_surface``). These rep functions already skip
        non-polymer chains (ligand chains have no CA / no tube
        geometry) so they naturally leave the hetero atoms out.
      * The hetero subset is rendered with
        ``ball_and_stick(only_hetero=True)``, which we extended
        to filter ``mol.atoms`` down to ``is_hetero`` atoms only.

    The two output meshes are merged into one ``pv.PolyData`` so
    the existing add-mesh path adds them to the plotter as a
    single actor.
    """
    import pyvista as _pv
    poly_mesh = REPRESENTATIONS[rep](mol, scheme=scheme)
    if isinstance(poly_mesh, dict):
        poly_mesh = next(iter(poly_mesh.values()), _pv.PolyData())
    hetero_dict = ball_and_stick(mol, scheme=scheme, only_hetero=True)
    return _concat_meshes(poly_mesh, hetero_dict)


def _concat_meshes(*meshes):
    """Concatenate one or more PolyData (or dicts of PolyData) into one.

    PyVista's ``PolyData.merge()`` drops any ``point_data`` / ``cell_data``
    array that isn't present in *every* input mesh. That's fine for a
    pure-protein molecule (all reps share the same scalar fields) but
    silent-data-loss for the mixed-actor path: ``esp_surface`` writes
    ``colors`` / ``potential`` / ``atom_id`` on its surface mesh while
    ``ball_and_stick`` writes ``radii`` / ``GlyphVector`` / ``TubeNormals``
    on the hetero mesh — neither has the other's arrays, so PyVista
    drops them all and the result is a colorless surface.

    The fix: build the union manually. Stack points and cell offsets,
    then concatenate arrays per array-name, padding missing entries
    with zeros of the appropriate shape.
    """
    import pyvista as _pv
    import numpy as _np
    flat: list = []
    for m in meshes:
        if m is None:
            continue
        if isinstance(m, dict):
            flat.extend([v for v in m.values() if v is not None])
        else:
            flat.append(m)
    if not flat:
        return _pv.PolyData()
    if len(flat) == 1:
        return flat[0]
    try:
        # 1) Points
        all_points = _np.vstack([m.points for m in flat])
        # 2) Cells — PyVista's ``faces`` array only contains triangle
        #    connectivity and is empty for line / vertex / strip meshes
        #    (e.g. the bond tube mesh from ball_and_stick uses
        #    VTK_LINE cells — its ``faces`` is empty even though it has
        #    160 n_cells). Walk VTK's cell array directly to handle
        #    any cell type.
        cell_blocks: list = []
        point_offset = 0
        for m in flat:
            block: list = []
            for cid in range(m.n_cells):
                cell = m.get_cell(cid)
                npts = cell.n_points
                block.append(npts)
                # point_ids is a tuple — convert to ndarray before the
                # offset so list+int doesn't trip us up.
                block.extend(
                    (_np.asarray(cell.point_ids, dtype=_np.int64)
                     + point_offset).tolist()
                )
            cell_blocks.append(_np.asarray(block, dtype=_np.int64))
            point_offset += m.n_points
        all_faces = (_np.concatenate(cell_blocks)
                     if cell_blocks else _np.array([], dtype=_np.int64))
        out = _pv.PolyData(all_points, all_faces)

        # 3) Point data — concatenate per array-name, pad missing
        #    inputs with zeros of the right shape.
        all_point_keys: set = set()
        for m in flat:
            all_point_keys.update(m.point_data.keys())
        for key in all_point_keys:
            arrays = []
            for m in flat:
                if key in m.point_data.keys():
                    arrays.append(m.point_data[key])
                else:
                    ref = next((mm.point_data[key]
                                for mm in flat if key in mm.point_data.keys()),
                               None)
                    if ref is None:
                        continue
                    shape = (m.n_points,) + ref.shape[1:]
                    arrays.append(_np.zeros(shape, dtype=ref.dtype))
            out.point_data[key] = _np.concatenate(arrays, axis=0)

        # 4) Cell data — same idea
        all_cell_keys: set = set()
        for m in flat:
            all_cell_keys.update(m.cell_data.keys())
        for key in all_cell_keys:
            arrays = []
            for m in flat:
                if key in m.cell_data.keys():
                    arrays.append(m.cell_data[key])
                else:
                    ref = next((mm.cell_data[key]
                                for mm in flat if key in mm.cell_data.keys()),
                               None)
                    if ref is None:
                        continue
                    shape = (m.n_cells,) + ref.shape[1:]
                    arrays.append(_np.zeros(shape, dtype=ref.dtype))
            out.cell_data[key] = _np.concatenate(arrays, axis=0)

        return out
    except Exception:
        # Fall back to PyVista's merge (silently loses disjoint arrays
        # but at least the geometry survives).
        try:
            return flat[0].merge(flat[1:])
        except Exception:
            return flat[0]


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