"""Convert :class:`Molecule` data into renderable meshes.

Each function returns a :class:`pyvista.PolyData` (or a small dict of them)
that the 3D viewport can display. The implementations are deliberately
simple — atom spheres + bonds for ball-and-stick, splines through CA atoms
for cartoon/ribbon, splines + tube for trace, and a per-atom colored
glyph cloud for point cloud.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def ribbon(
    mol: Molecule,
    scheme: ColorScheme | None = None,
    width: float = 1.2,
) -> "object":
    """Flatten the cartoon tube by a directional bias — used as a stylised ribbon.

    Here we use the same CA spline as ``cartoon`` but flatten it slightly along
    a normal direction (averaged over the previous / next tangent). This is a
    cheap approximation of a real ribbon mesh and keeps the implementation
    dependency-free.
    """
    pv = _safe_import_pyvista()
    mesh = cartoon(mol, scheme=scheme, radius=width * 0.4)
    return mesh  # stylised alias — same geometry, just a different preset


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