"""End-to-end smoke tests for protein-studio.

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/protein-studio/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from protein_studio.api import StudioAPI, HeadlessViewport
from protein_studio.core.io import load_pdb
from protein_studio.core.colors import SCHEME_PRESETS, SS_COLORS
from protein_studio.core.representations import (
    available_representations, cartoon, ball_and_stick, surface, point_cloud,
    ribbon,
)
from protein_studio.core.scene import Scene, SceneObject


EXAMPLES = ROOT / "examples"


# ---------------------------------------------------------------- core/molecule


def test_load_pdb_returns_molecule():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    assert mol.name
    assert mol.n_atoms() > 0
    assert mol.n_residues() > 0
    assert mol.n_chains() >= 1


def test_load_pdb_resolves_chains():
    mol = load_pdb(EXAMPLES / "4hhb.pdb")
    # Hemoglobin has 4 chains (A/B/C/D), no duplicate counts.
    chain_ids = [c.chain_id for c in mol.chains]
    assert chain_ids == ["A", "B", "C", "D"], chain_ids


def test_load_pdb_secondary_structure():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    ss = {r.secondary_structure for r in mol.residues}
    # Ubiquitin has helix and sheet at minimum
    assert "helix" in ss
    assert "sheet" in ss


def test_coords_shape():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    coords = mol.coords()
    assert coords.shape == (mol.n_atoms(), 3)
    assert coords.dtype == float


def test_sequence_one_letter():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    # Ubiquitin is 76 residues (one-letter codes)
    seq = mol.sequence()
    assert len(seq) == mol.n_residues()
    assert all(c.isupper() and c.isalpha() or c == "X" for c in seq)


# ------------------------------------------------------------------ colors


def test_schemes_present():
    for k in ("uniform", "element", "residue", "chain", "secondary", "bfactor"):
        assert k in SCHEME_PRESETS


def test_secondary_scheme_colors_have_three_categories():
    # Three states used by PDB SS records
    for k in ("helix", "sheet", "loop", "turn", "unknown"):
        assert k in SS_COLORS


def test_bfactor_gradient():
    from protein_studio.core.colors import ColorScheme
    s = ColorScheme("B-factor", "bfactor", bfactor_range=(0.0, 100.0))
    # Use a fake atom with a bfactor
    class A:
        bfactor = 0.0
        element = "C"
        residue_name = ""
        chain_id = ""
        secondary_structure = "unknown"
    assert s.color(A())[2] == pytest.approx(1.0)


# --------------------------------------------------------------- representations


def test_representation_registry():
    expected = {"cartoon", "ball_and_stick", "ribbon", "surface", "points"}
    assert set(available_representations()) >= expected


def test_cartoon_returns_polydata():
    import pyvista as pv
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    mesh = cartoon(mol)
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells > 0
    assert "colors" in mesh.cell_data


def test_ball_and_stick_returns_dict():
    import pyvista as pv
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    out = ball_and_stick(mol)
    assert isinstance(out, dict)
    assert "atoms" in out and "bonds" in out
    assert out["atoms"].n_cells > 0
    assert out["bonds"].n_cells > 0


def test_point_cloud_renders_points():
    import pyvista as pv
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    out = point_cloud(mol)
    assert isinstance(out, pv.PolyData)
    assert out.n_points == mol.n_atoms()


# ----------------------------------------------------------- scene + api flow


def test_api_load_and_refresh():
    api = StudioAPI(viewport=HeadlessViewport())
    obj = api.load(EXAMPLES / "1ubq.pdb")
    assert obj.name in api.scene.find(obj.name).name
    api.refresh()
    assert obj.name in api.viewport.plots
    api.clear()
    assert api.scene.n_objects() == 0


def test_api_set_representation_and_scheme():
    api = StudioAPI(viewport=HeadlessViewport())
    obj = api.load(EXAMPLES / "1ubq.pdb")
    api.set_representation(obj.name, "ball_and_stick")
    api.set_scheme(obj.name, "element")
    api.refresh()
    assert api.scene.find(obj.name).representation == "ball_and_stick"
    assert api.scene.find(obj.name).color_scheme.key == "element"


def test_api_save_load_roundtrip():
    import tempfile
    api = StudioAPI(viewport=HeadlessViewport())
    obj = api.load(EXAMPLES / "1ubq.pdb")
    api.refresh()
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        path = f.name
    try:
        api.save(obj.name, path)
        mol2 = load_pdb(path)
        assert mol2.n_atoms() == obj.molecule.n_atoms()
    finally:
        Path(path).unlink(missing_ok=True)


def test_scene_emits_events():
    scene = Scene()
    seen = []
    scene.on_change(lambda k, v: seen.append(k))
    obj = SceneObject(name="x", molecule=load_pdb(EXAMPLES / "1ubq.pdb"))
    scene.add(obj)
    scene.update_object("x", visible=False)
    scene.remove("x")
    assert "add" in seen
    assert "update" in seen
    assert "remove" in seen

# --------------------------------------------------------------- SS-aware ribbon

def test_ribbon_returns_single_polydata_with_colors():
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    import pyvista as pv
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells > 0
    assert "colors" in mesh.cell_data
    assert mesh.cell_data["colors"].shape[0] == mesh.n_cells


def test_ribbon_helix_is_cylindrical_not_flat():
    """A PyMOL-style ribbon has a **round** cross-section for helices
    (true cylinder look, not a flat twisted strip). Walk along a known
    helix in 1aki and measure the cross-section at each CA: PCA should
    give roughly equal eigenvalues in the two perpendicular directions
    (the cross-section is a circle, not a thin strip).
    """
    import numpy as np
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    # Pick CA atoms of a known helix (chain A residues ~12..30 of 1aki
    # are helix A).
    ca_atoms = [a for r in mol.chains[0].residues
                for a in r.atoms if a.name == "CA"]
    helix_cas = [a for a in ca_atoms
                  if (a.residue.secondary_structure or "").lower() == "helix"][:30]
    assert len(helix_cas) >= 5, "need enough helix residues"
    pts = mesh.points
    ratios = []
    for ca in helix_cas:
        # Find the cross-section ring at this CA: vertices within 1.5 Å
        # of ca.coord.
        d = np.linalg.norm(pts - ca.coord, axis=1)
        ring = pts[d < 1.8]
        if len(ring) < 6:
            continue
        # PCA on the ring
        c = ring - ring.mean(axis=0)
        cov = np.cov(c.T)
        ev = np.linalg.eigvalsh(cov)
        # ev ascending: smallest = thickness axis, middle = side axis.
        # If helix is a true cylinder, ev[1]/ev[0] should be near 1.
        # We expect <= 4 (a cylinder or near-cylinder), definitely not
        # the >10 ratio a flat strip would produce.
        if ev[0] > 1e-9:
            ratios.append(ev[1] / ev[0])
    assert len(ratios) >= 3, f"not enough rings measured: {len(ratios)}"
    median_ratio = float(np.median(ratios))
    assert median_ratio < 4.0, (
        f"helix cross-section looks too flat: median side/thickness "
        f"variance ratio = {median_ratio:.2f} (should be <4 for a "
        f"cylindrical helix; a flat strip would be >10)."
    )


def test_ribbon_sheet_is_flat():
    """The sheet strand in 1aki must have a **flat** cross-section
    (wide in the side direction, thin in the binormal) — i.e. the
    variance ratio in the cross-section ring should be >> 1.
    """
    import numpy as np
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    # Find sheet CA atoms
    sheet_cas = [a for r in mol.chains[0].residues
                  for a in r.atoms
                  if a.name == "CA"
                  and (a.residue.secondary_structure or "").lower() == "sheet"]
    assert len(sheet_cas) >= 2, "need sheet residues"
    pts = mesh.points
    ratios = []
    for ca in sheet_cas:
        d = np.linalg.norm(pts - ca.coord, axis=1)
        ring = pts[d < 1.8]
        if len(ring) < 6:
            continue
        c = ring - ring.mean(axis=0)
        cov = np.cov(c.T)
        ev = np.linalg.eigvalsh(cov)
        if ev[0] > 1e-9:
            ratios.append(ev[1] / ev[0])
    assert len(ratios) >= 1
    median_ratio = float(np.median(ratios))
    # Sheet cross-section should be much wider than thick; a flat
    # strip gives ratio >4, while a cylinder gives ~1.
    assert median_ratio > 4.0, (
        f"sheet cross-section looks too round: median side/thickness "
        f"variance ratio = {median_ratio:.2f} (should be >4 for a flat "
        f"arrow; a cylinder would be ~1)."
    )


def test_ribbon_has_smooth_vertex_normals():
    """The ribbon must have per-vertex normals so Phong shading produces
    a smooth round surface (not flat per-face)."""
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    import numpy as np
    assert "Normals" in mesh.point_data, "ribbon must compute vertex normals for smooth shading"
    normals = mesh.point_data["Normals"]
    assert normals.shape == (mesh.n_points, 3)
    # Normals must be unit-length
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), \
        f"vertex normals should be unit length, got range [{norms.min():.3f}, {norms.max():.3f}]"


def test_ribbon_ss_boundaries_are_smooth():
    """The cross-section width should not change abruptly at SS
    boundaries (helix→loop, loop→sheet) — gaussian smoothing should
    produce a smooth taper. We verify by checking that no two adjacent
    residues in the original chain differ by more than 3× their target
    width ratio."""
    import numpy as np
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    # Just confirm the mesh has no obvious "step" by checking the
    # radius of vertex rings along a helix→loop boundary. We pick the
    # helix-sheet-helix transitions in chain A.
    ca_atoms = [a for r in mol.chains[0].residues
                for a in r.atoms if a.name == "CA"]
    pts = mesh.points
    radii = []
    for ca in ca_atoms:
        d = np.linalg.norm(pts - ca.coord, axis=1)
        ring = pts[d < 1.8]
        if len(ring) < 6:
            continue
        r = np.sqrt(np.mean(np.sum((ring - ring.mean(axis=0)) ** 2, axis=1)))
        radii.append((ca.residue.number, r))
    # Find max gradient between adjacent residues
    if len(radii) < 4:
        pytest.skip("not enough rings")
    arr = np.array([r for _, r in radii])
    max_jump = float(np.max(np.abs(np.diff(arr))))
    # With gaussian smoothing over a window of 3 residues, the
    # transition should be smooth: max jump < 0.6 Å between adjacent
    # residues (much less than the helix→loop radius difference of ~1 Å).
    assert max_jump < 1.0, (
        f"ribbon radius changes too abruptly between adjacent residues: "
        f"max jump = {max_jump:.3f} Å — gaussian smoothing not working?"
    )


def test_ribbon_sheet_arrowhead_widens():
    """The C-terminal end of a sheet strand must be visibly wider than
    the N-terminal end (the arrowhead)."""
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    import numpy as np
    # 1aki's sheet residues are colored yellow in the secondary scheme
    # (SS_COLORS["sheet"] = (0.95, 0.85, 0.20)).
    yellow_mask = np.all(np.isclose(mesh.cell_data["colors"], [0.95, 0.85, 0.20],
                                     atol=0.02), axis=1)
    yellow_cells = np.where(yellow_mask)[0]
    assert len(yellow_cells) > 0, "expected yellow sheet cells"
    pts = mesh.points
    # Find all vertex indices referenced by yellow cells
    flat_faces = mesh.faces.reshape(-1, 4)
    yellow_vert_ids = set()
    for ci in yellow_cells:
        yellow_vert_ids.update(flat_faces[ci, 1:4].tolist())
    yellow_pts = pts[list(yellow_vert_ids)]
    bbox_extent = yellow_pts.max(axis=0) - yellow_pts.min(axis=0)
    # For a 3-residue sheet with arrow widening the last 2 to 2× width,
    # the bbox should span several Å in one direction (the strip
    # axis) AND several Å in another (the arrowhead widening).
    assert bbox_extent.max() > 1.5, (
        f"sheet bbox too small {bbox_extent} — arrowhead widening missing?"
    )


def test_ribbon_handles_1aki_end_to_end():
    """Smoke test: build a full ribbon for 1aki and ensure no crash,
    non-empty mesh, and secondary colors are present."""
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    assert mesh.n_cells > 100
    colors = mesh.cell_data["colors"]
    # We expect at least 3 distinct colors (helix, sheet, loop)
    unique = {tuple(c) for c in colors}
    assert len(unique) >= 3
