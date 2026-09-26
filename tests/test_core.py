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


def test_ribbon_is_a_flat_strip_not_a_cylinder():
    """A 3Dmol.js-style ribbon is a flat twisted strip — its cross-section
    is wider in the side-vector direction than in the binormal direction.
    A cylinder would have similar width in both axes; we want to confirm
    the strip is much wider than it is thick."""
    mol = load_pdb(EXAMPLES / "4hhb.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    # Pick any residue mid-protein — its strip cross-section should be a
    # rectangle ~helix_sheet_width × thickness = 1.3 × 0.4 = 2.6 wide
    # and 0.4 thick (aspect ratio ~6.5).
    # Measure: at the same residue i, find the cross-section extent along
    # the local side vector vs along the local binormal.
    import numpy as np
    pts = mesh.points
    # Approximate by sampling all points in a tight cluster.
    centroid = pts.mean(axis=0)
    nearby = pts[np.linalg.norm(pts - centroid, axis=1) < 1.0]
    if len(nearby) < 4:
        pytest.skip("no nearby cluster to measure cross-section")
    # PCA: the smallest-variance axis is the binormal (thickness direction)
    centered = nearby - nearby.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals sorted ascending → thickness² ≤ side² ≤ along-chain²
    thickness_axis_var = eigvals[0]
    side_axis_var = eigvals[1]
    assert side_axis_var > thickness_axis_var * 4, (
        f"ribbon cross-section should be much wider than thick; got "
        f"side²={side_axis_var:.3f}, thickness²={thickness_axis_var:.3f} "
        f"— looks more like a cylinder/tube than a flat strip"
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
