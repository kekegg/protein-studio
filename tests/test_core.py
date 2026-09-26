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
    ribbon, _local_frames, _group_runs, _helix_segment, _sheet_segment,
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

def test_local_frames_one_per_ca_residue():
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    chain = mol.chains[0]
    ca_count = sum(1 for r in chain.residues if r.atom("CA") is not None)
    frames = _local_frames(list(chain.residues))
    assert len(frames) == ca_count
    for f in frames:
        assert abs(np.linalg.norm(f.axis) - 1.0) < 1e-6
        assert abs(np.linalg.norm(f.up) - 1.0) < 1e-6
        assert abs(f.axis.dot(f.up)) < 1e-6  # perpendicular


def test_group_runs_splits_by_ss():
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    chain = mol.chains[0]
    frames = _local_frames(list(chain.residues))
    runs = _group_runs(frames)
    assert runs, "should produce at least one run"
    # The kinds must match the actual SS of each frame's residue
    for kind, run in runs:
        for f in run:
            ss = (f.residue.secondary_structure or "loop").lower()
            if ss == "helix":
                assert kind == "helix"
            elif ss == "sheet":
                assert kind == "sheet"
            else:
                assert kind == "loop"


def test_helix_segment_connected_manifold():
    """A helix segment must produce a single connected manifold mesh,
    not a stack of disconnected rings."""
    mol = load_pdb(EXAMPLES / "4hhb.pdb")
    chain = mol.chains[1]
    frames = _local_frames(list(chain.residues))
    runs = _group_runs(frames)
    helix_run = next(r for k, r in runs if k == "helix")
    mesh = _helix_segment(helix_run, 1.5, (1.0, 0.3, 0.1))
    assert mesh.n_cells > 0
    # Connectivity check — a properly-built cylinder should be 1 region
    import vtk
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(mesh)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    assert conn.GetNumberOfExtractedRegions() == 1, (
        f"helix segment has {conn.GetNumberOfExtractedRegions()} disconnected "
        f"pieces — likely only the first/last residue rings are connected "
        f"and the dense interpolated rings are orphaned"
    )


def test_helix_segment_volume_scales_with_length():
    """Helix volume should scale roughly linearly with the number of
    residue frames (longer helix = more cylinder)."""
    mol = load_pdb(EXAMPLES / "4hhb.pdb")
    chain = mol.chains[1]
    frames = _local_frames(list(chain.residues))
    runs = _group_runs(frames)
    helix_runs = [r for k, r in runs if k == "helix"]
    import vtk
    def volume(r):
        m = _helix_segment(r, 1.5, (1.0, 0.3, 0.1))
        mp = vtk.vtkMassProperties()
        mp.SetInputData(m)
        mp.Update()
        return mp.GetVolume()
    short_run = min(helix_runs, key=len)
    long_run = max(helix_runs, key=len)
    assert volume(long_run) > volume(short_run) * 1.5, (
        "longer helix should have substantially more cylinder volume"
    )


def test_ribbon_returns_single_polydata_with_colors():
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    mesh = ribbon(mol, scheme=SCHEME_PRESETS["secondary"])
    import pyvista as pv
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_cells > 0
    assert "colors" in mesh.cell_data
    assert mesh.cell_data["colors"].shape[0] == mesh.n_cells


def test_ribbon_sheet_segment_contains_arrowhead_extra_cells():
    """A sheet segment with N residues should have at least N strip
    quads + arrowhead triangles, so the cell count exceeds N."""
    mol = load_pdb(EXAMPLES / "1aki.pdb")
    chain = mol.chains[0]
    frames = _local_frames(list(chain.residues))
    runs = _group_runs(frames)
    sheet_run = next(r for k, r in runs if k == "sheet")
    mesh = _sheet_segment(sheet_run, 1.6, 0.3, (1.0, 1.0, 0.0),
                          arrowhead=True)
    n_res = len(sheet_run)
    assert mesh.n_cells > n_res, (
        "sheet must include arrowhead triangles beyond the per-residue quads"
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
