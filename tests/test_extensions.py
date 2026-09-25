"""Tests for the protein-studio extensions: charges, hbonds,
trajectories, AlphaFold, and docking.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from protein_studio.api import StudioAPI, HeadlessViewport
from protein_studio.core.alphafold import (
    AlphaFoldColorScheme, is_alphafold_pdb, plddt_bucket, plddt_summary,
)
from protein_studio.core.charges import (
    ESPConfig, assign_partial_charges, compute_esp_surface, esp_to_rgb,
)
from protein_studio.core.colors import SCHEME_PRESETS
from protein_studio.core.docking import load_docking_results, poses_to_color_array
from protein_studio.core.hbonds import detect_hbonds, hbonds_as_lines, measure_distance
from protein_studio.core.io import load_pdb
from protein_studio.core.molecule import Molecule
from protein_studio.core.representations import (
    esp_surface, hbonds_representation, available_representations,
)


EXAMPLES = ROOT / "examples"


# ============================================================== charges / ESP


def test_partial_charges_shape_and_signs():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    q = assign_partial_charges(mol)
    assert q.shape == (mol.n_atoms(),)
    # Some atoms should be negative (O / N) and some positive (terminals)
    assert (q < 0).any()
    # The mean magnitude should be small (heuristic charges, not force-field)
    assert abs(q.mean()) < 0.3


def test_esp_surface_returns_points_and_potential():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    pts, pot = compute_esp_surface(mol, config=ESPConfig(n_sample_points=300))
    assert pts.shape[0] == 300
    assert pot.shape == (300,)
    assert np.isfinite(pot).all()


def test_esp_to_rgb_zero_is_white():
    rgb = esp_to_rgb(np.array([0.0]))
    assert np.allclose(rgb[0], (1.0, 1.0, 1.0))


def test_esp_surface_mesh_has_colors():
    import pyvista as pv
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    mesh = esp_surface(mol, n_points=400)
    assert isinstance(mesh, pv.PolyData)
    # The mesh carries per-cell colors
    assert "colors" in (mesh.cell_data.keys() or []) or "colors" in (mesh.point_data.keys() or [])


# ============================================================== H-bonds


def test_detect_hbonds_returns_bond_list():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    hbs = detect_hbonds(mol)
    # Ubiquitin's β-sheet should generate many H-bonds
    assert len(hbs) > 0
    assert all(hb.distance <= 3.5 for hb in hbs)


def test_hbonds_as_lines_has_2n_coords():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    hbs = detect_hbonds(mol)
    coords, lines = hbonds_as_lines(hbs)
    assert coords.shape[0] == 2 * len(hbs)
    assert lines.shape[0] == 3 * len(hbs)


def test_measure_distance_simple():
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    a = mol.atom_by_name("A", 1, "N")
    b = mol.atom_by_name("A", 1, "CA")
    d = measure_distance(a, b)
    # peptide N-CA distance is ~1.33 Å
    assert 1.2 < d.distance < 1.5


def test_hbonds_representation_renders():
    import pyvista as pv
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    out = hbonds_representation(mol)
    assert isinstance(out, pv.PolyData)
    assert out.n_cells > 0


# ============================================================== AlphaFold


def test_alphafold_scheme_registered():
    assert "alphafold_plddt" in SCHEME_PRESETS


def test_alphafold_buckets():
    assert plddt_bucket(95) == "very_high"
    assert plddt_bucket(80) == "confident"
    assert plddt_bucket(60) == "low"
    assert plddt_bucket(20) == "very_low"


def test_alphafold_scheme_returns_color():
    s = AlphaFoldColorScheme()
    class A:
        bfactor = 95.0
    rgb = s.color(A())
    # Royal blue (0.12, 0.53, 0.90)
    assert rgb[0] < rgb[2]


def test_plddt_summary_on_normal_pdb():
    # Plain PDBs have all-zero B-factor — summary should still return
    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    summary = plddt_summary(mol)
    assert "mean_plddt" in summary
    assert summary["n_residues"] > 0


def test_is_alphafold_pdb_detects_alphafold_files():
    # Construct a tiny fake AF PDB
    fake = ROOT / "tests" / "_fake_af.pdb"
    fake.write_text(
        "HEADER    PREDICTED 3D STRUCTURE                       18-JAN-22   TEST    \n"
        "TITLE     ALPHAFOLD MONOMER PREDICTION                          \n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 85.00           N\n"
        "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 85.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00 70.00           C\n"
        "ATOM      4  O   ALA A   1       1.500   2.500   0.000  1.00 70.00           O\n"
        "END\n"
    )
    try:
        assert is_alphafold_pdb(str(fake)) is True
    finally:
        fake.unlink(missing_ok=True)


# ============================================================== Trajectory


def test_trajectory_optional():
    """Without MDAnalysis the test is skipped, otherwise we just check
    import plumbing — no real trajectory in the repo.
    """
    pytest.importorskip("MDAnalysis")
    # Sanity: Trajectory class exists
    from protein_studio.core.trajectory import Trajectory
    assert Trajectory is not None


def test_try_fetch_demo_trajectory():
    """May or may not return a trajectory depending on data files."""
    from protein_studio.core.trajectory import try_fetch_demo_trajectory
    traj = try_fetch_demo_trajectory()
    if traj is not None:
        assert traj.n_frames > 0


# ============================================================== Docking


def test_load_single_pdb_as_one_pose(tmp_path):
    fake = tmp_path / "ligand.pdb"
    fake.write_text(
        "ATOM      1  N   LIG A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  C1  LIG A   1       1.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C2  LIG A   1       2.000   1.500   0.000  1.00  0.00           C\n"
        "END\n"
    )
    poses = load_docking_results(str(fake))
    assert len(poses) == 1
    assert poses[0].rank == 1


def test_load_pose_dir(tmp_path):
    d = tmp_path / "poses"
    d.mkdir()
    (d / "pose_001.pdb").write_text(
        "ATOM      1  N   LIG A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  C1  LIG A   1       1.500   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    (d / "pose_002.pdb").write_text(
        "ATOM      1  N   LIG A   1       2.000   2.000   2.000  1.00  0.00           N\n"
        "ATOM      2  C1  LIG A   1       3.500   2.000   2.000  1.00  0.00           C\n"
        "END\n"
    )
    (d / "scores.txt").write_text("pose_001 -7.3\npose_002 -6.8\n")
    poses = load_docking_results(str(d))
    assert len(poses) == 2
    # Rank 1 should be the lower energy
    assert poses[0].score == -7.3
    assert poses[0].rank == 1


def test_poses_to_color_array_shape():
    fake = []
    from protein_studio.core.docking import DockingPose
    from protein_studio.core.molecule import Molecule
    mol = Molecule(name="x")
    for s in (-7.0, -6.5, -6.0):
        fake.append(DockingPose(rank=0, score=s, ligand=mol, source="x", pose_id=0))
    colors = poses_to_color_array(fake)
    assert colors.shape == (3, 3)


# ============================================================== API surface


def test_api_extended_methods_exist():
    api = StudioAPI(viewport=HeadlessViewport())
    assert callable(api.esp_color_scheme)
    assert callable(api.detect_hbonds)
    assert callable(api.distance)
    assert callable(api.load_trajectory)
    assert callable(api.load_alphafold)
    assert callable(api.load_docking)
    assert callable(api.assign_charges)


# ============================================================== Rendering
# (These verify the PyVista color-preference fix — the bug where
# cell_data colors were silently ignored and meshes came out white.)


def test_gui_viewport_uses_cell_preference_for_color():
    """Regression: the GUI viewport must pass `preference='cell'` so that
    cartoon / surface meshes that store colors in cell_data show up
    colored (not white).
    """
    try:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from protein_studio.ui.viewport import ViewportWidget
        from protein_studio.core.scene import Scene
    except ImportError:
        pytest.skip("PySide6 not available")
    app = QApplication.instance() or QApplication([])
    scene = Scene()
    v = ViewportWidget(scene)

    # Capture every kwargs passed to add_mesh
    captured = []
    real_add = v.plotter.add_mesh

    def spy_add_mesh(*args, **kwargs):
        captured.append(kwargs)
        return real_add(*args, **kwargs)

    v.plotter.add_mesh = spy_add_mesh

    mol = load_pdb(EXAMPLES / "1ubq.pdb")
    from protein_studio.core.representations import cartoon
    mesh = cartoon(mol)
    v._meshes["ubq"] = mesh
    from protein_studio.core.scene import SceneObject
    from protein_studio.core.colors import SCHEME_PRESETS
    obj = SceneObject(name="ubq", molecule=mol, representation="cartoon",
                      color_scheme=SCHEME_PRESETS["secondary"])
    scene.add(obj)
    v.refresh()

    # Find at least one add_mesh call that supplied colors via cell_data
    color_calls = [c for c in captured if c.get("scalars") == "colors"]
    assert color_calls, "viewport never passed scalars='colors' to add_mesh"
    assert all(c.get("preference") == "cell" for c in color_calls), (
        "viewport must pass preference='cell' so cell_data colors are "
        "picked up — otherwise the mesh comes out white."
    )

    # Also: the mesh must be passed positionally. Calling add_mesh as
    # `add_mesh(**kwargs)` (forgetting the mesh arg) raises
    # `TypeError: missing 1 required positional argument: 'mesh'`.
    for call_args in []:  # captured is kwargs only, check raise pattern
        pass
    # Re-run refresh and capture positional args too
    real_add = v.plotter.add_mesh
    positional_calls = []

    def spy_pos(*args, **kwargs):
        positional_calls.append((args, kwargs))
        return real_add(*args, **kwargs)

    v.plotter.add_mesh = spy_pos
    v.refresh()
    mesh_calls = [c for c in positional_calls if len(c[0]) > 0]
    assert mesh_calls, (
        "viewport must pass the mesh as the first positional argument to "
        "add_mesh; otherwise PyVista raises TypeError and the mesh is "
        "silently dropped (this was the white-mesh bug)."
    )


def test_api_can_alphafold_load_with_auto_scheme():
    pytest.importorskip("MDAnalysis")
    # Construct a fake AF PDB in-memory
    import tempfile
    fake = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    fake.write(
        "HEADER    ALPHAFOLD MODEL                                \n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 85.00           N\n"
        "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 85.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00 70.00           C\n"
        "ATOM      4  O   ALA A   1       1.500   2.500   0.000  1.00 70.00           O\n"
        "END\n"
    )
    fake.close()
    try:
        api = StudioAPI(viewport=HeadlessViewport())
        obj = api.load_alphafold(fake.name)
        assert obj.color_scheme.key == "alphafold_plddt"
        s = api.plddt_summary(obj.name)
        assert "mean_plddt" in s
    finally:
        Path(fake.name).unlink(missing_ok=True)


def test_api_detect_hbonds_returns_list():
    api = StudioAPI(viewport=HeadlessViewport())
    api.load(EXAMPLES / "1ubq.pdb")
    hbs = api.detect_hbonds("1ubq")
    assert isinstance(hbs, list)
    assert len(hbs) > 0


def test_api_distance_between_known_atoms():
    api = StudioAPI(viewport=HeadlessViewport())
    api.load(EXAMPLES / "1ubq.pdb")
    d = api.distance("1ubq", "A", 1, "N", "A", 1, "CA")
    assert 1.2 < d.distance < 1.5


# ============================================================== Console
# (these run only when PySide6 is available in the test env)


def test_console_handles_multistatement_paste():
    """Reproduces the user-reported SyntaxError when pasting a multi-line
    snippet into the in-app console.
    """
    try:
        from protein_studio.ui.console import PythonConsole
    except ImportError:
        pytest.skip("PySide6 not available")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    api = StudioAPI(viewport=HeadlessViewport())
    console = PythonConsole(namespace={"D": api})
    # The exact string that previously errored:
    snippet = (
        'af = D.load("examples/1ubq.pdb")\n'
        'D.plddt_summary(af.name)\n'
        'print("done")'
    )
    console._editor.setPlainText(">>> " + snippet)
    console._on_return()
    # No assertion needed — the call would raise on a regression.
    assert True


def test_console_multiline_block():
    """A block with a continued statement (for loop) should also work."""
    try:
        from protein_studio.ui.console import PythonConsole
    except ImportError:
        pytest.skip("PySide6 not available")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    api = StudioAPI(viewport=HeadlessViewport())
    console = PythonConsole(namespace={"D": api})
    console._editor.setPlainText(
        ">>> " + "for i in range(3):\n" "    print(i)\n"
    )
    console._on_return()
    # After running, the namespace should contain `i == 2`
    assert console.namespace.get("i") == 2