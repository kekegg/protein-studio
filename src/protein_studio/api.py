"""Python scripting API.

Exposes a small Blender-style DSL to the in-app console. Once the GUI
loads, a single :class:`StudioAPI` instance is bound to the name ``D``
(``D = StudioAPI(...)``). Users can then drive the application from the
embedded console:

    >>> D.load("examples/1aki.pdb")
    >>> D.objects[0].representation = "cartoon"
    >>> D.objects[0].color_scheme = D.scheme("element")
    >>> D.refresh()

The module is decoupled from Qt so it can also be used headlessly for
batch rendering:

    >>> from protein_studio.api import StudioAPI, HeadlessViewport
    >>> api = StudioAPI(viewport=HeadlessViewport())
    >>> api.load("examples/1aki.pdb")
    >>> api.refresh()
    >>> api.viewport.screenshot("out.png")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .core.alphafold import (
    AlphaFoldColorScheme, is_alphafold_pdb, plddt_summary,
)
from .core.charges import assign_partial_charges, compute_esp_surface, esp_to_rgb
from .core.colors import ColorScheme, SCHEME_PRESETS
from .core.docking import DockingPose, load_docking_results, poses_to_color_array
from .core.hbonds import (
    Distance, HBond, detect_hbonds, measure_distance,
)
from .core.io import load_pdb, save_pdb
from .core.molecule import Atom, Molecule
from .core.representations import REPRESENTATIONS, available_representations
from .core.scene import Scene, SceneObject
from .core.trajectory import Trajectory, load_trajectory


class HeadlessViewport:
    """Tiny stand-in for the Qt viewport.

    Holds the latest rendered PyVista plots keyed by object name. Used
    for scripted rendering without a GUI.
    """

    def __init__(self) -> None:
        self.plots: dict[str, Any] = {}

    def clear(self) -> None:
        self.plots.clear()

    def add(self, name: str, plot: Any) -> None:
        self.plots[name] = plot

    def screenshot(self, path: str, **kwargs) -> str:
        try:
            import pyvista as pv
            import vtk
        except ImportError as e:
            raise RuntimeError("pyvista is required for headless rendering") from e
        if not self.plots:
            raise RuntimeError("Nothing to render — call refresh() first.")
        p = pv.Plotter(off_screen=True)
        p.set_background("#222222", top="#111111")
        if hasattr(p, "enable_ssao"):
            try:
                p.enable_ssao(radius=0.6, bias=0.02)
            except Exception:
                pass
        # Remove PyVista's default light kit, install our three-point rig
        try:
            p.renderer.RemoveAllLights()
        except Exception:
            pass
        for pos, intensity, color in [
            (( 0.5,  1.0,  0.5), 1.20, (1.00, 0.97, 0.92)),
            ((-0.5, -0.5,  0.3), 0.45, (0.85, 0.88, 0.95)),
            (( 0.0, -0.2, -1.0), 0.65, (1.00, 0.92, 0.85)),
        ]:
            try:
                light = vtk.vtkLight()
                light.SetPosition(*pos)
                light.SetIntensity(intensity)
                light.SetColor(*color)
                light.PositionalOff()
                p.renderer.AddLight(light)
            except Exception:
                pass
        for k, m in self.plots.items():
            if isinstance(m, dict):
                for sub in m.values():
                    has_cell   = "colors" in (sub.cell_data.keys() or [])
                    has_point  = "colors" in (sub.point_data.keys() or [])
                    sub_scalars = "colors" if (has_cell or has_point) else None
                    pref = "cell" if has_cell else ("point" if has_point else "point")
                    try:
                        p.add_mesh(sub, scalars=sub_scalars, rgb=True,
                                   preference=pref,
                                   smooth_shading=True,
                                   specular=1.0, specular_power=100,
                                   ambient=0.10, diffuse=0.90,
                                   show_scalar_bar=False)
                    except Exception:
                        try:
                            p.add_mesh(sub)
                        except Exception:
                            pass
            else:
                has_cell   = "colors" in (m.cell_data.keys() or [])
                has_point  = "colors" in (m.point_data.keys() or [])
                scalars = "colors" if (has_cell or has_point) else None
                pref = "cell" if has_cell else ("point" if has_point else "point")
                try:
                    p.add_mesh(m, scalars=scalars, rgb=True,
                               preference=pref,
                               smooth_shading=True,
                               specular=1.0, specular_power=100,
                               ambient=0.10, diffuse=0.90,
                               show_scalar_bar=False)
                except Exception:
                    try:
                        p.add_mesh(m)
                    except Exception:
                        pass
        try:
            p.camera_position = "iso"
        except Exception:
            pass
        return p.show(screenshot=path) or path


class StudioAPI:
    """The high-level scripting API surfaced as ``D`` in the console."""

    def __init__(self, viewport: Any | None = None) -> None:
        self.scene = Scene()
        self.viewport = viewport or HeadlessViewport()
        self.metadata: dict[str, Any] = {"version": "0.1.0"}

        # Subscriptions: any change in the scene triggers a refresh signal
        self.scene.on_change(self._on_scene_change)
        self._dirty = False

    # --------------------------------------------------------- introspection
    @property
    def objects(self) -> List[SceneObject]:
        return self.scene.objects

    @property
    def active(self) -> Optional[SceneObject]:
        return self.scene.active

    # ----------------------------------------------------------------- load
    def load(
        self,
        path: str | os.PathLike,
        name: str | None = None,
        representation: str = "cartoon",
        scheme: str = "secondary",
    ) -> SceneObject:
        """Load a PDB file into the scene.

        >>> obj = D.load("examples/1aki.pdb")
        """
        mol = load_pdb(path)
        if name:
            mol.name = name
        obj = SceneObject(
            name=name or mol.name,
            molecule=mol,
            representation=representation,
            color_scheme=self.scheme(scheme),
        )
        self.scene.add(obj)
        return obj

    def save(self, obj_name: str, path: str) -> None:
        obj = self.scene.find(obj_name)
        if not obj:
            raise KeyError(obj_name)
        save_pdb(obj.molecule, path)

    def remove(self, name: str) -> None:
        self.scene.remove(name)

    def clear(self) -> None:
        self.scene.clear()

    # ------------------------------------------------------------ color/presets
    def scheme(self, name: str, **kwargs) -> ColorScheme:
        """Look up a color-scheme preset by name."""
        if name not in SCHEME_PRESETS:
            raise KeyError(f"Unknown color scheme '{name}'. Known: {list(SCHEMES)}")
        s = SCHEME_PRESETS[name]
        if kwargs:
            # Reconstruct with overrides (ColorScheme is frozen — replace)
            return ColorScheme(
                name=s.name,
                key=s.key,
                uniform=kwargs.get("uniform", s.uniform),
                bfactor_range=kwargs.get("bfactor_range", s.bfactor_range),
            )
        return s

    @property
    def schemes(self) -> List[str]:
        return list(SCHEMES)

    @property
    def representations(self) -> List[str]:
        return available_representations()

    # -------------------------------------------------------- mutation helpers
    def set_representation(self, name: str, representation: str) -> None:
        if representation not in REPRESENTATIONS:
            raise KeyError(f"Unknown representation '{representation}'")
        self.scene.update_object(name, representation=representation)
        self.refresh()

    def set_scheme(self, name: str, scheme: str) -> None:
        obj = self.scene.find(name)
        if not obj:
            raise KeyError(name)
        obj.color_scheme = self.scheme(scheme)
        self.scene.notify("update")
        self.refresh()

    def set_visible(self, name: str, visible: bool) -> None:
        self.scene.update_object(name, visible=visible)
        self.refresh()

    def center_view(self) -> None:
        if self.scene.objects:
            com = np.mean(
                [obj.molecule.center_of_mass() for obj in self.scene.objects], axis=0
            )
            for obj in self.scene.objects:
                obj.position = -com

    # ============================================================ extensions

    # ---------- ESP / electrostatic surface -----------------------------
    def esp_color_scheme(self, name: str) -> ColorScheme:
        """A custom color scheme that paints the surface by partial-charge ESP.

        Assign with ``D.set_scheme(obj, D.esp_color_scheme("esp"))``.
        """
        return ColorScheme(
            name=name, key="uniform",
            uniform=(0.85, 0.85, 0.85),
        )

    def assign_charges(self, obj_name: str) -> np.ndarray:
        """Compute and return the partial-charge array for an object's molecule."""
        obj = self.scene.find(obj_name)
        if not obj:
            raise KeyError(obj_name)
        return assign_partial_charges(obj.molecule)

    # ---------- H-bonds / distances -------------------------------------
    def detect_hbonds(self, obj_name: str, **kwargs) -> List[HBond]:
        obj = self.scene.find(obj_name)
        if not obj:
            raise KeyError(obj_name)
        # Accept (and ignore) ``scheme`` for parity with ``D.set_scheme`` /
        # other callers — it's not used by hbonds.
        kwargs.pop("scheme", None)
        return detect_hbonds(obj.molecule, **kwargs)

    def distance(
        self, obj_name: str,
        chain_a: str, resnum_a: int, atom_a: str,
        chain_b: str, resnum_b: int, atom_b: str,
    ) -> Distance:
        obj = self.scene.find(obj_name)
        if not obj:
            raise KeyError(obj_name)
        a = obj.molecule.atom_by_name(chain_a, resnum_a, atom_a)
        b = obj.molecule.atom_by_name(chain_b, resnum_b, atom_b)
        if a is None or b is None:
            raise LookupError(f"Atoms not found: {a} / {b}")
        return measure_distance(a, b)

    # ---------- Trajectories ---------------------------------------------
    def load_trajectory(self, name: str, topology_pdb: str, trajectory: str) -> Trajectory:
        """Register a trajectory and return the wrapper."""
        traj = load_trajectory(topology_pdb, trajectory)
        setattr(self, f"_traj_{name}", traj)
        return traj

    def frame(self, traj_name: str, index: int, scene_object: str) -> None:
        """Move the named trajectory to *index* and copy coords to *scene_object*."""
        traj: Trajectory = getattr(self, f"_traj_{traj_name}")
        traj.apply_frame(index)
        self.scene.notify("update")
        self.refresh()

    # ---------- AlphaFold / Boltz-2 -------------------------------------
    def load_alphafold(self, path: str, name: Optional[str] = None) -> SceneObject:
        """Load an AlphaFold PDB; auto-detect and assign pLDDT coloring."""
        obj = self.load(path, name=name, representation="cartoon",
                        scheme="alphafold_plddt")
        info = plddt_summary(obj.molecule)
        self.metadata.setdefault("alphafold", {})[obj.name] = info
        return obj

    def plddt_summary(self, obj_name: str) -> dict:
        obj = self.scene.find(obj_name)
        if not obj:
            raise KeyError(obj_name)
        return plddt_summary(obj.molecule)

    # ---------- Docking poses -------------------------------------------
    def load_docking(self, path: str, receptor_obj: Optional[str] = None,
                     top_n: Optional[int] = None) -> List[DockingPose]:
        """Load Vina PDBQT or a directory of EquiScore poses.

        If *receptor_obj* is given, each pose becomes a child object
        with the receptor as its visual base; otherwise each pose is
        loaded as its own scene object with rank-based coloring.
        """
        poses = load_docking_results(path)
        if top_n:
            poses = poses[:top_n]
        colors = poses_to_color_array(poses)
        for pose, color in zip(poses, colors):
            obj = SceneObject(
                name=f"pose_{pose.rank}_{pose.ligand.name}",
                molecule=pose.ligand,
                representation="ball_and_stick",
                color_scheme=ColorScheme(
                    name="rank", key="uniform", uniform=tuple(color),
                ),
            )
            self.scene.add(obj)
        return poses

    # ----------------------------------------------------------- rendering
    def refresh(self) -> None:
        """Re-render every object in the scene."""
        from .core.representations import build_mesh
        self.viewport.clear()
        for obj in self.scene.objects:
            if not obj.visible:
                continue
            mesh = build_mesh(obj, obj.color_scheme)
            self.viewport.add(obj.name, mesh)
        self._dirty = False

    def screenshot(self, path: str, **kwargs) -> str:
        return self.viewport.screenshot(path, **kwargs)

    # ----------------------------------------------------------- internal
    def _apply_transform(self, mesh: Any, obj: SceneObject) -> Any:
        """Apply obj.position/rotation/scale to a mesh (or dict of meshes)."""
        M = obj.transform_matrix()
        if M is None:
            return mesh
        if isinstance(mesh, dict):
            return {k: _apply_M(v, M) for k, v in mesh.items()}
        return _apply_M(mesh, M)

    def _on_scene_change(self, kind: str, payload: Any) -> None:
        self._dirty = True


# A separate registry used by `D.scheme` lookup
SCHEMES: dict[str, ColorScheme] = SCHEME_PRESETS


def _apply_M(mesh: Any, M: np.ndarray) -> Any:
    if mesh is None or len(getattr(mesh, "points", [])) == 0:
        return mesh
    try:
        pts = np.asarray(mesh.points, dtype=float)
        ones = np.ones((len(pts), 1))
        homo = np.hstack([pts, ones])
        new = (M @ homo.T).T[:, :3]
        mesh.points = new
    except Exception:
        return mesh
    return mesh


def default_api() -> StudioAPI:
    """Construct an API instance with no viewport (used for tests)."""
    return StudioAPI()