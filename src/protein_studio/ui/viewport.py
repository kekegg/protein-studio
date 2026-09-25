"""3D viewport widget — wraps a PyVista Qt interactor.

Renders the current scene's meshes and supports:

- Mouse-orbit, zoom, pan (PyVista default camera style)
- Reset to fit-all (center of mass)
- Toggle between solid / wireframe / surface representation per-object
- Background color picker (Blender-style dark gray by default)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QMenu,
)

from ..core.scene import Scene, SceneObject


class ViewportWidget(QWidget):
    """The main 3D viewport widget.

    Wraps a :class:`pyvistaqt.QtInteractor` for rendering.
    """

    selection_changed = Signal(str)  # object name picked (or "" if cleared)

    def __init__(self, scene: Scene, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._meshes: Dict[str, Any] = {}  # name -> pyvista mesh or dict-of-meshes
        self._actors: Dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar overlay (Blender-style minimal top bar)
        toolbar = QFrame()
        toolbar.setStyleSheet("background: #2d2d2d; border-bottom: 1px solid #1a1a1a;")
        tbar = QHBoxLayout(toolbar)
        tbar.setContentsMargins(8, 4, 8, 4)
        self._info_label = QLabel("No objects")
        self._info_label.setStyleSheet("color: #dcdcdc; font-size: 11px;")
        tbar.addWidget(self._info_label)
        tbar.addStretch()

        # Center, fit, axes buttons
        for label, slot in (
            ("Center", self.center_view),
            ("Fit",   self.fit_view),
            ("Reset", self.reset_camera),
            ("Axes",  self.toggle_axes),
        ):
            btn = QPushButton(label)
            btn.setStyleSheet(
                "QPushButton { background: #3a3a3a; color: #dcdcdc; "
                "padding: 3px 10px; border: 1px solid #555; border-radius: 3px; }"
                "QPushButton:hover { background: #4a4a4a; }"
            )
            btn.clicked.connect(slot)
            tbar.addWidget(btn)

        layout.addWidget(toolbar)

        # PyVista interactor
        try:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
        except Exception:
            # Fallback: PyVista's background plotter (still works without Qt event loop)
            import pyvista as pv
            self.plotter = pv.Plotter()
            placeholder = QLabel(
                "pyvistaqt is not available — falling back to offscreen rendering"
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder, 1)

        if hasattr(self.plotter, "set_background"):
            self.plotter.set_background("#333333", top="#222222")

        # Enable SSAO (screen-space ambient occlusion) for depth cues
        # between closely packed atoms.
        if hasattr(self.plotter, "enable_ssao"):
            try:
                self.plotter.enable_ssao(radius=1.5, bias=0.02, kernel_size=64)
            except Exception:
                pass

        # Remove PyVista's default light kit (it's bright and washes
        # out our three-point setup), then install key+fill+rim+rim2.
        self._install_lights()

        if hasattr(self.plotter, "add_axes"):
            self.plotter.add_axes()
            self._axes_visible = True
        else:
            self._axes_visible = False
        if hasattr(self, "plotter") and isinstance(self.plotter, QWidget):
            layout.addWidget(self.plotter, 1)

        # Scene → render hook
        self.scene.on_change(self._on_scene_change)

    # ----------------------------------------------------------- public ops
    def refresh(self) -> None:
        """Re-render every visible scene object."""
        if not hasattr(self, "plotter") or not hasattr(self.plotter, "clear"):
            return
        try:
            self.plotter.clear()
        except Exception:
            return
        # CRITICAL: `plotter.clear()` also removes lights and SSAO.
        # Re-install the lighting rig and SSAO every refresh, otherwise
        # the first render looks lit but every subsequent refresh goes
        # completely flat (no Phong shading, no shadow contact).
        if hasattr(self.plotter, "enable_ssao"):
            try:
                self.plotter.enable_ssao(radius=1.5, bias=0.02, kernel_size=64)
            except Exception:
                pass
        self._install_lights()
        if self._axes_visible and hasattr(self.plotter, "add_axes"):
            self.plotter.add_axes()
        # Render meshes
        for obj in self.scene.objects:
            if not obj.visible:
                continue
            mesh = self._meshes.get(obj.name)
            if mesh is None:
                continue
            self._add_object_to_plotter(obj, mesh)
        # Update info label
        n_obj = self.scene.n_objects()
        n_atoms = self.scene.n_atoms()
        self._info_label.setText(
            f"{n_obj} object{'s' if n_obj != 1 else ''} · "
            f"{n_atoms} atoms"
        )

    def fit_view(self) -> None:
        if hasattr(self.plotter, "reset_camera"):
            try:
                self.plotter.reset_camera()
            except Exception:
                pass

    def reset_camera(self) -> None:
        self.fit_view()

    def toggle_axes(self) -> None:
        self._axes_visible = not getattr(self, "_axes_visible", True)
        self.refresh()

    def center_view(self) -> None:
        if not self.scene.objects:
            return
        coms = [o.molecule.center_of_mass() for o in self.scene.objects if o.visible]
        if not coms:
            return
        com = np.mean(coms, axis=0)
        for o in self.scene.objects:
            o.position = -com
        self.scene.notify("update")
        self.refresh()

    def update_object(self, name: str, mesh: Any) -> None:
        self._meshes[name] = mesh
        # Re-render only that object
        obj = self.scene.find(name)
        if obj and obj.visible:
            self._add_object_to_plotter(obj, mesh)

    def remove_object(self, name: str) -> None:
        self._meshes.pop(name, None)
        # Remove actors by name
        actor_names = [a.name for a in self.plotter.actors if hasattr(a, "name")]
        for n in actor_names:
            if n == name or n.startswith(f"{name}:"):
                try:
                    self.plotter.remove_actor(n)
                except Exception:
                    pass
        self.refresh()

    # ----------------------------------------------------------- internal
    def _on_scene_change(self, kind: str, payload: Any) -> None:
        if kind == "add":
            # Will be populated by main_window -> refresh
            pass
        elif kind == "remove":
            if isinstance(payload, str):
                self.remove_object(payload)

    def _install_lights(self) -> None:
        """Install a three-point lighting rig (key + fill + rim).

        PyVista's default light kit is too bright and washes out our
        setup, so we RemoveAllLights() first then add exactly three
        directional lights.
        """
        try:
            import vtk
            LIGHT = vtk.vtkLight
        except ImportError:
            return

        # Clear default light kit so our three lights are the only ones
        renderer = getattr(self.plotter, "renderer", None)
        if renderer is None:
            return
        try:
            renderer.RemoveAllLights()
        except Exception:
            pass

        lights_spec = [
            # name,         position,                  intensity, color
            ("key_light",   ( 0.5,  1.0,  0.5),  1.20, (1.00, 0.97, 0.92)),
            ("fill_light",  (-0.5, -0.5,  0.3),  0.45, (0.85, 0.88, 0.95)),
            ("rim_light",   ( 0.0, -0.2, -1.0),  0.65, (1.00, 0.92, 0.85)),
            ("side_light",  ( 1.0,  0.0,  0.0),  0.50, (0.95, 0.95, 1.00)),
            ("side2_light", (-1.0,  0.0,  0.0),  0.40, (1.00, 0.92, 0.95)),
        ]
        for name, pos, intensity, color in lights_spec:
            try:
                light = LIGHT()
                light.SetPosition(*pos)
                light.SetIntensity(intensity)
                light.SetColor(*color)
                light.PositionalOff()
                renderer.AddLight(light)
            except Exception:
                continue

    def _add_object_to_plotter(self, obj: SceneObject, mesh: Any) -> None:
        # Per-representation shading hints. "cartoon"/"ribbon" read better
        # with Phong + edges; "esp_surface"/"surface" benefit from PBR
        # because they have a continuous shaded surface.
        rep = obj.representation
        use_pbr   = rep in ("esp_surface", "surface")
        use_edges = rep in ("cartoon", "ribbon", "ball_and_stick")

        def _add_one(sub, name: str) -> None:
            if sub is None:
                return
            # PyVista 0.49's add_mesh() defaults to point_data when
            # looking up scalars. Our meshes keep the color array on
            # *cell_data*, so without `preference='cell'` the renderer
            # silently falls back to a white default. Detect which side
            # the colors live on and pin the preference accordingly.
            has_cell   = "colors" in (sub.cell_data.keys() or [])
            has_point  = "colors" in (sub.point_data.keys() or [])
            scalars = "colors" if (has_cell or has_point) else None
            preference = "cell" if has_cell else ("point" if has_point else "point")
            kwargs = dict(
                name=name,
                color=None,
                scalars=scalars,
                preference=preference,
                rgb=True,
                smooth_shading=True,
                show_scalar_bar=False,
            )
            if use_pbr:
                kwargs.update(
                    pbr=True,
                    metallic=0.05,
                    roughness=0.55,
                )
            else:
                # specular_power must stay in PyVista's allowed range (0, 128)
                kwargs.update(
                    specular=1.0,
                    specular_power=100,
                    ambient=0.10,
                    diffuse=0.90,
                )
            attempts = [
                kwargs,
                # drop smooth_shading — some backends reject it on tube-like geometry
                {k: v for k, v in kwargs.items() if k != "smooth_shading"},
                # drop specular/ambient/diffuse — OpenGL ES backends may reject
                {k: v for k, v in kwargs.items()
                 if k not in ("specular", "specular_power", "ambient", "diffuse")},
                # keep scalars but drop smooth_shading AND lighting kwargs
                {k: v for k, v in kwargs.items()
                 if k not in ("smooth_shading", "specular", "specular_power",
                              "ambient", "diffuse", "pbr", "metallic", "roughness")},
            ]
            last_exc: Exception | None = None
            for attempt_kwargs in attempts:
                try:
                    self.plotter.add_mesh(sub, **attempt_kwargs)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                print(
                    f"[viewport] add_mesh failed for {name!r} "
                    f"(rep={obj.representation}): "
                    f"{type(last_exc).__name__}: {last_exc}",
                    flush=True,
                )
                try:
                    self.plotter.add_mesh(sub, name=name, color="#cccccc")
                except Exception as exc2:
                    print(
                        f"[viewport] fallback add_mesh also failed for {name!r}: "
                        f"{type(exc2).__name__}: {exc2}",
                        flush=True,
                    )

            # Add silhouette edges for cartoon-style reps
            if use_edges:
                # No outline overlay. Earlier versions drew the mesh a
                # second time with `style="wireframe"` (every triangle
                # edge), which on a 10-side tube produces a faceted
                # wireframe look instead of a smooth cartoon. A real
                # silhouette outline needs screen-space tricks (e.g.
                # inverse-hull), so leave that for a later iteration.
                pass

        if isinstance(mesh, dict):
            for k, sub in mesh.items():
                _add_one(sub, f"{obj.name}:{k}")
        else:
            _add_one(mesh, obj.name)

    def screenshot(self, path: str) -> str:
        if hasattr(self.plotter, "screenshot"):
            return self.plotter.screenshot(path, return_img=False) or path
        return ""

    # ----------------------------------------------------------- misc
    def wheelEvent(self, event) -> None:  # noqa: N802
        # PyVista handles camera zoom via its own event filters; keep this
        # empty so the OS doesn't scroll the parent widget.
        event.accept()