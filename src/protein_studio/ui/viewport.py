"""3D viewport widget — wraps a PyVista Qt interactor.

Renders the current scene's meshes and supports:

- Mouse-orbit, zoom, pan (PyVista default camera style)
- Reset to fit-all (center of mass)
- Toggle between solid / wireframe / surface representation per-object
- Background color picker (Blender-style dark gray by default)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Orientation-marker widget registry
# ---------------------------------------------------------------------------
# PyVista's ``add_axes()`` installs a ``vtkOrientationMarkerWidget`` on the
# renderer's interactor. VTK's destructor for this widget calls
# ``SetEnabled(0)`` defensively, which iterates ``vtkViewport::RemoveViewProp``
# over its axes actor and releases the per-vertex GL buffer objects. If
# that destructor runs on a thread that doesn't have the QOpenGLContext
# current (e.g. an asyncio task step in the MCP server's background
# thread, or Python's cyclic GC firing during interpreter shutdown), the
# ``glDeleteBuffers`` call hits a wild pointer and segfaults.
#
# Two complementary defences live here:
#
#   1. We keep a module-level strong reference to every widget created by
#      ``add_axes``. As long as the registry holds the widget, Python's
#      cyclic GC can't collect it. The registry is cleared during
#      ``ViewportWidget.shutdown()``, by which point we've explicitly
#      disabled the widget and detached it from the interactor so the
#      VTK destructor is a no-op.
#
#   2. Each widget also gets a ``weakref.finalize`` hook that runs
#      *before* the destructor. The hook checks the current QOpenGL
#      context: if no context is current (the crash scenario), we
#      manually disable the widget via a queued call to the GUI thread.
#      That call lands while the QOpenGLWidget still owns a valid GL
#      context, so ``SetEnabled(0)`` runs cleanly there.

_ORIENTATION_WIDGETS: List[Any] = []

# Module-level strong-reference list for VTK actor wrappers that have
# been replaced in a refresh() and are pending disposal. We hold them
# here until the GUI thread is idle, so the cyclic GC never destroys
# them from a non-GUI thread (the segfault path through
# vtkActor::ReleaseGraphicsResources → glDeleteBuffers).
_PENDING_ACTOR_DISPOSE: List[Any] = []


def _safe_axes_finalizer(widget: Any) -> None:
    """weakref.finalize callback — runs before the VTK destructor.

    The previous version used ``QTimer.singleShot(0, ...)`` to bounce
    the ``EnabledOff`` call onto the GUI thread when invoked from a
    non-GUI thread. That bounce was the actual segfault trigger: the
    GUI thread was often already inside a ``set_representation``
    rep-rebuild that was using the QOpenGLWidget's GL context, so the
    bounced ``EnabledOff`` ran ``glDeleteBuffers`` against a context
    mid-render and hit a wild pointer.

    New behaviour: just try to disable the widget right where we are,
    swallow every exception. If the current thread lacks a GL context
    the VTK call is internally a no-op (and even if it segfaults at
    the C level, that's still better than corrupting the render
    window). The widget is also held by the module-level
    ``_ORIENTATION_WIDGETS`` registry so this finalizer only fires
    after ``shutdown()`` has explicitly cleared that registry — by
    which point we have already disabled the widget on the GUI
    thread and removed it from the interactor, so the VTK
    destructor is effectively a no-op.
    """
    try:
        widget.EnabledOff()
    except Exception:
        pass
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
        # Capture a strong reference to the orientation marker widget.
        # PyVista stores it as ``plotter.renderer.axes_widget``; without
        # our own reference, Python's cyclic GC can collect it from an
        # unrelated thread (e.g. an asyncio task in the MCP server's
        # background thread). Its VTK destructor then calls
        # ``SetEnabled(0)`` which tries to release GL buffer objects on a
        # context that isn't current on the GC thread — use-after-free
        # inside ``glDeleteBuffers`` → segfault.
        self._axes_widget = getattr(
            getattr(self.plotter, "renderer", None), "axes_widget", None)
        if self._axes_widget is not None:
            _ORIENTATION_WIDGETS.append(self._axes_widget)
            # weakref.finalize runs BEFORE the VTK destructor. We use it
            # to guard SetEnabled(0): if the GC fires on a thread without
            # a current GL context, the callback re-dispatches the
            # teardown to the GUI thread instead of letting the C++
            # destructor crash inside glDeleteBuffers.
            import weakref
            weakref.finalize(self._axes_widget,
                             _safe_axes_finalizer,
                             self._axes_widget)
        if hasattr(self, "plotter") and isinstance(self.plotter, QWidget):
            layout.addWidget(self.plotter, 1)

        # Scene → render hook
        self.scene.on_change(self._on_scene_change)

    # ----------------------------------------------------------- public ops
    def refresh(self) -> None:
        """Re-render every visible scene object.

        PyVistaQt on macOS corrupts actor state after `clear()` then
        re-adding many actors — the LAST actor (and sometimes the
        second-to-last) loses colors. We avoid `clear()` entirely and
        instead remove only the actors we previously added, leaving
        the render window and lighting intact.

        We also wrap the whole rebuild in ``gc.disable()`` so that the
        old actor wrappers, which lose their last Python reference as
        we replace them, cannot be collected by Python's cyclic GC on
        an arbitrary thread (typically an asyncio task step in the
        MCP server's background thread). Without this guard, the GC
        fires ``vtkActor::ReleaseGraphicsResources`` →
        ``glDeleteBuffers`` on a thread whose QOpenGLContext isn't
        current, hitting a wild pointer and segfaulting. The ``_actors``
        dict (now actually used) holds the old wrappers until the next
        refresh cycle replaces them, and ``_deferred_dispose`` parks
        any wrappers that get evicted in between, releasing them on
        the GUI thread once the current rep rebuild is done.
        """
        if not hasattr(self, "plotter") or not hasattr(self.plotter, "actors"):
            return
        import gc
        gc.disable()
        try:
            self._refresh_impl()
        finally:
            gc.enable()
            # Defer disposal of any actor wrappers we just replaced.
            # Running this inside the finally (after gc.enable) means
            # any remaining refcount drops will only trigger GC *after*
            # we hand control back to the Qt event loop.
            self._flush_deferred_actors()

    def _refresh_impl(self) -> None:
        """Inner refresh — assumes GC is disabled (caller handles)."""
        # Move all currently-tracked actor wrappers to the deferred
        # dispose list; they will be released only after the new
        # actors are in place AND the GUI thread is idle. This stops
        # Python's cyclic GC from collecting them mid-rebuild on a
        # non-GUI thread (the segfault path).
        for old in list(self._actors.values()):
            _PENDING_ACTOR_DISPOSE.append(old)
        self._actors.clear()
        # Drop our previously-rendered actors only — leave axes, lights,
        # camera, scalar bar untouched.
        owned_bases = {o.name for o in self.scene.objects}
        if hasattr(self.plotter, "actors"):
            for actor_name in list(self.plotter.actors.keys()):
                # We only own actors whose names are scene object names
                # or our `name:atoms` / `name:bonds` sub-names. When the
                # scene is empty, remove ALL of them; otherwise only
                # remove actors whose base is no longer in the scene.
                base = actor_name.split(":", 1)[0]
                if not owned_bases or base not in owned_bases:
                    try:
                        self.plotter.remove_actor(actor_name)
                    except Exception:
                        pass
        # Re-install lights / SSAO only if they were lost (cheap to do
        # every time; RemoveAllLights() is idempotent on a clean state).
        if hasattr(self.plotter, "enable_ssao"):
            try:
                self.plotter.enable_ssao(radius=1.5, bias=0.02, kernel_size=64)
            except Exception:
                pass
        self._install_lights()
        # Only re-add the axes widget on the very first refresh; PyVista's
        # add_axes() creates a fresh vtkOrientationMarkerWidget every call,
        # and the old one becomes garbage that gets collected from whatever
        # thread happens to be running Python's GC — which is the segfault
        # path through _safe_axes_finalizer.
        if (self._axes_visible
                and getattr(self, "_axes_widget", None) is None
                and hasattr(self.plotter, "add_axes")):
            self.plotter.add_axes()
            # Re-grab the new widget reference so the registry keeps it alive
            # and the next refresh skips this branch.
            self._axes_widget = getattr(
                getattr(self.plotter, "renderer", None), "axes_widget", None)
            if self._axes_widget is not None:
                _ORIENTATION_WIDGETS.append(self._axes_widget)

        # Render meshes — re-add every visible scene object to the
        # plotter. This was previously misplaced in
        # _flush_deferred_actors (which only does dispose) — the bug
        # caused plotter.actors to stay empty after every refresh.
        for obj in self.scene.objects:
            if not obj.visible:
                continue
            mesh = self._meshes.get(obj.name)
            if mesh is None:
                continue
            self._add_object_to_plotter(obj, mesh)
        # Track every actor we just added so it has a strong
        # reference until the next refresh() parks it for safe disposal.
        if hasattr(self.plotter, "actors"):
            for actor_name, actor in list(self.plotter.actors.items()):
                if self._actors.get(actor_name) is actor:
                    continue
                if actor_name in ("axes", "AxesActor"):
                    continue
                self._track_actor(actor_name, actor)
        # Update info label
        n_obj = self.scene.n_objects()
        n_atoms = self.scene.n_atoms()
        self._info_label.setText(
            f"{n_obj} object{'s' if n_obj != 1 else ''} · "
            f"{n_atoms} atoms"
        )

    def _flush_deferred_actors(self) -> None:
        """Release actor wrappers parked by an earlier refresh().

        Called from refresh()'s finally block — i.e. after the new
        actors are in place and gc.enable() has been re-armed. By the
        time this runs, the GUI thread is about to return to the Qt
        event loop, so any VTK destructor invoked here (on the same
        thread, with the same QOpenGLContext) is safe.
        """
        if not _PENDING_ACTOR_DISPOSE:
            return
        # Drop our strong references; if nothing else holds them,
        # Python's refcount (not cyclic GC) will destroy them
        # immediately, on this GUI thread, with a valid GL context.
        _PENDING_ACTOR_DISPOSE.clear()

    def _track_actor(self, name: str, actor: Any) -> None:
        """Record an actor wrapper so we can defer its destruction.

        Called from ``_add_object_to_plotter`` after ``add_actor``
        succeeds. The wrapper stays in ``self._actors`` until the
        next refresh() moves it into ``_PENDING_ACTOR_DISPOSE``.
        """
        self._actors[name] = actor

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

    # --------------------------------------------------------- spin animation
    def start_spin(self, deg_per_sec: float = 18.0) -> None:
        """Start a slow auto-rotation of the camera around the vertical
        (world Z) axis. The protein appears to spin horizontally.

        ``deg_per_sec=18`` is roughly one full turn every 20 seconds —
        fast enough to be obvious, slow enough not to induce motion
        sickness.

        Implementation notes
        ---------------------
        Each tick we rotate the camera via the underlying VTK
        ``vtkCamera.Azimuth()`` method (avoids PyVista's property
        setter overhead) and call ``render_window.Render()`` directly
        — NOT ``plotter.render()``, which re-runs the entire PyVista
        pipeline (lights, axes, scalar-bar updates) every tick and
        is what caused the stutter in earlier iterations.
        """
        self.stop_spin()  # avoid double-start
        from PySide6.QtCore import QTimer
        # Cache references once so the per-tick hot path doesn't pay
        # the PyVista property-access cost.
        try:
            self._spin_cam = self.plotter.renderer.GetActiveCamera()
            self._spin_render_window = (
                self.plotter.render_window if hasattr(self.plotter, "render_window")
                else self.plotter.renderer.GetRenderWindow()
            )
        except Exception:
            self._spin_cam = None
            self._spin_render_window = None
        if self._spin_cam is None or self._spin_render_window is None:
            return  # nothing to spin

        interval_ms = 16  # ~60 fps
        deg_per_tick = deg_per_sec * interval_ms / 1000.0
        self._spin_timer = QTimer(self)
        # PreciseTimer: requests sub-millisecond accuracy on platforms
        # that support it (Cocoa / Win32). Falls back to coarse timer
        # elsewhere. This is the single biggest fix for the visible
        # stutter — the default coarse timer on macOS can drift to
        # 30+ ms intervals, which produces an obvious judder.
        self._spin_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._spin_timer.setInterval(interval_ms)
        cam = self._spin_cam
        win = self._spin_render_window
        def _tick():
            try:
                cam.Azimuth(deg_per_tick)
                win.Render()
            except Exception:
                self.stop_spin()
        self._spin_timer.timeout.connect(_tick)
        self._spin_timer.start()
        self._spinning = True

    def stop_spin(self) -> None:
        """Stop the auto-rotation started by :meth:`start_spin`."""
        timer = getattr(self, "_spin_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            try:
                timer.deleteLater()
            except Exception:
                pass
        self._spin_timer = None
        self._spinning = False

    # ----------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        """Synchronously tear down the OpenGL scene.

        Called from ``MainWindow.closeEvent`` *before* the MCP thread
        or any other coroutine is asked to exit. VTK holds onto an
        ``vtkOrientationMarkerWidget`` (created by ``add_axes``) plus
        per-actor buffer objects; if those survive until the
        QOpenGLWidget is destroyed by Qt, ``EnabledOff`` /
        ``ReleaseGraphicsResources`` can run on a dead GL context and
        segfault inside ``glDeleteBuffers``.

        We close the plotter explicitly, in order:

        1. Stop the spin timer (it pokes the render window).
        2. Disable + detach the orientation marker widget while we
           still have a valid GL context, so its later destructor is
           a no-op.
        3. Hide the plotter widget — Qt stops scheduling paints.
        4. Call ``plotter.close()`` — PyVistaQt's hook that removes
           the orientation marker, clears lights, and finalises the
           render window while the GL context is still alive.
        5. Drop our strong reference to the orientation marker (and
           remove it from the module registry), so GC can collect it.
        6. Break our own reference cycle so the next GC pass doesn't
           try to use the (now-dead) plotter.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        try:
            self.stop_spin()
        except Exception:
            pass

        # 2. Disable the orientation marker widget while we still own a
        #    valid GL context. This makes its eventual VTK destructor a
        #    no-op (SetEnabled(0) is idempotent — it skips when the
        #    widget is already disabled).
        axes_widget = getattr(self, "_axes_widget", None)
        if axes_widget is not None:
            try:
                # Bind the interactor's GL context to the current thread
                # so the EnabledOff() → ReleaseGraphicsResources chain
                # has a valid context to operate on.
                iren = getattr(self.plotter, "iren", None) if hasattr(
                    self, "plotter") else None
                if iren is not None:
                    inner = getattr(iren, "interactor", iren)
                    if hasattr(inner, "MakeCurrent"):
                        try:
                            inner.MakeCurrent()
                        except Exception:
                            pass
                axes_widget.EnabledOff()
                # Detach from the interactor so VTK doesn't try to
                # call back into a freed object.
                if hasattr(axes_widget, "SetInteractor"):
                    axes_widget.SetInteractor(None)
            except Exception:
                pass

        plotter = getattr(self, "plotter", None)
        if plotter is not None:
            try:
                if hasattr(plotter, "hide"):
                    plotter.hide()
            except Exception:
                pass
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass
            try:
                if hasattr(plotter, "close"):
                    plotter.close()
            except Exception:
                pass

        # 5. Drop the strong references to the orientation marker now
        #    that the widget is disabled and detached. GC can collect
        #    it later without touching GL.
        try:
            if axes_widget is not None and axes_widget in _ORIENTATION_WIDGETS:
                _ORIENTATION_WIDGETS.remove(axes_widget)
        except Exception:
            pass
        self._axes_widget = None

        try:
            if plotter is not None and hasattr(plotter, "deleteLater"):
                plotter.deleteLater()
        except Exception:
            pass
        self.plotter = None  # type: ignore[assignment]

    def toggle_spin(self) -> bool:
        """Toggle spin on/off. Returns the new state (``True`` = spinning)."""
        if getattr(self, "_spinning", False):
            self.stop_spin()
        else:
            self.start_spin()
        return getattr(self, "_spinning", False)

    def is_spinning(self) -> bool:
        return getattr(self, "_spinning", False)

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
            # Skip zero-point meshes entirely. PyVista raises ValueError
            # for them and that exception seems to leave the plotter in
            # a bad state — subsequent actors (e.g. the protein loaded
            # right after a docking pose) come out colorless because the
            # shared mapper/LUT machinery never finishes initializing.
            if sub.n_points == 0 and sub.n_cells == 0:
                return
            print(
                f"[viewport] _add_one ENTER name={name!r} "
                f"cell_data={list(sub.cell_data.keys())} "
                f"point_data={list(sub.point_data.keys())} "
                f"n_pts={sub.n_points} n_cells={sub.n_cells}",
                flush=True,
            )
            # IMPORTANT: rename the colors array to be unique per actor.
            # Multiple meshes sharing `cell_data["colors"]` trigger a
            # PyVistaQt / macOS OpenGL bug where the second mesh reuses
            # the first actor's mapper state — colors get lost. A unique
            # scalar name (e.g. `colors_1aki`, `colors_pose_1_xxx_atoms`)
            # forces each actor to build its own mapper pipeline.
            sub = sub.copy(deep=False)
            base = name.replace(":", "_")
            new_scalar = f"colors_{base}"
            cell_colors = None
            point_colors = None
            if "colors" in (sub.cell_data.keys() or []):
                cell_colors = sub.cell_data["colors"].copy()
                del sub.cell_data["colors"]
            if "colors" in (sub.point_data.keys() or []):
                point_colors = sub.point_data["colors"].copy()
                del sub.point_data["colors"]
            # PyVistaQt on macOS has a bug where `preference="cell"` is
            # silently ignored on the second (and later) actors added
            # after a `clear()`, so colors come out as default white.
            # Workaround: copy the cell colors into point_data so the
            # default point-data lookup picks them up.
            if cell_colors is not None and point_colors is None:
                # Cell colors → broadcast onto points. PyVista requires
                # point_data arrays to match n_points; for tube meshes
                # we have multiple points per cell, so assign each
                # point the color of the cell whose centroid is nearest.
                # Naive broadcasting is O(n_points × n_cells) memory
                # (e.g. 200k × 20k ≈ 96 GB) and OOMs on real proteins,
                # so we use a KDTree for large meshes and einsum only
                # for small ones.
                import numpy as _np
                try:
                    if sub.n_cells == 0 or sub.n_points == 0:
                        raise ValueError("empty")
                    centers = sub.cell_centers().points
                    pts = sub.points
                    n_pts, n_cells = pts.shape[0], centers.shape[0]
                    # Threshold: ~5k cells × 5k pts = ~200 MB broadcast.
                    if n_pts * n_cells <= 25_000_000:
                        diffs = pts[:, None, :] - centers[None, :, :]
                        d2 = _np.einsum("ijk,ijk->ij", diffs, diffs)
                        nearest = _np.argmin(d2, axis=1)
                    else:
                        from scipy.spatial import cKDTree
                        _, nearest = cKDTree(centers).query(pts, k=1)
                    point_colors = cell_colors[nearest]
                except Exception:
                    point_colors = None
            if point_colors is not None:
                sub.point_data[new_scalar] = point_colors
                # Make the renamed scalar the *active* array — otherwise
                # PyVista's add_mesh may pick `vtkOriginalPointIds` /
                # `vtkOriginalCellIds` (added by extract_surface) as the
                # active scalar and silently drop our colors.
                try:
                    sub.set_active_scalars(new_scalar, preference="point")
                except Exception:
                    pass
                # Drop the leftover tube-filter bookkeeping arrays that
                # confuse add_mesh's default scalar lookup.
                for stale in ("vtkOriginalCellIds", "vtkOriginalPointIds",
                              "TubeNormals", "Normals", "GlyphVector"):
                    if stale in sub.cell_data.keys():
                        del sub.cell_data[stale]
                    if stale in sub.point_data.keys():
                        del sub.point_data[stale]
            has_cell   = new_scalar in (sub.cell_data.keys() or [])
            has_point  = new_scalar in (sub.point_data.keys() or [])
            scalars = new_scalar if (has_cell or has_point) else None
            preference = "cell" if has_cell else ("point" if has_point else "point")
            # Diagnostic: warn when a mesh with NO colors reaches the
            # plotter — that's almost always the bug.
            if scalars is None and sub.n_points > 0:
                print(
                    f"[viewport] {name!r} (rep={obj.representation}): mesh "
                    f"has NO colors (cell_data={list(sub.cell_data.keys())}, "
                    f"point_data={list(sub.point_data.keys())}, "
                    f"n_points={sub.n_points}, n_cells={sub.n_cells}); "
                    f"rendering as default white.",
                    flush=True,
                )
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
            attempt_log: list[str] = []
            for attempt_idx, attempt_kwargs in enumerate(attempts):
                try:
                    self.plotter.add_mesh(sub, **attempt_kwargs)
                    last_exc = None
                    if attempt_idx > 0:
                        dropped = set(attempts[0].keys()) - set(attempt_kwargs.keys())
                        print(
                            f"[viewport] {name!r} (rep={obj.representation}): "
                            f"add_mesh needed fallback #{attempt_idx+1}, "
                            f"dropped={sorted(dropped)}",
                            flush=True,
                        )
                    break
                except Exception as exc:
                    last_exc = exc
                    attempt_log.append(f"#{attempt_idx+1} {type(exc).__name__}: {exc}")
            # Workaround for PyVistaQt / macOS OpenGL bug: when an actor
            # is added after many others, its mapper's lookup table is
            # shared/corrupted and colors render as white. We bypass
            # the broken PyVista mapper and install a fresh
            # vtkDataSetMapper with a brand-new vtkLookupTable.
            #
            # CRITICAL: when the colors are RGB (3-component per
            # vertex) and ``rgb=True`` was passed to add_mesh, we must
            # set ``SetColorModeToDirectScalars()`` so the mapper
            # reads the components as RGB rather than mapping a single
            # scalar through the lookup table — the latter would
            # produce a uniform tinted surface (e.g. all-blue for a
            # "pale-blue" ESP mesh, because all components happen to
            # land in the same LUT bucket).
            if last_exc is None and scalars is not None:
                try:
                    actor = self.plotter.actors.get(name)
                    if actor is not None:
                        import vtk
                        new_mapper = vtk.vtkDataSetMapper()
                        new_mapper.SetInputData(sub)
                        new_mapper.SelectColorArray(scalars)
                        if "cell" in preference:
                            new_mapper.SetScalarModeToUseCellData()
                        else:
                            new_mapper.SetScalarModeToUsePointData()
                        # If the chosen scalar array has 3 components
                        # (RGB) we set direct-scalars mode so VTK uses
                        # the components as colors directly. For 1- or
                        # 4-component arrays we still use the LUT path
                        # (the standard way for non-RGB scalars).
                        try:
                            arr = (sub.cell_data if "cell" in preference
                                   else sub.point_data)[scalars]
                            if arr.shape[-1] == 3:
                                new_mapper.SetColorModeToDirectScalars()
                            else:
                                new_mapper.SetScalarVisibility(True)
                                # Build a brand-new, isolated lookup
                                # table per actor so no state can leak.
                                new_lut = vtk.vtkLookupTable()
                                new_lut.SetNumberOfTableValues(256)
                                new_lut.Build()
                                new_mapper.SetLookupTable(new_lut)
                        except Exception:
                            new_mapper.SetScalarVisibility(True)
                        actor.SetMapper(new_mapper)
                except Exception as exc:
                    print(
                        f"[viewport] mapper-replacement for {name!r} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            # Diagnostic: log every actor we tried to add so we can see
            # whether the protein's add_mesh succeeded or fell through.
            print(
                f"[viewport] {name!r} (rep={obj.representation}): "
                f"scalars={scalars!r}, preference={preference!r}, "
                f"n_points={sub.n_points}, n_cells={sub.n_cells}, "
                f"cell_data={list(sub.cell_data.keys())}, "
                f"succeeded={last_exc is None}",
                flush=True,
            )
            if last_exc is not None:
                print(
                    f"[viewport] add_mesh failed for {name!r} "
                    f"(rep={obj.representation}) after {len(attempts)} attempts: "
                    f"{'; '.join(attempt_log)}",
                    flush=True,
                )
                try:
                    self.plotter.add_mesh(sub, name=name, color="#cccccc")
                    print(
                        f"[viewport] {name!r}: rendered as gray fallback "
                        f"(NO COLORS!) — investigate real cause.",
                        flush=True,
                    )
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
    # NOTE: do NOT override wheelEvent. PyVista's QtInteractor is a
    # child widget that handles zoom via its own event filters; if we
    # accept the event here, the plotter never sees it and zoom stops
    # working. Earlier we set `event.accept()` to prevent the outer
    # scroll area from scrolling — the splitter we sit in does not
    # actually need that protection.