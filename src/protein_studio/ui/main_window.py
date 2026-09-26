"""Main application window — Blender-like layout.

Layout
------

    ┌─ Menu bar ────────────────────────────────────────────┐
    ├─ Toolbar (Open, Save, Snapshot, Refresh, …) ──────────┤
    ├────────────┬───────────────────────────┬──────────────┤
    │  Outliner  │       3D Viewport         │  Properties  │
    │            │                           │              │
    ├────────────┴───────────────────────────┴──────────────┤
    │  Python console (collapsible)                          │
    └────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QSplitter, QWidget, QVBoxLayout,
    QToolBar, QStatusBar, QLabel, QDockWidget, QApplication, QPushButton,
    QHBoxLayout, QToolButton, QTabWidget,
)

from ..api import StudioAPI
from ..core.io import load_pdb, save_pdb
from ..core.scene import Scene
from .console import PythonConsole
from .editor import PythonEditor
from .outliner import OutlinerWidget
from .properties import PropertiesWidget
from .viewport import ViewportWidget


APP_NAME = "LingSu Protein"
APP_VERSION = "0.1.0"


class MainWindow(QMainWindow):
    """The main window."""

    def __init__(self, scene: Scene, viewport: ViewportWidget,
                 outliner: OutlinerWidget, properties: PropertiesWidget,
                 console: PythonConsole, editor: PythonEditor,
                 api: StudioAPI,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.viewport = viewport
        self.outliner = outliner
        self.properties = properties
        self.console = console
        self.editor = editor
        self.api = api

        self.setWindowTitle(APP_NAME)
        # Use the bundled icon (also set on QApplication, but having it
        # on the window too means alt-tab and screenshots pick it up).
        try:
            from ..resources import load_app_icon
            ic = load_app_icon()
            if not ic.isNull():
                self.setWindowIcon(ic)
        except Exception:
            pass
        self.resize(1280, 800)
        self.setStyleSheet(
            "QMainWindow { background: #2d2d2d; }"
            "QMenuBar { background: #2d2d2d; color: #dcdcdc; }"
            "QMenuBar::item:selected { background: #3a3a3a; }"
            "QMenu { background: #2d2d2d; color: #dcdcdc; border: 1px solid #1a1a1a; }"
            "QMenu::item:selected { background: #3b76c2; }"
            "QStatusBar { background: #1d1d1d; color: #888; }"
            "QToolBar { background: #2d2d2d; border-bottom: 1px solid #1a1a1a; spacing: 4px; }"
        )

        self._build_menus()
        self._build_toolbar()
        self._build_central()

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._status_label = QLabel("Ready")
        self.status.addWidget(self._status_label)

        # Wire events
        self.properties.object_modified.connect(self._on_object_modified)
        self.scene.on_change(self._on_scene_change)

    # --------------------------------------------------------------- ui build
    def _build_menus(self) -> None:
        menubar = self.menuBar()
        # File
        m_file = menubar.addMenu("&File")
        m_file.addAction(self._mk_action("Open PDB…", "Ctrl+O", self.open_pdb_dialog))
        m_file.addAction(self._mk_action("Open Docking…", "Ctrl+D", self.open_docking_dialog))
        m_file.addAction(self._mk_action("Open Trajectory…", "Ctrl+T", self.open_trajectory_dialog))
        m_file.addAction(self._mk_action("Open AlphaFold…", "Ctrl+Shift+A", self.open_alphafold_dialog))
        m_file.addAction(self._mk_action("Save PDB…", "Ctrl+S", self.save_pdb_dialog))
        m_file.addSeparator()
        m_file.addAction(self._mk_action("Snapshot…", "F12", self.take_snapshot))
        m_file.addSeparator()
        m_file.addAction(self._mk_action("Quit", "Ctrl+Q", self.close))

        # Edit
        m_edit = menubar.addMenu("&Edit")
        m_edit.addAction(self._mk_action("Clear Scene", "Ctrl+Shift+C", self.clear_scene))
        m_edit.addAction(self._mk_action("Center View",  "Numpad .",       self.center_view))

        # View
        m_view = menubar.addMenu("&View")
        m_view.addAction(self._mk_action("Fit Camera", "Home", self.viewport.fit_view))
        m_view.addAction(self._mk_action("Toggle Axes", None,   self.viewport.toggle_axes))

        # Render
        m_render = menubar.addMenu("&Render")
        m_render.addAction(self._mk_action("Refresh", "F5", self.refresh_viewport))
        m_render.addAction(self._mk_action("Toggle Wireframe", "Z", self.toggle_wireframe))

        # Help
        m_help = menubar.addMenu("&Help")
        m_help.addAction(self._mk_action("About", None, self.show_about))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        for label, slot in (
            ("Open",       self.open_pdb_dialog),
            ("Docking",    self.open_docking_dialog),
            ("Trajectory", self.open_trajectory_dialog),
            ("AlphaFold",  self.open_alphafold_dialog),
            ("Save",       self.save_pdb_dialog),
            ("Snapshot",   self.take_snapshot),
            ("Refresh",    self.refresh_viewport),
            ("Center",     self.center_view),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn.setStyleSheet(
                "QPushButton { background: #3a3a3a; color: #dcdcdc; "
                "padding: 4px 12px; border: 1px solid #555; border-radius: 3px; }"
                "QPushButton:hover { background: #4a4a4a; }"
            )
            tb.addWidget(btn)
        # Spin button — toggleable, starts/stops slow camera orbit around
        # the vertical (world Z) axis. Highlighted while active.
        self._spin_btn = QPushButton("Spin")
        self._spin_btn.setCheckable(True)
        self._spin_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #dcdcdc; "
            "padding: 4px 12px; border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { background: #4a4a4a; }"
            "QPushButton:checked { background: #3b76c2; color: #fff; "
            "border: 1px solid #5b96e2; }"
        )
        self._spin_btn.toggled.connect(self._on_spin_toggled)
        tb.addWidget(self._spin_btn)
        tb.addSeparator()
        self._render_label = QLabel("Cartoon")
        self._render_label.setStyleSheet("color: #888; padding: 4px 12px;")
        tb.addWidget(self._render_label)

    def _on_spin_toggled(self, checked: bool) -> None:
        """Toolbar Spin button handler — start/stop the slow orbit."""
        if checked:
            self.viewport.start_spin()
        else:
            self.viewport.stop_spin()

    def closeEvent(self, event) -> None:
        """Make sure the spin QTimer is stopped when the window closes,
        otherwise it could leak into the next session."""
        try:
            self.viewport.stop_spin()
        except Exception:
            pass
        super().closeEvent(event)

    def _build_central(self) -> None:
        # Splitter: outliner | viewport+console | properties
        outer = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self.outliner)
        outer.addWidget(self._build_center_stack())
        outer.addWidget(self.properties)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
        outer.setStretchFactor(2, 0)
        outer.setSizes([220, 760, 280])
        outer.setStyleSheet("QSplitter::handle { background: #1a1a1a; }")
        self.setCentralWidget(outer)

    def _build_center_stack(self) -> QWidget:
        v = QSplitter(Qt.Orientation.Vertical)
        v.addWidget(self.viewport)
        # Tabbed bottom pane: REPL console + script editor share the
        # same Python namespace so variables / imports / the API
        # handle (`D`) cross over freely.
        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setDocumentMode(True)
        self._bottom_tabs.setStyleSheet(
            "QTabWidget::pane { border-top: 1px solid #1a1a1a; background: #1d1d1d; }"
            "QTabBar::tab { background: #2a2a2a; color: #dcdcdc; padding: 6px 16px; "
            "border: 1px solid #1a1a1a; border-bottom: none; }"
            "QTabBar::tab:selected { background: #1d1d1d; color: #fff; "
            "border-bottom: 2px solid #3b76c2; }"
            "QTabBar::tab:hover { background: #353535; }"
        )
        self._bottom_tabs.addTab(self.console, "Console")
        self._bottom_tabs.addTab(self.editor, "Script")
        v.addWidget(self._bottom_tabs)
        v.setStretchFactor(0, 3)
        v.setStretchFactor(1, 1)
        v.setStyleSheet("QSplitter::handle { background: #1a1a1a; }")
        return v

    # --------------------------------------------------------------- helpers
    def _mk_action(self, label: str, shortcut, slot) -> QAction:
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        return a

    # --------------------------------------------------------------- slots
    def open_pdb_dialog(self) -> None:
        start_dir = str(Path(__file__).resolve().parents[3] / "examples")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PDB file", start_dir,
            "PDB files (*.pdb *.ent *.cif);;All files (*.*)",
        )
        if not path:
            return
        try:
            obj = self.api.load(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Failed to load PDB: {e}")
            return
        self._status_label.setText(f"Loaded {obj.molecule.summary()}")
        self.refresh_viewport()

    def save_pdb_dialog(self) -> None:
        if not self.scene.objects:
            QMessageBox.information(self, APP_NAME, "No objects to save.")
            return
        obj = self.scene.active or self.scene.objects[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDB file", f"{obj.name}.pdb",
            "PDB files (*.pdb)",
        )
        if not path:
            return
        try:
            save_pdb(obj.molecule, path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Failed to save: {e}")
            return
        self._status_label.setText(f"Saved {path}")

    def open_docking_dialog(self) -> None:
        """Open a Vina PDBQT or EquiScore directory of poses."""
        start = str(Path(__file__).resolve().parents[3] / "examples")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open docking results", start,
            "Docking files (*.pdbqt);;PDB files (*.pdb);;All files (*.*)",
        )
        if not path:
            return
        try:
            poses = self.api.load_docking(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Failed to load docking: {e}")
            return
        self._status_label.setText(
            f"Loaded {len(poses)} docking poses ({poses[0].score:.2f} to "
            f"{poses[-1].score:.2f} kcal/mol)"
        )
        self.refresh_viewport()

    def open_trajectory_dialog(self) -> None:
        """Open a topology PDB + trajectory file pair."""
        start = str(Path(__file__).resolve().parents[3] / "examples")
        top, _ = QFileDialog.getOpenFileName(
            self, "Open topology PDB", start,
            "PDB files (*.pdb)",
        )
        if not top:
            return
        traj_path, _ = QFileDialog.getOpenFileName(
            self, "Open trajectory", os.path.dirname(top),
            "Trajectory files (*.dcd *.xtc *.nc *.trr);;All files (*.*)",
        )
        if not traj_path:
            return
        # Load the topology as a scene object and bind the trajectory
        try:
            obj = self.api.load(top)
            traj = self.api.load_trajectory(obj.name, top, traj_path)
            self.api.metadata.setdefault("trajectories", {})[obj.name] = traj
            self._status_label.setText(
                f"Loaded trajectory {traj_path} ({traj.n_frames} frames) "
                f"— use D.frame('{obj.name}', i, '{obj.name}') in console"
            )
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Failed to load trajectory: {e}")

    def open_alphafold_dialog(self) -> None:
        """Open an AlphaFold / Boltz-2 PDB; auto-applies pLDDT colouring."""
        start = str(Path(__file__).resolve().parents[3] / "examples")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open AlphaFold prediction", start,
            "PDB files (*.pdb *.cif);;All files (*.*)",
        )
        if not path:
            return
        try:
            obj = self.api.load_alphafold(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Failed to load: {e}")
            return
        info = self.api.plddt_summary(obj.name)
        self._status_label.setText(
            f"Loaded AlphaFold {obj.name} — mean pLDDT {info['mean_plddt']:.1f}"
        )
        self.refresh_viewport()

    def take_snapshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", "screenshot.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        try:
            self.viewport.screenshot(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Snapshot failed: {e}")
            return
        self._status_label.setText(f"Snapshot saved to {path}")

    def clear_scene(self) -> None:
        self.scene.clear()
        self.refresh_viewport()
        self._status_label.setText("Scene cleared")

    def center_view(self) -> None:
        self.viewport.center_view()
        self._status_label.setText("Centered view on centroid")

    def refresh_viewport(self) -> None:
        # Rebuild meshes only for objects whose (representation, scheme,
        # molecule-identity) has changed; reuse the cached mesh otherwise.
        # This skips the ~250 ms `_propagate_cell_colors` cost on every
        # toggle. After mesh updates we re-render the GUI viewport only
        # (no need to round-trip through the headless path).
        from ..core.representations import build_mesh
        cache: dict[tuple, object] = getattr(self, "_mesh_cache", {})
        new_cache: dict[tuple, object] = {}
        for obj in self.scene.objects:
            if not obj.visible:
                continue
            key = (
                obj.name,
                obj.representation,
                id(obj.color_scheme),
                id(obj.molecule),
            )
            mesh = cache.get(key)
            if mesh is None:
                mesh = build_mesh(obj, obj.color_scheme)
            self.viewport._meshes[obj.name] = mesh
            new_cache[key] = mesh
        self._mesh_cache = new_cache
        # Drop meshes for objects that are no longer visible / present
        live = {o.name for o in self.scene.objects}
        for n in list(self.viewport._meshes):
            if n not in live:
                del self.viewport._meshes[n]
        self.viewport.refresh()

    def toggle_wireframe(self) -> None:
        # Toggle between solid and wireframe for every actor (best-effort)
        if not hasattr(self.viewport.plotter, "actors"):
            return
        # We don't track per-actor state — just request a re-render
        self.refresh_viewport()

    def show_about(self) -> None:
        QMessageBox.information(
            self, "About",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "A Blender-like molecular viewer.\n"
            "Python scripting via the embedded console.",
        )

    def _on_object_modified(self) -> None:
        self.refresh_viewport()

    def _on_scene_change(self, kind: str, payload) -> None:
        if kind in ("add", "remove", "clear"):
            self.refresh_viewport()
        if kind == "update":
            self.refresh_viewport()