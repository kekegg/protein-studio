"""Right-side properties panel — Blender-like.

Shows / edits the active object's:

- Name
- Representation (cartoon / ball-and-stick / surface / etc.)
- Color scheme
- Transform (position, rotation, scale)
- Visible / hidden
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QComboBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QGroupBox, QScrollArea, QHBoxLayout, QSlider,
)

from ..core.representations import available_representations
from ..core.colors import SCHEME_PRESETS
from ..core.scene import Scene


class PropertiesWidget(QWidget):
    """Blender-style properties panel for the active object."""

    object_modified = Signal()

    def __init__(self, scene: Scene, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._block = False  # avoid recursion when refreshing

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build())

        # Re-render when scene changes
        self.scene.on_change(self._on_scene_change)
        self.refresh()

    # ------------------------------------------------------- construction
    def _build(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: #1d1d1d; border: none; }"
            "QWidget#propsRoot { background: #1d1d1d; color: #dcdcdc; }"
        )
        root = QWidget()
        root.setObjectName("propsRoot")
        v = QVBoxLayout(root)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Header
        header = QLabel("Properties")
        header.setStyleSheet("color: #888; font-size: 11px; padding: 4px 0;")
        v.addWidget(header)

        self._empty_label = QLabel("No active object")
        self._empty_label.setStyleSheet("color: #888; padding: 8px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._empty_label)

        # Object panel
        self._obj_group = QGroupBox("Object")
        self._obj_form = QFormLayout(self._obj_group)
        self._obj_form.setContentsMargins(8, 8, 8, 8)
        self._obj_form.setSpacing(6)
        self._obj_summary = QLabel("")
        self._obj_summary.setWordWrap(True)
        self._obj_summary.setStyleSheet("color: #dcdcdc; font-size: 11px;")
        self._obj_form.addRow(self._obj_summary)

        self._representation = QComboBox()
        self._representation.addItems(available_representations())
        self._representation.currentTextChanged.connect(self._on_representation_changed)
        self._obj_form.addRow("Representation", self._representation)

        self._scheme = QComboBox()
        self._scheme.addItems(list(SCHEME_PRESETS.keys()))
        self._scheme.currentTextChanged.connect(self._on_scheme_changed)
        self._obj_form.addRow("Color scheme", self._scheme)

        self._visible = QCheckBox("Visible in viewport")
        self._visible.stateChanged.connect(self._on_visible_changed)
        self._obj_form.addRow(self._visible)

        v.addWidget(self._obj_group)
        self._obj_group.hide()

        # Transform panel
        self._tr_group = QGroupBox("Transform")
        tr_form = QFormLayout(self._tr_group)
        self._pos = [self._spin() for _ in range(3)]
        self._rot = [self._spin() for _ in range(3)]
        self._scl = [self._spin(value=1.0, minv=-10.0, maxv=10.0) for _ in range(3)]
        for label, group in (("Position", self._pos), ("Rotation (rad)", self._rot), ("Scale", self._scl)):
            row = QHBoxLayout()
            for sb in group:
                row.addWidget(sb)
                sb.editingFinished.connect(self._on_transform_changed)
            w = QWidget()
            w.setLayout(row)
            tr_form.addRow(label, w)
        v.addWidget(self._tr_group)
        self._tr_group.hide()

        # Misc
        v.addStretch()
        scroll.setWidget(root)
        return scroll

    def _spin(self, value: float = 0.0, minv: float = -1e6, maxv: float = 1e6) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(minv, maxv)
        sb.setDecimals(3)
        sb.setSingleStep(0.1)
        sb.setValue(value)
        sb.setStyleSheet("background: #2a2a2a; color: #dcdcdc; border: 1px solid #444;")
        return sb

    # ----------------------------------------------------- scene changes
    def _on_scene_change(self, kind: str, payload: Any) -> None:
        if kind in ("active", "add", "update", "remove", "clear"):
            self.refresh()

    def refresh(self) -> None:
        obj = self.scene.active
        self._block = True
        if obj is None:
            self._empty_label.show()
            self._obj_group.hide()
            self._tr_group.hide()
        else:
            self._empty_label.hide()
            self._obj_group.show()
            self._tr_group.show()
            self._obj_summary.setText(obj.molecule.summary())
            if self._representation.currentText() != obj.representation:
                self._representation.setCurrentText(obj.representation)
            if self._scheme.currentText() != obj.color_scheme.key:
                self._scheme.setCurrentText(obj.color_scheme.key)
            self._visible.setChecked(obj.visible)
            for i, sb in enumerate(self._pos):
                if sb.value() != obj.position[i]:
                    sb.setValue(float(obj.position[i]))
            for i, sb in enumerate(self._rot):
                if sb.value() != obj.rotation[i]:
                    sb.setValue(float(obj.rotation[i]))
            for i, sb in enumerate(self._scl):
                if sb.value() != obj.scale[i]:
                    sb.setValue(float(obj.scale[i]))
        self._block = False

    # ------------------------------------------------------- user edits
    def _on_representation_changed(self, name: str) -> None:
        if self._block:
            return
        obj = self.scene.active
        if not obj:
            return
        self.scene.update_object(obj.name, representation=name)
        self.scene.notify("update")
        self.object_modified.emit()

    def _on_scheme_changed(self, key: str) -> None:
        if self._block:
            return
        obj = self.scene.active
        if not obj:
            return
        if key in SCHEME_PRESETS:
            self.scene.update_object(obj.name, color_scheme=SCHEMES_PRESETS_FIXED[key])
            self.scene.notify("update")
            self.object_modified.emit()

    def _on_visible_changed(self, state: int) -> None:
        if self._block:
            return
        obj = self.scene.active
        if not obj:
            return
        self.scene.update_object(obj.name, visible=bool(state))
        self.scene.notify("update")
        self.object_modified.emit()

    def _on_transform_changed(self) -> None:
        if self._block:
            return
        obj = self.scene.active
        if not obj:
            return
        import numpy as np
        pos = np.array([sb.value() for sb in self._pos])
        rot = np.array([sb.value() for sb in self._rot])
        scl = np.array([sb.value() for sb in self._scl])
        self.scene.update_object(obj.name, position=pos, rotation=rot, scale=scl)
        self.scene.notify("update")
        self.object_modified.emit()


# A small re-export to keep the local name unambiguous
SCHEMES_PRESETS_FIXED = SCHEME_PRESETS