"""Outliner / scene-tree widget — a Blender-like left panel.

Lists all scene objects with their representation and color scheme.
Supports click-to-select, double-click-to-rename, and visibility toggles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QMenu, QInputDialog, QAbstractItemView,
)


class OutlinerWidget(QWidget):
    """Left-side panel listing scene objects."""

    selection_changed = Signal(str)        # active object name
    visibility_changed = Signal(str, bool)
    object_removed = Signal(str)

    def __init__(self, scene, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("Scene")
        header.setStyleSheet(
            "color: #888; padding: 4px 8px; background: #2d2d2d; "
            "border-bottom: 1px solid #1a1a1a; font-size: 11px;"
        )
        layout.addWidget(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Name", "Representation", "Color"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(
            "QTreeWidget { background: #1d1d1d; color: #dcdcdc; "
            "  alternate-background-color: #232323; border: none; }"
            "QTreeWidget::item:selected { background: #3b76c2; }"
            "QHeaderView::section { background: #2d2d2d; color: #888; "
            "  border: none; padding: 4px 8px; }"
        )
        self.tree.setColumnWidth(0, 24)
        self.tree.setColumnWidth(1, 160)
        self.tree.setColumnWidth(2, 110)
        layout.addWidget(self.tree, 1)

        # Buttons row
        btn_row = QHBoxLayout()
        for label, slot in (
            ("+",    self._on_add),
            ("-",    self._on_remove),
            ("Fit",  self._on_fit),
        ):
            b = QPushButton(label)
            b.setFixedWidth(28 if label != "Fit" else 50)
            b.setStyleSheet(
                "QPushButton { background: #3a3a3a; color: #dcdcdc; "
                "padding: 3px; border: 1px solid #555; border-radius: 3px; }"
                "QPushButton:hover { background: #4a4a4a; }"
            )
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Wire up
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_click)

        self.scene.on_change(self._on_scene_change)

    # --------------------------------------------------- scene -> tree sync
    def _on_scene_change(self, kind: str, payload: Any) -> None:
        if kind in ("add", "update", "clear", "remove"):
            self.refresh()

    def refresh(self) -> None:
        prev = self.tree.currentItem().text(1) if self.tree.currentItem() else None
        self.tree.blockSignals(True)
        self.tree.clear()
        for obj in self.scene.objects:
            self._add_item(obj)
        self.tree.blockSignals(False)
        # Restore selection
        if prev:
            for i in range(self.tree.topLevelItemCount()):
                it = self.tree.topLevelItem(i)
                if it.text(1) == prev:
                    self.tree.setCurrentItem(it)
                    break

    def _add_item(self, obj) -> None:
        item = QTreeWidgetItem()
        item.setText(1, obj.name)
        item.setText(2, obj.representation)
        item.setText(3, obj.color_scheme.name)
        item.setData(0, Qt.ItemDataRole.UserRole, obj.name)
        item.setToolTip(1, obj.molecule.summary())
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        # Visibility column
        item.setCheckState(0, Qt.CheckState.Checked if obj.visible else Qt.CheckState.Unchecked)
        self.tree.addTopLevelItem(item)

    # ----------------------------------------------------- user interaction
    def _on_selection_changed(self) -> None:
        it = self.tree.currentItem()
        if not it:
            return
        name = it.data(0, Qt.ItemDataRole.UserRole)
        self.scene.set_active(name)
        self.selection_changed.emit(name)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        name = item.data(0, Qt.ItemDataRole.UserRole)
        if column == 0:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self.scene.update_object(name, visible=visible)
            self.visibility_changed.emit(name, visible)
            self.scene.notify("update")
        elif column in (2, 3):
            # Rename / representation / scheme edited inline — handled by double-click
            pass

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 1:
            return
        old = item.text(1)
        new, ok = QInputDialog.getText(self, "Rename object", "Name:", text=old)
        if ok and new and new != old:
            obj = self.scene.find(old)
            if obj:
                obj.name = new
                self.scene._emit("update", obj)
                self.refresh()

    def _show_context_menu(self, pos) -> None:
        it = self.tree.itemAt(pos)
        if not it:
            return
        menu = QMenu(self)
        menu.addAction("Rename",   lambda: self._on_double_click(it, 1))
        menu.addAction("Duplicate", lambda: self._on_duplicate(it))
        menu.addSeparator()
        menu.addAction("Delete",    lambda: self._on_remove(it))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_duplicate(self, item: QTreeWidgetItem) -> None:
        name = item.data(0, Qt.ItemDataRole.UserRole)
        obj = self.scene.find(name)
        if not obj:
            return
        import copy
        new_obj = copy.deepcopy(obj)
        new_obj.name = f"{name}_copy"
        # Avoid deepcopying the molecule's atom coords (deepcopy is fine but heavy)
        self.scene.add(new_obj)

    # ------------------------------------------------------ toolbar slots
    def _on_add(self) -> None:
        # Trigger main window's open dialog
        if hasattr(self.parent(), "open_pdb_dialog"):
            self.parent().open_pdb_dialog()

    def _on_remove(self, item=None) -> None:
        if item is None:
            item = self.tree.currentItem()
        if not item:
            return
        name = item.data(0, Qt.ItemDataRole.UserRole)
        self.scene.remove(name)
        self.object_removed.emit(name)

    def _on_fit(self) -> None:
        if hasattr(self.parent(), "viewport"):
            self.parent().viewport.center_view()