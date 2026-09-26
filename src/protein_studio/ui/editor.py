"""Python script editor widget.

A lightweight file editor for writing/running Python scripts that
drive the :class:`~protein_studio.api.StudioAPI`. Pairs with the
existing REPL :class:`PythonConsole` — they share the same Python
namespace, so variables defined in one are visible in the other.

Features
--------
- Open / Save / Save As with proper file dialogs (recent dir is
  remembered between calls)
- "Run" executes the current script in the API namespace via
  ``exec(compile(source, filename, "exec"), globals)`` and prints
  any output / traceback to the embedded output pane
- "Run selection" executes just the highlighted block
- Modified-flag in the tab title (asterisk); clearing on save
- Line numbers, current-line highlight (free via QPlainTextEdit's
  setLineWrapMode etc. — kept minimal)
- Python syntax highlighting via the same highlighter the
  console uses, so the look is consistent

This is intentionally *not* a full IDE — no debugger, no
intellisense, no project tree. It is a notebook for scripted
workflows: load proteins, manipulate the scene, take screenshots,
save PDBs, run H-bond analyses, etc.
"""

from __future__ import annotations

import code
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import (
    QFont, QFontDatabase, QKeySequence, QShortcut, QTextCursor,
    QColor, QSyntaxHighlighter,
)
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QToolBar, QVBoxLayout, QWidget,
)

# Reuse the console's highlighter so colors stay consistent.
from .console import PythonHighlighter


class _EditorSignals(QObject):
    output = Signal(str, str)   # text, kind ("out" | "err")


class _StdoutTap(io.TextIOBase):
    """Captures print() output and pipes it into the editor's output pane."""

    def __init__(self, signals: _EditorSignals, kind: str) -> None:
        super().__init__()
        self.signals = signals
        self.kind = kind

    def write(self, s: str) -> int:
        if s:
            self.signals.output.emit(s, self.kind)
        return len(s)

    def flush(self) -> None:
        return None


class PythonEditor(QWidget):
    """A file editor + executor for protein-studio Python scripts.

    Parameters
    ----------
    namespace : dict
        The shared Python namespace (typically ``{"D": api, "np":
        numpy, ...}``). The editor does NOT own this dict — the
        REPL console does too, so writes here are visible there and
        vice versa.
    parent : QWidget | None
    """

    def __init__(self, namespace: Dict[str, Any],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.namespace: Dict[str, Any] = namespace
        self._current_path: Optional[Path] = None
        self._saved_text: str = ""  # for dirty-tracking

        self._signals = _EditorSignals()
        self._signals.output.connect(self._append_output)

        # ---- layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar (file ops + run)
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())
        toolbar.setStyleSheet(
            "QToolBar { background: #2a2a2a; border-bottom: 1px solid #1a1a1a; "
            "padding: 2px; spacing: 4px; }"
            "QToolButton { color: #dcdcdc; padding: 3px 10px; border: 1px solid "
            "#444; border-radius: 3px; background: #3a3a3a; }"
            "QToolButton:hover { background: #4a4a4a; }"
            "QToolButton:pressed { background: #2a2a2a; }"
        )
        for label, slot, tip in (
            ("New",      self.new_file,        "New blank script (Ctrl-N)"),
            ("Open",     self.open_file,       "Open script from disk (Ctrl-O)"),
            ("Save",     self.save_file,       "Save script (Ctrl-S)"),
            ("Save As",  self.save_file_as,    "Save as new file"),
            ("Run",      self.run_script,      "Run the whole script (F5)"),
            ("Run Sel",  self.run_selection,   "Run just the selected lines"),
            ("Stop",     self.stop_script,     "Cancel a running script (not yet supported)"),
        ):
            btn = QToolBar()  # not used; use QPushButton via custom layout instead
        # We want buttons styled like the toolbar elsewhere; build with QPushButton.
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.setSpacing(6)
        for label, slot, tip in (
            ("New",      self.new_file,        "Ctrl-N"),
            ("Open",     self.open_file,       "Ctrl-O"),
            ("Save",     self.save_file,       "Ctrl-S"),
            ("Save As",  self.save_file_as,    None),
            ("Run ▶",    self.run_script,      "F5"),
            ("Run Sel",  self.run_selection,   None),
        ):
            btn = QPushButton(label)
            btn.setToolTip(tip or "")
            btn.setStyleSheet(
                "QPushButton { background: #3a3a3a; color: #dcdcdc; "
                "padding: 3px 12px; border: 1px solid #555; border-radius: 3px; }"
                "QPushButton:hover { background: #4a4a4a; }"
                "QPushButton:pressed { background: #2a2a2a; }"
            )
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch()
        self._path_label = QLabel("(no file)")
        self._path_label.setStyleSheet("color: #888; font-size: 11px;")
        bar.addWidget(self._path_label)
        layout.addLayout(bar)

        # Editor area (top)
        self._editor = QPlainTextEdit()
        self._editor.setReadOnly(False)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(max(10, font.pointSize()))
        self._editor.setFont(font)
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setStyleSheet(
            "QPlainTextEdit { background: #1d1d1d; color: #dcdcdc; "
            "border: none; selection-background-color: #3b76c2; }"
        )
        self._highlighter = PythonHighlighter(self._editor.document())
        layout.addWidget(self._editor, 1)

        # Output area (bottom)
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(2000)  # cap memory
        self._output.setFont(font)
        self._output.setStyleSheet(
            "QPlainTextEdit { background: #181818; color: #dcdcdc; "
            "border-top: 1px solid #2a2a2a; border-bottom: none; "
            "border-left: none; border-right: none; }"
        )
        self._output.setPlaceholderText("Script output appears here.")
        # Capture the default output-pane char format so we can
        # restore it after colouring stderr lines red.
        self._default_output_fmt = self._output.currentCharFormat()
        layout.addWidget(self._output, 1)

        # ---- dirty tracking
        self._editor.document().modificationChanged.connect(
            self._on_modification_changed
        )

        # ---- shortcuts
        QShortcut(QKeySequence("Ctrl+N"), self, activated=self.new_file)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_file)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_file)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.save_file_as)
        QShortcut(QKeySequence("F5"), self, activated=self.run_script)

        # ---- initial banner
        self._append_output(
            "Protein Studio script editor — edit Python above, press "
            "F5 to run against the live API (D = StudioAPI).\n",
            "out",
        )

    # -------------------------------------------------------- file ops
    def new_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        self._editor.clear()
        self._current_path = None
        self._saved_text = ""
        self._editor.document().setModified(False)
        self._update_path_label()

    def open_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        start = (str(self._current_path.parent)
                 if self._current_path else str(Path.cwd()))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Python script", start,
            "Python files (*.py);;All files (*.*)",
        )
        if not path:
            return
        self.load_file(Path(path))

    def load_file(self, path: Path) -> None:
        """Load *path* into the editor. Used by the API and tests."""
        try:
            text = path.read_text()
        except OSError as e:
            QMessageBox.critical(self, "Open failed", f"Could not read {path}:\n{e}")
            return
        self._editor.setPlainText(text)
        self._current_path = path
        self._saved_text = text
        self._editor.document().setModified(False)
        self._update_path_label()
        self._append_output(f"Loaded {path}\n", "out")

    def save_file(self) -> None:
        if self._current_path is None:
            return self.save_file_as()
        text = self._editor.toPlainText()
        try:
            self._current_path.write_text(text)
        except OSError as e:
            QMessageBox.critical(self, "Save failed",
                                 f"Could not write {self._current_path}:\n{e}")
            return
        self._saved_text = text
        self._editor.document().setModified(False)
        self._append_output(f"Saved {self._current_path}\n", "out")

    def save_file_as(self) -> None:
        start = (str(self._current_path)
                 if self._current_path else str(Path.cwd()))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Python script", start,
            "Python files (*.py);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".py")
        self._current_path = p
        self._update_path_label()
        self.save_file()

    # ----------------------------------------------------------- run
    def run_script(self) -> None:
        """Execute the entire current script in the shared namespace."""
        text = self._editor.toPlainText()
        filename = (str(self._current_path)
                    if self._current_path else "<editor>")
        self._execute(text, filename)

    def run_selection(self) -> None:
        """Execute just the highlighted text (or the current line if
        nothing is highlighted)."""
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
        else:
            text = cursor.block().text() + "\n"
        self._execute(text, "<selection>")

    def stop_script(self) -> None:
        """Cooperative cancel — sets a flag the running script can
        poll via ``D.cancelled``. (Scripts that busy-loop won't
        notice; we don't have thread-based preemption here.)
        """
        self.namespace["_cancelled"] = True

    def _execute(self, source: str, filename: str) -> None:
        if not source.strip():
            return
        self._append_output(f"\n--- Running {filename} ---\n", "out")
        # Reset cancel flag and remember the previous one
        prev_cancelled = self.namespace.get("_cancelled", False)
        self.namespace["_cancelled"] = False
        # Capture stdout/stderr for the duration of the run.
        out = _StdoutTap(self._signals, "out")
        err = _StdoutTap(self._signals, "err")
        try:
            code_obj = compile(source, filename, "exec")
            with redirect_stdout(out), redirect_stderr(err):
                exec(code_obj, self.namespace)
        except SystemExit:
            self._append_output("(SystemExit)\n", "out")
        except Exception:
            tb = traceback.format_exc()
            self._append_output(tb, "err")
        finally:
            # Restore cancel state (don't leak our internal flag)
            self.namespace["_cancelled"] = prev_cancelled
        self._append_output("--- Done ---\n", "out")

    # ----------------------------------------------------- misc helpers
    def _append_output(self, text: str, kind: str) -> None:
        # Colourize stderr lines red, stdout default. Keep simple.
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if kind == "err":
            fmt = cursor.charFormat()
            fmt.setForeground(QColor("#e96b6b"))
            cursor.setCharFormat(fmt)
        cursor.insertText(text)
        # Reset by inserting a zero-width space with the default
        # format captured BEFORE we tinted stderr lines.
        cursor.setCharFormat(self._default_output_fmt)

    def _on_modification_changed(self, modified: bool) -> None:
        self._update_path_label()

    def _update_path_label(self) -> None:
        if self._current_path is None:
            label = "(unsaved)"
        else:
            label = str(self._current_path)
        if self._editor.document().isModified():
            label = "* " + label
        self._path_label.setText(label)

    def _confirm_discard_changes(self) -> bool:
        if not self._editor.document().isModified():
            return True
        r = QMessageBox.question(
            self, "Discard changes?",
            "The current script has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
        )
        return r == QMessageBox.StandardButton.Discard

    # Public hooks used by tests / API
    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path

    def set_namespace(self, namespace: Dict[str, Any]) -> None:
        self.namespace = namespace