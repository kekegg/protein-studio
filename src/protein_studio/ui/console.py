"""Embedded Python console widget.

A simple read-eval-print loop over Python, persisted into a history file,
with a couple of quality-of-life features:

- Up/down arrow keys browse history.
- ``Tab`` triggers a lightweight autocomplete for names exposed by the
  :class:`StudioAPI` (``D``, ``D.scene``, ``D.objects``, ``D.schemes``,
  ``D.representations``).
- Output streams are colored (stdout = light gray, stderr = red,
  exceptions are formatted with a traceback).
- Multi-statement / multi-line paste is handled gracefully — the user
  can paste a 4-line snippet and the console will run it as one block.

The console is *not* a full IPython replacement — it is a plain Python
interpreter so user scripts can do anything: load proteins, mutate
scene objects, run arbitrary matplotlib figures, etc.
"""

from __future__ import annotations

import code as _code
import io
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import (
    QFont, QFontDatabase, QKeyEvent, QTextCursor, QColor, QSyntaxHighlighter,
    QTextCharFormat, QTextDocument,
)
from PySide6.QtWidgets import (
    QPlainTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
)


# ANSI -> QColor conversion for stdout text styling
ANSI_COLORS = {
    "30": QColor("#262626"), "31": QColor("#e96b6b"), "32": QColor("#7fd47f"),
    "33": QColor("#e8c66b"), "34": QColor("#7fb1ff"), "35": QColor("#d28bff"),
    "36": QColor("#7fd1d1"), "37": QColor("#dcdcdc"),
    "90": QColor("#7f7f7f"), "91": QColor("#ff8a8a"), "92": QColor("#92e092"),
    "93": QColor("#ffd87f"), "94": QColor("#8fb7ff"), "95": QColor("#e69bff"),
    "96": QColor("#92dada"), "97": QColor("#ffffff"),
}


class PythonHighlighter(QSyntaxHighlighter):
    """Tiny Python syntax highlighter — keywords, strings, comments, numbers."""
    KEYWORDS = (
        "False None True and as assert async await break class continue def del "
        "elif else except finally for from global if import in is lambda "
        "nonlocal not or pass raise return try while with yield"
    ).split()

    def __init__(self, doc: QTextDocument) -> None:
        super().__init__(doc)
        self._kw_format = QTextCharFormat()
        self._kw_format.setForeground(QColor("#cf8ed4"))
        self._kw_format.setFontWeight(QFont.Bold)
        self._str_format = QTextCharFormat()
        self._str_format.setForeground(QColor("#a3dd94"))
        self._num_format = QTextCharFormat()
        self._num_format.setForeground(QColor("#e8c66b"))
        self._cmt_format = QTextCharFormat()
        self._cmt_format.setForeground(QColor("#7f7f7f"))
        self._fn_format = QTextCharFormat()
        self._fn_format.setForeground(QColor("#7fb1ff"))
        self._self_format = QTextCharFormat()
        self._self_format.setForeground(QColor("#ffa657"))

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt API)
        import re
        for kw in self.KEYWORDS:
            for m in re.finditer(rf"\b{kw}\b", text):
                self.setFormat(m.start(), m.end() - m.start(), self._kw_format)
        for m in re.finditer(r"\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"[^\n]*?\"|'[^\n]*?'", text):
            self.setFormat(m.start(), m.end() - m.start(), self._str_format)
        for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
            self.setFormat(m.start(), m.end() - m.start(), self._num_format)
        for m in re.finditer(r"#[^\n]*", text):
            self.setFormat(m.start(), m.end() - m.start(), self._cmt_format)
        for m in re.finditer(r"\bdef\s+(\w+)", text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), self._fn_format)
        for m in re.finditer(r"\bself\b", text):
            self.setFormat(m.start(), m.end() - m.start(), self._self_format)


class _ConsoleSignals(QObject):
    output = Signal(str, str)  # text, kind ("out" | "err")


class _StreamTap(io.TextIOBase):
    """Captures print() output and pipes it into the console widget."""

    def __init__(self, signals: _ConsoleSignals, kind: str) -> None:
        super().__init__()
        self.signals = signals
        self.kind = kind

    def write(self, s: str) -> int:
        if s:
            self.signals.output.emit(s, self.kind)
        return len(s)

    def flush(self) -> None:  # noqa: D401
        return None


class PythonConsole(QWidget):
    """A Blender-style Python scripting console.

    Parameters
    ----------
    namespace : dict
        Initial namespace — usually ``{"D": api}``.
    history_path : Path | None
        File used to persist command history (default ``~/.protein-studio-history``).
    """

    PROMPT = ">>> "

    def __init__(self, namespace: Optional[Dict[str, Any]] = None,
                 history_path: Optional[Path] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.namespace: Dict[str, Any] = dict(namespace or {})
        self.history_path = history_path or Path.home() / ".protein-studio-history"
        self.history: list[str] = []
        self._history_index = 0
        self._buffer: list[str] = []
        self._interp = _code.InteractiveConsole(self.namespace)

        self._signals = _ConsoleSignals()
        self._signals.output.connect(self._append_text)

        # ------------------------------ layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)
        self._mode_label = QLabel("Python")
        self._mode_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._mode_label)
        toolbar.addStretch()
        clear_btn = QLabel("<a href='#' style='color:#7fb1ff'>Clear</a>")
        clear_btn.linkActivated.connect(self._clear_console)
        toolbar.addWidget(clear_btn)
        layout.addLayout(toolbar)

        # Console text area
        self._editor = QPlainTextEdit()
        self._editor.setReadOnly(False)
        self._editor.setUndoRedoEnabled(False)
        self._editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        font = self._editor.font()
        font.setPointSize(max(10, font.pointSize()))
        self._editor.setFont(font)
        self._editor.setStyleSheet(
            "QPlainTextEdit { background: #1d1d1d; color: #dcdcdc; "
            "border: none; selection-background-color: #3b76c2; }"
        )
        self._highlighter = PythonHighlighter(self._editor.document())
        layout.addWidget(self._editor, 1)

        self._cursor_pos = 0
        self._editor.textChanged.connect(self._on_text_changed)
        self._editor.installEventFilter(self)

        # Initial banner
        self._append_text("Protein Studio Python console — type 'help(D)' for tips.\n", "out")
        self._write_prompt()

        # Restore history
        self._load_history()

    # ----------------------------------------------------------- history
    def _load_history(self) -> None:
        if self.history_path.exists():
            try:
                self.history = [l.rstrip("\n") for l in self.history_path.read_text().splitlines()]
            except OSError:
                self.history = []
        self._history_index = len(self.history)

    def _save_history(self) -> None:
        try:
            self.history_path.write_text("\n".join(self.history[-1000:]))
        except OSError:
            pass

    # ----------------------------------------------------------- prompt
    def _write_prompt(self) -> None:
        self._append_raw(self.PROMPT)
        self._cursor_pos = self._editor.textCursor().position()

    def _append_raw(self, text: str) -> None:
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._editor.setTextCursor(cursor)

    def _append_text(self, text: str, kind: str = "out") -> None:
        # Strip ANSI escapes (minimal handling)
        import re
        clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
        if kind == "err":
            color = QColor("#ff8a8a")
        else:
            color = QColor("#dcdcdc")
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = cursor.charFormat()
        fmt.setForeground(color)
        cursor.setCharFormat(fmt)
        cursor.insertText(clean)
        self._editor.setTextCursor(cursor)

    def _clear_console(self) -> None:
        self._editor.clear()
        self._write_prompt()

    def _on_text_changed(self) -> None:
        # Keep cursor past the prompt
        cursor = self._editor.textCursor()
        if cursor.position() < self._cursor_pos:
            cursor.setPosition(self._cursor_pos)
            self._editor.setTextCursor(cursor)

    # ----------------------------------------------------------- key handling
    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self._editor and isinstance(event, QKeyEvent):
            key = event.key()
            cursor = self._editor.textCursor()
            cursor.setPosition(self._editor.textCursor().position())

            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._on_return()
                return True
            if key == Qt.Key.Key_Up:
                self._history_prev()
                return True
            if key == Qt.Key.Key_Down:
                self._history_next()
                return True
            if key == Qt.Key.Key_Tab:
                self._complete()
                return True
            if key == Qt.Key.Key_Home:
                cursor = self._editor.textCursor()
                cursor.setPosition(self._cursor_pos)
                self._editor.setTextCursor(cursor)
                return True
            if key == Qt.Key.Key_Backspace:
                if self._editor.textCursor().position() <= self._cursor_pos:
                    return True
        return super().eventFilter(obj, event)

    def _current_line(self) -> str:
        text = self._editor.toPlainText()
        # Take everything after the last prompt on the current "line block"
        idx = text.rfind(self.PROMPT)
        if idx < 0:
            return ""
        return text[idx + len(self.PROMPT):]

    def _replace_current_line(self, new_text: str) -> None:
        cursor = self._editor.textCursor()
        cursor.setPosition(self._cursor_pos)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(new_text)

    def _history_prev(self) -> None:
        if not self.history:
            return
        if self._history_index > 0:
            self._history_index -= 1
        self._replace_current_line(self.history[self._history_index])

    def _history_next(self) -> None:
        if not self.history:
            return
        if self._history_index < len(self.history) - 1:
            self._history_index += 1
            self._replace_current_line(self.history[self._history_index])
        else:
            self._history_index = len(self.history)
            self._replace_current_line("")

    def _complete(self) -> None:
        line = self._current_line()
        if not line.strip():
            return
        # Only complete a single trailing identifier
        import re
        m = re.search(r"([A-Za-z_][A-Za-z0-9_.]*)$", line)
        if not m:
            return
        prefix = m.group(1)
        # Build candidate list
        candidates: list[str] = []
        # Local namespace
        for name in self.namespace:
            if name.startswith(prefix):
                candidates.append(name)
        # dunder attrs
        if "." in prefix:
            head, tail = prefix.rsplit(".", 1)
            try:
                obj = eval(head, self.namespace)
                for n in dir(obj):
                    if n.startswith(tail):
                        candidates.append(prefix[: -len(tail)] + n)
            except Exception:
                pass
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            self._replace_current_line(line[: m.start()] + candidates[0])
        elif len(candidates) > 1:
            self._append_text("\n" + "  ".join(candidates) + "\n", "out")
            self._write_prompt()
            self._append_raw(line)
            self._editor.moveCursor(QTextCursor.MoveOperation.End)

    def _on_return(self) -> None:
        line = self._current_line().rstrip()
        self._append_raw("\n")
        if line.strip():
            self.history.append(line)
            self._save_history()
            self._history_index = len(self.history)
            # If the line contains multiple statements (e.g. a paste of
            # several statements joined by newlines), runsource() will
            # compile + exec them in one go — push() cannot.
            out = _StreamTap(self._signals, "out")
            err = _StreamTap(self._signals, "err")
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    more = self._interp.runsource(line, filename="<console>", symbol="exec")
            except SystemExit:
                raise
            except Exception:
                tb = traceback.format_exc()
                self._append_text(tb, "err")
                more = False
            # If the source needs more input (e.g. unterminated `if` block)
            # show a `...` prompt and keep the buffer.
            if more:
                self._append_raw("... ")
                self._cursor_pos = self._editor.textCursor().position()
                return
        self._write_prompt()

    # ----------------------------------------------------------- public API
    def write_namespace(self, key: str, value: Any) -> None:
        """Inject a binding into the running interpreter namespace."""
        self.namespace[key] = value
        self._interp.locals[key] = value

    def exec_line(self, line: str) -> None:
        """Append an extra newline, then execute *line* as if typed.

        Handles multi-statement input via :meth:`runsource`.
        """
        self._append_raw("\n")
        out = _StreamTap(self._signals, "out")
        err = _StreamTap(self._signals, "err")
        try:
            with redirect_stdout(out), redirect_stderr(err):
                self._interp.runsource(line, filename="<console>", symbol="exec")
        except Exception:
            tb = traceback.format_exc()
            self._append_text(tb, "err")
        self._write_prompt()