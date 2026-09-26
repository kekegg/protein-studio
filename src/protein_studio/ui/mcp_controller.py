"""UI controller for starting/stopping the MCP server subprocess.

Wraps :class:`QProcess` so the MainWindow can spawn the stdio MCP server
that lives at ``protein_studio.mcp_server``. The server is a separate
process (MCP's stdio transport requires it); this controller just gives
the user start / stop / show-log actions in the GUI.

Signal contract
---------------
- ``state_changed(str)`` — emitted whenever the underlying process state
  changes. Payload is one of ``"stopped"``, ``"starting"``, ``"running"``,
  ``"stopping"``, ``"failed"``. UI elements (status-bar pill, menu
  actions, toolbar buttons) listen to this and toggle themselves.
- ``log_line(str)`` — one line of stdout or stderr, no trailing newline.

The controller is purely a process wrapper. It does NOT talk JSON-RPC to
the server — that is the MCP client's job (Claude Desktop, Cline, etc.).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


def _default_python() -> str:
    """Python interpreter used to spawn the MCP server. Defaults to the
    same interpreter that's running the GUI so they share a conda env.
    """
    return sys.executable


def _project_root() -> Path:
    """Path of the protein-studio checkout containing src/ and examples/.

    ``mcp_controller.py`` lives at
    ``<root>/src/protein_studio/ui/mcp_controller.py``, so ``parents[3]``
    is the project root.
    """
    return Path(__file__).resolve().parents[3]


class MCPController(QObject):
    """Manage the MCP server as a QProcess subprocess.

    Lifecycle::

        ctl = MCPController(parent)
        ctl.start()         # spawns the server
        ctl.stop()          # terminates it cleanly
        ctl.show_log()      # open a small window with stdout/stderr
    """

    state_changed = Signal(str)
    log_line = Signal(str)

    _STATE_STOPPED = "stopped"
    _STATE_STARTING = "starting"
    _STATE_RUNNING = "running"
    _STATE_STOPPING = "stopping"
    _STATE_FAILED = "failed"

    def __init__(self, parent: Optional[QObject] = None,
                 python: Optional[str] = None,
                 cwd: Optional[Path] = None) -> None:
        super().__init__(parent)
        self._python: str = python or _default_python()
        self._cwd: str = str(cwd or _project_root())
        self._process: Optional[QProcess] = None
        self._state: str = self._STATE_STOPPED

        # Optional log buffer (capped). Used by show_log().
        self._log_lines: list[str] = []
        self._log_cap = 2000

        # Lazy log dialog reference, set by show_log().
        self._log_dialog = None

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> str:
        return self._state

    def is_running(self) -> bool:
        return self._state == self._STATE_RUNNING

    def start(self) -> None:
        """Spawn the MCP server subprocess.

        Safe to call when already running — it's a no-op then.
        """
        if self._state in (self._STATE_RUNNING, self._STATE_STARTING):
            return
        if self._process is not None:
            self._process.deleteLater()
            self._process = None

        proc = QProcess(self)
        proc.setProgram(self._python)
        proc.setArguments(["-m", "protein_studio.mcp_server"])
        proc.setWorkingDirectory(self._cwd)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH",
                   str(Path(self._cwd) / "src") + os.pathsep +
                   env.value("PYTHONPATH", ""))
        # Stdio mode — don't allocate a controlling TTY.
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        proc.started.connect(self._on_started)
        proc.errorOccurred.connect(self._on_error)
        proc.finished.connect(self._on_finished)
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)

        self._set_state(self._STATE_STARTING)
        self._append_log(f"$ {self._python} -m protein_studio.mcp_server "
                         f"(cwd={self._cwd})")
        proc.start()
        self._process = proc

    def stop(self) -> None:
        """Terminate the server. No-op if not running.

        Sends SIGTERM first (QProcess.terminate), waits up to 3 s, then
        SIGKILL.
        """
        if self._process is None or self._state == self._STATE_STOPPED:
            return
        if self._state == self._STATE_STOPPING:
            return
        self._set_state(self._STATE_STOPPING)
        self._append_log("# stop requested — sending terminate()")
        self._process.terminate()
        if not self._process.waitForFinished(3000):
            self._append_log("# terminate() timed out — killing")
            self._process.kill()
            self._process.waitForFinished(1000)

    def show_log(self, parent: Optional[QObject] = None) -> None:
        """Open (or focus) a small dialog with the captured stdout/stderr."""
        from PySide6.QtWidgets import (QDialog, QPlainTextEdit,
                                       QVBoxLayout, QPushButton)
        if self._log_dialog is not None and self._log_dialog.isVisible():
            self._log_dialog.raise_()
            self._log_dialog.activateWindow()
            return

        dlg = QDialog(parent)
        dlg.setWindowTitle("MCP Server Log")
        dlg.resize(720, 360)
        layout = QVBoxLayout(dlg)
        edit = QPlainTextEdit(dlg)
        edit.setReadOnly(True)
        edit.setStyleSheet(
            "QPlainTextEdit { background: #1d1d1d; color: #dcdcdc; "
            "font-family: 'Menlo', 'Consolas', monospace; font-size: 11px; }"
        )
        edit.setPlainText("\n".join(self._log_lines))

        # Live-tail: every new log line appends to the view.
        def _on_line(line: str) -> None:
            edit.appendPlainText(line)
        self.log_line.connect(_on_line)

        def _on_close(*_):
            try:
                self.log_line.disconnect(_on_line)
            except (TypeError, RuntimeError):
                pass
        dlg.finished.connect(_on_close)
        layout.addWidget(edit)
        close_btn = QPushButton("Close", dlg)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        self._log_dialog = dlg
        dlg.show()

    def log_snapshot(self) -> str:
        """Return the current captured log as one big string."""
        return "\n".join(self._log_lines)

    # ----------------------------------------------------------------- Qt
    def _on_started(self) -> None:
        self._set_state(self._STATE_RUNNING)
        self._append_log("# process started")

    def _on_error(self, err: "QProcess.ProcessError") -> None:
        self._append_log(f"# QProcess error: {err}")
        if err == QProcess.ProcessError.FailedToStart:
            self._set_state(self._STATE_FAILED)

    def _on_finished(self, code: int, status: "QProcess.ExitStatus") -> None:
        self._append_log(f"# process finished, exit code={code}, "
                         f"status={status}")
        self._set_state(self._STATE_STOPPED)

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace")
        for line in data.splitlines():
            self._append_log(line)

    def _on_stderr(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardError()).decode(
            "utf-8", errors="replace")
        for line in data.splitlines():
            self._append_log(f"[stderr] {line}")

    def _set_state(self, new: str) -> None:
        if new != self._state:
            self._state = new
            self.state_changed.emit(new)

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > self._log_cap:
            del self._log_lines[: len(self._log_lines) - self._log_cap]
        self.log_line.emit(line)