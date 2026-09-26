"""UI controller for the in-process MCP server.

The GUI runs the MCP server **inside its own process** using
:func:`protein_studio.mcp_server.serve_http` on a TCP port (default
7820, localhost only). External MCP clients connect to
``http://127.0.0.1:7820/mcp`` and see exactly the same scene the GUI
is rendering, because the server is bound to the GUI's
``StudioAPI`` — not a fresh headless instance.

This replaces the earlier QProcess-spawned-stdio-subprocess design,
which had two unfixable problems:

1. The subprocess's ``HeadlessViewport`` was a different object from
   the GUI's, so load_pdb / set_representation calls never reached
   the actual viewport.
2. stdio is one-to-one with the parent process; external clients
   couldn't attach to the running subprocess.

Design
------
A :class:`QThread` runs an asyncio event loop that drives uvicorn.
The thread is created once and reused — ``start()`` and ``stop()`` are
just task lifecycle on the loop. Cancellation triggers uvicorn's
graceful shutdown, so in-flight requests finish cleanly.

Signal contract (unchanged from previous design):
- ``state_changed(str)`` — "stopped" / "starting" / "running" /
  "stopping" / "failed".
- ``log_line(str)`` — one captured line of stdout / stderr.

The controller is purely process / event-loop plumbing. It does not
talk JSON-RPC to the server — that's the MCP client's job.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7820


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]




class MCPController(QObject):
    """Manage the in-process MCP HTTP server.

    Lifecycle::

        ctl = MCPController(parent)
        ctl.set_api(api)        # call this once the GUI's api is ready
        ctl.start()             # spawns the server thread
        ctl.stop()              # asks uvicorn to shut down
        ctl.show_log()          # open a small window with stdout/stderr
    """

    state_changed = Signal(str)
    log_line = Signal(str)
    server_url_changed = Signal(str)  # emitted when the server URL is known

    _STATE_STOPPED = "stopped"
    _STATE_STARTING = "starting"
    _STATE_RUNNING = "running"
    _STATE_STOPPING = "stopping"
    _STATE_FAILED = "failed"

    def __init__(self, parent: Optional[QObject] = None,
                 host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT,
                 python: Optional[str] = None,
                 cwd: Optional[Path] = None) -> None:
        # python / cwd are accepted for backward-compat with the old
        # QProcess-based controller — they're ignored now since the
        # server runs in-process.
        del python, cwd
        super().__init__(parent)
        self._host = host
        self._port = port
        self._api = None  # set later via set_api()
        self._server: Optional["_QtMcpServer"] = None
        self._state = self._STATE_STOPPED
        self._url = ""
        self._log_lines: list[str] = []
        self._log_cap = 2000
        self._log_dialog = None

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> str:
        return self._state

    @property
    def url(self) -> str:
        return self._url

    def is_running(self) -> bool:
        return self._state == self._STATE_RUNNING

    def set_api(self, api) -> None:
        """Bind the GUI's StudioAPI. Must be called before start()."""
        self._api = api

    def start(self) -> None:
        """Start the MCP HTTP server in the GUI thread.

        ``_QtMcpServer`` uses ``QTcpServer`` to listen for HTTP/1.1
        requests and dispatches JSON-RPC to the bound ``StudioAPI``
        synchronously. Because the server runs on the thread that
        owns the QOpenGLWidget, viewport operations like
        ``plotter.screenshot`` work without any cross-thread
        marshalling.

        Safe to call when already running — it's a no-op then.
        Raises RuntimeError if ``set_api`` hasn't been called yet.
        """
        if self._state in (self._STATE_RUNNING, self._STATE_STARTING):
            return
        if self._api is None:
            raise RuntimeError("set_api() must be called before start()")
        # Lazy import to avoid pulling PySide6 at module import time
        # (lets CLI/headless mode load mcp_controller without Qt).
        from ..mcp_qt_server import _QtMcpServer
        server = _QtMcpServer(self._host, self._port, parent=self)
        server.bind_api(self._api)
        server.state_changed.connect(self._on_runner_state)
        server.log_line.connect(self._append_log)
        server.ready.connect(self._on_runner_ready)
        self._server = server
        self._set_state(self._STATE_STARTING)
        self._append_log(f"# starting GUI-thread server on "
                         f"{self._host}:{self._port}")
        if not server.start():
            self._server = None
            self._set_state(self._STATE_FAILED)

    def stop(self) -> None:
        """Ask the server to stop. No-op if not running."""
        if self._server is None or self._state == self._STATE_STOPPED:
            return
        if self._state == self._STATE_STOPPING:
            return
        self._set_state(self._STATE_STOPPING)
        self._append_log("# stop requested — closing QTcpServer")
        self._server.stop()

    def show_log(self, parent: Optional[QObject] = None) -> None:
        """Open (or focus) a small dialog with the captured log."""
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
        return "\n".join(self._log_lines)

    # ----------------------------------------------------------------- Qt
    def _on_runner_state(self, state: str) -> None:
        self._set_state(state)

    def _on_runner_ready(self, url: str) -> None:
        self._url = url
        self.server_url_changed.emit(url)
        self._append_log(f"# ready at {url}")

    def _set_state(self, new: str) -> None:
        if new != self._state:
            self._state = new
            self.state_changed.emit(new)

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > self._log_cap:
            del self._log_lines[: len(self._log_lines) - self._log_cap]
        self.log_line.emit(line)