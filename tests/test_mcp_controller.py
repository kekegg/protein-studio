"""Tests for the MCPController (UI wrapper around the MCP stdio server).

These tests spawn the actual ``protein_studio.mcp_server`` subprocess
the same way the GUI does, but with ``QT_QPA_PLATFORM=offscreen`` so
they run on CI / headless boxes. They verify:

- ``start()`` flips state ``stopped`` → ``starting`` → ``running``
- ``stop()`` flips it back to ``stopped`` via ``stopping``
- The log buffer accumulates output and exposes it via ``log_snapshot``
- Idempotent: calling ``start()`` twice while running is a no-op
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make sure the test process can import the package regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force a headless Qt backend before importing any Qt module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication     # noqa: E402

from protein_studio.ui.mcp_controller import MCPController  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Don't quit — pytest may share this across modules.


def _wait_for_state(ctl: MCPController, target: str, timeout_ms: int = 8000) -> list[str]:
    """Run the event loop until ``ctl`` reports ``target`` or timeout."""
    seen: list[str] = []
    ctl.state_changed.connect(seen.append)
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    def _check(s: str):
        if s == target:
            loop.quit()
    ctl.state_changed.connect(_check)
    timer.start(timeout_ms)
    loop.exec()
    timer.stop()
    ctl.state_changed.disconnect(_check)
    try:
        ctl.state_changed.disconnect(seen.append)
    except (TypeError, RuntimeError):
        pass
    return seen


def test_starts_and_stops_cleanly(qt_app):
    """start() → running, stop() → stopped."""
    ctl = MCPController(python=sys.executable, cwd=ROOT)
    assert ctl.state == "stopped"
    assert not ctl.is_running()

    # Connect a states buffer BEFORE start() so we capture the
    # synchronous 'starting' emission (state_changed fires from the
    # caller thread right after _set_state).
    states: list[str] = []
    ctl.state_changed.connect(states.append)

    ctl.start()
    _wait_for_state(ctl, "running", timeout_ms=10000)
    assert "starting" in states, f"missing 'starting' in {states}"
    assert ctl.state == "running"
    assert ctl.is_running()

    # Idempotent: start() while running is a no-op.
    before = list(ctl._log_lines)
    ctl.start()
    assert ctl.state == "running"
    assert list(ctl._log_lines) == before, "second start() should not log"

    ctl.stop()
    _wait_for_state(ctl, "stopped", timeout_ms=10000)
    assert "stopping" in states, f"missing 'stopping' in {states}"
    assert not ctl.is_running()
    assert ctl.state == "stopped"


def test_log_captures_command_line_and_lifecycle(qt_app):
    """The log should record the spawn command + at least the start line."""
    ctl = MCPController(python=sys.executable, cwd=ROOT)
    ctl.start()
    _wait_for_state(ctl, "running", timeout_ms=10000)

    snap = ctl.log_snapshot()
    assert "protein_studio.mcp_server" in snap, \
        "log should record the spawn command"
    assert any("started" in line for line in snap.splitlines()), \
        "log should record the started marker"

    ctl.stop()
    _wait_for_state(ctl, "stopped", timeout_ms=10000)


def test_log_is_capped(qt_app):
    """Repeated start/stop cycles shouldn't grow the log unbounded."""
    ctl = MCPController(python=sys.executable, cwd=ROOT)
    cap = ctl._log_cap
    # Force the cap low for the test so we don't need thousands of lines.
    ctl._log_cap = 20
    for _ in range(5):
        ctl.start()
        _wait_for_state(ctl, "running", timeout_ms=10000)
        ctl.stop()
        _wait_for_state(ctl, "stopped", timeout_ms=10000)
    assert len(ctl._log_lines) <= 20, \
        f"log should be capped at 20 lines, got {len(ctl._log_lines)}"
    # And the cap can be restored.
    ctl._log_cap = cap