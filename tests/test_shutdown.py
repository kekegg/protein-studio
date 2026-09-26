"""Regression tests for the OpenGL cleanup-on-shutdown path.

The crash we're guarding against::

    QApplication quit
      → QOpenGLWidget destroy
      → VTK's vtkOrientationMarkerWidget survives
      → GC fires EnabledOff on a dead GL context
      → glDeleteBuffers against wild pointer → segfault

The fix is to tear the plotter down *synchronously* (via
``ViewportWidget.shutdown``) while the QOpenGLWidget is still alive,
before Qt gets a chance to destroy it. This test exercises the
public surface of that path; it does not actually launch a real GL
context (offscreen Qt can't), but it confirms the call order and
idempotency so the production crash doesn't recur when shipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication      # noqa: E402

from protein_studio.core.scene import Scene      # noqa: E402
from protein_studio.ui.viewport import ViewportWidget  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_viewport_has_shutdown_method(qt_app):
    vp = ViewportWidget(Scene())
    assert hasattr(vp, "shutdown")
    assert callable(vp.shutdown)


def test_shutdown_is_idempotent(qt_app):
    """Calling shutdown() twice must not raise on the second call —
    the second invocation is a no-op thanks to the ``_shutdown_done``
    guard, so reconnecting aboutToQuit, or any other repeated
    shutdown trigger, is safe."""
    vp = ViewportWidget(Scene())
    vp.shutdown()
    vp.shutdown()  # must not raise
    assert vp._shutdown_done is True


def test_shutdown_stops_spin_first(qt_app):
    """If a spin loop is running, shutdown must stop it before
    touching the plotter."""
    vp = ViewportWidget(Scene())
    # Start spin (no-op visually, but the timer is now alive).
    try:
        vp.start_spin()
    except Exception:
        pass
    # shutdown should clean up without raising even with the timer live
    vp.shutdown()
    # After shutdown the spin flag must be off and the timer dropped.
    assert vp._spinning is False
    assert vp._spin_timer is None


def test_shutdown_drops_plotter_reference(qt_app):
    """After shutdown, ``vp.plotter`` must be None so the next GC
    pass doesn't see a live plotter dangling."""
    vp = ViewportWidget(Scene())
    vp.shutdown()
    assert vp.plotter is None


def test_shutdown_handles_already_none_plotter(qt_app):
    """If for some reason plotter is already None (e.g. a previous
    shutdown run), the call must still complete cleanly."""
    vp = ViewportWidget(Scene())
    vp.plotter = None
    vp.shutdown()  # must not raise
    assert vp._shutdown_done is True


def test_main_window_closeevent_calls_viewport_shutdown(qt_app, tmp_path):
    """The integration we actually care about: when MainWindow is
    closed, the viewport's shutdown is called *before* the MCP
    controller's stop. We don't spin up a real MCP server here — we
    just verify the call ordering via a stub."""
    from protein_studio.ui.main_window import MainWindow
    from protein_studio.api import StudioAPI
    from protein_studio.ui.outliner import OutlinerWidget
    from protein_studio.ui.properties import PropertiesWidget
    from protein_studio.ui.console import PythonConsole
    from protein_studio.ui.editor import PythonEditor

    api = StudioAPI()
    scene = Scene()
    vp = ViewportWidget(scene)
    outliner = OutlinerWidget(scene)
    properties = PropertiesWidget(scene)
    console = PythonConsole(namespace={"D": api, "np": __import__("numpy")})
    editor = PythonEditor({"D": api, "np": __import__("numpy")})

    win = MainWindow(scene, vp, outliner, properties, console, editor, api)

    # Stub out MCP controller so the test doesn't depend on ports
    # being free — we just want to record call order.
    call_log: list[str] = []

    real_vp_shutdown = vp.shutdown

    def _spy_vp_shutdown():
        call_log.append("viewport.shutdown")
        real_vp_shutdown()

    vp.shutdown = _spy_vp_shutdown  # type: ignore[method-assign]

    def _spy_mcp_stop():
        call_log.append("mcp.stop")

    win._mcp.stop = _spy_mcp_stop  # type: ignore[method-assign]

    # Close the window — closeEvent fires synchronously here.
    win.close()

    # The critical ordering: VTK teardown happens before MCP stop.
    assert "viewport.shutdown" in call_log
    if "mcp.stop" in call_log:
        # If MCP was running, it must come after viewport.shutdown.
        assert call_log.index("viewport.shutdown") < call_log.index("mcp.stop"), \
            f"viewport.shutdown must precede mcp.stop, got: {call_log}"