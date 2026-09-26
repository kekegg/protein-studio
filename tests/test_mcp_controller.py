"""Tests for the in-process MCPController.

The controller runs the MCP server inside a QThread using asyncio +
uvicorn. Tests verify:

- start() requires set_api() first
- start() flips state stopped → starting → running (with URL)
- stop() flips it back to stopped
- Idempotent: calling start() twice while running is a no-op
- The server actually accepts HTTP traffic on the bound port
  (full round-trip including shared-state visibility)
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Force a headless Qt backend before importing any Qt module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication     # noqa: E402

from protein_studio.api import HeadlessViewport, StudioAPI  # noqa: E402
from protein_studio.ui.mcp_controller import MCPController  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_url(ctl: MCPController, timeout_ms: int = 10000) -> None:
    """Run the event loop until ``ctl.url`` is non-empty (or timeout)."""
    if ctl.url:
        return
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)

    def _check(url: str):
        if url:
            loop.quit()
    ctl.server_url_changed.connect(_check)
    timer.start(timeout_ms)
    loop.exec()
    timer.stop()
    try:
        ctl.server_url_changed.disconnect(_check)
    except (TypeError, RuntimeError):
        pass


def _wait_for_state(ctl: MCPController, target: str,
                    timeout_ms: int = 15000) -> list[str]:
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
    try:
        ctl.state_changed.disconnect(_check)
    except (TypeError, RuntimeError):
        pass
    try:
        ctl.state_changed.disconnect(seen.append)
    except (TypeError, RuntimeError):
        pass
    return seen


def _http_post(url: str, payload: dict,
               session_id: list[str], timeout: float = 10.0) -> dict:
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id[0]:
        headers["mcp-session-id"] = session_id[0]
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        sid = resp.headers.get("mcp-session-id")
        if sid:
            session_id[0] = sid
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read().decode()
        if "text/event-stream" in ctype:
            for chunk in raw.split("\n\n"):
                for line in chunk.splitlines():
                    if line.startswith("data:"):
                        return json.loads(line[len("data:"):].strip())
        if not raw:
            return {}
        return json.loads(raw)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_requires_api(qt_app):
    ctl = MCPController(port=_free_port())
    with pytest.raises(RuntimeError, match="set_api"):
        ctl.start()


def test_starts_and_stops_cleanly(qt_app):
    api = StudioAPI(viewport=HeadlessViewport())
    ctl = MCPController(port=_free_port())
    ctl.set_api(api)
    assert ctl.state == "stopped"
    assert not ctl.is_running()

    # Capture states from BEFORE start() so the synchronous 'starting'
    # emission isn't lost.
    states: list[str] = []
    ctl.state_changed.connect(states.append)

    ctl.start()
    _wait_for_state(ctl, "running", timeout_ms=15000)
    assert "starting" in states, f"missing 'starting' in {states}"
    assert ctl.state == "running"
    assert ctl.is_running()
    assert ctl.url.startswith("http://127.0.0.1:")

    # Idempotent: start() while running is a no-op.
    before = list(ctl._log_lines)
    ctl.start()
    assert ctl.state == "running"
    assert list(ctl._log_lines) == before

    ctl.stop()
    _wait_for_state(ctl, "stopped", timeout_ms=15000)
    assert "stopping" in states, f"missing 'stopping' in {states}"
    assert not ctl.is_running()


# ---------------------------------------------------------------------------
# End-to-end: external client hits the GUI's server, sees shared state
# ---------------------------------------------------------------------------

def test_external_client_sees_shared_state(qt_app, tmp_path: Path):
    """Connect to the controller's HTTP server with urllib, prove that
    load_pdb from the client is visible to the GUI's api."""
    api = StudioAPI(viewport=HeadlessViewport())
    ctl = MCPController(port=_free_port())
    ctl.set_api(api)
    ctl.start()
    _wait_for_state(ctl, "running", timeout_ms=15000)
    # server_url_changed is emitted *after* state -> running, so wait
    # for it too — otherwise ctl.url can still be empty when we use it.
    _wait_for_url(ctl, timeout_ms=10000)
    try:
        assert ctl.url, f"empty URL after running (log: {ctl.log_snapshot()})"
        url = ctl.url
        session = [""]

        # initialize
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05",
                           "capabilities": {},
                           "clientInfo": {"name": "test", "version": "0"}}}
        r = _http_post(url, init, session)
        assert "result" in r, r
        # notifications/initialized
        _http_post(url, {"jsonrpc": "2.0",
                         "method": "notifications/initialized"}, session)

        # load_pdb
        pdb = ROOT / "examples" / "1aki.pdb"
        load = _http_post(url, {"jsonrpc": "2.0", "id": 2,
                                "method": "tools/call",
                                "params": {"name": "load_pdb",
                                           "arguments": {"path":
                                                         str(pdb)}}},
                          session)
        text = load["result"]["content"][0]["text"]
        load_obj = json.loads(text)
        assert load_obj["n_atoms"] > 0, load_obj

        # SHARED STATE — the GUI's api sees the object the client loaded
        assert api.scene.n_objects() == 1, \
            f"shared state broken: {api.scene.n_objects()} objects"

        # list_objects via HTTP returns the same scene
        lst = _http_post(url, {"jsonrpc": "2.0", "id": 3,
                               "method": "tools/call",
                               "params": {"name": "list_objects",
                                          "arguments": {}}}, session)
        lst_obj = json.loads(lst["result"]["content"][0]["text"])
        assert len(lst_obj) == 1
        assert lst_obj[0]["name"] == "1aki"
    finally:
        ctl.stop()
        _wait_for_state(ctl, "stopped", timeout_ms=15000)


def test_log_snapshot_records_start(qt_app):
    """The log buffer should at least record the spawn marker."""
    api = StudioAPI(viewport=HeadlessViewport())
    ctl = MCPController(port=_free_port())
    ctl.set_api(api)
    ctl.start()
    _wait_for_state(ctl, "running", timeout_ms=15000)
    try:
        snap = ctl.log_snapshot()
        assert "starting MCP HTTP server" in snap, snap
        assert str(ctl._port) in snap
    finally:
        ctl.stop()
        _wait_for_state(ctl, "stopped", timeout_ms=15000)