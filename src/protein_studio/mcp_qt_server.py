"""GUI-thread MCP server using pure Qt sockets.

Replaces the uvicorn + background-thread + asyncio loop stack with a
synchronous HTTP/1.1 server driven by ``QTcpServer`` in the GUI
thread. This means every MCP tool call is executed on the thread
that owns the QOpenGLWidget's GL context, so ``plotter.screenshot``,
``plotter.camera.Azimuth`` and friends work without any cross-thread
marshalling.

Protocol: a subset of MCP 2.x's streamable-HTTP transport. We
support:
  * ``POST /mcp`` with JSON-RPC body, response as either
    ``Content-Type: application/json`` (single response) or
    ``Content-Type: text/event-stream`` (SSE stream — supported
    but not currently used by tool calls since they're sync).
  * Session headers (``Mcp-Session-Id``) are accepted and echoed.
  * ``GET /mcp`` is not supported (MCP server doesn't push).

This is enough for the official ``mcp.client.streamable_http`` Python
SDK and the standard MCP Inspector.

Why a hand-rolled server instead of uvicorn on a background thread?
  * uvicorn needs its own asyncio loop; running it on a worker
    thread means viewport ops (which need the GUI thread's GL
    context) need cross-thread marshalling, which is fragile
    (previous attempts deadlock).
  * All ``_dispatch`` work is already sync; we just need a sync
    HTTP framing layer.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QTcpServer, QTcpSocket, QHostAddress

log = logging.getLogger("mcp_qt_server")

# --- Protocol constants ---
MCP_PATH = "/mcp"
JSON_CT = b"application/json"
SSE_CT = b"text/event-stream"
MAX_BODY = 16 * 1024 * 1024  # 16 MiB — generous


class _ClientHandler(QObject):
    """One per accepted TCP connection. Reads request, writes response."""

    def __init__(self, sock: QTcpSocket, parent: QObject, server: "_QtMcpServer"):
        super().__init__(parent)
        self._sock = sock
        self._server = server
        self._buf = bytearray()
        self._peer = sock.peerAddress().toString()
        sock.readyRead.connect(self._on_ready_read)
        sock.disconnected.connect(self._on_disconnected)
        # 10s idle timeout
        self._idle_timer = None  # set in on_ready_read
        self._install_idle_timer()

    def _install_idle_timer(self):
        from PySide6.QtCore import QTimer
        t = QTimer(self)
        t.setSingleShot(True)
        t.setInterval(10_000)
        t.timeout.connect(self._on_timeout)
        t.start()
        self._idle_timer = t

    def _on_timeout(self):
        log.warning("client %s idle timeout — closing", self._peer)
        self._sock.disconnectFromHost()

    def _on_disconnected(self):
        self.deleteLater()

    def _on_ready_read(self):
        # Reset idle timer on every read.
        if self._idle_timer is not None:
            self._idle_timer.start()
        chunk = self._sock.readAll()
        if not chunk:
            return
        self._buf.extend(chunk.data())
        # Try to parse one or more complete requests from the buffer.
        while True:
            req = self._try_parse_one()
            if req is None:
                break
            self._handle_request(req)

    def _try_parse_one(self) -> Optional[Dict[str, Any]]:
        """Look for a complete HTTP/1.1 request in self._buf. If found,
        strip it from the buffer and return the parsed components dict
        with keys: method, path, headers, body. Returns None if more
        data is needed.
        """
        sep = self._buf.find(b"\r\n\r\n")
        if sep < 0:
            return None
        head = bytes(self._buf[:sep]).decode("iso-8859-1")
        header_lines = head.split("\r\n")
        request_line = header_lines[0]
        try:
            method, target, _ = request_line.split(" ", 2)
        except ValueError:
            self._send_error(400, "bad request line")
            self._buf.clear()
            return None
        headers: Dict[str, str] = {}
        for h in header_lines[1:]:
            if ":" in h:
                k, v = h.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        clen = int(headers.get("content-length", "0"))
        total = sep + 4 + clen
        if len(self._buf) < total:
            return None  # need more body
        body = bytes(self._buf[sep + 4: total])
        # Consume the request from the buffer.
        del self._buf[:total]
        return {"method": method.upper(),
                "path": target.split("?", 1)[0],
                "headers": headers,
                "body": body}

    def _handle_request(self, req: Dict[str, Any]):
        method = req["method"]
        path = req["path"]
        body = req["body"]
        if method == "POST" and path == MCP_PATH:
            self._handle_mcp(body, req["headers"])
        elif method == "GET" and path == MCP_PATH:
            # The MCP streamable_http client opens a GET stream after
            # the initialized handshake so the server can push messages.
            # We have no server-initiated messages, so we just hold the
            # connection open with periodic SSE comment lines acting as
            # keepalives. The connection closes when the client
            # disconnects (which we detect via QTcpSocket.disconnected).
            self._handle_sse_get()
        elif method == "GET" and path == "/health":
            self._send_json(200, {"ok": True, "server": "protein-studio"})
        elif method == "DELETE" and path == MCP_PATH:
            # Terminate the session (per spec, we have no real session).
            self._send_json(200, {})
        elif method == "POST" and path == "/shutdown":
            # Convenience: lets tests cleanly stop the server.
            self._send_json(200, {"stopping": True})
            self._sock.flush()
            self._sock.disconnectFromHost()
            self._server.stop()
        else:
            self._send_error(404, f"no route for {method} {path}")

    def _handle_mcp(self, body: bytes, headers: Dict[str, str]):
        # Parse JSON-RPC.
        try:
            msg = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": f"parse error: {e}"},
            })
            return
        # Client may also send a batch; we accept and reply with a list.
        is_batch = isinstance(msg, list)
        requests = msg if is_batch else [msg]
        responses: List[Dict[str, Any]] = []
        for req in requests:
            resp = self._handle_one_jsonrpc(req)
            # Notifications get no response — filter them out so the
            # result list doesn't contain None entries that the client
            # then tries to parse as a JSON-RPC message.
            if resp is None:
                continue
            responses.append(resp)
        if not responses:
            # All-notification batch (or single notification) — per
            # JSON-RPC the server should send 204 No Content. Many
            # clients (including the MCP Python SDK) accept empty body
            # or 204, but not "null" which we used to send.
            self._send_no_content()
            return
        if is_batch:
            payload = responses
        else:
            payload = responses[0]
        self._send_json(200, payload)

    def _send_no_content(self):
        head = (
            "HTTP/1.1 204 No Content\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        ).encode("ascii")
        self._sock.write(head)
        self._sock.flush()
        self._buf.clear()

    def _handle_sse_get(self):
        """Hold an SSE stream open for server-initiated messages.

        The MCP client opens this after the 'initialized' notification
        to receive pushes from the server. We have no such messages
        right now, so we send a single SSE comment as a keepalive and
        then park on the socket until the client disconnects. The
        disconnected signal tears down this ClientHandler.
        """
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
        ).encode("ascii")
        self._sock.write(head)
        # Initial keepalive comment.
        self._sock.write(b": stream-open\n\n")
        self._sock.flush()
        # Install a periodic keepalive every 15s (SSE comments).
        from PySide6.QtCore import QTimer
        self._ka_timer = QTimer(self)
        self._ka_timer.setInterval(15_000)
        self._ka_timer.timeout.connect(self._send_keepalive)
        self._ka_timer.start()
        # When the client disconnects, the disconnected signal will
        # clean us up; nothing to do here.

    def _send_keepalive(self):
        if self._sock.state() == QTcpSocket.SocketState.UnconnectedState:
            return
        try:
            self._sock.write(b": keepalive\n\n")
            self._sock.flush()
        except Exception:
            pass

    def _handle_one_jsonrpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        rpc_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        # Notifications (no id) get no response.
        if rpc_id is None and "id" not in req:
            self._server._route_rpc(method, params)
            return None  # filtered out by caller
        if method == "initialize":
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"experimental": {},
                                     "tools": {"listChanged": False}},
                    "serverInfo": {"name": "protein-studio", "version": ""},
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": rpc_id, "result": {}}
        if method == "tools/list":
            tools = self._server._list_tools()
            return {"jsonrpc": "2.0", "id": rpc_id,
                    "result": {"tools": tools}}
        if method == "tools/call":
            return self._server._call_tool_jsonrpc(rpc_id, params)
        # notifications/initialized, etc. — accept and ignore
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {}}

    def _send_json(self, status: int, body: Any):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found",
                  500: "Internal Server Error"}.get(status, "OK")
        # Keep the socket open after the response so the client can
        # pipeline further JSON-RPC requests on the same connection
        # (the MCP streamable_http Python client does this).
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Cache-Control: no-cache\r\n"
            f"\r\n"
        ).encode("ascii")
        self._sock.write(head + payload)
        self._sock.flush()
        # Reset the buffer in case the client pipelines another
        # request on the same socket.
        self._buf.clear()

    def _send_error(self, status: int, message: str):
        self._send_json(status, {"error": message, "status": status})


class _QtMcpServer(QObject):
    """QTcpServer-based MCP server that runs in the GUI thread.

    All MCP tool calls execute synchronously on the thread that owns
    this QObject — which is the GUI thread, since MCPController
    constructs us as a child on itself (which is parented to the
    MainWindow). That means ``plotter.screenshot`` etc. have a valid
    GL context and work without any cross-thread marshalling.
    """

    # Emitted on the thread that calls start(). In our case that's
    # the GUI thread (start() is called from a slot).
    log_line = Signal(str)
    state_changed = Signal(str)
    ready = Signal(str)  # URL

    def __init__(self, host: str, port: int, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._server: Optional[QTcpServer] = None
        self._api = None  # set by bind_api()
        self._url = ""

    # --- API ---
    def bind_api(self, api):
        self._api = api

    def start(self) -> bool:
        if self._server is not None:
            return True
        s = QTcpServer(self)
        s.newConnection.connect(self._on_new_connection)
        if not s.listen(QHostAddress(self._host), self._port):
            self.state_changed.emit("failed")
            self.log_line.emit(f"# failed to bind {self._host}:{self._port}: {s.errorString()}")
            return False
        self._server = s
        self._url = f"http://{self._host}:{self._port}{MCP_PATH}"
        self.log_line.emit(f"# listening on {self._url}")
        self.state_changed.emit("running")
        self.ready.emit(self._url)
        return True

    def stop(self):
        if self._server is None:
            return
        # Closing the QTcpServer stops accepting new connections;
        # existing ClientHandler children will be torn down when their
        # sockets close (Connection: close after every response).
        self._server.close()
        self._server = None
        self.log_line.emit("# server stopped")
        self.state_changed.emit("stopped")

    @property
    def url(self) -> str:
        return self._url

    # --- Internals ---
    def _on_new_connection(self):
        s = self._server
        while s is not None and s.hasPendingConnections():
            sock = s.nextPendingConnection()
            _ClientHandler(sock, self, self)

    def _route_rpc(self, method: Optional[str], params: Dict[str, Any]):
        """Handle a notification-style RPC (no id, no response expected).

        We have no server-initiated messages and no notification
        handlers to invoke — accept and discard.
        """
        # Future: dispatch to on_progress / on_cancelled / etc. handlers
        # here. For now this is a no-op.
        pass

    # --- RPC dispatch (sync, runs on whatever thread called us) ---
    def _list_tools(self) -> List[Dict[str, Any]]:
        from .mcp_server import TOOL_CATALOG
        return [
            {"name": t["name"],
             "description": t["description"],
             "inputSchema": t["inputSchema"]}
            for t in TOOL_CATALOG
        ]

    def _call_tool_jsonrpc(self, rpc_id, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        from .mcp_server import _dispatch
        try:
            result = _dispatch(name, dict(arguments), self._api)
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {
                    "content": [{"type": "text",
                                 "text": _safe_json(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {
                    "content": [{"type": "text",
                                 "text": _safe_json({
                                     "error": str(e),
                                     "type": type(e).__name__,
                                     "traceback": traceback.format_exc(),
                                 })}],
                    "isError": True,
                },
            }


def _safe_json(obj: Any) -> str:
    def _default(o):
        try: return o.tolist()
        except AttributeError: return repr(o)
    return json.dumps(obj, default=_default, ensure_ascii=False)
