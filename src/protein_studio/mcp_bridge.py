"""stdio ↔ TCP bridge for the LingSu Protein MCP server.

Most MCP clients (Claude Desktop, Cline, Continue.dev, …) only know how
to launch a subprocess and pipe JSON-RPC over its stdin / stdout.
They have no way to talk to a server that's already listening on a
TCP port.

This bridge is the missing piece: it is itself a tiny MCP stdio server
whose only job is to forward each line of JSON-RPC from its stdin to
the GUI's HTTP endpoint, and write each response back to its stdout.

Usage in a client config::

    {
      "mcpServers": {
        "protein-studio": {
          "command": "python",
          "args": ["-m", "protein_studio.mcp_bridge"],
          "env": {
            "PROTEIN_STUDIO_MCP_URL": "http://127.0.0.1:7820/mcp"
          }
        }
      }
    }

The bridge uses ``MCP_PROTEIN_STUDIO_URL`` if set, otherwise defaults
to ``http://127.0.0.1:7820/mcp``. The connection is plain HTTP/1.1
POST with a JSON-RPC body — no SSE, no chunking — so a 200-line script
is all we need.

Protocol note: the streamable-http transport actually returns SSE by
default. We ask for plain JSON by sending the ``Accept: application/json,
text/event-stream`` header and parsing the response — uvicorn honours
``json_response=True`` on the server side, but that flag isn't set
here, so we read the first SSE ``data:`` payload and treat it as the
JSON-RPC response. For simple request/response RPC (the only thing an
agent does on the first turn), that works.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_URL = os.environ.get("MCP_PROTEIN_STUDIO_URL",
                             "http://127.0.0.1:7820/mcp")


# MCP streamable-http is stateful — the server returns an
# ``mcp-session-id`` header on ``initialize`` and we have to echo it
# on every subsequent request. Module-level so all _post_json calls
# share it.
_SESSION_ID: Optional[str] = None


def _post_json(url: str, payload: Dict[str, Any],
               timeout: float = 30.0) -> Dict[str, Any]:
    """POST a JSON-RPC payload, return the response dict.

    Handles plain-JSON / SSE-wrapped responses and tracks the
    mcp-session-id across calls.
    """
    global _SESSION_ID
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _SESSION_ID is not None:
        headers["mcp-session-id"] = _SESSION_ID
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        sid = resp.headers.get("mcp-session-id")
        if sid:
            _SESSION_ID = sid
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read().decode("utf-8")
        if "text/event-stream" in ctype:
            for chunk in raw.split("\n\n"):
                for line in chunk.splitlines():
                    if line.startswith("data:"):
                        return json.loads(line[len("data:"):].strip())
            raise RuntimeError(f"SSE response without data: {raw[:200]!r}")
        if not raw:
            return {}
        return json.loads(raw)


def _write_message(msg: Dict[str, Any]) -> None:
    """Write a JSON-RPC message as one line of stdout, then flush."""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _read_message() -> Optional[Dict[str, Any]]:
    """Read one JSON-RPC message from stdin (one line). Returns None at EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        return {"_parse_error": str(exc), "_raw": line}


def main() -> int:
    url = DEFAULT_URL
    # Line-buffer stdout so the client gets every message immediately.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stdin.reconfigure(line_buffering=True)

    # Sanity-check that the GUI is actually serving.  We send a
    # minimal initialize ping; if it fails we exit non-zero with a
    # helpful stderr message.
    init_payload = {
        "jsonrpc": "2.0", "id": "bridge-init", "method": "initialize",
        "params": {"protocolVersion": "2024-11-05",
                   "capabilities": {},
                   "clientInfo": {"name": "mcp-bridge", "version": "0.1"}},
    }
    try:
        _post_json(url, init_payload, timeout=5.0)
    except Exception as exc:
        print(f"# bridge: cannot reach {url}: {exc}", file=sys.stderr)
        print(f"#   is protein-studio running with the MCP server started?",
              file=sys.stderr)
        return 1

    # Main loop: read JSON-RPC from stdin, forward to HTTP, write back.
    while True:
        msg = _read_message()
        if msg is None:
            return 0
        if "_parse_error" in msg:
            err = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700,
                          "message": f"Parse error: {msg['_parse_error']}"},
            }
            _write_message(err)
            continue
        try:
            resp = _post_json(url, msg, timeout=120.0)
            _write_message(resp)
        except Exception as exc:
            err = msg.get("id")
            _write_message({
                "jsonrpc": "2.0", "id": err,
                "error": {"code": -32603,
                          "message": f"bridge: HTTP error: {exc!r}"},
            })


if __name__ == "__main__":
    sys.exit(main() or 0)