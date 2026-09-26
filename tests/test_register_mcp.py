"""Tests for scripts/register_mcp.py.

We don't shell out to the script — we import its functions and exercise
the parser / entry builders / atomic JSON writer directly. This keeps
the test fast and avoids touching real MCP client config files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from register_mcp import (  # noqa: E402
    CLIENTS, parse_compact_form, protein_studio_compact,
    protein_studio_entry, entry_from_compact,
    protein_studio_http_entry, protein_studio_http_url,
    read_json_or_empty, write_json_atomic,
)


# ----- compact form parser ---------------------------------------------

def test_parse_simple():
    e = parse_compact_form("foo:/usr/bin/python -m foo.server")
    assert e == {"name": "foo",
                 "command": "/usr/bin/python",
                 "args": ["-m", "foo.server"]}


def test_parse_quoted_path():
    e = parse_compact_form('pymol:/opt/x/python "/Users/me/path with space/server.py"')
    assert e["name"] == "pymol"
    assert e["args"] == ["/Users/me/path with space/server.py"]


def test_parse_no_colon_defaults_to_protein_studio():
    e = parse_compact_form("uvx mcp-server-fetch")
    assert e["name"] == "protein-studio"
    assert e["command"] == "uvx"
    assert e["args"] == ["mcp-server-fetch"]


def test_parse_bundled_compact_is_self_consistent():
    compact = protein_studio_compact()
    assert compact.startswith("protein-studio:")
    parsed = parse_compact_form(compact)
    assert parsed["name"] == "protein-studio"
    assert parsed["args"] == ["-m", "protein_studio.mcp_server"]


def test_parse_rejects_empty_command():
    with pytest.raises(ValueError, match="no command"):
        parse_compact_form("foo:")


def test_parse_rejects_empty_name():
    with pytest.raises(ValueError, match="no name"):
        parse_compact_form(":/usr/bin/python")


# ----- entry builders ---------------------------------------------------

def test_protein_studio_entry_has_required_fields():
    e = protein_studio_entry()
    for k in ("name", "command", "args", "cwd", "env"):
        assert k in e, f"missing {k}"
    assert e["name"] == "protein-studio"
    assert e["args"][-1] == "protein_studio.mcp_server"
    assert e["cwd"] == str(ROOT)
    assert "PYTHONPATH" in e["env"]
    assert e["env"]["QT_QPA_PLATFORM"] == "offscreen"


def test_protein_studio_entry_respects_env_override(monkeypatch):
    monkeypatch.setenv("PROTEIN_STUDIO_PYTHON", "/custom/python")
    e = protein_studio_entry()
    assert e["command"] == "/custom/python"


def test_entry_from_compact_uses_provided_token():
    e = entry_from_compact("fetch:uvx mcp-server-fetch")
    assert e["name"] == "fetch"
    assert e["command"] == "uvx"
    assert e["args"] == ["mcp-server-fetch"]


# ----- atomic JSON writer -----------------------------------------------

def test_atomic_write_creates_file(tmp_path: Path):
    p = tmp_path / "config.json"
    write_json_atomic(p, {"mcpServers": {"x": {"command": "y"}}})
    assert p.exists()
    data = json.loads(p.read_text())
    assert data == {"mcpServers": {"x": {"command": "y"}}}


def test_atomic_write_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "deep" / "nested" / "config.json"
    write_json_atomic(p, {"x": 1})
    assert p.exists()


def test_read_json_or_empty_handles_missing(tmp_path: Path):
    p = tmp_path / "missing.json"
    assert read_json_or_empty(p) == {}


def test_read_json_or_empty_handles_empty_file(tmp_path: Path):
    p = tmp_path / "empty.json"
    p.write_text("")
    assert read_json_or_empty(p) == {}


def test_read_json_or_empty_handles_real_json(tmp_path: Path):
    p = tmp_path / "real.json"
    p.write_text('{"a": 1, "b": {"c": 2}}')
    assert read_json_or_empty(p) == {"a": 1, "b": {"c": 2}}


def test_atomic_write_leaves_no_temp_file_on_success(tmp_path: Path):
    p = tmp_path / "config.json"
    write_json_atomic(p, {"x": 1})
    leftovers = list(tmp_path.glob("*.tmp-*"))
    assert leftovers == [], f"leftover temp files: {leftovers}"


# ----- supported clients list -------------------------------------------

def test_supported_clients_match_known_set():
    expected = {"claude-desktop", "cline", "cherry-studio", "continue"}
    assert set(CLIENTS.keys()) == expected


# ----- HTTP transport entry ---------------------------------------------

def test_http_entry_default_url():
    e = protein_studio_http_entry()
    assert e == {"type": "http", "url": "http://127.0.0.1:7820/mcp"}


def test_http_entry_respects_env_override(monkeypatch):
    monkeypatch.setenv("PROTEIN_STUDIO_MCP_URL",
                       "http://localhost:9000/mcp")
    e = protein_studio_http_entry()
    assert e["url"] == "http://localhost:9000/mcp"


def test_http_url_helper_matches_entry(monkeypatch):
    monkeypatch.delenv("PROTEIN_STUDIO_MCP_URL", raising=False)
    assert protein_studio_http_url() == protein_studio_http_entry()["url"]