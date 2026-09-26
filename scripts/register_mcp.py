#!/usr/bin/env python3
"""Register the LingSu Protein MCP server with one or more MCP clients.

Usage
-----
A. Register the bundled protein-studio server (all supported clients)::

       python scripts/register_mcp.py
       python scripts/register_mcp.py all
       python scripts/register_mcp.py claude-desktop

B. Register an arbitrary server in the compact "<name>:<command> [args...]"
   shorthand — same form you'd write in Claude Desktop's "command" column::

       python scripts/register_mcp.py \\
           --client claude-desktop \\
           protein-studio:/opt/.../python -m protein_studio.mcp_server

       python scripts/register_mcp.py --client cherry-studio --client cline \\
           fetch:uvx mcp-server-fetch

C. Just print the bundled protein-studio compact form (no writes)::

       python scripts/register_mcp.py --print

Supported clients
-----------------
- ``claude-desktop``   ``~/Library/Application Support/Claude/claude_desktop_config.json``
- ``cline``            ``~/.config/Code/.../cline_mcp_settings.json``
- ``cherry-studio``    ``~/.config/cherry-studio/mcp_settings.json``
- ``continue``         ``~/.continue/config.json``
- ``all``              register with all four (default when no client given)

Environment variables
---------------------
- ``PROTEIN_STUDIO_PYTHON``  override the bundled Python interpreter
- ``MCP_CWD``                override the cwd the entry sets on the server
- ``CLINE_SETTINGS``         override Cline config path (e.g. on Linux)
- ``CHERRY_STUDIO_SETTINGS`` override Cherry Studio config path
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PYTHON = (
    "/opt/homebrew/Caskroom/miniconda/base/envs/protein-studio/bin/python"
)
SERVER_KEY_PROTEIN_STUDIO = "protein-studio"


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

def _claude_desktop_path() -> Path:
    return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"


def _cline_path() -> Path:
    override = os.environ.get("CLINE_SETTINGS")
    if override:
        return Path(override)
    return (Path.home()
            / ".config/Code/User/globalStorage/saoudrizwan.claude-dev/settings"
            / "cline_mcp_settings.json")


def _cherry_studio_path() -> Path:
    override = os.environ.get("CHERRY_STUDIO_SETTINGS")
    if override:
        return Path(override)
    return Path.home() / ".config/cherry-studio/mcp_settings.json"


def _continue_path() -> Path:
    return Path.home() / ".continue/config.json"


CLIENTS: Dict[str, Callable[[], Path]] = {
    "claude-desktop": _claude_desktop_path,
    "cline":          _cline_path,
    "cherry-studio":  _cherry_studio_path,
    "continue":       _continue_path,
}


# ---------------------------------------------------------------------------
# Compact form parser — "<name>:<command> [args...]"
# ---------------------------------------------------------------------------

def parse_compact_form(token: str) -> dict:
    """Parse ``<name>:<command> [arg1 arg2 ...]`` into a dict.

    The first ``':'`` is the separator; everything after is tokenized
    with :mod:`shlex` so double-quoted segments may contain spaces.

    If the token has no ``':'``, the server name defaults to
    ``"protein-studio"``.
    """
    colon = token.find(":")
    if colon == -1:
        name = "protein-studio"
        rest = token.strip()
    else:
        name = token[:colon].strip()
        rest = token[colon + 1:].strip()

    if not name:
        raise ValueError(f"bad compact form (no name): {token!r}")
    if not rest:
        raise ValueError(f"bad compact form (no command): {token!r}")

    parts = shlex.split(rest, posix=True)
    if not parts:
        raise ValueError(f"bad compact form (no command): {token!r}")
    return {"name": name, "command": parts[0], "args": parts[1:]}


# ---------------------------------------------------------------------------
# Entry builders
# ---------------------------------------------------------------------------

def protein_studio_compact() -> str:
    """Compact form of the bundled protein-studio registration."""
    py = os.environ.get("PROTEIN_STUDIO_PYTHON", DEFAULT_PYTHON)
    return f"{SERVER_KEY_PROTEIN_STUDIO}:{py} -m protein_studio.mcp_server"


def _env_block() -> dict:
    cwd = Path(os.environ.get("MCP_CWD", str(PROJECT_ROOT)))
    return {
        "PYTHONPATH": str(cwd / "src"),
        "QT_QPA_PLATFORM": "offscreen",
    }


def protein_studio_entry() -> dict:
    parsed = parse_compact_form(protein_studio_compact())
    cwd = Path(os.environ.get("MCP_CWD", str(PROJECT_ROOT)))
    return {
        "name":    parsed["name"],
        "command": parsed["command"],
        "args":    parsed["args"],
        "cwd":     str(cwd),
        "env":     _env_block(),
    }


def entry_from_compact(token: str) -> dict:
    parsed = parse_compact_form(token)
    cwd = Path(os.environ.get("MCP_CWD", str(PROJECT_ROOT)))
    return {
        "name":    parsed["name"],
        "command": parsed["command"],
        "args":    parsed["args"],
        "cwd":     str(cwd),
        "env":     _env_block(),
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_json_or_empty(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not parse {path}: {exc}") from exc


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_one(client_name: str, factory: Callable[[], Path],
                 entry: dict) -> dict:
    config_path = factory()
    cfg = read_json_or_empty(config_path)
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    had = entry["name"] in servers
    servers[entry["name"]] = {
        "command": entry["command"],
        "args":    entry["args"],
        "cwd":     entry["cwd"],
        "env":     entry["env"],
    }
    cfg["mcpServers"] = servers
    write_json_atomic(config_path, cfg)
    return {"config_path": config_path, "had": had}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP_EPILOG = """
Examples:
  python scripts/register_mcp.py
  python scripts/register_mcp.py all
  python scripts/register_mcp.py claude-desktop
  python scripts/register_mcp.py --print

  python scripts/register_mcp.py --client claude-desktop \\
      protein-studio:/opt/.../python -m protein_studio.mcp_server
  python scripts/register_mcp.py --client cherry-studio --client cline \\
      fetch:uvx mcp-server-fetch
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="register_mcp.py",
        description=(
            "Register an MCP server (or the bundled protein-studio server) "
            "with one or more MCP clients."
        ),
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--client",
        action="append",
        metavar="NAME",
        help=(
            "MCP client to register with. May be repeated. "
            f"Known: {', '.join(list(CLIENTS) + ['all'])}. "
            "Defaults to 'all' when omitted."
        ),
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Print the bundled protein-studio compact form and exit "
             "(no config files are written).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help=(
            "Either a client name ('claude-desktop', 'cline', "
            "'cherry-studio', 'continue', 'all') — in which case the "
            "bundled protein-studio server is registered — or a compact "
            "'<name>:<command> [args...]' definition for an arbitrary "
            "MCP server."
        ),
    )
    p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="(compact form only) additional command-line tokens after "
             "the first positional; joined verbatim and re-tokenized.",
    )
    return p


def resolve_names(args: argparse.Namespace) -> List[str]:
    """Return the list of real client names to register with."""
    if args.client:
        names = list(args.client)
    elif args.target is None:
        names = list(CLIENTS)
    elif args.target == "all":
        names = list(CLIENTS)
    elif args.target in CLIENTS:
        names = [args.target]
    else:
        # Compact form — register with everyone.
        names = list(CLIENTS)
    # Expand any 'all' sentinel into real client names.
    if "all" in names:
        names = [n for n in names if n != "all"]
        names = sorted(set(names) | set(CLIENTS))
    return names


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print:
        print(protein_studio_compact())
        return 0

    # Build the entry — either from the bundled protein-studio server,
    # or from a compact "<name>:<command> [args...]" token.
    if args.target is None or (
        args.target in CLIENTS or args.target == "all"
    ):
        entry = protein_studio_entry()
    else:
        # Compact form — the user may have split it across argv tokens.
        # Rejoin everything after --client flags into one string.
        # We rely on argparse having separated --client + --print flags
        # from the rest, so anything left in target + rest is the form.
        token = args.target
        if args.rest:
            token = token + " " + " ".join(args.rest)
        entry = entry_from_compact(token)

    names = resolve_names(args)
    for name in names:
        if name not in CLIENTS:
            parser.error(f"unknown client: {name}\n"
                         f"known: {', '.join(list(CLIENTS) + ['all'])}")
        result = register_one(name, CLIENTS[name], entry)
        verb = "updated" if result["had"] else "added"
        print(f"  {verb:<8} {name:<15} {result['config_path']}")

    compact = f"{entry['name']}:{entry['command']} {' '.join(entry['args'])}"
    print("\nCompact form:")
    print(f"  {compact}")
    print("\nRestart the client to pick up the new server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())