"""MCP server for LingSu Protein — exposes the StudioAPI as tools.

Run with::

    python -m protein_studio.mcp_server

The server runs in stdio mode (no GUI) so an agent can pipe Python
through ``run_python`` and inspect outputs via the other tools. It
uses a :class:`~protein_studio.api.HeadlessViewport` under the hood
so screenshots actually rasterize to PNG.

Tool surface
------------
Scene ops:
    list_objects            — enumerate scene objects
    load_pdb                — load a PDB file (file path required)
    load_docking            — load Vina / EquiScore results
    load_alphafold          — load an AlphaFold prediction
    remove_object           — drop one object from the scene
    clear_scene             — drop everything

Representation / coloring:
    set_representation      — change rep on a scene object
    set_scheme              — change color scheme on a scene object
    list_representations    — enumerate available reps
    list_schemes            — enumerate available schemes

Visualisation:
    refresh                 — re-render after a change
    screenshot              — write the current view to PNG
    center_view             — translate scene so centroid is at origin
    rotate                  — orbit camera by an angle (deg)
    spin_start / spin_stop  — auto-orbit on/off (GUI only)

Analysis:
    detect_hbonds           — list H-bonds in a scene object
    compute_esp             — ESP stats (min/max/mean potential)
    get_info                — atom / residue / chain count for an object

Scripting:
    run_python              — execute Python in the API namespace
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

from .api import HeadlessViewport, StudioAPI
from .core.colors import SCHEME_PRESETS
from .core.representations import available_representations
from .core.scene import SceneObject


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------
# Tools are defined as plain dicts (not mcp.types.Tool) until the server
# actually boots, so that ``import protein_studio`` from the GUI doesn't
# pull pydantic/uvicorn transitively.

TOOL_CATALOG: List[Dict[str, Any]] = [
    {"name": "list_objects", "description": (
        "List all objects currently in the scene (name, representation, "
        "scheme, n_atoms, visible)."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "load_pdb", "description": (
        "Load a PDB file from disk into the scene. "
        "Returns the new object's name and atom count."),
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string",
                  "description": "Absolute path to a .pdb file on disk."},
         "name": {"type": "string",
                  "description": "Optional display name. Defaults to the "
                                 "file stem."},
         "representation": {"type": "string",
                            "description": f"One of: "
                            f"{', '.join(available_representations())}. "
                            f"Default 'ribbon'."},
         "scheme": {"type": "string",
                    "description": f"One of: {', '.join(SCHEME_PRESETS)}. "
                    f"Default 'secondary'."},
     }, "required": ["path"], "additionalProperties": False}},
    {"name": "load_docking", "description": (
        "Load a Vina PDBQT file or an EquiScore directory of poses. "
        "Returns the number of poses loaded and their ranks."),
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string",
                  "description": "Path to a .pdbqt file or directory "
                                 "containing poses."},
         "top_n": {"type": "integer",
                   "description": "If set, keep only the top-N best-scoring "
                                  "poses."},
     }, "required": ["path"], "additionalProperties": False}},
    {"name": "load_alphafold", "description": (
        "Load an AlphaFold PDB (pLDDT scores in the bfactor column). "
        "Returns the new object's name and pLDDT summary."),
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"},
         "name": {"type": "string"},
     }, "required": ["path"], "additionalProperties": False}},
    {"name": "remove_object", "description": (
        "Remove a single scene object by name."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
     }, "required": ["name"], "additionalProperties": False}},
    {"name": "clear_scene", "description": (
        "Remove every object from the scene."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "set_representation", "description": (
        f"Change the representation of an object. "
        f"Available: {', '.join(available_representations())}"),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "representation": {"type": "string"},
     }, "required": ["name", "representation"],
     "additionalProperties": False}},
    {"name": "set_scheme", "description": (
        f"Change the color scheme of an object. "
        f"Available: {', '.join(SCHEME_PRESETS)}"),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "scheme": {"type": "string"},
     }, "required": ["name", "scheme"],
     "additionalProperties": False}},
    {"name": "list_representations", "description": (
        "Enumerate available representations."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "list_schemes", "description": (
        "Enumerate available color schemes."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "refresh", "description": (
        "Re-render every visible scene object. "
        "Call this after making multiple changes so they take effect "
        "in one screenshot."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "screenshot", "description": (
        "Render the current scene to a PNG file at the given path. "
        "The viewport is auto-refreshed first."),
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string",
                  "description": "Absolute path for the output PNG. "
                                 "Parent directory must exist."},
     }, "required": ["path"], "additionalProperties": False}},
    {"name": "center_view", "description": (
        "Translate every scene object so the centroid is at the origin."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "rotate", "description": (
        "Orbit the camera by `angle` degrees around the vertical (Z) "
        "axis. Positive = clockwise from above."),
     "inputSchema": {"type": "object", "properties": {
         "angle": {"type": "number",
                   "description": "Degrees to rotate (can be negative or "
                                  "larger than 360)."},
     }, "required": ["angle"], "additionalProperties": False}},
    {"name": "spin_start", "description": (
        "Start auto-orbit (~20 s per full turn). Call spin_stop to halt. "
        "In headless mode this is a no-op; rotate manually instead."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "spin_stop", "description": (
        "Stop the auto-orbit started by spin_start. "
        "In headless mode this is a no-op."),
     "inputSchema": {"type": "object", "properties": {},
                     "additionalProperties": False}},
    {"name": "detect_hbonds", "description": (
        "Run Baker-Hubbard H-bond detection on a scene object. "
        "Returns up to `limit` H-bonds sorted by distance."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "limit": {"type": "integer", "default": 50},
     }, "required": ["name"], "additionalProperties": False}},
    {"name": "compute_esp", "description": (
        "Compute per-vertex ESP stats on a scene object "
        "(min/max/mean/median/std of the potential field). "
        "Useful for sanity-checking whether the electrostatic "
        "scheme is producing sensible values."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
     }, "required": ["name"], "additionalProperties": False}},
    {"name": "get_info", "description": (
        "Atom / residue / chain counts for a scene object, plus "
        "its bounding-box dimensions."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
     }, "required": ["name"], "additionalProperties": False}},
    {"name": "run_python", "description": (
        "Execute an arbitrary Python snippet in the StudioAPI "
        "namespace. `D` is the StudioAPI handle, `np` is numpy. "
        "Print statements and tracebacks are captured and returned. "
        "Use this for ad-hoc scripting: animations, custom exports, "
        "geometry inspection, etc."),
     "inputSchema": {"type": "object", "properties": {
         "code": {"type": "string"},
     }, "required": ["code"], "additionalProperties": False}},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obj_summary(obj: SceneObject) -> Dict[str, Any]:
    return {
        "name": obj.name,
        "representation": obj.representation,
        "scheme": obj.color_scheme.key,
        "n_atoms": obj.molecule.n_atoms(),
        "visible": obj.visible,
    }


def _run_python(api: StudioAPI, code: str) -> List[Dict[str, Any]]:
    """Execute Python in the API namespace; return captured stdout /
    stderr / exception as a list of dicts."""
    import numpy as np  # noqa: F401  (exposed to user script)
    ns = {"D": api, "np": np, "__name__": "__mcp_script__"}
    out, err = io.StringIO(), io.StringIO()
    try:
        compiled = compile(code, "<mcp-script>", "exec")
        with redirect_stdout(out), redirect_stderr(err):
            exec(compiled, ns)
        return [{"status": "ok",
                 "stdout": out.getvalue(),
                 "stderr": err.getvalue(),
                 "exports": {k: repr(v) for k, v in ns.items()
                             if k not in ("D", "np", "__name__")
                             and not k.startswith("_")}}]
    except SystemExit:
        return [{"status": "exit",
                 "stdout": out.getvalue(),
                 "stderr": err.getvalue()}]
    except Exception:
        return [{"status": "error",
                 "stdout": out.getvalue(),
                 "stderr": err.getvalue(),
                 "traceback": traceback.format_exc()}]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def _make_server():
    """Build an MCP Server wired to a fresh StudioAPI.

    Lazy imports of ``mcp`` so that importing ``protein_studio`` for the
    GUI doesn't pull in pydantic/uvicorn/etc. The Server constructor
    takes ``on_list_tools`` and ``on_call_tool`` keyword callbacks in
    MCP 2.2.0 — the decorator-based FastMCP pattern from older versions
    is not available here.
    """
    from mcp.server import Server
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    api = StudioAPI(viewport=HeadlessViewport())

    # Convert the catalog dicts into typed Tool objects.
    typed_tools: List[Tool] = [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in TOOL_CATALOG
    ]

    async def _on_list_tools(ctx, params):
        # ``params`` is PaginatedRequestParams | None — we return everything.
        return ListToolsResult(tools=typed_tools)

    async def _on_call_tool(ctx, params):
        # ``params`` is CallToolRequestParams — has .name and .arguments.
        name = params.name
        arguments = dict(params.arguments or {})
        try:
            result = _dispatch(name, arguments, api)
            return _wrap_result(result)
        except Exception as exc:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json_dumps({"error": str(exc),
                                      "type": type(exc).__name__,
                                      "traceback": traceback.format_exc()}),
                )],
                is_error=True,
            )

    return Server(
        "protein-studio",
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )


def _dispatch(name: str, arguments: Dict[str, Any],
              api: StudioAPI) -> Any:
    """Route a tool call to the right StudioAPI method."""
    if name == "list_objects":
        return [_obj_summary(o) for o in api.scene.objects]
    if name == "load_pdb":
        p = Path(arguments["path"])
        if not p.exists():
            raise FileNotFoundError(p)
        rep = arguments.get("representation", "ribbon")
        sch = arguments.get("scheme", "secondary")
        obj = api.load(str(p), representation=rep, scheme=sch)
        if "name" in arguments:
            obj.name = arguments["name"]
        return {"name": obj.name,
                "n_atoms": obj.molecule.n_atoms(),
                "representation": rep,
                "scheme": sch}
    if name == "load_docking":
        p = Path(arguments["path"])
        if not p.exists():
            raise FileNotFoundError(p)
        top_n = arguments.get("top_n")
        poses = api.load_docking(str(p), top_n=top_n)
        return {"n_poses": len(poses),
                "ranks": [pose.rank for pose in poses],
                "scores": [round(pose.score, 3) for pose in poses]}
    if name == "load_alphafold":
        p = Path(arguments["path"])
        if not p.exists():
            raise FileNotFoundError(p)
        name_arg = arguments.get("name")
        obj = api.load_alphafold(str(p), name=name_arg)
        return {"name": obj.name, "n_atoms": obj.molecule.n_atoms()}
    if name == "remove_object":
        api.remove(arguments["name"])
        return {"removed": arguments["name"]}
    if name == "clear_scene":
        api.clear()
        return {"cleared": True, "n_objects": api.scene.n_objects()}
    if name == "set_representation":
        api.set_representation(arguments["name"],
                               arguments["representation"])
        return {"ok": True}
    if name == "set_scheme":
        api.set_scheme(arguments["name"], arguments["scheme"])
        return {"ok": True}
    if name == "list_representations":
        return list(available_representations())
    if name == "list_schemes":
        return list(SCHEME_PRESETS.keys())
    if name == "refresh":
        api.refresh()
        return {"ok": True, "n_objects": api.scene.n_objects()}
    if name == "screenshot":
        p = Path(arguments["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        api.refresh()
        api.screenshot(str(p))
        return {"path": str(p), "size_bytes": p.stat().st_size}
    if name == "center_view":
        api.center_view()
        return {"ok": True}
    if name == "rotate":
        pv_plotter = api.viewport.plotter
        pv_plotter.camera.Azimuth(float(arguments["angle"]))
        pv_plotter.render()
        return {"azimuth_deg": pv_plotter.camera.azimuth}
    if name == "spin_start":
        api.refresh()
        return {"warning":
                "Spin loop requires the GUI; in headless mode use "
                "rotate() in a loop or run protein-studio's GUI."}
    if name == "spin_stop":
        return {"ok": True}
    if name == "detect_hbonds":
        from .core.hbonds import detect_hbonds as _detect
        obj = api.scene.find(arguments["name"])
        if obj is None:
            raise KeyError(arguments["name"])
        limit = arguments.get("limit", 50)
        hbs = _detect(obj.molecule)
        return [{
            "donor": f"{hb.donor_residue.resname}{hb.donor_residue.number}",
            "donor_atom": hb.donor.name,
            "acceptor": f"{hb.acceptor_residue.resname}"
                       f"{hb.acceptor_residue.number}",
            "acceptor_atom": hb.acceptor.name,
            "distance": round(hb.distance, 3),
        } for hb in hbs[:limit]]
    if name == "compute_esp":
        import numpy as np
        obj = api.scene.find(arguments["name"])
        if obj is None:
            raise KeyError(arguments["name"])
        # Force esp_surface rep so the mesh has potential data.
        api.set_representation(arguments["name"], "esp_surface")
        api.refresh()
        mesh = api.viewport.plots.get(arguments["name"])
        if mesh is None or not hasattr(mesh, "point_data"):
            return {"error": "no mesh"}
        atoms_mesh = mesh.get("atoms") if isinstance(mesh, dict) else mesh
        pot = atoms_mesh.point_data.get("potential")
        if pot is None:
            return {"error":
                    "no potential data — rep must be esp_surface"}
        return {
            "n_vertices": int(pot.shape[0]),
            "min": float(pot.min()),
            "max": float(pot.max()),
            "mean": float(pot.mean()),
            "median": float(np.median(pot)),
            "std": float(pot.std()),
        }
    if name == "get_info":
        obj = api.scene.find(arguments["name"])
        if obj is None:
            raise KeyError(arguments["name"])
        m = obj.molecule
        chains = list({a.residue.chain_id for a in m.atoms
                       if a.residue is not None})
        return {
            "name": obj.name,
            "n_atoms": m.n_atoms(),
            "n_residues": sum(len(c.residues) for c in m.chains),
            "n_chains": len(chains),
            "chains": chains,
            "representation": obj.representation,
            "scheme": obj.color_scheme.key,
            "visible": obj.visible,
            "center_of_mass": list(m.center_of_mass()),
        }
    if name == "run_python":
        return _run_python(api, arguments["code"])
    raise ValueError(f"unknown tool: {name}")


def json_dumps(obj: Any) -> str:
    """Safe JSON dump for tool results — falls back to repr for unknown
    objects (numpy arrays, vtkPolyData, etc.)."""
    import json
    def _default(o):
        try:
            return o.tolist()
        except AttributeError:
            return repr(o)
    return json.dumps(obj, default=_default, ensure_ascii=False)


def _wrap_result(result: Any) -> "CallToolResult":
    """Turn a Python value (dict / list / str / etc.) into the
    CallToolResult that MCP 2.2.0 expects."""
    from mcp.types import CallToolResult, TextContent
    return CallToolResult(
        content=[TextContent(type="text", text=json_dumps(result))],
        is_error=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main():
    server = _make_server()
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


def main() -> int:
    """Entry point — ``python -m protein_studio.mcp_server``."""
    import asyncio
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())