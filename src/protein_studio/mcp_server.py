"""MCP server for LingSu Protein — exposes the StudioAPI as tools.

Two transports
-------------
1. **stdio** (legacy / standalone / tests)::

       python -m protein_studio.mcp_server

   Runs as its own process, creates a fresh ``HeadlessViewport`` so it
   has *something* to render against. Used by ``/tmp/mcp_smoke.py`` and
   for one-off headless scripting.

2. **HTTP / streamable-http** (the GUI's transport)::

       from protein_studio.mcp_server import serve_http
       await serve_http(api, host='127.0.0.1', port=7820)

   Runs *inside* whatever process holds the ``StudioAPI`` (i.e. the
   GUI). External MCP clients connect to ``http://host:port/mcp``. They
   see exactly the same scene the GUI sees — because the dispatch
   callbacks close over the *passed* api, not a new one.

For stdio-only clients (most current MCP client setups), use
``python -m protein_studio.mcp_bridge`` as the ``command`` in the
client config — it forwards stdin/stdout to the GUI's TCP endpoint.

Tool surface
------------
Scene ops:      list_objects, load_pdb, load_docking, load_alphafold,
                remove_object, clear_scene
Rep / scheme:   set_representation, set_scheme, list_representations,
                list_schemes
Visualisation:  refresh, screenshot, center_view, rotate,
                spin_start, spin_stop
Analysis:       detect_hbonds, compute_esp, get_info
Scripting:      run_python
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    {"name": "fep_build_system", "description": (
        "Build an alchemical FEP system from a PDB file. Returns "
        "metadata (n_particles, n_waters, n_disappearing, etc.) "
        "without running any MD. Use `mutation` for sidechain "
        "truncation FEP (e.g. 'Lys6') or `ligand_resname` for "
        "ligand decouple FEP (e.g. 'LIG')."),
     "inputSchema": {"type": "object", "properties": {
         "pdb_path": {"type": "string"},
         "mutation": {"type": "string",
                      "description": "Residue id like 'Lys6' or '6'."},
         "ligand_resname": {"type": "string"},
         "restraint_k_k_jmol_nm2": {"type": "number", "default": 50000.0},
     }, "required": ["pdb_path"], "additionalProperties": False}},
    {"name": "fep_configure_recipe", "description": (
        "Return the named FEP recipe's full parameter set. "
        "Recipes bundle n_lambdas, equilibration/production steps, "
        "sampling stride, and neighbor-re-eval settings. Pass "
        "name='list' to see all available recipes."),
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string"},
     }, "required": ["name"], "additionalProperties": False}},
    {"name": "fep_run_sweep", "description": (
        "Run a full lambda-sweep + BAR analysis on a PDB and "
        "return delta_G. Long-running (minutes to hours for "
        "production recipes) — start with recipe_name='quick_test' "
        "to validate the pipeline. Returns per-window breakdown "
        "and uncertainty."),
     "inputSchema": {"type": "object", "properties": {
         "pdb_path": {"type": "string"},
         "mode": {"type": "string", "enum": ["mutation", "ligand_decouple"]},
         "recipe_name": {"type": "string", "default": "quick_test"},
         "mutation": {"type": "string"},
         "ligand_resname": {"type": "string"},
     }, "required": ["pdb_path", "mode"], "additionalProperties": False}},
    {"name": "fep_run_sweep_with_checkpoint", "description": (
        "Same as fep_run_sweep but writes a pickle checkpoint after "
        "every λ window. Pass checkpoint_path to override the default "
        "(derived from the sweep inputs). On cancellation the partial "
        "state is preserved — call fep_resume_sweep with the same "
        "checkpoint_path to continue."),
     "inputSchema": {"type": "object", "properties": {
         "pdb_path": {"type": "string"},
         "mode": {"type": "string", "enum": ["mutation", "ligand_decouple"]},
         "recipe_name": {"type": "string", "default": "quick_test"},
         "mutation": {"type": "string"},
         "ligand_resname": {"type": "string"},
         "checkpoint_path": {"type": "string"},
     }, "required": ["pdb_path", "mode"], "additionalProperties": False}},
    {"name": "fep_resume_sweep", "description": (
        "Resume a sweep from an existing checkpoint file. Skips the "
        "already-completed λ windows and runs only the remainder. "
        "Returns the same shape as fep_run_sweep on completion."),
     "inputSchema": {"type": "object", "properties": {
         "checkpoint_path": {"type": "string"},
     }, "required": ["checkpoint_path"], "additionalProperties": False}},
    {"name": "fep_checkpoint_info", "description": (
        "Inspect a checkpoint file without loading OpenMM. Returns "
        "missing / corrupt / ok status with progress info."),
     "inputSchema": {"type": "object", "properties": {
         "checkpoint_path": {"type": "string"},
     }, "required": ["checkpoint_path"], "additionalProperties": False}},
    {"name": "fep_checkpoint_delete", "description": (
        "Delete a checkpoint file. Returns whether it existed."),
     "inputSchema": {"type": "object", "properties": {
         "checkpoint_path": {"type": "string"},
     }, "required": ["checkpoint_path"], "additionalProperties": False}},
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
        pv_plotter = getattr(api.viewport, "plotter", None)
        if pv_plotter is None:
            return {"warning":
                    "rotate() requires a GUI viewport (the bundled "
                    "HeadlessViewport used for headless rendering has no "
                    "plotter). Run the protein-studio GUI to use rotate."}
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
            "donor": f"{hb.donor.residue.name}{hb.donor.residue.number}",
            "donor_atom": hb.donor.name,
            "acceptor": f"{hb.acceptor.residue.name}"
                       f"{hb.acceptor.residue.number}",
            "acceptor_atom": hb.acceptor.name,
            "distance": round(hb.distance, 3),
        } for hb in hbs[:limit]]
    if name == "compute_esp":
        import numpy as np
        obj = api.scene.find(arguments["name"])
        if obj is None:
            raise KeyError(arguments["name"])
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
    if name == "fep_build_system":
        # Lazy import — OpenMM is heavy.
        from .core.fep_mcp import fep_build_system as _fep_build
        kwargs = {}
        if "mutation" in arguments:
            kwargs["mutation"] = arguments["mutation"]
        if "ligand_resname" in arguments:
            kwargs["ligand_resname"] = arguments["ligand_resname"]
        if "restraint_k_k_jmol_nm2" in arguments:
            kwargs["restraint_k_kJmol_nm2"] = (
                arguments["restraint_k_k_jmol_nm2"]
            )
        return _fep_build(arguments["pdb_path"], **kwargs)
    if name == "fep_configure_recipe":
        from .core.fep_mcp import (
            fep_configure_recipe as _fep_cfg,
            fep_list_recipes as _fep_list,
        )
        if arguments["name"] == "list":
            return _fep_list()
        return _fep_cfg(arguments["name"])
    if name == "fep_run_sweep":
        from .core.fep_mcp import fep_run_sweep as _fep_run
        kwargs = {"recipe_name": arguments.get("recipe_name", "quick_test")}
        if "mutation" in arguments:
            kwargs["mutation"] = arguments["mutation"]
        if "ligand_resname" in arguments:
            kwargs["ligand_resname"] = arguments["ligand_resname"]
        return _fep_run(arguments["pdb_path"], arguments["mode"], **kwargs)
    if name == "fep_run_sweep_with_checkpoint":
        from .core.fep_mcp import (
            fep_run_sweep_with_checkpoint as _fep_run_cp,
        )
        kwargs = {"recipe_name": arguments.get("recipe_name", "quick_test")}
        if "mutation" in arguments:
            kwargs["mutation"] = arguments["mutation"]
        if "ligand_resname" in arguments:
            kwargs["ligand_resname"] = arguments["ligand_resname"]
        if "checkpoint_path" in arguments:
            kwargs["checkpoint_path"] = arguments["checkpoint_path"]
        return _fep_run_cp(
            arguments["pdb_path"], arguments["mode"], **kwargs,
        )
    if name == "fep_resume_sweep":
        from .core.fep_mcp import fep_resume_sweep as _fep_resume
        return _fep_resume(arguments["checkpoint_path"])
    if name == "fep_checkpoint_info":
        from .core.fep_mcp import fep_checkpoint_info as _fep_info
        return _fep_info(arguments["checkpoint_path"])
    if name == "fep_checkpoint_delete":
        from .core.fep_mcp import fep_checkpoint_delete as _fep_del
        return _fep_del(arguments["checkpoint_path"])
    raise ValueError(f"unknown tool: {name}")


def json_dumps(obj: Any) -> str:
    """Safe JSON dump for tool results — falls back to repr for unknown
    objects (numpy arrays, vtkPolyData, etc.)."""
    def _default(o):
        try:
            return o.tolist()
        except AttributeError:
            return repr(o)
    return json.dumps(obj, default=_default, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Server factory (transport-agnostic)
# ---------------------------------------------------------------------------

def make_server(api: StudioAPI, name: str = "protein-studio"):
    """Build an MCP Server wired to the given ``api``.

    The returned server is transport-agnostic — pass it to either
    ``server.run(read, write, ...)`` for stdio or
    ``server.streamable_http_app()`` for HTTP. The two transports live
    in :func:`serve_stdio` and :func:`serve_http` below.
    """
    from mcp.server import Server
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    typed_tools: List[Tool] = [
        Tool(name=t["name"],
             description=t["description"],
             inputSchema=t["inputSchema"])
        for t in TOOL_CATALOG
    ]

    async def _on_list_tools(ctx, params):
        return ListToolsResult(tools=typed_tools)

    async def _on_call_tool(ctx, params):
        name_ = params.name
        arguments = dict(params.arguments or {})
        try:
            result = _dispatch(name_, arguments, api)
            return CallToolResult(
                content=[TextContent(type="text", text=json_dumps(result))],
                is_error=False,
            )
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

    return Server(name,
                  on_list_tools=_on_list_tools,
                  on_call_tool=_on_call_tool)


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

async def serve_stdio() -> None:
    """Run the MCP server on stdio with a fresh HeadlessViewport.

    Used by ``python -m protein_studio.mcp_server`` for standalone /
    headless use. The GUI does NOT call this — it uses
    :func:`serve_http` instead, so the api it controls is the same one
    the GUI is rendering.
    """
    from mcp.server.stdio import stdio_server

    api = StudioAPI(viewport=HeadlessViewport())
    server = make_server(api)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


async def serve_http(api: StudioAPI,
                     host: str = "127.0.0.1",
                     port: int = 7820,
                     log_level: str = "warning") -> None:
    """Run the MCP server bound to the given ``api`` over HTTP.

    External MCP clients (Claude Desktop, Cline, etc.) connect to
    ``http://host:port/mcp``. The server uses the *passed* api, so any
    change made through a tool call is immediately reflected in the
    GUI's scene, and vice versa.

    Designed to be run inside a QThread by the GUI's MCPController:

        await serve_http(api, host="127.0.0.1", port=7820)

    Blocks until cancelled. Cancellation triggers uvicorn's shutdown,
    which is graceful — in-flight requests get a chance to complete.
    """
    import uvicorn

    server = make_server(api)
    app = server.streamable_http_app()
    config = uvicorn.Config(app,
                            host=host,
                            port=port,
                            log_level=log_level,
                            access_log=False)
    uv_server = uvicorn.Server(config)
    await uv_server.serve()


# ---------------------------------------------------------------------------
# Entry point (stdio)
# ---------------------------------------------------------------------------

async def _main_stdio():
    await serve_stdio()


def main() -> int:
    """Entry point — ``python -m protein_studio.mcp_server``.

    Starts the stdio server. The GUI does not call this; it spawns an
    in-process HTTP server via :func:`serve_http` instead.
    """
    try:
        asyncio.run(_main_stdio())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())