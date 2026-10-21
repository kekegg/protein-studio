#!/usr/bin/env python3
"""Regenerate the documentation figures used by README.md and docs/.

The figures are rendered from the *current* representation code, so they
cannot drift away from the implementation the way hand-made screenshots do.

Usage
-----
    python scripts/render_readme_images.py            # every figure
    python scripts/render_readme_images.py ubiquitin  # only figures whose
                                                       # name contains "ubiquitin"
    python scripts/render_readme_images.py --list     # show the table

Design notes
------------
* **Polymer-only figures exclude HETATM records** (waters, ligands, ions).
  ``core/representations.py`` documents the same intent for publication
  figures — "a clean cartoon with no ligand clutter".  Without this the
  solvent waters of 1ubq.pdb show up as a cloud of small spheres and the
  figure reads as a bug.

* **The lighting rig is the application's own three-point setup**
  (see ``core/api.py`` → ``HeadlessViewport.screenshot``), so the figures
  look exactly like the live app.

* **The `secondary` palette is the current one** — helix orange-red,
  sheet yellow, loop lavender.  Older figures were rendered with a
  previous palette and a plain-tube cartoon, which is why they no longer
  matched the code.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

DOCS = os.path.join(REPO, "docs")
EXAMPLES = os.path.join(REPO, "examples")

# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

#: Shared three-point light rig (mirrors HeadlessViewport.screenshot).
LIGHTS = [
    ((0.5, 1.0, 0.5), 1.20, (1.00, 0.97, 0.92)),
    ((-0.5, -0.5, 0.3), 0.45, (0.85, 0.88, 0.95)),
    ((0.0, -0.2, -1.0), 0.65, (1.00, 0.92, 0.85)),
]

BG_BOTTOM = "#222222"
BG_TOP = "#111111"


def _new_plotter(size: tuple[int, int], ssao: bool = True):
    import pyvista as pv
    import vtk

    p = pv.Plotter(off_screen=True, window_size=list(size))
    p.set_background(BG_BOTTOM, top=BG_TOP)
    if ssao and hasattr(p, "enable_ssao"):
        try:
            p.enable_ssao(radius=0.6, bias=0.02)
        except Exception:
            pass
    try:
        p.renderer.RemoveAllLights()
    except Exception:
        pass
    for pos, intensity, color in LIGHTS:
        try:
            light = vtk.vtkLight()
            light.SetPosition(*pos)
            light.SetIntensity(intensity)
            light.SetColor(*color)
            light.PositionalOff()
            p.renderer.AddLight(light)
        except Exception:
            pass
    return p


def _add(p, mesh, material: dict | None = None) -> None:
    """Add a mesh to the plotter.

    ``ball_and_stick`` returns a *dict* of meshes (``{"atoms": ...,
    "bonds": ...}``) rather than a single ``PolyData``; handle both, the
    same way the application's own screenshot path does.
    """
    if mesh is None:
        return
    if isinstance(mesh, dict):
        for sub in mesh.values():
            _add(p, sub, material)
        return
    if getattr(mesh, "n_cells", 0) == 0:
        return
    has_cell = "colors" in (mesh.cell_data.keys() or [])
    has_point = "colors" in (mesh.point_data.keys() or [])
    scalars = "colors" if (has_cell or has_point) else None
    pref = "cell" if has_cell else "point"
    mat = {
        "specular": 1.0,
        "specular_power": 100,
        "ambient": 0.10,
        "diffuse": 0.90,
    }
    mat.update(material or {})
    try:
        p.add_mesh(
            mesh, scalars=scalars, rgb=True, preference=pref,
            smooth_shading=True, show_scalar_bar=False, **mat,
        )
    except Exception:
        try:
            p.add_mesh(mesh)
        except Exception:
            pass


def render(
    out_path: str,
    meshes: Sequence,
    size: tuple[int, int] = (1400, 1050),
    azimuth: float = 0.0,
    elevation: float = 0.0,
    zoom: float = 1.0,
    material: dict | None = None,
    ssao: bool = True,
) -> str:
    """Render *meshes* off-screen to *out_path*."""
    p = _new_plotter(size, ssao=ssao)
    try:
        for m in meshes:
            _add(p, m, material)
        try:
            p.camera_position = "iso"
        except Exception:
            pass
        p.reset_camera()
        if azimuth:
            p.camera.azimuth = azimuth
        if elevation:
            p.camera.elevation = elevation
        if zoom != 1.0:
            p.camera.zoom(zoom)
        p.show(screenshot=out_path)
    finally:
        try:
            p.close()
        except Exception:
            pass
    return out_path


# ---------------------------------------------------------------------------
# figure table
# ---------------------------------------------------------------------------


@dataclass
class Figure:
    out: str                     # file name inside docs/
    build: Callable[[], list]   # -> list of meshes
    size: tuple[int, int] = (1400, 1050)
    azimuth: float = 0.0
    elevation: float = 0.0
    zoom: float = 1.0
    note: str = ""
    #: material overrides — see ``_add``.  The default (specular 1.0) is
    #: right for the ribbon / surface figures but blows out on the very
    #: dark CPK carbon of the ball-and-stick figure, where the highlights
    #: read as spurious white dots.
    material: dict = field(default_factory=dict)
    #: Screen-space ambient occlusion.  Must be off for dense ball-and-stick
    #: meshes — the geometry occludes itself so heavily that the whole
    #: structure goes black.
    ssao: bool = True


#: Matte material for the CPK ball-and-stick figure.
MATTE = {"specular": 0.16, "specular_power": 24, "ambient": 0.30, "diffuse": 0.80}


def _mol(path: str):
    from protein_studio.core.io import load_pdb
    return load_pdb(path)


def _scheme(key: str):
    from protein_studio.core.colors import SCHEME_PRESETS
    return SCHEME_PRESETS[key]


def _fig_ubiquitin_secondary() -> list:
    """Ubiquitin, SS-aware ribbon, coloured by secondary structure."""
    from protein_studio.core.representations import ribbon
    mol = _mol(os.path.join(EXAMPLES, "1ubq.pdb"))
    return [ribbon(mol, scheme=_scheme("secondary"))]


def _fig_hemoglobin_chain() -> list:
    """Haemoglobin, SS-aware ribbon, coloured by chain."""
    from protein_studio.core.representations import ribbon
    mol = _mol(os.path.join(EXAMPLES, "4hhb.pdb"))
    return [ribbon(mol, scheme=_scheme("chain"))]


def _fig_ubiquitin_balls() -> list:
    """Ubiquitin, ball-and-stick, CPK element colours (polymer only)."""
    from protein_studio.core.representations import ball_and_stick
    mol = _mol(os.path.join(EXAMPLES, "1ubq.pdb"))
    return [ball_and_stick(mol, scheme=_scheme("element"), include_hetero=False)]


def _fig_ubiquitin_hbonds() -> list:
    """Ubiquitin ribbon plus the detected backbone hydrogen bonds.

    The H-bond cut-off is matched to the one the secondary-structure
    detector uses (``core/secondary.py`` → ``HBOND_CUTOFF_DEFAULT``), so
    the yellow lines are exactly the bonds that define the helices and
    strands drawn in the ribbon — the two halves of the figure agree.
    """
    from protein_studio.core.representations import ribbon, hbonds_representation
    from protein_studio.core.secondary import HBOND_CUTOFF_DEFAULT
    mol = _mol(os.path.join(EXAMPLES, "1ubq.pdb"))
    return [
        ribbon(mol, scheme=_scheme("secondary")),
        hbonds_representation(mol, dist_cutoff=HBOND_CUTOFF_DEFAULT),
    ]


def _fig_alphafold_plddt() -> list:
    """Real AlphaFold prediction (calmodulin-1, P0DP23) coloured by pLDDT.

    Calmodulin-1 is a good showcase: the four EF-hand domains are predicted
    with very-high confidence while the inter-domain linkers fall into the
    confident / low buckets, so several palette colours are visible in one
    compact figure.
    """
    from protein_studio.core.representations import ribbon
    path = os.path.join(EXAMPLES, "AF-P0DP23-F1.pdb")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing — download it with:\n"
            "  curl -L -o examples/AF-P0DP23-F1.pdb \\\n"
            "    https://alphafold.ebi.ac.uk/files/AF-P0DP23-F1-model_v6.pdb"
        )
    mol = _mol(path)
    return [ribbon(mol, scheme=_scheme("alphafold_plddt"))]


def _fig_hemoglobin_esp() -> list:
    """Haemoglobin solvent-excluded surface coloured by electrostatic potential."""
    from protein_studio.core.representations import esp_surface
    mol = _mol(os.path.join(EXAMPLES, "4hhb.pdb"))
    return [esp_surface(mol, scheme=_scheme("electrostatic"), include_hetero=False)]


FIGURES: list[Figure] = [
    Figure("ubiquitin_ribbon_secondary.png", _fig_ubiquitin_secondary,
           azimuth=20, elevation=12, zoom=1.35,
           note="ubiquitin · ribbon · secondary-structure colours"),
    Figure("hemoglobin_ribbon_chain.png", _fig_hemoglobin_chain,
           azimuth=25, elevation=10, zoom=1.25,
           note="haemoglobin · ribbon · chain colours"),
    Figure("ubiquitin_balls_element.png", _fig_ubiquitin_balls,
           azimuth=15, elevation=8, zoom=1.30, material=MATTE, ssao=False,
           note="ubiquitin · ball-and-stick · CPK element colours"),
    Figure("ubiquitin_hbonds.png", _fig_ubiquitin_hbonds,
           azimuth=20, elevation=12, zoom=1.45,
           note="ubiquitin · ribbon + detected hydrogen bonds (yellow)"),
    Figure("alphafold_style_pLDDT.png", _fig_alphafold_plddt,
           azimuth=20, elevation=12, zoom=1.35,
           note="AlphaFold prediction AF-P0DP23-F1 (calmodulin-1) · ribbon · pLDDT colours"),
    Figure("hemoglobin_esp_surface.png", _fig_hemoglobin_esp,
           azimuth=20, elevation=10, zoom=1.20,
           note="haemoglobin · SES surface · electrostatic potential"),
]


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = {a for a in argv[1:] if a.startswith("-")}

    if "--list" in flags:
        for f in FIGURES:
            print(f"  {f.out:34} {f.note}")
        return 0

    os.makedirs(DOCS, exist_ok=True)
    selected = [
        f for f in FIGURES
        if not args or any(a in f.out for a in args)
    ]
    if not selected:
        print("no figures matched", file=sys.stderr)
        return 1

    failures = 0
    for f in selected:
        out = os.path.join(DOCS, f.out)
        try:
            meshes = f.build()
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {f.out:34} build failed: {exc}")
            failures += 1
            continue
        try:
            render(out, meshes, size=f.size, azimuth=f.azimuth,
                   elevation=f.elevation, zoom=f.zoom, material=f.material,
                   ssao=f.ssao)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {f.out:34} render failed: {exc}")
            failures += 1
            continue
        size_kb = os.path.getsize(out) / 1024
        print(f"  ✓ {f.out:34} {size_kb:7.1f} KB   {f.note}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
