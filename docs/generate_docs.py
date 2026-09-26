"""Generate developer and user documentation (docx + pptx).

Run with::

    python docs/generate_docs.py

Outputs are written to ``docs/``:

  - DeveloperGuide.docx / DeveloperGuide.pptx
  - UserGuide.docx       / UserGuide.pptx

Content is defined in two plain-Python data structures at the top of
the file (``DEVELOPER_GUIDE`` and ``USER_GUIDE``) so the .docx and
.pptx renderers stay in sync — change the content once, both files
update.

Headings / body are styled with a custom palette matching the
LingSu Protein app's dark theme.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Common docx/pptx units — both libraries export these with the same
# names; we import them up-front so the per-renderer helpers can
# reference them without each renderer re-importing.
from docx.shared import Pt, Inches, Cm, RGBColor as _RGBColorDocx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor as _RGBColorPptx
from pptx.enum.text import PP_ALIGN

# Resolve paths regardless of where the script is invoked from.
HERE = Path(__file__).resolve().parent
SCREENSHOTS = HERE / "screenshots"
OUT = HERE  # write into docs/ alongside the script

APP_NAME = "LingSu Protein"
APP_VERSION = "0.1.0"


# ============================================================ content data


DEVELOPER_GUIDE: List[Tuple[str, List]] = [
    ("Overview", [
        ("text", f"{APP_NAME} is a Blender-like molecular visualization desktop "
                  f"application. It pairs a PySide6 GUI with PyVista/VTK for 3D "
                  f"rendering, scikit-image for marching cubes, scipy for the "
                  f"Euclidean distance transform, and the AMBER ff99SB-style "
                  f"partial-charge table ported from 3Dmol.js."),
        ("text", "This guide is for contributors — it covers the module layout, "
                  "key design decisions, and how to add new representations, "
                  "color schemes, or extensions."),
    ]),
    ("Repository layout", [
        ("bullets", [
            "src/protein_studio/ — package source",
            "    api.py        StudioAPI — load/refresh/control surface for the GUI",
            "    app.py        application entry point (CLI flags + QApplication)",
            "    resources.py  icon loader (icon.icns → QIcon)",
            "    core/         data model + rendering",
            "        molecule.py    Atom / Residue / Chain / Molecule dataclasses",
            "        io.py          PDB parser",
            "        scene.py       Scene + SceneObject, add/remove/update_object",
            "        colors.py      ColorScheme + SCHEME_PRESETS",
            "        representations.py  7 reps (cartoon / ribbon / ball_and_stick / "
            "                                 surface / points / esp_surface / hbonds)",
            "        surface.py     EDTSurf-style SES surface generator",
            "        marching_cubes.py  skimage wrapper",
            "        charges.py     AMBER partial-charge table + ESP coloring",
            "        hbonds.py      Baker-Hubbard H-bond detection",
            "        trajectory.py  MDAnalysis wrapper",
            "        alphafold.py   pLDDT color scheme",
            "        docking.py     Vina PDBQT + EquiScore loaders",
            "    ui/          PySide6 widgets",
            "        viewport.py     3D viewport (PyVistaQt + macOS OpenGL workarounds)",
            "        outliner.py     left-side scene tree",
            "        properties.py   right-side inspector",
            "        console.py      REPL Python console",
            "        editor.py       file-editor + script runner (new)",
            "        main_window.py  4-pane Blender-like layout",
            "tests/        pytest suite (69 tests)",
            "examples/     sample PDBs (1aki.pdb, 1ubq.pdb, 4hhb.pdb)",
            "resources/    bundled icon (icon.icns / icon.ico / icon_*.png)",
            "docs/         this documentation",
        ]),
    ]),
    ("Quick start — running from source", [
        ("code", "conda activate protein-studio\n"
                  "python -m protein_studio.app examples/1aki.pdb"),
        ("text", "The above opens the GUI and auto-loads 1aki as a ribbon. "
                  "Pass additional PDB files as positional args to load them on "
                  "startup. Pass --headless to do a one-shot scripted render "
                  "(--output render.png, --representation cartoon, --scheme secondary)."),
    ]),
    ("Running tests", [
        ("code", "conda activate protein-studio\n"
                  "python -m pytest tests/ -v"),
        ("text", "The suite covers core molecule/scene ops, all 7 representations, "
                  "the AMBER charge table, ESP surface, the bundled Python editor, "
                  "and the resources module (icon + app name). Total: 69 tests."),
    ]),
    ("Design decisions worth knowing", [
        ("text", "1. EDTSurf-style SES algorithm"),
        ("text", "We port 3Dmol.js's ProteinSurface4 to Python: voxelize atoms → "
                  "scipy.ndimage.distance_transform_edt → skimage.measure.marching_cubes. "
                  "Defaults to scale_factor=2 (0.5 Å voxels). Three surface types: "
                  "VDW (1), SAS (3), MS/SES (2). See core/surface.py."),
        ("text", "2. AMBER partial-charge table"),
        ("text", "Full AMBER ff99SB-style lookup per (residue, atom) for the 20 "
                  "standard amino acids, ported from 3Dmol.js's partialCharges.ts. "
                  "Falls back to element-wise heuristic when atoms aren't in the "
                  "table. See core/charges.py."),
        ("text", "3. Per-SS ribbon cross-section"),
        ("text", "ribbon() picks a per-sample (width, thickness) ellipse in the "
                  "(side, binormal) plane depending on SS: helix becomes a full circle "
                  "(1.4 Å radius, looks like a real cylinder under Phong), loop is a "
                  "thin tube, sheet is a flat arrow with a 2× widening at the C-terminal "
                  "end. Smoothed along the chain with a Gaussian kernel so helix→loop "
                  "boundaries don't pop."),
        ("text", "4. macOS OpenGL mapper workaround"),
        ("text", "PyVistaQt on macOS corrupts actor color state when many actors "
                  "are added. _add_object_to_plotter installs a fresh vtkDataSetMapper "
                  "per actor and calls SetColorModeToDirectScalars() for 3-component "
                  "RGB arrays (so ESP / ball-and-stick colors reach the shader "
                  "verbatim). See _add_object_to_plotter in ui/viewport.py."),
        ("text", "5. Scene.add() deduplicates by name"),
        ("text", "Loading the same PDB twice replaces the existing object rather "
                  "than appending a second one (which would cause actor-name "
                  "collisions in the plotter and show the old version of the protein)."),
    ]),
    ("Adding a new representation", [
        ("text", "Implement a function in core/representations.py with the "
                  "signature (mol, scheme, **kwargs) -> pyvista.PolyData (or a "
                  "dict {\"atoms\": PolyData, \"bonds\": PolyData}). Register it in "
                  "the REPRESENTATIONS dict. Use build_mesh() (in the same file) "
                  "to get the canonical mesh-build pipeline."),
        ("text", "If your rep needs per-vertex atom_id or other metadata, "
                  "expose it as point_data so downstream properties (coloring, "
                  "ESP, H-bond overlay) can pick it up."),
    ]),
    ("Adding a new color scheme", [
        ("text", "Add an entry to SCHEME_PRESETS in core/colors.py with a unique "
                  "key (e.g. \"b_factor\"). The ColorScheme.color() method takes an "
                  "atom and returns an (r, g, b) tuple in [0, 1]. For schemes that "
                  "depend on per-vertex data (like electrostatic), handle the "
                  "coloring inside the rep function and have the scheme return "
                  "uniform — the viewport will pick up the rep's per-vertex "
                  "potential data via point_data['potential']."),
    ]),
    ("Adding an extension (H-bonds / trajectory / AlphaFold / docking)", [
        ("text", "Each extension lives in core/<name>.py and exposes top-level "
                  "functions plus dataclasses. Add a method to StudioAPI "
                  "(src/protein_studio/api.py) that wires the extension into the "
                  "scene. Wire a toolbar button in ui/main_window.py to call the "
                  "API method via a QFileDialog. Each extension has its own tests "
                  "in tests/test_extensions.py."),
    ]),
    ("Build / packaging", [
        ("text", "Currently shipped as a conda env (protein-studio) launched "
                  "via `python -m protein_studio.app`. A PyInstaller spec for a "
                  "double-clickable .app / .exe would be a natural next step "
                  "(the resources/ icon is already in place for that)."),
    ]),
]


USER_GUIDE: List[Tuple[str, List]] = [
    ("Welcome to LingSu Protein", [
        ("text", f"{APP_NAME} is a desktop molecular viewer built around the same "
                  f"representations you'd find in PyMOL and 3Dmol.js — cartoon, "
                  f"ribbon, ball-and-stick, molecular surface, ESP surface, "
                  f"ball-and-stick H-bonds, point cloud, AlphaFold pLDDT coloring."),
        ("text", "On first launch the app opens the bundled example 1aki.pdb as "
                  "a ribbon so you have something to look at immediately. Use "
                  "File → Clear Scene (Ctrl-Shift-C) any time to start fresh."),
    ]),
    ("GUI overview", [
        ("text", "The window is split into four Blender-style panels:"),
        ("bullets", [
            "Outliner (left)    — scene tree listing every loaded object",
            "Viewport (center)  — the 3D scene",
            "Properties (right) — selected object's representation + color scheme",
            "Console / Script (bottom) — REPL (>>> ) and file editor with Run button",
        ]),
    ]),
    ("Loading structures", [
        ("bullets", [
            "File → Open PDB… (Ctrl-O) — pick a .pdb file",
            "File → Open Docking… (Ctrl-D) — Vina .pdbqt or an EquiScore directory",
            "File → Open Trajectory… (Ctrl-T) — topology .pdb + trajectory .dcd/.xtc",
            "File → Open AlphaFold… (Ctrl-Shift-A) — load a predicted structure "
            "with pLDDT scores",
            "Drop the file into the viewport (drag-and-drop is in scope for the "
            "next iteration)",
        ]),
        ("text", "Loading the same PDB twice replaces the existing scene object "
                  "(so you can edit the representation without leaving ghost "
                  "copies)."),
    ]),
    ("Changing representation & color scheme", [
        ("text", "Click any object in the Outliner (or in the viewport) to "
                  "select it. The Properties panel on the right shows:"),
        ("bullets", [
            "Representation: cartoon | ribbon | ball_and_stick | surface | "
            "points | esp_surface | hbonds",
            "Color scheme:    uniform | element | residue | chain | secondary | "
            "b-factor | electrostatic | alphafold_plddt (when available)",
            "Visible in viewport: toggle the object's visibility",
            "Position / Rotation / Scale: numeric transform",
        ]),
        ("text", "Tip: with surface or esp_surface reps, pick the "
                  "\"Electrostatic\" scheme to get the red-white-blue ESP "
                  "coloring. Otherwise the surface is colored by the chosen "
                  "scheme (element / secondary / etc.)."),
    ]),
    ("View controls", [
        ("bullets", [
            "Left-drag — orbit the camera",
            "Right-drag (or Shift-drag) — pan",
            "Scroll wheel — zoom (PyVista default: dolly the camera along its "
            "view direction)",
            "View → Fit Camera (Home) — re-frame the whole scene",
            "View → Center (Numpad .) — translate every object so the scene "
            "centroid is at the origin",
            "Toolbar → Spin — toggles a slow ~20 s / turn orbit around the "
            "vertical (Z) axis. Click again to stop.",
        ]),
    ]),
    ("Snapshot / save", [
        ("bullets", [
            "F12 (Snapshot) — saves the current viewport to a PNG file",
            "Ctrl-S — save the active object back out as a PDB file",
            "Files → Open Docking lets you overlay a docked ligand pose; the "
            "Ligand is rendered as ball-and-stick and colored by docking rank",
        ]),
    ]),
    ("The Python console", [
        ("text", "The REPL at the bottom of the window exposes a `D` handle "
                  "(StudioAPI) and `np` (numpy) in its namespace — try "
                  "`D.scene.objects`, `D.schemes`, `D.representations`. Up/down "
                  "arrows browse history (persisted to ~/.protein-studio-history)."),
        ("text", "Examples:"),
        ("code", "D.load('examples/1aki.pdb', representation='ribbon')\n"
                  "D.refresh()\n"
                  "from protein_studio.core.hbonds import detect_hbonds\n"
                  "hb = detect_hbonds(D.scene.objects[0].molecule)\n"
                  "len(hb)"),
    ]),
    ("The Script editor", [
        ("text", "Switch to the **Script** tab at the bottom to write multi-line "
                  "Python scripts. The toolbar:"),
        ("bullets", [
            "New / Open / Save / Save As — standard file ops, with the current "
            "path shown on the right",
            "Run ▶ (F5) — execute the whole script in the shared namespace",
            "Run Sel — execute just the highlighted block",
            "Modified flag (*) appears in the path label while the script has "
            "unsaved changes",
        ]),
        ("text", "Example workflow — load, render, screenshot, save:"),
        ("code", "import os\n"
                  "out = '/tmp/work'\n"
                  "os.makedirs(out, exist_ok=True)\n"
                  "\n"
                  "p = D.load('examples/1aki.pdb',\n"
                  "           representation='ribbon', scheme='secondary')\n"
                  "D.refresh()\n"
                  "D.screenshot(os.path.join(out, '1aki.png'))\n"
                  "\n"
                  "from protein_studio.core.hbonds import detect_hbonds\n"
                  "hb = detect_hbonds(p.molecule)\n"
                  "print(f'{len(hb)} H-bonds')\n"
                  "\n"
                  "from protein_studio.core.io import save_pdb\n"
                  "save_pdb(p.molecule, os.path.join(out, '1aki.pdb'))"),
    ]),
    ("Tips & troubleshooting", [
        ("bullets", [
            "If colors look wrong after switching representations, press F5 "
            "or click Refresh — it forces a full mesh rebuild.",
            "Ball-and-stick atoms now use a display radius of ~0.3 Å (smaller "
            "than true van der Waals) so the bonds between them stay "
            "visible. Adjust in core/representations.py:ball_and_stick() if you "
            "want a different style.",
            "ESP surface defaults to range_kcal=0.1 — pass a higher value "
            "(e.g. range_kcal=1.0) for proteins with very strong fields, or a "
            "lower value (~0.05) for highly charged small molecules.",
            "The macOS dock icon is bundled as resources/icon.icns. On Linux "
            "the system uses icon_1024.png instead.",
        ]),
    ]),
    ("Where to get more help", [
        ("bullets", [
            "examples/ — bundled 1aki / 1ubq / 4hhb for quick experiments",
            "tests/ — pytest suite doubles as worked-example code",
            "3Dmol.js docs — https://3dmol.csb.pitt.edu — we mirror their "
            "representation semantics",
        ]),
    ]),
]


# ============================================================ shared helpers


def _img(path: str, width_in: float):
    return ("image", SCREENSHOTS / path, width_in)


# ============================================================ docx renderer


def _add_docx_heading(doc, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = _COLOR_HEADING


def _add_docx_paragraph(doc, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    for run in p.runs:
        run.font.color.rgb = _COLOR_BODY


def _add_docx_bullets(doc, items: Iterable[str]) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.color.rgb = _COLOR_BODY


def _add_docx_code(doc, code: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(code)
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    run.font.color.rgb = _COLOR_CODE


def _add_docx_image(doc, path: Path, width_in: float, caption: str = "") -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = c.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = _COLOR_MUTED


# ============================================================ pptx renderer


def _add_pptx_title(slide, title: str) -> None:
    """Add a title text box at the top of the slide (uses an explicit
    textbox instead of the layout's title placeholder, which doesn't
    exist on the blank layout we use)."""
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.4),
                                    Inches(9.0), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = _COLOR_HEADING


def _add_pptx_bullets(slide, items: List[str], left_in: float = 0.6,
                      top_in: float = 1.5, width_in: float = 8.8,
                      height_in: float = 5.0, font_size: int = 18) -> None:
    box = slide.shapes.add_textbox(Inches(left_in), Inches(top_in),
                                    Inches(width_in), Inches(height_in))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + item
        p.font.size = Pt(font_size)
        p.font.color.rgb = _COLOR_BODY
        p.space_after = Pt(6)


def _add_pptx_image(slide, path: Path, left_in: float = 1.0,
                    top_in: float = 1.6, width_in: float = 8.0,
                    height_in: float = 5.5, caption: str = "") -> None:
    slide.shapes.add_picture(str(path), Inches(left_in), Inches(top_in),
                             width=Inches(width_in), height=Inches(height_in))
    if caption:
        tx = slide.shapes.add_textbox(Inches(left_in),
                                       Inches(top_in + height_in + 0.05),
                                       Inches(width_in), Inches(0.4))
        p = tx.text_frame.paragraphs[0]
        p.text = caption
        p.font.size = Pt(12)
        p.font.italic = True
        p.font.color.rgb = _COLOR_MUTED
        p.alignment = PP_ALIGN.CENTER


# ============================================================ top-level renderers


def render_docx(guide: List[Tuple[str, List]], out_path: Path, title: str) -> None:
    """Render a structured guide into a .docx file."""
    global _COLOR_HEADING, _COLOR_BODY, _COLOR_MUTED, _COLOR_CODE
    from docx import Document

    _COLOR_HEADING = _RGBColorDocx(0x10, 0x35, 0x60)
    _COLOR_BODY    = _RGBColorDocx(0x20, 0x20, 0x20)
    _COLOR_MUTED   = _RGBColorDocx(0x70, 0x70, 0x70)
    _COLOR_CODE    = _RGBColorDocx(0x10, 0x40, 0x10)

    doc = Document()
    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title
    title_p = doc.add_paragraph()
    title_run = title_p.add_run(title)
    title_run.font.size = Pt(28)
    title_run.font.bold = True
    title_run.font.color.rgb = _COLOR_HEADING
    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run(f"LingSu Protein  ·  v{APP_VERSION}  ·  generated documentation")
    sub_run.font.size = Pt(10)
    sub_run.font.color.rgb = _COLOR_MUTED
    doc.add_paragraph()  # spacer

    for heading, blocks in guide:
        _add_docx_heading(doc, heading, level=1)
        for kind, *rest in blocks:
            if kind == "text":
                _add_docx_paragraph(doc, rest[0])
            elif kind == "bullets":
                _add_docx_bullets(doc, rest[0])
            elif kind == "code":
                _add_docx_code(doc, rest[0])
            elif kind == "image":
                path, width_in = rest[0], rest[1]
                if (SCREENSHOTS / path).exists():
                    _add_docx_image(doc, SCREENSHOTS / path, width_in)
                # else skip silently — image is optional

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"  wrote {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


def render_pptx(guide: List[Tuple[str, List]], out_path: Path, title: str) -> None:
    """Render a structured guide into a .pptx file.

    Each top-level (heading, blocks) becomes one or more slides:
    - if the section contains an image, that image gets its own slide
    - the rest of the blocks become a text slide
    - sections without images are a single text slide
    """
    global _COLOR_HEADING, _COLOR_BODY, _COLOR_MUTED, _COLOR_CODE
    from pptx import Presentation

    _COLOR_HEADING = _RGBColorPptx(0x10, 0x35, 0x60)
    _COLOR_BODY    = _RGBColorPptx(0x20, 0x20, 0x20)
    _COLOR_MUTED   = _RGBColorPptx(0x70, 0x70, 0x70)
    _COLOR_CODE    = _RGBColorPptx(0x10, 0x40, 0x10)

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]  # blank layout

    # Cover slide
    cover = prs.slides.add_slide(blank)
    tx = cover.shapes.add_textbox(Inches(0.5), Inches(2.0),
                                  Inches(9.0), Inches(2.5))
    tf = tx.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = _COLOR_HEADING
    p.alignment = PP_ALIGN.CENTER
    sub = tf.add_paragraph()
    sub.text = f"LingSu Protein  ·  v{APP_VERSION}"
    sub.font.size = Pt(20)
    sub.font.color.rgb = _COLOR_MUTED
    sub.alignment = PP_ALIGN.CENTER

    for heading, blocks in guide:
        # Split blocks into "image blocks" and "text blocks" so each
        # image gets its own slide.
        text_blocks = []
        image_block = None
        for kind, *rest in blocks:
            if kind == "image":
                image_block = (rest[0], rest[1])
            else:
                text_blocks.append((kind, *rest))

        # 1) Text slide with the section title + body content
        slide = prs.slides.add_slide(blank)
        _add_pptx_title(slide, heading)

        # Special handling: code-only sections get a wider monospace box
        if text_blocks and all(b[0] == "code" for b in text_blocks):
            box = slide.shapes.add_textbox(Inches(0.6), Inches(1.5),
                                           Inches(8.8), Inches(5.5))
            tf = box.text_frame
            tf.word_wrap = True
            for i, (kind, *rest) in enumerate(text_blocks):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = rest[0]
                p.font.name = "Consolas"
                p.font.size = Pt(14)
                p.font.color.rgb = _COLOR_CODE
                p.space_after = Pt(6)
        else:
            items = []
            for kind, *rest in text_blocks:
                if kind == "text":
                    items.append(rest[0])
                elif kind == "bullets":
                    items.extend(rest[0])
            _add_pptx_bullets(slide, items)

        # 2) Image slide if present
        if image_block is not None:
            img_path, width_in = image_block
            if (SCREENSHOTS / img_path).exists():
                img_slide = prs.slides.add_slide(blank)
                _add_pptx_title(img_slide, heading)
                _add_pptx_image(img_slide, SCREENSHOTS / img_path,
                                width_in=width_in)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"  wrote {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ============================================================ main


def main() -> int:
    print(f"Generating docs into {OUT}/")
    print("Developer guide:")
    render_docx(DEVELOPER_GUIDE, OUT / "DeveloperGuide.docx",
                title=f"{APP_NAME} Developer Guide")
    render_pptx(DEVELOPER_GUIDE, OUT / "DeveloperGuide.pptx",
                title=f"{APP_NAME} Developer Guide")
    print("User guide:")
    render_docx(USER_GUIDE, OUT / "UserGuide.docx",
                title=f"{APP_NAME} User Guide")
    render_pptx(USER_GUIDE, OUT / "UserGuide.pptx",
                title=f"{APP_NAME} User Guide")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())