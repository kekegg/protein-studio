#!/usr/bin/env node
/**
 * Generate developer and user documentation (docx + pptx).
 *
 * Run with::
 *
 *     npm run build         # from docs/
 *     node generate_docs.mjs
 *
 * Outputs are written next to this script (docs/):
 *
 *   - DeveloperGuide.docx / DeveloperGuide.pptx
 *   - UserGuide.docx       / UserGuide.pptx
 *
 * Content is defined in two plain-JS data structures
 * (``DEVELOPER_GUIDE`` and ``USER_GUIDE``) so the .docx and .pptx
 * renderers stay in sync — change the content once, both files update.
 *
 * Headings / body are styled with a custom palette matching the
 * LingSu Protein app's dark theme.
 */

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

import {
    Document, Paragraph, TextRun, ImageRun,
    AlignmentType, BorderStyle, HeadingLevel, Packer,
} from 'docx';
import PptxGenJS from 'pptxgenjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const HERE        = __dirname;
const SCREENSHOTS = path.join(HERE, 'screenshots');
const OUT         = HERE;

const APP_NAME    = 'LingSu Protein';
const APP_VERSION = '0.1.0';

// ============================================================ content data
//
// Each guide is an array of [heading, blocks[]] pairs.
// A block is one of:
//   { kind: 'text',    text: string }
//   { kind: 'bullets', items: string[] }
//   { kind: 'code',    code: string }
//   { kind: 'image',   file: string, widthIn: number }
//
// Screenshots referenced by `image` blocks live in docs/screenshots/.

const DEVELOPER_GUIDE = [
    ['Overview', [
        { kind: 'text', text:
            `${APP_NAME} is a Blender-like molecular visualization desktop `
            + `application. It pairs a PySide6 GUI with PyVista/VTK for 3D `
            + `rendering, scikit-image for marching cubes, scipy for the `
            + `Euclidean distance transform, and the AMBER ff99SB-style `
            + `partial-charge table ported from 3Dmol.js.` },
        { kind: 'text', text:
            'This guide is for contributors — it covers the module layout, '
            + 'key design decisions, and how to add new representations, '
            + 'color schemes, or extensions.' },
    ]],
    ['Repository layout', [
        { kind: 'bullets', items: [
            'src/protein_studio/ — package source',
            '    api.py        StudioAPI — load/refresh/control surface for the GUI',
            '    app.py        application entry point (CLI flags + QApplication)',
            '    resources.py  icon loader (icon.icns → QIcon)',
            '    mcp_server.py stdio MCP server (20 tools) over StudioAPI',
            '    core/         data model + rendering',
            '        molecule.py    Atom / Residue / Chain / Molecule dataclasses',
            '        io.py          PDB parser',
            '        scene.py       Scene + SceneObject, add/remove/update_object',
            '        colors.py      ColorScheme + SCHEME_PRESETS',
            '        representations.py  7 reps (cartoon / ribbon / ball_and_stick / '
            + '                                 surface / points / esp_surface / hbonds)',
            '        surface.py     EDTSurf-style SES surface generator',
            '        marching_cubes.py  skimage wrapper',
            '        charges.py     AMBER partial-charge table + ESP coloring',
            '        hbonds.py      Baker-Hubbard H-bond detection',
            '        trajectory.py  MDAnalysis wrapper',
            '        alphafold.py   pLDDT color scheme',
            '        docking.py     Vina PDBQT + EquiScore loaders',
            '    ui/          PySide6 widgets',
            '        viewport.py     3D viewport (PyVistaQt + macOS OpenGL workarounds)',
            '        outliner.py     left-side scene tree',
            '        properties.py   right-side inspector',
            '        console.py      REPL Python console',
            '        editor.py       file-editor + script runner',
            '        mcp_controller.py  QProcess wrapper that starts/stops the MCP server',
            '        main_window.py  4-pane Blender-like layout',
            'tests/        pytest suite',
            'examples/     sample PDBs (1aki.pdb, 1ubq.pdb, 4hhb.pdb)',
            'resources/    bundled icon (icon.icns / icon.ico / icon_*.png)',
            'docs/         this documentation',
        ]},
    ]],
    ['Quick start — running from source', [
        { kind: 'code', code:
            'conda activate protein-studio\n'
          + 'python -m protein_studio.app examples/1aki.pdb' },
        { kind: 'text', text:
            'The above opens the GUI and auto-loads 1aki as a ribbon. '
            + 'Pass additional PDB files as positional args to load them on '
            + 'startup. Pass --headless to do a one-shot scripted render '
            + '(--output render.png, --representation cartoon, --scheme secondary).' },
    ]],
    ['Running tests', [
        { kind: 'code', code:
            'conda activate protein-studio\n'
          + 'python -m pytest tests/ -v' },
        { kind: 'text', text:
            'The suite covers core molecule/scene ops, all 7 representations, '
            + 'the AMBER charge table, ESP surface, the bundled Python editor, '
            + 'MCP controller, and the resources module (icon + app name).' },
    ]],
    ['MCP server', [
        { kind: 'text', text:
            'A stdio MCP server lives in src/protein_studio/mcp_server.py and '
            + 'exposes StudioAPI as 20 tools (load_pdb, screenshot, run_python, '
            + 'detect_hbonds, compute_esp, etc.). Agents such as Claude '
            + 'Desktop, Cline, and Cherry Studio can drive the program through '
            + 'this server.' },
        { kind: 'text', text:
            'The GUI ships with a controller (ui/mcp_controller.py) that '
            + 'wraps QProcess so users can start / stop the server from the '
            + 'Tools menu, the toolbar MCP pill, or the status bar. '
            + 'closeEvent() terminates the subprocess when the window quits.' },
    ]],
    ['Design decisions worth knowing', [
        { kind: 'text', text: '1. EDTSurf-style SES algorithm' },
        { kind: 'text', text:
            'We port 3Dmol.js\'s ProteinSurface4 to Python: voxelize atoms → '
            + 'scipy.ndimage.distance_transform_edt → skimage.measure.marching_cubes. '
            + 'Defaults to scale_factor=2 (0.5 Å voxels). Three surface types: '
            + 'VDW (1), SAS (3), MS/SES (2). See core/surface.py.' },
        { kind: 'text', text: '2. AMBER partial-charge table' },
        { kind: 'text', text:
            'Full AMBER ff99SB-style lookup per (residue, atom) for the 20 '
            + 'standard amino acids, ported from 3Dmol.js\'s partialCharges.ts. '
            + 'Falls back to element-wise heuristic when atoms aren\'t in the '
            + 'table. See core/charges.py.' },
        { kind: 'text', text: '3. Per-SS ribbon cross-section' },
        { kind: 'text', text:
            'ribbon() picks a per-sample (width, thickness) ellipse in the '
            + '(side, binormal) plane depending on SS: helix becomes a full circle '
            + '(1.4 Å radius, looks like a real cylinder under Phong), loop is a '
            + 'thin tube, sheet is a flat arrow with a 2× widening at the C-terminal '
            + 'end. Smoothed along the chain with a Gaussian kernel so helix→loop '
            + 'boundaries don\'t pop.' },
        { kind: 'text', text: '4. macOS OpenGL mapper workaround' },
        { kind: 'text', text:
            'PyVistaQt on macOS corrupts actor color state when many actors '
            + 'are added. _add_object_to_plotter installs a fresh vtkDataSetMapper '
            + 'per actor and calls SetColorModeToDirectScalars() for 3-component '
            + 'RGB arrays (so ESP / ball-and-stick colors reach the shader '
            + 'verbatim). See _add_object_to_plotter in ui/viewport.py.' },
        { kind: 'text', text: '5. Scene.add() deduplicates by name' },
        { kind: 'text', text:
            'Loading the same PDB twice replaces the existing object rather '
            + 'than appending a second one (which would cause actor-name '
            + 'collisions in the plotter and show the old version of the protein).' },
    ]],
    ['Adding a new representation', [
        { kind: 'text', text:
            'Implement a function in core/representations.py with the '
            + 'signature (mol, scheme, **kwargs) -> pyvista.PolyData (or a '
            + 'dict {"atoms": PolyData, "bonds": PolyData}). Register it in '
            + 'the REPRESENTATIONS dict. Use build_mesh() (in the same file) '
            + 'to get the canonical mesh-build pipeline.' },
        { kind: 'text', text:
            'If your rep needs per-vertex atom_id or other metadata, '
            + 'expose it as point_data so downstream properties (coloring, '
            + 'ESP, H-bond overlay) can pick it up.' },
    ]],
    ['Adding a new color scheme', [
        { kind: 'text', text:
            'Add an entry to SCHEME_PRESETS in core/colors.py with a unique '
            + 'key (e.g. "b_factor"). The ColorScheme.color() method takes an '
            + 'atom and returns an (r, g, b) tuple in [0, 1]. For schemes that '
            + 'depend on per-vertex data (like electrostatic), handle the '
            + 'coloring inside the rep function and have the scheme return '
            + 'uniform — the viewport will pick up the rep\'s per-vertex '
            + 'potential data via point_data[\'potential\'].' },
    ]],
    ['Adding an extension (H-bonds / trajectory / AlphaFold / docking)', [
        { kind: 'text', text:
            'Each extension lives in core/<name>.py and exposes top-level '
            + 'functions plus dataclasses. Add a method to StudioAPI '
            + '(src/protein_studio/api.py) that wires the extension into the '
            + 'scene. Wire a toolbar button in ui/main_window.py to call the '
            + 'API method via a QFileDialog. Each extension has its own tests '
            + 'in tests/test_extensions.py.' },
    ]],
    ['Build / packaging', [
        { kind: 'text', text:
            'Currently shipped as a conda env (protein-studio) launched '
            + 'via `python -m protein_studio.app`. A PyInstaller spec for a '
            + 'double-clickable .app / .exe would be a natural next step '
            + '(the resources/ icon is already in place for that).' },
    ]],
];


const USER_GUIDE = [
    ['Welcome to LingSu Protein', [
        { kind: 'text', text:
            `${APP_NAME} is a desktop molecular viewer built around the same `
            + `representations you'd find in PyMOL and 3Dmol.js — cartoon, `
            + `ribbon, ball-and-stick, molecular surface, ESP surface, `
            + `ball-and-stick H-bonds, point cloud, AlphaFold pLDDT coloring.` },
        { kind: 'text', text:
            'On first launch the app opens the bundled example 1aki.pdb as '
            + 'a ribbon so you have something to look at immediately. Use '
            + 'File → Clear Scene (Ctrl-Shift-C) any time to start fresh.' },
    ]],
    ['GUI overview', [
        { kind: 'text', text:
            'The window is split into four Blender-style panels:' },
        { kind: 'bullets', items: [
            'Outliner (left)    — scene tree listing every loaded object',
            'Viewport (center)  — the 3D scene',
            'Properties (right) — selected object\'s representation + color scheme',
            'Console / Script (bottom) — REPL (>>> ) and file editor with Run button',
        ]},
    ]],
    ['Loading structures', [
        { kind: 'bullets', items: [
            'File → Open PDB… (Ctrl-O) — pick a .pdb file',
            'File → Open Docking… (Ctrl-D) — Vina .pdbqt or an EquiScore directory',
            'File → Open Trajectory… (Ctrl-T) — topology .pdb + trajectory .dcd/.xtc',
            'File → Open AlphaFold… (Ctrl-Shift-A) — load a predicted structure '
            + 'with pLDDT scores',
            'Drop the file into the viewport (drag-and-drop is in scope for the '
            + 'next iteration)',
        ]},
        { kind: 'text', text:
            'Loading the same PDB twice replaces the existing scene object '
            + '(so you can edit the representation without leaving ghost '
            + 'copies).' },
    ]],
    ['Changing representation & color scheme', [
        { kind: 'text', text:
            'Click any object in the Outliner (or in the viewport) to '
            + 'select it. The Properties panel on the right shows:' },
        { kind: 'bullets', items: [
            'Representation: cartoon | ribbon | ball_and_stick | surface | '
            + 'points | esp_surface | hbonds',
            'Color scheme:    uniform | element | residue | chain | secondary | '
            + 'b-factor | electrostatic | alphafold_plddt (when available)',
            'Visible in viewport: toggle the object\'s visibility',
            'Position / Rotation / Scale: numeric transform',
        ]},
        { kind: 'text', text:
            'Tip: with surface or esp_surface reps, pick the '
            + '"Electrostatic" scheme to get the red-white-blue ESP '
            + 'coloring. Otherwise the surface is colored by the chosen '
            + 'scheme (element / secondary / etc.).' },
    ]],
    ['View controls', [
        { kind: 'bullets', items: [
            'Left-drag — orbit the camera',
            'Right-drag (or Shift-drag) — pan',
            'Scroll wheel — zoom (PyVista default: dolly the camera along its '
            + 'view direction)',
            'View → Fit Camera (Home) — re-frame the whole scene',
            'View → Center (Numpad .) — translate every object so the scene '
            + 'centroid is at the origin',
            'Toolbar → Spin — toggles a slow ~20 s / turn orbit around the '
            + 'vertical (Z) axis. Click again to stop.',
        ]},
    ]],
    ['Snapshot / save', [
        { kind: 'bullets', items: [
            'F12 (Snapshot) — saves the current viewport to a PNG file',
            'Ctrl-S — save the active object back out as a PDB file',
            'Files → Open Docking lets you overlay a docked ligand pose; the '
            + 'Ligand is rendered as ball-and-stick and colored by docking rank',
        ]},
    ]],
    ['MCP server (driven by an AI agent)', [
        { kind: 'text', text:
            'Tools → Start MCP Server (or click the toolbar MCP pill) starts '
            + 'a stdio subprocess that exposes the entire application as 20 '
            + 'MCP tools. Tools → Stop MCP Server (or close the window) '
            + 'terminates it cleanly.' },
        { kind: 'text', text:
            'When the server is running, an MCP client such as Claude '
            + 'Desktop, Cline, or Cherry Studio can drive the viewer: load '
            + 'structures, change representations and color schemes, take '
            + 'screenshots, run Python in the app namespace (load_pdb, '
            + 'set_representation, screenshot, run_python, detect_hbonds, '
            + 'compute_esp, get_info, rotate, ...). The status-bar pill '
            + 'turns green while the server is running.' },
    ]],
    ['The Python console', [
        { kind: 'text', text:
            'The REPL at the bottom of the window exposes a `D` handle '
            + '(StudioAPI) and `np` (numpy) in its namespace — try '
            + '`D.scene.objects`, `D.schemes`, `D.representations`. Up/down '
            + 'arrows browse history (persisted to ~/.protein-studio-history).' },
        { kind: 'text', text: 'Examples:' },
        { kind: 'code', code:
            "D.load('examples/1aki.pdb', representation='ribbon')\n"
          + 'D.refresh()\n'
          + 'from protein_studio.core.hbonds import detect_hbonds\n'
          + 'hb = detect_hbonds(D.scene.objects[0].molecule)\n'
          + 'len(hb)' },
    ]],
    ['The Script editor', [
        { kind: 'text', text:
            'Switch to the **Script** tab at the bottom to write multi-line '
            + 'Python scripts. The toolbar:' },
        { kind: 'bullets', items: [
            'New / Open / Save / Save As — standard file ops, with the current '
            + 'path shown on the right',
            'Run ▶ (F5) — execute the whole script in the shared namespace',
            'Run Sel — execute just the highlighted block',
            'Modified flag (*) appears in the path label while the script has '
            + 'unsaved changes',
        ]},
        { kind: 'text', text:
            'Example workflow — load, render, screenshot, save:' },
        { kind: 'code', code:
            'import os\n'
          + "out = '/tmp/work'\n"
          + 'os.makedirs(out, exist_ok=True)\n'
          + '\n'
          + "p = D.load('examples/1aki.pdb',\n"
          + "           representation='ribbon', scheme='secondary')\n"
          + 'D.refresh()\n'
          + 'D.screenshot(os.path.join(out, \'1aki.png\'))\n'
          + '\n'
          + 'from protein_studio.core.hbonds import detect_hbonds\n'
          + 'hb = detect_hbonds(p.molecule)\n'
          + 'print(f\'{len(hb)} H-bonds\')\n'
          + '\n'
          + 'from protein_studio.core.io import save_pdb\n'
          + "save_pdb(p.molecule, os.path.join(out, '1aki.pdb'))" },
    ]],
    ['Tips & troubleshooting', [
        { kind: 'bullets', items: [
            'If colors look wrong after switching representations, press F5 '
            + 'or click Refresh — it forces a full mesh rebuild.',
            'Ball-and-stick atoms now use a display radius of ~0.3 Å (smaller '
            + 'than true van der Waals) so the bonds between them stay '
            + 'visible. Adjust in core/representations.py:ball_and_stick() if you '
            + 'want a different style.',
            'ESP surface defaults to range_kcal=0.1 — pass a higher value '
            + '(e.g. range_kcal=1.0) for proteins with very strong fields, or a '
            + 'lower value (~0.05) for highly charged small molecules.',
            'The macOS dock icon is bundled as resources/icon.icns. On Linux '
            + 'the system uses icon_1024.png instead.',
            'The MCP server needs a Python interpreter with the protein-studio '
            + 'env active. Run `which python` and check that the GUI and the '
            + 'server are launched from the same env.',
        ]},
    ]],
    ['Where to get more help', [
        { kind: 'bullets', items: [
            'examples/ — bundled 1aki / 1ubq / 4hhb for quick experiments',
            'tests/ — pytest suite doubles as worked-example code',
            '3Dmol.js docs — https://3dmol.csb.pitt.edu — we mirror their '
            + 'representation semantics',
        ]},
    ]],
];

// ============================================================ shared palette

const COLOR_HEADING = '103560';  // navy blue
const COLOR_BODY    = '202020';
const COLOR_MUTED   = '707070';
const COLOR_CODE    = '104010';

const FONT_BODY    = 'Calibri';
const FONT_MONO    = 'Consolas';
const FONT_HEADING = 'Calibri';

const PT  = (n) => n * 1;     // docx uses half-points but we wrap later
const IN  = (n) => n * 1440;  // pptxgenjs uses inches

function safeImagePath(rel) {
    const p = path.join(SCREENSHOTS, rel);
    return fs.existsSync(p) ? p : null;
}

// ============================================================ docx renderer

function buildDocx(guide, title) {
    const children = [];

    // Title block
    children.push(new Paragraph({
        alignment: AlignmentType.LEFT,
        spacing: { after: 80 },
        children: [new TextRun({
            text: title,
            bold: true, size: 56, font: FONT_HEADING, color: COLOR_HEADING,
        })],
    }));
    children.push(new Paragraph({
        spacing: { after: 240 },
        children: [new TextRun({
            text: `${APP_NAME}  ·  v${APP_VERSION}  ·  generated documentation`,
            size: 20, font: FONT_BODY, color: COLOR_MUTED, italics: true,
        })],
    }));
    children.push(new Paragraph({ children: [new TextRun({ text: '' })] }));

    for (const [heading, blocks] of guide) {
        children.push(new Paragraph({
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
            children: [new TextRun({
                text: heading,
                bold: true, size: 32, font: FONT_HEADING, color: COLOR_HEADING,
            })],
        }));

        for (const block of blocks) {
            if (block.kind === 'text') {
                children.push(new Paragraph({
                    spacing: { after: 120 },
                    children: [new TextRun({
                        text: block.text,
                        size: 22, font: FONT_BODY, color: COLOR_BODY,
                    })],
                }));
            } else if (block.kind === 'bullets') {
                for (const item of block.items) {
                    children.push(new Paragraph({
                        bullet: { level: 0 },
                        spacing: { after: 60 },
                        children: [new TextRun({
                            text: item,
                            size: 22, font: FONT_BODY, color: COLOR_BODY,
                        })],
                    }));
                }
            } else if (block.kind === 'code') {
                children.push(new Paragraph({
                    spacing: { before: 80, after: 160 },
                    indent: { left: 360 },
                    shading: { fill: 'F4F4F4', type: 'clear' },
                    border: {
                        top:    { style: BorderStyle.SINGLE, size: 2, color: 'D0D0D0' },
                        bottom: { style: BorderStyle.SINGLE, size: 2, color: 'D0D0D0' },
                        left:   { style: BorderStyle.SINGLE, size: 2, color: 'D0D0D0' },
                        right:  { style: BorderStyle.SINGLE, size: 2, color: 'D0D0D0' },
                    },
                    children: [new TextRun({
                        text: block.code,
                        size: 20, font: FONT_MONO, color: COLOR_CODE,
                    })],
                }));
            } else if (block.kind === 'image') {
                const imgPath = safeImagePath(block.file);
                if (!imgPath) continue;  // silently skip missing
                children.push(new Paragraph({
                    alignment: AlignmentType.CENTER,
                    spacing: { before: 80, after: 40 },
                    children: [new ImageRun({
                        data: fs.readFileSync(imgPath),
                        transformation: { width: block.widthIn * 96, height: 0 },
                    })],
                }));
            }
        }
    }

    return new Document({
        creator: APP_NAME,
        title,
        styles: {
            default: {
                document: { run: { font: FONT_BODY, size: 22 } },
            },
        },
        sections: [{
            properties: {},
            children,
        }],
    });
}

// ============================================================ pptx renderer

async function buildPptx(guide, title, outPath) {
    const pres = new PptxGenJS();
    pres.layout = 'LAYOUT_4x3';        // 10 × 7.5 in
    pres.title  = title;
    pres.author = APP_NAME;

    // Cover slide
    const cover = pres.addSlide();
    cover.background = { color: 'FFFFFF' };
    cover.addText(title, {
        x: 0.5, y: 2.0, w: 9.0, h: 1.2,
        fontFace: FONT_HEADING, fontSize: 40, bold: true,
        color: COLOR_HEADING, align: 'center',
    });
    cover.addText(`${APP_NAME}  ·  v${APP_VERSION}`, {
        x: 0.5, y: 3.4, w: 9.0, h: 0.6,
        fontFace: FONT_BODY, fontSize: 18,
        color: COLOR_MUTED, align: 'center', italic: true,
    });

    for (const [heading, blocks] of guide) {
        // Split into text blocks + an optional single image block.
        const textBlocks = [];
        let imageBlock = null;
        for (const block of blocks) {
            if (block.kind === 'image') {
                imageBlock = block;
            } else {
                textBlocks.push(block);
            }
        }

        // Text slide
        const textSlide = pres.addSlide();
        textSlide.background = { color: 'FFFFFF' };
        textSlide.addText(heading, {
            x: 0.5, y: 0.4, w: 9.0, h: 0.9,
            fontFace: FONT_HEADING, fontSize: 28, bold: true,
            color: COLOR_HEADING, align: 'left',
        });

        if (textBlocks.length > 0 &&
            textBlocks.every((b) => b.kind === 'code')) {
            // Render as one big monospace block
            const code = textBlocks.map((b) => b.code).join('\n\n');
            textSlide.addText(code, {
                x: 0.6, y: 1.5, w: 8.8, h: 5.5,
                fontFace: FONT_MONO, fontSize: 14,
                color: COLOR_CODE, align: 'left', valign: 'top',
            });
        } else {
            const lines = [];
            for (const b of textBlocks) {
                if (b.kind === 'text') {
                    lines.push({ text: b.text, options: { bullet: false } });
                } else if (b.kind === 'bullets') {
                    for (const item of b.items) {
                        lines.push({ text: item, options: { bullet: true } });
                    }
                }
            }
            textSlide.addText(
                lines.map((l) => ({
                    text: l.text,
                    options: {
                        bullet: l.options.bullet,
                        fontFace: FONT_BODY, fontSize: 18,
                        color: COLOR_BODY, paraSpaceAfter: 6,
                    },
                })),
                { x: 0.6, y: 1.5, w: 8.8, h: 5.5,
                  valign: 'top', align: 'left' },
            );
        }

        // Optional image slide
        if (imageBlock) {
            const imgPath = safeImagePath(imageBlock.file);
            if (imgPath) {
                const imgSlide = pres.addSlide();
                imgSlide.background = { color: 'FFFFFF' };
                imgSlide.addText(heading, {
                    x: 0.5, y: 0.4, w: 9.0, h: 0.9,
                    fontFace: FONT_HEADING, fontSize: 28, bold: true,
                    color: COLOR_HEADING, align: 'left',
                });
                imgSlide.addImage({
                    path: imgPath,
                    x: 1.0, y: 1.6, w: imageBlock.widthIn,
                    h: imageBlock.widthIn * 0.65,
                });
            }
        }
    }

    await pres.writeFile({ fileName: outPath });
}

// ============================================================ top-level

async function main() {
    console.log(`Generating docs into ${OUT}/`);

    const tasks = [
        { kind: 'docx', guide: DEVELOPER_GUIDE, title:
              `${APP_NAME} Developer Guide`, out: 'DeveloperGuide.docx' },
        { kind: 'pptx', guide: DEVELOPER_GUIDE, title:
              `${APP_NAME} Developer Guide`, out: 'DeveloperGuide.pptx' },
        { kind: 'docx', guide: USER_GUIDE, title:
              `${APP_NAME} User Guide`,       out: 'UserGuide.docx' },
        { kind: 'pptx', guide: USER_GUIDE, title:
              `${APP_NAME} User Guide`,       out: 'UserGuide.pptx' },
    ];

    console.log('Developer guide:');
    for (const t of tasks.filter((x) => x.out.startsWith('Developer'))) {
        await renderOne(t);
    }
    console.log('User guide:');
    for (const t of tasks.filter((x) => x.out.startsWith('User'))) {
        await renderOne(t);
    }
    console.log('Done.');
}

async function renderOne({ kind, guide, title, out }) {
    const outPath = path.join(OUT, out);
    if (kind === 'docx') {
        const doc = buildDocx(guide, title);
        const buf = await Packer.toBuffer(doc);
        fs.writeFileSync(outPath, buf);
    } else {
        await buildPptx(guide, title, outPath);
    }
    const sizeKb = Math.round(fs.statSync(outPath).size / 1024);
    console.log(`  wrote ${out}  (${sizeKb} KB)`);
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});