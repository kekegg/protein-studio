"""Application entry point.

Run with:

    protein-studio                    # launch with empty scene
    protein-studio path/to/file.pdb   # load a PDB on launch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="protein-studio",
                                     description="Blender-like molecular viewer")
    parser.add_argument("pdb", nargs="*", help="Optional PDB file(s) to load")
    parser.add_argument("--headless", action="store_true",
                        help="Run a one-shot scripted render (no GUI)")
    parser.add_argument("--output", default="render.png",
                        help="Headless output PNG path")
    parser.add_argument("--representation", default="cartoon",
                        help="Default representation for headless mode")
    parser.add_argument("--scheme", default="secondary",
                        help="Default color scheme for headless mode")
    args, _ = parser.parse_known_args(argv)

    # Late imports (so --help is fast)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt

    from .api import StudioAPI, HeadlessViewport
    from .core.scene import Scene
    from .resources import load_app_icon
    from .ui.viewport import ViewportWidget
    from .ui.outliner import OutlinerWidget
    from .ui.properties import PropertiesWidget
    from .ui.console import PythonConsole
    from .ui.editor import PythonEditor
    from .ui.main_window import MainWindow

    # Dark palette (Blender-like)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LingSu Protein")
    app.setApplicationDisplayName("LingSu Protein")
    # The window icon doubles as the macOS dock icon when running
    # without an .app bundle (which is the common case during
    # development). When packaged inside an .app bundle, the
    # CFBundleIconFile takes precedence.
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(29, 29, 29))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Link, QColor(127, 177, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(59, 118, 194))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Wire scene + UI
    scene = Scene()
    # Headless or GUI viewport
    headless = args.headless or "PROTEIN_STUDIO_HEADLESS" in os.environ
    if headless:
        viewport_impl = HeadlessViewport()
    else:
        viewport_impl = None  # use the GUI widget below

    api = StudioAPI(viewport=viewport_impl)
    api.scene = scene  # share the same Scene instance
    viewport_widget = ViewportWidget(scene) if not headless else None
    outliner  = OutlinerWidget(scene)
    properties = PropertiesWidget(scene)
    console  = PythonConsole(namespace={"D": api, "np": __import__("numpy")})

    if headless:
        # Load files, render, exit
        for p in args.pdb or []:
            api.load(p, representation=args.representation, scheme=args.scheme)
        api.refresh()
        api.screenshot(args.output)
        return 0

    main_window = MainWindow(scene, viewport_widget, outliner, properties,
                             console, PythonEditor({"D": api, "np": __import__("numpy")}),
                             api)
    main_window.show()

    # Open any PDB files passed on the command line
    pdb_paths = list(args.pdb or [])
    # If none passed, auto-load examples/1aki.pdb so the user has
    # something to look at immediately on first launch (easy to clear
    # via Edit → Clear Scene).
    if not pdb_paths:
        default_pdb = Path(__file__).parent.parent.parent / "examples" / "1aki.pdb"
        if default_pdb.exists():
            pdb_paths.append(default_pdb)
    for p in pdb_paths:
        if Path(p).exists():
            api.load(p, representation="ribbon", scheme="secondary")

    # Initial render
    main_window.refresh_viewport()

    # Welcome to the console
    console.write_namespace("D", api)
    console.exec_line('print(f"Loaded {D.scene.n_objects()} objects, {D.scene.n_atoms()} atoms")')

    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())