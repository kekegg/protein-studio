"""Application-wide icon + name resolution.

Centralizes the lookup of the bundled ``icon.icns`` / ``icon.png`` so
both the window (set via :class:`QMainWindow.setWindowIcon`) and the
macOS dock (set via :class:`QApplication.setWindowIcon`) pick up the
same asset.

The macOS dock uses the application icon set via
``QApplication.setWindowIcon`` *only* when the .app bundle's
``Info.plist`` does not provide a ``CFBundleIconFile`` — when the user
runs ``python -m protein_studio.app`` directly (no .app bundle), this
is the only path that puts a custom icon on the dock. When packaged
inside a .app bundle, macOS will use the bundle icon first.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon


def _resources_dir() -> Path:
    """Absolute path to the bundled ``resources/`` directory."""
    return Path(__file__).resolve().parent.parent.parent / "resources"


def load_app_icon() -> QIcon:
    """Return a :class:`QIcon` containing every size of the bundled icon.

    Picks the platform-preferred master first (``icon.icns`` on
    macOS, ``icon.ico`` on Windows, the largest PNG elsewhere) so
    PySide6 picks the right variant per-DPI.
    """
    rdir = _resources_dir()
    icon = QIcon()
    # Order matters: PySide6 prefers the first format that the
    # underlying QImageReader can decode. On macOS, ``icon.icns`` is
    # best; on Windows, ``icon.ico``; on Linux, the master PNG.
    for candidate in (
        rdir / "icon.icns",
        rdir / "icon.ico",
        rdir / "icon_1024.png",
    ):
        if candidate.exists():
            icon.addFile(str(candidate))
    # Add the smaller PNGs too — Qt will pick the closest match for
    # the target DPI/size when the OS gives it a generic bitmap.
    for s in (16, 24, 32, 48, 64, 128, 256, 512):
        p = rdir / f"icon_{s}.png"
        if p.exists():
            icon.addFile(str(p))
    return icon


def resources_dir() -> Path:
    """Public accessor for the bundled resources directory."""
    return _resources_dir()