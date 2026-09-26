"""Tests for the bundled app icon + application name.

The icon assets live under ``resources/`` (built by
``resources/generate_icon.py``). These tests verify they're wired up
correctly so the dock icon and window title match the brand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_resources_dir_exists():
    from protein_studio.resources import resources_dir
    rdir = resources_dir()
    assert rdir.exists(), f"resources dir missing: {rdir}"
    assert (rdir / "icon.icns").exists(), "icon.icns missing — run resources/generate_icon.py"
    assert (rdir / "icon_1024.png").exists(), "icon_1024.png missing"


def test_load_app_icon_returns_non_null():
    """The QIcon returned by load_app_icon must be valid and contain
    at least one master size."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from protein_studio.resources import load_app_icon
    icon = load_app_icon()
    assert not icon.isNull(), "loaded icon is null"
    sizes = sorted({s.width() for s in icon.availableSizes()})
    assert 1024 in sizes, f"1024x1024 PNG missing from icon; got sizes {sizes}"
    assert 64 in sizes, f"64x64 PNG missing from icon; got sizes {sizes}"


def test_app_name_is_lingsu_protein():
    """The QApplication name + display name + window title must all be
    ``"LingSu Protein"`` so macOS dock, window title and about box
    stay consistent."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from protein_studio.ui.main_window import APP_NAME
    assert APP_NAME == "LingSu Protein", (
        f"APP_NAME must be 'LingSu Protein', got {APP_NAME!r}"
    )