"""Tests for the embedded Python script editor."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QTextCursor

from protein_studio.api import StudioAPI, HeadlessViewport
from protein_studio.core.scene import Scene
from protein_studio.ui.editor import PythonEditor


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def editor_setup(qapp):
    """A bare editor with a HeadlessViewport API so we can call D.*."""
    api = StudioAPI(viewport=HeadlessViewport())
    api.scene = Scene()
    ns = {"D": api, "np": __import__("numpy")}
    ed = PythonEditor(ns)
    ed.show()
    yield ed, api
    ed.close()


def test_editor_executes_simple_print(editor_setup):
    ed, _ = editor_setup
    ed._editor.setPlainText("print('hello from editor')")
    ed.run_script()
    out = ed._output.toPlainText()
    assert "hello from editor" in out
    assert "--- Done ---" in out


def test_editor_captures_traceback_on_error(editor_setup):
    ed, _ = editor_setup
    ed._editor.setPlainText("raise ValueError('boom')")
    ed.run_script()
    out = ed._output.toPlainText()
    assert "ValueError: boom" in out
    assert "Traceback" in out


def test_editor_namespace_persists_between_runs(editor_setup):
    ed, _ = editor_setup
    ed._editor.setPlainText("shared = 42")
    ed.run_script()
    ed._editor.setPlainText("print(shared)")
    ed.run_script()
    out = ed._output.toPlainText()
    assert "42" in out


def test_editor_can_drive_api(editor_setup):
    """A real protein-studio workflow: load a PDB, refresh, screenshot,
    detect H-bonds, save PDB — all from a script."""
    ed, _ = editor_setup
    script = """
import os
from protein_studio.core.io import save_pdb
from protein_studio.core.hbonds import detect_hbonds
out_dir = '/tmp/protein-studio-test-editor'
os.makedirs(out_dir, exist_ok=True)
protein = D.load('examples/1aki.pdb', representation='ribbon', scheme='secondary')
D.refresh()
D.screenshot(os.path.join(out_dir, '1aki.png'))
hbonds = detect_hbonds(protein.molecule)
save_pdb(protein.molecule, os.path.join(out_dir, '1aki.pdb'))
print(f'OK hbonds={len(hbonds)} screenshot=1aki.png pdb=1aki.pdb')
"""
    ed._editor.setPlainText(script)
    ed.run_script()
    out = ed._output.toPlainText()
    assert "OK hbonds=" in out
    # The script's artifacts should be on disk.
    assert os.path.exists("/tmp/protein-studio-test-editor/1aki.png")
    assert os.path.exists("/tmp/protein-studio-test-editor/1aki.pdb")


def test_editor_save_and_load_file(editor_setup):
    ed, _ = editor_setup
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.py"
        body = "x = 1 + 2\nprint('x =', x)\n"
        ed._editor.setPlainText(body)
        ed._current_path = path  # pretend user picked this file
        ed.save_file()
        assert path.exists()
        assert path.read_text() == body
        # Reload into a fresh editor and run.
        ed2 = PythonEditor({"D": StudioAPI(viewport=HeadlessViewport())})
        ed2.load_file(path)
        ed2.run_script()
        assert "x = 3" in ed2._output.toPlainText()


def test_editor_run_selection(editor_setup):
    ed, _ = editor_setup
    ed._editor.setPlainText("foo = 1\nbar = 2\nprint(foo + bar)")
    cursor = ed._editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)  # select all
    ed._editor.setTextCursor(cursor)
    ed.run_selection()
    out = ed._output.toPlainText()
    assert "3" in out


def test_editor_dirty_flag_clears_on_save(editor_setup):
    """The path label shows an asterisk while the document is modified."""
    ed, _ = editor_setup
    ed._editor.setPlainText("print(1)")
    assert "*" in ed._path_label.text() or "unsaved" in ed._path_label.text()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved.py"
        ed._current_path = path
        ed.save_file()
    assert "*" not in ed._path_label.text()