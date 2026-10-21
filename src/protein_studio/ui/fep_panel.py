"""FEP panel — dock widget with Setup and Run tabs.

Minimum-runnable implementation: the user picks a PDB, a mode
(mutation / ligand_decouple), a target residue / ligand, and a
recipe preset, then clicks "Run sweep". The sweep runs in a
QThread so the GUI stays responsive; progress is reported via
signals.

This module intentionally stays thin — all the actual FEP work
lives in ``core.fep_mcp`` and is invoked through those wrappers.
The panel is the GUI front-end for those tools, nothing more.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import fep_recipes
from ..core.fep_mcp import (
    fep_build_system,
    fep_capture_lambda_snapshots,
    fep_configure_recipe,
    fep_get_disappearing_atoms,
    fep_run_sweep,
    fep_run_sweep_with_checkpoint,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class FEPRunWorker(QObject):
    """Runs ``fep_run_sweep_with_checkpoint`` in a worker thread.

    Emits ``progress(dict)`` periodically and ``finished(dict)`` /
    ``failed(str)`` when done. Cancellation flips ``self._cancel``
    to True; the underlying sweep's ``cancellation_check`` sees it
    and raises ``FEPInterrupted`` after the current window finishes,
    which the wrapper catches and surfaces as
    ``status="interrupted"`` (along with the preserved partial
    energies in the checkpoint file).
    """

    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        pdb_path: str,
        mode: str,
        recipe_name: str,
        mutation: Optional[str],
        ligand_resname: Optional[str],
        checkpoint_path: Optional[str] = None,
        resume_from: Optional[str] = None,
        position_snapshot_dir: Optional[str] = None,
    ):
        super().__init__()
        self._pdb_path = pdb_path
        self._mode = mode
        self._recipe_name = recipe_name
        self._mutation = mutation
        self._ligand_resname = ligand_resname
        self._checkpoint_path = checkpoint_path
        self._resume_from = resume_from
        self._snapshot_dir = position_snapshot_dir
        # Atomic flag the GUI can flip from the main thread.
        from threading import Event
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def _cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        try:
            recipe = fep_recipes.get_recipe(self._recipe_name)
            self.progress.emit({
                "stage": "starting",
                "message": (
                    f"{'Resume' if self._resume_from else 'Run'} "
                    f"{self._mode} FEP on {self._pdb_path} "
                    f"with recipe '{self._recipe_name}' "
                    f"({recipe['n_lambdas']} λ × "
                    f"{recipe['n_equilibration_steps'] + recipe['n_production_steps']} steps)"
                ),
            })

            def cb(progress_dict):
                self.progress.emit(progress_dict)

            # Use the checkpointed variant — supports cancellation and
            # resume. ``resume_from`` paths go through fep_resume_sweep,
            # which reads the existing checkpoint and only runs the
            # remaining windows.
            if self._resume_from:
                from ..core.fep_mcp import fep_resume_sweep
                result = fep_resume_sweep(
                    checkpoint_path=self._resume_from,
                    cancellation_check=self._cancelled,
                )
            else:
                from ..core.fep_mcp import fep_run_sweep_with_checkpoint
                result = fep_run_sweep_with_checkpoint(
                    pdb_path=self._pdb_path,
                    mode=self._mode,
                    recipe_name=self._recipe_name,
                    mutation=self._mutation,
                    ligand_resname=self._ligand_resname,
                    checkpoint_path=self._checkpoint_path,
                    cancellation_check=self._cancelled,
                    position_snapshot_dir=self._snapshot_dir,
                )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Panel widget
# ---------------------------------------------------------------------------

class FEPPanel(QWidget):
    """Two-tab FEP control panel: Setup and Run."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[FEPRunWorker] = None
        self._api = None  # StudioAPI handle; set by FEPDockWidget.set_api
        # Debounce timer for PDB-path changes — waits 300 ms of silence
        # before clearing the viewport + loading the new file.  Stops the
        # scene from flickering as the user types the path character by
        # character, and from going blank if they edit in the middle.
        self._pdb_load_timer = QTimer(self)
        self._pdb_load_timer.setSingleShot(True)
        self._pdb_load_timer.setInterval(300)
        self._pdb_load_timer.timeout.connect(self._apply_pdb_path)
        self._build_ui()
        self._on_mode_changed("mutation")
        self._on_recipe_changed("quick_test")

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        tabs = QTabWidget()

        # Setup tab ------------------------------------------------------
        setup = QWidget()
        form = QFormLayout(setup)

        # PDB file picker
        self._pdb_edit = QLineEdit()
        self._pdb_edit.setPlaceholderText("Absolute path to a .pdb file")
        # Whenever the path changes (browse or manual edit) clear the
        # viewport and load the new PDB. The user wants the FEP dock
        # to be the "single source of truth" for which structure is on
        # screen, so loading another FEP target always replaces the
        # current scene.
        self._pdb_edit.textChanged.connect(self._on_pdb_path_changed)
        pdb_browse = QPushButton("Browse…")
        pdb_browse.clicked.connect(self._on_pdb_browse)
        pdb_row = QHBoxLayout()
        pdb_row.addWidget(self._pdb_edit)
        pdb_row.addWidget(pdb_browse)
        pdb_widget = QWidget()
        pdb_widget.setLayout(pdb_row)
        form.addRow("PDB file:", pdb_widget)

        # Mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["mutation", "ligand_decouple"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self._mode_combo)

        # Mutation residue
        self._mutation_edit = QLineEdit()
        self._mutation_edit.setPlaceholderText("e.g. Lys6  /  6  /  LYS6")
        form.addRow("Mutation residue:", self._mutation_edit)

        # Ligand resname
        self._ligand_edit = QLineEdit()
        self._ligand_edit.setPlaceholderText("e.g. LIG  /  BOS")
        form.addRow("Ligand resname:", self._ligand_edit)

        # Recipe selector
        self._recipe_combo = QComboBox()
        self._recipe_combo.addItems(fep_recipes.list_recipes())
        self._recipe_combo.setCurrentText("quick_test")
        self._recipe_combo.currentTextChanged.connect(self._on_recipe_changed)
        form.addRow("Recipe:", self._recipe_combo)

        # Recipe info (read-only)
        self._recipe_info = QPlainTextEdit()
        self._recipe_info.setReadOnly(True)
        self._recipe_info.setMaximumHeight(140)
        form.addRow("Recipe info:", self._recipe_info)

        # Build button
        self._build_btn = QPushButton("Build system")
        self._build_btn.clicked.connect(self._on_build)
        form.addRow(self._build_btn)

        self._build_result = QLabel("No system built yet.")
        self._build_result.setWordWrap(True)
        form.addRow(self._build_result)

        # Highlight disappearing atoms button.
        self._highlight_btn = QPushButton(
            "Highlight disappearing atoms in viewport"
        )
        self._highlight_btn.clicked.connect(self._on_highlight)
        form.addRow(self._highlight_btn)
        self._highlight_result = QLabel("")
        self._highlight_result.setWordWrap(True)
        form.addRow(self._highlight_result)

        # Run tab --------------------------------------------------------
        run = QWidget()
        run_layout = QVBoxLayout(run)

        self._run_status = QLabel("Idle.")
        run_layout.addWidget(self._run_status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        run_layout.addWidget(self._progress)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run sweep")
        self._run_btn.clicked.connect(self._on_run)
        self._resume_btn = QPushButton("Resume checkpoint…")
        self._resume_btn.clicked.connect(self._on_resume)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._resume_btn)
        btn_row.addWidget(self._cancel_btn)
        run_layout.addLayout(btn_row)

        # Checkpoint path row (optional override).
        cp_row = QHBoxLayout()
        self._cp_edit = QLineEdit()
        self._cp_edit.setPlaceholderText(
            "(default — derived from sweep inputs)"
        )
        cp_browse = QPushButton("…")
        cp_browse.clicked.connect(self._on_cp_browse)
        cp_row.addWidget(QLabel("Checkpoint:"))
        cp_row.addWidget(self._cp_edit)
        cp_row.addWidget(cp_browse)
        run_layout.addLayout(cp_row)

        # Snapshot output row (per-λ PDBs for visualisation).
        snap_row = QHBoxLayout()
        from PySide6.QtWidgets import QCheckBox
        self._snap_check = QCheckBox("Save per-λ snapshots")
        self._snap_check.setChecked(True)
        snap_row.addWidget(self._snap_check)
        self._snap_edit = QLineEdit()
        self._snap_edit.setPlaceholderText(
            "(default — <pdb_dir>/fep_snapshots/...)"
        )
        self._snap_browse = QPushButton("…")
        self._snap_browse.clicked.connect(self._on_snap_browse)
        snap_row.addWidget(QLabel("Snapshots dir:"))
        snap_row.addWidget(self._snap_edit)
        snap_row.addWidget(self._snap_browse)
        run_layout.addLayout(snap_row)
        self._snap_result = QLabel("")
        self._snap_result.setWordWrap(True)
        run_layout.addWidget(self._snap_result)

        self._run_result = QPlainTextEdit()
        self._run_result.setReadOnly(True)
        run_layout.addWidget(self._run_result)

        self._open_analyze_btn = QPushButton("Open in Analyze tab →")
        self._open_analyze_btn.clicked.connect(self._on_open_analyze)
        self._open_analyze_btn.setEnabled(False)
        run_layout.addWidget(self._open_analyze_btn)

        # Analyze tab.
        from .fep_analyze import FEPAnalyzeWidget
        self._analyze_widget = FEPAnalyzeWidget()

        tabs.addTab(setup, "Setup")
        tabs.addTab(run, "Run")
        tabs.addTab(self._analyze_widget, "Analyze")
        self._tabs = tabs

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(tabs)

    # --------------------------------------------------------------- slots

    def _on_pdb_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PDB file", "", "PDB files (*.pdb);;All files (*)"
        )
        if path:
            self._pdb_edit.setText(path)

    def _on_pdb_path_changed(self, path: str) -> None:
        """Restart the debounce timer on every keystroke / Browse.

        The actual clear + load happens in :meth:`_apply_pdb_path`
        after 300 ms of silence so partial-typed paths don't blank
        the viewport mid-edit.  Browse also routes through here
        (``_pdb_edit.setText`` emits ``textChanged``), so both the
        picker and manual typing share the same debounced path.
        """
        # Don't bother restarting the timer if the user cleared the
        # field entirely — there's no work to do.
        if not (path or "").strip():
            self._pdb_load_timer.stop()
            return
        self._pdb_load_timer.start()

    def _apply_pdb_path(self) -> None:
        """Clear the viewport and load the selected PDB.

        Runs 300 ms after the last ``textChanged`` so typing a path
        doesn't flash the scene empty mid-keystroke.  We still
        validate the path here (a partial path may have looked
        plausible when ``textChanged`` fired).
        """
        # Read the current text from the line edit — the QTimer.timeout
        # signal has no arguments, so we can't receive the path
        # directly.  Reading from the widget gives us the latest value.
        path = self._pdb_edit.text().strip()
        if not path:
            return
        if not Path(path).is_file():
            return
        if self._api is None:
            return
        try:
            self._api.clear()
            # Let SceneAPI.load() auto-pick the rep/scheme based on
            # molecule type — proteins get cartoon + secondary;
            # pure-hetero (methane, ligand, single residue) get
            # ball_and_stick + element (CPK).
            obj = self._api.load(path)
            self._status_label_widget().setText(
                f"Loaded {obj.name} ({obj.molecule.n_atoms()} atoms, "
                f"rep={obj.representation})"
            )
        except Exception as e:
            self._status_label_widget().setText(
                f"⚠ Could not load PDB: {type(e).__name__}: {e}"
            )

    def _status_label_widget(self):
        """Return the (mutable) status label for transient feedback.

        We re-use the build-result label since it's already in the
        layout and visible to the user. If the panel ever grows a
        dedicated status row, swap this for the new widget.
        """
        return self._build_result

    def _on_mode_changed(self, mode: str) -> None:
        self._mutation_edit.setVisible(mode == "mutation")
        self._ligand_edit.setVisible(mode == "ligand_decouple")

    def _on_recipe_changed(self, name: str) -> None:
        cfg = fep_recipes.get_recipe(name)
        lines = [
            f"Name:        {cfg['name']}",
            f"λ windows:    {cfg['n_lambdas']}",
            f"Equilibration: {cfg['n_equilibration_steps']} steps",
            f"Production:    {cfg['n_production_steps']} steps",
            f"Sample every:  {cfg['sample_every']}",
            f"Neighbor re-eval: {cfg['sample_neighbor_lambdas']}",
            f"Restraint k:  {cfg['restraint_k_kJmol_nm2']} kJ/mol/nm²",
            "",
            cfg["description"],
        ]
        self._recipe_info.setPlainText("\n".join(lines))

    def _on_build(self) -> None:
        path = self._pdb_edit.text().strip()
        if not path:
            self._build_result.setText("⚠ Select a PDB file first.")
            return
        mode = self._mode_combo.currentText()
        kwargs = {}
        if mode == "mutation":
            kwargs["mutation"] = self._mutation_edit.text().strip() or None
        else:
            kwargs["ligand_resname"] = (
                self._ligand_edit.text().strip() or None
            )
        result = fep_build_system(pdb_path=path, **kwargs)
        if result.get("status") == "ok":
            parts = [
                f"✓ {result['mode']}: {result['n_particles']} particles, "
                f"{result.get('n_waters', '?')} waters"
            ]
            if "n_disappearing" in result:
                parts.append(f", {result['n_disappearing']} disappearing atoms")
            if "n_ligand_atoms" in result:
                parts.append(f", {result['n_ligand_atoms']} ligand atoms")
            self._build_result.setText("".join(parts))
        else:
            self._build_result.setText(
                f"✗ {result.get('error', 'unknown error')}"
            )

    def _on_highlight(self) -> None:
        """Build the system, write a filtered PDB of just the
        disappearing atoms, and load it as a new SceneObject so
        the user can see exactly which atoms the sweep is removing.
        """
        path = self._pdb_edit.text().strip()
        if not path:
            self._highlight_result.setText(
                "⚠ Select a PDB file first."
            )
            return
        mode = self._mode_combo.currentText()
        kwargs = {}
        if mode == "mutation":
            kwargs["mutation"] = self._mutation_edit.text().strip() or None
        else:
            kwargs["ligand_resname"] = (
                self._ligand_edit.text().strip() or None
            )

        # Write the filtered PDB to a sibling location so the GUI
        # can also keep it around for re-loads.
        from pathlib import Path
        pdb = Path(path)
        target = kwargs.get("mutation") or kwargs.get("ligand_resname") or "x"
        out_pdb = str(
            pdb.parent / f"{pdb.stem}_{mode}_disappearing_{target}.pdb"
        )

        result = fep_get_disappearing_atoms(
            pdb_path=path,
            mode=mode,
            output_pdb=out_pdb,
            **kwargs,
        )
        if result.get("status") != "ok":
            self._highlight_result.setText(
                f"✗ {result.get('error')}"
            )
            return

        # Hand the file off to the GUI's main API to load as a
        # separate SceneObject (ball+stick rep, "element" scheme
        # so each atom is colored by element type).
        api = self._api
        if api is None:
            # Fall back: just record the output path.
            self._highlight_result.setText(
                f"✓ {result['n_disappearing']} disappearing atoms. "
                f"Filtered PDB: {result.get('output_pdb', out_pdb)} "
                f"(couldn't access GUI API to load automatically — "
                f"open the file via File → Open PDB…)"
            )
            return

        try:
            stem = Path(out_pdb).stem
            obj = api.load(
                out_pdb,
                name=f"{stem} (alchemical)",
                representation="ball_and_stick",
                scheme="element",
            )
            res_summary = ", ".join(
                f"{r['residue']} ({len(r['atom_indices'])} atoms)"
                for r in result.get("residue_summary", [])
            )
            self._highlight_result.setText(
                f"✓ Highlighted {result['n_disappearing']} "
                f"disappearing atoms: {res_summary}"
            )
        except Exception as e:
            self._highlight_result.setText(
                f"✓ wrote filtered PDB to {out_pdb} but failed to "
                f"load: {e}"
            )

    def _on_run(self) -> None:
        path = self._pdb_edit.text().strip()
        if not path:
            self._run_status.setText("⚠ Select a PDB file first.")
            return
        mode = self._mode_combo.currentText()
        mutation = (
            self._mutation_edit.text().strip()
            if mode == "mutation" else None
        )
        ligand = (
            self._ligand_edit.text().strip()
            if mode == "ligand_decouple" else None
        )
        if mode == "mutation" and not mutation:
            self._run_status.setText("⚠ Enter a mutation residue (e.g. Lys6).")
            return
        if mode == "ligand_decouple" and not ligand:
            self._run_status.setText("⚠ Enter a ligand resname.")
            return

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._run_result.clear()
        self._run_status.setText("Starting…")

        cp_path = self._cp_edit.text().strip() or None
        snap_dir = (
            self._snap_edit.text().strip()
            if self._snap_check.isChecked() else None
        )
        if snap_dir:
            from pathlib import Path
            Path(snap_dir).mkdir(parents=True, exist_ok=True)
        self._thread = QThread()
        self._worker = FEPRunWorker(
            pdb_path=path,
            mode=mode,
            recipe_name=self._recipe_combo.currentText(),
            mutation=mutation,
            ligand_resname=ligand,
            checkpoint_path=cp_path,
            position_snapshot_dir=snap_dir,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_resume(self) -> None:
        """Pick a checkpoint file and continue a previously-cancelled sweep."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick checkpoint file", "",
            "FEP checkpoints (*.fepcp);;All files (*)"
        )
        if not path:
            return
        self._cp_edit.setText(path)
        self._run_status.setText(f"Resuming from {path}…")
        self._run_btn.setEnabled(False)
        self._resume_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)

        self._thread = QThread()
        self._worker = FEPRunWorker(
            pdb_path="",  # unused when resuming
            mode="",
            recipe_name="quick_test",
            mutation=None,
            ligand_resname=None,
            resume_from=path,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_cp_browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Pick checkpoint file", "",
            "FEP checkpoints (*.fepcp);;All files (*)"
        )
        if path:
            self._cp_edit.setText(path)

    def _on_snap_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Pick snapshot output directory", ""
        )
        if path:
            self._snap_edit.setText(path)

    def _on_progress(self, info: dict) -> None:
        stage = info.get("stage")
        if stage == "starting":
            self._run_status.setText(info.get("message", "Starting…"))
            self._progress.setValue(2)
        elif stage == "window":
            idx = info.get("lambda_idx", 0)
            n = info.get("n_lambdas", 1)
            lam = info.get("lam_value", 0.0)
            self._run_status.setText(
                f"λ = {lam:.3f} ({idx + 1}/{n}) "
                f"({info.get('n_samples', 0)} samples)"
            )
            # Reserve 5% for setup, 90% across windows, 5% for BAR.
            self._progress.setValue(5 + int(90 * (idx + 1) / max(n, 1)))

    def _on_finished(self, result: dict) -> None:
        self._run_btn.setEnabled(True)
        self._resume_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        import json
        self._last_result = result

        status = result.get("status", "error")
        if status == "ok":
            self._progress.setValue(100)
            dg = result.get("delta_g_kJmol", 0.0)
            sig = result.get("sigma_kJmol", 0.0)
            tag = " (resumed)" if result.get("resumed_from_checkpoint") else ""
            self._run_status.setText(
                f"✓ ΔG_FEP = {dg:.3f} ± {sig:.3f} kJ/mol{tag} "
                f"({result['delta_g_kcal_mol']:.3f} ± {sig/4.184:.3f} kcal/mol)"
            )
            self._run_result.setPlainText(json.dumps(result, indent=2))
            self._open_analyze_btn.setEnabled(True)
            # If snapshots were enabled, list what was captured.
            snap_dir = self._snap_edit.text().strip()
            if snap_dir and self._snap_check.isChecked():
                from pathlib import Path
                snap_files = sorted(Path(snap_dir).glob("lambda_*.pdb"))
                if snap_files:
                    self._snap_result.setText(
                        f"✓ {len(snap_files)} per-λ snapshots in {snap_dir}. "
                        f"Load them via File → Open PDB… to flip through."
                    )
        elif status == "interrupted":
            n_done = result.get("n_completed", 0)
            n_total = len(result.get("completed_lambdas", [])) + len(
                result.get("remaining_lambdas", [])
            )
            cp = result.get("checkpoint_path", "?")
            self._progress.setValue(int(100 * n_done / max(n_total, 1)))
            self._run_status.setText(
                f"⏸ Cancelled at λ window {n_done}/{n_total}. "
                f"Checkpoint: {cp}"
            )
            self._run_result.setPlainText(json.dumps(result, indent=2))
        else:  # error
            self._progress.setValue(0)
            self._run_status.setText(f"✗ {status}")
            self._run_result.setPlainText(json.dumps(result, indent=2))

    def _on_open_analyze(self) -> None:
        """Switch to the Analyze tab and load the most recent result."""
        if getattr(self, "_last_result", None) is None:
            return
        self._analyze_widget.set_result(self._last_result)
        # Switch the visible tab to the Analyze index.
        self._tabs.setCurrentIndex(self._tabs.indexOf(self._analyze_widget))

    def _on_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self._resume_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(0)
        self._run_status.setText("✗ Failed.")
        self._run_result.setPlainText(error)

    def _on_cancel(self) -> None:
        # Flip the worker's atomic cancel flag. The sweep's
        # cancellation_check sees it after the current λ window
        # finishes and raises FEPInterrupted; the wrapper catches
        # that and writes a final checkpoint with status="interrupted".
        if self._worker is None:
            return
        self._worker.request_cancel()
        self._run_status.setText(
            "Cancelling — finishing current window then writing checkpoint…"
        )
        self._cancel_btn.setEnabled(False)


# ---------------------------------------------------------------------------
# Dock wrapper
# ---------------------------------------------------------------------------

class FEPDockWidget(QDockWidget):
    """Dockable wrapper around :class:`FEPPanel`.

    Lazy-imports the panel so the GUI can decide whether to load
    OpenMM / FEP at all. Pass ``parent`` to integrate into a
    QMainWindow.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("FEP…", parent)
        self.setWidget(FEPPanel(self))
        # Sensible default size: a horizontal strip when first
        # detached, vertical strip when docked on the right.
        self.resize(520, 600)
        self._api = None  # set later via set_api() from main_window

    def set_api(self, api) -> None:
        """Hand the StudioAPI handle to the panel so visualisation
        actions (highlight disappearing atoms, load snapshot PDBs)
        can drop them straight into the scene."""
        self._api = api
        if self.widget() is not None:
            self.widget()._api = api