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
from typing import Optional

from PySide6.QtCore import Qt, QObject, QThread, Signal
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
    fep_configure_recipe,
    fep_run_sweep,
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
    ):
        super().__init__()
        self._pdb_path = pdb_path
        self._mode = mode
        self._recipe_name = recipe_name
        self._mutation = mutation
        self._ligand_resname = ligand_resname
        self._checkpoint_path = checkpoint_path
        self._resume_from = resume_from
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
        self._thread = QThread()
        self._worker = FEPRunWorker(
            pdb_path=path,
            mode=mode,
            recipe_name=self._recipe_combo.currentText(),
            mutation=mutation,
            ligand_resname=ligand,
            checkpoint_path=cp_path,
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