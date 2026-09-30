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
    """Runs ``fep_run_sweep`` in a worker thread.

    Emits ``progress(dict)`` periodically and ``finished(dict)`` /
    ``failed(str)`` when done. The worker holds the QThread alive
    via its lifetime; cancellation is a no-op for now (long runs
    will need a flag-based abort path in a future revision).
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
    ):
        super().__init__()
        self._pdb_path = pdb_path
        self._mode = mode
        self._recipe_name = recipe_name
        self._mutation = mutation
        self._ligand_resname = ligand_resname

    def run(self) -> None:
        try:
            recipe = fep_recipes.get_recipe(self._recipe_name)
            self.progress.emit({
                "stage": "starting",
                "message": (
                    f"Running {self._mode} FEP on "
                    f"{self._pdb_path} with recipe '{self._recipe_name}' "
                    f"({recipe['n_lambdas']} λ × "
                    f"{recipe['n_equilibration_steps'] + recipe['n_production_steps']} steps)"
                ),
            })

            def cb(lambda_idx, n_lambdas, lam_value, n_samples):
                    # Coarse-grained progress: 5% reserved for build,
                    # 90% split across λ windows, 5% reserved for
                    # BAR analysis.
                    self.progress.emit({
                        "stage": "window",
                        "lambda_idx": lambda_idx,
                        "n_lambdas": n_lambdas,
                        "lam_value": lam_value,
                        "n_samples": n_samples,
                    })

            result = fep_run_sweep(
                pdb_path=self._pdb_path,
                mode=self._mode,
                recipe_name=self._recipe_name,
                mutation=self._mutation,
                ligand_resname=self._ligand_resname,
                progress_callback=cb,
            )
            if result.get("status") == "ok":
                self.finished.emit(result)
            else:
                self.failed.emit(
                    result.get("error", "Unknown error")
                    + "\n"
                    + result.get("traceback", "")
                )
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
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setEnabled(False)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        run_layout.addLayout(btn_row)

        self._run_result = QPlainTextEdit()
        self._run_result.setReadOnly(True)
        run_layout.addWidget(self._run_result)

        tabs.addTab(setup, "Setup")
        tabs.addTab(run, "Run")

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

        self._thread = QThread()
        self._worker = FEPRunWorker(
            pdb_path=path,
            mode=mode,
            recipe_name=self._recipe_combo.currentText(),
            mutation=mutation,
            ligand_resname=ligand,
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
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(100)
        dg = result.get("delta_g_kJmol", 0.0)
        sig = result.get("sigma_kJmol", 0.0)
        self._run_status.setText(
            f"✓ ΔG_FEP = {dg:.3f} ± {sig:.3f} kJ/mol "
            f"({result['delta_g_kcal_mol']:.3f} ± {sig/4.184:.3f} kcal/mol)"
        )
        import json
        self._run_result.setPlainText(json.dumps(result, indent=2))

    def _on_failed(self, error: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(0)
        self._run_status.setText("✗ Failed.")
        self._run_result.setPlainText(error)

    def _on_cancel(self) -> None:
        # Best-effort: tear down the worker thread. Cancellation of an
        # in-flight OpenMM integration loop requires a flag the runner
        # checks — out of scope for this minimum-runnable revision.
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.quit()
            self._run_status.setText("Cancel requested.")


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