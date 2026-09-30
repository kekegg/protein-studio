"""FEP Analyze tab — three diagnostic plots of a finished sweep.

Reads a finished ``fep_run_sweep`` result dict and shows:

1. :class:`BarChartWidget`     ΔG per window, with sigma error bars.
2. :class:`ConvergenceWidget` Cumulative ΔG as the windows accumulate —
                              should plateau when sampling is converged.
3. :class:`ComparisonWidget`  BAR per-window vs trapezoidal TI per-window.
                              The two estimators should agree to within
                              a few kJ/mol; large divergences signal
                              insufficient sampling.

Plots are computed in this module from the per-window entries + raw
energies in the result dict — the wrapper layer (``fep_mcp``) hands
us ``per_window`` (BAR ΔG per pair of adjacent windows) and the raw
``energies_by_lambda`` map so we can run TI independently.

Public entry point:

    >>> panel = FEPAnalyzeWidget()
    >>> panel.set_result({"per_window": [...], "energies_by_lambda": {...}, ...})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import fep as fep_core
from ..core.fep_mcp import (
    ti_per_window,
    fep_best_reference,
    fep_compare_sweep_to_reference,
    fep_list_references,
)
from .fep_plots import (
    BarChartWidget,
    ComparisonWidget,
    ConvergenceWidget,
    PlotData,
)


class FEPAnalyzeWidget(QWidget):
    """Tab widget that renders the three diagnostic charts."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()
        self._populate_reference_combo()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # Toolbar.
        toolbar = QHBoxLayout()
        self._load_btn = QPushButton("Load result JSON…")
        self._load_btn.clicked.connect(self._on_load_json)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(self._load_btn)
        toolbar.addWidget(self._clear_btn)
        toolbar.addStretch(1)
        self._summary_label = QLabel("No result loaded.")
        toolbar.addWidget(self._summary_label)
        outer.addLayout(toolbar)

        # ΔG per-window bar chart.
        bar_box = QGroupBox("ΔG per window (BAR)")
        bar_layout = QVBoxLayout(bar_box)
        self._bar_chart = BarChartWidget()
        bar_layout.addWidget(self._bar_chart)
        outer.addWidget(bar_box)

        # Cumulative convergence.
        conv_box = QGroupBox("Cumulative ΔG — convergence")
        conv_layout = QVBoxLayout(conv_box)
        self._conv_chart = ConvergenceWidget()
        conv_layout.addWidget(self._conv_chart)
        outer.addWidget(conv_box)

        # BAR vs TI comparison.
        cmp_box = QGroupBox("BAR vs trapezoidal (TI)")
        cmp_layout = QVBoxLayout(cmp_box)
        self._cmp_chart = ComparisonWidget()
        cmp_layout.addWidget(self._cmp_chart)
        outer.addWidget(cmp_box)

        # Reference benchmark comparison.
        ref_box = QGroupBox("Compare to literature reference")
        ref_layout = QVBoxLayout(ref_box)

        ref_picker_row = QHBoxLayout()
        ref_picker_row.addWidget(QLabel("Reference:"))
        self._ref_combo = QComboBox()
        self._ref_combo.setMinimumWidth(320)
        ref_picker_row.addWidget(self._ref_combo)
        self._ref_compare_btn = QPushButton("Compare")
        self._ref_compare_btn.clicked.connect(self._on_compare_reference)
        ref_picker_row.addWidget(self._ref_compare_btn)
        ref_picker_row.addStretch(1)
        ref_layout.addLayout(ref_picker_row)

        self._ref_citation_label = QLabel("")
        self._ref_citation_label.setWordWrap(True)
        ref_layout.addWidget(self._ref_citation_label)

        self._ref_result_label = QLabel(
            "Pick a reference, then click Compare."
        )
        self._ref_result_label.setWordWrap(True)
        self._ref_result_label.setTextFormat(Qt.TextFormat.RichText)
        ref_layout.addWidget(self._ref_result_label)
        outer.addWidget(ref_box)

        # Raw result (collapsible-style via QPlainTextEdit).
        outer.addWidget(QLabel("Raw result:"))
        self._raw = QPlainTextEdit()
        self._raw.setReadOnly(True)
        self._raw.setMaximumHeight(140)
        outer.addWidget(self._raw)

    # ------------------------------------------------------------------ data

    def _populate_reference_combo(self) -> None:
        """Fill the reference dropdown with the shipped catalog."""
        refs = fep_list_references()["references"]
        self._ref_combo.blockSignals(True)
        self._ref_combo.clear()
        self._ref_combo.addItem("(none — pick a reference)", userData=None)
        for r in refs:
            label = r["name"]
            if not r.get("runnable", True):
                label += "  ⚠ not runnable"
            self._ref_combo.addItem(label, userData=r["name"])
        self._ref_combo.blockSignals(False)
        self._ref_combo.currentIndexChanged.connect(
            self._on_reference_changed)

    def _on_reference_changed(self, idx: int) -> None:
        """Show the citation for the selected reference."""
        name = self._ref_combo.itemData(idx)
        if not name:
            self._ref_citation_label.setText("")
            return
        refs = fep_list_references()["references"]
        ref = next((r for r in refs if r["name"] == name), None)
        if ref is None:
            return
        text = (
            f"<b>{ref['description']}</b><br>"
            f"<i>{ref['citation']}</i><br>"
            f"Reference ΔG = {ref['reference_dG_kJmol']:+.2f} ± "
            f"{ref['reference_dG_unc_kJmol']:.2f} kJ/mol "
            f"({ref['reference_dG_kcal_mol']:+.2f} kcal/mol)"
        )
        if ref.get("pdb_file"):
            text += f"<br>PDB: {ref['pdb_file']}"
        if ref.get("notes"):
            text += f"<br><span style='color:#9aa0a6'>{ref['notes']}</span>"
        self._ref_citation_label.setText(text)

    def _on_compare_reference(self) -> None:
        """Compare the current sweep result against the picked reference."""
        if self._current is None:
            self._ref_result_label.setText(
                "<span style='color:#dca85b'>⚠ Load a sweep result first.</span>"
            )
            return
        idx = self._ref_combo.currentIndex()
        name = self._ref_combo.itemData(idx)
        if not name:
            self._ref_result_label.setText(
                "<span style='color:#dca85b'>⚠ Pick a reference.</span>"
            )
            return
        cmp = fep_compare_sweep_to_reference(self._current, name)
        if cmp.get("status") != "ok":
            self._ref_result_label.setText(
                f"<span style='color:#dca85b'>✗ {cmp.get('error')}</span>"
            )
            return
        agreement = cmp["agreement"]
        z = cmp["z_score"]
        dev = cmp["deviation_kJmol"]
        combined = cmp["combined_unc_kJmol"]
        ref = cmp["reference"]
        if agreement == "within_1sigma":
            color = "#5b96e2"
            icon = "✓"
        elif agreement == "within_2sigma":
            color = "#dca85b"
            icon = "~"
        else:
            color = "#cc4040"
            icon = "✗"
        html = (
            f"<span style='color:{color}'><b>{icon} "
            f"{agreement.replace('_', ' ')}</b></span><br>"
            f"z = {z:+.2f} (deviation {dev:+.2f} kJ/mol / "
            f"combined σ {combined:.2f} kJ/mol)<br>"
            f"Reference: {ref['reference_dG_kJmol']:+.2f} ± "
            f"{ref['reference_dG_unc_kJmol']:.2f} kJ/mol "
            f"({ref['reference_dG_kcal_mol']:+.2f} kcal/mol)<br>"
            f"Your value: {cmp['measured_dG_kJmol']:+.2f} ± "
            f"{cmp['measured_sigma_kJmol']:.2f} kJ/mol"
        )
        self._ref_result_label.setText(html)

    def set_result(self, result: Dict[str, Any]) -> None:
        """Populate the charts from a finished ``fep_run_sweep`` result."""
        self._current = result
        per_window = result.get("per_window", [])
        if not per_window:
            self._on_clear()
            return

        labels = [f"{w['lambda']:.2f}" for w in per_window]
        bar_values = [w["delta_g_kJmol"] for w in per_window]
        bar_errors = [w["sigma_kJmol"] for w in per_window]

        # BAR vs TI comparison — recompute TI per window from raw energies.
        energies_by_lambda = result.get("energies_by_lambda")
        ti_values = []
        ti_total = None
        if energies_by_lambda:
            lambdas_sorted = sorted(float(l) for l in energies_by_lambda)
            ti_values = ti_per_window(energies_by_lambda, lambdas_sorted)
            ti_total = fep_core.analyze_fep_trapezoidal(
                energies_by_lambda, lambdas_sorted,
            )[0]

        # Cumulative ΔG (BAR-based).
        cum_values = []
        running = 0.0
        for v in bar_values:
            running += v
            cum_values.append(running)

        self._bar_chart.set_data(PlotData(
            x_labels=labels,
            y_values=bar_values,
            y_errors=bar_errors,
            title="ΔG per window (BAR with σ error bars)",
            y_label="ΔG per window (kJ/mol)",
        ))
        self._conv_chart.set_data(PlotData(
            x_labels=labels,
            y_values=cum_values,
            title=f"Cumulative ΔG_FEP (running sum, final = {cum_values[-1]:.2f})",
            y_label="Cumulative ΔG (kJ/mol)",
        ))
        if ti_values:
            self._cmp_chart.set_data(PlotData(
                x_labels=labels,
                y_values=bar_values,
                secondary_y=ti_values,
                title=(
                    f"BAR vs TI per window "
                    f"(TI total = {ti_total:.2f} kJ/mol)"
                    if ti_total is not None else
                    "BAR vs TI per window"
                ),
                y_label="ΔG per window (kJ/mol)",
                secondary_label="TI",
            ))

        # Summary.
        dg = result.get("delta_g_kJmol", 0.0)
        sig = result.get("sigma_kJmol", 0.0)
        ti_str = f" / TI {ti_total:+.2f}" if ti_total is not None else ""
        self._summary_label.setText(
            f"BAR ΔG_FEP = {dg:+.3f} ± {sig:.3f} kJ/mol"
            f"{ti_str} · {result.get('n_windows', '?')} windows · "
            f"recipe '{result.get('recipe', '?')}'"
        )

        self._raw.setPlainText(json.dumps(result, indent=2))

    def clear(self) -> None:
        self._on_clear()

    # --------------------------------------------------------------- slots

    def _on_load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FEP result", "",
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                result = json.load(f)
            self.set_result(result)
        except Exception as e:
            self._summary_label.setText(f"✗ Failed to load: {e}")

    def _on_clear(self) -> None:
        self._bar_chart.clear()
        self._conv_chart.clear()
        self._cmp_chart.clear()
        self._raw.clear()
        self._summary_label.setText("No result loaded.")
        self._current = None