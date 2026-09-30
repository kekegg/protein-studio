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
from ..core.fep_mcp import ti_per_window
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

        # Raw result (collapsible-style via QPlainTextEdit).
        outer.addWidget(QLabel("Raw result:"))
        self._raw = QPlainTextEdit()
        self._raw.setReadOnly(True)
        self._raw.setMaximumHeight(140)
        outer.addWidget(self._raw)

    # ------------------------------------------------------------------ data

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