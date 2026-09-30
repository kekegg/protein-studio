"""Lightweight QPainter chart widgets for the FEP Analyze tab.

Three plain-Qt charts:

- :class:`BarChartWidget`     ΔG per window, with error bars.
- :class:`ConvergenceWidget` Cumulative ΔG running sum (convergence
                              diagnostic — should plateau when the
                              sweep is long enough).
- :class:`ComparisonWidget`  BAR vs trapezoidal TI overlay.

We deliberately use QPainter instead of matplotlib / pyqtgraph to
avoid a heavy GUI-side dependency. The plots are small (typically
11 bars / lines per sweep) and don't need interactive zoom/pan —
diagnostic eye-candy is enough.

All three follow the protein-studio dark theme (#2d2d2d background,
#dcdcdc foreground, #3b76c2 accent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


_BG       = QColor("#1d1f23")
_FG       = QColor("#dcdcdc")
_FG_SOFT  = QColor("#9aa0a6")
_AXIS     = QColor("#555a60")
_GRID     = QColor("#33373c")
_ACCENT   = QColor("#3b76c2")
_ACCENT_2 = QColor("#5b96e2")
_ERROR    = QColor("#dca85b")


@dataclass
class PlotData:
    """Container for one chart's data.

    All values in kJ/mol. ``x_labels`` are the λ strings shown on
    the x-axis. Optional ``secondary_*`` fields let the comparison
    widget overlay a second series without making a second widget.
    """
    x_labels: List[str] = field(default_factory=list)
    y_values: List[float] = field(default_factory=list)
    y_errors: List[float] = field(default_factory=list)
    title: str = ""
    y_label: str = "ΔG (kJ/mol)"
    secondary_y: Optional[List[float]] = None
    secondary_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _ChartBase(QWidget):
    """Shared axis drawing for the three concrete chart widgets."""

    PADDING_LEFT = 56
    PADDING_RIGHT = 16
    PADDING_TOP = 24
    PADDING_BOTTOM = 36

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._data = PlotData()
        self.setMinimumHeight(150)

    # ------------------------------------------------------------------ data

    def set_data(self, data: PlotData) -> None:
        self._data = data
        self.update()

    def clear(self) -> None:
        self._data = PlotData()
        self.update()

    # --------------------------------------------------------------- painting

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        data = self._data
        if not data.y_values:
            self._draw_empty(painter)
            return

        plot_rect = self._plot_rect()
        y_min, y_max = self._y_range()

        self._draw_grid(painter, plot_rect, y_min, y_max)
        self._draw_axes(painter, plot_rect, y_min, y_max)
        self._draw_title(painter)

    # --------------------------------------------------------- helpers

    def _plot_rect(self) -> QRectF:
        r = self.rect()
        return QRectF(
            self.PADDING_LEFT,
            self.PADDING_TOP,
            r.width() - self.PADDING_LEFT - self.PADDING_RIGHT,
            r.height() - self.PADDING_TOP - self.PADDING_BOTTOM,
        )

    def _y_range(self) -> Tuple[float, float]:
        vals = list(self._data.y_values)
        errs = list(self._data.y_errors)
        if self._data.secondary_y is not None:
            vals.extend(self._data.secondary_y)
        if not vals:
            return (0.0, 1.0)
        lo = min(v - e for v, e in zip(vals, errs + [0.0] * len(vals)))
        hi = max(v + e for v, e in zip(vals, errs + [0.0] * len(vals)))
        if lo == hi:
            return (lo - 1.0, hi + 1.0)
        pad = max(0.1 * (hi - lo), 0.5)
        return (lo - pad, hi + pad)

    def _draw_grid(self, p: QPainter, r: QRectF,
                   y_min: float, y_max: float) -> None:
        p.setPen(QPen(_GRID, 1, Qt.PenStyle.DashLine))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = r.bottom() - frac * r.height()
            p.drawLine(r.left(), y, r.right(), y)

    def _draw_axes(self, p: QPainter, r: QRectF,
                   y_min: float, y_max: float) -> None:
        p.setPen(QPen(_AXIS, 1))
        # y-axis
        p.drawLine(r.left(), r.top(), r.left(), r.bottom())
        # x-axis
        p.drawLine(r.left(), r.bottom(), r.right(), r.bottom())

        p.setPen(QPen(_FG_SOFT))
        p.setFont(QFont("Sans", 8))
        # y tick labels
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = r.bottom() - frac * r.height()
            value = y_min + frac * (y_max - y_min)
            p.drawText(QRectF(r.left() - 50, y - 8, 44, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{value:+.1f}")
        # x tick labels
        n = max(len(self._data.x_labels), 1)
        for i, label in enumerate(self._data.x_labels):
            x = r.left() + (i + 0.5) * r.width() / n
            p.save()
            p.translate(x, r.bottom() + 4)
            p.rotate(-30)
            p.drawText(QRectF(0, 0, 60, 28),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       label)
            p.restore()
        # y-axis label
        p.save()
        p.translate(14, r.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-100, -8, 200, 16),
                   Qt.AlignmentFlag.AlignCenter,
                   self._data.y_label)
        p.restore()

    def _draw_title(self, p: QPainter) -> None:
        if not self._data.title:
            return
        p.setPen(QPen(_FG))
        p.setFont(QFont("Sans", 9, QFont.Weight.Bold))
        p.drawText(QRectF(self.PADDING_LEFT, 4,
                          self.rect().width() - self.PADDING_LEFT,
                          18),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self._data.title)

    def _draw_empty(self, p: QPainter) -> None:
        p.setPen(QPen(_FG_SOFT))
        p.setFont(QFont("Sans", 9))
        p.drawText(self.rect(),
                   Qt.AlignmentFlag.AlignCenter,
                   "No data yet — run a sweep or load a result.")


# ---------------------------------------------------------------------------
# Bar chart with error bars
# ---------------------------------------------------------------------------

class BarChartWidget(_ChartBase):
    """ΔG per window as a bar chart, with sigma error bars."""

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().paintEvent(event)
        data = self._data
        if not data.y_values:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        plot_rect = self._plot_rect()
        y_min, y_max = self._y_range()

        n = len(data.y_values)
        bar_w = plot_rect.width() / max(n, 1) * 0.7
        gap = plot_rect.width() / max(n, 1) * 0.3

        # Zero line if it falls inside the plot.
        if y_min < 0 < y_max:
            y_zero = plot_rect.bottom() - (
                (0.0 - y_min) / (y_max - y_min)
            ) * plot_rect.height()
            p.setPen(QPen(_AXIS, 1, Qt.PenStyle.DotLine))
            p.drawLine(plot_rect.left(), y_zero,
                       plot_rect.right(), y_zero)

        for i, (v, e) in enumerate(zip(data.y_values, data.y_errors)):
            x = plot_rect.left() + i * (bar_w + gap) + gap / 2
            y_top = plot_rect.bottom() - (
                (v - y_min) / (y_max - y_min)
            ) * plot_rect.height()

            bar = QRectF(x, y_top, bar_w, plot_rect.bottom() - y_top)
            p.fillRect(bar, QBrush(_ACCENT))
            p.setPen(QPen(_ACCENT_2, 1))
            p.drawRect(bar)

            # Error bar.
            if e > 0:
                y_top_e = plot_rect.bottom() - (
                    (v + e - y_min) / (y_max - y_min)
                ) * plot_rect.height()
                y_bot_e = plot_rect.bottom() - (
                    (v - e - y_min) / (y_max - y_min)
                ) * plot_rect.height()
                p.setPen(QPen(_ERROR, 1))
                cx = x + bar_w / 2
                p.drawLine(cx, y_top_e, cx, y_bot_e)
                p.drawLine(cx - 4, y_top_e, cx + 4, y_top_e)
                p.drawLine(cx - 4, y_bot_e, cx + 4, y_bot_e)

        p.end()


# ---------------------------------------------------------------------------
# Convergence line chart
# ---------------------------------------------------------------------------

class ConvergenceWidget(_ChartBase):
    """Cumulative ΔG convergence curve.

    Running sum of per-window ΔG across the sweep. Should plateau
    as more windows accumulate. The plateau value approximates the
    total free-energy difference.
    """

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().paintEvent(event)
        data = self._data
        if not data.y_values:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        plot_rect = self._plot_rect()
        y_min, y_max = self._y_range()
        n = len(data.y_values)

        cum = []
        running = 0.0
        for v in data.y_values:
            running += v
            cum.append(running)

        # Line.
        pen = QPen(_ACCENT, 2)
        p.setPen(pen)
        points = []
        for i, v in enumerate(cum):
            x = plot_rect.left() + (i + 0.5) * plot_rect.width() / max(n, 1)
            y = plot_rect.bottom() - (
                (v - y_min) / (y_max - y_min)
            ) * plot_rect.height()
            points.append(QPointF(x, y))
        for i in range(1, len(points)):
            p.drawLine(points[i - 1], points[i])

        # Markers.
        p.setBrush(QBrush(_ACCENT))
        for pt in points:
            p.drawEllipse(pt, 3, 3)

        p.end()


# ---------------------------------------------------------------------------
# BAR vs TI comparison
# ---------------------------------------------------------------------------

class ComparisonWidget(_ChartBase):
    """BAR per-window ΔG vs trapezoidal TI per-window.

    Both series share the same x-axis (λ). When the two estimators
    agree window to window, we have a good handle on convergence;
    large divergences usually signal insufficient sampling.
    """

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().paintEvent(event)
        data = self._data
        if not data.y_values or data.secondary_y is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        plot_rect = self._plot_rect()
        y_min, y_max = self._y_range()
        n = len(data.y_values)

        group_w = plot_rect.width() / max(n, 1)
        bar_w = group_w * 0.35
        gap = group_w * 0.1

        # Zero line.
        if y_min < 0 < y_max:
            y_zero = plot_rect.bottom() - (
                (0.0 - y_min) / (y_max - y_min)
            ) * plot_rect.height()
            p.setPen(QPen(_AXIS, 1, Qt.PenStyle.DotLine))
            p.drawLine(plot_rect.left(), y_zero,
                       plot_rect.right(), y_zero)

        for i, (v_bar, v_ti) in enumerate(
                zip(data.y_values, data.secondary_y)):
            x_base = plot_rect.left() + i * group_w
            for j, (v, color) in enumerate(
                    [(v_bar, _ACCENT), (v_ti, _ERROR)]):
                x = x_base + j * (bar_w + gap) + gap / 2
                y_top = plot_rect.bottom() - (
                    (v - y_min) / (y_max - y_min)
                ) * plot_rect.height()
                bar = QRectF(x, y_top, bar_w, plot_rect.bottom() - y_top)
                p.fillRect(bar, QBrush(color))

        # Legend.
        legend_y = plot_rect.top() + 4
        p.setFont(QFont("Sans", 8))
        legend_x = plot_rect.right() - 200
        p.setBrush(QBrush(_ACCENT))
        p.drawRect(QRectF(legend_x, legend_y, 14, 10))
        p.setPen(QPen(_FG))
        p.drawText(QRectF(legend_x + 18, legend_y - 2, 80, 14),
                   Qt.AlignmentFlag.AlignLeft, "BAR")
        p.setBrush(QBrush(_ERROR))
        p.drawRect(QRectF(legend_x + 100, legend_y, 14, 10))
        p.setPen(QPen(_FG))
        p.drawText(QRectF(legend_x + 118, legend_y - 2, 80, 14),
                   Qt.AlignmentFlag.AlignLeft,
                   data.secondary_label or "TI")

        p.end()