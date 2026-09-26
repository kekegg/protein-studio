"""Color schemes for protein visualization.

Provides Blender/UMol-style palette presets:
- "element": CPK colors per element
- "residue": color by residue type (amino acid)
- "chain":  color by chain id
- "secondary": color by secondary structure (helix/sheet/loop)
- "bfactor": color by B-factor (temperature)
- "uniform": single color for everything
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


# ---- CPK (Corey-Pauling-Koltun) element colors ----
CPK_COLORS: Dict[str, tuple[float, float, float]] = {
    "H":  (1.00, 1.00, 1.00),
    "C":  (0.20, 0.20, 0.20),
    "N":  (0.19, 0.30, 0.85),
    "O":  (0.85, 0.15, 0.15),
    "S":  (0.95, 0.85, 0.15),
    "P":  (0.95, 0.55, 0.20),
    "F":  (0.40, 0.85, 0.40),
    "CL": (0.20, 0.85, 0.20),
    "BR": (0.65, 0.15, 0.05),
    "I":  (0.55, 0.20, 0.55),
    "FE": (0.85, 0.45, 0.05),
    "MG": (0.40, 0.75, 0.20),
    "CA": (0.45, 0.45, 0.55),
    "ZN": (0.45, 0.55, 0.65),
    "NA": (0.45, 0.45, 0.85),
    "K":  (0.45, 0.30, 0.85),
    "CU": (0.85, 0.55, 0.25),
    "MN": (0.55, 0.55, 0.85),
}

# ---- Residue type colors (Taylor-style) ----
RESIDUE_COLORS: Dict[str, tuple[float, float, float]] = {
    "ALA": (0.85, 0.85, 0.85), "ARG": (0.05, 0.55, 0.95),
    "ASN": (0.95, 0.45, 0.45), "ASP": (0.95, 0.20, 0.20),
    "CYS": (0.85, 0.85, 0.05), "GLN": (0.45, 0.85, 0.45),
    "GLU": (0.20, 0.65, 0.20), "GLY": (0.95, 0.95, 0.95),
    "HIS": (0.45, 0.45, 0.85), "ILE": (0.20, 0.20, 0.85),
    "LEU": (0.05, 0.05, 0.85), "LYS": (0.95, 0.20, 0.85),
    "MET": (0.85, 0.45, 0.05), "PHE": (0.45, 0.20, 0.45),
    "PRO": (0.85, 0.45, 0.85), "SER": (0.95, 0.55, 0.85),
    "THR": (0.45, 0.85, 0.85), "TRP": (0.45, 0.20, 0.85),
    "TYR": (0.45, 0.65, 0.20), "VAL": (0.20, 0.85, 0.85),
    "SEC": (0.95, 0.85, 0.45), "PYL": (0.45, 0.85, 0.45),
    "HET": (0.85, 0.45, 0.65),
}

# ---- Secondary structure colors ----
SS_COLORS: Dict[str, tuple[float, float, float]] = {
    "helix":    (0.95, 0.40, 0.20),  # orange-red  (alpha-helix)
    "sheet":    (0.95, 0.85, 0.20),  # yellow      (beta-sheet)
    "loop":     (0.55, 0.55, 0.85),  # lavender    (coil/loop)
    "turn":     (0.20, 0.85, 0.45),  # green       (turn)
    "unknown":  (0.75, 0.75, 0.75),  # gray        (unassigned)
}

# Standard amino acid 3-letter -> 1-letter
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

# Default scene background (Blender-style dark gray)
DEFAULT_BG = (0.20, 0.20, 0.20)
DEFAULT_FG = (0.85, 0.85, 0.85)


def color_for_element(element: str, fallback: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> tuple[float, float, float]:
    """CPK color for an element symbol (case-insensitive)."""
    return CPK_COLORS.get(element.upper(), fallback)


def color_for_residue(resname: str, fallback: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> tuple[float, float, float]:
    """Standard residue type color."""
    return RESIDUE_COLORS.get(resname.upper(), fallback)


def color_for_chain(chain_id: str, palette: Iterable[tuple[float, float, float]] | None = None) -> tuple[float, float, float]:
    """Deterministic chain color from a palette (default 8-color)."""
    palette = palette or _CHAIN_PALETTE
    if not chain_id:
        return palette[0]
    return palette[ord(chain_id[0]) % len(palette)]


# A simple 8-color rainbow palette used to color chains deterministically
_CHAIN_PALETTE = (
    (0.85, 0.20, 0.20),  # red
    (0.20, 0.55, 0.95),  # blue
    (0.20, 0.75, 0.20),  # green
    (0.95, 0.65, 0.20),  # orange
    (0.65, 0.20, 0.85),  # purple
    (0.20, 0.85, 0.85),  # cyan
    (0.85, 0.85, 0.20),  # yellow
    (0.85, 0.40, 0.65),  # pink
)


@dataclass(frozen=True)
class ColorScheme:
    """A named color scheme that maps a per-atom key to an RGB color."""
    name: str
    key: str  # "element" | "residue" | "chain" | "secondary" | "bfactor" | "uniform" | "electrostatic"
    uniform: tuple[float, float, float] = (0.80, 0.80, 0.80)
    bfactor_range: tuple[float, float] = (0.0, 100.0)

    def color(self, atom) -> tuple[float, float, float]:  # noqa: ANN001
        if self.key == "uniform":
            return self.uniform
        if self.key == "element":
            return color_for_element(getattr(atom, "element", "C"))
        if self.key == "residue":
            return color_for_residue(getattr(atom, "residue_name", ""))
        if self.key == "chain":
            return color_for_chain(getattr(atom, "chain_id", ""))
        if self.key == "secondary":
            ss = getattr(atom, "secondary_structure", "unknown")
            return SS_COLORS.get(ss, SS_COLORS["unknown"])
        if self.key == "bfactor":
            bf = float(getattr(atom, "bfactor", 0.0))
            lo, hi = self.bfactor_range
            t = 0.0 if hi <= lo else (bf - lo) / (hi - lo)
            t = max(0.0, min(1.0, t))
            # Blue (low) -> white (mid) -> red (high)
            return _bfactor_color(t)
        if self.key == "electrostatic":
            # Per-atom electrostatic color is meaningless — fall back to
            # uniform. The rep function (esp_surface) handles per-vertex
            # potential → RWB coloring when this scheme is active.
            return self.uniform
        return self.uniform


def _bfactor_color(t: float) -> tuple[float, float, float]:
    """Blue -> white -> red gradient based on t in [0,1]."""
    if t < 0.5:
        s = t * 2.0
        return (s, s, 1.0)
    s = (t - 0.5) * 2.0
    return (1.0, 1.0 - s, 1.0 - s)


# Color schemes selectable from the UI
SCHEME_PRESETS: Dict[str, ColorScheme] = {
    "uniform":       ColorScheme("Uniform",       "uniform",   uniform=(0.80, 0.80, 0.85)),
    "element":       ColorScheme("Element",       "element"),
    "residue":       ColorScheme("Residue",       "residue"),
    "chain":         ColorScheme("Chain",         "chain"),
    "secondary":     ColorScheme("Secondary",     "secondary"),
    "bfactor":       ColorScheme("B-factor",      "bfactor"),
    "electrostatic": ColorScheme("Electrostatic", "electrostatic"),
}


# ----- AlphaFold pLDDT palette (defined inline to avoid circular imports) -----

ALPHAFOLD_COLORS: Dict[str, tuple[float, float, float]] = {
    "very_high":   (0.12, 0.53, 0.90),  # royal blue   (pLDDT > 90)
    "confident":   (0.40, 0.76, 0.65),  # teal/cyan    (70-90)
    "low":         (0.95, 0.76, 0.06),  # amber        (50-70)
    "very_low":    (0.85, 0.38, 0.31),  # brick red    (<50)
}


def _plddt_bucket(value: float) -> str:
    if value >= 90:
        return "very_high"
    if value >= 70:
        return "confident"
    if value >= 50:
        return "low"
    return "very_low"


class AlphaFoldColorScheme:
    """A scheme that interprets ``bfactor`` as pLDDT 0-100.

    Plain object (not a dataclass) so we can swap in a custom ``color``
    method. Exposes the same attributes that :class:`ColorScheme` does:
    ``name``, ``key``, ``uniform``, ``bfactor_range``, ``color``.
    """
    name: str = "AlphaFold pLDDT"
    key: str = "alphafold_plddt"
    uniform: tuple[float, float, float] = (0.5, 0.5, 0.5)
    bfactor_range: tuple[float, float] = (0.0, 100.0)

    def color(self, atom) -> tuple[float, float, float]:  # noqa: ANN001
        bf = float(getattr(atom, "bfactor", 0.0))
        if bf <= 1.0:
            bf *= 100.0
        return ALPHAFOLD_COLORS[_plddt_bucket(bf)]


SCHEME_PRESETS["alphafold_plddt"] = AlphaFoldColorScheme()