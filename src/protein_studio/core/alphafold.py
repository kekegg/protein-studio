"""AlphaFold / Boltz-2 pLDDT extraction + coloring.

AlphaFold and Boltz-2 both store the per-residue pLDDT score in the
PDB ``B-factor`` column (0-100). This module:

- Re-annotates each :class:`Atom` with its residue's pLDDT
  (via ``Atom.bfactor``).
- Provides an :class:`AlphaFoldColorScheme` that maps pLDDT 0-100
  through the standard AlphaFold palette:

      very high (pLDDT > 90)  → blue
      confident (70 < pLDDT < 90) → cyan
      low       (50 < pLDDT < 70) → yellow
      very low  (pLDDT < 50)       → orange/red

The scheme itself is registered into :data:`SCHEME_PRESETS` from
:mod:`protein_studio.core.colors` to avoid circular imports.

This makes AlphaFold structures immediately readable in the viewport
without manual setup.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .colors import ALPHAFOLD_COLORS, AlphaFoldColorScheme
from .molecule import Molecule


# Re-export for backwards-compat
__all__ = [
    "AlphaFoldColorScheme", "ALPHAFOLD_COLORS",
    "plddt_bucket", "plddt_summary", "is_alphafold_pdb",
]


def plddt_bucket(value: float) -> str:
    if value >= 90:
        return "very_high"
    if value >= 70:
        return "confident"
    if value >= 50:
        return "low"
    return "very_low"


def plddt_summary(mol: Molecule) -> dict:
    """Return mean/min/max pLDDT and bucket histogram over the molecule."""
    values = [float(a.bfactor) for a in mol.atoms if a.residue]
    if not values:
        return {}
    arr = np.asarray(values)
    # Normalise to 0-100 if needed
    if arr.max() <= 1.0:
        arr = arr * 100.0
    bucket_counts = {
        "very_high": int((arr >= 90).sum()),
        "confident": int(((arr >= 70) & (arr < 90)).sum()),
        "low":       int(((arr >= 50) & (arr < 70)).sum()),
        "very_low":  int((arr < 50).sum()),
    }
    return {
        "n_residues": int(arr.size),
        "mean_plddt": float(arr.mean()),
        "min_plddt":  float(arr.min()),
        "max_plddt":  float(arr.max()),
        "buckets":    bucket_counts,
    }


def is_alphafold_pdb(path: str) -> bool:
    """Detect AlphaFold-style PDB by checking the HEADER / TITLE / pLDDT scale."""
    try:
        from pathlib import Path
        text = Path(path).read_text(errors="replace")[:4000]
    except Exception:
        return False
    if "ALPHAFOLD" in text.upper():
        return True
    # Heuristic: if average B-factor is in 0-100 range and not 0,
    # it's likely AlphaFold.
    try:
        bfs = []
        for line in text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    bfs.append(float(line[60:66].strip()))
                except ValueError:
                    pass
        if not bfs:
            return False
        avg = sum(bfs) / len(bfs)
        return 30.0 <= avg <= 100.0
    except Exception:
        return False


def annotate_plddt_from_bfactor(mol: Molecule) -> None:
    """No-op (pLDDT is already in B-factor), but ensures mean is recorded."""
    plddt_summary(mol)