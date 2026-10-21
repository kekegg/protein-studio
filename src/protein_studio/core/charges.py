"""Partial-charge assignment + electrostatic-surface computation.

The aim is *not* to reproduce APBS-quality electrostatics — for that
you would need a Poisson–Boltzmann solver. Instead, we:

1. Assign per-atom charges from the AMBER ff99SB-style table
   (see ``_AMBER_PARTIAL_CHARGES``) — ported from 3Dmol.js's
   ``partialCharges.ts`` so behavior matches 3Dmol.js / PyMOL.
2. Sample the simplified Coulomb potential

       V(r) = sum_i  q_i / max(|r - r_i|, cutoff)

   on a molecular surface. The result is qualitatively correct
   (negative for acidic side chains, positive for basic ones) and
   runs in well under a second on ubiquitin.
"""

# -----------------------------------------------------------------------------
# Portions of this module are derived from 3Dmol.js (BSD-3-Clause).
#
# Upstream: 3Dmol.js src/parsers/utils/partialCharges.ts
#
#   3Dmol.js
#   Copyright (c) 2014, University of Pittsburgh and contributors
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#   3. Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .molecule import Atom, Molecule


# ============================================================ AMBER-style table
#
# Per-residue per-atom partial charges, in elementary-charge units.
# Ported from 3Dmol.js's partialCharges.ts (which itself cites the
# AMBER ff99SB force field). Lookup is keyed by ``f"{residue}:{atom}"``
# where residue is the 3-letter name and atom is the PDB atom name.
# These are NOT directly usable for MD — they're for visualization.

_AMBER_PARTIAL_CHARGES: dict[str, float] = {
    # ALA
    "ALA:N": -0.15, "ALA:CA": 0.10, "ALA:CB": 0.00,
    "ALA:C":  0.60, "ALA:O": -0.55,
    # ARG
    "ARG:N": -0.15, "ARG:CA": 0.10, "ARG:CB": 0.00,
    "ARG:CG": 0.00, "ARG:CD": 0.10, "ARG:NE": -0.10,
    "ARG:CZ": 0.50, "ARG:NH1": 0.25, "ARG:NH2": 0.25,
    "ARG:C":  0.60, "ARG:O": -0.55,
    # ASN
    "ASN:N": -0.15, "ASN:CA": 0.10, "ASN:CB": 0.00,
    "ASN:CG": 0.55, "ASN:OD1": -0.55, "ASN:ND2": 0.00,
    "ASN:C":  0.60, "ASN:O": -0.55,
    # ASP
    "ASP:N": -0.15, "ASP:CA": 0.10, "ASP:CB": 0.00,
    "ASP:CG": 0.14, "ASP:OD1": -0.57, "ASP:OD2": -0.57,
    "ASP:C":  0.60, "ASP:O": -0.55,
    # CYS
    "CYS:N": -0.15, "CYS:CA": 0.10, "CYS:CB": 0.19,
    "CYS:SG": -0.19, "CYS:C":  0.60, "CYS:O": -0.55,
    # GLN
    "GLN:N": -0.15, "GLN:CA": 0.10, "GLN:CB": 0.00,
    "GLN:CG": 0.00, "GLN:CD": 0.55, "GLN:OE1": -0.55,
    "GLN:NE2": 0.00, "GLN:C":  0.60, "GLN:O": -0.55,
    # GLU
    "GLU:N": -0.15, "GLU:CA": 0.10, "GLU:CB": 0.00,
    "GLU:CG": 0.00, "GLU:CD": 0.14,
    "GLU:OE1": -0.57, "GLU:OE2": -0.57,
    "GLU:C":  0.60, "GLU:O": -0.55,
    # GLY
    "GLY:N": -0.15, "GLY:CA": 0.10, "GLY:C":  0.60, "GLY:O": -0.55,
    # HIS
    "HIS:N": -0.15, "HIS:CA": 0.10, "HIS:CB": 0.00,
    "HIS:CG": 0.10, "HIS:ND1": -0.10, "HIS:CD2": 0.10,
    "HIS:NE2": -0.40, "HIS:CE1": 0.30,
    "HIS:C":  0.60, "HIS:O": -0.55,
    # ILE
    "ILE:N": -0.15, "ILE:CA": 0.10, "ILE:CB": 0.00,
    "ILE:CG2": 0.00, "ILE:CG1": 0.00, "ILE:CD": 0.00,
    "ILE:C":  0.60, "ILE:O": -0.55,
    # LEU
    "LEU:N": -0.15, "LEU:CA": 0.10, "LEU:CB": 0.00,
    "LEU:CG": 0.00, "LEU:CD1": 0.00, "LEU:CD2": 0.00,
    "LEU:C":  0.60, "LEU:O": -0.55,
    # LYS
    "LYS:N": -0.15, "LYS:CA": 0.10, "LYS:CB": 0.00,
    "LYS:CG": 0.00, "LYS:CD": 0.00, "LYS:CE": 0.25,
    "LYS:NZ": 0.75, "LYS:C":  0.60, "LYS:O": -0.55,
    # MET
    "MET:N": -0.15, "MET:CA": 0.10, "MET:CB": 0.00,
    "MET:CG": 0.06, "MET:SD": -0.12, "MET:CE": 0.06,
    "MET:C":  0.60, "MET:O": -0.55,
    # PHE
    "PHE:N": -0.15, "PHE:CA": 0.10, "PHE:CB": 0.00,
    "PHE:CG": 0.00, "PHE:CD1": 0.00, "PHE:CD2": 0.00,
    "PHE:CE1": 0.00, "PHE:CE2": 0.00, "PHE:CZ": 0.00,
    "PHE:C":  0.60, "PHE:O": -0.55,
    # PRO
    "PRO:N": -0.25, "PRO:CD": 0.10, "PRO:CA": 0.10,
    "PRO:CB": 0.00, "PRO:CG": 0.00, "PRO:C":  0.60, "PRO:O": -0.55,
    # SER
    "SER:N": -0.15, "SER:CA": 0.10, "SER:CB": 0.25,
    "SER:OG": -0.25, "SER:C":  0.60, "SER:O": -0.55,
    # THR
    "THR:N": -0.15, "THR:CA": 0.10, "THR:CB": 0.25,
    "THR:OG1": -0.25, "THR:CG2": 0.00,
    "THR:C":  0.60, "THR:O": -0.55,
    # TRP
    "TRP:N": -0.15, "TRP:CA": 0.10, "TRP:CB": 0.00,
    "TRP:CG": -0.03, "TRP:CD2": 0.10, "TRP:CE2": -0.04,
    "TRP:CE3": -0.03, "TRP:CD1": 0.06, "TRP:NE1": -0.06,
    "TRP:CZ2": 0.00, "TRP:CZ3": 0.00, "TRP:CH2": 0.00,
    "TRP:C":  0.60, "TRP:O": -0.55,
    # TYR
    "TYR:N": -0.15, "TYR:CA": 0.10, "TYR:CB": 0.00,
    "TYR:CG": 0.00, "TYR:CD1": 0.00, "TYR:CE1": 0.00,
    "TYR:CD2": 0.00, "TYR:CE2": 0.00, "TYR:CZ": 0.25,
    "TYR:OH": -0.25, "TYR:C":  0.60, "TYR:O": -0.55,
    # VAL
    "VAL:N": -0.15, "VAL:CA": 0.10, "VAL:CB": 0.00,
    "VAL:CG1": 0.00, "VAL:CG2": 0.00,
    "VAL:C":  0.60, "VAL:O": -0.55,
}


# Element-level fallback for atoms not covered by the AMBER table
# (heteroatoms, ions, hydrogens, missing sidechains).
_DEFAULT_PARTIAL_CHARGES: dict[str, float] = {
    "H":  0.05,
    "C":  0.00,
    "N": -0.30,
    "O": -0.40,
    "S": -0.15,
    "P":  0.30,
    "F": -0.20,
    "CL": -0.10,
    "BR": -0.10,
    "I": -0.10,
    "FE": 0.20,
    "MG": 0.50,
    "CA": 1.00,
    "ZN": 0.40,
    "NA": 1.00,
    "K":  1.00,
    "CU": 0.20,
    "MN": 0.20,
}


def assign_partial_charges(
    mol: Molecule,
    *,
    prefer_amber: bool = True,
    fallback_to_gasteiger: bool = True,
) -> np.ndarray:
    """Return a per-atom partial-charge array (length n_atoms).

    Lookup order for each atom:

    1. ``_AMBER_PARTIAL_CHARGES[f"{resn}:{atom}"]`` (only if
       ``prefer_amber`` is true and the residue name is a standard
       amino acid).
    2. ``_DEFAULT_PARTIAL_CHARGES[element]`` — coarse element-wise
       heuristic (only if ``fallback_to_gasteiger`` is true).
    3. 0.0 if neither matches.

    Behaviour matches 3Dmol.js's ``applyPartialCharges()`` when
    ``prefer_amber=True``, ``fallback_to_gasteiger=True``.
    """
    STANDARD_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU "
                      "LYS MET PHE PRO SER THR TRP TYR VAL".split())
    out = np.zeros(mol.n_atoms(), dtype=float)
    atoms = mol.atoms
    for i, a in enumerate(atoms):
        resn = (a.residue_name or "").upper()
        atom_name = (a.name or "").upper()
        # 1) AMBER table — only for standard amino acids, protein atoms
        if prefer_amber and resn in STANDARD_AA and not a.is_hetero:
            key = f"{resn}:{atom_name}"
            if key in _AMBER_PARTIAL_CHARGES:
                out[i] = _AMBER_PARTIAL_CHARGES[key]
                continue
        # 2) Element-wise fallback (Gasteiger-ish heuristic)
        if fallback_to_gasteiger:
            out[i] = _DEFAULT_PARTIAL_CHARGES.get(a.element.upper(), 0.0)
        # 3) else 0.0
    return out


# ---------------------------------------------------------------- surface potential


@dataclass
class ESPConfig:
    """Electrostatic-surface computation parameters."""
    probe_radius: float = 1.4
    n_sample_points: int = 4000    # surface samples
    decay: float = 1.5             # 1.0 = pure Coulomb, 2.0 = screened / sharper
    min_distance: float = 0.7      # avoid singularity at atom centers
    scale_kcal: float = 2.5        # scale e / eps to nicer color range
    include_hetero: bool = False   # ignore HETATM in ESP sum (HEM etc.)


def compute_esp_surface(
    mol: Molecule,
    charges: Optional[np.ndarray] = None,
    config: Optional[ESPConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a coarse molecular surface colored by approximate ESP.

    Returns ``(points, potential)`` where ``points`` is an (N, 3) array
    of surface samples and ``potential`` is the per-point ESP value.

    Implementation: build an SES mesh using the EDTSurf algorithm,
    then sample the simplified Coulomb potential on every vertex.
    If the mesh has more vertices than ``config.n_sample_points`` we
    subsample uniformly to keep the call fast.
    """
    cfg = config or ESPConfig()
    from .surface import generate_ses_surface, SURFACE_MS

    atoms_all = mol.atoms
    if cfg.include_hetero:
        atoms = atoms_all
    else:
        atoms = [a for a in atoms_all if not a.is_hetero]
    if not atoms:
        return np.zeros((0, 3)), np.zeros(0)
    coords = np.array([a.coord for a in atoms], dtype=float)
    elements = [a.element for a in atoms]

    ses = generate_ses_surface(
        coords, elements,
        probe_radius=cfg.probe_radius,
        surface_type=SURFACE_MS,
    )
    pts = ses.vertices
    if len(pts) == 0:
        return pts, np.zeros(0)

    # Subsample if the mesh is too big
    if len(pts) > cfg.n_sample_points:
        idx = np.linspace(0, len(pts) - 1, cfg.n_sample_points).astype(int)
        pts = pts[idx]

    # Per-atom charges for the participating atoms
    if charges is None:
        q_mol = assign_partial_charges(mol)
        if not cfg.include_hetero:
            mask = np.array([not a.is_hetero for a in atoms_all])
            q = q_mol[mask]
        else:
            q = q_mol
    else:
        q = charges

    # Vectorized Coulomb sum
    diff = pts[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, cfg.min_distance)
    pot = (q[None, :] / np.power(dist, cfg.decay)).sum(axis=1)
    # Saturate
    pot = np.tanh(pot / cfg.scale_kcal)
    return pts, pot


class _DummyUniformScheme:
    """All-grey scheme used internally for the ESP surface point cloud."""
    key = "uniform"
    name = "Uniform"
    uniform = (0.7, 0.7, 0.7)

    def color(self, atom):
        return self.uniform


# ----------------------------------------------------- Red-white-blue potential


def esp_to_rgb(potential: np.ndarray) -> np.ndarray:
    """Map potential values in [-1, 1] to RGB via red-white-blue gradient.

    Negative  → red
    Neutral   → white
    Positive  → blue
    """
    t = np.clip(np.asarray(potential, dtype=float), -1.0, 1.0)
    out = np.zeros((len(t), 3), dtype=float)
    if len(t) == 0:
        return out
    neg = t < 0
    pos = t > 0
    # Red gradient (negative potential)
    out[neg, 0] = 1.0
    out[neg, 1] = 1.0 + t[neg]
    out[neg, 2] = 1.0 + t[neg]
    # Blue gradient (positive potential)
    out[pos, 0] = 1.0 - t[pos]
    out[pos, 1] = 1.0 - t[pos]
    out[pos, 2] = 1.0
    # Neutral
    neutral = ~(neg | pos)
    out[neutral] = (1.0, 1.0, 1.0)
    return np.clip(out, 0.0, 1.0)