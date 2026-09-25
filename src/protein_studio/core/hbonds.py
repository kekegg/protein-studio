"""Hydrogen-bond detection + arbitrary distance measurement.

Implements a Baker-Hubbard-style geometric criterion:

    d(N..O) < 3.5 Å  AND  angle(D-H..A) > 120°

Donors: backbone N, sidechain N/O of ASN/GLN/HIS/SER/THR/TYR/TRP/LYS/ARG.
Acceptors: backbone O, sidechain O of ASN/ASP/GLN/GLU/HIS/SER/THR/TYR.

For distance measurement between arbitrary atoms we expose
:func:`measure_distance`, which returns the Euclidean distance plus the
two atom descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .molecule import Atom, Molecule


DONOR_RESIDUES = {
    "ASN", "GLN", "HIS", "SER", "THR", "TYR", "TRP", "LYS", "ARG", "CYS",
}
ACCEPTOR_RESIDUES = {
    "ASN", "ASP", "GLN", "GLU", "HIS", "SER", "THR", "TYR",
}
HBOND_DIST_CUTOFF = 3.5
HBOND_ANGLE_CUTOFF_DEG = 120.0


@dataclass
class HBond:
    """One detected hydrogen bond."""
    donor: Atom
    hydrogen_proxy: Optional[Atom]  # None for back-of-envelope detection
    acceptor: Atom
    distance: float
    angle_deg: Optional[float]


@dataclass
class Distance:
    """Distance between two arbitrary atoms."""
    a: Atom
    b: Atom
    distance: float

    def __repr__(self) -> str:
        return (
            f"{self.a.residue_name}{self.a.residue_number}.{self.a.name}"
            f"(chain {self.a.chain_id}) ↔ "
            f"{self.b.residue_name}{self.b.residue_number}.{self.b.name}"
            f"(chain {self.b.chain_id}): {self.distance:.3f} Å"
        )


def _is_donor(atom: Atom) -> bool:
    if atom.is_hetero:
        return False
    if atom.name == "N":
        return True  # backbone
    if atom.element != "N":
        return False
    return atom.residue_name.upper() in DONOR_RESIDUES


def _is_acceptor(atom: Atom) -> bool:
    if atom.is_hetero:
        # Accept HETATM hetero O (ligands, water) too
        return atom.element == "O"
    if atom.name == "O":
        return True  # backbone carbonyl
    if atom.element != "O":
        return False
    return atom.residue_name.upper() in ACCEPTOR_RESIDUES


def detect_hbonds(mol: Molecule,
                  dist_cutoff: float = HBOND_DIST_CUTOFF,
                  angle_cutoff: float = HBOND_ANGLE_CUTOFF_DEG,
                  include_hetero: bool = False) -> List[HBond]:
    """Find all geometric hydrogen bonds.

    Hydrogens are usually not present in X-ray PDB files, so we use a
    common trick: place the donor H along the donor-acceptor line at
    1.0 Å from the donor. This gives a usable angle estimate.
    """
    donors = [a for a in mol.atoms if _is_donor(a)]
    acceptors = [a for a in mol.atoms if _is_acceptor(a) and (include_hetero or not a.is_hetero)]
    if not donors or not acceptors:
        return []

    out: List[HBond] = []
    d_coords = np.array([a.coord for a in donors], dtype=float)
    a_coords = np.array([a.coord for a in acceptors], dtype=float)

    # All donor-acceptor pairs within cutoff (O(N*M) but small for proteins)
    for di, d in enumerate(donors):
        diff = a_coords - d_coords[di]
        d2 = np.einsum("ij,ij->i", diff, diff)
        cand = np.where(d2 < dist_cutoff ** 2)[0]
        if len(cand) == 0:
            continue
        for ai in cand:
            acceptor = acceptors[ai]
            dist = float(np.sqrt(d2[ai]))
            # Place proxy H along D→A vector at 1.0 Å from donor
            if dist > 1e-6:
                h_proxy = d.coord + (acceptor.coord - d.coord) * (1.0 / dist)
            else:
                h_proxy = d.coord
            # Angle D-H..A at the donor
            v1 = acceptor.coord - h_proxy
            v2 = d.coord - h_proxy
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = float(np.dot(v1, v2) / (n1 * n2))
            cos_a = max(-1.0, min(1.0, cos_a))
            angle_deg = float(np.degrees(np.arccos(cos_a)))
            if angle_deg >= angle_cutoff:
                out.append(HBond(
                    donor=d, hydrogen_proxy=None,
                    acceptor=acceptor, distance=dist, angle_deg=angle_deg,
                ))
    return out


def hbonds_as_lines(hbonds: List[HBond]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of HBond to (coords, lines) arrays usable by PyVista."""
    coords = []
    lines = []
    for i, hb in enumerate(hbonds):
        coords.append(hb.donor.coord)
        coords.append(hb.acceptor.coord)
        lines.extend([2, 2 * i, 2 * i + 1])
    if not coords:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    return np.asarray(coords, dtype=float), np.asarray(lines, dtype=int)


def measure_distance(atom_a: Atom, atom_b: Atom) -> Distance:
    """Euclidean distance between two atoms."""
    d = float(np.linalg.norm(atom_a.coord - atom_b.coord))
    return Distance(a=atom_a, b=atom_b, distance=d)


def hbonds_representation(mol: Molecule, **kwargs) -> "object":
    """Render detected hbonds as dashed lines between donor and acceptor."""
    from .representations import _safe_import_pyvista
    pv = _safe_import_pyvista()
    hbs = detect_hbonds(mol, **kwargs)
    coords, lines = hbonds_as_lines(hbs)
    if len(coords) == 0:
        return pv.PolyData()
    poly = pv.PolyData(coords, lines=lines)
    # Color: yellow by default
    color = np.tile(np.array([0.95, 0.85, 0.20], dtype=float), (len(hbs), 1))
    poly.cell_data["colors"] = color
    return poly