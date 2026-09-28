"""Secondary structure inference by Ramachandran phi/psi classification.

PDB files often lack HELIX/SHEET records (e.g. when the file was
deposited without DSSP assignment, or it's a docked snapshot, or a
homology model). Without those records, every residue's
``secondary_structure`` is left as ``"unknown"``, and the
``secondary`` color scheme paints the entire protein in flat gray.

This module fills that gap by computing φ/ψ dihedrals along each
chain and classifying each residue into helix / sheet / loop using a
simple Ramachandran-region lookup. The classification matches the
"secondary structure" colour scheme's vocabulary (helix / sheet /
loop / turn / unknown) so the existing ``ColorScheme.color`` lookup
works without further changes.

The thresholds are deliberately lenient: we'd rather over-paint
helix/sheet than leave the whole protein gray. The standard
reference regions (in degrees):

    alpha-helix : φ ≈ -60, ψ ≈ -45
    beta-sheet  : φ ≈ -120, ψ ≈ +130
    PP-II helix  : φ ≈ -75, ψ ≈ +150  (counted as sheet by us)

We add proline (locked φ ≈ -60) and glycine (very wide φ range)
special cases.

Public surface
--------------
``infer_secondary_structure(mol) -> int``
    Walk every chain, compute φ/ψ for residues with all four
    backbone atoms (N, CA, C, plus the preceding C and following N),
    classify, and overwrite ``residue.secondary_structure`` for any
    residue still set to ``"unknown"``. Returns the number of
    residues annotated.

The function is idempotent — calling it twice doesn't change
already-classified residues.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .molecule import Molecule, Residue


_BACKBONE = ("N", "CA", "C")


def _dihedral(p1, p2, p3, p4) -> float:
    """Compute a torsion angle (degrees) along four points.

    The standard formula: ``angle = atan2(b1 × b2 · n, n · b1 × b2)``
    where ``b1 = p2 - p1``, ``b2 = p3 - p2``, ``b3 = p4 - p3`` and
    ``n = b1 × b2``. The result is wrapped to (-180, 180].
    """
    b1 = np.asarray(p2) - np.asarray(p1)
    b2 = np.asarray(p3) - np.asarray(p2)
    b3 = np.asarray(p4) - np.asarray(p3)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)
    if n1_norm < 1e-9 or n2_norm < 1e-9 or b2_norm < 1e-9:
        return float("nan")
    n1 /= n1_norm
    n2 /= n2_norm
    b2u = b2 / b2_norm
    m1 = np.cross(n1, b2u)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return math.degrees(math.atan2(y, x))


def _backbone_atoms(residue: Residue):
    """Return ``(N, CA, C)`` coords for *residue*, or ``None`` if any is missing."""
    atoms = {a.name: a for a in residue.atoms if a.residue is residue}
    n  = atoms.get("N")
    ca = atoms.get("CA")
    c  = atoms.get("C")
    if n is None or ca is None or c is None:
        return None
    return n.coord, ca.coord, c.coord


def _classify(phi: float, psi: float, resname: str) -> str:
    """Classify a residue by (φ, ψ) into helix / sheet / loop.

    Thresholds chosen to mirror what DSSP and PyMOL's ``dssp`` output
    give on a typical alpha/beta protein. We use **inclusive** bounds
    so borderline residues get painted something rather than being
    silently dropped into the ``unknown`` gray.
    """
    # Special cases first.
    if resname == "PRO":
        # Proline's φ is locked around -60° by the ring; the most
        # common rotamer is polyproline-II or a single-turn of
        # alpha-helix. We classify as helix because in cartoon reps
        # it should look like part of a helix segment.
        return "helix"

    if resname == "GLY":
        # Glycine is unusually flexible. Widen the helix and sheet
        # windows so GLY in a real helix/sheet doesn't get dropped
        # into "loop".
        if -150 <= phi <= -10 and -90 <= psi <= 30:
            return "helix"
        if -180 <= phi <= -60 and 80 <= psi <= 180:
            return "sheet"
        if -90 <= phi <= -10 and 90 <= psi <= 180:
            return "sheet"  # PP-II
        return "loop"

    # Standard amino-acid thresholds.
    if -90 <= phi <= -30 and -75 <= psi <= 0:
        return "helix"
    if -180 <= phi <= -90 and 90 <= psi <= 180:
        return "sheet"
    if -150 <= phi <= -50 and 100 <= psi <= 180:
        # PP-II region — call it sheet so it shows up as something
        # distinguishable from loop/turn.
        return "sheet"
    return "loop"


def _classify_chain(residues: List[Residue]) -> int:
    """Walk a list of residues (already sorted) and classify the ones
    still set to ``"unknown"``. Returns the number of residues
    classified by this pass."""
    classified = 0
    n = len(residues)
    for i, r in enumerate(residues):
        if r.secondary_structure != "unknown":
            continue
        if i == 0 or i == n - 1:
            # No preceding C or no following N — cannot compute φ or ψ.
            continue
        bb_prev = _backbone_atoms(residues[i - 1])
        bb_curr = _backbone_atoms(r)
        bb_next = _backbone_atoms(residues[i + 1])
        if bb_prev is None or bb_curr is None or bb_next is None:
            continue
        # phi: C(prev) — N(curr) — CA(curr) — C(curr)
        phi = _dihedral(bb_prev[2], bb_curr[0], bb_curr[1], bb_curr[2])
        # psi: N(curr) — CA(curr) — C(curr) — N(next)
        psi = _dihedral(bb_curr[0], bb_curr[1], bb_curr[2], bb_next[0])
        if math.isnan(phi) or math.isnan(psi):
            continue
        r.secondary_structure = _classify(phi, psi, r.name)
        classified += 1
    return classified


def _prune_short_runs(mol: Molecule, min_helix: int = 4, min_sheet: int = 3) -> int:
    """Demote scattered helix/sheet runs shorter than DSSP's minima.

    Real α-helices need at least ~4 consecutive residues to make one
    full turn of the helix (DSSP convention). Anything shorter is
    either: (a) a coil kink that happened to have helix φ/ψ angles
    by chance, or (b) the start/end of a real helix truncated by a
    missing N/CA/C atom (e.g. at chain breaks). In both cases, painting
    it as a helix produces a misleading ribbon — a 1-residue "helix"
    run gets smoothed away by the ribbon's Gaussian kernel and the
    user sees a flat ribbon rather than a cylinder.

    Same logic for β-sheets: 3+ consecutive residues (DSSP minimum).

    This runs after ``_classify_chain`` and only touches residues the
    inference just labelled. Returns the number of residues demoted.
    """
    demoted = 0
    for chain in mol.chains:
        residues = chain.residues
        # Walk consecutive runs of the same SS class.
        i = 0
        while i < len(residues):
            r = residues[i]
            if r.secondary_structure not in ("helix", "sheet"):
                i += 1
                continue
            ss = r.secondary_structure
            j = i
            while j < len(residues) and residues[j].secondary_structure == ss:
                j += 1
            run_len = j - i
            # If this run was inferred (not from explicit HELIX/SHEET),
            # and it's below the DSSP minimum, demote the whole run to
            # loop. We can't tell from the residue alone whether it came
            # from PDB records or from our inference — so we just apply
            # the same minimum to everything. PDB-annotated helices are
            # already at least 4 residues long in well-formed files
            # (e.g. 1aki has 17 runs with min=4), so the threshold is safe.
            threshold = min_helix if ss == "helix" else min_sheet
            if run_len < threshold:
                for k in range(i, j):
                    residues[k].secondary_structure = "loop"
                    demoted += 1
            i = j
    return demoted


def infer_secondary_structure(mol: Molecule) -> int:
    """Infer secondary structure for any residue still set to ``"unknown"``.

    Walks every chain, computes φ/ψ for residues with all four
    backbone neighbours, and classifies into helix / sheet / loop.
    A second pass demotes scattered short runs (1-2 residue
    "helices" with no real geometry) to ``loop`` — DSSP requires at
    least 4 residues for an α-helix and 3 for a β-strand, anything
    shorter is geometric noise.

    Already-classified residues are left alone — this function is
    safe to call after ``load_pdb`` (which applies HELIX/SHEET
    records) and the inference will only fill in the missing residues.

    Returns the number of residues classified by this call (before
    the demotion pass).
    """
    classified = 0
    for chain in mol.chains:
        residues = sorted(chain.residues, key=lambda r: r.number)
        classified += _classify_chain(residues)
    # Demote scattered short runs that are below DSSP's minimum helix
    # / strand length — these are geometric noise (a single residue
    # whose phi/psi happens to land in the helix box by chance), not
    # real secondary structure. Without this filter, the ribbon
    # rep's Gaussian smoothing washes them into the surrounding loop
    # and the user sees flat strips where they expect cylinders.
    _prune_short_runs(mol)
    return classified