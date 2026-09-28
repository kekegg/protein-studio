"""Secondary structure detection by backbone hydrogen bonds.

Direct port of 3Dmol.js's ``computeSecondaryStructure`` +
``assignBackboneHBonds`` algorithm (see
``3Dmol.js-master/src/parsers/utils/computeSecondaryStructure.ts``).
The basic idea:

  * An alpha-helix is **defined** by an i→i+4 backbone N-H...O=C hydrogen
    bond — that is the physical structure. We detect those by finding
    the closest inter-residue N...O contact between backbone N and O
    atoms in the same chain.
  * A beta-sheet is detected by paired N...O contacts between two
    residues that aren't already helices.
  * Real SS lives in clusters of 4+ (helix) or 3+ (strand) residues;
    everything else is loop.

This is a strict improvement over φ/ψ-threshold classification
(which calls any single residue whose backbone dihedrals happen to
fall in the helix box a "helix") and matches the accuracy of
DSSP's geometric core without needing H-atom coordinates.

Public surface
--------------
``assign_backbone_hbonds(mol, hbond_cutoff=3.2)``
    For each backbone N and O atom in the molecule (skipping HETATM),
    find the closest non-trivial N...O contact (skipping same-chain
    within-3-residue neighbours). Sets ``atom.hbond_other`` and
    ``atom.hbond_distance_sq`` on every backbone N/O atom.

``compute_secondary_structure_from_hbonds(mol, hbond_cutoff=3.2)``
    Run hbond detection, then apply 3Dmol.js's classification:
    helix if hbonds to residue ±4, sheet if hbonds paired with another
    sheet, gap-plug 1-residue gaps, drop singletons. Writes the
    ``secondary_structure`` field on every residue.

The phi/psi based ``infer_secondary_structure`` (also in this module)
is kept as a fallback for molecules that lack backbone N/O atoms
(very short peptides, fully-heteroatom ligands, etc.).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .molecule import Molecule, Residue


HBOND_CUTOFF_DEFAULT = 3.2  # Å — same value 3Dmol.js uses.


def _backbone_no_atoms(mol: Molecule):
    """Return all backbone N / O atoms (skipping HETATM, skipping
    alternative conformations / duplicate atoms that share the same
    (residue, name) key — keep only the first one)."""
    seen: set = set()
    out = []
    for a in mol.atoms:
        if a.is_hetero:
            continue
        if a.residue is None:
            continue
        if a.name not in ("N", "O"):
            continue
        key = (a.residue.chain_id, a.residue.number, a.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _assign_backbone_hbonds(mol: Molecule, hbond_cutoff: float) -> None:
    """For each backbone N / O atom, find the closest neighbour
    outside its 4-residue window. Side-effect: sets ``hbond_other`` and
    ``hbond_distance_sq`` on every backbone N/O atom.

    This is the 3Dmol.js ``assignBackboneHBonds`` algorithm: sort
    backbone atoms by z, then for each pair in a 3.2 Å cube skip
    pairs in the same chain within 4 residues and pick the closest
    one. The O(n * k) sweep is fine for n in the thousands.
    """
    max_sq = hbond_cutoff * hbond_cutoff
    atoms = _backbone_no_atoms(mol)
    # Reset any previous state.
    for a in atoms:
        a.hbond_other = None
        a.hbond_distance_sq = math.inf
    atoms.sort(key=lambda a: a.coord[2])

    n = len(atoms)
    for i in range(n):
        ai = atoms[i]
        for j in range(i + 1, n):
            aj = atoms[j]
            if aj.coord[2] - ai.coord[2] > hbond_cutoff:
                break  # sorted by z — nothing else can be closer
            if abs(aj.coord[1] - ai.coord[1]) > hbond_cutoff:
                continue
            if abs(aj.coord[0] - ai.coord[0]) > hbond_cutoff:
                continue
            # Skip same-element pairs (N...N or O...O) — 3Dmol.js skips
            # them so an N doesn't end up bonded to another N.
            if aj.name == ai.name:
                continue
            # Skip close residues in the same chain.
            if (ai.residue is not None and aj.residue is not None
                    and ai.residue.chain_id == aj.residue.chain_id
                    and abs(ai.residue.number - aj.residue.number) < 4):
                continue
            dx = aj.coord[0] - ai.coord[0]
            dy = aj.coord[1] - ai.coord[1]
            dz = aj.coord[2] - ai.coord[2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 > max_sq:
                continue
            if d2 < ai.hbond_distance_sq:
                ai.hbond_other = aj
                ai.hbond_distance_sq = d2
            if d2 < aj.hbond_distance_sq:
                aj.hbond_other = ai
                aj.hbond_distance_sq = d2


def _classify_from_hbonds(mol: Molecule) -> int:
    """3Dmol.js-style classification: helix = i↔i±4 H-bond, sheet =
    paired H-bonds (both ends non-helix). Gap-plugs single-residue
    holes and drops singletons. Returns the number of residues whose
    ``secondary_structure`` was newly set.

    The molecule's atoms must have ``hbond_other`` / ``hbond_distance_sq``
    populated (run ``_assign_backbone_hbonds`` first).
    """
    # Group residues by chain (preserving order).
    chains = [c for c in mol.chains]
    if not chains:
        return 0

    # Per-chain lookup of residue-by-number → working SS string.
    chres: Dict[str, List[str]] = {c.chain_id: [] for c in chains}
    for c in chains:
        residues = sorted(c.residues, key=lambda r: r.number)
        # Build the indexable list — gaps in res numbering mean sparse
        # entries that we'll patch up with "loop".
        max_resnum = max((r.number for r in residues), default=0)
        lookup: List[Optional[str]] = [None] * (max_resnum + 1)
        for r in residues:
            # Pad from previous number up to this one.
            for k in range(len(lookup), r.number + 1):
                lookup.append(None)
            lookup[r.number] = None  # unknown by default
        chres[c.chain_id] = lookup

    # ---- 1. Helix pass: any N/O with an H-bond to residue ±4 → helix
    for a in mol.atoms:
        if a.name not in ("N", "O"):
            continue
        if a.hbond_other is None or not math.isfinite(a.hbond_distance_sq):
            continue
        other = a.hbond_other
        if (other.residue is None or a.residue is None
                or other.residue.chain_id != a.residue.chain_id):
            continue
        if abs(other.residue.number - a.residue.number) != 4:
            continue
        # Mark BOTH the residue this atom is on AND its partner as helix.
        chres[a.residue.chain_id][a.residue.number] = "h"
        chres[other.residue.chain_id][other.residue.number] = "h"

    # ---- 2. Plug single-residue gaps in helix runs.
    for cid, lookup in chres.items():
        for r in range(1, len(lookup) - 1):
            before = lookup[r - 1]
            after = lookup[r + 1]
            if before == "h" and before == after and lookup[r] != before:
                lookup[r] = before

    # ---- 3. Mark potential sheets: any remaining H-bonded residue
    # that isn't already helix becomes a sheet "candidate".
    # Track which atoms contributed so we can determine "is sheet"
    # by paired candidates (3Dmol.js: sheet only if both ends
    # of the H-bond pair are candidates or already sheet).
    candidate_residues: set = set()
    for a in mol.atoms:
        if a.name not in ("N", "O"):
            continue
        if a.hbond_other is None or not math.isfinite(a.hbond_distance_sq):
            continue
        cid = a.residue.chain_id if a.residue else None
        if cid is None:
            continue
        if chres[cid][a.residue.number] == "h":
            continue
        candidate_residues.add((cid, a.residue.number))

    # ---- 4. Promote candidates to sheet if both ends of their
    # H-bond pair are also candidates (or already sheet).
    sheet_set: set = set()
    for a in mol.atoms:
        if a.name not in ("N", "O"):
            continue
        if a.hbond_other is None or not math.isfinite(a.hbond_distance_sq):
            continue
        cid = a.residue.chain_id
        other_cid = a.hbond_other.residue.chain_id
        other_rnum = a.hbond_other.residue.number
        rnum = a.residue.number
        if (cid, rnum) not in candidate_residues:
            continue
        other_val = chres[other_cid][other_rnum]
        if other_val == "s" or (other_cid, other_rnum) in candidate_residues:
            chres[cid][rnum] = "s"
            sheet_set.add((cid, rnum))
            chres[other_cid][other_rnum] = "s"
            sheet_set.add((other_cid, other_rnum))

    # ---- 5. Plug single-residue gaps in sheets.
    for cid, lookup in chres.items():
        for r in range(1, len(lookup) - 1):
            before = lookup[r - 1]
            after = lookup[r + 1]
            if before == "s" and before == after and lookup[r] != before:
                lookup[r] = before

    # ---- 6. Drop singletons: a single helix/sheet residue flanked
    # by anything else gets demoted to loop.
    for cid, lookup in chres.items():
        for r in range(len(lookup)):
            v = lookup[r]
            if v not in ("h", "s"):
                continue
            prev_v = lookup[r - 1] if r > 0 else None
            next_v = lookup[r + 1] if r + 1 < len(lookup) else None
            if prev_v != v and next_v != v:
                lookup[r] = None

    # ---- 7. Write back to residues (only those still 'unknown').
    n_set = 0
    for c in chains:
        residues = sorted(c.residues, key=lambda r: r.number)
        for r in residues:
            if r.secondary_structure != "unknown":
                continue
            v = chres[c.chain_id][r.number]
            if v in ("h", "s"):
                r.secondary_structure = ("helix" if v == "h" else "sheet")
                n_set += 1
    return n_set


def compute_secondary_structure_from_hbonds(
    mol: Molecule, hbond_cutoff: float = HBOND_CUTOFF_DEFAULT,
) -> int:
    """Run the 3Dmol.js H-bond SS detection pipeline.

    Returns the number of residues newly annotated. Already-labelled
    residues (from PDB HELIX/SHEET records or from a previous call) are
    left alone.
    """
    _assign_backbone_hbonds(mol, hbond_cutoff)
    return _classify_from_hbonds(mol)


# ---------------------------------------------------------------------------
# Phi/psi based inference (kept as a fallback for molecules with too few
# backbone N/O atoms — e.g. very short peptides, or fully-heteroatom
# ligands with no backbone).
# ---------------------------------------------------------------------------


_BACKBONE = ("N", "CA", "C")


def _dihedral(p1, p2, p3, p4) -> float:
    """Compute a torsion angle (degrees) along four points.

    Standard formula: ``angle = atan2(b1 × b2 · n, n · b1 × b2)`` where
    ``b1 = p2 - p1``, ``b2 = p3 - p2``, ``b3 = p4 - p3`` and ``n = b1 × b2``.
    The result is wrapped to (-180, 180]. Colinear / degenerate inputs
    return NaN.
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
    """Classify a residue by (φ, ψ) into helix / sheet / loop."""
    if resname == "PRO":
        return "helix"
    if resname == "GLY":
        if -150 <= phi <= -10 and -90 <= psi <= 30:
            return "helix"
        if -180 <= phi <= -60 and 80 <= psi <= 180:
            return "sheet"
        if -90 <= phi <= -10 and 90 <= psi <= 180:
            return "sheet"
        return "loop"
    if -90 <= phi <= -30 and -75 <= psi <= 0:
        return "helix"
    if -180 <= phi <= -90 and 90 <= psi <= 180:
        return "sheet"
    if -150 <= phi <= -50 and 100 <= psi <= 180:
        return "sheet"
    return "loop"


def _classify_chain_phi_psi(residues: List[Residue]) -> int:
    """Walk a list of residues (already sorted) by φ/ψ. Returns the
    number of residues classified by this pass."""
    classified = 0
    n = len(residues)
    for i, r in enumerate(residues):
        if r.secondary_structure != "unknown":
            continue
        if i == 0 or i == n - 1:
            continue
        bb_prev = _backbone_atoms(residues[i - 1])
        bb_curr = _backbone_atoms(r)
        bb_next = _backbone_atoms(residues[i + 1])
        if bb_prev is None or bb_curr is None or bb_next is None:
            continue
        phi = _dihedral(bb_prev[2], bb_curr[0], bb_curr[1], bb_curr[2])
        psi = _dihedral(bb_curr[0], bb_curr[1], bb_curr[2], bb_next[0])
        if math.isnan(phi) or math.isnan(psi):
            continue
        r.secondary_structure = _classify(phi, psi, r.name)
        classified += 1
    return classified


def _prune_short_runs(mol: Molecule, min_helix: int = 4, min_sheet: int = 3) -> int:
    """Demote scattered helix/sheet runs shorter than DSSP's minima.

    Real α-helices need at least ~4 consecutive residues to make one
    full turn. Anything shorter is geometric noise (a single residue
    whose phi/psi happens to land in the helix box). Same logic for
    β-sheets (3+ consecutive residues).
    """
    demoted = 0
    for chain in mol.chains:
        residues = chain.residues
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
            threshold = min_helix if ss == "helix" else min_sheet
            if run_len < threshold:
                for k in range(i, j):
                    residues[k].secondary_structure = "loop"
                    demoted += 1
            i = j
    return demoted


def infer_secondary_structure(mol: Molecule) -> int:
    """Infer secondary structure for any residue still set to "unknown".

    The pipeline is two-phase:

    1. **H-bond pass** (primary, more accurate). Port of the
       3Dmol.js algorithm: assign each backbone N/O to its closest
       inter-residue N...O contact, then declare helix where i↔i±4
       H-bonds exist and sheet where paired H-bonds exist outside
       helices. Gap-plug 1-residue holes and drop singletons.
    2. **φ/ψ fallback**. For any residue still 'unknown' after
       hbond detection (typically chain-end residues that lack a
       neighbour), classify by Ramachandran regions. Then prune
       short scattered runs (DSSP minimum: 4 helix, 3 sheet).

    Already-labelled residues are left alone in both passes.

    Returns the number of residues newly classified across both
    passes (excluding the demote pass).
    """
    # Phase 1: H-bond based.
    n_hbonds = compute_secondary_structure_from_hbonds(mol)
    # Phase 2: φ/ψ fallback for whatever's left.
    n_phi = 0
    for chain in mol.chains:
        residues = sorted(chain.residues, key=lambda r: r.number)
        n_phi += _classify_chain_phi_psi(residues)
    # Post-pass: demote scattered short helix/sheet runs.
    _prune_short_runs(mol)
    return n_hbonds + n_phi