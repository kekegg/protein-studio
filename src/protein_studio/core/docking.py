"""Docking pose parsing — Vina (PDBQT) and EquiScore (multi-PDB).

AutoDock Vina outputs poses in PDBQT format. Each MODEL record holds a
single pose with its score in the first REMARK:

    REMARK  VINA RESULT:    -7.3      0.000      0.000      0.000

EquiScore / DiffDock-L and similar tools usually output one PDB per
pose in a directory. We detect both and return a unified list of
:class:`DockingPose` records.

Each pose knows:

- ``score``           : kcal/mol (Vina) or normalised confidence
- ``rank``            : 1..N from score ascending
- ``ligand``          : :class:`Molecule` containing the ligand atoms
- ``receptor_path``   : receptor PDB path (optional)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .io import load_pdb
from .molecule import Atom, Molecule


@dataclass
class DockingPose:
    """One docking pose."""
    rank: int
    score: float
    ligand: Molecule
    source: str       # path / identifier
    pose_id: int = 0  # model index in the file

    def __repr__(self) -> str:
        return f"DockingPose(rank={self.rank}, score={self.score:.3f}, ligand={self.ligand.name})"


# ---------------------------------------------------------------- PDBQT


# Map Vina's PDBQT atom-type field (the trailing ``<charge> <type>``
# pair AutoDock Vina writes after column 66) to the chemical element.
# The PDB element column (77-78) is usually blank in Vina output, so
# we recover the real element from the type token. Single-letter
# ambiguous types (``A``/``C``/``F``/``H``/``N``/``O``/``P``/``S``)
# map to a single element; two-letter halogen types and Vina's
# H-bond variants (``NA``/``OA``/``SA``/``HD``/``HS``) also map
# cleanly. Without this table most ligand atoms fall back to
# ``color_for_element("A")`` and render as a single gray fallback,
# which is what was making every docked pose look monochrome.
_VINA_TYPE_TO_ELEMENT = {
    "C":  "C", "A":  "C",
    "N":  "N", "NA": "N", "NS": "N",
    "O":  "O", "OA": "O", "OS": "O",
    "S":  "S", "SA": "S",
    "H":  "H", "HD": "H", "HS": "H",
    "F":  "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "P":  "P",
}


def _element_from_pdbqt(L: str, name: str) -> str:
    """Recover the chemical element from a Vina PDBQT line.

    AutoDock Vina writes the ``<charge> <atom_type>`` pair (where
    ``atom_type`` is e.g. ``A``/``C``/``NA``/``HD``/``Cl``) into the
    trailing whitespace + element column of the PDB line. The token
    that ends up in columns 77-78 of the output is just the last
    column of the atom-type field — which is ``"A"`` for aromatic
    carbons, ``"C"`` for aliphatic, ``"N"`` for nitrogen, and so on.
    That token IS the Vina atom type, not the chemical element, so
    we can't trust the PDB element column for Vina output at all.

    We always read the **last whitespace-separated token** of the
    line, look it up in the Vina-type table, and fall back to the
    first letter of the atom name only if the type isn't recognised
    (e.g. custom non-Vina PDBQT dialects).
    """
    tail = L.rstrip().split()[-1] if L.rstrip() else ""
    if tail in _VINA_TYPE_TO_ELEMENT:
        return _VINA_TYPE_TO_ELEMENT[tail]
    if tail:
        # Unknown Vina type — best-effort: first char upper-cased.
        return tail[0].upper()
    nm = name.strip()
    return nm[0].upper() if nm else "X"


def _pdbqt_to_pdb_atom_line(line: str) -> Optional[str]:
    """Convert a Vina PDBQT ATOM/HETATM line to PDB ATOM format.

    Vina appends ``<charge> <type>`` after the element column; we just
    drop them. Returns None if the line is not a record.
    """
    rec = line[:6].strip()
    if rec not in ("ATOM", "HETATM"):
        return None
    # Pad the line to the standard PDB column widths
    L = line.rstrip("\n").ljust(80)

    # PDB column layout (1-indexed → Python slice)
    # 1-6   record
    # 7-11  serial
    # 13-16 name
    # 17    altLoc
    # 18-20 resName
    # 22    chainID
    # 23-26 resSeq
    # 27    iCode
    # 31-38 x
    # 39-46 y
    # 47-54 z
    # 55-60 occupancy
    # 61-66 tempFactor
    # 77-78 element
    serial  = L[6:11]
    name    = L[12:16]
    resname = L[17:20]
    chain   = (L[21:22] or "A").strip() or "A"
    resseq  = L[22:26]
    icode   = (L[26:27] or " ").strip() or " "
    x       = L[30:38]
    y       = L[38:46]
    z       = L[46:54]
    occ     = (L[54:60] or "  1.00").strip() or "1.00"
    bfac    = (L[60:66] or "  0.00").strip() or "0.00"
    elem = _element_from_pdbqt(L, name)

    # Build a real 80-column PDB line that matches the column
    # layout the load_pdb reader expects. The reader's column map is
    # based on the standard PDB v3.3 layout:
    #   1-6   rec      7-11  serial   13-16 name    17    altLoc
    #   18-20 resName  22    chainID  23-26 resSeq  27    iCode
    #   31-38 x        39-46 y        47-54 z
    #   55-60 occ      61-66 bfac     77-78 element
    # The atom name is 4 chars at columns 13-16. For 1-2 char element
    # symbols (N, C, O, S, Cl, Br, ...) the *input* Vina name already
    # contains the leading space (e.g. " Cl "), so we emit it
    # verbatim. Resname is left-padded to 3 chars and lives at
    # cols 18-20. Element is left-padded to 2 chars at cols 77-78
    # so a 2-letter element like "Cl" lands in both positions.
    #
    # Earlier converters here either (a) double-padded the name and
    # shifted every downstream field by 1 column — making
    # load_pdb's element slice read the second character of "Cl"
    # and end up as "L" — or (b) used a 78-char format that left
    # the element column at the wrong offset. The version below
    # produces exactly 80 chars and aligns every field to its
    # canonical PDB position.
    pdb = (
        f"{rec:<6s}"          # 1-6   record (ATOM / HETATM)
        f"{serial:>5s}"       # 7-11  serial
        f" "                  # 12    (altLoc for 4-char names is ' '
                              #        at col 12; for 4-char aligned
                              #        names it lives at col 13-16
                              #        with the leading space already
                              #        part of the name field)
        f"{name:<4s}"         # 13-16 atom name
        f" "                  # 17    altLoc
        f"{resname:<3s}"      # 18-20 resname
        f" "                  # 21
        f"{chain:1s}"         # 22    chain
        f"{resseq:>4s}"       # 23-26 resSeq
        f"{icode:1s}"         # 27    iCode
        f"   "                # 28-30 (3 spaces before coords)
        f"{x}"                # 31-38 x
        f"{y}"                # 39-46 y
        f"{z}"                # 47-54 z
        f"{occ:>6s}"          # 55-60 occupancy
        f"{bfac:>6s}"         # 61-66 B-factor
        f"      "            # 67-72 (6 spaces, pre-segId padding)
        f"    "              # 73-76 segID (4 chars, blank here)
        f"{elem:<2s}"        # 77-78 element (left-aligned)
        f"  "               # 79-80 charge (2 chars, blank)
        f"\n"
    )
    return pdb


def _parse_vina_pdbqt(path: str) -> List[DockingPose]:
    poses: List[DockingPose] = []
    cur_lines: List[str] = []
    cur_score: Optional[float] = None
    pose_id = 0
    cur_name = Path(path).stem
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                cur_lines = []
                continue
            if line.startswith("ENDMDL"):
                if cur_lines:
                    # Write the buffered lines to a temp PDB-like buffer
                    pdb_text = "".join(cur_lines) + "END\n"
                    try:
                        mol = _parse_pdb_text(pdb_text, name=f"{cur_name}_pose{pose_id+1}")
                    except Exception:
                        mol = None
                    if mol is not None:
                        poses.append(DockingPose(
                            rank=0,  # filled after sorting
                            score=cur_score if cur_score is not None else 0.0,
                            ligand=mol, source=path, pose_id=pose_id,
                        ))
                    pose_id += 1
                    cur_lines = []
                continue
            if line.startswith("REMARK") and "VINA RESULT" in line:
                try:
                    parts = line.split()
                    # REMARK VINA RESULT: score rmsd_lb rmsd_ub
                    cur_score = float(parts[3])
                except (IndexError, ValueError):
                    pass
            pdb_line = _pdbqt_to_pdb_atom_line(line)
            if pdb_line:
                cur_lines.append(pdb_line)

    # Handle single-model file (no MODEL/ENDMDL)
    if not poses and cur_lines:
        pdb_text = "".join(cur_lines) + "END\n"
        try:
            mol = _parse_pdb_text(pdb_text, name=cur_name)
            poses.append(DockingPose(
                rank=1, score=cur_score or 0.0,
                ligand=mol, source=path, pose_id=0,
            ))
        except Exception:
            pass

    return _rank(poses)


def _parse_pdb_text(text: str, name: str = "ligand") -> Molecule:
    """Parse a single PDB blob using the existing PDB reader.

    The returned molecule's name is set to ``name`` — without this
    override, ``load_pdb`` derives the name from the temp file's stem
    (e.g. ``tmpk78876az``), which leaks into downstream SceneObject
    names like ``pose_1_tmpk78876az`` and makes it impossible to
    predict the object name after ``load_docking``.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        mol = load_pdb(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    mol.name = name
    return mol


# ---------------------------------------------------------------- multi-PDB (EquiScore)


def _parse_pose_dir(directory: str) -> List[DockingPose]:
    """Parse a directory of one-PDB-per-pose files (EquiScore / DiffDock).

    Conventionally:
      - File names sort lexicographically
      - A sidecar ``scores.txt`` or ``ranking.txt`` may provide scores
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(directory)
    pdb_files = sorted(d.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files in {directory}")
    score_path = d / "scores.txt"
    score_map: dict[str, float] = {}
    if score_path.exists():
        for line in score_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    score_map[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    poses = []
    for i, f in enumerate(pdb_files):
        try:
            mol = load_pdb(f)
            score = score_map.get(f.stem, score_map.get(f.name, -float(i + 1)))
            poses.append(DockingPose(
                rank=0, score=score,
                ligand=mol, source=str(f), pose_id=i,
            ))
        except Exception as e:
            warnings.warn(f"Failed to parse pose {f}: {e}")
    return _rank(poses)


def _rank(poses: List[DockingPose]) -> List[DockingPose]:
    """Reassign ranks 1..N by ascending score."""
    poses = sorted(poses, key=lambda p: p.score)
    for i, p in enumerate(poses):
        p.rank = i + 1
    return poses


# --------------------------------------------------------------- public API


def load_docking_results(path: str) -> List[DockingPose]:
    """Auto-detect Vina PDBQT vs EquiScore directory and return ranked poses."""
    p = Path(path)
    if p.is_dir():
        return _parse_pose_dir(str(p))
    if p.suffix.lower() in (".pdbqt", ".pdb"):
        if p.suffix.lower() == ".pdbqt":
            return _parse_vina_pdbqt(str(p))
        # Single PDB — wrap as a 1-pose result
        mol = load_pdb(p)
        return [DockingPose(rank=1, score=0.0, ligand=mol, source=str(p), pose_id=0)]
    raise FileNotFoundError(path)


def poses_to_color_array(poses: List[DockingPose]) -> np.ndarray:
    """Per-pose RGB color by rank: top pose = green, worst = red."""
    if not poses:
        return np.zeros((0, 3))
    n = len(poses)
    out = np.zeros((n, 3), dtype=float)
    # Rank 1 → bright green; rank N → bright red
    for i, p in enumerate(poses):
        t = 0.0 if n <= 1 else i / (n - 1)
        out[i] = (t, 1.0 - t, 0.2)
    return out