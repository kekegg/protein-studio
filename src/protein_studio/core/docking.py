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
    elem    = (L[76:78] or "").strip()
    if not elem:
        elem = name.strip()[0] if name.strip() else "X"

    pdb = (
        f"{rec:<6s}"
        f"{serial:>5s}"
        f" {name:<4s}"
        f"{resname:<3s} "
        f"{chain:1s}{resseq:>4s}{icode:1s}   "
        f"{x}{y}{z}"
        f"{occ:>6s}{bfac:>6s}"
        f"          "
        f"{elem:>2s}"
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
    """Parse a single PDB blob using the existing PDB reader."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        return load_pdb(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


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