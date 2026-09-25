"""Molecular-dynamics trajectory support.

Backed by MDAnalysis — handles DCD, XTC, NC, TRR and any other format
MDAnalysis supports. We expose a thin :class:`Trajectory` wrapper that
can:

- Hold N frames worth of coordinates for one topology.
- Update the coordinates of an existing :class:`Molecule` in place
  so the GUI/console can step through frames without re-parsing.
- Provide simple statistics (RMSD, frame-to-frame drift) on demand.

For convenience we also bundle a one-shot loader for the classic
ubiquitin benchmark trajectory from the MDAnalysis test data
(``MDAnalysis.tests.datafiles``), but in practice users will point us
at their own files.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .molecule import Atom, Molecule


@dataclass
class TrajectoryFrame:
    """A snapshot of the molecule at one frame."""
    index: int
    time_ps: Optional[float]
    coords: np.ndarray   # (N_atoms, 3)


class Trajectory:
    """MD trajectory wrapper.

    Parameters
    ----------
    topology_pdb : str
        PDB file describing the topology.
    trajectory : str
        Trajectory file (DCD, XTC, NC, TRR, ...).
    """

    def __init__(self, topology_pdb: str, trajectory: str) -> None:
        self.topology_pdb = topology_pdb
        self.trajectory_path = trajectory
        self._universe = None
        self._ref_mol: Optional[Molecule] = None
        self._load()

    # ------------------------------------------------------------ internals
    def _load(self) -> None:
        try:
            import MDAnalysis as mda
        except ImportError as e:
            raise RuntimeError(
                "MDAnalysis is required for trajectory support — "
                "install with `pip install MDAnalysis`"
            ) from e
        self._universe = mda.Universe(self.topology_pdb, self.trajectory_path)

    @property
    def universe(self):
        return self._universe

    @property
    def n_frames(self) -> int:
        return int(self._universe.trajectory.n_frames)

    @property
    def n_atoms(self) -> int:
        return int(self._universe.atoms.n_atoms)

    # ------------------------------------------------------- reference mol
    def reference_molecule(self) -> Molecule:
        """Return a :class:`Molecule` for the topology at frame 0.

        Useful as the canonical "Molecule" passed to representations.
        Call :meth:`apply_frame` afterwards to update coords in place.
        """
        if self._ref_mol is not None:
            return self._ref_mol
        from .io import load_pdb
        mol = load_pdb(self.topology_pdb)
        self._ref_mol = mol
        return mol

    def apply_frame(self, index: int) -> Molecule:
        """Move trajectory to *index* and copy coords into the ref Molecule."""
        mol = self.reference_molecule()
        self._universe.trajectory[index]
        coords = self._universe.atoms.positions.astype(float)
        if coords.shape != (mol.n_atoms(), 3):
            warnings.warn(
                f"Trajectory has {coords.shape[0]} atoms but topology has "
                f"{mol.n_atoms()} — coords truncated/padded."
            )
            n = min(coords.shape[0], mol.n_atoms())
            coords = coords[:n]
        atoms = mol.atoms
        for i, a in enumerate(atoms):
            if i < len(coords):
                a.coord = coords[i]
        return mol

    # ------------------------------------------------------------ queries
    def rmsd_to_first(self, atom_selection: str = "name CA") -> np.ndarray:
        """Per-frame RMSD to the first frame, restricted to a selection.

        Returns an array of length n_frames.
        """
        import MDAnalysis as mda
        sel = self._universe.select_atoms(atom_selection)
        if len(sel) == 0:
            raise ValueError(f"No atoms match selection: {atom_selection!r}")
        ref = sel.positions.copy()
        out = np.empty(self.n_frames, dtype=float)
        for i, ts in enumerate(self._universe.trajectory):
            cur = sel.positions
            d = cur - ref
            out[i] = float(np.sqrt((d ** 2).sum(axis=1).mean()))
        return out

    def frame_summary(self, index: int) -> TrajectoryFrame:
        self._universe.trajectory[index]
        return TrajectoryFrame(
            index=index,
            time_ps=float(self._universe.trajectory.time)
                   if self._universe.trajectory.time is not None else None,
            coords=self._universe.atoms.positions.astype(float),
        )

    def iter_frames(self) -> List[TrajectoryFrame]:
        return [
            self.frame_summary(i) for i in range(self.n_frames)
        ]


# ----------------------------------------------------------- convenience


def load_trajectory(topology_pdb: str, trajectory: str) -> Trajectory:
    """Convenience entry-point matching the rest of the loader API."""
    return Trajectory(topology_pdb, trajectory)


def try_fetch_demo_trajectory() -> Optional[Trajectory]:
    """Try to fetch MDAnalysis's bundled ubiquitin benchmark (DCD).

    Returns ``None`` if the data isn't available offline.
    """
    try:
        from MDAnalysis.tests.datafiles import PSF_UBQ, DCD_UBQ
        return Trajectory(PSF_UBQ, DCD_UBQ)
    except Exception:
        return None