"""Core molecule data model.

This module defines lightweight, pure-Python container types for atoms,
residues, chains, and molecules. It is intentionally backend-agnostic —
no rendering or parsing imports — so it can be reused by the GUI, the
Python scripting API, and external tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

from .colors import THREE_TO_ONE


@dataclass
class Atom:
    """A single atom record.

    Attributes
    ----------
    index : int
        PDB serial number (1-based).
    name : str
        Atom name (e.g. "CA", "N", "CB").
    element : str
        Element symbol, uppercase ("C", "N", "O", "S", ...).
    coord : np.ndarray
        Cartesian coordinate, shape (3,).
    bfactor : float
        Temperature factor from PDB.
    occupancy : float
        Occupancy from PDB.
    is_hetero : bool
        True if this atom belongs to a HETATM record (ligands, water, ions).
    """

    index: int
    name: str
    element: str
    coord: np.ndarray
    residue: "Residue"
    bfactor: float = 0.0
    occupancy: float = 1.0
    is_hetero: bool = False

    @property
    def chain_id(self) -> str:
        return self.residue.chain_id if self.residue else ""

    @property
    def residue_name(self) -> str:
        return self.residue.name if self.residue else ""

    @property
    def residue_number(self) -> int:
        return self.residue.number if self.residue else 0

    @property
    def secondary_structure(self) -> str:
        return self.residue.secondary_structure if self.residue else "unknown"

    @property
    def x(self) -> float: return float(self.coord[0])
    @property
    def y(self) -> float: return float(self.coord[1])
    @property
    def z(self) -> float: return float(self.coord[2])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Atom({self.name!r} {self.element} "
            f"res={self.residue_name}{self.residue_number} "
            f"chain={self.chain_id} {tuple(self.coord)})"
        )


@dataclass
class Residue:
    """A residue grouping atoms with a common residue number and chain."""
    name: str
    number: int
    chain_id: str
    atoms: List[Atom] = field(default_factory=list)
    secondary_structure: str = "unknown"  # "helix" | "sheet" | "loop" | "turn" | "unknown"

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.name.upper(), "X")

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)

    def __len__(self) -> int:
        return len(self.atoms)

    def atom(self, name: str) -> Optional[Atom]:
        for a in self.atoms:
            if a.name == name:
                return a
        return None


@dataclass
class Chain:
    """A chain grouping residues."""
    chain_id: str
    residues: List[Residue] = field(default_factory=list)

    @property
    def sequence(self) -> str:
        return "".join(r.one_letter for r in self.residues)

    def __iter__(self) -> Iterator[Residue]:
        return iter(self.residues)

    def __len__(self) -> int:
        return len(self.residues)


@dataclass
class Molecule:
    """A protein / nucleic-acid / ligand structure.

    Attributes
    ----------
    name : str
        Display name (defaults to the PDB HEADER/TITLE or filename).
    chains : list[Chain]
        All chains in the molecule, in order.
    """
    name: str = "unnamed"
    chains: List[Chain] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ atoms
    @property
    def atoms(self) -> List[Atom]:
        out: List[Atom] = []
        for c in self.chains:
            for r in c.residues:
                out.extend(r.atoms)
        return out

    @property
    def residues(self) -> List[Residue]:
        out: List[Residue] = []
        for c in self.chains:
            out.extend(c.residues)
        return out

    def n_atoms(self) -> int:
        return sum(len(r) for c in self.chains for r in c.residues)

    def n_residues(self) -> int:
        return sum(len(c) for c in self.chains)

    def n_chains(self) -> int:
        return len(self.chains)

    def sequence(self, chain_id: str | None = None) -> str:
        if chain_id is None:
            return "".join(c.sequence for c in self.chains)
        for c in self.chains:
            if c.chain_id == chain_id:
                return c.sequence
        return ""

    # ----------------------------------------------------------------- coords
    def coords(self, atoms: Sequence[Atom] | None = None) -> np.ndarray:
        """Stack selected (or all) atom coordinates into an (N,3) array."""
        src = atoms if atoms is not None else self.atoms
        if not src:
            return np.zeros((0, 3), dtype=float)
        return np.asarray([a.coord for a in src], dtype=float)

    def center_of_mass(self) -> np.ndarray:
        return self.coords().mean(axis=0) if self.atoms else np.zeros(3)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.coords()
        if len(c) == 0:
            z = np.zeros(3)
            return z, z
        return c.min(axis=0), c.max(axis=0)

    # ----------------------------------------------------------- convenience
    def chain(self, chain_id: str) -> Optional[Chain]:
        for c in self.chains:
            if c.chain_id == chain_id:
                return c
        return None

    def residues_in_range(self, chain_id: str, start: int, end: int) -> List[Residue]:
        c = self.chain(chain_id)
        if not c:
            return []
        return [r for r in c.residues if start <= r.number <= end]

    def atom_by_name(self, chain_id: str, resnum: int, name: str) -> Optional[Atom]:
        for r in self.residues_in_range(chain_id, resnum, resnum):
            return r.atom(name)
        return None

    # ---------------------------------------------------------------- display
    def summary(self) -> str:
        """One-line human-readable summary of the molecule."""
        n_atoms = self.n_atoms()
        n_res = self.n_residues()
        n_ch = self.n_chains()
        chains = ", ".join(c.chain_id for c in self.chains)
        return (
            f"{self.name}: {n_atoms} atoms / {n_res} residues / "
            f"{n_ch} chains ({chains})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Molecule({self.summary()})"