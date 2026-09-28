"""PDB file parsing and writing.

Lightweight pure-Python PDB reader. We intentionally avoid pulling in
heavyweight dependencies (biotite is optional and only used for SS
annotation if available). The parser supports:

- ATOM / HETATM records
- Multi-model NMR structures (kept together as one Molecule per model)
- HELIX / SHEET records for secondary structure annotation
- TER records (chain breaks)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .molecule import Atom, Chain, Molecule, Residue


# Standard PDB atom record column positions (1-indexed, converted to slices).
_ATOM_FIELDS = {
    "record":   (0,  6),  # ATOM / HETATM / TER
    "serial":   (6, 11),
    "name":    (12, 16),
    "alt":     (16, 17),
    "resname": (17, 20),
    "chain":   (21, 22),
    "resseq":  (22, 26),
    "icode":   (26, 27),
    "x":       (30, 38),
    "y":       (38, 46),
    "z":       (46, 54),
    "occ":     (54, 60),
    "bfac":    (60, 66),
    "element": (76, 78),
}


def _slice(line: str, rng: tuple[int, int]) -> str:
    a, b = rng
    return line[a:b].strip()


# ------------------------------------------------------------ secondary structure


def _annotate_secondary_structure(mol: Molecule, lines: Iterable[str]) -> None:
    """Walk HELIX/SHEET records and tag each residue's secondary_structure."""
    for line in lines:
        rec = line[:6].strip().upper()
        if rec == "HELIX":
            try:
                start = int(line[21:25])
                end   = int(line[33:37])
                chain = line[19:20].strip()
            except ValueError:
                continue
            for r in mol.residues_in_range(chain, start, end):
                if r.secondary_structure == "unknown":
                    r.secondary_structure = "helix"
        elif rec == "SHEET":
            try:
                start = int(line[22:26])
                end   = int(line[33:37])
                chain = line[21:22].strip()
            except ValueError:
                continue
            for r in mol.residues_in_range(chain, start, end):
                if r.secondary_structure == "unknown":
                    r.secondary_structure = "sheet"


# ------------------------------------------------------------------- main entry


def load_pdb(path: str | os.PathLike, model_index: int = 0) -> Molecule:
    """Load a PDB file and return a :class:`Molecule`.

    Parameters
    ----------
    path : str | Path
        Path to the .pdb (or .ent, .cif-ignored) file.
    model_index : int, default 0
        If the file has multiple MODEL records (NMR), pick which one to
        load. The default loads the first model only.

    Returns
    -------
    Molecule
        Parsed molecule with chains, residues, atoms, and SS annotation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDB file not found: {p}")

    name = p.stem
    mol = Molecule(name=name, metadata={"source_path": str(p.resolve())})

    # Key chains by chain_id so HETATM records and post-TER records
    # attach to the right chain instead of being duplicated.
    chain_by_id: dict[str, Chain] = {}
    current_chain: Chain | None = None
    current_residue: Residue | None = None
    current_model = 0
    parsing_model = -1  # start before any MODEL → load everything until the first one

    def get_chain(chain_id: str) -> Chain:
        if chain_id not in chain_by_id:
            c = Chain(chain_id=chain_id)
            chain_by_id[chain_id] = c
            mol.chains.append(c)
        return chain_by_id[chain_id]

    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = line[:6].strip().upper()

            # Track MODEL / ENDMDL for multi-model files
            if rec == "MODEL":
                parsing_model = current_model
                current_model += 1
                # Reset residue-level state; chains are kept so we can
                # reuse them across models of the same id
                current_residue = None
                if parsing_model > model_index:
                    continue  # skip lines until ENDMDL
                continue
            if rec == "ENDMDL":
                parsing_model = -1  # outside any model block
                if current_model - 1 > model_index:
                    return mol
                continue
            if parsing_model > model_index:
                continue

            if rec in ("ATOM", "HETATM"):
                try:
                    serial = int(_slice(line, _ATOM_FIELDS["serial"]))
                    x = float(_slice(line, _ATOM_FIELDS["x"]))
                    y = float(_slice(line, _ATOM_FIELDS["y"]))
                    z = float(_slice(line, _ATOM_FIELDS["z"]))
                    resname = _slice(line, _ATOM_FIELDS["resname"]).upper()
                    resseq = int(_slice(line, _ATOM_FIELDS["resseq"]))
                    chain_id = _slice(line, _ATOM_FIELDS["chain"]) or "A"
                    atom_name = _slice(line, _ATOM_FIELDS["name"]).strip()
                    element_raw = _slice(line, _ATOM_FIELDS["element"])
                    element = (element_raw or atom_name[0]).strip().upper()
                    try:
                        occ = float(_slice(line, _ATOM_FIELDS["occ"]))
                    except ValueError:
                        occ = 1.0
                    try:
                        bfac = float(_slice(line, _ATOM_FIELDS["bfac"]))
                    except ValueError:
                        bfac = 0.0
                except ValueError:
                    continue  # malformed line — skip

                # Chain: keyed by chain_id (deduplicates HETATM with same id)
                current_chain = get_chain(chain_id)

                # Residue change (including HETATM into same residue number)
                if (
                    current_residue is None
                    or current_residue.number != resseq
                    or current_residue.name != resname
                    or current_residue.chain_id != chain_id
                ):
                    current_residue = Residue(
                        name=resname, number=resseq, chain_id=chain_id
                    )
                    current_chain.residues.append(current_residue)

                atom = Atom(
                    index=serial,
                    name=atom_name,
                    element=element,
                    coord=np.array([x, y, z], dtype=float),
                    residue=current_residue,
                    bfactor=bfac,
                    occupancy=occ,
                    is_hetero=(rec == "HETATM"),
                )
                current_residue.atoms.append(atom)

            elif rec == "TER":
                # Chain break — next ATOM on the same chain id will start a
                # new residue; we keep the chain in chain_by_id so HETATM
                # records with the same id still attach to it.
                current_chain = None
                current_residue = None

            elif rec == "TITLE":
                title = line[10:80].strip()
                if title and (mol.metadata.get("title", "") == ""):
                    mol.metadata["title"] = title

        # Second pass for SS annotation
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            _annotate_secondary_structure(mol, fh)

        # Third pass: HELIX/SHEET records aren't always present in
        # PDB files (deposited snapshots, docked structures, homology
        # models). After the explicit annotation pass, any residue still
        # set to ``unknown`` gets φ/ψ-classified via Ramachandran
        # regions. This means the ``secondary`` colour scheme paints
        # a meaningful picture even for files that lack DSSP
        # annotation.
        from .secondary import infer_secondary_structure
        infer_secondary_structure(mol)

    return mol


def save_pdb(mol: Molecule, path: str | os.PathLike) -> None:
    """Write the molecule back to a .pdb file (best-effort format)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serial = 1
    with p.open("w", encoding="utf-8") as fh:
        fh.write(f"REMARK   Generated by protein-studio ({mol.name})\n")
        for chain in mol.chains:
            chain_started = False
            for res in chain.residues:
                for atom in res.atoms:
                    rec = "HETATM" if atom.is_hetero else "ATOM  "
                    line = (
                        f"{rec}{serial:5d} "
                        f"{atom.name:<4s} "
                        f"{res.name:<3s} "
                        f"{chain.chain_id:1s}"
                        f"{res.number:4d}    "
                        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
                        f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}"
                        f"          "
                        f"{atom.element:>2s}\n"
                    )
                    fh.write(line)
                    serial += 1
                if not chain_started:
                    chain_started = True
        fh.write("END\n")


def list_chains(mol: Molecule) -> List[str]:
    return [c.chain_id for c in mol.chains]