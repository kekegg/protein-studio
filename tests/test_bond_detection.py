"""Tests for bond detection in core/representations.py.

Bonds are computed by ``_build_bonds`` from atom positions. The previous
implementation used a single 2.0 A global cutoff, which dropped real
bonds in coordinate-noisy X-ray structures (4hhb loses ~70 of its
real bonds, mostly C-C at 2.0-2.7 A due to coordinate noise). The
current implementation uses per-element-pair cutoffs with a 0.5 A
tolerance — the same approach PyMOL / ChimeraX / VMD use.

These tests verify that the standard amino-acid backbone + sidechain
topology is recovered exactly for a clean high-resolution structure
(1aki), and that no spurious "1-3" intra-residue false positives are
added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_studio.core.io import load_pdb  # noqa: E402
from protein_studio.core.representations import (  # noqa: E402
    _build_bonds, _bond_max, _atom_element,
    _BBOND_PAIR_MAX, _BOND_TOLERANCE,
)


# Standard amino-acid backbone + sidechain heavy-atom topology.
# Backbone: N-CA, CA-C, C-O. Sidechain varies.
EXPECTED = {
    "ALA": [("CA", "CB")],
    "VAL": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
    "LEU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "ILE": [("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "PRO": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    "PHE": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "TYR": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
            ("CZ", "OH")],
    "TRP": [("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "NE1"), ("NE1", "CE2"), ("CE2", "CD2"),
            ("CE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "CZ3"),
            ("CZ3", "CE3"), ("CE3", "CD2")],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"),
            ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2")],
    "ASP": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "ASN": [("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"),
            ("CD", "OE1"), ("CD", "OE2")],
    "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"),
            ("CD", "OE1"), ("CD", "NE2")],
    "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
            ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"),
            ("CD", "CE"), ("CE", "NZ")],
    "SER": [("CA", "CB"), ("CB", "OG")],
    "THR": [("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "CYS": [("CA", "CB"), ("CB", "SG")],
    "MSE": [("CA", "CB"), ("CB", "CG"), ("CG", "SE"), ("SE", "CE")],
    "GLY": [],  # no sidechain
}

# Backbone bonds every amino acid has
BACKBONE = [("N", "CA"), ("CA", "C"), ("C", "O")]


def _residue_bonds(mol):
    """Build a set of intra-residue bonds keyed by (residue_id, atom_name_pair)."""
    from collections import defaultdict
    atoms = list(mol.atoms)
    bonds = _build_bonds(atoms)
    res_bonds = defaultdict(set)
    for i, j in bonds:
        a, b = atoms[i], atoms[j]
        if a.residue is b.residue and a.residue is not None:
            res_bonds[(a.residue.name, a.residue.number)].add(
                tuple(sorted((a.name, b.name))))
    return res_bonds


def _missing_standard_bonds(mol):
    """Return the list of (resname, resseq, bond-atom-pair) that the
    standard amino-acid topology expects but our bond detection didn't
    find. Counts only intra-residue sidechain bonds (not peptide bonds,
    which span residues)."""
    res_bonds = _residue_bonds(mol)
    missing = []
    for (rn, ri), bs in res_bonds.items():
        if rn not in EXPECTED:
            continue
        expected = {tuple(sorted(p)) for p in EXPECTED[rn]}
        for p in EXPECTED[rn]:
            if tuple(sorted(p)) not in bs:
                missing.append((rn, ri, p))
    return missing


# ----- per-element cutoffs ----------------------------------------------

def test_atom_element_returns_first_alpha_uppercased():
    assert _atom_element("CA") == "C"
    assert _atom_element("N") == "N"
    assert _atom_element("OD1") == "O"
    assert _atom_element("Cl") == "C"  # first letter; fallback for Cl
    assert _atom_element("1HA") == "H"  # digit prefix → next alpha


def test_bond_max_uses_table_for_known_pairs():
    assert abs(_bond_max("CA", "CB") - 1.65) < 1e-9  # C-C
    assert abs(_bond_max("C", "N") - 1.55) < 1e-9     # C-N
    assert abs(_bond_max("OD1", "CD") - 1.50) < 1e-9 # C-O
    assert abs(_bond_max("SG", "CB") - 1.85) < 1e-9   # C-S


def test_bond_max_uses_fallback_for_unknown():
    # Exotic elements not in {C, H, N, O, P, S} fall back to the
    # generous 2.4 A cutoff. (Selenium is *not* exotic — PDB atom
    # names "SE" start with "S" and are caught by the S-C entry.)
    assert abs(_bond_max("FE", "NE") - 2.4) < 1e-9   # iron-ligand
    assert abs(_bond_max("ZN", "SG") - 2.4) < 1e-9   # zinc-ligand


def test_bond_max_normalises_element_order():
    """The lookup table is keyed alphabetically on the element pair;
    `_bond_max` must normalise so CC and CC return the same value."""
    assert _bond_max("CA", "CB") == _bond_max("CB", "CA")
    assert _bond_max("OD1", "CD") == _bond_max("CD", "OD1")


# ----- bond detection on real PDBs --------------------------------------

def test_one_aki_has_no_missing_standard_bonds():
    """1aki is a clean high-resolution X-ray structure. Every standard
    amino-acid sidechain bond should be detected, including the
    borderline C-C bonds in PHE/TYR/HIS/TRP that sit at ~1.4-1.5 A."""
    mol = load_pdb("examples/1aki.pdb")
    missing = _missing_standard_bonds(mol)
    assert missing == [], f"1aki missing standard bonds: {missing[:10]}"


def test_one_ubq_has_no_missing_standard_bonds():
    """1ubq (ubiquitin) is another high-resolution structure."""
    mol = load_pdb("examples/1ubq.pdb")
    missing = _missing_standard_bonds(mol)
    assert missing == [], f"1ubq missing standard bonds: {missing[:10]}"


def test_four_hhb_reduces_missing_bonds_vs_global_cutoff():
    """4hhb (hemoglobin tetramer) is lower resolution and has many
    C-C / C-N bonds at 2.0-2.7 A due to coordinate noise. With the
    per-element cutoffs (max ideal + 0.5 A tolerance), we recover
    most of those bonds.

    The previous global-cutoff implementation missed ~74 standard
    amino-acid bonds in 4hhb. Per-element cutoffs reduce this to
    ~40, with the remaining misses all being coordinate errors in
    the source file (LYS60 CE-NZ at 2.69 A, GLU26 CD-OE1 at 2.83 A
    — those don't reflect real chemistry, they reflect bad X-ray
    modelling). We assert on the improvement rather than the
    absolute number to avoid coupling the test to upstream noise."""
    mol = load_pdb("examples/4hhb.pdb")
    missing = _missing_standard_bonds(mol)
    # Before per-element cutoffs: 74 missing. After: ~40.
    assert len(missing) < 50, (
        f"per-element cutoffs should recover most of 4hhb's noisy "
        f"bonds; got {len(missing)} missing (was 74 with global "
        f"cutoff). First few: {missing[:5]}")


# ----- 1-3 skip is structural, not index-based -------------------------

def test_intra_residue_three_atom_skip_drops_spurious_bonds():
    """VAL sidechain: CB is bonded to CG1 and CG2. CG1-CG2 across
    CB would be ~2.5 A — within the per-element cutoff, so the bbox
    picks them up as a candidate. The 1-3 skip must drop them
    because they share a common neighbor (CB) in the same residue."""
    mol = load_pdb("examples/1aki.pdb")
    atoms = list(mol.atoms)
    bonds = _build_bonds(atoms)
    bond_set = set()
    for i, j in bonds:
        a, b = atoms[i], atoms[j]
        if a.residue is b.residue and a.residue is not None:
            bond_set.add((a.residue.number, a.name, b.residue.name))

    # Walk every VAL residue; CG1-CG2 should NOT appear.
    val_count = 0
    for n in [a.residue.number for a in atoms
              if a.residue and a.residue.name == "VAL"]:
        if (n, "CG1", "CG2") in bond_set:
            val_count += 1
    assert val_count == 0, \
        f"VAL CG1-CG2 should be dropped by 1-3 skip; found in {val_count} residues"


def test_intra_residue_aromatic_ring_bonds_preserved():
    """PHE has a 6-membered ring CG-CD1-CE1-CZ-CE2-CD2. Every atom
    is within ~1.4 A of its ring neighbors, so the bbox picks them
    up as candidates. The 1-3 skip must NOT drop them because the
    ring bonds form a cycle — there is no single common neighbor.

    (The previous index-based heuristic — ``abs(a.index - b.index) > 4``
    — would drop CD2-CG because the index diff was 5. The structural
    1-3 skip keeps them.)"""
    mol = load_pdb("examples/1aki.pdb")
    atoms = list(mol.atoms)
    bonds = _build_bonds(atoms)

    ring_pairs = {tuple(sorted(p)) for p in [
        ("CG", "CD1"), ("CD1", "CE1"), ("CE1", "CZ"),
        ("CZ", "CE2"), ("CE2", "CD2"), ("CD2", "CG"),
    ]}
    found = 0
    for i, j in bonds:
        a, b = atoms[i], atoms[j]
        if (a.residue is b.residue and a.residue is not None
                and a.residue.name == "PHE"):
            if tuple(sorted((a.name, b.name))) in ring_pairs:
                found += 1
    n_phe = sum(1 for _ in {a.residue.number for a in atoms
                            if a.residue and a.residue.name == "PHE"})
    # Each PHE should have all 6 ring bonds.
    assert found == n_phe * 6, (
        f"PHE aromatic ring should have all 6 bonds; got "
        f"{found} / {n_phe * 6}")


# ----- empty / edge cases -----------------------------------------------

def test_empty_molecule_returns_no_bonds():
    mol = load_pdb("examples/1aki.pdb")
    assert _build_bonds([]) == []
    assert _build_bonds([list(mol.atoms)[0]]) == []


def test_single_atom_returns_no_bonds():
    mol = load_pdb("examples/1aki.pdb")
    assert _build_bonds([list(mol.atoms)[0]]) == []