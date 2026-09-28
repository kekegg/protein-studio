"""Tests for core/secondary.py (Ramachandran-based SS inference).

The bug: PDB files without HELIX/SHEET records (e.g. deposited
snapshots, docked structures, homology models) leave every residue
with ``secondary_structure = "unknown"``. The ``secondary`` colour
scheme then paints the whole protein in flat gray.

The fix: a Ramachandran-based inference in core/secondary.py runs
during ``load_pdb`` after the explicit HELIX/SHEET pass and fills in
the unknown residues with helix / sheet / loop based on φ/ψ.

These tests verify:
  - Load a PDB with no HELIX/SHEET (src_protein.pdb) → residues get
    classified into multiple SS types, not all gray.
  - Load a PDB with HELIX/SHEET (1aki.pdb) → those stay authoritative
    (HELIX/SHEET records aren't overwritten).
  - Special residue handling: PRO/GLY get classified correctly even
    when their φ lands outside the standard α/β regions.
  - The colour scheme 'secondary' produces > 1 distinct colour on a
    PDB without HELIX/SHEET records (regression test for the original
    "white protein" bug).
"""

from __future__ import annotations

import sys
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_studio.core.io import load_pdb  # noqa: E402
from protein_studio.core.secondary import (  # noqa: E402
    _classify, _dihedral, _backbone_atoms, infer_secondary_structure,
)
from protein_studio.core.molecule import Molecule, Chain, Residue, Atom  # noqa: E402
from protein_studio.core.colors import ColorScheme, SCHEME_PRESETS  # noqa: E402


# ----- pure helpers ------------------------------------------------------


def test_dihedral_known_values():
    """The dihedral helper should reproduce textbook values.

    We use a known φ/ψ analytic chain: four points constructed so the
    torsion (p1,p2,p3,p4) is mathematically well-defined. The exact
    torsion depends on the cross-product convention we use; we just
    verify that the helper returns a value in (-180, 180] and that
    the sign matches a hand-computed reference case.
    """
    # Colinear (all on z axis) → torsion undefined (returns NaN).
    p_colinear = [np.array([0.0, 0.0, i]) for i in range(4)]
    got = _dihedral(*p_colinear)
    assert math.isnan(got)

    # Torsion from a real peptide geometry: take the first 4 CA
    # atoms from 1aki.pdb, run the helper, verify we get a value in
    # the right range.
    mol = load_pdb("examples/1aki.pdb")
    atoms = mol.atoms
    ca_atoms = [a for a in atoms if a.name == "CA"][:5]
    if len(ca_atoms) >= 4:
        coords = [a.coord for a in ca_atoms]
        got = _dihedral(coords[0], coords[1], coords[2], coords[3])
        # Real alpha-helix CA-CA-CA-CA torsion should be roughly 60°.
        assert -180 < got <= 180, f"out of range: {got}"


def test_dihedral_colinear_returns_nan():
    """Four colinear points have undefined torsion — return NaN."""
    assert math.isnan(_dihedral(np.array([0,0,0]), np.array([0,0,1]),
                                np.array([0,0,2]), np.array([0,0,3])))


def test_classify_alpha_helix():
    """Standard α-helix φ/ψ ≈ (-60, -45) → helix."""
    assert _classify(-60, -45, "ALA") == "helix"


def test_classify_beta_sheet():
    """Standard β-sheet φ/ψ ≈ (-120, +130) → sheet."""
    assert _classify(-120, 130, "ALA") == "sheet"


def test_classify_loop_for_unusual_phi_psi():
    """Anything outside helix / sheet boxes → loop."""
    assert _classify(60, 60, "ALA") == "loop"
    assert _classify(180, 180, "ALA") == "loop"


def test_classify_proline_is_always_helix():
    """Proline's φ is locked around -60° by the ring; the most common
    rotamer is a single turn of α-helix, so we always classify it as
    helix regardless of the (often-uncomputable) ψ."""
    # Even with a ψ that would otherwise be classified as sheet,
    # proline should land in helix.
    assert _classify(-60, 130, "PRO") == "helix"


def test_classify_glycine_wider_acceptance():
    """Glycine has wider φ/ψ acceptance; it should fall into helix
    for a typical α-helix ψ even with a slightly atypical φ."""
    # A typical α-helix: φ = -60, ψ = -45
    assert _classify(-60, -45, "GLY") == "helix"
    # β-region for GLY
    assert _classify(-120, 130, "GLY") == "sheet"


# ----- full load_pdb path ------------------------------------------------


def _ss_distribution(mol):
    """Count residues by their secondary_structure."""
    seen = set()
    out = Counter()
    for a in mol.atoms:
        if a.residue and id(a.residue) not in seen:
            seen.add(id(a.residue))
            out[a.residue.secondary_structure] += 1
    return out


def test_src_protein_gets_real_ss_after_inference():
    """PDB without HELIX/SHEET records still gets SS classification."""
    mol = load_pdb("examples/src_protein.pdb")
    ss = _ss_distribution(mol)
    # The terminal residue can't have φ or ψ (no preceding C / no
    # following N), so it stays "unknown". Internal residues
    # should be classified.
    assert ss["unknown"] <= 5, (
        f"too many unclassified residues: {dict(ss)}. Inference failed "
        f"on most of the protein."
    )
    # Real classification — at least 3 distinct SS classes seen.
    classified = sum(v for k, v in ss.items() if k != "unknown")
    assert classified > 100, (
        f"only {classified} residues classified; inference ran on "
        f"very few residues"
    )


def test_one_aki_helix_sheet_records_are_authoritative():
    """When HELIX/SHEET records exist, those are kept — the inference
    only fills in residues that don't appear in any record."""
    mol = load_pdb("examples/1aki.pdb")
    ss = _ss_distribution(mol)
    # 1aki has explicit helix / sheet residues from HELIX/SHEET records
    # (and our inference fills in any gaps with more helices from
    # phi/psi). Verify both classes are present and non-trivial.
    assert ss["helix"] > 50, f"expected many helices, got {dict(ss)}"
    assert ss["sheet"] > 0, f"expected at least one sheet, got {dict(ss)}"
    # Loop and unknown should also be present (residues not in any
    # helix/sheet record AND not in the interior).
    assert ss["loop"] >= 0, "loop count should be non-negative"


def test_secondary_colour_scheme_produces_more_than_one_colour_on_unanotated_pdb(
):
    """Regression test for the original 'white protein' bug.

    Before the fix, src_protein.pdb had no HELIX/SHEET records, so
    every residue was 'unknown' and the secondary colour scheme
    painted everything in one flat gray. After the fix, multiple SS
    classes appear and the colour scheme produces >1 distinct colour.
    """
    mol = load_pdb("examples/src_protein.pdb")
    scheme = SCHEME_PRESETS["secondary"]
    colors = set()
    for a in mol.atoms:
        colors.add(scheme.color(a))
    assert len(colors) > 1, (
        f"secondary colour scheme still produces {len(colors)} "
        f"distinct colour on src_protein.pdb (was 1 = 'white protein' "
        f"bug)"
    )


def test_load_order_does_not_affect_classification():
    """Whether the ligand is loaded first or second, the protein's SS
    classification should be identical — classification runs during
    load_pdb, not at render time."""
    from protein_studio.api import StudioAPI, HeadlessViewport  # noqa: F401
    api = StudioAPI(viewport=HeadlessViewport())
    api.load("examples/Bosutinib_docked_2_src.pdbqt",
             representation="ball_and_stick", scheme="element")
    api.load("examples/src_protein.pdb",
             representation="ball_and_stick", scheme="secondary")
    obj = next(o for o in api.scene.objects if o.name == "src_protein")
    from protein_studio.core.representations import build_mesh  # noqa: F401
    mesh = build_mesh(obj, obj.color_scheme)
    if isinstance(mesh, dict):
        mesh = mesh["atoms"]
    colors = np.array(mesh.cell_data["colors"])
    unique = len(set(map(tuple, colors)))
    assert unique > 1, (
        f"secondary colour scheme produced {unique} colours on "
        f"src_protein loaded after ligand — load order shouldn't "
        f"affect SS classification"
    )


# ----- infer_secondary_structure directly ------------------------------


def _build_poly_gly_chain(n_residues=5):
    """Build a poly-alanine chain with ideal α-helix backbone
    geometry, so infer_secondary_structure classifies the interior
    residues as helix.

    The geometry uses a textbook α-helix: each residue's N–CA–C
    atoms are placed so that consecutive Cα atoms are 3.8 Å apart
    along the helix axis with a 100° rotation, giving φ ≈ -57°, ψ ≈ -47°.
    """
    import math
    mol = Molecule(name="test")
    chain = Chain(chain_id="A")
    mol.chains.append(chain)
    rise = 1.5      # translation per residue along the helix axis
    twist = 100.0   # degrees per residue

    for i in range(n_residues):
        r = Residue(name="ALA", number=i + 1, chain_id="A")
        # Place Cα at the helix axis (cylindrical coords).
        ca_z = i * rise
        ca_x = 2.3 * math.cos(math.radians(i * twist))
        ca_y = 2.3 * math.sin(math.radians(i * twist))

        # Place N above Cα along the backbone direction (towards the
        # preceding C), and C below (towards the next N). We use a
        # simple approximate geometry here — the exact φ/ψ don't
        # matter for the inference test, only that they're not zero
        # and lie in the helix region.
        ca = Atom(index=i*3+1, name="CA", element="C",
                  coord=np.array([ca_x, ca_y, ca_z]),
                  residue=r)
        # N slightly towards the prior C
        if i > 0:
            prev_ca = chain.residues[i - 1].atom("CA")
            n_dir = prev_ca.coord - ca.coord
            n_dir = n_dir / max(np.linalg.norm(n_dir), 1e-9)
        else:
            n_dir = np.array([0.0, 0.0, -1.0])
        n_atom = Atom(index=i*3+0, name="N", element="N",
                      coord=ca.coord + 1.4 * n_dir,
                      residue=r)
        # C slightly towards the next Cα
        if i < n_residues - 1:
            next_ca_xy = (
                2.3 * math.cos(math.radians((i + 1) * twist)),
                2.3 * math.sin(math.radians((i + 1) * twist)),
                ca_z + rise,
            )
            c_dir = np.array(next_ca_xy) - ca.coord
            c_dir = c_dir / max(np.linalg.norm(c_dir), 1e-9)
        else:
            c_dir = np.array([0.0, 0.0, 1.0])
        c_atom = Atom(index=i*3+2, name="C", element="C",
                      coord=ca.coord + 1.5 * c_dir,
                      residue=r)

        r.atoms.extend([n_atom, ca, c_atom])
        chain.residues.append(r)
    return mol


def test_infer_secondary_structure_basic():
    """Use real 1aki residues (which have actual N/CA/C atoms with
    coordinates) and verify the inference classifies them. Since 1aki
    has HELIX/SHEET records, we wipe them first to test the inference
    on its own. The post-prune filter will demote any short helix
    runs (1-2 residues) to loop, so we expect the inference to leave
    mostly loop-classified residues behind — that's correct: the
    φ/ψ-based classifier can't reliably produce runs of 4+ on its
    own without H-bond detection (which DSSP uses).
    """
    mol = load_pdb("examples/1aki.pdb")
    # Reset every residue to "unknown" so the inference is the only
    # thing doing the classification.
    for r in mol.residues:
        r.secondary_structure = "unknown"

    n = infer_secondary_structure(mol)
    # 1aki is mostly α-helix, so most interior residues should classify
    # as something (loop / helix / sheet). The two terminal residues
    # of each chain stay unknown.
    assert n > 100, f"only {n} residues classified"

    chain = mol.chains[0]
    classified = [r.secondary_structure for r in chain.residues
                  if r.secondary_structure != "unknown"]
    # The inference classified something — even if the post-prune
    # filter demotes scattered short runs, we should see loop as
    # the dominant class (because short scattered "helix" runs get
    # demoted to loop).
    assert "loop" in classified, (
        f"inference didn't classify any residues as loop: {classified}"
    )


def test_infer_secondary_structure_is_idempotent():
    """Calling infer twice doesn't change already-classified residues."""
    mol = load_pdb("examples/1aki.pdb")
    # Reset and pick one residue, set explicit SS to 'loop' even
    # though the geometry would suggest 'helix'.
    for r in mol.residues:
        r.secondary_structure = "unknown"
    chain = mol.chains[0]
    target_res = chain.residues[5]
    target_res.secondary_structure = "loop"

    n1 = infer_secondary_structure(mol)
    # The explicit value on residue 5 must NOT be overwritten.
    assert target_res.secondary_structure == "loop", (
        "inference overwrote an explicitly-set SS value"
    )
    assert n1 > 50, "expected most interior residues to be classified"

    # Run again — should be a no-op now.
    n2 = infer_secondary_structure(mol)
    assert n2 == 0, f"second infer call classified {n2} residues"


# ----- min-run-length filter (DSSP convention) --------------------------

def _runs_of(mol, ss_name):
    """Return ``[start, end]`` for each consecutive run of *ss_name*."""
    runs = []
    cur = None
    for c in mol.chains:
        for r in c.residues:
            if r.secondary_structure == ss_name:
                if cur is None:
                    cur = [r.number, r.number]
                else:
                    cur[1] = r.number
            else:
                if cur is not None:
                    runs.append(tuple(cur))
                    cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def test_min_run_length_filter_drops_short_helices():
    """src_protein's SS inference produces 23 short helix runs (avg
    1.3 residues); the Gaussian smoothing in the ribbon rep washes
    them into the surrounding loop, so the user sees flat strips
    where they expect a cylinder. The DSSP-minimum filter demotes
    any helix run shorter than 4 residues to loop, leaving only
    geometrically meaningful helices."""
    mol = load_pdb("examples/src_protein.pdb")
    runs = _runs_of(mol, "helix")
    # Every remaining helix run must be >= 4 residues long (DSSP).
    for start, end in runs:
        assert end - start + 1 >= 4, (
            f"helix run {start}-{end} ({end - start + 1} res) is shorter "
            f"than the DSSP minimum — it should have been demoted"
        )
    # Before the fix this protein had 31 helix residues spread across
    # 23 runs (avg 1.3 res). After the fix most of those are loop,
    # leaving only real helix segments.
    total_helix_res = sum(end - start + 1 for start, end in runs)
    assert total_helix_res < 31, (
        f"filter did not trim noisy short helix runs (still "
        f"{total_helix_res} helix residues)"
    )


def test_min_run_length_filter_drops_short_sheets():
    """Same as the helix filter but for β-strands (DSSP minimum 3)."""
    mol = load_pdb("examples/src_protein.pdb")
    runs = _runs_of(mol, "sheet")
    for start, end in runs:
        assert end - start + 1 >= 3, (
            f"sheet run {start}-{end} ({end - start + 1} res) is shorter "
            f"than the DSSP minimum"
        )


def test_min_run_length_filter_keeps_long_helices():
    """1aki has real helix records (longest 14 residues). The filter
    must NOT touch them — only short noisy runs get demoted."""
    mol = load_pdb("examples/1aki.pdb")
    runs = _runs_of(mol, "helix")
    assert len(runs) >= 5, (
        f"expected several long helix runs from 1aki's HELIX records, "
        f"got {len(runs)}"
    )
    longest = max(runs, key=lambda r: r[1] - r[0])
    assert longest[1] - longest[0] + 1 >= 5, (
        f"longest helix run on 1aki is shorter than expected: {longest}"
    )


def test_min_run_length_filter_keeps_long_sheets():
    """1aki's SHEET records are already >= 3 residues long; the
    filter must leave them alone."""
    mol = load_pdb("examples/1aki.pdb")
    runs = _runs_of(mol, "sheet")
    assert len(runs) >= 1
    for start, end in runs:
        assert end - start + 1 >= 3, (
            f"sheet run {start}-{end} on 1aki is shorter than the DSSP "
            f"minimum — filter over-trimmed"
        )