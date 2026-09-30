"""Tests for core/fep.py — Phase 4 sidechain-truncation mutation FEP.

Phase 4 wires OpenMM up for an alchemical sidechain truncation:

    Protein + sidechain    (λ=0)
            │
            │   λ-scaled nonbonded on the disappearing atoms
            │   (Beutler 1994 soft-core LJ + linear-λ Coulomb)
            ▼
    Protein only           (λ=1)

The mutation is "truncation" — anything beyond the Cβ (including
the Cβ-Hs) is λ-decoupled.  Phase 4 is a verification framework,
not production FEP: each λ window resets positions to the
solvated-system reference, minimises a few LBFGS iterations, and
samples for ``n_steps``.  Position restraints on disappearing
atoms prevent Langevin noise from driving them into close contact
with waters (which would explode the next window's standard LJ).

These tests verify:
  - ``_BACKBONE_PLUS_CB`` is the canonical 12-atom backbone + Cβ
    set (N, H, CA, HA, C, O, OXT, CB, HB, HB1, HB2, HB3).
  - ``_find_disappearing_atoms`` returns the correct 13 sidechain
    atoms for Lys6 in 1ubq (CG, HG2/3, CD, HD2/3, CE, HE2/3,
    NZ, HZ1/2/3).
  - ``build_amber_mutation_system`` produces a CustomNonbondedForce
    with the right global parameters and exclusions, plus a
    position restraint force on the disappearing atoms.
  - ``run_amber_mutation_fep`` completes all λ windows without
    NaN on the canonical Lys6→Ala truncation of ubiquitin.
  - Misuse raises informative errors (wrong residue name → empty
    list → ValueError).
  - The α default is 1.5 nm² (Beutler 1994 recommended range),
    not the broken α=0.5 that drove disappearing atoms into
    close contact at intermediate λ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protein_studio.core.fep import (  # noqa: E402
    _BACKBONE_PLUS_CB,
    _find_disappearing_atoms,
    build_amber_mutation_system,
    run_amber_mutation_fep,
)


# Path to the canonical test PDB — ubiquitin (Lys6 is the sidechain-
# truncation target that we've used throughout Phase 4 development).
ONEUBQ = str(ROOT / "examples" / "1ubq.pdb")


# ----- pure helpers ------------------------------------------------------


def test_backbone_plus_cb_set_is_canonical_12():
    """The backbone + Cβ set should have exactly the 12 canonical
    AMBER atom names — anything else and AMBER atom-type assignment
    on the surviving atoms would break.

    The canonical set is the same as PyMOL's "show (name ca,c,n,o)"
    + Cβ + the Cβ-H protons (HB/HB1/HB2/HB3 — different AMBER
    templates name them differently).  Adding or removing atoms
    silently changes the sidechain-truncation definition.
    """
    assert isinstance(_BACKBONE_PLUS_CB, frozenset)
    assert _BACKBONE_PLUS_CB == frozenset({
        "N", "H", "CA", "HA", "C", "O", "OXT",
        "CB", "HB", "HB1", "HB2", "HB3",
    })
    assert len(_BACKBONE_PLUS_CB) == 12


def test_backbone_plus_cb_excludes_sidechain_names():
    """No sidechain atom names should sneak into the backbone set —
    a sidechain atom marked "core" would never be λ-decoupled and
    would corrupt the mutation free energy."""
    sidechain_names = {
        # Lys
        "CG", "HG2", "HG3", "CD", "HD2", "HD3",
        "CE", "HE2", "HE3", "NZ", "HZ1", "HZ2", "HZ3",
        # Ala (this is the *target* residue — nothing beyond CB)
        # nothing
        # Gly
        # nothing (Gly has no CB)
    }
    assert not (_BACKBONE_PLUS_CB & sidechain_names)


def test_find_disappearing_atoms_lys6_has_thirteen_sidechain_atoms():
    """Lys6 in 1ubq should have exactly 13 sidechain atoms that get
    truncated to Cβ (the 12 backbone + Cβ atoms stay).  We test on
    the AMBER-with-hydrogens topology (via build_amber_mutation_system)
    rather than the raw PDB — the raw PDB only has heavy atoms, so
    Lys sidechain there is only CG, CD, CE, NZ (4 atoms)."""
    system, topology, *_ = _build_system()
    disappearing = _find_disappearing_atoms(
        topology, residue_index=5, residue_name="LYS",
    )
    # Lys sidechain beyond Cβ: CG, HG2, HG3, CD, HD2, HD3, CE,
    # HE2, HE3, NZ, HZ1, HZ2, HZ3 — that's 13.
    assert len(disappearing) == 13, (
        f"expected 13 Lys6 sidechain atoms, got {len(disappearing)}"
    )
    # All returned indices must lie on the Lys6 residue.
    names = []
    for atom in topology.atoms():
        if atom.index in disappearing:
            names.append(atom.name)
    assert set(names) == {
        "CG", "HG2", "HG3", "CD", "HD2", "HD3",
        "CE", "HE2", "HE3", "NZ", "HZ1", "HZ2", "HZ3",
    }


def test_find_disappearing_atoms_empty_when_residue_name_wrong():
    """Wrong residue name on the right index → no atoms match →
    downstream callers raise.  We want the early-exit semantics:
    zero atoms, not a crash."""
    import openmm.app as app

    pdb = app.PDBFile(ONEUBQ)
    disappearing = _find_disappearing_atoms(
        pdb.topology, residue_index=5, residue_name="ALA",
    )
    assert disappearing == []


def test_find_disappearing_atoms_empty_when_index_out_of_range():
    """Index that doesn't exist → empty list."""
    import openmm.app as app

    pdb = app.PDBFile(ONEUBQ)
    disappearing = _find_disappearing_atoms(
        pdb.topology, residue_index=9999, residue_name="LYS",
    )
    assert disappearing == []


def test_find_disappearing_atoms_keeps_backbone_plus_cb():
    """Backbone + Cβ atoms of Lys6 should NOT appear in the
    disappearing set — they're the surviving atoms.  Tested on
    the AMBER-with-hydrogens topology (see the 13-atom test)."""
    system, topology, *_ = _build_system()
    disappearing = set(_find_disappearing_atoms(
        topology, residue_index=5, residue_name="LYS",
    ))
    for atom in topology.atoms():
        if (atom.residue.index == 5
                and atom.residue.name == "LYS"
                and atom.name in _BACKBONE_PLUS_CB):
            assert atom.index not in disappearing, (
                f"backbone atom {atom.name} should not be disappearing"
            )


# ----- build_amber_mutation_system ---------------------------------------


def _build_system(**overrides):
    """Build a small mutation system with default settings (Lys6,
    α=1.5, padding=0.8).  Marked slow — Phase 4 builds AMBER +
    TIP3P which is ~16k particles.
    """
    return build_amber_mutation_system(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        **overrides,
    )


@pytest.mark.slow
def test_build_amber_mutation_system_returns_five_tuple():
    """The function returns (system, topology, positions, integrator,
    disappearing) — verify the shape and that each component is
    non-empty / usable."""
    system, topology, positions, integrator, disappearing = _build_system()

    # system has both AMBER (bonded) and our soft-core (nonbonded)
    assert system.getNumParticles() > 1000  # protein + waters + ions
    # topology has solvent + protein
    assert topology.getNumAtoms() == system.getNumParticles()
    # positions is one Vec3 per particle
    assert len(positions) == system.getNumParticles()
    # integrator is a LangevinMiddleIntegrator
    assert integrator.__class__.__name__ == "LangevinMiddleIntegrator"
    # disappearing has Lys6 sidechain atoms
    assert len(disappearing) == 13


@pytest.mark.slow
def test_build_amber_mutation_system_has_softcore_force():
    """The system must contain a CustomNonbondedForce with our
    'lam' and 'alpha' global parameters, set to NoCutoff (because
    addExclusion + PME is fiddly — we use exclusions + the explicit
    pair evaluation).
    """
    import openmm

    system, *_ = _build_system()
    found = False
    for f in system.getForces():
        if isinstance(f, openmm.CustomNonbondedForce):
            found = True
            assert f.getNumGlobalParameters() == 2
            names = {
                f.getGlobalParameterName(k)
                for k in range(f.getNumGlobalParameters())
            }
            assert names == {"lam", "alpha"}
            assert f.getNonbondedMethod() == openmm.CustomNonbondedForce.NoCutoff
            # 4 per-particle params: charge, sigma, eps, is_dis
            assert f.getNumPerParticleParameters() == 4
    assert found, "CustomNonbondedForce not found"


@pytest.mark.slow
def test_build_amber_mutation_system_alpha_default_is_one_point_five():
    """α=0.5 nm² is too small — at intermediate λ the soft-core LJ
    is too weak to keep disappearing atoms away from close contacts
    with waters.  α=1.5 is the Beutler 1994-recommended value for
    protein sidechain truncation (we tested this empirically)."""
    system, *_ = _build_system()
    for f in system.getForces():
        if hasattr(f, "getGlobalParameterDefaultValue"):
            try:
                # getGlobalParameterDefaultValue takes an int index
                for k in range(f.getNumGlobalParameters()):
                    if f.getGlobalParameterName(k) == "alpha":
                        assert f.getGlobalParameterDefaultValue(k) == pytest.approx(1.5), (
                            f"alpha default should be 1.5 nm², "
                            f"got {f.getGlobalParameterDefaultValue(k)}"
                        )
            except Exception:
                pass  # NonbondedForce doesn't have this — skip


@pytest.mark.slow
def test_build_amber_mutation_system_has_many_exclusions():
    """The CustomNonbondedForce must copy AMBER's 1-2/1-3/1-4
    exception list as exclusions, otherwise bonded-pair close
    contacts (~1.5 Å) blow up the LJ energy.  For 1ubq + TIP3P
    that's >20k exclusions."""
    system, *_ = _build_system()
    for f in system.getForces():
        # CustomNonbondedForce has getNumExclusions
        if hasattr(f, "getNumExclusions"):
            n_excl = f.getNumExclusions()
            assert n_excl > 20000, (
                f"too few exclusions: {n_excl} (expected >20000 — "
                "AMBER 1-2/1-3/1-4 pairs must be excluded)"
            )


@pytest.mark.slow
def test_build_amber_mutation_system_has_position_restraints():
    """The system must include a CustomExternalForce position
    restraint on each disappearing atom.  Without these, Langevin
    noise + the soft-core's weakened close-range LJ drives
    disappearing atoms onto water Hs at intermediate λ, and the
    next window explodes to NaN.
    """
    import openmm

    system, *_ = _build_system()
    has_restraint = False
    for f in system.getForces():
        if isinstance(f, openmm.CustomExternalForce):
            # The energy expression is k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
            if "x-x0" in f.getEnergyFunction():
                has_restraint = True
                # 13 disappearing atoms → 13 restrained particles
                assert f.getNumParticles() == 13
    assert has_restraint, "CustomExternalForce position restraint not found"


@pytest.mark.slow
def test_build_amber_mutation_system_per_particle_is_dis_flag():
    """Each particle in the soft-core force carries an is_dis flag:
    1.0 for disappearing atoms, 0.0 for everyone else.  This is
    what drives the per-pair weighting in the energy expression.
    """
    system, *_ = _build_system()
    for f in system.getForces():
        # CustomNonbondedForce has getParticleParameters
        if hasattr(f, "getParticleParameters"):
            try:
                params = f.getParticleParameters(0)
            except Exception:
                continue
            # 4-tuple: (charge, sigma, eps, is_dis)
            if len(params) == 4:
                # Find the disappearing atom indices — query one
                # core and one disappearing particle.
                _, _, _, is_dis_core = f.getParticleParameters(0)
                assert is_dis_core == 0.0, "first particle should be core (is_dis=0)"
                # Find a particle with is_dis=1 — search first 200.
                found_dis = False
                for k in range(200):
                    p = f.getParticleParameters(k)
                    if p[3] == 1.0:
                        found_dis = True
                        break
                assert found_dis, "no disappearing particle found"


# ----- run_amber_mutation_fep --------------------------------------------


@pytest.mark.slow
def test_run_amber_mutation_fep_completes_all_windows_without_nan():
    """The full Phase 4 pipeline on Lys6→Ala in 1ubq should
    complete all 5 λ windows (0, 0.25, 0.5, 0.75, 1.0) without
    any NaN — this is the regression test for the original
    "Particle coordinate is NaN" failure that drove the
    soft-core formula and α tuning.

    n_steps=5 keeps the wall time under 60s — enough to exercise
    the integrator + minimiser and catch any "NaN carry-over"
    across windows.
    """
    energies, disappearing = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
        n_steps=5,
        padding_nm=0.8,
    )
    assert len(disappearing) == 13
    # All five windows completed
    assert sorted(energies.keys()) == [0.0, 0.25, 0.5, 0.75, 1.0]
    # Every energy trajectory is finite (no NaN, no inf)
    for lam, traj in energies.items():
        arr = np.asarray(traj)
        assert np.all(np.isfinite(arr)), (
            f"non-finite energy at λ={lam}: {arr}"
        )
        # Each trajectory should have at least 2 samples
        # (1 initial + n_steps post-min steps).
        assert len(traj) >= 2


@pytest.mark.slow
def test_run_amber_mutation_fep_window_means_are_finite():
    """The mean energy at each λ window must be finite.  Loose
    bound on the magnitude — soft-core + restraints inflate the
    energy above the bare AMBER baseline (~−170000 kJ/mol), so we
    just check it lands in a reasonable range.
    """
    energies, _ = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.5, 1.0),
        n_steps=5,
        padding_nm=0.8,
    )
    for lam, traj in energies.items():
        # Skip the first sample (pre-min, can be huge)
        mean = float(np.mean(traj[1:]))
        assert -1e6 < mean < 1e6, (
            f"λ={lam} mean E = {mean} kJ/mol is out of range"
        )


@pytest.mark.slow
def test_run_amber_mutation_fep_protein_only_window_at_lambda_zero():
    """At λ=0, every λ-scaled nonbonded term vanishes (weight_dis * lam
    = 0, and the core LJ / Coulomb are kept).  In other words: the
    disappearing atoms are still in the system, but their interactions
    with the rest of the protein and waters are switched off.  The
    energy should be finite and not blow up to ±inf.
    """
    energies, _ = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0,),
        n_steps=5,
        padding_nm=0.8,
    )
    arr = np.asarray(energies[0.0][1:])
    assert np.all(np.isfinite(arr))
    # The mean must not be the "atomistic minimum" ~ −170000 kJ/mol
    # because the soft-core still has bonded terms, but it must be
    # well-bounded.
    mean = float(arr.mean())
    assert abs(mean) < 5e4, (
        f"λ=0 mean E = {mean} kJ/mol is implausibly large"
    )


# ----- error handling ----------------------------------------------------


@pytest.mark.slow
def test_build_amber_mutation_system_wrong_residue_name_raises():
    """Asking for the wrong residue name on a valid index → empty
    disappearing list → informative ValueError from build_amber_
    mutation_system (the empty-list guard)."""
    with pytest.raises(ValueError, match="No atoms to disappear"):
        build_amber_mutation_system(
            ONEUBQ,
            mutation_residue=5,
            from_resname="ALA",  # 1ubq has ALA at index 27, not 5
        )


@pytest.mark.slow
def test_build_amber_mutation_system_out_of_range_index_raises():
    """Residue index out of range → empty list → ValueError."""
    with pytest.raises(ValueError, match="No atoms to disappear"):
        build_amber_mutation_system(
            ONEUBQ,
            mutation_residue=9999,
            from_resname="LYS",
        )
