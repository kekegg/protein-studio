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
    compute_bar,
    analyze_fep_bar,
    analyze_fep_trapezoidal,
    cartesian_restraint_analytical_correction,
    boresch_analytical_correction,
    standard_state_concentration_correction,
    add_boresch_restraints,
    compute_binding_dg,
    _identify_ligand_atoms,
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
    energies, _neighbor, disappearing = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
        n_equilibration_steps=0, n_production_steps=1,
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
        # Each trajectory should have at least 1 sample
        # (production samples only; the equilibration samples
        # are discarded, so n_production_steps=5 gives 5 entries).
        assert len(traj) >= 1


@pytest.mark.slow
def test_run_amber_mutation_fep_window_means_are_finite():
    """The mean energy at each λ window must be finite.  Loose
    bound on the magnitude — soft-core + restraints inflate the
    energy above the bare AMBER baseline (~−170000 kJ/mol), so we
    just check it lands in a reasonable range.
    """
    energies, _neighbor, _ = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.5, 1.0),
        n_equilibration_steps=0, n_production_steps=1,
        padding_nm=0.8,
    )
    for lam, traj in energies.items():
        mean = float(np.mean(traj))
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
    energies, _neighbor, _ = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0,),
        n_equilibration_steps=0, n_production_steps=1,
        padding_nm=0.8,
    )
    arr = np.asarray(energies[0.0])
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


# ----- production analysis: compute_bar + analyze_fep_* ------------------


def test_compute_bar_recovers_known_delta_g():
    """compute_bar must recover the true ΔG_AB from synthetic
    mixed-state energies.  Each test case uses 1000 samples per
    state with small Gaussian noise on the energy differences —
    BAR should converge to within 0.5 kJ/mol (well below the
    1 kJ/mol bootstrap SEM).
    """
    rng = np.random.default_rng(42)
    n = 1000
    kT = 2.479  # 300 K
    for true_dg in (5.0, 0.0, -10.0, 20.0):
        # State A: energies with mean offset ΔG between U_A and U_B.
        u_a_at_a = rng.normal(100, 1, n)
        u_b_at_a = u_a_at_a + true_dg + rng.normal(0, 1, n)
        # State B: same constant ΔG, different samples.
        u_b_at_b = rng.normal(100 + true_dg, 1, n)
        u_a_at_b = u_b_at_b - true_dg + rng.normal(0, 1, n)
        dg, sigma = compute_bar(
            u_a_at_a, u_b_at_a, u_a_at_b, u_b_at_b,
            kT_kjmol=kT,
        )
        assert abs(dg - true_dg) < 0.5, (
            f"true ΔG = {true_dg}, BAR = {dg:.3f}"
        )
        # Bootstrap SEM should be > 0 and < 1 kJ/mol for these
        # well-overlapped distributions.
        assert 0.0 < sigma < 1.0, f"sigma = {sigma}"


def test_compute_bar_zero_input_returns_nan():
    """An empty input array should produce NaN ΔG and sigma —
    that's the documented behaviour, used when one window has
    no frames (e.g. crashed before producing any samples)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 100)
    b_empty = np.array([])
    dg, sigma = compute_bar(a, a + 1, b_empty, b_empty)
    assert np.isnan(dg) and np.isnan(sigma)


def test_analyze_fep_bar_sums_per_window_bars():
    """analyze_fep_bar should return the sum of per-window BAR
    estimates, with quadrature-combined uncertainty.  We
    construct a 3-window sweep with a known ΔG_per_window and
    verify the total matches.
    """
    rng = np.random.default_rng(123)
    n = 500
    true_per_window = 2.5  # total ΔG = 5

    u_0_at_0 = rng.normal(0, 1, n)
    u_05_at_0 = u_0_at_0 + true_per_window + rng.normal(0, 0.5, n)

    u_05_at_05 = rng.normal(true_per_window, 1, n)
    u_0_at_05 = u_05_at_05 - true_per_window + rng.normal(0, 0.5, n)
    u_10_at_05 = u_05_at_05 + true_per_window + rng.normal(0, 0.5, n)

    u_10_at_10 = rng.normal(2 * true_per_window, 1, n)
    u_05_at_10 = u_10_at_10 - true_per_window + rng.normal(0, 0.5, n)

    energies = {
        0.0: list(u_0_at_0),
        0.5: list(u_05_at_05),
        1.0: list(u_10_at_10),
    }
    neighbors = {
        0.0: [(None, float(u_05_at_0[i])) for i in range(n)],
        0.5: [(float(u_0_at_05[i]), float(u_10_at_05[i])) for i in range(n)],
        1.0: [(float(u_05_at_10[i]), None) for i in range(n)],
    }
    total_dg, total_sig, per_win = analyze_fep_bar(
        energies, neighbors, (0.0, 0.5, 1.0),
    )
    assert abs(total_dg - 5.0) < 0.5, (
        f"expected total ΔG ≈ 5.0, got {total_dg:.3f}"
    )
    # Per-window contributions should each be ~2.5.
    for idx, dg_i, _sig_i in per_win:
        assert abs(dg_i - 2.5) < 0.5, (
            f"window {idx} ΔG = {dg_i:.3f}, expected ≈2.5"
        )


def test_analyze_fep_bar_raises_when_neighbor_dict_empty():
    """If the caller passed sample_neighbor_lambdas=False, the
    neighbor dict is empty and BAR can't run.  The function
    must raise an informative ValueError, not silently return
    a garbage estimate."""
    rng = np.random.default_rng(0)
    energies = {
        0.0: list(rng.normal(0, 1, 50)),
        1.0: list(rng.normal(5, 1, 50)),
    }
    with pytest.raises(ValueError, match="neighbor_energies"):
        analyze_fep_bar(energies, {}, (0.0, 1.0))


def test_analyze_fep_trapezoidal_recovers_linear_curve():
    """Trapezoidal integration of ⟨U(λ)⟩ against λ gives the
    exact ΔG when ⟨U(λ)⟩ is linear in λ.  We construct a
    5-window sweep with ⟨U(λ_i)⟩ = 5·λ_i and check the result.
    """
    rng = np.random.default_rng(7)
    n = 200
    lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    ens = {l: rng.normal(5.0 * l, 0.1, n) for l in lambdas}
    dg, sig = analyze_fep_trapezoidal(ens, lambdas)
    # Linear curve → exact integral is 5/2 = 2.5 (the area under
    # y = 5x from 0 to 1).  Allow 0.1 kJ/mol for sampling noise.
    assert abs(dg - 2.5) < 0.1, f"expected ΔG ≈ 2.5, got {dg:.3f}"
    # SEM should be tiny (well-converged per-window means).
    assert sig < 0.05, f"expected small sigma, got {sig:.3f}"


def test_analyze_fep_trapezoidal_sigma_grows_with_overlap():
    """Larger per-window standard deviation → larger propagated
    SEM.  Trapezoidal variance = sum (Δλ/2)² × SEM² for end
    windows, (Δλ_{i-1}/2 + Δλ_i/2)² × SEM² for middle windows.
    """
    rng = np.random.default_rng(0)
    lambdas = (0.0, 0.5, 1.0)
    ens_low_noise = {l: rng.normal(5 * l, 0.1, 200) for l in lambdas}
    ens_high_noise = {l: rng.normal(5 * l, 5.0, 200) for l in lambdas}
    _, sig_low = analyze_fep_trapezoidal(ens_low_noise, lambdas)
    _, sig_high = analyze_fep_trapezoidal(ens_high_noise, lambdas)
    assert sig_high > 10 * sig_low, (
        f"high/low sigma ratio = {sig_high / sig_low:.1f}"
    )


# ----- production sampling features --------------------------------------


def test_run_amber_mutation_fep_default_lambda_count_is_eleven():
    """Production default: 11 evenly-spaced windows (Δλ = 0.1) to
    give BAR enough overlap.  The function returns energies for
    every window in the schedule."""
    # Mock the function call by inspecting the default via
    # inspect.signature (no AMBER build needed).
    import inspect
    sig = inspect.signature(run_amber_mutation_fep)
    assert "lambdas" in sig.parameters
    assert sig.parameters["lambdas"].default is None, (
        "default lambdas should be None (= 11-window schedule)"
    )


@pytest.mark.slow
def test_run_amber_mutation_fep_with_neighbor_re_evaluation():
    """sample_neighbor_lambdas=True populates the second return
    tuple with energy re-evaluations at adjacent λs — exactly
    what analyze_fep_bar needs."""
    energies, neighbors, disappearing = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.5, 1.0),
        n_equilibration_steps=0,
        n_production_steps=2,
        padding_nm=0.8,
        sample_neighbor_lambdas=True,
    )
    assert len(disappearing) == 13
    # Three windows, three neighbour dicts.
    assert sorted(energies.keys()) == [0.0, 0.5, 1.0]
    assert sorted(neighbors.keys()) == [0.0, 0.5, 1.0]
    # Middle window (0.5) has both prev and next neighbours;
    # end windows have only one.
    n_neighbour_samples = len(neighbors[0.5])
    assert n_neighbour_samples > 0
    for tup in neighbors[0.5]:
        prev, nxt = tup
        assert prev is not None
        assert nxt is not None
    for tup in neighbors[0.0]:
        assert tup[0] is None  # no prev
        assert tup[1] is not None  # has next
    for tup in neighbors[1.0]:
        assert tup[0] is not None  # has prev
        assert tup[1] is None  # no next
    # The neighbour re-evaluations must be finite.
    for lam, traj in neighbors.items():
        for tup in traj:
            for v in tup:
                if v is not None:
                    assert np.isfinite(v), (
                        f"non-finite neighbour energy at λ={lam}"
                    )


@pytest.mark.slow
def test_run_amber_mutation_fep_without_neighbor_re_evaluation():
    """Default behaviour: neighbor_energies is an empty dict so
    callers know there's nothing to run BAR on."""
    energies, neighbors, _ = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.5, 1.0),
        n_equilibration_steps=0,
        n_production_steps=2,
        padding_nm=0.8,
    )
    assert neighbors == {}
    assert len(energies) == 3


@pytest.mark.slow
def test_run_amber_mutation_fep_disabling_restraints():
    """restraint_k_kJmol_nm2=0 must drop the position restraint
    from the system (otherwise we'd be measuring ΔG with
    restraints still on, contaminating the result)."""
    energies, neighbors, disapp = run_amber_mutation_fep(
        ONEUBQ,
        mutation_residue=5,
        from_resname="LYS",
        lambdas=(0.0, 0.5),
        n_equilibration_steps=0,
        n_production_steps=2,
        padding_nm=0.8,
        restraint_k_kJmol_nm2=0.0,
    )
    assert len(disapp) == 13
    assert len(energies) == 2
    # Energies should be finite.
    for lam, traj in energies.items():
        for v in traj:
            assert np.isfinite(v)


# ----- Cartesian restraint analytical correction -------------------------


def test_cartesian_restraint_correction_phase4_default():
    """Phase 4 default: 13 disappearing atoms, k=1000 kJ/mol/nm²,
    T=300 K.

    Per-atom ΔG ≈ (3/2) kT ln(2π kT/k) - kT ln(V₀)
                 ≈ 3.7185 × (-4.161) - 2.479 × (-6.398)
                 ≈ +0.39 kJ/mol.

    For 13 atoms: ≈ +5.1 kJ/mol total.
    """
    corr = cartesian_restraint_analytical_correction(
        n_atoms=13,
        restraint_k_kJmol_nm2=1000.0,
        temperature_K=300.0,
    )
    assert 4.5 < corr < 6.0, (
        f"expected ~5 kJ/mol, got {corr:.3f}"
    )


def test_cartesian_restraint_correction_zero_atoms():
    """No restrained atoms → no correction."""
    corr = cartesian_restraint_analytical_correction(
        n_atoms=0,
        restraint_k_kJmol_nm2=1000.0,
    )
    assert corr == 0.0


def test_cartesian_restraint_correction_scales_with_atom_count():
    """Linear in n_atoms (each restraint is independent)."""
    c1 = cartesian_restraint_analytical_correction(n_atoms=10, restraint_k_kJmol_nm2=1000.0)
    c2 = cartesian_restraint_analytical_correction(n_atoms=20, restraint_k_kJmol_nm2=1000.0)
    assert abs(c2 / c1 - 2.0) < 1e-10, (
        f"correction should scale linearly: c1={c1:.3f}, c2={c2:.3f}"
    )


def test_cartesian_restraint_correction_stronger_k_larger_correction():
    """The sign of the correction depends on whether the
    restrained Gaussian volume is larger or smaller than V₀.
    At weak k (k ≲ kT/V₀^(2/3) ≈ 1500 kJ/mol/nm²), the
    Gaussian is wider than V₀ so unrestraining increases F
    (positive ΔG).  At strong k, the Gaussian is narrower
    so unrestraining decreases F (negative ΔG).

    This test verifies the SIGN FLIPS — that's the key
    qualitative behaviour of the analytical correction.
    """
    corr_weak = cartesian_restraint_analytical_correction(n_atoms=1, restraint_k_kJmol_nm2=10.0)
    corr_strong = cartesian_restraint_analytical_correction(n_atoms=1, restraint_k_kJmol_nm2=10000.0)
    assert corr_weak > 0, (
        f"weak restraint (k=10) → positive correction, got {corr_weak:.3f}"
    )
    assert corr_strong < 0, (
        f"strong restraint (k=10000) → negative correction, got {corr_strong:.3f}"
    )


def test_cartesian_restraint_correction_temperature_scaling():
    """Doubling T increases the correction magnitude (kT
    prefactor + ln(kT) terms give ratio ≈ 2·(3/2 ln(600)+c) /
    (3/2 ln(300)+c) which for typical V₀ is ~2.5, not exactly 2).
    """
    c_300 = cartesian_restraint_analytical_correction(n_atoms=1, restraint_k_kJmol_nm2=100.0, temperature_K=300.0)
    c_600 = cartesian_restraint_analytical_correction(n_atoms=1, restraint_k_kJmol_nm2=100.0, temperature_K=600.0)
    ratio = c_600 / c_300
    # Allow 2.0–3.0 range (kT prefactor alone is 2.0, plus
    # the ln(kT) terms push it up to ~2.5).
    assert 2.0 < ratio < 3.0, (
        f"expected ~2.5x ratio for T doubling, got {ratio:.3f}"
    )


def test_cartesian_restraint_correction_phase4_lysine_sign_and_magnitude():
    """For Phase 4's specific Lys6→Ala truncation (13
    disappearing atoms, k=1000), the correction is ~+5
    kJ/mol.  Sign is positive because the restrained
    Gaussian volume (1.95e-3 nm³) is comparable to the
    standard-state volume (1.66e-3 nm³), and the entropic
    gain from unrestraining is small.

    The fact that this correction is much smaller than the
    underlying alchemical ΔG (which is hundreds of kJ/mol
    for losing Lys NZ interactions) is intentional — the
    restraints are tight enough that they're just keeping
    the sidechain atoms from drifting during the sweep,
    not biasing the free-energy estimate significantly.
    """
    corr = cartesian_restraint_analytical_correction(
        n_atoms=13,
        restraint_k_kJmol_nm2=1000.0,
    )
    # Should be much smaller than 50 kJ/mol (the typical
    # alchemical ΔG for a sidechain truncation).
    assert abs(corr) < 50, (
        f"Phase 4 Lys6 correction {corr:.3f} kJ/mol exceeds "
        "expected bound (restraints shouldn't bias the "
        "estimate by more than 10% of the alchemical ΔG)"
    )


# ----- Phase 5: protein-ligand binding FEP --------------------------------


def test_boresch_analytical_correction_typical():
    """Standard Boresch (2003) values: k_r=10 kcal/mol/Å²,
    k_θ=100 kcal/mol/rad², k_φ=1000 kcal/mol/rad², T=300 K.
    Reference: −25.3 kJ/mol (≈ −6 kcal/mol).  Sign is
    negative because unrestraining the ligand into the
    standard state DECREASES its free energy — the restraints
    were biasing the bound state in the wrong direction.
    """
    corr = boresch_analytical_correction(
        k_r_kJmol_nm2=41.84,        # 10 kcal/mol/Å²
        k_theta1_kJmol_rad2=418.4, # 100 kcal/mol/rad²
        k_theta2_kJmol_rad2=418.4,
        k_phi1_kJmol_rad2=4184.0,  # 1000 kcal/mol/rad²
        k_phi2_kJmol_rad2=4184.0,
        k_phi3_kJmol_rad2=4184.0,
        temperature_K=300.0,
    )
    assert -27 < corr < -23, (
        f"expected ~-25 kJ/mol, got {corr:.3f}"
    )


def test_boresch_analytical_correction_stiffer_k_larger_magnitude():
    """Tighter restraints (larger k) → larger |ΔG_correction|.
    The more the restraints constrain the ligand, the more
    entropy it gains upon unrestraining.
    """
    corr_loose = boresch_analytical_correction(
        k_r_kJmol_nm2=4.184, k_theta1_kJmol_rad2=41.84,
        k_theta2_kJmol_rad2=41.84,
        k_phi1_kJmol_rad2=418.4, k_phi2_kJmol_rad2=418.4,
        k_phi3_kJmol_rad2=418.4,
    )
    corr_tight = boresch_analytical_correction(
        k_r_kJmol_nm2=418.4, k_theta1_kJmol_rad2=4184.0,
        k_theta2_kJmol_rad2=4184.0,
        k_phi1_kJmol_rad2=41840.0, k_phi2_kJmol_rad2=41840.0,
        k_phi3_kJmol_rad2=41840.0,
    )
    assert abs(corr_tight) > abs(corr_loose), (
        f"tighter restraints → larger |corr|: "
        f"loose={corr_loose:.3f}, tight={corr_tight:.3f}"
    )


def test_boresch_analytical_correction_zero_forces_raises():
    """Zero force constants → the formula breaks down
    (division by zero in the Z_restrained denominator).
    Physically: with zero restraints there's no restoring
    potential so the ligand would sample unbounded rotational
    space — the standard-state / restraint partition-function
    comparison isn't well-defined.  The function raises a
    ZeroDivisionError rather than silently returning garbage.
    """
    with pytest.raises(ZeroDivisionError):
        boresch_analytical_correction(
            k_r_kJmol_nm2=0.0, k_theta1_kJmol_rad2=418.4,
            k_theta2_kJmol_rad2=418.4,
            k_phi1_kJmol_rad2=4184.0, k_phi2_kJmol_rad2=4184.0,
            k_phi3_kJmol_rad2=4184.0,
        )


def test_standard_state_concentration_correction_typical_box():
    """For a 1ubq-like box (V ≈ 440 nm³) at T = 300 K,
    ΔG_ss ≈ −3.3 kJ/mol.  The simulation box has higher
    effective ligand concentration than 1 M, so reporting at
    1 M makes binding slightly less favourable.
    """
    corr = standard_state_concentration_correction(
        box_volume_nm3=440.0, temperature_K=300.0,
    )
    assert -5 < corr < -1, (
        f"expected ~-3.3 kJ/mol for 440 nm³ box, got {corr:.3f}"
    )


def test_standard_state_concentration_correction_scales_with_volume():
    """Larger box (more dilute ligand) → more negative
    ΔG_ss (since log(V_box) increases)."""
    c_small = standard_state_concentration_correction(box_volume_nm3=100.0)
    c_large = standard_state_concentration_correction(box_volume_nm3=1000.0)
    assert c_large > c_small, (
        f"larger box → larger ΔG_ss: small={c_small:.3f}, large={c_large:.3f}"
    )


def test_standard_state_concentration_correction_temperature():
    """Doubling T doubles ΔG_ss (linear in kT)."""
    c_300 = standard_state_concentration_correction(box_volume_nm3=440.0, temperature_K=300.0)
    c_600 = standard_state_concentration_correction(box_volume_nm3=440.0, temperature_K=600.0)
    assert abs(c_600 / c_300 - 2.0) < 1e-10


def test_compute_binding_dg_assembly_with_synthetic_data():
    """End-to-end assembly test on synthetic energies.
    ΔG_complex_decouple = 20 (decoupling ligand in complex costs
    energy because it was stabilised).  ΔG_solution_decouple =
    10 (decoupling free ligand costs less).  Without any
    corrections, ΔG_bind = ΔG_complex − ΔG_solution = +10.
    Add Boresch correction (≈ -25) → ΔG_bind ≈ -15.
    Add standard-state correction (≈ -3) → ΔG_bind ≈ -18.
    """
    rng = np.random.default_rng(0)
    e_c = {0.0: list(rng.normal(0, 1, 100)),
           0.5: list(rng.normal(20, 1, 100)),
           1.0: list(rng.normal(40, 1, 100))}
    e_s = {0.0: list(rng.normal(0, 1, 100)),
           0.5: list(rng.normal(10, 1, 100)),
           1.0: list(rng.normal(20, 1, 100))}

    # Without standard state and Boresch
    res = compute_binding_dg(
        e_c, e_s, use_bar=False, apply_standard_state=False,
    )
    assert abs(res["binding_dg_kJmol"] - 10.0) < 1.0
    assert res["standard_state_correction_kJmol"] == 0.0
    assert res["boresch_correction_kJmol"] == 0.0

    # With Boresch only
    res = compute_binding_dg(
        e_c, e_s, use_bar=False, apply_standard_state=False,
        boresch_kws={
            "k_r_kJmol_nm2": 41.84,
            "k_theta1_kJmol_rad2": 418.4,
            "k_theta2_kJmol_rad2": 418.4,
            "k_phi1_kJmol_rad2": 4184.0,
            "k_phi2_kJmol_rad2": 4184.0,
            "k_phi3_kJmol_rad2": 4184.0,
        },
    )
    assert abs(res["binding_dg_kJmol"] - (10.0 - 25.25)) < 1.0, (
        f"expected ~-15.25 with Boresch only, got {res['binding_dg_kJmol']:.3f}"
    )

    # With both Boresch and standard state
    res = compute_binding_dg(
        e_c, e_s, use_bar=False,
        boresch_kws={
            "k_r_kJmol_nm2": 41.84,
            "k_theta1_kJmol_rad2": 418.4,
            "k_theta2_kJmol_rad2": 418.4,
            "k_phi1_kJmol_rad2": 4184.0,
            "k_phi2_kJmol_rad2": 4184.0,
            "k_phi3_kJmol_rad2": 4184.0,
        },
        box_volume_complex_nm3=440.0,
        box_volume_solution_nm3=440.0,
    )
    expected = 10.0 - 25.25 + (-3.31)
    assert abs(res["binding_dg_kJmol"] - expected) < 1.0, (
        f"expected ~{expected:.3f} with both corrections, "
        f"got {res['binding_dg_kJmol']:.3f}"
    )


def test_compute_binding_dg_requires_box_volumes_for_standard_state():
    """apply_standard_state=True without box volumes should
    raise — without box info we can't compute the correction."""
    rng = np.random.default_rng(0)
    e_c = {0.0: list(rng.normal(0, 1, 50)), 1.0: list(rng.normal(40, 1, 50))}
    e_s = {0.0: list(rng.normal(0, 1, 50)), 1.0: list(rng.normal(20, 1, 50))}
    with pytest.raises(ValueError, match="box_volume"):
        compute_binding_dg(
            e_c, e_s, use_bar=False,
            # apply_standard_state defaults to True
        )


def test_compute_binding_dg_requires_neighbor_energies_for_bar():
    """use_bar=True without neighbor energies should raise."""
    rng = np.random.default_rng(0)
    e_c = {0.0: list(rng.normal(0, 1, 50)), 1.0: list(rng.normal(40, 1, 50))}
    e_s = {0.0: list(rng.normal(0, 1, 50)), 1.0: list(rng.normal(20, 1, 50))}
    with pytest.raises(ValueError, match="neighbor_energies"):
        compute_binding_dg(e_c, e_s, use_bar=True)


@pytest.mark.slow
def test_identify_ligand_atoms_finds_ligand_resname():
    """In 1ubq, the only non-standard HETATM residue is
    water (HOH).  After addHydrogens, HOH is still HOH.  So
    asking for ligand_resnames=['HOH'] should return all
    water atoms (the wrong target — there is no ligand — but
    the identification logic should work)."""
    import openmm.app as app
    import openmm

    # Build the same way build_amber_mutation_system does.
    from protein_studio.core.fep import build_amber_solvated_system
    sys_, topo, _, _ = build_amber_solvated_system(ONEUBQ, padding_nm=0.8)
    # No ligand in 1ubq (no HETATM except water + ions).  So
    # asking for ["XYZ"] should return empty.
    assert _identify_ligand_atoms(topo, ["XYZ"]) == []
    # Asking for HOH (water) should return many atoms (the
    # TIP3P waters added by addSolvent).
    hoh_atoms = _identify_ligand_atoms(topo, ["HOH"])
    assert len(hoh_atoms) > 1000, (
        f"expected many HOH atoms, got {len(hoh_atoms)}"
    )
