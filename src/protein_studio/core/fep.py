"""Phase 1: Free-Energy Perturbation (FEP) framework — minimal scaffolding.

Goal: verify the protein-studio ↔ OpenMM integration and the
lambda-coupling plumbing. A real FEP calculation (with two distinct
thermodynamic states connected by an alchemical transformation, a
lambda schedule, soft-core potentials and BAR analysis) is Phase 2 +
Phase 3. Phase 1 just gets the integration going and proves the
machinery works end-to-end.

What λ controls in Phase 1
----------------------------
A single lambda parameter ``lam`` scales the strength of the nonbonded
force group:

    lam = 0.0   harmonic position restraints only — no nonbonded
    lam = 1.0   harmonic restraints + full nonbonded (LJ + Coulomb)

This isn't an alchemical transformation (no atoms appearing or
disappearing, no bond terms changing). It's a minimal "decouple
intramolecular forces" demonstration of the lambda-coupling workflow
that any FEP calculation needs. The energies sampled at the two
endpoints differ — that's the ``ΔG`` we'd feed to BAR in Phase 3.

Public API
----------
* :class:`AlchemicalState` — one thermodynamic state (one Molecule + a
  lambda value).
* :class:`LambdaWindow` — sampling window at one lambda value.
* :class:`FEPResult` — aggregated energies, ``ΔG`` and standard errors.
* :func:`build_openmm_system` — convert a Molecule to an OpenMM System
  with a lambda-scaled nonbonded force.
* :func:`run_window` — sample energies at one lambda value.
* :func:`run_fep` — sample all windows in a lambda schedule and return
  the aggregated :class:`FEPResult`.

Dependencies
------------
OpenMM 8.x (``conda install -c conda-forge openmm``). If OpenMM isn't
installed, all functions raise :class:`ImportError` with a clear
install hint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .molecule import Molecule


# Element → atomic mass (a.m.u.). Used by build_openmm_system when the
# Molecule atoms have an ``element`` field but no mass. The values come
# from OpenMM's element data; only the biologically common ones are
# listed (anything else falls back to the most common isotope mass of
# the element symbol).
_ELEMENT_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "S": 32.06, "P": 30.974, "F": 18.998, "CL": 35.45,
    "BR": 79.904, "I": 126.904, "FE": 55.845, "ZN": 65.38,
    "MG": 24.305, "CA": 40.078, "MN": 54.938, "NA": 22.990,
    "K": 39.098, "CU": 63.546, "SE": 78.971,
}


@dataclass
class AlchemicalState:
    """One thermodynamic state (a Molecule + a lambda value)."""
    mol: Molecule
    lambda_value: float = 1.0


@dataclass
class LambdaWindow:
    """Sampling window at one lambda value.

    The ``energies`` list is populated by :func:`run_window` with the
    reduced potential (kJ/mol) at each sampled frame. The mean and SEM
    are computed by :func:`run_fep` and stored on the matching
    :class:`FEPResult`.
    """
    state: AlchemicalState
    lambda_value: float
    n_steps: int = 100
    energies: List[float] = field(default_factory=list)


@dataclass
class FEPResult:
    """Aggregated FEP results across a lambda schedule.

    All energy values are in kJ/mol. ``delta_g`` and ``delta_g_sem`` are
    computed by trapezoidal integration over the mean energies — a
    Phase 1 stand-in for proper BAR/MBAR analysis (Phase 3).
    """
    lambdas: List[float] = field(default_factory=list)
    mean_energies: List[float] = field(default_factory=list)
    sem_energies: List[float] = field(default_factory=list)
    delta_g: float = 0.0
    delta_g_sem: float = 0.0


def _require_openmm():
    """Import OpenMM or raise with an install hint."""
    try:
        import openmm  # noqa: F401
        import openmm.app  # noqa: F401
        import openmm.unit as unit  # noqa: F401
        return openmm, unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM is required for protein_studio.core.fep. "
            "Install with `conda install -c conda-forge openmm`."
        ) from exc


def _protein_atoms(mol: Molecule):
    """Return the non-hetero atom list (skip ligands / waters)."""
    return [a for a in mol.atoms if not a.is_hetero]


def build_openmm_system(mol: Molecule, lambda_value: float = 1.0,
                        temperature_kelvin: float = 300.0):
    """Build an OpenMM System for a protein-studio ``Molecule``.

    Sets up:

    * a Topology with one residue per protein residue (chain + name
      preserved from the Molecule so OpenMM reports them sanely).
    * a single :class:`openmm.HarmonicBondForce` connecting every pair
      of protein atoms within 4 Å of each other — a minimal bonded
      topology that keeps the structure from drifting without a full
      force-field parameterisation.
    * a single :class:`openmm.NonbondedForce` (no PBC, no
      CustomNonbondedForce yet — that's Phase 2). The lambda scales
      the LJ + Coulomb terms linearly: ``lam * LJ + lam * Coulomb``.
    * one :class:`openmm.HarmonicForce` per atom restraining it to its
      initial coordinate (so vacuum simulations don't blow up at
      lambda=0 when nonbonded is off).

    Parameters
    ----------
    mol : Molecule
        Source structure (only non-hetero atoms are used).
    lambda_value : float
        Coupling parameter in [0, 1]. 0 = no nonbonded, 1 = full
        nonbonded.
    temperature_kelvin : float
        Used only by the caller's integrator; not stored on the
        System.

    Returns
    -------
    system : openmm.System
        The constructed system.
    topology : openmm.app.Topology
        Topology matching ``system``.
    positions : list[openmm.unit.Quantity]
        Initial positions in nanometres (OpenMM's unit convention).
    """
    openmm, unit = _require_openmm()
    atoms = _protein_atoms(mol)
    if not atoms:
        raise ValueError("Molecule has no non-hetero atoms; nothing to simulate.")

    # ---- topology ----
    topology = openmm.app.Topology()
    chain_id_to_chain: dict[str, object] = {}
    res_to_top_res: dict = {}
    for a in atoms:
        cid = a.chain_id
        chain = chain_id_to_chain.get(cid)
        if chain is None:
            chain = topology.addChain(cid)
            chain_id_to_chain[cid] = chain
        res_key = (cid, a.residue_name, a.residue_number)
        top_res = res_to_top_res.get(res_key)
        if top_res is None:
            top_res = topology.addResidue(a.residue_name, chain)
            res_to_top_res[res_key] = top_res
        element = openmm.app.Element.getBySymbol(a.element or "C")
        topology.addAtom(a.name, element, top_res)

    # ---- positions in nm ----
    positions = [
        openmm.unit.Quantity(np.asarray(a.coord, dtype=float) / 10.0,
                              unit.nanometer)
        for a in atoms
    ]

    # ---- system ----
    system = openmm.System()
    for a in atoms:
        elem = a.element or "C"
        mass = _ELEMENT_MASS.get(elem.upper(), 12.011)
        system.addParticle(mass)

    # Position restraint to initial coordinates. Keeps vacuum
    # simulations bounded when lambda=0 (no nonbonded to hold the
    # structure together). OpenMM doesn't expose a built-in
    # HarmonicRestraintForce in this version, so we use one
    # CustomExternalForce with one particle index per atom.
    # kJ/mol/nm^2 = 10 (gentle enough for short Phase-1 simulations).
    restraint_force = openmm.CustomExternalForce(
        "k_restraint*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restraint_force.addGlobalParameter("k_restraint", 10.0)
    restraint_force.addPerParticleParameter("x0")
    restraint_force.addPerParticleParameter("y0")
    restraint_force.addPerParticleParameter("z0")
    for i, p in enumerate(positions):
        xyz = p.value_in_unit(unit.nanometer)
        restraint_force.addParticle(i, (xyz[0], xyz[1], xyz[2]))
    system.addForce(restraint_force)

    # Bonds between protein atoms within 4 Å of each other — minimal
    # covalent topology so the structure doesn't fall apart.
    bond_force = openmm.HarmonicBondForce()
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = float(np.linalg.norm(atoms[i].coord - atoms[j].coord))
            if d <= 4.0:
                bond_force.addBond(i, j, d / 10.0, 1000.0)  # nm, kJ/mol/nm^2
    system.addForce(bond_force)

    # Nonbonded force scaled by lambda. No PBC: the (small) system
    # doesn't need it for a Phase 1 sanity check.
    nb = openmm.NonbondedForce()
    for i, a in enumerate(atoms):
        # Rough element-based LJ params (sigma in nm, epsilon in kJ/mol).
        # Real FEP (Phase 2) reads these from a force field; Phase 1
        # uses simple element defaults so we don't pull in OpenFF.
        elem = (a.element or "C").upper()
        if elem == "H":
            sigma, epsilon, charge = 0.12, 0.10, 0.0
        elif elem == "C":
            sigma, epsilon, charge = 0.34, 0.36, 0.0
        elif elem == "N":
            sigma, epsilon, charge = 0.33, 0.71, 0.0
        elif elem == "O":
            sigma, epsilon, charge = 0.31, 0.88, 0.0
        elif elem == "S":
            sigma, epsilon, charge = 0.36, 1.86, 0.0
        else:
            sigma, epsilon, charge = 0.34, 0.36, 0.0
        nb.addParticle(charge, sigma, epsilon)

    # Phase 1 lambda coupling: store lambda as a global parameter and
    # multiply the entire force energy by it via a CustomNonbondedForce.
    # Easier: wrap the NonbondedForce's energy by a per-step scale —
    # OpenMM lets us set force group's contribution via a
    # CustomExternalForce that adds lam * U_nb? No, simpler: turn the
    # NonbondedForce on/off via its ``setForceGroup`` and the
    # integrator's ``getIntegrationForceGroups``. For Phase 1 we just
    # multiply every energy evaluation by ``lam`` by scaling epsilon
    # and charge by lam at force-build time. (Phase 2 replaces this
    # with proper soft-core potentials.)
    if lambda_value != 1.0:
        for i in range(nb.getNumParticles()):
            (q, sig, eps) = nb.getParticleParameters(i)
            nb.setParticleParameters(i, q * lambda_value, sig, eps * lambda_value)
    system.addForce(nb)
    return system, topology, positions


def run_window(mol: Molecule, lambda_value: float = 1.0,
               n_steps: int = 100,
               temperature_kelvin: float = 300.0,
               timestep_fs: float = 1.0):
    """Sample the reduced potential at one lambda value.

    Builds the OpenMM system, minimises, then runs ``n_steps`` of
    Langevin MD. Returns the list of per-frame reduced potentials
    (kJ/mol). ``n_steps`` ≤ 0 returns an empty list — useful for
    "no sampling" smoke tests.
    """
    openmm, unit = _require_openmm()
    if n_steps <= 0:
        return []
    system, topology, positions = build_openmm_system(
        mol, lambda_value=lambda_value, temperature_kelvin=temperature_kelvin,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    # OpenMM 8 dropped Context.minimizeEnergy(); use the module-level
    # LocalEnergyMinimizer instead. Phase 1 keeps it simple —
    # skip minimization if it's not available, since vacuum
    # simulations don't need it for a sanity check.
    if hasattr(openmm, "LocalEnergyMinimizer"):
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=20)

    energies: List[float] = []
    # Reporters see the per-frame state; we just collect energies.
    state = context.getState(getEnergy=True)
    energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

    for _ in range(n_steps):
        integrator.step(1)
        state = context.getState(getEnergy=True)
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    return energies


def run_fep(mol: Molecule,
            lambdas: Optional[Sequence[float]] = None,
            n_steps: int = 100,
            temperature_kelvin: float = 300.0):
    """Run a Phase 1 lambda-coupling sweep and return :class:`FEPResult`.

    Each lambda in ``lambdas`` gets its own :class:`LambdaWindow`; the
    energies are sampled by :func:`run_window` and aggregated. The
    ``ΔG`` is computed by trapezoidal integration over the mean
    energies (Phase 1 stand-in; Phase 3 will swap this for proper
    BAR/MBAR).

    Parameters
    ----------
    mol : Molecule
    lambdas : sequence of float, default ``(0.0, 0.25, 0.5, 0.75, 1.0)``
    n_steps : int
        MD steps per window.
    temperature_kelvin : float

    Returns
    -------
    FEPResult
    """
    if lambdas is None:
        lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    state = AlchemicalState(mol=mol, lambda_value=1.0)

    result = FEPResult()
    means: List[float] = []
    sems: List[float] = []
    for lam in lambdas:
        energies = run_window(mol, lambda_value=lam, n_steps=n_steps,
                              temperature_kelvin=temperature_kelvin)
        if not energies:
            means.append(float("nan"))
            sems.append(float("nan"))
        else:
            arr = np.asarray(energies, dtype=float)
            means.append(float(arr.mean()))
            sems.append(float(arr.std(ddof=1) / math.sqrt(len(arr))))
    result.lambdas = list(lambdas)
    result.mean_energies = means
    result.sem_energies = sems

    # Trapezoidal integration for ΔG (Phase 1 stand-in for BAR).
    finite = [(l, m) for l, m in zip(lambdas, means)
              if not math.isnan(m)]
    if len(finite) >= 2:
        ls, ms = zip(*finite)
        result.delta_g = float(np.trapezoid(ms, x=ls))
        if result.sem_energies and any(
            not math.isnan(s) for s in result.sem_energies
        ):
            # Error propagation: integrate SEM via trapezoidal too
            # (a conservative upper bound on the integrator's
            # contribution).
            sem_finite = [
                s for s in result.sem_energies
                if not math.isnan(s)
            ]
            if len(sem_finite) == len(finite):
                result.delta_g_sem = float(
                    math.sqrt(sum(s * s for s in sem_finite))
                )
    return result


# ============================================================================
# Phase 2: alchemical transformation — real atom deletion/addition with
# soft-core potentials. Adds an ``AlchemicalMapping`` data class plus
# ``build_alchemical_system`` / ``run_alchemical_window`` /
# ``run_alchemical_fep`` that connect two Molecules through an
# alchemical path.
# ============================================================================


@dataclass
class AlchemicalMapping:
    """Atom correspondence between two Molecules for an alchemical
    transformation.

    ``core_atom_pairs`` lists ``(a_idx, b_idx)`` tuples for atoms that
    exist in both states at corresponding positions. The core atoms
    keep their full nonbonded interactions throughout the lambda sweep.

    ``disappearing_atom_indices`` are indices in ``state_a`` for atoms
    that exist only at lambda=0 (i.e. they vanish as lambda goes 0→1).

    ``appearing_atom_indices`` are indices in ``state_b`` for atoms that
    exist only at lambda=1 (i.e. they appear as lambda goes 0→1).

    For a typical sidechain mutation (e.g. Ala→Gly, decouple CB),
    ``disappearing_atom_indices`` is ``[cb_idx]`` and
    ``appearing_atom_indices`` is empty. For a more elaborate
    transformation (e.g. mutation to a larger residue) both lists
    carry entries.
    """
    state_a: Molecule
    state_b: Molecule
    core_atom_pairs: List[Tuple[int, int]] = field(default_factory=list)
    disappearing_atom_indices: List[int] = field(default_factory=list)
    appearing_atom_indices: List[int] = field(default_factory=list)


# Soft-core LJ parameters (Beutler et al. 1994 style).  alpha=0.5
# (in nm^2) is the conventional choice — it keeps the soft-core
# potential finite at r=0 when lambda→0 or lambda→1.
SOFT_CORE_ALPHA_NM2 = 0.5

# Charge scaling used for disappearing / appearing atoms: Coulomb
# is multiplied by lambda or (1-lambda) directly.  No soft-core
# needed for charges — just linear scaling.
#
# (For a more thorough Phase 2.1 we could add a separate soft-core
# for charge (Zacharias et al. 1994), but linear scaling is a clean
# stand-in for the verification we're after here.)


def _non_hetero_atoms(mol: Molecule):
    """Non-hetero atom list with stable indices."""
    return [a for a in mol.atoms if not a.is_hetero]


def _sc_lj_formula(lam_global: str = "lam") -> str:
    """Return the OpenMM energy-expression string for soft-core LJ.

    U(r; λ) = λ · 4ε · [σ^12 / (α(1-λ)^2 + r^6)^2  −  σ^6 / (α(1-λ)^2 + r^6)]

    where α is the global soft-core alpha (in nm^2), ε is the
    per-pair epsilon (in kJ/mol), σ is the per-pair sigma (in nm),
    and lam is the global lambda parameter.
    """
    return (
        f"{lam_global} * 4 * epsilon * "
        f"((sigma^6 / (alpha * (1 - {lam_global})^2 + r^2)^2)^2 "
        f"- (sigma^6 / (alpha * (1 - {lam_global})^2 + r^2))); "
        f"alpha = {SOFT_CORE_ALPHA_NM2}"
    )


def build_alchemical_system(mapping: AlchemicalMapping,
                            lambda_value: float = 0.0,
                            temperature_kelvin: float = 300.0):
    """Build an OpenMM System for an alchemical transformation.

    Topology strategy
    -----------------
    The hybrid atom list used by OpenMM is:

      [core_a_0, core_a_1, ..., disappearing_0, ..., appearing_0, ...]

    Core atoms use the first len(slot) particle indices. Disappearing
    atoms use the next ``len(disappearing_atom_indices)`` indices.
    Appearing atoms use the last ``len(appearing_atom_indices)``
    indices. Position of core atoms comes from ``state_a`` (positions
    match via the ``core_atom_pairs`` map); disappearing atoms are
    placed at their state_a positions; appearing atoms at their
    state_b positions.

    Force strategy
    --------------
    Three force groups:

      * HarmonicRestraint (custom external) on every atom — keeps
        vacuum simulations bounded; the user is responsible for
        switching to a solvated system if needed (Phase 3).

      * HarmonicBondForce for state_a bonds only (≤ 4 Å). Bonds are
        not alchemically transformed in Phase 2.

      * NonbondedForce for **core–core** pairs at full strength.

      * CustomNonbondedForce for every pair that involves at least
        one *disappearing* or *appearing* atom. Uses the soft-core
        LJ formula above. Coulomb is handled by a separate
        CustomNonbondedForce (linear λ scaling) so we keep the two
        effects decoupled.

    The Coulomb soft-core uses linear scaling for now — a proper
    Zacharias-style soft-core is Phase 2.1.
    """
    openmm, unit = _require_openmm()
    a_atoms = _non_hetero_atoms(mapping.state_a)
    b_atoms = _non_hetero_atoms(mapping.state_b)

    core_pair_set = set(tuple(sorted(p)) for p in mapping.core_atom_pairs)
    n_core = len(mapping.core_atom_pairs)
    n_dis = len(mapping.disappearing_atom_indices)
    n_app = len(mapping.appearing_atom_indices)
    n_total = n_core + n_dis + n_app
    if n_total == 0:
        raise ValueError("AlchemicalMapping must declare at least one atom.")

    # ---- topology ----
    # OpenMM requires all atoms of a residue to be added contiguously.
    # Build a residue-grouped layout: for each residue in state_a,
    # append its core atoms first, then its disappearing atoms.
    # Appearing atoms from state_b get their own residue blocks
    # appended at the end (one per state_b residue that contributes).
    topology = openmm.app.Topology()
    chain_cache: dict = {}
    res_cache: dict = {}
    # hybrid_atoms / hybrid_labels parallel arrays; index in these
    # arrays = particle index in the OpenMM System.
    hybrid_atoms: list = []  # (top_atom, coord_angstrom, src_atom)
    hybrid_labels: list = []

    # Index lookup tables — for fast "is this atom core / disappearing / appearing?"
    core_a_set = {a for a, _ in mapping.core_atom_pairs}
    disappearing_a_set = set(mapping.disappearing_atom_indices)
    appearing_b_set = set(mapping.appearing_atom_indices)

    # Mapping residue key → set of state_a atom indices that live there.
    a_res_to_indices: dict = {}
    for idx, src in enumerate(a_atoms):
        rkey = (src.chain_id, src.residue_number)
        a_res_to_indices.setdefault(rkey, []).append(idx)
    # Same for state_b (appearing atoms only).
    b_res_to_indices: dict = {}
    for idx, src in enumerate(b_atoms):
        rkey = (src.chain_id, src.residue_number)
        b_res_to_indices.setdefault(rkey, []).append(idx)

    def _add_top_atom(src_atom, kind: str):
        cid = src_atom.chain_id or "X"
        chain = chain_cache.get(cid)
        if chain is None:
            chain = topology.addChain(cid)
            chain_cache[cid] = chain
        rkey = (cid, src_atom.residue_name, src_atom.residue_number)
        top_res = res_cache.get(rkey)
        if top_res is None:
            top_res = topology.addResidue(src_atom.residue_name, chain)
            res_cache[rkey] = top_res
        elem = openmm.app.Element.getBySymbol(src_atom.element or "C")
        top_atom = topology.addAtom(src_atom.name, elem, top_res)
        return top_atom

    # Pass 1: every residue in state_a — add its core atoms (in their
    # original residue order) followed by any disappearing atoms.
    for rkey, indices in a_res_to_indices.items():
        for idx in indices:
            src = a_atoms[idx]
            if idx in core_a_set:
                top_atom = _add_top_atom(src, "core")
                hybrid_atoms.append((top_atom, src.coord, src))
                hybrid_labels.append("core")
        for idx in indices:
            if idx in disappearing_a_set:
                src = a_atoms[idx]
                top_atom = _add_top_atom(src, "disappearing")
                hybrid_atoms.append((top_atom, src.coord, src))
                hybrid_labels.append("disappearing")

    # Pass 2: appearing atoms from state_b (each residue as a contiguous
    # block).
    for rkey, indices in b_res_to_indices.items():
        any_appearing = any(idx in appearing_b_set for idx in indices)
        if not any_appearing:
            continue
        for idx in indices:
            if idx in appearing_b_set:
                src = b_atoms[idx]
                top_atom = _add_top_atom(src, "appearing")
                hybrid_atoms.append((top_atom, src.coord, src))
                hybrid_labels.append("appearing")

    positions = [
        openmm.unit.Quantity(np.asarray(c, dtype=float) / 10.0, unit.nanometer)
        for _, c, _ in hybrid_atoms
    ]

    # ---- system ----
    system = openmm.System()
    for _, _, src_atom in hybrid_atoms:
        elem = (src_atom.element or "C").upper()
        mass = _ELEMENT_MASS.get(elem, 12.011)
        system.addParticle(mass)

    # Position restraints on every atom.
    restraint_force = openmm.CustomExternalForce(
        "k_restraint*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restraint_force.addGlobalParameter("k_restraint", 10.0)
    restraint_force.addPerParticleParameter("x0")
    restraint_force.addPerParticleParameter("y0")
    restraint_force.addPerParticleParameter("z0")
    for i, p in enumerate(positions):
        xyz = p.value_in_unit(unit.nanometer)
        restraint_force.addParticle(i, (xyz[0], xyz[1], xyz[2]))
    system.addForce(restraint_force)

    # Bonds for state_a only — bonds are not part of the alchemical
    # change in Phase 2.
    bond_force = openmm.HarmonicBondForce()
    for i in range(n_core + n_dis):
        src_i = hybrid_atoms[i][2]
        for j in range(i + 1, n_core + n_dis):
            src_j = hybrid_atoms[j][2]
            d = float(np.linalg.norm(src_i.coord - src_j.coord))
            if d <= 4.0:
                bond_force.addBond(i, j, d / 10.0, 1000.0)
    system.addForce(bond_force)

    # ---- soft-core LJ + Coulomb via a single CustomNonbondedForce ----
    # OpenMM's CustomNonbondedForce supports only per-particle
    # parameters (no per-pair), so we encode everything we need as
    # per-particle parameters and let the energy expression compute
    # pair properties from them. Per-particle parameters:
    #   sigmai, epsiloni, chargei — particle i's LJ / Coulomb params
    #   is_core_i, is_dis_i, is_app_i — which alchemical class i is in
    # The pair weight is computed from the kind labels and λ:
    #   core-core: 0 (no contribution — we still evaluate, weight=0)
    #   core ↔ disappearing: (1-λ)
    #   core ↔ appearing: λ
    #   disappearing ↔ disappearing: (1-λ)
    #   appearing ↔ appearing: λ
    #   disappearing ↔ appearing: 0 (no physical interaction — skip)
    #
    # Soft-core LJ (Steinbrecher-Case 2012 form):
    #   U_LJ(r) = 4 ε_ij λ × [σ_ij^6 / (α(1-λ)² + r^6)²
    #                          - σ_ij^6 / (α(1-λ)² + r^6)]
    # Coulomb is just q_i q_j λ_pair / r (linear in pair weight).
    # NOTE: the order matters. OpenMM parses each expression as it
    # is declared, so all intermediate (computed-value) variables
    # referenced by the energy expression need to be added BEFORE
    # the energy expression itself.
    nb = openmm.CustomNonbondedForce("")
    nb.addGlobalParameter("lam", lambda_value)
    nb.addGlobalParameter("alpha", SOFT_CORE_ALPHA_NM2)
    # r2 needs to be a global parameter (computed values are per-particle
    # in OpenMM CustomNonbondedForce). We update it on every step from
    # the integrator's distance, but for nonbonded forces the
    # distance `r` is already available — so we just inline r*r*r in
    # the energy expression. No need for `r2` here.
    nb.addPerParticleParameter("sigma")
    nb.addPerParticleParameter("eps")
    nb.addPerParticleParameter("charge")
    nb.addPerParticleParameter("is_core")
    nb.addPerParticleParameter("is_dis")
    nb.addPerParticleParameter("is_app")
    # Inlined single-line energy expression. Per-pair weight:
    #   core-core:                   0    (no contribution)
    #   core-dis / dis-core:         (1-lam)
    #   core-app / app-core:         lam
    #   dis-dis:                     (1-lam)
    #   app-app:                     lam
    #   dis-app / app-dis:           0    (no physical interaction)
    # Soft-core LJ: Steinbrecher-Case 2012 form, α(1-λ)² in the
    # denominator prevents the r=0 singularity at λ=0.
    # NB: OpenMM CustomNonbondedForce doesn't accept ";"-separated
    # sub-expression assignments in the energy function — everything
    # must be one expression. Per-particle parameters use the
    # ``name1`` / ``name2`` suffix convention in the energy function.
    w_dis = "(is_dis1*(1-is_app2) + (1-is_app1)*is_dis2)"
    w_app = "(is_app1*(1-is_dis2) + (1-is_dis1)*is_app2)"
    skip_dis_app = "(is_dis1*is_app2 + is_app1*is_dis2)"
    weight = f"((1-lam)*{w_dis} + lam*{w_app})"
    sigma_ij = "(0.5*(sigma1 + sigma2))"
    soft_lj = (
        f"(1-{skip_dis_app}) * {weight} * 4 * sqrt(eps1*eps2) * "
        f"({sigma_ij}^6/(alpha*(1-lam)^2 + r*r*r)^2 -"
        f" {sigma_ij}^6/(alpha*(1-lam)^2 + r*r*r))"
    )
    soft_q = (
        f"(1-{skip_dis_app}) * {weight} * charge1*charge2 / r"
    )
    nb.setEnergyFunction(f"{soft_lj} + {soft_q}")

    def _lp(src):
        elem = (src.element or "C").upper()
        if elem == "H":  return 0.12, 0.10, 0.0
        if elem == "C":  return 0.34, 0.36, 0.0
        if elem == "N":  return 0.33, 0.71, 0.0
        if elem == "O":  return 0.31, 0.88, 0.0
        if elem == "S":  return 0.36, 1.86, 0.0
        return 0.34, 0.36, 0.0

    for label, src in zip(hybrid_labels, [h[2] for h in hybrid_atoms]):
        sigma, eps, charge = _lp(src)
        is_core = 1.0 if label == "core" else 0.0
        is_dis = 1.0 if label == "disappearing" else 0.0
        is_app = 1.0 if label == "appearing" else 0.0
        nb.addParticle((sigma, eps, charge, is_core, is_dis, is_app))
        # Parameter order MUST match addPerParticleParameter declarations above:
        # (sigma, eps, charge, is_core, is_dis, is_app).

    nb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    system.addForce(nb)
    return system, topology, positions


def _set_lambda_in_system(system, lambda_value: float):
    """Set the ``lam`` global parameter on every CustomNonbondedForce in
    ``system`` that has one.  Called by :func:`run_alchemical_window`
    before each step."""
    import openmm
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, openmm.CustomNonbondedForce):
            if f.getNumGlobalParameters() > 0:
                # We always name our lambda parameter "lam".
                try:
                    f.setGlobalParameterByName("lam", lambda_value)
                except Exception:
                    pass


def run_alchemical_window(mapping: AlchemicalMapping,
                          lambda_value: float = 0.0,
                          n_steps: int = 100,
                          temperature_kelvin: float = 300.0,
                          timestep_fs: float = 1.0):
    """Sample the reduced potential at one alchemical lambda value.

    Returns a list of energies (kJ/mol) — one per frame including the
    pre-MD state.
    """
    openmm, unit = _require_openmm()
    if n_steps <= 0:
        return []
    system, topology, positions = build_alchemical_system(
        mapping, lambda_value=lambda_value,
        temperature_kelvin=temperature_kelvin,
    )
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    if hasattr(openmm, "LocalEnergyMinimizer"):
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=20)

    energies: List[float] = []
    state = context.getState(getEnergy=True)
    energies.append(
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    for _ in range(n_steps):
        integrator.step(1)
        state = context.getState(getEnergy=True)
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )
    return energies


def run_alchemical_fep(mapping: AlchemicalMapping,
                        lambdas: Optional[Sequence[float]] = None,
                        n_steps: int = 100,
                        temperature_kelvin: float = 300.0):
    """Run a Phase 2 alchemical FEP and return :class:`FEPResult`.

    The result records the energy mean / SEM at each lambda value,
    and the trapezoidal ``ΔG`` (Phase 3 will swap this for BAR/MBAR).
    """
    if lambdas is None:
        lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)
    result = FEPResult()
    means: List[float] = []
    sems: List[float] = []
    for lam in lambdas:
        energies = run_alchemical_window(
            mapping, lambda_value=lam, n_steps=n_steps,
            temperature_kelvin=temperature_kelvin,
        )
        if not energies:
            means.append(float("nan"))
            sems.append(float("nan"))
        else:
            arr = np.asarray(energies, dtype=float)
            means.append(float(arr.mean()))
            sems.append(float(arr.std(ddof=1) / math.sqrt(len(arr))))
    result.lambdas = list(lambdas)
    result.mean_energies = means
    result.sem_energies = sems

    finite = [(l, m) for l, m in zip(lambdas, means)
              if not math.isnan(m)]
    if len(finite) >= 2:
        ls, ms = zip(*finite)
        result.delta_g = float(np.trapezoid(ms, x=ls))
        sem_finite = [
            s for s in result.sem_energies if not math.isnan(s)
        ]
        if len(sem_finite) == len(finite):
            result.delta_g_sem = float(
                math.sqrt(sum(s * s for s in sem_finite))
            )
    return result


__all__ = [
    "AlchemicalState",
    "LambdaWindow",
    "FEPResult",
    "AlchemicalMapping",
    "build_openmm_system",
    "run_window",
    "run_fep",
    "build_alchemical_system",
    "run_alchemical_window",
    "run_alchemical_fep",
    "build_amber_solvated_system",
    "compute_bar",
    "run_amber_fep",
    "build_amber_mutation_system",
    "run_amber_mutation_fep",
]


# ============================================================================
# Phase 3: real force field (AMBER ff99SB-ILDN) + TIP3P solvent + BAR/MBAR.
# Builds a real OpenMM system with the AMBER force field and TIP3P explicit
# solvent, then samples energies at multiple lambda values where lambda
# scales the NonbondedForce's charge and epsilon parameters. This isn't
# yet a true alchemical transformation (no atoms appearing/disappearing
# with AMBER params) — it's a Phase-3-equivalent decoupling sweep on a
# real solvated system. Adding real alchemical transformations on top
# of AMBER (e.g. Lys→Ala mutation) requires hybrid residue templates
# and is the natural next step.
# ============================================================================


def build_amber_solvated_system(
    pdb_file: str,
    force_field_name: str = "amber99sbildn.xml",
    water_model: str = "tip3p.xml",
    padding_nm: float = 1.0,
    ionic_strength_M: float = 0.15,
    temperature_K: float = 300.0,
    timestep_fs: float = 2.0,
    cutoff_nm: float = 1.0,
):
    """Load a PDB, solvate with TIP3P + ions, build an AMBER system.

    Returns the OpenMM System, Topology, Positions, and an integrator.
    Use ``run_amber_fep`` to actually sample at multiple lambdas.

    Parameters
    ----------
    pdb_file : str
        Path to the input PDB.  Protein-only PDBs are fine — the
        function adds hydrogens, TIP3P waters and Na+/Cl- ions to
        neutralise.
    force_field_name : str
        OpenMM force-field XML name.  ``"amber99sbildn.xml"`` is the
        AMBER ff99SB-ILDN protein force field with TIP3P water.  Also
        accepts CHARMM (``"charmm36.xml"``) etc.
    water_model : str
        OpenMM water XML name.  ``"tip3p.xml"`` is the default.
    padding_nm : float
        Distance (nm) of TIP3P padding around the protein in each
        dimension.  Default 1.0 nm keeps the box small enough for a
        Phase-3 verification (~5000 waters on ubiquitin).
    ionic_strength_M : float
        NaCl concentration (M) used to neutralise the system.  Default
        0.15 M (physiological).
    """
    openmm, unit = _require_openmm()
    import openmm.app as app

    # Load the input structure.
    pdb = app.PDBFile(pdb_file)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Force field with protein + water.  ``amber99sbildn.xml``
    # pulls in all standard AMBER residue templates; ``tip3p.xml``
    # gives us TIP3P waters and Na+/Cl- ions for neutralisation.
    ff = app.ForceField(force_field_name, water_model)

    # Add hydrogens at pH 7 (default) so each residue has the right
    # number of H atoms for the AMBER templates.  OpenMM's Modeller
    # automatically assigns histidine protonation states.
    modeller.addHydrogens(ff)

    # Add TIP3P solvent + ions to the box.  The box is built so
    # there's at least ``padding_nm`` nm of water on each side.
    modeller.addSolvent(
        ff,
        model="tip3p",
        padding=padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_M * unit.molar,
        neutralize=True,
    )

    # Build the AMBER System with PME (periodic) for electrostatics
    # and HBonds-constrained bonds.
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5e-4,
    )

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )

    return (
        system,
        modeller.topology,
        modeller.positions,
        integrator,
    )


def compute_bar(
    u_a_at_a: Sequence[float],
    u_b_at_a: Sequence[float],
    u_a_at_b: Sequence[float],
    u_b_at_b: Sequence[float],
    kT_kjmol: float = 2.479,
    tol: float = 1e-6,
    max_iter: int = 200,
    n_bootstrap: int = 200,
) -> Tuple[float, float]:
    """Bennett Acceptance Ratio between two neighbouring states A and B.

    Given per-frame reduced potentials U(state A) and U(state B)
    evaluated at samples drawn from *both* state A and state B,
    return ``(delta_g, sigma)`` in kJ/mol where ::

        ΔG_AB = kT × log ⟨exp(-(U_B - U_A) / kT)⟩_A
                    /     ⟨exp(-(U_A - U_B) / kT)⟩_B

    This is the standard Bennett (1976) iterative solver: find ``C``
    such that

        sum_A 1 / (1 + exp((C + u_i)/kT)) = sum_B 1 / (1 + exp(-(C + u_j)/kT))

    where ``u_i = U_B(state_i) - U_A(state_i)`` for samples drawn
    from state A and ``u_j = U_A(state_j) - U_B(state_j)`` for
    samples drawn from state B.

    Parameters
    ----------
    u_a_at_a : sequence of float
        U_A evaluated at samples drawn from state A (kJ/mol).
    u_b_at_a : sequence of float
        U_B evaluated at the *same* samples from state A (kJ/mol).
    u_a_at_b : sequence of float
        U_A evaluated at samples drawn from state B (kJ/mol).
    u_b_at_b : sequence of float
        U_B evaluated at the *same* samples from state B (kJ/mol).
    kT_kjmol : float
        ``k_B T`` in kJ/mol at 300 K (default 2.479).
    tol, max_iter : float, int
        Convergence tolerance and iteration cap for Bennett's solver.
    n_bootstrap : int
        Number of bootstrap samples for uncertainty estimation
        (default 200).

    Notes
    -----
    With only single-state samples (no mixed-state re-evaluation),
    BAR reduces to the Zwanzig exponential-average estimator which
    is biased for windows that don't overlap.  For an 11-window
    sweep with Δλ = 0.1 and the soft-core parameters used in
    Phase 4, neighbouring windows have substantial overlap, so the
    histogram-overlap estimator is a reasonable approximation
    when mixed-state energies aren't available — see
    ``analyze_fep_trapezoidal``.
    """
    import math
    arr_a = np.asarray(u_a_at_a, dtype=float)
    arr_b_at_a = np.asarray(u_b_at_a, dtype=float)
    arr_a_at_b = np.asarray(u_a_at_b, dtype=float)
    arr_b = np.asarray(u_b_at_b, dtype=float)
    n_a = arr_a.size
    n_b = arr_b.size
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan")

    # Δu_A = U_B(state_i) - U_A(state_i) for samples from A.
    delta_u_a = (arr_b_at_a - arr_a) / kT_kjmol
    # Δu_B = U_A(state_j) - U_B(state_j) for samples from B.
    delta_u_b = (arr_a_at_b - arr_b) / kT_kjmol

    def _bar_solve(delta_a: np.ndarray, delta_b: np.ndarray,
                   c_init: float) -> float:
        """Iterate Bennett's overlap equation for given Δu arrays.

        Bennett 1976 (Shirts 2008 best-practices form):

            ⟨1/(1+exp(Δu_A − c))⟩_A = ⟨1/(1+exp(Δu_B + c))⟩_B

        where c = β·ΔF (i.e. C in Bennett's notation) and ΔF =
        F_B − F_A.  At convergence, ΔF = c · kT.

        Strategy: bracket the root then bisect.  The Bennett
        objective can have very flat regions far from the root
        (sums saturate to N_A or 0 once |Δu − c| ≫ 0), so a
        straight Newton solver diverges — every Newton step
        lands outside the useful region and the next derivative
        is essentially zero.
        """
        def _f(c: float) -> float:
            # Clip to avoid overflow in exp; fF at large |x| is
            # exactly 0 or 1 by the formula.
            arg_a = np.clip(delta_a - c, -500, 500)
            arg_b = np.clip(delta_b + c, -500, 500)
            return (
                float(np.sum(1.0 / (1.0 + np.exp(arg_a)))) -
                float(np.sum(1.0 / (1.0 + np.exp(arg_b))))
            )

        # Step 1: bracket.  Search outward from c_init until the
        # sign of f changes.  f(-∞) = -N_B (sum_B fF of large
        # positive → 0; sum_A fF of large negative → N_A; so
        # f(-∞) = N_A - 0 = N_A > 0).  f(+∞) = -N_B < 0.
        # Wait — recheck: f(c) = Σ_A fF(Δu_A - c) - Σ_B fF(Δu_B + c)
        #   As c → -∞: Δu_A - c → +∞, fF → 0; Δu_B + c → -∞, fF → 1.
        #     f → 0 - N_B = -N_B < 0.
        #   As c → +∞: Δu_A - c → -∞, fF → 1; Δu_B + c → +∞, fF → 0.
        #     f → N_A - 0 = N_A > 0.
        # So f(-∞) < 0 and f(+∞) > 0 — there's a unique root.
        c_low = c_init
        c_high = c_init
        step = 1.0
        for _ in range(60):
            if _f(c_low) < 0:
                break
            c_low -= step
            step *= 2
        step = 1.0
        for _ in range(60):
            if _f(c_high) > 0:
                break
            c_high += step
            step *= 2

        # Step 2: bisection.
        for _ in range(max_iter):
            c_mid = 0.5 * (c_low + c_high)
            f_mid = _f(c_mid)
            if abs(f_mid) < tol * 1e3 or (c_high - c_low) < tol:
                return c_mid
            if f_mid < 0:
                c_low = c_mid
            else:
                c_high = c_mid
        return 0.5 * (c_low + c_high)

    # Initial guess: Zwanzig exponential average (one-sided
    # estimator at state A).  ΔG = -kT · log⟨exp(-β(U_B - U_A))⟩_A
    # so c_init = ΔG/kT = -log⟨exp(-Δu_a)⟩_A.
    c_init = -float(np.log(np.mean(np.exp(-delta_u_a))))
    c = _bar_solve(delta_u_a, delta_u_b, c_init)
    delta_g = kT_kjmol * c

    # Bootstrap uncertainty.  Resample A and B independently and
    # solve BAR each time.
    rng = np.random.default_rng(0)
    boot = np.empty(max(n_bootstrap, 2), dtype=float)
    for bi in range(n_bootstrap):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        d_a_b = (arr_b_at_a[idx_a] - arr_a[idx_a]) / kT_kjmol
        d_b_b = (arr_a_at_b[idx_b] - arr_b[idx_b]) / kT_kjmol
        z = -float(np.log(np.mean(np.exp(-d_a_b))))
        boot[bi] = kT_kjmol * _bar_solve(d_a_b, d_b_b, z)
    sigma = float(boot.std(ddof=1)) if n_bootstrap > 1 else 0.0
    return delta_g, sigma


def analyze_fep_bar(
    energies_by_lambda: dict,
    neighbor_energies: dict,
    lambdas: Optional[Sequence[float]] = None,
    kT_kjmol: float = 2.479,
) -> Tuple[float, float, List[Tuple[int, float, float]]]:
    """Total ΔG via Bennett Acceptance Ratio between adjacent λ windows.

    For each adjacent pair (λ_i, λ_{i+1}), the function needs:

      From window-i frames:
        - u_i_at_i  = energies_by_lambda[λ_i][k]
        - u_{i+1}_at_i = neighbor_energies[λ_i][k].next (the right
          element of the tuple)

      From window-(i+1) frames:
        - u_{i+1}_at_{i+1} = energies_by_lambda[λ_{i+1}][k]
        - u_i_at_{i+1}    = neighbor_energies[λ_{i+1}][k].prev

    Each pair's BAR ΔG is added to the running total.

    Returns ``(total_delta_g_kJmol, total_sigma_kJmol,
    [(window_idx, dg, sigma), ...])``.
    """
    if lambdas is None:
        lambdas = sorted(energies_by_lambda.keys())
    else:
        lambdas = tuple(float(l) for l in lambdas)

    if not neighbor_energies:
        raise ValueError(
            "neighbor_energies is empty — pass sample_neighbor_lambdas=True "
            "to run_amber_mutation_fep to populate it."
        )

    per_window: List[Tuple[int, float, float]] = []
    total_dg = 0.0
    # Combine per-pair sigmas in quadrature (uncorrelated pairs
    # approximation — accurate enough for an 11-window sweep).
    total_var = 0.0
    for i in range(len(lambdas) - 1):
        lam_i = lambdas[i]
        lam_j = lambdas[i + 1]
        e_i = energies_by_lambda.get(lam_i, [])
        e_j = energies_by_lambda.get(lam_j, [])
        nbr_i = neighbor_energies.get(lam_i, [])
        nbr_j = neighbor_energies.get(lam_j, [])
        if not e_i or not e_j or not nbr_i or not nbr_j:
            per_window.append((i, float("nan"), float("nan")))
            continue
        # u_i at samples from i, u_j at samples from i
        u_i_at_i = np.asarray(e_i)
        u_j_at_i = np.asarray([t[1] for t in nbr_i])  # "next" entry
        # u_i at samples from j, u_j at samples from j
        u_i_at_j = np.asarray([t[0] for t in nbr_j])  # "prev" entry
        u_j_at_j = np.asarray(e_j)
        dg, sigma = compute_bar(
            u_i_at_i, u_j_at_i, u_i_at_j, u_j_at_j,
            kT_kjmol=kT_kjmol,
        )
        per_window.append((i, dg, sigma))
        total_dg += dg
        total_var += sigma * sigma
    return total_dg, float(np.sqrt(total_var)), per_window


def analyze_fep_trapezoidal(
    energies_by_lambda: dict,
    lambdas: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Total ΔG via trapezoidal integration of ⟨U⟩_λ vs λ.

    Returns ``(delta_g_kJmol, sigma_kJmol)`` where ``sigma`` is the
    standard error of the per-window mean propagated through the
    trapezoidal rule (assuming independent windows).
    """
    if lambdas is None:
        lambdas = sorted(energies_by_lambda.keys())
    else:
        lambdas = tuple(float(l) for l in lambdas)
    means = np.asarray(
        [np.mean(energies_by_lambda[l]) for l in lambdas]
    )
    sems = np.asarray(
        [
            np.std(energies_by_lambda[l], ddof=1) /
            max(np.sqrt(len(energies_by_lambda[l])), 1.0)
            for l in lambdas
        ]
    )
    # Trapezoidal rule over the mean energies.  Use
    # ``np.trapezoid`` (NumPy ≥ 2.0) if available, else fall
    # back to ``np.trapz``.
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    delta_g = float(trap(means, lambdas))
    # Per-window SEM → propagated variance = sum Δλ² × SEM²
    # (trapezoidal variance estimate).
    if len(lambdas) >= 2:
        dlam = np.diff(lambdas)
        var_terms = np.zeros(len(lambdas))
        var_terms[0] = (dlam[0] / 2.0) ** 2 * sems[0] ** 2
        var_terms[-1] = (dlam[-1] / 2.0) ** 2 * sems[-1] ** 2
        if len(lambdas) > 2:
            var_terms[1:-1] = (
                (dlam[:-1] / 2.0 + dlam[1:] / 2.0) ** 2 * sems[1:-1] ** 2
            )
        sigma = float(np.sqrt(var_terms.sum()))
    else:
        sigma = float("nan")
    return delta_g, sigma


def run_amber_fep(
    pdb_file: str,
    lambdas: Optional[Sequence[float]] = None,
    n_steps: int = 100,
    temperature_K: float = 300.0,
    padding_nm: float = 1.0,
):
    """Run FEP on a real AMBER / TIP3P-solvated system.

    λ rescales every nonbonded particle's ``charge`` and ``epsilon``
    by ``λ`` (the AMBER ``sigma`` is unchanged).  Bonds and angles
    are unaffected.  At ``λ = 1`` the system has full nonbonded
    interactions; at ``λ = 0`` all nonbonded forces are turned off —
    a "decoupling" sweep that's a Phase-3 stand-in for a real
    alchemical transformation.

    Returns a dict ``{lam: [energies...]}``.  Apply
    :func:`compute_bar` (or trapezoidal) to get ΔG.
    """
    openmm, unit = _require_openmm()
    if lambdas is None:
        lambdas = (0.0, 0.25, 0.5, 0.75, 1.0)

    system, topology, positions, integrator = build_amber_solvated_system(
        pdb_file,
        padding_nm=padding_nm,
        temperature_K=temperature_K,
    )

    # Capture the NonbondedForce (with all original AMBER params)
    # so we can re-scale per lambda.
    nb_idx = -1
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), openmm.NonbondedForce):
            nb_idx = i
            break
    if nb_idx == -1:
        raise RuntimeError("No NonbondedForce in system")
    nb = system.getForce(nb_idx)
    orig_q = []
    orig_sig = []
    orig_eps = []
    for j in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(j)
        orig_q.append(q)
        orig_sig.append(sig)
        orig_eps.append(eps)

    platform = openmm.Platform.getPlatformByName("CPU")
    energies_by_lambda: dict = {}

    for lam in lambdas:
        # Re-scale NonbondedForce per-particle params.  We don't
        # touch sigma (only charge and epsilon) so the LJ "shape"
        # is preserved across the sweep — that keeps the soft-core
        # idea implicit even at lambda=0 (epsilon is zero, so the
        # well goes away without the sigma going to zero).
        for j in range(nb.getNumParticles()):
            nb.setParticleParameters(
                j,
                orig_q[j] * lam,
                orig_sig[j],
                orig_eps[j] * lam,
            )
        nb.updateParametersInContext.__wrapped__ if False else None

        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        if hasattr(openmm, "LocalEnergyMinimizer"):
            openmm.LocalEnergyMinimizer.minimize(context, maxIterations=20)

        energies: List[float] = []
        state = context.getState(getEnergy=True)
        energies.append(
            state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
        )
        for _ in range(max(n_steps, 0)):
            integrator.step(1)
            state = context.getState(getEnergy=True)
            energies.append(
                state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
            )
        del context
        energies_by_lambda[float(lam)] = energies

    return energies_by_lambda


# ============================================================================
# Phase 4: real alchemical transformation with AMBER HybridResidue.
# Take a sidechain-truncation mutation (e.g., Lys→Ala: keep backbone
# + CB, drop everything beyond). Build an AMBER system with the source
# residue's atoms present, mark the sidechain atoms beyond CB as
# "disappearing", and replace the AMBER NonbondedForce with a soft-core
# CustomNonbondedForce so the disappearing atoms' nonbonded terms scale
# from full-strength at lambda=1 to zero at lambda=0. The bonded terms
# (bonds/angles/dihedrals touching the disappearing atoms) are kept as
# AMBER normally — at lambda=0 the sidechain atoms are bonded but
# non-interacting, which is the canonical "dummy atom" treatment for
# a sidechain truncation.
# ============================================================================


# AMBER backbone + Cβ atom names that are always retained in a
# sidechain-truncation mutation.  Any atom NOT in this set on the
# mutated residue is marked "disappearing" (lambda-controlled).
_BACKBONE_PLUS_CB = frozenset({
    "N", "H", "CA", "HA", "C", "O", "OXT",
    "CB", "HB", "HB1", "HB2", "HB3",
})


def _find_disappearing_atoms(topology, residue_index: int,
                             residue_name: str) -> List[int]:
    """Return the particle indices of the atoms to disappear when
    truncating ``residue_name`` at ``residue_index``.

    Anything not in the backbone + Cβ set is treated as sidechain.
    """
    indices: List[int] = []
    for atom in topology.atoms():
        if (atom.residue.index == residue_index
                and atom.residue.name == residue_name):
            if atom.name not in _BACKBONE_PLUS_CB:
                indices.append(atom.index)
    return indices


def _replace_amber_nb_with_alchemical(
    system,
    disappearing_set: set,
    lambda_value: float,
    alpha_nm2: float = 1.5,
):
    """Swap the AMBER NonbondedForce for a soft-core CustomNonbondedForce.

    The replacement force is a single inlined expression that does
    standard AMBER LJ + Coulomb for non-disappearing pairs and a
    Steinbrecher-Case 2012 soft-core LJ + linear-λ Coulomb for
    pairs involving at least one disappearing atom.  Per-particle
    parameters carry the AMBER (charge, sigma, ε) plus a boolean
    ``is_dis`` flag; ``is_dis = 1`` means the atom is the truncated
    sidechain.
    """
    openmm, unit = _require_openmm()

    # Find the existing NonbondedForce.
    nb_idx = -1
    old_nb = None
    for i, force in enumerate(system.getForces()):
        if isinstance(force, openmm.NonbondedForce):
            nb_idx = i
            old_nb = force
            break
    if old_nb is None:
        raise RuntimeError("No NonbondedForce in the AMBER system")

    n_particles = old_nb.getNumParticles()
    orig = []
    for j in range(n_particles):
        q, sig, eps = old_nb.getParticleParameters(j)
        orig.append((q, sig, eps))

    # Capture AMBER's exception list (1-2, 1-3 bonded pairs and
    # 1-4 scaled pairs).  CustomNonbondedForce evaluates ALL pairs by
    # default — we have to copy the exclusion list or bonded atoms
    # will see ~1.5 Å LJ energies and blow up the system.
    exceptions = []
    for k in range(old_nb.getNumExceptions()):
        i, j = old_nb.getExceptionParameters(k)[:2]
        exceptions.append((i, j))

    system.removeForce(nb_idx)

    # Build the soft-core CustomNonbondedForce.
    nb = openmm.CustomNonbondedForce("")
    nb.addGlobalParameter("lam", lambda_value)
    nb.addGlobalParameter("alpha", alpha_nm2)
    nb.addPerParticleParameter("charge")
    nb.addPerParticleParameter("sigma")
    nb.addPerParticleParameter("eps")
    nb.addPerParticleParameter("is_dis")

    for j in range(n_particles):
        is_dis = 1.0 if j in disappearing_set else 0.0
        nb.addParticle((orig[j][0], orig[j][1], orig[j][2], is_dis))

    # Single inlined expression — CustomNonbondedForce doesn't
    # accept "; subexpression = value" sub-statements (it expects
    # the expression to be one big formula), and addComputedValue
    # can't reference per-particle params with _1/_2 suffixes.  So
    # everything must be one expression.
    # Per-pair classification via per-particle is_dis flags:
    #   weight_dis = is_dis1 + is_dis2 - is_dis1*is_dis2
    #   weight_core = (1-is_dis1)*(1-is_dis2)
    # Non-disappearing pairs use standard AMBER LJ + Coulomb.
    # Disappearing pairs use soft-core LJ + Coulomb scaled by λ.
    nb.setEnergyFunction(
        # 1. AMBER LJ for non-disappearing pairs.
        "((1-is_dis1)*(1-is_dis2)) * 4 * sqrt(eps1*eps2) * "
        "(((0.5*(sigma1+sigma2))^12)/(r*r)^6 - "
        "((0.5*(sigma1+sigma2))^6)/(r*r)^3)"
        # 2. Coulomb for non-disappearing pairs.
        " + ((1-is_dis1)*(1-is_dis2)) * charge1*charge2 / r"
        # 3. Soft-core LJ for disappearing pairs (Beutler 1994 form — at
        # λ=1 it reduces exactly to standard LJ; Steinbrecher-Case 2012
        # uses σ^6 in the repulsion term which leaves an endpoint artefact
        # that explodes to ~1e9 kJ/mol when λ=1 and atoms are in close
        # contact after several λ windows of soft-core relaxation).
        " + ((is_dis1 + is_dis2 - is_dis1*is_dis2)) * lam * 4 * sqrt(eps1*eps2) * "
        "(((0.5*(sigma1+sigma2))^12) / (alpha*(1-lam)^2 + (r*r)*(r*r)*(r*r))^2 - "
        "((0.5*(sigma1+sigma2))^6)  / (alpha*(1-lam)^2 + (r*r)*(r*r)*(r*r)))"
        # 4. Coulomb for disappearing pairs (linear λ scaling).
        " + ((is_dis1 + is_dis2 - is_dis1*is_dis2)) * lam * charge1*charge2 / r"
    )
    # Copy AMBER's exception list (1-2, 1-3 bonded pairs are
    # excluded; 1-4 pairs are scaled — but our energy expression
    # doesn't model 1-4 scaling, so we just exclude all of them).
    for (i, j) in exceptions:
        if i < j:
            nb.addExclusion(i, j)
        else:
            nb.addExclusion(j, i)

    nb.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    system.addForce(nb)


def build_amber_mutation_system(
    pdb_file: str,
    mutation_residue: int,
    from_resname: str = "LYS",
    padding_nm: float = 1.0,
    ionic_strength_M: float = 0.15,
    temperature_K: float = 300.0,
    timestep_fs: float = 2.0,
    cutoff_nm: float = 1.0,
    lambda_value: float = 1.0,
    alpha_nm2: float = 1.5,
    restraint_k_kJmol_nm2: float = 1000.0,
):
    """Build a solvated AMBER system with the source residue's
    sidechain atoms marked alchemical.

    ``mutation_residue`` is the residue index (0-based in OpenMM's
    topology) of the source residue (e.g. LYS at index 8 for the
    typical Lys→Ala example on ubiquitin).  The function identifies
    the sidechain atoms beyond the Cβ backbone + Cβ set as
    "disappearing" — they'll be λ-decoupled during the FEP sweep.

    Disappearing atoms are weakly restrained to their initial
    positions via a per-particle ``CustomExternalForce`` with
    ``k = restraint_k_kJmol_nm2`` (default 1000 kJ/mol/nm²).  This
    is the standard FEP practice (e.g. Jorgensen's LIE-style setup,
    plus all production alchemical FEP in pmx / alchemlyb): without
    restraints the soft-core LJ is too weak to keep disappearing
    atoms away from close contacts with surrounding waters / protein
    at intermediate λ values, and Langevin noise eventually pushes
    one of them onto a water H at ~0.07 nm.  At the next λ window
    the standard LJ kicks in and the integrator explodes to NaN.

    Returns the system, topology, positions, integrator, and the
    list of disappearing atom indices (useful for analysis).
    """
    openmm, unit = _require_openmm()
    system, topology, positions, integrator = build_amber_solvated_system(
        pdb_file,
        padding_nm=padding_nm,
        ionic_strength_M=ionic_strength_M,
        temperature_K=temperature_K,
        timestep_fs=timestep_fs,
        cutoff_nm=cutoff_nm,
    )
    disappearing = _find_disappearing_atoms(
        topology, mutation_residue, from_resname,
    )
    if not disappearing:
        raise ValueError(
            f"No atoms to disappear on residue {from_resname}#{mutation_residue}. "
            "Check residue_index (0-based) and residue name."
        )
    _replace_amber_nb_with_alchemical(
        system, set(disappearing), lambda_value=lambda_value,
        alpha_nm2=alpha_nm2,
    )

    # Position restraints on disappearing atoms — see docstring.
    if restraint_k_kJmol_nm2 > 0:
        restraint = openmm.CustomExternalForce(
            "k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        restraint.addGlobalParameter("k", restraint_k_kJmol_nm2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        for idx in disappearing:
            pos = positions[idx].value_in_unit(unit.nanometer)
            restraint.addParticle(idx, (pos[0], pos[1], pos[2]))
        restraint.setForceGroup(1)
        system.addForce(restraint)

    return system, topology, positions, integrator, disappearing


def run_amber_mutation_fep(
    pdb_file: str,
    mutation_residue: int,
    from_resname: str = "LYS",
    lambdas: Optional[Sequence[float]] = None,
    n_equilibration_steps: int = 200,
    n_production_steps: int = 1000,
    temperature_K: float = 300.0,
    padding_nm: float = 0.8,
    minimize_iterations: int = 10,
    restraint_k_kJmol_nm2: Optional[float] = None,
    sample_every: int = 1,
    sample_neighbor_lambdas: bool = False,
):
    """Production-grade FEP for a sidechain-truncation mutation.

    Returns a 3-tuple ``(energies_by_lambda, neighbor_energies,
    disappearing_indices)`` where:

    ``energies_by_lambda[λ]`` is a list of *production* potential
    energies (kJ/mol) sampled at this λ after the equilibration
    phase.  Equilibration samples are discarded — this is what
    makes the returned trajectories a valid ensemble for free-
    energy estimation.

    ``neighbor_energies`` is a dict of the same shape as
    ``energies_by_lambda`` containing re-evaluations of the
    *same frames* at the two adjacent λ values (when they
    exist).  This is what makes Bennett Acceptance Ratio (BAR)
    possible — for two neighbouring states we need the
    difference U(λ_{i+1}) − U(λ_i) evaluated at frames drawn
    from state i (and vice versa).  Empty dict unless
    ``sample_neighbor_lambdas=True`` (3× the energy evaluations
    per saved frame, but no extra memory).

    Parameters
    ----------
    lambdas : sequence of float, optional
        Alchemical coupling schedule.  Default: 11 evenly-spaced
        windows from 0.0 to 1.0 (Δλ = 0.1).  Production FEP
        typically uses 11–21 windows; for non-overlapping
        neighbouring windows the soft-core is too weak to give
        meaningful histograms.
    n_equilibration_steps : int
        Number of MD steps to discard per window after the
        minimisation.  LangevinMiddleIntegrator needs ~50–200
        steps to forget the initial positions and equilibrate
        velocities; default 200.
    n_production_steps : int
        Number of MD steps to *keep* per window.  Default 1000;
        for published-quality ΔG expect 10⁴–10⁵.
    sample_every : int
        Stride for production sampling — energies are saved every
        Nth step to decorrelate the samples (autocorrelation time
        for protein-sidechain ΔG is ~5–10 steps at 2 fs).
    restraint_k_kJmol_nm2 : float or None, optional
        Force constant for the position restraints on disappearing
        atoms.  If ``None`` (default), uses the value set during
        ``build_amber_mutation_system`` (1000 kJ/mol/nm²).  Pass
        ``0.0`` to disable restraints entirely — only useful when
        the soft-core parameters are tuned to keep disappearing
        atoms out of close contacts without restraints.
    sample_neighbor_lambdas : bool
        When True, also re-evaluate every saved frame at the
        two adjacent λ values.  Pass to ``analyze_fep_bar`` for
        proper BAR / MBAR.  Default False to keep verification
        runs fast.
    """
    openmm, unit = _require_openmm()
    if lambdas is None:
        lambdas = tuple(i / 10.0 for i in range(11))  # 11 windows
    lambdas = tuple(float(l) for l in lambdas)
    n_windows = len(lambdas)

    if restraint_k_kJmol_nm2 is None:
        system, topology, positions, integrator, disappearing = (
            build_amber_mutation_system(
                pdb_file,
                mutation_residue=mutation_residue,
                from_resname=from_resname,
                padding_nm=padding_nm,
                temperature_K=temperature_K,
                lambda_value=1.0,
            )
        )
    else:
        system, topology, positions, integrator, disappearing = (
            build_amber_mutation_system(
                pdb_file,
                mutation_residue=mutation_residue,
                from_resname=from_resname,
                padding_nm=padding_nm,
                temperature_K=temperature_K,
                lambda_value=1.0,
                restraint_k_kJmol_nm2=restraint_k_kJmol_nm2,
            )
        )

    nb = None
    for f in system.getForces():
        if isinstance(f, openmm.CustomNonbondedForce):
            nb = f
            break
    if nb is None:
        raise RuntimeError("Soft-core CustomNonbondedForce not found")

    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)

    energies_by_lambda: dict = {lam: [] for lam in lambdas}
    neighbor_energies: dict = {lam: [] for lam in lambdas}

    # Defensive position reset.
    try:
        import numpy as np
        pos_check = context.getState(getPositions=True)
        if np.any(np.isnan(np.asarray(
            [[p.value_in_unit(unit.nanometer) for p in pr]
             for pr in pos_check.getPositions()]
        ))):
            context.setPositions(positions)
    except Exception:
        context.setPositions(positions)

    def _energy_at_current_lam() -> float:
        """Re-evaluate total energy at the *current* λ value
        (i.e. whatever was last set via setParameter)."""
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    for i, lam in enumerate(lambdas):
        # Set this window's λ and make sure the soft-core Force
        # sees it.
        context.setParameter("lam", float(lam))
        nb.updateParametersInContext(context)
        context.setPositions(positions)

        # Minimise at each λ before sampling — even with reference
        # positions, the λ=1 standard LJ needs a few LBFGS
        # iterations to relax clashes inherited from the
        # unminimised solvation structure.
        if hasattr(openmm, "LocalEnergyMinimizer"):
            openmm.LocalEnergyMinimizer.minimize(
                context, maxIterations=minimize_iterations,
            )

        # Equilibration phase: discard these steps so velocities
        # forget the zero-velocity initial state.
        for _ in range(max(n_equilibration_steps, 0)):
            integrator.step(1)

        # Production phase.  At each saved frame we evaluate the
        # current λ (always) plus the two adjacent λ values
        # (when sample_neighbor_lambdas=True and the neighbours
        # exist).
        prev_lam = lambdas[i - 1] if i > 0 else None
        next_lam = lambdas[i + 1] if i + 1 < n_windows else None
        for step in range(max(n_production_steps, 0)):
            integrator.step(1)
            if step % max(sample_every, 1) != 0:
                continue

            u_self = _energy_at_current_lam()
            energies_by_lambda[lam].append(u_self)

            if sample_neighbor_lambdas:
                u_prev: Optional[float] = None
                u_next: Optional[float] = None
                if prev_lam is not None:
                    context.setParameter("lam", float(prev_lam))
                    nb.updateParametersInContext(context)
                    u_prev = _energy_at_current_lam()
                if next_lam is not None:
                    context.setParameter("lam", float(next_lam))
                    nb.updateParametersInContext(context)
                    u_next = _energy_at_current_lam()
                # Restore own λ for the next MD step's force
                # calculation.
                context.setParameter("lam", float(lam))
                nb.updateParametersInContext(context)
                neighbor_energies[lam].append((u_prev, u_next))

    del context

    if not sample_neighbor_lambdas:
        # Backward-compatibility: caller that doesn't ask for
        # neighbor re-evaluation gets an empty dict instead of
        # three-element tuple surprise.
        neighbor_energies = {}

    return energies_by_lambda, neighbor_energies, disappearing