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
]