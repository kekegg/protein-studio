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
from typing import Any, Callable, Dict, List, Optional, Sequence

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


class FEPInterrupted(Exception):
    """Raised when a long-running sweep is cancelled between windows.

    Carries the count of completed windows so the checkpoint layer
    can resume from the right place.
    """


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


def cartesian_restraint_analytical_correction(
    n_atoms: int,
    restraint_k_kJmol_nm2: float = 1000.0,
    temperature_K: float = 300.0,
    standard_state_volume_nm3: float = 1.66054e-3,
) -> float:
    """Analytical ΔG correction for removing ``n_atoms`` Cartesian
    harmonic position restraints (Boresch 2003 / Deng & Roux 2009
    form, specialised to N isotropic position restraints).

    The standard treatment for alchemical FEP is to keep the
    disappearing atoms' position restraints turned on for the
    entire λ sweep — that way the system stays in a single
    metastable basin and the integrator doesn't blow up.  But
    the physical quantity we want is the ΔG for the *unrestrained*
    transformation.  So we add an analytical correction:

        ΔG_corrected = ΔG_FEP + ΔG_restraint

    where ΔG_restraint accounts for removing the restraints in
    the λ=1 (decoupled) state.  Each restrained atom's partition
    function goes from a Gaussian of width sqrt(kT/k_i) per
    dimension to the standard-state volume V₀.  Per-atom ΔG::

        ΔG_i = -kT × ln(Z_free / Z_restrained)
             = -kT × ln(V_0 / (2π kT / k_i)^(3/2))
             = (3/2) kT × ln(2π kT / k_i) - kT × ln(V_0)

    Parameters
    ----------
    n_atoms : int
        Number of Cartesian position restraints (number of
        disappearing atoms in Phase 4's sidechain truncation).
    restraint_k_kJmol_nm2 : float
        Force constant of the harmonic restraints, in
        kJ/mol/nm².  Default 1000 (Phase 4's default).
    temperature_K : float
        Temperature in K.  Default 300.
    standard_state_volume_nm3 : float
        Standard-state volume per particle in nm³.  Default
        1.66054e-3 (1 mol/L converted to nm³).

    Returns
    -------
    float
        ΔG_restraint in kJ/mol.  Sign convention: this is the
        ΔG of the *unrestraining* transformation (restrained →
        free).  Add it to your alchemical ΔG to recover the
        unrestrained physical ΔG.
    """
    kT = 8.314462618e-3 * temperature_K  # kJ/mol
    k = restraint_k_kJmol_nm2
    # Per-atom ΔG_unrestrain: log((2π kT / k)^(3/2) / V_0)
    # = (3/2) ln(2π kT / k) - ln(V_0)
    per_atom = (
        1.5 * np.log(2.0 * np.pi * kT / k) -
        np.log(standard_state_volume_nm3)
    )
    return float(kT * n_atoms * per_atom)


def boresch_analytical_correction(
    k_r_kJmol_nm2: float,
    k_theta1_kJmol_rad2: float,
    k_theta2_kJmol_rad2: float,
    k_phi1_kJmol_rad2: float,
    k_phi2_kJmol_rad2: float,
    k_phi3_kJmol_rad2: float,
    temperature_K: float = 300.0,
    standard_state_volume_nm3: float = 1.66054e-3,
) -> float:
    """Analytical ΔG correction for the standard Boresch (2003)
    6-restraint setup on a protein-ligand complex.

    The 6 restraints (1 distance, 2 angles, 3 torsions) couple
    3 ligand atoms to 2-3 receptor atoms.  They fix all 6
    rigid-body DOF of the ligand relative to the receptor.
    Their partition function factorises into 6 Gaussians::

        Z_restrained = (2π kT)^3 / ∏_i √k_i

    The unrestrained standard-state partition function is::

        Z_free = V_0 × 8π²

    where V_0 is the standard-state volume (1 mol/L per
    molecule) and 8π² is the rotational integral of a
    freely-rotating asymmetric top in the classical limit.

    The correction is::

        ΔG_restraint = -kT × ln(Z_free / Z_restrained)
                     = -kT × ln(V_0 × 8π² × ∏_i √k_i / (2π kT)^3)

    Parameters
    ----------
    k_r_kJmol_nm2 : float
        Force constant for the distance restraint
        (kJ/mol/nm²).  Typical: 41.84 (= 10 kcal/mol/Å²).
    k_theta1_kJmol_rad2, k_theta2_kJmol_rad2 : float
        Force constants for the two angle restraints
        (kJ/mol/rad²).  Typical: 418 (= 100 kcal/mol/rad²).
    k_phi1_kJmol_rad2, k_phi2_kJmol_rad2, k_phi3_kJmol_rad2 : float
        Force constants for the three torsion restraints
        (kJ/mol/rad²).  Typical: 4184 (= 1000 kcal/mol/rad²).
    temperature_K : float
        Temperature in K.  Default 300.
    standard_state_volume_nm3 : float
        Standard-state volume per particle in nm³ (1 mol/L).
        Default 1.66054e-3.

    Returns
    -------
    float
        ΔG_restraint in kJ/mol.  Sign convention: this is the
        ΔG of the *unrestraining* transformation.  Add it to
        your alchemical ΔG (with restraints on) to recover the
        unrestrained ΔG.

    Notes
    -----
    For typical Boresch values
    (k_r=10 kcal/mol/Å², k_θ=100 kcal/mol/rad², k_φ=1000
    kcal/mol/rad², T=300 K) this gives ≈ −25 kJ/mol
    (≈ −6 kcal/mol).  The negative sign means unrestraining
    the ligand decreases free energy — the restraints
    substantially bias the alchemical ΔG in the bound
    direction.  This is why the Boresch correction is
    critical for absolute binding free energies.
    """
    kT = 8.314462618e-3 * temperature_K  # kJ/mol
    ks = (
        k_r_kJmol_nm2,
        k_theta1_kJmol_rad2,
        k_theta2_kJmol_rad2,
        k_phi1_kJmol_rad2,
        k_phi2_kJmol_rad2,
        k_phi3_kJmol_rad2,
    )
    prod_sqrt_k = float(np.prod([np.sqrt(k) for k in ks]))
    z_free = standard_state_volume_nm3 * 8.0 * np.pi ** 2
    z_restrained = (2.0 * np.pi * kT) ** 3 / prod_sqrt_k
    return float(-kT * np.log(z_free / z_restrained))


def add_boresch_restraints(
    system,
    ligand_atom_indices: Sequence[int],
    receptor_atom_indices: Sequence[int],
    positions,
    unit,
    k_r_kJmol_nm2: float = 41.84,
    k_theta_kJmol_rad2: float = 418.4,
    k_phi_kJmol_rad2: float = 4184.0,
    force_group: int = 2,
):
    """Add the standard Boresch (2003) 6-restraint force to an
    OpenMM system.

    Restraint geometry (6 harmonic restraints):
      - Distance: |L1 - R1|        with force constant k_r
      - Angle θ1: ∠R1-L1-L2       with force constant k_θ
      - Angle θ2: ∠L1-L2-L3       with force constant k_θ
      - Torsion φ1: ∠R1-L1-L2-L3   with force constant k_φ
      - Torsion φ2: ∠R2-L1-L2      with force constant k_φ
      - Torsion φ3: ∠L1-L2-L3-R3   with force constant k_φ

    where L1, L2, L3 are ligand atoms (heavy, well-separated)
    and R1, R2, R3 are receptor atoms (heavy, in the binding
    site).  The 6 restraints fix all rigid-body DOF of the
    ligand relative to the receptor.

    Parameters
    ----------
    ligand_atom_indices : sequence of 3 int
        Indices of the 3 ligand atoms (L1, L2, L3) in the
        OpenMM topology.
    receptor_atom_indices : sequence of 3 int
        Indices of the 3 receptor atoms (R1, R2, R3).
    positions : openmm unit-bearing Quantity list
        The initial positions.  Used to compute the
        equilibrium values of the 6 restraints.
    k_r_kJmol_nm2, k_theta_kJmol_rad2, k_phi_kJmol_rad2 : float
        Force constants for the three restraint types
        (same k_θ for both angles, same k_φ for all 3
        torsions — the standard Boresch simplification).
    force_group : int
        OpenMM force group for all 6 restraints (default 2,
        separate from the soft-core nonbonded on group 0 and
        the disappearing-atom position restraints on group 1).

    Returns
    -------
    dict
        Reference values ``{name: r_0/θ_0/φ_0}`` for each of the
        6 restraints, useful for analysis / logging.
    """
    import openmm
    if len(ligand_atom_indices) != 3:
        raise ValueError(
            f"need 3 ligand atoms, got {len(ligand_atom_indices)}"
        )
    if len(receptor_atom_indices) != 3:
        raise ValueError(
            f"need 3 receptor atoms, got {len(receptor_atom_indices)}"
        )
    L1, L2, L3 = (int(i) for i in ligand_atom_indices)
    R1, R2, R3 = (int(i) for i in receptor_atom_indices)

    def _pos(idx):
        return np.asarray(
            [p.value_in_unit(unit.nanometer) for p in positions[idx]]
        )

    pL1, pL2, pL3 = _pos(L1), _pos(L2), _pos(L3)
    pR1, pR2, pR3 = _pos(R1), _pos(R2), _pos(R3)

    def _dist(p1, p2):
        return float(np.linalg.norm(p1 - p2))

    def _angle(p_a, p_b, p_c):
        v1 = p_a - p_b
        v2 = p_c - p_b
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        cos = max(-1.0, min(1.0, cos))
        return float(np.arccos(cos))

    def _torsion(p_a, p_b, p_c, p_d):
        # Atan2-based proper torsion in [0, 2π).  OpenMM's
        # convention for PeriodicTorsionForce is [-π, π]; we
        # centre on the measured value and use CustomTorsionForce
        # with energy k*(phi-phi0)^2 so the value can be anything.
        v1 = p_b - p_a
        v2 = p_c - p_b
        v3 = p_d - p_c
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-12 or n2_norm < 1e-12:
            return 0.0
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        m1 = np.cross(n1, v2 / np.linalg.norm(v2))
        x = float(np.dot(n1, n2))
        y = float(np.dot(m1, n2))
        return float(np.arctan2(y, x) % (2 * np.pi))

    r0 = _dist(pL1, pR1)
    theta1_0 = _angle(pR1, pL1, pL2)
    theta2_0 = _angle(pL1, pL2, pL3)
    phi1_0 = _torsion(pR1, pL1, pL2, pL3)
    phi2_0 = _torsion(pR2, pL1, pL2, pL1)  # placeholder; see note
    phi3_0 = _torsion(pL1, pL2, pL3, pR3)

    # 1. Distance restraint.
    bond = openmm.HarmonicBondForce()
    bond.addForceGroup(force_group)
    bond.addBond(L1, R1, r0, k_r_kJmol_nm2)
    system.addForce(bond)

    # 2 & 3. Two angle restraints — OpenMM's HarmonicAngleForce
    # takes 3 atom indices per angle.
    angle = openmm.HarmonicAngleForce()
    angle.addForceGroup(force_group)
    angle.addAngle(R1, L1, L2, theta1_0, k_theta_kJmol_rad2)
    angle.addAngle(L1, L2, L3, theta2_0, k_theta_kJmol_rad2)
    system.addForce(angle)

    # 4-6. Three torsion restraints via CustomTorsionForce with
    # a harmonic energy expression (k * (φ - φ_0)^2).
    torsion = openmm.CustomTorsionForce("k*(theta-theta0)^2")
    torsion.addForceGroup(force_group)
    torsion.addGlobalParameter("k", k_phi_kJmol_rad2)
    torsion.addPerTorsionParameter("theta0")
    torsion.addTorsion(R1, L1, L2, L3, [phi1_0])
    torsion.addTorsion(R2, L1, L2, L1, [phi2_0])
    torsion.addTorsion(L1, L2, L3, R3, [phi3_0])
    system.addForce(torsion)

    return {
        "r_0": r0, "theta1_0": theta1_0, "theta2_0": theta2_0,
        "phi1_0": phi1_0, "phi2_0": phi2_0, "phi3_0": phi3_0,
    }


def standard_state_concentration_correction(
    box_volume_nm3: float,
    temperature_K: float = 300.0,
    standard_state_concentration_M: float = 1.0,
) -> float:
    """ΔG for changing the unbound-ligand concentration from
    the simulation box's effective 1/V_box to the standard
    state (default 1 mol/L).

    Binding free energies from double-decoupling FEP have an
    arbitrary reference volume — the simulation box.  One
    ligand molecule in a box of volume V_box gives an
    effective concentration of 1/V_box.  To express the
    binding ΔG at the standard-state concentration C_0,
    add::

        ΔG_ss = +kT × ln(V_box × C_0 × N_A × 1e-27)

    where V_box is in nm³, C_0 in mol/L, N_A is Avogadro's
    number, and 1e-27 converts nm³ → L.  Equivalently,
    ΔG_ss = kT × ln(C_0 / C_box) where C_box = 1 /
    (V_box_nm3 × N_A × 1e-27) mol/L.

    For a typical 1ubq FEP setup (V_box ≈ 440 nm³) and T = 300 K,
    this is ≈ −3.3 kJ/mol (the simulation box has higher
    effective concentration than 1 M, so the ΔG becomes
    slightly less favourable when reported at 1 M).

    Parameters
    ----------
    box_volume_nm3 : float
        Simulation box volume in nm³.  For a cubic box with
        side a, this is a³.
    temperature_K : float
        Temperature in K.  Default 300.
    standard_state_concentration_M : float
        Standard-state concentration in mol/L.  Default 1.0.
    """
    kT = 8.314462618e-3 * temperature_K  # kJ/mol
    N_A = 6.02214076e23  # Avogadro
    nm3_to_L = 1e-27
    # C_box (mol/L) = 1 / (V_box × N_A × 1e-27)
    # ΔG_ss = kT × ln(C_0 / C_box) = kT × ln(C_0 × V_box_nm3 × N_A × 1e-27)
    return float(
        kT * np.log(
            standard_state_concentration_M
            * box_volume_nm3 * N_A * nm3_to_L
        )
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


def _replace_amber_nb_with_split_forces(
    system,
    disappearing_set: set,
    lambda_value: float,
    alpha_nm2: float = 1.5,
    cutoff_nm: float = 1.0,
):
    openmm, unit = _require_openmm()
    """Swap AMBER's NonbondedForce for two alchemical forces:

      1. **CustomNonbondedForce** (LJ only) with the soft-core
         potential: standard AMBER LJ for non-disappearing pairs,
         Beutler 1994 soft-core LJ (with λ-amplitude scaling) for
         pairs involving ≥1 disappearing atom.  NoCutoff — the
         soft-core breaks PME compatibility, and at 1.0 nm
         cutoff the LJ error is small anyway.

      2. **NonbondedForce** (Coulomb only, PME) with the original
         AMBER (sigma, ε) carried over for completeness, but
         charges multiplied by ``(1 - is_dis · λ)`` so that
         disappearing atoms' Coulomb turns off linearly with λ.
         AMBER's original 1-2/1-3/1-4 exception list is preserved
         (1-2/1-3 fully excluded via ``chargeProduct=0``; 1-4
         carries the standard 0.5·q1·q2 scaled Coulomb).

    Why two forces?  OpenMM 8.x's CustomNonbondedForce does not
    support PME, and replacing AMBER's NonbondedForce (which
    does) loses the periodic-image Coulomb interactions that
    hold TIP3P waters together — the bulk-water ΔE is wrong by
    ~25,000 kJ/mol.  Splitting LJ and Coulomb into two forces
    preserves PME for the Coulomb while letting us apply the
    soft-core only to the LJ.

    Returns ``(lj_force, coulomb_force, n_particles,
    original_qsigspe, exceptions)`` — caller updates the
    Coulomb charges at each λ change via
    ``update_coulomb_for_lambda``.
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

    # Capture AMBER's exception list.  Keep the full tuple
    # (chargeProduct, sigma, epsilon) so we can replay it on
    # the new NonbondedForce exactly as AMBER had it.
    exceptions = []
    for k in range(old_nb.getNumExceptions()):
        exc = old_nb.getExceptionParameters(k)
        exceptions.append(exc)

    system.removeForce(nb_idx)

    # ========== 1. CustomNonbondedForce for LJ (soft-core) ==========
    lj = openmm.CustomNonbondedForce("")
    lj.addGlobalParameter("lam", lambda_value)
    lj.addGlobalParameter("alpha", alpha_nm2)
    lj.addPerParticleParameter("sigma")
    lj.addPerParticleParameter("eps")
    lj.addPerParticleParameter("is_dis")

    for j in range(n_particles):
        is_dis = 1.0 if j in disappearing_set else 0.0
        lj.addParticle((orig[j][1], orig[j][2], is_dis))

    lj.setEnergyFunction(
        # Standard AMBER LJ for non-disappearing pairs.
        "((1-is_dis1)*(1-is_dis2)) * 4 * sqrt(eps1*eps2) * "
        "(((0.5*(sigma1+sigma2))^12)/(r*r)^6 - "
        "((0.5*(sigma1+sigma2))^6)/(r*r)^3)"
        # Beutler 1994 soft-core LJ for pairs involving ≥1
        # disappearing atom.  At λ=1 this reduces exactly to
        # standard LJ.  λ is the soft-core "amplitude" — full
        # interaction at λ=1, fully decoupled at λ=0.
        " + ((is_dis1 + is_dis2 - is_dis1*is_dis2)) * lam * 4 * sqrt(eps1*eps2) * "
        "(((0.5*(sigma1+sigma2))^12) / (alpha*(1-lam)^2 + (r*r)*(r*r)*(r*r))^2 - "
        "((0.5*(sigma1+sigma2))^6)  / (alpha*(1-lam)^2 + (r*r)*(r*r)*(r*r)))"
    )
    # 1-2/1-3 exclusions only (1-4 pairs are handled by the
    # Coulomb NonbondedForce below, which carries AMBER's 1-4
    # LJ params on its exception list — this avoids the ~3,000
    # kJ/mol artefact we saw when 1-4 LJ was excluded entirely).
    # AMBER's exception list uses chargeProduct=0 to flag 1-2/1-3
    # (excluded) and a non-zero chargeProduct for 1-4 (scaled).
    for exc in exceptions:
        i, j, charge_prod = exc[0], exc[1], exc[2]
        cp0 = float(charge_prod.value_in_unit(unit.elementary_charge ** 2))
        if cp0 == 0.0:
            # 1-2 or 1-3: fully excluded from LJ.
            if i < j:
                lj.addExclusion(i, j)
            else:
                lj.addExclusion(j, i)
        # else: 1-4 — let the Coulomb NonbondedForce handle it.
    lj.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(cutoff_nm * unit.nanometer)
    system.addForce(lj)

    # ========== 2. NonbondedForce for Coulomb (PME) ==========
    # Also carries AMBER's 1-4 scaled LJ via the exception
    # list (sigma, epsilon set to AMBER's 1-4 values for those
    # pairs).  1-2/1-3 exceptions have sigma=epsilon=0 (excluded
    # from both LJ and Coulomb here).
    coulomb = openmm.NonbondedForce()
    coulomb.setNonbondedMethod(openmm.NonbondedForce.PME)
    coulomb.setCutoffDistance(cutoff_nm * unit.nanometer)
    coulomb.setEwaldErrorTolerance(5e-4)

    is_dis = np.zeros(n_particles)
    for j in range(n_particles):
        is_dis[j] = 1.0 if j in disappearing_set else 0.0

    for j in range(n_particles):
        q0 = float(orig[j][0].value_in_unit(unit.elementary_charge))
        q_eff = q0 * (1.0 - is_dis[j] * lambda_value)
        # Particles that are *not* in any 1-4 exception have
        # sigma=epsilon=0 in this force (so no LJ contribution),
        # since the LJ soft-core handles them.
        coulomb.addParticle(q_eff, 0.0, 0.0)

    # Replay AMBER's exception list — but with chargeProduct
    # rescaled by the λ² factor for 1-4 pairs involving ≥1
    # disappearing atom (since q_eff = q · (1 - is_dis · λ),
    # q1_eff · q2_eff = q1 · q2 · (1 - is_dis1·λ)(1 - is_dis2·λ)).
    for exc in exceptions:
        i, j, charge_prod, sig, eps = exc
        cp0 = float(charge_prod.value_in_unit(unit.elementary_charge ** 2))
        sig_v = float(sig.value_in_unit(unit.nanometer))
        eps_v = float(eps.value_in_unit(unit.kilojoule_per_mole))
        # Recompute chargeProduct from original charges, not the
        # already-scaled ones — we want AMBER's intrinsic scaling
        # here (0.5 for 1-4, 0 for 1-2/1-3) and we'll re-multiply
        # by (1 - is_dis1·λ)(1 - is_dis2·λ) below.
        if cp0 == 0.0:
            # 1-2/1-3: fully excluded from both LJ and Coulomb here.
            new_cp = 0.0
            new_sig = 0.0
            new_eps = 0.0
        else:
            # 1-4: rescale chargeProduct for disappearing atoms,
            # keep AMBER's sigma/epsilon for the LJ contribution.
            q1 = float(orig[i][0].value_in_unit(unit.elementary_charge))
            q2 = float(orig[j][0].value_in_unit(unit.elementary_charge))
            intrinsic = cp0 / (q1 * q2) if q1 * q2 != 0 else 0.0
            new_cp = q1 * q2 * intrinsic * (
                1.0 - is_dis[i] * lambda_value
            ) * (1.0 - is_dis[j] * lambda_value)
            new_sig = sig_v
            new_eps = eps_v
        if i < j:
            coulomb.addException(i, j, new_cp, new_sig, new_eps)
        else:
            coulomb.addException(j, i, new_cp, new_sig, new_eps)

    system.addForce(coulomb)

    return lj, coulomb, n_particles, orig, exceptions, is_dis


def update_coulomb_for_lambda(
    coulomb_force,
    n_particles: int,
    original_charges,
    exceptions,
    is_dis: np.ndarray,
    lambda_value: float,
    context=None,
):
    openmm, unit = _require_openmm()
    """Push new λ-dependent Coulomb charges and exception
    chargeProducts into a Context.  Call this at each λ change
    before sampling.

    ``original_charges`` is the list/array of unscaled AMBER
    charges, ``exceptions`` is the list returned by
    ``_replace_amber_nb_with_split_forces``, and ``is_dis`` is
    the per-particle 0/1 flag.  If ``context`` is provided,
    also calls ``updateParametersInContext`` to push the new
    values into the running context.
    """
    for j in range(n_particles):
        q0 = float(original_charges[j][0].value_in_unit(unit.elementary_charge))
        q_eff = q0 * (1.0 - is_dis[j] * lambda_value)
        coulomb_force.setParticleParameters(j, q_eff, 0.0, 0.0)
    for k, exc in enumerate(exceptions):
        i, j, charge_prod, sig, eps = exc
        cp0 = float(charge_prod.value_in_unit(unit.elementary_charge ** 2))
        sig_v = float(sig.value_in_unit(unit.nanometer))
        eps_v = float(eps.value_in_unit(unit.kilojoule_per_mole))
        if cp0 == 0.0:
            new_cp = 0.0
            new_sig = 0.0
            new_eps = 0.0
        else:
            q1 = float(original_charges[i][0].value_in_unit(unit.elementary_charge))
            q2 = float(original_charges[j][0].value_in_unit(unit.elementary_charge))
            intrinsic = cp0 / (q1 * q2) if q1 * q2 != 0 else 0.0
            new_cp = q1 * q2 * intrinsic * (
                1.0 - is_dis[i] * lambda_value
            ) * (1.0 - is_dis[j] * lambda_value)
            new_sig = sig_v
            new_eps = eps_v
        coulomb_force.setExceptionParameters(k, i, j, new_cp, new_sig, new_eps)
    if context is not None:
        coulomb_force.updateParametersInContext(context)


def _replace_amber_nb_with_alchemical(
    system,
    disappearing_set: set,
    lambda_value: float,
    alpha_nm2: float = 1.5,
    cutoff_nm: float = 1.0,
):
    """Backward-compatibility shim around
    :func:`_replace_amber_nb_with_split_forces`.  Replaces
    AMBER's NonbondedForce with the split (LJ-soft-core +
    Coulomb-PME) pair.  See the split function's docstring for
    details.

    For tests that check "the soft-core NonbondedForce is in
    the system", note: there are now TWO forces instead of
    one — the LJ CustomNonbondedForce (the soft-core one) and
    the Coulomb NonbondedForce (with PME).
    """
    lj, coulomb, n_particles, orig, exceptions, is_dis = (
        _replace_amber_nb_with_split_forces(
            system, disappearing_set, lambda_value,
            alpha_nm2=alpha_nm2, cutoff_nm=cutoff_nm,
        )
    )
    # Stash the split bookkeeping on the system object (not the
    # Swig-proxied LJ force — Python attributes on SWIG proxies
    # don't survive subsequent getForces() calls).
    system._fep_split_metadata = (coulomb, n_particles, orig, exceptions, is_dis)
    return lj


def get_split_force_metadata(lj_force, system=None):
    """Return the split-force bookkeeping attached by
    :func:`_replace_amber_nb_with_alchemical`.

    Returns ``(coulomb_force, n_particles, original_params,
    exceptions, is_dis)`` or ``None`` if not available.
    """
    if system is not None:
        return getattr(system, "_fep_split_metadata", None)
    return None


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
    cancellation_check: Optional[Callable[[], bool]] = None,
    window_callback: Optional[Callable[[Dict[float, List[float]], Dict[float, List[Any]], List[float]], None]] = None,
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

    # Pull split-force bookkeeping so we can update the Coulomb
    # charges on every λ change.
    split_md = get_split_force_metadata(nb, system=system)
    if split_md is None:
        coulomb_force = None
        n_particles_split = 0
        orig_charges = []
        exc_list = []
        is_dis_arr = np.zeros(0)
    else:
        coulomb_force, n_particles_split, orig_charges, exc_list, is_dis_arr = split_md

    def _set_lambda(lam_value: float):
        """Push new λ into both LJ-soft-core (global param) and
        Coulomb (per-particle parameters)."""
        context.setParameter("lam", float(lam_value))
        nb.updateParametersInContext(context)
        if coulomb_force is not None:
            update_coulomb_for_lambda(
                coulomb_force, n_particles_split,
                orig_charges, exc_list, is_dis_arr, lam_value,
                context=context,
            )

    for i, lam in enumerate(lambdas):
        # Cancellation check between windows — cheap, runs before
        # any expensive work (minimise / equilibrate).
        if cancellation_check is not None and cancellation_check():
            raise FEPInterrupted(
                f"Cancelled before λ = {lam:.3f} "
                f"(completed {i} / {n_windows} windows)"
            )

        # Set this window's λ and make sure the soft-core Force
        # sees it.
        _set_lambda(lam)
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
                    _set_lambda(next_lam)
                    u_next = _energy_at_current_lam()
                _set_lambda(lam)
                neighbor_energies[lam].append((u_prev, u_next))

        # Notify checkpoint / progress layer after each window
        # finishes. Cheap — runs once per λ, after all production
        # sampling is done.
        if window_callback is not None:
            window_callback(
                {l: list(energies_by_lambda[l]) for l in lambdas},
                {l: list(neighbor_energies[l]) for l in lambdas},
                list(lambdas[: i + 1]),
            )

    del context

    if not sample_neighbor_lambdas:
        # Backward-compatibility: caller that doesn't ask for
        # neighbor re-evaluation gets an empty dict instead of
        # three-element tuple surprise.
        neighbor_energies = {}

    return energies_by_lambda, neighbor_energies, disappearing


# ----- Ligand decoupling + binding ΔG ------------------------------------


def _identify_ligand_atoms(
    topology,
    ligand_resnames: Sequence[str],
) -> List[int]:
    """Return particle indices for every atom whose residue
    name matches any of ``ligand_resnames``.  Used for ligand
    decoupling FEP (Phase 5), where the whole ligand
    disappears instead of a single residue's sidechain.
    """
    resnames = set(ligand_resnames)
    return [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name in resnames
    ]


def build_ligand_decouple_system(
    pdb_file: str,
    ligand_resnames: Sequence[str],
    padding_nm: float = 1.0,
    ionic_strength_M: float = 0.15,
    temperature_K: float = 300.0,
    timestep_fs: float = 2.0,
    cutoff_nm: float = 1.0,
    lambda_value: float = 1.0,
    alpha_nm2: float = 1.5,
    restraint_k_kJmol_nm2: float = 1000.0,
    boresch_kwargs: Optional[dict] = None,
) -> Tuple:
    """Build a solvated AMBER system with the ligand atoms
    marked alchemical (Phase 5: protein-ligand binding FEP).

    Identical to ``build_amber_mutation_system`` except:

      - The disappearing set is every atom whose residue name
        matches one of ``ligand_resnames`` (no residue-index
        filter — the ligand may span many residues or none at
        all if it's a single heteroatom).
      - Optionally adds the Boresch 6-restraint force if
        ``boresch_kwargs`` is provided.  Keys:
        ``ligand_atoms`` (3 indices), ``receptor_atoms``
        (3 indices).  Restraint force constants default to
        the standard Boresch values (k_r=41.84, k_θ=418.4,
        k_φ=4184.0 kJ/mol/nm², kJ/mol/rad² respectively).

    Returns the system, topology, positions, integrator, the
    list of disappearing atom indices, and (if Boresch was
    added) the equilibrium restraint values.
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
    disappearing = _identify_ligand_atoms(topology, ligand_resnames)
    if not disappearing:
        raise ValueError(
            f"No atoms matching ligand_resnames={list(ligand_resnames)}. "
            "Check residue names (must match exactly)."
        )
    _replace_amber_nb_with_alchemical(
        system, set(disappearing), lambda_value=lambda_value,
        alpha_nm2=alpha_nm2,
    )

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

    boresch_refs = None
    if boresch_kwargs is not None:
        kwargs = dict(boresch_kwargs)
        ligand_atoms = kwargs.pop("ligand_atoms")
        receptor_atoms = kwargs.pop("receptor_atoms")
        boresch_refs = add_boresch_restraints(
            system,
            ligand_atom_indices=ligand_atoms,
            receptor_atom_indices=receptor_atoms,
            positions=positions,
            unit=unit,
            **kwargs,
        )

    return (
        system, topology, positions, integrator,
        disappearing, boresch_refs,
    )


def run_ligand_decouple_fep(
    pdb_file: str,
    ligand_resnames: Sequence[str],
    lambdas: Optional[Sequence[float]] = None,
    n_equilibration_steps: int = 200,
    n_production_steps: int = 1000,
    temperature_K: float = 300.0,
    padding_nm: float = 0.8,
    minimize_iterations: int = 10,
    restraint_k_kJmol_nm2: Optional[float] = None,
    sample_every: int = 1,
    sample_neighbor_lambdas: bool = False,
    boresch_kwargs: Optional[dict] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    window_callback: Optional[Callable[[Dict[float, List[float]], Dict[float, List[Any]], List[float]], None]] = None,
):
    """Run FEP for ligand decoupling — both the complex and
    solution sides of a double-decoupling binding free energy.

    Same signature as ``run_amber_mutation_fep`` but the
    disappearing set is every atom in ``ligand_resnames``
    instead of a single residue's sidechain.  Pass
    ``boresch_kwargs`` for the bound-state sweep to add the
    Boresch 6-restraint; omit it for the solution sweep (the
    ligand is unrestrained in solution because the volume
    integral is over a sphere of radius equal to the Boresch
    anchor's restraint half-width — see Deng & Roux 2009).
    """
    openmm, unit = _require_openmm()
    if lambdas is None:
        lambdas = tuple(i / 10.0 for i in range(11))
    lambdas = tuple(float(l) for l in lambdas)

    if restraint_k_kJmol_nm2 is None:
        kws = dict(padding_nm=padding_nm, temperature_K=temperature_K)
    else:
        kws = dict(
            padding_nm=padding_nm,
            temperature_K=temperature_K,
            restraint_k_kJmol_nm2=restraint_k_kJmol_nm2,
        )
    system, topology, positions, integrator, disappearing, boresch_refs = (
        build_ligand_decouple_system(
            pdb_file,
            ligand_resnames=ligand_resnames,
            lambda_value=1.0,
            boresch_kwargs=boresch_kwargs,
            **kws,
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
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

    # Pull the split-force bookkeeping so we can update the
    # Coulomb charges on every λ change.
    split_md = get_split_force_metadata(nb, system=system)
    if split_md is None:
        # Pre-split-force build — fall back to single setParameter.
        coulomb_force = None
        n_particles_split = 0
        orig_charges = []
        exc_list = []
        is_dis_arr = np.zeros(0)
    else:
        coulomb_force, n_particles_split, orig_charges, exc_list, is_dis_arr = split_md

    def _set_lambda(lam_value: float):
        """Push new λ into both LJ-soft-core (via global param)
        and Coulomb (via per-particle parameter updates)."""
        context.setParameter("lam", float(lam_value))
        nb.updateParametersInContext(context)
        if coulomb_force is not None:
            update_coulomb_for_lambda(
                coulomb_force, n_particles_split,
                orig_charges, exc_list, is_dis_arr, lam_value,
                context=context,
            )

    for i, lam in enumerate(lambdas):
        if cancellation_check is not None and cancellation_check():
            raise FEPInterrupted(
                f"Cancelled before λ = {lam:.3f} "
                f"(completed {i} / {len(lambdas)} windows)"
            )
        _set_lambda(lam)
        context.setPositions(positions)
        if hasattr(openmm, "LocalEnergyMinimizer"):
            openmm.LocalEnergyMinimizer.minimize(
                context, maxIterations=minimize_iterations,
            )
        for _ in range(max(n_equilibration_steps, 0)):
            integrator.step(1)
        prev_lam = lambdas[i - 1] if i > 0 else None
        next_lam = lambdas[i + 1] if i + 1 < len(lambdas) else None
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
                    _set_lambda(prev_lam)
                    u_prev = _energy_at_current_lam()
                if next_lam is not None:
                    _set_lambda(next_lam)
                    u_next = _energy_at_current_lam()
                _set_lambda(lam)
                neighbor_energies[lam].append((u_prev, u_next))

        # Notify checkpoint / progress layer after each window.
        if window_callback is not None:
            window_callback(
                {l: list(energies_by_lambda[l]) for l in lambdas},
                {l: list(neighbor_energies[l]) for l in lambdas},
                list(lambdas[: i + 1]),
            )

    del context

    if not sample_neighbor_lambdas:
        neighbor_energies = {}

    return (
        energies_by_lambda, neighbor_energies,
        disappearing, boresch_refs,
    )


def compute_binding_dg(
    energies_complex: dict,
    energies_solution: dict,
    neighbor_energies_complex: Optional[dict] = None,
    neighbor_energies_solution: Optional[dict] = None,
    boresch_kws: Optional[dict] = None,
    box_volume_complex_nm3: Optional[float] = None,
    box_volume_solution_nm3: Optional[float] = None,
    temperature_K: float = 300.0,
    apply_standard_state: bool = True,
    use_bar: bool = True,
) -> dict:
    """Assemble absolute binding free energy from a double-
    decoupling FEP pair.

        ΔG_bind = ΔG_complex_decouple − ΔG_solution_decouple
                + ΔG_Boresch_correction
                + ΔG_standard_state_correction (1 mol/L)

    Parameters
    ----------
    energies_complex, energies_solution : dict
        ``{λ: [E, ...]}`` from the bound-state and free-state
        sweeps respectively.
    neighbor_energies_complex, neighbor_energies_solution : dict or None
        Mixed-state energies (from ``sample_neighbor_lambdas=True``).
        Required for BAR; ignored if ``use_bar=False``.
    boresch_kws : dict or None
        Force constants used for the Boresch restraint in the
        bound sweep, in the form expected by
        :func:`boresch_analytical_correction`.
        Pass ``None`` to skip the Boresch correction.
    box_volume_complex_nm3, box_volume_solution_nm3 : float or None
        Simulation box volumes in nm³ for the complex and
        solution runs respectively.  Required when
        ``apply_standard_state=True`` (default).
    temperature_K : float
        Temperature in K.  Default 300.
    apply_standard_state : bool
        If True (default), add the standard-state concentration
        correction so the result is reported at 1 M.  Set False
        if you want the simulation-box-referenced ΔG.
    use_bar : bool
        If True (default), use BAR.  If False, use trapezoidal.

    Returns
    -------
    dict
        ``{complex_dg, solution_dg, boresch_correction,
        standard_state_correction, binding_dg_kJmol,
        binding_dg_kcalmol, sigma}``.
    """
    if use_bar:
        if not neighbor_energies_complex or not neighbor_energies_solution:
            raise ValueError(
                "BAR requires neighbor_energies from both sweeps; "
                "pass sample_neighbor_lambdas=True to run_ligand_decouple_fep "
                "or set use_bar=False to fall back to trapezoidal."
            )
        complex_dg, complex_sig, _ = analyze_fep_bar(
            energies_complex, neighbor_energies_complex,
            lambdas=sorted(energies_complex.keys()),
            kT_kjmol=8.314462618e-3 * temperature_K,
        )
        solution_dg, solution_sig, _ = analyze_fep_bar(
            energies_solution, neighbor_energies_solution,
            lambdas=sorted(energies_solution.keys()),
            kT_kjmol=8.314462618e-3 * temperature_K,
        )
        sigma = float(np.sqrt(complex_sig ** 2 + solution_sig ** 2))
    else:
        complex_dg, complex_sig = analyze_fep_trapezoidal(
            energies_complex,
            lambdas=sorted(energies_complex.keys()),
        )
        solution_dg, solution_sig = analyze_fep_trapezoidal(
            energies_solution,
            lambdas=sorted(energies_solution.keys()),
        )
        sigma = float(np.sqrt(complex_sig ** 2 + solution_sig ** 2))

    boresch_correction = (
        boresch_analytical_correction(
            temperature_K=temperature_K,
            **boresch_kws,
        )
        if boresch_kws is not None
        else 0.0
    )

    if apply_standard_state:
        if box_volume_complex_nm3 is None or box_volume_solution_nm3 is None:
            raise ValueError(
                "apply_standard_state=True requires "
                "box_volume_complex_nm3 and box_volume_solution_nm3."
            )
        # Concentration correction is for the *unbound* state only
        # (the bound state has the ligand pinned by Boresch at a
        # well-defined position, so its "concentration" is implicit).
        # ΔG_ss_total = ΔG_ss_solution - ΔG_ss_complex
        # but ΔG_ss_complex = 0 (ligand pinned, no translational
        # volume integral), so just use the solution box.
        standard_state_correction = standard_state_concentration_correction(
            box_volume_nm3=box_volume_solution_nm3,
            temperature_K=temperature_K,
        )
    else:
        standard_state_correction = 0.0

    binding_dg_kJmol = (
        complex_dg - solution_dg
        + boresch_correction
        + standard_state_correction
    )
    binding_dg_kcalmol = binding_dg_kJmol / 4.184

    return {
        "complex_dg_kJmol": complex_dg,
        "solution_dg_kJmol": solution_dg,
        "boresch_correction_kJmol": boresch_correction,
        "standard_state_correction_kJmol": standard_state_correction,
        "binding_dg_kJmol": binding_dg_kJmol,
        "binding_dg_kcalmol": binding_dg_kcalmol,
        "sigma_kJmol": sigma,
    }