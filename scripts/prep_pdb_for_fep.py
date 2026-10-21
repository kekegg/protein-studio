#!/usr/bin/env python
"""Pre-clean a PDB file before running FEP.

OpenMM's ``addSolvent`` is the standard path for solvating a
system, but it occasionally places waters within ~0.1 nm of each
other.  When that happens, the Lennard-Jones force at those
short distances is large enough that the 2 fs Langevin
integrator NaNs within a few dozen steps before any minimise
or equilibration can recover.  This shows up as
``Particle coordinate is NaN`` in ``run_amber_mutation_fep``.

Production FEP tools (pmx, alchemlyb, FEP++) avoid this by
running their full prep pipeline (AmberTools' ``tleap`` or
GROMACS' ``grompp``) which guarantees a clean starting state
through tleap's built-in min-equil-min loop.  This script is a
*lightweight* equivalent — good enough for the protein-studio
test cases, not a substitute for the production toolchain.

What it does:
1. Load the PDB with PDBFixer.
2. Find and fill missing residues / atoms.
3. Strip heterogens (water, ligands, ions) — they don't survive
   ``addSolvent`` reliably anyway, and the protein-studio FEP
   panel re-solvates from scratch.
4. Add hydrogens at pH 7.0.
5. Run a 200-step L-BFGS minimise on the protein alone to fix
   any backbone clashes (heavy-atom only).
6. Save the cleaned PDB.

Usage::

    python scripts/prep_pdb_for_fep.py input.pdb output.pdb
"""
import argparse
import sys
from pathlib import Path

from openmm import unit, LangevinMiddleIntegrator, Platform
from openmm.app import PDBFile, HBonds
from pdbfixer import PDBFixer


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Clean a PDB for FEP sweep (PDBFixer + light minimise)."
    )
    p.add_argument("input_pdb", type=Path)
    p.add_argument("output_pdb", type=Path)
    p.add_argument(
        "--keep-heterogens", action="store_true",
        help="Don't strip ligands/waters/ions — useful when the "
             "input is already a pre-solvated structure.",
    )
    p.add_argument(
        "--minimize-iterations", type=int, default=200,
        help="Heavy-atom minimisation rounds (L-BFGS).",
    )
    p.add_argument(
        "--no-minimize", action="store_true",
        help="Skip the heavy-atom minimise step.",
    )
    p.add_argument(
        "--pH", type=float, default=7.0,
        help="pH for hydrogen placement (default 7.0).",
    )
    args = p.parse_args(argv)

    if not args.input_pdb.is_file():
        sys.exit(f"input PDB not found: {args.input_pdb}")

    print(f"[prep] loading {args.input_pdb}", flush=True)
    fixer = PDBFixer(filename=str(args.input_pdb))

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    if fixer.missingAtoms or fixer.missingResidues:
        print(
            f"[prep] filling {len(fixer.missingAtoms)} missing atoms "
            f"and {len(fixer.missingResidues)} missing residues",
            flush=True,
        )
        fixer.addMissingAtoms()
    else:
        print("[prep] structure already complete", flush=True)

    if not args.keep_heterogens:
        n_before = sum(1 for _ in fixer.topology.residues())
        fixer.removeHeterogens(keepWater=False)
        n_after = sum(1 for _ in fixer.topology.residues())
        print(
            f"[prep] stripped {n_before - n_after} heterogen residues "
            f"(kept protein / NA only)",
            flush=True,
        )

    fixer.addMissingHydrogens(pH=args.pH)
    print(f"[prep] added hydrogens at pH {args.pH}", flush=True)

    if not args.no_minimize:
        print(
            f"[prep] heavy-atom minimise: {args.minimize_iterations} iters",
            flush=True,
        )
        _heavy_atom_minimise(
            fixer.topology, fixer.positions,
            iterations=args.minimize_iterations,
        )

    print(f"[prep] writing {args.output_pdb}", flush=True)
    args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    n_atoms = sum(1 for _ in fixer.topology.atoms())
    print(f"[prep] done — {n_atoms} atoms written", flush=True)


def _heavy_atom_minimise(topology, positions, iterations):
    """Quick AMBER-Bondi heavy-atom minimise — no water, just the
    protein.  Used to fix clashes that would otherwise survive
    into the FEP sweep and NaN the integrator.
    """
    import numpy as np
    import openmm as _openmm
    from openmm.app import ForceField, NoCutoff

    ff = ForceField("amber99sbildn.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtosecond,
    )
    platform = Platform.getPlatformByName("CPU")
    ctx = _openmm.Context(system, integrator, platform)
    ctx.setPositions(positions)
    _openmm.LocalEnergyMinimizer.minimize(
        ctx, maxIterations=iterations,
    )
    state = ctx.getState(getPositions=True)
    pos_nm = np.asarray(
        [[p.value_in_unit(unit.nanometer) for p in pr]
         for pr in state.getPositions()],
        dtype=float,
    )
    # Rewrite back into the caller's positions list (Quantity[Vec3]).
    for i in range(pos_nm.shape[0]):
        positions[i] = (pos_nm[i, 0], pos_nm[i, 1], pos_nm[i, 2]) * unit.nanometer
    del ctx


if __name__ == "__main__":
    main()