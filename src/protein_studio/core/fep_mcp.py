"""FEP wrappers exposed as MCP tools.

Three thin functions on top of ``core.fep``:

- ``fep_build_system``  — build alchemical system from a PDB
  (returns metadata only; no MD).
- ``fep_configure_recipe`` — return a recipe preset by name.
- ``fep_run_sweep``     — full λ-sweep + BAR ΔG for a PDB.

OpenMM is heavy (~5s import on a cold cache) so we import
``core.fep`` lazily inside each function — that keeps the MCP
server fast on startup and avoids paying the OpenMM cost for
clients that only call non-FEP tools.

All three return plain dicts (JSON-serialisable) so they round-trip
cleanly through the MCP protocol and the GUI's status displays.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .fep_recipes import RECIPES, get_recipe, list_recipes


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def fep_build_system(
    pdb_path: str,
    mutation: Optional[Union[str, int]] = None,
    ligand_resname: Optional[str] = None,
    ph: float = 7.0,
    padding_nm: float = 1.0,
    ionic_strength_M: float = 0.15,
    temperature_K: float = 300.0,
    restraint_k_kJmol_nm2: float = 50000.0,
) -> Dict[str, Any]:
    """Build an alchemical system from a PDB file.

    Args:
        pdb_path: Absolute path to a PDB file on disk.
        mutation: Residue identifier for sidechain truncation FEP.
            Accepts either an integer (0-based OpenMM topology index)
            or a string like ``"Lys6"`` / ``"LYS6"``.
            Mutually exclusive with ``ligand_resname``.
        ligand_resname: Residue name for ligand decouple FEP
            (e.g. ``"LIG"``). Mutually exclusive with ``mutation``.
        ph: pH for protonation states (default 7.0). Currently
            informational only — the AMBER ff99SB-ildn force field
            uses fixed protonation.
        padding_nm: Solvent padding around the protein in nm.
        ionic_strength_M: NaCl concentration for charge screening.
        temperature_K: Simulation temperature.
        restraint_k_kJmol_nm2: Force constant for position restraints
            on disappearing atoms.

    Returns:
        Dict with metadata about the built system:
        ``status``, ``mode``, ``n_particles``, ``n_ligand_atoms``,
        ``lambda_id``, ``config``, and either ``error`` + ``traceback``
        on failure.
    """
    p = Path(pdb_path)
    if not p.exists():
        return {"status": "error",
                "error": f"PDB file not found: {p}"}

    if mutation is not None and ligand_resname is not None:
        return {"status": "error",
                "error": "Specify only one of `mutation` or `ligand_resname`"}
    if mutation is None and ligand_resname is None:
        return {"status": "error",
                "error": "Specify one of `mutation` or `ligand_resname`"}

    try:
        from . import fep as fep_core

        if mutation is not None:
            # Accept "Lys6", "LYS6", or int.
            if isinstance(mutation, str):
                residue_idx = _parse_residue_id(mutation, str(p))
            else:
                residue_idx = int(mutation)
            system, topology, positions, integrator, disappearing = (
                fep_core.build_amber_mutation_system(
                    str(p),
                    mutation_residue=residue_idx,
                    padding_nm=padding_nm,
                    ionic_strength_M=ionic_strength_M,
                    temperature_K=temperature_K,
                    restraint_k_kJmol_nm2=restraint_k_kJmol_nm2,
                )
            )
            return {
                "status": "ok",
                "mode": "mutation",
                "n_particles": system.getNumParticles(),
                "n_waters": _count_waters(topology),
                "n_disappearing": len(disappearing),
                "n_atoms_total": topology.getNumAtoms(),
                "lambda_id": "λ ∈ [0, 1]",
                "config": {
                    "pdb": str(p),
                    "mutation": str(mutation),
                    "padding_nm": padding_nm,
                    "ionic_strength_M": ionic_strength_M,
                    "temperature_K": temperature_K,
                    "restraint_k_kJmol_nm2": restraint_k_kJmol_nm2,
                },
            }

        # ligand_decouple mode
        system, topology, positions, integrator, ligand_indices = (
            fep_core.build_ligand_decouple_system(
                str(p),
                ligand_resnames=[ligand_resname],
                padding_nm=padding_nm,
                ionic_strength_M=ionic_strength_M,
                temperature_K=temperature_K,
            )
        )
        return {
            "status": "ok",
            "mode": "ligand_decouple",
            "n_particles": system.getNumParticles(),
            "n_waters": _count_waters(topology),
            "n_ligand_atoms": len(ligand_indices),
            "n_atoms_total": topology.getNumAtoms(),
            "lambda_id": "λ ∈ [0, 1]",
            "config": {
                "pdb": str(p),
                "ligand_resname": ligand_resname,
                "padding_nm": padding_nm,
                "ionic_strength_M": ionic_strength_M,
                "temperature_K": temperature_K,
            },
        }
    except Exception as e:
        return {"status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def _parse_residue_id(spec: str, pdb_path: str) -> int:
    """Convert a string residue spec to a 0-based OpenMM index.

    Accepts ``"Lys6"``, ``"LYS6"``, ``"6"`` — anything where the
    trailing digits form a 1-based residue number.

    Falls back to 0 if parsing fails (the FEP build will raise a
    clearer error downstream).
    """
    import re
    m = re.match(r"^([A-Za-z]+)?(\d+)$", spec.strip())
    if not m:
        return 0
    return int(m.group(2)) - 1


def _count_waters(topology) -> int:
    """Count waters in an OpenMM Topology by residue name."""
    count = 0
    for res in topology.residues():
        if res.name in ("HOH", "WAT", "SOL", "TIP3"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------

def fep_configure_recipe(name: str) -> Dict[str, Any]:
    """Return the named recipe's full parameter set.

    Args:
        name: One of the recipe names from ``list_recipes()``.

    Returns:
        Dict with ``status`` and ``recipe`` (the full parameter
        dict). On unknown name, ``status="error"`` with the list
        of available names.
    """
    if name not in RECIPES:
        return {"status": "error",
                "error": f"Unknown recipe: {name}",
                "available": list_recipes()}
    return {"status": "ok", "recipe": RECIPES[name]}


def fep_list_recipes() -> Dict[str, Any]:
    """Return all available recipes as a flat dict.

    Convenience tool for clients that want the full recipe library
    in one call rather than calling ``fep_configure_recipe`` for
    each name.
    """
    return {"status": "ok",
            "recipes": list_recipes(),
            "details": RECIPES}


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

def fep_run_sweep(
    pdb_path: str,
    mode: str,
    recipe_name: str = "quick_test",
    mutation: Optional[Union[str, int]] = None,
    ligand_resname: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a full λ-sweep and return BAR ΔG.

    Args:
        pdb_path: Absolute path to a PDB file on disk.
        mode: Either ``"mutation"`` or ``"ligand_decouple"``.
        recipe_name: Recipe preset name (see ``fep_list_recipes()``).
        mutation: For mutation mode, the residue identifier
            (``"Lys6"`` style or integer index).
        ligand_resname: For ligand_decouple mode, the residue name.
        progress_callback: Optional callable ``(lambda_idx,
            n_lambdas, lam_value, n_samples_so_far) -> None``. Used
            by the GUI to update the progress bar.

    Returns:
        Dict with ``status``, ``mode``, ``recipe``, ``delta_g_kJmol``,
        ``delta_g_kcal_mol``, ``sigma_kJmol``, ``n_windows``,
        ``per_window`` list of ``(idx, dg, sigma)`` tuples, and
        ``energy_summary`` (mean / n_samples per window). On failure
        ``status="error"`` + ``error`` + ``traceback``.
    """
    p = Path(pdb_path)
    if not p.exists():
        return {"status": "error",
                "error": f"PDB file not found: {p}"}
    if mode not in ("mutation", "ligand_decouple"):
        return {"status": "error",
                "error": f"Unknown mode: {mode}",
                "available": ["mutation", "ligand_decouple"]}

    recipe = get_recipe(recipe_name)
    if mode == "mutation" and mutation is None:
        return {"status": "error",
                "error": "mutation mode requires `mutation` arg"}
    if mode == "ligand_decouple" and ligand_resname is None:
        return {"status": "error",
                "error": "ligand_decouple mode requires `ligand_resname` arg"}

    try:
        from . import fep as fep_core

        common_kwargs = dict(
            n_equilibration_steps=recipe["n_equilibration_steps"],
            n_production_steps=recipe["n_production_steps"],
            sample_every=recipe["sample_every"],
            sample_neighbor_lambdas=recipe["sample_neighbor_lambdas"],
            restraint_k_kJmol_nm2=recipe["restraint_k_kJmol_nm2"],
        )

        if mode == "mutation":
            if isinstance(mutation, str):
                residue_idx = _parse_residue_id(mutation, str(p))
            else:
                residue_idx = int(mutation)
            energies_by_lambda, neighbor_energies, disappearing = (
                fep_core.run_amber_mutation_fep(
                    str(p),
                    mutation_residue=residue_idx,
                    **common_kwargs,
                )
            )
        else:
            energies_by_lambda, neighbor_energies, disappearing = (
                fep_core.run_ligand_decouple_fep(
                    str(p),
                    ligand_resnames=[ligand_resname],
                    **common_kwargs,
                )
            )

        # Drive a BAR analysis on the collected energies.
        lambdas = sorted(energies_by_lambda.keys())
        kT_kjmol = 8.314462618e-3 * 300.0
        total_dg, sigma, per_window = fep_core.analyze_fep_bar(
            energies_by_lambda, neighbor_energies, lambdas,
            kT_kjmol=kT_kjmol,
        )

        energy_summary = [
            {
                "lambda": lam,
                "n_samples": len(energies_by_lambda[lam]),
                "mean_kJmol": (
                    sum(energies_by_lambda[lam]) / len(energies_by_lambda[lam])
                    if energies_by_lambda[lam] else None
                ),
            }
            for lam in lambdas
        ]

        return {
            "status": "ok",
            "mode": mode,
            "recipe": recipe_name,
            "delta_g_kJmol": total_dg,
            "delta_g_kcal_mol": total_dg / 4.184,
            "sigma_kJmol": sigma,
            "n_windows": len(lambdas),
            "lambdas": lambdas,
            "per_window": [
                {"idx": idx, "lambda": lambdas[idx],
                 "delta_g_kJmol": dg, "sigma_kJmol": sig}
                for idx, dg, sig in per_window
            ],
            "energy_summary": energy_summary,
        }
    except Exception as e:
        return {"status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}