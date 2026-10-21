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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .fep_checkpoint import (
    FEPCheckpoint,
    checkpoint_status,
    default_checkpoint_path,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .fep_recipes import RECIPES, get_recipe, list_recipes
from .fep_reference import (
    REFERENCE_LIBRARY,
    list_references,
    list_runnable_references,
    find_reference,
    compare_to_reference,
    best_reference_for,
)


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
            minimize_iterations=recipe.get(
                "minimize_iterations", 100
            ),
            minimize_tolerance_kJmol=recipe.get(
                "minimize_tolerance_kJmol", 10.0
            ),
            padding_nm=0.8,  # retry will bump this on NaN
        )

        # Auto-retry: ``run_amber_mutation_fep`` can blow up to
        # NaN when OpenMM's ``addSolvent`` produces overlapping
        # waters (occasionally 1 in 5 runs for tight padding).
        # Each retry re-solvates with a 0.3 nm larger padding so
        # the water placement has more room.  We give up after 3
        # attempts — past that the user really does need an
        # external prep tool (AmberTools tleap / GROMACS grompp).
        max_padding_retries = 3
        attempt = 0
        energies_by_lambda = neighbor_energies = disappearing = None
        last_exc = None
        while attempt < max_padding_retries:
            try:
                if mode == "mutation":
                    if isinstance(mutation, str):
                        residue_idx = _parse_residue_id(mutation, str(p))
                    else:
                        residue_idx = int(mutation)
                    (energies_by_lambda,
                     neighbor_energies,
                     disappearing) = fep_core.run_amber_mutation_fep(
                        str(p),
                        mutation_residue=residue_idx,
                        **common_kwargs,
                    )
                else:
                    (energies_by_lambda,
                     neighbor_energies,
                     disappearing) = fep_core.run_ligand_decouple_fep(
                        str(p),
                        ligand_resnames=[ligand_resname],
                        **common_kwargs,
                    )
                break  # success
            except Exception as e:
                last_exc = e
                err = f"{type(e).__name__}: {e}"
                # Only retry on integrator-NaN style errors; let
                # other errors (file not found, bad residue index,
                # ...) surface immediately.
                is_nan = "NaN" in err or "nan" in err.lower()
                if not is_nan:
                    raise
                attempt += 1
                if attempt >= max_padding_retries:
                    raise
                # Retry with 0.3 nm more padding.
                common_kwargs["padding_nm"] = (
                    common_kwargs["padding_nm"] + 0.3
                )
                # Also wipe any cached checkpoint from a previous
                # attempt so we start fresh.
                continue

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
            "energies_by_lambda": {
                str(lam): list(map(float, vals))
                for lam, vals in energies_by_lambda.items()
            },
            "padding_attempts": attempt + 1,
            "final_padding_nm": common_kwargs["padding_nm"],
        }
    except Exception as e:
        return {"status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}

# ---------------------------------------------------------------------------
# Checkpointed sweep
# ---------------------------------------------------------------------------

def fep_run_sweep_with_checkpoint(
    pdb_path: str,
    mode: str,
    recipe_name: str = "quick_test",
    mutation: Optional[Union[str, int]] = None,
    ligand_resname: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable] = None,
    position_snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a full λ-sweep with optional checkpoint save + resume.

    On a clean run:
        - checkpoint_path is created and updated after each λ window.
        - If the user cancels (cancellation_check returns True), the
          partial results are saved and the result dict has
          ``status="interrupted"``.

    On resume (checkpoint_path already exists with matching inputs):
        - Already-completed λ windows are skipped.
        - The sweep continues from the first incomplete λ.

    Args:
        pdb_path: Absolute path to a PDB file.
        mode: ``"mutation"`` or ``"ligand_decouple"``.
        recipe_name: Recipe preset name.
        mutation: For mutation mode, the residue id (``"Lys6"``).
        ligand_resname: For ligand mode, the residue name.
        checkpoint_path: Where to write/read the pickle. Auto-derived
            from inputs if not given.
        cancellation_check: Callable returning True when the user
            has requested cancellation. Checked once per λ window.
        progress_callback: Callable invoked once per λ window with
            ``(lambda_idx, n_lambdas, lam_value, n_samples)``.

    Returns:
        Dict with status, mode, recipe, delta_g_kJmol, sigma_kJmol,
        n_windows, per_window, energy_summary, energies_by_lambda,
        and (when applicable) ``checkpoint_path`` / ``interrupted_at``.
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

    # Default checkpoint path derived from inputs.
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint_path(
            pdb_path, mode, mutation, ligand_resname, recipe_name,
        )

    existing = load_checkpoint(checkpoint_path)
    if existing is not None:
        # Verify the existing checkpoint matches this run's inputs.
        if (existing.pdb_path != pdb_path
                or existing.mode != mode
                or existing.recipe_name != recipe_name
                or existing.mutation != mutation
                or existing.ligand_resname != ligand_resname):
            return {
                "status": "error",
                "error": (
                    "Checkpoint exists at "
                    f"{checkpoint_path} but with different inputs. "
                    "Pass a new checkpoint_path to start fresh, or "
                    "delete the existing one."
                ),
                "existing": {
                    "pdb_path": existing.pdb_path,
                    "mode": existing.mode,
                    "recipe_name": existing.recipe_name,
                    "mutation": existing.mutation,
                    "ligand_resname": existing.ligand_resname,
                },
            }
        resume = True
        all_lambdas = list(existing.lambdas)
        remaining = [l for l in all_lambdas
                     if l not in existing.completed_lambdas]
    else:
        resume = False
        remaining = None  # default 11 windows
        all_lambdas = None

    try:
        from . import fep as fep_core

        common_kwargs = dict(
            n_equilibration_steps=recipe["n_equilibration_steps"],
            n_production_steps=recipe["n_production_steps"],
            sample_every=recipe["sample_every"],
            sample_neighbor_lambdas=recipe["sample_neighbor_lambdas"],
            restraint_k_kJmol_nm2=recipe["restraint_k_kJmol_nm2"],
            minimize_iterations=recipe.get(
                "minimize_iterations", 100
            ),
            minimize_tolerance_kJmol=recipe.get(
                "minimize_tolerance_kJmol", 10.0
            ),
            padding_nm=0.8,  # retry will bump on NaN
        )

        if mode == "mutation":
            residue_idx = (
                _parse_residue_id(mutation, str(p))
                if isinstance(mutation, str) else int(mutation)
            )
        else:
            residue_idx = None

        # Auto-retry on integrator NaN.  Each attempt re-solvates
        # with 0.3 nm more padding so OpenMM's ``addSolvent`` has
        # more room for water placement.  See notes in
        # ``fep_run_sweep``.
        max_padding_retries = 3
        attempt = 0
        energies = neighbors = disappearing = boresch_refs = None
        last_exc: Exception | None = None
        while attempt < max_padding_retries:
            try:
                run_kwargs = dict(common_kwargs)
                if remaining is not None:
                    run_kwargs["lambdas"] = remaining
                if position_snapshot_dir is not None:
                    run_kwargs["position_snapshot_dir"] = (
                        position_snapshot_dir
                    )
                if mode == "mutation":
                    (energies, neighbors,
                     disappearing) = fep_core.run_amber_mutation_fep(
                        str(p),
                        mutation_residue=residue_idx,
                        cancellation_check=cancellation_check,
                        window_callback=_make_window_callback(
                            checkpoint_path, pdb_path, mode, mutation,
                            ligand_resname, recipe_name,
                        ),
                        **run_kwargs,
                    )
                    boresch_refs = None
                else:
                    (energies, neighbors, disappearing,
                     boresch_refs) = fep_core.run_ligand_decouple_fep(
                        str(p),
                        ligand_resnames=[ligand_resname],
                        cancellation_check=cancellation_check,
                        window_callback=_make_window_callback(
                            checkpoint_path, pdb_path, mode, mutation,
                            ligand_resname, recipe_name,
                        ),
                        **run_kwargs,
                    )
                break
            except Exception as e:
                last_exc = e
                err = f"{type(e).__name__}: {e}"
                is_nan = "NaN" in err or "nan" in err.lower()
                if not is_nan:
                    raise
                attempt += 1
                if attempt >= max_padding_retries:
                    raise
                common_kwargs["padding_nm"] += 0.3
                continue

        # Merge with existing partial energies (resume case).
        if resume:
            for lam, vals in energies.items():
                existing.energies_by_lambda.setdefault(lam, []).extend(vals)
            for lam, vals in neighbors.items():
                existing.neighbor_energies.setdefault(lam, []).extend(vals)
            energies_by_lambda = existing.energies_by_lambda
            neighbor_energies = existing.neighbor_energies
            disappearing = existing.disappearing or disappearing
            all_lambdas = existing.lambdas
        else:
            energies_by_lambda = energies
            neighbor_energies = neighbors
            all_lambdas = sorted(energies_by_lambda.keys())

        # Drive BAR analysis on the merged energies.
        kT_kjmol = 8.314462618e-3 * 300.0
        total_dg, sigma, per_window = fep_core.analyze_fep_bar(
            energies_by_lambda, neighbor_energies, all_lambdas,
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
            for lam in all_lambdas
        ]

        # Clean up the checkpoint on a fully-successful sweep.
        try:
            delete_checkpoint(checkpoint_path)
        except Exception:
            pass

        return {
            "status": "ok",
            "mode": mode,
            "recipe": recipe_name,
            "resumed_from_checkpoint": resume,
            "delta_g_kJmol": total_dg,
            "delta_g_kcal_mol": total_dg / 4.184,
            "sigma_kJmol": sigma,
            "n_windows": len(all_lambdas),
            "lambdas": list(all_lambdas),
            "per_window": [
                {"idx": idx, "lambda": all_lambdas[idx],
                 "delta_g_kJmol": dg, "sigma_kJmol": sig}
                for idx, dg, sig in per_window
            ],
            "energy_summary": energy_summary,
            "energies_by_lambda": {
                str(lam): list(map(float, vals))
                for lam, vals in energies_by_lambda.items()
            },
        }
    except Exception as e:
        # Catches both FEPInterrupted (cancellation) and unexpected errors.
        # On cancellation we still want to surface the partial energies
        # so the user can analyse what was done.
        from . import fep as fep_core
        if isinstance(e, fep_core.FEPInterrupted):
            # The window_callback already wrote the latest checkpoint.
            # Surface the partial state for diagnostics.
            cp = load_checkpoint(checkpoint_path) if checkpoint_path else None
            return {
                "status": "interrupted",
                "mode": mode,
                "recipe": recipe_name,
                "checkpoint_path": checkpoint_path,
                "n_completed": (
                    len(cp.completed_lambdas) if cp else 0
                ),
                "completed_lambdas": (
                    cp.completed_lambdas if cp else []
                ),
                "remaining_lambdas": (
                    cp and [l for l in cp.lambdas
                            if l not in cp.completed_lambdas] or []
                ),
                "message": str(e),
            }
        return {"status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def _make_window_callback(
    checkpoint_path: Optional[str],
    pdb_path: str,
    mode: str,
    mutation: Optional[Union[str, int]],
    ligand_resname: Optional[str],
    recipe_name: str,
) -> Optional[Callable]:
    """Build a window_callback that writes the checkpoint after each λ.

    Returns None if no checkpoint_path was supplied (in which case
    there is nothing for the per-window hook to do).
    """
    if checkpoint_path is None:
        return None

    def _cb(energies, neighbors, completed):
        if checkpoint_path:
            cp = FEPCheckpoint(
                pdb_path=pdb_path,
                mode=mode,
                mutation=str(mutation) if mutation is not None else None,
                ligand_resname=ligand_resname,
                recipe_name=recipe_name,
                lambdas=list(energies.keys()),
                completed_lambdas=list(completed),
                energies_by_lambda={
                    float(k): list(v) for k, v in energies.items()
                },
                neighbor_energies={
                    float(k): list(v) for k, v in neighbors.items()
                },
                disappearing=[],
            )
            save_checkpoint(cp, checkpoint_path)

    return _cb


# ---------------------------------------------------------------------------
# Resume / status / delete
# ---------------------------------------------------------------------------

def fep_resume_sweep(
    checkpoint_path: str,
    cancellation_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Resume a sweep from ``checkpoint_path``.

    Reads the existing checkpoint and runs the remaining windows only.
    On completion, returns the same shape as :func:`fep_run_sweep`.
    """
    cp = load_checkpoint(checkpoint_path)
    if cp is None:
        return {"status": "error",
                "error": f"No checkpoint at {checkpoint_path}"}
    return fep_run_sweep_with_checkpoint(
        pdb_path=cp.pdb_path,
        mode=cp.mode,
        recipe_name=cp.recipe_name,
        mutation=cp.mutation,
        ligand_resname=cp.ligand_resname,
        checkpoint_path=checkpoint_path,
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )


def fep_checkpoint_info(path: str) -> Dict[str, Any]:
    """Inspect a checkpoint file.

    Returns the same shape as :func:`checkpoint_status`. Useful for
    MCP clients and GUI dialogs that want to show "you have 5 of 11
    windows done" before resuming.
    """
    return checkpoint_status(path)


def fep_checkpoint_delete(path: str) -> Dict[str, Any]:
    """Delete the checkpoint at ``path``. Returns whether it existed."""
    deleted = delete_checkpoint(path)
    return {"status": "ok" if deleted else "missing",
            "deleted": deleted,
            "path": path}


# ---------------------------------------------------------------------------
# Per-window TI helper (used by the Analyze tab)
# ---------------------------------------------------------------------------

def ti_per_window(
    energies_by_lambda: Dict[Any, List[float]],
    lambdas: List[float],
) -> List[float]:
    """Trapezoidal per-window ΔG contribution.

    Returns a list of len(lambdas) values; the first and last
    entries are zero (no neighbour on one side). The remaining
    entries are the negative of half the trapezoidal area between
    consecutive λ values.

    Diagnostic only — proper total ΔG comes from
    :func:`core.fep.analyze_fep_trapezoidal`.
    """
    if not lambdas:
        return []
    n = len(lambdas)
    out = [0.0] * n
    for i in range(1, n):
        lam_lo, lam_hi = lambdas[i - 1], lambdas[i]
        e_lo = energies_by_lambda.get(lam_lo, [])
        e_hi = energies_by_lambda.get(lam_hi, [])
        if not e_lo or not e_hi:
            continue
        mean_lo = sum(e_lo) / len(e_lo)
        mean_hi = sum(e_hi) / len(e_hi)
        out[i] = -0.5 * (lam_hi - lam_lo) * (mean_hi + mean_lo)
    return out


# ---------------------------------------------------------------------------
# Reference library + benchmark comparison
# ---------------------------------------------------------------------------

def fep_list_references(runnable_only: bool = False) -> Dict[str, Any]:
    """List reference ΔG benchmarks.

    Args:
        runnable_only: If True, skip entries that need additional
            parameterisation (e.g. Bosutinib needs antechamber).

    Returns:
        Dict with ``status``, ``n_references``, and ``references``
        (list of reference dicts).
    """
    refs = list_runnable_references() if runnable_only else list_references()
    return {
        "status": "ok",
        "n_references": len(refs),
        "references": refs,
    }


def fep_compare_to_reference(
    measured_dG_kJmol: float,
    measured_sigma_kJmol: float,
    reference_name: str,
) -> Dict[str, Any]:
    """Compare a measured ΔG against a named reference value.

    Args:
        measured_dG_kJmol: From your finished FEP sweep.
        measured_sigma_kJmol: From BAR (or TI propagated SEM).
        reference_name: One of the names from ``fep_list_references()``.

    Returns:
        Dict with agreement, z_score, deviation, combined
        uncertainty, and the full reference entry. On unknown
        reference name, status="unknown_reference" with the list
        of available names.
    """
    return compare_to_reference(
        measured_dG_kJmol=measured_dG_kJmol,
        measured_sigma_kJmol=measured_sigma_kJmol,
        reference_name=reference_name,
    )


def fep_compare_sweep_to_reference(
    sweep_result: Dict[str, Any],
    reference_name: str,
) -> Dict[str, Any]:
    """Compare a finished sweep result against a reference.

    Convenience wrapper around ``fep_compare_to_reference`` that
    pulls ``delta_g_kJmol`` and ``sigma_kJmol`` from the sweep
    result dict (the same shape ``fep_run_sweep`` returns).
    """
    dg = sweep_result.get("delta_g_kJmol")
    sig = sweep_result.get("sigma_kJmol")
    if dg is None or sig is None:
        return {"status": "error",
                "error": "sweep_result has no delta_g_kJmol / sigma_kJmol"}
    return compare_to_reference(
        measured_dG_kJmol=dg,
        measured_sigma_kJmol=sig,
        reference_name=reference_name,
    )


def fep_best_reference(
    mode: str,
    target: Optional[str] = None,
    pdb_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick the most likely matching reference for a sweep setup.

    Useful for auto-defaulting the GUI's reference dropdown.
    """
    ref = best_reference_for(mode=mode, target=target, pdb_file=pdb_file)
    if ref is None:
        return {"status": "no_match",
                "mode": mode,
                "target": target,
                "pdb_file": pdb_file}
    return {"status": "ok", "reference": ref.to_dict()}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def fep_get_disappearing_atoms(
    pdb_path: str,
    mode: str,
    mutation: Optional[Union[str, int]] = None,
    ligand_resname: Optional[str] = None,
    output_pdb: Optional[str] = None,
) -> Dict[str, Any]:
    """Identify the atoms being alchemically decoupled.

    Builds the system (no MD) and returns the disappearing-atom
    indices plus, optionally, a filtered PDB containing just those
    atoms. The GUI uses the filtered PDB to load a separate
    SceneObject (typically ball+stick) that highlights which atoms
    the sweep is removing.

    Args:
        pdb_path: Absolute path to a PDB file on disk.
        mode: ``"mutation"`` or ``"ligand_decouple"``.
        mutation: For mutation mode, the residue id (``"Lys6"``).
        ligand_resname: For ligand mode, the residue name.
        output_pdb: If given, write a filtered PDB containing only
            the disappearing atoms to this path. The GUI passes
            a path under its temp dir so the load is automatic.

    Returns:
        Dict with ``status``, ``mode``, ``n_disappearing``,
        ``atom_indices`` (list of int), ``residue_summary`` (one
        entry per residue that disappears), and ``output_pdb``
        if written.
    """
    p = Path(pdb_path)
    if not p.exists():
        return {"status": "error",
                "error": f"PDB file not found: {p}"}
    if mode not in ("mutation", "ligand_decouple"):
        return {"status": "error",
                "error": f"Unknown mode: {mode}"}

    try:
        from . import fep as fep_core

        if mode == "mutation":
            if mutation is None:
                return {"status": "error",
                        "error": "mutation mode requires `mutation` arg"}
            residue_idx = (
                _parse_residue_id(mutation, str(p))
                if isinstance(mutation, str) else int(mutation)
            )
            system, topology, _, _, disappearing = (
                fep_core.build_amber_mutation_system(
                    str(p), mutation_residue=residue_idx,
                )
            )
        else:
            if ligand_resname is None:
                return {"status": "error",
                        "error": "ligand_decouple mode requires `ligand_resname` arg"}
            system, topology, _, _, disappearing = (
                fep_core.build_ligand_decouple_system(
                    str(p), ligand_resnames=[ligand_resname],
                )
            )

        # Build a residue-level summary so the GUI can show
        # "this is what disappears" without parsing PDB records.
        by_residue: Dict[str, List[int]] = {}
        for atom_idx in disappearing:
            try:
                atom = topology.atom(atom_idx)
                resname = atom.residue.name
                resseq = atom.residue.id
                key = f"{resname}{resseq}"
                by_residue.setdefault(key, []).append(int(atom_idx))
            except Exception:
                continue

        residue_summary = [
            {"residue": k, "atom_indices": sorted(v)}
            for k, v in sorted(by_residue.items())
        ]

        result = {
            "status": "ok",
            "mode": mode,
            "n_disappearing": len(disappearing),
            "atom_indices": [int(i) for i in disappearing],
            "residue_summary": residue_summary,
        }

        if output_pdb:
            out = Path(output_pdb)
            out.parent.mkdir(parents=True, exist_ok=True)
            _write_subset_pdb(p, topology, disappearing, str(out))
            result["output_pdb"] = str(out)

        return result
    except Exception as e:
        return {"status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def _write_subset_pdb(
    src_pdb_path: str,
    topology,
    keep_indices,
    out_path: str,
) -> None:
    """Write a new PDB containing only the atoms at ``keep_indices``.

    Uses OpenMM's PDBFile infrastructure so the output has the same
    atom numbering scheme (serial numbers, chain IDs, residue
    records, CRYST1) as the source topology.
    """
    import openmm
    n_atoms = topology.getNumAtoms() if topology is not None else None
    keep_set = set(int(i) for i in keep_indices)

    # Read source PDB lines and filter.
    src_lines = open(src_pdb_path).readlines()
    out_lines = []
    serial_map = {}  # old serial → new serial
    new_serial = 1
    for line in src_lines:
        rec = line[:6]
        if rec in ("ATOM  ", "HETATM"):
            try:
                serial = int(line[6:11].strip())
            except ValueError:
                continue
            # Map source serial to atom index in topology: serial is
            # 1-based, atom index is 0-based.
            atom_idx = serial - 1
            if n_atoms is not None and not (
                0 <= atom_idx < n_atoms
            ):
                continue
            if atom_idx not in keep_set:
                continue
            if atom_idx not in serial_map:
                serial_map[atom_idx] = new_serial
                new_serial += 1
            new_s = serial_map[atom_idx]
            # Replace the serial in the line.
            new_line = (
                line[:6]
                + f"{new_s:>5d}"
                + line[11:]
            )
            out_lines.append(new_line)
        elif rec.strip() in (
            "CRYST1", "HEADER", "TITLE", "COMPND", "SOURCE",
            "KEYWDS", "EXPDTA", "AUTHOR", "REVDAT", "REMARK",
            "DBREF", "SEQRES", "FORMUL", "HET", "HETNAM",
            "HETSYN", "FORMUL", "SITE", "END", "MODEL",
            "ENDMDL", "TER", "ANISOU", "SIGUIJ", "TER",
            "MASTER", "END",
        ) or rec.startswith("CRYST"):
            out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)


def fep_capture_lambda_snapshots(
    pdb_path: str,
    mode: str,
    recipe_name: str = "quick_test",
    mutation: Optional[Union[str, int]] = None,
    ligand_resname: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a sweep and save one PDB per λ window.

    The output directory contains ``lambda_<value>.pdb`` files,
    one per completed λ window, that can be loaded as separate
    scene objects to visualise how the disappearing atoms decay
    across the sweep.

    Args:
        pdb_path: PDB file.
        mode: ``"mutation"`` or ``"ligand_decouple"``.
        recipe_name: Recipe preset.
        mutation / ligand_resname: target.
        output_dir: Where to write the per-λ PDBs. Defaults to
            ``<pdb_dir>/fep_snapshots/<pdb_stem>_<mode>_<target>``.

    Returns:
        Dict with ``status``, ``output_dir``, ``snapshots`` (list
        of dicts with ``lambda`` and ``pdb_path``), and the
        regular BAR analysis fields. On cancellation,
        ``status="interrupted"`` and only the completed snapshots
        are listed.
    """
    p = Path(pdb_path)
    if not p.exists():
        return {"status": "error",
                "error": f"PDB file not found: {p}"}

    if output_dir is None:
        target = mutation if mutation else ligand_resname or "x"
        safe_target = str(target).replace("/", "_").replace("\\", "_")
        output_dir = str(
            p.parent / "fep_snapshots" / f"{p.stem}_{mode}_{safe_target}"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Delegate to the checkpointed sweep — same cancellation /
    # resume semantics, with position_snapshot_dir plumbed through
    # so the core runner writes one PDB per completed λ window.
    result = fep_run_sweep_with_checkpoint(
        pdb_path=pdb_path,
        mode=mode,
        recipe_name=recipe_name,
        mutation=mutation,
        ligand_resname=ligand_resname,
        position_snapshot_dir=output_dir,
    )

    # Build the snapshots manifest from the output dir (it may be
    # partial if the sweep was interrupted).
    snapshots = []
    snap_path = Path(output_dir)
    if snap_path.exists():
        for f in sorted(snap_path.glob("lambda_*.pdb")):
            try:
                lam_str = f.stem.replace("lambda_", "")
                lam = float(lam_str)
            except ValueError:
                continue
            snapshots.append({"lambda": lam, "pdb_path": str(f)})

    return {
        **result,
        "snapshot_dir": output_dir,
        "snapshots": snapshots,
    }
