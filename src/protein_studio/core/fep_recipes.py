"""FEP recipe presets — predefined parameter combinations.

Recipes bundle the typical knobs (lambda count, equilibration /
production steps, sampling stride, neighbor re-eval, position
restraint k) into named presets so the GUI / MCP don't have to
expose every parameter.

Three shipped presets:

- ``quick_test``       — fast smoke test (5 windows × 200 + 500 steps).
                        Use to verify the pipeline before launching a
                        long production run.
- ``production_bar``   — production-quality BAR (11 windows × 2k +
                        10k steps, mixed-state re-eval enabled).
- ``benchmark_*``      — standard benchmarks used in our test suite.

Recipes are plain dicts so they JSON-serialize cleanly for the MCP
protocol and round-trip through the GUI's preset selector.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _recipe(
    name: str,
    n_lambdas: int,
    n_equilibration_steps: int,
    n_production_steps: int,
    sample_every: int,
    sample_neighbor_lambdas: bool,
    restraint_k_kJmol_nm2: float,
    description: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "n_lambdas": n_lambdas,
        "n_equilibration_steps": n_equilibration_steps,
        "n_production_steps": n_production_steps,
        "sample_every": sample_every,
        "sample_neighbor_lambdas": sample_neighbor_lambdas,
        "restraint_k_kJmol_nm2": restraint_k_kJmol_nm2,
        "description": description,
    }


RECIPES: Dict[str, Dict[str, Any]] = {
    "quick_test": _recipe(
        name="quick_test",
        n_lambdas=5,
        n_equilibration_steps=200,
        n_production_steps=500,
        sample_every=50,
        sample_neighbor_lambdas=True,
        restraint_k_kJmol_nm2=50000.0,
        description=(
            "Quick smoke test. 5 λ windows, 200 equilibration + 500 "
            "production steps each, mixed-state re-evaluation enabled. "
            "Use this to verify the pipeline end-to-end before launching "
            "a long production run."
        ),
    ),
    "production_bar": _recipe(
        name="production_bar",
        n_lambdas=11,
        n_equilibration_steps=2000,
        n_production_steps=10000,
        sample_every=50,
        sample_neighbor_lambdas=True,
        restraint_k_kJmol_nm2=50000.0,
        description=(
            "Production-quality BAR. 11 λ windows, 2k equilibration + "
            "10k production steps each, mixed-state re-evaluation enabled. "
            "Matches the parameter regime used in published small-molecule "
            "hydration studies."
        ),
    ),
    "benchmark_mutation": _recipe(
        name="benchmark_mutation",
        n_lambdas=11,
        n_equilibration_steps=1000,
        n_production_steps=2000,
        sample_every=50,
        sample_neighbor_lambdas=True,
        restraint_k_kJmol_nm2=50000.0,
        description=(
            "Standard benchmark for sidechain-truncation FEP "
            "(e.g. Lys6→Ala on ubiquitin). 11 λ windows, 1k + 2k "
            "steps, mixed-state re-evaluation."
        ),
    ),
    "benchmark_ligand": _recipe(
        name="benchmark_ligand",
        n_lambdas=11,
        n_equilibration_steps=1000,
        n_production_steps=2000,
        sample_every=50,
        sample_neighbor_lambdas=True,
        restraint_k_kJmol_nm2=50000.0,
        description=(
            "Standard benchmark for ligand decouple FEP "
            "(double decoupling for absolute ΔG_bind). Same step "
            "counts as benchmark_mutation."
        ),
    ),
}


def get_recipe(name: str) -> Dict[str, Any]:
    """Return recipe by name. Falls back to ``quick_test`` if unknown.

    A missing or unknown recipe is treated as a request for the
    smoke-test preset rather than raising — the GUI's preset
    selector should always have *some* sensible parameters to feed
    into the FEP runner.
    """
    return RECIPES.get(name, RECIPES["quick_test"])


def list_recipes() -> List[str]:
    """Return list of available recipe names."""
    return list(RECIPES.keys())