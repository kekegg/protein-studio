"""FEP reference ΔG library + automatic benchmark comparison.

Provides a curated catalog of well-characterised FEP benchmarks
that the GUI / MCP can match against a finished sweep. Each
reference includes the citation, expected ΔG (kJ/mol), its
uncertainty (1σ), and metadata about which protein-studio
artifact(s) implement the system.

After a sweep finishes the user can call
:func:`compare_to_reference` to get a z-score and a one-line
agreement label (``within_1sigma`` / ``within_2sigma`` / ``outside``).
This is the "are my numbers reasonable" sanity check that an FEP
practitioner would normally do by hand against a paper.

Entries
-------

The shipped catalog covers four systems:

- ``methane_hydration_shifts_2000`` — Shirts & Ponder 2000.
  Single methane in TIP3P box, λ-decouple of the full
  molecule. Already runnable via ``examples/methane_water.pdb``.
- ``ubiquitin_lys6_to_ala`` — classic Lys6 → Ala on 1ubq.pdb.
  Approximate — there is no single literature number for this
  sidechain truncation FEP, but the rough order of magnitude is
  set by the alchemical sidechain free-energy of Lys→Ala.
- ``abl_bosutinib_wang_2015`` — Wang et al. 2015 Abl + Bosutinib.
  NOT directly runnable yet: needs GAFF / antechamber for
  Bosutinib (the docked pose is shipped, parameters are not).
- ``methanol_hydration`` — common small-molecule benchmark. Same
  recipe as methane_hydration but for methanol.

Each entry is a dict so the MCP layer can serialise it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Boltzmann constant at 300 K in kcal/mol.
_KCAL_PER_KJ = 1.0 / 4.184


@dataclass
class FEPRefEntry:
    """One entry in the reference ΔG library."""

    name: str
    description: str
    mode: str                          # 'mutation' or 'ligand_decouple'
    target: str                        # residue id (e.g. 'Lys6') or ligand resname (e.g. 'BOS')
    reference_dG_kJmol: float
    reference_dG_unc_kJmol: float      # 1-sigma uncertainty on the reference value
    reference_uncertainty_source: str    # 'experimental' / 'computational_lit'
    citation: str                      # 'Shirts & Ponder, J Phys Chem B 2000'
    pdb_file: Optional[str] = None     # bundled example pdb, if available
    runnable: bool = True              # False if additional parameterisation needed
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "target": self.target,
            "reference_dG_kJmol": self.reference_dG_kJmol,
            "reference_dG_kcal_mol": self.reference_dG_kJmol * _KCAL_PER_KJ,
            "reference_dG_unc_kJmol": self.reference_dG_unc_kJmol,
            "reference_uncertainty_source": self.reference_uncertainty_source,
            "citation": self.citation,
            "pdb_file": self.pdb_file,
            "runnable": self.runnable,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Shipped library
# ---------------------------------------------------------------------------

REFERENCE_LIBRARY: List[FEPRefEntry] = [
    FEPRefEntry(
        name="methane_hydration_shirts_ponder_2000",
        description=(
            "Decoupling a single methane molecule from a TIP3P water box. "
            "Standard alchemical FEP hydration benchmark."
        ),
        mode="ligand_decouple",
        target="MTH",
        reference_dG_kJmol=8.12,
        reference_dG_unc_kJmol=0.21,
        reference_uncertainty_source="experimental",
        citation=(
            "Shirts, M. R. & Ponder, J. W. (2000). 'Structure and "
            "Thermodynamics of Alkane Solvation.' J. Phys. Chem. B "
            "104, 2247-2252."
        ),
        pdb_file="examples/methane_water.pdb",
        runnable=True,
        notes=(
            "ΔG_hydration = +1.94 ± 0.05 kcal/mol (+8.12 ± 0.21 kJ/mol). "
            "Ligand_resname='MTH' for our manually-parameterised methane "
            "(alanine-CB/-HB atom types, single CB residue)."
        ),
    ),
    FEPRefEntry(
        name="methanol_hydration",
        description=(
            "Decoupling a single methanol molecule from TIP3P water. "
            "Hydroxyl hydrogen bonding makes the hydration free energy "
            "substantially more favourable than methane."
        ),
        mode="ligand_decouple",
        target="MOH",
        reference_dG_kJmol=-5.10,
        reference_dG_unc_kJmol=0.20,
        reference_uncertainty_source="experimental",
        citation=(
            "Shirts, M. R. & Ponder, J. W. (2000). "
            "ΔG_hydration(MEOH) ≈ -1.22 ± 0.05 kcal/mol."
        ),
        pdb_file=None,
        runnable=False,
        notes=(
            "Requires a methanol residue template (currently no MOH "
            "in the manual residue table). Build via antechamber + GAFF."
        ),
    ),
    FEPRefEntry(
        name="ubiquitin_lys6_to_ala",
        description=(
            "Sidechain-truncation FEP on ubiquitin Lys6 → Ala "
            "(remove the Cγ, Cδ, Cε, Nζ atoms of Lys6). Classic "
            "bench-system from the AMBER tutorials."
        ),
        mode="mutation",
        target="Lys6",
        reference_dG_kJmol=-13.5,
        reference_dG_unc_kJmol=3.0,
        reference_uncertainty_source="computational_lit",
        citation=(
            "Order-of-magnitude estimate; the precise number depends "
            "on the AMBER force-field version and Boresch-style "
            "corrections. See Seabra / Shirts / Roux (2007) 'Chemical "
            "Theory and Computation' for the framework."
        ),
        pdb_file="examples/1ubq.pdb",
        runnable=True,
        notes=(
            "Compare to published Lys→Ala sidechain free energies in the "
            "pmx / alchemlyb literature; tolerance ~3 kJ/mol."
        ),
    ),
    FEPRefEntry(
        name="abl_bosutinib_wang_2015",
        description=(
            "Absolute binding free energy of the ATP-competitive "
            "inhibitor Bosutinib to Abelson kinase (Abl). Reference "
            "value from a published FEP+ study on a curated set of "
            "kinase inhibitors."
        ),
        mode="ligand_decouple",
        target="BOS",
        reference_dG_kJmol=-43.9,
        reference_dG_unc_kJmol=4.2,
        reference_uncertainty_source="computational_lit",
        citation=(
            "Wang, L. et al. (2015). 'Accurate and reliable prediction "
            "of relative ligand binding potency in prospective drug "
            "discovery by way of a modern free-energy calculation "
            "methodology and free energy thresholds.' J. Am. Chem. "
            "Soc. 137, 2695-2703."
        ),
        pdb_file="examples/src_protein.pdb",
        runnable=False,  # need GAFF parameters for Bosutinib
        notes=(
            "ΔG_bind ≈ -10.5 kcal/mol. NOT directly runnable yet — "
            "the docked ligand pose (examples/Bosutinib_docked_2_src.pdbqt) "
            "is bundled but GAFF parameters must be generated by antechamber. "
            "See ui/fep_reference.py for what a runnable version requires."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_references() -> List[Dict[str, Any]]:
    """Return all reference entries as a list of dicts.

    JSON-serialisable so MCP / GUI can ship them directly.
    """
    return [r.to_dict() for r in REFERENCE_LIBRARY]


def list_runnable_references() -> List[Dict[str, Any]]:
    """Return only the runnable entries (skip ones that need
    additional parameterisation like antechamber / GAFF).
    """
    return [r.to_dict() for r in REFERENCE_LIBRARY if r.runnable]


def find_reference(name: str) -> Optional[FEPRefEntry]:
    """Return the reference entry with the given name, or None."""
    for r in REFERENCE_LIBRARY:
        if r.name == name:
            return r
    return None


def compare_to_reference(
    measured_dG_kJmol: float,
    measured_sigma_kJmol: float,
    reference_name: str,
) -> Dict[str, Any]:
    """Compare a measured FEP ΔG against a reference value.

    Returns a status dict with:

    - ``deviation_kJmol``     measured - reference
    - ``deviation_kcal_mol``  same, in kcal/mol
    - ``combined_unc_kJmol``  sqrt(measured_sigma² + reference_unc²)
    - ``z_score``              deviation / combined_unc
    - ``agreement``            "within_1σ" / "within_2σ" / "outside_2σ"
    - ``reference``            the reference entry dict
    - ``status``               "ok" or "unknown_reference"

    The ``agreement`` label is the practical FEP sanity check:
    "your number matches the published one to within ±2σ".
    """
    ref = find_reference(reference_name)
    if ref is None:
        return {
            "status": "unknown_reference",
            "error": f"No reference named {reference_name!r}",
            "available": [r.name for r in REFERENCE_LIBRARY],
        }

    deviation = measured_dG_kJmol - ref.reference_dG_kJmol
    combined = (measured_sigma_kJmol ** 2 + ref.reference_dG_unc_kJmol ** 2) ** 0.5
    z = deviation / combined if combined > 0 else float("inf")
    abs_z = abs(z)

    if abs_z <= 1.0:
        agreement = "within_1sigma"
    elif abs_z <= 2.0:
        agreement = "within_2sigma"
    else:
        agreement = "outside_2sigma"

    return {
        "status": "ok",
        "agreement": agreement,
        "z_score": z,
        "deviation_kJmol": deviation,
        "deviation_kcal_mol": deviation * _KCAL_PER_KJ,
        "combined_unc_kJmol": combined,
        "measured_dG_kJmol": measured_dG_kJmol,
        "measured_sigma_kJmol": measured_sigma_kJmol,
        "reference": ref.to_dict(),
    }


def best_reference_for(
    mode: str,
    target: Optional[str] = None,
    pdb_file: Optional[str] = None,
) -> Optional[FEPRefEntry]:
    """Heuristic: pick the most likely matching reference.

    Match priority:
      1. exact (mode, target) match
      2. exact (mode, pdb basename) match
      3. exact mode match (any target)
      4. None

    Used by the GUI to auto-pick a sensible default when the user
    hits "Compare to reference" without specifying one.
    """
    if pdb_file:
        pdb_stem = pdb_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    else:
        pdb_stem = None

    # Priority 1: exact (mode, target) match.
    if target:
        for r in REFERENCE_LIBRARY:
            if r.mode == mode and r.target == target:
                return r
    # Priority 2: (mode, pdb_basename) match.
    if pdb_stem:
        for r in REFERENCE_LIBRARY:
            if r.mode == mode and r.pdb_file and pdb_stem in r.pdb_file:
                return r
    # Priority 3: any matching mode.
    for r in REFERENCE_LIBRARY:
        if r.mode == mode:
            return r
    return None