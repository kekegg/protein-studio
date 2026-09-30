"""Tests for the FEP reference ΔG library + automatic benchmark comparison."""

from __future__ import annotations

import pytest

from protein_studio.core.fep_reference import (
    REFERENCE_LIBRARY,
    best_reference_for,
    compare_to_reference,
    find_reference,
    list_references,
    list_runnable_references,
)


# ---------------------------------------------------------------------------
# Library shape
# ---------------------------------------------------------------------------

class TestLibrary:
    def test_list_references_returns_all_entries(self):
        refs = list_references()
        assert isinstance(refs, list)
        assert len(refs) == len(REFERENCE_LIBRARY)

    def test_list_runnable_excludes_unrunnable(self):
        all_refs = list_references()
        runnable = list_runnable_references()
        assert len(runnable) < len(all_refs)
        for r in runnable:
            assert r["runnable"] is True

    def test_shirts_ponder_methane_in_library(self):
        names = {r["name"] for r in list_references()}
        assert "methane_hydration_shirts_ponder_2000" in names

    def test_wang_abl_bosutinib_in_library(self):
        names = {r["name"] for r in list_references()}
        assert "abl_bosutinib_wang_2015" in names

    def test_each_entry_has_required_keys(self):
        required = {
            "name", "description", "mode", "target",
            "reference_dG_kJmol", "reference_dG_unc_kJmol",
            "citation", "runnable",
        }
        for r in list_references():
            assert required.issubset(r.keys()), (
                f"{r.get('name')} missing keys"
            )

    def test_methane_kcal_conversion(self):
        methane = find_reference("methane_hydration_shirts_ponder_2000")
        assert methane is not None
        # 8.12 kJ/mol ≈ 1.94 kcal/mol
        d = methane.to_dict()
        assert d["reference_dG_kcal_mol"] == pytest.approx(1.94, abs=0.05)

    def test_find_reference_returns_none_for_unknown(self):
        assert find_reference("does-not-exist") is None


# ---------------------------------------------------------------------------
# compare_to_reference
# ---------------------------------------------------------------------------

class TestCompare:
    def test_within_one_sigma(self):
        # Measured exactly at the reference value — z should be 0.
        methane = find_reference("methane_hydration_shirts_ponder_2000")
        assert methane is not None
        cmp = compare_to_reference(
            measured_dG_kJmol=methane.reference_dG_kJmol,
            measured_sigma_kJmol=0.5,
            reference_name=methane.name,
        )
        assert cmp["status"] == "ok"
        assert cmp["agreement"] == "within_1sigma"
        assert cmp["z_score"] == pytest.approx(0.0, abs=1e-9)
        assert cmp["deviation_kJmol"] == pytest.approx(0.0, abs=1e-9)

    def test_within_two_sigma(self):
        methane = find_reference("methane_hydration_shirts_ponder_2000")
        assert methane is not None
        # Move 1.5σ away using combined uncertainty.
        cmp = compare_to_reference(
            measured_dG_kJmol=methane.reference_dG_kJmol + 0.5,
            measured_sigma_kJmol=0.3,
            reference_name=methane.name,
        )
        # combined = sqrt(0.3² + 0.21²) ≈ 0.366
        # z = 0.5 / 0.366 ≈ 1.37 -> within_2sigma
        assert cmp["agreement"] == "within_2sigma"
        assert 1.0 < abs(cmp["z_score"]) < 2.0

    def test_outside_two_sigma(self):
        methane = find_reference("methane_hydration_shirts_ponder_2000")
        assert methane is not None
        cmp = compare_to_reference(
            measured_dG_kJmol=methane.reference_dG_kJmol + 5.0,
            measured_sigma_kJmol=0.1,
            reference_name=methane.name,
        )
        assert cmp["agreement"] == "outside_2sigma"
        assert abs(cmp["z_score"]) > 2.0

    def test_unknown_reference(self):
        cmp = compare_to_reference(
            measured_dG_kJmol=8.0,
            measured_sigma_kJmol=0.5,
            reference_name="not-a-real-reference",
        )
        assert cmp["status"] == "unknown_reference"
        assert "available" in cmp
        assert "methane_hydration_shirts_ponder_2000" in cmp["available"]

    def test_combined_uncertainty_is_quadrature_sum(self):
        methane = find_reference("methane_hydration_shirts_ponder_2000")
        assert methane is not None
        sigma_meas = 0.4
        sigma_ref = methane.reference_dG_unc_kJmol
        cmp = compare_to_reference(
            measured_dG_kJmol=methane.reference_dG_kJmol,
            measured_sigma_kJmol=sigma_meas,
            reference_name=methane.name,
        )
        expected = (sigma_meas ** 2 + sigma_ref ** 2) ** 0.5
        assert cmp["combined_unc_kJmol"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# best_reference_for
# ---------------------------------------------------------------------------

class TestBestReference:
    def test_exact_match_by_mode_and_target(self):
        # ligand_decouple + MTH should pick methane.
        ref = best_reference_for(mode="ligand_decouple", target="MTH")
        assert ref is not None
        assert ref.name == "methane_hydration_shirts_ponder_2000"

    def test_match_by_mode_and_pdb_basename(self):
        ref = best_reference_for(
            mode="ligand_decouple",
            pdb_file="examples/methane_water.pdb",
        )
        assert ref is not None
        assert ref.name == "methane_hydration_shirts_ponder_2000"

    def test_match_by_mode_only(self):
        # BOS isn't in any entry's target field, but pdb basename
        # might match for Abl+Bosutinib.
        ref = best_reference_for(
            mode="ligand_decouple",
            target="BOS",
        )
        assert ref is not None
        # Without pdb_file, only Abl+Bosutinib has target='BOS'.
        assert ref.name == "abl_bosutinib_wang_2015"

    def test_falls_back_to_mode_only(self):
        # Unknown target for mutation mode → ubiquitin Lys6 entry.
        ref = best_reference_for(mode="mutation", target="Glu34")
        assert ref is not None
        assert ref.mode == "mutation"

    def test_returns_none_for_completely_unknown(self):
        ref = best_reference_for(mode="impossible_mode")
        assert ref is None


# ---------------------------------------------------------------------------
# MCP wrappers
# ---------------------------------------------------------------------------

class TestMCPLayer:
    def test_list_references(self):
        from protein_studio.core.fep_mcp import fep_list_references
        r = fep_list_references()
        assert r["status"] == "ok"
        assert r["n_references"] == len(REFERENCE_LIBRARY)

    def test_list_references_runnable_only(self):
        from protein_studio.core.fep_mcp import fep_list_references
        r = fep_list_references(runnable_only=True)
        assert all(x["runnable"] for x in r["references"])

    def test_compare_sweep_to_reference(self):
        from protein_studio.core.fep_mcp import fep_compare_sweep_to_reference
        result = {"delta_g_kJmol": 8.12, "sigma_kJmol": 0.21}
        cmp = fep_compare_sweep_to_reference(
            result,
            "methane_hydration_shirts_ponder_2000",
        )
        assert cmp["status"] == "ok"
        assert cmp["agreement"] == "within_1sigma"

    def test_compare_sweep_missing_dg(self):
        from protein_studio.core.fep_mcp import fep_compare_sweep_to_reference
        cmp = fep_compare_sweep_to_reference(
            {}, "methane_hydration_shirts_ponder_2000",
        )
        assert cmp["status"] == "error"

    def test_best_reference(self):
        from protein_studio.core.fep_mcp import fep_best_reference
        r = fep_best_reference(mode="ligand_decouple", target="MTH")
        assert r["status"] == "ok"
        assert r["reference"]["name"] == (
            "methane_hydration_shirts_ponder_2000"
        )

    def test_best_reference_no_match(self):
        from protein_studio.core.fep_mcp import fep_best_reference
        r = fep_best_reference(mode="impossible_mode")
        assert r["status"] == "no_match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])