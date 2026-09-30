"""Tests for the FEP MCP wrappers and recipe presets.

These tests cover the new ``core.fep_mcp`` and ``core.fep_recipes``
modules — the layer that fronts the heavy ``core.fep`` functions for
the MCP server and GUI. They do NOT need OpenMM (the wrappers short-
circuit on missing PDB files before importing it), so they can run
on the fast test lane.

The actual MD-coupled paths in ``fep_run_sweep`` are exercised by
the slow tests in ``tests/test_fep.py``.
"""

from __future__ import annotations

import pytest

from protein_studio.core.fep_mcp import (
    fep_build_system,
    fep_configure_recipe,
    fep_list_recipes,
    fep_run_sweep,
)
from protein_studio.core.fep_recipes import RECIPES, get_recipe, list_recipes


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

class TestRecipes:
    def test_list_recipes_returns_all_keys(self):
        names = list_recipes()
        assert isinstance(names, list)
        assert set(names) == set(RECIPES.keys())

    def test_required_recipes_present(self):
        names = set(list_recipes())
        assert {"quick_test", "production_bar"}.issubset(names)

    def test_quick_test_is_fast(self):
        cfg = get_recipe("quick_test")
        assert cfg["n_lambdas"] <= 5
        assert cfg["n_production_steps"] <= 1000

    def test_production_bar_has_more_windows(self):
        cfg = get_recipe("production_bar")
        assert cfg["n_lambdas"] >= 11
        assert cfg["n_production_steps"] >= cfg["n_lambdas"] * 500

    def test_unknown_recipe_falls_back_to_quick_test(self):
        cfg = get_recipe("nonexistent-recipe")
        assert cfg["name"] == "quick_test"

    def test_recipes_have_descriptions(self):
        for name, cfg in RECIPES.items():
            assert "description" in cfg
            assert len(cfg["description"]) > 20, (
                f"{name} recipe missing description"
            )

    def test_recipes_json_serialisable(self):
        import json
        # Should round-trip cleanly so MCP / GUI can ship them.
        for name, cfg in RECIPES.items():
            json.dumps(cfg)


# ---------------------------------------------------------------------------
# Configure tool
# ---------------------------------------------------------------------------

class TestFepConfigureRecipe:
    def test_returns_full_recipe_for_known_name(self):
        r = fep_configure_recipe("quick_test")
        assert r["status"] == "ok"
        assert r["recipe"]["name"] == "quick_test"

    def test_returns_error_for_unknown(self):
        r = fep_configure_recipe("not-a-recipe")
        assert r["status"] == "error"
        assert "available" in r
        assert "quick_test" in r["available"]

    def test_list_recipes_returns_all_with_details(self):
        r = fep_list_recipes()
        assert r["status"] == "ok"
        assert set(r["recipes"]) == set(RECIPES.keys())
        assert r["details"] is RECIPES


# ---------------------------------------------------------------------------
# Build tool (no OpenMM needed)
# ---------------------------------------------------------------------------

class TestFepBuildSystem:
    def test_missing_pdb_returns_error(self, tmp_path):
        r = fep_build_system("/nonexistent/file.pdb", mutation="Lys6")
        assert r["status"] == "error"
        assert "not found" in r["error"].lower()

    def test_both_modes_returns_error(self, tmp_path):
        # Touch a fake pdb so the file check passes first.
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_build_system(
            str(fake), mutation="Lys6", ligand_resname="LIG"
        )
        assert r["status"] == "error"
        assert "one of" in r["error"]

    def test_neither_mode_returns_error(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_build_system(str(fake))
        assert r["status"] == "error"
        assert "Specify" in r["error"]

    def test_parses_residue_id_with_letters(self):
        # The parser is private but worth a sanity check via the
        # build path's failure mode — we can't actually build
        # without OpenMM, but the parser strips the letters and
        # passes the digit to the OpenMM call, which will then
        # raise a different error. We just confirm the build
        # function runs without crashing on a valid residue id.
        from protein_studio.core.fep_mcp import _parse_residue_id
        assert _parse_residue_id("Lys6", "/x") == 5
        assert _parse_residue_id("LYS6", "/x") == 5
        assert _parse_residue_id("6", "/x") == 5
        assert _parse_residue_id("  Lys42 ", "/x") == 41
        # Bad input falls back to 0 (OpenMM then raises a real error).
        assert _parse_residue_id("nonsense", "/x") == 0


# ---------------------------------------------------------------------------
# Run sweep (no-OpenMM paths)
# ---------------------------------------------------------------------------

class TestFepRunSweep:
    def test_missing_pdb(self):
        r = fep_run_sweep("/no.pdb", mode="mutation", mutation="Lys6")
        assert r["status"] == "error"
        assert "not found" in r["error"].lower()

    def test_unknown_mode(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep(str(fake), mode="decouple_nuclear")
        assert r["status"] == "error"
        assert "mode" in r["error"].lower()

    def test_mutation_mode_requires_mutation(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep(str(fake), mode="mutation")
        assert r["status"] == "error"
        assert "mutation" in r["error"].lower()

    def test_ligand_mode_requires_ligand(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep(str(fake), mode="ligand_decouple")
        assert r["status"] == "error"
        assert "ligand" in r["error"].lower()


# ---------------------------------------------------------------------------
# MCP catalog
# ---------------------------------------------------------------------------

class TestMcpCatalog:
    def test_fep_tools_registered(self):
        from protein_studio.mcp_server import TOOL_CATALOG
        names = {t["name"] for t in TOOL_CATALOG}
        assert {"fep_build_system",
                "fep_configure_recipe",
                "fep_run_sweep"}.issubset(names)

    def test_fep_tool_schemas_require_pdb_path(self):
        from protein_studio.mcp_server import TOOL_CATALOG
        by_name = {t["name"]: t for t in TOOL_CATALOG}
        for name in ("fep_build_system", "fep_run_sweep"):
            required = by_name[name]["inputSchema"]["required"]
            assert "pdb_path" in required, f"{name} missing pdb_path required"

    def test_fep_configure_recipe_schema(self):
        from protein_studio.mcp_server import TOOL_CATALOG
        by_name = {t["name"]: t for t in TOOL_CATALOG}
        t = by_name["fep_configure_recipe"]
        assert "name" in t["inputSchema"]["required"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])