"""Tests for the FEP visualisation helpers.

Covers:

- ``fep_get_disappearing_atoms`` error paths (no OpenMM needed) and
  the lightweight subset-PDB writer.
- ``fep_capture_lambda_snapshots`` argument validation and output-dir
  default derivation.
- The ``_write_subset_pdb`` helper (called when output_pdb is set).
- Tool catalog presence in MCP server.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# fep_get_disappearing_atoms error paths
# ---------------------------------------------------------------------------

class TestFepGetDisappearingAtoms:
    def test_missing_pdb(self):
        from protein_studio.core.fep_mcp import fep_get_disappearing_atoms
        r = fep_get_disappearing_atoms("/no.pdb", "mutation", mutation="Lys6")
        assert r["status"] == "error"
        assert "not found" in r["error"].lower()

    def test_unknown_mode(self, tmp_path):
        from protein_studio.core.fep_mcp import fep_get_disappearing_atoms
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_get_disappearing_atoms(str(fake), mode="decouple_nuclear")
        assert r["status"] == "error"
        assert "mode" in r["error"].lower()

    def test_mutation_mode_requires_mutation(self, tmp_path):
        from protein_studio.core.fep_mcp import fep_get_disappearing_atoms
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_get_disappearing_atoms(str(fake), mode="mutation")
        assert r["status"] == "error"
        assert "mutation" in r["error"].lower()

    def test_ligand_mode_requires_ligand(self, tmp_path):
        from protein_studio.core.fep_mcp import fep_get_disappearing_atoms
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_get_disappearing_atoms(str(fake), mode="ligand_decouple")
        assert r["status"] == "error"
        assert "ligand" in r["error"].lower()

    def test_subset_pdb_writer_filters_atoms(self, tmp_path):
        """The subset-PDB writer keeps only ATOM lines whose serial
        matches an atom index in keep_indices."""
        from protein_studio.core.fep_mcp import _write_subset_pdb

        # Build a tiny PDB with 5 atoms. Each line ends with \n so
        # ``readlines()`` in the helper yields one ATOM record per
        # line.
        lines = [
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n",
            "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n",
            "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n",
            "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O\n",
            "ATOM      5  CB  ALA A   1       1.500   1.500   0.000  1.00  0.00           C\n",
            "END\n",
        ]
        src = tmp_path / "src.pdb"
        src.write_text("".join(lines))

        # Keep atoms 2 and 4 (1-based serial).
        out = tmp_path / "out.pdb"
        _write_subset_pdb(
            str(src),
            topology=None,  # not used by this helper
            keep_indices=[1, 3],  # 0-based atom indices → serials 2 and 4
            out_path=str(out),
        )
        text = out.read_text()
        # We expect to keep the lines corresponding to original
        # serials 2 and 4 (CA and O of ALA). Serial numbers are
        # renumbered to 1, 2 in the output, but the residue + atom
        # name uniquely identify the kept records.
        atom_names = [
            m.group(1)
            for m in re.finditer(r"ATOM\s+\d+\s+([A-Z0-9]+)", text)
        ]
        assert sorted(atom_names) == ["CA", "O"]
        # And only two ATOM records remain.
        n_atoms = len(re.findall(r"^ATOM", text, re.MULTILINE))
        assert n_atoms == 2


# ---------------------------------------------------------------------------
# fep_capture_lambda_snapshots error paths
# ---------------------------------------------------------------------------

class TestFepCaptureLambdaSnapshots:
    def test_missing_pdb(self):
        from protein_studio.core.fep_mcp import fep_capture_lambda_snapshots
        r = fep_capture_lambda_snapshots("/no.pdb", "mutation", mutation="Lys6")
        assert r["status"] == "error"
        assert "not found" in r["error"].lower()


# ---------------------------------------------------------------------------
# MCP catalog
# ---------------------------------------------------------------------------

class TestMCPCatalog:
    def test_visualisation_tools_present(self):
        from protein_studio.mcp_server import TOOL_CATALOG
        names = {t["name"] for t in TOOL_CATALOG}
        assert "fep_get_disappearing_atoms" in names
        assert "fep_capture_lambda_snapshots" in names

    def test_get_disappearing_schema(self):
        from protein_studio.mcp_server import TOOL_CATALOG
        by_name = {t["name"]: t for t in TOOL_CATALOG}
        s = by_name["fep_get_disappearing_atoms"]["inputSchema"]
        assert "pdb_path" in s["required"]
        assert "mode" in s["required"]

    def test_capture_snapshots_schema(self):
        from protein_studio.mcp_server import TOOL_CATALOG
        by_name = {t["name"]: t for t in TOOL_CATALOG}
        s = by_name["fep_capture_lambda_snapshots"]["inputSchema"]
        assert "pdb_path" in s["required"]
        assert "mode" in s["required"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])