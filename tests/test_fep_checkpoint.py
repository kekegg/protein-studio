"""Tests for FEP checkpoint / resume and Analyze tab data plumbing.

These tests exercise:

- :class:`FEPCheckpoint` save / load / status round-trip.
- :func:`default_checkpoint_path` derivation from sweep inputs.
- :func:`fep_resume_sweep` and :func:`fep_checkpoint_info` error paths
  (no OpenMM needed).
- :func:`_ti_per_window` shape for the Analyze tab's BAR-vs-TI chart.
"""

from __future__ import annotations

import pytest

from protein_studio.core.fep_checkpoint import (
    FEPCheckpoint,
    checkpoint_status,
    default_checkpoint_path,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from protein_studio.core.fep_mcp import (
    fep_checkpoint_delete,
    fep_checkpoint_info,
    fep_resume_sweep,
    fep_run_sweep_with_checkpoint,
)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _make_cp(**overrides) -> FEPCheckpoint:
    base = dict(
        pdb_path="/tmp/foo.pdb",
        mode="mutation",
        mutation="Lys6",
        ligand_resname=None,
        recipe_name="quick_test",
        lambdas=[0.0, 0.25, 0.5, 0.75, 1.0],
        completed_lambdas=[0.0, 0.25],
        energies_by_lambda={
            0.0: [-100.0, -101.0, -102.0],
            0.25: [-50.0, -51.0, -52.0],
        },
        neighbor_energies={0.0: [(None, -50.0)], 0.25: [(-100.0, None)]},
        disappearing=[10, 12, 14, 16, 18, 20, 22],
        created_at=1.0,
        updated_at=2.0,
    )
    base.update(overrides)
    return FEPCheckpoint(**base)


class TestCheckpointIO:
    def test_save_load_round_trip(self, tmp_path):
        cp = _make_cp()
        path = tmp_path / "test.fepcp"
        cp.save(str(path))
        loaded = FEPCheckpoint.load(str(path))
        assert loaded.pdb_path == cp.pdb_path
        assert loaded.lambdas == cp.lambdas
        assert loaded.completed_lambdas == cp.completed_lambdas
        assert loaded.energies_by_lambda == cp.energies_by_lambda
        assert loaded.neighbor_energies == cp.neighbor_energies
        assert loaded.disappearing == cp.disappearing

    def test_load_missing_returns_none(self, tmp_path):
        assert load_checkpoint(str(tmp_path / "missing.fepcp")) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        cp = _make_cp()
        path = tmp_path / "subdir" / "deep" / "x.fepcp"
        cp.save(str(path))
        assert path.exists()

    def test_delete_removes_file(self, tmp_path):
        cp = _make_cp()
        path = tmp_path / "x.fepcp"
        cp.save(str(path))
        assert path.exists()
        assert delete_checkpoint(str(path)) is True
        assert not path.exists()

    def test_delete_missing_returns_false(self, tmp_path):
        assert delete_checkpoint(str(tmp_path / "missing.fepcp")) is False

    def test_status_missing(self, tmp_path):
        s = checkpoint_status(str(tmp_path / "missing.fepcp"))
        assert s["status"] == "missing"
        assert s["exists"] is False

    def test_status_ok(self, tmp_path):
        cp = _make_cp()
        path = tmp_path / "x.fepcp"
        cp.save(str(path))
        s = checkpoint_status(str(path))
        assert s["status"] == "ok"
        assert s["n_total_windows"] == 5
        assert s["n_completed"] == 2
        assert s["n_remaining"] == 3
        assert s["progress_pct"] == pytest.approx(40.0)
        assert s["completed_lambdas"] == [0.0, 0.25]
        assert s["remaining_lambdas"] == [0.5, 0.75, 1.0]
        assert s["mode"] == "mutation"
        assert s["recipe_name"] == "quick_test"


# ---------------------------------------------------------------------------
# default_checkpoint_path
# ---------------------------------------------------------------------------

class TestDefaultCheckpointPath:
    def test_path_is_stable_for_same_inputs(self, tmp_path):
        pdb = tmp_path / "1ubq.pdb"
        pdb.write_text("REMARK dummy\n")
        path_a = default_checkpoint_path(str(pdb), "mutation", "Lys6",
                                         None, "quick_test",
                                         out_dir=str(tmp_path / "ck"))
        path_b = default_checkpoint_path(str(pdb), "mutation", "Lys6",
                                         None, "quick_test",
                                         out_dir=str(tmp_path / "ck"))
        assert path_a == path_b

    def test_path_differs_per_recipe(self, tmp_path):
        pdb = tmp_path / "1ubq.pdb"
        pdb.write_text("REMARK dummy\n")
        path_a = default_checkpoint_path(str(pdb), "mutation", "Lys6",
                                         None, "quick_test",
                                         out_dir=str(tmp_path / "ck"))
        path_b = default_checkpoint_path(str(pdb), "mutation", "Lys6",
                                         None, "production_bar",
                                         out_dir=str(tmp_path / "ck"))
        assert path_a != path_b

    def test_path_differs_per_target(self, tmp_path):
        pdb = tmp_path / "1ubq.pdb"
        pdb.write_text("REMARK dummy\n")
        a = default_checkpoint_path(str(pdb), "mutation", "Lys6", None,
                                    "quick_test", out_dir=str(tmp_path))
        b = default_checkpoint_path(str(pdb), "mutation", "His7", None,
                                    "quick_test", out_dir=str(tmp_path))
        assert a != b

    def test_path_includes_ligand_when_in_ligand_mode(self, tmp_path):
        pdb = tmp_path / "complex.pdb"
        pdb.write_text("REMARK dummy\n")
        path = default_checkpoint_path(str(pdb), "ligand_decouple", None,
                                       "LIG", "quick_test",
                                       out_dir=str(tmp_path))
        assert "LIG" in path


# ---------------------------------------------------------------------------
# MCP wrapper error paths
# ---------------------------------------------------------------------------

class TestFepRunSweepWithCheckpoint:
    def test_missing_pdb(self):
        r = fep_run_sweep_with_checkpoint("/no.pdb", "mutation",
                                           mutation="Lys6")
        assert r["status"] == "error"
        assert "not found" in r["error"].lower()

    def test_unknown_mode(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep_with_checkpoint(str(fake), mode="decouple")
        assert r["status"] == "error"

    def test_mutation_mode_requires_mutation(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep_with_checkpoint(str(fake), mode="mutation")
        assert r["status"] == "error"
        assert "mutation" in r["error"].lower()

    def test_ligand_mode_requires_ligand(self, tmp_path):
        fake = tmp_path / "x.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep_with_checkpoint(str(fake), mode="ligand_decouple")
        assert r["status"] == "error"
        assert "ligand" in r["error"].lower()

    def test_mismatched_existing_checkpoint_rejected(self, tmp_path):
        # Pre-create a checkpoint with mismatched inputs.
        cp = _make_cp(pdb_path="/some/other.pdb")
        cp_path = tmp_path / "x.fepcp"
        cp.save(str(cp_path))

        # Try to resume a different PDB + residue.
        fake = tmp_path / "y.pdb"
        fake.write_text("REMARK dummy\n")
        r = fep_run_sweep_with_checkpoint(
            str(fake), mode="mutation", mutation="His7",
            checkpoint_path=str(cp_path),
        )
        assert r["status"] == "error"
        assert "different inputs" in r["error"]


class TestFepResumeSweep:
    def test_no_checkpoint(self):
        r = fep_resume_sweep("/no/checkpoint.fepcp")
        assert r["status"] == "error"
        assert "No checkpoint" in r["error"]


class TestFepCheckpointInfo:
    def test_missing(self, tmp_path):
        r = fep_checkpoint_info(str(tmp_path / "missing.fepcp"))
        assert r["status"] == "missing"

    def test_ok(self, tmp_path):
        cp = _make_cp()
        path = tmp_path / "x.fepcp"
        cp.save(str(path))
        r = fep_checkpoint_info(str(path))
        assert r["status"] == "ok"
        assert r["n_total_windows"] == 5


class TestFepCheckpointDelete:
    def test_delete_missing(self, tmp_path):
        r = fep_checkpoint_delete(str(tmp_path / "missing.fepcp"))
        assert r["status"] == "missing"
        assert r["deleted"] is False

    def test_delete_ok(self, tmp_path):
        cp = _make_cp()
        path = tmp_path / "x.fepcp"
        cp.save(str(path))
        r = fep_checkpoint_delete(str(path))
        assert r["status"] == "ok"
        assert r["deleted"] is True


# ---------------------------------------------------------------------------
# Analyze tab — TI per window
# ---------------------------------------------------------------------------

class TestTIPerWindow:
    def test_empty(self):
        from protein_studio.core.fep_mcp import ti_per_window
        assert ti_per_window({}, []) == []

    def test_single_window_is_zero(self):
        from protein_studio.core.fep_mcp import ti_per_window
        out = ti_per_window({0.0: [-100.0, -101.0]}, [0.0])
        assert out == [0.0]

    def test_two_windows_simple(self):
        from protein_studio.core.fep_mcp import ti_per_window
        out = ti_per_window({0.0: [-100.0], 0.5: [-50.0]}, [0.0, 0.5])
        # Trapezoid: -0.5 * (0.5 - 0) * (-50 + -100) = -0.5 * 0.5 * -150
        # = 37.5
        assert len(out) == 2
        assert out[0] == 0.0
        assert out[1] == pytest.approx(37.5, abs=0.01)

    def test_three_windows(self):
        from protein_studio.core.fep_mcp import ti_per_window
        energies = {0.0: [0.0], 0.5: [10.0], 1.0: [20.0]}
        lambdas = [0.0, 0.5, 1.0]
        out = ti_per_window(energies, lambdas)
        # out[0] = 0
        # out[1] = -0.5 * 0.5 * (10 + 0) = -2.5
        # out[2] = -0.5 * 0.5 * (20 + 10) = -7.5
        assert out[0] == 0.0
        assert out[1] == pytest.approx(-2.5)
        assert out[2] == pytest.approx(-7.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])