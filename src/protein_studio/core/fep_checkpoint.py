"""FEP checkpoint / resume.

A :class:`FEPCheckpoint` captures the state of a long-running
``run_amber_mutation_fep`` or ``run_ligand_decouple_fep`` sweep
that has been interrupted partway through. The wrapper layer
(``core.fep_mcp``) writes a checkpoint file after each completed
λ window; on resume it skips the already-completed windows and
continues from where the previous run stopped.

Storage format is **plain pickle** — the sweep's data is just
floats + lists (no OpenMM objects), and pickle is built-in. If a
checkpoint becomes corrupt or we ever need to inspect one, the
pickle can be loaded with any tool.

Public surface
--------------

- :class:`FEPCheckpoint`     — dataclass with all the state.
- :func:`save_checkpoint`    — pickle to disk.
- :func:`load_checkpoint`    — load from disk (returns None if missing).
- :func:`checkpoint_status`  — summary dict for MCP / GUI.
- :func:`delete_checkpoint`  — remove from disk.
- :func:`default_checkpoint_path` — derive a path from the inputs
  (so the GUI doesn't have to ask the user where to put it).
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def default_checkpoint_path(
    pdb_path: str,
    mode: str,
    mutation: Optional[str],
    ligand_resname: Optional[str],
    recipe_name: str,
    out_dir: Optional[str] = None,
) -> str:
    """Derive a stable checkpoint filename from the sweep inputs.

    The same sweep inputs (PDB + mode + residue/ligand + recipe)
    always map to the same path, so a resume after a restart picks
    up the existing file automatically.

    Output directory defaults to ``./fep_checkpoints`` next to the
    PDB if not given.
    """
    pdb = Path(pdb_path)
    target = mutation if mutation else ligand_resname or "unknown"
    safe_target = str(target).replace("/", "_").replace("\\", "_")
    stem = f"{pdb.stem}_{mode}_{recipe_name}_{safe_target}.fepcp"
    if out_dir is None:
        out_dir = str(pdb.parent / "fep_checkpoints")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(out_dir) / stem)


@dataclass
class FEPCheckpoint:
    """State of an in-progress FEP sweep.

    The ``completed_lambdas`` list is the source of truth for
    resume: the wrapper layer only runs λ values not in this list.
    """
    pdb_path: str
    mode: str                       # 'mutation' | 'ligand_decouple'
    mutation: Optional[str]
    ligand_resname: Optional[str]
    recipe_name: str
    lambdas: List[float]
    completed_lambdas: List[float] = field(default_factory=list)
    energies_by_lambda: Dict[float, List[float]] = field(default_factory=dict)
    neighbor_energies: Dict[float, List[Any]] = field(default_factory=dict)
    disappearing: List[int] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Pickle to ``path``. Atomic via temp-file rename."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = time.time()
        if not self.created_at:
            self.created_at = self.updated_at
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    @classmethod
    def load(cls, path: str) -> "FEPCheckpoint":
        """Load from ``path``. Raises FileNotFoundError if missing."""
        with open(path, "rb") as f:
            return pickle.load(f)


def save_checkpoint(checkpoint: FEPCheckpoint, path: str) -> None:
    """Convenience wrapper around :meth:`FEPCheckpoint.save`."""
    checkpoint.save(path)


def load_checkpoint(path: str) -> Optional[FEPCheckpoint]:
    """Load checkpoint from ``path`` or return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    return FEPCheckpoint.load(str(p))


def delete_checkpoint(path: str) -> bool:
    """Remove ``path``. Returns True if a file was deleted."""
    p = Path(path)
    if p.exists():
        p.unlink()
        # Also clean up the temp file if one was left behind by an
        # interrupted atomic-write.
        tmp = p.with_suffix(p.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        return True
    return False


def checkpoint_status(path: str) -> Dict[str, Any]:
    """Return a summary dict for the MCP / GUI to display.

    Status field values:
    - ``missing``  — no checkpoint at this path.
    - ``ok``       — checkpoint loaded successfully.
    - ``corrupt``  — pickle failed; checkpoint is unusable.
    """
    p = Path(path)
    if not p.exists():
        return {
            "status": "missing",
            "path": str(path),
            "exists": False,
        }
    try:
        cp = FEPCheckpoint.load(str(p))
    except Exception as e:
        return {
            "status": "corrupt",
            "path": str(path),
            "error": f"{type(e).__name__}: {e}",
        }

    remaining = [lam for lam in cp.lambdas
                 if lam not in cp.completed_lambdas]
    return {
        "status": "ok",
        "path": str(path),
        "mode": cp.mode,
        "recipe_name": cp.recipe_name,
        "pdb_path": cp.pdb_path,
        "n_total_windows": len(cp.lambdas),
        "n_completed": len(cp.completed_lambdas),
        "n_remaining": len(remaining),
        "progress_pct": (
            100.0 * len(cp.completed_lambdas) / max(len(cp.lambdas), 1)
        ),
        "completed_lambdas": cp.completed_lambdas,
        "remaining_lambdas": remaining,
        "created_at": cp.created_at,
        "updated_at": cp.updated_at,
    }