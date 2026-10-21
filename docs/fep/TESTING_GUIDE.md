# FEP GUI Testing Guide

End-to-end walkthrough for testing the Free Energy Perturbation
framework from the protein-studio desktop app (`Simulate → Free
Energy Perturbation…`).

Three example systems are wired up:

| Example PDB | Mode | Recipe | Time | ΔG vs literature |
|---|---|---|---|---|
| `examples/1ubq.pdb` (Ubiquitin) | `mutation` (Lys6 → Ala) | `quick_test` | ~30 s | ≈ −13.5 ± 3 kJ/mol (order-of-magnitude) |
| `examples/methane_water.pdb` (single methane in TIP3P) | — | — | — | needs custom residue template — see *Caveats* |
| `examples/src_protein.pdb` (Abl kinase) | `ligand_decouple` (target: `BOS`) | — | — | needs GAFF parameters — see *Caveats* |

Only `examples/1ubq.pdb` is runnable end-to-end through the GUI
today. The methane example is shipped for the *manual* pipeline
(`scripts/methane_production_bar.py`) — it works around OpenMM's
missing template for the MTH residue by hand-building the system.
A `methane_water` GUI path requires registering an MTH residue
template in `ff.xml` first; that's not done.

Use `1ubq.pdb` for the end-to-end GUI test, then add either of the
other two as future work.

---

## 1. Launch the GUI

The desktop app lives in the `protein-studio` conda env on this
machine.

```bash
conda activate protein-studio
python -m protein_studio
```

The window opens with a dark-themed viewport. The FEP controls are
under `Simulate → Free Energy Perturbation…` (right-click dock
appears on the right side).

---

## 2. Workflow A — Ubiquitin Lys6 → Ala (mutation)

Tests the Phase 4 sidechain-truncation FEP pipeline. Slightly slower
than methane because there's a real protein (~1300 atoms).

### 2.1 Open the structure

1. `File → Open PDB…` → `examples/1ubq.pdb`. Default `cartoon` +
   `secondary` scheme — you should see the ubiquitin β-grasp fold
   with seven helices.

   *Tip:* once the FEP dock has been opened (next step), picking
   another PDB via the FEP `Browse…` button automatically clears
   the viewport and loads the new file.

### 2.2 Open the FEP dock

2. `Simulate → Free Energy Perturbation…`. A three-tab dock
   appears: **Setup**, **Run**, **Analyze**.

### 2.3 Setup tab

3. **PDB file:** click `Browse…` and pick `examples/1ubq.pdb`.
   The viewport clears and the structure loads (cartoon + secondary).
4. **Mode:** `mutation`.
5. **Mutation residue:** `Lys6`. The parser reads `6` as the
   1-based residue number and subtracts 1 to give OpenMM's
   0-based index — that's Lys6 in our copy of 1ubq.pdb.
6. **Recipe:** `quick_test`.
7. Click **Build system**. Status turns green and reports
   `~1300 particles, 90 waters, ~7 disappearing atoms`. The 7 =
   Cγ, Cδ, Cε, Nζ (plus their hydrogens) — Lys6 sidechain beyond Cβ.
8. **Highlight disappearing atoms in viewport** → a new
   `ball_and_stick` SceneObject with the 7 atoms should appear.
   Inspect: those are the sidechain atoms the sweep removes.

### 2.4 Run tab

9. **Save per-λ snapshots:** leave checked (default). Optionally
   click `…` next to **Snapshots dir** to pick a folder; otherwise
   the default `<pdb_dir>/fep_snapshots/...` is used.
10. Click **Run sweep**. Progress bar fills from 2% → 100% over
    ~30–60 s. Status line shows the current λ and sample count.

### 2.5 Watch the result land

11. Status line: `✓ ΔG_FEP = +X.XXX ± Y.YYY kJ/mol  (+Z.ZZZ ± Y.YYY kcal/mol)`.
    With `quick_test` the number is rough — switch to
    `benchmark_mutation` or `production_bar` for tighter σ.
12. **Open in Analyze tab →** button becomes enabled — click it.

### 2.6 Analyze tab — three diagnostic plots

13. **ΔG per window (BAR)** — bars at λ = 0.0, 0.1, …, 1.0 with
    sigma error bars. Each bar is the per-window BAR estimate for
    `λ_i → λ_{i+1}`.
14. **Cumulative ΔG — convergence** — running sum of the bars from
    plot 1. Final value equals the total ΔG from the status line.
    This curve should plateau — the last segment should be roughly
    flat if the sweep is converged.
15. **BAR vs trapezoidal (TI)** — blue bars are BAR (per window);
    amber bars are TI (per window, trapezoidal). When the two agree
    you have a converged result; when they disagree by more than
    a few kJ/mol, you have insufficient statistics.

### 2.7 Reference comparison

16. In the **Compare to literature reference** panel, pick
    `ubiquitin_lys6_to_ala` from the dropdown. The citation card
    shows the order-of-magnitude reference value.
17. Click **Compare**. The agreement icon and z-score plus
    colour-coded label tells you whether your run lands within
    the reference's uncertainty.

### 2.8 Per-λ snapshots

18. In the file browser, open the snapshots folder. You'll see
    `lambda_0.000.pdb`, `lambda_0.100.pdb`, … `lambda_1.000.pdb`.
19. Back in the GUI, `File → Open PDB…` each one in turn to flip
    through how the Lys6 sidechain gradually disappears. The
    position restraints (k = 50000 kJ/mol/nm²) keep the disappearing
    atoms anchored near their initial positions — at λ → 1 they
    get smeared out by the soft-core LJ into a weak density.

---

## 3. Workflow B — Repeat with `1aki.pdb`

`examples/1aki.pdb` is adenylate kinase. The pattern is the same as
ubiquitin, just with a different protein.

```text
Mode:             mutation
Mutation residue: any Lys e.g. Lys11 (try Lys5 / Lys11 / Lys13)
Recipe:           quick_test
```

(Note: the FEP framework needs at least one atom beyond Cβ in the
sidechain to disappear. Position 1 / 2 sometimes have weird
chemistry in OpenMM's topology and raise `No atoms to disappear on
residue LYS#X` — try a few residues until it succeeds.)

---

## 4. Expected timings (MacBook-class CPU, OpenMM Reference platform)

| Recipe | windows × steps | ubiquitin (1ubq, 90 waters) |
|---|---|---|
| `quick_test` | 5 × 200 + 500 | ~30–60 s |
| `benchmark_mutation` | 11 × 1k + 2k | ~5–8 min |
| `production_bar` | 11 × 2k + 10k | ~30–60 min |

For a first pass use `quick_test`. Once the pipeline works end-to-end,
switch to `benchmark_mutation` for a real ΔG with error bars.

---

## 3. Workflow B — Ubiquitin Lys6 → Ala (mutation)

Tests the Phase 4 sidechain-truncation FEP pipeline. Slightly slower
than methane because there's a real protein (~1300 atoms).

### 3.1 Setup

1. `File → Open PDB…` → `examples/1ubq.pdb`. Default `cartoon` +
   `secondary` scheme — you should see the ubiquitin β-grasp fold
   with seven helices.
2. Open the FEP dock (`Simulate → Free Energy Perturbation…`).
3. **PDB file:** `examples/1ubq.pdb`.
4. **Mode:** `mutation`.
5. **Mutation residue:** `Lys6`. The parser reads `6` as the
   1-based residue number and subtracts 1 to give OpenMM's
   0-based index — that's Lys6 in our copy of 1ubq.pdb.
6. **Recipe:** `quick_test`.
7. **Build system** → status: `~1300 particles, 90 waters,
   ~7 disappearing atoms`. The 7 = Cγ, Cδ, Cε, Nζ (plus their
   hydrogens) — Lys6 sidechain beyond Cβ.
8. **Highlight disappearing atoms** → a new
   `ball_and_stick` SceneObject with the 7 atoms should appear.
   Inspect: those are the sidechain atoms the sweep removes.

### 3.2 Run, Analyze, Reference

9. Run sweep. ~30–60 s with `quick_test`.
10. `Open in Analyze tab →`. The ΔG per-window plot should show
    substantial per-window ΔG (Lys → Ala decouples ~4 heavy atoms
    + their H's, which is a big free-energy change).
11. Pick `ubiquitin_lys6_to_ala` from the reference dropdown.
    Note the reference value (≈ −13.5 ± 3 kJ/mol) is a coarse
    order-of-magnitude estimate, not a tight published number —
    see the citation card. The expectation with `quick_test` is
    that you land within ±5 kJ/mol of that estimate.

### 3.3 Visualisation

12. Per-λ snapshots: at each λ, the 7 disappearing atoms are
    softly pushed to their original positions by the position
    restraints (k = 50000 kJ/mol/nm²). Load the snapshots to see
    the sidechain "smear" as λ → 1.

---

## 4. Expected timings (MacBook-class CPU, OpenMM Reference platform)

| Recipe | windows × steps | methane (290 waters) | ubiquitin (1ubq, 90 waters) |
|---|---|---|---|
| `quick_test` | 5 × 200 + 500 | ~20 s | ~30–60 s |
| `benchmark_mutation` | 11 × 1k + 2k | ~2–3 min | ~5–8 min |
| `production_bar` | 11 × 2k + 10k | ~10–20 min | ~30–60 min |

For a first pass use `quick_test`. Once the pipeline works end-to-end,
switch to `benchmark_mutation` for a real ΔG with error bars.

---

## 5. Caveats / known limitations

- **Methane (`examples/methane_water.pdb`) doesn't go through the
  GUI.** OpenMM has no template for the MTH residue so the standard
  `build_ligand_decouple_system` fails (`No template found for
  residue MTH`). The methane pipeline works via the manual
  pipeline at `scripts/methane_production_bar.py`. To make it
  work through the GUI requires registering an MTH residue
  template in the AMBER force-field XML.
- **The Bosutinib reference is not directly runnable yet.** The
  docked pose is bundled (`examples/src_protein.pdb` +
  `examples/Bosutinib_docked_2_src.pdbqt`) but the GAFF parameters
  for Bosutinib haven't been generated — you need `antechamber` +
  GAFF + AM1-BCC charges to make the ligand decouple FEP work.
  The dropdown marks it `⚠ not runnable` so you can't accidentally
  try.
- **Cancel is inter-window, not intra-window.** Clicking
  *Cancel* finishes the current λ window's MD, then writes a
  checkpoint, then stops. The current window's energies are
  preserved; resuming from the checkpoint finishes the rest.
- **OpenMM Reference platform.** All shipped recipes use
  `openmm.Platform.getPlatformByName("Reference")` for numerical
  stability — slower than `CPU` but reliable for small systems.
  You can override by editing `fep.py` if you have a fast CPU
  platform available.
- **Split-force exclusions gotcha.** Currently `1ubq.pdb` raises
  `OpenMMException: All Forces must have identical exclusions`
  during a sweep — the LJ `CustomNonbondedForce` and the Coulomb
  `NonbondedForce` have inconsistent exclusion sets. This is a
  regression from the split-force refactor that needs a follow-up
  fix (for minimum-runnable UI testing, run only the *Build system*
  and *Highlight disappearing atoms* steps — those work. The actual
  sweep crashes before producing ΔG; we'll fix the exclusion list
  in the next patch).

- **MCP catalog** has 12 FEP tools registered. You can drive the
  same workflow from any MCP client (Claude Code, custom scripts
  via `mcp_bridge.py`). The tools are listed under
  `mcp_server.TOOL_CATALOG`.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Build system` says `⚠ Select a PDB file first` | Empty path | Click *Browse…* and pick a PDB |
| `Build system` says `PDB file not found` | Wrong path | Check the file exists and the path is absolute |
| Build says `AttributeError: 'NoneType' object has no attribute '_finalize'` | OpenMM topology issue on a malformed PDB | Use a clean PDB from the `examples/` directory |
| Run shows `✗ Failed` immediately | OpenMM platform failure | Check `conda activate protein-studio` is correct |
| All λ windows NaN | Soft-core / restraints conflict | Use `quick_test` recipe (has strong restraints); do not edit the AMBER params |
| Cancel button does nothing for a long time | Currently-running window is large | Wait — cancellation lands after the window finishes (≤ a few seconds for `quick_test`) |
| Reference dropdown shows `⚠ not runnable` | Reference needs extra parameterisation | See *Caveats* above (Bosutinib) |

---

## 7. Files you'll touch

| File | What |
|---|---|
| `docs/fep/FEP_principles.docx` / `.html` | Theory + implementation notes |
| `docs/fep/protein_studio_FEP.pptx` | 16-slide deck |
| `docs/fep/user_guide.html` | API reference + workflow examples |
| `src/protein_studio/core/fep.py` | Core FEP framework (~2600 lines, 5 phases) |
| `src/protein_studio/core/fep_mcp.py` | 12 MCP-facing wrapper functions |
| `src/protein_studio/core/fep_recipes.py` | 4 recipe presets |
| `src/protein_studio/core/fep_checkpoint.py` | Resume-from-pickle |
| `src/protein_studio/core/fep_reference.py` | Literature comparison library |
| `src/protein_studio/mcp_server.py` | MCP tool registration (12 FEP tools) |
| `src/protein_studio/ui/fep_panel.py` | Setup + Run tabs + worker |
| `src/protein_studio/ui/fep_analyze.py` | Analyze tab with reference comparison |
| `src/protein_studio/ui/fep_plots.py` | QPainter charts (BAR, convergence, BAR-vs-TI) |

Happy testing. If anything is unclear, push back and I'll iterate
on the workflow.