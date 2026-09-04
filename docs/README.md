# docs/ — documentation hub

Orientation documents for this repo. **New here? Read in this order:**

1. **`../readme.md`** — repo structure, how to run the pipeline, geometries table.
2. **`cnt_reflection_measurement_brief.md`** — *why measuring highly reflective (CNT)
   samples is hard, every geometry/trick we've tried, and where it stands.* The single
   orientation doc for the CNT reflection problem: conditioning (`r=−1` pole), the
   high-index window rescue, the air-gap problem, self-referencing, p-polarization, and
   the current verdict. Synthesises the sources below — **start here to evaluate a CNT
   technical idea.**
3. **`../PIPELINE_ARCHITECTURE.md`** — the pipeline data-flow map, the `processing_dict`
   contract, per-stage I/O. Read before touching a pipeline stage.
4. **`../ANALYSIS_NOTES.md`** — the physics-decision log (`§N`): every non-obvious design
   choice, the reasoning, validation, and gotchas. Read before touching inversion or
   phase handling.

## Full documentation map

**Top-level (repo root)** — kept in root deliberately (discoverable, and cross-linked by
`readme.md`, each other, and the memory files):
- `../readme.md`, `../ANALYSIS_NOTES.md`, `../PIPELINE_ARCHITECTURE.md`, `../TODO.md`

**docs/** — orientation / reference:
- `cnt_reflection_measurement_brief.md` — CNT reflective-sample technical brief (this hub's headline doc)
- `sandwich_extraction_explained.md` — substrate-sandwich (air|sub|sample|sub|air) transmission extraction
- `quantity_registry.md` — **the single source of truth for displayable/exportable quantities**: how it
  works, the `Quantity` fields, how to add one (`register(...)`), overlays, and who consumes it (viewer,
  CSV export, the `display` plotting adapter)

**Tools with their own README:**
- `../acquisition_editor/README.md` — lab-user guide to the standalone acquisition editor: walking
  through scans to judge purge equilibration, patching table-knock spikes (OPTP), excluding the odd
  noisy scan, cropping, the export format and its limitations

**reports/** — the CNT science narrative (findings log, tutorials, research):
- `CNT_measurement_lab_notebook.md` — **the never-erased running findings log (`F1`–`F29`)**; current interpretation lives here
- `reflection_interface_theory_tutorial.md` — impedance / the `r=−1` pole / high-index rescue
- `lineshape_and_inversion_tutorial.md` — signal-vs-artifact lineshapes
- `misalignment_lineshape_report.md` — wavefront distortion → group delay → n spikes
- `air_gap_deembed_research.md` — de-embed literature recanvas (min-phase / MEM / amplitude-KK / Si-anchor)
- `cnt_rough_gap_literature_canvas.md`, `gap_roughness_modelling_implementation_plan.md` — rough-gap modelling
- `uncertainty_propagation.md` — frequency-domain error model

> **If two docs disagree**, precedence is: lab notebook (`reports/CNT_measurement_lab_notebook.md`)
> → `ANALYSIS_NOTES.md` → this brief. The lab notebook and ANALYSIS_NOTES are maintained
> entry-by-entry; the brief is a periodic synthesis.
